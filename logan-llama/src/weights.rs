pub use crate::DType;
use crate::{
    ContractError, ContractResult, LlamaConfig,
    model::{DenseTensor, QuantizedPayload},
};
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

struct StrictObject(BTreeMap<String, Value>);

impl<'de> Deserialize<'de> for StrictObject {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ObjectVisitor;
        impl<'de> Visitor<'de> for ObjectVisitor {
            type Value = StrictObject;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON object with unique keys")
            }

            fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut object = BTreeMap::new();
                while let Some((key, value)) = access.next_entry::<String, Value>()? {
                    if object.insert(key.clone(), value).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate JSON key `{key}`"
                        )));
                    }
                }
                Ok(StrictObject(object))
            }
        }
        deserializer.deserialize_map(ObjectVisitor)
    }
}

/// Metadata for one tensor. `offset` addresses the payload from the beginning
/// of `shard`, not from the safetensors data section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub shard: PathBuf,
    pub offset: u64,
    pub len: u64,
    pub dtype: DType,
    pub shape: Vec<u64>,
}

impl TensorInfo {
    pub fn byte_span(&self) -> std::ops::Range<u64> {
        self.offset..self.offset.saturating_add(self.len)
    }
    pub fn elements(&self) -> u64 {
        self.shape
            .iter()
            .copied()
            .fold(1, |a, b| a.saturating_mul(b))
    }
}

/// Validated metadata for a MiniCPM5 checkpoint. No weight payload is loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightInventory {
    pub root: PathBuf,
    pub shards: Vec<PathBuf>,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub dtype_counts: BTreeMap<DType, u64>,
    pub total_bytes: u64,
}

impl WeightInventory {
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }
    pub fn len(&self) -> usize {
        self.tensors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}
/// Storage dtype used by an MLX quantized tensor or its affine metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QuantizedDType {
    U8,
    U32,
    F16,
    BF16,
    F32,
}

impl QuantizedDType {
    fn from_name(value: &str) -> Option<Self> {
        match value.to_ascii_uppercase().as_str() {
            "U8" | "UINT8" => Some(Self::U8),
            "U32" | "UINT32" => Some(Self::U32),
            "F16" | "FLOAT16" | "HALF" => Some(Self::F16),
            "BF16" | "BFLOAT16" => Some(Self::BF16),
            "F32" | "FLOAT32" => Some(Self::F32),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::U8 => "U8",
            Self::U32 => "U32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F32 => "F32",
        }
    }

    fn bytes_per_element(self) -> u64 {
        match self {
            Self::U8 => 1,
            Self::U32 | Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

/// Nominal weight-code precision of one MLX tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QuantizedBits {
    B4,
    B5,
    B6,
    B8,
}

impl QuantizedBits {
    fn from_u64(value: u64) -> Option<Self> {
        match value {
            4 => Some(Self::B4),
            5 => Some(Self::B5),
            6 => Some(Self::B6),
            8 => Some(Self::B8),
            _ => None,
        }
    }

    pub fn value(self) -> u8 {
        match self {
            Self::B4 => 4,
            Self::B5 => 5,
            Self::B6 => 6,
            Self::B8 => 8,
        }
    }
}

/// MLX quantization family. This is deliberately separate from execution
/// backends: an ANE/Metal buffer is not a quantized checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MlxQuantization {
    Affine,
    Oqe,
}

impl MlxQuantization {
    fn from_name(value: &str) -> Option<Self> {
        match value
            .to_ascii_lowercase()
            .replace('_', "")
            .replace('-', "")
            .as_str()
        {
            "affine" | "mlxaffine" => Some(Self::Affine),
            "oqe" | "mlxoqe" | "oq8e" | "mlxoq8e" => Some(Self::Oqe),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Affine => "mlx-affine",
            Self::Oqe => "mlx-oqe",
        }
    }
}

/// A byte-addressed tensor in an MLX checkpoint. Unlike `TensorInfo`, this
/// type can represent packed integer storage without pretending it is F16 or
/// BF16.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedTensorInfo {
    pub shard: PathBuf,
    pub offset: u64,
    pub len: u64,
    pub dtype: QuantizedDType,
    pub shape: Vec<u64>,
}

impl PackedTensorInfo {
    pub fn byte_span(&self) -> std::ops::Range<u64> {
        self.offset..self.offset.saturating_add(self.len)
    }
    pub fn elements(&self) -> u64 {
        self.shape
            .iter()
            .copied()
            .fold(1, |a, b| a.saturating_mul(b))
    }
}

/// Auxiliary scale/bias tensor metadata. Payload bytes are intentionally not
/// loaded by the inspector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedAuxTensorInfo {
    pub shard: PathBuf,
    pub offset: u64,
    pub len: u64,
    pub dtype: QuantizedDType,
    pub shape: Vec<u64>,
}

impl QuantizedAuxTensorInfo {
    pub fn byte_span(&self) -> std::ops::Range<u64> {
        self.offset..self.offset.saturating_add(self.len)
    }
    pub fn elements(&self) -> u64 {
        self.shape
            .iter()
            .copied()
            .fold(1, |a, b| a.saturating_mul(b))
    }
}

/// Validated metadata for one MLX affine/oQe tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedTensorInfo {
    /// The packed source tensor. This is the source representation identity;
    /// no dequantized replacement is ever synthesized.
    pub packed: PackedTensorInfo,
    pub bits: QuantizedBits,
    pub group_size: u64,
    pub rows: u64,
    pub columns: u64,
    pub scales: QuantizedAuxTensorInfo,
    pub biases: QuantizedAuxTensorInfo,
    pub source_quantization: String,
}

/// Validated inventory for an MLX quantized checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedWeightInventory {
    pub root: PathBuf,
    pub shards: Vec<PathBuf>,
    pub representation: MlxQuantization,
    pub source_quantization: String,
    pub tensors: BTreeMap<String, QuantizedTensorInfo>,
    pub auxiliaries: BTreeMap<String, QuantizedAuxTensorInfo>,
    pub total_packed_bytes: u64,
    pub total_auxiliary_bytes: u64,
}

impl QuantizedWeightInventory {
    pub fn tensor(&self, name: &str) -> Option<&QuantizedTensorInfo> {
        self.tensors.get(name)
    }
    pub fn len(&self) -> usize {
        self.tensors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
    pub fn total_bytes(&self) -> u64 {
        self.total_packed_bytes
            .saturating_add(self.total_auxiliary_bytes)
    }
}

/// Compatibility aliases make the representation explicit at call sites
/// while keeping the concise inventory API discoverable.
pub type MlxQuantizedTensorInfo = QuantizedTensorInfo;
pub type MlxQuantizedWeightInventory = QuantizedWeightInventory;

/// Inspect safetensors headers and validate the complete MiniCPM5 tensor
/// contract. Only bounded JSON headers are read; tensor payloads are untouched.
pub fn inspect_weights<P: AsRef<Path>>(
    root: P,
    config: &LlamaConfig,
) -> Result<WeightInventory, String> {
    inspect_weights_inner(root.as_ref(), config).map_err(|error| error.to_string())
}

/// Inspect an MLX affine/oQe checkpoint without reading or rewriting packed
/// weight bytes. The quantization manifest is normally
/// `mlx_quantization.json`; `config.json.quantization_config` is accepted as
/// the standard MLX sidecar spelling.
pub fn inspect_quantized_weights<P: AsRef<Path>>(
    root: P,
    config: &LlamaConfig,
) -> Result<QuantizedWeightInventory, String> {
    inspect_quantized_weights_inner(root.as_ref(), config).map_err(|error| error.to_string())
}

pub fn inspect_mlx_quantized_weights<P: AsRef<Path>>(
    root: P,
    config: &LlamaConfig,
) -> Result<QuantizedWeightInventory, String> {
    inspect_quantized_weights(root, config)
}

pub fn inspect_mlx_weights<P: AsRef<Path>>(
    root: P,
    config: &LlamaConfig,
) -> Result<QuantizedWeightInventory, String> {
    inspect_quantized_weights(root, config)
}
/// Materialize a validated MLX affine/oQe source into the dense tensor
/// contract. This is intentionally a compatibility path: the quantized bytes
/// are decoded once at load time, after which DenseModel uses its existing
/// resident BF16/Metal execution path.
pub(crate) fn load_quantized_tensors<P: AsRef<Path>>(
    root: P,
    config: &LlamaConfig,
) -> Result<BTreeMap<String, DenseTensor>, String> {
    let inventory = inspect_quantized_weights(root, config)?;
    let mut descriptors = BTreeMap::new();
    for shard in &inventory.shards {
        for (name, descriptor) in parse_quantized_shard(shard).map_err(|error| error.to_string())? {
            if descriptors.insert(name.clone(), descriptor).is_some() {
                return Err(format!("duplicate quantized tensor `{name}`"));
            }
        }
    }

    let mut tensors = BTreeMap::new();
    let mut consumed = BTreeSet::new();
    for (name, info) in &inventory.tensors {
        let packed = read_range(&info.packed.shard, info.packed.offset, info.packed.len)?;
        let scales = read_range(&info.scales.shard, info.scales.offset, info.scales.len)?;
        let biases = read_range(&info.biases.shard, info.biases.offset, info.biases.len)?;
        let bytes = dequantize_mlx_weight(info, &packed, &scales, &biases)?;
        let mut aux = scales;
        aux.extend_from_slice(&biases);
        tensors.insert(
            name.clone(),
            DenseTensor {
                dtype: DType::BF16,
                shape: vec![info.rows as usize, info.columns as usize],
                bytes,
                quantized: Some(QuantizedPayload {
                    weights: packed,
                    aux,
                    bits: info.bits.value(),
                    group_size: usize::try_from(info.group_size)
                        .map_err(|_| format!("{name}: group size is too large"))?,
                }),
            },
        );
        consumed.insert(span_key(
            &info.packed.shard,
            info.packed.offset,
            info.packed.len,
        ));
        consumed.insert(span_key(
            &info.scales.shard,
            info.scales.offset,
            info.scales.len,
        ));
        consumed.insert(span_key(
            &info.biases.shard,
            info.biases.offset,
            info.biases.len,
        ));
    }

    for (name, descriptor) in descriptors {
        if consumed.contains(&span_key(
            &descriptor.shard,
            descriptor.offset,
            descriptor.len,
        )) {
            continue;
        }
        let dtype = match descriptor.dtype {
            QuantizedDType::F16 => DType::F16,
            QuantizedDType::BF16 => DType::BF16,
            QuantizedDType::F32 => DType::BF16,
            QuantizedDType::U8 | QuantizedDType::U32 => {
                return Err(format!(
                    "unclaimed packed MLX tensor `{name}` is not present in the quantization manifest"
                ));
            }
        };
        let raw = read_range(&descriptor.shard, descriptor.offset, descriptor.len)?;
        let bytes = if descriptor.dtype == QuantizedDType::F32 {
            encode_bf16(&decode_float_values(descriptor.dtype, &raw)?)
        } else {
            raw
        };
        let shape = descriptor
            .shape
            .iter()
            .map(|&value| usize::try_from(value).map_err(|_| format!("{name}: shape is too large")))
            .collect::<Result<Vec<_>, _>>()?;
        tensors.insert(
            name,
            DenseTensor {
                dtype,
                shape,
                bytes,
                quantized: None,
            },
        );
    }
    Ok(tensors)
}

fn span_key(path: &Path, offset: u64, len: u64) -> (PathBuf, u64, u64) {
    (path.to_owned(), offset, len)
}

fn read_range(path: &Path, offset: u64, len: u64) -> Result<Vec<u8>, String> {
    let mut file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let len =
        usize::try_from(len).map_err(|_| format!("{}: tensor is too large", path.display()))?;
    let mut bytes = vec![0_u8; len];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(bytes)
}

fn dequantize_mlx_weight(
    info: &QuantizedTensorInfo,
    packed: &[u8],
    scales: &[u8],
    biases: &[u8],
) -> Result<Vec<u8>, String> {
    let rows = usize::try_from(info.rows).map_err(|_| "quantized rows are too large")?;
    let columns = usize::try_from(info.columns).map_err(|_| "quantized columns are too large")?;
    let group_size =
        usize::try_from(info.group_size).map_err(|_| "quantized group is too large")?;
    let scale_values = decode_float_values(info.scales.dtype, scales)?;
    let bias_values = decode_float_values(info.biases.dtype, biases)?;
    if scale_values.len() != bias_values.len() || scale_values.len() % rows != 0 {
        return Err(format!(
            "quantized sidecars have incompatible lengths for {}x{}",
            rows, columns
        ));
    }
    let groups_per_row = scale_values.len() / rows;
    if groups_per_row == 0 || groups_per_row * group_size < columns {
        return Err(format!(
            "quantized sidecars have {groups_per_row} groups for {columns} columns at group size {group_size}"
        ));
    }
    let bits = info.bits.value() as usize;
    let values_per_unit = match info.packed.dtype {
        QuantizedDType::U8 => {
            if bits != 8 {
                return Err("U8 MLX storage is only valid for 8-bit weights".into());
            }
            1
        }
        QuantizedDType::U32 => 32 / bits,
        other => return Err(format!("unsupported packed dtype {}", other.as_str())),
    };
    let expected_units = rows
        .checked_mul((columns + values_per_unit - 1) / values_per_unit)
        .ok_or("quantized payload shape overflows")?;
    let unit_bytes = match info.packed.dtype {
        QuantizedDType::U8 => 1,
        QuantizedDType::U32 => 4,
        _ => unreachable!(),
    };
    if packed.len() != expected_units * unit_bytes {
        return Err(format!(
            "quantized payload has {} bytes, expected {}",
            packed.len(),
            expected_units * unit_bytes
        ));
    }
    let mask = (1_u32 << bits) - 1;
    let mut values = Vec::with_capacity(rows * columns);
    for row in 0..rows {
        for column in 0..columns {
            let linear = row * columns + column;
            let code = match info.packed.dtype {
                QuantizedDType::U8 => packed[linear] as u32,
                QuantizedDType::U32 => {
                    let unit = linear / values_per_unit;
                    let shift = (linear % values_per_unit) * bits;
                    (u32::from_le_bytes([
                        packed[unit * 4],
                        packed[unit * 4 + 1],
                        packed[unit * 4 + 2],
                        packed[unit * 4 + 3],
                    ]) >> shift)
                        & mask
                }
                _ => unreachable!(),
            };
            let group = row * groups_per_row + column / group_size;
            values.push(code as f32 * scale_values[group] + bias_values[group]);
        }
    }
    Ok(encode_bf16(&values))
}

fn decode_float_values(dtype: QuantizedDType, bytes: &[u8]) -> Result<Vec<f32>, String> {
    let width = dtype.bytes_per_element() as usize;
    if width == 0 || bytes.len() % width != 0 {
        return Err(format!(
            "{} payload has {} bytes, not divisible by {width}",
            dtype.as_str(),
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(width)
        .map(|chunk| match dtype {
            QuantizedDType::F16 => f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])),
            QuantizedDType::BF16 => {
                f32::from_bits((u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16)
            }
            QuantizedDType::F32 => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            QuantizedDType::U8 | QuantizedDType::U32 => 0.0,
        })
        .collect())
}

fn encode_bf16(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for &value in values {
        bytes.extend_from_slice(&f32_to_bf16(value).to_le_bytes());
    }
    bytes
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = ((bits >> 16) & 1) + 0x7fff;
    ((bits.wrapping_add(rounding)) >> 16) as u16
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = (bits & 0x03ff) as u32;
    let value = if exponent == 0 {
        if fraction == 0 {
            sign
        } else {
            let mut frac = fraction;
            let mut exp = -14_i32;
            while frac & 0x400 == 0 {
                frac <<= 1;
                exp -= 1;
            }
            frac &= 0x3ff;
            sign | (((exp + 127) as u32) << 23) | (frac << 13)
        }
    } else if exponent == 31 {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        sign | (((exponent as u32 - 15 + 127) as u32) << 23) | (fraction << 13)
    };
    f32::from_bits(value)
}

fn inspect_quantized_weights_inner(
    root: &Path,
    config: &LlamaConfig,
) -> ContractResult<QuantizedWeightInventory> {
    if !root.is_dir() {
        return Err(ContractError::invalid(format!(
            "model root is not a directory: {}",
            root.display()
        )));
    }
    let manifest = read_quantization_manifest(root)?;
    let representation = manifest
        .mode
        .as_deref()
        .and_then(MlxQuantization::from_name)
        .ok_or_else(|| {
            ContractError::invalid(
                "unsupported quantized representation: expected MLX affine or oQe",
            )
        })?;
    let source_quantization = manifest.source_quantization.clone().ok_or_else(|| {
        ContractError::invalid("quantized checkpoint is missing source quantization identity")
    })?;
    if source_quantization.trim().is_empty() {
        return Err(ContractError::invalid(
            "source quantization identity must not be empty",
        ));
    }
    let shards = discover_quantized_shards(root)?;
    let mut descriptors = BTreeMap::new();
    for shard in &shards {
        for (name, descriptor) in parse_quantized_shard(shard)? {
            if descriptors.insert(name.clone(), descriptor).is_some() {
                return Err(ContractError::invalid(format!(
                    "duplicate quantized tensor `{name}` across shards"
                )));
            }
        }
    }

    let tensor_specs = if manifest.tensors.is_object() {
        manifest.tensors.clone()
    } else {
        infer_omlx_tensor_specs(&descriptors)
    };
    let specs = tensor_specs.as_object().ok_or_else(|| {
        ContractError::invalid("MLX quantization sidecar is missing object `tensors`")
    })?;
    if specs.is_empty() {
        return Err(ContractError::invalid(
            "MLX quantization sidecar has no tensor entries",
        ));
    }
    let mut tensors = BTreeMap::new();
    let mut auxiliaries = BTreeMap::new();
    let mut total_packed_bytes = 0_u64;
    let mut total_auxiliary_bytes = 0_u64;
    for (logical_name, value) in specs {
        let spec = value.as_object().ok_or_else(|| {
            ContractError::invalid(format!(
                "{logical_name}: quantization metadata must be an object"
            ))
        })?;
        let packed_name = metadata_ref(spec, &["packed", "weight", "qweight", "source"])
            .unwrap_or_else(|| logical_name.clone());
        let scales_name = metadata_ref(spec, &["scales", "scale"])
            .unwrap_or_else(|| format!("{logical_name}.scales"));
        let biases_name = metadata_ref(spec, &["biases", "bias"])
            .unwrap_or_else(|| format!("{logical_name}.biases"));
        let packed = descriptors.get(&packed_name).ok_or_else(|| {
            ContractError::invalid(format!(
                "{logical_name}: sidecar packed tensor reference `{packed_name}` is missing"
            ))
        })?;
        let scales = descriptors.get(&scales_name).ok_or_else(|| {
            ContractError::invalid(format!(
                "{logical_name}: sidecar scales reference `{scales_name}` is missing"
            ))
        })?;
        let biases = descriptors.get(&biases_name).ok_or_else(|| {
            ContractError::invalid(format!(
                "{logical_name}: sidecar biases reference `{biases_name}` is missing"
            ))
        })?;
        if !matches!(packed.dtype, QuantizedDType::U8 | QuantizedDType::U32) {
            return Err(ContractError::invalid(format!(
                "{logical_name}: unsupported quantized packed dtype `{}` (dequantized buffers are not checkpoints)",
                packed.dtype.as_str()
            )));
        }
        if !matches!(
            scales.dtype,
            QuantizedDType::F16 | QuantizedDType::BF16 | QuantizedDType::F32
        ) || !matches!(
            biases.dtype,
            QuantizedDType::F16 | QuantizedDType::BF16 | QuantizedDType::F32
        ) {
            return Err(ContractError::invalid(format!(
                "{logical_name}: sidecar scale/bias dtype must be F16, BF16, or F32"
            )));
        }
        let bits = metadata_bits(spec, &manifest).ok_or_else(|| {
            ContractError::invalid(format!(
                "{logical_name}: missing quantization bits (expected 4, 5, 6, or 8)"
            ))
        })?;
        let group_size = metadata_u64(spec, &["group_size", "group", "block_size"])
            .or(manifest.group_size)
            .ok_or_else(|| {
                ContractError::invalid(format!("{logical_name}: missing positive group_size"))
            })?;
        if group_size == 0 {
            return Err(ContractError::invalid(format!(
                "{logical_name}: group_size must be positive"
            )));
        }
        let (rows, columns) = logical_geometry(logical_name, spec, config)?;
        validate_quantized_geometry(
            logical_name,
            packed,
            scales,
            biases,
            bits,
            group_size,
            rows,
            columns,
        )?;
        let tensor_source =
            metadata_string(spec, &["source_quantization", "source_quantization_id"])
                .unwrap_or_else(|| source_quantization.clone());
        if tensor_source != source_quantization {
            return Err(ContractError::invalid(format!(
                "{logical_name}: source quantization identity `{tensor_source}` disagrees with `{source_quantization}`"
            )));
        }
        validate_declared_auxiliary_metadata(logical_name, spec, "scale", scales)?;
        validate_declared_auxiliary_metadata(logical_name, spec, "bias", biases)?;
        let packed = PackedTensorInfo {
            shard: packed.shard.clone(),
            offset: packed.offset,
            len: packed.len,
            dtype: packed.dtype,
            shape: packed.shape.clone(),
        };
        let scales = scales.clone();
        let biases = biases.clone();
        total_packed_bytes = total_packed_bytes
            .checked_add(packed.len)
            .ok_or_else(|| ContractError::invalid("quantized packed byte total overflows u64"))?;
        if auxiliaries
            .insert(scales_name.clone(), scales.clone())
            .is_none()
        {
            total_auxiliary_bytes =
                total_auxiliary_bytes
                    .checked_add(scales.len)
                    .ok_or_else(|| {
                        ContractError::invalid("quantized sidecar byte total overflows u64")
                    })?;
        }
        if auxiliaries
            .insert(biases_name.clone(), biases.clone())
            .is_none()
        {
            total_auxiliary_bytes =
                total_auxiliary_bytes
                    .checked_add(biases.len)
                    .ok_or_else(|| {
                        ContractError::invalid("quantized sidecar byte total overflows u64")
                    })?;
        }
        tensors.insert(
            logical_name.clone(),
            QuantizedTensorInfo {
                packed,
                bits,
                group_size,
                rows,
                columns,
                scales,
                biases,
                source_quantization: tensor_source,
            },
        );
    }
    Ok(QuantizedWeightInventory {
        root: root.to_owned(),
        shards,
        representation,
        source_quantization,
        tensors,
        auxiliaries,
        total_packed_bytes,
        total_auxiliary_bytes,
    })
}

#[derive(Debug)]
struct QuantizationManifest {
    mode: Option<String>,
    source_quantization: Option<String>,
    bits: Option<QuantizedBits>,
    group_size: Option<u64>,
    tensors: Value,
}

fn read_quantization_manifest(root: &Path) -> ContractResult<QuantizationManifest> {
    let candidates = [
        "mlx_quantization.json",
        "quantization.json",
        "quantization_config.json",
    ];
    for candidate in candidates {
        let path = root.join(candidate);
        if path.is_file() {
            let value = read_json_object(&path)?;
            return manifest_from_object(value);
        }
    }
    let config_path = root.join("config.json");
    if config_path.is_file() {
        let config = read_json_object(&config_path)?;
        if let Some(value) = config.get("quantization_config") {
            let mut manifest = manifest_from_object(value.clone())?;
            if manifest.source_quantization.is_none() {
                manifest.source_quantization = Some("omlx".to_owned());
            }
            return Ok(manifest);
        }
    }
    Err(ContractError::invalid(
        "missing MLX quantization sidecar (mlx_quantization.json or config.json.quantization_config)",
    ))
}

fn infer_omlx_tensor_specs(descriptors: &BTreeMap<String, QuantizedAuxTensorInfo>) -> Value {
    let mut tensors = serde_json::Map::new();
    for (name, packed) in descriptors {
        if !name.ends_with(".weight")
            || !matches!(packed.dtype, QuantizedDType::U8 | QuantizedDType::U32)
        {
            continue;
        }
        let base = &name[..name.len() - ".weight".len()];
        let scales_name = format!("{base}.scales");
        let biases_name = format!("{base}.biases");
        if !descriptors.contains_key(&scales_name) || !descriptors.contains_key(&biases_name) {
            continue;
        }
        tensors.insert(
            name.clone(),
            serde_json::json!({
                "packed": name,
                "scales": scales_name,
                "biases": biases_name,
            }),
        );
    }
    Value::Object(tensors)
}

fn manifest_from_object(value: Value) -> ContractResult<QuantizationManifest> {
    let object = value
        .as_object()
        .ok_or_else(|| ContractError::invalid("MLX quantization sidecar must be a JSON object"))?;
    let mode = metadata_string(
        object,
        &["mode", "quantization_mode", "quantization_type", "format"],
    );
    if mode.as_deref().is_some_and(|mode| {
        let lower = mode.to_ascii_lowercase();
        lower.contains("ane") || lower.contains("metal") || lower.contains("dequant")
    }) {
        return Err(ContractError::invalid(
            "unsupported quantized representation: ANE/Metal dequantized buffers are not MLX checkpoints",
        ));
    }
    let bits = object
        .get("bits")
        .and_then(Value::as_u64)
        .and_then(QuantizedBits::from_u64);
    let group_size = metadata_u64(object, &["group_size", "group", "block_size"]);
    let source_quantization = metadata_string(
        object,
        &[
            "source_quantization",
            "source_quantization_id",
            "source_format",
            "quantization",
        ],
    );
    let tensors = object
        .get("tensors")
        .or_else(|| object.get("weights"))
        .or_else(|| object.get("quantized_tensors"))
        .cloned()
        .unwrap_or(Value::Null);
    Ok(QuantizationManifest {
        mode,
        source_quantization,
        bits,
        group_size,
        tensors,
    })
}

fn read_json_object(path: &Path) -> ContractResult<Value> {
    let bytes = fs::read(path)
        .map_err(|error| ContractError::Io(format!("{}: {error}", path.display())))?;
    let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
        ContractError::Json(format!("{}: invalid JSON: {error}", path.display()))
    })?;
    if !value.is_object() {
        return Err(ContractError::invalid(format!(
            "{}: expected a JSON object",
            path.display()
        )));
    }
    Ok(value)
}

fn discover_quantized_shards(root: &Path) -> ContractResult<Vec<PathBuf>> {
    let mut shards = Vec::new();
    let mut entries = fs::read_dir(root)
        .map_err(|error| ContractError::Io(format!("{}: {error}", root.display())))?;
    while let Some(entry) = entries.next() {
        let path = entry
            .map_err(|error| ContractError::Io(error.to_string()))?
            .path();
        if path.extension().and_then(|x| x.to_str()) == Some("safetensors") {
            shards.push(path);
        }
    }
    shards.sort();
    if shards.is_empty() {
        return Err(ContractError::invalid(
            "quantized checkpoint has no safetensors shards",
        ));
    }
    Ok(shards)
}

fn parse_quantized_shard(path: &Path) -> ContractResult<Vec<(String, QuantizedAuxTensorInfo)>> {
    let mut file = File::open(path)
        .map_err(|error| ContractError::Io(format!("{}: {error}", path.display())))?;
    let file_len = file
        .metadata()
        .map_err(|error| ContractError::Io(format!("{}: {error}", path.display())))?
        .len();
    let mut bytes = [0_u8; 8];
    file.read_exact(&mut bytes)
        .map_err(|error| ContractError::Io(format!("{}: {error}", path.display())))?;
    let header_len = u64::from_le_bytes(bytes);
    let data_start = 8_u64
        .checked_add(header_len)
        .ok_or_else(|| ContractError::invalid("quantized header length overflows"))?;
    if data_start > file_len {
        return Err(ContractError::invalid(format!(
            "{}: header extends past EOF",
            path.display()
        )));
    }
    let mut header = vec![
        0_u8;
        usize::try_from(header_len).map_err(|_| ContractError::invalid(
            "quantized header is too large"
        ))?
    ];
    file.read_exact(&mut header)
        .map_err(|error| ContractError::Io(format!("{}: {error}", path.display())))?;
    let StrictObject(object) =
        serde_json::from_slice::<StrictObject>(&header).map_err(|error| {
            ContractError::Json(format!("{}: invalid header JSON: {error}", path.display()))
        })?;
    let payload_len = file_len - data_start;
    let mut ranges = Vec::new();
    let mut out = Vec::new();
    for (name, descriptor) in object {
        if name == "__metadata__" {
            continue;
        }
        let descriptor = descriptor.as_object().ok_or_else(|| {
            ContractError::invalid(format!("{name}: tensor descriptor is not an object"))
        })?;
        let dtype_name = descriptor
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| ContractError::invalid(format!("{name}: missing dtype")))?;
        let dtype = QuantizedDType::from_name(dtype_name).ok_or_else(|| {
            ContractError::invalid(format!("{name}: unsupported dtype `{dtype_name}`"))
        })?;
        let shape = descriptor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| ContractError::invalid(format!("{name}: missing shape")))?
            .iter()
            .map(|value| {
                value.as_u64().ok_or_else(|| {
                    ContractError::invalid(format!("{name}: shape must contain unsigned integers"))
                })
            })
            .collect::<ContractResult<Vec<_>>>()?;
        let offsets = descriptor
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| ContractError::invalid(format!("{name}: missing data_offsets")))?;
        if offsets.len() != 2 {
            return Err(ContractError::invalid(format!(
                "{name}: data_offsets must have two entries"
            )));
        }
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| ContractError::invalid(format!("{name}: invalid data_offsets start")))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| ContractError::invalid(format!("{name}: invalid data_offsets end")))?;
        if end < start || end > payload_len {
            return Err(ContractError::invalid(format!(
                "{name}: tensor span is outside payload"
            )));
        }
        let elements = shape
            .iter()
            .copied()
            .try_fold(1_u64, |a, b| a.checked_mul(b))
            .ok_or_else(|| ContractError::invalid(format!("{name}: shape overflows")))?;
        let expected = elements
            .checked_mul(dtype.bytes_per_element())
            .ok_or_else(|| ContractError::invalid(format!("{name}: byte span overflows")))?;
        if end - start != expected {
            return Err(ContractError::invalid(format!(
                "{name}: byte span {} does not equal shape elements × dtype width {expected}",
                end - start
            )));
        }
        for (old_start, old_end, old_name) in &ranges {
            if start < *old_end && *old_start < end {
                return Err(ContractError::invalid(format!(
                    "{name}: tensor span overlaps `{old_name}`"
                )));
            }
        }
        ranges.push((start, end, name.clone()));
        out.push((
            name,
            QuantizedAuxTensorInfo {
                shard: path.to_owned(),
                offset: data_start + start,
                len: end - start,
                dtype,
                shape,
            },
        ));
    }
    Ok(out)
}

fn inspect_weights_inner(root: &Path, config: &LlamaConfig) -> ContractResult<WeightInventory> {
    if !root.is_dir() {
        return Err(ContractError::invalid(format!(
            "model root is not a directory: {}",
            root.display()
        )));
    }
    let index_path = root.join("model.safetensors.index.json");
    let (shard_names, indexed_names) = if index_path.is_file() {
        read_index(&index_path)?
    } else {
        let mut names = Vec::new();
        let mut entries = fs::read_dir(root)
            .map_err(|e| ContractError::Io(format!("{}: {e}", root.display())))?;
        while let Some(entry) = entries.next() {
            let entry = entry.map_err(|e| ContractError::Io(format!("{}: {e}", root.display())))?;
            let path = entry.path();
            if path.extension().and_then(|x| x.to_str()) == Some("safetensors") {
                names.push(path.file_name().unwrap().to_string_lossy().into_owned());
            }
        }
        names.sort();
        if names.is_empty() {
            return Err(ContractError::invalid(
                "checkpoint has no safetensors shards",
            ));
        }
        (names, None)
    };

    let mut tensors = BTreeMap::new();
    let mut shards = Vec::new();
    for shard_name in shard_names {
        let relative = Path::new(&shard_name);
        if relative.is_absolute()
            || relative
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir))
        {
            return Err(ContractError::invalid(format!(
                "unsafe safetensors shard path `{shard_name}`"
            )));
        }
        let shard = root.join(relative);
        let records = parse_shard(&shard)?;
        for (name, tensor) in records {
            if tensors.insert(name.clone(), tensor).is_some() {
                return Err(ContractError::invalid(format!(
                    "duplicate tensor name `{name}` across shards"
                )));
            }
        }
        shards.push(shard);
    }
    if let Some(indexed_names) = indexed_names {
        let actual: BTreeSet<_> = tensors.keys().cloned().collect();
        if actual != indexed_names {
            let missing = indexed_names.difference(&actual).next().cloned();
            let extra = actual.difference(&indexed_names).next().cloned();
            return Err(ContractError::invalid(format!(
                "safetensors index coverage mismatch (missing={missing:?}, extra={extra:?})"
            )));
        }
    }
    validate_geometry(config, &tensors)?;
    let mut dtype_counts = BTreeMap::new();
    let mut total_bytes = 0_u64;
    for tensor in tensors.values() {
        total_bytes = total_bytes
            .checked_add(tensor.len)
            .ok_or_else(|| ContractError::invalid("tensor byte total overflows u64"))?;
        *dtype_counts.entry(tensor.dtype).or_insert(0) += 1;
    }
    Ok(WeightInventory {
        root: root.to_owned(),
        shards,
        tensors,
        dtype_counts,
        total_bytes,
    })
}

fn metadata_string(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        object
            .get(*key)
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    })
}

fn validate_declared_auxiliary_metadata(
    name: &str,
    object: &serde_json::Map<String, Value>,
    kind: &str,
    actual: &QuantizedAuxTensorInfo,
) -> ContractResult<()> {
    let dtype_key = if kind == "scale" {
        "scale_dtype"
    } else {
        "bias_dtype"
    };
    if let Some(declared) = object.get(dtype_key).and_then(Value::as_str) {
        let declared = QuantizedDType::from_name(declared).ok_or_else(|| {
            ContractError::invalid(format!(
                "{name}: unsupported sidecar {kind} dtype `{declared}`"
            ))
        })?;
        if declared != actual.dtype {
            return Err(ContractError::invalid(format!(
                "{name}: sidecar {kind} dtype `{declared:?}` disagrees with tensor dtype `{actual:?}`"
            )));
        }
    }
    let shape_key = if kind == "scale" {
        "scale_shape"
    } else {
        "bias_shape"
    };
    if let Some(declared) = object.get(shape_key).and_then(Value::as_array) {
        let declared = declared
            .iter()
            .map(|value| {
                value.as_u64().ok_or_else(|| {
                    ContractError::invalid(format!(
                        "{name}: sidecar {kind} shape must contain unsigned integers"
                    ))
                })
            })
            .collect::<ContractResult<Vec<_>>>()?;
        if declared != actual.shape {
            return Err(ContractError::invalid(format!(
                "{name}: sidecar {kind} shape {:?} disagrees with tensor shape {:?}",
                declared, actual.shape
            )));
        }
    }
    Ok(())
}

fn metadata_ref(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        let value = object.get(*key)?;
        value.as_str().map(ToOwned::to_owned).or_else(|| {
            let object = value.as_object()?;
            ["tensor", "name", "ref", "id"].iter().find_map(|field| {
                object
                    .get(*field)
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
        })
    })
}

fn metadata_u64(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| object.get(*key).and_then(Value::as_u64))
}

fn metadata_bits(
    object: &serde_json::Map<String, Value>,
    manifest: &QuantizationManifest,
) -> Option<QuantizedBits> {
    metadata_u64(object, &["bits", "weight_bits"])
        .and_then(QuantizedBits::from_u64)
        .or(manifest.bits)
}

fn logical_geometry(
    name: &str,
    object: &serde_json::Map<String, Value>,
    config: &LlamaConfig,
) -> ContractResult<(u64, u64)> {
    if let Some(shape) = object.get("shape").and_then(Value::as_array) {
        if shape.len() != 2 {
            return Err(ContractError::invalid(format!(
                "{name}: logical shape must have two entries"
            )));
        }
        let rows = shape[0].as_u64().ok_or_else(|| {
            ContractError::invalid(format!("{name}: logical rows must be an unsigned integer"))
        })?;
        let columns = shape[1].as_u64().ok_or_else(|| {
            ContractError::invalid(format!(
                "{name}: logical columns must be an unsigned integer"
            ))
        })?;
        if rows == 0 || columns == 0 {
            return Err(ContractError::invalid(format!(
                "{name}: logical shape must be positive"
            )));
        }
        return Ok((rows, columns));
    }
    let rows = metadata_u64(object, &["rows", "out_features"]);
    let columns = metadata_u64(object, &["columns", "cols", "in_features"]);
    if let (Some(rows), Some(columns)) = (rows, columns) {
        if rows == 0 || columns == 0 {
            return Err(ContractError::invalid(format!(
                "{name}: logical rows and columns must be positive"
            )));
        }
        return Ok((rows, columns));
    }
    expected_weight_shape(name, config).ok_or_else(|| {
        ContractError::invalid(format!(
            "{name}: sidecar must specify logical shape [rows, columns]"
        ))
    })
}

fn expected_weight_shape(name: &str, config: &LlamaConfig) -> Option<(u64, u64)> {
    let (rows, columns) = if name.ends_with("input_layernorm.weight")
        || name.ends_with("post_attention_layernorm.weight")
        || name.ends_with("norm.weight")
    {
        return None;
    } else if name.ends_with("embed_tokens.weight")
        || name.ends_with("tok_embeddings.weight")
        || name.ends_with("lm_head.weight")
    {
        (config.vocab_size, config.hidden_size)
    } else if name.ends_with("self_attn.q_proj.weight") {
        (
            (config.num_attention_heads as u64) * (config.head_dim as u64),
            config.hidden_size,
        )
    } else if name.ends_with("self_attn.k_proj.weight") || name.ends_with("self_attn.v_proj.weight")
    {
        (
            (config.num_key_value_heads as u64) * (config.head_dim as u64),
            config.hidden_size,
        )
    } else if name.ends_with("self_attn.o_proj.weight") {
        (
            config.hidden_size,
            (config.num_attention_heads as u64) * (config.head_dim as u64),
        )
    } else if name.ends_with("mlp.gate_proj.weight") || name.ends_with("mlp.up_proj.weight") {
        (config.intermediate_size, config.hidden_size)
    } else if name.ends_with("mlp.down_proj.weight") {
        (config.hidden_size, config.intermediate_size)
    } else {
        return None;
    };
    Some((rows, columns))
}

fn validate_quantized_geometry(
    name: &str,
    packed: &QuantizedAuxTensorInfo,
    scales: &QuantizedAuxTensorInfo,
    biases: &QuantizedAuxTensorInfo,
    bits: QuantizedBits,
    group_size: u64,
    rows: u64,
    columns: u64,
) -> ContractResult<()> {
    if packed.shape.len() != 2 || packed.shape[0] != rows {
        return Err(ContractError::invalid(format!(
            "{name}: packed tensor row geometry {:?} does not match logical [{rows}, {columns}]",
            packed.shape
        )));
    }
    let words_per_row = match packed.dtype {
        QuantizedDType::U32 => {
            let values_per_word = 32 / u64::from(bits.value());
            (columns + values_per_word - 1) / values_per_word
        }
        QuantizedDType::U8 if matches!(bits, QuantizedBits::B4 | QuantizedBits::B8) => {
            let values_per_byte = 8 / u64::from(bits.value());
            (columns + values_per_byte - 1) / values_per_byte
        }
        QuantizedDType::U8 => {
            return Err(ContractError::invalid(format!(
                "{name}: U8 packed storage cannot represent {}-bit MLX codes",
                bits.value()
            )));
        }
        _ => unreachable!(),
    };
    if packed.shape[1] != words_per_row {
        return Err(ContractError::invalid(format!(
            "{name}: packed columns {} do not match {}-bit row geometry (expected {words_per_row})",
            packed.shape[1],
            bits.value()
        )));
    }
    let groups = (columns + group_size - 1) / group_size;
    let expected_aux = [rows, groups];
    if scales.shape != expected_aux {
        return Err(ContractError::invalid(format!(
            "{name}: scales shape {:?}, expected {:?}",
            scales.shape, expected_aux
        )));
    }
    if biases.shape != expected_aux {
        return Err(ContractError::invalid(format!(
            "{name}: biases shape {:?}, expected {:?}",
            biases.shape, expected_aux
        )));
    }
    Ok(())
}

pub(crate) fn read_index(path: &Path) -> ContractResult<(Vec<String>, Option<BTreeSet<String>>)> {
    let bytes =
        fs::read(path).map_err(|e| ContractError::Io(format!("{}: {e}", path.display())))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|e| ContractError::Json(format!("{}: invalid JSON: {e}", path.display())))?;
    let map = value
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ContractError::invalid("safetensors index is missing object `weight_map`")
        })?;
    let mut shards = BTreeSet::new();
    let mut names = BTreeSet::new();
    for (name, shard) in map {
        let shard = shard.as_str().ok_or_else(|| {
            ContractError::invalid(format!("weight_map entry `{name}` is not a string"))
        })?;
        shards.insert(shard.to_owned());
        names.insert(name.to_owned());
    }
    if shards.is_empty() {
        return Err(ContractError::invalid(
            "safetensors index has an empty weight_map",
        ));
    }
    Ok((shards.into_iter().collect(), Some(names)))
}

pub(crate) fn parse_shard(path: &Path) -> ContractResult<Vec<(String, TensorInfo)>> {
    let mut file =
        File::open(path).map_err(|e| ContractError::Io(format!("{}: {e}", path.display())))?;
    let file_len = file
        .metadata()
        .map_err(|e| ContractError::Io(format!("{}: {e}", path.display())))?
        .len();
    let mut bytes = [0_u8; 8];
    file.read_exact(&mut bytes)
        .map_err(|e| ContractError::Io(format!("{}: {e}", path.display())))?;
    let header_len = u64::from_le_bytes(bytes);
    let data_start = 8_u64.checked_add(header_len).ok_or_else(|| {
        ContractError::invalid(format!("{}: header length overflows", path.display()))
    })?;
    if data_start > file_len {
        return Err(ContractError::invalid(format!(
            "{}: header extends past EOF",
            path.display()
        )));
    }
    let header_len_usize = usize::try_from(header_len)
        .map_err(|_| ContractError::invalid(format!("{}: header is too large", path.display())))?;
    let mut header = vec![0_u8; header_len_usize];
    file.read_exact(&mut header)
        .map_err(|e| ContractError::Io(format!("{}: {e}", path.display())))?;
    let StrictObject(object) = serde_json::from_slice::<StrictObject>(&header).map_err(|e| {
        ContractError::Json(format!("{}: invalid header JSON: {e}", path.display()))
    })?;
    let payload_len = file_len - data_start;
    let mut out = Vec::with_capacity(object.len());
    let mut ranges = Vec::<(u64, u64, String)>::new();
    for (name, descriptor) in object {
        if name == "__metadata__" {
            continue;
        }
        let descriptor = descriptor.as_object().ok_or_else(|| {
            ContractError::invalid(format!("{name}: tensor descriptor is not an object"))
        })?;
        let dtype_name = descriptor
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| ContractError::invalid(format!("{name}: missing dtype")))?;
        let dtype = DType::from_safetensors(dtype_name).ok_or_else(|| {
            ContractError::invalid(format!(
                "{name}: unsupported dtype `{dtype_name}` (expected F16 or BF16)"
            ))
        })?;
        let shape = descriptor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| ContractError::invalid(format!("{name}: missing shape")))?
            .iter()
            .map(|value| {
                value.as_u64().ok_or_else(|| {
                    ContractError::invalid(format!("{name}: shape must contain unsigned integers"))
                })
            })
            .collect::<ContractResult<Vec<_>>>()?;
        let offsets = descriptor
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| ContractError::invalid(format!("{name}: missing data_offsets")))?;
        if offsets.len() != 2 {
            return Err(ContractError::invalid(format!(
                "{name}: data_offsets must have two entries"
            )));
        }
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| ContractError::invalid(format!("{name}: invalid data_offsets start")))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| ContractError::invalid(format!("{name}: invalid data_offsets end")))?;
        if end < start || end > payload_len {
            return Err(ContractError::invalid(format!(
                "{name}: tensor span [{start}, {end}) is outside payload of {payload_len} bytes"
            )));
        }
        let elements = shape
            .iter()
            .copied()
            .try_fold(1_u64, |a, b| a.checked_mul(b))
            .ok_or_else(|| {
                ContractError::invalid(format!("{name}: shape element count overflows"))
            })?;
        let expected = elements
            .checked_mul(2)
            .ok_or_else(|| ContractError::invalid(format!("{name}: tensor byte span overflows")))?;
        if end - start != expected {
            return Err(ContractError::invalid(format!(
                "{name}: byte span {} does not equal shape element count {elements} × 2",
                end - start
            )));
        }
        for (old_start, old_end, old_name) in &ranges {
            if start < *old_end && *old_start < end {
                return Err(ContractError::invalid(format!(
                    "{name}: tensor span overlaps `{old_name}`"
                )));
            }
        }
        ranges.push((start, end, name.clone()));
        let offset = data_start
            .checked_add(start)
            .ok_or_else(|| ContractError::invalid(format!("{name}: file offset overflows")))?;
        out.push((
            name.clone(),
            TensorInfo {
                shard: path.to_owned(),
                offset,
                len: end - start,
                dtype,
                shape,
            },
        ));
    }
    Ok(out)
}

fn validate_geometry(
    config: &LlamaConfig,
    tensors: &BTreeMap<String, TensorInfo>,
) -> ContractResult<()> {
    let embedding = required(
        tensors,
        &[
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "model.tok_embeddings.weight",
            "tok_embeddings.weight",
        ],
        "token embedding",
    )?;
    shape_is(
        embedding,
        &[config.vocab_size, config.hidden_size],
        "token embedding",
    )?;
    let head = required(
        tensors,
        &["lm_head.weight", "model.lm_head.weight"],
        "untied lm_head",
    )?;
    shape_is(
        head,
        &[config.vocab_size, config.hidden_size],
        "untied lm_head",
    )?;
    let norm = required(tensors, &["model.norm.weight", "norm.weight"], "final norm")?;
    shape_is(norm, &[config.hidden_size], "final norm")?;

    for layer in 0..config.num_hidden_layers {
        let prefix = layer_prefix(tensors, layer)
            .ok_or_else(|| ContractError::invalid(format!("missing layer {layer}")))?;
        let layer_names = [
            ("input_layernorm.weight", vec![config.hidden_size]),
            (
                "self_attn.q_proj.weight",
                vec![
                    (config.num_attention_heads as u64) * (config.head_dim as u64),
                    config.hidden_size,
                ],
            ),
            (
                "self_attn.k_proj.weight",
                vec![
                    (config.num_key_value_heads as u64) * (config.head_dim as u64),
                    config.hidden_size,
                ],
            ),
            (
                "self_attn.v_proj.weight",
                vec![
                    (config.num_key_value_heads as u64) * (config.head_dim as u64),
                    config.hidden_size,
                ],
            ),
            (
                "self_attn.o_proj.weight",
                vec![
                    config.hidden_size,
                    (config.num_attention_heads as u64) * (config.head_dim as u64),
                ],
            ),
            ("post_attention_layernorm.weight", vec![config.hidden_size]),
            (
                "mlp.gate_proj.weight",
                vec![config.intermediate_size, config.hidden_size],
            ),
            (
                "mlp.up_proj.weight",
                vec![config.intermediate_size, config.hidden_size],
            ),
            (
                "mlp.down_proj.weight",
                vec![config.hidden_size, config.intermediate_size],
            ),
        ];
        for (suffix, expected) in layer_names {
            let full = format!("{prefix}{suffix}");
            let tensor = tensors
                .get(&full)
                .ok_or_else(|| ContractError::invalid(format!("missing tensor `{full}`")))?;
            shape_is(tensor, &expected, &full)?;
        }
    }
    for name in tensors.keys() {
        if let Some(index) = layer_number(name) {
            if index >= config.num_hidden_layers {
                return Err(ContractError::invalid(format!(
                    "tensor `{name}` refers to out-of-range layer {index}"
                )));
            }
        }
    }
    Ok(())
}

fn required<'a>(
    tensors: &'a BTreeMap<String, TensorInfo>,
    names: &[&str],
    role: &str,
) -> ContractResult<&'a TensorInfo> {
    names
        .iter()
        .find_map(|name| tensors.get(*name))
        .ok_or_else(|| ContractError::invalid(format!("missing {role}")))
}

fn shape_is(tensor: &TensorInfo, expected: &[u64], role: &str) -> ContractResult<()> {
    if tensor.shape != expected {
        return Err(ContractError::invalid(format!(
            "{role} has shape {:?}, expected {:?}",
            tensor.shape, expected
        )));
    }
    Ok(())
}

fn layer_prefix(tensors: &BTreeMap<String, TensorInfo>, layer: u32) -> Option<String> {
    [format!("model.layers.{layer}."), format!("layers.{layer}.")]
        .into_iter()
        .find(|prefix| tensors.keys().any(|name| name.starts_with(prefix)))
}

fn layer_number(name: &str) -> Option<u32> {
    for marker in ["model.layers.", "layers."] {
        if let Some(rest) = name.strip_prefix(marker) {
            let digits = rest.split('.').next()?;
            if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
                return digits.parse().ok();
            }
        }
    }
    None
}
