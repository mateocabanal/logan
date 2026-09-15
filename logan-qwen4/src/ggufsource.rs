//! Native GGUF v3 source for Qwen4Exp / Qwen3.8-Flash-Next.
//!
//! This module deliberately owns only the container/source boundary.  It is
//! not an inference backend and it never requantizes weights. Quantized tensor
//! payloads remain byte-for-byte GGUF blocks and are range-read directly from
//! the source file. The runtime decides how to execute those blocks.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

const GGUF_MAGIC: u32 = 0x4655_4747;
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_STRING_BYTES: u64 = 64 * 1024 * 1024;
const MAX_METADATA_ARRAY: u64 = 1 << 20;
const MAX_DIMS: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q5_0,
    Q8_0,
    Q4K,
    Q6K,
    Bf16,
}

impl GgmlType {
    pub fn from_id(id: u32) -> Result<Self, String> {
        match id {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            6 => Ok(Self::Q5_0),
            8 => Ok(Self::Q8_0),
            12 => Ok(Self::Q4K),
            14 => Ok(Self::Q6K),
            30 => Ok(Self::Bf16),
            other => Err(format!("unsupported GGML tensor type id {other}")),
        }
    }

    pub fn id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q5_0 => 6,
            Self::Q8_0 => 8,
            Self::Q4K => 12,
            Self::Q6K => 14,
            Self::Bf16 => 30,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q5_0 => "Q5_0",
            Self::Q8_0 => "Q8_0",
            Self::Q4K => "Q4_K",
            Self::Q6K => "Q6_K",
            Self::Bf16 => "BF16",
        }
    }

    /// (logical elements per stored block, stored bytes per block).
    pub fn block_geometry(self) -> (u64, u64) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::Bf16 => (1, 2),
            Self::Q5_0 => (32, 22),
            Self::Q8_0 => (32, 34),
            Self::Q4K => (256, 144),
            Self::Q6K => (256, 210),
        }
    }

    pub fn stored_bytes(self, elements: u64) -> Result<u64, String> {
        let (block, bytes) = self.block_geometry();
        if elements % block != 0 {
            return Err(format!(
                "{} tensor has {elements} elements, not divisible by block size {block}",
                self.name()
            ));
        }
        elements
            .checked_div(block)
            .and_then(|n| n.checked_mul(bytes))
            .ok_or_else(|| format!("{} tensor byte size overflow", self.name()))
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// GGUF logical dimension order, exactly as stored in the tensor descriptor.
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    /// Absolute byte offset to the first quant block / scalar in the GGUF file.
    pub offset: u64,
    pub stored_bytes: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().copied().product()
    }
}

#[derive(Clone, Debug)]
pub enum MetadataValue {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    U64Array(Vec<u64>),
    I64Array(Vec<i64>),
    F64Array(Vec<f64>),
    IgnoredArray(u64),
}

#[derive(Clone)]
pub struct GgufSource {
    path: PathBuf,
    file: Arc<Mutex<File>>,
    metadata: Arc<BTreeMap<String, MetadataValue>>,
    tensors: Arc<BTreeMap<String, TensorInfo>>,
    data_start: u64,
    alignment: u64,
}

impl GgufSource {
    pub fn open(path: &Path) -> Result<Self, String> {
        if !path.is_file() {
            return Err(format!("GGUF source is not a file: {}", path.display()));
        }
        let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let magic = read_u32(&mut file)?;
        if magic != GGUF_MAGIC {
            return Err(format!("{} is not a GGUF file", path.display()));
        }
        let version = read_u32(&mut file)?;
        if version != 3 {
            return Err(format!("GGUF version {version} is not supported; expected v3"));
        }
        let tensor_count = read_u64(&mut file)?;
        let kv_count = read_u64(&mut file)?;
        if tensor_count > 1_000_000 || kv_count > 1_000_000 {
            return Err("GGUF header counts exceed safety limits".into());
        }

        let mut metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = read_string(&mut file)?;
            let ty = read_u32(&mut file)?;
            let keep_array = key.starts_with("qwen4exp.") || key == "general.alignment";
            let value = read_metadata_value(&mut file, ty, keep_array)?;
            metadata.insert(key, value);
        }
        let alignment = metadata
            .get("general.alignment")
            .and_then(meta_u64)
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 || !alignment.is_power_of_two() || alignment > 1024 * 1024 {
            return Err(format!("invalid GGUF alignment {alignment}"));
        }

        struct PendingTensor {
            name: String,
            dims: Vec<u64>,
            dtype: GgmlType,
            relative_offset: u64,
            stored_bytes: u64,
        }
        let mut pending = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_string(&mut file)?;
            let n_dims = read_u32(&mut file)?;
            if n_dims == 0 || n_dims > MAX_DIMS {
                return Err(format!("{name}: invalid GGUF rank {n_dims}"));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut file)?);
            }
            let dtype = GgmlType::from_id(read_u32(&mut file)?)?;
            let relative_offset = read_u64(&mut file)?;
            let elements = dims.iter().try_fold(1_u64, |n, d| {
                n.checked_mul(*d)
                    .ok_or_else(|| format!("{name}: element count overflow"))
            })?;
            let stored_bytes = dtype.stored_bytes(elements)?;
            pending.push(PendingTensor {
                name,
                dims,
                dtype,
                relative_offset,
                stored_bytes,
            });
        }
        let header_end = file.stream_position().map_err(|e| e.to_string())?;
        let data_start = align_up(header_end, alignment)?;
        let file_len = file.metadata().map_err(|e| e.to_string())?.len();
        if data_start > file_len {
            return Err("GGUF data section begins beyond end of file".into());
        }

        let mut tensors = BTreeMap::new();
        for tensor in pending {
            if tensor.relative_offset % alignment != 0 {
                return Err(format!(
                    "{}: relative tensor offset {} is not {alignment}-byte aligned",
                    tensor.name, tensor.relative_offset
                ));
            }
            let offset = data_start
                .checked_add(tensor.relative_offset)
                .ok_or_else(|| format!("{}: tensor offset overflow", tensor.name))?;
            let end = offset
                .checked_add(tensor.stored_bytes)
                .ok_or_else(|| format!("{}: tensor extent overflow", tensor.name))?;
            if end > file_len {
                return Err(format!("{} extends beyond end of GGUF file", tensor.name));
            }
            let info = TensorInfo {
                name: tensor.name.clone(),
                dims: tensor.dims,
                dtype: tensor.dtype,
                offset,
                stored_bytes: tensor.stored_bytes,
            };
            if tensors.insert(tensor.name.clone(), info).is_some() {
                return Err(format!("duplicate GGUF tensor {}", tensor.name));
            }
        }

        let source = Self {
            path: path.to_owned(),
            file: Arc::new(Mutex::new(file)),
            metadata: Arc::new(metadata),
            tensors: Arc::new(tensors),
            data_start,
            alignment,
        };
        let arch = source.string("general.architecture").unwrap_or("");
        if arch != "qwen4exp" {
            return Err(format!(
                "unsupported GGUF architecture {arch:?}; this path currently requires qwen4exp"
            ));
        }
        Ok(source)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor(name).is_some()
    }

    pub fn metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    pub fn u64(&self, key: &str) -> Option<u64> {
        self.metadata(key).and_then(meta_u64)
    }

    pub fn i64(&self, key: &str) -> Option<i64> {
        match self.metadata(key)? {
            MetadataValue::I64(v) => Some(*v),
            MetadataValue::U64(v) => i64::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn f64(&self, key: &str) -> Option<f64> {
        match self.metadata(key)? {
            MetadataValue::F64(v) => Some(*v),
            MetadataValue::U64(v) => Some(*v as f64),
            MetadataValue::I64(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn string(&self, key: &str) -> Option<&str> {
        match self.metadata(key)? {
            MetadataValue::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn i64_array(&self, key: &str) -> Option<Vec<i64>> {
        match self.metadata(key)? {
            MetadataValue::I64Array(v) => Some(v.clone()),
            MetadataValue::U64Array(v) => v.iter().copied().map(i64::try_from).collect::<Result<Vec<_>, _>>().ok(),
            _ => None,
        }
    }

    pub fn u64_array(&self, key: &str) -> Option<Vec<u64>> {
        match self.metadata(key)? {
            MetadataValue::U64Array(v) => Some(v.clone()),
            MetadataValue::I64Array(v) => v.iter().copied().map(u64::try_from).collect::<Result<Vec<_>, _>>().ok(),
            _ => None,
        }
    }

    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>, String> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| format!("missing GGUF tensor {name}"))?;
        let len = usize::try_from(tensor.stored_bytes)
            .map_err(|_| format!("{name}: tensor is too large to materialize"))?;
        self.read_absolute(tensor.offset, len)
    }

    /// Stored bytes for one logical row. GGUF's first dimension is the
    /// contiguous input dimension; all later dimensions enumerate rows.
    pub fn row_stored_bytes(&self, name: &str) -> Result<usize, String> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| format!("missing GGUF tensor {name}"))?;
        let cols = *tensor
            .dims
            .first()
            .ok_or_else(|| format!("{name}: tensor has no dimensions"))?;
        usize::try_from(tensor.dtype.stored_bytes(cols)?)
            .map_err(|_| format!("{name}: row byte count exceeds usize"))
    }

    pub fn read_row(&self, name: &str, row: usize) -> Result<Vec<u8>, String> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| format!("missing GGUF tensor {name}"))?;
        let row_bytes = self.row_stored_bytes(name)?;
        let rows = tensor
            .dims
            .iter()
            .skip(1)
            .try_fold(1_u64, |n, d| n.checked_mul(*d))
            .ok_or_else(|| format!("{name}: row count overflow"))?;
        if row as u64 >= rows {
            return Err(format!("{name}: row {row} >= {rows}"));
        }
        self.read_tensor_range(name, row as u64 * row_bytes as u64, row_bytes)
    }

    pub fn read_row_f32(&self, name: &str, row: usize) -> Result<Vec<f32>, String> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| format!("missing GGUF tensor {name}"))?;
        let cols = tensor.dims[0] as usize;
        decode_row(tensor.dtype, &self.read_row(name, row)?, cols)
    }

    pub fn read_tensor_range(&self, name: &str, offset: u64, bytes: usize) -> Result<Vec<u8>, String> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| format!("missing GGUF tensor {name}"))?;
        let end = offset
            .checked_add(bytes as u64)
            .ok_or_else(|| format!("{name}: range overflow"))?;
        if end > tensor.stored_bytes {
            return Err(format!("{name}: range {offset}..{end} exceeds {} bytes", tensor.stored_bytes));
        }
        self.read_absolute(tensor.offset + offset, bytes)
    }

    pub fn read_absolute(&self, offset: u64, bytes: usize) -> Result<Vec<u8>, String> {
        let mut out = vec![0_u8; bytes];
        let mut file = self.file.lock().map_err(|_| "GGUF file lock poisoned".to_string())?;
        file.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
        file.read_exact(&mut out).map_err(|e| e.to_string())?;
        Ok(out)
    }

    /// Read one expert's byte-exact slice from a GGUF 3-D routed tensor.
    pub fn read_expert_slice(&self, name: &str, expert: usize) -> Result<Vec<u8>, String> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| format!("missing GGUF expert tensor {name}"))?;
        let experts = tensor
            .dims
            .last()
            .copied()
            .ok_or_else(|| format!("{name}: missing expert dimension"))? as usize;
        if expert >= experts {
            return Err(format!("{name}: expert {expert} >= {experts}"));
        }
        if tensor.stored_bytes % experts as u64 != 0 {
            return Err(format!("{name}: expert tensor is not evenly sliceable"));
        }
        let stride = tensor.stored_bytes / experts as u64;
        self.read_tensor_range(name, stride * expert as u64, stride as usize)
    }
}

fn meta_u64(value: &MetadataValue) -> Option<u64> {
    match value {
        MetadataValue::U64(v) => Some(*v),
        MetadataValue::I64(v) => u64::try_from(*v).ok(),
        _ => None,
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, String> {
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .ok_or_else(|| "GGUF alignment overflow".into())
}

fn read_exact<const N: usize>(file: &mut File) -> Result<[u8; N], String> {
    let mut out = [0_u8; N];
    file.read_exact(&mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

fn read_u8(file: &mut File) -> Result<u8, String> {
    Ok(read_exact::<1>(file)?[0])
}
fn read_i8(file: &mut File) -> Result<i8, String> {
    Ok(read_u8(file)? as i8)
}
fn read_u16(file: &mut File) -> Result<u16, String> {
    Ok(u16::from_le_bytes(read_exact(file)?))
}
fn read_i16(file: &mut File) -> Result<i16, String> {
    Ok(i16::from_le_bytes(read_exact(file)?))
}
fn read_u32(file: &mut File) -> Result<u32, String> {
    Ok(u32::from_le_bytes(read_exact(file)?))
}
fn read_i32(file: &mut File) -> Result<i32, String> {
    Ok(i32::from_le_bytes(read_exact(file)?))
}
fn read_u64(file: &mut File) -> Result<u64, String> {
    Ok(u64::from_le_bytes(read_exact(file)?))
}
fn read_i64(file: &mut File) -> Result<i64, String> {
    Ok(i64::from_le_bytes(read_exact(file)?))
}
fn read_f32(file: &mut File) -> Result<f32, String> {
    Ok(f32::from_le_bytes(read_exact(file)?))
}
fn read_f64(file: &mut File) -> Result<f64, String> {
    Ok(f64::from_le_bytes(read_exact(file)?))
}

fn read_string(file: &mut File) -> Result<String, String> {
    let len = read_u64(file)?;
    if len > MAX_STRING_BYTES {
        return Err(format!("GGUF string length {len} exceeds safety limit"));
    }
    let mut bytes = vec![0_u8; len as usize];
    file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    String::from_utf8(bytes).map_err(|e| format!("GGUF string is not UTF-8: {e}"))
}

fn skip_bytes(file: &mut File, bytes: u64) -> Result<(), String> {
    let delta = i64::try_from(bytes).map_err(|_| "GGUF skip exceeds i64".to_string())?;
    file.seek(SeekFrom::Current(delta)).map_err(|e| e.to_string())?;
    Ok(())
}

fn scalar_width(ty: u32) -> Option<u64> {
    match ty {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4 | 5 | 6 => Some(4),
        10 | 11 | 12 => Some(8),
        _ => None,
    }
}

fn read_metadata_value(file: &mut File, ty: u32, keep_array: bool) -> Result<MetadataValue, String> {
    match ty {
        0 => Ok(MetadataValue::U64(read_u8(file)? as u64)),
        1 => Ok(MetadataValue::I64(read_i8(file)? as i64)),
        2 => Ok(MetadataValue::U64(read_u16(file)? as u64)),
        3 => Ok(MetadataValue::I64(read_i16(file)? as i64)),
        4 => Ok(MetadataValue::U64(read_u32(file)? as u64)),
        5 => Ok(MetadataValue::I64(read_i32(file)? as i64)),
        6 => Ok(MetadataValue::F64(read_f32(file)? as f64)),
        7 => Ok(MetadataValue::Bool(read_u8(file)? != 0)),
        8 => Ok(MetadataValue::String(read_string(file)?)),
        9 => {
            let elem_ty = read_u32(file)?;
            let len = read_u64(file)?;
            if !keep_array || len > MAX_METADATA_ARRAY {
                skip_array(file, elem_ty, len)?;
                return Ok(MetadataValue::IgnoredArray(len));
            }
            match elem_ty {
                0 | 2 | 4 | 10 => {
                    let mut out = Vec::with_capacity(len as usize);
                    for _ in 0..len {
                        out.push(match elem_ty {
                            0 => read_u8(file)? as u64,
                            2 => read_u16(file)? as u64,
                            4 => read_u32(file)? as u64,
                            10 => read_u64(file)?,
                            _ => unreachable!(),
                        });
                    }
                    Ok(MetadataValue::U64Array(out))
                }
                1 | 3 | 5 | 11 => {
                    let mut out = Vec::with_capacity(len as usize);
                    for _ in 0..len {
                        out.push(match elem_ty {
                            1 => read_i8(file)? as i64,
                            3 => read_i16(file)? as i64,
                            5 => read_i32(file)? as i64,
                            11 => read_i64(file)?,
                            _ => unreachable!(),
                        });
                    }
                    Ok(MetadataValue::I64Array(out))
                }
                6 | 12 => {
                    let mut out = Vec::with_capacity(len as usize);
                    for _ in 0..len {
                        out.push(if elem_ty == 6 { read_f32(file)? as f64 } else { read_f64(file)? });
                    }
                    Ok(MetadataValue::F64Array(out))
                }
                _ => {
                    skip_array(file, elem_ty, len)?;
                    Ok(MetadataValue::IgnoredArray(len))
                }
            }
        }
        10 => Ok(MetadataValue::U64(read_u64(file)?)),
        11 => Ok(MetadataValue::I64(read_i64(file)?)),
        12 => Ok(MetadataValue::F64(read_f64(file)?)),
        other => Err(format!("unsupported GGUF metadata value type {other}")),
    }
}

fn skip_array(file: &mut File, elem_ty: u32, len: u64) -> Result<(), String> {
    if elem_ty == 8 {
        for _ in 0..len {
            let n = read_u64(file)?;
            if n > MAX_STRING_BYTES {
                return Err(format!("GGUF array string length {n} exceeds safety limit"));
            }
            skip_bytes(file, n)?;
        }
        return Ok(());
    }
    if elem_ty == 9 {
        return Err("nested GGUF arrays are unsupported".into());
    }
    let width = scalar_width(elem_ty)
        .ok_or_else(|| format!("unsupported GGUF array element type {elem_ty}"))?;
    let bytes = width
        .checked_mul(len)
        .ok_or_else(|| "GGUF array skip overflow".to_string())?;
    skip_bytes(file, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn push_string(out: &mut Vec<u8>, value: &str) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }

    #[test]
    fn ggml_block_sizes_match_q4km_checkpoint_formats() {
        assert_eq!(GgmlType::Q4K.stored_bytes(256).unwrap(), 144);
        assert_eq!(GgmlType::Q5_0.stored_bytes(32).unwrap(), 22);
        assert_eq!(GgmlType::Q6K.stored_bytes(256).unwrap(), 210);
        assert_eq!(GgmlType::Q8_0.stored_bytes(32).unwrap(), 34);
        assert_eq!(GgmlType::Bf16.stored_bytes(7).unwrap(), 14);
        assert!(GgmlType::Q4K.stored_bytes(255).is_err());
    }

    #[test]
    fn parses_v3_tensor_offsets_without_materializing_tokenizer_arrays() {
        let root = std::env::temp_dir().join(format!("logan-gguf-test-{}", std::process::id()));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes()); // tensors
        bytes.extend_from_slice(&3_u64.to_le_bytes()); // kv

        push_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_string(&mut bytes, "qwen4exp");

        push_string(&mut bytes, "general.alignment");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&32_u32.to_le_bytes());

        push_string(&mut bytes, "tokenizer.ggml.tokens");
        bytes.extend_from_slice(&9_u32.to_le_bytes());
        bytes.extend_from_slice(&8_u32.to_le_bytes()); // string element type
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        push_string(&mut bytes, "a");
        push_string(&mut bytes, "bc");

        push_string(&mut bytes, "weight");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&32_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&8_u32.to_le_bytes()); // Q8_0
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        let data_start = bytes.len();
        bytes.extend_from_slice(&[0_u8; 34]);
        let mut f = File::create(&root).unwrap();
        f.write_all(&bytes).unwrap();
        drop(f);

        let src = GgufSource::open(&root).unwrap();
        assert_eq!(src.data_start(), data_start as u64);
        assert_eq!(src.string("general.architecture"), Some("qwen4exp"));
        assert!(matches!(src.metadata("tokenizer.ggml.tokens"), Some(MetadataValue::IgnoredArray(2))));
        let t = src.tensor("weight").unwrap();
        assert_eq!(t.dtype, GgmlType::Q8_0);
        assert_eq!(t.stored_bytes, 34);
        assert_eq!(src.read_tensor("weight").unwrap().len(), 34);
        std::fs::remove_file(root).unwrap();
    }
}

// ---------------------------------------------------------------------------
// CPU quant oracle / fallback
// ---------------------------------------------------------------------------

/// Dot one logical GGML row with an f32 activation without materializing a
/// dequantized copy. This is the correctness oracle for the CUDA path
/// ([`cuda_q4k`]), which mirrors it — including its loop nesting, so the two
/// accumulate in the same order and agree far more closely than a general f32
/// reassociation would.
pub fn dot_row(dtype: GgmlType, row: &[u8], x: &[f32]) -> Result<f32, String> {
    let expected = dtype.stored_bytes(x.len() as u64)? as usize;
    if row.len() != expected {
        return Err(format!(
            "{} row has {} bytes, expected {expected} for {} values",
            dtype.name(),
            row.len(),
            x.len()
        ));
    }
    let mut acc = 0.0_f32;
    match dtype {
        GgmlType::F32 => {
            for (i, raw) in row.chunks_exact(4).enumerate() {
                acc += f32::from_le_bytes(raw.try_into().unwrap()) * x[i];
            }
        }
        GgmlType::F16 => {
            for (i, raw) in row.chunks_exact(2).enumerate() {
                acc += f16_to_f32(u16::from_le_bytes(raw.try_into().unwrap())) * x[i];
            }
        }
        GgmlType::Bf16 => {
            for (i, raw) in row.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes(raw.try_into().unwrap());
                acc += f32::from_bits((bits as u32) << 16) * x[i];
            }
        }
        GgmlType::Q8_0 => {
            for (block_index, block) in row.chunks_exact(34).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let xb = &x[block_index * 32..block_index * 32 + 32];
                for i in 0..32 {
                    acc += d * (block[2 + i] as i8 as f32) * xb[i];
                }
            }
        }
        GgmlType::Q5_0 => {
            for (block_index, block) in row.chunks_exact(22).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let qh = u32::from_le_bytes(block[2..6].try_into().unwrap());
                let qs = &block[6..22];
                let xb = &x[block_index * 32..block_index * 32 + 32];
                for i in 0..16 {
                    let lo_hi = (((qh >> i) & 1) as i32) << 4;
                    let hi_hi = (((qh >> (i + 16)) & 1) as i32) << 4;
                    let lo = ((qs[i] & 0x0f) as i32 | lo_hi) - 16;
                    let hi = ((qs[i] >> 4) as i32 | hi_hi) - 16;
                    acc += d * lo as f32 * xb[i];
                    acc += d * hi as f32 * xb[i + 16];
                }
            }
        }
        GgmlType::Q4K => {
            for (block_index, block) in row.chunks_exact(144).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                let scales = &block[4..16];
                let qs = &block[16..144];
                let xb = &x[block_index * 256..block_index * 256 + 256];
                for group in 0..8 {
                    let (scale, min) = q4k_scale_min(group, scales);
                    let ds = d * scale as f32;
                    let dm = dmin * min as f32;
                    let pair = group / 2;
                    let high = group & 1 != 0;
                    let xoff = group * 32;
                    let qoff = pair * 32;
                    for i in 0..32 {
                        let packed = qs[qoff + i];
                        let q = if high { packed >> 4 } else { packed & 0x0f };
                        acc += (ds * q as f32 - dm) * xb[xoff + i];
                    }
                }
            }
        }
        GgmlType::Q6K => {
            for (block_index, block) in row.chunks_exact(210).enumerate() {
                let ql = &block[..128];
                let qh = &block[128..192];
                let scales = &block[192..208];
                let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
                let xb = &x[block_index * 256..block_index * 256 + 256];
                for ip in 0..2 {
                    for il in 0..32 {
                        let is = 8 * ip + il / 16;
                        let low0 = ql[64 * ip + il];
                        let low1 = ql[64 * ip + il + 32];
                        let high = qh[32 * ip + il];
                        let vals = [
                            ((low0 & 0x0f) | ((high & 0x03) << 4)) as i32 - 32,
                            ((low1 & 0x0f) | (((high >> 2) & 0x03) << 4)) as i32 - 32,
                            ((low0 >> 4) | (((high >> 4) & 0x03) << 4)) as i32 - 32,
                            ((low1 >> 4) | (((high >> 6) & 0x03) << 4)) as i32 - 32,
                        ];
                        let offsets = [0, 32, 64, 96];
                        let scale_offsets = [0, 2, 4, 6];
                        for lane in 0..4 {
                            let scale = scales[is + scale_offsets[lane]] as i8 as f32;
                            let xi = 128 * ip + il + offsets[lane];
                            acc += d * scale * vals[lane] as f32 * xb[xi];
                        }
                    }
                }
            }
        }
    }
    Ok(acc)
}

pub fn decode_row(dtype: GgmlType, row: &[u8], elements: usize) -> Result<Vec<f32>, String> {
    let expected = dtype.stored_bytes(elements as u64)? as usize;
    if row.len() != expected {
        return Err(format!(
            "{} row has {} bytes, expected {expected} for {elements} values",
            dtype.name(), row.len()
        ));
    }
    let mut out = vec![0.0_f32; elements];
    match dtype {
        GgmlType::F32 => {
            for (dst, raw) in out.iter_mut().zip(row.chunks_exact(4)) {
                *dst = f32::from_le_bytes(raw.try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (dst, raw) in out.iter_mut().zip(row.chunks_exact(2)) {
                *dst = f16_to_f32(u16::from_le_bytes(raw.try_into().unwrap()));
            }
        }
        GgmlType::Bf16 => {
            for (dst, raw) in out.iter_mut().zip(row.chunks_exact(2)) {
                let bits = u16::from_le_bytes(raw.try_into().unwrap());
                *dst = f32::from_bits((bits as u32) << 16);
            }
        }
        GgmlType::Q8_0 => {
            for (bi, block) in row.chunks_exact(34).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                for i in 0..32 {
                    out[bi * 32 + i] = d * block[2 + i] as i8 as f32;
                }
            }
        }
        GgmlType::Q5_0 => {
            for (bi, block) in row.chunks_exact(22).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let qh = u32::from_le_bytes(block[2..6].try_into().unwrap());
                let qs = &block[6..22];
                for i in 0..16 {
                    let lo = ((qs[i] & 0x0f) as i32 | ((((qh >> i) & 1) as i32) << 4)) - 16;
                    let hi = ((qs[i] >> 4) as i32 | ((((qh >> (i + 16)) & 1) as i32) << 4)) - 16;
                    out[bi * 32 + i] = d * lo as f32;
                    out[bi * 32 + i + 16] = d * hi as f32;
                }
            }
        }
        GgmlType::Q4K => {
            for (bi, block) in row.chunks_exact(144).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                let scales = &block[4..16];
                let qs = &block[16..144];
                for group in 0..8 {
                    let (scale, min) = q4k_scale_min(group, scales);
                    let ds = d * scale as f32;
                    let dm = dmin * min as f32;
                    let pair = group / 2;
                    let high = group & 1 != 0;
                    for i in 0..32 {
                        let packed = qs[pair * 32 + i];
                        let q = if high { packed >> 4 } else { packed & 0x0f };
                        out[bi * 256 + group * 32 + i] = ds * q as f32 - dm;
                    }
                }
            }
        }
        GgmlType::Q6K => {
            for (bi, block) in row.chunks_exact(210).enumerate() {
                let ql = &block[..128];
                let qh = &block[128..192];
                let scales = &block[192..208];
                let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
                for ip in 0..2 {
                    for il in 0..32 {
                        let is = 8 * ip + il / 16;
                        let low0 = ql[64 * ip + il];
                        let low1 = ql[64 * ip + il + 32];
                        let high = qh[32 * ip + il];
                        let vals = [
                            ((low0 & 0x0f) | ((high & 0x03) << 4)) as i32 - 32,
                            ((low1 & 0x0f) | (((high >> 2) & 0x03) << 4)) as i32 - 32,
                            ((low0 >> 4) | (((high >> 4) & 0x03) << 4)) as i32 - 32,
                            ((low1 >> 4) | (((high >> 6) & 0x03) << 4)) as i32 - 32,
                        ];
                        let offsets = [0, 32, 64, 96];
                        let scale_offsets = [0, 2, 4, 6];
                        for lane in 0..4 {
                            let xi = 128 * ip + il + offsets[lane];
                            out[bi * 256 + xi] = d
                                * scales[is + scale_offsets[lane]] as i8 as f32
                                * vals[lane] as f32;
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Six-bit scale/minimum unpacking for Q4_K groups.
///
/// `pub(crate)` rather than private: [`cuda_q4k`] both transliterates this into
/// CUDA and checks its own block-packing helper against it, and duplicating the
/// bit-twiddling in a third place is how the kernel and the oracle would drift
/// apart without anyone noticing.
#[inline]
pub(crate) fn q4k_scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    debug_assert!(j < 8 && q.len() >= 12);
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// IEEE-754 binary16 -> f32 without a dependency on a half-precision crate.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x03ff) as u32;
    let out = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut mant = frac;
                let mut e = -14_i32;
                while mant & 0x0400 == 0 {
                    mant <<= 1;
                    e -= 1;
                }
                mant &= 0x03ff;
                sign | (((e + 127) as u32) << 23) | (mant << 13)
            }
        }
        31 => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | (((exp - 15 + 127) as u32) << 23) | (frac << 13),
    };
    f32::from_bits(out)
}

/// CUDA execution for Q4_K rows, next to the [`dot_row`] oracle it must agree
/// with.
///
/// A submodule of this file rather than of the crate root: the layout constants
/// (`q4k_scale_min`, the 144-byte block) are here, and the module is gated on
/// `target_arch` only by what it can load at runtime, not by compilation — the
/// loading path is `dlopen`/`LoadLibrary`, so this compiles everywhere and
/// simply reports itself unavailable where there is no CUDA runtime.
pub mod cuda_q4k;
