//! Native GGUF v3 source for Qwen4Exp / Qwen3.8-Flash-Next.
//!
//! This module deliberately owns only the container/source boundary.  It is
//! not an inference backend and it never requantizes weights. Quantized tensor
//! payloads remain byte-for-byte GGUF blocks and are range-read directly from
//! the source file. The runtime decides how to execute those blocks.

use crate::gguf_iq_tables::{
    IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS,
};

#[cfg(unix)]
use std::os::fd::AsRawFd;

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
    Q5K,
    Q6K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    Bf16,
    Q2_0,
}

impl GgmlType {
    pub fn from_id(id: u32) -> Result<Self, String> {
        match id {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            6 => Ok(Self::Q5_0),
            8 => Ok(Self::Q8_0),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            16 => Ok(Self::Iq2Xxs),
            17 => Ok(Self::Iq2Xs),
            18 => Ok(Self::Iq3Xxs),
            20 => Ok(Self::Iq4Nl),
            21 => Ok(Self::Iq3S),
            22 => Ok(Self::Iq2S),
            23 => Ok(Self::Iq4Xs),
            30 => Ok(Self::Bf16),
            42 => Ok(Self::Q2_0),
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
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Iq2Xxs => 16,
            Self::Iq2Xs => 17,
            Self::Iq3Xxs => 18,
            Self::Iq4Nl => 20,
            Self::Iq3S => 21,
            Self::Iq2S => 22,
            Self::Iq4Xs => 23,
            Self::Bf16 => 30,
            Self::Q2_0 => 42,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q5_0 => "Q5_0",
            Self::Q8_0 => "Q8_0",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Iq2Xs => "IQ2_XS",
            Self::Iq3Xxs => "IQ3_XXS",
            Self::Iq4Nl => "IQ4_NL",
            Self::Iq3S => "IQ3_S",
            Self::Iq2S => "IQ2_S",
            Self::Iq4Xs => "IQ4_XS",
            Self::Bf16 => "BF16",
            Self::Q2_0 => "Q2_0",
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
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Iq2Xxs => (256, 66),
            Self::Iq2Xs => (256, 74),
            Self::Iq3Xxs => (256, 98),
            Self::Iq4Nl => (32, 18),
            Self::Iq3S => (256, 110),
            Self::Iq2S => (256, 82),
            Self::Iq4Xs => (256, 136),
            Self::Q2_0 => (64, 18),
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
    /// Zero-based GGUF shard containing this tensor.
    pub shard: usize,
    /// Absolute byte offset to the first quant block / scalar within its shard.
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

#[cfg(unix)]
struct MappedShard {
    ptr: *const u8,
    len: usize,
}

#[cfg(unix)]
unsafe impl Send for MappedShard {}
#[cfg(unix)]
unsafe impl Sync for MappedShard {}

#[cfg(unix)]
impl MappedShard {
    fn map(file: &File) -> Result<Self, String> {
        let len = usize::try_from(file.metadata().map_err(|e| e.to_string())?.len())
            .map_err(|_| "GGUF shard is too large to map on this platform".to_string())?;
        if len == 0 {
            return Err("cannot mmap empty GGUF shard".into());
        }
        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return Err(format!(
                "mmap GGUF shard failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(Self {
            ptr: raw.cast::<u8>(),
            len,
        })
    }

    fn slice(&self, offset: u64, bytes: usize) -> Option<&[u8]> {
        let start = usize::try_from(offset).ok()?;
        let end = start.checked_add(bytes)?;
        if end > self.len {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(self.ptr.add(start), bytes) })
    }
}

#[cfg(unix)]
impl Drop for MappedShard {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.cast_mut().cast(), self.len);
        }
    }
}

#[derive(Clone)]
pub struct GgufSource {
    path: PathBuf,
    files: Arc<Vec<Mutex<File>>>,
    #[cfg(unix)]
    maps: Arc<Vec<Option<MappedShard>>>,
    metadata: Arc<BTreeMap<String, MetadataValue>>,
    tensors: Arc<BTreeMap<String, TensorInfo>>,
    data_start: u64,
    alignment: u64,
}

struct ParsedGguf {
    file: File,
    metadata: BTreeMap<String, MetadataValue>,
    tensors: BTreeMap<String, TensorInfo>,
    data_start: u64,
    alignment: u64,
}

fn parse_gguf_file(path: &Path, shard: usize) -> Result<ParsedGguf, String> {
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
        return Err(format!(
            "GGUF version {version} is not supported; expected v3"
        ));
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
            shard,
            offset,
            stored_bytes: tensor.stored_bytes,
        };
        if tensors.insert(tensor.name.clone(), info).is_some() {
            return Err(format!("duplicate GGUF tensor {}", tensor.name));
        }
    }

    Ok(ParsedGguf {
        file,
        metadata,
        tensors,
        data_start,
        alignment,
    })
}

fn split_sibling_path(path: &Path, index: usize, count: usize) -> Result<PathBuf, String> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("GGUF split path is not UTF-8: {}", path.display()))?;
    let first_suffix = format!("-{:05}-of-{:05}.gguf", 1, count);
    let prefix = name.strip_suffix(&first_suffix).ok_or_else(|| {
        format!(
            "split GGUF must be opened via its first shard (*{first_suffix}); got {}",
            path.display()
        )
    })?;
    let sibling = format!("{prefix}-{:05}-of-{:05}.gguf", index + 1, count);
    Ok(path.with_file_name(sibling))
}

impl GgufSource {
    pub fn open(path: &Path) -> Result<Self, String> {
        let first = parse_gguf_file(path, 0)?;
        let arch = first
            .metadata
            .get("general.architecture")
            .and_then(|value| match value {
                MetadataValue::String(value) => Some(value.as_str()),
                _ => None,
            })
            .unwrap_or("")
            .to_owned();
        if arch != "qwen4exp" {
            return Err(format!(
                "unsupported GGUF architecture {arch:?}; this path currently requires qwen4exp"
            ));
        }

        let split_count = first
            .metadata
            .get("split.count")
            .and_then(meta_u64)
            .unwrap_or(1);
        let split_count = usize::try_from(split_count)
            .map_err(|_| "GGUF split count exceeds usize".to_string())?;
        if split_count == 0 || split_count > 1024 {
            return Err(format!("invalid GGUF split count {split_count}"));
        }
        let first_split_no = first
            .metadata
            .get("split.no")
            .and_then(meta_u64)
            .unwrap_or(0);
        if split_count > 1 && first_split_no != 0 {
            return Err(format!(
                "split GGUF must be opened via shard 0; {} reports split.no={first_split_no}",
                path.display()
            ));
        }

        let data_start = first.data_start;
        let alignment = first.alignment;
        #[cfg(unix)]
        let first_map = MappedShard::map(&first.file).ok();
        let metadata = first.metadata;
        let mut tensors = first.tensors;
        let mut files = vec![Mutex::new(first.file)];
        #[cfg(unix)]
        let mut maps = vec![first_map];

        for shard in 1..split_count {
            let shard_path = split_sibling_path(path, shard, split_count)?;
            let parsed = parse_gguf_file(&shard_path, shard)?;
            let parsed_count = parsed
                .metadata
                .get("split.count")
                .and_then(meta_u64)
                .unwrap_or(1);
            let parsed_no = parsed
                .metadata
                .get("split.no")
                .and_then(meta_u64)
                .unwrap_or(u64::MAX);
            if parsed_count != split_count as u64 || parsed_no != shard as u64 {
                return Err(format!(
                    "{} has split.no={parsed_no}, split.count={parsed_count}; expected {shard}/{split_count}",
                    shard_path.display()
                ));
            }
            let parsed_arch = parsed
                .metadata
                .get("general.architecture")
                .and_then(|value| match value {
                    MetadataValue::String(value) => Some(value.as_str()),
                    _ => None,
                })
                .unwrap_or("");
            // Continuation shards produced by llama.cpp/gguf-split may omit
            // general.architecture entirely. The first shard is authoritative;
            // if a later shard does declare an architecture, it must agree.
            if !parsed_arch.is_empty() && parsed_arch != arch {
                return Err(format!(
                    "{} architecture {parsed_arch:?} does not match first shard {arch:?}",
                    shard_path.display()
                ));
            }
            for (name, info) in parsed.tensors {
                if tensors.insert(name.clone(), info).is_some() {
                    return Err(format!("duplicate GGUF tensor {name} across split shards"));
                }
            }
            #[cfg(unix)]
            let parsed_map = MappedShard::map(&parsed.file).ok();
            files.push(Mutex::new(parsed.file));
            #[cfg(unix)]
            maps.push(parsed_map);
        }

        if let Some(expected) = metadata.get("split.tensors.count").and_then(meta_u64) {
            if expected != tensors.len() as u64 {
                return Err(format!(
                    "split GGUF tensor count mismatch: metadata={expected}, parsed={}",
                    tensors.len()
                ));
            }
        }

        Ok(Self {
            path: path.to_owned(),
            files: Arc::new(files),
            #[cfg(unix)]
            maps: Arc::new(maps),
            metadata: Arc::new(metadata),
            tensors: Arc::new(tensors),
            data_start,
            alignment,
        })
    }

    pub fn shard_count(&self) -> usize {
        self.files.len()
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
            MetadataValue::U64Array(v) => v
                .iter()
                .copied()
                .map(i64::try_from)
                .collect::<Result<Vec<_>, _>>()
                .ok(),
            _ => None,
        }
    }

    pub fn u64_array(&self, key: &str) -> Option<Vec<u64>> {
        match self.metadata(key)? {
            MetadataValue::U64Array(v) => Some(v.clone()),
            MetadataValue::I64Array(v) => v
                .iter()
                .copied()
                .map(u64::try_from)
                .collect::<Result<Vec<_>, _>>()
                .ok(),
            _ => None,
        }
    }

    /// Borrow a tensor directly from the read-only shard mapping without
    /// copying it into the heap. On non-Unix targets or if mmap failed, the
    /// caller can fall back to the ordinary range-read path.
    pub fn mapped_tensor(&self, name: &str) -> Option<&[u8]> {
        let tensor = self.tensor(name)?;
        #[cfg(unix)]
        {
            let map = self.maps.get(tensor.shard)?.as_ref()?;
            let len = usize::try_from(tensor.stored_bytes).ok()?;
            return map.slice(tensor.offset, len);
        }
        #[cfg(not(unix))]
        {
            let _ = tensor;
            None
        }
    }

    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>, String> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| format!("missing GGUF tensor {name}"))?;
        let len = usize::try_from(tensor.stored_bytes)
            .map_err(|_| format!("{name}: tensor is too large to materialize"))?;
        self.read_absolute_from_shard(tensor.shard, tensor.offset, len)
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

    pub fn read_tensor_range(
        &self,
        name: &str,
        offset: u64,
        bytes: usize,
    ) -> Result<Vec<u8>, String> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| format!("missing GGUF tensor {name}"))?;
        let end = offset
            .checked_add(bytes as u64)
            .ok_or_else(|| format!("{name}: range overflow"))?;
        if end > tensor.stored_bytes {
            return Err(format!(
                "{name}: range {offset}..{end} exceeds {} bytes",
                tensor.stored_bytes
            ));
        }
        self.read_absolute_from_shard(tensor.shard, tensor.offset + offset, bytes)
    }

    pub fn read_absolute(&self, offset: u64, bytes: usize) -> Result<Vec<u8>, String> {
        self.read_absolute_from_shard(0, offset, bytes)
    }

    fn read_absolute_from_shard(
        &self,
        shard: usize,
        offset: u64,
        bytes: usize,
    ) -> Result<Vec<u8>, String> {
        let mut out = vec![0_u8; bytes];
        let file = self
            .files
            .get(shard)
            .ok_or_else(|| format!("GGUF shard index {shard} is out of range"))?;
        let mut file = file
            .lock()
            .map_err(|_| format!("GGUF shard {shard} file lock poisoned"))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| e.to_string())?;
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
            .ok_or_else(|| format!("{name}: missing expert dimension"))?
            as usize;
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
    file.seek(SeekFrom::Current(delta))
        .map_err(|e| e.to_string())?;
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

fn read_metadata_value(
    file: &mut File,
    ty: u32,
    keep_array: bool,
) -> Result<MetadataValue, String> {
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
                        out.push(if elem_ty == 6 {
                            read_f32(file)? as f64
                        } else {
                            read_f64(file)?
                        });
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

    fn deterministic_block(dtype: GgmlType, elements: usize, scale_bytes: usize) -> Vec<u8> {
        let bytes = dtype.stored_bytes(elements as u64).unwrap() as usize;
        let mut raw = (0..bytes)
            .map(|i| (((i * 73 + 19) ^ ((i * 11) >> 1)) & 0xff) as u8)
            .collect::<Vec<_>>();
        raw[0] = 0x00;
        raw[1] = 0x3c; // FP16 1.0
        if scale_bytes >= 4 {
            raw[2] = 0x00;
            raw[3] = 0x38; // FP16 0.5
        }
        raw
    }

    fn assert_reference_bits(
        dtype: GgmlType,
        elements: usize,
        scale_bytes: usize,
        expected: &[(usize, u32)],
    ) {
        let raw = deterministic_block(dtype, elements, scale_bytes);
        let decoded = decode_row(dtype, &raw, elements).unwrap();
        for &(index, bits) in expected {
            assert_eq!(
                decoded[index].to_bits(),
                bits,
                "{} index {index}",
                dtype.name()
            );
        }
    }

    fn write_split_fixture(
        path: &Path,
        shard_no: u64,
        include_arch: bool,
        tensor_name: &str,
        fill: u8,
    ) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes()); // one tensor in this shard
        let kv_count = if include_arch { 4_u64 } else { 2_u64 };
        bytes.extend_from_slice(&kv_count.to_le_bytes());

        if include_arch {
            push_string(&mut bytes, "general.architecture");
            bytes.extend_from_slice(&8_u32.to_le_bytes()); // string
            push_string(&mut bytes, "qwen4exp");
        }

        push_string(&mut bytes, "split.no");
        bytes.extend_from_slice(&10_u32.to_le_bytes()); // u64
        bytes.extend_from_slice(&shard_no.to_le_bytes());

        push_string(&mut bytes, "split.count");
        bytes.extend_from_slice(&10_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());

        if include_arch {
            push_string(&mut bytes, "split.tensors.count");
            bytes.extend_from_slice(&10_u32.to_le_bytes());
            bytes.extend_from_slice(&2_u64.to_le_bytes());
        }

        push_string(&mut bytes, tensor_name);
        bytes.extend_from_slice(&1_u32.to_le_bytes()); // rank
        bytes.extend_from_slice(&32_u64.to_le_bytes());
        bytes.extend_from_slice(&8_u32.to_le_bytes()); // Q8_0
        bytes.extend_from_slice(&0_u64.to_le_bytes()); // relative data offset

        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&[fill; 34]);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn split_gguf_merges_tensors_and_reads_from_owning_shard() {
        let root =
            std::env::temp_dir().join(format!("logan-gguf-split-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let first = root.join("model-00001-of-00002.gguf");
        let second = root.join("model-00002-of-00002.gguf");
        write_split_fixture(&first, 0, true, "first.weight", 0x11);
        // Real split continuations such as the ISTA Qwen3.8 PLE shard omit
        // general.architecture; shard 0 remains authoritative.
        write_split_fixture(&second, 1, false, "second.weight", 0x22);

        let src = GgufSource::open(&first).unwrap();
        assert_eq!(src.shard_count(), 2);
        assert_eq!(src.tensors().len(), 2);
        assert_eq!(src.tensor("first.weight").unwrap().shard, 0);
        assert_eq!(src.tensor("second.weight").unwrap().shard, 1);
        assert_eq!(src.read_tensor("first.weight").unwrap(), vec![0x11; 34]);
        assert_eq!(src.read_tensor("second.weight").unwrap(), vec![0x22; 34]);
        #[cfg(unix)]
        {
            assert_eq!(src.mapped_tensor("first.weight").unwrap(), &[0x11; 34]);
            assert_eq!(src.mapped_tensor("second.weight").unwrap(), &[0x22; 34]);

            let wt = crate::Wt {
                f: vec![],
                bytes: Some(crate::WtBytes::GgufMapped {
                    source: src.clone(),
                    name: "second.weight".to_string(),
                    dtype: GgmlType::Q8_0,
                }),
                o: 1,
                i: 32,
            };
            let x = vec![1.0_f32; 32];
            let mut y = vec![0.0_f32; 1];
            crate::matmul(&mut y, &x, &wt);
            let expected = dot_row(
                GgmlType::Q8_0,
                src.mapped_tensor("second.weight").unwrap(),
                &x,
            )
            .unwrap();
            assert_eq!(y[0].to_bits(), expected.to_bits());
        }

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mixed_iq_checkpoint_formats_match_upstream_reference_vectors() {
        // Reference bits were generated with the current upstream llama.cpp
        // dequantize_row_* implementations using deterministic raw blocks.
        // These catch block-geometry, lookup-table, sign, and scale packing drift.
        assert_reference_bits(
            GgmlType::Q2_0,
            64,
            2,
            &[
                (0, 0x3f800000),
                (7, 0x40000000),
                (31, 0x3f800000),
                (63, 0x3f800000),
            ],
        );
        assert_reference_bits(
            GgmlType::Q5K,
            256,
            4,
            &[
                (0, 0x4457e000),
                (15, 0xc1dc0000),
                (128, 0x43aac000),
                (255, 0x41c00000),
            ],
        );
        assert_reference_bits(
            GgmlType::Iq2Xxs,
            256,
            2,
            &[
                (0, 0x41980000),
                (7, 0xc26d8000),
                (128, 0x40a00000),
                (255, 0xc1f80000),
            ],
        );
        assert_reference_bits(
            GgmlType::Iq2Xs,
            256,
            2,
            &[
                (0, 0xc31be000),
                (15, 0x42b54000),
                (128, 0xc2548000),
                (255, 0xc23b8000),
            ],
        );
        assert_reference_bits(
            GgmlType::Iq2S,
            256,
            2,
            &[
                (0, 0x41a80000),
                (15, 0xc2e1c000),
                (128, 0x43066000),
                (255, 0xc1980000),
            ],
        );
        assert_reference_bits(
            GgmlType::Iq3Xxs,
            256,
            2,
            &[
                (0, 0x434f0000),
                (15, 0xc3210000),
                (128, 0x41880000),
                (255, 0x43a2c000),
            ],
        );
        assert_reference_bits(
            GgmlType::Iq3S,
            256,
            2,
            &[
                (0, 0x433d0000),
                (15, 0xc3670000),
                (128, 0x43070000),
                (255, 0x42960000),
            ],
        );
        assert_reference_bits(
            GgmlType::Iq4Nl,
            32,
            2,
            &[
                (0, 0x42b20000),
                (15, 0xc2d00000),
                (16, 0x41c80000),
                (31, 0x42180000),
            ],
        );
        assert_reference_bits(
            GgmlType::Iq4Xs,
            256,
            2,
            &[
                (0, 0xc1200000),
                (15, 0x42540000),
                (128, 0xc2a00000),
                (255, 0xc33e0000),
            ],
        );
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
        assert!(matches!(
            src.metadata("tokenizer.ggml.tokens"),
            Some(MetadataValue::IgnoredArray(2))
        ));
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

const IQ4_NL_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

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
        GgmlType::Q5K => {
            for (block_index, block) in row.chunks_exact(176).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                let scales = &block[4..16];
                let qh = &block[16..48];
                let ql = &block[48..176];
                let xb = &x[block_index * 256..block_index * 256 + 256];
                let mut is = 0usize;
                let mut u1 = 1u8;
                let mut u2 = 2u8;
                for j in (0..256).step_by(64) {
                    let (sc1, m1q) = q4k_scale_min(is, scales);
                    let (sc2, m2q) = q4k_scale_min(is + 1, scales);
                    let ds1 = d * sc1 as f32;
                    let dm1 = dmin * m1q as f32;
                    let ds2 = d * sc2 as f32;
                    let dm2 = dmin * m2q as f32;
                    let qoff = (j / 64) * 32;
                    for l in 0..32 {
                        let lo = (ql[qoff + l] & 0x0f) + if qh[l] & u1 != 0 { 16 } else { 0 };
                        let hi = (ql[qoff + l] >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 };
                        acc += (ds1 * lo as f32 - dm1) * xb[j + l];
                        acc += (ds2 * hi as f32 - dm2) * xb[j + 32 + l];
                    }
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        GgmlType::Q2_0 => {
            for (block_index, block) in row.chunks_exact(18).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let xb = &x[block_index * 64..block_index * 64 + 64];
                for j in 0..64 {
                    let q = (block[2 + j / 4] >> ((j % 4) * 2)) & 0x03;
                    acc += ((q as i32 - 1) as f32 * d) * xb[j];
                }
            }
        }
        GgmlType::Iq4Nl => {
            for (block_index, block) in row.chunks_exact(18).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let xb = &x[block_index * 32..block_index * 32 + 32];
                for j in 0..16 {
                    let packed = block[2 + j];
                    acc += d * IQ4_NL_VALUES[(packed & 0x0f) as usize] as f32 * xb[j];
                    acc += d * IQ4_NL_VALUES[(packed >> 4) as usize] as f32 * xb[j + 16];
                }
            }
        }
        GgmlType::Iq4Xs => {
            for (block_index, block) in row.chunks_exact(136).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let scales_h = u16::from_le_bytes([block[2], block[3]]);
                let scales_l = &block[4..8];
                let qs = &block[8..136];
                let xb = &x[block_index * 256..block_index * 256 + 256];
                for ib in 0..8 {
                    let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f;
                    let high = ((scales_h >> (2 * ib)) & 0x03) as u8;
                    let dl = d * (((low | (high << 4)) as i32 - 32) as f32);
                    let qoff = ib * 16;
                    let xoff = ib * 32;
                    for j in 0..16 {
                        let packed = qs[qoff + j];
                        acc += dl * IQ4_NL_VALUES[(packed & 0x0f) as usize] as f32 * xb[xoff + j];
                        acc +=
                            dl * IQ4_NL_VALUES[(packed >> 4) as usize] as f32 * xb[xoff + j + 16];
                    }
                }
            }
        }
        GgmlType::Iq2Xxs => {
            for (bi, block) in row.chunks_exact(66).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let qs = &block[2..66];
                let xb = &x[bi * 256..bi * 256 + 256];
                for ib32 in 0..8 {
                    let off = ib32 * 8;
                    let aux0 = u32::from_le_bytes(qs[off..off + 4].try_into().unwrap());
                    let aux1 = u32::from_le_bytes(qs[off + 4..off + 8].try_into().unwrap());
                    let aux0b = aux0.to_le_bytes();
                    let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
                    for l in 0..4 {
                        let grid = IQ2XXS_GRID[aux0b[l] as usize].to_le_bytes();
                        let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                        let xoff = ib32 * 32 + l * 8;
                        for j in 0..8 {
                            let sign = if signs & KMASK_IQ2XS[j] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            acc += db * grid[j] as f32 * sign * xb[xoff + j];
                        }
                    }
                }
            }
        }
        GgmlType::Iq2Xs => {
            for (bi, block) in row.chunks_exact(74).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let qs = &block[2..66];
                let scales = &block[66..74];
                let xb = &x[bi * 256..bi * 256 + 256];
                for ib32 in 0..8 {
                    let db = [
                        d * (0.5 + (scales[ib32] & 0x0f) as f32) * 0.25,
                        d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
                    ];
                    for l in 0..4 {
                        let qoff = 2 * (4 * ib32 + l);
                        let code = u16::from_le_bytes([qs[qoff], qs[qoff + 1]]);
                        let grid = IQ2XS_GRID[(code & 511) as usize].to_le_bytes();
                        let signs = KSIGNS_IQ2XS[(code >> 9) as usize];
                        let xoff = ib32 * 32 + l * 8;
                        for j in 0..8 {
                            let sign = if signs & KMASK_IQ2XS[j] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            acc += db[l / 2] * grid[j] as f32 * sign * xb[xoff + j];
                        }
                    }
                }
            }
        }
        GgmlType::Iq2S => {
            for (bi, block) in row.chunks_exact(82).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let qs_all = &block[2..66];
                let qs = &qs_all[..32];
                let signs_all = &qs_all[32..64];
                let qh = &block[66..74];
                let scales = &block[74..82];
                let xb = &x[bi * 256..bi * 256 + 256];
                for ib32 in 0..8 {
                    let db = [
                        d * (0.5 + (scales[ib32] & 0x0f) as f32) * 0.25,
                        d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
                    ];
                    for l in 0..4 {
                        let q = qs[4 * ib32 + l] as usize;
                        let hi = (((qh[ib32] as u16) << (8 - 2 * l)) & 0x300) as usize;
                        let grid = IQ2S_GRID[q | hi].to_le_bytes();
                        let signs = signs_all[4 * ib32 + l];
                        let xoff = ib32 * 32 + l * 8;
                        for j in 0..8 {
                            let sign = if signs & KMASK_IQ2XS[j] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            acc += db[l / 2] * grid[j] as f32 * sign * xb[xoff + j];
                        }
                    }
                }
            }
        }
        GgmlType::Iq3Xxs => {
            for (bi, block) in row.chunks_exact(98).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let qs = &block[2..98];
                let scales_and_signs = &qs[64..96];
                let xb = &x[bi * 256..bi * 256 + 256];
                for ib32 in 0..8 {
                    let aux = u32::from_le_bytes(
                        scales_and_signs[4 * ib32..4 * ib32 + 4].try_into().unwrap(),
                    );
                    let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                    let qbase = ib32 * 8;
                    for l in 0..4 {
                        let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                        let grid1 = IQ3XXS_GRID[qs[qbase + 2 * l] as usize].to_le_bytes();
                        let grid2 = IQ3XXS_GRID[qs[qbase + 2 * l + 1] as usize].to_le_bytes();
                        let xoff = ib32 * 32 + l * 8;
                        for j in 0..4 {
                            let sign0 = if signs & KMASK_IQ2XS[j] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            let sign1 = if signs & KMASK_IQ2XS[j + 4] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            acc += db * grid1[j] as f32 * sign0 * xb[xoff + j];
                            acc += db * grid2[j] as f32 * sign1 * xb[xoff + j + 4];
                        }
                    }
                }
            }
        }
        GgmlType::Iq3S => {
            for (bi, block) in row.chunks_exact(110).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let qs = &block[2..66];
                let qh = &block[66..74];
                let signs = &block[74..106];
                let scales = &block[106..110];
                let xb = &x[bi * 256..bi * 256 + 256];
                for pair in 0..4 {
                    let ib32 = pair * 2;
                    let db1 = d * (1 + 2 * (scales[pair] & 0x0f) as i32) as f32;
                    let db2 = d * (1 + 2 * (scales[pair] >> 4) as i32) as f32;
                    let qbase = pair * 16;
                    let sbase = pair * 8;
                    let h0 = qh[pair * 2];
                    let h1 = qh[pair * 2 + 1];

                    for l in 0..4 {
                        let idx1 = qs[qbase + 2 * l] as usize
                            | ((((h0 as u16) << (8 - 2 * l)) & 256) as usize);
                        let idx2 = qs[qbase + 2 * l + 1] as usize
                            | ((((h0 as u16) << (7 - 2 * l)) & 256) as usize);
                        let grid1 = IQ3S_GRID[idx1].to_le_bytes();
                        let grid2 = IQ3S_GRID[idx2].to_le_bytes();
                        let sb = signs[sbase + l];
                        let xoff = ib32 * 32 + l * 8;
                        for j in 0..4 {
                            let sign0 = if sb & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                            let sign1 = if sb & KMASK_IQ2XS[j + 4] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            acc += db1 * grid1[j] as f32 * sign0 * xb[xoff + j];
                            acc += db1 * grid2[j] as f32 * sign1 * xb[xoff + j + 4];
                        }
                    }

                    for l in 0..4 {
                        let idx1 = qs[qbase + 8 + 2 * l] as usize
                            | ((((h1 as u16) << (8 - 2 * l)) & 256) as usize);
                        let idx2 = qs[qbase + 8 + 2 * l + 1] as usize
                            | ((((h1 as u16) << (7 - 2 * l)) & 256) as usize);
                        let grid1 = IQ3S_GRID[idx1].to_le_bytes();
                        let grid2 = IQ3S_GRID[idx2].to_le_bytes();
                        let sb = signs[sbase + 4 + l];
                        let xoff = (ib32 + 1) * 32 + l * 8;
                        for j in 0..4 {
                            let sign0 = if sb & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                            let sign1 = if sb & KMASK_IQ2XS[j + 4] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            acc += db2 * grid1[j] as f32 * sign0 * xb[xoff + j];
                            acc += db2 * grid2[j] as f32 * sign1 * xb[xoff + j + 4];
                        }
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
            dtype.name(),
            row.len()
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
        GgmlType::Q5K => {
            for (bi, block) in row.chunks_exact(176).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                let scales = &block[4..16];
                let qh = &block[16..48];
                let ql = &block[48..176];
                let base = bi * 256;
                let mut is = 0usize;
                let mut u1 = 1u8;
                let mut u2 = 2u8;
                for j in (0..256).step_by(64) {
                    let (sc1, m1q) = q4k_scale_min(is, scales);
                    let (sc2, m2q) = q4k_scale_min(is + 1, scales);
                    let ds1 = d * sc1 as f32;
                    let dm1 = dmin * m1q as f32;
                    let ds2 = d * sc2 as f32;
                    let dm2 = dmin * m2q as f32;
                    let qoff = (j / 64) * 32;
                    for l in 0..32 {
                        let lo = (ql[qoff + l] & 0x0f) + if qh[l] & u1 != 0 { 16 } else { 0 };
                        let hi = (ql[qoff + l] >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 };
                        out[base + j + l] = ds1 * lo as f32 - dm1;
                        out[base + j + 32 + l] = ds2 * hi as f32 - dm2;
                    }
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        GgmlType::Q2_0 => {
            for (bi, block) in row.chunks_exact(18).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                for j in 0..64 {
                    let q = (block[2 + j / 4] >> ((j % 4) * 2)) & 0x03;
                    out[bi * 64 + j] = (q as i32 - 1) as f32 * d;
                }
            }
        }
        GgmlType::Iq4Nl => {
            for (bi, block) in row.chunks_exact(18).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                for j in 0..16 {
                    let packed = block[2 + j];
                    out[bi * 32 + j] = d * IQ4_NL_VALUES[(packed & 0x0f) as usize] as f32;
                    out[bi * 32 + j + 16] = d * IQ4_NL_VALUES[(packed >> 4) as usize] as f32;
                }
            }
        }
        GgmlType::Iq4Xs => {
            for (bi, block) in row.chunks_exact(136).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let scales_h = u16::from_le_bytes([block[2], block[3]]);
                let scales_l = &block[4..8];
                let qs = &block[8..136];
                for ib in 0..8 {
                    let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f;
                    let high = ((scales_h >> (2 * ib)) & 0x03) as u8;
                    let dl = d * (((low | (high << 4)) as i32 - 32) as f32);
                    let qoff = ib * 16;
                    let xoff = bi * 256 + ib * 32;
                    for j in 0..16 {
                        let packed = qs[qoff + j];
                        out[xoff + j] = dl * IQ4_NL_VALUES[(packed & 0x0f) as usize] as f32;
                        out[xoff + j + 16] = dl * IQ4_NL_VALUES[(packed >> 4) as usize] as f32;
                    }
                }
            }
        }
        GgmlType::Iq2Xxs => {
            for (bi, block) in row.chunks_exact(66).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let qs = &block[2..66];
                let base = bi * 256;
                for ib32 in 0..8 {
                    let off = ib32 * 8;
                    let aux0 = u32::from_le_bytes(qs[off..off + 4].try_into().unwrap());
                    let aux1 = u32::from_le_bytes(qs[off + 4..off + 8].try_into().unwrap());
                    let aux = [aux0.to_le_bytes(), aux1.to_le_bytes()].concat();
                    let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
                    for l in 0..4 {
                        let grid = IQ2XXS_GRID[aux[l] as usize].to_le_bytes();
                        let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                        for j in 0..8 {
                            let sign = if signs & KMASK_IQ2XS[j] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            out[base + ib32 * 32 + l * 8 + j] = db * grid[j] as f32 * sign;
                        }
                    }
                }
            }
        }
        GgmlType::Iq2Xs => {
            for (bi, block) in row.chunks_exact(74).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let qs = &block[2..66];
                let scales = &block[66..74];
                let base = bi * 256;
                for ib32 in 0..8 {
                    let db = [
                        d * (0.5 + (scales[ib32] & 0x0f) as f32) * 0.25,
                        d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
                    ];
                    for l in 0..4 {
                        let qoff = 2 * (4 * ib32 + l);
                        let code = u16::from_le_bytes([qs[qoff], qs[qoff + 1]]);
                        let grid = IQ2XS_GRID[(code & 511) as usize].to_le_bytes();
                        let signs = KSIGNS_IQ2XS[(code >> 9) as usize];
                        for j in 0..8 {
                            let sign = if signs & KMASK_IQ2XS[j] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            out[base + ib32 * 32 + l * 8 + j] = db[l / 2] * grid[j] as f32 * sign;
                        }
                    }
                }
            }
        }
        GgmlType::Iq2S => {
            for (bi, block) in row.chunks_exact(82).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let qs_all = &block[2..66];
                let qs = &qs_all[..32];
                let signs_all = &qs_all[32..64];
                let qh = &block[66..74];
                let scales = &block[74..82];
                let base = bi * 256;
                for ib32 in 0..8 {
                    let db = [
                        d * (0.5 + (scales[ib32] & 0x0f) as f32) * 0.25,
                        d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
                    ];
                    for l in 0..4 {
                        let q = qs[4 * ib32 + l] as usize;
                        let hi = (((qh[ib32] as u16) << (8 - 2 * l)) & 0x300) as usize;
                        let grid = IQ2S_GRID[q | hi].to_le_bytes();
                        let signs = signs_all[4 * ib32 + l];
                        for j in 0..8 {
                            let sign = if signs & KMASK_IQ2XS[j] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            out[base + ib32 * 32 + l * 8 + j] = db[l / 2] * grid[j] as f32 * sign;
                        }
                    }
                }
            }
        }
        GgmlType::Iq3Xxs => {
            for (bi, block) in row.chunks_exact(98).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let qs = &block[2..98];
                let scales_and_signs = &qs[64..96];
                let base = bi * 256;
                for ib32 in 0..8 {
                    let aux = u32::from_le_bytes(
                        scales_and_signs[4 * ib32..4 * ib32 + 4].try_into().unwrap(),
                    );
                    let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                    let qbase = ib32 * 8;
                    for l in 0..4 {
                        let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                        let grid1 = IQ3XXS_GRID[qs[qbase + 2 * l] as usize].to_le_bytes();
                        let grid2 = IQ3XXS_GRID[qs[qbase + 2 * l + 1] as usize].to_le_bytes();
                        for j in 0..4 {
                            let sign0 = if signs & KMASK_IQ2XS[j] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            let sign1 = if signs & KMASK_IQ2XS[j + 4] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            out[base + ib32 * 32 + l * 8 + j] = db * grid1[j] as f32 * sign0;
                            out[base + ib32 * 32 + l * 8 + j + 4] = db * grid2[j] as f32 * sign1;
                        }
                    }
                }
            }
        }
        GgmlType::Iq3S => {
            for (bi, block) in row.chunks_exact(110).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let qs = &block[2..66];
                let qh = &block[66..74];
                let signs = &block[74..106];
                let scales = &block[106..110];
                let base = bi * 256;
                for pair in 0..4 {
                    let ib32 = pair * 2;
                    let db1 = d * (1 + 2 * (scales[pair] & 0x0f) as i32) as f32;
                    let db2 = d * (1 + 2 * (scales[pair] >> 4) as i32) as f32;
                    let qbase = pair * 16;
                    let sbase = pair * 8;
                    let h0 = qh[pair * 2];
                    let h1 = qh[pair * 2 + 1];

                    for l in 0..4 {
                        let idx1 = qs[qbase + 2 * l] as usize
                            | ((((h0 as u16) << (8 - 2 * l)) & 256) as usize);
                        let idx2 = qs[qbase + 2 * l + 1] as usize
                            | ((((h0 as u16) << (7 - 2 * l)) & 256) as usize);
                        let grid1 = IQ3S_GRID[idx1].to_le_bytes();
                        let grid2 = IQ3S_GRID[idx2].to_le_bytes();
                        let sb = signs[sbase + l];
                        for j in 0..4 {
                            let sign0 = if sb & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                            let sign1 = if sb & KMASK_IQ2XS[j + 4] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            let out_base = base + ib32 * 32 + l * 8;
                            out[out_base + j] = db1 * grid1[j] as f32 * sign0;
                            out[out_base + j + 4] = db1 * grid2[j] as f32 * sign1;
                        }
                    }

                    for l in 0..4 {
                        let idx1 = qs[qbase + 8 + 2 * l] as usize
                            | ((((h1 as u16) << (8 - 2 * l)) & 256) as usize);
                        let idx2 = qs[qbase + 8 + 2 * l + 1] as usize
                            | ((((h1 as u16) << (7 - 2 * l)) & 256) as usize);
                        let grid1 = IQ3S_GRID[idx1].to_le_bytes();
                        let grid2 = IQ3S_GRID[idx2].to_le_bytes();
                        let sb = signs[sbase + 4 + l];
                        for j in 0..4 {
                            let sign0 = if sb & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                            let sign1 = if sb & KMASK_IQ2XS[j + 4] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            let out_base = base + (ib32 + 1) * 32 + l * 8;
                            out[out_base + j] = db2 * grid1[j] as f32 * sign0;
                            out[out_base + j + 4] = db2 * grid2[j] as f32 * sign1;
                        }
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
