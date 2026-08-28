//! Engine-neutral tensor storage.
//!
//! The weight source abstraction: engines declare tensors by stable name +
//! shape; the storage layer resolves them to resident RAM, streamed disk
//! ranges, or GPU-visible buffers. Mirrors the C engine's split between
//! resident dense weights, streamed experts, and MetalIO slots — but
//! engine-neutral: no qwen/qwen4/v4 names here.

use std::path::Path;

/// A resolved tensor: where its bytes live and how to read them.
#[derive(Debug, Clone)]
pub enum TensorStorage {
    /// Fully resident in RAM (dense weights, small tensors).
    Resident(Vec<u8>),
    /// Streamed from a package shard: (shard path, offset, len).
    /// The core's expert store / prefetch layer decides when to pull it.
    Streamed {
        shard: String,
        offset: u64,
        len: usize,
    },
    /// GPU-visible zero-copy (page-aligned, wrapped by Metal). The bytes
    /// are CPU-visible too (shared storage) — the CPU fallback reads them.
    Gpu {
        /// Opaque handle (MetalIO slot id or Metal buffer handle).
        handle: u64,
        /// CPU-visible pointer (valid while the handle is alive).
        ptr: *mut u8,
        len: usize,
    },
}

// SAFETY: Gpu storage is only ever accessed on the decode thread (the
// single-threaded per-token pipeline); the pointer is owned by the slot
// pool which outlives the storage handle.
unsafe impl Send for TensorStorage {}

/// A typed tensor handle: stable name + shape + dtype + storage.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub storage: TensorStorage,
}

impl Tensor {
    /// Byte length implied by shape + dtype ("f32"=4, "bf16"=2, "f8-e4m3"=1,
    /// "mxfp4-tile8x32"=136 bytes per 8x32 tile, "i4-g32"=5/8 per element).
    pub fn byte_len(&self) -> usize {
        match self.dtype.as_str() {
            "f32" => self.elems() * 4,
            "bf16" => self.elems() * 2,
            "f8-e4m3" => self.elems(),
            "i4-g32" => self.elems() * 5 / 8, // 4 bits + f32 scale per 32
            "mxfp4-tile8x32" => {
                // 8 rows x 32 cols per 136-byte tile (128 weight + 8 scale)
                let rows = self.shape.first().copied().unwrap_or(0);
                let cols = self.shape.get(1).copied().unwrap_or(0);
                let row_tiles = rows.div_ceil(8);
                let col_tiles = cols.div_ceil(32);
                (row_tiles * col_tiles) as usize * 136
            }
            other => panic!("unknown dtype {other}"),
        }
    }

    fn elems(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }
}

/// The weight source trait: an engine-neutral view of a model package.
///
/// Engines implement this over whatever package format they consume
/// (currently logan-format's CSF package; a raw safetensors dir is another
/// possible implementation). The core's expert store, prefetch, and
/// telemetry layers depend only on this trait — so a change to residency
/// policy or streaming benefits every engine.
pub trait WeightSource {
    /// Resolve a tensor by stable name. Returns None when the package has
    /// no such tensor (engines probe optional tensors this way).
    fn tensor(&self, name: &str) -> Option<Tensor>;

    /// The (layer, expert) triple for a routed expert, as streamed ranges.
    /// Returns None when the package stores experts in a non-streamable
    /// form (e.g. a raw safetensors dir with no shard layout).
    fn expert_ranges(&self, layer: u32, expert: u32) -> Option<[TensorStorage; 3]>;

    /// Package fingerprint (for plan-artifact staleness checks).
    fn fingerprint(&self) -> String;
}

/// Convenience: read a resident tensor's bytes as f32 (for small vectors
/// like norms and biases that engines decode at load time).
pub fn resident_f32(t: &Tensor) -> Option<Vec<f32>> {
    match &t.storage {
        TensorStorage::Resident(bytes) => {
            if t.dtype == "f32" {
                Some(
                    bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect(),
                )
            } else if t.dtype == "bf16" {
                Some(
                    bytes
                        .chunks_exact(2)
                        .map(|c| {
                            let u = u16::from_le_bytes(c.try_into().unwrap());
                            f32::from_bits((u as u32) << 16)
                        })
                        .collect(),
                )
            } else {
                None
            }
        }
        _ => None,
    }
}

/// A package directory path helper (engines that load from a dir).
pub fn package_dir(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_len_math() {
        let t = Tensor {
            name: "w".into(),
            shape: vec![640, 2560],
            dtype: "bf16".into(),
            storage: TensorStorage::Resident(vec![]),
        };
        assert_eq!(t.byte_len(), 640 * 2560 * 2);
        let t2 = Tensor {
            name: "e".into(),
            shape: vec![640, 2560],
            dtype: "mxfp4-tile8x32".into(),
            storage: TensorStorage::Resident(vec![]),
        };
        // 640x2560 = 80x80 tiles of 8x32 = 6400 tiles * 136 bytes
        assert_eq!(t2.byte_len(), 6400 * 136);
    }

    #[test]
    fn resident_f32_decode() {
        let bytes: Vec<u8> = [1.0f32, 2.5, -3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let t = Tensor {
            name: "v".into(),
            shape: vec![3],
            dtype: "f32".into(),
            storage: TensorStorage::Resident(bytes),
        };
        let v = resident_f32(&t).unwrap();
        assert_eq!(v, vec![1.0, 2.5, -3.0]);
    }

    #[test]
    fn resident_bf16_decode() {
        let bytes: Vec<u8> = [1.0f32, 2.0]
            .iter()
            .flat_map(|&v| crate::math::bf16_bytes(v))
            .collect();
        let t = Tensor {
            name: "v".into(),
            shape: vec![2],
            dtype: "bf16".into(),
            storage: TensorStorage::Resident(bytes),
        };
        let v = resident_f32(&t).unwrap();
        assert!((v[0] - 1.0).abs() < 0.01);
        assert!((v[1] - 2.0).abs() < 0.01);
    }
}
