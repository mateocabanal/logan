//! .coli package weight source for qwen4-rs (M2 real-model path).
//!
//! Mirrors the C runtime's dual-probe (HF-prefixed then canonical) and keeps
//! everything in its resident form:
//! - dense tensors: BF16 bytes (decode in matmul)
//! - experts: Apple8 MXFP4 tiles (rANS or raw) -> BF16 bytes, read on demand
//! - PLE ngram: per-shard record, rows read on demand (51 GB never resident)

use std::path::Path;
use std::{collections::HashMap, sync::Arc};

use logan_format::codecs::{
    apple8_mxfp4_decode, RansTable, INT4_MATH_FORMAT, INT4_SCALE_FORMAT, RANS_CODEC_ID,
};
use logan_format::package::{Package, RecordInfo};

const HF_PREFIX: &str = "model.language_model.";

#[derive(Clone)]
pub struct ColiSource {
    pkg: Package,
    // Immutable manifest indices only: never payloads or embedding rows.
    ple_shards: Arc<HashMap<i32, Vec<usize>>>,
}

fn index_ple_shards(records: &[RecordInfo]) -> Result<HashMap<i32, Vec<usize>>, String> {
    let mut grouped: HashMap<i32, Vec<(usize, usize)>> = HashMap::new();
    for (index, rec) in records.iter().enumerate() {
        if rec.kind != 1 {
            continue;
        }
        let Some(name) = rec.name.as_deref() else {
            continue;
        };
        let name = name.strip_prefix(HF_PREFIX).unwrap_or(name);
        let Some(rest) = name.strip_prefix("layers.") else {
            continue;
        };
        let Some((layer, shard)) = rest.split_once(".ple.ple_embedding.ngram_embedding.shard_")
        else {
            continue;
        };
        let layer: i32 = layer
            .parse()
            .map_err(|_| format!("invalid PLE layer in {name}"))?;
        let shard: usize = shard
            .parse()
            .map_err(|_| format!("invalid PLE shard in {name}"))?;
        grouped.entry(layer).or_default().push((shard, index));
    }
    grouped
        .into_iter()
        .map(|(layer, mut shards)| {
            shards.sort_unstable();
            if shards
                .iter()
                .enumerate()
                .any(|(expected, &(actual, _))| expected != actual)
            {
                return Err(format!(
                    "PLE layer {layer}: shard indices must be unique and contiguous from zero"
                ));
            }
            Ok((layer, shards.into_iter().map(|(_, index)| index).collect()))
        })
        .collect()
}

/// Raw GPU-ready expert matrix (Apple8 MXFP4 tiles + E8M0 scales).
#[derive(Clone)]
pub struct RawExpert {
    pub tiles: Vec<u8>,
    pub scales: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
    pub fmt: i32,
}

/// One cached expert whose three matrices live CONTIGUOUSLY in a MetalIO
/// slot (gate at 0, up at align16(gate_bytes), down at align16(up_end) — the
/// C engine's slot layout). The slot is the cache unit: the fused Apple8
/// moe_topk / swiglu kernels consume the slot bytes in native tile order, no
/// host copy, no per-matrix GPU handles. CPU fallback preads the slot bytes
/// (shared storage is CPU-visible).
pub struct SlotExpert {
    pub slot: i32,
    pub gate_bytes: usize,
    pub up_offset: usize,
    pub up_bytes: usize,
    pub down_offset: usize,
    pub down_bytes: usize,
    /// CPU-visible pointer to the slot's shared-storage MTLBuffer contents
    /// (valid while the slot is allocated).
    pub ptr: *mut u8,
    /// Pending MetalIO load event (0 = fully resident). Set on async issue;
    /// `Model::expert_wait` drains it before the expert is consumed.
    pub pending: std::cell::Cell<i64>,
    /// CPU fallback weights: decoded BF16 bytes (filled lazily on first CPU
    /// use; Metal users never pay for this).
    pub bf16_cache: std::cell::RefCell<Option<[Vec<u8>; 3]>>,
    pub rows: [usize; 3],
    pub cols: [usize; 3],
}

/// Borrowed view of a cached slot expert: offsets + raw pointer only, NO
/// Drop (must never free the slot — the cache-owned SlotExpert owns it).
pub struct SlotRef {
    pub slot: i32,
    pub gate_bytes: usize,
    pub up_offset: usize,
    pub up_bytes: usize,
    pub down_offset: usize,
    pub down_bytes: usize,
    pub ptr: *mut u8,
    pub rows: [usize; 3],
    pub cols: [usize; 3],
}

impl SlotExpert {
    /// Shared view without ownership of the slot.
    pub fn ref_view(&self) -> SlotRef {
        SlotRef {
            slot: self.slot,
            gate_bytes: self.gate_bytes,
            up_offset: self.up_offset,
            up_bytes: self.up_bytes,
            down_offset: self.down_offset,
            down_bytes: self.down_bytes,
            ptr: self.ptr,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl Drop for SlotExpert {
    fn drop(&mut self) {
        if self.slot >= 0 {
            unsafe { crate::ffi::metalio_slot_free(self.slot) };
            self.slot = -1;
        }
    }
}

/// The core's Slot contract: release = free the MetalIO slot (the LRU
/// store calls this on eviction/replace/drop).
impl logan_core::expert::Slot for SlotExpert {
    fn release(&mut self) {
        if self.slot >= 0 {
            unsafe { crate::ffi::metalio_slot_free(self.slot) };
            self.slot = -1;
        }
    }
}

/// Resident matrix in the physical representation carried by COLI.
pub struct ColiWt {
    pub bytes: Vec<u8>,
    pub scales: Vec<u8>,
    /// Logan Metal format: 5 = BF16 bytes, 7 = row-major MXFP4 + E8M0.
    pub fmt: i32,
    pub o: usize,
    pub i: usize,
}

impl ColiSource {
    pub fn open(dir: &Path) -> Result<ColiSource, String> {
        let pkg = Package::open(dir).map_err(|e| e.to_string())?;
        let ple_shards = Arc::new(index_ple_shards(pkg.records())?);
        Ok(ColiSource { pkg, ple_shards })
    }

    /// The underlying package for crate-internal MetalIO/plan region math.
    /// Keep this crate-private so external runtime code cannot bypass the
    /// bounded PLE n-gram row API and accidentally materialize streamed shards.
    pub(crate) fn pkg_ref(&self) -> &logan_format::package::Package {
        &self.pkg
    }

    /// Dual-probe record lookup: prefixed (resident HF) then bare canonical.
    pub(crate) fn rec(&self, name: &str) -> Option<&RecordInfo> {
        let pref = format!("{HF_PREFIX}{name}");
        self.pkg
            .record_by_name(&pref)
            .or_else(|| self.pkg.record_by_name(name))
    }

    fn reject_streamed_ple_full_read(name: &str) -> Result<(), String> {
        if name.contains("ple.ple_embedding.ngram_embedding.shard_")
            || name.ends_with("ple_embedding.ngram_embedding.weight")
        {
            return Err(format!(
                "{name}: PLE n-gram weights are streamed-only; use ple_ngram_row_f8 so the table remains on NVMe"
            ));
        }
        Ok(())
    }

    /// Dense vector tensor -> BF16 bytes.
    pub fn vec(&self, name: &str, want: usize) -> Result<Vec<u8>, String> {
        Self::reject_streamed_ple_full_read(name)?;
        let rec = self
            .rec(name)
            .ok_or_else(|| format!("missing dense tensor {name}"))?;
        let payload = self
            .pkg
            .read_tensor_payload(rec)
            .map_err(|e| e.to_string())?;
        if payload.len() != want * 2 {
            return Err(format!(
                "{name}: payload {} bytes != expected {}",
                payload.len(),
                want * 2
            ));
        }
        Ok(payload)
    }

    /// Dense matrix in the package's physical representation. Qwen3.6 MXFP4
    /// checkpoints keep their large matrices compressed all the way into the
    /// runtime; only the tiny affine-Q8 router/gate matrices are expanded once
    /// to BF16 at load so the established CPU routing path remains unchanged.
    pub fn wt(&self, name: &str, o: usize, i: usize) -> Result<ColiWt, String> {
        Self::reject_streamed_ple_full_read(name)?;
        let rec = self
            .rec(name)
            .ok_or_else(|| format!("missing dense matrix {name}"))?;
        let payload = self
            .pkg
            .read_tensor_payload(rec)
            .map_err(|e| e.to_string())?;
        match rec.math_format {
            0x0003 => {
                let want = o
                    .checked_mul(i)
                    .and_then(|n| n.checked_mul(2))
                    .ok_or_else(|| format!("{name}: BF16 matrix size overflows"))?;
                if payload.len() != want {
                    return Err(format!(
                        "{name}: BF16 payload {} bytes != expected {want} ({o}x{i})",
                        payload.len()
                    ));
                }
                Ok(ColiWt {
                    bytes: payload,
                    scales: Vec::new(),
                    fmt: 5,
                    o,
                    i,
                })
            }
            0x0020 => {
                let want_w = o
                    .checked_mul(i.div_ceil(2))
                    .ok_or_else(|| format!("{name}: MXFP4 matrix size overflows"))?;
                if payload.len() != want_w {
                    return Err(format!(
                        "{name}: MXFP4 payload {} bytes != expected {want_w} ({o}x{i})",
                        payload.len()
                    ));
                }
                let base = name
                    .strip_suffix(".weight")
                    .ok_or_else(|| format!("{name}: MXFP4 matrix is not a .weight record"))?;
                let scale_name = format!("{base}.scales");
                let scale_rec = self
                    .rec(&scale_name)
                    .ok_or_else(|| format!("missing MXFP4 scales {scale_name}"))?;
                let scales = self
                    .pkg
                    .read_tensor_payload(scale_rec)
                    .map_err(|e| e.to_string())?;
                let want_s = o
                    .checked_mul(i.div_ceil(32))
                    .ok_or_else(|| format!("{name}: MXFP4 scale size overflows"))?;
                if scales.len() != want_s {
                    return Err(format!(
                        "{scale_name}: payload {} bytes != expected {want_s} ({o}xceil({i}/32))",
                        scales.len()
                    ));
                }
                Ok(ColiWt {
                    bytes: payload,
                    scales,
                    fmt: 7,
                    o,
                    i,
                })
            }
            0x23..=0x26 => self.dequant_mlx_affine(name, rec, payload, rec.math_format, o, i),
            0x0005 => self.dequant_affine_q8(name, payload, o, i),
            other => Err(format!(
                "{name}: unsupported resident matrix math format 0x{other:04x}"
            )),
        }
    }

    fn dequant_mlx_affine(
        &self,
        name: &str,
        rec: &RecordInfo,
        payload: Vec<u8>,
        math: u16,
        o: usize,
        i: usize,
    ) -> Result<ColiWt, String> {
        let bits = mlx_affine_bits_from_math(math)
            .ok_or_else(|| format!("{name}: unsupported MLX affine math 0x{math:04x}"))?;
        let packed_bits = o
            .checked_mul(i)
            .and_then(|count| count.checked_mul(bits as usize))
            .ok_or_else(|| format!("{name}: MLX affine packed size overflows"))?;
        if packed_bits % 8 != 0 || payload.len() != packed_bits / 8 {
            return Err(format!(
                "{name}: MLX affine packed payload {} bytes != expected {} for {o}x{i} at {bits} bits",
                payload.len(),
                packed_bits.div_ceil(8)
            ));
        }
        let base = name
            .strip_suffix(".weight")
            .ok_or_else(|| format!("{name}: MLX affine matrix is not a .weight record"))?;
        let scale_name = format!("{base}.scales");
        let bias_name = format!("{base}.biases");
        let scales = self
            .pkg
            .read_tensor_payload(
                self.rec(&scale_name)
                    .ok_or_else(|| format!("missing MLX affine scales {scale_name}"))?,
            )
            .map_err(|e| e.to_string())?;
        let biases = self
            .pkg
            .read_tensor_payload(
                self.rec(&bias_name)
                    .ok_or_else(|| format!("missing MLX affine biases {bias_name}"))?,
            )
            .map_err(|e| e.to_string())?;
        let header = self
            .pkg
            .read_payload_range(rec, 0, 128)
            .map_err(|e| e.to_string())?;
        let group_size = u32::from_le_bytes(
            header[124..128]
                .try_into()
                .map_err(|_| format!("{name}: invalid COLITENS group-size field"))?,
        ) as usize;
        let decoded =
            decode_mlx_affine_dense_matrix(math, group_size, &payload, &scales, &biases, o, i)?;
        let mut out = Vec::with_capacity(decoded.len() * 2);
        for value in decoded {
            out.extend_from_slice(&f32_to_bf16(value).to_le_bytes());
        }
        Ok(ColiWt {
            bytes: out,
            scales: Vec::new(),
            fmt: 5,
            o,
            i,
        })
    }

    fn dequant_affine_q8(
        &self,
        name: &str,
        payload: Vec<u8>,
        o: usize,
        i: usize,
    ) -> Result<ColiWt, String> {
        let want = o
            .checked_mul(i)
            .ok_or_else(|| format!("{name}: Q8 matrix size overflows"))?;
        if payload.len() != want {
            return Err(format!(
                "{name}: Q8 payload {} bytes != expected {want} ({o}x{i})",
                payload.len()
            ));
        }
        let base = name
            .strip_suffix(".weight")
            .ok_or_else(|| format!("{name}: Q8 matrix is not a .weight record"))?;
        let scale_name = format!("{base}.scales");
        let bias_name = format!("{base}.biases");
        let scales = self
            .pkg
            .read_tensor_payload(
                self.rec(&scale_name)
                    .ok_or_else(|| format!("missing affine-Q8 scales {scale_name}"))?,
            )
            .map_err(|e| e.to_string())?;
        let biases = self
            .pkg
            .read_tensor_payload(
                self.rec(&bias_name)
                    .ok_or_else(|| format!("missing affine-Q8 biases {bias_name}"))?,
            )
            .map_err(|e| e.to_string())?;
        if scales.len() != biases.len() || scales.len() % (o * 2) != 0 {
            return Err(format!(
                "{name}: invalid affine-Q8 scale/bias payloads {}/{} bytes for {o} rows",
                scales.len(),
                biases.len()
            ));
        }
        let groups = scales.len() / (o * 2);
        if groups == 0 || i % groups != 0 {
            return Err(format!(
                "{name}: cannot infer affine-Q8 group size from {groups} groups across {i} columns"
            ));
        }
        let group_size = i / groups;
        let bf16_at = |bytes: &[u8], index: usize| -> f32 {
            let off = index * 2;
            bf16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]))
        };
        let mut out = Vec::with_capacity(want * 2);
        for row in 0..o {
            for col in 0..i {
                let group = row * groups + col / group_size;
                // MLX affine quantization reconstructs q*scale + bias. The
                // packed U32 source has already been exposed as raw U8 bytes
                // by the Logan compiler, so no further bit unpacking is needed.
                let value = payload[row * i + col] as f32 * bf16_at(&scales, group)
                    + bf16_at(&biases, group);
                out.extend_from_slice(&f32_to_bf16(value).to_le_bytes());
            }
        }
        Ok(ColiWt {
            bytes: out,
            scales: Vec::new(),
            fmt: 5,
            o,
            i,
        })
    }

    /// Routed expert (layer, expert) -> [gate, up, down] BF16 bytes,
    /// decoding Apple8 MXFP4 tiles or reading BF16 canonical payloads.
    /// Each call reads the record fresh (ponytail: no LRU yet; add when
    /// profiling shows disk-bound decode).
    pub fn expert_matrices(&self, layer: i32, expert: i32) -> Result<[ColiWt; 3], String> {
        let recs = self.pkg.expert_records(layer, expert);
        let rec = recs
            .first()
            .ok_or_else(|| format!("missing expert ({layer},{expert})"))?;
        let raw = self.pkg.read_record(rec).map_err(|e| e.to_string())?;
        assert_eq!(&raw[..8], b"COLIEXPT");
        let desc_size = u32::from_le_bytes(raw[28..32].try_into().unwrap()) as usize;
        let mut out: Vec<ColiWt> = Vec::with_capacity(3);
        for i in 0..3 {
            let d = 64 + i * desc_size;
            let role = u16::from_le_bytes(raw[d..d + 2].try_into().unwrap());
            let math = u16::from_le_bytes(raw[d + 4..d + 6].try_into().unwrap());
            let scale = u16::from_le_bytes(raw[d + 6..d + 8].try_into().unwrap());
            let wc = u16::from_le_bytes(raw[d + 8..d + 10].try_into().unwrap());
            let wt = u32::from_le_bytes(raw[d + 40..d + 44].try_into().unwrap());
            let rows = u64::from_le_bytes(raw[d + 16..d + 24].try_into().unwrap());
            let cols = u64::from_le_bytes(raw[d + 24..d + 32].try_into().unwrap());
            let w_off = u64::from_le_bytes(raw[d + 48..d + 56].try_into().unwrap());
            let w_stored = u64::from_le_bytes(raw[d + 56..d + 64].try_into().unwrap());
            let w_decoded = u64::from_le_bytes(raw[d + 64..d + 72].try_into().unwrap());
            let s_off = u64::from_le_bytes(raw[d + 72..d + 80].try_into().unwrap());
            let s_stored = u64::from_le_bytes(raw[d + 80..d + 88].try_into().unwrap());
            let w = &raw[w_off as usize..(w_off + w_stored) as usize];

            // All expert representations decode to BF16 bytes.
            let bytes: Vec<u8> = match (math, scale) {
                // BF16 canonical: raw bytes already
                (0x0003, 0x0000) => w.to_vec(),
                // INT4-G32: dequant to BF16
                (INT4_MATH_FORMAT, INT4_SCALE_FORMAT) => {
                    let s = &raw[s_off as usize..(s_off + s_stored) as usize];
                    let f =
                        logan_format::codecs::int4_grouped_decode(w, s, rows as usize, cols as usize)
                            .map_err(|e| e.to_string())?;
                    f.into_iter().flat_map(bf16_bytes).collect()
                }
                // MLX affine: packed U32-derived bytes plus one aux blob
                // containing BF16 scales followed immediately by BF16 biases.
                (0x23..=0x26, 0x0003) => {
                    let aux = &raw[s_off as usize..(s_off + s_stored) as usize];
                    let group_size =
                        u32::from_le_bytes(raw[d + 104..d + 108].try_into().unwrap()) as usize;
                    let f = decode_mlx_affine_expert_matrix(
                        math, group_size, w, aux, rows as usize, cols as usize,
                    )?;
                    f.into_iter().flat_map(bf16_bytes).collect()
                }
                // Apple8 MXFP4: rANS or raw tiles -> f32 -> BF16
                (0x0020, 0x0004) => {
                    let tiles: Vec<u8> = if wc == RANS_CODEC_ID {
                        let table = RansTable::from_manifest(&self.pkg.manifest_ref(), wt, wc)
                            .map_err(|e| e.to_string())?;
                        logan_format::codecs::apple8_decode(w, &table, rows, cols)
                            .map_err(|e| e.to_string())?
                    } else {
                        if w.len() as u64 != w_decoded {
                            return Err(format!("expert {layer}/{expert} m{i} raw tile size"));
                        }
                        w.to_vec()
                    };
                    let f = apple8_mxfp4_decode(&tiles, rows, cols).map_err(|e| e.to_string())?;
                    f.into_iter().flat_map(bf16_bytes).collect()
                }
                _ => {
                    return Err(format!(
                        "expert {layer}/{expert} m{i} unsupported math=0x{math:04x} scale=0x{scale:04x}"
                    ))
                }
            };
            out.push(ColiWt {
                bytes,
                scales: Vec::new(),
                fmt: 5,
                o: rows as usize,
                i: cols as usize,
            });
        }
        Ok([out.remove(0), out.remove(0), out.remove(0)])
    }

    /// The layer that actually carries PLE tensors in this package (the
    /// frontend's resolved ple_layer, which can differ from config's
    /// ple_layer_ids). Ground truth = first `layers.N.ple.` record.
    pub fn ple_layer(&self) -> Option<i32> {
        self.pkg.records().iter().find_map(|r| {
            let nm = r.name.as_deref()?;
            let rest = nm.strip_prefix("layers.")?;
            let dot = rest.find('.')?;
            let layer: i32 = rest[..dot].parse().ok()?;
            if nm.contains(".ple.") {
                Some(layer)
            } else {
                None
            }
        })
    }

    /// Raw resident expert tiles for (layer, expert): [gate, up, down],
    /// each as {tiles, scales, rows, cols, fmt}. Apple8 = fmt 7 (MXFP4:
    /// nibble bytes + raw E8M0 scale bytes) — the GPU consumes these
    /// directly, NO host decode. BF16 canonical = fmt 16 (CPU-only).
    pub fn expert_tiles(&self, layer: i32, expert: i32) -> Result<[RawExpert; 3], String> {
        let recs = self.pkg.expert_records(layer, expert);
        let rec = recs
            .first()
            .ok_or_else(|| format!("missing expert ({layer},{expert})"))?;
        let raw = self.pkg.read_record(rec).map_err(|e| e.to_string())?;
        assert_eq!(&raw[..8], b"COLIEXPT");
        let desc_size = u32::from_le_bytes(raw[28..32].try_into().unwrap()) as usize;
        let mut out: Vec<RawExpert> = Vec::with_capacity(3);
        for i in 0..3 {
            let d = 64 + i * desc_size;
            let math = u16::from_le_bytes(raw[d + 4..d + 6].try_into().unwrap());
            let scale = u16::from_le_bytes(raw[d + 6..d + 8].try_into().unwrap());
            let wc = u16::from_le_bytes(raw[d + 8..d + 10].try_into().unwrap());
            let rows = u64::from_le_bytes(raw[d + 16..d + 24].try_into().unwrap());
            let cols = u64::from_le_bytes(raw[d + 24..d + 32].try_into().unwrap());
            let w_off = u64::from_le_bytes(raw[d + 48..d + 56].try_into().unwrap());
            let w_stored = u64::from_le_bytes(raw[d + 56..d + 64].try_into().unwrap());
            let w_decoded = u64::from_le_bytes(raw[d + 64..d + 72].try_into().unwrap());
            let s_off = u64::from_le_bytes(raw[d + 72..d + 80].try_into().unwrap());
            let s_stored = u64::from_le_bytes(raw[d + 80..d + 88].try_into().unwrap());
            let w = &raw[w_off as usize..(w_off + w_stored) as usize];
            let s = &raw[s_off as usize..(s_off + s_stored) as usize];
            match (math, scale) {
                // Apple8 MXFP4: raw tiles are GPU-ready (fmt 7). rANS-compressed
                // tiles must be host-decoded first (wc == RANS_CODEC_ID).
                (0x0020, 0x0004) if wc != RANS_CODEC_ID => {
                    out.push(RawExpert {
                        tiles: w.to_vec(),
                        scales: s.to_vec(),
                        rows: rows as usize,
                        cols: cols as usize,
                        fmt: 7,
                    });
                }
                (0x0020, 0x0004) => {
                    let table = RansTable::from_manifest(&self.pkg.manifest_ref(), wc as u32, wc)
                        .map_err(|e| e.to_string())?;
                    let tiles = logan_format::codecs::apple8_decode(w, &table, rows, cols)
                        .map_err(|e| e.to_string())?;
                    out.push(RawExpert {
                        tiles,
                        scales: s.to_vec(),
                        rows: rows as usize,
                        cols: cols as usize,
                        fmt: 7,
                    });
                }
                _ => {
                    // BF16 canonical / INT4: no raw GPU form — the caller
                    // falls back to the decode path.
                    return Err(format!(
                        "expert {layer}/{expert} m{i} math=0x{math:04x} scale=0x{scale:04x} not raw-GPU"
                    ));
                }
            }
        }
        Ok([out.remove(0), out.remove(0), out.remove(0)])
    }

    /// PLE metadata from the package (ground truth, i64 records): per-head
    /// vocab sizes, cumulative offsets, and the layer multipliers. The
    /// config-derived prime math diverges on the real model (row 173M vs
    /// computed 160M capacity), so .coli mode reads these instead.
    pub fn ple_metadata(&self, layer: i32) -> Result<(Vec<i64>, Vec<i64>, Vec<u64>), String> {
        let sizes = self.i64_tensor(&format!(
            "layers.{layer}.ple.ple_embedding.ngram_heads_vocab_sizes"
        ))?;
        let offsets = self.i64_tensor(&format!(
            "layers.{layer}.ple.ple_embedding.ngram_heads_offsets"
        ))?;
        let mult = self.i64_tensor(&format!(
            "layers.{layer}.ple.ple_embedding.layer_multipliers"
        ))?;
        Ok((sizes, offsets, mult.into_iter().map(|m| m as u64).collect()))
    }

    fn i64_tensor(&self, name: &str) -> Result<Vec<i64>, String> {
        let rec = self
            .rec(name)
            .ok_or_else(|| format!("missing PLE metadata {name}"))?;
        let payload = self
            .pkg
            .read_tensor_payload(rec)
            .map_err(|e| e.to_string())?;
        // i64 records (8 bytes/elem: 16 heads x 8 = 128 bytes for
        // vocab_sizes/offsets; 3 x 8 = 24 for multipliers).
        if payload.len() % 8 != 0 {
            return Err(format!("{name}: payload {} not 8-aligned", payload.len()));
        }
        Ok(payload
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    /// PLE n-gram row fetch (F8 E4M3 shards): row `r` of the ngram table.
    /// The real package stores it as F8 shards (1 byte/elem, math=0x10) with
    /// a global BF16 scale; the C engine preads hd_per bytes per row, never
    /// loading the 400 MB shard. The tiny fixture's BF16 single-tensor form
    /// is handled by the safetensors loader (not this path).
    pub fn ple_ngram_row_f8(&self, layer: i32, r: u64, hd_per: usize) -> Result<Vec<u8>, String> {
        if hd_per == 0 {
            return Err("ngram row width must be nonzero".into());
        }
        let recs = self
            .ple_shards
            .get(&layer)
            .ok_or_else(|| format!("no ngram shards for layer {layer}"))?;
        let first = &self.pkg.records()[recs[0]];
        if first.decoded == 0 || first.decoded % hd_per as u64 != 0 {
            return Err(format!(
                "invalid ngram shard row geometry for layer {layer}"
            ));
        }
        let rps = first.decoded / hd_per as u64;
        let shard_idx = usize::try_from(r / rps).map_err(|_| "ngram shard index overflow")?;
        let &record_index = recs
            .get(shard_idx)
            .ok_or_else(|| format!("ngram row {r} out of range ({} shards)", recs.len()))?;
        let rec = &self.pkg.records()[record_index];
        let within = (r % rps) as u64;
        // pread only this row (F8: hd_per bytes) — never the whole shard
        self.pkg
            .read_tensor_payload_range(rec, within * hd_per as u64, hd_per)
            .map_err(|e| e.to_string())
    }

    /// Global BF16 scale for the F8 ngram table.
    pub fn ple_ngram_scale(&self, layer: i32) -> Result<f32, String> {
        let rec = self
            .rec(&format!(
                "layers.{layer}.ple.ple_embedding.ngram_embedding.weight_scale"
            ))
            .ok_or_else(|| format!("missing ngram weight_scale"))?;
        let payload = self
            .pkg
            .read_tensor_payload(rec)
            .map_err(|e| e.to_string())?;
        if payload.len() != 2 {
            return Err(format!("weight_scale payload {} != 2", payload.len()));
        }
        let u = u16::from_le_bytes(payload.try_into().unwrap());
        Ok(f32::from_bits((u as u32) << 16))
    }

    /// F8 E4M3 decode (bit-exact with the C engine's E4M3_LUT).
    pub fn e4m3_decode(b: u8) -> f32 {
        let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
        let exp = ((b >> 3) & 0x0f) as i32;
        let mant = (b & 0x07) as f32;
        match exp {
            0 => sign * mant * 0.001953125, // subnormal: 2^-9
            // OCP E4M3FN has no infinity encoding. Exponent 15 remains
            // finite through mantissa 6 (256..448); only mantissa 7 is NaN.
            0x0f if mant == 7.0 => f32::NAN,
            e => sign * (1.0 + mant * 0.125) * 2f32.powi(e - 7),
        }
    }
}

/// f32 -> BF16 (top 16 bits, round-to-nearest-even), as 2 LE bytes.
pub fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

pub fn bf16_bytes(f: f32) -> [u8; 2] {
    f32_to_bf16(f).to_le_bytes()
}

pub fn bf16_to_f32(u: u16) -> f32 {
    f32::from_bits((u as u32) << 16)
}

/// Decode MLX affine-packed row-major weights exactly as `mx.dequantize`: the
/// first code starts in the least-significant bits of the first U32 and
/// non-power-of-two widths continue as one contiguous LSB-first bitstream.
/// Each output is `code * scale + bias` for its quantization group.
fn mlx_affine_bits_from_math(math: u16) -> Option<u8> {
    match math {
        0x23 => Some(4),
        0x24 => Some(5),
        0x25 => Some(6),
        0x26 => Some(8),
        _ => None,
    }
}

fn decode_mlx_affine_expert_matrix(
    math: u16,
    group_size: usize,
    packed: &[u8],
    aux: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, String> {
    let bits = mlx_affine_bits_from_math(math)
        .ok_or_else(|| format!("unsupported MLX affine math format 0x{math:04x}"))?;
    if group_size == 0 || columns % group_size != 0 {
        return Err(format!(
            "invalid MLX affine expert group_size={group_size} for {columns} columns"
        ));
    }
    let param_bytes = rows
        .checked_mul(columns / group_size)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| "MLX affine expert parameter size overflows".to_owned())?;
    let want_aux = param_bytes
        .checked_mul(2)
        .ok_or_else(|| "MLX affine expert aux size overflows".to_owned())?;
    if aux.len() != want_aux {
        return Err(format!(
            "MLX affine expert aux payload {} bytes != expected {want_aux} ({param_bytes} scales + {param_bytes} biases)",
            aux.len()
        ));
    }
    let (scales, biases) = aux.split_at(param_bytes);
    mlx_affine_decode(packed, scales, biases, rows, columns, bits, group_size)
}

fn decode_mlx_affine_dense_matrix(
    math: u16,
    group_size: usize,
    packed: &[u8],
    scales: &[u8],
    biases: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, String> {
    let bits = mlx_affine_bits_from_math(math)
        .ok_or_else(|| format!("unsupported MLX affine math format 0x{math:04x}"))?;
    mlx_affine_decode(packed, scales, biases, rows, columns, bits, group_size)
}

pub(crate) fn mlx_affine_decode(
    packed: &[u8],
    scales: &[u8],
    biases: &[u8],
    rows: usize,
    columns: usize,
    bits: u8,
    group_size: usize,
) -> Result<Vec<f32>, String> {
    if !matches!(bits, 4 | 5 | 6 | 8) {
        return Err(format!("unsupported MLX affine bit width {bits}"));
    }
    if rows == 0 || columns == 0 || group_size == 0 || columns % group_size != 0 {
        return Err(format!(
            "invalid MLX affine geometry rows={rows} columns={columns} group_size={group_size}"
        ));
    }
    let row_bits = columns
        .checked_mul(bits as usize)
        .ok_or_else(|| "MLX affine row bit count overflows".to_owned())?;
    if row_bits % 32 != 0 {
        return Err(format!(
            "MLX affine row uses {row_bits} bits, not an integral number of U32 words"
        ));
    }
    let row_bytes = row_bits / 8;
    let want_weights = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| "MLX affine packed byte count overflows".to_owned())?;
    if packed.len() != want_weights {
        return Err(format!(
            "MLX affine packed payload {} bytes != expected {want_weights}",
            packed.len()
        ));
    }
    let groups_per_row = columns / group_size;
    let params = rows
        .checked_mul(groups_per_row)
        .ok_or_else(|| "MLX affine parameter count overflows".to_owned())?;
    let want_params = params
        .checked_mul(2)
        .ok_or_else(|| "MLX affine parameter byte count overflows".to_owned())?;
    if scales.len() != want_params || biases.len() != want_params {
        return Err(format!(
            "MLX affine scale/bias payloads {}/{} bytes != expected {want_params}",
            scales.len(),
            biases.len()
        ));
    }

    let bf16_at = |bytes: &[u8], index: usize| {
        let offset = index * 2;
        bf16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
    };
    let mask = (1_u64 << bits) - 1;
    let mut output = Vec::with_capacity(rows * columns);
    for row in 0..rows {
        let row_data = &packed[row * row_bytes..(row + 1) * row_bytes];
        for column in 0..columns {
            let bit = column * bits as usize;
            let word_index = bit / 32;
            let shift = bit % 32;
            let byte = word_index * 4;
            let lo = u32::from_le_bytes(row_data[byte..byte + 4].try_into().unwrap()) as u64;
            let combined = if shift + bits as usize > 32 {
                let next =
                    u32::from_le_bytes(row_data[byte + 4..byte + 8].try_into().unwrap()) as u64;
                lo | (next << 32)
            } else {
                lo
            };
            let code = ((combined >> shift) & mask) as f32;
            let group = row * groups_per_row + column / group_size;
            output.push(code * bf16_at(scales, group) + bf16_at(biases, group));
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::ColiSource;

    fn shard(name: &str) -> logan_format::package::RecordInfo {
        logan_format::package::RecordInfo {
            id: 0,
            kind: 1,
            codec: 0,
            math_format: 0x10,
            scale_format: 0,
            layout: 0,
            flags: 0,
            shard_id: 0,
            name: Some(name.into()),
            layer: 1,
            expert: -1,
            offset: 0,
            stored: 16,
            decoded: 16,
            stored_crc: 0,
            logical_crc: 0,
        }
    }

    #[test]
    fn ple_index_uses_numeric_shards_and_shares_no_payloads() {
        let records: Vec<_> = (0..12)
            .rev()
            .map(|i| {
                shard(&format!(
                    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}"
                ))
            })
            .collect();
        let index = super::index_ple_shards(&records).unwrap();
        assert_eq!(index[&1], (0..12).rev().collect::<Vec<_>>());
    }

    #[test]
    fn ple_index_rejects_gaps_and_duplicate_ordinals() {
        let a = shard("layers.1.ple.ple_embedding.ngram_embedding.shard_0");
        let b = shard("layers.1.ple.ple_embedding.ngram_embedding.shard_2");
        assert!(super::index_ple_shards(&[a.clone(), b]).is_err());
        assert!(super::index_ple_shards(&[a.clone(), a]).is_err());
    }

    #[test]
    #[ignore = "requires LOGAN_PLE_TEST_PACKAGE pointing to a local COLI model"]
    fn real_ple_shard_boundaries_match_direct_range_reads() {
        let path = std::env::var("LOGAN_PLE_TEST_PACKAGE").expect("set LOGAN_PLE_TEST_PACKAGE");
        let path = std::path::Path::new(&path);
        let cfg = crate::load_cfg(&path.join("config.json")).unwrap();
        assert!(cfg.ngram_heads > 0);
        let width = cfg.ple_embed_dim / cfg.ngram_heads;
        let src = ColiSource::open(path).unwrap();
        let layer = src.ple_layer().unwrap();
        let shards = &src.ple_shards[&layer];
        let rows_per_shard = src.pkg.records()[shards[0]].decoded / width as u64;
        let mut checks = 0;
        for (ordinal, &index) in shards.iter().enumerate() {
            let rec = &src.pkg.records()[index];
            assert_eq!(rec.decoded % width as u64, 0);
            let rows = rec.decoded / width as u64;
            assert!(rows > 0 && rows <= rows_per_shard);
            if ordinal + 1 < shards.len() {
                assert_eq!(rows, rows_per_shard);
            }
            for within in [0, rows - 1] {
                let expected = src
                    .pkg
                    .read_tensor_payload_range(rec, within * width as u64, width)
                    .unwrap();
                let got = src
                    .ple_ngram_row_f8(layer, ordinal as u64 * rows_per_shard + within, width)
                    .unwrap();
                assert_eq!(got, expected, "shard={ordinal} row={within}");
                checks += 1;
            }
        }
        assert!(src.ple_ngram_row_f8(layer, 0, 0).is_err());
        assert!(src
            .ple_ngram_row_f8(layer, shards.len() as u64 * rows_per_shard, width)
            .is_err());
        println!("PLE gate: {checks} boundary rows, width={width}, metadata_indices={}, no whole-shard reads", shards.len());
    }

    #[test]
    fn streamed_ple_ngram_cannot_use_resident_read_paths() {
        for name in [
            "layers.1.ple.ple_embedding.ngram_embedding.shard_0",
            "model.ple.ple_embedding.ngram_embedding.weight",
        ] {
            let err = ColiSource::reject_streamed_ple_full_read(name).unwrap_err();
            assert!(err.contains("streamed-only"));
        }
        assert!(ColiSource::reject_streamed_ple_full_read(
            "layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
        )
        .is_ok());
    }

    #[test]
    fn e4m3fn_exponent_15_is_finite_except_nan_code() {
        assert_eq!(ColiSource::e4m3_decode(0x78), 256.0);
        assert_eq!(ColiSource::e4m3_decode(0x79), 288.0);
        assert_eq!(ColiSource::e4m3_decode(0x7e), 448.0);
        assert_eq!(ColiSource::e4m3_decode(0xf8), -256.0);
        assert_eq!(ColiSource::e4m3_decode(0xfe), -448.0);
        assert!(ColiSource::e4m3_decode(0x7f).is_nan());
        assert!(ColiSource::e4m3_decode(0xff).is_nan());
    }

    #[test]
    fn e4m3fn_zero_subnormal_and_normal_boundaries() {
        assert_eq!(ColiSource::e4m3_decode(0x00).to_bits(), 0.0_f32.to_bits());
        assert_eq!(
            ColiSource::e4m3_decode(0x80).to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(ColiSource::e4m3_decode(0x01), 2.0_f32.powi(-9));
        assert_eq!(ColiSource::e4m3_decode(0x08), 2.0_f32.powi(-6));
    }
}

#[cfg(test)]
mod mlx_affine_decode_tests {
    use super::*;

    fn bf16_vec(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|&value| bf16_bytes(value)).collect()
    }

    fn words_le(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    #[test]
    fn mlx_affine_decode_matches_lsb_first_mlx_packing() {
        let cases: &[(u8, &[u32])] = &[
            (4, &[0x7654_3210, 0xfedc_ba98, 0x7654_3210, 0xfedc_ba98]),
            (
                5,
                &[
                    0x8a41_8820,
                    0xc5a9_2839,
                    0xca30_7b9a,
                    0x38bd_ab49,
                    0xffbb_cdeb,
                ],
            ),
            (
                6,
                &[
                    0x440c_2040,
                    0xa248_1c61,
                    0x3ce3_4c2c,
                    0x544d_2450,
                    0xa658_5d65,
                    0x7de7_5c6d,
                ],
            ),
            (
                8,
                &[
                    0x0302_0100,
                    0x0706_0504,
                    0x0b0a_0908,
                    0x0f0e_0d0c,
                    0x1312_1110,
                    0x1716_1514,
                    0x1b1a_1918,
                    0x1f1e_1d1c,
                ],
            ),
        ];

        for &(bits, words) in cases {
            // Two 32-value groups. The second repeats the same packed codes so
            // group-specific scale/bias application is visible independently
            // from bit unpacking.
            let mut packed = words_le(words);
            packed.extend_from_slice(&words_le(words));
            let scales = bf16_vec(&[1.0, 2.0]);
            let biases = bf16_vec(&[0.5, -1.0]);
            let decoded = mlx_affine_decode(&packed, &scales, &biases, 1, 64, bits, 32).unwrap();

            let mask = (1_u32 << bits) - 1;
            let expected: Vec<f32> = (0..64)
                .map(|column| {
                    let code = (column % 32) as u32 & mask;
                    if column < 32 {
                        code as f32 + 0.5
                    } else {
                        code as f32 * 2.0 - 1.0
                    }
                })
                .collect();
            assert_eq!(decoded, expected, "bits={bits}");
        }
    }

    #[test]
    fn mlx_affine_expert_aux_is_scale_then_bias() {
        let packed = words_le(&[
            0x7654_3210,
            0xfedc_ba98,
            0x7654_3210,
            0xfedc_ba98,
            0x7654_3210,
            0xfedc_ba98,
            0x7654_3210,
            0xfedc_ba98,
        ]);
        let mut aux = bf16_vec(&[2.0]);
        aux.extend_from_slice(&bf16_vec(&[-1.0]));
        let decoded = decode_mlx_affine_expert_matrix(0x23, 64, &packed, &aux, 1, 64).unwrap();
        let expected: Vec<f32> = (0..64)
            .map(|column| ((column % 16) as f32) * 2.0 - 1.0)
            .collect();
        assert_eq!(decoded, expected);
    }
}

#[cfg(test)]
mod mlx_affine_dense_tests {
    use super::*;

    fn bf16_vec(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|&value| bf16_bytes(value)).collect()
    }

    fn words_le(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    #[test]
    fn dense_affine_adapter_maps_math_and_group_metadata() {
        let packed = words_le(&[
            0x8a41_8820,
            0xc5a9_2839,
            0xca30_7b9a,
            0x38bd_ab49,
            0xffbb_cdeb,
            0x8a41_8820,
            0xc5a9_2839,
            0xca30_7b9a,
            0x38bd_ab49,
            0xffbb_cdeb,
        ]);
        let scales = bf16_vec(&[1.0, 2.0]);
        let biases = bf16_vec(&[0.5, -1.0]);
        let decoded =
            decode_mlx_affine_dense_matrix(0x24, 32, &packed, &scales, &biases, 1, 64).unwrap();
        let expected: Vec<f32> = (0..64)
            .map(|column| {
                let code = (column % 32) as u32 & 31;
                if column < 32 {
                    code as f32 + 0.5
                } else {
                    code as f32 * 2.0 - 1.0
                }
            })
            .collect();
        assert_eq!(decoded, expected);
    }
}
