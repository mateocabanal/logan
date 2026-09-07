//! Tiny Qwen4 (Qwen3.8-Flash-Next / Qwen4Exp) scalar reference in Rust.
//!
//! Extends the qwen-rs port with the qwen4 additions: hyper connections
//! (GatedResidual mixer), QSA indexer sparse attention, and the PLE n-gram
//! layer. GDN/full-attention/MoE math is unchanged from qwen-rs (C-identical
//! numerics, f32 accumulators, exact reduction order).
//!
//! Gate: `ref.json` greedy_new_ids (token identity).

use std::path::Path;

pub mod coliload;
pub mod colisource;
pub mod ggufsource;
mod ggufload;
pub use ggufload::load_cfg_gguf;
pub mod ffi;
mod gdn_ane;
pub mod plan;
pub mod scheduled;

use logan_core::expert::Slot as _; // for SlotExpert::release

// ---------------------------------------------------------------------------
// safetensors reader (same minimal F32 adapter as qwen-rs)
// ---------------------------------------------------------------------------

const MAX_RESIDENT_PLE_NGRAM_BYTES: usize = 64 * 1024 * 1024;

fn is_ple_ngram_weight(name: &str) -> bool {
    name.ends_with("ple_embedding.ngram_embedding.weight")
        || name.ends_with("ple.ngram_embedding.weight")
}

pub struct StFile {
    // Keep safetensors file-backed. The old reference loader used
    // std::fs::read(), which made the entire source file resident before any
    // tensor was requested. Production Flash-Next uses .coli, but keeping this
    // adapter range-read prevents accidental whole-file residency as well.
    file: std::sync::Mutex<std::fs::File>,
    tensors: std::collections::HashMap<String, (Vec<u64>, usize, usize)>,
}

impl StFile {
    pub fn open(path: &Path) -> Result<StFile, String> {
        use std::io::Read as _;

        let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let file_len = file.metadata().map_err(|e| e.to_string())?.len();
        let mut nbuf = [0_u8; 8];
        file.read_exact(&mut nbuf).map_err(|e| e.to_string())?;
        let n = u64::from_le_bytes(nbuf);
        let data_start_u64 = 8_u64
            .checked_add(n)
            .ok_or_else(|| "safetensors header length overflow".to_string())?;
        if data_start_u64 > file_len {
            return Err(format!(
                "safetensors header extends past EOF: {data_start_u64} > {file_len}"
            ));
        }
        let n_usize = usize::try_from(n)
            .map_err(|_| "safetensors header too large for this host".to_string())?;
        let mut header_bytes = vec![0_u8; n_usize];
        file.read_exact(&mut header_bytes)
            .map_err(|e| e.to_string())?;
        let header: serde_json::Value =
            serde_json::from_slice(&header_bytes).map_err(|e| e.to_string())?;
        let obj = header
            .as_object()
            .ok_or_else(|| "safetensors header is not an object".to_string())?;
        usize::try_from(data_start_u64)
            .map_err(|_| "safetensors data offset too large for this host".to_string())?;
        let mut tensors = std::collections::HashMap::new();
        for (name, spec) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = spec["dtype"]
                .as_str()
                .ok_or_else(|| format!("{name}: missing dtype"))?;
            let shape: Vec<u64> = spec["shape"]
                .as_array()
                .ok_or_else(|| format!("{name}: missing shape"))?
                .iter()
                .map(|v| v.as_u64().ok_or_else(|| format!("{name}: invalid shape")))
                .collect::<Result<_, _>>()?;
            let offs = spec["data_offsets"]
                .as_array()
                .ok_or_else(|| format!("{name}: missing data_offsets"))?;
            if offs.len() != 2 {
                return Err(format!("{name}: invalid data_offsets"));
            }
            let begin = offs[0]
                .as_u64()
                .ok_or_else(|| format!("{name}: invalid start offset"))?;
            let end = offs[1]
                .as_u64()
                .ok_or_else(|| format!("{name}: invalid end offset"))?;
            if end < begin || data_start_u64.checked_add(end).is_none_or(|v| v > file_len) {
                return Err(format!(
                    "{name}: tensor range is outside the safetensors file"
                ));
            }
            let offset = usize::try_from(data_start_u64 + begin)
                .map_err(|_| format!("{name}: tensor offset too large for this host"))?;
            let len = usize::try_from(end - begin)
                .map_err(|_| format!("{name}: tensor length too large for this host"))?;
            if dtype != "F32" {
                return Err(format!("{name}: only F32 supported, got {dtype}"));
            }
            tensors.insert(name.clone(), (shape, offset, len));
        }
        Ok(StFile {
            file: std::sync::Mutex::new(file),
            tensors,
        })
    }

    pub fn f32(&self, name: &str, expect: &[u64]) -> Result<Vec<f32>, String> {
        use std::io::{Read as _, Seek as _, SeekFrom};

        let (shape, offset, len) = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let want: u64 = expect.iter().product();
        let have: u64 = shape.iter().product();
        if have != want || *len != want as usize * 4 {
            return Err(format!(
                "{name}: shape {shape:?} len {} != expected {expect:?}",
                *len
            ));
        }
        if is_ple_ngram_weight(name) && *len > MAX_RESIDENT_PLE_NGRAM_BYTES {
            return Err(format!(
                "{name}: refusing to materialize {} bytes of PLE n-gram weights; compile/use a .coli package so n-gram rows stay on NVMe and are range-read on demand",
                *len
            ));
        }
        let mut raw = vec![0_u8; *len];
        let mut file = self
            .file
            .lock()
            .map_err(|_| "safetensors file lock poisoned".to_string())?;
        file.seek(SeekFrom::Start(*offset as u64))
            .map_err(|e| e.to_string())?;
        file.read_exact(&mut raw).map_err(|e| e.to_string())?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

/// Activation applied by Qwen4-Exp's gated RMSNorm at the GDN output.
/// Qwen3.8-Flash-Next explicitly selects Sigmoid; older Qwen variants fall
/// back to hidden_act (normally SiLU).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputGate {
    Silu,
    Sigmoid,
}

impl OutputGate {
    fn gdn_metal_code(self) -> i32 {
        match self {
            Self::Silu => 0,
            Self::Sigmoid => 1,
        }
    }

    fn from_config(v: &serde_json::Value) -> Result<Self, String> {
        let name = v
            .get("output_gate_type")
            .and_then(|x| x.as_str())
            .or_else(|| v.get("hidden_act").and_then(|x| x.as_str()))
            .unwrap_or("silu");
        match name {
            "silu" => Ok(Self::Silu),
            "sigmoid" => Ok(Self::Sigmoid),
            other => Err(format!(
                "unsupported Qwen4 output_gate_type/hidden_act {other:?}; expected silu or sigmoid"
            )),
        }
    }
}

#[derive(Clone)]
pub struct Cfg {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub theta: f32,
    pub experts: usize,
    pub topk: usize,
    pub moe_inter: usize,
    pub shared_inter: usize,
    lin_k_heads: usize,
    lin_k_dim: usize,
    lin_v_heads: usize,
    lin_v_dim: usize,
    conv_kernel: usize,
    max_t: usize,
    pub vocab: usize,
    pub eps: f32,
    pub output_gate: OutputGate,
    gdn_layers: Vec<bool>,
    qsa_layers: Vec<bool>,
    // qwen4 hyper connections
    pub hc_count: usize,
    pub hc_lowrank: usize,
    // qwen4 QSA indexer
    pub idx_n_heads: usize,
    pub idx_kv_heads: usize,
    pub idx_head_dim: usize,
    pub idx_budget: usize,
    pub idx_ratio: usize,
    // qwen4 PLE
    pub ple_layer: i64, // -1 = off
    pub ple_embed_dim: usize,
    pub ple_conv_kernel: usize,
    pub ngram_size: usize,
    pub ngram_heads: usize,
    pub ngram_vocab_base: i64,
    pub ngram_div: i64,
    pub seed: u64,
    pub eos: i64,
}

pub fn load_cfg(path: &Path) -> Result<Cfg, String> {
    let mut v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?).unwrap();
    // Real checkpoints wrap the text backbone under `text_config`; the tiny
    // fixture has it top-level. Read from text_config when present.
    if let Some(tc) = v.get("text_config").and_then(|x| x.as_object()) {
        v = serde_json::Value::Object(tc.clone());
    }
    let get = |k: &str| v.get(k).and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let num = |k: &str| v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
    let output_gate = OutputGate::from_config(&v)?;
    // Context window: the C engine sizes the KV/indexer caches by the
    // runner CTX (getenv CTX, default 65536 — qwen_moe_base.inc), NOT by
    // config max_position_embeddings. The real package advertises
    // 262144; sizing eagerly off that would zero ~51.6 GB of KV f32 on a
    // 16 GB M2 (swap storm: 113 s load + 10x inflated decode spans).
    // Mirror C parity: CTX env wins, capped by the config ceiling.
    let cfg_max_t = get("max_position_embeddings").max(1);
    let ctx = std::env::var("CTX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(65536);
    let max_t = ctx.min(cfg_max_t);
    let rope = v.get("rope_parameters");
    let theta = rope
        .and_then(|r| r.get("rope_theta"))
        .and_then(|x| x.as_f64())
        .map(|x| x as f32)
        .unwrap_or_else(|| num("rope_theta").max(10000000.0));
    let prf = rope
        .and_then(|r| r.get("partial_rotary_factor"))
        .and_then(|x| x.as_f64())
        .map(|x| x as f32)
        .unwrap_or(1.0);
    let head_dim = get("head_dim").max(get("hidden_size") / get("num_attention_heads").max(1));
    let layer_types = v
        .get("layer_types")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default();
    let gdn_layers: Vec<bool> = layer_types
        .iter()
        .map(|t| t.as_str() == Some("linear_attention"))
        .collect();

    // qwen4 keys are TOP-LEVEL in config.json (C engine reads from root)
    let idx_n_heads = get("indexer_n_heads");
    let idx_kv_heads = get("indexer_kv_heads");
    let idx_head_dim = get("indexer_head_dim");
    let idx_budget = get("indexer_budget");
    let idx_ratio = get("indexer_compress_ratio");
    // QSA layers = full_attention layers when the indexer is configured
    let qsa_layers: Vec<bool> = layer_types
        .iter()
        .map(|t| t.as_str() == Some("full_attention") && idx_n_heads > 0)
        .collect();

    let hc_count = get("hc_count");
    let hc_lowrank = get("hc_lowrank");

    let ple_layer = v
        .get("ple_layer_ids")
        .and_then(|x| x.as_array())
        .and_then(|a| a.first())
        .and_then(|x| x.as_i64())
        .unwrap_or(-1);
    let ngram_size = get("ngram_size");
    let heads_per = get("heads_per_ngram");
    let ngram_heads = if ngram_size > 1 {
        heads_per * (ngram_size - 1)
    } else {
        0
    };
    let ple_embed_dim = get("ple_embed_dim");
    let ple_conv_kernel = get("ple_conv_kernel_size");
    let ngram_vocab_base = v
        .get("ngram_vocab_size_base")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    let ngram_div = v
        .get("make_ngram_vocab_size_divisible_by")
        .and_then(|x| x.as_i64())
        .unwrap_or(1);

    let cfg = Cfg {
        hidden: get("hidden_size"),
        layers: get("num_hidden_layers"),
        heads: get("num_attention_heads"),
        kv_heads: get("num_key_value_heads"),
        head_dim,
        rotary_dim: (head_dim as f32 * prf) as usize,
        theta,
        experts: get("num_experts"),
        topk: get("num_experts_per_tok"),
        moe_inter: get("moe_intermediate_size"),
        shared_inter: get("shared_expert_intermediate_size"),
        lin_k_heads: get("linear_num_key_heads"),
        lin_k_dim: get("linear_key_head_dim"),
        lin_v_heads: get("linear_num_value_heads"),
        lin_v_dim: get("linear_value_head_dim"),
        conv_kernel: get("linear_conv_kernel_dim"),
        max_t,
        vocab: get("vocab_size"),
        eps: num("rms_norm_eps").max(1e-6),
        output_gate,
        gdn_layers,
        qsa_layers,
        hc_count,
        hc_lowrank,
        idx_n_heads,
        idx_kv_heads,
        idx_head_dim,
        idx_budget,
        idx_ratio,
        ple_layer,
        ple_embed_dim,
        ple_conv_kernel,
        ngram_size,
        ngram_heads,
        ngram_vocab_base,
        ngram_div,
        seed: v.get("seed").and_then(|x| x.as_u64()).unwrap_or(0),
        eos: v.get("eos_token_id").and_then(|x| x.as_i64()).unwrap_or(-1),
    };
    if cfg.layers != cfg.gdn_layers.len() {
        return Err(format!(
            "layer_types {} != num_hidden_layers {}",
            cfg.gdn_layers.len(),
            cfg.layers
        ));
    }
    Ok(cfg)
}

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------

/// Resume cursor for a scheduler-blocked token forward (issue #53). The
/// layer's MoE input and injectors are pure snapshots (no persistent state
/// is mutated in the MoE phase), so re-entering the MoE phase once the
/// experts are resident reproduces the canonical forward byte-identically.
pub struct TokenPause {
    pub layer: usize,
    /// The cold routed experts that blocked this layer (the driver loads
    /// them through the scheduler before resubmitting the op).
    pub experts: Vec<u32>,
    /// MoE input for `layer` (post hyper-connection mix).
    pub x: Vec<f32>,
    /// Hyper-connection injectors paired with `x`.
    pub inj: Vec<f32>,
    /// The token stream (all layers before `layer` applied).
    pub stream: Vec<f32>,
    pub token: usize,
    pub pos: usize,
}

/// Outcome of one scheduler-driven forward (issue #53 engine boundary: the
/// scheduler sees generic step outcomes, never model internals).
#[derive(Debug, Clone, PartialEq)]
pub enum SchedForward {
    /// The full token forward completed.
    Logits(Vec<f32>),
    /// The forward stopped at a layer whose routed experts are cold. The
    /// caller loads them, then resubmits the SAME (token, pos).
    NeedExperts { layer: usize, experts: Vec<u32> },
}

pub enum WtBytes {
    /// Canonical BF16 row-major matrix. The optional Metal tensor is created
    /// lazily and remains tied to this exact byte allocation.
    Bf16 {
        weights: Vec<u8>,
        metal_tensor: std::sync::Mutex<usize>,
    },
    /// OCP MXFP4 row-major packed E2M1 nibbles plus one E8M0 scale byte per
    /// 32 input columns. The optional Metal tensor is created lazily on first
    /// use and is owned by this weight representation.
    Mxfp4 {
        weights: Vec<u8>,
        scales: Vec<u8>,
        metal_tensor: std::sync::Mutex<usize>,
    },
    /// Signed INT8 with one FP32 scale per 32 input columns. This is an
    /// experimental GDN qualification format; weights remain row-major.
    Q8Block {
        weights: Vec<u8>,
        scales: Vec<u8>,
        block: usize,
        residuals: usize,
        metal_tensor: std::sync::Mutex<usize>,
    },
    /// Byte-exact native GGUF/GGML row blocks. These bytes are never
    /// requantized; CPU and CUDA kernels consume the source representation.
    Gguf {
        weights: Vec<u8>,
        dtype: ggufsource::GgmlType,
    },
}

impl Clone for WtBytes {
    fn clone(&self) -> Self {
        match self {
            Self::Bf16 { weights, .. } => Self::Bf16 {
                weights: weights.clone(),
                metal_tensor: std::sync::Mutex::new(0),
            },
            Self::Mxfp4 {
                weights, scales, ..
            } => Self::Mxfp4 {
                weights: weights.clone(),
                scales: scales.clone(),
                // A native tensor handle is tied to the original byte buffers.
                // Clones must create their own handle lazily.
                metal_tensor: std::sync::Mutex::new(0),
            },
            Self::Q8Block { weights, scales, block, residuals, .. } => Self::Q8Block {
                weights: weights.clone(),
                scales: scales.clone(),
                block: *block,
                residuals: *residuals,
                metal_tensor: std::sync::Mutex::new(0),
            },
            Self::Gguf { weights, dtype } => Self::Gguf {
                weights: weights.clone(),
                dtype: *dtype,
            },
        }
    }
}

impl Drop for WtBytes {
    fn drop(&mut self) {
        let metal_tensor = match self {
            Self::Bf16 { metal_tensor, .. }
            | Self::Mxfp4 { metal_tensor, .. }
            | Self::Q8Block { metal_tensor, .. } => Some(metal_tensor),
            Self::Gguf { .. } => None,
        };
        if let Some(metal_tensor) = metal_tensor {
            let raw = std::mem::take(
                metal_tensor
                    .get_mut()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            );
            if raw != 0 {
                unsafe {
                    logan_metal::coli_metal_tensor_free(raw as *mut logan_metal::ColiMetalTensor);
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct Wt {
    f: Vec<f32>,
    /// Physical bytes when loaded from a COLI package. The representation is
    /// explicit so resident MXFP4 matrices never masquerade as BF16.
    bytes: Option<WtBytes>,
    o: usize,
    i: usize,
}

impl Wt {
    fn bf16_bytes(&self) -> Option<&[u8]> {
        match self.bytes.as_ref()? {
            WtBytes::Bf16 { weights, .. } => Some(weights),
            WtBytes::Mxfp4 { .. } | WtBytes::Q8Block { .. } | WtBytes::Gguf { .. } => None,
        }
    }

    fn row_f32(&self, row: usize) -> Vec<f32> {
        assert!(row < self.o);
        if !self.f.is_empty() {
            return self.f[row * self.i..(row + 1) * self.i].to_vec();
        }
        match self
            .bytes
            .as_ref()
            .expect("resident weight has physical bytes")
        {
            WtBytes::Bf16 { weights, .. } => (0..self.i)
                .map(|col| {
                    let off = (row * self.i + col) * 2;
                    let u = u16::from_le_bytes([weights[off], weights[off + 1]]);
                    f32::from_bits((u as u32) << 16)
                })
                .collect(),
            WtBytes::Mxfp4 {
                weights, scales, ..
            } => {
                let rb = self.i.div_ceil(2);
                let ng = self.i.div_ceil(32);
                let wr = &weights[row * rb..(row + 1) * rb];
                let sr = &scales[row * ng..(row + 1) * ng];
                const MX4: [f32; 16] = [
                    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0,
                    -4.0, -6.0,
                ];
                (0..self.i)
                    .map(|col| {
                        let packed = wr[col / 2];
                        let code = if col & 1 == 0 {
                            packed & 0x0f
                        } else {
                            packed >> 4
                        };
                        let scale = f32::from_bits((sr[col / 32] as u32) << 23);
                        let mut value = MX4[code as usize] * scale;
                        let full_wb = self.o * rb;
                        let full_sb = self.o * ng;
                        if weights.len() >= full_wb * 2 && scales.len() >= full_sb * 2 {
                            let wr2 = &weights[full_wb + row * rb..full_wb + (row + 1) * rb];
                            let sr2 = &scales[full_sb + row * ng..full_sb + (row + 1) * ng];
                            let p2 = wr2[col / 2];
                            let c2 = if col & 1 == 0 { p2 & 0x0f } else { p2 >> 4 };
                            let s2 = f32::from_bits((sr2[col / 32] as u32) << 23);
                            value += MX4[c2 as usize] * s2;
                        }
                        if weights.len() >= full_wb * 3 && scales.len() >= full_sb * 3 {
                            let wr3 = &weights[2 * full_wb + row * rb..2 * full_wb + (row + 1) * rb];
                            let sr3 = &scales[2 * full_sb + row * ng..2 * full_sb + (row + 1) * ng];
                            let p3 = wr3[col / 2];
                            let c3 = if col & 1 == 0 { p3 & 0x0f } else { p3 >> 4 };
                            let s3 = f32::from_bits((sr3[col / 32] as u32) << 23);
                            value += MX4[c3 as usize] * s3;
                        }
                        value
                    })
                    .collect()
            },
            WtBytes::Q8Block { weights, scales, block, residuals, .. } => {
                let ng = self.i.div_ceil(*block);
                (0..self.i)
                    .map(|col| {
                        let q = weights[row * self.i + col] as i8 as f32;
                        let so = (row * ng + col / *block) * 4;
                        let scale = f32::from_le_bytes(scales[so..so + 4].try_into().unwrap());
                        let mut value = q * scale;
                        if *residuals == 1 && *block == 32 {
                            let base = self.o * self.i;
                            let rv_base = base;
                            let ri_base = rv_base + self.o * ng * 2;
                            let group = col / 32;
                            let idx = weights[ri_base + row * ng + group] as usize;
                            if col % 32 == idx {
                                let ro = rv_base + (row * ng + group) * 2;
                                let rb = u16::from_le_bytes([weights[ro], weights[ro + 1]]);
                                value += f32::from_bits((rb as u32) << 16);
                            }
                        }
                        value
                    })
                    .collect()
            }
            WtBytes::Gguf { weights, dtype } => {
                let row_bytes = dtype
                    .stored_bytes(self.i as u64)
                    .expect("validated GGUF row geometry") as usize;
                let start = row * row_bytes;
                ggufsource::decode_row(*dtype, &weights[start..start + row_bytes], self.i)
                    .expect("validated GGUF row")
            }
        }
    }
}

#[derive(Clone)]
struct Layer {
    in_ln: Vec<f32>,
    is_gdn: bool,
    is_qsa: bool,
    // GDN
    gdn_a_log: Vec<f32>,
    gdn_dt_bias: Vec<f32>,
    gdn_conv1d: Vec<f32>,
    gdn_in_a: Wt,
    gdn_in_b: Wt,
    gdn_in_qkv: Wt,
    gdn_in_z: Wt,
    gdn_norm: Vec<f32>,
    gdn_out: Wt,
    // full attention
    attn_q: Wt,
    attn_k: Wt,
    attn_v: Wt,
    attn_o: Wt,
    attn_qn: Vec<f32>,
    attn_kn: Vec<f32>,
    // QSA indexer
    index_qk: Wt,
    idx_qn: Vec<f32>,
    idx_kn: Vec<f32>,
    // hyper connections (attn + mlp sides)
    hc_norm: Vec<f32>,
    hc_mix_down: Wt,
    hc_mix_up: Wt,
    hc_inject: Wt,
    hc_mlp_norm: Vec<f32>,
    hc_mlp_mix_down: Wt,
    hc_mlp_mix_up: Wt,
    hc_mlp_inject: Wt,
    // MoE
    router: Wt,
    se_gate: Wt,
    se_up: Wt,
    se_down: Wt,
    se_g: Wt,
}

struct HcGlobal {
    norm: Vec<f32>,
    mix_down: Wt,
    mix_up: Wt,
}

impl Layer {
    /// Placeholder for the per-token mem::replace swap in forward_token
    /// (zero-sized weights; never used for compute).
    fn empty() -> Layer {
        Layer {
            in_ln: vec![],
            is_gdn: false,
            is_qsa: false,
            gdn_a_log: vec![],
            gdn_dt_bias: vec![],
            gdn_conv1d: vec![],
            gdn_in_a: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            gdn_in_b: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            gdn_in_qkv: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            gdn_in_z: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            gdn_norm: vec![],
            gdn_out: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            attn_q: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            attn_k: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            attn_v: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            attn_o: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            attn_qn: vec![],
            attn_kn: vec![],
            index_qk: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            idx_qn: vec![],
            idx_kn: vec![],
            hc_norm: vec![],
            hc_mix_down: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            hc_mix_up: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            hc_inject: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            hc_mlp_norm: vec![],
            hc_mlp_mix_down: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            hc_mlp_mix_up: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            hc_mlp_inject: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            router: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            se_gate: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            se_up: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            se_down: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
            se_g: Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            },
        }
    }
}

pub struct Model {
    cfg: Cfg,
    /// .coli package for on-demand expert/ngram fetches (None in safetensors
    /// mode). ponytail: no cache yet — each fetch re-reads the record; add a
    /// per-layer FIFO when disk shows in profiles.
    coli: Option<colisource::ColiSource>,
    /// Native GGUF source. Dense quantized weights are resident as their
    /// original GGML blocks; routed experts and PLE rows remain file-backed.
    gguf: Option<ggufsource::GgufSource>,
    /// GGUF Qwen4Exp reorders GDN V heads from HF grouped order to tiled
    /// broadcast order. Keeping that layout at runtime avoids rewriting any
    /// quantized matrix bytes; K-head selection becomes h % kheads.
    gdn_v_tiled: bool,
    /// Qwen4Exp GGUF keeps the checkpoint's interleaved rotary pairing.
    /// Existing COLI packages retain their historical physical convention.
    rope_interleaved: bool,
    embed: Wt,
    lm_head: Wt,
    /// Optional single-copy 16 KiB-aligned BF16 LM head storage. The generic
    /// Metal backend can wrap this allocation zero-copy instead of snapshotting
    /// the ~1.27 GiB Qwen3.8 head on first decode.
    lm_head_aligned: Option<AlignedBuf>,
    /// Persistent native wrapper for `lm_head_aligned` (stored as usize so the
    /// model remains Send without exposing the opaque C++ pointer type).
    lm_head_metal_tensor: usize,
    final_norm: Vec<f32>,
    layers: Vec<Layer>,
    experts: Vec<Vec<[Wt; 3]>>,
    hc_global: HcGlobal,
    // PLE (present when cfg.ple_layer >= 0)
    ple_ngram: Wt,
    ple_key_proj: Wt,
    ple_value_proj: Wt,
    ple_norm_key: Vec<f32>,
    ple_norm_query: Vec<f32>,
    ple_norm_conv: Vec<f32>,
    ple_conv1d: Vec<f32>,
    ple_offsets: Vec<i64>,
    ple_sizes: Vec<i64>,
    ple_mult: Vec<u64>,
    // state
    gdn_conv: Vec<Vec<f32>>,
    gdn_s: Vec<Vec<f32>>,
    // Long-context state is owned only by layers that consume it.
    // GDN layers keep empty KV vectors; QSA index storage is likewise sparse.
    kv_k: Vec<Vec<f32>>, // [layer][kv_head*max_t*head_dim + pos*head_dim + d]
    kv_v: Vec<Vec<f32>>,
    idx_cache: Vec<Vec<f32>>, // [layer][pos*nk], empty unless QSA
    ple_ring: Vec<i64>,
    ple_conv_state: Vec<f32>,
    // ponytail: FIFO expert cache (the C engine's CACHE 0->256 win was
    // 280->173 ms/tok; LRU upgrade if hit-rate plateaus low). Entries own
    // their Metal tensor handles — the C backend keys handles by weight
    // pointer, so stale handles would serve wrong weights.
    /// Pre-resolved package geometry for every routed expert. Canonical decode
    /// uses this just like the scheduled executor so a cache miss never has to
    /// read/parse the Apple8 descriptor from the shard on the hot path.
    expert_plan: Option<crate::plan::Plan>,
    /// LRU expert store (engine-neutral core; slot-owning values).
    expert_store: logan_core::expert::ExpertStore<crate::colisource::SlotExpert>,
    /// Per-token telemetry accumulator (LOGAN_PROFILE=1).
    spans: logan_core::telemetry::TokenSpans,
    /// Previous routed top-k per layer + overlap counters. This is cheap
    /// correctness-neutral instrumentation used to size layer-local expert
    /// residency from observed temporal locality rather than guesswork.
    route_prev: Vec<Vec<usize>>,
    route_overlap_common: Vec<u64>,
    route_overlap_total: Vec<u64>,
    route_overlap_pairs: Vec<u64>,
    /// Process-unique owner identity for native model-scoped resources. The
    /// native GDN cache is keyed by this ID + layer, never by layer alone.
    metal_model_id: u64,
    /// Metal direct path (fused Apple8 moe_topk + coalesced GDN kernels).
    /// Brought up lazily on the first decode token; failures leave it off and
    /// every caller falls back to the CPU reference (C contract).
    metal_direct: bool,
    /// C QWEN_APPLE8_OVERLAP: split-phase moe_topk (submit -> CPU shared
    /// expert -> wait). Default ON (measured +2.5% loss only when OFF).
    metal_overlap: bool,
    /// Per-GDN-layer 16 KiB page-aligned re-home of the BF16 weights the
    /// Metal GDN kernels wrap zero-copy (C qwen_moe.c contract: wqkv/wz/wa/
    /// wb/wout + recurrent state + conv state must be page-aligned, weights
    /// live for the model lifetime, state is MUTATED BY THE GPU). Backed by
    /// one aligned alloc per layer; CPU fallback reads the same memory, so
    /// there is exactly ONE copy of the GDN weights (moved, not duplicated —
    /// the 16 GB M2 budget).
    gdn_metal: Vec<Option<GdnMetalLayer>>,
    /// Experimental constant-weight ANE input-projection islands. Each entry
    /// stays uninitialized unless explicitly selected by QWEN_GDN_ANE.
    gdn_ane: Vec<gdn_ane::GdnAneState>,
    /// Per-attention-layer Metal BF16 projection buffers (lazy build,
    /// mirror of gdn_metal). QWEN_ATTN_METAL=0 opts out.
    attn_metal: Vec<Option<AttnMetalLayer>>,
    /// Scheduler-driven expert acquisition (QWEN_SCHED=1): the forward
    /// reports cold routed experts instead of loading them; the executor
    /// lane loads them through the plan and the forward resumes from a
    /// stashed cursor. Canonical path never reads these.
    sched_mode: bool,
    /// Cold experts of the layer currently being forward-passed (set by the
    /// MoE phase, consumed by `forward_layer`'s block point).
    sched_blocked: Option<Vec<u32>>,
    /// Resume cursor for a blocked token forward (same-op resubmission).
    sched_pause: Option<TokenPause>,
}

// Profiling-only routed-MoE fallback diagnostics. Each benchmark process owns
// one model in practice; counters are process-lifetime and printed only when
// LOGAN_PROFILE is enabled. They make CPU fallback causes attributable instead
// of folding every decline into the broad `fill` span.

fn next_metal_model_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    // Zero is reserved as "no owner" by the native API. Wraparound is not a
    // practical runtime concern, but fail closed rather than aliasing it.
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    assert_ne!(id, 0, "native Metal model id exhausted");
    id
}

impl Drop for Model {
    fn drop(&mut self) {
        // Drop native zero-copy wrappers *before* Rust destroys gdn_metal's
        // aligned backing allocations. The native call shares the GDN mutex
        // with token execution and therefore also waits for any synchronous
        // GDN submission to retire before releasing its wrappers.
        if self.lm_head_metal_tensor != 0 {
            unsafe {
                logan_metal::coli_metal_tensor_free(
                    self.lm_head_metal_tensor as *mut logan_metal::ColiMetalTensor,
                );
            }
            self.lm_head_metal_tensor = 0;
        }
        logan_metal::shared_mxfp4_drop_model(self.metal_model_id);
        logan_metal::gdn_mxfp4_drop_model(self.metal_model_id);
        crate::ffi::gdn_drop_model(self.metal_model_id);
    }
}

/// 16 KiB page-aligned allocation (Metal zero-copy wrap contract).
struct AlignedBuf {
    ptr: *mut u8,
    len: usize, // allocated (page-rounded) length
}

/// Lazy-zero f32 buffer (C calloc parity) for the big sparse caches.
/// KV is allocated only for full-attention layers; indexer storage only for
/// QSA layers. Eager
/// `vec![0.0; n]` forces every page dirty at load — on a 16 GB M2 that
/// swap-storms the box (113 s load, 10x inflated decode spans measured
/// before this). `alloc_zeroed` on the system allocator maps untouched
/// pages to the shared zero page (libmalloc mmap behavior), so the cache
/// costs ~0 RSS until a token actually writes its positions. Touched pages
/// are zeroed by the kernel on first write — same observable semantics as
/// `vec![0.0; n]`, so token math is unaffected.
fn lazy_zeroed_f32(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let layout = std::alloc::Layout::array::<f32>(n).expect("KV layout");
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) } as *mut f32;
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    // SAFETY: alloc_zeroed guarantees initialized-to-zero f32 storage;
    // capacity == len so Vec's drop (dealloc with the same layout) is exact.
    unsafe { Vec::from_raw_parts(ptr, n, n) }
}

// SAFETY: the raw allocation is owned exclusively by this struct; it is
// created and freed on the host thread that owns the Model (decode is a
// single-threaded per-token pipeline; the GPU reads the memory but Metal
// shared-storage buffers are explicitly designed for host+device access).
unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    fn uninitialized(len: usize) -> Option<AlignedBuf> {
        if len == 0 {
            return None;
        }
        let rounded = (len + 16383) & !16383usize;
        #[cfg(unix)]
        {
            let mut ptr: *mut u8 = std::ptr::null_mut();
            let rc = unsafe {
                libc::posix_memalign(
                    &mut ptr as *mut *mut u8 as *mut *mut libc::c_void,
                    16384,
                    rounded,
                )
            };
            if rc != 0 || ptr.is_null() {
                return None;
            }
            Some(AlignedBuf { ptr, len: rounded })
        }
        #[cfg(not(unix))]
        {
            let layout = std::alloc::Layout::from_size_align(rounded, 16384).ok()?;
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return None;
            }
            Some(AlignedBuf { ptr, len: rounded })
        }
    }

    fn zeroed(len: usize) -> Option<AlignedBuf> {
        let buf = Self::uninitialized(len)?;
        unsafe { std::ptr::write_bytes(buf.ptr, 0, buf.len) };
        Some(buf)
    }
    fn as_mut_f32(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut f32, self.len / 4) }
    }
    fn as_mut_u8(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            libc::free(self.ptr as *mut libc::c_void);
        }
        #[cfg(not(unix))]
        if let Ok(layout) = std::alloc::Layout::from_size_align(self.len, 16384) {
            unsafe { std::alloc::dealloc(self.ptr, layout) };
        }
    }
}

/// Page-aligned GDN weights + state for one layer (Metal zero-copy source of
/// truth). The CPU scalar path reads the same memory, so both paths see
/// identical state; layouts are [O, I] row-major to match the Rust `Wt`.
/// Attention projection weights homed into 16 KiB-aligned buffers for the
/// generic Metal BF16 GEMV seam (mirror of GdnMetalLayer: moved, not
/// duplicated; the CPU fallback reads the same memory).
struct AttnMetalLayer {
    /// [q|gate || k || v || optional index_qk, hidden] BF16 rows.
    /// QSA index rows share the same x input and can therefore execute in
    /// the existing QKV dispatch without another command buffer or host wait.
    qkv: *mut u8,
    qkv_rows: usize,
    /// [hidden, heads*hd] BF16 out projection
    o: *mut u8,
    _bufs: Vec<AlignedBuf>,
}

/// Outputs of the input-side attention projection. QSA may additionally
/// carry index_qk so selection does not repeat that matmul on the CPU.
struct AttnProjection {
    qg: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    index_qk: Option<Vec<f32>>,
    metal_ok: bool,
}

struct GdnMetalLayer {
    /// True when the five dense projections are BF16 and live in the aligned
    /// buffers below. MXFP4 layers use this object as a state-only holder;
    /// their persistent quantized weight buffers are owned by WtBytes.
    bf16_weights: bool,
    /// [cdim, hidden] BF16 in_proj_qkv (16 KiB-aligned, zero-copy wrapped)
    wqkv: *mut u8,
    /// [vdim, hidden] BF16 in_proj_z
    wz: *mut u8,
    /// [vheads, hidden] BF16 in_proj_a
    wa: *mut u8,
    /// [vheads, hidden] BF16 in_proj_b
    wb: *mut u8,
    /// [hidden, vdim] BF16 out_proj
    wout: *mut u8,
    /// Lazily-created zero-copy Metal wrappers for the five aligned BF16
    /// projections. They must be released before `_bufs` frees the backing pages.
    tqkv: *mut logan_metal::ColiMetalTensor,
    tz: *mut logan_metal::ColiMetalTensor,
    ta: *mut logan_metal::ColiMetalTensor,
    tb: *mut logan_metal::ColiMetalTensor,
    tout: *mut logan_metal::ColiMetalTensor,
    /// [vheads * kd * vd] recurrent state (f32, GPU-mutated in place)
    state: *mut f32,
    /// [cdim * (kk-1)] conv state (f32, GPU-mutated in place)
    conv_state: *mut f32,
    /// Keeps every allocation alive for the model lifetime.
    _bufs: Vec<AlignedBuf>,
}

impl Drop for GdnMetalLayer {
    fn drop(&mut self) {
        for tensor in [self.tqkv, self.tz, self.ta, self.tb, self.tout] {
            if !tensor.is_null() {
                unsafe { logan_metal::coli_metal_tensor_free(tensor) };
            }
        }
        self.tqkv = std::ptr::null_mut();
        self.tz = std::ptr::null_mut();
        self.ta = std::ptr::null_mut();
        self.tb = std::ptr::null_mut();
        self.tout = std::ptr::null_mut();
    }
}

/// Per-token scratch, allocated once at load (hidden/vocab sized).
struct TokScratch {
    mixed: Vec<f32>,
    attn: Vec<f32>,
    m2: Vec<f32>,
    moe: Vec<f32>,
    inj: Vec<f32>,
    inj2: Vec<f32>,
}

// ---------------------------------------------------------------------------
// math helpers (C-identical, same as qwen-rs)
// ---------------------------------------------------------------------------

/// NEON BF16 dot: y[o] = x[.] · w[o,.], weights BF16 (u16<<16 = f32).
/// 4-lane fma; fp-order differs from scalar (the gate decides).
#[cfg(target_arch = "aarch64")]
fn matmul_bf16_neon(y: &mut [f32], x: &[f32], w: &[u8], o: usize, i: usize) {
    use std::arch::aarch64::*;
    for oo in 0..o {
        let wr = &w[oo * i * 2..(oo + 1) * i * 2];
        let mut acc = unsafe { vdupq_n_f32(0.0) };
        let mut ii = 0;
        while ii + 8 <= i {
            // 8 bf16 -> 4 f32 (u16<<16), 4 x-f32
            unsafe {
                let wv = vld1q_u16(wr[ii * 2..].as_ptr() as *const u16);
                let w0 = vshlq_n_u32(vmovl_u16(vget_low_u16(wv)), 16);
                let w1 = vshlq_n_u32(vmovl_u16(vget_high_u16(wv)), 16);
                let wf0 = vreinterpretq_f32_u32(w0);
                let wf1 = vreinterpretq_f32_u32(w1);
                let x0 = vld1q_f32(x[ii..].as_ptr());
                let x1 = vld1q_f32(x[ii + 4..].as_ptr());
                acc = vfmaq_f32(acc, wf0, x0);
                acc = vfmaq_f32(acc, wf1, x1);
            }
            ii += 8;
        }
        let mut a = unsafe { vaddvq_f32(acc) };
        while ii < i {
            let u = u16::from_le_bytes([wr[ii * 2], wr[ii * 2 + 1]]);
            a += x[ii] * f32::from_bits((u as u32) << 16);
            ii += 1;
        }
        y[oo] = a;
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn matmul_bf16_neon(_y: &mut [f32], _x: &[f32], _w: &[u8], _o: usize, _i: usize) {
    unreachable!();
}

fn matmul_bf16_bytes(y: &mut [f32], x: &[f32], bytes: &[u8], o: usize, i: usize) {
    debug_assert!(bytes.len() >= o * i * 2);

    let bnns = std::env::var("QWEN_BNNS_BF16")
        .map(|v| v != "0")
        .unwrap_or(false);
    if bnns && logan_metal::bnns_bf16_matmul(bytes, x, y, o, i) {
        return;
    }

    let parallel = o * i >= 16_000_000
        && std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            > 1;

    let neon = std::env::var("QWEN_NEON_BF16")
        .map(|v| v != "0")
        .unwrap_or(true);

    #[cfg(target_arch = "aarch64")]
    let neon = neon && o * i >= 1 << 18;
    #[cfg(not(target_arch = "aarch64"))]
    let neon = false;

    if neon {
        if parallel {
            std::thread::scope(|scope| {
                let available = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);
                let nthreads = std::env::var("QWEN_BF16_THREADS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|&n| n > 0)
                    .unwrap_or(available)
                    .min(available)
                    .min(o.max(1));
                let chunk = o.div_ceil(nthreads);
                for (c, yslice) in y.chunks_mut(chunk).enumerate() {
                    let first_row = c * chunk;
                    let rows = yslice.len();
                    let byte_start = first_row * i * 2;
                    let byte_end = byte_start + rows * i * 2;
                    let wslice = &bytes[byte_start..byte_end];
                    scope.spawn(move || {
                        matmul_bf16_neon(yslice, x, wslice, rows, i);
                    });
                }
            });
        } else {
            matmul_bf16_neon(y, x, bytes, o, i);
        }
        return;
    }

    if parallel {
        std::thread::scope(|scope| {
            let nthreads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            let chunk = o.div_ceil(nthreads);

            for (c, yslice) in y.chunks_mut(chunk).enumerate() {
                let rows = c * chunk;
                scope.spawn(move || {
                    for (oo, yv) in yslice.iter_mut().enumerate() {
                        let oo = rows + oo;
                        let mut acc = 0.0_f32;
                        for ii in 0..i {
                            let off = (oo * i + ii) * 2;
                            let u = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                            acc += x[ii] * f32::from_bits((u as u32) << 16);
                        }
                        *yv = acc;
                    }
                });
            }
        });
    } else {
        for oo in 0..o {
            let mut acc = 0.0_f32;
            for ii in 0..i {
                let off = (oo * i + ii) * 2;
                let u = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                acc += x[ii] * f32::from_bits((u as u32) << 16);
            }
            y[oo] = acc;
        }
    }
}

fn matmul_mxfp4_bytes(y: &mut [f32], x: &[f32], weights: &[u8], scales: &[u8], o: usize, i: usize) {
    const MX4: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let rb = i.div_ceil(2);
    let ng = i.div_ceil(32);
    debug_assert!(weights.len() >= o * rb);
    debug_assert!(scales.len() >= o * ng);
    for row in 0..o {
        let wr = &weights[row * rb..(row + 1) * rb];
        let sr = &scales[row * ng..(row + 1) * ng];
        let mut acc = 0.0_f32;
        for col in 0..i {
            let packed = wr[col / 2];
            let code = if col & 1 == 0 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            let scale = f32::from_bits((sr[col / 32] as u32) << 23);
            acc += x[col] * MX4[code as usize] * scale;
        }
        y[row] = acc;
    }
}

fn mxfp4_storage_fmt(weights: &[u8], scales: &[u8], o: usize, i: usize) -> i32 {
    let wb = o.saturating_mul(i.div_ceil(2));
    let sb = o.saturating_mul(i.div_ceil(32));
    if weights.len() >= wb.saturating_mul(3) && scales.len() >= sb.saturating_mul(3) {
        10
    } else if weights.len() >= wb.saturating_mul(2) && scales.len() >= sb.saturating_mul(2) {
        9
    } else {
        7
    }
}

fn matmul_mxfp4_storage(y: &mut [f32], x: &[f32], weights: &[u8], scales: &[u8], o: usize, i: usize) {
    let wb = o * i.div_ceil(2);
    let sb = o * i.div_ceil(32);
    matmul_mxfp4_bytes(y, x, &weights[..wb], &scales[..sb], o, i);
    let fmt = mxfp4_storage_fmt(weights, scales, o, i);
    if fmt >= 9 {
        let mut residual = vec![0.0f32; o];
        matmul_mxfp4_bytes(&mut residual, x, &weights[wb..wb*2], &scales[sb..sb*2], o, i);
        for (dst, corr) in y.iter_mut().zip(residual.iter()) { *dst += *corr; }
    }
    if fmt == 10 {
        let mut residual = vec![0.0f32; o];
        matmul_mxfp4_bytes(&mut residual, x, &weights[wb*2..wb*3], &scales[sb*2..sb*3], o, i);
        for (dst, corr) in y.iter_mut().zip(residual.iter()) { *dst += *corr; }
    }
}

const MXFP4_GROUP_SIZE: usize = 32;
const MXFP4_MAX_E2M1: f32 = 6.0;
const MXFP4_E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

#[inline]
fn mxfp4_runtime_scale(code: u8) -> f32 {
    debug_assert!((1..=254).contains(&code));
    f32::from_bits(u32::from(code) << 23)
}

#[inline]
fn mxfp4_choose_scale(max_abs: f32) -> (u8, f32) {
    if max_abs == 0.0 {
        return (127, 1.0);
    }
    let bits = max_abs.to_bits();
    let biased = ((bits >> 23) & 0xff) as i32;
    let max_exp = if biased == 0 {
        let mantissa = bits & 0x007f_ffff;
        (31 - mantissa.leading_zeros() as i32) - 149
    } else {
        biased - 127
    };
    let mut scale_exp = (max_exp - 2).clamp(-126, 127);
    let mut code = (scale_exp + 127) as u8;
    let mut scale = mxfp4_runtime_scale(code);
    if max_abs > MXFP4_MAX_E2M1 * scale && scale_exp < 127 {
        scale_exp += 1;
        code = (scale_exp + 127) as u8;
        scale = mxfp4_runtime_scale(code);
    }
    (code, scale)
}

#[inline]
fn mxfp4_quantize_value(value: f32, scale: f32) -> u8 {
    let magnitude = (value.abs() / scale).min(MXFP4_MAX_E2M1);
    let mut best_code = 0u8;
    let mut best_error = f32::INFINITY;
    for (code, candidate) in MXFP4_E2M1_MAGNITUDES.iter().copied().enumerate() {
        let error = (magnitude - candidate).abs();
        if error < best_error || (error == best_error && (code & 1) == 0 && (best_code & 1) != 0) {
            best_error = error;
            best_code = code as u8;
        }
    }
    if value.is_sign_negative() { best_code | 0x8 } else { best_code }
}

#[inline]
fn mxfp4_decode_value(code: u8, scale: f32) -> f32 {
    let magnitude = MXFP4_E2M1_MAGNITUDES[(code & 0x7) as usize];
    let signed = if code & 0x8 != 0 { -magnitude } else { magnitude };
    signed * scale
}

/// Deterministic in-memory BF16 -> canonical OCP MXFP4 conversion used only
/// for runtime qualification of dense GDN weights. The permanent production
/// form should be emitted offline by logan-compiler once quality is qualified.
fn quantize_bf16_to_mxfp4(weights: &[u8], rows: usize, cols: usize) -> Option<(Vec<u8>, Vec<u8>)> {
    if rows == 0 || cols == 0 || weights.len() != rows.checked_mul(cols)?.checked_mul(2)? {
        return None;
    }
    let clip_rms = std::env::var("QWEN_GDN_RUNTIME_MXFP4_CLIP_RMS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0);
    let mse_scale = std::env::var("QWEN_GDN_RUNTIME_MXFP4_MSE")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    let mse_radius = std::env::var("QWEN_GDN_RUNTIME_MXFP4_MSE_RADIUS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(3)
        .clamp(1, 8);
    let src_row_bytes = cols * 2;
    let packed_row_bytes = cols.div_ceil(2);
    let scale_row_bytes = cols.div_ceil(MXFP4_GROUP_SIZE);
    let mut packed = vec![0u8; rows.checked_mul(packed_row_bytes)?];
    let mut scales = vec![0u8; rows.checked_mul(scale_row_bytes)?];
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(rows);
    let rows_per = rows.div_ceil(nthreads);

    std::thread::scope(|scope| {
        for ((src_rows, out_rows), scale_rows) in weights
            .chunks(src_row_bytes * rows_per)
            .zip(packed.chunks_mut(packed_row_bytes * rows_per))
            .zip(scales.chunks_mut(scale_row_bytes * rows_per))
        {
            scope.spawn(move || {
                for ((src, dst), dst_scales) in src_rows
                    .chunks_exact(src_row_bytes)
                    .zip(out_rows.chunks_mut(packed_row_bytes))
                    .zip(scale_rows.chunks_mut(scale_row_bytes))
                {
                    for (group, scale_out) in dst_scales.iter_mut().enumerate() {
                        let c0 = group * MXFP4_GROUP_SIZE;
                        let c1 = (c0 + MXFP4_GROUP_SIZE).min(cols);
                        let mut max_abs = 0.0f32;
                        let mut sum_sq = 0.0f32;
                        for col in c0..c1 {
                            let off = col * 2;
                            let bits = u16::from_le_bytes([src[off], src[off + 1]]);
                            let value = f32::from_bits(u32::from(bits) << 16);
                            if !value.is_finite() {
                                // Source model validation should already forbid this.
                                continue;
                            }
                            max_abs = max_abs.max(value.abs());
                            sum_sq += value * value;
                        }
                        let scale_target = if let Some(clip) = clip_rms {
                            let rms = (sum_sq / (c1 - c0).max(1) as f32).sqrt();
                            max_abs.min(rms * clip)
                        } else {
                            max_abs
                        };
                        let (mut scale_code, mut scale) = mxfp4_choose_scale(scale_target);
                        if mse_scale && max_abs != 0.0 {
                            // Search smaller neighboring power-of-two scales (which
                            // trade a small amount of saturation for finer E2M1
                            // resolution) plus one larger scale. The max-based
                            // code is always included, so MSE selection cannot be
                            // worse for weight reconstruction than canonical max.
                            let base = scale_code as i32;
                            let lo = (base - mse_radius).max(1);
                            let hi = (base + 1).min(254);
                            let mut best_code = scale_code;
                            let mut best_scale = scale;
                            let mut best_error = f64::INFINITY;
                            for candidate in lo..=hi {
                                let candidate_code = candidate as u8;
                                let candidate_scale = mxfp4_runtime_scale(candidate_code);
                                let mut error = 0.0f64;
                                for col in c0..c1 {
                                    let off = col * 2;
                                    let bits = u16::from_le_bytes([src[off], src[off + 1]]);
                                    let value = f32::from_bits(u32::from(bits) << 16);
                                    let code = mxfp4_quantize_value(value, candidate_scale);
                                    let delta = value - mxfp4_decode_value(code, candidate_scale);
                                    error += (delta as f64) * (delta as f64);
                                }
                                if error < best_error {
                                    best_error = error;
                                    best_code = candidate_code;
                                    best_scale = candidate_scale;
                                }
                            }
                            scale_code = best_code;
                            scale = best_scale;
                        }
                        *scale_out = scale_code;
                        let mut col = c0;
                        while col < c1 {
                            let off0 = col * 2;
                            let b0 = u16::from_le_bytes([src[off0], src[off0 + 1]]);
                            let v0 = f32::from_bits(u32::from(b0) << 16);
                            let low = mxfp4_quantize_value(v0, scale) & 0x0f;
                            let high = if col + 1 < c1 {
                                let off1 = (col + 1) * 2;
                                let b1 = u16::from_le_bytes([src[off1], src[off1 + 1]]);
                                let v1 = f32::from_bits(u32::from(b1) << 16);
                                (mxfp4_quantize_value(v1, scale) & 0x0f) << 4
                            } else { 0 };
                            dst[col / 2] = low | high;
                            col += 2;
                        }
                    }
                }
            });
        }
    });
    Some((packed, scales))
}

fn quantize_bf16_to_mxfp4x2(weights: &[u8], rows: usize, cols: usize) -> Option<(Vec<u8>, Vec<u8>)> {
    let (base_w, base_s) = quantize_bf16_to_mxfp4(weights, rows, cols)?;
    let rb = cols.div_ceil(2);
    let ng = cols.div_ceil(32);
    let mut residual_bf16 = vec![0u8; weights.len()];
    for row in 0..rows {
        let wr = &base_w[row * rb..(row + 1) * rb];
        let sr = &base_s[row * ng..(row + 1) * ng];
        for col in 0..cols {
            let src_off = (row * cols + col) * 2;
            let bits = u16::from_le_bytes([weights[src_off], weights[src_off + 1]]);
            let value = f32::from_bits(u32::from(bits) << 16);
            let packed = wr[col / 2];
            let code = if col & 1 == 0 { packed & 0x0f } else { packed >> 4 };
            let scale = mxfp4_runtime_scale(sr[col / 32]);
            let residual = value - mxfp4_decode_value(code, scale);
            let rb16 = crate::colisource::f32_to_bf16(residual).to_le_bytes();
            residual_bf16[src_off] = rb16[0];
            residual_bf16[src_off + 1] = rb16[1];
        }
    }
    let (res_w, res_s) = quantize_bf16_to_mxfp4(&residual_bf16, rows, cols)?;
    let mut all_w = Vec::with_capacity(base_w.len() + res_w.len());
    all_w.extend_from_slice(&base_w);
    all_w.extend_from_slice(&res_w);
    let mut all_s = Vec::with_capacity(base_s.len() + res_s.len());
    all_s.extend_from_slice(&base_s);
    all_s.extend_from_slice(&res_s);
    Some((all_w, all_s))
}

fn mxfp4_residual_bf16(source: &[u8], qweights: &[u8], qscales: &[u8], rows: usize, cols: usize) -> Option<Vec<u8>> {
    if source.len() != rows.checked_mul(cols)?.checked_mul(2)? { return None; }
    let rb = cols.div_ceil(2);
    let ng = cols.div_ceil(32);
    if qweights.len() < rows * rb || qscales.len() < rows * ng { return None; }
    let mut residual = vec![0u8; source.len()];
    for row in 0..rows {
        let wr = &qweights[row * rb..(row + 1) * rb];
        let sr = &qscales[row * ng..(row + 1) * ng];
        for col in 0..cols {
            let off = (row * cols + col) * 2;
            let bits = u16::from_le_bytes([source[off], source[off + 1]]);
            let value = f32::from_bits(u32::from(bits) << 16);
            let packed = wr[col / 2];
            let code = if col & 1 == 0 { packed & 0x0f } else { packed >> 4 };
            let scale = mxfp4_runtime_scale(sr[col / 32]);
            let delta = value - mxfp4_decode_value(code, scale);
            let out = crate::colisource::f32_to_bf16(delta).to_le_bytes();
            residual[off] = out[0]; residual[off + 1] = out[1];
        }
    }
    Some(residual)
}

fn quantize_bf16_to_mxfp4x3(weights: &[u8], rows: usize, cols: usize) -> Option<(Vec<u8>, Vec<u8>)> {
    let (w0, s0) = quantize_bf16_to_mxfp4(weights, rows, cols)?;
    let r1 = mxfp4_residual_bf16(weights, &w0, &s0, rows, cols)?;
    let (w1, s1) = quantize_bf16_to_mxfp4(&r1, rows, cols)?;
    let r2 = mxfp4_residual_bf16(&r1, &w1, &s1, rows, cols)?;
    let (w2, s2) = quantize_bf16_to_mxfp4(&r2, rows, cols)?;
    let mut all_w = Vec::with_capacity(w0.len() + w1.len() + w2.len());
    all_w.extend_from_slice(&w0); all_w.extend_from_slice(&w1); all_w.extend_from_slice(&w2);
    let mut all_s = Vec::with_capacity(s0.len() + s1.len() + s2.len());
    all_s.extend_from_slice(&s0); all_s.extend_from_slice(&s1); all_s.extend_from_slice(&s2);
    Some((all_w, all_s))
}

fn quantize_bf16_to_q8_block(weights: &[u8], rows: usize, cols: usize, block: usize) -> Option<(Vec<u8>, Vec<u8>)> {
    if rows == 0 || cols == 0 || weights.len() != rows.checked_mul(cols)?.checked_mul(2)? {
        return None;
    }
    if !matches!(block, 8 | 16 | 32) { return None; }
    let ng = cols.div_ceil(block);
    let mut q = vec![0u8; rows.checked_mul(cols)?];
    let mut scales = vec![0u8; rows.checked_mul(ng)?.checked_mul(4)?];
    let row_bytes = cols * 2;
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(rows);
    let rows_per = rows.div_ceil(nthreads);
    std::thread::scope(|scope| {
        for ((src_rows, q_rows), scale_rows) in weights
            .chunks(row_bytes * rows_per)
            .zip(q.chunks_mut(cols * rows_per))
            .zip(scales.chunks_mut(ng * 4 * rows_per))
        {
            scope.spawn(move || {
                for ((src, dst), sdst) in src_rows
                    .chunks_exact(row_bytes)
                    .zip(q_rows.chunks_mut(cols))
                    .zip(scale_rows.chunks_mut(ng * 4))
                {
                    for group in 0..ng {
                        let c0 = group * block;
                        let c1 = (c0 + block).min(cols);
                        let mut max_abs = 0.0f32;
                        for col in c0..c1 {
                            let off = col * 2;
                            let bits = u16::from_le_bytes([src[off], src[off + 1]]);
                            let value = f32::from_bits(u32::from(bits) << 16);
                            if value.is_finite() { max_abs = max_abs.max(value.abs()); }
                        }
                        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
                        let sb = scale.to_le_bytes();
                        sdst[group * 4..group * 4 + 4].copy_from_slice(&sb);
                        for col in c0..c1 {
                            let off = col * 2;
                            let bits = u16::from_le_bytes([src[off], src[off + 1]]);
                            let value = f32::from_bits(u32::from(bits) << 16);
                            let qi = if value.is_finite() {
                                (value / scale).round().clamp(-127.0, 127.0) as i8
                            } else { 0 };
                            dst[col] = qi as u8;
                        }
                    }
                }
            });
        }
    });
    Some((q, scales))
}

fn matmul_q8_block_bytes(y: &mut [f32], x: &[f32], weights: &[u8], scales: &[u8], o: usize, i: usize, block: usize, residuals: usize) {
    let ng = i.div_ceil(block);
    debug_assert!(weights.len() >= o * i);
    debug_assert!(scales.len() >= o * ng * 4);
    for row in 0..o {
        let wr = &weights[row * i..(row + 1) * i];
        let sr = &scales[row * ng * 4..(row + 1) * ng * 4];
        let mut acc = 0.0f32;
        for group in 0..ng {
            let so = group * 4;
            let scale = f32::from_le_bytes(sr[so..so + 4].try_into().unwrap());
            let c0 = group * block;
            let c1 = (c0 + block).min(i);
            for col in c0..c1 {
                acc += x[col] * (wr[col] as i8 as f32) * scale;
            }
        }
        if residuals == 1 && block == 32 {
            let base = o * i;
            let rv_base = base;
            let ri_base = rv_base + o * ng * 2;
            for group in 0..ng {
                let idx = weights[ri_base + row * ng + group] as usize;
                let col = group * 32 + idx;
                if col < i {
                    let ro = rv_base + (row * ng + group) * 2;
                    let bits = u16::from_le_bytes([weights[ro], weights[ro + 1]]);
                    acc += f32::from_bits((bits as u32) << 16) * x[col];
                }
            }
        }
        y[row] = acc;
    }
}

fn q8_append_residual1(source: &[u8], mut q: Vec<u8>, scales: &[u8], rows: usize, cols: usize) -> Option<Vec<u8>> {
    let block = 32usize;
    let ng = cols.div_ceil(block);
    if source.len() != rows.checked_mul(cols)?.checked_mul(2)? || q.len() != rows * cols || scales.len() < rows * ng * 4 {
        return None;
    }
    let mut residual_values = vec![0u8; rows * ng * 2];
    let mut residual_indices = vec![0u8; rows * ng];
    for row in 0..rows {
        for group in 0..ng {
            let so = (row * ng + group) * 4;
            let scale = f32::from_le_bytes(scales[so..so + 4].try_into().unwrap());
            let c0 = group * block;
            let c1 = (c0 + block).min(cols);
            let mut best_idx = 0usize;
            let mut best_residual = 0.0f32;
            for col in c0..c1 {
                let off = (row * cols + col) * 2;
                let bits = u16::from_le_bytes([source[off], source[off + 1]]);
                let value = f32::from_bits((bits as u32) << 16);
                let recon = (q[row * cols + col] as i8 as f32) * scale;
                let residual = value - recon;
                if residual.abs() > best_residual.abs() {
                    best_residual = residual;
                    best_idx = col - c0;
                }
            }
            let rb = crate::colisource::f32_to_bf16(best_residual).to_le_bytes();
            let ro = (row * ng + group) * 2;
            residual_values[ro] = rb[0]; residual_values[ro + 1] = rb[1];
            residual_indices[row * ng + group] = best_idx as u8;
        }
    }
    q.extend_from_slice(&residual_values);
    q.extend_from_slice(&residual_indices);
    Some(q)
}

fn quantize_wt_bf16_to_q8_block(w: &mut Wt, block: usize, residuals: usize) -> bool {
    let Some(source) = w.bf16_bytes() else {
        return matches!(w.bytes, Some(WtBytes::Q8Block { block: b, residuals: r, .. }) if b == block && r == residuals);
    };
    let Some((base_weights, scales)) = quantize_bf16_to_q8_block(source, w.o, w.i, block) else { return false; };
    let weights = if residuals == 1 {
        if block != 32 { return false; }
        let Some(v) = q8_append_residual1(source, base_weights, &scales, w.o, w.i) else { return false; };
        v
    } else {
        base_weights
    };
    w.bytes = Some(WtBytes::Q8Block {
        weights,
        scales,
        block,
        residuals,
        metal_tensor: std::sync::Mutex::new(0),
    });
    true
}

fn quantize_wt_bf16_to_mxfp4(w: &mut Wt) -> bool {
    let Some(source) = w.bf16_bytes() else { return matches!(w.bytes, Some(WtBytes::Mxfp4 { .. })); };
    let residual3 = std::env::var("QWEN_GDN_RUNTIME_MXFP4_RESIDUAL3")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    let residual2 = std::env::var("QWEN_GDN_RUNTIME_MXFP4_RESIDUAL2")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    let packed = if residual3 {
        quantize_bf16_to_mxfp4x3(source, w.o, w.i)
    } else if residual2 {
        quantize_bf16_to_mxfp4x2(source, w.o, w.i)
    } else {
        quantize_bf16_to_mxfp4(source, w.o, w.i)
    };
    let Some((weights, scales)) = packed else { return false; };
    w.bytes = Some(WtBytes::Mxfp4 {
        weights,
        scales,
        metal_tensor: std::sync::Mutex::new(0),
    });
    true
}

fn matmul(y: &mut [f32], x: &[f32], w: &Wt) {
    let (o, i) = (w.o, w.i);
    if let Some(bytes) = &w.bytes {
        match bytes {
            WtBytes::Bf16 {
                weights,
                metal_tensor,
            } => {
                // Large dense BF16 GEMVs (most importantly the ~1.27 GiB LM
                // head) are bandwidth-bound and dramatically faster on Metal
                // than the scalar/NEON fallback. Keep the threshold high so
                // small projections do not pay command-buffer overhead or
                // acquire duplicate Metal storage unnecessarily.
                let metal_bf16 = std::env::var("QWEN_METAL_BF16")
                    .map(|v| v != "0")
                    .unwrap_or(true)
                    && o >= 65_536;
                if metal_bf16 {
                    let mut handle = metal_tensor
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let mut tensor = *handle as *mut logan_metal::ColiMetalTensor;
                    if logan_metal::metal_matmul(&mut tensor, y, x, weights, &[], 5, i, o) {
                        *handle = tensor as usize;
                        return;
                    }
                    *handle = tensor as usize;
                }
                matmul_bf16_bytes(y, x, weights, o, i);
            }
            WtBytes::Mxfp4 {
                weights,
                scales,
                metal_tensor,
            } => {
                let mut handle = metal_tensor
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let mut tensor = *handle as *mut logan_metal::ColiMetalTensor;
                let fmt = mxfp4_storage_fmt(weights, scales, o, i);
                if logan_metal::metal_matmul(&mut tensor, y, x, weights, scales, fmt, i, o) {
                    *handle = tensor as usize;
                    return;
                }
                *handle = tensor as usize;
                drop(handle);
                matmul_mxfp4_storage(y, x, weights, scales, o, i);
            }
            WtBytes::Q8Block {
                weights,
                scales,
                block,
                residuals,
                metal_tensor,
            } => {
                let mut handle = metal_tensor
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let mut tensor = *handle as *mut logan_metal::ColiMetalTensor;
                let fmt = if *block == 32 && *residuals == 1 { 14 } else { match *block { 32 => 11, 16 => 12, 8 => 13, _ => 0 } };
                if fmt != 0 && logan_metal::metal_matmul(&mut tensor, y, x, weights, scales, fmt, i, o) {
                    *handle = tensor as usize;
                    return;
                }
                *handle = tensor as usize;
                drop(handle);
                matmul_q8_block_bytes(y, x, weights, scales, o, i, *block, *residuals);
            }
            WtBytes::Gguf { weights, dtype } => {
                let row_bytes = dtype
                    .stored_bytes(i as u64)
                    .expect("validated GGUF row geometry") as usize;
                debug_assert_eq!(weights.len(), row_bytes * o);
                let parallel = o * i >= 1_000_000
                    && std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1)
                        > 1;
                if parallel {
                    let workers = std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1);
                    let chunk = o.div_ceil(workers);
                    std::thread::scope(|scope| {
                        for (chunk_idx, ys) in y.chunks_mut(chunk).enumerate() {
                            let first_row = chunk_idx * chunk;
                            scope.spawn(move || {
                                for (local_row, dst) in ys.iter_mut().enumerate() {
                                    let row = first_row + local_row;
                                    let raw = &weights[row * row_bytes..(row + 1) * row_bytes];
                                    *dst = ggufsource::dot_row(*dtype, raw, x)
                                        .expect("validated GGUF quantized row");
                                }
                            });
                        }
                    });
                } else {
                    for (row, dst) in y.iter_mut().enumerate() {
                        let raw = &weights[row * row_bytes..(row + 1) * row_bytes];
                        *dst = ggufsource::dot_row(*dtype, raw, x)
                            .expect("validated GGUF quantized row");
                    }
                }
            }
        }
        return;
    }
    // ponytail: thread::scope per call costs ~50-100us of spawn; only
    // parallelize in-memory f32 matrices big enough to amortize it.
    let parallel = o * i >= 16_000_000
        && std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            > 1;
    if parallel {
        std::thread::scope(|s| {
            let nthreads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            let chunk = o.div_ceil(nthreads);
            for (c, yslice) in y.chunks_mut(chunk).enumerate() {
                let rows = c * chunk;
                let (x, w) = (&*x, &*w);
                s.spawn(move || {
                    for (oo, yv) in yslice.iter_mut().enumerate() {
                        let oo = rows + oo;
                        let mut acc = 0.0_f32;
                        for ii in 0..i {
                            acc += x[ii] * w.f[oo * i + ii];
                        }
                        *yv = acc;
                    }
                });
            }
        });
    } else {
        for oo in 0..o {
            let mut acc = 0.0_f32;
            for ii in 0..i {
                acc += x[ii] * w.f[oo * i + ii];
            }
            y[oo] = acc;
        }
    }
}

/// Encode several resident MXFP4 GEMVs that consume the same activation in a
/// single Metal command buffer. Returns false without changing numerical state
/// when any weight is not MXFP4 or Metal declines, so callers can fall back to
/// the established per-matrix path.
fn matmul_mxfp4_multi(ys: &mut [&mut [f32]], x: &[f32], ws: &[&Wt]) -> bool {
    if ys.is_empty() || ys.len() != ws.len() {
        return false;
    }
    let mut parts = Vec::with_capacity(ws.len());
    for &w in ws {
        let Some(WtBytes::Mxfp4 {
            weights,
            scales,
            metal_tensor,
        }) = w.bytes.as_ref()
        else {
            return false;
        };
        let fmt = mxfp4_storage_fmt(weights, scales, w.o, w.i);
        parts.push((
            weights.as_slice(),
            scales.as_slice(),
            metal_tensor,
            w.i,
            w.o,
            fmt,
        ));
    }

    let mut guards = Vec::with_capacity(parts.len());
    for (_, _, metal_tensor, _, _, _) in &parts {
        guards.push(
            metal_tensor
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
    }

    let mut descs = Vec::with_capacity(parts.len());
    for ((y, part), guard) in ys.iter_mut().zip(parts.iter()).zip(guards.iter()) {
        let (weights, scales, _, input, output, fmt) = *part;
        descs.push(logan_metal::MetalMatmulDesc {
            tensor: **guard as *mut logan_metal::ColiMetalTensor,
            y: &mut **y,
            weights,
            scales,
            fmt,
            i: input,
            o: output,
        });
    }

    let ok = logan_metal::metal_matmul_multi(x, &mut descs);
    for (guard, desc) in guards.iter_mut().zip(descs.iter()) {
        **guard = desc.tensor as usize;
    }
    ok
}

/// Full one-command-buffer quantized Gated DeltaNet decode. Supports MXFP4
/// qualification formats and block-scaled INT8 while sharing persistent Metal
/// tensor handles and state-only GdnMetalLayer storage.
fn gdn_mxfp4_full_token(
    model_id: u64,
    li: usize,
    layer: &Layer,
    gm: &mut GdnMetalLayer,
    cfg: &Cfg,
    x: &[f32],
    out: &mut [f32],
) -> i32 {
    if gm.bf16_weights {
        return 0;
    }
    let ws = [
        &layer.gdn_in_qkv,
        &layer.gdn_in_z,
        &layer.gdn_in_a,
        &layer.gdn_in_b,
        &layer.gdn_out,
    ];
    let mut parts = Vec::with_capacity(ws.len());
    for &w in &ws {
        let (weights, scales, metal_tensor, fmt): (&[u8], &[u8], &std::sync::Mutex<usize>, i32) = match w.bytes.as_ref() {
            Some(WtBytes::Bf16 { weights, metal_tensor }) => {
                (weights.as_slice(), &[], metal_tensor, 5)
            }
            Some(WtBytes::Mxfp4 { weights, scales, metal_tensor }) => {
                (weights.as_slice(), scales.as_slice(), metal_tensor, mxfp4_storage_fmt(weights, scales, w.o, w.i))
            }
            Some(WtBytes::Q8Block { weights, scales, block, residuals, metal_tensor }) => {
                let fmt = if *block == 32 && *residuals == 1 { 14 } else { match *block { 32 => 11, 16 => 12, 8 => 13, _ => return 0 } };
                (weights.as_slice(), scales.as_slice(), metal_tensor, fmt)
            }
            _ => return 0,
        };
        parts.push((
            weights,
            scales,
            metal_tensor,
            w.i,
            w.o,
            fmt,
        ));
    }
    let mut guards = Vec::with_capacity(parts.len());
    for (_, _, metal_tensor, _, _, _) in &parts {
        guards.push(
            metal_tensor
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
    }
    let mut descs = Vec::with_capacity(parts.len());
    for (part, guard) in parts.iter().zip(guards.iter()) {
        let (weights, scales, _, input, output, fmt) = *part;
        descs.push(logan_metal::MetalWeightDesc {
            tensor: **guard as *mut logan_metal::ColiMetalTensor,
            weights,
            scales,
            fmt,
            i: input,
            o: output,
        });
    }

    let state_len = cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim;
    let cdim = cfg.lin_k_heads * cfg.lin_k_dim * 2 + cfg.lin_v_heads * cfg.lin_v_dim;
    let conv_len = cdim * cfg.conv_kernel.saturating_sub(1);
    let state = unsafe { std::slice::from_raw_parts_mut(gm.state, state_len) };
    let conv_state = unsafe { std::slice::from_raw_parts_mut(gm.conv_state, conv_len) };
    let rc = logan_metal::gdn_mxfp4(
        model_id,
        li,
        &mut descs,
        x,
        out,
        &layer.gdn_a_log,
        &layer.gdn_dt_bias,
        &layer.gdn_conv1d,
        &layer.gdn_norm,
        state,
        conv_state,
        cfg.hidden,
        cfg.lin_k_heads,
        cfg.lin_k_dim,
        cfg.lin_v_heads,
        cfg.lin_v_dim,
        cfg.conv_kernel,
        cfg.output_gate.gdn_metal_code(),
        cfg.eps,
    );
    for (guard, desc) in guards.iter_mut().zip(descs.iter()) {
        **guard = desc.tensor as usize;
    }
    rc
}

fn rmsnorm_row(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let d = x.len();
    let mut ms = 0.0_f64;
    for i in 0..d {
        ms += x[i] as f64 * x[i] as f64;
    }
    let r = 1.0 / (ms as f32 / d as f32 + eps).sqrt();
    for i in 0..d {
        out[i] = x[i] * r * (1.0 + w[i]);
    }
}

/// MLX's Qwen3.5/3.6 sanitizer folds the Transformers RMSNorm `(1 + weight)`
/// into the stored checkpoint tensor. Those converted tensors therefore use
/// ordinary multiplicative RMSNorm at runtime. Keep this separate from the
/// Qwen4/raw-HF convention above so the two checkpoint families cannot be
/// silently mixed.
fn rmsnorm_row_shifted(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let d = x.len();
    let mut ms = 0.0_f64;
    for i in 0..d {
        ms += x[i] as f64 * x[i] as f64;
    }
    let r = 1.0 / (ms as f32 / d as f32 + eps).sqrt();
    for i in 0..d {
        out[i] = x[i] * r * w[i];
    }
}

fn rmsnorm_grouped(out: &mut [f32], x: &[f32], w: &[f32], hc: usize, d: usize, eps: f32) {
    for g in 0..hc {
        rmsnorm_row(
            &mut out[g * d..g * d + d],
            &x[g * d..g * d + d],
            &w[g * d..g * d + d],
            eps,
        );
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// One depthwise causal Conv1d sample using PyTorch/standard cross-correlation
/// tap order: weight[K-1] multiplies the newest/current sample and weight[0]
/// multiplies the oldest sample in the receptive field.
fn causal_conv1d_sample(current: f32, history: &[f32], weights: &[f32], dilation: usize) -> f32 {
    let mut acc = 0.0_f32;
    for (tap, &weight) in weights.iter().enumerate() {
        let lag = (weights.len() - 1 - tap) * dilation;
        let sample = if lag == 0 { current } else { history[lag - 1] };
        acc += weight * sample;
    }
    acc
}

fn rmsnorm_gated_row(out: &mut [f32], x: &[f32], z: &[f32], w: &[f32], eps: f32, gate: OutputGate) {
    let d = x.len();
    let mut ms = 0.0_f64;
    for i in 0..d {
        ms += x[i] as f64 * x[i] as f64;
    }
    let r = 1.0 / (ms as f32 / d as f32 + eps).sqrt();

    // Dispatch once per value-head row, not once per element. Sigmoid and
    // SiLU both require one exp; this adds no inner-loop branch to Qwen3.8.
    match gate {
        OutputGate::Silu => {
            for i in 0..d {
                out[i] = w[i] * (x[i] * r) * silu(z[i]);
            }
        }
        OutputGate::Sigmoid => {
            for i in 0..d {
                let sigmoid_z = 1.0 / (1.0 + (-z[i]).exp());
                out[i] = w[i] * (x[i] * r) * sigmoid_z;
            }
        }
    }
}

fn softmax_row(x: &mut [f32]) {
    let n = x.len();
    let mut m = -1e30_f32;
    for i in 0..n {
        if x[i] > m {
            m = x[i];
        }
    }
    let mut s = 0.0_f32;
    for i in 0..n {
        x[i] = (x[i] - m).exp();
        s += x[i];
    }
    for i in 0..n {
        x[i] /= s;
    }
}

fn l2norm(x: &mut [f32]) {
    let d = x.len();
    let mut s = 0.0_f64;
    for i in 0..d {
        s += x[i] as f64 * x[i] as f64;
    }
    let r = 1.0 / (s as f32 + 1e-6).sqrt();
    for i in 0..d {
        x[i] *= r;
    }
}

fn rope_angles(pos: usize, cfg: &Cfg) -> Vec<(f32, f32)> {
    let rd = cfg.rotary_dim;
    (0..rd / 2)
        .map(|j| {
            let inv = cfg.theta.powf(-2.0 * j as f32 / rd as f32);
            let ang = pos as f32 * inv;
            (ang.cos(), ang.sin())
        })
        .collect()
}

fn rope_partial_with_angles(
    v: &mut [f32],
    angles: &[(f32, f32)],
    rd: usize,
    interleaved: bool,
) {
    for (j, &(cs, sn)) in angles.iter().enumerate() {
        let (ia, ib) = if interleaved {
            (2 * j, 2 * j + 1)
        } else {
            (j, j + rd / 2)
        };
        let a = v[ia];
        let b = v[ib];
        v[ia] = a * cs - b * sn;
        v[ib] = b * cs + a * sn;
    }
}

fn rope_partial(v: &mut [f32], pos: usize, cfg: &Cfg, interleaved: bool) {
    let rd = cfg.rotary_dim;
    let angles = rope_angles(pos, cfg);
    rope_partial_with_angles(v, &angles, rd, interleaved);
}

// qwen4 PLE helpers (C-identical)
const PLE_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const PLE_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const PLE_M2: u64 = 0x94D0_49BB_1331_11EB;

fn ple_splitmix64(mut v: u64) -> u64 {
    v = v.wrapping_add(PLE_GAMMA);
    v = (v ^ (v >> 30)).wrapping_mul(PLE_M1);
    v = (v ^ (v >> 27)).wrapping_mul(PLE_M2);
    v ^ (v >> 31)
}

fn nth_prime_after(mut p: i64, count: i64) -> i64 {
    for _ in 0..count {
        p += 1;
        loop {
            let mut prime = p >= 2;
            if prime && p % 2 == 0 {
                prime = p == 2;
            }
            if prime {
                let mut d = 3_i64;
                while d * d <= p && d <= 46340 {
                    if p % d == 0 {
                        prime = false;
                        break;
                    }
                    d += 2;
                }
            }
            if prime {
                break;
            }
            p += 1;
        }
    }
    p
}

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

impl Model {
    fn hc_mix(
        &self,
        hc_norm: &[f32],
        hc_down: &Wt,
        hc_up: &Wt,
        hc_inj: Option<&Wt>,
        hx: &[f32],
        out: &mut [f32],
        inject: Option<&mut [f32]>,
    ) {
        let c = &self.cfg;
        let d = c.hidden;
        let hc = c.hc_count;
        let lr = c.hc_lowrank;
        let hcd = hc * d;
        let mut normed = vec![0.0; hcd];
        rmsnorm_grouped(&mut normed, hx, hc_norm, hc, d, c.eps);
        let mut lo = vec![0.0; lr];
        matmul(&mut lo, &normed, hc_down);
        for i in 0..lr {
            lo[i] = silu(lo[i] / hc as f32);
        }
        let mut hi = vec![0.0; hcd];
        matmul(&mut hi, &lo, hc_up);
        for i in 0..hcd {
            hi[i] = 1.0 / (1.0 + (-hi[i]).exp());
        }
        for i in 0..hcd {
            out[i % d] += (hi[i] * normed[i]) * (1.0 / hc as f32);
        }
        if let Some(inj) = inject {
            let bi_w = hc_inj.unwrap();
            let mut bi = vec![0.0; hc];
            matmul(&mut bi, &normed, bi_w);
            for g in 0..hc {
                inj[g] = 2.0 / (1.0 + (-bi[g] / hc as f32).exp());
            }
        }
    }

    /// Qualification-only load-time conversion of GDN dense matrices to
    /// signed INT8 with one FP32 scale per 32 input columns. The permanent
    /// representation should be emitted offline if quality/performance passes.
    pub(crate) fn quantize_gdn_runtime_q8(&mut self) -> bool {
        let enabled = std::env::var("QWEN_GDN_RUNTIME_Q8")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false);
        if !enabled { return false; }
        let scope = std::env::var("QWEN_GDN_RUNTIME_Q8_SCOPE").unwrap_or_else(|_| "all".into());
        let layer_spec = std::env::var("QWEN_GDN_RUNTIME_Q8_LAYERS").unwrap_or_else(|_| "all".into());
        let block = std::env::var("QWEN_GDN_RUNTIME_Q8_BLOCK")
            .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(32);
        if !matches!(block, 8 | 16 | 32) {
            eprintln!("qwen4-rs: unsupported Q8 GDN block size {block}; use 8, 16, or 32");
            return false;
        }
        let residuals = std::env::var("QWEN_GDN_RUNTIME_Q8_RESIDUALS")
            .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        if residuals > 1 || (residuals == 1 && block != 32) {
            eprintln!("qwen4-rs: Q8 GDN residuals currently supports only residuals=1 with block=32");
            return false;
        }
        let selected = |name: &str| scope == "all" || scope.split(',').any(|part| part.trim() == name);
        let selected_layer = |layer: usize| {
            if layer_spec.trim().eq_ignore_ascii_case("all") { return true; }
            layer_spec.split(',').any(|piece| {
                let piece = piece.trim();
                if piece.is_empty() { return false; }
                if let Some((lo, hi)) = piece.split_once('-') {
                    match (lo.trim().parse::<usize>(), hi.trim().parse::<usize>()) {
                        (Ok(lo), Ok(hi)) => lo <= layer && layer <= hi,
                        _ => false,
                    }
                } else {
                    piece.parse::<usize>().ok() == Some(layer)
                }
            })
        };
        let started = std::time::Instant::now();
        let mut converted = 0usize;
        let mut before = 0usize;
        let mut after = 0usize;
        for (li, layer) in self.layers.iter_mut().enumerate() {
            if !layer.is_gdn || !selected_layer(li) { continue; }
            let matrices = [
                ("qkv", &mut layer.gdn_in_qkv),
                ("z", &mut layer.gdn_in_z),
                ("a", &mut layer.gdn_in_a),
                ("b", &mut layer.gdn_in_b),
                ("out", &mut layer.gdn_out),
            ];
            for (name, w) in matrices {
                if !selected(name) { continue; }
                if let Some(bytes) = w.bf16_bytes() {
                    before += bytes.len();
                    if !quantize_wt_bf16_to_q8_block(w, block, residuals) {
                        eprintln!("qwen4-rs: runtime Q8 GDN quantization failed at layer {li} matrix {name}");
                        return false;
                    }
                    if let Some(WtBytes::Q8Block { weights, scales, .. }) = w.bytes.as_ref() {
                        after += weights.len() + scales.len();
                    }
                    converted += 1;
                } else if let Some(WtBytes::Q8Block { weights, scales, .. }) = w.bytes.as_ref() {
                    after += weights.len() + scales.len();
                } else {
                    eprintln!("qwen4-rs: runtime Q8 GDN encountered unsupported weight at layer {li} matrix {name}");
                    return false;
                }
            }
        }
        eprintln!(
            "qwen4-rs: runtime Q8 GDN scope={scope} layers={layer_spec} block={block} residuals={residuals} converted {converted} matrices in {:.1} ms ({:.1} MiB BF16 -> {:.1} MiB Q8)",
            started.elapsed().as_secs_f64() * 1e3,
            before as f64 / (1024.0 * 1024.0),
            after as f64 / (1024.0 * 1024.0),
        );
        converted != 0
    }

    /// Experimental load-time conversion of the five dense GDN matrices to
    /// canonical MXFP4. This deliberately runs before `build_gdn_metal`, so
    /// that object becomes state-only and `gdn_mxfp4_full_token` owns dense
    /// execution. The BF16 Vec is replaced immediately after each matrix is
    /// packed, keeping peak memory close to one matrix's packed output.
    pub(crate) fn quantize_gdn_runtime_mxfp4(&mut self) -> bool {
        let enabled = std::env::var("QWEN_GDN_RUNTIME_MXFP4")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false);
        if !enabled { return false; }
        let scope = std::env::var("QWEN_GDN_RUNTIME_MXFP4_SCOPE").unwrap_or_else(|_| "all".into());
        let layer_spec = std::env::var("QWEN_GDN_RUNTIME_MXFP4_LAYERS").unwrap_or_else(|_| "all".into());
        let selected = |name: &str| {
            scope == "all" || scope.split(',').any(|part| part.trim() == name)
        };
        let selected_layer = |layer: usize| {
            if layer_spec.trim().eq_ignore_ascii_case("all") { return true; }
            layer_spec.split(',').any(|piece| {
                let piece = piece.trim();
                if piece.is_empty() { return false; }
                if let Some((lo, hi)) = piece.split_once('-') {
                    match (lo.trim().parse::<usize>(), hi.trim().parse::<usize>()) {
                        (Ok(lo), Ok(hi)) => lo <= layer && layer <= hi,
                        _ => false,
                    }
                } else {
                    piece.parse::<usize>().ok() == Some(layer)
                }
            })
        };
        let started = std::time::Instant::now();
        let mut converted = 0usize;
        let mut before = 0usize;
        let mut after = 0usize;
        for (li, layer) in self.layers.iter_mut().enumerate() {
            if !layer.is_gdn || !selected_layer(li) { continue; }
            let matrices = [
                ("qkv", &mut layer.gdn_in_qkv),
                ("z", &mut layer.gdn_in_z),
                ("a", &mut layer.gdn_in_a),
                ("b", &mut layer.gdn_in_b),
                ("out", &mut layer.gdn_out),
            ];
            for (name, w) in matrices {
                if !selected(name) { continue; }
                if let Some(bytes) = w.bf16_bytes() {
                    before += bytes.len();
                    if !quantize_wt_bf16_to_mxfp4(w) {
                        eprintln!("qwen4-rs: runtime MXFP4 GDN quantization failed at layer {li} matrix {name}");
                        return false;
                    }
                    if let Some(WtBytes::Mxfp4 { weights, scales, .. }) = w.bytes.as_ref() {
                        after += weights.len() + scales.len();
                    }
                    converted += 1;
                } else if let Some(WtBytes::Mxfp4 { weights, scales, .. }) = w.bytes.as_ref() {
                    after += weights.len() + scales.len();
                } else {
                    eprintln!("qwen4-rs: runtime MXFP4 GDN encountered unsupported weight at layer {li} matrix {name}");
                    return false;
                }
            }
        }
        let scale_policy = if std::env::var("QWEN_GDN_RUNTIME_MXFP4_RESIDUAL3")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
        {
            "mx4+2xmx4res".to_string()
        } else if std::env::var("QWEN_GDN_RUNTIME_MXFP4_RESIDUAL2")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
        {
            "mx4+mx4res".to_string()
        } else if std::env::var("QWEN_GDN_RUNTIME_MXFP4_MSE")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
        {
            let radius = std::env::var("QWEN_GDN_RUNTIME_MXFP4_MSE_RADIUS").unwrap_or_else(|_| "3".into());
            format!("mse:r{radius}")
        } else {
            std::env::var("QWEN_GDN_RUNTIME_MXFP4_CLIP_RMS")
                .map(|v| format!("rms-clip:{v}"))
                .unwrap_or_else(|_| "max".into())
        };
        eprintln!(
            "qwen4-rs: runtime MXFP4 GDN scope={scope} layers={layer_spec} scale={scale_policy} converted {converted} matrices in {:.1} ms ({:.1} MiB BF16 -> {:.1} MiB MXFP4)",
            started.elapsed().as_secs_f64() * 1e3,
            before as f64 / (1024.0 * 1024.0),
            after as f64 / (1024.0 * 1024.0),
        );
        converted != 0
    }

    /// Move one GDN layer's BF16 weights into page-aligned buffers (C
    /// contract: newBufferWithBytesNoCopy requires 16 KiB-aligned pointers,
    /// length page-rounded). Returns None if any alloc fails (Metal GDN then
    /// stays off for that layer; CPU path unaffected).
    fn build_gdn_metal(layer: &mut Layer, cfg: &Cfg) -> Option<GdnMetalLayer> {
        if !layer.is_gdn || !crate::ffi::direct_available() {
            return None;
        }
        let kd = cfg.lin_k_dim;
        let kheads = cfg.lin_k_heads;
        let vd = cfg.lin_v_dim;
        let vheads = cfg.lin_v_heads;
        let cdim = kd * kheads * 2 + vd * vheads;
        let kk = cfg.conv_kernel;
        let move_bf16 = |w: &Wt| -> Option<AlignedBuf> {
            let bytes = w.bf16_bytes()?;
            let mut buf = AlignedBuf::zeroed(bytes.len())?;
            buf.as_mut_u8()[..bytes.len()].copy_from_slice(bytes);
            Some(buf)
        };
        let state_elems = vheads * kd * vd;
        let conv_elems = cdim * kk.saturating_sub(1);
        let state = AlignedBuf::zeroed(state_elems * 4)?;
        let conv_state = AlignedBuf::zeroed(conv_elems * 4)?;
        let state_ptr = state.ptr as *mut f32;
        let conv_state_ptr = conv_state.ptr as *mut f32;

        let all_bf16 = layer.gdn_in_qkv.bf16_bytes().is_some()
            && layer.gdn_in_z.bf16_bytes().is_some()
            && layer.gdn_in_a.bf16_bytes().is_some()
            && layer.gdn_in_b.bf16_bytes().is_some()
            && layer.gdn_out.bf16_bytes().is_some();

        if !all_bf16 {
            // MXFP4 path: only recurrent/conv state needs page-aligned,
            // model-lifetime storage. Dense weights remain in their existing
            // WtBytes and lazily create persistent ColiMetalTensor wrappers.
            return Some(GdnMetalLayer {
                bf16_weights: false,
                wqkv: std::ptr::null_mut(),
                wz: std::ptr::null_mut(),
                wa: std::ptr::null_mut(),
                wb: std::ptr::null_mut(),
                wout: std::ptr::null_mut(),
                tqkv: std::ptr::null_mut(),
                tz: std::ptr::null_mut(),
                ta: std::ptr::null_mut(),
                tb: std::ptr::null_mut(),
                tout: std::ptr::null_mut(),
                state: state_ptr,
                conv_state: conv_state_ptr,
                _bufs: vec![state, conv_state],
            });
        }

        let wqkv = move_bf16(&layer.gdn_in_qkv)?;
        let wz = move_bf16(&layer.gdn_in_z)?;
        let wa = move_bf16(&layer.gdn_in_a)?;
        let wb = move_bf16(&layer.gdn_in_b)?;
        let wout = move_bf16(&layer.gdn_out)?;
        let metal = GdnMetalLayer {
            bf16_weights: true,
            wqkv: wqkv.ptr,
            wz: wz.ptr,
            wa: wa.ptr,
            wb: wb.ptr,
            wout: wout.ptr,
            tqkv: std::ptr::null_mut(),
            tz: std::ptr::null_mut(),
            ta: std::ptr::null_mut(),
            tb: std::ptr::null_mut(),
            tout: std::ptr::null_mut(),
            state: state_ptr,
            conv_state: conv_state_ptr,
            _bufs: vec![wqkv, wz, wa, wb, wout, state, conv_state],
        };

        // The aligned Metal allocation is now the authoritative BF16
        // storage. The old code retained both copies for the model lifetime,
        // which duplicates ~3.9 GiB on Flash-Next's 36 GDN layers.
        let single_copy = std::env::var("QWEN_GDN_SINGLE_COPY")
            .map(|v| v != "0")
            .unwrap_or(true);
        if single_copy {
            layer.gdn_in_qkv.bytes = None;
            layer.gdn_in_z.bytes = None;
            layer.gdn_in_a.bytes = None;
            layer.gdn_in_b.bytes = None;
            layer.gdn_out.bytes = None;
        }
        Some(metal)
    }

    /// Compile/load explicitly selected ANE GDN input-projection islands during
    /// model load instead of charging compilation to the first decoded token.
    ///
    /// Selected BF16 GDN layers are also moved into the existing single-copy
    /// page-aligned storage here. The ANE compiler consumes those immutable
    /// bytes as constants; CPU/Metal fallback and the GDN output projection
    /// continue to use the same authoritative allocation.
    pub(crate) fn warm_selected_gdn_ane(&mut self) {
        if std::env::var("QWEN_GDN_ANE")
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(true)
        {
            return;
        }

        let cfg = self.cfg.clone();
        let kd = cfg.lin_k_dim;
        let kheads = cfg.lin_k_heads;
        let vd = cfg.lin_v_dim;
        let vheads = cfg.lin_v_heads;
        let cdim = kd * kheads * 2 + vd * vheads;
        let vdim = vd * vheads;
        let state_len = vheads * kd * vd;
        let conv_len = cdim * cfg.conv_kernel.saturating_sub(1);
        let started = std::time::Instant::now();
        let mut requested = 0usize;
        let mut warmed = 0usize;

        for li in 0..self.layers.len() {
            if !self.layers[li].is_gdn || !gdn_ane::requested(li) {
                continue;
            }
            requested += 1;

            if self.gdn_metal[li].is_none() {
                let built = Self::build_gdn_metal(&mut self.layers[li], &cfg);
                if let Some(gm) = built.as_ref() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(self.gdn_s[li].as_ptr(), gm.state, state_len);
                        if conv_len != 0 {
                            std::ptr::copy_nonoverlapping(
                                self.gdn_conv[li].as_ptr(),
                                gm.conv_state,
                                conv_len,
                            );
                        }
                    }
                }
                self.gdn_metal[li] = built;
            }

            let Some(gm) = self.gdn_metal[li].as_ref().filter(|gm| gm.bf16_weights) else {
                eprintln!(
                    "qwen4-rs: ANE GDN layer {li} warmup skipped: BF16 aligned weights unavailable"
                );
                continue;
            };

            let ok = unsafe {
                gdn_ane::warm(
                    &mut self.gdn_ane[li],
                    li,
                    cfg.hidden,
                    cdim,
                    vdim,
                    vheads,
                    std::slice::from_raw_parts(gm.wqkv, cdim * cfg.hidden * 2),
                    std::slice::from_raw_parts(gm.wz, vdim * cfg.hidden * 2),
                    std::slice::from_raw_parts(gm.wa, vheads * cfg.hidden * 2),
                    std::slice::from_raw_parts(gm.wb, vheads * cfg.hidden * 2),
                    &self.layers[li].gdn_conv1d,
                    cfg.conv_kernel,
                )
            };
            warmed += usize::from(ok);
        }

        if requested != 0 {
            eprintln!(
                "qwen4-rs: ANE GDN warmup {warmed}/{requested} layers in {:.1} ms",
                started.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    /// Re-home the resident BF16 LM head into one page-aligned allocation.
    /// The old Vec is dropped immediately after the copy, so steady-state
    /// residency remains single-copy and Metal can use newBufferWithBytesNoCopy.
    pub(crate) fn rehome_lm_head_bf16(&mut self) -> bool {
        let enabled = std::env::var("QWEN_LM_HEAD_ALIGNED")
            .map(|v| v != "0")
            .unwrap_or(true);
        if !enabled || self.lm_head_aligned.is_some() {
            return self.lm_head_aligned.is_some();
        }
        let need = match self
            .cfg
            .vocab
            .checked_mul(self.cfg.hidden)
            .and_then(|n| n.checked_mul(2))
        {
            Some(v) => v,
            None => return false,
        };
        let Some(src) = self.lm_head.bf16_bytes() else {
            return false;
        };
        if src.len() != need {
            return false;
        }
        let Some(mut aligned) = AlignedBuf::uninitialized(need) else {
            return false;
        };
        aligned.as_mut_u8()[..need].copy_from_slice(src);
        // The aligned allocation is now authoritative. The CPU fallback below
        // reads these same bytes, so clearing the Vec loses no fallback path.
        self.lm_head.bytes = None;
        self.lm_head_aligned = Some(aligned);
        true
    }

    /// Exact-BF16 LM-head GEMV over the aligned single-copy storage. Backend
    /// selection is deliberately separate from the rest of dense decode so an
    /// LM-head A/B cannot accidentally redirect all GDN projections too.
    /// QWEN_LM_HEAD_BACKEND accepts `cpu`, `bnns`, or `metal` (default `cpu`
    /// until full-runtime residency behaviour is qualified).
    fn lm_head_matmul(&mut self, y: &mut [f32], x: &[f32]) {
        let Some(aligned) = self.lm_head_aligned.as_ref() else {
            matmul(y, x, &self.lm_head);
            return;
        };
        let need = self.cfg.vocab * self.cfg.hidden * 2;
        let bytes = unsafe { std::slice::from_raw_parts(aligned.ptr, need) };
        let backend = std::env::var("QWEN_LM_HEAD_BACKEND")
            .unwrap_or_else(|_| "cpu".to_string());
        if backend.eq_ignore_ascii_case("metal") {
            let mut tensor = self.lm_head_metal_tensor as *mut logan_metal::ColiMetalTensor;
            let ok = logan_metal::metal_matmul(
                &mut tensor,
                y,
                x,
                bytes,
                &[],
                5,
                self.cfg.hidden,
                self.cfg.vocab,
            );
            self.lm_head_metal_tensor = tensor as usize;
            if ok {
                return;
            }
        } else if backend.eq_ignore_ascii_case("bnns")
            && logan_metal::bnns_bf16_matmul(
                bytes,
                x,
                y,
                self.cfg.vocab,
                self.cfg.hidden,
            )
        {
            return;
        }
        matmul_bf16_bytes(y, x, bytes, self.cfg.vocab, self.cfg.hidden);
    }

    fn build_attn_metal(layer: &Layer, cfg: &Cfg) -> Option<AttnMetalLayer> {
        if layer.is_gdn || !crate::ffi::direct_available() {
            return None;
        }
        let move_bf16 = |w: &Wt| -> Option<AlignedBuf> {
            let bytes = w.bf16_bytes()?;
            let mut buf = AlignedBuf::zeroed(bytes.len())?;
            buf.as_mut_u8()[..bytes.len()].copy_from_slice(bytes);
            Some(buf)
        };
        let o = move_bf16(&layer.attn_o)?;
        let q_bytes = layer.attn_q.bf16_bytes()?;
        let k_bytes = layer.attn_k.bf16_bytes()?;
        let v_bytes = layer.attn_v.bf16_bytes()?;
        let idx_bytes = if layer.is_qsa {
            layer.index_qk.bf16_bytes()
        } else {
            None
        };
        let idx_len = idx_bytes.map_or(0, |b| b.len());
        let total = q_bytes.len() + k_bytes.len() + v_bytes.len() + idx_len;
        let row_bytes = cfg.hidden.checked_mul(2)?;
        if total % row_bytes != 0 {
            return None;
        }

        let mut qkv = AlignedBuf::zeroed(total)?;
        let dst = qkv.as_mut_u8();
        let mut off = 0usize;
        for bytes in [q_bytes, k_bytes, v_bytes] {
            dst[off..off + bytes.len()].copy_from_slice(bytes);
            off += bytes.len();
        }
        if let Some(bytes) = idx_bytes {
            dst[off..off + bytes.len()].copy_from_slice(bytes);
        }

        Some(AttnMetalLayer {
            qkv: qkv.ptr,
            qkv_rows: total / row_bytes,
            o: o.ptr,
            _bufs: vec![qkv, o],
        })
    }

    fn gdn_chunk_batched(
        &mut self,
        layer: &mut Layer,
        li: usize,
        xs: &[Vec<f32>],
        outs: &mut [Vec<f32>],
    ) -> bool {
        let rows = xs.len();
        if rows <= 1 || outs.len() != rows || !layer.is_gdn {
            return false;
        }
        let enabled = std::env::var("QWEN_PREFILL_GDN_BATCH")
            .map(|v| v != "0")
            .unwrap_or(true);
        if !enabled {
            return false;
        }

        let c = self.cfg.clone();
        let kd = c.lin_k_dim;
        let kheads = c.lin_k_heads;
        let vd = c.lin_v_dim;
        let vheads = c.lin_v_heads;
        let kdim = kd * kheads;
        let vdim = vd * vheads;
        let cdim = kdim * 2 + vdim;
        let kk = c.conv_kernel;
        let d = c.hidden;
        if xs.iter().any(|x| x.len() != d) || outs.iter().any(|o| o.len() != d) {
            return false;
        }

        // Establish the same aligned single-copy BF16 storage/state used by
        // decode before any causal state is touched.
        if self.gdn_metal[li].is_none() {
            let built = Self::build_gdn_metal(layer, &self.cfg);
            if let Some(gm) = built.as_ref() {
                let state_len = vheads * kd * vd;
                let conv_len = cdim * (kk - 1);
                unsafe {
                    std::ptr::copy_nonoverlapping(self.gdn_s[li].as_ptr(), gm.state, state_len);
                    std::ptr::copy_nonoverlapping(
                        self.gdn_conv[li].as_ptr(),
                        gm.conv_state,
                        conv_len,
                    );
                }
            }
            self.gdn_metal[li] = built;
        }
        let Some(gm) = self.gdn_metal[li].as_ref() else {
            return false;
        };
        if !gm.bf16_weights {
            return false;
        }
        let (wqkv, wz, wa, wb, wout, state_ptr, conv_ptr) = (
            gm.wqkv,
            gm.wz,
            gm.wa,
            gm.wb,
            gm.wout,
            gm.state,
            gm.conv_state,
        );

        let mut x_all = Vec::with_capacity(rows * d);
        for x in xs {
            x_all.extend_from_slice(x);
        }
        let mut qkv_all = vec![0.0_f32; rows * cdim];
        let mut z_all = vec![0.0_f32; rows * vdim];
        let mut a_all = vec![0.0_f32; rows * vheads];
        let mut b_all = vec![0.0_f32; rows * vheads];

        let in_t0 = std::time::Instant::now();
        let input_ok = unsafe {
            crate::ffi::bnns_bf16_matmul_batch(
                std::slice::from_raw_parts(wqkv, cdim * d * 2),
                &x_all,
                &mut qkv_all,
                rows,
                cdim,
                d,
            ) && crate::ffi::bnns_bf16_matmul_batch(
                std::slice::from_raw_parts(wz, vdim * d * 2),
                &x_all,
                &mut z_all,
                rows,
                vdim,
                d,
            ) && crate::ffi::bnns_bf16_matmul_batch(
                std::slice::from_raw_parts(wa, vheads * d * 2),
                &x_all,
                &mut a_all,
                rows,
                vheads,
                d,
            ) && crate::ffi::bnns_bf16_matmul_batch(
                std::slice::from_raw_parts(wb, vheads * d * 2),
                &x_all,
                &mut b_all,
                rows,
                vheads,
                d,
            )
        };
        if !input_ok {
            return false;
        }
        if logan_core::telemetry::enabled() {
            self.spans.gdn_in_proj_ms += in_t0.elapsed().as_secs_f64() * 1e3;
        }

        let mut normed_all = vec![0.0_f32; rows * vdim];
        let rep = vheads / kheads;
        assert!(rep >= 1 && vheads % kheads == 0);
        let state_len = vheads * kd * vd;
        let conv_len = cdim * (kk - 1);
        let conv_st = unsafe { std::slice::from_raw_parts_mut(conv_ptr, conv_len) };
        let state = unsafe { std::slice::from_raw_parts_mut(state_ptr, state_len) };

        for row in 0..rows {
            let qkv = &qkv_all[row * cdim..(row + 1) * cdim];
            let a = &a_all[row * vheads..(row + 1) * vheads];
            let b = &b_all[row * vheads..(row + 1) * vheads];
            let z = &z_all[row * vdim..(row + 1) * vdim];

            let conv_t0 = std::time::Instant::now();
            let mut y = vec![0.0_f32; cdim];
            if kk > 1 {
                for ch in 0..cdim {
                    let mut acc = 0.0_f32;
                    for j in 0..kk {
                        let vv = if j == kk - 1 {
                            qkv[ch]
                        } else {
                            conv_st[ch * (kk - 1) + j]
                        };
                        acc += layer.gdn_conv1d[ch * kk + j] * vv;
                    }
                    y[ch] = silu(acc);
                }
                for ch in 0..cdim {
                    for s in 0..kk - 2 {
                        conv_st[ch * (kk - 1) + s] = conv_st[ch * (kk - 1) + s + 1];
                    }
                    conv_st[ch * (kk - 1) + (kk - 2)] = qkv[ch];
                }
            } else {
                for ch in 0..cdim {
                    y[ch] = silu(layer.gdn_conv1d[ch] * qkv[ch]);
                }
            }
            if logan_core::telemetry::enabled() {
                self.spans.gdn_conv_ms += conv_t0.elapsed().as_secs_f64() * 1e3;
            }

            let prep_t0 = std::time::Instant::now();
            let q_ = &y[..kdim];
            let k_ = &y[kdim..kdim * 2];
            let v_ = &y[kdim * 2..];
            let mut qh = vec![0.0_f32; vheads * kd];
            let mut kh = vec![0.0_f32; vheads * kd];
            let mut vh = vec![0.0_f32; vheads * vd];
            for h in 0..vheads {
                let khd = if self.gdn_v_tiled { h % kheads } else { h / rep };
                for dd in 0..kd {
                    qh[h * kd + dd] = q_[khd * kd + dd];
                    kh[h * kd + dd] = k_[khd * kd + dd];
                }
                for dd in 0..vd {
                    vh[h * vd + dd] = v_[h * vd + dd];
                }
                l2norm(&mut qh[h * kd..h * kd + kd]);
                l2norm(&mut kh[h * kd..h * kd + kd]);
                let sc = 1.0 / (kd as f32).sqrt();
                for dd in 0..kd {
                    qh[h * kd + dd] *= sc;
                }
            }
            if logan_core::telemetry::enabled() {
                self.spans.gdn_prepare_ms += prep_t0.elapsed().as_secs_f64() * 1e3;
            }

            let recur_t0 = std::time::Instant::now();
            let mut kv_mem = vec![0.0_f32; vd];
            for h in 0..vheads {
                let ga =
                    -layer.gdn_a_log[h].exp() * (1.0 + (a[h] + layer.gdn_dt_bias[h]).exp()).ln();
                let gt = ga.exp();
                let bt = 1.0 / (1.0 + (-b[h]).exp());
                let sh = &mut state[h * kd * vd..(h + 1) * kd * vd];
                let qhh = &qh[h * kd..(h + 1) * kd];
                let khh = &kh[h * kd..(h + 1) * kd];
                let vhh = &vh[h * vd..(h + 1) * vd];
                kv_mem.fill(0.0);
                for kk2 in 0..kd {
                    for dd in 0..vd {
                        let si = kk2 * vd + dd;
                        let sv = sh[si] * gt;
                        sh[si] = sv;
                        kv_mem[dd] += sv * khh[kk2];
                    }
                }
                for dd in 0..vd {
                    let delta = (vhh[dd] - kv_mem[dd]) * bt;
                    let mut acc = 0.0_f32;
                    for kk2 in 0..kd {
                        let si = kk2 * vd + dd;
                        let next_s = sh[si] + khh[kk2] * delta;
                        sh[si] = next_s;
                        acc += next_s * qhh[kk2];
                    }
                    kv_mem[dd] = acc;
                }
                vh[h * vd..(h + 1) * vd].copy_from_slice(&kv_mem);
            }
            if logan_core::telemetry::enabled() {
                self.spans.gdn_recur_ms += recur_t0.elapsed().as_secs_f64() * 1e3;
            }

            let gate_t0 = std::time::Instant::now();
            let normed = &mut normed_all[row * vdim..(row + 1) * vdim];
            for h in 0..vheads {
                rmsnorm_gated_row(
                    &mut normed[h * vd..h * vd + vd],
                    &vh[h * vd..h * vd + vd],
                    &z[h * vd..h * vd + vd],
                    &layer.gdn_norm,
                    c.eps,
                    c.output_gate,
                );
            }
            if logan_core::telemetry::enabled() {
                self.spans.gdn_gate_ms += gate_t0.elapsed().as_secs_f64() * 1e3;
            }
        }

        let out_t0 = std::time::Instant::now();
        let mut out_all = vec![0.0_f32; rows * d];
        let out_ok = unsafe {
            crate::ffi::bnns_bf16_matmul_batch(
                std::slice::from_raw_parts(wout, d * vdim * 2),
                &normed_all,
                &mut out_all,
                rows,
                d,
                vdim,
            )
        };
        if out_ok {
            for row in 0..rows {
                outs[row].copy_from_slice(&out_all[row * d..(row + 1) * d]);
            }
        } else {
            let wout_bytes = unsafe { std::slice::from_raw_parts(wout, d * vdim * 2) };
            for row in 0..rows {
                matmul_bf16_bytes(
                    &mut outs[row],
                    &normed_all[row * vdim..(row + 1) * vdim],
                    wout_bytes,
                    d,
                    vdim,
                );
            }
        }
        if logan_core::telemetry::enabled() {
            self.spans.gdn_out_proj_ms += out_t0.elapsed().as_secs_f64() * 1e3;
        }
        true
    }

    fn gdn_token(&mut self, layer: &mut Layer, li: usize, x: &[f32], out: &mut [f32]) {
        let c = self.cfg.clone();
        let kd = c.lin_k_dim;
        let kheads = c.lin_k_heads;
        let vd = c.lin_v_dim;
        let vheads = c.lin_v_heads;
        let kdim = kd * kheads;
        let vdim = vd * vheads;
        let cdim = kdim * 2 + vdim;
        let kk = c.conv_kernel;
        let profile_gdn_parts = logan_core::telemetry::enabled()
            || std::env::var("QWEN_PROFILE_GDN_PARTS")
                .map(|v| v != "0")
                .unwrap_or(false);
        let ane_requested = gdn_ane::requested(li);

        // One-time aligned re-home (C calloc_checked/coli_wt intercept): move
        // the five BF16 GDN matrices into 16 KiB-aligned buffers so Metal can
        // wrap them zero-copy. State lives in the same aligned blocks; the
        // CPU fallback reads the SAME memory (single source of truth).
        if self.gdn_metal[li].is_none() {
            let built = Self::build_gdn_metal(layer, &self.cfg);
            if let Some(gm) = built.as_ref() {
                // A RAM snapshot can be restored before the aligned buffers
                // exist. Seed the newly-created authoritative buffers from the
                // CPU state exactly once so lazy initialization never erases a
                // restored recurrent/conv state. Fresh models simply copy zero.
                let state_len = vheads * kd * vd;
                let conv_len = cdim * (kk - 1);
                unsafe {
                    std::ptr::copy_nonoverlapping(self.gdn_s[li].as_ptr(), gm.state, state_len);
                    std::ptr::copy_nonoverlapping(
                        self.gdn_conv[li].as_ptr(),
                        gm.conv_state,
                        conv_len,
                    );
                }
            }
            self.gdn_metal[li] = built;
        }
        // Metal direct path (C QWEN_GDN_METAL default ON): the coalesced
        // kernels consume the page-aligned re-home above; rc semantics per
        // C contract (0=decline pre-submit, <0 = fatal post-submit). The
        // kernel receives the checkpoint's gated-RMSNorm activation explicitly
        // (0=SiLU, 1=sigmoid), so Qwen3.8-Flash-Next no longer has to fall
        // back to CPU merely because it uses the newer sigmoid gate.
        let gdn_enabled = std::env::var("QWEN_GDN_METAL")
            .map(|v| v != "0")
            .unwrap_or(false);
        let gdn_mxfp4_full = std::env::var("QWEN_GDN_MXFP4_FULL")
            .map(|v| v != "0")
            .unwrap_or(false);
        // This gate is intentionally independent of legacy QWEN_GDN_METAL.
        // The latter remains off by default on Apple Silicon because the old
        // BF16 full-GDN path lost to BNNS; MXFP4 is qualified separately.
        if gdn_mxfp4_full && !ane_requested {
            if let Some(gm) = self.gdn_metal[li].as_mut() {
                if !gm.bf16_weights {
                    let rc = gdn_mxfp4_full_token(self.metal_model_id, li, layer, gm, &c, x, out);
                    if rc > 0 {
                        self.spans.gdn_metal_ok += 1;
                        return;
                    }
                    if rc < 0 {
                        eprintln!(
                            "qwen4-rs: full MXFP4 Metal GDN failed after submission (layer {li})"
                        );
                        std::process::exit(1);
                    }
                }
            }
        }

        if let Some(gm) = &self.gdn_metal[li] {
            if !ane_requested && gdn_enabled && gm.bf16_weights {
                // SAFETY: exact-length views over model-lifetime aligned
                // storage. In overlap mode begin() commits the complete GDN
                // command buffer, then the host uses the otherwise-idle wait
                // window to issue the previous route's MetalIO before finish().
                let (n_state, n_conv) = (vheads * kd * vd, cdim * (kk - 1));
                let overlap_prefetch = std::env::var("QWEN_GDN_PREFETCH_OVERLAP")
                    .map(|v| v != "0")
                    .unwrap_or(false);
                if overlap_prefetch {
                    let pending = unsafe {
                        crate::ffi::gdn_token_begin(
                            self.metal_model_id,
                            li,
                            x,
                            std::slice::from_raw_parts(gm.wqkv, cdim * c.hidden * 2),
                            std::slice::from_raw_parts(gm.wz, vdim * c.hidden * 2),
                            std::slice::from_raw_parts(gm.wa, vheads * c.hidden * 2),
                            std::slice::from_raw_parts(gm.wb, vheads * c.hidden * 2),
                            std::slice::from_raw_parts(gm.wout, c.hidden * vdim * 2),
                            &layer.gdn_a_log,
                            &layer.gdn_dt_bias,
                            &layer.gdn_conv1d,
                            &layer.gdn_norm,
                            std::slice::from_raw_parts_mut(gm.state, n_state),
                            std::slice::from_raw_parts_mut(gm.conv_state, n_conv),
                            c.hidden,
                            kheads,
                            kd,
                            vheads,
                            vd,
                            kk,
                            c.output_gate.gdn_metal_code(),
                            c.eps,
                        )
                    };
                    if let Some(pending) = pending {
                        self.prefetch_previous_route_now(li);
                        let rc = crate::ffi::gdn_token_finish(pending, out);
                        if rc > 0 {
                            self.spans.gdn_metal_ok += 1;
                            return;
                        }
                        if rc < 0 {
                            eprintln!("qwen4-rs: async Metal GDN failed after submission (layer {li})");
                            std::process::exit(1);
                        }
                    }
                } else {
                    let rc = unsafe {
                        crate::ffi::gdn_token(
                            self.metal_model_id,
                            li,
                            x,
                            out,
                            std::slice::from_raw_parts(gm.wqkv, cdim * c.hidden * 2),
                            std::slice::from_raw_parts(gm.wz, vdim * c.hidden * 2),
                            std::slice::from_raw_parts(gm.wa, vheads * c.hidden * 2),
                            std::slice::from_raw_parts(gm.wb, vheads * c.hidden * 2),
                            std::slice::from_raw_parts(gm.wout, c.hidden * vdim * 2),
                            &layer.gdn_a_log,
                            &layer.gdn_dt_bias,
                            &layer.gdn_conv1d,
                            &layer.gdn_norm,
                            std::slice::from_raw_parts_mut(gm.state, n_state),
                            std::slice::from_raw_parts_mut(gm.conv_state, n_conv),
                            c.hidden,
                            kheads,
                            kd,
                            vheads,
                            vd,
                            kk,
                            c.output_gate.gdn_metal_code(),
                            c.eps,
                        )
                    };
                    if rc > 0 {
                        self.spans.gdn_metal_ok += 1;
                        return;
                    }
                    if rc < 0 {
                        eprintln!("qwen4-rs: Metal GDN failed after submission (layer {li})");
                        std::process::exit(1);
                    }
                }
                // Pre-submit decline: scalar path below remains safe because
                // no recurrent state mutation has been submitted.
            }
        }

        let mut qkv = vec![0.0; cdim];
        let mut a = vec![0.0; vheads];
        let mut b = vec![0.0; vheads];
        let mut z = vec![0.0; vdim];
        let gdn_in_t0 = profile_gdn_parts.then(std::time::Instant::now);

        let ane_ok = if ane_requested {
            if let Some(gm) = self.gdn_metal[li].as_ref().filter(|gm| gm.bf16_weights) {
                // SAFETY: GdnMetalLayer owns these exact-length model-lifetime
                // BF16 buffers. ANE compilation consumes them as immutable
                // constants; recurrent state is not touched by this island.
                unsafe {
                    gdn_ane::try_input(
                        &mut self.gdn_ane[li],
                        li,
                        c.hidden,
                        cdim,
                        vdim,
                        vheads,
                        std::slice::from_raw_parts(gm.wqkv, cdim * c.hidden * 2),
                        std::slice::from_raw_parts(gm.wz, vdim * c.hidden * 2),
                        std::slice::from_raw_parts(gm.wa, vheads * c.hidden * 2),
                        std::slice::from_raw_parts(gm.wb, vheads * c.hidden * 2),
                        &layer.gdn_conv1d,
                        kk,
                        x,
                        &mut qkv,
                        &mut z,
                        &mut a,
                        &mut b,
                    )
                }
            } else {
                false
            }
        } else {
            false
        };

        if !ane_ok {
            if let Some(gm) = self.gdn_metal[li].as_mut().filter(|gm| gm.bf16_weights) {
                // These BF16 buffers are already 16 KiB aligned and page-rounded,
                // so the generic Metal GEMV can wrap them zero-copy. Fuse all four
                // input projections into one command buffer; a pre-submit decline
                // falls back to the established CPU/BNNS path below.
                let metal_bf16 = std::env::var("QWEN_GDN_BF16_METAL")
                    .map(|v| v != "0")
                    .unwrap_or(false);
                let metal_ok = if metal_bf16 {
                    let (wqkv, wz, wa, wb) = unsafe {
                        (
                            std::slice::from_raw_parts(gm.wqkv, cdim * c.hidden * 2),
                            std::slice::from_raw_parts(gm.wz, vdim * c.hidden * 2),
                            std::slice::from_raw_parts(gm.wa, vheads * c.hidden * 2),
                            std::slice::from_raw_parts(gm.wb, vheads * c.hidden * 2),
                        )
                    };
                    let mut descs = [
                        logan_metal::MetalMatmulDesc { tensor: gm.tqkv, y: &mut qkv, weights: wqkv, scales: &[], fmt: 5, i: c.hidden, o: cdim },
                        logan_metal::MetalMatmulDesc { tensor: gm.tz, y: &mut z, weights: wz, scales: &[], fmt: 5, i: c.hidden, o: vdim },
                        logan_metal::MetalMatmulDesc { tensor: gm.ta, y: &mut a, weights: wa, scales: &[], fmt: 5, i: c.hidden, o: vheads },
                        logan_metal::MetalMatmulDesc { tensor: gm.tb, y: &mut b, weights: wb, scales: &[], fmt: 5, i: c.hidden, o: vheads },
                    ];
                    let ok = logan_metal::metal_matmul_multi(x, &mut descs);
                    let handles = [descs[0].tensor, descs[1].tensor, descs[2].tensor, descs[3].tensor];
                    drop(descs);
                    gm.tqkv = handles[0];
                    gm.tz = handles[1];
                    gm.ta = handles[2];
                    gm.tb = handles[3];
                    ok
                } else {
                    false
                };
                if !metal_ok {
                    // SAFETY: GdnMetalLayer owns every aligned allocation for the
                    // entire model lifetime. These are the exact BF16 package bytes.
                    unsafe {
                        matmul_bf16_bytes(
                            &mut qkv,
                            x,
                            std::slice::from_raw_parts(gm.wqkv, cdim * c.hidden * 2),
                            cdim,
                            c.hidden,
                        );
                        matmul_bf16_bytes(
                            &mut z,
                            x,
                            std::slice::from_raw_parts(gm.wz, vdim * c.hidden * 2),
                            vdim,
                            c.hidden,
                        );
                        matmul_bf16_bytes(
                            &mut a,
                            x,
                            std::slice::from_raw_parts(gm.wa, vheads * c.hidden * 2),
                            vheads,
                            c.hidden,
                        );
                        matmul_bf16_bytes(
                            &mut b,
                            x,
                            std::slice::from_raw_parts(gm.wb, vheads * c.hidden * 2),
                            vheads,
                            c.hidden,
                        );
                    }
                }
            } else {
                let fused_input = std::env::var("QWEN_GDN_FUSED_INPUT")
                    .map(|v| v != "0")
                    .unwrap_or(true);
                let fused_ok = if fused_input {
                    let mut ys: [&mut [f32]; 4] = [&mut qkv, &mut a, &mut b, &mut z];
                    let ws = [
                        &layer.gdn_in_qkv,
                        &layer.gdn_in_a,
                        &layer.gdn_in_b,
                        &layer.gdn_in_z,
                    ];
                    matmul_mxfp4_multi(&mut ys, x, &ws)
                } else {
                    false
                };
                if !fused_ok {
                    matmul(&mut qkv, x, &layer.gdn_in_qkv);
                    matmul(&mut a, x, &layer.gdn_in_a);
                    matmul(&mut b, x, &layer.gdn_in_b);
                    matmul(&mut z, x, &layer.gdn_in_z);
                }
            }
        }
        if let Some(t0) = gdn_in_t0 {
            self.spans.gdn_in_proj_ms += t0.elapsed().as_secs_f64() * 1e3;
        }

        let gdn_conv_t0 = profile_gdn_parts.then(std::time::Instant::now);
        let mut y = vec![0.0; cdim];

        // Heterogeneous fused front-half: ANE produced qkv/z/a/b into
        // IOSurfaces; Metal consumes the SAME qkv IOSurface for causal
        // Conv1D+SiLU. The host conv state is advanced only after successful
        // GPU completion, so a declined Metal continuation can fall through
        // to the exact scalar path below without repairing state.
        let ane_conv_ok = if ane_ok {
            let conv_len = cdim * kk.saturating_sub(1);
            let (ane_states, metal_layers, cpu_conv) =
                (&mut self.gdn_ane, &mut self.gdn_metal, &mut self.gdn_conv);
            let conv_st: &mut [f32] = if conv_len == 0 {
                &mut []
            } else if let Some(gm) = metal_layers[li].as_mut() {
                unsafe { std::slice::from_raw_parts_mut(gm.conv_state, conv_len) }
            } else {
                &mut cpu_conv[li]
            };
            gdn_ane::try_conv_silu(&mut ane_states[li], li, &qkv, conv_st, &mut y)
        } else {
            false
        };

        if !ane_conv_ok {
            if kk > 1 {
                // build_gdn_metal() is also the single-copy BF16 re-home on Apple,
                // even when Metal GDN execution is disabled. Use its aligned conv
                // state directly so CPU decode does not maintain/copy a second
                // mirror every token.
                let conv_st: &mut [f32] = if let Some(gm) = self.gdn_metal[li].as_mut() {
                    unsafe { std::slice::from_raw_parts_mut(gm.conv_state, cdim * (kk - 1)) }
                } else {
                    &mut self.gdn_conv[li]
                };
                for ch in 0..cdim {
                    let mut acc = 0.0_f32;
                    for j in 0..kk {
                        let vv = if j == kk - 1 {
                            qkv[ch]
                        } else {
                            conv_st[ch * (kk - 1) + j]
                        };
                        acc += layer.gdn_conv1d[ch * kk + j] * vv;
                    }
                    y[ch] = silu(acc);
                }
                for ch in 0..cdim {
                    for s in 0..kk - 2 {
                        conv_st[ch * (kk - 1) + s] = conv_st[ch * (kk - 1) + s + 1];
                    }
                    conv_st[ch * (kk - 1) + (kk - 2)] = qkv[ch];
                }
            } else {
                for ch in 0..cdim {
                    y[ch] = silu(layer.gdn_conv1d[ch] * qkv[ch]);
                }
            }
        }
        if let Some(t0) = gdn_conv_t0 {
            self.spans.gdn_conv_ms += t0.elapsed().as_secs_f64() * 1e3;
        }

        let gdn_prepare_t0 = profile_gdn_parts.then(std::time::Instant::now);
        let q_ = &y[..kdim];
        let k_ = &y[kdim..kdim * 2];
        let v_ = &y[kdim * 2..];
        let rep = vheads / kheads;
        assert!(rep >= 1 && vheads % kheads == 0);

        let mut qh = vec![0.0; vheads * kd];
        let mut kh = vec![0.0; vheads * kd];
        let mut vh = vec![0.0; vheads * vd];
        for h in 0..vheads {
            let khd = if self.gdn_v_tiled { h % kheads } else { h / rep };
            for d in 0..kd {
                qh[h * kd + d] = q_[khd * kd + d];
                kh[h * kd + d] = k_[khd * kd + d];
            }
            for d in 0..vd {
                vh[h * vd + d] = v_[h * vd + d];
            }
            l2norm(&mut qh[h * kd..h * kd + kd]);
            l2norm(&mut kh[h * kd..h * kd + kd]);
            let sc = 1.0 / (kd as f32).sqrt();
            for d in 0..kd {
                qh[h * kd + d] *= sc;
            }
        }
        if let Some(t0) = gdn_prepare_t0 {
            self.spans.gdn_prepare_ms += t0.elapsed().as_secs_f64() * 1e3;
        }

        let gdn_recur_t0 = profile_gdn_parts.then(std::time::Instant::now);
        let state_len = vheads * kd * vd;
        {
            // Same recurrence and reduction order as the previous snew path,
            // but update the authoritative state in place. For Flash-Next this
            // removes a ~3 MiB zeroed temporary plus a ~3 MiB final copy for
            // each of 36 GDN layers on every token.
            let s: &mut [f32] = if let Some(gm) = self.gdn_metal[li].as_mut() {
                unsafe { std::slice::from_raw_parts_mut(gm.state, state_len) }
            } else {
                &mut self.gdn_s[li]
            };
            let mut kv_mem = vec![0.0; vd];
            for h in 0..vheads {
                let ga =
                    -layer.gdn_a_log[h].exp() * (1.0 + (a[h] + layer.gdn_dt_bias[h]).exp()).ln();
                let gt = ga.exp();
                let bt = 1.0 / (1.0 + (-b[h]).exp());
                let sh = &mut s[h * kd * vd..(h + 1) * kd * vd];
                let qhh = &qh[h * kd..(h + 1) * kd];
                let khh = &kh[h * kd..(h + 1) * kd];
                let vhh = &vh[h * vd..(h + 1) * vd];
                for d in 0..vd {
                    kv_mem[d] = 0.0;
                }
                for kk2 in 0..kd {
                    for d in 0..vd {
                        let si = kk2 * vd + d;
                        let sv = sh[si] * gt;
                        sh[si] = sv;
                        kv_mem[d] += sv * khh[kk2];
                    }
                }
                for d in 0..vd {
                    let delta = (vhh[d] - kv_mem[d]) * bt;
                    let mut acc = 0.0_f32;
                    for kk2 in 0..kd {
                        let si = kk2 * vd + d;
                        let next_s = sh[si] + khh[kk2] * delta;
                        sh[si] = next_s;
                        acc += next_s * qhh[kk2];
                    }
                    kv_mem[d] = acc;
                }
                for d in 0..vd {
                    vh[h * vd + d] = kv_mem[d];
                }
            }
        }
        if let Some(t0) = gdn_recur_t0 {
            self.spans.gdn_recur_ms += t0.elapsed().as_secs_f64() * 1e3;
        }

        let gdn_gate_t0 = profile_gdn_parts.then(std::time::Instant::now);
        let mut normed = vec![0.0; vdim];
        for h in 0..vheads {
            rmsnorm_gated_row(
                &mut normed[h * vd..h * vd + vd],
                &vh[h * vd..h * vd + vd],
                &z[h * vd..h * vd + vd],
                &layer.gdn_norm,
                c.eps,
                c.output_gate,
            );
        }
        if let Some(t0) = gdn_gate_t0 {
            self.spans.gdn_gate_ms += t0.elapsed().as_secs_f64() * 1e3;
        }

        let gdn_out_t0 = profile_gdn_parts.then(std::time::Instant::now);
        if let Some(gm) = self.gdn_metal[li].as_mut().filter(|gm| gm.bf16_weights) {
            let metal_bf16 = std::env::var("QWEN_GDN_BF16_METAL")
                .map(|v| v != "0")
                .unwrap_or(false);
            let wout = unsafe {
                std::slice::from_raw_parts(gm.wout, c.hidden * vdim * 2)
            };
            let mut tensor = gm.tout;
            let metal_ok = metal_bf16
                && logan_metal::metal_matmul(
                    &mut tensor,
                    out,
                    &normed,
                    wout,
                    &[],
                    5,
                    vdim,
                    c.hidden,
                );
            gm.tout = tensor;
            if !metal_ok {
                matmul_bf16_bytes(out, &normed, wout, c.hidden, vdim);
            }
        } else {
            matmul(out, &normed, &layer.gdn_out);
        }
        if let Some(t0) = gdn_out_t0 {
            self.spans.gdn_out_proj_ms += t0.elapsed().as_secs_f64() * 1e3;
        }
    }

    /// Prefill-only BF16 attention projection batch. The projections are pure
    /// functions of each row, so batching them cannot expose future KV or QSA
    /// index state; those causal updates remain in `attention_common` /
    /// `qsa_select` and are executed chronologically afterward.
    fn project_attention_chunk(
        &mut self,
        layer: &Layer,
        xs: &[Vec<f32>],
    ) -> Option<Vec<AttnProjection>> {
        let rows = xs.len();
        if rows <= 1 || layer.is_gdn {
            return None;
        }
        let enabled = std::env::var("QWEN_PREFILL_ATTN_BATCH")
            .map(|v| v != "0")
            .unwrap_or(true);
        if !enabled {
            return None;
        }
        let c = self.cfg.clone();
        let h = c.heads;
        let hd = c.head_dim;
        let kv = c.kv_heads;
        let rows_q = 2 * h * hd;
        let rows_kv = kv * hd;
        let q_bytes = layer.attn_q.bf16_bytes()?;
        let k_bytes = layer.attn_k.bf16_bytes()?;
        let v_bytes = layer.attn_v.bf16_bytes()?;
        if xs.iter().any(|x| x.len() != c.hidden) {
            return None;
        }
        let mut x_all = Vec::with_capacity(rows * c.hidden);
        for x in xs {
            x_all.extend_from_slice(x);
        }
        let mut q_all = vec![0.0_f32; rows * rows_q];
        let mut k_all = vec![0.0_f32; rows * rows_kv];
        let mut v_all = vec![0.0_f32; rows * rows_kv];
        if !crate::ffi::bnns_bf16_matmul_batch(q_bytes, &x_all, &mut q_all, rows, rows_q, c.hidden)
            || !crate::ffi::bnns_bf16_matmul_batch(
                k_bytes, &x_all, &mut k_all, rows, rows_kv, c.hidden,
            )
            || !crate::ffi::bnns_bf16_matmul_batch(
                v_bytes, &x_all, &mut v_all, rows, rows_kv, c.hidden,
            )
        {
            return None;
        }

        let index_rows = if layer.is_qsa {
            (c.idx_n_heads + c.idx_kv_heads) * c.idx_head_dim
        } else {
            0
        };
        let mut index_all = if index_rows > 0 {
            let bytes = layer.index_qk.bf16_bytes()?;
            let mut out = vec![0.0_f32; rows * index_rows];
            if !crate::ffi::bnns_bf16_matmul_batch(
                bytes, &x_all, &mut out, rows, index_rows, c.hidden,
            ) {
                return None;
            }
            Some(out)
        } else {
            None
        };

        let mut projected = Vec::with_capacity(rows);
        for row in 0..rows {
            projected.push(AttnProjection {
                qg: q_all[row * rows_q..(row + 1) * rows_q].to_vec(),
                k: k_all[row * rows_kv..(row + 1) * rows_kv].to_vec(),
                v: v_all[row * rows_kv..(row + 1) * rows_kv].to_vec(),
                index_qk: index_all
                    .as_mut()
                    .map(|all| all[row * index_rows..(row + 1) * index_rows].to_vec()),
                // This flag means the output projection is Metal-resident;
                // batched BNNS input projections intentionally leave it false.
                metal_ok: false,
            });
        }
        Some(projected)
    }

    /// Input-side attention projection. QSA can append index_qk to the packed
    /// Metal projection because all four matrices consume the same x.
    fn project_attention(
        &mut self,
        layer: &Layer,
        li: usize,
        x: &[f32],
        include_index: bool,
    ) -> AttnProjection {
        let c = self.cfg.clone();
        let h = c.heads;
        let hd = c.head_dim;
        let kv = c.kv_heads;

        let rows_q = 2 * h * hd;
        let rows_kv = kv * hd;
        let base_rows = rows_q + 2 * rows_kv;
        let index_rows = if include_index && layer.is_qsa {
            (c.idx_n_heads + c.idx_kv_heads) * c.idx_head_dim
        } else {
            0
        };

        let mut qg = vec![0.0; rows_q];
        let mut k = vec![0.0; rows_kv];
        let mut v = vec![0.0; rows_kv];
        let mut index_qk = None;
        let mut metal_ok = false;

        let metal_enabled = std::env::var("QWEN_ATTN_METAL")
            .map(|v| v != "0")
            .unwrap_or(true);

        if metal_enabled
            && self.attn_metal[li].is_none()
            && !layer.is_gdn
            && crate::ffi::direct_available()
        {
            self.attn_metal[li] = Self::build_attn_metal(layer, &self.cfg);
        }

        if let Some(am) = self.attn_metal[li].as_ref() {
            if metal_enabled {
                // Only include index rows when this QSA layer actually has
                // them packed. Otherwise retain Metal QKV and calculate the
                // index projection on the CPU below.
                let fused_index_rows = if index_rows > 0 && am.qkv_rows >= base_rows + index_rows {
                    index_rows
                } else {
                    0
                };
                let total_rows = base_rows + fused_index_rows;

                // SAFETY: AttnMetalLayer owns this aligned allocation for the
                // model lifetime.
                let weights = unsafe {
                    std::slice::from_raw_parts(am.qkv as *const u8, total_rows * c.hidden * 2)
                };

                let mut proj = vec![0.0; total_rows];
                let rc = crate::ffi::bf16_matmul(weights, x, &mut proj, 1, total_rows, c.hidden);

                if rc > 0 {
                    qg.copy_from_slice(&proj[..rows_q]);
                    k.copy_from_slice(&proj[rows_q..rows_q + rows_kv]);
                    v.copy_from_slice(&proj[rows_q + rows_kv..rows_q + 2 * rows_kv]);
                    if fused_index_rows > 0 {
                        index_qk = Some(proj[base_rows..].to_vec());
                    }
                    metal_ok = true;
                } else if rc < 0 {
                    eprintln!("qwen4-rs: Metal attention failed after submission (layer {li})");
                    std::process::exit(1);
                }
            }
        }

        if !metal_ok {
            // Default-on for MXFP4 decode: Q/K/V share one activation and
            // one Metal command buffer. The helper declines non-MXFP4 weights,
            // preserving the canonical path. Measured ~17% lower attention
            // span and a repeatable end-to-end win on the 16 GiB M2 workload.
            let fused_input = std::env::var("QWEN_ATTN_FUSED_INPUT")
                .map(|v| v != "0")
                .unwrap_or(true);
            // Qwen4's QSA index projection consumes the same activation.
            // Qualify this four-projection variant separately from Qwen3.6.
            let fused_index = index_rows > 0 && std::env::var("QWEN_QSA_FUSED_INPUT")
                .map(|v| v != "0").unwrap_or(false);
            let mut fused_ok = false;
            if fused_input && fused_index {
                let mut qk = vec![0.0; index_rows];
                let mut ys: [&mut [f32]; 4] = [&mut qg, &mut k, &mut v, &mut qk];
                let ws = [&layer.attn_q, &layer.attn_k, &layer.attn_v, &layer.index_qk];
                fused_ok = matmul_mxfp4_multi(&mut ys, x, &ws);
                if fused_ok { index_qk = Some(qk); }
            }
            if fused_input && !fused_ok {
                let mut ys: [&mut [f32]; 3] = [&mut qg, &mut k, &mut v];
                let ws = [&layer.attn_q, &layer.attn_k, &layer.attn_v];
                fused_ok = matmul_mxfp4_multi(&mut ys, x, &ws);
            }
            if !fused_ok {
                matmul(&mut qg, x, &layer.attn_q);
                matmul(&mut k, x, &layer.attn_k);
                matmul(&mut v, x, &layer.attn_v);
            }
        }

        // If QKV ran on Metal but index fusion was unavailable, preserve the
        // exact existing CPU indexer path rather than disabling Metal QKV.
        if include_index && layer.is_qsa && index_qk.is_none() {
            let mut qk = vec![0.0; (c.idx_n_heads + c.idx_kv_heads) * c.idx_head_dim];
            matmul(&mut qk, x, &layer.index_qk);
            index_qk = Some(qk);
        }

        AttnProjection {
            qg,
            k,
            v,
            index_qk,
            metal_ok,
        }
    }

    fn attention_common(
        &mut self,
        layer: &Layer,
        li: usize,
        x: &[f32],
        pos: usize,
        rope: &[(f32, f32)],
        selected: Option<&[usize]>,
        projected: Option<AttnProjection>,
        out: &mut [f32],
    ) {
        let c = self.cfg.clone();
        let h = c.heads;
        let hd = c.head_dim;
        let kv = c.kv_heads;
        let groups = h / kv;

        let AttnProjection {
            mut qg,
            mut k,
            v: vv,
            index_qk: _,
            metal_ok,
        } = projected.unwrap_or_else(|| self.project_attention(layer, li, x, false));
        let qg_snap = qg.clone();
        for hh in 0..h {
            let out = &mut qg[hh * 2 * hd..hh * 2 * hd + hd];
            let input = &qg_snap[hh * 2 * hd..hh * 2 * hd + hd];
            if c.hc_count == 0 {
                // MLX Qwen3.5/3.6 converted checkpoints already have the
                // Transformers `+1` folded into q_norm/k_norm weights.
                rmsnorm_row_shifted(out, input, &layer.attn_qn, c.eps);
            } else {
                rmsnorm_row(out, input, &layer.attn_qn, c.eps);
            }
        }
        let k_snap = k.clone();
        for g in 0..kv {
            let out = &mut k[g * hd..g * hd + hd];
            let input = &k_snap[g * hd..g * hd + hd];
            if c.hc_count == 0 {
                rmsnorm_row_shifted(out, input, &layer.attn_kn, c.eps);
            } else {
                rmsnorm_row(out, input, &layer.attn_kn, c.eps);
            }
        }
        for hh in 0..h {
            rope_partial_with_angles(
                &mut qg[hh * 2 * hd..hh * 2 * hd + 2 * hd],
                rope,
                c.rotary_dim,
                self.rope_interleaved,
            );
        }
        for g in 0..kv {
            rope_partial_with_angles(
                &mut k[g * hd..g * hd + hd],
                rope,
                c.rotary_dim,
                self.rope_interleaved,
            );
        }
        debug_assert_eq!(self.kv_k[li].len(), kv * c.max_t * hd);
        debug_assert_eq!(self.kv_v[li].len(), kv * c.max_t * hd);
        for g in 0..kv {
            let base = g * c.max_t * hd + pos * hd;
            self.kv_k[li][base..base + hd].copy_from_slice(&k[g * hd..g * hd + hd]);
            self.kv_v[li][base..base + hd].copy_from_slice(&vv[g * hd..g * hd + hd]);
        }

        let positions: Vec<usize> = match selected {
            Some(sel) => sel.to_vec(),
            None => (0..=pos).collect(),
        };
        let nsel = positions.len();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut scores = vec![0.0; nsel];
        let mut attn_out = vec![0.0; h * hd];
        for hh in 0..h {
            let qh = &qg[hh * 2 * hd..hh * 2 * hd + hd];
            let hg = hh / groups;
            let mut mx = -1e30_f32;
            for (s, &p) in positions.iter().enumerate() {
                let base = hg * c.max_t * hd + p * hd;
                let mut acc = 0.0_f32;
                for dd in 0..hd {
                    acc += qh[dd] * self.kv_k[li][base + dd];
                }
                scores[s] = acc * scale;
                if scores[s] > mx {
                    mx = scores[s];
                }
            }
            let mut ssum = 0.0_f32;
            for s in 0..nsel {
                scores[s] = (scores[s] - mx).exp();
                ssum += scores[s];
            }
            let oh = &mut attn_out[hh * hd..hh * hd + hd];
            for dd in 0..hd {
                oh[dd] = 0.0;
            }
            for (s, &p) in positions.iter().enumerate() {
                let base = hg * c.max_t * hd + p * hd;
                let w = scores[s] / ssum;
                for dd in 0..hd {
                    oh[dd] += w * self.kv_v[li][base + dd];
                }
            }
            let gh = &qg[(2 * hh + 1) * hd..(2 * hh + 2) * hd];
            for dd in 0..hd {
                oh[dd] *= 1.0 / (1.0 + (-gh[dd]).exp());
            }
        }
        if !metal_ok {
            matmul(out, &attn_out, &layer.attn_o);
        } else if let Some(am) = self.attn_metal[li].as_ref() {
            // SAFETY: exact-length view over the aligned o_proj block
            // (kept alive by am._bufs for the model lifetime).
            let wo =
                unsafe { std::slice::from_raw_parts(am.o as *const u8, c.hidden * h * hd * 2) };
            let rc = crate::ffi::bf16_matmul(wo, &attn_out, out, 1, c.hidden, h * hd);
            if rc < 0 {
                eprintln!(
                    "qwen4-rs: Metal attention out_proj failed after submission (layer {li})"
                );
                std::process::exit(1);
            }
            if rc == 0 {
                matmul(out, &attn_out, &layer.attn_o);
            }
        }
    }

    fn attention_token(
        &mut self,
        layer: &Layer,
        li: usize,
        x: &[f32],
        pos: usize,
        rope: &[(f32, f32)],
        out: &mut [f32],
    ) {
        self.attention_common(layer, li, x, pos, rope, None, None, out);
    }

    fn qsa_select(
        &mut self,
        layer: &Layer,
        li: usize,
        x: &[f32],
        pos: usize,
        rope: &[(f32, f32)],
        projected_qk: Option<&[f32]>,
    ) -> Vec<usize> {
        let c = self.cfg.clone();
        let ih = c.idx_head_dim;
        let in_ = c.idx_n_heads;
        let ik = c.idx_kv_heads;
        let ratio = c.idx_ratio;
        let budget = c.idx_budget;
        let nq = ih * in_;
        let nk = ih * ik;

        let qk = match projected_qk {
            Some(qk) => {
                debug_assert_eq!(qk.len(), nq + nk);
                qk.to_vec()
            }
            None => {
                let mut qk = vec![0.0; nq + nk];
                matmul(&mut qk, x, &layer.index_qk);
                qk
            }
        };
        let mut q = qk[..nq].to_vec();
        let q_snap = q.clone();
        for hh in 0..in_ {
            rmsnorm_row(
                &mut q[hh * ih..hh * ih + ih],
                &q_snap[hh * ih..hh * ih + ih],
                &layer.idx_qn,
                c.eps,
            );
        }
        for hh in 0..in_ {
            rope_partial_with_angles(
                &mut q[hh * ih..hh * ih + ih],
                rope,
                c.rotary_dim,
                self.rope_interleaved,
            );
        }
        // store raw indexer k for this position
        let cached = &mut self.idx_cache[li];
        debug_assert_eq!(cached.len(), c.max_t * nk);
        cached[pos * nk..pos * nk + nk].copy_from_slice(&qk[nq..nq + nk]);
        let len = pos + 1;
        let nblk = len / ratio;
        let mut sel: Vec<usize> = Vec::new();
        if nblk > 0 {
            let mut pool = vec![0.0; nblk * ih];
            let mut starts = vec![0_usize; nblk];
            for b in 0..nblk {
                starts[b] = b * ratio;
                for d in 0..ih {
                    let mut acc = 0.0_f64;
                    for r in 0..ratio {
                        acc += cached[(b * ratio + r) * nk + d] as f64;
                    }
                    pool[b * ih + d] = (acc / ratio as f64) as f32;
                }
            }
            let pool_snap = pool.clone();
            for b in 0..nblk {
                rmsnorm_row(
                    &mut pool[b * ih..b * ih + ih],
                    &pool_snap[b * ih..b * ih + ih],
                    &layer.idx_kn,
                    c.eps,
                );
            }
            let pool2 = pool.clone();
            for b in 0..nblk {
                let mut row = pool2[b * ih..b * ih + ih].to_vec();
                rope_partial(&mut row, starts[b], &c, self.rope_interleaved);
                pool[b * ih..b * ih + ih].copy_from_slice(&row);
            }
            let mut topk = budget / ratio;
            if topk > nblk {
                topk = nblk;
            }
            let mut sc = vec![0.0; nblk];
            let mut ord: Vec<usize> = (0..nblk).collect();
            for b in 0..nblk {
                let mut acc = 0.0_f32;
                for hh in 0..in_ {
                    let qh = &q[hh * ih..hh * ih + ih];
                    let kb = &pool[b * ih..b * ih + ih];
                    let mut dot = 0.0_f32;
                    for d in 0..ih {
                        dot += qh[d] * kb[d];
                    }
                    acc += if dot > 0.0 { dot } else { 0.0 };
                }
                sc[b] = acc / (ih as f32).sqrt();
            }
            // selection sort desc, lower index wins ties
            for i in 0..topk {
                let mut best = i;
                for j in i + 1..nblk {
                    if sc[j] > sc[best] || (sc[j] == sc[best] && ord[j] < ord[best]) {
                        best = j;
                    }
                }
                ord.swap(i, best);
                sc.swap(i, best);
            }
            for i in 0..topk {
                for r in 0..ratio {
                    sel.push(starts[ord[i]] + r);
                }
            }
        }
        for p in nblk * ratio..len {
            sel.push(p);
        }
        sel
    }

    fn sparse_attn_token(
        &mut self,
        layer: &Layer,
        li: usize,
        x: &[f32],
        pos: usize,
        rope: &[(f32, f32)],
        out: &mut [f32],
    ) {
        let index_metal = std::env::var("QWEN_QSA_INDEX_METAL")
            .map(|v| v != "0")
            .unwrap_or(true);
        let attn_metal = std::env::var("QWEN_ATTN_METAL")
            .map(|v| v != "0")
            .unwrap_or(true);

        if index_metal && attn_metal && crate::ffi::direct_available() {
            let projection = self.project_attention(layer, li, x, true);
            let sel = self.qsa_select(layer, li, x, pos, rope, projection.index_qk.as_deref());
            self.attention_common(layer, li, x, pos, rope, Some(&sel), Some(projection), out);
        } else {
            // Exact pre-fusion ordering for the A/B baseline.
            let sel = self.qsa_select(layer, li, x, pos, rope, None);
            self.attention_common(layer, li, x, pos, rope, Some(&sel), None, out);
        }
    }

    /// FIFO expert cache. On miss, stream the expert's three raw Apple8
    /// matrices async (MetalIO) into ONE slot with the C engine's layout
    /// (gate at 0, up at align16, down at align16(up_end)); the slot IS the
    /// cache unit — the fused moe_topk consumes it in native tile order, the
    /// CPU fallback preads its shared-storage bytes. One MTLIOFileHandle per
    /// shard (cached process-wide; the C table caps at 64). Eviction frees
    /// the slot (its Drop waits + releases).
    /// Phase 1 (async issue): ensure the expert is resident, enqueuing the
    /// MetalIO load WITHOUT waiting. Returns a borrowed view; the caller
    /// must drain the pending event via `expert_wait` before consuming the
    /// slot bytes (this is what lets the C engine overlap all K loads of a
    /// layer — the serialization fix).
    fn cached_expert_issue(
        &mut self,
        li: i32,
        ei: i32,
        speculative: bool,
    ) -> Option<std::rc::Rc<crate::colisource::SlotRef>> {
        // Demand hits promote/count; speculative probes deliberately do not
        // perturb hit-rate telemetry or recency when the expert is resident.
        if speculative {
            if let Some(v) = self.expert_store.peek((li as u32, ei as u32)) {
                return Some(std::rc::Rc::new(v.ref_view()));
            }
        } else if let Some(v) = self.expert_store.get((li as u32, ei as u32)) {
            return Some(std::rc::Rc::new(v.ref_view()));
        }
        let coli = self.coli.as_ref()?;
        let planned = self
            .expert_plan
            .as_ref()
            .and_then(|plan| plan.layers.get(li as usize))
            .and_then(|layer| layer.get(ei as usize))
            .cloned();
        let se: Option<crate::colisource::SlotExpert> = (|| {
            let (shard_id, regions, dims) = if let Some(planned) = planned.as_ref() {
                (
                    planned.shard_id,
                    [planned.regions[0], planned.regions[1], planned.regions[2]],
                    [planned.dims[0], planned.dims[1], planned.dims[2]],
                )
            } else {
                // Exact legacy lazy-descriptor path, retained for A/B and
                // non-preplanned packages.
                let recs = coli.pkg_ref().expert_records(li, ei);
                let rec = recs.first()?;
                let (regions, dims) = coli.pkg_ref().expert_matrix_regions(rec)?;
                if regions.len() < 3 || dims.len() < 3 {
                    return None;
                }
                (
                    rec.shard_id,
                    [regions[0], regions[1], regions[2]],
                    [dims[0], dims[1], dims[2]],
                )
            };
            let shard = coli.pkg_ref().shard_path(shard_id)?;
            let fid = crate::ffi::mio_file(&shard)?;
            let (slot, ev) = if speculative {
                crate::ffi::mio_prefetch_expert(fid, &regions)?
            } else {
                crate::ffi::mio_load_expert(fid, &regions)?
            };
            let ptr = unsafe { crate::ffi::metalio_slot_ptr(slot) } as *mut u8;
            if ptr.is_null() {
                unsafe { crate::ffi::metalio_slot_free(slot) };
                return None;
            }
            let gb = regions[0].1;
            let ub = regions[1].1;
            let db = regions[2].1;
            let up_off = (gb + 15) & !15usize;
            let down_off = (up_off + ub + 15) & !15usize;
            Some(crate::colisource::SlotExpert {
                slot,
                gate_bytes: gb,
                up_offset: up_off,
                up_bytes: ub,
                down_offset: down_off,
                down_bytes: db,
                ptr,
                pending: std::cell::Cell::new(ev),
                bf16_cache: std::cell::RefCell::new(None),
                rows: [dims[0].0, dims[1].0, dims[2].0],
                cols: [dims[0].1, dims[1].1, dims[2].1],
            })
        })();
        let se = se?;
        // LRU insert; the returned ref is the fresh value (no hit bump —
        // hits measure genuine reuse only). Evicted slot released by drop.
        let (mut evicted, v) = self.expert_store.insert((li as u32, ei as u32), se);
        if let Some(mut e) = evicted.take() {
            e.release();
        }
        Some(std::rc::Rc::new(v.ref_view()))
    }

    /// Issue the previous token's route early while the temporal block runs.
    /// With 8/layer residency this is normally a zero-I/O probe (the whole
    /// previous route is retained); it remains useful as an opt-in policy for
    /// smaller/global caches and records speculative MetalIO separately.
    fn prefetch_previous_route_now(&mut self, li: usize) {
        if self.sched_mode {
            return;
        }
        let previous = self.route_prev[li].clone();
        for ei in previous {
            let _ = self.cached_expert_issue(li as i32, ei as i32, true);
        }
    }

    fn prefetch_previous_route(&mut self, li: usize) {
        if !std::env::var("QWEN_PREV_ROUTE_PREFETCH")
            .map(|v| v != "0")
            .unwrap_or(false)
        {
            return;
        }
        self.prefetch_previous_route_now(li);
    }

    /// Phase 2 (drain): wait for a previously-issued expert's MetalIO event.
    /// Uses `peek` (not `get`) so the drain does NOT count as a cache hit —
    /// hits measure genuine reuse only. Returns false on I/O failure
    /// (caller falls back to the CPU path).
    fn expert_wait(&mut self, li: i32, ei: i32) -> bool {
        let ev = match self.expert_store.peek((li as u32, ei as u32)) {
            Some(v) => v.pending.get(),
            None => return true, // nothing pending
        };
        if ev == 0 {
            return true; // already resident
        }
        if unsafe { crate::ffi::metalio_wait(ev) } != 0 {
            return false;
        }
        if let Some(v) = self.expert_store.peek((li as u32, ei as u32)) {
            v.pending.set(0);
        }
        true
    }

    /// Prefill-only layer warmup: issue the union of routed experts for the
    /// current layer/chunk, then retire them behind one MetalIO completion
    /// point. This changes only load timing; the existing per-row fused MoE
    /// kernel still consumes experts in each row's canonical top-k order.
    fn preload_expert_set(&mut self, li: usize, experts: &[usize]) -> bool {
        if self.coli.is_none() || experts.is_empty() || experts.len() > cache_cap() {
            return false;
        }
        let mut refs: Vec<std::rc::Rc<crate::colisource::SlotRef>> =
            Vec::with_capacity(experts.len());
        for &ei in experts {
            match self.cached_expert_issue(li as i32, ei as i32, false) {
                Some(r) => refs.push(r),
                None => return false,
            }
        }
        let has_pending = experts.iter().any(|&ei| {
            self.expert_store
                .peek((li as u32, ei as u32))
                .is_some_and(|v| v.pending.get() != 0)
        });
        if !has_pending {
            return true;
        }
        if let Some(event) = crate::ffi::mio_batch_barrier() {
            let slots: Vec<i32> = refs.iter().map(|r| r.slot).collect();
            if !crate::ffi::mio_batch_wait(event, &slots) {
                return false;
            }
            for &ei in experts {
                if let Some(v) = self.expert_store.peek((li as u32, ei as u32)) {
                    v.pending.set(0);
                }
            }
            true
        } else {
            experts
                .iter()
                .all(|&ei| self.expert_wait(li as i32, ei as i32))
        }
    }

    /// Synchronous variant (CPU fallback paths): issue then drain.
    fn cached_expert_await(
        &mut self,
        li: i32,
        ei: i32,
    ) -> Option<std::rc::Rc<crate::colisource::SlotRef>> {
        let r = self.cached_expert_issue(li, ei, false)?;
        if !self.expert_wait(li, ei) {
            return None;
        }
        Some(r)
    }

    /// Direct-path expert descriptor for the fused moe_topk (slot + offsets).
    fn slot_descriptor(se: &crate::colisource::SlotRef) -> crate::ffi::ColiApple8MetalioExpert {
        crate::ffi::ColiApple8MetalioExpert {
            slot: se.slot,
            gate_offset: 0,
            gate_bytes: se.gate_bytes,
            up_offset: se.up_offset,
            up_bytes: se.up_bytes,
            down_offset: se.down_offset,
            down_bytes: se.down_bytes,
        }
    }

    /// CPU fallback for one slot-resident expert: decode the shared-storage
    /// tile bytes (lazily cached per expert) and run the BF16 matmuls.
    fn slot_expert_cpu(
        se: &crate::colisource::SlotRef,
        x: &[f32],
        gate: &mut [f32],
        up: &mut [f32],
        y: &mut [f32],
    ) {
        // SAFETY: slot bytes are CPU-visible shared storage, valid while the
        // cache entry owns the slot (ref_view borrows a live entry; eviction
        // only happens inside cached_expert while no ref is outstanding).
        let total = se.down_offset + se.down_bytes;
        let raw = unsafe { std::slice::from_raw_parts(se.ptr, total) };
        let parts = [
            &raw[0..se.gate_bytes],
            &raw[se.up_offset..se.up_offset + se.up_bytes],
            &raw[se.down_offset..se.down_offset + se.down_bytes],
        ];
        let mut cb: [Vec<u8>; 3] = Default::default();
        for (pi, p) in parts.iter().enumerate() {
            let f = logan_format::codecs::apple8_mxfp4_decode(
                p,
                se.rows[pi] as u64,
                se.cols[pi] as u64,
            )
            .unwrap_or_default();
            cb[pi] = f
                .into_iter()
                .flat_map(crate::colisource::bf16_bytes)
                .collect();
        }
        let [gb, ub, db] = cb;
        let g = Wt {
            f: vec![],
            bytes: Some(WtBytes::Bf16 {
                weights: gb,
                metal_tensor: std::sync::Mutex::new(0),
            }),
            o: se.rows[0],
            i: se.cols[0],
        };
        let u = Wt {
            f: vec![],
            bytes: Some(WtBytes::Bf16 {
                weights: ub,
                metal_tensor: std::sync::Mutex::new(0),
            }),
            o: se.rows[1],
            i: se.cols[1],
        };
        let dw = Wt {
            f: vec![],
            bytes: Some(WtBytes::Bf16 {
                weights: db,
                metal_tensor: std::sync::Mutex::new(0),
            }),
            o: se.rows[2],
            i: se.cols[2],
        };
        matmul(gate, x, &g);
        matmul(up, x, &u);
        let h: Vec<f32> = (0..gate.len()).map(|ii| silu(gate[ii]) * up[ii]).collect();
        matmul(y, &h, &dw);
    }

    fn shared_expert_value(&self, layer: &Layer, li: usize, x: &[f32]) -> (Vec<f32>, f32) {
        // The checkpoint keeps the tiny scalar shared-expert gate separate
        // from the three MXFP4 shared-MLP matrices. Preserve that scalar math
        // independently, then fuse gate_proj + up_proj + SwiGLU + down_proj
        // into one Metal command buffer when the large matrices are MXFP4.
        let mut sg = vec![0.0; 1];
        matmul(&mut sg, x, &layer.se_g);
        let gs = 1.0 / (1.0 + (-sg[0]).exp());

        let full_mxfp4 = std::env::var("QWEN_SHARED_MXFP4_FULL")
            .map(|v| v != "0")
            .unwrap_or(true);
        if full_mxfp4 {
            let ws = [&layer.se_gate, &layer.se_up, &layer.se_down];
            let mut parts = Vec::with_capacity(3);
            let mut all_mx = true;
            for &w in &ws {
                if let Some(WtBytes::Mxfp4 {
                    weights,
                    scales,
                    metal_tensor,
                }) = w.bytes.as_ref()
                {
                    parts.push((
                        weights.as_slice(),
                        scales.as_slice(),
                        metal_tensor,
                        w.i,
                        w.o,
                    ));
                } else {
                    all_mx = false;
                    break;
                }
            }
            if all_mx {
                let mut guards = Vec::with_capacity(3);
                for (_, _, metal_tensor, _, _) in &parts {
                    guards.push(
                        metal_tensor
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner()),
                    );
                }
                let mut descs = Vec::with_capacity(3);
                for (part, guard) in parts.iter().zip(guards.iter()) {
                    let (weights, scales, _, input, output) = *part;
                    descs.push(logan_metal::MetalWeightDesc {
                        tensor: **guard as *mut logan_metal::ColiMetalTensor,
                        weights,
                        scales,
                        fmt: 7,
                        i: input,
                        o: output,
                    });
                }
                let mut sy = vec![0.0f32; self.cfg.hidden];
                match logan_metal::shared_mxfp4(
                    self.metal_model_id,
                    li,
                    &mut descs,
                    x,
                    &mut sy,
                    self.cfg.hidden,
                    self.cfg.shared_inter,
                ) {
                    Ok(Some(())) => {
                        for (guard, desc) in guards.iter_mut().zip(descs.iter()) {
                            **guard = desc.tensor as usize;
                        }
                        return (sy, gs);
                    }
                    Ok(None) => {
                        for (guard, desc) in guards.iter_mut().zip(descs.iter()) {
                            **guard = desc.tensor as usize;
                        }
                    }
                    Err(()) => {
                        eprintln!("qwen4-rs: full MXFP4 shared expert failed after submission (layer {li})");
                        std::process::exit(1);
                    }
                }
            }
        }
        let c = &self.cfg;
        let d = c.hidden;
        let mut gv = vec![0.0; c.shared_inter];
        let mut h = vec![0.0; c.shared_inter];
        // Gate/up depend on the same activation; encode both MXFP4 GEMVs
        // into one Metal command buffer instead of synchronizing twice.
        // The helper declines non-MXFP4 weights. Default-on after paired and
        // reversed-order real-model A/Bs showed a repeatable wall-time win.
        let fused_input = std::env::var("QWEN_SHARED_FUSED_INPUT")
            .map(|v| v != "0")
            .unwrap_or(true);
        let fused_ok = if fused_input {
            let mut ys: [&mut [f32]; 2] = [&mut gv, &mut h];
            let ws = [&layer.se_gate, &layer.se_up];
            matmul_mxfp4_multi(&mut ys, x, &ws)
        } else {
            false
        };
        if !fused_ok {
            matmul(&mut gv, x, &layer.se_gate);
            matmul(&mut h, x, &layer.se_up);
        }
        for i in 0..c.shared_inter {
            h[i] = silu(gv[i]) * h[i];
        }
        let mut sy = vec![0.0; d];
        matmul(&mut sy, &h, &layer.se_down);
        (sy, gs)
    }

    fn route_topk(&mut self, layer: &Layer, x: &[f32]) -> (Vec<usize>, Vec<f32>, f32) {
        let c = self.cfg.clone();
        let e = c.experts;
        let k = c.topk;
        let mut _route_t = logan_core::telemetry::Span::begin("route");
        let mut logits = vec![0.0; e];
        matmul(&mut logits, x, &layer.router);
        softmax_row(&mut logits);

        let mut idx: Vec<usize> = (0..e).collect();
        let mut val = logits;
        for i in 0..k {
            let mut best = i;
            for j in i + 1..e {
                if val[j] > val[best] || (val[j] == val[best] && idx[j] < idx[best]) {
                    best = j;
                }
            }
            idx.swap(i, best);
            val.swap(i, best);
        }
        let wsum: f32 = val[..k].iter().sum();
        self.spans.route_ms += _route_t.end();
        (idx, val, wsum)
    }

    fn moe_token(&mut self, layer: &Layer, li: usize, x: &[f32], out: &mut [f32]) {
        let (idx, val, wsum) = self.route_topk(layer, x);
        self.moe_token_routed(layer, li, x, out, &idx, &val, wsum);
    }

    fn moe_token_routed(
        &mut self,
        layer: &Layer,
        li: usize,
        x: &[f32],
        out: &mut [f32],
        idx: &[usize],
        val: &[f32],
        wsum: f32,
    ) {
        let c = self.cfg.clone();
        let k = c.topk;
        let d = c.hidden;

        let current_route = &idx[..k.min(idx.len())];
        if std::env::var("QWEN_ROUTE_OVERLAP")
            .map(|v| v != "0")
            .unwrap_or(false)
        {
            let previous = &self.route_prev[li];
            if previous.len() == current_route.len() && !previous.is_empty() {
                let common = current_route
                    .iter()
                    .filter(|&&e| previous.contains(&e))
                    .count() as u64;
                self.route_overlap_common[li] += common;
                self.route_overlap_total[li] += current_route.len() as u64;
                self.route_overlap_pairs[li] += 1;
            }
        }
        self.route_prev[li].clear();
        self.route_prev[li].extend_from_slice(current_route);

        // Scheduler-driven mode (issue #53): the model never issues expert
        // loads. If any routed expert of this layer is not resident in the
        // physical store, report the cold set and abort the layer BEFORE any
        // compute, load, or state change. `forward_layer` stashes the resume
        // cursor; the scheduler loads the experts as separate actions and
        // the same op is resubmitted, re-entering this exact MoE phase with
        // resident slots (byte-identical: routing is a pure function of x).
        if self.sched_mode && self.coli.is_some() {
            let cold: Vec<u32> = (0..k)
                .filter(|i| {
                    self.expert_store
                        .peek((li as u32, idx[*i] as u32))
                        .is_none()
                })
                .map(|i| idx[i] as u32)
                .collect();
            if !cold.is_empty() {
                self.sched_blocked = Some(cold);
                return;
            }
        }
        let mut acc = vec![0.0; d];

        // Direct fused path (C QWEN_APPLE8_DIRECT): all K slot-resident
        // experts in ONE command buffer (gate+up+swiglu -> down -> weighted
        // reduce), consumed in top-k order with pre-renormalized weights.
        // Split-phase by default (C QWEN_APPLE8_OVERLAP=1): submit first, run
        // the CPU shared expert while the GPU works, then wait.
        let direct =
            self.metal_direct && crate::ffi::direct_available() && k <= 64 && self.coli.is_some();
        let mut pending: Option<crate::ffi::MoePending> = None;
        let mut direct_done = false;
        let mut fallback_reason: &'static str = if direct { "unknown" } else { "direct-disabled" };
        let mut shared_ready: Option<(Vec<f32>, f32)> = None;
        let shared_io_overlap = std::env::var("QWEN_SHARED_IO_OVERLAP")
            .map(|v| v != "0")
            .unwrap_or(true);
        let mut _io_t = logan_core::telemetry::Span::begin("io");
        if direct {
            // With a per-layer cache exactly equal to top-k, sequential miss
            // insertion can otherwise evict an expert that is already resident
            // and appears later in THIS SAME route. Protect the whole resident
            // route first; `promote_if_present` is telemetry-neutral, so the
            // subsequent `get` calls still count each genuine reuse exactly once.
            let pin_route_hits = std::env::var("QWEN_ROUTE_PIN_HITS")
                .map(|v| v != "0")
                .unwrap_or(true);
            if pin_route_hits {
                for i in 0..k {
                    self.expert_store
                        .promote_if_present((li as u32, idx[i] as u32));
                }
            }

            // Async issue: enqueue ALL K expert loads first (no waits), so
            // MTLIO pipelines them back-to-back instead of serializing each
            // load+wait (the C engine's exact-demand async issue). The
            // awaits below drain the events just before the fused submit.
            let mut refs: Vec<std::rc::Rc<crate::colisource::SlotRef>> = Vec::with_capacity(k);
            let mut all_ok = true;
            for i in 0..k {
                match self.cached_expert_issue(li as i32, idx[i] as i32, false) {
                    Some(ce) => refs.push(ce),
                    None => {
                        all_ok = false;
                        fallback_reason = "expert-issue";
                        break;
                    }
                }
            }
            // Queue one completion point immediately after all cold loads.
            // The IO queue is concurrent, so waiting on the numerically last
            // per-expert event is not sufficient; enqueueBarrier() is. We can
            // then overlap the shared expert with the whole outstanding batch
            // and perform one blocking wait before the fused MoE submit.
            let batch_wait_enabled = std::env::var("QWEN_MIO_BATCH_WAIT")
                .map(|v| v != "0")
                .unwrap_or(true);
            let has_pending = all_ok
                && (0..k).any(|i| {
                    self.expert_store
                        .peek((li as u32, idx[i] as u32))
                        .is_some_and(|v| v.pending.get() != 0)
                });
            let batch_event = if batch_wait_enabled && has_pending {
                crate::ffi::mio_batch_barrier()
            } else {
                None
            };

            // The shared expert depends only on x, not on routed-expert bytes.
            // Run it while MetalIO owns outstanding NVMe->UMA transfers instead of
            // spending the same CPU work after all I/O has already drained.
            if all_ok && shared_io_overlap {
                let mut _shared_t = logan_core::telemetry::Span::begin("shared");
                shared_ready = Some(self.shared_expert_value(layer, li, x));
                self.spans.shared_ms += _shared_t.end();
            }
            if all_ok {
                if let Some(event) = batch_event {
                    let slots: Vec<i32> = refs.iter().map(|r| r.slot).collect();
                    if crate::ffi::mio_batch_wait(event, &slots) {
                        for i in 0..k {
                            if let Some(v) = self.expert_store.peek((li as u32, idx[i] as u32)) {
                                v.pending.set(0);
                            }
                        }
                    } else {
                        all_ok = false;
                        fallback_reason = "batch-wait";
                    }
                } else {
                    for i in 0..k {
                        if !self.expert_wait(li as i32, idx[i] as i32) {
                            all_ok = false;
                            break;
                        }
                    }
                }
            }
            let mut ex: Vec<crate::ffi::ColiApple8MetalioExpert> = Vec::with_capacity(k);
            let mut ws: Vec<f32> = Vec::with_capacity(k);
            if all_ok {
                for (i, ce) in refs.iter().enumerate() {
                    ex.push(Self::slot_descriptor(ce));
                    ws.push(val[i] / wsum);
                }
            }
            if all_ok {
                let mut _gpu_t = logan_core::telemetry::Span::begin("gpu");
                if self.metal_overlap {
                    pending = crate::ffi::moe_topk_begin(&ex, &ws, x, d, c.moe_inter);
                } else if crate::ffi::moe_topk(&ex, &ws, x, &mut acc, d, c.moe_inter) {
                    direct_done = true;
                } else {
                    pending = None; // decline -> CPU per-expert loop below
                }
                self.spans.gpu_ms += _gpu_t.end();
            }
            if pending.is_none() && self.metal_overlap && all_ok {
                // begin() declined mid-run: fall through to the CPU loop
                // (weights unchanged; nothing was submitted).
                fallback_reason = "moe-begin";
            }
        }

        self.spans.io_ms += _io_t.end();
        if direct_done {
            let (sy, gs) = if let Some(v) = shared_ready.take() {
                v
            } else {
                let mut _shared_t = logan_core::telemetry::Span::begin("shared");
                let v = self.shared_expert_value(layer, li, x);
                self.spans.shared_ms += _shared_t.end();
                v
            };
            for dd in 0..d {
                out[dd] = acc[dd] + sy[dd] * gs;
            }
            return;
        }
        if pending.is_some() {
            // With QWEN_SHARED_IO_OVERLAP=1 this result was produced while
            // expert loads were outstanding. Opt-out retains the old GPU-overlap
            // placement for a same-binary A/B.
            let (sy, gs) = if let Some(v) = shared_ready.take() {
                v
            } else {
                let mut _shared_t = logan_core::telemetry::Span::begin("shared");
                let v = self.shared_expert_value(layer, li, x);
                self.spans.shared_ms += _shared_t.end();
                v
            };
            let p = pending.unwrap();
            let mut _gpu_wait = logan_core::telemetry::Span::begin("gpu-wait");
            let gpu_ok = crate::ffi::moe_topk_finish(p, &mut acc);
            self.spans.gpu_ms += _gpu_wait.end();
            if !gpu_ok {
                // GPU fault AFTER submit: C contract = redo those experts on
                // CPU. acc was scratch for the GPU result; recompute routed
                // experts on CPU into a fresh accumulator.
                acc = vec![0.0; d];
                for i in 0..k {
                    let w = val[i] / wsum;
                    if let Some(ce) = self.cached_expert_await(li as i32, idx[i] as i32) {
                        let mut gate = vec![0.0; c.moe_inter];
                        let mut up = vec![0.0; c.moe_inter];
                        let mut y = vec![0.0; d];
                        Self::slot_expert_cpu(&ce, x, &mut gate, &mut up, &mut y);
                        for dd in 0..d {
                            acc[dd] += y[dd] * w;
                        }
                    }
                }
            }
            for dd in 0..d {
                out[dd] = acc[dd] + sy[dd] * gs;
            }
            return;
        }

        if std::env::var("QWEN_MOE_FALLBACK_DIAG")
            .map(|v| v != "0")
            .unwrap_or(false)
        {
            eprintln!("[moe-fallback] layer={li} reason={fallback_reason}");
        }
        let mut _fill_t = logan_core::telemetry::Span::begin("fill");
        for i in 0..k {
            let w = val[i] / wsum;
            // .coli mode: slot-resident expert. The Metal fused path ran
            // above (returned early on success); this loop is the CPU
            // fallback (decodes the slot's shared-storage tiles lazily) or
            // the canonical decode path for non-Apple8 packages.
            // safetensors mode: preloaded experts.
            let mats: [Wt; 3] = if self.coli.is_some() {
                match self.cached_expert_await(li as i32, idx[i] as i32) {
                    Some(ce) => {
                        let mut gate = vec![0.0; c.moe_inter];
                        let mut up = vec![0.0; c.moe_inter];
                        let mut y = vec![0.0; d];
                        Self::slot_expert_cpu(&ce, x, &mut gate, &mut up, &mut y);
                        for dd in 0..d {
                            acc[dd] += y[dd] * w;
                        }
                        continue; // acc updated; skip the common tail
                    }
                    None => {
                        // BF16/INT4 canonical experts: decode path
                        let m = self
                            .coli
                            .as_ref()
                            .unwrap()
                            .expert_matrices(li as i32, idx[i] as i32)
                            .unwrap_or_else(|e| {
                                panic!("expert ({li},{}) fetch failed: {e}", idx[i])
                            });
                        [
                            Wt {
                                f: vec![],
                                bytes: Some(WtBytes::Bf16 {
                                    weights: m[0].bytes.clone(),
                                    metal_tensor: std::sync::Mutex::new(0),
                                }),
                                o: m[0].o,
                                i: m[0].i,
                            },
                            Wt {
                                f: vec![],
                                bytes: Some(WtBytes::Bf16 {
                                    weights: m[1].bytes.clone(),
                                    metal_tensor: std::sync::Mutex::new(0),
                                }),
                                o: m[1].o,
                                i: m[1].i,
                            },
                            Wt {
                                f: vec![],
                                bytes: Some(WtBytes::Bf16 {
                                    weights: m[2].bytes.clone(),
                                    metal_tensor: std::sync::Mutex::new(0),
                                }),
                                o: m[2].o,
                                i: m[2].i,
                            },
                        ]
                    }
                }
            } else {
                self.experts[li][idx[i]].clone()
            };
            let mut gate = vec![0.0; c.moe_inter];
            let mut up = vec![0.0; c.moe_inter];
            matmul(&mut gate, x, &mats[0]);
            matmul(&mut up, x, &mats[1]);
            let mut h = vec![0.0; c.moe_inter];
            for ii in 0..c.moe_inter {
                h[ii] = silu(gate[ii]) * up[ii];
            }
            let mut y = vec![0.0; d];
            matmul(&mut y, &h, &mats[2]);
            for dd in 0..d {
                acc[dd] += y[dd] * w;
            }
        }
        self.spans.fill_ms += _fill_t.end();
        let (sy, gs) = if let Some(v) = shared_ready.take() {
            v
        } else {
            let mut _shared_t = logan_core::telemetry::Span::begin("shared");
            let v = self.shared_expert_value(layer, li, x);
            self.spans.shared_ms += _shared_t.end();
            v
        };
        for dd in 0..d {
            out[dd] = acc[dd] + sy[dd] * gs;
        }
    }

    fn ple_forward(&mut self, stream: &mut [f32]) {
        let c = self.cfg.clone();
        if c.ple_layer < 0 {
            return;
        }
        let d = c.hidden;
        let hc = c.hc_count;
        let hcd = hc * d;
        let ns = c.ngram_size;
        let hpn = c.ngram_heads / (ns - 1);
        let heads = c.ngram_heads;
        let hd_per = c.ple_embed_dim / heads;
        let kk = c.ple_conv_kernel;
        let dil = ns;
        let pad = (kk - 1) * dil;
        let ctx = ns - 1;
        let t = ctx + 1;

        let mut hist = [0_i64; 16];
        for i in 0..ctx {
            hist[i] = self.ple_ring[i];
        }
        hist[ctx] = self.ple_ring[ctx];

        let mut shifted = [[0_i64; 16]; 8];
        for s in 0..ns {
            let mut seg = 0_usize;
            for p in 0..t {
                let mut v = c.eos;
                let src = p as i64 - s as i64;
                if src >= 0 && p - seg >= s {
                    v = hist[src as usize];
                }
                shifted[s][p] = v;
                if hist[p] == c.eos {
                    seg = p + 1;
                }
            }
        }

        let mut rows = [0_u64; 16];
        let mut n = 0;
        for ng in 2..=ns {
            for kk2 in 0..hpn {
                let h = (ng - 2) * hpn + kk2;
                let mut mixed = 0_u64;
                for j in 0..ng {
                    mixed ^= (shifted[j][t - 1] as u64).wrapping_mul(self.ple_mult[j]);
                }
                rows[n] = (mixed % self.ple_sizes[h] as u64) + self.ple_offsets[h] as u64;
                n += 1;
            }
        }

        // ponytail: sized from config — the tiny fixture's 256 hides this;
        // the real model's ple_embed_dim is 20480.
        let mut emb = vec![0.0_f32; c.ple_embed_dim.max(256)];
        // One immutable scalar is shared by every head. Keep row payloads
        // range-read from NVMe; do not cache the embedding table here.
        let ngram_scale = self.coli.as_ref()
            .map(|coli| coli.ple_ngram_scale(c.ple_layer as i32).unwrap_or(1.0));
        for h in 0..heads {
            let r = rows[h] as usize;
            // .coli mode: fetch the ngram row on demand (F8 E4M3 shards, one
            // pread per row — the 51 GB table is never resident).
            // safetensors mode: read the resident table.
            if let Some(coli) = &self.coli {
                let row_bytes = coli
                    .ple_ngram_row_f8(c.ple_layer as i32, r as u64, hd_per)
                    .unwrap_or_else(|e| panic!("ple ngram row {r} fetch failed: {e}"));
                let scale = ngram_scale.unwrap();
                for d in 0..hd_per {
                    emb[h * hd_per + d] = colisource::ColiSource::e4m3_decode(row_bytes[d]) * scale;
                }
            } else {
                let row = &self.ple_ngram.f[r * hd_per..(r + 1) * hd_per];
                emb[h * hd_per..(h + 1) * hd_per].copy_from_slice(row);
            }
        }

        let mut key = vec![0.0; hcd];
        let mut value = vec![0.0; d];
        let fused = std::env::var("QWEN_PLE_FUSED_INPUT")
            .map(|v| v != "0").unwrap_or(false)
            && matmul_mxfp4_multi(
                &mut [&mut key, &mut value], &emb,
                &[&self.ple_key_proj, &self.ple_value_proj],
            );
        if !fused {
            matmul(&mut key, &emb, &self.ple_key_proj);
            matmul(&mut value, &emb, &self.ple_value_proj);
        }
        let key_snap = key.clone();
        rmsnorm_grouped(&mut key, &key_snap, &self.ple_norm_key, hc, d, c.eps);
        let mut qn = vec![0.0; hcd];
        rmsnorm_grouped(&mut qn, stream, &self.ple_norm_query, hc, d, c.eps);
        let mut gated = vec![0.0; hcd];
        let r_d = 1.0 / (d as f32).sqrt();
        for g in 0..hc {
            let mut acc = 0.0_f32;
            for dd in 0..d {
                acc += key[g * d + dd] * qn[g * d + dd];
            }
            let tt = acc * r_d;
            let mut mag = tt.abs();
            if mag < 1e-6 {
                mag = 1e-6;
            }
            let sg = if tt >= 0.0 { 1.0 } else { -1.0 } * mag.sqrt();
            let sig = 1.0 / (1.0 + (-sg).exp());
            for dd in 0..d {
                gated[g * d + dd] = sig * value[dd];
            }
        }
        let mut conv_in = vec![0.0; hcd];
        rmsnorm_grouped(&mut conv_in, &gated, &self.ple_norm_conv, hc, d, c.eps);
        let st = pad + 1;
        for i in 0..hcd {
            let w = &self.ple_conv1d[i * kk..(i + 1) * kk];
            let state = &self.ple_conv_state[i * st..(i + 1) * st];
            let acc = causal_conv1d_sample(conv_in[i], state, w, dil);
            for s in (1..st).rev() {
                self.ple_conv_state[i * st + s] = self.ple_conv_state[i * st + s - 1];
            }
            self.ple_conv_state[i * st] = conv_in[i];
            stream[i] += gated[i] + silu(acc);
        }
    }

    fn init_token_stream(&self, token: usize) -> Vec<f32> {
        let d = self.cfg.hidden;
        let hc = self.cfg.hc_count;
        let row = self.embed.row_f32(token);
        if hc == 0 {
            return row;
        }
        let mut stream = vec![0.0; hc * d];
        for g in 0..hc {
            stream[g * d..(g + 1) * d].copy_from_slice(&row);
        }
        stream
    }

    fn push_ple_ring(&mut self, token: usize) {
        if self.cfg.ngram_size == 0 {
            return;
        }
        for i in 0..self.cfg.ngram_size - 1 {
            self.ple_ring[i] = self.ple_ring[i + 1];
        }
        self.ple_ring[self.cfg.ngram_size - 1] = token as i64;
    }

    fn forward_token_inner(&mut self, token: usize, pos: usize, want_logits: bool) -> Vec<f32> {
        let c = self.cfg.clone();
        let rope = rope_angles(pos, &c);
        let mut stream = self.init_token_stream(token);

        // Token-major execution may advance the PLE ring immediately because
        // only PLE consumes it. Layer-major prefill advances it immediately
        // before each chronological row reaches the PLE layer instead.
        self.push_ple_ring(token);

        for l in 0..c.layers {
            if !self.forward_layer(l, token, pos, &rope, &mut stream) {
                // Scheduler-driven block for cold experts: resume cursor
                // stashed, persistent state applied exactly once. The caller
                // must not consume these logits.
                return Vec::new();
            }
        }
        if want_logits {
            self.forward_tail(&stream)
        } else {
            // Intermediate prefill rows need only causal state (KV, GDN,
            // PLE/indexer state, etc.) committed by the layer loop. The global
            // HC tail and vocabulary projection are pure functions of this
            // row's final stream and are not inputs to any later prompt row.
            Vec::new()
        }
    }

    pub fn forward_token(&mut self, token: usize, pos: usize) -> Vec<f32> {
        self.forward_token_inner(token, pos, true)
    }

    /// Advance one prompt token without computing disposable final-row logits.
    /// This is numerically identical for all persistent causal state to
    /// `forward_token`; only the non-stateful global HC tail + LM head are
    /// omitted. Use `forward_token` for the final prompt token.
    pub fn prefill_token(&mut self, token: usize, pos: usize) {
        let _ = self.forward_token_inner(token, pos, false);
    }

    /// Bounded layer-major prompt prefill. This deliberately reuses the exact
    /// existing per-row layer implementation: only the traversal order changes
    /// from token-major to layer-major within the chunk. Each causal layer is
    /// still evaluated in increasing token position, so KV, GDN recurrent /
    /// convolution state and QSA state advance in the same order as canonical
    /// token-major execution.
    ///
    /// PLE is the only consumer of `ple_ring`; therefore its token history is
    /// advanced immediately before each row reaches the PLE layer rather than
    /// at chunk admission. The global HC tail + vocabulary projection run only
    /// for the final requested row.
    pub fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        want_logits_last: bool,
    ) -> Result<Option<Vec<f32>>, String> {
        if tokens.is_empty() {
            return Ok(None);
        }
        if self.sched_mode {
            return Err("layer-major prefill is not yet supported in scheduler mode".into());
        }
        let c = self.cfg.clone();
        if c.hc_count == 0 {
            // Correctness-first Qwen3.x prefill. Preserve canonical token-major
            // state evolution until the MXFP4 dense projections have their own
            // qualified layer-major batch path; routed expert caching/MetalIO
            // still use the shared engine on every row.
            let mut logits = None;
            for (row, &token) in tokens.iter().enumerate() {
                let pos = start_pos + row;
                if want_logits_last && row + 1 == tokens.len() {
                    logits = Some(self.forward_token(token as usize, pos));
                } else {
                    self.prefill_token(token as usize, pos);
                }
            }
            return Ok(logits);
        }
        let mut streams: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&token| self.init_token_stream(token as usize))
            .collect();
        let ropes: Vec<Vec<(f32, f32)>> = (0..tokens.len())
            .map(|row| rope_angles(start_pos + row, &c))
            .collect();

        for l in 0..c.layers {
            // Keep one layer borrowed for the entire chunk. Each row advances
            // this layer's causal state in increasing position, then pauses at
            // the routed-MoE seam so the chunk can acquire its expert union.
            let mut layer = std::mem::replace(&mut self.layers[l], Layer::empty());
            let mut moe_inputs: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
            let mut injectors: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());

            let d = c.hidden;
            let hc = c.hc_count;
            let mut mixed_rows: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
            let mut temporal_injectors: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());

            // First HC is row-independent. Stop all rows at the temporal seam
            // so GDN can batch its dense projections while retaining ordered
            // convolution/recurrent state updates inside gdn_chunk_batched().
            for row in 0..tokens.len() {
                let token = tokens[row] as usize;
                let stream = &mut streams[row];
                if c.ple_layer == l as i64 {
                    self.push_ple_ring(token);
                    self.ple_forward(stream);
                }
                let mut mixed = vec![0.0; d];
                let mut inj = vec![0.0; hc];
                let mut _hc_t = logan_core::telemetry::Span::begin("hc");
                self.hc_mix(
                    &layer.hc_norm,
                    &layer.hc_mix_down,
                    &layer.hc_mix_up,
                    Some(&layer.hc_inject),
                    stream,
                    &mut mixed,
                    Some(&mut inj),
                );
                self.spans.hc_ms += _hc_t.end();
                mixed_rows.push(mixed);
                temporal_injectors.push(inj);
            }

            let mut temporal_rows = vec![vec![0.0_f32; d]; tokens.len()];
            if layer.is_gdn {
                let mut _gdn_t = logan_core::telemetry::Span::begin("gdn");
                if !self.gdn_chunk_batched(&mut layer, l, &mixed_rows, &mut temporal_rows) {
                    for row in 0..tokens.len() {
                        self.gdn_token(&mut layer, l, &mixed_rows[row], &mut temporal_rows[row]);
                    }
                }
                self.spans.gdn_ms += _gdn_t.end();
            } else {
                let mut projected = self
                    .project_attention_chunk(&layer, &mixed_rows)
                    .map(std::collections::VecDeque::from);
                for row in 0..tokens.len() {
                    let pos = start_pos + row;
                    let mut _attn_t = logan_core::telemetry::Span::begin("attn");
                    if let Some(projection) = projected.as_mut().and_then(|q| q.pop_front()) {
                        if layer.is_qsa {
                            let sel = self.qsa_select(
                                &layer,
                                l,
                                &mixed_rows[row],
                                pos,
                                &ropes[row],
                                projection.index_qk.as_deref(),
                            );
                            self.attention_common(
                                &layer,
                                l,
                                &mixed_rows[row],
                                pos,
                                &ropes[row],
                                Some(&sel),
                                Some(projection),
                                &mut temporal_rows[row],
                            );
                        } else {
                            self.attention_common(
                                &layer,
                                l,
                                &mixed_rows[row],
                                pos,
                                &ropes[row],
                                None,
                                Some(projection),
                                &mut temporal_rows[row],
                            );
                        }
                    } else if layer.is_qsa {
                        self.sparse_attn_token(
                            &layer,
                            l,
                            &mixed_rows[row],
                            pos,
                            &ropes[row],
                            &mut temporal_rows[row],
                        );
                    } else {
                        self.attention_token(
                            &layer,
                            l,
                            &mixed_rows[row],
                            pos,
                            &ropes[row],
                            &mut temporal_rows[row],
                        );
                    }
                    self.spans.attn_ms += _attn_t.end();
                }
            }

            for row in 0..tokens.len() {
                let stream = &mut streams[row];
                let mut inj = std::mem::take(&mut temporal_injectors[row]);
                for g in 0..hc {
                    for dd in 0..d {
                        stream[g * d + dd] += inj[g] * temporal_rows[row][dd];
                    }
                }
                let mut m2 = vec![0.0; d];
                self.hc_mix(
                    &layer.hc_mlp_norm,
                    &layer.hc_mlp_mix_down,
                    &layer.hc_mlp_mix_up,
                    Some(&layer.hc_mlp_inject),
                    stream,
                    &mut m2,
                    Some(&mut inj),
                );
                moe_inputs.push(m2);
                injectors.push(inj);
            }

            // Route every row exactly once. Batched prefill groups route
            // occurrences by expert for GPU execution, then scatters the raw
            // expert outputs back to per-row/per-rank slots so floating-point
            // reduction still follows each token's canonical top-k order.
            let mut routes = Vec::with_capacity(tokens.len());
            let mut seen = vec![false; c.experts];
            let mut unique = Vec::new();
            for x in &moe_inputs {
                let route = self.route_topk(&layer, x);
                for &ei in route.0.iter().take(c.topk) {
                    if !seen[ei] {
                        seen[ei] = true;
                        unique.push(ei);
                    }
                }
                routes.push(route);
            }

            let batch_enabled = std::env::var("QWEN_PREFILL_MOE_BATCH")
                .map(|v| v != "0")
                .unwrap_or(false);
            let mut batch_done = false;
            if batch_enabled
                && self.metal_direct
                && crate::ffi::direct_available()
                && self.coli.is_some()
                && unique.len() <= cache_cap()
            {
                let mut _io_t = logan_core::telemetry::Span::begin("io");
                let resident = self.preload_expert_set(l, &unique);
                self.spans.io_ms += _io_t.end();
                if resident {
                    let mut refs: Vec<std::rc::Rc<crate::colisource::SlotRef>> =
                        Vec::with_capacity(unique.len());
                    let mut descriptors = Vec::with_capacity(unique.len());
                    let mut row_offsets = Vec::with_capacity(unique.len() + 1);
                    let mut grouped_x = Vec::with_capacity(tokens.len() * c.topk * c.hidden);
                    let mut scatter: Vec<(usize, usize)> =
                        Vec::with_capacity(tokens.len() * c.topk);
                    row_offsets.push(0_i32);

                    let mut bind_ok = true;
                    for &ei in &unique {
                        let Some(r) = self.cached_expert_issue(l as i32, ei as i32, false) else {
                            bind_ok = false;
                            break;
                        };
                        descriptors.push(Self::slot_descriptor(&r));
                        refs.push(r);
                        for row in 0..tokens.len() {
                            let (idx, _, _) = &routes[row];
                            for rank in 0..c.topk {
                                if idx[rank] == ei {
                                    grouped_x.extend_from_slice(&moe_inputs[row]);
                                    scatter.push((row, rank));
                                }
                            }
                        }
                        row_offsets.push(scatter.len() as i32);
                    }

                    if bind_ok && scatter.len() == tokens.len() * c.topk {
                        let mut grouped_y = vec![0.0_f32; scatter.len() * c.hidden];
                        let mut _gpu_t = logan_core::telemetry::Span::begin("gpu");
                        let gpu_ok = crate::ffi::moe_rows(
                            &descriptors,
                            &row_offsets,
                            &grouped_x,
                            &mut grouped_y,
                            c.hidden,
                            c.moe_inter,
                        );
                        self.spans.gpu_ms += _gpu_t.end();
                        if gpu_ok {
                            let mut contrib = vec![0.0_f32; tokens.len() * c.topk * c.hidden];
                            for (grouped_row, &(row, rank)) in scatter.iter().enumerate() {
                                let src = &grouped_y
                                    [grouped_row * c.hidden..(grouped_row + 1) * c.hidden];
                                let off = (row * c.topk + rank) * c.hidden;
                                contrib[off..off + c.hidden].copy_from_slice(src);
                            }
                            for row in 0..tokens.len() {
                                let (idx, val, wsum) = &routes[row];
                                debug_assert!(idx.len() >= c.topk);
                                let mut moe = vec![0.0_f32; c.hidden];
                                for rank in 0..c.topk {
                                    let w = val[rank] / *wsum;
                                    let off = (row * c.topk + rank) * c.hidden;
                                    for dd in 0..c.hidden {
                                        moe[dd] += contrib[off + dd] * w;
                                    }
                                }
                                let mut _shared_t = logan_core::telemetry::Span::begin("shared");
                                let (sy, gs) =
                                    self.shared_expert_value(&layer, l, &moe_inputs[row]);
                                self.spans.shared_ms += _shared_t.end();
                                for dd in 0..c.hidden {
                                    moe[dd] += sy[dd] * gs;
                                }
                                for g in 0..c.hc_count {
                                    for dd in 0..c.hidden {
                                        streams[row][g * c.hidden + dd] +=
                                            injectors[row][g] * moe[dd];
                                    }
                                }
                            }
                            batch_done = true;
                        }
                    }
                    drop(refs);
                }
            }

            if !batch_done {
                for row in 0..tokens.len() {
                    let mut moe = vec![0.0; c.hidden];
                    let (idx, val, wsum) = &routes[row];
                    self.moe_token_routed(&layer, l, &moe_inputs[row], &mut moe, idx, val, *wsum);
                    for g in 0..c.hc_count {
                        for dd in 0..c.hidden {
                            streams[row][g * c.hidden + dd] += injectors[row][g] * moe[dd];
                        }
                    }
                }
            }
            self.layers[l] = layer;
        }

        if want_logits_last {
            Ok(streams.last().map(|stream| self.forward_tail(stream)))
        } else {
            Ok(None)
        }
    }

    /// One transformer layer of the token stream — the exact canonical per-layer
    /// body. Returns true when the layer completed; false when scheduler-driven
    /// mode met cold routed experts, in which case the layer's MoE input and
    /// injectors are stashed in `sched_pause` for a byte-identical resume (all
    /// persistent state mutations — GDN conv/recurrent, KV, PLE — happen in the
    /// pre-MoE phases and are applied exactly once per token).
    fn forward_layer(
        &mut self,
        l: usize,
        token: usize,
        pos: usize,
        rope: &[(f32, f32)],
        stream: &mut [f32],
    ) -> bool {
        let c = self.cfg.clone();
        let d = c.hidden;
        let hc = c.hc_count;
        // Share-on-read: `self.layers[l].clone()` deep-copied EVERY weight
        // per token (~3.7 GB memcpy/token on the real model — it would
        // dominate any Metal win). Split-borrow: methods take &Layer, so
        // pull the layer out, run both sub-phases, put it back.
        let mut layer = std::mem::replace(&mut self.layers[l], Layer::empty());

        // Both residual layouts can overlap the previous route with temporal
        // work; the helper retains its opt-in and scheduler exclusion gates.
        self.prefetch_previous_route(l);

        // Qwen3.x classic residual block. The same temporal/MoE kernels and
        // expert residency machinery are shared with the hyper-connection
        // engine; only the residual plumbing and two RMSNorm sites differ.
        if hc == 0 {
            let mut mixed = vec![0.0; d];
            rmsnorm_row_shifted(&mut mixed, stream, &layer.in_ln, c.eps);
            let mut attn = vec![0.0; d];
            if layer.is_gdn {
                let mut _gdn_t = logan_core::telemetry::Span::begin("gdn");
                self.gdn_token(&mut layer, l, &mixed, &mut attn);
                self.spans.gdn_ms += _gdn_t.end();
            } else {
                let mut _attn_t = logan_core::telemetry::Span::begin("attn");
                self.attention_token(&layer, l, &mixed, pos, rope, &mut attn);
                self.spans.attn_ms += _attn_t.end();
            }
            for dd in 0..d {
                stream[dd] += attn[dd];
            }

            let mut m2 = vec![0.0; d];
            rmsnorm_row_shifted(&mut m2, stream, &layer.hc_mlp_norm, c.eps);
            let mut moe = vec![0.0; d];
            self.moe_token(&layer, l, &m2, &mut moe);
            if let Some(experts) = self.sched_blocked.take() {
                self.layers[l] = layer;
                self.sched_pause = Some(TokenPause {
                    layer: l,
                    experts,
                    x: m2,
                    inj: Vec::new(),
                    stream: stream.to_vec(),
                    token,
                    pos,
                });
                return false;
            }
            for dd in 0..d {
                stream[dd] += moe[dd];
            }
            self.layers[l] = layer;
            return true;
        }

        let mut mixed = vec![0.0; d];
        let mut attn = vec![0.0; d];
        if c.ple_layer == l as i64 {
            self.ple_forward(stream);
        }
        let mut inj = vec![0.0; hc];
        let mut _hc_t = logan_core::telemetry::Span::begin("hc");
        self.hc_mix(
            &layer.hc_norm,
            &layer.hc_mix_down,
            &layer.hc_mix_up,
            Some(&layer.hc_inject),
            &stream,
            &mut mixed,
            Some(&mut inj),
        );
        self.spans.hc_ms += _hc_t.end();
        if layer.is_gdn {
            let mut _gdn_t = logan_core::telemetry::Span::begin("gdn");
            self.gdn_token(&mut layer, l, &mixed, &mut attn);
            self.spans.gdn_ms += _gdn_t.end();
        } else {
            let mut _attn_t = logan_core::telemetry::Span::begin("attn");
            if layer.is_qsa {
                self.sparse_attn_token(&layer, l, &mixed, pos, &rope, &mut attn);
            } else {
                self.attention_token(&layer, l, &mixed, pos, &rope, &mut attn);
            }
            self.spans.attn_ms += _attn_t.end();
        }
        for g in 0..hc {
            for dd in 0..d {
                stream[g * d + dd] += inj[g] * attn[dd];
            }
        }
        let mut m2 = vec![0.0; d];
        let mut moe = vec![0.0; d];
        self.hc_mix(
            &layer.hc_mlp_norm,
            &layer.hc_mlp_mix_down,
            &layer.hc_mlp_mix_up,
            Some(&layer.hc_mlp_inject),
            &stream,
            &mut m2,
            Some(&mut inj),
        );
        self.moe_token(&layer, l, &m2, &mut moe);
        if let Some(experts) = self.sched_blocked.take() {
            // Block point: the layer's MoE phase reported cold expert(s).
            // Restore the layer and stash the exact resume cursor. Nothing
            // in the MoE phase mutates persistent state, so re-entering it
            // after the load reproduces the canonical order.
            self.layers[l] = layer;
            self.sched_pause = Some(TokenPause {
                layer: l,
                experts,
                x: m2,
                inj,
                stream: stream.to_vec(),
                token,
                pos,
            });
            return false;
        }
        for g in 0..hc {
            for dd in 0..d {
                stream[g * d + dd] += inj[g] * moe[dd];
            }
        }
        self.layers[l] = layer;
        true
    }

    /// Post-layer tail: global hyper-connection mixer, optional final norm,
    /// and the LM head. Shared by the full forward and the schedule resume.
    fn forward_tail(&mut self, stream: &[f32]) -> Vec<f32> {
        let c = self.cfg.clone();
        let d = c.hidden;
        if c.hc_count == 0 {
            let mut _head_t = logan_core::telemetry::Span::begin("head");
            let mut normed = vec![0.0; d];
            // MLX Qwen3.5/3.6 sanitize() folds the raw HF `(1 + weight)`
            // RMSNorm convention into model.norm.weight before quantization.
            rmsnorm_row_shifted(&mut normed, stream, &self.final_norm, c.eps);
            let mut logits = vec![0.0; c.vocab];
            self.lm_head_matmul(&mut logits, &normed);
            self.spans.head_ms += _head_t.end();
            return logits;
        }
        // final global hc_mix (no inject)
        let mut out = vec![0.0; d];
        let mut _hc3_t = logan_core::telemetry::Span::begin("hc");
        self.hc_mix(
            &self.hc_global.norm,
            &self.hc_global.mix_down,
            &self.hc_global.mix_up,
            None,
            &stream,
            &mut out,
            None,
        );
        self.spans.hc_ms += _hc3_t.end();
        // qwen4_no_final_norm: when hc is active and norm.weight is absent,
        // the mixer output goes straight to the head (C: qwen4_no_final_norm
        // = hc_count > 0 && !st_have(norm.weight)).
        let mut _head_t = logan_core::telemetry::Span::begin("head");
        let mut logits = vec![0.0; c.vocab];
        if !self.final_norm.is_empty() {
            let mut normed = vec![0.0; d];
            rmsnorm_row(&mut normed, &out, &self.final_norm, c.eps);
            self.lm_head_matmul(&mut logits, &normed);
        } else {
            self.lm_head_matmul(&mut logits, &out);
        }
        self.spans.head_ms += _head_t.end();
        logits
    }

    /// Switch the model to scheduler-driven expert acquisition (QWEN_SCHED=1).
    /// Must be called before any forward; the canonical path never does.
    pub fn enable_sched_mode(&mut self) {
        self.sched_mode = true;
    }

    /// Scheduler-driven forward (issue #53): one token forward with the
    /// routed-MoE phase reporting cold experts instead of loading them.
    /// A blocked forward stashes a resume cursor; resubmitting the SAME
    /// (token, pos) resumes byte-identically (persistent state is mutated
    /// exactly once per token). A stale cursor from an abandoned op is
    /// discarded.
    pub fn forward_scheduled(&mut self, token: usize, pos: usize) -> SchedForward {
        if let Some(pause) = self.sched_pause.take() {
            if pause.token == token && pause.pos == pos {
                return self.forward_resume(pause, pos);
            }
            // Stale pause from an abandoned op: fall through fresh.
        }
        let logits = self.forward_token(token, pos);
        match self.sched_pause.as_ref() {
            Some(pause) => SchedForward::NeedExperts {
                layer: pause.layer,
                experts: pause.experts.clone(),
            },
            None => SchedForward::Logits(logits),
        }
    }

    fn forward_resume(&mut self, pause: TokenPause, pos: usize) -> SchedForward {
        let c = self.cfg.clone();
        let d = c.hidden;
        let hc = c.hc_count;
        let rope = rope_angles(pause.pos, &c);
        let mut stream = pause.stream;
        // Re-enter the blocked layer's MoE phase only; its dense phases and
        // persistent state are already applied exactly once.
        let mut moe = vec![0.0; d];
        let layer = std::mem::replace(&mut self.layers[pause.layer], Layer::empty());
        self.moe_token(&layer, pause.layer, &pause.x, &mut moe);
        if let Some(experts) = self.sched_blocked.take() {
            // A resumed layer is still cold (eviction between ops): re-block.
            self.layers[pause.layer] = layer;
            let new_pause = TokenPause {
                layer: pause.layer,
                experts: experts.clone(),
                x: pause.x,
                inj: pause.inj,
                stream,
                token: pause.token,
                pos: pause.pos,
            };
            self.sched_pause = Some(new_pause);
            return SchedForward::NeedExperts {
                layer: pause.layer,
                experts,
            };
        }
        if hc == 0 {
            for dd in 0..d {
                stream[dd] += moe[dd];
            }
        } else {
            for g in 0..hc {
                for dd in 0..d {
                    stream[g * d + dd] += pause.inj[g] * moe[dd];
                }
            }
        }
        self.layers[pause.layer] = layer;
        for l in pause.layer + 1..c.layers {
            if !self.forward_layer(l, pause.token, pos, &rope, &mut stream) {
                let blocked = self
                    .sched_pause
                    .as_ref()
                    .expect("blocked layer stashes a pause");
                return SchedForward::NeedExperts {
                    layer: blocked.layer,
                    experts: blocked.experts.clone(),
                };
            }
        }
        SchedForward::Logits(self.forward_tail(&stream))
    }

    /// Scheduler-driven load of one planned expert (executor-lane role: the
    /// lane owns the MetalIO wait — the scheduler thread and the model's
    /// forward path never wait on it). Resolves nothing at runtime: the plan
    /// carries the record identity, stream regions, and dims. The load
    /// publishes into the physical store (the engine LRU); eviction frees
    /// the slot. No-op when already resident.
    pub fn load_expert_planned(
        &mut self,
        li: i32,
        ei: i32,
        planned: &crate::plan::PlannedExpert,
    ) -> Result<(), String> {
        if self.expert_store.peek((li as u32, ei as u32)).is_some() {
            return Ok(());
        }
        let coli = self
            .coli
            .as_ref()
            .ok_or_else(|| "load_expert_planned requires .coli mode".to_string())?;
        let shard = coli.pkg_ref().shard_path(planned.shard_id).ok_or_else(|| {
            format!(
                "planned expert ({li},{ei}) shard {} missing",
                planned.shard_id
            )
        })?;
        let fid = crate::ffi::mio_file(&shard)
            .ok_or_else(|| format!("MetalIO unavailable for expert ({li},{ei})"))?;
        let regions = [planned.regions[0], planned.regions[1], planned.regions[2]];
        let (slot, ev) = crate::ffi::mio_load_expert(fid, &regions)
            .ok_or_else(|| format!("MetalIO slot alloc failed for expert ({li},{ei})"))?;
        let ptr = unsafe { crate::ffi::metalio_slot_ptr(slot) } as *mut u8;
        if ptr.is_null() {
            unsafe { crate::ffi::metalio_slot_free(slot) };
            return Err(format!(
                "MetalIO slot has no CPU-visible memory for expert ({li},{ei})"
            ));
        }
        let [gb, ub, db] = [regions[0].1, regions[1].1, regions[2].1];
        let up_off = (gb + 15) & !15usize;
        let down_off = (up_off + ub + 15) & !15usize;
        let se = crate::colisource::SlotExpert {
            slot,
            gate_bytes: gb,
            up_offset: up_off,
            up_bytes: ub,
            down_offset: down_off,
            down_bytes: db,
            ptr,
            pending: std::cell::Cell::new(ev),
            bf16_cache: std::cell::RefCell::new(None),
            rows: [planned.dims[0].0, planned.dims[1].0, planned.dims[2].0],
            cols: [planned.dims[0].1, planned.dims[1].1, planned.dims[2].1],
        };
        let (mut evicted, _) = self.expert_store.insert((li as u32, ei as u32), se);
        if let Some(mut e) = evicted.take() {
            e.release();
        }
        if unsafe { crate::ffi::metalio_wait(ev) } != 0 {
            return Err(format!("MetalIO load failed for expert ({li},{ei})"));
        }
        if let Some(v) = self.expert_store.peek((li as u32, ei as u32)) {
            v.pending.set(0);
        }
        Ok(())
    }

    /// Emit the LOGAN_PROFILE=1 per-request summary (spans + Metal counters
    /// + LRU hit/miss). No-op when profiling is disabled.
    pub fn profile_summary(&self, tokens: usize, total_ms: f64) {
        if !logan_core::telemetry::enabled() {
            return;
        }
        let (e, s, w, k, fc, fe) = logan_metal::metal_profile();
        let mio = logan_metal::mio_stats();
        let metal = logan_core::telemetry::MetalCounters {
            encode_ns: e,
            submit_ns: s,
            wait_ns: w,
            kernel_ns: k,
            fused_calls: fc,
            fused_experts: fe,
            mio_loads: mio.loads,
            mio_bytes: mio.bytes,
            mio_waits: mio.waits,
            mio_fails: mio.fails,
        };
        let mut spans = self.spans.clone();
        spans.total_ms = total_ms;
        logan_core::telemetry::emit_request_summary(
            tokens,
            &spans,
            &metal,
            self.expert_store.hits,
            self.expert_store.misses,
        );
        if std::env::var("QWEN_ROUTE_OVERLAP")
            .map(|v| v != "0")
            .unwrap_or(false)
        {
            let common: u64 = self.route_overlap_common.iter().sum();
            let total: u64 = self.route_overlap_total.iter().sum();
            let pairs: u64 = self.route_overlap_pairs.iter().sum();
            eprintln!(
                "logan route-overlap: common={common} total={total} pairs={pairs} rate={:.3}",
                if total == 0 {
                    0.0
                } else {
                    common as f64 / total as f64
                }
            );
            let detail = self
                .route_overlap_common
                .iter()
                .zip(&self.route_overlap_total)
                .zip(&self.route_overlap_pairs)
                .enumerate()
                .map(|(li, ((&c, &t), &p))| {
                    format!(
                        "{li}:{:.3}/{p}",
                        if t == 0 { 0.0 } else { c as f64 / t as f64 }
                    )
                })
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!("logan route-overlap layers: {detail}");
        }
    }
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

/// Expert-cache capacity. An explicit `QWEN4_CACHE` always wins.
///
/// On 16 GiB Apple Silicon, a 256-slot expert cache measurably pressures UMA
/// residency: the GDN command-buffer scheduling wait rises sharply even though
/// the GDN GPU work itself is unchanged. Keep more headroom for dense/GDN GPU
/// resources on that tier; larger or unknown machines retain the established
/// 256-slot default until they have their own measured residency curve.
fn make_expert_store(
    layers: usize,
    topk: usize,
) -> logan_core::expert::ExpertStore<crate::colisource::SlotExpert> {
    // Explicit override: 0 restores the legacy global LRU for A/Bs.
    if let Some(per_layer) = std::env::var("QWEN4_CACHE_PER_LAYER")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        return if per_layer == 0 {
            logan_core::expert::ExpertStore::new(cache_cap())
        } else {
            logan_core::expert::ExpertStore::new_layered(layers, per_layer)
        };
    }

    // Measured on the 16 GiB M2 Qwen3.6-35B-A3B workload: 8/layer cuts
    // routed-expert misses ~30% versus the 128-entry global LRU and beats
    // 10/layer end-to-end because the latter adds UMA/residency pressure.
    // A layer-local cache must be able to hold one complete active route. Use
    // the model's declared routing top-k directly rather than assuming a value
    // from the model family: REAP/custom checkpoints may change it. A zero
    // value is malformed metadata, so keep a one-slot defensive floor here;
    // normal model validation should reject such a config earlier.
    if cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && detect_physical_ram_bytes().is_some_and(|bytes| bytes <= 16 * 1024 * 1024 * 1024)
    {
        return logan_core::expert::ExpertStore::new_layered(layers, topk.max(1));
    }
    logan_core::expert::ExpertStore::new(cache_cap())
}

pub fn cache_cap() -> usize {
    if let Some(cap) = std::env::var("QWEN4_CACHE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        return cap.max(1);
    }
    default_cache_cap_for_ram(detect_physical_ram_bytes())
}

fn default_cache_cap_for_ram(ram_bytes: Option<u64>) -> usize {
    const GIB: u64 = 1024 * 1024 * 1024;
    if cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && ram_bytes.is_some_and(|bytes| bytes <= 16 * GIB)
    {
        128
    } else {
        256
    }
}

fn detect_physical_ram_bytes() -> Option<u64> {
    if let Some(gib) = std::env::var("RAM_GB")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
    {
        return gib.checked_mul(1024 * 1024 * 1024);
    }
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8(output.stdout).ok()?;
        return text.trim().parse::<u64>().ok();
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

fn load_wt(st: &StFile, name: &str, o: usize, i: usize) -> Result<Wt, String> {
    Ok(Wt {
        f: st.f32(name, &[o as u64, i as u64])?,
        bytes: None,
        o,
        i,
    })
}

impl Model {
    pub fn load(st: &StFile, cfg: &Cfg) -> Result<Model, String> {
        let mut experts = Vec::new();
        let mut layers = Vec::new();
        for l in 0..cfg.layers {
            let lp = format!("model.layers.{l}");
            let is_gdn = cfg.gdn_layers[l];
            let is_qsa = cfg.qsa_layers[l];
            let cdim = cfg.lin_k_dim * cfg.lin_k_heads * 2 + cfg.lin_v_dim * cfg.lin_v_heads;
            let vdim = cfg.lin_v_dim * cfg.lin_v_heads;
            let hd = cfg.head_dim;
            let hcd = cfg.hc_count * cfg.hidden;
            // qwen4 hc path: no per-layer input/post norms (hc_mix normalizes)
            let in_ln: Vec<f32> = if cfg.hc_count > 0 {
                vec![]
            } else {
                st.f32(
                    &format!("{lp}.input_layernorm.weight"),
                    &[cfg.hidden as u64],
                )?
            };
            let layer = Layer {
                in_ln,
                is_gdn,
                is_qsa,
                gdn_a_log: if is_gdn {
                    st.f32(
                        &format!("{lp}.linear_attn.A_log"),
                        &[cfg.lin_v_heads as u64],
                    )?
                } else {
                    vec![]
                },
                gdn_dt_bias: if is_gdn {
                    st.f32(
                        &format!("{lp}.linear_attn.dt_bias"),
                        &[cfg.lin_v_heads as u64],
                    )?
                } else {
                    vec![]
                },
                gdn_conv1d: if is_gdn {
                    st.f32(
                        &format!("{lp}.linear_attn.conv1d.weight"),
                        &[(cdim * cfg.conv_kernel) as u64],
                    )?
                } else {
                    vec![]
                },
                gdn_in_a: if is_gdn {
                    load_wt(
                        st,
                        &format!("{lp}.linear_attn.in_proj_a.weight"),
                        cfg.lin_v_heads,
                        cfg.hidden,
                    )?
                } else {
                    Wt {
                        f: vec![],
                        bytes: None,
                        o: 0,
                        i: 0,
                    }
                },
                gdn_in_b: if is_gdn {
                    load_wt(
                        st,
                        &format!("{lp}.linear_attn.in_proj_b.weight"),
                        cfg.lin_v_heads,
                        cfg.hidden,
                    )?
                } else {
                    Wt {
                        f: vec![],
                        bytes: None,
                        o: 0,
                        i: 0,
                    }
                },
                gdn_in_qkv: if is_gdn {
                    load_wt(
                        st,
                        &format!("{lp}.linear_attn.in_proj_qkv.weight"),
                        cdim,
                        cfg.hidden,
                    )?
                } else {
                    Wt {
                        f: vec![],
                        bytes: None,
                        o: 0,
                        i: 0,
                    }
                },
                gdn_in_z: if is_gdn {
                    load_wt(
                        st,
                        &format!("{lp}.linear_attn.in_proj_z.weight"),
                        vdim,
                        cfg.hidden,
                    )?
                } else {
                    Wt {
                        f: vec![],
                        bytes: None,
                        o: 0,
                        i: 0,
                    }
                },
                gdn_norm: if is_gdn {
                    st.f32(
                        &format!("{lp}.linear_attn.norm.weight"),
                        &[cfg.lin_v_dim as u64],
                    )?
                } else {
                    vec![]
                },
                gdn_out: if is_gdn {
                    load_wt(
                        st,
                        &format!("{lp}.linear_attn.out_proj.weight"),
                        cfg.hidden,
                        vdim,
                    )?
                } else {
                    Wt {
                        f: vec![],
                        bytes: None,
                        o: 0,
                        i: 0,
                    }
                },
                attn_q: if !is_gdn {
                    load_wt(
                        st,
                        &format!("{lp}.self_attn.q_proj.weight"),
                        2 * cfg.heads * hd,
                        cfg.hidden,
                    )?
                } else {
                    Wt {
                        f: vec![],
                        bytes: None,
                        o: 0,
                        i: 0,
                    }
                },
                attn_k: if !is_gdn {
                    load_wt(
                        st,
                        &format!("{lp}.self_attn.k_proj.weight"),
                        cfg.kv_heads * hd,
                        cfg.hidden,
                    )?
                } else {
                    Wt {
                        f: vec![],
                        bytes: None,
                        o: 0,
                        i: 0,
                    }
                },
                attn_v: if !is_gdn {
                    load_wt(
                        st,
                        &format!("{lp}.self_attn.v_proj.weight"),
                        cfg.kv_heads * hd,
                        cfg.hidden,
                    )?
                } else {
                    Wt {
                        f: vec![],
                        bytes: None,
                        o: 0,
                        i: 0,
                    }
                },
                attn_o: if !is_gdn {
                    load_wt(
                        st,
                        &format!("{lp}.self_attn.o_proj.weight"),
                        cfg.hidden,
                        cfg.heads * hd,
                    )?
                } else {
                    Wt {
                        f: vec![],
                        bytes: None,
                        o: 0,
                        i: 0,
                    }
                },
                attn_qn: if !is_gdn {
                    st.f32(&format!("{lp}.self_attn.q_norm.weight"), &[hd as u64])?
                } else {
                    vec![]
                },
                attn_kn: if !is_gdn {
                    st.f32(&format!("{lp}.self_attn.k_norm.weight"), &[hd as u64])?
                } else {
                    vec![]
                },
                index_qk: if is_qsa {
                    load_wt(
                        st,
                        &format!("{lp}.self_attn.indexer.index_qk_proj.weight"),
                        cfg.idx_n_heads * cfg.idx_head_dim + cfg.idx_kv_heads * cfg.idx_head_dim,
                        cfg.hidden,
                    )?
                } else {
                    Wt {
                        f: vec![],
                        bytes: None,
                        o: 0,
                        i: 0,
                    }
                },
                idx_qn: if is_qsa {
                    st.f32(
                        &format!("{lp}.self_attn.indexer.q_layernorm.weight"),
                        &[cfg.idx_head_dim as u64],
                    )?
                } else {
                    vec![]
                },
                idx_kn: if is_qsa {
                    st.f32(
                        &format!("{lp}.self_attn.indexer.k_layernorm.weight"),
                        &[cfg.idx_head_dim as u64],
                    )?
                } else {
                    vec![]
                },
                hc_norm: st.f32(
                    &format!("{lp}.attn_hyper_connection.hc_norm.weight"),
                    &[hcd as u64],
                )?,
                hc_mix_down: load_wt(
                    st,
                    &format!("{lp}.attn_hyper_connection.input_mix_weight_down.weight"),
                    cfg.hc_lowrank,
                    hcd,
                )?,
                hc_mix_up: load_wt(
                    st,
                    &format!("{lp}.attn_hyper_connection.input_mix_weight_up.weight"),
                    hcd,
                    cfg.hc_lowrank,
                )?,
                hc_inject: load_wt(
                    st,
                    &format!("{lp}.attn_hyper_connection.block_inject_weight.weight"),
                    cfg.hc_count,
                    hcd,
                )?,
                hc_mlp_norm: st.f32(
                    &format!("{lp}.mlp_hyper_connection.hc_norm.weight"),
                    &[hcd as u64],
                )?,
                hc_mlp_mix_down: load_wt(
                    st,
                    &format!("{lp}.mlp_hyper_connection.input_mix_weight_down.weight"),
                    cfg.hc_lowrank,
                    hcd,
                )?,
                hc_mlp_mix_up: load_wt(
                    st,
                    &format!("{lp}.mlp_hyper_connection.input_mix_weight_up.weight"),
                    hcd,
                    cfg.hc_lowrank,
                )?,
                hc_mlp_inject: load_wt(
                    st,
                    &format!("{lp}.mlp_hyper_connection.block_inject_weight.weight"),
                    cfg.hc_count,
                    hcd,
                )?,
                router: load_wt(
                    st,
                    &format!("{lp}.mlp.gate.weight"),
                    cfg.experts,
                    cfg.hidden,
                )?,
                se_gate: load_wt(
                    st,
                    &format!("{lp}.mlp.shared_expert.gate_proj.weight"),
                    cfg.shared_inter,
                    cfg.hidden,
                )?,
                se_up: load_wt(
                    st,
                    &format!("{lp}.mlp.shared_expert.up_proj.weight"),
                    cfg.shared_inter,
                    cfg.hidden,
                )?,
                se_down: load_wt(
                    st,
                    &format!("{lp}.mlp.shared_expert.down_proj.weight"),
                    cfg.hidden,
                    cfg.shared_inter,
                )?,
                se_g: load_wt(
                    st,
                    &format!("{lp}.mlp.shared_expert_gate.weight"),
                    1,
                    cfg.hidden,
                )?,
            };
            let mut layer_experts = Vec::new();
            for e in 0..cfg.experts {
                let elp = format!("{lp}.mlp.experts.{e}");
                let gu = st.f32(
                    &format!("{elp}.gate_up_proj"),
                    &[(2 * cfg.moe_inter) as u64, cfg.hidden as u64],
                )?;
                let dn = st.f32(
                    &format!("{elp}.down_proj"),
                    &[cfg.hidden as u64, cfg.moe_inter as u64],
                )?;
                let half = cfg.moe_inter * cfg.hidden;
                let gate = Wt {
                    f: gu[..half].to_vec(),
                    bytes: None,
                    o: cfg.moe_inter,
                    i: cfg.hidden,
                };
                let up = Wt {
                    f: gu[half..].to_vec(),
                    bytes: None,
                    o: cfg.moe_inter,
                    i: cfg.hidden,
                };
                let down = Wt {
                    f: dn,
                    bytes: None,
                    o: cfg.hidden,
                    i: cfg.moe_inter,
                };
                layer_experts.push([gate, up, down]);
            }
            experts.push(layer_experts);
            layers.push(layer);
        }

        // PLE geometry
        let mut ple_offsets = Vec::new();
        let mut ple_sizes = Vec::new();
        if cfg.ple_layer >= 0 && cfg.ngram_heads > 0 {
            let mut total = 0_i64;
            for h in 0..cfg.ngram_heads {
                let size = nth_prime_after(cfg.ngram_vocab_base - 1, h as i64 + 1);
                ple_sizes.push(size);
                ple_offsets.push(total);
                total += size;
            }
        }
        let ple_embed: Wt = if cfg.ple_layer >= 0 && cfg.ngram_heads > 0 {
            let total: i64 = ple_sizes.iter().sum();
            let padded = (total + cfg.ngram_div - 1) / cfg.ngram_div * cfg.ngram_div;
            let hd_per = cfg.ple_embed_dim / cfg.ngram_heads;
            load_wt(
                st,
                "model.ple.ple_embedding.ngram_embedding.weight",
                padded as usize,
                hd_per,
            )?
        } else {
            Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            }
        };
        let ple_key_proj = if cfg.ple_layer >= 0 {
            load_wt(
                st,
                "model.ple.key_proj.weight",
                cfg.hc_count * cfg.hidden,
                cfg.ple_embed_dim,
            )?
        } else {
            Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            }
        };
        let ple_value_proj = if cfg.ple_layer >= 0 {
            load_wt(
                st,
                "model.ple.value_proj.weight",
                cfg.hidden,
                cfg.ple_embed_dim,
            )?
        } else {
            Wt {
                f: vec![],
                bytes: None,
                o: 0,
                i: 0,
            }
        };
        let hcd = cfg.hc_count * cfg.hidden;
        let ple_norm_key = if cfg.ple_layer >= 0 {
            st.f32("model.ple.norm_key.weight", &[hcd as u64])?
        } else {
            vec![]
        };
        let ple_norm_query = if cfg.ple_layer >= 0 {
            st.f32("model.ple.norm_query.weight", &[hcd as u64])?
        } else {
            vec![]
        };
        let ple_norm_conv = if cfg.ple_layer >= 0 {
            st.f32("model.ple.norm_conv.weight", &[hcd as u64])?
        } else {
            vec![]
        };
        let ple_conv1d = if cfg.ple_layer >= 0 {
            st.f32(
                "model.ple.conv1d.weight",
                &[(hcd * cfg.ple_conv_kernel) as u64],
            )?
        } else {
            vec![]
        };
        // ple_mult: odd multipliers from splitmix64
        let mut ple_mult = Vec::new();
        if cfg.ple_layer >= 0 && cfg.ngram_heads > 0 {
            let max_long = i64::MAX;
            let mult_max = max_long / (cfg.vocab.max(1) as i64);
            let half = (mult_max / 2).max(1);
            let base = cfg.seed as u64 + 10007_u64.wrapping_mul(cfg.ple_layer as u64);
            for i in 0..cfg.ngram_size {
                let v = ple_splitmix64(base.wrapping_add(PLE_GAMMA.wrapping_mul((i + 1) as u64)));
                ple_mult.push(2 * (v % half as u64) + 1);
            }
        }

        Ok(Model {
            cfg: cfg.clone(),
            coli: None,
            gguf: None,
            gdn_v_tiled: false,
            rope_interleaved: false,
            embed: load_wt(st, "model.embed_tokens.weight", cfg.vocab, cfg.hidden)?,
            lm_head: load_wt(st, "lm_head.weight", cfg.vocab, cfg.hidden)?,
            lm_head_aligned: None,
            lm_head_metal_tensor: 0,
            // qwen4 drops norm.weight when hyper connections are active
            final_norm: match st.f32("model.norm.weight", &[cfg.hidden as u64]) {
                Ok(v) => v,
                Err(_) if cfg.hc_count > 0 => vec![],
                Err(e) => return Err(e),
            },
            layers,
            experts,
            hc_global: HcGlobal {
                norm: st.f32("model.hyper_connection_mixer.hc_norm.weight", &[hcd as u64])?,
                mix_down: load_wt(
                    st,
                    "model.hyper_connection_mixer.input_mix_weight_down.weight",
                    cfg.hc_lowrank,
                    hcd,
                )?,
                mix_up: load_wt(
                    st,
                    "model.hyper_connection_mixer.input_mix_weight_up.weight",
                    hcd,
                    cfg.hc_lowrank,
                )?,
            },
            ple_ngram: ple_embed,
            ple_key_proj,
            ple_value_proj,
            ple_norm_key,
            ple_norm_query,
            ple_norm_conv,
            ple_conv1d,
            ple_offsets,
            ple_sizes,
            ple_mult,
            gdn_conv: cfg
                .gdn_layers
                .iter()
                .map(|&is_gdn| {
                    if is_gdn {
                        vec![0.0; (cdim_total(cfg)) * cfg.conv_kernel.saturating_sub(1)]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            gdn_s: cfg
                .gdn_layers
                .iter()
                .map(|&is_gdn| {
                    if is_gdn {
                        vec![0.0; cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            kv_k: cfg
                .gdn_layers
                .iter()
                .map(|&is_gdn| {
                    if is_gdn {
                        Vec::new()
                    } else {
                        lazy_zeroed_f32(cfg.kv_heads * cfg.max_t * cfg.head_dim)
                    }
                })
                .collect(),
            kv_v: cfg
                .gdn_layers
                .iter()
                .map(|&is_gdn| {
                    if is_gdn {
                        Vec::new()
                    } else {
                        lazy_zeroed_f32(cfg.kv_heads * cfg.max_t * cfg.head_dim)
                    }
                })
                .collect(),
            idx_cache: cfg
                .qsa_layers
                .iter()
                .map(|&is_qsa| {
                    if is_qsa {
                        lazy_zeroed_f32(cfg.max_t * cfg.idx_kv_heads * cfg.idx_head_dim)
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            ple_ring: vec![cfg.eos; cfg.ngram_size.max(1)],
            ple_conv_state: vec![
                0.0;
                hcd * ((cfg.ple_conv_kernel - 1) * cfg.ngram_size + 1).max(1)
            ],
            expert_plan: None,
            expert_store: make_expert_store(cfg.layers, cfg.topk),
            spans: logan_core::telemetry::TokenSpans::default(),
            route_prev: (0..cfg.layers).map(|_| Vec::new()).collect(),
            route_overlap_common: vec![0; cfg.layers],
            route_overlap_total: vec![0; cfg.layers],
            route_overlap_pairs: vec![0; cfg.layers],
            metal_model_id: next_metal_model_id(),
            // safetensors mode: no package profile, so the Apple8 direct path
            // never applies (C parity: direct requires the Apple8 target
            // profile). Keep the env gate for symmetry; the path is inert
            // without a .coli package anyway (moe_token requires coli).
            metal_direct: crate::ffi::direct_init()
                && std::env::var("QWEN_APPLE8_DIRECT")
                    .map(|v| v != "0")
                    .unwrap_or(true),
            metal_overlap: std::env::var("QWEN_APPLE8_OVERLAP")
                .map(|v| v != "0")
                .unwrap_or(true),
            gdn_metal: (0..cfg.layers).map(|_| None).collect(),
            gdn_ane: (0..cfg.layers)
                .map(|_| gdn_ane::GdnAneState::default())
                .collect(),
            attn_metal: (0..cfg.layers).map(|_| None).collect(),
            sched_mode: false,
            sched_blocked: None,
            sched_pause: None,
        })
    }
}

fn cdim_total(cfg: &Cfg) -> usize {
    cfg.lin_k_dim * cfg.lin_k_heads * 2 + cfg.lin_v_dim * cfg.lin_v_heads
}

/// Standalone runner for `coli run`: loads a .coli package (or safetensors
/// fixture dir) and greedy-decodes `max_new` tokens from `prompt`.
/// Returns the generated token ids (prompt excluded).
pub fn run_greedy(
    package_dir: &std::path::Path,
    prompt: &[u32],
    max_new: usize,
) -> Result<Vec<u32>, String> {
    let cfg = load_cfg(&package_dir.join("config.json"))?;
    let model = if package_dir.join("model.safetensors").exists() {
        let st = StFile::open(&package_dir.join("model.safetensors"))?;
        Model::load(&st, &cfg)?
    } else {
        let src = colisource::ColiSource::open(package_dir)?;
        Model::load_coli(&src, &cfg)?
    };
    Ok(run_greedy_with(model, cfg, prompt, max_new))
}

/// Greedy decode against an already-loaded model.
pub fn run_greedy_with(mut model: Model, _cfg: Cfg, prompt: &[u32], max_new: usize) -> Vec<u32> {
    let profile = logan_core::telemetry::enabled();
    let t0 = std::time::Instant::now();
    if prompt.is_empty() || max_new == 0 {
        return Vec::new();
    }

    // The final prompt forward already returns the logits that predict token
    // prompt.len(). Refeeding prompt.last() at that position duplicates the
    // final prompt token in recurrent/KV state and is not causal-LM decode.
    let mut logits = Vec::new();
    for (i, &t) in prompt.iter().enumerate() {
        logits = model.forward_token(t as usize, i);
    }

    let mut out = Vec::with_capacity(max_new);
    for step in 0..max_new {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        out.push(next);
        if step + 1 < max_new {
            logits = model.forward_token(next as usize, prompt.len() + step);
        }
    }
    if profile {
        model.profile_summary(max_new, t0.elapsed().as_secs_f64() * 1e3);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        causal_conv1d_sample, default_cache_cap_for_ram, quantize_bf16_to_mxfp4, rmsnorm_row,
        rmsnorm_row_shifted, StFile, MAX_RESIDENT_PLE_NGRAM_BYTES,
    };

    #[test]
    fn giant_ple_ngram_safetensor_is_refused_without_reading_payload() {
        use std::io::Write as _;

        let name = "model.ple.ple_embedding.ngram_embedding.weight";
        let payload_len = MAX_RESIDENT_PLE_NGRAM_BYTES + 4;
        let elems = payload_len / 4;
        let header = serde_json::to_vec(&serde_json::json!({
            name: {
                "dtype": "F32",
                "shape": [elems],
                "data_offsets": [0, payload_len]
            }
        }))
        .unwrap();
        let path = std::env::temp_dir().join(format!(
            "logan-ngram-residency-{}-{}.safetensors",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        // Sparse extension: if StFile::open ever reverts to std::fs::read(),
        // this regression test becomes expensive/fails instead of silently
        // accepting whole-file residency.
        file.set_len(8 + header.len() as u64 + payload_len as u64)
            .unwrap();
        drop(file);

        let st = StFile::open(&path).unwrap();
        let err = st.f32(name, &[elems as u64]).unwrap_err();
        assert!(err.contains("refusing to materialize"), "{err}");
        let _ = std::fs::remove_file(path);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn qsa_four_mxfp4_projections_match_scalar_and_mixed_format_declines() {
        use super::{Wt, WtBytes, matmul_mxfp4_multi, matmul_mxfp4_bytes};
        crate::ffi::metal_init();
        let x: Vec<f32> = (0..64).map(|i| ((i % 7) as f32 - 3.0) / 8.0).collect();
        let weights: Vec<Wt> = [(64, 0x22), (32, 0x44), (32, 0xaa), (16, 0xcc)]
            .into_iter().map(|(o, code)| Wt {
                f: vec![], o, i: 64,
                bytes: Some(WtBytes::Mxfp4 {
                    weights: vec![code; o * 32], scales: vec![127; o * 2],
                    metal_tensor: std::sync::Mutex::new(0),
                }),
            }).collect();
        let mut actual: Vec<Vec<f32>> = weights.iter().map(|w| vec![0.0; w.o]).collect();
        let refs: Vec<&Wt> = weights.iter().collect();
        assert!(matmul_mxfp4_multi(
            &mut actual.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>(), &x, &refs,
        ));
        for (w, got) in weights.iter().zip(&actual) {
            let WtBytes::Mxfp4 { weights, scales, .. } = w.bytes.as_ref().unwrap() else { unreachable!() };
            let mut expected = vec![0.0; w.o];
            matmul_mxfp4_bytes(&mut expected, &x, weights, scales, w.o, w.i);
            assert_eq!(*got, expected);
        }
        let bf16 = Wt { f: vec![], bytes: Some(WtBytes::Bf16(vec![0; 16 * 64 * 2])), o: 16, i: 64 };
        let before = actual.clone();
        assert!(!matmul_mxfp4_multi(
            &mut actual.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>(), &x,
            &[&weights[0], &weights[1], &weights[2], &bf16],
        ));
        assert_eq!(actual, before);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn expert_cache_leaves_uma_headroom_on_16g_apple_silicon() {
        const GIB: u64 = 1024 * 1024 * 1024;
        assert_eq!(default_cache_cap_for_ram(Some(16 * GIB)), 128);
        assert_eq!(default_cache_cap_for_ram(Some(32 * GIB)), 256);
        assert_eq!(default_cache_cap_for_ram(None), 256);
    }

    #[test]
    fn ple_causal_conv_uses_standard_tap_order() {
        // history[0] is t-1, history[1] is t-2. For kernel [1,10,100],
        // standard causal Conv1d computes 1*x[t-2] + 10*x[t-1] + 100*x[t].
        let history = [4.0_f32, 3.0, 2.0];
        let weights = [1.0_f32, 10.0, 100.0];
        assert_eq!(causal_conv1d_sample(5.0, &history, &weights, 1), 543.0);
    }

    #[test]
    fn runtime_mxfp4_quantizer_matches_canonical_nibble_order() {
        let values = [
            0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let mut bf16 = Vec::with_capacity(values.len() * 2);
        for value in values { bf16.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes()); }
        let (weights, scales) = quantize_bf16_to_mxfp4(&bf16, 1, values.len()).unwrap();
        assert_eq!(scales, vec![127]);
        assert_eq!(weights, vec![0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]);
    }

    #[test]
    fn mlx_qwen35_shifted_norm_matches_raw_hf_delta_norm() {
        // MLX sanitize() folds the Transformers `1 + weight` into the saved
        // Qwen3.5/3.6 norm tensor. Applying another +1 at runtime is the bug
        // that caused the real MXFP4 checkpoint to generate garbage.
        let x = [3.0_f32, -4.0, 1.5, -0.25];
        let raw_hf_delta = [0.25_f32, -0.5, 0.125, 0.75];
        let mlx_stored = [1.25_f32, 0.5, 1.125, 1.75];
        let mut raw_out = [0.0_f32; 4];
        let mut mlx_out = [0.0_f32; 4];
        rmsnorm_row(&mut raw_out, &x, &raw_hf_delta, 1e-6);
        rmsnorm_row_shifted(&mut mlx_out, &x, &mlx_stored, 1e-6);
        assert_eq!(raw_out, mlx_out);
    }
}

