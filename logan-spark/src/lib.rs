//! Spark-X2.5 resident inference runtime.
//!
//! The first production path deliberately mirrors MLX-LM's useful execution
//! choices for this architecture: keep affine-8 weights quantized, use fused
//! QKV, batch independent projections onto one Metal command buffer, allocate
//! full KV in chunks, and use fixed rotating KV for sliding-attention layers.

use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use serde_json::Value;

const AFFINE_GROUP: usize = 64;
const KV_GROW_TOKENS: usize = 256;
static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayerType {
    Full,
    Sliding,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub sliding_window: usize,
    pub max_context: usize,
    pub rms_eps: f32,
    layer_types: Vec<LayerType>,
    full_rope_dim: usize,
    full_theta: f32,
    sliding_rope_dim: usize,
    sliding_theta: f32,
}

pub fn load_cfg(path: &Path) -> Result<Config, String> {
    let v: Value = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if v.get("model_type").and_then(Value::as_str) != Some("spark2_5") {
        return Err("model_type is not spark2_5".into());
    }
    let n = |k: &str| -> Result<usize, String> {
        v.get(k)
            .and_then(Value::as_u64)
            .map(|x| x as usize)
            .ok_or_else(|| format!("missing/invalid {k}"))
    };
    let hidden = n("hidden_size")?;
    let intermediate = n("intermediate_size")?;
    let layers = n("num_hidden_layers")?;
    let heads = n("num_attention_heads")?;
    let kv_heads = n("num_key_value_heads")?;
    let head_dim = n("head_dim")?;
    if heads == 0 || kv_heads == 0 || heads % kv_heads != 0 || head_dim == 0 {
        return Err("invalid Spark attention geometry".into());
    }
    let layer_values = v
        .get("layer_types")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing layer_types".to_string())?;
    if layer_values.len() != layers {
        return Err("layer_types length mismatch".into());
    }
    let layer_types = layer_values
        .iter()
        .map(|x| match x.as_str() {
            Some("full_attention") => Ok(LayerType::Full),
            Some("sliding_attention") => Ok(LayerType::Sliding),
            other => Err(format!("unsupported Spark layer type {other:?}")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rope = v
        .get("rope_parameters")
        .ok_or_else(|| "missing rope_parameters".to_string())?;
    let rope_cfg = |kind: &str| -> Result<(usize, f32), String> {
        let r = rope
            .get(kind)
            .ok_or_else(|| format!("missing {kind} rope config"))?;
        let factor = r
            .get("partial_rotary_factor")
            .and_then(Value::as_f64)
            .unwrap_or(1.0) as f32;
        let theta = r
            .get("rope_theta")
            .and_then(Value::as_f64)
            .unwrap_or(10_000.0) as f32;
        let dim = (head_dim as f32 * factor) as usize;
        if dim == 0 || dim > head_dim || dim % 2 != 0 {
            return Err(format!("invalid {kind} rotary dim {dim}"));
        }
        Ok((dim, theta))
    };
    let (full_rope_dim, full_theta) = rope_cfg("full_attention")?;
    let (sliding_rope_dim, sliding_theta) = rope_cfg("sliding_attention")?;
    let quant = v
        .get("quantization_config")
        .or_else(|| v.get("quantization"))
        .ok_or_else(|| {
            "Spark resident runtime currently expects an MLX quantization config".to_string()
        })?;
    if quant.get("mode").and_then(Value::as_str) != Some("affine")
        || quant.get("bits").and_then(Value::as_u64) != Some(8)
        || quant.get("group_size").and_then(Value::as_u64) != Some(AFFINE_GROUP as u64)
    {
        return Err(
            "Spark resident runtime currently supports MLX affine 8-bit group_size=64".into(),
        );
    }
    Ok(Config {
        hidden,
        intermediate,
        layers,
        heads,
        kv_heads,
        head_dim,
        vocab: n("vocab_size")?,
        sliding_window: n("sliding_window")?,
        max_context: n("max_position_embeddings")?,
        rms_eps: v
            .get("rms_norm_eps")
            .and_then(Value::as_f64)
            .unwrap_or(1e-6) as f32,
        layer_types,
        full_rope_dim,
        full_theta,
        sliding_rope_dim,
        sliding_theta,
    })
}

#[derive(Debug)]
struct TensorStore {
    tensors: HashMap<String, Vec<u8>>,
}

impl TensorStore {
    fn open(root: &Path) -> Result<Self, String> {
        if root.join("manifest.coli").is_file() {
            Self::from_coli(root)
        } else {
            Self::from_safetensors(root)
        }
    }

    fn from_coli(root: &Path) -> Result<Self, String> {
        let pkg = logan_format::package::Package::open(root).map_err(|e| e.to_string())?;
        let mut tensors = HashMap::with_capacity(pkg.records().len());
        for rec in pkg.records() {
            if rec.kind != 1 {
                continue;
            }
            let Some(name) = rec.name.as_ref() else {
                continue;
            };
            let bytes = pkg.read_tensor_payload(rec).map_err(|e| e.to_string())?;
            tensors.insert(name.clone(), bytes);
        }
        Ok(Self { tensors })
    }

    fn from_safetensors(root: &Path) -> Result<Self, String> {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(root)
            .map_err(|e| e.to_string())?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(format!("{}: no safetensors files", root.display()));
        }
        let mut tensors = HashMap::new();
        for path in shards {
            read_safetensor_shard(&path, &mut tensors)?;
        }
        Ok(Self { tensors })
    }

    fn take(&mut self, name: &str) -> Result<Vec<u8>, String> {
        self.tensors
            .remove(name)
            .ok_or_else(|| format!("missing tensor {name}"))
    }
}

fn read_safetensor_shard(path: &Path, dst: &mut HashMap<String, Vec<u8>>) -> Result<(), String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut nbuf = [0u8; 8];
    file.read_exact(&mut nbuf).map_err(|e| e.to_string())?;
    let hlen = u64::from_le_bytes(nbuf) as usize;
    let mut hb = vec![0u8; hlen];
    file.read_exact(&mut hb).map_err(|e| e.to_string())?;
    let header: Value = serde_json::from_slice(&hb).map_err(|e| e.to_string())?;
    let obj = header
        .as_object()
        .ok_or_else(|| "invalid safetensors header".to_string())?;
    let data_start = 8u64 + hlen as u64;
    let mut spans = Vec::new();
    for (name, desc) in obj {
        if name == "__metadata__" {
            continue;
        }
        let offs = desc
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{name}: missing offsets"))?;
        if offs.len() != 2 {
            return Err(format!("{name}: invalid offsets"));
        }
        let a = offs[0]
            .as_u64()
            .ok_or_else(|| format!("{name}: invalid offset"))?;
        let b = offs[1]
            .as_u64()
            .ok_or_else(|| format!("{name}: invalid offset"))?;
        if b < a {
            return Err(format!("{name}: reversed offsets"));
        }
        spans.push((a, b, name.clone()));
    }
    spans.sort_by_key(|x| x.0);
    for (a, b, name) in spans {
        file.seek(SeekFrom::Start(data_start + a))
            .map_err(|e| e.to_string())?;
        let mut bytes = vec![0u8; (b - a) as usize];
        file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        if dst.insert(name.clone(), bytes).is_some() {
            return Err(format!("duplicate tensor {name}"));
        }
    }
    Ok(())
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
fn bf16_at(bytes: &[u8], idx: usize) -> f32 {
    let o = idx * 2;
    bf16(u16::from_le_bytes([bytes[o], bytes[o + 1]]))
}
fn f32_to_bf16(x: f32) -> f32 {
    let u = x.to_bits();
    let rounded = u.wrapping_add(0x7fff + ((u >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
}
fn round_bf16(v: &mut [f32]) {
    for x in v {
        *x = f32_to_bf16(*x);
    }
}

struct AffineQ8 {
    w: Vec<u8>,
    aux: Vec<u8>, // BF16 scales followed by BF16 biases
    o: usize,
    i: usize,
    metal_tensor: Mutex<usize>,
}

impl Drop for AffineQ8 {
    fn drop(&mut self) {
        let raw = *self
            .metal_tensor
            .get_mut()
            .unwrap_or_else(|p| p.into_inner());
        if raw != 0 {
            unsafe { logan_metal::coli_metal_tensor_free(raw as *mut logan_metal::ColiMetalTensor) }
        }
    }
}

impl AffineQ8 {
    fn load(store: &mut TensorStore, base: &str, o: usize, i: usize) -> Result<Self, String> {
        if i % AFFINE_GROUP != 0 {
            return Err(format!(
                "{base}: input dim {i} not divisible by {AFFINE_GROUP}"
            ));
        }
        let w = store.take(&format!("{base}.weight"))?;
        let scales = store.take(&format!("{base}.scales"))?;
        let biases = store.take(&format!("{base}.biases"))?;
        let ng = i / AFFINE_GROUP;
        if w.len() != o * i {
            return Err(format!("{base}: weight bytes {} != {}", w.len(), o * i));
        }
        if scales.len() != o * ng * 2 || biases.len() != scales.len() {
            return Err(format!(
                "{base}: scale/bias bytes do not match [{o},{ng}] BF16"
            ));
        }
        let mut aux = Vec::with_capacity(scales.len() + biases.len());
        aux.extend_from_slice(&scales);
        aux.extend_from_slice(&biases);
        Ok(Self {
            w,
            aux,
            o,
            i,
            metal_tensor: Mutex::new(0),
        })
    }

    fn row(&self, row: usize, out: &mut [f32]) {
        assert!(row < self.o && out.len() >= self.i);
        let ng = self.i / AFFINE_GROUP;
        let wr = &self.w[row * self.i..(row + 1) * self.i];
        let scale_base = row * ng;
        let bias_base = self.o * ng + row * ng;
        for g in 0..ng {
            let sc = bf16_at(&self.aux, scale_base + g);
            let bi = bf16_at(&self.aux, bias_base + g);
            let start = g * AFFINE_GROUP;
            for j in 0..AFFINE_GROUP {
                out[start + j] = f32_to_bf16(wr[start + j] as f32 * sc + bi);
            }
        }
    }

    fn matmul(&self, x: &[f32], y: &mut [f32]) {
        assert!(x.len() >= self.i && y.len() >= self.o);
        if logan_metal::metal_available() {
            let mut h = self.metal_tensor.lock().unwrap_or_else(|p| p.into_inner());
            let mut ptr = *h as *mut logan_metal::ColiMetalTensor;
            if logan_metal::metal_matmul(&mut ptr, y, x, &self.w, &self.aux, 15, self.i, self.o) {
                *h = ptr as usize;
                round_bf16(&mut y[..self.o]);
                return;
            }
        }
        self.matmul_cpu(x, y);
    }

    fn matmul_cpu(&self, x: &[f32], y: &mut [f32]) {
        let ng = self.i / AFFINE_GROUP;
        let mut xsum = vec![0.0f32; ng];
        for g in 0..ng {
            xsum[g] = x[g * AFFINE_GROUP..(g + 1) * AFFINE_GROUP].iter().sum();
        }
        for row in 0..self.o {
            let wr = &self.w[row * self.i..(row + 1) * self.i];
            let mut acc = 0.0f32;
            for g in 0..ng {
                let sc = bf16_at(&self.aux, row * ng + g);
                let bi = bf16_at(&self.aux, self.o * ng + row * ng + g);
                let start = g * AFFINE_GROUP;
                let mut dot = 0.0f32;
                for j in 0..AFFINE_GROUP {
                    dot += wr[start + j] as f32 * x[start + j];
                }
                acc += sc * dot + bi * xsum[g];
            }
            y[row] = f32_to_bf16(acc);
        }
    }
}

fn matmul_pair(a: &AffineQ8, b: &AffineQ8, x: &[f32], ya: &mut [f32], yb: &mut [f32]) {
    if a.i == b.i && logan_metal::metal_available() {
        let mut ha = a.metal_tensor.lock().unwrap_or_else(|p| p.into_inner());
        let mut hb = b.metal_tensor.lock().unwrap_or_else(|p| p.into_inner());
        let mut descs = [
            logan_metal::MetalMatmulDesc {
                tensor: *ha as *mut _,
                y: ya,
                weights: &a.w,
                scales: &a.aux,
                fmt: 15,
                i: a.i,
                o: a.o,
            },
            logan_metal::MetalMatmulDesc {
                tensor: *hb as *mut _,
                y: yb,
                weights: &b.w,
                scales: &b.aux,
                fmt: 15,
                i: b.i,
                o: b.o,
            },
        ];
        if logan_metal::metal_matmul_multi(x, &mut descs) {
            *ha = descs[0].tensor as usize;
            *hb = descs[1].tensor as usize;
            round_bf16(ya);
            round_bf16(yb);
            return;
        }
    }
    a.matmul_cpu(x, ya);
    b.matmul_cpu(x, yb);
}

struct Layer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    qkv: AffineQ8,
    gate: AffineQ8,
    out: AffineQ8,
    mlp_gate: AffineQ8,
    mlp_up: AffineQ8,
    mlp_down: AffineQ8,
    kind: LayerType,
}

impl Layer {
    fn load(store: &mut TensorStore, cfg: &Config, li: usize) -> Result<Self, String> {
        let p = format!("model.layers.{li}");
        let qdim = cfg.heads * cfg.head_dim;
        let kvdim = cfg.kv_heads * cfg.head_dim;
        Ok(Self {
            input_norm: load_bf16_vec(store, &format!("{p}.input_layernorm.weight"), cfg.hidden)?,
            post_norm: load_bf16_vec(
                store,
                &format!("{p}.post_attention_layernorm.weight"),
                cfg.hidden,
            )?,
            qkv: AffineQ8::load(
                store,
                &format!("{p}.self_attn.q_k_v_proj"),
                qdim + 2 * kvdim,
                cfg.hidden,
            )?,
            gate: AffineQ8::load(
                store,
                &format!("{p}.self_attn.g_proj"),
                cfg.heads,
                cfg.hidden,
            )?,
            out: AffineQ8::load(store, &format!("{p}.self_attn.out_proj"), cfg.hidden, qdim)?,
            mlp_gate: AffineQ8::load(
                store,
                &format!("{p}.mlp.gate_proj"),
                cfg.intermediate,
                cfg.hidden,
            )?,
            mlp_up: AffineQ8::load(
                store,
                &format!("{p}.mlp.up_proj"),
                cfg.intermediate,
                cfg.hidden,
            )?,
            mlp_down: AffineQ8::load(
                store,
                &format!("{p}.mlp.down_proj"),
                cfg.hidden,
                cfg.intermediate,
            )?,
            kind: cfg.layer_types[li],
        })
    }

    fn metal_token(
        &self,
        model_id: u64,
        li: usize,
        x: &mut [f32],
        cfg: &Config,
        pos: usize,
        encode_only: bool,
    ) -> i32 {
        let mut h0 = self
            .qkv
            .metal_tensor
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut h1 = self
            .gate
            .metal_tensor
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut h2 = self
            .out
            .metal_tensor
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut h3 = self
            .mlp_gate
            .metal_tensor
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut h4 = self
            .mlp_up
            .metal_tensor
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut h5 = self
            .mlp_down
            .metal_tensor
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut descs = [
            logan_metal::MetalWeightDesc {
                tensor: *h0 as *mut _,
                weights: &self.qkv.w,
                scales: &self.qkv.aux,
                fmt: 15,
                i: self.qkv.i,
                o: self.qkv.o,
            },
            logan_metal::MetalWeightDesc {
                tensor: *h1 as *mut _,
                weights: &self.gate.w,
                scales: &self.gate.aux,
                fmt: 15,
                i: self.gate.i,
                o: self.gate.o,
            },
            logan_metal::MetalWeightDesc {
                tensor: *h2 as *mut _,
                weights: &self.out.w,
                scales: &self.out.aux,
                fmt: 15,
                i: self.out.i,
                o: self.out.o,
            },
            logan_metal::MetalWeightDesc {
                tensor: *h3 as *mut _,
                weights: &self.mlp_gate.w,
                scales: &self.mlp_gate.aux,
                fmt: 15,
                i: self.mlp_gate.i,
                o: self.mlp_gate.o,
            },
            logan_metal::MetalWeightDesc {
                tensor: *h4 as *mut _,
                weights: &self.mlp_up.w,
                scales: &self.mlp_up.aux,
                fmt: 15,
                i: self.mlp_up.i,
                o: self.mlp_up.o,
            },
            logan_metal::MetalWeightDesc {
                tensor: *h5 as *mut _,
                weights: &self.mlp_down.w,
                scales: &self.mlp_down.aux,
                fmt: 15,
                i: self.mlp_down.i,
                o: self.mlp_down.o,
            },
        ];
        let (rd, theta) = match self.kind {
            LayerType::Full => (cfg.full_rope_dim, cfg.full_theta),
            LayerType::Sliding => (cfg.sliding_rope_dim, cfg.sliding_theta),
        };
        let rc = if encode_only {
            logan_metal::spark_layer_encode(
                model_id,
                li,
                &mut descs,
                &self.input_norm,
                &self.post_norm,
                cfg.hidden,
                cfg.intermediate,
                cfg.heads,
                cfg.kv_heads,
                cfg.head_dim,
                self.kind == LayerType::Sliding,
                cfg.sliding_window,
                pos,
                rd,
                theta,
                cfg.rms_eps,
            )
        } else {
            logan_metal::spark_layer(
                model_id,
                li,
                &mut descs,
                x,
                &self.input_norm,
                &self.post_norm,
                cfg.hidden,
                cfg.intermediate,
                cfg.heads,
                cfg.kv_heads,
                cfg.head_dim,
                self.kind == LayerType::Sliding,
                cfg.sliding_window,
                pos,
                rd,
                theta,
                cfg.rms_eps,
            )
        };
        *h0 = descs[0].tensor as usize;
        *h1 = descs[1].tensor as usize;
        *h2 = descs[2].tensor as usize;
        *h3 = descs[3].tensor as usize;
        *h4 = descs[4].tensor as usize;
        *h5 = descs[5].tensor as usize;
        rc
    }

    fn metal_prefill(
        &self,
        model_id: u64,
        li: usize,
        cfg: &Config,
        base: usize,
        srows: usize,
    ) -> i32 {
        let mut h0 = self
            .qkv
            .metal_tensor
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut h1 = self
            .gate
            .metal_tensor
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut h2 = self
            .out
            .metal_tensor
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut h3 = self
            .mlp_gate
            .metal_tensor
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut h4 = self
            .mlp_up
            .metal_tensor
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut h5 = self
            .mlp_down
            .metal_tensor
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut descs = [
            logan_metal::MetalWeightDesc {
                tensor: *h0 as *mut _,
                weights: &self.qkv.w,
                scales: &self.qkv.aux,
                fmt: 15,
                i: self.qkv.i,
                o: self.qkv.o,
            },
            logan_metal::MetalWeightDesc {
                tensor: *h1 as *mut _,
                weights: &self.gate.w,
                scales: &self.gate.aux,
                fmt: 15,
                i: self.gate.i,
                o: self.gate.o,
            },
            logan_metal::MetalWeightDesc {
                tensor: *h2 as *mut _,
                weights: &self.out.w,
                scales: &self.out.aux,
                fmt: 15,
                i: self.out.i,
                o: self.out.o,
            },
            logan_metal::MetalWeightDesc {
                tensor: *h3 as *mut _,
                weights: &self.mlp_gate.w,
                scales: &self.mlp_gate.aux,
                fmt: 15,
                i: self.mlp_gate.i,
                o: self.mlp_gate.o,
            },
            logan_metal::MetalWeightDesc {
                tensor: *h4 as *mut _,
                weights: &self.mlp_up.w,
                scales: &self.mlp_up.aux,
                fmt: 15,
                i: self.mlp_up.i,
                o: self.mlp_up.o,
            },
            logan_metal::MetalWeightDesc {
                tensor: *h5 as *mut _,
                weights: &self.mlp_down.w,
                scales: &self.mlp_down.aux,
                fmt: 15,
                i: self.mlp_down.i,
                o: self.mlp_down.o,
            },
        ];
        let (rd, theta) = match self.kind {
            LayerType::Full => (cfg.full_rope_dim, cfg.full_theta),
            LayerType::Sliding => (cfg.sliding_rope_dim, cfg.sliding_theta),
        };
        let rc = logan_metal::spark_prefill_layer_encode(
            model_id,
            li,
            &mut descs,
            &self.input_norm,
            &self.post_norm,
            cfg.hidden,
            cfg.intermediate,
            cfg.heads,
            cfg.kv_heads,
            cfg.head_dim,
            self.kind == LayerType::Sliding,
            cfg.sliding_window,
            base,
            srows,
            rd,
            theta,
            cfg.rms_eps,
        );
        *h0 = descs[0].tensor as usize;
        *h1 = descs[1].tensor as usize;
        *h2 = descs[2].tensor as usize;
        *h3 = descs[3].tensor as usize;
        *h4 = descs[4].tensor as usize;
        *h5 = descs[5].tensor as usize;
        rc
    }
}

fn load_bf16_vec(store: &mut TensorStore, name: &str, n: usize) -> Result<Vec<f32>, String> {
    let b = store.take(name)?;
    if b.len() != n * 2 {
        return Err(format!(
            "{name}: expected {} BF16 bytes, got {}",
            n * 2,
            b.len()
        ));
    }
    Ok((0..n).map(|i| bf16_at(&b, i)).collect())
}

#[derive(Debug)]
enum KvCache {
    Full {
        k: Vec<u16>,
        v: Vec<u16>,
        tokens: usize,
    },
    Sliding {
        k: Vec<u16>,
        v: Vec<u16>,
        tokens: usize,
        window: usize,
    },
}

impl KvCache {
    fn new(kind: LayerType, window: usize) -> Self {
        match kind {
            LayerType::Full => Self::Full {
                k: Vec::new(),
                v: Vec::new(),
                tokens: 0,
            },
            LayerType::Sliding => Self::Sliding {
                k: Vec::new(),
                v: Vec::new(),
                tokens: 0,
                window,
            },
        }
    }
    fn push(&mut self, krow: &[f32], vrow: &[f32]) {
        let d = krow.len();
        assert_eq!(d, vrow.len());
        match self {
            Self::Full { k, v, tokens } => {
                if k.len() + d > k.capacity() {
                    k.reserve_exact(KV_GROW_TOKENS * d);
                    v.reserve_exact(KV_GROW_TOKENS * d);
                }
                k.extend(
                    krow.iter()
                        .map(|&x| (f32_to_bf16(x).to_bits() >> 16) as u16),
                );
                v.extend(
                    vrow.iter()
                        .map(|&x| (f32_to_bf16(x).to_bits() >> 16) as u16),
                );
                *tokens += 1;
            }
            Self::Sliding {
                k,
                v,
                tokens,
                window,
            } => {
                if k.is_empty() {
                    k.resize(*window * d, 0);
                    v.resize(*window * d, 0);
                }
                let slot = *tokens % *window;
                let off = slot * d;
                for j in 0..d {
                    k[off + j] = (f32_to_bf16(krow[j]).to_bits() >> 16) as u16;
                    v[off + j] = (f32_to_bf16(vrow[j]).to_bits() >> 16) as u16;
                }
                *tokens += 1;
            }
        }
    }
    fn len(&self) -> usize {
        match self {
            Self::Full { tokens, .. } => *tokens,
            Self::Sliding { tokens, window, .. } => (*tokens).min(*window),
        }
    }
    fn token_at(&self, logical: usize, d: usize) -> (&[u16], &[u16]) {
        match self {
            Self::Full { k, v, .. } => (
                &k[logical * d..(logical + 1) * d],
                &v[logical * d..(logical + 1) * d],
            ),
            Self::Sliding {
                k,
                v,
                tokens,
                window,
            } => {
                let idx = logical % *window;
                let off = idx * d;
                let _ = tokens;
                (&k[off..off + d], &v[off..off + d])
            }
        }
    }
    fn logical_start(&self) -> usize {
        match self {
            Self::Full { .. } => 0,
            Self::Sliding { tokens, window, .. } => tokens.saturating_sub(*window),
        }
    }
}

pub struct Model {
    model_id: u64,
    metal_layers: bool,
    metal_token_chain: bool,
    cfg: Config,
    embedding: AffineQ8,
    norm: Vec<f32>,
    layers: Vec<Layer>,
    cache: Vec<KvCache>,
}

impl Model {
    pub fn load(root: &Path) -> Result<Self, String> {
        let cfg = load_cfg(&root.join("config.json"))?;
        let _ = logan_metal::metal_init();
        let mut store = TensorStore::open(root)?;
        let embedding = AffineQ8::load(&mut store, "model.embedding", cfg.vocab, cfg.hidden)?;
        let norm = load_bf16_vec(&mut store, "model.norm.weight", cfg.hidden)?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for li in 0..cfg.layers {
            layers.push(Layer::load(&mut store, &cfg, li)?);
        }
        let cache = cfg
            .layer_types
            .iter()
            .map(|&k| KvCache::new(k, cfg.sliding_window))
            .collect();
        let model_id = NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed).max(1);
        // The chained Metal path is the production default on Apple Silicon.
        // Set LOGAN_SPARK_CHAINED=0 only to debug/fall back to the older path.
        let metal_token_chain = logan_metal::metal_available()
            && std::env::var("LOGAN_SPARK_CHAINED").ok().as_deref() != Some("0");
        let metal_layers = metal_token_chain
            || (logan_metal::metal_available()
                && std::env::var("LOGAN_SPARK_LAYER_FAST").ok().as_deref() == Some("1"));
        Ok(Self {
            model_id,
            metal_layers,
            metal_token_chain,
            cfg,
            embedding,
            norm,
            layers,
            cache,
        })
    }
    pub fn context_limit(&self) -> usize {
        self.cfg.max_context
    }

    pub fn prefill_tokens(&mut self, tokens: &[u32], base: usize) -> Vec<f32> {
        let prefill_chunk = std::env::var("LOGAN_SPARK_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 2)
            .unwrap_or(1024);
        assert!(!tokens.is_empty());
        assert!(base + tokens.len() <= self.cfg.max_context);
        if self.metal_token_chain && tokens.len() > 1 {
            let mut logits = Vec::new();
            let mut off = 0usize;
            while off < tokens.len() {
                let n = (tokens.len() - off).min(prefill_chunk);
                // Small-M matrix prefill is not yet parity-clean on all real
                // prompts and underutilizes the GPU anyway. Use the proven chained
                // token path below 64 rows; matrix prefill starts where its SIMD
                // tiles are both useful and validated.
                if n < 64 {
                    for j in 0..n {
                        logits = self.forward_token(tokens[off + j] as usize, base + off + j);
                    }
                } else if let Some(batch_logits) = self.prefill_tokens_metal(
                    &tokens[off..off + n],
                    base + off,
                    off + n == tokens.len(),
                ) {
                    if !batch_logits.is_empty() {
                        logits = batch_logits;
                    }
                } else {
                    // prefill_tokens_metal aborts before commit on a declined batch,
                    // so falling back to the proven token path is safe.
                    for j in 0..n {
                        logits = self.forward_token(tokens[off + j] as usize, base + off + j);
                    }
                }
                off += n;
            }
            return logits;
        }
        let mut logits = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            logits = self.forward_token(tok as usize, base + i);
        }
        logits
    }

    fn prefill_tokens_metal(
        &mut self,
        tokens: &[u32],
        base: usize,
        need_logits: bool,
    ) -> Option<Vec<f32>> {
        let srows = tokens.len();
        let d = self.cfg.hidden;
        let mut x = vec![0.0f32; srows * d];
        for (r, &tok) in tokens.iter().enumerate() {
            if tok as usize >= self.cfg.vocab {
                return None;
            }
            self.embedding.row(tok as usize, &mut x[r * d..(r + 1) * d]);
        }
        if !logan_metal::spark_prefill_begin(self.model_id, &x, srows, d, base) {
            return None;
        }
        for (li, layer) in self.layers.iter().enumerate() {
            let rc = layer.metal_prefill(self.model_id, li, &self.cfg, base, srows);
            if rc <= 0 {
                logan_metal::spark_prefill_abort(self.model_id);
                return None;
            }
        }
        if !need_logits {
            let rc = logan_metal::spark_prefill_end(self.model_id, base, srows);
            if rc <= 0 {
                logan_metal::spark_prefill_abort(self.model_id);
                return None;
            }
            return Some(Vec::new());
        }
        let mut logits = vec![0.0f32; self.cfg.vocab];
        let mut head_handle = self
            .embedding
            .metal_tensor
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut head = logan_metal::MetalWeightDesc {
            tensor: *head_handle as *mut _,
            weights: &self.embedding.w,
            scales: &self.embedding.aux,
            fmt: 15,
            i: self.embedding.i,
            o: self.embedding.o,
        };
        let rc = logan_metal::spark_prefill_end_logits(
            self.model_id,
            &mut head,
            &self.norm,
            &mut logits,
            d,
            self.cfg.vocab,
            base,
            srows,
            self.cfg.rms_eps,
        );
        *head_handle = head.tensor as usize;
        if rc <= 0 {
            logan_metal::spark_prefill_abort(self.model_id);
            return None;
        }
        Some(logits)
    }

    pub fn forward_token_top1(&mut self, token: usize, pos: usize) -> u32 {
        if !self.metal_token_chain {
            let logits = self.forward_token(token, pos);
            return logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap();
        }
        assert!(token < self.cfg.vocab && pos < self.cfg.max_context);
        let d = self.cfg.hidden;
        let mut x = vec![0.0f32; d];
        self.embedding.row(token, &mut x);
        assert!(logan_metal::spark_token_begin(self.model_id, &x, d, pos));
        for (li, layer) in self.layers.iter().enumerate() {
            let rc = layer.metal_token(self.model_id, li, &mut x, &self.cfg, pos, true);
            if rc <= 0 {
                logan_metal::spark_token_abort(self.model_id);
                panic!("Spark Metal layer encode {li} failed at position {pos} (rc={rc})");
            }
        }
        let mut hh = self
            .embedding
            .metal_tensor
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut head = logan_metal::MetalWeightDesc {
            tensor: *hh as *mut _,
            weights: &self.embedding.w,
            scales: &self.embedding.aux,
            fmt: 15,
            i: self.embedding.i,
            o: self.embedding.o,
        };
        let mut next = 0u32;
        let rc = logan_metal::spark_token_end_top1(
            self.model_id,
            &mut head,
            &self.norm,
            &mut next,
            d,
            self.cfg.vocab,
            pos,
            self.cfg.rms_eps,
        );
        *hh = head.tensor as usize;
        assert!(
            rc > 0,
            "Spark Metal token+top1 end failed at position {pos} (rc={rc})"
        );
        next
    }

    pub fn forward_token(&mut self, token: usize, pos: usize) -> Vec<f32> {
        if self.metal_layers {
            return self.forward_token_metal(token, pos);
        }
        self.forward_token_cpu(token, pos)
    }

    fn forward_token_metal(&mut self, token: usize, pos: usize) -> Vec<f32> {
        assert!(token < self.cfg.vocab && pos < self.cfg.max_context);
        let d = self.cfg.hidden;
        let mut x = vec![0.0f32; d];
        self.embedding.row(token, &mut x);
        if self.metal_token_chain {
            assert!(
                logan_metal::spark_token_begin(self.model_id, &x, d, pos),
                "Spark Metal token begin failed at position {pos}"
            );
            for (li, layer) in self.layers.iter().enumerate() {
                let rc = layer.metal_token(self.model_id, li, &mut x, &self.cfg, pos, true);
                if rc <= 0 {
                    logan_metal::spark_token_abort(self.model_id);
                    panic!("Spark Metal layer encode {li} failed at position {pos} (rc={rc})");
                }
            }
            let mut logits = vec![0.0f32; self.cfg.vocab];
            let mut head_handle = self
                .embedding
                .metal_tensor
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let mut head = logan_metal::MetalWeightDesc {
                tensor: *head_handle as *mut _,
                weights: &self.embedding.w,
                scales: &self.embedding.aux,
                fmt: 15,
                i: self.embedding.i,
                o: self.embedding.o,
            };
            let rc = logan_metal::spark_token_end_logits(
                self.model_id,
                &mut head,
                &self.norm,
                &mut logits,
                d,
                self.cfg.vocab,
                pos,
                self.cfg.rms_eps,
            );
            *head_handle = head.tensor as usize;
            assert!(
                rc > 0,
                "Spark Metal token+head end failed at position {pos} (rc={rc})"
            );
            return logits;
        } else {
            for (li, layer) in self.layers.iter().enumerate() {
                let rc = layer.metal_token(self.model_id, li, &mut x, &self.cfg, pos, false);
                assert!(
                    rc > 0,
                    "Spark Metal layer {li} failed at position {pos} (rc={rc})"
                );
            }
        }
        let mut xn = vec![0.0; d];
        rmsnorm(&x, &self.norm, self.cfg.rms_eps, &mut xn);
        let mut logits = vec![0.0; self.cfg.vocab];
        self.embedding.matmul(&xn, &mut logits);
        logits
    }

    fn forward_token_cpu(&mut self, token: usize, pos: usize) -> Vec<f32> {
        assert!(token < self.cfg.vocab && pos < self.cfg.max_context);
        let d = self.cfg.hidden;
        let hd = self.cfg.head_dim;
        let qdim = self.cfg.heads * hd;
        let kvdim = self.cfg.kv_heads * hd;
        let mut x = vec![0.0f32; d];
        self.embedding.row(token, &mut x);
        for li in 0..self.layers.len() {
            let layer = &self.layers[li];
            let mut xn = vec![0.0; d];
            rmsnorm(&x, &layer.input_norm, self.cfg.rms_eps, &mut xn);
            let mut qkv = vec![0.0; qdim + 2 * kvdim];
            let mut gates = vec![0.0; self.cfg.heads];
            matmul_pair(&layer.qkv, &layer.gate, &xn, &mut qkv, &mut gates);
            let (q, rest) = qkv.split_at_mut(qdim);
            let (k, v) = rest.split_at_mut(kvdim);
            let (rd, theta) = match layer.kind {
                LayerType::Full => (self.cfg.full_rope_dim, self.cfg.full_theta),
                LayerType::Sliding => (self.cfg.sliding_rope_dim, self.cfg.sliding_theta),
            };
            for h in 0..self.cfg.heads {
                rope(&mut q[h * hd..(h + 1) * hd], pos, rd, theta);
            }
            for h in 0..self.cfg.kv_heads {
                rope(&mut k[h * hd..(h + 1) * hd], pos, rd, theta);
            }
            round_bf16(q);
            round_bf16(k);
            round_bf16(v);
            self.cache[li].push(k, v);
            let mut att = attention_decode(q, &self.cache[li], &self.cfg);
            for h in 0..self.cfg.heads {
                let g = f32_to_bf16(1.0 / (1.0 + (-gates[h]).exp()));
                for z in &mut att[h * hd..(h + 1) * hd] {
                    *z = f32_to_bf16(*z * g);
                }
            }
            let mut ao = vec![0.0; d];
            layer.out.matmul(&att, &mut ao);
            for j in 0..d {
                x[j] = f32_to_bf16(x[j] + ao[j]);
            }
            let mut pn = vec![0.0; d];
            rmsnorm(&x, &layer.post_norm, self.cfg.rms_eps, &mut pn);
            let mut mg = vec![0.0; self.cfg.intermediate];
            let mut mu = vec![0.0; self.cfg.intermediate];
            matmul_pair(&layer.mlp_gate, &layer.mlp_up, &pn, &mut mg, &mut mu);
            for j in 0..self.cfg.intermediate {
                let z = mg[j];
                let gelu = z * (1.0 + libm::erff(z / std::f32::consts::SQRT_2)) * 0.5;
                mg[j] = f32_to_bf16(f32_to_bf16(gelu) * mu[j]);
            }
            let mut mo = vec![0.0; d];
            layer.mlp_down.matmul(&mg, &mut mo);
            for j in 0..d {
                x[j] = f32_to_bf16(x[j] + mo[j]);
            }
        }
        let mut xn = vec![0.0; d];
        rmsnorm(&x, &self.norm, self.cfg.rms_eps, &mut xn);
        let mut logits = vec![0.0; self.cfg.vocab];
        self.embedding.matmul(&xn, &mut logits);
        logits
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        logan_metal::spark_drop_model(self.model_id);
    }
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ms = x.iter().map(|&z| z * z).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for i in 0..x.len() {
        out[i] = f32_to_bf16(x[i] * inv * w[i]);
    }
}
fn rope(x: &mut [f32], pos: usize, rd: usize, theta: f32) {
    let half = rd / 2;
    for j in 0..half {
        let inv = theta.powf(-2.0 * j as f32 / rd as f32);
        let a = pos as f32 * inv;
        let (sn, cs) = a.sin_cos();
        let u = x[j];
        let v = x[j + half];
        x[j] = u * cs - v * sn;
        x[j + half] = v * cs + u * sn;
    }
}
fn attention_decode(q: &[f32], cache: &KvCache, cfg: &Config) -> Vec<f32> {
    let hd = cfg.head_dim;
    let kvd = cfg.kv_heads * hd;
    let rep = cfg.heads / cfg.kv_heads;
    let n = cache.len();
    let start = cache.logical_start();
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; cfg.heads * hd];
    let mut scores = vec![0.0f32; n];
    for h in 0..cfg.heads {
        let kh = h / rep;
        let qh = &q[h * hd..(h + 1) * hd];
        let mut mx = f32::NEG_INFINITY;
        for t in 0..n {
            let (k, _) = cache.token_at(start + t, kvd);
            let kk = &k[kh * hd..(kh + 1) * hd];
            let mut s = 0.0;
            for j in 0..hd {
                s += qh[j] * bf16(kk[j]);
            }
            s *= scale;
            scores[t] = s;
            mx = mx.max(s);
        }
        let mut den = 0.0;
        for s in &mut scores[..n] {
            *s = (*s - mx).exp();
            den += *s;
        }
        let den = den.max(f32::MIN_POSITIVE);
        for t in 0..n {
            let (_, v) = cache.token_at(start + t, kvd);
            let vv = &v[kh * hd..(kh + 1) * hd];
            let a = scores[t] / den;
            for j in 0..hd {
                out[h * hd + j] += a * bf16(vv[j]);
            }
        }
    }
    round_bf16(&mut out);
    out
}

pub fn run_greedy(root: &Path, prompt: &[u32], max_new: usize) -> Result<Vec<u32>, String> {
    if prompt.is_empty() {
        return Err("prompt must contain at least one token".into());
    }
    let mut model = Model::load(root)?;
    let mut logits = model.prefill_tokens(prompt, 0);
    let mut out = Vec::with_capacity(max_new);
    for step in 0..max_new {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .ok_or("empty logits")?;
        out.push(next);
        logits = model.forward_token(next as usize, prompt.len() + step);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bf16_round_trip() {
        for x in [0.0, 1.0, -2.5, 123.25] {
            let y = f32_to_bf16(x);
            assert_eq!(bf16((y.to_bits() >> 16) as u16), y);
        }
    }
    #[test]
    fn sliding_cache_retains_only_the_latest_window() {
        let mut cache = KvCache::new(LayerType::Sliding, 2);
        cache.push(&[1.0, 2.0], &[11.0, 12.0]);
        cache.push(&[3.0, 4.0], &[13.0, 14.0]);
        cache.push(&[5.0, 6.0], &[15.0, 16.0]);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.logical_start(), 1);
        let (k1, v1) = cache.token_at(1, 2);
        assert_eq!((bf16(k1[0]), bf16(k1[1])), (3.0, 4.0));
        assert_eq!((bf16(v1[0]), bf16(v1[1])), (13.0, 14.0));
        let (k2, v2) = cache.token_at(2, 2);
        assert_eq!((bf16(k2[0]), bf16(k2[1])), (5.0, 6.0));
        assert_eq!((bf16(v2[0]), bf16(v2[1])), (15.0, 16.0));
    }

    #[test]
    fn affine_cpu_matches_definition() {
        let o = 2;
        let i = 64;
        let mut aux = Vec::new();
        for _ in 0..o {
            aux.extend_from_slice(&1.0f32.to_bits().to_be_bytes()[0..2]);
        } // overwritten below with canonical bf16 bits
        aux.clear();
        for _ in 0..o {
            aux.extend_from_slice(&(0x3f80u16).to_le_bytes());
        }
        for _ in 0..o {
            aux.extend_from_slice(&(0u16).to_le_bytes());
        }
        let m = AffineQ8 {
            w: vec![2; o * i],
            aux,
            o,
            i,
            metal_tensor: Mutex::new(0),
        };
        let x = vec![1.0; i];
        let mut y = vec![0.0; o];
        m.matmul_cpu(&x, &mut y);
        assert_eq!(y, vec![128.0, 128.0]);
    }
}
