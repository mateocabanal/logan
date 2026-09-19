use crate::{
    config::load_config,
    inspect_weights,
    kv::{KvCache, KvCheckpoint},
    metal::{self, BackendPreference, BackendReport, BackendUsed},
    DType, LlamaConfig, TensorInfo,
};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIdentity {
    pub model_type: String,
    pub dtype: Option<DType>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub layers: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct QuantizedPayload {
    pub(crate) weights: Vec<u8>,
    pub(crate) aux: Vec<u8>,
    pub(crate) bits: u8,
    pub(crate) group_size: usize,
}

#[derive(Debug, Clone)]
pub struct DenseTensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
    pub(crate) quantized: Option<QuantizedPayload>,
}
impl DenseTensor {
    pub fn from_f32(dtype: DType, shape: Vec<usize>, values: &[f32]) -> Result<Self, String> {
        let expected = shape.iter().copied().product::<usize>();
        if expected != values.len() {
            return Err(format!(
                "tensor has {} values, expected {expected}",
                values.len()
            ));
        }
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for &v in values {
            let bits = match dtype {
                DType::BF16 => (v.to_bits() >> 16) as u16,
                DType::F16 => f32_to_f16(v),
            };
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        Ok(Self {
            dtype,
            shape,
            bytes,
            quantized: None,
        })
    }
}

#[derive(Debug, Clone)]
struct Matrix {
    dtype: DType,
    rows: usize,
    cols: usize,
    values: Vec<f32>,
    bytes: Vec<u8>,
    quantized: Option<QuantizedPayload>,
    metal_tensor: metal::SharedMetalTensorHandle,
}

#[derive(Debug, Clone)]
struct Weights {
    embedding: Matrix,
    head: Matrix,
    final_norm: Vec<f32>,
    layers: Vec<LayerWeights>,
}
#[derive(Debug, Clone)]
struct LayerWeights {
    input_norm: Vec<f32>,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    o: Matrix,
    post_norm: Vec<f32>,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
}

/// Resident dense Llama/MiniCPM5 model. Matrices are decoded once and retained
/// for the lifetime of this object; BF16/F16 source bytes are retained for the
/// optional compatible Metal dense entry point.
#[derive(Debug, Clone)]
pub struct DenseModel {
    pub config: LlamaConfig,
    pub root: Option<PathBuf>,
    identity: ModelIdentity,
    weights: Arc<Weights>,
}

impl DenseModel {
    pub fn load<P: AsRef<Path>>(root: P) -> Result<Self, String> {
        let root = root.as_ref();
        let config = load_config(root.join("config.json"))?;
        Self::load_with_config(root, &config)
    }
    fn has_mlx_quantization_metadata(root: &Path) -> bool {
        if [
            "mlx_quantization.json",
            "quantization.json",
            "quantization_config.json",
        ]
        .iter()
        .any(|name| root.join(name).is_file())
        {
            return true;
        }
        let config = match fs::read(root.join("config.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        {
            Some(config) => config,
            None => return false,
        };
        ["quantization", "quantization_config"].iter().any(|key| {
            config
                .get(key)
                .and_then(serde_json::Value::as_object)
                .is_some()
        })
    }

    pub fn load_with_config<P: AsRef<Path>>(root: P, config: &LlamaConfig) -> Result<Self, String> {
        let root = root.as_ref();
        if Self::has_mlx_quantization_metadata(root) {
            let tensors = crate::weights::load_quantized_tensors(root, config)?;
            return Self::from_tensors_inner(config.clone(), tensors, Some(root.to_owned()));
        }
        let inventory = inspect_weights(root, config)?;
        let mut tensors = BTreeMap::new();
        for (name, info) in &inventory.tensors {
            tensors.insert(name.clone(), read_tensor(info)?);
        }
        Self::from_tensors_inner(config.clone(), tensors, Some(root.to_owned()))
    }

    /// Build a resident model from canonical safetensors names. This is useful
    /// for deterministic small reference models and does not alter checkpoint
    /// loading semantics.
    pub fn from_tensors(
        config: LlamaConfig,
        tensors: BTreeMap<String, DenseTensor>,
    ) -> Result<Self, String> {
        Self::from_tensors_inner(config, tensors, None)
    }

    pub fn identity(&self) -> &ModelIdentity {
        &self.identity
    }
    pub fn new_session(self: &Arc<Self>) -> DenseSession {
        DenseSession::new(self.clone())
    }
    pub fn session(&self) -> DenseSession {
        DenseSession::new(Arc::new(self.clone()))
    }

    fn from_tensors_inner(
        config: LlamaConfig,
        tensors: BTreeMap<String, DenseTensor>,
        root: Option<PathBuf>,
    ) -> Result<Self, String> {
        let get = |names: &[String]| -> Result<DenseTensor, String> {
            for n in names {
                if let Some(t) = tensors.get(n) {
                    return Ok(t.clone());
                }
            }
            Err(format!("missing tensor (tried {})", names.join(", ")))
        };
        let matrix =
            |t: DenseTensor, rows: usize, cols: usize, role: &str| -> Result<Matrix, String> {
                if t.shape != [rows, cols] {
                    return Err(format!(
                        "{role} shape {:?}, expected [{rows}, {cols}]",
                        t.shape
                    ));
                }
                decode_matrix(t)
            };
        let vecf = |t: DenseTensor, n: usize, role: &str| -> Result<Vec<f32>, String> {
            if t.shape != [n] {
                return Err(format!("{role} shape {:?}, expected [{n}]", t.shape));
            }
            decode_values(&t)
        };
        let embedding = matrix(
            get(&[
                "model.embed_tokens.weight".into(),
                "embed_tokens.weight".into(),
                "model.tok_embeddings.weight".into(),
                "tok_embeddings.weight".into(),
            ])?,
            config.vocab_size as usize,
            config.hidden_size as usize,
            "embedding",
        )?;
        let head = matrix(
            get(&["lm_head.weight".into(), "model.lm_head.weight".into()])?,
            config.vocab_size as usize,
            config.hidden_size as usize,
            "lm_head",
        )?;
        let final_norm = vecf(
            get(&["model.norm.weight".into(), "norm.weight".into()])?,
            config.hidden_size as usize,
            "final norm",
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for layer in 0..config.num_hidden_layers {
            let prefixes = [format!("model.layers.{layer}."), format!("layers.{layer}.")];
            let name = |suffix: &str| {
                prefixes
                    .iter()
                    .map(|p| format!("{p}{suffix}"))
                    .collect::<Vec<_>>()
            };
            layers.push(LayerWeights {
                input_norm: vecf(
                    get(&name("input_layernorm.weight"))?,
                    config.hidden_size as usize,
                    "input norm",
                )?,
                q: matrix(
                    get(&name("self_attn.q_proj.weight"))?,
                    config.num_attention_heads as usize * config.head_dim as usize,
                    config.hidden_size as usize,
                    "q",
                )?,
                k: matrix(
                    get(&name("self_attn.k_proj.weight"))?,
                    config.num_key_value_heads as usize * config.head_dim as usize,
                    config.hidden_size as usize,
                    "k",
                )?,
                v: matrix(
                    get(&name("self_attn.v_proj.weight"))?,
                    config.num_key_value_heads as usize * config.head_dim as usize,
                    config.hidden_size as usize,
                    "v",
                )?,
                o: matrix(
                    get(&name("self_attn.o_proj.weight"))?,
                    config.hidden_size as usize,
                    config.num_attention_heads as usize * config.head_dim as usize,
                    "o",
                )?,
                post_norm: vecf(
                    get(&name("post_attention_layernorm.weight"))?,
                    config.hidden_size as usize,
                    "post norm",
                )?,
                gate: matrix(
                    get(&name("mlp.gate_proj.weight"))?,
                    config.intermediate_size as usize,
                    config.hidden_size as usize,
                    "gate",
                )?,
                up: matrix(
                    get(&name("mlp.up_proj.weight"))?,
                    config.intermediate_size as usize,
                    config.hidden_size as usize,
                    "up",
                )?,
                down: matrix(
                    get(&name("mlp.down_proj.weight"))?,
                    config.hidden_size as usize,
                    config.intermediate_size as usize,
                    "down",
                )?,
            });
        }
        let dtype = tensors.values().next().map(|x| x.dtype);
        Ok(Self {
            identity: ModelIdentity {
                model_type: config.model_type.clone(),
                dtype,
                vocab_size: config.vocab_size as usize,
                hidden_size: config.hidden_size as usize,
                layers: config.num_hidden_layers as usize,
            },
            config,
            root,
            weights: Arc::new(Weights {
                embedding,
                head,
                final_norm,
                layers,
            }),
        })
    }
}

#[derive(Debug, Clone)]
pub struct LayerTap {
    pub layer: usize,
    pub residual: Vec<f32>,
    pub rows: usize,
    pub width: usize,
}

#[derive(Debug, Clone)]
pub struct ForwardOutput {
    pub logits: Vec<f32>,
    pub rows: usize,
    pub vocab_size: usize,
    pub processed_tokens: usize,
    pub taps: BTreeMap<usize, Vec<f32>>,
    pub backend: BackendReport,
}
impl ForwardOutput {
    pub fn logits_row(&self, row: usize) -> Option<&[f32]> {
        self.logits
            .get(row * self.vocab_size..(row + 1) * self.vocab_size)
    }
    pub fn tap(&self, layer: usize) -> Option<&[f32]> {
        self.taps.get(&layer).map(Vec::as_slice)
    }
}
fn fused_matrix_parts<'a>(matrix: &'a Matrix) -> (i32, &'a [u8], &'a [u8]) {
    // Native affine-8 QMV is exact, but currently slower than BF16 GEMV on
    // this workload. Keep it opt-in until a faster kernel wins the gate.
    let native =
        std::env::var_os("LOGAN_DENSE_NATIVE").as_deref() == Some(std::ffi::OsStr::new("1"));
    if native {
        if let Some(payload) = matrix
            .quantized
            .as_ref()
            .filter(|payload| payload.bits == 8 && payload.group_size == 64)
        {
            return (15, payload.weights.as_slice(), payload.aux.as_slice());
        }
    }
    (5, matrix.bytes.as_slice(), &[])
}

#[derive(Debug)]
pub struct DenseSession {
    model: Arc<DenseModel>,
    kv: KvCache,
    backend: BackendPreference,
    last_backend: BackendReport,
    gpu_model_id: u64,
}
static NEXT_GPU_MODEL_ID: AtomicU64 = AtomicU64::new(1);

impl Clone for DenseSession {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            kv: self.kv.clone(),
            backend: self.backend,
            last_backend: self.last_backend.clone(),
            gpu_model_id: NEXT_GPU_MODEL_ID.fetch_add(1, Ordering::Relaxed).max(1),
        }
    }
}

impl Drop for DenseSession {
    fn drop(&mut self) {
        metal::llama_drop_model(self.gpu_model_id);
    }
}

impl DenseSession {
    pub fn new(model: Arc<DenseModel>) -> Self {
        Self::new_with_backend(model, BackendPreference::Auto)
    }

    /// Construct a session with an explicit backend preference.
    ///
    /// Metal and the direct dense entry point are process-wide, once-only
    /// initializations. CPU sessions deliberately skip both initializers.
    pub fn new_with_backend(model: Arc<DenseModel>, backend: BackendPreference) -> Self {
        initialize_dense_backend(backend);
        let c = &model.config;
        Self {
            kv: KvCache::new(
                c.num_hidden_layers as usize,
                c.num_key_value_heads as usize * c.head_dim as usize,
            ),
            backend,
            last_backend: BackendReport::cpu(backend, "no forward executed"),
            model,
            gpu_model_id: NEXT_GPU_MODEL_ID.fetch_add(1, Ordering::Relaxed).max(1),
        }
    }

    pub fn model(&self) -> &DenseModel {
        &self.model
    }
    pub fn kv(&self) -> &KvCache {
        &self.kv
    }
    pub fn checkpoint(&self) -> KvCheckpoint {
        self.kv.checkpoint()
    }
    pub fn snapshot(&self) -> KvCheckpoint {
        self.checkpoint()
    }
    pub fn restore(&mut self, checkpoint: KvCheckpoint) -> Result<(), String> {
        metal::llama_drop_model(self.gpu_model_id);
        self.kv.restore(checkpoint).map_err(|e| e.to_string())
    }
    pub fn restore_snapshot(&mut self, checkpoint: KvCheckpoint) -> Result<(), String> {
        self.restore(checkpoint)
    }
    pub fn retain_verified(&mut self, checkpoint: KvCheckpoint) -> Result<(), String> {
        metal::llama_drop_model(self.gpu_model_id);
        self.kv
            .retain_verified(checkpoint)
            .map_err(|e| e.to_string())
    }
    pub fn is_current_checkpoint(&self, checkpoint: KvCheckpoint) -> bool {
        self.kv.is_current(checkpoint)
    }
    pub fn set_backend(&mut self, backend: BackendPreference) {
        metal::llama_drop_model(self.gpu_model_id);
        initialize_dense_backend(backend);
        self.backend = backend;
    }
    pub(crate) fn backend_preference(&self) -> BackendPreference {
        self.backend
    }
    pub fn backend(&self) -> BackendReport {
        self.last_backend.clone()
    }
    pub fn reset(&mut self) {
        metal::llama_drop_model(self.gpu_model_id);
        self.kv.clear();
    }
    fn forward_fused_decode(&mut self, tokens: &[u32]) -> Option<Result<ForwardOutput, String>> {
        if tokens.len() != 1 {
            return None;
        }
        let c = &self.model.config;
        let d = c.hidden_size as usize;
        let hd = c.head_dim as usize;
        let h = c.num_attention_heads as usize;
        let kh = c.num_key_value_heads as usize;
        let inter = c.intermediate_size as usize;
        let qdim = h * hd;
        let kw = kh * hd;
        let pos = self.kv.processed_tokens();
        let mut hidden = self.model.weights.embedding.values
            [tokens[0] as usize * d..(tokens[0] as usize + 1) * d]
            .to_vec();
        let mut staged_k = vec![Vec::<f32>::new(); c.num_hidden_layers as usize];
        let mut staged_v = vec![Vec::<f32>::new(); c.num_hidden_layers as usize];
        let mut report = BackendReport {
            requested: self.backend,
            used: BackendUsed::Metal,
            attempted: true,
            available: true,
            reason: "fused standard Llama layer",
        };
        let mut q_sink = vec![0.0; qdim];
        let mut k_sink = vec![0.0; kw];
        let mut v_sink = vec![0.0; kw];
        let mut o_sink = vec![0.0; d];
        let mut gate_sink = vec![0.0; inter];
        let mut up_sink = vec![0.0; inter];
        let mut down_sink = vec![0.0; d];
        for (li, layer) in self.model.weights.layers.iter().enumerate() {
            let mut q_handle = metal::lock_handle(&layer.q.metal_tensor);
            let mut k_handle = metal::lock_handle(&layer.k.metal_tensor);
            let mut v_handle = metal::lock_handle(&layer.v.metal_tensor);
            let mut o_handle = metal::lock_handle(&layer.o.metal_tensor);
            let mut gate_handle = metal::lock_handle(&layer.gate.metal_tensor);
            let mut up_handle = metal::lock_handle(&layer.up.metal_tensor);
            let mut down_handle = metal::lock_handle(&layer.down.metal_tensor);
            let (q_fmt, q_weights, q_scales) = fused_matrix_parts(&layer.q);
            let (k_fmt, k_weights, k_scales) = fused_matrix_parts(&layer.k);
            let (v_fmt, v_weights, v_scales) = fused_matrix_parts(&layer.v);
            let (o_fmt, o_weights, o_scales) = fused_matrix_parts(&layer.o);
            let (gate_fmt, gate_weights, gate_scales) = fused_matrix_parts(&layer.gate);
            let (up_fmt, up_weights, up_scales) = fused_matrix_parts(&layer.up);
            let (down_fmt, down_weights, down_scales) = fused_matrix_parts(&layer.down);
            let mut descs = [
                logan_metal::MetalMatmulDesc {
                    tensor: q_handle.raw(),
                    y: &mut q_sink,
                    weights: q_weights,
                    scales: q_scales,
                    fmt: q_fmt,
                    i: d,
                    o: qdim,
                },
                logan_metal::MetalMatmulDesc {
                    tensor: k_handle.raw(),
                    y: &mut k_sink,
                    weights: k_weights,
                    scales: k_scales,
                    fmt: k_fmt,
                    i: d,
                    o: kw,
                },
                logan_metal::MetalMatmulDesc {
                    tensor: v_handle.raw(),
                    y: &mut v_sink,
                    weights: v_weights,
                    scales: v_scales,
                    fmt: v_fmt,
                    i: d,
                    o: kw,
                },
                logan_metal::MetalMatmulDesc {
                    tensor: o_handle.raw(),
                    y: &mut o_sink,
                    weights: o_weights,
                    scales: o_scales,
                    fmt: o_fmt,
                    i: qdim,
                    o: d,
                },
                logan_metal::MetalMatmulDesc {
                    tensor: gate_handle.raw(),
                    y: &mut gate_sink,
                    weights: gate_weights,
                    scales: gate_scales,
                    fmt: gate_fmt,
                    i: d,
                    o: inter,
                },
                logan_metal::MetalMatmulDesc {
                    tensor: up_handle.raw(),
                    y: &mut up_sink,
                    weights: up_weights,
                    scales: up_scales,
                    fmt: up_fmt,
                    i: d,
                    o: inter,
                },
                logan_metal::MetalMatmulDesc {
                    tensor: down_handle.raw(),
                    y: &mut down_sink,
                    weights: down_weights,
                    scales: down_scales,
                    fmt: down_fmt,
                    i: inter,
                    o: d,
                },
            ];
            let mut k_row = vec![0.0; kw];
            let mut v_row = vec![0.0; kw];
            let rc = metal::llama_layer(
                self.gpu_model_id,
                li,
                &mut descs,
                &mut hidden,
                &mut k_row,
                &mut v_row,
                &layer.input_norm,
                &layer.post_norm,
                d,
                inter,
                h,
                kh,
                hd,
                pos,
                c.rope_theta as f32,
                c.rms_norm_eps as f32,
            );
            q_handle.set_raw(descs[0].tensor);
            k_handle.set_raw(descs[1].tensor);
            v_handle.set_raw(descs[2].tensor);
            o_handle.set_raw(descs[3].tensor);
            gate_handle.set_raw(descs[4].tensor);
            up_handle.set_raw(descs[5].tensor);
            down_handle.set_raw(descs[6].tensor);
            if rc != 1 {
                metal::llama_drop_model(self.gpu_model_id);
                return None;
            }
            staged_k[li] = k_row;
            staged_v[li] = v_row;
        }
        let mut norm = vec![0.0; d];
        rms_norm_into(
            &hidden,
            &self.model.weights.final_norm,
            c.rms_norm_eps as f32,
            &mut norm,
        );
        let mut logits = vec![0.0; c.vocab_size as usize];
        linear_into(
            &self.model.weights.head,
            &norm,
            &mut logits,
            self.backend,
            &mut report,
        );
        report = BackendReport {
            requested: self.backend,
            used: BackendUsed::Metal,
            attempted: true,
            available: true,
            reason: "fused standard Llama layer",
        };
        if let Err(error) = self.kv.commit(&staged_k, &staged_v, 1) {
            metal::llama_drop_model(self.gpu_model_id);
            return Some(Err(error.to_string()));
        }
        self.last_backend = report.clone();
        Some(Ok(ForwardOutput {
            logits,
            rows: 1,
            vocab_size: c.vocab_size as usize,
            processed_tokens: self.kv.processed_tokens(),
            taps: BTreeMap::new(),
            backend: report,
        }))
    }

    /// Execute a token chunk. `tap_ids` names layers whose post-FFN residual
    /// rows are returned in `ForwardOutput::taps`.
    pub fn forward(&mut self, tokens: &[u32], tap_ids: &[usize]) -> Result<ForwardOutput, String> {
        let rows = tokens.len();
        if rows == 0 {
            return Ok(ForwardOutput {
                logits: Vec::new(),
                rows: 0,
                vocab_size: self.model.config.vocab_size as usize,
                processed_tokens: self.kv.processed_tokens(),
                taps: BTreeMap::new(),
                backend: self.last_backend.clone(),
            });
        }
        let max_position_embeddings = self.model.config.max_position_embeddings as usize;
        let vocab_size = self.model.config.vocab_size;
        if self
            .kv
            .processed_tokens()
            .checked_add(rows)
            .ok_or("token position overflow")?
            > max_position_embeddings
        {
            return Err("token chunk exceeds max_position_embeddings".into());
        }
        if tokens.iter().any(|&t| t as u64 >= vocab_size) {
            return Err("token id exceeds vocabulary".into());
        }
        if rows == 1 && tap_ids.is_empty() && !matches!(self.backend, BackendPreference::Cpu) {
            if let Some(result) = self.forward_fused_decode(tokens) {
                return result;
            }
        } else if !matches!(self.backend, BackendPreference::Cpu) {
            // The fused cache is decode-only. Any batched/tapped CPU path must
            // invalidate it before a later single-token call can try again.
            metal::llama_drop_model(self.gpu_model_id);
        }
        let c = &self.model.config;
        let mut taps = BTreeMap::new();
        for &layer in tap_ids {
            if layer >= c.num_hidden_layers as usize {
                return Err(format!("tap layer {layer} out of range"));
            }
            taps.entry(layer)
                .or_insert_with(|| vec![0.0; rows * c.hidden_size as usize]);
        }
        let d = c.hidden_size as usize;
        let hd = c.head_dim as usize;
        let h = c.num_attention_heads as usize;
        let kh = c.num_key_value_heads as usize;
        let kw = kh * hd;
        let half = hd / 2;
        let mut rope_cache = vec![0.0; rows * half * 2];
        for r in 0..rows {
            let pos = self.kv.processed_tokens() + r;
            let angles = &mut rope_cache[r * half * 2..(r + 1) * half * 2];
            for i in 0..half {
                let theta = (pos as f32) / (c.rope_theta as f32).powf((2 * i) as f32 / hd as f32);
                angles[i] = theta.cos();
                angles[half + i] = theta.sin();
            }
        }
        let mut hidden = vec![0.0; rows * d];
        for (r, &token) in tokens.iter().enumerate() {
            hidden[r * d..(r + 1) * d].copy_from_slice(
                &self.model.weights.embedding.values[token as usize * d..(token as usize + 1) * d],
            );
        }
        let mut staged_k = vec![Vec::<f32>::with_capacity(rows * kw); c.num_hidden_layers as usize];
        let mut staged_v = vec![Vec::<f32>::with_capacity(rows * kw); c.num_hidden_layers as usize];
        let mut report = BackendReport::cpu(self.backend, "CPU reference path");
        let mut norm = vec![0.0; d];
        let mut proj = vec![0.0; d];
        for (li, layer) in self.model.weights.layers.iter().enumerate() {
            let old = self.kv.layer(li).map_err(|e| e.to_string())?;
            let old_rows = old.keys.len() / kw;
            let mut q_rows = vec![0.0; rows * h * hd];
            let mut k_rows = vec![0.0; rows * kw];
            let mut v_rows = vec![0.0; rows * kw];
            for r in 0..rows {
                rms_norm_into(
                    &hidden[r * d..(r + 1) * d],
                    &layer.input_norm,
                    c.rms_norm_eps as f32,
                    &mut norm,
                );
                let mut qkv = [
                    &mut q_rows[r * h * hd..(r + 1) * h * hd],
                    &mut k_rows[r * kw..(r + 1) * kw],
                    &mut v_rows[r * kw..(r + 1) * kw],
                ];
                linear_multi_into(
                    &[&layer.q, &layer.k, &layer.v],
                    &norm,
                    &mut qkv,
                    self.backend,
                    &mut report,
                );
                let angles = &rope_cache[r * half * 2..(r + 1) * half * 2];
                rope(&mut q_rows[r * h * hd..(r + 1) * h * hd], angles, h, hd);
                rope(&mut k_rows[r * kw..(r + 1) * kw], angles, kh, hd);
            }
            let mut attn = vec![0.0; rows * h * hd];
            let mut scores = Vec::new();
            for r in 0..rows {
                let key_count = old_rows + r + 1;
                for head in 0..h {
                    let group = head / (h / kh);
                    let q = &q_rows[r * h * hd + head * hd..r * h * hd + (head + 1) * hd];
                    scores.clear();
                    scores.resize(key_count, 0.0);
                    for j in 0..key_count {
                        let k = if j < old_rows {
                            &old.keys[j * kw + group * hd..j * kw + (group + 1) * hd]
                        } else {
                            let z = j - old_rows;
                            &k_rows[z * kw + group * hd..z * kw + (group + 1) * hd]
                        };
                        scores[j] =
                            q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() / (hd as f32).sqrt();
                    }
                    softmax(&mut scores);
                    for j in 0..key_count {
                        let v = if j < old_rows {
                            &old.values[j * kw + group * hd..j * kw + (group + 1) * hd]
                        } else {
                            let z = j - old_rows;
                            &v_rows[z * kw + group * hd..z * kw + (group + 1) * hd]
                        };
                        for z in 0..hd {
                            attn[r * h * hd + head * hd + z] += scores[j] * v[z];
                        }
                    }
                }
            }
            let mut gate = vec![0.0; layer.gate.rows];
            let mut up = vec![0.0; layer.up.rows];
            let mut down = vec![0.0; d];
            for r in 0..rows {
                linear_into(
                    &layer.o,
                    &attn[r * h * hd..(r + 1) * h * hd],
                    &mut proj,
                    self.backend,
                    &mut report,
                );
                for z in 0..d {
                    hidden[r * d + z] += proj[z];
                }
                rms_norm_into(
                    &hidden[r * d..(r + 1) * d],
                    &layer.post_norm,
                    c.rms_norm_eps as f32,
                    &mut norm,
                );
                let mut gate_up = [&mut gate[..], &mut up[..]];
                linear_multi_into(
                    &[&layer.gate, &layer.up],
                    &norm,
                    &mut gate_up,
                    self.backend,
                    &mut report,
                );
                for (g, u) in gate.iter_mut().zip(up.iter()) {
                    *g = *g / (1.0 + (-*g).exp()) * *u;
                }
                linear_into(&layer.down, &gate, &mut down, self.backend, &mut report);
                for z in 0..d {
                    hidden[r * d + z] += down[z];
                }
                if let Some(t) = taps.get_mut(&li) {
                    t[r * d..(r + 1) * d].copy_from_slice(&hidden[r * d..(r + 1) * d]);
                }
            }
            staged_k[li] = k_rows;
            staged_v[li] = v_rows;
        }
        let mut logits = vec![0.0; rows * c.vocab_size as usize];
        for r in 0..rows {
            rms_norm_into(
                &hidden[r * d..(r + 1) * d],
                &self.model.weights.final_norm,
                c.rms_norm_eps as f32,
                &mut norm,
            );
            linear_into(
                &self.model.weights.head,
                &norm,
                &mut logits[r * c.vocab_size as usize..(r + 1) * c.vocab_size as usize],
                self.backend,
                &mut report,
            );
        }
        self.kv
            .commit(&staged_k, &staged_v, rows)
            .map_err(|e| e.to_string())?;
        self.last_backend = report.clone();
        Ok(ForwardOutput {
            logits,
            rows,
            vocab_size: c.vocab_size as usize,
            processed_tokens: self.kv.processed_tokens(),
            taps,
            backend: report,
        })
    }

    /// Process only the valid logical prefix of a padded host buffer. Padded
    /// rows are neither projected nor committed to the KV cache.
    pub fn forward_padded(
        &mut self,
        tokens: &[u32],
        valid_rows: usize,
        tap_ids: &[usize],
    ) -> Result<ForwardOutput, String> {
        if valid_rows > tokens.len() {
            return Err("valid_rows exceeds token buffer".into());
        }
        self.forward(&tokens[..valid_rows], tap_ids)
    }

    pub fn forward_with_valid_len(
        &mut self,
        tokens: &[u32],
        valid_rows: usize,
        tap_ids: &[usize],
    ) -> Result<ForwardOutput, String> {
        self.forward_padded(tokens, valid_rows, tap_ids)
    }
    pub fn forward_tokens(&mut self, tokens: &[u32]) -> Result<ForwardOutput, String> {
        self.forward(tokens, &[])
    }
}

fn initialize_dense_backend(backend: BackendPreference) {
    if matches!(backend, BackendPreference::Auto | BackendPreference::Metal) {
        let _ = logan_metal::metal_init();

        let _ = logan_metal::direct_init();
    }
}
fn linear_multi_into(
    matrices: &[&Matrix],
    x: &[f32],
    ys: &mut [&mut [f32]],
    requested: BackendPreference,
    report: &mut BackendReport,
) {
    if matrices.len() != ys.len() || matrices.is_empty() {
        return;
    }
    let can_batch = matches!(
        requested,
        BackendPreference::Auto | BackendPreference::Metal
    ) && matrices.iter().all(|m| m.dtype == DType::BF16);
    if can_batch {
        // The CPU fallback rewrites every output row, so native writes can
        // target the caller's buffers directly.  This removes the temporary
        // candidate vectors and copies from the per-token QKV/FFN hot path.
        let mut handles = matrices
            .iter()
            .map(|m| metal::lock_handle(&m.metal_tensor))
            .collect::<Vec<_>>();
        let mut descs = Vec::with_capacity(matrices.len());
        for ((m, y), handle) in matrices.iter().zip(ys.iter_mut()).zip(handles.iter()) {
            descs.push(logan_metal::MetalMatmulDesc {
                tensor: handle.raw(),
                y: &mut **y,
                weights: &m.bytes,
                scales: &[],
                fmt: 5,
                i: m.cols,
                o: m.rows,
            });
        }
        let r = metal::matmul_bf16_multi(requested, DType::BF16, x, &mut descs);
        for (handle, desc) in handles.iter_mut().zip(descs.iter()) {
            handle.set_raw(desc.tensor);
        }
        *report = r.clone();
        if r.used == BackendUsed::Metal {
            return;
        }
    }
    for (m, y) in matrices.iter().zip(ys.iter_mut()) {
        linear_cpu_into(m, x, y);
    }
}

fn linear_cpu_into(m: &Matrix, x: &[f32], y: &mut [f32]) {
    for row in 0..m.rows {
        y[row] = m.values[row * m.cols..(row + 1) * m.cols]
            .iter()
            .zip(x)
            .map(|(a, b)| a * b)
            .sum();
    }
}

fn linear_into(
    m: &Matrix,
    x: &[f32],
    y: &mut [f32],
    requested: BackendPreference,
    report: &mut BackendReport,
) {
    if m.dtype == DType::BF16
        && matches!(
            requested,
            BackendPreference::Auto | BackendPreference::Metal
        )
    {
        if let Some(r) = metal::bnns_bf16(requested, &m.bytes, x, y, m.rows, m.cols) {
            *report = r;
            return;
        }
        let r = metal::matmul_bf16(requested, m.dtype, &m.bytes, x, y, 1, m.rows, m.cols);
        if r.used == BackendUsed::Metal {
            *report = r;
            return;
        }
        *report = r;
    }
    for row in 0..m.rows {
        y[row] = m.values[row * m.cols..(row + 1) * m.cols]
            .iter()
            .zip(x)
            .map(|(a, b)| a * b)
            .sum();
    }
}
fn rms_norm_into(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = (mean + eps).sqrt().recip();
    for ((out, value), weight) in out.iter_mut().zip(x).zip(w) {
        *out = value * inv * weight;
    }
}
fn softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    if s != 0.0 {
        for v in x {
            *v /= s;
        }
    }
}
fn rope(x: &mut [f32], angles: &[f32], heads: usize, hd: usize) {
    let half = hd / 2;
    for head in 0..heads {
        let base = head * hd;
        for i in 0..half {
            let co = angles[i];
            let si = angles[half + i];
            let a = x[base + i];
            let b = x[base + half + i];
            x[base + i] = a * co - b * si;
            x[base + half + i] = a * si + b * co;
        }
    }
}

fn decode_matrix(t: DenseTensor) -> Result<Matrix, String> {
    let rows = t.shape[0];
    let cols = t.shape[1];
    let values = decode_values(&t)?;
    Ok(Matrix {
        dtype: t.dtype,
        rows,
        cols,
        values,
        bytes: t.bytes,
        quantized: t.quantized,
        metal_tensor: Arc::new(Mutex::new(metal::MetalTensorHandle::default())),
    })
}
fn decode_values(t: &DenseTensor) -> Result<Vec<f32>, String> {
    let n = t.shape.iter().copied().product::<usize>();
    if t.bytes.len() != n * 2 {
        return Err(format!("tensor byte length {} != {}", t.bytes.len(), n * 2));
    }
    Ok(t.bytes
        .chunks_exact(2)
        .map(|p| {
            let b = u16::from_le_bytes([p[0], p[1]]);
            match t.dtype {
                DType::BF16 => f32::from_bits((b as u32) << 16),
                DType::F16 => f16_to_f32(b),
            }
        })
        .collect())
}
fn read_tensor(info: &TensorInfo) -> Result<DenseTensor, String> {
    let mut f = File::open(&info.shard).map_err(|e| e.to_string())?;
    f.seek(SeekFrom::Start(info.offset))
        .map_err(|e| e.to_string())?;
    let mut bytes = vec![0; info.len as usize];
    f.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    Ok(DenseTensor {
        dtype: info.dtype,
        shape: info.shape.iter().map(|&x| x as usize).collect(),
        bytes,
        quantized: None,
    })
}
fn f16_to_f32(bits: u16) -> f32 {
    let s = ((bits >> 15) & 1) as u32;
    let e = ((bits >> 10) & 0x1f) as u32;
    let f = (bits & 0x3ff) as u32;
    let out = if e == 0 {
        if f == 0 {
            s << 31
        } else {
            let mut frac = f;
            let mut exp = -14i32;
            while frac & 0x400 == 0 {
                frac <<= 1;
                exp -= 1;
            }
            (s << 31) | (((exp + 127) as u32) << 23) | ((frac & 0x3ff) << 13)
        }
    } else if e == 31 {
        (s << 31) | (0xff << 23) | (f << 13)
    } else {
        (s << 31) | ((e + 112) << 23) | (f << 13)
    };
    f32::from_bits(out)
}
fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let f = (b >> 13) & 0x3ff;
    if e <= 0 {
        if e < -10 {
            s
        } else {
            s | (((b & 0x7fffff) | 0x800000) >> (14 - e)) as u16
        }
    } else if e >= 31 {
        s | 0x7c00
    } else {
        s | ((e as u16) << 10) | f as u16
    }
}
