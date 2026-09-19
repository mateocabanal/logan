//! Dense Llama-family source frontend.
//!
//! This frontend validates the raw Hugging Face/SafeTensors vocabulary shared by
//! standard Llama checkpoints and MiniCPM5. It stops at source metadata and
//! deterministic lowering views without dequantizing or rewriting packed
//! quantized payloads.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::{
    error::{ColicError, Result},
    model::qwen_moe::mlx_affine_dtype,
    source::{self, SourceInventory, TensorRef},
};

/// The dense profile selected from the source configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaProfile {
    /// A conventional Llama-family `model_type=llama` checkpoint.
    StandardLlama,
    /// A MiniCPM5 (or explicitly MiniCPM-labelled) checkpoint.
    MiniCpm5,
}

impl LlamaProfile {
    /// Stable display name used by callers that persist source classification.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StandardLlama => "llama",
            Self::MiniCpm5 => "minicpm5",
        }
    }
}

/// Capabilities proven by [`LlamaFrontend::from_source`].
///
/// Quantized MLX affine/oQe weights are accepted as source-level packed views.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaCapabilities {
    pub dense: bool,
    pub grouped_query_attention: bool,
    pub untied_output_head: bool,
    pub f16: bool,
    pub bf16: bool,
    pub mlx_affine_quantization: bool,
}

impl LlamaCapabilities {
    pub const fn dense_source() -> Self {
        Self {
            dense: true,
            grouped_query_attention: true,
            untied_output_head: true,
            f16: true,
            bf16: true,
            mlx_affine_quantization: false,
        }
    }
}

/// Geometry and token metadata needed by dense Llama-family lowering.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaGeometry {
    pub vocab_size: u64,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub max_position_embeddings: u64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
}

/// Validated, target-independent source inventory for a dense Llama-family
/// checkpoint. Tensor references retain their source shard, byte offsets,
/// lengths, dtype spellings, and logical shapes exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaSource {
    pub profile: LlamaProfile,
    pub geometry: LlamaGeometry,
    pub capabilities: LlamaCapabilities,
    pub global_tensors: BTreeMap<String, TensorRef>,
    pub layer_tensors: BTreeMap<u32, BTreeMap<String, TensorRef>>,
    /// Source tensors not consumed by the dense execution contract. They are
    /// retained rather than discarded so inventory accounting remains exact.
    pub resident_tensors: BTreeMap<String, TensorRef>,
}

impl LlamaSource {
    pub fn tensor(&self, role: &str) -> Option<&TensorRef> {
        self.global_tensors.get(role)
    }

    pub fn layer_tensor(&self, layer: u32, role: &str) -> Option<&TensorRef> {
        self.layer_tensors.get(&layer)?.get(role)
    }
}

/// Deterministic lowering view. This is deliberately a source-level lowering,
/// not a `.logan`/`.coli` package representation.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaLowering {
    pub profile: LlamaProfile,
    pub geometry: LlamaGeometry,
    pub capabilities: LlamaCapabilities,
    pub global_tensors: BTreeMap<String, TensorRef>,
    pub layer_tensors: BTreeMap<u32, BTreeMap<String, TensorRef>>,
    pub resident_tensors: BTreeMap<String, TensorRef>,
}

pub struct LlamaFrontend;

impl LlamaFrontend {
    /// Return whether the source advertises a supported Llama-family model
    /// type. Full geometry and inventory validation is performed by
    /// [`Self::from_source`].
    pub fn probe(source: &SourceInventory) -> Result<bool> {
        let Some(config) = source::config(&source.root)? else {
            return Ok(false);
        };
        Ok(model_profile(&config).is_some())
    }

    /// Validate and classify a raw source inventory.
    pub fn from_source(source: &SourceInventory) -> Result<LlamaSource> {
        let config = source::config(&source.root)?.ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: "Llama source is missing config.json".into(),
        })?;
        let profile = model_profile(&config).ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!(
                "unsupported Llama-family model_type {:?}",
                config.get("model_type").and_then(Value::as_str)
            ),
        })?;
        let geometry = parse_geometry(source, &config)?;

        for (name, tensor) in &source.tensors {
            if !matches!(tensor.dtype.as_str(), "F16" | "BF16")
                && !(tensor.dtype == "F32"
                    && (name.ends_with(".scales") || name.ends_with(".biases")))
                && !is_packed_weight(name, tensor)
            {
                return invalid(
                    source,
                    format!(
                        "tensor `{name}` uses unsupported representation `{}`; expected F16/BF16 or MLX affine/oQe packed U32/U8",
                        tensor.dtype
                    ),
                );
            }
            validate_byte_span(source, name, tensor)?;
        }
        let mut consumed = BTreeSet::new();
        let mut global_tensors = BTreeMap::new();
        let embed = take_global(
            source,
            &config,
            &mut consumed,
            &[
                "model.embed_tokens.weight",
                "embed_tokens.weight",
                "model.tok_embeddings.weight",
                "tok_embeddings.weight",
            ],
            "token embedding",
            &[geometry.vocab_size, geometry.hidden_size],
        )?;
        global_tensors.insert("embed.weight".into(), embed);
        let head = take_global(
            source,
            &config,
            &mut consumed,
            &["lm_head.weight", "model.lm_head.weight"],
            "untied lm_head",
            &[geometry.vocab_size, geometry.hidden_size],
        )?;
        global_tensors.insert("head.weight".into(), head);
        let norm = take_global(
            source,
            &config,
            &mut consumed,
            &["model.norm.weight", "norm.weight"],
            "final norm",
            &[geometry.hidden_size],
        )?;
        global_tensors.insert("norm.weight".into(), norm);

        let mut layer_tensors = BTreeMap::new();
        for layer in 0..geometry.num_hidden_layers {
            let prefix = layer_prefix(source, layer)?;
            let mut roles = BTreeMap::new();
            for (role, suffix, shape) in [
                (
                    "input_layernorm.weight",
                    "input_layernorm.weight",
                    vec![geometry.hidden_size],
                ),
                (
                    "self_attn.q_proj.weight",
                    "self_attn.q_proj.weight",
                    vec![
                        checked_mul(
                            u64::from(geometry.num_attention_heads),
                            u64::from(geometry.head_dim),
                            source,
                            "query projection width",
                        )?,
                        geometry.hidden_size,
                    ],
                ),
                (
                    "self_attn.k_proj.weight",
                    "self_attn.k_proj.weight",
                    vec![
                        checked_mul(
                            u64::from(geometry.num_key_value_heads),
                            u64::from(geometry.head_dim),
                            source,
                            "key projection width",
                        )?,
                        geometry.hidden_size,
                    ],
                ),
                (
                    "self_attn.v_proj.weight",
                    "self_attn.v_proj.weight",
                    vec![
                        checked_mul(
                            u64::from(geometry.num_key_value_heads),
                            u64::from(geometry.head_dim),
                            source,
                            "value projection width",
                        )?,
                        geometry.hidden_size,
                    ],
                ),
                (
                    "self_attn.o_proj.weight",
                    "self_attn.o_proj.weight",
                    vec![
                        geometry.hidden_size,
                        checked_mul(
                            u64::from(geometry.num_attention_heads),
                            u64::from(geometry.head_dim),
                            source,
                            "output projection width",
                        )?,
                    ],
                ),
                (
                    "post_attention_layernorm.weight",
                    "post_attention_layernorm.weight",
                    vec![geometry.hidden_size],
                ),
                (
                    "mlp.gate_proj.weight",
                    "mlp.gate_proj.weight",
                    vec![geometry.intermediate_size, geometry.hidden_size],
                ),
                (
                    "mlp.up_proj.weight",
                    "mlp.up_proj.weight",
                    vec![geometry.intermediate_size, geometry.hidden_size],
                ),
                (
                    "mlp.down_proj.weight",
                    "mlp.down_proj.weight",
                    vec![geometry.hidden_size, geometry.intermediate_size],
                ),
            ] {
                let name = format!("{prefix}{suffix}");
                let tensor =
                    source
                        .tensors
                        .get(&name)
                        .ok_or_else(|| ColicError::InvalidSource {
                            path: source.root.clone(),
                            detail: format!("missing required tensor `{name}`"),
                        })?;
                let view = tensor_view(source, &config, &name, tensor, Some(&shape))?;
                consumed.insert(name);
                roles.insert(role.into(), view);
            }
            layer_tensors.insert(layer, roles);
        }

        let resident_tensors = source
            .tensors
            .iter()
            .filter(|(name, _)| !consumed.contains(*name))
            .map(|(name, tensor)| {
                let view = tensor_view(source, &config, name, tensor, None)?;
                Ok((name.clone(), view))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let mlx_affine_quantization = source
            .tensors
            .iter()
            .any(|(name, tensor)| is_packed_weight(name, tensor));

        Ok(LlamaSource {
            profile,
            geometry,
            capabilities: LlamaCapabilities {
                mlx_affine_quantization,
                ..LlamaCapabilities::dense_source()
            },
            global_tensors,
            layer_tensors,
            resident_tensors,
        })
    }

    /// Compatibility spelling used by the other compiler frontends.
    pub fn build(source: &SourceInventory) -> Result<LlamaSource> {
        Self::from_source(source)
    }

    /// Explicit validation entry point for callers that only need a yes/no
    /// source check and do not need to retain the lowering view.
    pub fn validate(source: &SourceInventory) -> Result<()> {
        Self::from_source(source).map(|_| ())
    }

    /// Lower validated source metadata into canonical dense roles.
    pub fn lower(source: &LlamaSource) -> Result<LlamaLowering> {
        Ok(LlamaLowering {
            profile: source.profile,
            geometry: source.geometry.clone(),
            capabilities: source.capabilities,
            global_tensors: source.global_tensors.clone(),
            layer_tensors: source.layer_tensors.clone(),
            resident_tensors: source.resident_tensors.clone(),
        })
    }
}

fn model_profile(config: &Value) -> Option<LlamaProfile> {
    let model_type = config.get("model_type")?.as_str()?;
    if model_type == "llama" {
        let minicpm_arch = config
            .get("architectures")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .any(|name| name.to_ascii_lowercase().contains("minicpm"));
        return Some(if minicpm_arch {
            LlamaProfile::MiniCpm5
        } else {
            LlamaProfile::StandardLlama
        });
    }
    if matches!(model_type, "minicpm" | "minicpm3" | "minicpm4" | "minicpm5") {
        return Some(LlamaProfile::MiniCpm5);
    }
    None
}

fn parse_geometry(source: &SourceInventory, config: &Value) -> Result<LlamaGeometry> {
    let vocab_size = required_u64(source, config, "vocab_size")?;
    let hidden_size = required_u64(source, config, "hidden_size")?;
    let intermediate_size = required_u64(source, config, "intermediate_size")?;
    let num_hidden_layers = required_u32(source, config, "num_hidden_layers")?;
    let num_attention_heads = required_u32(source, config, "num_attention_heads")?;
    let num_key_value_heads = config
        .get("num_key_value_heads")
        .map(|value| positive_u32(source, value, "num_key_value_heads"))
        .transpose()?
        .unwrap_or(num_attention_heads);
    let head_dim = config
        .get("head_dim")
        .map(|value| positive_u32(source, value, "head_dim"))
        .transpose()?
        .unwrap_or_else(|| (hidden_size / u64::from(num_attention_heads)) as u32);
    if hidden_size != u64::from(num_attention_heads) * u64::from(head_dim) {
        return invalid(
            source,
            format!(
                "hidden_size {hidden_size} != num_attention_heads {num_attention_heads} × head_dim {head_dim}"
            ),
        );
    }
    if num_attention_heads % num_key_value_heads != 0 {
        return invalid(
            source,
            format!(
                "num_attention_heads {num_attention_heads} is not divisible by num_key_value_heads {num_key_value_heads}"
            ),
        );
    }
    let rope_theta = rope_theta(source, config)?;
    let max_position_embeddings = config
        .get("max_position_embeddings")
        .map(|value| positive_u64(source, value, "max_position_embeddings"))
        .transpose()?
        .unwrap_or(2048);
    let rms_norm_eps = config
        .get("rms_norm_eps")
        .map(|value| positive_f64(source, value, "rms_norm_eps"))
        .transpose()?
        .unwrap_or(1e-5);
    let tie_word_embeddings = config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if tie_word_embeddings {
        return invalid(
            source,
            "untied output head is required; tie_word_embeddings must be false",
        );
    }
    let eos_token_ids = config
        .get("eos_token_id")
        .map(|value| token_ids(source, value, "eos_token_id"))
        .transpose()?
        .unwrap_or_default();
    let bos_token_id = config
        .get("bos_token_id")
        .map(|value| positive_u32(source, value, "bos_token_id"))
        .transpose()?;

    Ok(LlamaGeometry {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        max_position_embeddings,
        rms_norm_eps,
        rope_theta,
        bos_token_id,
        eos_token_ids,
    })
}

fn rope_theta(source: &SourceInventory, config: &Value) -> Result<f64> {
    let top = config
        .get("rope_theta")
        .map(|value| positive_f64(source, value, "rope_theta"))
        .transpose()?;
    let nested = config
        .get("rope_parameters")
        .map(|value| {
            let object = value.as_object().ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: "rope_parameters must be an object".into(),
            })?;
            object
                .get("rope_theta")
                .map(|value| positive_f64(source, value, "rope_parameters.rope_theta"))
                .transpose()
        })
        .transpose()?
        .flatten();
    match (top, nested) {
        (Some(top), Some(nested)) if top.to_bits() != nested.to_bits() => {
            invalid(source, "rope_theta and rope_parameters.rope_theta disagree")
        }
        (Some(value), _) | (_, Some(value)) => Ok(value),
        (None, None) => Ok(10_000.0),
    }
}

fn layer_prefix(source: &SourceInventory, layer: u32) -> Result<String> {
    let candidates = [format!("model.layers.{layer}."), format!("layers.{layer}.")];
    let present: Vec<_> = candidates
        .into_iter()
        .filter(|prefix| source.tensors.keys().any(|name| name.starts_with(prefix)))
        .collect();
    match present.as_slice() {
        [prefix] => Ok(prefix.clone()),
        [] => invalid(source, format!("missing layer {layer}")),
        _ => invalid(
            source,
            format!("layer {layer} has ambiguous model.layers/layers prefixes"),
        ),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MlxQuantizationMode {
    Affine,
    Oqe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MlxWeightQuant {
    mode: MlxQuantizationMode,
    bits: u8,
    group_size: u32,
}

fn is_packed_weight(name: &str, tensor: &TensorRef) -> bool {
    name.ends_with(".weight") && matches!(tensor.dtype.as_str(), "U32" | "U8")
}

fn take_global(
    source: &SourceInventory,
    config: &Value,
    consumed: &mut BTreeSet<String>,
    aliases: &[&str],
    role: &str,
    shape: &[u64],
) -> Result<TensorRef> {
    let present: Vec<_> = aliases
        .iter()
        .filter(|name| source.tensors.contains_key(**name))
        .copied()
        .collect();
    if present.len() > 1 {
        return invalid(
            source,
            format!("multiple source tensors provide {role}: {present:?}"),
        );
    }
    let name = present.first().ok_or_else(|| ColicError::InvalidSource {
        path: source.root.clone(),
        detail: format!("missing required {role}"),
    })?;
    let tensor = source.tensors.get(*name).expect("presence checked above");
    let view = tensor_view(source, config, name, tensor, Some(shape))?;
    consumed.insert((*name).to_owned());
    Ok(view)
}

fn tensor_view(
    source: &SourceInventory,
    config: &Value,
    name: &str,
    tensor: &TensorRef,
    logical_shape: Option<&[u64]>,
) -> Result<TensorRef> {
    if is_packed_weight(name, tensor) {
        return mlx_tensor_view(source, config, name, tensor, logical_shape);
    }
    if let Some(shape) = logical_shape {
        validate_tensor(source, name, tensor, shape)?;
    }
    Ok(tensor.clone())
}

fn mlx_tensor_view(
    source: &SourceInventory,
    config: &Value,
    name: &str,
    tensor: &TensorRef,
    logical_shape: Option<&[u64]>,
) -> Result<TensorRef> {
    let quant = mlx_weight_quant(source, config, name.trim_end_matches(".weight"))?;
    let packed_columns = tensor
        .shape
        .last()
        .copied()
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("packed tensor `{name}` has scalar shape"),
        })?;
    if tensor.shape.len() != 2 {
        return invalid(
            source,
            format!("packed tensor `{name}` must have rank-2 shape"),
        );
    }
    let rows = tensor.shape[0];
    let (rows, columns) = if let Some(shape) = logical_shape {
        if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
            return invalid(
                source,
                format!("tensor `{name}` has invalid logical shape {shape:?}"),
            );
        }
        (shape[0], shape[1])
    } else {
        let values_per_unit = packed_values_per_unit(source, name, tensor, quant.bits)?;
        (
            rows,
            packed_columns.checked_mul(values_per_unit).ok_or_else(|| {
                ColicError::InvalidSource {
                    path: source.root.clone(),
                    detail: format!("packed tensor `{name}` logical width overflows u64"),
                }
            })?,
        )
    };
    if rows != tensor.shape[0] {
        return invalid(
            source,
            format!(
                "packed tensor `{name}` rows {} do not match logical rows {rows}",
                tensor.shape[0]
            ),
        );
    }
    let values_per_unit = packed_values_per_unit(source, name, tensor, quant.bits)?;
    let expected_packed_columns = columns
        .checked_add(values_per_unit - 1)
        .and_then(|value| value.checked_div(values_per_unit))
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("packed tensor `{name}` packed width overflows u64"),
        })?;
    if packed_columns != expected_packed_columns {
        return invalid(
            source,
            format!(
                "packed tensor `{name}` columns {packed_columns} do not match logical [{rows}, {columns}] for {}-bit storage (expected {expected_packed_columns})",
                quant.bits
            ),
        );
    }
    validate_quant_sidecars(source, name, quant, rows, columns)?;
    let mut shape = tensor.shape.clone();
    shape[0] = rows;
    shape[1] = columns;
    Ok(TensorRef {
        source: tensor.source.clone(),
        offset: tensor.offset,
        len: tensor.len,
        dtype: mlx_affine_dtype(quant.bits, quant.group_size),
        shape,
    })
}

fn packed_values_per_unit(
    source: &SourceInventory,
    name: &str,
    tensor: &TensorRef,
    bits: u8,
) -> Result<u64> {
    let unit_bits = if tensor.dtype == "U32" { 32 } else { 8 };
    if tensor.dtype == "U8" && !matches!(bits, 4 | 8) {
        return invalid(
            source,
            format!("packed tensor `{name}` U8 storage cannot represent {bits}-bit MLX codes"),
        );
    }
    Ok(u64::from(unit_bits / u32::from(bits)))
}

fn mlx_weight_quant(
    source: &SourceInventory,
    config: &Value,
    name: &str,
) -> Result<MlxWeightQuant> {
    let quant = config
        .get("quantization_config")
        .or_else(|| config.get("quantization"))
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("MLX quantized tensor `{name}.weight` is missing quantization_config"),
        })?;
    let quant = quant.as_object().ok_or_else(|| ColicError::InvalidSource {
        path: source.root.clone(),
        detail: "MLX quantization_config must be an object".into(),
    })?;
    let scoped = quant
        .get(name)
        .or_else(|| quant.get(&format!("{name}.weight")))
        .or_else(|| quant.get("tensors").and_then(|value| value.get(name)))
        .and_then(Value::as_object);
    let mode_value = scoped
        .and_then(|value| {
            ["mode", "quantization_mode", "quantization_type", "format"]
                .iter()
                .find_map(|key| value.get(*key))
        })
        .or_else(|| {
            ["mode", "quantization_mode", "quantization_type", "format"]
                .iter()
                .find_map(|key| quant.get(*key))
        });
    let mode = mode_value
        .and_then(Value::as_str)
        .map(|value| value.to_ascii_lowercase().replace('_', "").replace('-', ""))
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("MLX quantization entry `{name}` is missing mode"),
        })?;
    let mode = match mode.as_str() {
        "affine" | "mlxaffine" => MlxQuantizationMode::Affine,
        "oqe" | "mlxoqe" | "oq8e" | "mlxoq8e" => MlxQuantizationMode::Oqe,
        other => {
            return invalid(
                source,
                format!("MLX tensor `{name}` uses unsupported quantization mode `{other}`"),
            );
        }
    };
    let metadata_u64 = |keys: &[&str]| {
        scoped
            .and_then(|value| keys.iter().find_map(|key| value.get(*key)))
            .and_then(Value::as_u64)
            .or_else(|| {
                keys.iter()
                    .find_map(|key| quant.get(*key))
                    .and_then(Value::as_u64)
            })
    };
    let bits = metadata_u64(&["bits", "weight_bits"])
        .and_then(|value| u8::try_from(value).ok())
        .filter(|bits| matches!(*bits, 4 | 5 | 6 | 8))
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("MLX quantized tensor `{name}` has unsupported bits"),
        })?;
    let group_size = metadata_u64(&["group_size", "group", "block_size"])
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("MLX quantized tensor `{name}` has invalid group_size"),
        })?;
    Ok(MlxWeightQuant {
        mode,
        bits,
        group_size,
    })
}

fn validate_quant_sidecars(
    source: &SourceInventory,
    name: &str,
    quant: MlxWeightQuant,
    rows: u64,
    columns: u64,
) -> Result<()> {
    let base = name.trim_end_matches(".weight");
    let scales_name = format!("{base}.scales");
    let biases_name = format!("{base}.biases");
    let scales = source
        .tensors
        .get(&scales_name)
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!(
                "MLX quantized tensor `{name}` is missing scales sidecar `{scales_name}`"
            ),
        })?;
    let biases = source
        .tensors
        .get(&biases_name)
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!(
                "MLX quantized tensor `{name}` is missing biases sidecar `{biases_name}`"
            ),
        })?;
    for (kind, sidecar_name, sidecar) in [
        ("scales", scales_name.as_str(), scales),
        ("biases", biases_name.as_str(), biases),
    ] {
        if !matches!(sidecar.dtype.as_str(), "F16" | "BF16" | "F32") {
            return invalid(
                source,
                format!(
                    "MLX quantized tensor `{name}` {kind} sidecar `{sidecar_name}` must use F16, BF16, or F32"
                ),
            );
        }
        validate_byte_span(source, sidecar_name, sidecar)?;
        let groups = columns
            .checked_add(u64::from(quant.group_size) - 1)
            .and_then(|value| value.checked_div(u64::from(quant.group_size)))
            .ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: format!("MLX quantized tensor `{name}` group geometry overflows"),
            })?;
        let expected = [rows, groups];
        if sidecar.shape != expected {
            return invalid(
                source,
                format!(
                    "MLX quantized tensor `{name}` {kind} sidecar shape {:?}, expected {expected:?}",
                    sidecar.shape
                ),
            );
        }
    }
    Ok(())
}

fn validate_tensor(
    source: &SourceInventory,
    name: &str,
    tensor: &TensorRef,
    shape: &[u64],
) -> Result<()> {
    if tensor.shape != shape {
        return invalid(
            source,
            format!(
                "tensor `{name}` has shape {:?}, expected {shape:?}",
                tensor.shape
            ),
        );
    }
    validate_byte_span(source, name, tensor)
}

fn validate_byte_span(source: &SourceInventory, name: &str, tensor: &TensorRef) -> Result<()> {
    let elements = tensor
        .shape
        .iter()
        .try_fold(1_u64, |acc, value| acc.checked_mul(*value));
    let Some(elements) = elements else {
        return invalid(
            source,
            format!("tensor `{name}` shape element count overflows"),
        );
    };
    let item_bytes = match tensor.dtype.as_str() {
        "F16" | "BF16" => 2,
        "F32" => 4,
        "U32" => 4,
        "U8" => 1,
        other => {
            return invalid(
                source,
                format!("tensor `{name}` uses unsupported representation `{other}`"),
            );
        }
    };
    let Some(expected) = elements.checked_mul(item_bytes) else {
        return invalid(source, format!("tensor `{name}` byte span overflows"));
    };
    if tensor.len != expected {
        return invalid(
            source,
            format!(
                "tensor `{name}` has byte length {}, expected {expected} for dtype {} and {} elements",
                tensor.len, tensor.dtype, elements
            ),
        );
    }
    Ok(())
}

fn checked_mul(a: u64, b: u64, source: &SourceInventory, role: &str) -> Result<u64> {
    a.checked_mul(b).ok_or_else(|| ColicError::InvalidSource {
        path: source.root.clone(),
        detail: format!("{role} overflows u64"),
    })
}

fn required_u64(source: &SourceInventory, config: &Value, key: &str) -> Result<u64> {
    let value = config.get(key).ok_or_else(|| ColicError::InvalidSource {
        path: source.root.clone(),
        detail: format!("missing `{key}`"),
    })?;
    positive_u64(source, value, key)
}

fn required_u32(source: &SourceInventory, config: &Value, key: &str) -> Result<u32> {
    let value = config.get(key).ok_or_else(|| ColicError::InvalidSource {
        path: source.root.clone(),
        detail: format!("missing `{key}`"),
    })?;
    positive_u32(source, value, key)
}

fn positive_u64(source: &SourceInventory, value: &Value, key: &str) -> Result<u64> {
    value
        .as_u64()
        .filter(|value| *value > 0)
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("`{key}` must be a positive unsigned integer"),
        })
}

fn positive_u32(source: &SourceInventory, value: &Value, key: &str) -> Result<u32> {
    let value = positive_u64(source, value, key)?;
    u32::try_from(value).map_err(|_| ColicError::InvalidSource {
        path: source.root.clone(),
        detail: format!("`{key}` does not fit u32"),
    })
}

fn positive_f64(source: &SourceInventory, value: &Value, key: &str) -> Result<f64> {
    let value = value.as_f64().ok_or_else(|| ColicError::InvalidSource {
        path: source.root.clone(),
        detail: format!("`{key}` must be a number"),
    })?;
    if !value.is_finite() || value <= 0.0 {
        return invalid(source, format!("`{key}` must be finite and positive"));
    }
    Ok(value)
}

fn token_ids(source: &SourceInventory, value: &Value, key: &str) -> Result<Vec<u32>> {
    let parse = |value: &Value| {
        let number = value.as_u64().ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("`{key}` must contain unsigned integer token ids"),
        })?;
        u32::try_from(number).map_err(|_| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("`{key}` token id does not fit u32"),
        })
    };
    match value {
        Value::Array(values) => {
            if values.is_empty() {
                return invalid(source, format!("`{key}` cannot be empty"));
            }
            values.iter().map(parse).collect()
        }
        Value::Number(_) => Ok(vec![parse(value)?]),
        _ => invalid(
            source,
            format!("`{key}` must be an unsigned integer or array"),
        ),
    }
}

fn invalid<T>(source: &SourceInventory, detail: impl Into<String>) -> Result<T> {
    Err(ColicError::InvalidSource {
        path: source.root.clone(),
        detail: detail.into(),
    })
}
