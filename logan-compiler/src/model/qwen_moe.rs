//! Qwen3.5 / Qwen3.6 / Qwen3.7 fine-grained MoE source frontend for colic.
//!
//! Mirrors `deepseek_v4.rs` in structure, but targets the HF `qwen3_5_moe`
//! family (model_type `qwen3_5_moe`, arch `Qwen3_5MoeForConditionalGeneration`
//! — the vision wrapper whose text backbone config lives in `text_config`).
//!
//! Key differences from DeepSeek V4:
//! - Text backbone tensors live under `model.language_model.*` (vision under
//!   `model.visual.*` is ignored; `lm_head.weight` is top-level, no prefix).
//! - Routed experts are FUSED per layer: `mlp.experts.gate_up_proj` [E,2·I,H]
//!   and `mlp.experts.down_proj` [E,H,I], all BF16, NO `.weight` suffix and NO
//!   per-expert scale. Each routed expert therefore becomes a sub-`TensorRef`
//!   slicing the fused payload on a per-layer `offset`/`len` (see `read_tensor`
//!   contract in target/mod.rs: it reads exactly `offset..offset+len`).
//! - Hybrid attention: `full_attention` layers (self_attn, every 4th) vs
//!   `linear_attention` layers (Gated DeltaNet / Mamba-style `linear_attn`).
//! - Shared expert + router gate + shared_expert_gate are layer-static.
//! - No HC, no hash router, no compression ratios, no FP8 scales.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use serde_json::Value;

use crate::{
    error::{ColicError, Result},
    ir::{Architecture, Matrix, ModelGeometry, RoutedExpert, SemanticModel},
    source::{self, SourceInventory, TensorRef},
};

pub struct QwenMoeFrontend;

impl QwenMoeFrontend {
    pub fn probe(source: &SourceInventory) -> Result<bool> {
        let Some(config) = source::config(&source.root)? else {
            return Ok(false);
        };
        Ok(config.get("model_type").and_then(Value::as_str) == Some("qwen3_5_moe"))
    }

    pub fn build(source: &SourceInventory) -> Result<SemanticModel> {
        let config = source::config(&source.root)?.ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: "Qwen MoE source is missing config.json".into(),
        })?;
        if config.get("model_type").and_then(Value::as_str) != Some("qwen3_5_moe") {
            return invalid(&source.root, "config model_type is not `qwen3_5_moe`");
        }
        // The real geometry lives in text_config (vision wrapper).
        let tc = config
            .get("text_config")
            .and_then(Value::as_object)
            .ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: "Qwen MoE config is missing `text_config`".into(),
            })?;
        let tc = Value::Object(tc.clone());

        let layers = required_u32(&source.root, &tc, "num_hidden_layers")?;
        let layer_types = tc
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: "text_config is missing `layer_types`".into(),
            })?;
        if layer_types.len() != layers as usize {
            return invalid(
                &source.root,
                format!(
                    "`layer_types` has {} entries but `num_hidden_layers` is {layers}",
                    layer_types.len()
                ),
            );
        }
        for (layer, layer_type) in layer_types.iter().enumerate() {
            match layer_type.as_str() {
                Some("full_attention" | "linear_attention") => {}
                other => {
                    return invalid(
                        &source.root,
                        format!("layer {layer} has unsupported layer type {other:?}"),
                    );
                }
            }
        }

        let shared_intermediate_size =
            required_u32(&source.root, &tc, "shared_expert_intermediate_size")?;
        let linear_num_key_heads = required_u32(&source.root, &tc, "linear_num_key_heads")?;
        let linear_num_value_heads = required_u32(&source.root, &tc, "linear_num_value_heads")?;
        let linear_value_head_dim = required_u32(&source.root, &tc, "linear_value_head_dim")?;
        let linear_conv_kernel_dim = required_u32(&source.root, &tc, "linear_conv_kernel_dim")?;

        let geometry = ModelGeometry {
            hidden_size: required_u32(&source.root, &tc, "hidden_size")?,
            layers,
            routed_experts_per_layer: required_u32(&source.root, &tc, "num_experts")?,
            moe_intermediate_size: required_u32(&source.root, &tc, "moe_intermediate_size")?,
            vocab_size: required_u32(&source.root, &tc, "vocab_size")?,
            // Fields below are unused by downstream for Qwen (colic only reads
            // geometry.layers), but remain in the shared struct.
            hc_mult: 1,
            num_hash_layers: 0,
            experts_per_token: required_u32(&source.root, &tc, "num_experts_per_tok")?,
            attention_heads: required_u32(&source.root, &tc, "num_attention_heads")?,
            head_dim: required_u32(&source.root, &tc, "head_dim")?,
            num_key_value_heads: required_u32(&source.root, &tc, "num_key_value_heads")?,
            linear_key_head_dim: required_u32(&source.root, &tc, "linear_key_head_dim")?,
            q_lora_rank: 0,
            o_groups: 1,
            o_lora_rank: 0,
            index_heads: 0,
            index_head_dim: 0,
            compression_ratios: vec![0; layers as usize],
        };

        let linear_key_width = u64::from(linear_num_key_heads)
            .checked_mul(u64::from(geometry.linear_key_head_dim))
            .ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: "linear-attention key width overflows u64".into(),
            })?;
        let linear_value_width = u64::from(linear_num_value_heads)
            .checked_mul(u64::from(linear_value_head_dim))
            .ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: "linear-attention value width overflows u64".into(),
            })?;
        let linear_qkv_width = linear_key_width
            .checked_mul(2)
            .and_then(|value| value.checked_add(linear_value_width))
            .ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: "linear-attention QKV width overflows u64".into(),
            })?;

        // MLX Qwen3.5 checkpoints use a different, already-packed text layout
        // (`language_model.model.*`) with routed experts under
        // `mlp.switch_mlp.{gate,up,down}_proj.*`. Preserve both standard MLX
        // MXFP4 and MLX affine quantization exactly instead of routing either
        // representation through BF16.
        if is_mlx_quantized_layout(source) {
            return build_mlx_quantized(source, geometry, &config);
        }

        // ---- routed experts: slice the fused per-layer tensors ----
        let inter = geometry.moe_intermediate_size;
        let hidden = geometry.hidden_size;
        let prefix = "model.language_model.layers";

        let mut routed_experts = BTreeMap::new();
        for layer in 0..geometry.layers {
            let lp = format!("{prefix}.{layer}.mlp.experts");
            let gu_name = format!("{lp}.gate_up_proj");
            let dn_name = format!("{lp}.down_proj");
            let guz = tensor_by_name(source, &gu_name)?;
            let dz = tensor_by_name(source, &dn_name)?;
            // gate_up_proj is [E, 2·I, H] and down_proj is [E, H, I], both BF16.
            let expected_gu_bytes =
                geometry.routed_experts_per_layer as u64 * 2 * inter as u64 * hidden as u64 * 2;
            let expected_d_bytes =
                geometry.routed_experts_per_layer as u64 * hidden as u64 * inter as u64 * 2;
            if guz.len != expected_gu_bytes {
                return invalid(
                    &source.root,
                    format!(
                        "layer {layer} gate_up_proj len {} != expected {expected_gu_bytes}",
                        guz.len
                    ),
                );
            }
            if dz.len != expected_d_bytes {
                return invalid(
                    &source.root,
                    format!(
                        "layer {layer} down_proj len {} != expected {expected_d_bytes}",
                        dz.len
                    ),
                );
            }
            for expert in 0..geometry.routed_experts_per_layer {
                let gate = slice_fused(guz, expert, 2 * inter, hidden, 0, inter)?;
                let up = slice_fused(guz, expert, 2 * inter, hidden, inter, inter)?;
                let down = slice_fused(dz, expert, hidden, inter, 0, hidden)?;
                routed_experts.insert(
                    (layer, expert),
                    RoutedExpert {
                        layer,
                        expert,
                        gate,
                        up,
                        down,
                    },
                );
            }
        }

        // ---- MTP routed experts: layer = n_layers + stage (speculative head) ----
        // The MTP head is one full-attention MoE layer per stage with the same
        // fused expert layout as the main layers; its experts ride the same
        // (layer, expert) index space at layer n_layers + stage.
        if let Some(mtp) = crate::model::qwen_mtp::inspect(source)? {
            for (stage, st) in mtp.stages.iter().enumerate() {
                let layer = geometry.layers + stage as u32;
                for expert in 0..mtp.experts {
                    use crate::model::qwen_mtp::QwenMtpExpertBank;
                    let (gate, up, down) = match &st.expert_bank {
                        QwenMtpExpertBank::FusedGateUp { gate_up, down } => (
                            slice_fused(gate_up, expert, 2 * inter, hidden, 0, inter)?,
                            slice_fused(gate_up, expert, 2 * inter, hidden, inter, inter)?,
                            slice_fused(down, expert, hidden, inter, 0, hidden)?,
                        ),
                        QwenMtpExpertBank::SplitGateUp { gate, up, down } => (
                            slice_fused(gate, expert, inter, hidden, 0, inter)?,
                            slice_fused(up, expert, inter, hidden, 0, inter)?,
                            slice_fused(down, expert, hidden, inter, 0, hidden)?,
                        ),
                    };
                    routed_experts.insert(
                        (layer, expert),
                        RoutedExpert {
                            layer,
                            expert,
                            gate,
                            up,
                            down,
                        },
                    );
                }
            }
        }

        // ---- global tensors (embed, final norm, lm_head) ----
        let mut global_tensors: BTreeMap<String, TensorRef> = BTreeMap::new();
        global_tensors.insert(
            "embed.weight".into(),
            validate_tensor(
                &source.root,
                &source.tensors,
                "model.language_model.embed_tokens.weight",
                "BF16",
                &[geometry.vocab_size as u64, geometry.hidden_size as u64],
            )?,
        );
        global_tensors.insert(
            "norm.weight".into(),
            validate_tensor(
                &source.root,
                &source.tensors,
                "model.language_model.norm.weight",
                "BF16",
                &[geometry.hidden_size as u64],
            )?,
        );
        global_tensors.insert(
            "head.weight".into(),
            validate_tensor(
                &source.root,
                &source.tensors,
                "lm_head.weight",
                "BF16",
                &[geometry.vocab_size as u64, geometry.hidden_size as u64],
            )?,
        );

        // ---- layer-static tensors ----
        let mut layer_static_tensors = BTreeMap::new();
        for layer in 0..geometry.layers {
            let lp = format!("{prefix}.{layer}");
            let mut static_tensors = BTreeMap::new();
            let is_full = layer_types[layer as usize].as_str() == Some("full_attention");
            if is_full {
                // attn_output_gate=true doubles q but o stays single-width.
                let q_width = 2 * geometry.attention_heads * geometry.head_dim;
                let kv_width = geometry.num_key_value_heads * geometry.head_dim;
                let o_width = geometry.attention_heads * geometry.head_dim;
                for (role, rows, columns) in [
                    ("q_proj", q_width, geometry.hidden_size),
                    ("k_proj", kv_width, geometry.hidden_size),
                    ("v_proj", kv_width, geometry.hidden_size),
                    ("o_proj", geometry.hidden_size, o_width),
                ] {
                    static_tensors.insert(
                        format!("attn.{role}.weight"),
                        validate_tensor(
                            &source.root,
                            &source.tensors,
                            &format!("{lp}.self_attn.{role}.weight"),
                            "BF16",
                            &[rows as u64, columns as u64],
                        )?,
                    );
                }
                for role in ["q_norm", "k_norm"] {
                    static_tensors.insert(
                        format!("attn.{role}.weight"),
                        validate_tensor(
                            &source.root,
                            &source.tensors,
                            &format!("{lp}.self_attn.{role}.weight"),
                            "BF16",
                            &[geometry.head_dim as u64],
                        )?,
                    );
                }
            } else {
                for (role, shape) in [
                    ("A_log", vec![u64::from(linear_num_value_heads)]),
                    ("dt_bias", vec![u64::from(linear_num_value_heads)]),
                    (
                        "conv1d.weight",
                        vec![linear_qkv_width, 1, u64::from(linear_conv_kernel_dim)],
                    ),
                    (
                        "in_proj_a.weight",
                        vec![
                            u64::from(linear_num_value_heads),
                            u64::from(geometry.hidden_size),
                        ],
                    ),
                    (
                        "in_proj_b.weight",
                        vec![
                            u64::from(linear_num_value_heads),
                            u64::from(geometry.hidden_size),
                        ],
                    ),
                    (
                        "in_proj_qkv.weight",
                        vec![linear_qkv_width, u64::from(geometry.hidden_size)],
                    ),
                    (
                        "in_proj_z.weight",
                        vec![linear_value_width, u64::from(geometry.hidden_size)],
                    ),
                    (
                        "out_proj.weight",
                        vec![u64::from(geometry.hidden_size), linear_value_width],
                    ),
                ] {
                    static_tensors.insert(
                        format!("linear_attn.{role}"),
                        validate_tensor(
                            &source.root,
                            &source.tensors,
                            &format!("{lp}.linear_attn.{role}"),
                            "BF16",
                            &shape,
                        )?,
                    );
                }
                static_tensors.insert(
                    "attn_norm.weight".into(),
                    validate_tensor(
                        &source.root,
                        &source.tensors,
                        &format!("{lp}.linear_attn.norm.weight"),
                        "BF16",
                        &[u64::from(linear_value_head_dim)],
                    )?,
                );
            }
            static_tensors.insert(
                "input_layernorm.weight".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("{lp}.input_layernorm.weight"),
                    "BF16",
                    &[geometry.hidden_size as u64],
                )?,
            );
            static_tensors.insert(
                "post_attention_layernorm.weight".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("{lp}.post_attention_layernorm.weight"),
                    "BF16",
                    &[geometry.hidden_size as u64],
                )?,
            );
            // router gate
            static_tensors.insert(
                "ffn.gate.weight".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("{lp}.mlp.gate.weight"),
                    "BF16",
                    &[
                        geometry.routed_experts_per_layer as u64,
                        geometry.hidden_size as u64,
                    ],
                )?,
            );
            // Shared expert projection gate must have a distinct semantic key
            // from the scalar shared_expert_gate. Keeping these separate avoids
            // silently replacing one record in the BTreeMap.
            static_tensors.insert(
                "ffn.shared_experts.gate_proj.weight".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("{lp}.mlp.shared_expert.gate_proj.weight"),
                    "BF16",
                    &[
                        u64::from(shared_intermediate_size),
                        u64::from(geometry.hidden_size),
                    ],
                )?,
            );
            static_tensors.insert(
                "ffn.shared_experts.up.weight".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("{lp}.mlp.shared_expert.up_proj.weight"),
                    "BF16",
                    &[
                        u64::from(shared_intermediate_size),
                        u64::from(geometry.hidden_size),
                    ],
                )?,
            );
            static_tensors.insert(
                "ffn.shared_experts.down.weight".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("{lp}.mlp.shared_expert.down_proj.weight"),
                    "BF16",
                    &[
                        u64::from(geometry.hidden_size),
                        u64::from(shared_intermediate_size),
                    ],
                )?,
            );
            static_tensors.insert(
                "ffn.shared_experts.gate.weight".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("{lp}.mlp.shared_expert_gate.weight"),
                    "BF16",
                    &[geometry.hidden_size as u64],
                )?,
            );
            layer_static_tensors.insert(layer, static_tensors);
        }

        // Map canonical layer-static roles back to real HF source names, for
        // resident-tensor exclusion.
        let mut static_sources = BTreeSet::new();
        for layer in 0..geometry.layers {
            let lp = format!("{prefix}.{layer}");
            for canon in layer_static_tensors[&layer].keys() {
                let src = if canon == "attn_norm.weight" {
                    format!("{lp}.linear_attn.norm.weight")
                } else if let Some(role) = canon.strip_prefix("attn.") {
                    format!("{lp}.self_attn.{role}")
                } else if canon.starts_with("linear_attn.") {
                    format!("{lp}.{canon}")
                } else if canon == "ffn.gate.weight" {
                    format!("{lp}.mlp.gate.weight")
                } else if canon == "ffn.shared_experts.gate.weight" {
                    format!("{lp}.mlp.shared_expert_gate.weight")
                } else if canon == "ffn.shared_experts.gate_proj.weight" {
                    format!("{lp}.mlp.shared_expert.gate_proj.weight")
                } else if canon.starts_with("ffn.shared_experts.") {
                    let role = canon
                        .trim_start_matches("ffn.shared_experts.")
                        .trim_end_matches(".weight");
                    format!("{lp}.mlp.shared_expert.{role}_proj.weight")
                } else if canon == "input_layernorm.weight" {
                    format!("{lp}.input_layernorm.weight")
                } else if canon == "post_attention_layernorm.weight" {
                    format!("{lp}.post_attention_layernorm.weight")
                } else {
                    format!("{lp}.{canon}")
                };
                static_sources.insert(src);
            }
        }
        let mut global_sources = BTreeSet::new();
        for canon in global_tensors.keys() {
            global_sources.insert(match canon.as_str() {
                "head.weight" => "lm_head.weight".to_owned(),
                "embed.weight" => "model.language_model.embed_tokens.weight".to_owned(),
                _ => format!("model.language_model.{canon}"),
            });
        }
        let resident_tensors = source
            .tensors
            .iter()
            .filter(|(name, _)| {
                // Exclude any tensor that is a global, a layer-static, or an
                // expert fused payload. Sub-slices share the same shard, so
                // exclusion must remain exact-name based rather than file based.
                !global_sources.contains(*name)
                    && !static_sources.contains(*name)
                    && !is_expert_fused_name(name)
            })
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect();

        Ok(SemanticModel {
            architecture: Architecture::Qwen3_5MoeMoE,
            geometry,
            routed_experts,
            global_tensors,
            layer_static_tensors,
            resident_tensors,
        })
    }
}

fn is_mlx_quantized_layout(source: &SourceInventory) -> bool {
    source
        .tensors
        .contains_key("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight")
}

fn build_mlx_quantized(
    source: &SourceInventory,
    geometry: ModelGeometry,
    config: &Value,
) -> Result<SemanticModel> {
    let mut consumed = BTreeSet::new();
    let routed_experts = build_mlx_switch_experts(source, &geometry, config, &mut consumed)?;

    let mut global_tensors = BTreeMap::new();
    for (canonical, source_name) in [
        ("embed.weight", "language_model.model.embed_tokens.weight"),
        ("embed.scales", "language_model.model.embed_tokens.scales"),
        ("head.weight", "language_model.lm_head.weight"),
        ("head.scales", "language_model.lm_head.scales"),
        ("norm.weight", "language_model.model.norm.weight"),
    ] {
        let tensor = tensor_by_name(source, source_name)?;
        global_tensors.insert(
            canonical.to_owned(),
            mlx_tensor_view(source, config, source_name, tensor)?,
        );
        consumed.insert(source_name.to_owned());
    }
    // Affine MLX tensors have an additive BF16 bias per quantization group.
    // MXFP4 does not, so these global sidecars are deliberately optional.
    for (canonical, source_name) in [
        ("embed.biases", "language_model.model.embed_tokens.biases"),
        ("head.biases", "language_model.lm_head.biases"),
    ] {
        if let Some(tensor) = source.tensors.get(source_name) {
            global_tensors.insert(canonical.to_owned(), tensor.clone());
            consumed.insert(source_name.to_owned());
        }
    }

    let mut layer_static_tensors = BTreeMap::new();
    for layer in 0..geometry.layers {
        let prefix = format!("language_model.model.layers.{layer}.");
        let mut tensors = BTreeMap::new();
        for (name, tensor) in &source.tensors {
            let Some(role) = name.strip_prefix(&prefix) else {
                continue;
            };
            if role.starts_with("mlp.switch_mlp.") {
                continue;
            }
            tensors.insert(
                role.to_owned(),
                mlx_tensor_view(source, config, name, tensor)?,
            );
            consumed.insert(name.clone());
        }
        if tensors.is_empty() {
            return invalid(
                &source.root,
                format!("MLX MXFP4 layer {layer} has no static tensors"),
            );
        }
        layer_static_tensors.insert(layer, tensors);
    }

    // Keep any future text-side tensors that were not classified above, but
    // deliberately ignore the vision tower: Logan's Qwen runtime is text-only.
    let mut resident_tensors = BTreeMap::new();
    for (name, tensor) in &source.tensors {
        if consumed.contains(name)
            || name.starts_with("vision_tower.")
            || name.starts_with("model.visual.")
            || name.starts_with("visual.")
        {
            continue;
        }
        if name.starts_with("language_model.") {
            resident_tensors.insert(name.clone(), mlx_tensor_view(source, config, name, tensor)?);
        }
    }

    Ok(SemanticModel {
        architecture: Architecture::Qwen3_5MoeMoE,
        geometry,
        routed_experts,
        global_tensors,
        layer_static_tensors,
        resident_tensors,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MlxWeightQuant {
    Mxfp4,
    Affine { bits: u8, group_size: u32 },
}

fn mlx_weight_quant(
    source: &SourceInventory,
    config: &Value,
    base: &str,
) -> Result<MlxWeightQuant> {
    let quant = config
        .get("quantization_config")
        .or_else(|| config.get("quantization"))
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: "MLX quantized checkpoint is missing quantization_config".into(),
        })?;
    let effective = quant
        .get(base)
        .filter(|value| value.is_object())
        .unwrap_or(quant);
    let mode = effective
        .get("mode")
        .and_then(Value::as_str)
        .or_else(|| quant.get("mode").and_then(Value::as_str))
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("MLX quantization entry `{base}` is missing mode"),
        })?;
    match mode {
        "mxfp4" => Ok(MlxWeightQuant::Mxfp4),
        "affine" => {
            let bits = effective
                .get("bits")
                .and_then(Value::as_u64)
                .or_else(|| quant.get("bits").and_then(Value::as_u64))
                .and_then(|value| u8::try_from(value).ok())
                .filter(|bits| matches!(*bits, 2 | 3 | 4 | 5 | 6 | 8))
                .ok_or_else(|| ColicError::InvalidSource {
                    path: source.root.clone(),
                    detail: format!("MLX affine entry `{base}` has unsupported bits"),
                })?;
            let group_size = effective
                .get("group_size")
                .and_then(Value::as_u64)
                .or_else(|| quant.get("group_size").and_then(Value::as_u64))
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| ColicError::InvalidSource {
                    path: source.root.clone(),
                    detail: format!("MLX affine entry `{base}` has invalid group_size"),
                })?;
            Ok(MlxWeightQuant::Affine { bits, group_size })
        }
        other => invalid(
            &source.root,
            format!("MLX tensor `{base}` uses unsupported quantization mode `{other}`"),
        ),
    }
}

pub(crate) fn mlx_affine_dtype(bits: u8, group_size: u32) -> String {
    format!("MLX_AFFINE:{bits}:{group_size}")
}

pub(crate) fn parse_mlx_affine_dtype(dtype: &str) -> Option<(u8, u32)> {
    let mut parts = dtype.strip_prefix("MLX_AFFINE:")?.split(':');
    let bits = parts.next()?.parse().ok()?;
    let group_size = parts.next()?.parse().ok()?;
    if parts.next().is_some() || !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) || group_size == 0 {
        return None;
    }
    Some((bits, group_size))
}

/// Reinterpret an MLX quantized tensor as the byte-level representation that
/// COLI stores. Safetensors U32 payloads are little-endian packed words; no
/// byte shuffle is needed on disk. For affine weights the pseudo dtype carries
/// the MLX bit width and group size while `shape` is restored to logical matrix
/// dimensions; `len` remains the exact packed source byte count.
fn mlx_tensor_view(
    source: &SourceInventory,
    config: &Value,
    name: &str,
    tensor: &TensorRef,
) -> Result<TensorRef> {
    match tensor.dtype.as_str() {
        "U32" => {
            let base = name
                .strip_suffix(".weight")
                .ok_or_else(|| ColicError::InvalidSource {
                    path: source.root.clone(),
                    detail: format!("packed U32 tensor `{name}` is not a weight tensor"),
                })?;
            let scale_name = format!("{base}.scales");
            let scale = tensor_by_name(source, &scale_name)?;
            match mlx_weight_quant(source, config, base)? {
                MlxWeightQuant::Mxfp4 => {
                    let mut shape = tensor.shape.clone();
                    let last = shape.last_mut().ok_or_else(|| ColicError::InvalidSource {
                        path: source.root.clone(),
                        detail: format!("packed tensor `{name}` has scalar shape"),
                    })?;
                    *last = last
                        .checked_mul(4)
                        .ok_or_else(|| ColicError::InvalidSource {
                            path: source.root.clone(),
                            detail: format!("packed tensor `{name}` byte shape overflows u64"),
                        })?;
                    let dtype = match scale.dtype.as_str() {
                        // Normal MXFP4 matrices: E2M1 nibbles plus E8M0 scales.
                        "U8" => "I8",
                        // mlx-community Qwen3.5 MXFP4 checkpoints keep tiny router
                        // matrices as affine-8 even though the checkpoint-level
                        // quantization mode is MXFP4. Their BF16 scales/biases are
                        // authoritative for this per-tensor exception.
                        "BF16" => {
                            let bias_name = format!("{base}.biases");
                            let bias = tensor_by_name(source, &bias_name)?;
                            if bias.dtype != "BF16" || bias.shape != scale.shape {
                                return invalid(
                                    &source.root,
                                    format!(
                                        "affine-8 tensor `{name}` requires matching BF16 scales/biases"
                                    ),
                                );
                            }
                            "U8"
                        }
                        other => {
                            return invalid(
                                &source.root,
                                format!(
                                    "MXFP4 tensor `{name}` has unsupported scale dtype `{other}`"
                                ),
                            );
                        }
                    };
                    Ok(TensorRef {
                        source: tensor.source.clone(),
                        offset: tensor.offset,
                        len: tensor.len,
                        dtype: dtype.into(),
                        shape,
                    })
                }
                MlxWeightQuant::Affine { bits, group_size } => {
                    let bias_name = format!("{base}.biases");
                    let bias = tensor_by_name(source, &bias_name)?;
                    if scale.dtype != "BF16" || bias.dtype != "BF16" || scale.shape != bias.shape {
                        return invalid(
                            &source.root,
                            format!(
                                "MLX affine tensor `{name}` requires matching BF16 scales/biases"
                            ),
                        );
                    }
                    let mut shape = tensor.shape.clone();
                    let words = *shape.last().ok_or_else(|| ColicError::InvalidSource {
                        path: source.root.clone(),
                        detail: format!("packed tensor `{name}` has scalar shape"),
                    })?;
                    let logical_bits =
                        words
                            .checked_mul(32)
                            .ok_or_else(|| ColicError::InvalidSource {
                                path: source.root.clone(),
                                detail: format!(
                                    "packed tensor `{name}` logical width overflows u64"
                                ),
                            })?;
                    if logical_bits % u64::from(bits) != 0 {
                        return invalid(
                            &source.root,
                            format!("packed tensor `{name}` width is not divisible by {bits} bits"),
                        );
                    }
                    let columns = logical_bits / u64::from(bits);
                    if columns % u64::from(group_size) != 0 {
                        return invalid(
                            &source.root,
                            format!(
                                "MLX affine tensor `{name}` columns={columns} not divisible by group_size={group_size}"
                            ),
                        );
                    }
                    let mut expected_params = shape.clone();
                    *expected_params.last_mut().unwrap() = columns / u64::from(group_size);
                    if scale.shape != expected_params {
                        return invalid(
                            &source.root,
                            format!(
                                "MLX affine params `{scale_name}` have {:?}, expected {:?}",
                                scale.shape, expected_params
                            ),
                        );
                    }
                    *shape.last_mut().unwrap() = columns;
                    Ok(TensorRef {
                        source: tensor.source.clone(),
                        offset: tensor.offset,
                        len: tensor.len,
                        dtype: mlx_affine_dtype(bits, group_size),
                        shape,
                    })
                }
            }
        }
        "U8" if name.ends_with(".scales") => {
            let base = name.trim_end_matches(".scales");
            if mlx_weight_quant(source, config, base)? == MlxWeightQuant::Mxfp4 {
                Ok(TensorRef {
                    source: tensor.source.clone(),
                    offset: tensor.offset,
                    len: tensor.len,
                    dtype: "F8_E8M0".into(),
                    shape: tensor.shape.clone(),
                })
            } else {
                Ok(tensor.clone())
            }
        }
        _ => Ok(tensor.clone()),
    }
}

fn build_mlx_switch_experts(
    source: &SourceInventory,
    geometry: &ModelGeometry,
    config: &Value,
    consumed: &mut BTreeSet<String>,
) -> Result<BTreeMap<(u32, u32), RoutedExpert>> {
    let mut routed = BTreeMap::new();
    for layer in 0..geometry.layers {
        let prefix = format!("language_model.model.layers.{layer}.mlp.switch_mlp");
        let gate_w = format!("{prefix}.gate_proj.weight");
        let gate_s = format!("{prefix}.gate_proj.scales");
        let gate_b = format!("{prefix}.gate_proj.biases");
        let up_w = format!("{prefix}.up_proj.weight");
        let up_s = format!("{prefix}.up_proj.scales");
        let up_b = format!("{prefix}.up_proj.biases");
        let down_w = format!("{prefix}.down_proj.weight");
        let down_s = format!("{prefix}.down_proj.scales");
        let down_b = format!("{prefix}.down_proj.biases");
        for name in [
            &gate_w, &gate_s, &gate_b, &up_w, &up_s, &up_b, &down_w, &down_s, &down_b,
        ] {
            consumed.insert(name.clone());
        }
        for expert in 0..geometry.routed_experts_per_layer {
            routed.insert(
                (layer, expert),
                RoutedExpert {
                    layer,
                    expert,
                    gate: slice_mlx_switch_bank(
                        source,
                        config,
                        &gate_w,
                        &gate_s,
                        expert,
                        geometry.routed_experts_per_layer,
                        geometry.moe_intermediate_size,
                        geometry.hidden_size,
                    )?,
                    up: slice_mlx_switch_bank(
                        source,
                        config,
                        &up_w,
                        &up_s,
                        expert,
                        geometry.routed_experts_per_layer,
                        geometry.moe_intermediate_size,
                        geometry.hidden_size,
                    )?,
                    down: slice_mlx_switch_bank(
                        source,
                        config,
                        &down_w,
                        &down_s,
                        expert,
                        geometry.routed_experts_per_layer,
                        geometry.hidden_size,
                        geometry.moe_intermediate_size,
                    )?,
                },
            );
        }
    }
    Ok(routed)
}

fn slice_mlx_switch_bank(
    source: &SourceInventory,
    config: &Value,
    weight_name: &str,
    scale_name: &str,
    expert: u32,
    experts: u32,
    rows: u32,
    columns: u32,
) -> Result<Matrix> {
    let base = weight_name
        .strip_suffix(".weight")
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("packed expert tensor `{weight_name}` is not a weight tensor"),
        })?;
    match mlx_weight_quant(source, config, base)? {
        MlxWeightQuant::Mxfp4 => slice_mlx_mxfp4_bank(
            source,
            weight_name,
            scale_name,
            expert,
            experts,
            rows,
            columns,
        ),
        MlxWeightQuant::Affine { bits, group_size } => slice_mlx_affine_bank(
            source,
            weight_name,
            scale_name,
            expert,
            experts,
            rows,
            columns,
            bits,
            group_size,
        ),
    }
}

fn slice_mlx_affine_bank(
    source: &SourceInventory,
    weight_name: &str,
    scale_name: &str,
    expert: u32,
    experts: u32,
    rows: u32,
    columns: u32,
    bits: u8,
    group_size: u32,
) -> Result<Matrix> {
    if columns % group_size != 0 {
        return invalid(
            &source.root,
            format!(
                "MLX affine matrix `{weight_name}` has columns={columns}, not divisible by group_size={group_size}"
            ),
        );
    }
    let packed_bits = u64::from(columns)
        .checked_mul(u64::from(bits))
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("MLX affine matrix `{weight_name}` packed width overflows u64"),
        })?;
    if packed_bits % 32 != 0 {
        return invalid(
            &source.root,
            format!(
                "MLX affine matrix `{weight_name}` needs a non-integral U32 row for {columns}x{bits}-bit"
            ),
        );
    }
    let weight = tensor_by_name(source, weight_name)?;
    let scale = tensor_by_name(source, scale_name)?;
    let bias_name = format!("{}.biases", weight_name.trim_end_matches(".weight"));
    let bias = tensor_by_name(source, &bias_name)?;
    let packed_words = packed_bits / 32;
    let groups = u64::from(columns / group_size);
    let expected_weight_shape = [u64::from(experts), u64::from(rows), packed_words];
    let expected_param_shape = [u64::from(experts), u64::from(rows), groups];
    if weight.dtype != "U32" || weight.shape != expected_weight_shape {
        return invalid(
            &source.root,
            format!(
                "MLX affine weight `{weight_name}` has {}/{:?}, expected U32/{expected_weight_shape:?}",
                weight.dtype, weight.shape
            ),
        );
    }
    for (name, tensor) in [(scale_name, scale), (bias_name.as_str(), bias)] {
        if tensor.dtype != "BF16" || tensor.shape != expected_param_shape {
            return invalid(
                &source.root,
                format!(
                    "MLX affine params `{name}` have {}/{:?}, expected BF16/{expected_param_shape:?}",
                    tensor.dtype, tensor.shape
                ),
            );
        }
    }
    let weight_bytes = u64::from(rows)
        .checked_mul(packed_words)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("MLX affine weight slice `{weight_name}` overflows u64"),
        })?;
    let param_bytes = u64::from(rows)
        .checked_mul(groups)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("MLX affine parameter slice `{scale_name}` overflows u64"),
        })?;
    let weight_offset = weight
        .offset
        .checked_add(u64::from(expert).checked_mul(weight_bytes).ok_or_else(|| {
            ColicError::InvalidSource {
                path: source.root.clone(),
                detail: format!("MLX affine expert offset `{weight_name}` overflows u64"),
            }
        })?)
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("MLX affine expert offset `{weight_name}` overflows u64"),
        })?;
    let scale_offset = scale
        .offset
        .checked_add(u64::from(expert).checked_mul(param_bytes).ok_or_else(|| {
            ColicError::InvalidSource {
                path: source.root.clone(),
                detail: format!("MLX affine expert offset `{scale_name}` overflows u64"),
            }
        })?)
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("MLX affine expert offset `{scale_name}` overflows u64"),
        })?;
    let bias_offset = bias
        .offset
        .checked_add(u64::from(expert).checked_mul(param_bytes).ok_or_else(|| {
            ColicError::InvalidSource {
                path: source.root.clone(),
                detail: format!("MLX affine expert offset `{bias_name}` overflows u64"),
            }
        })?)
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("MLX affine expert offset `{bias_name}` overflows u64"),
        })?;
    Ok(Matrix {
        source: TensorRef {
            source: weight.source.clone(),
            offset: weight_offset,
            len: weight_bytes,
            dtype: mlx_affine_dtype(bits, group_size),
            shape: vec![u64::from(rows), u64::from(columns)],
        },
        rows,
        columns,
        scale: Some(TensorRef {
            source: scale.source.clone(),
            offset: scale_offset,
            len: param_bytes,
            dtype: "BF16".into(),
            shape: vec![u64::from(rows), groups],
        }),
        bias: Some(TensorRef {
            source: bias.source.clone(),
            offset: bias_offset,
            len: param_bytes,
            dtype: "BF16".into(),
            shape: vec![u64::from(rows), groups],
        }),
    })
}

fn slice_mlx_mxfp4_bank(
    source: &SourceInventory,
    weight_name: &str,
    scale_name: &str,
    expert: u32,
    experts: u32,
    rows: u32,
    columns: u32,
) -> Result<Matrix> {
    if columns % 32 != 0 {
        return invalid(
            &source.root,
            format!("MXFP4 matrix `{weight_name}` has columns={columns}, not divisible by 32"),
        );
    }
    let weight = tensor_by_name(source, weight_name)?;
    let scale = tensor_by_name(source, scale_name)?;
    let packed_words = u64::from(columns) / 8;
    let groups = u64::from(columns) / 32;
    let expected_weight_shape = [u64::from(experts), u64::from(rows), packed_words];
    let expected_scale_shape = [u64::from(experts), u64::from(rows), groups];
    if weight.dtype != "U32" || weight.shape != expected_weight_shape {
        return invalid(
            &source.root,
            format!(
                "MXFP4 weight `{weight_name}` has {}/{:?}, expected U32/{expected_weight_shape:?}",
                weight.dtype, weight.shape
            ),
        );
    }
    if scale.dtype != "U8" || scale.shape != expected_scale_shape {
        return invalid(
            &source.root,
            format!(
                "MXFP4 scales `{scale_name}` have {}/{:?}, expected U8/{expected_scale_shape:?}",
                scale.dtype, scale.shape
            ),
        );
    }

    let weight_bytes = u64::from(rows)
        .checked_mul(u64::from(columns) / 2)
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("MXFP4 weight slice `{weight_name}` overflows u64"),
        })?;
    let scale_bytes =
        u64::from(rows)
            .checked_mul(groups)
            .ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: format!("MXFP4 scale slice `{scale_name}` overflows u64"),
            })?;
    let weight_offset = weight
        .offset
        .checked_add(u64::from(expert).checked_mul(weight_bytes).ok_or_else(|| {
            ColicError::InvalidSource {
                path: source.root.clone(),
                detail: format!("MXFP4 expert offset `{weight_name}` overflows u64"),
            }
        })?)
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("MXFP4 expert offset `{weight_name}` overflows u64"),
        })?;
    let scale_offset = scale
        .offset
        .checked_add(u64::from(expert).checked_mul(scale_bytes).ok_or_else(|| {
            ColicError::InvalidSource {
                path: source.root.clone(),
                detail: format!("MXFP4 expert offset `{scale_name}` overflows u64"),
            }
        })?)
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("MXFP4 expert offset `{scale_name}` overflows u64"),
        })?;

    Ok(Matrix {
        source: TensorRef {
            source: weight.source.clone(),
            offset: weight_offset,
            len: weight_bytes,
            dtype: "I8".into(),
            shape: vec![u64::from(rows), u64::from(columns) / 2],
        },
        rows,
        columns,
        scale: Some(TensorRef {
            source: scale.source.clone(),
            offset: scale_offset,
            len: scale_bytes,
            dtype: "F8_E8M0".into(),
            shape: vec![u64::from(rows), groups],
        }),
        bias: None,
    })
}

/// A tensor name that is one of the two fused per-layer expert payloads.
/// (Used only to avoid double-classifying the same shard tensor as resident.)
fn is_expert_fused_name(name: &str) -> bool {
    match name.rsplit_once(".mlp.experts.") {
        Some((_, tail)) => tail == "gate_up_proj" || tail == "down_proj",
        None => false,
    }
}

/// Split a fused expert tensor's payload for one expert.
///
/// `fused` has shape [E, M, K] (contiguous BF16). For expert `e`, this returns
/// a sub-`TensorRef` covering rows `[row_start, row_start+row_len)` of the
/// M-axis, i.e. byte offset `e*M*K + row_start*K` (×2 for BF16), length
/// `row_len*K*2`. `read_tensor` in target/mod.rs reads exactly
/// `offset..offset+len`.
fn slice_fused(
    fused: &TensorRef,
    expert: u32,
    m: u32,
    k: u32,
    row_start: u32,
    row_len: u32,
) -> Result<Matrix> {
    if fused.dtype != "BF16" {
        return invalid(
            &fused.source,
            format!("fused expert payload is not BF16 (got {})", fused.dtype),
        );
    }
    let bytes_per_row = (k as u64) * 2;
    let base = fused
        .offset
        .checked_add((expert as u64) * (m as u64) * (k as u64) * 2)
        .ok_or_else(|| ColicError::InvalidSource {
            path: fused.source.clone(),
            detail: format!("expert {expert} offset overflows u64"),
        })?;
    let len = (row_len as u64) * bytes_per_row;
    Ok(Matrix {
        source: TensorRef {
            source: fused.source.clone(),
            offset: base + (row_start as u64) * bytes_per_row,
            len,
            dtype: "BF16".into(),
            shape: vec![row_len as u64, k as u64],
        },
        rows: row_len,
        columns: k,
        scale: None,
        bias: None,
    })
}

fn tensor_by_name<'a>(source: &'a SourceInventory, name: &'a str) -> Result<&'a TensorRef> {
    source
        .tensors
        .get(name)
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("missing required tensor `{name}`"),
        })
}

fn validate_tensor(
    root: &Path,
    tensors: &BTreeMap<String, TensorRef>,
    name: &str,
    dtype: &str,
    shape: &[u64],
) -> Result<TensorRef> {
    let tensor = tensors.get(name).ok_or_else(|| ColicError::InvalidSource {
        path: root.to_owned(),
        detail: format!("missing required tensor `{name}`"),
    })?;
    let shape_matches = shape.is_empty()
        || tensor.shape == shape
        || (shape.len() == 1
            && tensor.shape.len() == 2
            && tensor.shape[0] == 1
            && tensor.shape[1] == shape[0]);
    if tensor.dtype != dtype || !shape_matches {
        return invalid(
            root,
            format!(
                "tensor `{name}` has dtype/shape {:?}/{:?}, expected {dtype}/{shape:?}",
                tensor.dtype, tensor.shape
            ),
        );
    }
    Ok(tensor.clone())
}

fn required_u32(root: &Path, config: &Value, field: &str) -> Result<u32> {
    config
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| value.try_into().ok())
        .filter(|value: &u32| *value > 0)
        .ok_or_else(|| ColicError::InvalidSource {
            path: root.to_owned(),
            detail: format!("config `{field}` must be a positive u32"),
        })
}

fn invalid<T>(path: &Path, detail: impl Into<String>) -> Result<T> {
    Err(ColicError::InvalidSource {
        path: path.to_owned(),
        detail: detail.into(),
    })
}
