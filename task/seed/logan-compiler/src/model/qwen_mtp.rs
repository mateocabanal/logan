//! Qwen native MTP source inventory.
//!
//! This module deliberately stops at semantic/source classification. Runtime
//! proposal, transactional state, and target-specific lowering live elsewhere.
//! Qwen3.x and Qwen4Exp use materially different MTP layouts, so they are
//! classified explicitly rather than normalized by tensor-name guesswork.

use std::{collections::BTreeMap, fs};

use serde_json::Value;

use crate::{
    error::{ColicError, Result},
    source::{SourceInventory, TensorRef},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QwenMtpFlavor {
    /// Qwen3.x: one concatenating `mtp.fc` plus the historical transformer/MoE
    /// draft layer. Routed experts are normally one fused gate+up bank.
    LegacyQwen3,
    /// Qwen4Exp / Qwen3.8-Flash-Next: separate HxH embedding/hidden fusion,
    /// four-stream HyperConnection state, QSA indexer, and one recursively used
    /// full-attention MoE draft layer.
    Qwen4Exp {
        hc_count: u32,
        hc_lowrank: u32,
        shared_expert_intermediate_size: u32,
        /// Standalone drafter checkpoints strip the outer `mtp.` namespace and
        /// share the target embedding + LM head at serve time.
        standalone: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QwenMtpExpertBank {
    /// [experts, 2*intermediate, hidden] + [experts, hidden, intermediate].
    FusedGateUp { gate_up: TensorRef, down: TensorRef },
    /// Qwen4Exp standalone drafter layout: three independently pageable banks.
    /// gate/up are [experts, intermediate, hidden], down is
    /// [experts, hidden, intermediate].
    SplitGateUp {
        gate: TensorRef,
        up: TensorRef,
        down: TensorRef,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenMtpStageInventory {
    pub stage: u32,
    pub expert_bank: QwenMtpExpertBank,
    /// Stage-local tensors other than the independently pageable expert bank.
    /// Keys are relative to the stage prefix (`mtp.layers.N.` for embedded
    /// checkpoints and `layers.N.` for a standalone Qwen4Exp drafter).
    pub static_tensors: BTreeMap<String, TensorRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenMtpInventory {
    pub flavor: QwenMtpFlavor,
    pub hidden_layers: u32,
    pub use_dedicated_embeddings: bool,
    pub hidden_size: u32,
    pub experts: u32,
    pub moe_intermediate_size: u32,
    /// MTP-global tensors such as fusion/norm/HC tensors. Keys preserve source
    /// names so embedded and standalone drafter packages remain distinguishable.
    pub global_tensors: BTreeMap<String, TensorRef>,
    pub stages: Vec<QwenMtpStageInventory>,
}

pub fn inspect(source: &SourceInventory) -> Result<Option<QwenMtpInventory>> {
    let config_path = source.root.join("config.json");
    let bytes = fs::read(&config_path).map_err(|source_error| ColicError::Io {
        path: config_path.clone(),
        source: source_error,
    })?;
    let config: Value =
        serde_json::from_slice(&bytes).map_err(|error| ColicError::InvalidSource {
            path: config_path.clone(),
            detail: format!("invalid config.json: {error}"),
        })?;
    let text_config = config
        .get("text_config")
        .and_then(Value::as_object)
        .ok_or_else(|| ColicError::InvalidSource {
            path: config_path.clone(),
            detail: "Qwen config is missing `text_config`".into(),
        })?;
    let text = Value::Object(text_config.clone());

    let standalone_qwen4 =
        config.get("model_type").and_then(Value::as_str) == Some("qwen4_exp_mtp");
    let namespace = if standalone_qwen4 { "" } else { "mtp." };

    let hidden_layers = optional_u32(
        &source.root,
        text_config.get("mtp_num_hidden_layers"),
        "mtp_num_hidden_layers",
    )?
    .or_else(|| {
        text_config
            .get("mtp")
            .and_then(|mtp| mtp.get("num_hidden_layers"))
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
    })
    .or_else(|| {
        standalone_qwen4
            .then(|| optional_u32_value(&text, "num_hidden_layers"))
            .flatten()
    })
    .unwrap_or(0);

    let has_mtp_tensors = if standalone_qwen4 {
        source.tensors.contains_key("fc_embedding.weight")
            || source.tensors.contains_key("fc_hidden.weight")
            || source
                .tensors
                .keys()
                .any(|name| name.starts_with("layers."))
    } else {
        source.tensors.keys().any(|name| name.starts_with("mtp."))
    };

    if hidden_layers == 0 {
        if has_mtp_tensors {
            return invalid(
                source,
                "checkpoint contains MTP tensors but config declares no MTP hidden layers",
            );
        }
        return Ok(None);
    }
    if !has_mtp_tensors {
        return invalid(
            source,
            format!(
                "config declares {hidden_layers} MTP hidden layer(s) but checkpoint contains no MTP tensors"
            ),
        );
    }

    let use_dedicated_embeddings = text_config
        .get("mtp_use_dedicated_embeddings")
        .map(|value| {
            value.as_bool().ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: "`mtp_use_dedicated_embeddings` is not a boolean".into(),
            })
        })
        .transpose()?
        .unwrap_or(false);
    let hidden_size = required_u32(source, text_config.get("hidden_size"), "hidden_size")?;
    let experts = required_u32(source, text_config.get("num_experts"), "num_experts")?;
    let moe_intermediate_size = required_u32(
        source,
        text_config.get("moe_intermediate_size"),
        "moe_intermediate_size",
    )?;

    let flavor = if source
        .tensors
        .contains_key(&format!("{namespace}fc_embedding.weight"))
        || source
            .tensors
            .contains_key(&format!("{namespace}fc_hidden.weight"))
    {
        inspect_qwen4(
            source,
            text_config,
            namespace,
            standalone_qwen4,
            hidden_layers,
            hidden_size,
            experts,
            moe_intermediate_size,
            use_dedicated_embeddings,
        )?
    } else if source
        .tensors
        .contains_key(&format!("{namespace}fc.weight"))
    {
        inspect_legacy(
            source,
            namespace,
            hidden_layers,
            hidden_size,
            experts,
            moe_intermediate_size,
            use_dedicated_embeddings,
        )?
    } else {
        return invalid(
            source,
            "MTP tensors do not match the Qwen3 `fc` or Qwen4Exp `fc_embedding`/`fc_hidden` layout",
        );
    };

    Ok(Some(flavor))
}

fn inspect_legacy(
    source: &SourceInventory,
    namespace: &str,
    hidden_layers: u32,
    hidden_size: u32,
    experts: u32,
    moe_intermediate_size: u32,
    use_dedicated_embeddings: bool,
) -> Result<QwenMtpInventory> {
    let two_hidden = hidden_size
        .checked_mul(2)
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: "MTP fc input width overflows u32".into(),
        })?;
    let two_intermediate =
        moe_intermediate_size
            .checked_mul(2)
            .ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: "MTP fused gate/up width overflows u32".into(),
            })?;

    for (name, expected) in [
        (
            format!("{namespace}fc.weight"),
            vec![hidden_size as u64, two_hidden as u64],
        ),
        (format!("{namespace}norm.weight"), vec![hidden_size as u64]),
        (
            format!("{namespace}pre_fc_norm_embedding.weight"),
            vec![hidden_size as u64],
        ),
        (
            format!("{namespace}pre_fc_norm_hidden.weight"),
            vec![hidden_size as u64],
        ),
    ] {
        require_shape(source, &name, &expected)?;
    }

    let global_tensors = collect_globals(source, namespace);
    let mut stages = Vec::with_capacity(hidden_layers as usize);
    for stage in 0..hidden_layers {
        let prefix = format!("{namespace}layers.{stage}.");
        let gate_up_name = format!("{prefix}mlp.experts.gate_up_proj");
        let down_name = format!("{prefix}mlp.experts.down_proj");
        let gate_up = require_shape(
            source,
            &gate_up_name,
            &[experts as u64, two_intermediate as u64, hidden_size as u64],
        )?
        .clone();
        let down = require_shape(
            source,
            &down_name,
            &[
                experts as u64,
                hidden_size as u64,
                moe_intermediate_size as u64,
            ],
        )?
        .clone();

        for role in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate.weight",
            "mlp.shared_expert.down_proj.weight",
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
            "mlp.shared_expert_gate.weight",
            "self_attn.k_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        ] {
            require_tensor(source, &format!("{prefix}{role}"))?;
        }

        stages.push(QwenMtpStageInventory {
            stage,
            expert_bank: QwenMtpExpertBank::FusedGateUp { gate_up, down },
            static_tensors: collect_stage_static(
                source,
                &prefix,
                &[gate_up_name.as_str(), down_name.as_str()],
            ),
        });
    }
    validate_stage_numbers(source, namespace, hidden_layers)?;

    Ok(QwenMtpInventory {
        flavor: QwenMtpFlavor::LegacyQwen3,
        hidden_layers,
        use_dedicated_embeddings,
        hidden_size,
        experts,
        moe_intermediate_size,
        global_tensors,
        stages,
    })
}

#[allow(clippy::too_many_arguments)]
fn inspect_qwen4(
    source: &SourceInventory,
    text_config: &serde_json::Map<String, Value>,
    namespace: &str,
    standalone: bool,
    hidden_layers: u32,
    hidden_size: u32,
    experts: u32,
    moe_intermediate_size: u32,
    use_dedicated_embeddings: bool,
) -> Result<QwenMtpInventory> {
    if use_dedicated_embeddings {
        return invalid(
            source,
            "Qwen4Exp MTP with dedicated embeddings is not supported by this inventory",
        );
    }
    let hc_count = required_u32(source, text_config.get("hc_count"), "hc_count")?;
    let hc_lowrank = required_u32(source, text_config.get("hc_lowrank"), "hc_lowrank")?;
    let shared_expert_intermediate_size = required_u32(
        source,
        text_config.get("shared_expert_intermediate_size"),
        "shared_expert_intermediate_size",
    )?;
    let hc_width = hidden_size
        .checked_mul(hc_count)
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: "Qwen4Exp MTP HC width overflows u32".into(),
        })?;

    for (name, expected) in [
        (
            format!("{namespace}fc_embedding.weight"),
            vec![hidden_size as u64, hidden_size as u64],
        ),
        (
            format!("{namespace}fc_hidden.weight"),
            vec![hidden_size as u64, hidden_size as u64],
        ),
        (
            format!("{namespace}pre_fc_norm_embedding.weight"),
            vec![hidden_size as u64],
        ),
        (
            format!("{namespace}pre_fc_norm_hidden.weight"),
            vec![hc_width as u64],
        ),
        (
            format!("{namespace}hyper_connection_mixer.hc_norm.weight"),
            vec![hc_width as u64],
        ),
        (
            format!("{namespace}hyper_connection_mixer.input_mix_weight_down.weight"),
            vec![hc_lowrank as u64, hc_width as u64],
        ),
        (
            format!("{namespace}hyper_connection_mixer.input_mix_weight_up.weight"),
            vec![hc_width as u64, hc_lowrank as u64],
        ),
    ] {
        require_shape(source, &name, &expected)?;
    }

    let global_tensors = collect_globals(source, namespace);
    let mut stages = Vec::with_capacity(hidden_layers as usize);
    for stage in 0..hidden_layers {
        let prefix = format!("{namespace}layers.{stage}.");
        let gate_name = format!("{prefix}mlp.switch_mlp.gate_proj.weight");
        let up_name = format!("{prefix}mlp.switch_mlp.up_proj.weight");
        let down_name = format!("{prefix}mlp.switch_mlp.down_proj.weight");
        let fused_gate_up_name = format!("{prefix}mlp.experts.gate_up_proj");
        let fused_down_name = format!("{prefix}mlp.experts.down_proj");

        let (expert_bank, expert_names): (QwenMtpExpertBank, Vec<String>) =
            if source.tensors.contains_key(&gate_name)
                || source.tensors.contains_key(&up_name)
                || source.tensors.contains_key(&down_name)
            {
                let gate = require_shape(
                    source,
                    &gate_name,
                    &[
                        experts as u64,
                        moe_intermediate_size as u64,
                        hidden_size as u64,
                    ],
                )?
                .clone();
                let up = require_shape(
                    source,
                    &up_name,
                    &[
                        experts as u64,
                        moe_intermediate_size as u64,
                        hidden_size as u64,
                    ],
                )?
                .clone();
                let down = require_shape(
                    source,
                    &down_name,
                    &[
                        experts as u64,
                        hidden_size as u64,
                        moe_intermediate_size as u64,
                    ],
                )?
                .clone();
                (
                    QwenMtpExpertBank::SplitGateUp { gate, up, down },
                    vec![gate_name, up_name, down_name],
                )
            } else {
                let two_intermediate = moe_intermediate_size.checked_mul(2).ok_or_else(|| {
                    ColicError::InvalidSource {
                        path: source.root.clone(),
                        detail: "Qwen4Exp MTP fused gate/up width overflows u32".into(),
                    }
                })?;
                let gate_up = require_shape(
                    source,
                    &fused_gate_up_name,
                    &[experts as u64, two_intermediate as u64, hidden_size as u64],
                )?
                .clone();
                let down = require_shape(
                    source,
                    &fused_down_name,
                    &[
                        experts as u64,
                        hidden_size as u64,
                        moe_intermediate_size as u64,
                    ],
                )?
                .clone();
                (
                    QwenMtpExpertBank::FusedGateUp { gate_up, down },
                    vec![fused_gate_up_name, fused_down_name],
                )
            };

        // These are the fixed Qwen4Exp MTP components consumed every recursive
        // draft step. Exact geometry is validated where it is architecture-
        // defining; the remaining tensor existence checks intentionally leave
        // attention/indexer output details to the Qwen4 lowering/runtime.
        for (role, expected) in [
            (
                "attn_hyper_connection.block_inject_weight.weight",
                vec![hc_count as u64, hc_width as u64],
            ),
            (
                "attn_hyper_connection.hc_norm.weight",
                vec![hc_width as u64],
            ),
            (
                "attn_hyper_connection.input_mix_weight_down.weight",
                vec![hc_lowrank as u64, hc_width as u64],
            ),
            (
                "attn_hyper_connection.input_mix_weight_up.weight",
                vec![hc_width as u64, hc_lowrank as u64],
            ),
            ("mlp.gate.weight", vec![experts as u64, hidden_size as u64]),
            (
                "mlp.shared_expert.down_proj.weight",
                vec![hidden_size as u64, shared_expert_intermediate_size as u64],
            ),
            (
                "mlp.shared_expert.gate_proj.weight",
                vec![shared_expert_intermediate_size as u64, hidden_size as u64],
            ),
            (
                "mlp.shared_expert.up_proj.weight",
                vec![shared_expert_intermediate_size as u64, hidden_size as u64],
            ),
            ("mlp.shared_expert_gate.weight", vec![1, hidden_size as u64]),
            (
                "mlp_hyper_connection.block_inject_weight.weight",
                vec![hc_count as u64, hc_width as u64],
            ),
            ("mlp_hyper_connection.hc_norm.weight", vec![hc_width as u64]),
            (
                "mlp_hyper_connection.input_mix_weight_down.weight",
                vec![hc_lowrank as u64, hc_width as u64],
            ),
            (
                "mlp_hyper_connection.input_mix_weight_up.weight",
                vec![hc_width as u64, hc_lowrank as u64],
            ),
        ] {
            require_shape(source, &format!("{prefix}{role}"), &expected)?;
        }
        for role in [
            "self_attn.indexer.index_qk_proj.weight",
            "self_attn.indexer.k_layernorm.weight",
            "self_attn.indexer.q_layernorm.weight",
            "self_attn.k_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        ] {
            require_tensor(source, &format!("{prefix}{role}"))?;
        }

        let excluded = expert_names.iter().map(String::as_str).collect::<Vec<_>>();
        stages.push(QwenMtpStageInventory {
            stage,
            expert_bank,
            static_tensors: collect_stage_static(source, &prefix, &excluded),
        });
    }
    validate_stage_numbers(source, namespace, hidden_layers)?;

    Ok(QwenMtpInventory {
        flavor: QwenMtpFlavor::Qwen4Exp {
            hc_count,
            hc_lowrank,
            shared_expert_intermediate_size,
            standalone,
        },
        hidden_layers,
        use_dedicated_embeddings,
        hidden_size,
        experts,
        moe_intermediate_size,
        global_tensors,
        stages,
    })
}

fn collect_globals(source: &SourceInventory, namespace: &str) -> BTreeMap<String, TensorRef> {
    source
        .tensors
        .iter()
        .filter(|(name, _)| {
            if namespace.is_empty() {
                !name.starts_with("layers.")
            } else {
                name.starts_with(namespace) && !name.starts_with(&format!("{namespace}layers."))
            }
        })
        .map(|(name, tensor)| (name.clone(), tensor.clone()))
        .collect()
}

fn collect_stage_static(
    source: &SourceInventory,
    prefix: &str,
    excluded: &[&str],
) -> BTreeMap<String, TensorRef> {
    source
        .tensors
        .iter()
        .filter_map(|(name, tensor)| {
            let role = name.strip_prefix(prefix)?;
            (!excluded.contains(&name.as_str())).then(|| (role.to_owned(), tensor.clone()))
        })
        .collect()
}

fn validate_stage_numbers(
    source: &SourceInventory,
    namespace: &str,
    hidden_layers: u32,
) -> Result<()> {
    let prefix = format!("{namespace}layers.");
    for name in source
        .tensors
        .keys()
        .filter(|name| name.starts_with(&prefix))
    {
        let rest = &name[prefix.len()..];
        let Some((stage, _)) = rest.split_once('.') else {
            return invalid(source, format!("invalid MTP layer tensor name `{name}`"));
        };
        let stage: u32 = stage.parse().map_err(|_| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("invalid MTP layer tensor name `{name}`"),
        })?;
        if stage >= hidden_layers {
            return invalid(
                source,
                format!(
                    "checkpoint contains MTP layer {stage} but config declares only {hidden_layers} layer(s)"
                ),
            );
        }
    }
    Ok(())
}

fn optional_u32_value(config: &Value, key: &str) -> Option<u32> {
    config.get(key)?.as_u64()?.try_into().ok()
}

fn optional_u32(root: &std::path::Path, value: Option<&Value>, key: &str) -> Result<Option<u32>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let raw = value.as_u64().ok_or_else(|| ColicError::InvalidSource {
        path: root.to_owned(),
        detail: format!("`{key}` is not a non-negative integer"),
    })?;
    let value = u32::try_from(raw).map_err(|_| ColicError::InvalidSource {
        path: root.to_owned(),
        detail: format!("`{key}` exceeds u32"),
    })?;
    Ok(Some(value))
}

fn required_u32(source: &SourceInventory, value: Option<&Value>, key: &str) -> Result<u32> {
    optional_u32(&source.root, value, key)?.ok_or_else(|| ColicError::InvalidSource {
        path: source.root.clone(),
        detail: format!("Qwen text_config is missing `{key}`"),
    })
}

fn require_tensor<'a>(source: &'a SourceInventory, name: &str) -> Result<&'a TensorRef> {
    source
        .tensors
        .get(name)
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("MTP checkpoint is missing required tensor `{name}`"),
        })
}

fn require_shape<'a>(
    source: &'a SourceInventory,
    name: &str,
    expected: &[u64],
) -> Result<&'a TensorRef> {
    let tensor = require_tensor(source, name)?;
    if tensor.shape != expected {
        return invalid(
            source,
            format!(
                "MTP tensor `{name}` has shape {:?}, expected {:?}",
                tensor.shape, expected
            ),
        );
    }
    Ok(tensor)
}

fn invalid<T>(source: &SourceInventory, detail: impl Into<String>) -> Result<T> {
    Err(ColicError::InvalidSource {
        path: source.root.clone(),
        detail: detail.into(),
    })
}
