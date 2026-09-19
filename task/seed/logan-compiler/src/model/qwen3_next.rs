//! Qwen3-Next / Qwen3-Coder-Next source frontend.
//!
//! Native Hugging Face Qwen3-Next checkpoints store routed experts separately
//! but fuse the Gated DeltaNet input projections as `in_proj_qkvz` and
//! `in_proj_ba`. Logan's runtime intentionally keeps the older four-projection
//! execution ABI (`qkv`, `z`, `a`, `b`) because the Metal/ANE paths are already
//! optimized around it. This frontend therefore exposes deterministic virtual
//! BF16 tensors; target lowering streams/reorders the fused source rows into
//! that canonical ABI without materializing a whole fused matrix.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use serde_json::Value;

use crate::{
    error::{ColicError, Result},
    ir::{Architecture, ModelGeometry, SemanticModel},
    source::{self, SourceInventory, TensorRef},
};

pub(crate) const QKVZ_SPLIT_PREFIX: &str = "QWEN3NEXT_QKVZ_BF16:";
pub(crate) const BA_SPLIT_PREFIX: &str = "QWEN3NEXT_BA_BF16:";

pub struct Qwen3NextFrontend;

impl Qwen3NextFrontend {
    pub fn probe(source: &SourceInventory) -> Result<bool> {
        let Some(config) = source::config(&source.root)? else {
            return Ok(false);
        };
        Ok(config.get("model_type").and_then(Value::as_str) == Some("qwen3_next"))
    }

    pub fn build(source: &SourceInventory) -> Result<SemanticModel> {
        let config = source::config(&source.root)?.ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: "Qwen3-Next source is missing config.json".into(),
        })?;
        if config.get("model_type").and_then(Value::as_str) != Some("qwen3_next") {
            return invalid(&source.root, "config model_type is not qwen3_next");
        }
        // Logan currently implements the all-MoE Qwen3-Next subset used by
        // Qwen3-Coder-Next. Fail closed on broader Qwen3-Next variants rather
        // than silently executing a dense MLP or different router contract as MoE.
        let sparse_step = config
            .get("decoder_sparse_step")
            .and_then(Value::as_u64)
            .unwrap_or(1);
        if sparse_step != 1 {
            return invalid(
                &source.root,
                format!("Qwen3-Next decoder_sparse_step={sparse_step} is unsupported; expected 1"),
            );
        }
        if config
            .get("mlp_only_layers")
            .and_then(Value::as_array)
            .is_some_and(|layers| !layers.is_empty())
        {
            return invalid(
                &source.root,
                "Qwen3-Next mlp_only_layers is non-empty; dense-only MLP layers are not implemented",
            );
        }
        if config
            .get("norm_topk_prob")
            .and_then(Value::as_bool)
            .is_some_and(|enabled| !enabled)
        {
            return invalid(
                &source.root,
                "Qwen3-Next norm_topk_prob=false is unsupported; Logan normalizes selected expert probabilities",
            );
        }
        if config
            .get("attention_bias")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return invalid(
                &source.root,
                "Qwen3-Next attention_bias=true is unsupported",
            );
        }
        if config
            .get("hidden_act")
            .and_then(Value::as_str)
            .is_some_and(|act| act != "silu")
        {
            return invalid(
                &source.root,
                "Qwen3-Next currently requires hidden_act=silu",
            );
        }
        if config
            .get("rope_scaling")
            .is_some_and(|value| !value.is_null())
        {
            return invalid(&source.root, "Qwen3-Next rope_scaling is not implemented");
        }

        let layers = required_u32(&source.root, &config, "num_hidden_layers")?;
        let hidden = required_u32(&source.root, &config, "hidden_size")?;
        let experts = required_u32(&source.root, &config, "num_experts")?;
        let inter = required_u32(&source.root, &config, "moe_intermediate_size")?;
        let vocab = required_u32(&source.root, &config, "vocab_size")?;
        let heads = required_u32(&source.root, &config, "num_attention_heads")?;
        let head_dim = required_u32(&source.root, &config, "head_dim")?;
        let kv_heads = required_u32(&source.root, &config, "num_key_value_heads")?;
        let lin_k_heads = required_u32(&source.root, &config, "linear_num_key_heads")?;
        let lin_k_dim = required_u32(&source.root, &config, "linear_key_head_dim")?;
        let lin_v_heads = required_u32(&source.root, &config, "linear_num_value_heads")?;
        let lin_v_dim = required_u32(&source.root, &config, "linear_value_head_dim")?;
        let conv_kernel = required_u32(&source.root, &config, "linear_conv_kernel_dim")?;
        let shared_inter = required_u32(&source.root, &config, "shared_expert_intermediate_size")?;
        if lin_v_heads % lin_k_heads != 0 {
            return invalid(
                &source.root,
                format!(
                    "linear value heads {lin_v_heads} are not divisible by key heads {lin_k_heads}"
                ),
            );
        }
        let layer_types = layer_types(&source.root, &config, layers)?;
        let base = "model";

        let geometry = ModelGeometry {
            hidden_size: hidden,
            layers,
            routed_experts_per_layer: experts,
            moe_intermediate_size: inter,
            vocab_size: vocab,
            hc_mult: 0,
            num_hash_layers: 0,
            experts_per_token: required_u32(&source.root, &config, "num_experts_per_tok")?,
            attention_heads: heads,
            head_dim,
            num_key_value_heads: kv_heads,
            linear_key_head_dim: lin_k_dim,
            q_lora_rank: 0,
            o_groups: 1,
            o_lora_rank: 0,
            index_heads: 0,
            index_head_dim: 0,
            // Qwen3-Next full-attention layers are ordinary gated attention,
            // not QSA/indexer layers. Keep the index-compression axis zero.
            compression_ratios: vec![0; layers as usize],
        };

        let mut consumed = BTreeSet::new();
        // Qwen4's expert builder is intentionally representation-generic: it
        // accepts the separate gate/up/down layout emitted by Qwen3-Coder-Next.
        let routed_experts =
            super::qwen4_exp::build_experts(source, base, &geometry, &mut consumed)?;

        let mut global_tensors = BTreeMap::new();
        add_required(
            source,
            &mut global_tensors,
            &mut consumed,
            "embed.weight",
            "model.embed_tokens.weight",
            &[vocab as u64, hidden as u64],
        )?;
        add_required(
            source,
            &mut global_tensors,
            &mut consumed,
            "norm.weight",
            "model.norm.weight",
            &[hidden as u64],
        )?;
        add_required(
            source,
            &mut global_tensors,
            &mut consumed,
            "head.weight",
            "lm_head.weight",
            &[vocab as u64, hidden as u64],
        )?;

        let mut layer_static_tensors = BTreeMap::new();
        for layer in 0..layers {
            let lp = format!("{base}.layers.{layer}");
            let mut out = BTreeMap::new();
            add_layer_required(
                source,
                &mut out,
                &mut consumed,
                "input_layernorm.weight",
                &format!("{lp}.input_layernorm.weight"),
                &[hidden as u64],
            )?;
            add_layer_required(
                source,
                &mut out,
                &mut consumed,
                "post_attention_layernorm.weight",
                &format!("{lp}.post_attention_layernorm.weight"),
                &[hidden as u64],
            )?;
            add_layer_required(
                source,
                &mut out,
                &mut consumed,
                "mlp.gate.weight",
                &format!("{lp}.mlp.gate.weight"),
                &[experts as u64, hidden as u64],
            )?;
            add_layer_required(
                source,
                &mut out,
                &mut consumed,
                "mlp.shared_expert.gate_proj.weight",
                &format!("{lp}.mlp.shared_expert.gate_proj.weight"),
                &[shared_inter as u64, hidden as u64],
            )?;
            add_layer_required(
                source,
                &mut out,
                &mut consumed,
                "mlp.shared_expert.up_proj.weight",
                &format!("{lp}.mlp.shared_expert.up_proj.weight"),
                &[shared_inter as u64, hidden as u64],
            )?;
            add_layer_required(
                source,
                &mut out,
                &mut consumed,
                "mlp.shared_expert.down_proj.weight",
                &format!("{lp}.mlp.shared_expert.down_proj.weight"),
                &[hidden as u64, shared_inter as u64],
            )?;
            add_layer_required(
                source,
                &mut out,
                &mut consumed,
                "mlp.shared_expert_gate.weight",
                &format!("{lp}.mlp.shared_expert_gate.weight"),
                &[1, hidden as u64],
            )?;

            if layer_types[layer as usize] == "linear_attention" {
                let value_dim = lin_v_heads
                    .checked_mul(lin_v_dim)
                    .ok_or_else(|| missing(source, "linear value dimension overflow"))?;
                let key_dim = lin_k_heads
                    .checked_mul(lin_k_dim)
                    .ok_or_else(|| missing(source, "linear key dimension overflow"))?;
                let conv_dim = key_dim
                    .checked_mul(2)
                    .and_then(|v| v.checked_add(value_dim))
                    .ok_or_else(|| missing(source, "linear conv dimension overflow"))?;
                add_layer_required(
                    source,
                    &mut out,
                    &mut consumed,
                    "linear_attn.A_log",
                    &format!("{lp}.linear_attn.A_log"),
                    &[lin_v_heads as u64],
                )?;
                add_layer_required(
                    source,
                    &mut out,
                    &mut consumed,
                    "linear_attn.dt_bias",
                    &format!("{lp}.linear_attn.dt_bias"),
                    &[lin_v_heads as u64],
                )?;
                add_layer_required(
                    source,
                    &mut out,
                    &mut consumed,
                    "linear_attn.conv1d.weight",
                    &format!("{lp}.linear_attn.conv1d.weight"),
                    &[conv_dim as u64, 1, conv_kernel as u64],
                )?;
                add_layer_required(
                    source,
                    &mut out,
                    &mut consumed,
                    "linear_attn.norm.weight",
                    &format!("{lp}.linear_attn.norm.weight"),
                    &[lin_v_dim as u64],
                )?;
                add_layer_required(
                    source,
                    &mut out,
                    &mut consumed,
                    "linear_attn.out_proj.weight",
                    &format!("{lp}.linear_attn.out_proj.weight"),
                    &[hidden as u64, value_dim as u64],
                )?;

                let qkvz_name = format!("{lp}.linear_attn.in_proj_qkvz.weight");
                let qkvz = required_bf16(
                    source,
                    &qkvz_name,
                    &[(key_dim * 2 + value_dim * 2) as u64, hidden as u64],
                )?;
                let qkv = virtual_qkvz_view(
                    &qkvz,
                    "qkv",
                    hidden,
                    lin_k_heads,
                    lin_k_dim,
                    lin_v_heads,
                    lin_v_dim,
                    conv_dim,
                )?;
                let z = virtual_qkvz_view(
                    &qkvz,
                    "z",
                    hidden,
                    lin_k_heads,
                    lin_k_dim,
                    lin_v_heads,
                    lin_v_dim,
                    value_dim,
                )?;
                out.insert("linear_attn.in_proj_qkv.weight".into(), qkv);
                out.insert("linear_attn.in_proj_z.weight".into(), z);
                consumed.insert(qkvz_name);

                let ba_name = format!("{lp}.linear_attn.in_proj_ba.weight");
                let ba =
                    required_bf16(source, &ba_name, &[(lin_v_heads * 2) as u64, hidden as u64])?;
                out.insert(
                    "linear_attn.in_proj_b.weight".into(),
                    virtual_ba_view(&ba, "b", hidden, lin_k_heads, lin_v_heads)?,
                );
                out.insert(
                    "linear_attn.in_proj_a.weight".into(),
                    virtual_ba_view(&ba, "a", hidden, lin_k_heads, lin_v_heads)?,
                );
                consumed.insert(ba_name);
            } else {
                add_layer_required(
                    source,
                    &mut out,
                    &mut consumed,
                    "self_attn.q_proj.weight",
                    &format!("{lp}.self_attn.q_proj.weight"),
                    &[(heads * head_dim * 2) as u64, hidden as u64],
                )?;
                add_layer_required(
                    source,
                    &mut out,
                    &mut consumed,
                    "self_attn.k_proj.weight",
                    &format!("{lp}.self_attn.k_proj.weight"),
                    &[(kv_heads * head_dim) as u64, hidden as u64],
                )?;
                add_layer_required(
                    source,
                    &mut out,
                    &mut consumed,
                    "self_attn.v_proj.weight",
                    &format!("{lp}.self_attn.v_proj.weight"),
                    &[(kv_heads * head_dim) as u64, hidden as u64],
                )?;
                add_layer_required(
                    source,
                    &mut out,
                    &mut consumed,
                    "self_attn.o_proj.weight",
                    &format!("{lp}.self_attn.o_proj.weight"),
                    &[hidden as u64, (heads * head_dim) as u64],
                )?;
                add_layer_required(
                    source,
                    &mut out,
                    &mut consumed,
                    "self_attn.q_norm.weight",
                    &format!("{lp}.self_attn.q_norm.weight"),
                    &[head_dim as u64],
                )?;
                add_layer_required(
                    source,
                    &mut out,
                    &mut consumed,
                    "self_attn.k_norm.weight",
                    &format!("{lp}.self_attn.k_norm.weight"),
                    &[head_dim as u64],
                )?;
            }
            layer_static_tensors.insert(layer, out);
        }

        let resident_tensors = source
            .tensors
            .iter()
            .filter(|(name, _)| !consumed.contains(*name))
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect();

        Ok(SemanticModel {
            architecture: Architecture::Qwen3Next,
            geometry,
            routed_experts,
            global_tensors,
            layer_static_tensors,
            resident_tensors,
        })
    }
}

fn layer_types(root: &Path, config: &Value, layers: u32) -> Result<Vec<String>> {
    if let Some(values) = config.get("layer_types").and_then(Value::as_array) {
        if values.len() != layers as usize {
            return invalid(
                root,
                format!(
                    "layer_types has {} entries, expected {layers}",
                    values.len()
                ),
            );
        }
        return values
            .iter()
            .enumerate()
            .map(|(index, value)| match value.as_str() {
                Some("linear_attention" | "full_attention") => {
                    Ok(value.as_str().unwrap().to_owned())
                }
                other => invalid(
                    root,
                    format!("unsupported layer type at {index}: {other:?}"),
                ),
            })
            .collect();
    }
    let interval = required_u32(root, config, "full_attention_interval")?;
    Ok((0..layers)
        .map(|layer| {
            if (layer + 1) % interval == 0 {
                "full_attention".to_owned()
            } else {
                "linear_attention".to_owned()
            }
        })
        .collect())
}

fn virtual_qkvz_view(
    tensor: &TensorRef,
    role: &str,
    hidden: u32,
    key_heads: u32,
    key_dim: u32,
    value_heads: u32,
    value_dim: u32,
    out_rows: u32,
) -> Result<TensorRef> {
    let len = u64::from(out_rows)
        .checked_mul(u64::from(hidden))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| ColicError::InvalidSource {
            path: tensor.source.clone(),
            detail: "Qwen3-Next virtual qkvz size overflow".into(),
        })?;
    Ok(TensorRef {
        source: tensor.source.clone(),
        offset: tensor.offset,
        len,
        dtype: format!(
            "{QKVZ_SPLIT_PREFIX}{role}:{key_heads}:{key_dim}:{value_heads}:{value_dim}:{hidden}"
        ),
        shape: vec![out_rows as u64, hidden as u64],
    })
}

fn virtual_ba_view(
    tensor: &TensorRef,
    role: &str,
    hidden: u32,
    key_heads: u32,
    value_heads: u32,
) -> Result<TensorRef> {
    let len = u64::from(value_heads)
        .checked_mul(u64::from(hidden))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| ColicError::InvalidSource {
            path: tensor.source.clone(),
            detail: "Qwen3-Next virtual ba size overflow".into(),
        })?;
    Ok(TensorRef {
        source: tensor.source.clone(),
        offset: tensor.offset,
        len,
        dtype: format!("{BA_SPLIT_PREFIX}{role}:{key_heads}:{value_heads}:{hidden}"),
        shape: vec![value_heads as u64, hidden as u64],
    })
}

fn add_required(
    source: &SourceInventory,
    out: &mut BTreeMap<String, TensorRef>,
    consumed: &mut BTreeSet<String>,
    role: &str,
    name: &str,
    shape: &[u64],
) -> Result<()> {
    out.insert(role.to_owned(), required_bf16(source, name, shape)?);
    consumed.insert(name.to_owned());
    Ok(())
}

fn add_layer_required(
    source: &SourceInventory,
    out: &mut BTreeMap<String, TensorRef>,
    consumed: &mut BTreeSet<String>,
    role: &str,
    name: &str,
    shape: &[u64],
) -> Result<()> {
    add_required(source, out, consumed, role, name, shape)
}

fn required_bf16(source: &SourceInventory, name: &str, shape: &[u64]) -> Result<TensorRef> {
    let tensor = source
        .tensors
        .get(name)
        .ok_or_else(|| missing(source, name))?;
    if tensor.dtype != "BF16" || tensor.shape != shape {
        return invalid(
            &source.root,
            format!(
                "tensor `{name}` has {}/{:?}, expected BF16/{shape:?}",
                tensor.dtype, tensor.shape
            ),
        );
    }
    let expected = shape
        .iter()
        .try_fold(2_u64, |bytes, dim| bytes.checked_mul(*dim))
        .ok_or_else(|| missing(source, format!("tensor `{name}` byte size overflow")))?;
    if tensor.len != expected {
        return invalid(
            &source.root,
            format!("tensor `{name}` len {} != {expected}", tensor.len),
        );
    }
    Ok(tensor.clone())
}

fn required_u32(root: &Path, config: &Value, field: &str) -> Result<u32> {
    config
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .filter(|v| *v > 0)
        .ok_or_else(|| ColicError::InvalidSource {
            path: root.to_owned(),
            detail: format!("config `{field}` must be a positive u32"),
        })
}

fn missing(source: &SourceInventory, detail: impl Into<String>) -> ColicError {
    ColicError::InvalidSource {
        path: source.root.clone(),
        detail: detail.into(),
    }
}

fn invalid<T>(path: &Path, detail: impl Into<String>) -> Result<T> {
    Err(ColicError::InvalidSource {
        path: path.to_owned(),
        detail: detail.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn add(
        tensors: &mut BTreeMap<String, TensorRef>,
        source: &Path,
        next: &mut u64,
        name: impl Into<String>,
        shape: &[u64],
    ) {
        let len = shape.iter().copied().product::<u64>() * 2;
        tensors.insert(
            name.into(),
            TensorRef {
                source: source.to_owned(),
                offset: *next,
                len,
                dtype: "BF16".into(),
                shape: shape.to_vec(),
            },
        );
        *next += len + 16;
    }

    fn fixture() -> (std::path::PathBuf, SourceInventory) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("logan-qwen3-next-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("config.json"),
            r#"{
              "model_type": "qwen3_next",
              "num_hidden_layers": 4,
              "full_attention_interval": 4,
              "hidden_size": 4,
              "num_experts": 2,
              "moe_intermediate_size": 3,
              "shared_expert_intermediate_size": 5,
              "vocab_size": 7,
              "num_experts_per_tok": 1,
              "num_attention_heads": 2,
              "head_dim": 2,
              "num_key_value_heads": 1,
              "linear_num_key_heads": 1,
              "linear_key_head_dim": 2,
              "linear_num_value_heads": 2,
              "linear_value_head_dim": 3,
              "linear_conv_kernel_dim": 4
            }"#,
        )
        .unwrap();
        let weights = root.join("weights.bin");
        fs::write(&weights, []).unwrap();
        let mut tensors = BTreeMap::new();
        let mut next = 4096;
        add(
            &mut tensors,
            &weights,
            &mut next,
            "model.embed_tokens.weight",
            &[7, 4],
        );
        add(&mut tensors, &weights, &mut next, "model.norm.weight", &[4]);
        add(&mut tensors, &weights, &mut next, "lm_head.weight", &[7, 4]);

        for layer in 0..4 {
            let lp = format!("model.layers.{layer}");
            add(
                &mut tensors,
                &weights,
                &mut next,
                format!("{lp}.input_layernorm.weight"),
                &[4],
            );
            add(
                &mut tensors,
                &weights,
                &mut next,
                format!("{lp}.post_attention_layernorm.weight"),
                &[4],
            );
            add(
                &mut tensors,
                &weights,
                &mut next,
                format!("{lp}.mlp.gate.weight"),
                &[2, 4],
            );
            add(
                &mut tensors,
                &weights,
                &mut next,
                format!("{lp}.mlp.shared_expert.gate_proj.weight"),
                &[5, 4],
            );
            add(
                &mut tensors,
                &weights,
                &mut next,
                format!("{lp}.mlp.shared_expert.up_proj.weight"),
                &[5, 4],
            );
            add(
                &mut tensors,
                &weights,
                &mut next,
                format!("{lp}.mlp.shared_expert.down_proj.weight"),
                &[4, 5],
            );
            add(
                &mut tensors,
                &weights,
                &mut next,
                format!("{lp}.mlp.shared_expert_gate.weight"),
                &[1, 4],
            );
            for expert in 0..2 {
                add(
                    &mut tensors,
                    &weights,
                    &mut next,
                    format!("{lp}.mlp.experts.{expert}.gate_proj.weight"),
                    &[3, 4],
                );
                add(
                    &mut tensors,
                    &weights,
                    &mut next,
                    format!("{lp}.mlp.experts.{expert}.up_proj.weight"),
                    &[3, 4],
                );
                add(
                    &mut tensors,
                    &weights,
                    &mut next,
                    format!("{lp}.mlp.experts.{expert}.down_proj.weight"),
                    &[4, 3],
                );
            }
            if layer == 3 {
                add(
                    &mut tensors,
                    &weights,
                    &mut next,
                    format!("{lp}.self_attn.q_proj.weight"),
                    &[8, 4],
                );
                add(
                    &mut tensors,
                    &weights,
                    &mut next,
                    format!("{lp}.self_attn.k_proj.weight"),
                    &[2, 4],
                );
                add(
                    &mut tensors,
                    &weights,
                    &mut next,
                    format!("{lp}.self_attn.v_proj.weight"),
                    &[2, 4],
                );
                add(
                    &mut tensors,
                    &weights,
                    &mut next,
                    format!("{lp}.self_attn.o_proj.weight"),
                    &[4, 4],
                );
                add(
                    &mut tensors,
                    &weights,
                    &mut next,
                    format!("{lp}.self_attn.q_norm.weight"),
                    &[2],
                );
                add(
                    &mut tensors,
                    &weights,
                    &mut next,
                    format!("{lp}.self_attn.k_norm.weight"),
                    &[2],
                );
            } else {
                add(
                    &mut tensors,
                    &weights,
                    &mut next,
                    format!("{lp}.linear_attn.A_log"),
                    &[2],
                );
                add(
                    &mut tensors,
                    &weights,
                    &mut next,
                    format!("{lp}.linear_attn.dt_bias"),
                    &[2],
                );
                add(
                    &mut tensors,
                    &weights,
                    &mut next,
                    format!("{lp}.linear_attn.conv1d.weight"),
                    &[10, 1, 4],
                );
                add(
                    &mut tensors,
                    &weights,
                    &mut next,
                    format!("{lp}.linear_attn.in_proj_qkvz.weight"),
                    &[16, 4],
                );
                add(
                    &mut tensors,
                    &weights,
                    &mut next,
                    format!("{lp}.linear_attn.in_proj_ba.weight"),
                    &[4, 4],
                );
                add(
                    &mut tensors,
                    &weights,
                    &mut next,
                    format!("{lp}.linear_attn.norm.weight"),
                    &[3],
                );
                add(
                    &mut tensors,
                    &weights,
                    &mut next,
                    format!("{lp}.linear_attn.out_proj.weight"),
                    &[4, 6],
                );
            }
        }

        let source_stored_bytes = tensors.values().map(|t| t.len).sum();
        let inventory = SourceInventory {
            root: root.clone(),
            files: vec![root.join("config.json"), weights],
            tensors,
            source_stored_bytes,
            dtype_counts: BTreeMap::from([("BF16".into(), 1)]),
            source_fingerprint: "fixture".into(),
            config_fingerprint: None,
            architecture_hint: Some("qwen3_next".into()),
        };
        (root, inventory)
    }

    fn set_config_field(root: &Path, key: &str, value: serde_json::Value) {
        let path = root.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        config
            .as_object_mut()
            .unwrap()
            .insert(key.to_owned(), value);
        fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
    }

    #[test]
    fn rejects_qwen3_next_variants_that_do_not_match_coder_next_execution_contract() {
        let cases = [
            (
                "decoder_sparse_step",
                serde_json::json!(2),
                "decoder_sparse_step=2",
            ),
            ("mlp_only_layers", serde_json::json!([0]), "mlp_only_layers"),
            (
                "norm_topk_prob",
                serde_json::json!(false),
                "norm_topk_prob=false",
            ),
            (
                "attention_bias",
                serde_json::json!(true),
                "attention_bias=true",
            ),
            ("hidden_act", serde_json::json!("gelu"), "hidden_act=silu"),
            (
                "rope_scaling",
                serde_json::json!({"rope_type":"yarn"}),
                "rope_scaling",
            ),
        ];
        for (key, value, expected) in cases {
            let (root, inventory) = fixture();
            set_config_field(&root, key, value);
            let error = Qwen3NextFrontend::build(&inventory)
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "{key}: {error}");
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn coder_next_frontend_derives_hybrid_schedule_and_virtual_gdn_projections() {
        let (root, inventory) = fixture();
        let model = Qwen3NextFrontend::build(&inventory).unwrap();
        assert_eq!(model.architecture, Architecture::Qwen3Next);
        assert_eq!(model.geometry.layers, 4);
        assert_eq!(model.geometry.routed_experts_per_layer, 2);
        assert_eq!(model.geometry.experts_per_token, 1);
        assert_eq!(model.geometry.compression_ratios, vec![0, 0, 0, 0]);
        assert_eq!(model.routed_experts.len(), 8);
        assert!(model.resident_tensors.is_empty());

        let gdn = &model.layer_static_tensors[&0];
        let qkv = &gdn["linear_attn.in_proj_qkv.weight"];
        assert_eq!(qkv.shape, vec![10, 4]);
        assert!(qkv.dtype.starts_with(QKVZ_SPLIT_PREFIX));
        let z = &gdn["linear_attn.in_proj_z.weight"];
        assert_eq!(z.shape, vec![6, 4]);
        let a = &gdn["linear_attn.in_proj_a.weight"];
        let b = &gdn["linear_attn.in_proj_b.weight"];
        assert_eq!(a.shape, vec![2, 4]);
        assert_eq!(b.shape, vec![2, 4]);
        assert!(a.dtype.starts_with(BA_SPLIT_PREFIX));

        let full = &model.layer_static_tensors[&3];
        assert_eq!(full["self_attn.q_proj.weight"].shape, vec![8, 4]);
        assert!(!full.contains_key("linear_attn.in_proj_qkv.weight"));
        fs::remove_dir_all(root).unwrap();
    }
}
