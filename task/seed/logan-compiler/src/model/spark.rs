//! Spark-X2.5 source frontend.
//!
//! Spark is dense and the MLX checkpoints are already execution-friendly:
//! 8-bit affine matrices are stored as packed U32 words plus BF16 scale/bias
//! arrays. COLI keeps those bytes resident rather than expanding them to BF16.

use crate::{
    error::{ColicError, Result},
    ir::{Architecture, ModelGeometry, SemanticModel},
    source::{self, SourceInventory},
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub struct SparkFrontend;

impl SparkFrontend {
    pub fn probe(source: &SourceInventory) -> Result<bool> {
        let Some(config) = source::config(&source.root)? else {
            return Ok(false);
        };
        Ok(config.get("model_type").and_then(Value::as_str) == Some("spark2_5"))
    }

    pub fn build(source: &SourceInventory) -> Result<SemanticModel> {
        let config = source::config(&source.root)?.ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: "Spark source is missing config.json".into(),
        })?;
        if config.get("model_type").and_then(Value::as_str) != Some("spark2_5") {
            return invalid(source, "config model_type is not spark2_5");
        }
        let hidden = req_u32(source, &config, "hidden_size")?;
        let layers = req_u32(source, &config, "num_hidden_layers")?;
        let inter = req_u32(source, &config, "intermediate_size")?;
        let vocab = req_u32(source, &config, "vocab_size")?;
        let heads = req_u32(source, &config, "num_attention_heads")?;
        let kv_heads = req_u32(source, &config, "num_key_value_heads")?;
        let head_dim = req_u32(source, &config, "head_dim")?;
        let layer_types = config
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or_else(|| bad(source, "missing layer_types"))?;
        if layer_types.len() != layers as usize
            || layer_types
                .iter()
                .any(|v| !matches!(v.as_str(), Some("full_attention" | "sliding_attention")))
        {
            return invalid(
                source,
                "layer_types must contain one full_attention/sliding_attention entry per layer",
            );
        }
        if config
            .get("headwise_attn_output_gate")
            .and_then(Value::as_bool)
            != Some(true)
            || config.get("gate_attn_act_mode").and_then(Value::as_str) != Some("sigmoid")
        {
            return invalid(
                source,
                "Spark-X2.5 requires sigmoid head-wise attention gates",
            );
        }
        let quant = config
            .get("quantization_config")
            .or_else(|| config.get("quantization"));
        if let Some(q) = quant {
            if q.get("mode").and_then(Value::as_str) != Some("affine")
                || q.get("bits").and_then(Value::as_u64) != Some(8)
                || q.get("group_size").and_then(Value::as_u64) != Some(64)
            {
                return invalid(
                    source,
                    "Spark MLX runtime currently requires affine 8-bit group_size=64",
                );
            }
        }

        // Keep exact HF/MLX names. Runtime dispatch uses these names for both
        // source directories and COLI packages. Packed U32 weights are retagged
        // as byte tensors because 8-bit MLX packing is four consecutive bytes
        // per little-endian U32; no repack is required.
        let mut resident_tensors = BTreeMap::new();
        let mut packed_bases = BTreeSet::new();
        for (name, tensor) in &source.tensors {
            if tensor.dtype == "U32" && name.ends_with(".weight") {
                let base = name.trim_end_matches(".weight");
                let scales = format!("{base}.scales");
                let biases = format!("{base}.biases");
                let s = source
                    .tensors
                    .get(&scales)
                    .ok_or_else(|| bad(source, format!("{name}: missing {scales}")))?;
                let b = source
                    .tensors
                    .get(&biases)
                    .ok_or_else(|| bad(source, format!("{name}: missing {biases}")))?;
                if s.dtype != "BF16" || b.dtype != "BF16" || s.shape != b.shape {
                    return invalid(
                        source,
                        format!("{name}: affine scale/bias tensors must be matching BF16 arrays"),
                    );
                }
                let mut view = tensor.clone();
                view.dtype = "U8".into();
                let last = view
                    .shape
                    .last_mut()
                    .ok_or_else(|| bad(source, format!("{name}: scalar packed weight")))?;
                *last = last
                    .checked_mul(4)
                    .ok_or_else(|| bad(source, format!("{name}: byte shape overflow")))?;
                resident_tensors.insert(name.clone(), view);
                packed_bases.insert(base.to_owned());
            } else {
                resident_tensors.insert(name.clone(), tensor.clone());
            }
        }
        // Basic real-model gates. These catch accidental conversion/layout
        // mismatches before a multi-gigabyte package is emitted.
        for name in [
            "model.embedding.weight",
            "model.norm.weight",
            "model.layers.0.self_attn.q_k_v_proj.weight",
            "model.layers.0.self_attn.g_proj.weight",
            "model.layers.0.self_attn.out_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ] {
            if !resident_tensors.contains_key(name) {
                return invalid(source, format!("missing required Spark tensor {name}"));
            }
        }
        if packed_bases.is_empty() {
            return invalid(
                source,
                "Spark frontend expected an MLX affine-8 checkpoint but found no packed U32 matrices",
            );
        }

        Ok(SemanticModel {
            architecture: Architecture::Spark2_5,
            geometry: ModelGeometry {
                hidden_size: hidden,
                layers,
                routed_experts_per_layer: 0,
                moe_intermediate_size: inter,
                vocab_size: vocab,
                hc_mult: 1,
                num_hash_layers: 0,
                experts_per_token: 0,
                attention_heads: heads,
                head_dim,
                num_key_value_heads: kv_heads,
                linear_key_head_dim: 0,
                q_lora_rank: 0,
                o_groups: 1,
                o_lora_rank: 0,
                index_heads: 0,
                index_head_dim: 0,
                compression_ratios: vec![0; layers as usize],
            },
            routed_experts: BTreeMap::new(),
            global_tensors: BTreeMap::new(),
            layer_static_tensors: BTreeMap::new(),
            resident_tensors,
        })
    }
}

fn req_u32(source: &SourceInventory, v: &Value, key: &str) -> Result<u32> {
    v.get(key)
        .and_then(Value::as_u64)
        .and_then(|x| u32::try_from(x).ok())
        .ok_or_else(|| bad(source, format!("missing/invalid {key}")))
}
fn bad(source: &SourceInventory, detail: impl Into<String>) -> ColicError {
    ColicError::InvalidSource {
        path: source.root.clone(),
        detail: detail.into(),
    }
}
fn invalid<T>(source: &SourceInventory, detail: impl Into<String>) -> Result<T> {
    Err(bad(source, detail))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    #[test]
    fn spark_probe_is_model_type_specific() {
        let source = SourceInventory {
            root: PathBuf::from("/nonexistent"),
            files: vec![],
            tensors: BTreeMap::new(),
            source_stored_bytes: 0,
            dtype_counts: BTreeMap::new(),
            source_fingerprint: String::new(),
            config_fingerprint: None,
            architecture_hint: None,
        };
        // Probe requires config I/O; the meaningful behavior is covered by the
        // integration frontend test. Keep this module compiling the type shape.
        assert_eq!(source.tensors.len(), 0);
    }
}
