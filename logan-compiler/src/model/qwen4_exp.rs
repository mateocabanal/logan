//! Qwen4Exp / Qwen3.8 Flash Next source frontend.
//!
//! The runtime already executes qwen4_exp COLI packages. This adapter closes
//! the compiler-side gap for native BF16 checkpoints, including REAP-pruned
//! checkpoints whose expert count differs from the upstream 512.
//!
//! Important source/layout properties:
//! - real checkpoints may use `model.language_model.*`; the deterministic
//!   fixture uses `model.*`;
//! - experts may be per-expert gate_up/down tensors or fused per-layer tensors;
//! - PLE is global in HF but layer-scoped in COLI (ple_layer_ids is 1-based);
//! - a BF16 PLE n-gram table is converted to streamed E4M3 shard records by
//!   the tensor lowerer. It is never materialized in RAM by the runtime.

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

pub(crate) const PLE_F8_FROM_BF16_S16: &str = "BF16_TO_F8_E4M3_S16";
pub(crate) const PLE_SCALE_CONST_S16: &str = "CONST_BF16_PLE_SCALE_S16";
const PLE_SHARDS: u64 = 128;

pub struct Qwen4ExpFrontend;

impl Qwen4ExpFrontend {
    pub fn probe(source: &SourceInventory) -> Result<bool> {
        let Some(config) = source::config(&source.root)? else {
            return Ok(false);
        };
        Ok(is_qwen4_model_type(model_type(&config)))
    }

    pub fn build(source: &SourceInventory) -> Result<SemanticModel> {
        let config = source::config(&source.root)?.ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: "Qwen4Exp source is missing config.json".into(),
        })?;
        if !is_qwen4_model_type(model_type(&config)) {
            return invalid(&source.root, "config model_type is not qwen4_exp/qwen4_exp_text");
        }
        let tc = config.get("text_config").unwrap_or(&config);
        let layers = required_u32(&source.root, tc, "num_hidden_layers")?;
        let hidden = required_u32(&source.root, tc, "hidden_size")?;
        let experts = required_u32(&source.root, tc, "num_experts")?;
        let inter = required_u32(&source.root, tc, "moe_intermediate_size")?;
        let vocab = required_u32(&source.root, tc, "vocab_size")?;
        let layer_types = required_layer_types(&source.root, tc, layers)?;
        let base = text_base(source)?;

        let geometry = ModelGeometry {
            hidden_size: hidden,
            layers,
            routed_experts_per_layer: experts,
            moe_intermediate_size: inter,
            vocab_size: vocab,
            hc_mult: optional_u32(tc, "hc_count").unwrap_or(1),
            num_hash_layers: 0,
            experts_per_token: required_u32(&source.root, tc, "num_experts_per_tok")?,
            attention_heads: required_u32(&source.root, tc, "num_attention_heads")?,
            head_dim: required_u32(&source.root, tc, "head_dim")?,
            num_key_value_heads: required_u32(&source.root, tc, "num_key_value_heads")?,
            linear_key_head_dim: required_u32(&source.root, tc, "linear_key_head_dim")?,
            q_lora_rank: 0,
            o_groups: 1,
            o_lora_rank: 0,
            index_heads: optional_u32(tc, "indexer_n_heads").unwrap_or(0),
            index_head_dim: optional_u32(tc, "indexer_head_dim").unwrap_or(0),
            compression_ratios: layer_types
                .iter()
                .map(|ty| {
                    if ty == "full_attention" {
                        optional_u32(tc, "indexer_compress_ratio").unwrap_or(0)
                    } else {
                        0
                    }
                })
                .collect(),
        };

        let mut consumed = BTreeSet::new();
        let routed_experts = build_experts(source, &base, &geometry, &mut consumed)?;

        let mut global_tensors = BTreeMap::new();
        let embed_name = format!("{base}.embed_tokens.weight");
        global_tensors.insert(
            "embed.weight".into(),
            required_bf16(source, &embed_name, &[vocab as u64, hidden as u64])?,
        );
        consumed.insert(embed_name);

        let head_name = first_existing_name(
            source,
            &["lm_head.weight".to_owned(), format!("{base}.lm_head.weight")],
        )
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: "missing lm_head.weight".into(),
        })?;
        global_tensors.insert(
            "head.weight".into(),
            required_bf16(source, &head_name, &[vocab as u64, hidden as u64])?,
        );
        consumed.insert(head_name);

        let norm_name = format!("{base}.norm.weight");
        if let Some(norm) = source.tensors.get(&norm_name) {
            global_tensors.insert("norm.weight".into(), norm.clone());
            consumed.insert(norm_name);
        }

        let mixer_prefix = format!("{base}.hyper_connection_mixer.");
        for (name, tensor) in &source.tensors {
            if let Some(rest) = name.strip_prefix(&mixer_prefix) {
                global_tensors.insert(format!("hyper_connection_mixer.{rest}"), tensor.clone());
                consumed.insert(name.clone());
            }
        }

        // Qwen4 runtime canonical names intentionally mirror the HF suffixes
        // (`self_attn.*`, `linear_attn.*`, hyper connections and `mlp.*`).
        let mut layer_static_tensors: BTreeMap<u32, BTreeMap<String, TensorRef>> = BTreeMap::new();
        for layer in 0..layers {
            let lp = format!("{base}.layers.{layer}.");
            let mut map = BTreeMap::new();
            for (name, tensor) in &source.tensors {
                let Some(rest) = name.strip_prefix(&lp) else { continue };
                if rest.starts_with("mlp.experts.") || rest == "mlp.experts.gate_up_proj" || rest == "mlp.experts.down_proj" {
                    continue;
                }
                map.insert(rest.to_owned(), tensor.clone());
                consumed.insert(name.clone());
            }
            layer_static_tensors.insert(layer, map);
        }

        // PLE is globally named in HF, but is injected at a specific model
        // layer. HF's ple_layer_ids is 1-based; COLI/runtime layers are 0-based.
        let ple_layer = ple_layer_index(tc, layers)?;
        add_ple_tensors(
            source,
            &base,
            ple_layer,
            &mut layer_static_tensors,
            &mut consumed,
        )?;

        // Keep unknown text tensors for forward compatibility, but deliberately
        // omit vision tensors: logan-qwen4 is currently a text runtime.
        let resident_tensors = source
            .tensors
            .iter()
            .filter(|(name, _)| {
                !consumed.contains(*name)
                    && !name.starts_with("model.visual.")
                    && !name.starts_with("visual.")
            })
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect();

        Ok(SemanticModel {
            // Today this enum variant is the compiler's Qwen fine-grained-MoE
            // family identity and is the gate that enables BF16->MXFP4 expert
            // lowering. Runtime architecture dispatch still uses config.json's
            // qwen4_exp model_type, so no Qwen3 execution path is involved.
            architecture: Architecture::Qwen3_5MoeMoE,
            geometry,
            routed_experts,
            global_tensors,
            layer_static_tensors,
            resident_tensors,
        })
    }
}

fn model_type(config: &Value) -> Option<&str> {
    config
        .get("model_type")
        .and_then(Value::as_str)
        .or_else(|| config.get("text_config")?.get("model_type")?.as_str())
}

fn is_qwen4_model_type(model_type: Option<&str>) -> bool {
    matches!(model_type, Some("qwen4_exp" | "qwen4_exp_text"))
}

fn text_base(source: &SourceInventory) -> Result<String> {
    for base in ["model.language_model", "model"] {
        let prefix = format!("{base}.layers.0.");
        if source.tensors.keys().any(|name| name.starts_with(&prefix)) {
            return Ok(base.to_owned());
        }
    }
    invalid(
        &source.root,
        "could not locate Qwen4 text backbone under model.language_model or model",
    )
}

fn build_experts(
    source: &SourceInventory,
    base: &str,
    geometry: &ModelGeometry,
    consumed: &mut BTreeSet<String>,
) -> Result<BTreeMap<(u32, u32), RoutedExpert>> {
    let mut out = BTreeMap::new();
    let h = geometry.hidden_size;
    let i = geometry.moe_intermediate_size;
    let ecount = geometry.routed_experts_per_layer;

    for layer in 0..geometry.layers {
        let ep = format!("{base}.layers.{layer}.mlp.experts");
        let first_per = [
            format!("{ep}.0.gate_up_proj"),
            format!("{ep}.0.gate_up_proj.weight"),
            format!("{ep}.0.gate_proj.weight"),
        ]
        .into_iter()
        .any(|name| source.tensors.contains_key(&name));

        if first_per {
            for expert in 0..ecount {
                let p = format!("{ep}.{expert}");
                let separate_gate = first_existing_name(
                    source,
                    &[format!("{p}.gate_proj.weight"), format!("{p}.gate_proj")],
                );
                let separate_up = first_existing_name(
                    source,
                    &[format!("{p}.up_proj.weight"), format!("{p}.up_proj")],
                );
                let down_name = first_existing_name(
                    source,
                    &[format!("{p}.down_proj"), format!("{p}.down_proj.weight")],
                )
                .ok_or_else(|| missing(source, format!("expert {layer}/{expert} down_proj")))?;

                let (gate, up) = if let (Some(gate_name), Some(up_name)) = (separate_gate, separate_up) {
                    let gate = whole_matrix(source, &gate_name, i, h)?;
                    let up = whole_matrix(source, &up_name, i, h)?;
                    consumed.insert(gate_name);
                    consumed.insert(up_name);
                    (gate, up)
                } else {
                    let gu_name = first_existing_name(
                        source,
                        &[format!("{p}.gate_up_proj"), format!("{p}.gate_up_proj.weight")],
                    )
                    .ok_or_else(|| missing(source, format!("expert {layer}/{expert} gate_up_proj")))?;
                    let gu = source.tensors.get(&gu_name).unwrap();
                    validate_bf16_shape(source, &gu_name, gu, &[2 * i as u64, h as u64])?;
                    let gate = slice_rows(gu, 0, i, h)?;
                    let up = slice_rows(gu, i, i, h)?;
                    consumed.insert(gu_name);
                    (gate, up)
                };
                let down = whole_matrix(source, &down_name, h, i)?;
                consumed.insert(down_name);
                out.insert(
                    (layer, expert),
                    RoutedExpert { layer, expert, gate, up, down },
                );
            }
            continue;
        }

        // Also accept the older fused-per-layer layout. This makes conversion
        // robust to checkpoint exporters without changing Apple8 lowering.
        let gu_name = first_existing_name(
            source,
            &[format!("{ep}.gate_up_proj"), format!("{ep}.gate_up_proj.weight")],
        )
        .ok_or_else(|| missing(source, format!("layer {layer} fused gate_up_proj")))?;
        let down_name = first_existing_name(
            source,
            &[format!("{ep}.down_proj"), format!("{ep}.down_proj.weight")],
        )
        .ok_or_else(|| missing(source, format!("layer {layer} fused down_proj")))?;
        let gu = source.tensors.get(&gu_name).unwrap();
        let down = source.tensors.get(&down_name).unwrap();
        validate_bf16_shape(source, &gu_name, gu, &[ecount as u64, 2 * i as u64, h as u64])?;
        validate_bf16_shape(source, &down_name, down, &[ecount as u64, h as u64, i as u64])?;
        for expert in 0..ecount {
            out.insert(
                (layer, expert),
                RoutedExpert {
                    layer,
                    expert,
                    gate: slice_fused(gu, expert, 2 * i, h, 0, i)?,
                    up: slice_fused(gu, expert, 2 * i, h, i, i)?,
                    down: slice_fused(down, expert, h, i, 0, h)?,
                },
            );
        }
        consumed.insert(gu_name);
        consumed.insert(down_name);
    }
    Ok(out)
}

fn add_ple_tensors(
    source: &SourceInventory,
    base: &str,
    ple_layer: u32,
    layers: &mut BTreeMap<u32, BTreeMap<String, TensorRef>>,
    consumed: &mut BTreeSet<String>,
) -> Result<()> {
    let prefix = format!("{base}.ple.");
    let ple = layers.entry(ple_layer).or_default();
    let single_name = format!("{prefix}ple_embedding.ngram_embedding.weight");
    let source_shard_prefix = format!("{prefix}ple_embedding.ngram_embedding.shard_");

    // First copy all small PLE tensors and metadata verbatim. The large ngram
    // tensor is handled separately below.
    for (name, tensor) in &source.tensors {
        let Some(rest) = name.strip_prefix(&prefix) else { continue };
        if name == &single_name || name.starts_with(&source_shard_prefix) {
            continue;
        }
        // Native BF16 checkpoints normally have no ngram scale. If an exporter
        // supplied one, preserve it unless we generate our own BF16->F8 scale.
        if rest == "ple_embedding.ngram_embedding.weight_scale" {
            continue;
        }
        ple.insert(format!("ple.{rest}"), tensor.clone());
        consumed.insert(name.clone());
    }

    if let Some(table) = source.tensors.get(&single_name) {
        add_single_bf16_ngram(source, table, ple)?;
        consumed.insert(single_name.clone());
        add_generated_ple_scale(table, ple);
        return Ok(());
    }

    let mut shards: Vec<(u64, String, TensorRef)> = source
        .tensors
        .iter()
        .filter_map(|(name, tensor)| {
            let suffix = name.strip_prefix(&source_shard_prefix)?;
            let index = suffix
                .trim_end_matches(".weight")
                .parse::<u64>()
                .ok()?;
            Some((index, name.clone(), tensor.clone()))
        })
        .collect();
    shards.sort_by_key(|(index, _, _)| *index);
    if shards.is_empty() {
        return invalid(&source.root, "missing PLE ngram embedding weight/shards");
    }
    for (ordinal, (_source_index, name, tensor)) in shards.into_iter().enumerate() {
        let out = f8_view_of_bf16(source, &name, &tensor)?;
        ple.insert(
            format!("ple.ple_embedding.ngram_embedding.shard_{ordinal:03}"),
            out,
        );
        consumed.insert(name);
    }
    let first = ple
        .values()
        .find(|t| t.dtype == PLE_F8_FROM_BF16_S16)
        .cloned()
        .ok_or_else(|| missing(source, "BF16 PLE ngram shard"))?;
    add_generated_ple_scale(&first, ple);
    Ok(())
}

fn add_single_bf16_ngram(
    source: &SourceInventory,
    table: &TensorRef,
    ple: &mut BTreeMap<String, TensorRef>,
) -> Result<()> {
    if table.dtype != "BF16" || table.shape.len() != 2 {
        return invalid(
            &source.root,
            format!("PLE ngram table must be rank-2 BF16, got {}/ {:?}", table.dtype, table.shape),
        );
    }
    let rows = table.shape[0];
    let cols = table.shape[1];
    let rows_per_shard = rows.div_ceil(PLE_SHARDS).max(1);
    let source_row_bytes = cols.checked_mul(2).ok_or_else(|| missing(source, "PLE row size overflow"))?;
    for shard in 0..PLE_SHARDS {
        let first_row = shard * rows_per_shard;
        if first_row >= rows { break; }
        let shard_rows = (rows - first_row).min(rows_per_shard);
        let offset = table
            .offset
            .checked_add(first_row.checked_mul(source_row_bytes).ok_or_else(|| missing(source, "PLE shard offset overflow"))?)
            .ok_or_else(|| missing(source, "PLE shard offset overflow"))?;
        let len = shard_rows.checked_mul(cols).ok_or_else(|| missing(source, "PLE shard size overflow"))?;
        ple.insert(
            format!("ple.ple_embedding.ngram_embedding.shard_{shard:03}"),
            TensorRef {
                source: table.source.clone(),
                offset,
                len,
                dtype: PLE_F8_FROM_BF16_S16.into(),
                shape: vec![shard_rows, cols],
            },
        );
    }
    Ok(())
}

fn f8_view_of_bf16(source: &SourceInventory, name: &str, tensor: &TensorRef) -> Result<TensorRef> {
    if tensor.dtype != "BF16" || tensor.shape.len() != 2 {
        return invalid(
            &source.root,
            format!("PLE shard `{name}` must be rank-2 BF16, got {}/{:?}", tensor.dtype, tensor.shape),
        );
    }
    if tensor.len % 2 != 0 {
        return invalid(&source.root, format!("PLE shard `{name}` has odd BF16 byte length"));
    }
    let mut out = tensor.clone();
    out.len /= 2;
    out.dtype = PLE_F8_FROM_BF16_S16.into();
    Ok(out)
}

fn add_generated_ple_scale(source_tensor: &TensorRef, ple: &mut BTreeMap<String, TensorRef>) {
    ple.insert(
        "ple.ple_embedding.ngram_embedding.weight_scale".into(),
        TensorRef {
            source: source_tensor.source.clone(),
            offset: source_tensor.offset,
            len: 2,
            dtype: PLE_SCALE_CONST_S16.into(),
            shape: vec![1],
        },
    );
}

fn ple_layer_index(config: &Value, layers: u32) -> Result<u32> {
    let id = config
        .get("ple_layer_ids")
        .and_then(Value::as_array)
        .and_then(|ids| ids.first())
        .and_then(Value::as_u64)
        .unwrap_or(1);
    let zero = id.checked_sub(1).ok_or_else(|| ColicError::InvalidSource {
        path: Path::new("config.json").to_owned(),
        detail: "ple_layer_ids must be 1-based and non-zero".into(),
    })?;
    let zero: u32 = zero.try_into().map_err(|_| ColicError::InvalidSource {
        path: Path::new("config.json").to_owned(),
        detail: "PLE layer index exceeds u32".into(),
    })?;
    if zero >= layers {
        return Err(ColicError::InvalidSource {
            path: Path::new("config.json").to_owned(),
            detail: format!("PLE layer {zero} is outside {layers} layers"),
        });
    }
    Ok(zero)
}

fn required_layer_types(root: &Path, config: &Value, layers: u32) -> Result<Vec<String>> {
    let array = config
        .get("layer_types")
        .and_then(Value::as_array)
        .ok_or_else(|| ColicError::InvalidSource {
            path: root.to_owned(),
            detail: "Qwen4 config is missing layer_types".into(),
        })?;
    if array.len() != layers as usize {
        return invalid(root, format!("layer_types has {} entries, expected {layers}", array.len()));
    }
    array
        .iter()
        .enumerate()
        .map(|(index, value)| match value.as_str() {
            Some("linear_attention" | "full_attention") => Ok(value.as_str().unwrap().to_owned()),
            other => invalid(root, format!("unsupported layer type at {index}: {other:?}")),
        })
        .collect()
}

fn required_bf16(source: &SourceInventory, name: &str, shape: &[u64]) -> Result<TensorRef> {
    let tensor = source.tensors.get(name).ok_or_else(|| missing(source, name))?;
    validate_bf16_shape(source, name, tensor, shape)?;
    Ok(tensor.clone())
}

fn validate_bf16_shape(
    source: &SourceInventory,
    name: &str,
    tensor: &TensorRef,
    shape: &[u64],
) -> Result<()> {
    if tensor.dtype != "BF16" || tensor.shape != shape {
        return invalid(
            &source.root,
            format!("tensor `{name}` has {}/{:?}, expected BF16/{shape:?}", tensor.dtype, tensor.shape),
        );
    }
    let expected = shape.iter().try_fold(2_u64, |bytes, dim| bytes.checked_mul(*dim))
        .ok_or_else(|| missing(source, format!("tensor `{name}` byte size overflow")))?;
    if tensor.len != expected {
        return invalid(&source.root, format!("tensor `{name}` len {} != {expected}", tensor.len));
    }
    Ok(())
}

fn whole_matrix(source: &SourceInventory, name: &str, rows: u32, cols: u32) -> Result<Matrix> {
    let tensor = required_bf16(source, name, &[rows as u64, cols as u64])?;
    Ok(Matrix { source: tensor, rows, columns: cols, scale: None })
}

fn slice_rows(tensor: &TensorRef, row_start: u32, rows: u32, cols: u32) -> Result<Matrix> {
    let row_bytes = u64::from(cols) * 2;
    let offset = tensor.offset
        .checked_add(u64::from(row_start) * row_bytes)
        .ok_or_else(|| ColicError::InvalidSource { path: tensor.source.clone(), detail: "expert row offset overflow".into() })?;
    Ok(Matrix {
        source: TensorRef {
            source: tensor.source.clone(),
            offset,
            len: u64::from(rows) * row_bytes,
            dtype: "BF16".into(),
            shape: vec![rows as u64, cols as u64],
        },
        rows,
        columns: cols,
        scale: None,
    })
}

fn slice_fused(
    tensor: &TensorRef,
    expert: u32,
    rows_per_expert: u32,
    cols: u32,
    row_start: u32,
    rows: u32,
) -> Result<Matrix> {
    let row_bytes = u64::from(cols) * 2;
    let expert_rows = u64::from(expert) * u64::from(rows_per_expert);
    let offset_rows = expert_rows + u64::from(row_start);
    let offset = tensor.offset
        .checked_add(offset_rows.checked_mul(row_bytes).ok_or_else(|| ColicError::InvalidSource {
            path: tensor.source.clone(), detail: "fused expert offset overflow".into()
        })?)
        .ok_or_else(|| ColicError::InvalidSource { path: tensor.source.clone(), detail: "fused expert offset overflow".into() })?;
    Ok(Matrix {
        source: TensorRef {
            source: tensor.source.clone(),
            offset,
            len: u64::from(rows) * row_bytes,
            dtype: "BF16".into(),
            shape: vec![rows as u64, cols as u64],
        },
        rows,
        columns: cols,
        scale: None,
    })
}

fn first_existing_name(source: &SourceInventory, names: &[String]) -> Option<String> {
    names.iter().find(|name| source.tensors.contains_key(*name)).cloned()
}

fn required_u32(root: &Path, config: &Value, field: &str) -> Result<u32> {
    optional_u32(config, field)
        .filter(|v| *v > 0)
        .ok_or_else(|| ColicError::InvalidSource {
            path: root.to_owned(),
            detail: format!("config `{field}` must be a positive u32"),
        })
}

fn optional_u32(config: &Value, field: &str) -> Option<u32> {
    config.get(field)?.as_u64()?.try_into().ok()
}

fn missing(source: &SourceInventory, detail: impl Into<String>) -> ColicError {
    ColicError::InvalidSource { path: source.root.clone(), detail: format!("missing required {0}", detail.into()) }
}

fn invalid<T>(path: &Path, detail: impl Into<String>) -> Result<T> {
    Err(ColicError::InvalidSource { path: path.to_owned(), detail: detail.into() })
}
