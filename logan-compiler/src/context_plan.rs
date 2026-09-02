use std::{collections::BTreeSet, fs, path::Path};

use logan_ir::{
    ContextCandidate, ContextConstraint, ContextConstraintKind, ContextPlan, ContextStateBytes,
    PlannerMemoryBudget,
};
use serde_json::Value;

use crate::{
    error::{ColicError, Result},
    target::MachineProfile,
};

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;
const F32_BYTES: u64 = 4;
const I64_BYTES: u64 = 8;
const MIN_CONTEXT_POINT: u64 = 8 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextGeometry {
    pub model_max_tokens: u64,
    pub hidden_size: u64,
    pub full_attention_layers: u64,
    pub gdn_layers: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub linear_key_heads: u64,
    pub linear_key_head_dim: u64,
    pub linear_value_heads: u64,
    pub linear_value_head_dim: u64,
    pub linear_conv_kernel: u64,
    pub qsa_layers: u64,
    pub qsa_kv_heads: u64,
    pub qsa_head_dim: u64,
    pub ple_enabled: bool,
    pub ple_embed_dim: u64,
    pub ple_conv_kernel: u64,
    pub ngram_size: u64,
    pub ngram_heads: u64,
    pub mtp_layers: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedContextPoint {
    pub candidate: ContextCandidate,
    pub plan: ContextPlan,
    pub budget: PlannerMemoryBudget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextPlanning {
    pub geometry: ContextGeometry,
    pub points: Vec<PlannedContextPoint>,
}

impl ContextPlanning {
    pub fn optimizer_candidates(&self) -> Vec<ContextCandidate> {
        self.points.iter().map(|point| point.candidate).collect()
    }

    pub fn point_for_tokens(&self, tokens: u64) -> Option<&PlannedContextPoint> {
        self.points.iter().find(|point| point.candidate.tokens == tokens)
    }
}

pub fn plan_from_package(
    package_root: &Path,
    constraint: ContextConstraint,
    machine: &MachineProfile,
    fixed_model_state: u64,
) -> Result<ContextPlanning> {
    let geometry = geometry_from_config(&package_root.join("config.json"), constraint.tokens)?;
    plan_geometry(geometry, constraint, machine, fixed_model_state)
}

pub fn plan_geometry(
    geometry: ContextGeometry,
    constraint: ContextConstraint,
    machine: &MachineProfile,
    fixed_model_state: u64,
) -> Result<ContextPlanning> {
    constraint
        .valid_for_model(geometry.model_max_tokens)
        .map_err(ColicError::Usage)?;
    let physical_memory = machine
        .ram_bytes
        .unwrap_or(crate::target::machine::DEFAULT_POOL_BUDGET);
    let (os_reserve, runtime_reserve, safety_reserve, execution_scratch) =
        reserve_policy(physical_memory);

    let mut points = Vec::new();
    for tokens in context_points(constraint, geometry.model_max_tokens) {
        let state_bytes = state_bytes(&geometry, tokens)?;
        let budget = PlannerMemoryBudget {
            physical_memory,
            os_reserve,
            runtime_reserve,
            safety_reserve,
            fixed_model_state,
            context_state: state_bytes,
            execution_scratch,
        };
        let overhead = os_reserve
            .checked_add(runtime_reserve)
            .and_then(|value| value.checked_add(safety_reserve))
            .and_then(|value| value.checked_add(execution_scratch))
            .and_then(|value| value.checked_add(state_bytes.total_bytes()))
            .ok_or_else(|| ColicError::Usage("context memory overhead overflows u64".into()))?;
        points.push(PlannedContextPoint {
            candidate: ContextCandidate {
                tokens,
                resident_bytes: overhead,
                // v1 has no measured context-dependent latency model. Keep
                // latency neutral rather than inventing a performance score.
                latency_cost: 0,
            },
            plan: ContextPlan {
                constraint,
                model_max_tokens: geometry.model_max_tokens,
                compiled_max_tokens: tokens,
                state_bytes,
            },
            budget,
        });
    }
    if points.is_empty() {
        return Err(ColicError::Usage(
            "no context points satisfy the requested context constraint".into(),
        ));
    }
    Ok(ContextPlanning { geometry, points })
}

pub fn geometry_from_config(path: &Path, requested_tokens: u64) -> Result<ContextGeometry> {
    let bytes = fs::read(path).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    let root: Value = serde_json::from_slice(&bytes).map_err(|error| ColicError::InvalidSource {
        path: path.to_owned(),
        detail: format!("invalid config.json: {error}"),
    })?;
    let config = root.get("text_config").unwrap_or(&root);
    let get = |key: &str| config.get(key).and_then(Value::as_u64).unwrap_or(0);

    let layers = get("num_hidden_layers");
    let hidden_size = get("hidden_size");
    if layers == 0 || hidden_size == 0 {
        return invalid(path, "context planning requires num_hidden_layers and hidden_size");
    }
    let model_max_tokens = get("max_position_embeddings");
    let model_max_tokens = if model_max_tokens == 0 {
        // Some older compiler fixtures do not advertise a ceiling. Treat the
        // requested point as the only proven ceiling; real optimized packages
        // with a declared ceiling are validated strictly below.
        requested_tokens
    } else {
        model_max_tokens
    };

    let layer_types = config.get("layer_types").and_then(Value::as_array);
    let (full_attention_layers, gdn_layers) = match layer_types {
        Some(types) => {
            if types.len() as u64 != layers {
                return invalid(
                    path,
                    format!("layer_types has {} entries but num_hidden_layers is {layers}", types.len()),
                );
            }
            let mut full = 0_u64;
            let mut gdn = 0_u64;
            for layer_type in types {
                match layer_type.as_str() {
                    Some("full_attention") => full += 1,
                    Some("linear_attention") => gdn += 1,
                    Some(other) => {
                        return invalid(path, format!("unsupported context layer type `{other}`"));
                    }
                    None => return invalid(path, "layer_types entries must be strings"),
                }
            }
            (full, gdn)
        }
        None => (layers, 0),
    };

    let heads = get("num_attention_heads");
    let head_dim = get("head_dim").max(if heads == 0 { 0 } else { hidden_size / heads });
    let kv_heads = get("num_key_value_heads").max(heads);
    if full_attention_layers != 0 && (head_dim == 0 || kv_heads == 0) {
        return invalid(path, "full-attention context planning requires KV-head geometry");
    }

    let linear_key_heads = get("linear_num_key_heads");
    let linear_key_head_dim = get("linear_key_head_dim");
    let linear_value_heads = get("linear_num_value_heads");
    let linear_value_head_dim = get("linear_value_head_dim");
    let linear_conv_kernel = get("linear_conv_kernel_dim");
    if gdn_layers != 0
        && [
            linear_key_heads,
            linear_key_head_dim,
            linear_value_heads,
            linear_value_head_dim,
            linear_conv_kernel,
        ]
        .contains(&0)
    {
        return invalid(path, "linear-attention context planning requires complete GDN geometry");
    }

    let idx_n_heads = get("indexer_n_heads");
    let qsa_layers = if idx_n_heads == 0 {
        0
    } else {
        full_attention_layers
    };
    let qsa_kv_heads = get("indexer_kv_heads");
    let qsa_head_dim = get("indexer_head_dim");
    if qsa_layers != 0 && (qsa_kv_heads == 0 || qsa_head_dim == 0) {
        return invalid(path, "QSA context planning requires indexer_kv_heads and indexer_head_dim");
    }

    let ple_enabled = config
        .get("ple_layer_ids")
        .and_then(Value::as_array)
        .is_some_and(|ids| !ids.is_empty());
    let ple_embed_dim = get("ple_embed_dim");
    let ple_conv_kernel = get("ple_conv_kernel_size");
    let ngram_size = get("ngram_size");
    let heads_per_ngram = get("heads_per_ngram");
    let ngram_heads = if ngram_size > 1 {
        heads_per_ngram.saturating_mul(ngram_size - 1)
    } else {
        0
    };
    if ple_enabled
        && (ple_embed_dim == 0
            || ple_conv_kernel == 0
            || ngram_size == 0
            || ngram_heads == 0
            || ple_embed_dim % ngram_heads != 0)
    {
        return invalid(path, "PLE context planning requires valid embed/conv/ngram geometry");
    }

    Ok(ContextGeometry {
        model_max_tokens,
        hidden_size,
        full_attention_layers,
        gdn_layers,
        kv_heads,
        head_dim,
        linear_key_heads,
        linear_key_head_dim,
        linear_value_heads,
        linear_value_head_dim,
        linear_conv_kernel,
        qsa_layers,
        qsa_kv_heads,
        qsa_head_dim,
        ple_enabled,
        ple_embed_dim,
        ple_conv_kernel,
        ngram_size,
        ngram_heads,
        mtp_layers: get("mtp_num_hidden_layers"),
    })
}

fn state_bytes(geometry: &ContextGeometry, tokens: u64) -> Result<ContextStateBytes> {
    let full_attention_kv = checked_product(&[
        geometry.full_attention_layers,
        2,
        geometry.kv_heads,
        tokens,
        geometry.head_dim,
        F32_BYTES,
    ], "full-attention KV")?;

    let gdn_recurrent = checked_product(&[
        geometry.gdn_layers,
        geometry.linear_value_heads,
        geometry.linear_key_head_dim,
        geometry.linear_value_head_dim,
        F32_BYTES,
    ], "GDN recurrent state")?;
    let cdim = geometry
        .linear_key_heads
        .checked_mul(geometry.linear_key_head_dim)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| {
            geometry
                .linear_value_heads
                .checked_mul(geometry.linear_value_head_dim)
                .and_then(|v| value.checked_add(v))
        })
        .ok_or_else(|| ColicError::Usage("GDN channel geometry overflows u64".into()))?;
    let gdn_conv = checked_product(&[
        geometry.gdn_layers,
        cdim,
        geometry.linear_conv_kernel.saturating_sub(1),
        F32_BYTES,
    ], "GDN convolution state")?;
    let qsa_index = checked_product(&[
        geometry.qsa_layers,
        tokens,
        geometry.qsa_kv_heads,
        geometry.qsa_head_dim,
        F32_BYTES,
    ], "QSA index state")?;

    let ple = if geometry.ple_enabled {
        let hcd = geometry.ple_embed_dim / geometry.ngram_heads;
        let history = geometry
            .ple_conv_kernel
            .saturating_sub(1)
            .checked_mul(geometry.ngram_size)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| ColicError::Usage("PLE history geometry overflows u64".into()))?;
        checked_product(&[geometry.ngram_size.max(1), I64_BYTES], "PLE token ring")?
            .checked_add(checked_product(&[hcd, history.max(1), F32_BYTES], "PLE conv state")?)
            .ok_or_else(|| ColicError::Usage("PLE state bytes overflow u64".into()))?
    } else {
        0
    };

    Ok(ContextStateBytes {
        full_attention_kv,
        gdn_recurrent,
        gdn_conv,
        qsa_index,
        ple,
        // The current Qwen4 runner does not execute MTP speculative stages,
        // so there is no runtime MTP state to reserve yet. The field remains
        // explicit so enabling MTP must add a bounded state contract here.
        mtp_speculative: 0,
        other: 0,
    })
}

fn context_points(constraint: ContextConstraint, model_max: u64) -> Vec<u64> {
    let mut points = BTreeSet::new();
    match constraint.kind {
        ContextConstraintKind::Maximum => {
            let ceiling = constraint.tokens.min(model_max);
            points.insert(ceiling);
            let mut point = ceiling;
            while point > MIN_CONTEXT_POINT {
                point /= 2;
                if point >= MIN_CONTEXT_POINT {
                    points.insert(point);
                }
            }
        }
        ContextConstraintKind::Required => {
            let mut point = constraint.tokens;
            points.insert(point);
            while point < model_max {
                let next = point.saturating_mul(2).min(model_max);
                if next == point {
                    break;
                }
                points.insert(next);
                point = next;
            }
        }
    }
    points
        .into_iter()
        .filter(|tokens| constraint.allows(*tokens) && *tokens <= model_max)
        .collect()
}

fn reserve_policy(physical_memory: u64) -> (u64, u64, u64, u64) {
    // Explicit deterministic desktop/UMA policy. The runtime reserve covers
    // allocator/framework overhead that has no bounded arena contract yet;
    // issue #2 can replace that portion with exact scratch accounting.
    let os_reserve = (physical_memory / 8).clamp(GIB, 4 * GIB);
    let runtime_reserve = 256 * MIB;
    let safety_reserve = (physical_memory / 16).clamp(512 * MIB, 2 * GIB);
    let execution_scratch = 256 * MIB;
    (os_reserve, runtime_reserve, safety_reserve, execution_scratch)
}

fn checked_product(values: &[u64], what: &str) -> Result<u64> {
    values.iter().copied().try_fold(1_u64, |acc, value| {
        acc.checked_mul(value)
            .ok_or_else(|| ColicError::Usage(format!("{what} bytes overflow u64")))
    })
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

    fn hybrid() -> ContextGeometry {
        ContextGeometry {
            model_max_tokens: 262_144,
            hidden_size: 2560,
            full_attention_layers: 12,
            gdn_layers: 36,
            kv_heads: 8,
            head_dim: 256,
            linear_key_heads: 16,
            linear_key_head_dim: 128,
            linear_value_heads: 48,
            linear_value_head_dim: 128,
            linear_conv_kernel: 4,
            qsa_layers: 12,
            qsa_kv_heads: 8,
            qsa_head_dim: 128,
            ple_enabled: true,
            ple_embed_dim: 1024,
            ple_conv_kernel: 4,
            ngram_size: 3,
            ngram_heads: 16,
            mtp_layers: 0,
        }
    }

    fn machine(bytes: u64) -> MachineProfile {
        let mut machine = MachineProfile::probe();
        machine.ram_bytes = Some(bytes);
        machine
    }

    #[test]
    fn hybrid_kv_only_counts_full_attention_layers() {
        let state = state_bytes(&hybrid(), 131_072).unwrap();
        assert_eq!(
            state.full_attention_kv,
            12 * 2 * 8 * 131_072 * 256 * 4
        );
        assert!(state.gdn_recurrent > 0);
        assert!(state.gdn_conv > 0);
        assert!(state.qsa_index > 0);
        assert!(state.ple > 0);
    }

    #[test]
    fn all_attention_uses_more_context_memory_than_hybrid() {
        let hybrid_state = state_bytes(&hybrid(), 65_536).unwrap();
        let mut full = hybrid();
        full.full_attention_layers = 48;
        full.gdn_layers = 0;
        full.qsa_layers = 0;
        let full_state = state_bytes(&full, 65_536).unwrap();
        assert!(full_state.full_attention_kv > hybrid_state.full_attention_kv);
        assert_eq!(full_state.gdn_recurrent, 0);
        assert_eq!(full_state.gdn_conv, 0);
    }

    #[test]
    fn max_context_generates_lower_tradeoff_points() {
        let planning = plan_geometry(
            hybrid(),
            ContextConstraint::maximum(131_072),
            &machine(16 * GIB),
            2 * GIB,
        )
        .unwrap();
        let tokens = planning
            .points
            .iter()
            .map(|point| point.candidate.tokens)
            .collect::<Vec<_>>();
        assert_eq!(tokens, vec![8192, 16384, 32768, 65536, 131072]);
    }

    #[test]
    fn required_context_never_offers_smaller_points() {
        let planning = plan_geometry(
            hybrid(),
            ContextConstraint::required(131_072),
            &machine(64 * GIB),
            2 * GIB,
        )
        .unwrap();
        let tokens = planning
            .points
            .iter()
            .map(|point| point.candidate.tokens)
            .collect::<Vec<_>>();
        assert_eq!(tokens, vec![131_072, 262_144]);
    }

    #[test]
    fn requested_context_above_model_ceiling_is_rejected() {
        let error = plan_geometry(
            hybrid(),
            ContextConstraint::required(262_145),
            &machine(64 * GIB),
            0,
        )
        .unwrap_err();
        assert!(error.to_string().contains("exceeds model ceiling"));
    }
}
