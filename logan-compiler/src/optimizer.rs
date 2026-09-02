use std::collections::{BTreeMap, BTreeSet};

use logan_ir::{
    ContextCandidate, ContextConstraint, DecisionGroup, OptimizeInput, ParetoPlan, PhysicalOption,
    Placement, optimize, select_plan,
};

use crate::{
    error::{ColicError, Result},
    ir::{Architecture, RoutedExpert, SemanticModel},
    quant::mxfp4_record,
    target::{self, MachineProfile, TargetProfile},
};

pub const COST_MODEL_VERSION: &str = "logan-expert-cost-v1";
const MIB: u64 = 1024 * 1024;
const EXECUTION_SCRATCH: u64 = 512 * MIB;
const RUNTIME_RESERVE: u64 = 256 * MIB;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertRepresentation {
    Exact,
    Mxfp4,
}

#[derive(Debug, Clone)]
pub(crate) struct SelectedOptimization {
    pub frontier: Vec<ParetoPlan>,
    pub selected: ParetoPlan,
    by_expert: BTreeMap<(u32, u32), ExpertRepresentation>,
}

impl SelectedOptimization {
    pub fn representation_for(&self, layer: u32, expert: u32) -> ExpertRepresentation {
        self.by_expert
            .get(&(layer, expert))
            .copied()
            .unwrap_or(ExpertRepresentation::Exact)
    }
}

fn exact_option_supported(expert: &RoutedExpert, target_profile: TargetProfile) -> bool {
    if target_profile == target::MACOS_ARM64_METAL_APPLE8_V1 {
        target::validate_apple8_exact_mxfp4_expert(expert).is_ok()
    } else {
        target::exact_expert_stored_bytes(expert).is_ok()
    }
}

fn mxfp4_option_supported(
    model: &SemanticModel,
    expert: &RoutedExpert,
    target_profile: TargetProfile,
) -> bool {
    if model.architecture != Architecture::Qwen3_5MoeMoE {
        return false;
    }
    if target_profile == target::MACOS_ARM64_METAL_APPLE8_V1 {
        target::validate_apple8_quantized_mxfp4_expert(expert).is_ok()
    } else {
        mxfp4_record::stored_bytes(expert).is_ok()
    }
}

fn stored_bytes(
    expert: &RoutedExpert,
    representation: ExpertRepresentation,
    target_profile: TargetProfile,
) -> Result<u64> {
    if target_profile == target::MACOS_ARM64_METAL_APPLE8_V1 {
        return target::apple8_expert_stored_bytes(expert);
    }
    match representation {
        ExpertRepresentation::Exact => target::exact_expert_stored_bytes(expert),
        ExpertRepresentation::Mxfp4 => mxfp4_record::stored_bytes(expert),
    }
}

fn decoded_bytes(
    expert: &RoutedExpert,
    representation: ExpertRepresentation,
    target_profile: TargetProfile,
) -> Result<u64> {
    if target_profile == target::MACOS_ARM64_METAL_APPLE8_V1 {
        return target::apple8_expert_decoded_bytes(expert);
    }
    match representation {
        ExpertRepresentation::Exact => target::exact_expert_decoded_bytes(expert),
        ExpertRepresentation::Mxfp4 => mxfp4_record::resident_bytes(expert),
    }
}

fn layer_sensitivity(layer: u32, layers: u32) -> u64 {
    let from_start = u64::from(layer) + 1;
    let from_end = u64::from(layers.saturating_sub(layer));
    let edge_distance = from_start.min(from_end).max(1);
    1_000 + 8_000 / edge_distance
}

fn expert_cache_slots() -> u64 {
    // Keep the optimizer deterministic and independent of hidden environment
    // knobs. Runtime cache tuning can still happen later, but compile-time
    // admissibility reserves the measured/default plateau.
    256
}

fn context_candidates(model: &SemanticModel, constraint: ContextConstraint) -> Vec<ContextCandidate> {
    let mut tokens = BTreeSet::new();
    match constraint.kind {
        logan_ir::ContextConstraintKind::Maximum => {
            tokens.insert((constraint.tokens / 4).max(1));
            tokens.insert((constraint.tokens / 2).max(1));
            tokens.insert(constraint.tokens);
        }
        logan_ir::ContextConstraintKind::Required => {
            tokens.insert(constraint.tokens);
        }
    }
    tokens
        .into_iter()
        .filter_map(|tokens| {
            estimate_context_bytes(model, tokens)
                .ok()
                .map(|state_bytes| ContextCandidate { tokens, state_bytes })
        })
        .collect()
}

fn estimate_context_bytes(model: &SemanticModel, tokens: u64) -> Result<u64> {
    // Conservative v1: treat every layer as full-attention KV at BF16. This
    // deliberately over-reserves hybrid GDN models until the semantic IR
    // carries exact per-layer attention/recurrent kinds. It cannot produce an
    // unsafe optimistic plan.
    let g = &model.geometry;
    let kv_heads = if g.num_key_value_heads > 0 {
        g.num_key_value_heads
    } else {
        g.attention_heads.max(1)
    };
    let head_dim = if g.head_dim > 0 {
        g.head_dim
    } else {
        g.linear_key_head_dim.max(1)
    };
    tokens
        .checked_mul(u64::from(g.layers.max(1)))
        .and_then(|value| value.checked_mul(u64::from(kv_heads)))
        .and_then(|value| value.checked_mul(u64::from(head_dim)))
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| ColicError::Usage("context-state estimate overflows u64".into()))
}

fn dense_resident_bytes(model: &SemanticModel) -> Result<u64> {
    model
        .global_tensors
        .values()
        .chain(model.layer_static_tensors.values().flat_map(|layer| layer.values()))
        .chain(model.resident_tensors.values())
        .try_fold(0_u64, |sum, tensor| {
            sum.checked_add(tensor.len)
                .ok_or_else(|| ColicError::Usage("dense resident bytes overflow u64".into()))
        })
}

fn machine_base_reserve(
    model: &SemanticModel,
    target_profile: TargetProfile,
    machine: &MachineProfile,
) -> Result<(u64, u64)> {
    let physical = machine
        .ram_bytes
        .unwrap_or(target::machine::DEFAULT_POOL_BUDGET);
    let dense = dense_resident_bytes(model)?;
    let mut worst_expert = 0_u64;
    for expert in model.routed_experts.values() {
        if exact_option_supported(expert, target_profile) {
            worst_expert = worst_expert.max(decoded_bytes(
                expert,
                ExpertRepresentation::Exact,
                target_profile,
            )?);
        }
        if mxfp4_option_supported(model, expert, target_profile) {
            worst_expert = worst_expert.max(decoded_bytes(
                expert,
                ExpertRepresentation::Mxfp4,
                target_profile,
            )?);
        }
    }
    let cache = expert_cache_slots()
        .checked_mul(worst_expert)
        .ok_or_else(|| ColicError::Usage("expert cache reservation overflows u64".into()))?;
    let os_reserve = physical / 8;
    let safety_reserve = physical / 10;
    let base = dense
        .checked_add(cache)
        .and_then(|value| value.checked_add(os_reserve))
        .and_then(|value| value.checked_add(safety_reserve))
        .and_then(|value| value.checked_add(RUNTIME_RESERVE))
        .and_then(|value| value.checked_add(EXECUTION_SCRATCH))
        .ok_or_else(|| ColicError::Usage("optimizer base memory reservation overflows u64".into()))?;
    Ok((base, physical))
}

fn layer_bands(model: &SemanticModel) -> Vec<Vec<u32>> {
    let layers = model
        .routed_experts
        .keys()
        .map(|(layer, _)| *layer)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if layers.is_empty() {
        return Vec::new();
    }
    let band_size = layers.len().div_ceil(8).max(1);
    layers
        .chunks(band_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn option_for_band(
    model: &SemanticModel,
    layers: &[u32],
    representation: ExpertRepresentation,
    target_profile: TargetProfile,
) -> Result<Option<PhysicalOption>> {
    let experts = model
        .routed_experts
        .values()
        .filter(|expert| layers.binary_search(&expert.layer).is_ok())
        .collect::<Vec<_>>();
    if experts.is_empty() {
        return Ok(None);
    }
    let supported = experts.iter().all(|expert| match representation {
        ExpertRepresentation::Exact => exact_option_supported(expert, target_profile),
        ExpertRepresentation::Mxfp4 => mxfp4_option_supported(model, expert, target_profile),
    });
    if !supported {
        return Ok(None);
    }
    let package_bytes = experts.iter().try_fold(0_u64, |sum, expert| {
        sum.checked_add(stored_bytes(expert, representation, target_profile)?)
            .ok_or_else(|| ColicError::Usage("optimizer package bytes overflow u64".into()))
    })?;
    let quality_loss = match representation {
        ExpertRepresentation::Exact => 0,
        ExpertRepresentation::Mxfp4 => layers.iter().try_fold(0_u64, |sum, &layer| {
            sum.checked_add(layer_sensitivity(layer, model.geometry.layers))
                .ok_or_else(|| ColicError::Usage("optimizer quality cost overflows u64".into()))
        })?,
    };
    let throughput_weight = match representation {
        ExpertRepresentation::Exact => 100_u64,
        ExpertRepresentation::Mxfp4 => {
            if target_profile == target::MACOS_ARM64_METAL_APPLE8_V1 { 42 } else { 62 }
        }
    };
    let latency_cost = package_bytes
        .div_ceil(MIB)
        .max(1)
        .checked_mul(throughput_weight)
        .ok_or_else(|| ColicError::Usage("optimizer latency cost overflows u64".into()))?;
    let (representation_name, layout) = match representation {
        ExpertRepresentation::Exact => ("exact", "source"),
        ExpertRepresentation::Mxfp4 if target_profile == target::MACOS_ARM64_METAL_APPLE8_V1 => {
            ("mxfp4", "apple8-tile8x32")
        }
        ExpertRepresentation::Mxfp4 => ("mxfp4", "canonical"),
    };
    Ok(Some(PhysicalOption {
        id: representation_name.into(),
        representation: representation_name.into(),
        layout: layout.into(),
        placement: Placement::Streamed,
        quality_loss,
        latency_cost,
        resident_bytes: 0,
        package_bytes,
    }))
}

pub(crate) fn build_plans(
    model: &SemanticModel,
    target_profile: TargetProfile,
    machine: &MachineProfile,
    constraint: ContextConstraint,
    forced: Option<ExpertRepresentation>,
) -> Result<(Vec<ParetoPlan>, BTreeMap<String, Vec<(u32, u32)>>)> {
    let mut groups = Vec::new();
    let mut members = BTreeMap::new();
    for layers in layer_bands(model) {
        let first = *layers.first().unwrap();
        let last = *layers.last().unwrap();
        let id = if first == last {
            format!("layer:{first}/experts")
        } else {
            format!("layers:{first}-{last}/experts")
        };
        let mut options = Vec::new();
        for representation in [ExpertRepresentation::Exact, ExpertRepresentation::Mxfp4] {
            if forced.is_some_and(|value| value != representation) {
                continue;
            }
            if let Some(option) = option_for_band(model, &layers, representation, target_profile)? {
                options.push(option);
            }
        }
        if options.is_empty() {
            return Err(ColicError::unsupported(
                "target planning",
                format!("no executable representation exists for optimizer group `{id}`"),
            ));
        }
        members.insert(
            id.clone(),
            model
                .routed_experts
                .keys()
                .filter(|(layer, _)| layers.binary_search(layer).is_ok())
                .copied()
                .collect(),
        );
        groups.push(DecisionGroup { id, options });
    }
    let (base_resident_bytes, memory_budget_bytes) = machine_base_reserve(model, target_profile, machine)?;
    let input = OptimizeInput {
        groups,
        contexts: context_candidates(model, constraint),
        context_constraint: constraint,
        base_resident_bytes,
        memory_budget_bytes,
        heterogeneity_penalty: 250,
    };
    let plans = optimize(&input).map_err(|message| ColicError::Usage(format!("optimizer: {message}")))?;
    if plans.is_empty() {
        return Err(ColicError::unsupported(
            "target planning",
            format!(
                "no Pareto plan fits context constraint {:?} {} within {} bytes of physical memory",
                constraint.kind, constraint.tokens, memory_budget_bytes
            ),
        ));
    }
    Ok((plans, members))
}

pub(crate) fn select(
    plans: Vec<ParetoPlan>,
    members: BTreeMap<String, Vec<(u32, u32)>>,
    selector: &str,
) -> Result<SelectedOptimization> {
    let selected = select_plan(&plans, selector).cloned().ok_or_else(|| {
        ColicError::Usage(format!(
            "unknown optimizer plan `{selector}`; available: {}",
            plans
                .iter()
                .map(|plan| format!("{} ({})", plan.label.as_deref().unwrap_or("unlabeled"), plan.id))
                .collect::<Vec<_>>()
                .join(", ")
        ))
    })?;
    let mut by_expert = BTreeMap::new();
    for decision in &selected.decisions {
        let representation = match decision.option_id.as_str() {
            "exact" => ExpertRepresentation::Exact,
            "mxfp4" => ExpertRepresentation::Mxfp4,
            other => {
                return Err(ColicError::Usage(format!(
                    "optimizer selected unknown expert representation `{other}`"
                )))
            }
        };
        for &(layer, expert) in members.get(&decision.group).ok_or_else(|| {
            ColicError::Usage(format!("optimizer lost group membership for `{}`", decision.group))
        })? {
            by_expert.insert((layer, expert), representation);
        }
    }
    Ok(SelectedOptimization {
        frontier: plans,
        selected,
        by_expert,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::{Matrix, ModelGeometry}, source::TensorRef};
    use std::path::PathBuf;

    fn tensor(name: &str, len: u64, shape: Vec<u64>) -> TensorRef {
        TensorRef {
            source: PathBuf::from(format!("fixture-{name}.bin")),
            offset: 0,
            len,
            dtype: "BF16".into(),
            shape,
        }
    }

    fn expert(layer: u32, id: u32, sensitive_scale: u32) -> RoutedExpert {
        let rows = 32 * sensitive_scale;
        let cols = 32;
        let matrix = |role: &str| Matrix {
            source: tensor(role, u64::from(rows) * u64::from(cols) * 2, vec![u64::from(rows), u64::from(cols)]),
            rows,
            columns: cols,
            scale: None,
        };
        RoutedExpert {
            layer,
            expert: id,
            gate: matrix("gate"),
            up: matrix("up"),
            down: Matrix {
                source: tensor("down", u64::from(cols) * u64::from(rows) * 2, vec![u64::from(cols), u64::from(rows)]),
                rows: cols,
                columns: rows,
                scale: None,
            },
        }
    }

    fn fixture() -> SemanticModel {
        let mut routed_experts = BTreeMap::new();
        for layer in 0..4 {
            routed_experts.insert((layer, 0), expert(layer, 0, 1));
        }
        SemanticModel {
            architecture: Architecture::Qwen3_5MoeMoE,
            geometry: ModelGeometry {
                hidden_size: 64,
                layers: 4,
                routed_experts_per_layer: 1,
                moe_intermediate_size: 32,
                vocab_size: 128,
                hc_mult: 1,
                num_hash_layers: 0,
                experts_per_token: 1,
                attention_heads: 1,
                head_dim: 16,
                num_key_value_heads: 1,
                linear_key_head_dim: 0,
                q_lora_rank: 0,
                o_groups: 0,
                o_lora_rank: 0,
                index_heads: 0,
                index_head_dim: 0,
                compression_ratios: vec![],
            },
            routed_experts,
            global_tensors: BTreeMap::new(),
            layer_static_tensors: BTreeMap::new(),
            resident_tensors: BTreeMap::new(),
        }
    }

    #[test]
    fn compiler_frontier_is_deterministic_and_contains_mixed_choice() {
        let model = fixture();
        let machine = MachineProfile {
            operating_system: "linux",
            architecture: "x86_64",
            ram_bytes: Some(8 * 1024 * MIB),
            unified_memory: false,
            metal_available: false,
            apple8_abi: false,
            avx2: true,
            apple_gpu_family_min: 8,
        };
        let (a, members) = build_plans(
            &model,
            target::LINUX_X86_64_AVX2_V1,
            &machine,
            ContextConstraint::required(1024),
            None,
        )
        .unwrap();
        let (b, _) = build_plans(
            &model,
            target::LINUX_X86_64_AVX2_V1,
            &machine,
            ContextConstraint::required(1024),
            None,
        )
        .unwrap();
        assert_eq!(a, b);
        assert!(!members.is_empty());
        assert!(a.iter().any(|plan| {
            let exact = plan.decisions.iter().filter(|decision| decision.option_id == "exact").count();
            let small = plan.decisions.iter().filter(|decision| decision.option_id == "mxfp4").count();
            exact > 0 && small > 0
        }));
    }
}
