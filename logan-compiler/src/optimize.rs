use std::{collections::BTreeMap, fs, path::Path};

use logan_ir::{
    BUILTIN_COST_MODEL_V1, CandidateGroup, ContextConstraint, OptimizerInput, ParetoPlan,
    Placement, QuantSpec, RepresentationCandidate, material_plans, select_plan,
};

use crate::{
    error::{ColicError, Result},
    ir::{RoutedExpert, SemanticModel},
    quant::mxfp4_record,
    target::{self, MachineProfile, TargetProfile},
    target_registry,
};

pub const CANDIDATE_KEEP: &str = "keep";
pub const CANDIDATE_MXFP4: &str = "mxfp4";
const DISPATCH_KEEP: u16 = 1;
const DISPATCH_MXFP4: u16 = 2;
const DEFAULT_EXPERT_CACHE_SLOTS: u64 = 256;
const HETEROGENEITY_SWITCH_PENALTY: u64 = 64;

#[derive(Debug, Clone, Default)]
pub struct CalibrationScores {
    scores: BTreeMap<(String, String), u64>,
}

impl CalibrationScores {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let Some(path) = path else {
            return Ok(Self::default());
        };
        let bytes = fs::read(path).map_err(|source| ColicError::Io {
            path: path.to_owned(),
            source,
        })?;
        let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
            ColicError::Usage(format!("invalid optimizer calibration JSON: {error}"))
        })?;
        let scores = value
            .get("scores")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                ColicError::Usage("optimizer calibration must contain a `scores` array".into())
            })?;
        let mut parsed = BTreeMap::new();
        for score in scores {
            let group = score
                .get("group")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| ColicError::Usage("calibration score needs `group`".into()))?;
            let candidate = score
                .get("candidate")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| ColicError::Usage("calibration score needs `candidate`".into()))?;
            let quality = score
                .get("quality_loss_ppm")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| {
                    ColicError::Usage(
                        "calibration score needs non-negative `quality_loss_ppm`".into(),
                    )
                })?;
            let key = (group.to_owned(), candidate.to_owned());
            if parsed.insert(key.clone(), quality).is_some() {
                return Err(ColicError::Usage(format!(
                    "duplicate calibration score for {}/{}",
                    key.0, key.1
                )));
            }
        }
        Ok(Self { scores: parsed })
    }

    pub(crate) fn quality(&self, group: &str, candidate: &str, fallback: u64) -> u64 {
        self.scores
            .get(&(group.to_owned(), candidate.to_owned()))
            .copied()
            .unwrap_or(fallback)
    }
}

#[derive(Debug, Clone)]
pub struct CompileOptimization {
    pub plans: Vec<ParetoPlan>,
}

impl CompileOptimization {
    pub fn select(&self, selector: &str) -> Result<ParetoPlan> {
        select_plan(&self.plans, selector).cloned().ok_or_else(|| {
            ColicError::Usage(format!(
                "unknown optimizer plan `{selector}`; expected one of {}",
                self.plans
                    .iter()
                    .flat_map(|plan| plan
                        .labels
                        .iter()
                        .cloned()
                        .chain(std::iter::once(plan.id.clone())))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
    }

    pub fn balanced_or_first(&self) -> Result<ParetoPlan> {
        self.plans
            .iter()
            .find(|plan| plan.labels.iter().any(|label| label == "balanced"))
            .or_else(|| self.plans.first())
            .cloned()
            .ok_or_else(|| ColicError::Usage("optimizer produced no material plans".into()))
    }
}

pub fn compile_plans(
    model: &SemanticModel,
    model_root: &Path,
    target_profile: TargetProfile,
    machine: &MachineProfile,
    context: ContextConstraint,
    calibration_path: Option<&Path>,
    cache_slots: Option<u64>,
) -> Result<CompileOptimization> {
    let calibration = CalibrationScores::load(calibration_path)?;
    let (input, context_planning) = compile_optimizer_input(
        model,
        model_root,
        target_profile,
        machine,
        context,
        &calibration,
        cache_slots.unwrap_or(DEFAULT_EXPERT_CACHE_SLOTS).max(1),
    )?;
    let mut plans = material_plans(&input).map_err(|detail| ColicError::Unsupported {
        stage: "target planning",
        detail,
    })?;
    context_planning.enrich_plans(&mut plans)?;
    Ok(CompileOptimization { plans })
}

fn compile_optimizer_input(
    model: &SemanticModel,
    model_root: &Path,
    target_profile: TargetProfile,
    machine: &MachineProfile,
    context: ContextConstraint,
    calibration: &CalibrationScores,
    cache_slots: u64,
) -> Result<(OptimizerInput, crate::context_plan::ContextPlanning)> {
    let memory_budget_bytes = machine
        .ram_bytes
        .unwrap_or(target::machine::DEFAULT_POOL_BUDGET);
    let mut base_resident_bytes = 0_u64;
    let mut base_package_bytes = 0_u64;
    let mut base_storage_traffic_bytes = 0_u64;
    let mut account_tensor = |name: &str, tensor: &crate::source::TensorRef| -> Result<()> {
        let stored = target::exact_tensor_stored_bytes(tensor)?;
        if !is_streamed_ple_tensor(name) {
            base_resident_bytes =
                checked_add(base_resident_bytes, tensor.len, "resident tensor bytes")?;
        }
        base_package_bytes = checked_add(base_package_bytes, stored, "package tensor bytes")?;
        base_storage_traffic_bytes =
            checked_add(base_storage_traffic_bytes, stored, "tensor storage traffic")?;
        Ok(())
    };
    for (name, tensor) in &model.global_tensors {
        account_tensor(name, tensor)?;
    }
    for tensors in model.layer_static_tensors.values() {
        for (name, tensor) in tensors {
            account_tensor(name, tensor)?;
        }
    }
    for (name, tensor) in &model.resident_tensors {
        account_tensor(name, tensor)?;
    }
    drop(account_tensor);

    let context_planning =
        crate::context_plan::plan_from_package(model_root, context, machine, base_resident_bytes)?;

    let layer_count = u64::from(model.geometry.layers.max(1));
    let slots_per_layer = cache_slots.div_ceil(layer_count).max(1);
    let mut groups = Vec::new();
    for layer in 0..model.geometry.layers {
        let experts = model
            .routed_experts
            .range((layer, 0)..=(layer, u32::MAX))
            .map(|(_, expert)| expert)
            .collect::<Vec<_>>();
        if experts.is_empty() {
            continue;
        }
        let key = format!("layer:{layer}:routed-experts");
        let mut options = Vec::new();

        if target_profile != target::MACOS_ARM64_METAL_APPLE8_V1 {
            let package_bytes = sum_experts(&experts, target::exact_expert_stored_bytes)?;
            let max_decoded = max_experts(&experts, target::exact_expert_decoded_bytes)?;
            let per_token = expert_token_traffic(
                &experts,
                model.geometry.experts_per_token,
                target::exact_expert_stored_bytes,
            )?;
            options.push(RepresentationCandidate {
                id: CANDIDATE_KEEP.into(),
                quant: QuantSpec {
                    kind: source_expert_kind(experts[0]).into(),
                    scale: None,
                },
                layout: 0,
                placement: Placement::Streamed,
                resident_bytes: checked_mul(max_decoded, slots_per_layer, "exact expert cache")?,
                package_bytes,
                storage_traffic_bytes: per_token,
                latency_cost: cost_units(per_token),
                quality_loss_ppm: calibration.quality(&key, CANDIDATE_KEEP, 0),
                dispatch_class: DISPATCH_KEEP,
                rationale:
                    "preserve the source expert representation and its verified runtime kernel"
                        .into(),
            });
        } else if experts.iter().all(|expert| canonical_mxfp4(expert)) {
            let package_bytes = sum_experts(&experts, target::apple8_expert_stored_bytes)?;
            let max_decoded = max_experts(&experts, target::apple8_expert_decoded_bytes)?;
            let per_token = expert_token_traffic(
                &experts,
                model.geometry.experts_per_token,
                target::apple8_expert_stored_bytes,
            )?;
            options.push(RepresentationCandidate {
                id: CANDIDATE_KEEP.into(),
                quant: QuantSpec {
                    kind: "mxfp4-tile8x32".into(),
                    scale: Some("f8-e8m0/1x32".into()),
                },
                layout: target_registry::APPLE8_MXFP4_TILE_LAYOUT,
                placement: Placement::Streamed,
                resident_bytes: checked_mul(max_decoded, slots_per_layer, "Apple8 expert cache")?,
                package_bytes,
                storage_traffic_bytes: per_token,
                latency_cost: cost_units(per_token),
                quality_loss_ppm: calibration.quality(&key, CANDIDATE_KEEP, 0),
                dispatch_class: DISPATCH_MXFP4,
                rationale:
                    "losslessly repack canonical MXFP4 into the target-native Apple8 tile ABI"
                        .into(),
            });
        }

        if experts.iter().all(|expert| bf16_unscaled(expert)) {
            let (package_bytes, max_decoded, per_token, layout, quant_kind, scale) =
                if target_profile == target::MACOS_ARM64_METAL_APPLE8_V1 {
                    (
                        sum_experts(&experts, target::apple8_expert_stored_bytes)?,
                        max_experts(&experts, target::apple8_expert_decoded_bytes)?,
                        expert_token_traffic(
                            &experts,
                            model.geometry.experts_per_token,
                            target::apple8_expert_stored_bytes,
                        )?,
                        target_registry::APPLE8_MXFP4_TILE_LAYOUT,
                        "mxfp4-tile8x32",
                        Some("f8-e8m0/1x32".into()),
                    )
                } else {
                    (
                        sum_experts(&experts, mxfp4_record::stored_bytes)?,
                        max_experts(&experts, mxfp4_record::resident_bytes)?,
                        expert_token_traffic(
                            &experts,
                            model.geometry.experts_per_token,
                            mxfp4_record::stored_bytes,
                        )?,
                        0,
                        "mxfp4",
                        Some("e8m0/1x32".into()),
                    )
                };
            let fallback_quality = layer_quant_quality_prior(layer, model.geometry.layers);
            options.push(RepresentationCandidate {
                id: CANDIDATE_MXFP4.into(),
                quant: QuantSpec {
                    kind: quant_kind.into(),
                    scale,
                },
                layout,
                placement: Placement::Streamed,
                resident_bytes: checked_mul(max_decoded, slots_per_layer, "MXFP4 expert cache")?,
                package_bytes,
                storage_traffic_bytes: per_token,
                latency_cost: cost_units(per_token),
                quality_loss_ppm: calibration.quality(
                    &key,
                    CANDIDATE_MXFP4,
                    fallback_quality,
                ),
                dispatch_class: DISPATCH_MXFP4,
                rationale: format!(
                    "fresh BF16→MXFP4 quantization; built-in quality prior={}ppm for layer {layer}, overrideable by calibration",
                    fallback_quality
                ),
            });
        }

        if options.is_empty() {
            return Err(ColicError::Unsupported {
                stage: "target planning",
                detail: format!(
                    "optimizer found no real runtime representation for routed expert group `{key}` on target `{}`",
                    target_profile.name
                ),
            });
        }
        groups.push(CandidateGroup { key, options });
    }

    Ok((
        OptimizerInput {
            cost_model: BUILTIN_COST_MODEL_V1.into(),
            groups,
            context_constraint: context,
            context_candidates: context_planning.optimizer_candidates(),
            memory_budget_bytes,
            base_resident_bytes,
            base_package_bytes,
            base_storage_traffic_bytes,
            base_latency_cost: 0,
            base_quality_loss_ppm: 0,
            heterogeneity_switch_penalty: HETEROGENEITY_SWITCH_PENALTY,
        },
        context_planning,
    ))
}

fn is_streamed_ple_tensor(name: &str) -> bool {
    name.contains("ple.ple_embedding.ngram_embedding.shard_")
}

pub fn selected_layer_quantization(plan: &ParetoPlan) -> Result<BTreeMap<u32, bool>> {
    let mut selected = BTreeMap::new();
    for decision in &plan.decisions {
        let Some(layer) = decision
            .group
            .strip_prefix("layer:")
            .and_then(|rest| rest.strip_suffix(":routed-experts"))
            .and_then(|value| value.parse::<u32>().ok())
        else {
            continue;
        };
        let mxfp4 = match decision.chosen.id.as_str() {
            CANDIDATE_KEEP => false,
            CANDIDATE_MXFP4 => true,
            other => {
                return Err(ColicError::Usage(format!(
                    "optimizer selected unknown compile candidate `{other}` for layer {layer}"
                )));
            }
        };
        selected.insert(layer, mxfp4);
    }
    Ok(selected)
}

pub fn plan_summary(plan: &ParetoPlan) -> String {
    format!(
        "{} [{}] quality={}ppm context={} latency={} resident={} package={} traffic={}",
        plan.id,
        if plan.labels.is_empty() {
            "unlabeled".into()
        } else {
            plan.labels.join(",")
        },
        plan.metrics.quality_loss_ppm,
        plan.metrics.context_tokens,
        plan.metrics.latency_cost,
        plan.metrics.resident_bytes,
        plan.metrics.package_bytes,
        plan.metrics.storage_traffic_bytes,
    )
}

fn source_expert_kind(expert: &RoutedExpert) -> &'static str {
    if canonical_mxfp4(expert) {
        "mxfp4"
    } else if bf16_unscaled(expert) {
        "bf16"
    } else {
        "exact"
    }
}

fn bf16_unscaled(expert: &RoutedExpert) -> bool {
    [&expert.gate, &expert.up, &expert.down]
        .into_iter()
        .all(|matrix| matrix.source.dtype == "BF16" && matrix.scale.is_none())
}

fn canonical_mxfp4(expert: &RoutedExpert) -> bool {
    [&expert.gate, &expert.up, &expert.down]
        .into_iter()
        .all(|matrix| {
            matrix.source.dtype == "I8"
                && matrix
                    .scale
                    .as_ref()
                    .is_some_and(|scale| matches!(scale.dtype.as_str(), "F8_E8M0" | "F8_E8M0FNU"))
        })
}

pub(crate) fn layer_quant_quality_prior(layer: u32, layers: u32) -> u64 {
    if layers <= 1 {
        return 4_000;
    }
    let last = layers - 1;
    let edge = layer.min(last.saturating_sub(layer));
    let edge_penalty = match edge {
        0 => 2_400,
        1 => 1_200,
        2 => 600,
        _ => 0,
    };
    // A modest deterministic shape prior, deliberately not uniform. Actual
    // calibration data overrides this per group/candidate.
    1_000 + edge_penalty + u64::from(layer % 5) * 75
}

fn expert_token_traffic(
    experts: &[&RoutedExpert],
    experts_per_token: u32,
    bytes: impl Fn(&RoutedExpert) -> Result<u64>,
) -> Result<u64> {
    let total = sum_experts(experts, bytes)?;
    let average = total.div_ceil(experts.len() as u64);
    checked_mul(
        average,
        u64::from(experts_per_token.max(1)),
        "expert token traffic",
    )
}

fn sum_experts(
    experts: &[&RoutedExpert],
    bytes: impl Fn(&RoutedExpert) -> Result<u64>,
) -> Result<u64> {
    experts.iter().try_fold(0_u64, |total, expert| {
        checked_add(total, bytes(expert)?, "expert byte sum")
    })
}

fn max_experts(
    experts: &[&RoutedExpert],
    bytes: impl Fn(&RoutedExpert) -> Result<u64>,
) -> Result<u64> {
    experts
        .iter()
        .map(|expert| bytes(expert))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .ok_or_else(|| ColicError::Usage("expert group is empty".into()))
}

fn cost_units(bytes: u64) -> u64 {
    bytes.div_ceil(4096).max(1)
}

fn checked_add(left: u64, right: u64, what: &str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| ColicError::Usage(format!("{what} overflows u64")))
}

fn checked_mul(left: u64, right: u64, what: &str) -> Result<u64> {
    left.checked_mul(right)
        .ok_or_else(|| ColicError::Usage(format!("{what} overflows u64")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_prior_is_not_uniform_and_protects_model_edges_more() {
        assert!(layer_quant_quality_prior(0, 40) > layer_quant_quality_prior(10, 40));
        assert!(layer_quant_quality_prior(39, 40) > layer_quant_quality_prior(20, 40));
        assert_ne!(
            layer_quant_quality_prior(10, 40),
            layer_quant_quality_prior(11, 40)
        );
    }

    #[test]
    fn calibration_json_overrides_one_group_without_affecting_others() {
        let root = std::env::temp_dir().join(format!(
            "logan-calibration-{}-{}.json",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        fs::write(
            &root,
            r#"{"scores":[{"group":"layer:7:routed-experts","candidate":"mxfp4","quality_loss_ppm":123}]}"#,
        )
        .unwrap();
        let scores = CalibrationScores::load(Some(&root)).unwrap();
        assert_eq!(scores.quality("layer:7:routed-experts", "mxfp4", 999), 123);
        assert_eq!(scores.quality("layer:8:routed-experts", "mxfp4", 999), 999);
        let _ = fs::remove_file(root);
    }
}
