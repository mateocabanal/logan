use serde::{Deserialize, Serialize};

use crate::{ContextConstraint, ContextPlan, Placement, PlannerMemoryBudget, QuantSpec};

pub const BUILTIN_COST_MODEL_V1: &str = "logan-builtin-cost-v1";
const MAX_FRONTIER_STATES: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentationCandidate {
    pub id: String,
    pub quant: QuantSpec,
    pub layout: u16,
    pub placement: Placement,
    pub resident_bytes: u64,
    pub package_bytes: u64,
    pub storage_traffic_bytes: u64,
    pub latency_cost: u64,
    pub quality_loss_ppm: u64,
    /// Zero exempts the candidate from heterogeneity accounting. Non-zero
    /// classes model runtime dispatch families; switching between adjacent
    /// groups incurs the optimizer's configured dispatch penalty.
    pub dispatch_class: u16,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateGroup {
    pub key: String,
    pub options: Vec<RepresentationCandidate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextCandidate {
    pub tokens: u64,
    pub resident_bytes: u64,
    pub latency_cost: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizerInput {
    pub cost_model: String,
    pub groups: Vec<CandidateGroup>,
    pub context_constraint: ContextConstraint,
    pub context_candidates: Vec<ContextCandidate>,
    pub memory_budget_bytes: u64,
    pub base_resident_bytes: u64,
    pub base_package_bytes: u64,
    pub base_storage_traffic_bytes: u64,
    pub base_latency_cost: u64,
    pub base_quality_loss_ppm: u64,
    pub heterogeneity_switch_penalty: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanMetrics {
    pub quality_loss_ppm: u64,
    pub context_tokens: u64,
    pub latency_cost: u64,
    pub resident_bytes: u64,
    pub package_bytes: u64,
    pub storage_traffic_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectedCandidate {
    pub id: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanDecision {
    pub group: String,
    pub chosen: RepresentationCandidate,
    pub rejected: Vec<RejectedCandidate>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParetoPlan {
    pub id: String,
    /// Human-friendly aliases such as `quality`, `balanced`,
    /// `long-context`, and `latency`. One physical plan can satisfy multiple
    /// aliases when the frontier collapses to fewer than four useful points.
    pub labels: Vec<String>,
    pub cost_model: String,
    pub metrics: PlanMetrics,
    pub decisions: Vec<PlanDecision>,
    /// Compiler-populated architecture-specific context state for this point.
    pub context_plan: Option<ContextPlan>,
    /// Compiler-populated complete memory budget used to admit this point.
    pub memory_budget: Option<PlannerMemoryBudget>,
}

#[derive(Debug, Clone)]
struct State {
    quality_loss_ppm: u64,
    latency_cost: u64,
    resident_bytes: u64,
    package_bytes: u64,
    storage_traffic_bytes: u64,
    last_dispatch_class: u16,
    choices: Vec<usize>,
}

impl State {
    fn metrics_with_context(&self, context: ContextCandidate) -> Option<PlanMetrics> {
        Some(PlanMetrics {
            quality_loss_ppm: self.quality_loss_ppm,
            context_tokens: context.tokens,
            latency_cost: self.latency_cost.checked_add(context.latency_cost)?,
            resident_bytes: self.resident_bytes.checked_add(context.resident_bytes)?,
            package_bytes: self.package_bytes,
            storage_traffic_bytes: self.storage_traffic_bytes,
        })
    }
}

pub fn pareto_plans(input: &OptimizerInput) -> Result<Vec<ParetoPlan>, String> {
    validate_input(input)?;

    let mut states = vec![State {
        quality_loss_ppm: input.base_quality_loss_ppm,
        latency_cost: input.base_latency_cost,
        resident_bytes: input.base_resident_bytes,
        package_bytes: input.base_package_bytes,
        storage_traffic_bytes: input.base_storage_traffic_bytes,
        last_dispatch_class: 0,
        choices: Vec::with_capacity(input.groups.len()),
    }];

    for group in &input.groups {
        let mut next = Vec::new();
        for state in &states {
            for (option_index, option) in group.options.iter().enumerate() {
                let mut latency_cost = state
                    .latency_cost
                    .checked_add(option.latency_cost)
                    .ok_or_else(|| "optimizer latency cost overflow".to_owned())?;
                if option.dispatch_class != 0
                    && state.last_dispatch_class != 0
                    && option.dispatch_class != state.last_dispatch_class
                {
                    latency_cost = latency_cost
                        .checked_add(input.heterogeneity_switch_penalty)
                        .ok_or_else(|| "optimizer heterogeneity cost overflow".to_owned())?;
                }
                let resident_bytes = state
                    .resident_bytes
                    .checked_add(option.resident_bytes)
                    .ok_or_else(|| "optimizer resident bytes overflow".to_owned())?;
                if resident_bytes > input.memory_budget_bytes {
                    continue;
                }
                let mut choices = state.choices.clone();
                choices.push(option_index);
                next.push(State {
                    quality_loss_ppm: state
                        .quality_loss_ppm
                        .checked_add(option.quality_loss_ppm)
                        .ok_or_else(|| "optimizer quality cost overflow".to_owned())?,
                    latency_cost,
                    resident_bytes,
                    package_bytes: state
                        .package_bytes
                        .checked_add(option.package_bytes)
                        .ok_or_else(|| "optimizer package bytes overflow".to_owned())?,
                    storage_traffic_bytes: state
                        .storage_traffic_bytes
                        .checked_add(option.storage_traffic_bytes)
                        .ok_or_else(|| "optimizer storage traffic overflow".to_owned())?,
                    last_dispatch_class: if option.dispatch_class == 0 {
                        state.last_dispatch_class
                    } else {
                        option.dispatch_class
                    },
                    choices,
                });
            }
        }
        states = prune_states(next);
        if states.is_empty() {
            return Err(format!(
                "no feasible physical representation remains after group `{}` within {} bytes",
                group.key, input.memory_budget_bytes
            ));
        }
    }

    let mut raw = Vec::<(PlanMetrics, Vec<usize>)>::new();
    for context in input
        .context_candidates
        .iter()
        .copied()
        .filter(|candidate| input.context_constraint.allows(candidate.tokens))
    {
        for state in &states {
            let Some(metrics) = state.metrics_with_context(context) else {
                continue;
            };
            if metrics.resident_bytes <= input.memory_budget_bytes {
                raw.push((metrics, state.choices.clone()));
            }
        }
    }
    if raw.is_empty() {
        return Err("no optimizer plan satisfies the context and memory constraints".into());
    }

    let keep = (0..raw.len())
        .filter(|&candidate| {
            !(0..raw.len())
                .any(|other| other != candidate && dominates(raw[other].0, raw[candidate].0))
        })
        .collect::<Vec<_>>();

    let mut plans = keep
        .into_iter()
        .map(|index| {
            let (metrics, choices) = &raw[index];
            build_plan(input, *metrics, choices)
        })
        .collect::<Vec<_>>();
    plans.sort_by(|left, right| plan_sort_key(left).cmp(&plan_sort_key(right)));
    plans.dedup_by(|left, right| left.id == right.id);
    Ok(plans)
}

/// Reduce a potentially larger frontier to at most four materially useful
/// user-facing points while preserving deterministic selection. Every
/// returned plan is still non-dominated because it is selected from the
/// already-pruned Pareto frontier.
pub fn material_plans(input: &OptimizerInput) -> Result<Vec<ParetoPlan>, String> {
    let frontier = pareto_plans(input)?;
    if frontier.is_empty() {
        return Ok(Vec::new());
    }

    let quality = best_index(&frontier, |plan| {
        (
            plan.metrics.quality_loss_ppm,
            u64::MAX - plan.metrics.context_tokens,
            plan.metrics.latency_cost,
            plan.metrics.resident_bytes,
            plan.metrics.package_bytes,
            plan.id.clone(),
        )
    });
    let latency = best_index(&frontier, |plan| {
        (
            plan.metrics.latency_cost,
            plan.metrics.quality_loss_ppm,
            u64::MAX - plan.metrics.context_tokens,
            plan.metrics.storage_traffic_bytes,
            plan.metrics.package_bytes,
            plan.id.clone(),
        )
    });
    let long_context = best_index(&frontier, |plan| {
        (
            u64::MAX - plan.metrics.context_tokens,
            plan.metrics.quality_loss_ppm,
            plan.metrics.latency_cost,
            plan.metrics.resident_bytes,
            plan.metrics.package_bytes,
            plan.id.clone(),
        )
    });
    let balanced = balanced_index(&frontier);

    let mut chosen = Vec::<ParetoPlan>::new();
    for (index, label) in [
        (quality, "quality"),
        (balanced, "balanced"),
        (long_context, "long-context"),
        (latency, "latency"),
    ] {
        if let Some(existing) = chosen.iter_mut().find(|plan| plan.id == frontier[index].id) {
            existing.labels.push(label.to_owned());
        } else {
            let mut plan = frontier[index].clone();
            plan.labels.push(label.to_owned());
            chosen.push(plan);
        }
    }
    for plan in &mut chosen {
        plan.labels.sort();
        plan.labels.dedup();
    }
    chosen.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(chosen)
}

pub fn select_plan<'a>(plans: &'a [ParetoPlan], selector: &str) -> Option<&'a ParetoPlan> {
    plans
        .iter()
        .find(|plan| plan.id == selector || plan.labels.iter().any(|label| label == selector))
}

fn validate_input(input: &OptimizerInput) -> Result<(), String> {
    if input.cost_model.trim().is_empty() {
        return Err("optimizer cost model id cannot be empty".into());
    }
    if input.memory_budget_bytes == 0 {
        return Err("optimizer memory budget must be non-zero".into());
    }
    if input.base_resident_bytes > input.memory_budget_bytes {
        return Err(format!(
            "fixed resident state needs {} bytes but the optimizer memory budget is {} bytes",
            input.base_resident_bytes, input.memory_budget_bytes
        ));
    }
    if input.context_candidates.is_empty() {
        return Err("optimizer requires at least one context candidate".into());
    }
    if !input
        .context_candidates
        .iter()
        .any(|candidate| input.context_constraint.allows(candidate.tokens))
    {
        return Err("no context candidate satisfies the requested context constraint".into());
    }
    for group in &input.groups {
        if group.key.is_empty() || group.options.is_empty() {
            return Err("optimizer groups require a non-empty key and at least one option".into());
        }
    }
    Ok(())
}

fn prune_states(mut states: Vec<State>) -> Vec<State> {
    states.sort_by(|left, right| state_sort_key(left).cmp(&state_sort_key(right)));
    let mut kept = Vec::<State>::new();
    'candidate: for state in states {
        for existing in &kept {
            if existing.last_dispatch_class == state.last_dispatch_class
                && state_dominates(existing, &state)
            {
                continue 'candidate;
            }
        }
        kept.retain(|existing| {
            existing.last_dispatch_class != state.last_dispatch_class
                || !state_dominates(&state, existing)
        });
        kept.push(state);
    }
    if kept.len() > MAX_FRONTIER_STATES {
        kept.sort_by(|left, right| state_trim_key(left).cmp(&state_trim_key(right)));
        kept.truncate(MAX_FRONTIER_STATES);
    }
    kept
}

fn state_dominates(left: &State, right: &State) -> bool {
    let no_worse = left.quality_loss_ppm <= right.quality_loss_ppm
        && left.latency_cost <= right.latency_cost
        && left.resident_bytes <= right.resident_bytes
        && left.package_bytes <= right.package_bytes
        && left.storage_traffic_bytes <= right.storage_traffic_bytes;
    let strictly_better = left.quality_loss_ppm < right.quality_loss_ppm
        || left.latency_cost < right.latency_cost
        || left.resident_bytes < right.resident_bytes
        || left.package_bytes < right.package_bytes
        || left.storage_traffic_bytes < right.storage_traffic_bytes;
    no_worse && strictly_better
}

fn dominates(left: PlanMetrics, right: PlanMetrics) -> bool {
    let no_worse = left.quality_loss_ppm <= right.quality_loss_ppm
        && left.context_tokens >= right.context_tokens
        && left.latency_cost <= right.latency_cost
        && left.resident_bytes <= right.resident_bytes
        && left.package_bytes <= right.package_bytes
        && left.storage_traffic_bytes <= right.storage_traffic_bytes;
    let strictly_better = left.quality_loss_ppm < right.quality_loss_ppm
        || left.context_tokens > right.context_tokens
        || left.latency_cost < right.latency_cost
        || left.resident_bytes < right.resident_bytes
        || left.package_bytes < right.package_bytes
        || left.storage_traffic_bytes < right.storage_traffic_bytes;
    no_worse && strictly_better
}

fn build_plan(input: &OptimizerInput, metrics: PlanMetrics, choices: &[usize]) -> ParetoPlan {
    let decisions = input
        .groups
        .iter()
        .zip(choices)
        .map(|(group, &choice)| {
            let chosen = group.options[choice].clone();
            let rejected = group
                .options
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != choice)
                .map(|(_, option)| RejectedCandidate {
                    id: option.id.clone(),
                    reason: format!(
                        "candidate q={}ppm latency={} resident={} package={} traffic={}: {}",
                        option.quality_loss_ppm,
                        option.latency_cost,
                        option.resident_bytes,
                        option.package_bytes,
                        option.storage_traffic_bytes,
                        option.rationale
                    ),
                })
                .collect();
            PlanDecision {
                group: group.key.clone(),
                chosen,
                rejected,
            }
        })
        .collect::<Vec<_>>();
    let id = stable_plan_id(&input.cost_model, metrics, &decisions);
    ParetoPlan {
        id,
        labels: Vec::new(),
        cost_model: input.cost_model.clone(),
        metrics,
        decisions,
        context_plan: None,
        memory_budget: None,
    }
}

fn stable_plan_id(cost_model: &str, metrics: PlanMetrics, decisions: &[PlanDecision]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    fn feed(hash: &mut u64, bytes: &[u8]) {
        for byte in bytes {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    feed(&mut hash, cost_model.as_bytes());
    for value in [
        metrics.quality_loss_ppm,
        metrics.context_tokens,
        metrics.latency_cost,
        metrics.resident_bytes,
        metrics.package_bytes,
        metrics.storage_traffic_bytes,
    ] {
        feed(&mut hash, &value.to_le_bytes());
    }
    for decision in decisions {
        feed(&mut hash, decision.group.as_bytes());
        feed(&mut hash, decision.chosen.id.as_bytes());
    }
    format!("p-{hash:016x}")
}

fn best_index<K: Ord>(plans: &[ParetoPlan], key: impl Fn(&ParetoPlan) -> K) -> usize {
    plans
        .iter()
        .enumerate()
        .min_by_key(|(_, plan)| key(plan))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn balanced_index(plans: &[ParetoPlan]) -> usize {
    let quality = ranks(plans, |plan| plan.metrics.quality_loss_ppm, false);
    let context = ranks(plans, |plan| plan.metrics.context_tokens, true);
    let latency = ranks(plans, |plan| plan.metrics.latency_cost, false);
    let resident = ranks(plans, |plan| plan.metrics.resident_bytes, false);
    let traffic = ranks(plans, |plan| plan.metrics.storage_traffic_bytes, false);
    (0..plans.len())
        .min_by_key(|&index| {
            let dimensions = [
                quality[index],
                context[index],
                latency[index],
                resident[index],
                traffic[index],
            ];
            (
                *dimensions.iter().max().unwrap_or(&0),
                dimensions.iter().sum::<usize>(),
                plans[index].id.clone(),
            )
        })
        .unwrap_or(0)
}

fn ranks(plans: &[ParetoPlan], value: impl Fn(&ParetoPlan) -> u64, reverse: bool) -> Vec<usize> {
    let mut order = (0..plans.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| {
        let value = value(&plans[index]);
        (
            if reverse { u64::MAX - value } else { value },
            plans[index].id.clone(),
        )
    });
    let mut ranks = vec![0; plans.len()];
    let mut previous = None;
    let mut tied_rank = 0_usize;
    for (position, index) in order.into_iter().enumerate() {
        let current = value(&plans[index]);
        if previous.is_some_and(|previous| previous != current) {
            tied_rank = position;
        }
        previous = Some(current);
        ranks[index] = tied_rank;
    }
    ranks
}

fn state_sort_key(state: &State) -> (u16, u64, u64, u64, u64, u64, Vec<usize>) {
    (
        state.last_dispatch_class,
        state.quality_loss_ppm,
        state.latency_cost,
        state.resident_bytes,
        state.package_bytes,
        state.storage_traffic_bytes,
        state.choices.clone(),
    )
}

fn state_trim_key(state: &State) -> (u64, u64, u64, u64, u64, u16, Vec<usize>) {
    (
        state
            .quality_loss_ppm
            .saturating_add(state.latency_cost)
            .saturating_add(state.resident_bytes / 4096)
            .saturating_add(state.storage_traffic_bytes / 4096),
        state.quality_loss_ppm,
        state.latency_cost,
        state.resident_bytes,
        state.package_bytes,
        state.last_dispatch_class,
        state.choices.clone(),
    )
}

fn plan_sort_key(plan: &ParetoPlan) -> (u64, u64, u64, u64, u64, u64, String) {
    (
        plan.metrics.quality_loss_ppm,
        u64::MAX - plan.metrics.context_tokens,
        plan.metrics.latency_cost,
        plan.metrics.resident_bytes,
        plan.metrics.package_bytes,
        plan.metrics.storage_traffic_bytes,
        plan.id.clone(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn option(
        id: &str,
        quality: u64,
        latency: u64,
        resident: u64,
        package: u64,
        class: u16,
    ) -> RepresentationCandidate {
        RepresentationCandidate {
            id: id.into(),
            quant: QuantSpec {
                kind: id.into(),
                scale: None,
            },
            layout: 0,
            placement: Placement::Resident,
            resident_bytes: resident,
            package_bytes: package,
            storage_traffic_bytes: package,
            latency_cost: latency,
            quality_loss_ppm: quality,
            dispatch_class: class,
            rationale: id.into(),
        }
    }

    fn input(groups: Vec<CandidateGroup>, memory: u64) -> OptimizerInput {
        OptimizerInput {
            cost_model: BUILTIN_COST_MODEL_V1.into(),
            groups,
            context_constraint: ContextConstraint::required(4096),
            context_candidates: vec![ContextCandidate {
                tokens: 4096,
                resident_bytes: 0,
                latency_cost: 0,
            }],
            memory_budget_bytes: memory,
            base_resident_bytes: 0,
            base_package_bytes: 0,
            base_storage_traffic_bytes: 0,
            base_latency_cost: 0,
            base_quality_loss_ppm: 0,
            heterogeneity_switch_penalty: 5,
        }
    }

    #[test]
    fn fixed_input_produces_a_deterministic_frontier_and_ids() {
        let spec = input(
            vec![
                CandidateGroup {
                    key: "layer:0".into(),
                    options: vec![
                        option("bf16", 0, 100, 60, 100, 1),
                        option("mxfp4", 2, 40, 20, 40, 2),
                    ],
                },
                CandidateGroup {
                    key: "layer:1".into(),
                    options: vec![
                        option("bf16", 0, 100, 60, 100, 1),
                        option("mxfp4", 8, 40, 20, 40, 2),
                    ],
                },
            ],
            120,
        );
        let first = material_plans(&spec).unwrap();
        let second = material_plans(&spec).unwrap();
        assert_eq!(first, second);
        assert!(first.iter().all(|plan| plan.id.starts_with("p-")));
    }

    #[test]
    fn mixed_plan_can_beat_both_global_choices_under_memory_and_quality_limits() {
        let spec = input(
            vec![
                CandidateGroup {
                    key: "layer:0".into(),
                    options: vec![
                        option("bf16", 0, 100, 60, 100, 1),
                        option("mxfp4", 1, 40, 20, 40, 2),
                    ],
                },
                CandidateGroup {
                    key: "layer:1".into(),
                    options: vec![
                        option("bf16", 0, 100, 60, 100, 1),
                        option("mxfp4", 20, 40, 20, 40, 2),
                    ],
                },
            ],
            80,
        );
        let frontier = pareto_plans(&spec).unwrap();
        let mixed = frontier.iter().find(|plan| {
            plan.decisions[0].chosen.id == "mxfp4" && plan.decisions[1].chosen.id == "bf16"
        });
        let mixed = mixed.expect("mixed plan must survive the Pareto frontier");
        assert!(mixed.metrics.resident_bytes <= 80);
        assert!(mixed.metrics.quality_loss_ppm <= 5);
        // Global BF16 is infeasible at 120 resident bytes. Global MXFP4 fits
        // memory but has quality loss 21ppm, so only the mixed plan satisfies
        // both the 80-byte memory and <=5ppm quality envelope.
        assert!(
            !frontier
                .iter()
                .any(|plan| { plan.decisions.iter().all(|d| d.chosen.id == "bf16") })
        );
        assert!(frontier.iter().any(|plan| {
            plan.decisions.iter().all(|d| d.chosen.id == "mxfp4")
                && plan.metrics.quality_loss_ppm == 21
        }));
    }

    #[test]
    fn material_plans_are_selectable_by_alias_or_stable_id() {
        let spec = input(
            vec![CandidateGroup {
                key: "layer:0".into(),
                options: vec![
                    option("bf16", 0, 100, 60, 100, 1),
                    option("mxfp4", 3, 40, 20, 40, 2),
                ],
            }],
            100,
        );
        let plans = material_plans(&spec).unwrap();
        let balanced = select_plan(&plans, "balanced").unwrap();
        assert_eq!(select_plan(&plans, &balanced.id).unwrap().id, balanced.id);
    }

    #[test]
    fn balanced_plan_surfaces_context_memory_compromise() {
        let spec = OptimizerInput {
            cost_model: BUILTIN_COST_MODEL_V1.into(),
            groups: Vec::new(),
            context_constraint: ContextConstraint::maximum(131_072),
            context_candidates: vec![
                ContextCandidate {
                    tokens: 32_768,
                    resident_bytes: 3,
                    latency_cost: 0,
                },
                ContextCandidate {
                    tokens: 65_536,
                    resident_bytes: 6,
                    latency_cost: 0,
                },
                ContextCandidate {
                    tokens: 131_072,
                    resident_bytes: 12,
                    latency_cost: 0,
                },
            ],
            memory_budget_bytes: 16,
            base_resident_bytes: 0,
            base_package_bytes: 0,
            base_storage_traffic_bytes: 0,
            base_latency_cost: 0,
            base_quality_loss_ppm: 0,
            heterogeneity_switch_penalty: 0,
        };
        let plans = material_plans(&spec).unwrap();
        let balanced = select_plan(&plans, "balanced").unwrap();
        let long = select_plan(&plans, "long-context").unwrap();
        assert_eq!(balanced.metrics.context_tokens, 65_536);
        assert_eq!(long.metrics.context_tokens, 131_072);
        assert_ne!(balanced.id, long.id);
    }

    #[test]
    fn switch_penalty_is_applied_between_dispatch_classes() {
        let mut spec = input(
            vec![
                CandidateGroup {
                    key: "layer:0".into(),
                    options: vec![option("a", 0, 10, 1, 1, 1)],
                },
                CandidateGroup {
                    key: "layer:1".into(),
                    options: vec![option("b", 0, 10, 1, 1, 2)],
                },
            ],
            10,
        );
        spec.heterogeneity_switch_penalty = 7;
        let plan = pareto_plans(&spec).unwrap().pop().unwrap();
        assert_eq!(plan.metrics.latency_cost, 27);
    }
}
