use serde::{Deserialize, Serialize};

use crate::{ContextConstraint, Placement};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalOption {
    pub id: String,
    pub representation: String,
    pub layout: String,
    pub placement: Placement,
    /// Fixed-point quality penalty. Zero is lossless relative to the source.
    pub quality_loss: u64,
    /// Relative deterministic latency cost; lower is better.
    pub latency_cost: u64,
    /// Incremental resident/runtime working-set bytes for this choice.
    pub resident_bytes: u64,
    /// Stored package bytes attributable to this group.
    pub package_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionGroup {
    pub id: String,
    pub options: Vec<PhysicalOption>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextCandidate {
    pub tokens: u64,
    pub state_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanMetrics {
    pub quality_loss: u64,
    pub latency_cost: u64,
    pub resident_bytes: u64,
    pub package_bytes: u64,
    pub context_tokens: u64,
    pub representation_switches: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanDecision {
    pub group: String,
    pub option_id: String,
    pub representation: String,
    pub layout: String,
    pub placement: Placement,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParetoPlan {
    pub id: String,
    pub label: Option<String>,
    pub metrics: PlanMetrics,
    pub decisions: Vec<PlanDecision>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizeInput {
    pub groups: Vec<DecisionGroup>,
    pub contexts: Vec<ContextCandidate>,
    pub context_constraint: ContextConstraint,
    pub base_resident_bytes: u64,
    pub base_package_bytes: u64,
    pub memory_budget_bytes: u64,
    pub heterogeneity_penalty: u64,
}

#[derive(Clone)]
struct State {
    quality_loss: u64,
    latency_cost: u64,
    resident_bytes: u64,
    package_bytes: u64,
    switches: u32,
    last_representation: Option<String>,
    decisions: Vec<PlanDecision>,
}

impl State {
    fn extend(&self, group: &DecisionGroup, option: &PhysicalOption, penalty: u64) -> Option<Self> {
        let switched = self
            .last_representation
            .as_deref()
            .is_some_and(|previous| previous != option.representation);
        Some(Self {
            quality_loss: self.quality_loss.checked_add(option.quality_loss)?,
            latency_cost: self
                .latency_cost
                .checked_add(option.latency_cost)?
                .checked_add(if switched { penalty } else { 0 })?,
            resident_bytes: self.resident_bytes.checked_add(option.resident_bytes)?,
            package_bytes: self.package_bytes.checked_add(option.package_bytes)?,
            switches: self.switches.checked_add(u32::from(switched))?,
            last_representation: Some(option.representation.clone()),
            decisions: {
                let mut decisions = self.decisions.clone();
                decisions.push(PlanDecision {
                    group: group.id.clone(),
                    option_id: option.id.clone(),
                    representation: option.representation.clone(),
                    layout: option.layout.clone(),
                    placement: option.placement,
                    reason: format!(
                        "selected {} / {} because it remains on the non-dominated frontier for quality, latency, memory and storage",
                        option.representation, option.layout
                    ),
                });
                decisions
            },
        })
    }
}

fn dominates_state(left: &State, right: &State) -> bool {
    let no_worse = left.quality_loss <= right.quality_loss
        && left.latency_cost <= right.latency_cost
        && left.resident_bytes <= right.resident_bytes
        && left.package_bytes <= right.package_bytes;
    let strictly_better = left.quality_loss < right.quality_loss
        || left.latency_cost < right.latency_cost
        || left.resident_bytes < right.resident_bytes
        || left.package_bytes < right.package_bytes;
    no_worse && strictly_better
}

fn prune_states(mut states: Vec<State>) -> Vec<State> {
    states.sort_by(|left, right| {
        (
            left.quality_loss,
            left.latency_cost,
            left.resident_bytes,
            left.package_bytes,
            left.switches,
            &left.last_representation,
        )
            .cmp(&(
                right.quality_loss,
                right.latency_cost,
                right.resident_bytes,
                right.package_bytes,
                right.switches,
                &right.last_representation,
            ))
    });
    states.dedup_by(|left, right| {
        left.quality_loss == right.quality_loss
            && left.latency_cost == right.latency_cost
            && left.resident_bytes == right.resident_bytes
            && left.package_bytes == right.package_bytes
            && left.last_representation == right.last_representation
    });
    let mut kept = Vec::new();
    'candidate: for state in states {
        for other in &kept {
            if dominates_state(other, &state)
                && other.last_representation == state.last_representation
            {
                continue 'candidate;
            }
        }
        kept.retain(|other| {
            !(dominates_state(&state, other)
                && other.last_representation == state.last_representation)
        });
        kept.push(state);
    }
    kept
}

fn dominates_plan(left: &ParetoPlan, right: &ParetoPlan) -> bool {
    let a = left.metrics;
    let b = right.metrics;
    let no_worse = a.quality_loss <= b.quality_loss
        && a.latency_cost <= b.latency_cost
        && a.resident_bytes <= b.resident_bytes
        && a.package_bytes <= b.package_bytes
        && a.context_tokens >= b.context_tokens;
    let strictly_better = a.quality_loss < b.quality_loss
        || a.latency_cost < b.latency_cost
        || a.resident_bytes < b.resident_bytes
        || a.package_bytes < b.package_bytes
        || a.context_tokens > b.context_tokens;
    no_worse && strictly_better
}

fn stable_plan_id(metrics: PlanMetrics, decisions: &[PlanDecision]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };
    for value in [
        metrics.quality_loss,
        metrics.latency_cost,
        metrics.resident_bytes,
        metrics.package_bytes,
        metrics.context_tokens,
        u64::from(metrics.representation_switches),
    ] {
        feed(&value.to_le_bytes());
    }
    for decision in decisions {
        feed(decision.group.as_bytes());
        feed(decision.option_id.as_bytes());
    }
    format!("p-{hash:016x}")
}

fn representative_indices(plans: &[ParetoPlan]) -> Vec<(usize, &'static str)> {
    if plans.is_empty() {
        return Vec::new();
    }
    let quality = (0..plans.len()).min_by_key(|&index| {
        let m = plans[index].metrics;
        (
            m.quality_loss,
            m.latency_cost,
            std::cmp::Reverse(m.context_tokens),
            m.package_bytes,
        )
    });
    let latency = (0..plans.len()).min_by_key(|&index| {
        let m = plans[index].metrics;
        (
            m.latency_cost,
            m.quality_loss,
            std::cmp::Reverse(m.context_tokens),
            m.package_bytes,
        )
    });
    let long_context = (0..plans.len()).min_by_key(|&index| {
        let m = plans[index].metrics;
        (
            std::cmp::Reverse(m.context_tokens),
            m.quality_loss,
            m.latency_cost,
            m.package_bytes,
        )
    });

    let min_q = plans.iter().map(|p| p.metrics.quality_loss).min().unwrap();
    let max_q = plans.iter().map(|p| p.metrics.quality_loss).max().unwrap();
    let min_l = plans.iter().map(|p| p.metrics.latency_cost).min().unwrap();
    let max_l = plans.iter().map(|p| p.metrics.latency_cost).max().unwrap();
    let min_s = plans.iter().map(|p| p.metrics.package_bytes).min().unwrap();
    let max_s = plans.iter().map(|p| p.metrics.package_bytes).max().unwrap();
    let min_c = plans
        .iter()
        .map(|p| p.metrics.context_tokens)
        .min()
        .unwrap();
    let max_c = plans
        .iter()
        .map(|p| p.metrics.context_tokens)
        .max()
        .unwrap();
    let span = |min: u64, max: u64| max.saturating_sub(min).max(1);
    let balanced = (0..plans.len()).min_by_key(|&index| {
        let m = plans[index].metrics;
        let q =
            (m.quality_loss.saturating_sub(min_q)).saturating_mul(1_000_000) / span(min_q, max_q);
        let l =
            (m.latency_cost.saturating_sub(min_l)).saturating_mul(1_000_000) / span(min_l, max_l);
        let s =
            (m.package_bytes.saturating_sub(min_s)).saturating_mul(1_000_000) / span(min_s, max_s);
        let c =
            (max_c.saturating_sub(m.context_tokens)).saturating_mul(1_000_000) / span(min_c, max_c);
        (
            q.saturating_add(l).saturating_add(s).saturating_add(c),
            m.quality_loss,
            m.latency_cost,
        )
    });

    let mut picked = Vec::new();
    for (index, label) in [
        (quality, "quality"),
        (balanced, "balanced"),
        (long_context, "long-context"),
        (latency, "latency"),
    ] {
        if let Some(index) = index {
            if picked.iter().all(|(existing, _)| *existing != index) {
                picked.push((index, label));
            }
        }
    }
    picked
}

pub fn optimize(input: &OptimizeInput) -> Result<Vec<ParetoPlan>, String> {
    if input.groups.iter().any(|group| group.options.is_empty()) {
        return Err("optimizer decision group has no executable options".into());
    }
    let mut states = vec![State {
        quality_loss: 0,
        latency_cost: 0,
        resident_bytes: 0,
        package_bytes: 0,
        switches: 0,
        last_representation: None,
        decisions: Vec::new(),
    }];
    for group in &input.groups {
        let mut next = Vec::new();
        for state in &states {
            for option in &group.options {
                if let Some(state) = state.extend(group, option, input.heterogeneity_penalty) {
                    next.push(state);
                }
            }
        }
        states = prune_states(next);
        if states.is_empty() {
            return Err(format!(
                "optimizer overflow while planning group `{}`",
                group.id
            ));
        }
    }

    let mut frontier = Vec::new();
    for context in input
        .contexts
        .iter()
        .copied()
        .filter(|context| input.context_constraint.allows(context.tokens))
    {
        for state in &states {
            let Some(resident_bytes) = input
                .base_resident_bytes
                .checked_add(context.state_bytes)
                .and_then(|value| value.checked_add(state.resident_bytes))
            else {
                continue;
            };
            if resident_bytes > input.memory_budget_bytes {
                continue;
            }
            let Some(package_bytes) = input.base_package_bytes.checked_add(state.package_bytes)
            else {
                continue;
            };
            let metrics = PlanMetrics {
                quality_loss: state.quality_loss,
                latency_cost: state.latency_cost,
                resident_bytes,
                package_bytes,
                context_tokens: context.tokens,
                representation_switches: state.switches,
            };
            frontier.push(ParetoPlan {
                id: stable_plan_id(metrics, &state.decisions),
                label: None,
                metrics,
                decisions: state.decisions.clone(),
            });
        }
    }
    frontier.sort_by(|left, right| {
        (
            left.metrics.quality_loss,
            left.metrics.latency_cost,
            std::cmp::Reverse(left.metrics.context_tokens),
            left.metrics.resident_bytes,
            left.metrics.package_bytes,
            &left.id,
        )
            .cmp(&(
                right.metrics.quality_loss,
                right.metrics.latency_cost,
                std::cmp::Reverse(right.metrics.context_tokens),
                right.metrics.resident_bytes,
                right.metrics.package_bytes,
                &right.id,
            ))
    });
    frontier
        .dedup_by(|left, right| left.metrics == right.metrics && left.decisions == right.decisions);
    let mut non_dominated = Vec::new();
    'candidate: for plan in frontier {
        if non_dominated
            .iter()
            .any(|other| dominates_plan(other, &plan))
        {
            continue 'candidate;
        }
        non_dominated.retain(|other| !dominates_plan(&plan, other));
        non_dominated.push(plan);
    }
    non_dominated.sort_by(|left, right| {
        (
            left.metrics.quality_loss,
            left.metrics.latency_cost,
            std::cmp::Reverse(left.metrics.context_tokens),
            left.metrics.package_bytes,
            &left.id,
        )
            .cmp(&(
                right.metrics.quality_loss,
                right.metrics.latency_cost,
                std::cmp::Reverse(right.metrics.context_tokens),
                right.metrics.package_bytes,
                &right.id,
            ))
    });

    let representatives = representative_indices(&non_dominated);
    for (index, label) in representatives {
        non_dominated[index].label = Some(label.to_string());
    }
    Ok(non_dominated)
}

pub fn select_plan<'a>(plans: &'a [ParetoPlan], selector: &str) -> Option<&'a ParetoPlan> {
    plans.iter().find(|plan| {
        plan.id == selector || plan.label.as_deref().is_some_and(|label| label == selector)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn option(
        id: &str,
        representation: &str,
        quality: u64,
        latency: u64,
        bytes: u64,
    ) -> PhysicalOption {
        PhysicalOption {
            id: id.into(),
            representation: representation.into(),
            layout: "test".into(),
            placement: Placement::Streamed,
            quality_loss: quality,
            latency_cost: latency,
            resident_bytes: 0,
            package_bytes: bytes,
        }
    }

    #[test]
    fn mixed_plan_can_dominate_both_global_choices() {
        let input = OptimizeInput {
            groups: vec![
                DecisionGroup {
                    id: "sensitive".into(),
                    options: vec![
                        option("exact", "bf16", 0, 100, 100),
                        option("small", "mxfp4", 50, 50, 50),
                    ],
                },
                DecisionGroup {
                    id: "tolerant".into(),
                    options: vec![
                        option("exact", "bf16", 0, 100, 100),
                        option("small", "mxfp4", 1, 50, 50),
                    ],
                },
            ],
            contexts: vec![ContextCandidate {
                tokens: 65_536,
                state_bytes: 0,
            }],
            context_constraint: ContextConstraint::required(65_536),
            base_resident_bytes: 0,
            base_package_bytes: 0,
            memory_budget_bytes: u64::MAX,
            heterogeneity_penalty: 5,
        };
        let plans = optimize(&input).unwrap();
        let all_exact = (0, 200, 200);
        let all_small = (51, 100, 100);
        assert!(plans.iter().any(|plan| {
            let m = plan.metrics;
            m.quality_loss == 1
                && m.latency_cost == 155
                && m.package_bytes == 150
                && (m.quality_loss < all_small.0 || m.latency_cost < all_exact.1)
                && m.package_bytes < all_exact.2
        }));
    }

    #[test]
    fn fixed_inputs_produce_fixed_frontier_and_ids() {
        let input = OptimizeInput {
            groups: vec![DecisionGroup {
                id: "layer.0".into(),
                options: vec![
                    option("a", "bf16", 0, 10, 20),
                    option("b", "mxfp4", 2, 5, 10),
                ],
            }],
            contexts: vec![
                ContextCandidate {
                    tokens: 32_768,
                    state_bytes: 100,
                },
                ContextCandidate {
                    tokens: 65_536,
                    state_bytes: 200,
                },
            ],
            context_constraint: ContextConstraint::maximum(65_536),
            base_resident_bytes: 100,
            base_package_bytes: 0,
            memory_budget_bytes: 1_000,
            heterogeneity_penalty: 7,
        };
        assert_eq!(optimize(&input).unwrap(), optimize(&input).unwrap());
    }

    #[test]
    fn required_context_and_memory_are_hard_constraints() {
        let input = OptimizeInput {
            groups: vec![DecisionGroup {
                id: "layer.0".into(),
                options: vec![option("a", "bf16", 0, 10, 20)],
            }],
            contexts: vec![
                ContextCandidate {
                    tokens: 32_768,
                    state_bytes: 100,
                },
                ContextCandidate {
                    tokens: 65_536,
                    state_bytes: 900,
                },
            ],
            context_constraint: ContextConstraint::required(65_536),
            base_resident_bytes: 200,
            base_package_bytes: 0,
            memory_budget_bytes: 1_000,
            heterogeneity_penalty: 0,
        };
        assert!(optimize(&input).unwrap().is_empty());
    }
}
