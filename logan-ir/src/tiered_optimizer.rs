use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    ContextConstraint, MemoryPoolId, Placement, QuantSpec, ResourceBudget, ResourcePlan,
    StoragePoolId,
};

pub const TIERED_COST_MODEL_V1: &str = "logan-tiered-cost-v1";
const MAX_FRONTIER_STATES: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TieredRepresentationCandidate {
    pub id: String,
    pub quant: QuantSpec,
    pub layout: u16,
    /// Transitional projection only. Capacity is governed by `resource`.
    pub legacy_placement: Placement,
    pub resource: ResourcePlan,
    pub package_bytes: u64,
    pub latency_cost: u64,
    pub quality_loss_ppm: u64,
    pub dispatch_class: u16,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TieredCandidateGroup {
    pub key: String,
    pub options: Vec<TieredRepresentationCandidate>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TieredContextCandidate {
    pub tokens: u64,
    /// Context may contain several independently backed/resident resources,
    /// e.g. fully resident KV layers plus file-backed streamed KV layers.
    pub resources: Vec<ResourcePlan>,
    pub latency_cost: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TieredOptimizerInput {
    pub cost_model: String,
    pub groups: Vec<TieredCandidateGroup>,
    pub context_constraint: ContextConstraint,
    pub context_candidates: Vec<TieredContextCandidate>,
    pub resource_budget: ResourceBudget,
    /// Always-present resources such as small resident state, fixed staging,
    /// or immutable package-backed tensors outside representation groups.
    pub base_resources: Vec<ResourcePlan>,
    pub base_package_bytes: u64,
    pub base_latency_cost: u64,
    pub base_quality_loss_ppm: u64,
    pub heterogeneity_switch_penalty: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPoolUsage {
    pub pool: MemoryPoolId,
    /// Pinned bytes plus the largest transient requirement in this pool.
    pub minimum_working_set_bytes: u64,
    pub target_resident_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoragePoolUsage {
    pub pool: StoragePoolId,
    pub mutable_backing_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TieredResourceUsage {
    pub memory_pools: Vec<MemoryPoolUsage>,
    pub storage_pools: Vec<StoragePoolUsage>,
    pub immutable_package_backing_bytes: u64,
    pub storage_read_bytes_per_step: u64,
    pub storage_write_bytes_per_step: u64,
}

impl TieredResourceUsage {
    pub fn minimum_working_set_bytes(&self) -> u64 {
        self.memory_pools
            .iter()
            .fold(0_u64, |total, pool| total.saturating_add(pool.minimum_working_set_bytes))
    }

    pub fn target_resident_bytes(&self) -> u64 {
        self.memory_pools
            .iter()
            .fold(0_u64, |total, pool| total.saturating_add(pool.target_resident_bytes))
    }

    pub fn mutable_backing_bytes(&self) -> u64 {
        self.storage_pools
            .iter()
            .fold(0_u64, |total, pool| total.saturating_add(pool.mutable_backing_bytes))
    }

    pub fn storage_traffic_bytes_per_step(&self) -> u64 {
        self.storage_read_bytes_per_step
            .saturating_add(self.storage_write_bytes_per_step)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TieredPlanMetrics {
    pub quality_loss_ppm: u64,
    pub context_tokens: u64,
    pub latency_cost: u64,
    pub resource_usage: TieredResourceUsage,
    pub package_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TieredRejectedCandidate {
    pub id: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TieredPlanDecision {
    pub group: String,
    pub chosen: TieredRepresentationCandidate,
    pub rejected: Vec<TieredRejectedCandidate>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TieredParetoPlan {
    pub id: String,
    pub labels: Vec<String>,
    pub cost_model: String,
    pub metrics: TieredPlanMetrics,
    pub decisions: Vec<TieredPlanDecision>,
}

#[derive(Debug, Clone, Default)]
struct MemoryAcc {
    pinned_bytes: u64,
    transient_working_set_bytes: u64,
    target_resident_bytes: u64,
}

impl MemoryAcc {
    fn minimum_working_set_bytes(&self) -> Option<u64> {
        self.pinned_bytes
            .checked_add(self.transient_working_set_bytes)
    }
}

#[derive(Debug, Clone, Default)]
struct ResourceAcc {
    memory: BTreeMap<MemoryPoolId, MemoryAcc>,
    storage: BTreeMap<StoragePoolId, u64>,
    immutable_package_backing_bytes: u64,
    storage_read_bytes_per_step: u64,
    storage_write_bytes_per_step: u64,
}

impl ResourceAcc {
    fn add_resource(&mut self, budget: &ResourceBudget, resource: &ResourcePlan) -> Result<bool, String> {
        resource.validate()?;
        let memory_capacity = budget
            .memory_capacity(&resource.residency.memory_pool)
            .ok_or_else(|| {
                format!(
                    "resource references unknown memory pool `{}`",
                    resource.residency.memory_pool.0
                )
            })?;
        let entry = self
            .memory
            .entry(resource.residency.memory_pool.clone())
            .or_default();
        entry.pinned_bytes = entry
            .pinned_bytes
            .checked_add(resource.residency.pinned_bytes)
            .ok_or_else(|| "pinned memory accounting overflow".to_owned())?;
        let transient = resource
            .residency
            .minimum_working_set_bytes
            .checked_sub(resource.residency.pinned_bytes)
            .ok_or_else(|| "resource pinned bytes exceed local working set".to_owned())?;
        entry.transient_working_set_bytes = entry.transient_working_set_bytes.max(transient);
        entry.target_resident_bytes = entry
            .target_resident_bytes
            .checked_add(resource.residency.target_resident_bytes)
            .ok_or_else(|| "resident target accounting overflow".to_owned())?;
        let minimum = entry
            .minimum_working_set_bytes()
            .ok_or_else(|| "minimum working-set accounting overflow".to_owned())?;
        if minimum > memory_capacity || entry.target_resident_bytes > memory_capacity {
            return Ok(false);
        }

        match resource.backing.kind {
            crate::BackingKind::RuntimeStateFile | crate::BackingKind::DevicePersistent => {
                let pool_id = resource.backing.storage_pool.as_ref().ok_or_else(|| {
                    "runtime/device backing is missing a storage pool after validation".to_owned()
                })?;
                let pool = budget.storage_pool(pool_id).ok_or_else(|| {
                    format!("resource references unknown storage pool `{}`", pool_id.0)
                })?;
                if !pool.writable {
                    return Ok(false);
                }
                let stored = self.storage.entry(pool_id.clone()).or_default();
                *stored = stored
                    .checked_add(resource.backing.bytes)
                    .ok_or_else(|| "mutable backing accounting overflow".to_owned())?;
                if *stored > pool.available_bytes {
                    return Ok(false);
                }
            }
            crate::BackingKind::PackageRecord => {
                self.immutable_package_backing_bytes = self
                    .immutable_package_backing_bytes
                    .checked_add(resource.backing.bytes)
                    .ok_or_else(|| "package backing accounting overflow".to_owned())?;
            }
            crate::BackingKind::ResidentOnly => {}
        }
        self.storage_read_bytes_per_step = self
            .storage_read_bytes_per_step
            .checked_add(resource.access.expected_read_bytes_per_step)
            .ok_or_else(|| "storage read accounting overflow".to_owned())?;
        self.storage_write_bytes_per_step = self
            .storage_write_bytes_per_step
            .checked_add(resource.access.expected_write_bytes_per_step)
            .ok_or_else(|| "storage write accounting overflow".to_owned())?;
        Ok(true)
    }

    fn add_resources(&mut self, budget: &ResourceBudget, resources: &[ResourcePlan]) -> Result<bool, String> {
        for resource in resources {
            if !self.add_resource(budget, resource)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn usage(&self) -> Result<TieredResourceUsage, String> {
        let memory_pools = self
            .memory
            .iter()
            .map(|(pool, usage)| {
                Ok(MemoryPoolUsage {
                    pool: pool.clone(),
                    minimum_working_set_bytes: usage
                        .minimum_working_set_bytes()
                        .ok_or_else(|| "minimum working-set accounting overflow".to_owned())?,
                    target_resident_bytes: usage.target_resident_bytes,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let storage_pools = self
            .storage
            .iter()
            .map(|(pool, bytes)| StoragePoolUsage {
                pool: pool.clone(),
                mutable_backing_bytes: *bytes,
            })
            .collect();
        Ok(TieredResourceUsage {
            memory_pools,
            storage_pools,
            immutable_package_backing_bytes: self.immutable_package_backing_bytes,
            storage_read_bytes_per_step: self.storage_read_bytes_per_step,
            storage_write_bytes_per_step: self.storage_write_bytes_per_step,
        })
    }

    fn no_worse_than(&self, other: &Self) -> bool {
        for (pool, right) in &other.memory {
            let left = self.memory.get(pool).cloned().unwrap_or_default();
            let Some(left_min) = left.minimum_working_set_bytes() else {
                return false;
            };
            let Some(right_min) = right.minimum_working_set_bytes() else {
                return false;
            };
            if left_min > right_min || left.target_resident_bytes > right.target_resident_bytes {
                return false;
            }
        }
        for (pool, right) in &other.storage {
            if self.storage.get(pool).copied().unwrap_or(0) > *right {
                return false;
            }
        }
        self.immutable_package_backing_bytes <= other.immutable_package_backing_bytes
            && self.storage_read_bytes_per_step <= other.storage_read_bytes_per_step
            && self.storage_write_bytes_per_step <= other.storage_write_bytes_per_step
    }

    fn strictly_better_than(&self, other: &Self) -> bool {
        if !self.no_worse_than(other) {
            return false;
        }
        if self.immutable_package_backing_bytes < other.immutable_package_backing_bytes
            || self.storage_read_bytes_per_step < other.storage_read_bytes_per_step
            || self.storage_write_bytes_per_step < other.storage_write_bytes_per_step
        {
            return true;
        }
        for (pool, right) in &other.memory {
            let left = self.memory.get(pool).cloned().unwrap_or_default();
            if left.minimum_working_set_bytes() < right.minimum_working_set_bytes()
                || left.target_resident_bytes < right.target_resident_bytes
            {
                return true;
            }
        }
        other
            .storage
            .iter()
            .any(|(pool, right)| self.storage.get(pool).copied().unwrap_or(0) < *right)
    }
}

#[derive(Debug, Clone)]
struct State {
    quality_loss_ppm: u64,
    latency_cost: u64,
    package_bytes: u64,
    resources: ResourceAcc,
    last_dispatch_class: u16,
    choices: Vec<usize>,
}

pub fn tiered_pareto_plans(input: &TieredOptimizerInput) -> Result<Vec<TieredParetoPlan>, String> {
    validate_input(input)?;
    let mut base = ResourceAcc::default();
    if !base.add_resources(&input.resource_budget, &input.base_resources)? {
        return Err("base resources exceed a hard memory working-set/resident target or backing-store capacity".into());
    }
    let mut states = vec![State {
        quality_loss_ppm: input.base_quality_loss_ppm,
        latency_cost: input.base_latency_cost,
        package_bytes: input.base_package_bytes,
        resources: base,
        last_dispatch_class: 0,
        choices: Vec::with_capacity(input.groups.len()),
    }];

    for group in &input.groups {
        let mut next = Vec::new();
        for state in &states {
            for (option_index, option) in group.options.iter().enumerate() {
                let mut resources = state.resources.clone();
                if !resources.add_resource(&input.resource_budget, &option.resource)? {
                    continue;
                }
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
                let mut choices = state.choices.clone();
                choices.push(option_index);
                next.push(State {
                    quality_loss_ppm: state
                        .quality_loss_ppm
                        .checked_add(option.quality_loss_ppm)
                        .ok_or_else(|| "optimizer quality cost overflow".to_owned())?,
                    latency_cost,
                    package_bytes: state
                        .package_bytes
                        .checked_add(option.package_bytes)
                        .ok_or_else(|| "optimizer package bytes overflow".to_owned())?,
                    resources,
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
                "no feasible physical representation remains after group `{}` within the memory/storage resource envelope",
                group.key
            ));
        }
    }

    let mut raw = Vec::<(TieredPlanMetrics, Vec<usize>)>::new();
    for context in input
        .context_candidates
        .iter()
        .filter(|candidate| input.context_constraint.allows(candidate.tokens))
    {
        for state in &states {
            let mut resources = state.resources.clone();
            if !resources.add_resources(&input.resource_budget, &context.resources)? {
                continue;
            }
            raw.push((
                TieredPlanMetrics {
                    quality_loss_ppm: state.quality_loss_ppm,
                    context_tokens: context.tokens,
                    latency_cost: state
                        .latency_cost
                        .checked_add(context.latency_cost)
                        .ok_or_else(|| "optimizer context latency overflow".to_owned())?,
                    resource_usage: resources.usage()?,
                    package_bytes: state.package_bytes,
                },
                state.choices.clone(),
            ));
        }
    }
    if raw.is_empty() {
        return Err("no optimizer plan satisfies the context and tiered resource constraints".into());
    }

    let keep = (0..raw.len())
        .filter(|&candidate| {
            !(0..raw.len()).any(|other| {
                other != candidate && metrics_dominate(&raw[other].0, &raw[candidate].0)
            })
        })
        .collect::<Vec<_>>();
    let mut plans = keep
        .into_iter()
        .map(|index| {
            let (metrics, choices) = &raw[index];
            build_plan(input, metrics.clone(), choices)
        })
        .collect::<Vec<_>>();
    plans.sort_by(|left, right| plan_sort_key(left).cmp(&plan_sort_key(right)));
    plans.dedup_by(|left, right| left.id == right.id);
    Ok(plans)
}

pub fn tiered_material_plans(input: &TieredOptimizerInput) -> Result<Vec<TieredParetoPlan>, String> {
    let frontier = tiered_pareto_plans(input)?;
    if frontier.is_empty() {
        return Ok(Vec::new());
    }
    let quality = best_index(&frontier, |plan| {
        (
            plan.metrics.quality_loss_ppm,
            u64::MAX - plan.metrics.context_tokens,
            plan.metrics.latency_cost,
            plan.metrics.resource_usage.target_resident_bytes(),
            plan.metrics.resource_usage.storage_traffic_bytes_per_step(),
            plan.id.clone(),
        )
    });
    let latency = best_index(&frontier, |plan| {
        (
            plan.metrics.latency_cost,
            plan.metrics.resource_usage.storage_traffic_bytes_per_step(),
            plan.metrics.quality_loss_ppm,
            u64::MAX - plan.metrics.context_tokens,
            plan.metrics.resource_usage.target_resident_bytes(),
            plan.id.clone(),
        )
    });
    let long_context = best_index(&frontier, |plan| {
        (
            u64::MAX - plan.metrics.context_tokens,
            plan.metrics.quality_loss_ppm,
            plan.metrics.latency_cost,
            plan.metrics.resource_usage.storage_traffic_bytes_per_step(),
            plan.metrics.resource_usage.target_resident_bytes(),
            plan.id.clone(),
        )
    });
    let balanced = balanced_index(&frontier);

    let mut chosen = Vec::<TieredParetoPlan>::new();
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

pub fn select_tiered_plan<'a>(plans: &'a [TieredParetoPlan], selector: &str) -> Option<&'a TieredParetoPlan> {
    plans
        .iter()
        .find(|plan| plan.id == selector || plan.labels.iter().any(|label| label == selector))
}

fn validate_input(input: &TieredOptimizerInput) -> Result<(), String> {
    if input.cost_model.trim().is_empty() {
        return Err("optimizer cost model id cannot be empty".into());
    }
    input.resource_budget.validate()?;
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
        for option in &group.options {
            option.resource.validate()?;
        }
    }
    for context in &input.context_candidates {
        for resource in &context.resources {
            resource.validate()?;
        }
    }
    Ok(())
}

fn prune_states(mut states: Vec<State>) -> Vec<State> {
    states.sort_by(|left, right| state_sort_key(left).cmp(&state_sort_key(right)));
    let mut kept = Vec::<State>::new();
    'candidate: for state in states {
        for existing in &kept {
            if existing.last_dispatch_class == state.last_dispatch_class && state_dominates(existing, &state) {
                continue 'candidate;
            }
        }
        kept.retain(|existing| {
            existing.last_dispatch_class != state.last_dispatch_class || !state_dominates(&state, existing)
        });
        kept.push(state);
    }
    if kept.len() > MAX_FRONTIER_STATES {
        kept.sort_by(|left, right| state_sort_key(left).cmp(&state_sort_key(right)));
        kept.truncate(MAX_FRONTIER_STATES);
    }
    kept
}

fn state_dominates(left: &State, right: &State) -> bool {
    let resource_no_worse = left.resources.no_worse_than(&right.resources);
    let no_worse = left.quality_loss_ppm <= right.quality_loss_ppm
        && left.latency_cost <= right.latency_cost
        && left.package_bytes <= right.package_bytes
        && resource_no_worse;
    let strictly_better = left.quality_loss_ppm < right.quality_loss_ppm
        || left.latency_cost < right.latency_cost
        || left.package_bytes < right.package_bytes
        || left.resources.strictly_better_than(&right.resources);
    no_worse && strictly_better
}

fn metrics_dominate(left: &TieredPlanMetrics, right: &TieredPlanMetrics) -> bool {
    let resource_no_worse = usage_no_worse(&left.resource_usage, &right.resource_usage);
    let no_worse = left.quality_loss_ppm <= right.quality_loss_ppm
        && left.context_tokens >= right.context_tokens
        && left.latency_cost <= right.latency_cost
        && left.package_bytes <= right.package_bytes
        && resource_no_worse;
    let strictly_better = left.quality_loss_ppm < right.quality_loss_ppm
        || left.context_tokens > right.context_tokens
        || left.latency_cost < right.latency_cost
        || left.package_bytes < right.package_bytes
        || usage_strictly_better(&left.resource_usage, &right.resource_usage);
    no_worse && strictly_better
}

fn usage_no_worse(left: &TieredResourceUsage, right: &TieredResourceUsage) -> bool {
    left.minimum_working_set_bytes() <= right.minimum_working_set_bytes()
        && left.target_resident_bytes() <= right.target_resident_bytes()
        && left.mutable_backing_bytes() <= right.mutable_backing_bytes()
        && left.immutable_package_backing_bytes <= right.immutable_package_backing_bytes
        && left.storage_read_bytes_per_step <= right.storage_read_bytes_per_step
        && left.storage_write_bytes_per_step <= right.storage_write_bytes_per_step
}

fn usage_strictly_better(left: &TieredResourceUsage, right: &TieredResourceUsage) -> bool {
    usage_no_worse(left, right)
        && (left.minimum_working_set_bytes() < right.minimum_working_set_bytes()
            || left.target_resident_bytes() < right.target_resident_bytes()
            || left.mutable_backing_bytes() < right.mutable_backing_bytes()
            || left.immutable_package_backing_bytes < right.immutable_package_backing_bytes
            || left.storage_read_bytes_per_step < right.storage_read_bytes_per_step
            || left.storage_write_bytes_per_step < right.storage_write_bytes_per_step)
}

fn build_plan(input: &TieredOptimizerInput, metrics: TieredPlanMetrics, choices: &[usize]) -> TieredParetoPlan {
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
                .map(|(_, option)| TieredRejectedCandidate {
                    id: option.id.clone(),
                    reason: format!(
                        "candidate q={}ppm latency={} resident_target={} min_working={} backing={} read/step={} write/step={}: {}",
                        option.quality_loss_ppm,
                        option.latency_cost,
                        option.resource.residency.target_resident_bytes,
                        option.resource.residency.minimum_working_set_bytes,
                        option.resource.mutable_backing_bytes(),
                        option.resource.access.expected_read_bytes_per_step,
                        option.resource.access.expected_write_bytes_per_step,
                        option.rationale
                    ),
                })
                .collect();
            TieredPlanDecision {
                group: group.key.clone(),
                chosen,
                rejected,
            }
        })
        .collect::<Vec<_>>();
    let id = stable_plan_id(&input.cost_model, &metrics, &decisions);
    TieredParetoPlan {
        id,
        labels: Vec::new(),
        cost_model: input.cost_model.clone(),
        metrics,
        decisions,
    }
}

fn stable_plan_id(cost_model: &str, metrics: &TieredPlanMetrics, decisions: &[TieredPlanDecision]) -> String {
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
        metrics.resource_usage.minimum_working_set_bytes(),
        metrics.resource_usage.target_resident_bytes(),
        metrics.resource_usage.mutable_backing_bytes(),
        metrics.resource_usage.immutable_package_backing_bytes,
        metrics.resource_usage.storage_read_bytes_per_step,
        metrics.resource_usage.storage_write_bytes_per_step,
        metrics.package_bytes,
    ] {
        feed(&mut hash, &value.to_le_bytes());
    }
    for pool in &metrics.resource_usage.memory_pools {
        feed(&mut hash, pool.pool.0.as_bytes());
        feed(&mut hash, &pool.minimum_working_set_bytes.to_le_bytes());
        feed(&mut hash, &pool.target_resident_bytes.to_le_bytes());
    }
    for pool in &metrics.resource_usage.storage_pools {
        feed(&mut hash, pool.pool.0.as_bytes());
        feed(&mut hash, &pool.mutable_backing_bytes.to_le_bytes());
    }
    for decision in decisions {
        feed(&mut hash, decision.group.as_bytes());
        feed(&mut hash, decision.chosen.id.as_bytes());
    }
    format!("tp-{hash:016x}")
}

fn best_index<K: Ord>(plans: &[TieredParetoPlan], key: impl Fn(&TieredParetoPlan) -> K) -> usize {
    plans
        .iter()
        .enumerate()
        .min_by_key(|(_, plan)| key(plan))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn balanced_index(plans: &[TieredParetoPlan]) -> usize {
    let quality = ranks(plans, |plan| plan.metrics.quality_loss_ppm, false);
    let context = ranks(plans, |plan| plan.metrics.context_tokens, true);
    let latency = ranks(plans, |plan| plan.metrics.latency_cost, false);
    let resident = ranks(plans, |plan| plan.metrics.resource_usage.target_resident_bytes(), false);
    let traffic = ranks(
        plans,
        |plan| plan.metrics.resource_usage.storage_traffic_bytes_per_step(),
        false,
    );
    let backing = ranks(plans, |plan| plan.metrics.resource_usage.mutable_backing_bytes(), false);
    (0..plans.len())
        .min_by_key(|&index| {
            let dimensions = [
                quality[index],
                context[index],
                latency[index],
                resident[index],
                traffic[index],
                backing[index],
            ];
            (
                *dimensions.iter().max().unwrap_or(&0),
                dimensions.iter().sum::<usize>(),
                plans[index].id.clone(),
            )
        })
        .unwrap_or(0)
}

fn ranks(plans: &[TieredParetoPlan], value: impl Fn(&TieredParetoPlan) -> u64, reverse: bool) -> Vec<usize> {
    let mut order = (0..plans.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| {
        let value = value(&plans[index]);
        (if reverse { u64::MAX - value } else { value }, plans[index].id.clone())
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

fn state_sort_key(state: &State) -> (u64, u64, u64, u64, u64, u16, Vec<usize>) {
    let usage = state.resources.usage().ok();
    (
        state.quality_loss_ppm,
        state.latency_cost,
        usage.as_ref().map_or(u64::MAX, TieredResourceUsage::target_resident_bytes),
        usage.as_ref().map_or(u64::MAX, TieredResourceUsage::mutable_backing_bytes),
        state.package_bytes,
        state.last_dispatch_class,
        state.choices.clone(),
    )
}

fn plan_sort_key(plan: &TieredParetoPlan) -> (u64, u64, u64, u64, u64, u64, String) {
    (
        plan.metrics.quality_loss_ppm,
        u64::MAX - plan.metrics.context_tokens,
        plan.metrics.latency_cost,
        plan.metrics.resource_usage.target_resident_bytes(),
        plan.metrics.resource_usage.mutable_backing_bytes(),
        plan.metrics.resource_usage.storage_traffic_bytes_per_step(),
        plan.id.clone(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryPoolBudget, StoragePoolBudget};

    const GIB: u64 = 1024 * 1024 * 1024;

    fn budget(ram_gib: u64, storage_gib: u64) -> ResourceBudget {
        ResourceBudget {
            memory_pools: vec![MemoryPoolBudget {
                id: MemoryPoolId::new("uma0"),
                capacity_bytes: ram_gib * GIB,
            }],
            storage_pools: vec![StoragePoolBudget {
                id: StoragePoolId::new("ssd0"),
                available_bytes: storage_gib * GIB,
                writable: true,
            }],
        }
    }

    fn empty_input(resource_budget: ResourceBudget, context: TieredContextCandidate) -> TieredOptimizerInput {
        TieredOptimizerInput {
            cost_model: TIERED_COST_MODEL_V1.into(),
            groups: Vec::new(),
            context_constraint: ContextConstraint::required(context.tokens),
            context_candidates: vec![context],
            resource_budget,
            base_resources: Vec::new(),
            base_package_bytes: 0,
            base_latency_cost: 0,
            base_quality_loss_ppm: 0,
            heterogeneity_switch_penalty: 0,
        }
    }

    #[test]
    fn ram_overcommitted_logical_state_is_feasible_when_backing_exists() {
        let context = TieredContextCandidate {
            tokens: 131_072,
            resources: vec![
                ResourcePlan::resident_only(12 * GIB, MemoryPoolId::new("uma0")),
                ResourcePlan::mutable_file_backed(
                    9 * GIB,
                    StoragePoolId::new("ssd0"),
                    MemoryPoolId::new("uma0"),
                    GIB / 2,
                    GIB / 2,
                    4 * 1024 * 1024,
                    9 * GIB,
                    64 * 1024,
                ),
            ],
            latency_cost: 100,
        };
        let plans = tiered_pareto_plans(&empty_input(budget(16, 10), context)).unwrap();
        assert_eq!(plans.len(), 1);
        let usage = &plans[0].metrics.resource_usage;
        assert_eq!(usage.mutable_backing_bytes(), 9 * GIB);
        assert_eq!(usage.target_resident_bytes(), 12 * GIB + GIB / 2);
        assert!(21 * GIB > 16 * GIB);
        assert!(usage.target_resident_bytes() < 16 * GIB);
    }

    #[test]
    fn genuine_backing_exhaustion_is_a_hard_failure() {
        let context = TieredContextCandidate {
            tokens: 131_072,
            resources: vec![ResourcePlan::mutable_file_backed(
                9 * GIB,
                StoragePoolId::new("ssd0"),
                MemoryPoolId::new("uma0"),
                GIB / 2,
                GIB / 2,
                4 * 1024 * 1024,
                9 * GIB,
                64 * 1024,
            )],
            latency_cost: 100,
        };
        let error = tiered_pareto_plans(&empty_input(budget(16, 8), context)).unwrap_err();
        assert!(error.contains("tiered resource constraints"));
    }

    #[test]
    fn immutable_hundred_gib_package_does_not_need_duplicate_spill_capacity() {
        let context = TieredContextCandidate {
            tokens: 4096,
            resources: vec![ResourcePlan::immutable_package(
                100 * GIB,
                MemoryPoolId::new("uma0"),
                GIB / 2,
                GIB,
                2 * GIB,
            )],
            latency_cost: 0,
        };
        let plans = tiered_pareto_plans(&empty_input(budget(16, 0), context)).unwrap();
        let usage = &plans[0].metrics.resource_usage;
        assert_eq!(usage.immutable_package_backing_bytes, 100 * GIB);
        assert_eq!(usage.mutable_backing_bytes(), 0);
        assert_eq!(usage.target_resident_bytes(), GIB);
    }

    #[test]
    fn irreducible_working_set_still_has_to_fit_ram() {
        let context = TieredContextCandidate {
            tokens: 4096,
            resources: vec![ResourcePlan::resident_only(18 * GIB, MemoryPoolId::new("uma0"))],
            latency_cost: 0,
        };
        assert!(tiered_pareto_plans(&empty_input(budget(16, 100), context)).is_err());
    }

    #[test]
    fn sequential_transient_working_sets_use_max_not_sum() {
        let mut input = empty_input(
            budget(8, 100),
            TieredContextCandidate {
                tokens: 4096,
                resources: Vec::new(),
                latency_cost: 0,
            },
        );
        input.groups = vec![
            TieredCandidateGroup {
                key: "layer0".into(),
                options: vec![TieredRepresentationCandidate {
                    id: "stream0".into(),
                    quant: QuantSpec { kind: "exact".into(), scale: None },
                    layout: 0,
                    legacy_placement: Placement::Streamed,
                    resource: ResourcePlan::immutable_package(
                        40 * GIB,
                        MemoryPoolId::new("uma0"),
                        4 * GIB,
                        4 * GIB,
                        4 * GIB,
                    ),
                    package_bytes: 40 * GIB,
                    latency_cost: 10,
                    quality_loss_ppm: 0,
                    dispatch_class: 1,
                    rationale: "layer0".into(),
                }],
            },
            TieredCandidateGroup {
                key: "layer1".into(),
                options: vec![TieredRepresentationCandidate {
                    id: "stream1".into(),
                    quant: QuantSpec { kind: "exact".into(), scale: None },
                    layout: 0,
                    legacy_placement: Placement::Streamed,
                    resource: ResourcePlan::immutable_package(
                        40 * GIB,
                        MemoryPoolId::new("uma0"),
                        4 * GIB,
                        4 * GIB,
                        4 * GIB,
                    ),
                    package_bytes: 40 * GIB,
                    latency_cost: 10,
                    quality_loss_ppm: 0,
                    dispatch_class: 1,
                    rationale: "layer1".into(),
                }],
            },
        ];
        let plans = tiered_pareto_plans(&input).unwrap();
        let usage = &plans[0].metrics.resource_usage;
        assert_eq!(usage.minimum_working_set_bytes(), 4 * GIB);
        assert_eq!(usage.target_resident_bytes(), 8 * GIB);
    }

    #[test]
    fn discrete_memory_pools_are_capacity_checked_independently() {
        let resource_budget = ResourceBudget {
            memory_pools: vec![
                MemoryPoolBudget {
                    id: MemoryPoolId::new("host"),
                    capacity_bytes: 8 * GIB,
                },
                MemoryPoolBudget {
                    id: MemoryPoolId::new("gpu0"),
                    capacity_bytes: 8 * GIB,
                },
            ],
            storage_pools: vec![],
        };
        let context = TieredContextCandidate {
            tokens: 4096,
            resources: vec![
                ResourcePlan::resident_only(6 * GIB, MemoryPoolId::new("host")),
                ResourcePlan::resident_only(6 * GIB, MemoryPoolId::new("gpu0")),
            ],
            latency_cost: 0,
        };
        let plans = tiered_pareto_plans(&empty_input(resource_budget, context)).unwrap();
        assert_eq!(plans[0].metrics.resource_usage.target_resident_bytes(), 12 * GIB);
    }
}
