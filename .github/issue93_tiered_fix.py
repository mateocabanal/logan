from pathlib import Path

p = Path('logan-ir/src/tiered_optimizer.rs')
s = p.read_text()

old = '''    fn no_worse_than(&self, other: &Self) -> bool {
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
'''
new = '''    fn no_worse_than(&self, other: &Self) -> bool {
        // A plan that consumes a pool the other plan does not use is not
        // automatically better just because aggregate bytes are smaller.
        // Compare each pool independently; a missing pool is zero usage.
        for (pool, left) in &self.memory {
            let right = other.memory.get(pool).cloned().unwrap_or_default();
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
        for (pool, left) in &self.storage {
            if *left > other.storage.get(pool).copied().unwrap_or(0) {
                return false;
            }
        }
        self.immutable_package_backing_bytes <= other.immutable_package_backing_bytes
            && self.storage_read_bytes_per_step <= other.storage_read_bytes_per_step
            && self.storage_write_bytes_per_step <= other.storage_write_bytes_per_step
    }
'''
if old not in s:
    raise SystemExit('ResourceAcc no_worse_than anchor missing')
s = s.replace(old, new, 1)

old = '''fn usage_no_worse(left: &TieredResourceUsage, right: &TieredResourceUsage) -> bool {
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
'''
new = '''fn usage_no_worse(left: &TieredResourceUsage, right: &TieredResourceUsage) -> bool {
    for left_pool in &left.memory_pools {
        let right_pool = right.memory_pools.iter().find(|pool| pool.pool == left_pool.pool);
        let (right_min, right_target) = right_pool
            .map(|pool| (pool.minimum_working_set_bytes, pool.target_resident_bytes))
            .unwrap_or((0, 0));
        if left_pool.minimum_working_set_bytes > right_min
            || left_pool.target_resident_bytes > right_target
        {
            return false;
        }
    }
    for left_pool in &left.storage_pools {
        let right_bytes = right
            .storage_pools
            .iter()
            .find(|pool| pool.pool == left_pool.pool)
            .map(|pool| pool.mutable_backing_bytes)
            .unwrap_or(0);
        if left_pool.mutable_backing_bytes > right_bytes {
            return false;
        }
    }
    left.immutable_package_backing_bytes <= right.immutable_package_backing_bytes
        && left.storage_read_bytes_per_step <= right.storage_read_bytes_per_step
        && left.storage_write_bytes_per_step <= right.storage_write_bytes_per_step
}

fn usage_strictly_better(left: &TieredResourceUsage, right: &TieredResourceUsage) -> bool {
    if !usage_no_worse(left, right) {
        return false;
    }
    if left.immutable_package_backing_bytes < right.immutable_package_backing_bytes
        || left.storage_read_bytes_per_step < right.storage_read_bytes_per_step
        || left.storage_write_bytes_per_step < right.storage_write_bytes_per_step
    {
        return true;
    }
    for right_pool in &right.memory_pools {
        let left_pool = left.memory_pools.iter().find(|pool| pool.pool == right_pool.pool);
        let (left_min, left_target) = left_pool
            .map(|pool| (pool.minimum_working_set_bytes, pool.target_resident_bytes))
            .unwrap_or((0, 0));
        if left_min < right_pool.minimum_working_set_bytes
            || left_target < right_pool.target_resident_bytes
        {
            return true;
        }
    }
    right.storage_pools.iter().any(|right_pool| {
        left.storage_pools
            .iter()
            .find(|pool| pool.pool == right_pool.pool)
            .map(|pool| pool.mutable_backing_bytes)
            .unwrap_or(0)
            < right_pool.mutable_backing_bytes
    })
}
'''
if old not in s:
    raise SystemExit('usage dominance anchor missing')
s = s.replace(old, new, 1)

# Add a regression proving pool-local usage survives the frontier.
anchor = '''    #[test]\n    fn discrete_memory_pools_are_capacity_checked_independently() {'''
if anchor not in s:
    raise SystemExit('test insertion anchor missing')
extra = '''    #[test]\n    fn pareto_dominance_does_not_merge_distinct_memory_pools() {\n        let resource_budget = ResourceBudget {\n            memory_pools: vec![\n                MemoryPoolBudget {\n                    id: MemoryPoolId::new("host"),\n                    capacity_bytes: 8 * GIB,\n                },\n                MemoryPoolBudget {\n                    id: MemoryPoolId::new("gpu0"),\n                    capacity_bytes: 8 * GIB,\n                },\n            ],\n            storage_pools: vec![],\n        };\n        let mut input = empty_input(\n            resource_budget,\n            TieredContextCandidate {\n                tokens: 4096,\n                resources: Vec::new(),\n                latency_cost: 0,\n            },\n        );\n        input.groups = vec![TieredCandidateGroup {\n            key: "placement".into(),\n            options: vec![\n                TieredRepresentationCandidate {\n                    id: "host".into(),\n                    quant: QuantSpec { kind: "exact".into(), scale: None },\n                    layout: 0,\n                    legacy_placement: Placement::Resident,\n                    resource: ResourcePlan::resident_only(4 * GIB, MemoryPoolId::new("host")),\n                    package_bytes: 1,\n                    latency_cost: 10,\n                    quality_loss_ppm: 0,\n                    dispatch_class: 1,\n                    rationale: "host".into(),\n                },\n                TieredRepresentationCandidate {\n                    id: "gpu".into(),\n                    quant: QuantSpec { kind: "exact".into(), scale: None },\n                    layout: 0,\n                    legacy_placement: Placement::Gpu,\n                    resource: ResourcePlan::resident_only(4 * GIB, MemoryPoolId::new("gpu0")),\n                    package_bytes: 1,\n                    latency_cost: 10,\n                    quality_loss_ppm: 0,\n                    dispatch_class: 2,\n                    rationale: "gpu".into(),\n                },\n            ],\n        }];\n        let frontier = tiered_pareto_plans(&input).unwrap();\n        assert_eq!(frontier.len(), 2);\n    }\n\n'''
s = s.replace(anchor, extra + anchor, 1)
p.write_text(s)
