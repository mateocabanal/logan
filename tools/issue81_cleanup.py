from pathlib import Path
import subprocess

# rustfmt followed module declarations from crate roots during the integration
# gate. Restore every file that was not semantically part of #81.
restore = [
    "logan-compiler/src/target/machine.rs",
    "logan-compiler/src/target/mod.rs",
    "logan-compiler/src/target/tests.rs",
    "logan-qwen4/src/colisource.rs",
    "logan-qwen4/src/plan.rs",
    "logan-qwen4/src/plan/prefix_cache.rs",
    "logan-qwen4/src/plan/runtime_stats.rs",
    "logan-qwen4/src/plan/snapshot.rs",
    "logan-qwen4/src/scheduled.rs",
]
subprocess.run(["git", "checkout", "origin/main", "--", *restore], check=True)

p = Path("logan-ir/src/optimizer.rs")
s = p.read_text()
old = '''fn balanced_index(plans: &[ParetoPlan]) -> usize {
    let quality = ranks(plans, |plan| plan.metrics.quality_loss_ppm, false);
    let context = ranks(plans, |plan| plan.metrics.context_tokens, true);
    let latency = ranks(plans, |plan| plan.metrics.latency_cost, false);
    let resident = ranks(plans, |plan| plan.metrics.resident_bytes, false);
    let traffic = ranks(plans, |plan| plan.metrics.storage_traffic_bytes, false);
    (0..plans.len())
        .min_by_key(|&index| {
            (
                quality[index] + context[index] + latency[index] + resident[index] + traffic[index],
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
    for (rank, index) in order.into_iter().enumerate() {
        ranks[index] = rank;
    }
    ranks
}
'''
new = '''fn balanced_index(plans: &[ParetoPlan]) -> usize {
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
'''
if old not in s:
    raise SystemExit("balanced selector anchor did not match")
s = s.replace(old, new, 1)

anchor = '''    #[test]
    fn switch_penalty_is_applied_between_dispatch_classes() {'''
test = '''    #[test]
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
    fn switch_penalty_is_applied_between_dispatch_classes() {'''
if anchor not in s:
    raise SystemExit("optimizer test anchor did not match")
s = s.replace(anchor, test, 1)
p.write_text(s)
print("issue81 cleanup applied")
