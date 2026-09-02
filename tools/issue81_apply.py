from pathlib import Path


def replace(path, old, new, count=1):
    p = Path(path)
    s = p.read_text()
    if old not in s:
        raise SystemExit(f"missing anchor in {path}: {old[:120]!r}")
    if count is not None and s.count(old) != count:
        raise SystemExit(f"unexpected anchor count in {path}: {s.count(old)} for {old[:80]!r}")
    p.write_text(s.replace(old, new))

# Export compiler context planner.
replace(
    "logan-compiler/src/lib.rs",
    "pub mod codec;\npub mod error;",
    "pub mod codec;\npub mod context_plan;\npub mod error;",
)

# Fix validation in the new planner and add plan enrichment.
replace(
    "logan-compiler/src/context_plan.rs",
    "use logan_ir::{\n    ContextCandidate, ContextConstraint, ContextConstraintKind, ContextPlan, ContextStateBytes,\n    PlannerMemoryBudget,\n};",
    "use logan_ir::{\n    ContextCandidate, ContextConstraint, ContextConstraintKind, ContextPlan, ContextStateBytes,\n    ParetoPlan, PlannerMemoryBudget,\n};",
)
replace(
    "logan-compiler/src/context_plan.rs",
    "    constraint\n        .valid_for_model(geometry.model_max_tokens)\n        .map_err(ColicError::Usage)?;",
    "    if constraint.tokens == 0 {\n        return Err(ColicError::Usage(\"context must be greater than zero\".into()));\n    }\n    if constraint.tokens > geometry.model_max_tokens {\n        return Err(ColicError::Usage(format!(\n            \"requested context {} exceeds model ceiling {}\",\n            constraint.tokens, geometry.model_max_tokens\n        )));\n    }",
)
replace(
    "logan-compiler/src/context_plan.rs",
    "    pub fn point_for_tokens(&self, tokens: u64) -> Option<&PlannedContextPoint> {\n        self.points.iter().find(|point| point.candidate.tokens == tokens)\n    }\n}",
    "    pub fn point_for_tokens(&self, tokens: u64) -> Option<&PlannedContextPoint> {\n        self.points.iter().find(|point| point.candidate.tokens == tokens)\n    }\n\n    pub fn enrich_plans(&self, plans: &mut [ParetoPlan]) -> Result<()> {\n        for plan in plans {\n            let point = self.point_for_tokens(plan.metrics.context_tokens).ok_or_else(|| {\n                ColicError::Usage(format!(\n                    \"optimizer returned unplanned context point {}\",\n                    plan.metrics.context_tokens\n                ))\n            })?;\n            plan.context_plan = Some(point.plan.clone());\n            plan.memory_budget = Some(point.budget);\n        }\n        Ok(())\n    }\n}",
)

# Pareto plans carry the exact context/budget reasoning selected by the compiler.
replace(
    "logan-ir/src/optimizer.rs",
    "use crate::{ContextConstraint, Placement, QuantSpec};",
    "use crate::{ContextConstraint, ContextPlan, Placement, PlannerMemoryBudget, QuantSpec};",
)
replace(
    "logan-ir/src/optimizer.rs",
    "    pub metrics: PlanMetrics,\n    pub decisions: Vec<PlanDecision>,\n}",
    "    pub metrics: PlanMetrics,\n    pub decisions: Vec<PlanDecision>,\n    /// Compiler-populated architecture-specific context state for this point.\n    pub context_plan: Option<ContextPlan>,\n    /// Compiler-populated complete memory budget used to admit this point.\n    pub memory_budget: Option<PlannerMemoryBudget>,\n}",
)
replace(
    "logan-ir/src/optimizer.rs",
    "        cost_model: input.cost_model.clone(),\n        metrics,\n        decisions,\n    }",
    "        cost_model: input.cost_model.clone(),\n        metrics,\n        decisions,\n        context_plan: None,\n        memory_budget: None,\n    }",
)

# Nested Pareto schema changed.
replace(
    "logan-ir/src/plan.rs",
    "pub const PLAN_ARTIFACT_VERSION: u32 = 2;",
    "pub const PLAN_ARTIFACT_VERSION: u32 = 3;",
)

# Compile optimizer: parse source config, account reserves/context, and enrich plans.
replace(
    "logan-compiler/src/optimize.rs",
    "    BUILTIN_COST_MODEL_V1, CandidateGroup, ContextCandidate, ContextConstraint, OptimizerInput,\n    ParetoPlan, Placement, QuantSpec, RepresentationCandidate, material_plans, select_plan,\n",
    "    BUILTIN_COST_MODEL_V1, CandidateGroup, ContextConstraint, OptimizerInput, ParetoPlan,\n    Placement, QuantSpec, RepresentationCandidate, material_plans, select_plan,\n",
)
replace(
    "logan-compiler/src/optimize.rs",
    "pub fn compile_plans(\n    model: &SemanticModel,\n    target_profile: TargetProfile,",
    "pub fn compile_plans(\n    model: &SemanticModel,\n    model_root: &Path,\n    target_profile: TargetProfile,",
)
replace(
    "logan-compiler/src/optimize.rs",
    "    let input = compile_optimizer_input(\n        model,\n        target_profile,",
    "    let (input, context_planning) = compile_optimizer_input(\n        model,\n        model_root,\n        target_profile,",
)
replace(
    "logan-compiler/src/optimize.rs",
    "    let plans = material_plans(&input).map_err(|detail| ColicError::Unsupported {\n        stage: \"target planning\",\n        detail,\n    })?;\n    Ok(CompileOptimization { plans })",
    "    let mut plans = material_plans(&input).map_err(|detail| ColicError::Unsupported {\n        stage: \"target planning\",\n        detail,\n    })?;\n    context_planning.enrich_plans(&mut plans)?;\n    Ok(CompileOptimization { plans })",
)
replace(
    "logan-compiler/src/optimize.rs",
    "fn compile_optimizer_input(\n    model: &SemanticModel,\n    target_profile: TargetProfile,",
    "fn compile_optimizer_input(\n    model: &SemanticModel,\n    model_root: &Path,\n    target_profile: TargetProfile,",
)
replace(
    "logan-compiler/src/optimize.rs",
    ") -> Result<OptimizerInput> {",
    ") -> Result<(OptimizerInput, crate::context_plan::ContextPlanning)> {",
)
old_loop = '''    for tensor in model
        .global_tensors
        .values()
        .chain(
            model
                .layer_static_tensors
                .values()
                .flat_map(|tensors| tensors.values()),
        )
        .chain(model.resident_tensors.values())
    {
        let stored = target::exact_tensor_stored_bytes(tensor)?;
        base_resident_bytes =
            checked_add(base_resident_bytes, tensor.len, "resident tensor bytes")?;
        base_package_bytes = checked_add(base_package_bytes, stored, "package tensor bytes")?;
        base_storage_traffic_bytes =
            checked_add(base_storage_traffic_bytes, stored, "tensor storage traffic")?;
    }
'''
new_loop = '''    let mut account_tensor = |name: &str, tensor: &crate::source::TensorRef| -> Result<()> {
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

    let context_planning = crate::context_plan::plan_from_package(
        model_root,
        context,
        machine,
        base_resident_bytes,
    )?;
'''
replace("logan-compiler/src/optimize.rs", old_loop, new_loop)
replace(
    "logan-compiler/src/optimize.rs",
    '''    Ok(OptimizerInput {
        cost_model: BUILTIN_COST_MODEL_V1.into(),
        groups,
        context_constraint: context,
        // #81 owns architecture-specific context-state expansion. #82 consumes
        // the shared contract and treats the requested point as fixed until
        // those additional admissible context points are available.
        context_candidates: vec![ContextCandidate {
            tokens: context.tokens,
            resident_bytes: 0,
            latency_cost: 0,
        }],''',
    '''    Ok((OptimizerInput {
        cost_model: BUILTIN_COST_MODEL_V1.into(),
        groups,
        context_constraint: context,
        context_candidates: context_planning.optimizer_candidates(),''',
)
replace(
    "logan-compiler/src/optimize.rs",
    "        heterogeneity_switch_penalty: HETEROGENEITY_SWITCH_PENALTY,\n    })\n}\n\npub fn selected_layer_quantization",
    "        heterogeneity_switch_penalty: HETEROGENEITY_SWITCH_PENALTY,\n    }, context_planning))\n}\n\nfn is_streamed_ple_tensor(name: &str) -> bool {\n    name.contains(\"ple.ple_embedding.ngram_embedding.shard_\")\n}\n\npub fn selected_layer_quantization",
)

# Thread source root into compile optimizer calls.
replace(
    "logan-compiler/src/pipeline.rs",
    "        model,\n        target_profile,\n        machine,",
    "        model,\n        &request.source,\n        target_profile,\n        machine,",
    count=1,
)

# Recompile: same architecture-aware context planner and enriched Pareto plans.
replace(
    "logan-compiler/src/recompile.rs",
    "    BUILTIN_COST_MODEL_V1, CandidateGroup, ContextCandidate, ContextConstraint, OptimizerInput,\n    ParetoPlan, Placement, QuantSpec, RepresentationCandidate, material_plans, select_plan,\n",
    "    BUILTIN_COST_MODEL_V1, CandidateGroup, ContextConstraint, OptimizerInput, ParetoPlan,\n    Placement, QuantSpec, RepresentationCandidate, material_plans, select_plan,\n",
)
replace(
    "logan-compiler/src/recompile.rs",
    "    let layer_count = by_layer.len().max(1) as u64;",
    "    let context_planning = crate::context_plan::plan_from_package(\n        &request.source,\n        context,\n        &machine,\n        base_resident_bytes,\n    )?;\n    let layer_count = by_layer.len().max(1) as u64;",
)
replace(
    "logan-compiler/src/recompile.rs",
    '''        context_constraint: context,
        context_candidates: vec![ContextCandidate {
            tokens: context.tokens,
            resident_bytes: 0,
            latency_cost: 0,
        }],''',
    '''        context_constraint: context,
        context_candidates: context_planning.optimizer_candidates(),''',
)
replace(
    "logan-compiler/src/recompile.rs",
    '''    material_plans(&input).map_err(|detail| ColicError::Unsupported {
        stage: "COLI optimization",
        detail,
    })
}''',
    '''    let mut plans = material_plans(&input).map_err(|detail| ColicError::Unsupported {
        stage: "COLI optimization",
        detail,
    })?;
    context_planning.enrich_plans(&mut plans)?;
    Ok(plans)
}''',
    count=1,
)

# CLI explains the full budget and clarifies quality metric semantics.
replace(
    "logan-compiler/src/main.rs",
    '"  [{}] {} aliases={} quality={}ppm context={} latency={} resident={} package={} traffic={}",',
    '"  [{}] {} aliases={} quant_loss={}ppm context={} latency={} resident={} package={} traffic={}",',
)
replace(
    "logan-compiler/src/main.rs",
    '''        );
    }
}

fn choose_optimizer_plan''',
    '''        );
        if let Some(budget) = plan.memory_budget {
            let state = budget.context_state;
            eprintln!(
                "      memory physical={} os={} runtime={} safety={} fixed={} scratch={} | kv={} gdn_recur={} gdn_conv={} qsa={} ple={} mtp={} | cache_headroom={}",
                human_bytes(budget.physical_memory),
                human_bytes(budget.os_reserve),
                human_bytes(budget.runtime_reserve),
                human_bytes(budget.safety_reserve),
                human_bytes(budget.fixed_model_state),
                human_bytes(budget.execution_scratch),
                human_bytes(state.full_attention_kv),
                human_bytes(state.gdn_recurrent),
                human_bytes(state.gdn_conv),
                human_bytes(state.qsa_index),
                human_bytes(state.ple),
                human_bytes(state.mtp_speculative),
                budget
                    .available_for_weights_and_cache()
                    .map(human_bytes)
                    .unwrap_or_else(|| "none".to_owned()),
            );
        }
    }
}

fn choose_optimizer_plan''',
)

# Runtime must allocate GDN state only for GDN layers, matching the planner.
for path in ["logan-qwen4/src/coliload.rs", "logan-qwen4/src/lib.rs", "logan-qwen/src/lib.rs"]:
    p = Path(path)
    s = p.read_text()
    old = '''            gdn_conv: vec![
                vec![0.0; (cdim_total(cfg)) * cfg.conv_kernel.saturating_sub(1)];
                cfg.layers
            ],
            gdn_s: vec![vec![0.0; cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim]; cfg.layers],'''
    if path.endswith("coliload.rs"):
        old = old.replace("cdim_total(cfg)", "cdim_total(&cfg)")
    if old not in s:
        # Qwen may use slightly different formatting; leave it untouched rather than guessing.
        continue
    cdim = "cdim_total(&cfg)" if path.endswith("coliload.rs") else "cdim_total(cfg)"
    new = f'''            gdn_conv: cfg
                .gdn_layers
                .iter()
                .map(|&is_gdn| {{
                    if is_gdn {{
                        vec![0.0; ({cdim}) * cfg.conv_kernel.saturating_sub(1)]
                    }} else {{
                        Vec::new()
                    }}
                }})
                .collect(),
            gdn_s: cfg
                .gdn_layers
                .iter()
                .map(|&is_gdn| {{
                    if is_gdn {{
                        vec![0.0; cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim]
                    }} else {{
                        Vec::new()
                    }}
                }})
                .collect(),'''
    p.write_text(s.replace(old, new, 1))

print("issue81 patch applied")
