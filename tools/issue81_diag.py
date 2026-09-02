from pathlib import Path

p = Path("logan-compiler/src/context_plan.rs")
s = p.read_text()
old = '''    if points.is_empty() {
        return Err(ColicError::Usage(
            "no context points satisfy the requested context constraint".into(),
        ));
    }
    Ok(ContextPlanning { geometry, points })
}'''
new = '''    if points.is_empty() {
        return Err(ColicError::Usage(
            "no context points satisfy the requested context constraint".into(),
        ));
    }
    if constraint.kind == ContextConstraintKind::Required {
        let required = points
            .iter()
            .find(|point| point.candidate.tokens == constraint.tokens)
            .ok_or_else(|| ColicError::Usage("required context point was not planned".into()))?;
        let reserved = required.budget.reserved_bytes().ok_or_else(|| {
            ColicError::Usage("required-context memory accounting overflowed u64".into())
        })?;
        if reserved > physical_memory {
            let state = required.plan.state_bytes;
            return Err(ColicError::Usage(format!(
                "required context {} cannot fit fixed runtime state: physical={} bytes reserved={} bytes deficit={} bytes before expert cache; os={} runtime={} safety={} fixed_model={} scratch={} kv={} gdn_recurrent={} gdn_conv={} qsa={} ple={} mtp={}",
                constraint.tokens,
                physical_memory,
                reserved,
                reserved - physical_memory,
                required.budget.os_reserve,
                required.budget.runtime_reserve,
                required.budget.safety_reserve,
                required.budget.fixed_model_state,
                required.budget.execution_scratch,
                state.full_attention_kv,
                state.gdn_recurrent,
                state.gdn_conv,
                state.qsa_index,
                state.ple,
                state.mtp_speculative,
            )));
        }
    }
    Ok(ContextPlanning { geometry, points })
}'''
if old not in s:
    raise SystemExit("plan_geometry return anchor did not match")
s = s.replace(old, new, 1)
old_test = '''    #[test]
    fn sixteen_gib_cannot_admit_128k_for_large_hybrid_geometry() {
        let planning = plan_geometry(
            hybrid(),
            ContextConstraint::required(131_072),
            &machine(16 * GIB),
            2 * GIB,
        )
        .unwrap();
        let point = planning.point_for_tokens(131_072).unwrap();
        assert_eq!(point.budget.available_for_weights_and_cache(), None);
        assert!(point.plan.state_bytes.full_attention_kv > 16 * GIB);
    }
'''
new_test = '''    #[test]
    fn sixteen_gib_rejects_required_128k_with_memory_breakdown() {
        let error = plan_geometry(
            hybrid(),
            ContextConstraint::required(131_072),
            &machine(16 * GIB),
            2 * GIB,
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("required context 131072"));
        assert!(message.contains("before expert cache"));
        assert!(message.contains("kv="));
        assert!(message.contains("qsa="));
    }

    #[test]
    fn maximum_context_keeps_lower_feasible_points_when_ceiling_is_too_large() {
        let planning = plan_geometry(
            hybrid(),
            ContextConstraint::maximum(131_072),
            &machine(16 * GIB),
            2 * GIB,
        )
        .unwrap();
        let low = planning.point_for_tokens(32_768).unwrap();
        let high = planning.point_for_tokens(131_072).unwrap();
        assert!(low.budget.available_for_weights_and_cache().is_some());
        assert!(high.budget.available_for_weights_and_cache().is_none());
    }
'''
if old_test not in s:
    raise SystemExit("128k test anchor did not match")
p.write_text(s.replace(old_test, new_test, 1))
print("issue81 diagnostics applied")
