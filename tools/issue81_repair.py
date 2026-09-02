from pathlib import Path


def replace(path, old, new, count=1):
    p = Path(path)
    s = p.read_text()
    if old not in s:
        raise SystemExit(f"missing repair anchor in {path}: {old[:120]!r}")
    if count is not None and s.count(old) != count:
        raise SystemExit(f"unexpected repair anchor count in {path}: {s.count(old)}")
    p.write_text(s.replace(old, new))

# Compile preview is a second compile_plans caller and must pass the source root.
replace(
    "logan-compiler/src/pipeline.rs",
    '''    Ok(crate::optimize::compile_plans(
        &model,
        target_profile,
        &machine,
        context,
        request.calibration.as_deref(),
        None,
    )?''',
    '''    Ok(crate::optimize::compile_plans(
        &model,
        &request.source,
        target_profile,
        &machine,
        context,
        request.calibration.as_deref(),
        None,
    )?''',
)

# Respect explicit GQA geometry. Falling back is valid only when the config
# omits the field; widening an explicit KV-head count silently overstates RAM.
replace(
    "logan-compiler/src/context_plan.rs",
    '''    let heads = get("num_attention_heads");
    let head_dim = get("head_dim").max(if heads == 0 { 0 } else { hidden_size / heads });
    let kv_heads = get("num_key_value_heads").max(heads);''',
    '''    let heads = get("num_attention_heads");
    let explicit_head_dim = get("head_dim");
    let head_dim = if explicit_head_dim == 0 {
        if heads == 0 { 0 } else { hidden_size / heads }
    } else {
        explicit_head_dim
    };
    let explicit_kv_heads = get("num_key_value_heads");
    let kv_heads = if explicit_kv_heads == 0 {
        heads
    } else {
        explicit_kv_heads
    };''',
)

# Add parser + infeasibility regressions immediately before the existing
# requested-context-ceiling test.
replace(
    "logan-compiler/src/context_plan.rs",
    '''    #[test]
    fn requested_context_above_model_ceiling_is_rejected() {''',
    r'''    #[test]
    fn config_parser_preserves_explicit_gqa_geometry() {
        let root = std::env::temp_dir().join(format!(
            "logan-context-plan-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("config.json");
        std::fs::write(
            &path,
            r#"{
                "hidden_size":1024,
                "num_hidden_layers":4,
                "max_position_embeddings":262144,
                "layer_types":["linear_attention","linear_attention","full_attention","full_attention"],
                "num_attention_heads":8,
                "num_key_value_heads":2,
                "head_dim":128,
                "linear_num_key_heads":4,
                "linear_key_head_dim":64,
                "linear_num_value_heads":8,
                "linear_value_head_dim":64,
                "linear_conv_kernel_dim":4,
                "indexer_n_heads":4,
                "indexer_kv_heads":2,
                "indexer_head_dim":64
            }"#,
        )
        .unwrap();
        let geometry = geometry_from_config(&path, 131_072).unwrap();
        std::fs::remove_dir_all(root).unwrap();
        assert_eq!(geometry.full_attention_layers, 2);
        assert_eq!(geometry.gdn_layers, 2);
        assert_eq!(geometry.kv_heads, 2);
        assert_eq!(geometry.head_dim, 128);
        assert_eq!(geometry.qsa_layers, 2);
        let state = state_bytes(&geometry, 131_072).unwrap();
        assert_eq!(state.full_attention_kv, 2 * 2 * 2 * 131_072 * 128 * 4);
    }

    #[test]
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

    #[test]
    fn requested_context_above_model_ceiling_is_rejected() {''',
)

print("issue81 repair applied")
