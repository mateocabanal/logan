from pathlib import Path


def replace(path, old, new, count=1):
    p = Path(path)
    text = p.read_text()
    actual = text.count(old)
    if actual < count:
        raise SystemExit(f"{path}: expected at least {count} matches, found {actual}: {old[:120]!r}")
    p.write_text(text.replace(old, new, count))

# Generic optimizer: account for constant package bytes while preserving full frontier.
replace("logan-ir/src/optimizer.rs",
'''    pub base_resident_bytes: u64,
    pub memory_budget_bytes: u64,''',
'''    pub base_resident_bytes: u64,
    pub base_package_bytes: u64,
    pub memory_budget_bytes: u64,''')
replace("logan-ir/src/optimizer.rs",
'''            let metrics = PlanMetrics {
                quality_loss: state.quality_loss,
                latency_cost: state.latency_cost,
                resident_bytes,
                package_bytes: state.package_bytes,
                context_tokens: context.tokens,
                representation_switches: state.switches,
            };''',
'''            let Some(package_bytes) = input.base_package_bytes.checked_add(state.package_bytes) else {
                continue;
            };
            let metrics = PlanMetrics {
                quality_loss: state.quality_loss,
                latency_cost: state.latency_cost,
                resident_bytes,
                package_bytes,
                context_tokens: context.tokens,
                representation_switches: state.switches,
            };''')
# Add base_package_bytes=0 to all three generic tests.
p = Path("logan-ir/src/optimizer.rs")
text = p.read_text()
text = text.replace("            base_resident_bytes: 0,\n            memory_budget_bytes:", "            base_resident_bytes: 0,\n            base_package_bytes: 0,\n            memory_budget_bytes:")
text = text.replace("            base_resident_bytes: 100,\n            memory_budget_bytes:", "            base_resident_bytes: 100,\n            base_package_bytes: 0,\n            memory_budget_bytes:")
text = text.replace("            base_resident_bytes: 200,\n            memory_budget_bytes:", "            base_resident_bytes: 200,\n            base_package_bytes: 0,\n            memory_budget_bytes:")
p.write_text(text)

# Compiler optimizer uses dense/static bytes as the constant package component.
replace("logan-compiler/src/optimizer.rs",
'''    let (base_resident_bytes, memory_budget_bytes) = machine_base_reserve(model, target_profile, machine)?;
    let input = OptimizeInput {
        groups,
        contexts: context_candidates(model, constraint),
        context_constraint: constraint,
        base_resident_bytes,
        memory_budget_bytes,''',
'''    let (base_resident_bytes, memory_budget_bytes) = machine_base_reserve(model, target_profile, machine)?;
    let base_package_bytes = dense_resident_bytes(model)?;
    let input = OptimizeInput {
        groups,
        contexts: context_candidates(model, constraint),
        context_constraint: constraint,
        base_resident_bytes,
        base_package_bytes,
        memory_budget_bytes,''')

# Share the interactive/noninteractive plan selector with recompile.
replace("logan-compiler/src/pipeline.rs",
'''fn choose_optimizer_plan(plans: &[logan_ir::ParetoPlan], requested: Option<&str>) -> Result<String> {''',
'''pub(crate) fn choose_optimizer_plan(
    plans: &[logan_ir::ParetoPlan],
    requested: Option<&str>,
) -> Result<String> {''')

# Recompile CLI: selected plan + explicit quant intent.
replace("logan-compiler/src/cli.rs",
'''    let mut optimize = false;
    let mut quant = RecompileQuantMode::Keep;
    let mut quant_rules = Vec::new();''',
'''    let mut optimize = false;
    let mut plan_choice = None;
    let mut quant = RecompileQuantMode::Keep;
    let mut quant_explicit = false;
    let mut quant_rules = Vec::new();''')
# Second --optimize occurrence is recompile.
needle = '''            "--optimize" => optimize = true,
            "--max-context" => {'''
replace("logan-compiler/src/cli.rs", needle,
'''            "--optimize" => optimize = true,
            "--select-plan" => plan_choice = Some(value(&mut args, "--select-plan")?),
            "--max-context" => {''')
replace("logan-compiler/src/cli.rs",
'''            "--quant" => quant = RecompileQuantMode::parse(&value(&mut args, "--quant")?)?,''',
'''            "--quant" => {
                quant = RecompileQuantMode::parse(&value(&mut args, "--quant")?)?;
                quant_explicit = true;
            }''')
replace("logan-compiler/src/cli.rs",
'''    if optimize && !target_explicit {
        target = "auto".to_owned();
    }

    if in_place && output.is_some() {''',
'''    if plan_choice.is_some() && !optimize {
        return Err(ColicError::Usage(
            "recompile --select-plan requires --optimize".into(),
        ));
    }
    if optimize && !target_explicit {
        target = "auto".to_owned();
    }

    if in_place && output.is_some() {''')
replace("logan-compiler/src/cli.rs",
'''        quant,
        quant_rules,
        codec,
        context,
        optimize,''',
'''        quant,
        quant_explicit,
        quant_rules,
        codec,
        context,
        optimize,
        plan_choice,''')

# Recompile imports.
replace("logan-compiler/src/recompile.rs",
'''use std::{
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};''',
'''use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};''')
replace("logan-compiler/src/recompile.rs",
'''use logan_ir::ContextConstraint;
use serde_json::json;''',
'''use logan_ir::{
    ContextCandidate, ContextConstraint, DecisionGroup, OptimizeInput, ParetoPlan, PhysicalOption,
    Placement, optimize, select_plan,
};
use serde_json::{Value, json};''')
replace("logan-compiler/src/recompile.rs",
'''const SCALE_E8M0: u16 = 0x0004;''',
'''const SCALE_E8M0: u16 = 0x0004;
const OPTIMIZER_MIB: u64 = 1024 * 1024;
const OPTIMIZER_RUNTIME_RESERVE: u64 = 256 * OPTIMIZER_MIB;
const OPTIMIZER_EXECUTION_SCRATCH: u64 = 512 * OPTIMIZER_MIB;
const OPTIMIZER_EXPERT_CACHE_SLOTS: u64 = 256;
const RECOMPILE_COST_MODEL_VERSION: &str = "logan-recompile-expert-cost-v1";''')

# Recompile request fields/defaults.
replace("logan-compiler/src/recompile.rs",
'''    pub target: String,
    pub quant: QuantMode,
    /// Ordered routed-expert overrides.''',
'''    pub target: String,
    pub quant: QuantMode,
    /// Whether --quant was explicitly supplied. In optimize mode an explicit
    /// base quant is a hard constraint unless a later --quant-rule overrides it.
    pub quant_explicit: bool,
    /// Ordered routed-expert overrides.''')
replace("logan-compiler/src/recompile.rs",
'''    pub optimize: bool,
    pub allow_requantize: bool,''',
'''    pub optimize: bool,
    /// Named Pareto label or stable plan id for non-interactive selection.
    pub plan_choice: Option<String>,
    pub allow_requantize: bool,''')
replace("logan-compiler/src/recompile.rs",
'''            target: "source".into(),
            quant: QuantMode::Keep,
            quant_rules: Vec::new(),''',
'''            target: "source".into(),
            quant: QuantMode::Keep,
            quant_explicit: false,
            quant_rules: Vec::new(),''')
replace("logan-compiler/src/recompile.rs",
'''            context: None,
            optimize: false,
            allow_requantize: false,''',
'''            context: None,
            optimize: false,
            plan_choice: None,
            allow_requantize: false,''')

# Insert recompile optimizer implementation before summary.
marker = '''#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecompileSummary {'''
insert = r'''#[derive(Debug, Clone)]
struct RecompileOptimization {
    frontier: Vec<ParetoPlan>,
    selected: ParetoPlan,
    by_record: BTreeMap<u64, QuantMode>,
}

impl RecompileOptimization {
    fn mode_for(&self, record: &RecordInfo) -> Option<QuantMode> {
        self.by_record.get(&record.id).copied()
    }
}

fn manual_quant_constraint(request: &RecompileRequest, record: &RecordInfo) -> Option<QuantMode> {
    request
        .quant_rules
        .iter()
        .filter(|rule| rule.matches(record))
        .last()
        .map(|rule| rule.mode)
        .or_else(|| request.quant_explicit.then_some(request.quant))
}

fn config_u64(config: &Value, key: &str) -> Option<u64> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .or_else(|| config.get("text_config").and_then(|text| text.get(key)).and_then(Value::as_u64))
}

fn package_context_candidates(
    source: &Path,
    constraint: ContextConstraint,
) -> Result<Vec<ContextCandidate>> {
    let path = source.join("config.json");
    let bytes = fs::read(&path).map_err(|source| ColicError::Io {
        path: path.clone(),
        source,
    })?;
    let config: Value = serde_json::from_slice(&bytes).map_err(|error| {
        ColicError::Usage(format!("cannot parse {} for context planning: {error}", path.display()))
    })?;
    let layers = config_u64(&config, "num_hidden_layers")
        .or_else(|| config_u64(&config, "n_layer"))
        .filter(|value| *value > 0)
        .ok_or_else(|| ColicError::Usage("config.json lacks a positive layer count needed by recompile --optimize".into()))?;
    let heads = config_u64(&config, "num_attention_heads")
        .or_else(|| config_u64(&config, "n_head"))
        .filter(|value| *value > 0)
        .ok_or_else(|| ColicError::Usage("config.json lacks a positive attention-head count needed by recompile --optimize".into()))?;
    let kv_heads = config_u64(&config, "num_key_value_heads")
        .filter(|value| *value > 0)
        .unwrap_or(heads);
    let head_dim = config_u64(&config, "head_dim")
        .filter(|value| *value > 0)
        .or_else(|| {
            config_u64(&config, "hidden_size")
                .filter(|hidden| hidden % heads == 0)
                .map(|hidden| hidden / heads)
        })
        .ok_or_else(|| ColicError::Usage("config.json lacks head_dim/hidden_size needed by recompile --optimize".into()))?;

    let state_bytes = |tokens: u64| -> Result<u64> {
        // Conservative v1: every layer is charged as full-attention BF16 KV.
        // Hybrid recurrent models therefore get headroom, never optimistic overcommit.
        tokens
            .checked_mul(layers)
            .and_then(|value| value.checked_mul(kv_heads))
            .and_then(|value| value.checked_mul(head_dim))
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| ColicError::Usage("recompile context-state estimate overflows u64".into()))
    };

    let mut tokens = Vec::new();
    match constraint.kind {
        logan_ir::ContextConstraintKind::Maximum => {
            tokens.push((constraint.tokens / 4).max(1));
            tokens.push((constraint.tokens / 2).max(1));
            tokens.push(constraint.tokens);
            tokens.sort_unstable();
            tokens.dedup();
        }
        logan_ir::ContextConstraintKind::Required => tokens.push(constraint.tokens),
    }
    tokens
        .into_iter()
        .map(|tokens| Ok(ContextCandidate { tokens, state_bytes: state_bytes(tokens)? }))
        .collect()
}

fn recompile_layer_sensitivity(layer: i32, max_layer: i32) -> u64 {
    let layer = u64::try_from(layer).unwrap_or(0);
    let layers = u64::try_from(max_layer.saturating_add(1)).unwrap_or(1).max(1);
    let from_start = layer + 1;
    let from_end = layers.saturating_sub(layer).max(1);
    1_000 + 8_000 / from_start.min(from_end).max(1)
}

fn probe_action_for_mode(
    package: &Package,
    record: &RecordInfo,
    request: &RecompileRequest,
    target: TargetProfile,
    target_kind: ExpertTarget,
    mode: QuantMode,
) -> Result<Action> {
    let mut probe = request.clone();
    probe.optimize = false;
    probe.plan_choice = None;
    probe.quant = mode;
    probe.quant_explicit = true;
    probe.quant_rules.clear();
    plan_record(package, record, &probe, target, target_kind)
}

fn action_latency_weight(
    kinds: &[MatrixKind; EXPERT_MATRICES],
    mode: QuantMode,
    target: TargetProfile,
) -> u64 {
    let all_mxfp4 = kinds
        .iter()
        .all(|kind| matches!(kind, MatrixKind::CanonicalMxfp4 | MatrixKind::Apple8Mxfp4));
    if mode == QuantMode::Mxfp4 || all_mxfp4 {
        if target.name == target_registry::APPLE8_PROFILE_NAME { 42 } else { 62 }
    } else if kinds.iter().all(|kind| *kind == MatrixKind::Int4G32) {
        68
    } else {
        100
    }
}

fn build_recompile_optimization(
    package: &Package,
    target: TargetProfile,
    target_kind: ExpertTarget,
    request: &RecompileRequest,
) -> Result<RecompileOptimization> {
    let constraint = request.context.ok_or_else(|| ColicError::Usage(
        "recompile --optimize requires a context constraint".into(),
    ))?;
    let mut layers: BTreeMap<i32, Vec<RecordInfo>> = BTreeMap::new();
    let mut base_package_bytes = 0_u64;
    let mut base_resident_bytes = 0_u64;
    for record in package.records() {
        if record.kind == 2 {
            layers.entry(record.layer).or_default().push(record.clone());
        } else {
            base_package_bytes = base_package_bytes
                .checked_add(record.stored)
                .ok_or_else(|| ColicError::Usage("recompile package-byte accounting overflows u64".into()))?;
            base_resident_bytes = base_resident_bytes
                .checked_add(record.decoded)
                .ok_or_else(|| ColicError::Usage("recompile resident-byte accounting overflows u64".into()))?;
        }
    }
    let max_layer = layers.keys().copied().max().unwrap_or(0);
    let mut groups = Vec::new();
    let mut members: BTreeMap<String, Vec<RecordInfo>> = BTreeMap::new();
    let mut worst_expert_decoded = 0_u64;

    for (layer, records) in layers {
        let group_id = format!("layer:{layer}/experts");
        let has_unconstrained = records
            .iter()
            .any(|record| manual_quant_constraint(request, record).is_none());
        let candidate_modes: &[QuantMode] = if has_unconstrained {
            &[QuantMode::Keep, QuantMode::Mxfp4]
        } else {
            &[QuantMode::Keep]
        };
        let mut options = Vec::new();
        for &free_mode in candidate_modes {
            let mut package_bytes = 0_u64;
            let mut latency_cost = 0_u64;
            let mut quality_numerator = 0_u64;
            let mut viable = true;
            for record in &records {
                let manual = manual_quant_constraint(request, record);
                let mode = manual.unwrap_or(free_mode);
                let descs = expert_descs(package, record)?;
                let kinds = [descs[0].kind()?, descs[1].kind()?, descs[2].kind()?];
                let action = match probe_action_for_mode(package, record, request, target, target_kind, mode) {
                    Ok(action) => action,
                    Err(error) if manual.is_none() => {
                        let _ = error;
                        viable = false;
                        break;
                    }
                    Err(error) => {
                        return Err(ColicError::Usage(format!(
                            "manual quant constraint for layer={} expert={} is not executable: {error}",
                            record.layer, record.expert
                        )));
                    }
                };
                package_bytes = package_bytes
                    .checked_add(action.lowered.stored_bytes)
                    .ok_or_else(|| ColicError::Usage("recompile optimizer package bytes overflow u64".into()))?;
                worst_expert_decoded = worst_expert_decoded.max(action.lowered.decoded_bytes);
                let weight = action_latency_weight(&kinds, mode, target);
                latency_cost = latency_cost
                    .checked_add(action.lowered.stored_bytes.div_ceil(OPTIMIZER_MIB).max(1).saturating_mul(weight))
                    .ok_or_else(|| ColicError::Usage("recompile optimizer latency cost overflows u64".into()))?;
                let source_all_mxfp4 = kinds.iter().all(|kind| {
                    matches!(kind, MatrixKind::CanonicalMxfp4 | MatrixKind::Apple8Mxfp4)
                });
                if mode == QuantMode::Mxfp4 && !source_all_mxfp4 {
                    let factor = if kinds.iter().any(|kind| *kind == MatrixKind::Int4G32) { 2 } else { 1 };
                    quality_numerator = quality_numerator
                        .checked_add(recompile_layer_sensitivity(layer, max_layer).saturating_mul(factor))
                        .ok_or_else(|| ColicError::Usage("recompile optimizer quality cost overflows u64".into()))?;
                }
            }
            if !viable {
                continue;
            }
            let quality_loss = if records.is_empty() {
                0
            } else {
                quality_numerator / u64::try_from(records.len()).unwrap_or(1)
            };
            let option_id = if has_unconstrained { free_mode.as_str() } else { "manual" };
            options.push(PhysicalOption {
                id: option_id.into(),
                representation: option_id.into(),
                layout: target.name.into(),
                placement: Placement::Streamed,
                quality_loss,
                latency_cost,
                resident_bytes: 0,
                package_bytes,
            });
        }
        if options.is_empty() {
            return Err(ColicError::unsupported(
                "COLI optimization",
                format!("no executable representation exists for {group_id}"),
            ));
        }
        members.insert(group_id.clone(), records);
        groups.push(DecisionGroup { id: group_id, options });
    }

    let machine = target::MachineProfile::probe();
    let physical = machine
        .ram_bytes
        .unwrap_or(target::machine::DEFAULT_POOL_BUDGET);
    let cache = OPTIMIZER_EXPERT_CACHE_SLOTS
        .checked_mul(worst_expert_decoded)
        .ok_or_else(|| ColicError::Usage("recompile expert-cache reservation overflows u64".into()))?;
    base_resident_bytes = base_resident_bytes
        .checked_add(cache)
        .and_then(|value| value.checked_add(physical / 8))
        .and_then(|value| value.checked_add(physical / 10))
        .and_then(|value| value.checked_add(OPTIMIZER_RUNTIME_RESERVE))
        .and_then(|value| value.checked_add(OPTIMIZER_EXECUTION_SCRATCH))
        .ok_or_else(|| ColicError::Usage("recompile base memory reservation overflows u64".into()))?;

    let input = OptimizeInput {
        groups,
        contexts: package_context_candidates(&request.source, constraint)?,
        context_constraint: constraint,
        base_resident_bytes,
        base_package_bytes,
        memory_budget_bytes: physical,
        heterogeneity_penalty: 250,
    };
    let plans = optimize(&input).map_err(|message| ColicError::Usage(format!("optimizer: {message}")))?;
    if plans.is_empty() {
        return Err(ColicError::unsupported(
            "COLI optimization",
            format!(
                "no Pareto plan fits {:?} context {} within {} bytes of physical memory",
                constraint.kind, constraint.tokens, physical
            ),
        ));
    }
    let selector = crate::pipeline::choose_optimizer_plan(&plans, request.plan_choice.as_deref())?;
    let selected = select_plan(&plans, &selector).cloned().ok_or_else(|| {
        ColicError::Usage(format!("unknown optimizer plan `{selector}`"))
    })?;
    let mut by_record = BTreeMap::new();
    for decision in &selected.decisions {
        let records = members.get(&decision.group).ok_or_else(|| {
            ColicError::Usage(format!("optimizer lost group membership for `{}`", decision.group))
        })?;
        let free_mode = match decision.option_id.as_str() {
            "keep" => Some(QuantMode::Keep),
            "mxfp4" => Some(QuantMode::Mxfp4),
            "manual" => None,
            other => return Err(ColicError::Usage(format!("optimizer selected unknown recompile option `{other}`"))),
        };
        for record in records {
            let mode = manual_quant_constraint(request, record)
                .or(free_mode)
                .ok_or_else(|| ColicError::Usage("manual-only optimizer group lost its quant constraint".into()))?;
            by_record.insert(record.id, mode);
        }
    }
    Ok(RecompileOptimization { frontier: plans, selected, by_record })
}

'''
replace("logan-compiler/src/recompile.rs", marker, insert + marker)

# Replace fail-closed optimize gate and hook optimizer after target resolution.
replace("logan-compiler/src/recompile.rs",
'''    if request.optimize {
        if request.context.is_none() {
            return Err(ColicError::Usage(
                "recompile --optimize requires exactly one of --max-context N or --require-context N".into(),
            ));
        }
        return Err(ColicError::unsupported(
            "COLI optimization",
            "--optimize is wired for recompile, including native target selection and context intent, but mixed-representation Pareto search is not implemented yet (see #82); no package was modified",
        ));
    }
''',
'''    if request.optimize && request.context.is_none() {
        return Err(ColicError::Usage(
            "recompile --optimize requires exactly one of --max-context N or --require-context N".into(),
        ));
    }
''')
replace("logan-compiler/src/recompile.rs",
'''    let target_kind = if target.name == target_registry::APPLE8_PROFILE_NAME {
        ExpertTarget::Apple8Mxfp4
    } else {
        ExpertTarget::CanonicalMxfp4
    };

    let mut actions = Vec::with_capacity(package.records().len());''',
'''    let target_kind = if target.name == target_registry::APPLE8_PROFILE_NAME {
        ExpertTarget::Apple8Mxfp4
    } else {
        ExpertTarget::CanonicalMxfp4
    };
    let optimization = if request.optimize {
        Some(build_recompile_optimization(&package, target, target_kind, request)?)
    } else {
        None
    };

    let mut actions = Vec::with_capacity(package.records().len());''')
replace("logan-compiler/src/recompile.rs",
'''    for record in package.records() {
        let action = plan_record(&package, record, request, target, target_kind)?;''',
'''    for record in package.records() {
        let action = if let Some(optimization) = &optimization {
            if let Some(mode) = optimization.mode_for(record) {
                probe_action_for_mode(&package, record, request, target, target_kind, mode)?
            } else {
                plan_record(&package, record, request, target, target_kind)?
            }
        } else {
            plan_record(&package, record, request, target, target_kind)?
        };''')
replace("logan-compiler/src/recompile.rs",
'''        request,
        &temporary,
    );''',
'''        request,
        optimization.as_ref(),
        &temporary,
    );''')

# Pass optimization into package/provenance writer.
replace("logan-compiler/src/recompile.rs",
'''    fingerprint: [u8; 32],
    request: &RecompileRequest,
    temporary: &Path,''',
'''    fingerprint: [u8; 32],
    request: &RecompileRequest,
    optimization: Option<&RecompileOptimization>,
    temporary: &Path,''')
replace("logan-compiler/src/recompile.rs",
'''    write_provenance(package, target, actions, request, temporary)?;''',
'''    write_provenance(package, target, actions, request, optimization, temporary)?;''')
replace("logan-compiler/src/recompile.rs",
'''    actions: &[Action],
    request: &RecompileRequest,
    output: &Path,''',
'''    actions: &[Action],
    request: &RecompileRequest,
    optimization: Option<&RecompileOptimization>,
    output: &Path,''')
replace("logan-compiler/src/recompile.rs",
'''        "optimize": request.optimize,
        "in_place": request.source == request.output,''',
'''        "optimize": request.optimize,
        "optimizer": optimization.map(|optimization| json!({
            "cost_model_version": RECOMPILE_COST_MODEL_VERSION,
            "selected": &optimization.selected,
            "frontier": &optimization.frontier,
        })),
        "in_place": request.source == request.output,''')

# Tests: manual rules are hard optimizer constraints at the decision boundary.
marker = '''    #[test]
    fn quant_rule_parser_rejects_reversed_and_unknown_selectors() {'''
insert = r'''    #[test]
    fn manual_quant_rules_are_optimizer_constraints() {
        let mut request =
            RecompileRequest::new(PathBuf::from("old.coli"), PathBuf::from("new.coli"));
        request.optimize = true;
        request.quant_rules = vec![
            QuantRule::parse("layer:2=mxfp4").unwrap(),
            QuantRule::parse("layer:2/expert:7=keep").unwrap(),
        ];
        let mut record = RecordInfo {
            id: 1,
            kind: 2,
            codec: 0,
            math_format: 0,
            scale_format: 0,
            layout: 0,
            flags: 0,
            shard_id: 0,
            name: None,
            layer: 2,
            expert: 3,
            offset: 0,
            stored: 0,
            decoded: 0,
            stored_crc: 0,
            logical_crc: 0,
        };
        assert_eq!(manual_quant_constraint(&request, &record), Some(QuantMode::Mxfp4));
        record.expert = 7;
        assert_eq!(manual_quant_constraint(&request, &record), Some(QuantMode::Keep));
        record.layer = 4;
        assert_eq!(manual_quant_constraint(&request, &record), None);

        request.quant = QuantMode::Mxfp4;
        request.quant_explicit = true;
        assert_eq!(manual_quant_constraint(&request, &record), Some(QuantMode::Mxfp4));
    }

'''
replace("logan-compiler/src/recompile.rs", marker, insert + marker)

# CLI test for noninteractive named plan parsing.
marker = '''    #[test]
    fn optimized_recompile_requires_context() {'''
insert = r'''    #[test]
    fn optimized_recompile_parses_named_plan_and_explicit_quant_constraint() {
        let command = parse(
            [
                "recompile",
                "model.coli",
                "--in-place",
                "--optimize",
                "--require-context",
                "65536",
                "--select-plan",
                "balanced",
                "--quant",
                "mxfp4",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        let Command::Recompile(request) = command else {
            panic!("expected recompile")
        };
        assert_eq!(request.plan_choice.as_deref(), Some("balanced"));
        assert!(request.quant_explicit);
        assert_eq!(request.quant, RecompileQuantMode::Mxfp4);
    }

'''
replace("logan-compiler/src/cli.rs", marker, insert + marker)
