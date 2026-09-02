from pathlib import Path


def replace_exact(path, old, new, count=1):
    p = Path(path)
    s = p.read_text()
    actual = s.count(old)
    assert actual == count, f'{path}: expected {count}, got {actual}: {old[:120]!r}'
    p.write_text(s.replace(old, new, count))

# Shared compiler quality prior/calibration lookup for COLI recompile adapter.
replace_exact(
    'logan-compiler/src/optimize.rs',
    '    fn quality(&self, group: &str, candidate: &str, fallback: u64) -> u64 {',
    '    pub(crate) fn quality(&self, group: &str, candidate: &str, fallback: u64) -> u64 {',
)
replace_exact(
    'logan-compiler/src/optimize.rs',
    'fn layer_quant_quality_prior(layer: u32, layers: u32) -> u64 {',
    'pub(crate) fn layer_quant_quality_prior(layer: u32, layers: u32) -> u64 {',
)

# ---------------------------------------------------------------------------
# recompile.rs
# ---------------------------------------------------------------------------
p = Path('logan-compiler/src/recompile.rs')
s = p.read_text()

def rep(old, new, count=1):
    global s
    actual = s.count(old)
    assert actual == count, f'recompile.rs expected {count}, got {actual}: {old[:120]!r}'
    s = s.replace(old, new, count)

rep(
    'use logan_ir::ContextConstraint;\n',
    '''use logan_ir::{\n    BUILTIN_COST_MODEL_V1, CandidateGroup, ContextCandidate, ContextConstraint, OptimizerInput,\n    ParetoPlan, Placement, QuantSpec, RepresentationCandidate, material_plans, select_plan,\n};\n''',
)
rep(
    '''    /// Hardware-specialized optimizer entrypoint. The candidate search itself\n    /// is tracked in #82 and fails closed until it exists.\n    pub optimize: bool,\n    pub allow_requantize: bool,\n''',
    '''    /// Hardware-specialized optimizer entrypoint.\n    pub optimize: bool,\n    /// Stable Pareto plan id or UX alias.\n    pub plan_choice: Option<String>,\n    /// Optional precomputed calibration-score JSON.\n    pub calibration: Option<PathBuf>,\n    pub allow_requantize: bool,\n''',
)
rep(
    '''            context: None,\n            optimize: false,\n            allow_requantize: false,\n''',
    '''            context: None,\n            optimize: false,\n            plan_choice: None,\n            calibration: None,\n            allow_requantize: false,\n''',
)
rep(
    '''pub struct RecompileSummary {\n    pub source_profile: String,\n    pub target_profile: String,\n    pub records: usize,\n    pub copied_records: usize,\n    pub rewritten_experts: usize,\n    pub requantized_experts: usize,\n    pub source_fingerprint: String,\n}\n''',
    '''pub struct RecompileSummary {\n    pub source_profile: String,\n    pub target_profile: String,\n    pub records: usize,\n    pub copied_records: usize,\n    pub rewritten_experts: usize,\n    pub requantized_experts: usize,\n    pub source_fingerprint: String,\n    pub optimizer_plan: Option<String>,\n}\n''',
)
start = s.index('pub fn recompile(request: &RecompileRequest) -> Result<RecompileSummary> {')
end = s.index('\nfn resolve_target(', start)
new_recompile = r'''pub fn recompile(request: &RecompileRequest) -> Result<RecompileSummary> {
    if request.optimize && request.context.is_none() {
        return Err(ColicError::Usage(
            "recompile --optimize requires --max-context or --require-context".into(),
        ));
    }
    if request.source == request.output && !request.force {
        return Err(ColicError::Usage(
            "recompiling in place requires --force; using a separate output path is safer".into(),
        ));
    }

    let package = Package::open(&request.source)?;
    let target = resolve_target(&package, &request.target)?;
    let target_kind = if target.name == target_registry::APPLE8_PROFILE_NAME {
        ExpertTarget::Apple8Mxfp4
    } else {
        ExpertTarget::CanonicalMxfp4
    };

    let selected_optimizer = if request.optimize {
        let plans = build_recompile_plans(&package, request, target, target_kind)?;
        let selector = request.plan_choice.as_deref().ok_or_else(|| {
            ColicError::Usage(
                "optimized recompile requires --plan-choice NAME|ID in non-interactive code paths; run the CLI without --plan-choice for the interactive picker or use --dry-run-equivalent preview tooling"
                    .into(),
            )
        })?;
        Some(
            select_plan(&plans, selector)
                .cloned()
                .ok_or_else(|| {
                    ColicError::Usage(format!(
                        "unknown optimizer plan `{selector}`; expected one of {}",
                        plans
                            .iter()
                            .flat_map(|plan| plan
                                .labels
                                .iter()
                                .cloned()
                                .chain(std::iter::once(plan.id.clone())))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                })?,
        )
    } else {
        None
    };
    let optimizer_modes = selected_optimizer
        .as_ref()
        .map(|plan| optimizer_modes_from_plan(request, &package, plan))
        .transpose()?;

    let mut actions = Vec::with_capacity(package.records().len());
    let mut copied_records = 0_usize;
    let mut rewritten_experts = 0_usize;
    let mut requantized_experts = 0_usize;

    for record in package.records() {
        let action = if let Some(mode) = optimizer_modes
            .as_ref()
            .and_then(|modes| modes.get(&record.id))
            .copied()
        {
            plan_record_with_quant(&package, record, request, target, target_kind, mode)?
        } else {
            plan_record(&package, record, request, target, target_kind)?
        };
        match action.kind {
            ActionKind::Copy => copied_records += 1,
            ActionKind::Rewrite { requantized, .. } => {
                rewritten_experts += 1;
                if requantized {
                    requantized_experts += 1;
                }
            }
        }
        actions.push(action);
    }

    let lowered = actions
        .iter()
        .map(|action| action.lowered.clone())
        .collect::<Vec<_>>();
    let plan = storage::plan_records(&lowered, target, 4 * 1024 * 1024 * 1024)?;
    let fingerprint = *package.fingerprint();
    let temporary = storage::temporary_package_path(&request.output)?;

    let write_result = write_package(
        &package,
        &actions,
        &plan,
        target,
        fingerprint,
        request,
        selected_optimizer.as_ref(),
        &temporary,
    );
    if let Err(error) = write_result {
        let _ = fs::remove_dir_all(&temporary);
        return Err(error);
    }

    if request.verify || request.source == request.output {
        let verification = crate::verify::verify_package(&temporary)
            .and_then(|_| crate::verify_target::verify_target_layouts(&temporary));
        if let Err(error) = verification {
            let _ = fs::remove_dir_all(&temporary);
            return Err(error);
        }
    }

    if request.force {
        storage::replace_package(&temporary, &request.output)?;
    } else {
        storage::publish_package(&temporary, &request.output)?;
    }

    Ok(RecompileSummary {
        source_profile: package.profile().to_owned(),
        target_profile: target.name.to_owned(),
        records: actions.len(),
        copied_records,
        rewritten_experts,
        requantized_experts,
        source_fingerprint: hex_fingerprint(package.fingerprint()),
        optimizer_plan: selected_optimizer.map(|plan| plan.id),
    })
}

pub fn preview_optimization(request: &RecompileRequest) -> Result<Vec<ParetoPlan>> {
    if !request.optimize {
        return Err(ColicError::Usage(
            "recompile optimization preview requires --optimize".into(),
        ));
    }
    if request.context.is_none() {
        return Err(ColicError::Usage(
            "recompile --optimize requires --max-context or --require-context".into(),
        ));
    }
    let package = Package::open(&request.source)?;
    let target = resolve_target(&package, &request.target)?;
    let target_kind = if target.name == target_registry::APPLE8_PROFILE_NAME {
        ExpertTarget::Apple8Mxfp4
    } else {
        ExpertTarget::CanonicalMxfp4
    };
    build_recompile_plans(&package, request, target, target_kind)
}
'''
s = s[:start] + new_recompile + s[end:]

marker = '\nfn plan_record(\n'
pos = s.index(marker)
helpers = r'''
fn build_recompile_plans(
    package: &Package,
    request: &RecompileRequest,
    target: TargetProfile,
    target_kind: ExpertTarget,
) -> Result<Vec<ParetoPlan>> {
    let context = request.context.ok_or_else(|| {
        ColicError::Usage("recompile --optimize requires a context constraint".into())
    })?;
    let calibration = crate::optimize::CalibrationScores::load(request.calibration.as_deref())?;
    let machine = target::MachineProfile::probe();
    let memory_budget_bytes = machine
        .ram_bytes
        .unwrap_or(target::machine::DEFAULT_POOL_BUDGET);

    let mut base_resident_bytes = 0_u64;
    let mut base_package_bytes = 0_u64;
    let mut by_layer = std::collections::BTreeMap::<i32, Vec<&RecordInfo>>::new();
    for record in package.records() {
        if record.kind == 2 {
            by_layer.entry(record.layer).or_default().push(record);
        } else {
            base_package_bytes = base_package_bytes
                .checked_add(record.stored)
                .ok_or_else(|| ColicError::Usage("recompile package bytes overflow".into()))?;
            base_resident_bytes = base_resident_bytes
                .checked_add(record.decoded)
                .ok_or_else(|| ColicError::Usage("recompile resident bytes overflow".into()))?;
        }
    }
    let layer_count = by_layer.len().max(1) as u64;
    let slots_per_layer = 256_u64.div_ceil(layer_count).max(1);
    let experts_per_token = package_experts_per_token(&request.source);
    let model_layers = by_layer
        .keys()
        .filter_map(|layer| u32::try_from(*layer).ok())
        .max()
        .map_or(1, |layer| layer.saturating_add(1));

    let mut groups = Vec::with_capacity(by_layer.len());
    for (&layer, records) in &by_layer {
        let group_key = format!("layer:{layer}:routed-experts");
        let mut options = Vec::new();
        let mut signatures = Vec::<(u64, u64, u64, u64, String, u16)>::new();
        for bias in [QuantMode::Keep, QuantMode::Mxfp4] {
            let mut package_bytes = 0_u64;
            let mut max_decoded = 0_u64;
            let mut traffic_bytes = 0_u64;
            let mut fresh_quant = false;
            let mut requantize = false;
            let mut quantized = 0_usize;
            let mut feasible = true;
            for record in records {
                let mode = optimizer_mode_for_record(request, record, bias);
                let descs = expert_descs(package, record)?;
                let kinds = [descs[0].kind()?, descs[1].kind()?, descs[2].kind()?];
                if mode == QuantMode::Mxfp4 {
                    quantized += 1;
                    fresh_quant |= kinds.iter().any(|kind| *kind == MatrixKind::Bf16);
                    requantize |= kinds.iter().any(|kind| *kind == MatrixKind::Int4G32);
                }
                let action = match plan_record_with_quant(
                    package,
                    record,
                    request,
                    target,
                    target_kind,
                    mode,
                ) {
                    Ok(action) => action,
                    Err(_) => {
                        feasible = false;
                        break;
                    }
                };
                package_bytes = package_bytes
                    .checked_add(action.lowered.stored_bytes)
                    .ok_or_else(|| ColicError::Usage("candidate package bytes overflow".into()))?;
                traffic_bytes = traffic_bytes
                    .checked_add(action.lowered.stored_bytes)
                    .ok_or_else(|| ColicError::Usage("candidate traffic bytes overflow".into()))?;
                max_decoded = max_decoded.max(action.lowered.decoded_bytes);
            }
            if !feasible {
                continue;
            }
            let average = traffic_bytes.div_ceil(records.len() as u64);
            let traffic = average
                .checked_mul(experts_per_token)
                .ok_or_else(|| ColicError::Usage("candidate token traffic overflows".into()))?;
            let resident = max_decoded
                .checked_mul(slots_per_layer)
                .ok_or_else(|| ColicError::Usage("candidate cache bytes overflow".into()))?;
            let fallback_quality = if requantize {
                crate::optimize::layer_quant_quality_prior(
                    u32::try_from(layer).unwrap_or(0),
                    model_layers,
                )
                .saturating_mul(2)
            } else if fresh_quant {
                crate::optimize::layer_quant_quality_prior(
                    u32::try_from(layer).unwrap_or(0),
                    model_layers,
                )
            } else {
                0
            };
            let candidate_id = bias.as_str();
            let quality = calibration.quality(&group_key, candidate_id, fallback_quality);
            let quant_kind = if quantized == 0 {
                "source"
            } else if quantized == records.len() {
                if target_kind == ExpertTarget::Apple8Mxfp4 {
                    "mxfp4-tile8x32"
                } else {
                    "mxfp4"
                }
            } else {
                "mixed"
            };
            let layout = if target_kind == ExpertTarget::Apple8Mxfp4 {
                target_registry::APPLE8_MXFP4_TILE_LAYOUT
            } else {
                0
            };
            let signature = (
                package_bytes,
                resident,
                traffic,
                quality,
                quant_kind.to_owned(),
                layout,
            );
            if signatures.contains(&signature) {
                continue;
            }
            signatures.push(signature);
            options.push(RepresentationCandidate {
                id: candidate_id.into(),
                quant: QuantSpec {
                    kind: quant_kind.into(),
                    scale: (quant_kind.starts_with("mxfp4"))
                        .then(|| "e8m0/1x32".to_owned()),
                },
                layout,
                placement: Placement::Streamed,
                resident_bytes: resident,
                package_bytes,
                storage_traffic_bytes: traffic,
                latency_cost: traffic.div_ceil(4096).max(1),
                quality_loss_ppm: quality,
                dispatch_class: if bias == QuantMode::Keep { 1 } else { 2 },
                rationale: format!(
                    "{}-biased layer plan after hard quant-rule pins: {quantized}/{} experts use MXFP4; fresh_quant={fresh_quant}; requantize={requantize}",
                    bias.as_str(),
                    records.len()
                ),
            });
        }
        if options.is_empty() {
            return Err(ColicError::Unsupported {
                stage: "COLI optimization",
                detail: format!(
                    "no feasible expert representation remains for `{group_key}` on `{}` after manual constraints",
                    target.name
                ),
            });
        }
        groups.push(CandidateGroup {
            key: group_key,
            options,
        });
    }

    let input = OptimizerInput {
        cost_model: BUILTIN_COST_MODEL_V1.into(),
        groups,
        context_constraint: context,
        context_candidates: vec![ContextCandidate {
            tokens: context.tokens,
            resident_bytes: 0,
            latency_cost: 0,
        }],
        memory_budget_bytes,
        base_resident_bytes,
        base_package_bytes,
        base_storage_traffic_bytes: 0,
        base_latency_cost: 0,
        base_quality_loss_ppm: 0,
        heterogeneity_switch_penalty: 64,
    };
    material_plans(&input).map_err(|detail| ColicError::Unsupported {
        stage: "COLI optimization",
        detail,
    })
}

fn optimizer_mode_for_record(
    request: &RecompileRequest,
    record: &RecordInfo,
    bias: QuantMode,
) -> QuantMode {
    if request.quant == QuantMode::Mxfp4 {
        return QuantMode::Mxfp4;
    }
    request
        .quant_rules
        .iter()
        .filter(|rule| rule.matches(record))
        .last()
        .map_or(bias, |rule| rule.mode)
}

fn optimizer_modes_from_plan(
    request: &RecompileRequest,
    package: &Package,
    plan: &ParetoPlan,
) -> Result<std::collections::BTreeMap<u64, QuantMode>> {
    let mut modes = std::collections::BTreeMap::new();
    for decision in &plan.decisions {
        let layer = decision
            .group
            .strip_prefix("layer:")
            .and_then(|rest| rest.strip_suffix(":routed-experts"))
            .and_then(|value| value.parse::<i32>().ok())
            .ok_or_else(|| {
                ColicError::Usage(format!(
                    "optimizer decision has invalid recompile group `{}`",
                    decision.group
                ))
            })?;
        let bias = QuantMode::parse(&decision.chosen.id)?;
        for record in package
            .records()
            .iter()
            .filter(|record| record.kind == 2 && record.layer == layer)
        {
            modes.insert(record.id, optimizer_mode_for_record(request, record, bias));
        }
    }
    Ok(modes)
}

fn package_experts_per_token(root: &Path) -> u64 {
    let Ok(text) = fs::read_to_string(root.join("config.json")) else {
        return 1;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&text) else {
        return 1;
    };
    ["num_experts_per_tok", "num_experts_per_token", "experts_per_token"]
        .into_iter()
        .find_map(|key| config.get(key).and_then(serde_json::Value::as_u64))
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

'''
s = s[:pos] + helpers + s[pos:]

old_sig = '''fn plan_record(\n    package: &Package,\n    record: &RecordInfo,\n    request: &RecompileRequest,\n    target_profile: TargetProfile,\n    target_kind: ExpertTarget,\n) -> Result<Action> {\n'''
new_sig = '''fn plan_record(\n    package: &Package,\n    record: &RecordInfo,\n    request: &RecompileRequest,\n    target_profile: TargetProfile,\n    target_kind: ExpertTarget,\n) -> Result<Action> {\n    let quant = effective_quant(request, record);\n    plan_record_with_quant(\n        package,\n        record,\n        request,\n        target_profile,\n        target_kind,\n        quant,\n    )\n}\n\nfn plan_record_with_quant(\n    package: &Package,\n    record: &RecordInfo,\n    request: &RecompileRequest,\n    target_profile: TargetProfile,\n    target_kind: ExpertTarget,\n    quant: QuantMode,\n) -> Result<Action> {\n'''
rep(old_sig, new_sig)
rep('    let quant = effective_quant(request, record);\n', '', 1)
rep(
    '''fn write_package(\n    package: &Package,\n    actions: &[Action],\n    plan: &storage::StoragePlan,\n    target: TargetProfile,\n    fingerprint: [u8; 32],\n    request: &RecompileRequest,\n    temporary: &Path,\n) -> Result<()> {\n''',
    '''fn write_package(\n    package: &Package,\n    actions: &[Action],\n    plan: &storage::StoragePlan,\n    target: TargetProfile,\n    fingerprint: [u8; 32],\n    request: &RecompileRequest,\n    optimizer: Option<&ParetoPlan>,\n    temporary: &Path,\n) -> Result<()> {\n''',
)
rep(
    '    write_provenance(package, target, actions, request, temporary)?;\n',
    '    write_provenance(package, target, actions, request, optimizer, temporary)?;\n',
)
rep(
    '''fn write_provenance(\n    package: &Package,\n    target: TargetProfile,\n    actions: &[Action],\n    request: &RecompileRequest,\n    output: &Path,\n) -> Result<()> {\n''',
    '''fn write_provenance(\n    package: &Package,\n    target: TargetProfile,\n    actions: &[Action],\n    request: &RecompileRequest,\n    optimizer: Option<&ParetoPlan>,\n    output: &Path,\n) -> Result<()> {\n''',
)
rep(
    '''        "optimize": request.optimize,\n        "in_place": request.source == request.output,\n''',
    '''        "optimize": request.optimize,\n        "optimizer_plan": optimizer,\n        "in_place": request.source == request.output,\n''',
)
p.write_text(s)
print('patched recompile.rs')

# ---------------------------------------------------------------------------
# cli.rs
# ---------------------------------------------------------------------------
p = Path('logan-compiler/src/cli.rs')
s = p.read_text()

def crep(old, new, count=1):
    global s
    actual = s.count(old)
    assert actual == count, f'cli.rs expected {count}, got {actual}: {old[:120]!r}'
    s = s.replace(old, new, count)

crep(
    'pub const USAGE: &str = "Usage:\\n  logan inspect-source MODEL_DIR\\n  logan verify PACKAGE_DIR\\n  logan compile MODEL_DIR (--max-context N | --require-context N) [--optimize] --target auto|native|PROFILE --quant exact|PROFILE --quant-floor bf16|exact --codec none|auto|PROFILE --opt default|size|latency -o OUTPUT [--plan PLAN_PATH] [--dry-run] [--verify] [--force]\\n  logan recompile PACKAGE_DIR (-o OUTPUT | --in-place) [--target source|auto|native|PROFILE] [--optimize (--max-context N | --require-context N)] [--quant keep|mxfp4] [--quant-rule SELECTOR=keep|mxfp4]... [--codec keep|none] [--allow-requantize] [--repack] [--verify] [--force]";',
    'pub const USAGE: &str = "Usage:\\n  logan inspect-source MODEL_DIR\\n  logan verify PACKAGE_DIR\\n  logan compile MODEL_DIR (--max-context N | --require-context N) [--optimize [--plan-choice NAME|ID] [--calibration FILE]] --target auto|native|PROFILE --quant exact|PROFILE --quant-floor bf16|exact --codec none|auto|PROFILE --opt default|size|latency -o OUTPUT [--plan PLAN_PATH] [--dry-run] [--verify] [--force]\\n  logan recompile PACKAGE_DIR (-o OUTPUT | --in-place) [--target source|auto|native|PROFILE] [--optimize (--max-context N | --require-context N) [--plan-choice NAME|ID] [--calibration FILE]] [--quant keep|mxfp4] [--quant-rule SELECTOR=keep|mxfp4]... [--codec keep|none] [--allow-requantize] [--repack] [--verify] [--force]";',
)
crep(
    '            "--optimize" => request.optimize = true,\n',
    '''            "--optimize" => request.optimize = true,\n            "--plan-choice" => {\n                request.plan_choice = Some(value(&mut args, "--plan-choice")?)\n            }\n            "--calibration" => {\n                request.calibration = Some(PathBuf::from(value(&mut args, "--calibration")?))\n            }\n''',
    1,
)
crep(
    '''    if request.context.is_none() {\n        return Err(ColicError::Usage(\n            "compile requires exactly one of --max-context N or --require-context N".into(),\n        ));\n    }\n    Ok(Command::Compile(request))\n''',
    '''    if request.context.is_none() {\n        return Err(ColicError::Usage(\n            "compile requires exactly one of --max-context N or --require-context N".into(),\n        ));\n    }\n    if !request.optimize && (request.plan_choice.is_some() || request.calibration.is_some()) {\n        return Err(ColicError::Usage(\n            "--plan-choice and --calibration require --optimize".into(),\n        ));\n    }\n    Ok(Command::Compile(request))\n''',
)
crep(
    '''    let mut context = None;\n    let mut optimize = false;\n    let mut quant = RecompileQuantMode::Keep;\n''',
    '''    let mut context = None;\n    let mut optimize = false;\n    let mut plan_choice = None;\n    let mut calibration = None;\n    let mut quant = RecompileQuantMode::Keep;\n''',
)
# second --optimize occurrence belongs to recompile
idx = s.find('            "--optimize" => optimize = true,')
assert idx >= 0
s = s[:idx] + s[idx:].replace(
    '            "--optimize" => optimize = true,\n',
    '''            "--optimize" => optimize = true,\n            "--plan-choice" => plan_choice = Some(value(&mut args, "--plan-choice")?),\n            "--calibration" => {\n                calibration = Some(PathBuf::from(value(&mut args, "--calibration")?))\n            }\n''',
    1,
)
crep(
    '''    if !optimize && context.is_some() {\n        return Err(ColicError::Usage(\n            "recompile context options require --optimize so they cannot be silently ignored".into(),\n        ));\n    }\n''',
    '''    if !optimize && context.is_some() {\n        return Err(ColicError::Usage(\n            "recompile context options require --optimize so they cannot be silently ignored".into(),\n        ));\n    }\n    if !optimize && (plan_choice.is_some() || calibration.is_some()) {\n        return Err(ColicError::Usage(\n            "recompile --plan-choice and --calibration require --optimize".into(),\n        ));\n    }\n''',
)
crep(
    '''        context,\n        optimize,\n        allow_requantize,\n''',
    '''        context,\n        optimize,\n        plan_choice,\n        calibration,\n        allow_requantize,\n''',
)
p.write_text(s)
print('patched cli.rs')

# ---------------------------------------------------------------------------
# main.rs interactive/noninteractive picker
# ---------------------------------------------------------------------------
p = Path('logan-compiler/src/main.rs')
s = p.read_text()

def mrep(old, new, count=1):
    global s
    actual = s.count(old)
    assert actual == count, f'main.rs expected {count}, got {actual}: {old[:120]!r}'
    s = s.replace(old, new, count)

insert_before = 'fn main() {\n'
assert s.count(insert_before) == 1
picker = r'''fn print_optimizer_plans(plans: &[logan_ir::ParetoPlan]) {
    for (index, plan) in plans.iter().enumerate() {
        eprintln!(
            "  [{}] {} aliases={} quality={}ppm context={} latency={} resident={} package={} traffic={}",
            index + 1,
            plan.id,
            if plan.labels.is_empty() {
                "-".to_owned()
            } else {
                plan.labels.join(",")
            },
            plan.metrics.quality_loss_ppm,
            plan.metrics.context_tokens,
            plan.metrics.latency_cost,
            human_bytes(plan.metrics.resident_bytes),
            human_bytes(plan.metrics.package_bytes),
            human_bytes(plan.metrics.storage_traffic_bytes),
        );
    }
}

fn choose_optimizer_plan(plans: &[logan_ir::ParetoPlan]) -> logan_compiler::Result<String> {
    if plans.is_empty() {
        return Err(logan_compiler::ColicError::Usage(
            "optimizer produced no selectable plans".into(),
        ));
    }
    print_optimizer_plans(plans);
    if !io::stdin().is_terminal() {
        return Err(logan_compiler::ColicError::Usage(
            "optimized compile/recompile is non-interactive here; pass --plan-choice NAME|ID (for example --plan-choice balanced)".into(),
        ));
    }
    eprint!("select optimizer plan [balanced]: ");
    let _ = io::stderr().flush();
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .map_err(|source| logan_compiler::ColicError::Io {
            path: "<stdin>".into(),
            source,
        })?;
    let input = input.trim();
    if input.is_empty() {
        return logan_ir::select_plan(plans, "balanced")
            .or_else(|| plans.first())
            .map(|plan| plan.id.clone())
            .ok_or_else(|| logan_compiler::ColicError::Usage("no optimizer plan".into()));
    }
    if let Ok(index) = input.parse::<usize>() {
        if let Some(plan) = index.checked_sub(1).and_then(|index| plans.get(index)) {
            return Ok(plan.id.clone());
        }
    }
    logan_ir::select_plan(plans, input)
        .map(|plan| plan.id.clone())
        .ok_or_else(|| logan_compiler::ColicError::Usage(format!("unknown optimizer plan `{input}`")))
}

'''
s = s.replace(insert_before, picker + insert_before, 1)

old_recompile = '''        Command::Recompile(request) => {\n            eprintln!("logan: offline COLI recompilation...");\n            let summary = logan_compiler::recompile::recompile(&request)?;\n'''
new_recompile = '''        Command::Recompile(mut request) => {\n            if request.optimize && request.plan_choice.is_none() {\n                eprintln!("logan: computing non-dominated recompile plans...");\n                let plans = logan_compiler::recompile::preview_optimization(&request)?;\n                request.plan_choice = Some(choose_optimizer_plan(&plans)?);\n            }\n            eprintln!("logan: offline COLI recompilation...");\n            let summary = logan_compiler::recompile::recompile(&request)?;\n'''
mrep(old_recompile, new_recompile)
mrep(
    '''            println!("source_fingerprint={}", summary.source_fingerprint);\n            Ok(())\n''',
    '''            println!("source_fingerprint={}", summary.source_fingerprint);\n            if let Some(plan) = summary.optimizer_plan {\n                println!("optimizer_plan={plan}");\n            }\n            Ok(())\n''',
    1,
)
# Dry-run: print frontier returned with summary.
mrep(
    '''            println!(\n                "projected_padding_bytes={}",\n                summary.plan.projected_padding_bytes\n            );\n            Ok(())\n''',
    '''            println!(\n                "projected_padding_bytes={}",\n                summary.plan.projected_padding_bytes\n            );\n            if !summary.optimizer_plans.is_empty() {\n                eprintln!("logan: non-dominated optimizer plans:");\n                print_optimizer_plans(&summary.optimizer_plans);\n            }\n            Ok(())\n''',
)
old_compile = '''        Command::Compile(request) => {\n            let mut progress = ConsoleProgress::new();\n            if logan_compiler::codec::compile::handles(&request) {\n                logan_compiler::codec::compile::compile(&request, &mut progress)\n            } else {\n                pipeline::compile(&request, &mut progress)\n            }\n        }\n'''
new_compile = '''        Command::Compile(mut request) => {\n            if request.optimize && request.plan_choice.is_none() {\n                eprintln!("logan: computing non-dominated compile plans...");\n                let plans = pipeline::preview_optimization(&request)?;\n                request.plan_choice = Some(choose_optimizer_plan(&plans)?);\n            }\n            let mut progress = ConsoleProgress::new();\n            if logan_compiler::codec::compile::handles(&request) {\n                logan_compiler::codec::compile::compile(&request, &mut progress)\n            } else {\n                pipeline::compile(&request, &mut progress)\n            }\n        }\n'''
mrep(old_compile, new_compile)
p.write_text(s)
print('patched main.rs')
