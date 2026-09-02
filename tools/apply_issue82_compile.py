from pathlib import Path


def replace(path, old, new, count=1):
    p = Path(path)
    text = p.read_text()
    actual = text.count(old)
    if actual < count:
        raise SystemExit(f"{path}: expected at least {count} matches, found {actual}: {old[:100]!r}")
    text = text.replace(old, new, count)
    p.write_text(text)

# CLI
replace("logan-compiler/src/cli.rs",
'''pub const USAGE: &str = "Usage:\\n  logan inspect-source MODEL_DIR\\n  logan verify PACKAGE_DIR\\n  logan compile MODEL_DIR (--max-context N | --require-context N) [--optimize] --target auto|native|PROFILE --quant exact|PROFILE --quant-floor bf16|exact --codec none|auto|PROFILE --opt default|size|latency -o OUTPUT [--plan PLAN_PATH] [--dry-run] [--verify] [--force]\\n  logan recompile PACKAGE_DIR (-o OUTPUT | --in-place) [--target source|auto|native|PROFILE] [--optimize (--max-context N | --require-context N)] [--quant keep|mxfp4] [--quant-rule SELECTOR=keep|mxfp4]... [--codec keep|none] [--allow-requantize] [--repack] [--verify] [--force]";''',
'''pub const USAGE: &str = "Usage:\\n  logan inspect-source MODEL_DIR\\n  logan verify PACKAGE_DIR\\n  logan compile MODEL_DIR (--max-context N | --require-context N) [--optimize] [--select-plan quality|balanced|long-context|latency|PLAN_ID] --target auto|native|PROFILE --quant exact|PROFILE --quant-floor bf16|exact --codec none|auto|PROFILE --opt default|size|latency -o OUTPUT [--plan PLAN_PATH] [--dry-run] [--verify] [--force]\\n  logan recompile PACKAGE_DIR (-o OUTPUT | --in-place) [--target source|auto|native|PROFILE] [--optimize (--max-context N | --require-context N)] [--select-plan quality|balanced|long-context|latency|PLAN_ID] [--quant keep|mxfp4] [--quant-rule SELECTOR=keep|mxfp4]... [--codec keep|none] [--allow-requantize] [--repack] [--verify] [--force]";''')
replace("logan-compiler/src/cli.rs",
'''            "--quant" => request.quant = QuantRequest::parse(&value(&mut args, "--quant")?)?,''',
'''            "--quant" => {
                request.quant = QuantRequest::parse(&value(&mut args, "--quant")?)?;
                request.quant_explicit = true;
            }''')
replace("logan-compiler/src/cli.rs",
'''            "--optimize" => request.optimize = true,
            "--max-context" => {''',
'''            "--optimize" => request.optimize = true,
            "--select-plan" => request.plan_choice = Some(value(&mut args, "--select-plan")?),
            "--max-context" => {''', 1)
replace("logan-compiler/src/cli.rs",
'''    if request.context.is_none() {
        return Err(ColicError::Usage(
            "compile requires exactly one of --max-context N or --require-context N".into(),
        ));
    }
    Ok(Command::Compile(request))''',
'''    if request.context.is_none() {
        return Err(ColicError::Usage(
            "compile requires exactly one of --max-context N or --require-context N".into(),
        ));
    }
    if request.plan_choice.is_some() && !request.optimize {
        return Err(ColicError::Usage(
            "--select-plan requires --optimize".into(),
        ));
    }
    Ok(Command::Compile(request))''')

# Pipeline imports and request fields.
replace("logan-compiler/src/pipeline.rs",
'''use std::{collections::BTreeMap, fs, io::Read, path::PathBuf};''',
'''use std::{
    collections::BTreeMap,
    fs,
    io::{self, IsTerminal, Read, Write},
    path::PathBuf,
};''')
replace("logan-compiler/src/pipeline.rs",
'''    plan::{MemoryPlan, Placement, PlanArtifact, QuantSpec},''',
'''    plan::{MemoryPlan, OptimizationRecord, Placement, PlanArtifact, QuantSpec},''')
replace("logan-compiler/src/pipeline.rs",
'''    model::qwen_moe::QwenMoeFrontend,
    quant::mxfp4_record,''',
'''    model::qwen_moe::QwenMoeFrontend,
    optimizer::{self, ExpertRepresentation, SelectedOptimization},
    quant::mxfp4_record,''')
replace("logan-compiler/src/pipeline.rs",
'''    pub quant: QuantRequest,
    pub codec: CodecRequest,''',
'''    pub quant: QuantRequest,
    /// Whether --quant was explicitly supplied; in optimize mode an explicit
    /// quant becomes a hard expert-representation constraint.
    pub quant_explicit: bool,
    pub codec: CodecRequest,''')
replace("logan-compiler/src/pipeline.rs",
'''    pub optimize: bool,
    pub dry_run: bool,''',
'''    pub optimize: bool,
    /// Named Pareto label or stable plan id for non-interactive selection.
    pub plan_choice: Option<String>,
    pub dry_run: bool,''')
replace("logan-compiler/src/pipeline.rs",
'''            quant: QuantRequest::Exact,
            codec: CodecRequest::None,''',
'''            quant: QuantRequest::Exact,
            quant_explicit: false,
            codec: CodecRequest::None,''')
replace("logan-compiler/src/pipeline.rs",
'''            optimize: false,
            dry_run: false,''',
'''            optimize: false,
            plan_choice: None,
            dry_run: false,''')
replace("logan-compiler/src/pipeline.rs",
'''enum ExpertQuantization {''',
'''pub(crate) enum ExpertQuantization {''')

# Per-expert quant decisions.
replace("logan-compiler/src/pipeline.rs",
'''fn record_inventory(
    model: &SemanticModel,
    expert_quantization: ExpertQuantization,
    target_profile: target::TargetProfile,
) -> Result<Vec<LoweredRecord>> {''',
'''fn record_inventory(
    model: &SemanticModel,
    expert_quantization: ExpertQuantization,
    optimization: Option<&SelectedOptimization>,
    target_profile: target::TargetProfile,
) -> Result<Vec<LoweredRecord>> {''')
replace("logan-compiler/src/pipeline.rs",
'''    for expert in model.routed_experts.values() {
        let (stored_bytes, decoded_bytes) = if target_profile == target::MACOS_ARM64_METAL_APPLE8_V1''',
'''    for expert in model.routed_experts.values() {
        let expert_quantization = optimization
            .map(|selected| match selected.representation_for(expert.layer, expert.expert) {
                ExpertRepresentation::Exact => ExpertQuantization::Exact,
                ExpertRepresentation::Mxfp4 => ExpertQuantization::Mxfp4,
            })
            .unwrap_or(expert_quantization);
        let (stored_bytes, decoded_bytes) = if target_profile == target::MACOS_ARM64_METAL_APPLE8_V1''')
replace("logan-compiler/src/pipeline.rs",
'''        ExpertQuantization::Exact,
        target::LINUX_X86_64_AVX2_V1,
    )''',
'''        ExpertQuantization::Exact,
        None,
        target::LINUX_X86_64_AVX2_V1,
    )''')
replace("logan-compiler/src/pipeline.rs",
'''fn exact_sources(
    model: &SemanticModel,
    expert_quantization: ExpertQuantization,
) -> Vec<ExactSource> {''',
'''fn exact_sources(
    model: &SemanticModel,
    expert_quantization: ExpertQuantization,
    optimization: Option<&SelectedOptimization>,
) -> Vec<ExactSource> {''')
replace("logan-compiler/src/pipeline.rs",
'''            .map(|expert| ExactSource::Expert {
                expert: Box::new(expert),
                quantization: expert_quantization,
            }),''',
'''            .map(|expert| {
                let quantization = optimization
                    .map(|selected| match selected.representation_for(expert.layer, expert.expert) {
                        ExpertRepresentation::Exact => ExpertQuantization::Exact,
                        ExpertRepresentation::Mxfp4 => ExpertQuantization::Mxfp4,
                    })
                    .unwrap_or(expert_quantization);
                ExactSource::Expert {
                    expert: Box::new(expert),
                    quantization,
                }
            }),''')

# Physical plan derives representation from each source expert now.
replace("logan-compiler/src/pipeline.rs",
'''    records: &[LoweredRecord],
    expert_quantization: ExpertQuantization,
    target: target::TargetProfile,''',
'''    records: &[LoweredRecord],
    target: target::TargetProfile,''')
replace("logan-compiler/src/pipeline.rs",
'''            ExactSource::Expert { .. } => {
                if target == target::MACOS_ARM64_METAL_APPLE8_V1 {
                    KIND_APPLE8
                } else {
                    match expert_quantization {
                        ExpertQuantization::Exact => KIND_EXACT,
                        ExpertQuantization::Mxfp4 => "mxfp4",
                    }
                }
            }''',
'''            ExactSource::Expert { quantization, .. } => {
                if target == target::MACOS_ARM64_METAL_APPLE8_V1 {
                    KIND_APPLE8
                } else {
                    match quantization {
                        ExpertQuantization::Exact => KIND_EXACT,
                        ExpertQuantization::Mxfp4 => "mxfp4",
                    }
                }
            }''')

# Add optimizer selection helpers before compile().
marker = '''pub fn compile(request: &CompileRequest, progress: &mut dyn ProgressSink) -> Result<()> {'''
insert = r'''fn requested_expert_constraint(request: &CompileRequest) -> Result<Option<ExpertRepresentation>> {
    if !request.quant_explicit {
        return Ok(None);
    }
    match &request.quant {
        QuantRequest::Exact => Ok(Some(ExpertRepresentation::Exact)),
        QuantRequest::Profile(profile) if profile == "mxfp4" => Ok(Some(ExpertRepresentation::Mxfp4)),
        QuantRequest::Profile(profile) => Err(ColicError::unsupported(
            Stage::TargetPlanning.as_str(),
            format!("quantization profile `{profile}` is not implemented"),
        )),
    }
}

fn choose_optimizer_plan(plans: &[logan_ir::ParetoPlan], requested: Option<&str>) -> Result<String> {
    if let Some(requested) = requested {
        return Ok(requested.to_owned());
    }
    if !io::stdin().is_terminal() {
        return Err(ColicError::Usage(format!(
            "--optimize needs --select-plan when stdin is not interactive; available: {}",
            plans.iter().map(|plan| format!("{} ({})", plan.label.as_deref().unwrap_or("unlabeled"), plan.id)).collect::<Vec<_>>().join(", ")
        )));
    }
    eprintln!("Logan found {} non-dominated deployment plans:", plans.len());
    for (index, plan) in plans.iter().enumerate() {
        let m = plan.metrics;
        eprintln!(
            "  {}. {:<12} {}  context={}  quality={}  latency={}  memory={} MiB  package={} MiB",
            index + 1,
            plan.label.as_deref().unwrap_or("unlabeled"),
            plan.id,
            m.context_tokens,
            m.quality_loss,
            m.latency_cost,
            m.resident_bytes / (1024 * 1024),
            m.package_bytes / (1024 * 1024),
        );
    }
    eprint!("Select plan [1-{}] (default balanced): ", plans.len());
    io::stderr().flush().map_err(|source| ColicError::Io { path: PathBuf::from("<stderr>"), source })?;
    let mut line = String::new();
    io::stdin().read_line(&mut line).map_err(|source| ColicError::Io { path: PathBuf::from("<stdin>"), source })?;
    let choice = line.trim();
    if choice.is_empty() {
        if let Some(plan) = plans.iter().find(|plan| plan.label.as_deref() == Some("balanced")) {
            return Ok(plan.id.clone());
        }
        return Ok(plans[0].id.clone());
    }
    if let Ok(index) = choice.parse::<usize>() {
        if (1..=plans.len()).contains(&index) {
            return Ok(plans[index - 1].id.clone());
        }
    }
    Ok(choice.to_owned())
}

fn resolve_optimization(
    request: &CompileRequest,
    model: &SemanticModel,
    target_profile: target::TargetProfile,
    machine: &target::MachineProfile,
) -> Result<Option<SelectedOptimization>> {
    if !request.optimize {
        return Ok(None);
    }
    let context = request.context.ok_or_else(|| ColicError::Usage(
        "--optimize requires a context constraint".into(),
    ))?;
    let forced = requested_expert_constraint(request)?;
    let (plans, members) = optimizer::build_plans(model, target_profile, machine, context, forced)?;
    let selector = choose_optimizer_plan(&plans, request.plan_choice.as_deref())?;
    let selected = optimizer::select(plans, members, &selector)?;
    eprintln!(
        "selected optimizer plan {} ({})",
        selected.selected.label.as_deref().unwrap_or("unlabeled"),
        selected.selected.id
    );
    Ok(Some(selected))
}

'''
replace("logan-compiler/src/pipeline.rs", marker, insert + marker)

# Dry run planning.
replace("logan-compiler/src/pipeline.rs",
'''    let quantization = resolve_expert_quantization(request, &model)?;
    let target_profile = target::resolve(&request.target, &target::MachineProfile::probe())?;
    let records = record_inventory(&model, quantization, target_profile)?;''',
'''    let machine = target::MachineProfile::probe();
    let target_profile = target::resolve(&request.target, &machine)?;
    let optimization = resolve_optimization(request, &model, target_profile, &machine)?;
    let quantization = if request.optimize {
        ExpertQuantization::Exact
    } else {
        resolve_expert_quantization(request, &model)?
    };
    let records = record_inventory(&model, quantization, optimization.as_ref(), target_profile)?;''')

# Compile path.
replace("logan-compiler/src/pipeline.rs",
'''    let expert_quantization = resolve_expert_quantization(request, &model)?;
    progress.stage(Stage::TargetPlanning);
    let machine = target::MachineProfile::probe();
    let target_profile = target::resolve(&request.target, &machine)?;''',
'''    progress.stage(Stage::TargetPlanning);
    let machine = target::MachineProfile::probe();
    let target_profile = target::resolve(&request.target, &machine)?;
    let optimization = resolve_optimization(request, &model, target_profile, &machine)?;
    let expert_quantization = if request.optimize {
        ExpertQuantization::Exact
    } else {
        resolve_expert_quantization(request, &model)?
    };''')
replace("logan-compiler/src/pipeline.rs",
'''    let sources = exact_sources(&model, expert_quantization);
    let records = record_inventory(&model, expert_quantization, target_profile)?;''',
'''    let sources = exact_sources(&model, expert_quantization, optimization.as_ref());
    let records = record_inventory(
        &model,
        expert_quantization,
        optimization.as_ref(),
        target_profile,
    )?;''')
replace("logan-compiler/src/pipeline.rs",
'''            &records,
            expert_quantization,
            target_profile,''',
'''            &records,
            target_profile,''')
replace("logan-compiler/src/pipeline.rs",
'''        fs::write(plan_path, plan.to_bytes().map_err(ColicError::Usage)?).map_err(|source| ColicError::Io {''',
'''        let plan = if let Some(optimization) = &optimization {
            plan.with_optimization(OptimizationRecord {
                cost_model_version: optimizer::COST_MODEL_VERSION.into(),
                selected: optimization.selected.clone(),
                alternatives: optimization.frontier.clone(),
            })
        } else {
            plan
        };
        fs::write(plan_path, plan.to_bytes().map_err(ColicError::Usage)?).map_err(|source| ColicError::Io {''')

# validate_supported_options: remove old fail-closed optimize block.
replace("logan-compiler/src/pipeline.rs",
'''    if request.optimize {
        return Err(ColicError::unsupported(
            Stage::TargetPlanning.as_str(),
            "--optimize is wired to the shared context contract, but mixed-representation Pareto search is not implemented yet (see #82); refusing to pretend a fixed quant is optimized",
        ));
    }
''', '')
replace("logan-compiler/src/pipeline.rs",
'''    if request.plan.is_some() && request.output.is_none() {''',
'''    if request.plan_choice.is_some() && !request.optimize {
        return Err(ColicError::Usage("--select-plan requires --optimize".into()));
    }
    if request.plan.is_some() && request.output.is_none() {''')

# Fix remaining record_inventory/exact_sources test callers by adding None.
text = Path("logan-compiler/src/pipeline.rs").read_text()
text = text.replace("record_inventory(&model, ExpertQuantization::Exact, target)", "record_inventory(&model, ExpertQuantization::Exact, None, target)")
text = text.replace("record_inventory(&model, ExpertQuantization::Mxfp4, target)", "record_inventory(&model, ExpertQuantization::Mxfp4, None, target)")
text = text.replace("exact_sources(&model, ExpertQuantization::Exact)", "exact_sources(&model, ExpertQuantization::Exact, None)")
text = text.replace("exact_sources(&model, ExpertQuantization::Mxfp4)", "exact_sources(&model, ExpertQuantization::Mxfp4, None)")
Path("logan-compiler/src/pipeline.rs").write_text(text)

# Keep tests explicit about quant flag state when checking parser.
replace("logan-compiler/src/cli.rs",
'''        assert_eq!(request.quant, QuantRequest::Exact);
        assert_eq!(request.codec, CodecRequest::None);''',
'''        assert_eq!(request.quant, QuantRequest::Exact);
        assert!(request.quant_explicit);
        assert_eq!(request.codec, CodecRequest::None);''')
