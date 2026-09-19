use std::{
    io::{self, IsTerminal, Write},
    time::Instant,
};

use logan_compiler::{
    cli::{self, Command, USAGE},
    pipeline::{self, ProgressSink, Stage},
    source::DiscoveryProgress,
    verify::VerificationProgress,
};

const PROGRESS_BAR_WIDTH: usize = 24;

struct ConsoleProgress {
    discovery_started: Option<Instant>,
    emission_started: Option<Instant>,
    verification_started: Option<Instant>,
    interactive: bool,
    active_line_width: usize,
}

impl ConsoleProgress {
    fn new() -> Self {
        Self {
            discovery_started: None,
            emission_started: None,
            verification_started: None,
            interactive: io::stderr().is_terminal(),
            active_line_width: 0,
        }
    }

    fn finish_active_line(&mut self) {
        if self.interactive && self.active_line_width != 0 {
            eprintln!();
            self.active_line_width = 0;
        }
    }

    fn progress_line(&mut self, line: String, complete: bool) {
        if !self.interactive {
            eprintln!("{line}");
            return;
        }

        let padding = self.active_line_width.saturating_sub(line.len());
        eprint!("\r{line}{:padding$}", "");
        let _ = io::stderr().flush();
        self.active_line_width = line.len();
        if complete {
            eprintln!();
            self.active_line_width = 0;
        }
    }

    fn verification(&mut self, update: VerificationProgress) {
        let started = self.verification_started.get_or_insert_with(Instant::now);
        let elapsed = started.elapsed().as_secs_f64();
        let complete = update.completed_records >= update.total_records;
        let percent = progress_percent(update.completed_records, update.total_records);
        let bar = progress_bar(update.completed_records, update.total_records);
        let eta = estimate_eta(update.completed_records, update.total_records, elapsed);
        let throughput = byte_rate(update.verified_bytes, elapsed);
        self.progress_line(
            format!(
                "colic: verify   [{bar}] {percent:5.1}%  {}/{} records  {} checked  {throughput}  ETA {eta}  shard {}/{}",
                update.completed_records,
                update.total_records,
                human_bytes(update.verified_bytes),
                update.current_shard + 1,
                update.total_shards,
            ),
            complete,
        );
    }
}

impl ProgressSink for ConsoleProgress {
    fn stage(&mut self, stage: Stage) {
        self.finish_active_line();
        if stage == Stage::Emission {
            self.emission_started = Some(Instant::now());
        }
        if stage == Stage::Verification {
            self.verification_started = Some(Instant::now());
        }
        eprintln!("colic: {}...", stage.as_str());
    }

    fn emission(&mut self, completed: u64, total: u64, bytes: u64, total_bytes: u64) {
        let elapsed = self
            .emission_started
            .map(|start| start.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let progress_done = if total_bytes != 0 { bytes } else { completed };
        let progress_total = if total_bytes != 0 { total_bytes } else { total };
        let complete = completed >= total && bytes >= total_bytes;
        let percent = progress_percent(progress_done, progress_total);
        let bar = progress_bar(progress_done, progress_total);
        let eta = estimate_eta(progress_done, progress_total, elapsed);
        let throughput = byte_rate(bytes, elapsed);
        self.progress_line(
            format!(
                "colic: emission [{bar}] {percent:5.1}%  {}/{}  {completed}/{total} records  {throughput}  ETA {eta}",
                human_bytes(bytes),
                human_bytes(total_bytes),
            ),
            complete,
        );
    }

    fn source_file(&mut self, update: &DiscoveryProgress) {
        let started = self.discovery_started.get_or_insert_with(Instant::now);
        let elapsed = started.elapsed().as_secs_f64();
        let complete = update.completed_files >= update.total_files;
        let completed = update.completed_files as u64;
        let total = update.total_files as u64;
        let percent = progress_percent(completed, total);
        let bar = progress_bar(completed, total);
        let eta = estimate_eta(completed, total, elapsed);
        self.progress_line(
            format!(
                "colic: source   [{bar}] {percent:5.1}%  {}/{} files  {} ({})  ETA {eta}",
                update.completed_files,
                update.total_files,
                update
                    .path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                human_bytes(update.bytes_hashed),
            ),
            complete,
        );
    }
}

fn progress_percent(completed: u64, total: u64) -> f64 {
    if total == 0 {
        return 100.0;
    }
    100.0 * completed.min(total) as f64 / total as f64
}

fn progress_bar(completed: u64, total: u64) -> String {
    if total == 0 {
        return "=".repeat(PROGRESS_BAR_WIDTH);
    }
    let completed = completed.min(total);
    let filled = ((completed as u128 * PROGRESS_BAR_WIDTH as u128) / total as u128) as usize;
    if filled >= PROGRESS_BAR_WIDTH {
        return "=".repeat(PROGRESS_BAR_WIDTH);
    }
    let mut bar = String::with_capacity(PROGRESS_BAR_WIDTH);
    bar.push_str(&"=".repeat(filled));
    bar.push('>');
    bar.push_str(&"-".repeat(PROGRESS_BAR_WIDTH - filled - 1));
    bar
}

fn estimate_eta(completed: u64, total: u64, elapsed_seconds: f64) -> String {
    if total == 0 || completed >= total {
        return "0s".to_owned();
    }
    if completed == 0 || elapsed_seconds <= 0.0 || !elapsed_seconds.is_finite() {
        return "--".to_owned();
    }
    let remaining = total - completed;
    let seconds = (elapsed_seconds * remaining as f64 / completed as f64).ceil();
    if !seconds.is_finite() || seconds < 0.0 {
        return "--".to_owned();
    }
    format_duration(seconds as u64)
}

fn format_duration(seconds: u64) -> String {
    if seconds >= 3600 {
        format!(
            "{}h {:02}m {:02}s",
            seconds / 3600,
            (seconds % 3600) / 60,
            seconds % 60
        )
    } else if seconds >= 60 {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

fn byte_rate(bytes: u64, elapsed_seconds: f64) -> String {
    if bytes == 0 || elapsed_seconds <= 0.0 || !elapsed_seconds.is_finite() {
        return "-- MiB/s".to_owned();
    }
    format!(
        "{:.1} MiB/s",
        bytes as f64 / elapsed_seconds / (1024.0 * 1024.0)
    )
}

fn human_bytes(bytes: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else {
        format!("{:.2} MiB", bytes as f64 / (1024 * 1024) as f64)
    }
}

fn print_optimizer_plans(plans: &[logan_ir::ParetoPlan]) {
    for (index, plan) in plans.iter().enumerate() {
        eprintln!(
            "  [{}] {} aliases={} quant_loss={}ppm context={} latency={} resident={} package={} traffic={}",
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
        .ok_or_else(|| {
            logan_compiler::ColicError::Usage(format!("unknown optimizer plan `{input}`"))
        })
}

/// Stack for the thread that does the work.
///
/// The platform main thread gets 1 MiB on Windows and 8 MiB on macOS, and this
/// binary overflows that: parsing a real checkpoint's safetensors header (152k
/// tensors for Qwen3.8-Flash-Next) recurses through serde's JSON parser deep
/// enough to blow it. The failure is a bare `thread 'main' has overflowed its
/// stack` with no other diagnostic, which reads like a corrupt model rather
/// than a stack limit.
///
/// 256 MiB is chosen to be obviously sufficient rather than tuned: it is
/// virtual address space, committed only as touched, and the whole point is
/// that this class of failure cannot come back by adding a larger model.
const WORKER_STACK_BYTES: usize = 256 * 1024 * 1024;

fn uses_modern_qwen_runtime(model_type: &str) -> bool {
    matches!(model_type, "qwen3_5_moe" | "qwen4_exp_text" | "qwen4_exp")
}

fn main() {
    // Do the work on a thread with a large stack rather than raising the main
    // thread's: `main`'s stack size is fixed by the platform at process start
    // and cannot be changed from inside the process on Windows.
    let worker = std::thread::Builder::new()
        .stack_size(WORKER_STACK_BYTES)
        .spawn(|| {
            if let Err(error) = run() {
                eprintln!("colic: {error}");
                eprintln!("{USAGE}");
                std::process::exit(2);
            }
        });
    match worker {
        Ok(handle) => {
            // A panic inside `run` is the thread's business; joining propagates
            // it here so the process still exits non-zero.
            if handle.join().is_err() {
                std::process::exit(101);
            }
        }
        Err(error) => {
            // Could not even start a thread: fall back to running inline rather
            // than refusing to work.
            eprintln!(
                "colic: could not start a worker thread ({error}); running on the main stack"
            );
            if let Err(error) = run() {
                eprintln!("colic: {error}");
                eprintln!("{USAGE}");
                std::process::exit(2);
            }
        }
    }
}

fn run() -> logan_compiler::Result<()> {
    match cli::parse(std::env::args().skip(1))? {
        Command::Help => {
            println!("{USAGE}");
            Ok(())
        }
        Command::InspectSource { source } => {
            eprintln!("logan: source discovery...");
            let mut progress = ConsoleProgress::new();
            let inventory =
                logan_compiler::source::discover_with_progress(&source, &mut |update| {
                    progress.source_file(&update);
                })?;
            println!("source={}", inventory.root.display());
            println!("files={}", inventory.files.len());
            println!("tensors={}", inventory.tensors.len());
            println!("source_stored_bytes={}", inventory.source_stored_bytes);
            println!("dtype_counts={:?}", inventory.dtype_counts);
            println!("source_fingerprint={}", inventory.source_fingerprint);
            if let Some(architecture_hint) = &inventory.architecture_hint {
                println!("architecture_hint={architecture_hint}");
            }
            if let Some(config_fingerprint) = &inventory.config_fingerprint {
                println!("config_fingerprint={config_fingerprint}");
            }
            if let Some(model) = pipeline::build_semantic_ir(&inventory)? {
                println!("semantic_architecture={:?}", model.architecture);
                println!("semantic_layers={}", model.geometry.layers);
                println!("semantic_routed_experts={}", model.routed_experts.len());
                println!(
                    "semantic_static_layers={}",
                    model.layer_static_tensors.len()
                );
                println!("semantic_resident_tensors={}", model.resident_tensors.len());
            }
            Ok(())
        }
        Command::AttachMtp { package, drafter } => {
            eprintln!("logan: attaching Qwen4Exp MTP drafter...");
            let summary = logan_compiler::mtp_attach::attach_qwen4_mtp(&package, &drafter)?;
            println!("package={}", summary.package.display());
            println!("drafter={}", summary.drafter.display());
            println!("stages={}", summary.stages);
            println!("experts={}", summary.experts);
            println!("static_tensors={}", summary.static_tensors);
            println!("virtual_layer_base={}", summary.virtual_layer_base);
            println!("new_records={}", summary.new_records);
            println!("new_shards={}", summary.new_shards);
            println!("added_stored_bytes={}", summary.added_stored_bytes);
            println!("backup_manifest={}", summary.backup_manifest.display());
            Ok(())
        }
        Command::ExportExperts {
            source,
            output_dir,
            verify,
            resume,
        } => {
            eprintln!("logan: quantizing routed experts to MXFP4 (one file per layer)...");
            let started = Instant::now();
            let mut last_layer = u32::MAX;
            let mut report = logan_compiler::export::export_experts(
                &source,
                &output_dir,
                resume,
                &mut |done, total, layer| {
                    if layer != last_layer {
                        last_layer = layer;
                        eprintln!("  layer {layer}: {done}/{total} experts");
                    }
                },
            )?;
            println!("output_dir={}", report.output_dir.display());
            println!("file_template={}", report.file_template);
            println!("total_bytes={}", report.total_bytes);
            println!("layers={}", report.layers);
            println!("experts_per_layer={}", report.experts_per_layer);
            println!("experts_exported={}", report.experts_exported);
            if let Some(first) = report.layer_bytes.first() {
                println!("bytes_per_layer={first}");
            }
            if !report.source_dtypes.is_empty() {
                let dtypes = report
                    .source_dtypes
                    .iter()
                    .map(|(d, n)| format!("{d}:{n}"))
                    .collect::<Vec<_>>()
                    .join(",");
                println!("source_expert_dtypes={dtypes}");
            }
            eprintln!(
                "logan: exported {} experts in {} layer file(s), {:.1} GB, in {:.1}s",
                report.experts_exported,
                report.layers,
                report.total_bytes as f64 / 1e9,
                started.elapsed().as_secs_f64()
            );

            if verify {
                // Read every emitted file back and prove the bytes decode.
                // This catches what matters: a file that parses but whose
                // offsets, shapes or nibble order are wrong.
                //
                // Every expert of every projection in every layer is checked: a
                // per-layer file is small enough that sampling would only hide
                // which layer is wrong. Layers are verified in parallel, and
                // each tensor is read once rather than once per expert.
                eprintln!("logan: verifying exported files...");
                let started_verify = Instant::now();
                let base = logan_compiler::export::qwen4_expert_base(&source)?;
                let checked = logan_compiler::export::verify_exported_files(
                    &report.output_dir,
                    &report.file_template,
                    report.layers,
                    report.experts_per_layer,
                    &base,
                )?;
                eprintln!(
                    "logan: verified all {checked} expert projections read back intact in {:.1}s",
                    started_verify.elapsed().as_secs_f64()
                );
            }
            Ok(())
        }
        Command::Verify { package } => {
            eprintln!("logan: verification...");
            let mut progress = ConsoleProgress::new();
            progress.verification_started = Some(Instant::now());
            let summary =
                logan_compiler::verify::verify_package_with_progress(&package, &mut |update| {
                    progress.verification(update);
                })?;
            logan_compiler::verify_target::verify_target_layouts(&package)?;
            println!("package={}", package.display());
            println!("shards={}", summary.shards);
            println!("records={}", summary.records);
            Ok(())
        }
        Command::Recompile(mut request) => {
            if request.optimize && request.plan_choice.is_none() {
                eprintln!("logan: computing non-dominated recompile plans...");
                let plans = logan_compiler::recompile::preview_optimization(&request)?;
                request.plan_choice = Some(choose_optimizer_plan(&plans)?);
            }
            eprintln!("logan: offline COLI recompilation...");
            let summary = logan_compiler::recompile::recompile(&request)?;
            println!("source_profile={}", summary.source_profile);
            println!("target_profile={}", summary.target_profile);
            println!("records={}", summary.records);
            println!("copied_records={}", summary.copied_records);
            println!("rewritten_experts={}", summary.rewritten_experts);
            println!("requantized_experts={}", summary.requantized_experts);
            println!("source_fingerprint={}", summary.source_fingerprint);
            if let Some(plan) = summary.optimizer_plan {
                println!("optimizer_plan={plan}");
            }
            Ok(())
        }
        Command::Run {
            package,
            prompt,
            max_new,
        } => {
            let prompt_ids: Vec<u32> = prompt
                .split_whitespace()
                .map(|t| {
                    t.parse().unwrap_or_else(|_| {
                        eprintln!("logan: invalid token id: {t}");
                        std::process::exit(2);
                    })
                })
                .collect();
            // Architecture dispatch (engine-neutral): read the package's
            // config.json model_type and hand the decode to the matching
            // engine crate. Both engines share the core (storage, LRU,
            // Metal backends, telemetry) — this is the neutral-CLI seam.
            let cfg_path = package.join("config.json");
            let model_type: String = std::fs::read_to_string(&cfg_path)
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .and_then(|v| {
                    v.get("model_type")
                        .and_then(|t| t.as_str())
                        .map(str::to_owned)
                })
                .unwrap_or_else(|| "qwen4_exp_text".to_string());
            let out = match model_type.as_str() {
                #[cfg(feature = "runtime")]
                "llama" => {
                    logan_llama::run_greedy(&package, &prompt_ids, max_new).map_err(|e| {
                        logan_compiler::ColicError::Unsupported {
                            stage: "run",
                            detail: e,
                        }
                    })?
                }
                #[cfg(feature = "runtime")]
                "spark2_5" => {
                    logan_spark::run_greedy(&package, &prompt_ids, max_new).map_err(|e| {
                        logan_compiler::ColicError::Unsupported {
                            stage: "run",
                            detail: e,
                        }
                    })?
                }
                #[cfg(feature = "runtime")]
                model_type if uses_modern_qwen_runtime(model_type) => {
                    let cfg = logan_qwen4::load_cfg(&cfg_path).map_err(|e| {
                        logan_compiler::ColicError::Unsupported {
                            stage: "run",
                            detail: e,
                        }
                    })?;
                    // A safetensors checkpoint directory loads through
                    // `run_greedy` (StFile::open_dir). A compiled .coli package
                    // takes the cached-coli path, which is the only one that
                    // manages a package's shards and prefix cache.
                    //
                    // Chosen by what is on disk rather than by model_type,
                    // because the same architecture arrives in both layouts and
                    // the layout -- not the arch -- decides which loader can
                    // read it.
                    if package.join("model.safetensors.index.json").is_file()
                        || package.join("model.safetensors").is_file()
                    {
                        logan_qwen4::run_greedy(&package, &prompt_ids, max_new).map_err(|e| {
                            logan_compiler::ColicError::Unsupported {
                                stage: "run",
                                detail: e,
                            }
                        })?
                    } else {
                        logan_qwen4::plan::run_greedy_cached_coli(
                            &package,
                            &cfg,
                            &prompt_ids,
                            max_new,
                        )
                        .map_err(|e| {
                            logan_compiler::ColicError::Unsupported {
                                stage: "run",
                                detail: e,
                            }
                        })?
                    }
                }
                #[cfg(feature = "runtime")]
                _ => {
                    // Default to the Qwen3 MoE engine for other qwen model
                    // types; unknown architectures are reported honestly.
                    match logan_qwen::run_greedy(&package, &prompt_ids, max_new) {
                        Ok(o) => o,
                        Err(qwen_err) => {
                            return Err(logan_compiler::ColicError::Unsupported {
                                stage: "run",
                                detail: format!("model_type={model_type}: {qwen_err}"),
                            });
                        }
                    }
                }
                // Built without the runtime feature: say exactly that, rather
                // than reporting every architecture as unsupported.
                #[cfg(not(feature = "runtime"))]
                _ => {
                    return Err(logan_compiler::ColicError::Unsupported {
                        stage: "run",
                        detail: format!(
                            "this build has no runtime backend (model_type={model_type}); \
                             rebuild with the `runtime` feature to use `logan run`"
                        ),
                    });
                }
            };
            println!("generated: {out:?}");
            Ok(())
        }
        Command::Compile(request) if request.dry_run => {
            let summary = if logan_compiler::codec::compile::handles(&request) {
                logan_compiler::codec::compile::dry_run(&request)?
            } else {
                pipeline::dry_run(&request)?
            };
            println!("target={}", summary.target_name);
            println!("source_tensors={}", summary.source_tensors);
            println!("source_stored_bytes={}", summary.source_stored_bytes);
            println!("projected_record_count={}", summary.plan.records.len());
            println!("projected_shard_count={}", summary.plan.shards);
            println!(
                "projected_stored_bytes={}",
                summary.plan.projected_stored_bytes
            );
            println!(
                "projected_padding_bytes={}",
                summary.plan.projected_padding_bytes
            );
            if !summary.optimizer_plans.is_empty() {
                eprintln!("logan: non-dominated optimizer plans:");
                print_optimizer_plans(&summary.optimizer_plans);
            }
            Ok(())
        }
        Command::Compile(mut request) => {
            if request.optimize && request.plan_choice.is_none() {
                eprintln!("logan: computing non-dominated compile plans...");
                let plans = pipeline::preview_optimization(&request)?;
                request.plan_choice = Some(choose_optimizer_plan(&plans)?);
            }
            let mut progress = ConsoleProgress::new();
            if logan_compiler::codec::compile::handles(&request) {
                logan_compiler::codec::compile::compile(&request, &mut progress)
            } else {
                pipeline::compile(&request, &mut progress)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_bar_has_stable_width_and_finishes_cleanly() {
        assert_eq!(progress_bar(0, 100).len(), PROGRESS_BAR_WIDTH);
        assert_eq!(progress_bar(50, 100).len(), PROGRESS_BAR_WIDTH);
        assert_eq!(progress_bar(100, 100), "=".repeat(PROGRESS_BAR_WIDTH));
        assert_eq!(progress_bar(200, 100), "=".repeat(PROGRESS_BAR_WIDTH));
    }

    #[test]
    fn eta_formats_human_scale_time() {
        assert_eq!(estimate_eta(0, 100, 10.0), "--");
        assert_eq!(estimate_eta(50, 100, 65.0), "1m 05s");
        assert_eq!(estimate_eta(25, 100, 1225.0), "1h 01m 15s");
        assert_eq!(estimate_eta(100, 100, 10.0), "0s");
    }

    #[test]
    fn byte_rate_handles_unavailable_and_known_rates() {
        assert_eq!(byte_rate(0, 1.0), "-- MiB/s");
        assert_eq!(byte_rate(10 * 1024 * 1024, 2.0), "5.0 MiB/s");
    }

    #[test]
    fn qwen35_moe_uses_modern_qwen_runtime() {
        assert!(uses_modern_qwen_runtime("qwen3_5_moe"));
        assert!(uses_modern_qwen_runtime("qwen4_exp_text"));
        assert!(uses_modern_qwen_runtime("qwen4_exp"));
        assert!(!uses_modern_qwen_runtime("spark2_5"));
    }
}
