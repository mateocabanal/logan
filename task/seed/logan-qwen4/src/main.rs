//! qwen4-rs: greedy decode on the tiny Qwen4 fixture, gated against
//! `ref.json` greedy_new_ids (short + mixed cases). Also loads `.coli`
//! packages (Apple8/MXFP4) via colibri-format.

use std::path::Path;

use logan_qwen4::{load_cfg, Model, StFile};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn apply_apple_runtime_defaults() {
    // Measured on the M2 Qwen3.8-Flash-Next Apple8 package: Accelerate/BNNS
    // BF16 dense execution beats the current Metal GDN path while preserving
    // greedy token identity. Keep both knobs overridable for A/B and fallback.
    if std::env::var_os("QWEN_BNNS_BF16").is_none() {
        std::env::set_var("QWEN_BNNS_BF16", "1");
    }
    if std::env::var_os("QWEN_GDN_MXFP4_FULL").is_none() {
        std::env::set_var("QWEN_GDN_MXFP4_FULL", "1");
    }
    if std::env::var_os("QWEN_SHARED_MXFP4_FULL").is_none() {
        std::env::set_var("QWEN_SHARED_MXFP4_FULL", "1");
    }
    if std::env::var_os("QWEN_GDN_METAL").is_none() {
        // The current generic BF16 Metal GDN path synchronously submits and
        // waits once per GDN layer and is substantially slower than BNNS at
        // decode batch S=1 on this M2. A future qualified FP8 dense island is
        // selected by package capability through its own policy; do not force
        // this generic path on merely because Metal is available.
        std::env::set_var("QWEN_GDN_METAL", "0");
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn apply_apple_runtime_defaults() {}

fn main() {
    apply_apple_runtime_defaults();

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: qwen4-rs <fixture-dir | .coli-package>");
        std::process::exit(2);
    }
    let dir = Path::new(&args[1]);
    let cfg = load_cfg(&dir.join("config.json")).unwrap_or_else(|e| {
        eprintln!("config error: {e}");
        std::process::exit(1);
    });
    // Three layouts, and the order matters:
    //   * a sharded safetensors checkpoint (model.safetensors.index.json)
    //   * a single-file safetensors fixture (model.safetensors)
    //   * a compiled .coli package
    //
    // The test was `!model.safetensors.exists()`, which misreads a SHARDED
    // checkpoint as a .coli package: a 131-shard export has no single
    // model.safetensors, so it took the .coli branch and died looking for
    // `manifest.coli`. That is the exact model this anchor exists to serve
    // (Qwen3.8-Flash-Next-FP8), so the bug read as "no loadable model on the
    // anchor host". `StFile::open_dir` already handles the sharded case --
    // including the HF backbone prefix alias -- so the fix is to ask whether
    // either safetensors layout is present.
    let has_sharded = dir.join("model.safetensors.index.json").exists();
    let has_single = dir.join("model.safetensors").exists();
    let is_safetensors = has_sharded || has_single;
    let is_coli = !is_safetensors;
    let ref_path = dir.join("ref.json");
    if is_coli && !ref_path.exists() {
        let prompt: Vec<u32> = std::env::var("QWEN_PROMPT")
            .unwrap_or_else(|_| "1 2 3 4 5".into())
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        let max_new: usize = std::env::var("QWEN_MAX_NEW")
            .unwrap_or_else(|_| "8".into())
            .parse()
            .unwrap();
        let t0 = std::time::Instant::now();
        let out = if std::env::var("QWEN_SCHED")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            logan_qwen4::scheduled::run_greedy_scheduled(dir, &prompt, max_new).unwrap_or_else(
                |e| {
                    eprintln!("scheduled decode error: {e}");
                    std::process::exit(1);
                },
            )
        } else {
            // Canonical direct path. When QWEN_PREFIX_CACHE=1 (or an explicit
            // LOGAN_PREFIX_CACHE_DIR is supplied), this restores the longest
            // previously persisted request-prefix and evaluates only the
            // uncached suffix. Cache failures reload/replay from a fresh model.
            logan_qwen4::plan::run_greedy_cached_coli(dir, &cfg, &prompt, max_new).unwrap_or_else(
                |e| {
                    eprintln!("decode error: {e}");
                    std::process::exit(1);
                },
            )
        };
        if logan_core::telemetry::enabled() {
            eprintln!(
                "logan qwen4: tokens={} total={:.1} ms/tok",
                out.len(),
                t0.elapsed().as_secs_f64() * 1e3 / out.len().max(1) as f64
            );
        }
        println!("generated: {out:?}");
        return;
    }
    let model = if is_coli {
        let src = logan_qwen4::colisource::ColiSource::open(dir).unwrap_or_else(|e| {
            eprintln!("coli error: {e}");
            std::process::exit(1);
        });
        Model::load_coli(&src, &cfg).unwrap_or_else(|e| {
            eprintln!("model error: {e}");
            std::process::exit(1);
        })
    } else {
        // `open_dir` subsumes both safetensors layouts: it reads the index when
        // one exists and falls back to model.safetensors otherwise.
        let st = StFile::open_dir(dir).unwrap_or_else(|e| {
            eprintln!("safetensors error: {e}");
            std::process::exit(1);
        });
        Model::load(&st, &cfg).unwrap_or_else(|e| {
            eprintln!("model error: {e}");
            std::process::exit(1);
        })
    };

    // A model WITHOUT ref.json is a real checkpoint, not a fixture: there is no
    // oracle to gate against, and the fixture cases below would panic on the
    // missing file. Run it on QWEN_PROMPT instead -- the same entry point the
    // .coli branch uses -- so a sharded FP8 checkpoint can actually generate.
    //
    // This is the path the anchor takes: Qwen3.8-Flash-Next-FP8 is 131 shards
    // with no ref.json, and before this it loaded the whole model and then died
    // demanding a fixture file.
    if !ref_path.exists() {
        let prompt: Vec<u32> = std::env::var("QWEN_PROMPT")
            .unwrap_or_else(|_| "1 2 3 4 5".into())
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        let max_new: usize = std::env::var("QWEN_MAX_NEW")
            .unwrap_or_else(|_| "8".into())
            .parse()
            .unwrap();
        let t0 = std::time::Instant::now();
        let mut model = model;
        // Prompt tokens except the last are prefilled without logits; the final
        // prompt forward is what predicts the first new token. Refeeding it
        // would append it to the recurrent/KV state twice.
        for (i, &t) in prompt.iter().enumerate() {
            if i + 1 == prompt.len() {
                break;
            }
            model.prefill_token(t as usize, i);
        }
        let mut logits = model.forward_token(*prompt.last().unwrap() as usize, prompt.len() - 1);
        let mut out: Vec<u32> = Vec::with_capacity(max_new);
        for step in 0..max_new {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            out.push(next);
            if step + 1 < max_new {
                logits = model.forward_token(next as usize, prompt.len() + step);
            }
        }
        if logan_core::telemetry::enabled() {
            eprintln!(
                "logan qwen4: tokens={} total={:.1} ms/tok",
                out.len(),
                t0.elapsed().as_secs_f64() * 1e3 / out.len().max(1) as f64
            );
        }
        println!("generated: {out:?}");
        return;
    }

    let ref_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ref_path).expect("ref.json"))
            .expect("ref.json parse");
    let cases = ref_json["cases"].as_object().expect("cases");

    let mut all_pass = true;
    for case_name in ["short", "mixed"] {
        let case = cases[case_name].as_object().unwrap();
        let prompt: Vec<u32> = case["prompt_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let expected: Vec<u32> = case["greedy_new_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let max_new = case["max_new_tokens"].as_u64().unwrap() as usize;

        // Fresh model per case (state reset), same source as the main load.
        let model = if is_coli {
            let src = logan_qwen4::colisource::ColiSource::open(dir).unwrap();
            Model::load_coli(&src, &cfg).unwrap()
        } else {
            let st = StFile::open_dir(dir).unwrap();
            Model::load(&st, &cfg).unwrap()
        };
        let mut model = model;
        for (i, &t) in prompt.iter().enumerate() {
            model.forward_token(t as usize, i);
        }
        let mut generated: Vec<u32> = Vec::new();
        let mut last = *prompt.last().unwrap();
        for pos in prompt.len()..prompt.len() + max_new {
            let logits = model.forward_token(last as usize, pos);
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            generated.push(next);
            last = next;
        }
        let pass = generated == expected;
        all_pass &= pass;
        println!(
            "case {case_name}: generated {:?} expected {:?} {}",
            generated,
            expected,
            if pass { "PASS" } else { "FAIL" }
        );
    }
    if all_pass {
        println!("GATE: PASS token-identity");
    } else {
        println!("GATE: FAIL token-identity");
        std::process::exit(1);
    }
}
