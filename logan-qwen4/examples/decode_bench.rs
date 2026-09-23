//! Decode-throughput bench driver for the real Qwen3.6 checkpoint.
//!
//! `cargo run --release -p logan-qwen4 --example decode_bench -- MODEL_DIR TOKENS MODE [PROMPT]`
//!
//! MODE:
//!   `greedy`  — argmax. Deterministic; the regression oracle.
//!   `sample`  — seeded multinomial sampling at `BENCH_TEMP` (default 1.0),
//!               optionally truncated by `BENCH_TOP_P` / `BENCH_TOP_K`.
//!               Deterministic *given identical logits*, which is what makes it
//!               usable as a benchmark: the token trajectory is a realistic
//!               non-greedy one, but a fixed `BENCH_SEED` replays it exactly.
//!
//! The prompt is real text (chat-templated) tokenized from the checkpoint's own
//! `tokenizer.json`, so routing reflects a real generation rather than a
//! synthetic id list.
//!
//! Reports only *decode* forwards in the throughput figures: model load, prompt
//! prefill and the final prompt forward are excluded and reported separately.
//! Everything is printed as `BENCH key=value` on stdout for a shell harness to
//! consume; diagnostics go to stderr.
//!
//! This is a measurement tool. It must not change what the runtime computes:
//! it drives the same `Model::forward_token` entry point the CLI uses, with the
//! same prefill/decode boundary and the same `begin_decode_measurement` call.

use std::path::{Path, PathBuf};
use std::time::Instant;

/// Chat template matching `logan-chat`'s non-thinking assistant turn. The
/// checkpoint ships a Jinja template; this is the same rendering applied
/// inline so the bench needs no template engine.
const ASSISTANT_PREFIX: &str = "<|im_start|>assistant\n";

/// Mirror `logan-qwen4/src/main.rs`'s `apply_apple_runtime_defaults` exactly.
///
/// The safetensors loader never applies `apply_max_performance_defaults` (only
/// the `.coli` and GGUF loaders do), so on raw MLX these four defaults come
/// solely from the binary entry point. A bench that skipped them would measure
/// a slower configuration than the one users actually run, and the difference
/// is large enough to swamp the effects this harness exists to detect.
///
/// Each knob stays overridable so an A/B can still set it explicitly.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn apply_apple_runtime_defaults() {
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
        // The generic BF16 Metal GDN path synchronously submits and waits once
        // per GDN layer and is substantially slower than BNNS at decode batch
        // S=1 on this host.
        std::env::set_var("QWEN_GDN_METAL", "0");
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn apply_apple_runtime_defaults() {}

fn render_prompt(user: &str) -> String {
    format!("<|im_start|>user\n{user}<|im_end|>\n{ASSISTANT_PREFIX}")
}

/// xorshift64* — the same generator `logan-chat::engine` uses, so a seed means
/// the same thing in both harnesses.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

/// Multinomial draw from the temperature-scaled, optionally truncated softmax.
///
/// `top_p == 1.0 && top_k == 0` is the plain temperature-1.0 sampler: the whole
/// vocabulary is live, which is the honest non-greedy arm.
fn sample(logits: &[f32], temp: f32, top_p: f32, top_k: usize, rng: &mut Rng) -> Result<u32, String> {
    if !logits.iter().all(|v| v.is_finite()) {
        return Err("nonfinite logits".into());
    }
    let temp = temp.max(0.0);
    if temp <= 0.001 {
        return Ok(argmax(logits));
    }
    let mut cand: Vec<(usize, f32)> = logits.iter().copied().enumerate().map(|(i, v)| (i, v / temp)).collect();
    let k = if top_k == 0 { cand.len() } else { top_k.min(cand.len()) };
    if k < cand.len() {
        cand.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        cand.truncate(k);
    }
    cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let max_logit = cand.first().map(|x| x.1).unwrap_or(0.0);
    let mut probs: Vec<f64> = cand.iter().map(|(_, l)| ((*l - max_logit) as f64).exp()).collect();
    let z = probs.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    for p in &mut probs {
        *p /= z;
    }
    // Truncate the tail once; `kept` then holds exactly the sampled support.
    let top_p = top_p.clamp(0.01, 1.0) as f64;
    let mut keep = probs.len();
    if top_p < 0.999_999 {
        let mut cum = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cum += *p;
            if cum >= top_p {
                keep = i + 1;
                break;
            }
        }
    }
    keep = keep.max(1);
    cand.truncate(keep);
    probs.truncate(keep);

    // Re-draw until the needle lands inside the truncated support so the
    // sampled distribution is exactly the truncated one (no mass leaking to
    // the tail we just cut).
    let support: f64 = probs.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    let mut needle = rng.next_f64() * support;
    for ((token, _), p) in cand.iter().zip(&probs) {
        if needle < *p {
            return Ok(*token as u32);
        }
        needle -= *p;
    }
    Ok(cand.last().map(|v| v.0 as u32).unwrap_or(0))
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() -> Result<(), String> {
    apply_apple_runtime_defaults();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: decode_bench MODEL_DIR TOKENS {greedy|sample} [PROMPT]".into());
    }
    let dir = PathBuf::from(&args[0]);
    let tokens: usize = args[1].parse().map_err(|_| "TOKENS must be an integer")?;
    let mode = args[2].as_str();
    if mode != "greedy" && mode != "sample" {
        return Err(format!("MODE must be greedy or sample, got {mode}"));
    }
    if tokens == 0 {
        return Err("TOKENS must be nonzero".into());
    }
    let user = args.get(3).cloned().unwrap_or_else(|| {
        "Explain why memory safety matters in systems programming and how Rust \
         achieves it without a garbage collector."
            .into()
    });

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let text = render_prompt(&user);
    let enc = tok.encode(text.as_str(), false).map_err(|e| e.to_string())?;
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    if prompt_ids.is_empty() {
        return Err("empty prompt".into());
    }

    let cfg = logan_qwen4::load_cfg(&dir.join("config.json"))?;
    let load_t0 = Instant::now();
    let mut model = load_model(&dir, &cfg)?;
    let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;

    // Prefill: every prompt token except the last advances state without
    // computing disposable final-row logits; the last one produces the logits
    // that predict the first generated token.
    let prefill_t0 = Instant::now();
    for (i, &t) in prompt_ids.iter().enumerate() {
        if i + 1 == prompt_ids.len() {
            break;
        }
        model.prefill_token(t as usize, i);
    }
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;

    let last_t0 = Instant::now();
    let mut logits = model.forward_token(*prompt_ids.last().unwrap() as usize, prompt_ids.len() - 1);
    let last_prompt_fwd_ms = last_t0.elapsed().as_secs_f64() * 1e3;
    if !logits.iter().all(|v| v.is_finite()) {
        return Err("nonfinite logits after prompt".into());
    }

    // Decode boundary. Everything before this point is excluded from the
    // per-token throughput figures.
    model.begin_decode_measurement();

    let temp = env_f32("BENCH_TEMP", 1.0);
    let top_p = env_f32("BENCH_TOP_P", 1.0);
    let top_k = env_usize("BENCH_TOP_K", 0);
    let seed = std::env::var("BENCH_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x9E37_79B9_7F4A_7C15u64);
    let mut rng = Rng::new(seed);

    let mut out: Vec<u32> = Vec::with_capacity(tokens);
    let mut step_ms: Vec<f64> = Vec::with_capacity(tokens.saturating_sub(1));
    for step in 0..tokens {
        let next = if mode == "greedy" {
            if !logits.iter().all(|v| v.is_finite()) {
                return Err(format!("nonfinite logits at step {step}"));
            }
            argmax(&logits)
        } else {
            sample(&logits, temp, top_p, top_k, &mut rng)?
        };
        out.push(next);
        if step + 1 < tokens {
            let t0 = Instant::now();
            logits = model.forward_token(next as usize, prompt_ids.len() + step);
            step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }

    let measured = step_ms.len();
    if measured == 0 {
        return Err("no measured decode forwards".into());
    }
    let mean_ms = step_ms.iter().sum::<f64>() / measured as f64;
    let mut sorted = step_ms.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median_ms = sorted[sorted.len() / 2];
    let p10_ms = sorted[sorted.len() / 10];
    let p90_ms = sorted[sorted.len() * 9 / 10];

    println!("BENCH mode={mode} measured_forwards={measured}");
    println!("BENCH prompt_tokens={}", prompt_ids.len());
    println!("BENCH load_ms={load_ms:.1}");
    println!("BENCH prefill_ms={prefill_ms:.1}");
    println!("BENCH last_prompt_fwd_ms={last_prompt_fwd_ms:.1}");
    println!("BENCH decode_mean_ms={mean_ms:.3}");
    println!("BENCH decode_median_ms={median_ms:.3}");
    println!("BENCH decode_p10_ms={p10_ms:.3}");
    println!("BENCH decode_p90_ms={p90_ms:.3}");
    println!("BENCH decode_tok_s={:.4}", 1000.0 / mean_ms);
    // Per-step times let a harness pool every observed forward across repeats
    // and take a median, which is far less sensitive to a single stalled step
    // than comparing arm means.
    println!(
        "BENCH step_ms={}",
        step_ms.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>().join(",")
    );
    if mode == "sample" {
        println!("BENCH temp={temp} top_p={top_p} top_k={top_k} seed={seed}");
    }
    println!("BENCH ids={}", out.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
    // Continuation text is a readability aid, not a metric.
    if let Ok(dec) = tok.decode(&out, false) {
        println!("BENCH continuation={dec:?}");
    }
    model.profile_summary(measured, prefill_ms + last_prompt_fwd_ms + mean_ms * measured as f64);
    Ok(())
}

fn load_model(dir: &Path, cfg: &logan_qwen4::Cfg) -> Result<logan_qwen4::Model, String> {
    if dir.join("model.safetensors.index.json").is_file() || dir.join("model.safetensors").is_file() {
        let st = logan_qwen4::StFile::open_dir(dir)?;
        logan_qwen4::Model::load(&st, cfg)
    } else {
        let src = logan_qwen4::colisource::ColiSource::open(dir)?;
        logan_qwen4::Model::load_coli(&src, cfg)
    }
}
