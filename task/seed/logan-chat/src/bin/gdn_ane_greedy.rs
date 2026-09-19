use std::{path::Path, time::Instant};

use logan_qwen4::{colisource::ColiSource, load_cfg, Model};
use tokenizers::Tokenizer;

fn argmax(values: &[f32]) -> Result<u32, String> {
    values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index as u32)
        .ok_or_else(|| "logits contain no finite values".to_string())
}

/// Peak resident set size in MiB, or 0.0 when unavailable.
///
/// macOS `getrusage` reports `ru_maxrss` in bytes (Linux reports KiB), and this
/// benchmark is Apple-Silicon-specific, so the native macOS value is exposed
/// directly. Windows has no `getrusage`; the probe reports 0.0 there, the same
/// "not measured" value the failure path already returns, so the CPU baseline
/// still runs and only the RSS column is absent.
#[cfg(target_os = "macos")]
fn peak_rss_mib() -> f64 {
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if rc != 0 {
        return 0.0;
    }
    usage.ru_maxrss as f64 / (1024.0 * 1024.0)
}

#[cfg(not(target_os = "macos"))]
fn peak_rss_mib() -> f64 {
    0.0
}

fn main() -> Result<(), String> {
    let package = std::env::args()
        .nth(1)
        .ok_or("usage: gdn_ane_greedy PACKAGE [max_new] [prompt]")?;
    let max_new = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(12)
        .max(1);
    let prompt = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "The capital of France is".to_string());
    let package = Path::new(&package);

    let tokenizer_path = package.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| format!("load {}: {error}", tokenizer_path.display()))?;
    let encoding = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|error| format!("tokenize: {error}"))?;
    let prompt_ids = encoding.get_ids().to_vec();
    if prompt_ids.is_empty() {
        return Err("prompt tokenized to zero tokens".into());
    }
    eprintln!(
        "prompt={prompt:?} prompt_tokens={} ids={prompt_ids:?}",
        prompt_ids.len()
    );
    eprintln!(
        "mode: ane={} layers={} max_new={max_new}",
        std::env::var("QWEN_GDN_ANE").unwrap_or_else(|_| "0".into()),
        std::env::var("QWEN_GDN_ANE_LAYERS").unwrap_or_else(|_| "<default>".into()),
    );

    let cfg = load_cfg(&package.join("config.json"))?;
    let source = ColiSource::open(package)?;
    let load_t0 = Instant::now();
    let mut model = Model::load_coli(&source, &cfg)?;
    println!("load_ms={:.3}", load_t0.elapsed().as_secs_f64() * 1e3);

    let prefill_t0 = Instant::now();
    let mut logits = Vec::new();
    for (position, &token) in prompt_ids.iter().enumerate() {
        logits = model.forward_token(token as usize, position);
        if logits.is_empty() {
            return Err(format!("prefill forward {position} returned empty logits"));
        }
    }
    println!(
        "prefill_ms={:.3} peak_rss_mib={:.1}",
        prefill_t0.elapsed().as_secs_f64() * 1e3,
        peak_rss_mib(),
    );

    let decode_t0 = Instant::now();
    let mut generated = Vec::with_capacity(max_new);
    for step in 0..max_new {
        let next = argmax(&logits)?;
        generated.push(next);
        println!("step={step} token={next}");
        if step + 1 < max_new {
            logits = model.forward_token(next as usize, prompt_ids.len() + step);
            if logits.is_empty() {
                return Err(format!("decode forward {step} returned empty logits"));
            }
        }
    }
    let decode_ms = decode_t0.elapsed().as_secs_f64() * 1e3;
    let text = tokenizer
        .decode(&generated, false)
        .map_err(|error| format!("decode generated tokens: {error}"))?;
    println!("generated_ids={generated:?}");
    println!("generated_text={text:?}");
    println!(
        "decode_ms={decode_ms:.3} generated={} tok_s={:.5} peak_rss_mib={:.1}",
        generated.len(),
        generated.len() as f64 / (decode_ms / 1e3),
        peak_rss_mib(),
    );

    if let Ok(path) = std::env::var("GREEDY_IDS_OUT") {
        let body = generated
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(&path, format!("{body}\n")).map_err(|error| error.to_string())?;
        println!("wrote_ids={path}");
    }
    Ok(())
}
