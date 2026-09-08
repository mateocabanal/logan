use std::{path::Path, time::Instant};

use logan_qwen4::{Model, colisource::ColiSource, load_cfg};

fn argmax(values: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn fingerprint(values: &[f32]) -> (f64, f64, f32, usize) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut max = f32::NEG_INFINITY;
    let mut maxi = 0usize;
    for (i, &v) in values.iter().enumerate() {
        sum += v as f64;
        sumsq += (v as f64) * (v as f64);
        if v > max {
            max = v;
            maxi = i;
        }
    }
    (sum, sumsq.sqrt(), max, maxi)
}

fn peak_rss_mib() -> f64 {
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if rc != 0 {
        return 0.0;
    }
    // macOS reports ru_maxrss in bytes (Linux reports KiB). This benchmark is
    // Apple-Silicon-specific, so expose the native macOS value directly.
    usage.ru_maxrss as f64 / (1024.0 * 1024.0)
}

fn write_logits(path: &str, values: &[f32]) -> Result<(), String> {
    use std::io::Write as _;
    let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    for &value in values {
        file.write_all(&value.to_le_bytes())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let package = std::env::args()
        .nth(1)
        .ok_or("usage: gdn_ane_e2e PACKAGE [forwards]")?;
    let forwards = std::env::args()
        .nth(2)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
        .max(1);
    let package = Path::new(&package);
    let cfg = load_cfg(&package.join("config.json"))?;
    let source = ColiSource::open(package)?;
    eprintln!(
        "mode: ane={} layers={} ane_fused={} gdn_metal={} ctx={}",
        std::env::var("QWEN_GDN_ANE").unwrap_or_else(|_| "0".into()),
        std::env::var("QWEN_GDN_ANE_LAYERS").unwrap_or_else(|_| "<default>".into()),
        std::env::var("QWEN_GDN_ANE_FUSED").unwrap_or_else(|_| "<default>".into()),
        std::env::var("QWEN_GDN_METAL").unwrap_or_else(|_| "<default>".into()),
        std::env::var("CTX").unwrap_or_else(|_| "<default>".into()),
    );
    let load_t0 = Instant::now();
    let mut model = Model::load_coli(&source, &cfg)?;
    println!("load_ms={:.3}", load_t0.elapsed().as_secs_f64() * 1e3);

    // Use a deterministic fixed token stream so separate baseline/ANE processes
    // can compare fingerprints without tokenizer or sampling differences.
    let vocab = model.runtime_stats().vocab_size.max(2);
    let tokens: Vec<usize> = (0..forwards)
        .map(|i| (17_321usize.wrapping_add(i * 7_919)) % vocab)
        .collect();

    let mut final_logits = Vec::new();
    let inference_t0 = Instant::now();
    for (pos, token) in tokens.into_iter().enumerate() {
        let before = model.runtime_stats();
        let t0 = Instant::now();
        let logits = model.forward_token(token, pos);
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        if logits.is_empty() {
            return Err(format!("forward {pos} returned empty logits"));
        }
        let delta = model.runtime_stats().delta_from(&before);
        let (sum, l2, max, maxi) = fingerprint(&logits);
        let rss = peak_rss_mib();
        println!(
            "forward={pos} token={token} wall_ms={wall_ms:.3} route_ms={:.3} gdn_ms={:.3} gdn_in_ms={:.3} gdn_conv_ms={:.3} gdn_prep_ms={:.3} gdn_recur_ms={:.3} gdn_gate_ms={:.3} gdn_out_ms={:.3} attn_ms={:.3} hc_ms={:.3} io_ms={:.3} shared_ms={:.3} gpu_ms={:.3} head_ms={:.3} hits={} misses={} resident={}/{} mio_loads={} mio_mb={:.1} metal_wait_ms={:.3} metal_kernel_ms={:.3} gdn_wait_ms={:.3} gdn_kernel_ms={:.3} gdn_calls={} moe_wait_ms={:.3} moe_kernel_ms={:.3} moe_calls={} peak_rss_mib={rss:.1} argmax={} fp_sum={sum:.9e} fp_l2={l2:.9e} max={max:.7} max_i={maxi}",
            delta.route_ms,
            delta.gdn_ms,
            delta.gdn_in_proj_ms,
            delta.gdn_conv_ms,
            delta.gdn_prepare_ms,
            delta.gdn_recur_ms,
            delta.gdn_gate_ms,
            delta.gdn_out_proj_ms,
            delta.attn_ms,
            delta.hc_ms,
            delta.io_ms,
            delta.shared_ms,
            delta.gpu_ms,
            delta.head_ms,
            delta.expert_hits,
            delta.expert_misses,
            delta.expert_resident,
            delta.expert_capacity,
            delta.mio_loads,
            delta.mio_bytes as f64 / (1024.0 * 1024.0),
            delta.metal_wait_ns as f64 / 1e6,
            delta.metal_kernel_ns as f64 / 1e6,
            delta.gdn_metal_wait_ns as f64 / 1e6,
            delta.gdn_metal_kernel_ns as f64 / 1e6,
            delta.gdn_metal_calls,
            delta.moe_metal_wait_ns as f64 / 1e6,
            delta.moe_metal_kernel_ns as f64 / 1e6,
            delta.moe_metal_calls,
            argmax(&logits),
        );
        final_logits = logits;
    }
    model.profile_summary(forwards, inference_t0.elapsed().as_secs_f64() * 1e3);
    if let Ok(path) = std::env::var("LOGAN_LOGITS_OUT") {
        write_logits(&path, &final_logits)?;
        println!("wrote_logits={path}");
    }
    Ok(())
}
