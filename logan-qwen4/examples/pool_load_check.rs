// Reproduce the anchor's submission pattern: 48 sequential layer batches, each
// waiting for its result, so the worker never gets a gap. This is the load shape
// the anchor actually produces, unlike a handful of isolated probe batches.
fn main() {
    let cfg = match logan_qwen4::pool::PoolConfig::from_env() {
        Some(c) => c,
        None => {
            println!("set LOGAN_POOL_COORDINATOR");
            return;
        }
    };
    let (d, m) = (2560usize, 640usize);
    let inp: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.001).sin() * 0.1).collect();
    let t_all = std::time::Instant::now();
    let mut times = Vec::new();
    for layer in 0..48u32 {
        let calls: Vec<_> = (0..10)
            .map(|e| logan_qwen4::pool::ExpertCall {
                layer,
                expert: (layer * 7 + e) % 512,
                input: inp.clone(),
            })
            .collect();
        let t = std::time::Instant::now();
        match logan_qwen4::pool::run_expert_batch(&cfg, &calls, d, m, "silu") {
            Ok(v) => times.push(t.elapsed().as_secs_f64() * 1000.0),
            Err(e) => {
                println!("  layer {layer} FAILED: {e}");
            }
        }
        let _ = times.len();
    }
    let total = t_all.elapsed().as_secs_f64() * 1000.0;
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("  48 layer batches: total {:.0} ms", total);
    println!(
        "  per layer: min {:.0}  median {:.0}  max {:.0} ms",
        sorted[0],
        sorted[sorted.len() / 2],
        sorted[sorted.len() - 1]
    );
    println!("  -> this is the `fill` cost the anchor reports");
}
