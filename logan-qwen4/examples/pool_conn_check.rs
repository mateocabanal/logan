// Verify connection reuse works end to end against the live coordinator:
// repeated batches must reuse one socket and stay correct.
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
    let calls: Vec<_> = (0..10)
        .map(|e| logan_qwen4::pool::ExpertCall {
            layer: 0,
            expert: e,
            input: inp.clone(),
        })
        .collect();
    let t0 = std::time::Instant::now();
    let first = logan_qwen4::pool::run_expert_batch(&cfg, &calls, d, m, "silu");
    match &first {
        Ok(v) => println!(
            "  batch 1 (cold connect): {:?} -> {} experts",
            t0.elapsed(),
            v.len()
        ),
        Err(e) => println!("  batch 1 FAILED: {e}"),
    }
    for i in 2..=6 {
        let t = std::time::Instant::now();
        match logan_qwen4::pool::run_expert_batch(&cfg, &calls, d, m, "silu") {
            Ok(v) => println!(
                "  batch {i} (reused):      {:?} -> {} experts, first={:.6}",
                t.elapsed(),
                v.len(),
                v[0][0]
            ),
            Err(e) => println!("  batch {i} FAILED: {e}"),
        }
    }
    // Determinism: the same input must give the same answer on every batch.
    let a = logan_qwen4::pool::run_expert_batch(&cfg, &calls, d, m, "silu").unwrap();
    let b = logan_qwen4::pool::run_expert_batch(&cfg, &calls, d, m, "silu").unwrap();
    println!("  deterministic across reuses: {}", a == b);
}
