// A pooled connection that the peer closed must not break the request.
// This is what the retry is for, and it is the failure mode a keep-alive pool
// introduces: the socket was valid when stored.
fn main() {
    let cfg = match logan_qwen4::pool::PoolConfig::from_env() {
        Some(c) => c,
        None => {
            println!("set LOGAN_POOL_COORDINATOR");
            return;
        }
    };
    let (d, m) = (2560usize, 640usize);
    let inp: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.001).sin()).collect();
    let calls: Vec<_> = (0..3)
        .map(|e| logan_qwen4::pool::ExpertCall {
            layer: 0,
            expert: e,
            input: inp.clone(),
        })
        .collect();
    // Warm the pool so a socket is stored.
    let _ = logan_qwen4::pool::run_expert_batch(&cfg, &calls, d, m, "silu");
    println!("  1. warmed (a connection is pooled)");
    // Idle long enough that the coordinator's keep-alive plausibly drops it, then
    // issue many requests: if the retry is missing, one of these fails.
    std::thread::sleep(std::time::Duration::from_secs(35));
    println!("  2. idled 35s");
    let mut ok = 0;
    let mut err = 0;
    for i in 0..12 {
        match logan_qwen4::pool::run_expert_batch(&cfg, &calls, d, m, "silu") {
            Ok(_) => ok += 1,
            Err(e) => {
                err += 1;
                println!("     request {i} FAILED: {e}");
            }
        }
    }
    println!("  3. after idle: {ok} ok, {err} failed");
    println!(
        "  => {}",
        if err == 0 {
            "stale connections recovered"
        } else {
            "RETRY MISSING"
        }
    );
}
