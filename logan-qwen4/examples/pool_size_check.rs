// Does payload size explain the 60ms -> 284ms gap?
// The probe sends the SAME input 10x (serde still serializes it 10x, so the wire
// size is identical). The real difference is the ANCHOR sends distinct inputs,
// which does not change the byte count -- so measure the actual bytes.
fn main() {
    let (d, m) = (2560usize, 640usize);
    let inp: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.001).sin() * 0.1).collect();
    let calls: Vec<_> = (0..10)
        .map(|e| logan_qwen4::pool::ExpertCall {
            layer: 0,
            expert: e,
            input: inp.clone(),
        })
        .collect();
    let body = logan_qwen4::pool::encode_batch_request(
        &calls,
        "qwen38-flash-next",
        Some("2a5fdfa55b5df665254c43d306f9d2839796fa1b89ff686b6cb374a75d99b15a"),
        d,
        m,
        "silu",
    );
    println!("  request body: {} KB for 10 experts", body.len() / 1024);
    println!(
        "  -> {} KB per token at 48 layers",
        48 * body.len() / 1024 / 1024
    );
    println!();
    println!("  response: 25600 f32 as JSON text ~= 200 KB");
    println!("  -> {} MB per token", 48 * 200 * 1024 / 1024 / 1024);
    println!();
    println!("  So ~10 MB/token in, ~9 MB/token out, as JSON TEXT.");
    println!("  The probe does 6 batches = 0.9 MB total, the anchor does 48/token.");
}
