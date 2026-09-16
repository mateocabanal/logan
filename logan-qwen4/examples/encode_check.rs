// The encoder must produce byte-identical output after the preallocation change,
// and the cost must actually drop. Both matter: an encoder that is fast and
// wrong corrupts every expert input.
fn main() {
    let d = 2560usize;
    let inp: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.001).sin() * 0.1).collect();
    let calls: Vec<_> = (0..10).map(|e| logan_qwen4::pool::ExpertCall {
        layer: 7, expert: e, input: inp.clone() }).collect();
    let body = logan_qwen4::pool::encode_batch_request(&calls, "fam", Some("deadbeef"), d, 640, "silu");
    println!("  body: {} KB", body.len()/1024);
    // Spot-check the structure the coordinator parses.
    for needle in ["\"family\":\"fam\"", "\"source_hash\":\"deadbeef\"",
                   "\"d_model\":2560", "\"layer\":7,\"expert\":0", "\"items\":["] {
        println!("  contains {needle:<32} {}", body.contains(needle));
    }
    println!("  ends with ']}}': {}", body.ends_with("]}"));
    // Every float present, in order: count separators.
    let commas = body.matches(',').count();
    println!("  commas: {commas} (expect > {})", d*10);
    let reps = 200;
    let t0 = std::time::Instant::now();
    for _ in 0..reps { std::hint::black_box(logan_qwen4::pool::encode_batch_request(&calls, "fam", Some("deadbeef"), d, 640, "silu")); }
    println!("  encode: {:.2} ms per call", t0.elapsed().as_secs_f64()*1000.0/reps as f64);
}
