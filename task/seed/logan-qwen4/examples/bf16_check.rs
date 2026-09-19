// Confirm matmul_bf16_bytes engages threads at and below the OLD 16M gate.
// Usage: bf16_check <OxI> <reps>
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (o, i) = { let (x, y) = a[1].split_once('x').unwrap(); (x.parse::<usize>().unwrap(), y.parse::<usize>().unwrap()) };
    let reps: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(40);
    let bytes: Vec<u8> = (0..o * i * 2).map(|k| (k % 251) as u8).collect();
    let x: Vec<f32> = (0..i).map(|k| ((k as f32) * 0.001).sin()).collect();
    let mut y = vec![0.0f32; o];
    for _ in 0..5 { logan_qwen4::matmul_bf16_exposed(&mut y, &x, &bytes, o, i); }
    let mut t = Vec::new();
    for _ in 0..reps { let t0 = Instant::now(); logan_qwen4::matmul_bf16_exposed(&mut y, &x, &bytes, o, i); t.push(t0.elapsed().as_secs_f64()*1000.0); }
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("bf16 {o}x{i}  threads={}  median_ms={:.3}", logan_qwen4::thread_count_exposed(), t[t.len()/2]);
}
