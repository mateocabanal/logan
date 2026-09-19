use std::{path::Path, time::Instant};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let root = Path::new(&a[1]);
    let n_prompt: usize = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(64);
    let n_decode: usize = a.get(3).and_then(|x| x.parse().ok()).unwrap_or(64);
    let t0 = Instant::now();
    let mut m = logan_spark::Model::load(root).unwrap();
    let load = t0.elapsed();
    let ids: Vec<u32> = (0..n_prompt).map(|pos| (pos % 100 + 1) as u32).collect();
    let t1 = Instant::now();
    let logits = m.prefill_tokens(&ids, 0);
    let pre = t1.elapsed();
    let mut next = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32;
    logan_metal::dense_profile_start();
    let t2 = Instant::now();
    for step in 0..n_decode {
        next = m.forward_token_top1(next as usize, n_prompt + step);
    }
    let dec = t2.elapsed();
    let (enc, sub, wait, gpu) = logan_metal::dense_profile_stop();
    println!(
        "load_s={:.4} prefill_s={:.4} prefill_tps={:.3} decode_s={:.4} decode_tps={:.3} metal_encode_ms={:.2} submit_ms={:.2} wait_ms={:.2} gpu_ms={:.2}",
        load.as_secs_f64(),
        pre.as_secs_f64(),
        n_prompt as f64 / pre.as_secs_f64(),
        dec.as_secs_f64(),
        n_decode as f64 / dec.as_secs_f64(),
        enc as f64 / 1e6,
        sub as f64 / 1e6,
        wait as f64 / 1e6,
        gpu as f64 / 1e6
    );
}
