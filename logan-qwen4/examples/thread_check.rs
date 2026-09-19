// Measure whether the parallel gates actually engage, and by how much.
//
// Usage: thread_check <shape> <path>
//   shape: o_cols, e.g. "2560x6144"
//   path:  "f32" or "mxfp4"
//
// ONE configuration per process. `QWEN_THREADS` is read once per call and
// `set_var` races once any thread exists, so measuring both settings in one
// process would measure the race rather than the gate. An earlier version of
// this benchmark did exactly that and reported iteration order (warm vs cold
// cache) as a 4.7x thread speedup; the numbers below come from separate
// processes with a proper warmup.
use std::time::Instant;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let shape = a.get(1).cloned().unwrap_or_else(|| "2560x6144".into());
    let path = a.get(2).cloned().unwrap_or_else(|| "f32".into());
    let (o, i) = {
        let (a, b) = shape.split_once('x').expect("shape as OxI");
        (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap())
    };
    let threads = logan_qwen4::thread_count_exposed();
    let reps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(40);

    // Warm the data, then take the MEDIAN of timed runs so a scheduler hiccup on
    // one run cannot masquerade as a result.
    let mut times = Vec::with_capacity(reps);
    let label;
    if path == "f32" {
        let w = fill(o * i, 7);
        let x = fill(i, 11);
        let mut y = vec![0.0f32; o];
        for _ in 0..5 {
            logan_qwen4::matmul_exposed(&mut y, &x, &w, o, i);
        }
        for _ in 0..reps {
            let t0 = Instant::now();
            logan_qwen4::matmul_exposed(&mut y, &x, &w, o, i);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        label = format!("f32   {o}x{i}");
    } else {
        let rb = i.div_ceil(2);
        let ng = i.div_ceil(32);
        let weights: Vec<u8> = (0..o * rb).map(|k| (k % 251) as u8).collect();
        let scales: Vec<u8> = (0..o * ng).map(|k| ((k % 100) + 100) as u8).collect();
        let x = fill(i, 9);
        let mut y = vec![0.0f32; o];
        for _ in 0..5 {
            logan_qwen4::matmul_mxfp4_exposed(&mut y, &x, &weights, &scales, o, i);
        }
        for _ in 0..reps {
            let t0 = Instant::now();
            logan_qwen4::matmul_mxfp4_exposed(&mut y, &x, &weights, &scales, o, i);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        label = format!("mxfp4 {o}x{i}");
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    let gf = 2.0 * (o * i) as f64 / (med / 1000.0) / 1e9;
    // Machine-readable so a driver can pair the two processes.
    println!("{label}  threads={threads}  median_ms={med:.3}  gflops={gf:.2}");
}
