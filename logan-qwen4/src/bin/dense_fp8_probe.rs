use logan_qwen4::colisource::ColiSource;
use std::{ptr, slice, time::Instant};

fn bf16(bytes: &[u8], idx: usize) -> f32 {
    let o = idx * 2;
    f32::from_bits((u16::from_le_bytes([bytes[o], bytes[o + 1]]) as u32) << 16)
}

fn e4m3_positive(code: u8) -> f32 {
    let exp = ((code >> 3) & 0x0f) as i32;
    let mant = (code & 7) as f32;
    if exp == 0 {
        mant * 0.001953125
    } else {
        (1.0 + mant * 0.125) * 2f32.powi(exp - 7)
    }
}

fn nearest_e4m3(value: f32, table: &[f32]) -> u8 {
    let a = value.abs().min(448.0);
    let hi = table.partition_point(|&x| x < a);
    let code = if hi == 0 {
        0
    } else if hi >= table.len() {
        table.len() - 1
    } else {
        let lo = hi - 1;
        if (a - table[lo]).abs() <= (table[hi] - a).abs() {
            lo
        } else {
            hi
        }
    } as u8;
    code | if value.is_sign_negative() { 0x80 } else { 0 }
}

fn quant_fp8(bytes: &[u8], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
    let bro = rows.div_ceil(128);
    let bri = cols.div_ceil(128);
    let table: Vec<f32> = (0u8..=0x7e).map(e4m3_positive).collect();
    let mut weights = vec![0u8; rows * cols];
    let mut scales = vec![1.0f32; bro * bri];
    for bo in 0..bro {
        for bi in 0..bri {
            let r0 = bo * 128;
            let r1 = (r0 + 128).min(rows);
            let c0 = bi * 128;
            let c1 = (c0 + 128).min(cols);
            let mut max_abs = 0.0f32;
            for r in r0..r1 {
                for c in c0..c1 {
                    max_abs = max_abs.max(bf16(bytes, r * cols + c).abs());
                }
            }
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 448.0 };
            scales[bo * bri + bi] = scale;
            for r in r0..r1 {
                for c in c0..c1 {
                    weights[r * cols + c] = nearest_e4m3(bf16(bytes, r * cols + c) / scale, &table);
                }
            }
        }
    }
    (weights, scales)
}

fn quality(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let (mut se, mut mx, mut dot, mut aa, mut bb) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = (x - y) as f64;
        se += d * d;
        mx = mx.max(d.abs());
        dot += x as f64 * y as f64;
        aa += (x as f64).powi(2);
        bb += (y as f64).powi(2);
    }
    (
        (se / a.len() as f64).sqrt(),
        mx,
        dot / (aa.sqrt() * bb.sqrt()),
    )
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> Result<(), String> {
    let pkg = std::env::args()
        .nth(1)
        .ok_or("usage: dense_fp8_probe PACKAGE")?;
    let src = ColiSource::open(std::path::Path::new(&pkg))?;
    let (rows, cols) = (10240usize, 2560usize);
    let name = "layers.0.linear_attn.in_proj_qkv.weight";
    let t0 = Instant::now();
    let wt = src.wt(name, rows, cols)?;
    println!(
        "bf16={:.2}MiB load={:.1}ms",
        wt.bytes.len() as f64 / 1048576.0,
        t0.elapsed().as_secs_f64() * 1e3
    );
    let t0 = Instant::now();
    let (qw, qs) = quant_fp8(&wt.bytes, rows, cols);
    println!(
        "fp8={:.2}MiB scales={} ratio={:.3} quant={:.1}ms",
        qw.len() as f64 / 1048576.0,
        qs.len(),
        (qw.len() + qs.len() * 4) as f64 / wt.bytes.len() as f64,
        t0.elapsed().as_secs_f64() * 1e3
    );
    let scale_bytes = unsafe { slice::from_raw_parts(qs.as_ptr() as *const u8, qs.len() * 4) };
    let x: Vec<f32> = (0..cols)
        .map(|i| ((i as f32 * 0.0137).sin() + (i as f32 * 0.0031).cos()) * 0.25)
        .collect();
    let mut yr = vec![0f32; rows];
    let mut yq = vec![0f32; rows];
    if !logan_metal::bnns_bf16_matmul(&wt.bytes, &x, &mut yr, rows, cols) {
        return Err("BNNS unavailable".into());
    }
    if !logan_metal::metal_init() {
        return Err("Metal unavailable".into());
    }
    let mut tensor: *mut logan_metal::ColiMetalTensor = ptr::null_mut();
    if !logan_metal::metal_matmul(&mut tensor, &mut yq, &x, &qw, scale_bytes, 8, cols, rows) {
        return Err("FP8 Metal failed".into());
    }
    let (rmse, mx, cos) = quality(&yr, &yq);
    println!("quality rmse={rmse:.6} max={mx:.6} cosine={cos:.8}");
    let mut bt = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        assert!(logan_metal::bnns_bf16_matmul(
            &wt.bytes, &x, &mut yr, rows, cols
        ));
        bt.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut mt = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        assert!(logan_metal::metal_matmul(
            &mut tensor,
            &mut yq,
            &x,
            &qw,
            scale_bytes,
            8,
            cols,
            rows
        ));
        mt.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let bm = median(bt.clone());
    let mm = median(mt.clone());
    println!("bnns_ms={bt:?}");
    println!("fp8_ms={mt:?}");
    println!(
        "median bnns={bm:.3}ms fp8={mm:.3}ms speedup={:.2}x",
        bm / mm
    );
    if !tensor.is_null() {
        unsafe { logan_metal::coli_metal_tensor_free(tensor) }
    }
    Ok(())
}
