use logan_qwen4::colisource::ColiSource;
use std::{ptr, time::Instant};
const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
fn bf16(b: &[u8], i: usize) -> f32 {
    let o = i * 2;
    f32::from_bits((u16::from_le_bytes([b[o], b[o + 1]]) as u32) << 16)
}
fn scale(b: &[u8], base: usize, a: usize, z: usize) -> (u8, f32) {
    let mut m = 0f32;
    for c in a..z {
        m = m.max(bf16(b, base + c).abs())
    }
    if m == 0.0 {
        return (127, 1.0);
    }
    let bits = m.to_bits();
    let bi = ((bits >> 23) & 255) as i32;
    let me = if bi == 0 {
        let x = bits & 0x7fffff;
        (31 - x.leading_zeros() as i32) - 149
    } else {
        bi - 127
    };
    let mut e = (me - 2).clamp(-126, 127);
    let mut code = (e + 127) as u8;
    let mut s = f32::from_bits((code as u32) << 23);
    if m > 6.0 * s && e < 127 {
        e += 1;
        code = (e + 127) as u8;
        s = f32::from_bits((code as u32) << 23)
    }
    (code, s)
}
fn quant(b: &[u8], r: usize, c: usize) -> (Vec<u8>, Vec<u8>) {
    let rb = c.div_ceil(2);
    let ng = c.div_ceil(32);
    let mut w = vec![0; r * rb];
    let mut s = vec![0; r * ng];
    for row in 0..r {
        let base = row * c;
        for g in 0..ng {
            let a = g * 32;
            let z = (a + 32).min(c);
            let (sc, sv) = scale(b, base, a, z);
            s[row * ng + g] = sc;
            for col in a..z {
                let v = bf16(b, base + col);
                let m = (v.abs() / sv).min(6.0);
                let (mut best, mut er) = (0usize, f32::INFINITY);
                for (k, q) in MAG.iter().copied().enumerate() {
                    let e = (m - q).abs();
                    if e < er || (e == er && k % 2 == 0 && best % 2 != 0) {
                        best = k;
                        er = e
                    }
                }
                let n = (best as u8) | if v.is_sign_negative() { 8 } else { 0 };
                let i = row * rb + col / 2;
                if col & 1 == 0 {
                    w[i] = (w[i] & 0xf0) | n
                } else {
                    w[i] = (w[i] & 0x0f) | (n << 4)
                }
            }
        }
    }
    (w, s)
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}
fn quality(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let (mut se, mut mx, mut dot, mut aa, mut bb) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = (x - y) as f64;
        se += d * d;
        mx = mx.max(d.abs());
        dot += x as f64 * y as f64;
        aa += (x as f64).powi(2);
        bb += (y as f64).powi(2)
    }
    (
        (se / a.len() as f64).sqrt(),
        mx,
        dot / (aa.sqrt() * bb.sqrt()),
    )
}
fn main() -> Result<(), String> {
    let pkg = std::env::args()
        .nth(1)
        .ok_or("usage: dense_mxfp4_probe PACKAGE")?;
    let src = ColiSource::open(std::path::Path::new(&pkg))?;
    let (r, c) = (10240usize, 2560usize);
    let name = "layers.0.linear_attn.in_proj_qkv.weight";
    let t = Instant::now();
    let wt = src.wt(name, r, c)?;
    println!(
        "bf16={:.2}MiB load={:.1}ms",
        wt.bytes.len() as f64 / 1048576.0,
        t.elapsed().as_secs_f64() * 1e3
    );
    let t = Instant::now();
    let (qw, qs) = quant(&wt.bytes, r, c);
    println!(
        "mxfp4={:.2}MiB ratio={:.3} quant={:.1}ms",
        (qw.len() + qs.len()) as f64 / 1048576.0,
        (qw.len() + qs.len()) as f64 / wt.bytes.len() as f64,
        t.elapsed().as_secs_f64() * 1e3
    );
    let x: Vec<f32> = (0..c)
        .map(|i| ((i as f32 * 0.0137).sin() + (i as f32 * 0.0031).cos()) * 0.25)
        .collect();
    let mut yr = vec![0f32; r];
    let mut yq = vec![0f32; r];
    if !logan_metal::bnns_bf16_matmul(&wt.bytes, &x, &mut yr, r, c) {
        return Err("BNNS unavailable".into());
    }
    if !logan_metal::metal_init() {
        return Err("Metal unavailable".into());
    }
    let mut tensor: *mut logan_metal::ColiMetalTensor = ptr::null_mut();
    if !logan_metal::metal_matmul(&mut tensor, &mut yq, &x, &qw, &qs, 7, c, r) {
        return Err("MXFP4 Metal failed".into());
    }
    let (rmse, mx, cos) = quality(&yr, &yq);
    println!("quality rmse={rmse:.6} max={mx:.6} cosine={cos:.8}");
    let mut bt = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        assert!(logan_metal::bnns_bf16_matmul(&wt.bytes, &x, &mut yr, r, c));
        bt.push(t.elapsed().as_secs_f64() * 1e3)
    }
    let mut mt = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        assert!(logan_metal::metal_matmul(
            &mut tensor,
            &mut yq,
            &x,
            &qw,
            &qs,
            7,
            c,
            r
        ));
        mt.push(t.elapsed().as_secs_f64() * 1e3)
    }
    let bm = median(bt.clone());
    let mm = median(mt.clone());
    println!("bnns_ms={bt:?}");
    println!("mxfp4_ms={mt:?}");
    println!(
        "median bnns={bm:.3}ms mxfp4={mm:.3}ms speedup={:.2}x",
        bm / mm
    );
    Ok(())
}
