use std::{path::Path, time::Instant};

use logan_ane::AneSurface;
use logan_metal::{MetalGdnConvSilu, MetalSharedSurface};
use logan_qwen4::colisource::ColiSource;

const C: usize = 10240;
const S: usize = 16;
const K: usize = 4;

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|p| f32::from_bits((u16::from_le_bytes([p[0], p[1]]) as u32) << 16))
        .collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn quality(a: &[f32], b: &[f32]) -> (f64, f32, f64) {
    let (mut se, mut mx, mut dot, mut aa, mut bb) = (0.0f64, 0.0f32, 0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = x - y;
        se += (d as f64).powi(2);
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pkg = std::env::args().nth(1).ok_or("PACKAGE")?;
    let src = ColiSource::open(Path::new(&pkg))?;
    let taps = bf16_to_f32(&src.vec("layers.0.linear_attn.conv1d.weight", C * K)?);

    let current = (0..C)
        .map(|i| ((i as f32 * 0.0013).sin()) * 0.2)
        .collect::<Vec<_>>();
    let history = (0..C * (K - 1))
        .map(|i| ((i as f32 * 0.0007).cos()) * 0.05)
        .collect::<Vec<_>>();
    let reference = (0..C)
        .map(|ch| {
            let mut acc = 0.0f32;
            for j in 0..K {
                let v = if j + 1 == K {
                    current[ch]
                } else {
                    history[ch * (K - 1) + j]
                };
                acc += taps[ch * K + j] * v;
            }
            silu(acc)
        })
        .collect::<Vec<_>>();

    let mut qkv = AneSurface::new(C * S * 4)?;
    let out = AneSurface::new(C * 4)?;
    {
        let mut map = qkv.write()?;
        for ch in 0..C {
            let base = ch * S * 4;
            let cb = current[ch].to_le_bytes();
            for lane in 0..S {
                let off = base + lane * 4;
                map[off..off + 4].copy_from_slice(&cb);
            }
            for j in 0..K - 1 {
                let off = base + j * 4;
                map[off..off + 4].copy_from_slice(&history[ch * (K - 1) + j].to_le_bytes());
            }
        }
    }

    let qkv_metal =
        unsafe { MetalSharedSurface::from_iosurface(qkv.as_raw_iosurface(), qkv.len()) }
            .ok_or("failed to import qkv IOSurface into Metal")?;
    let out_metal =
        unsafe { MetalSharedSurface::from_iosurface(out.as_raw_iosurface(), out.len()) }
            .ok_or("failed to import output IOSurface into Metal")?;
    let mut kernel = MetalGdnConvSilu::new(&qkv_metal, &out_metal, &taps, C, S, K)
        .ok_or("failed to create Metal Conv1D+SiLU continuation")?;

    // The continuation must retain IOSurfaces, not just no-copy MTLBuffers.
    drop(qkv_metal);
    drop(out_metal);
    drop(qkv);
    // Dropping an in-flight ticket must drain before the output is mapped.
    drop(unsafe { kernel.begin() }.ok_or("begin declined")?);
    let after_drop = out.read_f32()?;
    assert!(quality(&reference, &after_drop).1 < 1e-6);
    let (ok, gpu_ms) = unsafe { kernel.begin() }.ok_or("begin declined")?.finish();
    assert!(ok && gpu_ms >= 0.0);
    assert!(quality(&reference, &out.read_f32()?).1 < 1e-6);
    println!("pending_drop_and_surface_retention=passed gpu_ms={gpu_ms:.6}");

    for _ in 0..4 {
        assert!(kernel.run());
    }
    let mut times = Vec::new();
    for _ in 0..50 {
        let t = Instant::now();
        assert!(kernel.run());
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let got = out.read_f32()?;
    let (rmse, mx, cos) = quality(&reference, &got);
    assert!(mx < 1e-6 && cos > 0.999999);
    println!(
        "median_ms={:.4} p10_ms={:.4} p90_ms={:.4} rmse={rmse:.9} max={mx:.9} cosine={cos:.12}",
        times[times.len() / 2],
        times[times.len() / 10],
        times[times.len() * 9 / 10]
    );
    Ok(())
}
