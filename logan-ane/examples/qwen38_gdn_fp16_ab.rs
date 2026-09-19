use std::{path::Path, time::Instant};

use logan_ane::{AneRequest, AneRuntime, AneSurface, CompileOptions, mil::DenseProjection};
use logan_qwen4::colisource::ColiSource;

const HIDDEN: usize = 2560;
const QKV: usize = 10240;
const Z: usize = 6144;
const A: usize = 48;
const B: usize = 48;
const SPATIAL: usize = 16;

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return if mant == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7e00
        };
    }
    let he = exp - 127 + 15;
    if he >= 31 {
        return sign | 0x7c00;
    }
    if he <= 0 {
        if he < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - he) as u32;
        let mut hm = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && hm & 1 != 0) {
            hm += 1;
        }
        return sign | hm as u16;
    }
    let mut he_bits = (he as u16) << 10;
    let mut hm = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && hm & 1 != 0) {
        hm += 1;
        if hm == 0x400 {
            hm = 0;
            he_bits += 0x400;
        }
    }
    sign | he_bits | hm as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let f = (h & 0x03ff) as u32;
    let bits = if e == 0 {
        if f == 0 {
            s
        } else {
            let mut frac = f;
            let mut exp = 113u32;
            while frac & 0x400 == 0 {
                frac <<= 1;
                exp -= 1;
            }
            s | (exp << 23) | ((frac & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (f << 13)
    } else {
        s | ((e + 112) << 23) | (f << 13)
    };
    f32::from_bits(bits)
}

fn bf16_to_fp16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|p| {
            let bf = u16::from_le_bytes([p[0], p[1]]);
            f32_to_f16_bits(f32::from_bits((bf as u32) << 16))
        })
        .collect()
}

fn median(mut x: Vec<f64>) -> f64 {
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    x[x.len() / 2]
}
fn quality(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let (mut se, mut mx, mut dot, mut aa, mut bb) = (0., 0., 0., 0., 0.);
    for (&x, &y) in a.iter().zip(b) {
        let d = (x - y) as f64;
        se += d * d;
        mx = f64::max(mx, d.abs());
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

fn write_input(surface: &mut AneSurface, x: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let mut m = surface.write()?;
    for (c, &v) in x.iter().enumerate() {
        let h = f32_to_f16_bits(v).to_le_bytes();
        for s in 0..SPATIAL {
            let o = (c * SPATIAL + s) * 2;
            m[o..o + 2].copy_from_slice(&h);
        }
    }
    Ok(())
}
fn read_col(surface: &AneSurface, rows: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let m = surface.read()?;
    let mut y = Vec::with_capacity(rows);
    for r in 0..rows {
        let o = r * SPATIAL * 2;
        y.push(f16_to_f32(u16::from_le_bytes([m[o], m[o + 1]])));
    }
    Ok(y)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pkg = std::env::args()
        .nth(1)
        .ok_or("usage: qwen38_gdn_fp16_ab PACKAGE [layer]")?;
    let layer = std::env::args()
        .nth(2)
        .and_then(|x| x.parse().ok())
        .unwrap_or(0usize);
    let src = ColiSource::open(Path::new(&pkg)).map_err(|e| format!("open: {e}"))?;
    let p = format!("layers.{layer}.linear_attn");
    let q = src.wt(&format!("{p}.in_proj_qkv.weight"), QKV, HIDDEN)?;
    let z = src.wt(&format!("{p}.in_proj_z.weight"), Z, HIDDEN)?;
    let a = src.wt(&format!("{p}.in_proj_a.weight"), A, HIDDEN)?;
    let b = src.wt(&format!("{p}.in_proj_b.weight"), B, HIDDEN)?;
    if [q.fmt, z.fmt, a.fmt, b.fmt].iter().any(|&f| f != 5) {
        return Err("expected BF16 GDN weights".into());
    }
    let program = logan_ane::mil::parallel_dense_fp16_io(
        HIDDEN,
        SPATIAL,
        &[
            DenseProjection::new("qkv", QKV, bf16_to_fp16(&q.bytes)),
            DenseProjection::new("z", Z, bf16_to_fp16(&z.bytes)),
            DenseProjection::new("a", A, bf16_to_fp16(&a.bytes)),
            DenseProjection::new("b", B, bf16_to_fp16(&b.bytes)),
        ],
    )?;
    let rt = AneRuntime::load()?;
    let t = Instant::now();
    let mut model = rt.compile(&program, CompileOptions::default())?;
    let cm = t.elapsed().as_secs_f64() * 1e3;
    model.load()?;
    let x: Vec<f32> = (0..HIDDEN)
        .map(|i| ((i as f32 * 0.0137).sin() + (i as f32 * 0.0031).cos()) * 0.25)
        .collect();
    let mut rq = vec![0.; QKV];
    let mut rz = vec![0.; Z];
    let mut ra = vec![0.; A];
    let mut rb = vec![0.; B];
    for _ in 0..3 {
        assert!(logan_metal::bnns_bf16_matmul(
            &q.bytes, &x, &mut rq, QKV, HIDDEN
        ));
        assert!(logan_metal::bnns_bf16_matmul(
            &z.bytes, &x, &mut rz, Z, HIDDEN
        ));
        assert!(logan_metal::bnns_bf16_matmul(
            &a.bytes, &x, &mut ra, A, HIDDEN
        ));
        assert!(logan_metal::bnns_bf16_matmul(
            &b.bytes, &x, &mut rb, B, HIDDEN
        ));
    }
    let mut bt = Vec::new();
    for _ in 0..15 {
        let t = Instant::now();
        assert!(logan_metal::bnns_bf16_matmul(
            &q.bytes, &x, &mut rq, QKV, HIDDEN
        ));
        assert!(logan_metal::bnns_bf16_matmul(
            &z.bytes, &x, &mut rz, Z, HIDDEN
        ));
        assert!(logan_metal::bnns_bf16_matmul(
            &a.bytes, &x, &mut ra, A, HIDDEN
        ));
        assert!(logan_metal::bnns_bf16_matmul(
            &b.bytes, &x, &mut rb, B, HIDDEN
        ));
        bt.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut input = AneSurface::new(HIDDEN * SPATIAL * 2)?;
    write_input(&mut input, &x)?;
    let oq = AneSurface::new(QKV * SPATIAL * 2)?;
    let oz = AneSurface::new(Z * SPATIAL * 2)?;
    let oa = AneSurface::new(A * SPATIAL * 2)?;
    let ob = AneSurface::new(B * SPATIAL * 2)?;
    let req = AneRequest::new(&[&input], &[&oq, &oz, &oa, &ob], 0)?;
    for _ in 0..5 {
        model.evaluate(&req)?;
    }
    let mut at = Vec::new();
    for _ in 0..30 {
        let t = Instant::now();
        model.evaluate(&req)?;
        at.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let yq = read_col(&oq, QKV)?;
    let yz = read_col(&oz, Z)?;
    let ya = read_col(&oa, A)?;
    let yb = read_col(&ob, B)?;
    println!(
        "layer={layer} compile_ms={cm:.1} bnns_ms={:.3} ane_fp16_ms={:.3} speedup={:.2}x",
        median(bt.clone()),
        median(at.clone()),
        median(bt) / median(at)
    );
    for (name, r, y) in [
        ("qkv", &rq[..], &yq[..]),
        ("z", &rz[..], &yz[..]),
        ("a", &ra[..], &ya[..]),
        ("b", &rb[..], &yb[..]),
    ] {
        let (qr, mx, cos) = quality(r, y);
        println!("{name} rmse={qr:.6} max={mx:.6} cos={cos:.9}");
    }
    Ok(())
}
