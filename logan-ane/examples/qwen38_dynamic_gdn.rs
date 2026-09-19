use std::{path::Path, time::Instant};

use logan_ane::{AneRequest, AneRuntime, AneSurface, CompileOptions, DenseProjection};
use logan_qwen4::colisource::{ColiSource, ColiWt};

const HIDDEN: usize = 2560;
const K_HEADS: usize = 16;
const K_DIM: usize = 128;
const V_HEADS: usize = 48;
const V_DIM: usize = 128;
const QKV: usize = K_HEADS * K_DIM * 2 + V_HEADS * V_DIM; // 10240
const Z: usize = V_HEADS * V_DIM; // 6144
const A: usize = V_HEADS; // 48
const B: usize = V_HEADS; // 48
const SPATIAL: usize = 16; // honest decode padding required by current ANE MIL geometry

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// IEEE-754 binary32 -> binary16, round-to-nearest-even. This probe only accepts
// finite checkpoint values; infinities/NaNs are preserved if encountered so the
// numerical gate fails visibly rather than silently clipping them.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;

    if exp == 0xff {
        if mant == 0 {
            return sign | 0x7c00;
        }
        let payload = ((mant >> 13) as u16).max(1);
        return sign | 0x7c00 | payload;
    }

    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x80_0000;
        let shift = (14 - half_exp) as u32;
        let mut half_mant = mantissa >> shift;
        let remainder_mask = (1u32 << shift) - 1;
        let remainder = mantissa & remainder_mask;
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && (half_mant & 1) != 0) {
            half_mant += 1;
        }
        return sign | half_mant as u16;
    }

    let mut half_exp_bits = (half_exp as u16) << 10;
    let mut half_mant = mant >> 13;
    let remainder = mant & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (half_mant & 1) != 0) {
        half_mant += 1;
        if half_mant == 0x400 {
            half_mant = 0;
            half_exp_bits += 0x400;
            if half_exp_bits >= 0x7c00 {
                return sign | 0x7c00;
            }
        }
    }
    sign | half_exp_bits | half_mant as u16
}

fn bf16_bytes_to_fp16(bytes: &[u8]) -> Result<Vec<u16>, String> {
    if bytes.len() % 2 != 0 {
        return Err("BF16 tensor byte count is not even".into());
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let v = bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        if !v.is_finite() {
            return Err(format!("checkpoint contains non-finite weight {v}"));
        }
        out.push(f32_to_f16_bits(v));
    }
    Ok(out)
}

fn load_bf16(src: &ColiSource, name: &str, rows: usize) -> Result<ColiWt, String> {
    let wt = src.wt(name, rows, HIDDEN)?;
    if wt.fmt != 5 {
        return Err(format!("{name}: expected BF16 fmt=5, got fmt={}", wt.fmt));
    }
    if wt.bytes.len() != rows * HIDDEN * 2 {
        return Err(format!(
            "{name}: expected {} BF16 bytes, got {}",
            rows * HIDDEN * 2,
            wt.bytes.len()
        ));
    }
    Ok(wt)
}

#[derive(Debug)]
struct Quality {
    rmse: f64,
    rel_rmse: f64,
    max_abs: f64,
    cosine: f64,
}

fn quality(reference: &[f32], actual_surface: &[f32], spatial: usize) -> Quality {
    assert_eq!(actual_surface.len(), reference.len() * spatial);
    let mut se = 0.0f64;
    let mut ref_e = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut dot = 0.0f64;
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    for (row, &r) in reference.iter().enumerate() {
        let a = actual_surface[row * spatial] as f64;
        let r = r as f64;
        let d = a - r;
        se += d * d;
        ref_e += r * r;
        max_abs = max_abs.max(d.abs());
        dot += r * a;
        aa += r * r;
        bb += a * a;
    }
    Quality {
        rmse: (se / reference.len() as f64).sqrt(),
        rel_rmse: (se / ref_e.max(f64::MIN_POSITIVE)).sqrt(),
        max_abs,
        cosine: dot / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE),
    }
}

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn write_repeated(surface: &mut AneSurface, x: &[f32]) -> Result<(), String> {
    let mut map = surface.write().map_err(|e| e.to_string())?;
    for (channel, &value) in x.iter().enumerate() {
        let bytes = value.to_le_bytes();
        let base = channel * SPATIAL * 4;
        for lane in 0..SPATIAL {
            let off = base + lane * 4;
            map[off..off + 4].copy_from_slice(&bytes);
        }
    }
    Ok(())
}

fn read_column(surface: &AneSurface, rows: usize, dst: &mut [f32]) -> Result<(), String> {
    let map = surface.read().map_err(|e| e.to_string())?;
    for (row, value) in dst.iter_mut().enumerate().take(rows) {
        let off = row * SPATIAL * 4;
        *value = f32::from_le_bytes(map[off..off + 4].try_into().unwrap());
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let package = std::env::args().nth(1).unwrap_or_else(|| {
        "/Users/mateo/models/Qwen3.8-Flash-Next-REAP-288-MXFP4-Apple8.coli".into()
    });
    let layer: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mode = std::env::args().nth(3).unwrap_or_else(|| "qkv".into());
    let pn = format!("layers.{layer}.linear_attn");
    let src = ColiSource::open(Path::new(&package))?;
    let qkv = load_bf16(&src, &format!("{pn}.in_proj_qkv.weight"), QKV)?;
    let z = load_bf16(&src, &format!("{pn}.in_proj_z.weight"), Z)?;
    let a = load_bf16(&src, &format!("{pn}.in_proj_a.weight"), A)?;
    let b = load_bf16(&src, &format!("{pn}.in_proj_b.weight"), B)?;
    let selected: Vec<(&str, &ColiWt, usize)> = match mode.as_str() {
        "qkv" => vec![("qkv", &qkv, QKV)],
        "aux" => vec![("z", &z, Z), ("a", &a, A), ("b", &b, B)],
        _ => return Err("mode must be qkv or aux".into()),
    };
    let outs: Vec<usize> = selected.iter().map(|x| x.2).collect();
    let (program, layout) =
        logan_ane::mil::parallel_dense_packed_dynamic_f32_io(HIDDEN, SPATIAL, &outs)
            .map_err(|e| e.to_string())?;
    println!(
        "mode={mode} layer={layer} packed_spatial={} packed_mib={:.2}",
        layout.total_spatial,
        (HIDDEN * layout.total_spatial * 4) as f64 / 1048576.0
    );
    let runtime = AneRuntime::load().map_err(|e| e.to_string())?;
    let opts = CompileOptions {
        reuse_compiled_model: false,
        keep_temporary_files: true,
        ..CompileOptions::default()
    };
    let t = Instant::now();
    let mut model = runtime.compile(&program, opts).map_err(|e| e.to_string())?;
    println!("compile_ms={:.3}", t.elapsed().as_secs_f64() * 1e3);
    model.load().map_err(|e| e.to_string())?;
    let x: Vec<f32> = (0..HIDDEN)
        .map(|i| ((i as f32 * 0.0137).sin() + (i as f32 * 0.0031).cos()) * 0.25)
        .collect();
    let mut refs: Vec<Vec<f32>> = selected.iter().map(|x| vec![0f32; x.2]).collect();
    for (idx, (_, wt, o)) in selected.iter().enumerate() {
        if !logan_metal::bnns_bf16_matmul(&wt.bytes, &x, &mut refs[idx], *o, HIDDEN) {
            return Err("BNNS unavailable".into());
        }
    }
    let mut input =
        AneSurface::new(HIDDEN * layout.total_spatial * 4).map_err(|e| e.to_string())?;
    let t = Instant::now();
    for (k, (_, wt, o)) in selected.iter().enumerate() {
        input
            .pack_bf16_transposed_f32(
                layout.total_spatial,
                layout.weight_offsets[k],
                &wt.bytes,
                HIDDEN,
                *o,
            )
            .map_err(|e| e.to_string())?;
    }
    println!("neon_weight_pack_ms={:.3}", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    input
        .write_repeated_f32(layout.total_spatial, SPATIAL, &x)
        .map_err(|e| e.to_string())?;
    println!("activation_write_ms={:.3}", t.elapsed().as_secs_f64() * 1e3);
    let ys: Vec<AneSurface> = outs
        .iter()
        .map(|o| AneSurface::new(o * SPATIAL * 4).unwrap())
        .collect();
    let yrefs: Vec<&AneSurface> = ys.iter().collect();
    let req = AneRequest::new(&[&input], &yrefs, 0).map_err(|e| e.to_string())?;
    for _ in 0..3 {
        model.evaluate(&req).map_err(|e| e.to_string())?;
    }
    let mut samples = Vec::new();
    for _ in 0..20 {
        let t = Instant::now();
        model.evaluate(&req).map_err(|e| e.to_string())?;
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "eval_median_ms={:.3} samples={samples:?}",
        median_ms(samples.clone())
    );
    for (k, (name, _, _)) in selected.iter().enumerate() {
        let got = ys[k].read_f32().map_err(|e| e.to_string())?;
        let q = quality(&refs[k], &got, SPATIAL);
        println!(
            "{name} rmse={:.6} rel={:.6} max={:.6} cos={:.9}",
            q.rmse, q.rel_rmse, q.max_abs, q.cosine
        );
    }
    Ok(())
}
