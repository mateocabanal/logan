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
        .map(|s| {
            s.parse()
                .map_err(|_| "layer must be an integer".to_string())
        })
        .transpose()?
        .unwrap_or(0);
    let prefix = format!("layers.{layer}.linear_attn");
    println!("package={package}");
    println!("layer={layer} hidden={HIDDEN} qkv={QKV} z={Z} a={A} b={B} spatial={SPATIAL}");

    let src = ColiSource::open(Path::new(&package))?;
    let load_t0 = Instant::now();
    let qkv = load_bf16(&src, &format!("{prefix}.in_proj_qkv.weight"), QKV)?;
    let z = load_bf16(&src, &format!("{prefix}.in_proj_z.weight"), Z)?;
    let a = load_bf16(&src, &format!("{prefix}.in_proj_a.weight"), A)?;
    let b = load_bf16(&src, &format!("{prefix}.in_proj_b.weight"), B)?;
    println!(
        "loaded {:.2} MiB BF16 in {:.1} ms",
        (qkv.bytes.len() + z.bytes.len() + a.bytes.len() + b.bytes.len()) as f64 / 1048576.0,
        load_t0.elapsed().as_secs_f64() * 1e3
    );

    let convert_t0 = Instant::now();
    let qkv16 = bf16_bytes_to_fp16(&qkv.bytes)?;
    let z16 = bf16_bytes_to_fp16(&z.bytes)?;
    let a16 = bf16_bytes_to_fp16(&a.bytes)?;
    let b16 = bf16_bytes_to_fp16(&b.bytes)?;
    println!(
        "BF16->FP16 conversion {:.1} ms",
        convert_t0.elapsed().as_secs_f64() * 1e3
    );

    let program_t0 = Instant::now();
    let program = logan_ane::mil::parallel_dense_fp16_f32_io(
        HIDDEN,
        SPATIAL,
        &[
            DenseProjection::new("qkv", QKV, qkv16.clone()),
            DenseProjection::new("z", Z, z16.clone()),
            DenseProjection::new("a", A, a16.clone()),
            DenseProjection::new("b", B, b16.clone()),
        ],
    )
    .map_err(|e| e.to_string())?;
    println!(
        "MIL/blob build {:.1} ms",
        program_t0.elapsed().as_secs_f64() * 1e3
    );

    let runtime = AneRuntime::load().map_err(|e| e.to_string())?;
    let compile_t0 = Instant::now();
    let mut model = runtime
        .compile(&program, CompileOptions::default())
        .map_err(|e| e.to_string())?;
    let compile_ms = compile_t0.elapsed().as_secs_f64() * 1e3;
    let load_t0 = Instant::now();
    model.load().map_err(|e| e.to_string())?;
    let ane_load_ms = load_t0.elapsed().as_secs_f64() * 1e3;
    println!("ANE compile={compile_ms:.1} ms load={ane_load_ms:.1} ms");
    if std::env::var("LOGAN_ANE_RESIDENCY_HINT")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        let client = runtime.shared_client().map_err(|e| e.to_string())?;
        let mut hint_ms = Vec::new();
        for _ in 0..8 {
            let t = Instant::now();
            client
                .residency_hint(&model, logan_ane::AneQos::DEFAULT)
                .map_err(|e| e.to_string())?;
            hint_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "residency_hint ms={hint_ms:?} median={:.3}",
            median_ms(hint_ms.clone())
        );
    }

    // Deterministic activation with a distribution similar to normalized LLM hidden states.
    let x: Vec<f32> = (0..HIDDEN)
        .map(|i| ((i as f32 * 0.0137).sin() + (i as f32 * 0.0031).cos()) * 0.25)
        .collect();

    // BNNS baseline: four one-token BF16 GEMVs exactly as the current fallback path does.
    let mut ref_qkv = vec![0.0f32; QKV];
    let mut ref_z = vec![0.0f32; Z];
    let mut ref_a = vec![0.0f32; A];
    let mut ref_b = vec![0.0f32; B];
    let run_bnns =
        |rq: &mut [f32], rz: &mut [f32], ra: &mut [f32], rb: &mut [f32]| -> Result<(), String> {
            if !logan_metal::bnns_bf16_matmul(&qkv.bytes, &x, rq, QKV, HIDDEN)
                || !logan_metal::bnns_bf16_matmul(&z.bytes, &x, rz, Z, HIDDEN)
                || !logan_metal::bnns_bf16_matmul(&a.bytes, &x, ra, A, HIDDEN)
                || !logan_metal::bnns_bf16_matmul(&b.bytes, &x, rb, B, HIDDEN)
            {
                return Err("BNNS BF16 path unavailable".into());
            }
            Ok(())
        };
    for _ in 0..3 {
        run_bnns(&mut ref_qkv, &mut ref_z, &mut ref_a, &mut ref_b)?;
    }
    let mut bnns_samples = Vec::new();
    for _ in 0..15 {
        let t = Instant::now();
        run_bnns(&mut ref_qkv, &mut ref_z, &mut ref_a, &mut ref_b)?;
        bnns_samples.push(t.elapsed().as_secs_f64() * 1e3);
    }

    // Decode-honest ANE input: all 16 spatial slots carry the same token so we
    // can inspect any slot while paying the compiler's actual 16-position cost.
    let mut input_values = vec![0.0f32; HIDDEN * SPATIAL];
    for c in 0..HIDDEN {
        for s in 0..SPATIAL {
            input_values[c * SPATIAL + s] = x[c];
        }
    }
    let mut input = AneSurface::new(input_values.len() * 4).map_err(|e| e.to_string())?;
    input.write_f32(&input_values).map_err(|e| e.to_string())?;
    let out_qkv = AneSurface::new(QKV * SPATIAL * 4).map_err(|e| e.to_string())?;
    let out_z = AneSurface::new(Z * SPATIAL * 4).map_err(|e| e.to_string())?;
    let out_a = AneSurface::new(A * SPATIAL * 4).map_err(|e| e.to_string())?;
    let out_b = AneSurface::new(B * SPATIAL * 4).map_err(|e| e.to_string())?;
    let request = AneRequest::new(&[&input], &[&out_qkv, &out_z, &out_a, &out_b], 0)
        .map_err(|e| e.to_string())?;
    for _ in 0..5 {
        model.evaluate(&request).map_err(|e| e.to_string())?;
    }
    let mut ane_samples = Vec::new();
    for _ in 0..30 {
        let t = Instant::now();
        model.evaluate(&request).map_err(|e| e.to_string())?;
        ane_samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    drop(request);

    // Reusable async-channel cost on the same real Qwen GDN layer. This keeps
    // surface wrappers/request/shared-events/completion block alive and only
    // updates the shared-event value before each submit.
    let direct = std::env::var("LOGAN_ANE_ASYNC_DIRECT")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    let realtime = std::env::var("LOGAN_ANE_ASYNC_REALTIME")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    let mode = if realtime {
        2
    } else if direct {
        1
    } else {
        0
    };
    let premap = std::env::var("LOGAN_ANE_ASYNC_PREMAP")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    let mut fence = logan_metal::MetalAneFence::new(1)
        .ok_or_else(|| "Metal shared event unavailable".to_string())?;
    let mut channel = unsafe {
        model.async_channel(
            &[&input],
            &[&out_qkv, &out_z, &out_a, &out_b],
            0,
            fence.ane_shared_event(),
            mode,
            premap,
        )
    }
    .map_err(|e| e.to_string())?;
    let mut event_value = fence.value();
    for warm in 0..5 {
        if warm > 0 {
            event_value = fence
                .advance()
                .ok_or_else(|| "fence advance failed".to_string())?;
        }
        channel
            .submit(event_value)
            .map_err(|e| e.to_string())?
            .finish(2_000)
            .map_err(|e| e.to_string())?;
    }
    let mut async_submit_samples = Vec::new();
    let mut async_total_samples = Vec::new();
    for _ in 0..30 {
        event_value = fence
            .advance()
            .ok_or_else(|| "fence advance failed".to_string())?;
        let total_t0 = Instant::now();
        let submit_t0 = Instant::now();
        let pending = channel.submit(event_value).map_err(|e| e.to_string())?;
        async_submit_samples.push(submit_t0.elapsed().as_secs_f64() * 1e3);
        pending.finish(2_000).map_err(|e| e.to_string())?;
        async_total_samples.push(total_t0.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "ANE reusable async mode={mode} premap={premap} submit_median={:.3} ms total_median={:.3} ms",
        median_ms(async_submit_samples.clone()),
        median_ms(async_total_samples.clone())
    );

    if std::env::var("LOGAN_ANE_PACKED_PROJ")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        let packed_rows = QKV + Z + A + B;
        let mut packed_weights = Vec::with_capacity(packed_rows * HIDDEN);
        packed_weights.extend_from_slice(&qkv16);
        packed_weights.extend_from_slice(&z16);
        packed_weights.extend_from_slice(&a16);
        packed_weights.extend_from_slice(&b16);
        let packed_program = logan_ane::mil::parallel_dense_fp16_f32_io(
            HIDDEN,
            SPATIAL,
            &[DenseProjection::new("packed", packed_rows, packed_weights)],
        )
        .map_err(|e| e.to_string())?;
        let t0 = Instant::now();
        let mut packed_model = runtime
            .compile(&packed_program, CompileOptions::default())
            .map_err(|e| e.to_string())?;
        let packed_compile_ms = t0.elapsed().as_secs_f64() * 1e3;
        packed_model.load().map_err(|e| e.to_string())?;
        let packed_out = AneSurface::new(packed_rows * SPATIAL * 4).map_err(|e| e.to_string())?;
        let packed_req =
            AneRequest::new(&[&input], &[&packed_out], 0).map_err(|e| e.to_string())?;
        for _ in 0..5 {
            packed_model
                .evaluate(&packed_req)
                .map_err(|e| e.to_string())?;
        }
        let mut samples = Vec::new();
        for _ in 0..30 {
            let t = Instant::now();
            packed_model
                .evaluate(&packed_req)
                .map_err(|e| e.to_string())?;
            samples.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let po = packed_out.read_f32().map_err(|e| e.to_string())?;
        let mut row = 0usize;
        let pq = quality(&ref_qkv, &po[row * SPATIAL..(row + QKV) * SPATIAL], SPATIAL);
        row += QKV;
        let pz = quality(&ref_z, &po[row * SPATIAL..(row + Z) * SPATIAL], SPATIAL);
        row += Z;
        let pa = quality(&ref_a, &po[row * SPATIAL..(row + A) * SPATIAL], SPATIAL);
        row += A;
        let pb = quality(&ref_b, &po[row * SPATIAL..(row + B) * SPATIAL], SPATIAL);
        println!(
            "ANE packed projection compile={packed_compile_ms:.1} ms median={:.3} ms qkv_cos={:.9} z_cos={:.9} a_cos={:.9} b_cos={:.9}",
            median_ms(samples),
            pq.cosine,
            pz.cosine,
            pa.cosine,
            pb.cosine
        );
    }

    // Production-shaped cost: write one repeated token, build a request,
    // evaluate it, then read lane 0 from all four outputs. This captures the
    // IOSurface locking/copy and request-wrapper overhead that kernel-only
    // timing omits.
    let mut prod_qkv = vec![0.0f32; QKV];
    let mut prod_z = vec![0.0f32; Z];
    let mut prod_a = vec![0.0f32; A];
    let mut prod_b = vec![0.0f32; B];
    let mut total_samples = Vec::new();
    let mut write_samples = Vec::new();
    let mut request_eval_samples = Vec::new();
    let mut read_samples = Vec::new();
    for _ in 0..20 {
        let total_t0 = Instant::now();
        let t0 = Instant::now();
        write_repeated(&mut input, &x)?;
        write_samples.push(t0.elapsed().as_secs_f64() * 1e3);

        let t0 = Instant::now();
        let request = AneRequest::new(&[&input], &[&out_qkv, &out_z, &out_a, &out_b], 0)
            .map_err(|e| e.to_string())?;
        model.evaluate(&request).map_err(|e| e.to_string())?;
        request_eval_samples.push(t0.elapsed().as_secs_f64() * 1e3);
        drop(request);

        let t0 = Instant::now();
        read_column(&out_qkv, QKV, &mut prod_qkv)?;
        read_column(&out_z, Z, &mut prod_z)?;
        read_column(&out_a, A, &mut prod_a)?;
        read_column(&out_b, B, &mut prod_b)?;
        read_samples.push(t0.elapsed().as_secs_f64() * 1e3);
        total_samples.push(total_t0.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "production-shaped median total={:.3} ms write={:.3} ms request+eval={:.3} ms reads={:.3} ms",
        median_ms(total_samples),
        median_ms(write_samples),
        median_ms(request_eval_samples),
        median_ms(read_samples),
    );

    let aq = out_qkv.read_f32().map_err(|e| e.to_string())?;
    let az = out_z.read_f32().map_err(|e| e.to_string())?;
    let aa = out_a.read_f32().map_err(|e| e.to_string())?;
    let ab = out_b.read_f32().map_err(|e| e.to_string())?;
    let qq = quality(&ref_qkv, &aq, SPATIAL);
    let qz = quality(&ref_z, &az, SPATIAL);
    let qa = quality(&ref_a, &aa, SPATIAL);
    let qb = quality(&ref_b, &ab, SPATIAL);

    let bnns_med = median_ms(bnns_samples.clone());
    let ane_med = median_ms(ane_samples.clone());
    println!("BNNS samples ms={bnns_samples:?}");
    println!("ANE samples ms={ane_samples:?}");
    println!(
        "median current-BNNS={bnns_med:.3} ms ANE-padded16={ane_med:.3} ms speedup={:.2}x",
        bnns_med / ane_med
    );
    println!(
        "qkv quality rmse={:.6} rel={:.6} max={:.6} cos={:.9}",
        qq.rmse, qq.rel_rmse, qq.max_abs, qq.cosine
    );
    println!(
        "z   quality rmse={:.6} rel={:.6} max={:.6} cos={:.9}",
        qz.rmse, qz.rel_rmse, qz.max_abs, qz.cosine
    );
    println!(
        "a   quality rmse={:.6} rel={:.6} max={:.6} cos={:.9}",
        qa.rmse, qa.rel_rmse, qa.max_abs, qa.cosine
    );
    println!(
        "b   quality rmse={:.6} rel={:.6} max={:.6} cos={:.9}",
        qb.rmse, qb.rel_rmse, qb.max_abs, qb.cosine
    );

    // This is a research gate, not a production enablement gate. Fail only on
    // obviously broken numerics; token-identity/end-to-end quality comes next.
    for (name, q) in [("qkv", &qq), ("z", &qz), ("a", &qa), ("b", &qb)] {
        if !q.cosine.is_finite() || q.cosine < 0.999 {
            return Err(format!(
                "{name} ANE output failed cosine gate: {}",
                q.cosine
            ));
        }
    }
    Ok(())
}
