use half::f16;
use logan_ane::{mil, AneRequest, AneRuntime, AneSurface, CompileOptions};
use std::{fs, path::Path, time::Instant};

fn identity_rows(rows: usize, cols: usize) -> Vec<u16> {
    let mut w = vec![0u16; rows * cols];
    for row in 0..rows.min(cols) {
        w[row * cols + row] = f16::from_f32(1.0).to_bits();
    }
    w
}

fn read_fp16_weights(
    path: &Path,
    w0_len: usize,
    w1_len: usize,
    w2_len: usize,
) -> Result<(Vec<u16>, Vec<u16>, Vec<u16>), Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let expected = (w0_len + w1_len + w2_len) * 2;
    if bytes.len() != expected {
        return Err(format!(
            "RouteScout weights {} have {} bytes, expected {}",
            path.display(),
            bytes.len(),
            expected
        )
        .into());
    }
    let words: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    let (w0, rest) = words.split_at(w0_len);
    let (w1, w2) = rest.split_at(w1_len);
    Ok((w0.to_vec(), w1.to_vec(), w2.to_vec()))
}

fn fp16_linear(
    weights: &[u16],
    rows: usize,
    cols: usize,
    spatial: usize,
    input: &[f32],
    relu: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * spatial];
    for row in 0..rows {
        for lane in 0..spatial {
            let mut acc = 0.0f32;
            for col in 0..cols {
                acc += f16::from_bits(weights[row * cols + col]).to_f32()
                    * input[col * spatial + lane];
            }
            if relu {
                acc = acc.max(0.0);
            }
            // The MIL graph materializes an fp16 tensor at every layer boundary.
            out[row * spatial + lane] = f16::from_f32(acc).to_f32();
        }
    }
    out
}

fn cpu_reference(
    input: &[f32],
    spatial: usize,
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    i: usize,
    h: usize,
    l: usize,
    o: usize,
) -> Vec<f32> {
    let h0 = fp16_linear(w0, h, i, spatial, input, true);
    let h1 = fp16_linear(w1, l, h, spatial, &h0, true);
    fp16_linear(w2, o, l, spatial, &h1, false)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // RouteScout v0.1 physical ANE island. Eight live forecasts fit in the
    // first half of S=16; the remaining lanes are padding for M2 ANE geometry.
    const I: usize = 96;
    const H: usize = 96;
    const L: usize = 64;
    const O: usize = 256;
    const S: usize = 16;
    const LIVE: usize = 8;

    let (w0, w1, w2) = if let Some(path) = std::env::var_os("ROUTESCOUT_WEIGHTS") {
        let path = std::path::PathBuf::from(path);
        println!("RouteScout weights: {}", path.display());
        read_fp16_weights(&path, H * I, L * H, O * L)?
    } else {
        let w0 = identity_rows(H, I);
        let w1 = identity_rows(L, H);
        let mut w2 = vec![0u16; O * L];
        for out in 0..O {
            w2[out * L + (out % L)] = f16::from_f32(1.0).to_bits();
        }
        (w0, w1, w2)
    };

    let program = mil::route_scout_mlp_fp16(I, H, L, O, S, &w0, &w1, &w2)?;
    let runtime = AneRuntime::load()?;
    println!(
        "RouteScout geometry: {} -> {} -> {} -> {}, spatial={} ({} live)",
        I, H, L, O, S, LIVE
    );
    println!("ANE device: {:?}", runtime.device_info());

    let compile_start = Instant::now();
    let mut model = runtime.compile(
        &program,
        CompileOptions {
            cache_directory: Some(std::env::temp_dir().join("logan-ane-routescout-cache")),
            ..CompileOptions::default()
        },
    )?;
    model.load()?;
    println!(
        "compile+load: {:.3} ms",
        compile_start.elapsed().as_secs_f64() * 1e3
    );

    let mut input_values = vec![0.0_f32; I * S];
    for channel in 0..I {
        for lane in 0..LIVE {
            // Positive values make both ReLUs identity for this oracle.
            let value = 0.01 + ((channel * LIVE + lane) % 251) as f32 / 512.0;
            input_values[channel * S + lane] = value;
        }
    }

    let mut input = AneSurface::new(I * S * 4)?;
    input.write_f32(&input_values)?;
    let output = AneSurface::new(O * S * 4)?;
    let request = AneRequest::new(&[&input], &[&output], 0)?;

    for _ in 0..10 {
        model.evaluate(&request)?;
    }
    let iterations = 250usize;
    let started = Instant::now();
    for _ in 0..iterations {
        model.evaluate(&request)?;
    }
    let elapsed_us = started.elapsed().as_secs_f64() * 1e6 / iterations as f64;

    let got = output.read_f32()?;
    let expected = cpu_reference(&input_values, S, &w0, &w1, &w2, I, H, L, O);
    let mut max_abs = 0.0f32;
    let mut rms_sum = 0.0f64;
    let mut ref_sum = 0.0f64;
    let mut dot = 0.0f64;
    let mut got_norm = 0.0f64;
    let mut ref_norm = 0.0f64;
    let mut n = 0usize;
    for (&actual, &reference) in got.iter().zip(&expected) {
        let error = (actual - reference).abs();
        max_abs = max_abs.max(error);
        rms_sum += (error as f64) * (error as f64);
        ref_sum += (reference as f64) * (reference as f64);
        dot += actual as f64 * reference as f64;
        got_norm += (actual as f64) * (actual as f64);
        ref_norm += (reference as f64) * (reference as f64);
        n += 1;
    }
    let rms = (rms_sum / n as f64).sqrt();
    let rel_l2 = if ref_sum > 0.0 {
        (rms_sum / ref_sum).sqrt()
    } else {
        0.0
    };
    let cosine = if got_norm > 0.0 && ref_norm > 0.0 {
        dot / (got_norm.sqrt() * ref_norm.sqrt())
    } else {
        1.0
    };

    println!("evaluate: {elapsed_us:.3} us/dispatch");
    println!("oracle: max_abs={max_abs:.8} rms={rms:.8} rel_l2={rel_l2:.8} cosine={cosine:.9}");
    if max_abs > 0.05 || rel_l2 > 0.01 || cosine < 0.999 {
        return Err(format!(
            "RouteScout ANE oracle mismatch: max_abs={max_abs} rel_l2={rel_l2} cosine={cosine}"
        )
        .into());
    }
    println!("ROUTESCOUT_ANE_GATE: PASS");
    Ok(())
}
