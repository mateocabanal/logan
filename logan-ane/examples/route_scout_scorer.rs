//! RouteScout candidate-scorer direct-ANE gate.
//!
//! Two modes:
//!
//! * default (no weights): a deterministic identity-weights oracle check that the
//!   spatial packing and graph are exact.
//! * `--weights <scorer.fp16.bin>`: loads a real trained `16 -> 16 -> 8 -> 1`
//!   scorer exported by `tools/routescout_matrix.py scorer --export`, runs it on
//!   ANE, and checks it against a CPU reference. The gate on the trained path is
//!   *score ordering*, not absolute error: a residual scorer is only useful if it
//!   ranks candidates the same way the reference does, so the same top-k
//!   selection falls out.
use half::f16;
use logan_ane::{mil, AneRequest, AneRuntime, AneSurface, CompileOptions};
use std::{path::PathBuf, time::Instant};

const F: usize = 16;
const H: usize = 16;
const L: usize = 8;
const O: usize = 1;
const TARGETS: usize = 8;
const EXPERTS: usize = 256;
const S: usize = TARGETS * EXPERTS;

fn identity_weights() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut w0 = vec![0u16; H * F];
    for i in 0..H.min(F) {
        w0[i * F + i] = f16::from_f32(1.0).to_bits();
    }
    let mut w1 = vec![0u16; L * H];
    for i in 0..L.min(H) {
        w1[i * H + i] = f16::from_f32(1.0).to_bits();
    }
    let mut w2 = vec![0u16; O * L];
    for i in 0..L {
        w2[i] = f16::from_f32(1.0).to_bits();
    }
    (w0, w1, w2)
}

fn read_weights(path: &PathBuf) -> Result<(Vec<u16>, Vec<u16>, Vec<u16>), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let expected = (H * F + L * H + O * L) * 2;
    if bytes.len() != expected {
        return Err(format!(
            "{}: {} bytes, expected {expected}",
            path.display(),
            bytes.len()
        )
        .into());
    }
    let words: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    let (w0, rest) = words.split_at(H * F);
    let (w1, w2) = rest.split_at(L * H);
    Ok((w0.to_vec(), w1.to_vec(), w2.to_vec()))
}

/// Lanes ordered by descending score (ties broken by lane index).
fn rank_order(scores: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b)));
    order
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut weights_path: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--weights" {
            weights_path = Some(PathBuf::from(
                args.next().ok_or("--weights requires a path")?,
            ));
        }
    }

    let trained = weights_path.is_some();
    let (w0, w1, w2) = match &weights_path {
        Some(path) => read_weights(path)?,
        None => identity_weights(),
    };

    let program = mil::route_scout_mlp_fp16(F, H, L, O, S, &w0, &w1, &w2)?;
    let runtime = AneRuntime::load()?;
    println!(
        "RouteScout scorer geometry: {} -> {} -> {} -> {}, spatial={} ({}x{} candidates) mode={}",
        F,
        H,
        L,
        O,
        S,
        TARGETS,
        EXPERTS,
        if trained { "trained" } else { "identity" }
    );
    println!("ANE device: {:?}", runtime.device_info());

    let compile_start = Instant::now();
    let mut model = runtime.compile(
        &program,
        CompileOptions {
            cache_directory: Some(std::env::temp_dir().join("logan-ane-routescout-scorer-cache")),
            ..CompileOptions::default()
        },
    )?;
    model.load()?;
    println!(
        "compile+load: {:.3} ms",
        compile_start.elapsed().as_secs_f64() * 1e3
    );

    // Deterministic pseudo-features spanning the real feature ranges: the first
    // five features are "route-like" (sparse, non-negative), the router stats
    // are small, and the layer/expert encodings sweep their units.
    let mut values = vec![0.0f32; F * S];
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for feat in 0..F {
        for lane in 0..S {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f32 / (1u64 << 53) as f32;
            values[feat * S + lane] = match feat {
                0 | 1 | 4 => if unit > 0.9 { unit } else { 0.0 },
                2 | 3 => unit,
                14 => unit * if unit > 0.5 { 0.0 } else { 1.0 },
                15 => 1.0,
                11 | 12 | 13 => unit,
                _ => unit * 0.35,
            };
        }
    }

    let mut input = AneSurface::new(F * S * 4)?;
    input.write_f32(&values)?;
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
    let us = started.elapsed().as_secs_f64() * 1e6 / iterations as f64;
    let got = output.read_f32()?;

    // The graph is a 1x1 convolution, so output lane `s` depends only on input
    // lane `s`: expected[s] = MLP(x[:, s]). No cross-lane mixing.
    let cpu_lane = |lane: usize| -> f32 {
        let mut h0 = vec![0.0f32; H];
        for (row, slot) in h0.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for feat in 0..F {
                acc += f16::from_bits(w0[row * F + feat]).to_f32() * values[feat * S + lane];
            }
            *slot = acc.max(0.0);
        }
        let mut h1 = vec![0.0f32; L];
        for (row, slot) in h1.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for feat in 0..H {
                acc += f16::from_bits(w1[row * H + feat]).to_f32() * h0[feat];
            }
            *slot = acc.max(0.0);
        }
        let mut out = 0.0f32;
        for row in 0..L {
            out += f16::from_bits(w2[row]).to_f32() * h1[row];
        }
        out
    };

    let reference: Vec<f32> = (0..S).map(cpu_lane).collect();
    let mut max_abs = 0.0f32;
    let mut rms = 0.0f64;
    for lane in 0..S {
        let error = (got[lane] - reference[lane]).abs();
        max_abs = max_abs.max(error);
        rms += (error as f64) * (error as f64);
    }
    rms = (rms / S as f64).sqrt();

    let scale = reference
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()))
        .max(f32::MIN_POSITIVE);
    println!("evaluate: {us:.3} us/dispatch over {S} candidate lanes");
    println!("oracle: max_abs={max_abs:.6} rms={rms:.6} reference_scale={scale:.3}");

    // Rank agreement: a residual scorer is consumed by taking the top-k
    // candidates, so the gate is that ANE and CPU order candidates the same way.
    let ane_rank = rank_order(&got);
    let cpu_rank = rank_order(&reference);
    let top_k = EXPERTS;
    let ane_top: Vec<usize> = ane_rank[..top_k].iter().copied().collect();
    let cpu_top: Vec<usize> = cpu_rank[..top_k].iter().copied().collect();
    let overlap = ane_top.iter().filter(|lane| cpu_top.contains(lane)).count();
    let top1_match = ane_rank[0] == cpu_rank[0];
    println!(
        "ordering: top{top_k}_overlap={overlap}/{top_k} top1_match={top1_match} spread={:.6}",
        got.iter().fold(f32::MIN, |a, v| a.max(*v)) - got.iter().fold(f32::MAX, |a, v| a.min(*v))
    );

    let gate = if trained {
        // Trained fp16 weights summed over F*S terms in a different order make a
        // tight absolute tolerance meaningless; ordering is the contract.
        top1_match && overlap * 100 >= top_k * 99 && got.iter().all(|v| v.is_finite())
    } else {
        max_abs <= 0.01
    };

    println!(
        "ROUTESCOUT_SCORER_ANE_GATE: {}",
        if gate { "PASS" } else { "FAIL" }
    );
    if !gate {
        return Err("RouteScout scorer ANE gate failed".into());
    }
    Ok(())
}
