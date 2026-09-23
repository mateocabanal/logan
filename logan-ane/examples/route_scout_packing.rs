//! RouteScout ANE spatial-packing sweep.
//!
//! The scorer island packs `targets x experts` candidate lanes into the ANE
//! spatial axis. More targets per dispatch amortizes the (fixed) dispatch cost
//! over more layers but must still produce a correct result. This measures the
//! full per-dispatch cost across spatial widths so the runtime can pick a width
//! from evidence rather than from the single 8-target point EXP-013 established.
//!
//! Gate: every width must reproduce the per-lane CPU reference for its own
//! geometry, so a width that "fits" but computes wrongly cannot be selected.
use half::f16;
use logan_ane::{mil, AneRequest, AneRuntime, AneSurface, CompileOptions};
use std::time::Instant;

const F: usize = 16;
const H: usize = 16;
const L: usize = 8;
const O: usize = 1;
const EXPERTS: usize = 256;

fn weights(seed: u64) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    // Deterministic non-trivial weights: an all-identity graph would hide a
    // packing bug that permutes lanes.
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f32 / (1u64 << 53) as f32) * 0.6 - 0.3
    };
    let w0: Vec<u16> = (0..H * F).map(|_| f16::from_f32(next()).to_bits()).collect();
    let w1: Vec<u16> = (0..L * H).map(|_| f16::from_f32(next()).to_bits()).collect();
    let w2: Vec<u16> = (0..O * L).map(|_| f16::from_f32(next()).to_bits()).collect();
    (w0, w1, w2)
}

fn reference(input: &[f32], spatial: usize, w0: &[u16], w1: &[u16], w2: &[u16]) -> Vec<f32> {
    let mut out = vec![0.0f32; spatial];
    for lane in 0..spatial {
        let mut h0 = vec![0.0f32; H];
        for (row, slot) in h0.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for feat in 0..F {
                acc += f16::from_bits(w0[row * F + feat]).to_f32() * input[feat * spatial + lane];
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
        out[lane] = (0..L)
            .map(|row| f16::from_bits(w2[row]).to_f32() * h1[row])
            .sum();
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = AneRuntime::load()?;
    println!("ANE device: {:?}", runtime.device_info());
    println!("targets  spatial  compile_ms  us_per_dispatch  us_per_layer  max_abs  gate");

    let mut failures = 0usize;
    for targets in [1usize, 2, 4, 8, 16] {
        let spatial = targets * EXPERTS;
        let (w0, w1, w2) = weights(0x9E37_79B9_7F4A_7C15 ^ targets as u64);
        let program = match mil::route_scout_mlp_fp16(F, H, L, O, spatial, &w0, &w1, &w2) {
            Ok(program) => program,
            Err(error) => {
                println!("{targets:>7}  {spatial:>7}  unsupported: {error}");
                failures += 1;
                continue;
            }
        };

        let compile_start = Instant::now();
        let mut model = runtime.compile(
            &program,
            CompileOptions {
                cache_directory: Some(std::env::temp_dir().join(format!(
                    "logan-ane-routescout-pack-{targets}"
                ))),
                ..CompileOptions::default()
            },
        )?;
        model.load()?;
        let compile_ms = compile_start.elapsed().as_secs_f64() * 1e3;

        let mut state = 0x2545_F491_4F6C_DD1Du64 ^ spatial as u64;
        let mut values = vec![0.0f32; F * spatial];
        for value in values.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *value = ((state >> 11) as f32 / (1u64 << 53) as f32) - 0.5;
        }
        let mut input = AneSurface::new(F * spatial * 4)?;
        input.write_f32(&values)?;
        let output = AneSurface::new(O * spatial * 4)?;
        let request = AneRequest::new(&[&input], &[&output], 0)?;

        for _ in 0..10 {
            model.evaluate(&request)?;
        }
        let iterations = if spatial >= 8192 { 120 } else { 250 };
        let started = Instant::now();
        for _ in 0..iterations {
            model.evaluate(&request)?;
        }
        let us = started.elapsed().as_secs_f64() * 1e6 / iterations as f64;

        let got = output.read_f32()?;
        let expected = reference(&values, spatial, &w0, &w1, &w2);
        let max_abs = got
            .iter()
            .zip(&expected)
            .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
        let gate = max_abs <= 0.02 && got.iter().all(|v| v.is_finite());
        if !gate {
            failures += 1;
        }
        println!(
            "{targets:>7}  {spatial:>7}  {compile_ms:>10.1}  {us:>15.1}  {:>11.2}  {max_abs:>7.4}  {}",
            us / targets as f64,
            if gate { "PASS" } else { "FAIL" }
        );
    }

    println!(
        "ROUTESCOUT_ANE_PACKING_GATE: {}",
        if failures == 0 { "PASS" } else { "FAIL" }
    );
    if failures > 0 {
        return Err(format!("{failures} spatial widths failed").into());
    }
    Ok(())
}
