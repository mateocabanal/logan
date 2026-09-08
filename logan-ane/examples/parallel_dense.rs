use std::time::Instant;

use logan_ane::{AneRequest, AneRuntime, AneSurface, CompileOptions, DenseProjection, mil};

fn diagonal(channels: usize, fp16_value: u16) -> Vec<u16> {
    let mut values = vec![0u16; channels * channels];
    for channel in 0..channels {
        values[channel * channels + channel] = fp16_value;
    }
    values
}

fn max_error(actual: &[f32], expected: impl Iterator<Item = f32>) -> f32 {
    actual
        .iter()
        .copied()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = AneRuntime::load()?;
    let channels = 256usize;
    let spatial = 64usize;
    let elements = channels * spatial;
    let input_bytes = elements * std::mem::size_of::<f32>();

    let projections = vec![
        DenseProjection::new("identity", channels, diagonal(channels, 0x3c00)),
        DenseProjection::new("double", channels, diagonal(channels, 0x4000)),
    ];
    let program = mil::parallel_dense_fp16_f32_io(channels, spatial, &projections)?;

    let t0 = Instant::now();
    let mut model = runtime.compile(&program, CompileOptions::default())?;
    let compile_ms = t0.elapsed().as_secs_f64() * 1e3;
    model.load()?;

    let mut input = AneSurface::new(input_bytes)?;
    let out_identity = AneSurface::new(input_bytes)?;
    let out_double = AneSurface::new(input_bytes)?;
    let values: Vec<f32> = (0..elements)
        .map(|i| ((i as i32 % 17) - 8) as f32 / 8.0)
        .collect();
    input.write_f32(&values)?;

    let request = AneRequest::new(&[&input], &[&out_identity, &out_double], 0)?;
    for _ in 0..5 {
        model.evaluate(&request)?;
    }
    let iters = 100usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        model.evaluate(&request)?;
    }
    let eval_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let identity = out_identity.read_f32()?;
    let double = out_double.read_f32()?;
    let identity_error = max_error(&identity, values.iter().copied());
    let double_error = max_error(&double, values.iter().map(|v| v * 2.0));

    println!("device: {:#?}", runtime.device_info());
    println!("compile: {compile_ms:.3} ms");
    println!("evaluate: {eval_us:.3} us/dispatch for 2 dense projections");
    println!("identity max abs error: {identity_error:.8}");
    println!("double max abs error: {double_error:.8}");

    if identity_error > 0.001 || double_error > 0.001 {
        return Err("parallel ANE dense island validation failed".into());
    }
    Ok(())
}
