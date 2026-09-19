use std::time::Instant;

use logan_ane::{AneRequest, AneRuntime, AneSurface, CompileOptions, mil};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = AneRuntime::load()?;
    if !runtime.device_info().has_ane {
        return Err("this machine does not report an ANE".into());
    }

    let channels = 256usize;
    let spatial = 64usize;
    let elements = channels * spatial;
    let bytes = elements * std::mem::size_of::<f32>();
    let program = mil::relu_fp32(channels, spatial)?;

    let compile_t0 = Instant::now();
    let mut model = runtime.compile(&program, CompileOptions::default())?;
    let compile_ms = compile_t0.elapsed().as_secs_f64() * 1e3;
    println!("compiled_model_exists_after_compile={:?}", model.compiled_model_exists());
    println!("local_model_path_after_compile={:?}", model.local_model_path());
    println!("temporary_directory={}", model.temporary_directory().display());

    let load_t0 = Instant::now();
    model.load()?;
    let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;

    let mut input = AneSurface::new(bytes)?;
    let output = AneSurface::new(bytes)?;
    let values: Vec<f32> = (0..elements)
        .map(|i| ((i as i32 % 31) - 15) as f32 / 16.0)
        .collect();
    input.write_f32(&values)?;

    let echoed = input.read_f32()?;
    let input_error = echoed
        .iter()
        .zip(&values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    if input_error != 0.0 {
        return Err(format!("IOSurface CPU roundtrip failed: max error {input_error}").into());
    }

    let request = AneRequest::new(&[&input], &[&output], 0)?;
    for _ in 0..5 {
        model.evaluate(&request)?;
    }

    let iters = 100usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        model.evaluate(&request)?;
    }
    let eval_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let result = output.read_f32()?;
    let expected = values.iter().map(|&x| x.max(0.0));
    let max_abs_error = result
        .iter()
        .copied()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("device: {:#?}", runtime.device_info());
    println!("compile: {compile_ms:.3} ms");
    println!("load: {load_ms:.3} ms");
    println!("evaluate: {eval_us:.3} us/dispatch");
    println!("model state: {:#?}", model.state());
    println!("max abs error: {max_abs_error:.8}");

    if !max_abs_error.is_finite() || max_abs_error > 0.001 {
        return Err(format!("ReLU result failed validation: max error {max_abs_error}").into());
    }
    Ok(())
}
