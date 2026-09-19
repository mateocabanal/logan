use logan_ane::{AneRuntime, AneSurface, CompileOptions, mil};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = AneRuntime::load()?;
    if !runtime.device_info().has_ane { return Err("ANE unavailable".into()); }
    if !logan_metal::metal_init() { return Err("Metal unavailable".into()); }

    let channels = 64usize;
    let spatial = 16usize;
    let elements = channels * spatial;
    let bytes = elements * std::mem::size_of::<f32>();
    let mut model = runtime.compile(&mil::relu_fp32(channels, spatial)?, CompileOptions::default())?;
    model.load()?;
    let mut input = AneSurface::new(bytes)?;
    let output = AneSurface::new(bytes)?;
    let values: Vec<f32> = (0..elements).map(|i| ((i as i32 % 29) - 14) as f32 / 8.0).collect();
    input.write_f32(&values)?;

    let fence = logan_metal::MetalAneFence::new(1).ok_or("Metal shared event unavailable")?;
    let t0 = std::time::Instant::now();
    let pending = unsafe {
        model.evaluate_async_signal(
            &[&input], &[&output], 0,
            fence.ane_shared_event(), fence.value(),
            std::env::var("LOGAN_ANE_ASYNC_DIRECT").map(|v| v != "0").unwrap_or(false),
        )?
    };
    let submit_us = t0.elapsed().as_secs_f64() * 1e6;
    pending.finish(2_000)?;
    let result = output.read_f32()?;
    let max_abs = result.iter().zip(&values)
        .map(|(&got, &x)| (got - x.max(0.0)).abs())
        .fold(0.0f32, f32::max);
    println!("async_submit_us={submit_us:.3} max_abs_error={max_abs:.8}");
    if max_abs > 0.001 { return Err(format!("async ANE result mismatch: {max_abs}").into()); }
    Ok(())
}
