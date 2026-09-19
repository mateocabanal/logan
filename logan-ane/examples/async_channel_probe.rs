use logan_ane::{AneRuntime, AneSurface, CompileOptions, mil};

fn enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = AneRuntime::load()?;
    if !runtime.device_info().has_ane {
        return Err("ANE unavailable".into());
    }
    let channels = 64usize;
    let spatial = 16usize;
    let elements = channels * spatial;
    let bytes = elements * 4;
    let mut model = runtime.compile(
        &mil::relu_fp32(channels, spatial)?,
        CompileOptions::default(),
    )?;
    let realtime = enabled("LOGAN_ANE_ASYNC_REALTIME");
    if !realtime {
        model.load()?;
    }
    let mut input = AneSurface::new(bytes)?;
    let output = AneSurface::new(bytes)?;
    let values: Vec<f32> = (0..elements)
        .map(|i| ((i as i32 % 29) - 14) as f32 / 8.0)
        .collect();
    input.write_f32(&values)?;
    let mut fence = logan_metal::MetalAneFence::new(1).ok_or("Metal shared event unavailable")?;
    let mode = if enabled("LOGAN_ANE_ASYNC_REALTIME") {
        2
    } else if enabled("LOGAN_ANE_ASYNC_DIRECT") {
        1
    } else {
        0
    };
    let premap = enabled("LOGAN_ANE_ASYNC_PREMAP");
    let mut channel = unsafe {
        model.async_channel(
            &[&input],
            &[&output],
            0,
            fence.ane_shared_event(),
            mode,
            premap,
        )?
    };
    let mut submits = Vec::new();
    for i in 0..20 {
        if i > 0 {
            fence.advance().ok_or("fence advance failed")?;
        }
        let t0 = std::time::Instant::now();
        let p = channel.submit(fence.value())?;
        let us = t0.elapsed().as_secs_f64() * 1e6;
        p.finish(2_000)?;
        submits.push(us);
    }
    let result = output.read_f32()?;
    let max_abs = result
        .iter()
        .zip(&values)
        .map(|(&g, &x)| (g - x.max(0.0)).abs())
        .fold(0.0f32, f32::max);
    let warm = &submits[2..];
    let mean = warm.iter().sum::<f64>() / warm.len() as f64;
    let mut sorted = warm.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted[sorted.len() / 2];
    println!(
        "mode={mode} premap={premap} submit_mean_us={mean:.3} submit_median_us={med:.3} min_us={:.3} max_us={:.3} max_abs_error={max_abs:.8}",
        sorted[0],
        sorted[sorted.len() - 1]
    );
    if max_abs > 0.001 {
        return Err("reusable channel mismatch".into());
    }
    Ok(())
}
