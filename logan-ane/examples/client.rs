use std::time::Instant;

use logan_ane::{AneQos, AneRequest, AneRuntime, AneSurface, CompileOptions, mil};

fn bench(iters: usize, mut f: impl FnMut() -> logan_ane::Result<()>) -> logan_ane::Result<f64> {
    for _ in 0..5 {
        f()?;
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    Ok(t0.elapsed().as_secs_f64() * 1e6 / iters as f64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = AneRuntime::load()?;
    let client = runtime.shared_client()?;

    let channels = 256usize;
    let spatial = 64usize;
    let elements = channels * spatial;
    let bytes = elements * std::mem::size_of::<f32>();

    let mut model = runtime.compile(
        &mil::relu_fp32(channels, spatial)?,
        CompileOptions::default(),
    )?;
    model.load()?;

    let mut input = AneSurface::new(bytes)?;
    let output = AneSurface::new(bytes)?;
    let values: Vec<f32> = (0..elements)
        .map(|i| ((i as i32 % 31) - 15) as f32 / 16.0)
        .collect();
    input.write_f32(&values)?;
    let request = AneRequest::new(&[&input], &[&output], 0)?;

    let convenience_us = bench(100, || model.evaluate(&request))?;
    let client_us = bench(100, || client.evaluate(&model, &request, AneQos::DEFAULT))?;

    let direct_us = if runtime.capabilities().direct_client_evaluation {
        Some(bench(100, || {
            client.evaluate_direct(&model, &request, AneQos::DEFAULT)
        })?)
    } else {
        None
    };

    let mapped_direct_us = if runtime.capabilities().client_request_mapping
        && runtime.capabilities().direct_client_evaluation
    {
        client.map_request(&model, &request, true)?;
        let measured = bench(100, || {
            client.evaluate_direct(&model, &request, AneQos::DEFAULT)
        });
        let unmap = client.unmap_request(&model, &request);
        match (measured, unmap) {
            (Ok(value), Ok(())) => Some(value),
            (Err(error), _) => return Err(error.into()),
            (_, Err(error)) => return Err(error.into()),
        }
    } else {
        None
    };

    let result = output.read_f32()?;
    let max_abs_error = result
        .iter()
        .zip(values.iter())
        .map(|(&actual, &x)| (actual - x.max(0.0)).abs())
        .fold(0.0f32, f32::max);

    println!("device: {:#?}", runtime.device_info());
    println!("_ANEInMemoryModel evaluate: {convenience_us:.3} us/dispatch");
    println!("_ANEClient evaluate:       {client_us:.3} us/dispatch");
    match direct_us {
        Some(value) => println!("_ANEClient direct:         {value:.3} us/dispatch"),
        None => println!("_ANEClient direct:         unavailable"),
    }
    match mapped_direct_us {
        Some(value) => println!("_ANEClient mapped direct:  {value:.3} us/dispatch"),
        None => println!("_ANEClient mapped direct:  unavailable"),
    }
    println!("max abs error: {max_abs_error:.8}");

    if !max_abs_error.is_finite() || max_abs_error > 0.001 {
        return Err(format!("client-path ReLU validation failed: {max_abs_error}").into());
    }

    Ok(())
}
