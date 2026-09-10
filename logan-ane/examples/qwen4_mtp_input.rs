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

    // Same topology as Qwen3.8-Flash-Next's MTP input fusion: one H-wide
    // embedding stream and four H-wide hidden streams sharing fc_hidden.
    let hidden_size = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(256);
    let iterations = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100);
    let hc_count = 4usize;
    let token_spatial = 16usize;
    let hidden_spatial = hc_count * token_spatial;
    let packed_spatial = token_spatial + hidden_spatial;

    let program = mil::qwen4_mtp_input_projections_fp16_f32_io(
        hidden_size,
        token_spatial,
        hc_count,
        DenseProjection::new(
            "fc_embedding",
            hidden_size,
            diagonal(hidden_size, 0x3c00), // 1.0
        ),
        DenseProjection::new(
            "fc_hidden",
            hidden_size,
            diagonal(hidden_size, 0x3800), // 0.5
        ),
    )?;

    let t0 = Instant::now();
    let mut model = runtime.compile(&program, CompileOptions::default())?;
    let compile_ms = t0.elapsed().as_secs_f64() * 1e3;
    model.load()?;

    let embedding_values = (0..hidden_size * token_spatial)
        .map(|i| ((i as i32 % 17) - 8) as f32 / 8.0)
        .collect::<Vec<_>>();
    let hidden_values = (0..hidden_size * hidden_spatial)
        .map(|i| ((i as i32 % 23) - 11) as f32 / 16.0)
        .collect::<Vec<_>>();

    // ANE gets one activation surface. Each channel stores the embedding lanes
    // first, followed by the four HC branch regions.
    let mut packed_values = vec![0.0f32; hidden_size * packed_spatial];
    for channel in 0..hidden_size {
        let packed_base = channel * packed_spatial;
        let embedding_base = channel * token_spatial;
        let hidden_base = channel * hidden_spatial;
        packed_values[packed_base..packed_base + token_spatial]
            .copy_from_slice(&embedding_values[embedding_base..embedding_base + token_spatial]);
        packed_values[packed_base + token_spatial..packed_base + packed_spatial]
            .copy_from_slice(&hidden_values[hidden_base..hidden_base + hidden_spatial]);
    }

    let mut packed = AneSurface::new(packed_values.len() * 4)?;
    let embedding_out = AneSurface::new(packed_values.len() * 4)?;
    let hidden_out = AneSurface::new(packed_values.len() * 4)?;
    packed.write_f32(&packed_values)?;

    let request = AneRequest::new(&[&packed], &[&embedding_out, &hidden_out], 0)?;

    for _ in 0..5 {
        model.evaluate(&request)?;
    }
    let t0 = Instant::now();
    for _ in 0..iterations {
        model.evaluate(&request)?;
    }
    let eval_us = t0.elapsed().as_secs_f64() * 1e6 / iterations as f64;

    let embedding_actual = embedding_out.read_f32()?;
    let hidden_actual = hidden_out.read_f32()?;
    let embedding_error = max_error(&embedding_actual, packed_values.iter().copied());
    let hidden_error = max_error(&hidden_actual, packed_values.iter().map(|v| v * 0.5));

    println!("device: {:#?}", runtime.device_info());
    println!("compile: {compile_ms:.3} ms");
    println!(
        "evaluate: {eval_us:.3} us/dispatch for MTP fc_embedding + shared fc_hidden x{hc_count}"
    );
    println!("embedding max abs error: {embedding_error:.8}");
    println!("hidden max abs error: {hidden_error:.8}");

    if embedding_error > 0.001 || hidden_error > 0.001 {
        return Err("Qwen4 MTP ANE input-projection validation failed".into());
    }
    Ok(())
}
