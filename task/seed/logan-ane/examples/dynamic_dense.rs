use std::time::Instant;

use logan_ane::{AneRequest, AneRuntime, AneSurface, CompileOptions};

fn write_u16(surface: &mut AneSurface, values: &[u16]) -> logan_ane::Result<()> {
    if surface.len() != values.len() * 2 {
        return Err(logan_ane::AneError::InvalidArgument(
            "u16 surface size mismatch".into(),
        ));
    }
    let mut map = surface.write()?;
    for (chunk, value) in map.chunks_exact_mut(2).zip(values.iter().copied()) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

fn diagonal(channels: usize, value: u16) -> Vec<u16> {
    let mut w = vec![0u16; channels * channels];
    for i in 0..channels {
        w[i * channels + i] = value;
    }
    w
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let channels = 256usize;
    let spatial = 16usize;
    let runtime = AneRuntime::load()?;
    let program = logan_ane::mil::parallel_dense_dynamic_fp16_f32_io(
        channels,
        spatial,
        &[channels, channels],
    )?;
    let compile_t0 = Instant::now();
    let mut model = runtime.compile(&program, CompileOptions::default())?;
    println!(
        "compile: {:.3} ms",
        compile_t0.elapsed().as_secs_f64() * 1e3
    );
    model.load()?;

    let mut x = AneSurface::new(channels * spatial * 4)?;
    let mut w0 = AneSurface::new(channels * channels * 2)?;
    let mut w1 = AneSurface::new(channels * channels * 2)?;
    let y0 = AneSurface::new(channels * spatial * 4)?;
    let y1 = AneSurface::new(channels * spatial * 4)?;

    let values: Vec<f32> = (0..channels * spatial)
        .map(|i| ((i % 13) as i32 - 6) as f32 / 8.0)
        .collect();
    x.write_f32(&values)?;
    write_u16(&mut w0, &diagonal(channels, 0x3c00))?;
    write_u16(&mut w1, &diagonal(channels, 0x4000))?;

    let request = AneRequest::new(&[&x, &w0, &w1], &[&y0, &y1], 0)?;
    for _ in 0..5 {
        model.evaluate(&request)?;
    }
    let t0 = Instant::now();
    for _ in 0..100 {
        model.evaluate(&request)?;
    }
    println!(
        "evaluate: {:.3} us/dispatch",
        t0.elapsed().as_secs_f64() * 1e6 / 100.0
    );

    let o0 = y0.read_f32()?;
    let o1 = y1.read_f32()?;
    let e0 = o0
        .iter()
        .zip(&values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let e1 = o1
        .iter()
        .zip(&values)
        .map(|(a, b)| (a - 2.0 * b).abs())
        .fold(0.0f32, f32::max);
    println!("identity max abs error: {e0:.8}");
    println!("double max abs error:   {e1:.8}");
    if e0 > 1e-5 || e1 > 1e-5 {
        return Err(format!("dynamic dense correctness failure: {e0}, {e1}").into());
    }
    Ok(())
}
