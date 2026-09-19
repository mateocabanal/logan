use std::time::Instant;

use logan_ane::{AneRequest, AneRuntime, AneSurface, CompileOptions};

fn f32_to_f16_exact(v: f32) -> u16 {
    match v {
        -0.75 => 0xba00,
        -0.5 => 0xb800,
        -0.25 => 0xb400,
        0.0 => 0x0000,
        0.25 => 0x3400,
        0.5 => 0x3800,
        0.75 => 0x3a00,
        1.0 => 0x3c00,
        _ => panic!("fixture value is not encoded"),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let i = 256usize;
    let seq = 16usize;
    let (program, layout) =
        logan_ane::mil::parallel_dense_packed_dynamic_fp16_f32_io(i, seq, &[i, i])?;
    println!("layout: {layout:?}");

    let runtime = AneRuntime::load()?;
    let t = Instant::now();
    let mut model = runtime.compile(&program, CompileOptions::default())?;
    println!("compile: {:.3} ms", t.elapsed().as_secs_f64() * 1e3);
    model.load()?;

    let values: Vec<f32> = (0..i * seq)
        .map(|n| match n % 7 {
            0 => -0.75,
            1 => -0.5,
            2 => -0.25,
            3 => 0.0,
            4 => 0.25,
            5 => 0.5,
            _ => 0.75,
        })
        .collect();
    let mut packed = vec![0u16; i * layout.total_spatial];
    for c in 0..i {
        let base = c * layout.total_spatial;
        for s in 0..seq {
            packed[base + s] = f32_to_f16_exact(values[c * seq + s]);
        }
        // Packed weights are W^T[I,O]. Identity and 2*identity.
        packed[base + layout.weight_offsets[0] + c] = 0x3c00;
        packed[base + layout.weight_offsets[1] + c] = 0x4000;
    }

    let mut input = AneSurface::new(packed.len() * 2)?;
    {
        let mut map = input.write()?;
        for (dst, value) in map.chunks_exact_mut(2).zip(packed) {
            dst.copy_from_slice(&value.to_le_bytes());
        }
    }
    let y0 = AneSurface::new(i * seq * 4)?;
    let y1 = AneSurface::new(i * seq * 4)?;
    let req = AneRequest::new(&[&input], &[&y0, &y1], 0)?;
    for _ in 0..5 {
        model.evaluate(&req)?;
    }
    let t = Instant::now();
    for _ in 0..100 {
        model.evaluate(&req)?;
    }
    println!(
        "evaluate: {:.3} us/dispatch",
        t.elapsed().as_secs_f64() * 1e6 / 100.0
    );

    let o0 = y0.read_f32()?;
    let o1 = y1.read_f32()?;
    let e0 = o0
        .iter()
        .zip(&values)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let e1 = o1
        .iter()
        .zip(&values)
        .map(|(a, b)| (a - 2.0 * b).abs())
        .fold(0f32, f32::max);
    println!("identity max abs error: {e0:.8}");
    println!("double max abs error:   {e1:.8}");
    if e0 > 1e-5 || e1 > 1e-5 {
        return Err("packed dynamic matmul failed".into());
    }
    Ok(())
}
