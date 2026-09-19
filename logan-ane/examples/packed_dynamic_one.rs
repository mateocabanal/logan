use logan_ane::{AneRequest, AneRuntime, AneSurface, CompileOptions};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let i = 64usize;
    let seq = 64usize;
    let (program, layout) = logan_ane::mil::parallel_dense_packed_dynamic_f32_io(i, seq, &[i])?;
    let rt = AneRuntime::load()?;
    let opts = CompileOptions {
        reuse_compiled_model: false,
        keep_temporary_files: true,
        ..CompileOptions::default()
    };
    let t = Instant::now();
    let mut model = rt.compile(&program, opts)?;
    println!(
        "compile_ms={:.3} layout={layout:?}",
        t.elapsed().as_secs_f64() * 1e3
    );
    model.load()?;
    let mut packed = vec![0.0f32; i * layout.total_spatial];
    for d in 0..i {
        for s in 0..seq {
            packed[d * layout.total_spatial + s] = (d * seq + s) as f32 * 0.001;
        }
        for c in 0..i {
            packed[d * layout.total_spatial + layout.weight_offsets[0] + c] =
                if d == c { 1.0 } else { 0.0 };
        }
    }
    let mut input = AneSurface::new(packed.len() * 4)?;
    input.write_f32(&packed)?;
    let output = AneSurface::new(i * seq * 4)?;
    let req = AneRequest::new(&[&input], &[&output], 0)?;
    for _ in 0..5 {
        model.evaluate(&req)?;
    }
    let t = Instant::now();
    for _ in 0..100 {
        model.evaluate(&req)?;
    }
    println!("eval_us={:.3}", t.elapsed().as_secs_f64() * 1e6 / 100.0);
    let got = output.read_f32()?;
    let mut max = 0.0f32;
    for d in 0..i {
        for s in 0..seq {
            max = max.max((got[d * seq + s] - packed[d * layout.total_spatial + s]).abs());
        }
    }
    println!("max_err={max:.8}");
    if max > 0.01 {
        return Err("packed dynamic f32 mismatch".into());
    }
    Ok(())
}
