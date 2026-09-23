use half::f16;
use logan_ane::{mil, AneRequest, AneRuntime, AneSurface, CompileOptions, DenseProjection};
use std::time::Instant;

const INPUT: usize = 2048;
const OUTPUT: usize = 256;
const SPATIAL: usize = 16;
const LIVE: usize = 1;

fn projection(which: usize) -> DenseProjection {
    let mut w = vec![0u16; OUTPUT * INPUT];
    for out in 0..OUTPUT {
        let col = (out * 7 + which * 31) % INPUT;
        w[out * INPUT + col] = f16::from_f32(1.0).to_bits();
    }
    DenseProjection::new(format!("router_{which}"), OUTPUT, w)
}

fn run(runtime: &AneRuntime, count: usize) -> Result<(), Box<dyn std::error::Error>> {
    let projections: Vec<_> = (0..count).map(projection).collect();
    let program = mil::parallel_dense_fp16_f32_io(INPUT, SPATIAL, &projections)?;
    let cache = std::env::temp_dir().join(format!("logan-ane-routescout-router-{count}"));

    let t0 = Instant::now();
    let mut model = runtime.compile(
        &program,
        CompileOptions {
            cache_directory: Some(cache),
            ..CompileOptions::default()
        },
    )?;
    model.load()?;
    let compile_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut values = vec![0.0f32; INPUT * SPATIAL];
    for ch in 0..INPUT {
        values[ch * SPATIAL] = ((ch % 251) as f32 - 125.0) / 128.0;
    }
    let mut input = AneSurface::new(INPUT * SPATIAL * 4)?;
    input.write_f32(&values)?;
    let outputs: Vec<AneSurface> = (0..count)
        .map(|_| AneSurface::new(OUTPUT * SPATIAL * 4))
        .collect::<Result<_, _>>()?;
    let out_refs: Vec<&AneSurface> = outputs.iter().collect();
    let request = AneRequest::new(&[&input], &out_refs, 0)?;

    for _ in 0..10 {
        model.evaluate(&request)?;
    }
    let iters = 200usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        model.evaluate(&request)?;
    }
    let eval_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let mut max_abs = 0.0f32;
    for (which, output) in outputs.iter().enumerate() {
        let got = output.read_f32()?;
        for out in 0..OUTPUT {
            let col = (out * 7 + which * 31) % INPUT;
            let expected = values[col * SPATIAL];
            max_abs = max_abs.max((got[out * SPATIAL] - expected).abs());
            for lane in LIVE..SPATIAL {
                max_abs = max_abs.max(got[out * SPATIAL + lane].abs());
            }
        }
    }

    println!(
        "routers={count} weights={:.2} MiB compile+load={compile_ms:.3} ms evaluate={eval_us:.3} us max_abs={max_abs:.8}",
        count as f64 * OUTPUT as f64 * INPUT as f64 * 2.0 / (1024.0 * 1024.0)
    );
    if max_abs > 0.002 {
        return Err(format!("router island oracle failed for {count}: {max_abs}").into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = AneRuntime::load()?;
    println!("ANE device: {:?}", runtime.device_info());
    for count in [1usize, 2, 4, 8] {
        run(&runtime, count)?;
    }
    println!("ROUTESCOUT_FUTURE_ROUTER_GATE: PASS");
    Ok(())
}
