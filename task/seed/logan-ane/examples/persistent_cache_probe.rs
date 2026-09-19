//! Force a native cache miss with persistent inputs, then verify a warm hit.
use logan_ane::{AneRequest, AneRuntime, AneSurface, CompileOptions, mil};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = AneRuntime::load()?;
    let program = mil::relu_fp32(64, 16)?;
    let root = std::path::PathBuf::from(std::env::var_os("HOME").ok_or("HOME missing")?)
        .join(".cache/logan/ane/persistent-cache-probe");
    for reuse in [false, true] {
        let mut options = CompileOptions::default();
        options.cache_directory = Some(root.clone());
        options.reuse_compiled_model = reuse;
        let mut model = runtime.compile(&program, options)?;
        assert_eq!(model.native_cache_hit(), reuse);
        model.load()?;
        let mut input = AneSurface::new(64 * 16 * 4)?;
        let output = AneSurface::new(64 * 16 * 4)?;
        let values: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) / 16.0).collect();
        input.write_f32(&values)?;
        let request = AneRequest::new(&[&input], &[&output], 0)?;
        model.evaluate(&request)?;
        let got = output.read_f32()?;
        assert!(got.iter().zip(&values).all(|(&a, &b)| a == b.max(0.0)));
        println!("reuse={reuse} native_cache_hit={} exact_relu=true", model.native_cache_hit());
    }
    Ok(())
}
