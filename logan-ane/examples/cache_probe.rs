use logan_ane::{AneProgramCache, AneRuntime, CompileOptions, mil};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = AneRuntime::load()?;
    let program = mil::relu_fp32(64, 16)?;
    let mut cache = AneProgramCache::new(runtime);

    {
        let model =
            cache.get_or_compile("probe/relu-64x16", &program, CompileOptions::default())?;
        assert!(model.is_loaded());
    }
    let first = cache.stats();

    {
        let model =
            cache.get_or_compile("probe/relu-64x16", &program, CompileOptions::default())?;
        assert!(model.is_loaded());
    }
    let second = cache.stats();

    println!("first: {first:#?}");
    println!("second: {second:#?}");
    if first.misses != 1 || first.hits != 0 || second.misses != 1 || second.hits != 1 {
        return Err("ANE program cache did not record one miss followed by one hit".into());
    }
    Ok(())
}
