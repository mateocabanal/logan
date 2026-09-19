use logan_llama::DenseModel;
use logan_llama::dspark::{DSPARK_TAPS, DsparkGeometry, DsparkSession, DsparkWeights};
use std::{env, sync::Arc, time::Instant};

fn main() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let target_root = args
        .next()
        .ok_or("usage: dspark_projection_probe <target-model> <dspark-model>")?;
    let dspark_root = args
        .next()
        .ok_or("usage: dspark_projection_probe <target-model> <dspark-model>")?;

    let started = Instant::now();
    let target = DenseModel::load(&target_root)?;
    let geometry = DsparkGeometry::minicpm5(target.config.vocab_size as usize);
    let weights = DsparkWeights::load(&dspark_root, geometry)?;
    let mut session = DsparkSession::new(Arc::new(target), weights)?;
    let (_taps, projected) = session.prefill_projected(&[1, 2, 3, 4, 5], &DSPARK_TAPS)?;
    println!(
        "rows={} width={} backend={:?} first={:.6} elapsed_ms={:.3}",
        projected.rows,
        projected.width,
        projected.backend,
        projected.values.first().copied().unwrap_or_default(),
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    Ok(())
}
