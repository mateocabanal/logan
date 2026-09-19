use logan_llama::dspark::{
    DsparkGeometry, DsparkSession, DsparkWeights, VerificationOptions, DSPARK_TAPS,
};
use logan_llama::{BackendPreference, DenseModel};
use std::{env, sync::Arc, time::Instant};

fn main() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let target_root = args
        .next()
        .ok_or("usage: dspark_verify_probe <target-model> <dspark-model> [drafts]")?;
    let dspark_root = args
        .next()
        .ok_or("usage: dspark_verify_probe <target-model> <dspark-model> [drafts]")?;
    let drafts = args
        .next()
        .map(|value| value.parse::<usize>().map_err(|error| error.to_string()))
        .transpose()?
        .unwrap_or(7);

    let target = Arc::new(DenseModel::load(target_root)?);
    let geometry = DsparkGeometry::minicpm5(target.config.vocab_size as usize);
    let weights = DsparkWeights::load(dspark_root, geometry)?;
    let mut session = DsparkSession::new(target, weights)?;
    session.set_backend(BackendPreference::Auto);
    session.prefill(&[1], &DSPARK_TAPS)?;
    let proposal = session.propose(1, drafts)?;

    let started = Instant::now();
    let (result, _) = session.verify_block(1, &proposal.tokens, &VerificationOptions::default())?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    println!(
        "drafts={} verify_ms={elapsed_ms:.3} accepted={} retained={} all_match={} backend={:?}",
        proposal.tokens.len(),
        result.accepted_drafts,
        result.retained_rows,
        result.all_match,
        session.target().backend(),
    );
    Ok(())
}
