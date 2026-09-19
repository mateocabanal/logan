use logan_llama::{BackendPreference, DenseModel};
use std::{env, sync::Arc, time::Instant};

fn main() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let root = args
        .next()
        .ok_or("usage: minicpm5_decode_probe <model-root> [tokens] [cpu|auto|metal]")?;
    let tokens = args
        .next()
        .map(|value| value.parse::<usize>().map_err(|error| error.to_string()))
        .transpose()?
        .unwrap_or(24);
    let backend = match args.next().as_deref() {
        Some("cpu") => BackendPreference::Cpu,
        Some("metal") => BackendPreference::Metal,
        Some("auto") | None => BackendPreference::Auto,
        Some(value) => return Err(format!("unknown backend `{value}`")),
    };

    let model = Arc::new(DenseModel::load(root)?);
    if env::var_os("LOGAN_DENSE_COMPARE").is_some() {
        let mut reference = model.new_session();
        let mut fused = model.new_session();
        reference.set_backend(BackendPreference::Auto);
        fused.set_backend(BackendPreference::Auto);
        let mut max_abs = 0.0_f32;
        let mut mismatches = 0_usize;
        for ((token, reference_taps), (fused_token, fused_taps)) in
            [(1_u32, &[0_usize][..]), (2, &[0][..]), (3, &[0][..])]
                .into_iter()
                .zip([(1_u32, &[][..]), (2, &[][..]), (3, &[][..])])
        {
            // Run the fused session first so native quantized handles cannot
            // make the reference session accidentally take the fused route.
            let fused_output = fused.forward(&[fused_token], fused_taps)?;
            let reference_output = reference.forward(&[token], reference_taps)?;
            for (left, right) in reference_output.logits.iter().zip(&fused_output.logits) {
                max_abs = max_abs.max((left - right).abs());
            }
            let reference_top = reference_output
                .logits
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index);
            let fused_top = fused_output
                .logits
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index);
            if reference_top != fused_top {
                mismatches += 1;
            }
        }
        println!("compare_max_abs={max_abs:.6e} compare_argmax_mismatches={mismatches}");
        return Ok(());
    }
    let mut session = model.new_session();
    session.set_backend(backend);
    let fused = env::var("LOGAN_DENSE_FUSED").ok().as_deref() != Some("0");
    let tap_ids: &[usize] = if fused { &[] } else { &[0] };
    session.forward(&[1], tap_ids)?;
    let profile = env::var_os("LOGAN_PROFILE").is_some();
    if profile {
        logan_metal::dense_profile_start();
    }
    let started = Instant::now();
    let mut last_token = None;
    for token in 0..tokens {
        let output = session.forward(&[((token as u32) % 100) + 2], tap_ids)?;
        last_token = output
            .logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index);
    }
    if profile {
        let (encode, submit, wait, kernel) = logan_metal::dense_profile_stop();
        println!(
            "profile_encode_ms={:.3} profile_submit_ms={:.3} profile_wait_ms={:.3} profile_kernel_ms={:.3}",
            encode as f64 / 1_000_000.0,
            submit as f64 / 1_000_000.0,
            wait as f64 / 1_000_000.0,
            kernel as f64 / 1_000_000.0,
        );
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "backend={backend:?} tokens={tokens} elapsed_ms={:.3} tok_s={:.3} last_token={last_token:?} last={:?}",
        elapsed * 1_000.0,
        tokens as f64 / elapsed,
        session.backend(),
    );
    Ok(())
}
