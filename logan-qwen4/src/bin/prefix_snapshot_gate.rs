use std::path::Path;
use std::time::Instant;

use logan_qwen4::colisource::ColiSource;
use logan_qwen4::{load_cfg, Model};

fn parse_tokens(name: &str, default: &str) -> Vec<u32> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split_whitespace()
        .map(|t| t.parse::<u32>().expect("token id"))
        .collect()
}

fn load_model(path: &Path) -> Result<Model, String> {
    let cfg = load_cfg(&path.join("config.json"))?;
    let src = ColiSource::open(path)?;
    Model::load_coli(&src, &cfg)
}

fn prefill(model: &mut Model, tokens: &[u32], start_pos: usize) {
    for (i, &token) in tokens.iter().enumerate() {
        model.forward_token(token as usize, start_pos + i);
    }
}

fn generate_trace(
    model: &mut Model,
    mut last: u32,
    start_pos: usize,
    count: usize,
) -> (Vec<u32>, Vec<Vec<u32>>) {
    let mut tokens = Vec::with_capacity(count);
    let mut logits_bits = Vec::with_capacity(count);
    for pos in start_pos..start_pos + count {
        let logits = model.forward_token(last as usize, pos);
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        logits_bits.push(logits.iter().map(|v| v.to_bits()).collect());
        tokens.push(next);
        last = next;
    }
    (tokens, logits_bits)
}

fn main() -> Result<(), String> {
    let package = std::env::args()
        .nth(1)
        .ok_or_else(|| "usage: prefix_snapshot_gate <.coli-package>".to_string())?;
    let package = Path::new(&package);

    let prefix = parse_tokens("QWEN_GATE_PREFIX", "1 2 3 4 5");
    let suffix = parse_tokens("QWEN_GATE_SUFFIX", "7 9 11");
    if prefix.is_empty() || suffix.is_empty() {
        return Err("gate prefix and suffix must be non-empty".into());
    }
    let new_tokens = std::env::var("QWEN_GATE_NEW")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);
    let dirty: Vec<u32> = prefix.iter().map(|&t| t.saturating_add(1)).collect();

    // A: canonical fresh execution. Capture the reusable state immediately
    // after the common prefix, then continue normally.
    let mut a = load_model(package)?;
    let prefix_t0 = Instant::now();
    prefill(&mut a, &prefix, 0);
    let cold_prefix_ms = prefix_t0.elapsed().as_secs_f64() * 1e3;

    let snapshot_t0 = Instant::now();
    let prefix_snapshot = a.snapshot_state(prefix.len())?;
    let snapshot_ms = snapshot_t0.elapsed().as_secs_f64() * 1e3;

    prefill(&mut a, &suffix, prefix.len());
    let prompt_len = prefix.len() + suffix.len();
    let (a_tokens, a_logits) =
        generate_trace(&mut a, *suffix.last().unwrap(), prompt_len, new_tokens);
    let a_final = a.snapshot_state(prompt_len + new_tokens)?;
    drop(a);

    // B: use the same loaded-model regime a real server would use. First
    // warm every lazy weight/backend allocation with unrelated tokens. Then
    // reset causal state, measure a warmed prefix replay, dirty state again,
    // and finally measure restore of A's exact prefix checkpoint.
    let mut b = load_model(package)?;
    let empty = b.snapshot_state(0)?;
    prefill(&mut b, &dirty, 0);
    b.restore_state(&empty)?;

    let replay_t0 = Instant::now();
    prefill(&mut b, &prefix, 0);
    let replay_ms = replay_t0.elapsed().as_secs_f64() * 1e3;
    let replay_snapshot = b.snapshot_state(prefix.len())?;
    let replay_exact = replay_snapshot.exact_eq(&prefix_snapshot);

    // Prove restore actually replaces unrelated recurrent/KV/QSA/PLE state.
    b.restore_state(&empty)?;
    prefill(&mut b, &dirty, 0);
    let restore_t0 = Instant::now();
    b.restore_state(&prefix_snapshot)?;
    let restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
    let restored_exact = b.snapshot_state(prefix.len())?.exact_eq(&prefix_snapshot);

    prefill(&mut b, &suffix, prefix.len());
    let (b_tokens, b_logits) =
        generate_trace(&mut b, *suffix.last().unwrap(), prompt_len, new_tokens);
    let b_final = b.snapshot_state(prompt_len + new_tokens)?;

    let tokens_exact = a_tokens == b_tokens;
    let logits_exact = a_logits == b_logits;
    let final_state_exact = a_final.exact_eq(&b_final);
    let ratio = if replay_ms > 0.0 {
        restore_ms / replay_ms
    } else {
        f64::INFINITY
    };
    let perf_pass = ratio < 0.25;

    println!("prefix_tokens={}", prefix.len());
    println!(
        "snapshot_payload={:.2} MiB snapshot_create={:.2} ms",
        prefix_snapshot.payload_bytes() as f64 / (1024.0 * 1024.0),
        snapshot_ms
    );
    println!(
        "cold_prefix={cold_prefix_ms:.2} ms warmed_replay={replay_ms:.2} ms restore={restore_ms:.2} ms restore/replay={ratio:.4}x"
    );
    println!(
        "exact: replay_state={} restored_state={} tokens={} logits={} final_state={}",
        replay_exact, restored_exact, tokens_exact, logits_exact, final_state_exact
    );
    println!("generated={b_tokens:?}");

    if replay_exact
        && restored_exact
        && tokens_exact
        && logits_exact
        && final_state_exact
        && perf_pass
    {
        println!("GATE: PASS exact prefix restore and <25% warmed replay cost");
        Ok(())
    } else {
        Err(format!(
            "GATE: FAIL replay_exact={replay_exact} restored_exact={restored_exact} tokens={tokens_exact} logits={logits_exact} final_state={final_state_exact} perf={perf_pass}"
        ))
    }
}
