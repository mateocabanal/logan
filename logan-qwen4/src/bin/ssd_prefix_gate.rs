use std::fs;
use std::path::Path;
use std::time::Instant;

use sha2::{Digest, Sha256};

use logan_qwen4::colisource::ColiSource;
use logan_qwen4::plan::{
    digest_hex, live_prefix_state_digest, PrefixCacheKey, PrefixCacheStore,
};
use logan_qwen4::{load_cfg, Cfg, Model};

fn parse_tokens(name: &str, default: &str) -> Vec<u32> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split_whitespace()
        .map(|t| t.parse::<u32>().expect("token id"))
        .collect()
}

fn load_model(path: &Path) -> Result<(Cfg, ColiSource, Model), String> {
    let cfg = load_cfg(&path.join("config.json"))?;
    let src = ColiSource::open(path)?;
    let model = Model::load_coli(&src, &cfg)?;
    Ok((cfg, src, model))
}

fn prefill(model: &mut Model, tokens: &[u32], start_pos: usize) {
    for (i, &token) in tokens.iter().enumerate() {
        model.forward_token(token as usize, start_pos + i);
    }
}

fn logits_digest(logits: &[f32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update((logits.len() as u64).to_le_bytes());
    for &v in logits {
        h.update(v.to_bits().to_le_bytes());
    }
    h.finalize().into()
}

fn generate_trace(
    model: &mut Model,
    mut last: u32,
    start_pos: usize,
    count: usize,
) -> (Vec<u32>, Vec<[u8; 32]>) {
    let mut tokens = Vec::with_capacity(count);
    let mut logits = Vec::with_capacity(count);
    for pos in start_pos..start_pos + count {
        let out = model.forward_token(last as usize, pos);
        let next = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        logits.push(logits_digest(&out));
        tokens.push(next);
        last = next;
    }
    (tokens, logits)
}

fn main() -> Result<(), String> {
    let package = std::env::args()
        .nth(1)
        .ok_or_else(|| "usage: ssd_prefix_gate <.coli-package>".to_string())?;
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
    if new_tokens == 0 {
        return Err("QWEN_GATE_NEW must be > 0".into());
    }
    let reuse_existing = std::env::var("QWEN_GATE_REUSE")
        .map(|v| v != "0")
        .unwrap_or(false);
    let dirty: Vec<u32> = prefix.iter().map(|&t| t.saturating_add(1)).collect();
    let store = PrefixCacheStore::from_env()?;

    // A: canonical execution and durable cache creation. With
    // QWEN_GATE_REUSE=1 the old .lpfx is intentionally retained, proving a
    // cache made by an earlier process remains valid.
    let (cfg_a, src_a, mut a) = load_model(package)?;
    let key = PrefixCacheKey::new(&src_a, &cfg_a, &prefix);
    let cache_path = store.path_for(&key);
    if cache_path.exists() && !reuse_existing {
        fs::remove_file(&cache_path)
            .map_err(|e| format!("remove old gate cache {}: {e}", cache_path.display()))?;
    }

    let cold_t0 = Instant::now();
    prefill(&mut a, &prefix, 0);
    let cold_prefix_ms = cold_t0.elapsed().as_secs_f64() * 1e3;
    let prefix_digest = live_prefix_state_digest(&a, prefix.len())?;

    let write = store.store(&a, &key)?;
    let write_ms = write.elapsed.as_secs_f64() * 1e3;

    prefill(&mut a, &suffix, prefix.len());
    let prompt_len = prefix.len() + suffix.len();
    let (a_tokens, a_logits) = generate_trace(
        &mut a,
        *suffix.last().unwrap(),
        prompt_len,
        new_tokens,
    );
    let a_final = live_prefix_state_digest(&a, prompt_len + new_tokens)?;
    drop(a);

    // B: conservative replay baseline after warming lazy allocations/backends.
    let (_, _, mut b) = load_model(package)?;
    let empty = b.snapshot_state(0)?;
    prefill(&mut b, &dirty, 0);
    b.restore_state(&empty)?;
    let replay_t0 = Instant::now();
    prefill(&mut b, &prefix, 0);
    let replay_ms = replay_t0.elapsed().as_secs_f64() * 1e3;
    let replay_digest = live_prefix_state_digest(&b, prefix.len())?;
    let replay_exact = replay_digest == prefix_digest;
    drop(b);

    // C: a fresh model instance restores only from the persistent file.
    let (_, _, mut c) = load_model(package)?;
    let restore = store.restore(&mut c, &key)?;
    let restored_digest = live_prefix_state_digest(&c, prefix.len())?;
    let restored_exact = restored_digest == prefix_digest;

    prefill(&mut c, &suffix, prefix.len());
    let (c_tokens, c_logits) = generate_trace(
        &mut c,
        *suffix.last().unwrap(),
        prompt_len,
        new_tokens,
    );
    let c_final = live_prefix_state_digest(&c, prompt_len + new_tokens)?;

    let tokens_exact = a_tokens == c_tokens;
    let logits_exact = a_logits == c_logits;
    let final_state_exact = a_final == c_final;
    let restore_ms = restore.total.as_secs_f64() * 1e3;
    let ratio = if replay_ms > 0.0 {
        restore_ms / replay_ms
    } else {
        f64::INFINITY
    };
    let perf_pass = ratio < 0.25;

    println!("prefix_tokens={} reuse_existing={reuse_existing}", prefix.len());
    println!("cache_dir={}", store.root().display());
    println!("cache_file={}", write.path.display());
    println!(
        "cache_payload={:.2} MiB file={:.2} MiB write_fsync={write_ms:.2} ms already_existed={}",
        write.payload_bytes as f64 / (1024.0 * 1024.0),
        write.file_bytes as f64 / (1024.0 * 1024.0),
        write.already_existed,
    );
    println!(
        "cold_prefix={cold_prefix_ms:.2} ms warmed_replay={replay_ms:.2} ms"
    );
    println!(
        "ssd_verify={:.2} ms ssd_apply={:.2} ms ssd_total={restore_ms:.2} ms nocache={} restore/replay={ratio:.4}x",
        restore.verify.as_secs_f64() * 1e3,
        restore.apply.as_secs_f64() * 1e3,
        restore.nocache,
    );
    println!(
        "exact: replay_state={} restored_state={} tokens={} logits={} final_state={}",
        replay_exact, restored_exact, tokens_exact, logits_exact, final_state_exact
    );
    println!("prefix_state_sha256={}", digest_hex(&prefix_digest));
    println!("generated={c_tokens:?}");

    if replay_exact
        && restored_exact
        && tokens_exact
        && logits_exact
        && final_state_exact
        && perf_pass
    {
        println!("GATE: PASS persistent SSD restore is exact and <25% warmed replay cost");
        Ok(())
    } else {
        Err(format!(
            "GATE: FAIL replay_exact={replay_exact} restored_exact={restored_exact} tokens={tokens_exact} logits={logits_exact} final_state={final_state_exact} perf={perf_pass}"
        ))
    }
}
