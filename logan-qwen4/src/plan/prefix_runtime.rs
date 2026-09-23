//! Automatic persistent-prefix reuse for normal .coli greedy requests.
//!
//! Policy is deliberately small and reversible:
//! - validated performance features are on unless explicitly disabled;
//! - restore the longest previously persisted request-prefix;
//! - evaluate only the uncached prompt suffix;
//! - persist the completed input-prompt boundary before generation;
//! - on any restore failure, reload a fresh model before replaying so a
//!   partially applied recurrent state can never leak into the fallback.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::colisource::ColiSource;
use crate::{Cfg, Model};

use super::{CacheWriteStats, PrefixCacheKey, PrefixCacheStore};

fn set_if_unset(name: &str, value: &str) {
    if std::env::var_os(name).is_none() {
        std::env::set_var(name, value);
    }
}

/// Apply the fastest configuration that has already passed real-model A/B
/// validation. Explicit user values always win, so every path remains opt-out.
/// `Model::load_coli` invokes this automatically; it remains public for
/// runners/tests that need to establish policy before model construction.
pub fn apply_max_performance_defaults() {
    set_if_unset("QWEN_GDN_SINGLE_COPY", "1");
    set_if_unset("QWEN_GDN_MXFP4_FULL", "1");
    set_if_unset("QWEN_SHARED_MXFP4_FULL", "1");
    set_if_unset("QWEN_QSA_INDEX_METAL", "1");
    set_if_unset("QWEN_APPLE8_DIRECT", "1");
    set_if_unset("QWEN_APPLE8_OVERLAP", "1");
    set_if_unset("QWEN_SHARED_IO_OVERLAP", "1");
    set_if_unset("QWEN_PREFIX_CACHE", "1");
    set_if_unset("QWEN_PREFIX_CACHE_WRITE", "1");

    // Measured Apple Silicon winners: the legacy BF16 full-GDN path remains
    // slower than BNNS, while MXFP4 GDN has its own default-on full-GPU gate.
    // The generic Metal BF16 attention path pays
    // synchronous weight-buffer staging/waits that are substantially slower
    // than BNNS at decode batch S=1. Explicit env values remain authoritative.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        set_if_unset("QWEN_BNNS_BF16", "1");
        set_if_unset("QWEN_GDN_METAL", "0");
        set_if_unset("QWEN_ATTN_METAL", "0");
    }

    // Non-Apple targets keep the existing generic Metal-attention policy.
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    set_if_unset("QWEN_ATTN_METAL", "1");
}

/// Automatic prefix caching is default-on after real-model validation. Set
/// QWEN_PREFIX_CACHE=0 to opt out. The explicit cache directory still selects
/// where persistent checkpoints live, but is no longer required to enable the
/// feature.
pub fn auto_prefix_cache_enabled() -> bool {
    std::env::var("QWEN_PREFIX_CACHE")
        .map(|v| v != "0")
        .unwrap_or(true)
}

fn writes_enabled() -> bool {
    std::env::var("QWEN_PREFIX_CACHE_WRITE")
        .map(|v| v != "0")
        .unwrap_or(true)
}

fn min_persist_tokens() -> usize {
    std::env::var("QWEN_PREFIX_CACHE_MIN_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
}

fn cache_salt() -> Vec<u8> {
    std::env::var("LOGAN_PREFIX_CACHE_SALT")
        .map(|v| v.into_bytes())
        .unwrap_or_default()
}

fn load_model(package_dir: &Path, cfg: &Cfg) -> Result<Model, String> {
    let src = ColiSource::open(package_dir)?;
    Model::load_coli(&src, cfg)
}

fn parse_entry_prefix_len(name: &str) -> Option<usize> {
    let stem = name.strip_suffix(".lpfx")?;
    let (len, digest) = stem.split_once('-')?;
    if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let n = len.parse::<usize>().ok()?;
    (n > 0).then_some(n)
}

/// Return exact keys for all cache files that could be prefixes of `prompt`,
/// longest first. A state-only checkpoint equal to the complete prompt cannot
/// provide the final prompt logits, so only STRICT prefixes are candidates for
/// generation. Filenames are only a cheap candidate-length index; every key is
/// recomputed from the exact model + token prefix and restore validates the
/// header and full payload checksum.
fn candidate_keys(
    store: &PrefixCacheStore,
    model: &Model,
    prompt: &[u32],
    salt: &[u8],
) -> Result<Vec<PrefixCacheKey>, String> {
    if prompt.len() < 2 {
        return Ok(Vec::new());
    }

    // An empty-prefix key gives us the exact numerical-policy model namespace
    // without exposing the cache's internal model digest separately.
    let namespace = PrefixCacheKey::with_salt(model, &[], salt)?;
    let dir = store.root().join(namespace.model_hex());
    let entries = match fs::read_dir(&dir) {
        Ok(v) => v,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("read prefix cache {}: {e}", dir.display())),
    };

    let mut lengths = BTreeSet::new();
    for entry in entries {
        let entry = match entry {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ty = match entry.file_type() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if !ty.is_file() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if let Some(n) = parse_entry_prefix_len(&name) {
            if n < prompt.len() {
                lengths.insert(n);
            }
        }
    }

    let mut out = Vec::new();
    for n in lengths.into_iter().rev() {
        let key = PrefixCacheKey::with_salt(model, &prompt[..n], salt)?;
        if store.path_for(&key).is_file() {
            out.push(key);
        }
    }
    Ok(out)
}

#[derive(Clone, Debug, Default)]
pub struct PrefixRestoreSummary {
    pub cached_tokens: usize,
    pub restore_ms: f64,
    pub payload_bytes: u64,
}

/// Restore the longest exact persistent prefix into an already-loaded model.
///
/// `.lpfx` stores causal state only, not logits. Candidate lookup therefore
/// intentionally chooses a checkpoint STRICTLY shorter than `prompt`, so at
/// least one suffix token is forwarded after restore and supplies the logits
/// needed to select the first generated token.
///
/// A restore error is returned immediately because `restore` may have begun
/// applying state; callers must reload a pristine model before replaying.
pub fn restore_longest_prefix(
    model: &mut Model,
    prompt: &[u32],
) -> Result<Option<PrefixRestoreSummary>, String> {
    if !auto_prefix_cache_enabled() || prompt.len() < 2 {
        return Ok(None);
    }
    let store = PrefixCacheStore::from_env()?;
    let salt = cache_salt();
    let Some(key) = candidate_keys(&store, model, prompt, &salt)?
        .into_iter()
        .next()
    else {
        return Ok(None);
    };
    let stats = store.restore(model, &key)?;
    Ok(Some(PrefixRestoreSummary {
        cached_tokens: key.prefix_len(),
        restore_ms: stats.total.as_secs_f64() * 1e3,
        payload_bytes: stats.payload_bytes,
    }))
}

/// Persist an exact completed prompt boundary for future process/session reuse.
/// Existing immutable entries are cheap no-ops. Returns `None` when writes are
/// disabled or the boundary is below the configured minimum length.
pub fn persist_prefix_boundary(
    model: &Model,
    prompt: &[u32],
) -> Result<Option<CacheWriteStats>, String> {
    if !auto_prefix_cache_enabled()
        || !writes_enabled()
        || prompt.is_empty()
        || prompt.len() < min_persist_tokens()
    {
        return Ok(None);
    }
    let store = PrefixCacheStore::from_env()?;
    let salt = cache_salt();
    let key = PrefixCacheKey::with_salt(model, prompt, &salt)?;
    store.store(model, &key).map(Some)
}

fn prefill_suffix(model: &mut Model, prompt: &[u32], start: usize) -> Option<Vec<f32>> {
    if start >= prompt.len() {
        return None;
    }
    let chunk = std::env::var("QWEN_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8)
        .clamp(1, 64);
    let suffix = &prompt[start..];
    let mut logits = None;
    for (chunk_idx, rows) in suffix.chunks(chunk).enumerate() {
        let pos = start + chunk_idx * chunk;
        let final_chunk = pos + rows.len() == prompt.len();
        match model.prefill_chunk(rows, pos, final_chunk) {
            Ok(Some(v)) => logits = Some(v),
            Ok(None) => {}
            Err(e) => panic!("layer-major prefill failed: {e}"),
        }
    }
    logits
}

fn prefill_with_mtp(model: &mut Model, prompt: &[u32]) -> Result<Vec<f32>, String> {
    if prompt.is_empty() {
        return Err("Qwen4 MTP requires a non-empty prompt".into());
    }
    let mut previous_hidden: Vec<f32> = Vec::new();
    let mut logits = Vec::new();
    for (pos, &token) in prompt.iter().enumerate() {
        if pos + 1 == prompt.len() {
            logits = model.forward_token(token as usize, pos);
        } else {
            model.prefill_token(token as usize, pos);
        }
        let current_hidden = model.last_hidden_nextn().to_vec();
        if current_hidden.is_empty() {
            return Err(format!(
                "target did not export MTP hidden row at prompt position {pos}"
            ));
        }
        if previous_hidden.is_empty() {
            previous_hidden.resize(current_hidden.len(), 0.0);
        }
        model.mtp_catchup(token as usize, &previous_hidden, pos)?;
        previous_hidden = current_hidden;
    }
    Ok(logits)
}

/// Correctness-first MTP qualification loop. The draft model is genuinely run
/// on every next-token transition and its prediction is checked against the
/// target, but target verification remains serial. Therefore output tokens are
/// exactly the ordinary greedy target tokens even when a draft misses. Once
/// acceptance is qualified, this same state split can be upgraded to batched
/// target verification + rollback for actual speculative speedup.
fn mtp_block_len() -> usize {
    std::env::var("QWEN_MTP_BLOCK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
        .clamp(1, 4)
}

/// Multi-token Qwen4Exp MTP speculative decoding. The one-layer MTP block is
/// recursively applied up to `QWEN_MTP_BLOCK` (default 4) times, carrying its
/// post-block HC residual into the next draft step. The target verifies the
/// whole candidate block with the existing layer-major batched path.
///
/// Correctness policy for this first batched implementation:
/// - full-accept: keep the target state produced by the batched verifier;
/// - rejection: restore the exact pre-block target snapshot and replay only
///   the current token plus the accepted draft prefix. The correcting target
///   token remains unconsumed, matching ordinary greedy decode semantics.
///
/// MTP owns a separate KV row at virtual layer 48. Rejected rows do not need
/// zeroing: causal reads only observe <= current position and resumed drafting
/// overwrites the first rejected row before it becomes visible.
fn generate_from_logits_mtp_block(
    model: &mut Model,
    logits: Vec<f32>,
    prompt_len: usize,
    max_new: usize,
) -> Result<Vec<u32>, String> {
    if max_new == 0 {
        return Ok(Vec::new());
    }

    // Decode boundary: the caller has evaluated the full prompt (and the
    // drafter has caught up), so counter deltas from here are decode-only.
    model.begin_decode_measurement();
    let block_cap = mtp_block_len();
    let mut out = Vec::with_capacity(max_new);
    let mut current = argmax(&logits);
    out.push(current);

    while out.len() < max_new {
        let pos = prompt_len + out.len() - 1;
        let k = block_cap.min(max_new - out.len());
        let mut hidden = model.last_hidden_nextn().to_vec();
        if hidden.is_empty() {
            return Err(format!(
                "target did not export MTP hidden row before speculative block at position {pos}"
            ));
        }

        // Draft k future tokens without touching target causal state.
        let draft_before = model.runtime_stats();
        let draft_t0 = Instant::now();
        let mut drafts = Vec::with_capacity(k);
        let mut draft_token = current;
        for i in 0..k {
            let draft = model.mtp_draft(draft_token as usize, &hidden, pos + i)?;
            if draft.next_hidden_hc.is_empty() {
                return Err(format!("MTP step {i} returned no recursive HC state"));
            }
            draft_token = argmax(&draft.logits);
            drafts.push(draft_token);
            hidden = draft.next_hidden_hc;
        }
        let draft_ms = draft_t0.elapsed().as_secs_f64() * 1e3;
        let draft_delta = model.runtime_stats().delta_from(&draft_before);

        // To validate drafts d1..dk the target consumes [current,d1..d{k-1}]
        // and produces the authoritative distributions for [d1..dk].
        let mut verify_inputs = Vec::with_capacity(k);
        verify_inputs.push(current);
        verify_inputs.extend_from_slice(&drafts[..k.saturating_sub(1)]);

        let verify_before = model.runtime_stats();
        let verify_t0 = Instant::now();
        let verify = model.prefill_chunk_logits_all(&verify_inputs, pos)?;
        if verify.logits.len() != k || verify.boundaries.len() != k {
            return Err(format!(
                "MTP verifier returned {} logits / {} boundaries for {k} inputs",
                verify.logits.len(),
                verify.boundaries.len(),
            ));
        }

        let mut accepted = 0usize;
        while accepted < k && argmax(&verify.logits[accepted]) == drafts[accepted] {
            accepted += 1;
        }

        if accepted == k {
            // Speculative GDN recurrence is evaluated out-of-place so every
            // row remains a commit candidate. Publish only the fully accepted
            // final boundary; dk is emitted but intentionally not consumed
            // until the next block.
            model.mtp_commit_verified_boundary(&verify.boundaries[k - 1])?;
            out.extend_from_slice(&drafts);
            current = *drafts.last().unwrap();
        } else {
            let correction = argmax(&verify.logits[accepted]);

            // Rows after the mismatch were evaluated under a rejected prefix,
            // but rows 0..=accepted are already authoritative target work.
            // Commit the recurrent state captured immediately after the last
            // valid consumed row instead of restoring the whole block and
            // replaying that prefix. KV/QSA future rows remain physically
            // present but are causally invisible and will be overwritten.
            model.mtp_commit_verified_boundary(&verify.boundaries[accepted])?;

            out.extend_from_slice(&drafts[..accepted]);
            out.push(correction);
            current = correction;
        }

        let verify_ms = verify_t0.elapsed().as_secs_f64() * 1e3;
        let verify_delta = model.runtime_stats().delta_from(&verify_before);
        model.mtp_record_block(
            k,
            accepted,
            draft_ms,
            verify_ms,
            draft_delta.mio_bytes,
            verify_delta.mio_bytes,
        );
    }

    out.truncate(max_new);
    Ok(out)
}

fn print_mtp_stats(model: &Model) {
    let Some(stats) = model.mtp_stats() else {
        return;
    };
    let rate = if stats.drafted == 0 {
        0.0
    } else {
        100.0 * stats.accepted as f64 / stats.drafted as f64
    };
    let pos = (0..4)
        .map(|i| {
            let attempts = stats.attempted_by_pos[i];
            let pct = if attempts == 0 {
                0.0
            } else {
                100.0 * stats.accepted_by_pos[i] as f64 / attempts as f64
            };
            format!(
                "{}:{}/{}={pct:.1}%",
                i + 1,
                stats.accepted_by_pos[i],
                attempts
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!(
        "[qwen4-rs] MTP block verify: block_cap={} blocks={} catchup_rows={} drafted={} accepted={} acceptance={rate:.1}% positions=[{pos}] draft_ms={:.1} verify_ms={:.1} draft_io={:.2}MiB verify_io={:.2}MiB",
        mtp_block_len(),
        stats.blocks,
        stats.catchup_rows,
        stats.drafted,
        stats.accepted,
        stats.draft_ms,
        stats.verify_ms,
        stats.draft_mio_bytes as f64 / (1024.0 * 1024.0),
        stats.verify_mio_bytes as f64 / (1024.0 * 1024.0),
    );
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Decode from logits produced by the FINAL prompt token. For token `n+1`,
/// feed generated token `n` at its real position; never replay the final
/// prompt token at a synthetic extra position.
fn generate_from_logits(
    model: &mut Model,
    mut logits: Vec<f32>,
    prompt_len: usize,
    max_new: usize,
) -> Vec<u32> {
    // Decode boundary: the caller has already evaluated the final prompt
    // forward, so this helper is the exact point after which every reported
    // counter should measure decode only.
    model.begin_decode_measurement();
    let mut out = Vec::with_capacity(max_new);
    for step in 0..max_new {
        let next = argmax(&logits);
        out.push(next);
        if step + 1 < max_new {
            logits = model.forward_token(next as usize, prompt_len + step);
        }
    }
    out
}

/// Normal .coli greedy path with automatic persistent prefix reuse.
///
/// Validated performance defaults are applied before model load. Explicit env
/// values are preserved, so every fast path remains opt-out for A/B, fallback,
/// and debugging.
/// Greedy decode with MTP speculative blocks, against an ALREADY-LOADED model.
///
/// Split out of [`run_greedy_cached_coli`] so the safetensors path can use it
/// too: that entry point loads a `.coli` package, while the safetensors loader
/// builds the model itself and only shares the generation loop.
///
/// Requires a drafter to be attached (`model.mtp_enabled()`); callers decide
/// whether to take this path or plain decode.
pub fn run_greedy_mtp(
    model: &mut Model,
    prompt: &[u32],
    max_new: usize,
) -> Result<Vec<u32>, String> {
    if prompt.is_empty() || max_new == 0 {
        return Ok(Vec::new());
    }
    // Prefix snapshots cover target causal state only, so replay the prompt once
    // and let the drafter catch up on exactly shifted (h[p-1], x[p]) rows.
    let logits = prefill_with_mtp(model, prompt)?;
    generate_from_logits_mtp_block(model, logits, prompt.len(), max_new)
}

/// Emit the drafter's acceptance counters, for a caller that wants them.
pub fn print_mtp_stats_for(model: &Model) {
    print_mtp_stats(model)
}

pub fn run_greedy_cached_coli(
    package_dir: &Path,
    cfg: &Cfg,
    prompt: &[u32],
    max_new: usize,
) -> Result<Vec<u32>, String> {
    apply_max_performance_defaults();
    if std::env::var("LOGAN_SUPPRESS_LEGACY_FORMAT_WARNING")
        .map(|v| v == "0" || v.is_empty())
        .unwrap_or(true)
    {
        eprintln!(
            "[logan] legacy COLI package: supported for compatibility; .logan is the canonical compiled format"
        );
    }

    let profile = logan_core::telemetry::enabled();
    let total_t0 = Instant::now();
    let mut model = load_model(package_dir, cfg)?;

    if prompt.is_empty() || max_new == 0 {
        return Ok(Vec::new());
    }

    if model.mtp_enabled() {
        // Prefix snapshots currently cover target causal state only. Until MTP
        // KV is added to LPFX, replay the prompt once so the drafter catches up
        // with exactly shifted (h[p-1], x[p]) rows.
        eprintln!(
            "[qwen4-rs] MTP active: embedded drafter + batched speculative verification (prefix-cache bypassed)"
        );
        let logits = prefill_with_mtp(&mut model, prompt)?;
        let out = generate_from_logits_mtp_block(&mut model, logits, prompt.len(), max_new)?;
        print_mtp_stats(&model);
        if profile {
            model.profile_summary(max_new, total_t0.elapsed().as_secs_f64() * 1e3);
        }
        return Ok(out);
    }

    if !auto_prefix_cache_enabled() {
        let logits = prefill_suffix(&mut model, prompt, 0)
            .ok_or_else(|| "Qwen4 decode requires a non-empty prompt".to_string())?;
        let out = generate_from_logits(&mut model, logits, prompt.len(), max_new);
        if profile {
            model.profile_summary(max_new, total_t0.elapsed().as_secs_f64() * 1e3);
        }
        return Ok(out);
    }

    let salt = cache_salt();
    let store = match PrefixCacheStore::from_env() {
        Ok(v) => Some(v),
        Err(e) => {
            eprintln!("[qwen4-rs] prefix-cache disabled for request: {e}");
            None
        }
    };

    let mut prompt_start = 0usize;
    if let Some(store) = &store {
        match candidate_keys(store, &model, prompt, &salt) {
            Ok(keys) if keys.is_empty() => {
                eprintln!(
                    "[qwen4-rs] prefix-cache miss prompt_tokens={}",
                    prompt.len()
                );
            }
            Ok(keys) => {
                // Try longest first. A damaged/stale entry never poisons the
                // fallback: reload a pristine model before attempting a
                // shorter candidate or full replay.
                for key in keys {
                    match store.restore(&mut model, &key) {
                        Ok(stats) => {
                            prompt_start = key.prefix_len();
                            eprintln!(
                                "[qwen4-rs] prefix-cache hit cached_tokens={} suffix_tokens={} restore={:.2} ms",
                                prompt_start,
                                prompt.len().saturating_sub(prompt_start),
                                stats.total.as_secs_f64() * 1e3,
                            );
                            break;
                        }
                        Err(e) => {
                            eprintln!(
                                "[qwen4-rs] prefix-cache entry rejected tokens={}: {e}; trying fallback",
                                key.prefix_len()
                            );
                            model = load_model(package_dir, cfg)?;
                            prompt_start = 0;
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("[qwen4-rs] prefix-cache lookup failed: {e}; replaying prompt");
            }
        }
    }

    let logits = prefill_suffix(&mut model, prompt, prompt_start)
        .ok_or_else(|| "prefix cache restored full prompt without logits".to_string())?;

    // Persist only complete input-prompt boundaries. This remains synchronous
    // for now; QWEN_PREFIX_CACHE_WRITE=0 opts out independently of cache reads.
    if let Some(store) = &store {
        if writes_enabled() && prompt.len() >= min_persist_tokens() {
            match PrefixCacheKey::with_salt(&model, prompt, &salt)
                .and_then(|key| store.store(&model, &key))
            {
                Ok(stats) if !stats.already_existed => eprintln!(
                    "[qwen4-rs] prefix-cache stored tokens={} file={:.2} MiB write_fsync={:.2} ms",
                    prompt.len(),
                    stats.file_bytes as f64 / (1024.0 * 1024.0),
                    stats.elapsed.as_secs_f64() * 1e3,
                ),
                Ok(_) => {}
                Err(e) => eprintln!("[qwen4-rs] prefix-cache store failed (non-fatal): {e}"),
            }
        }
    }

    let out = generate_from_logits(&mut model, logits, prompt.len(), max_new);
    if profile {
        model.profile_summary(max_new, total_t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_only_final_cache_names() {
        assert_eq!(
            parse_entry_prefix_len(
                "00000005-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.lpfx"
            ),
            Some(5)
        );
        assert_eq!(parse_entry_prefix_len(".foo.tmp.1"), None);
        assert_eq!(parse_entry_prefix_len("00000005-deadbeef.lpfx"), None);
        assert_eq!(
            parse_entry_prefix_len(
                "00000000-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.lpfx"
            ),
            None
        );
    }

    #[test]
    fn explicit_opt_out_is_preserved() {
        let name = "LOGAN_TEST_PERF_DEFAULT";
        std::env::set_var(name, "0");
        set_if_unset(name, "1");
        assert_eq!(std::env::var(name).unwrap(), "0");
        std::env::remove_var(name);
    }

    #[test]
    fn argmax_chooses_largest_logit() {
        assert_eq!(argmax(&[-2.0, 7.0, 3.0]), 1);
    }
}
