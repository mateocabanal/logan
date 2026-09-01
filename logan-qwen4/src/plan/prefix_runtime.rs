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
/// Interactive/library clients should call this before `Model::load_coli`.
pub fn apply_max_performance_defaults() {
    set_if_unset("QWEN_GDN_SINGLE_COPY", "1");
    set_if_unset("QWEN_ATTN_METAL", "1");
    set_if_unset("QWEN_QSA_INDEX_METAL", "1");
    set_if_unset("QWEN_APPLE8_DIRECT", "1");
    set_if_unset("QWEN_APPLE8_OVERLAP", "1");
    set_if_unset("QWEN_SHARED_IO_OVERLAP", "1");
    set_if_unset("QWEN_PREFIX_CACHE", "1");
    set_if_unset("QWEN_PREFIX_CACHE_WRITE", "1");

    // Measured Apple Silicon winner: BNNS BF16 dense execution beats the
    // current Metal GDN path while preserving exact greedy output.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        set_if_unset("QWEN_BNNS_BF16", "1");
        set_if_unset("QWEN_GDN_METAL", "0");
    }
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
    let mut logits = None;
    for (i, &token) in prompt.iter().enumerate().skip(start) {
        logits = Some(model.forward_token(token as usize, i));
    }
    logits
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
pub fn run_greedy_cached_coli(
    package_dir: &Path,
    cfg: &Cfg,
    prompt: &[u32],
    max_new: usize,
) -> Result<Vec<u32>, String> {
    apply_max_performance_defaults();

    let profile = logan_core::telemetry::enabled();
    let total_t0 = Instant::now();
    let mut model = load_model(package_dir, cfg)?;

    if prompt.is_empty() || max_new == 0 {
        return Ok(Vec::new());
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
