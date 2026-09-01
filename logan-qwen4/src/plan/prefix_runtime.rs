//! Automatic persistent-prefix reuse for normal .coli greedy requests.
//!
//! Policy is deliberately small and reversible:
//! - opt-in until a disk-budget/eviction policy exists;
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

use super::{PrefixCacheKey, PrefixCacheStore};

/// Automatic prefix caching remains opt-in until cache size/eviction is
/// bounded. Explicitly setting a cache directory also counts as opting in.
pub fn auto_prefix_cache_enabled() -> bool {
    std::env::var("QWEN_PREFIX_CACHE")
        .map(|v| v != "0")
        .unwrap_or_else(|_| std::env::var_os("LOGAN_PREFIX_CACHE_DIR").is_some())
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
/// longest first. Filenames are only a cheap candidate-length index; every
/// candidate is recomputed from the exact model + token prefix and the normal
/// restore path still validates its header and full payload checksum.
fn candidate_keys(
    store: &PrefixCacheStore,
    model: &Model,
    prompt: &[u32],
    salt: &[u8],
) -> Result<Vec<PrefixCacheKey>, String> {
    if prompt.is_empty() {
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
            if n <= prompt.len() {
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

/// Normal .coli greedy path with automatic persistent prefix reuse.
///
/// When caching is disabled this is intentionally the old forward loop. When
/// enabled, previous complete request prompts serve as semantic checkpoints:
/// an append-only conversation naturally finds the previous request as its
/// longest reusable prefix.
pub fn run_greedy_cached_coli(
    package_dir: &Path,
    cfg: &Cfg,
    prompt: &[u32],
    max_new: usize,
) -> Result<Vec<u32>, String> {
    let profile = logan_core::telemetry::enabled();
    let total_t0 = Instant::now();
    let mut model = load_model(package_dir, cfg)?;

    if !auto_prefix_cache_enabled() {
        for (i, &t) in prompt.iter().enumerate() {
            model.forward_token(t as usize, i);
        }
        let out = generate(&mut model, prompt, max_new);
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
                eprintln!("[qwen4-rs] prefix-cache miss prompt_tokens={}", prompt.len());
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

    for (i, &t) in prompt.iter().enumerate().skip(prompt_start) {
        model.forward_token(t as usize, i);
    }

    // Persist only complete input-prompt boundaries. This is synchronous in
    // v1; the measured ~0.4 s write cost is tiny relative to current replay,
    // and keeping it here makes correctness simple. Async persistence can be
    // added independently after this integration is validated.
    if let Some(store) = &store {
        if writes_enabled() && !prompt.is_empty() && prompt.len() >= min_persist_tokens() {
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

    let out = generate(&mut model, prompt, max_new);
    if profile {
        model.profile_summary(max_new, total_t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok(out)
}

fn generate(model: &mut Model, prompt: &[u32], max_new: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(max_new);
    let mut last = *prompt.last().unwrap_or(&0);
    let start = prompt.len();
    for pos in start..start + max_new {
        let logits = model.forward_token(last as usize, pos);
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        out.push(next);
        last = next;
    }
    out
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
}
