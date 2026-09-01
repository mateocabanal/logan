//! Validated high-performance runtime defaults for Qwen4.
//!
//! Policy: use the fastest configuration we have actually measured unless the
//! caller explicitly overrides an environment variable. Rejected/experimental
//! paths are intentionally not enabled just because they are performance
//! related.

fn set_if_unset(name: &str, value: &str) {
    if std::env::var_os(name).is_none() {
        std::env::set_var(name, value);
    }
}

/// Apply validated max-performance defaults without overriding explicit user
/// choices. Callers can opt out of any fast path by setting its existing env
/// variable to `0` (or select an alternate backend explicitly).
pub fn apply_max_performance_defaults() {
    // Validated general fast paths.
    set_if_unset("QWEN_GDN_SINGLE_COPY", "1");
    set_if_unset("QWEN_ATTN_METAL", "1");
    set_if_unset("QWEN_QSA_INDEX_METAL", "1");
    set_if_unset("QWEN_APPLE8_DIRECT", "1");
    set_if_unset("QWEN_APPLE8_OVERLAP", "1");
    set_if_unset("QWEN_SHARED_IO_OVERLAP", "1");

    // Persistent prefix reuse is a validated end-to-end win. Writes remain on
    // so future requests can actually hit the cache; both knobs are opt-out.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_opt_out_is_preserved() {
        let name = "LOGAN_TEST_PERF_DEFAULT";
        std::env::set_var(name, "0");
        set_if_unset(name, "1");
        assert_eq!(std::env::var(name).unwrap(), "0");
        std::env::remove_var(name);
    }
}
