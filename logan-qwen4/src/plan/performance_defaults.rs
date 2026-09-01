//! Validated high-performance runtime defaults used by normal Qwen4 `.coli`
//! execution. Explicit environment values always win.

fn set_if_unset(name: &str, value: &str) {
    if std::env::var_os(name).is_none() {
        std::env::set_var(name, value);
    }
}

pub(super) fn apply_max_performance_defaults() {
    // Validated fast paths. These are already no-ops/fallback safely where a
    // backend is unavailable; setting them here makes every normal entry point
    // choose the fastest measured policy unless explicitly disabled.
    set_if_unset("QWEN_GDN_SINGLE_COPY", "1");
    set_if_unset("QWEN_ATTN_METAL", "1");
    set_if_unset("QWEN_QSA_INDEX_METAL", "1");
    set_if_unset("QWEN_APPLE8_DIRECT", "1");
    set_if_unset("QWEN_APPLE8_OVERLAP", "1");
    set_if_unset("QWEN_SHARED_IO_OVERLAP", "1");
    set_if_unset("QWEN_PREFIX_CACHE", "1");
    set_if_unset("QWEN_PREFIX_CACHE_WRITE", "1");

    // Measured M2 winner: Accelerate/BNNS BF16 dense execution with Metal GDN
    // disabled. Both remain explicitly overridable for A/B and fallback.
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
    fn explicit_value_is_never_overwritten() {
        let name = "LOGAN_TEST_PERF_DEFAULT";
        std::env::set_var(name, "0");
        set_if_unset(name, "1");
        assert_eq!(std::env::var(name).unwrap(), "0");
        std::env::remove_var(name);
    }
}
