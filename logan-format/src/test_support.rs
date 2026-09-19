//! Temporary-path helper shared by the crates whose tests need a unique scratch
//! file named after the running test.
//!
//! ## Why this lives here
//!
//! Three test sites (`logan-qwen4` ×2, `logan-compiler` ×1) built a temp
//! filename out of `std::thread::current().name()`. Under `cargo test` that
//! name is the full test path, e.g.
//! `tests::qwen3_next_config_derives_hybrid_schedule_and_raw_norm_semantics`,
//! which contains `::` — illegal in a Windows filename. The tests therefore
//! failed on Windows only, with `Os { code: 123, kind: InvalidFilename }`.
//!
//! `logan-format` is the only crate that both `logan-qwen4` and
//! `logan-compiler` already depend on, so it is the one place a single helper
//! can serve all of them without adding a workspace member. It is a plain
//! `pub fn` rather than `#[cfg(test)]` because `cfg(test)` only applies while a
//! crate is compiling *its own* tests — an external crate's tests would not see
//! it.
//!
//! No unsafe, no allocation beyond the returned `PathBuf`.

use std::path::PathBuf;

/// Longest sanitized name component retained. Keeps the whole path well inside
/// conservative `MAX_PATH` budgets once the temp directory and extension are
/// added.
const MAX_COMPONENT: usize = 96;

/// A unique, filesystem-legal temporary path for the calling test.
///
/// The name is `<prefix>-<pid>-<sanitized thread name>-<hash>.<extension>`
/// inside [`std::env::temp_dir`]. Every character that is not ASCII
/// alphanumeric, `.`, `_` or `-` becomes `_`, which covers the Windows-illegal
/// set (`< > : " / \ | ? *`), `::` in particular, and control characters.
///
/// `extension` is given without the leading dot; a supplied dot is tolerated.
/// An empty extension is allowed.
///
/// The trailing hash is taken over the *raw* thread name, so two test paths
/// that sanitize to the same component (say `a::b` and `a__b`) still get
/// distinct files.
pub fn test_temp_path(prefix: &str, extension: &str) -> PathBuf {
    // Bind the handle: `Thread::name()` borrows from it, so calling
    // `std::thread::current().name()` directly would drop the temporary at the
    // end of the statement while `raw` still borrowed it.
    let handle = std::thread::current();
    let raw = handle.name().unwrap_or("test");

    let mut name = String::with_capacity(prefix.len() + MAX_COMPONENT + 24);
    name.push_str(prefix);
    name.push('-');
    name.push_str(&std::process::id().to_string());
    name.push('-');
    name.push_str(&sanitize_component(raw));

    let extension = extension.strip_prefix('.').unwrap_or(extension);
    if !extension.is_empty() {
        name.push('.');
        name.push_str(extension);
    }

    std::env::temp_dir().join(name)
}

/// Replaces every character illegal in a Windows filename with `_`, bounds the
/// length, and appends a hash of the original so distinct inputs stay distinct.
fn sanitize_component(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len().min(MAX_COMPONENT) + 9);
    for ch in raw.chars() {
        if out.len() >= MAX_COMPONENT {
            break;
        }
        let legal = ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-');
        out.push(if legal { ch } else { '_' });
    }

    // FNV-1a over the raw bytes. Uniqueness must not depend on the sanitizer
    // being injective, because it is not.
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in raw.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    out.push_str(&format!("-{:08x}", hash as u32));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_a_path_with_no_windows_illegal_characters() {
        // The raw test name here contains `::`, exactly like the real callers.
        let path = test_temp_path("logan-check", "json");
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(
            !name.contains(['<', '>', ':', '"', '/', '\\', '|', '?', '*']),
            "illegal character survived sanitization: {name}"
        );
        assert!(name.starts_with("logan-check-"), "name={name}");
        assert!(name.ends_with(".json"), "name={name}");
        assert_eq!(path.parent(), Some(std::env::temp_dir().as_path()));
    }

    #[test]
    fn empty_extension_omits_the_dot() {
        let name = test_temp_path("logan-check", "")
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert!(!name.ends_with('.'), "name={name}");
    }

    #[test]
    fn sanitization_bounds_length_and_keeps_distinct_inputs_distinct() {
        let long = "x".repeat(500);
        let bounded = sanitize_component(&long);
        assert!(
            bounded.len() <= MAX_COMPONENT + 9,
            "unbounded component: {}",
            bounded.len()
        );

        // These two collapse to the same visible prefix after sanitization
        // (`::` -> `__`), so only the hash separates them. This is the property
        // that keeps distinct tests from sharing one scratch file.
        assert_ne!(
            sanitize_component("tests::a"),
            sanitize_component("tests__a")
        );

        // Same input, same output: the helper is deterministic.
        assert_eq!(
            sanitize_component("tests::a"),
            sanitize_component("tests::a")
        );
    }
}
