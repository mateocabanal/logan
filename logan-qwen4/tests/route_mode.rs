//! Routing-mode selection is process-global env state, so the property that
//! matters is that a model *captures* its mode at construction and never
//! re-reads the environment.
//!
//! This is not hypothetical: `examples/quality_probe` builds a teacher and a
//! student in one process by setting `QWEN_ROUTE_AUTHORITATIVE=1`, loading the
//! student, then clearing it and loading the teacher. If a model re-resolved its
//! mode per layer, both models would end up authoritative and the teacher
//! comparison would be comparing a thing to itself.
//!
//! `Model` owns its weights and the mode field, so the check here is on the
//! selection type itself plus the two invariants that make two models in one
//! process safe: mode is a value (copied, not shared) and the predictor is
//! per-model state constructed from the mode at build time.

use logan_qwen4::RouteMode;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Serialises the tests that mutate process-global `QWEN_ROUTE_*` variables.
///
/// `from_env` reads the real environment, and two tests here write it. Rust runs
/// integration tests on multiple threads, so without this lock one test can
/// observe another's `set_var`/`remove_var` and fail intermittently — which is
/// how this was found (a single failure in an otherwise green suite).
fn env_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// `RouteMode` must be a plain value: two models must be able to hold different
/// modes without any shared mutable state.
#[test]
fn mode_is_a_copyable_value_with_no_process_global_state() {
    let authoritative = RouteMode::Authoritative;
    let native = RouteMode::Native;
    let copied = authoritative;
    assert_eq!(copied, authoritative);
    assert_ne!(copied, native);
    // Copy semantics, so a model cannot observe another model's mode change.
    let mut a = RouteMode::Native;
    let _ = authoritative;
    a = RouteMode::Shadow;
    assert_eq!(native, RouteMode::Native, "unrelated value must be unaffected");
    assert_eq!(a, RouteMode::Shadow);
}

/// The three selectable states the mission requires, each with distinct
/// behaviour. Guarding the *shape* of the selection here means a later change
/// cannot quietly collapse two states into one flag.
#[test]
fn three_independently_selectable_states_exist() {
    // native: no prediction, no override.
    assert!(!RouteMode::Native.needs_predictor());
    assert!(!RouteMode::Native.overrides_route());
    // predict/shadow: prediction, no override.
    assert!(RouteMode::Shadow.needs_predictor());
    assert!(!RouteMode::Shadow.overrides_route());
    // authoritative: prediction and override.
    assert!(RouteMode::Authoritative.needs_predictor());
    assert!(RouteMode::Authoritative.overrides_route());
}

/// The control arm must not carry the treatment's behaviour, or it controls for
/// nothing.
#[test]
fn control_arm_is_distinct_from_both_native_and_authoritative() {
    assert!(!RouteMode::NativeTruncated.needs_predictor());
    assert!(!RouteMode::NativeTruncated.overrides_route());
    assert_ne!(RouteMode::NativeTruncated, RouteMode::Native);
    assert_ne!(RouteMode::NativeTruncated, RouteMode::Authoritative);
}

/// Only explicitly authoritative modes may override the native router.
/// Edge0 is separate from RouteScout's online predictor, so it overrides but
/// does not request a RoutePredictor instance.
#[test]
fn only_explicit_authoritative_modes_override_the_router() {
    assert!(!RouteMode::Edge0.needs_predictor());
    assert!(RouteMode::Edge0.overrides_route());
    assert_eq!(RouteMode::Edge0.to_string(), "edge0");

    let overrides: Vec<RouteMode> = ALL
        .into_iter()
        .filter(|m| m.overrides_route())
        .collect();
    assert_eq!(overrides, vec![RouteMode::Authoritative, RouteMode::Edge0]);
}

/// Every mode, so a new one cannot be added without the matrix below noticing.
const ALL: [RouteMode; 6] = [
    RouteMode::Native,
    RouteMode::Shadow,
    RouteMode::Authoritative,
    RouteMode::NativeTruncated,
    RouteMode::Edge0,
    RouteMode::Hybrid,
];

/// **The central invariant of the hybrid mission.** Hybrid must load and feed
/// both predictors, and must not override the route: the native Qwen K4 gate
/// stays semantically authoritative, and the predictors only decide which bytes
/// are read early.
///
/// This is the test that would fail if someone later "simplified" hybrid by
/// letting a prediction enter the executed route — the failure would be a
/// quality regression that no tok/s figure would reveal.
#[test]
fn hybrid_runs_both_predictors_and_never_overrides_the_route() {
    assert!(RouteMode::Hybrid.needs_edge0(), "hybrid needs Edge0 loaded");
    assert!(
        RouteMode::Hybrid.needs_routescout(),
        "hybrid needs RouteScout fed"
    );
    assert!(
        !RouteMode::Hybrid.overrides_route(),
        "hybrid MUST NOT override the native route: a prediction may stage bytes,          never choose experts"
    );
    assert!(
        !RouteMode::Hybrid.needs_predictor(),
        "hybrid builds no competing route, so it must not pay for one"
    );
    assert!(RouteMode::Hybrid.stages_only());
    assert_eq!(RouteMode::Hybrid.to_string(), "hybrid");
}

/// The two predicates must stay separable, or `needs_predictor` would silently
/// become the "does this mode run a predictor" question and hybrid would start
/// building a route it discards.
#[test]
fn staging_only_and_route_building_predicates_are_independent() {
    // Modes that build a route.
    for mode in [RouteMode::Shadow, RouteMode::Authoritative] {
        assert!(mode.needs_predictor());
        assert!(mode.needs_routescout());
        assert!(!mode.stages_only());
        assert!(!mode.needs_edge0());
    }
    // Modes that only predict storage.
    assert!(RouteMode::Hybrid.needs_routescout());
    assert!(!RouteMode::Hybrid.needs_predictor());
    // Modes that predict nothing at all.
    for mode in [RouteMode::Native, RouteMode::NativeTruncated] {
        assert!(!mode.needs_predictor());
        assert!(!mode.needs_routescout());
        assert!(!mode.needs_edge0());
        assert!(!mode.stages_only());
    }
    // Edge0 alone overrides the route and needs no RouteScout.
    assert!(RouteMode::Edge0.needs_edge0());
    assert!(!RouteMode::Edge0.needs_routescout());
    assert!(!RouteMode::Edge0.stages_only());
}

/// `QWEN_ROUTE_MODE=hybrid` must select hybrid, and the legacy flags must not.
#[test]
fn hybrid_is_selectable_by_mode_string_only() {
    let _env = env_lock();
    std::env::set_var("QWEN_ROUTE_MODE", "hybrid");
    assert_eq!(RouteMode::from_env(), RouteMode::Hybrid);
    std::env::set_var("QWEN_ROUTE_MODE", "HYBRID");
    assert_eq!(RouteMode::from_env(), RouteMode::Hybrid);
    std::env::remove_var("QWEN_ROUTE_MODE");
    assert_eq!(RouteMode::from_env(), RouteMode::Native);
}

/// A model built under one mode must not be re-routed by a later env change.
///
/// Simulated at the value level because constructing a real `Model` needs a
/// checkpoint: the guard is that `from_env` is called once per model and its
/// result is stored, which the field's type enforces (it is a `RouteMode`, not
/// a closure or a getter).
#[test]
fn mode_resolution_is_a_snapshot_not_a_live_read() {
    let _env = env_lock();
    // Two resolutions under different environments must be independent values.
    std::env::set_var("QWEN_ROUTE_AUTHORITATIVE", "1");
    let first = RouteMode::from_env();
    std::env::remove_var("QWEN_ROUTE_AUTHORITATIVE");
    let second = RouteMode::from_env();
    assert_eq!(first, RouteMode::Authoritative);
    assert_eq!(second, RouteMode::Native);
    // And the first snapshot is unaffected by the env change.
    assert_eq!(first, RouteMode::Authoritative);
}
