use std::str::FromStr;

use logan_llama::placement::*;

fn pair() -> ModelPairIdentity {
    ModelPairIdentity::new("target", "draft", "bf16", "M2", "macos-27")
}

fn key(context: ContextBucket, role: PlacementRole) -> CalibrationKey {
    CalibrationKey::new(pair(), ModelPhase::Decode, context, role)
}

fn round(metal_wall_ms: f64, ane_wall_ms: f64) -> RoundEvidence {
    RoundEvidence {
        wall_ms: ane_wall_ms,
        draft_ms: 1.0,
        verification_ms: 2.0,
        handoff_ms: 0.5,
        fallback_ms: 0.0,
        metal_wall_ms,
        ane_wall_ms,
        complete_round: true,
    }
}

#[test]
fn default_auto_is_metal_before_calibration() {
    let controller = PlacementController::default();
    let key = key(ContextBucket::Short, PlacementRole::Draft);
    let decision = controller.decide(PlacementRequest::new(
        &key,
        PlacementMode::Auto,
        PlacementRole::Draft,
    ));
    assert_eq!(decision.actual_backend(), "metal");
    assert!(!decision.ane_active);
    assert!(decision.fallback_reason.is_some());
}

#[test]
fn modes_parse_and_unknown_values_are_rejected() {
    for (text, expected) in [
        ("off", PlacementMode::Off),
        ("auto", PlacementMode::Auto),
        ("draft", PlacementMode::Draft),
        ("ffn", PlacementMode::Ffn),
        ("probe", PlacementMode::Probe),
    ] {
        assert_eq!(PlacementMode::from_str(text).unwrap(), expected);
    }
    assert!(PlacementMode::from_str("ane").is_err());
}

#[test]
fn explicit_modes_keep_target_ffn_and_draft_separate() {
    let controller = PlacementController::default();
    let target = key(ContextBucket::Short, PlacementRole::TargetFfn);
    let draft = key(ContextBucket::Short, PlacementRole::Draft);
    let target_from_draft = controller.decide(PlacementRequest::new(
        &target,
        PlacementMode::Draft,
        PlacementRole::TargetFfn,
    ));
    let draft_from_ffn = controller.decide(PlacementRequest::new(
        &draft,
        PlacementMode::Ffn,
        PlacementRole::Draft,
    ));
    assert_eq!(target_from_draft.actual_backend(), "metal");
    assert_eq!(draft_from_ffn.actual_backend(), "metal");
    assert!(target_from_draft.fallback_reason.is_some());
    assert!(draft_from_ffn.fallback_reason.is_some());
}

#[test]
fn sustained_gain_promotes_and_hysteresis_resists_noise() {
    let mut controller = PlacementController::default();
    let key = key(ContextBucket::Short, PlacementRole::Draft);
    for _ in 0..2 {
        controller.record_round(key.clone(), round(100.0, 90.0));
        assert_eq!(
            controller
                .decide(PlacementRequest::new(
                    &key,
                    PlacementMode::Auto,
                    PlacementRole::Draft
                ))
                .actual_backend(),
            "metal"
        );
    }
    controller.record_round(key.clone(), round(100.0, 90.0));
    assert_eq!(
        controller
            .decide(PlacementRequest::new(
                &key,
                PlacementMode::Auto,
                PlacementRole::Draft
            ))
            .actual_backend(),
        "ane"
    );
    controller.record_round(key.clone(), round(100.0, 100.0));
    assert_eq!(
        controller
            .decide(PlacementRequest::new(
                &key,
                PlacementMode::Auto,
                PlacementRole::Draft
            ))
            .actual_backend(),
        "ane"
    );
    controller.record_round(key.clone(), round(100.0, 110.0));
    assert_eq!(
        controller
            .decide(PlacementRequest::new(
                &key,
                PlacementMode::Auto,
                PlacementRole::Draft
            ))
            .actual_backend(),
        "ane"
    );
    controller.record_round(key.clone(), round(100.0, 110.0));
    assert_eq!(
        controller
            .decide(PlacementRequest::new(
                &key,
                PlacementMode::Auto,
                PlacementRole::Draft
            ))
            .actual_backend(),
        "metal"
    );
}

#[test]
fn phase_and_context_buckets_do_not_share_calibration() {
    let mut controller = PlacementController::default();
    let short_decode = key(ContextBucket::Short, PlacementRole::Draft);
    let medium_decode = key(ContextBucket::Medium, PlacementRole::Draft);
    let short_prefill = CalibrationKey::new(
        pair(),
        ModelPhase::Prefill,
        ContextBucket::Short,
        PlacementRole::Draft,
    );
    for _ in 0..3 {
        controller.record_round(short_decode.clone(), round(100.0, 90.0));
    }
    assert_eq!(
        controller
            .decide(PlacementRequest::new(
                &short_decode,
                PlacementMode::Auto,
                PlacementRole::Draft
            ))
            .actual_backend(),
        "ane"
    );
    assert_eq!(
        controller
            .decide(PlacementRequest::new(
                &medium_decode,
                PlacementMode::Auto,
                PlacementRole::Draft
            ))
            .actual_backend(),
        "metal"
    );
    assert_eq!(
        controller
            .decide(PlacementRequest::new(
                &short_prefill,
                PlacementMode::Auto,
                PlacementRole::Draft
            ))
            .actual_backend(),
        "metal"
    );
}

#[test]
fn unsupported_shape_or_abi_never_reports_ane_active() {
    let controller = PlacementController::default();
    let key = key(ContextBucket::Short, PlacementRole::Draft);
    let shape = PlacementRequest::new(&key, PlacementMode::Probe, PlacementRole::Draft)
        .qualified(true, false, true);
    let abi = PlacementRequest::new(&key, PlacementMode::Probe, PlacementRole::Draft)
        .qualified(true, true, false);
    for decision in [controller.decide(shape), controller.decide(abi)] {
        assert_eq!(decision.actual_backend(), "metal");
        assert!(!decision.ane_active);
        assert!(decision.fallback_reason.is_some());
    }
}

#[test]
fn calibration_key_isolates_model_pair_precision_and_host() {
    let mut controller = PlacementController::default();
    let trained = key(ContextBucket::Short, PlacementRole::Draft);
    for _ in 0..3 {
        controller.record_round(trained.clone(), round(100.0, 90.0));
    }
    let mut other_pair = pair();
    other_pair.precision = "fp16".into();
    other_pair.draft = "other-draft".into();
    other_pair.chip = "M3".into();
    let isolated = CalibrationKey::new(
        other_pair,
        ModelPhase::Decode,
        ContextBucket::Short,
        PlacementRole::Draft,
    );
    assert_eq!(
        controller
            .decide(PlacementRequest::new(
                &isolated,
                PlacementMode::Auto,
                PlacementRole::Draft
            ))
            .actual_backend(),
        "metal"
    );
}
