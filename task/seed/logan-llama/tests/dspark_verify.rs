use logan_llama::dspark::{
    DSPARK_BLOCK_WIDTH, DSPARK_HIDDEN_SIZE, DSPARK_MARKOV_RANK, DSPARK_MASK_TOKEN_ID, DSPARK_TAPS,
    DecodeControl, DraftAttentionState, DsparkGeometry, TokenizerGeometry, VerificationOptions,
    validate_target_draft, verify_greedy, verify_greedy_logits, verify_greedy_with_options,
};

fn options() -> VerificationOptions {
    VerificationOptions {
        eos_token_ids: vec![99],
        output_cap: None,
        control: DecodeControl::Continue,
    }
}

#[test]
fn contract_geometry_and_tokenizer_identity_are_strict() {
    let geometry = DsparkGeometry::minicpm5(100_000);
    assert_eq!(geometry.block_width, DSPARK_BLOCK_WIDTH);
    assert_eq!(geometry.hidden_size, DSPARK_HIDDEN_SIZE);
    assert_eq!(geometry.markov_rank, DSPARK_MARKOV_RANK);
    assert_eq!(geometry.mask_token_id, DSPARK_MASK_TOKEN_ID);
    assert_eq!(geometry.taps, DSPARK_TAPS);
    let target = TokenizerGeometry::new(100_000, DSPARK_HIDDEN_SIZE, vec![2, 73440]);
    let draft = TokenizerGeometry::new(100_000, DSPARK_HIDDEN_SIZE, vec![2, 73440]);
    validate_target_draft(&target, &draft, &geometry).unwrap();
    let wrong = TokenizerGeometry::new(100_001, DSPARK_HIDDEN_SIZE, vec![2, 73440]);
    assert!(validate_target_draft(&target, &wrong, &geometry).is_err());
}

#[test]
fn first_rejection_retains_anchor_and_emits_target_prediction_once() {
    let result = verify_greedy_with_options(&[8, 4, 5, 6], &[1, 2, 3], &options());
    assert_eq!(result.accepted_drafts, 0);
    assert_eq!(result.retained_rows, 1);
    assert_eq!(result.next_anchor, Some(8));
    assert_eq!(result.emitted, vec![8]);
    assert_eq!(result.rejected_drafts, vec![1, 2, 3]);
}

#[test]
fn every_mismatch_position_retains_only_anchor_and_accepted_inputs() {
    for mismatch in 0..3 {
        let drafts = vec![10, 11, 12];
        let mut target = vec![10, 11, 12, 13];
        target[mismatch] = 80;
        let result = verify_greedy(&target, &drafts);
        assert_eq!(result.accepted_drafts, mismatch);
        assert_eq!(result.retained_rows, mismatch + 1);
        assert_eq!(result.next_anchor, Some(80));
        assert_eq!(result.rejected_drafts, drafts[mismatch..]);
    }
}

#[test]
fn all_accepted_uses_post_block_prediction_and_full_retention() {
    let result = verify_greedy(&[10, 11, 12, 13], &[10, 11, 12]);
    assert_eq!(result.accepted_drafts, 3);
    assert_eq!(result.retained_rows, 4);
    assert_eq!(result.next_anchor, Some(13));
    assert_eq!(result.emitted, vec![10, 11, 12]);
    assert!(result.rejected_drafts.is_empty());
}

#[test]
fn eos_at_each_accepted_position_stops_inside_the_block() {
    for eos_position in 0..3 {
        let drafts = vec![10, 11, 12];
        let mut target = vec![10, 11, 12, 13];
        target[eos_position] = 99;
        let mut draft_with_eos = drafts.clone();
        draft_with_eos[eos_position] = 99;
        let result = verify_greedy_with_options(&target, &draft_with_eos, &options());
        assert_eq!(result.emitted.len(), eos_position + 1);
        assert_eq!(result.emitted.last(), Some(&99));
        assert!(result.stopped_on_eos);
        assert_eq!(result.retained_rows, eos_position + 2);
        assert!(result.next_anchor.is_none());
    }
}

#[test]
fn output_cap_and_cancellation_do_not_commit_rejected_rows() {
    let mut capped = options();
    capped.output_cap = Some(2);
    let result = verify_greedy_with_options(&[1, 2, 3, 4], &[1, 2, 3], &capped);
    assert_eq!(result.emitted, vec![1, 2]);
    assert_eq!(result.retained_rows, 3);
    assert!(result.output_cap_reached);
    let mut cancelled = options();
    cancelled.control = DecodeControl::Cancel;
    let result = verify_greedy_with_options(&[], &[1, 2, 3], &cancelled);
    assert!(result.cancelled);
    assert_eq!(result.retained_rows, 0);
    assert!(result.emitted.is_empty());
}

#[test]
fn logits_argmax_is_slot_zero_based_and_alignment_is_explicit() {
    let logits = vec![
        vec![0.1, 2.0, 1.0],
        vec![4.0, 3.0, 1.0],
        vec![9.0, 0.0, 0.0],
    ];
    let result = verify_greedy_logits(&logits, &[1, 0], &options());
    assert!(result.alignment.valid);
    assert_eq!(result.target_predictions, vec![1, 0, 0]);
    assert_eq!(result.accepted_drafts, 2);
    assert_eq!(result.retained_rows, 3);
}

#[test]
fn draft_attention_always_materializes_all_seven_slots_and_rolls_back_proposals() {
    let mut state = DraftAttentionState::new(7, DSPARK_MASK_TOKEN_ID).unwrap();
    state.commit_input_rows(&[4, 5]);
    let context = state.begin(&[6, 7]).unwrap();
    assert_eq!(context.len(), 7);
    assert_eq!(&context[0..4], &[75982, 75982, 75982, 4]);
    assert_eq!(&context[4..], &[5, 6, 7]);
    state.rollback();
    assert!(state.provisional().is_empty());
    assert_eq!(state.committed(), &[4, 5]);
}
