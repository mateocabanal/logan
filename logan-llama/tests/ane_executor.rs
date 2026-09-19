use logan_llama::ane::{
    AneExecutor, AneOperationSpec, AnePrecision, AnePrivateAbiProbe, AneProgramIdentity,
    AneShapeLayout, CompletionDisposition, CompletionStatus, ScratchDisposition, SubmitError,
};

fn identity() -> AneProgramIdentity {
    AneProgramIdentity {
        model_digest: [1; 32],
        weight_digest: [2; 32],
        precision: AnePrecision::Fp16,
        graph_version: 3,
        shape: AneShapeLayout {
            logical_width: 7,
            padded_width: 16,
            intermediate_width: 32,
            spatial: 16,
            layout_version: 1,
        },
        chip: "m2".into(),
        os: "macos-26".into(),
        private_abi: AnePrivateAbiProbe {
            compiler_available: true,
            runtime_available: true,
            async_channel_available: true,
            shared_event_available: false,
            probe_version: 1,
        },
    }
}

fn operation(id: u64, now_ms: u64, timeout_ms: u64, bytes: usize, fresh: bool) -> AneOperationSpec {
    AneOperationSpec {
        operation_id: id,
        program: identity(),
        bytes,
        scratch_bytes: 16,
        submitted_at_ms: now_ms,
        timeout_ms,
        allow_fresh_scratch: fresh,
    }
}

#[test]
fn timeout_quarantines_bytes_and_late_completion_is_safe() {
    let mut executor = AneExecutor::new(1, 256, 50);
    let (ticket, _) = executor.submit(operation(1, 0, 10, 64, true)).unwrap();
    assert_eq!(executor.expire(10)[0].ticket, ticket);
    assert_eq!(executor.quarantined_slots(), 1);
    assert_eq!(executor.used_bytes(), 80);
    assert_eq!(
        executor.submit(operation(2, 10, 10, 64, true)),
        Err(SubmitError::NoFreeSlot)
    );

    assert_eq!(
        executor.complete(ticket, CompletionStatus::Succeeded),
        CompletionDisposition::LateIgnored
    );
    assert_eq!(executor.used_bytes(), 0);
    let (replacement, _) = executor.submit(operation(2, 11, 10, 64, true)).unwrap();
    assert_ne!(replacement.generation, ticket.generation);
    assert_eq!(
        executor.complete(ticket, CompletionStatus::Succeeded),
        CompletionDisposition::LateIgnored
    );
    assert_eq!(
        executor.complete(replacement, CompletionStatus::Succeeded),
        CompletionDisposition::Completed(CompletionStatus::Succeeded)
    );
}

#[test]
fn duplicate_completion_is_ignored_and_completed_scratch_is_reused() {
    let mut executor = AneExecutor::new(1, 256, 10);
    let (ticket, first_scratch) = executor.submit(operation(7, 0, 20, 32, true)).unwrap();
    assert_eq!(first_scratch, ScratchDisposition::Fresh);
    assert_eq!(
        executor.complete(ticket, CompletionStatus::Succeeded),
        CompletionDisposition::Completed(CompletionStatus::Succeeded)
    );
    assert_eq!(
        executor.complete(ticket, CompletionStatus::Succeeded),
        CompletionDisposition::DuplicateIgnored
    );
    let (_, reused_scratch) = executor.submit(operation(8, 30, 20, 32, false)).unwrap();
    assert_eq!(reused_scratch, ScratchDisposition::Shared);
}

#[test]
fn bounded_bytes_include_scratch_and_require_fresh_fallback() {
    let mut executor = AneExecutor::new(2, 100, 10);
    assert_eq!(
        executor.submit(operation(1, 0, 20, 80, false)),
        Err(SubmitError::FreshScratchRequired)
    );
    let (first, _) = executor.submit(operation(1, 0, 20, 80, true)).unwrap();
    assert_eq!(
        executor.submit(operation(2, 0, 20, 5, true)),
        Err(SubmitError::CapacityExceeded)
    );
    assert_eq!(executor.used_bytes(), 96);
    assert_eq!(
        executor.complete(first, CompletionStatus::Failed),
        CompletionDisposition::Completed(CompletionStatus::Failed)
    );
    assert_eq!(executor.used_bytes(), 0);
}

#[test]
fn program_identity_key_changes_with_probe_and_layout() {
    let first = identity();
    let mut second = first.clone();
    second.private_abi.shared_event_available = true;
    assert_ne!(first.cache_key(), second.cache_key());
    second = first.clone();
    second.shape.layout_version += 1;
    assert_ne!(first.cache_key(), second.cache_key());
}
