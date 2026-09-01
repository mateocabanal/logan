//! Asynchronous accelerator submission/completion contract for issue #52.
//!
//! The scheduler issues an opaque accelerator action to a compiler-approved
//! `ExecutionTarget` and receives a typed ticket; submission is nonblocking
//! and the scheduler thread never waits on device/framework completion. This
//! module is pure contract and bookkeeping only: no platform backends, no
//! native APIs, no queues (worker-count and queue-shape belong to executor
//! #51).
//!
//! Ownership rules:
//! * ticket allocation and exact-once completion semantics are the V0 core's
//!   (`SchedulerCore::submit`/`complete`) — reuse, never reimplemented;
//! * placement is compiler-approved through the #56 `DeviceRegistry`: the
//!   submitted target must be the registered target for its device and must
//!   accept accelerator work;
//! * every resident resource generation referenced by an action is pinned
//!   through a #47 residency lease at submission and released exactly once
//!   by completion processing (or by `cancel_ticket` when the core revokes
//!   the ticket). While pinned, the generation cannot be evicted or reused;
//! * stale and duplicate completions are harmless no-ops.
//!
//! Native command buffers/streams/events and native resource handles are
//! owned by the backend; the contract only carries an opaque `NativeHandle`
//! slot for the backend to fill. Real ownership lands with #57/#58.

use std::collections::HashMap;

use super::core::{Action, ActionKind, Effect, Outcome, SchedError, SchedulerCore};
use super::device::{DeviceRegistry, ExecutionTarget};
use super::ids::{SessionId, Ticket};
use super::residency::{DeviceId, ExpertKey, Lease, ResidencyError, ResidencyManager};

/// Opaque engine-owned accelerator work: a payload plus the resident
/// resource generations the action references. Both fields are interpreted
/// by the backend and the model crate, never by `logan-core`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccelAction {
    pub payload: Vec<u8>,
    /// Resource keys whose current resident generation the submission must
    /// retain until completion processing releases it.
    pub references: Vec<ExpertKey>,
}

/// Opaque backend-owned native resource handle slot. A backend stores its
/// command buffer / stream / event identity here; the scheduler never
/// interprets the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeHandle(pub u64);

/// What the backend reports for one submitted ticket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccelOutcome {
    Success,
    Failed(String),
    Cancelled,
}

/// Typed completion event delivered to the scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccelCompletion {
    pub ticket: Ticket,
    pub outcome: AccelOutcome,
}

/// Effect the driver hands to the backend: execute `action` on `target`.
///
/// This replaces the V0 core's plain `Effect::Dispatch` for accelerator
/// work; the ticket and lifecycle semantics are still the core's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccelDispatch {
    pub ticket: Ticket,
    pub target: ExecutionTarget,
    pub action: AccelAction,
}

/// Invalid accelerator contract operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccelError {
    /// The submitted target is not the registered target for its device.
    UnknownTarget(DeviceId),
    /// The registered target does not accept accelerator work.
    UnsupportedTarget(DeviceId),
    /// A referenced resource is not resident at submission time.
    NotResident(ExpertKey),
    /// Scheduler-core rejection (unknown session, not runnable, inflight limit).
    Core(SchedError),
    /// A retained lease could not be returned during completion processing.
    Release {
        ticket: Ticket,
        error: ResidencyError,
    },
}

/// Scheduler-side contract bookkeeping: retained residency leases per live
/// accelerator ticket. Single-owner, like every other `sched` state machine.
#[derive(Debug, Default)]
pub struct Accel {
    /// Leases retained by each live ticket, held until completion reporting
    /// or `cancel_ticket` removes them. A lease appears here exactly once.
    inflight: HashMap<Ticket, Vec<Lease>>,
}

impl Accel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Nonblocking submission: validate the compiler-approved target, pin
    /// every referenced resident generation, and allocate the core ticket.
    ///
    /// Leases are acquired before the ticket exists so a rejected submission
    /// can never leak a ticket or a lease.
    pub fn submit(
        &mut self,
        core: &mut SchedulerCore,
        residency: &mut ResidencyManager,
        registry: &DeviceRegistry,
        session: SessionId,
        target: &ExecutionTarget,
        action: AccelAction,
    ) -> Result<(Ticket, AccelDispatch), AccelError> {
        let registered = registry
            .get(target.device)
            .filter(|registered| *registered == target)
            .ok_or(AccelError::UnknownTarget(target.device))?;
        if !registered.accepts(ActionKind::Accelerator) {
            return Err(AccelError::UnsupportedTarget(target.device));
        }
        let mut leases = Vec::with_capacity(action.references.len());
        for key in &action.references {
            match residency.acquire(key, session) {
                Ok(lease) => leases.push(lease),
                // Any pin failure means the referenced generation is not
                // resident. Roll back the references pinned so far.
                Err(_) => {
                    for lease in leases.drain(..) {
                        let _ = residency.release(lease);
                    }
                    return Err(AccelError::NotResident(key.clone()));
                }
            }
        }
        let (ticket, _core_dispatch) = match core.submit(
            session,
            Action {
                kind: ActionKind::Accelerator,
                payload: action.payload.clone(),
            },
        ) {
            Ok(submitted) => submitted,
            Err(error) => {
                for lease in leases.drain(..) {
                    let _ = residency.release(lease);
                }
                return Err(AccelError::Core(error));
            }
        };
        self.inflight.insert(ticket, leases);
        Ok((
            ticket,
            AccelDispatch {
                ticket,
                target: registered.clone(),
                action,
            },
        ))
    }

    /// Completion processing: release the ticket's retained generations,
    /// then forward the typed outcome into the core's exact-once ticket
    /// path. A stale, duplicate, or already-cancelled ticket has no lease
    /// record and releases nothing; the core treats the forwarded completion
    /// as a no-op.
    pub fn complete(
        &mut self,
        core: &mut SchedulerCore,
        residency: &mut ResidencyManager,
        completion: AccelCompletion,
    ) -> Result<Vec<Effect>, AccelError> {
        if let Some(leases) = self.inflight.remove(&completion.ticket) {
            for lease in leases {
                residency.release(lease).map_err(|error| AccelError::Release {
                    ticket: completion.ticket,
                    error,
                })?;
            }
        }
        let outcome = match completion.outcome {
            AccelOutcome::Success => Outcome::Ok,
            AccelOutcome::Failed(message) => Outcome::Err(message),
            AccelOutcome::Cancelled => Outcome::Err("accelerator action cancelled".into()),
        };
        core.complete(completion.ticket, outcome)
            .map_err(AccelError::Core)
    }

    /// Release the retained generations of one ticket whose session was
    /// cancelled, finished, or failed by the core. The backend's eventual
    /// report for that ticket then hits neither this bookkeeping nor the
    /// core. Idempotent: returns whether a record existed.
    pub fn cancel_ticket(&mut self, residency: &mut ResidencyManager, ticket: Ticket) -> bool {
        let Some(leases) = self.inflight.remove(&ticket) else {
            return false;
        };
        for lease in leases {
            let _ = residency.release(lease);
        }
        true
    }

    /// Number of tickets with retained leases awaiting completion.
    pub fn inflight_len(&self) -> usize {
        self.inflight.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched::core::{Budget, SessionState, Step};
    use crate::sched::device::DeviceKind;
    use crate::sched::ids::Generation;
    use crate::sched::residency::{
        CompletionDisposition, LoadCompletion, LoadDisposition, LoadResult, MemoryPoolId,
        RepresentationKey, ResidencyState,
    };

    // --- shared fixtures ---------------------------------------------------

    fn pool(manager: &mut ResidencyManager, id: u32, budget: usize) -> MemoryPoolId {
        let pool = MemoryPoolId(id);
        manager.add_pool(pool, budget).unwrap();
        pool
    }

    fn key(pool: MemoryPoolId, layout: u32) -> ExpertKey {
        ExpertKey {
            model: 1,
            package: 2,
            layer: 3,
            expert: 4,
            representation: RepresentationKey {
                layout,
                kernel_abi: 7,
                quant: 8,
            },
            pool,
        }
    }

    /// Load `bytes` for a key and publish it resident. The begin-load waiter
    /// lease is returned to the manager, leaving the entry unpinned.
    fn resident(manager: &mut ResidencyManager, pool: MemoryPoolId, layout: u32, bytes: usize) -> ExpertKey {
        let key = key(pool, layout);
        let load = match manager.begin_load(key.clone(), bytes, SessionId(0)).unwrap() {
            LoadDisposition::Started { load_id } => load_id,
            other => panic!("unexpected {other:?}"),
        };
        let disposition = manager
            .complete_load(LoadCompletion {
                load_id: load,
                result: LoadResult::Success,
            })
            .unwrap();
        let CompletionDisposition::Published { wakes, .. } = disposition else {
            panic!("load must publish")
        };
        for wake in wakes {
            if let Ok(lease) = wake.result {
                manager.release(lease).unwrap();
            }
        }
        key
    }

    fn gpu_target(registry: &mut DeviceRegistry, id: u32) -> ExecutionTarget {
        let target = ExecutionTarget {
            device: DeviceId(id),
            kind: DeviceKind::Gpu,
            capabilities: vec![ActionKind::Accelerator],
        };
        registry.register(target.clone()).unwrap();
        target
    }

    fn action(payload: &[u8], references: Vec<ExpertKey>) -> AccelAction {
        AccelAction {
            payload: payload.to_vec(),
            references,
        }
    }

    /// Mock backend target (tests only): records every dispatch it accepts
    /// and the opaque handle it would own; never touches real hardware.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct MockTarget {
        dispatches: Vec<MockDispatch>,
        next_handle: u64,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct MockDispatch {
        ticket: Ticket,
        target: ExecutionTarget,
        payload: Vec<u8>,
        handle: NativeHandle,
        references: usize,
    }

    impl MockTarget {
        fn new() -> Self {
            Self {
                dispatches: Vec::new(),
                next_handle: 0,
            }
        }

        fn submit(&mut self, dispatch: &AccelDispatch) {
            let handle = NativeHandle(self.next_handle);
            self.next_handle += 1;
            self.dispatches.push(MockDispatch {
                ticket: dispatch.ticket,
                target: dispatch.target.clone(),
                payload: dispatch.action.payload.clone(),
                handle,
                references: dispatch.action.references.len(),
            });
        }

        fn dispatch(&self, ticket: Ticket) -> &MockDispatch {
            self.dispatches
                .iter()
                .find(|record| record.ticket == ticket)
                .expect("mock must have recorded the dispatch")
        }

        fn completion(&self, ticket: Ticket, outcome: AccelOutcome) -> AccelCompletion {
            AccelCompletion { ticket, outcome }
        }
    }

    // --- contract behaviors ------------------------------------------------

    #[test]
    fn submit_dispatches_to_target_and_success_resumes_blocked_session() {
        let mut registry = DeviceRegistry::new();
        let target = gpu_target(&mut registry, 0);
        let mut residency = ResidencyManager::new();
        let pool = pool(&mut residency, 0, 100);
        let expert = resident(&mut residency, pool, 1, 40);
        let mut core = SchedulerCore::new(Budget::default());
        let session = core.create_session();
        let mut accel = Accel::new();
        let mut mock = MockTarget::new();

        let (ticket, dispatch) = accel
            .submit(
                &mut core,
                &mut residency,
                &registry,
                session,
                &target,
                action(b"decode", vec![expert.clone()]),
            )
            .unwrap();
        assert_eq!(dispatch.ticket, ticket);
        assert_eq!(dispatch.target, target);
        mock.submit(&dispatch);
        // The referenced generation is retained for the whole flight.
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 40);
        let recorded = mock.dispatch(ticket);
        assert_eq!(recorded.payload, b"decode");
        assert_eq!(recorded.references, 1);
        assert_eq!(recorded.handle, NativeHandle(0));

        core.block(session).unwrap();
        let effects = accel
            .complete(
                &mut core,
                &mut residency,
                mock.completion(ticket, AccelOutcome::Success),
            )
            .unwrap();
        assert_eq!(effects, vec![Effect::Wake { session }]);
        assert_eq!(core.session_state(session), Some(SessionState::Runnable));
        // Completion processing released the retained generation.
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 0);
        assert_eq!(accel.inflight_len(), 0);
        assert_eq!(core.step(), Step::Run(session));
    }

    #[test]
    fn failure_publication_terminates_session_and_duplicates_are_stale() {
        let mut registry = DeviceRegistry::new();
        let target = gpu_target(&mut registry, 0);
        let mut residency = ResidencyManager::new();
        let pool = pool(&mut residency, 0, 100);
        resident(&mut residency, pool, 1, 40);
        let mut core = SchedulerCore::new(Budget::default());
        let session = core.create_session();
        let mut accel = Accel::new();
        let mut mock = MockTarget::new();

        let (ticket, dispatch) = accel
            .submit(
                &mut core,
                &mut residency,
                &registry,
                session,
                &target,
                action(b"decode", vec![key(pool, 1)]),
            )
            .unwrap();
        mock.submit(&dispatch);
        core.block(session).unwrap();

        let effects = accel
            .complete(
                &mut core,
                &mut residency,
                mock.completion(ticket, AccelOutcome::Failed("kernel aborted".into())),
            )
            .unwrap();
        assert_eq!(
            effects,
            vec![Effect::Failed {
                session,
                error: "kernel aborted".into()
            }]
        );
        assert_eq!(core.session_state(session), Some(SessionState::Failed));
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 0);

        // Duplicate completion is a harmless no-op: same state, no release.
        assert!(accel
            .complete(
                &mut core,
                &mut residency,
                mock.completion(ticket, AccelOutcome::Success),
            )
            .unwrap()
            .is_empty());
        assert_eq!(core.session_state(session), Some(SessionState::Failed));
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 0);
        assert_eq!(accel.inflight_len(), 0);
    }

    #[test]
    fn cancellation_while_in_flight_releases_retention_and_late_reports_are_noops() {
        let mut registry = DeviceRegistry::new();
        let target = gpu_target(&mut registry, 0);
        let mut residency = ResidencyManager::new();
        let pool = pool(&mut residency, 0, 100);
        resident(&mut residency, pool, 1, 40);
        let mut core = SchedulerCore::new(Budget::default());
        let mut accel = Accel::new();
        let mut mock = MockTarget::new();

        let doomed = core.create_session();
        let (ticket, dispatch) = accel
            .submit(
                &mut core,
                &mut residency,
                &registry,
                doomed,
                &target,
                action(b"decode", vec![key(pool, 1)]),
            )
            .unwrap();
        mock.submit(&dispatch);
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 40);

        // The core revokes the session; the driver releases the retention.
        assert_eq!(core.cancel(doomed).unwrap(), vec![Effect::Cancel { ticket }]);
        assert!(accel.cancel_ticket(&mut residency, ticket));
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 0);

        // The backend's late report changes nothing and errors nothing.
        assert!(accel
            .complete(
                &mut core,
                &mut residency,
                mock.completion(ticket, AccelOutcome::Cancelled),
            )
            .unwrap()
            .is_empty());
        assert_eq!(core.session_state(doomed), Some(SessionState::Cancelling));
        assert!(!accel.cancel_ticket(&mut residency, ticket));

        // The contract still works for an unrelated session.
        let survivor = core.create_session();
        let (other, dispatch2) = accel
            .submit(
                &mut core,
                &mut residency,
                &registry,
                survivor,
                &target,
                action(b"decode", vec![key(pool, 1)]),
            )
            .unwrap();
        mock.submit(&dispatch2);
        core.block(survivor).unwrap();
        assert_eq!(
            accel
                .complete(
                    &mut core,
                    &mut residency,
                    mock.completion(other, AccelOutcome::Success),
                )
                .unwrap(),
            vec![Effect::Wake { session: survivor }]
        );
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 0);
    }

    #[test]
    fn retained_generation_blocks_eviction_until_completion_releases() {
        let mut registry = DeviceRegistry::new();
        let target = gpu_target(&mut registry, 0);
        let mut residency = ResidencyManager::new();
        let pool = pool(&mut residency, 0, 100);
        let expert = resident(&mut residency, pool, 1, 40);
        let mut core = SchedulerCore::new(Budget::default());
        let session = core.create_session();
        let mut accel = Accel::new();
        let mut mock = MockTarget::new();

        let (ticket, dispatch) = accel
            .submit(
                &mut core,
                &mut residency,
                &registry,
                session,
                &target,
                action(b"decode", vec![expert.clone()]),
            )
            .unwrap();
        mock.submit(&dispatch);
        // In flight: the pinned generation cannot be evicted or reused.
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 40);
        assert_eq!(residency.evict(pool).unwrap(), None);

        accel
            .complete(
                &mut core,
                &mut residency,
                mock.completion(ticket, AccelOutcome::Success),
            )
            .unwrap();
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 0);
        // Only after completion may the generation be evicted (reused).
        assert_eq!(residency.evict(pool).unwrap(), Some(expert.clone()));

        // The generation is reused: publish generation 1 (waiter lease
        // returned, so the entry is not pinned by the reload either).
        let reload = match residency.begin_load(expert.clone(), 40, SessionId(1)).unwrap() {
            LoadDisposition::Started { load_id } => load_id,
            other => panic!("unexpected {other:?}"),
        };
        let disposition = residency
            .complete_load(LoadCompletion {
                load_id: reload,
                result: LoadResult::Success,
            })
            .unwrap();
        let CompletionDisposition::Published { wakes, .. } = disposition else {
            panic!("reload must publish")
        };
        for wake in wakes {
            if let Ok(lease) = wake.result {
                residency.release(lease).unwrap();
            }
        }
        assert!(matches!(
            residency.state(&expert),
            ResidencyState::Resident { generation } if generation == Generation(1)
        ));

        // A stale duplicate completion after the reuse releases nothing and
        // cannot observe the new generation: the old lease was already
        // returned at first completion.
        assert!(accel
            .complete(
                &mut core,
                &mut residency,
                mock.completion(ticket, AccelOutcome::Success),
            )
            .unwrap()
            .is_empty());
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 0);
        assert!(matches!(
            residency.state(&expert),
            ResidencyState::Resident { generation } if generation == Generation(1)
        ));
    }

    #[test]
    fn stale_unknown_and_duplicate_completions_are_harmless() {
        let mut registry = DeviceRegistry::new();
        let target = gpu_target(&mut registry, 0);
        let mut residency = ResidencyManager::new();
        let pool = pool(&mut residency, 0, 100);
        resident(&mut residency, pool, 1, 40);
        let mut core = SchedulerCore::new(Budget::default());
        let session = core.create_session();
        let mut accel = Accel::new();
        let mut mock = MockTarget::new();

        // Completion for a ticket this contract never issued.
        assert!(accel
            .complete(
                &mut core,
                &mut residency,
                AccelCompletion {
                    ticket: Ticket(999),
                    outcome: AccelOutcome::Success,
                },
            )
            .unwrap()
            .is_empty());
        assert_eq!(accel.inflight_len(), 0);

        let (ticket, dispatch) = accel
            .submit(
                &mut core,
                &mut residency,
                &registry,
                session,
                &target,
                action(b"decode", vec![key(pool, 1)]),
            )
            .unwrap();
        mock.submit(&dispatch);
        // Other tickets do not disturb the live one.
        assert!(accel
            .complete(
                &mut core,
                &mut residency,
                AccelCompletion {
                    ticket: Ticket(998),
                    outcome: AccelOutcome::Failed("late stranger".into()),
                },
            )
            .unwrap()
            .is_empty());
        assert_eq!(accel.inflight_len(), 1);

        assert!(accel
            .complete(
                &mut core,
                &mut residency,
                mock.completion(ticket, AccelOutcome::Success),
            )
            .unwrap()
            .is_empty()); // session not blocked: nothing to wake
        assert!(accel
            .complete(
                &mut core,
                &mut residency,
                mock.completion(ticket, AccelOutcome::Success),
            )
            .unwrap()
            .is_empty());
        assert_eq!(accel.inflight_len(), 0);
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 0);
    }

    #[test]
    fn rejects_unapproved_targets_and_rolls_back_failed_submissions() {
        let mut registry = DeviceRegistry::new();
        let mut residency = ResidencyManager::new();
        let pool = pool(&mut residency, 0, 100);
        let target = gpu_target(&mut registry, 1);
        let mut core = SchedulerCore::new(Budget::default());
        let session = core.create_session();
        let mut accel = Accel::new();

        // No registered target for the device.
        let rogue = ExecutionTarget {
            device: DeviceId(9),
            kind: DeviceKind::Gpu,
            capabilities: vec![ActionKind::Accelerator],
        };
        assert_eq!(
            accel
                .submit(&mut core, &mut residency, &registry, session, &rogue, action(b"x", vec![]))
                .unwrap_err(),
            AccelError::UnknownTarget(DeviceId(9))
        );

        // Registered target that does not accept accelerator work.
        let cpu = ExecutionTarget {
            device: DeviceId(2),
            kind: DeviceKind::Cpu,
            capabilities: vec![ActionKind::Cpu],
        };
        registry.register(cpu.clone()).unwrap();
        assert_eq!(
            accel
                .submit(&mut core, &mut residency, &registry, session, &cpu, action(b"x", vec![]))
                .unwrap_err(),
            AccelError::UnsupportedTarget(DeviceId(2))
        );

        // Referenced resource never loaded: no ticket, no retained lease.
        let missing = key(pool, 2);
        assert_eq!(
            accel
                .submit(
                    &mut core,
                    &mut residency,
                    &registry,
                    session,
                    &target,
                    action(b"x", vec![missing.clone()]),
                )
                .unwrap_err(),
            AccelError::NotResident(missing)
        );
        assert_eq!(accel.inflight_len(), 0);
        assert_eq!(core.inflight_count(session), Some(0));
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 0);

        // Core rejection after pinning rolls the acquired leases back.
        let expert = resident(&mut residency, pool, 1, 40);
        core.block(session).unwrap();
        assert_eq!(
            accel
                .submit(
                    &mut core,
                    &mut residency,
                    &registry,
                    session,
                    &target,
                    action(b"x", vec![expert.clone()]),
                )
                .unwrap_err(),
            AccelError::Core(SchedError::NotRunnable(session))
        );
        assert_eq!(accel.inflight_len(), 0);
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 0);
    }

    #[test]
    fn out_of_order_completion_resumes_only_its_own_session() {
        let mut registry = DeviceRegistry::new();
        let target = gpu_target(&mut registry, 0);
        let mut residency = ResidencyManager::new();
        let pool = pool(&mut residency, 0, 100);
        resident(&mut residency, pool, 1, 40);
        let mut core = SchedulerCore::new(Budget::default());
        let mut accel = Accel::new();
        let mut mock = MockTarget::new();

        // Two independent sessions share one resident generation; each holds
        // its own lease on it.
        let first = core.create_session();
        let second = core.create_session();
        let (ticket_a, dispatch_a) = accel
            .submit(
                &mut core,
                &mut residency,
                &registry,
                first,
                &target,
                action(b"decode-a", vec![key(pool, 1)]),
            )
            .unwrap();
        let (ticket_b, dispatch_b) = accel
            .submit(
                &mut core,
                &mut residency,
                &registry,
                second,
                &target,
                action(b"decode-b", vec![key(pool, 1)]),
            )
            .unwrap();
        mock.submit(&dispatch_a);
        mock.submit(&dispatch_b);
        core.block(first).unwrap();
        core.block(second).unwrap();
        assert_eq!(accel.inflight_len(), 2);
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 40);

        // Device completes B first, out of order: only B resumes, and A's
        // lease still pins the generation.
        assert_eq!(
            accel
                .complete(
                    &mut core,
                    &mut residency,
                    mock.completion(ticket_b, AccelOutcome::Success),
                )
                .unwrap(),
            vec![Effect::Wake { session: second }]
        );
        assert_eq!(core.session_state(first), Some(SessionState::Blocked));
        assert_eq!(core.session_state(second), Some(SessionState::Runnable));
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 40);

        // A's completion releases the last lease.
        assert_eq!(
            accel
                .complete(
                    &mut core,
                    &mut residency,
                    mock.completion(ticket_a, AccelOutcome::Success),
                )
                .unwrap(),
            vec![Effect::Wake { session: first }]
        );
        assert_eq!(accel.inflight_len(), 0);
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 0);
    }
}