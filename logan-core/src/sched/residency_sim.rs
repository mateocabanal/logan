//! Deterministic fake residency driver, trace, and replay for issue #47.
//!
//! Mirrors `sim.rs` (V0 scheduler core): drives the real `ResidencyManager`
//! with explicit scripted operations and no threads, clocks, sleeps, I/O, or
//! devices. The same operation stream produces the same decisions, IDs,
//! effects, and errors on every run, so traces can be replayed byte-for-byte.

use super::ids::{LoadId, SessionId};
use super::residency::{
    CompletionDisposition, DeviceId, ExpertKey, Lease, LoadCompletion, LoadDisposition,
    MemoryPoolId, PoolStats, ResidencyError, ResidencyManager, ResidencyState,
};

/// One deterministic residency operation accepted by the simulator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidencyOp {
    AddPool {
        pool: MemoryPoolId,
        budget: usize,
    },
    AddDevice {
        device: DeviceId,
        pools: Vec<MemoryPoolId>,
    },
    BeginLoad {
        key: ExpertKey,
        bytes: usize,
        waiter: SessionId,
    },
    CancelWaiter {
        load_id: LoadId,
        session: SessionId,
    },
    CompleteLoad {
        completion: LoadCompletion,
    },
    Acquire {
        key: ExpertKey,
        owner: SessionId,
    },
    Release {
        lease: Lease,
    },
    Evict {
        pool: MemoryPoolId,
    },
}

/// Deterministic result of one simulated residency operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidencyOutcome {
    Configured(Result<(), ResidencyError>),
    Load(Result<LoadDisposition, ResidencyError>),
    WaiterCancelled(bool),
    Completion(Result<CompletionDisposition, ResidencyError>),
    Lease(Result<Lease, ResidencyError>),
    Released(Result<(), ResidencyError>),
    Evicted(Result<Option<ExpertKey>, ResidencyError>),
}

/// One residency trace record. `seq` is a logical sequence number, never
/// wall-clock time, matching the V0 trace convention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyTraceEvent {
    pub seq: u64,
    pub op: ResidencyOp,
    pub outcome: ResidencyOutcome,
}

/// A pure simulator around the production residency manager.
#[derive(Debug)]
pub struct ResidencySimulator {
    manager: ResidencyManager,
    trace: Vec<ResidencyTraceEvent>,
    next_seq: u64,
}

impl ResidencySimulator {
    pub fn new() -> Self {
        Self {
            manager: ResidencyManager::new(),
            trace: Vec::new(),
            next_seq: 0,
        }
    }

    /// Apply and record one operation, including benign stale/no-op outcomes.
    pub fn apply(&mut self, op: ResidencyOp) -> ResidencyOutcome {
        let outcome = self.execute(&op);
        self.trace.push(ResidencyTraceEvent {
            seq: self.next_seq,
            op,
            outcome: outcome.clone(),
        });
        self.next_seq += 1;
        #[cfg(debug_assertions)]
        self.manager.assert_invariants();
        outcome
    }

    fn execute(&mut self, op: &ResidencyOp) -> ResidencyOutcome {
        match op {
            ResidencyOp::AddPool { pool, budget } => {
                ResidencyOutcome::Configured(self.manager.add_pool(*pool, *budget))
            }
            ResidencyOp::AddDevice { device, pools } => {
                ResidencyOutcome::Configured(self.manager.add_device(*device, pools.clone()))
            }
            ResidencyOp::BeginLoad { key, bytes, waiter } => {
                ResidencyOutcome::Load(self.manager.begin_load(key.clone(), *bytes, *waiter))
            }
            ResidencyOp::CancelWaiter { load_id, session } => {
                ResidencyOutcome::WaiterCancelled(self.manager.cancel_waiter(*load_id, *session))
            }
            ResidencyOp::CompleteLoad { completion } => {
                ResidencyOutcome::Completion(self.manager.complete_load(completion.clone()))
            }
            ResidencyOp::Acquire { key, owner } => {
                ResidencyOutcome::Lease(self.manager.acquire(key, *owner))
            }
            ResidencyOp::Release { lease } => {
                ResidencyOutcome::Released(self.manager.release(lease.clone()))
            }
            ResidencyOp::Evict { pool } => ResidencyOutcome::Evicted(self.manager.evict(*pool)),
        }
    }

    pub fn trace(&self) -> &[ResidencyTraceEvent] {
        &self.trace
    }

    pub fn state(&self, key: &ExpertKey) -> ResidencyState {
        self.manager.state(key)
    }

    pub fn pool_stats(&self, pool: MemoryPoolId) -> Result<PoolStats, ResidencyError> {
        self.manager.pool_stats(pool)
    }

    pub fn device_pools(&self, device: DeviceId) -> Result<&[MemoryPoolId], ResidencyError> {
        self.manager.device_pools(device)
    }

    /// Replay an existing residency trace and compare every logical decision,
    /// including allocated IDs, effects, errors, and event sequence numbers.
    // The error type deliberately carries the full mismatched events for
    // diffing; same shape and lint as the V0 `SchedulerSimulator::replay`.
    #[allow(clippy::result_large_err)]
    pub fn replay(expected: &[ResidencyTraceEvent]) -> Result<(), ResidencyReplayError> {
        let mut simulator = Self::new();
        for expected_event in expected {
            let actual_outcome = simulator.apply(expected_event.op.clone());
            let actual_event = simulator.trace.last().expect("apply records an event");
            if actual_event != expected_event {
                return Err(ResidencyReplayError {
                    seq: expected_event.seq,
                    expected: expected_event.clone(),
                    actual: actual_event.clone(),
                });
            }
            debug_assert_eq!(actual_outcome, expected_event.outcome);
        }
        Ok(())
    }
}

impl Default for ResidencySimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyReplayError {
    pub seq: u64,
    pub expected: ResidencyTraceEvent,
    pub actual: ResidencyTraceEvent,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched::ids::Generation;
    use crate::sched::residency::{LoadResult, RepresentationKey};

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

    fn started(
        simulator: &mut ResidencySimulator,
        key: &ExpertKey,
        bytes: usize,
        session: SessionId,
    ) -> LoadId {
        match simulator.apply(ResidencyOp::BeginLoad {
            key: key.clone(),
            bytes,
            waiter: session,
        }) {
            ResidencyOutcome::Load(Ok(LoadDisposition::Started { load_id })) => load_id,
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn coalesced_load_then_pin_evict_cycle_replays_identically() {
        let pool = MemoryPoolId(0);
        let expert = key(pool, 1);
        let mut simulator = ResidencySimulator::new();
        simulator.apply(ResidencyOp::AddPool { pool, budget: 100 });
        simulator.apply(ResidencyOp::AddDevice {
            device: DeviceId(0),
            pools: vec![pool],
        });
        let load = started(&mut simulator, &expert, 40, SessionId(1));
        assert_eq!(
            simulator.apply(ResidencyOp::BeginLoad {
                key: expert.clone(),
                bytes: 40,
                waiter: SessionId(2),
            }),
            ResidencyOutcome::Load(Ok(LoadDisposition::Joined { load_id: load }))
        );
        // One waiter leaves; the shared load stays alive for the other.
        assert_eq!(
            simulator.apply(ResidencyOp::CancelWaiter {
                load_id: load,
                session: SessionId(1),
            }),
            ResidencyOutcome::WaiterCancelled(true)
        );
        let wake_lease = match simulator.apply(ResidencyOp::CompleteLoad {
            completion: LoadCompletion {
                load_id: load,
                result: LoadResult::Success,
            },
        }) {
            ResidencyOutcome::Completion(Ok(CompletionDisposition::Published {
                generation: Generation(0),
                wakes,
            })) if wakes.len() == 1 && wakes[0].session == SessionId(2) => {
                wakes[0].result.clone().expect("waiter got a lease")
            }
            other => panic!("unexpected result: {other:?}"),
        };
        // The surviving waiter's lease pins the published generation.
        assert_eq!(
            simulator.apply(ResidencyOp::Evict { pool }),
            ResidencyOutcome::Evicted(Ok(None))
        );
        assert_eq!(
            simulator.apply(ResidencyOp::Release { lease: wake_lease }),
            ResidencyOutcome::Released(Ok(()))
        );
        assert_eq!(
            simulator.apply(ResidencyOp::Evict { pool }),
            ResidencyOutcome::Evicted(Ok(Some(expert.clone())))
        );
        // Re-load after eviction starts a fresh load.
        let reload = started(&mut simulator, &expert, 40, SessionId(3));
        let reload_lease = match simulator.apply(ResidencyOp::CompleteLoad {
            completion: LoadCompletion {
                load_id: reload,
                result: LoadResult::Success,
            },
        }) {
            ResidencyOutcome::Completion(Ok(CompletionDisposition::Published {
                wakes, ..
            })) => wakes[0].result.clone().expect("waiter got a lease"),
            other => panic!("unexpected result: {other:?}"),
        };
        // Re-pin via an explicit acquire; eviction blocked until release.
        let lease = match simulator.apply(ResidencyOp::Acquire {
            key: expert.clone(),
            owner: SessionId(3),
        }) {
            ResidencyOutcome::Lease(Ok(lease)) => lease,
            other => panic!("unexpected result: {other:?}"),
        };
        assert_eq!(
            simulator.apply(ResidencyOp::Evict { pool }),
            ResidencyOutcome::Evicted(Ok(None))
        );
        assert_eq!(
            simulator.apply(ResidencyOp::Release { lease }),
            ResidencyOutcome::Released(Ok(()))
        );
        assert_eq!(
            simulator.apply(ResidencyOp::Release {
                lease: reload_lease,
            }),
            ResidencyOutcome::Released(Ok(()))
        );
        assert_eq!(
            simulator.apply(ResidencyOp::Evict { pool }),
            ResidencyOutcome::Evicted(Ok(Some(expert.clone())))
        );
        let stats = simulator.pool_stats(pool).unwrap();
        assert_eq!(stats.resident_bytes, 0);
        assert_eq!(stats.reserved_loading_bytes, 0);
        assert_eq!(stats.loads_started, 2);
        assert_eq!(stats.loads_joined, 1);
        assert_eq!(stats.evictions, 2);
        ResidencySimulator::replay(simulator.trace()).unwrap();
    }

    /// Load a key to resident and immediately release its wake lease so the
    /// entry is unpinned, the steady state the eviction tests need.
    fn resident(simulator: &mut ResidencySimulator, key: &ExpertKey, bytes: usize) {
        let load = started(simulator, key, bytes, SessionId(9));
        let wakes = match simulator.apply(ResidencyOp::CompleteLoad {
            completion: LoadCompletion {
                load_id: load,
                result: LoadResult::Success,
            },
        }) {
            ResidencyOutcome::Completion(Ok(CompletionDisposition::Published {
                wakes, ..
            })) => wakes,
            other => panic!("unexpected result: {other:?}"),
        };
        for wake in wakes {
            let lease = wake.result.expect("waiter got a lease");
            assert_eq!(
                simulator.apply(ResidencyOp::Release { lease }),
                ResidencyOutcome::Released(Ok(()))
            );
        }
        assert!(matches!(
            simulator.state(key),
            ResidencyState::Resident { .. }
        ));
    }

    #[test]
    fn pinned_and_loading_entries_are_never_evicted_under_pressure() {
        let pool = MemoryPoolId(0);
        let a = key(pool, 1);
        let b = key(pool, 2);
        let c = key(pool, 3);
        let mut simulator = ResidencySimulator::new();
        simulator.apply(ResidencyOp::AddPool { pool, budget: 100 });
        // a = 60 resident, b = 40 resident, both unpinned; a gets pinned.
        resident(&mut simulator, &a, 60);
        resident(&mut simulator, &b, 40);
        let lease = match simulator.apply(ResidencyOp::Acquire {
            key: a.clone(),
            owner: SessionId(1),
        }) {
            ResidencyOutcome::Lease(Ok(lease)) => lease,
            other => panic!("unexpected result: {other:?}"),
        };
        // c(60) needs 60: evicting b frees only 40, a is pinned -> hard failure.
        assert!(matches!(
            simulator.apply(ResidencyOp::BeginLoad {
                key: c.clone(),
                bytes: 60,
                waiter: SessionId(3),
            }),
            ResidencyOutcome::Load(Err(ResidencyError::BudgetExceeded { .. }))
        ));
        assert!(matches!(
            simulator.state(&a),
            ResidencyState::Resident { .. }
        ));
        assert!(matches!(simulator.state(&b), ResidencyState::Absent));
        assert_eq!(simulator.pool_stats(pool).unwrap().evictions, 1);
        // Releasing the pin makes a evictable; c now fits and loads.
        assert_eq!(
            simulator.apply(ResidencyOp::Release { lease }),
            ResidencyOutcome::Released(Ok(()))
        );
        let lc = started(&mut simulator, &c, 60, SessionId(3));
        assert!(matches!(simulator.state(&a), ResidencyState::Absent));
        simulator.apply(ResidencyOp::CompleteLoad {
            completion: LoadCompletion {
                load_id: lc,
                result: LoadResult::Success,
            },
        });
        let stats = simulator.pool_stats(pool).unwrap();
        assert_eq!(stats.resident_bytes, 60);
        assert_eq!(stats.reserved_loading_bytes, 0);
        assert_eq!(stats.evictions, 2);
        ResidencySimulator::replay(simulator.trace()).unwrap();
    }

    #[test]
    fn failed_load_wakes_every_waiter_and_frees_the_reservation() {
        let pool = MemoryPoolId(0);
        let expert = key(pool, 1);
        let mut simulator = ResidencySimulator::new();
        simulator.apply(ResidencyOp::AddPool { pool, budget: 50 });
        let load = started(&mut simulator, &expert, 40, SessionId(1));
        assert_eq!(
            simulator.apply(ResidencyOp::BeginLoad {
                key: expert.clone(),
                bytes: 40,
                waiter: SessionId(2),
            }),
            ResidencyOutcome::Load(Ok(LoadDisposition::Joined { load_id: load }))
        );
        assert!(matches!(
            simulator.apply(ResidencyOp::CompleteLoad {
                completion: LoadCompletion {
                    load_id: load,
                    result: LoadResult::Failed("io error".into()),
                },
            }),
            ResidencyOutcome::Completion(Ok(CompletionDisposition::Failed { wakes }))
                if wakes.len() == 2 && wakes.iter().all(|wake| wake.result.is_err())
        ));
        let stats = simulator.pool_stats(pool).unwrap();
        assert_eq!(stats.reserved_loading_bytes, 0);
        assert_eq!(stats.resident_bytes, 0);
        assert_eq!(stats.failures, 1);
        assert!(matches!(
            simulator.state(&expert),
            ResidencyState::Failed { .. }
        ));
        ResidencySimulator::replay(simulator.trace()).unwrap();
    }
}
