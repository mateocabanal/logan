//! Scheduler-owned residency state for issue #47.
//!
//! This module tracks logical residency only. It deliberately contains no
//! backend handles, raw pointers, file descriptors, device waits, or async
//! executor machinery. A runtime translates `LoadDisposition` and
//! `LoadCompletion` into backend work and scheduler commands.

use std::collections::HashMap;

use super::ids::{Generation, LoadId, SessionId};

/// A typed identifier for a physical memory budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MemoryPoolId(pub u32);

/// A typed identifier for an execution device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceId(pub u32);

/// A typed identifier for one retained lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LeaseId(pub u64);

/// The physical representation requested by a plan.
///
/// Layout and kernel ABI are intentionally separate from quantization: two
/// representations of one expert must never collapse into one cache entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RepresentationKey {
    pub layout: u32,
    pub kernel_abi: u32,
    pub quant: u16,
}

/// A fully qualified physical expert identity.
///
/// `layer` and `expert` alone are not sufficient. Model/package identity,
/// representation, and destination pool all affect the bytes and their legal
/// consumers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub model: u64,
    pub package: u64,
    pub layer: u32,
    pub expert: u32,
    pub representation: RepresentationKey,
    pub pool: MemoryPoolId,
}

/// Public residency state for diagnostics and simulator assertions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidencyState {
    Absent,
    Loading { load_id: LoadId },
    Resident { generation: Generation },
    Failed { message: String },
}

/// Result of asking the manager to make one key available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadDisposition {
    Started { load_id: LoadId },
    Joined { load_id: LoadId },
    AlreadyResident { generation: Generation },
}

/// Result reported by an external loader after its physical work ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadResult {
    Success,
    Failed(String),
    Partial(String),
    Cancelled,
}

/// A lease handed to a waiter or acquired by a ready session.
///
/// The lease retains the exact generation. Releasing a lease can therefore
/// never accidentally release a newer resident publication for the same key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lease {
    pub id: LeaseId,
    pub key: ExpertKey,
    pub generation: Generation,
    pub owner: SessionId,
}

/// A waiter wake-up produced by load completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadWake {
    pub session: SessionId,
    pub result: Result<Lease, String>,
}

/// Completion result returned to the runtime/driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionDisposition {
    Published {
        generation: Generation,
        wakes: Vec<LoadWake>,
    },
    Failed {
        wakes: Vec<LoadWake>,
    },
    Stale,
}

/// Completion envelope accepted from an external loader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadCompletion {
    pub load_id: LoadId,
    pub result: LoadResult,
}

/// Cumulative and current counters for one physical memory pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolStats {
    pub budget_bytes: usize,
    pub resident_bytes: usize,
    pub reserved_loading_bytes: usize,
    pub pinned_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub loads_started: u64,
    pub loads_joined: u64,
    pub evictions: u64,
    pub failures: u64,
    pub transfers: u64,
    pub transfer_bytes: u64,
    pub exposed_wait_ns: u64,
}

#[derive(Debug, Clone)]
struct PoolState {
    stats: PoolStats,
}

#[derive(Debug, Clone)]
struct Entry {
    state: ResidencyState,
    bytes: usize,
    waiters: Vec<SessionId>,
    leases: u32,
    last_touch: u64,
}

/// Errors from invalid topology or residency operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidencyError {
    DuplicatePool(MemoryPoolId),
    DuplicateDevice(DeviceId),
    UnknownPool(MemoryPoolId),
    UnknownDevice(DeviceId),
    UnknownKey,
    UnknownLease(LeaseId),
    StaleGeneration,
    SizeMismatch {
        expected: usize,
        requested: usize,
    },
    BudgetExceeded {
        pool: MemoryPoolId,
        requested: usize,
        available: usize,
    },
}

/// A deterministic, single-owner residency manager.
///
/// Callers must confine mutation to the scheduler owner, just as they do for
/// `SchedulerCore`. Every method is synchronous and performs only bounded
/// in-memory bookkeeping.
#[derive(Debug, Clone)]
pub struct ResidencyManager {
    pools: HashMap<MemoryPoolId, PoolState>,
    devices: HashMap<DeviceId, Vec<MemoryPoolId>>,
    entries: HashMap<ExpertKey, Entry>,
    loads: HashMap<LoadId, ExpertKey>,
    leases: HashMap<LeaseId, (ExpertKey, Generation, SessionId)>,
    next_load: u64,
    next_generation: HashMap<ExpertKey, u64>,
    next_lease: u64,
    clock: u64,
}

impl ResidencyManager {
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            devices: HashMap::new(),
            entries: HashMap::new(),
            loads: HashMap::new(),
            leases: HashMap::new(),
            next_load: 0,
            next_generation: HashMap::new(),
            next_lease: 0,
            clock: 0,
        }
    }

    pub fn add_pool(
        &mut self,
        pool: MemoryPoolId,
        budget_bytes: usize,
    ) -> Result<(), ResidencyError> {
        if self.pools.contains_key(&pool) {
            return Err(ResidencyError::DuplicatePool(pool));
        }
        self.pools.insert(
            pool,
            PoolState {
                stats: PoolStats {
                    budget_bytes,
                    ..PoolStats::default()
                },
            },
        );
        Ok(())
    }

    /// Register a device's explicitly accessible pools.
    ///
    /// Several devices may list the same pool (Apple CPU/Metal/ANE UMA), and
    /// a device may list more than one pool when explicit transfer rules allow
    /// it. Device count never changes a pool's budget.
    pub fn add_device(
        &mut self,
        device: DeviceId,
        pools: Vec<MemoryPoolId>,
    ) -> Result<(), ResidencyError> {
        if self.devices.contains_key(&device) {
            return Err(ResidencyError::DuplicateDevice(device));
        }
        for pool in &pools {
            if !self.pools.contains_key(pool) {
                return Err(ResidencyError::UnknownPool(*pool));
            }
        }
        self.devices.insert(device, pools);
        Ok(())
    }

    pub fn device_pools(&self, device: DeviceId) -> Result<&[MemoryPoolId], ResidencyError> {
        self.devices
            .get(&device)
            .map(Vec::as_slice)
            .ok_or(ResidencyError::UnknownDevice(device))
    }

    pub fn pool_stats(&self, pool: MemoryPoolId) -> Result<PoolStats, ResidencyError> {
        self.pools
            .get(&pool)
            .map(|state| state.stats)
            .ok_or(ResidencyError::UnknownPool(pool))
    }

    pub fn state(&self, key: &ExpertKey) -> ResidencyState {
        self.entries
            .get(key)
            .map(|entry| entry.state.clone())
            .unwrap_or(ResidencyState::Absent)
    }

    /// Reserve bytes and start or join one physical load.
    pub fn begin_load(
        &mut self,
        key: ExpertKey,
        bytes: usize,
        waiter: SessionId,
    ) -> Result<LoadDisposition, ResidencyError> {
        let pool = key.pool;
        let current = self.entries.get(&key).map(|entry| entry.state.clone());
        match current {
            Some(ResidencyState::Resident { generation }) => {
                self.touch(&key);
                self.pool_mut(pool)?.stats.hits += 1;
                return Ok(LoadDisposition::AlreadyResident { generation });
            }
            Some(ResidencyState::Loading { load_id }) => {
                let entry = self.entries.get_mut(&key).expect("loading entry exists");
                if entry.bytes != bytes {
                    return Err(ResidencyError::SizeMismatch {
                        expected: entry.bytes,
                        requested: bytes,
                    });
                }
                if !entry.waiters.contains(&waiter) {
                    entry.waiters.push(waiter);
                }
                self.pool_mut(pool)?.stats.loads_joined += 1;
                return Ok(LoadDisposition::Joined { load_id });
            }
            Some(ResidencyState::Failed { .. }) | Some(ResidencyState::Absent) | None => {}
        }

        self.ensure_space(pool, bytes)?;
        let load_id = self.alloc_load();
        let last_touch = self.tick();
        self.entries.insert(
            key.clone(),
            Entry {
                state: ResidencyState::Loading { load_id },
                bytes,
                waiters: vec![waiter],
                leases: 0,
                last_touch,
            },
        );
        self.loads.insert(load_id, key);
        let pool_state = self.pool_mut(pool)?;
        pool_state.stats.misses += 1;
        pool_state.stats.loads_started += 1;
        pool_state.stats.reserved_loading_bytes += bytes;
        Ok(LoadDisposition::Started { load_id })
    }

    /// Complete a physical load and publish one new generation on success.
    pub fn complete_load(
        &mut self,
        completion: LoadCompletion,
    ) -> Result<CompletionDisposition, ResidencyError> {
        let Some(key) = self.loads.remove(&completion.load_id) else {
            return Ok(CompletionDisposition::Stale);
        };
        let pool = key.pool;
        let (bytes, waiters) = {
            let Some(entry) = self.entries.get_mut(&key) else {
                return Ok(CompletionDisposition::Stale);
            };
            let ResidencyState::Loading { load_id } = entry.state else {
                return Ok(CompletionDisposition::Stale);
            };
            if load_id != completion.load_id {
                return Ok(CompletionDisposition::Stale);
            }
            (entry.bytes, std::mem::take(&mut entry.waiters))
        };

        let pool_state = self.pool_mut(pool)?;
        pool_state.stats.reserved_loading_bytes = pool_state
            .stats
            .reserved_loading_bytes
            .saturating_sub(bytes);

        match completion.result {
            LoadResult::Success => {
                let generation = self.alloc_generation(&key);
                let last_touch = self.tick();
                if let Some(entry) = self.entries.get_mut(&key) {
                    entry.state = ResidencyState::Resident { generation };
                    entry.last_touch = last_touch;
                }
                let mut wakes = Vec::with_capacity(waiters.len());
                for session in waiters {
                    let lease = self.new_lease(key.clone(), generation, session);
                    wakes.push(LoadWake {
                        session,
                        result: Ok(lease),
                    });
                }
                if let Some(entry) = self.entries.get_mut(&key) {
                    entry.leases = wakes.len() as u32;
                }
                self.pool_mut(pool)?.stats.resident_bytes += bytes;
                self.refresh_pinned(pool);
                Ok(CompletionDisposition::Published { generation, wakes })
            }
            LoadResult::Failed(message) | LoadResult::Partial(message) => {
                if let Some(entry) = self.entries.get_mut(&key) {
                    entry.state = ResidencyState::Failed {
                        message: message.clone(),
                    };
                }
                self.pool_mut(pool)?.stats.failures += 1;
                let wakes = waiters
                    .into_iter()
                    .map(|session| LoadWake {
                        session,
                        result: Err(message.clone()),
                    })
                    .collect();
                Ok(CompletionDisposition::Failed { wakes })
            }
            LoadResult::Cancelled => {
                if let Some(entry) = self.entries.get_mut(&key) {
                    entry.state = ResidencyState::Failed {
                        message: "load cancelled".into(),
                    };
                }
                self.pool_mut(pool)?.stats.failures += 1;
                let wakes = waiters
                    .into_iter()
                    .map(|session| LoadWake {
                        session,
                        result: Err("load cancelled".into()),
                    })
                    .collect();
                Ok(CompletionDisposition::Failed { wakes })
            }
        }
    }

    /// Remove one waiter without cancelling an issued load.
    pub fn cancel_waiter(&mut self, load_id: LoadId, session: SessionId) -> bool {
        let Some(key) = self.loads.get(&load_id) else {
            return false;
        };
        let Some(entry) = self.entries.get_mut(key) else {
            return false;
        };
        let before = entry.waiters.len();
        entry.waiters.retain(|candidate| *candidate != session);
        before != entry.waiters.len()
    }

    /// Acquire a fresh lease for an already published generation.
    pub fn acquire(&mut self, key: &ExpertKey, owner: SessionId) -> Result<Lease, ResidencyError> {
        let generation = match self.state(key) {
            ResidencyState::Resident { generation } => generation,
            _ => return Err(ResidencyError::UnknownKey),
        };
        let lease = self.new_lease(key.clone(), generation, owner);
        let last_touch = self.tick();
        let entry = self.entries.get_mut(key).expect("resident entry exists");
        entry.leases += 1;
        entry.last_touch = last_touch;
        self.refresh_pinned(key.pool);
        Ok(lease)
    }

    /// Release exactly one lease; stale generations can never release newer data.
    pub fn release(&mut self, lease: Lease) -> Result<(), ResidencyError> {
        let Some((stored_key, stored_generation, _owner)) = self.leases.get(&lease.id) else {
            return Err(ResidencyError::UnknownLease(lease.id));
        };
        if *stored_key != lease.key || *stored_generation != lease.generation {
            return Err(ResidencyError::StaleGeneration);
        }
        let key = stored_key.clone();
        let generation = *stored_generation;
        {
            let Some(entry) = self.entries.get_mut(&key) else {
                return Err(ResidencyError::StaleGeneration);
            };
            if !matches!(entry.state, ResidencyState::Resident { generation: current } if current == generation)
            {
                return Err(ResidencyError::StaleGeneration);
            }
            entry.leases = entry.leases.saturating_sub(1);
        }
        self.leases.remove(&lease.id);
        self.refresh_pinned(key.pool);
        Ok(())
    }

    /// Evict the least-recently-used unpinned resident entry in one pool.
    pub fn evict(&mut self, pool: MemoryPoolId) -> Result<Option<ExpertKey>, ResidencyError> {
        self.pool(pool)?;
        let candidate = self
            .entries
            .iter()
            .filter(|(key, entry)| {
                key.pool == pool
                    && entry.leases == 0
                    && matches!(entry.state, ResidencyState::Resident { .. })
            })
            .min_by_key(|(_, entry)| entry.last_touch)
            .map(|(key, _)| key.clone());
        let Some(key) = candidate else {
            return Ok(None);
        };
        let bytes = self.entries.remove(&key).expect("candidate exists").bytes;
        let pool_state = self.pool_mut(pool)?;
        pool_state.stats.resident_bytes = pool_state.stats.resident_bytes.saturating_sub(bytes);
        pool_state.stats.evictions += 1;
        self.refresh_pinned(pool);
        Ok(Some(key))
    }

    /// Record backend transfer telemetry without making the transfer implicit.
    pub fn record_transfer(
        &mut self,
        pool: MemoryPoolId,
        bytes: usize,
        exposed_wait_ns: u64,
    ) -> Result<(), ResidencyError> {
        let stats = &mut self.pool_mut(pool)?.stats;
        stats.transfers += 1;
        stats.transfer_bytes += bytes as u64;
        stats.exposed_wait_ns += exposed_wait_ns;
        Ok(())
    }

    /// Assert the hard-budget and lease invariants in debug/test builds.
    pub fn assert_invariants(&self) {
        for (pool, state) in &self.pools {
            debug_assert!(
                state.stats.resident_bytes + state.stats.reserved_loading_bytes
                    <= state.stats.budget_bytes,
                "pool {pool:?} exceeds hard budget"
            );
            let computed_resident = self
                .entries
                .iter()
                .filter(|(key, entry)| {
                    key.pool == *pool && matches!(entry.state, ResidencyState::Resident { .. })
                })
                .map(|(_, entry)| entry.bytes)
                .sum::<usize>();
            debug_assert_eq!(computed_resident, state.stats.resident_bytes);
        }
        for (lease_id, (key, generation, _)) in &self.leases {
            let Some(entry) = self.entries.get(key) else {
                panic!("lease {lease_id:?} points at absent key");
            };
            debug_assert!(
                matches!(entry.state, ResidencyState::Resident { generation: current } if current == *generation)
            );
        }
    }

    fn ensure_space(&mut self, pool: MemoryPoolId, bytes: usize) -> Result<(), ResidencyError> {
        let budget = self.pool(pool)?.stats.budget_bytes;
        if bytes > budget {
            return Err(ResidencyError::BudgetExceeded {
                pool,
                requested: bytes,
                available: budget,
            });
        }
        loop {
            let stats = self.pool(pool)?.stats;
            let used = stats
                .resident_bytes
                .saturating_add(stats.reserved_loading_bytes);
            if used.saturating_add(bytes) <= budget {
                return Ok(());
            }
            if self.evict(pool)?.is_none() {
                return Err(ResidencyError::BudgetExceeded {
                    pool,
                    requested: bytes,
                    available: budget.saturating_sub(used),
                });
            }
        }
    }

    fn pool(&self, pool: MemoryPoolId) -> Result<&PoolState, ResidencyError> {
        self.pools
            .get(&pool)
            .ok_or(ResidencyError::UnknownPool(pool))
    }

    fn pool_mut(&mut self, pool: MemoryPoolId) -> Result<&mut PoolState, ResidencyError> {
        self.pools
            .get_mut(&pool)
            .ok_or(ResidencyError::UnknownPool(pool))
    }

    fn touch(&mut self, key: &ExpertKey) {
        let now = self.tick();
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_touch = now;
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }

    fn alloc_load(&mut self) -> LoadId {
        let id = LoadId(self.next_load);
        self.next_load = self.next_load.saturating_add(1);
        id
    }

    fn alloc_generation(&mut self, key: &ExpertKey) -> Generation {
        let next = self.next_generation.entry(key.clone()).or_insert(0);
        let generation = Generation(*next);
        *next = next.saturating_add(1);
        generation
    }

    fn new_lease(&mut self, key: ExpertKey, generation: Generation, owner: SessionId) -> Lease {
        let id = LeaseId(self.next_lease);
        self.next_lease = self.next_lease.saturating_add(1);
        self.leases.insert(id, (key.clone(), generation, owner));
        Lease {
            id,
            key,
            generation,
            owner,
        }
    }

    fn refresh_pinned(&mut self, pool: MemoryPoolId) {
        let pinned = self
            .entries
            .iter()
            .filter(|(key, entry)| {
                key.pool == pool
                    && entry.leases > 0
                    && matches!(entry.state, ResidencyState::Resident { .. })
            })
            .map(|(_, entry)| entry.bytes)
            .sum();
        if let Ok(pool) = self.pool_mut(pool) {
            pool.stats.pinned_bytes = pinned;
        }
    }
}

impl Default for ResidencyManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched::ActionKind;

    fn key(pool: MemoryPoolId, representation: RepresentationKey) -> ExpertKey {
        ExpertKey {
            model: 1,
            package: 2,
            layer: 3,
            expert: 4,
            representation,
            pool,
        }
    }

    fn repr(layout: u32) -> RepresentationKey {
        RepresentationKey {
            layout,
            kernel_abi: 7,
            quant: 8,
        }
    }

    #[test]
    fn coalesces_same_physical_load_and_publishes_one_generation() {
        let pool = MemoryPoolId(0);
        let mut manager = ResidencyManager::new();
        manager.add_pool(pool, 100).unwrap();
        let expert = key(pool, repr(1));
        let first = manager
            .begin_load(expert.clone(), 40, SessionId(10))
            .unwrap();
        let load = match first {
            LoadDisposition::Started { load_id } => load_id,
            other => panic!("unexpected {other:?}"),
        };
        assert_eq!(
            manager
                .begin_load(expert.clone(), 40, SessionId(11))
                .unwrap(),
            LoadDisposition::Joined { load_id: load }
        );
        assert_eq!(manager.pool_stats(pool).unwrap().reserved_loading_bytes, 40);
        let completion = manager
            .complete_load(LoadCompletion {
                load_id: load,
                result: LoadResult::Success,
            })
            .unwrap();
        let CompletionDisposition::Published { generation, wakes } = completion else {
            panic!()
        };
        assert_eq!(generation, Generation(0));
        assert_eq!(wakes.len(), 2);
        assert_eq!(manager.pool_stats(pool).unwrap().resident_bytes, 40);
        assert_eq!(manager.pool_stats(pool).unwrap().reserved_loading_bytes, 0);
        manager.assert_invariants();
    }

    #[test]
    fn representations_and_pools_do_not_collapse() {
        let mut manager = ResidencyManager::new();
        manager.add_pool(MemoryPoolId(0), 100).unwrap();
        manager.add_pool(MemoryPoolId(1), 100).unwrap();
        let a = key(MemoryPoolId(0), repr(1));
        let b = key(MemoryPoolId(0), repr(2));
        let c = key(MemoryPoolId(1), repr(1));
        let la = manager.begin_load(a, 30, SessionId(1)).unwrap();
        let lb = manager.begin_load(b, 30, SessionId(1)).unwrap();
        let lc = manager.begin_load(c, 30, SessionId(1)).unwrap();
        assert!(matches!(la, LoadDisposition::Started { .. }));
        assert!(matches!(lb, LoadDisposition::Started { .. }));
        assert!(matches!(lc, LoadDisposition::Started { .. }));
        assert_eq!(
            manager
                .pool_stats(MemoryPoolId(0))
                .unwrap()
                .reserved_loading_bytes,
            60
        );
        assert_eq!(
            manager
                .pool_stats(MemoryPoolId(1))
                .unwrap()
                .reserved_loading_bytes,
            30
        );
    }

    #[test]
    fn reservation_is_hard_backpressure_and_failure_releases_it() {
        let pool = MemoryPoolId(0);
        let mut manager = ResidencyManager::new();
        manager.add_pool(pool, 50).unwrap();
        let a = key(pool, repr(1));
        let load = match manager.begin_load(a.clone(), 40, SessionId(1)).unwrap() {
            LoadDisposition::Started { load_id } => load_id,
            _ => unreachable!(),
        };
        assert!(matches!(
            manager.begin_load(key(pool, repr(2)), 20, SessionId(2)),
            Err(ResidencyError::BudgetExceeded { .. })
        ));
        let result = manager
            .complete_load(LoadCompletion {
                load_id: load,
                result: LoadResult::Partial("short read".into()),
            })
            .unwrap();
        assert!(matches!(result, CompletionDisposition::Failed { .. }));
        assert_eq!(manager.pool_stats(pool).unwrap().reserved_loading_bytes, 0);
        assert_eq!(manager.pool_stats(pool).unwrap().resident_bytes, 0);
        assert!(matches!(manager.state(&a), ResidencyState::Failed { .. }));
        manager.assert_invariants();
    }

    #[test]
    fn waiter_cancellation_keeps_shared_load_alive() {
        let pool = MemoryPoolId(0);
        let mut manager = ResidencyManager::new();
        manager.add_pool(pool, 50).unwrap();
        let expert = key(pool, repr(1));
        let load = match manager
            .begin_load(expert.clone(), 20, SessionId(1))
            .unwrap()
        {
            LoadDisposition::Started { load_id } => load_id,
            _ => unreachable!(),
        };
        assert_eq!(
            manager
                .begin_load(expert.clone(), 20, SessionId(2))
                .unwrap(),
            LoadDisposition::Joined { load_id: load }
        );
        assert!(manager.cancel_waiter(load, SessionId(1)));
        let CompletionDisposition::Published { wakes, .. } = manager
            .complete_load(LoadCompletion {
                load_id: load,
                result: LoadResult::Success,
            })
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(wakes.len(), 1);
        assert_eq!(wakes[0].session, SessionId(2));
    }

    #[test]
    fn leases_pin_and_generation_makes_old_release_harmless() {
        let pool = MemoryPoolId(0);
        let mut manager = ResidencyManager::new();
        manager.add_pool(pool, 30).unwrap();
        let expert = key(pool, repr(1));
        let load = match manager
            .begin_load(expert.clone(), 30, SessionId(1))
            .unwrap()
        {
            LoadDisposition::Started { load_id } => load_id,
            _ => unreachable!(),
        };
        let CompletionDisposition::Published {
            wakes, generation, ..
        } = manager
            .complete_load(LoadCompletion {
                load_id: load,
                result: LoadResult::Success,
            })
            .unwrap()
        else {
            panic!()
        };
        let lease = wakes[0].result.clone().unwrap();
        assert_eq!(manager.pool_stats(pool).unwrap().pinned_bytes, 30);
        assert_eq!(manager.evict(pool).unwrap(), None);
        manager.release(lease.clone()).unwrap();
        assert_eq!(manager.pool_stats(pool).unwrap().pinned_bytes, 0);
        assert_eq!(manager.evict(pool).unwrap(), Some(expert.clone()));
        let load2 = match manager
            .begin_load(expert.clone(), 30, SessionId(2))
            .unwrap()
        {
            LoadDisposition::Started { load_id } => load_id,
            _ => unreachable!(),
        };
        let CompletionDisposition::Published {
            generation: next, ..
        } = manager
            .complete_load(LoadCompletion {
                load_id: load2,
                result: LoadResult::Success,
            })
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(generation, Generation(0));
        assert_eq!(next, Generation(1));
        assert_eq!(
            manager.release(lease),
            Err(ResidencyError::UnknownLease(LeaseId(0)))
        );
        manager.assert_invariants();
    }

    #[test]
    fn cancelled_load_publishes_failure_and_retry_starts_fresh() {
        let pool = MemoryPoolId(0);
        let mut manager = ResidencyManager::new();
        manager.add_pool(pool, 50).unwrap();
        let expert = key(pool, repr(1));
        let load = match manager
            .begin_load(expert.clone(), 20, SessionId(1))
            .unwrap()
        {
            LoadDisposition::Started { load_id } => load_id,
            _ => unreachable!(),
        };
        manager
            .begin_load(expert.clone(), 20, SessionId(2))
            .unwrap();
        let disposition = manager
            .complete_load(LoadCompletion {
                load_id: load,
                result: LoadResult::Cancelled,
            })
            .unwrap();
        assert!(
            matches!(&disposition, CompletionDisposition::Failed { wakes } if wakes.len() == 2)
        );
        assert!(
            matches!(disposition, CompletionDisposition::Failed { wakes } if wakes.iter().all(|wake| wake.result.is_err()))
        );
        assert!(matches!(
            manager.state(&expert),
            ResidencyState::Failed { .. }
        ));
        assert_eq!(manager.pool_stats(pool).unwrap().reserved_loading_bytes, 0);
        assert_eq!(manager.pool_stats(pool).unwrap().resident_bytes, 0);
        assert_eq!(manager.pool_stats(pool).unwrap().failures, 1);
        // A failed entry is retryable: the next miss starts a fresh load.
        let retry = manager
            .begin_load(expert.clone(), 20, SessionId(3))
            .unwrap();
        assert!(matches!(retry, LoadDisposition::Started { .. }));
        assert!(matches!(
            manager.state(&expert),
            ResidencyState::Loading { .. }
        ));
        let load = match retry {
            LoadDisposition::Started { load_id } => load_id,
            _ => unreachable!(),
        };
        manager
            .complete_load(LoadCompletion {
                load_id: load,
                result: LoadResult::Success,
            })
            .unwrap();
        assert!(matches!(
            manager.state(&expert),
            ResidencyState::Resident { .. }
        ));
        manager.assert_invariants();
    }

    #[test]
    fn fake_apple_uma_counts_one_budget_for_three_devices() {
        let pool = MemoryPoolId(7);
        let mut manager = ResidencyManager::new();
        manager.add_pool(pool, 100).unwrap();
        manager.add_device(DeviceId(0), vec![pool]).unwrap(); // CPU
        manager.add_device(DeviceId(1), vec![pool]).unwrap(); // Metal
        manager.add_device(DeviceId(2), vec![pool]).unwrap(); // ANE
        let expert = key(pool, repr(1));
        let load = match manager.begin_load(expert, 90, SessionId(1)).unwrap() {
            LoadDisposition::Started { load_id } => load_id,
            _ => unreachable!(),
        };
        manager
            .complete_load(LoadCompletion {
                load_id: load,
                result: LoadResult::Success,
            })
            .unwrap();
        assert_eq!(manager.pool_stats(pool).unwrap().resident_bytes, 90);
        assert!(matches!(
            manager.begin_load(key(pool, repr(2)), 20, SessionId(2)),
            Err(ResidencyError::BudgetExceeded { .. })
        ));
        assert_eq!(manager.device_pools(DeviceId(0)).unwrap(), &[pool]);
        assert_eq!(manager.device_pools(DeviceId(1)).unwrap(), &[pool]);
        assert_eq!(manager.device_pools(DeviceId(2)).unwrap(), &[pool]);
    }

    #[test]
    fn fake_discrete_host_and_device_have_independent_budgets() {
        let host = MemoryPoolId(0);
        let gpu = MemoryPoolId(1);
        let mut manager = ResidencyManager::new();
        manager.add_pool(host, 50).unwrap();
        manager.add_pool(gpu, 50).unwrap();
        manager.add_device(DeviceId(0), vec![host, gpu]).unwrap();
        let host_key = key(host, repr(1));
        let gpu_key = key(gpu, repr(1));
        let host_load = match manager.begin_load(host_key, 40, SessionId(1)).unwrap() {
            LoadDisposition::Started { load_id } => load_id,
            _ => unreachable!(),
        };
        let gpu_load = match manager.begin_load(gpu_key, 40, SessionId(1)).unwrap() {
            LoadDisposition::Started { load_id } => load_id,
            _ => unreachable!(),
        };
        manager
            .complete_load(LoadCompletion {
                load_id: host_load,
                result: LoadResult::Success,
            })
            .unwrap();
        manager
            .complete_load(LoadCompletion {
                load_id: gpu_load,
                result: LoadResult::Success,
            })
            .unwrap();
        assert_eq!(manager.pool_stats(host).unwrap().resident_bytes, 40);
        assert_eq!(manager.pool_stats(gpu).unwrap().resident_bytes, 40);
    }

    #[test]
    fn thousands_of_evictions_never_reuse_a_generation() {
        let pool = MemoryPoolId(0);
        let mut manager = ResidencyManager::new();
        manager.add_pool(pool, 2).unwrap();
        let expert = key(pool, repr(1));
        let mut generations = Vec::new();
        for i in 0..1_000u64 {
            let load = match manager.begin_load(expert.clone(), 2, SessionId(i)).unwrap() {
                LoadDisposition::Started { load_id } => load_id,
                other => panic!("unexpected {other:?}"),
            };
            let CompletionDisposition::Published {
                generation, wakes, ..
            } = manager
                .complete_load(LoadCompletion {
                    load_id: load,
                    result: LoadResult::Success,
                })
                .unwrap()
            else {
                panic!()
            };
            generations.push(generation);
            for wake in wakes {
                if let Ok(lease) = wake.result {
                    manager.release(lease).unwrap();
                }
            }
            assert_eq!(manager.evict(pool).unwrap(), Some(expert.clone()));
        }
        assert_eq!(generations.len(), 1_000);
        assert!(generations.windows(2).all(|pair| pair[0].0 < pair[1].0));
    }

    #[test]
    fn telemetry_records_transfers_and_exposed_wait() {
        let mut manager = ResidencyManager::new();
        manager.add_pool(MemoryPoolId(0), 1).unwrap();
        manager.record_transfer(MemoryPoolId(0), 4096, 123).unwrap();
        let stats = manager.pool_stats(MemoryPoolId(0)).unwrap();
        assert_eq!(stats.transfers, 1);
        assert_eq!(stats.transfer_bytes, 4096);
        assert_eq!(stats.exposed_wait_ns, 123);
        let _ = ActionKind::Io;
    }
}
