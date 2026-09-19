//! Same-thread ANE ownership and plain-data completion lifecycle.
//!
//! The state machine is independent of Objective-C callbacks so fake devices
//! can exercise timeout, cancellation, and delayed completion behavior. A
//! production adapter drives it from the owner thread that also owns
//! `AneProgramCache`, `AneModel`, `AneSurface`, and `AneAsyncChannel`.

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::rc::Rc;

use logan_ane::{AneProgramCache, AneRuntime};

use super::program::AneProgramIdentity;

pub type OperationId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AneTicket {
    pub operation_id: OperationId,
    pub slot: u32,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScratchDisposition {
    Shared,
    Fresh,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AneOperationSpec {
    pub operation_id: OperationId,
    pub program: AneProgramIdentity,
    pub bytes: usize,
    pub scratch_bytes: usize,
    pub submitted_at_ms: u64,
    pub timeout_ms: u64,
    pub allow_fresh_scratch: bool,
}

impl AneOperationSpec {
    pub fn deadline_ms(&self) -> Option<u64> {
        self.submitted_at_ms.checked_add(self.timeout_ms)
    }

    fn total_bytes(&self) -> Option<usize> {
        self.bytes.checked_add(self.scratch_bytes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionStatus {
    Succeeded,
    Failed,
    Cancelled,
    TimedOut,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionDisposition {
    Completed(CompletionStatus),
    DuplicateIgnored,
    LateIgnored,
    UnknownTicket,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpiredOperation {
    pub ticket: AneTicket,
    pub status: CompletionStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmitError {
    InvalidOperationId,
    InvalidDeadline,
    InvalidBytes,
    CapacityExceeded,
    NoFreeSlot,
    FreshScratchRequired,
}

struct Slot {
    generation: u64,
    ticket: Option<AneTicket>,
    deadline_ms: u64,
    release_at_ms: u64,
    bytes: usize,
    scratch_reusable: bool,
    quarantined: bool,
}

impl Slot {
    fn free() -> Self {
        Self {
            generation: 0,
            ticket: None,
            deadline_ms: 0,
            release_at_ms: 0,
            bytes: 0,
            scratch_reusable: false,
            quarantined: false,
        }
    }

    fn is_free(&self) -> bool {
        self.ticket.is_none() && !self.quarantined
    }
}

/// Owner-thread lifecycle for bounded ANE dispatches.
///
/// This type is deliberately `!Send`/`!Sync`, matching the native wrappers it
/// is intended to sit beside. Every method is synchronous and plain-data;
/// callers submit a ticket to their native callback bridge and feed the
/// callback back through `complete` on this same owner thread.
pub struct AneExecutor {
    slots: Vec<Slot>,
    max_bytes: usize,
    used_bytes: usize,
    quarantine_ms: u64,
    retired: VecDeque<AneTicket>,
    retired_limit: usize,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl AneExecutor {
    pub fn new(max_slots: usize, max_bytes: usize, quarantine_ms: u64) -> Self {
        Self {
            slots: (0..max_slots).map(|_| Slot::free()).collect(),
            max_bytes,
            used_bytes: 0,
            quarantine_ms,
            retired: VecDeque::new(),
            retired_limit: max_slots.saturating_mul(2).max(1),
            _not_send_sync: PhantomData,
        }
    }
    fn remember_retired(&mut self, ticket: AneTicket) {
        self.retired.push_back(ticket);
        while self.retired.len() > self.retired_limit {
            self.retired.pop_front();
        }
    }

    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }
    pub fn active_slots(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.ticket.is_some() && !slot.quarantined)
            .count()
    }
    pub fn quarantined_slots(&self) -> usize {
        self.slots.iter().filter(|slot| slot.quarantined).count()
    }

    pub fn submit(
        &mut self,
        spec: AneOperationSpec,
    ) -> Result<(AneTicket, ScratchDisposition), SubmitError> {
        if spec.operation_id == 0 {
            return Err(SubmitError::InvalidOperationId);
        }
        let deadline_ms = spec.deadline_ms().ok_or(SubmitError::InvalidDeadline)?;
        let total_bytes = spec.total_bytes().ok_or(SubmitError::InvalidBytes)?;
        if total_bytes == 0 {
            return Err(SubmitError::InvalidBytes);
        }
        if total_bytes > self.max_bytes || self.used_bytes > self.max_bytes - total_bytes {
            return Err(SubmitError::CapacityExceeded);
        }

        let Some((slot_index, slot)) = self
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, slot)| slot.is_free())
        else {
            return Err(SubmitError::NoFreeSlot);
        };
        let scratch = if slot.scratch_reusable {
            ScratchDisposition::Shared
        } else if spec.allow_fresh_scratch {
            ScratchDisposition::Fresh
        } else {
            return Err(SubmitError::FreshScratchRequired);
        };
        slot.generation = slot.generation.wrapping_add(1).max(1);
        let ticket = AneTicket {
            operation_id: spec.operation_id,
            slot: slot_index as u32,
            generation: slot.generation,
        };
        slot.ticket = Some(ticket);
        slot.deadline_ms = deadline_ms;
        slot.release_at_ms = 0;
        slot.bytes = total_bytes;
        slot.quarantined = false;
        slot.scratch_reusable = false;
        self.used_bytes += total_bytes;
        Ok((ticket, scratch))
    }

    pub fn complete(
        &mut self,
        ticket: AneTicket,
        status: CompletionStatus,
    ) -> CompletionDisposition {
        let slot_index = ticket.slot as usize;
        let Some(slot) = self.slots.get(slot_index) else {
            return CompletionDisposition::UnknownTicket;
        };
        if slot.generation != ticket.generation || slot.ticket != Some(ticket) {
            return if self.retired.contains(&ticket) {
                CompletionDisposition::DuplicateIgnored
            } else {
                CompletionDisposition::LateIgnored
            };
        }
        let (bytes, quarantined) = {
            let slot = &self.slots[slot_index];
            (slot.bytes, slot.quarantined)
        };
        self.used_bytes = self.used_bytes.saturating_sub(bytes);
        let slot = &mut self.slots[slot_index];
        slot.ticket = None;
        slot.bytes = 0;
        slot.deadline_ms = 0;
        slot.quarantined = false;
        slot.scratch_reusable = !quarantined;
        if !quarantined {
            self.remember_retired(ticket);
        }
        if quarantined {
            CompletionDisposition::LateIgnored
        } else {
            CompletionDisposition::Completed(status)
        }
    }

    pub fn cancel(&mut self, ticket: AneTicket, now_ms: u64) -> CompletionDisposition {
        let Some(slot) = self.slots.get_mut(ticket.slot as usize) else {
            return CompletionDisposition::UnknownTicket;
        };
        if slot.generation != ticket.generation || slot.ticket != Some(ticket) {
            return if self.retired.contains(&ticket) {
                CompletionDisposition::DuplicateIgnored
            } else {
                CompletionDisposition::LateIgnored
            };
        }
        if slot.quarantined {
            return CompletionDisposition::LateIgnored;
        }
        slot.quarantined = true;
        slot.release_at_ms = now_ms.saturating_add(self.quarantine_ms);
        slot.scratch_reusable = false;
        CompletionDisposition::Completed(CompletionStatus::Cancelled)
    }

    pub fn expire(&mut self, now_ms: u64) -> Vec<ExpiredOperation> {
        let mut expired = Vec::new();
        for slot in &mut self.slots {
            if slot.quarantined || slot.ticket.is_none() || now_ms < slot.deadline_ms {
                continue;
            }
            let ticket = slot.ticket.expect("checked above");
            slot.quarantined = true;
            slot.release_at_ms = now_ms.saturating_add(self.quarantine_ms);
            slot.scratch_reusable = false;
            expired.push(ExpiredOperation {
                ticket,
                status: CompletionStatus::TimedOut,
            });
        }
        expired
    }

    /// Release quarantine slots after their grace period. A late completion
    /// after this point is harmless because its generation no longer matches.
    pub fn reap_quarantine(&mut self, now_ms: u64) -> usize {
        let mut reaped = 0;
        for slot in &mut self.slots {
            if !slot.quarantined || now_ms < slot.release_at_ms {
                continue;
            }
            self.used_bytes = self.used_bytes.saturating_sub(slot.bytes);
            slot.ticket = None;
            slot.bytes = 0;
            slot.quarantined = false;
            slot.scratch_reusable = false;
            reaped += 1;
        }
        reaped
    }
}
/// Native ANE cache owner. The cache and all model/channel/surface slots
/// remain behind this owner-thread boundary; only identities and tickets leave
/// it.
pub struct AneNativeOwner {
    cache: AneProgramCache,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    channels: Vec<logan_ane::AneAsyncChannel>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl AneNativeOwner {
    pub fn new(runtime: AneRuntime) -> Self {
        Self {
            cache: AneProgramCache::new(runtime),
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            channels: Vec::new(),
            _not_send_sync: PhantomData,
        }
    }

    pub fn cache(&self) -> &AneProgramCache {
        &self.cache
    }
    pub fn cache_mut(&mut self) -> &mut AneProgramCache {
        &mut self.cache
    }

    /// Store a channel only on this owner thread. The channel remains
    /// intentionally unavailable on non-macOS builds where the native type is
    /// not exported.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn retain_channel(&mut self, channel: logan_ane::AneAsyncChannel) {
        self.channels.push(channel);
    }
}
