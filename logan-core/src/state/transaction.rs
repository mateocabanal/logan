//! Generation-safe causal-state transactions.
//!
//! A transaction checkpoint belongs to exactly one manager/session. Rollback
//! restores the captured state; commit accepts the caller's live state and
//! invalidates older checkpoints.

use super::{CausalState, StateSchemaId};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Generation(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransactionCheckpoint {
    pub manager_id: u64,
    pub generation: Generation,
    pub transaction_id: u64,
    pub schema_id: StateSchemaId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionResult {
    Committed,
    RolledBack,
    StaleCheckpoint,
    WrongSession,
    Cancelled,
}

#[derive(Debug, Clone)]
struct TransactionInfo {
    checkpoint: TransactionCheckpoint,
    state: CausalState,
}

pub struct TransactionManager {
    manager_id: u64,
    schema_id: StateSchemaId,
    current_generation: Generation,
    next_transaction_id: u64,
    active: Vec<TransactionInfo>,
}

static NEXT_MANAGER_ID: AtomicU64 = AtomicU64::new(1);

impl TransactionManager {
    pub fn new(schema_id: StateSchemaId) -> Self {
        Self {
            manager_id: NEXT_MANAGER_ID.fetch_add(1, Ordering::Relaxed).max(1),
            schema_id,
            current_generation: Generation(0),
            next_transaction_id: 1,
            active: Vec::new(),
        }
    }

    pub fn begin(&mut self, current_state: &CausalState) -> TransactionCheckpoint {
        let checkpoint = TransactionCheckpoint {
            manager_id: self.manager_id,
            generation: self.current_generation,
            transaction_id: self.next_transaction_id,
            schema_id: self.schema_id.clone(),
        };
        self.next_transaction_id = self.next_transaction_id.wrapping_add(1).max(1);
        self.active.push(TransactionInfo {
            checkpoint: checkpoint.clone(),
            state: current_state.clone(),
        });
        checkpoint
    }

    /// Accept the caller's live state. Advancing the generation invalidates
    /// every checkpoint captured from the previous committed state.
    pub fn commit(&mut self, checkpoint: &TransactionCheckpoint) -> TransactionResult {
        match self.validate(checkpoint) {
            Ok(index) => {
                self.active.remove(index);
                self.advance_generation();
                TransactionResult::Committed
            }
            Err(result) => result,
        }
    }

    /// Restore the exact state captured by begin.
    pub fn rollback(
        &mut self,
        checkpoint: &TransactionCheckpoint,
        current_state: &mut CausalState,
    ) -> TransactionResult {
        match self.validate(checkpoint) {
            Ok(index) => {
                *current_state = self.active[index].state.clone();
                self.active.remove(index);
                self.advance_generation();
                TransactionResult::RolledBack
            }
            Err(result) => result,
        }
    }

    /// Restore a checkpoint and retain that verified prefix as the live state.
    pub fn retain_prefix(
        &mut self,
        checkpoint: &TransactionCheckpoint,
        current_state: &mut CausalState,
    ) -> TransactionResult {
        self.rollback(checkpoint, current_state)
    }

    pub fn cancel(&mut self) -> TransactionResult {
        self.active.clear();
        self.current_generation.0 = self.current_generation.0.wrapping_add(1);
        TransactionResult::Cancelled
    }

    pub fn current_generation(&self) -> Generation {
        self.current_generation
    }

    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    pub fn is_valid_checkpoint(&self, checkpoint: &TransactionCheckpoint) -> bool {
        self.validate(checkpoint).is_ok()
    }

    fn validate(&self, checkpoint: &TransactionCheckpoint) -> Result<usize, TransactionResult> {
        if checkpoint.manager_id != self.manager_id || checkpoint.schema_id != self.schema_id {
            return Err(TransactionResult::WrongSession);
        }
        if checkpoint.generation != self.current_generation {
            return Err(TransactionResult::StaleCheckpoint);
        }
        self.active
            .iter()
            .position(|info| info.checkpoint == *checkpoint)
            .ok_or(TransactionResult::StaleCheckpoint)
    }

    fn advance_generation(&mut self) {
        self.current_generation.0 = self.current_generation.0.wrapping_add(1);
        // Any concurrently captured checkpoint refers to the previous state and
        // must not become accidentally valid after a commit/rollback.
        self.active.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> StateSchemaId {
        StateSchemaId::new("test", 1, 0)
    }

    #[test]
    fn rollback_restores_exact_state() {
        let mut manager = TransactionManager::new(schema());
        let mut state = CausalState::Opaque(vec![1, 2, 3]);
        let checkpoint = manager.begin(&state);
        state = CausalState::Opaque(vec![9, 9]);
        assert_eq!(
            manager.rollback(&checkpoint, &mut state),
            TransactionResult::RolledBack
        );
        assert_eq!(state, CausalState::Opaque(vec![1, 2, 3]));
        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn commit_invalidates_other_same_generation_checkpoints() {
        let mut manager = TransactionManager::new(schema());
        let state = CausalState::Opaque(vec![1]);
        let first = manager.begin(&state);
        let second = manager.begin(&state);
        assert_eq!(manager.commit(&first), TransactionResult::Committed);
        assert_eq!(manager.commit(&second), TransactionResult::StaleCheckpoint);
    }

    #[test]
    fn checkpoints_cannot_cross_sessions() {
        let mut a = TransactionManager::new(schema());
        let mut b = TransactionManager::new(schema());
        let state = CausalState::Opaque(vec![1]);
        let checkpoint = a.begin(&state);
        assert_eq!(b.commit(&checkpoint), TransactionResult::WrongSession);
    }

    #[test]
    fn cancellation_invalidates_checkpoint() {
        let mut manager = TransactionManager::new(schema());
        let state = CausalState::Opaque(vec![]);
        let checkpoint = manager.begin(&state);
        assert_eq!(manager.cancel(), TransactionResult::Cancelled);
        assert!(!manager.is_valid_checkpoint(&checkpoint));
    }
}
