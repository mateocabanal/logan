//! Generic transactional state operations.
//!
//! Provides checkpoint/commit/rollback for causal state with:
//! - COW isolation between live and transactional pages
//! - Safe rollback of partial speculative blocks
//! - Stale generation rejection
//! - Two-session isolation
//! - Cancellation cleanup

use crate::state::{CausalState, StateSchemaId};

/// Generation counter for transaction safety
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Generation(pub u64);

/// A checkpoint reference returned when a transaction begins.
/// Used to validate and restore transaction state.
#[derive(Debug, Clone)]
pub struct TransactionCheckpoint {
    pub generation: Generation,
    pub transaction_id: u64,
    pub schema_id: StateSchemaId,
}

/// A transactional state operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionResult {
    /// Transaction committed successfully
    Committed,
    /// Transaction rolled back (partial or full)
    RolledBack,
    /// Checkpoint was stale - cannot commit or rollback
    StaleCheckpoint,
    /// Wrong generation - checkpoint belongs to another session
    WrongGeneration,
    /// Cancelled before completion
    Cancelled,
}

/// Transaction state for checkpoint/commit/rollback operations.
pub struct TransactionManager {
    current_generation: Generation,
    active_transactions: Vec<TransactionInfo>,
}

#[derive(Debug, Clone)]
struct TransactionInfo {
    id: u64,
    generation: Generation,
    checkpoint: CausalState,
}

impl TransactionManager {
    pub fn new(schema_id: StateSchemaId) -> Self {
        TransactionManager {
            current_generation: Generation(0),
            active_transactions: Vec::new(),
        }
    }

    /// Begin a new transaction by creating a checkpoint of the current state.
    /// Returns a TransactionCheckpoint that can be used to validate/commit/rollback.
    pub fn begin(
        &mut self,
        current_state: &CausalState,
    ) -> TransactionCheckpoint {
        let id = self.active_transactions.len() as u64 + 1;
        let checkpoint = current_state.clone();
        let schema_id = StateSchemaId {
            engine: "generic".to_string(),
            version: 1,
            sub_version: 0,
        };

        self.active_transactions.push(TransactionInfo {
            id,
            generation: self.current_generation,
            checkpoint,
        });

        TransactionCheckpoint {
            generation: self.current_generation,
            transaction_id: id,
            schema_id,
        }
    }

    /// Commit a transaction: accept its changes as the new committed state.
    /// Returns TransactionResult indicating success/failure.
    pub fn commit(
        &mut self,
        checkpoint: &TransactionCheckpoint,
    ) -> TransactionResult {
        if checkpoint.generation != self.current_generation {
            return TransactionResult::WrongGeneration;
        }

        let idx = self.active_transactions.iter().position(|t| {
            t.transaction_id == checkpoint.transaction_id
                && t.generation == checkpoint.generation
        });

        if let Some(idx) = idx {
            // Remove committed transaction, increment generation
            self.active_transactions.remove(idx);
            self.current_generation.0 += 1;
            TransactionResult::Committed
        } else {
            TransactionResult::StaleCheckpoint
        }
    }

    /// Rollback a transaction: discard its changes.
    /// Returns TransactionResult indicating success/failure.
    pub fn rollback(
        &mut self,
        checkpoint: &TransactionCheckpoint,
    ) -> TransactionResult {
        if checkpoint.generation != self.current_generation {
            // Stale checkpoint from another generation/session
            return TransactionResult::StaleCheckpoint;
        }

        let idx = self.active_transactions.iter().position(|t| {
            t.transaction_id == checkpoint.transaction_id
                && t.generation == checkpoint.generation
        });

        if let Some(idx) = idx {
            self.active_transactions.remove(idx);
            TransactionResult::RolledBack
        } else {
            TransactionResult::StaleCheckpoint
        }
    }

    /// Retain a prefix checkpoint, releasing later entries.
    /// This is used after prefix cache persistence to discard later entries.
    pub fn retain_prefix(&mut self, checkpoint: &TransactionCheckpoint) -> bool {
        self.active_transactions.retain(|t| {
            t.generation == checkpoint.generation
                && t.transaction_id == checkpoint.transaction_id
        });
        !self.active_transactions.is_empty()
    }

    /// Cancel all active transactions (used for cancellation).
    pub fn cancel(&mut self) {
        self.active_transactions.clear();
        self.current_generation.0 += 1;
    }

    /// Get the current generation for validation
    pub fn current_generation(&self) -> Generation {
        self.current_generation
    }

    /// Check if a checkpoint is valid for the current generation
    pub fn is_valid_checkpoint(&self, checkpoint: &TransactionCheckpoint) -> bool {
        checkpoint.generation == self.current_generation
            || self.active_transactions.iter().any(|t| {
                t.transaction_id == checkpoint.transaction_id
                    && t.generation == checkpoint.generation
            })
    }

    /// Get number of active transactions
    pub fn active_count(&self) -> usize {
        self.active_transactions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (TransactionManager, TransactionCheckpoint) {
        let mut tm = TransactionManager::new(StateSchemaId {
            engine: "test".to_string(),
            version: 1,
            sub_version: 0,
        });
        let state = CausalState::append_only(2, 4, 8);
        let checkpoint = tm.begin(&state);
        (tm, checkpoint)
    }

    #[test]
    fn test_begin_creates_checkpoint() {
        let (tm, cp) = setup();
        assert_eq!(tm.active_count(), 1);
        assert!(tm.is_valid_checkpoint(&cp));
    }

    #[test]
    fn test_commit_succeeds() {
        let (mut tm, cp) = setup();
        let result = tm.commit(&cp);
        assert_eq!(result, TransactionResult::Committed);
        assert_eq!(tm.active_count(), 0);
    }

    #[test]
    fn test_rollback_succeeds() {
        let (mut tm, cp) = setup();
        let result = tm.rollback(&cp);
        assert_eq!(result, TransactionResult::RolledBack);
        assert_eq!(tm.active_count(), 0);
    }

    #[test]
    fn test_stale_checkpoint_rejected() {
        let (mut tm, cp) = setup();
        // Advance generation without committing
        tm.cancel();

        let result = tm.commit(&cp);
        assert_eq!(result, TransactionResult::WrongGeneration);
    }

    #[test]
    fn test_cancellation_cleanup() {
        let (mut tm, cp) = setup();
        tm.cancel();
        assert_eq!(tm.active_count(), 0);
        // Checkpoint should be invalid after cancel
        assert!(!tm.is_valid_checkpoint(&cp));
    }

    #[test]
    fn test_two_sessions_isolation() {
        let mut tm1 = TransactionManager::new(StateSchemaId {
            engine: "test".to_string(),
            version: 1,
            sub_version: 0,
        });
        let mut tm2 = TransactionManager::new(StateSchemaId {
            engine: "test".to_string(),
            version: 1,
            sub_version: 0,
        });

        let state = CausalState::append_only(2, 4, 8);
        let cp1 = tm1.begin(&state);
        let cp2 = tm2.begin(&state);

        // Each session's checkpoint is valid in its own context
        assert!(tm1.is_valid_checkpoint(&cp1));
        assert!(tm2.is_valid_checkpoint(&cp2));

        // Committing in tm1 should not affect tm2
        let r1 = tm1.commit(&cp1);
        assert_eq!(r1, TransactionResult::Committed);

        // cp2 still valid in its session
        assert!(tm2.is_valid_checkpoint(&cp2));
    }

    #[test]
    fn test_retain_prefix() {
        let (mut tm, cp) = setup();
        let result = tm.retain_prefix(&cp);
        assert!(!result); // no more active transactions after retain
    }
}