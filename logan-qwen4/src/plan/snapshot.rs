//! Qwen4 adapter onto the model-neutral logan-core state snapshot.

use crate::Model;
use logan_core::prefix::hash_tokens;
use logan_core::state::{CausalState, CausalStateCodec, StateSchemaId, StateSnapshot};

#[derive(Debug, Clone, Copy, Default)]
pub struct QwenStateCodec;

impl QwenStateCodec {
    pub fn schema_id() -> StateSchemaId {
        StateSchemaId::new("qwen4-causal", 1, 0)
    }
}

impl CausalStateCodec for QwenStateCodec {
    type EngineState = Model;

    fn schema_id(&self) -> StateSchemaId {
        Self::schema_id()
    }

    fn export_state(
        &self,
        model: &Self::EngineState,
        prefix_len: usize,
    ) -> Result<CausalState, String> {
        let snapshot = model.snapshot_state(prefix_len)?;
        Ok(CausalState::Opaque(snapshot.payload))
    }

    fn import_state(
        &self,
        model: &mut Self::EngineState,
        prefix_len: usize,
        causal: &CausalState,
    ) -> Result<(), String> {
        let CausalState::Opaque(payload) = causal else {
            return Err("Qwen4 state codec requires an opaque causal payload".into());
        };
        crate::plan::prefix_cache::restore_payload_from_bytes(model, prefix_len, payload)
    }
}

/// Generic, checksummed Qwen causal-state checkpoint.
///
/// The cache policy remains outside the engine; this type only binds Qwen's
/// exact state semantics to logan-core's versioned snapshot container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenStateSnapshot {
    inner: StateSnapshot,
}

impl QwenStateSnapshot {
    pub fn capture(model: &Model, prefix_len: usize, prefix_hash: u64) -> Result<Self, String> {
        Ok(Self {
            inner: StateSnapshot::capture(&QwenStateCodec, model, prefix_len, prefix_hash)?,
        })
    }

    pub fn capture_tokens(model: &Model, tokens: &[u32]) -> Result<Self, String> {
        Self::capture(model, tokens.len(), hash_tokens(tokens))
    }

    pub fn restore_to(&self, model: &mut Model, expected_prefix_hash: u64) -> Result<(), String> {
        self.inner
            .restore_with(&QwenStateCodec, model, expected_prefix_hash)
    }

    pub fn restore_tokens(&self, model: &mut Model, tokens: &[u32]) -> Result<(), String> {
        if tokens.len() != self.prefix_len() {
            return Err(format!(
                "Qwen snapshot has {} tokens, restore key has {}",
                self.prefix_len(),
                tokens.len()
            ));
        }
        self.restore_to(model, hash_tokens(tokens))
    }

    pub fn prefix_len(&self) -> usize {
        self.inner.prefix_len
    }

    pub fn payload_bytes(&self) -> usize {
        self.inner.payload_bytes()
    }

    pub fn exact_eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    pub fn inner(&self) -> &StateSnapshot {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_identity_is_stable() {
        assert_eq!(
            QwenStateCodec::schema_id(),
            StateSchemaId::new("qwen4-causal", 1, 0)
        );
    }
}
