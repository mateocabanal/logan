//! Dense Llama/MiniCPM causal-state adapter for logan-core.

use crate::DenseSession;
use logan_core::state::{CausalState, CausalStateCodec, StateSchemaId};

#[derive(Debug, Clone, Copy, Default)]
pub struct LlamaStateCodec;

impl LlamaStateCodec {
    pub fn schema_id() -> StateSchemaId {
        StateSchemaId::new("llama-dense-kv", 1, 0)
    }
}

impl CausalStateCodec for LlamaStateCodec {
    type EngineState = DenseSession;

    fn schema_id(&self) -> StateSchemaId {
        Self::schema_id()
    }

    fn export_state(
        &self,
        session: &Self::EngineState,
        prefix_len: usize,
    ) -> Result<CausalState, String> {
        if session.kv().processed_tokens() != prefix_len {
            return Err(format!(
                "dense session has {} committed tokens, snapshot requested at {prefix_len}",
                session.kv().processed_tokens()
            ));
        }
        Ok(CausalState::Opaque(session.export_prefix_state()))
    }

    fn import_state(
        &self,
        session: &mut Self::EngineState,
        prefix_len: usize,
        causal: &CausalState,
    ) -> Result<(), String> {
        let CausalState::Opaque(bytes) = causal else {
            return Err("Llama dense state codec requires an opaque KV payload".into());
        };
        session.import_prefix_state(bytes)?;
        if session.kv().processed_tokens() != prefix_len {
            return Err(format!(
                "restored dense session has {} tokens, expected {prefix_len}",
                session.kv().processed_tokens()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_identity_is_stable() {
        assert_eq!(
            LlamaStateCodec::schema_id(),
            StateSchemaId::new("llama-dense-kv", 1, 0)
        );
    }
}
