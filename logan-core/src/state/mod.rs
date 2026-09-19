//! Generic model-neutral causal-state subsystem.
//!
//! Core owns the lifecycle and persistence mechanics. Each engine supplies a
//! small codec that maps its live causal state into this neutral representation
//! and back again. Engine math never leaks into the cache/persistence layer.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Model-neutral causal state.
#[derive(Debug, Clone, PartialEq)]
pub enum CausalState {
    /// Append-only KV-like state.
    AppendOnly(AppendOnlyState),
    /// Circular/recurrent state.
    Ring(RingState),
    /// Fixed-size mutable state.
    MutableFixed(MutableFixedState),
    /// Sparse paged state.
    SparsePaged(SparsePagedState),
    /// Engine-defined bytes. This is the escape hatch for engines whose state
    /// has several coupled regions that must be restored atomically.
    Opaque(Vec<u8>),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AppendOnlyState {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub position: usize,
    pub n_heads: usize,
    pub dim: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RingState {
    pub buffer: Vec<f32>,
    pub capacity: usize,
    pub write_pos: usize,
    pub read_pos: usize,
    pub len: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct MutableFixedState {
    pub data: Vec<f32>,
    pub write_pos: usize,
    pub len: usize,
    pub has_wrapped: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SparsePagedState {
    pub page_table: Vec<Option<u64>>,
    pub pages: Vec<Vec<f32>>,
    pub page_entries: usize,
    pub max_pages: usize,
    pub pinned: bool,
}

impl CausalState {
    pub fn append_only(n_heads: usize, dim: usize, max_seq: usize) -> Self {
        let size = n_heads.saturating_mul(max_seq).saturating_mul(dim);
        Self::AppendOnly(AppendOnlyState {
            keys: vec![0.0; size],
            values: vec![0.0; size],
            position: 0,
            n_heads,
            dim,
        })
    }

    pub fn ring(capacity: usize, width: usize) -> Self {
        Self::Ring(RingState {
            buffer: vec![0.0; capacity.saturating_mul(width)],
            capacity,
            write_pos: 0,
            read_pos: 0,
            len: 0,
        })
    }

    pub fn mutable_fixed(capacity: usize) -> Self {
        Self::MutableFixed(MutableFixedState {
            data: vec![0.0; capacity],
            write_pos: 0,
            len: 0,
            has_wrapped: false,
        })
    }

    pub fn sparse_paged(max_pages: usize, page_entries: usize) -> Self {
        Self::SparsePaged(SparsePagedState {
            page_table: vec![None; max_pages],
            pages: vec![vec![0.0; page_entries]; max_pages],
            page_entries,
            max_pages,
            pinned: false,
        })
    }

    pub fn opaque(data: Vec<u8>) -> Self {
        Self::Opaque(data)
    }

    pub fn position(&self) -> Option<usize> {
        match self {
            Self::AppendOnly(s) => Some(s.position),
            Self::Ring(s) => Some(s.read_pos),
            Self::MutableFixed(s) => Some(s.write_pos),
            Self::SparsePaged(_) | Self::Opaque(_) => None,
        }
    }

    pub fn len(&self) -> Option<usize> {
        match self {
            Self::AppendOnly(s) => Some(s.position),
            Self::Ring(s) => Some(s.len),
            Self::MutableFixed(s) => Some(s.len),
            Self::SparsePaged(s) => Some(s.page_table.iter().filter(|p| p.is_some()).count()),
            Self::Opaque(_) => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }
}

/// Versioned identity for one engine's causal-state ABI.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateSchemaId {
    pub engine: String,
    pub version: u32,
    pub sub_version: u32,
}

impl StateSchemaId {
    pub fn new(engine: impl Into<String>, version: u32, sub_version: u32) -> Self {
        Self {
            engine: engine.into(),
            version,
            sub_version,
        }
    }

    /// Same engine and major representation version. A sub-version change may
    /// add validation/numerical metadata but must not silently change layout.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.engine == other.engine && self.version == other.version
    }
}

impl fmt::Display for StateSchemaId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}.{}", self.engine, self.version, self.sub_version)
    }
}

/// Engine-specific state adapter.
///
/// The engine owns semantics. Core only asks the adapter to export/import an
/// exact completed-token prefix.
pub trait CausalStateCodec {
    type EngineState;

    fn schema_id(&self) -> StateSchemaId;

    fn export_state(
        &self,
        state: &Self::EngineState,
        prefix_len: usize,
    ) -> Result<CausalState, String>;

    fn import_state(
        &self,
        state: &mut Self::EngineState,
        prefix_len: usize,
        causal: &CausalState,
    ) -> Result<(), String>;
}

pub mod page;
pub mod schema;
pub mod snapshot;
pub mod transaction;

pub use page::{PageId, StatePage};
pub use schema::{DataType, RegionKind, StateRegion, StateSchema};
pub use snapshot::{Section, SectionKind, SnapshotError, SnapshotPayload, StateSnapshot};
pub use transaction::{Generation, TransactionCheckpoint, TransactionManager, TransactionResult};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_constructors_are_consistent() {
        let state = CausalState::append_only(2, 4, 8);
        let CausalState::AppendOnly(s) = state else {
            panic!("wrong state kind");
        };
        assert_eq!(s.keys.len(), 64);
        assert_eq!(s.values.len(), 64);
        assert_eq!(s.position, 0);

        let ring = CausalState::ring(16, 4);
        assert_eq!(ring.len(), Some(0));

        let mutable = CausalState::mutable_fixed(8);
        assert_eq!(mutable.position(), Some(0));

        let paged = CausalState::sparse_paged(4, 4);
        assert_eq!(paged.len(), Some(0));

        let opaque = CausalState::opaque(vec![1, 2, 3]);
        assert_eq!(opaque.len(), None);
    }

    #[test]
    fn schema_compatibility_is_explicit() {
        let a = StateSchemaId::new("llama", 1, 0);
        let b = StateSchemaId::new("llama", 1, 1);
        let c = StateSchemaId::new("llama", 2, 0);
        assert!(a.is_compatible_with(&b));
        assert!(!a.is_compatible_with(&c));
    }
}
