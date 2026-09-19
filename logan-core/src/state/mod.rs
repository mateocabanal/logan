//! Generic model-neutral causal-state subsystem.
//!
//! Owns reusable runtime mechanisms for representing and managing causal state
//! across engine families. Mathematical meaning stays in engine-specific codecs;
//! core provides storage, paging, transactions, and snapshot format.

use std::fmt;

/// A model-neutral causal state that can represent various persistent patterns.
/// Engines provide codec adapters to export/import their concrete state into
/// this neutral container.
#[derive(Debug, Clone)]
pub enum CausalState {
    /// Keys/values appended as new tokens are processed. Position advances monotonically.
    AppendOnly(AppendOnlyState),
    /// Circular buffer where old entries wrap around. Useful for recurrent state,
    /// convolution history, and windowed attention.
    Ring(RingState),
    /// Fixed-size mutable buffer. State is updated in-place, size never changes.
    MutableFixed(MutableFixedState),
    /// Sparse paged layout: entries mapped to pages, with COW semantics for
    /// transactions. KV cache with paging is the primary use case.
    SparsePaged(SparsePagedState),
    /// Opaque engine-defined state. The core stores bytes and manages lifecycle
    /// (paging, COW, transactions) without interpreting the meaning.
    Opaque(Vec<u8>),
}

/// Append-only causal state: keys/values are appended as new tokens are processed.
/// Position advances monotonically. Used for Llama V1/V2 KV and MiniCPM.
#[derive(Debug, Clone, Default)]
pub struct AppendOnlyState {
    /// Keys rows, stored head-major: [head, seq, dim] -> flattened
    pub keys: Vec<f32>,
    /// Values rows, stored head-major: [head, seq, dim] -> flattened
    pub values: Vec<f32>,
    /// Current position (number of tokens processed so far)
    pub position: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Key/value dimension per head
    pub dim: usize,
}

/// Ring (circular buffer) causal state: old entries wrap around when the buffer
/// is full. Used for GDN recurrent state, convolution history, and windowed KV.
#[derive(Debug, Clone, Default)]
pub struct RingState {
    /// Pre-allocated buffer of maximum capacity
    pub buffer: Vec<f32>,
    /// Maximum number of entries to retain
    pub capacity: usize,
    /// Current write position (wraps around)
    pub write_pos: usize,
    /// Current read position (what has been consumed)
    pub read_pos: usize,
    /// Number of valid entries currently in the ring
    pub len: usize,
}

/// Fixed-size mutable state: buffer size is fixed, content updates in-place.
/// Used for GDN convolution history and partial compressor state that needs to
/// be retained but overwritten.
#[derive(Debug, Clone, Default)]
pub struct MutableFixedState {
    /// Pre-allocated buffer
    pub data: Vec<f32>,
    /// Current write position (overwrites in place)
    pub write_pos: usize,
    /// Number of valid elements
    pub len: usize,
    /// Whether the buffer has wrapped (been filled once)
    pub has_wrapped: bool,
}

/// Sparse paged state: KV entries mapped to pages with COW semantics.
/// Primary use case: paged KV cache with SSD backing.
#[derive(Debug, Clone, Default)]
pub struct SparsePagedState {
    /// Page table mapping logical page index -> file-backed page offset
    pub page_table: Vec<Option<u64>>,
    /// Per-page data (RAM or SSD)
    pub pages: Vec<Vec<f32>>,
    /// Number of valid entries per page
    pub page_entries: usize,
    /// Maximum number of pages retained
    pub max_pages: usize,
    /// Whether pages are pinned (not eligible for eviction)
    pub pinned: bool,
}

impl CausalState {
    /// Create a new AppendOnly state for the given architecture
    pub fn append_only(n_heads: usize, dim: usize, max_seq: usize) -> Self {
        let size = n_heads * max_seq * dim;
        CausalState::AppendOnly(AppendOnlyState {
            keys: vec![0.0; size],
            values: vec![0.0; size],
            position: 0,
            n_heads,
            dim,
        })
    }

    /// Create a new Ring state with given capacity
    pub fn ring(capacity: usize, width: usize) -> Self {
        CausalState::Ring(RingState {
            buffer: vec![0.0; capacity * width],
            capacity,
            write_pos: 0,
            read_pos: 0,
            len: 0,
        })
    }

    /// Create a new MutableFixed state
    pub fn mutable_fixed(capacity: usize) -> Self {
        CausalState::MutableFixed(MutableFixedState {
            data: vec![0.0; capacity],
            write_pos: 0,
            len: 0,
            has_wrapped: false,
        })
    }

    /// Create a new SparsePaged state
    pub fn sparse_paged(max_pages: usize, page_entries: usize) -> Self {
        CausalState::SparsePaged(SparsePagedState {
            page_table: vec![None; max_pages],
            pages: vec![vec![0.0; page_entries]; max_pages],
            page_entries,
            max_pages,
            pinned: false,
        })
    }

    /// Create Opaque state from raw bytes
    pub fn opaque(data: Vec<u8>) -> Self {
        CausalState::Opaque(data)
    }

    /// Get the position if applicable
    pub fn position(&self) -> Option<usize> {
        match self {
            CausalState::AppendOnly(s) => Some(s.position),
            CausalState::Ring(s) => Some(s.read_pos),
            CausalState::MutableFixed(s) => Some(s.write_pos),
            CausalState::SparsePaged(_) => None,
            CausalState::Opaque(_) => None,
        }
    }

    /// Get the number of valid entries if applicable
    pub fn len(&self) -> Option<usize> {
        match self {
            CausalState::AppendOnly(s) => Some(s.position),
            CausalState::Ring(s) => Some(s.len),
            CausalState::MutableFixed(s) => Some(s.len),
            CausalState::SparsePaged(s) => Some(s.page_table.iter().filter(|p| p.is_some()).count()),
            CausalState::Opaque(_) => None,
        }
    }
}

/// Trait for engine-specific codec adapters that export/import state.
/// Core owns the lifecycle/storage/versioning; engine owns the semantics.
pub trait CausalStateCodec {
    /// Return a schema ID identifying this engine's state representation.
    fn schema_id(&self) -> StateSchemaId;

    /// Export engine state into the neutral CausalState.
    fn export_state(&self, state: &Self) -> Result<CausalState, String>;

    /// Import neutral CausalState into engine-specific state.
    fn import_state(&self, state: &mut Self, causal: &CausalState) -> Result<(), String>;
}

/// Engine-state trait: core stores/manages state, engine knows semantics.
pub trait EngineState: Clone {
    /// Check if this state is valid after a restore
    fn is_valid(&self) -> bool;
}

/// Versioned snapshot of state schema
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateSchemaId {
    /// Engine identifier (e.g. "llama", "qwen4", "v4")
    pub engine: &'static str,
    /// State representation version
    pub version: u32,
    /// Optional sub-version for numerical policy changes
    pub sub_version: u32,
}

pub mod snapshot;
pub use snapshot::{StateSnapshot, SnapshotPayload, Section, SectionKind, SnapshotError};

#[cfg(test)]
mod tests {
    use super::*;

    fn test_schema() -> StateSchemaId {
        StateSchemaId {
            engine: "test",
            version: 1,
            sub_version: 0,
        }
    }

    #[test]
    fn test_append_only_creation() {
        let state = CausalState::append_only(32, 64, 8192);
        match state {
            CausalState::AppendOnly(s) => {
                assert_eq!(s.n_heads, 32);
                assert_eq!(s.dim, 64);
                assert_eq!(s.position, 0);
            }
            _ => panic!("Expected AppendOnly"),
        }
    }

    #[test]
    fn test_ring_creation() {
        let state = CausalState::ring(4096, 128);
        match state {
            CausalState::Ring(s) => {
                assert_eq!(s.capacity, 4096);
                assert_eq!(s.write_pos, 0);
                assert_eq!(s.read_pos, 0);
                assert_eq!(s.len, 0);
            }
            _ => panic!("Expected Ring"),
        }
    }

    #[test]
    fn test_mutable_fixed_creation() {
        let state = CausalState::mutable_fixed(1024);
        match state {
            CausalState::MutableFixed(s) => {
                assert_eq!(s.data.len(), 1024);
                assert_eq!(s.write_pos, 0);
                assert_eq!(s.len, 0);
            }
            _ => panic!("Expected MutableFixed"),
        }
    }

    #[test]
    fn test_sparse_paged_creation() {
        let state = CausalState::sparse_paged(256, 256);
        match state {
            CausalState::SparsePaged(s) => {
                assert_eq!(s.max_pages, 256);
                assert_eq!(s.page_entries, 256);
                assert_eq!(s.pinned, false);
            }
            _ => panic!("Expected SparsePaged"),
        }
    }

    #[test]
    fn test_opaque_creation() {
        let data = vec![1u8, 2, 3, 4];
        let state = CausalState::opaque(data.clone());
        match state {
            CausalState::Opaque(s) => assert_eq!(s, data),
            _ => panic!("Expected Opaque"),
        }
    }

    #[test]
    fn test_state_position() {
        let a = CausalState::append_only(2, 4, 8);
        assert_eq!(a.position(), Some(0));

        let r = CausalState::ring(16, 4);
        assert_eq!(r.position(), Some(0));

        let m = CausalState::mutable_fixed(8);
        assert_eq!(m.position(), Some(0));

        let s = CausalState::sparse_paged(4, 4);
        assert_eq!(s.position(), None);

        let o = CausalState::opaque(vec![1, 2, 3]);
        assert_eq!(o.position(), None);
    }

    #[test]
    fn test_state_len() {
        let a = CausalState::append_only(2, 4, 8);
        assert_eq!(a.len(), Some(0));

        let r = CausalState::ring(16, 4);
        assert_eq!(r.len(), Some(0));

        let m = CausalState::mutable_fixed(8);
        assert_eq!(m.len(), Some(0));

        let s = CausalState::sparse_paged(4, 4);
        assert_eq!(s.len(), Some(0));

        let o = CausalState::opaque(vec![1, 2, 3]);
        assert_eq!(o.len(), None);
    }
}