//! Versioned state snapshot for persistence.

use crate::state::{CausalState, StateSchemaId};
use std::fmt;

/// A persisted snapshot of causal state.
/// Used for SSD prefix cache and cross-session restore.
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    /// Schema ID for versioning/validation
    pub schema_id: StateSchemaId,
    /// Prefix token count this snapshot represents
    pub prefix_len: usize,
    /// Prefix token hash for identity verification
    pub prefix_hash: u64,
    /// The causal state payload (bytes for serialization)
    pub payload: SnapshotPayload,
    /// Checksum for integrity verification
    pub checksum: [u8; 32],
    /// Generation for staleness detection
    pub generation: u64,
}

/// Payload types for storage
#[derive(Debug, Clone)]
pub enum SnapshotPayload {
    /// Raw byte payload (for Opaque state)
    Raw(Vec<u8>),
    /// Structured sections (for structured state types)
    Sections(Vec<Section>),
}

/// A section within a snapshot payload
#[derive(Debug, Clone)]
pub struct Section {
    /// Section type identifier
    pub kind: SectionKind,
    /// Offset into the payload data
    pub offset: usize,
    /// Length in bytes
    pub length: usize,
}

/// Section kind identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    /// Append-only KV keys
    AppendOnlyKeys,
    /// Append-only KV values
    AppendOnlyValues,
    /// Position/marker
    Position,
    /// Ring buffer data
    RingBuffer,
    /// Mutable fixed state
    MutableFixed,
    /// Paged index
    PageTable,
    /// Engine-defined opaque data
    Opaque,
}

/// Snapshot error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotError {
    /// Checksum mismatch
    ChecksumMismatch,
    /// Wrong schema version
    WrongSchema,
    /// Truncated data
    Truncated,
    /// Wrong prefix hash
    PrefixHashMismatch,
    /// Stale generation
    StaleGeneration,
    /// Invalid section
    InvalidSection,
    /// IO error
    Io(String),
}

impl fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SnapshotError::ChecksumMismatch => write!(f, "checksum mismatch"),
            SnapshotError::WrongSchema => write!(f, "wrong schema version"),
            SnapshotError::Truncated => write!(f, "truncated data"),
            SnapshotError::PrefixHashMismatch => write!(f, "prefix hash mismatch"),
            SnapshotError::StaleGeneration => write!(f, "stale generation"),
            SnapshotError::InvalidSection => write!(f, "invalid section"),
            SnapshotError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for SnapshotError {}

impl StateSnapshot {
    /// Current container format version
    pub const FORMAT_VERSION: u32 = 1;
    /// Current state ABI version
    pub const STATE_ABI_VERSION: u32 = 1;
    /// Magic bytes for format identification
    pub const MAGIC: [u8; 8] = *b"LOGANST0";

    /// Create a new snapshot from causal state
    pub fn new(
        schema_id: StateSchemaId,
        prefix_len: usize,
        prefix_hash: u64,
        state: &CausalState,
    ) -> Result<Self, String> {
        let payload = Self::serialize_state(state)?;
        let checksum = Self::compute_checksum(&payload)?;

        Ok(StateSnapshot {
            schema_id,
            prefix_len,
            prefix_hash,
            payload,
            checksum,
            generation: 0,
        })
    }

    /// Serialize causal state to payload
    fn serialize_state(state: &CausalState) -> Result<SnapshotPayload, String> {
        match state {
            CausalState::AppendOnly(s) => {
                let mut bytes = Vec::new();
                // Keys + Values as f32 bytes
                bytes.extend_from_slice(bytemuck::cast_slice(&s.keys));
                bytes.extend_from_slice(bytemuck::cast_slice(&s.values));
                Ok(SnapshotPayload::Raw(bytes))
            }
            CausalState::Ring(s) => {
                Ok(SnapshotPayload::Raw(bytemuck::cast_slice(&s.buffer).to_vec()))
            }
            CausalState::MutableFixed(s) => {
                Ok(SnapshotPayload::Raw(bytemuck::cast_slice(&s.data).to_vec()))
            }
            CausalState::SparsePaged(s) => {
                let mut bytes = Vec::new();
                for page in &s.pages {
                    bytes.extend_from_slice(bytemuck::cast_slice(page));
                }
                Ok(SnapshotPayload::Raw(bytes))
            }
            CausalState::Opaque(data) => {
                Ok(SnapshotPayload::Raw(data.clone()))
            }
        }
    }

    /// Compute checksum of payload
    fn compute_checksum(payload: &SnapshotPayload) -> Result<[u8; 32], String> {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        match payload {
            SnapshotPayload::Raw(bytes) => {
                hasher.update(bytes);
            }
            SnapshotPayload::Sections(sections) => {
                for s in sections {
                    hasher.update(&s.kind.to_be_bytes());
                    hasher.update(&s.offset.to_be_bytes());
                    hasher.update(&s.length.to_be_bytes());
                }
            }
        }
        let result = hasher.finalize();
        let mut checksum = [0u8; 32];
        checksum.copy_from_slice(&result);
        Ok(checksum)
    }

    /// Validate snapshot integrity
    pub fn validate(&self) -> Result<(), SnapshotError> {
        // Verify checksum
        let expected = Self::compute_checksum(&self.payload)
            .map_err(|_| SnapshotError::Truncated)?;
        if expected != self.checksum {
            return Err(SnapshotError::ChecksumMismatch);
        }
        Ok(())
    }

    /// Deserialize into causal state
    pub fn into_causal_state(&self) -> Result<CausalState, String> {
        self.validate().map_err(|e| format!("{:?}", e))?;
        match &self.payload {
            SnapshotPayload::Raw(bytes) => {
                // Try to interpret as append-only (most common case)
                Ok(CausalState::Opaque(bytes.clone()))
            }
            SnapshotPayload::Sections(_) => {
                Err("sections not yet implemented".to_string())
            }
        }
    }
}

impl SectionKind {
    pub fn to_be_bytes(&self) -> [u8; 1] {
        match self {
            SectionKind::AppendOnlyKeys => [1],
            SectionKind::AppendOnlyValues => [2],
            SectionKind::Position => [3],
            SectionKind::RingBuffer => [4],
            SectionKind::MutableFixed => [5],
            SectionKind::PageTable => [6],
            SectionKind::Opaque => [7],
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            SectionKind::AppendOnlyKeys => "ak",
            SectionKind::AppendOnlyValues => "av",
            SectionKind::Position => "pos",
            SectionKind::RingBuffer => "rb",
            SectionKind::MutableFixed => "mf",
            SectionKind::PageTable => "pt",
            SectionKind::Opaque => "op",
        }
    }
}

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
    fn test_snapshot_creation() {
        let state = CausalState::append_only(2, 4, 8);
        let snap = StateSnapshot::new(
            test_schema(),
            100,
            0xabc123,
            &state,
        ).unwrap();

        assert_eq!(snap.schema_id, test_schema());
        assert_eq!(snap.prefix_len, 100);
        assert_eq!(snap.prefix_hash, 0xabc123);
    }

    #[test]
    fn test_snapshot_validation() {
        let state = CausalState::append_only(2, 4, 8);
        let snap = StateSnapshot::new(
            test_schema(),
            50,
            0xdef456,
            &state,
        ).unwrap();

        assert!(snap.validate().is_ok());
    }

    #[test]
    fn test_snapshot_checksum_mismatch() {
        let state = CausalState::append_only(2, 4, 8);
        let mut snap = StateSnapshot::new(
            test_schema(),
            50,
            0x000000,
            &state,
        ).unwrap();

        // Corrupt checksum
        snap.checksum[0] ^= 0xFF;
        assert!(snap.validate().is_err());
    }

    #[test]
    fn test_snapshot_deserialize() {
        let state = CausalState::append_only(2, 4, 8);
        let snap = StateSnapshot::new(
            test_schema(),
            50,
            0x123456,
            &state,
        ).unwrap();

        let roundtrip = snap.into_causal_state().unwrap();
        match roundtrip {
            CausalState::Opaque(_) => {} // Expected
            _ => panic!("Expected Opaque variant"),
        }
    }
}