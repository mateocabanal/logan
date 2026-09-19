//! Versioned snapshot container format.
//!
//! Stable, generic container for persistent prefix snapshots.
//! Does not serialize Rust structs directly - uses versioned binary format.

use crate::state::StateSnapshot;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

/// Container magic
pub const CONTAINER_MAGIC: [u8; 8] = *b"LOGANPF1";
/// Container format version
pub const CONTAINER_VERSION: u32 = 1;

/// Snapshot container header
#[derive(Debug, Clone)]
pub struct ContainerHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub state_abi_version: u32,
    pub model_fingerprint: [u8; 32],
    pub tokenizer_fingerprint: [u8; 32],
    pub state_schema_id: [u8; 32],
    pub prefix_token_hash: u64,
    pub prefix_len: usize,
    pub payload_size: u64,
    pub section_count: u32,
    pub checksum: [u8; 32],
}

impl ContainerHeader {
    /// Serialize header to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.magic);
        bytes.extend_from_slice(&self.version.to_le_bytes());
        bytes.extend_from_slice(&self.state_abi_version.to_le_bytes());
        bytes.extend_from_slice(&self.model_fingerprint);
        bytes.extend_from_slice(&self.tokenizer_fingerprint);
        bytes.extend_from_slice(&self.state_schema_id);
        bytes.extend_from_slice(&self.prefix_token_hash.to_le_bytes());
        bytes.extend_from_slice(&(self.prefix_len as u64).to_le_bytes());
        bytes.extend_from_slice(&self.payload_size.to_le_bytes());
        bytes.extend_from_slice(&self.section_count.to_le_bytes());
        bytes.extend_from_slice(&self.checksum);
        bytes
    }

    /// Parse header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 8 + 4 + 4 + 32 + 32 + 32 + 8 + 8 + 8 + 4 + 32 {
            return Err("header too short".to_string());
        }
        let mut offset = 0;
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[offset..offset + 8]);
        offset += 8;
        let version = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let state_abi_version = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let mut model_fingerprint = [0u8; 32];
        model_fingerprint.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;
        let mut tokenizer_fingerprint = [0u8; 32];
        tokenizer_fingerprint.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;
        let mut state_schema_id = [0u8; 32];
        state_schema_id.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;
        let prefix_token_hash = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        offset += 8;
        let prefix_len = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        let payload_size = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        offset += 8;
        let section_count = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let mut checksum = [0u8; 32];
        checksum.copy_from_slice(&bytes[offset..offset + 32]);

        Ok(ContainerHeader {
            magic,
            version,
            state_abi_version,
            model_fingerprint,
            tokenizer_fingerprint,
            state_schema_id,
            prefix_token_hash,
            prefix_len,
            payload_size,
            section_count,
            checksum,
        })
    }
}

/// Section descriptor in container
#[derive(Debug, Clone)]
pub struct SectionDescriptor {
    pub kind: u8,
    pub offset: u64,
    pub length: u64,
    pub checksum: [u8; 32],
}

/// Write snapshot to writer
pub fn write_snapshot<W: Write>(
    writer: &mut W,
    state: &StateSnapshot,
    schema_id: [u8; 32],
) -> Result<(), std::io::Error> {
    let payload = serialize_state(state)?;
    let checksum = compute_checksum(&payload);

    let header = ContainerHeader {
        magic: CONTAINER_MAGIC,
        version: CONTAINER_VERSION,
        state_abi_version: StateSnapshot::STATE_ABI_VERSION,
        model_fingerprint: state.schema_id.engine.as_bytes()[..32.min(state.schema_id.engine.as_bytes().len())].to_vec().try_into().unwrap_or([0; 32]),
        tokenizer_fingerprint: [0; 32], // TODO: fill from schema
        state_schema_id: schema_id,
        prefix_token_hash: state.prefix_hash,
        prefix_len: state.prefix_len,
        payload_size: payload.len() as u64,
        section_count: 1,
        checksum,
    };

    writer.write_all(&header.to_bytes())?;
    writer.write_all(&payload)?;

    Ok(())
}

/// Read and validate snapshot from reader
pub fn read_snapshot<R: Read>(
    reader: &mut R,
) -> Result<(ContainerHeader, Vec<u8>), std::io::Error> {
    let mut header_bytes = vec![0u8; 128];
    reader.read_exact(&mut header_bytes)?;
    let header = ContainerHeader::from_bytes(&header_bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    if header.magic != CONTAINER_MAGIC {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
    }
    if header.version != CONTAINER_VERSION {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("unsupported version: {}", header.version)));
    }

    let mut payload = vec![0u8; header.payload_size as usize];
    reader.read_exact(&mut payload)?;

    // Verify checksum
    let expected = compute_checksum(&payload);
    if expected != header.checksum {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "checksum mismatch"));
    }

    Ok((header, payload))
}

/// Serialize state to payload bytes
fn serialize_state(state: &StateSnapshot) -> Result<Vec<u8>, std::io::Error> {
    // Simplified - just return the raw payload
    match &state.payload {
        crate::state::SnapshotPayload::Raw(bytes) => Ok(bytes.clone()),
        crate::state::SnapshotPayload::Sections(sections) => {
            let mut out = Vec::new();
            for s in sections {
                out.extend_from_slice(&s.kind.to_be_bytes());
                out.extend_from_slice(&s.offset.to_be_bytes());
                out.extend_from_slice(&s.length.to_be_bytes());
            }
            Ok(out)
        }
    }
}

/// Compute SHA-256 checksum
fn compute_checksum(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut checksum = [0u8; 32];
    checksum.copy_from_slice(&result);
    checksum
}

/// Compatibility check
pub fn is_compatible(
    header: &ContainerHeader,
    expected_model: &[u8; 32],
    expected_tokenizer: &[u8; 32],
    expected_schema: &[u8; 32],
) -> bool {
    header.model_fingerprint == *expected_model
        && header.tokenizer_fingerprint == *expected_tokenizer
        && header.state_schema_id == *expected_schema
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_header_roundtrip() {
        let header = ContainerHeader {
            magic: CONTAINER_MAGIC,
            version: CONTAINER_VERSION,
            state_abi_version: 1,
            model_fingerprint: [1; 32],
            tokenizer_fingerprint: [2; 32],
            state_schema_id: [3; 32],
            prefix_token_hash: 0xabc123,
            prefix_len: 100,
            payload_size: 1024,
            section_count: 1,
            checksum: [4; 32],
        };

        let bytes = header.to_bytes();
        let parsed = ContainerHeader::from_bytes(&bytes).unwrap();

        assert_eq!(header.magic, parsed.magic);
        assert_eq!(header.version, parsed.version);
        assert_eq!(header.state_abi_version, parsed.state_abi_version);
        assert_eq!(header.model_fingerprint, parsed.model_fingerprint);
        assert_eq!(header.tokenizer_fingerprint, parsed.tokenizer_fingerprint);
        assert_eq!(header.state_schema_id, parsed.state_schema_id);
        assert_eq!(header.prefix_token_hash, parsed.prefix_token_hash);
        assert_eq!(header.prefix_len, parsed.prefix_len);
        assert_eq!(header.payload_size, parsed.payload_size);
        assert_eq!(header.section_count, parsed.section_count);
        assert_eq!(header.checksum, parsed.checksum);
    }

    #[test]
    fn test_checksum() {
        let data = b"test data";
        let c1 = compute_checksum(data);
        let c2 = compute_checksum(data);
        assert_eq!(c1, c2);
        assert_ne!(c1, compute_checksum(b"other"));
    }
}