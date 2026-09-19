//! Stable binary container for persistent generic prefix snapshots.
//!
//! The header carries all cache-compatibility axes. The body carries the exact
//! token sequence followed by a versioned StateSnapshot. Longest-prefix lookup
//! can therefore be verified after a process restart instead of trusting hashes.

use crate::prefix::{
    ModelFingerprint, PlanFingerprint, PrefixKey, StateSchemaFingerprint, TokenizerFingerprint,
};
use crate::state::StateSnapshot;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

pub const CONTAINER_MAGIC: [u8; 8] = *b"LOGANPF1";
pub const CONTAINER_VERSION: u32 = 2;
pub const HEADER_BYTES: usize = 204;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContainerHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub state_abi_version: u32,
    pub model_fingerprint: [u8; 32],
    pub tokenizer_fingerprint: [u8; 32],
    pub state_schema_fingerprint: [u8; 32],
    pub plan_fingerprint: [u8; 32],
    pub prefix_token_hash: u64,
    pub prefix_len: usize,
    /// StateSnapshot byte length, excluding the token vector.
    pub payload_size: u64,
    pub section_count: u32,
    /// SHA-256(tokens || snapshot bytes).
    pub checksum: [u8; 32],
}

impl ContainerHeader {
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let prefix_len = u64::try_from(self.prefix_len)
            .map_err(|_| "prefix length exceeds container format".to_string())?;
        let mut bytes = Vec::with_capacity(HEADER_BYTES);
        bytes.extend_from_slice(&self.magic);
        bytes.extend_from_slice(&self.version.to_le_bytes());
        bytes.extend_from_slice(&self.state_abi_version.to_le_bytes());
        bytes.extend_from_slice(&self.model_fingerprint);
        bytes.extend_from_slice(&self.tokenizer_fingerprint);
        bytes.extend_from_slice(&self.state_schema_fingerprint);
        bytes.extend_from_slice(&self.plan_fingerprint);
        bytes.extend_from_slice(&self.prefix_token_hash.to_le_bytes());
        bytes.extend_from_slice(&prefix_len.to_le_bytes());
        bytes.extend_from_slice(&self.payload_size.to_le_bytes());
        bytes.extend_from_slice(&self.section_count.to_le_bytes());
        bytes.extend_from_slice(&self.checksum);
        debug_assert_eq!(bytes.len(), HEADER_BYTES);
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != HEADER_BYTES {
            return Err(format!(
                "prefix header has {} bytes, expected {HEADER_BYTES}",
                bytes.len()
            ));
        }
        let mut c = Cursor::new(bytes);
        let magic = c.array::<8>()?;
        let version = c.u32()?;
        let state_abi_version = c.u32()?;
        let model_fingerprint = c.array::<32>()?;
        let tokenizer_fingerprint = c.array::<32>()?;
        let state_schema_fingerprint = c.array::<32>()?;
        let plan_fingerprint = c.array::<32>()?;
        let prefix_token_hash = c.u64()?;
        let prefix_len = usize::try_from(c.u64()?)
            .map_err(|_| "prefix length exceeds host usize".to_string())?;
        let payload_size = c.u64()?;
        let section_count = c.u32()?;
        let checksum = c.array::<32>()?;
        if !c.done() {
            return Err("trailing bytes in prefix header".into());
        }
        Ok(Self {
            magic,
            version,
            state_abi_version,
            model_fingerprint,
            tokenizer_fingerprint,
            state_schema_fingerprint,
            plan_fingerprint,
            prefix_token_hash,
            prefix_len,
            payload_size,
            section_count,
            checksum,
        })
    }

    pub fn key_for_tokens(&self, tokens: Vec<u32>) -> Result<PrefixKey, String> {
        if tokens.len() != self.prefix_len {
            return Err("prefix token count does not match header".into());
        }
        let key = PrefixKey::new(
            ModelFingerprint {
                digest: self.model_fingerprint,
            },
            StateSchemaFingerprint {
                digest: self.state_schema_fingerprint,
            },
            TokenizerFingerprint {
                digest: self.tokenizer_fingerprint,
            },
            PlanFingerprint {
                digest: self.plan_fingerprint,
            },
            tokens,
        );
        if key.prefix_token_hash != self.prefix_token_hash {
            return Err("prefix token hash does not match header".into());
        }
        Ok(key)
    }
}

pub fn write_snapshot<W: Write>(
    writer: &mut W,
    key: &PrefixKey,
    state: &StateSnapshot,
) -> Result<(), std::io::Error> {
    if state.prefix_len != key.prefix_len() || state.prefix_hash != key.prefix_token_hash {
        return Err(invalid_data(
            "state snapshot prefix identity does not match cache key",
        ));
    }
    let snapshot = state.to_bytes().map_err(invalid_data)?;
    let tokens = encode_tokens(&key.prefix_tokens);
    let checksum = compute_checksum(&tokens, &snapshot);
    let header = ContainerHeader {
        magic: CONTAINER_MAGIC,
        version: CONTAINER_VERSION,
        state_abi_version: StateSnapshot::STATE_ABI_VERSION,
        model_fingerprint: key.model_fingerprint.digest,
        tokenizer_fingerprint: key.tokenizer_fingerprint.digest,
        state_schema_fingerprint: key.state_schema_fingerprint.digest,
        plan_fingerprint: key.plan_fingerprint.digest,
        prefix_token_hash: key.prefix_token_hash,
        prefix_len: key.prefix_len(),
        payload_size: snapshot.len() as u64,
        section_count: 1,
        checksum,
    };
    writer.write_all(&header.to_bytes().map_err(invalid_data)?)?;
    writer.write_all(&tokens)?;
    writer.write_all(&snapshot)?;
    Ok(())
}

pub fn read_snapshot<R: Read>(
    reader: &mut R,
) -> Result<(ContainerHeader, PrefixKey, StateSnapshot), std::io::Error> {
    let mut header_bytes = [0u8; HEADER_BYTES];
    reader.read_exact(&mut header_bytes)?;
    let header = ContainerHeader::from_bytes(&header_bytes).map_err(invalid_data)?;
    if header.magic != CONTAINER_MAGIC {
        return Err(invalid_data("bad prefix cache magic"));
    }
    if header.version != CONTAINER_VERSION {
        return Err(invalid_data(format!(
            "unsupported prefix container version {}",
            header.version
        )));
    }
    if header.state_abi_version != StateSnapshot::STATE_ABI_VERSION {
        return Err(invalid_data(format!(
            "unsupported state ABI version {}",
            header.state_abi_version
        )));
    }

    let token_bytes_len = header
        .prefix_len
        .checked_mul(4)
        .ok_or_else(|| invalid_data("prefix token byte length overflow"))?;
    let mut token_bytes = vec![0u8; token_bytes_len];
    reader.read_exact(&mut token_bytes)?;
    let tokens = decode_tokens(&token_bytes);
    let key = header.key_for_tokens(tokens).map_err(invalid_data)?;

    let payload_len = usize::try_from(header.payload_size)
        .map_err(|_| invalid_data("prefix snapshot exceeds host address space"))?;
    let mut snapshot_bytes = vec![0u8; payload_len];
    reader.read_exact(&mut snapshot_bytes)?;
    if compute_checksum(&token_bytes, &snapshot_bytes) != header.checksum {
        return Err(invalid_data("prefix cache checksum mismatch"));
    }
    let snapshot = StateSnapshot::from_bytes(&snapshot_bytes).map_err(invalid_data)?;
    if snapshot.prefix_len != key.prefix_len() || snapshot.prefix_hash != key.prefix_token_hash {
        return Err(invalid_data(
            "embedded state snapshot prefix identity mismatch",
        ));
    }

    Ok((header, key, snapshot))
}

pub fn is_compatible(header: &ContainerHeader, query: &PrefixKey) -> bool {
    header.model_fingerprint == query.model_fingerprint.digest
        && header.tokenizer_fingerprint == query.tokenizer_fingerprint.digest
        && header.state_schema_fingerprint == query.state_schema_fingerprint.digest
        && header.plan_fingerprint == query.plan_fingerprint.digest
}

fn encode_tokens(tokens: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes
}

fn decode_tokens(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|raw| u32::from_le_bytes(raw.try_into().unwrap()))
        .collect()
}

fn compute_checksum(tokens: &[u8], snapshot: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(tokens);
    hasher.update(snapshot);
    hasher.finalize().into()
}

fn invalid_data(error: impl ToString) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
}

struct Cursor<'a> {
    bytes: &'a [u8],
    off: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, off: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self
            .off
            .checked_add(len)
            .ok_or_else(|| "prefix header offset overflow".to_string())?;
        let out = self
            .bytes
            .get(self.off..end)
            .ok_or_else(|| "truncated prefix header".to_string())?;
        self.off = end;
        Ok(out)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        self.take(N)?
            .try_into()
            .map_err(|_| "truncated prefix header".to_string())
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn done(&self) -> bool {
        self.off == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefix::{
        ModelFingerprint, PlanFingerprint, StateSchemaFingerprint, TokenizerFingerprint,
    };
    use crate::state::{CausalState, StateSchemaId};

    fn key(tokens: &[u32]) -> PrefixKey {
        PrefixKey::new(
            ModelFingerprint { digest: [1; 32] },
            StateSchemaFingerprint { digest: [2; 32] },
            TokenizerFingerprint { digest: [3; 32] },
            PlanFingerprint { digest: [4; 32] },
            tokens.to_vec(),
        )
    }

    #[test]
    fn header_has_one_exact_size_and_round_trips() {
        let key = key(&[1, 2, 3]);
        let header = ContainerHeader {
            magic: CONTAINER_MAGIC,
            version: CONTAINER_VERSION,
            state_abi_version: 1,
            model_fingerprint: key.model_fingerprint.digest,
            tokenizer_fingerprint: key.tokenizer_fingerprint.digest,
            state_schema_fingerprint: key.state_schema_fingerprint.digest,
            plan_fingerprint: key.plan_fingerprint.digest,
            prefix_token_hash: key.prefix_token_hash,
            prefix_len: key.prefix_len(),
            payload_size: 42,
            section_count: 1,
            checksum: [5; 32],
        };
        let bytes = header.to_bytes().unwrap();
        assert_eq!(bytes.len(), HEADER_BYTES);
        assert_eq!(ContainerHeader::from_bytes(&bytes).unwrap(), header);
    }

    #[test]
    fn full_container_round_trip_preserves_tokens_and_state() {
        let key = key(&[7, 8, 9]);
        let state = StateSnapshot::new(
            StateSchemaId::new("test", 1, 0),
            key.prefix_len(),
            key.prefix_token_hash,
            &CausalState::Opaque(vec![10, 11]),
        )
        .unwrap();
        let mut bytes = Vec::new();
        write_snapshot(&mut bytes, &key, &state).unwrap();
        let (_, decoded_key, decoded_state) = read_snapshot(&mut bytes.as_slice()).unwrap();
        assert_eq!(decoded_key, key);
        assert_eq!(decoded_state, state);
    }

    #[test]
    fn token_corruption_is_rejected() {
        let key = key(&[7, 8, 9]);
        let state = StateSnapshot::new(
            StateSchemaId::new("test", 1, 0),
            key.prefix_len(),
            key.prefix_token_hash,
            &CausalState::Opaque(vec![10]),
        )
        .unwrap();
        let mut bytes = Vec::new();
        write_snapshot(&mut bytes, &key, &state).unwrap();
        bytes[HEADER_BYTES] ^= 1;
        assert!(read_snapshot(&mut bytes.as_slice()).is_err());
    }
}
