//! Versioned, checksummed causal-state snapshots.

use super::{
    AppendOnlyState, CausalState, CausalStateCodec, MutableFixedState, RingState, SparsePagedState,
    StateSchemaId,
};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateSnapshot {
    pub schema_id: StateSchemaId,
    pub prefix_len: usize,
    pub prefix_hash: u64,
    pub payload: SnapshotPayload,
    pub checksum: [u8; 32],
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotPayload {
    Raw(Vec<u8>),
    Sections(Vec<Section>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub kind: SectionKind,
    pub offset: usize,
    pub length: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    AppendOnlyKeys,
    AppendOnlyValues,
    Position,
    RingBuffer,
    MutableFixed,
    PageTable,
    Opaque,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotError {
    ChecksumMismatch,
    WrongSchema,
    Truncated,
    WrongPrefixHash,
    StaleGeneration,
    InvalidSection,
    InvalidPayload(String),
    Io(String),
}

impl fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChecksumMismatch => write!(f, "checksum mismatch"),
            Self::WrongSchema => write!(f, "wrong state schema"),
            Self::Truncated => write!(f, "truncated snapshot"),
            Self::WrongPrefixHash => write!(f, "wrong prefix hash"),
            Self::StaleGeneration => write!(f, "stale generation"),
            Self::InvalidSection => write!(f, "invalid section"),
            Self::InvalidPayload(e) => write!(f, "invalid payload: {e}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for SnapshotError {}

impl StateSnapshot {
    pub const FORMAT_VERSION: u32 = 1;
    pub const STATE_ABI_VERSION: u32 = 1;
    pub const MAGIC: [u8; 8] = *b"LOGANST0";

    pub fn new(
        schema_id: StateSchemaId,
        prefix_len: usize,
        prefix_hash: u64,
        state: &CausalState,
    ) -> Result<Self, String> {
        let payload = SnapshotPayload::Raw(encode_causal_state(state)?);
        let checksum = compute_checksum(&payload);
        Ok(Self {
            schema_id,
            prefix_len,
            prefix_hash,
            payload,
            checksum,
            generation: 0,
        })
    }

    pub fn capture<C: CausalStateCodec>(
        codec: &C,
        engine: &C::EngineState,
        prefix_len: usize,
        prefix_hash: u64,
    ) -> Result<Self, String> {
        let state = codec.export_state(engine, prefix_len)?;
        Self::new(codec.schema_id(), prefix_len, prefix_hash, &state)
    }

    pub fn restore_with<C: CausalStateCodec>(
        &self,
        codec: &C,
        engine: &mut C::EngineState,
        expected_prefix_hash: u64,
    ) -> Result<(), String> {
        self.validate().map_err(|e| e.to_string())?;
        if self.schema_id != codec.schema_id() {
            return Err(SnapshotError::WrongSchema.to_string());
        }
        if self.prefix_hash != expected_prefix_hash {
            return Err(SnapshotError::WrongPrefixHash.to_string());
        }
        let state = self.into_causal_state()?;
        codec.import_state(engine, self.prefix_len, &state)
    }

    pub fn payload_bytes(&self) -> usize {
        match &self.payload {
            SnapshotPayload::Raw(bytes) => bytes.len(),
            SnapshotPayload::Sections(sections) => sections.len() * (1 + 8 + 8),
        }
    }

    pub fn validate(&self) -> Result<(), SnapshotError> {
        let expected = compute_checksum(&self.payload);
        if expected != self.checksum {
            return Err(SnapshotError::ChecksumMismatch);
        }
        Ok(())
    }

    pub fn into_causal_state(&self) -> Result<CausalState, String> {
        self.validate().map_err(|e| e.to_string())?;
        match &self.payload {
            SnapshotPayload::Raw(bytes) => decode_causal_state(bytes),
            SnapshotPayload::Sections(_) => Err("section snapshots are metadata-only".into()),
        }
    }

    /// Stable snapshot encoding used by the generic SSD prefix store.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate().map_err(|e| e.to_string())?;
        let engine = self.schema_id.engine.as_bytes();
        let engine_len = u32::try_from(engine.len())
            .map_err(|_| "state schema engine name too long".to_string())?;
        let prefix_len = u64::try_from(self.prefix_len)
            .map_err(|_| "prefix length does not fit snapshot format".to_string())?;

        let (payload_kind, payload) = match &self.payload {
            SnapshotPayload::Raw(bytes) => (0u8, bytes.clone()),
            SnapshotPayload::Sections(sections) => {
                let mut bytes = Vec::with_capacity(sections.len() * 17);
                for section in sections {
                    bytes.push(section.kind.tag());
                    push_u64(&mut bytes, section.offset as u64);
                    push_u64(&mut bytes, section.length as u64);
                }
                (1u8, bytes)
            }
        };

        let mut out = Vec::with_capacity(96 + engine.len() + payload.len());
        out.extend_from_slice(&Self::MAGIC);
        out.extend_from_slice(&Self::FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&engine_len.to_le_bytes());
        out.extend_from_slice(engine);
        out.extend_from_slice(&self.schema_id.version.to_le_bytes());
        out.extend_from_slice(&self.schema_id.sub_version.to_le_bytes());
        push_u64(&mut out, prefix_len);
        push_u64(&mut out, self.prefix_hash);
        push_u64(&mut out, self.generation);
        out.push(payload_kind);
        push_u64(&mut out, payload.len() as u64);
        out.extend_from_slice(&payload);
        out.extend_from_slice(&self.checksum);
        Ok(out)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut cursor = ByteCursor::new(bytes);
        if cursor.take(8)? != Self::MAGIC {
            return Err("bad state snapshot magic".into());
        }
        let format = cursor.u32()?;
        if format != Self::FORMAT_VERSION {
            return Err(format!("unsupported state snapshot version {format}"));
        }
        let engine_len = cursor.u32()? as usize;
        let engine = std::str::from_utf8(cursor.take(engine_len)?)
            .map_err(|_| "state schema engine is not UTF-8".to_string())?
            .to_string();
        let version = cursor.u32()?;
        let sub_version = cursor.u32()?;
        let prefix_len = usize::try_from(cursor.u64()?)
            .map_err(|_| "prefix length exceeds host usize".to_string())?;
        let prefix_hash = cursor.u64()?;
        let generation = cursor.u64()?;
        let payload_kind = cursor.u8()?;
        let payload_len = usize::try_from(cursor.u64()?)
            .map_err(|_| "snapshot payload exceeds host usize".to_string())?;
        let payload_bytes = cursor.take(payload_len)?.to_vec();
        let checksum: [u8; 32] = cursor
            .take(32)?
            .try_into()
            .map_err(|_| "truncated snapshot checksum".to_string())?;
        if !cursor.is_done() {
            return Err("trailing bytes after state snapshot".into());
        }

        let payload = match payload_kind {
            0 => SnapshotPayload::Raw(payload_bytes),
            1 => {
                if payload_bytes.len() % 17 != 0 {
                    return Err("invalid section payload length".into());
                }
                let mut sections = Vec::with_capacity(payload_bytes.len() / 17);
                let mut c = ByteCursor::new(&payload_bytes);
                while !c.is_done() {
                    let kind = SectionKind::from_tag(c.u8()?)
                        .ok_or_else(|| "invalid section kind".to_string())?;
                    let offset = usize::try_from(c.u64()?)
                        .map_err(|_| "section offset exceeds host usize".to_string())?;
                    let length = usize::try_from(c.u64()?)
                        .map_err(|_| "section length exceeds host usize".to_string())?;
                    sections.push(Section {
                        kind,
                        offset,
                        length,
                    });
                }
                SnapshotPayload::Sections(sections)
            }
            _ => return Err(format!("unknown snapshot payload kind {payload_kind}")),
        };

        let snapshot = Self {
            schema_id: StateSchemaId::new(engine, version, sub_version),
            prefix_len,
            prefix_hash,
            payload,
            checksum,
            generation,
        };
        snapshot.validate().map_err(|e| e.to_string())?;
        Ok(snapshot)
    }
}

impl SectionKind {
    pub fn to_be_bytes(self) -> [u8; 1] {
        [self.tag()]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::AppendOnlyKeys => "ak",
            Self::AppendOnlyValues => "av",
            Self::Position => "pos",
            Self::RingBuffer => "rb",
            Self::MutableFixed => "mf",
            Self::PageTable => "pt",
            Self::Opaque => "op",
        }
    }

    fn tag(self) -> u8 {
        match self {
            Self::AppendOnlyKeys => 1,
            Self::AppendOnlyValues => 2,
            Self::Position => 3,
            Self::RingBuffer => 4,
            Self::MutableFixed => 5,
            Self::PageTable => 6,
            Self::Opaque => 7,
        }
    }

    fn from_tag(tag: u8) -> Option<Self> {
        Some(match tag {
            1 => Self::AppendOnlyKeys,
            2 => Self::AppendOnlyValues,
            3 => Self::Position,
            4 => Self::RingBuffer,
            5 => Self::MutableFixed,
            6 => Self::PageTable,
            7 => Self::Opaque,
            _ => return None,
        })
    }
}

const STATE_PAYLOAD_MAGIC: [u8; 4] = *b"LCS1";

fn encode_causal_state(state: &CausalState) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.extend_from_slice(&STATE_PAYLOAD_MAGIC);
    match state {
        CausalState::AppendOnly(s) => {
            out.push(0);
            push_usize(&mut out, s.position)?;
            push_usize(&mut out, s.n_heads)?;
            push_usize(&mut out, s.dim)?;
            push_f32_vec(&mut out, &s.keys)?;
            push_f32_vec(&mut out, &s.values)?;
        }
        CausalState::Ring(s) => {
            out.push(1);
            push_usize(&mut out, s.capacity)?;
            push_usize(&mut out, s.write_pos)?;
            push_usize(&mut out, s.read_pos)?;
            push_usize(&mut out, s.len)?;
            push_f32_vec(&mut out, &s.buffer)?;
        }
        CausalState::MutableFixed(s) => {
            out.push(2);
            push_usize(&mut out, s.write_pos)?;
            push_usize(&mut out, s.len)?;
            out.push(u8::from(s.has_wrapped));
            push_f32_vec(&mut out, &s.data)?;
        }
        CausalState::SparsePaged(s) => {
            out.push(3);
            push_usize(&mut out, s.page_entries)?;
            push_usize(&mut out, s.max_pages)?;
            out.push(u8::from(s.pinned));
            push_usize(&mut out, s.page_table.len())?;
            for page in &s.page_table {
                match page {
                    Some(offset) => {
                        out.push(1);
                        push_u64(&mut out, *offset);
                    }
                    None => out.push(0),
                }
            }
            push_usize(&mut out, s.pages.len())?;
            for page in &s.pages {
                push_f32_vec(&mut out, page)?;
            }
        }
        CausalState::Opaque(data) => {
            out.push(4);
            push_usize(&mut out, data.len())?;
            out.extend_from_slice(data);
        }
    }
    Ok(out)
}

fn decode_causal_state(bytes: &[u8]) -> Result<CausalState, String> {
    let mut c = ByteCursor::new(bytes);
    if c.take(4)? != STATE_PAYLOAD_MAGIC {
        return Err("bad causal-state payload magic".into());
    }
    let tag = c.u8()?;
    let state = match tag {
        0 => CausalState::AppendOnly(AppendOnlyState {
            position: c.usize()?,
            n_heads: c.usize()?,
            dim: c.usize()?,
            keys: c.f32_vec()?,
            values: c.f32_vec()?,
        }),
        1 => CausalState::Ring(RingState {
            capacity: c.usize()?,
            write_pos: c.usize()?,
            read_pos: c.usize()?,
            len: c.usize()?,
            buffer: c.f32_vec()?,
        }),
        2 => CausalState::MutableFixed(MutableFixedState {
            write_pos: c.usize()?,
            len: c.usize()?,
            has_wrapped: match c.u8()? {
                0 => false,
                1 => true,
                _ => return Err("invalid mutable-state wrapped flag".into()),
            },
            data: c.f32_vec()?,
        }),
        3 => {
            let page_entries = c.usize()?;
            let max_pages = c.usize()?;
            let pinned = match c.u8()? {
                0 => false,
                1 => true,
                _ => return Err("invalid sparse-state pinned flag".into()),
            };
            let table_len = c.usize()?;
            let mut page_table = Vec::with_capacity(table_len);
            for _ in 0..table_len {
                page_table.push(match c.u8()? {
                    0 => None,
                    1 => Some(c.u64()?),
                    _ => return Err("invalid sparse page-table tag".into()),
                });
            }
            let page_count = c.usize()?;
            let mut pages = Vec::with_capacity(page_count);
            for _ in 0..page_count {
                pages.push(c.f32_vec()?);
            }
            CausalState::SparsePaged(SparsePagedState {
                page_table,
                pages,
                page_entries,
                max_pages,
                pinned,
            })
        }
        4 => {
            let len = c.usize()?;
            CausalState::Opaque(c.take(len)?.to_vec())
        }
        _ => return Err(format!("unknown causal-state kind {tag}")),
    };
    if !c.is_done() {
        return Err("trailing bytes in causal-state payload".into());
    }
    Ok(state)
}

fn compute_checksum(payload: &SnapshotPayload) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    match payload {
        SnapshotPayload::Raw(bytes) => {
            hasher.update([0]);
            hasher.update(bytes);
        }
        SnapshotPayload::Sections(sections) => {
            hasher.update([1]);
            for section in sections {
                hasher.update([section.kind.tag()]);
                hasher.update((section.offset as u64).to_le_bytes());
                hasher.update((section.length as u64).to_le_bytes());
            }
        }
    }
    hasher.finalize().into()
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_usize(out: &mut Vec<u8>, value: usize) -> Result<(), String> {
    push_u64(
        out,
        u64::try_from(value).map_err(|_| "usize exceeds snapshot format".to_string())?,
    );
    Ok(())
}

fn push_f32_vec(out: &mut Vec<u8>, values: &[f32]) -> Result<(), String> {
    push_usize(out, values.len())?;
    for value in values {
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    Ok(())
}

struct ByteCursor<'a> {
    bytes: &'a [u8],
    off: usize,
}

impl<'a> ByteCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, off: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self
            .off
            .checked_add(len)
            .ok_or_else(|| "snapshot offset overflow".to_string())?;
        let slice = self
            .bytes
            .get(self.off..end)
            .ok_or_else(|| "truncated snapshot payload".to_string())?;
        self.off = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn usize(&mut self) -> Result<usize, String> {
        usize::try_from(self.u64()?).map_err(|_| "snapshot length exceeds host usize".to_string())
    }

    fn f32_vec(&mut self) -> Result<Vec<f32>, String> {
        let len = self.usize()?;
        let bytes_len = len
            .checked_mul(4)
            .ok_or_else(|| "f32 vector byte length overflow".to_string())?;
        let raw = self.take(bytes_len)?;
        Ok(raw
            .chunks_exact(4)
            .map(|chunk| f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap())))
            .collect())
    }

    fn is_done(&self) -> bool {
        self.off == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct FakeEngine(Vec<u8>);

    struct FakeCodec;

    impl CausalStateCodec for FakeCodec {
        type EngineState = FakeEngine;

        fn schema_id(&self) -> StateSchemaId {
            StateSchemaId::new("fake", 1, 0)
        }

        fn export_state(
            &self,
            state: &Self::EngineState,
            _prefix_len: usize,
        ) -> Result<CausalState, String> {
            Ok(CausalState::Opaque(state.0.clone()))
        }

        fn import_state(
            &self,
            state: &mut Self::EngineState,
            _prefix_len: usize,
            causal: &CausalState,
        ) -> Result<(), String> {
            let CausalState::Opaque(bytes) = causal else {
                return Err("fake codec requires opaque state".into());
            };
            state.0.clone_from(bytes);
            Ok(())
        }
    }

    #[test]
    fn structured_state_round_trips_without_type_loss() {
        let state = CausalState::AppendOnly(AppendOnlyState {
            keys: vec![1.0, 2.0],
            values: vec![3.0, 4.0],
            position: 1,
            n_heads: 1,
            dim: 2,
        });
        let snapshot =
            StateSnapshot::new(StateSchemaId::new("test", 1, 0), 1, 123, &state).unwrap();
        assert_eq!(snapshot.into_causal_state().unwrap(), state);
    }

    #[test]
    fn codec_capture_restore_checks_prefix_identity() {
        let source = FakeEngine(vec![1, 2, 3]);
        let snapshot = StateSnapshot::capture(&FakeCodec, &source, 3, 0x55).unwrap();
        let mut target = FakeEngine(vec![9]);
        assert!(
            snapshot
                .restore_with(&FakeCodec, &mut target, 0x56)
                .is_err()
        );
        snapshot
            .restore_with(&FakeCodec, &mut target, 0x55)
            .unwrap();
        assert_eq!(target.0, vec![1, 2, 3]);
    }

    #[test]
    fn snapshot_binary_round_trip_preserves_checksum_and_schema() {
        let state = CausalState::Opaque(vec![7, 8, 9]);
        let mut snapshot =
            StateSnapshot::new(StateSchemaId::new("llama", 2, 3), 4, 0xabc, &state).unwrap();
        snapshot.generation = 11;
        let bytes = snapshot.to_bytes().unwrap();
        let decoded = StateSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, snapshot);
        assert_eq!(decoded.into_causal_state().unwrap(), state);
    }

    #[test]
    fn checksum_detects_payload_corruption() {
        let state = CausalState::Opaque(vec![1, 2, 3]);
        let mut snapshot =
            StateSnapshot::new(StateSchemaId::new("test", 1, 0), 3, 1, &state).unwrap();
        let SnapshotPayload::Raw(bytes) = &mut snapshot.payload else {
            unreachable!();
        };
        bytes[0] ^= 0xff;
        assert_eq!(snapshot.validate(), Err(SnapshotError::ChecksumMismatch));
    }
}
