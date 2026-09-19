use crate::{AneError, Result};

const ALIGNMENT: usize = 64;
const HEADER_BYTES: usize = 64;
const METADATA_BYTES: usize = 64;
const SENTINEL: u32 = 0xDEAD_BEEF;
const VERSION: u32 = 2;

/// Element type stored in a CoreML MIL Blob v2 weight record.
///
/// The discriminants mirror CoreML's private/semi-private BlobDataType values
/// used by `weight.bin`. Keep these separate from MIL protobuf DataType values:
/// the two enums intentionally use different numeric assignments.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlobDataType {
    Float16 = 1,
    Float32 = 2,
    UInt8 = 3,
    Int8 = 4,
    BFloat16 = 5,
    Int16 = 6,
    UInt16 = 7,
    Int4 = 8,
    UInt1 = 9,
    UInt2 = 10,
    UInt4 = 11,
    UInt3 = 12,
    UInt6 = 13,
    Int32 = 14,
    UInt32 = 15,
    Float8E4M3Fn = 16,
    Float8E5M2 = 17,
}

/// Byte offset returned by [`BlobV2Builder::push`].
///
/// This points at the record's 64-byte metadata block and is the number that
/// belongs in MIL's `BLOBFILE(..., offset = uint64(...))` expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlobOffset(u64);

impl BlobOffset {
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Builder for CoreML MIL `weight.bin` Blob v2 files.
///
/// Records are 64-byte aligned and consist of a 64-byte metadata block followed
/// by the raw tensor payload. `push` returns the metadata offset used by MIL.
#[derive(Clone, Debug)]
pub struct BlobV2Builder {
    bytes: Vec<u8>,
    count: u32,
}

impl Default for BlobV2Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl BlobV2Builder {
    pub fn new() -> Self {
        let mut bytes = vec![0u8; HEADER_BYTES];
        bytes[4..8].copy_from_slice(&VERSION.to_le_bytes());
        Self { bytes, count: 0 }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn record_count(&self) -> u32 {
        self.count
    }

    /// Append one raw tensor record and return its MIL BLOBFILE offset.
    pub fn push(&mut self, data_type: BlobDataType, payload: &[u8]) -> Result<BlobOffset> {
        if payload.is_empty() {
            return Err(AneError::InvalidArgument(
                "Blob v2 record payload must be non-empty".into(),
            ));
        }
        if self.count == u32::MAX {
            return Err(AneError::InvalidArgument(
                "Blob v2 record count exceeds u32::MAX".into(),
            ));
        }

        align_vec(&mut self.bytes, ALIGNMENT);
        let metadata_offset = self.bytes.len();
        let payload_offset = metadata_offset
            .checked_add(METADATA_BYTES)
            .ok_or_else(|| AneError::InvalidArgument("Blob v2 offset overflow".into()))?;
        let payload_len = u64::try_from(payload.len())
            .map_err(|_| AneError::InvalidArgument("Blob v2 payload is too large".into()))?;
        let payload_offset_u64 = u64::try_from(payload_offset)
            .map_err(|_| AneError::InvalidArgument("Blob v2 offset is too large".into()))?;

        self.bytes.resize(payload_offset, 0);
        let meta = &mut self.bytes[metadata_offset..payload_offset];
        meta[0..4].copy_from_slice(&SENTINEL.to_le_bytes());
        meta[4..8].copy_from_slice(&(data_type as u32).to_le_bytes());
        meta[8..16].copy_from_slice(&payload_len.to_le_bytes());
        meta[16..24].copy_from_slice(&payload_offset_u64.to_le_bytes());
        self.bytes.extend_from_slice(payload);

        self.count += 1;
        self.bytes[0..4].copy_from_slice(&self.count.to_le_bytes());

        Ok(BlobOffset(metadata_offset as u64))
    }

    /// Append little-endian IEEE fp16 values.
    pub fn push_fp16(&mut self, values: &[u16]) -> Result<BlobOffset> {
        if values.is_empty() {
            return Err(AneError::InvalidArgument(
                "fp16 Blob v2 tensor must be non-empty".into(),
            ));
        }
        let byte_len = values
            .len()
            .checked_mul(2)
            .ok_or_else(|| AneError::InvalidArgument("fp16 byte count overflow".into()))?;
        let mut payload = Vec::with_capacity(byte_len);
        for &value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        self.push(BlobDataType::Float16, &payload)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

fn align_vec(bytes: &mut Vec<u8>, alignment: usize) {
    let remainder = bytes.len() % alignment;
    if remainder != 0 {
        bytes.resize(bytes.len() + (alignment - remainder), 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u32_at(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    fn u64_at(bytes: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
    }

    #[test]
    fn writes_single_fp16_record() {
        let mut blob = BlobV2Builder::new();
        let offset = blob.push_fp16(&[0x3c00, 0xbc00]).unwrap();
        assert_eq!(offset.get(), 64);
        let bytes = blob.as_bytes();
        assert_eq!(u32_at(bytes, 0), 1);
        assert_eq!(u32_at(bytes, 4), 2);
        assert_eq!(u32_at(bytes, 64), SENTINEL);
        assert_eq!(u32_at(bytes, 68), BlobDataType::Float16 as u32);
        assert_eq!(u64_at(bytes, 72), 4);
        assert_eq!(u64_at(bytes, 80), 128);
        assert_eq!(&bytes[128..132], &[0x00, 0x3c, 0x00, 0xbc]);
    }

    #[test]
    fn aligns_multiple_records_and_updates_count() {
        let mut blob = BlobV2Builder::new();
        let first = blob.push(BlobDataType::UInt8, &[1, 2, 3]).unwrap();
        let second = blob.push(BlobDataType::Float32, &[0; 8]).unwrap();
        assert_eq!(first.get(), 64);
        assert_eq!(second.get(), 192);
        assert_eq!(blob.record_count(), 2);
        assert_eq!(u32_at(blob.as_bytes(), 0), 2);
        assert_eq!(u64_at(blob.as_bytes(), 192 + 16), 256);
    }

    #[test]
    fn rejects_empty_payload() {
        let mut blob = BlobV2Builder::new();
        assert!(blob.push(BlobDataType::Float16, &[]).is_err());
    }
}
