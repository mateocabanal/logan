//! Safe reader/writer for Logan's canonical `.logan` v1 artifact framing.
//!
//! This crate deliberately does not know model architectures or execution
//! kernels. It validates the immutable container contract and exposes bounded
//! section/file metadata to compiler and runtime crates.
#![forbid(unsafe_code)]

use std::fmt;

pub const MANIFEST_MAGIC: &[u8; 10] = b"LOGANMODEL";
pub const TRAILER_MAGIC: &[u8; 8] = b"LOGANEND";
pub const FORMAT_MAJOR: u16 = 1;
pub const FORMAT_MINOR: u16 = 0;
pub const HEADER_BYTES: usize = 64;
pub const SECTION_ENTRY_BYTES: usize = 32;
pub const FILE_ENTRY_BYTES: usize = 24;
pub const TRAILER_BYTES: usize = 16;

pub mod section {
    pub const STRINGS: u16 = 0x0001;
    pub const IDENTITY: u16 = 0x0002;
    pub const CAPS: u16 = 0x0003;
    pub const SEGMENTS: u16 = 0x0004;
    pub const SYMBOLS: u16 = 0x0005;
    pub const STATE_REQ: u16 = 0x0006;
    pub const COSTS: u16 = 0x0007;
    pub const MODEL: u16 = 0x0008;
    pub const VARIANTS: u16 = 0x0009;
}

pub mod file_kind {
    pub const MANIFEST: u32 = 0;
    pub const PLAN: u32 = 1;
    pub const PAYLOAD: u32 = 2;
    pub const JOURNAL: u32 = 3;
    pub const PROVENANCE: u32 = 4;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactError {
    Truncated(&'static str),
    Invalid(&'static str),
    InvalidDetail(String),
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated(what) => write!(f, "truncated {what}"),
            Self::Invalid(what) => write!(f, "invalid {what}"),
            Self::InvalidDetail(what) => f.write_str(what),
        }
    }
}

impl std::error::Error for ArtifactError {}

pub type Result<T> = std::result::Result<T, ArtifactError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManifestHeader {
    pub format_major: u16,
    pub format_minor: u16,
    pub min_reader_major: u16,
    pub min_reader_minor: u16,
    pub flags: u32,
    pub generation: u64,
    pub section_count: u32,
    pub file_count: u32,
    pub section_table_offset: u64,
    pub file_table_offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionEntry {
    pub kind: u16,
    pub entry_flags: u16,
    pub offset: u64,
    pub length: u64,
    pub crc32c: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileEntry {
    pub file_id: u32,
    pub name_id: u32,
    pub file_bytes: u64,
    pub file_kind: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub header: ManifestHeader,
    pub sections: Vec<SectionEntry>,
    pub files: Vec<FileEntry>,
    bytes: Vec<u8>,
}

fn take<const N: usize>(bytes: &[u8], at: usize, what: &'static str) -> Result<[u8; N]> {
    let end = at
        .checked_add(N)
        .ok_or(ArtifactError::Invalid("offset overflow"))?;
    let src = bytes.get(at..end).ok_or(ArtifactError::Truncated(what))?;
    Ok(src.try_into().expect("slice length checked"))
}

fn u16_at(bytes: &[u8], at: usize, what: &'static str) -> Result<u16> {
    Ok(u16::from_le_bytes(take(bytes, at, what)?))
}

fn u32_at(bytes: &[u8], at: usize, what: &'static str) -> Result<u32> {
    Ok(u32::from_le_bytes(take(bytes, at, what)?))
}

fn u64_at(bytes: &[u8], at: usize, what: &'static str) -> Result<u64> {
    Ok(u64::from_le_bytes(take(bytes, at, what)?))
}

fn checked_table(
    bytes_len: usize,
    offset: u64,
    count: u32,
    entry_bytes: usize,
    what: &'static str,
) -> Result<std::ops::Range<usize>> {
    let start = usize::try_from(offset).map_err(|_| ArtifactError::Invalid("offset too large"))?;
    let len = usize::try_from(count)
        .ok()
        .and_then(|n| n.checked_mul(entry_bytes))
        .ok_or(ArtifactError::Invalid("table size overflow"))?;
    let end = start
        .checked_add(len)
        .ok_or(ArtifactError::Invalid("table end overflow"))?;
    if end > bytes_len {
        return Err(ArtifactError::Truncated(what));
    }
    Ok(start..end)
}

impl Manifest {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_BYTES + TRAILER_BYTES {
            return Err(ArtifactError::Truncated("manifest"));
        }
        if bytes.get(..MANIFEST_MAGIC.len()) != Some(MANIFEST_MAGIC.as_slice()) {
            return Err(ArtifactError::Invalid("manifest magic"));
        }

        let format_major = u16_at(bytes, 10, "format major")?;
        let format_minor = u16_at(bytes, 12, "format minor")?;
        let header_bytes = u32_at(bytes, 14, "header size")?;
        let min_reader_major = u16_at(bytes, 18, "minimum reader major")?;
        let min_reader_minor = u16_at(bytes, 20, "minimum reader minor")?;
        let flags = u32_at(bytes, 22, "flags")?;
        let generation = u64_at(bytes, 26, "generation")?;
        let section_count = u32_at(bytes, 34, "section count")?;
        let file_count = u32_at(bytes, 38, "file count")?;
        let section_table_offset = u64_at(bytes, 42, "section table offset")?;
        let file_table_offset = u64_at(bytes, 50, "file table offset")?;
        let expected_crc = u32_at(bytes, 58, "header crc")?;
        let reserved = u16_at(bytes, 62, "header reserved")?;

        if format_major != FORMAT_MAJOR {
            return Err(ArtifactError::InvalidDetail(format!(
                "unsupported .logan major version {format_major}"
            )));
        }
        if header_bytes != HEADER_BYTES as u32 {
            return Err(ArtifactError::Invalid("manifest header size"));
        }
        if min_reader_major > FORMAT_MAJOR
            || (min_reader_major == FORMAT_MAJOR && min_reader_minor > FORMAT_MINOR)
        {
            return Err(ArtifactError::InvalidDetail(format!(
                ".logan requires reader {min_reader_major}.{min_reader_minor}"
            )));
        }
        if generation == 0 {
            return Err(ArtifactError::Invalid("zero generation"));
        }
        if reserved != 0 {
            return Err(ArtifactError::Invalid("manifest header reserved field"));
        }
        let actual_crc = crc32c::crc32c(&bytes[..58]);
        if actual_crc != expected_crc {
            return Err(ArtifactError::InvalidDetail(format!(
                "manifest header CRC32C mismatch: stored {expected_crc:#010x}, computed {actual_crc:#010x}"
            )));
        }

        let section_range = checked_table(
            bytes.len(),
            section_table_offset,
            section_count,
            SECTION_ENTRY_BYTES,
            "section table",
        )?;
        let file_range = checked_table(
            bytes.len(),
            file_table_offset,
            file_count,
            FILE_ENTRY_BYTES,
            "file table",
        )?;
        if section_range.start < HEADER_BYTES || file_range.start < HEADER_BYTES {
            return Err(ArtifactError::Invalid("table overlaps fixed header"));
        }

        let trailer_at = bytes.len() - TRAILER_BYTES;
        if bytes.get(trailer_at..trailer_at + 8) != Some(TRAILER_MAGIC.as_slice()) {
            return Err(ArtifactError::Invalid("manifest trailer magic"));
        }
        if u32_at(bytes, trailer_at + 12, "trailer reserved")? != 0 {
            return Err(ArtifactError::Invalid("manifest trailer reserved field"));
        }

        let mut sections = Vec::with_capacity(section_count as usize);
        let mut last_kind = None;
        for i in 0..section_count as usize {
            let at = section_range.start + i * SECTION_ENTRY_BYTES;
            let kind = u16_at(bytes, at, "section kind")?;
            let entry_flags = u16_at(bytes, at + 2, "section flags")?;
            if u32_at(bytes, at + 4, "section reserved")? != 0
                || u32_at(bytes, at + 28, "section reserved2")? != 0
            {
                return Err(ArtifactError::Invalid("section reserved field"));
            }
            if last_kind.is_some_and(|previous| kind <= previous) {
                return Err(ArtifactError::Invalid(
                    "section table must be strictly ordered by kind",
                ));
            }
            last_kind = Some(kind);
            let offset = u64_at(bytes, at + 8, "section offset")?;
            let length = u64_at(bytes, at + 16, "section length")?;
            let crc32c = u32_at(bytes, at + 24, "section crc")?;
            let start = usize::try_from(offset)
                .map_err(|_| ArtifactError::Invalid("section offset too large"))?;
            let len = usize::try_from(length)
                .map_err(|_| ArtifactError::Invalid("section length too large"))?;
            let end = start
                .checked_add(len)
                .ok_or(ArtifactError::Invalid("section range overflow"))?;
            if end > trailer_at {
                return Err(ArtifactError::Truncated("section payload"));
            }
            if ranges_overlap(start..end, section_range.clone())
                || ranges_overlap(start..end, file_range.clone())
                || start < HEADER_BYTES
            {
                return Err(ArtifactError::Invalid("section overlaps metadata"));
            }
            let actual = crc32c::crc32c(&bytes[start..end]);
            if actual != crc32c {
                return Err(ArtifactError::InvalidDetail(format!(
                    "section {kind:#06x} CRC32C mismatch: stored {crc32c:#010x}, computed {actual:#010x}"
                )));
            }
            sections.push(SectionEntry {
                kind,
                entry_flags,
                offset,
                length,
                crc32c,
            });
        }

        let mut files = Vec::with_capacity(file_count as usize);
        for i in 0..file_count as usize {
            let at = file_range.start + i * FILE_ENTRY_BYTES;
            let file_id = u32_at(bytes, at, "file id")?;
            if file_id != i as u32 {
                return Err(ArtifactError::Invalid(
                    "file table must be ordered densely by file_id",
                ));
            }
            let name_id = u32_at(bytes, at + 4, "file name id")?;
            let file_bytes = u64_at(bytes, at + 8, "file bytes")?;
            let file_kind = u32_at(bytes, at + 16, "file kind")?;
            if u32_at(bytes, at + 20, "file reserved")? != 0 {
                return Err(ArtifactError::Invalid("file reserved field"));
            }
            if file_kind > file_kind::PROVENANCE {
                return Err(ArtifactError::Invalid("unknown file kind"));
            }
            files.push(FileEntry {
                file_id,
                name_id,
                file_bytes,
                file_kind,
            });
        }

        if let Some(self_file) = files
            .iter()
            .find(|entry| entry.file_kind == file_kind::MANIFEST)
        {
            if self_file.file_bytes != bytes.len() as u64 {
                return Err(ArtifactError::Invalid(
                    "manifest file table size does not match actual file",
                ));
            }
        }

        Ok(Self {
            header: ManifestHeader {
                format_major,
                format_minor,
                min_reader_major,
                min_reader_minor,
                flags,
                generation,
                section_count,
                file_count,
                section_table_offset,
                file_table_offset,
            },
            sections,
            files,
            bytes: bytes.to_vec(),
        })
    }

    pub fn section(&self, kind: u16) -> Option<&[u8]> {
        let entry = self.sections.iter().find(|entry| entry.kind == kind)?;
        let start = usize::try_from(entry.offset).ok()?;
        let len = usize::try_from(entry.length).ok()?;
        self.bytes.get(start..start.checked_add(len)?)
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

fn ranges_overlap(a: std::ops::Range<usize>, b: std::ops::Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}

#[derive(Debug, Clone)]
pub struct ManifestBuilder {
    flags: u32,
    generation: u64,
    min_reader: (u16, u16),
    sections: Vec<(u16, u16, Vec<u8>)>,
    files: Vec<FileEntry>,
}

impl ManifestBuilder {
    pub fn new(generation: u64) -> Result<Self> {
        if generation == 0 {
            return Err(ArtifactError::Invalid("zero generation"));
        }
        Ok(Self {
            flags: 0,
            generation,
            min_reader: (FORMAT_MAJOR, FORMAT_MINOR),
            sections: Vec::new(),
            files: Vec::new(),
        })
    }

    pub fn flags(mut self, flags: u32) -> Self {
        self.flags = flags;
        self
    }

    pub fn min_reader(mut self, major: u16, minor: u16) -> Self {
        self.min_reader = (major, minor);
        self
    }

    pub fn add_section(&mut self, kind: u16, entry_flags: u16, payload: Vec<u8>) -> Result<()> {
        if self
            .sections
            .iter()
            .any(|(existing, _, _)| *existing == kind)
        {
            return Err(ArtifactError::Invalid("duplicate section kind"));
        }
        self.sections.push((kind, entry_flags, payload));
        Ok(())
    }

    /// Add an external file. The MANIFEST/self file is synthesized by encode
    /// as file id 0, so external file ids begin at 1.
    pub fn add_file(&mut self, name_id: u32, file_bytes: u64, file_kind: u32) -> Result<u32> {
        if file_kind == file_kind::MANIFEST || file_kind > file_kind::PROVENANCE {
            return Err(ArtifactError::Invalid("invalid external file kind"));
        }
        let file_id = u32::try_from(self.files.len() + 1)
            .map_err(|_| ArtifactError::Invalid("too many files"))?;
        self.files.push(FileEntry {
            file_id,
            name_id,
            file_bytes,
            file_kind,
        });
        Ok(file_id)
    }

    pub fn encode(mut self) -> Result<Vec<u8>> {
        self.sections.sort_unstable_by_key(|(kind, _, _)| *kind);
        let section_count = u32::try_from(self.sections.len())
            .map_err(|_| ArtifactError::Invalid("too many sections"))?;
        let file_count = u32::try_from(self.files.len() + 1)
            .map_err(|_| ArtifactError::Invalid("too many files"))?;
        let section_table_offset = HEADER_BYTES;
        let section_table_bytes = self
            .sections
            .len()
            .checked_mul(SECTION_ENTRY_BYTES)
            .ok_or(ArtifactError::Invalid("section table size overflow"))?;
        let file_table_offset = section_table_offset
            .checked_add(section_table_bytes)
            .ok_or(ArtifactError::Invalid("file table offset overflow"))?;
        let file_table_bytes = (self.files.len() + 1)
            .checked_mul(FILE_ENTRY_BYTES)
            .ok_or(ArtifactError::Invalid("file table size overflow"))?;
        let mut payload_at = file_table_offset
            .checked_add(file_table_bytes)
            .ok_or(ArtifactError::Invalid("payload offset overflow"))?;

        let payload_bytes = self
            .sections
            .iter()
            .try_fold(0usize, |sum, (_, _, payload)| {
                sum.checked_add(payload.len())
            })
            .ok_or(ArtifactError::Invalid("manifest payload size overflow"))?;
        let total = payload_at
            .checked_add(payload_bytes)
            .and_then(|n| n.checked_add(TRAILER_BYTES))
            .ok_or(ArtifactError::Invalid("manifest size overflow"))?;

        let mut out = vec![0u8; total];
        out[..10].copy_from_slice(MANIFEST_MAGIC);
        put_u16(&mut out, 10, FORMAT_MAJOR);
        put_u16(&mut out, 12, FORMAT_MINOR);
        put_u32(&mut out, 14, HEADER_BYTES as u32);
        put_u16(&mut out, 18, self.min_reader.0);
        put_u16(&mut out, 20, self.min_reader.1);
        put_u32(&mut out, 22, self.flags);
        put_u64(&mut out, 26, self.generation);
        put_u32(&mut out, 34, section_count);
        put_u32(&mut out, 38, file_count);
        put_u64(&mut out, 42, section_table_offset as u64);
        put_u64(&mut out, 50, file_table_offset as u64);
        put_u16(&mut out, 62, 0);

        for (i, (kind, entry_flags, payload)) in self.sections.iter().enumerate() {
            let at = section_table_offset + i * SECTION_ENTRY_BYTES;
            put_u16(&mut out, at, *kind);
            put_u16(&mut out, at + 2, *entry_flags);
            put_u64(&mut out, at + 8, payload_at as u64);
            put_u64(&mut out, at + 16, payload.len() as u64);
            put_u32(&mut out, at + 24, crc32c::crc32c(payload));
            let end = payload_at + payload.len();
            out[payload_at..end].copy_from_slice(payload);
            payload_at = end;
        }

        // File id 0 is the manifest itself. name_id=0 is valid only when the
        // eventual STRINGS section chooses it; the core framing does not assign
        // string semantics.
        put_file_entry(
            &mut out,
            file_table_offset,
            FileEntry {
                file_id: 0,
                name_id: 0,
                file_bytes: total as u64,
                file_kind: file_kind::MANIFEST,
            },
        );
        for (i, entry) in self.files.iter().copied().enumerate() {
            put_file_entry(
                &mut out,
                file_table_offset + (i + 1) * FILE_ENTRY_BYTES,
                entry,
            );
        }

        let header_crc = crc32c::crc32c(&out[..58]);
        put_u32(&mut out, 58, header_crc);
        let trailer_at = total - TRAILER_BYTES;
        out[trailer_at..trailer_at + 8].copy_from_slice(TRAILER_MAGIC);
        // The v1 design reserves a trailer CRC field but does not yet specify
        // its coverage. Emit zero rather than inventing an incompatible rule.
        put_u32(&mut out, trailer_at + 8, 0);
        put_u32(&mut out, trailer_at + 12, 0);

        // Self-check keeps the writer and reader contracts from drifting.
        Manifest::parse(&out)?;
        Ok(out)
    }
}

fn put_file_entry(out: &mut [u8], at: usize, entry: FileEntry) {
    put_u32(out, at, entry.file_id);
    put_u32(out, at + 4, entry.name_id);
    put_u64(out, at + 8, entry.file_bytes);
    put_u32(out, at + 16, entry.file_kind);
    put_u32(out, at + 20, 0);
}

fn put_u16(out: &mut [u8], at: usize, value: u16) {
    out[at..at + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(out: &mut [u8], at: usize, value: u32) {
    out[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut [u8], at: usize, value: u64) {
    out[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_round_trip_is_byte_stable() {
        let mut builder = ManifestBuilder::new(1)
            .unwrap()
            .flags(0b110)
            .min_reader(1, 0);
        builder
            .add_section(section::STRINGS, 0, b"hello".to_vec())
            .unwrap();
        builder
            .add_section(section::COSTS, 0, vec![1, 2, 3, 4, 5, 6])
            .unwrap();
        builder.add_file(1, 4096, file_kind::PLAN).unwrap();
        builder.add_file(2, 8192, file_kind::PAYLOAD).unwrap();

        let bytes = builder.clone().encode().unwrap();
        let parsed = Manifest::parse(&bytes).unwrap();
        assert_eq!(parsed.header.format_major, 1);
        assert_eq!(parsed.header.generation, 1);
        assert_eq!(parsed.sections.len(), 2);
        assert_eq!(parsed.files.len(), 3);
        assert_eq!(parsed.section(section::STRINGS), Some(&b"hello"[..]));

        // Rebuilding the same logical manifest is deterministic.
        assert_eq!(bytes, builder.encode().unwrap());
    }

    #[test]
    fn rejects_truncated_tables_before_allocating_from_counts() {
        let mut bytes = ManifestBuilder::new(1).unwrap().encode().unwrap();
        put_u32(&mut bytes, 34, u32::MAX);
        let crc = crc32c::crc32c(&bytes[..58]);
        put_u32(&mut bytes, 58, crc);
        assert!(matches!(
            Manifest::parse(&bytes),
            Err(ArtifactError::Truncated("section table"))
                | Err(ArtifactError::Invalid("table size overflow"))
                | Err(ArtifactError::Invalid("table end overflow"))
        ));
    }

    #[test]
    fn rejects_section_crc_corruption() {
        let mut builder = ManifestBuilder::new(1).unwrap();
        builder
            .add_section(section::MODEL, 0, b"model".to_vec())
            .unwrap();
        let mut bytes = builder.encode().unwrap();
        let parsed = Manifest::parse(&bytes).unwrap();
        let offset = parsed.sections[0].offset as usize;
        bytes[offset] ^= 0x80;
        assert!(matches!(
            Manifest::parse(&bytes),
            Err(ArtifactError::InvalidDetail(_))
        ));
    }

    #[test]
    fn rejects_section_metadata_overlap() {
        let mut builder = ManifestBuilder::new(1).unwrap();
        builder
            .add_section(section::MODEL, 0, b"model".to_vec())
            .unwrap();
        let mut bytes = builder.encode().unwrap();
        let table = HEADER_BYTES;
        put_u64(&mut bytes, table + 8, HEADER_BYTES as u64);
        let fake_crc = crc32c::crc32c(&bytes[HEADER_BYTES..HEADER_BYTES + 5]);
        put_u32(&mut bytes, table + 24, fake_crc);
        assert!(matches!(
            Manifest::parse(&bytes),
            Err(ArtifactError::Invalid("section overlaps metadata"))
        ));
    }

    #[test]
    fn rejects_unsorted_or_duplicate_section_kinds() {
        let mut builder = ManifestBuilder::new(1).unwrap();
        builder.add_section(1, 0, vec![1]).unwrap();
        builder.add_section(2, 0, vec![2]).unwrap();
        let mut bytes = builder.encode().unwrap();
        put_u16(&mut bytes, HEADER_BYTES + SECTION_ENTRY_BYTES, 1);
        assert!(matches!(
            Manifest::parse(&bytes),
            Err(ArtifactError::Invalid(
                "section table must be strictly ordered by kind"
            ))
        ));
    }
}
