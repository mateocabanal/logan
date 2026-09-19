//! Persistent SSD prefix store.

use crate::prefix::format;
use crate::prefix::{PrefixFingerprint, PrefixKey};
use crate::state::StateSnapshot;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct SsdPrefixStore {
    root: PathBuf,
}

impl SsdPrefixStore {
    pub fn new<P: AsRef<Path>>(root: P) -> std::io::Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn snapshot_path(&self, fingerprint: &PrefixFingerprint) -> PathBuf {
        let mut hasher = Sha256::new();
        hasher.update(fingerprint.model.digest);
        hasher.update(fingerprint.state_schema.digest);
        hasher.update(fingerprint.tokenizer.digest);
        hasher.update(fingerprint.plan.digest);
        hasher.update(fingerprint.prefix_token_hash.to_le_bytes());
        hasher.update((fingerprint.prefix_len as u64).to_le_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        self.root.join(format!("{}.lpfx", hex::encode(digest)))
    }

    pub fn exists(&self, fingerprint: &PrefixFingerprint) -> bool {
        self.snapshot_path(fingerprint).is_file()
    }

    pub fn write_for(&self, key: &PrefixKey, snapshot: &StateSnapshot) -> std::io::Result<PathBuf> {
        let fingerprint = key.fingerprint();
        let path = self.snapshot_path(&fingerprint);
        let tmp = self.root.join(format!(
            ".{}.{}.tmp",
            path.file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("prefix"),
            std::process::id()
        ));
        let file = File::create(&tmp)?;
        {
            let mut writer = BufWriter::new(file);
            format::write_snapshot(&mut writer, key, snapshot)?;
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }
        fs::rename(&tmp, &path)?;
        Ok(path)
    }

    pub fn read_for(
        &self,
        fingerprint: &PrefixFingerprint,
    ) -> std::io::Result<(PrefixKey, StateSnapshot)> {
        let file = File::open(self.snapshot_path(fingerprint))?;
        let (_, key, snapshot) = format::read_snapshot(&mut BufReader::new(file))?;
        if key.fingerprint() != *fingerprint {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "prefix file identity does not match requested fingerprint",
            ));
        }
        Ok((key, snapshot))
    }

    /// Load valid entries already present in the cache directory.
    ///
    /// Corrupted/truncated files are ignored as cache misses; they cannot
    /// become live state because the container and StateSnapshot checksums are
    /// both verified before an entry is returned.
    pub fn scan_entries(&self) -> std::io::Result<Vec<(PrefixKey, StateSnapshot)>> {
        let mut entries = Vec::new();
        for dir_entry in fs::read_dir(&self.root)? {
            let dir_entry = dir_entry?;
            let path = dir_entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("lpfx") {
                continue;
            }
            let Ok(file) = File::open(&path) else {
                continue;
            };
            let Ok((_, key, snapshot)) = format::read_snapshot(&mut BufReader::new(file)) else {
                continue;
            };
            entries.push((key, snapshot));
        }
        Ok(entries)
    }

    pub fn remove(&self, fingerprint: &PrefixFingerprint) -> std::io::Result<bool> {
        let path = self.snapshot_path(fingerprint);
        match fs::remove_file(path) {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefix::{
        ModelFingerprint, PlanFingerprint, StateSchemaFingerprint, TokenizerFingerprint,
    };
    use crate::state::{CausalState, StateSchemaId};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("logan-prefix-test-{}-{id}", std::process::id()))
    }

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
    fn write_read_and_scan_round_trip() {
        let root = temp_dir();
        let store = SsdPrefixStore::new(&root).unwrap();
        let key = key(&[1, 2, 3]);
        let snapshot = StateSnapshot::new(
            StateSchemaId::new("test", 1, 0),
            key.prefix_len(),
            key.prefix_token_hash,
            &CausalState::Opaque(vec![5, 6]),
        )
        .unwrap();

        let path = store.write_for(&key, &snapshot).unwrap();
        assert!(path.is_file());
        let (read_key, read_snapshot) = store.read_for(&key.fingerprint()).unwrap();
        assert_eq!(read_key, key);
        assert_eq!(read_snapshot, snapshot);
        assert_eq!(store.scan_entries().unwrap().len(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn path_identity_includes_model_contract() {
        let root = temp_dir();
        let store = SsdPrefixStore::new(&root).unwrap();
        let a = key(&[1]);
        let mut b = key(&[1]);
        b.model_fingerprint = ModelFingerprint { digest: [9; 32] };
        assert_ne!(
            store.snapshot_path(&a.fingerprint()),
            store.snapshot_path(&b.fingerprint())
        );
        fs::remove_dir_all(root).unwrap();
    }
}
