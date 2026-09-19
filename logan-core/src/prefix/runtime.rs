//! Model-neutral prefix-cache runtime.
//!
//! The runtime owns lookup policy and RAM/SSD tiers. Engines own only the
//! StateSnapshot codec used to capture and restore causal state.

use crate::prefix::{CacheStats, PrefixIndex, PrefixKey, RamPrefixCache, SsdPrefixStore};
use crate::state::StateSnapshot;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct PrefixRuntimeConfig {
    pub ram_cache_bytes: usize,
    pub ssd_cache_bytes: usize,
    pub ssd_root: Option<PathBuf>,
    pub min_cache_tokens: usize,
    pub writes_enabled: bool,
    pub salt: Vec<u8>,
}

impl Default for PrefixRuntimeConfig {
    fn default() -> Self {
        Self {
            ram_cache_bytes: 512 * 1024 * 1024,
            ssd_cache_bytes: 4 * 1024 * 1024 * 1024,
            ssd_root: None,
            min_cache_tokens: 4,
            writes_enabled: true,
            salt: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrefixLookupResult {
    pub hit: bool,
    pub ram_hit: bool,
    pub ssd_hit: bool,
    pub prefix_len: usize,
    pub remaining_tokens: usize,
    pub snapshot: Option<StateSnapshot>,
}

impl PrefixLookupResult {
    fn miss(query_len: usize) -> Self {
        Self {
            hit: false,
            ram_hit: false,
            ssd_hit: false,
            prefix_len: 0,
            remaining_tokens: query_len,
            snapshot: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrefixRuntime {
    ram_cache: RamPrefixCache<StateSnapshot>,
    ssd_store: Option<SsdPrefixStore>,
    ssd_index: PrefixIndex,
    config: PrefixRuntimeConfig,
}

impl PrefixRuntime {
    pub fn new(config: PrefixRuntimeConfig) -> Result<Self, String> {
        let ssd_store = match &config.ssd_root {
            Some(root) if config.ssd_cache_bytes > 0 => {
                Some(SsdPrefixStore::new(root).map_err(|e| format!("open SSD prefix cache: {e}"))?)
            }
            _ => None,
        };
        let mut runtime = Self {
            ram_cache: RamPrefixCache::new(config.ram_cache_bytes),
            ssd_index: PrefixIndex::new(config.ssd_cache_bytes),
            ssd_store,
            config,
        };
        runtime.rebuild_ssd_index()?;
        Ok(runtime)
    }

    pub fn lookup_prefix(&mut self, query: &PrefixKey) -> Result<PrefixLookupResult, String> {
        let query = self.effective_key(query);
        let query_len = query.prefix_len();

        if let Some(hit) = self.ram_cache.lookup(&query) {
            return Ok(PrefixLookupResult {
                hit: true,
                ram_hit: true,
                ssd_hit: false,
                prefix_len: hit.prefix_len,
                remaining_tokens: query_len.saturating_sub(hit.prefix_len),
                snapshot: Some(hit.value),
            });
        }

        let Some(fingerprint) = self.ssd_index.lookup(&query) else {
            return Ok(PrefixLookupResult::miss(query_len));
        };
        let Some(store) = &self.ssd_store else {
            return Ok(PrefixLookupResult::miss(query_len));
        };

        let (stored_key, snapshot) = match store.read_for(&fingerprint) {
            Ok(entry) => entry,
            Err(error) => {
                // The index may outlive a file that was manually removed or
                // corrupted. Drop the stale metadata and treat it as a miss.
                self.ssd_index.remove(&fingerprint);
                return if error.kind() == std::io::ErrorKind::NotFound
                    || error.kind() == std::io::ErrorKind::InvalidData
                {
                    Ok(PrefixLookupResult::miss(query_len))
                } else {
                    Err(format!("read SSD prefix cache: {error}"))
                };
            }
        };
        if !stored_key.is_prefix_of(&query) {
            self.ssd_index.remove(&fingerprint);
            return Ok(PrefixLookupResult::miss(query_len));
        }

        let size = cache_entry_bytes(&stored_key, &snapshot)?;
        if size <= self.config.ram_cache_bytes {
            let _ = self
                .ram_cache
                .insert(stored_key.clone(), snapshot.clone(), size);
        }
        let prefix_len = stored_key.prefix_len();
        Ok(PrefixLookupResult {
            hit: true,
            ram_hit: false,
            ssd_hit: true,
            prefix_len,
            remaining_tokens: query_len.saturating_sub(prefix_len),
            snapshot: Some(snapshot),
        })
    }

    pub fn cache_prefix(&mut self, key: PrefixKey, snapshot: StateSnapshot) -> Result<(), String> {
        if key.prefix_len() < self.config.min_cache_tokens || !self.config.writes_enabled {
            return Ok(());
        }
        if snapshot.prefix_len != key.prefix_len() || snapshot.prefix_hash != key.prefix_token_hash
        {
            return Err("snapshot prefix identity does not match cache key".into());
        }

        let key = self.effective_key(&key);
        let size = cache_entry_bytes(&key, &snapshot)?;

        if self.config.ram_cache_bytes > 0 && size <= self.config.ram_cache_bytes {
            self.ram_cache.insert(key.clone(), snapshot.clone(), size)?;
        }

        if let Some(store) = &self.ssd_store {
            if size <= self.config.ssd_cache_bytes {
                let evicted = self.ssd_index.insert_with_evictions(key.clone(), size)?;
                for fingerprint in evicted {
                    let _ = store.remove(&fingerprint);
                }
                if let Err(error) = store.write_for(&key, &snapshot) {
                    self.ssd_index.remove(&key.fingerprint());
                    return Err(format!("write SSD prefix cache: {error}"));
                }
            }
        }
        Ok(())
    }

    pub fn ram_stats(&self) -> CacheStats {
        self.ram_cache.stats()
    }

    pub fn ssd_stats(&self) -> CacheStats {
        self.ssd_index.stats()
    }

    pub fn clear_ram(&mut self) {
        self.ram_cache.clear();
    }

    fn effective_key(&self, key: &PrefixKey) -> PrefixKey {
        if self.config.salt.is_empty() {
            key.clone()
        } else {
            key.with_salt(&self.config.salt)
        }
    }

    fn rebuild_ssd_index(&mut self) -> Result<(), String> {
        let Some(store) = &self.ssd_store else {
            return Ok(());
        };
        for (key, snapshot) in store
            .scan_entries()
            .map_err(|e| format!("scan SSD prefix cache: {e}"))?
        {
            let size = cache_entry_bytes(&key, &snapshot)?;
            if size > self.config.ssd_cache_bytes {
                let _ = store.remove(&key.fingerprint());
                continue;
            }
            let evicted = self.ssd_index.insert_with_evictions(key, size)?;
            for fingerprint in evicted {
                let _ = store.remove(&fingerprint);
            }
        }
        Ok(())
    }
}

fn cache_entry_bytes(key: &PrefixKey, snapshot: &StateSnapshot) -> Result<usize, String> {
    let snapshot_bytes = snapshot.to_bytes()?.len();
    key.prefix_len()
        .checked_mul(4)
        .and_then(|tokens| tokens.checked_add(snapshot_bytes))
        .ok_or_else(|| "prefix cache entry size overflow".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefix::{
        ModelFingerprint, PlanFingerprint, StateSchemaFingerprint, TokenizerFingerprint,
    };
    use crate::state::{CausalState, StateSchemaId};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn key(tokens: &[u32]) -> PrefixKey {
        PrefixKey::new(
            ModelFingerprint { digest: [1; 32] },
            StateSchemaFingerprint { digest: [2; 32] },
            TokenizerFingerprint { digest: [3; 32] },
            PlanFingerprint { digest: [4; 32] },
            tokens.to_vec(),
        )
    }

    fn snapshot(key: &PrefixKey, value: u8) -> StateSnapshot {
        StateSnapshot::new(
            StateSchemaId::new("test", 1, 0),
            key.prefix_len(),
            key.prefix_token_hash,
            &CausalState::Opaque(vec![value]),
        )
        .unwrap()
    }

    fn temp_dir() -> PathBuf {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("logan-runtime-prefix-{}-{id}", std::process::id()))
    }

    #[test]
    fn ram_lookup_restores_longest_actual_prefix() {
        let mut runtime = PrefixRuntime::new(PrefixRuntimeConfig {
            min_cache_tokens: 1,
            ..Default::default()
        })
        .unwrap();
        let a = key(&[1, 2]);
        let b = key(&[1, 2, 3]);
        runtime.cache_prefix(a.clone(), snapshot(&a, 1)).unwrap();
        runtime.cache_prefix(b.clone(), snapshot(&b, 2)).unwrap();

        let hit = runtime.lookup_prefix(&key(&[1, 2, 3, 4])).unwrap();
        assert!(hit.ram_hit);
        assert_eq!(hit.prefix_len, 3);
        assert_eq!(hit.remaining_tokens, 1);
    }

    #[test]
    fn ssd_survives_runtime_restart_and_promotes_to_ram() {
        let root = temp_dir();
        let config = PrefixRuntimeConfig {
            ram_cache_bytes: 1024 * 1024,
            ssd_cache_bytes: 1024 * 1024,
            ssd_root: Some(root.clone()),
            min_cache_tokens: 1,
            writes_enabled: true,
            salt: vec![],
        };
        let cached = key(&[5, 6, 7]);
        {
            let mut runtime = PrefixRuntime::new(config.clone()).unwrap();
            runtime
                .cache_prefix(cached.clone(), snapshot(&cached, 9))
                .unwrap();
        }
        let mut runtime = PrefixRuntime::new(config).unwrap();
        let first = runtime.lookup_prefix(&key(&[5, 6, 7, 8])).unwrap();
        assert!(first.ssd_hit);
        let second = runtime.lookup_prefix(&key(&[5, 6, 7, 9])).unwrap();
        assert!(second.ram_hit);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn snapshot_identity_mismatch_is_rejected() {
        let mut runtime = PrefixRuntime::new(PrefixRuntimeConfig {
            min_cache_tokens: 1,
            ..Default::default()
        })
        .unwrap();
        let a = key(&[1, 2]);
        let b = key(&[1, 3]);
        assert!(runtime.cache_prefix(a, snapshot(&b, 1)).is_err());
    }
}
