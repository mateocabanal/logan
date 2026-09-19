//! Metadata-only prefix index with exact token-prefix validation.

use crate::prefix::{PrefixFingerprint, PrefixKey};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_bytes: usize,
    pub max_bytes: usize,
    pub total_lookups: usize,
    pub total_hits: u64,
    pub total_misses: u64,
    pub hit_rate: f64,
}

#[derive(Debug, Clone)]
struct IndexedPrefix {
    key: PrefixKey,
    size: usize,
}

#[derive(Debug, Clone)]
pub struct PrefixIndex {
    entries: BTreeMap<PrefixFingerprint, IndexedPrefix>,
    lru: Vec<PrefixFingerprint>,
    total_bytes: usize,
    max_bytes: usize,
    total_lookups: u64,
    total_hits: u64,
    total_misses: u64,
}

impl PrefixIndex {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            lru: Vec::new(),
            total_bytes: 0,
            max_bytes,
            total_lookups: 0,
            total_hits: 0,
            total_misses: 0,
        }
    }

    pub fn insert(&mut self, key: PrefixKey, size: usize) -> Result<(), String> {
        self.insert_with_evictions(key, size).map(|_| ())
    }

    pub fn insert_with_evictions(
        &mut self,
        key: PrefixKey,
        size: usize,
    ) -> Result<Vec<PrefixFingerprint>, String> {
        if size > self.max_bytes {
            return Err(format!(
                "prefix entry of {size} bytes exceeds cache budget {}",
                self.max_bytes
            ));
        }
        let fingerprint = key.fingerprint();
        self.remove(&fingerprint);
        let evicted = self.evict_for(size);
        self.total_bytes += size;
        self.lru.push(fingerprint);
        self.entries
            .insert(fingerprint, IndexedPrefix { key, size });
        Ok(evicted)
    }

    pub fn lookup(&mut self, query: &PrefixKey) -> Option<PrefixFingerprint> {
        self.total_lookups = self.total_lookups.saturating_add(1);
        let best = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.key.is_prefix_of(query))
            .max_by_key(|(_, entry)| entry.key.prefix_len())
            .map(|(fingerprint, _)| *fingerprint);

        if let Some(fingerprint) = best {
            self.total_hits = self.total_hits.saturating_add(1);
            self.touch(fingerprint);
            Some(fingerprint)
        } else {
            self.total_misses = self.total_misses.saturating_add(1);
            None
        }
    }

    pub fn remove(&mut self, fingerprint: &PrefixFingerprint) -> bool {
        let Some(entry) = self.entries.remove(fingerprint) else {
            return false;
        };
        self.total_bytes = self.total_bytes.saturating_sub(entry.size);
        self.lru.retain(|candidate| candidate != fingerprint);
        true
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            total_entries: self.entries.len(),
            total_bytes: self.total_bytes,
            max_bytes: self.max_bytes,
            total_lookups: self.total_lookups as usize,
            total_hits: self.total_hits,
            total_misses: self.total_misses,
            hit_rate: if self.total_lookups == 0 {
                0.0
            } else {
                self.total_hits as f64 / self.total_lookups as f64
            },
        }
    }

    fn touch(&mut self, fingerprint: PrefixFingerprint) {
        self.lru.retain(|candidate| *candidate != fingerprint);
        self.lru.push(fingerprint);
    }

    fn evict_for(&mut self, incoming: usize) -> Vec<PrefixFingerprint> {
        let mut evicted = Vec::new();
        while self.total_bytes.saturating_add(incoming) > self.max_bytes {
            let Some(oldest) = self.lru.first().copied() else {
                break;
            };
            if self.remove(&oldest) {
                evicted.push(oldest);
            }
        }
        evicted
    }
}

pub trait PrefixLookup {
    fn lookup(&mut self, key: &PrefixKey) -> Option<PrefixFingerprint>;
    fn insert(&mut self, key: PrefixKey, size: usize) -> Result<(), String>;
    fn stats(&self) -> CacheStats;
}

impl PrefixLookup for PrefixIndex {
    fn lookup(&mut self, key: &PrefixKey) -> Option<PrefixFingerprint> {
        PrefixIndex::lookup(self, key)
    }

    fn insert(&mut self, key: PrefixKey, size: usize) -> Result<(), String> {
        PrefixIndex::insert(self, key, size)
    }

    fn stats(&self) -> CacheStats {
        PrefixIndex::stats(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefix::{
        ModelFingerprint, PlanFingerprint, StateSchemaFingerprint, TokenizerFingerprint,
    };

    fn key(id: u8, tokens: &[u32]) -> PrefixKey {
        PrefixKey::new(
            ModelFingerprint { digest: [id; 32] },
            StateSchemaFingerprint { digest: [2; 32] },
            TokenizerFingerprint { digest: [3; 32] },
            PlanFingerprint { digest: [4; 32] },
            tokens.to_vec(),
        )
    }

    #[test]
    fn lookup_returns_longest_actual_token_prefix() {
        let mut index = PrefixIndex::new(1024);
        index.insert(key(1, &[1, 2]), 100).unwrap();
        index.insert(key(1, &[1, 2, 3]), 100).unwrap();
        let hit = index.lookup(&key(1, &[1, 2, 3, 4])).unwrap();
        assert_eq!(hit.prefix_len, 3);
        assert_eq!(index.stats().total_hits, 1);
    }

    #[test]
    fn mismatch_counts_as_miss() {
        let mut index = PrefixIndex::new(1024);
        index.insert(key(1, &[1, 2]), 100).unwrap();
        assert!(index.lookup(&key(2, &[1, 2, 3])).is_none());
        assert_eq!(index.stats().total_misses, 1);
    }

    #[test]
    fn oversized_insert_fails_without_underflow() {
        let mut index = PrefixIndex::new(32);
        assert!(index.insert(key(1, &[1]), 64).is_err());
        assert_eq!(index.stats().total_bytes, 0);
    }

    #[test]
    fn eviction_is_lru() {
        let mut index = PrefixIndex::new(20);
        let a = key(1, &[1]);
        let b = key(1, &[2]);
        let c = key(1, &[3]);
        index.insert(a.clone(), 10).unwrap();
        index.insert(b.clone(), 10).unwrap();
        assert!(index.lookup(&a).is_some());
        index.insert(c, 10).unwrap();
        assert!(index.lookup(&b).is_none());
    }
}
