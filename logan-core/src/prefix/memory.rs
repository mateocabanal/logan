//! Byte-budgeted RAM hot-prefix cache.
//!
//! Policy is model-agnostic: callers supply a value (normally a causal-state
//! snapshot plus any boundary metadata) and the cache owns prefix matching/LRU.

use crate::prefix::{CacheStats, PrefixFingerprint, PrefixKey};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct CacheHit<V> {
    pub fingerprint: PrefixFingerprint,
    pub prefix_len: usize,
    pub value: V,
}

#[derive(Debug, Clone)]
struct CacheEntry<V> {
    key: PrefixKey,
    value: V,
    size: usize,
}

#[derive(Debug, Clone)]
pub struct RamPrefixCache<V> {
    entries: BTreeMap<PrefixFingerprint, CacheEntry<V>>,
    lru: Vec<PrefixFingerprint>,
    total_bytes: usize,
    max_bytes: usize,
    total_lookups: u64,
    total_hits: u64,
    total_misses: u64,
}

impl<V: Clone> RamPrefixCache<V> {
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

    pub fn lookup(&mut self, query: &PrefixKey) -> Option<CacheHit<V>> {
        self.total_lookups = self.total_lookups.saturating_add(1);
        let best = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.key.is_prefix_of(query))
            .max_by_key(|(_, entry)| entry.key.prefix_len())
            .map(|(fingerprint, entry)| {
                (*fingerprint, entry.key.prefix_len(), entry.value.clone())
            });

        if let Some((fingerprint, prefix_len, value)) = best {
            self.total_hits = self.total_hits.saturating_add(1);
            self.touch(fingerprint);
            Some(CacheHit {
                fingerprint,
                prefix_len,
                value,
            })
        } else {
            self.total_misses = self.total_misses.saturating_add(1);
            None
        }
    }

    pub fn insert(&mut self, key: PrefixKey, value: V, size: usize) -> Result<(), String> {
        if size > self.max_bytes {
            return Err(format!(
                "prefix entry of {size} bytes exceeds RAM cache budget {}",
                self.max_bytes
            ));
        }
        let fingerprint = key.fingerprint();
        self.remove(&fingerprint);
        self.evict_for(size);
        self.total_bytes += size;
        self.lru.push(fingerprint);
        self.entries
            .insert(fingerprint, CacheEntry { key, value, size });
        Ok(())
    }

    pub fn remove(&mut self, fingerprint: &PrefixFingerprint) -> bool {
        let Some(entry) = self.entries.remove(fingerprint) else {
            return false;
        };
        self.total_bytes = self.total_bytes.saturating_sub(entry.size);
        self.lru.retain(|candidate| candidate != fingerprint);
        true
    }

    pub fn capacity_bytes(&self) -> usize {
        self.max_bytes
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

    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
        self.total_bytes = 0;
    }

    fn touch(&mut self, fingerprint: PrefixFingerprint) {
        self.lru.retain(|candidate| *candidate != fingerprint);
        self.lru.push(fingerprint);
    }

    fn evict_for(&mut self, incoming: usize) {
        while self.total_bytes.saturating_add(incoming) > self.max_bytes {
            let Some(oldest) = self.lru.first().copied() else {
                break;
            };
            self.remove(&oldest);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefix::{
        ModelFingerprint, PlanFingerprint, StateSchemaFingerprint, TokenizerFingerprint,
    };

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
    fn longest_prefix_returns_value() {
        let mut cache = RamPrefixCache::new(1024);
        cache.insert(key(&[1, 2]), "short", 100).unwrap();
        cache.insert(key(&[1, 2, 3]), "long", 100).unwrap();

        let hit = cache.lookup(&key(&[1, 2, 3, 4])).unwrap();
        assert_eq!(hit.prefix_len, 3);
        assert_eq!(hit.value, "long");
    }

    #[test]
    fn miss_accounting_survives_eviction() {
        let mut cache = RamPrefixCache::new(10);
        assert!(cache.lookup(&key(&[9])).is_none());
        cache.insert(key(&[1]), 1u8, 10).unwrap();
        cache.insert(key(&[2]), 2u8, 10).unwrap();
        let stats = cache.stats();
        assert_eq!(stats.total_misses, 1);
        assert_eq!(stats.total_entries, 1);
        assert_eq!(stats.total_bytes, 10);
    }

    #[test]
    fn oversized_entry_is_rejected() {
        let mut cache = RamPrefixCache::new(4);
        assert!(cache.insert(key(&[1]), (), 5).is_err());
        assert_eq!(cache.stats().total_bytes, 0);
    }

    #[test]
    fn clear_preserves_lifetime_telemetry() {
        let mut cache = RamPrefixCache::new(64);
        cache.insert(key(&[1]), 1u8, 1).unwrap();
        assert!(cache.lookup(&key(&[1, 2])).is_some());
        cache.clear();
        assert_eq!(cache.stats().total_entries, 0);
        assert_eq!(cache.stats().total_hits, 1);
    }
}
