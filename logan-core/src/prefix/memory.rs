//! RAM hot prefix cache.

use crate::prefix::{PrefixFingerprint, PrefixKey, CacheStats};
use std::collections::BTreeMap;

/// RAM prefix cache: hot in-memory cache for recently used prefixes.
#[derive(Debug)]
pub struct RamPrefixCache {
    /// Cache entries by fingerprint
    entries: BTreeMap<PrefixFingerprint, CacheEntry>,
    /// LRU order
    lru: Vec<PrefixFingerprint>,
    /// Total bytes in cache
    total_bytes: usize,
    /// Maximum bytes
    max_bytes: usize,
    /// Next timestamp
    next_timestamp: u64,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    /// Prefix token count
    prefix_len: usize,
    /// Size in bytes
    size: usize,
    /// Timestamp of last access
    last_access: u64,
    /// Timestamp of last update
    last_update: u64,
    /// Number of hits
    hits: u64,
    /// Number of misses
    misses: u64,
}

impl RamPrefixCache {
    /// Create a new RAM prefix cache with given byte budget
    pub fn new(max_bytes: usize) -> Self {
        RamPrefixCache {
            entries: BTreeMap::new(),
            lru: Vec::new(),
            total_bytes: 0,
            max_bytes,
            next_timestamp: 0,
        }
    }

    /// Look up a prefix in the RAM cache
    pub fn lookup(&mut self, key: &PrefixKey) -> Option<PrefixFingerprint> {
        let query = PrefixFingerprint::new(
            key.model_fingerprint,
            key.state_schema_fingerprint,
            key.tokenizer_fingerprint,
            key.plan_fingerprint,
            key.prefix_token_hash,
            key.prefix_len,
        );

        // Find longest compatible prefix
        let mut best: Option<&PrefixFingerprint> = None;
        for (fp, entry) in &self.entries {
            if fp.is_compatible_prefix(&query) {
                if best.map_or(true, |b| fp.prefix_len > b.prefix_len) {
                    best = Some(fp);
                }
            }
        }

        if let Some(fp_ref) = best {
            let fp = *fp_ref;
            // Update LRU
            if let Some(idx) = self.lru.iter().position(|f| *f == fp) {
                let removed = self.lru.remove(idx);
                self.lru.push(removed);
            }
            self.next_timestamp += 1;

            // Update stats
            if let Some(entry) = self.entries.get_mut(&fp) {
                entry.hits += 1;
                entry.last_access = self.next_timestamp;
            }

            Some(fp)
        } else {
            None
        }
    }

    /// Insert a prefix entry
    pub fn insert(&mut self, key: PrefixKey, size: usize) -> Result<(), String> {
        let fingerprint = PrefixFingerprint::new(
            key.model_fingerprint,
            key.state_schema_fingerprint,
            key.tokenizer_fingerprint,
            key.plan_fingerprint,
            key.prefix_token_hash,
            key.prefix_len,
        );

        // Remove old entry if updating
        if let Some(old_fp) = self.entries
            .range(..=fingerprint)
            .next_back()
            .filter(|(fp, _)| **fp == fingerprint)
            .map(|(fp, _)| *fp)
        {
            self.remove(&old_fp);
        }

        // Evict if needed
        if self.total_bytes + size > self.max_bytes {
            self.evict_until(self.max_bytes - size);
        }

        // Insert
        self.entries.insert(fingerprint.clone(), CacheEntry {
            prefix_len: key.prefix_len,
            size,
            last_access: self.next_timestamp,
            last_update: self.next_timestamp,
            hits: 0,
            misses: 0,
        });
        self.lru.push(fingerprint);
        self.total_bytes += size;
        self.next_timestamp += 1;

        Ok(())
    }

    /// Remove an entry
    fn remove(&mut self, fingerprint: &PrefixFingerprint) {
        if let Some(entry) = self.entries.remove(fingerprint) {
            self.total_bytes -= entry.size;
            self.lru.retain(|f| f != fingerprint);
        }
    }

    /// Evict entries until total bytes is under target
    fn evict_until(&mut self, target: usize) {
        while self.total_bytes > target && !self.lru.is_empty() {
            let fingerprint = self.lru.remove(0);
            if let Some(entry) = self.entries.remove(&fingerprint) {
                self.total_bytes -= entry.size;
            }
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let total_hits: u64 = self.entries.values().map(|e| e.hits).sum();
        let total_misses: u64 = self.entries.values().map(|e| e.misses).sum();
        let total_lookups = total_hits + total_misses;

        CacheStats {
            total_entries: self.entries.len(),
            total_bytes: self.total_bytes,
            max_bytes: self.max_bytes,
            total_lookups: total_lookups as usize,
            total_hits,
            total_misses,
            hit_rate: if total_lookups > 0 {
                total_hits as f64 / total_lookups as f64
            } else {
                0.0
            },
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
        self.total_bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefix::{ModelFingerprint, StateSchemaFingerprint, TokenizerFingerprint, PlanFingerprint};

    fn create_key(id: u8, prefix_len: usize) -> PrefixKey {
        PrefixKey {
            model_fingerprint: ModelFingerprint { digest: [id; 32] },
            state_schema_fingerprint: StateSchemaFingerprint { digest: [(id + 1) as u8; 32] },
            tokenizer_fingerprint: TokenizerFingerprint { digest: [(id + 2) as u8; 32] },
            plan_fingerprint: PlanFingerprint { digest: [(id + 3) as u8; 32] },
            prefix_token_hash: 0xabc123,
            prefix_len,
        }
    }

    #[test]
    fn test_ram_cache_insert_lookup() {
        let mut cache = RamPrefixCache::new(1024);

        let key = create_key(1, 100);
        let result = cache.insert(key, 500);
        assert!(result.is_ok());

        let lookup_key = create_key(1, 200);
        let result = cache.lookup(&lookup_key);
        assert!(result.is_some());
    }

    #[test]
    fn test_ram_cache_miss() {
        let mut cache = RamPrefixCache::new(1024);

        let key = create_key(1, 100);
        let lookup_key = create_key(2, 100);

        let result = cache.lookup(&lookup_key);
        assert!(result.is_none());
    }

    #[test]
    fn test_ram_cache_eviction() {
        let mut cache = RamPrefixCache::new(100);

        // Insert entries until we exceed capacity
        for i in 0..5 {
            let key = create_key(i as u8, 10);
            let result = cache.insert(key, 30);
            assert!(result.is_ok());
        }

        // Should have evicted some entries
        assert!(cache.stats().total_bytes <= 100);
    }

    #[test]
    fn test_ram_cache_clear() {
        let mut cache = RamPrefixCache::new(1024);

        let key = create_key(1, 100);
        cache.insert(key, 500).unwrap();

        cache.clear();
        assert_eq!(cache.stats().total_entries, 0);
        assert_eq!(cache.stats().total_bytes, 0);
    }
}