//! Prefix index for efficient lookup of cached prefixes.

use crate::prefix::{PrefixFingerprint, PrefixKey};
use std::collections::BTreeMap;

/// Prefix index manages cached prefixes and their metadata.
#[derive(Debug)]
pub struct PrefixIndex {
    /// Index of prefix fingerprints to their metadata
    entries: BTreeMap<PrefixFingerprint, PrefixMetadata>,
    /// LRU ordering (timestamp for LRU eviction)
    lru_order: Vec<(u64, PrefixFingerprint)>,
    /// Total bytes in cache
    total_bytes: usize,
    /// Maximum cache size (in bytes)
    max_bytes: usize,
    /// Current LRU timestamp
    next_timestamp: u64,
}

#[derive(Debug, Clone)]
struct PrefixMetadata {
    /// Size of this prefix in bytes
    size: usize,
    /// Timestamp when last accessed
    last_access: u64,
    /// Timestamp when last updated
    last_update: u64,
    /// Number of hits
    hits: u64,
    /// Number of misses
    misses: u64,
}

impl PrefixIndex {
    /// Create a new prefix index
    pub fn new(max_bytes: usize) -> Self {
        PrefixIndex {
            entries: BTreeMap::new(),
            lru_order: Vec::new(),
            total_bytes: 0,
            max_bytes,
            next_timestamp: 0,
        }
    }

    /// Insert or update a prefix entry
    pub fn insert(&mut self, key: PrefixKey, metadata: PrefixMetadata) -> Result<(), String> {
        let fingerprint = PrefixFingerprint::new(
            key.model_fingerprint,
            key.state_schema_fingerprint,
            key.tokenizer_fingerprint,
            key.plan_fingerprint,
            key.prefix_token_hash,
            key.prefix_len,
        );

        // Remove old entry if updating
        if let Some(old_fingerprint) = self.entries
            .range(..=fingerprint)
            .next_back()
            .filter(|(f, _)| **f == fingerprint)
            .map(|(f, _)| *f)
        {
            self.remove_entry(&old_fingerprint);
        }

        // Check if we need to evict
        let metadata_size = metadata.size;
        if self.total_bytes + metadata_size > self.max_bytes {
            // Evict LRU entries
            self.evict_until(self.max_bytes - metadata_size);
        }

        // Insert new entry
        let metadata_size = metadata.size;
        self.entries.insert(fingerprint.clone(), metadata);
        self.lru_order.push((self.next_timestamp, fingerprint.clone()));
        self.total_bytes += metadata_size;
        self.next_timestamp += 1;

        Ok(())
    }

    /// Look up the best matching prefix for a query
    pub fn lookup(&mut self, key: &PrefixKey) -> Option<PrefixFingerprint> {
        let query_fingerprint = PrefixFingerprint::new(
            key.model_fingerprint,
            key.state_schema_fingerprint,
            key.tokenizer_fingerprint,
            key.plan_fingerprint,
            key.prefix_token_hash,
            key.prefix_len,
        );

        // Find all compatible prefixes (same model, state, tokenizer, plan)
        let mut compatible = self.entries
            .iter()
            .filter(|(f, _)| f.is_compatible_prefix(&query_fingerprint))
            .map(|(f, _)| *f)
            .collect::<Vec<_>>();

        compatible.sort_by(|a, b| b.prefix_len.cmp(&a.prefix_len)); // Longest first

        if let Some(best) = compatible.first() {
            // Update LRU
            if let Some(idx) = self.lru_order.iter().position(|(_, f)| f == best) {
                let (ts, f) = self.lru_order.remove(idx);
                self.lru_order.push((self.next_timestamp, f));
                self.next_timestamp += 1;
            }

            // Update stats
            if let Some(meta) = self.entries.get_mut(best) {
                meta.hits += 1;
                meta.last_access = self.next_timestamp;
            }

            Some(*best)
        } else {
            // No match
            None
        }
    }

    /// Remove an entry from the index
    fn remove_entry(&mut self, fingerprint: &PrefixFingerprint) {
        if let Some(meta) = self.entries.remove(fingerprint) {
            self.total_bytes -= meta.size;

            // Remove from LRU order
            self.lru_order.retain(|(_, f)| f != fingerprint);
        }
    }

    /// Evict entries until total bytes is under the target
    fn evict_until(&mut self, target: usize) {
        while self.total_bytes > target && !self.lru_order.is_empty() {
            let (_, fingerprint) = self.lru_order.remove(0);
            let meta = self.entries.remove(&fingerprint).unwrap();
            self.total_bytes -= meta.size;
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let total_lookups = self.lru_order.len();
        let total_hits: u64 = self.entries.values()
            .map(|m| m.hits)
            .sum();
        let total_misses: u64 = self.entries.values()
            .map(|m| m.misses)
            .sum();

        CacheStats {
            total_entries: self.entries.len(),
            total_bytes: self.total_bytes,
            max_bytes: self.max_bytes,
            total_lookups,
            total_hits,
            total_misses,
            hit_rate: if total_lookups > 0 {
                total_hits as f64 / (total_hits + total_misses) as f64
            } else {
                0.0
            },
        }
    }
}

/// Prefix lookup trait for integration with runtime
pub trait PrefixLookup {
    fn lookup(&mut self, key: &PrefixKey) -> Option<PrefixFingerprint>;
    fn insert(&mut self, key: PrefixKey, size: usize) -> Result<(), String>;
    fn stats(&self) -> CacheStats;
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total number of entries in cache
    pub total_entries: usize,
    /// Total bytes used by cache
    pub total_bytes: usize,
    /// Maximum cache size in bytes
    pub max_bytes: usize,
    /// Total number of lookups performed
    pub total_lookups: usize,
    /// Total number of cache hits
    pub total_hits: u64,
    /// Total number of cache misses
    pub total_misses: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefix::{ModelFingerprint, StateSchemaFingerprint, TokenizerFingerprint, PlanFingerprint};

    fn create_fingerprint(id: u8, prefix_len: usize) -> PrefixFingerprint {
        PrefixFingerprint::new(
            ModelFingerprint { digest: [id; 32] },
            StateSchemaFingerprint { digest: [(id + 1) as u8; 32] },
            TokenizerFingerprint { digest: [(id + 2) as u8; 32] },
            PlanFingerprint { digest: [(id + 3) as u8; 32] },
            0xabc123,
            prefix_len,
        )
    }

    #[test]
    fn test_prefix_index_insert_lookup() {
        let mut index = PrefixIndex::new(1024);

        // Insert entries
        let key1 = PrefixKey {
            model_fingerprint: ModelFingerprint { digest: [1; 32] },
            state_schema_fingerprint: StateSchemaFingerprint { digest: [2; 32] },
            tokenizer_fingerprint: TokenizerFingerprint { digest: [3; 32] },
            plan_fingerprint: PlanFingerprint { digest: [4; 32] },
            prefix_token_hash: 0xabc123,
            prefix_len: 100,
        };

        let metadata = PrefixMetadata {
            size: 500,
            last_access: 0,
            last_update: 0,
            hits: 0,
            misses: 0,
        };

        let result = index.insert(key1.clone(), metadata);
        assert!(result.is_ok());

        // Lookup existing entry
        let result = index.lookup(&key1);
        assert!(result.is_some());

        // Lookup non-existent entry
        let key2 = PrefixKey {
            model_fingerprint: ModelFingerprint { digest: [5; 32] },
            state_schema_fingerprint: StateSchemaFingerprint { digest: [6; 32] },
            tokenizer_fingerprint: TokenizerFingerprint { digest: [7; 32] },
            plan_fingerprint: PlanFingerprint { digest: [8; 32] },
            prefix_token_hash: 0xdef456,
            prefix_len: 200,
        };

        let result = index.lookup(&key2);
        assert!(result.is_none());
    }

    #[test]
    fn test_prefix_index_eviction() {
        let mut index = PrefixIndex::new(30);

        // Insert small entries until we reach capacity
        for i in 0..5 {
            let key = PrefixKey {
                model_fingerprint: ModelFingerprint { digest: [i as u8; 32] },
                state_schema_fingerprint: StateSchemaFingerprint { digest: [(i + 1) as u8; 32] },
                tokenizer_fingerprint: TokenizerFingerprint { digest: [(i + 2) as u8; 32] },
                plan_fingerprint: PlanFingerprint { digest: [(i + 3) as u8; 32] },
                prefix_token_hash: i as u64 * 0xabc123,
                prefix_len: 10,
            };

            let metadata = PrefixMetadata {
                size: i + 10, // 10, 11, 12, 13, 14 bytes
                last_access: i as u64,
                last_update: i as u64,
                hits: 0,
                misses: 0,
            };

            let result = index.insert(key, metadata);
            assert!(result.is_ok());
        }

        // Should have evicted earliest entries
        assert!(index.stats().total_entries <= 3); // May have fewer due to capacity
        assert!(index.stats().total_bytes <= 100);
    }

    #[test]
    fn test_prefix_compatibility() {
        let a = create_fingerprint(1, 100);
        let b = create_fingerprint(1, 200); // Same prefix hash, different length
        let c = create_fingerprint(2, 100); // Different model

        assert!(a.is_compatible_prefix(&b));
        assert!(!a.is_compatible_with(&b));
        assert!(!a.is_compatible_prefix(&c));
        assert!(!a.is_compatible_with(&c));
    }
}