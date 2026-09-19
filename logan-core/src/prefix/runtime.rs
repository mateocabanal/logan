//! Prefix cache runtime integration.
//!
//! Integrates the generic prefix cache into the inference pipeline:
//! - Tokenize incoming request
//! - Find longest cached prefix
//! - Restore causal state
//! - Forward only uncached suffix
//! - Persist useful prefix boundaries

use crate::prefix::{PrefixKey, RamPrefixCache};
use crate::state::CausalState;
use std::collections::HashMap;

/// Runtime configuration for prefix caching
#[derive(Debug, Clone)]
pub struct PrefixRuntimeConfig {
    /// Maximum RAM cache size in bytes
    pub ram_cache_bytes: usize,
    /// Maximum SSD cache size in bytes
    pub ssd_cache_bytes: usize,
    /// Minimum prefix length to cache (in tokens)
    pub min_cache_tokens: usize,
    /// Whether writes are enabled
    pub writes_enabled: bool,
    /// Cache salt for user-specific isolation
    pub salt: Vec<u8>,
}

impl Default for PrefixRuntimeConfig {
    fn default() -> Self {
        PrefixRuntimeConfig {
            ram_cache_bytes: 512 * 1024 * 1024, // 512 MB
            ssd_cache_bytes: 4 * 1024 * 1024 * 1024, // 4 GB
            min_cache_tokens: 4,
            writes_enabled: true,
            salt: Vec::new(),
        }
    }
}

/// Result of a prefix lookup operation
#[derive(Debug, Clone)]
pub struct PrefixLookupResult {
    /// Whether there was a cache hit (RAM or SSD)
    pub hit: bool,
    /// Whether the hit was from RAM
    pub ram_hit: bool,
    /// Whether the hit was from SSD
    pub ssd_hit: bool,
    /// Length of the cached prefix (in tokens)
    pub prefix_len: usize,
    /// Number of tokens that still need to be forwarded
    pub remaining_tokens: usize,
}

/// Runtime integration for prefix caching
#[derive(Debug)]
pub struct PrefixRuntime {
    /// RAM cache
    ram_cache: RamPrefixCache,
    /// Runtime configuration
    config: PrefixRuntimeConfig,
    /// Active session state
    session_state: HashMap<String, String>,
}

impl PrefixRuntime {
    /// Create a new prefix runtime
    pub fn new(config: PrefixRuntimeConfig) -> Self {
        PrefixRuntime {
            ram_cache: RamPrefixCache::new(config.ram_cache_bytes),
            config,
            session_state: HashMap::new(),
        }
    }

    /// Look up the longest cached prefix for a request
    pub fn lookup_prefix(&mut self, key: &PrefixKey) -> PrefixLookupResult {
        // First check RAM cache
        if let Some(fp) = self.ram_cache.lookup(key) {
            return PrefixLookupResult {
                hit: true,
                ram_hit: true,
                ssd_hit: false,
                prefix_len: fp.prefix_len,
                remaining_tokens: key.prefix_len - fp.prefix_len,
            };
        }

        // Then check SSD store (would be implemented in full version)
        // For now, this is a placeholder
        let _ssd_store = std::path::Path::new(".");

        PrefixLookupResult {
            hit: false,
            ram_hit: false,
            ssd_hit: false,
            prefix_len: 0,
            remaining_tokens: key.prefix_len,
        }
    }

    /// Cache a completed prefix
    pub fn cache_prefix(&mut self, key: PrefixKey, state: &CausalState) -> Result<(), String> {
        if key.prefix_len < self.config.min_cache_tokens {
            return Ok(());
        }

        // Estimate size (simplified)
        let size = self.estimate_state_size(state);
        self.ram_cache.insert(key, size)
    }

    /// Estimate state size for caching purposes
    fn estimate_state_size(&self, state: &CausalState) -> usize {
        match state {
            CausalState::AppendOnly(s) => s.keys.len() * 4 + s.values.len() * 4,
            CausalState::Ring(s) => s.buffer.len() * 4,
            CausalState::MutableFixed(s) => s.data.len() * 4,
            CausalState::SparsePaged(s) => {
                s.pages.iter().map(|p| p.len() * 4).sum()
            }
            CausalState::Opaque(data) => data.len(),
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> crate::prefix::CacheStats {
        self.ram_cache.stats()
    }

    /// Clear all caches
    pub fn clear(&mut self) {
        self.ram_cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PrefixRuntimeConfig::default();
        assert_eq!(config.ram_cache_bytes, 512 * 1024 * 1024);
        assert_eq!(config.ssd_cache_bytes, 4 * 1024 * 1024 * 1024);
        assert_eq!(config.min_cache_tokens, 4);
        assert!(config.writes_enabled);
    }

    #[test]
    fn test_lookup_miss() {
        let config = PrefixRuntimeConfig::default();
        let mut runtime = PrefixRuntime::new(config);

        let key = PrefixKey {
            model_fingerprint: crate::prefix::ModelFingerprint { digest: [1; 32] },
            state_schema_fingerprint: crate::prefix::StateSchemaFingerprint { digest: [2; 32] },
            tokenizer_fingerprint: crate::prefix::TokenizerFingerprint { digest: [3; 32] },
            plan_fingerprint: crate::prefix::PlanFingerprint { digest: [4; 32] },
            prefix_token_hash: 0xabc123,
            prefix_len: 100,
        };

        let result = runtime.lookup_prefix(&key);
        assert!(!result.hit);
        assert_eq!(result.remaining_tokens, 100);
    }

    #[test]
    fn test_cache_and_lookup() {
        let config = PrefixRuntimeConfig::default();
        let mut runtime = PrefixRuntime::new(config);

        let key = PrefixKey {
            model_fingerprint: crate::prefix::ModelFingerprint { digest: [1; 32] },
            state_schema_fingerprint: crate::prefix::StateSchemaFingerprint { digest: [2; 32] },
            tokenizer_fingerprint: crate::prefix::TokenizerFingerprint { digest: [3; 32] },
            plan_fingerprint: crate::prefix::PlanFingerprint { digest: [4; 32] },
            prefix_token_hash: 0xabc123,
            prefix_len: 100,
        };

        let state = crate::state::CausalState::append_only(2, 4, 8);
        let result = runtime.cache_prefix(key.clone(), &state);
        assert!(result.is_ok());

        let lookup_result = runtime.lookup_prefix(&key);
        assert!(lookup_result.hit);
        assert!(lookup_result.ram_hit);
    }
}