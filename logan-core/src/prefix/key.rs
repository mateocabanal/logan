//! Prefix key and fingerprint definitions.

use std::fmt;

/// Fingerprint of a model artifact or checkpoint.
/// Used to reject cache entries for incompatible models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ModelFingerprint {
    /// SHA-256 digest of model artifact
    pub digest: [u8; 32],
}

impl fmt::Display for ModelFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.digest))
    }
}

/// Fingerprint of tokenizer/template identity.
/// Used to reject cache entries for incompatible tokenizers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct TokenizerFingerprint {
    /// SHA-256 digest of tokenizer/template identity
    pub digest: [u8; 32],
}

impl fmt::Display for TokenizerFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.digest))
    }
}

/// Fingerprint of state representation/schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct StateSchemaFingerprint {
    /// SHA-256 digest of state schema
    pub digest: [u8; 32],
}

impl fmt::Display for StateSchemaFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.digest))
    }
}

/// Fingerprint of compiled plan/package identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct PlanFingerprint {
    /// SHA-256 digest of compiled plan/package
    pub digest: [u8; 32],
}

impl fmt::Display for PlanFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.digest))
    }
}

/// Complete prefix cache identity.
/// Rejects any snapshot that doesn't belong to exactly the compatible
/// model/runtime state contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct PrefixFingerprint {
    /// Model fingerprint
    pub model: ModelFingerprint,
    /// State schema fingerprint
    pub state_schema: StateSchemaFingerprint,
    /// Tokenizer/template fingerprint
    pub tokenizer: TokenizerFingerprint,
    /// Compiled plan/package fingerprint
    pub plan: PlanFingerprint,
    /// Prefix token hash
    pub prefix_token_hash: u64,
    /// Prefix token count
    pub prefix_len: usize,
}

impl PrefixFingerprint {
    /// Create new prefix fingerprint
    pub fn new(
        model: ModelFingerprint,
        state_schema: StateSchemaFingerprint,
        tokenizer: TokenizerFingerprint,
        plan: PlanFingerprint,
        prefix_token_hash: u64,
        prefix_len: usize,
    ) -> Self {
        PrefixFingerprint {
            model,
            state_schema,
            tokenizer,
            plan,
            prefix_token_hash,
            prefix_len,
        }
    }

    /// Check compatibility with another fingerprint
    pub fn is_compatible_with(&self, other: &PrefixFingerprint) -> bool {
        self.model == other.model
            && self.state_schema == other.state_schema
            && self.tokenizer == other.tokenizer
            && self.plan == other.plan
            && self.prefix_token_hash == other.prefix_token_hash
            && self.prefix_len == other.prefix_len
    }

    /// Check if compatible except prefix length (for longest-prefix lookup)
    pub fn is_compatible_prefix(&self, other: &PrefixFingerprint) -> bool {
        self.model == other.model
            && self.state_schema == other.state_schema
            && self.tokenizer == other.tokenizer
            && self.plan == other.plan
            && self.prefix_token_hash == other.prefix_token_hash
    }
}

/// Hash function for token sequences.
pub fn hash_tokens(tokens: &[u32]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    for token in tokens {
        token.hash(&mut hasher);
        // Include token boundaries to avoid ambiguity
        0xFFu32.hash(&mut hasher);
    }
    hasher.finish()
}

/// Checksum for prefix token sequence.
pub fn checksum_tokens(tokens: &[u32]) -> [u8; 32] {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(&token.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut checksum = [0u8; 32];
    checksum.copy_from_slice(&digest);
    checksum
}

/// Query key for prefix cache lookup
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrefixKey {
    pub model_fingerprint: ModelFingerprint,
    pub state_schema_fingerprint: StateSchemaFingerprint,
    pub tokenizer_fingerprint: TokenizerFingerprint,
    pub plan_fingerprint: PlanFingerprint,
    pub prefix_token_hash: u64,
    pub prefix_len: usize,
}
#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint() -> PrefixFingerprint {
        PrefixFingerprint::new(
            ModelFingerprint { digest: [1; 32] },
            StateSchemaFingerprint { digest: [2; 32] },
            TokenizerFingerprint { digest: [3; 32] },
            PlanFingerprint { digest: [4; 32] },
            0xabc123,
            100,
        )
    }

    #[test]
    fn test_fingerprint_compatible() {
        let a = fingerprint();
        let b = fingerprint();
        assert!(a.is_compatible_with(&b));
    }

    #[test]
    fn test_fingerprint_prefix_compatible() {
        let a = fingerprint();
        let b = PrefixFingerprint::new(
            a.model,
            a.state_schema,
            a.tokenizer,
            a.plan,
            0xabc123,
            150,
        );
        assert!(a.is_compatible_prefix(&b));
        assert!(!a.is_compatible_with(&b));
    }

    #[test]
    fn test_fingerprint_mismatch() {
        let a = fingerprint();
        let b = PrefixFingerprint::new(
            ModelFingerprint { digest: [5; 32] },
            a.state_schema,
            a.tokenizer,
            a.plan,
            0xabc123,
            100,
        );
        assert!(!a.is_compatible_with(&b));
        assert!(!a.is_compatible_prefix(&b));
    }

    #[test]
    fn test_token_hash() {
        assert_eq!(hash_tokens(&[]), hash_tokens(&[]));
        assert_ne!(hash_tokens(&[1, 2, 3]), hash_tokens(&[1, 2, 4]));
    }

    #[test]
    fn test_token_checksum() {
        let a = checksum_tokens(&[1, 2, 3]);
        let b = checksum_tokens(&[1, 2, 3]);
        assert_eq!(a, b);
        assert_ne!(a, checksum_tokens(&[1, 2, 4]));
    }
}