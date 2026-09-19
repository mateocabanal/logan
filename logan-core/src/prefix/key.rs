//! Prefix identity and compatibility.
//!
//! Exact fingerprints identify persisted entries. Longest-prefix lookup uses the
//! actual token sequence; a hash of a longer sequence can never prove that a
//! shorter sequence is its prefix.

use sha2::{Digest, Sha256};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ModelFingerprint {
    pub digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct TokenizerFingerprint {
    pub digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct StateSchemaFingerprint {
    pub digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct PlanFingerprint {
    pub digest: [u8; 32],
}

macro_rules! impl_display {
    ($ty:ty) => {
        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", hex::encode(self.digest))
            }
        }
    };
}

impl_display!(ModelFingerprint);
impl_display!(TokenizerFingerprint);
impl_display!(StateSchemaFingerprint);
impl_display!(PlanFingerprint);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct PrefixFingerprint {
    pub model: ModelFingerprint,
    pub state_schema: StateSchemaFingerprint,
    pub tokenizer: TokenizerFingerprint,
    pub plan: PlanFingerprint,
    pub prefix_token_hash: u64,
    pub prefix_len: usize,
}

impl PrefixFingerprint {
    pub fn new(
        model: ModelFingerprint,
        state_schema: StateSchemaFingerprint,
        tokenizer: TokenizerFingerprint,
        plan: PlanFingerprint,
        prefix_token_hash: u64,
        prefix_len: usize,
    ) -> Self {
        Self {
            model,
            state_schema,
            tokenizer,
            plan,
            prefix_token_hash,
            prefix_len,
        }
    }

    pub fn same_contract(&self, other: &Self) -> bool {
        self.model == other.model
            && self.state_schema == other.state_schema
            && self.tokenizer == other.tokenizer
            && self.plan == other.plan
    }

    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.same_contract(other)
            && self.prefix_token_hash == other.prefix_token_hash
            && self.prefix_len == other.prefix_len
    }
}

/// Query/cache key. Tokens are retained so longest-prefix lookup is exact.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrefixKey {
    pub model_fingerprint: ModelFingerprint,
    pub state_schema_fingerprint: StateSchemaFingerprint,
    pub tokenizer_fingerprint: TokenizerFingerprint,
    pub plan_fingerprint: PlanFingerprint,
    pub prefix_tokens: Vec<u32>,
    pub prefix_token_hash: u64,
}

impl PrefixKey {
    pub fn new(
        model_fingerprint: ModelFingerprint,
        state_schema_fingerprint: StateSchemaFingerprint,
        tokenizer_fingerprint: TokenizerFingerprint,
        plan_fingerprint: PlanFingerprint,
        prefix_tokens: impl Into<Vec<u32>>,
    ) -> Self {
        let prefix_tokens = prefix_tokens.into();
        let prefix_token_hash = hash_tokens(&prefix_tokens);
        Self {
            model_fingerprint,
            state_schema_fingerprint,
            tokenizer_fingerprint,
            plan_fingerprint,
            prefix_tokens,
            prefix_token_hash,
        }
    }

    pub fn prefix_len(&self) -> usize {
        self.prefix_tokens.len()
    }

    pub fn fingerprint(&self) -> PrefixFingerprint {
        PrefixFingerprint::new(
            self.model_fingerprint,
            self.state_schema_fingerprint,
            self.tokenizer_fingerprint,
            self.plan_fingerprint,
            self.prefix_token_hash,
            self.prefix_len(),
        )
    }

    pub fn same_contract(&self, other: &Self) -> bool {
        self.model_fingerprint == other.model_fingerprint
            && self.state_schema_fingerprint == other.state_schema_fingerprint
            && self.tokenizer_fingerprint == other.tokenizer_fingerprint
            && self.plan_fingerprint == other.plan_fingerprint
    }

    pub fn is_prefix_of(&self, query: &Self) -> bool {
        self.same_contract(query) && query.prefix_tokens.starts_with(&self.prefix_tokens)
    }

    /// Derive a namespace-isolated key without discarding tokenizer/state/plan
    /// identity. This is appropriate for per-user cache isolation.
    pub fn with_salt(&self, salt: impl AsRef<[u8]>) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"logan-prefix-salt-v1");
        hasher.update(self.model_fingerprint.digest);
        hasher.update(salt.as_ref());
        let digest: [u8; 32] = hasher.finalize().into();
        let mut key = self.clone();
        key.model_fingerprint = ModelFingerprint { digest };
        key
    }
}

/// Stable token-sequence hash for filenames/indexing. Prefix correctness never
/// relies on this alone; lookup also checks the actual token sequence.
pub fn hash_tokens(tokens: &[u32]) -> u64 {
    let checksum = checksum_tokens(tokens);
    u64::from_le_bytes(checksum[..8].try_into().unwrap())
}

pub fn checksum_tokens(tokens: &[u32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"logan-prefix-tokens-v1");
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn longest_prefix_uses_tokens_not_equal_hashes() {
        let short = key(&[1, 2, 3]);
        let long = key(&[1, 2, 3, 4, 5]);
        assert_ne!(short.prefix_token_hash, long.prefix_token_hash);
        assert!(short.is_prefix_of(&long));
        assert!(!long.is_prefix_of(&short));
    }

    #[test]
    fn contract_mismatch_rejects_prefix() {
        let a = key(&[1, 2]);
        let mut b = key(&[1, 2, 3]);
        b.model_fingerprint = ModelFingerprint { digest: [9; 32] };
        assert!(!a.is_prefix_of(&b));
    }

    #[test]
    fn token_hash_is_stable_and_order_sensitive() {
        assert_eq!(hash_tokens(&[1, 2, 3]), hash_tokens(&[1, 2, 3]));
        assert_ne!(hash_tokens(&[1, 2, 3]), hash_tokens(&[1, 3, 2]));
    }

    #[test]
    fn salt_preserves_non_model_contract_identity() {
        let original = key(&[1, 2, 3]);
        let salted = original.with_salt(b"alice");
        assert_ne!(salted.model_fingerprint, original.model_fingerprint);
        assert_eq!(
            salted.state_schema_fingerprint,
            original.state_schema_fingerprint
        );
        assert_eq!(salted.tokenizer_fingerprint, original.tokenizer_fingerprint);
        assert_eq!(salted.plan_fingerprint, original.plan_fingerprint);
        assert_eq!(salted.prefix_tokens, original.prefix_tokens);
    }
}
