//! Generic persistent prefix cache subsystem.
//!
//! Moves reusable prefix-cache infrastructure out of Qwen-specific ownership
//! into Logan-wide shared runtime code.
//!
//! Hierarchy:
//! active state -> RAM hot prefix cache -> miss -> persistent SSD prefix cache
//! -> miss -> prompt replay
//!
//! Lookup supports longest reusable prefix (not just exact matches).

pub mod disk;
pub mod format;
pub mod index;
pub mod key;
pub mod memory;
pub mod runtime;

pub use disk::SsdPrefixStore;
pub use index::{CacheStats, PrefixIndex, PrefixLookup};
pub use key::{
    ModelFingerprint, PlanFingerprint, PrefixFingerprint, PrefixKey, StateSchemaFingerprint,
    TokenizerFingerprint, checksum_tokens, hash_tokens,
};
pub use memory::{CacheHit, RamPrefixCache};
pub use runtime::{PrefixLookupResult, PrefixRuntime, PrefixRuntimeConfig};
