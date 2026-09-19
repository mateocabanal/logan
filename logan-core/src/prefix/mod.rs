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

pub mod key;
pub mod index;
pub mod memory;
pub mod disk;
pub mod format;
pub mod runtime;
pub use key::{ModelFingerprint, StateSchemaFingerprint, TokenizerFingerprint, PlanFingerprint};
pub use key::{PrefixKey, PrefixFingerprint};
pub use index::{PrefixIndex, PrefixLookup, CacheStats};
pub use memory::RamPrefixCache;
