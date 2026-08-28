//! Metal FFI re-export shim: the backend now lives in the engine-neutral
//! `logan-metal` crate (slice 3 of the neutral backend design). Zero
//! call-site changes — this module re-exports the whole surface.

pub use logan_metal::*;
