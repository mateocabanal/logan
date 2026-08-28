//! Logan core — the engine-neutral runtime foundation.
//!
//! Slice 2 of the neutral backend design (docs/design_neutral_backend.md):
//! math primitives + tensor storage extracted from the qwen4 engine,
//! verbatim (C-identical numerics are the contract — the token-identity
//! gates prove extraction is behavior-neutral).

pub mod expert;
pub mod math;
pub mod sched;
pub mod storage;
pub mod telemetry;
