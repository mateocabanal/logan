//! `colic` is the offline compiler for target-compiled COLI artifacts.
//!
//! It deliberately has no link-time relationship with the C inference runtime.
//! Hardware-specialized Pareto planning lives in `optimize`; emission remains
//! delegated to the same verified target lowerers used by manual compilation.
//! Pareto choices use stable IDs so interactive and scripted selection replay
//! the same physical representation decisions.

pub mod cli;
pub mod codec;
pub mod context_plan;
pub mod error;
pub mod format;
pub mod generated;
pub mod ir;
pub mod model;
pub mod optimize;
pub mod passes;
pub mod pipeline;
pub mod quant;
pub mod recompile;
pub mod source;
pub mod storage;
pub mod target;
pub mod target_registry;
pub mod verify;
pub mod verify_target;

pub use error::{ColicError, Result};
