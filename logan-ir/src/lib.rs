//! Logan IR — the shared language between the compiler and the runtime.
//!
//! `logan compile` emits a plan artifact (graph + placement + quant) that
//! `logan run` executes. The runtime can also build a graph itself from a
//! package when no plan exists. This crate holds the types both sides speak.
//!
//! Design: docs/design_neutral_backend.md (slice 1).

pub mod graph;
pub mod plan;

pub use graph::{AttentionKind, Graph, Node, NodeId, Op, Value, ValueId, ValueType};
pub use plan::{MemoryPlan, Placement, PlanArtifact, QuantSpec};
