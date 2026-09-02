//! Logan IR — the shared language between the compiler and the runtime.
//!
//! `logan compile` emits a plan artifact (graph + physical resource plan) that
//! `logan run` executes. The runtime can also build a graph itself from a
//! package when no plan exists. This crate holds the types both sides speak.
//!
//! Design: docs/design_neutral_backend.md (slice 1).

pub mod context;
pub mod graph;
pub mod optimizer;
pub mod plan;
pub mod resource;
pub mod tiered_optimizer;

pub use context::{
    ContextConstraint, ContextConstraintKind, ContextPlan, ContextStateBytes, PlannerMemoryBudget,
};
pub use graph::{AttentionKind, Graph, Node, NodeId, Op, Value, ValueId, ValueType};
pub use optimizer::{
    BUILTIN_COST_MODEL_V1, CandidateGroup, ContextCandidate, OptimizerInput, ParetoPlan,
    PlanDecision, PlanMetrics, RejectedCandidate, RepresentationCandidate, material_plans,
    pareto_plans, select_plan,
};
pub use plan::{MemoryPlan, Placement, PlanArtifact, QuantSpec};
pub use resource::{
    AccessKind, AccessPlan, BackingKind, BackingPlan, DataMutability, MemoryPoolBudget,
    MemoryPoolId, ResidencyPlan, ResourceBudget, ResourcePlan, StoragePoolBudget, StoragePoolId,
};
pub use tiered_optimizer::{
    MemoryPoolUsage, StoragePoolUsage, TIERED_COST_MODEL_V1, TieredCandidateGroup,
    TieredContextCandidate, TieredOptimizerInput, TieredParetoPlan, TieredPlanDecision,
    TieredPlanMetrics, TieredRejectedCandidate, TieredRepresentationCandidate,
    TieredResourceUsage, select_tiered_plan, tiered_material_plans, tiered_pareto_plans,
};
