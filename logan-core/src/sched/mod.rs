//! Logan inference scheduler (issue #43 umbrella; V0 deterministic core).
//!
//! One scheduler owner decides; platform I/O, CPU workers and accelerators
//! execute elsewhere; every result returns as a typed completion event.
//!
//! The core deliberately contains no Rust async executor machinery.

pub mod core;
pub mod ids;
pub mod residency;
pub mod runtime;
pub mod sim;

pub use core::{
    Action, ActionKind, Budget, Effect, Outcome, SchedError, SchedulerCore, Session, SessionState,
    Step,
};
pub use ids::{Generation, LoadId, SessionId, Ticket};
pub use residency::{
    CompletionDisposition, DeviceId, ExpertKey, Lease, LeaseId, LoadCompletion, LoadDisposition,
    LoadResult, LoadWake, MemoryPoolId, PoolStats, RepresentationKey, ResidencyError,
    ResidencyManager, ResidencyState,
};
pub use runtime::{
    BoundedExecutor, CompletionSendError, CompletionSink, DispatchRequest, ExecutorSet,
    RequestSendError, RuntimeConfig, RuntimeConfigError, RuntimeHandle, RuntimeReply,
    RuntimeRequest, SchedulerRuntime, ShutdownMode, SubmitDisposition, SubmitResult,
};
pub use sim::{
    FakeCompletionScript, ReplayError, SchedulerSimulator, SimOp, SimOutcome, TraceEvent,
};
