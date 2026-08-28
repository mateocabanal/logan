//! Logan inference scheduler (issue #43 umbrella; V0 deterministic core).
//!
//! One scheduler owner decides; platform I/O, CPU workers and accelerators
//! execute elsewhere; every result returns as a typed completion event.
//!
//! The core deliberately contains no Rust async executor machinery.

pub mod core;
pub mod ids;

pub use core::{
    Action, ActionKind, Budget, Effect, Outcome, SchedError, SchedulerCore, Session, SessionState,
    Step,
};
pub use ids::{Generation, LoadId, SessionId, Ticket};
