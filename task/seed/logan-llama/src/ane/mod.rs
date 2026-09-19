//! Qualification-only ANE execution seam for Logan model runtimes.
//!
//! The public boundary is plain data. Native `logan_ane` values are owned by
//! the same thread as `AneNativeOwner`; no model, channel, or IOSurface wrapper
//! is exposed as `Send`/`Sync`.

mod executor;
mod program;

pub use executor::{
    AneExecutor, AneNativeOwner, AneOperationSpec, AneTicket, CompletionDisposition,
    CompletionStatus, ExpiredOperation, OperationId, ScratchDisposition, SubmitError,
};
pub use program::{
    AnePrecision, AnePrivateAbiProbe, AneProgramIdentity, AneProgramSpec, AneShapeLayout,
};
