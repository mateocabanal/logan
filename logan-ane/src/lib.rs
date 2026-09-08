//! Low-level Apple Neural Engine access for Logan.
//!
//! `logan-ane` deliberately sits below CoreML's public model API. On Apple
//! Silicon macOS it dynamically resolves Apple's private ANE Objective-C
//! classes, compiles raw MIL text into an `_ANEInMemoryModel`, and evaluates
//! requests over IOSurface-backed memory.
//!
//! The private ABI is undocumented and version-fragile. This crate is intended
//! for local research, profiling, and Logan runtime experiments; it is not an
//! App Store compatibility layer.

pub mod blob;
mod cache;
mod error;
pub mod mil;

pub use blob::{BlobDataType, BlobOffset, BlobV2Builder};
pub use cache::{AneProgramCache, AneProgramCacheStats};
pub use error::{AneError, Result};
pub use mil::{DenseProjection, MilProgram, WeightBlob};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod model;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod raw;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod surface;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use model::{
    AneClient, AneDeviceInfo, AneModel, AneQos, AneRequest, AneRuntime, CompileOptions, ModelState,
    MutableWeightMapping, RuntimeCapabilities,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use surface::{AneSurface, SurfaceRead, SurfaceWrite};

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
mod unsupported;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub use unsupported::*;

/// Names of the private classes/selectors currently wrapped by this crate.
/// These are exposed for diagnostics and ABI probes without making the raw
/// Objective-C messaging helpers part of the stable Rust API.
pub mod private_abi {
    pub const APPLE_NEURAL_ENGINE_FRAMEWORK: &str =
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";
    pub const ANE_COMPILER_FRAMEWORK: &str =
        "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler";

    pub const IN_MEMORY_DESCRIPTOR: &str = "_ANEInMemoryModelDescriptor";
    pub const IN_MEMORY_MODEL: &str = "_ANEInMemoryModel";
    pub const REQUEST: &str = "_ANERequest";
    pub const IOSURFACE_OBJECT: &str = "_ANEIOSurfaceObject";
    pub const CLIENT: &str = "_ANEClient";
    pub const DEVICE_INFO: &str = "_ANEDeviceInfo";

    pub const DESCRIPTOR_FROM_MIL: &str = "modelWithMILText:weights:optionsPlist:";
    pub const MODEL_FROM_DESCRIPTOR: &str = "inMemoryModelWithDescriptor:";
    pub const COMPILE: &str = "compileWithQoS:options:error:";
    pub const LOAD: &str = "loadWithQoS:options:error:";
    pub const EVALUATE: &str = "evaluateWithQoS:options:request:error:";
    pub const UNLOAD: &str = "unloadWithQoS:error:";
    pub const MAP_MUTABLE_WEIGHTS: &str =
        "mapMutableWeightsForModel:andProcedure:mappedWeightsBuffer:size:error:";
    pub const SYNC_MUTABLE_WEIGHTS: &str =
        "syncMutableWeightsForModel:andProcedure:fromOffset:withSize:error:";
    pub const UNMAP_MUTABLE_WEIGHTS: &str = "unmapMutableWeightsForModel:andProcedure:";
}
