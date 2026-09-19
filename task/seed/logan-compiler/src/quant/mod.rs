//! Offline quantization passes used by target lowering.

#[cfg(target_arch = "x86_64")]
pub mod avx2;
pub mod mxfp4;
pub mod mxfp4_record;
pub mod precision_policy;
