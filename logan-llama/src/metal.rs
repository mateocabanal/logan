use crate::DType;
use std::sync::{Arc, Mutex, MutexGuard};

/// Opaque Metal weight state retained by one resident matrix.
///
/// `metal_matmul_multi` lazily creates the native tensor handle and returns it
/// through each descriptor. Keeping the owning value behind an `Arc<Mutex<_>>`
/// lets cloned models/sessions share one handle without double-freeing it.
#[derive(Debug, Default)]
pub(crate) struct MetalTensorHandle {
    raw: usize,
}

impl MetalTensorHandle {
    pub(crate) fn raw(&self) -> *mut logan_metal::ColiMetalTensor {
        self.raw as *mut logan_metal::ColiMetalTensor
    }

    pub(crate) fn set_raw(&mut self, raw: *mut logan_metal::ColiMetalTensor) {
        let raw = raw as usize;
        if self.raw != raw {
            if self.raw != 0 {
                unsafe {
                    logan_metal::coli_metal_tensor_free(
                        self.raw as *mut logan_metal::ColiMetalTensor,
                    );
                }
            }
            self.raw = raw;
        }
    }
}

impl Drop for MetalTensorHandle {
    fn drop(&mut self) {
        let raw = std::mem::replace(&mut self.raw, 0);
        if raw != 0 {
            unsafe {
                logan_metal::coli_metal_tensor_free(raw as *mut logan_metal::ColiMetalTensor);
            }
        }
    }
}

pub(crate) type SharedMetalTensorHandle = Arc<Mutex<MetalTensorHandle>>;

/// Lock a persistent matrix handle, recovering from a poisoned lock after a
/// panic so one failed inference cannot strand native resources.
pub(crate) fn lock_handle(handle: &SharedMetalTensorHandle) -> MutexGuard<'_, MetalTensorHandle> {
    handle
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Requested execution backend. `Auto` may attempt Metal only when the source
/// weights are BF16; all unsupported cases remain on the deterministic CPU path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPreference {
    Cpu,
    Auto,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendUsed {
    Cpu,
    Metal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendReport {
    pub requested: BackendPreference,
    pub used: BackendUsed,
    pub attempted: bool,
    pub available: bool,
    pub reason: &'static str,
}

impl BackendReport {
    pub fn cpu(requested: BackendPreference, reason: &'static str) -> Self {
        Self {
            requested,
            used: BackendUsed::Cpu,
            attempted: false,
            available: false,
            reason,
        }
    }
}

/// One-token batch of BF16 GEMV projections sharing the same activation.
/// Native output is written only into temporary buffers by the caller, so a
/// declined batch can always fall back to the exact scalar CPU calculation.
pub(crate) fn matmul_bf16_multi(
    requested: BackendPreference,
    dtype: DType,
    x: &[f32],
    descs: &mut [logan_metal::MetalMatmulDesc<'_>],
) -> BackendReport {
    if matches!(requested, BackendPreference::Cpu) {
        return BackendReport::cpu(requested, "CPU explicitly selected");
    }
    if dtype != DType::BF16 {
        return BackendReport::cpu(requested, "Metal dense entry point requires BF16 weights");
    }
    let available = logan_metal::metal_available();
    if !available {
        return BackendReport {
            requested,
            used: BackendUsed::Cpu,
            attempted: true,
            available: false,
            reason: "Metal unavailable; CPU fallback",
        };
    }
    if logan_metal::metal_matmul_multi(x, descs) {
        BackendReport {
            requested,
            used: BackendUsed::Metal,
            attempted: true,
            available: true,
            reason: "BF16 dense entry point",
        }
    } else {
        BackendReport {
            requested,
            used: BackendUsed::Cpu,
            attempted: true,
            available: true,
            reason: "BF16 dense entry point declined; CPU fallback",
        }
    }
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn llama_layer(
    model_id: u64,
    layer: usize,
    descs: &mut [logan_metal::MetalMatmulDesc<'_>],
    x: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
    input_norm: &[f32],
    post_norm: &[f32],
    d: usize,
    inter: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
    eps: f32,
) -> i32 {
    logan_metal::llama_layer(
        model_id, layer, descs, x, k_out, v_out, input_norm, post_norm, d, inter, heads, kv_heads,
        head_dim, pos, theta, eps,
    )
}

pub(crate) fn llama_drop_model(model_id: u64) {
    logan_metal::llama_drop_model(model_id);
}

/// Optional Accelerate/BNNS BF16 GEMV.  Qwen's decode path uses this seam
/// because CPU BLAS can beat a generic Metal launch for one-token rows.
pub(crate) fn bnns_bf16(
    requested: BackendPreference,
    weights: &[u8],
    x: &[f32],
    y: &mut [f32],
    out: usize,
    input: usize,
) -> Option<BackendReport> {
    if std::env::var("LOGAN_BNNS_BF16").ok().as_deref() != Some("1") {
        return None;
    }
    if logan_metal::bnns_bf16_matmul(weights, x, y, out, input) {
        Some(BackendReport {
            requested,
            used: BackendUsed::Cpu,
            attempted: true,
            available: true,
            reason: "opt-in BNNS BF16 dense entry point",
        })
    } else {
        None
    }
}

/// Single BF16 GEMV seam retained for projections that cannot be batched.
pub(crate) fn matmul_bf16(
    requested: BackendPreference,
    dtype: DType,
    weights: &[u8],
    x: &[f32],
    y: &mut [f32],
    rows: usize,
    out: usize,
    input: usize,
) -> BackendReport {
    if matches!(requested, BackendPreference::Cpu) {
        return BackendReport::cpu(requested, "CPU explicitly selected");
    }
    if dtype != DType::BF16 {
        return BackendReport::cpu(requested, "Metal dense entry point requires BF16 weights");
    }
    let available = logan_metal::metal_available();
    if !available {
        return BackendReport {
            requested,
            used: BackendUsed::Cpu,
            attempted: true,
            available: false,
            reason: "Metal unavailable; CPU fallback",
        };
    }
    let ok = logan_metal::bf16_matmul(weights, x, y, rows, out, input) > 0;
    if ok {
        BackendReport {
            requested,
            used: BackendUsed::Metal,
            attempted: true,
            available: true,
            reason: "BF16 dense entry point",
        }
    } else {
        BackendReport {
            requested,
            used: BackendUsed::Cpu,
            attempted: true,
            available: true,
            reason: "BF16 dense entry point declined; CPU fallback",
        }
    }
}
