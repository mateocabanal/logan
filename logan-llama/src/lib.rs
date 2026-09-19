//! Pure-Rust MiniCPM5/Llama checkpoint and dense runtime.
//!
//! The crate validates safetensors contracts, materializes resident dense
//! weights, and exposes the Metal-backed decode path used by Logan.

use std::{fmt, path::Path, sync::Arc};

pub mod ane;
pub mod codec_adapter;
pub mod config;
pub mod kv;
pub mod metal;
pub mod model;
pub mod weights;

pub mod placement;

pub mod dspark;
pub use codec_adapter::LlamaStateCodec;
pub use config::{LlamaConfig, load_config};
pub use placement::{
    CalibrationConfig, CalibrationKey, CalibrationStatus, ContextBucket, ModelPairIdentity,
    ModelPhase, PlacementBackend, PlacementController, PlacementDecision, PlacementMode,
    PlacementModeParseError, PlacementRequest, PlacementRole, RoundEvidence,
};

pub use kv::{KvCache, KvCheckpoint, KvError};
pub use metal::{BackendPreference, BackendReport, BackendUsed};
pub use model::{DenseModel, DenseSession, DenseTensor, ForwardOutput, LayerTap, ModelIdentity};
pub use weights::{
    MlxQuantization, MlxQuantizedTensorInfo, MlxQuantizedWeightInventory, PackedTensorInfo,
    QuantizedAuxTensorInfo, QuantizedBits, QuantizedDType, QuantizedTensorInfo,
    QuantizedWeightInventory, TensorInfo, WeightInventory, inspect_mlx_quantized_weights,
    inspect_mlx_weights, inspect_quantized_weights, inspect_weights,
};

/// Greedy decode for the neutral `logan run` CLI.
///
/// The prompt is a sequence of token IDs. MLX quantized checkpoints are
/// detected by `DenseModel::load` and materialized to the resident BF16 path
/// before decoding.
pub fn run_greedy(root: &Path, prompt: &[u32], max_new: usize) -> Result<Vec<u32>, String> {
    if prompt.is_empty() {
        return Err("prompt must contain at least one token".into());
    }
    let model = Arc::new(DenseModel::load(root)?);
    let mut session = model.new_session();
    let mut output = session.forward(prompt, &[])?;
    let mut logits = output
        .logits_row(prompt.len() - 1)
        .ok_or("prompt forward returned no final logits")?
        .to_vec();
    let mut generated = Vec::with_capacity(max_new);
    for step in 0..max_new {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(token, _)| token as u32)
            .ok_or("empty logits")?;
        generated.push(next);
        if step + 1 < max_new {
            output = session.forward(&[next], &[])?;
            logits = output
                .logits_row(0)
                .ok_or("decode forward returned no logits")?
                .to_vec();
        }
    }
    Ok(generated)
}
/// Storage dtypes accepted by the MiniCPM5 tensor contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DType {
    F16,
    BF16,
}

impl DType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "F16",
            Self::BF16 => "BF16",
        }
    }

    pub(crate) fn from_safetensors(value: &str) -> Option<Self> {
        match value {
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::BF16),
            _ => None,
        }
    }

    pub(crate) fn from_torch(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "float16" | "f16" | "half" => Some(Self::F16),
            "bfloat16" | "bf16" => Some(Self::BF16),
            _ => None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ContractError {
    Io(String),
    Json(String),
    Invalid(String),
}

impl ContractError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) | Self::Json(message) | Self::Invalid(message) => {
                f.write_str(message)
            }
        }
    }
}

impl std::error::Error for ContractError {}

pub(crate) type ContractResult<T> = Result<T, ContractError>;

pub type TensorDType = DType;
