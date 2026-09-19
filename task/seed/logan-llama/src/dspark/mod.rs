//! MiniCPM5 DSpark: a model-specific speculative draft/verify contract.
//!
//! Greedy verification remains independent from stochastic sampling.  The
//! equality verifier is exact for greedy target decoding; positive-temperature
//! decoding uses the proposal-aware sampler only when its support is explicit,
//! otherwise it falls back to ordinary target decoding.

mod model;
mod sampling;
mod verify;
mod weights;

pub use model::{
    DraftAttentionState, DraftProposal, DsparkModel, DsparkSession, DsparkStateCheckpoint,
    MaterializedTaps, ProjectedTargetStates, ProposalContext, TargetHistory,
};
pub use sampling::{
    ConditionalDistribution, Distribution, MarkovModel, RandomSource, SamplingError, SequenceRng,
    SpeculativeSample, route_for_sampling, sample_speculative, supports_speculative_sampling,
};
pub use verify::{
    AlignmentDiagnostic, DecodeControl, VerificationOptions, VerificationResult, verify_greedy,
    verify_greedy_logits, verify_greedy_with_options,
};
pub use weights::{
    DSPARK_BLOCK_WIDTH, DSPARK_HIDDEN_SIZE, DSPARK_MARKOV_RANK, DSPARK_MASK_TOKEN_ID, DSPARK_TAPS,
    DsparkGeometry, DsparkProjection, DsparkTensorInventory, DsparkWeights, TokenizerGeometry,
    validate_target_draft,
};

/// User-facing decoding route.  The stochastic route is distinct from the
/// equality verifier: matching greedy tokens does not establish distributional
/// correctness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeRoute {
    DsparkSampling,
    DsparkGreedy,
    OrdinaryTarget,
}

/// Select a route for a temperature-based decode.  Non-positive and
/// non-finite temperatures deliberately fall back to ordinary target decode;
/// callers that want greedy DSpark verification call it explicitly.
pub fn route_for_temperature(temperature: f32) -> DecodeRoute {
    if supports_speculative_sampling(temperature) {
        DecodeRoute::DsparkSampling
    } else {
        DecodeRoute::OrdinaryTarget
    }
}
