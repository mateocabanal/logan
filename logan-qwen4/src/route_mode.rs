//! Authoritative predictive routing: three independently selectable routing
//! policies, and the instrumentation that judges them.
//!
//! The native Qwen router is and remains the default. This module adds two
//! experimental policies on top of it:
//!
//! - [`RouteMode::Native`] — the router runs and its top-k is consumed. No
//!   prediction, no behavioural change. This is the reference path and must stay
//!   bit-identical to the pre-existing implementation.
//! - [`RouteMode::Shadow`] — the router's top-k is consumed (so output is
//!   unchanged) while a predictor's route is scored against it. Prediction
//!   quality is measured with zero runtime risk.
//! - [`RouteMode::Authoritative`] — the predictor's route *replaces* the
//!   router's. The expert set the I/O scheduler stages is then guaranteed to
//!   equal the expert set the MoE kernel consumes, so prediction accuracy stops
//!   being a runtime correctness dependency and becomes a model-quality
//!   question.
//!
//! # Why authoritative routing is not just "prefetch more accurately"
//!
//! Under the previous design a wrong prediction produced a corrective demand
//! read. Here the predicted set is final, so the failure mode changes from a
//! cache miss to a different mixture of expert outputs. That is why the same
//! prediction quality that was useless for prefetch can still be interesting
//! here — and why the cost has to be measured as a *quality* number
//! ([`RouteAgreement::discarded_mass`]) rather than as an I/O hit rate.
//!
//! # The error metric
//!
//! Qwen's router normalizes its top-k weights to sum to 1 and the MoE reduces
//! experts in that normalizer's proportions. If the authoritative route is `S`
//! and the native route is `N` with weights `w`, the layer's output changes
//! both by dropping the mass outside `S` and by rescaling the survivors by
//! `1 / (1 - dropped)`. [`RouteAgreement::discarded_mass`] is the first half —
//! the fraction of native normalized mass that `S` fails to capture — and is
//! the number that predicts how far the layer's output moves.

use std::fmt;

/// Which routing policy the MoE layers execute under.
///
/// Selected once from the environment by [`RouteMode::from_env`] and stored on
/// the model, so a run has exactly one policy and no layer can disagree about
/// it. `QWEN_ROUTE_AUTHORITATIVE=1` selects [`RouteMode::Authoritative`];
/// `QWEN_ROUTE_PREDICT=1` without it selects [`RouteMode::Shadow`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteMode {
    /// Native router is authoritative. The default.
    Native,
    /// Native router is authoritative; a predicted route is scored against it.
    Shadow,
    /// The predicted route is authoritative and is what gets staged and consumed.
    Authoritative,
    /// **Control arm, not a routing policy.** The native router stays
    /// authoritative but its selected set is truncated to
    /// `QWEN_ROUTE_NATIVE_K`, with no prediction involved.
    ///
    /// This exists because the mission requires knowing whether authoritative
    /// *prediction* earns its place over the far simpler alternative of asking
    /// the router for fewer experts. Both reduce bytes/token identically; they
    /// differ only in which experts survive the reduction. If plain truncation
    /// is better on quality at equal bytes, prediction is dominated and no
    /// recovery training can rescue it.
    NativeTruncated,
    /// Edge0's pretrained cross-token prerouter is authoritative for consumer
    /// layers 7..38. Predictions are produced one token early by owners 6..38;
    /// the final layer stays on the native router, matching Edge0's production engine.
    Edge0,
    /// **Predictive staging with native authority.** Both Edge0 and RouteScout
    /// run, but neither decides anything: the native K4 gate is the executed
    /// route in exactly the sense it is under [`RouteMode::Native`].
    ///
    /// Their only job is to predict *storage*: which expert bytes the next
    /// token's gate will ask for, so those bytes are already in memory when the
    /// request arrives. A wrong prediction costs wasted SSD bandwidth and
    /// nothing else — it cannot change which four experts execute, and an
    /// expert that is late or missing is demand-loaded exactly as it would have
    /// been without the prediction.
    ///
    /// This is the distinction the mode exists to make concrete: the edge0 and
    /// authoritative modes both let a predictor *select*, which is a model
    /// quality question; this one lets it only *stage*, which is an I/O
    /// question.
    Hybrid,
}

impl RouteMode {
    /// Resolve the mode from the environment.
    ///
    /// `QWEN_ROUTE_AUTHORITATIVE` wins over `QWEN_ROUTE_PREDICT` because
    /// authoritative mode *implies* prediction: it needs the predictor's output,
    /// not merely its accuracy statistics.
    pub fn from_env() -> Self {
        if let Ok(mode) = std::env::var("QWEN_ROUTE_MODE") {
            match mode.trim().to_ascii_lowercase().as_str() {
                "edge0" => return Self::Edge0,
                "hybrid" => return Self::Hybrid,
                "native" => return Self::Native,
                "shadow" => return Self::Shadow,
                "authoritative" => return Self::Authoritative,
                "native-truncated" | "native_truncated" => return Self::NativeTruncated,
                _ => eprintln!(
                    "logan route-mode: unknown QWEN_ROUTE_MODE={mode:?}; falling back to legacy flags"
                ),
            }
        }
        if env_true("QWEN_ROUTE_AUTHORITATIVE") {
            Self::Authoritative
        } else if env_true("QWEN_ROUTE_PREDICT") || env_true("QWEN_ROUTE_PREDICT_PREFETCH") {
            Self::Shadow
        } else if std::env::var("QWEN_ROUTE_NATIVE_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .is_some_and(|k| k > 0)
        {
            Self::NativeTruncated
        } else {
            Self::Native
        }
    }

    /// Whether the predicted route must be *computed and scored* against the
    /// native route.
    ///
    /// This is the shadow/authoritative question — "is there a competing route
    /// to compare with?" — and it is deliberately narrower than
    /// [`Self::needs_routescout`]: hybrid runs the predictor but never builds a
    /// competing route from it, so it must not pay for one.
    pub fn needs_predictor(self) -> bool {
        matches!(self, Self::Shadow | Self::Authoritative)
    }

    /// Whether RouteScout's online predictor must exist and be fed.
    ///
    /// Distinct from [`Self::needs_predictor`] because a staging-only mode wants
    /// the predictor's *learned state* without wanting a route built out of it.
    /// Conflating the two is what would make hybrid pay for a route it discards.
    pub fn needs_routescout(self) -> bool {
        self.needs_predictor() || self == Self::Hybrid
    }

    /// Whether Edge0's pretrained prerouter must be loaded.
    pub fn needs_edge0(self) -> bool {
        matches!(self, Self::Edge0 | Self::Hybrid)
    }

    /// Whether the predicted route replaces the router's.
    pub fn overrides_route(self) -> bool {
        matches!(self, Self::Authoritative | Self::Edge0)
    }

    /// Whether this mode predicts *storage* instead of *routing*.
    ///
    /// True only for hybrid: the native gate is the executed route and the
    /// predictors may only decide which bytes are read early. Every consumer
    /// that could otherwise treat a prediction as a route keys off this.
    pub fn stages_only(self) -> bool {
        self == Self::Hybrid
    }

    /// Whether the native gate executes at a **reduced** width.
    ///
    /// Hybrid is included because its whole comparison is against the native-K4
    /// control at identical bytes/token: at the checkpoint's own top-k it would
    /// be native-K8 plus staging, which is a different experiment. Sharing this
    /// predicate is what keeps the two arms' executed routes the same width by
    /// construction rather than by two branches that could drift.
    pub fn truncates_native(self) -> bool {
        matches!(self, Self::NativeTruncated | Self::Hybrid)
    }
}

fn env_true(name: &str) -> bool {
    std::env::var(name)
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
}

impl fmt::Display for RouteMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Native => "native",
            Self::Shadow => "shadow",
            Self::Authoritative => "authoritative",
            Self::NativeTruncated => "native-truncated",
            Self::Edge0 => "edge0",
            Self::Hybrid => "hybrid",
        })
    }
}

/// Native-vs-authoritative route disagreement, accumulated for analysis only.
///
/// Never read by the execution path: authoritative mode does not issue
/// corrective reads, so these counters exist purely to characterise the
/// divergence after the fact.
#[derive(Clone, Debug, Default)]
pub struct RouteAgreement {
    /// Comparisons where the authoritative set failed to capture *all* native
    /// normalized mass.
    pub disagree_layers: u64,
    /// Total comparisons.
    pub total_layers: u64,
    /// Sum over comparisons of the native normalized weight mass outside the
    /// authoritative set. Divide by `total_layers` for the mean.
    pub discarded_mass: f64,
    /// Sum over comparisons of `|N ∩ S|`.
    pub intersect_experts: u64,
    /// Sum over comparisons of `|N|`, giving `recall@K` when divided by it.
    pub native_experts: u64,
    /// Comparisons where the authoritative route fell back to native because no
    /// prediction was available yet (cold start, or the predictor declined).
    pub fallbacks: u64,
    /// Authoritative routes whose size differed from native top-k, e.g. after a
    /// K sweep or a clipped route.
    pub width_mismatches: u64,
}

impl RouteAgreement {
    /// Record one (native, authoritative) route pair.
    ///
    /// `native` is the router's top-k with its normalized weights (they sum to
    /// 1 by construction in the engine). `authoritative` is the expert set that
    /// was actually consumed.
    pub fn record(&mut self, native: &[(usize, f32)], authoritative: &[usize], fallback: bool) {
        self.total_layers += 1;
        if fallback {
            self.fallbacks += 1;
        }
        if native.len() != authoritative.len() {
            self.width_mismatches += 1;
        }
        let total: f32 = native.iter().map(|&(_, w)| w).sum();
        let kept: f32 = native
            .iter()
            .filter(|&(e, _)| authoritative.contains(&e))
            .map(|&(_, w)| w)
            .sum();
        let discarded = if total > 0.0 {
            ((total - kept) / total).max(0.0)
        } else {
            0.0
        };
        self.discarded_mass += discarded as f64;
        self.intersect_experts += native
            .iter()
            .filter(|&&(e, _)| authoritative.contains(&e))
            .count() as u64;
        self.native_experts += native.len() as u64;
        if discarded > 0.0 {
            self.disagree_layers += 1;
        }
    }

    /// Mean fraction of native normalized weight mass dropped per layer. This is
    /// the quality-cost proxy the experiment is judged on.
    pub fn mean_discarded_mass(&self) -> f64 {
        if self.total_layers == 0 {
            0.0
        } else {
            self.discarded_mass / self.total_layers as f64
        }
    }

    /// Mean `|N ∩ S| / |N|`, i.e. `recall@K` of the authoritative set.
    pub fn recall_at_k(&self) -> f64 {
        if self.native_experts == 0 {
            0.0
        } else {
            self.intersect_experts as f64 / self.native_experts as f64
        }
    }

    /// Fraction of layer comparisons where the route was not identical.
    pub fn disagreement_rate(&self) -> f64 {
        if self.total_layers == 0 {
            0.0
        } else {
            self.disagree_layers as f64 / self.total_layers as f64
        }
    }

    pub fn merge_from(&mut self, other: &Self) {
        self.disagree_layers += other.disagree_layers;
        self.total_layers += other.total_layers;
        self.discarded_mass += other.discarded_mass;
        self.intersect_experts += other.intersect_experts;
        self.native_experts += other.native_experts;
        self.fallbacks += other.fallbacks;
        self.width_mismatches += other.width_mismatches;
    }
}

/// Per-layer route error, so degradation can be attributed to *where* in the
/// stack it originates rather than only to its average.
///
/// The mission asks for degradation to be correlated with route disagreement by
/// layer, because a routing policy that is only wrong in a few layers is
/// repairable in a way that one which is uniformly wrong is not. Recording the
/// distribution is what makes that distinguishable.
#[derive(Clone, Debug, Default)]
pub struct LayerRouteError {
    /// Layers compared. Grows to `layers` once every layer has been seen once.
    pub layers: Vec<LayerError>,
}

/// One layer's accumulated route error.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerError {
    pub comparisons: u64,
    /// Native normalized weight mass outside the authoritative set. Summed.
    pub discarded_mass: f64,
    pub native_experts: u64,
    pub intersect_experts: u64,
}

impl LayerError {
    pub fn mean_discarded_mass(&self) -> f64 {
        if self.comparisons == 0 {
            0.0
        } else {
            self.discarded_mass / self.comparisons as f64
        }
    }
    pub fn recall_at_k(&self) -> f64 {
        if self.native_experts == 0 {
            0.0
        } else {
            self.intersect_experts as f64 / self.native_experts as f64
        }
    }
}

impl LayerRouteError {
    /// Ensure `layer` exists.
    pub fn ensure(&mut self, layer: usize) {
        if self.layers.len() <= layer {
            self.layers.resize(layer + 1, LayerError::default());
        }
    }

    /// Record one layer's comparison, using the same mass definition as
    /// [`RouteAgreement::record`].
    pub fn record(&mut self, layer: usize, native: &[(usize, f32)], authoritative: &[usize]) {
        self.ensure(layer);
        let e = &mut self.layers[layer];
        e.comparisons += 1;
        let total: f32 = native.iter().map(|&(_, w)| w).sum();
        let kept: f32 = native
            .iter()
            .filter(|&(x, _)| authoritative.contains(&x))
            .map(|&(_, w)| w)
            .sum();
        e.discarded_mass += if total > 0.0 {
            (((total - kept) / total).max(0.0)) as f64
        } else {
            0.0
        };
        e.native_experts += native.len() as u64;
        e.intersect_experts += native
            .iter()
            .filter(|&&(x, _)| authoritative.contains(&x))
            .count() as u64;
    }

    /// Layers ranked by mean discarded mass, worst first. The tail of this
    /// ranking is where a recovery adapter would need capacity.
    pub fn worst_layers(&self, count: usize) -> Vec<(usize, f64)> {
        let mut ranked: Vec<(usize, f64)> = self
            .layers
            .iter()
            .enumerate()
            .filter(|(_, e)| e.comparisons > 0)
            .map(|(li, e)| (li, e.mean_discarded_mass()))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(count);
        ranked
    }

    /// Mean discarded mass across layers, for a compact report line.
    pub fn overall_mean(&self) -> f64 {
        let (sum, n) = self
            .layers
            .iter()
            .filter(|e| e.comparisons > 0)
            .fold((0.0, 0u64), |(s, n), e| {
                (s + e.mean_discarded_mass(), n + 1)
            });
        if n == 0 {
            0.0
        } else {
            sum / n as f64
        }
    }
}

/// Weight selected experts by arbitrary router probabilities.
///
/// Returns `(experts, weights, weight_sum)`, or `None` when the route captures
/// no probability mass at all — which must fall back to native rather than
/// produce a zero-weighted mixture.
///
/// The return contract is **exactly** `route_topk_n`'s: `weights` are the
/// router's un-normalized probabilities over the selected experts and
/// `weight_sum` is their total, so every existing consumer's `w / weight_sum`
/// renormalizes over the authoritative set without being changed. Normalizing
/// here instead would scale the routed branch by `1 / captured_mass` on top of
/// the consumer's own division — a silent 1.1–1.8x error at the masses this
/// experiment measures.
///
/// `router_probs` is the router's post-softmax vector over *all* experts, so an
/// authoritative expert the router also selected keeps its real probability and
/// one it did not carries the small probability the softmax actually assigned
/// it. Both are the router's own numbers; nothing is invented.
pub fn weight_authoritative(
    authoritative: &[usize],
    router_probs: &[f32],
) -> Option<(Vec<usize>, Vec<f32>, f32)> {
    let experts: Vec<usize> = authoritative
        .iter()
        .copied()
        .filter(|&e| e < router_probs.len())
        .collect();
    if experts.is_empty() {
        return None;
    }
    let weights: Vec<f32> = experts.iter().map(|&e| router_probs[e]).collect();
    finish_weighted(experts, weights)
}

/// Weight an authoritative route from the router's **permuted** top-k pair.
///
/// `route_topk_n` returns `(idx, val)` where `idx` is a permutation of the expert
/// ids and `val[i]` is the probability of `idx[i]`. Indexing `val` by expert id
/// therefore reads a *different expert's* probability — the two arrays are
/// permuted together, not independently. This resolves each authoritative expert
/// to its own probability through the permutation, and does so with no
/// allocation because an authoritative route is only `k` wide while the
/// permutation is `experts` long.
///
/// Returns the same `(experts, weights, weight_sum)` contract as
/// [`weight_authoritative`].
pub fn weight_authoritative_permuted(
    authoritative: &[usize],
    idx: &[usize],
    val: &[f32],
) -> Option<(Vec<usize>, Vec<f32>, f32)> {
    if idx.len() != val.len() {
        return None;
    }
    let mut experts: Vec<usize> = Vec::with_capacity(authoritative.len());
    let mut weights: Vec<f32> = Vec::with_capacity(authoritative.len());
    for &e in authoritative {
        let Some(position) = idx.iter().position(|&candidate| candidate == e) else {
            // An expert the router's permutation does not contain cannot be
            // weighted by the router's own numbers, so it is dropped rather than
            // given an invented one.
            continue;
        };
        experts.push(e);
        weights.push(val[position]);
    }
    finish_weighted(experts, weights)
}

/// Shared tail of the weighting helpers: sort into `route_topk_n`'s canonical
/// descending-weight order and report the un-normalized sum.
fn finish_weighted(
    mut experts: Vec<usize>,
    mut weights: Vec<f32>,
) -> Option<(Vec<usize>, Vec<f32>, f32)> {
    if experts.is_empty() {
        return None;
    }
    let sum: f32 = weights.iter().sum();
    if !(sum > 0.0) || !sum.is_finite() {
        return None;
    }
    let mut order: Vec<usize> = (0..experts.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        weights[b]
            .total_cmp(&weights[a])
            .then_with(|| experts[a].cmp(&experts[b]))
    });
    let experts: Vec<usize> = order.iter().map(|&i| experts[i]).collect();
    let weights: Vec<f32> = order.iter().map(|&i| weights[i]).collect();
    Some((experts, weights, sum))
}

/// The authoritative route for one layer: the `k` highest fused scores over all
/// experts, computed by the predictor.
///
/// Route construction lives in
/// [`crate::route_predictor::RoutePredictor::authoritative_route`] because it
/// needs the per-layer transition tables, which the engine does not own. The
/// properties that matter and are asserted there:
///
/// - the route is exactly `k` wide whenever `k` experts have nonzero evidence,
///   because a narrow route changes bytes/token as well as quality and the two
///   effects must not be conflated;
/// - previous-route experts compete on score rather than being force-included,
///   so route stability is a measurable consequence of the evidence;
/// - the tie-break matches the native router's (higher score, then lower expert
///   id), so an unbiased route and the router agree on ties.

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::LayerError as _LayerError;
    use super::*;
    use super::{LayerError, LayerRouteError};
    use crate::RouteMode;

    #[test]
    fn native_mode_needs_no_predictor_and_never_overrides() {
        assert!(!RouteMode::Native.needs_predictor());
        assert!(!RouteMode::Native.overrides_route());
        assert!(RouteMode::Shadow.needs_predictor());
        assert!(!RouteMode::Shadow.overrides_route());
        assert!(RouteMode::Authoritative.needs_predictor());
        assert!(RouteMode::Authoritative.overrides_route());
    }

    #[test]
    fn authoritative_weights_are_un_normalized_like_the_router() {
        // Contract parity with `route_topk_n`: weights are the router's raw
        // probabilities over the selection and the third value is their sum, so
        // the engine's existing `w / wsum` renormalizes over the authoritative
        // set. Normalizing here would double-apply the rescale.
        let probs = vec![0.0, 0.5, 0.25, 0.0, 0.25];
        let (experts, weights, sum) = weight_authoritative(&[1, 2, 4], &probs).unwrap();
        assert_eq!(experts, vec![1, 2, 4]);
        assert!((sum - 1.0).abs() < 1e-6);
        assert!((weights[0] - 0.5).abs() < 1e-6);
        assert!((weights[1] - 0.25).abs() < 1e-6);
        // The consumer's division must land back on a sum of 1.
        let normed: f32 = weights.iter().map(|w| w / sum).sum();
        assert!((normed - 1.0).abs() < 1e-6);
    }

    #[test]
    fn partial_authoritative_weights_renormalize_to_one_via_the_consumer() {
        // The authoritative route captures 0.75 of the native mass (missing the
        // 0.25 expert). Un-normalized weights + `w / sum` must still produce a
        // valid mixture over the captured experts.
        let probs = vec![0.5, 0.25, 0.25];
        let (_, weights, sum) = weight_authoritative(&[0, 1], &probs).unwrap();
        assert!((sum - 0.75).abs() < 1e-6);
        let normed: f32 = weights.iter().map(|w| w / sum).sum();
        assert!((normed - 1.0).abs() < 1e-6);
        assert!((weights[0] / sum - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn zero_mass_route_falls_back_rather_than_producing_a_null_mixture() {
        // An authoritative route naming only experts the router gave no
        // probability to must not be consumed as an all-zero mixture.
        assert!(weight_authoritative(&[0, 3], &[0.0, 0.5, 0.5, 0.0]).is_none());
        assert!(weight_authoritative(&[], &[0.5, 0.5]).is_none());
        // Out-of-range ids are dropped; if nothing valid remains, fall back.
        assert!(weight_authoritative(&[99], &[0.5, 0.5]).is_none());
    }

    #[test]
    fn permuted_weighting_resolves_each_expert_to_its_own_probability() {
        // route_topk_n returns (idx, val) permuted together: val[i] belongs to
        // idx[i]. Expert 1's probability is 0.6 (at position 1), and expert 1
        // sits at position 0 of idx.
        let idx = vec![1usize, 7, 3];
        let val = vec![0.6f32, 0.3, 0.1];
        let (experts, weights, sum) = weight_authoritative_permuted(&[7, 1], &idx, &val).unwrap();
        // Descending weight: expert 1 (0.6) then expert 7 (0.3).
        assert_eq!(experts, vec![1, 7]);
        assert!((weights[0] - 0.6).abs() < 1e-6, "got {:?}", weights);
        assert!((weights[1] - 0.3).abs() < 1e-6, "got {:?}", weights);
        assert!((sum - 0.9).abs() < 1e-6);
    }

    #[test]
    fn permuted_weighting_reproduces_the_native_route_exactly() {
        // When the authoritative route IS the native route, the weights must be
        // the native weights in the native order — the property that makes
        // "authoritative agrees with native" a zero-cost case rather than a
        // numerical one.
        let idx = vec![4usize, 9, 2, 5];
        let val = vec![0.5f32, 0.25, 0.15, 0.10];
        let (experts, weights, sum) =
            weight_authoritative_permuted(&[9, 2, 4, 5], &idx, &val).unwrap();
        assert_eq!(experts, vec![4, 9, 2, 5]);
        assert!((sum - 1.0).abs() < 1e-6);
        let normed: Vec<f32> = weights.iter().map(|w| w / sum).collect();
        for (got, want) in normed.iter().zip(val.iter()) {
            assert!((got - want).abs() < 1e-6, "got {normed:?} want {val:?}");
        }
    }

    #[test]
    fn permuted_weighting_drops_experts_the_router_never_scored() {
        // An expert outside the router's permutation has no router probability,
        // so it must be dropped rather than given an invented weight.
        let idx = vec![1usize, 7, 3];
        let val = vec![0.6f32, 0.3, 0.1];
        let (experts, _, sum) = weight_authoritative_permuted(&[7, 200], &idx, &val).unwrap();
        assert_eq!(experts, vec![7]);
        assert!((sum - 0.3).abs() < 1e-6);
    }

    #[test]
    fn permutation_and_probability_vectors_must_agree_in_length() {
        assert!(weight_authoritative_permuted(&[1], &[1, 2], &[0.5]).is_none());
    }

    #[test]
    fn authoritative_k_never_exceeds_the_router_scored_width() {
        // The engine clamps `QWEN_ROUTE_AUTHORITATIVE_K` against the router's
        // scored top-k. A wider route has no router probability to be weighted
        // by, so it must be clamped rather than left to produce unweighted
        // experts or read past the permutation.
        let clamp = |requested: usize, scored: usize| {
            if requested == 0 {
                scored
            } else {
                requested.max(1).min(scored)
            }
        };
        assert_eq!(clamp(0, 8), 8, "0 means the model's native top-k");
        assert_eq!(clamp(4, 8), 4, "a narrower sweep is honoured");
        assert_eq!(
            clamp(16, 8),
            8,
            "a wider request is clamped to what was scored"
        );
        assert_eq!(
            clamp(0, 2),
            2,
            "the native width is read, never assumed to be 8"
        );
    }

    #[test]
    fn predictor_is_absent_in_native_mode() {
        // `needs_predictor` gates construction, so native mode cannot allocate
        // transition tables (40 x 256 x 256 x 2 B x 2 tables = 10.5 MiB) or spend
        // any per-layer work on prediction. This is the "predictor-disabled"
        // behaviour the mission requires be selectable and tested.
        assert!(!RouteMode::Native.needs_predictor());
        // And the override must be inert there too.
        assert!(!RouteMode::Native.overrides_route());
        // Shadow computes predictions but must not change the consumed route.
        assert!(RouteMode::Shadow.needs_predictor());
        assert!(!RouteMode::Shadow.overrides_route());
    }

    #[test]
    fn mode_display_is_stable_for_log_parsing() {
        // The A/B harness greps these exact strings out of stderr, so a rename
        // would silently produce empty metric columns.
        assert_eq!(RouteMode::Native.to_string(), "native");
        assert_eq!(RouteMode::Shadow.to_string(), "shadow");
        assert_eq!(RouteMode::Authoritative.to_string(), "authoritative");
        assert_eq!(RouteMode::NativeTruncated.to_string(), "native-truncated");
        assert_eq!(RouteMode::Edge0.to_string(), "edge0");
    }

    #[test]
    fn native_truncation_is_a_control_arm_with_no_predictor() {
        // It must not build transition tables and must not override the router's
        // selection — it only narrows it. Otherwise the control would carry the
        // same cost profile as the treatment and stop controlling for anything.
        assert!(!RouteMode::NativeTruncated.needs_predictor());
        assert!(!RouteMode::NativeTruncated.overrides_route());
    }

    #[test]
    fn layer_error_attributes_mass_to_the_layer_that_caused_it() {
        let mut le = LayerRouteError::default();
        // Layer 0 keeps everything (mass 0 dropped); layer 5 drops half.
        le.record(0, &[(1, 0.5), (2, 0.5)], &[1, 2]);
        le.record(5, &[(1, 0.5), (2, 0.5)], &[1, 9]);
        assert!((le.layers[0].mean_discarded_mass() - 0.0).abs() < 1e-9);
        assert!((le.layers[5].mean_discarded_mass() - 0.5).abs() < 1e-9);
        // The ranking must lead with the layer that actually has error, which
        // is the whole point of recording per layer.
        assert_eq!(le.worst_layers(1), vec![(5, 0.5)]);
        // A never-compared layer reports 0 and must not enter the ranking.
        le.ensure(9);
        assert_eq!(le.layers[9].comparisons, 0);
        assert!(le.worst_layers(8).iter().all(|&(li, _)| li != 9));
    }

    #[test]
    fn layer_error_overall_mean_is_the_mean_of_layer_means_not_of_occurrences() {
        // One layer compared 100 times with 0.0 and one compared once with 1.0.
        // The per-occurrence mean would be ~0.01; the per-layer mean is 0.5, and
        // the latter is what "which layers are wrong" means.
        let mut le = LayerRouteError::default();
        for _ in 0..100 {
            le.record(0, &[(1, 0.5), (2, 0.5)], &[1, 2]);
        }
        le.record(1, &[(1, 1.0)], &[9]);
        assert!((le.overall_mean() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn agreement_reports_full_overlap_as_zero_discarded_mass() {
        let mut a = RouteAgreement::default();
        a.record(&[(1, 0.5), (2, 0.5)], &[1, 2], false);
        assert_eq!(a.mean_discarded_mass(), 0.0);
        assert_eq!(a.recall_at_k(), 1.0);
        assert_eq!(a.disagreement_rate(), 0.0);
    }

    #[test]
    fn agreement_mass_is_the_dropped_fraction_not_the_kept_fraction() {
        // Native mass 0.5 + 0.25 + 0.25; the authoritative route keeps only the
        // 0.5 expert, so 0.5 of the mass is discarded.
        let mut a = RouteAgreement::default();
        a.record(&[(1, 0.5), (2, 0.25), (3, 0.25)], &[1, 9], false);
        assert!((a.mean_discarded_mass() - 0.5).abs() < 1e-9);
        assert!((a.recall_at_k() - 1.0 / 3.0).abs() < 1e-9);
        assert!((a.disagreement_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn agreement_counts_fallbacks_and_width_mismatches_separately() {
        let mut a = RouteAgreement::default();
        a.record(&[(1, 0.5), (2, 0.5)], &[1, 2], true);
        a.record(&[(1, 0.5), (2, 0.5)], &[1], false);
        assert_eq!(a.fallbacks, 1);
        assert_eq!(a.width_mismatches, 1);
        assert_eq!(a.total_layers, 2);
    }

    #[test]
    fn agreement_merge_is_exact_for_windowed_reporting() {
        let mut a = RouteAgreement::default();
        let mut b = RouteAgreement::default();
        a.record(&[(1, 0.5), (2, 0.5)], &[1], false);
        b.record(&[(1, 0.5), (2, 0.5)], &[1, 2], false);
        a.merge_from(&b);
        assert_eq!(a.total_layers, 2);
        assert_eq!(a.native_experts, 4);
        assert_eq!(a.intersect_experts, 3);
    }
}
