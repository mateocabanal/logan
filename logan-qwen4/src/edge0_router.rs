//! Edge0-compatible pretrained cross-token MoE prerouter.
//!
//! Routing semantics are independently implemented from Edge0's Apache-2.0
//! prerouter design: https://github.com/Edge0-AI/Edge0/tree/main/src/edge0/prerouter
//!
//! One head is owned by layer N. At token t it consumes the MoE input hidden
//! state, this token's executed route one-hot, and the previous token's executed
//! route one-hot, then predicts layer N+1 for token t+1. Predictions are
//! double-buffered so they cannot become visible to the token that produced them.

use std::path::{Path, PathBuf};

use crate::{Cfg, StFile};

pub(crate) const EDGE0_START_LAYER: usize = 7;
pub(crate) const EDGE0_OWNER_FIRST: usize = 6;
pub(crate) const EDGE0_OWNER_LAST: usize = 38;
pub(crate) const EDGE0_HEAD_HIDDEN: usize = 512;
pub(crate) const DEFAULT_EDGE0_TOPK: usize = 4;

#[derive(Clone, Debug)]
pub(crate) struct Edge0Prediction {
    pub experts: Vec<usize>,
    /// Already normalized across the selected experts.
    pub scores: Vec<f32>,
}

#[derive(Debug)]
struct Edge0Head {
    fc1: Vec<u16>,
    fc2: Vec<u16>,
    linear_init: Vec<u16>,
}

#[derive(Debug)]
pub(crate) struct Edge0Router {
    layers: usize,
    hidden: usize,
    experts: usize,
    heads: Vec<Option<Edge0Head>>,
    /// Predictions produced by the previous token and consumable now.
    current: Vec<Option<Edge0Prediction>>,
    /// Predictions being produced by this token for the next token.
    next: Vec<Option<Edge0Prediction>>,
    /// Executed route at each owner layer on the previous token.
    prev_executed: Vec<Vec<usize>>,
    token_generation: u64,
    /// Number of experts emitted by the authoritative Edge0 route. Defaults to
    /// the published K4 behavior; Logan-trained adapters may opt into another K.
    route_k: usize,
}

impl Edge0Router {
    pub(crate) fn route_k(&self) -> usize {
        self.route_k
    }

    /// Locate the pretrained prerouter adapter.
    ///
    /// Three locations, in order: an explicit `QWEN_EDGE0_PREROUTER`, *beside*
    /// the checkpoint directory, then *inside* it. The sibling location is the
    /// one the adapter is actually published at — next to the model directory
    /// rather than inside it — so searching only inside would make every
    /// documented setup require the env override.
    pub(crate) fn resolve_weights_path(model: &StFile) -> Result<PathBuf, String> {
        const ADAPTER: &str = "prerouter_edge0_35b.safetensors";
        if let Some(path) = std::env::var_os("QWEN_EDGE0_PREROUTER") {
            let path = PathBuf::from(path);
            if path.is_file() {
                return Ok(path);
            }
            return Err(format!(
                "QWEN_EDGE0_PREROUTER points to missing file {}",
                path.display()
            ));
        }

        let model_dir = model.first_path().and_then(Path::parent);
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Some(dir) = model_dir {
            // Beside the checkpoint directory: `<models>/prerouter_edge0_35b.safetensors`.
            if let Some(parent) = dir.parent() {
                candidates.push(parent.join(ADAPTER));
            }
            // Inside it, for a checkpoint that carries its own adapter.
            candidates.push(dir.join(ADAPTER));
        }
        for candidate in candidates {
            if candidate.is_file() {
                return Ok(candidate);
            }
        }

        Err(format!(
            "route modes 'edge0' and 'hybrid' require {ADAPTER}; set \
             QWEN_EDGE0_PREROUTER=/path/to/{ADAPTER}, or place it beside or inside \
             the model checkpoint directory"
        ))
    }

    pub(crate) fn load_from_env(cfg: &Cfg) -> Result<Self, String> {
        let path = std::env::var_os("QWEN_EDGE0_PREROUTER")
            .map(PathBuf::from)
            .ok_or_else(|| {
                "route mode 'edge0' requires QWEN_EDGE0_PREROUTER for COLI/GGUF sources".to_string()
            })?;
        if !path.is_file() {
            return Err(format!(
                "QWEN_EDGE0_PREROUTER points to missing file {}",
                path.display()
            ));
        }
        Self::load(&path, cfg)
    }

    pub(crate) fn load(path: &Path, cfg: &Cfg) -> Result<Self, String> {
        if cfg.layers != 40 || cfg.hidden != 2048 || cfg.experts != 256 || cfg.moe_inter != 512 {
            return Err(format!(
                "Edge0-35B prerouter is trained for Qwen3.6-35B-A3B                  (40 layers, hidden=2048, experts=256, moe_inter=512); model has                  layers={} hidden={} experts={} moe_inter={}",
                cfg.layers, cfg.hidden, cfg.experts, cfg.moe_inter
            ));
        }
        let route_k = match std::env::var("QWEN_EDGE0_K") {
            Ok(raw) => raw
                .parse::<usize>()
                .map_err(|_| format!("QWEN_EDGE0_K must be an integer, got {raw:?}"))?,
            Err(_) => DEFAULT_EDGE0_TOPK,
        };
        if route_k == 0 || route_k > cfg.topk {
            return Err(format!(
                "Edge0 prerouter route K must be in 1..={}, got {}",
                cfg.topk, route_k
            ));
        }

        let st = StFile::open(path)?;
        let feature_width = cfg.hidden + 2 * cfg.experts;
        let mut heads: Vec<Option<Edge0Head>> = (0..cfg.layers).map(|_| None).collect();
        for owner in EDGE0_OWNER_FIRST..=EDGE0_OWNER_LAST {
            let prefix = format!("layers.{owner}");
            let fc1 = st.f16_bits(
                &format!("{prefix}.fc1.weight"),
                &[EDGE0_HEAD_HIDDEN as u64, feature_width as u64],
            )?;
            let fc2 = st.f16_bits(
                &format!("{prefix}.fc2.weight"),
                &[cfg.experts as u64, EDGE0_HEAD_HIDDEN as u64],
            )?;
            let linear_init = st.f16_bits(
                &format!("{prefix}.linear_init.weight"),
                &[cfg.experts as u64, feature_width as u64],
            )?;
            heads[owner] = Some(Edge0Head {
                fc1,
                fc2,
                linear_init,
            });
        }

        Ok(Self {
            layers: cfg.layers,
            hidden: cfg.hidden,
            experts: cfg.experts,
            heads,
            current: (0..cfg.layers).map(|_| None).collect(),
            next: (0..cfg.layers).map(|_| None).collect(),
            prev_executed: (0..cfg.layers).map(|_| Vec::new()).collect(),
            token_generation: 0,
            route_k,
        })
    }

    /// Advance the cross-token double buffer exactly once at decode-token entry.
    pub(crate) fn begin_token(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
        for prediction in &mut self.next {
            *prediction = None;
        }
        self.token_generation = self.token_generation.saturating_add(1);
    }

    pub(crate) fn reset(&mut self) {
        for prediction in &mut self.current {
            *prediction = None;
        }
        for prediction in &mut self.next {
            *prediction = None;
        }
        for route in &mut self.prev_executed {
            route.clear();
        }
        self.token_generation = 0;
    }

    pub(crate) fn prediction(&self, consumer: usize) -> Option<&Edge0Prediction> {
        // Edge0's production Qwen engine keeps the final layer exact/native:
        // staged/prerouter consumers are 7..=38 for a 40-layer model.  The
        // shipped owner-38 head still produces a consumer-39 prediction, but
        // that prediction is intentionally not consumed.
        if consumer < EDGE0_START_LAYER || consumer >= self.layers.saturating_sub(1) {
            return None;
        }
        self.current.get(consumer)?.as_ref()
    }

    /// Run owner N's published head and write consumer N+1 into NEXT token's
    /// prediction buffer. Returns the consumer and expert set so the caller can
    /// start MetalIO immediately.
    pub(crate) fn predict_next(
        &mut self,
        owner: usize,
        hidden: &[f32],
        executed: &[usize],
    ) -> Result<Option<(usize, Vec<usize>)>, String> {
        Ok(self
            .predict_next_ranked(owner, hidden, executed, self.route_k)?
            .map(|(consumer, ranked)| {
                let experts = ranked.iter().map(|c| c.expert).collect();
                (consumer, experts)
            }))
    }

    /// Run owner N's head and return the consumer's top-`max_candidates` experts
    /// ranked best-first, with their softmax probabilities.
    ///
    /// The head always produces a full `experts`-wide logit vector.
    /// `route_k` is the width the authoritative mode consumes. The published
    /// adapter defaults to K4; Logan-trained adapters may set `QWEN_EDGE0_K`.
    /// A staging consumer can request a wider ranking without changing the
    /// authoritative width.
    ///
    /// `max_candidates` is clamped to at least `route_k` so a caller can
    /// never accidentally ask for a ranking narrower than the authoritative
    /// route that shares this code path.
    pub(crate) fn predict_next_ranked(
        &mut self,
        owner: usize,
        hidden: &[f32],
        executed: &[usize],
        max_candidates: usize,
    ) -> Result<Option<(usize, Vec<RankedCandidate>)>, String> {
        if owner < EDGE0_OWNER_FIRST || owner > EDGE0_OWNER_LAST {
            return Ok(None);
        }
        if hidden.len() != self.hidden {
            return Err(format!(
                "Edge0 owner {owner}: hidden width {} != {}",
                hidden.len(),
                self.hidden
            ));
        }
        let Some(head) = self.heads[owner].as_ref() else {
            return Err(format!("Edge0 owner {owner}: missing prerouter head"));
        };

        let feature_width = self.hidden + 2 * self.experts;
        let mut features = Vec::with_capacity(feature_width);
        features.extend(hidden.iter().copied().map(f32_to_f16_bits));
        append_onehot(&mut features, executed, self.experts);
        append_onehot(&mut features, &self.prev_executed[owner], self.experts);
        debug_assert_eq!(features.len(), feature_width);

        // Edge0 exports FP16 heads. BNNS accumulates to f32; round each linear
        // boundary back to FP16 before the next operation so the adapter stays
        // close to its deployed dtype semantics rather than becoming an FP32
        // network accidentally.
        let mut fc1 = vec![0.0_f32; EDGE0_HEAD_HIDDEN];
        if !crate::ffi::bnns_f16_matmul(
            &head.fc1,
            &features,
            &mut fc1,
            EDGE0_HEAD_HIDDEN,
            feature_width,
        ) {
            return Err(format!("Edge0 owner {owner}: BNNS fc1 failed"));
        }

        let mut activated_bits = Vec::with_capacity(EDGE0_HEAD_HIDDEN);
        for value in fc1 {
            let rounded = f16_to_f32(f32_to_f16_bits(value));
            activated_bits.push(f32_to_f16_bits(gelu_erf(rounded)));
        }

        let mut nonlinear = vec![0.0_f32; self.experts];
        let mut linear = vec![0.0_f32; self.experts];
        if !crate::ffi::bnns_f16_matmul(
            &head.fc2,
            &activated_bits,
            &mut nonlinear,
            self.experts,
            EDGE0_HEAD_HIDDEN,
        ) {
            return Err(format!("Edge0 owner {owner}: BNNS fc2 failed"));
        }
        if !crate::ffi::bnns_f16_matmul(
            &head.linear_init,
            &features,
            &mut linear,
            self.experts,
            feature_width,
        ) {
            return Err(format!("Edge0 owner {owner}: BNNS linear_init failed"));
        }

        let mut logits = vec![0.0_f32; self.experts];
        for i in 0..self.experts {
            let a = f16_to_f32(f32_to_f16_bits(nonlinear[i]));
            let b = f16_to_f32(f32_to_f16_bits(linear[i]));
            logits[i] = f16_to_f32(f32_to_f16_bits(a + b));
        }
        let width = max_candidates.max(self.route_k).min(self.experts);
        let (_, probs) = softmax_over(&logits);
        let ranked = rank_by_score(&probs, width);
        let prediction = select_topk_softmax(&logits, self.route_k);
        debug_assert_eq!(
            prediction.experts,
            ranked
                .iter()
                .take(self.route_k)
                .map(|c| c.expert)
                .collect::<Vec<_>>(),
            "wide ranking must agree with the authoritative top-k on its prefix"
        );

        let consumer = owner + 1;
        self.next[consumer] = Some(prediction);
        self.prev_executed[owner].clear();
        self.prev_executed[owner].extend_from_slice(executed);
        Ok(Some((consumer, ranked)))
    }
}

/// One ranked staging candidate: an expert id and the head's softmax
/// probability for it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RankedCandidate {
    pub expert: usize,
    pub score: f32,
}

/// Softmax over a logit vector, returning `(max, probabilities)`.
fn softmax_over(logits: &[f32]) -> (f32, Vec<f32>) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let denom: f32 = probs.iter().sum();
    if denom.is_finite() && denom > 0.0 {
        for p in &mut probs {
            *p /= denom;
        }
    }
    (max, probs)
}

/// The `width` highest-probability experts, best first, tie-broken by lower id.
///
/// The tie-break matches [`select_topk_softmax`] exactly, so the head has one
/// definition of "best" rather than two that could drift.
fn rank_by_score(probs: &[f32], width: usize) -> Vec<RankedCandidate> {
    let mut ids: Vec<usize> = (0..probs.len()).collect();
    ids.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]).then_with(|| a.cmp(&b)));
    ids.truncate(width.min(ids.len()));
    ids.into_iter()
        .map(|expert| RankedCandidate {
            expert,
            score: probs[expert],
        })
        .collect()
}

fn append_onehot(dst: &mut Vec<u16>, selected: &[usize], experts: usize) {
    let zero = f32_to_f16_bits(0.0);
    let one = f32_to_f16_bits(1.0);
    let start = dst.len();
    dst.resize(start + experts, zero);
    for &expert in selected {
        if expert < experts {
            dst[start + expert] = one;
        }
    }
}

/// The authoritative top-`k`, with its scores renormalized over the selection.
///
/// Built on [`softmax_over`]/[`rank_by_score`] so the ranking rule has exactly
/// one definition — the wide staging ranking and the authoritative route must
/// agree on their shared prefix by construction, not by convention.
fn select_topk_softmax(logits: &[f32], k: usize) -> Edge0Prediction {
    let (_, probs) = softmax_over(logits);
    let ranked = rank_by_score(&probs, k);

    let mut experts: Vec<usize> = Vec::with_capacity(ranked.len());
    let mut scores: Vec<f32> = Vec::with_capacity(ranked.len());
    for candidate in ranked {
        experts.push(candidate.expert);
        scores.push(candidate.score);
    }

    let selected_sum: f32 = scores.iter().sum();
    if selected_sum > 0.0 && selected_sum.is_finite() {
        for score in &mut scores {
            *score /= selected_sum;
        }
    }

    Edge0Prediction { experts, scores }
}

#[inline]
fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x / std::f32::consts::SQRT_2))
}

/// IEEE-754 binary32 -> binary16, round-to-nearest-even.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        if mant == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7c00 | ((mant >> 13) as u16).max(1);
    }

    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x80_0000;
        let shift = (14 - half_exp) as u32;
        let mut half_mant = mantissa >> shift;
        let remainder = mantissa & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && (half_mant & 1) != 0) {
            half_mant += 1;
        }
        return sign | half_mant as u16;
    }

    let mut half_exp_bits = (half_exp as u16) << 10;
    let mut half_mant = mant >> 13;
    let remainder = mant & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (half_mant & 1) != 0) {
        half_mant += 1;
        if half_mant == 0x400 {
            half_mant = 0;
            half_exp_bits += 0x400;
            if half_exp_bits >= 0x7c00 {
                return sign | 0x7c00;
            }
        }
    }
    sign | half_exp_bits | half_mant as u16
}

fn f16_to_f32(bits: u16) -> f32 {
    crate::ggufsource::f16_to_f32(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_known_values() {
        for &value in &[0.0, -0.0, 1.0, -2.0, 0.5, 65504.0] {
            assert_eq!(f16_to_f32(f32_to_f16_bits(value)), value);
        }
    }

    #[test]
    fn onehot_is_a_set() {
        let mut v = Vec::new();
        append_onehot(&mut v, &[3, 1, 3], 5);
        let got: Vec<f32> = v.into_iter().map(f16_to_f32).collect();
        assert_eq!(got, vec![0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn softmax_topk_is_normalized_and_stable() {
        let p = select_topk_softmax(&[0.0, 3.0, 2.0, 3.0, -1.0], 3);
        assert_eq!(p.experts, vec![1, 3, 2]);
        assert!((p.scores.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
