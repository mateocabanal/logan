//! Online, correctness-neutral expert-arrival prediction.
//!
//! Routing stays authoritative. The predictor observes the top-k routes the
//! model already chose and predicts only *arrivals*: experts likely to appear
//! on the next token that are not in the previous route. Previous-route experts
//! are normally already resident, so predicting them cannot hide a cold load.
//!
//! Each layer keeps a bounded u16 expert->expert transition matrix plus one
//! observation count per source expert. Candidate scores sum conditional
//! probabilities P(next expert | previous expert), avoiding raw-count bias.

const DECAY_EVERY: u32 = 4096;

/// Normalised transition evidence with an explicit weight.
///
/// The evidence is peak-normalised *per term* before weighting. Without that,
/// summing two unnormalised conditional tables makes the weight a function of
/// whichever table happens to hold larger raw counts, so a weight sweep would
/// really be measuring table scale rather than signal influence. EXP-034
/// measured this fused form against the runtime's previous equal-weight sum of
/// unnormalised tables and found `temporal = 0.25, spatial = 1.0` better in all
/// 16 holdout x budget cells.
fn accumulate_normalized(
    out: &mut [f32],
    scores: &mut [f32],
    transitions: &[u16],
    from_observations: &[u16],
    experts: usize,
    sources: &[usize],
    weight: f32,
) -> usize {
    scores.fill(0.0);
    let learned =
        accumulate_transition_scores(scores, transitions, from_observations, experts, sources);
    if learned == 0 || weight == 0.0 {
        return learned;
    }
    let peak = scores.iter().copied().fold(0.0_f32, f32::max);
    if peak <= 0.0 {
        return learned;
    }
    let scale = weight / peak;
    for (dst, &value) in out.iter_mut().zip(scores.iter()) {
        *dst += value * scale;
    }
    learned
}

fn accumulate_transition_scores(
    scores: &mut [f32],
    transitions: &[u16],
    from_observations: &[u16],
    experts: usize,
    sources: &[usize],
) -> usize {
    let mut learned_sources = 0usize;
    for &from in sources {
        let observations = from_observations.get(from).copied().unwrap_or(0);
        if observations == 0 {
            continue;
        }
        learned_sources += 1;
        let inv = 1.0_f32 / observations as f32;
        let row = &transitions[from * experts..(from + 1) * experts];
        for (score, &count) in scores.iter_mut().zip(row) {
            *score += count as f32 * inv;
        }
    }
    learned_sources
}

#[derive(Debug)]
pub(crate) struct RoutePredictor {
    experts: usize,
    temporal_weight: f32,
    layers: Vec<LayerPredictor>,
}

/// `QWEN_ROUTE_PREDICT_W_TEMPORAL`, clamped to a sane range.
///
/// EXP-034 measured this fused form offline against the four real prompt
/// families and found 0.25 better than equal weighting in all 16
/// holdout x budget cells; 1.0 restores the previous behaviour for A/B.
pub(crate) fn default_temporal_weight() -> f32 {
    std::env::var("QWEN_ROUTE_PREDICT_W_TEMPORAL")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.25)
        .clamp(0.0, 8.0)
}

#[derive(Debug)]
struct LayerPredictor {
    transitions: Vec<u16>,
    from_observations: Vec<u16>,
    spatial_transitions: Vec<u16>,
    spatial_from_observations: Vec<u16>,
    scores: Vec<f32>,
    /// Scratch for the spatial term so it can be peak-normalised separately from
    /// the temporal term before the two are fused.
    spatial_scores: Vec<f32>,
    /// Scratch for the temporal term (see `spatial_scores`).
    temporal_scores: Vec<f32>,
    /// Relative weight of temporal evidence against spatial (fixed at 1.0).
    /// EXP-034: 0.25 beats the previous equal weighting in every holdout.
    temporal_weight: f32,
    pending_prediction: Vec<usize>,
    pending_previous: Vec<usize>,
    observations_since_decay: u32,
    predicted_common: u64,
    predicted_total: u64,
    actual_arrivals: u64,
    prediction_pairs: u64,
}

impl LayerPredictor {
    fn new(experts: usize, temporal_weight: f32) -> Self {
        Self {
            transitions: vec![0; experts.saturating_mul(experts)],
            from_observations: vec![0; experts],
            spatial_transitions: vec![0; experts.saturating_mul(experts)],
            spatial_from_observations: vec![0; experts],
            scores: vec![0.0; experts],
            spatial_scores: vec![0.0; experts],
            temporal_scores: vec![0.0; experts],
            temporal_weight,
            pending_prediction: Vec::new(),
            pending_previous: Vec::new(),
            observations_since_decay: 0,
            predicted_common: 0,
            predicted_total: 0,
            actual_arrivals: 0,
            prediction_pairs: 0,
        }
    }

    /// Ranked cold-arrival candidates with their scores, best first.
    ///
    /// The caller needs scores, not just a fixed-size list: the budget policy
    /// has to decide how many of these are worth reading from the *shape* of the
    /// ranking (top score and how far the tail falls off), which a pre-trimmed
    /// `Vec<ExpertId>` cannot express.
    fn predict_arrivals_ranked(
        &mut self,
        experts: usize,
        previous: &[usize],
        spatial_previous: &[usize],
        max_budget: usize,
    ) -> Vec<(usize, f32)> {
        self.rank_arrivals(experts, previous, spatial_previous, max_budget, true)
    }

    /// Same ranking, leaving `pending_prediction`/`pending_previous` untouched.
    ///
    /// See [`RoutePredictor::peek_arrivals_ranked`] for why a staging-only
    /// consumer needs the second entry point rather than reusing the scoring one.
    fn peek_arrivals_ranked(
        &mut self,
        experts: usize,
        previous: &[usize],
        spatial_previous: &[usize],
        max_budget: usize,
    ) -> Vec<(usize, f32)> {
        self.rank_arrivals(experts, previous, spatial_previous, max_budget, false)
    }

    /// Shared body of the two ranked readers.
    ///
    /// `record` is the only behavioural difference: it decides whether the
    /// selection is cached for the next `observe` to score. Splitting the
    /// *evidence* computation from the *bookkeeping* is what lets a staging
    /// consumer read the same ranking without corrupting the arrival statistics
    /// the shadow/authoritative modes exist to report.
    fn rank_arrivals(
        &mut self,
        experts: usize,
        previous: &[usize],
        spatial_previous: &[usize],
        max_budget: usize,
        record: bool,
    ) -> Vec<(usize, f32)> {
        if record {
            self.pending_prediction.clear();
            self.pending_previous.clear();
        }
        if experts == 0 || max_budget == 0 {
            return Vec::new();
        }

        let valid_previous: Vec<usize> = previous
            .iter()
            .copied()
            .filter(|&expert| expert < experts)
            .collect();
        let valid_spatial: Vec<usize> = spatial_previous
            .iter()
            .copied()
            .filter(|&expert| expert < experts)
            .collect();
        if valid_previous.is_empty() && valid_spatial.is_empty() {
            return Vec::new();
        }
        if record {
            self.pending_previous.extend_from_slice(&valid_previous);
        }

        self.scores.fill(0.0);
        let mut learned_sources = accumulate_normalized(
            &mut self.scores,
            &mut self.temporal_scores,
            &self.transitions,
            &self.from_observations,
            experts,
            &valid_previous,
            self.temporal_weight,
        );
        learned_sources += accumulate_normalized(
            &mut self.scores,
            &mut self.spatial_scores,
            &self.spatial_transitions,
            &self.spatial_from_observations,
            experts,
            &valid_spatial,
            1.0,
        );

        // Do not speculate during cold start: with no evidence every candidate
        // would score 0.0 and any budget policy reading scores would be reading
        // noise.
        if learned_sources == 0 {
            return Vec::new();
        }

        let want = max_budget.min(experts.saturating_sub(valid_previous.len()));
        let mut ranked: Vec<(usize, f32)> = Vec::with_capacity(want);
        let mut selected = vec![false; experts];
        while ranked.len() < want {
            let mut best: Option<usize> = None;
            for candidate in 0..experts {
                if selected[candidate] || valid_previous.contains(&candidate) {
                    continue;
                }
                if self.scores[candidate] <= 0.0 {
                    continue;
                }
                best = match best {
                    None => Some(candidate),
                    Some(current) => {
                        if self.scores[candidate] > self.scores[current]
                            || (self.scores[candidate] == self.scores[current]
                                && candidate < current)
                        {
                            Some(candidate)
                        } else {
                            Some(current)
                        }
                    }
                };
            }
            let Some(best) = best else {
                break;
            };
            selected[best] = true;
            ranked.push((best, self.scores[best]));
            if record {
                self.pending_prediction.push(best);
            }
        }

        // Truncate the pending list to what was actually selected so `observe`
        // scores precision against issued predictions, not against candidates
        // the budget policy declined to read.
        ranked
    }

    fn predict_arrivals(
        &mut self,
        experts: usize,
        previous: &[usize],
        spatial_previous: &[usize],
        budget: usize,
    ) -> Vec<usize> {
        self.predict_arrivals_ranked(experts, previous, spatial_previous, budget)
            .into_iter()
            .map(|(expert, _)| expert)
            .collect()
    }

    /// The authoritative route: the `k` highest fused scores over *all*
    /// experts, best first.
    ///
    /// This is deliberately not [`Self::predict_arrivals`]. That method exists to
    /// choose speculative reads, so it excludes the previous route (already
    /// resident) and only names *cold arrivals*. An authoritative route is the
    /// route the kernel will execute, so it must be a full `k`-wide set — a
    /// route of arrivals only would be narrower than the native one and change
    /// bytes/token, which is the wrong variable to move when the question is
    /// whether prediction can replace the router.
    ///
    /// Previous-route experts are therefore *in* the ranking with real fused
    /// scores rather than forced in. Keeping them is then a consequence of the
    /// evidence, not an ordering rule — which is what makes a stability bias a
    /// tunable instead of a hard-coded policy.
    ///
    /// `stability_bias` adds `bias x (peak fused score)` to every expert in
    /// `previous`, and `resident_bias` the same for experts in `resident` (the
    /// ones an I/O policy expects to still be staged). `bias == 0.0` is the
    /// unbiased ranking. The scale is the score's own peak, so the knob means
    /// the same thing at every layer despite per-layer score magnitudes
    /// differing.
    fn authoritative_route(
        &mut self,
        experts: usize,
        previous: &[usize],
        spatial_previous: &[usize],
        k: usize,
        stability_bias: f32,
        resident: &[usize],
        resident_bias: f32,
    ) -> Vec<(usize, f32)> {
        if experts == 0 || k == 0 {
            return Vec::new();
        }
        let valid_previous: Vec<usize> = previous
            .iter()
            .copied()
            .filter(|&expert| expert < experts)
            .collect();
        let valid_spatial: Vec<usize> = spatial_previous
            .iter()
            .copied()
            .filter(|&expert| expert < experts)
            .collect();
        if valid_previous.is_empty() && valid_spatial.is_empty() {
            return Vec::new();
        }

        self.scores.fill(0.0);
        let mut learned_sources = accumulate_normalized(
            &mut self.scores,
            &mut self.temporal_scores,
            &self.transitions,
            &self.from_observations,
            experts,
            &valid_previous,
            self.temporal_weight,
        );
        learned_sources += accumulate_normalized(
            &mut self.scores,
            &mut self.spatial_scores,
            &self.spatial_transitions,
            &self.spatial_from_observations,
            experts,
            &valid_spatial,
            1.0,
        );
        // No evidence at all: an authoritative route here would be an arbitrary
        // set, so decline and let the caller run the native router.
        if learned_sources == 0 {
            return Vec::new();
        }

        let peak = self.scores.iter().copied().fold(0.0_f32, f32::max);
        if peak > 0.0 && (stability_bias > 0.0 || resident_bias > 0.0) {
            for &expert in &valid_previous {
                if stability_bias > 0.0 {
                    self.scores[expert] += peak * stability_bias;
                }
            }
            for &expert in resident {
                if expert < experts && resident_bias > 0.0 {
                    self.scores[expert] += peak * resident_bias;
                }
            }
        }

        // Selection sort over the score vector, same tie-break as the router's
        // own top-k (higher score, then lower expert id) so an unbiased
        // authoritative route and the native router break ties identically.
        let mut ranked: Vec<(usize, f32)> = Vec::with_capacity(k.min(experts));
        let mut selected = vec![false; experts];
        while ranked.len() < k {
            let mut best: Option<usize> = None;
            for candidate in 0..experts {
                if selected[candidate] || self.scores[candidate] <= 0.0 {
                    continue;
                }
                best = match best {
                    None => Some(candidate),
                    Some(current) => {
                        if self.scores[candidate] > self.scores[current]
                            || (self.scores[candidate] == self.scores[current]
                                && candidate < current)
                        {
                            Some(candidate)
                        } else {
                            Some(current)
                        }
                    }
                };
            }
            let Some(best) = best else {
                break;
            };
            selected[best] = true;
            ranked.push((best, self.scores[best]));
        }
        ranked
    }

    fn observe(
        &mut self,
        experts: usize,
        previous: &[usize],
        spatial_previous: &[usize],
        current: &[usize],
    ) {
        if !self.pending_previous.is_empty() {
            let actual: Vec<usize> = current
                .iter()
                .copied()
                .filter(|expert| !self.pending_previous.contains(expert))
                .collect();
            let common = self
                .pending_prediction
                .iter()
                .filter(|expert| actual.contains(expert))
                .count() as u64;

            self.predicted_common = self.predicted_common.saturating_add(common);
            self.predicted_total = self
                .predicted_total
                .saturating_add(self.pending_prediction.len() as u64);
            self.actual_arrivals = self.actual_arrivals.saturating_add(actual.len() as u64);
            self.prediction_pairs = self.prediction_pairs.saturating_add(1);
            self.pending_prediction.clear();
            self.pending_previous.clear();
        }

        let valid_previous: Vec<usize> = previous
            .iter()
            .copied()
            .filter(|&expert| expert < experts)
            .collect();
        let valid_spatial: Vec<usize> = spatial_previous
            .iter()
            .copied()
            .filter(|&expert| expert < experts)
            .collect();
        let valid_current: Vec<usize> = current
            .iter()
            .copied()
            .filter(|&expert| expert < experts)
            .collect();
        if valid_current.is_empty() || (valid_previous.is_empty() && valid_spatial.is_empty()) {
            return;
        }

        for &from in &valid_previous {
            self.from_observations[from] = self.from_observations[from].saturating_add(1);
            let row = &mut self.transitions[from * experts..(from + 1) * experts];
            for &to in &valid_current {
                row[to] = row[to].saturating_add(1);
            }
        }
        for &from in &valid_spatial {
            self.spatial_from_observations[from] =
                self.spatial_from_observations[from].saturating_add(1);
            let row = &mut self.spatial_transitions[from * experts..(from + 1) * experts];
            for &to in &valid_current {
                row[to] = row[to].saturating_add(1);
            }
        }

        self.observations_since_decay += 1;
        if self.observations_since_decay >= DECAY_EVERY {
            for count in &mut self.transitions {
                *count >>= 1;
            }
            for count in &mut self.from_observations {
                *count >>= 1;
            }
            for count in &mut self.spatial_transitions {
                *count >>= 1;
            }
            for count in &mut self.spatial_from_observations {
                *count >>= 1;
            }
            self.observations_since_decay = 0;
        }
    }
}

impl RoutePredictor {
    pub(crate) fn new(layers: usize, experts: usize) -> Self {
        Self::with_temporal_weight(layers, experts, default_temporal_weight())
    }

    pub(crate) fn with_temporal_weight(
        layers: usize,
        experts: usize,
        temporal_weight: f32,
    ) -> Self {
        Self {
            experts,
            temporal_weight,
            layers: (0..layers)
                .map(|_| LayerPredictor::new(experts, temporal_weight))
                .collect(),
        }
    }

    pub(crate) fn push_layer(&mut self) {
        self.layers
            .push(LayerPredictor::new(self.experts, self.temporal_weight));
    }

    pub(crate) fn predict_arrivals(
        &mut self,
        layer: usize,
        previous: &[usize],
        spatial_previous: &[usize],
        budget: usize,
    ) -> Vec<usize> {
        let Some(state) = self.layers.get_mut(layer) else {
            return Vec::new();
        };
        state.predict_arrivals(self.experts, previous, spatial_previous, budget)
    }

    /// Ranked `(expert, score)` cold arrivals for one layer, best first.
    pub(crate) fn predict_arrivals_ranked(
        &mut self,
        layer: usize,
        previous: &[usize],
        spatial_previous: &[usize],
        max_budget: usize,
    ) -> Vec<(usize, f32)> {
        let Some(state) = self.layers.get_mut(layer) else {
            return Vec::new();
        };
        state.predict_arrivals_ranked(self.experts, previous, spatial_previous, max_budget)
    }

    /// Ranked cold arrivals for one layer **without disturbing the predictor's
    /// prediction bookkeeping**.
    ///
    /// `predict_arrivals_ranked` caches `pending_prediction`/`pending_previous`
    /// so that the next `observe` can score it. A staging-only consumer reads
    /// this ranking *in addition to* whatever prediction the mode already makes,
    /// so routing through the scoring path would overwrite that pending state
    /// with a second, unobserved prediction and silently corrupt the arrival
    /// statistics (predicted_total would count a set nothing ever scored).
    ///
    /// The evidence itself is a pure read of the transition tables, so the
    /// separate entry point computes the same ranking and writes nothing.
    pub(crate) fn peek_arrivals_ranked(
        &mut self,
        layer: usize,
        previous: &[usize],
        spatial_previous: &[usize],
        max_budget: usize,
    ) -> Vec<(usize, f32)> {
        let Some(state) = self.layers.get_mut(layer) else {
            return Vec::new();
        };
        state.peek_arrivals_ranked(self.experts, previous, spatial_previous, max_budget)
    }

    /// The authoritative route for one layer, best first. See
    /// [`LayerPredictor::authoritative_route`].
    pub(crate) fn authoritative_route(
        &mut self,
        layer: usize,
        previous: &[usize],
        spatial_previous: &[usize],
        k: usize,
        stability_bias: f32,
        resident: &[usize],
        resident_bias: f32,
    ) -> Vec<(usize, f32)> {
        let Some(state) = self.layers.get_mut(layer) else {
            return Vec::new();
        };
        state.authoritative_route(
            self.experts,
            previous,
            spatial_previous,
            k,
            stability_bias,
            resident,
            resident_bias,
        )
    }

    pub(crate) fn observe(
        &mut self,
        layer: usize,
        previous: &[usize],
        spatial_previous: &[usize],
        current: &[usize],
    ) {
        if let Some(state) = self.layers.get_mut(layer) {
            state.observe(self.experts, previous, spatial_previous, current);
        }
    }

    /// (correct predictions, issued predictions, actual cold arrivals, pairs)
    pub(crate) fn stats(&self) -> (u64, u64, u64, u64) {
        self.layers.iter().fold((0, 0, 0, 0), |acc, layer| {
            (
                acc.0.saturating_add(layer.predicted_common),
                acc.1.saturating_add(layer.predicted_total),
                acc.2.saturating_add(layer.actual_arrivals),
                acc.3.saturating_add(layer.prediction_pairs),
            )
        })
    }

    pub(crate) fn layer_stats_at(&self, layer: usize) -> (u64, u64, u64, u64) {
        self.layers
            .get(layer)
            .map(|layer| {
                (
                    layer.predicted_common,
                    layer.predicted_total,
                    layer.actual_arrivals,
                    layer.prediction_pairs,
                )
            })
            .unwrap_or((0, 0, 0, 0))
    }

    pub(crate) fn layer_stats(&self) -> Vec<(u64, u64, u64, u64)> {
        (0..self.layers.len())
            .map(|layer| self.layer_stats_at(layer))
            .collect()
    }

    pub(crate) fn transition_bytes(&self) -> usize {
        self.layers.len().saturating_mul(
            self.experts
                .saturating_mul(self.experts)
                .saturating_mul(std::mem::size_of::<u16>())
                .saturating_add(self.experts.saturating_mul(std::mem::size_of::<u16>()))
                .saturating_mul(2),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_start_does_not_speculate() {
        let mut predictor = RoutePredictor::new(1, 8);
        assert!(predictor.predict_arrivals(0, &[6, 2], &[], 1).is_empty());
    }

    #[test]
    fn learns_a_cold_arrival() {
        let mut predictor = RoutePredictor::new(1, 8);
        // Train [0,1] -> [1,4]: expert 1 persists, expert 4 arrives.
        predictor.observe(0, &[0, 1], &[], &[1, 4]);

        assert_eq!(predictor.predict_arrivals(0, &[0, 1], &[], 1), vec![4]);
        predictor.observe(0, &[0, 1], &[], &[1, 4]);

        assert_eq!(predictor.stats(), (1, 1, 1, 1));
    }

    #[test]
    fn conditional_normalization_avoids_frequency_bias() {
        let mut predictor = RoutePredictor::new(1, 4);

        for _ in 0..60 {
            predictor.observe(0, &[0], &[], &[0, 2]);
        }
        for _ in 0..40 {
            predictor.observe(0, &[0], &[], &[0, 3]);
        }
        for _ in 0..2 {
            predictor.observe(0, &[1], &[], &[1, 3]);
        }

        // Raw counts favor 2 (60 vs 42); conditional evidence from rare expert
        // 1 makes 3 the stronger arrival given [0,1].
        assert_eq!(predictor.predict_arrivals(0, &[0, 1], &[], 1), vec![3]);
    }

    #[test]
    fn never_predicts_an_already_resident_previous_expert() {
        let mut predictor = RoutePredictor::new(1, 4);
        for _ in 0..4 {
            predictor.observe(0, &[0, 1], &[], &[0, 2]);
        }
        let prediction = predictor.predict_arrivals(0, &[0, 1], &[], 2);
        assert!(!prediction.contains(&0));
        assert!(!prediction.contains(&1));
    }

    #[test]
    fn prediction_is_layer_local_and_bounded() {
        let mut predictor = RoutePredictor::new(2, 4);
        predictor.observe(0, &[0], &[], &[0, 3]);

        assert_eq!(predictor.predict_arrivals(0, &[0], &[], 1), vec![3]);
        assert!(predictor.predict_arrivals(1, &[0], &[], 1).is_empty());
        assert_eq!(predictor.transition_bytes(), 2 * 2 * (4 * 4 * 2 + 4 * 2));
    }

    #[test]
    fn learns_spatial_arrival_signal() {
        let mut predictor = RoutePredictor::new(2, 6);
        for _ in 0..4 {
            predictor.observe(1, &[], &[4], &[0, 3]);
        }

        // No temporal transition was trained. Expert 4 from the current token's
        // previous layer is enough to recover cold arrival 3; expert 0 is
        // excluded because it is already present in the previous same-layer route.
        assert_eq!(predictor.predict_arrivals(1, &[0], &[4], 1), vec![3]);
    }

    #[test]
    fn invalid_expert_ids_are_ignored() {
        let mut predictor = RoutePredictor::new(1, 4);
        predictor.observe(0, &[99, 2], &[], &[2, 3, 88]);
        assert_eq!(predictor.predict_arrivals(0, &[2], &[], 1), vec![3]);
    }
}
