//! Proposal-aware speculative sampling over a finite vocabulary.
//!
//! This module intentionally does not share the greedy verifier.  Greedy
//! verification compares argmax tokens; this module samples from the exact
//! target distribution using the proposal-correction rule.

use std::{error::Error, fmt};

/// An error raised before sampling when a probability distribution is not a
/// finite, non-negative, non-zero-mass vector.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingError {
    EmptyDistribution,
    InvalidProbability { index: usize, value: f32 },
    NonFiniteMass,
    ZeroMass,
    VocabularyMismatch { expected: usize, actual: usize },
    InvalidTemperature(f32),
    InvalidRepetitionPenalty(f32),
    InvalidFilter { expected: usize, actual: usize },
    InvalidToken { token: u32, vocabulary: usize },
    InvalidRandomValue(f32),
    RandomSourceExhausted,
    UnsupportedSpeculation(&'static str),
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDistribution => f.write_str("distribution has an empty vocabulary"),
            Self::InvalidProbability { index, value } => write!(
                f,
                "distribution probability {value} at index {index} is invalid"
            ),
            Self::NonFiniteMass => f.write_str("distribution mass is not finite"),
            Self::ZeroMass => f.write_str("distribution has zero mass"),
            Self::VocabularyMismatch { expected, actual } => write!(
                f,
                "distribution vocabulary is {actual}, expected {expected}"
            ),
            Self::InvalidTemperature(value) => {
                write!(f, "temperature {value} is not positive and finite")
            }
            Self::InvalidRepetitionPenalty(value) => {
                write!(f, "repetition penalty {value} is not positive and finite")
            }
            Self::InvalidFilter { expected, actual } => {
                write!(f, "filter has length {actual}, expected {expected}")
            }
            Self::InvalidToken { token, vocabulary } => {
                write!(f, "token {token} is outside vocabulary of {vocabulary}")
            }
            Self::InvalidRandomValue(value) => write!(f, "random value {value} is outside [0, 1)"),
            Self::RandomSourceExhausted => f.write_str("deterministic random source is exhausted"),
            Self::UnsupportedSpeculation(reason) => {
                write!(f, "speculative sampling is unsupported: {reason}")
            }
        }
    }
}

impl Error for SamplingError {}

/// A normalized finite-vocabulary probability distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct Distribution {
    probabilities: Vec<f32>,
}

impl Distribution {
    /// Normalize a non-negative finite vector.  Accepting unnormalized mass
    /// keeps logits/probability callers equivalent while retaining strict
    /// validation of invalid inputs.
    pub fn new(probabilities: &[f32]) -> Result<Self, SamplingError> {
        if probabilities.is_empty() {
            return Err(SamplingError::EmptyDistribution);
        }
        let mut mass = 0.0f32;
        for (index, &value) in probabilities.iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(SamplingError::InvalidProbability { index, value });
            }
            mass += value;
        }
        if !mass.is_finite() {
            return Err(SamplingError::NonFiniteMass);
        }
        if mass <= 0.0 {
            return Err(SamplingError::ZeroMass);
        }
        Ok(Self {
            probabilities: probabilities.iter().map(|value| value / mass).collect(),
        })
    }

    pub fn from_logits(logits: &[f32], temperature: f32) -> Result<Self, SamplingError> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(SamplingError::InvalidTemperature(temperature));
        }
        if logits.is_empty() {
            return Err(SamplingError::EmptyDistribution);
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return Err(SamplingError::NonFiniteMass);
        }
        let mut probabilities = Vec::with_capacity(logits.len());
        for (index, &logit) in logits.iter().enumerate() {
            if !logit.is_finite() {
                return Err(SamplingError::InvalidProbability {
                    index,
                    value: logit,
                });
            }
            probabilities.push(((logit - max) / temperature).exp());
        }
        Self::new(&probabilities)
    }

    pub fn from_logits_filtered(
        logits: &[f32],
        temperature: f32,
        allowed: &[bool],
    ) -> Result<Self, SamplingError> {
        if logits.len() != allowed.len() {
            return Err(SamplingError::InvalidFilter {
                expected: logits.len(),
                actual: allowed.len(),
            });
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(SamplingError::InvalidTemperature(temperature));
        }
        let mut filtered = logits.to_vec();
        for (index, (logit, &is_allowed)) in filtered.iter_mut().zip(allowed).enumerate() {
            if !is_allowed {
                *logit = f32::NEG_INFINITY;
            } else if !logit.is_finite() {
                return Err(SamplingError::InvalidProbability {
                    index,
                    value: *logit,
                });
            }
        }
        if filtered.is_empty() {
            return Err(SamplingError::EmptyDistribution);
        }
        let max = filtered.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return Err(SamplingError::ZeroMass);
        }
        let probabilities = filtered
            .iter()
            .map(|&logit| {
                if logit == f32::NEG_INFINITY {
                    0.0
                } else {
                    ((logit - max) / temperature).exp()
                }
            })
            .collect::<Vec<_>>();
        Self::new(&probabilities)
    }

    pub fn probabilities(&self) -> &[f32] {
        &self.probabilities
    }
    pub fn len(&self) -> usize {
        self.probabilities.len()
    }
    pub fn is_empty(&self) -> bool {
        self.probabilities.is_empty()
    }
    pub fn probability(&self, token: u32) -> Result<f32, SamplingError> {
        self.probabilities
            .get(token as usize)
            .copied()
            .ok_or(SamplingError::InvalidToken {
                token,
                vocabulary: self.len(),
            })
    }

    /// Return normalized max(p-q, 0), the exact rejection distribution.
    pub fn residual_distribution(&self, proposal: &Self) -> Result<Self, SamplingError> {
        if self.len() != proposal.len() {
            return Err(SamplingError::VocabularyMismatch {
                expected: self.len(),
                actual: proposal.len(),
            });
        }
        let values = self
            .probabilities
            .iter()
            .zip(&proposal.probabilities)
            .map(|(&p, &q)| (p - q).max(0.0))
            .collect::<Vec<_>>();
        Self::new(&values)
    }

    /// Compute min(1,p(x)/q(x)).  q(x)=0 is always rejected (the proposal
    /// cannot legitimately produce that event); its mass is recovered by the
    /// residual distribution instead.
    pub fn acceptance_probability(
        &self,
        proposal: &Self,
        token: u32,
    ) -> Result<f32, SamplingError> {
        if self.len() != proposal.len() {
            return Err(SamplingError::VocabularyMismatch {
                expected: self.len(),
                actual: proposal.len(),
            });
        }
        let p = self.probability(token)?;
        let q = proposal.probability(token)?;
        Ok(if q > 0.0 { (p / q).min(1.0) } else { 0.0 })
    }

    fn residual(&self, proposal: &Self) -> Result<Self, SamplingError> {
        self.residual_distribution(proposal)
    }

    fn sample<R: RandomSource>(&self, rng: &mut R) -> Result<u32, SamplingError> {
        let random = rng.next_unit()?;
        validate_random(random)?;
        let mut cumulative = 0.0f32;
        for (token, &probability) in self.probabilities.iter().enumerate() {
            cumulative += probability;
            if random < cumulative {
                return Ok(token as u32);
            }
        }
        // Normalization can leave the final sum one ulp below one.  Choose
        // the final positive-mass entry, never an explicitly filtered token.
        self.probabilities
            .iter()
            .enumerate()
            .rev()
            .find_map(|(token, &probability)| (probability > 0.0).then_some(token as u32))
            .ok_or(SamplingError::ZeroMass)
    }
}

/// RNG seam used by production callers and deterministic tests.  Values must
/// be in [0, 1); the sampler never silently clamps malformed randomness.
pub trait RandomSource {
    fn next_unit(&mut self) -> Result<f32, SamplingError>;
}

/// Deterministic sequence RNG for tests and reproducible decoding.
#[derive(Debug, Clone)]
pub struct SequenceRng {
    values: Vec<f32>,
    cursor: usize,
}

impl SequenceRng {
    pub fn new(values: Vec<f32>) -> Self {
        Self { values, cursor: 0 }
    }
    pub fn remaining(&self) -> usize {
        self.values.len().saturating_sub(self.cursor)
    }
}

impl RandomSource for SequenceRng {
    fn next_unit(&mut self) -> Result<f32, SamplingError> {
        let value = *self
            .values
            .get(self.cursor)
            .ok_or(SamplingError::RandomSourceExhausted)?;
        self.cursor += 1;
        validate_random(value)?;
        Ok(value)
    }
}

fn validate_random(value: f32) -> Result<(), SamplingError> {
    if value.is_finite() && (0.0..1.0).contains(&value) {
        Ok(())
    } else {
        Err(SamplingError::InvalidRandomValue(value))
    }
}

/// A distribution conditional on the already generated prefix.
pub trait ConditionalDistribution {
    fn distribution(&self, prefix: &[u32]) -> Result<Distribution, SamplingError>;
}

impl<F> ConditionalDistribution for F
where
    F: Fn(&[u32]) -> Result<Distribution, SamplingError>,
{
    fn distribution(&self, prefix: &[u32]) -> Result<Distribution, SamplingError> {
        self(prefix)
    }
}

/// A finite first-order Markov model whose rows are logits.  The first row is
/// used for an empty prefix; otherwise the row indexed by the last token is
/// used.  Repetition penalties and support filters are applied to every row
/// after selecting it, so both p and q remain prefix-conditioned.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkovModel {
    initial_logits: Vec<f32>,
    transition_logits: Vec<Vec<f32>>,
    temperature: f32,
    repetition_penalty: Option<f32>,
    allowed: Option<Vec<bool>>,
}

impl MarkovModel {
    pub fn new(
        initial_logits: Vec<f32>,
        transition_logits: Vec<Vec<f32>>,
    ) -> Result<Self, SamplingError> {
        let vocabulary = initial_logits.len();
        if vocabulary == 0 {
            return Err(SamplingError::EmptyDistribution);
        }
        if transition_logits.len() != vocabulary {
            return Err(SamplingError::VocabularyMismatch {
                expected: vocabulary,
                actual: transition_logits.len(),
            });
        }
        for row in &transition_logits {
            if row.len() != vocabulary {
                return Err(SamplingError::VocabularyMismatch {
                    expected: vocabulary,
                    actual: row.len(),
                });
            }
        }
        // Validate all values while preserving logits (including -inf only via
        // filters, never as an unfiltered model input).
        for row in std::iter::once(&initial_logits).chain(transition_logits.iter()) {
            for (index, &value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(SamplingError::InvalidProbability { index, value });
                }
            }
        }
        Ok(Self {
            initial_logits,
            transition_logits,
            temperature: 1.0,
            repetition_penalty: None,
            allowed: None,
        })
    }

    pub fn with_temperature(mut self, temperature: f32) -> Result<Self, SamplingError> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(SamplingError::InvalidTemperature(temperature));
        }
        self.temperature = temperature;
        Ok(self)
    }

    pub fn with_repetition_penalty(mut self, penalty: f32) -> Result<Self, SamplingError> {
        if !penalty.is_finite() || penalty <= 0.0 {
            return Err(SamplingError::InvalidRepetitionPenalty(penalty));
        }
        self.repetition_penalty = Some(penalty);
        Ok(self)
    }

    pub fn with_allowed(mut self, allowed: Vec<bool>) -> Result<Self, SamplingError> {
        if allowed.len() != self.initial_logits.len() {
            return Err(SamplingError::InvalidFilter {
                expected: self.initial_logits.len(),
                actual: allowed.len(),
            });
        }
        self.allowed = Some(allowed);
        Ok(self)
    }

    pub fn vocabulary(&self) -> usize {
        self.initial_logits.len()
    }

    fn logits_for(&self, prefix: &[u32]) -> Result<Vec<f32>, SamplingError> {
        if let Some(&token) = prefix.last() {
            if token as usize >= self.vocabulary() {
                return Err(SamplingError::InvalidToken {
                    token,
                    vocabulary: self.vocabulary(),
                });
            }
        }
        let mut logits = prefix.last().map_or_else(
            || self.initial_logits.clone(),
            |&token| self.transition_logits[token as usize].clone(),
        );
        if let Some(penalty) = self.repetition_penalty {
            let mut seen = vec![false; self.vocabulary()];
            for &token in prefix {
                let index = token as usize;
                if !seen[index] {
                    let value = &mut logits[index];
                    if *value >= 0.0 {
                        *value /= penalty;
                    } else {
                        *value *= penalty;
                    }
                    seen[index] = true;
                }
            }
        }
        Ok(logits)
    }
}

impl ConditionalDistribution for MarkovModel {
    fn distribution(&self, prefix: &[u32]) -> Result<Distribution, SamplingError> {
        let logits = self.logits_for(prefix)?;
        match &self.allowed {
            Some(allowed) => Distribution::from_logits_filtered(&logits, self.temperature, allowed),
            None => Distribution::from_logits(&logits, self.temperature),
        }
    }
}

/// Result of one speculative block.  `tokens` contains the accepted draft
/// prefix followed by either the residual replacement or the target bonus.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeSample {
    pub tokens: Vec<u32>,
    pub proposals: Vec<u32>,
    pub accepted: usize,
    pub rejected: bool,
    pub bonus: bool,
}

impl SpeculativeSample {
    pub fn replacement(&self) -> Option<u32> {
        if self.rejected {
            self.tokens.last().copied()
        } else {
            None
        }
    }
}

/// Perform exact proposal-aware speculative sampling.
///
/// Each proposal is drawn from q at its actual prefix.  It is accepted with
/// min(1,p(x)/q(x)); on rejection the replacement is drawn from normalized
/// max(p-q,0).  If every draft is accepted, one target bonus token is drawn
/// from p after the complete accepted prefix.
pub fn sample_speculative<R, P, Q>(
    prefix: &[u32],
    draft_len: usize,
    target: &P,
    proposal: &Q,
    rng: &mut R,
) -> Result<SpeculativeSample, SamplingError>
where
    R: RandomSource,
    P: ConditionalDistribution,
    Q: ConditionalDistribution,
{
    let mut context = prefix.to_vec();
    let mut proposals = Vec::with_capacity(draft_len);
    let mut tokens = Vec::with_capacity(draft_len.saturating_add(1));
    let mut accepted = 0usize;

    for _ in 0..draft_len {
        let q = proposal.distribution(&context)?;
        let p = target.distribution(&context)?;
        if p.len() != q.len() {
            return Err(SamplingError::VocabularyMismatch {
                expected: p.len(),
                actual: q.len(),
            });
        }
        let candidate = q.sample(rng)?;
        proposals.push(candidate);
        let acceptance = p.acceptance_probability(&q, candidate)?;
        let accept_draw = rng.next_unit()?;
        // Keep zero-probability events rejected even when the deterministic
        // seam returns exactly zero; ratio one is accepted at every draw.
        if acceptance > 0.0 && (acceptance >= 1.0 || accept_draw <= acceptance) {
            tokens.push(candidate);
            context.push(candidate);
            accepted += 1;
            continue;
        }
        let replacement = p.residual(&q)?.sample(rng)?;
        tokens.push(replacement);
        return Ok(SpeculativeSample {
            tokens,
            proposals,
            accepted,
            rejected: true,
            bonus: false,
        });
    }

    let bonus = target.distribution(&context)?.sample(rng)?;
    tokens.push(bonus);
    Ok(SpeculativeSample {
        tokens,
        proposals,
        accepted,
        rejected: false,
        bonus: true,
    })
}

/// Whether a temperature can use stochastic speculative sampling.  Greedy
/// decoding remains available through the independent equality verifier.
pub fn supports_speculative_sampling(temperature: f32) -> bool {
    temperature.is_finite() && temperature > 0.0
}

/// Select the stochastic route only when the target/proposal pair advertises
/// support.  Unsupported model combinations use ordinary target decoding.
pub fn route_for_sampling(temperature: f32, supported: bool) -> crate::dspark::DecodeRoute {
    if supported && supports_speculative_sampling(temperature) {
        crate::dspark::DecodeRoute::DsparkSampling
    } else {
        crate::dspark::DecodeRoute::OrdinaryTarget
    }
}
