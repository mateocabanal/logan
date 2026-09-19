//! Deterministic, plain-data placement policy and calibration.
//!
//! Placement is deliberately a policy seam: execution backends report what
//! actually ran, while this module decides whether a qualified ANE candidate
//! may be selected.  The default is always Metal until complete-round evidence
//! proves a sustained end-to-end gain.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlacementMode {
    Off,
    Auto,
    Draft,
    Ffn,
    Probe,
}

impl Default for PlacementMode {
    fn default() -> Self {
        Self::Auto
    }
}

impl PlacementMode {
    pub const VALUES: [&'static str; 5] = ["off", "auto", "draft", "ffn", "probe"];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::Draft => "draft",
            Self::Ffn => "ffn",
            Self::Probe => "probe",
        }
    }

    pub fn parse(value: &str) -> Result<Self, PlacementModeParseError> {
        value.parse()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementModeParseError {
    pub value: String,
}

impl fmt::Display for PlacementModeParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown placement mode {:?}; expected off|auto|draft|ffn|probe",
            self.value
        )
    }
}

impl std::error::Error for PlacementModeParseError {}

impl FromStr for PlacementMode {
    type Err = PlacementModeParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            "draft" => Ok(Self::Draft),
            "ffn" => Ok(Self::Ffn),
            "probe" => Ok(Self::Probe),
            _ => Err(PlacementModeParseError {
                value: value.to_string(),
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum ModelPhase {
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum ContextBucket {
    Empty,
    Short,
    Medium,
    Long,
}

impl ContextBucket {
    pub fn from_tokens(tokens: usize) -> Self {
        match tokens {
            0 => Self::Empty,
            1..=512 => Self::Short,
            513..=4096 => Self::Medium,
            _ => Self::Long,
        }
    }
}

/// Stable model-pair identity used to prevent target/draft and host changes
/// from sharing calibration results.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct ModelPairIdentity {
    pub target: String,
    pub draft: String,
    pub precision: String,
    pub chip: String,
    pub os: String,
}

impl ModelPairIdentity {
    pub fn new(
        target: impl Into<String>,
        draft: impl Into<String>,
        precision: impl Into<String>,
        chip: impl Into<String>,
        os: impl Into<String>,
    ) -> Self {
        Self {
            target: target.into(),
            draft: draft.into(),
            precision: precision.into(),
            chip: chip.into(),
            os: os.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum PlacementRole {
    TargetFfn,
    Draft,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct CalibrationKey {
    pub pair: ModelPairIdentity,
    pub phase: ModelPhase,
    pub context: ContextBucket,
    pub role: PlacementRole,
}

impl CalibrationKey {
    pub fn new(
        pair: ModelPairIdentity,
        phase: ModelPhase,
        context: ContextBucket,
        role: PlacementRole,
    ) -> Self {
        Self {
            pair,
            phase,
            context,
            role,
        }
    }
}

/// Wall-clock measurements from one complete target/draft round.  All fields
/// are caller measurements; the policy never synthesizes missing timings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RoundEvidence {
    pub wall_ms: f64,
    pub draft_ms: f64,
    pub verification_ms: f64,
    pub handoff_ms: f64,
    pub fallback_ms: f64,
    pub metal_wall_ms: f64,
    pub ane_wall_ms: f64,
    pub complete_round: bool,
}

impl RoundEvidence {
    pub fn valid(self) -> bool {
        self.complete_round
            && self.wall_ms.is_finite()
            && self.wall_ms > 0.0
            && self.metal_wall_ms.is_finite()
            && self.metal_wall_ms > 0.0
            && self.ane_wall_ms.is_finite()
            && self.ane_wall_ms > 0.0
            && self.draft_ms.is_finite()
            && self.draft_ms >= 0.0
            && self.verification_ms.is_finite()
            && self.verification_ms >= 0.0
            && self.handoff_ms.is_finite()
            && self.handoff_ms >= 0.0
            && self.fallback_ms.is_finite()
            && self.fallback_ms >= 0.0
    }

    pub fn gain(self) -> f64 {
        if self.valid() {
            (self.metal_wall_ms - self.ane_wall_ms) / self.metal_wall_ms
        } else {
            0.0
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationConfig {
    pub promote_after_rounds: u32,
    pub demote_after_rounds: u32,
    pub promote_gain: f64,
    pub demote_loss: f64,
    pub max_reprobes: u32,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            promote_after_rounds: 3,
            demote_after_rounds: 2,
            promote_gain: 0.05,
            demote_loss: 0.02,
            max_reprobes: 3,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationStatus {
    pub valid_rounds: u32,
    pub gain_rounds: u32,
    pub loss_rounds: u32,
    pub reprobes: u32,
    pub promoted: bool,
    pub last_gain: f64,
    pub last_round: Option<RoundEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlacementController {
    pub config: CalibrationConfig,
    pub states: BTreeMap<CalibrationKey, CalibrationStatus>,
}

impl Default for PlacementController {
    fn default() -> Self {
        Self::new(CalibrationConfig::default())
    }
}

impl PlacementController {
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            config,
            states: BTreeMap::new(),
        }
    }

    pub fn status(&self, key: &CalibrationKey) -> CalibrationStatus {
        self.states.get(key).cloned().unwrap_or_default()
    }

    pub fn can_probe(&self, key: &CalibrationKey) -> bool {
        let status = self.status(key);
        status.promoted || status.reprobes < self.config.max_reprobes
    }

    /// Add one complete round. Incomplete, non-finite, or zero timings are
    /// ignored so partial/cancelled rounds cannot bias a decision.
    pub fn record_round(
        &mut self,
        key: CalibrationKey,
        evidence: RoundEvidence,
    ) -> CalibrationStatus {
        let config = self.config.clone();
        let state = self.states.entry(key).or_default();
        if !evidence.valid() {
            return state.clone();
        }
        state.valid_rounds = state.valid_rounds.saturating_add(1);
        state.last_gain = evidence.gain();
        state.last_round = Some(evidence);
        if state.last_gain >= config.promote_gain {
            state.gain_rounds = state.gain_rounds.saturating_add(1);
            state.loss_rounds = 0;
            if !state.promoted
                && state.gain_rounds >= config.promote_after_rounds
                && state.reprobes < config.max_reprobes
            {
                state.promoted = true;
            }
        } else if state.last_gain <= -config.demote_loss {
            state.loss_rounds = state.loss_rounds.saturating_add(1);
            state.gain_rounds = 0;
            if state.promoted && state.loss_rounds >= config.demote_after_rounds {
                state.promoted = false;
                state.reprobes = state.reprobes.saturating_add(1);
            }
        } else {
            // Neutral/noisy rounds reset neither backend immediately, but they
            // cannot count toward sustained promotion or demotion.
            state.gain_rounds = 0;
            state.loss_rounds = 0;
        }
        state.clone()
    }
    pub fn observe(&mut self, key: CalibrationKey, evidence: RoundEvidence) -> CalibrationStatus {
        self.record_round(key, evidence)
    }

    pub fn select(&self, request: PlacementRequest) -> PlacementDecision {
        self.decide(request)
    }

    pub fn decide(&self, request: PlacementRequest) -> PlacementDecision {
        let mut decision = PlacementDecision::metal(request.mode, request.role);
        let wants_ane = match request.mode {
            PlacementMode::Off => false,
            PlacementMode::Probe => true,
            PlacementMode::Draft => request.role == PlacementRole::Draft,
            PlacementMode::Ffn => request.role == PlacementRole::TargetFfn,
            PlacementMode::Auto => self.status(&request.key).promoted,
        };
        if !wants_ane {
            decision.fallback_reason = Some(match request.mode {
                PlacementMode::Off => "ANE disabled by placement policy".into(),
                PlacementMode::Auto => "calibration has not proved a sustained ANE gain".into(),
                PlacementMode::Draft => "draft mode does not place target FFN on ANE".into(),
                PlacementMode::Ffn => "ffn mode does not place draft computation on ANE".into(),
                PlacementMode::Probe => unreachable!(),
            });
            return decision;
        }
        if request.role == PlacementRole::Draft && request.mode == PlacementMode::Ffn {
            decision.fallback_reason = Some("target FFN mode cannot select draft ANE".into());
            return decision;
        }
        if request.role == PlacementRole::TargetFfn && request.mode == PlacementMode::Draft {
            decision.fallback_reason = Some("draft mode cannot select target FFN ANE".into());
            return decision;
        }
        if !request.ane_available {
            decision.fallback_reason = Some("ANE unavailable; Metal fallback".into());
        } else if !request.shape_supported {
            decision.fallback_reason = Some("ANE shape unsupported; Metal fallback".into());
        } else if !request.abi_supported {
            decision.fallback_reason = Some("ANE private ABI unsupported; Metal fallback".into());
        } else {
            decision.backend = PlacementBackend::Ane;
            decision.ane_active = true;
            decision.fallback_reason = None;
        }
        decision
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlacementBackend {
    Metal,
    Ane,
}

impl PlacementBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Ane => "ane",
        }
    }
}

impl fmt::Display for PlacementBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementRequest {
    pub key: CalibrationKey,
    pub mode: PlacementMode,
    pub role: PlacementRole,
    pub ane_available: bool,
    pub shape_supported: bool,
    pub abi_supported: bool,
}

impl PlacementRequest {
    pub fn new(key: &CalibrationKey, mode: PlacementMode, role: PlacementRole) -> Self {
        Self {
            key: key.clone(),
            mode,
            role,
            ane_available: true,
            shape_supported: true,
            abi_supported: true,
        }
    }

    pub fn qualified(
        mut self,
        available: bool,
        shape_supported: bool,
        abi_supported: bool,
    ) -> Self {
        self.ane_available = available;
        self.shape_supported = shape_supported;
        self.abi_supported = abi_supported;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementDecision {
    pub requested_mode: PlacementMode,
    pub role: PlacementRole,
    pub backend: PlacementBackend,
    pub ane_active: bool,
    pub fallback_reason: Option<String>,
}

impl PlacementDecision {
    fn metal(mode: PlacementMode, role: PlacementRole) -> Self {
        Self {
            requested_mode: mode,
            role,
            backend: PlacementBackend::Metal,
            ane_active: false,
            fallback_reason: None,
        }
    }

    pub fn selected_backend(&self) -> PlacementBackend {
        self.backend
    }
    pub fn is_ane_active(&self) -> bool {
        self.ane_active && self.backend == PlacementBackend::Ane
    }
    pub fn actual_backend(&self) -> &'static str {
        self.backend.as_str()
    }
}
