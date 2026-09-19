//! Plain-data identity and shape contracts for ANE qualification programs.
//!
//! These values deliberately contain no `logan_ane` native handles.  They can
//! cross the model/runtime boundary and are safe to use as cache keys, while
//! the owner thread keeps compiled models, IOSurfaces, and channels private.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AnePrecision {
    Fp16,
    Fp32,
    Bf16,
}

impl fmt::Display for AnePrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Fp16 => "fp16",
            Self::Fp32 => "fp32",
            Self::Bf16 => "bf16",
        })
    }
}

/// Physical tensor geometry and the logical model geometry it carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AneShapeLayout {
    pub logical_width: usize,
    pub padded_width: usize,
    pub intermediate_width: usize,
    pub spatial: usize,
    pub layout_version: u32,
}

impl AneShapeLayout {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.logical_width == 0 || self.padded_width == 0 || self.intermediate_width == 0 {
            return Err("ANE shape dimensions must be non-zero");
        }
        if self.logical_width > self.padded_width {
            return Err("ANE logical width exceeds padded width");
        }
        if self.padded_width % 16 != 0 {
            return Err("ANE padded width must be a multiple of 16");
        }
        if self.spatial < 16 || self.spatial % 16 != 0 {
            return Err("ANE spatial dimension must be >= 16 and a multiple of 16");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AnePrivateAbiProbe {
    pub compiler_available: bool,
    pub runtime_available: bool,
    pub async_channel_available: bool,
    pub shared_event_available: bool,
    pub probe_version: u32,
}

impl AnePrivateAbiProbe {
    pub const fn unavailable() -> Self {
        Self {
            compiler_available: false,
            runtime_available: false,
            async_channel_available: false,
            shared_event_available: false,
            probe_version: 0,
        }
    }

    pub const fn is_usable(&self) -> bool {
        self.compiler_available && self.runtime_available
    }
}

/// Every field which can alter generated MIL, weights, or the private ABI
/// execution contract belongs in this identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AneProgramIdentity {
    pub model_digest: [u8; 32],
    pub weight_digest: [u8; 32],
    pub precision: AnePrecision,
    pub graph_version: u32,
    pub shape: AneShapeLayout,
    pub chip: String,
    pub os: String,
    pub private_abi: AnePrivateAbiProbe,
}

impl AneProgramIdentity {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.shape.validate()?;
        if self.graph_version == 0 {
            return Err("ANE graph version must be non-zero");
        }
        if self.chip.trim().is_empty() || self.os.trim().is_empty() {
            return Err("ANE chip and OS identity must be non-empty");
        }
        Ok(())
    }

    /// Stable, human-readable key for `AneProgramCache`.
    pub fn cache_key(&self) -> String {
        format!(
            "ane/v{}/model-{}-weights-{}/{}-shape-{}x{}x{}x{}-layout{}-{}-{}-abi{}{}{}{}-p{}",
            self.graph_version,
            hex_digest(&self.model_digest),
            hex_digest(&self.weight_digest),
            self.precision,
            self.shape.logical_width,
            self.shape.padded_width,
            self.shape.intermediate_width,
            self.shape.spatial,
            self.shape.layout_version,
            self.chip,
            self.os,
            u8::from(self.private_abi.compiler_available),
            u8::from(self.private_abi.runtime_available),
            u8::from(self.private_abi.async_channel_available),
            u8::from(self.private_abi.shared_event_available),
            self.private_abi.probe_version,
        )
    }
}

fn hex_digest(digest: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// A qualification graph plus its identity.  This is still plain data; the
/// native model is compiled only by the same-thread owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AneProgramSpec {
    pub identity: AneProgramIdentity,
    pub mil_text: String,
}

impl AneProgramSpec {
    pub fn new(
        identity: AneProgramIdentity,
        mil_text: impl Into<String>,
    ) -> Result<Self, &'static str> {
        identity.validate()?;
        let mil_text = mil_text.into();
        if mil_text.trim().is_empty() {
            return Err("ANE MIL program must be non-empty");
        }
        Ok(Self { identity, mil_text })
    }
}
