//! Compiler-wide quantization integrity policy.
//!
//! The compiler must never make a quantized source appear to have more
//! precision than it actually contains.  Target-specific layout changes are
//! allowed only when they are lossless repacks of the same quantization
//! semantics.  Numeric requantization is an explicit operation and may only
//! keep or reduce the source tensor's nominal weight-code bit width.
//!
//! Important: `Q4_K` is called a 4-bit quant because its weight codes are
//! 4-bit.  Its block scales/min metadata make its effective bytes-per-weight
//! larger than exactly 4 bits; that metadata overhead is part of Q4_K and is
//! not an upward requantization.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    GgmlQ4K,
    GgmlQ5_0,
    GgmlQ6K,
    GgmlQ8_0,
}

impl QuantFormat {
    /// Nominal number of bits used by each quantized weight code.  This is
    /// deliberately not "effective bpw", which also includes scale/min block
    /// metadata.
    pub const fn nominal_weight_bits(self) -> u8 {
        match self {
            Self::GgmlQ4K => 4,
            Self::GgmlQ5_0 => 5,
            Self::GgmlQ6K => 6,
            Self::GgmlQ8_0 => 8,
        }
    }

    /// Tensor-level GGML/GGUF quantization names currently required by the
    /// Qwen3.8-Flash-Next REAP-288 Q4_K_M source package.
    pub fn from_ggml_name(name: &str) -> Option<Self> {
        match name {
            "Q4_K" => Some(Self::GgmlQ4K),
            "Q5_0" => Some(Self::GgmlQ5_0),
            "Q6_K" => Some(Self::GgmlQ6K),
            "Q8_0" => Some(Self::GgmlQ8_0),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantTransform {
    /// Byte-for-byte storage preservation, or a target layout transform that
    /// has a proven exact inverse and retains the same quantization semantics.
    LosslessRepack,
    /// A real numeric decode + quantize operation explicitly requested by the
    /// user.  It may keep or lower nominal precision, never raise it.
    ExplicitRequantize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantPolicyError {
    /// A "lossless" path tried to change quantization semantics.
    LosslessFormatChanged {
        source: QuantFormat,
        target: QuantFormat,
    },
    /// A real requantization attempted to increase nominal weight precision.
    UpwardRequantization {
        source: QuantFormat,
        target: QuantFormat,
    },
}

/// Validate a quantized tensor transition before any target lowering occurs.
///
/// This is intentionally fail-closed:
/// - normal compilation uses `LosslessRepack` and therefore cannot silently
///   change Q4_K into Q4_0, INT4-G32, Q5, Q8, FP16, etc.;
/// - an explicitly requested requantization may change families, but its
///   target nominal bit width must be <= the source tensor's bit width.
pub fn validate_quant_transition(
    source: QuantFormat,
    target: QuantFormat,
    transform: QuantTransform,
) -> core::result::Result<(), QuantPolicyError> {
    match transform {
        QuantTransform::LosslessRepack => {
            if source == target {
                Ok(())
            } else {
                Err(QuantPolicyError::LosslessFormatChanged { source, target })
            }
        }
        QuantTransform::ExplicitRequantize => {
            if target.nominal_weight_bits() <= source.nominal_weight_bits() {
                Ok(())
            } else {
                Err(QuantPolicyError::UpwardRequantization { source, target })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_never_moves_above_four_bits() {
        assert_eq!(
            validate_quant_transition(
                QuantFormat::GgmlQ4K,
                QuantFormat::GgmlQ5_0,
                QuantTransform::ExplicitRequantize,
            ),
            Err(QuantPolicyError::UpwardRequantization {
                source: QuantFormat::GgmlQ4K,
                target: QuantFormat::GgmlQ5_0,
            })
        );
        assert!(matches!(
            validate_quant_transition(
                QuantFormat::GgmlQ4K,
                QuantFormat::GgmlQ8_0,
                QuantTransform::ExplicitRequantize,
            ),
            Err(QuantPolicyError::UpwardRequantization { .. })
        ));
    }

    #[test]
    fn preserve_mode_requires_identical_quant_semantics() {
        assert!(
            validate_quant_transition(
                QuantFormat::GgmlQ4K,
                QuantFormat::GgmlQ4K,
                QuantTransform::LosslessRepack,
            )
            .is_ok()
        );

        // Equal or lower nominal precision is NOT enough to call a transform
        // lossless.  Changing quant families requires explicit requantization.
        assert!(matches!(
            validate_quant_transition(
                QuantFormat::GgmlQ8_0,
                QuantFormat::GgmlQ6K,
                QuantTransform::LosslessRepack,
            ),
            Err(QuantPolicyError::LosslessFormatChanged { .. })
        ));
    }

    #[test]
    fn explicit_requantization_may_only_hold_or_reduce_precision() {
        assert!(
            validate_quant_transition(
                QuantFormat::GgmlQ8_0,
                QuantFormat::GgmlQ6K,
                QuantTransform::ExplicitRequantize,
            )
            .is_ok()
        );
        assert!(
            validate_quant_transition(
                QuantFormat::GgmlQ6K,
                QuantFormat::GgmlQ4K,
                QuantTransform::ExplicitRequantize,
            )
            .is_ok()
        );
        assert!(
            validate_quant_transition(
                QuantFormat::GgmlQ4K,
                QuantFormat::GgmlQ4K,
                QuantTransform::ExplicitRequantize,
            )
            .is_ok()
        );
    }

    #[test]
    fn recognizes_required_reap_288_tensor_formats() {
        assert_eq!(
            QuantFormat::from_ggml_name("Q4_K"),
            Some(QuantFormat::GgmlQ4K)
        );
        assert_eq!(
            QuantFormat::from_ggml_name("Q5_0"),
            Some(QuantFormat::GgmlQ5_0)
        );
        assert_eq!(
            QuantFormat::from_ggml_name("Q6_K"),
            Some(QuantFormat::GgmlQ6K)
        );
        assert_eq!(
            QuantFormat::from_ggml_name("Q8_0"),
            Some(QuantFormat::GgmlQ8_0)
        );
        assert_eq!(QuantFormat::from_ggml_name("BF16"), None);
    }
}
