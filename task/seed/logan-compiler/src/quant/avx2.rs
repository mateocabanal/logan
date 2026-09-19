//! AVX2 kernels for the MXFP4 offline quantizer.
//!
//! ## Why this file exists
//!
//! The exporter builds for a baseline x86-64 target, so nothing in the
//! quantizer used AVX2 even on a CPU that has it: each of the eight
//! `E2M1_MAGNITUDES` candidates was compared scalar, one value at a time, on
//! one core out of twelve. A full 48-layer export of 512 experts measured 99%
//! of a single core. These kernels take the two hot operations eight lanes at
//! a time.
//!
//! ## Bit-exactness, not merely closeness
//!
//! `logan_core::math_x86` documents that its accumulator order differs from
//! scalar and that results match only "to f32 reassociation tolerance". That
//! is not acceptable here: the exporter promises byte-reproducible output and
//! the inference pool ingests those bytes. So both kernels below are written
//! to agree with the scalar path bit-for-bit, on every input, not just
//! typical ones. Two facts make that hold:
//!
//! * **The scale maximum is exact.** Group values are non-negative after
//!   clearing the sign, so `andnot` (clear sign) + `max` over lanes selects
//!   precisely the same maximum as the scalar `fold`. No addition is involved,
//!   so there is no reassociation that could introduce a difference.
//!
//! * **Code selection is a sum of exact midpoint comparisons.** The scalar
//!   path searches all eight magnitudes for the smallest error and breaks a
//!   tie toward the even code. Equivalently: the chosen code is the smallest
//!   even code attaining the minimum error, or the smallest code if no even
//!   code attains it. Since `E2M1_MAGNITUDES` strictly increases, only two
//!   adjacent candidates can ever tie, so comparing `magnitude` against the
//!   exact midpoint between each adjacent pair decides the same thing. Ties
//!   fall to whichever code is even, which is why each threshold carries its
//!   own strict/inclusive operator below.
//!
//! A group reaching these kernels has already been checked finite, so no NaN
//! can reach `_mm256_max_ps` or the magnitude comparisons — both of which have
//! NaN semantics that would otherwise need guarding.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::mxfp4::{E2M1_MAGNITUDES, GROUP_SIZE, MAX_E2M1, VALUES_PER_BYTE};

/// Midpoints between adjacent `E2M1_MAGNITUDES`, paired with whether the
/// comparison is inclusive.
///
/// `MAGNITUDES` is `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]`. The `inclusive`
/// flag is true exactly when the *upper* code of the pair is the even one, so
/// a value sitting on the midpoint selects the even code — matching the
/// scalar tie-break. Both endpoints are even, hence a strictly-greater test
/// for pairs 0-1, 2-3, 4-5 and 6-7, and an inclusive one for 1-2, 3-4, 5-6.
const THRESHOLDS: [(f32, bool); 7] = [
    (0.25, false), // codes 0/1, lower even
    (0.75, true),  // codes 1/2, upper even
    (1.25, false), // codes 2/3, lower even
    (1.75, true),  // codes 3/4, upper even
    (2.5, false),  // codes 4/5, lower even
    (3.5, true),   // codes 5/6, upper even
    (5.0, false),  // codes 6/7, lower even
];

/// Maximum `|v|` over `values`, identical to the scalar fold.
///
/// # Safety
///
/// CPU must support `avx2`. Values must be finite (NaN would make the scalar
/// and vector `max` disagree).
#[target_feature(enable = "avx2")]
pub unsafe fn max_abs_avx2(values: &[f32]) -> f32 {
    let n = values.len();
    let sign = _mm256_set1_ps(-0.0);
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        let v = unsafe { _mm256_loadu_ps(values.as_ptr().add(i)) };
        let magnitude = _mm256_andnot_ps(sign, v);
        acc = _mm256_max_ps(acc, magnitude);
        i += 8;
    }
    let mut lanes = [0.0_f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut best = 0.0_f32;
    for lane in lanes {
        if lane > best {
            best = lane;
        }
    }
    while i < n {
        let magnitude = unsafe { *values.get_unchecked(i) }.abs();
        if magnitude > best {
            best = magnitude;
        }
        i += 1;
    }
    best
}

/// Write the E2M1 code for each value in `values` into `codes`.
///
/// Codes are the packed-nibble form: bits 0..2 are the magnitude code, bit 3
/// is the sign.
///
/// # Safety
///
/// CPU must support `avx2`. `values.len() == codes.len()`. All values must be
/// finite.
#[target_feature(enable = "avx2")]
pub unsafe fn codes_avx2(values: &[f32], scale: f32, codes: &mut [u8]) {
    let n = values.len();
    let sign = _mm256_set1_ps(-0.0);
    let scale_v = _mm256_set1_ps(scale);
    let ceiling = _mm256_set1_ps(MAX_E2M1);

    let mut i = 0usize;
    while i + 8 <= n {
        unsafe {
            let v = _mm256_loadu_ps(values.as_ptr().add(i));
            // `|v| / scale`, clipped at the largest representable magnitude.
            let magnitude =
                _mm256_min_ps(_mm256_div_ps(_mm256_andnot_ps(sign, v), scale_v), ceiling);
            // Each true comparison contributes 1. A comparison result is an
            // all-ones mask, which is a NaN as a float, NOT -1.0 — subtracting
            // it directly would poison the accumulator. Masking against 1.0
            // first turns true into exactly 1.0 and false into 0.0, so the sum
            // stays an exact small integer.
            let one = _mm256_set1_ps(1.0);
            let mut code = _mm256_setzero_ps();
            for &(threshold, inclusive) in THRESHOLDS.iter() {
                let t = _mm256_set1_ps(threshold);
                // Both operators are instantiated literally: a const generic
                // cannot come from a runtime flag.
                let hit = if inclusive {
                    _mm256_cmp_ps::<_CMP_GE_OQ>(magnitude, t)
                } else {
                    _mm256_cmp_ps::<_CMP_GT_OQ>(magnitude, t)
                };
                code = _mm256_add_ps(code, _mm256_and_ps(hit, one));
            }
            // `code` is now an exact small integer in 0..=7.
            let mut packed = _mm256_cvtps_epi32(code);
            let negative = _mm256_slli_epi32(_mm256_srli_epi32(_mm256_castps_si256(v), 31), 3);
            packed = _mm256_or_si256(packed, negative);

            let mut lanes = [0_i32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, packed);
            for (slot, lane) in codes[i..i + 8].iter_mut().zip(lanes) {
                *slot = lane as u8;
            }
        }
        i += 8;
    }
    // A trailing partial group is decided scalar; rows are 32-wide in practice
    // so this is a boundary case, not the hot path.
    while i < n {
        let value = unsafe { *values.get_unchecked(i) };
        codes[i] = super::mxfp4::quantize_value(value, scale);
        i += 1;
    }
}

/// Codes for a whole group, laid out exactly as `pack_group` expects.
///
/// Kept as the reference implementation the AVX2 path is checked against.
pub(crate) fn codes_scalar(values: &[f32], scale: f32, codes: &mut [u8]) {
    for (slot, &value) in codes.iter_mut().zip(values.iter()) {
        *slot = super::mxfp4::quantize_value(value, scale);
    }
}

/// Pack per-value codes into two-nibbles-per-byte form.
///
/// Even column in the low nibble, odd in the high. A final unpaired value
/// leaves a zero high nibble, which the decoder never reads because it stops
/// at the declared column count.
pub(crate) fn pack_group(codes: &[u8], out: &mut Vec<u8>) {
    let mut pending: Option<u8> = None;
    for &code in codes {
        match pending.take() {
            None => pending = Some(code & 0x0f),
            Some(low) => out.push(low | ((code & 0x0f) << 4)),
        }
    }
    if let Some(low) = pending {
        out.push(low);
    }
}

/// Whether the AVX2 kernels should be used.
///
/// `LOGAN_MXFP4_SCALAR` forces the scalar path, mirroring the existing
/// `QWEN_NEON_BF16` opt-out: a caller can A/B the two encodings or work around
/// a mis-reported CPU feature.
pub(crate) fn enabled() -> bool {
    if std::env::var_os("LOGAN_MXFP4_SCALAR").is_some() {
        return false;
    }
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every adjacent pair's midpoint must be exactly representable, or the
    /// "exact comparison" argument in the module docs fails.
    #[test]
    fn thresholds_are_the_exact_midpoints() {
        for (index, &(threshold, _)) in THRESHOLDS.iter().enumerate() {
            let low = E2M1_MAGNITUDES[index];
            let high = E2M1_MAGNITUDES[index + 1];
            assert_eq!(threshold, (low + high) / 2.0, "pair {index}");
        }
    }

    #[test]
    fn code_geometry_matches_the_packer() {
        assert_eq!(VALUES_PER_BYTE, 2);
        assert_eq!(GROUP_SIZE, 32);
        assert_eq!(E2M1_MAGNITUDES.len(), 8);
    }
}
