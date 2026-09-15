//! Deterministic BF16 -> OCP MXFP4 packing for offline target lowering.
//!
//! The runtime already consumes MXFP4 as row-major E2M1 nibbles (low nibble
//! first) plus one raw E8M0 scale byte per 32 input values. This module owns
//! the compiler side of that contract so target lowering does not need to
//! rediscover or reimplement the quantization rules.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

use crate::{
    error::{ColicError, Result},
    ir::Matrix,
};

pub const GROUP_SIZE: usize = 32;
pub const VALUES_PER_BYTE: usize = 2;
pub const MAX_E2M1: f32 = 6.0;

/// Positive E2M1 magnitudes. Bit 3 of the packed nibble is the sign bit.
pub const E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedMatrix {
    pub rows: u32,
    pub columns: u32,
    /// Row-major E2M1 nibbles, two logical values per byte. The even column is
    /// stored in the low nibble and the odd column in the high nibble.
    pub weights: Vec<u8>,
    /// Row-major raw E8M0 scales, one byte per 32-column group.
    pub scales: Vec<u8>,
}

impl PackedMatrix {
    pub fn row_bytes(&self) -> usize {
        (self.columns as usize).div_ceil(VALUES_PER_BYTE)
    }

    pub fn scale_bytes_per_row(&self) -> usize {
        (self.columns as usize).div_ceil(GROUP_SIZE)
    }
}

/// Quantize one semantic BF16 matrix without materializing its safetensors
/// shard. The source tensor span is opened/seeks once and then consumed one
/// matrix row at a time, so hot-path I/O remains sequential.
pub fn quantize_matrix(matrix: &Matrix) -> Result<PackedMatrix> {
    if matrix.source.dtype != "BF16" {
        return Err(ColicError::unsupported(
            "MXFP4 quantization",
            format!(
                "matrix at {} has dtype `{}`; Qwen MXFP4 lowering currently requires BF16 source weights",
                matrix.source.source.display(),
                matrix.source.dtype
            ),
        ));
    }
    if matrix.scale.is_some() {
        return Err(ColicError::unsupported(
            "MXFP4 quantization",
            "pre-scaled matrices are not accepted by the BF16 -> MXFP4 pass",
        ));
    }

    let row_source_bytes = u64::from(matrix.columns)
        .checked_mul(2)
        .ok_or_else(|| ColicError::Usage("MXFP4 source row size overflows u64".into()))?;
    let expected = u64::from(matrix.rows)
        .checked_mul(row_source_bytes)
        .ok_or_else(|| ColicError::Usage("MXFP4 source matrix size overflows u64".into()))?;
    if matrix.source.len != expected {
        return Err(ColicError::InvalidSource {
            path: matrix.source.source.clone(),
            detail: format!(
                "BF16 matrix payload is {} bytes, expected {expected} for {}x{}",
                matrix.source.len, matrix.rows, matrix.columns
            ),
        });
    }

    let row_bytes = usize::try_from(row_source_bytes)
        .map_err(|_| ColicError::Usage("MXFP4 row is too large for this host".into()))?;
    let packed_row_bytes = (matrix.columns as usize).div_ceil(VALUES_PER_BYTE);
    let scale_row_bytes = (matrix.columns as usize).div_ceil(GROUP_SIZE);
    let weight_capacity = (matrix.rows as usize)
        .checked_mul(packed_row_bytes)
        .ok_or_else(|| ColicError::Usage("MXFP4 packed matrix size overflows usize".into()))?;
    let scale_capacity = (matrix.rows as usize)
        .checked_mul(scale_row_bytes)
        .ok_or_else(|| ColicError::Usage("MXFP4 scale matrix size overflows usize".into()))?;

    let mut file = File::open(&matrix.source.source).map_err(|source| ColicError::Io {
        path: matrix.source.source.clone(),
        source,
    })?;
    file.seek(SeekFrom::Start(matrix.source.offset))
        .map_err(|source| ColicError::Io {
            path: matrix.source.source.clone(),
            source,
        })?;

    let mut weights = Vec::with_capacity(weight_capacity);
    let mut scales = Vec::with_capacity(scale_capacity);
    let mut source_row = vec![0_u8; row_bytes];
    for _ in 0..matrix.rows {
        file.read_exact(&mut source_row)
            .map_err(|source| ColicError::Io {
                path: matrix.source.source.clone(),
                source,
            })?;
        quantize_bf16_row(&source_row, &mut weights, &mut scales)?;
    }

    debug_assert_eq!(weights.len(), weight_capacity);
    debug_assert_eq!(scales.len(), scale_capacity);
    Ok(PackedMatrix {
        rows: matrix.rows,
        columns: matrix.columns,
        weights,
        scales,
    })
}

/// Pack one little-endian BF16 row. This is public so target lowering can
/// stream directly into an expert record later without retaining all packed
/// weight bytes in memory.
pub fn quantize_bf16_row(
    row_bytes: &[u8],
    packed_weights: &mut Vec<u8>,
    scales: &mut Vec<u8>,
) -> Result<()> {
    if !row_bytes.len().is_multiple_of(2) {
        return Err(ColicError::Usage(
            "BF16 row has an odd byte count during MXFP4 quantization".into(),
        ));
    }
    let columns = row_bytes.len() / 2;
    let mut values = Vec::with_capacity(columns.min(GROUP_SIZE));
    let mut nibbles = Vec::with_capacity(columns);

    for group_start in (0..columns).step_by(GROUP_SIZE) {
        let group_end = (group_start + GROUP_SIZE).min(columns);
        values.clear();
        for column in group_start..group_end {
            let offset = column * 2;
            let bits = u16::from_le_bytes([row_bytes[offset], row_bytes[offset + 1]]);
            let value = f32::from_bits(u32::from(bits) << 16);
            if !value.is_finite() {
                return Err(ColicError::Usage(format!(
                    "MXFP4 quantization refuses non-finite BF16 value at column {column}"
                )));
            }
            values.push(value);
        }

        let (scale_code, scale) = choose_scale(&values);
        scales.push(scale_code);
        for &value in &values {
            nibbles.push(quantize_value(value, scale));
        }
    }

    for pair in nibbles.chunks(2) {
        let low = pair[0] & 0x0f;
        let high = pair.get(1).copied().unwrap_or(0) & 0x0f;
        packed_weights.push(low | (high << 4));
    }
    Ok(())
}

/// Choose the smallest runtime-supported power-of-two scale that can hold the
/// largest magnitude without E2M1 saturation. This is a deterministic PTQ
/// policy; OCP permits conversion algorithms other than its baseline recipe.
///
/// The existing kernels decode E8M0 through the Float32 exponent-bit fast path,
/// which is exact for codes 1..=254. Code 0 is a valid OCP E8M0 encoding for
/// 2^-127, but those kernels intentionally do not implement that denormal edge.
/// We therefore never emit code 0; an all-zero block uses scale 1.0 instead.
fn choose_scale(values: &[f32]) -> (u8, f32) {
    choose_scale_from_max(values.iter().fold(0.0_f32, |acc, value| acc.max(value.abs())))
}

/// Scale selection from a group's maximum magnitude.
///
/// Split out from `choose_scale` so a SIMD path can compute the maximum with
/// its own kernel and still share this decision exactly. Everything below is
/// integer or exact-fp, so a different maximum is the only way to reach a
/// different answer — which is what makes the vectorized path bit-identical
/// rather than merely close.
pub(crate) fn choose_scale_from_max(max_abs: f32) -> (u8, f32) {
    if max_abs == 0.0 {
        return (127, 1.0);
    }

    let bits = max_abs.to_bits();
    let biased = ((bits >> 23) & 0xff) as i32;
    let max_exp = if biased == 0 {
        // f32 subnormal: value = mantissa * 2^-149.
        let mantissa = bits & 0x007f_ffff;
        (31 - mantissa.leading_zeros() as i32) - 149
    } else {
        biased - 127
    };
    let mut scale_exp = max_exp - 2; // E2M1's largest power-of-two magnitude is 4.
    scale_exp = scale_exp.clamp(-126, 127);
    let mut scale_code = (scale_exp + 127) as u8;
    let mut scale = runtime_e8m0_to_f32(scale_code);

    // Values in [6*scale, 8*scale) need the next power-of-two scale to avoid
    // clipping at the E2M1 maximum magnitude 6.
    if max_abs > MAX_E2M1 * scale && scale_exp < 127 {
        scale_exp += 1;
        scale_code = (scale_exp + 127) as u8;
        scale = runtime_e8m0_to_f32(scale_code);
    }
    (scale_code, scale)
}

pub(crate) fn quantize_value(value: f32, scale: f32) -> u8 {
    let magnitude = (value.abs() / scale).min(MAX_E2M1);
    let mut best_code = 0_u8;
    let mut best_error = f32::INFINITY;
    for (code, candidate) in E2M1_MAGNITUDES.iter().copied().enumerate() {
        let error = (magnitude - candidate).abs();
        // E2M1's positive code parity tracks the low mantissa bit for the
        // adjacent representable values. Prefer even codes on an exact tie,
        // matching round-to-nearest-even at midpoint values.
        if error < best_error || (error == best_error && (code & 1) == 0 && (best_code & 1) != 0) {
            best_error = error;
            best_code = code as u8;
        }
    }
    if value.is_sign_negative() {
        best_code | 0x8
    } else {
        best_code
    }
}

#[inline]
pub fn runtime_e8m0_to_f32(code: u8) -> f32 {
    debug_assert!((1..=254).contains(&code));
    f32::from_bits(u32::from(code) << 23)
}

// ---------------------------------------------------------------------------
// FP8 (E4M3) source support
// ---------------------------------------------------------------------------

/// Decode one byte of OCP FP8 E4M3 (bias 7) to f32.
///
/// Layout: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits. Two bit
/// patterns are NaN: `S.1111.111` (all-ones exponent AND mantissa), and — in
/// the FNUZ variant — `S.0000.000`, which ordinary E4M3 uses for zero. This
/// follows the ordinary (non-FNUZ) convention, which is what Qwen FP8
/// checkpoints use.
///
/// Deliberately not delegated to an existing helper: the compiler's existing
/// E4M3 code is an *encoder* with a saturation refusal
/// (`encode_bf16_e4m3`/`e4m3_positive` in `target/lowering_tensor.rs`) written
/// for the PLE path, not a general decoder. Reusing it here would mean
/// unpicking its scale handling to get a plain value.
pub fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 3) & 0xf;
    let mant = byte & 0x7;

    // S.1111.111 is NaN. Everything else with exponent 15 is finite E4M3
    // (largest magnitude 448), unlike IEEE-754 binary16/32 where a max
    // exponent means infinity.
    if exp == 0xf && mant == 0x7 {
        return f32::NAN;
    }

    let magnitude = if exp == 0 {
        // Subnormal: mantissa * 2^-9 (2^(1-7) * mant/8 with mant as 3 bits).
        (mant as f32) * (1.0 / 512.0)
    } else {
        let e = exp as i32 - 7;
        (1.0 + (mant as f32) / 8.0) * (2.0_f32).powi(e)
    };
    if sign == 1 { -magnitude } else { magnitude }
}

/// Number of bytes one element of `dtype` occupies.
pub fn dtype_element_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "BF16" | "F16" => Some(2),
        "F32" => Some(4),
        // E4M3FN has the finite-only E4M3 encoding decoded by
        // `e4m3_to_f32`.  FNUZ, E5M2 and E8M0 have different encodings and
        // must not be accepted until they have explicit decoders; treating
        // every one-byte float as E4M3 silently corrupts source weights.
        "F8_E4M3" | "F8_E4M3FN" => Some(1),
        _ => None,
    }
}

/// Decode IEEE binary16 to f32. Kept here (rather than in individual importers)
/// so every MXFP4 source path agrees on what `F16` means.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x03ff;
    if exp == 0x1f {
        return if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }
    let magnitude = if exp == 0 {
        (mant as f32) * (2.0_f32).powi(-24)
    } else {
        (1.0 + (mant as f32) / 1024.0) * (2.0_f32).powi(exp as i32 - 15)
    };
    if sign == 1 { -magnitude } else { magnitude }
}

/// Quantize one row of f32 values into the runtime MXFP4 contract.
///
/// The single place groups, scales and nibble packing are decided, so every
/// source dtype takes an identical path once it has been decoded to f32. The
/// BF16 entry point (`quantize_bf16_row`) feeds this; a dtype whose decode
/// produces f32 feeds it directly.
pub fn quantize_f32_row(
    values: &[f32],
    packed_weights: &mut Vec<u8>,
    scales: &mut Vec<u8>,
) -> Result<()> {
    // Fast path: a full 32-value group, one scale code and 16 packed bytes.
    // Groups are 32 wide for every real geometry, so this covers all but the
    // final partial group of a row whose column count is not a multiple of 32.
    #[cfg(target_arch = "x86_64")]
    if crate::quant::avx2::enabled() {
        let mut codes = [0_u8; GROUP_SIZE];
        for chunk in values.chunks(GROUP_SIZE) {
            for &value in chunk {
                if !value.is_finite() {
                    return Err(ColicError::Usage(
                        "MXFP4 quantization refuses a non-finite source value".into(),
                    ));
                }
            }
            let max_abs = unsafe { crate::quant::avx2::max_abs_avx2(chunk) };
            let (scale_code, scale) = choose_scale_from_max(max_abs);
            scales.push(scale_code);
            let codes = &mut codes[..chunk.len()];
            unsafe { crate::quant::avx2::codes_avx2(chunk, scale, codes) };
            crate::quant::avx2::pack_group(codes, packed_weights);
        }
        return Ok(());
    }

    let mut group = Vec::with_capacity(GROUP_SIZE.min(values.len()));
    for chunk in values.chunks(GROUP_SIZE) {
        group.clear();
        for &value in chunk {
            if !value.is_finite() {
                return Err(ColicError::Usage(
                    "MXFP4 quantization refuses a non-finite source value".into(),
                ));
            }
            group.push(value);
        }

        let (scale_code, scale) = choose_scale(&group);
        scales.push(scale_code);
        let nibbles = group.iter().map(|&v| quantize_value(v, scale));
        // Pack two nibbles per byte: even column in the low nibble, odd in the
        // high. A final unpaired value leaves a zero high nibble, which the
        // decoder never reads because it stops at the declared column count.
        let mut pending: Option<u8> = None;
        for nibble in nibbles {
            match pending.take() {
                None => pending = Some(nibble & 0x0f),
                Some(low) => packed_weights.push(low | ((nibble & 0x0f) << 4)),
            }
        }
        if let Some(low) = pending {
            packed_weights.push(low);
        }
    }
    Ok(())
}

/// Quantize one row of source bytes of the given dtype.
///
/// Rejects a dtype it cannot decode rather than guessing, and names the dtype
/// in the error so a caller sees which tensor is unsupported.
pub fn quantize_row(
    row_bytes: &[u8],
    dtype: &str,
    packed_weights: &mut Vec<u8>,
    scales: &mut Vec<u8>,
) -> Result<()> {
    let elem = dtype_element_bytes(dtype).ok_or_else(|| {
        ColicError::unsupported(
            "MXFP4 quantization",
            format!("source dtype `{dtype}` has no decoder for the BF16 -> MXFP4 pass"),
        )
    })?;
    if !row_bytes.len().is_multiple_of(elem) {
        return Err(ColicError::Usage(format!(
            "{dtype} row has {} bytes, not a multiple of its {elem}-byte element",
            row_bytes.len()
        )));
    }

    let decode = |offset: usize| -> f32 {
        match dtype {
            "F8_E4M3" | "F8_E4M3FN" => e4m3_to_f32(row_bytes[offset]),
            "BF16" => {
                let bits = u16::from_le_bytes([row_bytes[offset], row_bytes[offset + 1]]);
                f32::from_bits(u32::from(bits) << 16)
            }
            "F16" => {
                let bits = u16::from_le_bytes([row_bytes[offset], row_bytes[offset + 1]]);
                f16_to_f32(bits)
            }
            "F32" => f32::from_le_bytes([
                row_bytes[offset],
                row_bytes[offset + 1],
                row_bytes[offset + 2],
                row_bytes[offset + 3],
            ]),
            _ => unreachable!("dtype_element_bytes accepted an unhandled dtype"),
        }
    };
    let values: Vec<f32> = (0..row_bytes.len() / elem)
        .map(|i| decode(i * elem))
        .collect();
    quantize_f32_row(&values, packed_weights, scales)
}

#[cfg(test)]
fn decode_nibble(code: u8, scale: u8) -> f32 {
    let magnitude = E2M1_MAGNITUDES[(code & 0x7) as usize];
    let signed = if code & 0x8 != 0 {
        -magnitude
    } else {
        magnitude
    };
    signed * runtime_e8m0_to_f32(scale)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::{ir::Matrix, source::TensorRef};

    /// The AVX2 path must produce the same bytes as the scalar path for every
    /// group it accepts — not merely close ones, since the exporter's output is
    /// consumed by the pool as authoritative weights.
    ///
    /// Values are swept densely around every decision boundary rather than
    /// sampled randomly: the failure mode this guards is a threshold that is
    /// off by one ULP or an inclusive/strict comparison that picks the odd code
    /// on an exact tie, and random floats almost never land on a midpoint.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_every_magnitude_boundary() {
        if !crate::quant::avx2::enabled() {
            return;
        }
        let mut values: Vec<f32> = Vec::new();
        // Each E2M1 magnitude, every adjacent midpoint, and each nudged by a
        // few ULP either way, in both signs.
        let mut probes: Vec<f32> = E2M1_MAGNITUDES.to_vec();
        for pair in E2M1_MAGNITUDES.windows(2) {
            probes.push((pair[0] + pair[1]) / 2.0);
        }
        probes.push(MAX_E2M1);
        for probe in probes {
            for delta in [-3_i32, -1, 0, 1, 3] {
                for sign in [1.0_f32, -1.0] {
                    // Nudging by signed bits wraps below zero into NaN, so the
                    // candidate is filtered rather than trusted: the quantizer
                    // is required to refuse non-finite input, so feeding it one
                    // would test the guard, not the kernel.
                    let bits = (probe.to_bits() as i64 + delta as i64) as u32;
                    let nudged = f32::from_bits(bits);
                    if nudged.is_finite() {
                        values.push(sign * nudged);
                    }
                }
            }
        }
        // Plus a range of scales so the per-group scale choice is exercised,
        // and subnormals, which take a different exponent branch.
        for exponent in -30..30 {
            let scale = 2.0_f32.powi(exponent);
            for base in [0.0_f32, 0.25, 1.0, 3.5, 5.0, 6.0] {
                let value = scale * base;
                if value.is_finite() {
                    values.push(value);
                }
            }
        }
        values.extend_from_slice(&[f32::from_bits(1), f32::from_bits(0x007f_ffff)]);
        assert!(values.iter().all(|v| v.is_finite()));

        // Pad to a whole number of 32-wide groups by repeating the sweep, so
        // every group the kernel sees is a full one.
        let target = values.len().div_ceil(GROUP_SIZE) * GROUP_SIZE;
        let mut index = 0usize;
        while values.len() < target {
            let value = values[index % values.len()];
            values.push(value);
            index += 1;
        }
        assert_eq!(values.len() % GROUP_SIZE, 0);

        let mut scalar_weights = Vec::new();
        let mut scalar_scales = Vec::new();
        // Force the scalar path by draining through the reference helpers.
        let mut codes = [0_u8; GROUP_SIZE];
        for chunk in values.chunks(GROUP_SIZE) {
            let (scale_code, scale) = choose_scale(chunk);
            scalar_scales.push(scale_code);
            let codes = &mut codes[..chunk.len()];
            crate::quant::avx2::codes_scalar(chunk, scale, codes);
            crate::quant::avx2::pack_group(codes, &mut scalar_weights);
        }

        let mut fast_weights = Vec::new();
        let mut fast_scales = Vec::new();
        quantize_f32_row(&values, &mut fast_weights, &mut fast_scales)
            .expect("swept values are finite");

        assert_eq!(
            scalar_scales, fast_scales,
            "scale codes differ between scalar and AVX2 paths"
        );
        assert_eq!(
            scalar_weights.len(),
            fast_weights.len(),
            "packed length differs"
        );
        let first_diff = scalar_weights
            .iter()
            .zip(fast_weights.iter())
            .position(|(a, b)| a != b);
        if let Some(at) = first_diff {
            // Locate the value that produced the differing nibble so the
            // failure names an input rather than a byte offset.
            let value_index = at * 2;
            let lo = scalar_weights[at] & 0x0f;
            let hi = scalar_weights[at] >> 4;
            let flo = fast_weights[at] & 0x0f;
            let fhi = fast_weights[at] >> 4;
            let window_lo = value_index.saturating_sub(4);
            let window_hi = (value_index + 6).min(values.len());
            panic!(
                "packed bytes differ at {at}: scalar=0x{:02x} (lo={lo} hi={hi}) \
                 avx2=0x{:02x} (lo={flo} fhi={fhi}); values[{window_lo}..{window_hi}]={:?}",
                scalar_weights[at],
                fast_weights[at],
                &values[window_lo..window_hi]
            );
        }
        assert_eq!(first_diff, None, "packed bytes differ");
    }

    /// A partial final group is 1..=31 values; the tail must still round-trip.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_handles_partial_final_group() {
        if !crate::quant::avx2::enabled() {
            return;
        }
        for len in 1..=GROUP_SIZE + 5 {
            let values: Vec<f32> = (0..len).map(|i| (i as f32) * 0.37 - 3.1).collect();
            let mut weights = Vec::new();
            let mut scales = Vec::new();
            quantize_f32_row(&values, &mut weights, &mut scales).unwrap();
            assert_eq!(scales.len(), values.len().div_ceil(GROUP_SIZE), "len {len}");
            assert_eq!(
                weights.len(),
                values.len().div_ceil(VALUES_PER_BYTE),
                "len {len}"
            );
        }
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            let bf16 = (value.to_bits() >> 16) as u16;
            bytes.extend_from_slice(&bf16.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn packs_all_e2m1_codes_low_nibble_first() {
        let values = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        quantize_bf16_row(&bf16_bytes(&values), &mut weights, &mut scales).unwrap();
        assert_eq!(scales, vec![127]);
        assert_eq!(
            weights,
            vec![0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]
        );
    }

    #[test]
    fn zero_block_uses_finite_nonzero_e8m0_scale() {
        let values = [0.0_f32; GROUP_SIZE];
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        quantize_bf16_row(&bf16_bytes(&values), &mut weights, &mut scales).unwrap();
        assert_eq!(scales, vec![127]);
        assert!(weights.iter().all(|byte| *byte == 0));
    }

    #[test]
    fn midpoint_rounding_is_ties_to_even() {
        // With scale=1 these are exact midpoints between adjacent positive
        // E2M1 values. OCP requires roundTiesToEven support.
        let values = [0.25_f32, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        quantize_bf16_row(&bf16_bytes(&values), &mut weights, &mut scales).unwrap();
        assert_eq!(scales, vec![127]);
        let codes: Vec<u8> = (0..values.len())
            .map(|i| {
                let byte = weights[i / 2];
                if i % 2 == 0 { byte & 0x0f } else { byte >> 4 }
            })
            .collect();
        assert_eq!(codes, vec![0, 2, 2, 4, 4, 6, 6]);
    }

    #[test]
    fn scale_is_power_of_two_and_avoids_saturation() {
        let values = [7.0_f32; GROUP_SIZE];
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        quantize_bf16_row(&bf16_bytes(&values), &mut weights, &mut scales).unwrap();
        assert_eq!(scales, vec![128]); // 2^(128-127) = 2
        let code = weights[0] & 0x0f;
        assert_eq!(decode_nibble(code, scales[0]), 8.0);
    }

    #[test]
    fn scale_groups_restart_for_every_row() {
        let path = std::env::temp_dir().join(format!("colic-mxfp4-{}", std::process::id()));
        let mut source = Vec::new();
        source.extend_from_slice(&bf16_bytes(&[1.0; 33]));
        source.extend_from_slice(&bf16_bytes(&[16.0; 33]));
        fs::write(&path, &source).unwrap();
        let matrix = Matrix {
            source: TensorRef {
                source: path.clone(),
                offset: 0,
                len: source.len() as u64,
                dtype: "BF16".into(),
                shape: vec![2, 33],
            },
            rows: 2,
            columns: 33,
            scale: None,
        };
        let packed = quantize_matrix(&matrix).unwrap();
        assert_eq!(packed.row_bytes(), 17);
        assert_eq!(packed.scale_bytes_per_row(), 2);
        assert_eq!(packed.weights.len(), 34);
        assert_eq!(packed.scales, vec![125, 125, 129, 129]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn sliced_matrix_starts_at_tensor_offset() {
        let path = std::env::temp_dir().join(format!("colic-mxfp4-offset-{}", std::process::id()));
        let prefix = bf16_bytes(&[64.0; 32]);
        let wanted = bf16_bytes(&[1.0; 32]);
        let mut source = prefix.clone();
        source.extend_from_slice(&wanted);
        fs::write(&path, &source).unwrap();
        let matrix = Matrix {
            source: TensorRef {
                source: path.clone(),
                offset: prefix.len() as u64,
                len: wanted.len() as u64,
                dtype: "BF16".into(),
                shape: vec![1, 32],
            },
            rows: 1,
            columns: 32,
            scale: None,
        };
        let packed = quantize_matrix(&matrix).unwrap();
        assert_eq!(packed.scales, vec![125]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_nonfinite_bf16_source() {
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        let error = quantize_bf16_row(&bf16_bytes(&[f32::INFINITY]), &mut weights, &mut scales)
            .unwrap_err();
        assert!(error.to_string().contains("non-finite"));
    }

    #[test]
    fn f16_is_not_misdecoded_as_bf16() {
        // 0x3c00 is F16 1.0 but BF16 0.0078125. Both source encodings must
        // quantize to the same MXFP4 bytes when they represent the same value.
        let f16 = 0x3c00_u16.to_le_bytes();
        let bf16 = ((1.0_f32.to_bits() >> 16) as u16).to_le_bytes();
        let (mut fw, mut fs) = (Vec::new(), Vec::new());
        let (mut bw, mut bs) = (Vec::new(), Vec::new());
        quantize_row(&f16, "F16", &mut fw, &mut fs).unwrap();
        quantize_row(&bf16, "BF16", &mut bw, &mut bs).unwrap();
        assert_eq!((fw, fs), (bw, bs));
    }

    #[test]
    fn unsupported_one_byte_float_formats_fail_closed() {
        for dtype in ["F8_E4M3FNUZ", "F8_E5M2", "F8_E5M2FNUZ", "F8_E8M0"] {
            let mut weights = Vec::new();
            let mut scales = Vec::new();
            let error = quantize_row(&[0x38], dtype, &mut weights, &mut scales).unwrap_err();
            assert!(error.to_string().contains(dtype));
            assert!(weights.is_empty());
            assert!(scales.is_empty());
        }
    }
}
