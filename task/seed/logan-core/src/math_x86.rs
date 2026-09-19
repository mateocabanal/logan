//! x86_64 AVX2 kernels for the engine-neutral math primitives.
//!
//! ## Why this file exists
//!
//! `MachineProfile` has recorded `avx2` since before this module did: the
//! compiler *plans* for it (`target/mod.rs` branches on it, `resources.rs`
//! serialises it), but `math.rs` had no AVX2 kernel, so an x86 host quietly ran
//! the scalar path while its own target profile claimed otherwise. Planning for
//! a capability that never executes is the same lie as advertising Metal
//! without a Metal kernel.
//!
//! AVX2 is not guaranteed on x86_64 (the GPD's Atom x7 has SSE4.2 only), so
//! every entry point here is `#[target_feature(enable = "avx2")]` and reached
//! only through a runtime `is_x86_feature_detected!` probe at each caller:
//! `math::matmul` and `logan-qwen4`'s bf16 GEMV. Each caller asserts on
//! [`bf16_calls`], because a kernel that is verified and never reached is the
//! same dead code this module was written to replace.
//!
//! ## Numerics
//!
//! The NEON path already documents that its fp-order differs from scalar and
//! that "the token-identity gate decides". These kernels carry the same
//! caveat: the accumulator order differs, results match to f32 reassociation
//! tolerance, not bit-equality. Callers that need byte-identity must stay on
//! the scalar path, which is why the env opt-out (`QWEN_NEON_BF16`, honoured in
//! `math::matmul`) disables both.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use std::sync::atomic::{AtomicU64, Ordering};

/// BF16 AVX2 kernels entered, as evidence the path executed.
///
/// The same guard [`crate::cuda::launches`] provides for the GPU path, for the
/// same reason: a kernel can be verified correct and measured faster and still
/// reach no production path, and only a counter distinguishes "wired" from
/// "advertised". `math::matmul`'s and `logan-qwen4`'s wiring tests assert on it.
///
/// One relaxed increment per GEMV *entry* -- not per row -- against `o * i`
/// multiply-adds, which is below the noise floor even at the smallest shape the
/// SIMD gate admits (256K MACs).
#[cfg(target_arch = "x86_64")]
static BF16_CALLS: AtomicU64 = AtomicU64::new(0);

/// How many times [`matmul_bf16_avx2`] has been entered in this process.
///
/// Read this instead of trusting a capability probe, exactly as
/// `cuda_q4k::kernel_launches` is read instead of trusting `available`.
#[cfg(target_arch = "x86_64")]
pub fn bf16_calls() -> u64 {
    BF16_CALLS.load(Ordering::Relaxed)
}

/// BF16 dot products, eight f32 per iteration.
///
/// `y[o] = x[.] · w[o,.]` where `w` is BF16 bytes: each u16 widens to f32 by
/// shifting into the high half, exactly as the scalar and NEON paths do.
///
/// # Safety
///
/// CPU must support `avx2` and `fma`. `w` must hold `o * i * 2` bytes,
/// `x.len() >= i`, `y.len() >= o`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_bf16_avx2(y: &mut [f32], x: &[f32], w: &[u8], o: usize, i: usize) {
    BF16_CALLS.fetch_add(1, Ordering::Relaxed);
    for oo in 0..o {
        let wr = &w[oo * i * 2..(oo + 1) * i * 2];
        let mut acc0 = unsafe { _mm256_setzero_ps() };
        let mut acc1 = unsafe { _mm256_setzero_ps() };
        let mut ii = 0usize;
        // Two 8-wide chains: the fma latency is ~4 cycles, one chain stalls.
        while ii + 16 <= i {
            unsafe {
                let w0 = widen_bf16x8(wr.as_ptr().add(ii * 2));
                let w1 = widen_bf16x8(wr.as_ptr().add(ii * 2 + 16));
                acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x.as_ptr().add(ii)), acc0);
                acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x.as_ptr().add(ii + 8)), acc1);
            }
            ii += 16;
        }
        while ii + 8 <= i {
            unsafe {
                let w0 = widen_bf16x8(wr.as_ptr().add(ii * 2));
                acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x.as_ptr().add(ii)), acc0);
            }
            ii += 8;
        }
        let mut s = unsafe { hsum256_ps(_mm256_add_ps(acc0, acc1)) };
        // Scalar tail, same widening as everywhere else.
        while ii < i {
            let u = u16::from_le_bytes([wr[ii * 2], wr[ii * 2 + 1]]);
            s += x[ii] * f32::from_bits((u as u32) << 16);
            ii += 1;
        }
        y[oo] = s;
    }
}

/// Load 8 BF16 values and widen to f32 (u16 << 16 reinterpreted).
///
/// # Safety
///
/// CPU must support `avx2`. `p` must point at 16 readable bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn widen_bf16x8(p: *const u8) -> __m256 {
    unsafe {
        // 8 u16 = 128 bits.
        let raw = _mm_loadu_si128(p as *const __m128i);
        // Zero-extend u16 -> u32, then shift left 16 to place the value in the
        // high half of each f32 lane. This is the exact scalar transform,
        // vectorised -- not an approximation of it.
        let wide = _mm256_cvtepu16_epi32(raw);
        let shifted = _mm256_slli_epi32(wide, 16);
        _mm256_castsi256_ps(shifted)
    }
}

/// Horizontal sum of eight f32 lanes.
///
/// # Safety
///
/// CPU must support `avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    unsafe {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(s);
        let sums = _mm_add_ps(s, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
    }
}

/// f32 dot products, eight lanes per iteration.
///
/// # Safety
///
/// CPU must support `avx2` and `fma`. `w` must hold `o * i` f32,
/// `x.len() >= i`, `y.len() >= o`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_f32_avx2(y: &mut [f32], x: &[f32], w: &[f32], o: usize, i: usize) {
    for oo in 0..o {
        let wr = &w[oo * i..(oo + 1) * i];
        let mut acc0 = unsafe { _mm256_setzero_ps() };
        let mut acc1 = unsafe { _mm256_setzero_ps() };
        let mut ii = 0usize;
        while ii + 16 <= i {
            unsafe {
                acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(wr.as_ptr().add(ii)),
                    _mm256_loadu_ps(x.as_ptr().add(ii)),
                    acc0,
                );
                acc1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(wr.as_ptr().add(ii + 8)),
                    _mm256_loadu_ps(x.as_ptr().add(ii + 8)),
                    acc1,
                );
            }
            ii += 16;
        }
        while ii + 8 <= i {
            unsafe {
                acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(wr.as_ptr().add(ii)),
                    _mm256_loadu_ps(x.as_ptr().add(ii)),
                    acc0,
                );
            }
            ii += 8;
        }
        let mut s = unsafe { hsum256_ps(_mm256_add_ps(acc0, acc1)) };
        while ii < i {
            s += x[ii] * wr[ii];
            ii += 1;
        }
        y[oo] = s;
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    fn scalar_bf16(y: &mut [f32], x: &[f32], w: &[u8], o: usize, i: usize) {
        for oo in 0..o {
            let mut acc = 0.0f32;
            for ii in 0..i {
                let u = u16::from_le_bytes([w[(oo * i + ii) * 2], w[(oo * i + ii) * 2 + 1]]);
                acc += x[ii] * f32::from_bits((u as u32) << 16);
            }
            y[oo] = acc;
        }
    }

    fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| crate::math::bf16_bytes(*v)).collect()
    }

    /// Skip rather than pass when the CPU lacks AVX2: a pass would be a lie.
    fn has_avx2() -> bool {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }

    #[test]
    fn bf16_avx2_matches_scalar_when_supported() {
        if !has_avx2() {
            eprintln!("skipping: no avx2+fma on this CPU");
            return;
        }
        // Ragged i exercises both the 16- and 8-wide bodies and the tail.
        for i in [1usize, 7, 8, 9, 15, 16, 17, 33, 512] {
            let o = 3;
            let wv: Vec<f32> = (0..o * i).map(|k| ((k % 37) as f32 - 18.0) / 8.0).collect();
            let x: Vec<f32> = (0..i).map(|k| ((k % 11) as f32 - 5.0) / 4.0).collect();
            let wb = bf16_bytes(&wv);
            let mut got = vec![0.0f32; o];
            let mut want = vec![0.0f32; o];
            unsafe { matmul_bf16_avx2(&mut got, &x, &wb, o, i) };
            scalar_bf16(&mut want, &x, &wb, o, i);
            let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-30);
            for r in 0..o {
                assert!(
                    (got[r] - want[r]).abs() / scale < 1e-5,
                    "i={i} row={r}: got {} want {}",
                    got[r],
                    want[r]
                );
            }
        }
    }

    #[test]
    fn f32_avx2_matches_scalar_when_supported() {
        if !has_avx2() {
            eprintln!("skipping: no avx2+fma on this CPU");
            return;
        }
        for i in [1usize, 7, 8, 16, 17, 100] {
            let o = 3;
            let w: Vec<f32> = (0..o * i).map(|k| ((k % 23) as f32 - 11.0) / 3.0).collect();
            let x: Vec<f32> = (0..i).map(|k| ((k % 13) as f32 - 6.0) / 2.0).collect();
            let mut got = vec![0.0f32; o];
            unsafe { matmul_f32_avx2(&mut got, &x, &w, o, i) };
            for r in 0..o {
                let want: f32 = (0..i).map(|c| x[c] * w[r * i + c]).sum();
                let scale = want.abs().max(1e-30);
                assert!((got[r] - want).abs() / scale < 1e-5, "i={i} row={r}");
            }
        }
    }
}
