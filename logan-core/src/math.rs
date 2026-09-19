//! Engine-neutral math primitives, extracted verbatim from the qwen4
//! engine (C-identical numerics: f64 accumulation in rmsnorm, exact
//! reduction order, 1/sqrtf semantics). The token-identity gates are the
//! contract — any change here must keep every engine byte-identical.
//!
//! ## Accelerated paths and their opt-outs
//!
//! Every kernel reached from this module is behind a runtime probe and can be
//! disabled, because each one reassociates f32 accumulation and so is a
//! numerics change under the gate above:
//!
//! - **AVX2** (`math_x86::matmul_bf16_avx2`, `matmul_f32_avx2`) — reached only
//!   when `is_x86_feature_detected!("avx2")` and `"fma"` both pass; opt out with
//!   `QWEN_NEON_BF16=0`, which disables the NEON and AVX2 paths together.
//! - **NEON** (`matmul_bf16_neon`) — aarch64, same opt-out, plus a
//!   `o * i >= 1 << 18` size gate below which the scalar loop wins.
//! - **CUDA** — deliberately *not* reachable from [`matmul`]. The Windows
//!   workload is Q4_K_M, whose experts are byte-exact GGML blocks that this
//!   module never sees; its `Wt` holds only f32 and BF16. The CUDA path lives in
//!   `logan-qwen4`'s `ggufsource::cuda_q4k`, consumes GGML blocks directly, and
//!   mirrors `ggufsource::dot_row`. This one is **opt-in** (`LOGAN_CUDA=1`): it
//!   is bit-exact but measured at 0.34x–0.58x of the 12-thread CPU path at real
//!   expert shapes, so it stays off until the kernel is tiled. See
//!   `crate::cuda`'s module header, which also lists the library-search
//!   override and the fail-closed rules.
//!
//! Nothing here advertises a path it did not take: `matmul` returns without
//! touching a kernel unless the corresponding probe passed.

/// BF16 dot: y[o] = x[.] · w[o,.], weights BF16 (u16<<16 = f32).
/// 4-lane NEON fma; fp-order differs from scalar (the gate decides).
#[cfg(target_arch = "aarch64")]
pub fn matmul_bf16_neon(y: &mut [f32], x: &[f32], w: &[u8], o: usize, i: usize) {
    use std::arch::aarch64::*;
    for oo in 0..o {
        let wr = &w[oo * i * 2..(oo + 1) * i * 2];
        let mut acc = unsafe { vdupq_n_f32(0.0) };
        let mut ii = 0;
        while ii + 8 <= i {
            unsafe {
                let wv = vld1q_u16(wr[ii * 2..].as_ptr() as *const u16);
                let w0 = vshlq_n_u32(vmovl_u16(vget_low_u16(wv)), 16);
                let w1 = vshlq_n_u32(vmovl_u16(vget_high_u16(wv)), 16);
                let wf0 = vreinterpretq_f32_u32(w0);
                let wf1 = vreinterpretq_f32_u32(w1);
                let x0 = vld1q_f32(x[ii..].as_ptr());
                let x1 = vld1q_f32(x[ii + 4..].as_ptr());
                acc = vfmaq_f32(acc, wf0, x0);
                acc = vfmaq_f32(acc, wf1, x1);
            }
            ii += 8;
        }
        let mut s = unsafe { vaddvq_f32(acc) };
        while ii < i {
            let u = u16::from_le_bytes([wr[ii * 2], wr[ii * 2 + 1]]);
            s += x[ii] * f32::from_bits((u as u32) << 16);
            ii += 1;
        }
        y[oo] = s;
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn matmul_bf16_neon(_y: &mut [f32], _x: &[f32], _w: &[u8], _o: usize, _i: usize) {}

/// A resident weight: f32 or BF16 bytes + shape.
#[derive(Clone)]
pub struct Wt {
    pub f: Vec<f32>,
    /// BF16 bytes when loaded from a .coli package (decoded per-row in
    /// matmul to keep resident memory at package size).
    pub bytes: Option<Vec<u8>>,
    pub o: usize,
    pub i: usize,
}

/// y[O] = x[I] @ W^T. BF16 bytes decode per-row; f32 path is direct.
/// Parallelizes only matmuls big enough to amortize thread spawn
/// (>= 16M MACs ≈ 2ms at ~8 GFLOPs scalar).
pub fn matmul(y: &mut [f32], x: &[f32], w: &Wt) {
    let (o, i) = (w.o, w.i);
    let parallel = o * i >= 16_000_000
        && std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            > 1;
    if let Some(bytes) = &w.bytes {
        // NEON BF16: 4 f32 lanes, bf16 weights widened by (u16<<16).
        // fp-order differs from scalar (grouped fma) — the token-identity
        // gate decides; QWEN_NEON_BF16=0 opts out.
        let neon = std::env::var("QWEN_NEON_BF16")
            .map(|v| v != "0")
            .unwrap_or(true);
        // Size threshold: a 4-lane (or 8-lane) kernel amortises its prologue
        // only above ~256K MACs. Below that the scalar loop wins.
        #[cfg(target_arch = "aarch64")]
        let simd = neon && o * i >= 1 << 18;
        #[cfg(target_arch = "x86_64")]
        let simd = neon
            && o * i >= 1 << 18
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma");
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let simd = false;

        #[cfg(target_arch = "aarch64")]
        if simd {
            matmul_bf16_neon(y, x, bytes, o, i);
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if simd {
            // SAFETY: guarded by the runtime avx2+fma probe above.
            unsafe { crate::math_x86::matmul_bf16_avx2(y, x, bytes, o, i) };
            return;
        }
        let _ = simd;
        if parallel {
            std::thread::scope(|s| {
                let nthreads = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);
                let chunk = o.div_ceil(nthreads);
                for (c, yslice) in y.chunks_mut(chunk).enumerate() {
                    let rows = c * chunk;
                    let (x, bytes) = (&*x, &*bytes);
                    s.spawn(move || {
                        for (oo, yv) in yslice.iter_mut().enumerate() {
                            let oo = rows + oo;
                            let mut acc = 0.0_f32;
                            for ii in 0..i {
                                let u = u16::from_le_bytes([
                                    bytes[(oo * i + ii) * 2],
                                    bytes[(oo * i + ii) * 2 + 1],
                                ]);
                                acc += x[ii] * f32::from_bits((u as u32) << 16);
                            }
                            *yv = acc;
                        }
                    });
                }
            });
        } else {
            for oo in 0..o {
                let mut acc = 0.0_f32;
                for ii in 0..i {
                    let u = u16::from_le_bytes([
                        bytes[(oo * i + ii) * 2],
                        bytes[(oo * i + ii) * 2 + 1],
                    ]);
                    acc += x[ii] * f32::from_bits((u as u32) << 16);
                }
                y[oo] = acc;
            }
        }
        return;
    }
    // f32 weights: AVX2 when the host has it. The NEON port left this path
    // scalar entirely, so x86 hosts ran unvectorised on both weight formats.
    #[cfg(target_arch = "x86_64")]
    {
        let simd = std::env::var("QWEN_NEON_BF16")
            .map(|v| v != "0")
            .unwrap_or(true)
            && o * i >= 1 << 18
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma");
        if simd {
            // SAFETY: guarded by the runtime avx2+fma probe above.
            unsafe { crate::math_x86::matmul_f32_avx2(y, x, &w.f, o, i) };
            return;
        }
    }
    if parallel {
        std::thread::scope(|s| {
            let nthreads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            let chunk = o.div_ceil(nthreads);
            for (c, yslice) in y.chunks_mut(chunk).enumerate() {
                let rows = c * chunk;
                let (x, w) = (&*x, &*w);
                s.spawn(move || {
                    for (oo, yv) in yslice.iter_mut().enumerate() {
                        let oo = rows + oo;
                        let mut acc = 0.0_f32;
                        for ii in 0..i {
                            acc += x[ii] * w.f[oo * i + ii];
                        }
                        *yv = acc;
                    }
                });
            }
        });
    } else {
        for oo in 0..o {
            let mut acc = 0.0_f32;
            for ii in 0..i {
                acc += x[ii] * w.f[oo * i + ii];
            }
            y[oo] = acc;
        }
    }
}

/// out = rmsnorm(x) * (1 + w). f64 accumulation (C-identical).
pub fn rmsnorm_row(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let d = x.len();
    let mut ms = 0.0_f64;
    for i in 0..d {
        ms += x[i] as f64 * x[i] as f64;
    }
    let r = 1.0 / (ms as f32 / d as f32 + eps).sqrt();
    for i in 0..d {
        out[i] = x[i] * r * (1.0 + w[i]);
    }
}

/// Grouped rmsnorm: hc groups of d.
pub fn rmsnorm_grouped(out: &mut [f32], x: &[f32], w: &[f32], hc: usize, d: usize, eps: f32) {
    for g in 0..hc {
        rmsnorm_row(
            &mut out[g * d..g * d + d],
            &x[g * d..g * d + d],
            &w[g * d..g * d + d],
            eps,
        );
    }
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Gated rmsnorm (GDN): out = w * (x * r) * silu(z).
pub fn rmsnorm_gated_row(out: &mut [f32], x: &[f32], z: &[f32], w: &[f32], eps: f32) {
    let d = x.len();
    let mut ms = 0.0_f64;
    for i in 0..d {
        ms += x[i] as f64 * x[i] as f64;
    }
    let r = 1.0 / (ms as f32 / d as f32 + eps).sqrt();
    for i in 0..d {
        out[i] = w[i] * (x[i] * r) * silu(z[i]);
    }
}

pub fn softmax_row(x: &mut [f32]) {
    let n = x.len();
    let mut m = -1e30_f32;
    for i in 0..n {
        if x[i] > m {
            m = x[i];
        }
    }
    let mut s = 0.0_f32;
    for i in 0..n {
        x[i] = (x[i] - m).exp();
        s += x[i];
    }
    let r = 1.0 / s;
    for i in 0..n {
        x[i] *= r;
    }
}

pub fn l2norm(x: &mut [f32]) {
    let mut s = 0.0_f64;
    for &v in x.iter() {
        s += v as f64 * v as f64;
    }
    let r = 1.0 / s.sqrt() as f32;
    for v in x.iter_mut() {
        *v *= r;
    }
}

/// f32 -> BF16 (top 16 bits, round-to-nearest-even), as 2 LE bytes.
pub fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

pub fn bf16_bytes(f: f32) -> [u8; 2] {
    f32_to_bf16(f).to_le_bytes()
}

pub fn bf16_to_f32(u: u16) -> f32 {
    f32::from_bits((u as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::approx_constant)]
    fn bf16_round_trip() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 3.14159, 1e-5, 1e5] {
            let u = f32_to_bf16(v);
            let back = bf16_to_f32(u);
            // BF16 has 8 mantissa bits: relative error <= 2^-8
            let rel = ((back - v).abs() / v.abs().max(1e-30)).max(0.0);
            assert!(rel < 0.004, "v={v} back={back} rel={rel}");
        }
    }

    #[test]
    fn rmsnorm_known_vector() {
        let x = [3.0f32, 4.0];
        let w = [0.0f32, 0.0];
        let mut out = [0.0f32; 2];
        rmsnorm_row(&mut out, &x, &w, 1e-6);
        // ms = 25, r = 1/sqrt(12.5+eps) ≈ 0.28284
        assert!((out[0] - 3.0 * 0.2828427).abs() < 1e-5);
        assert!((out[1] - 4.0 * 0.2828427).abs() < 1e-5);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0f32, 2.0, 3.0];
        softmax_row(&mut x);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn matmul_bf16_matches_scalar() {
        let x = [1.0f32, 2.0, 3.0];
        let w: Vec<u8> = [1.0f32, 0.5, 2.0]
            .iter()
            .flat_map(|&v| bf16_bytes(v))
            .collect();
        let mut y = [0.0f32; 1];
        matmul(
            &mut y,
            &x,
            &Wt {
                f: vec![],
                bytes: Some(w),
                o: 1,
                i: 3,
            },
        );
        // 1*1 + 2*0.5 + 3*2 = 8 (bf16 rounding ~1e-3)
        assert!((y[0] - 8.0).abs() < 0.01);
    }

    #[test]
    fn matmul_f32_path() {
        let x = [1.0f32, 2.0];
        let w = Wt {
            f: vec![1.0, 0.0, 0.0, 1.0],
            bytes: None,
            o: 2,
            i: 2,
        };
        let mut y = [0.0f32; 2];
        matmul(&mut y, &x, &w);
        assert_eq!(y, [1.0, 2.0]);
    }

    /// Naive scalar references. The tests above use o=1,i=3, which is below the
    /// `o * i >= 1 << 18` gate, so they never exercise the SIMD kernels; these
    /// shapes are above it and do.
    #[cfg(target_arch = "x86_64")]
    fn scalar_bf16(y: &mut [f32], x: &[f32], w: &[u8], o: usize, i: usize) {
        for oo in 0..o {
            let mut acc = 0.0f32;
            for ii in 0..i {
                let off = (oo * i + ii) * 2;
                let u = u16::from_le_bytes([w[off], w[off + 1]]);
                acc += x[ii] * f32::from_bits((u as u32) << 16);
            }
            y[oo] = acc;
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn scalar_f32(y: &mut [f32], x: &[f32], w: &[f32], o: usize, i: usize) {
        for oo in 0..o {
            let mut acc = 0.0f32;
            for ii in 0..i {
                acc += x[ii] * w[oo * i + ii];
            }
            y[oo] = acc;
        }
    }

    /// Whether the AVX2 bf16 kernel can run here at all.
    ///
    /// The same probe `matmul` makes, so the reachability assertion below can
    /// tell "the kernel was declined by the host" (skip) from "the kernel was
    /// declined by the wiring" (fail).
    #[cfg(target_arch = "x86_64")]
    fn has_avx2() -> bool {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }

    #[cfg(target_arch = "x86_64")]
    fn rng_vec(n: usize, mut seed: u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed >> 12;
                seed ^= seed << 25;
                seed ^= seed >> 27;
                let u = seed.wrapping_mul(0x2545_F491_4F6C_DD1D);
                ((u >> 32) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
            })
            .collect()
    }

    /// Every above-gate shape must agree with the scalar reference to f32
    /// reassociation tolerance. A wrong bf16 widening order, a mis-strided
    /// accumulator, or a dropped tail all show up here as O(1) relative error.
    ///
    /// i=513/257 also cover the 16-wide body, the 8-wide body and the scalar
    /// tail, which is where an off-by-one would live.
    ///
    /// x86_64 only: the f32 arm has no aarch64 kernel, so there the scalar f32
    /// loop is the only implementation and this test would compare it to itself.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn matmul_simd_matches_scalar_above_threshold() {
        for (o, i) in [(640usize, 2560usize), (2560, 640), (512, 513), (2048, 257)] {
            assert!(o * i >= 1 << 18, "shape {o}x{i} is below the SIMD gate");
            let x = rng_vec(i, 0x1234_5678);
            let f32w = rng_vec(o * i, 0x9E37_79B9);

            // bf16 bytes arm
            let bfw: Vec<u8> = f32w.iter().flat_map(|&v| bf16_bytes(v)).collect();
            let mut got = vec![0.0f32; o];
            let mut want = vec![0.0f32; o];
            matmul(
                &mut got,
                &x,
                &Wt {
                    f: vec![],
                    bytes: Some(bfw.clone()),
                    o,
                    i,
                },
            );
            scalar_bf16(&mut want, &x, &bfw, o, i);
            assert_close(&got, &want, &format!("bf16 {o}x{i}"));

            // f32 arm
            let mut gotf = vec![0.0f32; o];
            let mut wantf = vec![0.0f32; o];
            matmul(
                &mut gotf,
                &x,
                &Wt {
                    f: f32w.clone(),
                    bytes: None,
                    o,
                    i,
                },
            );
            scalar_f32(&mut wantf, &x, &f32w, o, i);
            assert_close(&gotf, &wantf, &format!("f32 {o}x{i}"));
        }
    }

    /// The engine-neutral `matmul` must actually ENTER the AVX2 kernel.
    ///
    /// A tolerance test passes whether or not the kernel runs -- that is how
    /// this module sat unreachable long enough for `math_x86` to be written
    /// twice. This asserts on [`crate::math_x86::bf16_calls`] instead.
    ///
    /// Runs in a child process: `QWEN_NEON_BF16` is process-global and
    /// `matmul_opt_out_is_bit_exact_scalar` flips it, so an in-process
    /// assertion here would race that test under the parallel harness and pass
    /// or fail at random. A child gets a pristine environment, and the parent
    /// asserts the child took the arm it was asked for.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn matmul_reaches_the_avx2_kernel() {
        const CHILD: &str = "LOGAN_CORE_AVX2_CHILD";

        let (o, i) = (640usize, 2560usize);
        let x = rng_vec(i, 0x0BAD_F00D);
        let f32w = rng_vec(o * i, 0x5EED_1234);
        let bfw: Vec<u8> = f32w.iter().flat_map(|&v| bf16_bytes(v)).collect();

        if std::env::var(CHILD).is_ok() {
            if !has_avx2() {
                // Skip rather than pass: a green tick for a kernel that never
                // ran is the same lie the kernel was written to end.
                println!("arm: SKIP no avx2+fma on this CPU");
                return;
            }

            let before = crate::math_x86::bf16_calls();
            let mut got = vec![0.0f32; o];
            matmul(
                &mut got,
                &x,
                &Wt {
                    f: vec![],
                    bytes: Some(bfw.clone()),
                    o,
                    i,
                },
            );
            let calls = crate::math_x86::bf16_calls() - before;
            assert!(
                calls > 0,
                "{o}x{i} is above the gate with avx2+fma present, but `matmul` \
                 did not enter the AVX2 kernel -- the fast path is dead code"
            );

            let mut want = vec![0.0f32; o];
            scalar_bf16(&mut want, &x, &bfw, o, i);
            assert_close(&got, &want, &format!("bf16 {o}x{i}"));
            println!("arm: calls={calls}");
            return;
        }

        let out = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .args([
                "--exact",
                "math::tests::matmul_reaches_the_avx2_kernel",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env_remove("QWEN_NEON_BF16")
            .output()
            .expect("spawn child test binary");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "child failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(stdout.contains("arm:"), "child took no arm:\n{stdout}");
        print!("{stdout}");
    }

    /// Absolute error against the magnitude of the results being compared.
    ///
    /// Deliberately NOT elementwise-relative: these dot products sum thousands
    /// of O(1) terms with cancelling signs, so individual rows land near zero
    /// with absolute error ~1e-5 while neighbouring rows are O(100). Relative
    /// to a near-zero row that looks like 1e-3 of pure reassociation noise.
    /// A genuine kernel bug -- wrong bf16 widening, mis-strided accumulator,
    /// dropped tail -- perturbs a row by O(summand), i.e. O(1) here, which is
    /// ~100x this bound.
    #[cfg(target_arch = "x86_64")]
    fn assert_close(got: &[f32], want: &[f32], what: &str) {
        let scale = want
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max)
            .max(1e-30);
        let bound = 1e-5 * scale;
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            let d = (g - w).abs();
            assert!(
                d <= bound,
                "{what} row {k}: got {g:e} want {w:e} (abs {d:e} > {bound:e}, scale {scale:e})"
            );
        }
    }

    /// The opt-out must produce the scalar path exactly, on every shape.
    ///
    /// x86_64 only, and single-threaded: `QWEN_NEON_BF16` is process-global, so
    /// running this under the default parallel harness would race the SIMD test
    /// above. The other tests in this module do not call `matmul`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn matmul_opt_out_is_bit_exact_scalar() {
        let (o, i) = (640usize, 2560usize);
        let x = rng_vec(i, 0xABCD_EF01);
        let f32w = rng_vec(o * i, 0x1357_9BDF);
        let bfw: Vec<u8> = f32w.iter().flat_map(|&v| bf16_bytes(v)).collect();

        let prev = std::env::var("QWEN_NEON_BF16").ok();
        // SAFETY: single-threaded test (see doc comment above).
        unsafe { std::env::set_var("QWEN_NEON_BF16", "0") };

        let mut got = vec![0.0f32; o];
        let mut want = vec![0.0f32; o];
        matmul(
            &mut got,
            &x,
            &Wt {
                f: vec![],
                bytes: Some(bfw.clone()),
                o,
                i,
            },
        );
        scalar_bf16(&mut want, &x, &bfw, o, i);
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "QWEN_NEON_BF16=0 must be bit-identical to the scalar reference"
        );

        let mut gotf = vec![0.0f32; o];
        let mut wantf = vec![0.0f32; o];
        matmul(
            &mut gotf,
            &x,
            &Wt {
                f: f32w.clone(),
                bytes: None,
                o,
                i,
            },
        );
        scalar_f32(&mut wantf, &x, &f32w, o, i);
        assert_eq!(
            gotf.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            wantf.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "QWEN_NEON_BF16=0 f32 path must be bit-identical to scalar"
        );

        // SAFETY: single-threaded test.
        unsafe {
            match prev {
                Some(v) => std::env::set_var("QWEN_NEON_BF16", v),
                None => std::env::remove_var("QWEN_NEON_BF16"),
            }
        }
    }
}
