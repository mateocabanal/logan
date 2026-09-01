//! Engine-neutral math primitives, extracted verbatim from the qwen4
//! engine (C-identical numerics: f64 accumulation in rmsnorm, exact
//! reduction order, 1/sqrtf semantics). The token-identity gates are the
//! contract — any change here must keep every engine byte-identical.

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
        #[cfg(target_arch = "aarch64")]
        let neon = neon && o * i >= 1 << 18;
        #[cfg(not(target_arch = "aarch64"))]
        let neon = false;
        if neon {
            matmul_bf16_neon(y, x, bytes, o, i);
            return;
        }
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
}
