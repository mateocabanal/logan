//! Qwen3.8 MTP-E speculative draft head — package/runtime state + numerical helpers.
//!
//! Exact algorithm from llama.cpp qwen4exp `graph_mtp` (PR #27836) and the
//! checkpoint's `mtp.*` weights. The draft is:
//!
//!   h = trunk WIDE hyper-connection residual [hc*n_embd] (last layer, pre-head-collapse)
//!   e = embed(prev_token) [n_embd]
//!   e_norm = rmsnorm(e, enorm)                      // mtp.pre_fc_norm_embedding
//!   h_norm = rmsnorm_grouped(h, hnorm, each hc stream) // mtp.pre_fc_norm_hidden [hc*d]
//!   res_hc[g] = eh_proj @ concat([e_norm ; h_norm[g]]) per stream g
//!            == fc_embedding@e_norm + fc_hidden@h_norm[g]   (e broadcast to all streams)
//!   -> ONE dense full-attention + MoE trunk block over res_hc   // CALLER (lib.rs layer-48)
//!   -> head_mix: hc_mix(res_hc, hc_head_*)  collapses streams, doubles as output norm
//!   logits = trunk lm_head(cur)
//!
//! This module holds the two MTP-specific pieces that are NOT a normal trunk
//! layer: the `combiner` (eh_proj with per-stream broadcast) and the `head_mix`
//! (hc_head collapse). The full-attention/MoE block between them is a standard
//! trunk layer (`forward_layer(48)` in Logan) and is NOT reproduced here.
//! Self-contained + unit-tested so numerics are pinned before the lib.rs wiring.

use super::Wt;

/// Row-major [o][i] f32 weights used only by the scalar helper tests below.
type W = Vec<f32>;

/// Loaded embedded MTP state. The actual transformer block lives at
/// `layer_index` in `Model::layers`; keeping it there lets MTP reuse the
/// production attention, HC, shared-expert, routed-expert and MetalIO paths
/// without adding the draft layer to the target model's `cfg.layers`.
pub(crate) struct MtpRuntime {
    pub layer_index: usize,
    pub experts: usize,
    pub topk: usize,
    pub fc_embedding: Wt,
    pub fc_hidden: Wt,
    pub enorm: Vec<f32>,
    pub hnorm: Vec<f32>,
    pub head_norm: Vec<f32>,
    pub head_down: Wt,
    pub head_up: Wt,
    pub catchup_rows: u64,
    pub drafted: u64,
    pub accepted: u64,
    pub blocks: u64,
    pub attempted_by_pos: [u64; 4],
    pub accepted_by_pos: [u64; 4],
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub draft_mio_bytes: u64,
    pub verify_mio_bytes: u64,
}

#[derive(Debug)]
pub(crate) struct MtpDraft {
    pub logits: Vec<f32>,
    /// Pre-final-mixer HC residual emitted by the MTP transformer block. This
    /// is the hidden state consumed by the next recursive MTP step.
    pub next_hidden_hc: Vec<f32>,
}

#[derive(Clone, Default)]
pub(crate) struct MtpVerifyBoundary {
    pub gdn_s: Vec<Vec<f32>>,
    pub gdn_conv: Vec<Vec<f32>>,
    pub ple_ring: Vec<i64>,
    pub ple_conv_state: Vec<f32>,
    pub hidden_hc: Vec<f32>,
}

pub(crate) struct MtpVerifyBatch {
    pub logits: Vec<Vec<f32>>,
    pub boundaries: Vec<MtpVerifyBoundary>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MtpStats {
    pub catchup_rows: u64,
    pub drafted: u64,
    pub accepted: u64,
    pub blocks: u64,
    pub attempted_by_pos: [u64; 4],
    pub accepted_by_pos: [u64; 4],
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub draft_mio_bytes: u64,
    pub verify_mio_bytes: u64,
}

/// The MTP combiner + head-mixer weights (block weights live in `Layer(48)`).
#[derive(Clone)]
pub struct MtpHead {
    pub fc_embedding: W,   // [d][d]
    pub fc_hidden: W,      // [d][d]
    pub enorm: Vec<f32>,   // [d]   (mtp.pre_fc_norm_embedding)
    pub hnorm: Vec<f32>,   // [hc*d] (mtp.pre_fc_norm_hidden)
    pub hc_norm: Vec<f32>, // [hc*d] (mtp.hyper_connection_mixer.hc_norm)
    pub hc_down: W,        // [lr][hc*d]
    pub hc_up: W,          // [hc*d][lr]
}

fn rmsnorm_row(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let mut sq = 0.0f32;
    for &v in x {
        sq += v * v;
    }
    let inv = 1.0 / (sq / x.len() as f32 + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * (1.0 + w[i]);
    }
}

/// Grouped RMSNorm over `hc` contiguous `d`-width streams, each scaled by its
/// own `d`-wide gamma slice (the engine's `rmsnorm_grouped`).
fn rmsnorm_grouped(out: &mut [f32], x: &[f32], w: &[f32], hc: usize, d: usize, eps: f32) {
    for g in 0..hc {
        rmsnorm_row(
            &mut out[g * d..(g + 1) * d],
            &x[g * d..(g + 1) * d],
            &w[g * d..(g + 1) * d],
            eps,
        );
    }
}

fn matmul(y: &mut [f32], x: &[f32], w: &W, o: usize) {
    for oo in 0..o {
        let mut acc = 0.0f32;
        for ii in 0..x.len() {
            acc += x[ii] * w[oo * x.len() + ii];
        }
        y[oo] = acc;
    }
}

impl MtpHead {
    /// Combiner: turn (prev-token embedding, wide residual) into the wide
    /// `res_hc` that feeds the dense block. Returns `hc*d` (per-stream
    /// `fc_embedding@e_norm + fc_hidden@h_norm[g]`).
    pub fn combiner(&self, prev_emb: &[f32], residual: &[f32], hc: usize, eps: f32) -> Vec<f32> {
        let d = prev_emb.len();
        let hcd = hc * d;
        debug_assert_eq!(residual.len(), hcd);
        let mut e_norm = vec![0.0; d];
        let mut h_norm = vec![0.0; hcd];
        rmsnorm_row(&mut e_norm, prev_emb, &self.enorm, eps);
        rmsnorm_grouped(&mut h_norm, residual, &self.hnorm, hc, d, eps);
        let mut fe = vec![0.0; d]; // fc_embedding @ e_norm (same for all streams)
        matmul(&mut fe, &e_norm, &self.fc_embedding, d);
        let mut out = vec![0.0; hcd];
        for g in 0..hc {
            let mut fh = vec![0.0; d];
            matmul(&mut fh, &h_norm[g * d..(g + 1) * d], &self.fc_hidden, d);
            for dd in 0..d {
                out[g * d + dd] = fe[dd] + fh[dd];
            }
        }
        out
    }

    /// Head collapse: after the dense block, `hc_mix(res_hc, hc_head_*)` with no
    /// inject collapses the streams and doubles as the output norm. Returns `d`.
    pub fn head_mix(&self, res_hc: &[f32], hc: usize, lr: usize, eps: f32) -> Vec<f32> {
        let d = self.enorm.len();
        let hcd = hc * d;
        debug_assert_eq!(res_hc.len(), hcd);
        let mut normed = vec![0.0; hcd];
        rmsnorm_grouped(&mut normed, res_hc, &self.hc_norm, hc, d, eps);
        let mut lo = vec![0.0; lr];
        matmul(&mut lo, &normed, &self.hc_down, lr);
        for v in lo.iter_mut() {
            let s = *v / hc as f32;
            *v = s / (1.0 + (-s).exp());
        }
        let mut hi = vec![0.0; hcd];
        matmul(&mut hi, &lo, &self.hc_up, hcd);
        let mut out = vec![0.0; d];
        for i in 0..hcd {
            let sig = 1.0 / (1.0 + (-hi[i]).exp());
            out[i % d] += (sig * normed[i]) * (1.0 / hc as f32);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(init: fn(usize, usize) -> f32, o: usize, i: usize) -> W {
        (0..o * i).map(|k| init(k / i, k % i)).collect()
    }

    #[test]
    fn combiner_split_equals_fused_concat() {
        // fc_embedding@e + fc_hidden@h (per stream) must equal the fused
        // [fc_embedding | fc_hidden] @ concat([e;h]) — the exact identity the
        // checkpoint relies on (A·e + B·h == [A|B]·concat(e,h)).
        let d = 3usize;
        let hc = 2usize;
        let eps = 1e-5f32;
        let m = MtpHead {
            fc_embedding: mk(|o, i| (o as f32 + 1.0) * 0.1 + i as f32 * 0.2, d, d),
            fc_hidden: mk(|o, i| (o as f32 + 1.0) * 0.3 - i as f32 * 0.1, d, d),
            enorm: vec![1.0, 2.0, 0.5],
            hnorm: vec![0.8, 1.1, 1.6, 1.2, 0.9, 1.4], // hc*d
            hc_norm: vec![1.0; hc * d],
            hc_down: mk(|o, i| (o as f32 + i as f32) * 0.01, 2, hc * d),
            hc_up: mk(|o, i| (o as f32 + i as f32) * 0.02, hc * d, 2),
        };
        let emb = [1.0f32, 0.5, -1.0];
        let res = [1.0f32, -0.5, 0.25, 0.75, -1.0, 2.0]; // hc*d
        let out = m.combiner(&emb, &res, hc, eps);
        assert_eq!(out.len(), hc * d);

        // independent scalar reference
        let norm = |x: &[f32], w: &[f32]| -> Vec<f32> {
            let sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let inv = 1.0 / (sq + eps).sqrt();
            x.iter().zip(w).map(|(x, w)| x * inv * (1.0 + w)).collect()
        };
        let e = norm(&emb, &m.enorm);
        let h0 = norm(&res[0..3], &m.hnorm[0..3]);
        let h1 = norm(&res[3..6], &m.hnorm[3..6]);
        let dot = |w: &W, v: &[f32]| -> Vec<f32> {
            (0..d)
                .map(|i| {
                    w[i * d..(i + 1) * d]
                        .iter()
                        .zip(v)
                        .map(|(wi, v)| wi * v)
                        .sum::<f32>()
                })
                .collect()
        };
        for g in 0..hc {
            let hs = if g == 0 { &h0 } else { &h1 };
            for dd in 0..d {
                let want = dot(&m.fc_embedding, &e)[dd] + dot(&m.fc_hidden, hs)[dd];
                assert!(
                    (out[g * d + dd] - want).abs() < 1e-4,
                    "combiner[g={g}][{dd}]"
                );
            }
        }
        // determinism
        assert_eq!(m.combiner(&emb, &res, hc, eps), out);
    }

    #[test]
    fn head_mix_deterministic() {
        let hc = 1usize;
        let lr = 1usize;
        let m = MtpHead {
            fc_embedding: vec![1.0, 0.0, 0.0, 1.0],
            fc_hidden: vec![0.0, 1.0, 1.0, 0.0],
            enorm: vec![1.0, 1.0],
            hnorm: vec![1.0, 1.0],
            hc_norm: vec![1.0, 1.0],
            hc_down: vec![1.0, 1.0],
            hc_up: vec![1.0, 1.0],
        };
        let x = [0.1f32, 0.2];
        assert_eq!(m.head_mix(&x, hc, lr, 1e-5), m.head_mix(&x, hc, lr, 1e-5));
    }
}
