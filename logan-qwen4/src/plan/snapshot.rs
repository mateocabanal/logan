//! Exact Qwen4 causal-state checkpoint/restore primitive.
//!
//! This is intentionally NOT a cache policy. It captures one completed-token
//! prefix state compactly so callers can A/B the value of prefix reuse before
//! Logan grows an LRU/radix/disk format around it.

use crate::Model;

#[derive(Clone)]
pub struct QwenStateSnapshot {
    prefix_len: usize,
    // Empty entries correspond to non-GDN layers.
    gdn_s: Vec<Vec<f32>>,
    gdn_conv: Vec<Vec<f32>>,
    // Attention KV is compacted from [head, max_t, dim] to
    // [head, prefix_len, dim]. Empty entries correspond to GDN layers.
    kv_k: Vec<Vec<f32>>,
    kv_v: Vec<Vec<f32>>,
    // QSA index cache is already [position, index_key_dim], so its active
    // prefix is contiguous. Empty entries correspond to non-QSA layers.
    idx_cache: Vec<Vec<f32>>,
    ple_ring: Vec<i64>,
    ple_conv_state: Vec<f32>,
}

impl QwenStateSnapshot {
    pub fn prefix_len(&self) -> usize {
        self.prefix_len
    }

    /// Payload bytes copied by this checkpoint (Vec metadata excluded).
    pub fn payload_bytes(&self) -> usize {
        let f32_bytes = |v: &Vec<Vec<f32>>| v.iter().map(|x| x.len() * 4).sum::<usize>();
        f32_bytes(&self.gdn_s)
            + f32_bytes(&self.gdn_conv)
            + f32_bytes(&self.kv_k)
            + f32_bytes(&self.kv_v)
            + f32_bytes(&self.idx_cache)
            + self.ple_ring.len() * std::mem::size_of::<i64>()
            + self.ple_conv_state.len() * 4
    }

    /// Bit-exact state equality (distinguishes +0/-0 and NaN payloads).
    pub fn exact_eq(&self, other: &Self) -> bool {
        self.prefix_len == other.prefix_len
            && nested_f32_exact(&self.gdn_s, &other.gdn_s)
            && nested_f32_exact(&self.gdn_conv, &other.gdn_conv)
            && nested_f32_exact(&self.kv_k, &other.kv_k)
            && nested_f32_exact(&self.kv_v, &other.kv_v)
            && nested_f32_exact(&self.idx_cache, &other.idx_cache)
            && self.ple_ring == other.ple_ring
            && f32_exact(&self.ple_conv_state, &other.ple_conv_state)
    }
}

fn f32_exact(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn nested_f32_exact(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| f32_exact(x, y))
}

fn pack_head_major(
    src: &[f32],
    heads: usize,
    max_t: usize,
    width: usize,
    prefix_len: usize,
) -> Vec<f32> {
    let active = prefix_len * width;
    let mut out = vec![0.0; heads * active];
    for h in 0..heads {
        let src_off = h * max_t * width;
        let dst_off = h * active;
        out[dst_off..dst_off + active].copy_from_slice(&src[src_off..src_off + active]);
    }
    out
}

fn unpack_head_major(
    compact: &[f32],
    dst: &mut [f32],
    heads: usize,
    max_t: usize,
    width: usize,
    prefix_len: usize,
) {
    let active = prefix_len * width;
    for h in 0..heads {
        let src_off = h * active;
        let dst_off = h * max_t * width;
        dst[dst_off..dst_off + active].copy_from_slice(&compact[src_off..src_off + active]);
    }
}

impl Model {
    /// Capture the exact causal state after `prefix_len` completed token
    /// positions. Only live GDN layers and active KV/QSA prefix ranges are
    /// copied. Expert residency and telemetry are deliberately excluded:
    /// they change performance, not model semantics.
    pub fn snapshot_state(&self, prefix_len: usize) -> Result<QwenStateSnapshot, String> {
        if prefix_len > self.cfg.max_t {
            return Err(format!(
                "snapshot prefix {prefix_len} exceeds context {}",
                self.cfg.max_t
            ));
        }
        if self.sched_pause.is_some() || self.sched_blocked.is_some() {
            return Err("cannot snapshot a scheduler-paused token".into());
        }

        let state_len = self.cfg.lin_v_heads * self.cfg.lin_k_dim * self.cfg.lin_v_dim;
        let cdim = self.cfg.lin_k_dim * self.cfg.lin_k_heads * 2
            + self.cfg.lin_v_dim * self.cfg.lin_v_heads;
        let conv_len = cdim * self.cfg.conv_kernel.saturating_sub(1);

        let mut gdn_s = Vec::with_capacity(self.cfg.layers);
        let mut gdn_conv = Vec::with_capacity(self.cfg.layers);
        for li in 0..self.cfg.layers {
            if !self.cfg.gdn_layers[li] {
                gdn_s.push(Vec::new());
                gdn_conv.push(Vec::new());
                continue;
            }

            // The aligned buffers are authoritative after a Metal GDN token;
            // the CPU vectors are authoritative before that buffer exists.
            // The CPU path keeps both synchronized, so preferring gm when it
            // exists is correct for both backends.
            if let Some(gm) = self.gdn_metal[li].as_ref() {
                unsafe {
                    gdn_s.push(std::slice::from_raw_parts(gm.state, state_len).to_vec());
                    gdn_conv.push(std::slice::from_raw_parts(gm.conv_state, conv_len).to_vec());
                }
            } else {
                gdn_s.push(self.gdn_s[li].clone());
                gdn_conv.push(self.gdn_conv[li].clone());
            }
        }

        let h = self.cfg.kv_heads;
        let hd = self.cfg.head_dim;
        let mut kv_k = Vec::with_capacity(self.cfg.layers);
        let mut kv_v = Vec::with_capacity(self.cfg.layers);
        for li in 0..self.cfg.layers {
            if self.cfg.gdn_layers[li] {
                kv_k.push(Vec::new());
                kv_v.push(Vec::new());
                continue;
            }
            let expected = h * self.cfg.max_t * hd;
            if self.kv_k[li].len() != expected || self.kv_v[li].len() != expected {
                return Err(format!("layer {li}: malformed attention KV storage"));
            }
            kv_k.push(pack_head_major(
                &self.kv_k[li],
                h,
                self.cfg.max_t,
                hd,
                prefix_len,
            ));
            kv_v.push(pack_head_major(
                &self.kv_v[li],
                h,
                self.cfg.max_t,
                hd,
                prefix_len,
            ));
        }

        let idx_width = self.cfg.idx_kv_heads * self.cfg.idx_head_dim;
        let mut idx_cache = Vec::with_capacity(self.cfg.layers);
        for li in 0..self.cfg.layers {
            if !self.cfg.qsa_layers[li] {
                idx_cache.push(Vec::new());
                continue;
            }
            let active = prefix_len * idx_width;
            if self.idx_cache[li].len() < active {
                return Err(format!("layer {li}: malformed QSA index storage"));
            }
            idx_cache.push(self.idx_cache[li][..active].to_vec());
        }

        Ok(QwenStateSnapshot {
            prefix_len,
            gdn_s,
            gdn_conv,
            kv_k,
            kv_v,
            idx_cache,
            ple_ring: self.ple_ring.clone(),
            ple_conv_state: self.ple_conv_state.clone(),
        })
    }

    /// Restore a snapshot into this already-loaded model. Positions beyond
    /// the restored prefix are intentionally left untouched: causal reads
    /// cannot observe them and sequential continuation overwrites them before
    /// they become visible. Avoiding that zero-fill is part of the cache win.
    pub fn restore_state(&mut self, snapshot: &QwenStateSnapshot) -> Result<(), String> {
        if snapshot.prefix_len > self.cfg.max_t {
            return Err(format!(
                "snapshot prefix {} exceeds target context {}",
                snapshot.prefix_len, self.cfg.max_t
            ));
        }
        if self.sched_pause.is_some() || self.sched_blocked.is_some() {
            return Err("cannot restore during a scheduler-paused token".into());
        }
        for (name, n) in [
            ("gdn_s", snapshot.gdn_s.len()),
            ("gdn_conv", snapshot.gdn_conv.len()),
            ("kv_k", snapshot.kv_k.len()),
            ("kv_v", snapshot.kv_v.len()),
            ("idx_cache", snapshot.idx_cache.len()),
        ] {
            if n != self.cfg.layers {
                return Err(format!(
                    "snapshot {name} has {n} layers; model has {}",
                    self.cfg.layers
                ));
            }
        }

        let state_len = self.cfg.lin_v_heads * self.cfg.lin_k_dim * self.cfg.lin_v_dim;
        let cdim = self.cfg.lin_k_dim * self.cfg.lin_k_heads * 2
            + self.cfg.lin_v_dim * self.cfg.lin_v_heads;
        let conv_len = cdim * self.cfg.conv_kernel.saturating_sub(1);
        for li in 0..self.cfg.layers {
            if !self.cfg.gdn_layers[li] {
                continue;
            }
            if snapshot.gdn_s[li].len() != state_len || snapshot.gdn_conv[li].len() != conv_len {
                return Err(format!("layer {li}: incompatible GDN snapshot geometry"));
            }
            if self.gdn_s[li].len() != state_len || self.gdn_conv[li].len() != conv_len {
                return Err(format!("layer {li}: incompatible target GDN geometry"));
            }
            self.gdn_s[li].copy_from_slice(&snapshot.gdn_s[li]);
            self.gdn_conv[li].copy_from_slice(&snapshot.gdn_conv[li]);
            if let Some(gm) = self.gdn_metal[li].as_ref() {
                unsafe {
                    std::ptr::copy_nonoverlapping(snapshot.gdn_s[li].as_ptr(), gm.state, state_len);
                    std::ptr::copy_nonoverlapping(
                        snapshot.gdn_conv[li].as_ptr(),
                        gm.conv_state,
                        conv_len,
                    );
                }
            }
        }

        let h = self.cfg.kv_heads;
        let hd = self.cfg.head_dim;
        let compact_len = h * snapshot.prefix_len * hd;
        for li in 0..self.cfg.layers {
            if self.cfg.gdn_layers[li] {
                continue;
            }
            if snapshot.kv_k[li].len() != compact_len || snapshot.kv_v[li].len() != compact_len {
                return Err(format!("layer {li}: incompatible KV snapshot geometry"));
            }
            let expected = h * self.cfg.max_t * hd;
            if self.kv_k[li].len() != expected || self.kv_v[li].len() != expected {
                return Err(format!("layer {li}: incompatible target KV geometry"));
            }
            unpack_head_major(
                &snapshot.kv_k[li],
                &mut self.kv_k[li],
                h,
                self.cfg.max_t,
                hd,
                snapshot.prefix_len,
            );
            unpack_head_major(
                &snapshot.kv_v[li],
                &mut self.kv_v[li],
                h,
                self.cfg.max_t,
                hd,
                snapshot.prefix_len,
            );
        }

        let idx_width = self.cfg.idx_kv_heads * self.cfg.idx_head_dim;
        let idx_active = snapshot.prefix_len * idx_width;
        for li in 0..self.cfg.layers {
            if !self.cfg.qsa_layers[li] {
                continue;
            }
            if snapshot.idx_cache[li].len() != idx_active || self.idx_cache[li].len() < idx_active {
                return Err(format!("layer {li}: incompatible QSA snapshot geometry"));
            }
            self.idx_cache[li][..idx_active].copy_from_slice(&snapshot.idx_cache[li]);
        }

        if self.ple_ring.len() != snapshot.ple_ring.len()
            || self.ple_conv_state.len() != snapshot.ple_conv_state.len()
        {
            return Err("incompatible PLE snapshot geometry".into());
        }
        self.ple_ring.copy_from_slice(&snapshot.ple_ring);
        self.ple_conv_state
            .copy_from_slice(&snapshot.ple_conv_state);
        self.sched_blocked = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_head_major_round_trip_preserves_inactive_tail() {
        let heads = 2;
        let max_t = 4;
        let width = 2;
        let prefix = 2;
        let src: Vec<f32> = (0..heads * max_t * width).map(|x| x as f32).collect();
        let packed = pack_head_major(&src, heads, max_t, width, prefix);
        assert_eq!(packed.len(), heads * prefix * width);

        let mut dst = vec![-1.0; src.len()];
        unpack_head_major(&packed, &mut dst, heads, max_t, width, prefix);
        for h in 0..heads {
            let base = h * max_t * width;
            let active = prefix * width;
            assert_eq!(&dst[base..base + active], &src[base..base + active]);
            assert!(
                dst[base + active..base + max_t * width]
                    .iter()
                    .all(|&v| v == -1.0)
            );
        }
    }
}
