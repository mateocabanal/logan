#!/usr/bin/env python3
"""Apply the Qwen4-Exp sigmoid-GDN correctness fix to logan-qwen4/src/lib.rs.

This is intentionally assertion-heavy because lib.rs is a large monolithic runtime
file. Every replacement must match the exact reviewed source or the script aborts
without writing a partial patch.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "logan-qwen4" / "src" / "lib.rs"
text = PATH.read_text()
original = text


def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one source match, found {count}")
    text = text.replace(old, new, 1)


replace_once(
    """// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Cfg {
""",
    """// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

/// Activation applied by Qwen4-Exp's gated RMSNorm at the GDN output.
/// Qwen3.8-Flash-Next explicitly selects Sigmoid; older Qwen variants fall
/// back to hidden_act (normally SiLU).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputGate {
    Silu,
    Sigmoid,
}

impl OutputGate {
    fn from_config(v: &serde_json::Value) -> Result<Self, String> {
        let name = v
            .get("output_gate_type")
            .and_then(|x| x.as_str())
            .or_else(|| v.get("hidden_act").and_then(|x| x.as_str()))
            .unwrap_or("silu");
        match name {
            "silu" => Ok(Self::Silu),
            "sigmoid" => Ok(Self::Sigmoid),
            other => Err(format!(
                "unsupported Qwen4 output_gate_type/hidden_act {other:?}; expected silu or sigmoid"
            )),
        }
    }
}

#[derive(Clone)]
pub struct Cfg {
""",
    "insert OutputGate",
)

replace_once(
    """    pub vocab: usize,
    pub eps: f32,
    gdn_layers: Vec<bool>,
""",
    """    pub vocab: usize,
    pub eps: f32,
    pub output_gate: OutputGate,
    gdn_layers: Vec<bool>,
""",
    "Cfg.output_gate field",
)

replace_once(
    """    let get = |k: &str| v.get(k).and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let num = |k: &str| v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
""",
    """    let get = |k: &str| v.get(k).and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let num = |k: &str| v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
    let output_gate = OutputGate::from_config(&v)?;
""",
    "parse output gate",
)

replace_once(
    """        vocab: get("vocab_size"),
        eps: num("rms_norm_eps").max(1e-6),
        gdn_layers,
""",
    """        vocab: get("vocab_size"),
        eps: num("rms_norm_eps").max(1e-6),
        output_gate,
        gdn_layers,
""",
    "Cfg output_gate init",
)

replace_once(
    """fn rmsnorm_gated_row(out: &mut [f32], x: &[f32], z: &[f32], w: &[f32], eps: f32) {
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
""",
    """fn rmsnorm_gated_row(
    out: &mut [f32],
    x: &[f32],
    z: &[f32],
    w: &[f32],
    eps: f32,
    gate: OutputGate,
) {
    let d = x.len();
    let mut ms = 0.0_f64;
    for i in 0..d {
        ms += x[i] as f64 * x[i] as f64;
    }
    let r = 1.0 / (ms as f32 / d as f32 + eps).sqrt();

    // Dispatch once per value-head row, not once per element. Sigmoid and
    // SiLU both require one exp; this adds no inner-loop branch to Qwen3.8.
    match gate {
        OutputGate::Silu => {
            for i in 0..d {
                out[i] = w[i] * (x[i] * r) * silu(z[i]);
            }
        }
        OutputGate::Sigmoid => {
            for i in 0..d {
                let sigmoid_z = 1.0 / (1.0 + (-z[i]).exp());
                out[i] = w[i] * (x[i] * r) * sigmoid_z;
            }
        }
    }
}
""",
    "configurable GDN gated norm",
)

replace_once(
    """        let gdn_enabled = std::env::var("QWEN_GDN_METAL")
            .map(|v| v != "0")
            .unwrap_or(true);
""",
    """        // The current Metal kernel encodes the historical SiLU gate.
        // Fail closed for sigmoid checkpoints instead of silently running the
        // wrong model. Apple defaults already prefer the faster BNNS CPU GDN.
        let gdn_enabled = std::env::var("QWEN_GDN_METAL")
            .map(|v| v != "0")
            .unwrap_or(true)
            && c.output_gate == OutputGate::Silu;
""",
    "fail-closed Metal sigmoid gate",
)

replace_once(
    """                // rc == 0: declined pre-submit. GPU state is the truth from
                // any earlier Metal tokens -> pull it into the CPU state
                // before the scalar reference below runs.
                let n_state = vheads * kd * vd;
                let n_conv = cdim * (kk - 1);
                unsafe {
                    std::ptr::copy_nonoverlapping(gm.state, self.gdn_s[li].as_mut_ptr(), n_state);
                    std::ptr::copy_nonoverlapping(
                        gm.conv_state,
                        self.gdn_conv[li].as_mut_ptr(),
                        n_conv,
                    );
                }
""",
    """                // rc == 0: declined pre-submit. The scalar path below
                // operates directly on the aligned state when it exists, so
                // no multi-megabyte GPU->CPU mirror copy is needed here.
""",
    "remove Metal decline state copy",
)

replace_once(
    """        if kk > 1 {
            let conv_st = &mut self.gdn_conv[li];
""",
    """        if kk > 1 {
            // build_gdn_metal() is also the single-copy BF16 re-home on Apple,
            // even when Metal GDN execution is disabled. Use its aligned conv
            // state directly so CPU decode does not maintain/copy a second
            // mirror every token.
            let conv_st: &mut [f32] = if let Some(gm) = self.gdn_metal[li].as_mut() {
                unsafe { std::slice::from_raw_parts_mut(gm.conv_state, cdim * (kk - 1)) }
            } else {
                &mut self.gdn_conv[li]
            };
""",
    "single-source conv state",
)

replace_once(
    """        let s = &mut self.gdn_s[li];
        let mut qh = vec![0.0; vheads * kd];
""",
    """        let mut qh = vec![0.0; vheads * kd];
""",
    "defer recurrent state borrow",
)

old_recurrence = """        let mut snew = vec![0.0; vheads * kd * vd];
        let mut kv_mem = vec![0.0; vd];
        for h in 0..vheads {
            let ga = -layer.gdn_a_log[h].exp() * (1.0 + (a[h] + layer.gdn_dt_bias[h]).exp()).ln();
            let gt = ga.exp();
            let bt = 1.0 / (1.0 + (-b[h]).exp());
            let sh = &s[h * kd * vd..(h + 1) * kd * vd];
            let sn = &mut snew[h * kd * vd..(h + 1) * kd * vd];
            let qhh = &qh[h * kd..(h + 1) * kd];
            let khh = &kh[h * kd..(h + 1) * kd];
            let vhh = &vh[h * vd..(h + 1) * vd];
            for d in 0..vd {
                kv_mem[d] = 0.0;
            }
            for kk2 in 0..kd {
                let srow = &sh[kk2 * vd..(kk2 + 1) * vd];
                for d in 0..vd {
                    let sv = srow[d] * gt;
                    sn[kk2 * vd + d] = sv;
                    kv_mem[d] += sv * khh[kk2];
                }
            }
            for d in 0..vd {
                let delta = (vhh[d] - kv_mem[d]) * bt;
                let mut acc = 0.0_f32;
                for kk2 in 0..kd {
                    let si = kk2 * vd + d;
                    let next_s = sn[si] + khh[kk2] * delta;
                    sn[si] = next_s;
                    acc += next_s * qhh[kk2];
                }
                kv_mem[d] = acc;
            }
            for d in 0..vd {
                vh[h * vd + d] = kv_mem[d];
            }
        }
        self.gdn_s[li].copy_from_slice(&snew);
        // Keep the aligned (GPU-visible) state in sync after a CPU token so
        // the next Metal token starts from the same recurrence (C has ONE
        // state; here we sync on the CPU path only).
        if let Some(gm) = &self.gdn_metal[li] {
            unsafe {
                std::ptr::copy_nonoverlapping(self.gdn_s[li].as_ptr(), gm.state, vheads * kd * vd);
                std::ptr::copy_nonoverlapping(
                    self.gdn_conv[li].as_ptr(),
                    gm.conv_state,
                    cdim * (kk - 1),
                );
            }
        }
"""
new_recurrence = """        let state_len = vheads * kd * vd;
        {
            // Same recurrence and reduction order as the previous snew path,
            // but update the authoritative state in place. For Flash-Next this
            // removes a ~3 MiB zeroed temporary plus a ~3 MiB final copy for
            // each of 36 GDN layers on every token.
            let s: &mut [f32] = if let Some(gm) = self.gdn_metal[li].as_mut() {
                unsafe { std::slice::from_raw_parts_mut(gm.state, state_len) }
            } else {
                &mut self.gdn_s[li]
            };
            let mut kv_mem = vec![0.0; vd];
            for h in 0..vheads {
                let ga = -layer.gdn_a_log[h].exp()
                    * (1.0 + (a[h] + layer.gdn_dt_bias[h]).exp()).ln();
                let gt = ga.exp();
                let bt = 1.0 / (1.0 + (-b[h]).exp());
                let sh = &mut s[h * kd * vd..(h + 1) * kd * vd];
                let qhh = &qh[h * kd..(h + 1) * kd];
                let khh = &kh[h * kd..(h + 1) * kd];
                let vhh = &vh[h * vd..(h + 1) * vd];
                for d in 0..vd {
                    kv_mem[d] = 0.0;
                }
                for kk2 in 0..kd {
                    for d in 0..vd {
                        let si = kk2 * vd + d;
                        let sv = sh[si] * gt;
                        sh[si] = sv;
                        kv_mem[d] += sv * khh[kk2];
                    }
                }
                for d in 0..vd {
                    let delta = (vhh[d] - kv_mem[d]) * bt;
                    let mut acc = 0.0_f32;
                    for kk2 in 0..kd {
                        let si = kk2 * vd + d;
                        let next_s = sh[si] + khh[kk2] * delta;
                        sh[si] = next_s;
                        acc += next_s * qhh[kk2];
                    }
                    kv_mem[d] = acc;
                }
                for d in 0..vd {
                    vh[h * vd + d] = kv_mem[d];
                }
            }
        }
"""
replace_once(old_recurrence, new_recurrence, "in-place recurrent state")

replace_once(
    """                &layer.gdn_norm,
                c.eps,
            );
""",
    """                &layer.gdn_norm,
                c.eps,
                c.output_gate,
            );
""",
    "GDN output-gate dispatch",
)

old_driver = """pub fn run_greedy_with(mut model: Model, cfg: Cfg, prompt: &[u32], max_new: usize) -> Vec<u32> {
    let profile = logan_core::telemetry::enabled();
    let t0 = std::time::Instant::now();
    for (i, &t) in prompt.iter().enumerate() {
        model.forward_token(t as usize, i);
    }
    let mut out = Vec::with_capacity(max_new);
    let mut last = *prompt.last().unwrap_or(&0);
    let start = prompt.len();
    for pos in start..start + max_new {
        let logits = model.forward_token(last as usize, pos);
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        out.push(next);
        last = next;
    }
    if profile {
        model.profile_summary(max_new, t0.elapsed().as_secs_f64() * 1e3);
    }
    out
}
"""
new_driver = """pub fn run_greedy_with(mut model: Model, _cfg: Cfg, prompt: &[u32], max_new: usize) -> Vec<u32> {
    let profile = logan_core::telemetry::enabled();
    let t0 = std::time::Instant::now();
    if prompt.is_empty() || max_new == 0 {
        return Vec::new();
    }

    // The final prompt forward already returns the logits that predict token
    // prompt.len(). Refeeding prompt.last() at that position duplicates the
    // final prompt token in recurrent/KV state and is not causal-LM decode.
    let mut logits = Vec::new();
    for (i, &t) in prompt.iter().enumerate() {
        logits = model.forward_token(t as usize, i);
    }

    let mut out = Vec::with_capacity(max_new);
    for step in 0..max_new {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        out.push(next);
        if step + 1 < max_new {
            logits = model.forward_token(next as usize, prompt.len() + step);
        }
    }
    if profile {
        model.profile_summary(max_new, t0.elapsed().as_secs_f64() * 1e3);
    }
    out
}
"""
replace_once(old_driver, new_driver, "canonical causal decode driver")

if text == original:
    raise SystemExit("no changes made")
PATH.write_text(text)
print(f"patched {PATH.relative_to(ROOT)}")
