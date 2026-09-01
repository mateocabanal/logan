from pathlib import Path


def replace_exact(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {n}\n--- OLD ---\n{old}")
    p.write_text(text.replace(old, new, 1))


# Do not build the large aligned attention re-home when Metal attention is
# disabled. Explicit QWEN_ATTN_METAL=1 still builds and uses the old path.
replace_exact(
    "logan-qwen4/src/lib.rs",
    '''        if self.attn_metal[li].is_none() && !layer.is_gdn && crate::ffi::direct_available() {
            self.attn_metal[li] = Self::build_attn_metal(layer, &self.cfg);
        }

        if let Some(am) = self.attn_metal[li].as_ref() {
            let gate = std::env::var("QWEN_ATTN_METAL")
                .map(|v| v != "0")
                .unwrap_or(true);

            if gate {
''',
    '''        let metal_enabled = std::env::var("QWEN_ATTN_METAL")
            .map(|v| v != "0")
            .unwrap_or(true);

        if metal_enabled
            && self.attn_metal[li].is_none()
            && !layer.is_gdn
            && crate::ffi::direct_available()
        {
            self.attn_metal[li] = Self::build_attn_metal(layer, &self.cfg);
        }

        if let Some(am) = self.attn_metal[li].as_ref() {
            if metal_enabled {
''',
)

# GDN sub-phase telemetry. Timers are only created under LOGAN_PROFILE so
# ordinary decode pays only the boolean branch checks.
replace_exact(
    "logan-qwen4/src/lib.rs",
    '''        let cdim = kdim * 2 + vdim;
        let kk = c.conv_kernel;

        // One-time aligned re-home''',
    '''        let cdim = kdim * 2 + vdim;
        let kk = c.conv_kernel;
        let profile_gdn_parts = logan_core::telemetry::enabled();

        // One-time aligned re-home''',
)

replace_exact(
    "logan-qwen4/src/lib.rs",
    '''        let mut qkv = vec![0.0; cdim];
        let mut a = vec![0.0; vheads];
        let mut b = vec![0.0; vheads];
        let mut z = vec![0.0; vdim];

        if let Some(gm) = self.gdn_metal[li].as_ref() {''',
    '''        let mut qkv = vec![0.0; cdim];
        let mut a = vec![0.0; vheads];
        let mut b = vec![0.0; vheads];
        let mut z = vec![0.0; vdim];
        let gdn_in_t0 = profile_gdn_parts.then(std::time::Instant::now);

        if let Some(gm) = self.gdn_metal[li].as_ref() {''',
)

replace_exact(
    "logan-qwen4/src/lib.rs",
    '''            matmul(&mut z, x, &layer.gdn_in_z);
        }

        let mut y = vec![0.0; cdim];''',
    '''            matmul(&mut z, x, &layer.gdn_in_z);
        }
        if let Some(t0) = gdn_in_t0 {
            self.spans.gdn_in_proj_ms += t0.elapsed().as_secs_f64() * 1e3;
        }

        let gdn_conv_t0 = profile_gdn_parts.then(std::time::Instant::now);
        let mut y = vec![0.0; cdim];''',
)

replace_exact(
    "logan-qwen4/src/lib.rs",
    '''        } else {
            for ch in 0..cdim {
                y[ch] = silu(layer.gdn_conv1d[ch] * qkv[ch]);
            }
        }

        let q_ = &y[..kdim];''',
    '''        } else {
            for ch in 0..cdim {
                y[ch] = silu(layer.gdn_conv1d[ch] * qkv[ch]);
            }
        }
        if let Some(t0) = gdn_conv_t0 {
            self.spans.gdn_conv_ms += t0.elapsed().as_secs_f64() * 1e3;
        }

        let gdn_prepare_t0 = profile_gdn_parts.then(std::time::Instant::now);
        let q_ = &y[..kdim];''',
)

replace_exact(
    "logan-qwen4/src/lib.rs",
    '''                qh[h * kd + d] *= sc;
            }
        }

        let state_len = vheads * kd * vd;''',
    '''                qh[h * kd + d] *= sc;
            }
        }
        if let Some(t0) = gdn_prepare_t0 {
            self.spans.gdn_prepare_ms += t0.elapsed().as_secs_f64() * 1e3;
        }

        let gdn_recur_t0 = profile_gdn_parts.then(std::time::Instant::now);
        let state_len = vheads * kd * vd;''',
)

replace_exact(
    "logan-qwen4/src/lib.rs",
    '''            }
        }

        let mut normed = vec![0.0; vdim];
        for h in 0..vheads {''',
    '''            }
        }
        if let Some(t0) = gdn_recur_t0 {
            self.spans.gdn_recur_ms += t0.elapsed().as_secs_f64() * 1e3;
        }

        let gdn_gate_t0 = profile_gdn_parts.then(std::time::Instant::now);
        let mut normed = vec![0.0; vdim];
        for h in 0..vheads {''',
)

replace_exact(
    "logan-qwen4/src/lib.rs",
    '''                c.output_gate,
            );
        }
        if let Some(gm) = self.gdn_metal[li].as_ref() {''',
    '''                c.output_gate,
            );
        }
        if let Some(t0) = gdn_gate_t0 {
            self.spans.gdn_gate_ms += t0.elapsed().as_secs_f64() * 1e3;
        }

        let gdn_out_t0 = profile_gdn_parts.then(std::time::Instant::now);
        if let Some(gm) = self.gdn_metal[li].as_ref() {''',
)

replace_exact(
    "logan-qwen4/src/lib.rs",
    '''        } else {
            matmul(out, &normed, &layer.gdn_out);
        }
    }

    /// Input-side attention projection.''',
    '''        } else {
            matmul(out, &normed, &layer.gdn_out);
        }
        if let Some(t0) = gdn_out_t0 {
            self.spans.gdn_out_proj_ms += t0.elapsed().as_secs_f64() * 1e3;
        }
    }

    /// Input-side attention projection.''',
)

# Engine-neutral counters and rendering.
replace_exact(
    "logan-core/src/telemetry.rs",
    '''    /// GDN layer phase (Metal direct calls + CPU fallback), ms.
    pub gdn_ms: f64,
    /// Full-attention/QSA layer phase, ms.''',
    '''    /// GDN layer phase (Metal direct calls + CPU fallback), ms.
    pub gdn_ms: f64,
    /// GDN input dense projections (qkv + z + a + b), ms.
    pub gdn_in_proj_ms: f64,
    /// GDN depthwise convolution + conv-state update, ms.
    pub gdn_conv_ms: f64,
    /// GDN q/k expansion, L2 normalization, and q scaling, ms.
    pub gdn_prepare_ms: f64,
    /// GDN recurrent-state decay/update/readout, ms.
    pub gdn_recur_ms: f64,
    /// GDN gated RMSNorm/output gate, ms.
    pub gdn_gate_ms: f64,
    /// GDN output dense projection, ms.
    pub gdn_out_proj_ms: f64,
    /// Full-attention/QSA layer phase, ms.''',
)

replace_exact(
    "logan-core/src/telemetry.rs",
    '''        "logan profile: tokens={tokens} route={:.1} io={:.1} shared={:.1} gpu={:.1} fill={:.1} gdn={:.1} attn={:.1} hc={:.1} head={:.1} total={:.1} ms/tok | cache hits={cache_hits} misses={cache_misses} | gdn_metal_ok={} | metal encode={} submit={} wait={} kernel={} ns fused_calls={} fused_experts={} | mio loads={} bytes={} waits={} fails={}",''',
    '''        "logan profile: tokens={tokens} route={:.1} io={:.1} shared={:.1} gpu={:.1} fill={:.1} gdn={:.1} attn={:.1} hc={:.1} head={:.1} total={:.1} ms/tok | gdn_parts in={:.1} conv={:.1} prep={:.1} recur={:.1} gate={:.1} out={:.1} | cache hits={cache_hits} misses={cache_misses} | gdn_metal_ok={} | metal encode={} submit={} wait={} kernel={} ns fused_calls={} fused_experts={} | mio loads={} bytes={} waits={} fails={}",''',
)

replace_exact(
    "logan-core/src/telemetry.rs",
    '''        spans.head_ms / tokens.max(1) as f64,
        spans.total_ms / tokens.max(1) as f64,
        spans.gdn_metal_ok,''',
    '''        spans.head_ms / tokens.max(1) as f64,
        spans.total_ms / tokens.max(1) as f64,
        spans.gdn_in_proj_ms / tokens.max(1) as f64,
        spans.gdn_conv_ms / tokens.max(1) as f64,
        spans.gdn_prepare_ms / tokens.max(1) as f64,
        spans.gdn_recur_ms / tokens.max(1) as f64,
        spans.gdn_gate_ms / tokens.max(1) as f64,
        spans.gdn_out_proj_ms / tokens.max(1) as f64,
        spans.gdn_metal_ok,''',
)

replace_exact(
    "logan-core/src/telemetry.rs",
    '''            gdn_metal_ok: 0,
            total_ms: 52.0,
        };''',
    '''            gdn_metal_ok: 0,
            total_ms: 52.0,
            ..Default::default()
        };''',
)

print("materialized Qwen4 BNNS-attention + GDN telemetry patch")
