//! Engine-neutral telemetry: per-token decode spans + Metal/MetalIO
//! counters, gated by `LOGAN_PROFILE=1`.
//!
//! Port of the C engine's span model (moe_route/io/shared/gpu/fill_ms) —
//! the regime-independent metrics that make A/B verdicts trustworthy
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Per-token decode spans. Engines call `begin`/`end` around their phases;
/// the core's decode loop (when it owns the loop) fills these itself.
#[derive(Debug, Default, Clone)]
pub struct TokenSpans {
    pub route_ms: f64,
    /// Learned-prerouter (RouteScout/Edge0 head) evaluation, ms. Distinct from
    /// `route_ms`, which is the native gate: a predictor's own cost has to be
    /// comparable against the I/O it is trying to save.
    pub predict_ms: f64,
    pub io_ms: f64,
    pub shared_ms: f64,
    pub gpu_ms: f64,
    pub fill_ms: f64,
    /// GDN layer phase (Metal direct calls + CPU fallback), ms.
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
    /// Full-attention/QSA layer phase, ms.
    pub attn_ms: f64,
    /// Hyper-connection mixer phase (both hc_mix calls per layer), ms.
    pub hc_ms: f64,
    /// Head phase (final norm + lm_head), ms.
    pub head_ms: f64,
    /// Count of gdn_token calls that ran on Metal (rc > 0).
    pub gdn_metal_ok: u64,
    pub total_ms: f64,
}

impl TokenSpans {
    /// Component-wise difference between two *cumulative* span snapshots
    /// (self minus `before`). `total_ms` is a caller-supplied whole-request
    /// figure, not an accumulator, so it is not differenced — the caller sets
    /// it on the result.
    ///
    /// This is the decode-boundary measurement primitive: spans accumulate
    /// across prefill and decode, but the reported per-token figures must
    /// divide only the decode window by the decode forward count.
    pub fn delta_from(&self, before: &TokenSpans) -> TokenSpans {
        TokenSpans {
            route_ms: self.route_ms - before.route_ms,
            predict_ms: self.predict_ms - before.predict_ms,
            io_ms: self.io_ms - before.io_ms,
            shared_ms: self.shared_ms - before.shared_ms,
            gpu_ms: self.gpu_ms - before.gpu_ms,
            fill_ms: self.fill_ms - before.fill_ms,
            gdn_ms: self.gdn_ms - before.gdn_ms,
            gdn_in_proj_ms: self.gdn_in_proj_ms - before.gdn_in_proj_ms,
            gdn_conv_ms: self.gdn_conv_ms - before.gdn_conv_ms,
            gdn_prepare_ms: self.gdn_prepare_ms - before.gdn_prepare_ms,
            gdn_recur_ms: self.gdn_recur_ms - before.gdn_recur_ms,
            gdn_gate_ms: self.gdn_gate_ms - before.gdn_gate_ms,
            gdn_out_proj_ms: self.gdn_out_proj_ms - before.gdn_out_proj_ms,
            attn_ms: self.attn_ms - before.attn_ms,
            hc_ms: self.hc_ms - before.hc_ms,
            head_ms: self.head_ms - before.head_ms,
            gdn_metal_ok: self.gdn_metal_ok.saturating_sub(before.gdn_metal_ok),
            total_ms: 0.0,
        }
    }
}

/// A running span timer.
pub struct Span {
    name: &'static str,
    start: Instant,
    acc: f64,
}

impl Span {
    pub fn begin(name: &'static str) -> Span {
        Span {
            name,
            start: Instant::now(),
            acc: 0.0,
        }
    }

    /// Accumulate elapsed time since the last `checkpoint` (or begin).
    pub fn checkpoint(&mut self) -> f64 {
        let now = Instant::now();
        let dt = now.duration_since(self.start).as_secs_f64() * 1e3;
        self.start = now;
        self.acc += dt;
        dt
    }

    pub fn end(mut self) -> f64 {
        self.checkpoint();
        self.acc
    }

    pub fn name(&self) -> &'static str {
        self.name
    }
}

/// Metal/MetalIO counters (from the C backend's profile_get + metalio_stats).
#[derive(Debug, Default, Clone)]
pub struct MetalCounters {
    pub encode_ns: u64,
    pub submit_ns: u64,
    pub wait_ns: u64,
    pub kernel_ns: u64,
    pub fused_calls: u64,
    pub fused_experts: u64,
    pub mio_loads: u64,
    pub mio_bytes: u64,
    pub mio_waits: u64,
    pub mio_fails: u64,
}
/// Complete-round wall-clock components used by placement calibration.
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlacementRoundTelemetry {
    pub wall_ms: f64,
    pub draft_ms: f64,
    pub verification_ms: f64,
    pub handoff_ms: f64,
    pub fallback_ms: f64,
}

/// Compact, backend-neutral placement counters.  Values are populated only
/// from observed execution; zero means that no measurement was supplied.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlacementTelemetry {
    pub requested_mode: String,
    pub selected_backend: String,
    pub fallback_reason: Option<String>,
    pub ane_calls: u64,
    pub ane_fallbacks: u64,
    pub ane_probe_rounds: u64,
    pub ane_promotions: u64,
    pub ane_demotions: u64,
    pub round: PlacementRoundTelemetry,
}

/// Profile gate: LOGAN_PROFILE=1 enables span collection + emission.
pub fn enabled() -> bool {
    std::env::var("LOGAN_PROFILE")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
}

/// Emit a per-request summary line (C-style, one line per request).
pub fn emit_request_summary(
    tokens: usize,
    spans: &TokenSpans,
    metal: &MetalCounters,
    cache_hits: u64,
    cache_misses: u64,
) {
    emit_request_summary_with_placement(
        tokens,
        spans,
        metal,
        cache_hits,
        cache_misses,
        &PlacementTelemetry::default(),
    );
}

/// Emit the summary with actual placement selection and fallback evidence.
pub fn emit_request_summary_with_placement(
    tokens: usize,
    spans: &TokenSpans,
    metal: &MetalCounters,
    cache_hits: u64,
    cache_misses: u64,
    placement: &PlacementTelemetry,
) {
    if !enabled() {
        return;
    }
    eprintln!(
        "logan profile: tokens={tokens} route={:.1} predict={:.1} io={:.1} shared={:.1} gpu={:.1} fill={:.1} gdn={:.1} attn={:.1} hc={:.1} head={:.1} total={:.1} ms/tok | gdn_parts in={:.1} conv={:.1} prep={:.1} recur={:.1} gate={:.1} out={:.1} | cache hits={cache_hits} misses={cache_misses} | gdn_metal_ok={} | metal encode={} submit={} wait={} kernel={} ns fused_calls={} fused_experts={} | mio loads={} bytes={} waits={} fails={} | placement requested={} selected={} fallback={} ane_calls={} ane_fallbacks={} probes={} promotions={} demotions={} round wall={:.1} draft={:.1} verify={:.1} handoff={:.1} fallback_ms={:.1}",
        spans.route_ms / tokens.max(1) as f64,
        spans.predict_ms / tokens.max(1) as f64,
        spans.io_ms / tokens.max(1) as f64,
        spans.shared_ms / tokens.max(1) as f64,
        spans.gpu_ms / tokens.max(1) as f64,
        spans.fill_ms / tokens.max(1) as f64,
        spans.gdn_ms / tokens.max(1) as f64,
        spans.attn_ms / tokens.max(1) as f64,
        spans.hc_ms / tokens.max(1) as f64,
        spans.head_ms / tokens.max(1) as f64,
        spans.total_ms / tokens.max(1) as f64,
        spans.gdn_in_proj_ms / tokens.max(1) as f64,
        spans.gdn_conv_ms / tokens.max(1) as f64,
        spans.gdn_prepare_ms / tokens.max(1) as f64,
        spans.gdn_recur_ms / tokens.max(1) as f64,
        spans.gdn_gate_ms / tokens.max(1) as f64,
        spans.gdn_out_proj_ms / tokens.max(1) as f64,
        spans.gdn_metal_ok,
        metal.encode_ns,
        metal.submit_ns,
        metal.wait_ns,
        metal.kernel_ns,
        metal.fused_calls,
        metal.fused_experts,
        metal.mio_loads,
        metal.mio_bytes,
        metal.mio_waits,
        metal.mio_fails,
        placement.requested_mode,
        placement.selected_backend,
        placement.fallback_reason.as_deref().unwrap_or("none"),
        placement.ane_calls,
        placement.ane_fallbacks,
        placement.ane_probe_rounds,
        placement.ane_promotions,
        placement.ane_demotions,
        placement.round.wall_ms,
        placement.round.draft_ms,
        placement.round.verification_ms,
        placement.round.handoff_ms,
        placement.round.fallback_ms,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn span_accumulates() {
        let mut s = Span::begin("test");
        assert_eq!(s.name(), "test");
        thread::sleep(Duration::from_millis(5));
        let dt = s.checkpoint();
        assert!(dt >= 4.0, "dt={dt}");
        thread::sleep(Duration::from_millis(5));
        let total = s.end();
        assert!(total >= 9.0, "total={total}");
    }

    #[test]
    fn profile_gate_default_off() {
        // unset env in test: default off
        unsafe { std::env::remove_var("LOGAN_PROFILE") };
        assert!(!enabled());
    }

    #[test]
    fn summary_line_emits_when_enabled() {
        unsafe { std::env::set_var("LOGAN_PROFILE", "1") };
        let spans = TokenSpans {
            route_ms: 10.0,
            predict_ms: 0.0,
            io_ms: 20.0,
            shared_ms: 5.0,
            gpu_ms: 15.0,
            fill_ms: 2.0,
            gdn_ms: 0.0,
            attn_ms: 0.0,
            hc_ms: 0.0,
            head_ms: 0.0,
            gdn_metal_ok: 0,
            total_ms: 52.0,
            ..Default::default()
        };
        let metal = MetalCounters {
            mio_loads: 61,
            mio_bytes: 1_400_000_000,
            ..Default::default()
        };
        emit_request_summary(8, &spans, &metal, 100, 50);
        unsafe { std::env::remove_var("LOGAN_PROFILE") };
    }
}
