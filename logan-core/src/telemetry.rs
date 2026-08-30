//! Engine-neutral telemetry: per-token decode spans + Metal/MetalIO
//! counters, gated by `LOGAN_PROFILE=1`.
//!
//! Port of the C engine's span model (moe_route/io/shared/gpu/fill_ms) —
//! the regime-independent metrics that make A/B verdicts trustworthy
//! (wall time on a throttled box is garbage; loads/bytes/waits are not).

use std::time::Instant;

/// Per-token decode spans. Engines call `begin`/`end` around their phases;
/// the core's decode loop (when it owns the loop) fills these itself.
#[derive(Debug, Default, Clone)]
pub struct TokenSpans {
    pub route_ms: f64,
    pub io_ms: f64,
    pub shared_ms: f64,
    pub gpu_ms: f64,
    pub fill_ms: f64,
    /// GDN layer phase (Metal direct calls + CPU fallback), ms.
    pub gdn_ms: f64,
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
    if !enabled() {
        return;
    }
    eprintln!(
        "logan profile: tokens={tokens} route={:.1} io={:.1} shared={:.1} gpu={:.1} fill={:.1} gdn={:.1} attn={:.1} hc={:.1} head={:.1} total={:.1} ms/tok | cache hits={cache_hits} misses={cache_misses} | gdn_metal_ok={} | metal encode={} submit={} wait={} kernel={} ns fused_calls={} fused_experts={} | mio loads={} bytes={} waits={} fails={}",
        spans.route_ms / tokens.max(1) as f64,
        spans.io_ms / tokens.max(1) as f64,
        spans.shared_ms / tokens.max(1) as f64,
        spans.gpu_ms / tokens.max(1) as f64,
        spans.fill_ms / tokens.max(1) as f64,
        spans.gdn_ms / tokens.max(1) as f64,
        spans.attn_ms / tokens.max(1) as f64,
        spans.hc_ms / tokens.max(1) as f64,
        spans.head_ms / tokens.max(1) as f64,
        spans.total_ms / tokens.max(1) as f64,
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
