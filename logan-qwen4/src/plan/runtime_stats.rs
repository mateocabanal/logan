//! Lightweight live runtime observability for interactive clients.
//!
//! Unlike `profile_summary`, this API does not print or require stderr parsing.
//! Counters are cumulative for the model/process lifetime; callers can take two
//! snapshots and use `delta_from` to obtain a per-turn view.

use crate::{Model, cache_cap};

#[derive(Clone, Debug, Default)]
pub struct RuntimeFeatures {
    pub metal_direct: bool,
    pub metal_overlap: bool,
    pub bnns_bf16: bool,
    pub gdn_metal: bool,
    pub gdn_single_copy: bool,
    pub attn_metal: bool,
    pub qsa_index_metal: bool,
    pub shared_io_overlap: bool,
    pub prefix_cache: bool,
    pub prefix_cache_write: bool,
}

#[derive(Clone, Debug, Default)]
pub struct RuntimeStats {
    pub route_ms: f64,
    pub io_ms: f64,
    pub shared_ms: f64,
    pub gpu_ms: f64,
    pub fill_ms: f64,
    pub gdn_ms: f64,
    pub attn_ms: f64,
    pub hc_ms: f64,
    pub head_ms: f64,
    pub gdn_metal_ok: u64,

    pub expert_hits: u64,
    pub expert_misses: u64,
    pub expert_evictions: u64,
    pub expert_resident: usize,
    pub expert_capacity: usize,

    pub metal_encode_ns: u64,
    pub metal_submit_ns: u64,
    pub metal_wait_ns: u64,
    pub metal_kernel_ns: u64,
    pub fused_calls: u64,
    pub fused_experts: u64,

    pub mio_loads: u64,
    pub mio_bytes: u64,
    pub mio_waits: u64,
    pub mio_fails: u64,
    pub mio_prefetch_loads: u64,
    pub mio_prefetch_used: u64,
    pub mio_prefetch_wasted: u64,
    pub mio_outstanding: u64,
    pub mio_peak_outstanding: u64,
    pub mio_latency_samples: u64,
    pub mio_total_latency_s: f64,

    pub context_limit: usize,
    pub vocab_size: usize,
    pub eos_token_id: i64,
    pub features: RuntimeFeatures,
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name).map(|v| v != "0").unwrap_or(default)
}

impl RuntimeStats {
    /// Counter/timing delta between two cumulative snapshots. Gauge-like
    /// fields (resident slots, limits, feature flags, outstanding I/O) retain
    /// their current value from `self`.
    pub fn delta_from(&self, before: &RuntimeStats) -> RuntimeStats {
        RuntimeStats {
            route_ms: self.route_ms - before.route_ms,
            io_ms: self.io_ms - before.io_ms,
            shared_ms: self.shared_ms - before.shared_ms,
            gpu_ms: self.gpu_ms - before.gpu_ms,
            fill_ms: self.fill_ms - before.fill_ms,
            gdn_ms: self.gdn_ms - before.gdn_ms,
            attn_ms: self.attn_ms - before.attn_ms,
            hc_ms: self.hc_ms - before.hc_ms,
            head_ms: self.head_ms - before.head_ms,
            gdn_metal_ok: self.gdn_metal_ok.saturating_sub(before.gdn_metal_ok),
            expert_hits: self.expert_hits.saturating_sub(before.expert_hits),
            expert_misses: self.expert_misses.saturating_sub(before.expert_misses),
            expert_evictions: self
                .expert_evictions
                .saturating_sub(before.expert_evictions),
            expert_resident: self.expert_resident,
            expert_capacity: self.expert_capacity,
            metal_encode_ns: self.metal_encode_ns.saturating_sub(before.metal_encode_ns),
            metal_submit_ns: self.metal_submit_ns.saturating_sub(before.metal_submit_ns),
            metal_wait_ns: self.metal_wait_ns.saturating_sub(before.metal_wait_ns),
            metal_kernel_ns: self.metal_kernel_ns.saturating_sub(before.metal_kernel_ns),
            fused_calls: self.fused_calls.saturating_sub(before.fused_calls),
            fused_experts: self.fused_experts.saturating_sub(before.fused_experts),
            mio_loads: self.mio_loads.saturating_sub(before.mio_loads),
            mio_bytes: self.mio_bytes.saturating_sub(before.mio_bytes),
            mio_waits: self.mio_waits.saturating_sub(before.mio_waits),
            mio_fails: self.mio_fails.saturating_sub(before.mio_fails),
            mio_prefetch_loads: self
                .mio_prefetch_loads
                .saturating_sub(before.mio_prefetch_loads),
            mio_prefetch_used: self
                .mio_prefetch_used
                .saturating_sub(before.mio_prefetch_used),
            mio_prefetch_wasted: self
                .mio_prefetch_wasted
                .saturating_sub(before.mio_prefetch_wasted),
            mio_outstanding: self.mio_outstanding,
            mio_peak_outstanding: self.mio_peak_outstanding,
            mio_latency_samples: self
                .mio_latency_samples
                .saturating_sub(before.mio_latency_samples),
            mio_total_latency_s: self.mio_total_latency_s - before.mio_total_latency_s,
            context_limit: self.context_limit,
            vocab_size: self.vocab_size,
            eos_token_id: self.eos_token_id,
            features: self.features.clone(),
        }
    }

    pub fn expert_hit_rate(&self) -> f64 {
        let total = self.expert_hits + self.expert_misses;
        if total == 0 {
            0.0
        } else {
            self.expert_hits as f64 / total as f64
        }
    }

    pub fn mio_avg_latency_ms(&self) -> f64 {
        if self.mio_latency_samples == 0 {
            0.0
        } else {
            self.mio_total_latency_s * 1e3 / self.mio_latency_samples as f64
        }
    }
}

impl Model {
    pub fn runtime_stats(&self) -> RuntimeStats {
        let (encode, submit, wait, kernel, fused_calls, fused_experts) =
            logan_metal::metal_profile();
        let mio = logan_metal::mio_stats();
        RuntimeStats {
            route_ms: self.spans.route_ms,
            io_ms: self.spans.io_ms,
            shared_ms: self.spans.shared_ms,
            gpu_ms: self.spans.gpu_ms,
            fill_ms: self.spans.fill_ms,
            gdn_ms: self.spans.gdn_ms,
            attn_ms: self.spans.attn_ms,
            hc_ms: self.spans.hc_ms,
            head_ms: self.spans.head_ms,
            gdn_metal_ok: self.spans.gdn_metal_ok,
            expert_hits: self.expert_store.hits,
            expert_misses: self.expert_store.misses,
            expert_evictions: self.expert_store.evictions,
            expert_resident: self.expert_store.len(),
            expert_capacity: cache_cap(),
            metal_encode_ns: encode,
            metal_submit_ns: submit,
            metal_wait_ns: wait,
            metal_kernel_ns: kernel,
            fused_calls,
            fused_experts,
            mio_loads: mio.loads,
            mio_bytes: mio.bytes,
            mio_waits: mio.waits,
            mio_fails: mio.fails,
            mio_prefetch_loads: mio.prefetch_loads,
            mio_prefetch_used: mio.prefetch_used,
            mio_prefetch_wasted: mio.prefetch_wasted,
            mio_outstanding: mio.outstanding,
            mio_peak_outstanding: mio.peak_outstanding,
            mio_latency_samples: mio.latency_samples,
            mio_total_latency_s: mio.total_latency_s,
            context_limit: self.cfg.max_t,
            vocab_size: self.cfg.vocab,
            eos_token_id: self.cfg.eos,
            features: RuntimeFeatures {
                metal_direct: self.metal_direct,
                metal_overlap: self.metal_overlap,
                bnns_bf16: env_bool("QWEN_BNNS_BF16", false),
                gdn_metal: env_bool("QWEN_GDN_METAL", true)
                    && self.cfg.output_gate == crate::OutputGate::Silu,
                gdn_single_copy: env_bool("QWEN_GDN_SINGLE_COPY", true),
                attn_metal: env_bool("QWEN_ATTN_METAL", true),
                qsa_index_metal: env_bool("QWEN_QSA_INDEX_METAL", true),
                shared_io_overlap: env_bool("QWEN_SHARED_IO_OVERLAP", true),
                prefix_cache: super::auto_prefix_cache_enabled(),
                prefix_cache_write: env_bool("QWEN_PREFIX_CACHE_WRITE", true),
            },
        }
    }

    pub fn context_limit(&self) -> usize {
        self.cfg.max_t
    }

    pub fn eos_token_id(&self) -> i64 {
        self.cfg.eos
    }
}
