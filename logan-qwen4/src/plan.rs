//! Qwen4 scheduler plan: package representation/backend/layout choices
//! pre-resolved ONCE from the validated package (issue #53 engine boundary).
//!
//! The planned representation for a routed expert is the raw Apple8 MXFP4
//! MetalIO record (math 0x20, wc 0) — the same record the direct path
//! streams. This module resolves the record identity, stream regions, and
//! dims for every (layer, expert) at startup. The scheduler runtime never
//! re-discovers layout (`expert_records().first()` lives only here).

use logan_format::package::Package;

pub mod prefix_cache;
pub mod prefix_runtime;
pub mod runtime_stats;
pub mod snapshot;
pub use prefix_cache::{
    digest_hex, live_prefix_state_digest, CacheRestoreStats, CacheWriteStats, PrefixCacheKey,
    PrefixCacheStore,
};
pub use prefix_runtime::{auto_prefix_cache_enabled, run_greedy_cached_coli};
pub use runtime_stats::{RuntimeFeatures, RuntimeStats};
pub use snapshot::QwenStateSnapshot;

/// One planned expert load: the exact record identity + absolute stream
/// regions the validated plan resolved. The runtime performs the MetalIO
/// load from this alone — no manifest/record lookups, no layout math at miss time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedExpert {
    pub shard_id: u32,
    /// Absolute file regions (offset, bytes) for [gate, up, down] — the
    /// same regions `Package::expert_matrix_regions` produces.
    pub regions: [(u64, usize); 3],
    pub dims: [(usize, usize); 3],
}

impl PlannedExpert {
    /// Bytes the MetalIO slot must hold (gate + up + down packed).
    pub fn slot_bytes(&self) -> usize {
        self.regions.iter().map(|r| r.1).sum()
    }
}

/// Per-run plan: the complete routed-expert footprint of the package,
/// pre-resolved once. Also the residency budget source.
#[derive(Clone, Debug)]
pub struct Plan {
    pub layers: Vec<Vec<PlannedExpert>>,
    /// Largest single-expert slot, for the residency pool budget.
    pub max_slot_bytes: usize,
}

impl Plan {
    /// Resolve every (layer, expert) against the validated package. `layers`
    /// and `experts` come from the loaded config; a package that cannot
    /// provide the planned Apple8 representation for any expert is rejected
    /// up front (the scheduled path is MetalIO-only).
    pub fn resolve(pkg: &Package, layers: usize, experts: usize) -> Result<Plan, String> {
        let mut max_slot_bytes = 0usize;
        let mut out = Vec::with_capacity(layers);
        for l in 0..layers {
            let mut layer = Vec::with_capacity(experts);
            for e in 0..experts {
                let recs = pkg.expert_records(l as i32, e as i32);
                let rec = recs
                    .first()
                    .ok_or_else(|| format!("planned expert ({l},{e}) has no record"))?;
                let (regions, dims) = pkg
                    .expert_matrix_regions(rec)
                    .ok_or_else(|| format!("planned expert ({l},{e}) is not raw Apple8 MetalIO"))?;
                if regions.len() < 3 || dims.len() < 3 {
                    return Err(format!("planned expert ({l},{e}) has malformed regions"));
                }
                let planned = PlannedExpert {
                    shard_id: rec.shard_id,
                    regions: [regions[0], regions[1], regions[2]],
                    dims: [dims[0], dims[1], dims[2]],
                };
                max_slot_bytes = max_slot_bytes.max(planned.slot_bytes());
                layer.push(planned);
            }
            out.push(layer);
        }
        Ok(Plan {
            layers: out,
            max_slot_bytes,
        })
    }

    /// Total planned expert bytes (the logical residency ceiling: the view
    /// never evicts, so the pool budget must admit any subset of the plan).
    pub fn total_bytes(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|layer| layer.iter())
            .map(|planned| planned.slot_bytes())
            .sum()
    }

    pub fn planned(&self, layer: usize, expert: usize) -> Option<&PlannedExpert> {
        self.layers.get(layer).and_then(|l| l.get(expert))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_bytes_sums_regions() {
        let planned = PlannedExpert {
            shard_id: 3,
            regions: [(10, 128), (200, 256), (1024, 64)],
            dims: [(1, 1); 3],
        };
        assert_eq!(planned.slot_bytes(), 128 + 256 + 64);
    }

    #[test]
    fn total_bytes_and_max_cover_the_plan() {
        let plan = Plan {
            layers: vec![
                vec![
                    PlannedExpert { shard_id: 0, regions: [(0, 100), (0, 100), (0, 100)], dims: [(1, 1); 3] },
                    PlannedExpert { shard_id: 1, regions: [(0, 50), (0, 50), (0, 50)], dims: [(1, 1); 3] },
                ],
                vec![
                    PlannedExpert { shard_id: 2, regions: [(0, 400), (0, 400), (0, 400)], dims: [(1, 1); 3] },
                ],
            ],
            max_slot_bytes: 1200,
        };
        assert_eq!(plan.total_bytes(), 300 + 150 + 1200);
        assert_eq!(plan.max_slot_bytes, 1200);
        assert_eq!(plan.planned(1, 0).unwrap().shard_id, 2);
        assert!(plan.planned(2, 0).is_none());
    }
}
