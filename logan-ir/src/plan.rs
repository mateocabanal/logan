//! The plan artifact: graph + placement + quant + memory plan, serialized
//! beside the `.coli` package. `logan compile` emits it; `logan run`
//! executes it. Versioned and fingerprint-pinned so a stale plan is
//! rejected pre-load (the compiler already rejects mismatches pre-emission).

use serde::{Deserialize, Serialize};

use crate::graph::Graph;

/// Artifact format version. Bump on any breaking change to the schema.
pub const PLAN_ARTIFACT_VERSION: u32 = 1;

/// Where a weight lives at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Placement {
    /// Resident in RAM (dense weights, small tensors).
    Resident,
    /// Streamed from disk on demand (experts, PLE shards).
    Streamed,
    /// GPU-visible zero-copy (page-aligned, wrapped by Metal).
    Gpu,
}

/// Quantization applied to a weight by the compiler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantSpec {
    /// "exact" (bf16/f32 passthrough), "mxfp4-tile8x32" (Apple8),
    /// "i4-g32", "f8-e4m3".
    pub kind: String,
    /// Per-tensor or per-block scale metadata (opaque to the core; the
    /// engine's decode path interprets it).
    pub scale: Option<String>,
}

/// Memory plan: the compiler's placement decisions, replayable by the
/// runtime without re-running the planner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPlan {
    /// value name -> placement
    pub placement: Vec<(String, Placement)>,
    /// value name -> quant
    pub quant: Vec<(String, QuantSpec)>,
    /// Peak resident budget the plan was built for (bytes).
    pub ram_budget_bytes: u64,
}

/// The full plan artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanArtifact {
    pub version: u32,
    /// Fingerprint of the source package this plan was compiled from
    /// (rejects stale plans pre-load).
    pub package_fingerprint: String,
    pub graph: Graph,
    pub memory: MemoryPlan,
}

impl PlanArtifact {
    pub fn new(package_fingerprint: String, graph: Graph, memory: MemoryPlan) -> PlanArtifact {
        PlanArtifact {
            version: PLAN_ARTIFACT_VERSION,
            package_fingerprint,
            graph,
            memory,
        }
    }

    /// Serialize to a compact binary form (bincode-style framing via serde).
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| format!("plan serialize: {e}"))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<PlanArtifact, String> {
        let plan: PlanArtifact =
            bincode::deserialize(bytes).map_err(|e| format!("plan deserialize: {e}"))?;
        if plan.version != PLAN_ARTIFACT_VERSION {
            return Err(format!(
                "plan artifact version {} != supported {}",
                plan.version, PLAN_ARTIFACT_VERSION
            ));
        }
        Ok(plan)
    }

    /// Reject a plan whose package fingerprint doesn't match the package
    /// being run (stale-plan guard, mirrors the compiler's pre-emission
    /// fingerprint pin).
    pub fn check_fingerprint(&self, package_fingerprint: &str) -> Result<(), String> {
        if self.package_fingerprint != package_fingerprint {
            return Err(format!(
                "plan fingerprint {} != package {} (stale plan?)",
                self.package_fingerprint, package_fingerprint
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Op, ValueType};
    use std::collections::BTreeMap;

    fn sample_plan() -> PlanArtifact {
        let mut g = crate::graph::Graph::new();
        let x = g.add_value(
            ValueType {
                shape: vec![1, 64],
                dtype: "f32".into(),
            },
            Some("input".into()),
        );
        let w = g.add_value(
            ValueType {
                shape: vec![64, 64],
                dtype: "bf16".into(),
            },
            Some("layers.0.mlp.gate.weight".into()),
        );
        let y = g.add_value(
            ValueType {
                shape: vec![1, 64],
                dtype: "f32".into(),
            },
            None,
        );
        g.add_node(Op::MatMul, vec![x, w], vec![y], BTreeMap::new());
        g.inputs = vec![x];
        g.outputs = vec![y];
        let memory = MemoryPlan {
            placement: vec![
                ("layers.0.mlp.gate.weight".into(), Placement::Streamed),
                ("input".into(), Placement::Resident),
            ],
            quant: vec![(
                "layers.0.mlp.gate.weight".into(),
                QuantSpec {
                    kind: "mxfp4-tile8x32".into(),
                    scale: None,
                },
            )],
            ram_budget_bytes: 4 * 1024 * 1024 * 1024,
        };
        PlanArtifact::new("abc123".into(), g, memory)
    }

    #[test]
    fn round_trip_bytes() {
        let plan = sample_plan();
        let bytes = plan.to_bytes().unwrap();
        let back = PlanArtifact::from_bytes(&bytes).unwrap();
        assert_eq!(plan, back);
    }

    #[test]
    fn fingerprint_guard() {
        let plan = sample_plan();
        assert!(plan.check_fingerprint("abc123").is_ok());
        assert!(plan.check_fingerprint("other").is_err());
    }

    #[test]
    fn version_guard() {
        let plan = sample_plan();
        let mut bytes = plan.to_bytes().unwrap();
        // corrupt the version field (first 4 bytes, little-endian u32)
        bytes[0] = 99;
        assert!(PlanArtifact::from_bytes(&bytes).is_err());
    }

    #[test]
    fn placement_and_quant_round_trip() {
        let plan = sample_plan();
        let bytes = plan.to_bytes().unwrap();
        let back = PlanArtifact::from_bytes(&bytes).unwrap();
        assert_eq!(back.memory.placement.len(), 2);
        assert_eq!(back.memory.quant[0].1.kind, "mxfp4-tile8x32");
    }
}
