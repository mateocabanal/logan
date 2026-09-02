//! The plan artifact: graph + representation + tiered resource plan,
//! serialized beside the `.coli` package. `logan compile` emits it; `logan run`
//! executes it. Versioned and fingerprint-pinned so a stale plan is rejected
//! pre-load.

use serde::{Deserialize, Serialize};

use crate::{graph::Graph, optimizer::ParetoPlan, resource::ResourcePlan};

/// Artifact format version. Bump on any breaking change to the schema.
pub const PLAN_ARTIFACT_VERSION: u32 = 4;

/// Transitional compatibility projection for runtimes that have not migrated
/// to `ResourcePlan` yet.
///
/// This enum is intentionally no longer the authoritative capacity model:
/// `Gpu` conflates execution visibility with storage and `Streamed` conflates
/// backing with residency. New optimizer/resource decisions live in
/// `MemoryPlan::resources`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Placement {
    Resident,
    Streamed,
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

/// Physical resource plan replayable by the runtime without re-running the
/// optimizer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPlan {
    /// Transitional value name -> legacy placement projection.
    pub placement: Vec<(String, Placement)>,
    /// value name -> quant representation.
    pub quant: Vec<(String, QuantSpec)>,
    /// value name -> physical execution layout id (0 means canonical/none).
    pub layout: Vec<(String, u16)>,
    /// value/state name -> authoritative backing/residency/access contract.
    pub resources: Vec<(String, ResourcePlan)>,
    /// Fast resident-pool target/budget the plan was optimized against.
    /// This is not a whole-model capacity ceiling.
    pub ram_budget_bytes: u64,
}

impl MemoryPlan {
    pub fn resource(&self, name: &str) -> Option<&ResourcePlan> {
        self.resources
            .iter()
            .find_map(|(candidate, plan)| (candidate == name).then_some(plan))
    }

    pub fn validate_resources(&self) -> Result<(), String> {
        for (name, resource) in &self.resources {
            resource
                .validate()
                .map_err(|detail| format!("resource `{name}`: {detail}"))?;
        }
        Ok(())
    }
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
    /// The selected Pareto point, including per-group reasoning. `None` for
    /// legacy/manual plans that did not run the optimizer.
    pub optimizer: Option<ParetoPlan>,
}

impl PlanArtifact {
    pub fn new(package_fingerprint: String, graph: Graph, memory: MemoryPlan) -> PlanArtifact {
        PlanArtifact {
            version: PLAN_ARTIFACT_VERSION,
            package_fingerprint,
            graph,
            memory,
            optimizer: None,
        }
    }

    pub fn with_optimizer(mut self, optimizer: ParetoPlan) -> PlanArtifact {
        self.optimizer = Some(optimizer);
        self
    }

    /// Serialize to a compact binary form (bincode-style framing via serde).
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        self.memory.validate_resources()?;
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
        plan.memory.validate_resources()?;
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
    use crate::{
        graph::{Op, ValueType},
        resource::{MemoryPoolId, ResourcePlan, StoragePoolId},
    };
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
            layout: vec![("layers.0.mlp.gate.weight".into(), 0x0103)],
            resources: vec![
                (
                    "layers.0.mlp.gate.weight".into(),
                    ResourcePlan::immutable_package(
                        4096,
                        MemoryPoolId::new("uma0"),
                        512,
                        1024,
                        4096,
                    ),
                ),
                (
                    "kv".into(),
                    ResourcePlan::mutable_file_backed(
                        16 * 1024,
                        StoragePoolId::new("ssd0"),
                        MemoryPoolId::new("uma0"),
                        1024,
                        4 * 1024,
                        4096,
                        12 * 1024,
                        256,
                    ),
                ),
            ],
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
        assert_eq!(back.version, 4);
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
    fn resource_axes_round_trip_independently_from_legacy_placement() {
        let plan = sample_plan();
        let bytes = plan.to_bytes().unwrap();
        let back = PlanArtifact::from_bytes(&bytes).unwrap();
        assert_eq!(back.memory.placement.len(), 2);
        assert_eq!(back.memory.quant[0].1.kind, "mxfp4-tile8x32");
        assert_eq!(back.memory.layout[0].1, 0x0103);
        let kv = back.memory.resource("kv").unwrap();
        assert_eq!(kv.backing.bytes, 16 * 1024);
        assert_eq!(kv.residency.target_resident_bytes, 4 * 1024);
        assert_eq!(kv.mutable_backing_bytes(), 16 * 1024);
        assert!(back.optimizer.is_none());
    }
}
