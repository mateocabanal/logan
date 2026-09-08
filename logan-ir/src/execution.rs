//! Backend execution placement for heterogeneous Logan plans.
//!
//! Storage placement and execution placement are deliberately separate. On
//! Apple Silicon the CPU, Metal GPU, and ANE can all touch the same UMA-backed
//! allocation, while a graph node still has exactly one chosen execution
//! backend. `ExecutionPlan` groups adjacent graph nodes into backend "islands"
//! so transfers/synchronization happen at island boundaries rather than after
//! every primitive op.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::{Graph, NodeId, ValueId};

/// Runtime execution backend. `Auto` is a compiler/runtime negotiation hint,
/// never a concrete device submission target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionBackend {
    Auto,
    Cpu,
    Metal,
    Ane,
}

/// How a value crosses an execution-island boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransferKind {
    /// Same backend and same physical allocation; no transfer or handoff.
    Alias,
    /// Shared physical allocation visible to both devices (e.g. IOSurface on
    /// Apple Silicon). A synchronization edge may still be required.
    SharedMemory,
    /// Explicit host-side copy between allocations.
    HostCopy,
    /// Data is materialized from backing storage rather than from the producer
    /// device's allocation (e.g. SSD-backed weight streaming).
    Streamed,
}

pub type IslandId = u32;

/// A compiler-selected contiguous region intended to execute as one backend
/// unit. Backends are free to fuse nodes inside an island into one native
/// program/command buffer (for ANE, typically one MIL program).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionIsland {
    pub id: IslandId,
    pub backend: ExecutionBackend,
    pub nodes: Vec<NodeId>,
    /// Values entering from graph inputs or another island.
    pub inputs: Vec<ValueId>,
    /// Values consumed outside this island or exported as graph outputs.
    pub outputs: Vec<ValueId>,
    /// Fixed-shape islands are eligible for ahead-of-time compilation/caching
    /// by backends such as ANE.
    pub fixed_shape: bool,
    /// Optional stable key used to cache a backend-native compiled artifact.
    pub cache_key: Option<String>,
    /// Human-readable planner rationale for diagnostics/dashboard display.
    pub rationale: Option<String>,
}

/// One value handoff between two islands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionEdge {
    pub from: IslandId,
    pub to: IslandId,
    pub value: ValueId,
    pub transfer: TransferKind,
}

/// Heterogeneous execution overlay for a graph.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub islands: Vec<ExecutionIsland>,
    pub edges: Vec<ExecutionEdge>,
}

impl ExecutionPlan {
    pub fn backend_for_node(&self, node: NodeId) -> Option<ExecutionBackend> {
        self.islands
            .iter()
            .find(|island| island.nodes.contains(&node))
            .map(|island| island.backend)
    }

    pub fn island(&self, id: IslandId) -> Option<&ExecutionIsland> {
        self.islands.iter().find(|island| island.id == id)
    }

    /// Validate graph references and the one-node/one-island ownership rule.
    /// This intentionally does not require every graph node to be placed yet;
    /// partial plans are useful while bringing up a new backend.
    pub fn validate(&self, graph: &Graph) -> Result<(), String> {
        let mut island_ids = BTreeSet::new();
        let mut owner = BTreeMap::<NodeId, IslandId>::new();
        for island in &self.islands {
            if !island_ids.insert(island.id) {
                return Err(format!("duplicate execution island id {}", island.id));
            }
            if island.backend == ExecutionBackend::Auto {
                return Err(format!(
                    "execution island {} has non-concrete Auto backend",
                    island.id
                ));
            }
            if island.nodes.is_empty() {
                return Err(format!("execution island {} has no nodes", island.id));
            }
            for &node in &island.nodes {
                if node as usize >= graph.nodes.len() {
                    return Err(format!(
                        "execution island {} references node {node} outside graph",
                        island.id
                    ));
                }
                if let Some(previous) = owner.insert(node, island.id) {
                    return Err(format!(
                        "graph node {node} appears in execution islands {previous} and {}",
                        island.id
                    ));
                }
            }
            for (&value, role) in island
                .inputs
                .iter()
                .map(|v| (v, "input"))
                .chain(island.outputs.iter().map(|v| (v, "output")))
            {
                if value as usize >= graph.values.len() {
                    return Err(format!(
                        "execution island {} {role} value {value} outside graph",
                        island.id
                    ));
                }
            }
        }

        for edge in &self.edges {
            if edge.from == edge.to {
                return Err(format!(
                    "execution edge for value {} is a self-edge on island {}",
                    edge.value, edge.from
                ));
            }
            if !island_ids.contains(&edge.from) || !island_ids.contains(&edge.to) {
                return Err(format!(
                    "execution edge {} -> {} references unknown island",
                    edge.from, edge.to
                ));
            }
            if edge.value as usize >= graph.values.len() {
                return Err(format!(
                    "execution edge references value {} outside graph",
                    edge.value
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Op, ValueType};
    use std::collections::BTreeMap;

    fn graph() -> Graph {
        let mut g = Graph::new();
        let x = g.add_value(
            ValueType {
                shape: vec![1, 64],
                dtype: "f32".into(),
            },
            Some("x".into()),
        );
        let a = g.add_value(
            ValueType {
                shape: vec![1, 64],
                dtype: "f32".into(),
            },
            None,
        );
        let y = g.add_value(
            ValueType {
                shape: vec![1, 64],
                dtype: "f32".into(),
            },
            None,
        );
        g.add_node(Op::RmsNorm, vec![x], vec![a], BTreeMap::new());
        g.add_node(Op::Silu, vec![a], vec![y], BTreeMap::new());
        g.inputs = vec![x];
        g.outputs = vec![y];
        g
    }

    #[test]
    fn ane_island_can_feed_metal_through_shared_memory() {
        let g = graph();
        let plan = ExecutionPlan {
            islands: vec![
                ExecutionIsland {
                    id: 0,
                    backend: ExecutionBackend::Ane,
                    nodes: vec![0],
                    inputs: vec![0],
                    outputs: vec![1],
                    fixed_shape: true,
                    cache_key: Some("norm-1x64".into()),
                    rationale: Some("fixed dense island".into()),
                },
                ExecutionIsland {
                    id: 1,
                    backend: ExecutionBackend::Metal,
                    nodes: vec![1],
                    inputs: vec![1],
                    outputs: vec![2],
                    fixed_shape: true,
                    cache_key: None,
                    rationale: None,
                },
            ],
            edges: vec![ExecutionEdge {
                from: 0,
                to: 1,
                value: 1,
                transfer: TransferKind::SharedMemory,
            }],
        };
        plan.validate(&g).unwrap();
        assert_eq!(plan.backend_for_node(0), Some(ExecutionBackend::Ane));
        assert_eq!(plan.backend_for_node(1), Some(ExecutionBackend::Metal));
    }

    #[test]
    fn rejects_duplicate_node_ownership() {
        let g = graph();
        let island = |id, backend| ExecutionIsland {
            id,
            backend,
            nodes: vec![0],
            inputs: vec![0],
            outputs: vec![1],
            fixed_shape: true,
            cache_key: None,
            rationale: None,
        };
        let plan = ExecutionPlan {
            islands: vec![
                island(0, ExecutionBackend::Ane),
                island(1, ExecutionBackend::Metal),
            ],
            edges: vec![],
        };
        assert!(plan.validate(&g).is_err());
    }
}
