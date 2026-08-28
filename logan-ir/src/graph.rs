//! The tensor graph IR: nodes, ops, values, and the graph container.
//!
//! Neutral by construction: the op set is the intersection of what LLM
//! engines do (matmul, norms, attention kinds, MoE routing), with an
//! `Extension` escape hatch for engine-specific ops that declare their data
//! dependencies so the core scheduler can route around them.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Opaque node identifier (index into `Graph::nodes`).
pub type NodeId = u32;

/// Opaque value identifier (index into `Graph::values`).
pub type ValueId = u32;

/// A value flowing between nodes: a tensor (or scalar) with a declared type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Value {
    pub id: ValueId,
    pub ty: ValueType,
    /// Stable semantic name (e.g. "layers.0.attn.q_proj.weight") when the
    /// value is a model weight; empty for intermediates.
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueType {
    pub shape: Vec<u64>,
    /// "f32", "bf16", "f8-e4m3", "mxfp4-tile8x32", "i4-g32", "i32", "u32", "i64"
    pub dtype: String,
}

/// Attention variants the core knows how to schedule. The engine supplies
/// the per-variant compute; the core supplies the KV cache, position
/// handling, and (for GDN) the recurrent-state lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionKind {
    Gqa,
    Mla,
    Qsa,
    Gdn,
}

/// The neutral op set. `Extension` is the escape hatch: an engine-specific
/// op the core doesn't know, executed by the engine, scheduled by the core
/// through its declared data dependencies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    // ---- storage / IO -----------------------------------------------------
    /// Materialize a weight into its placement (resident / streamed / GPU).
    LoadWeight,
    /// Read a streamed row range (e.g. PLE n-gram rows) into a value.
    ReadRange,

    // ---- compute (neutral) -------------------------------------------------
    MatMul,
    RmsNorm,
    Silu,
    Softmax,
    RoPE,
    Add,
    Mul,
    /// y = silu(gate(x)) * up(x); down(y) — the MoE expert block.
    ExpertBlock,
    /// Router: logits -> top-k indices + weights (engine supplies the
    /// selection policy; core supplies the renormalization contract).
    Router,

    // ---- attention ---------------------------------------------------------
    Attention(AttentionKind),

    // ---- engine extension --------------------------------------------------
    Extension(String),
}

/// A graph node: an op with declared inputs/outputs and attributes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    /// Free-form attributes (shapes, quant hints, placement hints, engine
    /// extension payloads). Kept as a map so the core can pass through
    /// unknown attributes without understanding them.
    pub attrs: BTreeMap<String, String>,
}

/// The tensor graph. Values are the edges; nodes are the ops.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub values: Vec<Value>,
    pub nodes: Vec<Node>,
    /// Model inputs (e.g. token id stream).
    pub inputs: Vec<ValueId>,
    /// Model outputs (logits).
    pub outputs: Vec<ValueId>,
}

impl Graph {
    pub fn new() -> Graph {
        Graph {
            values: Vec::new(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn add_value(&mut self, ty: ValueType, name: Option<String>) -> ValueId {
        let id = self.values.len() as ValueId;
        self.values.push(Value { id, ty, name });
        id
    }

    pub fn add_node(
        &mut self,
        op: Op,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
        attrs: BTreeMap<String, String>,
    ) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(Node {
            id,
            op,
            inputs,
            outputs,
            attrs,
        });
        id
    }

    /// Topological order of node ids (Kahn's algorithm). The core scheduler
    /// executes in this order; extension nodes are scheduled by their
    /// declared dependencies like any other node.
    pub fn topo_order(&self) -> Result<Vec<NodeId>, String> {
        let n = self.nodes.len();
        // Validate value refs first.
        for node in &self.nodes {
            for &v in &node.inputs {
                if (v as usize) >= self.values.len() {
                    return Err(format!("node {}: input value {v} out of range", node.id));
                }
            }
            for &v in &node.outputs {
                if (v as usize) >= self.values.len() {
                    return Err(format!("node {}: output value {v} out of range", node.id));
                }
            }
        }
        // value -> producer node (a value is produced by exactly one node;
        // graph inputs have no producer).
        let mut producer: Vec<Option<NodeId>> = vec![None; self.values.len()];
        for node in &self.nodes {
            for &v in &node.outputs {
                producer[v as usize] = Some(node.id);
            }
        }
        // value -> consumers
        let mut consumers: Vec<Vec<NodeId>> = vec![Vec::new(); self.values.len()];
        for node in &self.nodes {
            for &v in &node.inputs {
                consumers[v as usize].push(node.id);
            }
        }
        // Edge producer(v) -> consumer(v) for every value v that has both.
        // A node consuming its own output (c == p) is a self-loop = cycle;
        // it must NOT be skipped.
        let mut indeg = vec![0usize; n];
        let mut adj: Vec<Vec<NodeId>> = vec![Vec::new(); n];
        for (v, &prod) in producer.iter().enumerate() {
            if let Some(p) = prod {
                for &c in &consumers[v] {
                    adj[p as usize].push(c);
                    indeg[c as usize] += 1;
                }
            }
        }
        // Kahn
        let mut queue: Vec<NodeId> = (0..n)
            .filter(|&i| indeg[i] == 0)
            .map(|i| i as NodeId)
            .collect();
        let mut order = Vec::with_capacity(n);
        while let Some(id) = queue.pop() {
            order.push(id);
            for &c in &adj[id as usize] {
                indeg[c as usize] -= 1;
                if indeg[c as usize] == 0 {
                    queue.push(c);
                }
            }
        }
        if order.len() != n {
            return Err("graph contains a cycle".to_string());
        }
        Ok(order)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_graph() -> Graph {
        let mut g = Graph::new();
        let x = g.add_value(
            ValueType { shape: vec![1, 64], dtype: "f32".into() },
            Some("input".into()),
        );
        let w = g.add_value(
            ValueType { shape: vec![64, 64], dtype: "bf16".into() },
            Some("layers.0.mlp.gate.weight".into()),
        );
        let y = g.add_value(
            ValueType { shape: vec![1, 64], dtype: "f32".into() },
            None,
        );
        g.add_node(
            Op::MatMul,
            vec![x, w],
            vec![y],
            BTreeMap::new(),
        );
        g.inputs = vec![x];
        g.outputs = vec![y];
        g
    }

    #[test]
    fn topo_order_single_node() {
        let g = tiny_graph();
        let order = g.topo_order().unwrap();
        assert_eq!(order, vec![0]);
    }

    #[test]
    fn topo_order_chain() {
        let mut g = Graph::new();
        let a = g.add_value(ValueType { shape: vec![1], dtype: "f32".into() }, None);
        let b = g.add_value(ValueType { shape: vec![1], dtype: "f32".into() }, None);
        let c = g.add_value(ValueType { shape: vec![1], dtype: "f32".into() }, None);
        g.add_node(Op::Silu, vec![a], vec![b], BTreeMap::new());
        g.add_node(Op::Softmax, vec![b], vec![c], BTreeMap::new());
        let order = g.topo_order().unwrap();
        assert_eq!(order, vec![0, 1]);
    }

    #[test]
    fn topo_order_detects_cycle() {
        let mut g = Graph::new();
        let a = g.add_value(ValueType { shape: vec![1], dtype: "f32".into() }, None);
        let b = g.add_value(ValueType { shape: vec![1], dtype: "f32".into() }, None);
        g.add_node(Op::Add, vec![a, b], vec![a], BTreeMap::new()); // a -> a cycle
        assert!(g.topo_order().is_err());
    }

    #[test]
    fn topo_order_rejects_bad_value_refs() {
        let mut g = Graph::new();
        g.add_node(Op::Silu, vec![99], vec![], BTreeMap::new());
        assert!(g.topo_order().is_err());
    }

    #[test]
    fn extension_node_schedules_by_deps() {
        let mut g = Graph::new();
        let a = g.add_value(ValueType { shape: vec![1], dtype: "f32".into() }, None);
        let b = g.add_value(ValueType { shape: vec![1], dtype: "f32".into() }, None);
        let c = g.add_value(ValueType { shape: vec![1], dtype: "f32".into() }, None);
        g.add_node(Op::Extension("qwen4.ple".into()), vec![a], vec![b], BTreeMap::new());
        g.add_node(Op::Add, vec![b, a], vec![c], BTreeMap::new());
        let order = g.topo_order().unwrap();
        assert_eq!(order, vec![0, 1]);
    }
}
