# Logan Neutral Backend Design

**Date:** 2026-08-28
**Status:** Approved (architecture) — implementation in progress
**Author:** Mateo + Hermes

## The thesis

Logan is the mutant compiler: it *mutates* model checkpoints into a form
optimized for the hardware they run on. Today that mutation ends at the
`.coli` package (weights only). This design extends it one step further:

> **`logan compile` emits a plan artifact — the tensor graph + placement +
> quant decisions — and `logan run` executes that graph.**

The compiler and the runtime speak the same language. The runtime can also
build the graph itself from a package when no plan exists (ad-hoc runs, the
current behavior). This is the mini-LLVM shape: compiler = frontend +
optimizer, runtime = IR executor.

## Design decisions (approved)

1. **Scope: runtime core + compiler core.** Both the runtime (weight source,
   expert store/residency, MetalIO+Metal backend, math primitives, telemetry)
   and the compiler (IR, planner, quant) become engine-neutral.
2. **Core owns the decode loop.** Engines provide per-layer compute via a
   trait; the core handles scheduling, prefetch, residency, GPU overlap, and
   telemetry. "Any change benefits all engines" is structurally true.
3. **Full tensor graph IR.** Engines declare their model as a graph; the core
   schedules and executes it. The compiler's existing semantic IR
   (`SemanticModel`/`ModelGeometry`) and `StoragePlan` evolve into this graph.
4. **Compiler plan artifact is the runtime's input.** `logan compile` emits
   graph + placement + quant; `logan run` executes it. Runtime falls back to
   building the graph itself when no plan exists.
5. **DeepSeek-V4 is the third engine** (`logan-v4`), ported from the C v4
   engine on top of the neutral core — the proof that the abstractions hold
   for an engine sharing nothing with qwen/qwen4 (different attention, quant
   family, speculative loop).

## Crate layout

```
logan/ (workspace)
├── logan-ir        NEW  — tensor graph IR + plan artifact (serde)
├── logan-core      NEW  — runtime core: storage, expert store (LRU +
│                          residency), decode loop, telemetry, math primitives
├── logan-metal     NEW  — Metal/MetalIO backend (kernels, device, streaming);
│                          engine-neutral; CPU-only fallback when absent
├── logan-compiler  (exists) — frontends → IR → planner → quant → package+plan
│   └── frontends/  deepseek_v4, qwen_moe, qwen4 (model → IR)
├── logan-qwen      (exists, thins out) — graph builder + engine-specific ops
├── logan-qwen4     (exists, thins out) — graph builder + engine-specific ops
├── logan-v4        NEW  — DeepSeek-V4 engine (the neutrality proof)
└── logan-cli       NEW  — the `logan` binary: compile / run / inspect
```

## Neutral vs engine-specific

| Core (neutral) | Engine (specific) |
|---|---|
| Tensor graph IR + plan artifact | Model definition: which ops, shapes, order |
| Storage: resident / streamed / GPU placement | Attention variants (GQA, MLA, QSA, GDN) |
| Expert store: slots, LRU, residency policy, prefetch | Router semantics (mostly generic already) |
| Decode loop: scheduling, overlap, telemetry | PLE / n-gram (qwen4) |
| MetalIO async streaming + Metal kernels | MTP / speculative heads (becomes a core feature) |
| Math primitives: rmsnorm, silu, softmax, rope, bf16/fp8 decode | — |
| Speculative decoding (MTP/DSpark as a core wrapper) | — |

The C engine already proved this shape: `expert_store`, `metalio`,
`backend_metal`, and adaptive residency are shared across qwen_moe and
deepseek_v4. The Rust rewrite never extracted them. This design does, and
adds the graph IR on top.

## The IR (logan-ir)

```rust
pub struct Graph {
    pub nodes: Vec<Node>,
    pub inputs: Vec<Port>,   // model inputs (token ids)
    pub outputs: Vec<Port>, // logits
}

pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
    pub attrs: Attrs, // shapes, quant, placement hints
}

pub enum Op {
    // storage/IO
    LoadTensor(TensorRef),      // resident or streamed
    // compute (neutral set)
    MatMul, RmsNorm, Silu, Softmax, RoPE, Add, Mul,
    // attention
    Attention(AttentionKind),   // GQA, MLA, QSA, GDN
    // MoE
    Router, ExpertMatMul, ExpertReduce,
    // extension (engine-specific, opaque to the core scheduler but with
    // declared data dependencies)
    Extension(ExtensionId),
}
```

**Plan artifact** = Graph + placement (which tensors resident/streamed/GPU) +
quant decisions + memory plan. Serialized (serde + bincode or similar) as
`plan.logan` beside the `.coli` package.

**Extension ops** are the escape hatch: an engine can ship an op the core
doesn't know, as long as it declares its data dependencies. The core
schedules around it; the engine executes it. This keeps the neutral set
small and the core honest.

## Core components (logan-core)

- **Storage**: tensor handles (TensorId + shape + dtype + storage kind).
  Resident (RAM), streamed (disk→GPU via MetalIO), GPU (zero-copy wrapped).
- **Expert store**: slot pool (MetalIO), LRU cache, residency policy
  (C's adaptive residency port), prefetch (C measured prefetch-on-pread as a
  loss; MetalIO async prefetch is the only variant worth having).
- **Decode loop**: owns the per-token schedule: route → issue expert loads →
  overlap CPU/GPU → reduce → next layer. Telemetry spans around each phase.
- **Telemetry**: per-token `route/io/shared/gpu/fill_ms` + Metal profile
  counters + MetalIO stats. Gated by `LOGAN_PROFILE=1`. This is what makes
  A/B verdicts trustworthy (regime-independent metrics, not wall time).
- **Math primitives**: rmsnorm, silu, softmax, rope, bf16/fp8 decode,
  matmul (NEON BF16, scalar fallback).

## Metal backend (logan-metal)

The three C layers, engine-neutral:
- `backend_metal.mm` — quantized GEMV (fmt 7 MXFP4/Apple8), small ops
- `metalio.mm` — async NVMe→MTLBuffer streaming, slot pool
- `apple8_metalio_direct.mm` — fused expert execution + coalesced GDN kernels

All entry points decline cleanly to CPU when unavailable (the C contract).
Non-macOS builds get stub decliners. The GDN kernels are qwen-specific today
— they move to the core as "coalesced BF16 matmul + recurrence" primitives,
with the qwen4 engine providing the recurrence semantics.

## Engines

- **logan-qwen** (1k lines): simplest — proves loop + storage + expert store.
- **logan-qwen4** (3.4k lines): proves Metal integration + exotic ops
  (GDN, QSA, PLE).
- **logan-v4** (new): proves full neutrality — different attention (sparse +
  MLA-style), different quant family (MXFP4 fmt7 + fp8), speculative loop
  (MTP/DSpark as a core wrapper).

## Data flow

```
checkpoint (HF safetensors)
    │  logan compile
    ▼
.coli package (weights)  +  plan.logan (graph + placement + quant)
    │  logan run
    ▼
logan-core loads plan → builds storage → decode loop executes graph
    (or: runtime builds the graph itself from the package when no plan)
```

## Error handling

- Metal/MetalIO: every entry point declines (returns 0/None) → CPU fallback.
  Post-submit GPU failure (rc<0) is fatal per the C contract (state may have
  advanced).
- Storage: missing tensor / shape mismatch → typed errors at graph build
  time, not mid-decode.
- Plan artifact: versioned; a stale/mismatched plan (fingerprint vs package)
  is rejected pre-emission (the compiler already does this) and pre-load.

## Testing strategy

- **Token-identity gates** (the existing ref.json gates) stay the ground
  truth for every refactor slice: qwen, qwen4, and v4 must produce
  byte-identical output before/after each extraction.
- **Cross-path identity**: Metal vs CPU-fallback on the real package
  (already proven for qwen4; becomes a CI-able gate).
- **IR round-trip**: graph → serialize → deserialize → identical execution.
- **C-oracle differentials** (rANS/Apple8) stay skip-guarded when the C tree
  is absent.

## Migration path (slices)

1. **logan-ir** — graph IR types + plan artifact serialization + round-trip
   tests. New code, zero risk to existing crates.
2. **logan-core** — extract neutral pieces from qwen4 (math primitives,
   storage abstraction, expert store) into the new crate; qwen4 depends on
   it. Token-identity gate keeps it honest.
3. **logan-metal** — move the Metal FFI + metalio + apple8-direct into a
   neutral crate.
4. **Engine thinning** — qwen4 and qwen build graphs on the core.
5. **logan-v4** — DeepSeek-V4 engine on the core (the proof).
6. **logan-cli** — unified binary: compile / run / inspect.

Each slice ends green: `cargo test --workspace --all-targets` + the tiny
gates.
