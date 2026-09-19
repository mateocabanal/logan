# Qwen3.8-Flash-Next / Qwen4Exp Runtime Research

**Date:** 2026-09-08
**Scope:** llama.cpp, vLLM, SGLang, and implications for Logan
**Model family name used by runtimes:** `Qwen4Exp` / `qwen4_exp`

## Executive summary

Qwen3.8-Flash-Next is not difficult primarily because of its arithmetic. The hard part is its **heterogeneous persistent state** and storage hierarchy:

- recurrent Gated DeltaNet state,
- GDN convolution history,
- normal attention KV state,
- QSA indexer raw/pending state,
- QSA compressed block state,
- PLE/n-gram convolution history,
- the extremely large (~51B parameter) PLE/n-gram embedding table,
- MoE expert residency,
- prefix-cache serialization / restoration of all of the above.

All three major runtimes independently converged on the same high-level design principle:

> **Qwen3.8 should extend the proven Qwen3.5/Qwen3-Next runtime paths rather than creating a completely separate engine.**

The most useful ideas for Logan are distributed across the projects:

- **llama.cpp:** simple architecture decomposition, four-stream HC layout, reuse of existing GDN/MoE primitives.
- **vLLM:** explicit persistent QSA side-state, true sparse QSA execution, clear recurrent-state ownership.
- **SGLang:** strongest QSA kernel/state implementation and the most relevant PLE storage design for NVMe-backed inference.

The most important Logan-specific finding is that Logan already performs **algorithmically sparse QSA attention** over selected positions, but its **QSA indexer state is not incremental**. It reconstructs historical compressed blocks from raw index-key history repeatedly. This should be the first Qwen3.8 structural optimization.

---

# 1. Qwen3.8-Flash-Next architecture relevant to runtime design

The current model is internally referred to as Qwen4Exp by the major runtimes.

Important architectural properties include:

- 48 transformer/recurrent layers.
- Repeating pattern of roughly `3 × Gated DeltaNet + 1 × Qwen Sparse Attention`.
- Hidden size around 2560.
- Four residual / HyperConnection streams (`hc_count = 4`).
- 512 MoE experts with top-k routing (official configuration uses top-10; custom Logan variants may differ and must respect model metadata).
- QSA compressed-key selection with a budget around 2048 tokens and compression ratio 4.
- PLE/n-gram embedding module with a very large random-access embedding table (~51B parameters).
- Stateful dilated/depthwise convolution inside PLE.
- MTP / draft-layer support in optimized runtimes.

The critical observation is that Qwen3.8 combines several state machines with very different access patterns. A runtime that represents all of them as ordinary resident tensors will either waste memory or destroy performance.

---

# 2. llama.cpp implementation

## 2.1 General architecture

llama.cpp added Qwen4Exp support by reusing existing GGML primitives rather than introducing a completely separate runtime.

The key decomposition is:

- existing delta-net / recurrent machinery for GDN,
- existing MoE machinery,
- existing RoPE machinery,
- Qwen4Exp-specific HyperConnection / gated residual composition,
- Qwen4Exp-specific QSA indexer logic,
- Qwen4Exp-specific PLE/n-gram logic.

This is the correct architectural direction for Logan as well: Qwen3.8 should remain an extension of the Qwen3.5/Qwen-Next execution machinery.

## 2.2 Gated DeltaNet

The GDN path is fundamentally the prior delta-net implementation with the Qwen3.8-specific output gate behavior.

Conceptually:

```text
QKV projection
    ↓
causal depthwise convolution
    ↓
SiLU
    ↓
normalize Q/K
    ↓
recurrent delta-state update
    ↓
RMSNorm
    ↓
sigmoid(z) × normalized output
    ↓
output projection
```

The important Qwen3.8 detail is the **sigmoid output gate** rather than blindly inheriting an older SiLU-gated variant.

## 2.3 Four-stream HyperConnection representation

llama.cpp keeps the widened residual representation alive across layers rather than constantly packing and unpacking it.

Conceptually:

```text
[hidden × 4 HC streams]
        ↓
HC read/mix
        ↓
GDN or QSA
        ↓
HC gated combine
        ↓
HC read/mix
        ↓
MoE
        ↓
HC gated combine
```

Only near the final output does the runtime collapse the four streams.

**Implication for Logan:** HC=4 should remain a first-class physical hidden-state layout.

## 2.4 PLE / n-gram embedding

llama.cpp computes n-gram row IDs from token history and performs row gathers from the giant PLE table. It treats the table lazily rather than eagerly materializing it.

The PLE path is approximately:

```text
n-gram row lookup
    ↓
K and V projections
    ↓
Q/K-like normalized gating computation
    ↓
gated V contribution
    ↓
normalization
    ↓
dilated depthwise convolution
    ↓
SiLU
    ↓
residual update
```

PLE convolution history is persistent model state, not scratch.

## 2.5 QSA weakness in current llama.cpp path

The original/merged llama.cpp QSA design correctly computes top-k index positions, but its attention execution historically built a mask over the full KV cache rather than immediately gathering only selected K/V entries.

That means the **selection is sparse while the memory traffic can remain effectively dense**.

An open gather-based path demonstrates why this matters: compacting selected K/V before attention improves long-context decode significantly.

**Do not copy the masked full-cache design into Logan.**

---

# 3. vLLM implementation

## 3.1 General architecture

vLLM reuses Qwen3.5/Qwen3-Next components wherever possible and adds Qwen4Exp-specific modules around them.

Important decomposition:

- existing optimized GDN implementation,
- existing MoE blocks,
- dedicated gated residual / HC layer,
- dedicated PLE module,
- dedicated QSA indexer,
- dedicated QSA sparse-attention backend,
- MTP support integrated into normal runtime state handling.

## 3.2 QSA persistent side state

This is one of vLLM's strongest design choices.

It keeps separate state for:

```text
normal attention KV cache
+
raw / pending QSA index-key state
+
compressed QSA block-key state
```

A new token updates only the pending/raw side state. Once a compression group completes, the runtime computes the compressed representation once and persists it.

Historical compressed blocks are **not reconstructed every token**.

## 3.3 True sparse QSA execution

The indexer produces selected logical token positions, and the sparse-attention backend consumes those positions directly against the paged KV cache.

Conceptually:

```text
QSA indexer
    ↓
selected logical token IDs
    ↓
sparse paged attention
    ↓
only selected KV rows touched
```

This is the correct execution model for long-context QSA.

## 3.4 PLE state ownership

vLLM treats PLE as a stateful recurrent-style component rather than as a stateless embedding trick.

Persistent PLE state includes the short/dilated convolution history required for exact continuation, batching, prefix caching, swapping, and speculative execution.

## 3.5 PLE CPU offload

vLLM includes a CPU-side/offload path for the giant PLE embedding table.

This is useful evidence that the table must be treated as a separate storage tier, but its process/CPU-RAM-oriented design is not the best fit for Apple Silicon Logan.

For Logan, SGLang's file-backed strategy is more relevant.

---

# 4. SGLang implementation

## 4.1 General architecture

SGLang's active Qwen4Exp work similarly derives from Qwen3.5 components rather than replacing them.

Its Qwen4-specific additions primarily cover:

- gated residual / HC,
- QSA,
- PLE,
- Qwen4-specific model wiring.

## 4.2 Incremental QSA index state

SGLang keeps only the uncompressed pending group and completed compressed state.

For compression ratio 4:

```text
up to 4 raw K rows
    ↓
mean
    ↓
RMSNorm
    ↓
MRoPE
    ↓
persistent compressed-K row
```

Completed historical blocks are never recomputed.

This is exactly the state model Logan should adopt.

## 4.3 Fused QSA maintenance kernels

SGLang fuses operations such as:

- Q normalization + RoPE + state store,
- block mean + normalization + RoPE + compressed-cache store.

This avoids fragmenting the QSA indexer into many tiny device dispatches.

## 4.4 True sparse attention

SGLang uses custom sparse kernels / compacted K/V paths that consume only selected positions.

This confirms the same architecture as vLLM:

> The QSA selector and the attention backend should be designed together.

## 4.5 QSA overlap

SGLang can overlap indexer work with the main QKV path because much of the indexer is independent until attention consumes the selection.

Conceptually:

```text
                   ┌─ main QKV projection ──────┐
hidden ────────────┤                             ├─ sparse attention
                   └─ QSA indexer / top-k ──────┘
```

On Apple Silicon, this maps naturally to separate Metal command buffers / command queues where dependency structure permits it.

## 4.6 PLE file-backed / NVMe design

This is the most relevant PLE implementation for Logan.

SGLang explicitly treats PLE as a storage-hierarchy problem and supports a file-backed mapping strategy with controls intended to prevent the giant embedding from consuming all host memory through page-cache residency.

Important ideas:

- file-backed PLE storage,
- large table remains outside normal model residency,
- bounded working-set behavior,
- prefetch for useful batch sizes,
- page-cache/RSS trimming,
- local fast NVMe as the backing store.

The key lesson is that **mmap alone does not guarantee cold residency**. Random accesses can progressively populate the OS page cache until most of the giant table becomes resident.

Logan should therefore make PLE residency explicit rather than relying entirely on OS mmap behavior.

---

# 5. Logan current-state comparison

This section reflects the Logan tree inspected on 2026-09-08.

## 5.1 What Logan is already doing correctly

### Sparse QSA attention is genuinely sparse

Logan's selected-attention path iterates only over positions returned by the QSA selector. It does not require a full-KV `-inf` mask path like the older llama.cpp implementation.

This means Logan is already algorithmically ahead of that particular llama.cpp weakness.

### PLE remains an NVMe resource

The giant PLE table is accessed with bounded row reads from COLI shards rather than being eagerly loaded into RAM.

This preserves the key invariant:

> **The ~51B PLE table must never become an ordinary resident tensor.**

### HC=4 is represented structurally

The current engine already has the right conceptual direction for the four residual streams.

### Existing QKV/index projection fusion is useful

Current projection fusion should be preserved. The missing QSA work is mostly after the projections.

---

# 6. Primary Logan QSA problem

The current Logan QSA selector stores raw index-key history and repeatedly reconstructs the compressed representation.

Current behavior is approximately:

```text
new raw index K
    ↓
append to complete raw K history
    ↓
for every completed historical block:
    average raw keys
    normalize
    apply RoPE
    materialize compressed block
    ↓
score every block
    ↓
select top blocks
```

This is the wrong state machine.

The compressed index must be **persistent incremental state**, not a derived temporary rebuilt every token.

---

# 7. Proposed Logan QSA state machine

Use approximately:

```rust
struct QsaLayerState {
    /// At most compression_ratio - 1 uncompressed index-K rows.
    pending_k: Vec<f32>,

    /// Completed blocks after mean + norm + RoPE.
    compressed_k: Vec<f32>,

    completed_blocks: usize,
}
```

For compression ratio 4:

```text
token N index-K
    ↓
append to pending ring
    ↓
group complete?
    ├─ no  → keep pending state
    └─ yes → mean four K rows
             ↓
           K norm
             ↓
          MRoPE
             ↓
       append compressed K
             ↓
        clear pending group
```

Then selection becomes:

```text
index Q
   ↓
Q norm + MRoPE
   ↓
dot against persistent compressed K
   ↓
top-k blocks
   ↓
expand each block to token positions
   +
pending/uncompressed tail positions
```

Historical compression work disappears from decode.

---

# 8. Top-k selection problem

The inspected Logan path uses a host-side selection-sort-style top-k implementation.

With approximately:

- compression ratio = 4,
- sparse token budget ≈ 2048,
- top blocks ≈ 512,

selection work grows roughly like:

```text
O(number_of_blocks × 512)
```

per QSA layer per token.

This should be removed.

Recommended progression:

1. **Deterministic bounded heap / partial selection on CPU** while preserving exact tie-breaking.
2. Metal score + top-k kernel.
3. Fuse score/top-k more aggressively only if profiling justifies it.

A parity oracle should verify selected block/token ordering exactly against the old implementation before enabling the new path by default.

---

# 9. Metal QSA plan

## Phase A: persistent state + CPU reference selector

First make the state machine correct and incremental without changing numerical behavior.

Keep the current selector as an oracle path.

## Phase B: Metal indexer island

Move to device:

```text
index Q norm + RoPE
pending K state update
completed-block mean
completed-block norm + RoPE
Q × compressed-K scores
top-k
```

Preserve deterministic tie behavior where required.

## Phase C: selected-KV attention on Metal

Do not immediately build the most complex sparse kernel.

Start with:

```text
selected token IDs (~2050)
        ↓
Metal gather K/V
        ↓
compact K/V scratch buffer
        ↓
optimized dense-style attention over compact width
```

Benefits:

- easy parity against CPU sparse attention,
- contiguous memory after gather,
- bounded maximum sparse width,
- straightforward vectorization,
- simpler Metal implementation.

Only later consider a fully fused indexed sparse-attention kernel.

## Phase D: overlap indexer and QKV

Once correctness is stable, issue the indexer work independently from the main QKV path and synchronize only before sparse attention consumes the selected IDs.

---

# 10. PLE store refactor

The current PLE row path keeps the giant table safely on disk, but it performs avoidable metadata work around row reads.

A better long-lived descriptor should be created at model load time:

```rust
struct PleStore {
    shards: Vec<PleShard>,
    rows_per_shard: u64,
    row_bytes: usize,
    scale: f32,
    cache: PleRowCache,
}

struct PleShard {
    payload_offset: u64,
    rows: u64,
    // cached file / MetalIO resource handle
}
```

Hot row access should become approximately:

```rust
ple_store.gather_rows(&row_ids, dst)
```

and should not repeatedly:

- rescan record metadata,
- rebuild shard-name prefixes,
- rediscover rows-per-shard,
- reread a global FP8 scale for every head.

---

# 11. Bounded PLE hot-row cache

The cache should be explicit and small relative to the full table.

Recommended design:

```text
PLE table on NVMe
      ↓
small bounded FP8 row cache in RAM
      ↓
FP8 decode / projection on device
```

Important rule:

> **Cache raw FP8 rows, not decoded F32 rows.**

That maximizes useful rows per RAM byte and keeps the storage invariant obvious.

A reasonable initial default is in the tens of MiB, with the ResidencyManager owning the actual policy.

Telemetry should expose:

- row-cache hits,
- misses,
- hit rate,
- bytes read from NVMe,
- bytes served from RAM cache,
- latency by storage tier,
- evictions.

---

# 12. PLE batched row gathering

Instead of one row read per head in isolation:

```rust
let ids = ple_row_ids(...);
ple_store.gather_rows(&ids, scratch);
```

The store can:

- deduplicate duplicate row IDs,
- group rows by shard,
- sort nearby rows,
- coalesce reads where safe,
- batch MetalIO work,
- scatter results back into canonical head order.

For prompt chunks, compute all row IDs for the chunk and prefetch them as one storage workload.

---

# 13. PLE prefetch opportunity

PLE row IDs depend on **token history**, not hidden activations.

Therefore PLE reads can begin before execution reaches the PLE layer.

Decode:

```text
token admitted
    ├──────── start PLE read ───────────────┐
    ↓                                       │
embedding                                  NVMe
    ↓                                       │
layer 0                                     │
    ↓                                       │
layer 1                                     │
    ↓                                       │
PLE layer ◄────────────── completion ───────┘
```

Prefill:

```text
compute PLE row IDs for entire chunk
    ↓
deduplicate/group/prefetch rows
    ↓
execute earlier layers
    ↓
consume prefetched rows
```

Important implementation rule:

- make row-ID calculation a **pure function**,
- keep causal history advancement separate,
- speculative/prefetch work must not mutate committed token history.

---

# 14. Prefix cache implications

Current QSA prefix snapshots should eventually stop serializing complete raw index-key history.

After the QSA state refactor, continuation requires only:

```text
persistent compressed QSA blocks
+
small pending raw group
```

At compression ratio 4, this reduces QSA snapshot state by roughly the compression ratio before considering lower-precision compressed storage.

The persistent state ABI must be bumped when changing serialized QSA layout.

Prefix-cache parity tests must verify that:

- cold run,
- RAM prefix restore,
- SSD prefix restore,
- cancellation/restart,

all generate identical continuation tokens.

---

# 15. Memory planner correction

While auditing PLE state, a planner mismatch was identified.

PLE convolution runtime state operates over the HC-wide channel dimension, approximately:

```text
hc_count × hidden_size × history
```

The planner should budget that same physical representation rather than deriving the PLE convolution width solely from PLE embedding dimensions.

The planner must describe actual allocated state, not a model-theory approximation.

---

# 16. Recommended implementation order for Logan

## P0 — Incremental QSA compressed state

- pending raw-K ring,
- persistent compressed-K cache,
- no historical recompression,
- exact selector parity tests.

## P0.5 — Replace host selection sort

- deterministic bounded heap / partial top-k,
- exact tie-breaking parity,
- benchmark separately.

## P1 — Update planner and prefix snapshots

- compressed QSA accounting,
- pending-group accounting,
- persistent state ABI bump,
- restore continuation parity tests.

## P2 — Metal QSA selector island

- Q norm + MRoPE,
- pending K update,
- completed block compression,
- compressed-K scoring,
- top-k.

Keep CPU reference implementation available behind a debug/oracle path.

## P3 — Metal selected-KV attention

- selected ID gather,
- compact K/V scratch,
- compact attention,
- later fuse only if beneficial.

## P4 — PLE descriptor/store refactor

- resolve shards once,
- cache FP8 scale once,
- cache file/MetalIO handles,
- batched gather API.

## P5 — Bounded FP8 PLE row cache

- explicit memory cap,
- residency owned by ResidencyManager,
- table can never be promoted wholesale,
- expose hit/miss/read telemetry.

## P6 — Async PLE prefetch

- token-admission prefetch,
- chunk-wide prefill prefetch,
- pure row-ID calculation separated from state mutation.

## P7 — PLE Metal island

Only after I/O is no longer dominant:

- FP8 decode,
- K/V projections,
- gate,
- norm,
- convolution,
- residual update.

---

# 17. Things not to disturb during this pass

Do **not** unnecessarily rewrite these while implementing the above:

- existing QKV + index projection fusion,
- four-stream HC physical representation,
- established GDN path,
- MoE routing/streaming work,
- current PLE hashing mathematics,
- expert MetalIO residency machinery.

The comparison does not identify them as the immediate missing performance opportunity.

---

# 18. Correctness / performance gates

Every optimization should maintain a reference path and clear acceptance gate.

## QSA state parity

For the same input sequence:

- compressed block vectors match within expected floating-point tolerance,
- selected block IDs match exactly,
- selected token IDs match exactly,
- decode logits/tokens remain within established oracle tolerance.

## Prefix restore parity

Validate at multiple positions, including positions where the QSA pending group contains 0, 1, 2, and 3 rows.

## PLE parity

Validate:

- row-ID hashes,
- shard resolution,
- FP8 decode,
- scale application,
- convolution history,
- cold-row and cache-hit behavior produce identical numerical results.

## Storage invariant

During full runs:

- the complete PLE table is never allocated in RAM,
- RSS growth from PLE caching is bounded,
- cache telemetry proves the configured maximum is respected.

## Performance measurements

Track independently:

- QSA compression/update time,
- QSA scoring time,
- QSA top-k time,
- selected-KV gather time,
- sparse/compact attention time,
- PLE row-ID generation,
- PLE metadata overhead,
- PLE storage latency,
- PLE cache hit rate,
- PLE compute,
- end-to-end ms/token and tok/s.

---

# 19. Source references

Primary upstream implementation references inspected during research:

- llama.cpp Qwen4Exp model implementation:
  https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen4exp.cpp

- llama.cpp initial Qwen3.8/Qwen4Exp support PR:
  https://github.com/ggml-org/llama.cpp/pull/27742

- llama.cpp gather/sparse-QSA follow-up work:
  https://github.com/ggml-org/llama.cpp/pull/28213

- vLLM Qwen4Exp model implementation:
  https://github.com/vllm-project/vllm/tree/main/vllm/models/qwen4_exp

- vLLM Qwen4Exp support PR:
  https://github.com/vllm-project/vllm/pull/53896

- SGLang Qwen4Exp support work:
  https://github.com/sgl-project/sglang/pull/36497

- SGLang QSA implementation (active Qwen4Exp branch at time of research):
  `python/sglang/srt/layers/attention/qsa/`

- Qwen model/config sources:
  https://huggingface.co/Qwen/Qwen3.8-Flash-Next

---

# 20. Durable conclusions

1. **Qwen3.8 is an extension of Qwen3.5 runtime machinery, not a separate engine.**
2. **QSA compressed index state must be persistent and incremental.**
3. **Logan already has true selected-position QSA attention; the immediate structural waste is index reconstruction and host-side selection.**
4. **The QSA selector and sparse-attention backend should eventually be one coordinated Metal execution island.**
5. **The 51B PLE table must remain an explicit NVMe resource and must never become an ordinary resident tensor.**
6. **A bounded FP8 hot-row cache is preferable to an uncontrolled giant mmap working set on Apple Silicon.**
7. **PLE row lookup should use a persistent resolved store descriptor and batched gather API.**
8. **PLE reads can be prefetched because row IDs depend only on token history.**
9. **QSA + PLE + GDN persistent state must participate coherently in prefix snapshots and restore.**
10. **The next concrete Logan milestone is persistent QSA compression + exact faster top-k before attempting more exotic kernels.**
