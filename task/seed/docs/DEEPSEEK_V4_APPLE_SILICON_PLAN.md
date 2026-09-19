# DeepSeek V4 on Apple Silicon: implementation plan

Status: planning only, updated 2026-09-07. Repository baseline: Logan `132efcd1abd311973acf657c8a5967b0a7d72e04`.

Start with **DeepSeek-V4-Flash target-only inference on the M2 MacBook Air with 16 GB unified memory**. Reuse Logan's compiler, compact expert representation, residency accounting, MetalIO pool, and scheduling infrastructure. Implement V4's attention and state semantics explicitly. Establish correct prefill and decode before enabling prefix reuse, speculative decoding, or long-context claims.

This document proposes work and acceptance gates; it does not report a working Logan V4 engine or a measured V4 speed. Colibri is a read-only architectural and fixture reference. Do not build or run that fork as the implementation path.

## 1. Evidence and checkpoint identity

The official family includes Flash at 284B total / 13B active parameters, and Pro at 1.6T / 49B. Their shared architecture uses compressed attention and manifold-constrained hyper-connections. Model context capacity is not a promise that the complete runtime fits at that context on this Mac. [Official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)

Pin the exact checkpoint revision before compiling. The original Flash and Flash-0731 are separate inputs:

| Field | Original Flash | Flash-0731 |
|---|---:|---:|
| Target layers / hidden width | 43 / 4096 | 43 / 4096 |
| Routed experts / selected / expert width | 256 / 6 / 2048 | 256 / 6 / 2048 |
| Shared experts / hash-routed layers | 1 / first 3 | 1 / first 3 |
| Attention heads / head width / window | 64 / 512 / 128 | 64 / 512 / 128 |
| Indexer heads / width / top-k | 64 / 128 / 512 | 64 / 128 / 512 |
| HC width multiplier / Sinkhorn iterations | 4 / 20 | 4 / 20 |
| Compression entries | 44 | 46 |
| DSpark config fields | Absent | Block 5; noise token 128799; target layers 40, 41, 42; Markov rank 256 |

Both target schedules begin with two uncompressed window layers, then alternate ratios 4 and 128 through layer 42. Extra compression entries belong outside that 43-layer target loop; their exact mapping must come from the selected revision's inference configuration and tensor inventory. Do not infer auxiliary stage count from array length or `num_nextn_predict_layers` alone. [Original configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json), [0731 configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/config.json)

Capture a manifest containing repository ID, immutable revision, configuration hashes, tokenizer and chat-template hashes, shard sizes and hashes, tensor names/dtypes/shapes, and compiler/layout ABI. Verify local shard completeness before interpreting a failure as an engine bug. The local checkpoint revision, available disk space, complete tensor census, and actual Logan allocation total remain unverified in this planning pass.

## 2. What already exists, and what must be established

The inspected `logan-compiler/src/model/deepseek_v4.rs` contains a DeepSeek V4 frontend. It probes `model_type`, collects routed gate/up/down members, validates expert completeness and geometry, distinguishes hash-routing tables from router biases, and validates static attention/HC/global roles. Existing fixture tests include incomplete inventories, missing static tensors, UE8M0 spelling, and indexed-compression classification. These are compiler foundations; they do not establish runtime support or real-checkpoint parity.

The historical Colibri reference includes `c/deepseek_v4.c`, `c/deepseek_v4_dspark.inc`, tiny fixtures, oracle generators, and `docs/deepseek-v4.md`. That document describes an older target-only CPU scope and excludes DSpark even though separate DSpark source exists. Treat each reference's actual source/revision as authoritative over its stale overview. Its residency estimates are not a measured Logan/M2 footprint.

| Reusable Logan work | V4-specific work or validation |
|---|---|
| Native compact expert payloads and GPU matvec | Exact FP4 block/scale semantics, V4 gate/up/down roles and activation limits |
| Persistent prepared SSD cache | Versioned V4 representation key, byte accounting, complete cache publication |
| Bounded MetalIO slot pool and completion tracking | Lease lifetime through V4 attention, shared expert, and routed expert consumers |
| Residency manager and scheduler | Physical allocations charged once across experts, static tensors, KV, scratch, prefix snapshots |
| Low-level FP8 GEMV and HC primitives (model integration still requires qualification) | V4 tensor shapes, mHC coefficients/order, norms, head reduction |
| Qwen4 QSA indexing/compression work | Independent V4 CSA/HCA oracle; reuse primitives only after matching semantics |
| Prefix/snapshot infrastructure | V4 window state, compressed history, partial compressor states, and rollback |

Do not transplant a Qwen full forward pass. Make shared low-level operations reusable while keeping model semantics in a V4 engine. Inventory executable/runtime dispatch before naming an existing component “V4-ready.”

## 3. Compiler and package contract

First produce a small, inspectable V4 package manifest and compiler fixture. Then compile a pinned real checkpoint without executing it.

1. Preserve source quantization semantics. Distinguish routed FP4 with per-32-value scales from FP8 dense matrices with block scales. Preserve BF16/F32 tensors where specified, especially sensitive norms, HC coefficients, and compressor inputs. A model-wide FP8 label is insufficient to choose every tensor's kernel.
2. Normalize all execution parameters into versioned metadata: target layer count and per-layer compression kind, hash-routing depth, shared-expert count, scoring and normalization rules, route scale, activation limit, RoPE/YaRN parameters, attention/indexer geometry, sink values, HC settings, tokenizer, and auxiliary configuration. Reject unsupported combinations explicitly rather than substituting Qwen defaults.
3. Give target, MTP, and DSpark tensors distinct namespaces. For target-only packages, explicitly record which auxiliary tensors were omitted and why. For a future DSpark package, preserve all required auxiliary tensors and configuration, not only a generic MTP count.
4. Produce role-level byte totals, tensor-relative offsets, strides, scale shapes, padded layout sizes, and checked range arithmetic. Validate scale aliases through the existing canonical dtype path. A shape-correct matrix with incorrect scale orientation is a correctness failure.
5. Reuse the prepared-cache mechanism with identity including checkpoint hash, tensor/expert identity, source representation, output layout version, and kernel ABI. A cache from another revision or architecture must miss cleanly. Publish complete records atomically and detect truncated/corrupt records before device use.

Acceptance: deterministic tiny-package manifests; rejected incomplete/mis-shaped inputs; decode of every selected representation against scalar fixtures; a real-checkpoint inventory that accounts for every target tensor and explicitly classifies extras. Do not equate “compiler accepted the input” with “runtime can execute it.”

## 4. Correct target execution

Use the pinned official reference to make small deterministic numerical fixtures, including teacher-forced intermediate tensors. The initial oracle may run on a tiny CPU model and emulate quantization; it must not use old Colibri execution as the source of truth.

The reference requires separate window and compressed histories. Ratio-4 compression overlaps adjacent blocks; ratio-128 does not. Compressor pooling accumulates in FP32. The indexer has its own compressor and uses rotated, FP4-simulated values for selection. Attention applies quantization simulation to non-RoPE KV dimensions while retaining positional precision. Preserve these operations and their ordering in the oracle before trying compact cache storage. [Official attention, compressor, and indexer implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py)

The proposed target path is:

1. Token embedding and HC expansion.
2. Per-layer HC reduction, normalization, attention projections, indexer/compressor updates, attention, and HC residual update.
3. HC reduction for the FFN, hash or learned routing, shared expert plus selected routed experts, then HC residual update.
4. Final HC reduction, norm, output head, and sampler.

Resolve exact operation ordering against the oracle rather than this abbreviated list. V4 needs its own fixtures for router scoring, bias-only selection versus output weighting, normalization, shared-expert addition, activation clipping, attention sinks, output projection grouping, and positional transforms.

State design:

- One request owns logical token position, bounded window rings, compressed KV pages, indexer pages, and all incomplete/overlapping compressor states.
- Allocate compressed pages incrementally. A 128-original-token alignment can coordinate ratio-4 and ratio-128 page lifetimes, but measure the useful allocation granularity rather than hardcoding huge pages.
- Retain a compressed block only once. Attention selection indexes logical entries and translates through a checked page table; no accidental duplicate contribution from window and compressed ranges.
- Implement chunked prefill with causal visibility at each query position. Incremental decode and prefill must agree across every compressor boundary.
- Start with explicit, completed command batches. Then combine compatible GPU stages and overlap expert reads where dependencies permit. Do not hide unfinished reads behind a “resident” flag.

The official report also distinguishes persistent compressed history from bounded window/incomplete-compression state, and explains why prefix reuse must restore or recompute unfinished tails. Use that distinction for Logan snapshots. [Technical report, cache management](https://arxiv.org/html/2606.19348v1#S3.SS5)

## 5. Memory and I/O budget for the 16 GB Mac

All sizes below use binary MiB/GiB unless marked GB. They are arithmetic models, not measurements. Physical file padding, Metal allocations, scales converted to wider types, and staging copies can increase them.

For a target expert with three `4096 × 2048` matrices and standard MXFP4 packing:

```text
weights per expert = 3 × 4096 × 2048 = 25,165,824
bytes per expert   = weights × (1/2 + 1/32) = 13,369,344 = 12.75 MiB
all target experts = 43 × 256 × 12.75 MiB = 137.0625 GiB
selected per token = 43 × 6 × 12.75 MiB = 3.21240234375 GiB
```

These counts exclude static weights, auxiliary stages, package metadata, and layout expansion. The source and prepared package can coexist on disk; calculate peak disk requirements before a full compilation. Avoid creating a second complete expanded representation by default.

A BF16 `129280 × 4096` embedding or output head is **0.986328125 GiB**, approximately **1.059 GB**. The historical Colibri overview's “1.06 GiB head” is not a precise binary-size estimate. Determine whether each actual static total already includes embedding/head before adding them.

With BF16 storage for one 512-wide KV entry and one 128-wide indexer entry, a target-only state estimate is:

```text
window bytes = 43 × 128 × 512 × 2
CSA bytes    = 21 × floor(context / 4) × (512 + 128) × 2
HCA bytes    = 20 × floor(context / 128) × 512 × 2
```

| Context tokens | Window + compressed KV + indexer |
|---:|---:|
| 8,192 | 59.125 MiB |
| 65,536 | 435.375 MiB |
| 1,048,576 | 6,885.375 MiB |

This excludes compressor FP32 states, page padding, position tables, scratch, logits, snapshots, batch multiplicity, and DSpark. Quantization simulation does not imply compact physical cache storage. A million-token allocation is therefore outside the first M2 milestone.

Use this **proposed initial admission envelope**, then replace each provisional number with the manifest and allocation census:

| Account | Initial upper allocation |
|---|---:|
| Static weights, including any resident embedding/head | 7.50 GiB |
| Expert cache and active expert leases together | 1.00 GiB |
| Attention/compressor/request state at initial context | 0.25 GiB |
| Prefill/decode scratch, staging, logits | 1.00 GiB |
| Allocator/runtime reserve | 1.25 GiB |
| Total process envelope | 11.00 GiB |

This is an admission target, not an assertion that the current package fits it. Keep roughly 5 GiB outside that envelope on the 16 GiB machine, and shrink admission further if current OS usage requires it. If the actual static set exceeds 7.5 GiB, choose tiled/streamed static weights or a separately validated compact representation; do not rely on swap. Count shared CPU/Metal views once, distinct staging/private copies separately, and pinned plus evictable experts in one ledger. A 1 GiB expert pool holds only about 80 unpadded experts, so strong temporal locality must be measured rather than assumed.

The I/O lower bound is:

```text
expert seconds/token >= actual selected miss bytes / measured effective read bandwidth
```

With zero hits, five tokens/second would require at least **16.06 GiB/s of expert payload reads alone** under the compact assumptions above. Qwen3.6 performance is not a V4-Flash performance forecast. First measure selected bytes, hit rate by bytes, physical SSD reads, MetalIO wait exposure, dense compute, and command overhead. A prepared SSD cache avoids repacking; it does not make missed expert reads disappear.

## 6. Milestones and acceptance gates

| Milestone | Deliverable | Gate before proceeding |
|---|---|---|
| V0: identity and census | Pinned Flash revision, package contract, tensor/byte report | Every target tensor accounted for; auxiliary namespaces explicit; no invented fit estimate |
| V1: tiny target oracle | CPU/scalar semantics and tiny V4 fixtures | Agreed per-operation numerical tolerances, exact selected indices on non-tied fixtures, finite logits, correct teacher forcing |
| V2: target Metal path | Persistent GPU state with streamed compact experts | Same-package CPU/Metal parity; native quantization fixtures; correct causal prefill/decode |
| V3: real Flash smoke run | Normal text prompt and complete decoded continuation | Several readable prompts plus teacher-forced logit comparison to pinned reference; no unexplained repeated-token failure |
| V4: bounded scheduling and prefix reuse | Request isolation, completion ownership, snapshot restore | Cancellation, backpressure, I/O failure, eviction, prefix hit/miss, and fresh-session equivalence within the memory envelope |
| V5: measured optimization | Timings, cold/warm profile, one controlled optimization at a time | Repeatable end-to-end improvement under equivalent conditions and preserved correctness; report TTFT and decode separately |
| V6: longer context and speculation | 64K admission, then optional DSpark | Memory/state gates pass independently; DSpark adds a verified speed gain after accounting for extra memory and reads |

The numerical tolerances are to be chosen from precision-aware oracle baselines before optimization, not adjusted afterwards to bless a failing kernel. Token identity is useful for equivalent execution paths; it does not replace logit/error analysis across different quantization representations.

Boundary fixtures should include lengths around 4, 128, window wraparound, and the point where compressed history exceeds top-k 512 (around 2,048 input tokens), plus chunk splits immediately before and after those boundaries. Include nonzero starting positions, a restored partial compressor block, two interleaved sessions, and cancelled work with outstanding device leases.

The first real run should use a small context cap and ordinary visible text. Expand to 8K, then 64K only after admission and state behavior are measured. Keep Pro and million-token context outside these initial acceptance gates.

## 7. DSpark: a separate, concrete follow-up

Flash-0731's official inference code contains DSpark stages under `mtp.*`, a Markov head and confidence head. Target conditioning takes mean-reduced HC outputs from configured target layers. This is concrete upstream support; it is not a generic MTP flag. The public DeepSpec repository's Qwen/Gemma draft table is not evidence that V4's packaged DSpark is absent. [0731 DSpark implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/model.py), [DeepSpec repository](https://github.com/deepseek-ai/DeepSpec)

Plan DSpark after V5:

1. Reconcile 0731's public model config, inference config, converter, and actual `mtp.*` tensor shapes. Measure the complete draft footprint and verify which target buffers it shares. Never infer all stage geometry from the model config's MTP count.
2. Preserve the target conditioning positions and reductions exactly. Add distinct typed draft attention, Markov, confidence, and auxiliary head operations rather than routing through Qwen's MTP topology.
3. Verify a proposed block through the ordinary causal target path. Publish only the accepted prefix. Roll back every rejected suffix's window writes, compressed entries, partial compressor state, logical positions, and RNG state as required by the selected sampler.
4. Prove greedy on/off token identity, including zero acceptance, full acceptance, EOS inside a block, context exhaustion, cancellation, and prefix restoration. For non-greedy decoding, validate the exact acceptance/correction algorithm and output distribution; matching seeds alone is not a sufficient test.
5. Evaluate fixed short blocks first, then confidence scheduling. Compare wall time per emitted token, accepted length, draft compute, target verification, SSD bytes, memory, and TTFT. Default to target-only if extra draft residency displaces enough expert cache to lose end-to-end performance.

The DSpark paper's production results concern a serving system with its own throughput profile. They do not establish a speedup on this 16 GB Apple Silicon machine. [DSpark paper](https://arxiv.org/abs/2607.05147)

## 8. Next implementation slice

After the Qwen port is stable, implement **V0 and V1 together**: a pinned manifest/tensor census plus a tiny target-only V4 fixture covering one window layer, one CSA layer, one HCA layer, hash/learned routing, and HC transitions. That creates a reviewable compiler/runtime contract and a trustworthy correctness oracle before committing to the full streaming engine.

Planning verification: repository guidance and compiler/reference files were inspected read-only; current public official configs, model implementations, model card, and papers were reviewed. No Colibri executable or DeepSeek workload was run, and no DeepSeek runtime changes or speed measurements are claimed here.

