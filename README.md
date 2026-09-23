# Logan

**The mutant compiler and runtime for local LLM inference.**

Logan is a Rust-native inference system built for running models that do not fit neatly in memory. It combines hardware-aware compilation, model-format planning, disk-streamed sparse execution, native accelerator backends, and a correctness-first runtime so large local models can be transformed to fit the machine they actually run on.

The name is a reference to Wolverine: Logan *mutates* checkpoints into hardware-specific execution plans rather than treating the original model layout as sacred.

Today Logan is especially focused on Apple Silicon, large sparse/MoE models, raw MLX/safetensors execution, Metal + MetalIO, and aggressive reduction of CPU↔GPU synchronization and dispatch overhead.

## Philosophy

Logan is guided by a few principles:

- **Models should adapt to hardware, not the other way around.** Compilation may change storage format, quantization, layout, residency policy, and execution placement while preserving model semantics.
- **Models larger than RAM are a normal workload.** Cold or sparse structures should remain on NVMe when possible; residency is a budgeted runtime decision, not an assumption that everything fits.
- **Measure the critical path.** An optimization is not a win because a microbenchmark, cache hit rate, or I/O counter improved. It must survive controlled end-to-end A/B testing.
- **Correctness gates come before speed.** Token identity, numerical tolerances, state progression, deterministic replay, and fallback behavior are treated as promotion gates.
- **Minimize boundaries and submissions.** On modern accelerators, command-buffer count, synchronization, packing, and handoff overhead can dominate the arithmetic itself.
- **Mechanism and policy stay separate.** MetalIO, caching, RouteScout, residency, scheduling, and model formats expose mechanisms; the runtime decides when they are actually worthwhile.
- **Prediction is advisory.** Speculative systems such as RouteScout may prefetch or stage work, but the model's native router remains authoritative.
- **Failed experiments are part of the architecture.** `EXPERIMENTS.md` records rejected, superseded, and inconclusive ideas so Logan does not repeatedly rediscover the same false assumptions.

## What Logan can do

### Hardware-aware model execution

Logan supports multiple runtime sources and representations rather than forcing every model through one container:

- raw **safetensors / MLX** checkpoints are first-class runtime inputs;
- **`.logan`** is the canonical compiled-format direction;
- legacy **COLI** packages remain supported for compatibility while the `.logan` path matures;
- quantization semantics are preserved by the compiler rather than silently requantized;
- execution planning is designed around the admitted working set, backing storage, target accelerator, and context/concurrency budget.

### Qwen sparse/MoE runtime

`logan-qwen4` contains the high-performance Qwen path used for Qwen3-Next / Qwen3.6-class MoE checkpoints and Qwen4Exp / Qwen3.8-Flash-Next work.

The Apple Silicon path includes:

- Metal and MetalIO execution;
- SSD-streamed routed experts;
- Gated DeltaNet support;
- QSA, PLE and hyper-connections for Qwen4Exp-family models;
- native MLX affine 4/5/6/8-bit kernels;
- grouped routed-expert execution;
- whole-route expert I/O issue;
- GPU-resident GDN paths;
- fused shared-expert execution;
- reusable Metal weight objects;
- source-neutral expert prefetch hooks.

Recent work reduced the routed-MoE compute phase from many small synchronous submissions to **two Metal command buffers per layer** in the normal top-k path.

### Dense Llama / MiniCPM runtime

`logan-llama` provides the dense Llama-compatible runtime used by MiniCPM5-class checkpoints.

Current work includes:

- direct raw-checkpoint execution;
- MiniCPM5 tensor/config validation;
- dense Metal projection paths;
- model-agnostic state/prefix-cache integration;
- DSpark-specific speculative-draft plumbing;
- ANE experiments for fixed-shape islands where they make architectural sense.

### RouteScout

RouteScout is Logan's predictive MoE research path.

It learns temporal and cross-layer routing structure so future expert demand can be predicted several layers ahead. The important rule is that RouteScout never decides the actual route: the model router remains authoritative.

Experiments on Qwen3.6 show that useful routing structure is genuinely predictable, including several layers into the future. On the current 16 GiB M2, however, speculative SSD prefetch is not a throughput win even when predicted reads arrive before demand. The machine is already fast enough at expert transport that speculative traffic mainly creates shared-memory pressure.

That makes RouteScout more interesting for workloads where transport is genuinely expensive — larger storage misses, remote workers, or distributed MoE — than as a blanket optimization for the current M2 path.

### `logand`

The workspace includes a persistent inference daemon with:

- an embedded web dashboard;
- model loading and generation settings;
- Hugging Face download / quantization workflow support;
- OpenAI-compatible `POST /v1/chat/completions`;
- OpenAI-compatible `POST /v1/responses`;
- model-specific prompt rendering;
- Qwen and Llama/MiniCPM runtime routing.

The long-term product direction is a local inference service that owns model state, caching, scheduling, residency, and hardware resources while exposing a simple API and UI to clients.

## Recent performance work

The current Qwen3.6 raw-MLX path is a good example of Logan's measurement philosophy.

The initial hypothesis was that SSD expert streaming was the dominant bottleneck. The experiment ledger disproved that. The major wins came instead from:

1. batching routed-expert Metal submissions;
2. reducing the MoE compute phase to two command buffers per layer;
3. re-enabling whole-route concurrent I/O after compute-side dispatch overhead was reduced;
4. vectorizing MLX affine 4/5/6/8-bit kernels;
5. keeping eligible GDN work entirely on GPU;
6. fusing the shared expert;
7. reusing Metal weight objects instead of recreating them every token.

On the same canonical harness, those changes moved the Qwen3.6 path from roughly **1.8 tok/s to about 3.7–3.8 tok/s on an M2 MacBook Air**, approximately a **2× end-to-end improvement**, while retaining the same deterministic greedy trajectory.

The exact absolute rate varies with host state, so Logan reports paired/alternating A/B results rather than treating one fast run as truth.

See **[EXPERIMENTS.md](EXPERIMENTS.md)** for the full evidence trail.

## Workspace

| Crate | Role |
|---|---|
| `logan-abi` | Stable target, representation and semantic ABI identities |
| `logan-artifact` | Canonical `.logan` framing, manifest/table I/O and structural validation |
| `logan-format` | Legacy COLI compatibility |
| `logan-ir` | Model/compiler intermediate representation |
| `logan-core` | Scheduler, state, residency and model-agnostic runtime foundations |
| `logan-metal` | Metal kernels, buffers, MetalIO integration and Apple-Silicon primitives |
| `logan-compiler` | Hardware-aware compiler, quantization, physical planning and artifact emission |
| `logan-qwen` | Scalar/reference Qwen implementation and numerical oracle work |
| `logan-qwen4` | High-performance Qwen MoE / Qwen4Exp runtime |
| `logan-llama` | Dense Llama-compatible / MiniCPM5 runtime |
| `logan-spark` | Spark model support |
| `logan-ane` | Native Apple Neural Engine experiments and execution islands |
| `logan-chat` | Interactive chat client |
| `logand` | Persistent inference daemon, dashboard and OpenAI-compatible APIs |

## Build and test

```bash
cargo build --release --workspace
cargo test --workspace --all-targets
```

The C-oracle differential tests skip automatically when the C reference tree is absent. The C Colibri fork remains a numerical/parity oracle and is not vendored into this repository.

## Run

A tiny deterministic fixture:

```bash
cargo run --release -p logan-qwen4 -- fixtures/qwen4_moe_tiny
```

Run the daemon:

```bash
cargo run --release -p logand
```

The daemon exposes its dashboard and API from the configured bind address.

For raw MLX/safetensors expert-streaming experiments:

```bash
LOGAN_EXPERT_NOCACHE=1 \
  target/release/logan-qwen4 /path/to/model
```

On macOS, that mode bypasses the normal file cache for expert-streaming descriptors and uses MetalIO when available.

Legacy compiled-package emission remains available during the `.logan` migration:

```bash
target/release/logan compile MODEL_DIR \
  --target macos-arm64-metal-apple8-v1 \
  --quant mxfp4 \
  --codec none \
  --opt default \
  -o MODEL.Apple8.coli \
  --verify
```

## Experiment discipline

Performance and architecture experiments belong in `EXPERIMENTS.md`.

A promoted optimization should normally have:

- a falsifiable hypothesis;
- a named baseline;
- correctness gates;
- comparable A/B conditions;
- phase-level measurements when relevant;
- an explicit KEEP / REJECT / INCONCLUSIVE decision.

This has already prevented several misleading optimizations from becoming permanent defaults: larger expert caches, speculative prefetch on the current M2, excessive MetalIO queue depth, unnecessary copies removed from the wrong lifetime boundary, and attention batching that looked obvious but lost in practice.

## Lineage

Logan is the Rust successor to the Colibri C inference engine.

Colibri remains useful as a reference implementation and numerical oracle. Logan is the forward path: a more general, model-aware runtime built around compilation, scheduling, tiered storage, accelerator execution, and models whose working set may exceed RAM.

## License

Apache-2.0
