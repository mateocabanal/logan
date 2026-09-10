# Logan heterogeneous CPU / Metal / ANE execution

**Status:** heterogeneous substrate implemented; experimental Qwen3.8 GDN ANE→Metal front-half path is hardware-qualified and remains opt-in.

The goal is not to turn Logan into an ANE-only runtime. Apple Silicon has three
useful execution engines sharing one physical memory system, and Logan should
place *islands* of work on the engine that wins for that island while minimizing
handoff cost.

## Implemented substrate

### `logan-ir`

`ExecutionPlan` is an optional overlay on the tensor graph:

- `ExecutionBackend::{Cpu, Metal, Ane}`
- `ExecutionIsland`: a fixed group of graph nodes assigned to one backend
- `ExecutionEdge`: a value crossing island boundaries
- `TransferKind::{Alias, SharedMemory, HostCopy, Streamed}`

`plan.logan` v5 serializes the execution overlay. The reader accepts v4 plans
and upgrades them in memory with `execution = None`.

### `logan-core`

`SharedAllocationRegistry` describes physical allocations independently from
backend-native handles. `DeviceVisibility::APPLE_UMA` means one allocation is
visible to CPU, GPU, and Neural Engine; visibility does not imply completion or
coherency ordering.

`TensorStorage::Shared` can reference a byte range in one such allocation.

The scheduler already had `DeviceKind::Neural`. `DeviceRegistry` now supports
`find_kind_capable`, so an `(Neural, Accelerator)` submission cannot
accidentally dispatch to a GPU accelerator target.

### `logan-ane`

The private ANE runtime is loaded dynamically and type-encoding checked before
unsafe Objective-C calls. Current capabilities include:

- raw MIL compilation/load/evaluation
- IOSurface-backed requests
- lower-level `_ANEClient` direct evaluation
- request pre-mapping
- CoreML Blob v2 constant weights
- multiple fixed dense projections in one MIL program

`parallel_dense_fp16_f32_io` expresses `[O,I]` matrices as 1x1 convolutions,
casts one fp32 activation to fp16 once, evaluates multiple projections, then
returns fp32 outputs. This matches the input-side shape of Qwen GDN's
qkv/z/a/b projections.

### `logan-metal`

`MetalSharedSurface` imports an IOSurface as a persistent shared `MTLBuffer`
using `newBufferWithBytesNoCopy`. The bridge retains the IOSurface and declines
when Metal cannot import it without copying.

Hardware validation on the M2 passed both directions with zero error:

1. Metal mutates an IOSurface -> ANE consumes the same bytes.
2. ANE writes an IOSurface -> Metal mutates the same bytes -> CPU observes it.

## GPU-resident ANE continuation experiment

`QWEN_GDN_ANE_GPU_TAIL=1` keeps completed ANE projections on shared surfaces,
then runs GPU gather, Conv1D, recurrence, gated RMSNorm and BF16 output projection
in one Metal command buffer. It remains opt-in along with ANE. See
[the implementation and validation results](ane_async_20260908.md) for numerical
differences, measured CPU map reduction, cache repair, and the bounded private
shared-event hardware probe. End-to-end acceleration is not yet established.

## Synchronization contract

Zero-copy is not synchronization.

The initial safe contract is deliberately synchronous:

- ANE `evaluate` / `_ANEClient` direct evaluate must return before Metal consumes
  ANE-written bytes.
- A Metal command buffer must complete before ANE consumes Metal-written bytes.
- CPU mappings must not overlap a device write.

Later work can replace these host waits with shared-event/fence integration.
`_ANESharedWaitEvent` exists on the validated OS and is a candidate for this.

## Island selection rules

An ANE island should initially satisfy all of these:

1. fixed shape and stable MIL program;
2. enough work to amortize roughly 0.1--0.2 ms dispatch overhead;
3. weights can stay resident/compiled for many calls;
4. input/output handoff can use shared IOSurface memory;
5. numerical conversion (usually fp16 today) passes a model-level quality gate;
6. no dynamic control flow that forces recompilation each token.

Do not select ANE merely because an op is a matmul. Placement is an end-to-end
latency decision including conversion, handoff, scheduling and memory pressure.

## First production experiment: Qwen3.8 GDN

### Phase A — parallel input projections only

For each GDN layer, compile one ANE island containing the BF16 checkpoint's
input projections after conversion to qualified fp16 constants:

```text
                    ┌─ qkv
fp32 activation ────┼─ z
   one cast fp16    ├─ a
                    └─ b
```

Keep recurrence, conv, gate and output projection on the current path. Use
IOSurface outputs so the next backend consumes them without copies.

This isolates the known BNNS-heavy bottleneck and gives an apples-to-apples
comparison before moving stateful recurrence onto a private backend.

### Qualified decode front-half: ANE projections → Metal Conv1D/SiLU

The current private ANE compiler on the validated M2/macOS build rejects the
single-program formulation needed for `qkv/z/a/b -> causal depthwise Conv1D ->
SiLU`. The individual operations compile, but several combined forms fail with
`InvalidMILProgram`/`CompilationFailure`, including grouped/depthwise conv and
history slicing combined with extra per-channel BLOB constants. Supplying a
runtime tensor as a dynamic conv weight compiles but evaluation is cancelled
with status `0x1d`.

The qualified workaround keeps the projection island on ANE and continues on
Metal through the *same qkv IOSurface*:

```text
x
│
├─ ANE: qkv / z / a / b
│        │
│        └─ qkv IOSurface (no activation copy)
│                    │
└────────────────────▼
             Metal: causal Conv1D(k=4) + SiLU
                         │
                         ▼
              existing DeltaNet recurrence
```

The Metal continuation patches the three causal-history values into reserved
lanes of the ANE qkv surface, leaves the current qkv value in the final lane,
and dispatches one thread per channel. Host conv state is advanced only after
successful GPU completion, preserving scalar-fallback correctness.

Real layer-0 bridge validation measured about 0.22--0.24 ms median per 10,240-
channel Conv1D+SiLU dispatch with max absolute error around `4e-9` versus the
CPU reference. A four-forward full-model history-order gate retained identical
argmax and top-100 ordering; final-logit cosine was `0.999999999999179` versus
the projection-only ANE path.

All 36 Qwen3.8 GDN layers can be resident with the fused continuation. In one
steady second-forward run the model measured 4.21 s wall / 2.20 s GDN, but
whole-model timing remains noisy and should not be treated as a stable speedup
claim yet. Against the original BF16 baseline, the all-36 fused path measured
about 0.0532% relative logit RMSE, cosine `0.9999998624`, identical argmax and
identical top-50 ordering. Sharing the identical Metal device/queue/pipeline
across layers reduced peak process RSS in the paired all-36 run from roughly
4.47 GiB to 3.12 GiB with bit-identical logits.

Enable experimentally with `QWEN_GDN_ANE=1`; select layers with
`QWEN_GDN_ANE_LAYERS`, and disable only the Metal continuation with
`QWEN_GDN_ANE_FUSED=0` for projection-only A/B testing.

### Phase B — complete GDN island

Once Phase A wins and passes quality:

```text
RMSNorm -> qkv/z/a/b -> Conv1D -> recurrent update -> gate -> out projection
```

Compile that as one fixed MIL program with recurrent state represented through
persistent/shared buffers if the compiler accepts the required operations.

### Phase C — prefill

Benchmark the same island at sequence chunks S=16/32/64/128. ANE is expected to
be more attractive when each dispatch carries more arithmetic than decode S=1.

## Acceptance gates

No ANE path becomes default until it passes:

- reference output / token-identity gate on deterministic prompts;
- per-island numerical error measurements;
- end-to-end tok/s and TTFT, not kernel-only timings;
- peak RSS / UMA pressure comparison;
- warm and cold compiled-model behavior;
- cancellation/unload correctness;
- fallback when private ABI or MIL compilation is unavailable.

Useful telemetry to add before production enablement:

- `ane_compile_ms`, `ane_load_ms`, `ane_dispatch_us`
- ANE island calls / failures / fallbacks
- shared-memory handoffs vs copied handoffs
- bytes crossing each island boundary
- compiled-island cache hit rate
- host synchronization time between Metal and ANE

## Longer-term opportunities

- ANE speculative draft model running concurrently with Metal verifier.
- Always-resident shared MoE experts on ANE while routed experts stream to
  Metal.
- ANE prefill islands.
- dynamic/mutable-weight kernels.
- multimodal/audio encoders isolated from the GPU generation workload.
- typed shared events for GPU/ANE pipelining without host waits.
- LoRA or other small training workloads after mutable-weight procedure ABI is
  validated end-to-end.
