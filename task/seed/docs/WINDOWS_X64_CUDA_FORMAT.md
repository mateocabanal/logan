# Logan Windows x86-64 + CUDA package format

**Status:** design draft / implementation target
**Date:** 2026-09-09
**Initial hardware target:** Windows x86-64, Ryzen 5 5600X, GeForce GTX 1080 8 GiB (SM 6.1), NVMe SSD
**Initial model:** Qwen3.8-Flash-Next REAP-288 GGUF Q4_K_M

## 1. Goal

Define a Logan target package that is optimized for discrete NVIDIA GPUs and NVMe-backed MoE execution without pretending that compilation can add precision that does not exist in the source model.

The first implementation should favor three things over cleverness:

1. preserve the source quantization exactly whenever possible;
2. make routed experts independently streamable from NVMe;
3. keep directly executable quantized bytes compressed all the way into CUDA kernels.

This is not the Apple8 format with Metal names replaced by CUDA names. Windows + a discrete GPU has a different memory hierarchy and should get a different physical package layout.

## 2. Quantization integrity invariant

### 2.1 Compilation does not increase source precision

For every quantized source tensor, the compiler records the source tensor quantization type and treats its nominal weight-code precision as a hard ceiling.

Examples:

| Source tensor | Nominal weight bits | Legal automatic compile result |
| --- | ---: | --- |
| GGML Q4_K | 4 | Q4_K preserved / losslessly repacked |
| GGML Q5_0 | 5 | Q5_0 preserved / losslessly repacked |
| GGML Q6_K | 6 | Q6_K preserved / losslessly repacked |
| GGML Q8_0 | 8 | Q8_0 preserved / losslessly repacked |

An explicit requantization request may keep or reduce nominal precision. It may never raise it.

Therefore:

- Q4_K -> Q5/Q6/Q8/FP16 stored weights: **forbidden**;
- Q5_0 -> Q6/Q8/FP16 stored weights: **forbidden**;
- Q8_0 -> Q6_K: allowed only as an explicit lossy requantization;
- Q4_K -> another 4-bit family: still a real requantization and must not be called a lossless repack;
- Q4_K -> Q4_K with only byte/block relocation: allowed as a lossless repack.

The compiler implementation for this rule lives in `logan-compiler/src/quant/precision_policy.rs`.

### 2.2 Nominal bits are not effective bits per weight

Q4_K is called a 4-bit quant because the quantized weight codes are 4-bit. Q4_K blocks also contain scale/min metadata, so the actual stored bytes per weight are greater than exactly 4 bits.

That metadata overhead is part of the original quant format and is not an illegal precision increase.

The compiler must never claim that `effective_bpw <= 4.0` is required for a Q4_K source. The invariant is about the weight-code precision and quantization semantics, not container padding or scale metadata.

### 2.3 Runtime arithmetic is separate from stored model precision

CUDA kernels may dequantize Q4/Q5/Q6/Q8 values into registers and accumulate in FP32. That is normal execution and does not change the compiled model's stored precision.

The prohibition is against writing a higher-precision replacement tensor into the compiled package and presenting it as if the extra precision came from the source.

## 3. Actual REAP-288 source package

The downloaded source GGUF is approximately 83.8 GB and contains 1,224 tensors.

Model metadata observed from the file:

- architecture: `qwen4exp`;
- layers: 48;
- hidden size: 2560;
- experts: 288;
- routed experts per token: 10;
- expert FFN size: 640;
- context length: 262,144;
- PLE/n-gram table: `per_layer_token_embd.weight`, shape `[160, 320001536]`;
- PLE table quantization: Q5_0.

Tensor storage by source type:

| Source type | Tensor count | Stored bytes |
| --- | ---: | ---: |
| Q4_K | 568 | 27,320,832,000 |
| Q5_0 | 146 | 43,231,400,960 |
| Q6_K | 49 | 979,507,200 |
| Q8_0 | 48 | 12,074,188,800 |
| F32 | 388 | 152,112,640 |
| BF16 | 24 | 39,321,600 |
| F16 | 1 | 81,920 |

This is why `Q4_K_M` must not be treated as a global four-bit tensor type. It is a mixed recipe. The per-tensor source type is authoritative.

The two dominant storage classes are approximately:

- PLE / n-gram embedding: 35.6 GB;
- routed expert tensors: 45.3 GB.

Neither can be made normally resident on a 32 GB host. They require explicit storage-tier behavior.

## 4. Proposed target profile

Working profile name:

`windows-x86_64-cuda-ggml-v1`

The profile should eventually be registered in `abi/coli-target-registry.toml` once the lowerer and CUDA runtime are ready to consume it.

Proposed properties:

- OS: Windows;
- architecture: x86_64;
- backend: CUDA;
- minimum CUDA GPU capability for v1: SM 6.1;
- record alignment: 4096 bytes;
- preferred file-I/O granularity: 4096 bytes;
- resident CUDA alignment: at least 256 bytes;
- top-level codec for directly executable quantized weights: `none`;
- target execution layout: GGML-compatible native quant blocks.

The package should remain usable on newer NVIDIA GPUs. SM 6.1 is the minimum implementation target, not an instruction to emit GTX-1080-only model data.

## 5. Physical package classes

The package should separate data by access pattern rather than reproducing the GGUF tensor ordering.

### 5.1 Dense / always-hot tensors

Dense model tensors remain individually indexed COLI tensor records.

For quantized dense tensors:

- preserve the original GGML quant block bytes;
- do not dequantize/requantize during compilation;
- align record starts for Windows async I/O and CUDA staging;
- keep `codec = none` so CUDA can consume the quantized representation directly.

The non-PLE, non-expert portion of this model is only a few GB, so the runtime should attempt to keep the highest-value dense tensors resident on the GPU and spill according to the residency planner when context state competes for VRAM.

### 5.2 Routed experts: one streamable record per `(layer, expert)`

The source GGUF stores each expert as a contiguous slice inside each 3-D expert tensor. That lets the compiler build one Logan expert record without numerically touching the weights.

Observed examples:

Layer 0:

- `ffn_gate_exps`: Q4_K, 921,600 bytes per expert;
- `ffn_up_exps`: Q4_K, 921,600 bytes per expert;
- `ffn_down_exps`: Q8_0, 1,740,800 bytes per expert.

A layer whose down projection is Q5_0 has:

- gate Q4_K: 921,600 bytes;
- up Q4_K: 921,600 bytes;
- down Q5_0: 1,126,400 bytes.

The compiler should copy those exact source slices into one expert envelope:

```text
4 KiB-aligned expert record
  header / descriptors
  gate: exact source GGML block bytes
  up:   exact source GGML block bytes
  down: exact source GGML block bytes
  padding to I/O alignment
```

The padding is container storage only. It does not change quantization precision.

Reasons to bundle all three matrices:

- one routed-expert cache key;
- one async storage request per miss where practical;
- one residency lifetime across gate/up/down use;
- no three-way metadata lookup on the hot decode path;
- easier cancellation and lease ownership.

The matrix descriptor must retain each matrix's own quant type. An expert is allowed to be mixed, e.g. Q4_K gate/up plus Q8_0 down.

### 5.3 PLE table: 4 KiB page-packed Q5_0 rows

`per_layer_token_embd.weight` has 160 values per row and is stored as Q5_0.

A Q5_0 row of width 160 is 110 bytes in the current GGUF. Random 110-byte reads are a poor match for uncached Windows/NVMe I/O.

For this model, pack PLE rows into fixed 4096-byte storage pages without altering row bytes:

```text
37 rows x 110 bytes = 4070 bytes
26 bytes padding
---------------------------
4096-byte PLE page
```

This adds less than 1% storage overhead while giving every cold PLE fetch one naturally aligned 4 KiB read.

The PLE record metadata should encode at minimum:

- source quant format = Q5_0;
- row width = 160 values;
- row stored bytes = 110;
- page bytes = 4096;
- rows per page = 37;
- logical row count;
- page count;
- final-page valid row count.

Lookup becomes:

```text
page = row_id / rows_per_page
slot = row_id % rows_per_page
offset = page * 4096 + slot * 110
```

The runtime cache should cache raw 4 KiB pages or raw Q5_0 rows, never expanded FP32 PLE rows by default.

## 6. Quantized execution layout

The first CUDA layout should intentionally stay close to the GGML block definitions instead of inventing a CUDA-specific reordered quantization on day one.

Proposed logical execution formats:

- GGML Q4_K;
- GGML Q5_0;
- GGML Q6_K;
- GGML Q8_0;
- F32;
- BF16/F16 where the source already contains them.

A later CUDA-specific bit permutation is acceptable only if:

1. the transform is exactly invertible;
2. the quantization family and nominal weight bits do not change;
3. byte-level or decoded-value parity tests prove the transform;
4. package metadata marks it as `lossless_repack`, never `requantize`.

## 7. CUDA execution strategy for GTX 1080

The GTX 1080 is Pascal GP104 / SM 6.1. It should not be treated like an Ampere/Hopper tensor-core target.

For v1:

- keep weights in Q4/Q5/Q6/Q8 form in VRAM;
- dequantize inside the CUDA kernel into registers;
- use FP32 accumulation for correctness and good Pascal behavior;
- evaluate DP4A-style paths where they help Q8 or activation-dot products;
- avoid storing expanded FP16/FP32 copies of experts in VRAM;
- fuse dequantization with matvec/matmul rather than materializing a dequantized weight matrix.

The source quantization invariant does not prevent temporary FP32 arithmetic inside the kernel.

## 8. Three-tier residency model

### VRAM: execution-hot

Use VRAM for:

- dense hot tensors;
- current recurrent/QSA state and active KV working set;
- a dynamically sized expert cache;
- bounded PLE gather buffers;
- scratch / CUDA graphs or launch state.

Do not hardcode all remaining VRAM to experts. Long context increases state pressure, so expert-cache capacity must come from the runtime residency manager.

### System RAM: staging and second-level hot cache

Use host RAM for:

- a bounded compressed expert cache if profiling proves useful;
- pinned async I/O staging buffers;
- prefix/cache state;
- metadata and routing structures.

Do not let Windows file cache consume the entire PLE table. The PLE path should use explicit bounded caching and uncached/low-pollution I/O where available.

### NVMe: authoritative cold store

Keep:

- all expert records;
- all PLE pages;
- any dense tensors not selected for higher residency.

The NVMe representation is already execution-quantized. There is no decode-to-FP16 storage stage between disk and GPU.

## 9. Windows I/O path

The first robust implementation should prefer ordinary Windows primitives over a dependency on DirectStorage:

```text
NVMe
  -> overlapped ReadFile / IOCP
  -> aligned pinned host staging ring
  -> cudaMemcpyAsync
  -> CUDA stream
  -> fused quantized kernel
```

Design requirements:

- multiple outstanding expert reads;
- per-request cancellation/lifetime ownership;
- buffers aligned for the selected Windows file mode;
- optional no-buffering path when sector/alignment constraints are satisfied;
- coalesce adjacent expert records when it actually reduces I/O;
- never block the scheduler thread on storage completion;
- overlap layer-N compute with safe prefetch for future routed/storage work when dependencies permit.

DirectStorage can be investigated later, but it should not be a prerequisite for correctness or first usable CUDA inference.

## 10. Expert cache identity

A cached expert must be keyed by execution semantics, not only `(layer, expert)`:

```text
ExpertKey {
    model_fingerprint,
    layer,
    expert,
    representation/layout ABI,
    gate quant format,
    up quant format,
    down quant format,
    kernel ABI,
}
```

This prevents a future CUDA repack or kernel ABI change from accidentally reusing incompatible bytes.

## 11. Compilation provenance

The compiler should emit a machine-readable quantization provenance report as part of the package or alongside the manifest.

For every tensor/expert matrix record it should be possible to answer:

- source tensor name;
- source file fingerprint;
- source GGML quant type;
- source byte range;
- target math/quant type;
- transform: `identity`, `lossless_repack`, or `explicit_requantize`;
- source nominal weight bits;
- target nominal weight bits;
- logical CRC/hash of the source quantized payload where practical.

The verifier must fail a package when provenance claims `lossless_repack` but the source/target quant semantics differ.

The compiler must never use terminology such as "upquantized", "restored precision", or similar wording that implies information was recovered from a lower-bit source.

## 12. Default compilation policy

Default command behavior for an already-quantized GGUF should be equivalent to:

```text
quant_policy = preserve
expert_layout = per-expert bundled
ple_layout = aligned-page-packed
storage_codec = none for executable quant blocks
requantize = disabled
```

A future explicit flag may allow lower/equal-bit requantization, but it must be visibly opt-in and the compilation report must say which tensors changed.

There should be no flag that permits upward requantization of a quantized source tensor.

## 13. Correctness gates before enabling CUDA output

The Windows target should not be marked production-emittable until all of these pass:

1. GGUF tensor inventory matches source names, shapes, types and byte counts.
2. Source fingerprint is deterministic.
3. Expert slicing reconstructs the exact source matrix byte ranges.
4. Lossless expert bundle unpacking reproduces the original Q4_K/Q5_0/Q8_0 slices byte-for-byte.
5. PLE page packing reproduces every original 110-byte row byte-for-byte.
6. Quant precision policy rejects Q4->Q5/Q6/Q8 and analogous upward transitions.
7. CUDA Q4_K/Q5_0/Q6_K/Q8_0 kernels match a trusted CPU/GGML reference within quantized arithmetic tolerance.
8. End-to-end logits match the reference runtime before performance tuning.
9. Routed top-10 expert identities come from model metadata and match the reference implementation.
10. Cancellation cannot release pinned host buffers or VRAM leases before I/O/CUDA completion.

## 14. Implementation sequence

Recommended order:

1. Add native GGUF source inventory/parser to `logan-compiler` without dequantization.
2. Attach per-tensor `QuantFormat` to the compiler IR and enforce the precision policy before lowering.
3. Implement byte-exact expert slicing and per-expert bundle emission.
4. Implement PLE Q5_0 4 KiB page packing and parity tests.
5. Register `windows-x86_64-cuda-ggml-v1` as a non-emittable target while the runtime is incomplete.
6. Add CUDA runtime crate/FFI and direct quantized matvec kernels for Q4_K, Q5_0, Q6_K and Q8_0.
7. Add Windows IOCP/pinned staging and expert residency.
8. Turn on compiler emission only after byte parity + logit parity tests pass.
9. Benchmark and then consider CUDA-specific lossless repacks only where measured kernel gains justify the added ABI.

## 15. Non-goals for v1

Do not initially:

- convert all quantized tensors to one Logan INT4 format;
- expand experts to FP16 on disk or in the persistent VRAM cache;
- assume the `Q4_K_M` filename means every tensor is four-bit;
- make DirectStorage mandatory;
- use the Windows file cache as the PLE residency manager;
- create a Pascal-only on-disk layout that cannot be consumed by newer CUDA GPUs;
- silently lower precision during a normal compile.

The core principle is simple: **the source model's quantized information is authoritative; compilation changes layout and storage locality, not history.**
