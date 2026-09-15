# Logan Windows x86-64: build, CUDA and AVX2 status

**Status:** session record — all work uncommitted
**Date:** 2026-09-15
**Scope:** Windows x86-64 buildability, the CUDA Q4_K backend, the AVX2 kernels, and two pre-existing bugs fixed along the way.

---

## 1. Windows build status

`logan-qwen4`, `logan-chat` (and therefore `logand`) and the **whole workspace** now
build on Windows x86-64 with `--no-default-features`.

```
cargo build --release -p logan-qwen4 --no-default-features   # Finished
cargo build --release -p logan-chat  --no-default-features   # Finished
cargo build --release --workspace    --no-default-features   # Finished
target\release\logand.exe --help                             # runs
```

**Root cause.** Several non-Apple stub modules already existed and were correctly
cfg-gated, but each was **missing members that the (correctly ungated) call sites
use**. The fix was *completing the stubs*, not gating call sites — this matches the
repo's existing convention (logan-metal's header: "every entry point returns 0/None
on decline; the caller falls back to the CPU reference path"). No Apple code was
deleted or weakened.

| File | Gained |
|---|---|
| `logan-metal/src/lib.rs` | Non-macOS `imp`: `MetalAneFence` (uninhabited, never constructible) + `gdn_ane_token_begin` (24-arg macOS signature, returns `None`) |
| `logan-qwen4/src/gdn_ane.rs` | Non-Apple `imp`: `GdnAneDynamicPending` + `evaluate_layer_async`, `evaluate_layer`, `gpu_fence`, `gpu_surfaces` on `GdnAneDynamicEngine` |
| `logan-ane/src/unsupported.rs` | `impl AneModel { fn load() }` (was a bare unit struct; `cache.rs:71` calls `model.load()?`) |
| `logan-qwen4/src/bin/gdn_ane_e2e.rs` | `peak_rss_mib` split macos / non-macos |
| `logan-chat/src/bin/gdn_ane_greedy.rs` | same treatment |
| `logan-chat/src/bin/logand.rs` | comment only (its `getrusage` use was already block-gated) |

**Caveat.** `--no-default-features` is **required** on Windows: the ANE/Metal path is
Apple-only, so `cargo build -p logan-qwen4` (default features) fails by design.

**macOS unchanged.** `cargo check -p logan-qwen4` and `cargo check --workspace` both
`Finished` with no new diagnostics.

---

## 2. CUDA backend

Two new files:

- `logan-core/src/cuda.rs` — mechanism: `dlopen`/`LoadLibrary` + NVRTC compile-to-PTX.
  No Cargo dependency (no `cudarc`), fail-closed. 5 unit tests, all env-independent.
- `logan-qwen4/src/ggufsource/cuda_q4k.rs` — the Q4_K GEMV kernel, next to the
  `dot_row` oracle. Declared at `logan-qwen4/src/ggufsource.rs:969` (`pub mod cuda_q4k;`).

**Correctness: bit-exact vs `dot_row`** — `0e0` max absolute and relative difference,
verified on the real **GTX 1080**.

**Opt-in via `LOGAN_CUDA=1`; default OFF** (accepts `1`/`true`, case-insensitive).

**Why default OFF:** measured **0.34–0.58× of the 12-thread CPU oracle** at
o=640/i=2560 and o=2048/i=2560 — i.e. **2–3× SLOWER** than the CPU path it replaces.

**Mechanism (row kernel):** one CUDA thread per output row, serial accumulation chain,
~15% occupancy. Weight upload is only 8–12% of runtime, so **PCIe is NOT the cause** —
the kernel itself is the bottleneck.

**Tiled kernel result:** 1.05–1.46× faster than the row kernel, flat µs/row across `o`,
but still only **0.32–0.58× of CPU**. So it does not rescue the backend.

**Diagnosed next step.** A 2560-value row is 80 groups, so a 256-thread block uses
only 80 threads and the shared reduction now costs as much as the work it combines.
Fix by giving **each thread several groups**, or **sizing the block to the group
count** — *not* by raising thread count.

---

## 3. AVX2 status

`logan-core/src/math_x86.rs` kernels are **correct and do engage**. Verified with
temporary instrumented atomics on Zen 3 (counters since removed); measured against an
independent scalar reference:

| shape | path | scalar | AVX2 | speedup | max abs vs scalar ref |
|---|---|---|---|---|---|
| o=640 i=2560 | bf16 | 1.687 ms | 0.559 ms | **3.0×** | 7.82e-5 (1.71e-6 of result scale) |
| o=640 i=2560 | f32 | 1.965 ms | 1.171 ms | **1.68×** | 9.16e-5 (2.00e-6 of result scale) |

Differences match f32 reassociation exactly (predicted 1.7e-6, observed 1.71e-6);
the scalar arm is bit-identical to the reference. Not a bug.

> **⚠ Open architectural issue (not fixed).**
> `logan_core::math::matmul` has **zero production callers**. `logan-qwen4/src/lib.rs`
> defines its **own private `matmul` with its own `WtBytes` enum**, and *that* is what
> actually runs. The AVX2 kernels are therefore currently **unreachable from
> production**, and there is a **duplicated matmul implementation** between
> `logan-core` and `logan-qwen4` that will drift. This was left alone deliberately.

---

## 4. Two pre-existing bugs fixed

**(a) `::` in temp filenames — 3 tests, Windows only (`Os { code: 123, InvalidFilename }`).**

Tests built a temp path from `std::thread::current().name()`, which under `cargo test`
is the full test path (`tests::foo`) and contains `::`, illegal on Windows.

Fixed with one shared helper, `logan_format::test_support::test_temp_path`, used at the
**3 broken call sites**: `logan-qwen4/src/lib.rs` ×2 and
`logan-compiler/src/optimize.rs` (plus 2 self-tests inside the helper module). It maps
every Windows-illegal character to `_`, bounds the component to 96 chars, and appends an
FNV-1a hash of the *raw* name so distinct tests cannot collide after sanitisation.

It is a plain `pub fn`, **not `#[cfg(test)]`**, because `cfg(test)` does not apply to an
external crate's tests.

**(b) `logan-core`'s `--release` test target did not build at HEAD (11× E0599).**

`SchedulerCore::assert_invariants` was gated `#[cfg(debug_assertions)]` while its test
callers are not — and `cargo test --release` compiles tests with `debug_assertions`
**off**. Gate changed to `#[cfg(any(debug_assertions, test))]`.

Proof that production still excludes it: `nm -C` on the **release rlib** gives **0**
symbols matching `sched::core::SchedulerCore::assert_invariants` (debug rlib: 6). The
single `assert_invariants` symbol present in the release rlib is
`sched::residency::ResidencyManager::assert_invariants` — a different, separately gated
function.

**Bonus (not in the original report):** a third Windows-only bug,
`recompile::tests::low_space_in_place_retarget_...` failing with `Os { code: 5,
PermissionDenied }`. Two Windows-only atomic-replace paths in
`logan-compiler/src/recompile.rs` (`write_json_synced`, `replace_manifest`) did
`fs::copy(next, path)` then `File::open(path).sync_all()`. On Windows `sync_all` is
`FlushFileBuffers`, which requires GENERIC_WRITE; a read-only handle fails permanently.
Reproduced standalone (`File::open(..).sync_all()` → code 5; write handle → OK).
Fixed by extracting one `#[cfg(windows)] fn commit_replaced_file` that opens for write.

---

## 5. Known-broken / surprising commands

- `cargo test --release -p logan-core --lib` — **broken before fix (b), works after.**
  On macOS prefer the debug profile for `logan-core` if anything looks odd.
- `cargo build -p logan-qwen4` **without** `--no-default-features` — fails on Windows
  by design (Apple-only ANE/Metal).
- `cargo test --release --no-default-features -p logan-qwen4 --lib` on Windows: **54
  passed / 0 failed / 1 ignored** (was 52/2 before the temp-filename fix).
- Windows `set VAR=0 && cmd` appends a **trailing space**, so `QWEN_NEON_BF16` becomes
  `"0 "` which `!= "0"` and the opt-out silently does not engage. Use `set "VAR=0"`.

---

## 6. Attribution: this session's work vs pre-existing

This section records what this session actually authored, which is what the session's
commit contains. At the time of the audit `git status --short` reported **27 files
modified and 6 untracked entries** (4 files plus the directories
`.qwen3-coder-next-scratch/` and `logan-qwen4/src/ggufsource/`).

> **Correction — read this before trusting any earlier revision of this section.**
> The first revision of §6 under-counted this session's contribution. It was written
> by the `avx2-windows-verify` worker, which had no visibility into the
> `logan-cuda-matmul` worker editing `logan-qwen4/src/lib.rs` **concurrently**, and so
> mis-attributed that file's CUDA hunks to "concurrent/other work". They were
> concurrent, but they were still this session's. In particular the earlier claim
> that "only the two `test_temp_path` edits are this session's" in `lib.rs` is
> **wrong**: all four of its hunks are this session's. The lists below supersede it.

**Committed as one commit** (wholly this session's work):

| Path | What this session did |
|---|---|
| `logan-core/src/cuda.rs` (new) | NVRTC + `dlopen`/`LoadLibrary` CUDA mechanism, no Cargo dependency |
| `logan-qwen4/src/ggufsource/cuda_q4k.rs` (new) | Q4_K GEMV kernel, next to the `dot_row` oracle |
| `logan-format/src/test_support.rs` (new) | shared `test_temp_path` helper |
| `logan-ane/src/unsupported.rs` | `AneModel::load` (`cache.rs:71` calls it) |
| `logan-metal/src/lib.rs` | non-macOS `MetalAneFence` + the 24-arg `gdn_ane_token_begin` |
| `logan-qwen4/src/gdn_ane.rs` | non-Apple `GdnAneDynamicEngine`/`GdnAneDynamicPending` stub block (one hunk; the file's large rustfmt reflow is not this session's) |
| `logan-qwen4/src/ggufsource.rs` | `pub mod cuda_q4k;`, `q4k_scale_min` → `pub(crate)`, `dot_row` doc (3 hunks; its rustfmt-only hunks are not this session's) |
| `logan-qwen4/src/lib.rs` | **all four hunks**: CUDA early-return in `matmul` (~line 1976), two `test_temp_path` edits, ~481-line CUDA test block (`+507 / -10` total) |
| `logan-qwen4/src/bin/gdn_ane_e2e.rs` | `peak_rss_mib` split macos / non-macos |
| `logan-chat/src/bin/gdn_ane_greedy.rs` | same treatment |
| `logan-chat/src/bin/logand.rs` | comment only |
| `logan-core/src/sched/core.rs` | `assert_invariants` gate widened to `any(debug_assertions, test)` |
| `logan-core/src/lib.rs` | `pub mod cuda;` (only — see below) |
| `logan-format/src/lib.rs` | `test_support` module + `test_temp_path` re-export |
| `logan-compiler/src/optimize.rs` | temp-path fix |
| `logan-compiler/src/recompile.rs` | `commit_replaced_file` (the `sync_all`-on-read-only-handle bug) |
| `docs/WINDOWS_CUDA_AND_X86_STATUS.md` | this document |

> **⚠ A reviewer MUST NOT attribute the whole working-tree diff to this session.**
> The tree also contains substantial **pre-existing uncommitted work by others**
> (Metal/ANE, prefix-cache and session work) which this session deliberately did not
> touch, and which the commit therefore does **not** contain:
> `logan-qwen4/src/coliload.rs` (+325), `colisource.rs` (+110), `ggufload.rs` (+458),
> `mtp.rs`, `main.rs`, `plan/prefix_runtime.rs` (+215), `plan/runtime_stats.rs`,
> `scheduled.rs`, `examples/dump_ple.rs`, `bin/prefix_snapshot_gate.rs`,
> `bin/ssd_prefix_gate.rs`, `README.md`, `docs/logan_model_format_v1.md`, and
> `.qwen3-coder-next-scratch/`.
>
> `logan-qwen4/src/gdn_ane.rs`'s diff is ~571 lines but only one stub impl-block is
> this session's; the rest predates it. `logan-qwen4/src/ggufsource.rs` likewise
> carries unrelated rustfmt reflow.

**Left uncommitted although this session touched it — `logan-core/src/math.rs`.**
`math_x86.rs` is *earlier*-session work (see §3), still untracked when this session
began. This session's `math.rs` additions (the module header, the f32 AVX2 arm, its
two tests) call `crate::math_x86`, so committing `math.rs` without `math_x86.rs` would
not build. Rather than sweep an earlier session's untracked file into this commit, the
whole `logan-core` AVX2 surface — `math.rs`, `math_x86.rs`, and the
`#[cfg(target_arch = "x86_64")] pub mod math_x86;` line in `logan-core/src/lib.rs` —
was left uncommitted. The AVX2 *findings* this session established are recorded in §3
and are not lost.

---

## ⚠ HEAD is broken: `logan-qwen4` does not compile at HEAD

**This is a pre-existing condition, not something this session introduced.** It was
discovered while verifying this session's commit in a pristine worktree at HEAD, and it
is recorded here because it will waste a future session's afternoon otherwise: **anyone
bisecting, branching from, or clean-building HEAD will find `logan-qwen4` does not
compile.**

**Symptom.** `cargo check --workspace` at HEAD fails with **17 errors**:
13–16 × E0425/E0422 `cannot find type/struct MtpRuntime, MtpStats, MtpDraft,
MtpVerifyBatch, MtpVerifyBoundary in crate::mtp`, in `logan-qwen4/src/lib.rs`, plus
E0063 missing-field errors in `logan-qwen4/src/coliload.rs` and `ggufload.rs`.

**Mechanism.** The MTP feature is **half-landed in HEAD** (commits `8a520e9` /
`6950297`). HEAD's own committed `logan-qwen4/src/lib.rs` declares
`Model.last_hidden_nextn` and *uses* `crate::mtp::MtpRuntime`, `MtpStats`, `MtpDraft`,
`MtpVerifyBatch` and `MtpVerifyBoundary` — but HEAD's own committed `mtp.rs` **does not
define them**. HEAD's `coliload.rs` likewise does not set `Model`'s new fields, and
HEAD's `ggufload.rs` does not set `Cfg.zero_centered_norm`. The feature is completed
only by three files that are **uncommitted**: `mtp.rs` (+75), `coliload.rs` (+325) and
`ggufload.rs` (+458).

| Check | Result |
|---|---|
| `cargo check --workspace` on a pristine worktree at HEAD | **17 errors** |
| `cargo check --workspace` on HEAD + this commit's 17 files | **the same 17 errors**, byte-identical; only line numbers shift |
| adding *just* those 3 uncommitted files (`mtp.rs`, `coliload.rs`, `ggufload.rs`) | **17 errors → 0** |

**Consequence for this commit.** The session's commit neither causes nor worsens the
breakage: every error site is byte-identical to HEAD's. It is not ours to fix — fixing
it would mean committing another session's in-flight MTP work, which was deliberately
not done. It is recorded here **for the user to act on**.

---

## 7. Next steps

1. **CUDA tiled kernel:** give each thread several groups, or size the block to the
   group count (80 for a 2560-value row). Do *not* simply raise thread count.
   Re-evaluate whether CUDA is worth keeping at all: currently 0.32–0.58× of the
   12-thread CPU oracle, and default OFF.
2. **Duplicated matmul:** reconcile `logan_core::math::matmul` with
   `logan-qwen4`'s private `matmul`. Until then the AVX2 kernels (3.0× bf16) are dead
   code in production.
3. **Commit this session's work**, separated from the pre-existing in-flight changes,
   so the next session does not re-derive the above.
4. **Unreproduced flake:** one run of `cargo test -p logan-core --lib` reported
   `82 passed; 1 failed` on macOS; not reproduced in 48 subsequent runs. The core test
   binary now includes 5 new CUDA tests that probe for a driver and compile kernels at
   runtime; a plausible (unconfirmed) cause is machine load from concurrent builds.
5. `logan-compiler/src/recompile.rs` fix is in the **committed** file, so it will show
   up as a normal diff — it is not in-flight work.

---

## 8. Assessment: is a GPU worth it for this workload?

**This corrects §7.1**, which frames CUDA as a kernel to fix and then re-evaluate. The
measurements support something stronger: for *this* model on *this* card the GPU premise
fails on **design** grounds, not just kernel quality.

**Measured position.** Bit-exact against the `dot_row` oracle (0e0 max abs and rel,
verified on the real GTX 1080) and nonetheless **0.34–0.58× of the 12-thread CPU oracle**
at o=640/i=2560 and o=2048/i=2560 — i.e. **2–3× slower**. Hence opt-in via `LOGAN_CUDA=1`,
default OFF.

**Premise, and why it does not hold here.** The case for the GPU was bandwidth: ~352 GB/s
VRAM against ~35–45 GB/s DDR4, so weights resident in VRAM would be ~8× cheaper to
re-read. That is **conditional on the weights fitting**. This model is
Qwen3.8-Flash-Next Q4_K_M with 512 experts per layer across 48 layers; the expert set is
far larger than the card's 8 GiB. The GPU therefore cannot act as a resident cache for
this workload — at best a small cache with an eviction policy, where every miss pays a
PCIe transfer on top of the kernel. **The premise fails for this model on this card.**

**Where the time goes.** Weight upload is only 8–12% of the call, so PCIe is not the
bottleneck. GPU time scales cleanly with per-thread serial work:

| `i` | 256 | 512 | 1280 | 2560 |
|---|---|---|---|---|
| ms | 0.185 | 0.285 | 0.590 | 1.080 |

exactly proportional — the kernel is **work-bound, not overhead-bound**. A block-size
sweep (64/128/256/512 threads) was **flat** (0.864 / 0.863 / 0.866 / 0.878 ms),
disproving the occupancy/reduction hypothesis. Launch+sync+DtoH is 0.027 ms. The limit is
the **per-thread dependency chain**: not parallelism, not transfer, not launch overhead.

**Conclusion.** For a memory-bound quantized MoE workload whose weights do not fit in
VRAM and are streamed from NVMe, **12 CPU cores with AVX2 is a reasonable and currently
FASTER backend**. The GPU path stays available, correct and opt-in — an **experiment with
a recorded negative result, not a pending optimisation**. A future attempt needs a
genuinely different kernel (several groups per thread to break the dependency chain, plus
overlapping transfers rather than a full `cuCtxSynchronize` + DtoH per call) **AND** a
plan for resident weights — otherwise it tunes something whose premise does not hold.

**Do not delete it.** Keep the code and the opt-in: it is bit-exact, the NVRTC/dlopen
loading mechanism is reusable with no Cargo dependency and no toolkit, and it is worth
keeping for a workload where experts **do** fit, or for a card with more VRAM. Do not
remove it.
