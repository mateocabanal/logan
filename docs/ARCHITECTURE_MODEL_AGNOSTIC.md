# Logan Model-Agnostic Runtime Architecture

Last verified: 2026-09-19

## Status

Logan now has a compiled, tested model-neutral causal-state and prefix-cache layer in
`logan-core`. Engines still own the semantics and physical representation of their
causal state through small codec adapters.

The important boundary is:

- **core owns policy and lifecycle**: identity, prefix matching, RAM/SSD policy,
  checksums, transactions, persistence containers, cache budgets, and telemetry;
- **engines own state semantics and performance-critical codecs**: how live KV,
  recurrent state, QSA/GDN state, or other model-specific state is captured and
  restored.

This is intentionally not a requirement that every model use the same physical
serialization algorithm.

## Shared Core

| Subsystem | File(s) | Responsibility |
|---|---|---|
| Causal-state abstraction | `logan-core/src/state/mod.rs` | `CausalState`, state patterns, `CausalStateCodec`, schema identity |
| State schemas | `logan-core/src/state/schema.rs` | Logical region geometry, dtype, validation, state-size accounting |
| State pages | `logan-core/src/state/page.rs` | Page identity/reference/pinning primitives |
| State transactions | `logan-core/src/state/transaction.rs` | Generation-safe checkpoint, commit, rollback, cancellation |
| State snapshots | `logan-core/src/state/snapshot.rs` | Versioned checksummed causal-state snapshots and codec capture/restore |
| Prefix identity | `logan-core/src/prefix/key.rs` | Model/state/tokenizer/plan identity and stable token hashing |
| Prefix index | `logan-core/src/prefix/index.rs` | Exact token-prefix matching, longest-prefix selection, LRU accounting |
| RAM prefix cache | `logan-core/src/prefix/memory.rs` | Byte-budgeted hot cache storing real snapshot values |
| Generic SSD store | `logan-core/src/prefix/disk.rs`, `format.rs` | Atomic checksummed persistence and restart-safe token/state identity |
| Prefix runtime | `logan-core/src/prefix/runtime.rs` | RAM/SSD lookup, promotion, persistence, budgets, cache isolation |

The generic persistent prefix container is `LOGANPF1`. It stores the actual token
sequence plus a versioned `StateSnapshot`; a hash alone is never treated as proof
that one sequence is a prefix of another.

## Engine Boundary

### Dense Llama / MiniCPM

`logan-llama/src/codec_adapter.rs` implements `LlamaStateCodec`.

The dense chat path uses:

1. the generic `RamPrefixCache` for hot snapshots plus boundary logits;
2. the generic `PrefixRuntime` for persistent SSD state;
3. exact model, tokenizer, state-schema, and numerical-plan fingerprints;
4. strict-prefix SSD restore, followed by a real suffix token to regenerate boundary
   logits. No synthetic token position is replayed.

`DenseModel::cache_fingerprint()` binds persistent state to the effective resident
model, including model geometry and weight representation.

### Qwen4 / Qwen4Exp

`logan-qwen4/src/plan/snapshot.rs` implements `QwenStateCodec` and exposes the
core-backed `QwenStateSnapshot` used by the hot-cache path.

Qwen also deliberately retains its specialized streaming persistent `.lpfx` codec in
`logan-qwen4/src/plan/prefix_cache.rs`. That is an engine-specific physical codec,
not a separate architectural policy layer. Qwen hybrid state can be very large
(GDN, attention KV, QSA, PLE), so direct validated streaming restore avoids allocating
a second full in-memory state image.

The specialized Qwen restore now validates exact payload size, decodes little-endian
values without unaligned typed-pointer casts, and restores attention KV using the
correct per-head stride.

## State Patterns

Core supports reusable logical state patterns:

- `AppendOnly` — conventional KV-like state;
- `Ring` — sliding-window/circular state;
- `MutableFixed` — fixed-size recurrent or compressor state;
- `SparsePaged` — page-oriented sparse state;
- `Opaque` — engine-owned compound state that must be captured/restored atomically.

`Opaque` is intentional: model-agnostic lifecycle management does not require core
to understand every engine's internal tensor layout.

## Cache Identity and Correctness

A generic prefix key includes:

- model fingerprint;
- state-schema fingerprint;
- tokenizer fingerprint;
- execution/numerical-plan fingerprint;
- the exact prefix token sequence;
- a stable SHA-256-derived token hash for indexing.

Longest-prefix lookup compares the real token sequence. Two different-length prefixes
are not expected to have equal hashes.

Generic SSD entries use:

- a versioned container;
- exact token storage;
- state snapshot checksums;
- atomic temp-file + fsync + rename publication;
- byte-budgeted eviction;
- restart-time index reconstruction.

State transactions are session-bound and generation-bound so a checkpoint from a
different session or an older committed state cannot become valid accidentally.

## Verification

The exact integrated tree was verified on 2026-09-19 with:

```text
cargo fmt --all -- --check
PASS

git diff --check
PASS

cargo test --workspace --all-targets
PASS (exit 0)
```

Important targeted regressions that pass include:

- Qwen head-major KV restore uses the real per-head stride and preserves unused tail;
- Qwen byte restore accepts unaligned byte sources safely;
- MiniCPM exact-prompt RAM cache restore returns the exact committed causal position;
- generic SSD cache survives runtime reconstruction and promotes restored entries;
- state rollback restores exact state and rejects stale/cross-session checkpoints.

### Real MiniCPM5 checks

The installed `MiniCPM5-2B-oQ8e` checkpoint was run through the release Logan
dense path. A one-token decode completed on the Metal backend:

```text
backend=Auto
tokens=1
tok_s=1.425
used=Metal
reason="fused standard Llama layer"
```

A separate release-mode real-model SSD test used a 2-token prefix and 3-token
continuation, explicitly cleared the RAM hot cache, and verified the persistent state
restore:

```text
real_ssd_prefix=ok
first_tokens=2
second_tokens=3
ssd_hits=1
ssd_misses=1
live_tokens=3
```

This demonstrates that the dense generic SSD path is not only a synthetic/unit-test
implementation.

## Deliberate Remaining Specialization

Model-agnostic does **not** mean model-identical.

It remains appropriate for an engine to specialize:

- state capture/restore layout when copying a generic snapshot would be prohibitively
  expensive;
- attention or recurrent-state mathematics;
- router/expert behavior;
- tokenizer/chat-template handling;
- Metal/CUDA/ANE kernels;
- speculative drafter conditioning and verification geometry.

Those pieces should remain behind the engine boundary while lifecycle, identity,
cache policy, transactions, and resource management stay reusable.

## Remaining Qualification Work

The following are not claimed by this verification pass:

- a full real-model Qwen4Exp persistent-SSD restore A/B on a large production
  `.coli` checkpoint;
- real-model DSpark/MTP speculative performance qualification;
- before/after TTFT and throughput benchmarks for the cache refactor.

These are performance/engine qualification tasks rather than blockers for the shared
state/cache architecture.

## Integration History

- `ce5b657` — original model-agnostic runtime refactor base.
- `11fe4df` — verified repair integrating state/cache correctness fixes and dense
  Llama/MiniCPM adapters.
- The repair is merged onto `main`; see Git history for the merge commit and this
  documentation update.
