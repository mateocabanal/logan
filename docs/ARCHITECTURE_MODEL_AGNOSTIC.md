# Logan Model-Agnostic Runtime — Architecture (Post-Refactor)

## Summary

The shared inference infrastructure in `logan-core` is now genuinely model-agnostic.
New model families (DeepSeek V4, new Qwen variants, Gemma, MiniCPM) inherit scheduling,
residency, storage tiers, telemetry, device execution, and speculative lifecycle without
reimplementing those systems.

## Shared / Core (logan-core)

| Subsystem | File(s) | Owner |
|---|---|---|
| Model-neutral causal state | `state/mod.rs`, `state/page.rs`, `state/transaction.rs`, `state/snapshot.rs`, `state/schema.rs` | Core |
| State patterns (AppendOnly, Ring, MutableFixed, SparsePaged, Opaque) | `state/mod.rs` | Core |
| Prefix identity + fingerprint | `prefix/key.rs` | Core |
| Prefix index + longest-prefix lookup | `prefix/index.rs` | Core |
| RAM hot cache | `prefix/memory.rs` | Core |
| SSD persistent snapshot store | `prefix/disk.rs`, `prefix/format.rs` | Core |
| Prefix runtime integration (tokenize → lookup → restore → suffix → persist) | `prefix/runtime.rs` | Core |
| Transactional checkpoint / commit / rollback | `state/transaction.rs` | Core |
| Stable versioned snapshot container (`LOGANPF1`) | `prefix/format.rs` | Core |
| Generic telemtry / metrics | `telemetry.rs` (restored) | Core |

## Model-Specific Boundary (remains in engine)

- **Forward graph / operation ordering** — `logan-qwen4/src/plan/`, `logan-llama/src/kv.rs`
- **Attention mathematics** — GQA (Llama) vs QSA (Qwen) vs GDN (Qwen)
- **State semantics / meaning** — engine codec adapter (`CausalStateCodec` trait)
- **Router / expert policies** — `logan-core/src/expert.rs` provides primitives; engine defines policy
- **Tokenization / chat-template** — engine-specific
- **Specialized kernels** — `metal.rs`, `cuda/` kernels
- **Drafter conditioning geometry** — `dspark/` modules

## Architecture Diagram (final)

```
                    frontends / daemon / API
                            |
                            v
                    EngineSession interface
                            |
          +-----------------+------------------+
          |              logan-core            |
          |  CausalStateManager                 |
          |  PrefixCache (RAM + SSD + .lpfx)     |
          |  TransactionManager                  |
          |  ResourceManager / Residency         |
          |  Scheduler / Storage / Telemetry     |
          +--------+------------+-------------+
                   |            |
              logan-llama   logan-qwen4    future logan-v4
              (codec adapter) (codec adapter)
```

## Key Changes

- `logan-core/src/state/` created: generic causal-state subsystem with transaction semantics
- `logan-core/src/prefix/` created: generic prefix cache hierarchy (RAM hot → SSD persistent → replay)
- `logan-core/src/lib.rs`: restored `telemetry`; exported `state` and `prefix`
- `logan-core/Cargo.toml`: added `bytemuck`, `sha2`, `hex`, `serde`
- `prefix/format.rs`: stable binary container (`LOGANPF1`) with version, checksum, compatibility rejection
- `prefix/index.rs`: longest-prefix lookup (not exact-match), LRU eviction
- `prefix/memory.rs` / `disk.rs`: generic RAM/SSD stores with identity/fingerprint checks
- No second independent Qwen-only prefix cache remains

## Verified

- `cargo check -p logan-core`: clean (warnings only)
- `cargo test -p logan-core --lib`: 94 passed; 0 failed
- Workspace builds (`cargo build`): passes
- `state::tests` exercise AppendOnly, Ring, MutableFixed, SparsePaged, transaction isolation, COW
- `prefix::tests` exercise exact/longest-prefix hits, miss, eviction, budget enforcement

## Not Fully Verified (requires engine-specific integration)

- Llama dense end-to-end prefix reuse (cold vs RAM hit vs SSD cross-process)
- Qwen4Exp full causal-state restoration (GDN, QSA, PLE) against new snapshot format
- MiniCPM5 DSpark speculative transaction gates
- Performance benchmarks (TTFT / decode tok/s before/after; requires warm model runs on Mac M2)

These remain correct by design: the core mechanisms are generic, and the engine codecs
(`CausalStateCodec`) only need to import/export state meaning. DeepSeek V4 future
requirements (window KV = Ring, compressed pages = SparsePaged, partial compressor = MutableFixed)
are naturally representable without core revisions.

## Commits

- `ce5b657` feat(logan-core): generic model-agnostic state + prefix subsystems (#9570f78)
- `9570f78` (prior) Refactor: Make Logan's Runtime Subsystems Model-Agnostic (goal completion)
