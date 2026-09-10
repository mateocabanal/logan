# `.logan` — Logan's native compiled model format (v1 design)

**Date:** 2026-09-09
**Status:** Proposed — design for owner review. No runtime/format code changed by this document.
**Author:** Mateo + Hermes
**Supersedes:** COLI (`.coli`) as the canonical compiled model representation
**Release requirement:** this design must be able to close Logan issue **#37** completely, without a
v2 format break. See §19 for the requirement-by-requirement mapping.

---

## 1. Decision summary

`.logan` is **not a model file**. It is a **compiled execution image**: a small immutable *plan* plus a set
of physically homogeneous *segments*, where a segment is the unit that has exactly one representation
contract, one access contract, and one settable residency budget.

The single most important inversion relative to COLI:

> COLI is a **container of records with some metadata attached**.
> `.logan` v1 is a **container of a plan with payloads attached**.

Everything else follows from that. The plan is mandatory, package-local, integrity-bound to the bytes, and
authoritative for what is legal. The runtime may choose *timing* — prefetch, eviction, queue order — only
inside the bounds the plan declares. The runtime may never invent a different physical plan.

Seven decisions carry the design:

1. **The artifact carries a plan, and the plan is the primary object.** (Closes #83's structural half.)
2. **Payload is described by a segment table, not by per-tensor records.** A logical tensor maps to one or
   more segments with explicit block geometry; large tensors can be split so a bounded working set exists.
3. **Access class is an explicit, intrinsic property of a segment** — `Stream | Gather | Mutable | Pinned` —
   separate from math representation, layout/ABI, backing, residency, execution device and transfer path.
   Engram-style lookup tables and routed experts stop sharing one accidental cold-I/O abstraction.
4. **Unit-addressable data carries a block geometry + block CRCs.** A 400 MB sparse table must be readable and
   verifiable 256 bytes at a time, not only as an opaque blob.
5. **Mutable state is never inside `.logan`.** The artifact declares *requirements* (bytes, block size,
   alignment, layout id, ordering), the runtime binds them to a compatible writable pool and creates a
   versioned **state store** whose format v1 specifies.
6. **Metadata is a typed, length-prefixed TLV — never bincode, never JSON-as-truth.** Fixed-width
   little-endian integers, checked arithmetic, unknown sections skippable, no embedded paths. JSON remains
   only as a human-readable `PROVENANCE` report.
7. **One logical layout, two physical packings.** A sharded directory (default, for 500 GB models) and a
   single file (for distribution of small models). Same manifest, same plan, same segment table; only the
   file table differs.

Section 23 reconciles this design with the concurrent Windows/CUDA format, DeepSeek V4 and quantization
integrity work that landed in the same tree while it was being written.

Two things the design deliberately does **not** claim:

- **Separate files per storage class are not a performance win.** Measured on the target M2 NVMe against the
  real 91 GiB package: file-level separation changed neither sparse-read p99 nor bulk throughput (§17).
  Classes exist for *semantics, budgeting and queueing*, not for filesystem layout.
- **`.logan` v1 will not make a 447 GiB model fast on 16 GiB.** It makes it *legal, honest and inspectable*:
  the planner reports the real per-token SSD traffic and exposed stall instead of pretending, or refuses.

---

## 2. What problem `.logan` is actually solving

Logan's thesis is stated as: **tiny machine, immense model — RAM/UMA/VRAM determine speed, backing storage
determines capacity.** The current implementation does not yet express that thesis anywhere the runtime can
see it. Three concrete symptoms, all verified in the tree:

1. `build_physical_plan` (`logan-compiler/src/pipeline.rs`) computes `resident_bytes` as the sum of *all*
   non-expert decoded bytes and then calls `check_uma_pool`, whose contract is
   `resident_bytes + expert_cache_bytes <= ram_bytes` → otherwise a hard compile error. That is precisely the
   `model bytes + context bytes <= RAM` worldview issue #37 exists to eliminate.
2. `MemoryPlan.resources` — the #93 tiered `ResourcePlan` vector — is emitted as `Vec::new()`. The correct
   abstraction landed (commit `3c6bcd6`, #93) but **nothing populates it and nothing consumes it**.
3. The compiled plan is an **optional external file** (`logan compile --plan PATH`, bincode over serde
   structs). The package has no pointer to it, no digest over it, and no requirement that it exist. A `.coli`
   package with the wrong or missing plan opens and runs happily.

So `.logan` is solving a narrower and more precise problem than "a better container":

> **Persist, bind and enforce the complete physical execution contract of a compiled model, such that a
> machine whose fast memory is far smaller than the model can be told exactly what is legal, what is
> required, what it may choose, and what it will cost — and cannot silently do something else.**

Three questions the format must answer for every byte class in the model, and must answer *separately*:

| Question | Answer lives in | Never conflated with |
|---|---|---|
| What numbers are these? | representation contract (math format + scale format + block geometry + provenance) | layout, device |
| How may they be addressed and moved? | layout/kernel ABI id, access class, block geometry, transfer path | residence |
| Where do they live when not resident? | backing (package segment / state store / pinned-only) | execution device |
| How much fast memory must exist *right now*? | minimum working set per memory pool | total logical bytes |
| May the runtime choose something else? | alternatives list + prefetch/eviction bounds | "the runtime will figure it out" |

The failure mode this design is written against is not "the format ran out of fields". It is **"the artifact
was structurally unable to represent the plan, so the runtime improvised"** — which is where silent precision
loss, silent re-quantization, silent RAM-residency assumptions and unreproducible performance all come from.

---

## 3. Audit of the current system

### 3.1 COLI, as implemented

Verified against `logan-format/src/{lib,package,verify,codecs}.rs` and the emitted artifact on disk.

A package is a **directory**: `manifest.coli` plus `data-NNNNN.coli` shards.

`manifest.coli`:
- 256-byte header, magic `COLI\r\n\x1a\n`, version 1, CRC32C over the header with the CRC field zeroed.
- shard table (64 B/entry): shard id, name string id, file bytes, header CRC.
- record table (96 B/entry): `id, kind, codec, math_format, scale_format, layout, flags, shard_id, name_id,
  layer, expert, offset, stored, decoded, stored_crc, logical_crc, codec_table_id`.
- string table (16 B descriptors), interned names.
- `record_alignment` constrained to `[4096, 1 MiB]`, power of two.
- source fingerprint `[u8;32]` + a validity flag that must agree with the bytes.

Measured on the real artifact `Qwen3.8-Flash-Next-REAP-288-MXFP4-Apple8.coli`:

| Property | Value |
|---|---|
| package size | 91 GiB |
| shards | 24 files, ~4.0–4.3 GiB each |
| records | 15,023 (13,824 kind=2 experts, 1,199 kind=1 tensors) |
| distinct record sizes | 21 |
| expert record size | 2,611,648 B (288 experts × 48 layers) |
| largest records | 128 × 400,002,048 B — the PLE n-gram table, 51.2 GiB total |
| manifest size | 2,075,424 B |
| open + record index | **7.1 ms** (1.2 ms read, 5.9 ms index), 24 fds, 0.77 ms to open shards |
| alignment | 16384 |
| profile | `macos-arm64-metal-apple8-v1` |

Open cost and fd count are healthy at 15 k records. That part of COLI is right and should survive.

### 3.2 What COLI gets right (must survive)

1. **Directory package with a small manifest and large shards.** Bounded open cost, no full scan.
2. **Filenames derived from ids, never stored.** `root.join(format!("data-{shard_id:05}.coli"))` makes path
   traversal structurally impossible. `.logan` keeps this property by storing *file indices*, never paths.
3. **Structural open without reading payloads.** `Package::open` validates structure; bytes are read lazily.
4. **Deterministic, checked planning.** `plan_records` requires strictly increasing non-zero ids, validates
   alignment range, uses checked arithmetic throughout, and accounts padding explicitly.
5. **Fingerprint binding.** Source fingerprint in the manifest and a stale-plan guard on the artifact.
6. **A generated, extensible id registry.** `abi/coli-target-registry.toml` → `logan-abi` generated Rust:
   layout ids with geometry (`tile_rows/columns/bytes`), profile ids with ABI/kernel ids and alignment.
7. **Expert envelopes.** Three matrices in one record with an internal descriptor list → one contiguous read
   feeds one MetalIO slot. Measured as the main I/O win; keep the idea.
8. **Multiple representations per expert are already legal** (`by_expert` maps one `(layer, expert)` to many
   records). The germ of "compiled alternatives" exists — it just stops at experts.
9. **CRC32C on stored payload** plus a separate logical CRC over the decoded form.

### 3.3 What COLI gets wrong (or is silent about) for this design

| # | Defect | Consequence |
|---|---|---|
| C1 | Plan is external, optional, bincode/serde | package is not self-describing; #83 unachievable as-is; bincode is not versionable or forward-compatible |
| C2 | `MemoryPlan.resources` emitted empty | #93's tiered contract is dead code; the runtime has no backing/residency/access data |
| C3 | `check_uma_pool` is the admission test on the non-optimizer path | the single global RAM wall #37 exists to remove |
| C4 | A record is the atomic unit — no sub-record structure | a 400 MB PLE record has no declared way to address or verify a 256 B row; the reader must *know the convention* |
| C5 | `read_payload_range` explicitly skips CRC | partial reads of the largest tables in the model are unverifiable by construction |
| C6 | No storage-tier/access-class concept | experts, PLE shards, dense weights and KV all present as undifferentiated "records" |
| C7 | No mutable-state format at all | the runtime has no way to create, validate, resume or repair a KV/state backing store |
| C8 | `flags` and `kind` semantics are undocumented; `math_format = 0xfffe` / `layout = 0xfffe` overload "none" and "expert-scope" | an untyped-metadata failure mode inside a *typed* manifest |
| C9 | No capability/requirement declaration | the runtime cannot validate compatibility before loading; #84 step 5 impossible |
| C10 | Alternatives exist for experts only | no general mechanism for "compiler-approved equivalent implementations" |
| C11 | No compiled context ceiling in the package | #84's "runtime needs <= compiled max context" has nothing to compare against |
| C12 | `Package::open` / `expert_matrix_regions` hardcodes an *expert-specific* descriptor ABI (`0x20`, `wc == 0`, 88-byte descriptors) inside the generic format reader | a layering violation: the neutral reader knows one model family's payload layout |
| C13 | `compiler` field is a fixed literal `"colic-0.1.0"` | no compiler provenance; two compilers are indistinguishable |
| C14 | No algorithm anywhere for reclaiming space from superseded payload | #85 has no substrate to build on |

### 3.4 Compiler pipeline, as implemented

`logan-compiler` (`logan` binary) is `source discovery → semantic frontend → lowering → quant/codec →
StoragePlan → shard write → manifest → optional external plan`.

- **Frontends** exist for Qwen MoE, Qwen4-Exp (Qwen3.8-Flash-Next), Qwen MTP and **DeepSeek-V4**
  (`model/deepseek_v4.rs`, `Architecture::DeepSeekV4Flash`, with hash layers via a `tid2eid` int64 tensor).
- **Targets** are `TargetProfile` (Apple8/Metal, Linux x86-64 AVX2), resolved from a `MachineProfile` probe
  (`target/machine.rs`) that reads OS/arch/RAM and a few capability gates from env — no storage probing at all.
- **Quant** is per-tensor (`QuantSpec.kind` ∈ `exact | bf16 | mxfp4-tile8x32 | i4-g32 | f8-e4m3`), with a
  "sensitive dense" veto floor (embed/head/norm/gate must stay ≥ bf16 unless waived).
- **Codec**: rANS256 and Apple8 packers, with C-oracle differential identity tests.
- **Recompile** (`recompile.rs`) supports selective per-record quant/layout rewriting with
  `ActionKind::Rewrite{requantized}`, and writes `recompile.json` provenance including
  `source_model_fingerprint`, `parent_manifest_sha256`, `in_place`, and requantization counts. This is the
  seed of the provenance model in §14 — it is *good* and should be promoted into the artifact.

### 3.5 Planner and capacity model, as implemented

This is the most important part of the audit, because the design must not fight it.

`logan-ir` already contains the **correct** abstraction, landed by #93:

```rust
enum DataMutability { Immutable, Mutable }
enum BackingKind { PackageRecord, RuntimeStateFile, ResidentOnly, DevicePersistent }
struct BackingPlan  { kind, storage_pool: Option<StoragePoolId>, bytes, alignment, page_or_block_bytes }
struct ResidencyPlan{ memory_pool, minimum_working_set_bytes, target_resident_bytes,
                      pinned_bytes, eviction_priority }
struct AccessPlan   { kind, prefetch_depth, expected_read_bytes_per_step, expected_write_bytes_per_step }
struct ResourcePlan { mutability, backing, residency, access }
struct ResourceBudget { memory_pools: Vec<MemoryPoolBudget>, storage_pools: Vec<StoragePoolBudget> }
```

`tiered_optimizer.rs` implements admission correctly and has the tests to prove it:

- `add_resource` checks `minimum_working_set_bytes <= pool.capacity_bytes` **and**
  `target_resident_bytes <= pool.capacity_bytes`, and checks mutable backing against a *storage* pool. It never
  compares total logical bytes to RAM.
- `ram_overcommitted_logical_state_is_feasible_when_backing_exists` — 21 GiB logical, 12 GiB target resident,
  16 GiB RAM ⇒ feasible.
- `irreducible_working_set_still_has_to_fit_ram` — the correct hard failure.
- `immutable_hundred_gib_package_does_not_need_duplicate_spill_capacity`.
- `discrete_memory_pools_are_capacity_checked_independently`.
- `TieredResourceUsage` reports minimum working set, target resident, mutable backing, immutable package
  backing, per-pool usage, and storage read/write traffic per step.

`context_plan.rs` is architecture-aware (separate byte models for full-attention KV, GDN recurrent, GDN conv,
QSA index, PLE, MTP, scratch) — exactly what #81 asked for.

**Conclusion of the audit: the planner side of #37 is substantially landed. The package side is not.**
`.logan` v1's job is the missing half: *emit what the planner already knows, bind it to the bytes, and make
the runtime read it.*

### 3.6 Runtime, as implemented

`logan-core/src/sched/` — one scheduler owner, typed tickets, generation-safe leases, deterministic replay.

- `residency.rs` (`ResidencyManager`): pools, devices→pools, entries keyed by
  `ExpertKey { model, package, layer, expert, representation, pool }`, states `Absent/Loading/Resident/Failed`,
  `LoadDisposition { Started | Joined | AlreadyResident }`, `Lease{id,key,generation,owner}`, LRU eviction of
  unpinned entries, `PoolStats` with resident/reserved/pinned bytes, hits/misses/evictions/transfers/bytes/
  `exposed_wait_ns`. Invariants asserted in debug builds.
- `device.rs`: `DeviceKind{Cpu,Gpu,Neural,Io,Other}`, `ExecutionTarget{device,kind,capabilities}`, registry
  with `find_kind_capable`. Good, and already models "devices are not memory pools".
- `storage.rs` (core): `TensorStorage { Resident | Streamed{shard,offset,len} | Gpu{handle,ptr} | Shared }`.
- `expert.rs`: O(1) intrusive-list LRU with optional per-layer capacity; C-measured plateau at ≥256 slots.

Runtime limitations that matter for `.logan`:

| # | Limitation | Consequence |
|---|---|---|
| R1 | `ResidencyManager` is **expert-centric** (`ExpertKey` has layer/expert). It cannot express a KV layer, a PLE block range, a dense island or a scratch arena as first-class residents | the same tier system is not usable by all large state, which #82 explicitly requires |
| R2 | No backing-store identity per entry: no file, no offset, no write, no dirty state | mutable out-of-core state is unrepresentable |
| R3 | Eviction is strict LRU with no priority input, even though `ResidencyPlan.eviction_priority` exists in the IR | the compiler cannot express "evict this first" |
| R4 | Nothing reads a plan at load | #84 is entirely unstarted |
| R5 | `TensorStorage::Streamed` carries `(shard: String, offset, len)` with no block geometry, no access class and no checksum | the streaming path cannot verify or coalesce |
| R6 | Prefix-cache identity is computed by hashing the *loaded* model and numerical policy (`model_digest`) | expensive, and no stable artifact identity exists to key caches safely against a specific physical plan |

### 3.7 Platform paths, as implemented

- **Apple Silicon / Metal + MetalIO** is the mature path. `logan-metal` exposes `metalio_file_add`,
  `metalio_slot_alloc/free/ptr/bytes`, `metalio_loadv`, `metalio_wait`, `metalio_batch_barrier/wait`,
  `metalio_slot_consumed`, `metalio_stats`. `ColiSource::expert_matrix_regions` exists specifically so
  `metalio_loadv` can stream the resident tiles of one expert straight into one slot with a scatter list.
  This is the design's strongest existing asset and `.logan` must feed it *better*, not around it.
- **Windows x86-64 + CUDA** is planned, not built: `.claude/specs/deepseek-v4-engine/{README,design,
  kernels-cuda,requirements}.md` propose `logan-cuda` + `logan-v4`, native MSVC (explicitly *not* WSL), SM100
  and SM120 kept distinct, with residency/streaming as separate performance regimes. `target/mod.rs` today
  knows only Apple8 and Linux-x86_64-AVX2 (`PROFILES`).
- The **neutral-backend design** (`docs/design_neutral_backend.md`, approved 2026-08-28) already committed to
  the mini-LLVM shape: compiler emits graph+plan, runtime executes it. `.logan` is the serialization of that
  contract. It also records the honest finding that "the current compiler graph is a weight-load inventory,
  not a complete executable transformer" — the plan must not pretend otherwise.

---

## 4. Requirements and non-goals

### 4.1 Requirements

**R1 Plan authority.** A `.logan` artifact must carry the complete physical execution contract, integrity-bound
to the payload. Runtime startup validates it. The runtime may never silently produce a materially different
physical plan.

**R2 No global-RAM admission.** No admission decision may compare total virtual/model/context bytes to a fast
memory tier. The only memory-related hard failure is an irreducible per-operation working set that fits no
accessible pool.

**R3 Axis separation.** Math representation, physical layout/kernel ABI, execution backend/device, backing
home, resident working set, transfer/access path and compiled alternatives are independently representable.

**R4 Access class.** Storage/access classes are explicit and intrinsic, distinguishing bulk streaming from
indexed, small-unit, prefetchable lookup traffic.

**R5 Unit addressability.** Any class of data whose natural read unit is much smaller than its total size must
declare unit/block geometry and be readable and verifiable at that granularity.

**R6 Mutable state out of core.** Mutable state (KV, recurrent, indexer, prefix snapshots) must have a declared
backing requirement and a specified on-disk state-store format, created and validated by the runtime against a
writable pool.

**R7 Truthful precision.** The artifact must distinguish source mathematical representation, lossless repacking,
target-specific physical layout, quantized representation and explicit requantization. A mixed-precision model
must be auditable per tensor; no single model-level label.

**R8 Deterministic identity.** Two compilations of the same source with the same plan inputs produce a
byte-identical artifact with a stable content identity usable as a cache key and as a plan/package binding.

**R9 Bounded-memory compilation.** Compiling a checkpoint larger than RAM must be possible with bounded memory,
streaming source traversal, resumability, crash recovery and atomic publication.

**R10 Robustness.** Safe-Rust-parseable, fuzzable, no path traversal, no allocation from unvalidated counts,
checked arithmetic, partial-corruption detectability, repairability, unknown-section skippability.

**R11 Evolvability.** A new model architecture must be representable without a format version bump. A new
*structural* feature must be addable via an optional section that older readers skip, or a required-flag that
older readers reject — never by a silent reinterpretation.

**R12 Portability with explicit specialization.** Artifacts declare capability requirements, not machine
identities. One artifact may carry bounded, explicitly-listed alternatives; duplication is budgeted, never casual.

**R13 Platform neutrality.** The container must not be Apple-specific. Metal, CUDA, ROCm, Vulkan, CPU and
future accelerators must be expressible as capability sets and stored layout ids.

**R14 Inspectability.** The plan's derived metrics (working set, backing bytes, storage traffic, exposed stall,
headroom, quality loss, context capability, latency/throughput estimates) must be readable and printable without
executing the model.

### 4.2 Non-goals (v1)

- Not a training or incremental-update format.
- Not a distributed/remote-shard format (single-node artifact; distribution is a layer above).
- Not a universal graph optimizer. The plan may remain a weight/state-load inventory plus explicit execution
  islands, as the neutral-backend design already acknowledges.
- Not an encrypted or DRM format.
- Not a compression container for transport. Transport encoding (if any) is a separate concern; `Zstd`
  skippable frames are noted as a future *optional* wrapper, not a v1 requirement.
- Not a replacement for the tokenizer/config sidecar files. Those remain source metadata copied into the package
  for self-containment, with a declared digest, but they are not part of the compiled contract.

---

## 5. Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Keep COLI, add a bigger header** | The problem is not header width. COLI's record model cannot express sub-record geometry, storage class, mutable state or a mandatory plan; adding those as more record fields produces the "untyped metadata dumping ground" failure. COLI also conflates the neutral reader with one expert payload ABI (C12). |
| **One giant single file, no directory** | Loses resumable/parallel streaming compilation, bounded per-writer streaming, partial distribution and cheap per-shard atomicity. Kept as a *variant* (§6) rather than the default. Not justified as a performance win — see §17. |
| **Separate file per storage class, as the primary organizing principle** | Measured: no effect on sparse p99 or bulk throughput (§17). Files-per-class would be an unjustified complexity. Classes are a *semantic and scheduling* axis; file layout is derived from them for convenience only. |
| **Safetensors-style flat header, no plan** | Safe and simple, but zero execution contract; #37 needs a plan, not just tensors. Adopted its good idea (offset-ordered, mmap-friendly payload) rather than its scope. |
| **GGUF-style arbitrary key-value metadata** | Solves "new architecture invents something unusual" by making everything untyped. Correctness then depends on conventions no parser can enforce. Rejected in favor of typed sections + an explicit extension range. |
| **Object-file model (ELF-style with symbols and relocations)** | The closest fit conceptually, and the plan *is* shaped like a linker output with symbol bindings. Rejected as a wholesale metaphor: real relocations and dynamic linking solve address binding, which is not the problem; we need *resource* binding. The useful parts — segment table, program headers, lazy mapping, symbol/version tables — are adopted. |
| **Purely content-addressed CAS with a separate manifest (OCI-like)** | Excellent for dedup, partial download and atomicity, and the *payload* half is adopted (immutable, digest-named, never mutated). Rejected as the whole design because a 450 GiB model has essentially no dedup opportunity, and pure CAS makes the byte ranges non-contiguous, which hurts the bulk streaming path that dominates expert I/O. |
| **A database/LSM tree (RocksDB-style) for the sparse tables** | Right instinct for the lookup class (block index + block cache + per-block checksums are adopted). Rejected as the storage substrate: an LSM rewrites and compacts, which is exactly wrong for immutable model bytes, and its read path adds write-ahead and level bookkeeping to what is a read-only lookup. |
| **KV-cache-style paged blocks for everything** | Correct for mutable KV (adopted for the state store), wrong for immutable weights where the natural unit is a whole expert or a contiguous dense tensor. |
| **Leave the plan outside the artifact (status quo, `--plan PATH`)** | Directly contradicts #83 and R1. A plan that can silently mismatch its weights is worse than no plan. |
| **Store the plan as JSON** | Human-friendly, untyped, unbounded, slow, and impossible to validate structurally. JSON stays as a generated *report*. |
| **Keep bincode-over-serde for the plan** | Not self-describing, no schema evolution, no unknown-field skipping, silently wrong across a struct change. `PLAN_ARTIFACT_VERSION` hand-migration (`LegacyPlanArtifactV4`) already exists and is exactly the maintenance burden a tagged format removes. |
| **One artifact = exactly one physical plan, no alternatives** | Simplest, and the v1 *default*. But it forces a full recompile for a second accelerator, and #37 requires "approved alternatives are explicit". Rejected as an absolute rule; adopted as the default with a bounded opt-in. |
| **One artifact = every possible alternative** | Duplicates hundreds of GB for portability that is rarely needed. Rejected outright. |

---

## 6. Recommended artifact topology

### 6.1 One logical layout

```
foo.logan                     (single-file packing)
   └── MANIFEST ‖ payload segments ‖ PLAN  — one file, file table has 1 entry

foo.logan/                    (sharded directory packing — DEFAULT)
   ├── MANIFEST               small immutable root; validated fully on open
   ├── PLAN                   mandatory compiled physical plan
   ├── PROVENANCE             human-readable JSON audit report (not authoritative)
   ├── JOURNAL                optional; only exists during in-place mutation
   └── data/
        ├── dense-00000.lgd
        ├── stream-00003.lgd
        ├── gather-00001.lgd
        ├── pinned-00000.lgd
        └── ...
```

Rules:

- **`.lgd` files are immutable and content-addressed.** A published segment is never rewritten. This gives
  crash-safe publication for free: a partially written segment is simply never referenced by a committed
  manifest, so recovery is "delete unreferenced segments".
- **A `.lgd` file contains segments of exactly one access class.** Not for speed (§17) — so the runtime can
  open only the files it needs, budget each class independently, and keep bulk extents contiguous.
- **File names are conventional, not authoritative.** The manifest's file table is authoritative; the reader
  may *check* the conventional name but must never *derive* a path from metadata. Preserves COLI's
  path-traversal immunity (§3.2.2).
- **Default shard target 4 GiB** (matches the current working 24 × ~4 GiB layout), exposed as a compile flag.
  Segment size must be able to exceed the shard target for a single large tensor region; the file table
  declares the actual size.

### 6.2 Why two packings and not one

The sharded directory is right for 450 GiB: parallel writers, resumable streaming compilation, per-shard
atomicity, bounded scratch, and partial distribution.

The single file is right for a 4 GB model someone wants to `scp`. It costs almost nothing to support because
the only difference is the *file table*: extents are always `(file_id, offset, length)`, and the single-file
packing is simply `file_id 0 == the MANIFEST itself`, with payloads appended after the manifest and the plan.
The manifest is at offset 0 with a fixed header, so it is still readable without any seek. This is the
ELF shape (header → program headers → sections) and it means the format has exactly one reader.

### 6.3 What is *not* in the artifact

| Thing | Where it lives | Why not in `.logan` |
|---|---|---|
| Machine profile (RAM, VRAM pools, storage pools, queue concurrency) | Runtime probe + planner input | It is a fact about a machine, not about a model. The plan stores only a *digest* of the profile used, for reproducibility. |
| Live free space | Runtime probe | Volatile. The plan stores *required* bytes; the runtime verifies actual free space at bind time (#68's rule). |
| Mutable state bytes | Runtime-created state store on a compatible writable pool | The artifact is immutable. |
| Caches (expert slots, PLE blocks, prefix snapshots) | Runtime-owned, keyed by artifact identity | Policy and observation, not contract. |
| Absolute paths | Nowhere | Requirements only; `--state-dir` or a pool default binds them. |
| Scheduler timing decisions | Runtime | The compiler provides bounds and advice. |

---

## 7. The `.logan` v1 logical model

Nine concepts. Everything else is a field.

### 7.1 Artifact identity

- `artifact_id` — BLAKE3 over the canonical serialization of `MANIFEST` (with the id field zeroed) ‖ `PLAN`.
  A content identity for cache keys, prefix-cache directories, state-store directories and dedup.
- `source_fingerprint` — carried forward from COLI: a digest of the source checkpoint's identity.
- `plan_digest` — BLAKE3 of `PLAN`, stored in the manifest.
- `generation` — monotonically increasing publish counter, 1-based. Two artifacts with identical `artifact_id`
  and different `generation` differ only in packaging, never in contract.

Consequence for R6/R5 of §3.6: prefix caches key on `artifact_id` + state-layout id + prefix digest. Two
packages compiled from the same checkpoint with different precision/layouts get different `artifact_id`s and
therefore cannot silently share caches — which the deepseek-v4 engine spec asked for explicitly.

### 7.2 Representation contract (per segment)

```
math_format:  u16   # registry id: 0x0000 canonical/opaque, 0x0020 apple mxfp4 tile8x32, 0x0003 bf16, ...
scale_format: u16   # 0x0000 none, 0x0004 e8m0, 0x000a f32, ...
block:        { rows, cols }        # quantization block geometry (0,0 = none)
stored_bytes: u64
logical_bytes:u64
provenance:   rep_id -> ProvenanceEntry     # §14
```

`RepresentationKey` for a runtime cache must therefore be `{ layout, kernel_abi, repr_contract }`, where
`repr_contract` is a registry id enumerating the representation tuple of a **bundled** segment (for an
expert: the per-matrix triple, e.g. `Q4_K/Q4_K/Q8_0`). See §23.5(1) for why a single-quant form is
insufficient.

`math_format` alone is **not** the representation: `0x0020 + e8m0 + 8×32` and `0x0020 + e8m0 + 1×32` are
different contracts. The triple (math, scale, block) is the identity. This is why `RepresentationKey` in
`residency.rs` is already `{layout, kernel_abi, quant}` — extend it to include the block geometry.

### 7.3 Layout and kernel ABI (per segment)

`layout: u16` (execution layout id, registry-backed), `kernel_abi: u16`. Distinct from representation: the
same numbers in the same format can have different physical orders (canonical row-major vs Apple8 tiling), and
the same layout can be consumed by different kernels. Registry lives in `abi/*.toml` → generated Rust; this is
existing, working machinery to reuse, not replace.

### 7.4 Access class (per segment) — the new axis

```
enum AccessClass {
  Stream,   // bulk; consumed in large contiguous extents; linear order; no per-unit address
  Gather,   // indexed small units; addresses come from a computation; coalescable; prefetchable with lead time
  Mutable,  // written and read; requires a state store; ordering and durability matter
  Pinned,   // must be resident; no legal backing; part of the irreducible working set
}
```

Class is **intrinsic** — it describes how the bytes *must* be addressed, and it is decided by the compiler
from the data's role. It is **not** a `Placement` enum: `Placement::{Resident,Streamed,Gpu}` folds backing,
residency and device visibility into one value; `AccessClass` folds none of them and is orthogonal to all of
them. A `Stream` segment can be pinned or cached; a `Gather` segment can live in a package or (in principle) a
device-persistent pool.

Two scheduling properties are **policy**, not intrinsic, and therefore live in the plan's access/advice
structures, not in the class:

- `prefetch_lead` — how many execution steps ahead the addresses become knowable (tokens = whole batch;
  router output = 1–2 layers; query-dependent = 0).
- `queue_class` — which I/O queue/priority the runtime should use (bulk vs latency-critical).

Why the split matters: MoE experts and Engram tables differ in *both* respects, and the difference is not a
DeepSeek/Qwen quirk:
- routed experts → `Stream`, lead ≈ 1–2 layers, bulk queue, large units;
- Engram tables → `Gather`, lead = whole step, latency queue, 256 B units, must be coalesced.

With a single "streamed tensor" abstraction these two fight through one queue and one cache and one budget.
With the class + policy split they get independent residency budgets, independent queue classes and independent
traffic accounting — which is precisely what #37 requires and what §12/§13 show is needed.

### 7.5 Block geometry and addressability (per segment)

```
unit_bytes:   u64   # 0 = not unit-addressable
unit_count:   u64
block_bytes:  u64   # 0 = the whole segment is one block
block_units:  u64   # units per block (fixed stride); 0 => use the index section
index:        IndexKind
```

`IndexKind::FixedStride` — unit *i* is at `segment.offset + i*unit_bytes`, blocks are `block_bytes` apart.
No index array is needed; only optional per-block CRCs. Covers Engram and every uniform sparse table.

`IndexKind::SortedKeyTable` — an explicit, sorted `(u64 key → u64 byte-offset)` table plus a per-block CRC
array. Covers variable-size sparse dictionaries and any future non-uniform lookup structure. *Specified in v1,
optional to implement.*

`IndexKind::None` — the segment is only meaningful as a whole.

This is what makes R5 real: a 400 MB sparse shard becomes 400 MB / `block_bytes` independently verifiable
blocks, and a 256 B row read becomes a legal, checksummed operation rather than an unwritten convention.

### 7.6 Backing (per segment)

Exactly #93's `BackingPlan`, kept:

```
enum BackingKind { PackageSegment, StateStore, PinnedOnly, DevicePersistent }
struct BackingPlan { kind, storage_pool: Option<StoragePoolId>, bytes, alignment, page_or_block_bytes }
```

`PackageSegment` is COLI's `PackageRecord` renamed for accuracy (a segment, not a record). Immutable package
bytes do **not** require duplicate spill capacity — this is the invariant
`immutable_hundred_gib_package_does_not_need_duplicate_spill_capacity` already tests.

### 7.7 Residency (per segment / per logical resource)

`ResidencyPlan { memory_pool, minimum_working_set_bytes, target_resident_bytes, pinned_bytes,
eviction_priority }`, kept from #93. The semantics that must not drift:

- `minimum_working_set_bytes` — the **hard** requirement. Across sequentially-executed resources the optimizer
  takes the **max**, not the sum (already implemented).
- `target_resident_bytes` — the **performance** choice.
- `pinned_bytes` — permanent floor, sums across resources.
- `eviction_priority` — ordering input the runtime must honor (currently the IR has it and the runtime ignores
  it; R3 of §3.6).

### 7.8 Access and transfer path (per segment)

```
enum AccessKind { DirectShared, Mapped, AsyncStream, Staged }
struct AccessPlan { kind, prefetch_depth: u16, queue_class: u8,
                    expected_read_bytes_per_step: u64, expected_write_bytes_per_step: u64 }
```

`queue_class` is the addition that makes §17's measured contention actionable. Transfer *paths* between
pools come from the runtime's device-link graph (issue #56), not from the artifact; the artifact declares the
`AccessKind` its kernels can consume.

### 7.9 Execution islands and alternatives

`ExecutionPlan { islands, edges }` is kept unchanged from `logan-ir` (it is well-formed and validated).

Alternatives get a first-class, *bounded* representation:

```
struct VariantGroup { key: SymbolId, variants: Vec<SegmentId>, selection: Selection }
enum Selection {
  Exclusive,                     // choose exactly one
  Preferred(Vec<SegmentId>),     // ordered preference; runtime takes the first it can execute
  CapabilityExpr(String),        // optional: registry-defined predicate over the runtime capability set
}
```

Rules:
- **v1 default: no variant groups.** One physical plan.
- The compiler emits more than one variant only on explicit request (`--variants <set>`), and the plan reports
  the extra bytes in `costs`.
- A variant is legal only if the compiler emitted it; the runtime may **never** synthesize a representation.
- `--variant` byte accounting is reported in `package_bytes` and in the plan's declared metrics, so casual
  duplication cannot happen silently.

This satisfies "if multiple equivalent compiled alternatives are allowed, represent them explicitly" without
paying for portability nobody asked for.

---

## 8. Physical / on-disk specification

All integers little-endian. All offsets are byte offsets into the file named by `file_id`. All lengths are
checked against the containing file's declared size before any allocation. No pointer-sized fields. No
embedded paths. No NUL-terminated strings.

### 8.1 `MANIFEST`

```
MANIFEST := Header ‖ SectionTable ‖ FileTable ‖ Sections… ‖ Trailer

Header (64 bytes, fixed):
   0  u8[10]  magic              "LOGANMODEL"
  10  u16     format_major       1
  12  u16     format_minor       0
  14  u32     header_bytes       64 (must equal)
  18  u16     min_reader_major   1
  20  u16     min_reader_minor   0
  22  u32     flags              bit0 = single-file packing, bit1 = plan present,
                                 bit2 = provenance file present, bit3 = journal required for recovery
  26  u64     generation         ≥1, monotonic
  34  u32     section_count
  38  u32     file_count
  42  u64     section_table_offset
  50  u64     file_table_offset
  58  u32     header_crc32c      over bytes 0..58 with this field zeroed
  62  u16     reserved

SectionTable: section_count × 32 bytes, ascending by (kind):
   0  u16     kind
   2  u16     entry_flags
   4  u32     reserved
   8  u64     offset
  16  u64     length
  24  u32     crc32c
  28  u32     reserved2

FileTable: file_count × 24 bytes, ascending by file_id:
   0  u32     file_id
   4  u32     name_id          # string id; the reader MAY check the conventional
                                 # name and MUST NOT derive a path from it
   8  u64     file_bytes
  16  u32     file_kind        # 0 MANIFEST(self) 1 PLAN 2 PAYLOAD 3 JOURNAL 4 PROVENANCE
  20  u32     reserved

Trailer (16 bytes):  u8[8] magic "LOGANEND" ‖ u32 trailer_crc32c ‖ u32 reserved
```

`artifact_id` is stored as a 32-byte field inside the `IDENTITY` section, zeroed while hashing.

### 8.2 Section kinds

| kind | name | contents |
|---|---|---|
| 0x0001 | `STRINGS` | interned UTF-8 blob + descriptor array (len u32, off u64) |
| 0x0002 | `IDENTITY` | `artifact_id[32]`, `source_fingerprint[32]`, `plan_digest[32]`, `compiler_id`, `compiler_version`, `profile_digest[32]` |
| 0x0003 | `CAPS` | capability requirements (§8.5) |
| 0x0004 | `SEGMENTS` | segment table (§8.3) |
| 0x0005 | `SYMBOLS` | logical value/state → segment bindings (§8.4) |
| 0x0006 | `STATE_REQ` | mutable backing requirements (§8.6) |
| 0x0007 | `COSTS` | declared resource metrics (§8.7) |
| 0x0008 | `MODEL` | compacted logical model description (architecture kind, geometry, per-layer facts, tensor roles/shapes/dtypes) |
| 0x0009 | `VARIANTS` | variant groups (§7.9); absent in the default single-plan case |
| 0x0010–0x7FFF | *reserved for `.logan` minor extensions* | |
| 0x8000–0xFFFF | *implementation/experiment range, never required for correctness* | |

Unknown kinds are skipped by length. Unknown **required** feature bits (see `CAPS`) are a hard rejection.

### 8.3 `SEGMENTS`

```
SEGMENTS := Header ‖ Entry[count] ‖ Digest[count]? ‖ BlockCrc[…]

Header (32 bytes):
   0  u32  count
   4  u32  entry_bytes      96
   8  u32  digest_bytes     32 (0 => digests absent)
  12  u32  flags            bit0 = digests present, bit1 = block CRC arrays present
  16  u64  entry_table_offset
  24  u64  aux_offset

Entry (96 bytes):
   0  u64  segment_id            nonzero, unique, strictly increasing
   8  u16  access_class          1 Stream, 2 Gather, 3 Mutable, 4 Pinned
  10  u16  mutability            0 immutable, 1 mutable
  12  u16  math_format           registry id (0x0000 = canonical/opaque)
  14  u16  scale_format          registry id (0x0000 = none)
  16  u16  layout                execution layout registry id (0xFFFF = none)
  18  u16  flags                 bit0 unit-addressable, bit1 block CRCs present,
                                 bit2 index present, bit3 verified-whole, bit4 sparse-allocated
  20  u32  file_id
  24  u64  offset                MUST be a multiple of the artifact record alignment
  32  u64  stored_bytes
  40  u64  logical_bytes
  48  u64  unit_bytes            0 = not unit-addressable
  56  u64  unit_count
  64  u64  block_bytes           0 = single block
  72  u64  block_units           0 = fixed-stride derived from block_bytes/unit_bytes
  80  u64  index_offset          0 = none; else offset into the containing file
  88  u16  kernel_abi
  90  u16  quant_block_rows
  92  u16  quant_block_cols
  94  u16  reserved

Digest[count]: 32 bytes each, in segment_id order. BLAKE3 over stored bytes.
BlockCrc[]:    u32 × ceil(block_count) per flagged segment, concatenated in segment_id order,
               each CRC32C over that block's stored bytes.
```

Validation performed on open (all before any payload read):

1. `offset + stored_bytes <= file_bytes` for the declared `file_id`, and `offset % alignment == 0`.
2. `segment_id != 0`, strictly increasing.
3. `unit_bytes > 0 ⇒ unit_count > 0 ∧ unit_bytes*unit_count <= stored_bytes` (checked multiply).
4. `block_bytes > 0 ⇒ block_count = ceil(stored_bytes/block_bytes)` and the block CRC array length matches
   exactly when flag bit1 is set.
5. `access_class == Mutable ⇒ mutability == 1 ∧ backing != PackageSegment` (immutable segments cannot be
   declared mutable; mutable segments cannot live in the immutable package).
6. `access_class == Pinned ⇒ backing == PinnedOnly`.
7. `math_format`, `scale_format`, `layout`, `kernel_abi` must be registered ids (unknown ids are rejected
   pre-load, not at first use).

### 8.4 `SYMBOLS` — logical bindings

```
SYMBOLS := Header ‖ Entry[count]
Entry (40 bytes):
   0  u64  symbol_id
   4  u32  name_id               # into STRINGS
   8  u16  role                  # 1 weight, 2 scale, 3 index, 4 bias/aux, 5 kv-main, 6 kv-swa,
                                 # 7 recurrent, 8 conv-history, 9 indexer, 10 prefix-snapshot,
                                 # 11 scratch, 12 speculative, 13 multimodal-asset
  10  u16  layer                 # 0xFFFF = not layer-scoped
  12  u32  expert                # 0xFFFFFFFF = not expert-scoped
  16  u32  variant_group         # 0xFFFFFFFF = none
  20  u32  segment_first
  24  u32  segment_count
  28  u64  logical_bytes
  36  u32  reserved
```

A logical tensor (e.g. one routed expert's three matrices) binds to **one or more** segments. This is what
lets an expert bundle be one contiguous `Stream` segment feeding one MetalIO slot (preserving COLI's best
property) while a large dense tensor is split into N row-block segments so a bounded working set exists.

### 8.5 `CAPS` — capability requirements

```
CAPS := Header ‖ Requirement[]
Requirement (16 bytes):
   0  u16  kind    1 abi_version(min,max) 2 execution_layout 3 kernel_abi 4 backend_class
                   5 gpu_family_min 6 cpu_feature_mask 7 io_feature 8 memory_pool_class
                   9 storage_pool_class 10 accelerator_memory_min
  10  u16  flags   bit0 required, bit1 optional-but-preferred
  12  u32  value
```

Admission is a **subset check** over the runtime's advertised capability set, never an exact device match
(#84's rule). A package specialized for one machine runs on another when ABI, layout ids, required pool
classes, access paths and capacities are satisfied. There is deliberately **no machine fingerprint
requirement**; the profile digest is recorded in `IDENTITY` for reproducibility only.

### 8.6 `STATE_REQ` — mutable backing requirements

```
STATE_REQ := Header ‖ Entry[count]
Entry (64 bytes):
   0  u32  state_id
   4  u32  name_id
   8  u16  kind         1 kv-main, 2 kv-swa, 3 recurrent, 4 conv-history, 5 indexer,
                        6 prefix-snapshot, 7 scratch-journal, 8 speculative
  10  u16  repr         state-store layout registry id
  12  u32  layout_rows
  16  u64  bytes                 # required capacity, exact
  24  u64  block_bytes
  32  u64  alignment
  40  u64  read_bytes_per_step
  48  u64  write_bytes_per_step
  56  u16  durability  0 none, 1 crash-consistent, 2 prefix-reusable
  58  u16  min_working_set_bytes
  60  u32  reserved
```

The plan stores **requirements**, never a path (#83's rule). The runtime binds them to a compatible writable
storage pool or an explicit `--state-dir` and verifies actual free space. On insufficiency it fails before
inference with required bytes, available bytes, the exact deficit and which state needs it.

### 8.7 `COSTS` — declared resource metrics

Stored as a TLV list of `(metric_id: u16, value: u64)`. Categories (matching #37 and #82):

```
0x0001 virtual_logical_bytes
0x0002 immutable_package_bytes
0x0003 mutable_backing_bytes
0x0004..0x000B minimum_working_set_bytes[pool_class]      (one per pool class present)
0x000C..0x0013 target_resident_bytes[pool_class]
0x0014 scratch_bytes
0x0015 journal_bytes
0x0016 storage_read_bytes_per_decode_step
0x0017 storage_write_bytes_per_decode_step
0x0018 storage_read_bytes_per_prefill_token
0x0019 transfer_bytes_per_decode_step
0x0020 exposed_stall_ns_per_decode_step_estimate
0x0021 headroom_bytes[pool_class]
0x0022 quality_loss_ppm
0x0023 context_max_tokens_compiled
0x0024 latency_cost
0x0025 throughput_estimate_tok_s
```

These are **declared** (compiler-computed, authoritative for admission and reporting). The runtime's
*observed* counters are separate telemetry. A declared/observed divergence is a diagnostic, never a silent
replan.

### 8.8 `PLAN`

A separate file, magic `LOGANPLAN`, same TLV framing, mandatory whenever `flags.bit1` is set (mandatory in
v1). It carries:

- `plan_version`, `artifact_id`, `source_fingerprint`, `compiler_id`/`version`, `profile_digest`
- `CAPS` (mirrored — the plan is independently admissible without the manifest's model section)
- `CONTEXT` — `{ model_max_tokens, compiled_max_tokens, constraint_kind, per-component byte geometry }`
- `RESOURCES` — one `ResourcePlan` (§7.6/7.7/7.8) per logical resource, **populated**
- `EXECUTION` — islands + edges (existing `ExecutionPlan` shape)
- `STATE_REQ` — mirrored from the manifest, so the plan is self-sufficient
- `COSTS` — declared metrics
- `OBJECTIVE` — cost-model id, selected Pareto point id, and the rejected alternatives with reasons
- `EXT` — extension range

`plan_digest` in the manifest is BLAKE3 over the whole `PLAN` file, so a stale or substituted plan is rejected
pre-load. The `artifact_id` binds them transitively.

### 8.9 Alignment, extents and addresses

- `record_alignment` is declared in `CAPS`/`IDENTITY` and constrained to `[4096, 1 MiB]`, power of two.
  Segment `offset` must be a multiple of it. Rationale: preserve the property that made MetalIO's scatter-list
  path work and keep page-cache behaviour predictable. Apple8 uses 16384 today; a 4096 default with an Apple8
  override is fine.
- Overlapping extents are **structurally rejected** in v1. A segment's bytes may be *shared* by being
  referenced twice, not by overlapping ranges — this removes an entire class of adversarial metadata.
- Sparse allocation (holes) is allowed but must be declared (`flags.bit4`); a hole reads as zero and is not an
  error.

### 8.10 The state store (runtime-created, format specified in v1)

Created by the runtime, outside the artifact, at `<state-dir>/<artifact_id>/<state_id>.lgs`.

```
Header (128 bytes):
   0  u8[8]  magic "LOGANST1"
   8  u8[32] artifact_id      # refuses to bind state to a different artifact
  40  u32  state_id          # must equal the STATE_REQ entry
  44  u16  repr              # state-store layout id
  46  u16  flags             bit0 dirty-tracking, bit1 crash-consistent
  48  u64  block_bytes
  56  u64  block_count
  64  u64  capacity_bytes
  72  u64  committed_blocks   # prefix of valid blocks
  80  u32  epoch
  84  u32  header_crc32c
  88  u8[40] reserved
Blocks (block_bytes each, aligned), followed by
BlockMeta[block_count]: u32 crc32c ‖ u32 state_bitmap  (per-block validity)
```

Recovery: a state store whose `artifact_id` mismatches, whose header CRC fails or whose `committed_blocks`
exceeds `block_count` is refused. Blocks beyond `committed_blocks` are discarded, not trusted. Prefix-reusable
state (`durability == 2`) records the token position its blocks correspond to in the `SYMBOL` binding for that
state, so a prefix hit can be validated rather than assumed.

This is a new subsystem, not a format detail — but its *existence and shape* must be fixed by v1, because the
plan has to declare and the runtime has to enforce the requirement (#84 step 7).

### 8.11 The journal (optional, only for in-place mutation)

v1's default is "never mutate a published segment". The low-space in-place path of #85 therefore has two
compliant strategies:

1. **Add-then-commit** (no journal): write new segments into free space, commit with one `MANIFEST` rename,
   reclaim unreferenced segments afterwards. Extra disk = new segment bytes; old space is freed after commit.
   Crash-safe with no journal at all.
2. **Extent-reuse with journal** (needed only when free space cannot hold the new bytes): the `JOURNAL` file is
   an append-only sequence of records
   `{ magic, generation, segment_file, offset, length, new_digest, crc32c }`, fsynced before the in-place write.
   Commit increments `generation` and flips `flags.bit3` off. Recovery replays: for each record, either the new
   digest matches (write completed) or the extent is re-written from the plan. A manifest/plan/segment triple
   that disagrees is never exposed, because the commit is the single `MANIFEST` rename.

Submission of `.logan` v1 only requires strategy 1 plus the *declared* existence and format of the journal so
strategy 2 is implementable without a format change. This is exactly the "no v2 escape hatch" requirement.

### 8.12 Parser and fuzz surface

- `#![forbid(unsafe_code)]` for the whole format crate (COLI already does this; keep it).
- Every length is validated against the containing file's declared size **before** allocation; every
  allocation is bounded by a caller-supplied cap.
- All arithmetic on offsets/lengths is checked; counts are compared against the byte budget that must exist
  for them (`count * entry_bytes <= section.length`).
- Strings: length-prefixed UTF-8, no NULs, total bounded; invalid UTF-8 rejected.
- Unknown section kinds are skipped by length; unknown *required* capability bits reject.
- Overlapping extents rejected; unreferenced file-table entries tolerated (they are garbage awaiting reclaim).
- Fuzz targets: (1) `MANIFEST` header+section table, (2) `SEGMENTS` + block-CRC arrays, (3) `PLAN`, (4) the
  state-store header. Each must be fuzzed with a seed corpus of truncated/bit-flipped real artifacts.

### 8.13 Determinism

Given the same source fingerprint, the same compile request and the same plan inputs, emission is byte-identical
regardless of thread count or filesystem. This means: stable ordering everywhere (ascending `segment_id`,
ascending `symbol_id`, `BTreeMap` not `HashMap` in the emitting path), no timestamps in authoritative sections
(the `PROVENANCE` report may carry a timestamp because it is not authoritative), and a fixed padding fill
byte. COLI already has determinism tests (`apple8_package_determinism.rs`); `.logan` keeps them and extends
them to `artifact_id`.
---

## 9. Compiler and planner architecture

### 9.1 Pipeline

```
source checkpoint
   │  (streaming, bounded memory, resumable)
   ▼
import / source inventory          ← existing: source::SourceInventory, tensor index
   ▼
semantic model (frontend)          ← existing: SemanticModel + ModelGeometry (qwen_moe, qwen4_exp, deepseek_v4)
   ▼
logical IR                         ← NEW: roles + access class + representation candidates per symbol
   ▼
machine + storage profile          ← EXTENDED: MachineProfile v2 with memory AND storage pools (#68)
   ▼
legal representations / placements ← existing: candidate groups (quant × layout × backing × residency × device)
   ▼
cost model / planner               ← existing: tiered_optimizer (Pareto, per-pool admission)
   ▼
chosen physical plan               ← existing: TieredParetoPlan → ResourcePlan per resource
   ▼
.logan emission                    ← NEW: segments + bindings + caps + state_req + costs + plan
   ▼
validation                         ← NEW: structural + digest + capability + capacity preflight
   ▼
runtime storage/residency views    ← NEW: artifact-driven, replaces ad-hoc engine init
   ▼
scheduler execution                ← existing: SchedulerCore / SchedulerRuntime
```

The only *new* stages are the logical-IR annotation (roles/access class), emission and validation. Everything
upstream of that exists and is preserved deliberately.

### 9.2 The one required change to the planner

`build_physical_plan` must stop being the admission authority. Concretely:

- Delete `check_uma_pool` as an admission gate. Its arithmetic is a legitimate *input* to a pool budget, but it
  is not a feasibility test.
- Build a `ResourceBudget` from the extended machine profile (memory pools + storage pools) and route all
  admission through `tiered_optimizer`'s `add_resource`, which already implements the correct test.
- Populate `MemoryPlan.resources` from the selected plan's `ResourcePlan`s. This is the single highest-value
  change in the whole programme: it converts #93 from dead code into the emitted contract.
- Replace the `Placement::{Resident,Streamed,Gpu}` *authority* with `ResourcePlan`; keep `Placement` only as a
  compatibility projection for engines that have not migrated (as the current code comment already intends).

When no optimizer ran, the compiler must still emit a complete `ResourcePlan` per resource by a simple,
deterministic defaulting rule: `PinnedOnly` for scratch and irreducible state; `PackageSegment` + `Stream` for
experts; `PackageSegment` + `Stream` with a residency target for dense weights; `PackageSegment` + `Gather` for
sparse tables; `StateStore` + `Mutable` for context state. No admission failure other than "the irreducible
working set fits no pool" or "the required backing bytes exceed available storage".

### 9.3 Machine and storage profile (`MachineProfile` v2, #68)

```
struct MachineProfile {
  // identity / ABI (existing)
  operating_system, architecture, backend capabilities, avx2, gpu family, ane, ...
  // memory pools (existing, corrected)
  memory_pools: Vec<MemoryPoolProfile>,   // id, class(UMA|Host|Pinned|DevicePrivate), capacity_bytes,
                                          //   bandwidth_class, addressable_by: Vec<DeviceId>
  // storage pools (NEW — the #68 gap)
  storage_pools: Vec<StoragePoolProfile>, // id, capacity_bytes, free_bytes_probed,
                                          //   device_class, seq_read_bw_class, random_iops_class,
                                          //   writable, sparse_ok, mmap_ok, native_async_io,
                                          //   direct_io, alignment
  // concurrency (NEW — needed for §17's finding)
  io_queue_depth: u32,                    // bits a scheduler can use to hide latency
  devices: Vec<DeviceProfile>             // id, class, queues, pools, links
}
```

`free_bytes` is a volatile observation, never baked into the artifact (#68's rule). The *requirement* is
persisted; the runtime re-probes.

On Apple Silicon this yields one UMA memory pool (CPU/Metal/ANE share it) plus one storage pool. On a
Windows/CUDA machine it yields host RAM, pinned staging and per-GPU VRAM as distinct pools plus one or more
NVMe storage pools — the topology the deepseek-v4 engine spec already insists on keeping distinct.

### 9.4 Physical-plan identity and reproducibility

The plan records `profile_digest` (digest of the `MachineProfile` used) and the cost-model id. Two compiles on
different machines therefore produce different plans with different `artifact_id`s, and the planner can explain
*why* a plan was chosen. This is the "deterministic cost/objective decisions with rejected-alternative
explanations" theme from #37.

### 9.5 Streaming, bounded-memory compilation

Preserved from the current implementation and made explicit as a format property:

- Source traversal streams shard by shard; only the current tensor/expert is resident plus one output staging
  buffer. Peak compiler memory is set by the largest single unit, not by the model.
- Segments are written with `writev`-style batching directly from the lowering buffer into the final layout —
  no intermediate copy of a different layout. (COLI already avoids a decode+repack round trip for Apple8; keep
  it and make it the default for every target.)
- **Resumability**: because segments are immutable and content-addressed, a crashed compile is resumed by
  re-emitting only segments whose digest is absent. A `${artifact}.progress` JSON records the planning decisions
  (deterministic) and the set of completed segment digests. No journal needed for the add path.
- **Deterministic identities**: `artifact_id` depends only on source fingerprint + plan inputs, so interrupted
  and repeated compiles converge.
- **Low free space**: the compiler preflights projected bytes (`projected_stored_bytes + projected_padding`)
  against the target pool's free bytes *before* any destructive work and reports
  `required / available / deficit` precisely (#85's "exact preflight"). With immutable segments, an aborted
  compile leaves only reclaimable garbage.

### 9.6 What "the plan is authoritative" means for the compiler

The compiler must never emit a segment the runtime cannot execute on the declared `CAPS`, and must never rely
on the runtime to fix up a layout. Because `CAPS` is a subset check, this is enforceable at emission: resolve
the target capability set, validate every segment's `layout`/`kernel_abi`/`math_format` against it, and refuse
to emit otherwise. COLI already has the seed of this in `resolve()`'s refusal to emit under a profile whose
producer is unimplemented; `.logan` generalizes it.

---

## 10. Runtime architecture

### 10.1 Startup contract (implements #84)

```
open artifact
  1. read MANIFEST header; verify magic, sizes, CRC32C, trailer
  2. walk the section table; reject unknown REQUIRED kinds/features
  3. read IDENTITY; verify file table sizes on disk
  4. read + verify PLAN; check plan_digest == IDENTITY.plan_digest
  5. probe live capabilities (devices, memory pools, storage pools, io queue depth)
  6. admission = CAPS ⊆ advertised capabilities            → else reject with the missing capability
  7. admission = for every ResourcePlan, minimum_working_set_bytes <= some accessible pool
                                                        → else reject with pool, required, available, deficit
  8. admission = required backing bytes <= available writable storage
                                                        → else reject with state id and exact deficit
  9. bind state requirements: create/open state stores; verify artifact_id, header CRC, capacity
 10. instantiate residency pools and caches at the declared target_resident_bytes (never more)
 11. install the artifact's access classes, queue classes and prefetch depths into the scheduler
 12. execute
```

Steps 6–8 are the whole point: **step 7 is a per-pool working-set test, and there is no step that compares
total virtual bytes to RAM.** The runtime is forbidden from rejecting a plan because
`virtual_logical_bytes > physical_memory`.

Explicitly *not* done at startup: reading payload, computing a model digest over payload, or building a graph
from scratch. The graph and the resource plan are in the artifact. Building a graph ad hoc remains available as
a debug path from a COLI package only, and it must be visibly marked as unplanned in telemetry.

### 10.2 Residency, generalized (fixes R1/R2/R3)

`ResidencyManager` becomes pool-and-**region**-oriented rather than expert-oriented:

```
struct RegionKey {          // was ExpertKey
  artifact: ArtifactId,
  segment:  SegmentId,      // the physical identity
  pool:     MemoryPoolId,   // destination capacity domain
  repr:     RepresentationKey,   // { layout, kernel_abi, quant, quant_block }
}
```

`segment_id` replaces `(layer, expert)`: it is already the physical identity, it works for a KV layer block, a
PLE block range, a dense island and an expert alike, and it removes the need for the format to know that
experts exist. `LayerKey`/`ExpertKey` views remain as thin adapters for existing engines.

Additions:
- `backing: Option<BackingRef>` — `{ file_id | state_id, offset, length, block_bytes, mutable }`, so
  Mutable regions have a real home and can be written back.
- `eviction_priority: u16` honored from `ResourcePlan.residency.eviction_priority` instead of strict LRU.
- `access_class` and `queue_class` on the entry, so the pool can apply class-specific admission
  (e.g. do not let a bulk `Stream` fill evict a latency-critical `Gather` working set).
- `PoolStats` extends to per-class: resident/target/bytes-per-class, queue depths, exposed wait per class.

Nothing else in `residency.rs` changes: the generation/lease discipline, the typed completions, the
`Started/Joined/AlreadyResident` dispositions and the debug invariants are correct and hard-won.

### 10.3 Scheduler and the class split (fixes the §17 finding)

The measured result — that concurrent bulk streaming triples the p99 of small random reads *regardless of file
layout* — is a **scheduling** problem, and the format must give the scheduler the vocabulary to solve it:

- Each `ResourcePlan` carries `queue_class` and `prefetch_depth`.
- The runtime budgets **in-flight bytes per class** rather than globally. Concretely: a `Stream` class gets a
  bounded number of large outstanding reads; a `Gather` class gets a bounded, *reserved* slice of the device
  queue so it cannot be starved by bulk work.
- `prefetch_lead` is what makes this cheap for the Engram case: its addresses are known from the token stream
  for the whole step, so its I/O can be issued early and amortized instead of demanded at the point of use.
  With 9,200 IOPS measured and 48 lookups/token, demand fetching would burn ~5 ms/token; issued as coalesced
  blocks with queue depth, the same bytes cost a fraction of that. The artifact is what makes the prefetch
  *legal and schedulable*.

This is the concrete meaning of "expert streaming and Engram traffic are independently laid out, budgeted,
queued and scheduled instead of fighting through an accidental single cold-I/O abstraction."

### 10.4 What the runtime may change dynamically

| Decision | Compiler | Runtime |
|---|---|---|
| math representation | decides, persists | may not change |
| physical layout / kernel ABI | decides, persists | may not change |
| execution backend/device | decides a legal set | may choose **only** an emitted variant |
| backing home | decides (package / state store / pinned) | may not change the *kind*; may bind a state requirement to a compatible pool |
| minimum working set | decides, persists as a hard requirement | must satisfy; may not exceed |
| target residency | decides | may lower under pressure; may raise only up to headroom |
| prefetch depth | advises | decides within the advised bound |
| eviction order | gives relative priorities | decides the exact victim within a priority tier |
| queue class | assigns | decides submission order and in-flight budget |
| context length | decides the ceiling | may request ≤ ceiling; above it is a hard error with a recompile instruction |

### 10.5 Prefix caches and state identity (fixes R6)

Prefix-cache directories key on `artifact_id` + state-store layout id + prefix digest. This is cheaper than
the current `model_digest` over the loaded model, and — more importantly — it is *safe*: two artifacts compiled
from one checkpoint with different precision or layout cannot share a prefix cache by accident. COLI's
`recompile.json` already tracks exactly the information needed to distinguish them
(`source_model_fingerprint`, `parent_manifest_sha256`, requantization counts); that data moves into
`IDENTITY`/`PROVENANCE`.

---

## 11. The resource model

One table, because this is the part that must not be muddled.

| Layer | Owns | Lives in | Example values |
|---|---|---|---|
| **Source model description** | identity of the checkpoint | `IDENTITY` (fingerprint + digest), `MODEL` (compacted geometry/roles) | source fingerprint, architecture kind, layer count, per-layer attention mode |
| **Logical IR** | symbols, roles, shapes, dtypes, access class, candidate representations | `SYMBOLS`, `MODEL`, `SEGMENTS` | `layers.7.ffn.experts.123` → role=weight, access=Stream |
| **Physical plan** | chosen representation, layout, backing, residency, access, alternatives, declared costs | `PLAN`, `SEGMENTS`, `COSTS`, `VARIANTS` | `mxfp4-tile8x32 + e8m0 + 8x32`, layout `0x0103`, kernel abi 1 |
| **Machine profile** | live hardware/resource facts | runtime; digest recorded in `IDENTITY` | UMA 16 GiB; NVMe seq 3.0 GB/s, 9.2 k IOPS, queue depth |
| **Immutable model data** | payload bytes | `data/*.lgd` | 253 GiB FP4 experts, 183 GiB FP8 lookup tables |
| **Mutable state** | backing bytes + validity | runtime state store `.lgs` | 890 B/token FP4 main KV; recurrent state |
| **Caches** | residency choice within bounds | runtime memory pools | 256 expert slots; PLE block cache; prefix snapshots |
| **Scheduler policy** | timing | runtime, bounded by plan | prefetch depth, queue class order, eviction victim |

Hard-failure taxonomy (the *only* legal refusals):

| Failure | Reported as |
|---|---|
| irreducible working set fits no accessible pool | pool, required bytes, available bytes, deficit |
| required backing capacity unavailable | state id, required, available, deficit |
| required storage/access path unsupported | capability name that failed the subset check |
| incompatible kernel ABI / layout / representation | the id and the required-vs-advertised set |
| context beyond the compiled ceiling | requested vs compiled max, with a recompile instruction |
| address/offset/format limit exceeded | the specific limit and the offending field |
| insufficient free disk for the planned artifact or recompile | preflight required/available/deficit |

There is deliberately **no** "model + context does not fit in RAM" entry.

---

## 12. DeepSeek-V4.1-Flash mapping

Authoritative sources used: the official technical report (`DeepSeek_V41_Tech_Report.pdf`, in
`~/Downloads`) and the official `config.json` for `deepseek-ai/DeepSeek-V4.1-Flash`. Both were read directly
for this design; nothing below is inferred from aggregator coverage.

### 12.1 What the model actually is

- **552B backbone parameters + 196B Engram parameters**, 40 layers split 20 causal-encoder / 20 decoder,
  hidden 5120, vocab 129280. Activates **16B/token on decode, 8B/token on prefill**. 1M-token context
  (YaRN ×16 from 65536). 45T multimodal training tokens.
- **MoE**: 384 routed experts + 1 shared per layer, `moe_intermediate_size` 2304, **top-6**. Expert weight
  matrices are **FP4**; everything else is **FP8 with UE8M0 scales and 32×32 weight blocks**.
- **CSA2** (Compressed Sparse Attention 2) with three statically assigned modes. Encoder layers 2–19 use
  compression `m=2` in three groups of six (first layer Full, five Reuse). Decoder layers use `m=1` in five
  groups of four (first Full, rest Reuse for group 1; first Reindex, rest Reuse for groups 2–5). Query heads
  64, head dim 512, q-lora 1280, o-groups 8, o-lora 1024. Indexer: 32 heads × 128 dims, top-512.
  Hierarchical Sparse Indexer: ≤2048 blocks × 8 positions = ≤16384 candidates.
- **KV**: FP4 main KV (MXFP4 E2M1, one E4M3 scale per 16 channels) = **890 bytes/token, always in HBM**.
  FP8 SWA KV for `n_win = 128`. SWA KV is *not* persisted to SSD — it lives in a host-DRAM pool with minutes
  TTL; persistent global KV lives on SSD with ≥72 h retention under LRU. **SWA Bounded Replay** reconstructs
  missing SWA state by replaying only `n_win` tokens.
- **Engram**: 2 modules (layers 1 and 14), 98B params each. N-gram orders {2,3,4} × 8 hash heads; each head
  indexes a table of ~16M entries (distinct primes) → **384,006,168 entries per module**; 2048 embedding dims
  per order = 256 per head. Tables and projections in **FP8**. The report states plainly that "deterministic
  addressing enables embeddings to be prefetched from host memory via background RDMA transfers".
- **DSpark** speculative decoding: 3 Transformer blocks, window 128, 5 parallel draft positions, Markov head
  (rank 256), confidence head, its own 128 routed experts at top-3.
- **Vision**: DeepSeek-ViT, 32 layers, hidden 1024, 16 heads, patch 14, 3×3 pixel-unshuffle, 544²–1344².
- **mHC** with `hc_mult = 4` residual streams, Single-Pass formulation, fused into a Mega-mHC kernel.

### 12.2 Computed byte budget

| Component | Params | Format | Bytes |
|---|---|---|---|
| routed experts | 543.6 B | FP4 | **253.1 GiB** |
| Engram tables (2 modules) | 196.0 B | FP8 | **182.5 GiB** (+0.36 GiB scales) |
| attention (q/kv/o/indexer, 40 layers) | 6.9 B | FP8 | 6.4 GiB |
| shared experts | 1.4 B | FP8 | 1.3 GiB |
| embedding + head | 1.3 B | FP8 | 1.2 GiB |
| vision encoder | ~0.3 B | FP8 | 0.3 GiB |
| DSpark drafter + 128 experts | ~2.4 B | FP4/FP8 | 2.3 GiB |
| **total on disk** | | | **≈447 GiB** |

**447 GiB of artifact vs 16 GiB of machine RAM — a factor of 28.** Two components are 97% of it: routed
experts (253 GiB) and Engram (183 GiB). If those two were treated as one "streamed tensor" class, the design
would be dead on arrival. They are not, and here is why in numbers:

| | routed experts | Engram tables |
|---|---|---|
| access class | `Stream` | `Gather` |
| total bytes | 253 GiB | 183 GiB |
| natural unit | one expert = 3 matrices ≈ **17.7 MB** | one entry = 256 dims FP8 + scale ≈ **257 B** |
| address space | 15,360 discrete bundles | **384 M entries per module** |
| address source | router output (layer L−1 activations) | **input token sequence only** |
| address lead time | 1–2 layers | **the entire step / batch** |
| per-token volume (top-6, 40 layers) | **≈4.25 GB** | 48 lookups ≈ **12.3 KB** |
| sensitivity | bandwidth + residency/locality | **IOPS + latency** |
| prefetchability | limited (router-dependent) | **unlimited and deterministic** |
| storage class budget | bulk queue, large extents | latency queue, coalesced small blocks |
| `prefetch_lead` | 1–2 | whole step |
| `queue_class` | bulk | latency-critical |

The 4.25 GB/token figure is the honest reason DeepSeek's own deployment keeps Engram tables GPU-resident and
serves experts with expert parallelism. On a 16 GiB laptop that is not available, so the design's job is to
make the *consequences* visible and correct rather than to hide them: `COSTS` reports
`storage_read_bytes_per_decode_step` and `exposed_stall_ns_per_decode_step_estimate`, and admission still
passes because the **irreducible working set** — one layer's six experts (106 MB) plus scratch and state — fits
comfortably. That is the whole thesis, measured.

### 12.3 How the design represents each component

| Component | access class | unit / block | backing | residency | notes |
|---|---|---|---|---|---|
| routed experts (`w1/w2/w3` + scales) | `Stream` | one **bundle** per `(layer, expert)`; 3 matrices with declared sub-offsets | `PackageSegment` | LRU cache, `eviction_priority` per layer group | bundle stays contiguous so one MetalIO/CUDA scatter load feeds one slot — preserves COLI's best property |
| Engram tables (48 sub-tables) | `Gather` | `unit_bytes = 257`, `block_bytes = 4096` | `PackageSegment` | tiny hot cache; **never** a whole-table target | block CRCs make a 257 B read verifiable; `prefetch_lead = step`; latency `queue_class` |
| shared experts, attention proj., embeddings | `Stream` or `Pinned` | tensor (may split into row blocks) | `PackageSegment` | sized so the per-layer working set fits | ~9 GiB; the parts read every step are the natural pinning candidates |
| main KV (FP4), indexer state, compressor tails | `Mutable` | block = compressor/attention block granularity | `StateStore` | layer/block-aware windows, **not** a fractional global cache | 890 B/token; CSA2 means KV is **shared across layer groups**, so the state layout must express sharing, not one-cache-per-layer |
| SWA KV | `Mutable` (low durability) | `n_win` window | `StateStore` | minutes TTL, replayed, not persisted | `durability = 0`; the plan states the replay cost instead of budgeting SSD for it |
| prefix snapshots | `Mutable`, `durability = 2` | block | `StateStore` | prefix cache | keyed on `artifact_id` |
| DSpark drafter + its 128 experts | `Stream`, own resource group | expert bundle | `PackageSegment` | separate cache, own budget | a distinct `SYMBOL` group; the scheduler can deprioritize it when the queue is saturated |
| vision encoder weights | `Stream` or `Pinned` | layer tensors | `PackageSegment` | resident when multimodal is in use | separate execution island |
| multimodal image assets | not model weights | n/a | outside the artifact | n/a | inputs, not model state |
| scratch / mHC workspaces | `Pinned` | — | `PinnedOnly` | irreducible | contributes to `minimum_working_set_bytes`; with `hc_mult = 4` this is a real, non-trivial number |

Two format features exist *because* of this model and would otherwise have required DeepSeek-specific hacks:

1. **`Gather` + `prefetch_lead` + `queue_class`.** Engram's "deterministic addressing ⇒ prefetchable" property
   is representable, so the runtime can issue its reads a step early and coalesce them into blocks, instead of
   demanding 257-byte reads at the point of use where the measured 9,200 IOPS device would cost ~5 ms/token.
2. **Block geometry + block CRCs.** A 384M-entry table cannot be validated as a whole (183 GiB), and its reads
   are 257 bytes. Without per-block checksums and fixed-stride addressing the format would be forced to either
   trust unverified bytes or read 700 MB blocks. This is now generic — Qwen's PLE table needs exactly the same
   two features, which is the test that it is not a DeepSeek hack.

Also note what the design *refuses* to do: it does not put `engram`, `csa2`, `dspark` or `mhc` anywhere in the
format. Those are model semantics, and they live in the frontend and in `MODEL`/`SYMBOLS` role data. The format
knows about *streams, gathers, mutable blocks and pinned scratch*; the model supplies the arrangement.

---

## 13. Qwen3.8-Flash-Next mapping

This one is grounded in a **real compiled artifact** measured for this design:
`~/models/Qwen3.8-Flash-Next-REAP-288-MXFP4-Apple8.coli` — 91 GiB, 24 shards, 15,023 records.

### 13.1 Observed structure → `.logan` segments

| COLI record | count | size | `.logan` class | unit/block | residency |
|---|---|---|---|---|---|
| `layers.*.ple.ple_embedding.ngram_embedding.shard_NNN` | 128 | 400,002,048 B each (51.2 GiB total) | `Gather` | unit = one n-gram row; block = declared | small hot cache, never resident as a whole |
| expert records (kind 2, `(layer, expert)`) | 13,824 | 2,611,648 B each | `Stream` | one bundle (3 matrices) | LRU cache, layer-partitioned |
| dense/GDN/attention tensors (kind 1) | 1,199 | 21 distinct sizes, 224 B … 1.27 GB | `Stream` or `Pinned` | tensor, splittable into row blocks | sized to the per-layer working set |
| KV / GDN recurrent / GDN conv / QSA index / MTP state | **absent** | — | `Mutable` | block | state store, layer/block-aware windows |

### 13.2 The specific defects `.logan` fixes here

1. **A 400 MB PLE record is an opaque blob.** Verified: the reader's own comment says the streaming path
   performs *no* CRC, and `expert_matrix_regions` exists because a reader must be taught a payload ABI out of
   band. In `.logan` the n-gram shard declares `unit_bytes`/`block_bytes` and per-block CRC32C, so a row read is
   addressable, verifiable, coalescable and repairable — and the generic reader needs to know nothing about
   PLE. The brain page for this project already records Mateo's standing requirement that the ~51B-parameter
   n-gram table "must remain NVMe-backed/cold and must never be fully materialized in RAM"; the format now
   *enforces* it by declaring a residency target of a small block cache rather than a whole-tensor obligation.
2. **The 51.2 GiB table currently competes with expert streaming through one cold path.** With `Gather` vs
   `Stream` classes they get separate residency budgets, separate queue classes and separate traffic
   accounting. §17 shows why this is a scheduling, not a file-layout, fix.
3. **GDN/QSA/PLE-conv state has no declared backing at all** today, even though the project's own research page
   concludes that GDN state, PLE convolution state, QSA indexer/cache state, hyper-connection streams and MTP
   draft state "are persistent runtime objects, not disposable per-forward temporaries" needing "explicit
   ownership, generation IDs, cancellation semantics and prefix-cache invalidation". `.logan`'s `STATE_REQ` +
   state store + `artifact_id`-keyed prefix caches are the format-level answer to that conclusion.
4. **`check_uma_pool` currently rejects the honest plan.** On this 16 GiB machine the compiler sums all
   non-expert decoded bytes and refuses when they exceed RAM. `.logan` admission instead asks whether the
   per-layer working set fits and whether the backing exists.

### 13.3 Apple Silicon specifics

- One `uma0` memory pool for CPU/Metal/ANE; one NVMe storage pool. MetalIO slot use becomes a *residency
  implementation detail* driven by `ResourcePlan.residency.target_resident_bytes`, not a hardcoded 256-slot
  default (`expert_cache_slots()` today reads `QWEN4_CACHE`/`CACHE` with a default of 256).
- Apple8 `layout 0x0103`, `math_format 0x0020`, `scale_format 0x0004`, 8×32 tiles of 136 B stay exactly as they
  are — they are a **registered layout id**, which is precisely what the registry mechanism exists for.
- `ANIMATION`: the ANE path (`logan-ane`) contributes `ExecutionIsland{ backend: Ane }` entries with
  `cache_key`s; the plan already models this and it survives unchanged.

### 13.4 Windows x86-64 + CUDA specifics

The plan must be able to describe the same logical model compiled for a different target without a format
change:

- a different `layout`/`kernel_abi` id per segment (e.g. CUDA FP4 block layouts, SM100 vs SM120 kept distinct as
  the engine spec requires);
- a different `CAPS` set: several memory pools (host, pinned staging, per-GPU VRAM), `io_feature` requiring
  overlapped I/O + IOCP rather than MetalIO, `backend_class = Cuda`;
- `AccessKind::Staged` with an explicit `min_working_set` for the pinned staging pool, which is a distinct
  capacity domain on discrete-GPU machines and must never be summed with host RAM;
- `queue_class` for copy engines distinct from compute queues.

None of this requires a new format version — it requires new *registry ids* and new *capability bits*, which is
the evolvability property R11 asks for. Conversely, an Apple8 artifact and a CUDA artifact of the same source
have different `artifact_id`s and can coexist and be selected explicitly, which is the "several physical
alternatives" case handled by `VARIANTS` when the user asks for it and by two separate artifacts when they do
not.

---

## 14. Quantization and representation provenance

### 14.1 The rule

> Logan must never lie about model precision. Any statement about a tensor's numerical content must be
> traceable to one of: the source checkpoint, a lossless repack, or a recorded lossy transformation — and the
> three must be distinguishable by inspection.

### 14.2 Five distinct things, never collapsed

| Concept | Meaning | Recorded as |
|---|---|---|
| **source representation** | what the checkpoint actually contained | `source_math_format`, `source_dtype`, `source_block`, `source_scale_format`, tensor name |
| **lossless repack** | bits reordered/tiled, no value changed | `op = LosslessRepack`, `bit_exact = true`, verified by a decode-compare gate |
| **target physical layout** | the order the kernels want | `layout`, `kernel_abi` — an execution property, orthogonal to numbers |
| **quantized representation** | values were quantized at some point | `op = Quantize(from, to, block, scale_fmt)` with `bit_exact = false` |
| **explicit requantization** | already-quantized source was converted to a different quant | `op = ExplicitRequantize`, `allow_requantize` flag, count of affected segments |

### 14.3 `ProvenanceEntry` (per emitted segment)

```
struct ProvenanceEntry {
  rep_id:          u32,
  source_tensors:  Vec<StringId>,        // source names that produced this segment
  source_math:     u16,                  // registry id
  source_scale:    u16,
  source_block:    (u32, u32),
  emitted_math:    u16,
  emitted_scale:   u16,
  emitted_block:   (u32, u32),
  source_nominal_weight_bits:  u16,      // from the source quant format
  emitted_nominal_weight_bits: u16,      // must be <= source when requantized
  ops:             Vec<Op>,              // ordered LosslessRepack | ExplicitRequantize | Passthrough
  bit_exact:       bool,                 // true only when op chain is empty or all-LosslessRepack,
                                         // and a decode-compare gate passed
  gate:            Option<GateId>,       // which regression gate certified it
}
```

`bit_exact = true` may only be set when the op chain is empty or all-`LosslessRepack` **and** the
corresponding decode-compare gate passed. `ExplicitRequantize` requires `allow_requantize` to have been
explicitly set on the request (mirroring the existing `recompile.rs` behaviour), and the count of requantized
segments is reported in `COSTS`-adjacent provenance and in `PROVENANCE`.

The **rule** is owned by `logan-compiler/src/quant/precision_policy.rs` (`validate_quant_transition`,
`QuantFormat::nominal_weight_bits`), which is already implemented and fail-closed: `LosslessRepack` requires
source == target, and `ExplicitRequantize` forbids raising nominal weight-code bits. `.logan` stores the
*evidence* (source and emitted bits, the op list, `bit_exact`) and the verifier calls the same function — it does
not re-derive the policy. Note that a **target layout change is not a transform**: moving `Q4_K` bytes into
`ggml_native_blocks_v1` changes the `layout` id and leaves `math_format` and the op list untouched — which is
exactly why the Windows/CUDA work could register a new layout with no policy change (§23.3).

### 14.4 What this preserves from existing Logan work

- The **sensitive-dense floor** (embed/head/norm/gate must not be silently narrowed below bf16 unless waived)
  is preserved as a planner policy. `.logan` makes it *visible*: the waive reason is recorded in `PROVENANCE`.
- `recompile.rs`'s `ActionKind::Rewrite{ requantized }` and its `recompile.json`
  (`source_model_fingerprint`, `parent_manifest_sha256`, `rewritten_experts`, `requantized_experts`,
  `allow_requantize`) is the seed. Its semantics are promoted into the artifact; the JSON stays as the
  human-readable report.
- The Apple8 C-oracle differential tests (`apple8_rans_c_identity`, `rans_identity`,
  `apple8_rust_c_identity`) remain the certification mechanism referenced by `ProvenanceEntry.gate`.

### 14.5 Mixed precision is normal, and must be legible

The DeepSeek-V4.1 case is the proof: FP4 expert matrices with block scales, FP8 shared experts and attention
with UE8M0 32×32 blocks, BF16 norms, F32 gate biases — in one artifact. `.logan` has **no model-level
precision field**. A reviewer asking "what precision is this model?" gets an answer per segment, with counts of
`bit_exact` vs quantized vs requantized segments, rather than a label. The `PROVENANCE` report prints exactly
that summary.

---

## 15. Compatibility, versioning and security

### 15.1 Versioning

- `format_major` / `format_minor` in the manifest header. A reader rejects `format_major > supported`.
- `min_reader_major`/`min_reader_minor` let a writer say "readers older than X cannot be trusted with this",
  which is how a *required* new semantic is signalled without bumping `format_major`.
- `flags` in the header are **required** feature bits: an unknown bit ⇒ reject. This is the only place a new
  feature can force a rejection.
- **Section-kind skipping** is the extension mechanism for anything that need not be understood to execute —
  provenance additions, reports, statistics, tuning hints. `0x0010–0x7FFF` is reserved for `.logan` minor
  extensions, `0x8000–0xFFFF` for implementations and never required.
- **Registry ids are the model-architecture extension mechanism.** A new quantization format or tile layout is
  a new id in `abi/*.toml` → generated Rust. No format bump. Unknown ids are rejected pre-load with the id
  named, rather than misread.
- Explicit **legacy readers** are permitted to be deleted only via the ADR process; the successor rule is that
  a reader must handle every artifact whose `format_major` it claims to support.

### 15.2 What is deliberately *not* versioned into the format

- Model architecture specifics. `MODEL` carries a free-form but *length-bounded* architecture descriptor
  section; unknown architecture fields are skipped by the frontend, and the *execution* contract lives in
  segments and bindings, which are generic. A new architecture must be representable by new roles/geometry,
  not by a new reader.

### 15.3 Security and robustness

| Threat | Mitigation |
|---|---|
| path traversal via metadata | no paths in metadata; file names come from the file table by index and are derived from `file_id` |
| unbounded allocation from a claimed count | every count is validated against the section's byte budget before allocation; a caller-supplied allocation cap is mandatory |
| integer overflow in offsets/lengths | checked arithmetic throughout; `offset + length <= file_bytes` validated before use |
| truncated file | header/trailer CRC32C, section CRC32C, declared sizes vs actual file sizes |
| partial corruption | per-segment BLAKE3 digest + optional per-block CRC32C; block-granular repair is possible when block CRCs are present |
| overlapping/adversarial extents | overlapping extents structurally rejected in v1 |
| oversized dimensions | `unit_count * unit_bytes <= stored_bytes`, `block_count` derived and cross-checked against the CRC array length |
| malicious section length | section must lie inside the manifest/file and not overlap the header, section table or file table |
| silent precision change | provenance records every op; `bit_exact` requires a passing gate |
| stale/substituted plan | `plan_digest` in the manifest; `artifact_id` binds plan and payload transitively |
| state store belonging to another model | `artifact_id` in the state-store header |
| fuzzing | four fuzz targets (§8.12) seeded with truncated and bit-flipped real artifacts |

### 15.4 Open cost

Bounded by construction: header (64 B) → section table → file table → `SEGMENTS`/`SYMBOLS`/`CAPS`/`STATE_REQ`.
No payload read. Measured COLI baseline for the equivalent operation on the 91 GiB, 15,023-record artifact is
**7.1 ms**; `.logan` adds digests (32 B/segment ≈ 0.5 MB) and one plan file, so the same operation should stay
in single-digit milliseconds. If a future model exceeds a few hundred thousand segments, the segment table
itself becomes a payload-backed section — a compatible extension, since it is already length-prefixed and
offset-addressed.

---

## 16. COLI migration and deprecation

### 16.1 What actually needs to migrate

Evidence-based audit of COLI dependencies, from the tree:

| Consumer | What it uses | Migration |
|---|---|---|
| `logan-format` (`package.rs`, `codecs.rs`, `verify.rs`) | the entire format | becomes the **COLI legacy reader**; `.logan` gets its own crate (`logan-artifact`) so the legacy code is untouched |
| `logan-qwen4::colisource` (`ColiSource`, `SlotExpert`, `expert_matrix_regions`) | record lookup, expert region math, PLE row reads | grows a `LoganSource` sibling; `ColiSource` keeps working unchanged |
| `logan-qwen4::coliload` (`Model::load_coli`) | dual-probe name lookup, resident formats 5/7 | reads segment bindings instead of records when the artifact is `.logan` |
| `logan-qwen4::plan::{prefix_cache,prefix_runtime,snapshot}` | model digest for cache keys | key on `artifact_id` when available, fall back to the current digest for COLI |
| `logan-metal` MetalIO | `metalio_file_add` + scatter loads | unchanged: the artifact supplies `(file, offset, length)` triples, which is exactly MetalIO's input |
| `logan-compiler` (`pipeline`, `recompile`, `storage`, `codec`) | emitter | emits `.logan` alongside `.coli` during the transition |
| tests (`apple8_*_identity`, `qwen4_exp_apple8_compile`, `apple8_package_determinism`, …) | COLI artifacts | kept as-is for the legacy path; new `.logan` equivalents added, and the C-oracle identity tests re-pointed at `.logan` payloads |

### 16.2 Staged plan

**Stage 0 — design (this document).** Owner review. No code.

**Stage 1 — dual emit, dual load, no behaviour change.**
- `logan-artifact` crate: manifest/plan TLV reader+writer, `#![forbid(unsafe_code)]`, fuzz targets.
- `logan compile` gains `--format logan|coli|both`. Default stays `coli` until the runtime can execute `.logan`.
- `logan run` accepts either; `PROVENANCE`/telemetry records which path executed and whether it was planned.
- Correctness gate: for every existing fixture and the real Qwen3.8 package, a `.logan` artifact produced from
  the same source yields **token-identical** output via the existing gates, and identical Apple8 payload bytes
  (`apple8_rust_c_identity` re-pointed). This is the migration's safety net.

**Stage 2 — `.logan` becomes the default emit.** COLI emission becomes opt-in (`--format coli`) for reproducing
old artifacts. Runtime admission for `.logan` implements §10.1 in full.

**Stage 3 — the tiered capabilities go live.** `MemoryPlan.resources` populated; `check_uma_pool` removed as an
admission gate; state store implemented; scheduler queue classes and per-class in-flight budgets implemented;
the #37 acceptance tests (§20) pass.

**Stage 4 — deprecate COLI.** `logan compile --format coli` warns; the legacy reader remains for running
existing artifacts. A COLI→`.logan` converter is worth building only for the *shape*, not the content:

> **Converter verdict (evidence-based):** a COLI→`.logan` converter can recover **segments, representations,
> layouts, offsets and CRCs losslessly** — all of that is in the manifest. It **cannot** recover the plan
> (there is none), the storage classes (implicit), the block geometry of the 400 MB PLE records (the reader
> only knows the convention, not the contract), the access/queue policy, the mutable-state requirements or the
> declared costs. So a converter is useful as a *bulk repackaging* tool that produces a `.logan` with a
> **degraded, explicitly-marked plan** (single `Stream` class everywhere, no alternatives, no state
> requirements, `provenance.bit_exact = true` for unchanged bytes). It must be marked as needing recompile for
> any tiered or out-of-core deployment. The real migration path is **recompile from the original source
> checkpoint**, which is cheap precisely because compilation is streaming and bounded-memory.

**Stage 5 — remove COLI** once no runtime path, fixture, test or user artifact depends on it. Not scheduled by
this document.

### 16.3 What must not break during migration

- `logan-qwen4`'s working Apple8/MetalIO execution against the existing 91 GiB package.
- The C-oracle differential identity tests (`mateocabanal/colibri` remains the parity oracle).
- Prefix-cache identity safety: `.logan` introduces a *new, stronger* key. During dual support, COLI-keyed
  caches must not be re-used for `.logan` artifacts and vice versa.
---

## 17. Prototype and benchmark evidence

All numbers below were produced for this design on the target machine. Commands and scripts are named so they
can be re-run.

### 17.1 The machine

`MacBook Air (Mac14,15)`, Apple M2, **16 GiB RAM** (`hw.memsize` = 17179869184), internal NVMe, 926 GiB
filesystem with 54 GiB free at measurement time. This is the "tiny machine" in the thesis, and it is the same
machine that holds the real 91 GiB artifact.

### 17.2 The real artifact, parsed

`~/models/Qwen3.8-Flash-Next-REAP-288-MXFP4-Apple8.coli` — parsed directly from `manifest.coli` (no Logan code):

| Measurement | Result |
|---|---|
| package bytes | 97,144,168,346 (90.5 GiB) |
| shards | 24, sizes 2.27–4.29 GB |
| records | 15,023 — 13,824 expert (kind 2), 1,199 tensor (kind 1) |
| alignment | 16384 |
| distinct stored sizes | 21 |
| manifest bytes | 2,075,424 |
| **open + build record index** | **7.11 ms** (1.18 ms read, 5.93 ms index) |
| open 24 shard fds | 0.77 ms |
| largest records | 128 × 400,002,048 B (the PLE n-gram table: 51.2 GiB, 56% of the package) |
| expert record | 2,611,648 B, uniform |

Two conclusions feed the design directly:

1. **Manifest-scale metadata is a non-problem at this size.** 15 k segments in 2 MB and 7 ms means the
   segment-table design of §8 scales to the models Logan cares about without an indirection layer.
   (V4.1-Flash would be ~17 k segments — same order.)
2. **56% of the package is one sparse lookup table represented as 128 opaque 400 MB blobs.** That is the single
   most important structural fact about the current format, and it is the motivation for `Gather` +
   block geometry + block CRCs.

### 17.3 The decisive experiment: does storage-class separation need separate files?

Script: `ioexp2.py` (run on the Mac, read-only against the real package, `F_NOCACHE` on every descriptor so the
page cache is bypassed and real device behaviour is measured).

Two concurrent workloads:
- **bulk** — 2 MiB sequential reads (expert-streaming-shaped);
- **sparse** — small random reads at 4 KiB-aligned offsets over a 4 GiB file (lookup-table-shaped).

Variants: `alone` (no bulk load), `same` (both on `data-00000.coli`), `other` (sparse on `data-00020.coli`, same
NVMe). Median of 3 runs, 5 s each.

```
=== block 256 B ===
  alone  iops=  9253  p50= 111.2us  p99= 140.8us  max= 1831.8us
  same   iops=  8338  p50=  75.6us  p99= 523.4us  max= 9948.2us | bulk 2864 MB/s
  other  iops=  4795  p50= 127.4us  p99= 515.2us  max= 2177.6us | bulk 2982 MB/s
=== block 4096 B ===
  alone  iops=  9162  p50= 111.4us  p99= 144.4us  max= 1773.0us
  same   iops=  8875  p50=  32.9us  p99= 488.8us  max= 1921.2us | bulk 3086 MB/s
  other  iops=  4688  p50= 129.7us  p99= 511.6us  max= 2178.6us | bulk 2976 MB/s
```

Findings:

- **Device profile**: ~3.0 GB/s bulk sequential; **~9,200 IOPS** at both 256 B and 4 KiB random reads, p50
  ~111 µs, p99 ~141 µs. IOPS-bound, not bandwidth-bound — which is exactly the Engram/PLE profile.
- **Concurrent bulk streaming degrades the sparse workload's tail by ~3.6×** (p99 141 µs → ~490–520 µs) and
  fills the maximum-latency tail (1.8 ms → 5–10 ms).
- **The file the sparse reads come from makes no measurable difference.** Bulk throughput is 2.86–3.09 GB/s
  `same` vs 2.98 GB/s `other`; sparse p99 is ~490 µs `same` vs ~512–515 µs `other`. The differences in IOPS and
  p50 are head-of-line artefacts, not a layout effect — the two files are interleaved on the same device, so
  `same` vs `other` tests descriptor bookkeeping, not physical placement.
- **Therefore: separate files per storage class are not justified as a performance measure, and the design does
  not claim that they are.** The artifact separates classes so that the *runtime can budget and queue them
  independently* — which is the only lever the data supports.

The arithmetic that makes the class split matter for V4.1-Flash: 48 Engram lookups per token at 9,200 IOPS
demand-fetched is ~5 ms/token of pure lookup latency, before any expert I/O. Issued as coalesced 4 KiB blocks
with bounded queue depth and a whole-step prefetch lead, the same bytes cost a fraction of that. The format's
contribution is making those reads *addressable, verifiable, classifiable and schedulable*; the scheduling
itself is #56's queue model.

### 17.4 What this changes in the design (and what it removes)

**Changes:**
- `queue_class` is promoted from a nice-to-have to a required field, and the runtime must budget in-flight
  bytes **per class**, not globally.
- `prefetch_lead` becomes load-bearing for the `Gather` class.
- The plan must declare `exposed_stall` so a user can see the 3.6× tail degradation *before* running.
- Block-coalescing geometry (`block_bytes`, `block_units`) is required for `Gather` segments, because
  coalescing is the only way to escape the IOPS ceiling.

**Removes (Ponytail):**
- Any justification for a "one file per class" topology *as a performance feature*. Classes group shard files
  because it is tidy and makes independent budgeting legible — a one-line rationale, not an architectural claim.
- Any plan to add storage-tier-specific file alignment/padding as a tuning knob, because the measurement does
  not support an effect.

**Not measured, and therefore not claimed:** mmap vs pread for the lookup class; per-class alignment effects;
behaviour on a second device; NVMe command-queue depth effects. The design leaves these as runtime policy
choices (`AccessKind::Mapped` vs `AsyncStream`) rather than baking a conclusion.

### 17.5 A deliberate non-experiment and the decision it forces

`mmap` is attractive for immutable bytes (safetensors' whole thesis, llama.cpp's model loading). For a 253 GiB
expert set on a 16 GiB machine it is the wrong default: the page-fault path has no notion of working-set budget,
no queue class, no prefetch lead and no class-locality control, and it will happily thrash a resident set that
the scheduler had carefully bounded. **Decision: `AccessKind::AsyncStream` (pread / MetalIO / io_uring /
overlapped I/O) is the default for every large class; `Mapped` is available and must be explicitly selected.**
This is a design judgement, stated as one, not a measurement.

---

## 18. Risks and unresolved questions

### 18.1 Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Admission can pass while throughput is useless (V4.1-Flash: 4.25 GB/token of expert I/O on a 3 GB/s device ≈ 1.4 s/token before compute) | high | `COSTS` must report `storage_read_bytes_per_decode_step`, `exposed_stall_ns_per_decode_step_estimate` and `throughput_estimate_tok_s`; the CLI should warn when the estimate is below a user-settable floor. **Open question for the owner: warn, or refuse above a threshold?** The design chooses *warn + declare*, because refusing would violate #37's "backing storage determines capacity" rule — but the user must not be surprised. |
| `queue_class` cannot be enforced by the OS on macOS — there is no I/O priority API for `pread` | medium | Enforce by *in-flight byte budgets per class* and separate worker/queue assignment, which is what the measurement supports. Do not depend on kernel priority. Document the limitation. |
| Block CRCs cost ~51 MB for a 51 GiB lookup table (~0.1%) | low | Accepted; make them optional per segment so the bulk class can skip them. |
| New `BLAKE3` dependency for digests | low | `sha2` is already a dependency in the compiler; BLAKE3 is faster and tree-hashable. **Open question for the owner: add the dependency, or use SHA-256?** The design works with either; only digest width/algorithm changes, and the algorithm is recorded in the manifest. |
| Per-class in-flight budgeting can under-utilise the device for pure-bulk workloads | low | Budgets are derived from the declared `target_resident_bytes` and queue depth, not fixed constants. |
| A `.logan` artifact with a degraded converter-produced plan could be mistaken for a first-class plan | medium | The converter must mark the plan `degraded = true` in `flags`, and the runtime must print it and refuse tiered/out-of-core operation. |
| Two compilers producing "the same" plan with different cost models | low | `artifact_id` covers the plan; `IDENTITY` records `compiler_id`/`version` and `profile_digest`. |
| Journal-based in-place mutation (§8.11) is subtle | medium | It is not required for v1. Implement, test it against crash injection, and keep it behind the journal flag; the default add-then-commit path needs no journal. |
| Concurrent access by two runtime processes to one artifact | medium | Read-only payload is safe by construction. The state store needs an exclusive lock (flock/`LockFileEx`) — **not specified in v1; flagged as an open item.** |
| Windows atomic publish via `ReplaceFile` has different semantics from POSIX `rename` | low | The commit step is one small-file replacement; specify `ReplaceFile(MOVEFILE_REPLACE_EXISTING)` and verify with a crash test on real Windows hardware (the engine spec already forbids WSL for this class of verification). |

### 18.2 Unresolved questions for the owner

1. **Throughput floor policy.** Warn, or refuse, when a legal plan's estimated throughput is below a threshold
   (§18.1)? The design proposes warn-and-declare; #37's rules arguably permit refusing.
2. **Digest algorithm and dependency.** BLAKE3 (new dependency) vs SHA-256 (existing dependency).
3. **`SortedKeyTable` index kind.** Specified in v1 but optional to implement, or dropped until a model needs it?
   Current position: specified, not implemented — it costs nothing but spec text and prevents a v2 break.
4. **Where the model-level "compiled context ceiling" is enforced.** The design puts it in `PLAN.CONTEXT` and
   requires the runtime to refuse above it. Should the runtime also refuse *below* it when the plan's state
   requirement is sized for the ceiling (i.e. reserve for the ceiling, or for the requested length)? The design
   chooses: **reserve for the compiled ceiling**, because lazy growth past a verified capacity is exactly the
   overcommit #84 warns about.
5. **Batch/concurrency multiplication of the working set.** The design refines #37's "irreducible per-operation
   working set" to "per *admitted concurrent operation set*", since batching multiplies the requirement. This is
   a real refinement to #37's wording and should be agreed explicitly.
6. **Whether `.logan` v1 emission should be the default before the runtime can enforce it.** The design says no
   (Stage 1 dual-emit, default unchanged) — confirm.

---

## 19. How `.logan` v1 completely resolves issue #37

Legend: **[D]** SOLVED BY V1 DESIGN · **[I]** REQUIRES IMPLEMENTATION (the design is complete; code is not) ·
**[B]** BLOCKED/UNSOLVED.

**There are no [B] items.**

### 19.1 Vision and core thesis

| #37 requirement | Status | Where |
|---|---|---|
| Checkpoint + machine profile + context requirement + objective ⇒ automatically chosen per-tensor math format, physical layout/kernel ABI, backend/device, **backing store and bounded resident working set**, emitted as a deterministic mixed-format, out-of-core package **plus an executable plan** | **[D]** for the representation and emission contract; **[I]** for the optimizer→emitter wiring | §7, §8, §9.2. The candidate-generation and Pareto machinery already exists; the missing piece is population of `ResourcePlan` and emission, not design. |
| "The planner must not require the complete model or context state to fit physical memory" | **[D]** | §7.6, §10.1 step 7. Admission is per-pool working-set, per §8.7 `COSTS`. |
| "Immutable data may remain in the package and stream through bounded caches" | **[D]** | `BackingKind::PackageSegment` + `ResidencyPlan.target_resident_bytes`, §7.6/§7.7. |
| "Mutable state may use explicit runtime backing storage and bounded resident windows/pages" | **[D]** + **[I]** | `BackingKind::StateStore`, `STATE_REQ` (§8.6), state-store format (§8.10). |
| "The only memory-related hard failure should be an irreducible per-operation working set that cannot fit any legal memory pool" | **[D]**, with one **correction** | §11's failure taxonomy. **Correction:** the working set is per *admitted concurrent operation set*, not per operation — batching multiplies it, and #37's wording would otherwise admit a plan that fits for one sequence and fails for eight. See §18.2(5). |
| "Whole-model/context capacity failures should instead be caused by architectural limits, unsupported paths, address/format limits, or insufficient backing-store/disk capacity" | **[D]** | §11 failure taxonomy; no "does not fit RAM" entry exists. |

### 19.2 Required separation (the seven axes)

| #37 axis | Status | Where |
|---|---|---|
| 1. math representation | **[D]** | `math_format` + `scale_format` + `block` (§7.2) |
| 2. physical ABI/layout | **[D]** | `layout` + `kernel_abi` (§7.3), registry-backed |
| 3. backing home / storage tier | **[D]** | `BackingPlan` (§7.6) |
| 4. resident cache/working-set policy | **[D]** | `ResidencyPlan` (§7.7) |
| 5. backend/device execution target | **[D]** | `ExecutionPlan` islands (§7.9), capability-checked |
| 6. legal transfer/access path | **[D]** | `AccessPlan` (§7.8) + `AccessClass` (intrinsic) + `queue_class`; the pool-level link graph is #56's job and the format consumes it |
| 7. legal runtime alternative targets | **[D]** | `VariantGroup` (§7.9), bounded and explicitly listed |
| "`Placement::{Resident,Streamed,Gpu}` is transitional because it conflates backing, residency and execution visibility. The end-state must represent those axes separately." | **[D]** | `AccessClass` is deliberately **not** a placement enum and folds none of the three; `Placement` survives only as a compatibility projection for un-migrated engines (§9.2). |

### 19.3 Universal tiered storage model

| #37 data class | Status | Where |
|---|---|---|
| dense/shared weights — package-backed, optionally resident/cached/streamed | **[D]** | `Stream` segment, splittable into row blocks so a bounded window exists (§7.5, §8.4) |
| routed experts — package-backed with bounded residency | **[D]** | `Stream` bundle per `(layer, expert)`; preserved as one contiguous load (§12.3) |
| PLE/ngram tables — package-backed streaming | **[D]** | `Gather` with unit/block geometry + block CRCs (§7.4, §7.5, §13.2) |
| full-attention KV — mutable backing + layer/block-aware resident windows | **[D]** | `Mutable` + `STATE_REQ` with block geometry (§8.6) |
| QSA/indexer state — mutable backing when needed | **[D]** | same, `role = indexer`, `kind = indexer` |
| prefix/speculative snapshots — explicit backing/residency policy | **[D]** | `durability = 2` state entries + `artifact_id`-keyed caches (§8.10, §10.5) |
| execution scratch — bounded resident-only working set | **[D]** | `AccessClass::Pinned` + `BackingKind::PinnedOnly`, contributes to `minimum_working_set_bytes` |
| "Immutable package records do not need duplicate spill storage" | **[D]** | `PackageSegment` never requires spill capacity; the invariant already has a passing test (§3.5) |
| "Mutable state gets an explicit backing-store allocation whose size is planned and checked" | **[D]** + **[I]** | §8.6 declared bytes; §10.1 step 8/9 checks and creates |
| "For cyclic full-attention scans, do not assume a fractional cache gives a proportional hit rate. Prefer layer/block residency choices and charge realistic storage bytes read per decode step." | **[D]** | `STATE_REQ.block_bytes` + `COSTS.storage_read_bytes_per_decode_step` (§8.7); residency is expressed per state region, never as a global fraction |
| "Future lower-precision KV/state representations are representation candidates only when kernels and quality evidence exist." | **[D]** | `STATE_REQ.repr` is a registry id; a `VariantGroup` for state is legal **only** if the compiler emitted it, which requires a kernel ABI and recorded `quality_loss_ppm` |

### 19.4 Heterogeneous machine + storage profile

| #37 requirement | Status | Where |
|---|---|---|
| CPU ISA/topology and memory-bandwidth class | **[D]** + **[I]** | `MachineProfile` v2 (§9.3); the probe exists for ISA, bandwidth class is new |
| one or more GPUs/accelerators and backend capabilities | **[D]** + **[I]** | capability set in `CAPS`; multi-device probing extends the existing probe |
| memory-pool topology: UMA, host RAM, pinned staging, per-device VRAM, accelerator-private | **[D]** + **[I]** | `memory_pools` with `class` and `addressable_by` (§9.3) — the IR already models separate pools (`discrete_host_and_vram_pools_remain_separate`) |
| **storage pools**: filesystem/device identity, capacity/free bytes when planning locally, sequential/random bandwidth/latency class, direct/native async I/O capabilities, alignment constraints | **[D]** + **[I]** | `StoragePoolProfile` (§9.3) and `StoragePoolBudget` (already in the IR). The probe is entirely new — `MachineProfile` today has no storage notion at all. |
| device/queue concurrency and synchronization/transfer constraints | **[D]** + **[I]** | `io_queue_depth` + `devices[].queues/links`; the queue/link model is #56's, and the format consumes it |
| "On Apple Silicon, CPU/GPU/ANE may be distinct execution devices while sharing one UMA capacity. SSD/backing storage is a distinct capacity tier, not extra UMA." | **[D]** | §13.3; `MemoryPoolBudget`/`StoragePoolBudget` are separate domains by construction |
| "free_bytes is a volatile runtime/planning observation and should not be baked into the stable execution ABI" | **[D]** | §9.3; the plan stores only required bytes |

### 19.5 Capacity model and reporting

| #37 metric | Status | Where |
|---|---|---|
| virtual/logical bytes | **[D]** | `COSTS 0x0001` |
| minimum resident working set per pool (**hard memory requirement**) | **[D]** | `ResidencyPlan.minimum_working_set_bytes`; `COSTS 0x0004..0x000B` |
| resident target/cache bytes per pool (**performance choice**) | **[D]** | `ResidencyPlan.target_resident_bytes`; `COSTS 0x000C..0x0013` |
| immutable package-backed bytes | **[D]** | `BackingPlan{kind: PackageSegment}`; `COSTS 0x0002` |
| mutable backing-store bytes (**hard storage requirement**) | **[D]** | `STATE_REQ.bytes`; `COSTS 0x0003` |
| package/recompile journal/scratch storage | **[D]** | `COSTS 0x0014 scratch_bytes`, `0x0015 journal_bytes`; §8.11 |
| expected storage/transfer bytes per token or phase | **[D]** | `COSTS 0x0016..0x0019`; `AccessPlan.expected_read/write_bytes_per_step` |
| headroom / memory pressure | **[D]** | `COSTS 0x0021 headroom_bytes[pool]` |
| quality/context/latency/throughput metrics | **[D]** | `COSTS 0x0020, 0x0022..0x0025` |
| "**Never** use `total_state_bytes <= physical_ram` as feasibility" | **[D]** | §10.1: no such step exists; §11: no such failure reason exists |

### 19.6 Planner/runtime boundary

| #37 requirement | Status | Where |
|---|---|---|
| compiler decides legal representations, backing stores, resident/cache policies, execution targets | **[D]** | §7, §9 |
| runtime decides timing, eviction, prefetch **within those explicit bounds** | **[D]** | §10.4 permission table |
| runtime may choose only compiler-approved alternatives | **[D]** | `VariantGroup`; §10.4 row 3 |
| startup validates execution ABI/kernel capabilities | **[D]** + **[I]** | §10.1 step 6 |
| startup validates irreducible working-set fit | **[D]** + **[I]** | §10.1 step 7 |
| startup validates required backing-store capacity | **[D]** + **[I]** | §10.1 steps 8–9 |
| startup validates storage/access-path availability | **[D]** + **[I]** | §10.1 steps 6, 8 |
| startup validates the compiled context ceiling | **[D]** + **[I]** | `PLAN.CONTEXT`; §10.1 step 12 / §8.8 |
| "must not reject a plan merely because total virtual/pageable state exceeds physical RAM" | **[D]** | §10.1; the check does not exist |

### 19.7 Optimization objective

| #37 requirement | Status | Where |
|---|---|---|
| Pareto over quality loss, context, compute latency/throughput, resident memory pressure, immutable/mutable backing bytes, storage/transfer traffic and exposed stall, package size | **[D]** (engine) + **[I]** (persisted) | `tiered_optimizer` already trades these; `COSTS` + `OBJECTIVE` persist the chosen point and the rejected alternatives with reasons |
| "A valid 16 GiB machine plan may deliberately carry >16 GiB of model+context state if its resident working set is bounded and the remainder has legal backing storage" | **[D]** | Directly tested today by `ram_overcommitted_logical_state_is_feasible_when_backing_exists`; §20 T1/T2 make it an end-to-end acceptance test |

### 19.8 Completion themes

| Theme | Status | Where |
|---|---|---|
| **#68** machine profiles with memory **and storage** topology/capabilities | **[D]** + **[I]** | §9.3. Format side complete; the storage probe and `StoragePoolProfile` are new code. |
| **#81** architecture-aware context geometry + tiered state placement | **[D]** + **[I]** | Geometry is implemented (`context_plan.rs`: KV/GDN-recurrent/GDN-conv/QSA/PLE/MTP/scratch separately) and richer than #37 assumed; **placement** is the missing half and is exactly `STATE_REQ` + `ResourcePlan` for state (§8.6). #37's own correction — "do not treat all non-expert model bytes as permanently resident fixed state; the ~10 GiB `fixed_model_state` bucket is not an acceptable permanent abstraction" — is satisfied: `fixed_model_state` disappears and is replaced by per-resource `ResourcePlan` with `PackageSegment` backing (§9.2). |
| **#82** Pareto optimizer with resident-vs-backing decisions and **no global RAM capacity wall** | **[D]** + **[I]** | `tiered_optimizer` already has no RAM wall and is per-pool. **The wall lives in one function**: `check_uma_pool` in `pipeline.rs` on the non-optimizer path. #81/#82's reopening is therefore closed by *deleting that gate and populating `resources`*, not by redesign. |
| **#83/#84** package/runtime plan persistence and enforcement | **[D]** + **[I]** | §8 (mandatory, digest-bound, package-local plan) and §10.1 (12-step startup contract). #83's recommended layout is adopted with two corrections: the optimizer report is a **non-authoritative** `PROVENANCE` report (not part of the contract), and the plan is a first-class TLV, not a serialized in-memory struct. |
| **#85** low-space package mutation | **[D]** + **[I]** | §8.11's two strategies. v1 makes strategy 1 (add-then-commit, no journal) the default and *specifies* the journal so strategy 2 needs no format change. The format does not structurally prevent #85's stronger `extra_disk ≈ new_bytes` property; it is not implemented in v1. |
| shared runtime out-of-core state/backing abstraction + Qwen vertical proof | **[D]** + **[I]** | §10.2 generalization of `ResidencyManager` from `ExpertKey` to `RegionKey`; §13 is the Qwen vertical mapping |
| deterministic cost/objective decisions with rejected-alternative explanations | **[D]** + **[I]** | `OBJECTIVE` section (§8.8); the `TieredRejectedCandidate` data already exists |
| mixed representation package/runtime dispatch | **[D]** | per-segment `math_format`/`scale_format`/`block` + `layout` + `kernel_abi`; `RepresentationKey` extended with block geometry (§7.2) |
| calibrated quality sensitivity | **[I]** | Out of format scope. The format provides `COSTS.quality_loss_ppm` as the carrier and `ProvenanceEntry.gate` as the evidence link; calibration is a modelling task. |
| Metal/CUDA/Vulkan/ROCm target families as capabilities justify them | **[D]** + **[I]** | Capability bits + registry ids (§12.4, §15.1). The format is target-neutral; each backend is a registry/lowering task, not a format task. |
| explicit I/O, transfer and framework-boundary costs rather than treating storage/accelerator FLOPS as free | **[D]** + **[I]** | `COSTS 0x0016..0x0021`, `AccessPlan.expected_*_per_step`, and §17's measured device profile as the calibration input |

### 19.9 Acceptance criteria from #37

| #37 acceptance criterion | Status | Where |
|---|---|---|
| "A normal user can request automatic target/context/objective and receive an **inspectable deterministic plan** without understanding ISA/layout/storage details" | **[D]** for inspectable/deterministic; **[I]** for the CLI/UX surface | `--target auto --max-context/--require-context N --optimize --plan-choice` already exist; `.logan` adds `logan inspect <artifact>` printing `COSTS` and `OBJECTIVE` with rejected alternatives. The report is the deliverable, not a debugging aid. |
| "A model/context larger than RAM remains runnable when legal backing storage exists" | **[D]** + **[I]** | §20 T1/T2. This is the acceptance scenario. |
| "The selected plan reports its minimum working set, resident targets, backing-store need and I/O consequences" | **[D]** | §8.7 |
| "disk/backing exhaustion is reported precisely" | **[D]** | §8.6 + §11: required, available, deficit, and which state |
| "runtime validates the plan without silently replanning" | **[D]** | §10.1; there is no replan path at load |
| "exact placements cannot migrate" | **[D]** | `CAPS` is a subset check over ABI/layout/kernel/pool/access-path capabilities; it is not an identity match, so a *compatible* machine runs the artifact and an *incompatible* one is refused (§8.5) |
| "approved alternatives are explicit" | **[D]** | `VariantGroup` (§7.9) |
| "manual overrides remain available for reproducible benchmarking" | **[D]** + **[I]** | Manual quant/layout/backing constraints are already hard-constraint inputs to the optimizer; the format records the resulting plan. The `--quant-rules`, `--quant-floor` surfaces persist. |

### 19.10 Where I disagree with #37 (stated explicitly, not quietly ignored)

1. **"Per-operation working set" should be "per admitted concurrent operation set."** Batching multiplies the
   irreducible requirement; the current wording would admit a plan that fits for one sequence and fails for
   eight. §10.1 step 7 and §11 use the corrected formulation. **This should be amended in #37.**
2. **`Placement` should be deleted, not just deprecated.** #37 calls it transitional. Keeping it as a
   compatibility projection for un-migrated engines is the pragmatic reading, but it must not remain in any
   emitted plan as an authority — `.logan` emits no `placement` field at all, only `ResourcePlan`s. §9.2.
3. **`optimization.json` in the package should be explicitly non-authoritative.** #83 lists it as optional; the
   design makes it a `PROVENANCE` *report* whose absence or disagreement never affects correctness. §8, §14.
4. **#37's "immutable data may remain in the COLI package" should now read "in the `.logan` artifact."** The
   requirement is about immutability and backing, not about COLI; the sentence should be re-pointed so it does
   not become an argument for keeping COLI.
5. **Block-granular addressability and verification should be an explicit requirement, not an implied one.**
   #37 covers KV block residency but never states that a *package* payload must be unit-addressable. Without it
   the En/gram/PLE workloads cannot be made verifiable or coalescable — this design adds it as R5 and it should
   be added to #37.
6. **The plan must declare a throughput/stall estimate, not only bytes.** #37 lists "storage/transfer traffic
   and exposed stall" as an objective axis; §17 shows a case where the bytes are legal but the throughput is
   catastrophic (4.25 GB/token). The format must carry the estimate so the user sees it *before* the first
   token (§18.1).

---

## 20. Acceptance tests that would prove #37 is closed

Test machine: **MacBook Air M2 (Mac14,15), 16 GiB unified memory, internal NVMe** — measured at ~3.0 GB/s bulk
and ~9.2 k IOPS random (§17.3). All tests assert against `logan inspect` output and runtime telemetry, so they
verify the *artifact*, not just the process exit code.

**T1 — Immutable state far larger than RAM runs with a bounded working set.**
Artifact: the real `Qwen3.8-Flash-Next-REAP-288-MXFP4-Apple8` compiled to `.logan` (90.5 GiB) with
`--require-context 8192 --optimize`.
Assert: `COSTS.virtual_logical_bytes > 16 GiB`; `COSTS.minimum_working_set_bytes[uma0] < 12 GiB`;
`PROVENANCE` shows zero `Requantize` ops; execution completes; peak RSS < 12 GiB; every segment digest verifies.

**T2 — Mutable state larger than RAM has a legal backing plan and executes.** *(the headline #37 scenario)*
Artifact: same model, `--require-context 131072`. The compiled context state (KV + GDN recurrent + GDN conv +
QSA index + MTP) is deliberately forced beyond the resident budget.
Assert: `COSTS.mutable_backing_bytes > COSTS.target_resident_bytes[uma0]`; a state store is created whose size
equals `STATE_REQ.bytes`; the run proceeds; peak RSS < 12 GiB; `COSTS.storage_read_bytes_per_decode_step > 0`;
`exposed_stall_ns_per_decode_step_estimate > 0` and is non-zero in telemetry.
**Assert explicitly that no error message anywhere contains "does not fit in RAM" or an equivalent.**

**T3 — A 51.2 GiB sparse lookup table is never materialized.**
Assert: for the `Gather` class, resident bytes never exceed the declared `target_resident_bytes`; the observed
resident delta attributable to the lookup class stays under a declared bound (proposed: 256 MiB); block reads
are coalesced (observed read size distribution shows blocks, not 256 B units); block CRCs verify on every read.

**T4 — Deterministic identity.**
Compile the same source twice with the same request. Assert byte-identical `MANIFEST` and `PLAN` and equal
`artifact_id`. Then compile with a different quant rule and assert a **different** `artifact_id` and that the
two artifacts cannot share a prefix cache directory.

**T5 — Precise refusal, never a RAM refusal.**
Four cases: (a) state-store requirement exceeding free space ⇒ error names the state id, required, available,
deficit; (b) artifact requiring an unadvertised `kernel_abi` ⇒ error names the id; (c) `--require-context`
above the compiled ceiling ⇒ error names requested vs compiled; (d) a synthetic irreducible working set larger
than any pool ⇒ error names the pool, required and available. Assert that no refusal reason is phrased in terms
of total model or total context size.

**T6 — Stale/substituted plan is rejected pre-load.**
Replace `PLAN` with one from a different artifact. Assert rejection before any payload read, naming the digest
mismatch. Repeat with a bit flipped inside `PLAN` and inside `MANIFEST`.

**T7 — Corruption and truncation.**
Truncate `MANIFEST`, each section, and one payload shard; flip one bit in a `Gather` block. Assert: metadata
corruption rejected at open; payload corruption detected by the block CRC and reported with the segment id and
block index; the segment-level digest detects corruption where block CRCs are absent; repair (re-emit that
segment) restores a verifying artifact. Run the four fuzz targets over a corpus of truncated and bit-flipped
real artifacts with no panic and no allocation above the declared cap.

**T8 — Class isolation under load.** *(operationalizes §17)*
While expert `Stream` streaming runs at full tilt, assert that `Gather` p99 latency stays within a bound the
plan declares, and that bulk throughput is not reduced by more than a declared fraction — i.e. per-class
in-flight budgeting actually works on macOS, where there is no kernel I/O priority to lean on.

**T9 — Token identity against the current runtime.**
For the tiny committed fixture (`fixtures/qwen4_moe_tiny`) and the real Qwen3.8 package, a `.logan` artifact
produces token-identical greedy output to the existing COLI artifact, and Apple8 payload bytes are
bit-identical to the C oracle (`apple8_rust_c_identity` re-pointed at `.logan` payloads). This is the migration
safety net and the correctness gate for §14.

**T10 — Compilation is bounded-memory and resumable.**
Compile the 90.5 GiB source under a hard RSS cap (e.g. 3 GiB). Assert success. Kill at 50%, restart, assert
completion and a byte-identical `artifact_id` to an uninterrupted compile. Assert that an aborted compile
leaves only reclaimable garbage and that the preflight deficit message appears when free space is short.

---

## 21. Staged implementation plan

Ordered so that every stage is independently valuable and the risky parts come after the cheap proof.

**Stage A — artifact crate and round-trip (no behaviour change).**
1. `logan-artifact`: TLV reader/writer, `#![forbid(unsafe_code)]`, manifests, section tables, file tables,
   segment table, symbols, caps, state requirements, costs. Round-trip + structural-rejection tests.
2. Four fuzz targets seeded with truncated/bit-flipped real COLI manifests.
3. `logan inspect <artifact>` reading a `.logan` and printing `COSTS`/`OBJECTIVE`.
*Evidence of done:* round-trip byte-identity test; fuzzers run clean; `inspect` prints the real model's segment
inventory.

**Stage B — emission from the existing planner.**
4. Populate `MemoryPlan.resources` from the selected plan (the single highest-value change).
5. Emit `SEGMENTS`/`SYMBOLS`/`CAPS`/`STATE_REQ`/`COSTS` alongside the existing COLI writer; `--format logan|coli|both`.
6. Carry block geometry through from the frontends: expert bundles, dense tensors (splittable), PLE/n-gram
   shards (`unit_bytes`/`block_bytes`), multimodal assets.
*Evidence:* T9 token identity and T4 determinism.

**Stage C — runtime admission and the tiered path.**
7. `MachineProfile` v2 with `memory_pools` + `storage_pools` + `io_queue_depth`; probe storage
   (seq/random class, free bytes, async-I/O and alignment capabilities).
8. `.logan`-driven startup per §10.1, including **deleting `check_uma_pool` as an admission gate**.
9. State store: create/validate/resume (§8.10) and bind `STATE_REQ` to a writable pool or `--state-dir`.
10. `ResidencyManager` → `RegionKey` generalization plus `eviction_priority`, `access_class`, `queue_class`.
*Evidence:* T1, T2, T5, T6.

**Stage D — class scheduling and the out-of-core vertical.**
11. Per-class in-flight budgets, prefetch leads, coalescing for `Gather`, `exposed_stall` telemetry.
12. Qwen3.8-Flash-Next vertical: PLE as `Gather` with a small hot block cache; KV/GDN/QSA state in the state
    store with layer/block residency; prefix caches keyed on `artifact_id`.
*Evidence:* T2, T3, T8, T10.

**Stage E — V4/DSpark/CUDA readiness.**
13. DeepSeek-V4 frontend emits segments/bindings for CSA2 KV sharing, Engram `Gather` tables, DSpark's separate
    expert group and vision islands. (The frontend already exists; it emits records today.)
14. CUDA capability set + layout/KCUDA ABI registry ids + `AccessKind::Staged` for pinned staging pools.
*Evidence:* a V4-shaped synthetic fixture (already used in `pipeline.rs` tests) compiles, inspects and admits on
a machine with insufficient VRAM for the full model.

**Stage F — low-space mutation and deprecation.**
15. Compaction/reclaim pass and the optional journal (§8.11), with crash-injection tests.
16. COLI emission off by default; the legacy reader stays; converter with `degraded = true` marking.
*Evidence:* #85's `extra_disk ≈ new_bytes` property reproduced; an interrupted in-place recompile never
exposes a manifest/plan/segment disagreement.

**Explicitly not in this plan:** removing COLI, implementing every backend, distributed shards, training,
encryption. Each is either out of scope or a separate initiative.

---

## 22. Summary of repository changes made by this work

Two documents were added. **No code was changed, no format was implemented, and no existing artifact was
modified.** Untracked research directories (`.research/`, `.ngram-research/`, `tools/proto/`) were left
untouched.

**They are left uncommitted, deliberately.** The working tree is mid-merge (unresolved conflicts in
`Cargo.toml`, `logan-metal/metal/backend_metal.mm` and `logan-metal/src/lib.rs`) and carries staged in-flight
work from other sessions. Committing now could land a partial tree on top of unresolved conflicts. Review and
commit explicitly:

```bash
git add docs/logan_model_format_v1.md docs/adr/0001-logan-format-supersedes-coli.md
git commit -m "docs: design .logan v1, the successor to COLI (#37)"
```

- `docs/logan_model_format_v1.md` — this document.
- `docs/adr/0001-logan-format-supersedes-coli.md` — the decision record establishing `.logan` as COLI's
  successor, that v1 is designed around tiered/out-of-core execution, that v1 is intended to close #37, and
  that COLI remains temporarily supported.

No user-facing material claims `.logan` is implemented or production-ready. It is not.

---

## 23. Reconciliation with concurrent Logan design work

While this design was being written, other sessions landed substantial related work in the same tree
(GGUF source loading, a quantization integrity policy, a Windows/CUDA package-format draft, a DeepSeek V4
Apple-Silicon plan, and a new `logan-spark` crate). Those artefacts were read and this design is reconciled
with them below. **They do not conflict with `.logan`; they constrain and complete it.**

### 23.1 Relationship

`docs/WINDOWS_X64_CUDA_FORMAT.md`, `docs/DEEPSEEK_V4_APPLE_SILICON_PLAN.md` and
`logan-compiler/src/quant/precision_policy.rs` are **target-specific compiler policy and package-layout
decisions**. They are exactly the kind of content `.logan` is supposed to carry. They are written against
COLI's record model because that is what exists today; every one of their decisions maps onto a `.logan`
segment, binding, capability or provenance entry, as follows.

### 23.2 Mapping of the Windows x86-64 + CUDA format onto `.logan`

| Windows/CUDA doc decision | `.logan` representation |
|---|---|
| "separate data by access pattern rather than reproducing GGUF tensor ordering" | `AccessClass` (§7.4) — the same principle, made a first-class field |
| §5.1 dense/always-hot tensors stay individually indexed records | `Stream` or `Pinned` segments, splittable into row blocks so a bounded window exists |
| §5.2 one streamable record per `(layer, expert)`, gate/up/down bundled, one cache key, one async request, one residency lifetime | one `Stream` segment per expert with three sub-matrix descriptors declared in its `SYMBOLS` binding. The bundling rationale is preserved verbatim — this is the design's expert path (§12.3) |
| §5.2 "the matrix descriptor must retain each matrix's own quant type; an expert is allowed to be mixed (Q4_K gate/up + Q8_0 down)" | **amendment adopted** — see §23.5(1): `RepresentationKey.repr_contract` must enumerate the bundle's per-matrix formats, not assume one |
| §5.3 PLE rows packed into fixed 4 KiB pages without altering row bytes | **exactly** `Gather` with `unit_bytes = 110`, `block_bytes = 4096`, `block_units = 37`, plus `unit_count`/`block_count` so the final partial page is declared rather than guessed (§7.5). The doc's proposed metadata list is the `.logan` block-geometry fields, spelled the same way |
| §5.3 "the runtime cache should cache raw 4 KiB pages or raw Q5_0 rows, never expanded FP32 PLE rows" | `math_format`/`scale_format` on the segment are authoritative; there is no expanded representation in the artifact, and `ProvenanceEntry.bit_exact` records that no numeric transform occurred |
| §6 native GGML quant blocks as the execution layout, `codec = none` | **already registered** as `layouts.ggml_native_blocks_v1 = 0x0201`; `.logan` carries it as the segment `layout` id. `codec = none` is `stored_bytes == logical_bytes` with no codec field at all — the design has no per-segment codec, which is a simplification the doc's requirement justifies |
| §7 dequantize inside the kernel, FP32 accumulate, no expanded expert copies in VRAM | unaffected by the format; the artifact's job is to guarantee no expanded copy is *stored* |
| §8 three-tier residency: VRAM execution-hot / host RAM staging and second-level cache / NVMe authoritative | `memory_pools` (device-private VRAM, host, pinned staging) + `storage_pools` (NVMe), with residency targets and minimum working sets per pool (§9.3, §11). The doc's "do not hardcode all remaining VRAM to experts" is `ResidencyPlan.target_resident_bytes` being a policy output, not a constant |
| §9 overlapped `ReadFile`/IOCP → aligned pinned staging ring → `cudaMemcpyAsync` → stream → fused kernel | `AccessKind::Staged` (a distinct access kind, §7.8) + an `io_feature` capability (`overlapped_iocp`, `direct_io`, `alignment`) + `queue_class` separating copy from compute. `AccessPlan.expected_read_bytes_per_step` sizes the ring |
| §9 "multiple outstanding expert reads", "never block the scheduler thread on storage completion", "coalesce adjacent expert records" | the `Stream` class's in-flight budget and the scheduler's typed completions; `.logan` adds nothing synchronous. §17's per-class budgeting is the mechanism |
| §9 cancellation cannot release pinned host buffers or VRAM leases before I/O/CUDA completion | unchanged from #43/#56; the artifact must not introduce a synchronous read path, which §17.5's "no default mmap" decision supports |
| §10 `ExpertKey{model_fingerprint, layer, expert, layout ABI, gate/up/down quant, kernel ABI}` | `RegionKey{artifact, segment, pool, repr}` where `repr = RepresentationKey{layout, kernel_abi, repr_contract}` (§23.5(1)). `artifact_id` replaces `model_fingerprint` and additionally covers the physical plan, so a repack or ABI change changes the identity |
| §11 machine-readable per-tensor provenance report with `identity`/`lossless_repack`/`explicit_requantize`, nominal bits both sides, transform recorded | `ProvenanceEntry` (§14.3), persisted in the artifact rather than emitted alongside it, so it cannot drift from the bytes |
| §11 "the verifier must fail a package when provenance claims `lossless_repack` but the source/target quant semantics differ" | `validate_quant_transition` is the single source of truth for the rule; the artifact carries the evidence and the runtime re-checks it. See §23.5(2) |
| §12 default policy `quant_policy = preserve`, `requantize = disabled` | the planner's default; `COSTS`/`PROVENANCE` report every segment's transform so a non-default run is visible without reading flags |
| §13 correctness gates 1–10 | §20's acceptance tests; gates 1–5 (inventory, fingerprint, byte-exact slicing, bundle byte parity, PLE page byte parity) are the `.logan` migration safety net (§16.2 Stage 1) |
| §4 profile `windows-x86_64-cuda-ggml-v1`, alignment 4096, resident alignment 256, min SM 6.1, `compiler_emission_supported = false` | **already registered** as `profile_id = 3`, `allowed_layouts = [0x0000, 0x0201]`, `gpu_family_min = 61`. `.logan`'s `CAPS` must reuse the registry's own field names (`required_runtime_features`, `allowed_layouts`, `gpu_family_min`, `target_profile_abi`, `execution_layout_abi`, `kernel_abi`, `record_alignment`, `io_granularity`, `resident_alignment`) rather than invent parallel ones — see §23.5(4) |

**Net effect: the Windows/CUDA document needs no changes to its content.** Its container sentences ("COLI
tensor records", "register in `abi/coli-target-registry.toml`") are re-pointed to segments and to the shared
registry, both of which are rename-level, not redesign-level.

### 23.3 Mapping of the quantization integrity policy

`logan-compiler/src/quant/precision_policy.rs` is implemented and is the authority. The design defers to it:

- `QuantFormat { GgmlQ4K, GgmlQ5_0, GgmlQ6K, GgmlQ8_0 }` and `nominal_weight_bits()` supply the per-tensor
  nominal ceiling that §14 describes.
- `QuantTransform { LosslessRepack, ExplicitRequantize }` is the vocabulary `ProvenanceEntry.ops` must use.
  The design's five-concept table (§14.2) does not add new operations; it only insists that *target physical
  layout* is not one of the two transforms — it is a registry `layout` id, which is why `ggml_native_blocks_v1`
  required no policy change at all. That is the test of whether the model of numerical provenance is right.
- `validate_quant_transition` is fail-closed on `LosslessRepack` (source must equal target) and forbids upward
  `ExplicitRequantize`. `.logan` must record the *inputs and outcome*, not re-derive the rule: the artifact
  stores `source_nominal_weight_bits`/`emitted_nominal_weight_bits` and `bit_exact`, and the verifier calls the
  same function.
- The doc's warning about "effective bits per weight" versus nominal weight-code bits is adopted as a stated
  invariant: `.logan` reports `stored_bytes`/`logical_bytes` as container facts and
  `nominal_weight_bits` as a precision fact, and never converts one into a claim about the other.

### 23.4 Mapping of the DeepSeek V4 Apple-Silicon plan

Its §3 "Compiler and package contract" asks for, and `.logan` provides:

| Requirement | `.logan` |
|---|---|
| "a small, inspectable V4 package manifest and compiler fixture" | `MANIFEST` + `PLAN` + `logan inspect` (§8, §21 Stage A) |
| "preserve source quantization semantics … a model-wide FP8 label is insufficient to choose every tensor's kernel" | per-segment representation contract (§7.2, §14.5) |
| "normalize all execution parameters into versioned metadata … reject unsupported combinations explicitly" | `MODEL` (bounded, length-prefixed, unknown fields skipped) + `CAPS` (§8.2, §8.5) |
| "give target, MTP and DSpark tensors distinct namespaces" | `SYMBOLS.role` + a separate resource group per component (§12.3) |
| **"explicitly record which auxiliary tensors were omitted and why"** | **amendment adopted** — see §23.5(3) |
| "role-level byte totals, tensor-relative offsets, strides, scale shapes, padded layout sizes, and checked range arithmetic" | `SEGMENTS` (offset/stored/logical/unit/block), `SYMBOLS` (`logical_bytes`), `COSTS`; all range arithmetic checked at open (§8.3) |
| "a shape-correct matrix with incorrect scale orientation is a correctness failure" | `scale_format` + `block` geometry are part of the representation identity, so a scale-orientation error is a *different* contract, not a silent success (§7.2) |
| "reuse the prepared-cache mechanism with identity including checkpoint hash, tensor/expert identity, source representation, output layout version, and kernel ABI" | exactly `artifact_id` + `SegmentId` + `RepresentationKey` (§7.1, §10.5). The prepared SSD cache should key on `.logan` identities rather than a bespoke scheme |
| "publish complete records atomically and detect truncated/corrupt records before device use" | immutable content-addressed segments + per-segment digest/block CRCs (§6.1, §8.3, §8.12) |

Its §5 arithmetic (window vs compressed history, incrementally allocated compressed pages, "measure the useful
allocation granularity rather than hardcoding huge pages", prefix reuse restoring unfinished tails) is
`STATE_REQ`'s job: block geometry per state component, `durability = 2` for prefix-reusable state, and declared
per-step read/write bytes so the granularity choice is a *measured* plan input rather than a constant.

### 23.5 Amendments to this design, adopted from the concurrent work

1. **`RepresentationKey` must carry a bundle contract, not a single quant.**
   `repr = RepresentationKey { layout: u16, kernel_abi: u16, repr_contract: u32 }`, where `repr_contract` is a
   registry id enumerating the per-matrix representation tuple of a bundled segment (e.g. `Q4_K/Q4_K/Q8_0`).
   Without this, a future CUDA repack or kernel ABI change could reuse incompatible bytes — the exact hazard the
   Windows doc's §10 identifies. This supersedes the design's earlier `quant: u16` field.
2. **Provenance op names follow `precision_policy.rs`.** `ProvenanceEntry.ops` uses
   `Repack | ExplicitRequantize | Passthrough`, and `ProvenanceEntry` records
   `source_nominal_weight_bits`/`emitted_nominal_weight_bits`. The policy function is the single source of
   truth; the artifact stores the evidence.
3. **`MODEL` records omissions.** A new `omissions: Vec<(name_id, reason_id)>` list, because the DeepSeek plan
   correctly insists that a target-only package state *which* auxiliary tensors were left out and why. Silently
   absent tensors are indistinguishable from lost ones.
4. **`CAPS` reuses the target registry's field names.** `required_runtime_features`, `allowed_layouts`,
   `gpu_family_min`, `target_profile_abi`, `execution_layout_abi`, `kernel_abi`, `record_alignment`,
   `io_granularity`, `resident_alignment`. The registry is already the authority for these; the format must not
   define a second vocabulary.
5. **`abi/coli-target-registry.toml` should eventually be renamed** to reflect that it is the shared target/layout
   registry, not a COLI artefact — it now serves Apple8, Linux CPU and Windows CUDA, and will serve `.logan`.
   This is a **rename-only change**, deliberately not performed here, because the tree is mid-merge and the file
   is consumed by `tools/gen_target_registry.py` and by generated Rust that other work in flight depends on.

### 23.6 Repository state note

At the time of writing, the Logan working tree has **an unresolved merge in progress** (`Cargo.toml`,
`logan-metal/metal/backend_metal.mm`, `logan-metal/src/lib.rs`), staged in-flight work from other sessions
(`logan-spark`, GGUF loading, `precision_policy`) and untracked research directories. **Nothing was committed by
this work.** The two design documents are left untracked in the working tree for the owner to review and commit
deliberately, so that a partial commit cannot land on top of unresolved conflicts.
