# ADR 0001 — `.logan` replaces COLI as the canonical compiled model format

**Date:** 2026-09-09
**Status:** Accepted — owner-approved 2026-09-22
**Author:** Mateo + Hermes
**Related:** #37 (umbrella), #43, #56, #68, #81, #82, #83, #84, #85, #93
**Note:** Logan had no ADR convention before this record. This is the first, using a minimal format
(Context / Decision / Consequences / Status). If a different convention is preferred, this file should be
renamed rather than duplicated.

## Context

Logan compiles model checkpoints into a hardware-specialized artifact and executes them with a bounded fast-memory
footprint. The current native representation is **COLI** (`.coli`): a directory package of `manifest.coli` plus
`data-NNNNN.coli` shards, indexed by a record table, with CRC32C and a target-profile registry.

Three facts about the current state forced this decision:

1. **The compiled plan is not in the artifact.** `logan compile --plan PATH` writes an optional, external,
   bincode-serialized plan. The package has no pointer to it and no digest over it. A package with the wrong or
   missing plan opens and runs. Issue #83 requires the opposite.
2. **The tiered resource abstraction landed but is unused.** Commit `3c6bcd6` (#93) added the correct
   `ResourcePlan`/`ResourceBudget` model to `logan-ir` — backing, residency, access as separate axes, with
   per-pool working-set admission and tests proving RAM overcommit is feasible when backing exists. But
   `build_physical_plan` emits `MemoryPlan.resources = vec![]`, and the non-optimizer path still admits through
   `check_uma_pool`, i.e. `resident_bytes + expert_cache_bytes <= ram_bytes` — the exact global-RAM worldview
   issue #37 exists to remove.
3. **The payload model cannot express the workloads Logan now targets.** A record is atomic. The real
   Qwen3.8-Flash-Next artifact on the target machine is 90.5 GiB in 24 shards and 15,023 records, of which
   **56% is 128 opaque 400 MB blocks of n-gram lookup table** with no declared sub-record geometry and no
   checksum on partial reads. DeepSeek-V4.1-Flash adds ~183 GiB of Engram lookup tables and needs mutable KV,
   indexer and recurrent state with declared backing. COLI has no storage class, no block geometry, no mutable
   state format and no capability declaration.

## Decision

1. **`.logan` supersedes COLI as Logan's canonical compiled model representation.** COLI becomes legacy in
   architectural direction.
2. **`.logan` v1 is designed around tiered / out-of-core execution.** A `.logan` artifact is a compiled
   execution image: a small immutable plan plus physically homogeneous segments. It is not a tensor container.
3. **`.logan` v1 is intended to satisfy issue #37 completely**, with no requirement deferred to a v2 format
   break. The requirement-by-requirement mapping is in `docs/logan_model_format_v1.md` §19.
4. **The artifact carries a mandatory, package-local, digest-bound physical plan** (backing, residency, access
   class, transfer path, capability requirements, declared costs, and any compiler-approved alternatives).
   The runtime validates it at startup and may choose only timing within its declared bounds.
5. **COLI is deprecated but remains temporarily supported.** The legacy reader and existing runtime path
   stay available for old artifacts and parity work during migration, but new architecture and new source-neutral
   runtime work must not depend on COLI-specific records or naming.
6. **Safetensors/MLX remain first-class input/runtime sources.** Native `.logan` is the preferred compiled
   artifact, not a requirement that users convert open checkpoints before Logan can run them. Source adapters may
   share MetalIO, residency, scheduling and RouteScout through engine-neutral interfaces.
7. **Implementation is authorized.** Owner review was completed on 2026-09-22; staged migration may proceed
   while preserving correctness gates and legacy artifact readability.

## Consequences

- `logan-format` becomes the **COLI legacy reader**; `.logan` gets its own crate so legacy code is untouched.
- `MemoryPlan.resources` must be populated from the planner's existing `ResourcePlan`s, and `check_uma_pool`
  must stop being an admission gate. This converts #81/#82/#93 from partially-landed to actually effective, and
  is the single highest-value change in the programme.
- `MachineProfile` must gain storage pools and queue depth (#68); it currently has no storage notion at all.
- `ResidencyManager` must generalize from `ExpertKey` to a segment/pool key so the same tier system covers
  dense weights, experts, lookup tables, KV and scratch.
- A new runtime subsystem appears: the mutable **state store**, whose format v1 fixes but which does not live
  inside the artifact.
- A COLI→`.logan` converter is only useful as bulk repackaging; it cannot recover a plan that never existed.
  The real migration path is recompilation from the source checkpoint, which is cheap because compilation is
  streaming and bounded-memory.
- The C-oracle differential identity tests (`mateocabanal/colibri` remains the parity oracle) and the existing
  token-identity gates are the correctness safety net across the migration.

## Relationship to concurrent work

While this ADR was being drafted, other sessions landed `docs/WINDOWS_X64_CUDA_FORMAT.md`,
`docs/DEEPSEEK_V4_APPLE_SILICON_PLAN.md` and `logan-compiler/src/quant/precision_policy.rs`. These are
**complementary**, not competing: they define target-specific compiler policy and package layout, which is
exactly the content `.logan` must carry. Every one of their decisions maps onto a `.logan` segment, binding,
capability or provenance entry; the mapping is in `docs/logan_model_format_v1.md` §23. In particular the
quantization policy file is adopted as the single source of truth for representation provenance, and the
Windows/CUDA profile and GGML layout are already registered in `abi/coli-target-registry.toml` and need no
change. The registry file's name should eventually change to reflect that it is shared, but that is a
rename-only change deliberately not performed while the tree is mid-merge.

## Alternatives rejected

Keeping COLI with more fields; one monolithic file; one file per storage class as the organizing principle;
safetensors-style flat headers with no plan; GGUF-style untyped key–value metadata; a wholesale ELF/dynamic-link
model; a pure content-addressed store; an LSM-tree substrate for the sparse tables. Rationale for each is in
`docs/logan_model_format_v1.md` §5.

## Status

Accepted on 2026-09-22. COLI is now a deprecated legacy compatibility format; `.logan` is the canonical
compiled-format direction. Raw safetensors/MLX remains a supported first-class runtime source rather than a
second-class import path.
