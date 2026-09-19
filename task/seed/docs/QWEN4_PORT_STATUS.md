# Qwen3.6 → Qwen4 port status

Updated 2026-09-07. Based on `132efcd1abd311973acf657c8a5967b0a7d72e04` in the isolated `feat/qwen4-port-deepseek-plan-20260906` branch.

## Inherited shared improvements

The Qwen3.6 milestone already put route-hit pinning, cached expert-region geometry, persistent shard handles, shared-expert/I/O overlap, native MXFP4 GDN and shared-expert execution, and prefix-cache correctness in code also used by Qwen4. These do not need duplicate model-specific implementations.

The compressed dense kernels dispatch on the actual weight representation. An MXFP4 expert package can still contain BF16 dense weights. Do not relabel those weights or apply Qwen3.6's MLX-folded RMSNorm convention to Qwen4's HC norms.

## Changes in this branch

- Build a shared, immutable PLE shard index once when opening the package. It stores manifest indices only. Sort shard ordinals numerically and reject duplicate/gapped ordinals.
- Read the common PLE scale once per token instead of once per head. Each n-gram head still issues a bounded row read from the original NVMe package. No row, shard, or full-table cache is introduced.
- Extend previous-route prefetch to the HC residual branch. `QWEN_PREV_ROUTE_PREFETCH` remains opt-in and scheduler mode continues to exclude speculative loads.
- Add `QWEN_QSA_FUSED_INPUT=1` to submit compressed Q/K/V/indexer projections together. If the indexer is not MXFP4, retain the existing QKV batch plus independent indexer projection.
- Add `QWEN_PLE_FUSED_INPUT=1` to batch compressed PLE key/value projections. Non-MXFP4 weights retain the existing path.
- Add a visible-text `greedy_smoke` example and an explicit real-package PLE boundary gate.

The new QSA and PLE switches default to off pending real compatible-package performance qualification. Existing Qwen3.6 defaults are unchanged.

## Validation

On the M2 Mac, the unmodified baseline passed 23 library tests. The modified branch passed 26 library tests, including the real Metal four-output MXFP4 check against scalar outputs, mixed-format decline without output mutation, and numerical shard ordering/gap rejection. Baseline and candidate release smoke binaries both built successfully.

The additional real-package boundary test is intentionally ignored by default and requires `LOGAN_PLE_TEST_PACKAGE`. It checks the first and last row of every shard against direct bounded reads, plus zero-width and out-of-range failures. It never reads a whole shard.

Real text validation uses the local `Qwen3.8-Flash-Next-REAP-288-MXFP4-Apple8.coli` checkpoint, prompt `The capital of France is`, eight generated tokens, context cap 128, prefix caching disabled, and a layer cache sized from the checkpoint's `num_experts_per_tok` (10 when inspected). No top-k override is applied to routing. Full run results must be collected before claiming token identity or a performance improvement.

Commands:

```sh
cargo test -p logan-qwen4 --lib -- --test-threads=1
LOGAN_PLE_TEST_PACKAGE=/path/to/model.coli cargo test -p logan-qwen4 real_ple_shard_boundaries_match_direct_range_reads -- --ignored --nocapture --test-threads=1
cargo run --release -p logan-chat --example greedy_smoke -- /path/to/model.coli 'The capital of France is' 8
```

## Remaining compressed-dense work

Native FP8 dense integration is a separate representation task. The generic Metal backend already understands E4M3FN with F32 block-128 scales, but runtime `WtBytes` and the full GDN/shared pipelines currently select MXFP4 explicitly. Changing only the format allowlist would be wrong: the native encoder also hardcodes format 7. A correct FP8 port needs an explicit weight variant, scale encoding conversion/validation, CPU fallback, ownership checks, and GPU parity for both GDN output gates.

Existing uncommitted work in the main checkout covers model-derived cache sizing and streamed-only n-gram API safeguards. This isolated branch leaves that work intact and does not claim it as part of its diff.

