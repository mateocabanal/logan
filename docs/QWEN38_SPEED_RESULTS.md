# Qwen3.8-Flash-Next speed work — 2026-09-07

Implemented in the isolated `logan-qwen4-port` checkout on top of draft PR #97. Main checkout changes were preserved. This follow-up has not been published or merged.

## Retained change

`run_greedy_with` now uses the existing `prefill_token` for every prompt token except the last. This avoids discarded global HC tail and vocabulary projections without changing causal state. The final prompt token still produces logits exactly once. Normal chat's chunked/prefix prefill already skips these heads; this closes a gap in the standalone greedy runner, not a new chat-prefill algorithm.

Added `logan-chat/examples/phase_bench.rs`: separate model-load, prompt and decode timings; per-step token IDs and full-logit fingerprints; rejection of nonfinite logits. The `full` mode retains the old prompt behavior as a comparison oracle. The `skip` mode exercises the retained prompt change. Fingerprints are same-binary comparison aids, not portable reference hashes.

## Measurements

M2 MacBook Air, 16 GB; existing `Qwen3.8-Flash-Next-REAP-288-MXFP4-Apple8.coli`; configuration routing top-k 10, unchanged. Prompt `The capital of France is` encoded as plain text, five prompt tokens; six generated tokens, therefore five timed decode forwards. CTX=128, LOGAN_PROFILE=1, prefix reads/writes disabled. Runs were sequential on the live Mac; file caches and memory residency were not reset. These are short diagnostic runs, not a throughput qualification.

| Run | Cache | Prompt heads | Load s | Prompt s | Decode s / 5 steps | Whole process s |
|---|---|---|---:|---:|---:|---:|
| Baseline | 10/layer | Every row | 8.46 | 33.85 | 14.15 | 57.39 |
| No GDN aligned copy (rejected) | 10/layer | Last row | 8.46 | 82.30 | 12.70 | 104.11 |
| Prompt optimization | 10/layer | Last row | 8.27 | 21.01 | 14.94 | 44.81 |
| Smaller cache | Global 128 | Last row | 8.34 | 13.72 | 11.69 | 34.33 |
| Original cache repeat | 10/layer | Last row | 8.56 | 33.37 | 11.52 | 54.08 |
| Smaller cache repeat | Global 128 | Last row | 8.27 | 34.52 | 9.96 | 53.26 |

All six runs produced identical six-token outputs and identical fingerprints for all six vocabulary-logit vectors. This checks equivalence to the existing runtime; it does not establish model quality or agreement with the upstream model oracle.

The first prompt-only comparison was 22% shorter end to end, but repeated prompt latency varied widely. Do not claim a stable percentage speedup from these runs. Global-128 decode was 0.43–0.50 tokens/s versus 0.33–0.43 for the tested layer-local runs; ranges nearly overlap and sample sizes are small. Keep cache defaults unchanged pending longer alternating comparisons under controlled memory conditions.

The global cache read 12.53 GB of expert payload across the request (zero retained route hits), versus 7.93 GB with 10/layer (1,762 hits). Thus fewer I/O bytes alone did not predict lower elapsed time on this Mac. Memory residency and first-use overhead merit separate profiling. Full GDN Metal success count was zero; the inherited default uses BNNS for BF16 GDN.

## Rejected experiment

An opt-in path avoided aligning/copying BF16 GDN weights when full Metal GDN was disabled. It preserved the observed logits but raised request latency from 57 to 104 seconds. The runtime change and its flag were removed. Original aligned weights and state ownership remain intact. PLE stays on NVMe and uses bounded row reads.

## Reproduce

From the isolated checkout:

```sh
cargo build --release -p logan-chat --example phase_bench
CTX=128 LOGAN_PROFILE=1 QWEN_PREFIX_CACHE=0 QWEN_PREFIX_CACHE_WRITE=0 \
QWEN_GDN_METAL=0 QWEN4_CACHE_PER_LAYER=10 \
target/release/examples/phase_bench \
/Users/mateo/models/Qwen3.8-Flash-Next-REAP-288-MXFP4-Apple8.coli \
'The capital of France is' 6 skip
```

Use `full` for old prompt behavior. For the cache experiment, change to `QWEN4_CACHE_PER_LAYER=0 QWEN4_CACHE=128`; this does not change model routing top-k. Do not compare profile `total ms/tok` directly to steady decode: that existing counter divides prompt plus decode time by generated-token count. Use the harness's separate decode timings.

Next substantial work should qualify compressed dense GDN/shared weights on this checkpoint and profile steady decode with a longer fixed token stream. The MXFP4 expert package name does not imply compressed dense weights. Preserve sigmoid gating and HC semantics, and gate any repack or native-format change against logits and recurrent state.

## Verification

Final retained source: 26 library tests passed; the separately enabled real PLE shard-boundary test passed; release builds of `phase_bench` and `greedy_smoke` passed. The PLE test performs bounded reads at actual shard boundaries and closes the pending gate from PR #97. Existing warnings remain. Six real-model comparison runs completed, with six matching logit fingerprints and token IDs each. No benchmark process is intentionally left running.
