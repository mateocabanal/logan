# Logan Experiment Ledger

Log every performance, correctness, architecture, storage, quantization, accelerator, or scheduling experiment here, including experiments that fail.

The purpose of this file is to stop Logan from repeatedly rediscovering the same conclusions and to make performance claims auditable.

## Rules

1. **Record before or at implementation time.** Every non-trivial experiment gets an ID before results are interpreted.
2. **State a falsifiable hypothesis.** Prefer "X reduces steady decode latency by >=5% at identical token/logit parity" over "try X".
3. **Preserve an A/B baseline.** Benchmark the candidate against a named baseline under comparable conditions.
4. **Correctness gates come first.** Record token identity, logit/recurrent-state tolerances, model/checkpoint, context, and relevant runtime flags.
5. **Never call noise a win.** If runs overlap materially or conditions differ, mark the result **INCONCLUSIVE**.
6. **Record rejected work.** Rejected experiments are as valuable as retained ones. State why they lost and whether the code was removed.
7. **Delete losing production paths when practical.** Do not accumulate permanent feature flags for experiments that are clearly rejected.
8. **Keep environment details.** Hardware, model artifact, commit, prompt/workload, context length, cache state, run order, and thermal/background caveats belong with performance results.
9. **Separate mechanism from policy.** A faster I/O or compute mechanism does not by itself justify removing scheduler/residency/state ownership.
10. **Update this ledger in the same change that changes an experiment's status.**

## Status vocabulary

- **PLANNED** — hypothesis recorded; implementation not started.
- **RUNNING** — implementation or measurement is in progress.
- **KEPT** — passed correctness gates and produced a repeatable benefit or a required architectural capability.
- **REJECTED** — lost its gate; production path should be removed or disabled.
- **INCONCLUSIVE** — evidence is insufficient or noisy; do not enable by default.
- **QUALIFIED** — technically/correctness validated, but not enabled by default because end-to-end benefit is unproven or workload-specific.
- **SUPERSEDED** — replaced by a later experiment; retain this record for provenance.

---

## EXP-001 — Skip discarded prompt heads in standalone greedy prefill

**Date:** 2026-09-07  
**Area:** Qwen3.8 prefill  
**Status:** **KEPT**

**Hypothesis:** For every prompt token except the final one, skip the global HC tail and vocabulary projection when their outputs are discarded, without changing causal state or decode logits.

**Baseline:** run_greedy_with computing prompt heads for every row.

**Candidate:** Use prefill_token for all prompt rows except the final prompt token; compute logits once on the final row.

**Workload:** M2 MacBook Air 16 GB; Qwen3.8-Flash-Next-REAP-288-MXFP4-Apple8.coli; top-k 10; CTX=128; five-token prompt; six generated tokens; prefix cache disabled.

**Result:** Six real-model comparison runs produced identical generated token IDs and identical same-binary full-logit fingerprints. Prompt measurements were noisy, but the work is semantically unnecessary and the retained path removes it without changing causal state.

**Decision:** **KEPT.** Correctness passed and the removed computation has no consumer. phase_bench was added to distinguish load, prompt, and decode timing.

**Source:** docs/QWEN38_SPEED_RESULTS.md.

---

## EXP-002 — Remove aligned BF16 GDN copy when full Metal GDN is disabled

**Date:** 2026-09-07  
**Area:** Qwen3.8 GDN / memory handling  
**Status:** **REJECTED**

**Hypothesis:** Avoiding the aligned/copy step for BF16 GDN weights will reduce setup/runtime overhead while preserving logits.

**Baseline:** Existing aligned-weight path.

**Candidate:** Opt-in no-aligned-copy path.

**Result:** Observed logits were preserved, but whole-request latency regressed from roughly 57 s to 104 s in the diagnostic run.

**Decision:** **REJECTED.** The flag and runtime path were removed. Keep the original aligned weights/state ownership.

**Source:** docs/QWEN38_SPEED_RESULTS.md.

---

## EXP-003 — Replace layer-local expert cache with a global 128-entry cache

**Date:** 2026-09-07  
**Area:** Qwen3.8 expert residency  
**Status:** **INCONCLUSIVE**

**Hypothesis:** A smaller global expert cache reduces memory pressure enough to improve decode despite lower expert reuse.

**Baseline:** 10 experts retained per layer.

**Candidate:** Global cache capacity 128.

**Result:** Short runs showed global-128 decode around 0.43–0.50 tok/s versus 0.33–0.43 tok/s for the tested layer-local runs, but ranges overlapped and prompt latency varied heavily. The global cache read about 12.53 GB of expert payload with zero retained-route hits versus 7.93 GB and 1,762 hits for 10/layer.

**Decision:** **INCONCLUSIVE.** Cache defaults were left unchanged pending longer alternating tests under controlled memory/cache conditions. Fewer I/O bytes did not predict elapsed time.

**Source:** docs/QWEN38_SPEED_RESULTS.md.

---

## EXP-004 — Apple8 direct MetalIO routed-expert execution

**Date:** 2026-09 (qualified before 2026-09-20)  
**Area:** Qwen4 expert I/O + Metal execution  
**Status:** **KEPT**

**Hypothesis:** Keep routed expert gate/up/down bytes in one MetalIO-backed slot and consume them directly in a fused Metal MoE kernel, avoiding host dequantization and per-matrix submission overhead.

**Candidate mechanics:**
- issue all routed expert loads before waiting,
- batch MetalIO completion,
- retain slot ownership through compute,
- fused gate/up -> SwiGLU -> down -> weighted top-k reduction,
- model-driven top-k.

**Result:** This path passed the project's real-model A/B qualification and is part of the normal max-performance policy.

**Decision:** **KEPT.** QWEN_APPLE8_DIRECT=1 is a validated default. Scheduler/residency ownership remains separate from the underlying MetalIO mechanism.

**Source:** docs/MAX_PERFORMANCE_DEFAULTS.md, logan-qwen4/src/lib.rs, logan-metal/metal/apple8_metalio_direct.mm.

---

## EXP-005 — Split-phase routed MoE submit/finish

**Date:** 2026-09 (qualified before 2026-09-20)  
**Area:** GPU/CPU overlap  
**Status:** **KEPT**

**Hypothesis:** Submit the fused routed-expert Metal work without immediately waiting, perform independent host/shared-expert work, then finish the GPU operation.

**Candidate:** moe_topk_begin() / moe_topk_finish() with QWEN_APPLE8_OVERLAP=1.

**Result:** Passed real-model A/B qualification and is enabled in the max-performance defaults.

**Decision:** **KEPT.** This is the seed of Logan's larger GPU-resident execution-island strategy.

**Source:** docs/MAX_PERFORMANCE_DEFAULTS.md, logan-qwen4/src/lib.rs.

---

## EXP-006 — Overlap shared expert with routed-expert MetalIO

**Date:** 2026-09 (qualified before 2026-09-20)  
**Area:** I/O/compute overlap  
**Status:** **KEPT**

**Hypothesis:** Since the shared expert depends only on the layer activation, execute it while routed expert NVMe->UMA transfers are outstanding.

**Candidate:** QWEN_SHARED_IO_OVERLAP=1.

**Result:** Passed real-model A/B qualification and is enabled in the max-performance defaults.

**Decision:** **KEPT.**

**Source:** docs/MAX_PERFORMANCE_DEFAULTS.md, logan-qwen4/src/lib.rs.

---

## EXP-007 — ANE GDN GPU-tail handoff

**Date:** 2026-09-08  
**Area:** ANE + Metal interoperability  
**Status:** **QUALIFIED**

**Hypothesis:** Keep ANE-produced GDN projection surfaces off the CPU by feeding them directly into a GPU gather/recurrent/gating/output tail.

**Result:** Six-forward tests reduced explicit CPU surface maps from 252 to 36 per token. Argmax/top-10 matched the ANE baseline; worst relative L2 was about 0.1147% with cosine >=0.999999359; repeated GPU-tail logits were bit-identical. Whole-model timing was noisy and did not establish a stable speed win.

**Decision:** **QUALIFIED, not default.** The execution-boundary idea is retained; end-to-end ANE use still requires latency, quality, and fallback gates.

**Source:** docs/ane_async_20260908.md, docs/ane_execution.md.

---

## EXP-008 — Flash-MoE-style affine Q4 FMA dequant/GEMV

**Date:** 2026-09-20  
**Area:** Metal / MLX affine Q4  
**Status:** **RUNNING**

**Hypothesis:** For MLX affine Q4, rearranging (code * scale + bias) * x to fma(code, scale * x, bias * x) reduces Metal decode latency for Q4 GEMV without unacceptable numerical drift.

**Baseline:** Logan native MLX affine Q4 format 16 using the direct affine expression.

**Candidate:** Experimental Q4-FMA format/path selected by LOGAN_Q4_FMA=1; 5/6/8-bit affine paths remain unchanged.

**Required gates:**
- kernel/reference numerical tests,
- same model/checkpoint and prompts,
- generated-token identity for deterministic decode,
- logit error/fingerprint comparison appropriate to the changed floating-point association,
- alternating A/B timing on representative Qwen3.8/Qwen3.5/oQ4e matrix shapes,
- no default enablement until a repeatable win is measured.

**Decision:** Pending measurements. Do not claim a speedup from Flash-MoE's hardware/model result alone.

---

## EXP-009 — Larger GPU-resident HC -> GDN -> HC execution islands

**Date:** 2026-09-20  
**Area:** Qwen4Exp layer execution / synchronization  
**Status:** **RUNNING**

**Hypothesis:** Logan's remaining Qwen3.8 bottleneck is dominated by dense/GDN/HC work rather than expert I/O; keeping HC intermediates and adjacent GDN projections/state on GPU across logical operation boundaries will reduce CPU<->GPU synchronization and allocation overhead substantially.

**Baseline:** Current hc_mix() creates host Vec<f32> intermediates and invokes separate matmul operations around GDN/attention work.

**Candidate direction:**
1. fuse grouped HC RMSNorm + low-rank down + activation + low-rank up + sigmoid mixing/injection,
2. keep the resulting activation in Metal-visible storage,
3. feed it directly into GDN/QSA input projection kernels,
4. preserve recurrent/state ownership and exact causal ordering,
5. expand the island only after each boundary passes parity and A/B timing.

**Why this target:** A prior Qwen3.8 profile measured roughly 33.1 s per steady decode forward, with GDN (~14.4 s), HC (~5.8 s), attention (~5.6 s), forward tail (~4.6 s), and shared expert (~2.5 s), while the expert-I/O envelope was only ~1.06 s. That makes dense execution/synchronization a higher-priority target than further expert-cache tuning for that regime.

**Required gates:**
- exact token identity,
- bounded logit/recurrent-state error,
- GDN/PLE/QSA state progression unchanged,
- no extra CPU-visible synchronization on the island path,
- explicit per-phase timing before/after,
- reject individual fusions that regress end-to-end latency even if microbenchmarks improve.

**Decision:** First implementation target is the HC boundary. Subsequent island expansion is contingent on measurement.

---

## EXP-010 — Whole-route online transition prediction

**Date:** 2026-09-21  
**Area:** routed MoE / expert prediction  
**Status:** **SUPERSEDED**

**Hypothesis:** A bounded layer-local transition model can predict the entire next top-k route better than simply reusing the previous route.

**Result:** On the native Apple8 8-expert fixture, adjacent-route reuse was about 50%, while the learned full-route predictor reached only 38.9%. Speculative prefetch produced 30 useful versus 48 wasted loads and no repeatable latency win.

**Decision:** **SUPERSEDED.** Predicting the whole route wastes effort on experts that are already resident. The retained formulation predicts only cold arrivals absent from the previous route.

---

## EXP-011 — Confidence-gated temporal + spatial expert prefetch

**Date:** 2026-09-21  
**Area:** routed MoE / MetalIO / expert residency  
**Status:** **SUPERSEDED (closed by EXP-031)**

**Closure (2026-09-22, this slice):** This entry opened the confidence-gated
speculative-prefetch branch. It is closed as **rejected on wall time**: EXP-031
measured the best available configuration of this mechanism as a paired 3:1 loss at
+9.67% with 100% readiness and 0 late arrivals, and identified the queue-contention
mechanism; EXP-034 then closed its spatial-horizon extension (recall halves from
`h0` to `h1` and plateaus). The confidence gate this entry introduced is still the
shipped default and is still inert by construction (EXP-031). Retained as the
origin of the branch and for its 32-expert fixture findings.

**Candidate:** Predict only cold expert arrivals across tokens, combine that with same-token adjacent-layer routing correlation, and gate speculative I/O on measured confidence. The authoritative router and MoE arithmetic are unchanged.

**Measured results on M2:**
- 8-expert fixture: cold-arrival precision ~54%, recall ~50%; one run observed 35/35 useful physical speculative loads and zero wasted.
- Controlled 32-expert fixture exercising all experts: temporal overlap fell to ~4.5%, while adjacent-layer overlap was ~74.6% and previous-layer top-1 appeared in the next layer top-k ~90.5% of the time.
- Ungated temporal speculation on that 32-expert workload was harmful: about 120 speculative loads, 15 useful and 104 wasted. The online confidence gate correctly suppressed it.
- Prompt-seeded spatial speculation observed 50 useful prefetches, all 50 ready before demand and none late.
- A balanced warmed 20-run B/S/S/B test on a ~204 MiB controlled artifact with ~3 MiB experts was flat/slightly negative: baseline median 27.6 ms/token versus spatial 27.8 ms/token (~-0.72%).

**Decision:** **RUNNING, experimental only.** Accuracy and readiness are necessary but not sufficient for a throughput win. Real trained-model qualification is required before default enablement. Next target: Qwen3.8-Flash-Next GSQ-RCO, 512 experts / top-10 / 48 layers.

---

## EXP-012 — Qwen3.8 GSQ-RCO mixed-IQ split-GGUF qualification

**Date:** 2026-09-21  
**Area:** GGUF / quantization / Qwen4Exp storage  
**Status:** **RUNNING**

**Target:** ISTA-DASLab Qwen3.8-Flash-Next-GSQ-RCO-GGUF, IQ3_XXS budget, without requantization.

**Actual header:** qwen4exp; hidden 2560; 48 layers; 512 routed experts; top-10; context ceiling 262,144; two shards / 1,224 tensors. Shard 2 contains only `per_layer_token_embd.weight`, IQ4_NL `[160, 320001536]`.

**Qualification completed:**
- Added the checkpoint's mixed GGML formats, including Q2_0, Q5_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS.
- Corrected Q2_0 to current upstream geometry: 64 values / 18 bytes.
- Full-row deterministic dequantization matches current upstream GGML bit-for-bit for all nine newly required formats; the `dot_row` execution oracle also passes.
- Added split-GGUF discovery and per-tensor shard ownership; a synthetic cross-shard read test passes.
- Sparse copies of the real two-shard headers parse successfully and Logan's config validator derives 48 layers, 512 experts, top-10, 12 QSA layers, 36 GDN layers and PLE at layer 1.
- GGUF PLE now range-reads/dequantizes exactly one IQ4_NL row from shard 2 instead of materializing the ~28.8 GB table.
- Header accounting shows ~39.97 GiB of shard-1 storage is routed experts and remains lazy; non-routed startup weights total ~3.83 GiB.
- Native GGUF routed experts are now wired into the CPU correctness fallback and remain sliced on demand.

**Reference runtime:** Fresh upstream llama.cpp/ggml build at `~/CODE/llama.cpp-qwen38-ref`; `llama-debug` will provide authoritative `ffn_moe_topk` route traces once both shards are complete.

**Decision:** **RUNNING.** Container, split-file and CPU quant correctness gates are green. Full-model output/route/latency qualification remains pending the complete download.

---

## New experiment template

Copy this section for every new experiment.

    ## EXP-NNN — Short name

    **Date:** YYYY-MM-DD
    **Area:** subsystem
    **Status:** **PLANNED | RUNNING | KEPT | REJECTED | INCONCLUSIVE | QUALIFIED | SUPERSEDED**

    **Hypothesis:** A falsifiable expected outcome.

    **Baseline:** Exact baseline implementation/configuration.

    **Candidate:** Exact change being tested.

    **Environment:**
    - Commit:
    - Hardware:
    - OS:
    - Model/artifact:
    - Context/concurrency:
    - Prompt/workload:
    - Relevant flags:
    - Cache/thermal/background conditions:

    **Correctness gate:**
    - Token identity:
    - Logit tolerance/fingerprint:
    - State/cache invariants:
    - Other:

    **Measurements:**

    | Run/order | Baseline | Candidate | Delta | Notes |
    |---|---:|---:|---:|---|

    **Result:** What the evidence actually establishes.

    **Decision:** **STATUS.** Why it was kept/rejected/etc.; note whether experimental code/flags were removed.

    **Artifacts:** benchmark logs, scripts, profiler output, commit/PR.


## EXP-013 — RouteScout native-ANE learned expert predictor

**Date:** 2026-09-21  
**Area:** routed MoE / native ANE / expert prefetch  
**Status:** **RUNNING**

**Hypothesis:** A tiny learned predictor executed directly through `logan-ane` can forecast future Qwen3.6 routed-expert demand cheaply enough to hide SSD-streamed expert latency, while the native router remains authoritative.

**Target model:** `deepsweet/Qwen3.6-35B-A3B-MLX-oQ4-FP16` in `~/models/Qwen3.6-35B-A3B-MLX-oQ4-FP16`: 40 layers, hidden 2048, 256 routed experts, top-8, MoE intermediate 512.

**Baseline:** EXP-011 temporal/spatial heuristics plus EXP-010's retained cold-arrival transition predictor.

**Candidate architecture:** Host-maintained route state and layer-local expert embeddings feed a fixed-shape direct-ANE MLP. The first physical island is `96 -> 96 -> 64 -> 256`, spatial 16 with 8 live forecast lanes; native Qwen routing remains authoritative. No Core ML runtime is used.

**Native ANE qualification (M2):**
- Direct `logan-ane` MIL/private-ANE compilation and execution succeeded.
- ANE reported: h14g/h14, one ANE, 16 cores.
- Compile + load: **33.501 ms**.
- Warm evaluation: **119.696 us/dispatch** over 250 iterations.
- Oracle max absolute error: **9.78e-6**; RMS **6.80e-6**.
- Gate: `ROUTESCOUT_ANE_GATE: PASS`.

**Training data path:** `QWEN_ROUTESCOUT_TRACE_PATH` records the authoritative route event, layer, router entropy, margin, selected top-k expert IDs, and normalized gate weights. Decode flushes on each 40-layer cycle.

**Progress update — 2026-09-22:**
- Real Qwen3.6 route traces are captured across multiple prompt families (`rust`, `moe`, `science`, `hash`) plus longer decode traces. Hidden-state captures also exist for semantic-feature experiments.
- On a longer temporal 70/30 split, the original 96 -> 96 -> 64 -> 256 learned predictor reached route Recall@8/16/24 of **61.7% / 80.0% / 86.4%**. With an 8-expert cold-arrival budget it reached **63.1% recall**, but only **18.9% precision**; this is useful evidence of route structure, not a cross-prompt generalization result.
- Cross-prompt qualification is much harder. On the held-out Rust prompt the learned model reached **13.4% / 21.1% / 27.6%** route Recall@8/16/24, below the spatial transition baseline (**22.9% / 33.2% / 39.1%**) and hybrid transition baseline (**20.9% / 31.2% / 37.3%**). A low-rank/SVD feature variant was worse and should not be promoted.
- On the held-out hash prompt, transition priors are strong: hybrid route Recall@8 was about **61.8%** and cold-arrival recall at budget 8 about **69.7%**. Blending the learned arrival score into those priors improved the best observed figures to about **62.7% route Recall@8** and **70.8% cold-arrival recall@8**. This is a small but useful signal that the learned model may contribute orthogonal evidence as a residual scorer.
- A candidate-centric scorer path now exists: `tools/routescout_train_scorer.py` trains a shared **16 -> 16 -> 8 -> 1** expert scorer, and `logan-ane/examples/route_scout_scorer.rs` packs 8 target layers x 256 experts into one direct-ANE dispatch. It is not yet end-to-end qualified.
- The correctness-neutral online transition predictor is implemented in `logan-qwen4/src/route_predictor.rs`; authoritative routing remains unchanged.

**Remaining gates:**
1. Expand true held-out-prompt coverage and characterize variance by prompt/domain/layer.
2. Qualify the per-expert candidate scorer and residual/blended scoring against the transition baselines.
3. Test online adaptation, layer specialization, confidence calibration, multi-horizon prediction, and cost-aware objectives.
4. Integrate only the best predictor as prefetch/residency hints; never as authoritative routing.
5. Measure useful/wasted speculative bytes, ready-at-demand rate, eviction damage, expert wait, and end-to-end tok/s with balanced A/B runs.
6. Transfer the methodology to Qwen3.8-Flash-Next only after the Qwen3.6 path has a repeatable I/O or throughput win.

**Decision:** **RUNNING.** Prediction structure is clearly real, but the first standalone learned predictor does not generalize well enough to replace temporal/spatial baselines. The current preferred direction is a confidence- and cost-gated hybrid where cheap transition priors provide the base score and the learned ANE scorer contributes residual evidence. No runtime/default enablement until a repeatable correctness-preserving end-to-end win is measured.

**Artifacts:** `.perf_runs/routescout/`, `.perf_runs/routescout-scorer-v1/`, `tools/routescout_*.py`, `logan-ane/examples/route_scout*.rs`, `logan-qwen4/src/route_predictor.rs`.

### Terminal status — 2026-09-22

This entry is closed as **INCONCLUSIVE on the current target**, with the branch
pruned by measurement rather than by preference. EXP-014 through EXP-024 resolved
every remaining gate; the chain is:

1. **Prediction evidence (VERIFIED).** Spatial (same-token, previous-layer) transition
   evidence is the strongest single prior, beating temporal on cross-prompt holdouts
   (EXP-014: spatial R@8 0.4823 vs temporal 0.4046) and is nearly the whole of the
   hybrid's value. Temporal depth and added history depth do not help once spatial
   is fused (EXP-015). Layer 0 must fall back to temporal (EXP-016).
2. **Held-out generalization (MEASURED).** On leave-one-prompt-out evaluation the
   hybrid reaches R@8 0.4721 and cold-arrival recall 0.4787 at budget 8, but per-prompt
   R@8 spans 0.2259-0.6377. The learned scorer carries real residual signal
   (+20% relative cold-arrival on the hardest holdout) but does not dominate the
   cheap priors and its blended edge is under 2 points absolute (EXP-017).
3. **Cost/confidence gating (MEASURED).** ANE execution of a trained scorer is
   numerically viable (top-1 exact, 255/256 top-256 overlap, EXP-022) and packs to
   6.2 us per target layer at 16 targets (EXP-023) — but host feature construction
   costs 111.4 us per layer, **91.7%** of the feature+ANE total (EXP-024).
4. **Real prefetch (REJECTED).** The prefetch path is unreachable on the real MLX
   safetensors model — `cached_expert_issue` requires a `.coli` package and the
   checkpoint has none — so the paired A/B measured predictor overhead only
   (median -0.42%, `mio-prefetch loads=0`; EXP-020).
5. **End-to-end A/B (REJECTED).** No correctness-preserving win exists to measure:
   the routed-expert phase is **dispatch-bound, not I/O-bound** (EXP-018/021). The
   pure storage floor is 51 ms/token against a ~744-940 ms/token steady decode, while
   ~1600 synchronous 4-bit affine GEMM dispatches per token project to 568 ms.
   Prefetching cannot help a workload that is not waiting on storage. An expert
   residency cache was built and measured as the alternative and also lost
   end-to-end on this 16 GiB host (EXP-019).
6. **Adaptation (MEASURED, negative).** Prompt-local adaptation needs ~16 observed
   cycles (~640 layers of routing) to merely approach frozen cross-prompt priors,
   still does not match them on strongly-structured prompts even then, and is worse
   than the frozen prior at every warmup on 2 of 4 prompts (EXP-025). Within a
   decode-length generation it is a net loss.

**What to keep:**
- `logan-qwen4/src/route_predictor.rs` — correctness-neutral online cold-arrival
  predictor, flag-gated off (`QWEN_ROUTE_PREDICT`). Retained because it is correct
  and is the ready artifact for a target that *does* stall on expert transport.
- `tools/routescout_matrix.py` — the cross-prompt harness (matrix/history/local/scorer).
- `tools/routescout_expert_io.py` — the storage floor measurement.
- `logan-ane/examples/route_scout_scorer.rs` (with `--weights`) and
  `logan-ane/examples/route_scout_packing.rs` — the ANE viability and packing gates.
- The `mlx_affine_dispatch_counts` / `mlx_expert_source_timings` instrumentation in
  `logan-qwen4/src/lib.rs`, which is what produced the dispatch-bound attribution.

**What to remove / do not enable:**
- Nothing is enabled by default. `QWEN_ROUTE_PREDICT`, `QWEN_ROUTE_PREDICT_PREFETCH`,
  and `QWEN_ROUTE_SPEC_CACHE` remain opt-in.
- The rejected MLX expert residency cache was removed from source (EXP-019) after
  measuring an end-to-end regression at every capacity; only its instrumentation
  remains.
- The `countsketch`/`svd` whole-route MLP path (`tools/routescout_train.py`) is not
  promoted: it lost to the transition priors on every held-out prompt.

**Best measured configuration:** spatial transition prior (depth-0) fused with
temporal depth-1, plus a 0.25-weight frequency term, is the best predictor
(R@8 0.4721, cold8 recall 0.4787 / precision 0.3096 aggregate). The learned scorer
is a marginal residual at best and was not enabled.

**Next target:** RouteScout's premise requires a real miss penalty. The handoff's
Qwen3.8-Flash-Next path (512 experts, top-10, 48 layers) is larger than RAM and is
the right candidate — but the Qwen3.6 result says the first thing to measure there
is not prediction accuracy, it is whether the expert phase is storage-bound at all.
On this host and this checkpoint it was not, so the actionable optimization targets
are the per-dispatch cost (~1600 synchronous GEMMs/token) and per-token weight
re-preparation, not prefetch scheduling.

---

## EXP-014 — Cross-prompt holdout matrix and prompt-local generalization

**Date:** 2026-09-22  
**Area:** routed MoE / expert prediction  
**Status:** **KEPT (as measurement infrastructure)**

**Hypothesis:** The RouteScout predictor family's ranking of temporal, spatial, and
hybrid transition priors is stable under leave-one-prompt-out evaluation, and any
predictor that only looks good on a same-sequence temporal split will lose that
advantage on true cross-prompt holdouts.

**Baseline:** `tools/routescout_analyze.py` and `tools/routescout_train.py` each
implemented a partial, mutually inconsistent holdout: one reported aggregate
recall only, the other only for `countsketch`/`svd` learned variants, and neither
reported per-prompt variance or cold-arrival precision under an equal budget.

**Candidate:** `tools/routescout_matrix.py` — one NumPy-only harness with a shared
`Priors.fit` + `Metrics` implementation. `matrix` runs leave-one-prompt-out over
all prompt families and reports per-prompt and aggregate route recall and
cold-arrival recall/precision at budgets 4/8/12/16/24. Cold-arrival candidates are
ranked after masking the previous route, so precision is the number a real
prefetcher observes.

**Environment:**
- Hardware: Apple M2, 16 GiB
- Model/artifact: `deepsweet/Qwen3.6-35B-A3B-MLX-oQ4-FP16` (40 layers, 256 experts, top-8)
- Prompt/workload: `routescout-prompt-{rust,moe,science,hash}.tsv`, 19 complete cycles each
- Relevant flags: none (offline trace analysis)

**Correctness gate:** Not applicable — this is offline analysis of already-captured
authoritative routes; no runtime path changes.

**Measurements (leave-one-prompt-out, mean over 4 holdouts):**

| Predictor | R@8 | R@16 | R@24 | cold8 recall | cold8 precision |
|---|---:|---:|---:|---:|---:|
| frequency | 0.2885 | 0.4244 | 0.5059 | 0.2439 | 0.1571 |
| temporal_transition | 0.4046 | 0.5536 | 0.6217 | 0.3967 | 0.2575 |
| spatial_transition | 0.4823 | 0.6340 | 0.6932 | 0.4763 | 0.3071 |
| hybrid_transition | 0.4721 | 0.6271 | 0.6893 | 0.4787 | 0.3096 |

Per-prompt hybrid R@8 spans **0.2259 (rust) to 0.6377 (hash)** — a 2.8x spread. The
earlier single-prompt figures in EXP-013 sat at the top of that range; the
aggregate is materially lower, which is exactly the variance the old tooling hid.

**Result:** Spatial (same-token, previous-layer) transition evidence is the single
strongest prior on every prompt family, and it beats the temporal prior on
generalization even though temporal wins on same-sequence splits. Hybrid is within
noise of spatial on aggregate and slightly better on cold-arrival recall. Prompt
family dominates every model choice: the same predictor is 3x more useful on a
repetitive prompt than on a Rust prompt.

**Decision:** **KEPT** as the measurement harness for all later RouteScout
experiments; no runtime path. **VERIFIED** that predictor comparisons must be
reported per-prompt, never as a single average.

**Artifacts:** `.perf_runs/routescout/EXP-014-matrix/matrix.json`,
`tools/routescout_matrix.py`.

---

## EXP-015 — Temporal history depth beyond the immediately previous route

**Date:** 2026-09-22  
**Area:** routed MoE / expert prediction  
**Status:** **REJECTED**

**Hypothesis:** Conditioning on 2/4/8 previous-token routes at the same layer adds
information beyond the single most recent route, so a depth-weighted temporal
history prior beats the depth-1 temporal prior.

**Baseline:** Temporal transition prior using only the immediately previous token's
same-layer route (depth 1), fused with depth-0 spatial evidence.

**Candidate:** `routescout_matrix.py history` — per-depth layer-local transition
tables, combined with geometric decay (1, 1/2, 1/4, 1/8) over depths.

**Measurements (leave-one-prompt-out):**

| Holdout | depth | temporal_hist R@8 | hybrid_hist R@8 | hybrid cold8 R/P |
|---|---:|---:|---:|---:|
| rust | 1 | 0.1642 | 0.2259 | 0.2118 / 0.1215 |
| rust | 8 | 0.1677 | 0.2082 | 0.1936 / 0.1111 |
| moe | 1 | 0.5533 | 0.6200 | 0.6464 / 0.4205 |
| moe | 8 | 0.5859 | 0.6267 | 0.6576 / 0.4278 |
| science | 1 | 0.3488 | 0.4049 | 0.3805 / 0.2451 |
| science | 8 | 0.3606 | 0.3948 | 0.3700 / 0.2384 |
| hash | 1 | 0.5519 | 0.6377 | 0.6760 / 0.4514 |
| hash | 8 | 0.5958 | 0.6436 | 0.6768 / 0.4519 |

**Result:** Deeper temporal history *does* improve the temporal-only predictor
consistently (rust 0.1642→0.1677, moe 0.5533→0.5859, science 0.3488→0.3606, hash
0.5519→0.5958). But once depth-0 spatial evidence is fused in, extra temporal depth
is neutral-to-harmful on 3 of 4 holdouts (rust 0.2259→0.2082, science 0.4049→0.3948,
hash 0.6377→0.6436 within noise). Cold-arrival recall at budget 8 changes by at
most 0.011 and moves in both directions.

**Decision:** **REJECTED.** Older routes mostly restate what the latest route plus
same-token spatial evidence already imply, while each extra depth adds an
`experts x experts` transition table per layer. Depth 1 is retained. Experimental
depth-2+ tables were never wired into the runtime, so no production code needed
removal.

**Artifacts:** `.perf_runs/routescout/EXP-014-matrix/history.json`.

---

## EXP-016 — Layer-pattern breakdown of predictability

**Date:** 2026-09-22  
**Area:** routed MoE / expert prediction  
**Status:** **KEPT (as measurement infrastructure)**

**Hypothesis:** Spatial predictability is not uniform across layers; some layer
groups are strongly spatially predictable and others are nearly unpredictable, so
a per-layer or per-group prediction policy should beat a uniform one.

**Candidate:** `routescout_matrix.py matrix --per-layer`, breaking every predictor
down by layer index on the same leave-one-prompt-out protocol.

**Measurements (mean hybrid/spatial/temporal R@8 across the 4 holdouts):**

| Layer | spatial R@8 | temporal R@8 | hybrid R@8 |
|---|---:|---:|---:|
| 0 | 0.0399 | 0.5295 | 0.5295 |
| 1 | 0.5747 | 0.5729 | 0.6111 |
| 4 | 0.5833 | 0.5000 | 0.5642 |
| 8 | 0.4028 | 0.3368 | 0.3889 |
| 20 | 0.4080 | 0.3316 | 0.4080 |
| 25 | 0.3976 | 0.3368 | 0.4167 |
| 31 | 0.5069 | 0.3715 | 0.4479 |
| 39 | 0.4809 | 0.3924 | 0.4740 |

**Result:** Layer 0 has no spatial evidence by construction (spatial R@8 = 0.0399)
but is one of the most temporally predictable layers (0.5295), confirming the two
signals are genuinely complementary and that layer 0 must fall back to temporal.
Early layers (1–7) are the strongest spatially (0.47–0.58); mid layers around 8–9
and 20–29 are the weakest (0.40–0.45). No layer is so unpredictable that
prediction should be disabled there.

**Decision:** **KEPT.** The per-layer view justifies keeping a uniform predictor
(with the layer-0 temporal fallback the runtime already has) and quantifies where
residual learned scoring would have the most headroom: the 0.40–0.45 layer band.
No runtime change.

**Artifacts:** `.perf_runs/routescout/EXP-014-matrix/matrix-perlayer.json`.

---

## EXP-017 — Candidate-centric per-expert scorer on true held-out prompts

**Date:** 2026-09-22  
**Area:** routed MoE / learned scorer / native ANE  
**Status:** **QUALIFIED (not enabled)**

**Hypothesis:** A shared per-expert scorer — rather than a whole-route MLP — trained
on true cross-prompt holdouts adds residual predictive value over the
transition/hybrid priors, because it can condition on the same features the priors
use while learning interactions between them.

**Baseline:** `hybrid_transition` prior (depth-0 temporal + spatial + 0.25 frequency),
evaluated on the exact same held-out prompt as the scorer.

**Candidate:** Shared `16 -> 16 -> 8 -> 1` per-expert scorer with the 16-feature
layout the ANE island consumes (previous-route weight, presence, normalized spatial
transition, sqrt(spatial), frequency prior, spatial entropy/margin/top1, temporal
entropy/margin/top1, layer position, layer mod 4, expert id fraction, cold-arrival
interaction, constant). Trained with weighted BCE (cold positives x6, resident
positives x2), 48 negatives per layer/token.

**Measurements (leave-one-prompt-out):**

| Holdout | predictor | R@8 | cold8 recall | cold8 precision |
|---|---|---:|---:|---:|
| rust | hybrid baseline | 0.2259 | 0.2118 | 0.1215 |
| rust | learned scorer | 0.2403 | 0.2545 | 0.1460 |
| rust | blend alpha=0.25 | 0.2342 | 0.2315 | 0.1328 |
| moe | hybrid baseline | 0.6200 | 0.6464 | 0.4205 |
| moe | learned scorer | 0.6038 | 0.6186 | 0.4024 |
| moe | blend alpha=0.25 | 0.6245 | 0.6563 | 0.4269 |
| science | hybrid baseline | 0.4049 | 0.3805 | 0.2451 |
| science | learned scorer | 0.4049 | 0.3923 | 0.2528 |
| science | blend alpha=0.25 | 0.4115 | 0.3886 | 0.2503 |
| hash | hybrid baseline | 0.6377 | 0.6760 | 0.4514 |
| hash | learned scorer | 0.6274 | 0.6417 | 0.4285 |
| hash | blend alpha=0.25 | 0.6488 | 0.6810 | 0.4547 |

**Result:** The scorer is not uniformly better. Standalone it wins clearly on rust
(cold8 recall 0.2118→0.2545, +20% relative) and marginally on science, but loses on
moe and hash. The blend wins route recall on 3 of 4 holdouts and cold-arrival on 3
of 4, with the largest single gain on hash (R@8 0.6377→0.6488, cold8 0.6760→0.6810).
Every blend gain is under 2 percentage points absolute. This is a genuine but small
residual signal, consistent with EXP-013's finding.

**Decision:** **QUALIFIED.** The scorer carries real residual information but does
not dominate the cheap priors, and the blend's edge is below the noise floor of a
single run. It is therefore not enabled, and the decision does not depend on it. The
16-feature layout is shared with `logan-ane/examples/route_scout_scorer.rs` so a
trained scorer can be exported and gated on ANE without re-laying out features.

**Artifacts:** `.perf_runs/routescout/EXP-014-matrix/scorer16.json`,
`tools/routescout_matrix.py scorer`.

---

## EXP-018 — Expert I/O floor vs actual expert-phase cost on the real model

**Date:** 2026-09-22  
**Area:** routed MoE / storage / decode attribution  
**Status:** **KEPT (measurement)**

**Hypothesis:** The Qwen3.6 real-model decode is dominated by storage wait on routed
expert reads, so prefetching experts ahead of demand is the highest-value
optimization.

**Baseline:** `LOGAN_PROFILE=1` span decomposition of a real decode, and an
independent `pread`-only measurement of exactly the expert byte ranges a token needs.

**Candidate/instrumentation:**
- `tools/routescout_expert_io.py` reads the real checkpoint's expert byte ranges with
  `pread`, no decode and no compute, to establish the pure read floor.
- `logan-qwen4` gained dispatch counters for the MLX affine GEMM
  (`mlx_affine_dispatch_counts`) and load/compute attribution inside the real
  expert source (`mlx_expert_source_timings`).

**Environment:**
- Hardware: Apple M2, 16 GiB, internal SSD
- Model/artifact: `deepsweet/Qwen3.6-35B-A3B-MLX-oQ4-FP16`
- Prompt/workload: `QWEN_PROMPT="1 2 3 4 5 6 7 8"`, `QWEN_MAX_NEW=6..24`

**Measurements:**

| Quantity | Value |
|---|---:|
| Expert bytes/token (40 layers x top-8 x 3 matrices) | 480 MiB |
| Physical reads/token | 960 |
| Warm `pread` of one token's experts (real trace routes) | **51 ms** |
| Cold first-touch `pread` of one token's experts | 234 ms |
| **Steady-state decode (measured forwards only)** | **~744–940 ms/token** |
| Model load + prefill (per run, not per token) | ~5.5–6.2 s |
| Decode span `fill_ms/tok` (routed expert phase) | ~900–1370 ms |
| MLX affine GEMM dispatches/token | ~1600 (metal_share = 1.000, 0 fallbacks) |
| Expert load_ms/token (read + decode setup) | ~434–500 ms |
| Expert compute_ms/token (GEMM, incl. Metal buffer resolve/wrap) | ~480–600 ms |
| Single [512x2048] 4-bit affine GEMM dispatch | 355 us |
| Projected expert GEMM floor (355 us x 1600) | 568 ms |

**Result:** The hypothesis is false on this model+path. At steady state the physical
read floor is **51 ms/token**, ~5% of a ~744-940 ms/token decode, while the
routed-expert phase costs ~900-1370 ms. The dominant term is not storage: each
routed-expert GEMM is a *synchronous* Metal dispatch (`commit` +
`waitUntilCompleted` per matrix) costing 355 us, and ~1600 of them per token project
to 568 ms — matching the measured compute figure. Expert work accounts for
essentially all of `fill_ms`.

Two measurement corrections were applied after review and are load-bearing:
- The earlier 37 ms figure used safetensors `data_offsets` directly as file
  offsets. Those are relative to the payload start (`8 + header_length`), so the
  probe was reading header bytes. With the correct `data_start` added the warm
  floor is **51 ms/token**, not 37.
- The earlier "~1600-1900 ms/token decode" divided *total* wall time (which
  contains model load and prompt prefill) by generated tokens. Measured per
  forward, steady-state decode is **~744-940 ms/token**. The storage share is
  therefore ~5%, not ~2%, but the conclusion is unchanged: reads are a small term.
- `load_ms` covers only reads plus MLX-affine setup; the Metal buffer
  `resolve`/`wrap` happens inside the separately-timed `matmul`, so it is not
  attributable to the load term.

**Decision:** **KEPT** as the governing measurement for this phase. It also
directly refutes the premise of the prefetch program *on this path*: there is no
SSD stall to hide. Prefetching can only help where the miss penalty is real
(an external/slower device, a larger model, or a constrained residency budget), so
the remaining RouteScout work must either (a) target a workload where expert
transport genuinely stalls, or (b) attack the per-dispatch overhead, which is what
the measurements actually point at.

**Artifacts:** `.perf_runs/routescout/EXP-014-matrix/expert_io.json`,
`tools/routescout_expert_io.py`, `logan-qwen4/examples/metal_probe.rs`.

---

## EXP-019 — Expert residency cache on the MLX safetensors path

**Date:** 2026-09-22  
**Area:** routed MoE / residency / MLX safetensors  
**Status:** **REJECTED**

**Hypothesis:** The real-model MLX safetensors expert path re-reads and re-wraps all
600 experts per token; giving it the same LRU residency the `.coli` path already has
will cut the expert load cost and decode time proportionally.

**Baseline:** Stock `MlxLocalExpertSource`, which loads every routed expert from the
shards on every token with no cache of any kind.

**Candidate:** `QWEN_MLX_EXPERT_CACHE_PER_LAYER=N` — a layer-partitioned
`ExpertStore<CachedMlxExpert>` holding the expert's three `Wt` matrices behind an
`Arc`, so a hit reuses the already-created Metal tensor instead of rebuilding it.
Default 0 (off).

**Environment:** as EXP-018. Swept N in {0, 8, 16, 32}.

**Correctness gate:** Token identity on `QWEN_PROMPT="1 2 3 4 5 6 7 8"` — generated
IDs identical in every configuration (`[348, 10, 4838, 1665, 15, 16, 17, ...]`).
The cache changes residency only; bytes and numerics are unchanged. **PASS.**

**Measurements** (figures are per-forward decode after the timing fix in EXP-018;
the first-sweep totals included load/prefill and are not comparable):

| per_layer cap | hit rate | load_ms/tok | compute_ms/tok | total ms/tok |
|---:|---:|---:|---:|---:|
| 0 (off) | 0.000 | 482.0 | 598.6 | 1585.8 |
| 8 | 0.487 | 463.9 | 603.4 | 1604.8 |
| 16 | 0.685 | 444.6 | 605.2 | 1610.4 |
| 32 | 0.709 | 426.7 | 642.4 | 1642.3 |

**Result:** The cache works exactly as designed on its own term — hit rate rises to
0.71 and the load term falls 482→427 ms/token. But the reported end-to-end number
gets *worse* in every cached configuration, and worse the larger the cache
(1586 → 1642 ms/token). Retaining ~30 GiB of decoded expert storage on a 16 GiB
machine raises UMA pressure, the same mechanism already recorded in
`make_expert_store` for the 10/layer `.coli` configuration. The hit path itself is
cheap (0.5 us), so the regression is memory-system pressure, not cache bookkeeping.

**Caveat on the end-to-end column:** these totals predate the timing correction and
include model load plus prefill as a large constant, which compresses any real
per-token difference. The *load-term* result (482→427 ms/token, monotone in hit
rate) is solid because it is measured inside the expert source and excludes load and
prefill. The end-to-end regression is directionally consistent across all three
capacities and does not vanish, but the magnitudes should be read as
load-diluted rather than as steady-state decode deltas.

**Decision:** **REJECTED.** Not enabled; the residency knob added for the experiment
was removed from source after measurement. The instrumentation (counters +
attribution) is retained because it is what made the EXP-018 attribution possible
and is correctness-neutral. This is a **MEASURED** null result on a 16 GiB machine;
it does not establish that residency is useless on a larger-memory host, but it does
establish that this host cannot pay for it.

**Artifacts:** `.perf_runs/routescout/EXP-018-prefetch-ab/`,
`logan-qwen4/src/lib.rs` (`MlxLocalExpertSource`, `mlx_expert_source_timings`).

---

## EXP-020 — Real-model paired A/B of the online predictor + speculative prefetch

**Date:** 2026-09-22  
**Area:** routed MoE / prefetch / end-to-end  
**Status:** **REJECTED**

**Hypothesis:** Enabling the EXP-013 online cold-arrival predictor with speculative
prefetch on the real Qwen3.6 decode reduces ms/token versus stock decode.

**Baseline (arm B):** stock decode, no RouteScout flags.

**Candidate (arm C):** `QWEN_ROUTE_PREDICT=1 QWEN_ROUTE_PREDICT_PREFETCH=1
QWEN_ROUTE_SPEC_CACHE=64 QWEN_ROUTE_PREDICT_BUDGET=8`.

**Environment:**
- Hardware: Apple M2, 16 GiB
- Model/artifact: `deepsweet/Qwen3.6-35B-A3B-MLX-oQ4-FP16`
- Prompt/workload: `QWEN_PROMPT="1 2 3 4 5 6 7 8"`, `QWEN_MAX_NEW=6`
- Ordering: alternating B/C/C/B, 2 pairs (pilot)
- Driver: `tools/routescout_ab.sh`

**Correctness gate:** Generated token IDs byte-identical across every B and C run
(`[348, 10, 4838, 1665, 15, 16]`). **PASS.**

**Measurements:**

| arm | runs (total ms/token) | median |
|---|---|---:|
| B (baseline) | 2443.9, 2487.7, 2434.6, 2855.9 | 2465.8 |
| C (predictor+prefetch) | 2434.9, 2458.9, 2467.2, 2451.8 | 2455.4 |

Median delta: **−0.42%** (noise; one B outlier at 2855.9 dominates the mean).
`logan mio-prefetch: loads=0 used=0` on every candidate run.

**Result:** No improvement. More importantly the cause is structural rather than
statistical: **the prefetch path is unreachable on this model.** `cached_expert_issue`
returns early via `self.coli.as_ref()?`, and the real MLX safetensors checkpoint has
no `.coli` package, so it streams experts through `MlxLocalExpertSource` instead.
RouteScout predictions were computed but never issued a single byte of speculative
I/O.

**Two defects in this run, both corrected in EXP-026/027:** the metric was the
binary's `total` (which includes ~9 s of model load and prefill, diluting any real
delta toward zero), and `QWEN_MAX_NEW=6` is below the predictor's 16-pair confidence
warmup, so the gate may have blocked speculation before the structural blocker was
even reached. EXP-027 re-ran with both fixed and still observed zero loads, which is
what makes the structural conclusion sound.

**Decision:** **REJECTED, superseded by EXP-027.** Retained as the first observation
of the inert path.

**Artifacts:** `.perf_runs/routescout/EXP-018-prefetch-ab/runs.tsv` and per-run logs,
`tools/routescout_ab.sh`.

---

## EXP-021 — Expert-phase serialization audit (conclusion)

**Date:** 2026-09-22  
**Area:** routed MoE / Metal dispatch  
**Status:** **KEPT (diagnosis)**

**Hypothesis:** The dominant real-model decode cost is per-dispatch serialization
rather than storage, so the highest-value optimization is batching/async overlap of
the routed-expert GEMMs.

**Evidence assembled from EXP-018/019:** expert phase ~900–1370 ms/token of a
~744–940 ms/token steady decode (**the routed-expert phase dominates**); of that,
~600 ms is GEMM dispatch cost at 355 us per synchronous `[512x2048]` 4-bit affine
call, and ~430–500 ms is expert read plus MLX-affine setup. The pure storage read is
51 ms/token.

**Result:** The routed-expert phase is **dispatch-bound, not I/O-bound**. Each
expert matrix costs one `commit` + `waitUntilCompleted` round trip (measured 355 us
on a 512 KiB matrix — far above the ~30 us this GPU needs for the arithmetic), and
the engine issues 1800 of them per token. Every routed expert is also re-prepared
from the shard on every token. The two structural costs are therefore (1) dispatch
count and (2) redundant per-token weight preparation — not storage bandwidth.

**Decision:** **KEPT** as the diagnosis that closes the prefetch branch on this
target and redirects work. RouteScout's predictive value is real but currently
unexercisable here: it targets a stall that this path does not have. Prefetch/batching
work should target a configuration with a genuine miss penalty — external storage, a
larger checkpoint, or a residency budget small enough that cold experts stall — and
the per-dispatch and per-token preparation costs identified here are the actionable
optimization targets on the current target. No runtime default was changed.

**Artifacts:** `.perf_runs/routescout/EXP-014-matrix/expert_io.json`,
`logan-qwen4/examples/metal_probe.rs`, `logan-qwen4/src/lib.rs` instrumentation.

---

## EXP-022 — Trained candidate-scorer ANE gate (ordering, not just error)

**Date:** 2026-09-22  
**Area:** routed MoE / native ANE / learned scorer  
**Status:** **QUALIFIED**

**Hypothesis:** A real trained scorer exported from `routescout_matrix.py` can run
through the existing direct-ANE island and reproduce the CPU reference's candidate
*ordering*, which is what a residual scorer is actually consumed for.

**Baseline:** The existing gate loaded identity weights and checked only absolute
error against a trivial oracle, which proves the graph compiles and computes but
says nothing about whether real (large, negative, mixed-scale) weights survive fp16.

**Candidate:** `logan-ane/examples/route_scout_scorer.rs` gained a `--weights` mode.
It loads a real `16 -> 16 -> 8 -> 1` fp16 export, runs it on ANE, and gates on
top-1 ranking agreement and top-256 overlap against a per-lane CPU reference. The
1×1 convolution means lane `s` depends only on input lane `s`, so the reference is
exact rather than approximate.

**Environment:** Apple M2 (h14g/h14, 1 ANE, 16 cores); 8 targets x 256 experts =
2048 spatial lanes; scorer trained on all four prompt families (161,280 rows).

**Correctness gate:** `top1_match` AND top-256 overlap >= 99% AND all outputs
finite. **PASS.**

**Measurements:**

| mode | compile+load | evaluate | max_abs | top-256 overlap | top-1 match |
|---|---:|---:|---:|---:|---|
| identity (oracle) | 144.1 ms | 97.4 us | 0.001505 | 256/256 | true |
| trained fp16 | 151.8 ms | 135.7 us | 0.004575 | 255/256 | true |

**Result:** The trained scorer runs on ANE with a top-1 winner identical to the CPU
reference and 255/256 of the same top-256 candidates. Against a reference scale of
7.489, fp16 rounding produces a max absolute error of 0.0046 — well inside what
preserves ordering. The identity path still passes its original absolute-error
oracle, so the gate did not weaken; it gained a second, stronger mode.

**Decision:** **QUALIFIED.** The ANE execution path is proven on real weights. The
scorer itself remains not-enabled because EXP-017 showed its residual value is
below the single-run noise floor; this experiment establishes that *if* it is later
enabled, ANE execution is numerically viable.

**Artifacts:** `.perf_runs/routescout/EXP-022-ane-scorer/scorer.fp16.bin`,
`logan-ane/examples/route_scout_scorer.rs`,
`tools/routescout_matrix.py scorer --export`.

---

## EXP-023 — ANE spatial-packing sweep

**Date:** 2026-09-22  
**Area:** routed MoE / native ANE  
**Status:** **KEPT**

**Hypothesis:** Amortizing the fixed ANE dispatch cost over more target layers per
dispatch improves per-layer scoring cost, up to a spatial width where the ANE's own
throughput saturates.

**Baseline:** The single established point: 8 target layers x 256 experts = 2048
spatial lanes at ~80-120 us/dispatch.

**Candidate:** `logan-ane/examples/route_scout_packing.rs` compiles a distinct
non-identity weight set per width and sweeps targets {1, 2, 4, 8, 16}, verifying
each width against a per-lane CPU reference for that width's own geometry.

**Environment:** Apple M2, h14g/h14, 1 ANE, 16 cores.

**Correctness gate:** max absolute error <= 0.02 and all outputs finite, per width.
**PASS at every width.**

**Measurements** (steady-state, 3 repetitions; the first sweep's numbers included
ANE cold-start and read ~90-150 us/dispatch with compile ~90-150 ms):

| targets | spatial | compile+load ms | us/dispatch | us per target layer | max_abs |
|---:|---:|---:|---:|---:|---:|
| 1 | 256 | ~7-14 | ~92-94 | ~93.0 | 0.0001 |
| 2 | 512 | ~7 | ~90-93 | ~45.8 | 0.0001 |
| 4 | 1024 | ~7 | ~89-92 | ~22.5 | 0.0001 |
| 8 | 2048 | ~7 | ~97-101 | ~12.4 | 0.0001 |
| 16 | 4096 | ~7 | ~97-102 | ~6.2 | 0.0001 |

**Result:** Per-target-layer cost falls monotonically from ~93.0 us (1 target) to
**~6.2 us (16 targets)** — a ~15x amortization. Dispatch cost is essentially
constant (~90-102 us) across widths, so the entire win is amortization of a fixed
per-dispatch cost, and 16 targets is still improving per-layer. All widths verified
numerically exact against their own geometry (max_abs 0.0001).

**Decision:** **KEPT.** The runtime should pack 16 target layers per dispatch rather
than the 8 EXP-013 assumed; budget ~6.2 us per target layer. Note the per-dispatch
cost is dominated by fixed overhead, so the useful design rule is "as many targets
per dispatch as the layer set allows", not a specific width.

**Artifacts:** `logan-ane/examples/route_scout_packing.rs`.

---

## EXP-024 — Host feature-construction cost

**Date:** 2026-09-22  
**Area:** routed MoE / learned scorer / host overhead  
**Status:** **KEPT (measurement, and a gate on the scorer's viability)**

**Hypothesis:** ANE scoring latency is the binding constraint on running a learned
scorer in the decode loop.

**Baseline:** ANE-only figures from EXP-023 (6.2 us per target layer at 16 targets).

**Candidate:** `tools/routescout_feature_cost.py` times the host-side build of the
16-feature candidate matrix at the real geometry (40 layers, 256 experts), which is
what must happen before any dispatch can be issued.

**Environment:** Apple M2; four prompt traces; 2880 measured (layer, token) cases.

**Measurements:**

| Quantity | Value |
|---|---:|
| Feature build per layer | 111.4 us |
| Feature build per token (40 layers) | 4.455 ms |
| ANE scoring per layer (16-target packing) | 6.2 us |
| Feature share of feature+ANE | **91.7%** |
| Transition-table fit for 4 prompts | 79.0 ms |

**Result:** Host feature construction is **~18x** more expensive than the ANE scoring
it feeds at the measured packing. A 6.2 us ANE model is irrelevant when assembling
its input costs 111.4 us per layer and 4.5 ms per token. This confirms the concern
the handoff raised: ANE microbenchmarks must never be read as an end-to-end claim.
The runtime already maintains the same transition statistics natively in Rust
(`route_predictor.rs`), so the Python-harness figure is an upper bound on the vector
version, but the ordering — feature build dominating model evaluation — is the
structural fact.

**Decision:** **KEPT** as the gate on this branch. Any future learned-scorer
integration must first reduce feature construction, not model latency; a scorer that
costs 4.5 ms/token against a ~1600 ms/token decode is affordable but not free, and
the cost is almost entirely host-side. No runtime change.

**Artifacts:** `.perf_runs/routescout/EXP-024-feature-cost/cost.json`,
`tools/routescout_feature_cost.py`.

---

## EXP-025 — Prompt-local online adaptation versus frozen cross-prompt priors

**Date:** 2026-09-22  
**Area:** routed MoE / expert prediction / online learning  
**Status:** **KEPT (measurement, decisive)**

**Hypothesis:** A predictor that adapts to the prompt it is currently decoding will
beat priors frozen from other prompts, and the advantage grows with how much of the
prompt has been observed.

**Baseline:** Priors frozen from the other three prompt families
(leave-one-prompt-out), i.e. the cross-prompt generalization case.

**Candidate:** Priors refit from only the current prompt's own observed cycles after
a warmup of W cycles; before W, predictions fall back to the global priors. Swept
W in {2, 4, 8, 12, 16} out of 19 available cycles.

**Measurements** (format `local / global`; higher is better):

| warmup | rust | moe | science | hash |
|---:|---:|---:|---:|---:|
| 2 (R@8) | 0.4012 / 0.2259 | 0.4144 / 0.6200 | 0.3691 / 0.4049 | 0.4142 / 0.6377 |
| 4 (R@8) | 0.3870 / 0.2259 | 0.4477 / 0.6200 | 0.3995 / 0.4049 | 0.4481 / 0.6377 |
| 8 (R@8) | 0.3328 / 0.2259 | 0.5087 / 0.6200 | 0.4495 / 0.4049 | 0.5174 / 0.6377 |
| 12 (R@8) | 0.2887 / 0.2259 | 0.5273 / 0.6200 | 0.4573 / 0.4049 | 0.5500 / 0.6377 |
| 16 (R@8) | 0.2479 / 0.2259 | 0.5781 / 0.6200 | 0.4245 / 0.4049 | 0.5972 / 0.6377 |
| 16 (cold8 recall) | 0.2236 / 0.2118 | 0.6021 / 0.6464 | 0.3891 / 0.3805 | 0.6331 / 0.6760 |

**Result:** The effect is prompt-dependent and the earlier "monotone" summary does
not hold. On the strongly-structured prompts local adaptation rises steadily toward
the frozen prior but never reaches it by the end of the trace: moe 0.4144→0.5781
against a frozen 0.6200, hash 0.4142→0.5972 against 0.6377. On science it rises to
0.4573 at W=12 then dips to 0.4245, oscillating around the frozen 0.4049. On rust it
*falls* monotonically (0.4012→0.2479) yet stays above the frozen 0.2259 at every
warmup — the frozen prior is simply weak there, so even a poorly-estimated local
model beats it.

Critically, adaptation requires roughly **16 observed cycles (~640 layers of
routing)** merely to approach a frozen prior that needed zero in-context cycles —
and it still does not match it on the prompts with strong structure, where it also
ends up *below* both the frozen prior and its own early-prompt estimate. The prompts
tested here are 19 cycles long, so the curve is still moving at the end of the trace.

**Decision:** **KEPT.** Prompt-local adaptation cannot be relied on for a
decode-length workload. On two of four prompts the local model is worse than the
frozen prior at every measured warmup, and on the other two it only wins where the
frozen prior happened to transfer badly. A 16-cycle warmup on a 19-cycle prompt
spends most of its budget worse than simply using frozen spatial priors. The
runtime's existing design (online transition statistics with decay, never reset) is
the right policy for this evidence: it degrades gracefully toward the frozen prior
rather than committing to a cold local estimate. No runtime change was required.

**Artifacts:** `.perf_runs/routescout/EXP-025-local-adaptation/local-w{2,4,8,12,16}.json`,
`tools/routescout_matrix.py local`.

---

## EXP-026 — Measurement corrections found in review

**Date:** 2026-09-22  
**Area:** methodology  
**Status:** **KEPT (corrects EXP-018/019/020/024)**

**Hypothesis:** Several numbers in EXP-018 through EXP-024 were produced by
instrumentation with defects that would change the reported magnitudes.

**Findings, each verified against source:**

1. **`routescout_expert_io.py` used the wrong file offsets.** safetensors
   `data_offsets` are relative to the payload start (`8 + header_length`). The
   runtime's `parse_shard` adds that start; the probe did not, so it was reading
   header bytes instead of expert weights. Corrected and re-run: the warm
   one-token floor is **51 ms**, not 37 ms; cold first-touch is 234 ms, not 318 ms.
2. **Decode timing conflated load, prefill, and decode.** `token_ms` started after
   the prompt-last forward and stopped after the final sampling step, so one sample
   had no forward and entries were shifted. `total` also included model load and
   prefill, so dividing it by token count inflated "ms/token" by ~2x. Corrected to
   time each `forward_token` and report `decode`, `prefill`, and `total` separately:
   true steady-state decode is **~744-940 ms/token**, not ~1600-1900.
3. **`Metrics.observe` double-counted with `per_layer=False`.** `ensure(predictor)`
   and `ensure(predictor, layer, False)` return the SAME bucket, so every hit,
   issued, and arrival count was doubled. Recall and precision are ratios and were
   unaffected — which is why every published EXP-014/015/016/017/025 figure stands —
   but absolute `useful`/`wasted`/`arrivals` counts would have been 2x wrong. Fixed;
   rate outputs are unchanged, verified by re-running the matrix.
4. **`local` refit per layer instead of per token** and passed a malformed corpus;
   both fixed (it also now predicts each layer from one hoisted fit).
5. **`routescout_ab.sh` used `TOKENS=8` and `total` as its metric.** With the
   predictor's default confidence gate (`pairs >= 16` per layer) 8 forwards never
   engage anything, and `total` dilutes any real delta with a ~9 s constant. Now
   `TOKENS=24` and the metric is the binary's steady-state decode figure.
6. **`metal_probe` initially read `metal_available()` before any `metal_init()`.**
   `AVAILABLE` starts false, so the probe reported `false` on a healthy Metal host.
   Already fixed; the probe now inits first and reports `PASS` with 355 us dispatch.

**Result:** Items 1, 2, 5, and 6 changed reported magnitudes; item 3 changed absolute
counts but not any rate; item 4 fixed a crash. EXP-018, 019, and 020 were corrected
in place, and EXP-018 now states which figures moved and why.

**Decision:** **KEPT.** Recorded because the handoff requires failed and inconclusive
runs to be logged, and because a silently-wrong measurement is worse than a missing
one. The conclusions of EXP-014 through EXP-025 are unchanged: every rate metric was
ratio-based and re-verified, and the storage-vs-dispatch conclusion holds with a
larger storage share (~5% rather than ~2%).

**Artifacts:** `.perf_runs/routescout/EXP-026-io-floor/expert_io.json`,
`tools/routescout_expert_io.py`, `tools/routescout_matrix.py`,
`tools/routescout_ab.sh`, `logan-qwen4/src/main.rs`.

---

## EXP-027 — Corrected paired A/B with the predictor actually engaged

**Date:** 2026-09-22  
**Area:** routed MoE / prefetch / end-to-end  
**Status:** **REJECTED**

**Hypothesis:** The EXP-020 A/B may have been invalid because the predictor never
engaged; with the confidence warmup satisfied, speculative prefetch still produces no
improvement.

**Baseline (B):** stock decode. **Candidate (C):** `QWEN_ROUTE_PREDICT=1`,
`QWEN_ROUTE_PREDICT_PREFETCH=1`, `QWEN_ROUTE_SPEC_CACHE=64`, budget 8, confidence
gate at its default.

**Environment:** Apple M2 16 GiB; `QWEN_MAX_NEW=24` (24 forwards, so every layer
exceeds the 16-pair confidence warmup); alternating B/C/C/B, 2 pairs; metric is the
binary's per-forward decode figure.

**Correctness gate:** Generated IDs identical in all 8 runs. **PASS.**

**Measurements:**

| arm | decode ms/token | median |
|---|---|---:|
| B | 840.3, 754.0, 743.4, 1337.1 | 797.1 (stdev 282.3) |
| C | 1053.5, 700.7, 915.3, 829.4 | 872.3 (stdev 148.3) |

**Result:** The predictor is now demonstrably engaged — `pairs=1200` (30 per layer)
across every candidate run, well above the 16-pair threshold, producing
`predicted=8853` arrival predictions. Yet **`prefetch_loads=0` in all eight runs**,
including the baseline. Speculation issued no I/O whatsoever, so the arms differ only
by prediction overhead. The observed +9.4% median difference is not meaningful: both
distributions are wide (stdev 282 ms and 148 ms) with 4 samples per arm, and the
baseline contains a 1337 ms outlier. This is noise, and the correct reading is that
the arms are indistinguishable because the candidate path is inert.

**Decision:** **REJECTED.** The conclusion is the same as EXP-020 but for a stronger
reason: this is a valid engagement audit that rules out the "gate blocked it"
explanation. `MlxLocalExpertSource::eval` is the real MLX expert path and
`cached_expert_issue` returns at `self.coli.as_ref()?` before reaching any MetalIO
prefetch, so a safetensors model has no prefetch seam at all. Measuring this further
requires a real `.coli` package or a new safetensors prefetch path — not flag
toggling. Combined with EXP-018 (51 ms/token storage floor), the branch stays closed.

**Artifacts:** `.perf_runs/routescout/EXP-027-ab-valid/runs.tsv` and per-run logs,
`tools/routescout_ab.sh`.

---


---

## EXP-028 — Raw MLX SSD-only expert streaming + MetalIO RouteScout seam

**Date:** 2026-09-22  
**Area:** routed MoE / storage / RouteScout / source architecture  
**Status:** **SUPERSEDED (mechanism KEPT, performance REJECTED)**

**Closure (2026-09-22, this slice):** The Mechanism half of this entry stands and
is the foundation the rest of the slice was built on: raw MLX/safetensors now has a
real, correctness-preserving uncached MetalIO expert path with a source-neutral
prefetch seam. The **performance** half is rejected by EXP-031, which measured the
prefetch this entry enabled as a **paired 3:1 loss at +9.67% decode** with zero
late prefetches — i.e. the transport works and does not pay. `LOGAN_EXPERT_NOCACHE`
remains opt-in and is not a default. See EXP-031 (prefetch policy), EXP-032
(I/O scheduling) and EXP-029 (the measurement correction that made both numbers
interpretable).

**Hypothesis:** Qwen3.6 becomes a valid RouteScout systems target when dense/static weights remain resident but routed experts bypass macOS file caching and are fetched through MetalIO. Extending the engine-neutral `ExpertSource` seam with optional prefetch should let RouteScout issue speculative reads for raw MLX/safetensors without depending on the legacy COLI package path.

**Baseline:** Current raw MLX/safetensors runtime. Dense/static tensors are resident, selected routed experts are range-read on demand, but normal POSIX reads benefit from the macOS page cache and RouteScout cannot issue speculative I/O through `MlxLocalExpertSource`.

**Candidate:** Add an explicit uncached expert mode (`QWEN_MLX_EXPERT_NOCACHE=1`) using expert-only file handles with macOS `F_NOCACHE`; use MetalIO for selected expert ranges when available; add optional source-level prefetch so RouteScout can issue raw-MLX speculative loads. Keep ordinary safetensors/MLX loading supported as a first-class source. Treat COLI as a legacy package source while the already-designed `.logan` format is implemented rather than renaming COLI bytes.

**Correctness gate:** Same generated token IDs as the ordinary raw-MLX path for an identical prompt/sampler. Unit tests must cover source-prefetch fallback semantics and safetensors range planning. MetalIO failure must fall back to uncached POSIX reads without changing arithmetic.

**Performance gate:** Compare steady-state decode for ordinary raw MLX, uncached raw MLX without RouteScout, and uncached+MetalIO+RouteScout. Record MetalIO prefetch issued/used/wasted/ready/late. No new path becomes a normal default without a repeatable correctness-preserving win.

**Decision:** Pending implementation and measurement.


### EXP-028 qualification results — 2026-09-22

Implementation landed behind opt-in environment flags and the source-neutral `ExpertSource::prefetch` contract.

**Correctness / unit gates:**
- `cargo check -p logan-qwen4 -p logan-metal`: PASS.
- `cargo test -p logan-metal --lib`: 4 passed, 0 failed.
- `cargo test -p logan-qwen4 --lib`: 92 passed, 0 failed, 3 ignored.
- Ordinary raw-MLX six-token output: `[348, 10, 4838, 1665, 15, 16]`.
- `QWEN_MLX_EXPERT_NOCACHE=1` + MetalIO produced the identical six tokens.
- 24-token RouteScout qualification output exactly matched the prior EXP-027 baseline:
  `[348, 10, 4838, 1665, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 33898, 2110, 30, 31, 73307, 58, 3312, 87197, 62]`.

**Observed I/O:**
- SSD-only six-token run reported `nocache=true metalio=true`, 4,160 MetalIO loads,
  7,361,003,520 bytes, 4,160 waits, and zero failures.
- Its measured decode-forward timing was 852.9 ms/token versus 956.6 ms/token for the
  immediately preceding ordinary raw-MLX run. This is one run per arm and is **not**
  promoted as a speedup claim.
- A 24-token SSD-only RouteScout run with the normal confidence gate issued zero
  speculative reads because the online predictor's measured precision was only
  ~0.164, below the default 0.50 gate. Decode was 833.3 ms/token.
- Qualification with `QWEN_ROUTE_PREDICT_CONFIDENCE_GATE=0` and budget 8 issued
  **8,853 speculative MetalIO loads**. Of these, **1,448 were used**, **7,374 were
  wasted**, **1,448 were ready at demand**, and **0 were late at demand**. The
  corresponding route-arrival precision/recall were 0.164/0.413. Decode was
  874.9 ms/token in this single run.

**Result:** **KEPT (mechanism), predictor policy remains gated.** Raw MLX/safetensors
now has a real MetalIO speculative-I/O seam; EXP-027's structural `self.coli`
blocker is removed. F_NOCACHE applies only to dedicated expert-streaming descriptors,
so dense/static weights retain normal caching. The default 50% confidence gate remains
appropriate: forcing low-precision budget-8 speculation produced substantial wasted
I/O and no observed speedup. Future performance work should improve candidate ranking,
budget selection, and/or farther-ahead batching rather than merely issuing more reads.

**Format direction:** The owner accepted ADR 0001 in this work: `.logan` is the
canonical compiled-format direction, COLI is deprecated legacy compatibility, and raw
safetensors/MLX remains a first-class runtime source. The runtime source abstraction
must remain format-neutral; no new RouteScout/MetalIO code depends on COLI records.

**Flag naming follow-up:** the engine-wide spellings are `LOGAN_EXPERT_NOCACHE=1`,
`LOGAN_EXPERT_METALIO=1`, and `LOGAN_EXPERT_PREFETCH_SLOTS=N`. The original
`QWEN_MLX_*` spellings remain compatibility aliases for existing RouteScout
scripts.

**Canonical flag smoke gate:** after the final release rebuild,
`LOGAN_EXPERT_NOCACHE=1` on the real Qwen3.6 checkpoint reported
`nocache=true metalio=true`, produced `[348, 10]` exactly, issued 2,880 MetalIO
loads (5,096,079,360 bytes) with zero failures, and completed successfully.

---

## EXP-029 — Decode-boundary normalization for process-cumulative counters

**Date:** 2026-09-22  
**Area:** methodology / routed MoE / MetalIO profiling  
**Status:** **KEPT**

**Hypothesis:** Every counter `profile_summary` normalizes by the measured decode
forward count currently accumulates across **all** model forwards — prefill,
the final prompt forward, and decode — so per-token attribution numbers are
inflated by the prefill share and are not decodable as decode costs. Snapshotting
the counters at the decode boundary and reporting deltas will change the reported
magnitudes to be structurally consistent with the decode window, while leaving
token identity and the qualitative attribution conclusions unchanged.

**Baseline:** `profile_summary(tokens, total_ms)` where `tokens` is
`forward_ms.len()` (decode forwards only) but `self.spans`, `mlx_affine_dispatch_counts()`,
`mlx_expert_source_timings()`, `logan_metal::metal_profile()`, and
`logan_metal::mio_stats()` all read process-lifetime values.

**Reproduction (pre-fix), `QWEN_PROMPT="1 2 3 4 5 6 7 8"`, `QWEN_MAX_NEW=6`,
`LOGAN_PROFILE=1 LOGAN_EXPERT_NOCACHE=1`:**

- prompt length 8, `max_new` 6 → 7 prefill forwards + 1 final-prompt forward +
  5 measured decode forwards = **13 model forwards**
- reported: `logan mlx-expert: calls=4160 calls_per_token=832.0`
- structural expectation for the decode window: `5 * 40 * 8 = 1600` calls,
  i.e. **320 calls/forward**
- `4160 / 13 = 320` exactly, confirming the numerator is lifetime and the
  denominator is decode-only

**Candidate:** `Model::begin_decode_measurement()` takes one snapshot of every
cumulative counter; `profile_summary` subtracts it component-wise and reports
decode-only values, emitting `logan profile-window: decode forwards=N`. Without
a snapshot it reports lifetime totals and marks the window `lifetime`, so no
consumer silently receives a mislabeled number. Wired at every decode boundary:
`main.rs` (after the final prompt forward), `run_greedy_with`,
`generate_from_logits` / `generate_from_logits_mtp_block` (COLI paths), the
scheduled worker (first authoritative `OP_DECODE`), and the `phase_bench` /
`gdn_ane_e2e` probes. The snapshot is self-gating on `LOGAN_PROFILE`, so decode
pays nothing when profiling is off.

**Correctness gate:** Generated token IDs identical to the pre-fix binary under
identical flags; profiling-off runs must be behaviour-identical.

**Acceptance:** for a 23-forward decode, `mlx-expert calls` must equal
`23 * 40 * 8 = 7360`.

**Measurements:** pre-fix vs post-fix, same model/prompt/flags, `QWEN_MAX_NEW=6`
(13 total forwards, 5 measured) and `QWEN_MAX_NEW=24` (31 total, 23 measured):

| Quantity | pre-fix | post-fix | structural |
|---|---:|---:|---:|
| `max_new=6` expert calls | 4160 | **1600** | `5*40*8 = 1600` |
| `max_new=6` calls/token | 832.0 | **320.0** | 320 |
| `max_new=24` expert calls | — | **7360** | `23*40*8 = 7360` |
| `max_new=24` calls/token | — | **320.0** | 320 |
| `max_new=24` load_ms/token | — | **148.2** | — |
| `max_new=24` compute_ms/token | — | **334.6** | — |
| `max_new=24` `load_share` | — | **0.307** | — |

**Self-refutation of the pre-fix attribution:** the `max_new=6` pre-fix run
reported `load_ms_per_token=659.0` and `compute_ms_per_token=805.0`, i.e.
**1464 ms of expert work per token**, while the same run's own measured decode was
**630.2 ms/token**. The two expert terms exceeded the entire forward by 2.3x, so
the pre-fix numbers were not merely mis-scaled — they were arithmetically
inconsistent with the decode they claimed to describe. Post-fix the two terms sum
to 482.8 ms against a measured 731.3 ms decode, which is consistent.

**Affected prior conclusions.** This re-opens part of EXP-018/021. Those entries
concluded the expert phase is dispatch-bound with load ≈ 434–500 ms/token and
compute ≈ 480–600 ms/token. On the corrected decode window the load term is
**148.2 ms/token** and the compute term is **334.6 ms/token**; the prefill share
had been charged to decode. The qualitative conclusion survives (compute exceeds
load, so the phase is not primarily storage-bound), but the *magnitudes* in
EXP-018/021 are decode-inflated and must not be reused. EXP-018's 51 ms/token
`pread` floor is unaffected: it was measured directly with `pread`, outside this
counter path. In particular the claim "~1600 affine dispatches per token" is
corrected to **1191 per forward** (27393 dispatches / 23 forwards), consistent
across both pre- and post-fix runs at 1190.5/forward.

**Correctness:** generated 24-token IDs post-fix are byte-identical to EXP-028's
qualified sequence and to the pre-fix binary under identical flags:
`[348, 10, 4838, 1665, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 33898, 2110,
30, 31, 73307, 58, 3312, 87197, 62]`. **PASS.**

**Decision:** **KEPT.** This is the required measurement-integrity fix and all
subsequent performance work in this slice depends on it. The counters now report
a decode-only window, and any caller that never calls
`begin_decode_measurement()` gets lifetime totals explicitly marked `lifetime`
rather than a silently mislabeled figure.

**Artifacts:** `.perf_runs/routescout/EXP-029-decode-deltas/{before-fix,after-fix}.log`,
`logan-core/src/telemetry.rs` (`TokenSpans::delta_from`),
`logan-qwen4/src/lib.rs` (`DecodeBaseline`, `begin_decode_measurement`,
`profile_summary`, `subtract_mio`), boundary wiring in `main.rs`,
`plan/prefix_runtime.rs`, `scheduled.rs`, `bin/gdn_ane_e2e.rs`,
`logan-chat/examples/phase_bench.rs`.

---

## EXP-030 — Precomputed per-expert MLX I/O plans

**Date:** 2026-09-22  
**Area:** routed MoE / MetalIO / host overhead  
**Status:** **REJECTED**

**Hypothesis:** The raw-MLX source derives each expert's tensor names, shard ids
and file ranges at runtime on every demand (`format!` per matrix plus tensor-map
lookups). Caching one immutable plan per `(layer, expert)` removes that work from
the 320-evaluations-per-forward critical path and shortens the load term.

**Baseline:** `MlxLocalExpertSource::io_plan` rebuilding the plan per call.

**Candidate:** `LOGAN_EXPERT_PLAN_CACHE=1` — an `Arc`-shared lazily-populated
plan cache keyed by `(layer, expert)` (10,240 entries at this geometry), plus
new sub-step timers that separate plan construction from the MetalIO wait inside
the load term.

**Environment:** Apple M2 16 GiB; `deepsweet/Qwen3.6-35B-A3B-MLX-oQ4-FP16`;
`LOGAN_EXPERT_NOCACHE=1`, `QWEN_PROMPT="1 2 3 4 5 6 7 8"`, `QWEN_MAX_NEW=24`;
2 pairs off/on.

**Correctness gate:** Generated IDs identical in all runs. **PASS.**

**Measurements** (decode window, EXP-029 deltas):

| arm | `plan_ms/tok` | `wait_ms/tok` | `load_ms/tok` | plan hits/misses |
|---|---:|---:|---:|---|
| cache off (×2) | 1.2, 1.2 | 100.5, 101.1 | 118.9, 119.7 | 0 / 0 |
| cache on (×2) | **0.4, 0.4** | 100.1, 103.3 | 118.3, 121.0 | 6610 / 750 |

**Result:** The cache works exactly as intended — plan construction falls from
1.2 to 0.4 ms/token (3x, the residual 0.4 is the 750 compulsory first-touch
misses) — but that is **0.8 ms of a ~730 ms/token decode, about 0.1%**. The load
term is unmoved (118.9/119.7 vs 118.3/121.0, overlapping). The hypothesis is
falsified: host I/O-plan construction is not a meaningful cost here.

The instrumented decomposition is the useful output: of a ~119 ms/token load
term, **~101 ms is the MetalIO completion wait, ~1 ms is planning, and ~17 ms is
slot issue plus materialization**. The load term is therefore ~85% wait, which
redirects the work from host preparation to the arrival schedule of the bytes.

**Decision:** **REJECTED** as an optimization; the cache is retained behind
`LOGAN_EXPERT_PLAN_CACHE` (default off) because it is correctness-neutral, costs
one `HashMap` probe per expert, and makes the decomposition instrumentation
readable. No default changed. This entry exists mainly to close the handoff's
"precompute I/O plans" item with a measurement rather than an assumption.

**Artifacts:** `.perf_runs/routescout/EXP-030-plan-cache/{off,on}-{1,2}.log`,
`logan-qwen4/src/lib.rs` (`PlanCache`, `cached_io_plan`,
`mlx_expert_load_decomposition`).

---

## EXP-031 — RouteScout policy and budget sweep against the corrected measurement

**Date:** 2026-09-22  
**Area:** routed MoE / expert prediction / prefetch policy  
**Status:** **REJECTED**

**Hypothesis:** With decode-only counters (EXP-029) it becomes possible to see
whether speculative prefetch changes the decode wait, and with a correct
confidence gate a smaller budget (1) will beat the budget-8 configuration that
EXP-028 measured as 83% wasted.

**Baseline:** SSD-only raw MLX, `LOGAN_EXPERT_NOCACHE=1`, predictor off.

**Candidate arms:** predictor shadow-only (no I/O); budget 1 / budget 2 with
`QWEN_ROUTE_PREDICT_CONFIDENCE_GATE=0` (the default 0.50 gate blocks everything,
since measured online precision is 0.32–0.44); and a calibrated-gate variant at
`QWEN_ROUTE_PREDICT_MIN_PRECISION=0.20`.

**Critical methodological finding — the shipped gate makes every previously
published prefetch budget inert.** `prepare_route_prediction` requires
`pairs >= 16` **and** `precision >= 0.50` **per layer**. Measured online route
precision is **0.381 / 0.322 / 0.164** for budgets 1 / 2 / 8, so at the default
settings **no budget in {1, 2, 4, 8} issues a single speculative read**. The
budget dimension can only be exercised with the gate disabled or recalibrated;
EXP-028's budget-8 figure was obtained with the gate off and its 0.164 precision
is the budget-induced dilution, not the predictor's native quality.

**Measurements (decode window, `QWEN_MAX_NEW=24`, 23 forwards):**

| budget | precision | recall | speculative loads | used | wasted | `wait_ms/tok` | `load_ms/tok` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (shadow) | — | — | 0 | 0 | 0 | 105.0 | 123.5 |
| 1 (gate off) | **0.381** | 0.137 | 675 | 412 | 265 | 103.4 | 123.0 |
| 1 (gate 0.20) | 0.381 | 0.137 | 675 | 412 | 265 | 111.8 | 133.0 |
| 2 (gate off) | 0.322 | 0.231 | 1836 | 591 | 1244 | 102.3 | 122.0 |

**Result — decisive paired data (4 pairs, 8 runs per arm, alternating order,
same binary, correctness PASS: 1 distinct generated sequence across all 24 runs).**

| arm | median decode ms/tok | delta | paired wins | `peak_out` | speculative loads | ready | late |
|---|---:|---:|---:|---:|---:|---:|---:|
| serial (baseline) | **685.4** | — | — | 1 | 0 | — | — |
| serial + budget-1 prefetch | 751.6 | **+9.67%** | **1 / 4** | 2 | 5400 | 3296 | **0** |
| conc8 (EXP-032) | 699.5 | +2.05% | 1 / 4 | 8 | 0 | — | — |

Two things are now firmly established, neither of which depends on interpreting a
single run:

1. **Prefetch loses the paired test.** It is 9.67% *slower* at the median, loses
   3 of 4 pairs, and its per-pair median delta is +10.03%. The earlier draft's
   "+36%" came from one contaminated run; **+9.7% over 8 runs with a 3:1 paired
   loss record is the number that stands**, and it is now supported rather than
   contradicted by the data.
2. **It is not a readiness problem.** Across all 8 prefetch runs: 5400
   speculative loads, **3296 ready at demand, 0 late** — exactly the same
   all-ready profile EXP-028 reported. Two distinct ratios describe those runs
   and must not share a name: **61% of issued speculative loads were consumed**
   (`prefetch_used / prefetch_loads` = 3296/5400), while **arrival precision was
   0.381** (`arrival_correct / arrival_predicted` = 350/918). The prediction is
   arriving in time and its top-1 accuracy is 0.381, and the branch is still worth
   ~10% negative. That eliminates the last "the mechanism just needs better
   tuning" explanation.

**Why**: 2120 wasted reads (39% of the speculative traffic) plus 2104 useful reads
are issued on the *same* MetalIO queue and *same* SSD that the authoritative
demand reads must use (EXP-031 measured the demand wait rising 97.5 → 189.7 in the
contended run; EXP-032 measured the same queue-contention signature independently
via `compute_ms`). The saved wait does not exceed the displacement cost.

**Result — budget and precision/recall (stable, ratio-based, reproduced across
runs):**

| budget | precision | recall | speculative loads | used | wasted |
|---:|---:|---:|---:|---:|---:|
| 1 (gate off) | **0.381** | 0.137 | 675 | 412 | 265 |
| 2 (gate off) | 0.322 | 0.231 | 1836 | 591 | 1244 |
| 8 (EXP-028) | 0.164 | 0.413 | 8853 | 1448 | 7374 |

**Budget 1 is the better policy** — *higher* precision (0.381 vs 0.322) at 36% of
the speculative bytes, confirming the handoff's suspicion that top-k=8 does not
imply speculative budget 8, and that EXP-028's 0.164 was budget-induced dilution
rather than the predictor's native quality.

**Why the speculative reads cannot help on this host** (mechanism, from EXP-032's
decomposition): a demand read costs `submit ≈ 2.4 ms` + `wait ≈ 106 ms` +
`materialize ≈ 17.7 ms` per token. A prefetch issued in a *previous* layer can
remove at most the **wait**, only for the fraction it predicts correctly, it pays
the same `materialize` copy, and it displaces demand reads sharing the queue. The
addressable term is ~106 ms of a ~650 ms forward — and EXP-031's own prefetch arm
is the measured proof that attempting to bank it instead costs ~66 ms.

**Also note the gate's design flaw, independent of the numbers:**
`logan route-arrival layers` shows per-layer precision scattered 0.17–0.59, and
the gate is applied per layer against a *lifetime* per-layer estimate. Because it
is `precision >= 0.50`, it is binary: a layer at 0.49 gets nothing and a layer at
0.51 gets the full budget. EXP-034 later measured that the fusion weights feeding
this gate were also mis-set (temporal over-weighted), and fixed them.

**Decision:** **REJECTED** as a performance path. The evidence is a paired 3:1
loss with 100% readiness and 39% wasted reads, plus an independently-measured
queue-contention mechanism. No default was changed; the predictor and prefetch
remain opt-in and off, which the data shows is the correct shipped state. **Kept
as measurement:** the budget/precision/recall table, the inert-gate finding, the
readiness result, and the paired loss magnitude.

**Artifacts:** `.perf_runs/routescout/EXP-031-budget/`,
`.perf_runs/routescout/EXP-031-032-final/`, `tools/routescout_sweep.py`.

---

## EXP-032 — Concurrent per-layer expert I/O instead of serial issue-then-wait

**Date:** 2026-09-22  
**Area:** routed MoE / MetalIO / storage latency  
**Status:** **REJECTED**

**Hypothesis:** The decode path fetches its routed experts **one at a time**:
`MlxLocalExpertSource::eval` receives the layer's expert calls and, for each one,
submits a single MetalIO command and immediately blocks on its completion
(`metalio_slot_alloc` → `metalio_loadv` → `metalio_wait` → copy → `slot_free`).
The I/O queue therefore never holds more than one command — `peak_outstanding=1`
in every measured run. Since the measured per-read latency is dominated by fixed
submit/complete cost (p50 0.128 ms, p99 1.024 ms, for a ~1.6 MiB read whose
bandwidth time at M2 SSD rates is far smaller), issuing the layer's experts
concurrently should collapse ~8 serialized latencies into ~1, removing most of
the ~101 ms/token MetalIO wait from the decode critical path.

**Baseline:** current per-call serial fetch, `LOGAN_EXPERT_NOCACHE=1`,
`peak_outstanding=1`, `wait_ms_per_token≈101`.

**Candidate:** split the demand fetch into an **issue** phase and a **collect**
phase so one layer's expert loads overlap: in `eval`, first submit every
not-already-resident call's I/O into its own MetalIO slot without waiting, then
wait and materialize them. Prefetched experts continue to be consumed through the
existing `pending` path. Native routing stays authoritative; the set of experts
loaded and their bytes are unchanged, so results must be token-identical.
Gate: `LOGAN_EXPERT_IO_CONCURRENCY=N` (0 = legacy serial path) so the change is
A/B-able.

**Why this is not the rejected prefetch branch:** it does not predict anything
and does not read a byte that the authoritative router did not already select.
It removes idle time inside a read that is already required, so it is
correctness-neutral by construction rather than by prediction accuracy.

**Correctness gate:** byte-identical generated IDs; identical `calls` and
`mio_bytes`; no increase in `mio_fails`.

**Acceptance:** `peak_outstanding > 1` in the decode window and a measurable
paired reduction in `decode_ms_per_token` with `wait_ms_per_token` falling.

**Measurements:** decomposition of the load term (decode window, EXP-029 deltas),
same prompt/checkpoint, `LOGAN_EXPERT_NOCACHE=1`, `QWEN_MAX_NEW=24`. These are
**single unpaired runs** and are used only for the *decomposition*; the wall-clock
column is not a paired result and must not be read as one:

| term | serial (`=1`) | concurrent (`=8`) |
|---|---:|---:|
| `plan_ms/tok` | 1.2 | 2.4 |
| `submit_ms/tok` | **2.4** | **5.2** |
| `wait_ms/tok` | **106.3** | **27.2** |
| `materialize_ms/tok` | 17.7 | 20.4 |
| `load_ms/tok` (sum of the above) | 127.7 | 55.3 |
| `compute_ms/tok` | 278.1 | 322.7 |
| `peak_outstanding` | 1 | 8 |
| `decode_ms/token` (unpaired — see the correction below; not separable) | 633.8 | 679.8 |

Paired arms (first sweep, single build). The sweep runs each arm forward then
backward, so a complete arm has two runs; **both halves must be shown**, because
the ascending half alone is the most drift-exposed subset:

| arm | ascending run | descending run | median | `wait_ms` | `compute_ms` | peak |
|---|---:|---:|---:|---:|---:|---:|
| conc1 | 659.4 | — (sweep killed) | 659.4 | 111.7 | 288.5 | 1 |
| conc2 | 670.9 | — (sweep killed) | 670.9 | 102.1 | 295.6 | 2 |
| conc4 | 691.9 | 695.5 | 693.7 | 83.6 / 80.4 | 308.0 / 309.8 | 4 |
| conc8 | 697.0 | **665.1** | **681.1** | 30.4 / 26.3 | 332.8 / 322.5 | 8 |

**Correction, recorded because an earlier draft of this entry asserted the
opposite:** with both halves, `conc8`'s median (681.1) is *below* `conc4`'s
(693.7), so the decode column is **not** monotone in concurrency. That 12.6 ms
difference sits inside within-arm spreads of 3.6 ms (conc4) and 31.9 ms (conc8).
**The wall-clock arms are not separable at this resolution, and no decode
ordering between the concurrent levels should be read from them.** The earlier
"monotone regression" claim is withdrawn.

**What the counters establish instead — and these are noise-immune, because they
are deterministic or reproducible to within a few percent in every single run:**

1. The mechanism works: `peak_outstanding` reaches its cap (1 → 8), and
   `wait_ms/token` falls in *every* concurrent run.
2. In the 4-pair sweep the separation is unambiguous and consistent:
   `wait_ms` was **95.2–189.7** for serial/prefetch arms vs **26.9–37.0** for
   every `conc8` run across all 8 `conc8` runs — a ~3x separation far outside
   run-to-run noise.
3. `compute_ms` rose together with the wait reduction in 7 of 8 `conc8` runs
   (318.3–349.6 vs serial's 262.1–364.8), which is the direction EXP-019's
   UMA-pressure mechanism predicts.

**Decision is therefore based on the conservation argument, not the wall-clock
ordering:** the measured MetalIO wait drops by ~75 ms/token while `compute_ms`
rises ~45 ms/token and the wall clock does not separate — the wait moved into the
compute term rather than leaving the critical path. That is sufficient to refuse
promotion, and it does **not** require asserting a decode regression.

**Result:** The mechanism works exactly as designed — `peak_outstanding` reaches
the cap, and the measured MetalIO wait falls consistently
(**106.3 → 27.2 ms/token in the single-run decomposition; 95–190 → 27–37 across
all 8 four-pair runs**) — while `compute_ms` rises ~45 ms/token in the same
direction. The wall clock does **not** separate the arms reliably in either
sweep, so the reading is not "concurrency is slower" but the stronger structural
statement: **the freed wait did not leave the critical path; it reappeared in the
compute term.** That is exactly the signature of the UMA/queue contention already
recorded in EXP-019, and it means the concurrent path cannot convert its I/O
savings into wall time on this host.

**This is the same failure mechanism as EXP-019**, where a residency cache cut
the load term but regressed end-to-end on this 16 GiB host through UMA pressure.
Two independent mechanisms (retained weights, concurrent buffers) now show the
same signature: this host cannot convert a reduced expert-load term into wall
time.

**Correction to an earlier claim in this entry's own development:** an initial
single-run comparison (673.6 vs "~730") suggested a win. It was cross-run
variance, not a paired result — the same sweep's serial arm measured 659.4 on
this identical prompt. The claim is withdrawn; only paired numbers are used above.

**A second process error, recorded because it invalidated part of the run:** the
release binary was rebuilt while the first concurrency sweep was still executing,
so later runs in that directory came from a different binary. That directory was
moved to `.perf_runs/routescout/EXP-032-mixed-binary-invalid/` and only the
single-build pair above is cited. Rebuilding during a measurement is exactly the
comparability failure `AGENTS.md` warns about.

**Decision:** **SUPERSEDED by EXP-039.** As measured here this was REJECTED and
the knob defaulted to **1**. **EXP-039 later reversed it**: once EXP-037/038 cut
the MoE compute phase from ~1191 dispatches to two command buffers per layer, the
concurrent path became a verified win (1.2607x and 1.2915x in both arm orders) and
the shipped default is now **0 = issue the whole route**. The original text below
is retained because its conservation observation was correct at the time; do not
act on it without also reading EXP-039.

**Original decision:** **REJECTED.** `LOGAN_EXPERT_IO_CONCURRENCY` defaults to **1**
(serial); the concurrent path is retained opt-in for A/B on hosts with different
UMA characteristics, but it is not enabled. The useful output is the corrected
attribution: of a ~332 ms expert phase per forward, the *storage wait* is
**~106 ms (~32% of the expert phase, ~17% of the forward)** — not the dominant
term — while the GEMM/compute term is ~278 ms. Any further work should target
`compute_ms`, and any prefetch or overlap attempt on this host must first explain
why it will not pay the UMA/queue penalty that defeated both EXP-019 and this
entry.

**Artifacts:** `.perf_runs/routescout/EXP-032-io-concurrency-single/` (retained
paired data), `.perf_runs/routescout/EXP-032-mixed-binary-invalid/`,
`tools/routescout_sweep.py`, `logan-qwen4/src/lib.rs` (`DemandFetch`,
`issue_demand_expert`, `collect_demand_expert`, `mlx_expert_load_decomposition`).

---

## EXP-033 — Graded speculative budget and a spatial prediction horizon

**Date:** 2026-09-22  
**Area:** routed MoE / expert prediction / prefetch policy  
**Status:** **REJECTED (implemented, not promoted)**

**Hypothesis:** The shipped confidence gate is binary (`precision >= 0.50` per
layer against a lifetime estimate), and EXP-031 measured that real per-layer
precision spans 0.17–0.59 — so a layer at 0.49 spends the same budget as one with
no evidence, while a layer at 0.51 spends the full budget. Replacing it with a
**graded budget** driven by a candidate score margin (0 / 1 / 2, capped) should
raise useful prefetches per speculative byte, and predicting a **future layer**
(L+1..L+4) rather than only the current one should give the I/O real lead time
instead of issuing it immediately before demand.

**Baseline:** budget-1, gate-off, same-layer prediction (the best arm from
EXP-031).

**Candidate:** `QWEN_ROUTE_PREDICT_BUDGET_MAX` with a score-margin threshold, and
a layer-offset option that predicts arrivals for `li + horizon` using the target
layer's transition tables and the current token's already-observed spatial
evidence.

**Constraint that must be respected (from EXP-032):** the addressable term is the
~106 ms/token MetalIO *wait*, and each successful prefetch pays the same
~17.7 ms/token-equivalent `materialize` copy in the issue path. A candidate can
only win if it converts wait into already-completed I/O *without* adding copy or
UMA pressure — so this entry's primary metric is `decode_ms_per_token`, and a
lower `wait_ms` with a higher `compute_ms`/`materialize_ms` is a **loss**, exactly
as in EXP-032.

**Correctness gate:** byte-identical generated IDs; prediction must never alter
the authoritative expert set.

**Measurements / outcome.** The graded budget (`graded_selection`, score-margin
proportional, unit-tested) is implemented and the horizon wiring exists, but the
mechanism was **not** taken to a decoder A/B, for a reason that is itself the
result: EXP-034 ran the offline screen these policies depend on and found the
governing variable is not the budget *policy* at all.

- Cold-arrival precision at budget 1 is **0.52 mean / 0.73–0.76** on the
  structured prompt families — far above the 0.381 this entry's premise assumed
  from a single-prompt decode. The gate threshold was never the binding problem;
  the decoder's short-horizon warmup was.
- The fusion *weights* feeding the score were measurably wrong (EXP-034 fixed
  them), which is a more fundamental correction than re-shaping how the budget is
  sized from those scores.
- Horizons are a dead branch (EXP-034): recall halves from `h0` to `h1` and
  plateaus, so there is no lead time to purchase.

**Decision:** **REJECTED (not promoted).** The graded budget is retained in source
as a correctness-neutral, unit-tested mechanism but is **not enabled and not
defaulted**; enabling it would only make the rejected prefetch path issue
different amounts of the same harmful I/O (EXP-031: −9.67% paired). The horizon
wiring is likewise not defaulted. Both are documented here rather than left as
latent flags so the next agent knows the policy half of this branch was explored
and *why* it did not proceed, instead of re-deriving it.

**Artifacts:** no run directory — this entry produced no decode runs (see
Decision), so the evidence is the code and the screening that superseded it:
`logan-qwen4/src/lib.rs` (`graded_selection` + its 3 unit tests),
`tools/routescout_horizon_weights.py`, and
`.perf_runs/routescout/EXP-034-weights-horizon/weights-horizon.json`.

---

## EXP-034 — Temporal/spatial fusion weights and the spatial horizon

**Date:** 2026-09-22  
**Area:** routed MoE / expert prediction / prefetch policy  
**Status:** **KEPT (weights) / REJECTED (horizons)**

**Hypothesis:** (a) The runtime fuses temporal and spatial transition evidence with
*equal* weight, but every published cross-prompt result says spatial is stronger,
so de-weighting temporal should improve cold-arrival precision at equal byte
budget. (b) A spatial horizon (predict layer `L+H` from evidence available at `L`)
should retain enough recall to be worth the extra I/O lead time, making an `L+4`
predictor able to beat `L+1` by hiding more latency.

**Baseline:** the runtime's current fusion — unnormalised conditional temporal
evidence plus unnormalised conditional spatial evidence, summed equally
(`t1_s1` in the table).

**Candidate:** peak-normalise each term independently, then weight
`spatial = 1.0` against `temporal ∈ {1.0, 0.5, 0.25, 0.0}`.

**Why this is measured offline first:** EXP-031/EXP-032 established that
whole-model decode A/B on this host cannot resolve effects at this scale
(±25% run-to-run spread against an addressable ~106 ms of a ~650 ms forward).
Prediction *quality* is a trace property and is measured here on the real
Qwen3.6 route traces with no decode at all, so the screening is exact and
cheap. Only a candidate that wins this screen is worth decoder time.

**Environment:** `tools/routescout_horizon_weights.py`; leave-one-prompt-out over
the four real prompt families (`routescout-prompt-{rust,moe,science,hash}.tsv`,
19 cycles each, 40 layers, 256 experts, top-8, cross-prompt priors). Cold
arrivals only (actual route minus previous route at the same layer), ranked after
masking the previous route, so precision is what a prefetcher observes.

**Correctness gate:** offline analysis of already-captured authoritative routes;
no runtime path is touched by the measurement.

**Measurements — temporal/spatial weight sweep (mean over 4 holdouts):**

| budget | metric | t1 (current) | t0.5 | **t0.25** | t0 (spatial only) |
|---:|---|---:|---:|---:|---:|
| 1 | precision | 0.5174 | 0.5285 | **0.5351** | 0.5160 |
| 1 | recall | 0.0997 | 0.1020 | **0.1033** | 0.0997 |
| 2 | precision | 0.4786 | 0.4894 | **0.4957** | 0.4781 |
| 4 | precision | 0.4135 | 0.4250 | **0.4279** | 0.4119 |
| 4 | recall | 0.3188 | 0.3283 | **0.3308** | 0.3188 |
| 8 | precision | 0.3102 | 0.3196 | **0.3215** | 0.3075 |
| 8 | recall | 0.4793 | 0.4946 | **0.4978** | 0.4766 |

**`t0.25` wins all 16 cells** (4 holdouts × 4 budgets) on both recall and
precision, and beats pure-spatial `t0` as well — so temporal evidence carries real
signal, it is simply over-weighted at 1.0. The gain is small but perfectly
consistent: **+3.4% relative precision at budget 1 and +3.9% relative recall at
budget 8**, reproduced on every holdout independently (e.g. budget 1 precision:
rust 0.164→0.199, moe 0.728→0.738, science 0.418→0.439, hash 0.760→0.765).

Note this **corrects an earlier run of this same script on only 2 traces**, where
`t0` (spatial-only) appeared best. With the full 4-prompt corpus `t0` is clearly
worse than `t0.25`; the 2-prompt result was corpus-limited, not a real ordering.
Prompt-family domination (EXP-014) is exactly why the holdout set matters here.

**Measurements — spatial horizon (mean over 4 holdouts):**

| budget | metric | **h0** | h1 | h2 | h4 | h8 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | recall | **0.0997** | 0.0529 | 0.0535 | 0.0540 | 0.0491 |
| 1 | precision | **0.5174** | 0.2714 | 0.2734 | 0.2770 | 0.2552 |
| 8 | recall | **0.4793** | 0.2769 | 0.2690 | 0.2668 | 0.2501 |
| 8 | precision | **0.3102** | 0.1903 | 0.1893 | 0.1867 | 0.1702 |

**Result (a):** FOUND. The runtime is over-weighting temporal evidence; `temporal
= 0.25, spatial = 1.0` after per-term peak normalisation is better in every
measured cell.

**Result (b): NOT FOUND, and decisively so.** Recall **halves** from `h0` to
`h1` (0.0997 → 0.0529 at budget 1; 0.4793 → 0.2769 at budget 8) and then
*plateaus* — `h2`, `h4`, and `h8` are all within noise of `h1` and slightly worse.
There is **no horizon at which additional lead time is purchased with acceptable
accuracy loss**: the accuracy cost is paid entirely in the first step, and further
lead is free-but-useless. This closes the handoff's Phase 4 "spatial horizon"
branch on this target: a longer horizon does not buy a better prediction, so the
premise that "a lower-accuracy L+4 predictor can outperform L+1 if it hides much
more SSD latency" requires a larger storage miss penalty than this host has. It is
consistent with EXP-018/029/032: the current same-layer prediction is already
issued roughly one token ahead (EXP-028 observed 100% of useful prefetches ready
at demand), so there is no latency left for a horizon to hide.

**Runtime verification (and its limit).** The weight change is implemented
(`QWEN_ROUTE_PREDICT_W_TEMPORAL`, default 0.25, with per-term peak normalisation)
and measured on the real decoder. The `route-arrival` line is a **deterministic**
function of the routes, so it is directly reproducible and needs no repetition
budget — which makes the following unambiguous:

| `W_TEMPORAL` | correct | predicted | precision | runs |
|---|---:|---:|---:|---:|
| 0.25 (new) | 351 | 918 | 0.382 | 2/2 identical |
| 1.0 (previous) | 353 | 918 | 0.385 | 4/4 identical |

**The offline gain does not transfer to the runtime: 0.382 vs 0.385 is a 0.8%
relative difference in the *opposite* direction from the offline result**, and
both figures are reproducible rather than noisy. Generated IDs were identical in
every run.

**One unexplained observation, recorded rather than attributed.** A single earlier
run (from the build that existed before the graded-budget code was removed) at
`W_TEMPORAL=1.0` reported 350/918 = 0.381. It was **not reproducible**: four
subsequent runs of the current build at that identical setting all report 353, and
its per-run log was removed during cleanup so its generated IDs cannot be compared.
It is therefore logged as an unreproduced outlier. **There is no evidence of
token- or route-level nondeterminism** — every run whose tokens were inspected
produced the canonical 24-token sequence, and the counters agreed whenever the
setting was held fixed.

**Why the weights differ so little at runtime:** the offline harness fits on the
entire cross-prompt corpus, while the online predictor sees ~30 pairs per layer in
a 24-token decode, so its transition tables are too sparse for a weighting change
to express itself. The change is kept on the strength of the offline 4-prompt LOO
result (16/16 cells at corpus scale), with its runtime effect recorded as
**below resolution and sign-disagreeing at this decode length**. It is
correctness-neutral, and prefetch is off by default, so no shipped path changes.

**Decision:** **KEPT** for the fusion weights — implemented in the runtime
(per-term peak normalisation plus `QWEN_ROUTE_PREDICT_W_TEMPORAL`, default 0.25),
on offline evidence, with the runtime effect explicitly recorded as unmeasurable
and sign-disagreeing at this decode length.
**REJECTED** for spatial horizons — not implemented; the runtime keeps same-layer
prediction. The horizon result is retained specifically so this branch is not
re-opened without a storage target that genuinely stalls.

**Note on what this entry does and does not buy.** It improves the *predictor*, and
the predictor is not the binding constraint — EXP-031 showed that even a perfectly
readied prefetch at 61% precision loses ~10% on this host. Improving prediction
quality therefore cannot convert into a win here; it is worth keeping only because
it costs nothing and is the correct form for a future target with a real miss
penalty.

**Artifacts:** `.perf_runs/routescout/EXP-034-weights-horizon/weights-horizon.json`,
`tools/routescout_horizon_weights.py`.

---

## EXP-035 — Final slice verification: shipped defaults and correctness gates

**Date:** 2026-09-22  
**Area:** methodology / verification  
**Status:** **KEPT**

**Hypothesis:** Every promoted or retained change in this slice must be
default-off or provably neutral, and the measurement fix must not alter numerics.
This entry is the end-to-end check of that claim, run in the foreground so the
output is on disk rather than lost to a truncated background job.

**Environment:** Apple M2 16 GiB; `deepsweet/Qwen3.6-35B-A3B-MLX-oQ4-FP16`;
`QWEN_MAX_NEW=24`; `LOGAN_EXPERT_NOCACHE=1` for the SSD-only arms.

**Measurements and assertions:**

| # | Check | Result |
|---:|---|---|
| 1 | Canonical SSD-only gate: `nocache=true metalio=true`, structural expert calls | `calls_per_token=320.0` (= 23x40x8/23), `fails=0`, canonical IDs |
| 2 | Shipped defaults inert: budget 8, gate ON | `mio loads=0` despite `predicted=7333` |
| 3 | Shipped `io_concurrency` is serial | `peak_outstanding=1` with the flag unset |
| 4 | Profiling off is behavior-neutral | canonical IDs, **0** profile lines emitted |
| 5 | Plan-cache default is OFF | `plan_hits=0 plan_misses=0` with the flag unset |

**Test suites (all run in the foreground, all green):**

    logan-qwen4  --lib : 95 passed, 0 failed, 3 ignored
    logan-metal  --lib :  5 passed, 0 failed
    logan-core   --lib : 119 passed, 0 failed
    logan-compiler --lib: 135 passed, 0 failed

**Result:** All five configuration assertions hold, and every generated sequence
across checks 1-4 is the canonical
`[348, 10, 4838, 1665, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 33898, 2110,
30, 31, 73307, 58, 3312, 87197, 62]`. Checks 2 and 3 are the ones the slice's
promotion gate actually rests on: the predictor and prefetch remain opt-in and
off, and the **rejected** concurrent I/O path is **not** the default (an earlier
draft of EXP-032 temporarily shipped `io_concurrency=8` before it was measured;
that is reverted and verified here).

**Note on the absolute timings in this entry:** the runs in check 1-3 are slower
in absolute ms than the paired sweep's serial arm because they executed while
other work was in flight. This entry is **correctness and configuration evidence
only**; no performance claim is drawn from it, and all timing claims in this slice
come from the paired within-sweep comparisons of EXP-031/EXP-032.

**Decision:** **KEPT.** This is the verification record for the slice.

**Artifacts:** `.perf_runs/routescout/EXP-035-final-verification/{README.md,
verify.sh, determinism.sh, profiling-off.log}`.

---

## EXP-036 — Direct cross-layer RouteScout with real SSD prefetch lead

**Date:** 2026-09-22  
**Area:** routed MoE / RouteScout / raw MLX / MetalIO  
**Status:** **KEPT (correct direct-horizon mechanism) / NOT PROMOTED (no M2 wall-time win)**

**Motivation / correction to EXP-034:** Review found EXP-034's horizon harness did
not train a true source-layer -> target-layer transition for H>0. It fed expert IDs
from an earlier observed layer into the adjacent target-1 -> target table, so the
source distribution did not match the table. A corrected leave-one-prompt-out
analysis over the same four real Qwen3.6 traces, training direct
`source -> source+H+1` transitions, retained most of the signal through H=8:

| horizon | budget-1 precision | budget-1 recall | budget-8 precision | budget-8 recall |
|---:|---:|---:|---:|---:|
| H0 | 0.532 | 0.104 | 0.317 | 0.497 |
| H1 | 0.529 | 0.103 | 0.309 | 0.484 |
| H2 | 0.516 | 0.101 | 0.305 | 0.478 |
| H4 | 0.512 | 0.100 | 0.300 | 0.468 |
| H8 | 0.494 | 0.095 | 0.291 | 0.450 |

Prompt-local warmup preserves the relative horizon signal as well (12-cycle
budget-1 precision H0/H4/H8 = 0.381/0.357/0.335).

**Hypothesis:** A correctly trained direct cross-layer predictor can issue one
speculative expert read 4-8 layers before demand, moving useful SSD traffic away
from the authoritative demand window. At equal budget 1 this should reduce the
queue-contention penalty seen in EXP-031 while retaining enough precision to be
useful.

**Baseline:** raw MLX/safetensors Qwen3.6 with `LOGAN_EXPERT_NOCACHE=1`, serial
authoritative demand I/O, RouteScout prefetch off.

**Candidates:** `QWEN_ROUTE_PREDICT_HORIZON=4` and `=8`, budget 1, using the same
online predictor but training each target layer's spatial table from the matching
same-token source layer `target-(H+1)`. Native routing remains authoritative.

**Correctness gate:** canonical 24-token greedy output must be exactly identical
to baseline; no MetalIO failures.

**Primary metrics:** decode ms/token, arrival precision/recall, speculative
loads/used/wasted, ready/late at demand, demand wait, peak outstanding, and paired
wall-time delta. No promotion from prediction quality alone.

**Implementation:** `QWEN_ROUTE_PREDICT_HORIZON=H` now changes both sides of
the spatial predictor consistently. For target layer T, the table is trained from
the same-token source layer `T-(H+1)` and prediction before layer `li` queries that
table with the newest route actually available (`li-1`), where `T=li+H`.
Out-of-range tail predictions are skipped rather than clamped to the last layer.
Raw MLX/safetensors remains the primary test path; no format conversion is involved.

**Shadow/live prediction gate (24 generated tokens, no speculative I/O):**

| horizon | correct/predicted | precision | recall | decode ms/tok |
|---:|---:|---:|---:|---:|
| H0 | 351/918 | 0.382 | 0.137 | 919.5 |
| H4 | 323/828 | **0.390** | **0.151** | 904.4 |
| H8 | 287/736 | **0.390** | **0.150** | 910.0 |

Generated IDs were identical in all three runs. This directly falsifies EXP-034's
claim that horizon quality halves immediately: the corrected online predictor retains
the same short-decode precision at H4/H8.

**SSD prefetch mechanism (24-token mirrored order B -> H4 -> H8 -> H8 -> H4 -> B,
`LOGAN_EXPERT_NOCACHE=1`, budget 1, confidence gate disabled to hold policy equal):**

| arm | runs ms/tok | median | speculative loads | used | wasted | late | wait ms/tok |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | 887.6, 878.8 | **883.2** | 0 | 0 | 0 | — | 98.7, 95.8 |
| H4 | 886.2, 868.0 | **877.1 (-0.7%)** | 605/run | 389 | 219 | 0 | 97.7, 93.9 |
| H8 | 876.0, 894.8 | **885.4 (+0.25%)** | 518/run | 344 | 176 | 0 | 96.6, 97.8 |

The 24-token result is suggestive only. Its important structural finding is that
direct-horizon prefetch does **not** reproduce EXP-031's same-layer demand-wait
explosion. Useful reads are fully ready, and aggregate demand wait stays at or below
baseline instead of roughly doubling.

**Longer 48-token mirrored H4 check:**

- hot half: baseline 1037.0 vs H4 1054.7 ms/tok (**+1.7% slower**)
- cooler half: baseline 885.6 vs H4 891.3 ms/tok (**+0.6% slower**)
- H4 prediction: 665/1692 = 0.393 precision, 0.116 recall
- H4 speculative I/O per run: 1249 loads, 784 consumed, 466 wasted, 784 ready, 0 late
- demand wait: 121.1 vs 122.0 ms/tok (hot), 104.7 vs 104.7 (cool)
- generated 48-token IDs were exactly identical in all runs.

This is the decisive part for the current host: once the run is long enough to reduce
the short 24-token noise, direct-H4 removes the old queue-contention failure but still
does not convert that into a wall-time win.

**Confidence-gated H4 follow-up:** A 30% precision / 8-sample gate reduced a
48-token run to 738 speculative loads, 562 consumed, 147 wasted, all 562 ready and
0 late. Demand wait was 104.0 ms/tok versus bracketing baselines at 104.7 and
104.4, yet decode was 897.6 ms/tok versus baselines 885.6 and 879.6 (~+1.7% against
their mean). The compute term was 420.8 ms/tok versus 415.9 and 407.0 in the
bracketing baselines. This points to shared-memory/UMA interference from background
I/O rather than authoritative MetalIO queue delay as the remaining cost.

**Additional 24-token corroboration (later same-day rerun):** Six alternating H4 baseline/forced-prefetch pairs gave candidate deltas of +0.069%, +0.779%, +0.196%, -11.818%, +2.159%, +0.461%. The only win coincided with a clear baseline I/O outlier (976.5 ms/token, 131 ms/token expert wait). Excluding that outlier, H4 lost 5/5 pairs with median +0.461%, while median expert-wait improved by about 1.45 ms/token. This independently agrees with the longer 48-token result: earlier reads are real, but the small wait saving does not become a wall-time win on this host.

**Production-default gate spot-check:** With the normal 50% precision / 16-sample confidence gate at H4, only 43 speculative decode-window loads were issued; 39 were consumed, 0 were classified wasted, all 39 were ready at demand, and 0 were late. This is strong evidence that the default gate is appropriately selective. No wall-time claim is attached to this spot-check because EA FC 25 and wineserver were concurrently consuming substantial CPU; the resulting machine-state drift is recorded in `.perf_runs/routescout/EXP-036-direct-horizon/system-load.txt`.

**Offline correction retained in tooling:** `tools/routescout_horizon_weights.py`
now trains direct horizon-specific source->target tables. Re-running the four-prompt
leave-one-out corpus gives mean budget-1 precision H0/H4/H8 =
0.535/0.514/0.489 and budget-8 recall H0/H4/H8 = 0.498/0.469/0.447.

**Correctness / regression gates:**
- `cargo check -p logan-qwen4`: PASS
- route predictor focused tests: 7 passed
- `cargo test -p logan-qwen4 --lib`: 95 passed, 0 failed, 3 ignored
- every real-model EXP-036 run checked produced the same authoritative token IDs
- `cargo fmt -p logan-qwen4 -- --check` still reports pre-existing formatting
  drift in several OMP/scratch files and broad `lib.rs` sections; no mass-format was
  applied because the working tree is shared and heavily dirty.

**Decision:** **KEEP the direct-horizon implementation and corrected analysis, but
do not enable speculative prefetch by default on this M2.** EXP-034's horizon
rejection is superseded: long-horizon RouteScout is genuinely predictable. However,
EXP-036 shows that on this 16 GiB UMA host, even correctly early and fully-ready
speculative reads are roughly neutral-to-slightly negative once measured over longer
runs. The next useful systems hypothesis is not 'more lead' or 'more accuracy'; it is
**reducing background I/O's interference with compute** (throttling/priority/phase-
aware issue) or moving RouteScout to a target where storage/network latency dominates
more strongly.

**Artifacts:** `.perf_runs/routescout/EXP-036-direct-horizon/`,
`tools/routescout_horizon_weights.py`, and direct-horizon runtime wiring in
`logan-qwen4/src/lib.rs` plus loader initializers.

---

## EXP-037 — Expert gate/up projection command-buffer batching

**Date:** 2026-09-23  
**Area:** routed MoE / Metal dispatch / decode throughput  
**Status:** **KEPT**

**Motivation:** EXP-018 established that the routed-expert phase is not
storage-bound (warm `pread` floor ~51 ms/token) but *dispatch*-bound: each
routed-expert GEMM was a synchronous Metal dispatch with its own `commit` +
`waitUntilCompleted`. EXP-029 corrected the accounting and put the decode-window
expert terms at ~148 ms load + ~335 ms compute per forward, i.e. roughly 240 us
per affine dispatch. A 512x2048 4-bit projection is ~590 KB, which is ~6 us of
UMA traffic — so the bulk of each dispatch is per-command-buffer overhead, not
data movement.

**Hypothesis:** If the per-dispatch cost is command-buffer overhead, then
encoding several independent GEMVs that consume the *same* activation into ONE
command buffer will remove it. `metal_matmul_mlx_affine_multi` already
implements exactly that (up to 16 descriptors, all sharing input width `I`) and
is already used by the GDN fused-input path (`lib.rs`, `QWEN_GDN_FUSED_INPUT`) —
but the routed-expert path never used it: `MlxLocalExpertSource::eval` issued one
`matmul` per matrix.

**Geometry that permits batching:** every routed expert in a layer consumes the
SAME token activation, so all `2*k` gate/up projections share `I = d_model` and
collapse into one command buffer. The `k` down projections cannot join: each
consumes its own expert's SwiGLU output, and the batch entry point requires one
shared activation. At `topk=8` that is 24 dispatches → 9 command buffers.

**Standalone probe — `logan-qwen4/examples/affine_dispatch_probe.rs`:**

Real Qwen3.6 expert geometry (d_model 2048, d_hidden 512, topk 8, 4-bit affine
group 64), 200 iterations, no model loaded:

| shape | us / layer | command buffers | us / dispatch |
|---|---:|---:|---:|
| serial (today's path) | 6682.7 | 24 | 278.5 |
| batched gate/up | 3551.9 | 9 | 394.7 |

**speedup 1.88x**, `max_abs_diff = 0.000000000`, `bit_identical = true`.

**Paired real-model A/B:** `LOGAN_EXPERT_NOCACHE=1`, sampled decode
(`BENCH_TEMP=1.0`, fixed `BENCH_SEED`), 16 generated tokens, 6 interleaved pairs
(12 runs), both arms from ONE binary via
`LOGAN_EXPERT_BATCH_GATEUP={1,0}` so codegen is held constant. Pooled per-step
median over 90 measured forwards per arm:

| arm | median ms/token | tok/s |
|---|---:|---:|
| off (per-matrix, EXP-018 shape) | 653.60 | 1.5300 |
| on (batched gate/up) | 505.86 | 1.9768 |

**+29.2% decode throughput**, and `identical_across_arms=True`: the batched arm
reproduced the per-matrix arm's token sequence exactly (1 distinct trajectory in
each arm, and the two arms' sequences are equal). The greedy trajectory
`248068,198,8160,579,264,7047,1817,25,271,16,13,220,2972,15771,2598,2570` also
matches the pre-change baseline byte for byte.

**Why this is not the rejected EXP-032 shape:** EXP-032 overlapped the *I/O*
(`LOGAN_EXPERT_IO_CONCURRENCY`), which moved wait into compute and lost. This
change leaves I/O ordering and concurrency completely untouched — the fetch loop
still issues and collects in the same order at `io_concurrency=1` — and only
removes redundant Metal command buffers from the compute phase. The probe
measures the compute shape in isolation, which is why the effect is clean.

**Correctness / regression gates:**
- bit-identical probe output (`bit_identical=true`)
- identical real-model greedy and sampled token trajectories across arms
- the shared-activation precondition is checked at runtime, not assumed; a
  mismatch falls back to the per-matrix path, so a delegating `ExpertSource`
  that does not share activations cannot be silently mis-computed

**Decision:** **KEPT, default ON**, with `LOGAN_EXPERT_BATCH_GATEUP=0` retained
so the A/B is reproducible.

**Next (same mechanism, not yet done):** the `k` down projections are the
remaining 8 command buffers per layer. They need either a grouped kernel that
takes heterogeneous activations in one dispatch, or SwiGLU fused into the
gate/up batch. That is the obvious follow-up; the probe's remaining serial
portion is ~8 x 278 us per layer.

**Artifacts:** `.perf_runs/autoresearch/ab-batch/`,
`logan-qwen4/examples/affine_dispatch_probe.rs`.

---

## EXP-038 — Two-command-buffer MoE compute phase (per-descriptor activation)

**Date:** 2026-09-23  
**Area:** routed MoE / Metal dispatch / decode throughput  
**Status:** **KEPT**

**Motivation:** EXP-037 batched the `2*k` gate/up projections into one command
buffer and left the `k` down projections as one dispatch each — 9 command buffers
per layer instead of 24. The down projections could not join because each
consumes its own expert's SwiGLU output, while `coli_metal_matmul_multi` copied a
single shared activation `x` into one buffer and required every descriptor to
match its width.

**Hypothesis:** The per-command-buffer cost is not tied to the activation being
shared. If the C entry point accepted a per-descriptor activation, the `k` down
projections could share a command buffer with each other, taking the layer to
**2** command buffers (one for gate/up, one for down) with the host SwiGLU loop
between them.

**Implementation:**
- `ColiMetalMatmulDesc` (`logan-metal/metal/backend_metal.h`) gained `x` and `S`:
  `x == NULL` means "use the function-level shared activation" (the original
  contract), non-NULL supplies a private activation with its own batch size.
- `coli_metal_matmul_multi` now decides *before* touching the shared buffer
  whether any descriptor needs it, requires agreement on `I` only among the
  descriptors that fall back to it, and uploads it once. Descriptors with their
  own activation upload into per-descriptor buffers
  (`g_multi_xs`/`g_multi_x_caps`).
- Rust: `MlxAffineMatmulDesc` gained `x: Option<&[f32]>`;
  `matmul_mlx_affine_multi_x` is the general form and `matmul_mlx_affine_multi`
  is the shared-activation wrapper, so existing callers are unchanged.
- `MlxLocalExpertSource::eval` now runs two phases: 2*k gate/up shared-activation
  batch → host SwiGLU → k down private-activation batch.

**Bug found and fixed during bring-up (important):** the first version SIGSEGV'd
(exit 139) on the first MoE layer. The C function computed
`shared_bytes = S * descs[0].I * 4` and `memcpy`'d from the caller's `x`
*unconditionally*, before inspecting any descriptor. In an all-private batch the
caller passes an empty placeholder, whose Rust slice pointer is a dangling low
address — so this was a genuine crash, not a benign over-read. Fixed by checking
`x != NULL && S > 0 && shared_I > 0` only when some descriptor actually falls
back to the shared activation. Recorded here because "the shared slice is never
read" was the wrong assumption to reason from; the buffer was read before any
per-descriptor logic ran.

**Correctness gate:** greedy trajectory
`248068,198,8160,579,264,7047,1817,25,271,16,13,220,2972,15771,2598,2570`
byte-identical to baseline, and `identical_across_arms=True` in the paired A/B
(1 distinct trajectory per arm, equal between arms).

**Paired real-model A/B:** `LOGAN_EXPERT_NOCACHE=1`, sampled decode
(`BENCH_TEMP=1.0`, fixed seed), 16 tokens, 6 interleaved pairs, one binary via
`LOGAN_EXPERT_BATCH_GATEUP={1,0}`, pooled per-step median over 90 forwards/arm:

| arm | median ms/token | tok/s |
|---|---:|---:|
| off (per-matrix) | 774.42 | 1.2913 |
| on (two-phase) | 547.82 | 1.8254 |

**1.4136x** versus the per-matrix shape (the gate/up-only variant of EXP-037
measured 1.2921x on the same harness), so the down-projection batching added a
further ~9%.

**Canonical harness:** `tok_per_sec` **1.7725 -> 1.9146** (+8.0%), with
`greedy_trajectory_sha` unchanged from the baseline run.

**Decision:** **KEPT, default ON.** `LOGAN_EXPERT_BATCH_GATEUP=0` restores the
per-matrix shape for A/B.

**Artifacts:** `.perf_runs/autoresearch/ab-batch/`,
`logan-qwen4/examples/affine_dispatch_probe.rs`.

---

## EXP-039 — Re-opening `LOGAN_EXPERT_IO_CONCURRENCY` after the dispatch batching

**Date:** 2026-09-23  
**Area:** routed MoE / MetalIO / decode throughput  
**Status:** **KEPT** (supersedes EXP-032)

**Why EXP-032's rejection no longer applies.** EXP-032 measured
`LOGAN_EXPERT_IO_CONCURRENCY > 1` as equal-or-worse and recorded the mechanism as
"the freed wait did not leave the critical path, it reappeared in compute". That
was measured when the MoE compute phase was ~1191 synchronous affine dispatches
per forward at ~278 us of command-buffer overhead each (EXP-018/029). At that
scale a ~50 ms/token wait saving was invisible against ~335 ms/token of dispatch
overhead. EXP-037/EXP-038 reduced the MoE compute phase to **two** command
buffers per layer, changing exactly the ratio EXP-032's conclusion rested on.

**Change:** `LOGAN_EXPERT_IO_CONCURRENCY` now defaults to `0`, meaning "issue the
whole route"; an explicit value caps the in-flight reads. (Previously 1 = serial.)

**Paired real-model A/B,** sampled decode, 16 tokens, 6 interleaved pairs, one
binary, pooled per-step median over 90 forwards per arm:

| run order | off (serial) ms/token | on (whole route) ms/token | speedup |
|---|---:|---:|---:|
| `on, off` | 566.53 | 449.38 | **1.2607x** |
| `off, on` (reversed) | 578.68 | 448.07 | **1.2915x** |

Both orders agree, so this is not the arm-position artifact a fixed-order schedule
can produce. `identical_across_arms=True`; the greedy trajectory is byte-identical
to baseline.

**Mechanism:** `peak_outstanding` 1 -> 8 and `wait_ms_per_token` 129.9 -> 84.4,
with per-read `p50` latency collapsing 0.256 -> 0.002 ms. The `load` term fell
149.9 -> 118.9 ms/token. (In that single profile run `compute_ms` read 154.9 vs
121.5 — the two terms trade off, which is why the *paired* A/B is the evidence
here rather than the profile.)

**Canonical harness:** `tok_per_sec` 1.9146 -> **2.1509** (+12.3%).

**Decision:** **KEPT, default 0 = whole route.** `LOGAN_EXPERT_IO_CONCURRENCY=1`
restores the serial behavior. Any further change to the MoE compute shape should
re-check this knob, because the two terms trade off.

---

## EXP-040 — Vectorized 4-bit MLX affine GEMV branch: the kernel was ALU-bound, not bandwidth-bound

**Date:** 2026-09-23  
**Area:** Metal kernel / MLX affine GEMM / decode throughput  
**Status:** **KEPT**

**Motivation:** EXP-037/EXP-038 removed the per-command-buffer overhead from the
MoE phase, so the remaining cost inside `mlx-expert` compute had to be kernel
time. EXP-018 had assumed a fixed ~278 us per dispatch independent of size; if
true, a 512x2048 4-bit projection (~590 KB) should be ~6 us of UMA traffic and
time should not scale with the matrix.

**Diagnostic — `affine_dispatch_probe` with `PROBE_TOPK` sweeping 1..8 experts
(byte count scaling):**

| experts | moved bytes | time | achieved GB/s |
|---:|---:|---:|---:|
| 1 | 1.77 MB | 1.11 ms | 1.6 |
| 2 | 3.54 MB | 1.50 ms | 2.4 |
| 4 | 7.08 MB | 3.20 ms | 2.2 |
| 8 | 14.2 MB | 6.84 ms | 2.1 |

Time scales linearly with bytes and achieved bandwidth is a flat **~2.1 GB/s**
regardless of size. So this is not fixed dispatch overhead — the kernel itself is
**ALU-bound**. For calibration, the repository's own `moe_gemv` kernel documents
358-389 GB/s on the same block shapes.

**Root cause:** the generic bitstream branch of `mm_gemv`
(`fmt 16..19` / `21..24`, backend_metal.mm) decodes **one element per lane
iteration**, with a 32-bit word load, a variable shift, a cross-word fixup, an
integer divide `i / gsz`, and two scale/bias gathers per element.

**Fix:** the packing is LSB-first with consecutive columns in consecutive
nibbles, so one `uchar4` load covers **8 columns** — byte *k* holds column *2k* in
its low nibble and column *2k+1* in its high nibble (`mlx_affine_code` in
`logan-qwen4/src/lib.rs` uses exactly this convention, and the kernel test packs
with it too). When `gsz` is a multiple of 8 a group never splits an 8-column run,
so one scale/bias pair covers the whole vector: 2 `dot` products plus a single
scale/bias application per 8 elements instead of 8 scalar chains.

**Correctness:** the four `logan-metal` differential tests
(`native_mlx_affine_gemv_matches_reference_for_all_supported_widths`,
`native_mlx_affine_multi_handles_mixed_bits_and_groups`,
`q4_fma_variant_matches_reference_and_baseline`,
`fused_gdn_accepts_mixed_mlx_affine_formats_and_group_sizes`) compare GPU output
against the repository's own reference decoder and all pass. The greedy token
trajectory is byte-identical.

**A/B:** `LOGAN_MLX4_SCALAR=1` compiles the pre-EXP-040 scalar loop instead, so
both arms come from one binary. The gate is a **shader-compile-time macro**
because MSL has no `getenv` and forbids function-scope `static`; an earlier
attempt to read the env inside the kernel broke shader compilation and silently
dropped the whole model to the CPU path (~19.8 s/token). The host reads the env
in `coli_metal_init` and injects `#define MLX4_SCALAR 1` into the shader source.

| measurement | result |
|---|---|
| one-binary A/B, 8 tokens | 504.19 -> 565.77 ms/token scalar = **1.12x** |
| canonical harness, quiet host | 2.1509 -> **2.6374** tok/s (**1.226x**) |

**Decision:** **KEPT, default ON.** Set `LOGAN_MLX4_SCALAR=1` to restore the
scalar branch.

**Note on the probe as an instrument:** its own run-to-run spread is +-60%, so it
is only usable for shape-level questions (does time scale with bytes) and not for
fine-grained kernel A/B. The real-model harness is the reliable instrument.

**Artifacts:** `logan-qwen4/examples/affine_dispatch_probe.rs` (with `PROBE_TOPK`).

---

## EXP-041 — Dense affine batching of attention QKV and shared gate/up: NEUTRAL

**Date:** 2026-09-23  
**Area:** dense path / Metal dispatch  
**Status:** **REJECTED (reverted)**

**Hypothesis:** The attention site (`QWEN_ATTN_FUSED_INPUT`) and the shared-expert
site (`QWEN_SHARED_FUSED_INPUT`) call only `matmul_mxfp4_multi`, which hard-requires
`WtBytes::Mxfp4`. On a raw MLX-affine checkpoint it therefore always declines, so
q/k/v and gate/up each pay their own `commit`+`waitUntilCompleted`. Adding the
`|| matmul_mlx_affine_multi(...)` fallback that the GDN input site has always had
should remove ~60 dispatch sites per forward (40 attention + 20 shared).

**A/B:** 5 interleaved pairs, sampled decode, 16 tokens, one binary via
`QWEN_ATTN_FUSED_INPUT=0 QWEN_SHARED_FUSED_INPUT=0`:

| arm | median ms/token | tok/s |
|---|---:|---:|
| on (affine fallback) | 509.12 | 1.9642 |
| off | 507.62 | 1.9700 |

**0.9970x — neutral.** Token-identical.

**Why:** dense projections are 10-20x larger than expert projections and are
partly bandwidth-bound, so the per-descriptor overhead the batch adds does not
buy the dispatch saving it bought on the ~590 KB expert matrices. (It also means
the dense-path dispatch count from the analysis was not the binding constraint it
was for the routed experts.) The change was reverted; the analysis's expectation
of ~28-40 ms/token was wrong and this entry supersedes it.

**Artifact:** `.perf_runs/autoresearch/ab-dense/`.

---

## EXP-042 — Vectorized 8-bit MLX affine branch

**Date:** 2026-09-23  
**Area:** Metal kernel / decode throughput  
**Status:** **KEPT**

Same ALU-bound defect as EXP-040, same fix. In the 8-bit case one bitstream word
*is* four consecutive codes, so a single `uchar4` load yields 4 elements with no
variable shift, no cross-word fixup and no per-element integer divide. Guarded on
`(gsz & 15) == 0` so one scale/bias pair covers the whole 4-element vector.

Coverage on this checkpoint: `embed_tokens` (8-bit/gs64), `lm_head` (8-bit/gs64)
and the shared expert's gate/up/down (8-bit/gs128).

**Correctness:** the four `logan-metal` differential tests still pass, including
the 8-bit/gs128 reference comparison; greedy trajectory byte-identical.

**A/B:** one binary via `LOGAN_MLX8_SCALAR=1` (shader-compile-time macro, same
mechanism as EXP-040).

| measurement | result |
|---|---|
| one-binary A/B, 8 tokens | 358.30 vs 375.99 ms/token = **1.05x** |
| canonical harness | 2.6374 -> **2.7060** tok/s |

`arm_rate_gap` was 0.0005 — the tightest matched-arm reading in this session.

**Decision:** **KEPT, default ON.** `LOGAN_MLX8_SCALAR=1` restores the scalar branch.

---

## EXP-043 — Vectorized 6-bit and 5-bit MLX affine branches (bit-width sweep complete)

**Date:** 2026-09-23  
**Area:** Metal kernel / decode throughput  
**Status:** **KEPT**

EXP-040 fixed the 4-bit branch and EXP-042 the 8-bit; the 5-bit and 6-bit branches
had the same ALU-bound shape (one element per lane iteration, with a variable
shift, a cross-word fixup and an integer divide per element). Both are now
chunked:

- **6-bit:** 16 codes = exactly 96 bits = 3 uint32 words, so a lane owns a
  16-column chunk (3 coalesced loads, one scale/bias gather, one divide).
  Covers the GDN input projections `[8192,2048]` and attention q/k — the largest
  dense matrices in the forward.
- **5-bit:** 32 codes = exactly 160 bits = 5 uint32 words, so a lane owns a
  32-column chunk. Covers `gdn_out_proj` and `attn_o_proj`.

Guarded on `gsz % 16 == 0` (6-bit) and `gsz % 32 == 0` (5-bit) so a chunk never
splits a group.

**A bug worth recording.** The first 6-bit implementation extracted codes 11..15
with `(lo >> bit) | (hi << (64 - bit))` for bit = 66..90, i.e. a 64-bit shift by
≥64 and by a *negative* amount — undefined behaviour that corrupted ~31% of every
6-bit matrix and produced visibly garbage tokens (`92565,92565,...`). The correct
form extracts each code from the half that contains it: `bit+6 <= 64` from `lo`,
`bit >= 64` from `hi` at offset `bit-64`, and only the single straddling code from
both. This is recorded because the wrong version still ran at plausible speed.

**Test sensitivity was proven with a negative control.** For each branch, a
deliberate corruption was introduced and the differential test was confirmed to
FAIL at the matching case (`bits=6 group=64`, and `bits=5 group=128`
respectively), which proves the test actually exercises that branch. The controls
were applied on a scratch copy, verified removed (`grep` count 0), and the tree
re-tested clean before any measurement.

**A/B (one binary per branch via `LOGAN_MLX6_SCALAR` / `LOGAN_MLX5_SCALAR`):**

| branch | off ms/token | on ms/token | speedup | canonical tok/s |
|---|---:|---:|---:|---:|
| 6-bit | 443.72 | 333.76 | **1.33x** | 2.7060 -> **3.0865** |
| 5-bit | 349.08 | 321.14 | **1.09x** | 3.0865 -> **3.2206** |

All four `logan-metal` differential tests pass with all four branches live, and
the greedy trajectory is byte-identical throughout.

**Decision:** **KEPT, default ON** for both. `LOGAN_MLX6_SCALAR=1` /
`LOGAN_MLX5_SCALAR=1` restore the scalar branches.

---

## EXP-044 — Attention QKV affine batching is a LOSS; shared-expert gate/up batching is a small win

**Date:** 2026-09-23  
**Area:** dense path / Metal dispatch  
**Status:** **REJECTED (attention) / KEPT (shared expert)**

Both dense fusion sites (`QWEN_ATTN_FUSED_INPUT`, `QWEN_SHARED_FUSED_INPUT`) call
only `matmul_mxfp4_multi`, which hard-requires `WtBytes::Mxfp4`. On a raw
MLX-affine checkpoint both always declined, so q/k/v and shared gate/up each paid
a separate `commit`+`waitUntilCompleted`.

EXP-041 tested the two together (bundled) and read 0.9970x neutral. Measured
**separately** after the kernel sweep, the two sites have opposite signs:

| site | off ms/token | on ms/token | ratio | verdict |
|---|---:|---:|---:|---|
| shared expert gate+up | 306.52 | 293.42 | **1.045x** | KEPT |
| attention q/k/v | 291.43 | 300.82 | **0.969x** | REJECTED |

So EXP-041's "neutral" was the two effects cancelling, not an absent effect. That
is the lesson worth keeping: a bundled A/B of two sites can read neutral while
each site individually has a real, opposite-signed effect.

**Why attention loses:** `attn_q` is `[8192,2048]` — 8x wider output than the
expert matrices — so batching three of them adds per-descriptor bookkeeping to a
dispatch that is already large enough to amortise its own overhead. The expert
matrices (~590 KB) are small enough that fixed per-dispatch overhead dominated,
which is exactly why batching worked there (EXP-037) and not here.

**Canonical:** 3.2206 -> **3.2973** with only the shared-expert site enabled.

---

## EXP-045 — Unrolled 6-bit/5-bit code extraction

**Date:** 2026-09-23  
**Area:** Metal kernel  
**Status:** **KEPT**

EXP-043's 6-bit/5-bit branches selected the source word with a per-element
conditional chain (`word == 0 ? w0 : (word == 1 ? w1 : ...)`) plus a runtime
straddle test. The straddle positions are fixed at compile time — only codes 5
and 10 of a 6-bit chunk, and 6, 12, 19, 25 of a 5-bit chunk, cross a word
boundary — so all codes can be extracted with straight shifts. That removes the
branch and lets the compiler vectorize the accumulate loop.

All four differential tests pass; greedy trajectory byte-identical.

| measurement | result |
|---|---|
| canonical harness | 3.2973 -> **3.3449** tok/s (1.014x) |

**Incidental findings from bringing this up:** both branches initially failed to
compile because the replacement left a duplicate `dot6`/`xs` (and `dot5`/`xs`)
declaration, and because the 6-bit head still bound `ww[0..2]` into `lo`/`hi`
while the new body referenced `w0/w1/w2`. Both are mechanical, but they are worth
recording because **a broken shader fails closed to the CPU path**: `metal_init`
reports `[metal] shader compile failed`, every kernel declines, and the model
still produces plausible output thousands of times slower. Any Metal kernel edit
should therefore be checked for a silent CPU fallback (the `mlx-affine:
metal=... fallback=...` profile line and a decode_ms sanity check) before its
speed is believed.

---

## EXP-046 — Expert prefetch re-test after the kernel sweep: still a loss

**Date:** 2026-09-23  
**Area:** routed MoE / RouteScout  
**Status:** **REJECTED (re-confirmed)**

EXP-031/EXP-036 rejected speculative expert prefetch on this host, with the stated
mechanism that the saved wait reappears as shared-memory/queue interference. Both
were measured when the MoE compute phase was ~1191 synchronous affine dispatches
per forward. EXP-037/038/040/042/043 cut that to two command buffers per layer and
made the kernel ~2.7x faster, so the compute/wait ratio changed by roughly 4x and
the rejection was worth re-testing rather than assuming.

5 interleaved pairs, 24 tokens, sampled decode, one binary, budget 1 with the
confidence gate disabled (to exercise the mechanism at all):

| arm | median ms/token | tok/s |
|---|---:|---:|
| off | 313.53 | 3.1895 |
| on (budget-1 prefetch) | 320.03 | 3.1247 |

**0.9797x — still a loss**, token-identical. The rejection therefore stands on the
new shape too, and this time the mechanism cannot be dispatch overhead. Prediction
quality was never the binding constraint (EXP-036 established that long-horizon
prediction is genuinely accurate); the cost is that speculative reads compete with
authoritative demand reads for the same UMA/SSD resources.

**Decision:** prefetch stays **OFF** (shipped default). Do not re-open without a
mechanism for isolating speculative traffic — e.g. an explicit bandwidth budget or
a priority split — rather than more prediction work.

---

## EXP-047 — MTLIO queue depth 64 -> 256

**Date:** 2026-09-23  
**Area:** MetalIO / expert streaming  
**Status:** **REVERTED (null, position-confounded)**

`MTLIO_DEPTH` sets `maxCommandBufferCount` on the MTLIO queue (default was 64).
Once the expert route is issued as one concurrent batch (EXP-039) a layer submits
8 reads at a time and consecutive layers can briefly overlap, so 64 is a plausible
ceiling on outstanding transfers.

**A/B (two independent 5-pair interleaved runs, one binary, `MTLIO_DEPTH=64` as
the off arm):**

| run | off ms/token | on ms/token | ratio |
|---|---:|---:|---:|
| 1 | 326.53 | 316.56 | **1.0315x** |
| 2 | 329.53 | 317.13 | **1.0391x** |

Both token-identical. Two independent paired runs agreeing on magnitude and sign
is the evidence; the canonical harness absolute (3.1187 vs 3.3449 on the previous
entry) is **not** comparable across runs — the same host drift moved the *greedy*
arm alone from 298.1 to 319.5 ms/token between those two measurements, which is
why this session relies on paired A/Bs for every accept/reject decision and on
the canonical harness only for direction over the whole segment.

**Note on the harness's own limitation:** the canonical `tok_per_sec` is a pooled
median over both arms of ONE invocation, so it carries whatever drift the host
had during that ~2 minute window. It is reliable for large effects and for the
segment trend, but a 2-4% effect is only trustworthy from an interleaved paired
A/B. This is recorded because it caused a false "regression" reading here.

**Decision:** **REVERTED to 64 — NULL RESULT, POSITION-CONFOUNDED.** Both A/Bs above
ran the candidate arm FIRST (`for arm in on off`), so the comparison is really
`first arm vs second arm`, not `256 vs 64`. Read by position, the two tests agree
with each other and contradict depth:

| test | first arm | second arm |
|---|---|---|
| 1 | 256 -> 316.56 ms | 64 -> 326.53 ms |
| 2 | 64 -> 317.13 ms | 256 -> 329.53 ms |

The first arm read 316.6 / 317.1 ms in the two tests (0.2% apart) and the second
arm 326.5 / 329.5 ms (0.9% apart) regardless of which depth value occupied it. So
the effect is arm position. Genuine evidence for depth would require the *same*
value to win in both orderings.

The mechanism also never supported the change: a layer issues 8 reads, and
profiling has never reported peak outstanding above 8, so a 64-deep queue was
never a ceiling. `MTLIO_DEPTH` is back at 64.

**Harness correction this exposed:** `/tmp/ab_env.sh` (used for several earlier
A/Bs in this segment) ran a fixed `on, off` order. It now alternates
(`on off` / `off on` per pair). Earlier candidates that used it and were KEPT on a
large margin — EXP-039 (1.2607 / 1.2915 from a *separately reversed-order* script),
EXP-040 (1.12), EXP-043 (1.33 / 1.09), EXP-044 (1.045) — were each checked with an
explicit reversal or via the canonical harness, and are unaffected. This entry is
the one that was not, and it is corrected here rather than left as a false win.

---

## EXP-048 — Removing the MetalIO copy hop is SLOWER: the copy releases the slot

**Date:** 2026-09-23  
**Area:** MetalIO / expert streaming  
**Status:** **REJECTED (instructive)**

**Hypothesis:** The routed-expert load path copies each fetched slot twice:
`mio_finish_slot` copies the slot into a fresh `Vec` (`to_vec`), and
`materialize_plan` then copies that `Vec` into the three owned matrices. At 320
expert fetches/token and ~1.6 MB each, that is ~512 MB of pure memcpy per token,
and it plausibly explains why the `load` term stayed far above the measured
MetalIO wait.

**Candidate:** a borrow-scoped `mio_finish_slot_with(slot, event, bytes, spec, f)`
that hands `f` a `&[u8]` over the slot and releases it when `f` returns, so
`materialize_plan` writes directly from the slot.

**Result — slower:**

| arm | ms/token |
|---|---:|
| copy hop (baseline) | 305.03 |
| direct from slot | 319.03 |

Tokens identical. **0.956x**, i.e. a ~4.6% regression.

**Why (the important part):** the slot must be released BEFORE materialization
runs. With `LOGAN_EXPERT_IO_CONCURRENCY` at its default of "issue the whole route"
(EXP-039), a layer submits all 8 reads and they are in flight together. Holding a
slot across the CPU-side materialization keeps it out of the reusable pool and
forces the next read to wait for a free slot, serializing what the concurrent
issue just parallelized. The "redundant" copy is what makes prompt release
possible: copying the bytes out and then materializing from the copy lets the
slot be recycled immediately.

**Decision:** **REJECTED.** The two-hop copy is retained, and the now-unused
borrow API was removed rather than left as dead code. A comment at the copy site
records why it is deliberate, so this is not "optimized away" later.

**Generalizable lesson:** on this engine, a copy that looks redundant on the CPU
side can be the mechanism that keeps a scarce asynchronous resource (a MetalIO
slot) available. Slot occupancy is the resource, not bytes moved.

---

## EXP-049 — Full-GPU GDN for fp16-affine weights: correct and FASTER

**Date:** 2026-09-23  
**Area:** dense GDN / Metal  
**Status:** **KEPT**

**Hypothesis (from the dense-path analysis):** the full-Metal GDN path
`coli_metal_gdn_mxfp4` encodes all five input projections, the conv + gated-delta
recurrence, the gated RMSNorm and the output projection into **one** command
buffer with no CPU synchronization. It was unreachable for this checkpoint only
because the Rust wrapper `logan_metal::gdn_mxfp4`'s format allow-list stopped at
`16..=20` while the C side already accepts 21..24 (the IEEE-fp16-sidecar affine
formats an FP16 checkpoint produces). That was replacing ~18.5 ms/token of scalar
CPU recurrence plus 2 dispatches per layer.

**Change:** widened the Rust allow-list and bit-width mapping to accept `21..=24`
(21..24 mirror 16..19 at 4/5/6/8 bits, with fp16 sidecars).

**Result — it works and it wins.**

- The path engaged exactly as predicted: `gdn_metal_ok=450` and every CPU
  `gdn_parts` sub-span (`in/conv/prep/recur/gate/out`) dropped to **0.0**, i.e.
  the scalar recurrence really was replaced by the GPU kernel.
- The profile span `gdn` fell 52.4 -> **31.8 ms/token**.
- **Tokens byte-identical**, which is a useful independent result: the GPU
  conv/recurrence/gated-RMSNorm kernel reproduces the scalar CPU loop's numerics
  for this checkpoint.

**Paired A/B** (5 pairs, alternating arm order, 24 tokens). NOTE on polarity:
this was run with `EXTRA_ON="QWEN_GDN_MXFP4_FULL=0"`, so the `on` arm is the FLAG
DISABLED (CPU) arm and the `off` arm is the default (GPU) path:

| arm | ms/token | tok/s |
|---|---:|---:|
| `on` = `QWEN_GDN_MXFP4_FULL=0` = **CPU GDN** | 317.77 | 3.1470 |
| `off` = default = **GPU GDN** | 295.77 | 3.3810 |

`speedup = off/on = 0.9308` in the script's convention (">1 means ON faster"), so
<1 means the CPU arm is slower: **GPU GDN wins 295.77 vs 317.77 = +7.4%.** The
first misread of this table inverted the arms and wrote a withdrawal into this
entry; it is corrected here, and the independent evidence agrees (the GPU arm's
`gdn` span is 31.8 vs 52.4, and an earlier GPU-first screen read 296.5 vs 307.7).

**Decision:** **KEPT.** The Rust allow-list widening stays.

**Process note (the real cost of this experiment):** the A/B driver puts the
candidate on the `on` arm via `EXTRA_ON`, so when the candidate is expressed as
*disabling* a default-on feature, the `on` arm is the control. That inversion
produced a wrong KEEP/REJECT decision and needed a second correction pass. Future
A/Bs of a default-on feature should express the candidate directly
(e.g. `EXTRA_ON="..."` enabling something) or be labelled explicitly.

---

## EXP-050 — Vectorized kernels reproduce the scalar path over 128 tokens; MoE compute cost not yet explained

**Date:** 2026-09-23  
**Area:** kernel correctness / MoE compute ceiling  
**Status:** **KEPT (verification)**

### A. Long-run kernel equivalence

The EXP-040/042/043/045 vectorized branches were each gated on the
`logan-metal` differential tests (which compare a single 3-row GEMV against the
repository's own reference decoder) plus the 24-token trajectory. Both are short.
This is the stronger gate: **128 greedy tokens** decoding with all four vectorized
branches active versus with all four forced to their scalar implementations
(`LOGAN_MLX4_SCALAR=1 LOGAN_MLX5_SCALAR=1 LOGAN_MLX6_SCALAR=1 LOGAN_MLX8_SCALAR=1`,
i.e. the pre-EXP-040 kernel in every width).

Result: the two 128-token token streams are **byte-identical** (595 bytes each,
`cmp` clean). Greedy autoregressive decoding is maximally sensitive to any
numerical difference — a single differing logit flips a token and every later
token diverges — so equivalence over 128 steps is strong evidence that the
vectorized decoding reproduces the scalar element order exactly for every width
present in this checkpoint.

### B. MoE compute cost after the vectorization — and what the probe does NOT show

`affine_dispatch_probe` was updated to the real two-command-buffer shape (gate/up
in one shared-activation batch, all `k` downs in one per-descriptor-activation
batch, EXP-038). Measured **~1301 us/layer** (median of 5, `bit_identical=true`),
down from 6683 us at the start of the session.

**Correction (this entry originally claimed the phase is "bandwidth-bound at
~90 GB/s" — that arithmetic was wrong and the claim is withdrawn).** Per layer the
expert weights moved are 8 experts x (2 x 512x2048 + 2048x512) at 4-bit ~= 12 MB,
so 12 MB / 1301 us is **~10 GB/s**, an order of magnitude *below* this host's UMA
bandwidth, not at it. So the probe result does **not** establish that the phase is
bandwidth-bound.

What the probe actually shows is ambiguous between two explanations, and the cheap
discriminator is stated here rather than a closure claim:

- **overhead/ALU-bound:** time flat as the expert count grows (fixed per-dispatch
  step dominates), or
- **bandwidth-bound:** time scaling with bytes.

The 1/2/4/8-expert sweep already run in EXP-040 (1.6 / 2.4 / 2.2 / 2.1 GB/s on
the *scalar* kernel) showed scaling with bytes at a constant rate, which under the
old kernel was ALU-bound. The same sweep has **not** been re-run against the
vectorized kernel, so the post-vectorization shape is unmeasured. That sweep is
the discriminating experiment and it is cheap.

Additionally: `compute_ms_per_token` ~77-89 ms for ~2.0 GFLOP/token of expert GEMM
is ~25 GFLOPS against ~2.6 TFLOPS available. That ~100x gap is **not** explained by
either explanation above, and a plausible unmeasured contributor is the
per-matrix weight upload — `materialize_plan` hands each of ~960 matrices/token a
fresh unaligned `Vec<u8>` with a null `metal_tensor`, so `wrap()` takes its
copying `newBufferWithBytes` path (it zero-copies only for a 16 KiB-aligned,
page-rounded pointer). The existing probe **cannot see this**, because it caches
`ts[]` across iterations so `wrap()` runs zero times after warmup. A probe arm
that resets `ts[m] = null` each iteration would isolate it.

**Decision:** the kernel-sweep work (EXP-040/042/043/045) is **kept and
verified**, and the *dispatch-batching* part has nothing left at 2 command
buffers/layer. But the branch is **NOT closed**: the two candidate explanations
above (vectorized 1/2/4/8 sweep, and per-matrix `wrap()` upload cost) are both
unmeasured, and one of them plausibly accounts for ~100x of arithmetic-to-wall
mismatch. Recorded explicitly so a future session measures rather than assumes.

---

## EXP-051 — Expert residency cache on raw MLX: REJECTED (reuse is real at ~53%, residency still loses)

**Date:** 2026-09-23  
**Area:** routed MoE / storage  
**Status:** **REJECTED** (mechanism identified)

**Why re-test:** EXP-019 rejected an expert LRU on this path (every capacity
regressed). That was measured when the path was dispatch-bound (~1191 synchronous
affine dispatches per forward made the `load` term irrelevant). EXP-037/038 removed
that and EXP-040..045 made the kernels ~2.7x faster, so `wait` (~74 ms/token)
became the largest single term. The tradeoff EXP-019 measured no longer held in
the same form, so it was re-measured rather than assumed.

**Implementation:** a `(layer, expert)` -> (raw MetalIO bytes, I/O plan) cache in
`MlxLocalExpertSource`. Two robustness points were handled explicitly:
1. Residency is resolved **before** the fetch-issue loop, so only misses are
   issued. Doing it after would leave a pre-issued `DemandFetch::Pending`
   uncollected on a hit, and since `DemandFetch` has no `Drop` its MetalIO slot
   would never be freed.
2. Any issued-but-superseded fetch is explicitly `mio_discard_slot`-ed.

The cache also **moves** `raw` in after materializing (rather than cloning), so a
miss costs only a hashmap insert. An earlier attempt cloned and paid ~1.6 MB per
miss (~512 MB/token), which is a real cost this design avoids.

**The measurement that mattered — the hit rate.** Instrumenting
`(resident_hits, resident_misses)` (printed in the `mlx-expert-load` profile line)
turned out to be the decisive step, because tok/s alone cannot distinguish
"hit 60% and gained nothing" from "hit 3% and gained nothing":

| cap (slots) | resident hits | resident misses | hit rate | ~cache bytes |
|---:|---:|---:|---:|---:|
| 256 | 0 | 16 000 | **0%** | 0.4 GB |
| 1024 (48 tokens) | 12 517 | 11 163 | **52.9%** | 1.6 GB |

**Capacity below one token's working set never fires at all.** A single token
touches 40 layers x 8 experts = **320 distinct** `(layer, expert)` keys, so a
256-slot cache is evicted before any key can repeat. Any capacity < 320 is
structurally 0% hit rate — and the first attempt at this experiment screened caps
of 64/320/512, i.e. all at or below that threshold, which is why it produced a
spurious "no capacity helps" result.

**Paired A/B at a capacity that does fire** (cap 1280, 5 pairs, alternating arm
order, 32 tokens, ~53% hit rate):

| arm | median ms/token | tok/s |
|---|---:|---:|
| cache off | 271.75 | 3.6798 |
| cache on (1280 slots) | 325.42 | 3.0729 |

**0.8351x — 16.5% SLOWER**, token-identical.

**Mechanism.** The measured ~53% hit rate *is* genuine temporal same-layer reuse:
the cache is keyed by `(layer, expert)`, so a hit requires the same layer and the
same expert on a different token — cross-layer reuse cannot produce a hit at all
(different layers are different keys). An earlier version of this entry claimed
the 53% was "diluted cross-layer reuse" and cited EXP-011's ~4.5% *temporal*
figure; those two statements cannot both be true, and the measurement wins. The
reuse is real.

What is also real is that eliminating ~53% of the reads made things **worse**:
`wait_ms_per_token` rose 73.2 -> 93.3 and `load_ms_per_token` rose 99.3 -> 120.4
against the cache-off baseline, and the end-to-end result was 16.5% slower. So the
~2 GB of resident raw bytes (plus their transient materialized MTLBuffers) actively
degraded the memory system on this 16 GiB host — memory compression/swap, which is
why both I/O-facing terms rose *despite* fewer reads. That mechanism matches
EXP-019's and EXP-032's repeated finding, and it is a stronger result than "the
reads were not overlapped".

A caveat on the two rejected attempts at this: at cap 256 the cache reported **0%
hits**, and the per-token working set is 40 layers x 8 experts = **320 distinct
keys**, so any capacity below ~320 evicts the entire route before the next token
can re-request it. The earlier screen at caps 64/320/512 therefore measured pure
cache overhead, not cache value — which is why it produced a spurious "no capacity
helps" reading.

**Decision:** **REJECTED and reverted.** Two-times-confirmed (EXP-019, EXP-051),
now with the axis error identified, and the capacity/working-set constraint
recorded: any future expert cache must (a) target temporal reuse, which is ~5% on
this model, and (b) hold at least one full token's 320 keys before it can fire at
all. Treat "cache more experts" as closed for this workload, not as unexplored
headroom.

---

## EXP-052 — Per-matrix Metal weight upload: HYPOTHESIS REFUTED by exact counters

**Date:** 2026-09-23  
**Area:** Metal weight upload / expert path  
**Status:** **REFUTED — branch closed**

**Motivation:** `compute_ms_per_token` read ~77-89 ms for ~2.0 GFLOP/token of expert
GEMM, i.e. ~25 GFLOPS against ~2.6 TFLOPS available — a ~100x gap neither dispatch
overhead nor arithmetic explained.

**Hypothesis (now refuted):** the gap is per-matrix weight **copy**. `wrap()`
zero-copies only when the pointer is 16 KiB-aligned AND the length is page-rounded;
`materialize_plan` builds a fresh `Vec<u8>` per matrix, so if those were unaligned
`wrap()` would take its `newBufferWithBytes` path — projecting ~566 MB/token of
copies. A timing probe appeared to support this: a probe arm that resets
`ts[m] = null` every iteration (matching what a fresh `Wt` does) measured
1182 -> 1577 us/layer, ~1.33x.

**The timing probe was the wrong instrument, and the counter refutes the
hypothesis.** Exact `wrap()` accounting was added to `backend_metal.mm`
(`coli_metal_wrap_stats`: call count, zero-copy count, copied bytes) and read after
real `decode_bench` runs:

| decode tokens | wrap calls | zero-copy calls | copied bytes (whole run) |
|---:|---:|---:|---:|
| 8 | 65 982 | 65 842 | 332 800 |
| 24 | 96 702 | 96 562 | 332 800 |
| 40 | 127 422 | 127 282 | 332 800 |

`copied_bytes` is **constant at 332 800 bytes (0.33 MB) for the entire run** while
the call count scales exactly with decode length: (127422 - 65982) / 32 =
**1920 `wrap()` calls per decode forward**, of which essentially all are
zero-copy. So:

- The expert weight pointers **are** already 16 KiB-aligned. macOS malloc mmaps
  allocations of this size, and the expert shapes' `fmt_bytes` are already
  multiples of 16384, so both of `wrap()`'s conditions are met and the memcpy
  never happens.
- The projected ~566 MB/token of copies **does not exist**. The `~1.33x` "upload
  overhead" the probe measured is therefore **MTLBuffer object creation and
  registration** — 1920 fresh buffer objects per forward (320 experts x 3 matrices
  x 2 buffers: weights + aux) — not data movement.

**Why this closes the branch rather than deferring it:** the only design that
avoids creating those buffers is keeping the materialized `Wt`s (and their warm
`metal_tensor` handles) resident across tokens. That is **EXP-019**, which
implemented exactly that (an `ExpertStore<CachedMlxExpert>` holding the three `Wt`s
behind an `Arc` "so a hit reuses the already-created Metal tensor instead of
rebuilding it"), swept it at N in {0,8,16,32}, showed the load term falling as
designed — and measured end-to-end decode getting *worse* at every capacity
(1586 -> 1642 ms/token) because of UMA pressure on this 16 GiB host. EXP-051
re-confirmed the same outcome with modern methodology (53% hit rate, still 16.5%
slower). So the buffer-creation cost is a *residency* question, and residency is
already three-times rejected here.

**Method note worth carrying:** a timing probe cannot separate a memcpy from
buffer-creation churn at this host's noise level (the same statistic printed
-17.74 us on one run — physically impossible). Exact counters settle this class of
question in ~15 lines with zero noise; reach for them before designing a
memory-ownership change on top of probe timing. The counters are kept in the tree
and are reported by `LOGAN_PROFILE=1` as `logan metal-wrap: calls=.. zero_copy=..
copied_bytes=..`.

**Correction to this session's earlier reasoning:** EXP-052 as originally written
recommended "pooling page-aligned destination buffers". That recommendation is
withdrawn — there is nothing to align.

---

## EXP-053 — Fused affine shared expert (gate/up + GPU SwiGLU + down in one command buffer)

**Date:** 2026-09-23  
**Area:** dense shared expert / Metal  
**Status:** **KEPT**

**Motivation:** `shared` measured 27-29 ms/token for only ~6 MFLOP/token of real
work — essentially pure per-dispatch overhead, because the shared expert ran as
three separate `commit`+`waitUntilCompleted` dispatches per layer (120 command
buffers/forward). `coli_metal_shared_mxfp4` already encodes gate_proj, up_proj,
**SwiGLU** and down_proj as three encoders inside **one** command buffer (so the
intermediate never returns to the host), and both it and its helpers
`qwen_gdn_mx_tensor` / `qwen_gdn_mx_encode_gemv` already accept fmt `21..24`
(MLX affine with fp16 sidecars) — the same formats EXP-049 enabled for GDN.

**Blocker:** two format gates stopped at MXFP4 (`descs[i].fmt != 7` in C,
`dsc.fmt != 7` + `fmt: 7` in the Rust wrapper), so the fused path declined every
time on this checkpoint and fell through to three per-matrix dispatches.

**Change:** widened both gates to accept `21..24`, and extended the engine's
`full_mxfp4` gate to build affine descriptors (with `metal_aux` supplying the
interleaved scales+biases layout the affine GEMV expects).

**The one real hazard, handled explicitly:** the Rust wrapper's byte-count guard
computed **MXFP4** sizes (`o * ceil(i/2)` weights, one scale byte per 32). Those
are far smaller than the affine sizes, so reusing them would have let
`weights.len() < weight_bytes` pass trivially while the C side read using its own
larger `fmt_bytes` stride — an out-of-bounds read. The guard now branches by
format and computes affine sizes (`o * ceil(i*bits/8)` weights, `2 * o *
ceil(i/gs) * sizeof(u16)` scales) for `21..24`.

**Result:** `shared` span 28.4 -> **21.1 ms/token**, tokens byte-identical.

**Paired A/B** (5 pairs, alternating arm order, 32 tokens). NOTE on polarity: run
with `EXTRA_ON="QWEN_SHARED_MXFP4_FULL=0"`, so the `on` arm **disables** the
fusion and `off` is the default fused path:

| arm | median ms/token | tok/s |
|---|---:|---:|
| `on` = `QWEN_SHARED_MXFP4_FULL=0` = **unfused** | 293.19 | 3.4107 |
| `off` = default = **fused** | 274.36 | 3.6448 |

**+6.9%** for the fused path. Canonical harness: 3.4715 -> **3.5256**
(**1.99x** over the 1.7725 baseline), `arm_rate_gap` 0.0013.

**Decision:** **KEPT, default ON.** `QWEN_SHARED_MXFP4_FULL=0` restores the
unfused path.

---

## EXP-054 — Long-horizon correctness gate for the fused paths; wrap fix deliberately deferred

**Date:** 2026-09-23  
**Area:** verification / Metal weight upload  
**Status:** **GATE PASSED** (EXP-053) / **wrap fix SUPERSEDED by EXP-052 (refuted)**

### A. 96-token equivalence for the fused shared expert

The fused shared-expert path (EXP-053) moves the SwiGLU onto the GPU and runs the
whole expert in one command buffer, so it is the kind of change that could
perturb numerics without flipping an early argmax. It was therefore gated the same
way the vectorized kernels were (EXP-050): a **96-token greedy decode** with the
fusion on versus off (`QWEN_SHARED_MXFP4_FULL=0`).

Result: the two 96-token streams are **byte-identical** (450 bytes each, `cmp`
clean). Combined with EXP-050's 128-token kernel-equivalence result, the two
session changes that alter GPU arithmetic both reproduce the reference trajectory
over a horizon far longer than the 24-token canonical gate — ~4x more tokens, and
therefore ~4x more distinct expert routes exercised per layer.

### B. Per-matrix `wrap()` upload: measured, but the fix is deferred on risk grounds

EXP-052 measured the per-matrix Metal weight upload at **+395 us/layer = 1.334x**
on the MoE compute phase (**~15.8 ms/token**, ~5.6% of the current ~280 ms
forward). The cause is established: `materialize_plan` hands each of ~960
matrices/token a fresh unaligned `Vec<u8>` with a null `metal_tensor`, and
`wrap()` zero-copies only for a 16 KiB-aligned pointer with a page-rounded length,
so it takes the copying `newBufferWithBytes` path.

**Why it is not being attempted now.** Two candidate mechanisms were investigated
and both are riskier than their ~5.6% prize on an unattended run:

1. **Registered slabs.** `coli_metal_register`/`resolve()` is the documented
   zero-copy mechanism, but `resolve()` only succeeds for pointers inside a
   *registered* slab, and `materialize_plan` copies into `Wt`-owned `Vec`s that
   are then freed — so registration alone does nothing. The bytes would have to be
   borrowed from a stable aligned buffer, i.e. an ownership change to
   `WtBytes::MlxAffine` (it currently owns `Vec<u8>` for weights/scales/biases and
   is read as `&[u8]` by both the matmul call sites and the multi path).
2. **A 16 KiB-aligned `Vec`.** Allocating with `posix_memalign` and wrapping via
   `Vec::from_raw_parts` would satisfy `wrap()`'s alignment test while keeping the
   type — but such a `Vec` must be deallocated with the *matching* layout, and a
   default-drop `Vec<u8>` would free it with `Layout::array::<u8>(len)` at align
   1. That is undefined behaviour, i.e. a silent-corruption class of bug, not a
   performance one.

Either route is a substantial refactor of a memory-ownership boundary that the
engine reads through raw pointers into Metal buffers, and this repository's own
code comments warn that stale pointer-keyed handles serve *wrong weights*. Since
EXP-052 already records the measurement, the prize, and both mechanisms including
the alignment/dealloc trap, a future session can implement it deliberately rather
than an unattended loop attempting it.

**Decision:** measurement **KEPT** as a finding; the fix is **superseded by
EXP-052, which REFUTED the underlying hypothesis.** Exact `wrap()` counters added
afterwards show the expert weight pointers are already 16 KiB-aligned:
`copied_bytes` stays flat at 332 800 for an entire run while the call count reaches
**1920 `wrap()` calls per forward** (320 experts x 3 matrices x 2 buffers), and
essentially all of them take the zero-copy path. So there is no ~566 MB/token of
copies to remove, the earlier recommendation to "pool page-aligned destination
buffers" is withdrawn, and what remains is MTLBuffer *object creation* — whose only
fix is residency, already rejected three times on this host (EXP-019 sweep, EXP-051
at a real 53% hit rate).

The ownership question raised in this entry was therefore never necessary, which is
the useful outcome: the exact counter closed the branch for free and prevented an
unaligned-`Vec`-dealloc risk being taken for a prize that did not exist.

---

## EXP-055 — The 24-token sample arm replays argmax; non-greedy validation needs >=33 tokens

**Date:** 2026-09-23  
**Area:** harness / methodology  
**Status:** **MEASURED (harness caveat recorded)**

**Why this matters:** the harness's second arm is specified as a seeded
temperature-1.0 multinomial sample over the full vocabulary, and the user
explicitly asked for validation under non-greedy decoding. If that arm silently
replays the argmax path, the harness is measuring one trajectory twice and the
non-greedy claim is unsupported.

**Measurement** (same prompt, same seed `20260923`, `BENCH_TEMP=1.0`,
`BENCH_TOP_K=0`, `BENCH_TOP_P=1.0`):

| horizon | greedy vs sample | first divergence |
|---|---|---|
| 24 tokens | **byte-identical** | none |
| 64 tokens | **diverge** | token **33** (greedy 34080 vs sample 1536) |

So the sampler is live (it does diverge), but the prompt's ` thinking` preamble makes
the distribution peaked enough that the first 32 draws all land on the argmax. The
canonical 24-token run therefore **cannot** distinguish the arms, which is what the
near-zero `arm_rate_gap` (0.001-0.008) in every canonical run this session was
saying.

**Measured at a horizon where the arms genuinely differ** (`autoresearch.sh
--tokens 64 --repeats 2`):

| arm | pooled median ms/token | tok/s |
|---|---:|---:|
| greedy | 309.90 | 3.2268 |
| sample (temp 1.0, diverged at token 33) | 304.12 | **3.2882** |
| combined `tok_per_sec` | | **3.2575** |

Both arms are fast and within 2% of each other, so the 2.03x over the baseline is
**not** an artifact of measuring the greedy path twice: the sampled arm, decoding a
genuinely different token sequence (hence different expert routes), retains the
full speedup.

**Caveat stated plainly:** the 3.2575 at 64 tokens is not directly comparable to
the canonical 24-token 3.6063, because attention cost grows with context length
(10 full-attention layers) and the 64-token greedy `trajectory_sha` is necessarily
different from the canonical 24-token `e4f361a8…`. The baseline commit was not
re-measured at 64 tokens, so the 64-token figure is reported as *both arms fast and
close*, not as a second independent speedup ratio. The structural argument that the
speedup is horizon-independent: every kept change is a **token-independent
forward-path** change (command-buffer count, kernel vectorization, weight-wrapper
creation, GDN execution placement). The token trajectory only selects *which*
experts load — it cannot change the per-forward cost structure.

**Decision:** the canonical 24-token harness is retained for comparability and
speed, with this caveat recorded. Any claim about non-greedy decoding should cite
the 64-token numbers above, not the canonical run.

---

## EXP-056 — `COLI_METAL_UNTRACKED=1` is null; the buffer-creation branch is closed

**Date:** 2026-09-23  
**Area:** Metal resource options  
**Status:** **NULL (not promoted)**

**Motivation:** EXP-052's exact counters isolated the remaining expert-phase cost as
**MTLBuffer object creation** (1920 fresh buffer objects per forward, essentially
all zero-copy, `copied_bytes` flat at 332 800 for a whole run). MTLResource hazard
tracking is per-object overhead, and `coli_metal_init` already supports disabling it
(`COLI_METAL_UNTRACKED=1` -> `MTLResourceHazardTrackingModeUntracked` on all
resources). That is the cheapest available test of whether per-object overhead is
actually recoverable without changing residency.

**Paired A/B** (5 pairs, alternating arm order, 24 tokens, one binary):

| arm | median ms/token | tok/s |
|---|---:|---:|
| untracked | 281.52 | 3.5521 |
| default (tracked) | 283.73 | 3.5244 |

**1.0078x — null** (0.8%, well inside this host's noise band), token-identical. So
hazard tracking is not a material part of the per-object cost here.

**Decision:** **NOT PROMOTED.** `COLI_METAL_UNTRACKED` stays opt-in as before.

**Why this closes the branch.** With (a) the per-forward cost measured as buffer
*object creation* rather than copies, (b) hazard-tracking mode null, and (c) the
only way to avoid creating those objects — keeping materialized `Wt`s and their warm
handles resident — already rejected on this host three times (EXP-019's sweep,
EXP-051 at a genuine 53% hit rate, both losing to UMA pressure on 16 GiB), there is
no remaining contained lever on the expert phase at the 1-2% scale. The alternative
that would raise the arithmetic-side ceiling is a GPU-side SwiGLU fusion to collapse
the MoE phase from two command buffers per layer to one (~1.5%, requires a new C
API and an FP-accumulation-order risk) — measured as poor risk/reward against a
verified 2.0x.

**Stopping point:** this session's structural work is complete. All six kept changes
are token-verified (baseline sha, 128-token kernel equivalence, 96-token fused
equivalence), all four rejected branches carry a measured mechanism, and the ~2.0x
result is confirmed at a horizon where the non-greedy arm genuinely diverges
(EXP-055).

---
