# RouteScout SSD-Streaming Optimization — Agent Handoff

**Created:** 2026-09-22  
**Repository:** `~/CODE/logan`  
**Primary model:** `~/models/Qwen3.6-35B-A3B-MLX-oQ4-FP16`  
**Experiment ledger:** `EXPERIMENTS.md`  
**Current completed experiment:** EXP-028  
**Primary runtime source for this slice:** raw MLX + safetensors

## Mission

Continue RouteScout as a **correctness-neutral SSD-streaming optimization** for Logan.

The immediate target is the real raw-MLX Qwen3.6 checkpoint running with routed experts streamed from the Mac SSD while dense/static weights remain resident.

The authoritative model router always decides the actual experts. RouteScout may predict, prefetch, rank, stage, batch, or schedule future expert work, but a wrong prediction must never change model numerics.

The goal for this slice is:

> Turn the already-working raw-MLX MetalIO prefetch mechanism into a repeatable end-to-end decode win by improving measurement integrity, prediction policy, lead time, and I/O efficiency.

Do not stop at better Recall@K. The promotion gate is **correctness-preserving end-to-end performance**.

---

## Scope / format policy

### MLX + safetensors are first-class

Raw MLX/safetensors is **not** an import-only path and does not need to be converted to another Logan package format before optimization.

For this slice:

- use the raw Qwen3.6 MLX/safetensors checkpoint directly,
- improve the source-neutral `ExpertSource` / MetalIO path,
- preserve raw MLX as a first-class runtime source,
- do not prioritize `.logan` format work,
- do not regress or delete existing `.logan` work,
- do not make RouteScout depend on COLI or `.logan` metadata.

MLX/safetensors and native Logan artifacts should remain peers at the runtime source boundary.

### COLI

COLI is legacy compatibility. Do not build new RouteScout behavior around `self.coli`.

The structural blocker from EXP-027 has already been removed.

---

## oh-my-pi mode guidance

### Use `/goal` as the primary mode

This task is an implementation + experiment program with clear acceptance criteria.

Recommended prompt:

```text
Read AGENTS.md, EXPERIMENTS.md, and docs/handoffs/ROUTESCOUT_SSD_STREAMING_HANDOFF.md in ~/CODE/logan.

Execute the RouteScout handoff autonomously. Preserve the dirty working tree. Use raw MLX/safetensors as the primary Qwen3.6 source. Do not prioritize .logan format work. Record every non-trivial performance/architecture experiment in EXPERIMENTS.md before or alongside the run. Fix measurement integrity first, then optimize expected stall avoided, prefetch lead time, and MetalIO efficiency. Native routing must remain authoritative and every promoted path must be token-identical.
```

### `/loop`

Useful once a candidate exists for repeated:

```text
build -> run paired A/B -> parse metrics -> inspect -> adjust -> rerun
```

Do not let Loop silently change the hypothesis between runs. Each material hypothesis needs its own EXP entry.

### `/autoresearch`

Use only for a narrow unresolved question, such as:

- a specific learned-routing paper/algorithm that could improve cross-prompt cold-arrival prediction,
- Apple MetalIO queue/coalescing semantics not resolvable from SDK/repo code,
- statistical methodology for a noisy paired benchmark.

Do not use autoresearch as the main mode. We already have enough evidence to continue experimentally.

### Avoid `/vibe`

This is performance work on a dirty repo. Explicit hypotheses and A/B gates are more useful.

---

## Repository safety

Read `AGENTS.md` first.

The working tree contains valuable uncommitted work from multiple Logan threads, including:

- RouteScout,
- Qwen3.8 / GGUF,
- Metal kernels,
- ANE,
- compiler/format work.

Never run destructive cleanup such as:

- `git reset --hard`
- `git clean`
- broad checkout/restore
- mass reformatting unrelated files

Inspect diffs before editing already-modified files.

If committing, stage only intentional files.

Local run artifacts under `.perf_runs/` are intentionally Git-ignored but should be preserved when they contain experiment evidence.

---

# Current verified state

## Target model

`deepsweet/Qwen3.6-35B-A3B-MLX-oQ4-FP16`

Local path:

```text
~/models/Qwen3.6-35B-A3B-MLX-oQ4-FP16
```

Geometry used by the current runtime work:

- 40 layers
- hidden size 2048
- 256 routed experts
- top-8
- MoE intermediate 512

This checkpoint is the primary systems target for this slice.

---

## Raw MLX SSD-only mode works

Canonical flags:

```text
LOGAN_EXPERT_NOCACHE=1
LOGAN_EXPERT_METALIO=1
LOGAN_EXPERT_PREFETCH_SLOTS=N
```

`LOGAN_EXPERT_NOCACHE=1`:

- reopens only routed-expert safetensors shard descriptors,
- applies macOS `F_NOCACHE` to those expert-only descriptors,
- leaves dense/static loading normally cached,
- automatically enables MetalIO when available.

The older `QWEN_MLX_*` spellings remain compatibility aliases only.

A real Qwen3.6 smoke gate with the canonical flag produced:

```text
nocache=true
metalio=true
generated: [348, 10]
MetalIO loads: 2,880
MetalIO bytes: 5,096,079,360
failures: 0
```

So the SSD-only path is real and token-correct.

---

## Source-neutral speculative prefetch works

`ExpertSource` now has optional prefetch capability.

Raw MLX/safetensors can stage selected experts via arbitrary-range MetalIO without going through COLI.

MetalIO supports ranges from multiple source files in one request; the corresponding regression test passes.

The raw MLX source maintains bounded speculative slots and consumes a prefetched expert if authoritative routing later requests the same `(layer, expert)`.

---

## EXP-028 result

With Qwen3.6, SSD-only raw MLX, RouteScout enabled, budget 8, and the normal confidence gate disabled purely for mechanism qualification:

```text
speculative MetalIO loads: 8,853
used:                       1,448
wasted:                     7,374
ready at demand:            1,448
late at demand:                 0

route precision:            ~0.164
route recall:               ~0.413
```

The generated 24-token sequence matched baseline exactly:

```text
[348, 10, 4838, 1665, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
 25, 33898, 2110, 30, 31, 73307, 58, 3312, 87197, 62]
```

Single qualification timings:

- SSD-only with normal confidence gate: ~833.3 ms/decode-forward
- forced low-precision budget-8 speculation: ~874.9 ms/decode-forward

Do not over-interpret those single runs. The important result is:

> The transport mechanism works. Low-precision over-prefetching wastes substantial I/O and did not win.

The default confidence gate was correct to suppress this candidate.

---

# Critical measurement issue — fix this first

Before making any RouteScout performance claim, fix the surviving profiling-normalization bug.

## Problem

The runner does prompt prefill, then one final prompt forward, then measures only subsequent decode forwards:

```text
prefill_token(...) x prompt_len-1
forward_token(last_prompt)       <- not in forward_ms
forward_token(generated token)   <- measured
...
```

But several global/model counters accumulate across **all** forwards, while `profile_summary()` divides by only `forward_ms.len()`.

Affected or potentially affected metrics include:

- `mlx_affine_dispatch_counts()`
- `mlx_expert_source_timings()`
- model `self.spans`
- MetalIO stats
- RouteScout lifetime counters if interpreted as decode-only

Example from earlier qualification:

For prompt length 8 and `QWEN_MAX_NEW=24`:

- 7 prefill forwards
- 1 final-prompt forward
- 23 measured decode forwards
- 31 total model forwards

With 40 layers x top-8:

```text
31 * 40 * 8 = 9,920 expert calls
```

That exact total appeared in profiling, but it was divided by 23, yielding a false ~431 calls/token instead of the structural 320 calls/model-forward.

Do **not** fix this by multiplying old numbers by 23/31. Prefill and decode can have different costs.

## Required fix

Create a decode-boundary snapshot/delta mechanism.

The comparison boundary for metrics aligned to `forward_ms` should be:

```text
after:
    model.forward_token(last_prompt, ...)

before:
    the generated-token decode loop
```

Snapshot or reset the relevant counters there, then report decode deltas.

Prefer snapshots/deltas over global reset if resetting could affect another consumer.

### Measurement acceptance check

For raw Qwen3.6 expert-source decode with 23 measured forwards, expert calls should be structurally consistent with:

```text
23 * 40 * 8 = 7,360
```

unless an explicitly documented runtime path changes the number of authoritative expert evaluations.

Record this as the next experiment/measurement-integrity entry. At handoff creation the next free number appears to be **EXP-029**, but scan `EXPERIMENTS.md` before allocating.

---

# Current predictor

`logan-qwen4/src/route_predictor.rs`

The online predictor maintains, per layer:

- temporal expert -> expert transitions,
- spatial expert -> expert transitions,
- source observation counts,
- decay,
- pending prediction metrics.

It predicts **cold arrivals** only: experts absent from the previous same-layer route.

Current scoring effectively sums conditional temporal and spatial transition evidence.

Cold start does not speculate.

Current tests cover:

- cold start,
- learned cold arrival,
- conditional normalization,
- excluding previous resident experts,
- layer locality/bounds,
- spatial signal,
- invalid IDs.

Native routing remains authoritative.

---

# Historical predictor evidence worth preserving

Cross-prompt leave-one-out results from earlier RouteScout work:

```text
Predictor             R@8      R@16     R@24    cold8 recall
frequency             .2885    .4244    .5059       .2439
temporal              .4046    .5536    .6217       .3967
spatial               .4823    .6340    .6932       .4763
hybrid                .4721    .6271    .6893       .4787
```

Important conclusion:

> Spatial transition is the strongest simple single prior on this Qwen3.6 corpus.

Additional findings:

- Temporal history deeper than 1 did not improve the fused predictor reliably.
- Prompt-local adaptation needed too much warmup and was prompt-dependent.
- The candidate-centric learned scorer only added small residual gains.
- ANE scorer execution is cheap enough; prediction quality/policy and I/O value are the problem.

Do not resurrect rejected variants unchanged without a new hypothesis.

---

# Primary objective

Optimize **expected stall avoided**, not classifier accuracy.

A useful candidate is roughly:

```text
P(authoritative demand before expiry)
* stall_ms_that_can_be_hidden
- speculative_read_cost
- queue_contention_cost
- residency/eviction_cost
- predictor/scheduling overhead
```

For the current raw-MLX no-residency-cache SSD mode, eviction cost may be low, but:

- SSD bandwidth is finite,
- MetalIO slots/queue depth are finite,
- wrong predictions consume real bytes and command overhead.

A 16% precision predictor can still be useful if reads are very cheap and highly overlapped, but EXP-028 suggests budget-8 is too aggressive.

---

# Experiment program

Do not mechanically run every item if an earlier result makes a later branch redundant. Follow evidence.

## Phase 0 — measurement integrity

### EXP-029 candidate: decode-only profiling

Hypothesis:

> Correct decode-boundary deltas will materially change the attribution numbers but preserve the structural conclusion that expert loading and many small expert GEMMs/dispatches are important costs.

Implement and validate decode-only deltas for:

- model phase spans,
- MLX expert calls/load/compute,
- affine dispatch counts,
- MetalIO loads/bytes/waits/fails,
- speculative MetalIO used/wasted/ready/late,
- RouteScout metrics if reported as decode-only.

Keep lifetime totals available only if clearly labeled.

Acceptance:

- structural expert-call count matches decode forwards,
- no token changes,
- profiling off has no meaningful runtime behavior difference.

Do this before any performance A/B.

---

## Phase 1 — establish a reliable SSD-only baseline

Use:

```bash
LOGAN_EXPERT_NOCACHE=1
```

and raw MLX.

Baseline RouteScout states:

1. predictor completely off,
2. predictor shadow-only, no speculative I/O,
3. predictor prefetch with production confidence gate.

Use deterministic greedy output unless there is a reason not to.

Start with shorter runs for iteration, then use longer decode for promotion.

Suggested workload progression:

- 24 generated tokens for fast debugging,
- 64+ generated tokens for meaningful steady-state behavior,
- several prompt families for final qualification.

Alternate paired order, e.g.:

```text
B C C B
C B B C
```

Do not compare one candidate run against one baseline run.

Record temperature/background/thermal anomalies if obvious.

---

## Phase 2 — budget policy before a bigger neural model

The current forced budget-8 result wastes ~83% of speculative loads.

First determine whether a simpler policy fixes most of the problem.

### Fixed budget sweep

Compare equal conditions for:

```text
budget = 0, 1, 2, 4, 8
```

Do not assume top-k=8 implies speculative budget 8.

Measure:

- precision,
- cold-arrival recall,
- useful prefetches,
- wasted prefetches,
- speculative bytes/token,
- ready-at-demand,
- late-at-demand,
- expert load wait,
- decode ms/token.

A budget of 1 or 2 may beat 8 even with worse recall.

### Adaptive budget

Try a bounded adaptive policy such as:

```text
0 = weak/no evidence
1 = moderate
2 = strong
4 = exceptionally strong and I/O queue has headroom
```

Avoid dynamic 8+ until evidence justifies it.

Inputs may include:

- top candidate score,
- score margin,
- support count,
- per-layer historical precision,
- router entropy/margin if available before useful lead time,
- MetalIO outstanding queue depth,
- recent prefetch precision,
- measured read size/cost.

The policy must never change authoritative routing.

---

## Phase 3 — improve simple scoring

Do this before making the neural scorer larger.

### Spatial/temporal weighting

Current online fusion sums temporal and spatial conditional evidence roughly equally.

Historical evidence says spatial is stronger.

Sweep simple weights offline/shadow first, e.g.:

```text
temporal: 0, .25, .5, 1
spatial:  .5, 1, 2, 4
```

Normalize carefully so evidence count does not accidentally dominate score scale.

Evaluate true cross-prompt holdouts and live decode.

### Candidate score API

Consider changing the predictor from:

```rust
Vec<ExpertId>
```

to an internal ranked candidate structure like:

```rust
Prediction {
    expert,
    score,
    temporal_support,
    spatial_support,
}
```

if needed for adaptive budgets/confidence.

Keep the external execution behavior simple.

### Calibration

The existing confidence gate uses empirical layer precision with defaults:

- min samples ~16
- min precision 0.50

That is a blunt gate.

Test whether a calibrated candidate-score threshold or lower confidence bound predicts **expected I/O value** better than aggregate historical precision.

Do not lower the threshold merely to issue more reads.

---

## Phase 4 — increase prefetch lead time

This may be more valuable than squeezing a few points of precision.

Current same-layer prediction is prepared just before the current layer router executes. The fact that EXP-028 reported all useful prefetches ready at demand is encouraging, but it does not prove same-layer lead is optimal.

Test predictions for future layers while earlier layers compute.

### Spatial horizon

Use current layer routing evidence to predict target layers:

```text
L + 1
L + 2
L + 4
L + 8
```

Measure:

- precision decay,
- recall decay,
- available overlap time,
- ready-at-demand,
- late-at-demand,
- wasted bytes,
- wall time.

A lower-accuracy L+4 predictor can outperform L+1 if it hides much more SSD latency.

### Earliest-use objective

Consider predicting:

> Will expert E be needed anywhere in the next N layers?

rather than only one exact future layer.

This may align better with SSD staging.

Be careful: the pending cache is keyed by `(layer, expert)`, so cross-layer expert IDs are not interchangeable weights.

---

## Phase 5 — MetalIO submission efficiency

Only after policy is reasonable.

Current raw-MLX `ExpertSource::prefetch` may still issue one expert staging operation at a time.

Investigate:

- submitting several predicted expert slots in one MetalIO batch,
- queue-depth tuning,
- merging physically adjacent ranges where it genuinely reduces I/O,
- avoiding redundant file-handle lookup/planning work,
- reusing immutable per-expert I/O plans instead of rebuilding names/ranges every token.

Do not merge ranges if it causes significant over-read.

Measure command/CPU overhead separately from physical read time.

### Important potential win: precompute I/O plans

The raw MLX source currently derives expert tensor names/ranges at runtime.

For a fixed checkpoint, the physical gate/up/down ranges are immutable.

Test caching compact per-`(layer,expert)` plans:

```text
file id
source offset
byte count
destination offset
matrix metadata
```

This can reduce host allocation/string/hash-map work even before storage latency.

Watch memory use: 40 x 256 experts is only 10,240 experts, so compact descriptors should be manageable.

Record an experiment rather than assuming it helps.

---

## Phase 6 — learned residual scorer only if simple policy plateaus

The candidate-centric scorer is already qualified on ANE, but earlier cross-prompt gains were small.

Use it as a **reranker/residual**, not a replacement for strong spatial priors.

Promote it only if, at equal speculative byte budget, it improves:

- useful speculative reads,
- expected stall avoided,
- ready-at-demand,
- end-to-end decode.

ANE inference cost is not the main concern; host feature construction previously dominated predictor compute.

If revisiting the scorer:

- minimize feature construction,
- use spatial/temporal base score explicitly,
- train on cold-arrival or expected-cost targets,
- evaluate cross-prompt.

---

## Phase 7 — only then consider future-work GPU batching

There is a broader idea worth testing after SSD prefetch is useful:

> RouteScout can predict a window of future expert work and let Logan batch/prepare larger Metal workloads rather than only fetching bytes early.

Do not begin here.

First prove useful SSD prefetch. Otherwise batching obscures whether RouteScout or GPU scheduling caused the result.

If pursued later, native routing must still validate/authorize actual expert execution.

---

# Metrics that must become first-class

For each candidate, record at least:

### End-to-end

- decode ms/token
- tok/s
- paired delta vs baseline

### Authoritative expert work

- expert calls/decode-forward
- expert load ms/decode-forward
- expert compute ms/decode-forward
- physical demand reads
- demand bytes

### Speculation

- predicted candidates
- speculative loads
- speculative bytes
- useful/consumed
- wasted
- ready-at-demand
- late-at-demand
- precision
- cold-arrival recall

### MetalIO

- loads
- bytes
- waits
- failures
- outstanding/peak outstanding
- average latency if trustworthy

### Predictor overhead

- scoring time
- host feature construction
- I/O plan construction
- queue/submission overhead

### Correctness

- generated token IDs or stable fingerprint
- exact equality vs baseline

Never report a speedup without the correctness result beside it.

---

# Promotion criteria

A RouteScout candidate can be enabled more broadly only when:

1. native router remains authoritative,
2. output/token identity passes,
3. decode-only measurement is correct,
4. useful speculative reads materially exceed harmful/wasted cost,
5. ready-at-demand increases without pathological queue pressure,
6. paired end-to-end runs show a repeatable win larger than noise.

For a serious promotion run, target roughly:

- >=20 paired comparisons when practical,
- positive median paired improvement around >=2%,
- >=15/20 paired wins,
- no correctness failures.

If variance makes those gates inappropriate, justify a better statistical test in the experiment entry.

Do not promote based on:

- Recall@K alone,
- one fast run,
- ANE microbenchmark,
- lower expert-load time with worse wall time,
- more ready-at-demand with excessive speculative bytes.

---

# Pruning rules

Reject/de-prioritize a branch if:

- it needs the confidence gate disabled to issue useful work but loses wall time,
- it improves recall mainly by flooding SSD bandwidth,
- it is prompt-specific and fails cross-prompt,
- host feature/preparation cost erases I/O savings,
- longer lead time causes too much precision decay,
- queue pressure makes authoritative reads slower,
- it changes authoritative routing,
- it depends on converting raw MLX to another format.

Preserve the result in `EXPERIMENTS.md`.

---

# Required experiment discipline

`AGENTS.md` is authoritative.

For every non-trivial performance/architecture experiment:

1. allocate the next free `EXP-NNN`,
2. write hypothesis + baseline + candidate before or alongside execution,
3. save raw logs under:
   `.perf_runs/routescout/EXP-NNN-<slug>/`,
4. record exact commands/flags/model/prompt,
5. record correctness,
6. record result even if negative,
7. keep or reject explicitly.

Do not overwrite EXP-028.

---

# Suggested first actions

1. Read `AGENTS.md`.
2. Read EXP-018 through EXP-028 in `EXPERIMENTS.md`.
3. Inspect the current dirty diff before touching `logan-qwen4/src/lib.rs`, `main.rs`, `logan-metal`, or RouteScout tools.
4. Allocate EXP-029 for decode-boundary measurement correction.
5. Implement decode-only counter snapshots/deltas.
6. Verify the Qwen3.6 structural expert-call count.
7. Run an SSD-only baseline with `LOGAN_EXPERT_NOCACHE=1`.
8. Run fixed RouteScout budgets 0/1/2/4/8 in shadow and real-prefetch modes.
9. Follow the evidence into adaptive budgets or better spatial weighting.
10. Only after that investigate longer spatial horizons and MetalIO batching.

---

# Known-green regression gates at handoff creation

Recent verified results:

```text
cargo test -p logan-qwen4 --lib
92 passed, 0 failed, 3 ignored

cargo test -p logan-metal --lib
5 passed, 0 failed

cargo test -p logan-compiler --lib
135 passed, 0 failed

cargo test -p logan-artifact
5 passed, 0 failed
```

Relevant MetalIO test includes multi-file vectored source ranges.

Do not rerun unrelated broad suites after every tiny edit, but run the relevant tests before claiming completion.

---

# Completion criteria for this slice

The agent should keep going until either:

## Success

A RouteScout configuration on real raw-MLX Qwen3.6 demonstrates a repeatable, correctness-preserving SSD-streaming decode win under controlled paired testing.

or:

## Decisive negative result

After correct profiling and reasonable budget/lead-time/policy exploration, evidence shows RouteScout cannot materially improve Qwen3.6 under this storage/runtime regime.

In that case:

- preserve the best shadow predictor,
- keep harmful prefetch gated/off,
- document why,
- identify whether a model with a larger expert working set / slower storage / distributed latency is a better systems target.

Do not manufacture a win.

---

# Final report expected from the agent

Report:

1. measurement-integrity fixes,
2. experiments added,
3. best predictor/policy,
4. best budget and horizon,
5. useful/wasted/ready/late prefetch metrics,
6. MetalIO byte/queue behavior,
7. paired wall-time result,
8. correctness evidence,
9. rejected ideas,
10. files changed,
11. exact next bottleneck.

If RouteScout wins, explain **where the saved time came from**.

If it loses, explain **what prevented predictions from becoming wall-time savings**.
