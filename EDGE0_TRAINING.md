# Edge0-style prerouter training for Logan

**Status:** completed through EXP-078; 50k RouteScout head promoted
**Final training branch:** `exp/routescout-train-v1`
**Target checkpoint:** `~/models/Qwen3.6-35B-A3B-MLX-oQ4-FP16`
**Published initialization:** `~/models/prerouter_edge0_35b.safetensors`

This document is the reproducibility log for training a Logan-specific Edge0-style
predictive router. The goal of this phase is **prediction only**: the native Qwen
K4 gate remains the teacher and semantic authority.

## 1. Why train our own head

EXP-073 established two facts:

1. The published Edge0 head is a better predictor than RouteScout on this
   checkpoint (recall@4 0.316 vs 0.229 in the measured M=4 staging comparison).
2. It is still far too inaccurate to make speculative SSD reads economical, and
   using the published head as an authoritative router has a large quality cost
   on this untouched Qwen checkpoint.

The hypothesis here is that most of that gap is checkpoint/domain mismatch:
train the same small architecture directly against **our model's own native K4
routes** before changing any model semantics.

## 2. Preserving the existing experiment tree

The starting tree contained the uncommitted RouteArena/MoE-island/RouteScout/
Edge0/Hybrid experiments from EXP-067..073. Before this work, it was snapshotted
outside the repo at:

`~/CODE/logan-checkpoints/edge0-train-pre-20260923/`

The snapshot contains `head.txt`, `status.txt`, `tracked.patch`, and
`untracked.tgz`. No reset/stash was used. A new branch,
`exp/edge0-train`, was created on top of the preserved dirty tree.

## 3. Exact training target

The temporal relationship matches the Edge0 prerouter used by Logan:

```text
owner layer N, token t:
    hidden_N(t)
    native route_N(t)
    native route_N(t-1)
        |
        v
target = native route_(N+1)(t+1)
```

Feature width is therefore:

- hidden: 2048
- current-route one-hot: 256
- previous-route one-hot: 256
- total: **2560**

The deployed head remains:

```text
fc1:        2560 -> 512, no bias
exact-erf GELU
fc2:         512 -> 256, no bias
linear_init: 2560 -> 256, no bias
logits = fc2(gelu(fc1(x))) + linear_init(x)
```

For the quality-preserving Logan path we train owners **6..37**, predicting
consumers **7..38**. Consumer layer 39 remains exact/native. The published
owner-38 tensors are retained in the adapter but are not changed by this pilot.

## 4. Trace collector

Implementation: `logan-qwen4/src/edge0_train_trace.rs`

Enable it with:

```bash
QWEN_EDGE0_TRACE_DIR=.perf_runs/edge0-train-v1 \
QWEN_ROUTE_MODE=native-truncated \
QWEN_ROUTE_NATIVE_K=4 \
LOGAN_EXPERT_NOCACHE=1 \
target/release/examples/decode_bench \
  ~/models/Qwen3.6-35B-A3B-MLX-oQ4-FP16 64 sample "prompt"
```

The collector is observational. It is called **after the native gate has selected
K4 and before any predictor can override a route**. It does not choose experts or
issue I/O.

### Binary format: `edge0-logan-trace-v1`

One appendable file per owner: `owner-06.e0trace` ... `owner-37.e0trace`.

Header: 32 bytes. Record: 4144 bytes.

Each record contains:

| field | representation |
|---|---|
| run id | u64 |
| token generation | u64 |
| owner hidden state | 2048 x FP16 |
| current owner route | 4 x u16 |
| previous owner route | 4 x u16 |
| next-token consumer target route | 4 x u16 |
| normalized native target weights | 4 x FP16 |

Every process invocation receives a distinct run ID so validation can split by
**whole generation runs**, not neighboring tokens.

### Double-buffer rule

The collector has `CURRENT` and `NEXT` pending-feature buffers. This is
mandatory: on token t+1, owner N executes before consumer N+1, so a single
pending slot would overwrite token t's feature before its target route arrives.

This bug was caught by the first two smoke runs: both created valid headers but
zero records. After introducing the same current/next lifetime used by the
runtime Edge0 router, the smoke produced exactly:

- 8 generated tokens
- 7 decode forwards
- 6 trainable temporal pairs/head
- 32 heads
- **192 records total**

A parsed owner-06 record had finite hidden values, legal expert IDs, and target
weights summing to 0.999878 after FP16 storage.

## 5. Training environment

Created an isolated environment rather than altering system Python:

```bash
uv venv ~/.venvs/logan-edge0-train
uv pip install --python ~/.venvs/logan-edge0-train/bin/python \
  mlx numpy safetensors
```

Pilot versions:

- Python 3.12.11
- MLX 0.32.2
- NumPy 2.5.3
- safetensors 0.8.0

The published adapter was inspected directly: 99 FP16 tensors = 33 heads x
`{fc1, fc2, linear_init}`, with expected shapes
`(512,2560)`, `(256,512)`, and `(256,2560)`.

## 6. Trainer

Implementation: `tools/train_edge0_router.py`.

The first phase **fine-tunes the published head**, rather than random
initialization. Weights are optimized in FP32 and exported to FP16 with the same
tensor names Logan already loads.

Loss:

```text
soft cross entropy(predicted 256-expert logits,
                   native K4 target distribution)
```

The four selected native experts receive their native router weights
renormalized over K4; all other experts receive target probability zero.

Validation metrics include:

- exact native top-1 accuracy
- candidate recall@1 / @4 / @8 / @12
- weighted recall at each M
- full-route coverage@4
- soft cross entropy

**Metric convention:** recall@M is `hits / (4 * samples)`, matching the
candidate-recall convention used by EXP-073. Thus recall@1 has a mathematical
maximum of 0.25.

Validation is split by `run_id` whenever at least two generation runs exist.
The trainer only falls back to a sample-level split for tiny smoke traces and
labels that case `sample_fallback`.

Example:

```bash
~/.venvs/logan-edge0-train/bin/python tools/train_edge0_router.py \
  --trace-dir .perf_runs/edge0-train-v1 \
  --base-adapter ~/models/prerouter_edge0_35b.safetensors \
  --output ~/models/prerouter_logan_qwen36_v1.safetensors \
  --epochs 8 --batch-size 32 --lr 1e-4
```

The output adapter preserves untouched published tensors and replaces only the
trained owner heads. A sibling `.metrics.json` contains the complete
per-head baseline, epoch history, final validation metrics, and split details.

## 7. Pilot corpus

Collection script: `tools/collect_edge0_traces.py`.

The first pilot uses eight distinct prompts spanning systems programming,
debugging, arithmetic reasoning, MoE/inference, Python/data work, networking,
science explanation, and creative prose.

Fixed collection settings:

- native K4 teacher
- sample decoding
- temperature 0.8
- top-p 0.95
- unique deterministic seed per run
- 64 generated tokens/run
- expert cache disabled to keep the runtime regime aligned with EXP-073

Command:

```bash
~/.venvs/logan-edge0-train/bin/python tools/collect_edge0_traces.py \
  --model ~/models/Qwen3.6-35B-A3B-MLX-oQ4-FP16 \
  --trace-dir .perf_runs/edge0-train-v1 \
  --tokens 64 --runs 8
```

## 8. Evidence gates

This phase does **not** promote the learned router to authoritative routing.

The first questions are:

1. Does Logan-specific training materially beat the published head on held-out
   native-route prediction?
2. Is the improvement broad across layers rather than concentrated in a few?
3. Does candidate density become good enough to revisit early I/O?
4. If prediction becomes excellent, is a Recover-LoRA/joint-recovery phase worth
   testing before any authoritative deployment?

The existing native route remains the correctness oracle throughout this phase.

## 9. Current verification

- `cargo check -p logan-qwen4`: passes.
- trace module unit tests: **4 passed, 0 failed** (including the current/next lifetime regression).
- real-model trace smoke: passes, 192 correctly aligned records.
- MLX one-head trainer/export smoke: passes.
- clean full corpus integrity: **12 runs, 1512 examples/head, 48,384 total, zero partial records**.
- full 32-head Logan-specific training: completed in **55.67 s**.
- exact Logan FP16/BNNS adapter A/B: completed on an unseen prompt with token-identical native-K4 output.
- final serialized release suite: **149 passed, 0 failed, 3 ignored**; integration tests also pass.

## 10. Full v1 training result

The final v1 run uses the larger clean corpus at `.perf_runs/edge0-train-v1-clean`,
not the earlier 8×64 pilot. The first attempted shared-path batch was quarantined as
`.perf_runs/edge0-train-v1-contaminated` after concurrent collection was detected; none of
those records were used.

### Dataset

- 12 independent generated continuations
- 128 generated tokens/run → 127 decode forwards/run → **126 labeled temporal pairs/head/run**
- 1512 examples/head
- 32 trained heads (owners 6..37)
- 48,384 total examples
- 191.2 MiB trace payload
- each run has generations 1..126 exactly
- all target expert IDs are valid and target-weight sums are 0.999634..1.000366 after FP16 rounding

### Training

```bash
~/.venvs/logan-edge0-train/bin/python tools/train_edge0_router.py \
  --trace-dir .perf_runs/edge0-train-v1-clean \
  --base-adapter ~/models/prerouter_edge0_35b.safetensors \
  --output .perf_runs/edge0-train-v1-clean/prerouter_logan_qwen36_v1.safetensors \
  --epochs 12 --batch-size 64 --eval-batch-size 256 \
  --lr 1e-4 --weight-decay 1e-4
```

Validation is held out by complete run ID. Best checkpoint is selected independently for each
head by weighted native K4 mass, then recall@4, then loss. Mean best epoch was 6.81 and
median 7, showing that keeping the per-head best checkpoint avoids measurable late-epoch
overfit.

### Held-out metrics (mean over 32 heads)

| metric | published adapter | Logan-trained v1 | delta |
|---|---:|---:|---:|
| soft CE | 3.6161 | **2.9421** | -0.6740 |
| exact native top-1 | 28.35% | **37.80%** | +9.45 pp |
| predicted top-1 ∈ native K4 | 59.93% | **72.07%** | +12.14 pp |
| recall@1 (max 25%) | 14.98% | **18.02%** | +3.04 pp |
| recall@4 | 43.21% | **53.16%** | +9.95 pp |
| recall@8 | 61.00% | **69.77%** | +8.77 pp |
| recall@12 | 69.94% | **78.18%** | +8.25 pp |
| weighted native mass@4 | 46.90% | **57.80%** | +10.90 pp |
| full native K4 coverage@4 | 3.72% | **9.78%** | +6.06 pp |

Recall@4 improved on 30/32 heads; weighted mass@4 improved on 31/32. The absolute offline
numbers are evaluator/corpus-specific, so the before/after within this evaluator is what is
meaningful.

### Exact Logan deployment-path check

To rule out a Python-vs-FP16/BNNS discrepancy, the published and trained adapters were run
through Logan's actual Edge0 implementation on the same unseen B-tree/LSM-tree prompt:

```text
QWEN_ROUTE_MODE=hybrid
QWEN_ROUTE_NATIVE_K=4
QWEN_HYBRID_FUSION=edge0
QWEN_HYBRID_RESIDENT_PRIOR=0
QWEN_HYBRID_STAGE_M=4
LOGAN_EXPERT_NOCACHE=1
```

| adapter | runtime recall@4 | full-route coverage@4 | staged hits |
|---|---:|---:|---:|
| published Edge0 | 28.90% | 1.33% | 2950 |
| **Logan-trained v1** | **44.54%** | **13.71%** | **4547** |

The two runs emitted the **exact same 64 generated token IDs**, because native K4 remains
the semantic authority. Both reported `duplicate_reads=0` and `late=0`.

This gives the trained adapter a **+15.64 percentage-point runtime recall@4 gain** and over
10× the full-route coverage on this unseen prompt. That is direct evidence that the learned
weights transfer through Logan's real FP16+BNNS inference path.

Do not interpret the sequential wall times as a speed benchmark. EXP-073 already established
that M=4 speculative reads are too expensive at current prediction density, and host state
varied between these two runs. EXP-074 is a **prediction-quality win**.


### M=1 staging follow-up after training

Because EXP-073 showed speculative bandwidth is the practical constraint, the trained adapter was
also tested in the most favorable narrow staging shape: **Edge0-only M=1**, resident prior off,
native K4 still authoritative. A 16-token greedy prompt was run in both arm orders.

All arms emitted the exact same token IDs.

| order | arm | tok/s | M=1 hits | stage efficiency | overall recall |
|---|---|---:|---:|---:|---:|
| A | native K4 | **5.6608** | — | — | — |
| A | published Edge0 M=1 | 4.3003 | 185 | 0.361 | 0.0732 |
| A | trained Edge0 M=1 | **4.5445** | **375** | **0.732** | **0.1483** |
| B | trained Edge0 M=1 | **4.7540** | **375** | **0.732** | **0.1483** |
| B | published Edge0 M=1 | 4.1770 | 185 | 0.361 | 0.0732 |
| B | native K4 | **6.2070** | — | — | — |

The learned head therefore **doubled useful M=1 staged hits** on this trajectory
(185 -> 375) and roughly doubled staged-byte efficiency (36.1% -> 73.2%). It also beat the
published M=1 arm in both orderings.

That still does **not** make speculative staging a speed win on this M2: trained M=1 remained
about 20-23% slower than native K4 in these two matched sequences. The result strengthens the
case for learned prediction while preserving EXP-073's I/O conclusion: better prediction alone
does not remove predictor/orchestration cost or the cost of wrong speculative reads.

### Artifacts

- deployed trained adapter: `~/models/prerouter_logan_qwen36_v1.safetensors`
- reproducible adapter + report:
  `.perf_runs/edge0-train-v1-clean/prerouter_logan_qwen36_v1.safetensors`
  `.perf_runs/edge0-train-v1-clean/prerouter_logan_qwen36_v1.metrics.json`
- clean traces: `.perf_runs/edge0-train-v1-clean/`
- training implementation: `tools/train_edge0_router.py`

### Decision

Checkpoint-specific learned route prediction is **worth continuing**. It materially repairs
the published Edge0 mismatch. The next slice should grow corpus diversity/size and test richer
features (including RouteScout/locality signals) before Recover-LoRA or authoritative routing.
Native K4 stays authoritative for now.


## 11. Native-K quality sweep and K4 operating point (EXP-075/076)

### Goal

Find the lowest native routed-expert width that keeps Qwen3.6 close enough to the
full native K8 teacher to justify using it as the semantic target for a new
Edge0-style prerouter.

The sweep uses `examples/quality_probe` in `native-truncated` control mode:
the full K8 model supplies a greedy teacher stream and per-position logits; each
lower-K student is teacher-forced on the exact same token stream. This separates
local logit damage from generation cascades.

### Primary 4-prompt / 128-position sweep

| K | top-1 agreement | KL(teacher||student) | divergences |
|---:|---:|---:|---:|
| 7 | **100.00%** | **0.00206** | 0 / 128 |
| 6 | 99.22% | 0.00624 | 1 / 128 |
| 5 | 97.66% | 0.01347 | 3 / 128 |
| 4 | 97.66% | 0.03915 | 3 / 128 |
| 3 | 94.53% | 0.09024 | 7 / 128 |
| 2 | 78.12% | 0.6702 | historical EXP-072 control |

K3 was explicitly tested because K4-vs-K2 left the boundary unresolved. It is
not a hidden free win: the distribution damage roughly doubles again from K4
and top-token changes become common.

### Disjoint 8-prompt confirmation

A separate systems/math/networking/compiler/cache corpus used 8 prompts x 24
positions:

| K | top-1 agreement | KL(teacher||student) | divergences |
|---:|---:|---:|---:|
| 7 | **98.96%** | **0.00395** | 2 / 192 |
| 6 | 98.44% | 0.01411 | 3 / 192 |

Combining both quality corpora:

- K7: **99.376%** top-1 agreement, weighted KL **0.00319**, 2/320 changes.
- K6: **98.752%** top-1 agreement, weighted KL **0.01096**, 4/320 changes.

K7 is the conservative near-transparent setting. K6 has a small but repeatable
distribution shift; K5 and below are where the curve degrades much more clearly.

### Throughput A/B

Two opposite-order 24-token runs used the same prompt and
`LOGAN_EXPERT_NOCACHE=1`.

| K | pass A tok/s | pass B tok/s | mean within-pass speedup vs K8 |
|---:|---:|---:|---:|
| 8 | 4.6594 | 4.5108 | baseline |
| 7 | 5.0346 | 4.8408 | **+7.68%** |
| 6 | 5.0674 | 5.1609 | **+11.58%** |
| 5 | 5.5846 | 5.7779 | **+23.97%** |

K5 changed the actual greedy continuation in this throughput probe. K6 and K7
matched K8's 24-token continuation.

### Operating-point decision

**Use K4 as the active Qwen3.6 execution and Edge0-training target.**

The sweep shows that K6/K7 preserve the K8 distribution more closely, but K4 is the selected
performance/quality tradeoff because it removes substantially more routed-expert bandwidth. K3 is
the first clear quality cliff below it: 94.53% top-1 agreement and KL 0.09024 versus K4's 97.66%
and KL 0.03915 on the primary 128-position probe.

This is not a claim that K4 is lossless. It is the intentionally more aggressive operating point.
EXP-074 already trained the Logan-specific Edge0 v1 adapter against native K4, so no target-width
migration is needed.

### K-aware training pipeline

The training path was generalized so the target width is no longer hard-coded
to K4:

- `QWEN_EDGE0_TRACE_K=<K>` selects trace route width.
- `edge0_train_trace.rs` stores K in each file header and computes record size
  as `16 + 2*hidden + 8*K`.
- Existing trace files are checked before append; a K mismatch fails closed.
- `tools/collect_edge0_training.py --k K` sets both native execution K and trace K.
- `tools/train_edge0_router.py` reads K from the trace header, trains against
  all K target experts, and writes `target_k` into safetensors metadata.
- `tools/validate_edge0_traces.py` validates arbitrary K datasets.

K6 smoke:

- record size: **4160 bytes**
- 8 generated tokens -> 7 measured decode forwards -> **6 pairs/head**
- 32 heads -> **192 examples**
- normalized FP16 target-weight max error: **0.000244**
- collector focused tests: 4/4 pass
- `cargo check -p logan-qwen4`: pass

### Archived K6 corpus

A K6 collection had already started during the sweep. It was stopped once K4 was selected and moved
to:

`.perf_runs/edge0-train-k6-abandoned-20260923`

The directory is intentionally retained as archival evidence, but it is incomplete and must not be
used as an active training corpus. The K-aware pipeline remains useful infrastructure for future
experiments; the active Qwen3.6 target stays K4.

## 12. RouteScout K4 scaling campaign (EXP-078)

EXP-078 scales the same architecture, optimizer, and selection rule to a
substantially larger corpus and measures where the curve flattens. The procedure
is in `EXPERIMENTS.md`; this section records only what a later run needs to
reproduce it.

### Terminology (changed here)

- **Edge0 head** = `~/models/prerouter_edge0_35b.safetensors` (published).
- **RouteScout head** = our checkpoint-specific learned router. v1 is
  `~/models/prerouter_logan_qwen36_v1.safetensors`; EXP-078 trains v2.

The name "Logan-trained Edge0 head" used earlier in this document for v1 is
retired: it conflated the published Edge0 head with our own trained head.

### Corpus and split

- Collector: `tools/collect_routescout_corpus.py`, one process per prompt.
- Prompts: `tools/routescout_prompts.py` — 212 pool / 8 val / 8 test, disjoint,
  pool ordered round-robin across 18 domains so every prefix spans all topics.
- Settings: native-truncated K4, 256 tokens/run, sampled (T 0.8, top-p 0.95,
  top-k 50), deterministic per-prompt seed, `LOGAN_EXPERT_NOCACHE=1`.
- Split by whole run, never by token: pool = training, `val` bank = model
  selection at every scale, `final/` dir = untouched final evaluation.
- Scale points are **nested collection-order prefixes**, so `5k ⊂ 10k ⊂ 25k ⊂ 50k`
  by construction and a checkpoint's dataset is exactly "the first N pool runs".

### Trainer interface added for scaling

```text
--max-runs N        train on the first N pool runs (nested prefix)
--live-prefix       read the longest whole-run prefix of a still-collecting corpus
--val-bank NAME     which corpus-index bank supplies model-selection runs
--init adapter|random   Arm A/B (adapter) vs Arm C (random)
--selection exp078|exp074   checkpoint rule; exp074 reproduces the v1 numbers
```

The loader is a structured mmap view; a head costs address space, not resident
memory, so a 50k-example corpus does not need ~6.5 GiB of RAM.

### Runtime cost instrumentation

`LOGAN_PROFILE=1` now prints `predict=`, the learned-head evaluation time,
separate from the native gate's `route=`. Measured on this host: 45–48 ms/token
for the head on the deployed FP16/BNNS path (control with no predictor: 0.0).

### Reproducing the EXP-078 measurement and verification chain

Verification, in the order it should be run:

```bash
# 1. Corpus integrity, including the temporal pairing the format checks cannot see.
python tools/validate_edge0_traces.py .perf_runs/routescout-train-v1/final --check-alignment
#    -> VALID temporal_alignment_pairs=62744 status=ok

# 1b. A still-collecting corpus: validate only whole runs, ignore the partial tail.
python tools/validate_edge0_traces.py .perf_runs/routescout-train-v1/corpus --prefix-runs 14

# 2. Held-out comparison of every adapter on the frozen bank (identical records).
python tools/eval_routescout.py --trace-dir .perf_runs/routescout-train-v1/final \
  --adapter edge0_published=~/models/prerouter_edge0_35b.safetensors \
  --adapter routescout_v1=~/models/prerouter_logan_qwen36_v1.safetensors \
  --owners 6-37 --all-holdout --live-prefix --output .../results/compare-test.json

# 3. Scaling curve + best-checkpoint nomination + per-layer findings.
python tools/summarize_routescout_sweep.py --runs-dir .../runs --output .../results/summary.json \
  --compare test=.../results/compare-test.json
python tools/analyze_routescout_layers.py --eval test=.../results/compare-test.json \
  --output .../results/layers.json --curve <scale>=<metrics.json> ...

# 4. Cost. Both refuse to run while any other model process is alive.
python tools/measure_routescout_latency.py --trace-dir .../final --adapter <n>=<path> --output .../latency.json
python tools/measure_routescout_deployed.py --adapter <n>=<path> --output .../deployed.json
```

**Scale points are nested prefixes.** Because the trace is append-only and the
pool bank is ordered, "the first N runs" is the same bytes whenever it is trained.
A scale point is therefore trained exactly once, and the sweep fails closed if a
point asks for more runs than exist rather than silently relabelling a smaller
corpus.

**Two invariants worth not breaking:**

1. `load_trace_prefix` must return the same record count for every owner head.
   Records are written layer by layer within a token, so raw file lengths differ
   mid-token; only the prefix view is comparable. If a future change makes this
   per-file, the 32 heads silently train on different data at the same scale point.
2. `select_run_sets` must never fall back to a smaller run set. A mislabelled
   scale point is indistinguishable from a real result and would corrupt the
   scaling conclusion.
