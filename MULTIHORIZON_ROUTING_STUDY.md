# Multi-horizon K4 routing predictability study

**Date:** 2026-09-23
**Status:** completed, offline only — **no training performed**
**Checkpoint:** `~/models/Qwen3.6-35B-A3B-MLX-oQ4-FP16`
**Canonical route width:** K4
**Trace corpus:** `.perf_runs/edge0-train-v1-clean`

## Question

Does predicting several tokens ahead expose enough expert locality to be useful for Logan's
SSD/residency scheduler, even if exact future K4 routes become hard to predict?

The study separates:

1. exact future-route predictability;
2. future-window compressibility (how many unique experts are actually needed);
3. simple history/RouteScout-style working-set prediction;
4. the existing published and Logan-trained Edge0 heads scored at +1/+2/+4/+8 without any retraining.

## Corpus

The canonical K4 corpus validates cleanly:

- 12 independent generation runs
- 126 temporal records/head/run
- 1512 records/head
- 32 owner layers (6..37)
- 48,384 total examples
- zero malformed/partial records

For RouteScout-style transition-table evaluation, 10 complete runs were used to build statistics
and 2 whole runs were held out. Oracle/locality measurements use route structure only and are
reported separately from predictor accuracy.

## 1. Native route autocorrelation

Same-layer overlap between the current K4 route and the route H tokens later:

| horizon | same-route recall | exact K4 set repeated |
|---:|---:|---:|
| +1 | 30.30% | 1.49% |
| +2 | 23.79% | 0.88% |
| +4 | 19.16% | 0.48% |
| +8 | 17.08% | 0.38% |

Exact-route identity therefore decays quickly. Simply keeping the current K4 route is not enough
for a far-horizon predictor.

Layer heterogeneity is significant. At +1, overlap ranges from 22.45% (layer 34) to 39.35%
(layer 20). At H=4 with a 12-expert recent-history working set, layer 20 covers 57.1% of future
expert-use events while layers 32/34 cover only ~30.4%.

## 2. The future working set is much smaller than raw expert-use count

Each token uses 4 routed experts per layer. Across a future H-token window the raw number of
expert-use events is therefore 4H, but many experts repeat.

| window | raw expert-use events | mean unique experts | p50 unique | p90 unique |
|---:|---:|---:|---:|---:|
| 2 tokens | 8 | 6.79 | 7 | 8 |
| 4 tokens | 16 | **11.36** | 11 | 14 |
| 8 tokens | 32 | **18.56** | 19 | 23 |

A perfect retention policy would therefore need ~29% fewer distinct expert blocks than raw
event count over 4 tokens, and ~42% fewer over 8 tokens.

### Oracle budget needed for future expert-use coverage

| window | 80% coverage | 90% coverage | 95% coverage |
|---:|---:|---:|---:|
| 2 tokens | 5.81 | 6.79 | 6.79 |
| 4 tokens | **8.39** | **10.36** | **11.36** |
| 8 tokens | **12.65** | **15.57** | **17.56** |

At fixed budgets, an oracle can cover:

| window | B4 | B8 | B12 | B16 |
|---:|---:|---:|---:|---:|
| 2 | 65.1% | 100% | 100% | 100% |
| 4 | 53.1% | **78.7%** | **96.7%** | 100% |
| 8 | 44.4% | 66.3% | **79.3%** | **90.4%** |

This is the main positive result: substantial multi-token reuse exists.

## 3. Simple non-neural prediction

### Recent-4-token frequency baseline

Rank experts by frequency over the previous four routes and retain the top B:

| future window | B4 | B8 | B12 | B16 |
|---:|---:|---:|---:|---:|
| 1 | 29.45% | 43.26% | 49.11% | 50.11% |
| 2 | 27.44% | 40.08% | 46.15% | 47.21% |
| 4 | 24.94% | 36.55% | **42.89%** | 43.99% |
| 8 | 22.35% | 33.16% | 39.68% | **40.88%** |

This beats a held-out RouteScout-style transition-table predictor through most small/medium
budgets.

### Held-out RouteScout-style horizon tables

The table predictor uses peak-normalized temporal + spatial transition scores with the same
0.25 temporal weighting as in-tree RouteScout, but is fit directly for each horizon/window.
Future-window event coverage:

| future window | B4 | B8 | B12 | B16 |
|---:|---:|---:|---:|---:|
| 1 | 21.57% | 33.81% | 42.57% | 49.39% |
| 2 | 20.63% | 32.06% | 40.55% | 47.22% |
| 4 | 18.60% | 29.59% | **37.68%** | 44.04% |
| 8 | 17.43% | 27.83% | 35.69% | **41.88%** |

RouteScout-style discrete transition counts do not capture enough of the longer-horizon working
set. At H=4/B12, recent frequency is 42.9% vs RouteScout-style 37.7%.

## 4. Existing Edge0 heads scored farther ahead — no retraining

The existing +1 logits were frozen and scored against held-out native K4 consumer targets at
+1/+2/+4/+8.

### Exact future K4 route

| adapter | +1 recall@4 | +2 | +4 | +8 |
|---|---:|---:|---:|---:|
| published Edge0 | 43.21% | 28.12% | 21.59% | 17.64% |
| **Logan-trained K4** | **53.16%** | **30.32%** | **22.74%** | **18.65%** |

The learned +1 signal decays very quickly. Exact +4/+8 routing should not be assumed to fall out
of the existing +1 head automatically.

### Use the same logits as a future working-set ranking

Logan-trained K4 Edge0 event coverage:

| future window | B4 | B8 | B12 | B16 |
|---:|---:|---:|---:|---:|
| 1 | 53.16% | 69.77% | 78.19% | 83.09% |
| 2 | 41.75% | 57.44% | **66.62%** | 72.49% |
| 4 | 32.58% | **46.67%** | **55.63%** | 61.97% |
| 8 | 25.98% | 38.61% | **47.16%** | **53.58%** |

Published Edge0 is consistently worse; for example H=4/B12 is 50.19%, versus 55.63% for the
Logan-trained head.

Even though the current head was never trained for a multi-token union target, it beats both
recent-history and RouteScout-style baselines as a future working-set ranker.

At H=4/B12:

- oracle: 96.69%
- trained Edge0: **55.63%**
- recent-4 history: 42.89%
- RouteScout-style table: 37.68%

So the head captures only ~58% of the available oracle coverage. There is substantial room for a
purpose-built multi-horizon objective.

## Interpretation

### What does *not* look promising

Treating +4/+8 as exact K4 route prediction. Existing +1 Edge0 recall collapses to 22.7% / 18.7%;
native route autocorrelation shows the same qualitative decay.

### What *does* look promising

Predicting a **future expert working set** for retention/residency rather than exact token-specific
routes.

There is real reuse:

- 4 tokens = 16 routed-expert uses but only ~11.36 unique experts on average.
- 8 tokens = 32 uses but only ~18.56 unique experts.
- An oracle B12 set covers 96.7% of four-token expert uses.
- Current trained Edge0 B12 already covers 55.6% without being trained for that objective.

This changes the economics compared with speculative reads. A wrong residency prediction can
merely keep the wrong already-loaded block longer; it does not have to cause a new SSD read.

### Likely operating shape

The data points toward a short horizon, roughly **H=4**, with an adaptive per-layer residency
budget. H=8 has more total reuse but predictability has decayed farther and the useful set grows
substantially.

For a fixed one-buffer retention cache, the existing expert stride is ~1.6875 MiB:

- B4/layer across 40 layers: ~270 MiB
- B8/layer: ~540 MiB
- B12/layer: ~810 MiB
- B16/layer: ~1080 MiB

B8 is an attractive memory-efficiency knee in the current untrained working-set ranking:
H=4 coverage rises to 46.7%, while B12 costs another ~270 MiB for ~9 percentage points more.

Layer-adaptive allocation should be better than a uniform budget because measured predictability
varies strongly by layer.

## Conclusion

**Multi-token prediction is worth pursuing, but as future-working-set/residency prediction, not
as exact far-future routing.**

The strongest next research target is a model that predicts:

`P(expert e is used at least once / how often e is used over tokens t+1..t+4)`

while retaining the existing +1 exact-route objective as a separate output. This study deliberately
does not train such a model.

## Reproducible artifacts

- structural study: `tools/study_multihorizon_routes.py`
- frozen-adapter horizon evaluator: `tools/eval_edge0_multihorizon.py`
- structural results: `.perf_runs/edge0-multihorizon-study-v1/results.json`
- Edge0 horizon results: `.perf_runs/edge0-multihorizon-study-v1/edge0-horizons.json`
