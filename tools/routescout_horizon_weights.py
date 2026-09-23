#!/usr/bin/env python3
"""RouteScout Phase 3/4: cold-arrival prediction quality for two things the
runtime work made newly actionable.

Why this is offline and why it runs before any further decode A/B: EXP-029's
corrected counters showed that a decode-window effect at this scale (the
addressable MetalIO wait is ~106 ms of a ~650 ms forward) cannot be resolved by
whole-model A/B on this host, where run-to-run spread is +-25%. Candidate
predictor policies therefore have to be *screened* on their prediction statistics
first, and only the ones that clear the byte budget go to the decoder.

Two questions, both cross-prompt (leave-one-prompt-out), both scored on
cold arrivals only (arrivals = actual route minus the previous route at the same
layer, since previous-route experts are already resident and predicting them
cannot hide a load):

1. What relative weight should the temporal and spatial transition evidence carry?
   The runtime sums them with equal weight. Every published cross-prompt result
   says spatial is the stronger prior, which suggests the runtime is
   under-weighting it. Measure at equal speculative byte budget.

2. How fast does prediction quality decay with a *direct spatial horizon*
   (predict target layer L+H from the newest route available before layer L)?
   The horizon table must be trained on the same source distribution it will
   query at runtime: source=target-(H+1) -> target. EXP-036 corrected an older
   version of this tool that accidentally queried an adjacent-layer table with
   expert IDs from a more distant source layer, which understated horizon quality.

Both are reported per prompt, because EXP-014 established that prompt family
dominates every model choice (R@8 spans 0.23-0.64 across families) and a single
average hides that.

Usage:
    tools/routescout_horizon_weights.py TRACE... [--budgets 1,2,4,8] [--horizons 0,1,2,4,8]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from routescout_matrix import Corpus, load_corpus, top_k
from routescout_trace import transition_scores

# Temporal weights paired with the spatial weight held at 1.0. The runtime's
# current behaviour is the first entry; the historical evidence says the last
# entries should be better.
WEIGHT_GRID = ((1.0, 1.0), (0.5, 1.0), (0.25, 1.0), (0.0, 1.0))


def score_layer(
    temporal_tables,
    spatial_tables,
    experts: int,
    layer: int,
    previous: list[int],
    spatial_ids: list[int],
    temporal_weight: float,
    spatial_weight: float,
) -> np.ndarray:
    """Weighted sum of conditional temporal and spatial evidence.

    Both terms are peak-normalised before weighting, so the weight controls the
    *relative influence* of the two signals rather than the scale of whichever
    table happens to hold larger raw counts. Without that normalisation a weight
    sweep is really a sweep over table sizes.
    """
    out = np.zeros(experts, dtype=np.float32)
    term = np.zeros(experts, dtype=np.float32)
    for expert, value in transition_scores(temporal_tables[layer], list(previous), experts).items():
        term[expert] = value
    peak = float(term.max()) if term.size else 0.0
    if peak > 0:
        out += temporal_weight * (term / peak)

    if layer > 0 and spatial_tables[layer - 1] is not None:
        term[:] = 0.0
        for expert, value in transition_scores(
            spatial_tables[layer - 1], list(spatial_ids), experts
        ).items():
            term[expert] = value
        peak = float(term.max()) if term.size else 0.0
        if peak > 0:
            out += spatial_weight * (term / peak)
    return out


def fit_tables(corpora: list[Corpus]) -> tuple[list, list]:
    """Layer-local temporal (same layer) and spatial (layer-1 -> layer) tables.

    Kept as the same `defaultdict` shape the rest of the harness uses so
    `routescout_trace.transition_scores` consumes them unchanged.
    """
    from collections import defaultdict

    from routescout_trace import update_transitions

    layers = corpora[0].layers
    temporal = [defaultdict(lambda: defaultdict(int)) for _ in range(layers)]
    spatial = [defaultdict(lambda: defaultdict(int)) for _ in range(layers - 1)]
    for corpus in corpora:
        cycles = corpus.cycles
        for t in range(1, len(cycles)):
            for layer in range(layers):
                update_transitions(
                    temporal[layer], cycles[t - 1][layer].ids, cycles[t][layer].ids
                )
                if layer + 1 < layers:
                    update_transitions(
                        spatial[layer], cycles[t][layer].ids, cycles[t][layer + 1].ids
                    )
    return temporal, spatial


def fit_direct_horizon_tables(corpora: list[Corpus], horizon: int) -> tuple[list, list]:
    """Temporal target tables plus direct same-token source->target tables.

    At runtime, before layer li executes, the newest same-token route known is
    li-1. If the target is li+horizon, the direct spatial source is therefore
    target-(horizon+1). Training exactly that mapping avoids the EXP-034 bug
    where distant source IDs were scored through an adjacent-layer table.
    """
    from collections import defaultdict
    from routescout_trace import update_transitions

    layers = corpora[0].layers
    temporal = [defaultdict(lambda: defaultdict(int)) for _ in range(layers)]
    direct = [defaultdict(lambda: defaultdict(int)) for _ in range(layers)]
    for corpus in corpora:
        cycles = corpus.cycles
        for t in range(1, len(cycles)):
            for target in range(layers):
                update_transitions(
                    temporal[target],
                    cycles[t - 1][target].ids,
                    cycles[t][target].ids,
                )
                source = target - (horizon + 1)
                if source >= 0:
                    update_transitions(
                        direct[target],
                        cycles[t][source].ids,
                        cycles[t][target].ids,
                    )
    return temporal, direct


def cold_metrics(scores: np.ndarray, actual: list[int], previous: list[int], budget: int) -> tuple[int, int, int]:
    """(hits, issued, arrivals) for a budget-limited cold-arrival prediction."""
    arrivals = set(actual) - set(previous)
    pred = top_k(scores, budget, forbid=set(previous))
    return len(set(pred) & arrivals), len(pred), len(arrivals)


def sweep_weights(corpus: list[Corpus], budgets: list[int]) -> dict:
    out = {}
    for holdout in corpus:
        train = [c for c in corpus if c.name != holdout.name]
        temporal, spatial = fit_tables(train)
        expert_count = holdout.experts
        cycles = holdout.cycles
        per_weight = {}
        for temporal_weight, spatial_weight in WEIGHT_GRID:
            hits = {b: 0 for b in budgets}
            issued = {b: 0 for b in budgets}
            arrivals = {b: 0 for b in budgets}
            for t in range(1, len(cycles)):
                for layer in range(holdout.layers):
                    previous = list(cycles[t - 1][layer].ids)
                    spatial_ids = list(cycles[t][layer - 1].ids) if layer else []
                    actual = list(cycles[t][layer].ids)
                    scores = score_layer(
                        temporal,
                        spatial,
                        expert_count,
                        layer,
                        previous,
                        spatial_ids,
                        temporal_weight,
                        spatial_weight,
                    )
                    for b in budgets:
                        h, i, a = cold_metrics(scores, actual, previous, b)
                        hits[b] += h
                        issued[b] += i
                        arrivals[b] += a
            per_weight[f"t{temporal_weight:g}_s{spatial_weight:g}"] = {
                f"budget_{b}": {
                    "recall": hits[b] / arrivals[b] if arrivals[b] else 0.0,
                    "precision": hits[b] / issued[b] if issued[b] else 0.0,
                    "hits": hits[b],
                    "issued": issued[b],
                    "arrivals": arrivals[b],
                }
                for b in budgets
            }
        out[holdout.name] = per_weight
    return out


def sweep_horizons(corpus: list[Corpus], budgets: list[int], horizons: list[int]) -> dict:
    """Predict layer L+H from evidence available at layer L of the same token.

    For H=0 this is the runtime's current same-layer prediction. For H>0 the
    temporal term uses the previous token's route at the *target* layer (which is
    known before the current token starts) and the spatial term uses the current
    token's already-observed layer L, so no future information is consumed.
    """
    out = {}
    for holdout in corpus:
        train = [c for c in corpus if c.name != holdout.name]
        experts = holdout.experts
        cycles = holdout.cycles
        layers = holdout.layers
        per_horizon = {}
        for horizon in horizons:
            temporal, direct = fit_direct_horizon_tables(train, horizon)
            hits = {b: 0 for b in budgets}
            issued = {b: 0 for b in budgets}
            arrivals = {b: 0 for b in budgets}
            scored = 0
            for t in range(1, len(cycles)):
                cur = cycles[t]
                prev_cycle = cycles[t - 1]
                for layer in range(layers):
                    target = layer + horizon
                    if target >= layers:
                        continue
                    # Temporal evidence about the TARGET layer, from the previous
                    # token's route at that layer: available before this token.
                    temporal_ids = list(prev_cycle[target].ids)
                    # Spatial evidence: the current token's most recent observed
                    # layer. At H=0 that is layer-1; at H>0 it is still the last
                    # layer whose route this token has already produced.
                    spatial_source = layer if layer > 0 else None
                    spatial_ids = None
                    if spatial_source is not None:
                        spatial_ids = list(cur[spatial_source - 1].ids)
                    scores = np.zeros(experts, dtype=np.float32)
                    term = np.zeros(experts, dtype=np.float32)
                    for expert, value in transition_scores(
                        temporal[target], temporal_ids, experts
                    ).items():
                        term[expert] = value
                    peak = float(term.max()) if term.size else 0.0
                    if peak > 0:
                        scores += 0.25 * (term / peak)
                    if spatial_ids and target > 0:
                        term[:] = 0.0
                        for expert, value in transition_scores(
                            direct[target], spatial_ids, experts
                        ).items():
                            term[expert] = value
                        peak = float(term.max()) if term.size else 0.0
                        if peak > 0:
                            scores += term / peak

                    previous = list(prev_cycle[target].ids)
                    actual = list(cur[target].ids)
                    for b in budgets:
                        h, i, a = cold_metrics(scores, actual, previous, b)
                        hits[b] += h
                        issued[b] += i
                        arrivals[b] += a
                    scored += 1
            per_horizon[f"h{horizon}"] = {
                "scored_positions": scored,
                "budgets": {
                    f"budget_{b}": {
                        "recall": hits[b] / arrivals[b] if arrivals[b] else 0.0,
                        "precision": hits[b] / issued[b] if issued[b] else 0.0,
                    }
                    for b in budgets
                },
            }
        out[holdout.name] = per_horizon
    return out


def report_weights(results: dict, budgets: list[int]) -> None:
    print("== temporal/spatial weight sweep (cold arrivals, cross-prompt holdout) ==")
    for budget in budgets:
        print(f"\n  budget {budget}")
        labels = list(next(iter(results.values())).keys())
        print(f"    {'holdout':<28}" + "".join(f"{lab:>16}" for lab in labels))
        for holdout, per_weight in results.items():
            cells = []
            for lab in labels:
                m = per_weight[lab][f"budget_{budget}"]
                cells.append(f"{m['recall']:.4f}/{m['precision']:.3f}")
            print(f"    {holdout:<28}" + "".join(f"{c:>16}" for c in cells))
        # Aggregate: unweighted mean recall and precision across holdouts.
        print(f"    {'MEAN recall':<28}", end="")
        for lab in labels:
            mean_recall = statistics.fmean(
                per_weight[lab][f"budget_{budget}"]["recall"] for per_weight in results.values()
            )
            print(f"{mean_recall:>16.4f}", end="")
        print()
        print(f"    {'MEAN precision':<28}", end="")
        for lab in labels:
            mean_precision = statistics.fmean(
                per_weight[lab][f"budget_{budget}"]["precision"] for per_weight in results.values()
            )
            print(f"{mean_precision:>16.4f}", end="")
        print()


def report_horizons(results: dict, budgets: list[int]) -> None:
    print("\n== spatial horizon sweep (predict layer L+H from evidence at L) ==")
    horizons = list(next(iter(results.values())).keys())
    for budget in budgets:
        print(f"\n  budget {budget}  (recall: how much of the true cold arrivals a "
              f"budget-{budget} prediction still covers)")
        print(f"    {'holdout':<28}" + "".join(f"{h:>12}" for h in horizons))
        for holdout, per_horizon in results.items():
            cells = [
                f"{per_horizon[h]['budgets'][f'budget_{budget}']['recall']:.4f}"
                for h in horizons
            ]
            print(f"    {holdout:<28}" + "".join(f"{c:>12}" for c in cells))
        print(f"    {'MEAN recall':<28}", end="")
        for h in horizons:
            mean_recall = statistics.fmean(
                per_horizon[h]["budgets"][f"budget_{budget}"]["recall"]
                for per_horizon in results.values()
            )
            print(f"{mean_recall:>12.4f}", end="")
        print()
        print(f"    {'MEAN precision':<28}", end="")
        for h in horizons:
            mean_precision = statistics.fmean(
                per_horizon[h]["budgets"][f"budget_{budget}"]["precision"]
                for per_horizon in results.values()
            )
            print(f"{mean_precision:>12.4f}", end="")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--budgets", default="1,2,4,8")
    ap.add_argument("--horizons", default="0,1,2,4,8")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",") if b]
    horizons = [int(h) for h in args.horizons.split(",") if h]
    corpus = load_corpus(args.traces)
    print(
        f"corpus={len(corpus)} layers={corpus[0].layers} experts={corpus[0].experts} "
        f"topk={corpus[0].meta['topk']} prompts={[c.name for c in corpus]}"
    )

    weights = sweep_weights(corpus, budgets)
    report_weights(weights, budgets)
    horizon = sweep_horizons(corpus, budgets, horizons)
    report_horizons(horizon, budgets)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"weights": weights, "horizons": horizon}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
