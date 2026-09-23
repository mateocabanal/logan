#!/usr/bin/env python3
"""RouteScout host feature-construction cost.

The ANE scorer is ~5-100 us/dispatch, which is irrelevant if assembling its input
features costs milliseconds. This times the host-side feature build for the
16-feature candidate layout at the real model's geometry (40 layers, 256 experts,
top-8): one layer's worth of candidates is 256 lanes, and the runtime packs 8
target layers per dispatch.

Reports both the pure feature build and the incremental cost of maintaining the
transition tables it depends on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from routescout_matrix import (  # noqa: E402
    Priors,
    load_corpus,
    scorer_matrix,
    iter_layers_with_history,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    corpus = load_corpus(args.traces)
    priors = Priors.fit(corpus)
    layers = corpus[0].layers
    experts = corpus[0].experts
    print(f"corpus={len(corpus)} layers={layers} experts={experts}")

    # Collect the (layer, histories, spatial) inputs once; the measurement is the
    # feature build, not trace parsing.
    cases = []
    for c in corpus:
        for _t, layer, ids, weights, sids, sweights in iter_layers_with_history(c, 1):
            cases.append((layer, ids, weights, sids, sweights))
    print(f"cases={len(cases)}")

    # Warm up.
    for layer, ids, weights, sids, sweights in cases[:50]:
        scorer_matrix(priors, layer, ids, weights, sids, sweights)

    timings = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        for layer, ids, weights, sids, sweights in cases:
            scorer_matrix(priors, layer, ids, weights, sids, sweights)
        elapsed = time.perf_counter() - started
        timings.append(elapsed / len(cases))

    per_case_us = float(np.median(timings)) * 1e6
    # The runtime scores one candidate set per layer; 40 layers per token.
    per_token_ms = per_case_us * layers / 1e3
    print(f"feature_build_us_per_layer={per_case_us:.1f}")
    print(f"feature_build_ms_per_token={per_token_ms:.3f} (40 layers)")
    print(f"ane_dispatch_us_per_8_layers=80.5 -> per_layer_us=10.07")
    print(
        f"feature_share={per_case_us / (per_case_us + 10.07):.3f} "
        f"(feature build as a fraction of feature+ANE scoring)"
    )

    # Transition-table maintenance: the runtime updates these from observed
    # routes, so it must be cheap relative to a token's decode budget.
    table_costs = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        Priors.fit(corpus)
        table_costs.append(time.perf_counter() - started)
    fit_ms = float(np.median(table_costs)) * 1e3
    print(f"transition_fit_ms_for_{len(corpus)}_prompts={fit_ms:.1f}")

    payload = {
        "layers": layers,
        "experts": experts,
        "cases": len(cases),
        "feature_build_us_per_layer": per_case_us,
        "feature_build_ms_per_token": per_token_ms,
        "ane_us_per_layer": 10.07,
        "feature_share": per_case_us / (per_case_us + 10.07),
        "transition_fit_ms": fit_ms,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
