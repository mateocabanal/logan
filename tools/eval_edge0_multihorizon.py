#!/usr/bin/env python3
"""Evaluate existing Edge0 adapters against farther-future K4 labels.

No optimization/training is performed. Uses the same per-head run-level validation
split as train_edge0_router.py, then holds the input at token t fixed while
shifting the native consumer target from t+1 to t+2/t+4/t+8.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def import_trainer():
    spec = importlib.util.spec_from_file_location("edge0_train", "tools/train_edge0_router.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--adapter", action="append", required=True, help="name=path")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    args = ap.parse_args()

    tr = import_trainer()
    adapters = {}
    for item in args.adapter:
        name, path = item.split("=", 1)
        adapters[name] = tr.load_adapter(Path(path))

    horizons = (1, 2, 4, 8)
    budgets = (4, 8, 12, 16)
    acc = {
        name: {
            h: {b: [] for b in budgets}
            for h in horizons
        }
        for name in adapters
    }
    mass = {
        name: {
            h: {b: [] for b in budgets}
            for h in horizons
        }
        for name in adapters
    }
    full = {name: {h: [] for h in horizons} for name in adapters}
    window_cov = {name: {h: {b: [] for b in budgets} for h in horizons} for name in adapters}
    sample_counts = {h: 0 for h in horizons}

    for owner in range(tr.OWNER_FIRST, tr.OWNER_LAST + 1):
        trace = tr.load_trace(args.trace_dir / f"owner-{owner:02}.e0trace")
        _, val_ix, _ = tr.split_indices(trace.run_id, args.val_fraction, args.seed + owner)
        val_set = set(int(x) for x in val_ix)
        # Index each run in generation order so a target shift never crosses runs.
        by_run = {}
        for i, rid in enumerate(trace.run_id):
            by_run.setdefault(int(rid), []).append(i)
        for rid in by_run:
            by_run[rid].sort(key=lambda i: int(trace.generation[i]))

        params_by_name = {}
        prefix = f"layers.{owner}"
        for name, base in adapters.items():
            params_by_name[name] = [
                mx.array(base[f"{prefix}.fc1.weight"].astype(np.float32)),
                mx.array(base[f"{prefix}.fc2.weight"].astype(np.float32)),
                mx.array(base[f"{prefix}.linear_init.weight"].astype(np.float32)),
            ]

        # Batch all valid inputs per horizon for this head.
        for h in horizons:
            pairs = []
            for rid, seq in by_run.items():
                for pos in range(0, len(seq) - h + 1):
                    input_i = seq[pos]
                    target_i = seq[pos + h - 1]  # record j targets consumer at generation j+1
                    if input_i in val_set:
                        pairs.append((input_i, target_i))
            if not pairs:
                continue
            input_ix = np.array([a for a, _ in pairs], dtype=np.int64)
            target_ix = np.array([b for _, b in pairs], dtype=np.int64)
            x, _, _ = tr.make_batch(trace, input_ix)
            target = trace.target[target_ix].astype(np.int64, copy=False)
            tw = np.asarray(trace.target_weights[target_ix], dtype=np.float32)
            tw = tw / np.maximum(tw.sum(axis=1, keepdims=True), 1e-8)
            sample_counts[h] += len(pairs)

            for name, params in params_by_name.items():
                logits = tr.logits_for(params, x)
                pred16 = mx.argpartition(logits, kth=-16, axis=1)[:, -16:]
                # Argpartition isn't sorted, but nested set membership is all we need.
                mx.eval(pred16)
                p12 = np.asarray(pred16)
                # recover actual top-B by sorting the selected 16 logits
                log_np = np.asarray(logits)
                for row_logits, row_candidates, row_t, row_w in zip(log_np, p12, target, tw):
                    ordered = sorted((int(e) for e in row_candidates), key=lambda e: -float(row_logits[e]))
                    tset = set(int(e) for e in row_t)
                    for b in budgets:
                        chosen = set(ordered[:b])
                        acc[name][h][b].append(len(chosen & tset) / trace.k)
                        mass[name][h][b].append(
                            sum(float(row_w[j]) for j, e in enumerate(row_t) if int(e) in chosen)
                        )
                    full[name][h].append(1.0 if tset.issubset(set(ordered[:trace.k])) else 0.0)

                # Treat the same ranked prediction as a retained working set over
                # consumer routes t+1..t+h; this asks whether +1 logits remain
                # useful for a multi-token residency policy without retraining.
                for row_i, (input_i, _) in enumerate(pairs):
                    rid = int(trace.run_id[input_i])
                    seq = by_run[rid]
                    pos = seq.index(input_i)
                    future_ix = [seq[pos + u] for u in range(0, h)]
                    events = [int(e) for j in future_ix for e in trace.target[j]]
                    ordered = sorted(
                        (int(e) for e in p12[row_i]),
                        key=lambda e: -float(log_np[row_i, e]),
                    )
                    for b in budgets:
                        chosen = set(ordered[:b])
                        window_cov[name][h][b].append(
                            sum(e in chosen for e in events) / len(events)
                        )

    result = {
        "format": "edge0-existing-adapter-multihorizon-v1",
        "note": "No training. Existing t+1 logits scored against farther-future held-out native K4 targets.",
        "sample_counts": {str(h): sample_counts[h] for h in horizons},
        "adapters": {},
    }
    for name in adapters:
        result["adapters"][name] = {}
        for h in horizons:
            result["adapters"][name][str(h)] = {
                "recall": {str(b): float(np.mean(acc[name][h][b])) for b in budgets},
                "weighted_mass": {str(b): float(np.mean(mass[name][h][b])) for b in budgets},
                "full_k4_at_4": float(np.mean(full[name][h])),
                "window_event_coverage": {
                    str(b): float(np.mean(window_cov[name][h][b])) for b in budgets
                },
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    for name in adapters:
        print(name)
        for h in horizons:
            x = result["adapters"][name][str(h)]
            print(
                f"  +{h}: recall@4={x['recall']['4']:.4f} "
                f"mass@4={x['weighted_mass']['4']:.4f} "
                f"recall@8={x['recall']['8']:.4f} "
                f"windowB12={x['window_event_coverage']['12']:.4f} "
                f"full4={x['full_k4_at_4']:.4f}"
            )
    print(f"OUTPUT {args.output}")


if __name__ == "__main__":
    main()
