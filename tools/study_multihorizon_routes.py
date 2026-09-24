#!/usr/bin/env python3
"""Offline multi-horizon route predictability study for Logan K4 traces.

No model training occurs here.

Questions:
1. How quickly does same-layer K4 overlap decay at +1/+2/+4/+8 tokens?
2. How compact is the *oracle* future expert working set over 2/4/8 tokens?
3. How well do simple past-only baselines cover future expert use?
4. Can RouteScout-style transition tables, fit on complete training runs only,
   predict exact future routes or future working sets on held-out runs?
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MAGIC = b"E0TRC001"
HEADER_BYTES = 32
EXPERTS = 256
HIDDEN = 2048
OWNER_FIRST = 6
OWNER_LAST = 37
HORIZONS = (1, 2, 4, 8)
BUDGETS = (4, 6, 8, 12, 16)
THRESHOLDS = (0.80, 0.90, 0.95)


def load_current_routes(path: Path):
    raw = path.read_bytes()[:HEADER_BYTES]
    magic, version, owner, hidden, experts, k, header_bytes, record_bytes, reserved = struct.unpack(
        "<8sIHHHHIII", raw
    )
    if magic != MAGIC or version != 1 or header_bytes != HEADER_BYTES or reserved != 0:
        raise ValueError(f"{path}: bad header")
    if hidden != HIDDEN or experts != EXPERTS:
        raise ValueError(f"{path}: unexpected geometry")
    expected = 16 + 2 * hidden + 8 * k
    if record_bytes != expected:
        raise ValueError(f"{path}: record_bytes={record_bytes}, expected={expected}")
    payload = path.stat().st_size - HEADER_BYTES
    if payload <= 0 or payload % record_bytes:
        raise ValueError(f"{path}: malformed payload")
    n = payload // record_bytes
    mm = np.memmap(path, mode="r", dtype=np.uint8, offset=HEADER_BYTES, shape=(n, record_bytes))
    run_id = np.ascontiguousarray(mm[:, 0:8]).view("<u8").reshape(-1)
    generation = np.ascontiguousarray(mm[:, 8:16]).view("<u8").reshape(-1)
    off = 16 + hidden * 2
    current = np.ascontiguousarray(mm[:, off : off + 2 * k]).view("<u2").reshape(n, k)
    return int(owner), int(k), run_id, generation, current.astype(np.int16)


def load_dataset(trace_dir: Path):
    by_layer: dict[int, dict[int, list[tuple[int, tuple[int, ...]]]]] = {}
    ks = set()
    all_runs = set()
    for owner in range(OWNER_FIRST, OWNER_LAST + 1):
        path = trace_dir / f"owner-{owner:02}.e0trace"
        owner2, k, run_ids, generations, current = load_current_routes(path)
        assert owner2 == owner
        ks.add(k)
        layer_runs: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
        for rid, gen, route in zip(run_ids, generations, current):
            route_t = tuple(int(x) for x in route)
            layer_runs[int(rid)].append((int(gen), route_t))
            all_runs.add(int(rid))
        for rid in layer_runs:
            layer_runs[rid].sort(key=lambda x: x[0])
            gens = [g for g, _ in layer_runs[rid]]
            if gens != list(range(gens[0], gens[0] + len(gens))):
                raise ValueError(f"layer {owner} run {rid}: noncontiguous generations")
        by_layer[owner] = layer_runs
    if len(ks) != 1:
        raise ValueError(f"mixed K dataset: {ks}")
    return by_layer, ks.pop(), sorted(all_runs)


def mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def percentile(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def split_runs(runs, seed=20260923, val_runs=2):
    arr = np.array(runs, dtype=np.uint64)
    rng = np.random.default_rng(seed)
    rng.shuffle(arr)
    val = set(int(x) for x in arr[:val_runs])
    train = set(int(x) for x in arr[val_runs:])
    return train, val


def overlap_metrics(by_layer, runs, k):
    out = {}
    for h in HORIZONS:
        recalls, exact = [], []
        for layer, layer_runs in by_layer.items():
            for rid, seq in layer_runs.items():
                if rid not in runs:
                    continue
                routes = [r for _, r in seq]
                for t in range(0, len(routes) - h):
                    a, b = set(routes[t]), set(routes[t + h])
                    inter = len(a & b)
                    recalls.append(inter / k)
                    exact.append(1.0 if a == b else 0.0)
        out[str(h)] = {
            "same_route_recall": mean(recalls),
            "exact_set_repeat": mean(exact),
            "samples": len(recalls),
        }
    return out


def future_window_stats(by_layer, runs, k):
    out = {}
    for h in HORIZONS:
        union_sizes = []
        oracle_needed = {thr: [] for thr in THRESHOLDS}
        current_cover = []
        for layer_runs in by_layer.values():
            for rid, seq in layer_runs.items():
                if rid not in runs:
                    continue
                routes = [r for _, r in seq]
                for t in range(0, len(routes) - h):
                    window = routes[t + 1 : t + 1 + h]
                    events = [e for route in window for e in route]
                    counts = Counter(events)
                    union_sizes.append(len(counts))
                    cur = set(routes[t])
                    current_cover.append(sum(1 for e in events if e in cur) / len(events))
                    freqs = sorted(counts.values(), reverse=True)
                    for thr in THRESHOLDS:
                        need_events = math.ceil(thr * len(events) - 1e-12)
                        acc = 0
                        n = 0
                        for c in freqs:
                            acc += c
                            n += 1
                            if acc >= need_events:
                                break
                        oracle_needed[thr].append(n)
        out[str(h)] = {
            "union_mean": mean(union_sizes),
            "union_p50": percentile(union_sizes, 50),
            "union_p90": percentile(union_sizes, 90),
            "current_route_event_coverage": mean(current_cover),
            "oracle_budget": {
                str(thr): {
                    "mean": mean(vals),
                    "p50": percentile(vals, 50),
                    "p90": percentile(vals, 90),
                }
                for thr, vals in oracle_needed.items()
            },
            "windows": len(union_sizes),
        }
    return out


def recent_frequency_working_set(by_layer, runs, k, history=4):
    result = {}
    for h in HORIZONS:
        budget_cov = {b: [] for b in BUDGETS}
        for layer_runs in by_layer.values():
            for rid, seq in layer_runs.items():
                if rid not in runs:
                    continue
                routes = [r for _, r in seq]
                for t in range(history - 1, len(routes) - h):
                    hist = routes[t - history + 1 : t + 1]
                    scores = Counter()
                    last_seen = {}
                    for age, route in enumerate(hist):
                        for e in route:
                            scores[e] += 1
                            last_seen[e] = age
                    ranked = sorted(scores, key=lambda e: (-scores[e], -last_seen[e], e))
                    events = [e for route in routes[t + 1 : t + 1 + h] for e in route]
                    for b in BUDGETS:
                        chosen = set(ranked[:b])
                        budget_cov[b].append(sum(1 for e in events if e in chosen) / len(events))
        result[str(h)] = {
            str(b): mean(vals) for b, vals in budget_cov.items()
        }
    return result


class TransitionModel:
    def __init__(self, layers):
        self.layers = layers
        self.temporal = {l: [Counter() for _ in range(EXPERTS)] for l in layers}
        self.temporal_n = {l: np.zeros(EXPERTS, dtype=np.int64) for l in layers}
        self.spatial = {l: [Counter() for _ in range(EXPERTS)] for l in layers}
        self.spatial_n = {l: np.zeros(EXPERTS, dtype=np.int64) for l in layers}

    def add(self, layer, temporal_src, spatial_src, dest_events):
        for s in temporal_src:
            self.temporal_n[layer][s] += 1
            for d in dest_events:
                self.temporal[layer][s][d] += 1
        for s in spatial_src:
            self.spatial_n[layer][s] += 1
            for d in dest_events:
                self.spatial[layer][s][d] += 1

    def rank(self, layer, temporal_src, spatial_src):
        ts = np.zeros(EXPERTS, dtype=np.float64)
        ss = np.zeros(EXPERTS, dtype=np.float64)
        for s in temporal_src:
            n = self.temporal_n[layer][s]
            if n:
                for d, c in self.temporal[layer][s].items():
                    ts[d] += c / n
        for s in spatial_src:
            n = self.spatial_n[layer][s]
            if n:
                for d, c in self.spatial[layer][s].items():
                    ss[d] += c / n
        # Same peak-normalized fusion as RouteScout; temporal_weight defaults 0.25.
        tp = ts.max()
        sp = ss.max()
        scores = np.zeros(EXPERTS, dtype=np.float64)
        if tp > 0:
            scores += 0.25 * ts / tp
        if sp > 0:
            scores += ss / sp
        ranked = np.argsort(-scores, kind="stable")
        return [int(x) for x in ranked if scores[x] > 0]


def aligned_routes(by_layer, rid):
    layers = sorted(by_layer)
    n = min(len(by_layer[l][rid]) for l in layers if rid in by_layer[l])
    return layers, {
        l: [r for _, r in by_layer[l][rid]][:n]
        for l in layers
    }, n


def train_horizon_model(by_layer, train_runs, h, window=False):
    layers = sorted(by_layer)
    model = TransitionModel(layers)
    for rid in train_runs:
        if not all(rid in by_layer[l] for l in layers):
            continue
        _, routes, n = aligned_routes(by_layer, rid)
        for t in range(0, n - h):
            for l in layers:
                temporal_src = routes[l][t]
                spatial_src = routes[l - 1][t] if l - 1 in routes else ()
                if window:
                    dest = [e for u in range(1, h + 1) for e in routes[l][t + u]]
                else:
                    dest = list(routes[l][t + h])
                model.add(l, temporal_src, spatial_src, dest)
    return model


def eval_transition_models(by_layer, train_runs, val_runs, k):
    exact = {}
    working = {}
    layers = sorted(by_layer)
    for h in HORIZONS:
        model = train_horizon_model(by_layer, train_runs, h, window=False)
        rec = {b: [] for b in BUDGETS}
        full4 = []
        for rid in val_runs:
            if not all(rid in by_layer[l] for l in layers):
                continue
            _, routes, n = aligned_routes(by_layer, rid)
            for t in range(0, n - h):
                for l in layers:
                    ranked = model.rank(l, routes[l][t], routes[l - 1][t] if l - 1 in routes else ())
                    target = set(routes[l][t + h])
                    for b in BUDGETS:
                        chosen = set(ranked[:b])
                        rec[b].append(len(chosen & target) / k)
                    full4.append(1.0 if target.issubset(set(ranked[:k])) else 0.0)
        exact[str(h)] = {
            "recall": {str(b): mean(vals) for b, vals in rec.items()},
            "full_target_at_k": mean(full4),
        }

        modelw = train_horizon_model(by_layer, train_runs, h, window=True)
        cov = {b: [] for b in BUDGETS}
        for rid in val_runs:
            if not all(rid in by_layer[l] for l in layers):
                continue
            _, routes, n = aligned_routes(by_layer, rid)
            for t in range(0, n - h):
                for l in layers:
                    ranked = modelw.rank(l, routes[l][t], routes[l - 1][t] if l - 1 in routes else ())
                    events = [e for u in range(1, h + 1) for e in routes[l][t + u]]
                    for b in BUDGETS:
                        chosen = set(ranked[:b])
                        cov[b].append(sum(1 for e in events if e in chosen) / len(events))
        working[str(h)] = {str(b): mean(vals) for b, vals in cov.items()}
    return exact, working


def layer_overlap(by_layer, runs, k, h=1):
    vals = []
    for layer, layer_runs in by_layer.items():
        xs = []
        for rid, seq in layer_runs.items():
            if rid not in runs:
                continue
            routes = [r for _, r in seq]
            for t in range(len(routes) - h):
                xs.append(len(set(routes[t]) & set(routes[t + h])) / k)
        vals.append((layer, mean(xs)))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    by_layer, k, runs = load_dataset(args.trace_dir)
    if k != 4:
        raise SystemExit(f"study requested canonical K4 corpus; got K={k}")
    train_runs, val_runs = split_runs(runs)

    all_runs = set(runs)
    overlap = overlap_metrics(by_layer, all_runs, k)
    future = future_window_stats(by_layer, all_runs, k)
    recent = recent_frequency_working_set(by_layer, all_runs, k)
    trans_exact, trans_work = eval_transition_models(by_layer, train_runs, val_runs, k)
    per_layer_h1 = layer_overlap(by_layer, all_runs, k, 1)

    result = {
        "format": "logan-k4-multihorizon-study-v1",
        "trace_dir": str(args.trace_dir),
        "k": k,
        "layers": [OWNER_FIRST, OWNER_LAST],
        "runs": len(runs),
        "train_runs": sorted(train_runs),
        "val_runs": sorted(val_runs),
        "horizons": list(HORIZONS),
        "budgets": list(BUDGETS),
        "overlap": overlap,
        "future_windows": future,
        "recent_frequency": recent,
        "routescout_style_exact": trans_exact,
        "routescout_style_working_set": trans_work,
        "per_layer_h1_overlap": [{"layer": l, "recall": v} for l, v in per_layer_h1],
    }

    out = args.output or args.trace_dir / "multihorizon-study-v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"STUDY k={k} runs={len(runs)} train={len(train_runs)} val={len(val_runs)}")
    print("HORIZON same_route_recall exact_repeat union_mean current_window_cover")
    for h in HORIZONS:
        o = overlap[str(h)]
        f = future[str(h)]
        print(
            f"+{h} {o['same_route_recall']:.4f} {o['exact_set_repeat']:.4f} "
            f"{f['union_mean']:.3f} {f['current_route_event_coverage']:.4f}"
        )
    print("ORACLE budgets mean needed for 80/90/95% of future selection events")
    for h in HORIZONS:
        f = future[str(h)]["oracle_budget"]
        print(
            f"H={h} "
            + " ".join(f"{int(float(th)*100)}%={f[th]['mean']:.2f}" for th in map(str, THRESHOLDS))
        )
    print("HELDOUT RouteScout-style exact-route recall topB")
    for h in HORIZONS:
        x = trans_exact[str(h)]["recall"]
        print(f"+{h} " + " ".join(f"B{b}={x[str(b)]:.4f}" for b in BUDGETS))
    print("HELDOUT RouteScout-style future-window event coverage")
    for h in HORIZONS:
        x = trans_work[str(h)]
        print(f"H={h} " + " ".join(f"B{b}={x[str(b)]:.4f}" for b in BUDGETS))
    print("RECENT4 future-window event coverage")
    for h in HORIZONS:
        x = recent[str(h)]
        print(f"H={h} " + " ".join(f"B{b}={x[str(b)]:.4f}" for b in BUDGETS))
    print(f"OUTPUT {out}")


if __name__ == "__main__":
    main()
