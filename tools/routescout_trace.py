#!/usr/bin/env python3
"""Analyze RouteScout v1 routing traces without external dependencies.

The trace is intentionally simple TSV so experiments remain inspectable:
  event layer entropy margin wsum expert:normalized_weight ...

Only complete, monotonically ordered 0..layers-1 cycles are admitted. This
makes partially-written traces and layer-major/broken traversals fail closed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict


@dataclass(frozen=True)
class Event:
    event: int
    layer: int
    entropy: float
    margin: float
    wsum: float
    ids: tuple[int, ...]
    weights: tuple[float, ...]


def parse(path: Path):
    lines = path.read_text().splitlines()
    if not lines or not lines[0].startswith("# routescout-v1\t"):
        raise SystemExit("not a routescout-v1 trace")
    meta = {}
    for item in lines[0].split("\t")[1:]:
        k, v = item.split("=", 1)
        meta[k] = int(v)
    layers = meta["layers"]
    experts = meta["experts"]
    topk = meta["topk"]
    events = []
    for lineno, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 5 + topk:
            raise SystemExit(f"{path}:{lineno}: expected {5+topk} fields, got {len(fields)}")
        pairs = [x.split(":", 1) for x in fields[5:]]
        ev = Event(
            int(fields[0]), int(fields[1]), float(fields[2]), float(fields[3]),
            float(fields[4]), tuple(int(p[0]) for p in pairs),
            tuple(float(p[1]) for p in pairs),
        )
        if any(e < 0 or e >= experts for e in ev.ids):
            raise SystemExit(f"{path}:{lineno}: expert outside 0..{experts-1}")
        events.append(ev)

    cycles = []
    i = 0
    while i + layers <= len(events):
        chunk = events[i:i+layers]
        if [e.layer for e in chunk] == list(range(layers)):
            cycles.append(chunk)
            i += layers
        else:
            i += 1
    return meta, events, cycles


def recall(pred, actual):
    if not actual:
        return 0.0
    return len(set(pred) & set(actual)) / len(set(actual))


def ranked(scores, k):
    return [e for e, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def update_transitions(table, source_ids, target_ids):
    for src in source_ids:
        row = table[src]
        for dst in target_ids:
            row[dst] += 1


def transition_scores(table, source_ids, experts):
    scores = defaultdict(float)
    for src in source_ids:
        row = table.get(src)
        if not row:
            continue
        total = sum(row.values())
        if total:
            inv = 1.0 / total
            for dst, count in row.items():
                scores[dst] += count * inv
    return scores


def evaluate(meta, cycles):
    layers, experts, topk = meta["layers"], meta["experts"], meta["topk"]
    if len(cycles) < 3:
        raise SystemExit(f"need >=3 complete cycles, got {len(cycles)}")

    # Last max(2, 30%) complete cycles are held out entirely.
    test_n = max(2, round(len(cycles) * 0.30))
    test_n = min(test_n, len(cycles) - 1)
    train = cycles[:-test_n]
    test = cycles[-test_n:]

    temporal_overlap = []
    spatial_overlap = []
    temporal_spatial_union = []
    for t in range(1, len(cycles)):
        for l in range(layers):
            temporal_overlap.append(recall(cycles[t-1][l].ids, cycles[t][l].ids))
            if l:
                spatial_overlap.append(recall(cycles[t][l-1].ids, cycles[t][l].ids))
                temporal_spatial_union.append(
                    recall(cycles[t-1][l].ids + cycles[t][l-1].ids, cycles[t][l].ids)
                )

    # Per-layer route-frequency baseline.
    freq = [defaultdict(int) for _ in range(layers)]
    # Per-layer temporal transition P(route_t | route_t-1).
    temporal = [defaultdict(lambda: defaultdict(int)) for _ in range(layers)]
    # Per edge spatial transition P(route_L+1 | route_L), same token.
    spatial = [defaultdict(lambda: defaultdict(int)) for _ in range(layers - 1)]

    for t, cyc in enumerate(train):
        for l, ev in enumerate(cyc):
            for e in ev.ids:
                freq[l][e] += 1
            if t:
                update_transitions(temporal[l], train[t-1][l].ids, ev.ids)
            if l + 1 < layers:
                update_transitions(spatial[l], ev.ids, cyc[l+1].ids)

    metrics = {name: {8: [], 16: [], 24: []} for name in (
        "frequency", "temporal_transition", "spatial_transition", "hybrid_transition"
    )}

    prev = train[-1]
    for cyc in test:
        for l, ev in enumerate(cyc):
            fs = {e: float(c) for e, c in freq[l].items()}
            ts = transition_scores(temporal[l], prev[l].ids, experts)
            ss = transition_scores(spatial[l-1], cyc[l-1].ids, experts) if l else {}
            # Normalize each evidence source by its own max before combining.
            def norm(s):
                m = max(s.values(), default=0.0)
                return {k: v/m for k, v in s.items()} if m > 0 else {}
            nts, nss, nfs = norm(ts), norm(ss), norm(fs)
            hs = defaultdict(float)
            for src, scale in ((nts, 1.0), (nss, 1.0), (nfs, 0.25)):
                for e, v in src.items():
                    hs[e] += scale * v
            for k in (8, 16, 24):
                metrics["frequency"][k].append(recall(ranked(fs, k), ev.ids))
                metrics["temporal_transition"][k].append(recall(ranked(ts, k), ev.ids))
                metrics["spatial_transition"][k].append(recall(ranked(ss, k), ev.ids))
                metrics["hybrid_transition"][k].append(recall(ranked(hs, k), ev.ids))
        prev = cyc

    entropies = [e.entropy for c in cycles for e in c]
    margins = [e.margin for c in cycles for e in c]
    wsums = [e.wsum for c in cycles for e in c]

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    print(f"cycles={len(cycles)} events={len(cycles)*layers} train={len(train)} test={len(test)}")
    print(f"geometry=layers:{layers} experts:{experts} topk:{topk}")
    print(f"router mean_entropy={mean(entropies):.4f} mean_top1_margin={mean(margins):.5f} mean_topk_mass={mean(wsums):.4f}")
    print(f"carry temporal_same_layer_R@8={mean(temporal_overlap):.4f}")
    print(f"carry spatial_prev_layer_R@8={mean(spatial_overlap):.4f}")
    print(f"carry union_prev_token+prev_layer_R@16={mean(temporal_spatial_union):.4f}")
    for name, ks in metrics.items():
        print(name + " " + " ".join(f"R@{k}={mean(vals):.4f}" for k, vals in ks.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    args = ap.parse_args()
    meta, events, cycles = parse(args.trace)
    print(f"raw_events={len(events)} complete_cycles={len(cycles)} discarded={len(events)-len(cycles)*meta['layers']}")
    evaluate(meta, cycles)


if __name__ == "__main__":
    main()
