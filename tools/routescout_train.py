#!/usr/bin/env python3
"""Train RouteScout's fixed 96->96->64->256 no-bias MLP from route traces.

This is intentionally NumPy-only so it runs on a stock Logan development Mac.
Every feature is available before the authoritative router for the target layer:

  0..31   previous-token / same-layer route CountSketch
  32..63  same-token / previous-layer route CountSketch
  64..71  previous-token route gate weights (top-k padded)
  72..79  previous-layer route gate weights (top-k padded)
  80..82  previous-token router entropy, margin, top-k mass
  83..85  previous-layer router entropy, margin, top-k mass
  86..95  layer-position Fourier features (5 sin/cos pairs)

The target is the authoritative current top-k route. The network is trained as a
multi-label classifier; inference ranks raw logits, so the exported graph exactly
matches logan_ane::mil::route_scout_mlp_fp16 (no sigmoid is required on ANE).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from routescout_trace import Event, parse, ranked, recall, transition_scores, update_transitions

IN_FEATURES = 96
HIDDEN = 96
LATENT = 64
FEATURE_VERSIONS = {"countsketch": "routescout-features-v1", "svd": "routescout-features-v2-svd5"}


@dataclass
class SampleMeta:
    temporal_ids: tuple[int, ...]
    target_ids: tuple[int, ...]


def _norm_router_stats(ev: Event | None, experts: int) -> tuple[float, float, float]:
    if ev is None:
        return (0.0, 0.0, 0.0)
    # Entropy is bounded by ln(experts); margin and selected mass are already
    # probabilities/masses. Clamp defensively for malformed/experimental traces.
    entropy_scale = max(math.log(max(experts, 2)), 1e-6)
    return (
        float(np.clip(ev.entropy / entropy_scale, 0.0, 1.5)),
        float(np.clip(ev.margin, 0.0, 1.0)),
        float(np.clip(ev.wsum, 0.0, 1.0)),
    )


def _sketch(ev: Event | None, width: int = 32) -> np.ndarray:
    out = np.zeros(width, dtype=np.float32)
    if ev is None:
        return out
    # Two deterministic CountSketch projections per routed expert. The pair of
    # buckets carries much more identity than e % width while remaining trivial
    # to reproduce in Rust before dispatching the ANE island.
    for expert, weight in zip(ev.ids, ev.weights):
        e = int(expert) & 0xFFFFFFFF
        h1 = ((e * 0x9E3779B1) & 0xFFFFFFFF) >> 27
        h2 = ((e * 0x85EBCA6B + 0xC2B2AE35) & 0xFFFFFFFF) >> 27
        s1 = 1.0 if ((e * 0x27D4EB2D) & 0x80000000) == 0 else -1.0
        s2 = 1.0 if ((e * 0x165667B1 + 17) & 0x40000000) == 0 else -1.0
        out[h1 % width] += np.float32(weight * s1)
        out[h2 % width] += np.float32(weight * 0.5 * s2)
    return out


def build_svd_embedding(groups: list[list[list[Event]]], experts: int, dims: int = 5) -> np.ndarray:
    """Compress training-only expert->future-route distributions to a tiny table."""
    transitions = np.zeros((experts, experts), dtype=np.float32)
    seen = np.zeros(experts, dtype=np.float32)

    def add(source: Event, target: Event):
        for src in source.ids:
            if not (0 <= src < experts):
                continue
            seen[src] += 1.0
            for dst, weight in zip(target.ids, target.weights):
                if 0 <= dst < experts:
                    transitions[src, dst] += np.float32(weight)

    for cycles in groups:
        for t, cyc in enumerate(cycles):
            for layer, ev in enumerate(cyc):
                if t:
                    add(cycles[t - 1][layer], ev)
                if layer:
                    add(cyc[layer - 1], ev)

    row_sum = transitions.sum(axis=1, keepdims=True)
    normalized = np.divide(
        transitions,
        row_sum,
        out=np.zeros_like(transitions),
        where=row_sum > 0,
    )
    u, s, _ = np.linalg.svd(normalized, full_matrices=False)
    embedding = (u[:, :dims] * s[:dims]).astype(np.float32)
    # Give each retained dimension comparable numeric scale for the MLP. The
    # transform is fit on training-only routes and exported for host lookup.
    std = embedding.std(axis=0, keepdims=True)
    embedding = np.divide(
        embedding,
        std,
        out=np.zeros_like(embedding),
        where=std > 1e-8,
    )
    embedding = np.clip(embedding, -4.0, 4.0)
    print(
        f"svd_embedding_seen={int(np.count_nonzero(seen))}/{experts} "
        f"rank={dims} explained_energy={float(np.sum(s[:dims] ** 2) / max(np.sum(s ** 2), 1e-12)):.4f}"
    )
    return embedding


def features(
    temporal: Event,
    spatial: Event | None,
    layer: int,
    layers: int,
    topk: int,
    experts: int,
    feature_mode: str,
    expert_embedding: np.ndarray | None,
) -> np.ndarray:
    if topk > 8:
        raise ValueError("RouteScout feature layout supports top-k <= 8")
    x = np.zeros(IN_FEATURES, dtype=np.float32)

    if feature_mode == "countsketch":
        x[0:32] = _sketch(temporal)
        x[32:64] = _sketch(spatial)
        x[64 : 64 + topk] = np.asarray(temporal.weights, dtype=np.float32)
        if spatial is not None:
            x[72 : 72 + topk] = np.asarray(spatial.weights, dtype=np.float32)
    elif feature_mode == "svd":
        if expert_embedding is None or expert_embedding.shape != (experts, 5):
            raise ValueError("RouteScout SVD features require an [experts,5] embedding")
        # Eight rank slots x five values = 40 dimensions per route. Scaling by
        # top-k*gate_weight preserves router confidence while keeping ordinary
        # selected experts near unit scale.
        for rank, (expert, weight) in enumerate(zip(temporal.ids[:8], temporal.weights[:8])):
            x[rank * 5 : rank * 5 + 5] = expert_embedding[expert] * np.float32(weight * topk)
        if spatial is not None:
            for rank, (expert, weight) in enumerate(zip(spatial.ids[:8], spatial.weights[:8])):
                off = 40 + rank * 5
                x[off : off + 5] = expert_embedding[expert] * np.float32(weight * topk)
    else:
        raise ValueError(f"unknown feature mode: {feature_mode}")

    x[80:83] = _norm_router_stats(temporal, experts)
    x[83:86] = _norm_router_stats(spatial, experts)

    # Five Fourier scales encode layer identity without a learned lookup table.
    phase = (layer + 0.5) / max(layers, 1)
    off = 86
    for freq in (1.0, 2.0, 4.0, 8.0, 16.0):
        angle = 2.0 * math.pi * freq * phase
        x[off] = math.sin(angle)
        x[off + 1] = math.cos(angle)
        off += 2
    return x


def make_samples(
    meta: dict,
    cycles: list[list[Event]],
    feature_mode: str,
    expert_embedding: np.ndarray | None,
    target_mode: str,
):
    layers, experts, topk = meta["layers"], meta["experts"], meta["topk"]
    xs, ys, smeta = [], [], []
    for t in range(1, len(cycles)):
        prev = cycles[t - 1]
        cur = cycles[t]
        for layer in range(layers):
            temporal = prev[layer]
            spatial = cur[layer - 1] if layer else None
            target = cur[layer]
            xs.append(
                features(
                    temporal,
                    spatial,
                    layer,
                    layers,
                    topk,
                    experts,
                    feature_mode,
                    expert_embedding,
                )
            )
            y = np.zeros(experts, dtype=np.float32)
            previous_ids = set(temporal.ids)
            for expert in target.ids:
                if target_mode == "route" or expert not in previous_ids:
                    y[expert] = 1.0
            ys.append(y)
            smeta.append(SampleMeta(tuple(temporal.ids), tuple(target.ids)))
    if not xs:
        return (
            np.empty((0, IN_FEATURES), dtype=np.float32),
            np.empty((0, experts), dtype=np.float32),
            [],
        )
    return np.stack(xs), np.stack(ys), smeta


def split_traces(paths: list[Path], feature_mode: str, target_mode: str):
    parsed = [parse(p) for p in paths]
    geometry = [tuple(x[0][k] for k in ("layers", "experts", "topk")) for x in parsed]
    if len(set(geometry)) != 1:
        raise SystemExit(f"trace geometry mismatch: {geometry}")
    meta = parsed[0][0]

    if len(parsed) >= 2:
        train_cycles = [p[2] for p in parsed[:-1]]
        test_cycles = [parsed[-1][2]]
        split_name = f"held-out-trace:{paths[-1].name}"
    else:
        cycles = parsed[0][2]
        if len(cycles) < 5:
            raise SystemExit(f"need >=5 complete cycles, got {len(cycles)}")
        test_n = max(2, round(len(cycles) * 0.30))
        test_n = min(test_n, len(cycles) - 2)
        train_cycles = [cycles[:-test_n]]
        # Keep the immediately preceding cycle as inference state for the first
        # held-out target, but never train on held-out targets.
        test_cycles = [cycles[-test_n - 1 :]]
        split_name = "temporal-70/30"

    expert_embedding = (
        build_svd_embedding(train_cycles, meta["experts"])
        if feature_mode == "svd"
        else None
    )

    def merge(groups):
        x_all, y_all, m_all = [], [], []
        for cycles in groups:
            x, y, m = make_samples(
                meta, cycles, feature_mode, expert_embedding, target_mode
            )
            if len(x):
                x_all.append(x)
                y_all.append(y)
                m_all.extend(m)
        return np.concatenate(x_all), np.concatenate(y_all), m_all

    return meta, split_name, merge(train_cycles), merge(test_cycles), expert_embedding


def heldout_transition_baseline(paths: list[Path]):
    """Evaluate the hand-built baseline on the exact same held-out prompt split."""
    if len(paths) < 2:
        return None
    parsed = [parse(p) for p in paths]
    meta = parsed[0][0]
    layers, experts = meta["layers"], meta["experts"]

    # Never create transitions across prompt/file boundaries.
    freq = [dict() for _ in range(layers)]
    temporal = [defaultdict(lambda: defaultdict(int)) for _ in range(layers)]
    spatial = [defaultdict(lambda: defaultdict(int)) for _ in range(layers - 1)]
    for _, _, cycles in parsed[:-1]:
        for t, cyc in enumerate(cycles):
            for layer, ev in enumerate(cyc):
                row = freq[layer]
                for expert in ev.ids:
                    row[expert] = row.get(expert, 0) + 1
                if t:
                    update_transitions(temporal[layer], cycles[t - 1][layer].ids, ev.ids)
                if layer + 1 < layers:
                    update_transitions(spatial[layer], ev.ids, cyc[layer + 1].ids)

    test = parsed[-1][2]
    if len(test) < 2:
        return None
    metrics = {name: {8: [], 16: [], 24: []} for name in (
        "frequency", "temporal_transition", "spatial_transition", "hybrid_transition"
    )}
    hybrid_rows = []
    for t in range(1, len(test)):
        prev, cyc = test[t - 1], test[t]
        for layer, ev in enumerate(cyc):
            fs = {e: float(c) for e, c in freq[layer].items()}
            ts = transition_scores(temporal[layer], prev[layer].ids, experts)
            ss = transition_scores(spatial[layer - 1], cyc[layer - 1].ids, experts) if layer else {}

            def norm(scores):
                peak = max(scores.values(), default=0.0)
                return {k: v / peak for k, v in scores.items()} if peak > 0 else {}

            hs = defaultdict(float)
            for source, scale in ((norm(ts), 1.0), (norm(ss), 1.0), (norm(fs), 0.25)):
                for expert, value in source.items():
                    hs[expert] += scale * value
            hybrid_rows.append(
                np.asarray([hs.get(expert, 0.0) for expert in range(experts)], dtype=np.float32)
            )
            for k in (8, 16, 24):
                metrics["frequency"][k].append(recall(ranked(fs, k), ev.ids))
                metrics["temporal_transition"][k].append(recall(ranked(ts, k), ev.ids))
                metrics["spatial_transition"][k].append(recall(ranked(ss, k), ev.ids))
                metrics["hybrid_transition"][k].append(recall(ranked(hs, k), ev.ids))

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    summary = {
        name: {f"R@{k}": mean(values) for k, values in ks.items()}
        for name, ks in metrics.items()
    }
    scores = np.stack(hybrid_rows) if hybrid_rows else np.empty((0, experts), dtype=np.float32)
    return summary, scores


def sigmoid(x):
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


class MLP:
    def __init__(self, out_features: int, seed: int):
        rng = np.random.default_rng(seed)
        self.w0 = (rng.standard_normal((HIDDEN, IN_FEATURES)) * math.sqrt(2 / IN_FEATURES)).astype(np.float32)
        self.w1 = (rng.standard_normal((LATENT, HIDDEN)) * math.sqrt(2 / HIDDEN)).astype(np.float32)
        self.w2 = (rng.standard_normal((out_features, LATENT)) * math.sqrt(1 / LATENT)).astype(np.float32)

    def forward(self, x):
        z0 = x @ self.w0.T
        h0 = np.maximum(z0, 0.0)
        z1 = h0 @ self.w1.T
        h1 = np.maximum(z1, 0.0)
        logits = h1 @ self.w2.T
        return z0, h0, z1, h1, logits


def train(model: MLP, x, y, epochs: int, batch: int, lr: float, pos_weight: float, seed: int):
    rng = np.random.default_rng(seed + 1)
    params = [model.w0, model.w1, model.w2]
    ms = [np.zeros_like(p) for p in params]
    vs = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x))
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(x), batch):
            idx = order[start : start + batch]
            xb, yb = x[idx], y[idx]
            z0, h0, z1, h1, logits = model.forward(xb)
            p = sigmoid(logits)
            weights = 1.0 + yb * (pos_weight - 1.0)
            # Stable BCE for reporting only.
            bce = np.maximum(logits, 0) - logits * yb + np.log1p(np.exp(-np.abs(logits)))
            loss_sum += float(np.sum(bce * weights))
            seen += yb.size

            dlogits = (p - yb) * weights / max(len(idx), 1)
            gw2 = dlogits.T @ h1
            dh1 = dlogits @ model.w2
            dz1 = dh1 * (z1 > 0)
            gw1 = dz1.T @ h0
            dh0 = dz1 @ model.w1
            dz0 = dh0 * (z0 > 0)
            gw0 = dz0.T @ xb
            grads = [gw0, gw1, gw2]

            # Global gradient clipping keeps tiny trace experiments stable.
            norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads))
            if norm > 5.0:
                scale = 5.0 / norm
                grads = [g * scale for g in grads]

            step += 1
            for i, (p_arr, g) in enumerate(zip(params, grads)):
                ms[i] = beta1 * ms[i] + (1 - beta1) * g
                vs[i] = beta2 * vs[i] + (1 - beta2) * (g * g)
                mhat = ms[i] / (1 - beta1**step)
                vhat = vs[i] / (1 - beta2**step)
                p_arr -= lr * mhat / (np.sqrt(vhat) + eps)

        if epoch == 1 or epoch % max(epochs // 6, 1) == 0 or epoch == epochs:
            print(f"epoch={epoch:4d} weighted_bce={loss_sum/max(seen,1):.6f}")


def topk_indices(row: np.ndarray, k: int):
    k = min(k, len(row))
    if k <= 0:
        return []
    part = np.argpartition(-row, k - 1)[:k]
    return part[np.argsort(-row[part], kind="stable")].tolist()


def metrics_from_scores(scores: np.ndarray, metas: list[SampleMeta]):
    route = {8: [], 16: [], 24: []}
    cold_precision = {4: [], 8: []}
    cold_recall = {4: [], 8: []}
    useful = {4: 0, 8: 0}
    wasted = {4: 0, 8: 0}

    for row, meta in zip(scores, metas):
        actual = set(meta.target_ids)
        for k in route:
            pred = set(topk_indices(row, k))
            route[k].append(len(pred & actual) / max(len(actual), 1))

        previous = set(meta.temporal_ids)
        arrivals = actual - previous
        masked = row.copy()
        if previous:
            masked[list(previous)] = -np.inf
        for budget in (4, 8):
            pred = set(topk_indices(masked, budget))
            hit = len(pred & arrivals)
            useful[budget] += hit
            wasted[budget] += len(pred) - hit
            cold_precision[budget].append(hit / max(len(pred), 1))
            cold_recall[budget].append(hit / max(len(arrivals), 1) if arrivals else 1.0)

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {
        "route_recall": {f"R@{k}": mean(v) for k, v in route.items()},
        "cold_arrival": {
            f"budget_{b}": {
                "precision": mean(cold_precision[b]),
                "recall": mean(cold_recall[b]),
                "useful": useful[b],
                "wasted": wasted[b],
            }
            for b in (4, 8)
        },
    }


def evaluate(model: MLP, x, metas: list[SampleMeta]):
    return metrics_from_scores(model.forward(x)[-1], metas)


def evaluate_blends(
    model: MLP,
    x: np.ndarray,
    metas: list[SampleMeta],
    baseline_scores: np.ndarray | None,
):
    if baseline_scores is None or len(baseline_scores) != len(x):
        return None
    learned = model.forward(x)[-1]

    def row_zscore(values):
        mean = values.mean(axis=1, keepdims=True)
        std = values.std(axis=1, keepdims=True)
        return (values - mean) / np.maximum(std, 1e-6)

    base_z = row_zscore(baseline_scores)
    learned_z = row_zscore(learned)
    results = {}
    for alpha in (0.0, 0.05, 0.10, 0.25, 0.50, 1.0):
        scores = base_z + np.float32(alpha) * learned_z
        results[f"alpha_{alpha:.2f}"] = metrics_from_scores(scores, metas)
    return results


def export(
    model: MLP,
    out_prefix: Path,
    meta: dict,
    split: str,
    metrics: dict,
    feature_mode: str,
    expert_embedding: np.ndarray | None,
    target_mode: str,
):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    weights_path = out_prefix.with_suffix(".fp16.bin")
    manifest_path = out_prefix.with_suffix(".json")
    arrays = [model.w0.astype("<f2"), model.w1.astype("<f2"), model.w2.astype("<f2")]
    offsets = []
    cursor = 0
    with weights_path.open("wb") as f:
        for name, arr in zip(("w0", "w1", "w2"), arrays):
            blob = arr.tobytes(order="C")
            offsets.append({"name": name, "offset": cursor, "bytes": len(blob), "shape": list(arr.shape)})
            f.write(blob)
            cursor += len(blob)
    embedding_name = None
    if expert_embedding is not None:
        embedding_path = out_prefix.with_suffix(".embed.fp16.bin")
        embedding_path.write_bytes(expert_embedding.astype("<f2").tobytes(order="C"))
        embedding_name = embedding_path.name
        print(f"exported_embedding={embedding_path}")

    manifest = {
        "format": "routescout-ane-v1",
        "feature_version": FEATURE_VERSIONS[feature_mode],
        "feature_mode": feature_mode,
        "target_mode": target_mode,
        "dimensions": {
            "input": IN_FEATURES,
            "hidden": HIDDEN,
            "latent": LATENT,
            "output": int(meta["experts"]),
            "spatial": 16,
            "live_lanes": 8,
        },
        "model_geometry": meta,
        "split": split,
        "weights": weights_path.name,
        "expert_embedding": embedding_name,
        "arrays": offsets,
        "metrics": metrics,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"exported_weights={weights_path}")
    print(f"exported_manifest={manifest_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path(".perf_runs/routescout/model"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pos-weight", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--features", choices=sorted(FEATURE_VERSIONS), default="countsketch")
    ap.add_argument("--target", choices=("route", "arrivals"), default="route")
    args = ap.parse_args()

    meta, split, train_set, test_set, expert_embedding = split_traces(
        args.trace, args.features, args.target
    )
    x_train, y_train, _ = train_set
    x_test, _, m_test = test_set
    baseline_info = heldout_transition_baseline(args.trace)
    baseline, baseline_scores = baseline_info if baseline_info is not None else (None, None)
    print(
        f"geometry=layers:{meta['layers']} experts:{meta['experts']} topk:{meta['topk']} "
        f"train_samples={len(x_train)} test_samples={len(x_test)} split={split} "
        f"features={args.features} target={args.target}"
    )
    if baseline is not None:
        print("heldout_baseline=" + json.dumps(baseline, sort_keys=True))
    if len(x_train) == 0 or len(x_test) == 0:
        raise SystemExit("empty train/test set")

    model = MLP(meta["experts"], args.seed)
    train(model, x_train, y_train, args.epochs, args.batch, args.lr, args.pos_weight, args.seed)
    metrics = evaluate(model, x_test, m_test)
    blends = evaluate_blends(model, x_test, m_test, baseline_scores)
    if baseline is not None:
        metrics["heldout_baseline"] = baseline
    if blends is not None:
        metrics["prior_blends"] = blends
    print("metrics=" + json.dumps(metrics, sort_keys=True))
    export(
        model,
        args.out,
        meta,
        split,
        metrics,
        args.features,
        expert_embedding,
        args.target,
    )


if __name__ == "__main__":
    main()
