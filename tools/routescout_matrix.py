#!/usr/bin/env python3
"""RouteScout Phase-A/B harness: cross-prompt holdout matrix, layer breakdown,
temporal history depth, online-vs-global priors, and a candidate-centric learned
scorer evaluated on true held-out prompts.

NumPy only, no external deps. Every predictor consumes only information that is
available before the authoritative router executes for the target layer:

  * previous token(s), same layer route (temporal history)
  * current token, previous layer route (spatial)
  * training-only per-layer frequency / transition priors

Nothing here touches the authoritative router; the harness only scores
predictions against the routes the model actually chose.

Metrics
-------
route R@K     recall of the full top-k route (diagnostic only).
cold-arrival  arrivals = actual - previous-route. Candidates are ranked after
              masking previous-route experts, so precision is what a prefetcher
              would observe (useful loads / issued loads).

Subcommands
-----------
matrix   leave-one-prompt-out holdout matrix over every prompt family.
layers   per-layer breakdown for the holdout matrix.
history  temporal history depth 1/2/4/8 with the same holdout protocol.
local    prompt-local online adaptation curve versus frozen global priors.
scorer   candidate-centric shared per-expert scorer on held-out prompts.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from routescout_trace import Event, parse, transition_scores, update_transitions

BUDGETS = (4, 8, 12, 16, 24)
ROUTE_KS = (8, 16, 24)
BASELINES = ("frequency", "temporal_transition", "spatial_transition", "hybrid_transition")


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


@dataclass
class Corpus:
    name: str
    meta: dict
    cycles: list[list[Event]]

    @property
    def layers(self) -> int:
        return self.meta["layers"]

    @property
    def experts(self) -> int:
        return self.meta["experts"]


def load_corpus(paths: list[Path]) -> list[Corpus]:
    out = []
    for path in paths:
        meta, _events, cycles = parse(path)
        if cycles:
            out.append(Corpus(path.stem, meta, cycles))
    if not out:
        raise SystemExit("no usable traces")
    geometry = {tuple(c.meta[k] for k in ("layers", "experts", "topk")) for c in out}
    if len(geometry) != 1:
        raise SystemExit(f"trace geometry mismatch: {sorted(geometry)}")
    return out


def norm_vec(values: list[float], experts: int) -> np.ndarray:
    x = np.zeros(experts, dtype=np.float32)
    for expert, weight in values:
        x[expert] = weight
    total = float(x.sum())
    return x / total if total > 0 else x


def event_vec(ev: Event, experts: int) -> np.ndarray:
    return norm_vec(list(zip(ev.ids, ev.weights)), experts)


# ---------------------------------------------------------------------------
# priors: frequency + transition tables trained per corpus subset
# ---------------------------------------------------------------------------


@dataclass
class Priors:
    layers: int
    experts: int
    freq: list[np.ndarray] = field(default_factory=list)
    temporal: list = field(default_factory=list)
    spatial: list = field(default_factory=list)
    temporal_history: list = field(default_factory=list)
    spatial_history: list = field(default_factory=list)

    @classmethod
    def fit(cls, corpora: list[Corpus], history: int = 1) -> "Priors":
        layers = corpora[0].layers
        experts = corpora[0].experts
        freq = [np.zeros(experts, dtype=np.float32) for _ in range(layers)]
        temporal = [defaultdict(lambda: defaultdict(int)) for _ in range(layers)]
        spatial = [defaultdict(lambda: defaultdict(int)) for _ in range(layers - 1)]
        history_temporal = [
            [defaultdict(lambda: defaultdict(int)) for _ in range(layers)] for _ in range(history)
        ]
        history_spatial = [
            [defaultdict(lambda: defaultdict(int)) for _ in range(layers)] for _ in range(history)
        ]
        for corpus in corpora:
            cycles = corpus.cycles
            for t, cycle in enumerate(cycles):
                for layer, ev in enumerate(cycle):
                    for expert in ev.ids:
                        freq[layer][expert] += 1.0
                    if t:
                        update_transitions(temporal[layer], cycles[t - 1][layer].ids, ev.ids)
                    if layer + 1 < layers:
                        update_transitions(spatial[layer], ev.ids, cycle[layer + 1].ids)
                    for depth in range(history):
                        if not t:
                            break
                        back = t - 1 - depth
                        if back < 0:
                            break
                        update_transitions(
                            history_temporal[depth][layer],
                            cycles[back][layer].ids,
                            ev.ids,
                        )
                        if layer + 1 < layers:
                            update_transitions(
                                history_spatial[depth][layer],
                                cycles[back][layer].ids,
                                cycle[layer + 1].ids,
                            )
        for row in freq:
            total = float(row.sum())
            if total > 0:
                row /= total
        return cls(layers, experts, freq, temporal, spatial, history_temporal, history_spatial)

    def temporal_scores(self, layer: int, ids: list[int], depth: int = 0) -> dict[int, float]:
        table = self.temporal[layer] if depth == 0 else self.temporal_history[depth][layer]
        return transition_scores(table, ids, self.experts)

    def spatial_scores(self, layer: int, ids: list[int], depth: int = 0) -> dict[int, float]:
        if layer == 0:
            return {}
        table = self.spatial[layer - 1] if depth == 0 else self.spatial_history[depth][layer - 1]
        return transition_scores(table, ids, self.experts)

    def vector(self, scores: dict[int, float], normalize: bool = True) -> np.ndarray:
        out = np.zeros(self.experts, dtype=np.float32)
        for expert, value in scores.items():
            out[expert] = value
        if normalize:
            peak = float(out.max()) if out.size else 0.0
            if peak > 0:
                out = out / peak
        return out

    def predict(self, layer: int, temporal_ids: list[int], spatial_ids: list[int]) -> dict[str, np.ndarray]:
        temporal = self.vector(self.temporal_scores(layer, temporal_ids))
        spatial = self.vector(self.spatial_scores(layer, spatial_ids))
        return {
            "frequency": self.freq[layer].copy(),
            "temporal_transition": temporal,
            "spatial_transition": spatial,
            "hybrid_transition": temporal + spatial + 0.25 * self.freq[layer],
        }

    def predict_history(self, layer: int, histories: list[list[int]], spatial_ids: list[int]) -> dict[str, np.ndarray]:
        """Depth-weighted temporal evidence plus depth-0 spatial."""
        temporal = np.zeros(self.experts, dtype=np.float32)
        weight = 1.0
        for depth, ids in enumerate(histories[: len(self.temporal_history)]):
            if not ids:
                break
            temporal += weight * self.vector(self.temporal_scores(layer, ids, depth))
            weight *= 0.5
        spatial = self.vector(self.spatial_scores(layer, spatial_ids))
        return {
            "temporal_history": temporal,
            "hybrid_history": temporal + spatial + 0.25 * self.freq[layer],
        }


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def top_k(scores: np.ndarray, k: int, forbid: set[int] | None = None) -> list[int]:
    if forbid:
        scores = scores.copy()
        scores[list(forbid)] = -np.inf
    k = min(k, scores.size)
    if k <= 0:
        return []
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")].tolist()


@dataclass
class Accum:
    hits: int = 0
    total: int = 0
    issued: int = 0

    def snapshot(self) -> dict:
        return {
            "recall": self.hits / self.total if self.total else 0.0,
            "precision": self.hits / self.issued if self.issued else 0.0,
            "useful": self.hits,
            "wasted": self.issued - self.hits,
            "arrivals": self.total,
        }


class Metrics:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def ensure(self, predictor: str, layer: int | None = None, per_layer: bool = False) -> dict:
        row = self.rows.setdefault(
            predictor,
            {"route": {k: Accum() for k in ROUTE_KS}, "cold": {b: Accum() for b in BUDGETS},
             "layers": {}},
        )
        if per_layer and layer is not None:
            return row["layers"].setdefault(
                layer,
                {"route": {k: Accum() for k in ROUTE_KS}, "cold": {b: Accum() for b in BUDGETS}},
            )
        return row

    def observe(
        self,
        predictor: str,
        layer: int,
        scores: np.ndarray,
        actual: list[int],
        previous: list[int],
        per_layer: bool,
    ) -> None:
        actual_set = set(actual)
        previous_set = set(previous)
        arrivals = actual_set - previous_set
        # With per_layer=False, ensure(predictor) and ensure(predictor, layer, True)
        # are the SAME bucket; iterating both would double every count (rates
        # cancel, but useful/wasted/arrival totals would be 2x wrong).
        buckets = [self.ensure(predictor)]
        if per_layer:
            buckets.append(self.ensure(predictor, layer, per_layer))
        for bucket in buckets:
            for k in ROUTE_KS:
                bucket["route"][k].hits += len(set(top_k(scores, k)) & actual_set)
                bucket["route"][k].total += len(actual_set)
            for budget in BUDGETS:
                pred = top_k(scores, budget, forbid=previous_set)
                hit = len(set(pred) & arrivals)
                acc = bucket["cold"][budget]
                acc.hits += hit
                acc.total += len(arrivals)
                acc.issued += len(pred)

    def summary(self, per_layer: bool) -> dict:
        out = {}
        for name, row in self.rows.items():
            entry = {
                "route": {f"R@{k}": row["route"][k].snapshot()["recall"] for k in ROUTE_KS},
                "cold": {f"budget_{b}": row["cold"][b].snapshot() for b in BUDGETS},
            }
            if per_layer:
                entry["per_layer"] = {
                    str(layer): {
                        "route": {f"R@{k}": bucket["route"][k].snapshot()["recall"] for k in ROUTE_KS},
                        "cold": {f"budget_{b}": bucket["cold"][b].snapshot() for b in BUDGETS},
                    }
                    for layer, bucket in sorted(row["layers"].items())
                }
            out[name] = entry
        return out


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def run_baselines(
    train: list[Corpus],
    holdout: Corpus,
    per_layer: bool,
    history: int = 1,
) -> tuple[Metrics, Priors]:
    priors = Priors.fit(train, history=history)
    metrics = Metrics()
    cycles = holdout.cycles
    for t in range(1, len(cycles)):
        cur = cycles[t]
        for layer in range(holdout.layers):
            previous = cycles[t - 1][layer].ids
            spatial_ids = cur[layer - 1].ids if layer else []
            predictions = priors.predict(layer, list(previous), list(spatial_ids))
            histories = []
            for depth in range(max(history, 2)):
                back = t - 1 - depth
                histories.append(list(cycles[back][layer].ids) if back >= 0 else [])
            predictions.update(priors.predict_history(layer, histories, list(spatial_ids)))
            for name, scores in predictions.items():
                metrics.observe(name, layer, scores, list(cur[layer].ids), list(previous), per_layer)
    return metrics, priors


# ---------------------------------------------------------------------------
# candidate-centric scorer
# ---------------------------------------------------------------------------

SCORER_FEATURES = 16


def _score_stats(scores: dict[int, float]) -> tuple[float, float, float]:
    if not scores:
        return (0.0, 0.0, 0.0)
    values = np.fromiter(scores.values(), dtype=np.float32)
    total = float(values.sum())
    probs = values / total if total > 0 else values
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    ordered = np.sort(probs)[::-1]
    margin = float(ordered[0] - ordered[1]) if ordered.size > 1 else float(ordered[0])
    return (entropy, margin, float(ordered[0]))


def scorer_matrix(
    priors: Priors,
    layer: int,
    histories: list[list[int]],
    history_weights: list[list[float]],
    spatial_ids: list[int],
    spatial_weights: list[float],
) -> np.ndarray:
    """[experts, SCORER_FEATURES] candidate feature matrix."""
    experts = priors.experts
    layers = priors.layers
    temporal_vec = norm_vec(list(zip(histories[0], history_weights[0])), experts) if histories and histories[0] else np.zeros(experts, np.float32)
    spatial_vec = norm_vec(list(zip(spatial_ids, spatial_weights)), experts) if layer else np.zeros(experts, np.float32)
    spatial_raw = np.zeros(experts, np.float32)
    for expert, value in priors.spatial_scores(layer, spatial_ids).items():
        spatial_raw[expert] = value
    spatial_peak = float(spatial_raw.max()) if spatial_raw.size else 0.0
    if spatial_peak > 0:
        spatial_raw /= spatial_peak

    prev_flag = (temporal_vec > 0).astype(np.float32)
    tent = _score_stats(priors.temporal_scores(layer, histories[0]))
    sent = _score_stats(priors.spatial_scores(layer, spatial_ids))

    x = np.empty((experts, SCORER_FEATURES), dtype=np.float32)
    x[:, 0] = temporal_vec
    x[:, 1] = prev_flag
    x[:, 2] = spatial_raw
    x[:, 3] = np.sqrt(spatial_raw)
    x[:, 4] = priors.freq[layer]
    x[:, 5] = sent[0]
    x[:, 6] = sent[1]
    x[:, 7] = sent[2]
    x[:, 8] = tent[0]
    x[:, 9] = tent[1]
    x[:, 10] = tent[2]
    x[:, 11] = layer / max(1, layers - 1)
    x[:, 12] = (layer % 4) / 3.0
    x[:, 13] = np.arange(experts, dtype=np.float32) / max(1, experts - 1)
    x[:, 14] = spatial_raw * (1.0 - prev_flag)
    x[:, 15] = 1.0
    return x


class Scorer:
    def __init__(self, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.w0 = (rng.standard_normal((16, SCORER_FEATURES)) * math.sqrt(2 / SCORER_FEATURES)).astype(np.float32)
        self.w1 = (rng.standard_normal((8, 16)) * math.sqrt(2 / 16)).astype(np.float32)
        self.w2 = (rng.standard_normal((1, 8)) * math.sqrt(1 / 8)).astype(np.float32)
        self.params = [self.w0, self.w1, self.w2]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.step = 0

    def forward(self, x: np.ndarray):
        z0 = x @ self.w0.T
        h0 = np.maximum(z0, 0.0)
        z1 = h0 @ self.w1.T
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.w2.T
        return z2[:, 0], (x, z0, h0, z1, h1)

    def scores(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    def train_batch(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, lr: float) -> float:
        logits, cache = self.forward(x)
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        denom = max(1e-6, float(weights.sum()))
        dz = ((p - y) * weights / denom)[:, None]
        xin, z0, h0, z1, h1 = cache
        g2 = dz.T @ h1
        dh1 = dz @ self.w2
        dz1 = dh1 * (z1 > 0)
        g1 = dz1.T @ h0
        dh0 = dz1 @ self.w1
        dz0 = dh0 * (z0 > 0)
        g0 = dz0.T @ xin
        self.step += 1
        beta1, beta2 = 0.9, 0.999
        for i, (param, grad) in enumerate(zip(self.params, [g0, g1, g2])):
            np.clip(grad, -2, 2, out=grad)
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad * grad)
            mhat = self.m[i] / (1 - beta1**self.step)
            vhat = self.v[i] / (1 - beta2**self.step)
            param -= lr * mhat / (np.sqrt(vhat) + 1e-8)
        eps = 1e-7
        return float(-(weights * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))).sum() / denom)


def iter_layers_with_history(corpus: Corpus, history: int):
    """Yield (t, layer, histories_ids, histories_weights, spatial_ids, spatial_weights)."""
    cycles = corpus.cycles
    for t in range(history, len(cycles)):
        cur = cycles[t]
        for layer in range(corpus.layers):
            ids = []
            weights = []
            for depth in range(history):
                ev = cycles[t - 1 - depth][layer]
                ids.append(list(ev.ids))
                weights.append(list(ev.weights))
            spatial_ev = cur[layer - 1] if layer else None
            yield (
                t,
                layer,
                ids,
                weights,
                list(spatial_ev.ids) if spatial_ev else [],
                list(spatial_ev.weights) if spatial_ev else [],
            )


def build_dataset(
    corpora: list[Corpus],
    priors: Priors,
    history: int,
    target_mode: str,
    neg_per: int,
    seed: int,
    include_prior_blend: bool,
):
    rng = np.random.default_rng(seed)
    xs, ys, ws, blends = [], [], [], []
    for corpus in corpora:
        cycles = corpus.cycles
        for t, layer, ids, weights, sids, sweights in iter_layers_with_history(corpus, history):
            x = scorer_matrix(priors, layer, ids, weights, sids, sweights)
            actual = set(cycles[t][layer].ids)
            previous = set(ids[0]) if ids else set()
            y = np.zeros(priors.experts, np.float32)
            pos = []
            for expert in actual:
                if target_mode == "route" or expert not in previous:
                    y[expert] = 1.0
                    pos.append(expert)
            neg = np.flatnonzero(y == 0)
            choose = rng.choice(neg, size=min(neg_per, neg.size), replace=False)
            idx = np.concatenate([np.asarray(pos, dtype=np.int64), choose])
            xs.append(x[idx])
            ys.append(y[idx])
            weight = np.ones(len(idx), np.float32)
            cold_mask = np.array([e in actual and e not in previous for e in idx])
            weight[: len(pos)] = np.where(cold_mask[: len(pos)], 6.0, 2.0)
            ws.append(weight)
            if include_prior_blend:
                base = priors.predict(layer, ids[0], sids)["hybrid_transition"]
                blends.append(base[idx])
    x = np.concatenate(xs) if xs else np.empty((0, SCORER_FEATURES), np.float32)
    y = np.concatenate(ys) if ys else np.empty((0,), np.float32)
    w = np.concatenate(ws) if ws else np.empty((0,), np.float32)
    b = np.concatenate(blends) if blends else None
    return x, y, w, b


def train_scorer(x, y, w, epochs: int, batch: int, lr: float, seed: int) -> Scorer:
    model = Scorer(seed)
    rng = np.random.default_rng(seed + 1)
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        losses = []
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            losses.append(model.train_batch(x[idx], y[idx], w[idx], lr))
        if epoch == 0 or (epoch + 1) % max(epochs // 4, 1) == 0:
            print(f"  epoch {epoch + 1:>3} weighted_bce={np.mean(losses):.5f}")
    return model


def zscore_rows(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)


def evaluate_scorer(
    model: Scorer,
    priors: Priors,
    holdout: Corpus,
    history: int,
    alpha: float,
    per_layer: bool,
) -> Metrics:
    metrics = Metrics()
    cycles = holdout.cycles
    for t, layer, ids, weights, sids, sweights in iter_layers_with_history(holdout, history):
        x = scorer_matrix(priors, layer, ids, weights, sids, sweights)
        learned = model.scores(x)
        base = priors.predict(layer, ids[0], sids)["hybrid_transition"]
        scores = zscore_rows(base[None, :])[0]
        if alpha > 0:
            scores = scores + alpha * zscore_rows(learned[None, :])[0]
        previous = list(ids[0]) if ids else []
        metrics.observe("learned_scorer", layer, learned, list(cycles[t][layer].ids), previous, per_layer)
        metrics.observe("hybrid_transition", layer, base, list(cycles[t][layer].ids), previous, per_layer)
        metrics.observe(
            f"blend_alpha_{alpha:.2f}", layer, scores, list(cycles[t][layer].ids), previous, per_layer
        )
    return metrics


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def cmd_matrix(args) -> dict:
    corpus = load_corpus(args.traces)
    print(f"corpus={len(corpus)} layers={corpus[0].layers} experts={corpus[0].experts} topk={corpus[0].meta['topk']}")
    per_prompt = {}
    for holdout in corpus:
        train = [c for c in corpus if c.name != holdout.name]
        metrics, _priors = run_baselines(train, holdout, args.per_layer, history=1)
        summary = metrics.summary(args.per_layer)
        per_prompt[holdout.name] = {
            "train": [c.name for c in train],
            "holdout_cycles": len(holdout.cycles),
            "summary": summary,
        }
        hy = summary["hybrid_transition"]
        sp = summary["spatial_transition"]
        tp = summary["temporal_transition"]
        print(
            f"holdout={holdout.name:<30} cycles={len(holdout.cycles):>3} "
            f"hyb R@8={hy['route']['R@8']:.4f} cold8={hy['cold']['budget_8']['recall']:.4f}"
            f"/{hy['cold']['budget_8']['precision']:.4f} | "
            f"spat R@8={sp['route']['R@8']:.4f} cold8={sp['cold']['budget_8']['recall']:.4f} | "
            f"temp R@8={tp['route']['R@8']:.4f} cold8={tp['cold']['budget_8']['recall']:.4f}"
        )
    aggregate = {}
    for predictor in BASELINES:
        aggregate[predictor] = {
            "route": {
                f"R@{k}": float(np.mean([m["summary"][predictor]["route"][f"R@{k}"] for m in per_prompt.values()]))
                for k in ROUTE_KS
            },
            "cold": {
                f"budget_{b}": {
                    "recall": float(np.mean([m["summary"][predictor]["cold"][f"budget_{b}"]["recall"] for m in per_prompt.values()])),
                    "precision": float(np.mean([m["summary"][predictor]["cold"][f"budget_{b}"]["precision"] for m in per_prompt.values()])),
                }
                for b in BUDGETS
            },
        }
    print("\naggregate (unweighted mean over prompts)")
    for predictor in BASELINES:
        row = aggregate[predictor]
        print(
            f"{predictor:<22} R@8={row['route']['R@8']:.4f} R@16={row['route']['R@16']:.4f} "
            f"R@24={row['route']['R@24']:.4f} cold8={row['cold']['budget_8']['recall']:.4f}"
            f"/{row['cold']['budget_8']['precision']:.4f}"
        )
    return {"geometry": corpus[0].meta, "per_prompt": per_prompt, "aggregate": aggregate}


def cmd_history(args) -> dict:
    corpus = load_corpus(args.traces)
    out = {}
    for holdout in corpus:
        train = [c for c in corpus if c.name != holdout.name]
        depth_results = {}
        for history in (1, 2, 4, 8):
            if len(holdout.cycles) <= history:
                continue
            metrics, _ = run_baselines(train, holdout, False, history=history)
            summary = metrics.summary(False)
            depth_results[history] = {
                "temporal_history": summary["temporal_history"]["route"],
                "hybrid_history": summary["hybrid_history"]["route"],
                "temporal_history_cold8": summary["temporal_history"]["cold"]["budget_8"],
                "hybrid_history_cold8": summary["hybrid_history"]["cold"]["budget_8"],
            }
            print(
                f"holdout={holdout.name:<30} depth={history} "
                f"temporal_hist R@8={summary['temporal_history']['route']['R@8']:.4f} "
                f"hybrid_hist R@8={summary['hybrid_history']['route']['R@8']:.4f} "
                f"cold8={summary['hybrid_history']['cold']['budget_8']['recall']:.4f}"
                f"/{summary['hybrid_history']['cold']['budget_8']['precision']:.4f}"
            )
        out[holdout.name] = depth_results
    return out


def cmd_local(args) -> dict:
    """Prompt-local online adaptation: measure how fast local-only priors approach
    the frozen cross-prompt global priors. Both are evaluated on the same prompt."""
    corpus = load_corpus(args.traces)
    out = {}
    for holdout in corpus:
        train = [c for c in corpus if c.name != holdout.name]
        global_priors = Priors.fit(train)
        metrics = Metrics()
        seen: list[Corpus] = []
        cycles = holdout.cycles
        warmup = args.warmup
        global_row = Metrics()
        local_row = Metrics()
        for t in range(1, len(cycles)):
            # Refit once per token, not once per layer: the local priors depend
            # only on the cycles observed so far, so refitting inside the layer
            # loop repeats identical work 40x.
            local_priors = Priors.fit(seen) if len(seen) >= warmup else global_priors
            for layer in range(holdout.layers):
                previous = list(cycles[t - 1][layer].ids)
                spatial = list(cycles[t][layer - 1].ids) if layer else []
                gp = global_priors.predict(layer, previous, spatial)["hybrid_transition"]
                lp = local_priors.predict(layer, previous, spatial)["hybrid_transition"]
                actual = list(cycles[t][layer].ids)
                global_row.observe("global", layer, gp, actual, previous, False)
                local_row.observe("local", layer, lp, actual, previous, False)
            seen.append(Corpus("seen", holdout.meta, cycles[:t]))
        g = global_row.summary(False)["global"]
        l = local_row.summary(False)["local"]
        out[holdout.name] = {
            "global": g,
            "local_after_warmup": l,
            "warmup_cycles": warmup,
        }
        print(
            f"holdout={holdout.name:<30} cycles={len(cycles):>3} warmup={warmup} "
            f"global R@8={g['route']['R@8']:.4f} cold8={g['cold']['budget_8']['recall']:.4f} | "
            f"local R@8={l['route']['R@8']:.4f} cold8={l['cold']['budget_8']['recall']:.4f}"
        )
    return out


def cmd_scorer(args) -> dict:
    corpus = load_corpus(args.traces)
    out = {}
    for holdout in corpus:
        train = [c for c in corpus if c.name != holdout.name]
        priors = Priors.fit(train)
        x, y, w, _ = build_dataset(train, priors, args.history, args.target, args.neg_per, 3, False)
        xh, yh, wh, _ = build_dataset([holdout], priors, args.history, args.target, args.neg_per, 3, False)
        print(
            f"holdout={holdout.name} train_rows={len(y)} pos={int(y.sum())} "
            f"holdout_rows={len(yh)} pos={int(yh.sum())}"
        )
        model = train_scorer(x, y, w, args.epochs, args.batch, args.lr, args.seed)
        metrics = evaluate_scorer(model, priors, holdout, args.history, args.alpha, False)
        summary = metrics.summary(False)
        out[holdout.name] = {"summary": summary, "train_rows": int(len(y)), "holdout_rows": int(len(yh))}
        ls = summary["learned_scorer"]
        bs = summary["hybrid_transition"]
        bl = summary[f"blend_alpha_{args.alpha:.2f}"]
        print(
            f"  learned  R@8={ls['route']['R@8']:.4f} cold8={ls['cold']['budget_8']['recall']:.4f}"
            f"/{ls['cold']['budget_8']['precision']:.4f}"
        )
        print(
            f"  baseline R@8={bs['route']['R@8']:.4f} cold8={bs['cold']['budget_8']['recall']:.4f}"
            f"/{bs['cold']['budget_8']['precision']:.4f}"
        )
        print(
            f"  blend    R@8={bl['route']['R@8']:.4f} cold8={bl['cold']['budget_8']['recall']:.4f}"
            f"/{bl['cold']['budget_8']['precision']:.4f}"
        )
    if args.export:
        # Export the scorer trained on the full corpus (every prompt), which is
        # the artifact an ANE gate and a runtime would actually consume.
        priors = Priors.fit(corpus)
        x, y, w, _ = build_dataset(corpus, priors, args.history, args.target, args.neg_per, 3, False)
        model = train_scorer(x, y, w, args.epochs, args.batch, args.lr, args.seed)
        args.export.mkdir(parents=True, exist_ok=True)
        blob = bytearray()
        arrays = []
        cursor = 0
        for name, param in (("w0", model.w0), ("w1", model.w1), ("w2", model.w2)):
            payload = param.astype("<f2").tobytes()
            arrays.append(
                {"name": name, "offset": cursor, "bytes": len(payload), "shape": list(param.shape)}
            )
            blob.extend(payload)
            cursor += len(payload)
        (args.export / "scorer.fp16.bin").write_bytes(bytes(blob))
        (args.export / "scorer.json").write_text(
            json.dumps(
                {
                    "format": "routescout-scorer-ane-v1",
                    "features": SCORER_FEATURES,
                    "hidden": 16,
                    "latent": 8,
                    "output": 1,
                    "train_rows": int(len(y)),
                    "train_prompts": [c.name for c in corpus],
                    "arrays": arrays,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"exported {args.export}/scorer.fp16.bin and scorer.json (train_rows={len(y)})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    def common(parser):
        parser.add_argument("traces", nargs="+", type=Path)
        parser.add_argument("--out", type=Path, default=None)

    p = sub.add_parser("matrix")
    common(p)
    p.add_argument("--per-layer", action="store_true")

    p = sub.add_parser("history")
    common(p)

    p = sub.add_parser("local")
    common(p)
    p.add_argument("--warmup", type=int, default=8)

    p = sub.add_parser("scorer")
    common(p)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--neg-per", type=int, default=48)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--history", type=int, default=1)
    p.add_argument("--target", choices=("route", "arrivals"), default="route")
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--export", type=Path, default=None)

    args = ap.parse_args()
    handlers = {"matrix": cmd_matrix, "history": cmd_history, "local": cmd_local, "scorer": cmd_scorer}
    payload = handlers[args.command](args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
