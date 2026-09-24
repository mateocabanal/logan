#!/usr/bin/env python3
"""Offline gate for authoritative predictive routing.

An *authoritative* route is one decided before the native router runs, so it is
also the route the I/O scheduler stages. Its quality cost depends only on how
much of the native router's weighted mixture it fails to reproduce.

Two figures are reported per policy, and the difference matters:

  discarded_mass   fraction of the native top-K *normalized* weight mass that
                   falls outside the authoritative set. This is the mixture mass
                   the layer simply drops (no renormalization).
  renormalized_*   the authoritative set's weights rescaled to sum to 1. The
                   kept experts' outputs are then amplified by 1/(1-discarded),
                   which is a different and usually larger error than dropping
                   the tail. Qwen's own router renormalizes over the selected
                   top-k, so this is the behaviour-compatible choice.

Neither number is a quality measurement — expert output norms are unknown
offline — but they bound the mixture perturbation and rank the policies, which
is what decides whether runtime work is justified.
"""

import sys
from collections import defaultdict


def load(path):
    events = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            ev, layer = int(p[0]), int(p[1])
            route = []
            for tok in p[5:]:
                e, _, w = tok.partition(":")
                if e:
                    route.append((int(e), float(w)))
            events.append((ev, layer, route))
    return events


def tokens_of(events, layers):
    out, cur = [], {}
    for _ev, layer, route in events:
        cur[layer] = route
        if layer + 1 == layers:
            out.append(cur)
            cur = {}
    if cur:
        out.append(cur)
    return out


def score(cur, chosen):
    """(discarded_mass, renormalized_kept_mass) over the native normalized route."""
    total = sum(w for _, w in cur)
    if total <= 0:
        return 0.0, 0.0
    kept = sum(w for e, w in cur if e in chosen)
    disc = (total - kept) / total
    renorm = kept / total / (1.0 - disc) if disc < 1.0 else 0.0
    return disc, renorm


class Predictor:
    """In-tree RouteScout: per-layer P(next | prev) temporal + spatial tables."""

    def __init__(self, layers, experts, tw=0.25):
        self.layers, self.experts, self.tw = layers, experts, tw
        self.T = [defaultdict(lambda: defaultdict(int)) for _ in range(layers)]
        self.Tn = [defaultdict(int) for _ in range(layers)]
        self.S = [defaultdict(lambda: defaultdict(int)) for _ in range(layers)]
        self.Sn = [defaultdict(int) for _ in range(layers)]

    def _raw(self, tbl, cnts, srcs):
        out = [0.0] * self.experts
        for s in srcs:
            n = cnts[s]
            if n:
                for d, c in tbl[s].items():
                    out[d] += c / n
        return out

    def _fuse(self, a, wa, b, wb):
        pa, pb = (max(a) if a else 0.0), (max(b) if b else 0.0)
        return [
            (a[i] / pa * wa if pa > 0 else 0.0) + (b[i] / pb * wb if pb > 0 else 0.0)
            for i in range(len(a))
        ]

    def scores(self, layer, prev, spatial):
        prev = [e for e in prev if e < self.experts]
        spatial = [e for e in spatial if e < self.experts]
        return self._fuse(
            self._raw(self.T[layer], self.Tn[layer], prev), self.tw,
            self._raw(self.S[layer], self.Sn[layer], spatial), 1.0,
        )

    def route(self, layer, prev, spatial, k):
        """Authoritative route: top-k of the fused score over ALL experts.

        Previous-route experts are NOT excluded. They are exactly the experts the
        predictor should keep if they are still likely, and their score is a real
        fused score, so route stability is captured by the model rather than
        forced by a warm-set-first ordering. Keeping this a plain top-k is what
        holds bytes/token equal to native routing, which any decode speedup
        requires.
        """
        f = self.scores(layer, prev, spatial)
        ranked = sorted(
            (i for i in range(self.experts) if f[i] > 0), key=lambda i: (-f[i], i)
        )
        return ranked[:k]

    def observe(self, layer, prev, spatial, current):
        for p in prev:
            if p < self.experts:
                self.Tn[layer][p] += 1
                for c in current:
                    if c < self.experts:
                        self.T[layer][p][c] += 1
        for p in spatial:
            if p < self.experts:
                self.Sn[layer][p] += 1
                for c in current:
                    if c < self.experts:
                        self.S[layer][p][c] += 1


def evaluate(toks, layers, topk, horizons, warm_frac=0.25):
    warm = max(1, int(len(toks) * warm_frac))
    idx = list(range(warm, len(toks)))

    def agg(fn):
        s = 0.0
        for t in idx:
            for l in range(layers):
                cur = toks[t].get(l, [])
                if cur:
                    s += fn(t, l, cur)
        return s / max(1, len(idx) * layers)

    print("  policy                                    discarded_mass  "
          "recall@K  retained_weight_share")

    # --- Reference points that are NOT early-decidable, as ceilings ---------
    for k in (2, 4, 6, topk):
        disc = agg(lambda t, l, cur, k=k: score(cur, {e for e, _ in
                  sorted(cur, key=lambda x: -x[1])[:k]})[0])
        print(f"  native top-{k} truncation (ceiling)             "
              f"{disc:>14.4f}  {k/topk:>8.4f}  {1-disc:>14.4f}")

    # --- Early-decidable policies -------------------------------------------
    # P1: this token's route == previous token's same-layer route.
    disc = agg(lambda t, l, cur: score(cur, {e for e, _ in toks[t - 1].get(l, [])})[0])
    print(f"  P1 prev-token same-layer route              {disc:>14.4f}")

    # P2: previous token's route UNION this token's previous layer (spatial),
    #     i.e. the union of what is already warm in either direction.
    def p2(t, l, cur):
        s = {e for e, _ in toks[t - 1].get(l, [])}
        s |= {e for e, _ in toks[t].get(l - 1, [])} if l > 0 else set()
        return score(cur, s)[0]
    disc = agg(p2)
    print(f"  P2 prev-token U same-token prev-layer       {disc:>14.4f}")

    # --- Online predictor at several K, authoritative top-K semantics -------
    for k in (topk, 6, 4, 2):
        pred = Predictor(layers, 256)
        experts = 256
        disc_acc, rec = 0.0, 0
        for t in range(1, len(toks)):
            for l in range(layers):
                prev = [e for e, _ in toks[t - 1].get(l, [])]
                spatial = [e for e, _ in toks[t].get(l - 1, [])] if l > 0 else []
                cur = toks[t].get(l, [])
                if not cur:
                    continue
                route = pred.route(l, prev, spatial, k)
                if t >= warm:
                    d, _ = score(cur, set(route))
                    disc_acc += d
                pred.observe(l, prev, spatial, [e for e, _ in cur])
        n = len(idx) * layers
        print(f"  P3 authoritative top-{k} (fused score)        "
              f"{disc_acc/n:>14.4f}")

    # --- Cross-layer horizons, authoritative top-k --------------------------
    for H in horizons:
        pred = Predictor(layers, 256)
        disc_acc = 0.0
        for t in range(1, len(toks)):
            for l in range(layers):
                prev = [e for e, _ in toks[t - 1].get(l, [])]
                spatial = [e for e, _ in toks[t].get(l - 1, [])] if l > 0 else []
                cur = toks[t - 1].get(l, [])
                pred.observe(l, prev, spatial, [e for e, _ in cur])
            for l in range(layers):
                tgt = l + H
                if tgt >= layers:
                    continue
                src_prev = [e for e, _ in toks[t - 1].get(tgt, [])]
                spatial = [e for e, _ in toks[t].get(tgt - 1, [])] if tgt > 0 else []
                cur = toks[t].get(tgt, [])
                if not cur:
                    continue
                route = pred.route(tgt, src_prev, spatial, topk)
                if t >= warm:
                    d, _ = score(cur, set(route))
                    disc_acc += d
        n = len(idx) * layers
        print(f"  P4 authoritative H={H} (cross-layer)         "
              f"{disc_acc/n:>14.4f}")

    # --- P6: route self-stability (the locality mechanism) ------------------
    # Byte counters cannot distinguish "8 scattered reads" from "the same 8
    # regions as last token". If an authoritative route repeats itself far more
    # than the native one, the OS page cache serves the repeats and MetalIO wait
    # falls while the *logical* bytes/token stay identical. That is a candidate
    # explanation for a wait reduction at constant byte count, so it is measured
    # rather than assumed.
    def self_overlap(routes):
        """Mean fraction of token t's route that was also token t-1's route."""
        acc, n = 0.0, 0
        for t in range(1, len(routes)):
            a, b = routes[t], routes[t - 1]
            if not a or not b:
                continue
            acc += len(set(a) & set(b)) / len(a)
            n += 1
        return acc / max(1, n)

    native_routes, auth_routes = [], []
    for t in range(len(toks)):
        native_routes.append(
            {l: [e for e, _ in toks[t].get(l, [])] for l in range(layers)}
        )
    pred = Predictor(layers, 256)
    for t in range(1, len(toks)):
        for l in range(layers):
            prev = [e for e, _ in toks[t - 1].get(l, [])]
            spatial = [e for e, _ in toks[t].get(l - 1, [])] if l > 0 else []
            cur = toks[t].get(l, [])
            pred.observe(l, prev, spatial, [e for e, _ in cur])
        auth_routes.append(
            {
                l: pred.route(
                    l,
                    [e for e, _ in toks[t - 1].get(l, [])],
                    [e for e, _ in toks[t].get(l - 1, [])] if l > 0 else [],
                    topk,
                )
                for l in range(layers)
            }
        )
    for label, routes in (("native", native_routes[1:]), ("authoritative", auth_routes)):
        per_layer = []
        for l in (0, layers // 2, layers - 1):
            per_layer.append(
                f"L{l}={self_overlap([r.get(l, []) for r in routes]):.3f}"
            )
        allr = []
        for t in range(1, len(routes)):
            a = [e for l in range(layers) for e in routes[t].get(l, [])]
            b = [e for l in range(layers) for e in routes[t - 1].get(l, [])]
            if a and b:
                allr.append(len(set(a) & set(b)) / len(a))
        print(f"  P6 self-overlap {label:<16} overall={sum(allr)/max(1,len(allr)):.4f} "
              f"{' '.join(per_layer)}")
    # Ranking the fused score over ALL experts throws away the strongest single
    # piece of evidence available: the expert is already the same one the
    # previous token routed to. A bias term recovers it without forcing it, so
    # the predictor can still displace an unstable resident. lambda is in units
    # of the fused score's own peak, so it is scale-free across layers.
    for lam in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        for H in (0, 4):
            pred = Predictor(layers, 256)
            disc_acc = 0.0
            for t in range(1, len(toks)):
                if H:
                    for l in range(layers):
                        prev = [e for e, _ in toks[t - 1].get(l, [])]
                        spatial = [e for e, _ in toks[t].get(l - 1, [])] if l > 0 else []
                        cur = toks[t - 1].get(l, [])
                        pred.observe(l, prev, spatial, [e for e, _ in cur])
                for l in range(layers):
                    tgt = l + H
                    if tgt >= layers:
                        continue
                    prev = [e for e, _ in toks[t - 1].get(tgt, [])]
                    spatial = [e for e, _ in toks[t].get(tgt - 1, [])] if tgt > 0 else []
                    cur = toks[t].get(tgt, [])
                    if not cur:
                        continue
                    f = pred.scores(tgt, prev, spatial)
                    peak = max(f) if f else 0.0
                    if peak > 0:
                        warmset = set(prev)
                        f = [v + (lam * peak if i in warmset else 0.0) for i, v in enumerate(f)]
                    route = sorted(
                        (i for i in range(256) if f[i] > 0), key=lambda i: (-f[i], i)
                    )[:topk]
                    if t >= warm:
                        d, _ = score(cur, set(route))
                        disc_acc += d
                    if not H:
                        pred.observe(tgt, prev, spatial, [e for e, _ in cur])
            n = len(idx) * layers
            print(f"  P5 stability bias lam={lam:<4} H={H}              "
                  f"{disc_acc/n:>14.4f}")


def main():
    for path in sys.argv[1:]:
        with open(path) as f:
            head = f.readline().strip()
        meta = dict(kv.split("=") for kv in head.lstrip("# ").split("\t")[1:])
        layers, experts, topk = int(meta["layers"]), int(meta["experts"]), int(meta["topk"])
        toks = tokens_of(load(path), layers)
        print(f"\n######## {path}  layers={layers} experts={experts} topk={topk} "
              f"tokens={len(toks)}")
        evaluate(toks, layers, topk, (1, 2, 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
