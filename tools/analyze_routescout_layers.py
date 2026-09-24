#!/usr/bin/env python3
"""Per-layer analysis of adapter arms from `eval_routescout.py` output.

Answers the layer questions the handoff asks for: which layers are strongest,
which are weakest, which fail to improve across a corpus scaling step, and
whether the spread justifies per-layer capacity/data decisions.

Two modes:

* `--eval a.json b.json` — compare adapters on one held-out set per head.
* `--curve runs/routescout_*.metrics.json` — find heads whose held-out quality
  *did not* improve between two consecutive corpus scales, which is direct
  evidence for or against a uniform data-limited story.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_eval(path: Path) -> dict:
    return json.loads(path.read_text())


def head_metrics(blob: dict, adapter: str) -> dict[int, dict]:
    return {int(h["owner"]): h for h in blob["adapters"][adapter]["heads"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="append", default=[],
                    help="name=path pairs from eval_routescout.py")
    ap.add_argument("--curve", action="append", default=[],
                    help="metrics.json files from the sweep, ordered by scale")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--metric", default="weighted_mass4")
    args = ap.parse_args()

    out: dict[str, object] = {}

    if args.eval:
        names = []
        per_adapter: dict[str, dict[int, dict]] = {}
        for item in args.eval:
            name, path = item.split("=", 1)
            blob = load_eval(Path(path).expanduser())
            # The first adapter listed is the reference; compare all to it.
            for adapter in blob["adapters"]:
                per_adapter[adapter] = head_metrics(blob, adapter)
                names.append(adapter)
        owners = sorted(set().union(*(set(h) for h in per_adapter.values())))
        rows = []
        for owner in owners:
            row = {"owner": owner}
            for adapter, heads in per_adapter.items():
                row[adapter] = heads.get(owner, {}).get(args.metric, float("nan"))
            rows.append(row)
        out["per_layer"] = rows
        out["metric"] = args.metric

        print(f"### Per-layer {args.metric}, held-out\n")
        header = "layer".ljust(6) + "".join(a.rjust(14) for a in per_adapter)
        print(header)
        for row in rows:
            print(str(row["owner"]).ljust(6) + "".join(f"{row[a]:14.4f}" for a in per_adapter))

        print("\n### Summary per adapter\n")
        for adapter, heads in per_adapter.items():
            vals = [h.get(args.metric, np.nan) for h in heads.values()]
            vals = [v for v in vals if np.isfinite(v)]
            ranked = sorted(heads.items(), key=lambda kv: kv[1].get(args.metric, -1), reverse=True)
            print(
                f"  {adapter:22} mean={np.mean(vals):.4f} min={np.min(vals):.4f} "
                f"max={np.max(vals):.4f} spread={np.max(vals) - np.min(vals):.4f}"
            )
            print(f"    strongest: {[o for o, _ in ranked[:4]]}")
            print(f"    weakest:   {[o for o, _ in ranked[-4:]]}")

        if "edge0_published" in per_adapter and len(per_adapter) > 1:
            ref = per_adapter["edge0_published"]
            for adapter, heads in per_adapter.items():
                if adapter == "edge0_published":
                    continue
                worse = [
                    o for o in owners
                    if o in ref and o in heads
                    and heads[o].get(args.metric, np.nan) <= ref[o].get(args.metric, np.nan)
                ]
                print(f"  {adapter}: heads not beating published Edge0 on {args.metric}: {worse}")

    if args.curve:
        blobs = []
        for item in args.curve:
            name, path = item.split("=", 1)
            blobs.append((name, json.loads(Path(path).expanduser().read_text())))
        if len(blobs) >= 2:
            print("\n### Per-layer improvement between consecutive scales\n")
            out["curve_layer_deltas"] = []
            for (na, a), (nb, b) in zip(blobs, blobs[1:]):
                ha = {int(h["owner"]): h["best"].get(args.metric, np.nan) for h in a["heads"]}
                hb = {int(h["owner"]): h["best"].get(args.metric, np.nan) for h in b["heads"]}
                owners = sorted(set(ha) & set(hb))
                deltas = {o: hb[o] - ha[o] for o in owners}
                stalled = [o for o in owners if deltas[o] <= 0.0]
                weak = sorted(owners, key=lambda o: deltas[o])[:6]
                print(
                    f"  {na} -> {nb}: mean_delta={np.mean(list(deltas.values())):+.4f} "
                    f"improved={sum(1 for d in deltas.values() if d > 0)}/{len(owners)} "
                    f"stalled={stalled}"
                )
                print(f"    smallest deltas (layer: delta): {[(o, round(deltas[o], 4)) for o in weak]}")
                out["curve_layer_deltas"].append({
                    "from": na, "to": nb,
                    "mean_delta": float(np.mean(list(deltas.values()))),
                    "improved": sum(1 for d in deltas.values() if d > 0),
                    "total": len(owners),
                    "stalled_layers": stalled,
                    "deltas": {str(o): float(deltas[o]) for o in owners},
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nOUTPUT {args.output}")


if __name__ == "__main__":
    main()
