#!/usr/bin/env python3
"""Reduce sweep metric files into the EXP-078 learning-curve and comparison tables.

Consumes the per-run `*.metrics.json` written by `run_routescout_sweep.py` and
the comparison JSON from `eval_routescout.py`, and emits one JSON plus
markdown-ready tables. Kept separate from measurement so the numbers in the
report are a mechanical reduction of preserved artifacts rather than retyped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = (
    "loss", "exact_top1", "top1_in_k4", "recall1", "recall4", "recall8",
    "recall12", "weighted_mass4", "weighted_mass8", "full4",
)


def mean_over_heads(blob: dict, phase: str) -> dict[str, float]:
    out = {}
    for key in KEYS:
        vals = []
        for h in blob["heads"]:
            block = h[phase]
            # The trainer writes `top1_in_target`; the ledger writes `top1_in_k4`.
            value = block.get(key)
            if value is None and key == "top1_in_k4":
                value = block.get("top1_in_target")
            if value is not None and np.isfinite(value):
                vals.append(value)
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def mean_final_epoch(blob: dict) -> dict[str, float]:
    """Mean over heads of the LAST epoch's validation metrics.

    `best_epoch` is chosen to maximise weighted mass@4 *on this same validation
    bank*, so `best` is optimistic: every head is scored at the epoch that
    happened to look best on the data doing the selecting. The final-epoch column
    removes that bias, at the cost of including any late-epoch overfit. Both are
    reported; the frozen test bank scored with `--all-holdout` is the unbiased
    headline because no checkpoint selection touches it.
    """
    out = {}
    for key in KEYS:
        vals = []
        for h in blob["heads"]:
            history = h.get("history") or []
            if not history:
                continue
            block = history[-1]
            value = block.get(key)
            if value is None and key == "top1_in_k4":
                value = block.get("top1_in_target")
            if value is not None and np.isfinite(value):
                vals.append(value)
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def scale_sort_key(name: str) -> float:
    try:
        return float(name.rstrip("kK")) * 1000
    except ValueError:
        return float("inf")


def md_table(rows: list[list[str]], header: list[str]) -> str:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    out = [line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--compare", action="append", default=[],
                    help="name=path of eval_routescout.py JSON for the comparison table")
    args = ap.parse_args()

    blobs = []
    for path in sorted(args.runs_dir.glob("routescout_*.metrics.json")):
        blob = json.loads(path.read_text())
        if not blob.get("heads"):
            continue
        blobs.append(blob)

    # Learning curve: one row per (scale, arm).
    row_map: dict[tuple[str, str], dict] = {}
    for blob in blobs:
        key = (blob.get("scale_name", "?"), blob.get("arm", "?"))
        row_map[key] = blob

    curve = []
    for (scale, arm), blob in row_map.items():
        base = mean_over_heads(blob, "baseline")
        best = mean_over_heads(blob, "best")
        final = mean_final_epoch(blob)
        best_epochs = [h["best_epoch"] for h in blob["heads"]]
        curve.append({
            "scale": scale,
            "arm": arm,
            "runs": blob.get("run_count"),
            "train_examples_per_head": blob.get("train_examples_per_head"),
            "val_examples_per_head": blob.get("val_examples_per_head"),
            "wall_seconds": blob.get("wall_seconds"),
            "init": blob.get("init"),
            "best_epoch_mean": float(np.mean(best_epochs)),
            "best_epoch_median": float(np.median(best_epochs)),
            "baseline": base,
            "best": best,
            "final_epoch": final,
            "delta": {k: best[k] - base[k] for k in KEYS},
        })
    curve.sort(key=lambda r: (scale_sort_key(r["scale"]), r["arm"]))

    result: dict[str, object] = {"learning_curve": curve}

    print("\n### Learning curve (mean over 32 heads)\n")
    print("`best` is selected ON this bank (per-head best weighted mass@4), so it is")
    print("optimistically biased; `final` is the last epoch, unbiased but includes any")
    print("late-epoch overfit. The frozen test-bank comparison is the unbiased headline.\n")
    rows = []
    for r in curve:
        rows.append([
            r["scale"], r["arm"],
            str(r["train_examples_per_head"]),
            f"{r['best']['recall4']:.4f}",
            f"{r['final_epoch']['recall4']:.4f}",
            f"{r['best']['weighted_mass4']:.4f}",
            f"{r['final_epoch']['weighted_mass4']:.4f}",
            f"{r['best']['top1_in_k4']:.4f}",
            f"{r['best']['recall8']:.4f}",
            f"{r['best']['full4']:.4f}",
            f"{r['best']['loss']:.4f}",
            f"{r['best_epoch_mean']:.1f}",
            f"{r['wall_seconds']:.0f}" if r["wall_seconds"] else "n/a",
        ])
    table = md_table(
        rows,
        ["scale", "arm", "train/head", "r@4 best", "r@4 final", "mass@4 best",
         "mass@4 final", "top1 best", "recall@8", "full@4", "CE", "best_ep", "wall_s"],
    )
    print(table)
    result["learning_curve_md"] = table

    if args.compare:
        comparisons = {}
        for item in args.compare:
            name, path = item.split("=", 1)
            obj = json.loads(Path(path).expanduser().read_text())
            comparisons[name] = obj
        result["comparisons"] = comparisons
        print("\n### Adapter comparison on the held-out set\n")
        rows = []
        columns = ("recall4", "weighted_mass4", "top1_in_k4", "recall8", "full4", "loss")
        for group, obj in comparisons.items():
            for adapter, blob in obj["adapters"].items():
                m = blob.get("mean") or {}
                row = [group, adapter]
                for key in columns:
                    value = m.get(key)
                    if value is None and key == "top1_in_k4":
                        value = m.get("top1_in_target")
                    row.append(f"{value:.4f}" if value is not None else "n/a")
                rows.append(row)
        table = md_table(
            rows,
            ["split", "adapter", "recall@4", "mass@4", "top1_in_K4",
             "recall@8", "full@4", "CE"],
        )
        print(table)
        result["comparison_md"] = table

        # Nominate the best RouteScout checkpoint by the handoff's rule on
        # held-out data: weighted mass@4, then recall@4, then top1-in-K4, then CE.
        #
        # Control arms are excluded. `routescout_v1` is the previous head (the
        # baseline being beaten), and the `random` arm is a deliberate
        # from-scratch control that exists to measure init quality — nominating
        # either as "the best RouteScout head" would be a category error even if
        # one happened to score highest.
        candidates = []
        for group, obj in comparisons.items():
            for adapter, blob in obj["adapters"].items():
                if not adapter.startswith("routescout_") or adapter == "routescout_v1":
                    continue
                if adapter.endswith("_random"):
                    continue
                m = blob.get("mean") or {}
                mass = m.get("weighted_mass4")
                if mass is None or not np.isfinite(mass):
                    # Without the primary key this candidate cannot be ranked. Skip
                    # rather than raise: an unattended run must not lose the whole
                    # summary to one adapter that predates the full metric set.
                    print(f"note: {adapter} has no weighted_mass4; excluded from ranking")
                    continue
                # Secondary keys are optional; a missing one contributes a neutral 0
                # so ranking still falls back to the next key in the rule.
                def _f(key: str) -> float:
                    value = m.get(key)
                    if value is None and key == "top1_in_k4":
                        value = m.get("top1_in_target")
                    return float(value) if value is not None and np.isfinite(value) else 0.0
                candidates.append((
                    (mass, _f("recall4"), _f("top1_in_k4"), -_f("loss")),
                    adapter, m,
                ))
        candidates.sort(reverse=True, key=lambda c: c[0])
        if candidates:
            _key, best_name, best_m = candidates[0]
            result["best_checkpoint"] = {"adapter": best_name, "metrics": best_m}
            print("\n### Best RouteScout checkpoint by held-out rule (mass@4, recall@4, top1-in-K4, CE)\n")
            print(f"  {best_name}")
            for k in ("recall4", "weighted_mass4", "top1_in_k4", "recall8", "full4", "loss"):
                value = best_m.get(k)
                if value is None and k == "top1_in_k4":
                    value = best_m.get("top1_in_target")
                print(f"    {k:16} {value:.4f}" if value is not None else f"    {k:16} n/a")
            print("\n  full ranking (score = mass@4, recall@4, top1-in-K4):")

            def _fmt(m: dict, key: str) -> str:
                value = m.get(key)
                if value is None and key == "top1_in_k4":
                    value = m.get("top1_in_target")
                return f"{value:.4f}" if value is not None else "n/a"

            for key, name, m in candidates:
                print(f"    {name:34} mass@4={_fmt(m, 'weighted_mass4')} "
                      f"recall@4={_fmt(m, 'recall4')} top1={_fmt(m, 'top1_in_k4')} "
                      f"CE={_fmt(m, 'loss')}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nOUTPUT {args.output}")


if __name__ == "__main__":
    main()
