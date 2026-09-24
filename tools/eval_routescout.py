#!/usr/bin/env python3
"""Evaluate any number of Edge0-architecture adapters on one held-out corpus.

All arms are scored on the *same* records: same run set, same per-head
validation split, same metrics function as `train_edge0_router.py`. That is the
only way the numbers are comparable, because the split is a property of the
corpus rather than of the adapter.

Usage (Phase 3 comparison):

```bash
python tools/eval_routescout.py \
  --trace-dir .perf_runs/routescout-train-v1/test \
  --adapter edge0_published=~/models/prerouter_edge0_35b.safetensors \
  --adapter routescout_v1=~/models/prerouter_logan_qwen36_v1.safetensors \
  --adapter routescout_v2=<new>.safetensors \
  --output .perf_runs/routescout-train-v1/compare-test.json
```

With `--split-bank test` the whole dir is normally held out, so every record in
it is scored (`--val-bank test --all-holdout`).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def import_trainer():
    import sys

    spec = importlib.util.spec_from_file_location(
        "edge0_train", Path(__file__).resolve().parent / "train_edge0_router.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


METRIC_KEYS = (
    "loss",
    "exact_top1",
    "top1_in_k4",
    "recall1",
    "recall4",
    "recall8",
    "recall12",
    "weighted_mass1",
    "weighted_mass4",
    "weighted_mass8",
    "weighted_mass12",
    "full4",
    "full8",
    "full12",
)


def normalize_head(head: dict) -> dict:
    """Accept the trainer's `top1_in_target` and the report's `top1_in_k4`.

    The trainer names the metric by K because K is a parameter; the experiment
    ledger calls it `top1-in-K4` because the operating point is fixed. Both names
    denote the same quantity, and every consumer should see one.
    """
    if "top1_in_k4" not in head and "top1_in_target" in head:
        head = dict(head)
        head["top1_in_k4"] = head["top1_in_target"]
    return head


def mean_metrics(per_head: list[dict]) -> dict[str, float]:
    out = {}
    for key in METRIC_KEYS:
        vals = [h[key] for h in per_head if key in h and np.isfinite(h[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--adapter", action="append", required=True, help="name=path")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--owners", default="6-37")
    ap.add_argument("--val-bank", default="test")
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--all-holdout", action="store_true",
                    help="score every record in the directory (no train/val split)")
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--live-prefix", action="store_true",
                    help="read only the longest whole-run prefix (still-collecting corpus)")
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args()

    tr = import_trainer()
    if "-" in args.owners and "," not in args.owners:
        lo, hi = (int(x) for x in args.owners.split("-", 1))
        owners = list(range(lo, hi + 1))
    else:
        owners = [int(x) for x in args.owners.split(",") if x.strip()]

    adapters: dict[str, dict[str, np.ndarray]] = {}
    for item in args.adapter:
        name, path = item.split("=", 1)
        adapters[name] = tr.load_adapter(Path(path).expanduser())
    if not adapters:
        raise SystemExit("no adapters given")

    result: dict[str, object] = {
        "trace_dir": str(args.trace_dir),
        "all_holdout": args.all_holdout,
        "val_bank": args.val_bank,
        "max_runs": args.max_runs,
        "owners": owners,
        "adapters": {},
    }
    per_adapter_heads: dict[str, list[dict]] = {name: [] for name in adapters}
    split_seen: str | None = None
    scored_records: dict[str, int] = {}

    for owner in owners:
        path = args.trace_dir / f"owner-{owner:02}.e0trace"
        if args.live_prefix:
            index = tr.load_corpus_index(args.trace_dir)
            if index is None:
                raise SystemExit(f"{args.trace_dir}: --live-prefix requires corpus-index.json")
            entries = tr.load_corpus_entries(args.trace_dir)
            allowed = set()
            for ids in index["banks"].values():
                allowed |= set(ids)
            expected = {rid: entries[rid]["tokens"] - 2 for rid in allowed if rid in entries}
            trace = tr.load_trace_prefix(path, allowed, expected)
        else:
            trace = tr.load_trace(path)
        if args.all_holdout:
            val_ix = np.arange(trace.n)
            split_desc = "all-records holdout"
        else:
            train_ix, val_ix, split_desc, _detail = tr.select_run_sets(
                args.trace_dir,
                trace.run_id,
                args.max_runs,
                args.val_bank,
                args.val_fraction if hasattr(args, "val_fraction") else 0.2,
                args.seed + owner,
            )
            if len(train_ix) and len(val_ix) == 0:
                raise SystemExit("split produced no validation records")
        if split_seen is None:
            split_seen = split_desc
        scored_records[str(owner)] = int(len(val_ix))

        for name, tensors in adapters.items():
            prefix = f"layers.{owner}"
            params = [
                tr.mx.array(tensors[f"{prefix}.fc1.weight"].astype(np.float32)),
                tr.mx.array(tensors[f"{prefix}.fc2.weight"].astype(np.float32)),
                tr.mx.array(tensors[f"{prefix}.linear_init.weight"].astype(np.float32)),
            ]
            m = tr.metrics(params, trace, val_ix, args.eval_batch_size)
            m = normalize_head(m)
            m["owner"] = owner
            per_adapter_heads[name].append(m)

    for name, heads in per_adapter_heads.items():
        result["adapters"][name] = {
            "mean": mean_metrics(heads),
            "heads": heads,
        }
    result["split"] = split_seen
    result["val_records_per_head"] = scored_records

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"EVAL split={split_seen} owners={len(owners)} records/head={next(iter(scored_records.values()), 0)}")
    header = "adapter".ljust(22) + "".join(
        k.rjust(12) for k in ("recall@4", "mass@4", "top1_in_K4", "recall@8", "full@4", "soft_CE")
    )
    print(header)
    for name, blob in result["adapters"].items():
        m = blob["mean"]
        row = name.ljust(22) + "".join(
            f"{v:12.4f}" for v in (
                m["recall4"], m["weighted_mass4"], m["top1_in_k4"],
                m["recall8"], m["full4"], m["loss"],
            )
        )
        print(row)
    print(f"OUTPUT {args.output}")


if __name__ == "__main__":
    main()
