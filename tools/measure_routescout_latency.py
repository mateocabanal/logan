#!/usr/bin/env python3
"""Measure RouteScout/Edge0 head inference cost on this host (MLX path only).

This measures the head as an MLX module over real trace hidden states: the
offline/training-path cost, and the cleanest apples-to-apples number across
adapters because it holds the evaluator fixed and varies only the weights.

It does **not** measure the deployed FP16/BNNS path. That number comes from the
runtime's own `predict=` span:

```bash
LOGAN_PROFILE=1 QWEN_ROUTE_MODE=hybrid QWEN_HYBRID_FUSION=edge0 \
  QWEN_EDGE0_PREROUTER=<adapter> \
  target/release/examples/decode_bench <model> <tokens> greedy "<prompt>" 2>&1 |
  grep -oE 'route=[0-9.]+ predict=[0-9.]+'
```

`predict=` isolates learned-head evaluation from the native gate's `route=`. The
control that proves the span is specific: with `QWEN_ROUTE_MODE=native-truncated`
and no predictor loaded, `predict=` reads exactly `0.0`.

NEITHER number authorizes a speed claim: EXP-073 established that speculative
staging is bandwidth-bound regardless of predictor accuracy.

**This refuses to run while another model process is alive.** On this 16 GB host a
second mmap'd copy of the 20 GB checkpoint puts the machine into its documented
swap-storm regime and inflates every timing, so a contended reading is worse than
no reading. `--allow-contended` records the reading but labels it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def competing_processes() -> list[str]:
    out = subprocess.run(["ps", "-eo", "pid,command"], text=True,
                         stdout=subprocess.PIPE).stdout
    return [
        line.strip() for line in out.splitlines()[1:]
        if "decode_bench" in line or "collect_routescout_corpus" in line
    ]


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--adapter", action="append", required=True, help="name=path")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--owners", default="6-37")
    ap.add_argument("--allow-contended", action="store_true")
    args = ap.parse_args()

    busy = competing_processes()
    if busy and not args.allow_contended:
        raise SystemExit(
            "refusing to measure under contention; these processes hold a second "
            "copy of the model:\n  " + "\n  ".join(busy) +
            "\nStop collection first, or pass --allow-contended to record a "
            "deliberately-labelled contended reading."
        )

    here = Path(__file__).resolve().parent
    tr = import_module("edge0_train", here / "train_edge0_router.py")
    mx = tr.mx

    if "-" in args.owners and "," not in args.owners:
        lo, hi = (int(x) for x in args.owners.split("-", 1))
        owners = list(range(lo, hi + 1))
    else:
        owners = [int(x) for x in args.owners.split(",") if x.strip()]

    adapters = {}
    for item in args.adapter:
        name, path = item.split("=", 1)
        adapters[name] = tr.load_adapter(Path(path).expanduser())

    result: dict[str, object] = {
        "trace_dir": str(args.trace_dir),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "contended": bool(busy),
        "contending_processes": busy,
        "adapters": {},
    }

    for name, tensors in adapters.items():
        per_head_us: list[float] = []
        for owner in owners:
            trace = tr.load_trace(args.trace_dir / f"owner-{owner:02}.e0trace")
            n = min(args.repeats + args.warmup, trace.n)
            ix = np.arange(n)
            x, _target, _w = tr.make_batch(trace, ix)
            prefix = f"layers.{owner}"
            params = [
                mx.array(tensors[f"{prefix}.fc1.weight"].astype(np.float32)),
                mx.array(tensors[f"{prefix}.fc2.weight"].astype(np.float32)),
                mx.array(tensors[f"{prefix}.linear_init.weight"].astype(np.float32)),
            ]
            mx.eval(params, x)

            def head(p, xx):
                return tr.logits_for(p, xx)

            # Warm up, then time single-sample forwards (the decode shape).
            for _ in range(args.warmup):
                out = head(params, x[:1])
                mx.eval(out)
            samples: list[float] = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                out = head(params, x[:1])
                mx.eval(out)
                samples.append(time.perf_counter() - t0)
            per_head_us.append(statistics.median(samples) * 1e6)

        result["adapters"][name] = {
            "per_head_us_median": per_head_us,
            "mean_head_us": float(np.mean(per_head_us)),
            "median_head_us": float(np.median(per_head_us)),
            "total_32_heads_ms": float(np.sum(per_head_us) / 1000.0),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"latency: heads={len(owners)} repeats={args.repeats} (single-sample MLX forward)")
    for name, blob in result["adapters"].items():
        print(
            f"  {name:22} median_head={blob['median_head_us']:7.1f} us  "
            f"mean_head={blob['mean_head_us']:7.1f} us  "
            f"sum_32_heads={blob['total_32_heads_ms']:6.2f} ms/token"
        )
    print(f"OUTPUT {args.output}")


if __name__ == "__main__":
    main()
