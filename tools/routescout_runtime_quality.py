#!/usr/bin/env python3
"""Deployed-path runtime-quality sweep across RouteScout checkpoints.

Runs the real engine (not the Python evaluator) with `QWEN_ROUTE_MODE=hybrid` for
each adapter on the same prompt and seed, and collects the staging arena's own
`recall=` and `full_route_coverage=` counters.

Why this exists separately from `measure_routescout_deployed.py`:

- that tool measures *predictor cost* and asserts token-identity across arms;
- this one measures *runtime prediction quality vs corpus scale*, which is the
  mission's "inference overhead / per-layer distribution" companion question — does
  the offline improvement survive the FP16/BNNS path?

Native K4 remains authoritative throughout: a wrong prediction changes which bytes
are staged and never which experts execute, and this script asserts that by
comparing the emitted token ids across every arm.

It does NOT refuse under contention, because its output is a *quality* number
(recall/coverage ratios), which is far less host-sensitive than a timing. Contention
is recorded in the output so a reader can see the conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

STAGE_RE = re.compile(
    r"hybrid-stage arm=(\S+) M=(\d+) hits=(\d+) late=(\d+) misses=(\d+) "
    r"demand_reads=(\d+) duplicate_reads=(\d+) stale_rejected=(\d+) unplaced=(\d+) "
    r".*?efficiency=([0-9.]+) recall=([0-9.]+) full_route_coverage=([0-9.]+)"
)
IDS_RE = re.compile(r"^BENCH ids=([0-9,]+)$", re.MULTILINE)


def live_model_processes() -> list[str]:
    out = subprocess.run(["ps", "-eo", "pid,command"], text=True,
                         stdout=subprocess.PIPE).stdout
    return [l.strip() for l in out.splitlines()[1:]
            if "decode_bench" in l or "collect_routescout_corpus" in l]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path,
                    default=Path.home() / "models/Qwen3.6-35B-A3B-MLX-oQ4-FP16")
    ap.add_argument("--binary", type=Path,
                    default=Path("target/release/examples/decode_bench"))
    ap.add_argument("--adapter", action="append", required=True, help="name=path")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt",
                    default="Explain cache locality and why it matters for expert I/O.")
    args = ap.parse_args()

    adapters = []
    for item in args.adapter:
        name, path = item.split("=", 1)
        adapters.append((name, Path(path).expanduser()))

    busy = live_model_processes()
    result: dict = {
        "prompt": args.prompt,
        "tokens": args.tokens,
        "seed": args.seed,
        "contended": bool(busy),
        "contending_processes": busy,
        "note": (
            "runtime recall/coverage are quality ratios and far less host-sensitive "
            "than timings, but this IS a single-prompt probe, not the held-out "
            "comparison; the frozen test-bank table is the unbiased headline"
        ),
        "arms": {},
    }

    for name, path in adapters:
        env = os.environ.copy()
        env.update({
            "QWEN_ROUTE_MODE": "hybrid",
            "QWEN_ROUTE_NATIVE_K": "4",
            "QWEN_HYBRID_FUSION": "edge0",
            "QWEN_HYBRID_RESIDENT_PRIOR": "0",
            "QWEN_HYBRID_STAGE_M": "4",
            "LOGAN_EXPERT_NOCACHE": "1",
            "LOGAN_PROFILE": "1",
            "BENCH_SEED": str(args.seed),
            "QWEN_EDGE0_PREROUTER": str(path),
        })
        cmd = [str(args.binary), str(args.model), str(args.tokens), "greedy", args.prompt]
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        arm: dict = {"rc": proc.returncode, "wall_s": round(time.time() - t0, 2),
                     "adapter_sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                     if path.exists() else None}
        m = STAGE_RE.search(proc.stderr)
        if m:
            arm["stage"] = {
                "arm": m.group(1), "M": int(m.group(2)), "hits": int(m.group(3)),
                "late": int(m.group(4)), "misses": int(m.group(5)),
                "duplicate_reads": int(m.group(7)), "efficiency": float(m.group(10)),
                "recall": float(m.group(11)), "full_route_coverage": float(m.group(12)),
            }
        ids = IDS_RE.search(proc.stdout)
        if ids:
            arm["ids_sha1"] = hashlib.sha1(ids.group(1).encode()).hexdigest()[:16]
        result["arms"][name] = arm
        st = arm.get("stage") or {}
        print(f"{name:16} recall={st.get('recall')} coverage={st.get('full_route_coverage')} "
              f"efficiency={st.get('efficiency')} late={st.get('late')} "
              f"dup={st.get('duplicate_reads')}")

    ids = {n: a.get("ids_sha1") for n, a in result["arms"].items() if a.get("ids_sha1")}
    result["token_ids_identical"] = len(set(ids.values())) <= 1
    print(f"token ids identical across arms: {result['token_ids_identical']}")
    if not result["token_ids_identical"]:
        print("WARNING: arms diverged; prediction changed what executed", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"OUTPUT {args.output}")


if __name__ == "__main__":
    main()
