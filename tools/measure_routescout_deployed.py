#!/usr/bin/env python3
"""Measure the deployed FP16/BNNS predictor path on an idle host.

Two numbers, both required by the EXP-078 mission and neither available from the
MLX harness:

1. **Predictor cost per token** — `predict=` from the runtime's own profile
   summary, A/B'd against a control with no predictor loaded (`predict=` must
   read exactly 0.0 there). `route=` is reported alongside so the predictor's
   cost can be compared with the native gate it sits beside.
2. **Runtime prediction quality** — `recall=` and `full_route_coverage=` from the
   hybrid staging arena, which is the evaluator-independent confirmation that a
   trained adapter actually helps on the path Logan runs.

**This refuses to run while another model process is alive.** On this 16 GB host a
second mmap'd copy of the 20 GB checkpoint puts the machine into its documented
swap-storm regime, which inflates decode spans by roughly an order of magnitude.
Measurements taken under contention are worse than no measurement, because they
look like data. `--allow-contended` exists only to re-take a deliberately labelled
contended reading.

Run it after collection has stopped. `load_ms` is reported so a contended reading
is obvious after the fact (an idle load is ~1 s; a swap-storm load is minutes).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

PROFILE_RE = re.compile(r"route=([0-9.]+) predict=([0-9.]+)")
STAGE_RE = re.compile(
    r"hybrid-stage arm=(\S+) M=(\d+) hits=(\d+) late=(\d+) misses=(\d+) "
    r"demand_reads=(\d+) duplicate_reads=(\d+) .*?efficiency=([0-9.]+) "
    r"recall=([0-9.]+) full_route_coverage=([0-9.]+)"
)
IDS_RE = re.compile(r"^BENCH ids=([0-9,]+)$", re.MULTILINE)
LOAD_RE = re.compile(r"^BENCH load_ms=([0-9.]+)$", re.MULTILINE)
TOKS_RE = re.compile(r"^BENCH decode_tok_s=([0-9.]+)$", re.MULTILINE)


def competing_processes() -> list[str]:
    out = subprocess.run(["ps", "-eo", "pid,command"], text=True,
                         stdout=subprocess.PIPE).stdout
    bad = []
    for line in out.splitlines()[1:]:
        if "decode_bench" in line or "collect_routescout_corpus" in line:
            bad.append(line.strip())
    return bad


def run_bench(binary: Path, model: Path, adapter: Path | None, tokens: int,
              prompt: str, seed: int, mode: str) -> dict:
    env = os.environ.copy()
    env.update({
        "LOGAN_EXPERT_NOCACHE": "1",
        "LOGAN_PROFILE": "1",
        "BENCH_SEED": str(seed),
        "QWEN_ROUTE_NATIVE_K": "4",
    })
    if mode == "hybrid":
        env.update({
            "QWEN_ROUTE_MODE": "hybrid",
            "QWEN_HYBRID_FUSION": "edge0",
            "QWEN_HYBRID_RESIDENT_PRIOR": "0",
            "QWEN_HYBRID_STAGE_M": "4",
        })
        if adapter is not None:
            env["QWEN_EDGE0_PREROUTER"] = str(adapter)
    else:
        env["QWEN_ROUTE_MODE"] = "native-truncated"

    cmd = [str(binary), str(model), str(tokens), "greedy", prompt]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    wall = time.time() - t0
    return {"stdout": proc.stdout, "stderr": proc.stderr, "rc": proc.returncode,
            "wall_s": round(wall, 2)}


def parse(res: dict) -> dict:
    out: dict = {"returncode": res["rc"], "wall_s": res["wall_s"]}
    m = PROFILE_RE.search(res["stderr"])
    if m:
        out["route_ms_per_tok"] = float(m.group(1))
        out["predict_ms_per_tok"] = float(m.group(2))
    m = LOAD_RE.search(res["stdout"])
    if m:
        out["load_ms"] = float(m.group(1))
    m = TOKS_RE.search(res["stdout"])
    if m:
        out["decode_tok_s"] = float(m.group(1))
    m = IDS_RE.search(res["stdout"])
    if m:
        out["ids_sha1"] = __import__("hashlib").sha1(
            m.group(1).encode()).hexdigest()[:16]
    m = STAGE_RE.search(res["stderr"])
    if m:
        out["stage"] = {
            "arm": m.group(1), "M": int(m.group(2)), "hits": int(m.group(3)),
            "late": int(m.group(4)), "misses": int(m.group(5)),
            "duplicate_reads": int(m.group(7)), "efficiency": float(m.group(8)),
            "recall": float(m.group(9)), "full_route_coverage": float(m.group(10)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path.home() / "models/Qwen3.6-35B-A3B-MLX-oQ4-FP16")
    ap.add_argument("--adapter", action="append", required=True, help="name=path")
    ap.add_argument("--binary", type=Path, default=Path("target/release/examples/decode_bench"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default="Explain how a B-tree differs from an LSM-tree for a write-heavy workload.")
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

    adapters = {}
    for item in args.adapter:
        name, path = item.split("=", 1)
        adapters[name] = Path(path).expanduser()

    result: dict = {
        "contended": bool(busy),
        "contending_processes": busy,
        "tokens": args.tokens,
        "seed": args.seed,
        "prompt": args.prompt,
        "control_native_k4": None,
        "adapters": {},
    }

    # Control: no predictor at all. predict= must be exactly 0.0, which is what
    # proves the span measures the predictor rather than some share of the gate.
    ctrl = parse(run_bench(args.binary, args.model, None, args.tokens,
                           args.prompt, args.seed, "native-truncated"))
    result["control_native_k4"] = ctrl
    print(f"control native K4: route={ctrl.get('route_ms_per_tok')} "
          f"predict={ctrl.get('predict_ms_per_tok')} load_ms={ctrl.get('load_ms')}")

    for name, path in adapters.items():
        arm = parse(run_bench(args.binary, args.model, path, args.tokens,
                              args.prompt, args.seed, "hybrid"))
        result["adapters"][name] = arm
        print(f"{name:22} route={arm.get('route_ms_per_tok')} "
              f"predict={arm.get('predict_ms_per_tok')} "
              f"recall={(arm.get('stage') or {}).get('recall')} "
              f"coverage={(arm.get('stage') or {}).get('full_route_coverage')} "
              f"tok_s={arm.get('decode_tok_s')} load_ms={arm.get('load_ms')}")

    # Every arm must emit the same tokens: prediction may change which bytes are
    # staged, never which experts execute. A mismatch invalidates the comparison.
    ids = {name: blob.get("ids_sha1") for name, blob in result["adapters"].items()}
    ids["control_native_k4"] = ctrl.get("ids_sha1")
    distinct = {v for v in ids.values() if v}
    result["token_ids_identical"] = len(distinct) <= 1
    result["token_ids"] = ids
    print(f"token ids identical across arms: {result['token_ids_identical']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"OUTPUT {args.output}")


if __name__ == "__main__":
    main()
