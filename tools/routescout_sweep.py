#!/usr/bin/env python3
"""Paired RouteScout budget/policy sweep on the raw-MLX SSD-streaming path.

Runs N arms in an alternating order within each pair (`B C C B` for two arms),
which cancels monotonic drift (thermal, page-cache warm-up, background load) that
a single block per arm absorbs into the comparison.

Records the decode-window figures that EXP-029 made interpretable: because
`profile_summary` now reports deltas from the decode boundary, `wait_ms_per_token`
and `load_ms_per_token` are comparable across arms, and the speculative counters
are this sweep's own bytes rather than lifetime totals including prefill.

Usage:
    tools/routescout_sweep.py OUT_DIR --pairs 3 --tokens 24 \
        base: \
        b1:QWEN_ROUTE_PREDICT=1,QWEN_ROUTE_PREDICT_PREFETCH=1,QWEN_ROUTE_PREDICT_BUDGET=1

The first arm is the baseline. `BASE_ENV` (default `LOGAN_EXPERT_NOCACHE=1`)
applies to every arm; the swept variable is predictor policy, never the storage
path.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

# Every per-token metric the handoff requires be first-class.
METRIC_PATTERNS = {
    "decode_ms_per_token": r"decode_ms_per_token=([0-9.]+)|decode=([0-9.]+) ms/tok",
    "prefill_ms": r"prefill=([0-9.]+) ms",
    "total_ms": r"total=([0-9.]+) ms",
    "calls": r"calls=(\d+) ",
    "calls_per_token": r"calls_per_token=([0-9.]+)",
    "load_ms": r"load_ms_per_token=([0-9.]+)",
    "compute_ms": r"compute_ms_per_token=([0-9.]+)",
    "plan_ms": r"plan_ms_per_token=([0-9.]+)",
    "wait_ms": r"wait_ms_per_token=([0-9.]+)",
    "mio_loads": r"mio loads=(\d+)",
    "mio_bytes": r"mio bytes=(\d+)",
    "mio_waits": r"mio waits=(\d+)",
    "mio_fails": r"mio fails=(\d+)",
    "pref_loads": r"mio-prefetch: loads=(\d+)",
    "pref_used": r"mio-prefetch: loads=\d+ used=(\d+)",
    "pref_wasted": r"used=\d+ wasted=(\d+)",
    "ready": r"wasted=\d+ ready_at_demand=(\d+)",
    "late": r"ready_at_demand=\d+ late_at_demand=(\d+)",
    "peak_out": r"late_at_demand=\d+ outstanding=\d+ peak=(\d+)",
    "wait_total_ms": r"wait_total_ms=([0-9.]+)",
    "p50_ms": r"p50_ms=([0-9.]+)",
    "p99_ms": r"p99_ms=([0-9.]+)",
    "max_ms": r"max_ms=([0-9.]+)",
    "predicted": r"route-arrival: correct=\d+ predicted=(\d+)",
    "arrival_correct": r"route-arrival: correct=(\d+)",
    "arrival_actual": r"correct=\d+ predicted=\d+ actual=(\d+)",
    "pairs": r"predicted=\d+ actual=\d+ pairs=(\d+)",
    "precision": r"pairs=\d+ precision=([0-9.]+)",
    "recall": r"precision=[0-9.]+ recall=([0-9.]+)",
    "transition_bytes": r"transition_bytes=(\d+)",
    "plan_hits": r"plan_hits=(\d+)",
    "plan_misses": r"plan_misses=(\d+)",
    "affine_metal": r"mlx-affine: metal=(\d+)",
    "fill_ms": r"fill=([0-9.]+)",
}
GENERATED = r"generated: (\[[^\]]*\])"


def parse(log: str) -> dict:
    import re

    out: dict = {}
    for name, pat in METRIC_PATTERNS.items():
        ms = re.findall(pat, log)
        if ms:
            last = ms[-1]
            # Alternation patterns yield tuples; take the first populated group.
            if isinstance(last, tuple):
                last = next((v for v in last if v), "")
            if last != "":
                out[name] = float(last)
    g = re.findall(GENERATED, log)
    out["generated"] = g[-1] if g else "MISSING"
    out["window"] = (
        "decode" if "logan profile-window: decode" in log else "lifetime"
    )
    return out


def run_one(binary: str, model: str, env_base: dict, extra: dict, tokens: int,
            prompt: str, log_path: Path) -> dict:
    env = dict(os.environ)
    env.update(env_base)
    env.update(extra)
    env["LOGAN_PROFILE"] = "1"
    env["QWEN_MAX_NEW"] = str(tokens)
    env["QWEN_PROMPT"] = prompt
    with log_path.open("w") as fh:
        proc = subprocess.run([binary, model], stdout=fh, stderr=subprocess.STDOUT, env=env)
    log = log_path.read_text()
    row = parse(log)
    row["exit"] = proc.returncode
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("arms", nargs="+", help="name:ENV=VAL,ENV=VAL (first is baseline)")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--prompt", default="1 2 3 4 5 6 7 8")
    ap.add_argument("--model", default=os.path.expanduser(
        "~/models/Qwen3.6-35B-A3B-MLX-oQ4-FP16"))
    ap.add_argument("--bin", default="./target/release/logan-qwen4")
    ap.add_argument("--base-env", default="LOGAN_EXPERT_NOCACHE=1")
    args = ap.parse_args()

    def kv(spec: str) -> dict:
        d = {}
        for part in filter(None, spec.split(",")):
            k, _, v = part.partition("=")
            d[k] = v
        return d

    names, extras = [], []
    for spec in args.arms:
        name, _, rest = spec.partition(":")
        names.append(name)
        extras.append(kv(rest))

    env_base = kv(args.base_env)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Alternating order within each pair: forward then reversed, so each arm
    # occupies the early and late slot equally often.
    order = list(range(len(names))) + list(reversed(range(len(names))))

    rows = []
    index = 0
    for pair in range(1, args.pairs + 1):
        for arm in order:
            index += 1
            name = names[arm]
            log_path = out / f"run-{index:03d}-p{pair}-{name}.log"
            row = run_one(args.bin, args.model, env_base, extras[arm], args.tokens,
                          args.prompt, log_path)
            row.update({"index": index, "pair": pair, "arm": name})
            rows.append(row)
            print(f"[{index:3d} p{pair} {name:>6}] "
                  f"decode={row.get('decode_ms_per_token', 'NA')} "
                  f"wait={row.get('wait_ms', 0)} load={row.get('load_ms', 0)} "
                  f"compute={row.get('compute_ms', 0)} "
                  f"pref={row.get('pref_loads', 0)} used={row.get('pref_used', 0)} "
                  f"wasted={row.get('pref_wasted', 0)} peak={row.get('peak_out', 0)} "
                  f"window={row['window']}", flush=True)

    keys = sorted({k for r in rows for k in r})
    tsv = out / "sweep.tsv"
    with tsv.open("w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")

    # Every arm must have marked the decode boundary, or its counters silently
    # report the prefill-inflated lifetime values this sweep exists to avoid
    # (EXP-029). A missing boundary is a wiring bug, not a slow run.
    bad_window = [r for r in rows if r["window"] != "decode"]
    if bad_window:
        print("\n!! WINDOW FAILURE: these runs did not report a decode window:")
        for r in bad_window:
            print(f"   index={r['index']} arm={r['arm']} window={r['window']}")
        print("   Counters in those runs are lifetime totals; do not compare them.")

    # Per-run logs are the auditable evidence; keep the decode figures beside the
    # counters so a reviewer can re-derive the medians without re-running.
    print("\n== per-run decode (ms/token) ==")
    for r in rows:
        print(f"  {r['index']:3d} p{r['pair']} {r['arm']:>16} "
              f"decode={r.get('decode_ms_per_token', 'NA')}")

    # Correctness gate: every arm must produce identical greedy IDs.
    gens = {r["arm"]: r["generated"] for r in rows}
    distinct = set(gens.values())
    print("\n== correctness ==")
    print(f"distinct generated sequences: {len(distinct)}"
          f"{' OK' if len(distinct) == 1 else ' MISMATCH'}")
    for arm, g in gens.items():
        print(f"  {arm}: {g[:80]}{'...' if len(g) > 80 else ''}")

    base = names[0]
    print(f"\n== paired decode ms/token (baseline = {base}) ==")
    summary = {}
    for name in names:
        vals = [r["decode_ms_per_token"] for r in rows
                if r["arm"] == name and "decode_ms_per_token" in r]
        if not vals:
            continue
        summary[name] = {
            "median": statistics.median(vals),
            "mean": statistics.fmean(vals),
            "n": len(vals),
            "values": vals,
        }
    for name, s in summary.items():
        delta = ""
        if name != base and base in summary:
            b = summary[base]["median"]
            pct = (s["median"] - b) / b * 100 if b else 0.0
            delta = f"  delta={pct:+.2f}%"
        print(f"  {name:>6}: median={s['median']:.1f} mean={s['mean']:.1f} "
              f"n={s['n']}{delta}")

    # Paired wins, matched within pair: same pair index, arm vs baseline.
    if base in summary:
        print(f"\n== paired wins vs {base} (per pair median) ==")
        by_pair: dict = {}
        for r in rows:
            if "decode_ms_per_token" not in r:
                continue
            by_pair.setdefault((r["pair"], r["arm"]), []).append(r["decode_ms_per_token"])
        for name in names:
            if name == base:
                continue
            wins = ties = losses = 0
            deltas = []
            for pair in range(1, args.pairs + 1):
                bv = by_pair.get((pair, base))
                cv = by_pair.get((pair, name))
                if not bv or not cv:
                    continue
                b = statistics.median(bv)
                c = statistics.median(cv)
                deltas.append((c - b) / b * 100 if b else 0.0)
                if c < b:
                    wins += 1
                elif c > b:
                    losses += 1
                else:
                    ties += 1
            if deltas:
                print(f"  {name:>6}: wins={wins} losses={losses} ties={ties} "
                      f"median_delta={statistics.median(deltas):+.2f}%")

    print(f"\n== speculation (decode window) ==")
    for name in names:
        sel = [r for r in rows if r["arm"] == name]
        if not sel:
            continue
        def tot(k):
            return sum(r.get(k, 0) for r in sel)
        print(f"  {name:>6}: pref_loads={tot('pref_loads'):.0f} used={tot('pref_used'):.0f} "
              f"wasted={tot('pref_wasted'):.0f} ready={tot('ready'):.0f} "
              f"late={tot('late'):.0f} pref_MiB={tot('mio_bytes')/1048576:.0f} "
              f"peak_out={max((r.get('peak_out', 0) for r in sel), default=0):.0f}")

    (out / "summary.json").write_text(json.dumps(
        {"rows": rows, "medians": summary, "pairs": args.pairs}, indent=2))
    print(f"\nwrote {tsv} and {out/'summary.json'}")
    return 0 if len(distinct) == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
