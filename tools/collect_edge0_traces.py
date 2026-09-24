#!/usr/bin/env python3
"""Collect a reproducible pilot corpus for Edge0-style Logan prerouter training."""

from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path

PROMPTS = [
    "Explain Rust ownership and borrowing, then show a safe linked-list design tradeoff.",
    "Write a careful plan for debugging a data race in a multithreaded C++ program.",
    "Solve this step by step: if a cache has a 70% hit rate and hits take 2 ns while misses take 80 ns, what is average access time?",
    "Explain mixture-of-experts routing and why expert locality matters for SSD-streamed inference.",
    "Design a small Python program that parses a CSV, groups rows by user, and reports the top three totals.",
    "Compare TCP and QUIC for a latency-sensitive application, including failure and congestion behavior.",
    "Explain why cancer is difficult to cure without using metaphors; distinguish mutation, selection, and treatment resistance.",
    "Write a short science-fiction scene in which an engineer discovers that a machine predicts which memory pages will be needed next.",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--binary", type=Path, default=Path("target/release/examples/decode_bench"))
    ap.add_argument("--runs", type=int, default=len(PROMPTS))
    args = ap.parse_args()
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    t0 = time.time()
    for i in range(args.runs):
        prompt = PROMPTS[i % len(PROMPTS)]
        seed = 2026092300 + i
        env = os.environ.copy()
        env.update({
            "QWEN_EDGE0_TRACE_DIR": str(args.trace_dir.resolve()),
            "QWEN_ROUTE_MODE": "native-truncated",
            "QWEN_ROUTE_NATIVE_K": "4",
            "LOGAN_EXPERT_NOCACHE": "1",
            "BENCH_SEED": str(seed),
            "BENCH_TEMP": "0.8",
            "BENCH_TOP_P": "0.95",
            "BENCH_TOP_K": "0",
        })
        cmd = [str(args.binary.resolve()), str(args.model.resolve()), str(args.tokens), "sample", prompt]
        print(json.dumps({"event":"run_start","run":i,"seed":seed,"prompt":prompt}), flush=True)
        started = time.time()
        proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
        entry = {
            "run": i,
            "seed": seed,
            "prompt": prompt,
            "returncode": proc.returncode,
            "seconds": time.time() - started,
            "stdout": proc.stdout,
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        }
        logs.append(entry)
        print(json.dumps({"event":"run_done","run":i,"returncode":proc.returncode,"seconds":entry["seconds"]}), flush=True)
        if proc.returncode != 0:
            (args.trace_dir / "collection.json").write_text(json.dumps(logs, indent=2))
            raise SystemExit(proc.returncode)
    manifest = {
        "format": "logan-edge0-pilot-collection-v1",
        "model": str(args.model.resolve()),
        "tokens_per_run": args.tokens,
        "mode": "sample",
        "temperature": 0.8,
        "top_p": 0.95,
        "route_mode": "native-truncated",
        "native_k": 4,
        "runs": logs,
        "total_seconds": time.time() - t0,
    }
    (args.trace_dir / "collection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event":"collection_done","runs":len(logs),"seconds":manifest["total_seconds"],"trace_dir":str(args.trace_dir)}))

if __name__ == "__main__":
    main()
