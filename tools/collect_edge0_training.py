#!/usr/bin/env python3
"""Collect diverse native-K4 decode traces for Edge0-style prerouter training.

This intentionally launches separate decode runs: each process receives a unique
run_id from Logan's collector, which lets train_edge0_router.py hold out entire
continuations rather than adjacent tokens from the same continuation.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROMPTS = [
    "Explain Rust ownership and borrowing to an experienced C++ programmer, including one subtle lifetime example.",
    "Design a lock-free bounded queue and discuss the memory-ordering choices you would make on ARM64.",
    "Write a careful explanation of mixture-of-experts routing, expert load balancing, and why storage locality matters.",
    "Solve this step by step: derive the gradient of softmax cross entropy and explain why subtracting the maximum logit is numerically stable.",
    "Compare virtual memory, mmap, direct I/O, and asynchronous I/O for streaming a model larger than RAM from an NVMe SSD.",
    "Explain how a modern compiler lowers a high-level loop into SSA, performs optimization, and finally emits machine code.",
    "You are reviewing a production inference engine. List the highest-risk correctness bugs around caching, cancellation, and GPU synchronization.",
    "Describe how attention KV caches scale with context length and propose practical ways to reduce their memory footprint without changing model weights.",
    "Teach me the difference between TCP congestion control and application-level backpressure using concrete examples.",
    "Give a detailed plan for benchmarking an optimization where host thermals and filesystem cache can create misleading speedups.",
    "Explain Bayesian inference from first principles, then work through a small numerical example with a biased coin.",
    "Write a short technical essay about why predictive systems should distinguish prediction quality from the cost of acting on a prediction.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--k", type=int, default=4, help="native route width and trace target width")
    ap.add_argument("--binary", type=Path, default=Path("target/release/examples/decode_bench"))
    ap.add_argument("--start", type=int, default=0, help="zero-based prompt index")
    ap.add_argument("--limit", type=int, default=len(PROMPTS), help="number of prompts from --start")
    args = ap.parse_args()

    args.trace_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.trace_dir / "collection.log"
    env_base = os.environ.copy()
    env_base.update(
        {
            "QWEN_EDGE0_TRACE_DIR": str(args.trace_dir.resolve()),
            "QWEN_ROUTE_MODE": "native-truncated",
            "QWEN_ROUTE_NATIVE_K": str(args.k),
            "QWEN_EDGE0_TRACE_K": str(args.k),
            "LOGAN_EXPERT_NOCACHE": "1",
            "BENCH_TEMP": "0.8",
            "BENCH_TOP_P": "0.95",
            "BENCH_TOP_K": "50",
        }
    )

    start = max(0, min(args.start, len(PROMPTS) - 1))
    stop = min(len(PROMPTS), start + max(1, args.limit))
    prompts = PROMPTS[start:stop]
    prompt_indices = list(range(start, stop))
    t0 = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        for local_i, (prompt_i, prompt) in enumerate(zip(prompt_indices, prompts)):
            env = dict(env_base)
            env["BENCH_SEED"] = str(0xE0D000 + prompt_i * 104729)
            cmd = [
                str(args.binary),
                str(args.model),
                str(args.tokens),
                "sample",
                prompt,
            ]
            print(
                f"COLLECT run={local_i+1}/{len(prompts)} prompt_index={prompt_i} "
                f"tokens={args.tokens} k={args.k} seed={env['BENCH_SEED']}",
                flush=True,
            )
            log.write(f"\n=== prompt_index {prompt_i} ({local_i+1}/{len(prompts)}) ===\n")
            log.write(f"prompt={prompt}\n")
            log.write(f"seed={env['BENCH_SEED']}\n")
            log.flush()
            proc = subprocess.run(
                cmd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            log.write(proc.stdout)
            log.write(proc.stderr)
            log.write(f"exit_code={proc.returncode}\n")
            log.flush()
            if proc.returncode != 0:
                print(proc.stdout, file=sys.stderr)
                print(proc.stderr, file=sys.stderr)
                raise SystemExit(proc.returncode)
            for line in proc.stdout.splitlines():
                if line.startswith("BENCH decode_tok_s=") or line.startswith("BENCH ids="):
                    print(f"  {line}", flush=True)

    elapsed = time.time() - t0
    print(f"COLLECT done runs={len(prompts)} elapsed_s={elapsed:.1f} trace_dir={args.trace_dir}", flush=True)


if __name__ == "__main__":
    main()
