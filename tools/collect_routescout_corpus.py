#!/usr/bin/env python3
"""Collect the RouteScout K4 scaling corpus (EXP-078) into one appendable dir.

Difference from `collect_edge0_training.py`:

* prompts come from `routescout_prompts.py`, split into disjoint train / val /
  test banks;
* the run order is written to `corpus-index.json`, so a *scale point* is an
  exact prefix of the corpus by collection order rather than a guess;
* `--target-tokens` stops as soon as the requested generated-token count is
  reached, which is how the 5k/10k/25k/50k scale points are produced by
  growing one directory instead of recollecting four.

Each run is a separate process, so it gets its own `run_id` from Logan's
collector and whole runs — not neighbouring tokens — can be held out later.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from routescout_prompts import TRAIN_BANK, VAL_BANK, TEST_BANK  # noqa: E402

BANKS = {"train": TRAIN_BANK, "val": VAL_BANK, "test": TEST_BANK}

# Distinct seed base per bank so val/test runs can never collide with a pool
# run's seed even if the prompt lists were ever edited to overlap.
SEED_BASE = {"train": 0xE0D000, "val": 0xE0D100, "test": 0xE0D200}


def load_index(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"format": "logan-routescout-corpus-v1", "dir": str(path.parent), "runs": []}


def save_index(path: Path, index: dict) -> None:
    path.write_text(json.dumps(index, indent=2) + "\n")


def truncate_orphan_runs(trace_dir: Path, index: dict, log_path: Path | None = None) -> int:
    """Drop records from runs that have no index entry (crashed collections).

    A run's index entry is appended only after its child process exits
    successfully. If the collector is killed mid-run, that run's records are
    already in the trace files with no index entry — and because every consumer
    reads only *indexed* runs, those orphan records would permanently block all
    later data (a prefix walk stops at the first unindexed run).

    Truncating each owner file back to the end of the last indexed run restores
    the append-only prefix invariant. Returns the number of files changed.
    """
    import struct

    known = {int(r["run_id"]) for r in index["runs"] if r.get("run_id")}
    changed = 0
    for path in sorted(trace_dir.glob("owner-*.e0trace")):
        raw = path.read_bytes()[:32]
        if len(raw) != 32:
            continue
        _m, _v, _o, hidden, _e, _k, _hb, record_bytes, _r = struct.unpack(
            "<8sIHHHHIII", raw
        )
        if record_bytes == 0:
            continue
        payload = path.stat().st_size - 32
        n = payload // record_bytes
        if n == 0:
            continue
        # Walk from the end, ignoring a partial trailing record. Reads one 8-byte
        # run id per step: reading the whole file per iteration would be
        # O(records^2) on a 50k-example corpus.
        keep = n
        with path.open("rb") as fh:
            while keep > 0:
                fh.seek(32 + (keep - 1) * record_bytes)
                rid = struct.unpack("<Q", fh.read(8))[0]
                if int(rid) in known:
                    break
                keep -= 1
        new_size = 32 + keep * record_bytes
        if new_size != path.stat().st_size:
            with path.open("r+b") as fh:
                fh.truncate(new_size)
                fh.flush()
                os.fsync(fh.fileno())
            changed += 1
            if log_path is not None:
                with log_path.open("a", encoding="utf-8") as lg:
                    lg.write(
                        f"recovered {path.name}: truncated {n} -> {keep} records "
                        f"(dropped orphan run)\n"
                    )
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--bank", choices=sorted(BANKS), default="train")
    ap.add_argument("--tokens", type=int, default=256, help="generated tokens per run")
    ap.add_argument("--k", type=int, default=4, help="native route width and trace target width")
    ap.add_argument("--target-tokens", type=int, default=0,
                    help="stop once this many generated tokens exist in the bank (0 = run the whole bank once)")
    ap.add_argument("--passes", type=int, default=1, help="how many times to cycle the bank")
    ap.add_argument("--binary", type=Path, default=Path("target/release/examples/decode_bench"))
    ap.add_argument("--start", type=int, default=0, help="zero-based prompt index within the bank")
    ap.add_argument("--limit", type=int, default=0, help="prompts from --start (0 = to end of bank)")
    ap.add_argument("--resume", action="store_true",
                    help="skip (prompt, pass) pairs already recorded in corpus-index.json")
    ap.add_argument("--max-failures", type=int, default=8,
                    help="stop after this many failed runs (a systematic fault is not transient)")
    args = ap.parse_args()

    if args.tokens < 4:
        raise SystemExit("--tokens must be at least 4")
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.trace_dir / f"collection-{args.bank}.log"
    index_path = args.trace_dir / "corpus-index.json"
    index = load_index(index_path)

    # Self-heal before appending: a previous crash may have left records from a
    # run that never got an index entry, which would block every later run from
    # being read. This is a no-op when the directory is intact.
    if index["runs"]:
        recovered = truncate_orphan_runs(args.trace_dir, index, log_path)
        if recovered:
            print(f"COLLECT recovered {recovered} owner file(s) from an interrupted run", flush=True)

    # The same prompt may legitimately be collected twice (second pass, new
    # seed) — that is a second *run*, and run-level splitting keeps those
    # separate. What must never happen is a prompt appearing in two banks.
    done_bank_tokens = sum(
        int(r["tokens"]) for r in index["runs"] if r.get("bank") == args.bank
    )

    bank = BANKS[args.bank]
    start = max(0, min(args.start, len(bank) - 1))
    stop = len(bank) if args.limit <= 0 else min(len(bank), start + args.limit)
    order = list(range(start, stop))

    # `--resume` keys off (prompt_index, pass) rather than `--start` arithmetic,
    # so a restart cannot silently re-collect an already-indexed prompt: replaying
    # one would add a near-duplicate run to the training pool with a fresh run_id
    # that the split would then treat as independent evidence.
    done_pairs = {
        (int(r["prompt_index"]), int(r.get("pass", 0)))
        for r in index["runs"]
        if r.get("bank") == args.bank and r.get("run_id") is not None
    }
    if args.resume and done_pairs:
        print(f"COLLECT resume: skipping {len(done_pairs)} already-collected "
              f"(prompt, pass) pairs", flush=True)

    failures: list[dict] = []

    t0 = time.time()
    collected = 0
    with log_path.open("a", encoding="utf-8") as log:
        for pass_i in range(args.passes):
            for step, prompt_i in enumerate(order):
                if args.target_tokens and done_bank_tokens + collected >= args.target_tokens:
                    break
                if args.resume and (prompt_i, pass_i) in done_pairs:
                    continue
                prompt = bank[prompt_i]
                seed = SEED_BASE[args.bank] + prompt_i * 104729 + pass_i
                env = os.environ.copy()
                env.update(
                    {
                        "QWEN_EDGE0_TRACE_DIR": str(args.trace_dir.resolve()),
                        "QWEN_ROUTE_MODE": "native-truncated",
                        "QWEN_ROUTE_NATIVE_K": str(args.k),
                        "QWEN_EDGE0_TRACE_K": str(args.k),
                        "LOGAN_EXPERT_NOCACHE": "1",
                        "BENCH_TEMP": "0.8",
                        "BENCH_TOP_P": "0.95",
                        "BENCH_TOP_K": "50",
                        "BENCH_SEED": str(seed),
                    }
                )
                cmd = [
                    str(args.binary),
                    str(args.model),
                    str(args.tokens),
                    "sample",
                    prompt,
                ]
                label = f"{args.bank}[{prompt_i}] pass={pass_i}"
                print(
                    f"COLLECT {label} step={step+1}/{len(order)} "
                    f"bank_tokens={done_bank_tokens + collected} k={args.k} seed={seed}",
                    flush=True,
                )
                log.write(f"\n=== {label} seed={seed} ===\n{prompt}\n")
                log.flush()
                run_t0 = time.time()
                proc = subprocess.run(cmd, env=env, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                seconds = time.time() - run_t0
                log.write(proc.stdout)
                log.write(proc.stderr)
                log.write(f"exit_code={proc.returncode} seconds={seconds:.1f}\n")
                log.flush()
                if proc.returncode != 0:
                    # One crashed run must not end a multi-hour collection. Drop
                    # any records it did write (they have no index entry and
                    # would block every later run from being read), then move on
                    # to the next prompt. There are spare prompts in the bank
                    # precisely so a few losses do not shrink the target corpus.
                    print(proc.stdout, file=sys.stderr)
                    print(proc.stderr, file=sys.stderr)
                    dropped = truncate_orphan_runs(args.trace_dir, index, log_path)
                    failures.append(
                        {"prompt_index": prompt_i, "pass": pass_i,
                         "returncode": proc.returncode, "seconds": round(seconds, 1)}
                    )
                    print(
                        f"COLLECT {label} FAILED rc={proc.returncode}; dropped "
                        f"{dropped} partial file(s); continuing. "
                        f"failures={len(failures)}",
                        flush=True,
                    )
                    log.write(f"FAILED rc={proc.returncode} dropped_files={dropped}\n")
                    log.flush()
                    if len(failures) >= args.max_failures:
                        print(
                            f"COLLECT stopping: {len(failures)} failures reached "
                            f"--max-failures={args.max_failures}",
                            flush=True,
                        )
                        break
                    continue

                run_id = None
                decode_tok_s = None
                for line in proc.stdout.splitlines():
                    if line.startswith("BENCH decode_tok_s="):
                        decode_tok_s = float(line.split("=", 1)[1])
                        print(f"  {line}", flush=True)
                for line in proc.stderr.splitlines():
                    if "armed dir=" in line and "run_id=" in line:
                        run_id = int(line.rsplit("run_id=", 1)[1].split()[0])

                index["runs"].append(
                    {
                        "bank": args.bank,
                        "prompt_index": prompt_i,
                        "pass": pass_i,
                        "prompt": prompt,
                        "seed": seed,
                        "tokens": args.tokens,
                        "k": args.k,
                        "run_id": run_id,
                        "seconds": round(seconds, 2),
                        "decode_tok_s": decode_tok_s,
                    }
                )
                index["failures"] = failures
                save_index(index_path, index)
                collected += args.tokens

            if args.target_tokens and done_bank_tokens + collected >= args.target_tokens:
                break

    elapsed = time.time() - t0
    total_bank = done_bank_tokens + collected
    print(
        f"COLLECT done bank={args.bank} new_runs={len(index['runs'])} "
        f"new_tokens={collected} bank_tokens={total_bank} elapsed_s={elapsed:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
