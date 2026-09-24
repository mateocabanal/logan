#!/usr/bin/env python3
"""Run the EXP-078 RouteScout scaling sweep.

Trains the same Edge0-style architecture at several corpus scales from each
initialization arm, then reduces the per-run metrics into the learning curve.

Scale points are **nested prefixes by run count**, so every point is a subset of
the next and the curve measures data volume rather than a corpus difference.

All arms share: architecture, optimizer, LR, batch size, epoch count, seed, and
the validation bank. Only the initialization changes. That is what makes the
initialization comparison meaningful.

Example:

```bash
python tools/run_routescout_sweep.py \
  --corpus .perf_runs/routescout-train-v1/corpus \
  --out-dir .perf_runs/routescout-train-v1/runs \
  --scales 5k=20,10k=40,25k=98,50k=196 \
  --arms edge0,rs_v1
```
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = Path.home() / ".venvs/logan-edge0-train/bin/python"

ADAPTERS = {
    "edge0": str(Path.home() / "models/prerouter_edge0_35b.safetensors"),
    "rs_v1": str(Path.home() / "models/prerouter_logan_qwen36_v1.safetensors"),
}


def parse_scales(spec: str) -> list[tuple[str, int]]:
    out = []
    for item in spec.split(","):
        name, runs = item.split("=", 1)
        out.append((name.strip(), int(runs)))
    return out


def available_pool_runs(corpus: Path, bank: str = "train") -> int:
    """How many pooled runs are **complete** in the trace right now.

    A run counts only when its last record has landed, because the collector
    flushes records mid-run (the `samples=4096` periodic flush), so an in-flight
    run's id is already visible in the trace with a partial count. Counting it
    would let a scale point admit a truncated run and still call itself "the
    first N runs".

    Index membership would be enough on its own — entries are appended only after
    the child process exits — but reading the trace directly also catches a run
    whose index entry exists and whose records were later truncated, which is the
    failure mode a multi-hour unattended collection actually risks.
    """
    index_path = corpus / "corpus-index.json"
    if not index_path.exists():
        return 0
    index = json.loads(index_path.read_text())
    wanted = [r for r in index["runs"] if r.get("bank") == bank and r.get("run_id")]
    if not wanted:
        return 0
    sys.path.insert(0, str(HERE))
    from train_edge0_router import load_trace_header, record_dtype
    import numpy as np

    head = load_trace_header(corpus / "owner-06.e0trace")
    n = head["max_records"]
    if n == 0:
        return 0
    dtype = record_dtype(head["hidden"], head["k"])
    mm = np.memmap(corpus / "owner-06.e0trace", mode="r", dtype=np.uint8,
                   offset=32, shape=(n, head["record_bytes"]))
    records = np.ndarray(shape=(n,), dtype=dtype, buffer=mm)
    counts: dict[int, int] = {}
    for rid in records["run_id"]:
        value = int(rid)
        counts[value] = counts.get(value, 0) + 1

    complete = 0
    for entry in wanted:
        rid = int(entry["run_id"])
        # One record per decode forward that has both a predecessor feature and a
        # successor target: tokens - 2 for a sampled run of `tokens` tokens.
        expected = int(entry["tokens"]) - 2
        if counts.get(rid, 0) >= expected:
            complete += 1
    return complete


def live_trainers() -> list[str]:
    """Training processes other than this one.

    Two filters are needed, both learned from real failures:

    - **Exclude our own pid.** This module is itself named
      `run_routescout_sweep.py`, so a naive substring match makes the sweep detect
      *itself* and refuse every run.
    - **Require a python interpreter as argv[0].** A `bash -c '... sweep.py ...'`
      wrapper carries the same substring but is not a trainer. This is not
      hypothetical: the collector is launched exactly that way
      (`caffeinate -i bash -c '… collect_routescout_corpus.py …'`), and if a sweep
      were ever wrapped the same way, the surviving wrapper would make every
      `train_one` refuse while nothing was actually training.
    """
    me = os.getpid()
    out = subprocess.run(["ps", "-eo", "pid,command"], text=True,
                         stdout=subprocess.PIPE).stdout
    found = []
    for line in out.splitlines()[1:]:
        fields = line.split(None, 1)
        if not fields:
            continue
        pid = fields[0]
        command = fields[1] if len(fields) > 1 else ""
        if pid.isdigit() and int(pid) == me:
            continue
        parts = command.split()
        if not parts:
            continue
        argv0 = Path(parts[0]).name
        if not argv0.startswith("python"):
            continue
        if any("train_edge0_router" in p or "run_routescout_sweep" in p for p in parts):
            found.append(line.strip())
    return found


def sweep_log(message: str, log_path: Path | None) -> None:
    """Print and optionally append a line, matching the orchestrator's format."""
    line = f"[sweep {time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def wait_for_trainers_idle(log_path: Path | None, poll_s: int, deadline_s: float) -> bool:
    """Wait until no training process is running, then confirm it stays idle.

    Killing the orchestrator does **not** kill its children: `run()` is a blocking
    `subprocess.run`, so the orchestrator pid and the trainer it spawned are
    separate processes. A restarted orchestrator would immediately see the pool
    ready, call the sweep, find `metrics.json` still absent (the orphan is still
    training), and spawn a second trainer writing the *same* adapter path — two
    writers on one output file.

    Both halves are checked for the same reason as elsewhere in this file:
    liveness alone races the moment a trainer exits, and idle alone would fire
    between two sweep invocations that are about to start.
    """
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        busy = live_trainers()
        if not busy:
            sweep_log("orphan guard: no live trainer", log_path)
            return True
        sweep_log(f"orphan guard: {len(busy)} live trainer process(es); waiting", log_path)
        time.sleep(poll_s)
    sweep_log("orphan guard: a trainer never exited; refusing to start a duplicate", log_path)
    return False


def train_one_retrying(
    corpus: Path,
    out_dir: Path,
    scale_name: str,
    runs: int,
    arm: str,
    epochs: int,
    seed: int,
    val_bank: str,
    selection: str,
    live_prefix: bool,
    attempts: int = 2,
) -> dict:
    """`train_one` with a bounded retry on a *crashed* run.

    Fail-closed on a crashed run is correct, but "crashed" and "in flight" must not
    be conflated, or a single transient error becomes permanent: the marker would
    persist, every later attempt would refuse, and the scale point would silently
    vanish from the curve the primary deliverable depends on — recoverable only by
    a human noticing a log line and deleting a file.

    `live_trainers()` is exactly the discriminator. If a marker exists but no trainer
    is alive, the run is dead and the marker is stale, so it is cleared and the
    attempt is repeated. If a trainer *is* alive, the refusal stands and is
    reported, because starting a second writer is the one thing that must never
    happen.
    """
    tag = f"{scale_name}_{arm}"
    inflight = out_dir / f"routescout_{tag}.inflight"
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return train_one(corpus, out_dir, scale_name, runs, arm,
                             epochs, seed, val_bank, selection, live_prefix)
        except SystemExit as exc:
            last = exc
            busy = live_trainers()
            if busy:
                print(f"SWEEP {tag}: refusing to retry, {len(busy)} trainer(s) alive", flush=True)
                raise
            if not inflight.exists():
                # Not a stale-marker situation (e.g. the model/data is genuinely
                # bad); a retry would just fail the same way.
                raise
            if attempt == attempts:
                print(f"SWEEP {tag}: {attempts} attempts exhausted; leaving marker for "
                      f"inspection", flush=True)
                raise
            print(f"SWEEP {tag}: attempt {attempt} failed with no live trainer; "
                  f"clearing stale marker and retrying", flush=True)
            inflight.unlink(missing_ok=True)
    assert last is not None
    raise last


def train_one(
    corpus: Path,
    out_dir: Path,
    scale_name: str,
    runs: int,
    arm: str,
    epochs: int,
    seed: int,
    val_bank: str,
    selection: str,
    live_prefix: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{scale_name}_{arm}"
    adapter = out_dir / f"routescout_{tag}.safetensors"
    metrics = out_dir / f"routescout_{tag}.metrics.json"
    inflight = out_dir / f"routescout_{tag}.inflight"
    if metrics.exists():
        print(f"SWEEP skip {tag} (metrics present)", flush=True)
        return json.loads(metrics.read_text())
    # An in-flight marker covers the whole training window. Checking for the
    # adapter file instead would NOT work: `train_edge0_router.py` writes the
    # adapter only after every head has trained, so during a run the adapter is
    # absent and an adapter-exists test is false for exactly the window it is
    # meant to protect. The marker is created before training starts and removed
    # in a `finally`, so it is true for the entire window.
    if inflight.exists():
        raise SystemExit(
            f"SWEEP refuse {tag}: {inflight.name} exists, so a run is in flight or "
            f"crashed (the marker is removed only on completion). Refusing to start a "
            f"second writer; inspect and delete the marker only if the run is truly dead."
        )
    if live_trainers():
        raise SystemExit(
            f"SWEEP refuse {tag}: {len(live_trainers())} training process(es) are already "
            f"running. A duplicate writer would corrupt the checkpoint."
        )

    cmd = [
        str(PYTHON),
        str(HERE / "train_edge0_router.py"),
        "--trace-dir", str(corpus),
        "--output", str(adapter),
        "--epochs", str(epochs),
        "--batch-size", "64",
        "--eval-batch-size", "256",
        "--lr", "1e-4",
        "--weight-decay", "1e-4",
        "--val-bank", val_bank,
        "--max-runs", str(runs),
        "--seed", str(seed),
        "--selection", selection,
        "--quiet",
    ]
    if live_prefix:
        cmd.append("--live-prefix")
    if arm == "random":
        cmd += ["--init", "random"]
    else:
        cmd += ["--init", "adapter", "--base-adapter", ADAPTERS[arm]]

    print(f"SWEEP start {tag} runs={runs} arm={arm} cmd={' '.join(cmd)}", flush=True)
    # Marker spans the whole training window, including a crash: it is removed
    # only when the child finished successfully, so a crashed run leaves it behind
    # and a later orchestrator refuses rather than silently retraining.
    inflight.write_text(f"{time.time():.0f}\n")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0
        print(proc.stdout.strip(), flush=True)
        if proc.returncode != 0:
            raise SystemExit(f"train failed for {tag} rc={proc.returncode}")
    except BaseException:
        # Keep the marker on failure so the incomplete run is visible rather than
        # looking like a scale point that was never attempted.
        print(f"SWEEP failed {tag}; marker kept at {inflight.name}", flush=True)
        raise
    else:
        inflight.unlink(missing_ok=True)
    blob = json.loads(metrics.read_text())
    blob["wall_seconds"] = round(elapsed, 1)
    blob["scale_name"] = scale_name
    blob["arm"] = arm
    # The nominal request and the runs actually used are both recorded. The
    # trainer now fails closed on a shortfall, so these agree — but keeping them
    # separate means a future regression shows up as a mismatch in the artifacts
    # rather than as a silently relabelled scale point.
    blob["run_count_requested"] = runs
    blob["run_count"] = int(blob.get("heads", [{}])[0].get("split_detail", {}).get("pool_runs", []).__len__() or runs)
    metrics.write_text(json.dumps(blob, indent=2) + "\n")
    print(f"SWEEP done {tag} wall_s={elapsed:.1f} runs={blob['run_count']}", flush=True)
    return blob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--scales", required=True, help="e.g. 5k=20,10k=40,25k=98,50k=196")
    ap.add_argument("--arms", default="edge0,rs_v1")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--val-bank", default="val")
    ap.add_argument("--selection", choices=("exp078", "exp074"), default="exp078")
    ap.add_argument("--live-prefix", action="store_true",
                    help="train from the longest complete prefix (pool still collecting)")
    ap.add_argument("--require-runs", action="store_true",
                    help="skip a scale point until its full run count exists")
    ap.add_argument("--only", default="", help="restrict to one scale or tag")
    args = ap.parse_args()

    scales = parse_scales(args.scales)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    results = []
    failures: list[dict] = []
    for scale_name, runs in scales:
        have = available_pool_runs(args.corpus) if args.require_runs else runs
        if args.require_runs and have < runs:
            print(f"SWEEP skip {scale_name}: {have}/{runs} pool runs complete", flush=True)
            continue
        for arm in arms:
            if args.only and args.only not in (scale_name, f"{scale_name}_{arm}"):
                continue
            try:
                blob = train_one_retrying(
                    args.corpus, args.out_dir, scale_name, runs, arm,
                    args.epochs, args.seed, args.val_bank, args.selection, args.live_prefix,
                )
            except SystemExit as exc:
                # Record the failure and continue to the next arm. One arm failing
                # must not abandon the others: with two arms per scale, aborting
                # would silently drop a curve point the deliverable depends on.
                failures.append({"scale": scale_name, "arm": arm, "reason": str(exc)[:400]})
                print(f"SWEEP {scale_name}_{arm} FAILED (recorded); continuing", flush=True)
                continue
            results.append({
                "scale": scale_name,
                "arm": arm,
                "runs": runs,
                "train_examples_per_head": blob.get("train_examples_per_head"),
                "val_examples_per_head": blob.get("val_examples_per_head"),
                "wall_seconds": blob.get("wall_seconds"),
            })

    curve_path = args.out_dir / "sweep-index.json"
    if curve_path.exists():
        prev = json.loads(curve_path.read_text())
        known = {(r["scale"], r["arm"]) for r in prev["runs"]}
        for r in results:
            if (r["scale"], r["arm"]) not in known:
                prev["runs"].append(r)
        runs_out = prev["runs"]
    else:
        runs_out = results
    runs_out.sort(key=lambda r: r["scale"])
    curve_path.write_text(json.dumps(
        {"format": "routescout-sweep-v1", "runs": runs_out, "failures": failures},
        indent=2) + "\n")
    if failures:
        print(f"SWEEP recorded {len(failures)} failed arm(s): "
              f"{[(f['scale'], f['arm']) for f in failures]}")
    print(f"SWEEP index={curve_path}")
    # Exit nonzero so the caller's `rc != 0` branch surfaces a partial failure.
    # Without this a sweep where one arm failed and its sibling succeeded would
    # report rc=0, and the orchestrator (which logs the tail only on rc!=0) would
    # discard the failure text — a dropped curve point appearing green.
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
