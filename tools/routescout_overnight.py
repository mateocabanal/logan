#!/usr/bin/env python3
"""Overnight orchestrator for EXP-078: wait for the corpus, then run the sweep.

Designed to run unattended. It:

1. waits until the pool bank has reached a target number of runs (polling the
   trace files, not the index, so a half-written run never counts);
2. trains the scaling sweep over the requested scale points and both/additional
   initialization arms, using `--live-prefix` so a still-collecting corpus is
   usable;
3. re-runs the sweep once collection finishes, so the final scale point uses the
   complete corpus;
4. evaluates every arm plus both baselines on the frozen test bank;
5. measures latency and writes the per-layer analysis;
6. reduces everything into the learning-curve/comparison JSON.

Every stage is idempotent: a stage whose metrics file already exists is skipped,
so a restarted orchestrator resumes rather than redoing work. Failures are
recorded and the remaining independent stages still run — a broken latency
measurement must not lose a completed sweep.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = Path.home() / ".venvs/logan-edge0-train/bin/python"


def load_summary(path: Path) -> dict | None:
    """Read a JSON artifact, or None if absent/unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def log(msg: str, log_path: Path | None = None) -> None:
    line = f"[orch {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def run(cmd: list[str], log_path: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout or ""
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(" ".join(cmd) + "\n" + out + "\n")
    return proc.returncode, out


def pool_runs_complete(corpus: Path) -> int:
    """Pool runs whose records are fully present in the trace files."""
    sys.path.insert(0, str(HERE))
    from run_routescout_sweep import available_pool_runs

    try:
        return available_pool_runs(corpus)
    except Exception:
        return 0


def wait_for_pool(corpus: Path, target_runs: int, log_path: Path, poll_s: int, deadline_s: float,
                  stall_polls: int = 3) -> int:
    """Wait for `target_runs` complete pool runs, or for a dead collector.

    Waiting only for the target or a timeout is not enough: nothing supervises the
    collector, so a process-level death (OOM or swap on this documented 16 GB /
    20 GB-model host) looks exactly like slow collection. The orchestrator would
    poll until the multi-hour deadline and only then consider a top-up, by which
    point there is no time left to train the largest scale — losing the headline
    answer to a crash that was detectable within minutes.

    So a third exit condition is added: **no live collector AND no progress across
    several polls**. Both halves are required. No-progress alone would fire during
    a slow run; no-live-collector alone would fire in the gap between the
    collector's processes. Together they mean the collection has actually stopped
    rather than merely pausing. Returning here hands control to the top-up path,
    which has its own idle guard, so this cannot double-write the traces.
    """
    t0 = time.time()
    last_have = -1
    stalled = 0
    while True:
        have = pool_runs_complete(corpus)
        if have >= target_runs:
            log(f"pool ready: {have}/{target_runs} runs", log_path)
            return have
        if have == last_have:
            stalled += 1
        else:
            stalled = 0
        if stalled >= stall_polls and not live_collectors():
            log(f"pool STALLED: {have}/{target_runs} runs, no live collector for "
                f"{stalled} polls; handing off to the top-up path", log_path)
            return have
        if time.time() - t0 > deadline_s:
            log(f"pool wait timeout: {have}/{target_runs} runs; proceeding with what exists", log_path)
            return have
        if stalled:
            log(f"pool waiting: {have}/{target_runs} runs "
                f"(no progress for {stalled} poll(s))", log_path)
        else:
            log(f"pool waiting: {have}/{target_runs} runs", log_path)
        last_have = have
        time.sleep(poll_s)


def live_trainers() -> list[str]:
    """Any training process currently writing an adapter."""
    out = subprocess.run(["ps", "-eo", "pid,command"], text=True,
                         stdout=subprocess.PIPE).stdout
    return [
        line.strip() for line in out.splitlines()[1:]
        if "train_edge0_router" in line or "run_routescout_sweep" in line
    ]


def wait_for_trainers_idle(log_path, poll_s: int, deadline_s: float) -> bool:
    """Wait until no training process is running.

    `run()` is a blocking `subprocess.run`, so killing the orchestrator leaves the
    trainer it spawned alive and still writing its adapter. A restarted
    orchestrator would otherwise start a second writer on the same path.
    """
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        busy = live_trainers()
        if not busy:
            log("orphan guard: no live trainer", log_path)
            return True
        log(f"orphan guard: {len(busy)} live trainer(s) alive; waiting", log_path)
        time.sleep(poll_s)
    log("orphan guard: trainer never exited; refusing to duplicate it", log_path)
    return False


def live_collectors() -> list[str]:
    """Any process currently appending to a trace corpus."""
    out = subprocess.run(["ps", "-eo", "pid,command"], text=True,
                         stdout=subprocess.PIPE).stdout
    return [
        line.strip() for line in out.splitlines()[1:]
        if "collect_routescout_corpus" in line or "decode_bench" in line
    ]


def wait_for_writer_idle(corpus: Path, log_path: Path, poll_s: int, deadline_s: float) -> bool:
    """Wait until nothing is appending to the corpus, then confirm it is stable.

    Used by two callers with the same requirement:

    * **Top-up** — two writers on the same append-only `owner-*.e0trace` files
      would interleave records and shift every subsequent record boundary, so a
      top-up must never start while the primary collector is alive.
    * **Timing stages** — the MLX-latency and deployed measurements both refuse
      under contention, so waiting here keeps the predictor-cost deliverable from
      being skipped on a technicality.

    Liveness alone is not enough: the last poll of a dying collector can race its
    final flush, so the trace size is also required to stop changing across two
    consecutive polls before the corpus is treated as quiescent.
    """
    t0 = time.time()
    probe = corpus / "owner-06.e0trace"
    last_size = None
    stable = 0
    while time.time() - t0 < deadline_s:
        busy = live_collectors()
        size = probe.stat().st_size if probe.exists() else 0
        if not busy and size == last_size:
            stable += 1
            if stable >= 2:
                log("writer-idle guard: writer idle and trace size stable", log_path)
                return True
        else:
            stable = 0
            if busy:
                log(f"writer-idle guard: {len(busy)} writer(s) alive; waiting", log_path)
        last_size = size
        time.sleep(poll_s)
    log("writer-idle guard: writer never went idle within budget", log_path)
    return False


def top_up_pool(corpus: Path, needed: int, model: Path, binary: Path, log_path: Path,
                poll_s: int = 30) -> int:
    """Try to reach `needed` complete pool runs with a second collector pass.

    The pool bank has 200 prompts and a 196-run target, so only 4 failed runs can
    be absorbed by a single pass — and a scale point that silently disappears is
    the one outcome that would cost this experiment its headline.

    A second pass is the recovery, and it works because the collector skips by
    `(prompt_index, pass)` identity: pass 0 re-attempts exactly the prompts that
    failed (they never got an index entry), and if that is still not enough, pass 1
    contributes fresh runs of the same prompts under new seeds. Re-running with
    `--passes 1` would do nothing at all — every pair is already recorded.

    **The primary writer must be confirmed dead first.** `wait_for_pool` returns on
    the deadline regardless of count, so a timed-out wait can leave the collector
    still appending; launching a second one there would interleave two writers on
    the same files.
    """
    have = pool_runs_complete(corpus)
    if have >= needed:
        return have
    if not wait_for_writer_idle(corpus, log_path, poll_s, 1800):
        return pool_runs_complete(corpus)
    # Re-read after the wait: the collector may have finished the last few runs
    # while the guard was confirming it was idle.
    have = pool_runs_complete(corpus)
    if have >= needed:
        log(f"top-up not needed after all: {have}/{needed} runs", log_path)
        return have
    log(f"top-up: pool has {have}/{needed} runs; starting a second collector pass", log_path)
    cmd = [
        str(PYTHON), str(HERE / "collect_routescout_corpus.py"),
        "--model", str(model), "--trace-dir", str(corpus),
        "--bank", "train", "--tokens", "256",
        "--target-tokens", str(needed * 256),
        "--resume", "--passes", "2",
        "--binary", str(binary),
    ]
    rc, out = run(cmd, log_path)
    log(f"top-up rc={rc} tail:\n{out[-800:]}", log_path)
    return pool_runs_complete(corpus)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="experiment root, e.g. .perf_runs/routescout-train-v1")
    ap.add_argument("--scales", required=True, help="scale=run_count,... in ascending order")
    ap.add_argument("--arms", default="edge0,rs_v1")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--deadline-hours", type=float, default=8.0)
    ap.add_argument("--random-scale", default="",
                    help="scale name to additionally train with Arm C (random init)")
    ap.add_argument("--model", type=Path,
                    default=Path.home() / "models/Qwen3.6-35B-A3B-MLX-oQ4-FP16")
    ap.add_argument("--binary", type=Path,
                    default=Path("target/release/examples/decode_bench"))
    ap.add_argument("--top-up", action="store_true",
                    help="run a second collector pass if the pool is short of the top scale")
    args = ap.parse_args()

    root: Path = args.root
    corpus = root / "corpus"
    test_dir = root / "final"
    runs_dir = root / "runs"
    results = root / "results"
    runs_dir.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    orch_log = root / "orchestrator.log"
    deadline = args.deadline_hours * 3600
    t_start = time.time()

    scales = []
    for item in args.scales.split(","):
        name, runs = item.split("=", 1)
        scales.append((name.strip(), int(runs)))
    scales.sort(key=lambda s: s[1])

    log("EXP-078 orchestrator starting", orch_log)
    log(f"root={root} scales={scales} arms={args.arms}", orch_log)
    # Record the complete invocation. This process may be restarted many times
    # during an overnight run, and the flags that matter are the ones that are easy
    # to drop by hand: without `--top-up` the stall/death detection returns but the
    # scale point is merely skipped, and without `--random-scale` the Arm C point
    # silently disappears. A log line makes a restart auditable rather than assumed.
    log(f"argv: {' '.join(sys.argv)}", orch_log)
    log(
        f"flags: epochs={args.epochs} poll_seconds={args.poll_seconds} "
        f"deadline_hours={args.deadline_hours} top_up={args.top_up} "
        f"random_scale={args.random_scale or 'none'} scales={args.scales}",
        orch_log,
    )

    def sweep(scales_spec: str, arms: str, require: bool) -> None:
        # A restarted orchestrator must not race an orphaned trainer from the
        # previous instance: `run()` is a blocking `subprocess.run`, so killing the
        # orchestrator leaves its trainer alive and still writing this adapter.
        if not wait_for_trainers_idle(orch_log, args.poll_seconds, 1800):
            log(f"sweep skipped for {scales_spec}: a trainer never exited", orch_log)
            return
        cmd = [
            str(PYTHON), str(HERE / "run_routescout_sweep.py"),
            "--corpus", str(corpus), "--out-dir", str(runs_dir),
            "--scales", scales_spec, "--arms", arms,
            "--epochs", str(args.epochs), "--live-prefix",
        ]
        if require:
            cmd.append("--require-runs")
        rc, out = run(cmd, orch_log)
        log(f"sweep rc={rc} scales={scales_spec} arms={arms}", orch_log)
        # Log failures whether or not rc is nonzero: the sweep exits 1 on a partial
        # failure, but a future caller might not, and the failure list is the thing
        # that must never be silently dropped.
        if "recorded" in out and "failed arm" in out:
            for line in out.splitlines():
                if "failed arm" in line:
                    log(f"  {line.strip()}", orch_log)
        if rc != 0:
            log(f"sweep FAILED; tail:\n{out[-1500:]}", orch_log)

    # Scale points are nested prefixes of one append-only corpus, so "the first
    # N runs" is the same bytes whenever it is trained. Each scale is therefore
    # trained exactly once, as soon as N runs are complete — there is no need to
    # retrain later, and no provisional artifacts to reconcile.
    for name, runs in scales:
        remaining = deadline - (time.time() - t_start)
        if remaining <= 0:
            log(f"deadline reached before {name}; stopping scale loop", orch_log)
            break
        have = wait_for_pool(corpus, runs, orch_log, args.poll_seconds, min(remaining, deadline))
        if have < runs:
            # Only top up once the primary collector has actually stopped. While
            # it is still running, a shortfall means "not finished yet", and a
            # second writer would corrupt the append-only traces.
            if live_collectors():
                log(f"{name}: {have}/{runs} runs and a collector is still running; "
                    f"waiting for it to finish rather than topping up", orch_log)
                have = wait_for_pool(corpus, runs, orch_log, args.poll_seconds,
                                     max(0.0, remaining - (time.time() - t_start)))
            if have < runs and args.top_up:
                have = top_up_pool(corpus, runs, args.model, args.binary, orch_log,
                                   args.poll_seconds)
        if have < runs:
            log(f"{name}: only {have}/{runs} runs within budget; skipping rather than "
                f"training a mislabeled scale point", orch_log)
            continue
        sweep(f"{name}={runs}", args.arms, require=True)
        if args.random_scale and name == args.random_scale:
            sweep(f"{name}={runs}", "random", require=True)

    # Preserve the trace validator's output. The handoff's artifact list explicitly
    # includes "trace validator output", and until now it was only ever printed to a
    # terminal — so the Phase 0 correctness evidence for the corpus would not survive
    # the session. Recorded for both banks with the alignment check, which is the
    # part that can actually fail.
    #
    # The pool may still be appending (its last record can be partial), so it is
    # validated as the first N *complete* runs, using the corpus index to supply the
    # expected per-run record counts. The test bank is quiescent and validated whole.
    for bank_dir, bank_name in ((corpus, "pool"), (test_dir, "test")):
        val = [
            str(PYTHON), str(HERE / "validate_edge0_traces.py"), str(bank_dir),
            "--index", str(bank_dir / "corpus-index.json"),
        ]
        if bank_dir is corpus:
            # A run's index entry is appended only after its child exits, so every
            # *indexed* run is complete; the sole incomplete run is the one currently
            # being collected, which sits after them in file order. Validating the
            # first len(index) runs therefore covers every complete run and stops
            # before the partial one.
            manifest = load_summary(bank_dir / "corpus-index.json") or {}
            val += ["--prefix-runs", str(len(manifest.get("runs") or []))]
        val.append("--check-alignment")
        rc, out = run(val, orch_log)
        dest = results / f"validate-{bank_name}.txt"
        dest.write_text(out if out.endswith("\n") else out + "\n")
        log(f"validator ({bank_name}) rc={rc} -> {dest}", orch_log)

    # Held-out evaluation of every produced adapter plus both baselines.
    #
    # Every adapter in runs_dir is trained on a nested prefix of the same corpus,
    # so they are all legitimate *curve* points — but only the largest scale is a
    # candidate for "the best RouteScout head". All of them are still scored,
    # because the curve is the experiment's primary output; the largest scale is
    # identified in the summary rather than by excluding the smaller arms here.
    adapters = [
        f"edge0_published={Path.home() / 'models/prerouter_edge0_35b.safetensors'}",
        f"routescout_v1={Path.home() / 'models/prerouter_logan_qwen36_v1.safetensors'}",
    ]
    produced = sorted(runs_dir.glob("routescout_*.safetensors"))
    for path in produced:
        adapters.append(f"{path.stem}={path}")
    if not produced:
        log("no trained adapters found; skipping held-out evaluation", orch_log)
    else:
        # Each heavy stage is skipped when its artifact already covers every adapter
        # we just produced. Every stage is idempotent, so a restart must not repeat
        # them: re-running the comparison would take a second look at the frozen test
        # bank, and the deployed measurement holds the whole model for minutes. A new
        # scale point adds an adapter, which changes the covered set and re-runs the
        # stage exactly once.
        want = {item.split("=", 1)[0] for item in adapters}

        def covers(stage_name: str, filename: str) -> bool:
            blob = load_summary(results / filename)
            if not blob:
                return False
            covered = set(blob.get("adapters") or blob.get("arms") or {})
            if not covered:
                return False
            if want.issubset(covered):
                log(f"{stage_name}: reusing {filename} (covers all "
                    f"{len(want)} adapters)", orch_log)
                return True
            log(f"{stage_name}: {filename} covers {len(covered)}/"
                f"{len(want)} adapters; re-running", orch_log)
            return False

        if covers("comparison", "compare-test.json"):
            pass
        else:
            cmd = [
                str(PYTHON), str(HERE / "eval_routescout.py"),
                "--trace-dir", str(test_dir), "--owners", "6-37", "--all-holdout",
                "--live-prefix", "--output", str(results / "compare-test.json"),
            ]
            for item in adapters:
                cmd += ["--adapter", item]
            rc, out = run(cmd, orch_log)
            log(f"comparison eval rc={rc}", orch_log)
            if rc != 0:
                log(f"comparison eval FAILED; tail:\n{out[-1200:]}", orch_log)

        if covers("latency (MLX)", "latency.json"):
            pass
        else:
            # Both timing stages refuse under contention, so wait for the collector
            # to exit first. Otherwise the predictor-cost deliverable would be
            # skipped on a technicality (the collector reaches 196 runs ~40 min
            # before 50k training finishes, so this normally returns immediately).
            wait_for_writer_idle(corpus, orch_log, args.poll_seconds,
                                 max(0.0, deadline - (time.time() - t_start)))
            lat = [
                str(PYTHON), str(HERE / "measure_routescout_latency.py"),
                "--trace-dir", str(test_dir), "--owners", "6-37", "--repeats", "200",
                "--output", str(results / "latency.json"),
            ]
            for item in adapters:
                lat += ["--adapter", item]
            rc, out = run(lat, orch_log)
            log(f"latency (MLX) rc={rc}", orch_log)

        # Deployed FP16/BNNS measurement. Refuses to run if anything else holds a
        # copy of the model, so this is attempted only after collection is done;
        # a refusal is recorded rather than forced through.
        if covers("deployed measurement", "deployed.json"):
            pass
        else:
            dep = [
                str(PYTHON), str(HERE / "measure_routescout_deployed.py"),
                "--output", str(results / "deployed.json"),
            ]
            for item in adapters:
                dep += ["--adapter", item]
            rc, out = run(dep, orch_log)
            log(f"deployed measurement rc={rc}\n{out[-600:]}", orch_log)

        # Runtime prediction quality across every trained head. Without this stage
        # the report would render whatever `runtime-quality.json` happened to exist
        # — a stale, contended probe over fewer heads — as the final table.
        if covers("runtime-quality", "runtime-quality.json"):
            pass
        else:
            rq = [
                str(PYTHON), str(HERE / "routescout_runtime_quality.py"),
                "--tokens", "12", "--seed", "42",
                "--output", str(results / "runtime-quality.json"),
            ]
            for item in adapters:
                rq += ["--adapter", item]
            rc, out = run(rq, orch_log)
            log(f"runtime-quality rc={rc}\n{out[-800:]}", orch_log)

    if (results / "compare-test.json").exists():
        # Per-layer analysis is a mission-required deliverable ("per-layer
        # distribution", "layers that fail to improve"), so it runs here rather
        # than being left to a manual invocation.
        layers = [
            str(PYTHON), str(HERE / "analyze_routescout_layers.py"),
            "--eval", f"test={results / 'compare-test.json'}",
            "--output", str(results / "layers.json"),
        ]
        # Scale-vs-layer deltas: consecutive scale points for the Edge0-init arm,
        # which is the arm with a checkpoint at every scale. Ordered by numeric
        # scale magnitude, not mtime: mtime could interleave 25k before 10k if a
        # file were touched, and the delta would then be computed backwards.
        def _scale_num(path) -> float:
            stem = path.stem.replace("routescout_", "")
            scale = stem.rsplit("_", 1)[0]
            try:
                return float(scale.rstrip("kK")) * 1000
            except ValueError:
                return float("inf")

        ordered = sorted(runs_dir.glob("routescout_*_edge0.metrics.json"), key=_scale_num)
        for path in ordered:
            layers += ["--curve", f"{path.stem.replace('routescout_', '')}={path}"]
        rc, out = run(layers, orch_log)
        log(f"per-layer analysis rc={rc}\n{out[-3000:]}", orch_log)

        cmd = [
            str(PYTHON), str(HERE / "summarize_routescout_sweep.py"),
            "--runs-dir", str(runs_dir), "--output", str(results / "summary.json"),
            "--compare", f"test={results / 'compare-test.json'}",
        ]
        rc, out = run(cmd, orch_log)
        log(f"summarize rc={rc}\n{out[-3000:]}", orch_log)

        # Final report, derived mechanically from the artifacts above. Run last so
        # it includes the comparison, layers, and latency sections in one pass.
        rep = [
            str(PYTHON), str(HERE / "routescout_report.py"),
            "--root", str(root), "--output", str(results / "REPORT.md"),
        ]
        rc, out = run(rep, orch_log)
        log(f"report rc={rc}", orch_log)

        # Cross-check the ledger's tables against the artifacts. This catches the
        # failure mode that reads plausibly: a hand-transcribed number that no
        # other gate would flag.
        chk = [
            str(PYTHON), str(HERE / "check_routescout_ledger.py"),
            "--runs-dir", str(runs_dir),
        ]
        rc, out = run(chk, orch_log)
        log(f"ledger cross-check rc={rc}\n{out[-800:]}", orch_log)

        # Promotion, gated on the held-out comparison. The gate itself refuses
        # unless the candidate beats every baseline on weighted mass@4, so this
        # stage can safely run unattended: a candidate that does not win stays in
        # `.perf_runs` as evidence rather than displacing the known-good model.
        best = (load_summary(results / "summary.json") or {}).get("best_checkpoint") or {}
        best_name = best.get("adapter")
        best_path = runs_dir / f"{best_name}.safetensors" if best_name else None
        if best_path is not None and best_path.exists():
            promo = [
                str(PYTHON), str(HERE / "promote_routescout.py"),
                "--candidate", str(best_path),
                "--candidate-name", str(best_name),
                "--comparison", str(results / "compare-test.json"),
                "--version", "1",
                "--metrics", str(best_path.with_suffix(".metrics.json")),
            ]
            rc, out = run(promo, orch_log)
            log(f"promotion rc={rc}\n{out[-1200:]}", orch_log)
        else:
            log(f"promotion skipped: no nominated checkpoint ({best_name})", orch_log)

    log("EXP-078 orchestrator done", orch_log)


if __name__ == "__main__":
    main()
