#!/usr/bin/env python3
"""Snapshot a whole-run prefix of a live trace corpus.

The collectors append records continuously, so a corpus directory that is
*still being written* ends in a partial record and `load_trace` correctly
refuses it. That is the right behaviour for training, but it means a scaling
sweep can never start before collection finishes.

This produces a stable, readable snapshot of the first N runs of a bank, so
scale points can be trained while collection continues.

Two properties matter:

1. **Whole runs only.** The byte length is computed from the runs actually
   present in `corpus-index.json` and their recorded token counts, never from
   the current file size, so a partially written trailing record cannot enter
   the snapshot.
2. **Near-zero cost on APFS.** Files are cloned with `cp -c` (copy-on-write)
   and then truncated to the whole-run boundary. The shared prefix keeps sharing
   blocks with the original as it grows, so a 5k snapshot does not consume 5k
   worth of new disk.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_edge0_router import HEADER_BYTES, expected_record_bytes, load_trace  # noqa: E402

OWNER_FIRST, OWNER_LAST = 6, 37


def run_prefix(trace_dir: Path, bank: str, max_runs: int) -> list[dict]:
    index_path = trace_dir / "corpus-index.json"
    if not index_path.exists():
        raise SystemExit(f"{trace_dir}: no corpus-index.json")
    index = json.loads(index_path.read_text())
    runs = [r for r in index["runs"] if r.get("bank") == bank and r.get("run_id")]
    if max_runs > 0:
        runs = runs[:max_runs]
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bank", default="train")
    ap.add_argument("--max-runs", type=int, required=True)
    ap.add_argument("--extra-banks", default="val",
                    help="additional banks included so validation runs are present")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out.exists():
        if not args.force:
            raise SystemExit(f"{args.out} exists (use --force)")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    wanted = run_prefix(args.corpus, args.bank, args.max_runs)
    if not wanted:
        raise SystemExit(f"no runs of bank {args.bank!r} in {args.corpus}")
    included = list(wanted)
    for extra in filter(None, (b.strip() for b in args.extra_banks.split(","))):
        included += run_prefix(args.corpus, extra, 0)

    wanted_ids = {int(r["run_id"]) for r in included}
    tokens_by_run = {int(r["run_id"]): int(r["tokens"]) for r in included}
    print(f"SNAPSHOT pool_runs={len(wanted)} extra={len(included) - len(wanted)} total_runs={len(included)}")

    # Every included run contributes (tokens - 2) records per head: one record
    # per decode forward that has a successor token.
    records_per_run = {}
    for run_id, tokens in tokens_by_run.items():
        n = tokens - 2
        if n <= 0:
            raise SystemExit(f"run {run_id}: tokens={tokens} yields no records")
        records_per_run[run_id] = n

    for owner in range(OWNER_FIRST, OWNER_LAST + 1):
        src = args.corpus / f"owner-{owner:02}.e0trace"
        if not src.exists():
            raise SystemExit(f"missing {src}")
        dst = args.out / src.name
        # Clone (CoW) then truncate to the whole-run boundary. A plain copy would
        # cost the full prefix size per scale point.
        subprocess.run(["cp", "-c", str(src), str(dst)], check=True)
        trace = load_trace_unsafe(owner, src)
        # The snapshot is a *byte prefix*, so the wanted runs must occupy a
        # contiguous range starting at record 0. Find the last record belonging
        # to a wanted run, then prove nothing before it is unwanted — otherwise
        # truncating there would silently admit a run that was never selected.
        keep = 0
        for pos in range(trace["n"]):
            if int(trace["run_ids"][pos]) in wanted_ids:
                keep = pos + 1
        admitted = set(int(x) for x in trace["run_ids"][:keep])
        if admitted != wanted_ids:
            stray = sorted(admitted - wanted_ids)[:3]
            missing = sorted(wanted_ids - admitted)[:3]
            raise SystemExit(
                f"owner-{owner:02}: wanted runs are not a contiguous prefix of "
                f"{src}; stray={stray} missing={missing}. Collect the selection "
                f"banks before the pool bank so snapshots are well defined."
            )
        expected = sum(records_per_run[r] for r in wanted_ids)
        if keep != expected:
            raise SystemExit(
                f"owner-{owner:02}: prefix record count {keep} != expected {expected}"
            )
        size = HEADER_BYTES + keep * expected_record_bytes(2048, trace["k"])
        subprocess.run(["truncate", "-s", str(size), str(dst)], check=True)

    # Manifests and a matching index, so select_run_sets sees exactly the
    # snapshot's runs rather than every run in the source directory.
    for run_id in wanted_ids:
        (args.out / f"run-{run_id}.json").write_text(
            (args.corpus / f"run-{run_id}.json").read_text()
        )
    out_index = {
        "format": "logan-routescout-corpus-v1",
        "dir": str(args.out),
        "snapshot_of": str(args.corpus),
        "pool_bank": args.bank,
        "max_runs": args.max_runs,
        "runs": included,
    }
    (args.out / "corpus-index.json").write_text(json.dumps(out_index, indent=2) + "\n")
    print(f"SNAPSHOT out={args.out} runs={len(wanted_ids)}")


def load_trace_unsafe(owner: int, path: Path) -> dict:
    """Read run ids from a possibly-incomplete file (header + complete records)."""
    import numpy as np
    import struct

    raw = path.read_bytes()[:HEADER_BYTES]
    _magic, _version, _owner, hidden, _experts, k, _hb, record_bytes, _res = struct.unpack(
        "<8sIHHHHIII", raw
    )
    size = path.stat().st_size
    n = (size - HEADER_BYTES) // record_bytes  # floor: ignore a partial tail
    mm = np.memmap(path, mode="r", dtype=np.uint8, offset=HEADER_BYTES,
                   shape=(n, record_bytes))
    run_ids = np.ascontiguousarray(mm[:, 0:8]).view("<u8").reshape(-1)
    return {"n": int(n), "run_ids": run_ids, "k": int(k), "hidden": int(hidden)}


if __name__ == "__main__":
    main()
