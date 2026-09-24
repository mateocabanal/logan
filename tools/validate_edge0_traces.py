#!/usr/bin/env python3
"""Fail-closed validator for edge0-logan-trace-v1 datasets of any route K."""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MAGIC = b"E0TRC001"
HEADER_BYTES = 32
HIDDEN = 2048
EXPERTS = 256
OWNER_FIRST = 6
OWNER_LAST = 37


def expected_record_bytes(hidden: int, k: int) -> int:
    return 16 + hidden * 2 + k * 2 * 4


def read_header(path: Path) -> tuple[int, int, int]:
    raw = path.read_bytes()[:HEADER_BYTES]
    if len(raw) != HEADER_BYTES:
        raise ValueError(f"{path}: truncated header")
    magic, version, owner, hidden, experts, k, header_bytes, record_bytes, reserved = struct.unpack(
        "<8sIHHHHIII", raw
    )
    expected = expected_record_bytes(hidden, k)
    if magic != MAGIC or version != 1:
        raise ValueError(f"{path}: bad magic/version {magic!r}/{version}")
    if hidden != HIDDEN or experts != EXPERTS or header_bytes != HEADER_BYTES or record_bytes != expected or reserved != 0:
        raise ValueError(
            f"{path}: bad geometry owner={owner} hidden={hidden} experts={experts} k={k} "
            f"header={header_bytes} record={record_bytes} expected={expected} reserved={reserved}"
        )
    if not (1 <= k <= experts):
        raise ValueError(f"{path}: invalid K={k}")
    return int(owner), int(k), int(record_bytes)


def validate_file(
    path: Path,
    known_run_ids: set[int],
    expected_counts: dict[int, int] | None = None,
    max_run_prefix: int = 0,
) -> dict:
    owner, k, record_bytes = read_header(path)
    size = path.stat().st_size
    payload = size - HEADER_BYTES
    n_full = payload // record_bytes if max_run_prefix > 0 else None
    if max_run_prefix == 0:
        if payload < 0 or payload % record_bytes:
            raise ValueError(
                f"{path}: partial record payload={payload} mod={record_bytes}"
            )
    # A still-collecting corpus is expected to end in a partial record; that is
    # ignored here and only the whole-run prefix is validated.
    n = n_full if n_full is not None else payload // record_bytes
    if n == 0:
        raise ValueError(f"{path}: empty dataset")

    # Runs are contiguous in an append-only trace, so a nested scale point is a
    # byte prefix. The truncation is computed from the run-id column *before* any
    # interleaved field block is sliced, so every block is cut to the same length.
    mm = np.memmap(path, mode="r", dtype=np.uint8, offset=HEADER_BYTES, shape=(n, record_bytes))
    all_run_ids = np.ascontiguousarray(mm[:, 0:8]).view("<u8").reshape(-1)
    if max_run_prefix > 0:
        boundaries: list[int] = []
        start = 0
        for pos in range(1, n + 1):
            if pos == n or all_run_ids[pos] != all_run_ids[start]:
                boundaries.append(pos)
                start = pos
        if len(boundaries) > max_run_prefix:
            n = boundaries[max_run_prefix - 1]
    if n == 0:
        raise ValueError(f"{path}: no complete run within the requested prefix")

    run_ids = all_run_ids[:n]
    generations = np.ascontiguousarray(mm[:n, 8:16]).view("<u8").reshape(-1)

    off = 16 + HIDDEN * 2
    rb = k * 2
    current = np.ascontiguousarray(mm[:n, off : off + rb]).view("<u2").reshape(n, k)
    off += rb
    previous = np.ascontiguousarray(mm[:n, off : off + rb]).view("<u2").reshape(n, k)
    off += rb
    target = np.ascontiguousarray(mm[:n, off : off + rb]).view("<u2").reshape(n, k)
    off += rb
    weights = np.ascontiguousarray(mm[:n, off : off + rb]).view("<f2").reshape(n, k).astype(np.float32)
    off += rb
    if off != record_bytes:
        raise ValueError(f"{path}: parser ended at {off}, expected {record_bytes}")

    unknown = sorted(set(int(x) for x in np.unique(run_ids)) - known_run_ids)
    if unknown:
        raise ValueError(f"{path}: {len(unknown)} unknown run_ids, first={unknown[:3]}")
    if np.any(current >= EXPERTS) or np.any(target >= EXPERTS):
        raise ValueError(f"{path}: invalid current/target expert id")
    if np.any((previous >= EXPERTS) & (previous != np.uint16(65535))):
        raise ValueError(f"{path}: invalid previous-route expert id")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(f"{path}: invalid target weights")
    sums = weights.sum(axis=1)
    if np.max(np.abs(sums - 1.0)) > 0.01:
        raise ValueError(
            f"{path}: target weights not normalized; max_err={np.max(np.abs(sums - 1))}"
        )

    by_run: dict[int, list[int]] = defaultdict(list)
    order: list[int] = []
    for rid, gen in zip(run_ids, generations):
        key = int(rid)
        if key not in by_run:
            order.append(key)
        by_run[key].append(int(gen))
    for rid, gens in by_run.items():
        if gens != list(range(1, len(gens) + 1)):
            raise ValueError(
                f"{path}: run {rid} generations not contiguous from 1; "
                f"first={gens[:6]} last={gens[-6:]}"
            )
    # Per-run record count. The generation-contiguity check above accepts a run
    # truncated in the middle (1..k is still contiguous from 1), which would
    # silently shrink a scale point, so the expected `tokens - 2` is asserted
    # when the corpus index supplies it.
    short = []
    if expected_counts:
        for rid, gens in by_run.items():
            want = expected_counts.get(rid)
            if want is not None and len(gens) != want:
                short.append((rid, len(gens), want))
    if short:
        raise ValueError(
            f"{path}: {len(short)} run(s) have the wrong record count; "
            f"first={short[:3]} (run_id, got, expected)"
        )

    return {
        "owner": owner,
        "k": k,
        "record_bytes": record_bytes,
        "records": int(n),
        "runs": len(by_run),
        "run_order": order,
        "per_run": {rid: len(gens) for rid, gens in by_run.items()},
        "weight_sum_max_error": float(np.max(np.abs(sums - 1.0))),
    }


def check_temporal_alignment(dirpath: Path, owners: list[int],
                             max_records: int | None = None) -> dict:
    """Verify the owner-N -> consumer-(N+1) temporal pairing across head files.

    This is the Phase 0 requirement that the trace format is *correct*, not merely
    well-formed. For each adjacent owner pair the target route recorded by owner N
    must equal the current route recorded by owner N+1 on the following
    generation, and the previous route of owner N must equal its own current route
    from the generation before.

    Two boundary rules are required for this to be valid, and getting either wrong
    produces a false failure:

    * **generation g+1 must exist.** Records stop at the final generated token,
      whose feature has no consumer, so `target(N, g_max)` has nothing to pair
      with. Only generations strictly below each run's maximum are checked.
    * **generation 1's previous route is a sentinel.** `begin_decode` clears the
      previous-route history, so `previous(N, 1)` is missing-expert padding and
      is compared only for generations >= 2.
    """
    frames: dict[int, dict] = {}
    for owner in owners:
        path = dirpath / f"owner-{owner:02}.e0trace"
        _owner, k, record_bytes = read_header(path)
        payload = path.stat().st_size - HEADER_BYTES
        n = payload // record_bytes
        if max_records is not None:
            # Bound every head to the same record count. A still-appending corpus
            # can have one head a record or two ahead of another, and comparing a
            # head against a *shorter* neighbour produces spurious
            # "no consumer record" mismatches. The bound is the caller's
            # `--prefix-runs`-derived count, so both checks see identical evidence.
            n = min(n, max_records)
        mm = np.memmap(path, mode="r", dtype=np.uint8, offset=HEADER_BYTES,
                       shape=(n, record_bytes))
        off = 16 + HIDDEN * 2
        rb = k * 2
        def block(at: int) -> np.ndarray:
            return np.ascontiguousarray(mm[:, at : at + rb]).view("<u2").reshape(n, k)
        frames[owner] = {
            "run": np.ascontiguousarray(mm[:, 0:8]).view("<u8").reshape(-1),
            "gen": np.ascontiguousarray(mm[:, 8:16]).view("<u8").reshape(-1),
            "k": k,
            "current": block(off),
            "previous": block(off + rb),
            "target": block(off + 2 * rb),
        }

    checked = 0
    mismatches: list[str] = []
    for owner in owners:
        if owner + 1 not in frames:
            continue
        a, b = frames[owner], frames[owner + 1]
        # Index consumer (owner+1) records by (run, gen) for lookup.
        b_index: dict[tuple[int, int], int] = {}
        for pos in range(b["gen"].shape[0]):
            b_index[(int(b["run"][pos]), int(b["gen"][pos]))] = pos
        # Per-run maximum generation, to skip the unpaired final record.
        max_gen: dict[int, int] = {}
        for pos in range(a["gen"].shape[0]):
            rid, gen = int(a["run"][pos]), int(a["gen"][pos])
            if gen > max_gen.get(rid, 0):
                max_gen[rid] = gen
        a_index: dict[tuple[int, int], int] = {
            (int(a["run"][p]), int(a["gen"][p])): p for p in range(a["gen"].shape[0])
        }
        for pos in range(a["gen"].shape[0]):
            rid, gen = int(a["run"][pos]), int(a["gen"][pos])
            if gen >= max_gen[rid]:
                continue  # unpaired final record: no consumer at gen+1
            nxt = b_index.get((rid, gen + 1))
            if nxt is None:
                mismatches.append(f"owner{owner}->{owner+1} run {rid} gen {gen+1}: no consumer record")
                continue
            if not np.array_equal(a["target"][pos], b["current"][nxt]):
                mismatches.append(
                    f"owner{owner}->{owner+1} run {rid} gen {gen}: "
                    f"target {a['target'][pos].tolist()} != consumer current {b['current'][nxt].tolist()}"
                )
            if gen >= 2:
                prev = a_index.get((rid, gen - 1))
                if prev is not None and not np.array_equal(a["previous"][pos], a["current"][prev]):
                    mismatches.append(
                        f"owner{owner} run {rid} gen {gen}: previous != own current at gen {gen-1}"
                    )
            checked += 1
            if len(mismatches) >= 5:
                break
        if len(mismatches) >= 5:
            break
    return {"checked": checked, "mismatches": mismatches}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir", type=Path)
    ap.add_argument("--prefix-runs", type=int, default=0,
                    help="validate only the first N runs of a still-collecting corpus")
    ap.add_argument("--index", type=Path,
                    help="corpus-index.json supplying expected per-run record counts")
    ap.add_argument("--check-alignment", action="store_true",
                    help="verify the owner->consumer temporal pairing across heads")
    args = ap.parse_args()

    manifests = list(args.trace_dir.glob("run-*.json"))
    index_obj = None
    if args.index is not None:
        index_obj = json.loads(args.index.read_text())
    elif (args.trace_dir / "corpus-index.json").exists():
        index_obj = json.loads((args.trace_dir / "corpus-index.json").read_text())

    known: set[int] = set()
    manifest_ks: set[int] = set()
    for p in manifests:
        obj = json.loads(p.read_text())
        known.add(int(obj["run_id"]))
        manifest_ks.add(int(obj["k"]))
    expected_counts: dict[int, int] = {}
    if index_obj:
        for entry in index_obj.get("runs", []):
            if entry.get("run_id") is not None and entry.get("tokens"):
                expected_counts[int(entry["run_id"])] = int(entry["tokens"]) - 2
        known |= set(expected_counts)
    if not known:
        raise SystemExit("no run manifests and no corpus index")

    reports = []
    for owner in range(OWNER_FIRST, OWNER_LAST + 1):
        path = args.trace_dir / f"owner-{owner:02}.e0trace"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        report = validate_file(path, known, expected_counts, args.prefix_runs)
        if report["owner"] != owner:
            raise SystemExit(f"{path}: header owner={report['owner']}")
        reports.append(report)

    record_counts = {r["records"] for r in reports}
    run_counts = {r["runs"] for r in reports}
    ks = {r["k"] for r in reports}
    rec_bytes = {r["record_bytes"] for r in reports}
    per_runs = [r["per_run"] for r in reports]
    if len(record_counts) != 1 or len(run_counts) != 1 or len(ks) != 1 or len(rec_bytes) != 1:
        raise SystemExit("head files disagree on record/run/K/layout")
    if any(x != per_runs[0] for x in per_runs[1:]):
        raise SystemExit("head files disagree on per-run record counts")
    orders = [r["run_order"] for r in reports]
    if any(x != orders[0] for x in orders[1:]):
        raise SystemExit("head files disagree on run order")
    if manifest_ks and ks != manifest_ks:
        raise SystemExit(f"manifest K {manifest_ks} disagrees with trace K {ks}")

    first = reports[0]
    print(f"VALID format=edge0-logan-trace-v1 k={first['k']} record_bytes={first['record_bytes']} heads={len(reports)}")
    print(f"VALID manifests={len(manifests)} completed_runs={first['runs']} records_per_head={first['records']}")
    print(f"VALID examples_total={first['records'] * len(reports)}")
    print(f"VALID per_run_counts={list(first['per_run'].values())}")
    print(f"VALID weight_sum_max_error={max(r['weight_sum_max_error'] for r in reports):.6f}")
    if expected_counts:
        print(f"VALID per_run_counts_asserted_against_index={len(expected_counts)} runs")

    if args.check_alignment:
        owners = list(range(OWNER_FIRST, OWNER_LAST + 1))
        # Same record bound the format check used, so a partially-appended corpus
        # cannot make the alignment check read a head's truncated tail and report
        # "no consumer record" as if the corpus were corrupt. `min` rather than the
        # first head's count: the bound must be the length all heads share.
        record_bound = min(r["records"] for r in reports)
        aligned = check_temporal_alignment(
            args.trace_dir, owners, max_records=record_bound
        )
        if aligned["mismatches"]:
            for m in aligned["mismatches"]:
                print(f"ALIGN-FAIL {m}")
            raise SystemExit(
                f"temporal alignment failed: {len(aligned['mismatches'])} mismatch(es)"
            )
        print(f"VALID temporal_alignment_pairs={aligned['checked']} status=ok")


if __name__ == "__main__":
    main()
