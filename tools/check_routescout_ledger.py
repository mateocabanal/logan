#!/usr/bin/env python3
"""Cross-check every number written in the EXP-078 ledger against its artifact.

The ledger's tables are hand-written prose over machine-produced numbers, which is
exactly where a transcription error hides: it reads plausibly and nothing fails.
This script re-derives each metric from the metrics JSON and asserts the literal
value appears in the ledger section.

Found a real one on first use: Arm C's `top1-in-K4` was written as 0.7377 while the
artifact said 0.6757 — a 6-point error in the initialization comparison that no
other check would have caught, because both values look reasonable.

Usage:

    python tools/check_routescout_ledger.py            # cross-check, exit 1 on drift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# §4's comparison table names adapters in prose, so the label->adapter mapping and the
# metric columns have to be declared. These mirror `routescout_report.py`'s METRIC_ROWS
# for the four columns §4 actually tabulates.
COMPARISON_LABELS: dict[str, str] = {
    "Edge0 head (published)": "edge0_published",
    "RouteScout v1 (previous)": "routescout_v1",
    "RouteScout 10k random (Arm C)": "routescout_10k_random",
    "RouteScout 10k (Arm A)": "routescout_10k_edge0",
    "RouteScout 10k (Arm B)": "routescout_10k_rs_v1",
    "RouteScout 25k (Arm A)": "routescout_25k_edge0",
    "RouteScout 25k (Arm B)": "routescout_25k_rs_v1",
    "RouteScout 50k (Arm A)": "routescout_50k_edge0",
    "RouteScout 50k (Arm B)": "routescout_50k_rs_v1",
}
COMPARISON_FIELDS: dict[str, str] = {
    "recall4": "recall@4",
    "weighted_mass4": "mass@4",
    "top1_in_k4": "top1-in-K4",
    "full4": "full@4",
}


def comparison_row(section: str, label: str) -> str | None:
    """The §4 row whose first cell names this adapter label.

    Scoped to §4 specifically: the string "Edge0 head (published)" also appears in
    an earlier six-column baseline table, so a section-wide search could match the
    wrong row and then pass or fail on unrelated numbers.
    """
    if "### 4. Comparison" not in section:
        return None
    start = section.index("### 4. Comparison")
    end = section.find("### 5.", start)
    block = section[start:end if end != -1 else len(section)]
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip().replace("**", "") for c in stripped.strip("|").split("|")]
        if cells and cells[0] == label:
            return stripped
    return None


def discover_checks(runs_dir: Path) -> list[tuple[str, str, str, str]]:
    """Derive the check list from the artifacts that actually exist.

    A hardcoded list silently stops covering the experiment as it grows: the 25k
    and 50k points would land with no cross-check on their ledger rows, which is
    exactly where a transcription error would survive. Discovery means every
    completed scale point is checked by construction.
    """
    checks = []
    for path in sorted(runs_dir.glob("routescout_*.metrics.json")):
        stem = path.name[len("routescout_"):-len(".metrics.json")]
        # Scale names are `5k`/`10k`/`25k`/`50k` and arms are `edge0`/`rs_v1`/`random`.
        # `rpartition("_")` breaks on `10k_rs_v1` (-> `10k_rs`/`v1`), so match the
        # scale pattern explicitly and treat the remainder as the arm.
        import re as _re
        m = _re.match(r"^(\d+k)_(.+)$", stem)
        if not m:
            continue
        scale, arm = m.group(1), m.group(2)
        if not scale or not arm:
            continue
        # Only the metrics the ledger's curve table actually tabulates. Checking
        # more would demand the ledger print columns it does not claim, which is a
        # false-alarm generator rather than a correctness check.
        for key, column in (("recall4", "recall@4"),
                            ("weighted_mass4", "mass@4"),
                            ("top1_in_target", "top1-in-K4")):
            checks.append((scale, arm, key, column))
    return checks


def row_for_scale_arm(section: str, scale: str, arm: str) -> str | None:
    """The markdown table line that names this exact scale and arm.

    Matching against the whole section is unsound: a 4-decimal value like `0.5486`
    appears six times in the EXP-078 section (measured), so a bare substring test can
    pass on some *other* row and hide real drift. Requiring the values to appear in
    the row that names this scale and arm removes that collision class, and is
    layout-agnostic — the ledger's tables have different column orders (the curve
    table carries `best`/`final` pairs), so positional comparison would be fragile.
    """
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        if cells[0] != scale:
            continue
        # The arm may be its own cell (`edge0`) or embedded in the label (`10k_edge0`).
        if cells[1] == arm or arm in stripped:
            return stripped
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, default=Path("EXPERIMENTS.md"))
    ap.add_argument("--runs-dir", type=Path,
                    default=Path(".perf_runs/routescout-train-v1/runs"))
    ap.add_argument("--section", default="## EXP-078")
    args = ap.parse_args()

    text = args.ledger.read_text()
    if args.section not in text:
        print(f"FAIL: ledger has no {args.section} section", file=sys.stderr)
        sys.exit(1)
    section = text[text.index(args.section):]

    checks = discover_checks(args.runs_dir)
    if not checks:
        print("FAIL: no metrics artifacts found to check", file=sys.stderr)
        sys.exit(1)

    by_row: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for scale, arm, key, column in checks:
        by_row.setdefault((scale, arm), []).append((key, column))

    missing, checked, skipped, no_row = [], 0, 0, []
    for (scale, arm), wanted in sorted(by_row.items()):
        path = args.runs_dir / f"routescout_{scale}_{arm}.metrics.json"
        blob = json.loads(path.read_text())
        heads = blob.get("heads") or []
        if not heads:
            skipped += len(wanted)
            continue
        row = row_for_scale_arm(section, scale, arm)
        if row is None:
            no_row.append(f"{scale}/{arm}")
            continue
        for key, column in wanted:
            if key not in heads[0]["best"]:
                skipped += 1
                continue
            value = float(np.mean([h["best"][key] for h in heads]))
            literal = f"{value:.4f}"
            checked += 1
            if literal not in row:
                missing.append(f"{scale}/{arm} {column} = {literal} "
                               f"(row: {row[:70]}…)")

    print(f"row-scoped check: {checked} value(s) across {len(by_row)} scale/arm "
          f"artifact(s) ({skipped} skipped as absent)")

    # The §4 comparison table is NOT covered by the scale/arm row check above: its rows
    # are named by adapter (e.g. "RouteScout 50k (Arm A)"), not by scale/arm, and its
    # values come from the frozen-bank comparison rather than the per-scale metrics. It
    # is also the table a reader is most likely to trust, so it gets its own check.
    compare_path = args.runs_dir.parent / "results" / "compare-test.json"
    frozen_checked, frozen_bad = 0, []
    if compare_path.exists():
        compare = json.loads(compare_path.read_text())
        for label, key in COMPARISON_LABELS.items():
            if key not in compare.get("adapters", {}):
                continue
            mean = compare["adapters"][key]["mean"]
            for field, col in COMPARISON_FIELDS.items():
                value = mean.get(field)
                if value is None and field == "top1_in_k4":
                    value = mean.get("top1_in_target")
                if value is None:
                    continue
                literal = f"{value:.4f}"
                frozen_checked += 1
                row = comparison_row(section, label)
                if row is None or literal not in row:
                    frozen_bad.append(
                        f"{label} {col} = {literal} (row: "
                        f"{(row or 'NOT FOUND')[:60]}…)"
                    )
        print(f"frozen-bank check: {frozen_checked} value(s) across "
              f"{len(COMPARISON_LABELS)} labeled adapter(s)")
    else:
        print(f"frozen-bank check: skipped ({compare_path.name} absent)")

    if frozen_bad:
        print("DRIFT: frozen-bank value(s) do not match their ledger row:")
        for m in frozen_bad:
            print(f"  - {m}")
        sys.exit(1)
    if no_row:
        print(f"note: no ledger row naming {no_row} (not yet written?)")
    if missing:
        print(f"DRIFT: {len(missing)} value(s) do not match their ledger row:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
    print("OK: every checked value matches its own ledger row")


if __name__ == "__main__":
    main()
