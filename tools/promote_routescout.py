#!/usr/bin/env python3
"""Promote a validated RouteScout checkpoint to the model directory.

The handoff is explicit that a new checkpoint must not overwrite the current
known-good one until it has been validated on held-out data. This tool enforces
that as a gate rather than a convention:

1. the candidate's held-out metrics must be present in the comparison JSON;
2. it must beat **both** baselines (published Edge0 head and the previous
   RouteScout head) on the selection rule's primary metric, weighted mass@4;
3. the destination is a versioned path and is never an existing file;
4. the adapter is checked to carry the tensor names and shapes the runtime loader
   requires, so a malformed export cannot be promoted;
5. SHA-256 of both the source and the promoted copy is recorded, alongside the
   corpus, scale, and initialization that produced it.

Refusing to promote is a normal outcome, not an error: a candidate that does not
beat the baselines should stay in `.perf_runs` as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from safetensors import safe_open

# The runtime loader (`logan-qwen4/src/edge0_router.rs`) requires these exact
# names for owners 6..38 and checks these exact shapes.
OWNER_FIRST, OWNER_LAST = 6, 38
FEATURES, HEAD_HIDDEN, EXPERTS = 2560, 512, 256


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_adapter(path: Path) -> dict:
    """Verify the tensor surface the runtime loader expects."""
    with safe_open(str(path), framework="numpy") as f:
        keys = set(f.keys())
        missing = []
        for owner in range(OWNER_FIRST, OWNER_LAST + 1):
            for part, shape in (
                ("fc1.weight", (HEAD_HIDDEN, FEATURES)),
                ("fc2.weight", (EXPERTS, HEAD_HIDDEN)),
                ("linear_init.weight", (EXPERTS, FEATURES)),
            ):
                name = f"layers.{owner}.{part}"
                if name not in keys:
                    missing.append(name)
                    continue
                got = tuple(f.get_slice(name).get_shape())
                if got != shape:
                    raise SystemExit(f"{path}: {name} shape {got} != {shape}")
        if missing:
            raise SystemExit(
                f"{path}: missing {len(missing)} tensor(s) the runtime needs; "
                f"first={missing[:3]}"
            )
        meta = f.metadata() or {}
    return {"tensors": len(keys), "metadata": meta}


def metric_of(comparison: dict, adapter: str, key: str) -> float | None:
    blob = comparison.get("adapters", {}).get(adapter)
    if not blob:
        return None
    value = blob["mean"].get(key)
    return None if value is None else float(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--candidate-name", required=True,
                    help="adapter name as it appears in the comparison JSON")
    ap.add_argument("--comparison", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, default=Path.home() / "models")
    ap.add_argument("--baseline", action="append", default=["edge0_published", "routescout_v1"])
    ap.add_argument("--metric", default="weighted_mass4")
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--metrics", type=Path, action="append", default=[],
                    help="training metrics JSON to preserve alongside the checkpoint")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    comparison = json.loads(args.comparison.read_text())
    cand = metric_of(comparison, args.candidate_name, args.metric)
    if cand is None:
        raise SystemExit(
            f"{args.candidate_name!r} has no {args.metric} in {args.comparison}; "
            f"available adapters: {sorted(comparison.get('adapters', {}))}"
        )

    verdicts = {}
    missing_baselines = []
    for baseline in args.baseline:
        base = metric_of(comparison, baseline, args.metric)
        verdicts[baseline] = base
        if base is None:
            missing_baselines.append(baseline)

    print(f"candidate {args.candidate_name}: {args.metric}={cand:.4f}")
    for baseline, base in verdicts.items():
        if base is None:
            print(f"  baseline {baseline}: ABSENT from comparison")
        else:
            mark = "BEATS" if cand > base else "does not beat"
            print(f"  baseline {baseline}: {base:.4f} -> {mark} ({cand - base:+.4f})")

    if missing_baselines:
        # Fail closed. A gate that silently compares against fewer baselines than
        # it claims is worse than no gate: it would report a pass while having
        # skipped the very comparison the promotion rule exists to enforce.
        raise SystemExit(
            f"REFUSED: baseline(s) {missing_baselines} are absent from "
            f"{args.comparison}. Available adapters: "
            f"{sorted(comparison.get('adapters', {}))}. Re-run the evaluation so "
            f"the candidate and every baseline are scored on the same records."
        )

    beaten = all(cand > base for base in verdicts.values() if base is not None)
    if not beaten:
        print("\nREFUSED: candidate does not beat every baseline on the primary metric.")
        print("The checkpoint stays in .perf_runs as evidence; nothing was promoted.")
        raise SystemExit(2)

    surface = check_adapter(args.candidate)
    if surface["metadata"].get("target_k") not in (None, "4"):
        raise SystemExit(
            f"{args.candidate}: target_k={surface['metadata'].get('target_k')} is not K4"
        )
    if surface["metadata"].get("init") == "random":
        print("note: candidate was trained from random initialization")

    dest = args.models_dir / f"routescout_qwen36_k4_v{args.version}.safetensors"
    if dest.exists():
        raise SystemExit(f"{dest} already exists; refusing to overwrite a promoted model")

    manifest = {
        "source": str(args.candidate),
        "destination": str(dest),
        "sha256_source": sha256(args.candidate),
        "promoted_metric": args.metric,
        "candidate_value": cand,
        "baselines": verdicts,
        "adapter_tensors": surface["tensors"],
        "adapter_metadata": surface["metadata"],
        "comparison": str(args.comparison),
        "comparison_split": comparison.get("split"),
        "val_records_per_head": comparison.get("val_records_per_head"),
    }
    if args.dry_run:
        print("\nDRY RUN: would promote")
        print(json.dumps(manifest, indent=2))
        return

    args.models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.candidate, dest)
    manifest["sha256_destination"] = sha256(dest)
    if manifest["sha256_source"] != manifest["sha256_destination"]:
        raise SystemExit("copy hash mismatch; promoting nothing")

    for extra in args.metrics:
        if extra.exists():
            shutil.copy2(extra, dest.with_suffix(".metrics.json"))
            manifest["metrics_preserved"] = str(dest.with_suffix(".metrics.json"))

    dest.with_suffix(".provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nPROMOTED {dest}")
    print(f"  sha256 {manifest['sha256_destination']}")
    print(f"  provenance {dest.with_suffix('.provenance.json')}")


if __name__ == "__main__":
    main()
