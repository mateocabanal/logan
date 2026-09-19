#!/usr/bin/env python3
"""Fixed Logan architecture scorer for Dream-RSI candidate workspaces.

The candidate cannot edit this file: it lives outside task/seed. The score is
only valid when the complete workspace test suite remains at least as strong as
the measured seed. Architecture points reward observable shared-runtime
surface area; they are deliberately secondary to the correctness gate.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

BASELINE_PASSED = 477
BASELINE_IGNORED = 3
TEST_TIMEOUT_SECONDS = 1800


def run_tests(root: Path) -> tuple[int, int, int, str, float, str | None]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            ["cargo", "test", "--workspace", "--all-targets"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=TEST_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "")
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return 0, 0, 0, output, time.monotonic() - started, "timeout"
    output = proc.stdout
    passed = failed = ignored = 0
    for match in re.finditer(
        r"test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;",
        output,
    ):
        passed += int(match.group(1))
        failed += int(match.group(2))
        ignored += int(match.group(3))
    if proc.returncode != 0 and failed == 0:
        failed = 1
    return passed, failed, ignored, output, time.monotonic() - started, None


def source_text(root: Path) -> str:
    chunks: list[str] = []
    for path in root.rglob("*.rs"):
        if any(part in {"target", ".git", ".dream-rsi"} for part in path.parts):
            continue
        try:
            chunks.append(path.read_text(errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def architecture_points(root: Path) -> tuple[int, list[str]]:
    text = source_text(root)
    checks = [
        ("generic causal-state surface", ("CausalState", "StateRegionKind")),
        ("all persistent region kinds", ("AppendOnly", "Ring", "MutableFixed", "SparsePaged", "Opaque")),
        ("transaction generations and COW", ("CowPage", "checkpoint", "rollback", "generation")),
        ("shared prefix runtime", ("pub struct PrefixCache {", "longest_prefix")),
        ("versioned snapshot integrity", ("SnapshotContainer", "checksum", "atomic")),
        ("engine codec boundary", ("CausalStateCodec", "export_state", "import_state")),
        ("dense engine adoption", ("EngineSession", "DenseSession")),
        ("Qwen engine adoption", ("EngineSession", "Qwen", "prefix")),
        ("model-neutral resources", ("ResourceKey", "ResidencyManager")),
        ("generic speculative lifecycle", ("SpeculativeExecutor", "accepted", "rollback", "commit")),
    ]
    earned: list[str] = []
    points = 0
    for label, needles in checks:
        if all(needle.lower() in text.lower() for needle in needles):
            points += 10
            earned.append(label)
    return points, earned


def main() -> int:
    root = Path.cwd()
    started = time.monotonic()
    passed, failed, ignored, output, test_seconds, timeout = run_tests(root)
    arch_score, earned = architecture_points(root)
    valid = (
        timeout is None
        and failed == 0
        and passed >= BASELINE_PASSED
        and ignored <= BASELINE_IGNORED
    )
    # Correctness dominates. Architecture points are intentionally bounded so
    # a candidate cannot win by deleting tests or weakening the runtime gate.
    score = (1000.0 if valid else 0.0) + float(arch_score)
    fail_class = "ok" if valid else (timeout or "correctness")
    result = {
        "score": score,
        "valid": valid,
        "fail_class": fail_class,
        "error": None if valid else output[-4000:],
        "passed": passed,
        "failed": failed,
        "ignored": ignored,
        "baseline_passed": BASELINE_PASSED,
        "architecture_points": arch_score,
        "architecture_checks": earned,
        "test_seconds": round(test_seconds, 3),
        "scored_seconds": round(time.monotonic() - started, 3),
    }
    eval_dir = root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
