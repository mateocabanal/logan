#!/usr/bin/env python3
"""Train Logan's Edge0-style next-token MoE prerouter on native Qwen3.6 traces.

Trace K is read from each file header, so the same architecture can be trained
against native K4, K6, or another explicitly collected route width. The feature
width remains 2560 because current/previous routes are represented as 256-wide
sets, independent of how many experts are active.

The script fine-tunes a supplied Edge0-compatible adapter and writes a full
99-tensor adapter, preserving any untrained head (currently owner 38).
Validation is split by whole run_id whenever multiple generation runs exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

MAGIC = b"E0TRC001"
HEADER_BYTES = 32
HIDDEN = 2048
EXPERTS = 256
FEATURES = HIDDEN + 2 * EXPERTS
HEAD_HIDDEN = 512
OWNER_FIRST = 6
OWNER_LAST = 37


@dataclass
class TraceHead:
    owner: int
    k: int
    record_bytes: int
    records: np.ndarray
    """Structured view over the whole file, one element per record.

    Field access (`records["hidden"]`) is a *strided view* into the mmap, so a
    head costs address space rather than resident memory and indexed reads
    materialize only the rows actually asked for. The previous per-field
    `ascontiguousarray` copies were 4 KiB per record, i.e. ~205 MiB per head at
    a 50k-example corpus and ~6.5 GiB for all 32 heads at once.
    """

    @property
    def n(self) -> int:
        return int(self.records.shape[0])

    @property
    def run_id(self) -> np.ndarray:
        return self.records["run_id"]

    @property
    def generation(self) -> np.ndarray:
        return self.records["generation"]

    @property
    def hidden(self) -> np.ndarray:
        return self.records["hidden"]

    @property
    def current(self) -> np.ndarray:
        return self.records["current"]

    @property
    def previous(self) -> np.ndarray:
        return self.records["previous"]

    @property
    def target(self) -> np.ndarray:
        return self.records["target"]

    @property
    def target_weights(self) -> np.ndarray:
        return self.records["weights"]


def expected_record_bytes(hidden: int, k: int) -> int:
    return 16 + hidden * 2 + k * 2 * 4


def record_dtype(hidden: int, k: int) -> np.dtype:
    return np.dtype(
        [
            ("run_id", "<u8"),
            ("generation", "<u8"),
            ("hidden", "<f2", (hidden,)),
            ("current", "<u2", (k,)),
            ("previous", "<u2", (k,)),
            ("target", "<u2", (k,)),
            ("weights", "<f2", (k,)),
        ]
    )


def load_trace(path: Path) -> TraceHead:
    raw_header = path.read_bytes()[:HEADER_BYTES]
    if len(raw_header) != HEADER_BYTES:
        raise ValueError(f"{path}: truncated header")
    magic, version, owner, hidden, experts, k, header_bytes, record_bytes, reserved = struct.unpack(
        "<8sIHHHHIII", raw_header
    )
    if magic != MAGIC or version != 1:
        raise ValueError(f"{path}: unsupported trace header {magic!r} v{version}")
    expected = expected_record_bytes(hidden, k)
    if hidden != HIDDEN or experts != EXPERTS or header_bytes != HEADER_BYTES or record_bytes != expected or reserved != 0:
        raise ValueError(
            f"{path}: geometry/layout mismatch hidden={hidden} experts={experts} k={k} "
            f"header={header_bytes} record={record_bytes} expected_record={expected} reserved={reserved}"
        )
    if not (1 <= k <= EXPERTS):
        raise ValueError(f"{path}: invalid k={k}")

    size = path.stat().st_size
    payload = size - HEADER_BYTES
    if payload < 0 or payload % record_bytes:
        raise ValueError(f"{path}: partial record payload={payload} record_bytes={record_bytes}")
    n = payload // record_bytes
    if n == 0:
        raise ValueError(f"{path}: no examples")

    dtype = record_dtype(hidden, k)
    if dtype.itemsize != record_bytes:
        raise AssertionError(f"{path}: dtype itemsize {dtype.itemsize} != record {record_bytes}")
    mm = np.memmap(path, mode="r", dtype=np.uint8, offset=HEADER_BYTES, shape=(n, record_bytes))
    records = np.ndarray(shape=(n,), dtype=dtype, buffer=mm)

    return TraceHead(
        owner=int(owner),
        k=int(k),
        record_bytes=int(record_bytes),
        records=records,
    )


def load_trace_prefix(
    path: Path,
    allowed_runs: set[int],
    expected_counts: dict[int, int] | None = None,
) -> TraceHead:
    """Load the longest whole-run prefix of `path` that uses only `allowed_runs`.

    Collectors append continuously, so a corpus under active collection ends in
    a partial record and (mid-run) an incomplete final run. Both are fine to
    *ignore*: what is never fine is training on a fragment of a run, because the
    held-out split is defined over runs.

    Two conditions are enforced, and both matter:

    1. **Runs are contiguous.** An append-only trace writes runs back to back, so
       the result is a byte prefix. If any allowed run also has records *after*
       the chosen boundary, the boundary is in the middle of a run and the loader
       refuses rather than silently truncating.
    2. **Runs are complete.** Each owner file is written layer by layer within a
       token, so at any instant owner 6 can be one record ahead of owner 37. A run
       that is complete in one head's file and one record short in another would
       make heads train on different data at the same nominal scale point, so
       `expected_counts` (from the corpus index) is required to be satisfied per
       run and the prefix stops before the first short run.
    """
    head = load_trace_header(path)
    n_total = head["max_records"]
    if n_total == 0:
        raise ValueError(f"{path}: no complete records yet")
    dtype = record_dtype(head["hidden"], head["k"])
    mm = np.memmap(
        path, mode="r", dtype=np.uint8, offset=HEADER_BYTES,
        shape=(n_total, head["record_bytes"]),
    )
    records = np.ndarray(shape=(n_total,), dtype=dtype, buffer=mm)

    # Walk runs in file order, tracking each run's record count and where it ends.
    run_ids = np.asarray(records["run_id"])
    boundaries: list[tuple[int, int]] = []  # (run_id, exclusive end position)
    start = 0
    for pos in range(1, n_total + 1):
        if pos == n_total or run_ids[pos] != run_ids[start]:
            boundaries.append((int(run_ids[start]), pos))
            start = pos

    keep = 0
    prev_end = 0
    for run_id, end in boundaries:
        if run_id not in allowed_runs:
            break
        run_records = end - prev_end
        if expected_counts is not None:
            expected = expected_counts.get(run_id)
            if expected is not None and run_records < expected:
                break
        keep = end
        prev_end = end
    if keep == 0:
        raise ValueError(
            f"{path}: no complete allowed run is present yet "
            f"({len(allowed_runs)} requested)"
        )
    admitted = set(int(x) for x in np.unique(run_ids[:keep]))
    if admitted != {r for r in allowed_runs if r in admitted}:
        raise AssertionError(f"{path}: admitted runs escaped the allowed set")
    return TraceHead(
        owner=head["owner"],
        k=head["k"],
        record_bytes=head["record_bytes"],
        records=records[:keep],
    )


def load_trace_header(path: Path) -> dict:
    raw_header = path.read_bytes()[:HEADER_BYTES]
    if len(raw_header) != HEADER_BYTES:
        raise ValueError(f"{path}: truncated header")
    magic, version, owner, hidden, experts, k, header_bytes, record_bytes, reserved = struct.unpack(
        "<8sIHHHHIII", raw_header
    )
    if magic != MAGIC or version != 1:
        raise ValueError(f"{path}: unsupported trace header {magic!r} v{version}")
    expected = expected_record_bytes(hidden, k)
    if hidden != HIDDEN or experts != EXPERTS or header_bytes != HEADER_BYTES or record_bytes != expected or reserved != 0:
        raise ValueError(
            f"{path}: geometry/layout mismatch hidden={hidden} experts={experts} k={k} "
            f"header={header_bytes} record={record_bytes} expected_record={expected} reserved={reserved}"
        )
    if not (1 <= k <= EXPERTS):
        raise ValueError(f"{path}: invalid k={k}")
    payload = path.stat().st_size - HEADER_BYTES
    if payload < 0:
        raise ValueError(f"{path}: truncated payload")
    return {
        "owner": int(owner),
        "k": int(k),
        "hidden": int(hidden),
        "record_bytes": int(record_bytes),
        # Floor: a partially written trailing record is ignored, never read.
        "max_records": payload // record_bytes,
    }


def load_corpus_entries(trace_dir: Path) -> dict[int, dict]:
    """Full per-run index entries keyed by run id (token counts, bank, prompt)."""
    path = trace_dir / "corpus-index.json"
    if not path.exists():
        return {}
    obj = json.loads(path.read_text())
    return {
        int(entry["run_id"]): entry
        for entry in obj.get("runs", [])
        if entry.get("run_id") is not None
    }


def load_corpus_index(trace_dir: Path) -> dict | None:
    path = trace_dir / "corpus-index.json"
    if not path.exists():
        return None
    obj = json.loads(path.read_text())
    banks: dict[str, list[int]] = {}
    for entry in obj.get("runs", []):
        if entry.get("run_id") is None:
            continue
        banks.setdefault(str(entry["bank"]), []).append(int(entry["run_id"]))
    return {"banks": {name: ids for name, ids in banks.items()}}


def select_run_sets(
    trace_dir: Path,
    run_ids: np.ndarray,
    train_runs: int,
    val_bank: str,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str, dict]:
    """Resolve which whole runs train and which select the checkpoint.

    With a `corpus-index.json` present (the EXP-078 scaling corpus) the split is
    explicit and reproducible: the pool runs are taken as a **nested prefix** in
    collection order, and the validation runs come from a bank of prompts that
    is never trained on at any scale point. Without an index (the EXP-074
    corpus) the previous seeded random run-level split is used unchanged, so a
    v1 reproduction on that corpus still reproduces.
    """
    present = set(int(x) for x in np.unique(run_ids))
    index = load_corpus_index(trace_dir)
    if index is None:
        train_ix, val_ix, mode = split_indices(run_ids, val_fraction, seed)
        return train_ix, val_ix, mode, {"mode": "random-run-split", "val_runs": []}

    pool = [r for r in index["banks"].get("train", []) if r in present]
    pool_set = set(pool)
    if not pool_set:
        raise ValueError(
            f"{trace_dir}: corpus-index.json has no 'train'-bank runs present in the traces"
        )
    if train_runs > 0:
        if len(pool_set) < train_runs:
            # Refuse rather than taking fewer runs than asked for: silently
            # training on 80 runs while the scale point is labelled "50k" is the
            # one failure that would make the scaling curve dishonest.
            raise ValueError(
                f"{trace_dir}: scale point requests {train_runs} pool runs but only "
                f"{len(pool_set)} complete runs are present. Collect more, or pass a "
                f"lower --max-runs, so the point cannot be mislabelled."
            )
        pool_set = set(pool[:train_runs])

    val_ids = [r for r in index["banks"].get(val_bank, []) if r in present]
    if not val_ids:
        raise ValueError(
            f"{trace_dir}: corpus-index.json has no '{val_bank}'-bank runs present in the traces"
        )
    val_set = set(val_ids)

    train_mask = np.array([int(x) in pool_set for x in run_ids], dtype=bool)
    val_mask = np.array([int(x) in val_set for x in run_ids], dtype=bool)
    mode = (
        f"bank-split train={len(pool_set)} pool runs "
        f"val={len(val_set)} runs from bank '{val_bank}'"
    )
    return (
        np.flatnonzero(train_mask),
        np.flatnonzero(val_mask),
        mode,
        {"mode": "bank-split", "pool_runs": sorted(pool_set), "val_runs": sorted(val_set)},
    )


def split_indices(run_ids: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    unique = np.unique(run_ids)
    rng = np.random.default_rng(seed)
    if len(unique) >= 2:
        shuffled = unique.copy()
        rng.shuffle(shuffled)
        n_val_runs = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * val_fraction))))
        val_runs = set(int(x) for x in shuffled[:n_val_runs])
        val_mask = np.array([int(x) in val_runs for x in run_ids], dtype=bool)
        mode = f"run-level ({n_val_runs}/{len(unique)} runs)"
    else:
        order = np.arange(len(run_ids))
        rng.shuffle(order)
        n_val = max(1, min(len(order) - 1, int(round(len(order) * val_fraction))))
        val_mask = np.zeros(len(order), dtype=bool)
        val_mask[order[:n_val]] = True
        mode = "sample-level fallback (only one run present)"
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask), mode


def load_adapter(path: Path) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            tensors[key] = np.array(f.get_tensor(key), copy=True)
    return tensors


def make_batch(trace: TraceHead, indices: np.ndarray) -> tuple[mx.array, mx.array, mx.array]:
    h = np.asarray(trace.hidden[indices], dtype=np.float32)
    b = len(indices)
    features = np.zeros((b, FEATURES), dtype=np.float32)
    features[:, :HIDDEN] = h
    rows = np.arange(b)

    cur = np.asarray(trace.current[indices], dtype=np.int64)
    prv = np.asarray(trace.previous[indices], dtype=np.int64)
    for j in range(trace.k):
        valid_cur = cur[:, j] < EXPERTS
        features[rows[valid_cur], HIDDEN + cur[valid_cur, j]] = 1.0
        valid_prev = prv[:, j] < EXPERTS
        features[rows[valid_prev], HIDDEN + EXPERTS + prv[valid_prev, j]] = 1.0

    target = np.asarray(trace.target[indices], dtype=np.int32)
    weights = np.asarray(trace.target_weights[indices], dtype=np.float32)
    sums = weights.sum(axis=1, keepdims=True)
    weights = weights / np.maximum(sums, 1e-8)
    return mx.array(features), mx.array(target), mx.array(weights)


def gelu_erf(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def logits_for(params: list[mx.array], x: mx.array) -> mx.array:
    w1, w2, wlin = params
    hidden = gelu_erf(x @ w1.T)
    return hidden @ w2.T + x @ wlin.T


def loss_fn(params: list[mx.array], x: mx.array, target: mx.array, weights: mx.array) -> mx.array:
    logits = logits_for(params, x)
    selected = mx.take_along_axis(logits, target, axis=1)
    return mx.mean(mx.logsumexp(logits, axis=1) - mx.sum(selected * weights, axis=1))


def metric_widths(k: int) -> tuple[int, ...]:
    return tuple(sorted({1, k, 8, 12}))


def metrics(params: list[mx.array], trace: TraceHead, indices: np.ndarray, batch_size: int) -> dict[str, float]:
    widths = metric_widths(trace.k)
    empty = {
        "loss": float("nan"),
        "exact_top1": float("nan"),
        "top1_in_target": float("nan"),
        **{f"recall{m}": float("nan") for m in widths},
        **{f"weighted_mass{m}": float("nan") for m in widths},
        **{f"full{m}": float("nan") for m in widths if m >= trace.k},
    }
    if len(indices) == 0:
        return empty

    losses: list[tuple[float, int]] = []
    exact_top1 = 0
    top1_in_target = 0
    hits = {m: 0 for m in widths}
    weighted_mass = {m: 0.0 for m in widths}
    full = {m: 0 for m in widths if m >= trace.k}
    seen = 0
    max_width = max(widths)

    for start in range(0, len(indices), batch_size):
        ix = indices[start : start + batch_size]
        x, target, weights = make_batch(trace, ix)
        logits = logits_for(params, x)
        loss = loss_fn(params, x, target, weights)
        pred = mx.argpartition(logits, kth=-max_width, axis=1)[:, -max_width:]
        argmax = mx.argmax(logits, axis=1)
        mx.eval(loss, pred, argmax, logits)

        losses.append((float(loss.item()), len(ix)))
        p = np.asarray(pred)
        a = np.asarray(argmax)
        t = trace.target[ix].astype(np.int64, copy=False)
        tw = np.asarray(trace.target_weights[ix], dtype=np.float32)
        tw = tw / np.maximum(tw.sum(axis=1, keepdims=True), 1e-8)
        logits_np = np.asarray(logits)

        for row_p, row_a, row_t, row_w, row_logits in zip(p, a, t, tw, logits_np):
            ranked = sorted((int(v) for v in row_p), key=lambda e: (-float(row_logits[e]), e))
            target_set = set(int(v) for v in row_t)
            exact_top1 += int(int(row_a) == int(row_t[0]))
            top1_in_target += int(int(row_a) in target_set)
            for m in widths:
                pred_set = set(ranked[:m])
                inter = pred_set & target_set
                hits[m] += len(inter)
                weighted_mass[m] += sum(
                    float(row_w[j])
                    for j, expert in enumerate(row_t)
                    if int(expert) in pred_set
                )
                if m >= trace.k:
                    full[m] += int(len(inter) == trace.k)
        seen += len(ix)

    result = {
        "loss": sum(v * n for v, n in losses) / seen,
        "exact_top1": exact_top1 / seen,
        "top1_in_target": top1_in_target / seen,
    }
    for m in widths:
        # Candidate recall follows EXP-073 convention: recovered target experts
        # divided by K*sample_count. recall@1 therefore maxes at 1/K.
        result[f"recall{m}"] = hits[m] / (seen * trace.k)
        result[f"weighted_mass{m}"] = weighted_mass[m] / seen
        if m >= trace.k:
            result[f"full{m}"] = full[m] / seen
    return result


class AdamW:
    def __init__(
        self,
        params: list[mx.array],
        lr: float,
        weight_decay: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [mx.zeros_like(p) for p in params]
        self.v = [mx.zeros_like(p) for p in params]

    def step(self, params: list[mx.array], grads: list[mx.array]) -> list[mx.array]:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bias1 = 1.0 - b1 ** self.t
        bias2 = 1.0 - b2 ** self.t
        out = []
        new_m = []
        new_v = []
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * (g * g)
            update = (m / bias1) / (mx.sqrt(v / bias2) + self.eps)
            if self.weight_decay:
                update = update + self.weight_decay * p
            p = p - self.lr * update
            out.append(p)
            new_m.append(m)
            new_v.append(v)
        self.m, self.v = new_m, new_v
        return out


def init_params(
    base: dict[str, np.ndarray] | None,
    owner: int,
    k: int,
    mode: str,
    seed: int,
) -> list[mx.array]:
    """Build one head's three weight tensors.

    `adapter` copies the supplied head (the published Edge0 head for Arm A, the
    previous Logan RouteScout head for Arm B). `random` draws the published
    architecture's own initialization: fan-in scaled uniform for fc1/fc2 and a
    zero `linear_init`, which is the same shape family the published adapter
    uses and keeps the skip path from injecting noise.
    """
    prefix = f"layers.{owner}"
    if mode == "adapter":
        if base is None:
            raise ValueError("--init adapter requires --base-adapter")
        return [
            mx.array(base[f"{prefix}.fc1.weight"].astype(np.float32)),
            mx.array(base[f"{prefix}.fc2.weight"].astype(np.float32)),
            mx.array(base[f"{prefix}.linear_init.weight"].astype(np.float32)),
        ]
    if mode != "random":
        raise ValueError(f"unknown init mode {mode!r}")
    rng = np.random.default_rng(seed + 7919 * owner)
    limit1 = 1.0 / math.sqrt(FEATURES)
    limit2 = 1.0 / math.sqrt(HEAD_HIDDEN)

    def uniform(shape: tuple[int, int], limit: float) -> mx.array:
        return mx.array(rng.uniform(-limit, limit, size=shape).astype(np.float32))

    return [
        uniform((HEAD_HIDDEN, FEATURES), limit1),
        uniform((EXPERTS, HEAD_HIDDEN), limit2),
        mx.zeros((EXPERTS, FEATURES), dtype=mx.float32),
    ]


def train_owner(
    trace: TraceHead,
    base: dict[str, np.ndarray] | None,
    args: argparse.Namespace,
    run_select: int,
    val_bank: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    prefix = f"layers.{trace.owner}"
    params = init_params(base, trace.owner, trace.k, args.init, args.seed)

    train_ix, val_ix, split_mode, split_detail = select_run_sets(
        args.trace_dir,
        trace.run_id,
        run_select,
        val_bank,
        args.val_fraction,
        args.seed + trace.owner,
    )
    if len(train_ix) == 0:
        raise ValueError(
            f"owner {trace.owner}: no training records after run selection "
            f"({split_mode}); raise --max-runs or collect more pool runs"
        )
    baseline = metrics(params, trace, val_ix, args.eval_batch_size)
    opt = AdamW(params, args.lr, args.weight_decay)
    value_and_grad = mx.value_and_grad(loss_fn)
    rng = np.random.default_rng(args.seed + 1000 + trace.owner)

    best_params = [mx.array(p) for p in params]
    best = dict(baseline)
    best_epoch = 0
    history = []
    key_mass = f"weighted_mass{trace.k}"
    key_recall = f"recall{trace.k}"

    def selection_key(m: dict) -> tuple[float, float, float, float]:
        """Model selection: weighted mass@K, recall@K, top1-in-K4, then loss.

        This is the ordering the EXP-078 handoff specifies. It is *not* the
        ordering the EXP-074/v1 run used, which stopped at (mass, recall, loss)
        and omitted top1-in-K4. Both are recorded, because the v1 numbers in
        EDGE0_TRAINING.md were produced by the three-key rule and a like-for-like
        reproduction has to use the same rule as the run it reproduces.
        """
        if args.selection == "exp074":
            return (m[key_mass], m[key_recall], -m["loss"])
        return (m[key_mass], m[key_recall], m["top1_in_target"], -m["loss"])

    for epoch in range(1, args.epochs + 1):
        order = train_ix.copy()
        rng.shuffle(order)
        train_loss_num = 0.0
        train_seen = 0
        for start in range(0, len(order), args.batch_size):
            ix = order[start : start + args.batch_size]
            x, target, weights = make_batch(trace, ix)
            loss, grads = value_and_grad(params, x, target, weights)
            params = opt.step(params, grads)
            mx.eval(loss, params, opt.m, opt.v)
            train_loss_num += float(loss.item()) * len(ix)
            train_seen += len(ix)

        val = metrics(params, trace, val_ix, args.eval_batch_size)
        record = {"epoch": epoch, "train_loss": train_loss_num / max(train_seen, 1), **val}
        history.append(record)
        key = selection_key(val)
        best_key = selection_key(best)
        if key > best_key:
            best = dict(val)
            best_epoch = epoch
            best_params = [mx.array(p) for p in params]
            mx.eval(best_params)
        if not args.quiet:
            print(
                f"owner={trace.owner:02} k={trace.k} epoch={epoch:02} "
                f"train_loss={record['train_loss']:.4f} val_loss={val['loss']:.4f} "
                f"top1_in_target={val['top1_in_target']:.3f} "
                f"recall@k={val[key_recall]:.3f} full@k={val[f'full{trace.k}']:.3f} "
                f"mass@k={val[key_mass]:.3f}"
            )

    trained = {
        f"{prefix}.fc1.weight": np.asarray(best_params[0].astype(mx.float16)),
        f"{prefix}.fc2.weight": np.asarray(best_params[1].astype(mx.float16)),
        f"{prefix}.linear_init.weight": np.asarray(best_params[2].astype(mx.float16)),
    }
    report = {
        "owner": trace.owner,
        "k": trace.k,
        "samples": trace.n,
        "runs": int(len(np.unique(trace.run_id))),
        "train_samples": int(len(train_ix)),
        "val_samples": int(len(val_ix)),
        "split": split_mode,
        "split_detail": split_detail,
        "init": args.init,
        "base_adapter": str(args.base_adapter) if args.base_adapter else None,
        "baseline": baseline,
        "best": best,
        "best_epoch": best_epoch,
        "history": history,
    }
    return trained, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--base-adapter", type=Path, default=None,
                    help="initialization adapter; required unless --init random")
    ap.add_argument("--init", choices=("adapter", "random"), default="adapter")
    ap.add_argument("--template-adapter", type=Path, default=Path("/Users/mateo/models/prerouter_edge0_35b.safetensors"),
                    help="adapter supplying untrained heads (owner 38) for the random-init arm")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--owners", default=f"{OWNER_FIRST}-{OWNER_LAST}", help="e.g. 6-37 or 6,7,8")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--val-bank", default="val",
                    help="corpus-index bank used for model selection")
    ap.add_argument("--max-runs", type=int, default=0,
                    help="use only the first N pool runs in collection order (0 = all)")
    ap.add_argument("--live-prefix", action="store_true",
                    help="read only the longest whole-run prefix (for a still-collecting corpus)")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--selection", choices=("exp078", "exp074"), default="exp078",
                    help="checkpoint rule: exp078 = mass@K, recall@K, top1-in-K4, loss")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.init == "adapter" and args.base_adapter is None:
        ap.error("--base-adapter is required when --init adapter")

    if "-" in args.owners and "," not in args.owners:
        lo, hi = (int(x) for x in args.owners.split("-", 1))
        owners = list(range(lo, hi + 1))
    else:
        owners = [int(x) for x in args.owners.split(",") if x.strip()]

    traces: dict[int, TraceHead] = {}
    trace_k: int | None = None
    live_note = ""
    for owner in owners:
        path = args.trace_dir / f"owner-{owner:02}.e0trace"
        if args.live_prefix:
            # Training against a corpus that is still collecting: read only the
            # longest whole-run prefix. Runs the selection does not want (and any
            # partially written trailing run) are excluded, so a scale point is
            # always a clean nested subset rather than a truncated fragment.
            index = load_corpus_index(args.trace_dir)
            if index is None:
                raise ValueError(f"{args.trace_dir}: --live-prefix requires corpus-index.json")
            entries = load_corpus_entries(args.trace_dir)
            pool = index["banks"].get("train", [])
            allowed = set(pool[: args.max_runs] if args.max_runs > 0 else pool)
            allowed |= set(index["banks"].get(args.val_bank, []))
            if not allowed:
                raise ValueError(f"{args.trace_dir}: no usable runs in corpus-index.json yet")
            expected = {
                rid: entries[rid]["tokens"] - 2
                for rid in allowed
                if rid in entries
            }
            trace = load_trace_prefix(path, allowed, expected)
            live_note = (
                f"live-prefix: {len(allowed)} runs allowed, "
                f"{trace.n} complete records/head loaded"
            )
        else:
            trace = load_trace(path)
        if trace.owner != owner:
            raise ValueError(f"{path}: header owner {trace.owner} != expected {owner}")
        if trace_k is None:
            trace_k = trace.k
        elif trace.k != trace_k:
            raise ValueError(f"{path}: K={trace.k} disagrees with dataset K={trace_k}")
        traces[owner] = trace
    if trace_k is None:
        raise ValueError("no owners selected")
    if live_note:
        print(f"TRAIN {live_note}")

    base = load_adapter(args.base_adapter) if args.base_adapter is not None else None
    # A random-init arm still writes a complete 99-tensor adapter, so it needs a
    # template for the heads it does not train (owner 38 and any unselected
    # owner). The published Edge0 adapter is that template for every arm.
    template = base if base is not None else load_adapter(args.template_adapter)
    output_tensors = {name: np.array(v, copy=True) for name, v in template.items()}
    reports = []
    t0 = time.time()

    for owner in owners:
        trained, report = train_owner(traces[owner], base, args, args.max_runs, args.val_bank)
        output_tensors.update(trained)
        reports.append(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        output_tensors,
        str(args.output),
        metadata={
            "format": "logan-edge0-prerouter-v1",
            "base_adapter": str(args.base_adapter) if args.base_adapter else "random",
            "init": args.init,
            "target_k": str(trace_k),
            "objective": f"next-token native-K{trace_k} weighted cross-entropy",
        },
    )

    # The handoff requires the checkpoint hash in the experiment record, and a
    # hash taken here is the only one that provably describes the artifact this
    # run wrote (a later copy could differ if anything re-exports it).
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    summary = {
        "checkpoint_sha256": _sha256(args.output),
        "checkpoint_bytes": args.output.stat().st_size,
        "trace_dir": str(args.trace_dir),
        "trace_k": trace_k,
        "base_adapter": str(args.base_adapter) if args.base_adapter else None,
        "init": args.init,
        "max_runs": args.max_runs,
        "val_bank": args.val_bank,
        "output": str(args.output),
        "owners": owners,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_fraction": args.val_fraction,
        "selection": args.selection,
        "seed": args.seed,
        "elapsed_s": time.time() - t0,
        "train_examples_per_head": reports[0]["train_samples"] if reports else 0,
        "val_examples_per_head": reports[0]["val_samples"] if reports else 0,
        "heads": reports,
    }
    report_path = args.report or args.output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"TRAIN output={args.output}")
    print(f"TRAIN checkpoint_sha256={summary['checkpoint_sha256']}")
    print(f"TRAIN report={report_path}")
    print(
        f"TRAIN k={trace_k} heads={len(reports)} split={reports[0]['split'] if reports else 'n/a'} "
        f"elapsed_s={summary['elapsed_s']:.2f}"
    )
    key_recall = f"recall{trace_k}"
    key_mass = f"weighted_mass{trace_k}"
    before = np.mean([r["baseline"][key_recall] for r in reports])
    after = np.mean([r["best"][key_recall] for r in reports])
    mass_before = np.mean([r["baseline"][key_mass] for r in reports])
    mass_after = np.mean([r["best"][key_mass] for r in reports])
    print(f"TRAIN mean_val_recall@k={before:.4f}->{after:.4f}")
    print(f"TRAIN mean_val_weighted_mass@k={mass_before:.4f}->{mass_after:.4f}")


if __name__ == "__main__":
    main()
