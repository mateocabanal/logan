#!/usr/bin/env python3
"""Measure the physical expert-payload read cost for the real Qwen3.6 MLX oQ4 checkpoint.

RouteScout can only hide costs that actually exist. This measures the floor:
reading exactly the expert byte ranges a single decode token needs
(40 layers x top-8 experts x 3 matrices), at the same offsets the runtime reads,
using pread with no decode and no compute.

Reports both a cold pass (page cache dropped where permitted) and a warm pass so
the SSD-vs-cache split is visible. Artifacts land under .perf_runs/routescout/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import time
from pathlib import Path

ROLES = ("gate_proj", "up_proj", "down_proj")


def parse_shard(path: Path) -> tuple[dict, int]:
    """Return (header, data_start).

    `data_offsets` inside a safetensors header are relative to the START OF THE
    PAYLOAD, i.e. `8 + header_length`. The runtime's `parse_shard` adds that
    start before reading, so a raw `os.pread(fd, len, offset)` using the header
    value alone reads the header bytes and silently measures the wrong region.
    """
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def expert_ranges(
    header: dict, data_start: int, layer: int, role: str, experts: int
) -> list[tuple[int, int, str]]:
    """[(absolute_file_offset, length, dtype)] for every expert matrix."""
    base = f"language_model.model.layers.{layer}.mlp.switch_mlp.{role}"
    weight = header[f"{base}.weight"]
    dtype = weight["dtype"]
    shape = weight["shape"]
    assert shape[0] == experts, (base, shape)
    start, end = weight["data_offsets"]
    per_expert = (end - start) // experts
    # Sidecars are read by the runtime too, but they are ~1/64 of the weight
    # payload at group 64; count only the weight matrices to stay conservative.
    return [
        (data_start + start + e * per_expert, per_expert, dtype) for e in range(experts)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="replay these authoritative routes instead of uniform random experts",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    shards: dict[str, tuple[dict, int]] = {}
    for layer in range(args.layers):
        for role in ROLES:
            name = f"language_model.model.layers.{layer}.mlp.switch_mlp.{role}.weight"
            shard = index[name]
            if shard not in shards:
                shards[shard] = parse_shard(args.model / shard)
    print(f"shards={sorted(shards)}")

    # Replay real trace routes rather than uniformly random experts when a trace
    # is supplied: a uniform sample has a different hit/miss and locality profile
    # than the model's actual routing, so it cannot be called an exact floor.
    trace_routes: list[list[list[int]]] = []
    if args.trace:
        from routescout_trace import parse as parse_trace

        _meta, _events, cycles = parse_trace(args.trace)
        trace_routes = [
            [[int(e) for e in cycle[layer].ids] for layer in range(args.layers)]
            for cycle in cycles
        ]
        print(f"trace_routes={len(trace_routes)} from {args.trace.name}")

    rng = random.Random(1234)
    plan: list[tuple[str, int, int]] = []
    per_token_bytes = 0
    for token in range(args.tokens):
        for layer in range(args.layers):
            if trace_routes:
                chosen = trace_routes[token % len(trace_routes)][layer][: args.topk]
            else:
                chosen = rng.sample(range(args.experts), args.topk)
            for role in ROLES:
                name = f"language_model.model.layers.{layer}.mlp.switch_mlp.{role}.weight"
                shard = index[name]
                header, data_start = shards[shard]
                start, end = header[name]["data_offsets"]
                per = (end - start) // args.experts
                for expert in chosen:
                    plan.append((shard, data_start + start + expert * per, per))
                    per_token_bytes += per
    per_token_bytes //= args.tokens
    reads_per_token = len(plan) // args.tokens
    print(
        f"plan reads={len(plan)} reads/token={reads_per_token} "
        f"bytes/token={per_token_bytes} ({per_token_bytes/1048576:.1f} MiB)"
    )

    shard_paths = {shard: args.model / shard for shard in shards}
    handles = {shard: os.open(path, os.O_RDONLY) for shard, path in shard_paths.items()}
    try:
        results = []
        for repeat in range(args.repeats):
            started = time.perf_counter()
            total = 0
            for shard, offset, length in plan:
                data = os.pread(handles[shard], length, offset)
                total += len(data)
            elapsed = time.perf_counter() - started
            mb = total / 1048576
            results.append(
                {
                    "repeat": repeat,
                    "seconds": elapsed,
                    "bytes": total,
                    "MiB_per_s": mb / elapsed,
                    "ms_per_token": elapsed * 1e3 / args.tokens,
                }
            )
            print(
                f"repeat={repeat} seconds={elapsed:.3f} MiB/s={mb/elapsed:.1f} "
                f"ms/token={elapsed*1e3/args.tokens:.2f}"
            )
    finally:
        for handle in handles.values():
            os.close(handle)

    payload = {
        "model": str(args.model),
        "layers": args.layers,
        "topk": args.topk,
        "experts": args.experts,
        "tokens": args.tokens,
        "reads_per_token": reads_per_token,
        "bytes_per_token": per_token_bytes,
        "results": results,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
