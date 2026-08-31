# Qwen4 performance five-pack gates

Base: `413461184dadf80a0845cd0694ab428f8575c630` (`perf/qwen4-gdn-single-copy`).

The installer `tools/apply_qwen4_five_pack.py --commit` emits five local commits. Do not push the stack until the compile/correctness gates pass. Each runtime-changing feature remains independently comparable.

## Apply in an isolated worktree

```bash
cd ~/CODE/logan
git fetch origin perf/qwen4-five-pack
git worktree add -b perf/qwen4-five-pack-local \
  ~/CODE/logan-five-pack origin/perf/qwen4-five-pack
cd ~/CODE/logan-five-pack
python3 tools/apply_qwen4_five_pack.py --commit
```

## Build / unit gates

```bash
cargo fmt --all
git diff --check
cargo test -p logan-core
cargo test -p logan-metal
cargo test -p logan-compiler
cargo test -p logan-qwen4
cargo build --release -p logan-qwen4
cargo build --release -p logan-compiler
```

Warnings already present on the baseline are not a gate. New errors, Metal/BNNS compile failures, or `mio fails != 0` are.

## Common real-model gate

Unless a section says otherwise:

```bash
CTX=262144
QWEN4_CACHE=32
QWEN_PROMPT="1 2 3 4 5"
QWEN_MAX_NEW=8
```

Expected greedy IDs for the BF16 Apple8 package:

```text
[99048, 96252, 1977, 287, 99709, 43184, 11, 261]
```

Remember that the current profiler divides phase totals by generated-token count. With prompt5 + max_new8 there are 13 full forwards, so actual phase/full-forward is displayed phase × 8 / 13.

---

## 1. Fused recurrent update + output pass

This is a commit-level A/B because the rewrite has no runtime branch. Compare the task-1 commit with its parent using the same command line. It removes one full updated-state traversal in both the Metal and canonical CPU recurrence while keeping ascending-`kk` output accumulation.

Gate:

- exact generated IDs on the BF16 package;
- GDN median improves or is neutral within ~1%;
- wall does not regress >2%;
- no recurrent-state divergence over at least 32 generated tokens.

If it regresses, revert only task 1.

---

## 2. Shared expert under MetalIO

Same binary A/B:

```bash
for mode in 0 1 0 1; do
  /usr/bin/time -l env \
    CTX=262144 QWEN4_CACHE=32 LOGAN_PROFILE=1 \
    QWEN_ATTN_METAL=1 QWEN_QSA_INDEX_METAL=1 \
    QWEN_GDN_METAL=1 QWEN_GDN_SINGLE_COPY=1 \
    QWEN_SHARED_IO_OVERLAP=$mode \
    QWEN_PROMPT="1 2 3 4 5" QWEN_MAX_NEW=8 \
    ./target/release/logan-qwen4 \
    ~/models/Qwen3.8-Flash-Next-FP8.Apple8.coli \
    2>&1 | tee "/tmp/shared-io-${mode}-$(date +%s).txt"
done
```

Gate:

- exact IDs;
- median wall improves >=3%, or the MetalIO+shared critical path clearly shrinks with wall neutral;
- `mio fails=0`;
- CPU shared work must be computed exactly once on both direct-success and fallback paths.

Side effect to watch: CPU shared GEMVs consume UMA bandwidth while MetalIO writes expert slots. If storage throughput drops enough to erase the overlap, keep `QWEN_SHARED_IO_OVERLAP=0` as default or revert.

---

## 3. BNNS BF16 CPU backend

`QWEN_BNNS_BF16=1` enables Accelerate/BNNS for BF16 CPU dense matmuls. It does not keep a second weight copy; only BNNS workspace is retained per thread.

First compare CPU GDN implementations:

```bash
for mode in 0 1 0 1; do
  /usr/bin/time -l env \
    CTX=262144 QWEN4_CACHE=32 LOGAN_PROFILE=1 \
    QWEN_GDN_METAL=0 QWEN_BNNS_BF16=$mode \
    QWEN_ATTN_METAL=1 QWEN_QSA_INDEX_METAL=1 \
    QWEN_GDN_SINGLE_COPY=1 \
    QWEN_PROMPT="1 2 3 4 5" QWEN_MAX_NEW=8 \
    ./target/release/logan-qwen4 \
    ~/models/Qwen3.8-Flash-Next-FP8.Apple8.coli \
    2>&1 | tee "/tmp/bnns-${mode}-$(date +%s).txt"
done
```

Then compare best BNNS CPU result with the normal Metal GDN path.

Gate:

- exact generated IDs if BNNS is to be used by the exact-quality profile;
- CPU-GDN wall should beat the custom NEON CPU path by >=8% to justify the backend;
- only replace Metal GDN if end-to-end wall beats Metal by >=3%;
- peak footprint must stay near the single-copy baseline (no multi-GiB packed-weight copy).

BNNS can choose a different floating reduction order and may use internal threads. If greedy IDs diverge, retain it only as an opt-in throughput backend.

---

## 4. Layer-aware expert LRU

The layer policy protects a small per-layer floor while all capacity above the floors remains one global LRU overflow pool. Defaults:

- `QWEN4_CACHE_POLICY=layer`;
- auto floor 2 if `cap >= 2*layers`, floor 1 if `cap >= layers`, otherwise 0;
- `QWEN4_CACHE_POLICY=global` reproduces the old policy;
- `QWEN4_CACHE_PER_LAYER=N` overrides the auto floor.

At `QWEN4_CACHE=32` the auto floor is 0 for this 48-layer model, so the controlled low-memory baseline remains effectively global.

Capture a routing trace without using it as a performance run:

```bash
env \
  CTX=262144 QWEN4_CACHE=32 QWEN_EXPERT_TRACE=1 \
  QWEN_PROMPT="1 2 3 4 5" QWEN_MAX_NEW=32 \
  ./target/release/logan-qwen4 \
  ~/models/Qwen3.8-Flash-Next-FP8.Apple8.coli \
  2>/tmp/qwen-expert-trace.txt

python3 tools/analyze_expert_trace.py /tmp/qwen-expert-trace.txt \
  --caps 32,48,96,128,192,256 --floors 0,1,2,4
```

Only benchmark policies whose simulated hit rate is materially above global LRU. Suggested runtime A/B:

```bash
QWEN4_CACHE=96 QWEN4_CACHE_POLICY=global ...
QWEN4_CACHE=96 QWEN4_CACHE_POLICY=layer QWEN4_CACHE_PER_LAYER=1 ...
```

Gate:

- higher hit rate at the same capacity;
- exact IDs;
- median wall improves >=3%;
- peak footprint stays within ~5% because capacity, not policy, controls slots;
- avoid capacities that recreate the known UMA-pressure regression.

---

## 5. GDN W8-G64 physical package

This is intentionally a separate package/quality tier. The exact BF16 package is unchanged.

The compiler profile quantizes only:

- `linear_attn.in_proj_qkv.weight`;
- `linear_attn.in_proj_z.weight`;
- `linear_attn.out_proj.weight`.

`in_proj_a`, `in_proj_b`, norms, A_log/dt_bias, convolution weights, and recurrent state remain BF16/f32. W8 uses symmetric signed int8 weights and one f32 scale per 64 input columns.

Compile:

```bash
cargo run --release -p logan-compiler -- \
  compile ~/models/Qwen3.8-Flash-Next-FP8 \
  --target macos-arm64-metal-apple8-v1 \
  --quant gdn-w8-g64 \
  --codec none --opt default \
  -o ~/models/Qwen3.8-Flash-Next-W8GDN.Apple8.coli \
  --plan /tmp/qwen38-w8gdn.plan \
  --verify --force
```

The first runtime implementation consumes W8+scales directly with an AArch64 CPU kernel; it deliberately does **not** silently expand the matrices back to BF16 and does not feed W8 bytes to the BF16 Metal kernel. This proves the physical representation, memory reduction, and quality before a dedicated Metal W8 projection kernel is allowed to become the performance default.

Required gates before considering W8 a default profile:

1. Package size / resident GDN bytes fall close to the expected ~2x reduction for qkv/z/out.
2. No load-time BF16 expansion of W8 records.
3. Evaluate more than one greedy prompt; exact token identity is **not** required for a lossy physical profile, but divergence must be characterized.
4. Long-sequence quality/recurrent drift test (at least several thousand tokens or a perplexity/eval slice).
5. End-to-end speed must beat the BF16 exact profile before calling W8 a performance win. If the CPU W8 kernel loses, keep the format but do not enable it by default; the next implementation should be a native Metal W8-G64 GDN projection kernel.

## Push only after gates

```bash
git log --oneline --decorate -8
git status --short
git push -u origin HEAD:perf/qwen4-five-pack
```

Do not merge the five commits as a unit merely because the stack compiles. Keep/revert each slice according to its own gate.