# Logan

**The mutant compiler.**

Logan is an ahead-of-time compiler + runtime for local LLM inference, written
in Rust. It *mutates* model checkpoints into a form optimized for the hardware
they run on — quantized, target-compiled, disk-streamed — so a 119 GB model
runs on a 16 GB laptop.

The name is a reference to The Wolverine: Logan is a mutant, and that's what
this compiler does. It takes a model and mutates it into a local version
optimized for the current hardware.

## What's in the workspace

| Crate | Role |
|---|---|
| `logan-abi` | Stable ABI identities: target registry, semantic IDs, representation contracts (generated from `abi/coli-target-registry.toml`) |
| `logan-format` | COLI CSF artifact framing: checksums, manifest/data-shard constants, package reader |
| `logan-compiler` | The compiler (`logan` binary): machine probe, physical IR, memory planner, quant (exact / MXFP4-Apple8 / INT4-G32), rANS codec, package emission |
| `logan-qwen` | Scalar Qwen MoE reference (C-identical numerics, token-identity gated) |
| `logan-qwen4` | Qwen4 (Qwen3.8-Flash-Next / Qwen4Exp) runtime: hyper connections, QSA sparse attention, PLE n-gram layer, **Metal/MetalIO direct execution** (fused Apple8 experts + coalesced GDN kernels) |

## The Metal/MetalIO path (macOS)

The runtime streams quantized experts off disk straight into GPU-visible
buffers (MetalIO) and executes them in native Apple8 tile order — no host
decode, no repack. The GDN (Gated DeltaNet) layers run coalesced Metal
kernels over page-aligned zero-copy weights. Every Metal entry point declines
cleanly to the CPU reference path when unavailable; the two paths are
token-identical by gate.

Env gates (all default ON, `=0` opts out): `QWEN_GDN_METAL`,
`QWEN_APPLE8_DIRECT`, `QWEN_APPLE8_OVERLAP`.

## Build & test

```bash
cargo build --release --workspace
cargo test --workspace --all-targets
```

The C-oracle differential tests (rANS/Apple8 identity vs the C decoder)
skip automatically when the C reference tree is absent — the C fork
(`mateocabanal/colibri`) remains the parity oracle and is not vendored here.

## Run

```bash
# tiny fixture gate (deterministic, committed):
cargo run --release -p logan-qwen4 -- fixtures/qwen4_moe_tiny

# compile a checkpoint into a COLI package:
target/release/logan compile MODEL_DIR --target native --quant exact \
  --codec none --opt default -o OUT.coli --verify

# decode a real package (macOS, Metal path):
QWEN_PROMPT="1 2 3 4 5" QWEN_MAX_NEW=8 \
  target/release/logan-qwen4 ~/models/Qwen3.8-Flash-Next-FP8.Apple8.coli
```

## Lineage

Logan is the Rust rewrite of the colibri C engine (pure-C local LLM
inference, "tiny engine, immense model"). The C fork is the reference
implementation and parity oracle; Logan is the forward path.

## License

Apache-2.0
