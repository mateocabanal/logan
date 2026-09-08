# logan-ane

Low-level Apple Neural Engine research crate for Logan.

`logan-ane` bypasses CoreML's public model/prediction layer and dynamically resolves the private ANE runtime underneath it. The current implementation targets Apple Silicon macOS and uses raw MIL text, `_ANEInMemoryModelDescriptor`, `_ANEInMemoryModel`, `_ANERequest`, `_ANEIOSurfaceObject`, `_ANEClient`, and IOSurface shared memory.

The private ABI is undocumented and can change with any macOS update. Treat this crate as a local research/runtime backend, not an App Store API.

## API shape

```rust
use logan_ane::{mil, AneRequest, AneRuntime, AneSurface, CompileOptions};

let runtime = AneRuntime::load()?;
let program = mil::relu_fp32(256, 64)?;
let mut model = runtime.compile(&program, CompileOptions::default())?;
model.load()?;

let mut input = AneSurface::new(256 * 64 * 4)?;
let output = AneSurface::new(256 * 64 * 4)?;
input.write_f32(&vec![1.0; 256 * 64])?;

let request = AneRequest::new(&[&input], &[&output], 0)?;
model.evaluate(&request)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## What is wrapped

- Private framework loading via `dlopen` rather than private-framework link flags.
- Runtime ABI checks against the private Objective-C classes/selectors before use.
- ANE hardware/device discovery through `_ANEDeviceInfo`.
- Raw MIL + BLOBFILE weights, including a crate-owned CoreML Blob v2 `weight.bin` builder with 64-byte record alignment and returned MIL offsets.
- In-memory compile/load/evaluate/unload lifecycle.
- IOSurface-backed shared input/output buffers with RAII lock guards.
- Explicit request IOSurface map/unmap hooks.
- `_ANEClient::sharedConnection` access for lower-level experiments.
- Typed `_ANEClient` evaluate/direct-evaluate and IOSurface map/unmap paths over Rust-owned models and requests.
- Unsafe mutable-weight map/sync/unmap bridge with a bounds-checked `MutableWeightMapping` after successful mapping.
- Runtime capability detection for mutable weights, shared events, real-time evaluation, and direct-client evaluation.
- Unsafe raw private-object escape hatches for selectors not yet promoted into the Rust API.
- Non-Apple/non-arm64 stubs that fail with `UnsupportedPlatform` rather than silently falling back.

## Local ABI verification

The implementation was checked against the Objective-C runtime on the development M2 rather than assuming old class-dump headers. On macOS 27.0 build `26A5378j`, `_ANEDeviceInfo` reports one `h14g` ANE with 16 cores, and the installed framework exposes all selectors currently wrapped by the crate.

See [`docs/private-abi.md`](docs/private-abi.md) for the observed selectors and Objective-C type encodings.

## Validation

```bash
cargo check -p logan-ane
cargo test -p logan-ane
cargo run -p logan-ane --example probe
cargo run --release -p logan-ane --example relu
cargo run --release -p logan-ane --example weighted_conv
cargo run --release -p logan-ane --example parallel_dense
cargo run --release -p logan-ane --example metal_bridge
cargo run --release -p logan-ane --example cache_probe
cargo run --release -p logan-ane --example client
./logan-ane/tools/probe-private-abi.sh > /tmp/logan-ane-abi.txt
```

`probe` prints the detected private-runtime capabilities and ANE hardware information.

`relu` compiles raw MIL and executes it through `_ANEInMemoryModel` over IOSurface buffers. On the development M2 it returns the exact expected ReLU output and measures roughly 0.14 ms per synchronous dispatch for this tiny fixture. It is primarily a correctness/ABI smoke test, not an ANE throughput benchmark.

`weighted_conv` builds a real CoreML Blob v2 `weight.bin` containing a 256x256 fp16 identity matrix, references it through MIL `BLOBFILE`, and runs a 1x1 convolution on ANE. The validation M2 returns exact identity output. See [`docs/blob-v2.md`](docs/blob-v2.md) for the binary layout.

`parallel_dense` compiles multiple fixed dense projections into one MIL program, sharing one input cast and one Blob v2 file. This is the first primitive shaped specifically for Qwen GDN's qkv/z/a/b input projections; two 256x256 projections currently execute together at roughly 0.16 ms/dispatch on the validation M2 with exact outputs.

`metal_bridge` proves the heterogeneous memory path on hardware. Metal mutates an ANE input IOSurface through a zero-copy `MTLBuffer`, ANE consumes those same bytes, then Metal mutates the ANE output IOSurface. Both directions validate with zero numerical error. See [`../docs/ane_execution.md`](../docs/ane_execution.md) for the synchronization and execution-island contract.

`cache_probe` validates `AneProgramCache`, the process-local compiled/loaded program cache intended for the ANE executor thread. The cache key must include every model/island/numerical-policy detail that changes generated MIL or weights.

`client` exercises the lower-level `_ANEClient` paths on the same model, including direct evaluation and explicit IOSurface mapping. Tiny-dispatch timings vary noticeably between runs (roughly 0.09-0.17 ms on the validation M2), so the example reports each path side by side but does not claim a stable direct-path speedup.

`tools/probe-private-abi.sh` recompiles the Objective-C runtime dumper and prints every `_ANE*`/`ANE*` class method and Objective-C type encoding visible on the running OS. Keep a probe output when upgrading macOS so ABI changes can be diffed explicitly.

## Mutable weights

The current OS exposes `_ANEClient` selectors to map, synchronize, and unmap a mutable-weight buffer. `logan-ane` exposes them through `unsafe AneClient::map_mutable_weights_raw` because Apple's private `andProcedure:` object contract is undocumented. Once mapping succeeds, byte access is lifetime-bound and `MutableWeightMapping::sync(range)` performs bounds checking before calling the private synchronization selector.

The next step here is to build a known-good mutable-weight MIL fixture and then promote the procedure-object construction into a typed Rust wrapper.

## Research references

The design is informed by `rane`, `rustane`, maderix/ANE, and related ANE reverse-engineering projects. Logan keeps its own implementation so the runtime ABI, lifecycle, diagnostics, and future Metal/ANE shared-memory integration can evolve with the rest of the engine.
