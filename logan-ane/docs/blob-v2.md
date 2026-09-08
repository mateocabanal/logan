# CoreML MIL Blob v2 notes

`logan-ane::BlobV2Builder` writes the binary `weight.bin` format referenced by MIL `BLOBFILE(...)` constants.

This format is separate from the MIL protobuf `DataType` enum. It is the blob-storage container consumed by the CoreML/ANE compiler.

## File layout

All metadata records start on 64-byte boundaries.

### 64-byte file header

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | `u32` | record count |
| 4 | `u32` | format version (`2`) |
| 8 | 56 bytes | zero padding |

### 64-byte record metadata

| Offset within record | Type | Meaning |
| ---: | --- | --- |
| 0 | `u32` | sentinel `0xDEADBEEF` |
| 4 | `u32` | BlobDataType |
| 8 | `u64` | payload size in bytes |
| 16 | `u64` | absolute file offset of payload |
| 24 | 40 bytes | zero padding |

The raw tensor payload begins at the recorded payload offset. A following record's metadata is aligned up to the next 64-byte boundary.

MIL's `BLOBFILE(path = ..., offset = uint64(N))` uses the **metadata-record offset**, not the payload offset. For the first record in a normal Blob v2 file, `N == 64`.

The `_ANEInMemoryModelDescriptor` weight dictionary receives the complete blob file. `WeightBlob::descriptor_offset(0)` is the normal value for a self-contained `weight.bin`; this descriptor offset is independent of the per-constant BLOBFILE metadata offset.

## Blob data types

The values currently mirrored by `BlobDataType` are:

| Value | Type |
| ---: | --- |
| 1 | Float16 |
| 2 | Float32 |
| 3 | UInt8 |
| 4 | Int8 |
| 5 | BFloat16 |
| 6 | Int16 |
| 7 | UInt16 |
| 8 | Int4 |
| 9 | UInt1 |
| 10 | UInt2 |
| 11 | UInt4 |
| 12 | UInt3 |
| 13 | UInt6 |
| 14 | Int32 |
| 15 | UInt32 |
| 16 | Float8E4M3FN |
| 17 | Float8E5M2 |

## Hardware validation

`examples/weighted_conv.rs` builds one Blob v2 file containing a 256x256 fp16 identity matrix, references it from a MIL 1x1 convolution, compiles it through `_ANEInMemoryModelDescriptor`, and evaluates over IOSurface memory.

On the validation M2 this path compiled and returned exact identity output (`max abs error = 0`). This verifies the complete chain:

`Rust tensor bytes -> Blob v2 -> WeightBlob descriptor -> MIL BLOBFILE -> private ANE compiler -> ANE execution`.

## Provenance / compatibility

The layout matches the `MILBlob` storage format documented in Apple's open-source `coremltools` implementation and independently reproduced by Chromium's CoreML WebNN backend. Private ANE behavior remains version-fragile even when the on-disk blob format itself is open-source; keep the runtime ABI probe in `tools/` as the compatibility gate.
