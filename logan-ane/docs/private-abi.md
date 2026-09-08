# Apple Neural Engine private ABI notes

These notes record the ABI that `logan-ane` was validated against. They are observations, **not** an Apple compatibility contract. Re-run the probe after major macOS updates before trusting private selectors.

## Validation host

- macOS 27.0, build `26A5378j`
- Apple M2, 16 GiB unified memory
- `_ANEDeviceInfo`: `hasANE = true`, `numANEs = 1`, `numANECores = 16`
- ANE architecture: `h14g`, subtype `h14`

The implementation dynamically opens:

- `/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine`
- `/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler`

`CoreML.framework` was also loaded during the discovery probe, but `logan-ane` does not need to link against CoreML's public prediction API.

## In-memory compilation

Observed Objective-C runtime methods:

```text
_ANEInMemoryModelDescriptor
+ modelWithMILText:weights:optionsPlist:          @40@0:8@16@24@32
+ modelWithNetworkDescription:weights:optionsPlist: @40@0:8@16@24@32

_ANEInMemoryModel
+ inMemoryModelWithDescriptor:                   @24@0:8@16
- compileWithQoS:options:error:                  B36@0:8I16@20^@28
- loadWithQoS:options:error:                     B36@0:8I16@20^@28
- evaluateWithQoS:options:request:error:         B44@0:8I16@20@28^@36
- unloadWithQoS:error:                           B28@0:8I16^@20
- mapIOSurfacesWithRequest:cacheInference:error: B36@0:8@16B24^@28
- unmapIOSurfacesWithRequest:                    v24@0:8@16
```

The in-memory model also exposes `model`, `state`, `programHandle`, `intermediateBufferHandle`, `queueDepth`, `perfStatsMask`, and `hexStringIdentifier`.

QoS `21` is the current known-good in-memory compile/load/evaluate value and remains caller-overridable through `AneQos`.

## Requests and IOSurface

```text
_ANEIOSurfaceObject
+ objectWithIOSurface: @24@0:8^{__IOSurface=}16

_ANERequest
+ requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:
  @72@0:8@16@24@32@40@48@56@64
+ requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:
  @80@0:8@16@24@32@40@48@56@64@72
```

`logan-ane` owns the IOSurface lifetime directly and retains the private Objective-C request wrapper. Request lifetimes are tied to their Rust surface borrows.

## `_ANEClient`

The process-wide client exposes the lower-level path underneath `_ANEInMemoryModel`:

```text
+ sharedConnection
+ sharedPrivateConnection
- compileModel:options:qos:error:
- loadModel:options:qos:error:
- evaluateWithModel:options:request:qos:error:
- doEvaluateDirectWithModel:options:request:qos:error:
- evaluateRealTimeWithModel:options:request:error:
- mapIOSurfacesWithModel:request:cacheInference:error:
- unmapIOSurfacesWithModel:request:
- prepareChainingWithModel:options:chainingReq:qos:error:
```

The runtime also exposes mutable-weight operations:

```text
- mapMutableWeightsForModel:andProcedure:mappedWeightsBuffer:size:error:
  B56@0:8@16@24^^v32^Q40^@48
- syncMutableWeightsForModel:andProcedure:fromOffset:withSize:error:
  B56@0:8@16@24Q32Q40^@48
- unmapMutableWeightsForModel:andProcedure:
  B32@0:8@16@24
```

`logan-ane` wraps these behind `unsafe AneClient::map_mutable_weights_raw`. The mapping itself is bounds-checked after creation, and `MutableWeightMapping::sync` synchronizes a byte range. The initial call stays unsafe because the concrete private *procedure object* contract is undocumented and can vary with compiled MIL.

Related classes observed during discovery:

```text
_ANEProgramProcedurePriv
- initWithSymbolName:
- symbolName
- mutableWeightsBuffer                 {ANEBufferMapping=QQ}
- mutableWeightsBufferPtr              ^{ANEBufferMapping=QQ}

_ANEProcedureData
+ procedureDataWithSymbol:weightArray:
- procedureSymbol
- mutableWeightsBufferSize             Q16@0:8
- mutableWeightsBufferID               Q16@0:8
```

These are deliberately not promoted into the safe API until a mutable-weight MIL fixture exercises them end-to-end.

## Correctness probe

`cargo run --release -p logan-ane --example relu` compiles a weight-free MIL program:

```text
fp32 IOSurface -> fp16 cast -> relu -> fp32 cast -> IOSurface
```

On the validation host, the probe produced exact expected output (`max abs error = 0`) and a measured synchronous dispatch around 0.14 ms. The benchmark is intended as an ABI/correctness smoke test, not a throughput characterization of the ANE.

A second probe exercises the lower-level `_ANEClient` execution path with the same compiled model and IOSurfaces. All tested paths produce exact output. Tiny-dispatch timings vary significantly between runs (roughly 0.09-0.17 ms on the validation M2), including which direct/mapped variant wins, so these measurements are useful for ABI/overhead experiments but are not a stable performance guarantee. Re-measure with the actual Logan kernel and after OS updates.

The source for the runtime class/selector dumper is kept in `tools/abi_probe.m`; `tools/probe-private-abi.sh` recompiles it against the current SDK/runtime so these type encodings can be diffed across macOS versions.
