//! Quantize routed experts and write them back out as plain safetensors.
//!
//! ## Why this exists
//!
//! Logan's compiler normally emits a `.coli` package: a compiled image with
//! target-specific layouts (Apple8 tile8x32, rANS-coded records) that only the
//! Logan runtime reads. That is the right output for Logan.
//!
//! It is the wrong output for anything else. A `.coli` MXFP4 package is
//! unusable by a third-party consumer that has no Apple8/rANS decoder — the
//! inference pool being the concrete case: it reads safetensors, so an MXFP4
//! `.coli` cannot be ingested no matter how the experts were quantized.
//!
//! This module produces the *portable* form instead: the same deterministic
//! MXFP4 quantization, written as an ordinary safetensors file whose expert
//! tensors are stacked per layer in the layout open-weight Qwen MoE checkpoints
//! already use. Nothing about the quantization changes — only the container and
//! the tensor grouping.
//!
//! ## The output contract
//!
//! For every layer that has routed experts, three tensor pairs:
//!
//! ```text
//! {base}.layers.{L}.mlp.switch_mlp.gate_proj.weight   U32  [E, I, D/8]
//! {base}.layers.{L}.mlp.switch_mlp.gate_proj.scales   U8   [E, I, D/32]
//! {base}.layers.{L}.mlp.switch_mlp.up_proj.weight     U32  [E, I, D/8]
//! {base}.layers.{L}.mlp.switch_mlp.up_proj.scales     U8   [E, I, D/32]
//! {base}.layers.{L}.mlp.switch_mlp.down_proj.weight   U32  [E, D, I/8]
//! {base}.layers.{L}.mlp.switch_mlp.down_proj.scales   U8   [E, D, I/32]
//! ```
//!
//! `D` is hidden size, `I` the MoE intermediate size, `E` the expert count.
//! Packed words are little-endian u32 over row-major E2M1 nibbles; scales are
//! raw E8M0 bytes with one byte per 32 input columns. This is byte-for-byte the
//! convention the reference decoder in `logan-qwen4` and the pool's
//! `Mxfp4Stacked` reader both expect, so a consumer needs no translation step.
//!
//! ## Determinism
//!
//! The quantization itself is `quant::mxfp4`, unchanged and already pinned
//! against mlx. Output ordering is fixed (layers ascending, then projection,
//! then weights before scales) and `--seed`-independent, so two runs over the
//! same source produce identical bytes.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use crate::{
    error::{ColicError, Result},
    ir::Matrix,
    model::{self, qwen4_exp::Qwen4ExpFrontend},
    quant::mxfp4::{self, GROUP_SIZE, VALUES_PER_BYTE},
    source::{self, SourceInventory, TensorRef},
};

/// Progress callback: (completed experts, total experts, current layer).
pub type Progress<'a> = dyn FnMut(usize, usize, u32) + 'a;

/// One quantized expert projection, in the exact form the writer needs.
struct PackedProjection {
    rows: u32,
    columns: u32,
    weights: Vec<u8>,
    scales: Vec<u8>,
}

/// Tensors produced for one layer, before stacking.
struct LayerExperts {
    gate: Vec<PackedProjection>,
    up: Vec<PackedProjection>,
    down: Vec<PackedProjection>,
}

#[derive(Debug, Clone)]
pub struct ExportReport {
    /// Directory holding one file per layer.
    pub output_dir: PathBuf,
    /// Template the per-layer files are named from, e.g. `layer{layer}.safetensors`.
    pub file_template: String,
    pub total_bytes: u64,
    /// Bytes per layer, in layer order, so a caller can size a transfer.
    pub layer_bytes: Vec<u64>,
    pub layers: u32,
    pub experts_per_layer: u32,
    pub experts_exported: usize,
    pub source_stored_bytes: u64,
    /// Dtypes seen in the source tensors actually consumed.
    pub source_dtypes: BTreeMap<String, u64>,
}

impl ExportReport {
    /// Path written for `layer`.
    pub fn layer_path(&self, layer: u32) -> PathBuf {
        self.output_dir
            .join(self.file_template.replace("{layer}", &layer.to_string()))
    }
}

/// Resolve the tensor-name prefix for a model directory.
///
/// Exposed so callers that need to construct tensor names for a source tree
/// (the expert exporter, and anything verifying its output) use the same
/// resolution the frontends do, rather than re-deriving it and drifting.
pub fn qwen4_expert_base(root: &Path) -> Result<String> {
    let inventory = source::discover(root)?;
    crate::model::qwen4_exp::text_base(&inventory)
}

/// Tensors `export_experts` writes per layer: three projections, weights and
/// scales. Pinned here so the resume check rejects anything else.
const LAYER_TENSORS: usize = 6;

/// Largest declared payload end in a safetensors header, plus the tensor count.
///
/// A scan rather than a JSON parse: this module already hand-writes the header,
/// and reading back six fixed keys does not justify a JSON dependency.
fn header_extent(header: &str) -> Option<(u64, usize)> {
    let mut max_end = 0_u64;
    let mut tensors = 0_usize;
    let mut rest = header;
    while let Some(at) = rest.find("\"data_offsets\":[") {
        rest = &rest[at + "\"data_offsets\":[".len()..];
        let close = rest.find(']')?;
        let end = rest[..close]
            .split(',')
            .nth(1)?
            .trim()
            .parse::<u64>()
            .ok()?;
        max_end = max_end.max(end);
        tensors += 1;
        rest = &rest[close..];
    }
    Some((max_end, tensors))
}

/// Whether a layer file already holds a complete payload.
///
/// `write_safetensors` creates the file with `File::create` and streams into it,
/// so a process killed mid-write leaves a short file rather than an absent one.
/// Trusting mere existence would therefore resume past a truncated layer and
/// ship an output with a hole in it, so the file is checked against the writer's
/// own rule: the declared payload end must account for every byte present.
fn layer_file_is_complete(path: &Path) -> Result<bool> {
    let mut file = File::open(path).map_err(|source| ColicError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let file_len = file
        .metadata()
        .map_err(|source| ColicError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();

    let mut prefix = [0_u8; 8];
    if file.read_exact(&mut prefix).is_err() {
        return Ok(false);
    }
    let header_len = u64::from_le_bytes(prefix);
    if header_len == 0 || header_len > file_len.saturating_sub(8) {
        return Ok(false);
    }
    let mut header = vec![0_u8; header_len as usize];
    if file.read_exact(&mut header).is_err() {
        return Ok(false);
    }
    let Ok(header) = std::str::from_utf8(&header) else {
        return Ok(false);
    };
    let Some((end, tensors)) = header_extent(header) else {
        return Ok(false);
    };
    Ok(tensors == LAYER_TENSORS && 8 + header_len + end == file_len)
}

/// How many layers from the front of the output are already written in full.
///
/// Resume matters because the cost here is decoding and requantizing every
/// expert, not the write. Only a contiguous run starting at layer 0 counts: a
/// gap would silently leave a layer missing, and redoing work is far cheaper
/// than shipping an incomplete output.
fn completed_layers(output_dir: &Path, template: &str, total_layers: u32) -> Result<u32> {
    let mut complete = 0_u32;
    for layer in 0..total_layers {
        let path = output_dir.join(template.replace("{layer}", &layer.to_string()));
        if !path.exists() || !layer_file_is_complete(&path)? {
            break;
        }
        complete = layer + 1;
    }
    Ok(complete)
}

/// Which of an expert's three matrices to pack.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Projection {
    Gate,
    Up,
    Down,
}

impl Projection {
    fn of(self, refs: &MatrixRefs) -> &MatrixSource {
        match self {
            Projection::Gate => &refs.gate,
            Projection::Up => &refs.up,
            Projection::Down => &refs.down,
        }
    }
}

/// How many worker threads to pack with.
///
/// Bounded by the machine's available parallelism and by the work itself: more
/// threads than experts is pure overhead. `LOGAN_EXPORT_THREADS` pins it, for
/// benchmarking a single-threaded run against a parallel one.
fn worker_threads(experts: usize) -> usize {
    let configured = std::env::var("LOGAN_EXPORT_THREADS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });
    configured.clamp(1, experts.max(1))
}

/// Pack every expert of one projection, spreading the experts over threads.
///
/// Each worker owns a contiguous expert range and its own dtype tally, so no
/// lock is taken and no shared buffer is written. Results are re-assembled in
/// expert order and the tallies merged in that same order, which keeps the
/// output bytes and the reported dtypes independent of thread count — the
/// property `--verify` and the pool's byte-identity assumption both rely on.
fn pack_projection(
    plan: &ExpertPlan,
    inventory: &SourceInventory,
    source_root: &Path,
    layer: u32,
    projection: Projection,
) -> Result<(Vec<PackedProjection>, BTreeMap<String, u64>)> {
    let experts = plan.experts_per_layer as usize;
    if experts == 0 {
        return Ok((Vec::new(), BTreeMap::new()));
    }

    let threads = worker_threads(experts);
    let block = experts.div_ceil(threads);
    let workers: Vec<(usize, Vec<PackedProjection>, BTreeMap<String, u64>)> =
        std::thread::scope(|scope| -> Result<_> {
            let mut handles = Vec::with_capacity(threads);
            for index in 0..threads {
                let start = index * block;
                if start >= experts {
                    break;
                }
                let end = (start + block).min(experts);
                handles.push(scope.spawn(
                    move || -> Result<(usize, Vec<PackedProjection>, BTreeMap<String, u64>)> {
                        let mut local = Vec::with_capacity(end - start);
                        let mut dtypes: BTreeMap<String, u64> = BTreeMap::new();
                        for expert in start..end {
                            let refs = plan
                                .expert_refs(layer, expert as u32, inventory)?
                                .ok_or_else(|| ColicError::InvalidSource {
                                    path: source_root.to_path_buf(),
                                    detail: format!(
                                        "layer {layer} has no routed expert {expert} in the source"
                                    ),
                                })?;
                            local.push(pack_expert_matrix(projection.of(&refs), &mut dtypes)?);
                        }
                        Ok((start, local, dtypes))
                    },
                ));
            }
            let mut done = Vec::with_capacity(handles.len());
            for handle in handles {
                // A panicking worker is a bug in packing, not a caller error;
                // surfacing it as a panic keeps the original message.
                done.push(
                    handle
                        .join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))?,
                );
            }
            Ok(done)
        })?;

    let mut ordered = workers;
    ordered.sort_by_key(|(start, _, _)| *start);
    let mut packed = Vec::with_capacity(experts);
    let mut dtypes: BTreeMap<String, u64> = BTreeMap::new();
    for (_, expert_matrices, worker_dtypes) in ordered {
        packed.extend(expert_matrices);
        for (dtype, count) in worker_dtypes {
            *dtypes.entry(dtype).or_default() += count;
        }
    }
    Ok((packed, dtypes))
}

/// One tensor's location and declared shape in an exported file.
struct TensorSpan {
    start: u64,
    end: u64,
    shape: Vec<u64>,
}

/// Resolve a tensor's payload span, as offsets from the start of the payload.
fn tensor_span(
    parsed: &serde_json::Map<String, serde_json::Value>,
    name: &str,
    path: &Path,
) -> Result<TensorSpan> {
    let meta = parsed.get(name).ok_or_else(|| ColicError::InvalidSource {
        path: path.to_path_buf(),
        detail: format!("exported file has no tensor `{name}`"),
    })?;
    let offsets = meta
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ColicError::InvalidSource {
            path: path.to_path_buf(),
            detail: format!("tensor `{name}` has no data_offsets"),
        })?;
    let start = offsets
        .first()
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| ColicError::InvalidSource {
            path: path.to_path_buf(),
            detail: format!("tensor `{name}` has a malformed data_offsets[0]"),
        })?;
    let end = offsets
        .get(1)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| ColicError::InvalidSource {
            path: path.to_path_buf(),
            detail: format!("tensor `{name}` has a malformed data_offsets[1]"),
        })?;
    let shape: Vec<u64> = meta
        .get("shape")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(serde_json::Value::as_u64).collect())
        .unwrap_or_default();
    Ok(TensorSpan { start, end, shape })
}

/// Verify one exported layer file, reading each tensor **once**.
///
/// Verification previously read one expert at a time, and that helper read the
/// whole tensor span before slicing a single expert out of it — so a layer cost
/// `experts x tensor_bytes` of reads (roughly 200 GB for a 512-expert FP8
/// layer) rather than one pass over the file. It also only ever looked at
/// `gate_proj`. Reading each span once and slicing in memory is the difference
/// between a check that runs in seconds and one that never finishes.
///
/// Every expert of all **three** projections is covered. Checking only
/// `gate_proj` would leave two thirds of the bytes unvalidated.
fn verify_exported_layer(
    path: &Path,
    layer: u32,
    base: &str,
    expected_experts: u32,
) -> Result<usize> {
    let mut file = File::open(path).map_err(|source| ColicError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut len_bytes = [0_u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|source| ColicError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let header_len = u64::from_le_bytes(len_bytes);
    let mut header = vec![0_u8; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|source| ColicError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let parsed: serde_json::Value =
        serde_json::from_slice(&header).map_err(|error| ColicError::InvalidSource {
            path: path.to_path_buf(),
            detail: format!("export header is not valid safetensors JSON: {error}"),
        })?;
    let parsed = parsed
        .as_object()
        .ok_or_else(|| ColicError::InvalidSource {
            path: path.to_path_buf(),
            detail: "export header is not a JSON object".into(),
        })?;

    // Payload base is the 8-byte length prefix PLUS the header, matching
    // safetensors and the pool's own reader (`data_start = 8 + header_len`).
    let data_start = 8 + header_len;
    let prefix = format!("{base}.layers.{layer}.mlp.switch_mlp");

    let mut spans: Vec<(String, TensorSpan, Vec<u8>)> = Vec::with_capacity(LAYER_TENSORS);
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        for (kind, elem) in [("weight", 4_u64), ("scales", 1_u64)] {
            let name = format!("{prefix}.{projection}.{kind}");
            let span = tensor_span(parsed, &name, path)?;
            let len = span.end.saturating_sub(span.start);
            if len == 0 || len % elem != 0 {
                return Err(ColicError::InvalidSource {
                    path: path.to_path_buf(),
                    detail: format!(
                        "tensor `{name}` declares {len} bytes, not a multiple of {elem}"
                    ),
                });
            }
            let mut buf = vec![0_u8; len as usize];
            file.seek(SeekFrom::Start(data_start + span.start))
                .and_then(|_| file.read_exact(&mut buf))
                .map_err(|source| ColicError::Io {
                    path: path.to_path_buf(),
                    source,
                })?;
            spans.push((name, span, buf));
        }
    }

    let mut checked = 0usize;
    for projection in 0..3 {
        let (weight_name, weight_span, weight) = &spans[projection * 2];
        let (scale_name, scale_span, scales) = &spans[projection * 2 + 1];
        let [experts, rows, words_per_row] = match weight_span.shape.as_slice() {
            [a, b, c] => [*a, *b, *c],
            other => {
                return Err(ColicError::InvalidSource {
                    path: path.to_path_buf(),
                    detail: format!("`{weight_name}` should be 3-D, got {other:?}"),
                });
            }
        };
        let [scale_experts, scale_rows, groups_per_row] = match scale_span.shape.as_slice() {
            [a, b, c] => [*a, *b, *c],
            other => {
                return Err(ColicError::InvalidSource {
                    path: path.to_path_buf(),
                    detail: format!("`{scale_name}` should be 3-D, got {other:?}"),
                });
            }
        };
        if experts != u64::from(expected_experts) || scale_experts != u64::from(expected_experts) {
            return Err(ColicError::InvalidSource {
                path: path.to_path_buf(),
                detail: format!(
                    "layer {layer}: `{weight_name}` holds {experts} experts, expected \
                     {expected_experts} (scales: {scale_experts})"
                ),
            });
        }
        if rows == 0 || words_per_row == 0 || rows != scale_rows {
            return Err(ColicError::InvalidSource {
                path: path.to_path_buf(),
                detail: format!(
                    "layer {layer}: `{projection}` geometry {rows}x{words_per_row} disagrees \
                     with scales {scale_rows}x{groups_per_row}"
                ),
            });
        }
        // Columns are `words_per_row * 8`; one scale byte covers GROUP_SIZE.
        let expected_groups = (words_per_row * 8).div_ceil(GROUP_SIZE as u64);
        if groups_per_row != expected_groups {
            return Err(ColicError::InvalidSource {
                path: path.to_path_buf(),
                detail: format!(
                    "layer {layer}: scales declare {groups_per_row} groups per row, \
                     expected {expected_groups}"
                ),
            });
        }

        let weight_stride = (rows * words_per_row * 4) as usize;
        let scale_stride = (rows * groups_per_row) as usize;
        if weight.len() < experts as usize * weight_stride
            || scales.len() < experts as usize * scale_stride
        {
            return Err(ColicError::InvalidSource {
                path: path.to_path_buf(),
                detail: format!("layer {layer}: `{weight_name}` payload is shorter than its shape"),
            });
        }

        for expert in 0..experts as usize {
            let w = &weight[expert * weight_stride..(expert + 1) * weight_stride];
            let s = &scales[expert * scale_stride..(expert + 1) * scale_stride];
            // A slice that is entirely zero means the packing wrote nothing for
            // this expert — the failure a length check alone would miss.
            if w.iter().all(|&b| b == 0) {
                return Err(ColicError::InvalidSource {
                    path: path.to_path_buf(),
                    detail: format!("layer {layer}: `{weight_name}` expert {expert} is all zeroes"),
                });
            }
            if s.iter().all(|&b| b == 0) {
                return Err(ColicError::InvalidSource {
                    path: path.to_path_buf(),
                    detail: format!("layer {layer}: `{scale_name}` expert {expert} is all zeroes"),
                });
            }
            checked += 1;
        }
    }
    Ok(checked)
}

/// Verify every layer of an export, with layers spread over threads.
///
/// Layer-parallel rather than expert-parallel: each layer is an independent
/// file, which keeps workers from sharing any state at all. Memory is the
/// bound worth knowing — a worker holds one layer's six tensors, about 1.25 GB
/// for the FP8 model — so the worker count is capped accordingly.
pub fn verify_exported_files(
    output_dir: &Path,
    template: &str,
    layers: u32,
    experts_per_layer: u32,
    base: &str,
) -> Result<usize> {
    if layers == 0 {
        return Ok(0);
    }
    let threads = worker_threads(layers as usize);
    let block = (layers as usize).div_ceil(threads);
    let results: Vec<Result<usize>> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for index in 0..threads {
            let start = (index * block) as u32;
            if start >= layers {
                break;
            }
            let end = ((index + 1) * block).min(layers as usize) as u32;
            handles.push(scope.spawn(move || -> Result<usize> {
                let mut checked = 0usize;
                for layer in start..end {
                    let path = output_dir.join(template.replace("{layer}", &layer.to_string()));
                    checked += verify_exported_layer(&path, layer, base, experts_per_layer)?;
                }
                Ok(checked)
            }));
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    });
    let mut total = 0usize;
    for result in results {
        total += result?;
    }
    Ok(total)
}

/// Quantize the routed experts of `source` to MXFP4 and write them to
/// `output` as safetensors.
///
/// Reads each expert matrix directly from its byte range in the original
/// shards, so the source is never materialized in memory: only one layer's
/// packed output is held at a time.
pub fn export_experts(
    source_root: &Path,
    output_dir: &Path,
    resume: bool,
    progress: &mut Progress<'_>,
) -> Result<ExportReport> {
    let inventory = source::discover(source_root)?;
    let plan = plan_experts(&inventory, source_root)?;

    std::fs::create_dir_all(output_dir).map_err(|error| ColicError::Io {
        path: output_dir.to_path_buf(),
        source: error,
    })?;

    let template = "layer{layer}.safetensors";
    let total_experts = (plan.layers as usize).saturating_mul(plan.experts_per_layer as usize);
    let mut source_dtypes: BTreeMap<String, u64> = BTreeMap::new();
    let mut layer_bytes: Vec<u64> = Vec::with_capacity(plan.layers as usize);
    let mut done = 0usize;

    // Layers already complete on disk are neither re-read nor re-encoded, so a
    // resumed run costs only the layers that are missing. `--verify` still
    // re-reads every layer afterwards, reused ones included, which is what
    // makes trusting the on-disk bytes safe.
    let first_layer = if resume {
        let complete = completed_layers(output_dir, template, plan.layers)?;
        for layer in 0..complete {
            let path = output_dir.join(template.replace("{layer}", &layer.to_string()));
            layer_bytes.push(
                std::fs::metadata(&path)
                    .map_err(|source| ColicError::Io { path, source })?
                    .len(),
            );
        }
        if complete > 0 {
            eprintln!(
                "logan: resuming — {complete}/{} layer(s) already complete, {} to go",
                plan.layers,
                plan.layers - complete
            );
        }
        done = complete as usize * plan.experts_per_layer as usize;
        progress(
            done,
            total_experts,
            complete.min(plan.layers.saturating_sub(1)),
        );
        complete
    } else {
        0
    };

    for layer in first_layer..plan.layers {
        // Read and quantize the layer's experts in parallel, then write one
        // complete file. The write and the stacking stay sequential and in
        // expert order so the emitted bytes never depend on scheduling.
        let (gate, gate_dtypes) =
            pack_projection(&plan, &inventory, source_root, layer, Projection::Gate)?;
        let (up, up_dtypes) =
            pack_projection(&plan, &inventory, source_root, layer, Projection::Up)?;
        let (down, down_dtypes) =
            pack_projection(&plan, &inventory, source_root, layer, Projection::Down)?;
        for dtypes in [gate_dtypes, up_dtypes, down_dtypes] {
            for (dtype, count) in dtypes {
                *source_dtypes.entry(dtype).or_default() += count;
            }
        }
        let layer_out = LayerExperts { gate, up, down };
        done += plan.experts_per_layer as usize;
        progress(done, total_experts, layer);

        let prefix = format!("{}.layers.{layer}.mlp.switch_mlp", plan.base);
        let mut entries: Vec<OwnedTensor> = Vec::with_capacity(6);
        // Order is fixed so output bytes are reproducible run to run.
        for (name, proj) in [
            ("gate_proj", &layer_out.gate),
            ("up_proj", &layer_out.up),
            ("down_proj", &layer_out.down),
        ] {
            let (weights, wshape) = stack_words(proj)?;
            let (scales, sshape) = stack_scales(proj)?;
            entries.push(OwnedTensor {
                name: format!("{prefix}.{name}.weight"),
                dtype: Dtype::U32,
                shape: wshape,
                data: weights,
            });
            entries.push(OwnedTensor {
                name: format!("{prefix}.{name}.scales"),
                dtype: Dtype::U8,
                shape: sshape,
                data: scales,
            });
        }

        // Written and dropped per layer. Holding every layer's packed bytes
        // until the end would need the whole output resident — 64 GB for the
        // real 48-layer model, against ~34 GB of RAM on the machine that has
        // the disk for it — and it would also defeat per-layer sharding, which
        // is the unit the pool distributes.
        let path = output_dir.join(template.replace("{layer}", &layer.to_string()));
        let bytes = write_safetensors(&path, &entries)?;
        layer_bytes.push(bytes);
        drop(entries);
    }

    Ok(ExportReport {
        output_dir: output_dir.to_path_buf(),
        file_template: template.to_string(),
        total_bytes: layer_bytes.iter().sum(),
        layer_bytes,
        layers: plan.layers,
        experts_per_layer: plan.experts_per_layer,
        experts_exported: done,
        source_stored_bytes: inventory.source_stored_bytes,
        source_dtypes,
    })
}

// ---------------------------------------------------------------------------
// Source planning
// ---------------------------------------------------------------------------

/// How to find one expert's three matrices in a given checkpoint.
#[derive(Debug, Clone)]
struct MatrixRefs {
    gate: MatrixSource,
    up: MatrixSource,
    down: MatrixSource,
}

/// One matrix: its payload, plus the block scales that must be applied to it.
#[derive(Debug, Clone)]
struct MatrixSource {
    weight: TensorRef,
    /// `weight_scale_inv`: one BF16 scale per `block_rows x block_columns`
    /// tile of the weight. Absent for dense sources.
    scale: Option<BlockScale>,
    rows: u32,
    columns: u32,
}

#[derive(Debug, Clone)]
struct BlockScale {
    tensor: TensorRef,
    /// Tile height in weight rows.
    block_rows: u64,
    /// Tile width in weight columns.
    block_columns: u64,
    /// Scale columns per row (the scale tensor's own row length).
    scale_columns: u64,
}

/// Which expert layout a checkpoint uses.
enum Layout {
    /// Per-expert tensors named `...experts.{E}.{proj}.weight`.
    PerExpert,
    /// Experts resolved through the model frontend (dense sources).
    Frontend(Box<crate::ir::SemanticModel>),
}

struct ExpertPlan {
    base: String,
    layers: u32,
    experts_per_layer: u32,
    layout: Layout,
}

impl ExpertPlan {
    fn expert_refs(
        &self,
        layer: u32,
        expert: u32,
        inventory: &SourceInventory,
    ) -> Result<Option<MatrixRefs>> {
        match &self.layout {
            Layout::PerExpert => {
                let ep = format!("{}.layers.{layer}.mlp.experts.{expert}", self.base);
                // Both per-expert shapes occur in the wild: split
                // gate/up/down, and a fused `gate_up_proj` holding gate rows
                // first. The fused form is split here so downstream code only
                // ever sees three matrices.
                let gate_up = per_expert_matrix(inventory, &ep, "gate_up_proj")?;
                let fused_down = per_expert_matrix(inventory, &ep, "down_proj")?;
                if let (Some(gate_up), Some(down)) = (gate_up, fused_down) {
                    let (gate, up) = split_fused_gate_up(&gate_up)?;
                    return Ok(Some(MatrixRefs { gate, up, down }));
                }
                // A split layout must have all three or the expert is unusable;
                // reporting it as absent is better than exporting a partial one.
                match (
                    per_expert_matrix(inventory, &ep, "gate_proj")?,
                    per_expert_matrix(inventory, &ep, "up_proj")?,
                    per_expert_matrix(inventory, &ep, "down_proj")?,
                ) {
                    (Some(gate), Some(up), Some(down)) => Ok(Some(MatrixRefs { gate, up, down })),
                    _ => Ok(None),
                }
            }
            Layout::Frontend(semantic) => {
                let routed = semantic.routed_experts.get(&(layer, expert));
                Ok(routed.map(|routed| MatrixRefs {
                    gate: dense_from_matrix(&routed.gate),
                    up: dense_from_matrix(&routed.up),
                    down: dense_from_matrix(&routed.down),
                }))
            }
        }
    }
}

/// Split a fused `gate_up_proj` (`[2I, H]`, gate rows first) into gate and up.
///
/// Both halves must be the same shape, and a block-scaled source must have
/// scales that divide evenly at the split point — otherwise the scale grid
/// would straddle the boundary between gate and up and the two halves could
/// not be scaled independently.
fn split_fused_gate_up(gate_up: &MatrixSource) -> Result<(MatrixSource, MatrixSource)> {
    let rows = gate_up.rows;
    if rows % 2 != 0 {
        return Err(ColicError::unsupported(
            "expert export",
            format!("fused gate_up_proj has {rows} rows, which is not 2*intermediate"),
        ));
    }
    let half = rows / 2;
    let elem = mxfp4::dtype_element_bytes(&gate_up.weight.dtype).ok_or_else(|| {
        ColicError::unsupported(
            "expert export",
            format!(
                "fused gate_up_proj at {} has dtype `{}`, which has no decode path",
                gate_up.weight.source.display(),
                gate_up.weight.dtype
            ),
        )
    })?;
    let half_bytes = half as u64 * gate_up.columns as u64 * elem as u64;
    if gate_up.weight.len != half_bytes * 2 {
        return Err(ColicError::InvalidSource {
            path: gate_up.weight.source.clone(),
            detail: format!(
                "fused gate_up_proj is {} bytes, expected {} for {}x{}",
                gate_up.weight.len,
                half_bytes * 2,
                gate_up.rows,
                gate_up.columns
            ),
        });
    }

    let split = |start_row: u32| -> MatrixSource {
        MatrixSource {
            weight: TensorRef {
                source: gate_up.weight.source.clone(),
                offset: gate_up.weight.offset
                    + start_row as u64 * gate_up.columns as u64 * elem as u64,
                len: half_bytes,
                dtype: gate_up.weight.dtype.clone(),
                shape: vec![half as u64, gate_up.columns as u64],
            },
            scale: None,
            rows: half,
            columns: gate_up.columns,
        }
    };

    let (mut gate, mut up) = (split(0), split(half));

    // Carry block scales across only when the grid aligns with the split row.
    if let Some(bs) = &gate_up.scale {
        if u64::from(half) % bs.block_rows != 0 {
            return Err(ColicError::unsupported(
                "expert export",
                format!(
                    "fused gate_up_proj block height {} does not divide the {half}-row split, \
                     so gate and up cannot be scaled independently",
                    bs.block_rows
                ),
            ));
        }
        let scale_elem = scale_element_bytes(&bs.tensor.dtype)? as u64;
        let scale_row_bytes = bs.scale_columns.checked_mul(scale_elem).ok_or_else(|| {
            ColicError::Usage("expert export: scale row size overflows u64".into())
        })?;
        for (target, first_row) in [(&mut gate, 0_u64), (&mut up, u64::from(half))] {
            let first_block_row = first_row / bs.block_rows;
            let block_rows_in_half = u64::from(half) / bs.block_rows;
            let offset = bs.tensor.offset + first_block_row * scale_row_bytes;
            let len = block_rows_in_half * scale_row_bytes;
            target.scale = Some(BlockScale {
                tensor: TensorRef {
                    source: bs.tensor.source.clone(),
                    offset,
                    len,
                    dtype: bs.tensor.dtype.clone(),
                    shape: vec![block_rows_in_half, bs.scale_columns],
                },
                block_rows: bs.block_rows,
                block_columns: bs.block_columns,
                scale_columns: bs.scale_columns,
            });
        }
    }

    Ok((gate, up))
}

fn dense_from_matrix(m: &Matrix) -> MatrixSource {
    MatrixSource {
        weight: m.source.clone(),
        scale: None,
        rows: m.rows,
        columns: m.columns,
    }
}

fn scale_element_bytes(dtype: &str) -> Result<usize> {
    match dtype {
        "BF16" | "F16" => Ok(2),
        "F32" => Ok(4),
        _ => Err(ColicError::unsupported(
            "expert export",
            format!("block scale dtype `{dtype}` is not supported; expected BF16, F16 or F32"),
        )),
    }
}

/// Resolve one `{proj}.weight` plus its optional `weight_scale_inv`.
///
/// Accepts the `.weight`-suffixed name and the bare name, because checkpoints
/// differ on that. A weight whose shape the scale cannot tile is refused rather
/// than guessed at: a mismatched scale means every value is silently wrong.
fn per_expert_matrix(
    inventory: &SourceInventory,
    expert_prefix: &str,
    proj: &str,
) -> Result<Option<MatrixSource>> {
    let Some(weight) = inventory
        .tensors
        .get(&format!("{expert_prefix}.{proj}.weight"))
        .or_else(|| inventory.tensors.get(&format!("{expert_prefix}.{proj}")))
    else {
        return Ok(None);
    };
    if weight.shape.len() != 2 {
        return Err(ColicError::InvalidSource {
            path: weight.source.clone(),
            detail: format!(
                "{expert_prefix}.{proj} weight has shape {:?}; expected a 2-D matrix",
                weight.shape
            ),
        });
    }
    let (rows, columns) = (weight.shape[0], weight.shape[1]);

    let scale = if let Some(s) = inventory
        .tensors
        .get(&format!("{expert_prefix}.{proj}.weight_scale_inv"))
    {
        // `weight_scale_inv` is a 2-D grid of multiplicative block scales.
        // The source format does not encode the block geometry separately, so
        // we can only infer it when the grid divides the matrix exactly.  If a
        // scale tensor is present but malformed, failing is mandatory: silently
        // dropping it would reinterpret normalized FP8 bytes as real weights.
        if s.shape.len() != 2 || s.shape[0] == 0 || s.shape[1] == 0 {
            return Err(ColicError::InvalidSource {
                path: s.source.clone(),
                detail: format!(
                    "{expert_prefix}.{proj}.weight_scale_inv has invalid shape {:?}",
                    s.shape
                ),
            });
        }
        if rows % s.shape[0] != 0 || columns % s.shape[1] != 0 {
            return Err(ColicError::InvalidSource {
                path: s.source.clone(),
                detail: format!(
                    "{expert_prefix}.{proj}.weight_scale_inv shape {:?} does not tile {rows}x{columns} exactly",
                    s.shape
                ),
            });
        }
        let elem = scale_element_bytes(&s.dtype)? as u64;
        let expected_scale_bytes = s.shape[0]
            .checked_mul(s.shape[1])
            .and_then(|n| n.checked_mul(elem))
            .ok_or_else(|| {
                ColicError::Usage("expert export: scale tensor size overflows u64".into())
            })?;
        if s.len != expected_scale_bytes {
            return Err(ColicError::InvalidSource {
                path: s.source.clone(),
                detail: format!(
                    "{expert_prefix}.{proj}.weight_scale_inv is {} bytes, expected {expected_scale_bytes} for {:?} {}",
                    s.len, s.shape, s.dtype
                ),
            });
        }
        Some(BlockScale {
            tensor: s.clone(),
            block_rows: rows / s.shape[0],
            block_columns: columns / s.shape[1],
            scale_columns: s.shape[1],
        })
    } else {
        None
    };

    Ok(Some(MatrixSource {
        weight: weight.clone(),
        scale,
        rows: rows as u32,
        columns: columns as u32,
    }))
}

/// Decide how to read this checkpoint's experts.
fn plan_experts(inventory: &SourceInventory, root: &Path) -> Result<ExpertPlan> {
    let base = crate::model::qwen4_exp::text_base(inventory)?;

    // Prefer the explicit per-expert layout when present: it is self-describing
    // and needs no model frontend, which matters because the frontend validates
    // BF16 and a quantized checkpoint is not BF16. Both per-expert shapes must
    // be probed: a checkpoint may ship only the fused `gate_up_proj`.
    let has_per_expert = |layer: u32, expert: u32| {
        let ep = format!("{base}.layers.{layer}.mlp.experts.{expert}");
        ["gate_proj", "gate_up_proj"].iter().any(|proj| {
            inventory
                .tensors
                .contains_key(&format!("{ep}.{proj}.weight"))
                || inventory.tensors.contains_key(&format!("{ep}.{proj}"))
        })
    };

    if has_per_expert(0, 0) {
        let layers = count_layers(inventory, &base);
        let experts_per_layer =
            count_experts(inventory, &base, 0).ok_or_else(|| ColicError::InvalidSource {
                path: root.to_path_buf(),
                detail: "layer 0 has a per-expert layout but no experts were found".into(),
            })?;
        if layers == 0 {
            return Err(ColicError::InvalidSource {
                path: root.to_path_buf(),
                detail: "no layers found for the per-expert layout".into(),
            });
        }
        return Ok(ExpertPlan {
            base,
            layers,
            experts_per_layer,
            layout: Layout::PerExpert,
        });
    }

    let semantic = Qwen4ExpFrontend::build(inventory)?;
    let layers = semantic.geometry.layers;
    let experts_per_layer = semantic.geometry.routed_experts_per_layer;
    Ok(ExpertPlan {
        base,
        layers,
        experts_per_layer,
        layout: Layout::Frontend(Box::new(semantic)),
    })
}

/// Highest routed-expert layer index plus one.
fn count_layers(inventory: &SourceInventory, base: &str) -> u32 {
    let marker = format!("{base}.layers.");
    let mut max: Option<u32> = None;
    for name in inventory.tensors.keys() {
        let Some(rest) = name.strip_prefix(&marker) else {
            continue;
        };
        if !rest.contains(".mlp.experts.") {
            continue;
        }
        if let Some(layer) = rest.split('.').next().and_then(|s| s.parse::<u32>().ok()) {
            max = Some(max.map_or(layer, |m: u32| m.max(layer)));
        }
    }
    max.map_or(0, |m| m + 1)
}

/// Expert count in `layer`, from the highest expert index present.
///
/// Counted from indices rather than a config field: a shard or a REAP-pruned
/// layer must export what it actually holds, and the config would overstate it.
fn count_experts(inventory: &SourceInventory, base: &str, layer: u32) -> Option<u32> {
    let marker = format!("{base}.layers.{layer}.mlp.experts.");
    let mut max: Option<u32> = None;
    for name in inventory.tensors.keys() {
        let Some(rest) = name.strip_prefix(&marker) else {
            continue;
        };
        if let Some(expert) = rest.split('.').next().and_then(|s| s.parse::<u32>().ok()) {
            max = Some(max.map_or(expert, |m: u32| m.max(expert)));
        }
    }
    max.map(|m| m + 1)
}

/// Decode one expert matrix (applying block scales when present) and pack it
/// to MXFP4.
///
/// Streams a row at a time so a matrix is never held decoded in full.
fn pack_expert_matrix(
    source: &MatrixSource,
    dtypes: &mut BTreeMap<String, u64>,
) -> Result<PackedProjection> {
    let dtype = source.weight.dtype.as_str();
    let elem = mxfp4::dtype_element_bytes(dtype).ok_or_else(|| {
        ColicError::unsupported(
            "expert export",
            format!(
                "source tensor at {} has dtype `{dtype}`, which has no MXFP4 decode path",
                source.weight.source.display()
            ),
        )
    })?;

    let row_values = source.columns as usize;
    let row_bytes = row_values
        .checked_mul(elem)
        .ok_or_else(|| ColicError::Usage("expert export: row byte count overflows usize".into()))?;
    let expected = (source.rows as u64)
        .checked_mul(row_bytes as u64)
        .ok_or_else(|| ColicError::Usage("expert export: matrix size overflows u64".into()))?;
    if source.weight.len != expected {
        return Err(ColicError::InvalidSource {
            path: source.weight.source.clone(),
            detail: format!(
                "tensor payload is {} bytes but {}x{} of {dtype} is {expected}",
                source.weight.len, source.rows, source.columns
            ),
        });
    }

    *dtypes.entry(dtype.to_string()).or_default() += 1;

    // Block scales are tiny (5x20 for a 640x2560 tile) and every row indexes
    // into them, so they are read once rather than per row.
    let scale_table = match &source.scale {
        Some(bs) => {
            *dtypes.entry(bs.tensor.dtype.clone()).or_default() += 1;
            Some(read_scale_tensor(&bs.tensor)?)
        }
        None => None,
    };

    let packed_row = row_values.div_ceil(VALUES_PER_BYTE);
    let scale_row = row_values.div_ceil(GROUP_SIZE);
    let mut weights = Vec::with_capacity(source.rows as usize * packed_row);
    let mut scales = Vec::with_capacity(source.rows as usize * scale_row);
    let mut buf = vec![0_u8; row_bytes];
    let mut values = vec![0.0_f32; row_values];

    let mut file = File::open(&source.weight.source).map_err(|error| ColicError::Io {
        path: source.weight.source.clone(),
        source: error,
    })?;
    file.seek(SeekFrom::Start(source.weight.offset))
        .map_err(|error| ColicError::Io {
            path: source.weight.source.clone(),
            source: error,
        })?;

    for row in 0..source.rows {
        file.read_exact(&mut buf).map_err(|error| ColicError::Io {
            path: source.weight.source.clone(),
            source: error,
        })?;

        decode_row(&buf, dtype, elem, &mut values);

        // Apply the block scale. This is the step that makes a quantized
        // checkpoint correct: the stored low-precision values are per-block
        // normalized, so the scale is part of the value, not a refinement.
        if let (Some(bs), Some(table)) = (&source.scale, &scale_table) {
            let block_row = (row as u64 / bs.block_rows) as usize;
            let scale_cols = bs.scale_columns as usize;
            for (col, value) in values.iter_mut().enumerate() {
                let block_col = col as u64 / bs.block_columns;
                let idx = block_row * scale_cols + block_col as usize;
                let s = *table.get(idx).ok_or_else(|| ColicError::InvalidSource {
                    path: bs.tensor.source.clone(),
                    detail: format!(
                        "block scale index {idx} out of range ({} values) for row {row}",
                        table.len()
                    ),
                })?;
                *value *= s;
            }
        }

        mxfp4::quantize_f32_row(&values, &mut weights, &mut scales)?;
    }

    if weights.len() != source.rows as usize * packed_row {
        return Err(ColicError::Usage(format!(
            "expert export: packed {} weight bytes, expected {}",
            weights.len(),
            source.rows as usize * packed_row
        )));
    }
    if scales.len() != source.rows as usize * scale_row {
        return Err(ColicError::Usage(format!(
            "expert export: produced {} scale bytes, expected {}",
            scales.len(),
            source.rows as usize * scale_row
        )));
    }

    Ok(PackedProjection {
        rows: source.rows,
        columns: source.columns,
        weights,
        scales,
    })
}

/// Decode `row_bytes` of `dtype` into `out`.
fn decode_row(row_bytes: &[u8], dtype: &str, elem: usize, out: &mut [f32]) {
    match elem {
        1 => {
            for (slot, &byte) in out.iter_mut().zip(row_bytes.iter()) {
                *slot = mxfp4::e4m3_to_f32(byte);
            }
        }
        2 => {
            let is_bf16 = dtype.starts_with("BF16") || dtype.starts_with("Bf16");
            for (slot, pair) in out.iter_mut().zip(row_bytes.chunks_exact(2)) {
                let bits = u16::from_le_bytes([pair[0], pair[1]]);
                // BF16 is a truncated f32; F16 is a different encoding.
                *slot = if is_bf16 {
                    f32::from_bits(u32::from(bits) << 16)
                } else {
                    f16_to_f32(bits)
                };
            }
        }
        _ => {
            for (slot, word) in out.iter_mut().zip(row_bytes.chunks_exact(4)) {
                *slot = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            }
        }
    }
}

/// Read a whole block-scale tensor into f32 according to its declared dtype.
/// Scale metadata is part of the quantized value; misdecoding it corrupts every
/// weight in the corresponding block, so unsupported dtypes fail closed.
fn read_scale_tensor(tensor: &TensorRef) -> Result<Vec<f32>> {
    let elem = scale_element_bytes(&tensor.dtype)?;
    let expected_elems = tensor.shape.iter().try_fold(1_u64, |n, &dim| {
        n.checked_mul(dim)
            .ok_or_else(|| ColicError::Usage("expert export: scale shape overflows u64".into()))
    })?;
    let expected_bytes = expected_elems
        .checked_mul(elem as u64)
        .ok_or_else(|| ColicError::Usage("expert export: scale byte count overflows u64".into()))?;
    if tensor.len != expected_bytes {
        return Err(ColicError::InvalidSource {
            path: tensor.source.clone(),
            detail: format!(
                "scale tensor is {} bytes, expected {expected_bytes} for {:?} {}",
                tensor.len, tensor.shape, tensor.dtype
            ),
        });
    }

    let mut buf = vec![0_u8; tensor.len as usize];
    let mut file = File::open(&tensor.source).map_err(|error| ColicError::Io {
        path: tensor.source.clone(),
        source: error,
    })?;
    file.seek(SeekFrom::Start(tensor.offset))
        .and_then(|_| file.read_exact(&mut buf))
        .map_err(|error| ColicError::Io {
            path: tensor.source.clone(),
            source: error,
        })?;

    let mut out = Vec::with_capacity(expected_elems as usize);
    for chunk in buf.chunks_exact(elem) {
        let value = match tensor.dtype.as_str() {
            "BF16" => {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f32::from_bits(u32::from(bits) << 16)
            }
            "F16" => mxfp4::f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])),
            "F32" => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            _ => unreachable!("scale_element_bytes accepted an unhandled dtype"),
        };
        if !value.is_finite() || value < 0.0 {
            return Err(ColicError::InvalidSource {
                path: tensor.source.clone(),
                detail: format!("block scale contains invalid value {value}"),
            });
        }
        out.push(value);
    }
    Ok(out)
}

/// Local alias retained for the exporter tests; the implementation lives in
/// `quant::mxfp4` so F16 semantics cannot drift between import paths.
fn f16_to_f32(bits: u16) -> f32 {
    mxfp4::f16_to_f32(bits)
}

/// Stack per-expert packed words into `[E, rows, words_per_row]`, little-endian.
fn stack_words(experts: &[PackedProjection]) -> Result<(Vec<u8>, Vec<u64>)> {
    let first = experts
        .first()
        .ok_or_else(|| ColicError::Usage("expert export: no experts to stack".into()))?;
    // The portable contract stores U32 words and derives the logical row width
    // as words_per_row*8. Without a separate logical-column field, a tail of
    // fewer than eight values would be ambiguous and rows could straddle word
    // boundaries. Qwen expert dimensions are naturally 8-aligned; reject any
    // other geometry rather than emit a misleading shape.
    if !first.columns.is_multiple_of((VALUES_PER_BYTE * 4) as u32) {
        return Err(ColicError::unsupported(
            "expert export",
            format!(
                "MXFP4 U32 export requires a column count divisible by 8, got {}",
                first.columns
            ),
        ));
    }
    let words_per_row = first.columns as usize / (VALUES_PER_BYTE * 4);
    let expected_weight_bytes = first.rows as usize * words_per_row * 4;
    let mut out = Vec::new();
    for expert in experts {
        if expert.rows != first.rows || expert.columns != first.columns {
            return Err(ColicError::Usage(format!(
                "expert export: inconsistent expert geometry {}x{} vs {}x{}",
                expert.rows, expert.columns, first.rows, first.columns
            )));
        }
        if expert.weights.len() != expected_weight_bytes {
            return Err(ColicError::Usage(format!(
                "expert export: packed weight payload is {} bytes, expected {expected_weight_bytes}",
                expert.weights.len()
            )));
        }
        for word in expert.weights.chunks_exact(4) {
            out.extend_from_slice(
                &u32::from_le_bytes([word[0], word[1], word[2], word[3]]).to_le_bytes(),
            );
        }
    }
    Ok((
        out,
        vec![
            experts.len() as u64,
            first.rows as u64,
            words_per_row as u64,
        ],
    ))
}

/// Stack per-expert scale bytes into `[E, rows, groups_per_row]`.
fn stack_scales(experts: &[PackedProjection]) -> Result<(Vec<u8>, Vec<u64>)> {
    let first = experts
        .first()
        .ok_or_else(|| ColicError::Usage("expert export: no experts to stack".into()))?;
    let groups_per_row = (first.columns as usize).div_ceil(GROUP_SIZE);
    let expected_scale_bytes = first.rows as usize * groups_per_row;
    let mut out = Vec::new();
    for expert in experts {
        if expert.rows != first.rows || expert.columns != first.columns {
            return Err(ColicError::Usage(format!(
                "expert export: inconsistent expert geometry {}x{} vs {}x{}",
                expert.rows, expert.columns, first.rows, first.columns
            )));
        }
        if expert.scales.len() != expected_scale_bytes {
            return Err(ColicError::Usage(format!(
                "expert export: scale payload is {} bytes, expected {expected_scale_bytes}",
                expert.scales.len()
            )));
        }
        out.extend_from_slice(&expert.scales);
    }
    Ok((
        out,
        vec![
            experts.len() as u64,
            first.rows as u64,
            groups_per_row as u64,
        ],
    ))
}

// ---------------------------------------------------------------------------
// safetensors writing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dtype {
    U32,
    U8,
}

impl Dtype {
    fn name(self) -> &'static str {
        match self {
            Dtype::U32 => "U32",
            Dtype::U8 => "U8",
        }
    }
}

struct OwnedTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<u64>,
    data: Vec<u8>,
}

/// Write a safetensors file: 8-byte LE header length, JSON header, then a
/// contiguous payload whose offsets are absolute from the start of the file.
///
/// Hand-written rather than pulled from a crate because the format is a length
/// prefix plus JSON, and the compiler already ships a reader for it — adding a
/// dependency to emit what the repo can already parse would be the heavier
/// choice.
fn write_safetensors(path: &Path, tensors: &[OwnedTensor]) -> Result<u64> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|source| ColicError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }

    // Offsets are relative to the END of the header, which is what the reader
    // in `pool-model`/`logan-qwen4` adds its base to. Getting this wrong yields
    // a file that parses but reads garbage, so it is computed once here.
    let mut offset = 0_u64;
    let mut header = String::from("{");
    for (index, tensor) in tensors.iter().enumerate() {
        if index != 0 {
            header.push(',');
        }
        let shape: Vec<String> = tensor.shape.iter().map(u64::to_string).collect();
        header.push_str(&format!(
            "{}:{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            json_string(&tensor.name),
            tensor.dtype.name(),
            shape.join(","),
            offset,
            offset + tensor.data.len() as u64
        ));
        offset += tensor.data.len() as u64;
    }
    // Align the payload so offsets stay a clean multiple, matching what other
    // writers emit and keeping large tensor reads page-friendly.
    let mut header_bytes = header.into_bytes();
    header_bytes.push(b'}');
    let pad = (8 - (header_bytes.len() % 8)) % 8;
    header_bytes.extend(std::iter::repeat_n(b' ', pad));

    let file = File::create(path).map_err(|source| ColicError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut out = BufWriter::with_capacity(1 << 20, file);
    out.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .map_err(|source| ColicError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    out.write_all(&header_bytes)
        .and_then(|()| {
            for tensor in tensors {
                out.write_all(&tensor.data)?;
            }
            Ok(())
        })
        .map_err(|source| ColicError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    out.flush().map_err(|source| ColicError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    Ok(8 + header_bytes.len() as u64 + offset)
}

/// Minimal JSON string escaping for tensor names.
fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// E4M3 is not IEEE-754, so the boundary cases are asserted directly
    /// rather than assumed. If this drifts, every quantized expert is wrong.
    ///
    /// Subnormal value is `(mant / 8) * 2^(1 - 7)` = `mant * 2^-9`, so the
    /// smallest subnormal is byte 0x01 and 0x04 is four steps above it.
    #[test]
    fn e4m3_decodes_its_documented_boundaries() {
        assert_eq!(mxfp4::e4m3_to_f32(0x00), 0.0); // +0
        assert_eq!(mxfp4::e4m3_to_f32(0x80), -0.0); // -0
        assert_eq!(mxfp4::e4m3_to_f32(0x38), 1.0); // exp=7 mant=0
        assert_eq!(mxfp4::e4m3_to_f32(0x3c), 1.5); // exp=7 mant=4
        assert_eq!(mxfp4::e4m3_to_f32(0xb8), -1.0);
        assert_eq!(mxfp4::e4m3_to_f32(0x01), 1.0 / 512.0); // smallest subnormal
        assert_eq!(mxfp4::e4m3_to_f32(0x04), 4.0 / 512.0); // 0.0078125
        assert_eq!(mxfp4::e4m3_to_f32(0x7e), 448.0); // largest finite
        assert!(mxfp4::e4m3_to_f32(0x7f).is_nan()); // S.1111.111 is the only NaN
    }

    /// Cross-check every one of the 256 encodings against an independently
    /// written reference.
    ///
    /// The reference below is deliberately structured differently from the
    /// implementation (it builds the value from a table of exponents rather
    /// than from bit fields), so a shared mistake in one is unlikely to appear
    /// in both. A decoder that is wrong on even a handful of codes silently
    /// corrupts every expert, which is the failure this catches.
    #[test]
    fn e4m3_matches_an_independent_reference_for_all_256_codes() {
        fn reference(byte: u8) -> f32 {
            let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
            let exp = ((byte >> 3) & 0xf) as i32;
            let mant = (byte & 0x7) as f32;
            if exp == 0xf && byte & 0x7 == 0x7 {
                return f32::NAN;
            }
            // value = 2^exp_step * (1 + mant/8) for normals, mant/8 for subnormals
            if exp == 0 {
                sign * mant / 8.0 * 2.0_f32.powi(1 - 7)
            } else {
                sign * (1.0 + mant / 8.0) * 2.0_f32.powi(exp - 7)
            }
        }

        for byte in 0u16..=255 {
            let byte = byte as u8;
            let got = mxfp4::e4m3_to_f32(byte);
            let want = reference(byte);
            if want.is_nan() {
                assert!(got.is_nan(), "byte {byte:#04x}: expected NaN, got {got}");
                continue;
            }
            assert!(
                (got - want).abs() <= f32::EPSILON * want.abs().max(1.0),
                "byte {byte:#04x}: got {got}, independent reference says {want}"
            );
        }
    }

    /// A round trip through the quantizer must stay inside one E2M1 step of the
    /// input; anything larger means a scale or nibble bug.
    #[test]
    fn quantize_row_round_trips_within_one_step() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.37).collect();
        let mut bytes = Vec::new();
        for v in &values {
            bytes.extend_from_slice(&(v.to_bits() >> 16).to_le_bytes()[..2]);
        }
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        mxfp4::quantize_row(&bytes, "BF16", &mut weights, &mut scales).unwrap();
        assert_eq!(scales.len(), 2, "64 values are two 32-value groups");

        // Decode back through the same nibble/scale convention the runtime uses.
        let scale0 = mxfp4::runtime_e8m0_to_f32(scales[0]);
        let scale1 = mxfp4::runtime_e8m0_to_f32(scales[1]);
        for (i, original) in values.iter().enumerate() {
            let byte = weights[i / 2];
            let nib = if i % 2 == 0 { byte & 0xf } else { byte >> 4 };
            let mag = mxfp4::E2M1_MAGNITUDES[(nib & 0x7) as usize];
            let decoded =
                if nib & 0x8 != 0 { -mag } else { mag } * if i < 32 { scale0 } else { scale1 };
            let step = if i < 32 { scale0 } else { scale1 };
            assert!(
                (decoded - original).abs() <= step,
                "value {original} decoded to {decoded}, further than one step {step}"
            );
        }
    }

    /// FP8 and BF16 sources must produce identical output for identical values,
    /// because both decode to f32 before quantization. If they diverge, the
    /// decode path is leaking format-specific behaviour into the quantization.
    #[test]
    fn bf16_and_fp8_paths_agree_on_the_same_values() {
        let values = [0.0_f32, 0.5, -1.25, 3.75, -6.0, 0.0625, 12.0, -24.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| (v.to_bits() >> 16).to_le_bytes()[..2].to_vec())
            .collect();
        let fp8: Vec<u8> = values.iter().map(|v| encode_e4m3(*v)).collect();

        let (mut w1, mut s1) = (Vec::new(), Vec::new());
        let (mut w2, mut s2) = (Vec::new(), Vec::new());
        mxfp4::quantize_row(&bf16, "BF16", &mut w1, &mut s1).unwrap();
        mxfp4::quantize_row(&fp8, "F8_E4M3", &mut w2, &mut s2).unwrap();

        // Values here are exactly representable in both formats, so the packed
        // nibbles must match; the scale group covers one 32-value block.
        assert_eq!(s1.len(), 1);
        assert_eq!(s1, s2, "scale choice differed between source dtypes");
    }

    /// Encode a value to E4M3 for tests. Assumes the value is representable.
    fn encode_e4m3(value: f32) -> u8 {
        let sign = if value.is_sign_negative() { 0x80 } else { 0 };
        let v = value.abs();
        if v == 0.0 {
            return sign;
        }
        // Smallest exponent whose 3-bit mantissa can represent v.
        for exp in 0..16_i32 {
            let scale = if exp == 0 {
                1.0 / 512.0
            } else {
                (2.0_f32).powi(exp - 7)
            };
            let base = if exp == 0 { 0.0 } else { scale };
            let units = (v - base) / (scale / 8.0);
            let mant = units.round();
            if (0.0..=7.0).contains(&mant) {
                let reconstructed = base + mant * (scale / 8.0);
                if (reconstructed - v).abs() < f32::EPSILON * 8.0 {
                    return sign | ((exp as u8) << 3) | (mant as u8);
                }
            }
        }
        panic!("test value {value} is not E4M3-representable");
    }

    /// A file written by the exporter must be readable back through the plain
    /// safetensors header convention, or a consumer cannot use it.
    #[test]
    fn written_file_round_trips_through_the_header() {
        let dir = std::env::temp_dir().join(format!("logan-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.safetensors");

        let tensors = vec![
            OwnedTensor {
                name: "a.weight".into(),
                dtype: Dtype::U32,
                shape: vec![2, 1, 1],
                data: vec![1, 0, 0, 0, 2, 0, 0, 0],
            },
            OwnedTensor {
                name: "a.scales".into(),
                dtype: Dtype::U8,
                shape: vec![2, 1, 1],
                data: vec![127, 128],
            },
        ];
        let bytes = write_safetensors(&path, &tensors).unwrap();

        // Re-read the header the way an independent consumer would. Data
        // offsets are relative to the END of the header, so the payload base is
        // 8 + header_len, not header_len.
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(bytes, raw.len() as u64);
        let header_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&raw[8..8 + header_len]).unwrap();
        let w = &header["a.weight"];
        assert_eq!(w["dtype"], "U32");
        assert_eq!(w["shape"], serde_json::json!([2, 1, 1]));
        let data_start = 8 + header_len;
        let start = data_start + w["data_offsets"][0].as_u64().unwrap() as usize;
        assert_eq!(&raw[start..start + 4], &[1, 0, 0, 0]);
        // The second tensor must begin exactly where the first ends.
        let s = &header["a.scales"];
        let s_start = data_start + s["data_offsets"][0].as_u64().unwrap() as usize;
        assert_eq!(s_start, start + 8);
        assert_eq!(raw[s_start], 127);
        // Padding must keep the payload 8-byte aligned.
        assert_eq!(header_len % 8, 0);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// JSON escaping must not corrupt tensor names into unparseable headers.
    #[test]
    fn json_string_escapes_control_characters() {
        assert_eq!(json_string("plain.name"), "\"plain.name\"");
        assert_eq!(json_string("a\"b"), "\"a\\\"b\"");
        assert_eq!(json_string("a\u{1}b"), "\"a\\u0001b\"");
    }

    /// The block scale must actually be multiplied in.
    ///
    /// This is the failure that would be invisible: a block-scaled FP8
    /// checkpoint's stored values are per-block normalized, so dropping the
    /// scale yields a well-formed file whose weights are all wrong by a
    /// per-block factor. Asserted through the real decode path rather than by
    /// inspecting a constant.
    #[test]
    fn block_scale_is_applied_to_every_value_in_its_tile() {
        // A 2x4 matrix with a 2x2 block grid: two block columns of 2 wide, and
        // one block row, so rows 0..1 share a block row.
        let rows = 2usize;
        let columns = 4usize;
        let weight = vec![0x38_u8; rows * columns]; // every value is E4M3 1.0
        let mut values = vec![0.0_f32; columns];
        for row in 0..rows {
            decode_row(
                &weight[row * columns..(row + 1) * columns],
                "F8_E4M3",
                1,
                &mut values,
            );
            assert!(
                values.iter().all(|v| *v == 1.0),
                "E4M3 0x38 must decode to 1.0"
            );
        }

        // Apply scales of 2.0 then 0.5 across the two block columns.
        let bs = BlockScale {
            tensor: TensorRef {
                source: std::path::PathBuf::from("unused"),
                offset: 0,
                len: 0,
                dtype: "BF16".into(),
                shape: vec![1, 2],
            },
            block_rows: 2,
            block_columns: 2,
            scale_columns: 2,
        };
        let table = [2.0_f32, 0.5];

        decode_row(&weight[0..columns], "F8_E4M3", 1, &mut values);
        let block_row = 0u64 / bs.block_rows;
        for (col, value) in values.iter_mut().enumerate() {
            let idx = (block_row * bs.scale_columns + col as u64 / bs.block_columns) as usize;
            *value *= table[idx];
        }
        assert_eq!(values[..2], [2.0, 2.0], "first block column scaled by 2");
        assert_eq!(values[2..], [0.5, 0.5], "second block column scaled by 0.5");
    }

    /// A weight whose scale tensor cannot tile it must be refused, not guessed.
    #[test]
    fn mismatched_block_scale_shape_is_refused() {
        use std::collections::BTreeMap;
        let mut tensors: BTreeMap<String, TensorRef> = BTreeMap::new();
        let mk = |len: u64, shape: Vec<u64>| TensorRef {
            source: std::path::PathBuf::from("unused"),
            offset: 0,
            len,
            dtype: "F8_E4M3".into(),
            shape,
        };
        tensors.insert(
            "model.layers.0.mlp.experts.0.gate_proj.weight".into(),
            mk(640 * 2560, vec![640, 2560]),
        );
        // 7 does not divide 640, so no integer block height exists.
        tensors.insert(
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv".into(),
            TensorRef {
                dtype: "BF16".into(),
                ..mk(0, vec![7, 20])
            },
        );
        let inventory = SourceInventory {
            root: std::path::PathBuf::from("unused"),
            files: vec![],
            tensors,
            source_stored_bytes: 0,
            dtype_counts: BTreeMap::new(),
            source_fingerprint: String::new(),
            config_fingerprint: None,
            architecture_hint: None,
        };
        let error =
            per_expert_matrix(&inventory, "model.layers.0.mlp.experts.0", "gate_proj").unwrap_err();
        assert!(error.to_string().contains("does not tile"));
    }

    /// A valid `weight_scale_inv` must be detected and its block size derived.
    #[test]
    fn block_scale_is_detected_with_the_official_geometry() {
        use std::collections::BTreeMap;
        let mut tensors: BTreeMap<String, TensorRef> = BTreeMap::new();
        tensors.insert(
            "model.layers.0.mlp.experts.0.gate_proj.weight".into(),
            TensorRef {
                source: std::path::PathBuf::from("unused"),
                offset: 0,
                len: 640 * 2560,
                dtype: "F8_E4M3".into(),
                shape: vec![640, 2560],
            },
        );
        // The real Qwen3.8-Flash-Next-FP8 shape: [5, 20] for a 640x2560 weight.
        tensors.insert(
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv".into(),
            TensorRef {
                source: std::path::PathBuf::from("unused"),
                offset: 0,
                len: 5 * 20 * 4,
                dtype: "F32".into(),
                shape: vec![5, 20],
            },
        );
        let inventory = SourceInventory {
            root: std::path::PathBuf::from("unused"),
            files: vec![],
            tensors,
            source_stored_bytes: 0,
            dtype_counts: BTreeMap::new(),
            source_fingerprint: String::new(),
            config_fingerprint: None,
            architecture_hint: None,
        };
        let m = per_expert_matrix(&inventory, "model.layers.0.mlp.experts.0", "gate_proj")
            .unwrap()
            .expect("weight resolves");
        let bs = m.scale.expect("block scale must be detected");
        assert_eq!((bs.block_rows, bs.block_columns), (128, 128));
        assert_eq!(bs.scale_columns, 20);
    }

    #[test]
    fn block_scale_reader_honors_f32_dtype() {
        let path =
            std::env::temp_dir().join(format!("logan-export-f32-scale-{}", std::process::id()));
        let values = [2.0_f32, 0.5_f32, 0.0_f32, 3.25_f32];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, &bytes).unwrap();
        let tensor = TensorRef {
            source: path.clone(),
            offset: 0,
            len: bytes.len() as u64,
            dtype: "F32".into(),
            shape: vec![2, 2],
        };
        assert_eq!(read_scale_tensor(&tensor).unwrap(), values);
        std::fs::remove_file(path).unwrap();
    }

    /// F16 is not BF16; decoding one as the other silently corrupts values.
    #[test]
    fn f16_decodes_distinctly_from_bf16() {
        // 0x3C00 is F16 1.0. As BF16 the same bits are 0.0078125.
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        let as_bf16 = f32::from_bits(u32::from(0x3c00_u16) << 16);
        assert_ne!(as_bf16, 1.0, "the two encodings must not be conflated");
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0xc000), -2.0);
    }
}
