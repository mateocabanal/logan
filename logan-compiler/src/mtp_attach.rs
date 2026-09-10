//! Append a standalone Qwen4Exp MTP drafter to an existing Apple8 COLI package.
//!
//! The base package is not rebuilt. Existing data shards are left byte-for-byte
//! untouched; new MTP records are written to new shards and a replacement
//! manifest is atomically published only after every new shard is complete.

use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use logan_format::package::Package;
use serde_json::{Value, json};

use crate::{
    error::{ColicError, Result},
    ir::{Matrix, RoutedExpert},
    model::qwen_mtp::{self, QwenMtpExpertBank, QwenMtpFlavor, QwenMtpInventory},
    source::{self, TensorRef},
    storage::{self, LoweredRecord, ManifestRecord, PlannedRecord, StoragePlan},
    target,
};

const SHARD_LIMIT: u64 = 4 * 1024 * 1024 * 1024;
const MTP_METADATA_FILE: &str = "mtp.json";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttachMtpSummary {
    pub package: PathBuf,
    pub drafter: PathBuf,
    pub static_tensors: usize,
    pub experts: u32,
    pub stages: u32,
    pub virtual_layer_base: u32,
    pub new_shards: u32,
    pub new_records: usize,
    pub added_stored_bytes: u64,
    pub backup_manifest: PathBuf,
}

#[derive(Clone)]
enum NewSource {
    Tensor {
        name: String,
        layer: i32,
        tensor: TensorRef,
    },
    Expert {
        name: String,
        expert: RoutedExpert,
    },
}

/// Attach a standalone `qwen4_exp_mtp` checkpoint to an existing
/// `macos-arm64-metal-apple8-v1` COLI package.
///
/// Dense/HC/attention tensors are retained as exact BF16 tensor records.
/// Routed MTP experts are lowered from the original BF16 weights to Logan's
/// pageable Apple8 MXFP4 execution representation, using virtual model layers
/// immediately after the target model's base transformer layers.
pub fn attach_qwen4_mtp(package_root: &Path, drafter_root: &Path) -> Result<AttachMtpSummary> {
    let package = open_package(package_root)?;
    if package.profile() != target::MACOS_ARM64_METAL_APPLE8_V1.name {
        return unsupported(
            package_root,
            format!(
                "MTP append currently targets `{}` packages; got `{}`",
                target::MACOS_ARM64_METAL_APPLE8_V1.name,
                package.profile()
            ),
        );
    }
    if package.alignment() != target::MACOS_ARM64_METAL_APPLE8_V1.record_alignment {
        return unsupported(package_root, "package record alignment does not match Apple8 v1");
    }
    if package
        .records()
        .iter()
        .any(|record| record.name.as_deref().is_some_and(|name| name.starts_with("mtp.")))
        || package_root.join(MTP_METADATA_FILE).exists()
    {
        return Err(ColicError::Usage(format!(
            "package already contains an MTP attachment: {}",
            package_root.display()
        )));
    }
    // The current manifest re-encoder deliberately has no codec-table copy
    // path. Fail closed rather than accidentally stripping compressed-record
    // metadata from a future package.
    if package.records().iter().any(|record| record.codec != 0) {
        return unsupported(
            package_root,
            "MTP append does not yet preserve nonzero outer record codecs",
        );
    }

    let base_config = read_json(&package_root.join("config.json"))?;
    let base_text = base_config.get("text_config").unwrap_or(&base_config);
    let virtual_layer_base = required_config_u32(package_root, base_text, "num_hidden_layers")?;
    let base_hidden = required_config_u32(package_root, base_text, "hidden_size")?;
    let base_inter = required_config_u32(package_root, base_text, "moe_intermediate_size")?;
    let base_hc = required_config_u32(package_root, base_text, "hc_count")?;
    let base_hc_lowrank = required_config_u32(package_root, base_text, "hc_lowrank")?;
    let base_shared =
        required_config_u32(package_root, base_text, "shared_expert_intermediate_size")?;

    let inventory = source::discover(drafter_root)?;
    let mtp = qwen_mtp::inspect(&inventory)?.ok_or_else(|| ColicError::InvalidSource {
        path: drafter_root.to_owned(),
        detail: "checkpoint does not declare MTP".into(),
    })?;
    let (hc_count, hc_lowrank, shared_intermediate) = match mtp.flavor {
        QwenMtpFlavor::Qwen4Exp {
            hc_count,
            hc_lowrank,
            shared_expert_intermediate_size,
            standalone: true,
        } => (hc_count, hc_lowrank, shared_expert_intermediate_size),
        QwenMtpFlavor::Qwen4Exp {
            standalone: false, ..
        } => {
            return unsupported(
                drafter_root,
                "attach-mtp expects a standalone qwen4_exp_mtp checkpoint, not embedded mtp.* tensors",
            );
        }
        QwenMtpFlavor::LegacyQwen3 => {
            return unsupported(drafter_root, "attach-mtp currently supports Qwen4Exp MTP only");
        }
    };
    for (label, base, draft) in [
        ("hidden_size", base_hidden, mtp.hidden_size),
        ("moe_intermediate_size", base_inter, mtp.moe_intermediate_size),
        ("hc_count", base_hc, hc_count),
        ("hc_lowrank", base_hc_lowrank, hc_lowrank),
        (
            "shared_expert_intermediate_size",
            base_shared,
            shared_intermediate,
        ),
    ] {
        if base != draft {
            return unsupported(
                package_root,
                format!("drafter `{label}`={draft} does not match target model `{label}`={base}"),
            );
        }
    }
    if mtp.use_dedicated_embeddings {
        return unsupported(drafter_root, "dedicated MTP embeddings are not supported");
    }
    if let Some(declared) = base_text.get("mtp_num_hidden_layers").and_then(Value::as_u64) {
        if declared != u64::from(mtp.hidden_layers) {
            return unsupported(
                package_root,
                format!(
                    "target declares {declared} MTP layer(s), drafter contains {}",
                    mtp.hidden_layers
                ),
            );
        }
    }

    let sources = build_sources(&mtp, virtual_layer_base)?;
    ensure_unique_names(&package, &sources, package_root)?;

    let first_new_id = package
        .records()
        .iter()
        .map(|record| record.id)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| ColicError::Usage("record ID overflows u64".into()))?;
    let lowered = lowered_records(&sources, first_new_id)?;
    let local_plan = storage::plan_records(
        &lowered,
        target::MACOS_ARM64_METAL_APPLE8_V1,
        SHARD_LIMIT,
    )?;

    let existing_shards = existing_shard_count(&package);
    validate_existing_shards(&package, existing_shards)?;
    let shifted_new = local_plan
        .records
        .iter()
        .cloned()
        .map(|mut record| {
            record.shard_id = record
                .shard_id
                .checked_add(existing_shards)
                .ok_or_else(|| ColicError::Usage("shard ID overflows u32".into()))?;
            Ok(record)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut temp_shards = Vec::with_capacity(local_plan.shards as usize);
    let mut final_shards = Vec::with_capacity(local_plan.shards as usize);
    let pid = std::process::id();
    for local in 0..local_plan.shards {
        let shard = existing_shards
            .checked_add(local)
            .ok_or_else(|| ColicError::Usage("shard ID overflows u32".into()))?;
        temp_shards.push(package_root.join(format!("data-{shard:05}.coli.mtp-next-{pid}")));
        final_shards.push(package_root.join(format!("data-{shard:05}.coli")));
    }
    if temp_shards.iter().chain(&final_shards).any(|path| path.exists()) {
        return Err(ColicError::Usage(
            "MTP append target/temp shard path already exists; refusing overwrite".into(),
        ));
    }

    let result = write_attachment(
        package_root,
        drafter_root,
        &package,
        &inventory.source_fingerprint,
        &mtp,
        virtual_layer_base,
        existing_shards,
        &sources,
        &shifted_new,
        &temp_shards,
        &final_shards,
    );
    if result.is_err() {
        for path in temp_shards.iter().chain(&final_shards) {
            let _ = fs::remove_file(path);
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn write_attachment(
    package_root: &Path,
    drafter_root: &Path,
    package: &Package,
    drafter_fingerprint: &str,
    mtp: &QwenMtpInventory,
    virtual_layer_base: u32,
    existing_shards: u32,
    sources: &[NewSource],
    shifted_new: &[PlannedRecord],
    temp_shards: &[PathBuf],
    final_shards: &[PathBuf],
) -> Result<AttachMtpSummary> {
    let fingerprint = *package.fingerprint();
    let mut writers = temp_shards
        .iter()
        .enumerate()
        .map(|(local, path)| {
            storage::DataShardWriter::create(
                path,
                existing_shards + local as u32,
                package.alignment(),
                fingerprint,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let mut new_metadata = Vec::with_capacity(sources.len());
    for (source, planned) in sources.iter().zip(shifted_new) {
        let writer_index = usize::try_from(planned.shard_id - existing_shards)
            .map_err(|_| ColicError::Usage("MTP writer index exceeds usize".into()))?;
        let writer = writers
            .get_mut(writer_index)
            .ok_or_else(|| ColicError::Usage("MTP planned record references missing writer".into()))?;
        new_metadata.push(write_new_record(writer, planned, source)?);
    }

    let mut new_header_crcs = Vec::with_capacity(writers.len());
    for (writer, (temp, final_path)) in writers
        .into_iter()
        .zip(temp_shards.iter().zip(final_shards))
    {
        writer.finish()?;
        fs::rename(temp, final_path).map_err(|source| ColicError::Io {
            path: final_path.clone(),
            source,
        })?;
        new_header_crcs.push(read_shard_header_crc(final_path)?);
    }

    let mut combined_records = Vec::with_capacity(package.records().len() + shifted_new.len());
    let mut combined_metadata = Vec::with_capacity(package.records().len() + new_metadata.len());
    for record in package.records() {
        combined_records.push(PlannedRecord {
            record: LoweredRecord {
                id: record.id,
                kind: record.kind,
                stored_bytes: record.stored,
                decoded_bytes: record.decoded,
            },
            shard_id: record.shard_id,
            payload_offset: record.offset,
        });
        combined_metadata.push(ManifestRecord {
            id: record.id,
            name: record.name.clone(),
            layer: record.layer,
            expert: record.expert,
            kind: record.kind,
            codec: record.codec,
            math_format: record.math_format,
            scale_format: record.scale_format,
            layout: record.layout,
            flags: record.flags,
            stored_crc32c: record.stored_crc,
            logical_crc32c: record.logical_crc,
            codec_table_id: 0,
        });
    }
    combined_records.extend_from_slice(shifted_new);
    combined_metadata.extend(new_metadata);

    let shard_count = existing_shards
        .checked_add(temp_shards.len() as u32)
        .ok_or_else(|| ColicError::Usage("combined shard count overflows u32".into()))?;
    let combined_plan = StoragePlan {
        record_alignment: package.alignment(),
        shard_size_limit: SHARD_LIMIT,
        shards: shard_count,
        records: combined_records,
        projected_stored_bytes: package
            .records()
            .iter()
            .map(|record| record.stored)
            .chain(shifted_new.iter().map(|record| record.record.stored_bytes))
            .try_fold(0_u64, |sum, bytes| sum.checked_add(bytes))
            .ok_or_else(|| ColicError::Usage("combined stored-byte total overflows u64".into()))?,
        projected_padding_bytes: 0,
    };

    let mut shard_header_crcs = Vec::with_capacity(shard_count as usize);
    for shard in 0..existing_shards {
        let path = package
            .shard_path(shard)
            .ok_or_else(|| ColicError::Usage("existing package has missing shard path".into()))?;
        shard_header_crcs.push(read_shard_header_crc(Path::new(&path))?);
    }
    shard_header_crcs.extend(new_header_crcs);

    let manifest = storage::encode_manifest_with_records(
        &combined_plan,
        package.profile(),
        fingerprint,
        &combined_metadata,
        &shard_header_crcs,
    )?;
    let manifest_path = package_root.join("manifest.coli");
    let backup_manifest = package_root.join(format!("manifest.coli.pre-mtp-{pid}", pid = std::process::id()));
    fs::copy(&manifest_path, &backup_manifest).map_err(|source| ColicError::Io {
        path: backup_manifest.clone(),
        source,
    })?;

    let metadata = json!({
        "version": 1,
        "kind": "qwen4_exp_mtp",
        "embedded": true,
        "source": drafter_root.to_string_lossy(),
        "source_fingerprint": drafter_fingerprint,
        "target_fingerprint": hex_fingerprint(package.fingerprint()),
        "virtual_layer_base": virtual_layer_base,
        "stages": mtp.hidden_layers,
        "experts": mtp.experts,
        "experts_per_stage": mtp.experts,
        "hidden_size": mtp.hidden_size,
        "moe_intermediate_size": mtp.moe_intermediate_size,
        "expert_representation": "apple8-mxfp4",
        "static_representation": "bf16-exact",
        "record_name_prefix": "mtp.",
        "shard_first": existing_shards,
        "shard_count": temp_shards.len(),
    });
    let metadata_bytes = serde_json::to_vec_pretty(&metadata)
        .map_err(|error| ColicError::Usage(format!("failed to encode MTP metadata: {error}")))?;
    let metadata_next = package_root.join(format!("{MTP_METADATA_FILE}.mtp-next-{}", std::process::id()));
    fs::write(&metadata_next, &metadata_bytes).map_err(|source| ColicError::Io {
        path: metadata_next.clone(),
        source,
    })?;
    fs::rename(&metadata_next, package_root.join(MTP_METADATA_FILE)).map_err(|source| {
        ColicError::Io {
            path: package_root.join(MTP_METADATA_FILE),
            source,
        }
    })?;

    let manifest_next = package_root.join(format!("manifest.coli.mtp-next-{}", std::process::id()));
    fs::write(&manifest_next, manifest).map_err(|source| ColicError::Io {
        path: manifest_next.clone(),
        source,
    })?;
    fs::rename(&manifest_next, &manifest_path).map_err(|source| ColicError::Io {
        path: manifest_path.clone(),
        source,
    })?;

    if let Err(error) = post_attach_verify(package_root, mtp, virtual_layer_base) {
        let _ = fs::copy(&backup_manifest, &manifest_path);
        let _ = fs::remove_file(package_root.join(MTP_METADATA_FILE));
        for path in final_shards {
            let _ = fs::remove_file(path);
        }
        return Err(error);
    }

    Ok(AttachMtpSummary {
        package: package_root.to_owned(),
        drafter: drafter_root.to_owned(),
        static_tensors: sources
            .iter()
            .filter(|source| matches!(source, NewSource::Tensor { .. }))
            .count(),
        experts: mtp.experts,
        stages: mtp.hidden_layers,
        virtual_layer_base,
        new_shards: temp_shards.len() as u32,
        new_records: sources.len(),
        added_stored_bytes: shifted_new.iter().map(|r| r.record.stored_bytes).sum(),
        backup_manifest,
    })
}

fn build_sources(mtp: &QwenMtpInventory, virtual_layer_base: u32) -> Result<Vec<NewSource>> {
    let mut sources = Vec::new();
    for (name, tensor) in &mtp.global_tensors {
        require_bf16(tensor, name)?;
        sources.push(NewSource::Tensor {
            name: format!("mtp.{name}"),
            layer: -3,
            tensor: tensor.clone(),
        });
    }
    for stage in &mtp.stages {
        let virtual_layer = virtual_layer_base
            .checked_add(stage.stage)
            .ok_or_else(|| ColicError::Usage("MTP virtual layer overflows u32".into()))?;
        for (role, tensor) in &stage.static_tensors {
            require_bf16(tensor, role)?;
            sources.push(NewSource::Tensor {
                name: format!("mtp.layers.{}.{role}", stage.stage),
                layer: i32::try_from(virtual_layer)
                    .map_err(|_| ColicError::Usage("MTP virtual layer exceeds i32".into()))?,
                tensor: tensor.clone(),
            });
        }
        for expert in 0..mtp.experts {
            let routed = routed_expert(stage, expert, virtual_layer, mtp)?;
            sources.push(NewSource::Expert {
                name: format!("mtp.layers.{}.ffn.experts.{expert}", stage.stage),
                expert: routed,
            });
        }
    }
    Ok(sources)
}

fn routed_expert(
    stage: &qwen_mtp::QwenMtpStageInventory,
    expert: u32,
    virtual_layer: u32,
    mtp: &QwenMtpInventory,
) -> Result<RoutedExpert> {
    let h = mtp.hidden_size;
    let i = mtp.moe_intermediate_size;
    let (gate, up, down) = match &stage.expert_bank {
        QwenMtpExpertBank::SplitGateUp { gate, up, down } => (
            slice_bank(gate, mtp.experts, expert, i, h)?,
            slice_bank(up, mtp.experts, expert, i, h)?,
            slice_bank(down, mtp.experts, expert, h, i)?,
        ),
        QwenMtpExpertBank::FusedGateUp { gate_up, down } => (
            slice_bank_rows(gate_up, mtp.experts, expert, 2 * i, h, 0, i)?,
            slice_bank_rows(gate_up, mtp.experts, expert, 2 * i, h, i, i)?,
            slice_bank(down, mtp.experts, expert, h, i)?,
        ),
    };
    Ok(RoutedExpert {
        layer: virtual_layer,
        expert,
        gate,
        up,
        down,
    })
}

fn slice_bank(
    tensor: &TensorRef,
    experts: u32,
    expert: u32,
    rows: u32,
    columns: u32,
) -> Result<Matrix> {
    slice_bank_rows(tensor, experts, expert, rows, columns, 0, rows)
}

fn slice_bank_rows(
    tensor: &TensorRef,
    experts: u32,
    expert: u32,
    rows_per_expert: u32,
    columns: u32,
    first_row: u32,
    rows: u32,
) -> Result<Matrix> {
    require_bf16(tensor, "MTP expert bank")?;
    if tensor.shape != [
        u64::from(experts),
        u64::from(rows_per_expert),
        u64::from(columns),
    ] || expert >= experts
        || first_row.checked_add(rows).is_none_or(|end| end > rows_per_expert)
    {
        return Err(ColicError::InvalidSource {
            path: tensor.source.clone(),
            detail: format!(
                "invalid MTP expert-bank slice: shape={:?}, expert={expert}/{experts}, rows={first_row}+{rows}/{rows_per_expert}, columns={columns}",
                tensor.shape
            ),
        });
    }
    let row_bytes = u64::from(columns)
        .checked_mul(2)
        .ok_or_else(|| ColicError::Usage("MTP expert row size overflows u64".into()))?;
    let row_index = u64::from(expert)
        .checked_mul(u64::from(rows_per_expert))
        .and_then(|value| value.checked_add(u64::from(first_row)))
        .ok_or_else(|| ColicError::Usage("MTP expert offset overflows u64".into()))?;
    let offset = tensor
        .offset
        .checked_add(
            row_index
                .checked_mul(row_bytes)
                .ok_or_else(|| ColicError::Usage("MTP expert offset overflows u64".into()))?,
        )
        .ok_or_else(|| ColicError::Usage("MTP expert offset overflows u64".into()))?;
    let len = u64::from(rows)
        .checked_mul(row_bytes)
        .ok_or_else(|| ColicError::Usage("MTP expert slice size overflows u64".into()))?;
    Ok(Matrix {
        source: TensorRef {
            source: tensor.source.clone(),
            offset,
            len,
            dtype: "BF16".into(),
            shape: vec![u64::from(rows), u64::from(columns)],
        },
        rows,
        columns,
        scale: None,
    })
}

fn lowered_records(sources: &[NewSource], first_id: u64) -> Result<Vec<LoweredRecord>> {
    sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let id = first_id
                .checked_add(index as u64)
                .ok_or_else(|| ColicError::Usage("record ID overflows u64".into()))?;
            match source {
                NewSource::Tensor { tensor, .. } => Ok(LoweredRecord {
                    id,
                    kind: 1,
                    stored_bytes: target::exact_tensor_stored_bytes(tensor)?,
                    decoded_bytes: tensor.len,
                }),
                NewSource::Expert { expert, .. } => {
                    target::validate_apple8_quantized_mxfp4_expert(expert)?;
                    Ok(LoweredRecord {
                        id,
                        kind: 2,
                        stored_bytes: target::apple8_expert_stored_bytes(expert)?,
                        decoded_bytes: target::apple8_expert_decoded_bytes(expert)?,
                    })
                }
            }
        })
        .collect()
}

fn write_new_record(
    writer: &mut storage::DataShardWriter,
    planned: &PlannedRecord,
    source: &NewSource,
) -> Result<ManifestRecord> {
    match source {
        NewSource::Tensor {
            name,
            layer,
            tensor,
        } => {
            let mut checksums = (0_u32, 0_u32);
            writer.write_record_stream(planned, |file| {
                checksums = target::stream_exact_tensor(tensor, file)?;
                Ok(planned.record.stored_bytes)
            })?;
            Ok(ManifestRecord {
                id: planned.record.id,
                name: Some(name.clone()),
                layer: *layer,
                expert: -1,
                kind: 1,
                codec: 0,
                math_format: target::math_format_for_dtype(&tensor.dtype)?,
                scale_format: 0,
                layout: 0,
                flags: 0b10,
                stored_crc32c: checksums.1,
                logical_crc32c: checksums.0,
                codec_table_id: 0,
            })
        }
        NewSource::Expert { name, expert } => {
            let bytes = target::lower_apple8_quantized_mxfp4_expert(expert)?;
            if bytes.len() as u64 != planned.record.stored_bytes {
                return Err(ColicError::Usage(
                    "MTP Apple8 expert lowerer disagrees with storage plan".into(),
                ));
            }
            let crc = storage::crc32c(&bytes);
            writer.write_record(planned, &bytes)?;
            Ok(ManifestRecord {
                id: planned.record.id,
                name: Some(name.clone()),
                layer: i32::try_from(expert.layer)
                    .map_err(|_| ColicError::Usage("MTP expert layer exceeds i32".into()))?,
                expert: i32::try_from(expert.expert)
                    .map_err(|_| ColicError::Usage("MTP expert ID exceeds i32".into()))?,
                kind: 2,
                codec: 0,
                math_format: 0xfffe,
                scale_format: 0xfffe,
                layout: 0xfffe,
                flags: 0,
                stored_crc32c: crc,
                logical_crc32c: 0,
                codec_table_id: 0,
            })
        }
    }
}

fn ensure_unique_names(package: &Package, sources: &[NewSource], root: &Path) -> Result<()> {
    let mut names = std::collections::BTreeSet::new();
    for source in sources {
        let name = match source {
            NewSource::Tensor { name, .. } | NewSource::Expert { name, .. } => name,
        };
        if !names.insert(name.as_str()) || package.record_by_name(name).is_some() {
            return Err(ColicError::InvalidSource {
                path: root.to_owned(),
                detail: format!("MTP record name collision `{name}`"),
            });
        }
    }
    Ok(())
}

fn existing_shard_count(package: &Package) -> u32 {
    package
        .records()
        .iter()
        .map(|record| record.shard_id)
        .max()
        .map_or(0, |id| id + 1)
}

fn validate_existing_shards(package: &Package, shards: u32) -> Result<()> {
    let first_offset = storage::align_up(storage::DATA_SHARD_HEADER_BYTES, package.alignment())?;
    for shard in 0..shards {
        let path = package
            .shard_path(shard)
            .ok_or_else(|| ColicError::Usage(format!("missing existing shard {shard}")))?;
        let expected = package
            .records()
            .iter()
            .filter(|record| record.shard_id == shard)
            .map(|record| record.offset + record.stored)
            .max()
            .unwrap_or(first_offset);
        let actual = fs::metadata(&path)
            .map_err(|source| ColicError::Io {
                path: PathBuf::from(&path),
                source,
            })?
            .len();
        if expected != actual {
            return unsupported(
                Path::new(&path),
                format!(
                    "existing shard has trailing/unindexed bytes ({actual} != manifest-derived {expected}); append re-encoding would not preserve it"
                ),
            );
        }
    }
    Ok(())
}

fn post_attach_verify(root: &Path, mtp: &QwenMtpInventory, virtual_layer_base: u32) -> Result<()> {
    let package = open_package(root)?;
    for required in [
        "mtp.fc_embedding.weight",
        "mtp.fc_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ] {
        let record = package.record_by_name(required).ok_or_else(|| ColicError::InvalidSource {
            path: root.to_owned(),
            detail: format!("attached package is missing `{required}`"),
        })?;
        let _ = package.read_tensor_payload(record).map_err(|error| ColicError::InvalidSource {
            path: root.to_owned(),
            detail: format!("attached MTP tensor `{required}` failed CRC/read validation: {error}"),
        })?;
    }
    for stage in 0..mtp.hidden_layers {
        let layer = virtual_layer_base + stage;
        let records = package.expert_records(layer as i32, 0);
        if records.len() != 1 {
            return Err(ColicError::InvalidSource {
                path: root.to_owned(),
                detail: format!("MTP virtual layer {layer} expert 0 has {} records, expected 1", records.len()),
            });
        }
        let _ = package.read_record(records[0]).map_err(|error| ColicError::InvalidSource {
            path: root.to_owned(),
            detail: format!("MTP expert {layer}/0 failed CRC validation: {error}"),
        })?;
        if package.expert_records(layer as i32, (mtp.experts - 1) as i32).len() != 1 {
            return Err(ColicError::InvalidSource {
                path: root.to_owned(),
                detail: format!("MTP virtual layer {layer} is missing its final expert"),
            });
        }
    }
    Ok(())
}

fn read_shard_header_crc(path: &Path) -> Result<u32> {
    let mut file = File::open(path).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    let mut header = [0_u8; 76];
    file.read_exact(&mut header).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    if &header[..8] != storage::DATA_MAGIC {
        return Err(ColicError::InvalidSource {
            path: path.to_owned(),
            detail: "data shard header magic is invalid".into(),
        });
    }
    Ok(u32::from_le_bytes(header[72..76].try_into().unwrap()))
}

fn require_bf16(tensor: &TensorRef, role: &str) -> Result<()> {
    if tensor.dtype != "BF16" {
        return Err(ColicError::InvalidSource {
            path: tensor.source.clone(),
            detail: format!("MTP tensor `{role}` is {}, expected BF16", tensor.dtype),
        });
    }
    Ok(())
}

fn open_package(root: &Path) -> Result<Package> {
    Package::open(root).map_err(|error| ColicError::InvalidSource {
        path: root.to_owned(),
        detail: format!("invalid COLI package: {error}"),
    })
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|error| ColicError::InvalidSource {
        path: path.to_owned(),
        detail: format!("invalid JSON: {error}"),
    })
}

fn required_config_u32(root: &Path, config: &Value, key: &str) -> Result<u32> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value != 0)
        .ok_or_else(|| ColicError::InvalidSource {
            path: root.to_owned(),
            detail: format!("target config is missing positive `{key}`"),
        })
}

fn hex_fingerprint(fingerprint: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in fingerprint {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn unsupported<T>(path: &Path, detail: impl Into<String>) -> Result<T> {
    Err(ColicError::InvalidSource {
        path: path.to_owned(),
        detail: detail.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_bank_slice_preserves_row_major_expert_geometry() {
        let tensor = TensorRef {
            source: PathBuf::from("weights.safetensors"),
            offset: 128,
            len: 2 * 3 * 5 * 7,
            dtype: "BF16".into(),
            shape: vec![3, 5, 7],
        };
        let matrix = slice_bank_rows(&tensor, 3, 2, 5, 7, 1, 2).unwrap();
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.columns, 7);
        assert_eq!(matrix.source.shape, vec![2, 7]);
        assert_eq!(matrix.source.offset, 128 + ((2 * 5 + 1) * 7 * 2));
        assert_eq!(matrix.source.len, 2 * 7 * 2);
    }
}
