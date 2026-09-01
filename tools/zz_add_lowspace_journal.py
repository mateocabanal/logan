from pathlib import Path

p = Path('logan-compiler/src/recompile.rs')
s = p.read_text()

# Constants for the bounded on-disk transaction state.
anchor = 'const SCALE_E8M0: u16 = 0x0004;\n'
insert = anchor + '''const INPLACE_STATE_FILE: &str = ".logan-recompile-state.json";\nconst INPLACE_RECORD_META_FILE: &str = ".logan-recompile-record.json";\nconst INPLACE_RECORD_BACKUP_FILE: &str = ".logan-recompile-record.bin";\nconst INPLACE_MANIFEST_BACKUP_FILE: &str = ".logan-recompile-manifest.bak";\nconst INPLACE_FINAL_MANIFEST_BACKUP_FILE: &str = ".logan-recompile-final-manifest.bak";\nconst INPLACE_HEADERS_BACKUP_FILE: &str = ".logan-recompile-headers.bak";\n'''
if anchor not in s:
    raise SystemExit('constant anchor missing')
s = s.replace(anchor, insert, 1)

# Recover an interrupted bounded transaction before Package::open validates the
# possibly-hybrid package.
old = '''pub fn recompile(request: &RecompileRequest) -> Result<RecompileSummary> {\n    if request.source == request.output && !request.force {\n        return Err(ColicError::Usage(\n            "recompiling in place requires --force; using a separate output path is safer".into(),\n        ));\n    }\n\n    let package = Package::open(&request.source)?;\n'''
new = '''pub fn recompile(request: &RecompileRequest) -> Result<RecompileSummary> {\n    if request.source == request.output {\n        if !request.force {\n            return Err(ColicError::Usage(\n                "recompiling in place requires --force; using a separate output path is safer".into(),\n            ));\n        }\n        recover_in_place_if_needed(request)?;\n    }\n\n    let package = Package::open(&request.source)?;\n'''
if old not in s:
    raise SystemExit('recompile entry anchor missing')
s = s.replace(old, new, 1)

old = '''        let (plan, shard_sizes) = plan_in_place(&package, &actions, target)?;\n        write_package_in_place(\n            &package,\n            &actions,\n            &plan,\n            &shard_sizes,\n            target,\n            fingerprint,\n            request,\n        )?;\n        crate::verify::verify_package(&request.output)?;\n        crate::verify_target::verify_target_layouts(&request.output)?;\n'''
new = '''        let (plan, shard_sizes) = plan_in_place(&package, &actions, target)?;\n        ensure_in_place_state(\n            request,\n            package.profile(),\n            target.name,\n            plan.shards,\n        )?;\n        write_package_in_place(\n            &package,\n            &actions,\n            &plan,\n            &shard_sizes,\n            target,\n            fingerprint,\n            request,\n        )?;\n'''
if old not in s:
    raise SystemExit('in-place call anchor missing')
s = s.replace(old, new, 1)

# Replace the destructive writer with a per-record checkpointed writer and
# bounded recovery helpers.
start = s.index('fn write_package_in_place(\n')
end = s.index('\nfn copy_manifest_record(', start)
replacement = r'''fn write_package_in_place(
    package: &Package,
    actions: &[Action],
    plan: &StoragePlan,
    shard_sizes: &[u64],
    target: TargetProfile,
    fingerprint: [u8; 32],
    request: &RecompileRequest,
) -> Result<()> {
    if actions.len() != plan.records.len() {
        return Err(ColicError::Usage(
            "in-place action/plan order mismatch".into(),
        ));
    }

    let source_alignment = u64::from(u32_at(package.manifest_ref(), 24)?);
    let source_header_crcs = read_shard_header_crcs(package, plan.shards)?;
    let source_rans_table = source_rans_table(package)?;
    let mut checkpoint_plan = checkpoint_plan_from_package(package, shard_sizes)?;
    let mut metadata = actions
        .iter()
        .map(copy_manifest_record)
        .collect::<Vec<_>>();

    for (index, (action, planned)) in actions.iter().zip(&plan.records).enumerate() {
        if planned.shard_id != action.source.shard_id
            || planned.payload_offset != action.source.offset
        {
            return Err(ColicError::Usage(
                "low-space in-place plan attempted to relocate a record".into(),
            ));
        }
        let ActionKind::Rewrite {
            descs,
            target: expert_target,
            ..
        } = &action.kind
        else {
            continue;
        };

        // Build the complete replacement from the still-valid source record
        // before touching disk. Peak scratch is one expert payload in memory.
        let packed = [
            matrix_to_mxfp4(package, &action.source, descs[0], request.allow_requantize)?,
            matrix_to_mxfp4(package, &action.source, descs[1], request.allow_requantize)?,
            matrix_to_mxfp4(package, &action.source, descs[2], request.allow_requantize)?,
        ];
        let payload = match expert_target {
            ExpertTarget::Apple8Mxfp4 => build_apple8_expert(
                action.source.layer,
                action.source.expert,
                [&packed[0], &packed[1], &packed[2]],
            )?,
            ExpertTarget::CanonicalMxfp4 => build_canonical_mxfp4_expert(
                action.source.layer,
                action.source.expert,
                [&packed[0], &packed[1], &packed[2]],
            )?,
        };
        if payload.len() as u64 != planned.record.stored_bytes {
            return Err(ColicError::Usage(format!(
                "rewritten expert {}:{} produced {} bytes, planned {}",
                action.source.layer,
                action.source.expert,
                payload.len(),
                planned.record.stored_bytes
            )));
        }

        // Persist only the current source record plus the current manifest.
        // If the process dies anywhere before the new checkpoint manifest is
        // committed, the next `logan recompile --in-place` restores these few
        // bytes and resumes from the last structurally-valid checkpoint.
        begin_record_journal(&request.source, package, &action.source)?;
        let mutation = (|| -> Result<()> {
            write_record_payload(
                &request.source,
                planned.shard_id,
                planned.payload_offset,
                &payload,
            )?;
            checkpoint_plan.records[index].record = planned.record.clone();
            metadata[index] = rewritten_manifest_record(action, &payload);
            write_checkpoint_manifest(
                package,
                &checkpoint_plan,
                shard_sizes,
                source_alignment,
                &metadata,
                &source_header_crcs,
                source_rans_table.as_ref(),
                fingerprint,
                &request.source,
            )
        })();
        if let Err(error) = mutation {
            if let Err(recovery) = recover_record_journal(&request.source) {
                return Err(ColicError::Usage(format!(
                    "in-place rewrite failed: {error}; record rollback also failed: {recovery}"
                )));
            }
            return Err(error);
        }
        clear_record_journal(&request.source)?;
    }

    // Final target headers and profile are a tiny second transaction. Back up
    // only the current manifest and 128-byte shard headers so a crash while
    // flipping the package to the new target can roll back to the last hybrid
    // checkpoint without a model-sized copy.
    begin_finalization_journal(&request.source, plan.shards)?;
    set_in_place_phase(&request.source, "finalizing")?;
    let finalization = (|| -> Result<()> {
        let mut header_crcs = Vec::with_capacity(plan.shards as usize);
        for shard_id in 0..plan.shards {
            let file_bytes = *shard_sizes
                .get(shard_id as usize)
                .ok_or_else(|| ColicError::Usage("missing source shard size".into()))?;
            let header = storage::encode_data_shard_header(
                shard_id,
                file_bytes,
                plan.record_alignment,
                fingerprint,
            )?;
            write_shard_header(&request.source, shard_id, &header)?;
            header_crcs.push(u32::from_le_bytes(header[72..76].try_into().unwrap()));
        }

        let mut manifest = storage::encode_manifest_with_records(
            plan,
            target.name,
            fingerprint,
            &metadata,
            &header_crcs,
        )?;
        if actions.iter().any(|action| action.keeps_rans_table) {
            let table = rans256::table_from_manifest(
                package.manifest_ref(),
                RANS_TABLE_ID,
                RANS_CODEC_ID,
            )?;
            manifest = rans256::manifest_table_region(manifest, Some(&table))?;
        }
        patch_manifest_shard_sizes(&mut manifest, shard_sizes)?;
        replace_manifest(&request.source, &manifest)?;
        crate::verify::verify_package(&request.output)?;
        crate::verify_target::verify_target_layouts(&request.output)?;
        write_provenance(package, target, actions, request, &request.output)
    })();
    if let Err(error) = finalization {
        if let Err(recovery) = rollback_finalization(&request.source) {
            return Err(ColicError::Usage(format!(
                "in-place finalization failed: {error}; finalization rollback also failed: {recovery}"
            )));
        }
        return Err(error);
    }

    set_in_place_phase(&request.source, "committed")?;
    clear_finalization_journal(&request.source)?;
    clear_record_journal(&request.source)?;
    remove_if_exists(&request.source.join(INPLACE_STATE_FILE))?;
    Ok(())
}

fn request_signature(request: &RecompileRequest) -> String {
    serde_json::to_string(&json!({
        "target": request.target,
        "quant": request.quant.as_str(),
        "quant_rules": request.quant_rules.iter().map(|rule| rule.as_spec()).collect::<Vec<_>>(),
        "codec": request.codec.as_str(),
        "allow_requantize": request.allow_requantize,
        "repack": request.repack,
    }))
    .expect("recompile request signature is always JSON-serializable")
}

fn ensure_in_place_state(
    request: &RecompileRequest,
    source_profile: &str,
    target_profile: &str,
    shards: u32,
) -> Result<()> {
    let path = request.source.join(INPLACE_STATE_FILE);
    if path.exists() {
        let state = read_json_file(&path)?;
        ensure_state_signature(&state, request)?;
        return Ok(());
    }
    let state = json!({
        "version": 1,
        "phase": "records",
        "request": request_signature(request),
        "source_profile": source_profile,
        "target_profile": target_profile,
        "shards": shards,
    });
    write_json_synced(&path, &state)
}

fn recover_in_place_if_needed(request: &RecompileRequest) -> Result<()> {
    let state_path = request.source.join(INPLACE_STATE_FILE);
    if !state_path.exists() {
        // A crash before the record-journal marker is durable can leave only
        // orphan backup files. No model bytes were touched in that state.
        clear_record_journal(&request.source)?;
        clear_finalization_journal(&request.source)?;
        return Ok(());
    }
    let mut state = read_json_file(&state_path)?;
    let phase = state
        .get("phase")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ColicError::Usage("in-place recompile state has no phase".into()))?
        .to_owned();

    match phase.as_str() {
        "records" => recover_record_journal(&request.source)?,
        "finalizing" => {
            recover_record_journal(&request.source)?;
            rollback_finalization(&request.source)?;
            state = read_json_file(&state_path)?;
        }
        "committed" => {
            crate::verify::verify_package(&request.source)?;
            crate::verify_target::verify_target_layouts(&request.source)?;
            clear_record_journal(&request.source)?;
            clear_finalization_journal(&request.source)?;
            remove_if_exists(&state_path)?;
            return Ok(());
        }
        other => {
            return Err(ColicError::Usage(format!(
                "unknown in-place recompile recovery phase `{other}`"
            )));
        }
    }
    ensure_state_signature(&state, request)
}

fn ensure_state_signature(state: &serde_json::Value, request: &RecompileRequest) -> Result<()> {
    let stored = state
        .get("request")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ColicError::Usage("in-place recompile state has no request signature".into()))?;
    let current = request_signature(request);
    if stored != current {
        return Err(ColicError::Usage(
            "an unfinished low-space in-place recompile exists; rerun with the same target/quant/codec options to resume it"
                .into(),
        ));
    }
    Ok(())
}

fn set_in_place_phase(root: &Path, phase: &str) -> Result<()> {
    let path = root.join(INPLACE_STATE_FILE);
    let mut state = read_json_file(&path)?;
    state["phase"] = serde_json::Value::String(phase.to_owned());
    write_json_synced(&path, &state)
}

fn begin_record_journal(root: &Path, package: &Package, record: &RecordInfo) -> Result<()> {
    clear_record_journal(root)?;
    let payload = package.read_record(record)?;
    let manifest = fs::read(root.join("manifest.coli")).map_err(|source| ColicError::Io {
        path: root.join("manifest.coli"),
        source,
    })?;
    write_synced(&root.join(INPLACE_RECORD_BACKUP_FILE), &payload)?;
    write_synced(&root.join(INPLACE_MANIFEST_BACKUP_FILE), &manifest)?;
    // The metadata marker is durable last. Until it exists no data shard is
    // modified, so orphan backup files can simply be discarded on recovery.
    write_json_synced(
        &root.join(INPLACE_RECORD_META_FILE),
        &json!({
            "version": 1,
            "shard": record.shard_id,
            "offset": record.offset,
            "bytes": record.stored,
        }),
    )
}

fn recover_record_journal(root: &Path) -> Result<()> {
    let meta_path = root.join(INPLACE_RECORD_META_FILE);
    if !meta_path.exists() {
        remove_if_exists(&root.join(INPLACE_RECORD_BACKUP_FILE))?;
        remove_if_exists(&root.join(INPLACE_MANIFEST_BACKUP_FILE))?;
        return Ok(());
    }
    let meta = read_json_file(&meta_path)?;
    let shard = json_u64(&meta, "shard")?;
    let shard: u32 = shard
        .try_into()
        .map_err(|_| ColicError::Usage("record journal shard id exceeds u32".into()))?;
    let offset = json_u64(&meta, "offset")?;
    let bytes = json_u64(&meta, "bytes")?;
    let backup_path = root.join(INPLACE_RECORD_BACKUP_FILE);
    let backup = fs::read(&backup_path).map_err(|source| ColicError::Io {
        path: backup_path.clone(),
        source,
    })?;
    if backup.len() as u64 != bytes {
        return Err(ColicError::Usage(
            "record journal payload length does not match its metadata".into(),
        ));
    }
    write_record_payload(root, shard, offset, &backup)?;
    let manifest_backup_path = root.join(INPLACE_MANIFEST_BACKUP_FILE);
    let manifest = fs::read(&manifest_backup_path).map_err(|source| ColicError::Io {
        path: manifest_backup_path.clone(),
        source,
    })?;
    replace_manifest(root, &manifest)?;
    clear_record_journal(root)
}

fn clear_record_journal(root: &Path) -> Result<()> {
    remove_if_exists(&root.join(INPLACE_RECORD_META_FILE))?;
    remove_if_exists(&root.join(INPLACE_RECORD_BACKUP_FILE))?;
    remove_if_exists(&root.join(INPLACE_MANIFEST_BACKUP_FILE))
}

fn begin_finalization_journal(root: &Path, shards: u32) -> Result<()> {
    clear_finalization_journal(root)?;
    let manifest_path = root.join("manifest.coli");
    let manifest = fs::read(&manifest_path).map_err(|source| ColicError::Io {
        path: manifest_path,
        source,
    })?;
    write_synced(&root.join(INPLACE_FINAL_MANIFEST_BACKUP_FILE), &manifest)?;
    let mut headers = Vec::with_capacity(shards as usize * storage::DATA_SHARD_HEADER_BYTES as usize);
    for shard_id in 0..shards {
        let path = root.join(format!("data-{shard_id:05}.coli"));
        let mut header = vec![0_u8; storage::DATA_SHARD_HEADER_BYTES as usize];
        File::open(&path)
            .map_err(|source| ColicError::Io {
                path: path.clone(),
                source,
            })?
            .read_exact(&mut header)
            .map_err(|source| ColicError::Io { path, source })?;
        headers.extend_from_slice(&header);
    }
    write_synced(&root.join(INPLACE_HEADERS_BACKUP_FILE), &headers)
}

fn rollback_finalization(root: &Path) -> Result<()> {
    let state = read_json_file(&root.join(INPLACE_STATE_FILE))?;
    let shards: u32 = json_u64(&state, "shards")?
        .try_into()
        .map_err(|_| ColicError::Usage("in-place state shard count exceeds u32".into()))?;
    let headers_path = root.join(INPLACE_HEADERS_BACKUP_FILE);
    let headers = fs::read(&headers_path).map_err(|source| ColicError::Io {
        path: headers_path.clone(),
        source,
    })?;
    let header_bytes = storage::DATA_SHARD_HEADER_BYTES as usize;
    if headers.len() != shards as usize * header_bytes {
        return Err(ColicError::Usage(
            "finalization header backup has the wrong length".into(),
        ));
    }
    for shard_id in 0..shards {
        let start = shard_id as usize * header_bytes;
        let header: [u8; storage::DATA_SHARD_HEADER_BYTES as usize] = headers
            [start..start + header_bytes]
            .try_into()
            .unwrap();
        write_shard_header(root, shard_id, &header)?;
    }
    let manifest_path = root.join(INPLACE_FINAL_MANIFEST_BACKUP_FILE);
    let manifest = fs::read(&manifest_path).map_err(|source| ColicError::Io {
        path: manifest_path.clone(),
        source,
    })?;
    replace_manifest(root, &manifest)?;
    set_in_place_phase(root, "records")?;
    clear_finalization_journal(root)
}

fn clear_finalization_journal(root: &Path) -> Result<()> {
    remove_if_exists(&root.join(INPLACE_FINAL_MANIFEST_BACKUP_FILE))?;
    remove_if_exists(&root.join(INPLACE_HEADERS_BACKUP_FILE))
}

fn checkpoint_plan_from_package(package: &Package, shard_sizes: &[u64]) -> Result<StoragePlan> {
    let records = package
        .records()
        .iter()
        .map(|record| PlannedRecord {
            record: LoweredRecord {
                id: record.id,
                kind: record.kind,
                stored_bytes: record.stored,
                decoded_bytes: record.decoded,
            },
            shard_id: record.shard_id,
            payload_offset: record.offset,
        })
        .collect::<Vec<_>>();
    let projected_stored_bytes = records.iter().try_fold(0_u64, |total, record| {
        total
            .checked_add(record.record.stored_bytes)
            .ok_or_else(|| ColicError::Usage("checkpoint stored-byte total overflows u64".into()))
    })?;
    Ok(StoragePlan {
        record_alignment: u64::from(u32_at(package.manifest_ref(), 24)?),
        shard_size_limit: shard_sizes.iter().copied().max().unwrap_or(0),
        shards: u32_at(package.manifest_ref(), 40)?,
        records,
        projected_stored_bytes,
        projected_padding_bytes: 0,
    })
}

fn read_shard_header_crcs(package: &Package, shards: u32) -> Result<Vec<u32>> {
    let mut result = Vec::with_capacity(shards as usize);
    for shard_id in 0..shards {
        let path = PathBuf::from(
            package
                .shard_path(shard_id)
                .ok_or_else(|| ColicError::Usage("source shard id is invalid".into()))?,
        );
        let mut header = [0_u8; storage::DATA_SHARD_HEADER_BYTES as usize];
        File::open(&path)
            .map_err(|source| ColicError::Io {
                path: path.clone(),
                source,
            })?
            .read_exact(&mut header)
            .map_err(|source| ColicError::Io { path, source })?;
        result.push(u32::from_le_bytes(header[72..76].try_into().unwrap()));
    }
    Ok(result)
}

fn source_rans_table(package: &Package) -> Result<Option<RansTable>> {
    if u32_at(package.manifest_ref(), 160)? == 0 {
        return Ok(None);
    }
    Ok(Some(rans256::table_from_manifest(
        package.manifest_ref(),
        RANS_TABLE_ID,
        RANS_CODEC_ID,
    )?))
}

#[allow(clippy::too_many_arguments)]
fn write_checkpoint_manifest(
    package: &Package,
    plan: &StoragePlan,
    shard_sizes: &[u64],
    source_alignment: u64,
    metadata: &[ManifestRecord],
    header_crcs: &[u32],
    rans_table: Option<&RansTable>,
    fingerprint: [u8; 32],
    root: &Path,
) -> Result<()> {
    let mut checkpoint_plan = plan.clone();
    checkpoint_plan.record_alignment = source_alignment;
    let mut manifest = storage::encode_manifest_with_records(
        &checkpoint_plan,
        package.profile(),
        fingerprint,
        metadata,
        header_crcs,
    )?;
    if let Some(table) = rans_table {
        manifest = rans256::manifest_table_region(manifest, Some(table))?;
    }
    patch_manifest_shard_sizes(&mut manifest, shard_sizes)?;
    replace_manifest(root, &manifest)
}

fn write_record_payload(root: &Path, shard_id: u32, offset: u64, payload: &[u8]) -> Result<()> {
    let path = root.join(format!("data-{shard_id:05}.coli"));
    let mut shard = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|source| ColicError::Io {
            path: path.clone(),
            source,
        })?;
    shard.seek(SeekFrom::Start(offset)).map_err(|source| ColicError::Io {
        path: path.clone(),
        source,
    })?;
    shard.write_all(payload).map_err(|source| ColicError::Io {
        path: path.clone(),
        source,
    })?;
    shard.sync_data().map_err(|source| ColicError::Io { path, source })
}

fn write_shard_header(
    root: &Path,
    shard_id: u32,
    header: &[u8; storage::DATA_SHARD_HEADER_BYTES as usize],
) -> Result<()> {
    write_record_payload(root, shard_id, 0, header)
}

fn replace_manifest(root: &Path, manifest: &[u8]) -> Result<()> {
    let path = root.join("manifest.coli");
    let next = root.join("manifest.coli.recompile-next");
    write_synced(&next, manifest)?;
    #[cfg(not(windows))]
    fs::rename(&next, &path).map_err(|source| ColicError::Io {
        path: path.clone(),
        source,
    })?;
    #[cfg(windows)]
    {
        fs::copy(&next, &path).map_err(|source| ColicError::Io {
            path: path.clone(),
            source,
        })?;
        File::open(&path)
            .and_then(|file| file.sync_all())
            .map_err(|source| ColicError::Io {
                path: path.clone(),
                source,
            })?;
        remove_if_exists(&next)?;
    }
    Ok(())
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = File::create(path).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    file.write_all(bytes).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    file.sync_all().map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })
}

fn write_json_synced(path: &Path, value: &serde_json::Value) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| ColicError::Usage(format!("cannot encode recompile state: {error}")))?;
    write_synced(path, &bytes)
}

fn read_json_file(path: &Path) -> Result<serde_json::Value> {
    let bytes = fs::read(path).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    serde_json::from_slice(&bytes)
        .map_err(|error| ColicError::Usage(format!("invalid recompile state JSON: {error}")))
}

fn json_u64(value: &serde_json::Value, field: &str) -> Result<u64> {
    value
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| ColicError::Usage(format!("recompile state field `{field}` is invalid")))
}

fn remove_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(ColicError::Io {
            path: path.to_owned(),
            source,
        }),
    }
}
'''
s = s[:start] + replacement + s[end:]

# Extend the existing end-to-end test with explicit record-journal and
# finalization-journal crash recovery checks before the real conversion.
test_anchor = '''        crate::verify::verify_package(&root).unwrap();\n        crate::verify_target::verify_target_layouts(&root).unwrap();\n\n        let before = Package::open(&root).unwrap();\n'''
test_insert = '''        crate::verify::verify_package(&root).unwrap();\n        crate::verify_target::verify_target_layouts(&root).unwrap();\n\n        let mut request = RecompileRequest::new(root.clone(), root.clone());\n        request.target = target::LINUX_X86_64_AVX2_V1.name.into();\n        request.force = true;\n        ensure_in_place_state(\n            &request,\n            target::MACOS_ARM64_METAL_APPLE8_V1.name,\n            target::LINUX_X86_64_AVX2_V1.name,\n            1,\n        )\n        .unwrap();\n\n        // Simulate a crash after a record overwrite but before its checkpoint\n        // manifest is committed. Recovery restores exactly the old record and\n        // manifest using only one-record scratch.\n        {\n            let source = Package::open(&root).unwrap();\n            begin_record_journal(&root, &source, &source.records()[0]).unwrap();\n            write_record_payload(&root, 0, before_offset, &[0_u8; 32]).unwrap();\n        }\n        recover_in_place_if_needed(&request).unwrap();\n        crate::verify::verify_package(&root).unwrap();\n        crate::verify_target::verify_target_layouts(&root).unwrap();\n\n        // Simulate a crash in the tiny finalization transaction after shard\n        // headers start changing. The header+manifest backup rolls it back.\n        begin_finalization_journal(&root, 1).unwrap();\n        set_in_place_phase(&root, "finalizing").unwrap();\n        write_shard_header(&root, 0, &[0_u8; storage::DATA_SHARD_HEADER_BYTES as usize]).unwrap();\n        recover_in_place_if_needed(&request).unwrap();\n        crate::verify::verify_package(&root).unwrap();\n        crate::verify_target::verify_target_layouts(&root).unwrap();\n\n        let before = Package::open(&root).unwrap();\n'''
if test_anchor not in s:
    raise SystemExit('e2e insertion anchor missing')
s = s.replace(test_anchor, test_insert, 1)

# Reuse the request created above instead of shadowing it later.
old_req = '''        let mut request = RecompileRequest::new(root.clone(), root.clone());\n        request.target = target::LINUX_X86_64_AVX2_V1.name.into();\n        request.force = true;\n        let summary = recompile(&request).unwrap();\n'''
new_req = '''        let summary = recompile(&request).unwrap();\n'''
if old_req not in s:
    raise SystemExit('e2e request anchor missing')
s = s.replace(old_req, new_req, 1)

p.write_text(s)
Path('tools/zz_add_lowspace_journal.py').unlink(missing_ok=True)
Path('.github/workflows/zz-add-lowspace-journal.yml').unlink(missing_ok=True)
