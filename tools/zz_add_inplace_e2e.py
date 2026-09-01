from pathlib import Path
p = Path('logan-compiler/src/recompile.rs')
s = p.read_text()
anchor = '''    #[test]\n    fn quant_rule_parser_rejects_reversed_and_unknown_selectors() {\n        assert!(QuantRule::parse("layer:9-3=mxfp4").is_err());\n        assert!(QuantRule::parse("dense=mxfp4").is_err());\n        assert!(QuantRule::parse("layer:2=bogus").is_err());\n    }\n'''
if anchor not in s:
    raise SystemExit('test anchor missing')
insert = r'''
    #[test]
    fn low_space_in_place_retarget_preserves_shard_size_and_record_offset() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "logan-recompile-in-place-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();

        let matrices = [packed(9, 32), packed(9, 32), packed(32, 32)];
        let payload = build_apple8_expert(
            0,
            0,
            [&matrices[0], &matrices[1], &matrices[2]],
        )
        .unwrap();
        let fingerprint = [0x5a; 32];
        let lowered = LoweredRecord {
            id: 1,
            kind: 2,
            stored_bytes: payload.len() as u64,
            decoded_bytes: u64::from_le_bytes(payload[48..56].try_into().unwrap()),
        };
        let source_plan = storage::plan_records(
            &[lowered],
            target::MACOS_ARM64_METAL_APPLE8_V1,
            64 * 1024,
        )
        .unwrap();
        let shard_path = root.join("data-00000.coli");
        let mut writer = storage::DataShardWriter::create(
            &shard_path,
            0,
            source_plan.record_alignment,
            fingerprint,
        )
        .unwrap();
        writer.write_record(&source_plan.records[0], &payload).unwrap();
        writer.finish().unwrap();
        let before_shard_bytes = fs::metadata(&shard_path).unwrap().len();
        let before_offset = source_plan.records[0].payload_offset;

        let mut header = [0_u8; storage::DATA_SHARD_HEADER_BYTES as usize];
        File::open(&shard_path)
            .unwrap()
            .read_exact(&mut header)
            .unwrap();
        let metadata = [ManifestRecord {
            id: 1,
            name: Some("layers.0.ffn.experts.0".into()),
            layer: 0,
            expert: 0,
            kind: 2,
            codec: 0,
            math_format: 0xfffe,
            scale_format: 0xfffe,
            layout: 0xfffe,
            flags: 0,
            stored_crc32c: storage::crc32c(&payload),
            logical_crc32c: 0,
            codec_table_id: 0,
        }];
        let manifest = storage::encode_manifest_with_records(
            &source_plan,
            target::MACOS_ARM64_METAL_APPLE8_V1.name,
            fingerprint,
            &metadata,
            &[u32::from_le_bytes(header[72..76].try_into().unwrap())],
        )
        .unwrap();
        fs::write(root.join("manifest.coli"), manifest).unwrap();
        crate::verify::verify_package(&root).unwrap();
        crate::verify_target::verify_target_layouts(&root).unwrap();

        let before = Package::open(&root).unwrap();
        assert_eq!(before.records()[0].offset, before_offset);
        let before_stored = before.records()[0].stored;

        let mut request = RecompileRequest::new(root.clone(), root.clone());
        request.target = target::LINUX_X86_64_AVX2_V1.name.into();
        request.force = true;
        let summary = recompile(&request).unwrap();
        assert_eq!(summary.rewritten_experts, 1);

        let after = Package::open(&root).unwrap();
        assert_eq!(after.profile(), target::LINUX_X86_64_AVX2_V1.name);
        assert_eq!(after.records()[0].shard_id, 0);
        assert_eq!(after.records()[0].offset, before_offset);
        assert!(after.records()[0].stored < before_stored);
        assert_eq!(fs::metadata(&shard_path).unwrap().len(), before_shard_bytes);
        crate::verify::verify_package(&root).unwrap();
        crate::verify_target::verify_target_layouts(&root).unwrap();

        fs::remove_dir_all(root).unwrap();
    }

'''
s = s.replace(anchor, insert + anchor, 1)
p.write_text(s)
Path('tools/zz_add_inplace_e2e.py').unlink(missing_ok=True)
Path('.github/workflows/zz-add-inplace-e2e.yml').unlink(missing_ok=True)
