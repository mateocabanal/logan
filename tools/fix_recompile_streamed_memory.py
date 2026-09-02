from pathlib import Path

path = Path("logan-compiler/src/recompile.rs")
text = path.read_text()

anchor = '''fn build_recompile_plans(
    package: &Package,
'''
helper = '''fn fixed_resident_bytes(record: &RecordInfo) -> u64 {
    // COLI v1 does not yet persist placement for tensor records. Qwen4 PLE
    // n-gram shards are explicitly row-streamed by the runtime and must not
    // be charged as permanently resident just because their decoded payload
    // is large. Keep this classification narrow until the package-local
    // execution plan (#83) becomes the authoritative placement source.
    if record.kind == 1
        && record.name.as_deref().is_some_and(|name| {
            name.contains("ple.ple_embedding.ngram_embedding.shard_")
        })
    {
        0
    } else {
        record.decoded
    }
}

'''
if anchor not in text:
    raise SystemExit("build_recompile_plans anchor not found")
text = text.replace(anchor, helper + anchor, 1)

old = '''            base_resident_bytes = base_resident_bytes
                .checked_add(record.decoded)
                .ok_or_else(|| ColicError::Usage("recompile resident bytes overflow".into()))?;
'''
new = '''            base_resident_bytes = base_resident_bytes
                .checked_add(fixed_resident_bytes(record))
                .ok_or_else(|| ColicError::Usage("recompile resident bytes overflow".into()))?;
'''
if old not in text:
    raise SystemExit("resident accounting block not found")
text = text.replace(old, new, 1)

marker = '''    #[test]
    fn quant_rule_parser_rejects_reversed_and_unknown_selectors() {
'''
test = '''    #[test]
    fn streamed_ple_ngram_shards_are_not_fixed_resident() {
        let streamed = RecordInfo {
            id: 1,
            kind: 1,
            codec: 0,
            math_format: 0,
            scale_format: 0,
            layout: 0,
            flags: 0,
            shard_id: 0,
            name: Some(
                "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_17"
                    .into(),
            ),
            layer: 0,
            expert: -1,
            offset: 0,
            stored: 400_000_000,
            decoded: 400_000_000,
            stored_crc: 0,
            logical_crc: 0,
        };
        assert_eq!(fixed_resident_bytes(&streamed), 0);

        let mut dense = streamed.clone();
        dense.name = Some("model.language_model.layers.0.self_attn.q_proj.weight".into());
        dense.decoded = 123_456;
        assert_eq!(fixed_resident_bytes(&dense), 123_456);

        let mut ngram_scale = streamed;
        ngram_scale.name = Some(
            "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.weight_scale"
                .into(),
        );
        ngram_scale.decoded = 2;
        assert_eq!(fixed_resident_bytes(&ngram_scale), 2);
    }

'''
if marker not in text:
    raise SystemExit("test insertion marker not found")
text = text.replace(marker, test + marker, 1)
path.write_text(text)
