use std::{collections::BTreeMap, fs, path::Path, time::{SystemTime, UNIX_EPOCH}};

use logan_compiler::pipeline::{
    compile, CompileRequest, NoProgress, QuantRequest, TargetRequest,
};
use logan_format::package::Package;

fn write_safetensors(root: &Path) {
    let mut specs = BTreeMap::<String, (&'static str, Vec<u64>)>::new();
    let mut add = |name: &str, dtype: &'static str, shape: &[u64]| {
        specs.insert(name.to_owned(), (dtype, shape.to_vec()));
    };

    add("model.language_model.embed_tokens.weight", "BF16", &[8, 32]);
    add("lm_head.weight", "BF16", &[8, 32]);
    add(
        "model.language_model.layers.0.mlp.experts.0.gate_up_proj",
        "BF16",
        &[16, 32],
    );
    add(
        "model.language_model.layers.0.mlp.experts.0.down_proj",
        "BF16",
        &[32, 8],
    );
    add("model.language_model.layers.0.mlp.gate.weight", "BF16", &[1, 32]);
    add(
        "model.language_model.layers.0.ple.key_proj.weight",
        "BF16",
        &[32, 32],
    );
    add(
        "model.language_model.layers.0.ple.ple_embedding.ngram_heads_vocab_sizes",
        "I64",
        &[2],
    );
    add(
        "model.language_model.layers.0.ple.ple_embedding.ngram_heads_offsets",
        "I64",
        &[2],
    );
    add(
        "model.language_model.layers.0.ple.ple_embedding.layer_multipliers",
        "I64",
        &[3],
    );
    add(
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.weight",
        "BF16",
        &[128, 8],
    );

    let mut offset = 0_u64;
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, (dtype, shape)) in specs {
        let element_bytes = match dtype {
            "BF16" => 2_u64,
            "I64" => 8_u64,
            _ => unreachable!(),
        };
        let bytes = shape.iter().product::<u64>() * element_bytes;
        header.insert(
            name,
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + bytes]
            }),
        );
        payload.resize(payload.len() + bytes as usize, 0);
        offset += bytes;
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&payload);
    fs::write(root.join("model.safetensors"), file).unwrap();
}

fn source_fixture(root: &Path) {
    fs::create_dir_all(root).unwrap();
    fs::write(
        root.join("config.json"),
        r#"{
          "architectures":["Qwen4ExpForCausalLM"],
          "model_type":"qwen4_exp_text",
          "num_hidden_layers":1,
          "layer_types":["linear_attention"],
          "hidden_size":32,
          "num_experts":1,
          "num_experts_per_tok":1,
          "moe_intermediate_size":8,
          "shared_expert_intermediate_size":8,
          "vocab_size":8,
          "num_attention_heads":1,
          "num_key_value_heads":1,
          "head_dim":32,
          "linear_num_key_heads":1,
          "linear_key_head_dim":8,
          "linear_num_value_heads":1,
          "linear_value_head_dim":8,
          "linear_conv_kernel_dim":4,
          "hc_count":1,
          "ple_layer_ids":[1],
          "split_ngram_parts":128
        }"#,
    )
    .unwrap();
    write_safetensors(root);
}

#[test]
fn bf16_qwen4_compiles_to_verified_apple8_with_mxfp4_expert_and_f8_ple() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "logan-qwen4-apple8-e2e-{}-{nonce}",
        std::process::id()
    ));
    let source = root.join("source");
    let output = root.join("qwen4-apple8.coli");
    source_fixture(&source);

    let mut request = CompileRequest::new(source);
    request.output = Some(output.clone());
    request.target = TargetRequest::Profile("macos-arm64-metal-apple8-v1".into());
    request.quant = QuantRequest::Profile("mxfp4".into());
    request.verify = true;
    compile(&request, &mut NoProgress).unwrap();

    let package = Package::open(&output).unwrap();
    assert_eq!(package.profile(), "macos-arm64-metal-apple8-v1");

    let expert = package.expert_records(0, 0);
    assert_eq!(expert.len(), 1);
    let raw = package.read_record(expert[0]).unwrap();
    assert_eq!(&raw[..8], b"COLIEXPT");
    let desc_size = u32::from_le_bytes(raw[28..32].try_into().unwrap()) as usize;
    for matrix in 0..3 {
        let desc = 64 + matrix * desc_size;
        let math = u16::from_le_bytes(raw[desc + 4..desc + 6].try_into().unwrap());
        let scale = u16::from_le_bytes(raw[desc + 6..desc + 8].try_into().unwrap());
        assert_eq!(math, 0x20, "matrix {matrix} is not MXFP4");
        assert_eq!(scale, 0x04, "matrix {matrix} does not use E8M0 scales");
    }

    let shard = package
        .record_by_name("layers.0.ple.ple_embedding.ngram_embedding.shard_000")
        .unwrap();
    assert_eq!(shard.math_format, 0x10);
    assert_eq!(shard.decoded, 8); // one [1,8] E4M3 row in this tiny split
    assert_eq!(package.read_tensor_payload(shard).unwrap(), vec![0_u8; 8]);

    let scale = package
        .record_by_name("layers.0.ple.ple_embedding.ngram_embedding.weight_scale")
        .unwrap();
    assert_eq!(scale.math_format, 3);
    assert_eq!(package.read_tensor_payload(scale).unwrap(), vec![0x80, 0x3f]); // BF16 1.0

    fs::remove_dir_all(root).unwrap();
}
