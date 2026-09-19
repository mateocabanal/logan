use logan_llama::{DType, inspect_weights, load_config};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

fn temp_dir(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "logan-llama-{label}-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn config(dtype: &str, rope: &str) -> String {
    format!(
        r#"{{
      "model_type":"llama", "vocab_size":6, "hidden_size":4,
      "intermediate_size":8, "num_hidden_layers":1,
      "num_attention_heads":2, "num_key_value_heads":1, "head_dim":2,
      "max_position_embeddings":64, "rms_norm_eps":0.00001,
      "rope_theta":{rope}, "rope_parameters":{{"rope_theta":{rope}}},
      "bos_token_id":1, "eos_token_id":[1,130073],
      "tie_word_embeddings":false, "torch_dtype":"{dtype}"
    }}"#
    )
}

fn tensor_specs() -> Vec<(&'static str, Vec<u64>)> {
    vec![
        ("model.embed_tokens.weight", vec![6, 4]),
        ("model.layers.0.input_layernorm.weight", vec![4]),
        ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
        ("model.layers.0.self_attn.k_proj.weight", vec![2, 4]),
        ("model.layers.0.self_attn.v_proj.weight", vec![2, 4]),
        ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
        ("model.layers.0.post_attention_layernorm.weight", vec![4]),
        ("model.layers.0.mlp.gate_proj.weight", vec![8, 4]),
        ("model.layers.0.mlp.up_proj.weight", vec![8, 4]),
        ("model.layers.0.mlp.down_proj.weight", vec![4, 8]),
        ("model.norm.weight", vec![4]),
        ("lm_head.weight", vec![6, 4]),
    ]
}

fn write_shard(path: &Path, dtype: &str, specs: &[(&str, Vec<u64>)]) {
    let mut offset = 0_u64;
    let mut entries = Vec::new();
    for (name, shape) in specs {
        let elements = shape.iter().product::<u64>();
        let end = offset + elements * 2;
        entries.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":{:?},"data_offsets":[{offset},{end}]}}"#,
            shape
        ));
        offset = end;
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(bytes.len() + offset as usize, 0);
    fs::write(path, bytes).unwrap();
}

fn write_index(root: &Path, names: &[(&str, &str)]) {
    let weight_map = names
        .iter()
        .map(|(name, shard)| format!(r#""{name}":"{shard}""#))
        .collect::<Vec<_>>()
        .join(",");
    fs::write(
        root.join("model.safetensors.index.json"),
        format!(r#"{{"weight_map":{{{weight_map}}}}}"#),
    )
    .unwrap();
}

#[test]
fn official_bf16_and_local_f16_spellings_preserve_dtype() {
    for (dtype_name, expected) in [("bfloat16", DType::BF16), ("float16", DType::F16)] {
        let root = temp_dir(dtype_name);
        fs::write(root.join("config.json"), config(dtype_name, "10000.0")).unwrap();
        write_shard(
            &root.join("model.safetensors"),
            expected.as_str(),
            &tensor_specs(),
        );
        let cfg = load_config(root.join("config.json")).unwrap();
        let inventory = inspect_weights(&root, &cfg).unwrap();
        assert_eq!(
            inventory.tensor("model.embed_tokens.weight").unwrap().dtype,
            expected
        );
        assert_eq!(
            inventory.tensor("model.embed_tokens.weight").unwrap().len,
            48
        );
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn accepts_multieos_without_inventing_a_default() {
    let root = temp_dir("eos");
    fs::write(root.join("config.json"), config("bfloat16", "10000")).unwrap();
    let cfg = load_config(root.join("config.json")).unwrap();
    assert_eq!(cfg.eos_token_ids, vec![1, 130073]);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejects_inconsistent_rope_spellings() {
    let root = temp_dir("rope");
    let mut text = config("bfloat16", "10000");
    text = text.replace(
        r#""rope_parameters":{"rope_theta":10000}"#,
        r#""rope_parameters":{"rope_theta":20000}"#,
    );
    fs::write(root.join("config.json"), text).unwrap();
    assert!(
        load_config(root.join("config.json"))
            .unwrap_err()
            .contains("disagree")
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejects_tied_head_and_bad_geometry() {
    let root = temp_dir("geometry");
    fs::write(
        root.join("config.json"),
        config("bfloat16", "10000").replace(
            "\"tie_word_embeddings\":false",
            "\"tie_word_embeddings\":true",
        ),
    )
    .unwrap();
    assert!(load_config(root.join("config.json")).is_err());
    fs::write(root.join("config.json"), config("bfloat16", "10000")).unwrap();
    let mut specs = tensor_specs();
    specs[2].1 = vec![3, 4];
    write_shard(&root.join("model.safetensors"), "BF16", &specs);
    let cfg = load_config(root.join("config.json")).unwrap();
    assert!(inspect_weights(&root, &cfg).unwrap_err().contains("shape"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejects_missing_output_head() {
    let root = temp_dir("head");
    fs::write(root.join("config.json"), config("bfloat16", "10000")).unwrap();
    let specs = tensor_specs()
        .into_iter()
        .filter(|(name, _)| *name != "lm_head.weight")
        .collect::<Vec<_>>();
    write_shard(&root.join("model.safetensors"), "BF16", &specs);
    let cfg = load_config(root.join("config.json")).unwrap();
    assert!(
        inspect_weights(&root, &cfg)
            .unwrap_err()
            .contains("lm_head")
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejects_bad_offsets_and_spans() {
    let root = temp_dir("offsets");
    fs::write(root.join("config.json"), config("bfloat16", "10000")).unwrap();
    let shard = root.join("model.safetensors");
    write_shard(&shard, "BF16", &tensor_specs());
    let mut bytes = fs::read(&shard).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header = String::from_utf8(bytes[8..8 + header_len].to_vec()).unwrap();
    header = header.replace("\"data_offsets\":[0,48]", "\"data_offsets\":[0,46]");
    bytes[8..8 + header_len].copy_from_slice(header.as_bytes());
    fs::write(&shard, bytes).unwrap();
    let cfg = load_config(root.join("config.json")).unwrap();
    assert!(
        inspect_weights(&root, &cfg)
            .unwrap_err()
            .contains("byte span")
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejects_duplicate_names_and_index_coverage_gaps() {
    let root = temp_dir("index");
    fs::write(root.join("config.json"), config("bfloat16", "10000")).unwrap();
    let specs = tensor_specs();
    write_shard(&root.join("a.safetensors"), "BF16", &specs);
    write_shard(&root.join("b.safetensors"), "BF16", &specs);
    assert!(
        inspect_weights(&root, &load_config(root.join("config.json")).unwrap())
            .unwrap_err()
            .contains("duplicate")
    );
    let root = temp_dir("coverage");
    fs::write(root.join("config.json"), config("bfloat16", "10000")).unwrap();
    write_shard(&root.join("model-00001.safetensors"), "BF16", &specs);
    write_index(&root, &[("not_in_header", "model-00001.safetensors")]);
    let cfg = load_config(root.join("config.json")).unwrap();
    assert!(
        inspect_weights(&root, &cfg)
            .unwrap_err()
            .contains("coverage")
    );
    fs::remove_dir_all(root).unwrap();
}
