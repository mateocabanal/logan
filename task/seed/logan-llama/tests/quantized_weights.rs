use logan_llama::{
    LlamaConfig, MlxQuantization, QuantizedBits, QuantizedDType, inspect_quantized_weights,
};
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
    let path = std::env::temp_dir().join(format!("logan-llama-quant-{label}-{nonce}"));
    fs::create_dir_all(&path).unwrap();
    path
}

fn config() -> LlamaConfig {
    LlamaConfig {
        model_type: "minicpm5".into(),
        vocab_size: 8,
        hidden_size: 8,
        intermediate_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 8,
        max_position_embeddings: 16,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        bos_token_id: None,
        eos_token_ids: vec![],
        tie_word_embeddings: false,
        torch_dtype: None,
    }
}

fn write_shard(path: &Path, specs: &[(&str, &str, Vec<u64>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0_u64;
    let mut payload = Vec::new();
    for (name, dtype, shape, bytes) in specs {
        let end = offset + bytes.len() as u64;
        header.insert(
            (*name).into(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}),
        );
        offset = end;
        payload.extend_from_slice(bytes);
    }
    let encoded = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut output = Vec::new();
    output.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
    output.extend_from_slice(&encoded);
    output.extend_from_slice(&payload);
    fs::write(path, output).unwrap();
}

fn manifest(path: &Path, sidecar_shape: Option<&str>) {
    manifest_with_mode(path, sidecar_shape, "affine");
}

fn manifest_with_mode(path: &Path, sidecar_shape: Option<&str>, mode: &str) {
    let shape = sidecar_shape.unwrap_or("[2,1]");
    fs::write(path.join("mlx_quantization.json"), format!(r#"{{
      "mode":"{mode}", "source_quantization":"mlx-test-v1",
      "tensors": {{
        "q4": {{"packed":"q4", "scales":"q4.scales", "biases":"q4.biases", "bits":4, "group_size":8, "shape":[2,8]}},
        "q5": {{"packed":"q5", "scales":"q5.scales", "biases":"q5.biases", "bits":5, "group_size":8, "shape":[2,8]}},
        "q6": {{"packed":"q6", "scales":"q6.scales", "biases":"q6.biases", "bits":6, "group_size":8, "shape":[2,8]}},
        "q8": {{"packed":"q8", "scales":"q8.scales", "biases":"q8.biases", "bits":8, "group_size":8, "shape":[2,8], "scale_shape":{shape}}}
      }}
    }}"#)).unwrap();
}

#[test]
fn mixed_bits_inventory_preserves_packed_source_identity() {
    let root = temp_dir("mixed");
    let specs = [
        ("q4", "U32", vec![2, 1], vec![0x11; 8]),
        ("q4.scales", "F16", vec![2, 1], vec![0; 4]),
        ("q4.biases", "F16", vec![2, 1], vec![0; 4]),
        ("q5", "U32", vec![2, 2], vec![0x22; 16]),
        ("q5.scales", "F16", vec![2, 1], vec![0; 4]),
        ("q5.biases", "F16", vec![2, 1], vec![0; 4]),
        ("q6", "U32", vec![2, 2], vec![0x33; 16]),
        ("q6.scales", "F16", vec![2, 1], vec![0; 4]),
        ("q6.biases", "F16", vec![2, 1], vec![0; 4]),
        ("q8", "U32", vec![2, 2], vec![0x44; 16]),
        ("q8.scales", "F16", vec![2, 1], vec![0; 4]),
        ("q8.biases", "F16", vec![2, 1], vec![0; 4]),
    ];
    write_shard(&root.join("model.safetensors"), &specs);
    manifest(&root, None);
    let inventory = inspect_quantized_weights(&root, &config()).unwrap();
    assert_eq!(inventory.representation, MlxQuantization::Affine);
    assert_eq!(inventory.tensor("q4").unwrap().bits, QuantizedBits::B4);
    assert_eq!(inventory.tensor("q5").unwrap().bits, QuantizedBits::B5);
    assert_eq!(inventory.tensor("q6").unwrap().bits, QuantizedBits::B6);
    assert_eq!(inventory.tensor("q8").unwrap().bits, QuantizedBits::B8);
    assert_eq!(
        inventory.tensor("q4").unwrap().packed.dtype,
        QuantizedDType::U32
    );
    assert_eq!(inventory.len(), 4);
    let source = fs::read(root.join("model.safetensors")).unwrap();
    let span = inventory.tensor("q4").unwrap().packed.byte_span();
    assert_eq!(&source[span.start as usize..span.end as usize], &[0x11; 8]);
    for mode in ["oQ8e", "MLX-oQ8e"] {
        manifest_with_mode(&root, None, mode);
        let alias_inventory = inspect_quantized_weights(&root, &config()).unwrap();
        assert_eq!(
            alias_inventory.representation,
            MlxQuantization::Oqe,
            "mode {mode}"
        );
    }
}

#[test]
fn sidecar_shape_and_dtype_mismatches_fail_before_execution() {
    let root = temp_dir("mismatch");
    write_shard(
        &root.join("model.safetensors"),
        &[
            ("q", "U32", vec![2, 1], vec![0; 8]),
            ("q.scales", "F32", vec![2, 1], vec![0; 8]),
            ("q.biases", "F16", vec![2, 1], vec![0; 4]),
        ],
    );
    fs::write(root.join("mlx_quantization.json"), r#"{"mode":"affine","source_quantization":"mlx-test-v1","tensors":{"q":{"bits":4,"group_size":8,"shape":[2,8],"scale_dtype":"F16","scales":"q.scales","biases":"q.biases","scale_shape":[1,2]}}}"#).unwrap();
    let error = inspect_quantized_weights(&root, &config()).unwrap_err();
    assert!(error.contains("scale"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn unsupported_execution_representation_is_rejected_explicitly() {
    let root = temp_dir("unsupported");
    fs::write(
        root.join("mlx_quantization.json"),
        r#"{"mode":"metal-dequantized","source_quantization":"mlx-test-v1","tensors":{}}"#,
    )
    .unwrap();
    let error = inspect_quantized_weights(&root, &config()).unwrap_err();
    assert!(error.contains("unsupported quantized representation"));
    fs::remove_dir_all(root).unwrap();
}
