use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use logan_compiler::{
    model::llama::{LlamaFrontend, LlamaProfile},
    source::{SourceInventory, TensorRef},
};

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(1);

struct Fixture {
    root: PathBuf,
    inventory: SourceInventory,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn add_tensor(
    tensors: &mut BTreeMap<String, TensorRef>,
    source: &PathBuf,
    name: impl Into<String>,
    dtype: &str,
    shape: &[u64],
    offset: &mut u64,
) {
    let len = shape.iter().copied().product::<u64>() * 2;
    tensors.insert(
        name.into(),
        TensorRef {
            source: source.clone(),
            offset: *offset,
            len,
            dtype: dtype.into(),
            shape: shape.to_vec(),
        },
    );
    *offset += len;
}
fn add_typed_tensor(
    tensors: &mut BTreeMap<String, TensorRef>,
    source: &PathBuf,
    name: impl Into<String>,
    dtype: &str,
    shape: &[u64],
    item_bytes: u64,
    offset: &mut u64,
) {
    let len = shape.iter().copied().product::<u64>() * item_bytes;
    tensors.insert(
        name.into(),
        TensorRef {
            source: source.clone(),
            offset: *offset,
            len,
            dtype: dtype.into(),
            shape: shape.to_vec(),
        },
    );
    *offset += len;
}

fn affine_fixture() -> Fixture {
    let mut fixture = fixture("minicpm5", None);
    let config_path = fixture.root.join("config.json");
    fs::write(
        &config_path,
        fs::read_to_string(&config_path)
            .unwrap()
            .replace(
                r#""tie_word_embeddings":false"#,
                r#""tie_word_embeddings":false, "quantization_config":{"mode":"affine","bits":4,"group_size":2}"#,
            ),
    )
    .unwrap();
    let shard = fixture.root.join("model.safetensors");
    let mut inventory = fixture.inventory.clone();
    let packed = inventory
        .tensors
        .get_mut("model.layers.0.mlp.gate_proj.weight")
        .unwrap();
    packed.dtype = "U32".into();
    packed.shape = vec![8, 1];
    packed.len = 8 * 1 * 4;
    let mut offset = fs::metadata(&shard).unwrap().len();
    add_typed_tensor(
        &mut inventory.tensors,
        &shard,
        "model.layers.0.mlp.gate_proj.scales",
        "BF16",
        &[8, 2],
        2,
        &mut offset,
    );
    add_typed_tensor(
        &mut inventory.tensors,
        &shard,
        "model.layers.0.mlp.gate_proj.biases",
        "F32",
        &[8, 2],
        4,
        &mut offset,
    );
    let mut bytes = fs::read(&shard).unwrap();
    bytes.resize(offset as usize, 0);
    fs::write(&shard, bytes).unwrap();
    inventory.source_stored_bytes = offset;
    fixture.inventory = inventory;
    fixture
}

fn fixture(model_type: &str, architecture: Option<&str>) -> Fixture {
    let id = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!("logan-minicpm5-{}-{id}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let architecture_json = architecture
        .map(|value| format!(r#", "architectures":["{value}"]"#))
        .unwrap_or_default();
    fs::write(
        root.join("config.json"),
        format!(
            r#"{{
              "model_type":"{model_type}",
              "vocab_size":6,
              "hidden_size":4,
              "intermediate_size":8,
              "num_hidden_layers":1,
              "num_attention_heads":2,
              "num_key_value_heads":1,
              "head_dim":2,
              "max_position_embeddings":64,
              "rms_norm_eps":0.00001,
              "rope_theta":10000,
              "rope_parameters":{{"rope_theta":10000}},
              "bos_token_id":1,
              "eos_token_id":[2,73440],
              "tie_word_embeddings":false{architecture_json}
            }}"#
        ),
    )
    .unwrap();

    let shard = root.join("model.safetensors");
    let mut tensors = BTreeMap::new();
    let mut offset = 0;
    add_tensor(
        &mut tensors,
        &shard,
        "model.embed_tokens.weight",
        "BF16",
        &[6, 4],
        &mut offset,
    );
    add_tensor(
        &mut tensors,
        &shard,
        "model.norm.weight",
        "F16",
        &[4],
        &mut offset,
    );
    add_tensor(
        &mut tensors,
        &shard,
        "lm_head.weight",
        "BF16",
        &[6, 4],
        &mut offset,
    );
    for (name, shape) in [
        ("input_layernorm.weight", vec![4]),
        ("self_attn.q_proj.weight", vec![4, 4]),
        ("self_attn.k_proj.weight", vec![2, 4]),
        ("self_attn.v_proj.weight", vec![2, 4]),
        ("self_attn.o_proj.weight", vec![4, 4]),
        ("post_attention_layernorm.weight", vec![4]),
        ("mlp.gate_proj.weight", vec![8, 4]),
        ("mlp.up_proj.weight", vec![8, 4]),
        ("mlp.down_proj.weight", vec![4, 8]),
    ] {
        add_tensor(
            &mut tensors,
            &shard,
            format!("model.layers.0.{name}"),
            "BF16",
            &shape,
            &mut offset,
        );
    }
    fs::write(&shard, vec![0_u8; offset as usize]).unwrap();
    let inventory = SourceInventory {
        root: root.clone(),
        files: vec![root.join("config.json"), shard],
        tensors,
        source_stored_bytes: offset,
        dtype_counts: BTreeMap::new(),
        source_fingerprint: String::new(),
        config_fingerprint: None,
        architecture_hint: None,
    };
    Fixture { root, inventory }
}

#[test]
fn valid_minicpm5_source_preserves_exact_inventory_and_metadata() {
    let fixture = fixture("llama", Some("MiniCPM5ForCausalLM"));
    assert!(LlamaFrontend::probe(&fixture.inventory).unwrap());
    let source = LlamaFrontend::from_source(&fixture.inventory).unwrap();
    assert_eq!(source.profile, LlamaProfile::MiniCpm5);
    assert_eq!(source.geometry.num_key_value_heads, 1);
    assert_eq!(source.geometry.eos_token_ids, vec![2, 73440]);
    assert_eq!(source.global_tensors["embed.weight"].dtype, "BF16");
    assert_eq!(source.global_tensors["norm.weight"].dtype, "F16");
    assert_eq!(source.layer_tensors[&0].len(), 9);
    assert!(source.resident_tensors.is_empty());

    let lowered = LlamaFrontend::lower(&source).unwrap();
    assert!(lowered.capabilities.dense);
    assert!(lowered.capabilities.grouped_query_attention);
    assert!(!lowered.capabilities.mlx_affine_quantization);
}

#[test]
fn generic_llama_profile_is_not_minicpm5() {
    let fixture = fixture("llama", None);
    let source = LlamaFrontend::from_source(&fixture.inventory).unwrap();
    assert_eq!(source.profile, LlamaProfile::StandardLlama);
}

#[test]
fn rejects_wrong_gqa_geometry_and_missing_head() {
    let gqa_fixture = fixture("minicpm5", None);
    fs::write(
        gqa_fixture.root.join("config.json"),
        fs::read_to_string(gqa_fixture.root.join("config.json"))
            .unwrap()
            .replace(r#""num_key_value_heads":1"#, r#""num_key_value_heads":3"#),
    )
    .unwrap();
    let error = LlamaFrontend::from_source(&gqa_fixture.inventory)
        .unwrap_err()
        .to_string();
    assert!(error.contains("divisible"), "{error}");

    let valid_fixture = fixture("minicpm5", None);
    let mut missing_head = valid_fixture.inventory.clone();
    missing_head
        .tensors
        .remove("lm_head.weight")
        .expect("fixture has output head");
    let error = LlamaFrontend::from_source(&missing_head)
        .unwrap_err()
        .to_string();
    assert!(error.contains("untied lm_head"), "{error}");
}

#[test]
fn accepts_mlx_affine_mixed_bits_without_rewriting_source_spans() {
    let fixture = affine_fixture();
    let packed = fixture
        .inventory
        .tensors
        .get("model.layers.0.mlp.gate_proj.weight")
        .unwrap();
    let source = LlamaFrontend::from_source(&fixture.inventory).unwrap();
    let view = &source.layer_tensors[&0]["mlp.gate_proj.weight"];
    assert!(source.capabilities.mlx_affine_quantization);
    assert_eq!(view.dtype, "MLX_AFFINE:4:2");
    assert_eq!(view.shape, vec![8, 4]);
    assert_eq!(view.offset, packed.offset);
    assert_eq!(view.len, packed.len);
    assert_eq!(
        source.resident_tensors["model.layers.0.mlp.gate_proj.scales"].shape,
        vec![8, 2]
    );
    assert_eq!(
        source.resident_tensors["model.layers.0.mlp.gate_proj.biases"].dtype,
        "F32"
    );
}

#[test]
fn accepts_mlx_oq8e_config_mode_alias() {
    let fixture = affine_fixture();
    fs::write(
        fixture.root.join("config.json"),
        fs::read_to_string(fixture.root.join("config.json"))
            .unwrap()
            .replace(r#""mode":"affine""#, r#""mode":"oq8e""#),
    )
    .unwrap();
    assert!(LlamaFrontend::from_source(&fixture.inventory).is_ok());
}

#[test]
fn rejects_unsupported_mlx_mode_and_bits() {
    let fixture = affine_fixture();
    fs::write(
        fixture.root.join("config.json"),
        fs::read_to_string(fixture.root.join("config.json"))
            .unwrap()
            .replace(r#""mode":"affine""#, r#""mode":"metal""#),
    )
    .unwrap();
    let error = LlamaFrontend::from_source(&fixture.inventory)
        .unwrap_err()
        .to_string();
    assert!(error.contains("unsupported quantization mode"), "{error}");

    let fixture = affine_fixture();
    fs::write(
        fixture.root.join("config.json"),
        fs::read_to_string(fixture.root.join("config.json"))
            .unwrap()
            .replace(r#""bits":4"#, r#""bits":7"#),
    )
    .unwrap();
    let error = LlamaFrontend::from_source(&fixture.inventory)
        .unwrap_err()
        .to_string();
    assert!(error.contains("unsupported bits"), "{error}");
}

#[test]
fn rejects_mlx_sidecar_geometry_mismatch() {
    let fixture = affine_fixture();
    let mut malformed = fixture.inventory.clone();
    let scales = malformed
        .tensors
        .get_mut("model.layers.0.mlp.gate_proj.scales")
        .unwrap();
    scales.shape = vec![8, 1];
    scales.len = 8 * 2;
    let error = LlamaFrontend::from_source(&malformed)
        .unwrap_err()
        .to_string();
    assert!(error.contains("scales sidecar shape"), "{error}");
}
