use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use logan_compiler::{
    model::qwen_moe::QwenMoeFrontend,
    source::{SourceInventory, TensorRef},
};

struct Fixture {
    root: PathBuf,
    inventory: SourceInventory,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn add(
    tensors: &mut BTreeMap<String, TensorRef>,
    file: &PathBuf,
    off: &mut u64,
    name: String,
    shape: Vec<u64>,
    dtype: &str,
) {
    let elem = match dtype {
        "BF16" => 2,
        "I64" => 8,
        _ => 1,
    };
    let len = shape.iter().product::<u64>() * elem;
    tensors.insert(
        name,
        TensorRef {
            source: file.clone(),
            offset: *off,
            len,
            dtype: dtype.into(),
            shape,
        },
    );
    *off += len + 16;
}

fn fixture() -> Fixture {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "logan-qwen4-reap288-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("config.json"),
        r#"{
      "model_type":"qwen4_exp_text",
      "text_config":{
        "model_type":"qwen4_exp_text",
        "num_hidden_layers":1,
        "layer_types":["linear_attention"],
        "hidden_size":4,
        "num_experts":288,
        "num_experts_per_tok":10,
        "moe_intermediate_size":2,
        "shared_expert_intermediate_size":3,
        "vocab_size":8,
        "num_attention_heads":2,
        "num_key_value_heads":1,
        "head_dim":2,
        "linear_num_key_heads":1,
        "linear_key_head_dim":2,
        "linear_num_value_heads":2,
        "linear_value_head_dim":2,
        "linear_conv_kernel_dim":4,
        "hc_count":4,
        "ple_layer_ids":[1]
      }
    }"#,
    )
    .unwrap();
    let file = root.join("weights.bin");
    let mut tensors = BTreeMap::new();
    let mut off = 4096;
    add(
        &mut tensors,
        &file,
        &mut off,
        "model.language_model.embed_tokens.weight".into(),
        vec![8, 4],
        "BF16",
    );
    add(
        &mut tensors,
        &file,
        &mut off,
        "lm_head.weight".into(),
        vec![8, 4],
        "BF16",
    );
    for e in 0..288 {
        add(
            &mut tensors,
            &file,
            &mut off,
            format!("model.language_model.layers.0.mlp.experts.{e}.gate_up_proj"),
            vec![4, 4],
            "BF16",
        );
        add(
            &mut tensors,
            &file,
            &mut off,
            format!("model.language_model.layers.0.mlp.experts.{e}.down_proj"),
            vec![4, 2],
            "BF16",
        );
    }
    add(
        &mut tensors,
        &file,
        &mut off,
        "model.language_model.layers.0.mlp.gate.weight".into(),
        vec![288, 4],
        "BF16",
    );
    // Real Qwen3.8 uses model.language_model.layers.(ple_id-1).ple.*.
    add(
        &mut tensors,
        &file,
        &mut off,
        "model.language_model.layers.0.ple.key_proj.weight".into(),
        vec![4, 4],
        "BF16",
    );
    add(
        &mut tensors,
        &file,
        &mut off,
        "model.language_model.layers.0.ple.ple_embedding.ngram_heads_vocab_sizes".into(),
        vec![2],
        "I64",
    );
    add(
        &mut tensors,
        &file,
        &mut off,
        "model.language_model.layers.0.ple.ple_embedding.ngram_heads_offsets".into(),
        vec![2],
        "I64",
    );
    add(
        &mut tensors,
        &file,
        &mut off,
        "model.language_model.layers.0.ple.ple_embedding.layer_multipliers".into(),
        vec![3],
        "I64",
    );
    add(
        &mut tensors,
        &file,
        &mut off,
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.weight".into(),
        vec![256, 2],
        "BF16",
    );
    // The frontend's PLE scale census reads the BF16 bytes. Zero-filled input
    // is deterministic and yields the fallback exact BF16 scale 1.0 (0x3f80).
    fs::write(&file, vec![0_u8; off as usize]).unwrap();

    let inventory = SourceInventory {
        root: root.clone(),
        files: vec![root.join("config.json"), file],
        source_stored_bytes: tensors.values().map(|t| t.len).sum(),
        dtype_counts: BTreeMap::new(),
        source_fingerprint: "00".repeat(32),
        config_fingerprint: None,
        architecture_hint: Some("Qwen4ExpForCausalLM".into()),
        tensors,
    };
    Fixture { root, inventory }
}

#[test]
fn reap_288_is_first_class_geometry_and_expert_slices_are_bf16() {
    let f = fixture();
    assert!(QwenMoeFrontend::probe(&f.inventory).unwrap());
    let model = QwenMoeFrontend::build(&f.inventory).unwrap();
    assert_eq!(model.geometry.routed_experts_per_layer, 288);
    assert_eq!(model.geometry.experts_per_token, 10);
    assert_eq!(model.routed_experts.len(), 288);
    let expert = &model.routed_experts[&(0, 287)];
    assert_eq!(expert.gate.rows, 2);
    assert_eq!(expert.gate.columns, 4);
    assert_eq!(expert.up.rows, 2);
    assert_eq!(expert.down.rows, 4);
    assert_eq!(expert.down.columns, 2);
    assert_eq!(expert.gate.source.dtype, "BF16");
}

#[test]
fn bf16_ple_is_lowered_to_streamed_f8_shards_with_derived_scale() {
    let f = fixture();
    let model = QwenMoeFrontend::build(&f.inventory).unwrap();
    let layer = &model.layer_static_tensors[&0];
    let shards: Vec<_> = layer
        .keys()
        .filter(|name| name.starts_with("ple.ple_embedding.ngram_embedding.shard_"))
        .collect();
    assert_eq!(shards.len(), 128);
    assert_eq!(
        layer["ple.ple_embedding.ngram_embedding.shard_000"].dtype,
        "BF16_TO_F8_E4M3:3f80"
    );
    assert_eq!(
        layer["ple.ple_embedding.ngram_embedding.shard_000"].shape,
        vec![2, 2]
    );
    assert_eq!(
        layer["ple.ple_embedding.ngram_embedding.weight_scale"].dtype,
        "CONST_BF16:3f80"
    );
    assert!(layer.contains_key("ple.ple_embedding.ngram_heads_vocab_sizes"));
    assert!(!model.resident_tensors.contains_key(
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.weight"
    ));
}
