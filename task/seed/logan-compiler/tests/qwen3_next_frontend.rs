use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use logan_compiler::{
    ir::Architecture,
    model::qwen3_next::Qwen3NextFrontend,
    source::{SourceInventory, TensorRef},
    target,
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

fn len(shape: &[u64]) -> u64 {
    shape.iter().product::<u64>() * 2
}
fn tref(path: &PathBuf, shape: &[u64]) -> TensorRef {
    TensorRef {
        source: path.clone(),
        offset: 0,
        len: len(shape),
        dtype: "BF16".into(),
        shape: shape.to_vec(),
    }
}
fn add(
    t: &mut BTreeMap<String, TensorRef>,
    path: &PathBuf,
    name: impl Into<String>,
    shape: &[u64],
) {
    t.insert(name.into(), tref(path, shape));
}
fn patterned_rows(path: &PathBuf, rows: usize, cols: usize) {
    let mut bytes = Vec::with_capacity(rows * cols * 2);
    for r in 0..rows {
        for c in 0..cols {
            bytes.extend_from_slice(&((r * 10 + c) as u16).to_le_bytes());
        }
    }
    fs::write(path, bytes).unwrap();
}
fn decoded_rows(bytes: &[u8], cols: usize) -> Vec<Vec<u16>> {
    bytes
        .chunks_exact(cols * 2)
        .map(|row| {
            row.chunks_exact(2)
                .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
                .collect()
        })
        .collect()
}

fn fixture() -> Fixture {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("logan-qwen3-next-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("config.json"),
        r#"{
      "model_type":"qwen3_next",
      "num_hidden_layers":4,
      "full_attention_interval":4,
      "hidden_size":2,
      "num_experts":2,
      "num_experts_per_tok":2,
      "moe_intermediate_size":1,
      "shared_expert_intermediate_size":2,
      "vocab_size":3,
      "num_attention_heads":1,
      "num_key_value_heads":1,
      "head_dim":2,
      "linear_num_key_heads":2,
      "linear_key_head_dim":1,
      "linear_num_value_heads":4,
      "linear_value_head_dim":1,
      "linear_conv_kernel_dim":4
    }"#,
    )
    .unwrap();
    let dummy = root.join("dummy.bin");
    fs::write(&dummy, []).unwrap();
    let qkvz = root.join("qkvz.bin");
    patterned_rows(&qkvz, 12, 2);
    let ba = root.join("ba.bin");
    patterned_rows(&ba, 8, 2);
    let mut tensors = BTreeMap::new();
    add(&mut tensors, &dummy, "model.embed_tokens.weight", &[3, 2]);
    add(&mut tensors, &dummy, "model.norm.weight", &[2]);
    add(&mut tensors, &dummy, "lm_head.weight", &[3, 2]);
    for layer in 0..4 {
        let lp = format!("model.layers.{layer}");
        add(
            &mut tensors,
            &dummy,
            format!("{lp}.input_layernorm.weight"),
            &[2],
        );
        add(
            &mut tensors,
            &dummy,
            format!("{lp}.post_attention_layernorm.weight"),
            &[2],
        );
        add(
            &mut tensors,
            &dummy,
            format!("{lp}.mlp.gate.weight"),
            &[2, 2],
        );
        add(
            &mut tensors,
            &dummy,
            format!("{lp}.mlp.shared_expert.gate_proj.weight"),
            &[2, 2],
        );
        add(
            &mut tensors,
            &dummy,
            format!("{lp}.mlp.shared_expert.up_proj.weight"),
            &[2, 2],
        );
        add(
            &mut tensors,
            &dummy,
            format!("{lp}.mlp.shared_expert.down_proj.weight"),
            &[2, 2],
        );
        add(
            &mut tensors,
            &dummy,
            format!("{lp}.mlp.shared_expert_gate.weight"),
            &[1, 2],
        );
        for expert in 0..2 {
            add(
                &mut tensors,
                &dummy,
                format!("{lp}.mlp.experts.{expert}.gate_proj.weight"),
                &[1, 2],
            );
            add(
                &mut tensors,
                &dummy,
                format!("{lp}.mlp.experts.{expert}.up_proj.weight"),
                &[1, 2],
            );
            add(
                &mut tensors,
                &dummy,
                format!("{lp}.mlp.experts.{expert}.down_proj.weight"),
                &[2, 1],
            );
        }
        if layer < 3 {
            add(
                &mut tensors,
                &dummy,
                format!("{lp}.linear_attn.A_log"),
                &[4],
            );
            add(
                &mut tensors,
                &dummy,
                format!("{lp}.linear_attn.dt_bias"),
                &[4],
            );
            add(
                &mut tensors,
                &dummy,
                format!("{lp}.linear_attn.conv1d.weight"),
                &[8, 1, 4],
            );
            add(
                &mut tensors,
                &dummy,
                format!("{lp}.linear_attn.norm.weight"),
                &[1],
            );
            add(
                &mut tensors,
                &dummy,
                format!("{lp}.linear_attn.out_proj.weight"),
                &[2, 4],
            );
            tensors.insert(
                format!("{lp}.linear_attn.in_proj_qkvz.weight"),
                tref(&qkvz, &[12, 2]),
            );
            tensors.insert(
                format!("{lp}.linear_attn.in_proj_ba.weight"),
                tref(&ba, &[8, 2]),
            );
        } else {
            add(
                &mut tensors,
                &dummy,
                format!("{lp}.self_attn.q_proj.weight"),
                &[4, 2],
            );
            add(
                &mut tensors,
                &dummy,
                format!("{lp}.self_attn.k_proj.weight"),
                &[2, 2],
            );
            add(
                &mut tensors,
                &dummy,
                format!("{lp}.self_attn.v_proj.weight"),
                &[2, 2],
            );
            add(
                &mut tensors,
                &dummy,
                format!("{lp}.self_attn.o_proj.weight"),
                &[2, 2],
            );
            add(
                &mut tensors,
                &dummy,
                format!("{lp}.self_attn.q_norm.weight"),
                &[2],
            );
            add(
                &mut tensors,
                &dummy,
                format!("{lp}.self_attn.k_norm.weight"),
                &[2],
            );
        }
    }
    let source_stored_bytes = tensors.values().map(|t| t.len).sum();
    Fixture {
        root: root.clone(),
        inventory: SourceInventory {
            root,
            files: vec![dummy, qkvz, ba],
            tensors,
            source_stored_bytes,
            dtype_counts: BTreeMap::from([("BF16".to_owned(), source_stored_bytes)]),
            source_fingerprint: "00".repeat(32),
            config_fingerprint: None,
            architecture_hint: Some("qwen3_next".to_owned()),
        },
    }
}

#[test]
fn coder_next_derives_three_to_one_hybrid_schedule_and_preserves_topk() {
    let f = fixture();
    assert!(Qwen3NextFrontend::probe(&f.inventory).unwrap());
    let model = Qwen3NextFrontend::build(&f.inventory).unwrap();
    assert_eq!(model.architecture, Architecture::Qwen3Next);
    assert_eq!(model.geometry.layers, 4);
    assert_eq!(model.geometry.routed_experts_per_layer, 2);
    assert_eq!(model.geometry.experts_per_token, 2);
    assert_eq!(model.routed_experts.len(), 8);
    for layer in 0..3 {
        let static_tensors = &model.layer_static_tensors[&layer];
        assert!(static_tensors.contains_key("linear_attn.in_proj_qkv.weight"));
        assert!(static_tensors.contains_key("linear_attn.in_proj_z.weight"));
        assert!(!static_tensors.contains_key("self_attn.q_proj.weight"));
    }
    assert!(model.layer_static_tensors[&3].contains_key("self_attn.q_proj.weight"));
    assert!(!model.layer_static_tensors[&3].contains_key("linear_attn.in_proj_qkv.weight"));
    assert!(model.resident_tensors.is_empty());
}

#[test]
fn fused_deltanet_rows_are_stream_reordered_into_runtime_abi() {
    let f = fixture();
    let model = Qwen3NextFrontend::build(&f.inventory).unwrap();
    let layer = &model.layer_static_tensors[&0];
    let cases = [
        (
            "linear_attn.in_proj_qkv.weight",
            vec![0, 6, 1, 7, 2, 3, 8, 9],
        ),
        ("linear_attn.in_proj_z.weight", vec![4, 5, 10, 11]),
        ("linear_attn.in_proj_b.weight", vec![0, 1, 4, 5]),
        ("linear_attn.in_proj_a.weight", vec![2, 3, 6, 7]),
    ];
    for (name, expected_rows) in cases {
        let payload = target::lower_exact_tensor(&layer[name]).unwrap();
        let rows = decoded_rows(&payload[128..], 2);
        let ids: Vec<u16> = rows.iter().map(|row| row[0] / 10).collect();
        assert_eq!(ids, expected_rows, "{name}");
        assert!(rows.iter().all(|row| row[1] == row[0] + 1), "{name}");
    }
}
