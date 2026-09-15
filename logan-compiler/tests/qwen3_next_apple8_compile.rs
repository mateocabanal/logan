use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use logan_compiler::pipeline::{CompileRequest, NoProgress, QuantRequest, TargetRequest, compile};
use logan_format::package::Package;

fn zeros(shape: &[u64]) -> Vec<u8> {
    vec![0; shape.iter().product::<u64>() as usize * 2]
}

fn patterned_bf16_rows(rows: usize, cols: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(rows * cols * 2);
    for row in 0..rows {
        for col in 0..cols {
            bytes.extend_from_slice(&((row * 64 + col) as u16).to_le_bytes());
        }
    }
    bytes
}

fn write_safetensors(root: &Path) {
    let mut tensors = BTreeMap::<String, (Vec<u64>, Vec<u8>)>::new();
    let mut add_zero = |name: &str, shape: &[u64]| {
        tensors.insert(name.to_owned(), (shape.to_vec(), zeros(shape)));
    };

    add_zero("model.embed_tokens.weight", &[8, 32]);
    add_zero("model.norm.weight", &[32]);
    add_zero("lm_head.weight", &[8, 32]);

    let lp = "model.layers.0";
    add_zero(&format!("{lp}.input_layernorm.weight"), &[32]);
    add_zero(&format!("{lp}.post_attention_layernorm.weight"), &[32]);
    add_zero(&format!("{lp}.mlp.gate.weight"), &[1, 32]);
    add_zero(
        &format!("{lp}.mlp.shared_expert.gate_proj.weight"),
        &[8, 32],
    );
    add_zero(&format!("{lp}.mlp.shared_expert.up_proj.weight"), &[8, 32]);
    add_zero(
        &format!("{lp}.mlp.shared_expert.down_proj.weight"),
        &[32, 8],
    );
    add_zero(&format!("{lp}.mlp.shared_expert_gate.weight"), &[1, 32]);
    add_zero(&format!("{lp}.mlp.experts.0.gate_proj.weight"), &[8, 32]);
    add_zero(&format!("{lp}.mlp.experts.0.up_proj.weight"), &[8, 32]);
    add_zero(&format!("{lp}.mlp.experts.0.down_proj.weight"), &[32, 8]);
    add_zero(&format!("{lp}.linear_attn.A_log"), &[1]);
    add_zero(&format!("{lp}.linear_attn.dt_bias"), &[1]);
    add_zero(&format!("{lp}.linear_attn.conv1d.weight"), &[24, 1, 4]);
    add_zero(&format!("{lp}.linear_attn.norm.weight"), &[8]);
    add_zero(&format!("{lp}.linear_attn.out_proj.weight"), &[32, 8]);
    tensors.insert(
        format!("{lp}.linear_attn.in_proj_qkvz.weight"),
        (vec![32, 32], patterned_bf16_rows(32, 32)),
    );
    tensors.insert(
        format!("{lp}.linear_attn.in_proj_ba.weight"),
        (vec![2, 32], patterned_bf16_rows(2, 32)),
    );

    let mut offset = 0_u64;
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, (shape, data)) in tensors {
        let bytes = data.len() as u64;
        header.insert(
            name,
            serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [offset, offset + bytes]
            }),
        );
        payload.extend_from_slice(&data);
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
          "architectures":["Qwen3NextForCausalLM"],
          "model_type":"qwen3_next",
          "num_hidden_layers":1,
          "full_attention_interval":4,
          "hidden_size":32,
          "num_experts":1,
          "num_experts_per_tok":1,
          "decoder_sparse_step":1,
          "mlp_only_layers":[],
          "norm_topk_prob":true,
          "moe_intermediate_size":8,
          "shared_expert_intermediate_size":8,
          "vocab_size":8,
          "num_attention_heads":1,
          "num_key_value_heads":1,
          "head_dim":32,
          "partial_rotary_factor":0.25,
          "rope_theta":5000000,
          "linear_num_key_heads":1,
          "linear_key_head_dim":8,
          "linear_num_value_heads":1,
          "linear_value_head_dim":8,
          "linear_conv_kernel_dim":4,
          "hidden_act":"silu",
          "attention_bias":false,
          "rope_scaling":null,
          "max_position_embeddings":262144
        }"#,
    )
    .unwrap();
    write_safetensors(root);
}

#[test]
fn qwen3_coder_next_compiles_to_verified_apple8_coli() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "logan-qwen3-next-apple8-e2e-{}-{nonce}",
        std::process::id()
    ));
    let source = root.join("source");
    let output = root.join("qwen3-next-apple8.coli");
    source_fixture(&source);

    let mut request = CompileRequest::new(source);
    request.output = Some(output.clone());
    request.target = TargetRequest::Profile("macos-arm64-metal-apple8-v1".into());
    request.quant = QuantRequest::Profile("mxfp4".into());
    request.verify = true;
    compile(&request, &mut NoProgress).unwrap();

    let package = Package::open(&output).unwrap();
    assert_eq!(package.profile(), "macos-arm64-metal-apple8-v1");

    // Routed experts use the same validated Apple8 MXFP4 record ABI as the
    // existing Qwen3/Qwen4 paths.
    let expert = package.expert_records(0, 0);
    assert_eq!(expert.len(), 1);
    let raw = package.read_record(expert[0]).unwrap();
    assert_eq!(&raw[..8], b"COLIEXPT");

    // The compiler must have physically split the official fused GDN weights
    // into Logan's existing execution ABI before package verification.
    let qkv = package
        .record_by_name("layers.0.linear_attn.in_proj_qkv.weight")
        .unwrap();
    assert_eq!(qkv.math_format, 3);
    assert_eq!(qkv.decoded, 24 * 32 * 2);
    let qkv_payload = package.read_tensor_payload(qkv).unwrap();
    let first_values: Vec<u16> = qkv_payload
        .chunks_exact(2)
        .step_by(32)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]) / 64)
        .collect();
    // key_heads=value_heads=1 in this compile fixture, so fused [q,k,v,z]
    // becomes canonical [q(8), k(8), v(8)] with z split away.
    assert_eq!(first_values, (0_u16..24).collect::<Vec<_>>());

    let z = package
        .record_by_name("layers.0.linear_attn.in_proj_z.weight")
        .unwrap();
    let z_payload = package.read_tensor_payload(z).unwrap();
    let z_rows: Vec<u16> = z_payload
        .chunks_exact(2)
        .step_by(32)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]) / 64)
        .collect();
    assert_eq!(z_rows, (24_u16..32).collect::<Vec<_>>());

    // Metadata copied into the package is what the runtime uses to derive
    // raw zero-centred RMSNorm, 5M RoPE and the hybrid layer schedule.
    let copied: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
    assert_eq!(copied["model_type"], "qwen3_next");
    assert_eq!(copied["rope_theta"], 5_000_000);

    // Close the loop through the real runtime loader. With zero model weights,
    // a token forward is numerically uninteresting but still exercises config
    // semantics, resident tensor loading and on-demand Apple8 expert fetch.
    let cfg = logan_qwen4::load_cfg(&output.join("config.json")).unwrap();
    assert!(cfg.zero_centered_norm);
    assert_eq!(cfg.rotary_dim, 8);
    let source = logan_qwen4::colisource::ColiSource::open(&output).unwrap();
    let mut model = logan_qwen4::Model::load_coli(&source, &cfg).unwrap();
    let logits = model.forward_token(0, 0);
    assert_eq!(logits.len(), 8);
    assert!(logits.iter().all(|value| value.is_finite()));

    fs::remove_dir_all(root).unwrap();
}
