use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use logan_qwen4::{load_cfg, OutputGate};

fn write_cfg(body: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("logan-qwen4-gate-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.json");
    fs::write(&path, body).unwrap();
    path
}

#[test]
fn explicit_sigmoid_overrides_silu_hidden_act() {
    let path = write_cfg(
        r#"{
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "head_dim": 64,
            "max_position_embeddings": 64,
            "layer_types": ["linear_attention"],
            "hidden_act": "silu",
            "output_gate_type": "sigmoid"
        }"#,
    );
    let cfg = load_cfg(&path).unwrap();
    assert_eq!(cfg.output_gate, OutputGate::Sigmoid);
    let _ = fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn missing_output_gate_falls_back_to_hidden_act() {
    let path = write_cfg(
        r#"{
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "head_dim": 64,
            "max_position_embeddings": 64,
            "layer_types": ["linear_attention"],
            "hidden_act": "silu"
        }"#,
    );
    let cfg = load_cfg(&path).unwrap();
    assert_eq!(cfg.output_gate, OutputGate::Silu);
    let _ = fs::remove_dir_all(path.parent().unwrap());
}
