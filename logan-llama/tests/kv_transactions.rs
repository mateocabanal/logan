use logan_llama::{DType, DenseModel, DenseTensor, KvCache, LlamaConfig};
use std::{collections::BTreeMap, sync::Arc};

#[test]
fn checkpoint_restore_and_retain_verified_are_generation_safe() {
    let mut kv = KvCache::new(1, 2);
    let c0 = kv.checkpoint();
    let c1 = kv.commit(&[vec![1.0, 2.0]], [vec![3.0]].as_slice(), 1);
    assert!(c1.is_err());
    let c1 = kv.commit(&[vec![1.0, 2.0]], &[vec![3.0, 4.0]], 1).unwrap();
    assert!(kv.is_current(c1));
    assert!(kv.retain_verified(c1).is_ok());
    assert!(!kv.is_current(c0));
    let c2 = kv.commit(&[vec![5.0, 6.0]], &[vec![7.0, 8.0]], 1).unwrap();
    assert!(!kv.is_current(c1));
    kv.retain_verified(c1).unwrap();
    assert_eq!(kv.checkpoint(), c1);
    assert_ne!(c1, c2);
}

fn tiny() -> Arc<DenseModel> {
    let c = LlamaConfig {
        model_type: "llama".into(),
        vocab_size: 3,
        hidden_size: 2,
        intermediate_size: 2,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        max_position_embeddings: 8,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        bos_token_id: None,
        eos_token_ids: vec![0],
        tie_word_embeddings: false,
        torch_dtype: Some(DType::F16),
    };
    let mut t = BTreeMap::new();
    let add = |t: &mut BTreeMap<String, DenseTensor>, n: &str, sh: Vec<usize>, x: Vec<f32>| {
        t.insert(n.into(), DenseTensor::from_f32(DType::F16, sh, &x).unwrap());
    };
    add(
        &mut t,
        "model.embed_tokens.weight",
        vec![3, 2],
        vec![0.1; 6],
    );
    add(&mut t, "lm_head.weight", vec![3, 2], vec![0.2; 6]);
    add(&mut t, "model.norm.weight", vec![2], vec![1.; 2]);
    add(
        &mut t,
        "model.layers.0.input_layernorm.weight",
        vec![2],
        vec![1.; 2],
    );
    add(
        &mut t,
        "model.layers.0.self_attn.q_proj.weight",
        vec![2, 2],
        vec![0.1; 4],
    );
    add(
        &mut t,
        "model.layers.0.self_attn.k_proj.weight",
        vec![2, 2],
        vec![0.1; 4],
    );
    add(
        &mut t,
        "model.layers.0.self_attn.v_proj.weight",
        vec![2, 2],
        vec![0.1; 4],
    );
    add(
        &mut t,
        "model.layers.0.self_attn.o_proj.weight",
        vec![2, 2],
        vec![0.1; 4],
    );
    add(
        &mut t,
        "model.layers.0.post_attention_layernorm.weight",
        vec![2],
        vec![1.; 2],
    );
    add(
        &mut t,
        "model.layers.0.mlp.gate_proj.weight",
        vec![2, 2],
        vec![0.1; 4],
    );
    add(
        &mut t,
        "model.layers.0.mlp.up_proj.weight",
        vec![2, 2],
        vec![0.1; 4],
    );
    add(
        &mut t,
        "model.layers.0.mlp.down_proj.weight",
        vec![2, 2],
        vec![0.1; 4],
    );
    Arc::new(DenseModel::from_tensors(c, t).unwrap())
}

#[test]
fn session_rollback_replays_same_logits() {
    let m = tiny();
    let mut s = m.session();
    let mark = s.checkpoint();
    let first = s.forward(&[1, 2], &[]).unwrap();
    s.restore(mark).unwrap();
    let replay = s.forward(&[1, 2], &[]).unwrap();
    assert_eq!(first.logits, replay.logits);
    assert_eq!(first.processed_tokens, 2);
}
