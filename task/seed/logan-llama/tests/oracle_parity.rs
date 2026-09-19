use logan_llama::{DType, DenseModel, DenseTensor, LlamaConfig};
use std::{collections::BTreeMap, sync::Arc};

fn cfg(dtype: DType) -> LlamaConfig {
    LlamaConfig {
        model_type: "llama".into(),
        vocab_size: 7,
        hidden_size: 4,
        intermediate_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 2,
        max_position_embeddings: 32,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        bos_token_id: Some(1),
        eos_token_ids: vec![0, 6],
        tie_word_embeddings: false,
        torch_dtype: Some(dtype),
    }
}
fn model(dtype: DType) -> Arc<DenseModel> {
    let c = cfg(dtype);
    let d = 4;
    let v = 7;
    let mut t = BTreeMap::new();
    let put =
        |t: &mut BTreeMap<String, DenseTensor>, n: String, shape: Vec<usize>, vals: Vec<f32>| {
            t.insert(n, DenseTensor::from_f32(dtype, shape, &vals).unwrap());
        };
    put(
        &mut t,
        "model.embed_tokens.weight".into(),
        vec![v, d],
        (0..v * d).map(|x| x as f32 / 20.0).collect(),
    );
    put(
        &mut t,
        "lm_head.weight".into(),
        vec![v, d],
        (0..v * d).map(|x| (x as f32 - 5.0) / 30.0).collect(),
    );
    put(&mut t, "model.norm.weight".into(), vec![d], vec![1.0; d]);
    put(
        &mut t,
        "model.layers.0.input_layernorm.weight".into(),
        vec![d],
        vec![1.0; d],
    );
    put(
        &mut t,
        "model.layers.0.self_attn.q_proj.weight".into(),
        vec![4, d],
        vec![0.1; 16],
    );
    put(
        &mut t,
        "model.layers.0.self_attn.k_proj.weight".into(),
        vec![2, d],
        vec![0.1; 8],
    );
    put(
        &mut t,
        "model.layers.0.self_attn.v_proj.weight".into(),
        vec![2, d],
        vec![0.1; 8],
    );
    put(
        &mut t,
        "model.layers.0.self_attn.o_proj.weight".into(),
        vec![d, 4],
        vec![0.1; 16],
    );
    put(
        &mut t,
        "model.layers.0.post_attention_layernorm.weight".into(),
        vec![d],
        vec![1.0; d],
    );
    put(
        &mut t,
        "model.layers.0.mlp.gate_proj.weight".into(),
        vec![8, d],
        vec![0.02; 32],
    );
    put(
        &mut t,
        "model.layers.0.mlp.up_proj.weight".into(),
        vec![8, d],
        vec![0.03; 32],
    );
    put(
        &mut t,
        "model.layers.0.mlp.down_proj.weight".into(),
        vec![d, 8],
        vec![0.02; 32],
    );
    Arc::new(DenseModel::from_tensors(c, t).unwrap())
}

#[test]
fn cached_chunks_match_one_prefill_for_f16_and_bf16() {
    for dtype in [DType::F16, DType::BF16] {
        let m = model(dtype);
        let mut full = m.session();
        let a = full.forward(&[1, 2, 3], &[0]).unwrap();
        let mut chunk = m.session();
        let b = chunk.forward(&[1], &[0]).unwrap();
        let c = chunk.forward(&[2, 3], &[0]).unwrap();
        assert_eq!(a.logits.len(), b.logits.len() + c.logits.len());
        assert!(
            a.logits
                .iter()
                .zip(b.logits.iter().chain(c.logits.iter()))
                .all(|(x, y)| (x - y).abs() < 2e-2)
        );
        assert_eq!(a.tap(0).unwrap().len(), 12);
    }
}

#[test]
fn causal_multirow_and_padding_boundaries_are_deterministic() {
    let m = model(DType::BF16);
    let mut s = m.session();
    let out = s.forward_padded(&[1, 2, 3, 0, 0], 3, &[]).unwrap();
    assert_eq!(out.rows, 3);
    assert_eq!(out.processed_tokens, 3);
    assert_eq!(s.kv().processed_tokens(), 3);
    assert_eq!(m.config.eos_token_ids, vec![0, 6]);
    assert!(
        out.logits_row(0)
            .unwrap()
            .iter()
            .zip(out.logits_row(1).unwrap())
            .any(|(a, b)| a != b)
    );
}
