use std::{
    collections::BTreeMap,
    sync::{atomic::AtomicBool, Arc},
};

use logan_chat::{
    engine::{DenseMiniCpm, FamilyEngineHandle, GenerationSettings, StopReason},
    openai::ApiMessage,
    runtime::{render_prompt, ModelFamily, ProtocolOptions},
};
use logan_llama::{DType, DenseModel, DenseTensor, LlamaConfig};
use tokenizers::{models::wordlevel::WordLevel, Tokenizer};

fn tiny_adapter() -> DenseMiniCpm {
    let config = LlamaConfig {
        model_type: "minicpm".into(),
        vocab_size: 2,
        hidden_size: 2,
        intermediate_size: 2,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        max_position_embeddings: 8,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        bos_token_id: None,
        eos_token_ids: vec![1],
        tie_word_embeddings: false,
        torch_dtype: Some(DType::F16),
    };
    let mut tensors = BTreeMap::new();
    let add = |tensors: &mut BTreeMap<String, DenseTensor>,
               name: &str,
               shape: Vec<usize>,
               values: Vec<f32>| {
        tensors.insert(
            name.into(),
            DenseTensor::from_f32(DType::F16, shape, &values).unwrap(),
        );
    };
    add(
        &mut tensors,
        "model.embed_tokens.weight",
        vec![2, 2],
        vec![0.1; 4],
    );
    add(&mut tensors, "lm_head.weight", vec![2, 2], vec![0.1; 4]);
    add(&mut tensors, "model.norm.weight", vec![2], vec![1.0; 2]);
    add(
        &mut tensors,
        "model.layers.0.input_layernorm.weight",
        vec![2],
        vec![1.0; 2],
    );
    add(
        &mut tensors,
        "model.layers.0.self_attn.q_proj.weight",
        vec![2, 2],
        vec![0.1; 4],
    );
    add(
        &mut tensors,
        "model.layers.0.self_attn.k_proj.weight",
        vec![2, 2],
        vec![0.1; 4],
    );
    add(
        &mut tensors,
        "model.layers.0.self_attn.v_proj.weight",
        vec![2, 2],
        vec![0.1; 4],
    );
    add(
        &mut tensors,
        "model.layers.0.self_attn.o_proj.weight",
        vec![2, 2],
        vec![0.1; 4],
    );
    add(
        &mut tensors,
        "model.layers.0.post_attention_layernorm.weight",
        vec![2],
        vec![1.0; 2],
    );
    add(
        &mut tensors,
        "model.layers.0.mlp.gate_proj.weight",
        vec![2, 2],
        vec![0.1; 4],
    );
    add(
        &mut tensors,
        "model.layers.0.mlp.up_proj.weight",
        vec![2, 2],
        vec![0.1; 4],
    );
    add(
        &mut tensors,
        "model.layers.0.mlp.down_proj.weight",
        vec![2, 2],
        vec![0.1; 4],
    );
    let model = Arc::new(DenseModel::from_tensors(config, tensors).unwrap());
    let tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(
                [("hello".into(), 0), ("<unk>".into(), 1)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("<unk>".into())
            .build()
            .unwrap(),
    );
    DenseMiniCpm::from_model(model, tokenizer)
}

#[test]
fn dense_load_rejects_non_package_explicitly() {
    let error = match DenseMiniCpm::load("/definitely/not/a/minicpm5/package") {
        Ok(_) => panic!("non-package unexpectedly loaded"),
        Err(error) => error,
    };
    assert!(error.contains("not a directory"));
}

#[test]
fn dense_adapter_honors_both_minicpm5_eos_ids_and_commits_prompt() {
    let mut adapter = tiny_adapter();
    assert!(adapter.eos_ids().contains(&2));
    assert!(adapter.eos_ids().contains(&73440));
    let output = adapter
        .generate(
            "hello",
            &GenerationSettings {
                max_new: 0,
                ..Default::default()
            },
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(output.input_tokens, 1);
    assert!(output.token_ids.is_empty());
    assert_eq!(output.stop_reason, StopReason::MaxTokens);
}

#[test]
fn minicpm5_prompt_and_family_dispatch_stay_separate_from_qwen() {
    let messages = [ApiMessage {
        role: "user".into(),
        text: "hello".into(),
    }];
    let prompt = render_prompt(
        &logan_chat::runtime::PromptAdapter::for_family(ModelFamily::MiniCpm5),
        &messages,
        None,
    )
    .unwrap();
    assert!(prompt.ends_with("<|im_start|>assistant\n"));
    assert!(!prompt.contains("<think>\n\n</think>"));
    assert_eq!(ProtocolOptions::default().mini_cpm5().tools.len(), 0);

    let handle = logan_chat::engine::spawn_for_family(
        "/missing/qwen-package".into(),
        String::new(),
        ModelFamily::Qwen4,
    )
    .unwrap();
    assert!(matches!(handle, FamilyEngineHandle::Qwen(_)));
}
