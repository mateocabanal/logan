use logan_qwen4::{ggufsource::GgufSource, load_cfg_gguf};
use std::{collections::BTreeMap, path::Path};

fn main() {
    let p = std::env::args().nth(1).expect("gguf path");
    let src = GgufSource::open(Path::new(&p)).expect("open GGUF");
    println!("arch={:?}", src.string("general.architecture"));
    println!("tensors={} data_start={} alignment={}", src.tensors().len(), src.data_start(), src.alignment());
    for k in [
        "qwen4exp.block_count",
        "qwen4exp.embedding_length",
        "qwen4exp.context_length",
        "qwen4exp.expert_count",
        "qwen4exp.expert_used_count",
        "qwen4exp.expert_feed_forward_length",
        "qwen4exp.expert_shared_feed_forward_length",
        "qwen4exp.hyper_connection.count",
        "qwen4exp.attention.indexer.top_k",
        "qwen4exp.ple.ngram_size",
        "qwen4exp.ple.heads_per_ngram",
    ] {
        println!("{k}={:?}", src.u64(k));
    }
    let mut all = BTreeMap::<&str, usize>::new();
    let mut expert = BTreeMap::<&str, usize>::new();
    for (name, t) in src.tensors() {
        *all.entry(t.dtype.name()).or_default() += 1;
        if name.contains("_exps.") || name.contains(".experts.") {
            *expert.entry(t.dtype.name()).or_default() += 1;
        }
    }
    println!("dtype_all={all:?}");
    println!("dtype_expert={expert:?}");
    match load_cfg_gguf(&src) {
        Ok(c) => println!(
            "cfg hidden={} layers={} experts={} topk={} moe_inter={} shared_inter={} vocab={} hc={} ple_layer={} ple_dim={}",
            c.hidden, c.layers, c.experts, c.topk, c.moe_inter, c.shared_inter, c.vocab,
            c.hc_count, c.ple_layer, c.ple_embed_dim
        ),
        Err(e) => println!("cfg_error={e}"),
    }
}
