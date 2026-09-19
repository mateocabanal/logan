//! Visible-text greedy smoke check for the shared Qwen engine.
//! cargo run --release -p logan-chat --example greedy_smoke -- PACKAGE PROMPT [TOKENS]
use std::path::PathBuf;
fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err("usage: greedy_smoke PACKAGE PROMPT [TOKENS]".into());
    }
    let package = PathBuf::from(&args[0]);
    let count: usize = args
        .get(2)
        .map(|s| s.parse())
        .transpose()
        .map_err(|_| "TOKENS must be an integer")?
        .unwrap_or(16);
    let tokenizer = tokenizers::Tokenizer::from_file(package.join("tokenizer.json"))
        .map_err(|e| e.to_string())?;
    let prompt = tokenizer
        .encode(args[1].as_str(), false)
        .map_err(|e| e.to_string())?;
    if prompt.is_empty() || count == 0 {
        return Err("prompt and token count must be nonempty".into());
    }
    let cfg = logan_qwen4::load_cfg(&package.join("config.json"))?;
    println!(
        "prompt: {}\nprompt_ids: {:?}\nmodel_topk: {}",
        args[1],
        prompt.get_ids(),
        cfg.topk
    );
    let out = logan_qwen4::run_greedy(&package, prompt.get_ids(), count)?;
    println!(
        "generated_ids: {out:?}\ncontinuation: {}",
        tokenizer.decode(&out, false).map_err(|e| e.to_string())?
    );
    Ok(())
}
