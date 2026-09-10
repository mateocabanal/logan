//! Bounded real-model probe: PACKAGE PROMPT TOKENS [full|skip].
//! Reports load, prefill and decode separately; prefix reuse is not used.
use std::{
    hash::{Hash, Hasher},
    path::PathBuf,
    time::Instant,
};
fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if !(3..=4).contains(&a.len()) {
        return Err("PACKAGE PROMPT TOKENS [full|skip]".into());
    }
    let p = PathBuf::from(&a[0]);
    let n: usize = a[2].parse().map_err(|_| "invalid token count")?;
    let full = match a.get(3).map(String::as_str).unwrap_or("skip") {
        "full" => true,
        "skip" => false,
        _ => return Err("mode must be full or skip".into()),
    };
    let tok =
        tokenizers::Tokenizer::from_file(p.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let enc = tok
        .encode(a[1].as_str(), false)
        .map_err(|e| e.to_string())?;
    let ids = enc.get_ids();
    if n == 0 || ids.is_empty() {
        return Err("empty request".into());
    }
    let cfg = logan_qwen4::load_cfg(&p.join("config.json"))?;
    eprintln!("prompt_ids={ids:?} topk={} full={full}", cfg.topk);
    let t = Instant::now();
    let src = logan_qwen4::colisource::ColiSource::open(&p)?;
    let mut model = logan_qwen4::Model::load_coli(&src, &cfg)?;
    eprintln!("load_ms={:.3}", t.elapsed().as_secs_f64() * 1000.0);
    let mut logits = Vec::new();
    let mut prefill_ms = 0.0;
    for (i, &id) in ids.iter().enumerate() {
        let t = Instant::now();
        if full || i + 1 == ids.len() {
            logits = model.forward_token(id as usize, i);
        } else {
            model.prefill_token(id as usize, i);
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        prefill_ms += ms;
        eprintln!("prefill_row={i} ms={ms:.3}");
    }
    let mut out = Vec::new();
    let mut decode_ms = 0.0;
    for i in 0..n {
        if logits.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite logits".into());
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for x in &logits {
            x.to_bits().hash(&mut h);
        }
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
        out.push(next);
        eprintln!("token={i} id={next} logits_hash={:016x}", h.finish());
        if i + 1 < n {
            let t = Instant::now();
            logits = model.forward_token(next as usize, ids.len() + i);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            decode_ms += ms;
            eprintln!("decode_step={i} ms={ms:.3}");
        }
    }
    println!(
        "generated_ids: {out:?}\ncontinuation: {}",
        tok.decode(&out, false).map_err(|e| e.to_string())?
    );
    eprintln!(
        "prefill_ms={prefill_ms:.3} decode_ms={decode_ms:.3} decode_steps={}",
        n - 1
    );
    model.profile_summary(n, prefill_ms + decode_ms);
    Ok(())
}
