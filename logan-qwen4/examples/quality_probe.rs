//! Teacher/student quality harness for authoritative predictive routing.
//!
//! `cargo run --release -p logan-qwen4 --example quality_probe -- MODEL_DIR [--tokens N] [--corpus FILE]`
//!
//! Runs the SAME checkpoint twice over the SAME token stream:
//!
//! - **teacher**: native router. `forward_token` gives the true next-token
//!   distribution, and the reference token stream is the teacher's own greedy
//!   continuation.
//! - **student**: authoritative RouteScout (`QWEN_ROUTE_AUTHORITATIVE=1`),
//!   teacher-forced on the teacher's stream so the comparison is per-position
//!   and not a divergence cascade. The first position where the two disagree is
//!   reported separately as the generation divergence point.
//!
//! For every student position this reports, against the teacher's distribution
//! at the same position: top-1 agreement, top-10 overlap, KL(teacher||student),
//! teacher cross-entropy, and logit cosine similarity. Those are the numbers the
//! mission asks for before any recovery training is considered, so the cost of
//! authoritative routing is quantified instead of eyeballed.
//!
//! The two models must be built in one process (the checkpoint is 20 GB on this
//! host and a second load would thrash swap, which would also corrupt the
//! timing). Native and authoritative routing are selected per model at
//! construction, so the student's environment variables are set before its load
//! and unset before the teacher's.

use std::path::PathBuf;

use logan_qwen4::{Cfg, Model, StFile};

/// The corpus is a small fixed set of prompts so the run is reproducible and the
/// result is a distribution over several different routing regimes rather than
/// one prompt's idiosyncrasies.
const DEFAULT_CORPUS: &[&str] = &[
    "Explain why memory safety matters in systems programming and how Rust achieves it without a garbage collector.",
    "Write a short explanation of how mixture-of-experts routing works in a transformer language model.",
    "Describe the tradeoffs between reading model weights from an SSD and keeping them resident in unified memory.",
    "What is the difference between speculative decoding and speculative prefetching in LLM inference?",
];

fn render_prompt(user: &str) -> String {
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exps: Vec<f64> = logits.iter().map(|&v| ((v as f64) - max).exp()).collect();
    let z: f64 = exps.iter().sum();
    exps.into_iter().map(|v| v / z).collect()
}

fn topk(logits: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    idx.truncate(k);
    idx
}

/// KL(teacher || student) — the direction that penalizes missing teacher mass.
fn kl_divergence(teacher: &[f32], student: &[f32]) -> f64 {
    let t = softmax(teacher);
    let s = softmax(student);
    t.iter()
        .zip(s.iter())
        .zip(teacher.iter())
        .filter(|((&p, _), &l)| p > 0.0 && l.is_finite())
        .map(|((&p, &q), _)| p * (p / q.max(f64::MIN_POSITIVE)).ln())
        .sum()
}

fn cross_entropy(teacher: &[f32], student: &[f32], target: usize) -> f64 {
    let s = softmax(student);
    -(s.get(target).copied().unwrap_or(0.0).max(f64::MIN_POSITIVE)).ln()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    if na <= 0.0 || nb <= 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

struct Accum {
    positions: u64,
    top1: u64,
    top10_overlap: u64,
    kl: f64,
    ce: f64,
    cosine: f64,
    top1_margin_logit: f64,
    first_divergence: Option<usize>,
    divergence_tokens: Vec<usize>,
    divergences: u64,
}

impl Accum {
    fn new() -> Self {
        Self {
            positions: 0,
            top1: 0,
            top10_overlap: 0,
            kl: 0.0,
            ce: 0.0,
            cosine: 0.0,
            top1_margin_logit: 0.0,
            first_divergence: None,
            divergence_tokens: Vec::new(),
            divergences: 0,
        }
    }
    fn finish(&self) -> String {
        let n = self.positions.max(1) as f64;
        format!(
            "positions={} top1_agree={:.4} top10_overlap={:.4} kl={:.5} \
             teacher_ce={:.5} logit_cosine={:.6} mean_top1_margin={:.3} \
             divergence_point={:?} divergences={}",
            self.positions,
            self.top1 as f64 / n,
            self.top10_overlap as f64 / n,
            self.kl / n,
            self.ce / n,
            self.cosine / n,
            self.top1_margin_logit / n,
            self.first_divergence,
            self.divergences,
        )
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err(
            "usage: quality_probe MODEL_DIR [--tokens N] [--prompt TEXT ...] [--control|--edge0]".into(),
        );
    }
    let dir = PathBuf::from(&args[0]);
    let mut tokens = 32usize;
    let mut prompts: Vec<String> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--tokens" => {
                tokens = args
                    .get(i + 1)
                    .ok_or("--tokens needs a value")?
                    .parse()
                    .map_err(|_| "--tokens must be an integer")?;
                i += 2;
            }
            "--prompt" => {
                prompts.push(args.get(i + 1).ok_or("--prompt needs a value")?.clone());
                i += 2;
            }
            "--control" | "--edge0" => {
                i += 1;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if prompts.is_empty() {
        prompts = DEFAULT_CORPUS.iter().map(|s| s.to_string()).collect();
    }

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| e.to_string())?;
    let cfg: Cfg = logan_qwen4::load_cfg(&dir.join("config.json"))?;
    let st = StFile::open_dir(&dir)?;

    // Construct one student mode, then clear its process-level selector before
    // constructing the native teacher. Model snapshots RouteMode at load time.
    let control = args.iter().any(|a| a == "--control");
    let edge0 = args.iter().any(|a| a == "--edge0");
    if control && edge0 {
        return Err("--control and --edge0 are mutually exclusive".into());
    }
    let student_name = if edge0 {
        std::env::set_var("QWEN_ROUTE_MODE", "edge0");
        // Quality measurement must not be polluted by speculative storage I/O.
        std::env::set_var("QWEN_EDGE0_PREFETCH", "0");
        "edge0 pretrained prerouter"
    } else if control {
        std::env::set_var(
            "QWEN_ROUTE_NATIVE_K",
            std::env::var("QWEN_ROUTE_AUTHORITATIVE_K").unwrap_or_else(|_| "0".into()),
        );
        "native-truncated control"
    } else {
        std::env::set_var("QWEN_ROUTE_AUTHORITATIVE", "1");
        "authoritative RouteScout"
    };
    if std::env::var("QWEN_ROUTE_AUTHORITATIVE_K").is_err() {
        std::env::set_var("QWEN_ROUTE_AUTHORITATIVE_K", "0");
    }
    eprintln!("quality-probe: loading STUDENT ({student_name})");
    let mut student = Model::load(&st, &cfg)?;
    std::env::remove_var("QWEN_ROUTE_MODE");
    std::env::remove_var("QWEN_EDGE0_PREFETCH");
    std::env::remove_var("QWEN_ROUTE_AUTHORITATIVE");
    std::env::remove_var("QWEN_ROUTE_NATIVE_K");

    eprintln!("quality-probe: loading TEACHER (native)");
    let mut teacher = Model::load(&st, &cfg)?;

    let mut total = Accum::new();
    for (pi, prompt) in prompts.iter().enumerate() {
        let text = render_prompt(prompt);
        let enc = tok.encode(text.as_str(), false).map_err(|e| e.to_string())?;
        let ids: Vec<u32> = enc.get_ids().to_vec();
        if ids.is_empty() {
            continue;
        }
        // A fresh sequence per prompt: recurrent state must not leak across
        // samples (the engine has reset_sequence_state for exactly this).
        teacher.reset_sequence_state();
        student.reset_sequence_state();

        for (k, &t) in ids.iter().enumerate() {
            let last = k + 1 == ids.len();
            if last {
                break;
            }
            teacher.prefill_token(t as usize, k);
            student.prefill_token(t as usize, k);
        }

        let mut teacher_logits = teacher.forward_token(*ids.last().unwrap() as usize, ids.len() - 1);
        teacher.begin_decode_measurement();
        // Store the teacher's distribution at every position up front. Re-deriving
        // it by re-running forwards would both be O(n^2) and mutate the teacher's
        // recurrent state, corrupting the reference trajectory.
        let mut teacher_trace: Vec<Vec<f32>> = Vec::with_capacity(tokens);
        let mut stream: Vec<usize> = Vec::with_capacity(tokens);
        for step in 0..tokens {
            let next = argmax(&teacher_logits);
            teacher_trace.push(teacher_logits.clone());
            stream.push(next);
            if step + 1 < tokens {
                teacher_logits = teacher.forward_token(next, ids.len() + step);
            }
        }

        // Student teacher-forced over the teacher's stream, so a single
        // disagreement cannot cascade and mask the per-position quality.
        let mut student_logits =
            student.forward_token(*ids.last().unwrap() as usize, ids.len() - 1);
        student.begin_decode_measurement();
        let mut acc = Accum::new();
        for (step, &target) in stream.iter().enumerate() {
            let tref = &teacher_trace[step];
            let t_top = topk(tref, 10);
            let s_top = topk(&student_logits, 10);
            let t1 = argmax(tref);
            let s1 = argmax(&student_logits);
            if t1 == s1 {
                acc.top1 += 1;
            } else {
                acc.divergences += 1;
                acc.divergence_tokens.push(target);
                if acc.first_divergence.is_none() {
                    acc.first_divergence = Some(step);
                }

            }
            acc.top10_overlap += t_top
                .iter()
                .filter(|e| s_top.contains(e))
                .count() as u64;
            acc.kl += kl_divergence(tref, &student_logits);
            acc.ce += cross_entropy(tref, &student_logits, target);
            acc.cosine += cosine(tref, &student_logits);
            let mut sorted = tref.clone();
            sorted.sort_unstable_by(|a, b| b.total_cmp(a));
            acc.top1_margin_logit +=
                (sorted.first().copied().unwrap_or(0.0) - sorted.get(1).copied().unwrap_or(0.0))
                    as f64;
            acc.positions += 1;
            if step + 1 < tokens {
                student_logits = student.forward_token(target, ids.len() + step);
            }
        }
        eprintln!("quality-probe: prompt {pi} student {acc:?}", acc = acc.finish());
        total.positions += acc.positions;
        total.top1 += acc.top1;
        total.top10_overlap += acc.top10_overlap;
        total.kl += acc.kl;
        total.ce += acc.ce;
        total.cosine += acc.cosine;
        total.top1_margin_logit += acc.top1_margin_logit;
        total.divergences += acc.divergences;
        if total.first_divergence.is_none() {
            total.first_divergence = acc.first_divergence;
        }
        eprintln!(
            "quality-probe: prompt {pi} teacher stream sha={}",
            sha8(&stream)
        );
    }
    println!("QUALITY {}", total.finish());
    Ok(())
}

fn sha8(v: &[usize]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    v.hash(&mut h);
    format!("{:016x}", h.finish())
}
