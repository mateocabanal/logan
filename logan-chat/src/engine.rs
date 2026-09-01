use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use logan_qwen4::colisource::ColiSource;
use logan_qwen4::plan::prefix_runtime::{
    apply_max_performance_defaults, persist_prefix_boundary, restore_longest_prefix,
};
use logan_qwen4::plan::{RuntimeFeatures, RuntimeStats};
use logan_qwen4::{load_cfg, Cfg, Model};
use tokenizers::Tokenizer;

#[derive(Clone, Debug)]
pub struct GenerationSettings {
    pub max_new: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
}

impl Default for GenerationSettings {
    fn default() -> Self {
        Self {
            max_new: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.05,
        }
    }
}

#[derive(Clone, Debug)]
pub enum StopReason {
    EndOfTurn,
    Eos,
    MaxTokens,
    Cancelled,
    ContextFull,
}

impl StopReason {
    pub fn label(&self) -> &'static str {
        match self {
            Self::EndOfTurn => "end-of-turn",
            Self::Eos => "eos",
            Self::MaxTokens => "max-tokens",
            Self::Cancelled => "cancelled",
            Self::ContextFull => "context-full",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TurnMetrics {
    pub input_tokens: usize,
    pub forwarded_prompt_tokens: usize,
    pub live_reused_tokens: usize,
    pub ssd_cached_tokens: usize,
    pub cache_restore_ms: f64,
    pub cache_write_ms: f64,
    pub prompt_ms: f64,
    pub first_token_ms: f64,
    pub generation_ms: f64,
    pub total_ms: f64,
    pub generated_tokens: usize,
    pub forward_tokens: usize,
    pub context_tokens: usize,
    pub stop_reason: Option<StopReason>,
}

#[derive(Clone, Debug)]
pub enum EngineCommand {
    Send {
        text: String,
        settings: GenerationSettings,
    },
    Reset {
        system_prompt: String,
    },
    Shutdown,
}

#[derive(Clone, Debug)]
pub enum EngineEvent {
    Loading(String),
    Ready {
        model_name: String,
        context_limit: usize,
        features: RuntimeFeatures,
        cache_dir: PathBuf,
    },
    Warning(String),
    TurnStarted {
        metrics: TurnMetrics,
    },
    Token {
        text: String,
        token_id: u32,
        metrics: TurnMetrics,
        stats: RuntimeStats,
    },
    TurnDone {
        text: String,
        metrics: TurnMetrics,
        stats: RuntimeStats,
    },
    ResetDone,
    Error(String),
}

pub struct EngineHandle {
    pub tx: mpsc::Sender<EngineCommand>,
    pub rx: mpsc::Receiver<EngineEvent>,
    pub cancel: Arc<AtomicBool>,
}

pub fn spawn(package: PathBuf, system_prompt: String) -> EngineHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (event_tx, event_rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = Arc::clone(&cancel);

    std::thread::spawn(move || {
        let mut worker = match ChatWorker::load(package, system_prompt, &event_tx) {
            Ok(worker) => worker,
            Err(error) => {
                let _ = event_tx.send(EngineEvent::Error(error));
                return;
            }
        };

        while let Ok(command) = cmd_rx.recv() {
            match command {
                EngineCommand::Send { text, settings } => {
                    worker_cancel.store(false, Ordering::Relaxed);
                    if let Err(error) =
                        worker.run_turn(&text, &settings, &worker_cancel, &event_tx)
                    {
                        let _ = event_tx.send(EngineEvent::Error(error));
                    }
                }
                EngineCommand::Reset { system_prompt } => {
                    worker_cancel.store(true, Ordering::Relaxed);
                    let _ = event_tx.send(EngineEvent::Loading("resetting model state".into()));
                    match worker.reset(system_prompt) {
                        Ok(()) => {
                            let _ = event_tx.send(EngineEvent::ResetDone);
                        }
                        Err(error) => {
                            let _ = event_tx.send(EngineEvent::Error(error));
                        }
                    }
                }
                EngineCommand::Shutdown => break,
            }
        }
    });

    EngineHandle {
        tx: cmd_tx,
        rx: event_rx,
        cancel,
    }
}

struct ChatWorker {
    package: PathBuf,
    cfg: Cfg,
    model: Model,
    tokenizer: Tokenizer,
    system_prompt: String,
    im_end_id: u32,
    eos_id: Option<u32>,
    tokens: Vec<u32>,
    pending: Vec<u32>,
    position: usize,
    turns: usize,
    rng: TinyRng,
}

impl ChatWorker {
    fn load(
        package: PathBuf,
        system_prompt: String,
        events: &mpsc::Sender<EngineEvent>,
    ) -> Result<Self, String> {
        apply_max_performance_defaults();
        let _ = events.send(EngineEvent::Loading("loading tokenizer".into()));
        let tokenizer_path = package.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("load {}: {e}", tokenizer_path.display()))?;
        let im_end_id = tokenizer
            .token_to_id("<|im_end|>")
            .ok_or_else(|| "tokenizer is missing Qwen <|im_end|> token".to_string())?;

        let _ = events.send(EngineEvent::Loading("loading Qwen4 package".into()));
        let cfg = load_cfg(&package.join("config.json"))?;
        let model = load_model(&package, &cfg)?;
        let stats = model.runtime_stats();
        let eos_id = (stats.eos_token_id >= 0).then_some(stats.eos_token_id as u32);
        let model_name = package
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or("Qwen4")
            .to_string();
        let cache_dir = logan_qwen4::plan::PrefixCacheStore::from_env()
            .map(|s| s.root().to_path_buf())
            .unwrap_or_default();

        let _ = events.send(EngineEvent::Ready {
            model_name,
            context_limit: stats.context_limit,
            features: stats.features.clone(),
            cache_dir,
        });

        Ok(Self {
            package,
            cfg,
            model,
            tokenizer,
            system_prompt,
            im_end_id,
            eos_id,
            tokens: Vec::new(),
            pending: Vec::new(),
            position: 0,
            turns: 0,
            rng: TinyRng::new(),
        })
    }

    fn reset(&mut self, system_prompt: String) -> Result<(), String> {
        apply_max_performance_defaults();
        self.model = load_model(&self.package, &self.cfg)?;
        self.system_prompt = system_prompt;
        self.tokens.clear();
        self.pending.clear();
        self.position = 0;
        self.turns = 0;
        Ok(())
    }

    fn reload_pristine(&mut self) -> Result<(), String> {
        self.model = load_model(&self.package, &self.cfg)?;
        self.position = 0;
        self.pending.clear();
        Ok(())
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|e| format!("tokenize: {e}"))
    }

    fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| format!("decode: {e}"))
    }

    fn system_prefix(&self) -> String {
        format!("<|im_start|>system\n{}<|im_end|>\n", self.system_prompt)
    }

    fn first_user_suffix(&self, user: &str) -> String {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user
        )
    }

    fn continuation_prompt(&self, user: &str) -> String {
        format!(
            "\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user
        )
    }

    fn run_turn(
        &mut self,
        user: &str,
        settings: &GenerationSettings,
        cancel: &AtomicBool,
        events: &mpsc::Sender<EngineEvent>,
    ) -> Result<(), String> {
        if user.trim().is_empty() {
            return Ok(());
        }
        let turn_t0 = Instant::now();
        let mut stats_before = self.model.runtime_stats();
        let live_reused = self.position;
        let mut forward_tokens = 0usize;

        // Consume the final generated token / chat terminator from the previous
        // turn only when another turn actually begins. This removes a full
        // model forward from end-of-response latency while keeping live state
        // causal when the conversation continues.
        for token in std::mem::take(&mut self.pending) {
            if self.position >= self.model.context_limit() {
                return Err("context is full; use /clear before continuing".into());
            }
            self.model.forward_token(token as usize, self.position);
            self.position += 1;
            forward_tokens += 1;
        }

        let (input_ids, system_boundary) = if self.turns == 0 {
            let system_ids = self.encode(&self.system_prefix())?;
            let mut all = system_ids.clone();
            all.extend(self.encode(&self.first_user_suffix(user))?);
            (all, Some(system_ids.len()))
        } else {
            (self.encode(&self.continuation_prompt(user))?, None)
        };
        let input_tokens = input_ids.len();

        let prompt_start = Instant::now();
        let mut cached_tokens = 0usize;
        let mut cache_restore_ms = 0.0;
        let mut cache_write_ms = 0.0;

        if self.turns == 0 && self.position == 0 {
            self.tokens = input_ids.clone();
            match restore_longest_prefix(&mut self.model, &self.tokens) {
                Ok(Some(hit)) => {
                    cached_tokens = hit.cached_tokens;
                    cache_restore_ms = hit.restore_ms;
                    self.position = hit.cached_tokens;
                }
                Ok(None) => {}
                Err(error) => {
                    let _ = events.send(EngineEvent::Warning(format!(
                        "persistent prefix rejected; replaying from a fresh model: {error}"
                    )));
                    self.reload_pristine()?;
                    stats_before = self.model.runtime_stats();
                }
            }
        } else {
            self.tokens.extend_from_slice(&input_ids);
        }

        let target_end = self.tokens.len();
        if target_end > self.model.context_limit() {
            return Err(format!(
                "context {} exceeds model limit {}; use /clear or a smaller history",
                target_end,
                self.model.context_limit()
            ));
        }

        // First-turn system state is a high-value semantic checkpoint shared by
        // every new chat using the same system prompt. If SSD restore already
        // passed this boundary there is nothing to do; otherwise stop exactly
        // at it, persist, then continue into the user suffix.
        if let Some(system_end) = system_boundary {
            while self.position < system_end {
                let token = self.tokens[self.position];
                self.model.forward_token(token as usize, self.position);
                self.position += 1;
                forward_tokens += 1;
            }
            if self.position == system_end {
                match persist_prefix_boundary(&self.model, &self.tokens[..system_end]) {
                    Ok(Some(write)) if !write.already_existed => {
                        cache_write_ms += write.elapsed.as_secs_f64() * 1e3;
                    }
                    Ok(_) => {}
                    Err(error) => {
                        let _ = events.send(EngineEvent::Warning(format!(
                            "system-prefix cache write failed (non-fatal): {error}"
                        )));
                    }
                }
            }
        }

        let mut logits = None;
        while self.position < target_end {
            let token = self.tokens[self.position];
            logits = Some(self.model.forward_token(token as usize, self.position));
            self.position += 1;
            forward_tokens += 1;
        }
        let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1e3;

        match persist_prefix_boundary(&self.model, &self.tokens) {
            Ok(Some(write)) if !write.already_existed => {
                cache_write_ms += write.elapsed.as_secs_f64() * 1e3;
            }
            Ok(_) => {}
            Err(error) => {
                let _ = events.send(EngineEvent::Warning(format!(
                    "prefix cache write failed (non-fatal): {error}"
                )));
            }
        }

        let mut metrics = TurnMetrics {
            input_tokens,
            forwarded_prompt_tokens: forward_tokens,
            live_reused_tokens: live_reused,
            ssd_cached_tokens: cached_tokens,
            cache_restore_ms,
            cache_write_ms,
            prompt_ms,
            context_tokens: self.tokens.len(),
            ..Default::default()
        };
        let _ = events.send(EngineEvent::TurnStarted {
            metrics: metrics.clone(),
        });

        let Some(mut logits) = logits else {
            return Err("chat prompt produced no logits".into());
        };
        let generation_t0 = Instant::now();
        let mut generated = Vec::with_capacity(settings.max_new);
        let mut text = String::new();
        let mut first_token_seen = false;
        let mut stop_reason = StopReason::MaxTokens;

        for step in 0..settings.max_new {
            // Sampling another visible token is not useful if there is no
            // context slot left to consume it on a future step/turn.
            if self.position >= self.model.context_limit() {
                stop_reason = StopReason::ContextFull;
                break;
            }
            if cancel.load(Ordering::Relaxed) {
                stop_reason = StopReason::Cancelled;
                self.append_chat_closure();
                break;
            }

            let next = sample_token(&mut logits, &self.tokens, settings, &mut self.rng);

            if next == self.im_end_id {
                self.tokens.push(next);
                self.pending.push(next);
                stop_reason = StopReason::EndOfTurn;
                break;
            }
            if self.eos_id == Some(next) {
                self.tokens.push(next);
                self.pending.push(next);
                stop_reason = StopReason::Eos;
                break;
            }

            generated.push(next);
            self.tokens.push(next);
            text = self.decode(&generated)?;

            if !first_token_seen {
                metrics.first_token_ms = turn_t0.elapsed().as_secs_f64() * 1e3;
                first_token_seen = true;
            }
            metrics.generated_tokens = generated.len();
            metrics.generation_ms = generation_t0.elapsed().as_secs_f64() * 1e3;
            metrics.total_ms = turn_t0.elapsed().as_secs_f64() * 1e3;
            metrics.context_tokens = self.tokens.len();
            metrics.forward_tokens = forward_tokens;
            let stats = self.model.runtime_stats().delta_from(&stats_before);
            let _ = events.send(EngineEvent::Token {
                text: text.clone(),
                token_id: next,
                metrics: metrics.clone(),
                stats,
            });

            if step + 1 >= settings.max_new {
                self.pending.push(next);
                self.append_chat_closure();
                stop_reason = StopReason::MaxTokens;
                break;
            }

            if cancel.load(Ordering::Relaxed) {
                self.pending.push(next);
                self.append_chat_closure();
                stop_reason = StopReason::Cancelled;
                break;
            }

            logits = self.model.forward_token(next as usize, self.position);
            self.position += 1;
            forward_tokens += 1;
        }

        self.turns += 1;
        metrics.generated_tokens = generated.len();
        metrics.generation_ms = generation_t0.elapsed().as_secs_f64() * 1e3;
        metrics.total_ms = turn_t0.elapsed().as_secs_f64() * 1e3;
        metrics.context_tokens = self.tokens.len();
        metrics.forward_tokens = forward_tokens;
        metrics.stop_reason = Some(stop_reason);
        let stats = self.model.runtime_stats().delta_from(&stats_before);
        let _ = events.send(EngineEvent::TurnDone {
            text,
            metrics,
            stats,
        });
        Ok(())
    }

    fn append_chat_closure(&mut self) {
        if self.tokens.last().copied() != Some(self.im_end_id) {
            self.tokens.push(self.im_end_id);
            self.pending.push(self.im_end_id);
        }
    }
}

fn load_model(package: &Path, cfg: &Cfg) -> Result<Model, String> {
    let src = ColiSource::open(package)?;
    Model::load_coli(&src, cfg)
}

fn apply_repeat_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    if penalty <= 1.0 {
        return;
    }
    let start = history.len().saturating_sub(256);
    let mut seen = HashSet::with_capacity(history.len() - start);
    for &token in &history[start..] {
        if !seen.insert(token) {
            continue;
        }
        let Some(logit) = logits.get_mut(token as usize) else {
            continue;
        };
        if *logit >= 0.0 {
            *logit /= penalty;
        } else {
            *logit *= penalty;
        }
    }
}

fn sample_token(
    logits: &mut [f32],
    history: &[u32],
    settings: &GenerationSettings,
    rng: &mut TinyRng,
) -> u32 {
    apply_repeat_penalty(logits, history, settings.repeat_penalty.max(1.0));

    if settings.temperature <= 0.001 || settings.top_k == 1 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    let temperature = settings.temperature.max(0.01);
    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (i, v / temperature))
        .collect();

    let top_k = if settings.top_k == 0 {
        candidates.len()
    } else {
        settings.top_k.min(candidates.len())
    };
    if top_k < candidates.len() {
        candidates.select_nth_unstable_by(top_k - 1, |a, b| b.1.total_cmp(&a.1));
        candidates.truncate(top_k);
    }
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let max_logit = candidates.first().map(|x| x.1).unwrap_or(0.0);
    let mut probs: Vec<f64> = candidates
        .iter()
        .map(|(_, logit)| ((*logit - max_logit) as f64).exp())
        .collect();
    let z: f64 = probs.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    for p in &mut probs {
        *p /= z;
    }

    let top_p = settings.top_p.clamp(0.01, 1.0) as f64;
    let mut keep = probs.len();
    if top_p < 0.999_999 {
        let mut cumulative = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cumulative += *p;
            if cumulative >= top_p {
                keep = i + 1;
                break;
            }
        }
    }
    candidates.truncate(keep.max(1));
    probs.truncate(keep.max(1));
    let kept_z = probs.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    let mut needle = rng.next_f64() * kept_z;
    for ((token, _), p) in candidates.iter().zip(&probs) {
        if needle <= *p {
            return *token as u32;
        }
        needle -= *p;
    }
    candidates.last().map(|v| v.0 as u32).unwrap_or(0)
}

struct TinyRng(u64);

impl TinyRng {
    fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
            ^ (std::process::id() as u64).rotate_left(17);
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}
