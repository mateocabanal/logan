use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use logan_qwen4::colisource::ColiSource;
use logan_qwen4::plan::prefix_runtime::{
    apply_max_performance_defaults, persist_prefix_boundary, restore_longest_prefix,
};
use logan_qwen4::plan::{QwenStateSnapshot, RuntimeFeatures, RuntimeStats};
use logan_qwen4::{Cfg, Model, load_cfg};
use tokenizers::Tokenizer;

/// Qwen3.8-Flash-Next's official non-thinking assistant generation prefix.
///
/// The model's chat template does not start generation directly after
/// `<|im_start|>assistant\n`. Even with thinking disabled it emits an empty
/// think block first.
const ASSISTANT_NON_THINKING_PREFIX: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

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
            top_p: 0.8,
            top_k: 20,
            repeat_penalty: 1.0,
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
    pub ram_cached_tokens: usize,
    pub ssd_cached_tokens: usize,
    pub ram_cache_restore_ms: f64,
    pub ram_cache_write_ms: f64,
    pub cache_restore_ms: f64,
    pub cache_write_ms: f64,
    pub prompt_ms: f64,
    pub first_token_ms: f64,
    pub generation_ms: f64,
    pub total_ms: f64,
    pub generated_tokens: usize,
    pub forward_tokens: usize,
    pub context_tokens: usize,
    /// Exact bytes held by reusable in-RAM prefix snapshots.
    pub hot_cache_bytes: u64,
    pub hot_cache_entries: usize,
    pub hot_cache_tokens: usize,
    /// Exact payload bytes represented by the currently active causal state.
    /// This excludes model weights and allocator/container overhead.
    pub active_state_bytes: u64,
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
    ClearHot,
    Complete {
        prompt: String,
        settings: GenerationSettings,
        updates: mpsc::Sender<CompletionUpdate>,
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
        chunk: String,
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

#[derive(Clone, Debug)]
pub enum CompletionUpdate {
    Started {
        metrics: TurnMetrics,
    },
    Token {
        chunk: String,
        token_id: u32,
        metrics: TurnMetrics,
        stats: RuntimeStats,
    },
    Done {
        text: String,
        metrics: TurnMetrics,
        stats: RuntimeStats,
    },
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
        let panic_events = event_tx.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
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
                    EngineCommand::ClearHot => {
                        worker_cancel.store(true, Ordering::Relaxed);
                        worker.hot_cache.clear();
                        let system_prompt = worker.system_prompt.clone();
                        let _ =
                            event_tx.send(EngineEvent::Loading("clearing RAM prefix cache".into()));
                        match worker.reset(system_prompt) {
                            Ok(()) => {
                                let _ = event_tx.send(EngineEvent::ResetDone);
                            }
                            Err(error) => {
                                let _ = event_tx.send(EngineEvent::Error(error));
                            }
                        }
                    }
                    EngineCommand::Complete {
                        prompt,
                        settings,
                        updates,
                    } => {
                        worker_cancel.store(false, Ordering::Relaxed);
                        if let Err(error) = worker.run_prompt(
                            &prompt,
                            &settings,
                            &worker_cancel,
                            &event_tx,
                            &updates,
                        ) {
                            let _ = updates.send(CompletionUpdate::Error(error.clone()));
                            let _ = event_tx.send(EngineEvent::Error(error));
                        }
                    }
                    EngineCommand::Shutdown => break,
                }
            }
        }));
        if let Err(payload) = result {
            let message = payload
                .downcast_ref::<&str>()
                .map(|value| (*value).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown worker panic".into());
            let _ = panic_events.send(EngineEvent::Error(format!(
                "inference worker panicked: {message}"
            )));
        }
    });

    EngineHandle {
        tx: cmd_tx,
        rx: event_rx,
        cancel,
    }
}

struct HotPrefixEntry {
    tokens: Vec<u32>,
    snapshot: QwenStateSnapshot,
    /// Logits emitted by the final cached token. Keeping these in RAM lets
    /// an exact prompt hit resume decode without replaying a boundary token.
    last_logits: Vec<f32>,
    bytes: u64,
    last_used: u64,
}

struct HotPrefixCache {
    entries: Vec<HotPrefixEntry>,
    bytes: u64,
    budget: u64,
    clock: u64,
}

struct HotRestoreStats {
    cached_tokens: usize,
    restore_ms: f64,
    last_logits: Vec<f32>,
}

struct HotWriteStats {
    write_ms: f64,
}

impl HotPrefixCache {
    fn from_env() -> Self {
        const DEFAULT_BUDGET: u64 = 512 * 1024 * 1024;
        let budget = std::env::var("LOGAN_HOT_PREFIX_CACHE_BYTES")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .or_else(|| {
                std::env::var("LOGAN_HOT_PREFIX_CACHE_MB")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .map(|mb| mb.saturating_mul(1024 * 1024))
            })
            .unwrap_or(DEFAULT_BUDGET);
        Self {
            entries: Vec::new(),
            bytes: 0,
            budget,
            clock: 0,
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
        self.clock = 0;
    }

    fn entry_count(&self) -> usize {
        self.entries.len()
    }

    fn resident_bytes(&self) -> u64 {
        self.bytes
    }

    fn cached_tokens(&self) -> usize {
        self.entries
            .iter()
            .fold(0usize, |sum, entry| sum.saturating_add(entry.tokens.len()))
    }

    fn restore_longest(
        &mut self,
        model: &mut Model,
        prompt: &[u32],
    ) -> Result<Option<HotRestoreStats>, String> {
        let Some(index) = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.tokens.len() <= prompt.len() && prompt.starts_with(&entry.tokens)
            })
            .max_by_key(|(_, entry)| entry.tokens.len())
            .map(|(index, _)| index)
        else {
            return Ok(None);
        };

        let started = Instant::now();
        model.restore_state(&self.entries[index].snapshot)?;
        let restore_ms = started.elapsed().as_secs_f64() * 1e3;
        self.clock = self.clock.wrapping_add(1);
        self.entries[index].last_used = self.clock;
        Ok(Some(HotRestoreStats {
            cached_tokens: self.entries[index].tokens.len(),
            restore_ms,
            last_logits: self.entries[index].last_logits.clone(),
        }))
    }

    fn insert(
        &mut self,
        model: &Model,
        tokens: &[u32],
        last_logits: &[f32],
    ) -> Result<Option<HotWriteStats>, String> {
        if self.budget == 0 || tokens.is_empty() || last_logits.is_empty() {
            return Ok(None);
        }
        if let Some(index) = self.entries.iter().position(|entry| entry.tokens == tokens) {
            self.clock = self.clock.wrapping_add(1);
            self.entries[index].last_used = self.clock;
            return Ok(None);
        }

        let started = Instant::now();
        let snapshot = model.snapshot_state(tokens.len())?;
        let bytes = (snapshot.payload_bytes() as u64)
            .saturating_add((last_logits.len() as u64).saturating_mul(4))
            .saturating_add((tokens.len() as u64).saturating_mul(4));
        if bytes > self.budget {
            return Ok(None);
        }

        while self.bytes.saturating_add(bytes) > self.budget && !self.entries.is_empty() {
            let lru = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(index, _)| index)
                .unwrap();
            let evicted = self.entries.swap_remove(lru);
            self.bytes = self.bytes.saturating_sub(evicted.bytes);
        }

        self.clock = self.clock.wrapping_add(1);
        self.entries.push(HotPrefixEntry {
            tokens: tokens.to_vec(),
            snapshot,
            last_logits: last_logits.to_vec(),
            bytes,
            last_used: self.clock,
        });
        self.bytes = self.bytes.saturating_add(bytes);
        Ok(Some(HotWriteStats {
            write_ms: started.elapsed().as_secs_f64() * 1e3,
        }))
    }
}

struct ChatWorker {
    package: PathBuf,
    cfg: Cfg,
    model: Model,
    zero_state: QwenStateSnapshot,
    tokenizer: Tokenizer,
    system_prompt: String,
    im_end_id: u32,
    eos_id: Option<u32>,
    /// Logical chat token history.
    tokens: Vec<u32>,
    /// Number of logical `tokens` already consumed by the model.
    consumed: usize,
    /// Next physical model position. In the corrected causal driver this
    /// advances one-for-one with consumed logical tokens.
    position: usize,
    /// Logits emitted by the most recently consumed logical token. These are
    /// the logits that predict the NEXT token; keeping them avoids replaying
    /// the final prompt token at a synthetic duplicate position.
    last_logits: Option<Vec<f32>>,
    hot_cache: HotPrefixCache,
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
        let zero_state = model.snapshot_state(0)?;
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
            zero_state,
            tokenizer,
            system_prompt,
            im_end_id,
            eos_id,
            tokens: Vec::new(),
            consumed: 0,
            position: 0,
            last_logits: None,
            hot_cache: HotPrefixCache::from_env(),
            turns: 0,
            rng: TinyRng::new(),
        })
    }

    fn reset(&mut self, system_prompt: String) -> Result<(), String> {
        apply_max_performance_defaults();
        self.restore_zero_state()?;
        self.system_prompt = system_prompt;
        self.turns = 0;
        Ok(())
    }

    fn reload_pristine(&mut self) -> Result<(), String> {
        self.restore_zero_state()
    }

    fn restore_zero_state(&mut self) -> Result<(), String> {
        self.model.restore_state(&self.zero_state)?;
        self.tokens.clear();
        self.consumed = 0;
        self.position = 0;
        self.last_logits = None;
        Ok(())
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|e| format!("tokenize: {e}"))
    }

    fn system_prefix(&self) -> String {
        render_system_prefix(&self.system_prompt)
    }

    fn first_user_suffix(&self, user: &str) -> String {
        render_first_user_suffix(user)
    }

    fn continuation_prompt(&self, user: &str) -> String {
        render_continuation_prompt(user)
    }

    fn consume_logical_until(&mut self, end: usize) -> Result<usize, String> {
        let mut forwards = 0usize;
        while self.consumed < end {
            if self.position >= self.model.context_limit() {
                return Err("context is full; use /clear before continuing".into());
            }
            let token = *self
                .tokens
                .get(self.consumed)
                .ok_or_else(|| "logical/model cursor drift".to_string())?;
            self.last_logits = Some(self.model.forward_token(token as usize, self.position));
            self.consumed += 1;
            self.position += 1;
            forwards += 1;
        }
        Ok(forwards)
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
        let live_reused = self.consumed;
        let mut forward_tokens = 0usize;

        // Consume any sampled token / turn terminator left outstanding by the
        // previous turn so the live recurrent/KV state exactly matches history.
        forward_tokens += self.consume_logical_until(self.tokens.len())?;

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
        let mut ram_cached_tokens = 0usize;
        let mut ssd_cached_tokens = 0usize;
        let mut ram_cache_restore_ms = 0.0;
        let mut ram_cache_write_ms = 0.0;
        let mut cache_restore_ms = 0.0;
        let mut cache_write_ms = 0.0;

        if self.turns == 0 && self.consumed == 0 && self.position == 0 {
            self.tokens = input_ids.clone();
            let mut try_ssd = true;
            match self
                .hot_cache
                .restore_longest(&mut self.model, &self.tokens)
            {
                Ok(Some(hit)) => {
                    ram_cached_tokens = hit.cached_tokens;
                    ram_cache_restore_ms = hit.restore_ms;
                    self.consumed = hit.cached_tokens;
                    self.position = hit.cached_tokens;
                    self.last_logits = Some(hit.last_logits);
                    try_ssd = false;
                }
                Ok(None) => {}
                Err(error) => {
                    let _ = events.send(EngineEvent::Warning(format!(
                        "RAM prefix rejected; clearing hot cache and retrying from SSD: {error}"
                    )));
                    self.hot_cache.clear();
                    self.reload_pristine()?;
                    stats_before = self.model.runtime_stats();
                }
            }

            if try_ssd {
                match restore_longest_prefix(&mut self.model, &self.tokens) {
                    Ok(Some(hit)) => {
                        ssd_cached_tokens = hit.cached_tokens;
                        cache_restore_ms = hit.restore_ms;
                        self.consumed = hit.cached_tokens;
                        self.position = hit.cached_tokens;
                        // Prefix snapshots contain causal state but no logits.
                        // The restore helpers only accept strict prefixes, so
                        // consuming the suffix repopulates last_logits.
                        self.last_logits = None;
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
            }
        } else {
            self.tokens.extend_from_slice(&input_ids);
        }

        let target_end = self.tokens.len();
        let remaining_prompt = target_end.saturating_sub(self.consumed);
        if self.position.saturating_add(remaining_prompt) >= self.model.context_limit() {
            return Err(format!(
                "prompt would exhaust context at model position {}; use /clear or a smaller history",
                self.position.saturating_add(remaining_prompt)
            ));
        }

        // First-turn system state is a high-value checkpoint. Causal state and
        // logical position are one-to-one, so the ordinary prefix format is
        // exact and reusable.
        if let Some(system_end) = system_boundary {
            if self.consumed < system_end {
                forward_tokens += self.consume_logical_until(system_end)?;
            }
            if self.consumed == system_end && self.position == self.consumed {
                match self.hot_cache.insert(
                    &self.model,
                    &self.tokens[..system_end],
                    self.last_logits.as_deref().unwrap_or(&[]),
                ) {
                    Ok(Some(write)) => ram_cache_write_ms += write.write_ms,
                    Ok(None) => {}
                    Err(error) => {
                        let _ = events.send(EngineEvent::Warning(format!(
                            "system-prefix RAM cache write failed (non-fatal): {error}"
                        )));
                    }
                }
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

        forward_tokens += self.consume_logical_until(target_end)?;
        let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1e3;
        let prompt_forward_tokens = forward_tokens;

        // Persist the exact completed prompt before generation. Unlike the old
        // synthetic-repeat driver, logical and physical positions remain equal.
        if self.position == self.consumed {
            match self.hot_cache.insert(
                &self.model,
                &self.tokens,
                self.last_logits.as_deref().unwrap_or(&[]),
            ) {
                Ok(Some(write)) => ram_cache_write_ms += write.write_ms,
                Ok(None) => {}
                Err(error) => {
                    let _ = events.send(EngineEvent::Warning(format!(
                        "prompt RAM cache write failed (non-fatal): {error}"
                    )));
                }
            }
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
        }

        let mut metrics = TurnMetrics {
            input_tokens,
            forwarded_prompt_tokens: prompt_forward_tokens,
            live_reused_tokens: live_reused,
            ram_cached_tokens,
            ssd_cached_tokens,
            ram_cache_restore_ms,
            ram_cache_write_ms,
            cache_restore_ms,
            cache_write_ms,
            prompt_ms,
            context_tokens: self.position,
            hot_cache_bytes: self.hot_cache.resident_bytes(),
            hot_cache_entries: self.hot_cache.entry_count(),
            hot_cache_tokens: self.hot_cache.cached_tokens(),
            active_state_bytes: logan_qwen4::plan::prefix_state_payload_bytes(
                &self.model,
                self.position,
            )
            .unwrap_or(0),
            ..Default::default()
        };
        let _ = events.send(EngineEvent::TurnStarted {
            metrics: metrics.clone(),
        });

        // Causal-LM decode starts from logits emitted by the FINAL prompt
        // token. Refeeding that token at prompt.len() duplicates it in the
        // recurrent/KV state and changes the model's context.
        let mut logits = self
            .last_logits
            .take()
            .ok_or_else(|| "chat prompt produced no final-token logits".to_string())?;

        let generation_t0 = Instant::now();
        let tokenizer = self.tokenizer.clone();
        let mut decode_stream = tokenizer.decode_stream(true);
        let mut text = String::new();
        let mut generated_tokens = 0usize;
        let mut first_token_seen = false;
        let mut stop_reason = StopReason::MaxTokens;

        for step in 0..settings.max_new {
            if cancel.load(Ordering::Relaxed) {
                stop_reason = StopReason::Cancelled;
                self.append_chat_closure();
                break;
            }

            let next = sample_token(&mut logits, &self.tokens, settings, &mut self.rng);

            if next == self.im_end_id {
                self.tokens.push(next);
                stop_reason = StopReason::EndOfTurn;
                break;
            }
            if self.eos_id == Some(next) {
                self.tokens.push(next);
                stop_reason = StopReason::Eos;
                break;
            }

            self.tokens.push(next);
            generated_tokens += 1;
            let chunk = decode_stream
                .step(next)
                .map_err(|e| format!("decode token {next}: {e}"))?
                .unwrap_or_default();
            text.push_str(&chunk);

            if !first_token_seen {
                metrics.first_token_ms = turn_t0.elapsed().as_secs_f64() * 1e3;
                first_token_seen = true;
            }
            metrics.generated_tokens = generated_tokens;
            metrics.generation_ms = generation_t0.elapsed().as_secs_f64() * 1e3;
            metrics.total_ms = turn_t0.elapsed().as_secs_f64() * 1e3;
            metrics.context_tokens = self.position;
            metrics.hot_cache_bytes = self.hot_cache.resident_bytes();
            metrics.hot_cache_entries = self.hot_cache.entry_count();
            metrics.hot_cache_tokens = self.hot_cache.cached_tokens();
            metrics.active_state_bytes =
                logan_qwen4::plan::prefix_state_payload_bytes(&self.model, self.position)
                    .unwrap_or(0);
            metrics.forward_tokens = forward_tokens;
            let stats = self.model.runtime_stats().delta_from(&stats_before);
            let _ = events.send(EngineEvent::Token {
                chunk,
                token_id: next,
                metrics: metrics.clone(),
                stats,
            });

            if step + 1 >= settings.max_new {
                self.append_chat_closure();
                stop_reason = StopReason::MaxTokens;
                break;
            }

            if cancel.load(Ordering::Relaxed) {
                self.append_chat_closure();
                stop_reason = StopReason::Cancelled;
                break;
            }

            if self.position >= self.model.context_limit() {
                self.append_chat_closure();
                stop_reason = StopReason::ContextFull;
                break;
            }

            // Consume the just-sampled token at its real logical/physical
            // position to obtain logits for the following token.
            logits = self.model.forward_token(next as usize, self.position);
            self.position += 1;
            self.consumed += 1;
            forward_tokens += 1;
        }

        self.turns += 1;
        metrics.generated_tokens = generated_tokens;
        metrics.generation_ms = generation_t0.elapsed().as_secs_f64() * 1e3;
        metrics.total_ms = turn_t0.elapsed().as_secs_f64() * 1e3;
        metrics.context_tokens = self.position;
        metrics.hot_cache_bytes = self.hot_cache.resident_bytes();
        metrics.hot_cache_entries = self.hot_cache.entry_count();
        metrics.hot_cache_tokens = self.hot_cache.cached_tokens();
        metrics.active_state_bytes =
            logan_qwen4::plan::prefix_state_payload_bytes(&self.model, self.position).unwrap_or(0);
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

    fn run_prompt(
        &mut self,
        prompt: &str,
        settings: &GenerationSettings,
        cancel: &AtomicBool,
        events: &mpsc::Sender<EngineEvent>,
        updates: &mpsc::Sender<CompletionUpdate>,
    ) -> Result<(), String> {
        if prompt.trim().is_empty() {
            return Err("completion prompt is empty".into());
        }

        self.restore_zero_state()?;
        let turn_t0 = Instant::now();
        let mut stats_before = self.model.runtime_stats();
        self.tokens = self.encode(prompt)?;
        let input_tokens = self.tokens.len();
        if input_tokens == 0 {
            return Err("completion prompt tokenized to zero tokens".into());
        }
        if input_tokens >= self.model.context_limit() {
            return Err(format!(
                "prompt has {input_tokens} tokens but model context is {}",
                self.model.context_limit()
            ));
        }

        let prompt_start = Instant::now();
        let mut ram_cached_tokens = 0usize;
        let mut ssd_cached_tokens = 0usize;
        let mut ram_cache_restore_ms = 0.0;
        let mut ram_cache_write_ms = 0.0;
        let mut cache_restore_ms = 0.0;
        let mut cache_write_ms = 0.0;
        let mut forward_tokens = 0usize;
        let mut try_ssd = true;

        match self
            .hot_cache
            .restore_longest(&mut self.model, &self.tokens)
        {
            Ok(Some(hit)) => {
                ram_cached_tokens = hit.cached_tokens;
                ram_cache_restore_ms = hit.restore_ms;
                self.consumed = hit.cached_tokens;
                self.position = hit.cached_tokens;
                self.last_logits = Some(hit.last_logits);
                try_ssd = false;
            }
            Ok(None) => {}
            Err(error) => {
                let _ = events.send(EngineEvent::Warning(format!(
                    "RAM prefix rejected; clearing hot cache and retrying from SSD: {error}"
                )));
                self.hot_cache.clear();
                self.restore_zero_state()?;
                self.tokens = self.encode(prompt)?;
                stats_before = self.model.runtime_stats();
            }
        }

        if try_ssd {
            match restore_longest_prefix(&mut self.model, &self.tokens) {
                Ok(Some(hit)) => {
                    ssd_cached_tokens = hit.cached_tokens;
                    cache_restore_ms = hit.restore_ms;
                    self.consumed = hit.cached_tokens;
                    self.position = hit.cached_tokens;
                    self.last_logits = None;
                }
                Ok(None) => {}
                Err(error) => {
                    let _ = events.send(EngineEvent::Warning(format!(
                        "persistent prefix rejected; replaying from zero state: {error}"
                    )));
                    self.restore_zero_state()?;
                    self.tokens = self.encode(prompt)?;
                    stats_before = self.model.runtime_stats();
                }
            }
        }

        forward_tokens += self.consume_logical_until(self.tokens.len())?;
        let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1e3;
        let prompt_forward_tokens = forward_tokens;

        if self.position == self.consumed {
            match self.hot_cache.insert(
                &self.model,
                &self.tokens,
                self.last_logits.as_deref().unwrap_or(&[]),
            ) {
                Ok(Some(write)) => ram_cache_write_ms += write.write_ms,
                Ok(None) => {}
                Err(error) => {
                    let _ = events.send(EngineEvent::Warning(format!(
                        "prompt RAM cache write failed (non-fatal): {error}"
                    )));
                }
            }
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
        }

        let mut metrics = TurnMetrics {
            input_tokens,
            forwarded_prompt_tokens: prompt_forward_tokens,
            live_reused_tokens: 0,
            ram_cached_tokens,
            ssd_cached_tokens,
            ram_cache_restore_ms,
            ram_cache_write_ms,
            cache_restore_ms,
            cache_write_ms,
            prompt_ms,
            context_tokens: self.position,
            hot_cache_bytes: self.hot_cache.resident_bytes(),
            hot_cache_entries: self.hot_cache.entry_count(),
            hot_cache_tokens: self.hot_cache.cached_tokens(),
            active_state_bytes: logan_qwen4::plan::prefix_state_payload_bytes(
                &self.model,
                self.position,
            )
            .unwrap_or(0),
            ..Default::default()
        };
        let _ = events.send(EngineEvent::TurnStarted {
            metrics: metrics.clone(),
        });
        let _ = updates.send(CompletionUpdate::Started {
            metrics: metrics.clone(),
        });

        let mut logits = self
            .last_logits
            .take()
            .ok_or_else(|| "completion prompt produced no final-token logits".to_string())?;
        let generation_t0 = Instant::now();
        let tokenizer = self.tokenizer.clone();
        let mut decode_stream = tokenizer.decode_stream(true);
        let mut text = String::new();
        let mut generated_tokens = 0usize;
        let mut first_token_seen = false;
        let mut stop_reason = StopReason::MaxTokens;

        for step in 0..settings.max_new {
            if cancel.load(Ordering::Relaxed) {
                stop_reason = StopReason::Cancelled;
                break;
            }

            let next = sample_token(&mut logits, &self.tokens, settings, &mut self.rng);
            if next == self.im_end_id {
                stop_reason = StopReason::EndOfTurn;
                break;
            }
            if self.eos_id == Some(next) {
                stop_reason = StopReason::Eos;
                break;
            }

            self.tokens.push(next);
            generated_tokens += 1;
            let chunk = decode_stream
                .step(next)
                .map_err(|e| format!("decode token {next}: {e}"))?
                .unwrap_or_default();
            text.push_str(&chunk);

            if !first_token_seen {
                metrics.first_token_ms = turn_t0.elapsed().as_secs_f64() * 1e3;
                first_token_seen = true;
            }
            metrics.generated_tokens = generated_tokens;
            metrics.generation_ms = generation_t0.elapsed().as_secs_f64() * 1e3;
            metrics.total_ms = turn_t0.elapsed().as_secs_f64() * 1e3;
            metrics.context_tokens = self.position;
            metrics.hot_cache_bytes = self.hot_cache.resident_bytes();
            metrics.hot_cache_entries = self.hot_cache.entry_count();
            metrics.hot_cache_tokens = self.hot_cache.cached_tokens();
            metrics.active_state_bytes =
                logan_qwen4::plan::prefix_state_payload_bytes(&self.model, self.position)
                    .unwrap_or(0);
            metrics.forward_tokens = forward_tokens;
            let stats = self.model.runtime_stats().delta_from(&stats_before);
            let _ = events.send(EngineEvent::Token {
                chunk: chunk.clone(),
                token_id: next,
                metrics: metrics.clone(),
                stats: stats.clone(),
            });
            let _ = updates.send(CompletionUpdate::Token {
                chunk,
                token_id: next,
                metrics: metrics.clone(),
                stats,
            });

            if step + 1 >= settings.max_new {
                stop_reason = StopReason::MaxTokens;
                break;
            }
            if cancel.load(Ordering::Relaxed) {
                stop_reason = StopReason::Cancelled;
                break;
            }
            if self.position >= self.model.context_limit() {
                stop_reason = StopReason::ContextFull;
                break;
            }

            logits = self.model.forward_token(next as usize, self.position);
            self.position += 1;
            self.consumed += 1;
            forward_tokens += 1;
        }

        metrics.generated_tokens = generated_tokens;
        metrics.generation_ms = generation_t0.elapsed().as_secs_f64() * 1e3;
        metrics.total_ms = turn_t0.elapsed().as_secs_f64() * 1e3;
        metrics.context_tokens = self.position;
        metrics.hot_cache_bytes = self.hot_cache.resident_bytes();
        metrics.hot_cache_entries = self.hot_cache.entry_count();
        metrics.hot_cache_tokens = self.hot_cache.cached_tokens();
        metrics.active_state_bytes =
            logan_qwen4::plan::prefix_state_payload_bytes(&self.model, self.position).unwrap_or(0);
        metrics.forward_tokens = forward_tokens;
        metrics.stop_reason = Some(stop_reason);
        let stats = self.model.runtime_stats().delta_from(&stats_before);
        let _ = events.send(EngineEvent::TurnDone {
            text: text.clone(),
            metrics: metrics.clone(),
            stats: stats.clone(),
        });
        let _ = updates.send(CompletionUpdate::Done {
            text,
            metrics,
            stats,
        });
        Ok(())
    }

    fn append_chat_closure(&mut self) {
        if self.tokens.last().copied() != Some(self.im_end_id) {
            self.tokens.push(self.im_end_id);
        }
    }
}

fn render_system_prefix(system: &str) -> String {
    format!("<|im_start|>system\n{system}<|im_end|>\n")
}

fn render_first_user_suffix(user: &str) -> String {
    format!("<|im_start|>user\n{user}<|im_end|>\n{ASSISTANT_NON_THINKING_PREFIX}")
}

fn render_continuation_prompt(user: &str) -> String {
    format!("\n<|im_start|>user\n{user}<|im_end|>\n{ASSISTANT_NON_THINKING_PREFIX}")
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
    let z = probs.iter().sum::<f64>().max(f64::MIN_POSITIVE);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_non_thinking_generation_prefix_is_exact() {
        assert_eq!(
            ASSISTANT_NON_THINKING_PREFIX,
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn first_turn_uses_qwen_non_thinking_template() {
        assert_eq!(
            render_system_prefix("Be concise."),
            "<|im_start|>system\nBe concise.<|im_end|>\n"
        );
        assert_eq!(
            render_first_user_suffix("Hi"),
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn continuation_uses_same_generation_prefix() {
        assert_eq!(
            render_continuation_prompt("Again"),
            "\n<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn non_thinking_sampling_defaults_match_qwen_guidance() {
        let settings = GenerationSettings::default();
        assert_eq!(settings.temperature, 0.7);
        assert_eq!(settings.top_p, 0.8);
        assert_eq!(settings.top_k, 20);
        assert_eq!(settings.repeat_penalty, 1.0);
    }

    #[test]
    fn repeat_penalty_applies_once_per_recent_token() {
        let mut logits = vec![0.0, 4.0, -4.0];
        apply_repeat_penalty(&mut logits, &[1, 1, 2, 2], 2.0);
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], -8.0);
    }

    #[test]
    fn greedy_sampling_returns_max_logit() {
        let mut logits = vec![1.0, 7.0, 3.0];
        let settings = GenerationSettings {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = TinyRng(1);
        assert_eq!(sample_token(&mut logits, &[], &settings, &mut rng), 1);
    }
}
