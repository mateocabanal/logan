use std::fs;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use logan_qwen4::plan::{RuntimeFeatures, RuntimeStats};

use crate::engine::{
    EngineCommand, EngineEvent, GenerationSettings, StopReason, TurnMetrics,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    Notice,
}

#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug)]
pub enum UiAction {
    None,
    Send(EngineCommand),
    Quit,
}

pub struct App {
    pub model_name: String,
    pub status: String,
    pub loaded: bool,
    pub generating: bool,
    pub resetting: bool,
    pub messages: Vec<Message>,
    pub input: String,
    pub cursor: usize,
    pub input_history: Vec<String>,
    pub history_index: Option<usize>,
    pub transcript_scroll: u16,
    pub show_stats: bool,
    pub show_help: bool,
    pub system_prompt: String,
    pub settings: GenerationSettings,
    pub context_limit: usize,
    pub features: RuntimeFeatures,
    pub last_metrics: TurnMetrics,
    pub live_metrics: TurnMetrics,
    pub last_stats: RuntimeStats,
    pub last_token_id: Option<u32>,
    pub cache_dir: PathBuf,
    pub cache_entries: usize,
    pub cache_bytes: u64,
    pub peak_rss_bytes: u64,
    pub cancel: Arc<std::sync::atomic::AtomicBool>,
}

impl App {
    pub fn new(system_prompt: String, cancel: Arc<std::sync::atomic::AtomicBool>) -> Self {
        Self {
            model_name: "Qwen4".into(),
            status: "starting inference worker".into(),
            loaded: false,
            generating: false,
            resetting: false,
            messages: Vec::new(),
            input: String::new(),
            cursor: 0,
            input_history: Vec::new(),
            history_index: None,
            transcript_scroll: 0,
            show_stats: true,
            show_help: false,
            system_prompt,
            settings: GenerationSettings::default(),
            context_limit: 0,
            features: RuntimeFeatures::default(),
            last_metrics: TurnMetrics::default(),
            live_metrics: TurnMetrics::default(),
            last_stats: RuntimeStats::default(),
            last_token_id: None,
            cache_dir: PathBuf::new(),
            cache_entries: 0,
            cache_bytes: 0,
            peak_rss_bytes: 0,
            cancel,
        }
    }

    pub fn on_engine_event(&mut self, event: EngineEvent) {
        match event {
            EngineEvent::Loading(stage) => {
                self.status = stage;
                self.loaded = false;
            }
            EngineEvent::Ready {
                model_name,
                context_limit,
                features,
                cache_dir,
            } => {
                self.model_name = model_name;
                self.context_limit = context_limit;
                self.features = features;
                self.cache_dir = cache_dir;
                self.loaded = true;
                self.status = "ready".into();
                self.refresh_cache_usage();
            }
            EngineEvent::Warning(message) => {
                self.messages.push(Message {
                    role: Role::Notice,
                    content: message.clone(),
                });
                self.status = message;
            }
            EngineEvent::TurnStarted { metrics } => {
                self.live_metrics = metrics;
                self.generating = true;
                self.status = "generating — Esc cancels".into();
            }
            EngineEvent::Token {
                text,
                token_id,
                metrics,
                stats,
            } => {
                self.last_token_id = Some(token_id);
                self.live_metrics = metrics;
                self.last_stats = stats;
                self.set_streaming_assistant(text);
                self.status = format!(
                    "generating {} tokens · {:.2} tok/s",
                    self.live_metrics.generated_tokens,
                    generation_rate(&self.live_metrics)
                );
            }
            EngineEvent::TurnDone {
                text,
                metrics,
                stats,
            } => {
                self.set_streaming_assistant(text);
                self.last_metrics = metrics.clone();
                self.live_metrics = metrics;
                self.last_stats = stats;
                self.generating = false;
                self.cancel.store(false, Ordering::Relaxed);
                self.status = format!(
                    "ready · {} · {:.2} tok/s · {:.1}s",
                    self.last_metrics
                        .stop_reason
                        .as_ref()
                        .map(StopReason::label)
                        .unwrap_or("done"),
                    generation_rate(&self.last_metrics),
                    self.last_metrics.total_ms / 1e3,
                );
                self.refresh_cache_usage();
            }
            EngineEvent::ResetDone => {
                self.loaded = true;
                self.resetting = false;
                self.generating = false;
                self.last_metrics = TurnMetrics::default();
                self.live_metrics = TurnMetrics::default();
                self.last_stats = RuntimeStats::default();
                self.last_token_id = None;
                self.status = "new session ready".into();
            }
            EngineEvent::Error(error) => {
                self.messages.push(Message {
                    role: Role::Notice,
                    content: format!("error: {error}"),
                });
                self.generating = false;
                self.resetting = false;
                self.status = format!("error: {error}");
            }
        }
    }

    fn set_streaming_assistant(&mut self, text: String) {
        if let Some(last) = self.messages.last_mut() {
            if last.role == Role::Assistant {
                last.content = text;
                return;
            }
        }
        self.messages.push(Message {
            role: Role::Assistant,
            content: text,
        });
    }

    pub fn submit(&mut self) -> UiAction {
        if !self.loaded || self.generating || self.resetting {
            return UiAction::None;
        }
        let raw = self.input.trim_end().to_string();
        if raw.trim().is_empty() {
            return UiAction::None;
        }
        self.input_history.push(raw.clone());
        self.history_index = None;
        self.clear_input();

        if let Some(command) = raw.strip_prefix('/') {
            return self.run_command(command.trim());
        }

        self.messages.push(Message {
            role: Role::User,
            content: raw.clone(),
        });
        self.messages.push(Message {
            role: Role::Assistant,
            content: String::new(),
        });
        self.transcript_scroll = 0;
        self.generating = true;
        self.status = "prefilling prompt".into();
        UiAction::Send(EngineCommand::Send {
            text: raw,
            settings: self.settings.clone(),
        })
    }

    fn run_command(&mut self, command: &str) -> UiAction {
        let (name, arg) = command
            .split_once(char::is_whitespace)
            .map(|(a, b)| (a, b.trim()))
            .unwrap_or((command, ""));
        match name {
            "q" | "quit" | "exit" => UiAction::Quit,
            "help" | "?" => {
                self.show_help = true;
                UiAction::None
            }
            "stats" => {
                self.show_stats = !self.show_stats;
                UiAction::None
            }
            "clear" | "new" => {
                self.messages.clear();
                self.transcript_scroll = 0;
                self.resetting = true;
                self.status = "resetting model state".into();
                UiAction::Send(EngineCommand::Reset {
                    system_prompt: self.system_prompt.clone(),
                })
            }
            "system" if !arg.is_empty() => {
                self.system_prompt = arg.to_string();
                self.messages.clear();
                self.resetting = true;
                self.status = "reloading with new system prompt".into();
                UiAction::Send(EngineCommand::Reset {
                    system_prompt: self.system_prompt.clone(),
                })
            }
            "max" => self.set_usize("max tokens", arg, 1, 4096, |s, v| s.max_new = v),
            "top-k" | "topk" => {
                self.set_usize("top-k", arg, 0, 4096, |s, v| s.top_k = v)
            }
            "temp" | "temperature" => {
                self.set_f32("temperature", arg, 0.0, 5.0, |s, v| s.temperature = v)
            }
            "top-p" | "topp" => self.set_f32("top-p", arg, 0.01, 1.0, |s, v| s.top_p = v),
            "repeat" => self.set_f32("repeat penalty", arg, 1.0, 2.0, |s, v| {
                s.repeat_penalty = v
            }),
            "greedy" => {
                self.settings.temperature = 0.0;
                self.settings.top_k = 1;
                self.notice("sampling: greedy");
                UiAction::None
            }
            "save" => {
                let path = if arg.is_empty() { "logan-chat.txt" } else { arg };
                match self.save_transcript(path) {
                    Ok(()) => self.notice(&format!("saved transcript to {path}")),
                    Err(e) => self.notice(&format!("save failed: {e}")),
                }
                UiAction::None
            }
            _ => {
                self.notice(&format!("unknown command /{name}; /help lists commands"));
                UiAction::None
            }
        }
    }

    fn set_usize(
        &mut self,
        label: &str,
        arg: &str,
        min: usize,
        max: usize,
        set: impl FnOnce(&mut GenerationSettings, usize),
    ) -> UiAction {
        match arg.parse::<usize>() {
            Ok(v) if (min..=max).contains(&v) => {
                set(&mut self.settings, v);
                self.notice(&format!("{label}: {v}"));
            }
            _ => self.notice(&format!("/{label} expects {min}..={max}")),
        }
        UiAction::None
    }

    fn set_f32(
        &mut self,
        label: &str,
        arg: &str,
        min: f32,
        max: f32,
        set: impl FnOnce(&mut GenerationSettings, f32),
    ) -> UiAction {
        match arg.parse::<f32>() {
            Ok(v) if v.is_finite() && v >= min && v <= max => {
                set(&mut self.settings, v);
                self.notice(&format!("{label}: {v:.3}"));
            }
            _ => self.notice(&format!("{label} expects {min:.2}..={max:.2}")),
        }
        UiAction::None
    }

    fn notice(&mut self, text: &str) {
        self.messages.push(Message {
            role: Role::Notice,
            content: text.to_string(),
        });
        self.status = text.to_string();
    }

    fn save_transcript(&self, path: &str) -> Result<(), String> {
        let mut out = String::new();
        for message in &self.messages {
            match message.role {
                Role::User => out.push_str("User:\n"),
                Role::Assistant => out.push_str("Assistant:\n"),
                Role::Notice => out.push_str("[Logan]\n"),
            }
            out.push_str(&message.content);
            out.push_str("\n\n");
        }
        fs::write(path, out).map_err(|e| e.to_string())
    }

    pub fn clear_input(&mut self) {
        self.input.clear();
        self.cursor = 0;
    }

    pub fn insert_char(&mut self, ch: char) {
        self.input.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    pub fn insert_str(&mut self, text: &str) {
        self.input.insert_str(self.cursor, text);
        self.cursor += text.len();
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = self.input[..self.cursor]
            .char_indices()
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.input.drain(prev..self.cursor);
        self.cursor = prev;
    }

    pub fn delete(&mut self) {
        if self.cursor >= self.input.len() {
            return;
        }
        let next = self.input[self.cursor..]
            .char_indices()
            .nth(1)
            .map(|(i, _)| self.cursor + i)
            .unwrap_or(self.input.len());
        self.input.drain(self.cursor..next);
    }

    pub fn move_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.cursor = self.input[..self.cursor]
            .char_indices()
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    pub fn move_right(&mut self) {
        if self.cursor >= self.input.len() {
            return;
        }
        self.cursor = self.input[self.cursor..]
            .char_indices()
            .nth(1)
            .map(|(i, _)| self.cursor + i)
            .unwrap_or(self.input.len());
    }

    pub fn delete_word(&mut self) {
        while self.cursor > 0
            && self.input[..self.cursor]
                .chars()
                .last()
                .is_some_and(char::is_whitespace)
        {
            self.backspace();
        }
        while self.cursor > 0
            && self.input[..self.cursor]
                .chars()
                .last()
                .is_some_and(|c| !c.is_whitespace())
        {
            self.backspace();
        }
    }

    pub fn history_prev(&mut self) {
        if self.input_history.is_empty() {
            return;
        }
        let index = match self.history_index {
            None => self.input_history.len() - 1,
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.history_index = Some(index);
        self.input = self.input_history[index].clone();
        self.cursor = self.input.len();
    }

    pub fn history_next(&mut self) {
        let Some(index) = self.history_index else {
            return;
        };
        if index + 1 >= self.input_history.len() {
            self.history_index = None;
            self.clear_input();
        } else {
            self.history_index = Some(index + 1);
            self.input = self.input_history[index + 1].clone();
            self.cursor = self.input.len();
        }
    }

    pub fn cancel_generation(&mut self) {
        if self.generating {
            self.cancel.store(true, Ordering::Relaxed);
            self.status = "cancelling after current token".into();
        }
    }

    pub fn refresh_cache_usage(&mut self) {
        let (entries, bytes) = scan_lpfx(&self.cache_dir);
        self.cache_entries = entries;
        self.cache_bytes = bytes;
        self.peak_rss_bytes = peak_rss_bytes();
    }
}

pub fn generation_rate(metrics: &TurnMetrics) -> f64 {
    if metrics.generated_tokens == 0 || metrics.generation_ms <= 0.0 {
        0.0
    } else {
        metrics.generated_tokens as f64 / (metrics.generation_ms / 1e3)
    }
}

fn scan_lpfx(root: &PathBuf) -> (usize, u64) {
    if root.as_os_str().is_empty() {
        return (0, 0);
    }
    let mut stack = vec![root.clone()];
    let mut entries = 0usize;
    let mut bytes = 0u64;
    while let Some(dir) = stack.pop() {
        let Ok(read) = fs::read_dir(dir) else {
            continue;
        };
        for entry in read.flatten() {
            let path = entry.path();
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if meta.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|v| v.to_str()) == Some("lpfx") {
                entries += 1;
                bytes = bytes.saturating_add(meta.len());
            }
        }
    }
    (entries, bytes)
}

#[cfg(target_os = "macos")]
fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return 0;
    }
    unsafe { usage.assume_init().ru_maxrss.max(0) as u64 }
}

#[cfg(all(unix, not(target_os = "macos")))]
fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return 0;
    }
    unsafe { (usage.assume_init().ru_maxrss.max(0) as u64).saturating_mul(1024) }
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> u64 {
    0
}
