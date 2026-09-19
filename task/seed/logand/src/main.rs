use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use logan_chat::engine::{
    self, CompletionUpdate, EngineCommand, EngineEvent, GenerationSettings, StopReason, TurnMetrics,
};
use logan_chat::openai::{
    ApiMessage, ChatCompletionRequest, ResponsesRequest, normalize_chat_messages,
    normalize_responses_input, settings_from_chat, settings_from_responses,
};
use logan_chat::runtime::{self, ModelFamily};
use logan_qwen4::plan::{PrefixCacheStore, RuntimeFeatures, RuntimeStats};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_stream::wrappers::ReceiverStream;

const DASHBOARD: &str = include_str!("dashboard.html");
const DEFAULT_PORT: u16 = 11435;
const DEFAULT_SYSTEM_PROMPT: &str = "You are an effective, careful assistant. Infer intent from the conversation, complete authorized work, and communicate directly. Ask only when missing information materially changes the result.";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelSettings {
    system_prompt: String,
    max_new: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repeat_penalty: f32,
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            system_prompt: DEFAULT_SYSTEM_PROMPT.into(),
            max_new: 256,
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            repeat_penalty: 1.0,
        }
    }
}

impl ModelSettings {
    fn generation(&self) -> GenerationSettings {
        GenerationSettings {
            max_new: self.max_new,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            repeat_penalty: self.repeat_penalty,
        }
    }
}

struct SettingsStore {
    path: PathBuf,
    models: HashMap<String, ModelSettings>,
}

impl SettingsStore {
    fn load(path: PathBuf) -> Self {
        let models = fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        Self { path, models }
    }

    fn get(&self, path: &Path) -> ModelSettings {
        self.models
            .get(&path.display().to_string())
            .cloned()
            .unwrap_or_default()
    }

    fn set(&mut self, path: &Path, settings: ModelSettings) -> Result<(), String> {
        self.models.insert(path.display().to_string(), settings);
        let parent = self
            .path
            .parent()
            .ok_or_else(|| "settings path has no parent".to_string())?;
        fs::create_dir_all(parent).map_err(|e| format!("create settings directory: {e}"))?;
        let temporary = self.path.with_extension("json.tmp");
        let body = serde_json::to_vec_pretty(&self.models)
            .map_err(|e| format!("encode model settings: {e}"))?;
        fs::write(&temporary, body).map_err(|e| format!("write model settings: {e}"))?;
        fs::rename(&temporary, &self.path).map_err(|e| format!("publish model settings: {e}"))
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelJob {
    id: String,
    kind: String,
    status: String,
    phase: String,
    repo_id: Option<String>,
    source: Option<String>,
    output: Option<String>,
    message: Option<String>,
    error: Option<String>,
    started_at_ms: u128,
    updated_at_ms: u128,
}

#[derive(Clone)]
struct AppState {
    snapshot: Arc<Mutex<DaemonSnapshot>>,
    control: mpsc::Sender<Control>,
    model_root: PathBuf,
    responses: Arc<Mutex<HashMap<String, Vec<ApiMessage>>>>,
    logs: Arc<Mutex<VecDeque<LogEntry>>>,
    settings: Arc<Mutex<SettingsStore>>,
    jobs: Arc<Mutex<Vec<ModelJob>>>,
}

#[derive(Clone, Debug)]
enum Control {
    Load {
        package: PathBuf,
        settings: ModelSettings,
    },
    Unload,
    Generate {
        text: String,
        settings: GenerationSettings,
    },
    Cancel,
    Reset {
        system_prompt: Option<String>,
    },
    ClearHot,
    ClearCold,
    Complete {
        prompt: String,
        settings: GenerationSettings,
        updates: mpsc::Sender<CompletionUpdate>,
    },
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelView {
    name: String,
    path: String,
    context_limit: usize,
    settings: ModelSettings,
}
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct PerformanceView {
    prompt_tokens_per_second: f64,
    generation_tokens_per_second: f64,
    ttft_ms: f64,
    prompt_ms: f64,
    generation_ms: f64,
    total_ms: f64,
    generated_tokens: usize,
    forwarded_prompt_tokens: usize,
    context_tokens: usize,
    context_limit: usize,
    last_token_id: Option<u32>,
    stop_reason: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct CacheView {
    hot_entries: usize,
    hot_tokens: usize,
    hot_bytes: u64,
    active_state_bytes: u64,
    live_reuse_tokens: usize,
    last_hot_reuse_tokens: usize,
    hot_restore_ms: f64,
    hot_write_ms: f64,
    cold_entries: usize,
    cold_bytes: u64,
    last_cold_hit_tokens: usize,
    restore_ms: f64,
    write_ms: f64,
    directory: String,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct FeaturesView {
    metal_direct: bool,
    metal_overlap: bool,
    bnns_bf16: bool,
    gdn_metal: bool,
    gdn_ane: bool,
    gdn_ane_fused: bool,
    gdn_single_copy: bool,
    attn_metal: bool,
    qsa_index_metal: bool,
    shared_io_overlap: bool,
    prefix_cache: bool,
    prefix_cache_write: bool,
}

impl From<&RuntimeFeatures> for FeaturesView {
    fn from(v: &RuntimeFeatures) -> Self {
        Self {
            metal_direct: v.metal_direct,
            metal_overlap: v.metal_overlap,
            bnns_bf16: v.bnns_bf16,
            gdn_metal: v.gdn_metal,
            gdn_ane: v.gdn_ane,
            gdn_ane_fused: v.gdn_ane_fused,
            gdn_single_copy: v.gdn_single_copy,
            attn_metal: v.attn_metal,
            qsa_index_metal: v.qsa_index_metal,
            shared_io_overlap: v.shared_io_overlap,
            prefix_cache: v.prefix_cache,
            prefix_cache_write: v.prefix_cache_write,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeView {
    route_ms: f64,
    io_ms: f64,
    shared_ms: f64,
    gpu_ms: f64,
    fill_ms: f64,
    gdn_ms: f64,
    attn_ms: f64,
    hc_ms: f64,
    head_ms: f64,
    expert_hits: u64,
    expert_misses: u64,
    expert_evictions: u64,
    expert_resident: usize,
    expert_capacity: usize,
    expert_hit_rate: f64,
    metal_encode_ms: f64,
    metal_submit_ms: f64,
    metal_wait_ms: f64,
    metal_kernel_ms: f64,
    fused_calls: u64,
    fused_experts: u64,
    metal_io_loads: u64,
    metal_io_bytes: u64,
    metal_io_waits: u64,
    metal_io_failures: u64,
    metal_io_outstanding: u64,
    metal_io_peak_outstanding: u64,
    metal_io_avg_latency_ms: f64,
    prefetch_loads: u64,
    prefetch_used: u64,
    prefetch_wasted: u64,
    features: FeaturesView,
}

impl From<&RuntimeStats> for RuntimeView {
    fn from(v: &RuntimeStats) -> Self {
        Self {
            route_ms: v.route_ms,
            io_ms: v.io_ms,
            shared_ms: v.shared_ms,
            gpu_ms: v.gpu_ms,
            fill_ms: v.fill_ms,
            gdn_ms: v.gdn_ms,
            attn_ms: v.attn_ms,
            hc_ms: v.hc_ms,
            head_ms: v.head_ms,
            expert_hits: v.expert_hits,
            expert_misses: v.expert_misses,
            expert_evictions: v.expert_evictions,
            expert_resident: v.expert_resident,
            expert_capacity: v.expert_capacity,
            expert_hit_rate: v.expert_hit_rate(),
            metal_encode_ms: v.metal_encode_ns as f64 / 1e6,
            metal_submit_ms: v.metal_submit_ns as f64 / 1e6,
            metal_wait_ms: v.metal_wait_ns as f64 / 1e6,
            metal_kernel_ms: v.metal_kernel_ns as f64 / 1e6,
            fused_calls: v.fused_calls,
            fused_experts: v.fused_experts,
            metal_io_loads: v.mio_loads,
            metal_io_bytes: v.mio_bytes,
            metal_io_waits: v.mio_waits,
            metal_io_failures: v.mio_fails,
            metal_io_outstanding: v.mio_outstanding,
            metal_io_peak_outstanding: v.mio_peak_outstanding,
            metal_io_avg_latency_ms: v.mio_avg_latency_ms(),
            prefetch_loads: v.mio_prefetch_loads,
            prefetch_used: v.mio_prefetch_used,
            prefetch_wasted: v.mio_prefetch_wasted,
            features: FeaturesView::from(&v.features),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DaemonSnapshot {
    status: String,
    phase: String,
    model: Option<ModelView>,
    performance: PerformanceView,
    cache: CacheView,
    runtime: RuntimeView,
    assistant_text: String,
    warning: Option<String>,
    error: Option<String>,
    peak_rss_bytes: u64,
    uptime_seconds: u64,
    updated_at_ms: u128,
}

impl DaemonSnapshot {
    fn idle(cache_dir: PathBuf) -> Self {
        let (cold_entries, cold_bytes) = scan_cold_cache(&cache_dir);
        Self {
            status: "idle".into(),
            phase: "No model loaded".into(),
            model: None,
            performance: PerformanceView::default(),
            cache: CacheView {
                cold_entries,
                cold_bytes,
                directory: cache_dir.display().to_string(),
                ..Default::default()
            },
            runtime: RuntimeView::default(),
            assistant_text: String::new(),
            warning: None,
            error: None,
            peak_rss_bytes: peak_rss_bytes(),
            uptime_seconds: 0,
            updated_at_ms: now_ms(),
        }
    }

    fn touch(&mut self, started: Instant) {
        self.updated_at_ms = now_ms();
        self.uptime_seconds = started.elapsed().as_secs();
        self.peak_rss_bytes = peak_rss_bytes();
    }
}

#[derive(Serialize)]
struct ApiReply {
    ok: bool,
    message: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LogEntry {
    ts_ms: u128,
    level: String,
    source: String,
    message: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelCandidate {
    name: String,
    path: String,
    size_bytes: u64,
    settings: ModelSettings,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LoadRequest {
    path: String,
    system_prompt: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateRequest {
    text: String,
    max_new: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repeat_penalty: Option<f32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ResetRequest {
    system_prompt: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveSettingsRequest {
    path: String,
    system_prompt: Option<String>,
    max_new: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repeat_penalty: Option<f32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DownloadRequest {
    repo_id: String,
    destination: Option<String>,
    revision: Option<String>,
    include: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct QuantizeRequest {
    source: String,
    output: String,
    max_context: Option<usize>,
    target: Option<String>,
    quant: Option<String>,
    quant_floor: Option<String>,
    codec: Option<String>,
    optimization: Option<String>,
    verify: Option<bool>,
    force: Option<bool>,
}

#[tokio::main]
async fn main() {
    let (host, port, model_root) = match parse_args() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("logand: {e}");
            print_usage();
            std::process::exit(2);
        }
    };

    let cache_dir = PrefixCacheStore::from_env()
        .map(|s| s.root().to_path_buf())
        .unwrap_or_default();
    let snapshot = Arc::new(Mutex::new(DaemonSnapshot::idle(cache_dir)));
    let logs = Arc::new(Mutex::new(VecDeque::new()));
    let settings = Arc::new(Mutex::new(SettingsStore::load(settings_file_path())));
    let jobs = Arc::new(Mutex::new(Vec::new()));
    push_log(&logs, "info", "daemon", "logand starting");
    let (control_tx, control_rx) = mpsc::channel();
    spawn_supervisor(Arc::clone(&snapshot), Arc::clone(&logs), control_rx);

    let state = AppState {
        snapshot,
        control: control_tx,
        model_root,
        responses: Arc::new(Mutex::new(HashMap::new())),
        logs,
        settings,
        jobs,
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/v1/models", get(openai_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses_create))
        .route("/api/state", get(api_state))
        .route("/api/logs", get(api_logs))
        .route("/api/models", get(api_models))
        .route("/api/jobs", get(api_jobs))
        .route("/api/model/load", post(api_load_model))
        .route("/api/model/unload", post(api_unload_model))
        .route("/api/model/settings", post(api_save_settings))
        .route("/api/models/download", post(api_download_model))
        .route("/api/models/quantize", post(api_quantize_model))
        .route("/api/generate", post(api_generate))
        .route("/api/cancel", post(api_cancel))
        .route("/api/reset", post(api_reset))
        .route("/api/cache/hot/clear", post(api_clear_hot))
        .route("/api/cache/cold/clear", post(api_clear_cold))
        .with_state(state);

    let addr = SocketAddr::new(host, port);
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(v) => v,
        Err(e) => {
            eprintln!("logand: bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    println!("Logan dashboard: http://{addr}");
    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("logand: server error: {e}");
    }
}

async fn index() -> Html<&'static str> {
    Html(DASHBOARD)
}

async fn health() -> &'static str {
    "ok"
}

#[derive(Clone)]
struct CompletionOutcome {
    text: String,
    metrics: TurnMetrics,
    stats: RuntimeStats,
}

async fn openai_models(State(state): State<AppState>) -> Response {
    let loaded = state
        .snapshot
        .lock()
        .unwrap()
        .model
        .as_ref()
        .map(|model| model.name.clone());
    let data: Vec<Value> = discover_models(&state.model_root)
        .into_iter()
        .map(|model| {
            json!({
                "id": model.name,
                "object": "model",
                "created": 0,
                "owned_by": "logan",
                "logan_path": model.path,
                "logan_size_bytes": model.size_bytes,
                "logan_loaded": loaded.as_deref() == Some(model.name.as_str())
            })
        })
        .collect();
    Json(json!({"object":"list","data":data})).into_response()
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    push_log(
        &state.logs,
        "info",
        "api",
        format!(
            "POST /v1/chat/completions model={} stream={} messages={}",
            request.model,
            request.stream,
            request.messages.len()
        ),
    );
    let model = match require_loaded_model(&state, Some(&request.model)) {
        Ok(model) => model,
        Err(response) => return response,
    };
    let messages = match normalize_chat_messages(&request.messages) {
        Ok(messages) => messages,
        Err(error) => return openai_error(StatusCode::BAD_REQUEST, error, "invalid_request_error"),
    };
    let prompt = match render_loaded_prompt(&state, &messages, None) {
        Ok(prompt) => prompt,
        Err(error) => return openai_error(StatusCode::BAD_REQUEST, error, "invalid_request_error"),
    };
    let settings = match chat_generation_settings(&state, &request) {
        Ok(settings) => settings,
        Err(error) => return openai_error(StatusCode::BAD_REQUEST, error, "invalid_request_error"),
    };
    let updates = match queue_completion(&state, prompt, settings.clone()) {
        Ok(updates) => updates,
        Err(response) => return response,
    };
    let id = new_id("chatcmpl");
    let created = unix_seconds();
    if request.stream {
        return chat_completion_stream(
            updates,
            id,
            created,
            model,
            request
                .stream_options
                .is_some_and(|options| options.include_usage),
        );
    }

    match wait_completion(updates).await {
        Ok(outcome) => {
            let finish = finish_reason(outcome.metrics.stop_reason.as_ref());
            Json(json!({
                "id": id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": outcome.text,
                        "refusal": null
                    },
                    "logprobs": null,
                    "finish_reason": finish
                }],
                "usage": chat_usage(&outcome.metrics),
                "x_logan": logan_stats(&outcome.metrics, &outcome.stats)
            }))
            .into_response()
        }
        Err(error) => openai_error(StatusCode::INTERNAL_SERVER_ERROR, error, "server_error"),
    }
}

async fn responses_create(
    State(state): State<AppState>,
    Json(request): Json<ResponsesRequest>,
) -> Response {
    push_log(
        &state.logs,
        "info",
        "api",
        format!(
            "POST /v1/responses model={} stream={} previous={}",
            request.model.as_deref().unwrap_or("loaded"),
            request.stream,
            request.previous_response_id.as_deref().unwrap_or("none")
        ),
    );
    let model = match require_loaded_model(&state, request.model.as_deref()) {
        Ok(model) => model,
        Err(response) => return response,
    };
    let settings = match responses_generation_settings(&state, &request) {
        Ok(settings) => settings,
        Err(error) => return openai_error(StatusCode::BAD_REQUEST, error, "invalid_request_error"),
    };

    let mut history = if let Some(previous) = request.previous_response_id.as_deref() {
        match state.responses.lock().unwrap().get(previous).cloned() {
            Some(history) => history,
            None => {
                return openai_error(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "previous_response_id {previous:?} is not known to this logand process"
                    ),
                    "invalid_request_error",
                );
            }
        }
    } else {
        Vec::new()
    };
    let new_input = match normalize_responses_input(request.input.as_ref()) {
        Ok(messages) => messages,
        Err(error) => return openai_error(StatusCode::BAD_REQUEST, error, "invalid_request_error"),
    };
    if new_input.is_empty() && history.is_empty() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "Responses request has no input",
            "invalid_request_error",
        );
    }
    history.extend(new_input);
    let prompt = match render_loaded_prompt(&state, &history, request.instructions.as_deref()) {
        Ok(prompt) => prompt,
        Err(error) => return openai_error(StatusCode::BAD_REQUEST, error, "invalid_request_error"),
    };
    let updates = match queue_completion(&state, prompt, settings.clone()) {
        Ok(updates) => updates,
        Err(response) => return response,
    };

    let response_id = new_id("resp");
    let message_id = new_id("msg");
    let created = unix_seconds();
    if request.stream {
        return responses_stream(
            updates,
            response_id,
            message_id,
            created,
            model,
            request.instructions,
            request.previous_response_id,
            request.metadata.unwrap_or_else(|| json!({})),
            settings,
            Arc::clone(&state.responses),
            history,
        );
    }

    match wait_completion(updates).await {
        Ok(outcome) => {
            let mut stored = history;
            stored.push(ApiMessage {
                role: "assistant".into(),
                text: outcome.text.clone(),
            });
            store_response_history(&state.responses, response_id.clone(), stored);
            Json(response_object(
                &response_id,
                &message_id,
                created,
                &model,
                request.instructions.as_deref(),
                request.previous_response_id.as_deref(),
                request.metadata.unwrap_or_else(|| json!({})),
                &settings,
                &outcome,
            ))
            .into_response()
        }
        Err(error) => openai_error(StatusCode::INTERNAL_SERVER_ERROR, error, "server_error"),
    }
}

fn queue_completion(
    state: &AppState,
    prompt: String,
    settings: GenerationSettings,
) -> Result<mpsc::Receiver<CompletionUpdate>, Response> {
    let (updates_tx, updates_rx) = mpsc::channel();
    state
        .control
        .send(Control::Complete {
            prompt,
            settings,
            updates: updates_tx,
        })
        .map_err(|_| {
            openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "daemon supervisor exited",
                "server_error",
            )
        })?;
    Ok(updates_rx)
}

fn loaded_model_settings(state: &AppState) -> ModelSettings {
    state
        .snapshot
        .lock()
        .unwrap()
        .model
        .as_ref()
        .map(|model| model.settings.clone())
        .unwrap_or_default()
}

fn chat_generation_settings(
    state: &AppState,
    request: &ChatCompletionRequest,
) -> Result<GenerationSettings, String> {
    let parsed = settings_from_chat(request)?;
    let defaults = loaded_model_settings(state);
    let mut settings = defaults.generation();
    if request.max_completion_tokens.is_some() || request.max_tokens.is_some() {
        settings.max_new = parsed.max_new;
    }
    if request.temperature.is_some() {
        settings.temperature = parsed.temperature;
    }
    if request.top_p.is_some() {
        settings.top_p = parsed.top_p;
    }
    if request.top_k.is_some() {
        settings.top_k = parsed.top_k;
    }
    Ok(settings)
}

fn responses_generation_settings(
    state: &AppState,
    request: &ResponsesRequest,
) -> Result<GenerationSettings, String> {
    let parsed = settings_from_responses(request)?;
    let defaults = loaded_model_settings(state);
    let mut settings = defaults.generation();
    if request.max_output_tokens.is_some() {
        settings.max_new = parsed.max_new;
    }
    if request.temperature.is_some() {
        settings.temperature = parsed.temperature;
    }
    if request.top_p.is_some() {
        settings.top_p = parsed.top_p;
    }
    if request.top_k.is_some() {
        settings.top_k = parsed.top_k;
    }
    Ok(settings)
}

fn render_loaded_prompt(
    state: &AppState,
    messages: &[ApiMessage],
    instructions: Option<&str>,
) -> Result<String, String> {
    let (family, default_system) = {
        let snapshot = state.snapshot.lock().unwrap();
        let Some(model) = snapshot.model.as_ref() else {
            return Err("no Logan model is loaded".into());
        };
        let config = Path::new(&model.path).join("config.json");
        let family = if logan_llama::load_config(config).is_ok() {
            ModelFamily::MiniCpm5
        } else {
            ModelFamily::Qwen4
        };
        (family, model.settings.system_prompt.clone())
    };
    let instructions = instructions
        .or_else(|| (family == ModelFamily::MiniCpm5).then_some(default_system.as_str()));
    runtime::render_prompt(
        &runtime::PromptAdapter::for_family(family),
        messages,
        instructions,
    )
}

async fn wait_completion(
    updates: mpsc::Receiver<CompletionUpdate>,
) -> Result<CompletionOutcome, String> {
    tokio::task::spawn_blocking(move || {
        loop {
            match updates.recv() {
                Ok(CompletionUpdate::Done {
                    text,
                    metrics,
                    stats,
                }) => {
                    return Ok(CompletionOutcome {
                        text,
                        metrics,
                        stats,
                    });
                }
                Ok(CompletionUpdate::Error(error)) => return Err(error),
                Ok(_) => {}
                Err(_) => return Err("inference update channel closed".into()),
            }
        }
    })
    .await
    .map_err(|error| format!("completion task failed: {error}"))?
}

fn chat_completion_stream(
    updates: mpsc::Receiver<CompletionUpdate>,
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
    tokio::task::spawn_blocking(move || {
        let initial = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]
        });
        if tx
            .blocking_send(Ok(Event::default().data(initial.to_string())))
            .is_err()
        {
            return;
        }
        while let Ok(update) = updates.recv() {
            match update {
                CompletionUpdate::Token { chunk, .. } if !chunk.is_empty() => {
                    let event = json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index":0,"delta":{"content":chunk},"finish_reason":null}]
                    });
                    if tx
                        .blocking_send(Ok(Event::default().data(event.to_string())))
                        .is_err()
                    {
                        return;
                    }
                }
                CompletionUpdate::Done { metrics, stats, .. } => {
                    let finish = finish_reason(metrics.stop_reason.as_ref());
                    let final_chunk = json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index":0,"delta":{},"finish_reason":finish}],
                        "x_logan": logan_stats(&metrics, &stats)
                    });
                    let _ = tx.blocking_send(Ok(Event::default().data(final_chunk.to_string())));
                    if include_usage {
                        let usage = json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": chat_usage(&metrics)
                        });
                        let _ = tx.blocking_send(Ok(Event::default().data(usage.to_string())));
                    }
                    let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                    return;
                }
                CompletionUpdate::Error(error) => {
                    let payload = json!({"error":{"message":error,"type":"server_error","param":null,"code":null}});
                    let _ = tx.blocking_send(Ok(Event::default().data(payload.to_string())));
                    let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                    return;
                }
                _ => {}
            }
        }
    });
    Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[allow(clippy::too_many_arguments)]
fn responses_stream(
    updates: mpsc::Receiver<CompletionUpdate>,
    response_id: String,
    message_id: String,
    created: u64,
    model: String,
    instructions: Option<String>,
    previous_response_id: Option<String>,
    metadata: Value,
    settings: GenerationSettings,
    responses: Arc<Mutex<HashMap<String, Vec<ApiMessage>>>>,
    history: Vec<ApiMessage>,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
    tokio::task::spawn_blocking(move || {
        let partial = response_shell(
            &response_id,
            created,
            &model,
            "in_progress",
            instructions.as_deref(),
            previous_response_id.as_deref(),
            metadata.clone(),
            &settings,
        );
        for (event_name, payload) in [
            (
                "response.created",
                json!({"type":"response.created","response":partial.clone()}),
            ),
            (
                "response.in_progress",
                json!({"type":"response.in_progress","response":partial}),
            ),
            (
                "response.output_item.added",
                json!({
                    "type":"response.output_item.added","output_index":0,
                    "item":{"id":message_id,"type":"message","status":"in_progress","role":"assistant","content":[]}
                }),
            ),
            (
                "response.content_part.added",
                json!({
                    "type":"response.content_part.added","item_id":message_id,"output_index":0,"content_index":0,
                    "part":{"type":"output_text","text":"","annotations":[]}
                }),
            ),
        ] {
            if tx
                .blocking_send(Ok(Event::default()
                    .event(event_name)
                    .data(payload.to_string())))
                .is_err()
            {
                return;
            }
        }

        while let Ok(update) = updates.recv() {
            match update {
                CompletionUpdate::Token { chunk, .. } if !chunk.is_empty() => {
                    let payload = json!({
                        "type":"response.output_text.delta",
                        "item_id":message_id,"output_index":0,"content_index":0,"delta":chunk
                    });
                    if tx
                        .blocking_send(Ok(Event::default()
                            .event("response.output_text.delta")
                            .data(payload.to_string())))
                        .is_err()
                    {
                        return;
                    }
                }
                CompletionUpdate::Done {
                    text,
                    metrics,
                    stats,
                } => {
                    let outcome = CompletionOutcome {
                        text: text.clone(),
                        metrics,
                        stats,
                    };
                    let text_part = json!({"type":"output_text","text":text,"annotations":[]});
                    let done_text = json!({
                        "type":"response.output_text.done","item_id":message_id,
                        "output_index":0,"content_index":0,"text":outcome.text
                    });
                    let done_part = json!({
                        "type":"response.content_part.done","item_id":message_id,"output_index":0,"content_index":0,
                        "part":text_part
                    });
                    let done_item = json!({
                        "type":"response.output_item.done","output_index":0,
                        "item":{"id":message_id,"type":"message","status":"completed","role":"assistant","content":[text_part]}
                    });
                    let final_response = response_object(
                        &response_id,
                        &message_id,
                        created,
                        &model,
                        instructions.as_deref(),
                        previous_response_id.as_deref(),
                        metadata.clone(),
                        &settings,
                        &outcome,
                    );
                    for (event_name, payload) in [
                        ("response.output_text.done", done_text),
                        ("response.content_part.done", done_part),
                        ("response.output_item.done", done_item),
                        (
                            "response.completed",
                            json!({"type":"response.completed","response":final_response}),
                        ),
                    ] {
                        let _ = tx.blocking_send(Ok(Event::default()
                            .event(event_name)
                            .data(payload.to_string())));
                    }
                    let mut stored = history;
                    stored.push(ApiMessage {
                        role: "assistant".into(),
                        text: outcome.text,
                    });
                    store_response_history(&responses, response_id.clone(), stored);
                    return;
                }
                CompletionUpdate::Error(error) => {
                    let payload = json!({
                        "type":"response.failed",
                        "response":{
                            "id":response_id,"object":"response","created_at":created,"status":"failed",
                            "error":{"code":"server_error","message":error}
                        }
                    });
                    let _ = tx.blocking_send(Ok(Event::default()
                        .event("response.failed")
                        .data(payload.to_string())));
                    return;
                }
                _ => {}
            }
        }
    });
    Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn response_shell(
    id: &str,
    created: u64,
    model: &str,
    status: &str,
    instructions: Option<&str>,
    previous_response_id: Option<&str>,
    metadata: Value,
    settings: &GenerationSettings,
) -> Value {
    json!({
        "id":id,"object":"response","created_at":created,"status":status,
        "error":null,"incomplete_details":null,"instructions":instructions,
        "max_output_tokens":settings.max_new,"model":model,"output":[],
        "parallel_tool_calls":true,"previous_response_id":previous_response_id,
        "reasoning":{"effort":null,"summary":null},"store":false,
        "temperature":settings.temperature,"text":{"format":{"type":"text"}},
        "tool_choice":"none","tools":[],"top_p":settings.top_p,"truncation":"disabled",
        "metadata":metadata
    })
}

#[allow(clippy::too_many_arguments)]
fn response_object(
    id: &str,
    message_id: &str,
    created: u64,
    model: &str,
    instructions: Option<&str>,
    previous_response_id: Option<&str>,
    metadata: Value,
    settings: &GenerationSettings,
    outcome: &CompletionOutcome,
) -> Value {
    let incomplete = matches!(
        outcome.metrics.stop_reason,
        Some(StopReason::MaxTokens | StopReason::ContextFull)
    );
    json!({
        "id":id,"object":"response","created_at":created,"completed_at":unix_seconds(),
        "status":if incomplete {"incomplete"} else {"completed"},"error":null,
        "incomplete_details":if incomplete {json!({"reason":"max_output_tokens"})} else {Value::Null},
        "instructions":instructions,"max_output_tokens":settings.max_new,"model":model,
        "output":[{"id":message_id,"type":"message","status":"completed","role":"assistant",
            "content":[{"type":"output_text","text":outcome.text,"annotations":[]}]}],
        "output_text":outcome.text,"parallel_tool_calls":true,
        "previous_response_id":previous_response_id,"reasoning":{"effort":null,"summary":null},
        "store":false,"temperature":settings.temperature,"text":{"format":{"type":"text"}},
        "tool_choice":"none","tools":[],"top_p":settings.top_p,"truncation":"disabled",
        "usage":responses_usage(&outcome.metrics),"metadata":metadata,
        "x_logan":logan_stats(&outcome.metrics,&outcome.stats)
    })
}

fn require_loaded_model(state: &AppState, requested: Option<&str>) -> Result<String, Response> {
    let snapshot = state.snapshot.lock().unwrap();
    let Some(model) = snapshot.model.as_ref() else {
        return Err(openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "no Logan model is loaded; load one from the dashboard first",
            "server_error",
        ));
    };
    if snapshot.status == "loading" {
        return Err(openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "the Logan model is still loading",
            "server_error",
        ));
    }
    if snapshot.status == "error" {
        return Err(openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            snapshot
                .error
                .clone()
                .unwrap_or_else(|| "Logan runtime error".into()),
            "server_error",
        ));
    }
    if let Some(requested) = requested.filter(|value| !value.is_empty()) {
        let basename = Path::new(&model.path)
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or(&model.name);
        if requested != model.name && requested != basename && requested != "logan" {
            return Err(openai_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "model {requested:?} is not loaded; current Logan model is {:?}",
                    model.name
                ),
                "invalid_request_error",
            ));
        }
    }
    Ok(model.name.clone())
}

fn openai_error(status: StatusCode, message: impl Into<String>, kind: &str) -> Response {
    (
        status,
        Json(json!({
            "error": {"message":message.into(),"type":kind,"param":null,"code":null}
        })),
    )
        .into_response()
}

fn finish_reason(reason: Option<&StopReason>) -> &'static str {
    match reason {
        Some(StopReason::MaxTokens | StopReason::ContextFull) => "length",
        _ => "stop",
    }
}

fn cached_tokens(metrics: &TurnMetrics) -> usize {
    metrics.ram_cached_tokens.max(metrics.ssd_cached_tokens)
}

fn chat_usage(metrics: &TurnMetrics) -> Value {
    json!({
        "prompt_tokens":metrics.input_tokens,
        "completion_tokens":metrics.generated_tokens,
        "total_tokens":metrics.input_tokens.saturating_add(metrics.generated_tokens),
        "prompt_tokens_details":{"cached_tokens":cached_tokens(metrics)},
        "completion_tokens_details":{"reasoning_tokens":0}
    })
}

fn responses_usage(metrics: &TurnMetrics) -> Value {
    json!({
        "input_tokens":metrics.input_tokens,
        "input_tokens_details":{"cached_tokens":cached_tokens(metrics),"cache_write_tokens":0},
        "output_tokens":metrics.generated_tokens,
        "output_tokens_details":{"reasoning_tokens":0},
        "total_tokens":metrics.input_tokens.saturating_add(metrics.generated_tokens)
    })
}

fn logan_stats(metrics: &TurnMetrics, stats: &RuntimeStats) -> Value {
    let generation_tps = if metrics.generation_ms > 0.0 {
        metrics.generated_tokens as f64 * 1000.0 / metrics.generation_ms
    } else {
        0.0
    };
    let prompt_tps = if metrics.prompt_ms > 0.0 {
        metrics.forwarded_prompt_tokens as f64 * 1000.0 / metrics.prompt_ms
    } else {
        0.0
    };
    json!({
        "generation_tokens_per_second":generation_tps,
        "prompt_tokens_per_second":prompt_tps,
        "ttft_ms":metrics.first_token_ms,
        "prompt_ms":metrics.prompt_ms,
        "generation_ms":metrics.generation_ms,
        "total_ms":metrics.total_ms,
        "forwarded_prompt_tokens":metrics.forwarded_prompt_tokens,
        "ram_cached_tokens":metrics.ram_cached_tokens,
        "ssd_cached_tokens":metrics.ssd_cached_tokens,
        "ram_cache_restore_ms":metrics.ram_cache_restore_ms,
        "ssd_cache_restore_ms":metrics.cache_restore_ms,
        "ram_cache_bytes":metrics.hot_cache_bytes,
        "ram_cache_entries":metrics.hot_cache_entries,
        "active_state_bytes":metrics.active_state_bytes,
        "expert_hit_rate":stats.expert_hit_rate(),
        "expert_resident":stats.expert_resident,
        "expert_capacity":stats.expert_capacity,
        "expert_evictions":stats.expert_evictions,
        "metal_kernel_ms":stats.metal_kernel_ns as f64 / 1e6,
        "metal_wait_ms":stats.metal_wait_ns as f64 / 1e6,
        "metal_io_bytes":stats.mio_bytes,
        "metal_io_loads":stats.mio_loads,
        "metal_io_avg_latency_ms":stats.mio_avg_latency_ms(),
        "gdn_ms":stats.gdn_ms,"attention_ms":stats.attn_ms,"routed_io_ms":stats.io_ms,
        "shared_ms":stats.shared_ms,"gpu_moe_ms":stats.gpu_ms,"route_ms":stats.route_ms
    })
}

fn store_response_history(
    responses: &Arc<Mutex<HashMap<String, Vec<ApiMessage>>>>,
    id: String,
    history: Vec<ApiMessage>,
) {
    let mut responses = responses.lock().unwrap();
    if responses.len() >= 256 {
        if let Some(key) = responses.keys().next().cloned() {
            responses.remove(&key);
        }
    }
    responses.insert(id, history);
}

fn new_id(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{prefix}_{nanos:032x}{:08x}", std::process::id())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn api_state(State(state): State<AppState>) -> Json<DaemonSnapshot> {
    Json(state.snapshot.lock().unwrap().clone())
}

async fn api_logs(State(state): State<AppState>) -> Json<Vec<LogEntry>> {
    Json(state.logs.lock().unwrap().iter().cloned().collect())
}

async fn api_models(State(state): State<AppState>) -> Json<Vec<ModelCandidate>> {
    let settings = state.settings.lock().unwrap();
    let models = discover_models(&state.model_root)
        .into_iter()
        .map(|mut model| {
            model.settings = settings.get(Path::new(&model.path));
            model
        })
        .collect();
    Json(models)
}

async fn api_jobs(State(state): State<AppState>) -> Json<Vec<ModelJob>> {
    Json(state.jobs.lock().unwrap().clone())
}

async fn api_load_model(
    State(state): State<AppState>,
    Json(req): Json<LoadRequest>,
) -> impl IntoResponse {
    let package = expand_home(&req.path);
    if !is_loadable_model(&package) {
        return api_error(
            StatusCode::BAD_REQUEST,
            format!(
                "not a loadable model directory (expected a Logan .coli package, or a \
                 safetensors checkpoint with config.json + model.safetensors[.index.json]): {}",
                package.display()
            ),
        );
    }
    let mut settings = state.settings.lock().unwrap().get(&package);
    if let Some(system_prompt) = req.system_prompt.filter(|v| !v.trim().is_empty()) {
        settings.system_prompt = system_prompt;
    }
    send_control(&state, Control::Load { package, settings }, "loading model")
}
async fn api_save_settings(
    State(state): State<AppState>,
    Json(req): Json<SaveSettingsRequest>,
) -> impl IntoResponse {
    let package = expand_home(&req.path);
    if !is_loadable_model(&package) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "settings target is not a loadable model",
        );
    }
    let mut settings = state.settings.lock().unwrap().get(&package);
    if let Some(value) = req.system_prompt.filter(|v| !v.trim().is_empty()) {
        settings.system_prompt = value;
    }
    if let Some(value) = req.max_new {
        settings.max_new = value.clamp(1, 65_536);
    }
    if let Some(value) = req.temperature {
        if !(0.0..=2.0).contains(&value) {
            return api_error(StatusCode::BAD_REQUEST, "temperature must be in 0..=2");
        }
        settings.temperature = value;
    }
    if let Some(value) = req.top_p {
        if !(0.01..=1.0).contains(&value) {
            return api_error(StatusCode::BAD_REQUEST, "top-p must be in 0.01..=1");
        }
        settings.top_p = value;
    }
    if let Some(value) = req.top_k {
        settings.top_k = value;
    }
    if let Some(value) = req.repeat_penalty {
        if !(1.0..=2.0).contains(&value) {
            return api_error(StatusCode::BAD_REQUEST, "repeat penalty must be in 1..=2");
        }
        settings.repeat_penalty = value;
    }
    if let Err(error) = state
        .settings
        .lock()
        .unwrap()
        .set(&package, settings.clone())
    {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, error);
    }

    let mut reset = false;
    {
        let mut snapshot = state.snapshot.lock().unwrap();
        if let Some(model) = snapshot.model.as_mut() {
            if model.path == package.display().to_string() {
                reset = model.settings.system_prompt != settings.system_prompt;
                model.settings = settings.clone();
            }
        }
    }
    if reset {
        let _ = state.control.send(Control::Reset {
            system_prompt: Some(settings.system_prompt.clone()),
        });
    }
    (
        StatusCode::OK,
        Json(ApiReply {
            ok: true,
            message: "model settings saved".into(),
        }),
    )
        .into_response()
}

async fn api_unload_model(State(state): State<AppState>) -> impl IntoResponse {
    send_control(&state, Control::Unload, "unloading model")
}

async fn api_generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> impl IntoResponse {
    if req.text.trim().is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "prompt is empty");
    }
    let mut settings = loaded_model_settings(&state).generation();
    if let Some(v) = req.max_new {
        settings.max_new = v.clamp(1, 8192);
    }
    if let Some(v) = req.temperature {
        settings.temperature = v.clamp(0.0, 5.0);
    }
    if let Some(v) = req.top_p {
        settings.top_p = v.clamp(0.01, 1.0);
    }
    if let Some(v) = req.top_k {
        settings.top_k = v;
    }
    if let Some(v) = req.repeat_penalty {
        settings.repeat_penalty = v.clamp(1.0, 2.0);
    }
    send_control(
        &state,
        Control::Generate {
            text: req.text,
            settings,
        },
        "generation queued",
    )
}

async fn api_cancel(State(state): State<AppState>) -> impl IntoResponse {
    send_control(&state, Control::Cancel, "cancel requested")
}

async fn api_reset(
    State(state): State<AppState>,
    Json(req): Json<ResetRequest>,
) -> impl IntoResponse {
    send_control(
        &state,
        Control::Reset {
            system_prompt: req.system_prompt,
        },
        "reset requested",
    )
}

async fn api_clear_hot(State(state): State<AppState>) -> impl IntoResponse {
    send_control(&state, Control::ClearHot, "hot cache reset requested")
}

async fn api_clear_cold(State(state): State<AppState>) -> impl IntoResponse {
    send_control(&state, Control::ClearCold, "cold cache clear requested")
}
fn settings_file_path() -> PathBuf {
    std::env::var_os("LOGAN_CONFIG_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(|home| PathBuf::from(home).join("Library/Application Support/Logan"))
        })
        .unwrap_or_else(|| PathBuf::from(".logan"))
        .join("model-settings.json")
}

fn repo_leaf(repo_id: &str) -> Option<&str> {
    let mut parts = repo_id.split('/');
    let owner = parts.next()?;
    let name = parts.next()?;
    if parts.next().is_some()
        || owner.is_empty()
        || name.is_empty()
        || matches!(owner, "." | "..")
        || matches!(name, "." | "..")
        || !owner
            .chars()
            .chain(name.chars())
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
    {
        return None;
    }
    Some(name)
}

fn safe_model_path(root: &Path, raw: Option<&str>, default_name: &str) -> Result<PathBuf, String> {
    let root = fs::canonicalize(root).map_err(|e| format!("model root is unavailable: {e}"))?;
    if raw.is_some_and(|value| value.trim().is_empty()) {
        return Err("model path must not be empty".into());
    }
    let raw = raw.filter(|value| !value.trim().is_empty());

    let candidate = match raw {
        Some(value) => {
            let value = expand_home(value);
            if value.is_absolute() {
                value
            } else {
                root.join(value)
            }
        }
        None => root.join(default_name),
    };
    let resolved = if candidate.exists() {
        fs::canonicalize(&candidate).map_err(|e| format!("resolve model path: {e}"))?
    } else {
        let parent = candidate
            .parent()
            .ok_or_else(|| "model path has no parent".to_string())?;
        let parent = fs::canonicalize(parent)
            .map_err(|e| format!("model path parent is unavailable: {e}"))?;
        parent.join(
            candidate
                .file_name()
                .ok_or_else(|| "model path has no filename".to_string())?,
        )
    };
    if !resolved.starts_with(&root) {
        return Err(format!("model path must remain inside {}", root.display()));
    }
    Ok(resolved)
}

fn resolve_tool(name: &str) -> Result<PathBuf, String> {
    if name == "logan" {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(path) = exe.parent().map(|parent| parent.join(name)) {
                if path.is_file() {
                    return Ok(path);
                }
            }
        }
    }
    if let Some(path) = std::env::var_os("PATH")
        .into_iter()
        .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .map(|path| path.join(name))
        .find(|path| path.is_file())
    {
        return Ok(path);
    }
    if name == "hf" {
        for path in [
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/bin/hf")),
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".pyenv/shims/hf")),
            Some(PathBuf::from("/opt/homebrew/bin/hf")),
            Some(PathBuf::from("/usr/local/bin/hf")),
        ]
        .into_iter()
        .flatten()
        {
            if path.is_file() {
                return Ok(path);
            }
        }
    }
    Err(format!(
        "could not find `{name}`; install it or put it on the daemon PATH"
    ))
}

fn output_tail(output: &std::process::Output) -> String {
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    let mut lines = text.lines().rev().take(12).collect::<Vec<_>>();
    lines.reverse();
    let text = lines.join("\n");
    if text.len() > 2_000 {
        text.chars()
            .rev()
            .take(2_000)
            .collect::<String>()
            .chars()
            .rev()
            .collect()
    } else {
        text
    }
}

fn push_job(jobs: &Arc<Mutex<Vec<ModelJob>>>, job: ModelJob) {
    let mut jobs = jobs.lock().unwrap();
    jobs.push(job);
    if jobs.len() > 32 {
        let remove = jobs
            .iter()
            .position(|job| job.status == "completed" || job.status == "failed");
        if let Some(index) = remove {
            jobs.remove(index);
        }
    }
}

fn update_job(
    jobs: &Arc<Mutex<Vec<ModelJob>>>,
    id: &str,
    status: &str,
    phase: &str,
    message: Option<String>,
    error: Option<String>,
) {
    if let Some(job) = jobs.lock().unwrap().iter_mut().find(|job| job.id == id) {
        job.status = status.into();
        job.phase = phase.into();
        job.message = message;
        job.error = error;
        job.updated_at_ms = now_ms();
    }
}

async fn api_download_model(
    State(state): State<AppState>,
    Json(req): Json<DownloadRequest>,
) -> impl IntoResponse {
    let repo_id = req.repo_id.trim().to_string();
    let Some(leaf) = repo_leaf(&repo_id) else {
        return api_error(
            StatusCode::BAD_REQUEST,
            "repoId must be an owner/name Hugging Face model id",
        );
    };
    let destination = match safe_model_path(&state.model_root, req.destination.as_deref(), leaf) {
        Ok(path) => path,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, error),
    };
    let job = ModelJob {
        id: new_id("download"),
        kind: "download".into(),
        status: "queued".into(),
        phase: "Queued".into(),
        repo_id: Some(repo_id.clone()),
        source: None,
        output: Some(destination.display().to_string()),
        message: None,
        error: None,
        started_at_ms: now_ms(),
        updated_at_ms: now_ms(),
    };
    let job_id = job.id.clone();
    push_job(&state.jobs, job);
    let jobs = Arc::clone(&state.jobs);
    let revision = req.revision.filter(|value| !value.trim().is_empty());
    let include = req.include.filter(|value| !value.trim().is_empty());
    let thread_job_id = job_id.clone();
    thread::spawn(move || {
        let job_id = thread_job_id;
        update_job(
            &jobs,
            &job_id,
            "running",
            "Downloading from Hugging Face",
            None,
            None,
        );
        let result = (|| {
            let executable = resolve_tool("hf")?;
            fs::create_dir_all(&destination)
                .map_err(|e| format!("create download directory: {e}"))?;
            let mut command = Command::new(executable);
            command
                .arg("download")
                .arg(&repo_id)
                .arg("--type")
                .arg("model")
                .arg("--local-dir")
                .arg(&destination);
            if let Some(revision) = revision {
                command.arg("--revision").arg(revision);
            }
            if let Some(include) = include {
                command.arg("--include").arg(include);
            }
            let output = command
                .output()
                .map_err(|e| format!("run hf download: {e}"))?;
            if !output.status.success() {
                return Err(format!("hf download failed: {}", output_tail(&output)));
            }
            Ok(output_tail(&output))
        })();
        match result {
            Ok(message) => update_job(
                &jobs,
                &job_id,
                "completed",
                "Download complete",
                Some(if message.is_empty() {
                    "model downloaded".into()
                } else {
                    message
                }),
                None,
            ),
            Err(error) => update_job(
                &jobs,
                &job_id,
                "failed",
                "Download failed",
                None,
                Some(error),
            ),
        }
    });
    (
        StatusCode::ACCEPTED,
        Json(json!({"ok":true,"jobId":job_id,"message":"download queued"})),
    )
        .into_response()
}

async fn api_quantize_model(
    State(state): State<AppState>,
    Json(req): Json<QuantizeRequest>,
) -> impl IntoResponse {
    let source = match safe_model_path(&state.model_root, Some(&req.source), "") {
        Ok(path) => path,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, error),
    };
    if !source.is_dir() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "quantization source is not a directory",
        );
    }
    let output = match safe_model_path(&state.model_root, Some(&req.output), "") {
        Ok(path) => path,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, error),
    };
    if source == output {
        return api_error(
            StatusCode::BAD_REQUEST,
            "quantization output must differ from source",
        );
    }
    let max_context = req.max_context.unwrap_or(131_072).clamp(1, 1_000_000);
    let target = req.target.unwrap_or_else(|| "native".into());
    let quant = req.quant.unwrap_or_else(|| "exact".into());
    let quant_floor = req.quant_floor.unwrap_or_else(|| "bf16".into());
    let codec = req.codec.unwrap_or_else(|| "none".into());
    let optimization = req.optimization.unwrap_or_else(|| "default".into());
    for (label, value) in [
        ("target", &target),
        ("quant", &quant),
        ("quantFloor", &quant_floor),
        ("codec", &codec),
        ("optimization", &optimization),
    ] {
        if value.is_empty()
            || value.len() > 64
            || !value
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
        {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("{label} contains unsupported characters"),
            );
        }
    }
    let job = ModelJob {
        id: new_id("quantize"),
        kind: "quantize".into(),
        status: "queued".into(),
        phase: "Queued".into(),
        repo_id: None,
        source: Some(source.display().to_string()),
        output: Some(output.display().to_string()),
        message: None,
        error: None,
        started_at_ms: now_ms(),
        updated_at_ms: now_ms(),
    };
    let job_id = job.id.clone();
    push_job(&state.jobs, job);
    let jobs = Arc::clone(&state.jobs);
    let verify = req.verify.unwrap_or(true);
    let force = req.force.unwrap_or(false);
    let thread_job_id = job_id.clone();
    thread::spawn(move || {
        let job_id = thread_job_id;
        update_job(
            &jobs,
            &job_id,
            "running",
            "Compiling and quantizing",
            None,
            None,
        );
        let result = (|| {
            let executable = resolve_tool("logan")?;
            let mut command = Command::new(executable);
            command
                .arg("compile")
                .arg(&source)
                .arg("--max-context")
                .arg(max_context.to_string())
                .arg("--target")
                .arg(target)
                .arg("--quant")
                .arg(quant)
                .arg("--quant-floor")
                .arg(quant_floor)
                .arg("--codec")
                .arg(codec)
                .arg("--opt")
                .arg(optimization)
                .arg("-o")
                .arg(&output);
            if verify {
                command.arg("--verify");
            }
            if force {
                command.arg("--force");
            }
            let output = command
                .output()
                .map_err(|e| format!("run Logan compiler: {e}"))?;
            if !output.status.success() {
                return Err(format!("Logan compile failed: {}", output_tail(&output)));
            }
            Ok(output_tail(&output))
        })();
        match result {
            Ok(message) => update_job(
                &jobs,
                &job_id,
                "completed",
                "Quantization complete",
                Some(if message.is_empty() {
                    "model compiled".into()
                } else {
                    message
                }),
                None,
            ),
            Err(error) => update_job(
                &jobs,
                &job_id,
                "failed",
                "Quantization failed",
                None,
                Some(error),
            ),
        }
    });
    (
        StatusCode::ACCEPTED,
        Json(json!({"ok":true,"jobId":job_id,"message":"quantization queued"})),
    )
        .into_response()
}

fn send_control(
    state: &AppState,
    control: Control,
    message: impl Into<String>,
) -> axum::response::Response {
    match state.control.send(control) {
        Ok(()) => (
            StatusCode::ACCEPTED,
            Json(ApiReply {
                ok: true,
                message: message.into(),
            }),
        )
            .into_response(),
        Err(_) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "daemon supervisor exited",
        ),
    }
}

fn api_error(status: StatusCode, message: impl Into<String>) -> axum::response::Response {
    (
        status,
        Json(ApiReply {
            ok: false,
            message: message.into(),
        }),
    )
        .into_response()
}

fn spawn_supervisor(
    snapshot: Arc<Mutex<DaemonSnapshot>>,
    logs: Arc<Mutex<VecDeque<LogEntry>>>,
    control_rx: mpsc::Receiver<Control>,
) {
    thread::Builder::new()
        .name("logan-daemon-supervisor".into())
        .spawn(move || supervisor_loop(snapshot, logs, control_rx))
        .expect("spawn Logan daemon supervisor");
}

fn supervisor_loop(
    snapshot: Arc<Mutex<DaemonSnapshot>>,
    logs: Arc<Mutex<VecDeque<LogEntry>>>,
    control_rx: mpsc::Receiver<Control>,
) {
    let started = Instant::now();
    let mut handle: Option<engine::EngineHandle> = None;
    let mut loaded_path: Option<PathBuf> = None;
    let mut system_prompt = DEFAULT_SYSTEM_PROMPT.to_string();
    let mut last_cache_scan = Instant::now() - Duration::from_secs(2);

    loop {
        while let Ok(control) = control_rx.try_recv() {
            match control {
                Control::Load { package, settings } => {
                    push_log(
                        &logs,
                        "info",
                        "model",
                        format!("loading {}", package.display()),
                    );
                    if let Some(old) = handle.take() {
                        let _ = old.tx.send(EngineCommand::Shutdown);
                    }
                    system_prompt = settings.system_prompt.clone();
                    loaded_path = Some(package.clone());
                    {
                        let mut s = snapshot.lock().unwrap();
                        s.status = "loading".into();
                        s.phase = "Starting inference worker".into();
                        s.model = Some(ModelView {
                            name: package
                                .file_name()
                                .and_then(|v| v.to_str())
                                .unwrap_or("model")
                                .to_string(),
                            path: package.display().to_string(),
                            context_limit: 0,
                            settings: settings.clone(),
                        });
                        s.performance = PerformanceView::default();
                        s.runtime = RuntimeView::default();
                        clear_hot_view(&mut s.cache);
                        s.assistant_text.clear();
                        s.error = None;
                        s.warning = None;
                        s.touch(started);
                    }
                    handle = Some(engine::spawn(package, system_prompt.clone()));
                }
                Control::Unload => {
                    push_log(&logs, "info", "model", "unloading current model");
                    if let Some(old) = handle.take() {
                        old.cancel.store(true, Ordering::Relaxed);
                        let _ = old.tx.send(EngineCommand::Shutdown);
                    }
                    loaded_path = None;
                    let mut s = snapshot.lock().unwrap();
                    s.status = "idle".into();
                    s.phase = "No model loaded".into();
                    s.model = None;
                    s.performance = PerformanceView::default();
                    s.runtime = RuntimeView::default();
                    clear_hot_view(&mut s.cache);
                    s.assistant_text.clear();
                    s.error = None;
                    s.touch(started);
                }
                Control::Generate { text, settings } => {
                    if let Some(h) = handle.as_ref() {
                        h.cancel.store(false, Ordering::Relaxed);
                        match h.tx.send(EngineCommand::Send { text, settings }) {
                            Ok(()) => {
                                let mut s = snapshot.lock().unwrap();
                                s.status = "generating".into();
                                s.phase = "Preparing prompt".into();
                                s.performance = PerformanceView {
                                    context_limit: s
                                        .model
                                        .as_ref()
                                        .map(|m| m.context_limit)
                                        .unwrap_or(0),
                                    ..Default::default()
                                };
                                s.assistant_text.clear();
                                s.error = None;
                                s.touch(started);
                            }
                            Err(_) => {
                                set_error(&snapshot, &logs, started, "inference worker exited")
                            }
                        }
                    } else {
                        set_error(&snapshot, &logs, started, "load a model before generating");
                    }
                }
                Control::Cancel => {
                    push_log(&logs, "info", "request", "cancellation requested");
                    if let Some(h) = handle.as_ref() {
                        h.cancel.store(true, Ordering::Relaxed);
                        let mut s = snapshot.lock().unwrap();
                        s.phase = "Cancelling after current token".into();
                        s.touch(started);
                    }
                }
                Control::Reset {
                    system_prompt: replacement,
                } => {
                    if let Some(replacement) = replacement.filter(|v| !v.trim().is_empty()) {
                        system_prompt = replacement;
                    }
                    if let Some(h) = handle.as_ref() {
                        h.cancel.store(true, Ordering::Relaxed);
                        let _ = h.tx.send(EngineCommand::Reset {
                            system_prompt: system_prompt.clone(),
                        });
                        let mut s = snapshot.lock().unwrap();
                        s.status = "loading".into();
                        s.phase = "Resetting hot causal state".into();
                        clear_hot_view(&mut s.cache);
                        s.performance = PerformanceView::default();
                        s.assistant_text.clear();
                        s.touch(started);
                    }
                }
                Control::ClearHot => {
                    push_log(&logs, "info", "cache", "clearing RAM prompt cache");
                    if let Some(h) = handle.as_ref() {
                        h.cancel.store(true, Ordering::Relaxed);
                        let _ = h.tx.send(EngineCommand::ClearHot);
                    }
                    let mut s = snapshot.lock().unwrap();
                    clear_hot_view(&mut s.cache);
                    s.performance = PerformanceView::default();
                    s.assistant_text.clear();
                    s.status = if handle.is_some() { "loading" } else { "idle" }.into();
                    s.phase = if handle.is_some() {
                        "Clearing hot prompt cache"
                    } else {
                        "No model loaded"
                    }
                    .into();
                    s.touch(started);
                }
                Control::ClearCold => {
                    push_log(
                        &logs,
                        "warn",
                        "cache",
                        "clearing persistent SSD prompt cache",
                    );
                    let cache_dir = snapshot.lock().unwrap().cache.directory.clone();
                    if !cache_dir.is_empty() {
                        let root = PathBuf::from(&cache_dir);
                        if let Err(e) = fs::remove_dir_all(&root) {
                            if e.kind() != std::io::ErrorKind::NotFound {
                                set_error(
                                    &snapshot,
                                    &logs,
                                    started,
                                    format!("clear cold cache: {e}"),
                                );
                            }
                        }
                        let _ = fs::create_dir_all(&root);
                    }
                    let mut s = snapshot.lock().unwrap();
                    s.cache.cold_entries = 0;
                    s.cache.cold_bytes = 0;
                    s.cache.last_cold_hit_tokens = 0;
                    s.cache.restore_ms = 0.0;
                    s.cache.write_ms = 0.0;
                    s.touch(started);
                }
                Control::Complete {
                    prompt,
                    settings,
                    updates,
                } => {
                    if let Some(h) = handle.as_ref() {
                        h.cancel.store(false, Ordering::Relaxed);
                        if h.tx
                            .send(EngineCommand::Complete {
                                prompt,
                                settings,
                                updates: updates.clone(),
                            })
                            .is_err()
                        {
                            let _ = updates
                                .send(CompletionUpdate::Error("inference worker exited".into()));
                        } else {
                            let mut s = snapshot.lock().unwrap();
                            s.status = "generating".into();
                            s.phase = "Preparing API request".into();
                            s.assistant_text.clear();
                            s.error = None;
                            s.touch(started);
                        }
                    } else {
                        let _ = updates.send(CompletionUpdate::Error(
                            "load a model before generating".into(),
                        ));
                    }
                }
            }
        }

        if let Some(h) = handle.as_ref() {
            while let Ok(event) = h.rx.try_recv() {
                apply_engine_event(&snapshot, &logs, started, &loaded_path, event);
            }
        }

        if last_cache_scan.elapsed() >= Duration::from_secs(1) {
            let cache_dir = snapshot.lock().unwrap().cache.directory.clone();
            if !cache_dir.is_empty() {
                let (entries, bytes) = scan_cold_cache(Path::new(&cache_dir));
                let mut s = snapshot.lock().unwrap();
                s.cache.cold_entries = entries;
                s.cache.cold_bytes = bytes;
                s.touch(started);
            } else {
                snapshot.lock().unwrap().touch(started);
            }
            last_cache_scan = Instant::now();
        }

        thread::sleep(Duration::from_millis(20));
    }
}

fn apply_engine_event(
    snapshot: &Arc<Mutex<DaemonSnapshot>>,
    logs: &Arc<Mutex<VecDeque<LogEntry>>>,
    started: Instant,
    loaded_path: &Option<PathBuf>,
    event: EngineEvent,
) {
    let mut s = snapshot.lock().unwrap();
    match event {
        EngineEvent::Loading(phase) => {
            push_log(logs, "info", "engine", phase.clone());
            s.status = "loading".into();
            s.phase = phase;
        }
        EngineEvent::Ready {
            model_name,
            context_limit,
            features,
            cache_dir,
        } => {
            push_log(
                logs,
                "info",
                "model",
                format!("{model_name} ready; context={context_limit}"),
            );
            let settings = s
                .model
                .as_ref()
                .map(|model| model.settings.clone())
                .unwrap_or_default();
            s.status = "ready".into();
            s.phase = "Model ready".into();
            s.model = Some(ModelView {
                name: model_name,
                path: loaded_path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default(),
                context_limit,
                settings,
            });
            s.performance.context_limit = context_limit;
            s.runtime.features = FeaturesView::from(&features);
            if !cache_dir.as_os_str().is_empty() {
                s.cache.directory = cache_dir.display().to_string();
            }
            s.error = None;
        }
        EngineEvent::Warning(warning) => {
            push_log(logs, "warn", "engine", warning.clone());
            s.warning = Some(warning);
        }
        EngineEvent::TurnStarted { metrics } => {
            s.status = "generating".into();
            s.phase = "Decoding".into();
            update_metrics(&mut s, &metrics, None);
        }
        EngineEvent::Token {
            chunk,
            token_id,
            metrics,
            stats,
        } => {
            s.status = "generating".into();
            s.phase = "Decoding".into();
            s.assistant_text.push_str(&chunk);
            update_metrics(&mut s, &metrics, Some(token_id));
            s.runtime = RuntimeView::from(&stats);
        }
        EngineEvent::TurnDone {
            text,
            metrics,
            stats,
        } => {
            push_log(
                logs,
                "info",
                "request",
                format!(
                    "completed: {} output tok, {:.2} tok/s, TTFT {:.1} ms, RAM hit {}, SSD hit {}",
                    metrics.generated_tokens,
                    if metrics.generation_ms > 0.0 {
                        metrics.generated_tokens as f64 * 1000.0 / metrics.generation_ms
                    } else {
                        0.0
                    },
                    metrics.first_token_ms,
                    metrics.ram_cached_tokens,
                    metrics.ssd_cached_tokens
                ),
            );
            s.status = "ready".into();
            s.phase = "Model ready".into();
            s.assistant_text = text;
            update_metrics(&mut s, &metrics, None);
            s.runtime = RuntimeView::from(&stats);
        }
        EngineEvent::ResetDone => {
            push_log(logs, "info", "engine", "model state reset complete");
            s.status = "ready".into();
            s.phase = "Model ready".into();
            clear_hot_view(&mut s.cache);
            s.performance = PerformanceView {
                context_limit: s.model.as_ref().map(|m| m.context_limit).unwrap_or(0),
                ..Default::default()
            };
        }
        EngineEvent::Error(error) => {
            push_log(logs, "error", "engine", error.clone());
            s.status = "error".into();
            s.phase = "Runtime error".into();
            s.error = Some(error);
        }
    }
    s.touch(started);
}

fn update_metrics(s: &mut DaemonSnapshot, m: &TurnMetrics, token_id: Option<u32>) {
    let context_limit = s.model.as_ref().map(|m| m.context_limit).unwrap_or(0);
    let prompt_tps = if m.prompt_ms > 0.0 {
        m.forwarded_prompt_tokens as f64 * 1000.0 / m.prompt_ms
    } else {
        0.0
    };
    let generation_tps = if m.generation_ms > 0.0 {
        m.generated_tokens as f64 * 1000.0 / m.generation_ms
    } else {
        0.0
    };
    s.performance = PerformanceView {
        prompt_tokens_per_second: prompt_tps,
        generation_tokens_per_second: generation_tps,
        ttft_ms: m.first_token_ms,
        prompt_ms: m.prompt_ms,
        generation_ms: m.generation_ms,
        total_ms: m.total_ms,
        generated_tokens: m.generated_tokens,
        forwarded_prompt_tokens: m.forwarded_prompt_tokens,
        context_tokens: m.context_tokens,
        context_limit,
        last_token_id: token_id.or(s.performance.last_token_id),
        stop_reason: m.stop_reason.as_ref().map(|v| v.label().to_string()),
    };
    s.cache.hot_entries = m.hot_cache_entries;
    s.cache.hot_tokens = m.hot_cache_tokens;
    s.cache.hot_bytes = m.hot_cache_bytes;
    s.cache.active_state_bytes = m.active_state_bytes;
    s.cache.live_reuse_tokens = m.live_reused_tokens;
    s.cache.last_hot_reuse_tokens = m.ram_cached_tokens;
    s.cache.hot_restore_ms = m.ram_cache_restore_ms;
    s.cache.hot_write_ms = m.ram_cache_write_ms;
    s.cache.last_cold_hit_tokens = m.ssd_cached_tokens;
    s.cache.restore_ms = m.cache_restore_ms;
    s.cache.write_ms = m.cache_write_ms;
}

fn clear_hot_view(cache: &mut CacheView) {
    cache.hot_entries = 0;
    cache.hot_tokens = 0;
    cache.hot_bytes = 0;
    cache.active_state_bytes = 0;
    cache.live_reuse_tokens = 0;
    cache.last_hot_reuse_tokens = 0;
    cache.hot_restore_ms = 0.0;
    cache.hot_write_ms = 0.0;
}

fn push_log(
    logs: &Arc<Mutex<VecDeque<LogEntry>>>,
    level: impl Into<String>,
    source: impl Into<String>,
    message: impl Into<String>,
) {
    const MAX_LOGS: usize = 1000;
    let mut logs = logs.lock().unwrap();
    logs.push_back(LogEntry {
        ts_ms: now_ms(),
        level: level.into(),
        source: source.into(),
        message: message.into(),
    });
    while logs.len() > MAX_LOGS {
        logs.pop_front();
    }
}

fn set_error(
    snapshot: &Arc<Mutex<DaemonSnapshot>>,
    logs: &Arc<Mutex<VecDeque<LogEntry>>>,
    started: Instant,
    error: impl Into<String>,
) {
    let error = error.into();
    push_log(logs, "error", "daemon", error.clone());
    let mut s = snapshot.lock().unwrap();
    s.status = "error".into();
    s.phase = "Runtime error".into();
    s.error = Some(error);
    s.touch(started);
}

fn scan_cold_cache(root: &Path) -> (usize, u64) {
    if root.as_os_str().is_empty() || !root.exists() {
        return (0, 0);
    }
    fn walk(path: &Path, entries: &mut usize, bytes: &mut u64) {
        let Ok(read_dir) = fs::read_dir(path) else {
            return;
        };
        for entry in read_dir.flatten() {
            let p = entry.path();
            if p.is_dir() {
                walk(&p, entries, bytes);
            } else if p.extension().and_then(|v| v.to_str()) == Some("lpfx") {
                *entries += 1;
                *bytes = bytes.saturating_add(entry.metadata().map(|m| m.len()).unwrap_or(0));
            }
        }
    }
    let mut entries = 0;
    let mut bytes = 0;
    walk(root, &mut entries, &mut bytes);
    (entries, bytes)
}

fn is_logan_package(path: &Path) -> bool {
    path.is_dir()
        && path.join("config.json").is_file()
        && path.join("tokenizer.json").is_file()
        && path.join("manifest.coli").is_file()
}

/// A safetensors checkpoint directory: `config.json` plus either a sharded
/// index or a single `model.safetensors`.
///
/// This is an open-weight HF layout, not a compiled Logan package, and it is
/// supported first-class: `StFile::open_dir` reads every shard named by the
/// index and decodes the stored dtypes (F32/BF16/F16/F8_E4M3), so a checkpoint
/// can run without a `.coli` build step. No `tokenizer.json` requirement here —
/// a raw checkpoint ships `tokenizer.json` OR the `vocab`/`merges` pair, and
/// `logand` resolves whichever is present.
fn is_safetensors_checkpoint(path: &Path) -> bool {
    path.is_dir()
        && path.join("config.json").is_file()
        && (path.join("model.safetensors.index.json").is_file()
            || path.join("model.safetensors").is_file())
}

/// Either supported source layout.
fn is_loadable_model(path: &Path) -> bool {
    is_logan_package(path) || is_safetensors_checkpoint(path)
}

fn discover_models(root: &Path) -> Vec<ModelCandidate> {
    let mut out = Vec::new();
    discover_models_inner(root, 0, &mut out);
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out
}

fn discover_models_inner(path: &Path, depth: usize, out: &mut Vec<ModelCandidate>) {
    if depth > 2 || !path.is_dir() {
        return;
    }
    if is_loadable_model(path) {
        out.push(ModelCandidate {
            name: path
                .file_name()
                .and_then(|v| v.to_str())
                .unwrap_or("model")
                .to_string(),
            path: path.display().to_string(),
            size_bytes: directory_size(path),
            settings: ModelSettings::default(),
        });
        return;
    }
    let Ok(read_dir) = fs::read_dir(path) else {
        return;
    };
    for entry in read_dir.flatten() {
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            discover_models_inner(&entry.path(), depth + 1, out);
        }
    }
}

fn directory_size(path: &Path) -> u64 {
    fn walk(path: &Path, total: &mut u64) {
        let Ok(read_dir) = fs::read_dir(path) else {
            return;
        };
        for entry in read_dir.flatten() {
            let p = entry.path();
            if p.is_dir() {
                walk(&p, total);
            } else {
                *total = total.saturating_add(entry.metadata().map(|m| m.len()).unwrap_or(0));
            }
        }
    }
    let mut total = 0;
    walk(path, &mut total);
    total
}

fn peak_rss_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        let ok = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0;
        if ok {
            return unsafe { usage.assume_init().ru_maxrss.max(0) as u64 };
        }
    }
    // Windows has no `getrusage`: report "not measured" rather than failing.
    // The daemon's status endpoint renders this as an absent RSS figure.
    0
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn expand_home(raw: &str) -> PathBuf {
    if raw == "~" {
        return std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(raw));
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(raw)
}

fn parse_args() -> Result<(IpAddr, u16, PathBuf), String> {
    let mut host: IpAddr = "127.0.0.1".parse().unwrap();
    let mut port = DEFAULT_PORT;
    let mut model_root = std::env::var_os("LOGAN_MODEL_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join("models")))
        .unwrap_or_else(|| PathBuf::from("models"));
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--host" => {
                let value = args.next().ok_or("--host requires an address")?;
                host = value
                    .parse()
                    .map_err(|_| format!("invalid --host: {value}"))?;
            }
            "--port" => {
                let value = args.next().ok_or("--port requires a number")?;
                port = value
                    .parse()
                    .map_err(|_| format!("invalid --port: {value}"))?;
            }
            "--model-dir" => {
                let value = args.next().ok_or("--model-dir requires a path")?;
                model_root = expand_home(&value);
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }
    Ok((host, port, model_root))
}

fn print_usage() {
    eprintln!(
        "Usage: logand [--host 127.0.0.1] [--port 11435] [--model-dir ~/models]\n\n\
         Background Logan inference daemon with an embedded web dashboard."
    );
}
