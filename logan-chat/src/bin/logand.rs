use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
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
    normalize_responses_input, render_qwen_prompt, settings_from_chat, settings_from_responses,
};
use logan_qwen4::plan::{PrefixCacheStore, RuntimeFeatures, RuntimeStats};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_stream::wrappers::ReceiverStream;

const DASHBOARD: &str = include_str!("../dashboard.html");
const DEFAULT_PORT: u16 = 11435;
const DEFAULT_SYSTEM_PROMPT: &str = "You are an effective, careful assistant. Infer intent from the conversation, complete authorized work, and communicate directly. Ask only when missing information materially changes the result.";

#[derive(Clone)]
struct AppState {
    snapshot: Arc<Mutex<DaemonSnapshot>>,
    control: mpsc::Sender<Control>,
    model_root: PathBuf,
    responses: Arc<Mutex<HashMap<String, Vec<ApiMessage>>>>,
    logs: Arc<Mutex<VecDeque<LogEntry>>>,
}

#[derive(Clone, Debug)]
enum Control {
    Load {
        package: PathBuf,
        system_prompt: String,
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
    push_log(&logs, "info", "daemon", "logand starting");
    let (control_tx, control_rx) = mpsc::channel();
    spawn_supervisor(Arc::clone(&snapshot), Arc::clone(&logs), control_rx);

    let state = AppState {
        snapshot,
        control: control_tx,
        model_root,
        responses: Arc::new(Mutex::new(HashMap::new())),
        logs,
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
        .route("/api/model/load", post(api_load_model))
        .route("/api/model/unload", post(api_unload_model))
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
    let prompt = match render_qwen_prompt(&messages, None) {
        Ok(prompt) => prompt,
        Err(error) => return openai_error(StatusCode::BAD_REQUEST, error, "invalid_request_error"),
    };
    let settings = match settings_from_chat(&request) {
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
    let settings = match settings_from_responses(&request) {
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
    let prompt = match render_qwen_prompt(&history, request.instructions.as_deref()) {
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
    Json(discover_models(&state.model_root))
}

async fn api_load_model(
    State(state): State<AppState>,
    Json(req): Json<LoadRequest>,
) -> impl IntoResponse {
    let package = expand_home(&req.path);
    if !is_logan_package(&package) {
        return api_error(
            StatusCode::BAD_REQUEST,
            format!("not a Logan .coli package: {}", package.display()),
        );
    }
    let system_prompt = req
        .system_prompt
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());
    send_control(
        &state,
        Control::Load {
            package,
            system_prompt,
        },
        "loading model",
    )
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
    let mut settings = GenerationSettings::default();
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
                Control::Load {
                    package,
                    system_prompt: new_system,
                } => {
                    push_log(
                        &logs,
                        "info",
                        "model",
                        format!("loading {}", package.display()),
                    );
                    if let Some(old) = handle.take() {
                        let _ = old.tx.send(EngineCommand::Shutdown);
                    }
                    system_prompt = new_system;
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
            s.status = "ready".into();
            s.phase = "Model ready".into();
            s.model = Some(ModelView {
                name: model_name,
                path: loaded_path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default(),
                context_limit,
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
    if is_logan_package(path) {
        out.push(ModelCandidate {
            name: path
                .file_name()
                .and_then(|v| v.to_str())
                .unwrap_or("model")
                .to_string(),
            path: path.display().to_string(),
            size_bytes: directory_size(path),
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
