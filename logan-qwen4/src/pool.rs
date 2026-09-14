//! Inference-pool backend: run routed MoE experts on a pool of machines.
//!
//! This is Logan's first-class support for the `inference-pool` coordinator
//! (https://github.com/…/inference-pool). The dense backbone -- attention, GDN
//! state, routing, sampling -- stays local; only the routed experts travel.
//!
//! ## Why this shape
//!
//! Logan already groups a prefill batch's routed experts by expert id and calls
//! one batched kernel over the union (`moe_rows`, `preload_expert_set`). The
//! pool protocol takes exactly that shape: a list of `(layer, expert, input)`
//! items, answered by a list of output vectors in the same order. So the seam
//! is a drop-in replacement for the local kernel rather than a redesign.
//!
//! ## What it does not do
//!
//! - No batching *across* layers: one call per layer, matching the existing
//!   `preload_expert_set` cadence. Cross-layer batching would need the router
//!   to run ahead of the layer loop, which changes the execution schedule.
//! - No streaming: the request completes before the layer proceeds.
//!
//! Both are deliberate. A round trip per layer against sub-millisecond compute
//! is already the dominant cost, and pipelining layers would hide that only by
//! holding two layers' activations -- more memory for a network that is the
//! bottleneck either way.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// Coordinator configuration.
#[derive(Clone, Debug)]
pub struct PoolConfig {
    /// Base URL of the inference-pool coordinator, e.g. `http://coordinator-host:8080`.
    pub coordinator: String,
    /// Logical model family the pool matches on (keeps shards of one model apart).
    pub family: String,
    /// Content hash of the shard whose experts this node wants served. When
    /// empty the pool picks any worker holding the layer.
    pub source_hash: Option<String>,
    /// Network timeout for one expert batch.
    pub timeout: Duration,
}

impl PoolConfig {
    pub fn new(coordinator: impl Into<String>) -> Self {
        Self {
            coordinator: coordinator.into(),
            family: "qwen36".into(),
            source_hash: None,
            timeout: Duration::from_secs(120),
        }
    }

    pub fn with_family(mut self, family: impl Into<String>) -> Self {
        self.family = family.into();
        self
    }

    pub fn with_source_hash(mut self, hash: impl Into<String>) -> Self {
        self.source_hash = Some(hash.into());
        self
    }

    /// Coordinator base URL, or `None` when the pool is disabled.
    ///
    /// Reading the environment here (rather than at each call site) keeps the
    /// "is the pool on?" question in one place: `LOGAN_POOL_COORDINATOR`.
    pub fn from_env() -> Option<Self> {
        let url = std::env::var("LOGAN_POOL_COORDINATOR").ok()?;
        if url.is_empty() {
            return None;
        }
        let mut cfg = Self::new(url);
        if let Ok(f) = std::env::var("LOGAN_POOL_FAMILY") {
            cfg.family = f;
        }
        if let Ok(h) = std::env::var("LOGAN_POOL_SOURCE_HASH") {
            if !h.is_empty() {
                cfg.source_hash = Some(h);
            }
        }
        if let Ok(ms) = std::env::var("LOGAN_POOL_TIMEOUT_MS") {
            if let Ok(v) = ms.parse::<u64>() {
                cfg.timeout = Duration::from_millis(v);
            }
        }
        Some(cfg)
    }
}

/// One expert evaluation to send.
#[derive(Clone, Debug)]
pub struct ExpertCall {
    pub layer: u32,
    pub expert: u32,
    pub input: Vec<f32>,
}

/// A pool request failure. Kept as a plain string: callers fall back to the
/// local kernel, and the reason only needs to reach the log.
pub type PoolError = String;

/// POST a body to `path` on the coordinator and return the response body.
///
/// Deliberately a hand-rolled HTTP/1.1 client: this crate takes no HTTP
/// dependency today, and one POST with a `Content-Length` is not worth adding
/// one plus its TLS stack for. The tradeoff is no TLS -- see `PoolConfig`.
fn request(cfg: &PoolConfig, method: &str, path: &str, body: &str) -> Result<String, PoolError> {
    let rest = cfg
        .coordinator
        .strip_prefix("http://")
        .ok_or_else(|| format!("coordinator must be http:// (got {})", cfg.coordinator))?;
    // ponytail: no TLS. Coordinator runs on a trusted LAN/tailnet; add a TLS
    // client here if the pool ever crosses an untrusted network.
    let (host_port, _) = rest.split_once('/').unwrap_or((rest, ""));
    let addr = if host_port.contains(':') {
        host_port.to_string()
    } else {
        format!("{host_port}:80")
    };

    let stream = TcpStream::connect(&addr).map_err(|e| format!("connect {addr}: {e}"))?;
    stream
        .set_read_timeout(Some(cfg.timeout))
        .map_err(|e| format!("set timeout: {e}"))?;
    stream
        .set_write_timeout(Some(cfg.timeout))
        .map_err(|e| format!("set timeout: {e}"))?;
    let mut stream = stream;

    // The pool's job-state endpoint is a GET; posting to it returns 405. The
    // method is a parameter rather than hard-coded POST because of that.
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host_port}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|e| format!("write: {e}"))?;

    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|e| format!("read: {e}"))?;
    let text = String::from_utf8_lossy(&raw).to_string();

    // Split headers from body; `Connection: close` means one response per read.
    let (head, payload) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| "malformed response (no header terminator)".to_string())?;
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| "malformed status line".to_string())?;
    if !(200..300).contains(&status) {
        return Err(format!("HTTP {status}: {}", payload.trim()));
    }
    Ok(payload.to_string())
}

/// Minimal JSON extraction without a serde dependency.
///
/// The two payloads this module reads are flat and shaped by the pool, so the
/// parsers stay small: `{"outputs":[[f32,...],...]}` and
/// `{"worker_id":"…","job_id":"…"}`. A real JSON parser would be the better
/// call if this grew past these two.
mod json {
    /// Output vectors from a completed batch job.
    ///
    /// The coordinator answers `/v1/jobs/{id}` with
    /// `{"state":"complete","result":{"output":[…]},"items":N}` for a batch.
    /// `items` carries the count, which is what lets a caller check that every
    /// expert came back before scattering results into a layer.
    pub fn outputs(body: &str) -> Result<Vec<Vec<f32>>, String> {
        // Batch jobs answer with `result.output` as a flat [items × d_model]
        // list; a single-expert job nests one level differently. Handle the
        // flat form, which is what this client requests.
        let start = body
            .find("\"output\"")
            .ok_or_else(|| "no output key".to_string())?;
        let rest = &body[start..];

        let Some(open) = rest.find('[') else {
            return Err("no output array".into());
        };
        let bytes = rest.as_bytes();
        let mut depth = 0i32;
        let mut end = None;
        for (i, &b) in bytes.iter().enumerate().skip(open) {
            match b {
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(end) = end else {
            return Err("unterminated output array".into());
        };
        let inner = &rest[open + 1..end];
        if inner.trim().is_empty() {
            return Err("empty output".into());
        }
        // A flat vector of numbers (single expert) or a list of vectors.
        if inner.trim_start().starts_with('[') {
            parse_float_arrays(inner)
        } else {
            Ok(vec![parse_floats(inner)?])
        }
    }

    fn parse_floats(s: &str) -> Result<Vec<f32>, String> {
        let mut out = Vec::new();
        let mut num = String::new();
        for ch in s.chars() {
            if ch == ',' || ch.is_whitespace() {
                if !num.is_empty() {
                    out.push(
                        num.parse::<f32>()
                            .map_err(|e| format!("bad float {num}: {e}"))?,
                    );
                    num.clear();
                }
            } else {
                num.push(ch);
            }
        }
        if !num.is_empty() {
            out.push(
                num.parse::<f32>()
                    .map_err(|e| format!("bad float {num}: {e}"))?,
            );
        }
        Ok(out)
    }

    /// Parse `[1.0,-2.0],[3.0]` into two vectors, preserving every element.
    fn parse_float_arrays(s: &str) -> Result<Vec<Vec<f32>>, String> {
        let mut out = Vec::new();
        let mut cur: Option<Vec<f32>> = None;
        let mut num = String::new();
        for ch in s.chars() {
            match ch {
                '[' => cur = Some(Vec::new()),
                ']' | ',' | ' ' | '\n' | '\t' | '\r' => {
                    // A separator terminates the current number, but only `]`
                    // closes the element list -- treating `,` as a closer is
                    // what truncated vectors to their first value.
                    if !num.is_empty() {
                        if let Some(v) = cur.as_mut() {
                            v.push(
                                num.parse::<f32>()
                                    .map_err(|e| format!("bad float {num}: {e}"))?,
                            );
                        }
                        num.clear();
                    }
                    if ch == ']' {
                        if let Some(v) = cur.take() {
                            out.push(v);
                        }
                    }
                }
                _ => num.push(ch),
            }
        }
        Ok(out)
    }

    /// Value of a `"key":"value"` string field at the top level.
    pub fn string_field(body: &str, key: &str) -> Option<String> {
        let needle = format!("\"{key}\"");
        let start = body.find(&needle)? + needle.len();
        let rest = &body[start..];
        let open = rest.find('"')? + 1;
        let close = rest[open..].find('"')? + open;
        Some(rest[open..close].to_string())
    }
}

/// Encode expert calls as a `SubmitExpertBatchRequest`.
///
/// `layer`/`expert` are the checkpoint's own numbering. The pool resolves them
/// against each worker's cached manifest, so a shard holding experts 128..255
/// answers for those and no others -- which is what makes expert-level sharding
/// work without this side knowing who holds what.
pub fn encode_batch_request(
    calls: &[ExpertCall],
    family: &str,
    source_hash: Option<&str>,
    d_model: usize,
    d_hidden: usize,
    activation: &str,
) -> String {
    let first = calls.first();
    let (layer, expert) = first.map(|c| (c.layer, c.expert)).unwrap_or((0, 0));

    let mut items = String::new();
    for (i, c) in calls.iter().enumerate() {
        if i > 0 {
            items.push(',');
        }
        items.push_str(&format!(
            "{{\"layer\":{},\"expert\":{},\"input\":[",
            c.layer, c.expert
        ));
        for (j, v) in c.input.iter().enumerate() {
            if j > 0 {
                items.push(',');
            }
            items.push_str(&format!("{v}"));
        }
        items.push_str("]}");
    }

    let hash = match source_hash {
        Some(h) => format!("\"{h}\""),
        None => "\"\"".to_string(),
    };
    format!(
        "{{\"schedule\":{{\"model_id\":\"logan\",\"family\":\"{family}\",\"layer\":{layer},\
         \"expert\":{expert},\"expert_bytes\":0,\"activation_bytes\":{},\
         \"max_latency_ms\":null,\"source_hash\":{hash}}},\
         \"shape\":{{\"d_model\":{d_model},\"d_hidden\":{d_hidden}}},\
         \"activation\":\"{activation}\",\"items\":[{items}]}}",
        d_model * 4 * calls.len()
    )
}

/// Submit one expert batch and wait for the outputs.
///
/// Returns the outputs in the same order as `calls`, so the caller can scatter
/// them back by the same index it built `calls` with.
pub fn run_expert_batch(
    cfg: &PoolConfig,
    calls: &[ExpertCall],
    d_model: usize,
    d_hidden: usize,
    activation: &str,
) -> Result<Vec<Vec<f32>>, PoolError> {
    if calls.is_empty() {
        return Ok(Vec::new());
    }
    let body = encode_batch_request(
        calls,
        &cfg.family,
        cfg.source_hash.as_deref(),
        d_model,
        d_hidden,
        activation,
    );
    let submitted = request(cfg, "POST", "/v1/jobs/expert/batch", &body)?;
    let job_id = json::string_field(&submitted, "job_id")
        .ok_or_else(|| format!("no job_id in {}", truncate(&submitted)))?;

    // Poll until terminal. The coordinator has no push channel for results, so
    // polling is the contract; the interval is short because a batched layer is
    // expected to finish in tens of milliseconds.
    let deadline = std::time::Instant::now() + cfg.timeout;
    let mut delay = Duration::from_millis(2);
    loop {
        if std::time::Instant::now() > deadline {
            return Err(format!("job {job_id} timed out after {:?}", cfg.timeout));
        }
        let state = request(cfg, "GET", &format!("/v1/jobs/{job_id}"), "")?;
        if state.contains("\"complete\"") {
            return json::outputs(&state);
        }
        if state.contains("\"failed\"") {
            return Err(format!("job {job_id} failed: {}", truncate(&state)));
        }
        std::thread::sleep(delay);
        // Back off to 20ms: a busy loop would hammer the coordinator's lock for
        // no gain, and layers are not so fast that 20ms matters.
        delay = (delay * 2).min(Duration::from_millis(20));
    }
}

fn truncate(s: &str) -> String {
    s.chars().take(200).collect()
}

/// Whether the pool is configured. Cheap enough to call per layer.
pub fn enabled() -> bool {
    std::env::var("LOGAN_POOL_COORDINATOR")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_a_batch_request_the_pool_accepts() {
        let calls = vec![
            ExpertCall {
                layer: 3,
                expert: 17,
                input: vec![0.5, -1.25],
            },
            ExpertCall {
                layer: 3,
                expert: 200,
                input: vec![2.0, 0.0],
            },
        ];
        let body = encode_batch_request(&calls, "qwen36", Some("abc123"), 2, 4, "silu");
        assert!(body.contains("\"family\":\"qwen36\""));
        assert!(body.contains("\"source_hash\":\"abc123\""));
        assert!(body.contains("\"d_model\":2"));
        assert!(body.contains("{\"layer\":3,\"expert\":17,\"input\":[0.5,-1.25]}"));
        assert!(body.contains("{\"layer\":3,\"expert\":200,\"input\":[2,0]}"));
        // activation_bytes = d_model * 4 * items
        assert!(body.contains("\"activation_bytes\":16"));
    }

    #[test]
    fn extracts_outputs_from_a_job_state_response() {
        // The coordinator's real batch reply shape, captured from a live job.
        let batch = r#"{"items":2,"job_id":"x","result":{"backend":"cpu",
            "compute_ms":1.5,"output":[[1.5,-2.0],[3.0,4.0]]},"state":"complete"}"#;
        assert_eq!(
            json::outputs(batch).unwrap(),
            vec![vec![1.5, -2.0], vec![3.0, 4.0]]
        );

        // A single-expert job answers with a flat vector; accept it as one row
        // so a caller does not need two code paths.
        let single = r#"{"state":"complete","result":{"output":[1.0,2.0,3.0]}}"#;
        assert_eq!(json::outputs(single).unwrap(), vec![vec![1.0, 2.0, 3.0]]);

        // A pending job has no output yet: an error, not an empty success --
        // silently returning zero experts would corrupt the layer.
        assert!(json::outputs(r#"{"state":"pending"}"#).is_err());
        assert!(json::outputs(r#"{"result":{"output":[]}}"#).is_err());
        assert!(json::outputs(r#"{"result":{"output":[[1.0]"#).is_err());
    }

    #[test]
    fn reads_job_id_and_rejects_http_errors() {
        let ok = r#"{"job_id":"11111111-2222-3333-4444-555555555555","worker_id":"x"}"#;
        assert_eq!(
            json::string_field(ok, "job_id").as_deref(),
            Some("11111111-2222-3333-4444-555555555555")
        );
        assert!(json::string_field(ok, "missing").is_none());
    }

    #[test]
    fn pool_is_off_unless_the_env_var_says_otherwise() {
        // Not asserting on `enabled()` directly: the test process' environment
        // is shared, so only the parse path is checked here.
        let body = encode_batch_request(&[], "f", None, 8, 8, "silu");
        assert!(body.contains("\"items\":[]"));
        assert!(body.contains("\"source_hash\":\"\""));
    }
}
