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

    // Reuse one connection per coordinator instead of dialing per request.
    //
    // This is the dominant cost of the whole expert path, and it is not the
    // work: a fresh TCP connect from the worker's host measures **398 ms** on
    // the first request after idle and 15-34 ms afterwards, while the worker's
    // own persistent connection reports a 2.5 ms RTT. The path runs
    // the worker's LAN -> consumer-router NAT -> anchor, so a new connection pays
    // established one does not.
    //
    // The anchor sends one submit and one wait per layer, so per token that was
    // ~96 connections. Measured as `fill` = 24-30 s per token, which is the
    // entire expert phase and was previously misread as compute.
    // The pool's job-state endpoint is a GET; posting to it returns 405. The
    // method is a parameter rather than hard-coded POST because of that.
    //
    // `Connection: close` is gone: the response is read to a declared
    // Content-Length boundary instead, and the socket goes back to the pool.
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host_port}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\n\r\n{body}",
        body.len()
    );

    // A pooled socket may have been closed by the peer while idle, in which case
    // the write or read fails on a connection that was fine when it was stored.
    // Retry ONCE on a fresh connection rather than probing liveness up front: a
    // probe would cost a round trip on the happy path, which is the path that
    // matters here, and a genuinely dead coordinator still fails on the retry.
    let mut retried = false;
    let (head, payload) = loop {
        let mut stream = connection(cfg, &addr)?;
        let attempt = (|| -> Result<(String, String), PoolError> {
            stream
                .write_all(req.as_bytes())
                .map_err(|e| format!("write: {e}"))?;
            // Read headers, then exactly Content-Length bytes of body.
            // `read_to_end` cannot be used once the connection is kept alive: it
            // would block until the peer closed, which it no longer does.
            read_response(&mut stream)
        })();
        match attempt {
            Ok(ok) => {
                // Return the socket for reuse only on success, so a failed
                // exchange never leaves a half-read stream in the pool.
                if let Ok(mut pool) = pool_conns().lock() {
                    if pool.len() < 4 {
                        pool.insert(addr.clone(), stream);
                    }
                }
                break ok;
            }
            Err(e) => {
                drop(stream);
                if retried {
                    return Err(e);
                }
                retried = true;
                continue;
            }
        }
    };

    let head = head.as_str();
    let payload = payload.as_str();
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

/// Idle keep-alive connections, keyed by coordinator address.
///
/// Small and unbounded-free: a caller talks to one coordinator, and the cap
/// keeps a misconfigured caller from accumulating sockets.
fn pool_conns() -> &'static std::sync::Mutex<std::collections::HashMap<String, TcpStream>> {
    static CELL: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, TcpStream>>> =
        std::sync::OnceLock::new();
    CELL.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// A pooled connection to `addr`, or a fresh one.
///
/// A pooled socket may have been closed by the peer while idle, so the caller's
/// first write can fail. That is handled by retrying once on a new connection
/// rather than by probing, which would cost a round trip on the happy path.
fn connection(cfg: &PoolConfig, addr: &str) -> Result<TcpStream, PoolError> {
    if let Ok(mut pool) = pool_conns().lock() {
        if let Some(s) = pool.remove(addr) {
            return Ok(s);
        }
    }
    let stream = TcpStream::connect(addr).map_err(|e| format!("connect {addr}: {e}"))?;
    stream
        .set_read_timeout(Some(cfg.timeout))
        .map_err(|e| format!("set timeout: {e}"))?;
    stream
        .set_write_timeout(Some(cfg.timeout))
        .map_err(|e| format!("set timeout: {e}"))?;
    Ok(stream)
}

/// Read one HTTP/1.1 response: `(headers, body)`.
///
/// Content-Length delimited rather than read-to-EOF, because the connection is
/// reused. A chunked response is refused rather than guessed at: the coordinator
/// always sets Content-Length, so a chunked reply means something is wrong and
/// silently mis-parsing it would corrupt expert output.
fn read_response(stream: &mut TcpStream) -> Result<(String, String), PoolError> {
    use std::io::Read as _;
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    let head_end = loop {
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos + 4;
        }
        let mut chunk = [0u8; 4096];
        let n = stream.read(&mut chunk).map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            return Err("connection closed before headers completed".into());
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let len = header_value(&head, "content-length")
        .and_then(|v| v.trim().parse::<usize>().ok())
        .ok_or_else(|| format!("response has no Content-Length: {}", head.lines().next().unwrap_or("")))?;
    while buf.len() - head_end < len {
        let mut chunk = [0u8; 8192];
        let n = stream.read(&mut chunk).map_err(|e| format!("read body: {e}"))?;
        if n == 0 {
            return Err(format!(
                "connection closed after {} of {len} body bytes",
                buf.len() - head_end
            ));
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    Ok((head, String::from_utf8_lossy(&buf[head_end..head_end + len]).to_string()))
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Case-insensitive header lookup.
fn header_value(head: &str, name: &str) -> Option<String> {
    head.lines()
        .skip(1)
        .find(|l| {
            l.split_once(':')
                .is_some_and(|(k, _)| k.trim().eq_ignore_ascii_case(name))
        })
        .and_then(|l| l.split_once(':').map(|(_, v)| v.to_string()))
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
    /// Parse `result.output` into one vector per item.
    ///
    /// `row_width` is the output width of ONE item (`d_model`). It is needed
    /// because the coordinator answers a batch with a FLAT
    /// `[items x row_width]` array — 10 items of 2560 comes back as 25600
    /// numbers with no structure marking the boundaries. Without the width the
    /// only honest reading of a flat array is "one very long vector", which is
    /// how a 10-expert layer silently became a 1-expert one.
    ///
    /// Bodies that ARE nested (a single-expert job, or a future coordinator that
    /// nests) are returned as-is, with the width used only to validate.
    pub fn outputs_with_width(body: &str, row_width: usize) -> Result<Vec<Vec<f32>>, String> {
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
        // A list of vectors (nested) ...
        if inner.trim_start().starts_with('[') {
            return parse_float_arrays(inner);
        }
        // ... or a flat run of numbers, which must be chunked by item width.
        let flat = parse_floats(inner)?;
        // An empty output is "no experts ran", which must never look like a
        // successful layer: a caller accumulating this would compute with the
        // experts silently missing.
        if flat.is_empty() {
            return Err("empty output".into());
        }
        if row_width == 0 {
            return Err("flat output needs a nonzero row width to split".into());
        }
        if flat.len() % row_width != 0 {
            return Err(format!(
                "flat output of {} is not a whole number of {row_width}-wide items",
                flat.len()
            ));
        }
        Ok(flat.chunks(row_width).map(|c| c.to_vec()).collect())
    }

    /// Back-compat wrapper for a caller that has no item width.
    ///
    /// Only useful for a NESTED body or a genuine single-item flat body; a flat
    /// multi-item body is indistinguishable from one long vector without a
    /// width, so callers handling batches must use
    /// [`outputs_with_width`]. Kept so a probe or test can parse a reply
    /// without knowing the model's width.
    pub fn outputs(body: &str) -> Result<Vec<Vec<f32>>, String> {
        // A flat multi-item body cannot be split here; the width is unknown by
        // construction. Nested bodies are unambiguous, so parse those, and for a
        // flat body return it as ONE vector (the historical single-expert shape)
        // after the same validation the width-aware path applies.
        //
        // Truncation must still be detected: an unterminated array is an error
        // rather than a short read that looks successful.
        let start = body.find("\"output\"").ok_or("no output key")?;
        let rest = &body[start..];
        let open = rest.find('[').ok_or("no output array")?;
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
        let end = end.ok_or("unterminated output array")?;
        let inner = &rest[open + 1..end];
        if inner.trim().is_empty() {
            return Err("empty output".into());
        }
        if inner.trim_start().starts_with('[') {
            return parse_float_arrays(inner);
        }
        let v = parse_floats(inner)?;
        if v.is_empty() {
            return Err("empty output".into());
        }
        Ok(vec![v])
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

    use std::fmt::Write as _;

    // Build the body into ONE preallocated buffer, writing floats directly into
    // it.
    //
    // The previous version called `format!("{v}")` per float, which allocates a
    // fresh `String` for every value: 2560 values x 10 experts x 48 layers x 7
    // token-passes = 8.6 million allocations per run, and the resulting pieces
    // were then copied again into `items` and copied again into the final body.
    // Float formatting itself is unavoidable here, but the allocations are not.
    //
    // Capacity is estimated from the inputs so the buffer usually does not
    // reallocate: a float prints in at most ~14 bytes, plus ~48 of framing per
    // item.
    let est: usize = calls
        .iter()
        .map(|c| c.input.len() * 14 + 48)
        .sum::<usize>()
        + 256;
    let mut out = String::with_capacity(est);

    let _ = write!(
        out,
        "{{\"schedule\":{{\"model_id\":\"logan\",\"family\":\"{family}\",\"layer\":{layer},\
         \"expert\":{expert},\"expert_bytes\":0,\"activation_bytes\":{},\
         \"max_latency_ms\":null,\"source_hash\":",
        d_model * 4 * calls.len()
    );
    match source_hash {
        Some(h) => {
            let _ = write!(out, "\"{h}\"");
        }
        None => out.push_str("\"\""),
    }
    let _ = write!(
        out,
        "}},\"shape\":{{\"d_model\":{d_model},\"d_hidden\":{d_hidden}}},\
         \"activation\":\"{activation}\",\"items\":["
    );
    for (i, c) in calls.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{{\"layer\":{},\"expert\":{},\"input\":[", c.layer, c.expert);
        for (j, v) in c.input.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            let _ = write!(out, "{v}");
        }
        out.push_str("]}");
    }
    out.push_str("]}");
    out
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

    // Wait for the result with the coordinator's LONG POLL, not a sleep loop.
    //
    // `/v1/jobs/{id}?wait=true&wait_ms=N` blocks on the coordinator until the job
    // is terminal or the window expires, and the worker's 16-byte result-announce
    // datagram wakes it — so the reply arrives as soon as the work does.
    //
    // This matters more than it looks. The previous version slept 2ms, then 4, 8,
    // 16, and settled at **20 ms per check**, so a layer that finished in 6ms was
    // still reported up to 20ms late, and the anchor paid that 48 times per token.
    // Measured end to end: 508 ms per layer with the sleep loop. The sleep was
    // also pure overhead on a path whose whole job is to be short.
    let deadline = std::time::Instant::now() + cfg.timeout;
    // One long-poll window per request, bounded so a cancellation or a dead
    // coordinator is still noticed. Re-issued until the overall deadline.
    let window_ms = 5000u64;
    loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            return Err(format!("job {job_id} timed out after {:?}", cfg.timeout));
        }
        let this_window = window_ms.min(remaining.as_millis().max(1) as u64);
        let state = request(
            cfg,
            "GET",
            &format!("/v1/jobs/{job_id}?wait=true&wait_ms={this_window}"),
            "",
        )?;
        if state.contains("\"complete\"") {
            // A batch answers flat, so the item width is required to split it.
            // `d_model` is the output width of one expert evaluation.
            return json::outputs_with_width(&state, d_model);
        }
        if state.contains("\"failed\"") {
            return Err(format!("job {job_id} failed: {}", truncate(&state)));
        }
        // Still pending after a full window: loop and wait again. No sleep here —
        // the coordinator already waited, and adding one would reintroduce the
        // latency this replaced.
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
    fn splits_a_flat_batch_reply_by_item_width() {
        // Captured from a live 10-item batch: the coordinator returns a FLAT
        // `items x d_model` run with nothing marking item boundaries. Reading
        // that as one vector is how a 10-expert layer silently became 1 expert
        // and the layer computed with 9 experts missing.
        let mut nums = Vec::new();
        for item in 0..10 {
            for col in 0..4 {
                nums.push(format!("{}", item as f32 + col as f32 / 10.0));
            }
        }
        let body = format!(
            r#"{{"items":10,"state":"complete","result":{{"output":[{}]}}}}"#,
            nums.join(",")
        );
        let out = json::outputs_with_width(&body, 4).unwrap();
        assert_eq!(out.len(), 10, "one vector per item");
        assert!(out.iter().all(|v| v.len() == 4));
        assert_eq!(out[0], vec![0.0, 0.1, 0.2, 0.3]);
        assert_eq!(out[9], vec![9.0, 9.1, 9.2, 9.3]);

        // A run that is not a whole number of items is an error, never a
        // silently truncated or padded layer.
        let ragged = r#"{"state":"complete","result":{"output":[1.0,2.0,3.0]}}"#;
        assert!(json::outputs_with_width(ragged, 2).is_err());

        // Nested replies still parse, and the width only validates them.
        let nested = r#"{"state":"complete","result":{"output":[[1.0,2.0],[3.0,4.0]]}}"#;
        assert_eq!(
            json::outputs_with_width(nested, 2).unwrap(),
            vec![vec![1.0, 2.0], vec![3.0, 4.0]]
        );
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
