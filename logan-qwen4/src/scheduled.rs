//! Thin scheduler driver for Qwen4.
//!
//! The scheduler owns lifecycle/ticket state only. A single bounded worker
//! owns the model and performs forward passes, including Metal/MetalIO calls.
//! This keeps heavy work and device waits off the scheduler thread while
//! reusing the existing direct backend and CPU fallback.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{mpsc::SyncSender, Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use logan_core::sched::{
    Action, ActionKind, BoundedExecutor, Budget, CompletionSendError, DispatchRequest,
    ExecutorSet, Outcome, RuntimeConfig, RuntimeHandle, RuntimeReply, RuntimeRequest,
    SchedulerRuntime, SubmitResult,
};

use crate::Model;

const OP_PREFILL: u8 = 0;
const OP_DECODE: u8 = 1;
const OP_BYTES: usize = 17;

fn encode_op(kind: u8, token: usize, pos: usize) -> Vec<u8> {
    let mut payload = Vec::with_capacity(OP_BYTES);
    payload.push(kind);
    payload.extend_from_slice(&(token as u64).to_le_bytes());
    payload.extend_from_slice(&(pos as u64).to_le_bytes());
    payload
}

fn decode_op(payload: &[u8]) -> Result<(u8, usize, usize), String> {
    if payload.len() != OP_BYTES {
        return Err(format!("invalid scheduled Qwen action: {} bytes", payload.len()));
    }
    let token = u64::from_le_bytes(payload[1..9].try_into().unwrap()) as usize;
    let pos = u64::from_le_bytes(payload[9..17].try_into().unwrap()) as usize;
    Ok((payload[0], token, pos))
}

fn greedy_next(logits: &[f32]) -> Result<u32, String> {
    let mut best = None;
    for (index, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("non-finite Qwen logit at index {index}"));
        }
        if best.map_or(true, |(_, current)| value >= current) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index as u32)
        .ok_or_else(|| "empty Qwen logits".to_string())
}

fn execute_op(model: &mut Model, payload: &[u8]) -> Result<Option<u32>, String> {
    match decode_op(payload)? {
        (OP_PREFILL, token, pos) => {
            model.forward_token(token, pos);
            Ok(None)
        }
        (OP_DECODE, token, pos) => {
            let logits = model.forward_token(token, pos);
            let next = greedy_next(&logits)?;
            Ok(Some(next))
        }
        (kind, _, _) => Err(format!("unknown scheduled Qwen op {kind}")),
    }
}

#[derive(Default)]
struct Completed {
    items: VecDeque<(logan_core::sched::Ticket, Result<Option<u32>, String>)>,
}

struct CompletionLog {
    state: Mutex<Completed>,
    ready: Condvar,
}

impl CompletionLog {
    fn push(
        &self,
        ticket: logan_core::sched::Ticket,
        result: Result<Option<u32>, String>,
    ) {
        self.state.lock().unwrap().items.push_back((ticket, result));
        self.ready.notify_all();
    }

    fn wait(&self, ticket: logan_core::sched::Ticket) -> Result<Option<u32>, String> {
        let mut state = self.state.lock().unwrap();
        loop {
            if let Some(index) = state.items.iter().position(|(known, _)| *known == ticket) {
                return state.items.remove(index).unwrap().1;
            }
            state = self.ready.wait(state).unwrap();
        }
    }
}

struct QwenExecutor {
    sender: Option<SyncSender<DispatchRequest>>,
    worker: Option<JoinHandle<()>>,
}

impl QwenExecutor {
    fn new(package_dir: PathBuf, cfg: crate::Cfg, completed: Arc<CompletionLog>) -> Self {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<DispatchRequest>(1);
        let worker_completed = Arc::clone(&completed);
        let worker = thread::Builder::new()
            .name("logan-qwen-worker".into())
            .spawn(move || {
                let mut model = match crate::colisource::ColiSource::open(&package_dir)
                    .and_then(|src| Model::load_coli(&src, &cfg))
                {
                    Ok(model) => Some(model),
                    Err(error) => {
                        eprintln!("scheduled model load error: {error}");
                        None
                    }
                };
                let run_started = std::time::Instant::now();
                let mut generated = 0usize;
                while let Ok(request) = receiver.recv() {
                    let result = match model.as_mut() {
                        Some(model) => execute_op(model, &request.action.payload),
                        None => Err("scheduled model failed to load".into()),
                    };
                    match result {
                        Ok(output) => {
                            if output.is_some() {
                                generated += 1;
                            }
                            worker_completed.push(request.ticket, Ok(output));
                            send_completion(&request, Outcome::Ok);
                        }
                        Err(error) => {
                            worker_completed.push(request.ticket, Err(error.clone()));
                            send_completion(&request, Outcome::Err(error));
                        }
                    }
                }
                if generated > 0 {
                    if let Some(model) = model.as_mut() {
                        model.profile_summary(
                            generated,
                            run_started.elapsed().as_secs_f64() * 1e3,
                        );
                    }
                }
            })
            .expect("Qwen worker thread must be creatable");
        Self {
            sender: Some(sender),
            worker: Some(worker),
        }
    }
}

fn send_completion(request: &DispatchRequest, outcome: Outcome) {
    // The runtime channel is bounded. Retry happens on this worker, never on
    // the scheduler owner; each attempt remains nonblocking.
    loop {
        match request.completion.try_complete(request.ticket, outcome.clone()) {
            Ok(()) | Err(CompletionSendError::Closed) => return,
            Err(CompletionSendError::Full) => thread::yield_now(),
        }
    }
}

impl BoundedExecutor for QwenExecutor {
    fn try_submit(&mut self, request: DispatchRequest) -> SubmitResult {
        let Some(sender) = &self.sender else {
            return SubmitResult::Closed(request);
        };
        match sender.try_send(request) {
            Ok(()) => SubmitResult::Accepted,
            Err(std::sync::mpsc::TrySendError::Full(request)) => SubmitResult::Full(request),
            Err(std::sync::mpsc::TrySendError::Disconnected(request)) => {
                SubmitResult::Closed(request)
            }
        }
    }

    fn try_cancel(&mut self, _ticket: logan_core::sched::Ticket) -> bool {
        false
    }

    fn shutdown(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

struct ClosedExecutor;

impl BoundedExecutor for ClosedExecutor {
    fn try_submit(&mut self, request: DispatchRequest) -> SubmitResult {
        SubmitResult::Closed(request)
    }

    fn try_cancel(&mut self, _ticket: logan_core::sched::Ticket) -> bool {
        false
    }

    fn shutdown(&mut self) {}
}

fn recv_reply(receiver: std::sync::mpsc::Receiver<RuntimeReply>) -> Result<RuntimeReply, String> {
    receiver.recv().map_err(|_| "scheduler stopped".into())
}

fn request(handle: &RuntimeHandle, request: RuntimeRequest) -> Result<RuntimeReply, String> {
    let receiver = handle
        .try_request(request)
        .map_err(|error| format!("scheduler request rejected: {error:?}"))?;
    recv_reply(receiver)
}

fn submit_and_complete(
    handle: &RuntimeHandle,
    session: logan_core::sched::SessionId,
    payload: Vec<u8>,
    completed: &CompletionLog,
) -> Result<Option<u32>, String> {
    let reply = request(
        handle,
        RuntimeRequest::Submit {
            session,
            action: Action {
                kind: ActionKind::Accelerator,
                payload,
            },
        },
    )?;
    let ticket = match reply {
        RuntimeReply::Submitted { ticket, .. } => ticket,
        RuntimeReply::Error(error) => return Err(error),
        other => return Err(format!("unexpected submit reply: {other:?}")),
    };
    let output = completed.wait(ticket)?;
    // The worker has already submitted the real completion. This explicit
    // completion is harmless if it races or arrives second, and gives the
    // driver a deterministic acknowledgement before issuing the next quantum.
    let reply = request(
        handle,
        RuntimeRequest::Complete {
            ticket,
            outcome: Outcome::Ok,
        },
    )?;
    if let RuntimeReply::Error(error) = reply {
        return Err(error);
    }
    Ok(output)
}

/// Decode a `.coli` package through the bounded scheduler runtime.
pub fn run_greedy_scheduled(
    package_dir: &Path,
    prompt: &[u32],
    max_new: usize,
) -> Result<Vec<u32>, String> {
    let cfg = crate::load_cfg(&package_dir.join("config.json"))?;
    let completed = Arc::new(CompletionLog {
        state: Mutex::new(Completed::default()),
        ready: Condvar::new(),
    });
    let executors = ExecutorSet::new(
        Box::new(ClosedExecutor),
        Box::new(ClosedExecutor),
        Box::new(QwenExecutor::new(
            package_dir.to_path_buf(),
            cfg,
            Arc::clone(&completed),
        )),
    );
    let (handle, runtime) = SchedulerRuntime::spawn(
        Budget::default(),
        RuntimeConfig { command_capacity: 64 },
        executors,
    )
    .map_err(|error| format!("scheduler startup failed: {error:?}"))?;
    let result = (|| {
        let session = match request(&handle, RuntimeRequest::CreateSession)? {
            RuntimeReply::SessionCreated(session) => session,
            other => return Err(format!("unexpected session reply: {other:?}")),
        };
        for (pos, &token) in prompt.iter().enumerate() {
            submit_and_complete(&handle, session, encode_op(OP_PREFILL, token as usize, pos), &completed)?;
        }
        let mut output = Vec::with_capacity(max_new);
        let mut last = *prompt.last().unwrap_or(&0);
        for pos in prompt.len()..prompt.len() + max_new {
            let next = submit_and_complete(&handle, session, encode_op(OP_DECODE, last as usize, pos), &completed)?
                .ok_or_else(|| "decode completion had no token".to_string())?;
            output.push(next);
            last = next;
        }
        let _ = request(&handle, RuntimeRequest::Finish { session })?;
        Ok(output)
    })();
    let shutdown = request(&handle, RuntimeRequest::Shutdown { mode: logan_core::sched::ShutdownMode::Drain });
    let joined = runtime.join();
    if let Err(error) = joined {
        return Err(format!("scheduler thread failed: {error:?}"));
    }
    match (result, shutdown) {
        (Ok(output), Ok(_)) => Ok(output),
        (Err(error), _) | (_, Err(error)) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_encoding_round_trips() {
        assert_eq!(decode_op(&encode_op(OP_DECODE, 123, 456)).unwrap(), (OP_DECODE, 123, 456));
    }

    #[test]
    fn non_finite_logits_are_reported() {
        let error = greedy_next(&[0.0, f32::NAN]).unwrap_err();
        assert_eq!(error, "non-finite Qwen logit at index 1");
    }

    #[test]
    fn greedy_ties_keep_max_by_order() {
        assert_eq!(greedy_next(&[1.0, 3.0, 3.0]).unwrap(), 2);
    }
}
