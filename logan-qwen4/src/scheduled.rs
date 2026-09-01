//! Thin scheduler driver for Qwen4 (issue #53, WS2d slice).
//!
//! The scheduler owns lifecycle/ticket state only. A single bounded worker
//! owns the model and performs forward passes, including Metal/MetalIO calls
//! (the executor lane owns backend waits; the scheduler thread never waits
//! on I/O or device work).
//!
//! Scheduler-driven decode (QWEN_SCHED=1): the routed-MoE phase reports cold
//! experts instead of loading them (`Model::forward_scheduled`). The driver
//! then runs the #52 contract per cold expert: residency reserve (`#47`
//! `begin_load`) -> exactly one async native record load (MetalIO, the macOS
//! compiled-record I/O provider) issued and drained on the executor lane ->
//! completion ticket -> resident publish (`complete_load`) -> same op
//! resubmitted, which resumes the forward from its stashed cursor. The
//! engine SlotExpert LRU remains the physical store; the residency manager
//! is the logical key/state view over it (bridged, never rebuilt).

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{mpsc::SyncSender, Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use logan_core::sched::{
    device::{DeviceKind, DeviceRegistry, ExecutionTarget},
    ids::{LoadId, SessionId},
    residency::{
        CompletionDisposition, DeviceId, ExpertKey, LoadCompletion, LoadDisposition, LoadResult,
        MemoryPoolId, RepresentationKey, ResidencyManager, ResidencyState,
    },
    Action, ActionKind, BoundedExecutor, Budget, CompletionSendError, DispatchRequest,
    ExecutorSet, Outcome, RuntimeConfig, RuntimeHandle, RuntimeReply, RuntimeRequest,
    SchedulerRuntime, SubmitResult,
};

use crate::plan::Plan;
use crate::{Model, SchedForward};

const OP_PREFILL: u8 = 0;
const OP_DECODE: u8 = 1;
const OP_LOAD: u8 = 2;
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

/// Generic step outcome the worker reports for one dispatched action
/// (issue #53 engine boundary: the scheduler sees step outcomes, never
/// model internals).
#[derive(Debug, Clone, PartialEq)]
pub enum SchedOutcome {
    /// The action completed: `Some(token)` for a decode action.
    Token(Option<u32>),
    /// The forward stopped at `layer` because the listed routed experts are
    /// cold. The driver loads them and resubmits the SAME action.
    NeedExperts { layer: usize, experts: Vec<u32> },
}

fn execute_op(model: &mut Model, plan: &Plan, payload: &[u8]) -> Result<SchedOutcome, String> {
    match decode_op(payload)? {
        (OP_PREFILL, token, pos) => match model.forward_scheduled(token, pos) {
            SchedForward::Logits(_) => Ok(SchedOutcome::Token(None)),
            SchedForward::NeedExperts { layer, experts } => {
                Ok(SchedOutcome::NeedExperts { layer, experts })
            }
        },
        (OP_DECODE, token, pos) => match model.forward_scheduled(token, pos) {
            SchedForward::Logits(logits) => Ok(SchedOutcome::Token(Some(greedy_next(&logits)?))),
            SchedForward::NeedExperts { layer, experts } => {
                Ok(SchedOutcome::NeedExperts { layer, experts })
            }
        },
        (OP_LOAD, li, ei) => {
            let planned = plan.planned(li, ei).ok_or_else(|| {
                format!("planned expert ({li},{ei}) is not in the run plan")
            })?;
            model.load_expert_planned(li as i32, ei as i32, planned)?;
            Ok(SchedOutcome::Token(None))
        }
        (kind, _, _) => Err(format!("unknown scheduled Qwen op {kind}")),
    }
}

#[derive(Default)]
struct Completed {
    items: VecDeque<(logan_core::sched::Ticket, Result<SchedOutcome, String>)>,
}

struct CompletionLog {
    state: Mutex<Completed>,
    ready: Condvar,
}

impl CompletionLog {
    fn push(&self, ticket: logan_core::sched::Ticket, result: Result<SchedOutcome, String>) {
        self.state.lock().unwrap().items.push_back((ticket, result));
        self.ready.notify_all();
    }

    fn wait(&self, ticket: logan_core::sched::Ticket) -> Result<SchedOutcome, String> {
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
    fn new(
        coli: crate::colisource::ColiSource,
        cfg: crate::Cfg,
        plan: Arc<Plan>,
        completed: Arc<CompletionLog>,
    ) -> Self {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<DispatchRequest>(1);
        let worker_completed = Arc::clone(&completed);
        let worker = thread::Builder::new()
            .name("logan-qwen-worker".into())
            .spawn(move || {
                let t_open = std::time::Instant::now();
                let mut model = match Model::load_coli(&coli, &cfg) {
                    Ok(mut model) => {
                        // Scheduler-driven expert acquisition: the model
                        // reports cold experts instead of loading them.
                        model.enable_sched_mode();
                        Some(model)
                    }
                    Err(error) => {
                        eprintln!("scheduled model load error: {error}");
                        None
                    }
                };
                if std::env::var("LOGAN_PROFILE").map(|v| v != "0").unwrap_or(false) {
                    eprintln!(
                        "logan load: {:.1} s (package open + weights)",
                        t_open.elapsed().as_secs_f64()
                    );
                }
                let run_started = std::time::Instant::now();
                let mut generated = 0usize;
                while let Ok(request) = receiver.recv() {
                    let result = match model.as_mut() {
                        Some(model) => execute_op(model, &plan, &request.action.payload),
                        None => Err("scheduled model failed to load".into()),
                    };
                    match result {
                        Ok(outcome) => {
                            if matches!(outcome, SchedOutcome::Token(Some(_))) {
                                generated += 1;
                            }
                            worker_completed.push(request.ticket, Ok(outcome));
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
                        model.profile_summary(generated, run_started.elapsed().as_secs_f64() * 1e3);
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

/// One worker shared by the accelerator and IO lanes (the worker performs
/// both compute and the MetalIO record loads; V1 is strictly sequential, so
/// a single bounded queue is honest). The mutex only ever contends on the
/// scheduler owner thread.
struct SharedExecutor(Arc<Mutex<QwenExecutor>>);

impl BoundedExecutor for SharedExecutor {
    fn try_submit(&mut self, request: DispatchRequest) -> SubmitResult {
        self.0.lock().unwrap().try_submit(request)
    }

    fn try_cancel(&mut self, ticket: logan_core::sched::Ticket) -> bool {
        self.0.lock().unwrap().try_cancel(ticket)
    }

    fn shutdown(&mut self) {
        self.0.lock().unwrap().shutdown();
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
    session: SessionId,
    kind: ActionKind,
    payload: Vec<u8>,
    completed: &CompletionLog,
) -> Result<SchedOutcome, String> {
    let reply = request(
        handle,
        RuntimeRequest::Submit {
            session,
            action: Action {
                kind,
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

/// Residency bridge (issue #47): the engine SlotExpert LRU stays the
/// physical store; this manager is the scheduler's logical key/state view,
/// driven by the load-ticket lifecycle. Single-owner: the driver thread.
struct ResidencyBridge<'a> {
    residency: ResidencyManager,
    registry: DeviceRegistry,
    target: ExecutionTarget,
    plan: &'a Plan,
}

impl<'a> ResidencyBridge<'a> {
    fn new(plan: &'a Plan) -> Result<Self, String> {
        let mut registry = DeviceRegistry::new();
        let target = ExecutionTarget {
            device: DeviceId(0),
            kind: DeviceKind::Gpu,
            capabilities: vec![ActionKind::Accelerator, ActionKind::Io],
        };
        registry
            .register(target.clone())
            .map_err(|e| format!("execution target registration: {e:?}"))?;
        let mut residency = ResidencyManager::new();
        let pool = MemoryPoolId(0);
        residency
            .add_pool(pool, plan.total_bytes())
            .map_err(|e| format!("residency pool: {e:?}"))?;
        residency
            .add_device(DeviceId(0), vec![pool])
            .map_err(|e| format!("residency device: {e:?}"))?;
        Ok(ResidencyBridge {
            residency,
            registry,
            target,
            plan,
        })
    }

    /// The planned representation identity for one expert. Convention for
    /// this package: layout 1 = Apple8 MetalIO slot (gate/up/down packed),
    /// kernel_abi 1 = fused moe_topk ABI, quant 1 = MXFP4. Only the bridge
    /// interprets these; the scheduler never does.
    fn key(&self, layer: u32, expert: u32) -> ExpertKey {
        ExpertKey {
            model: 1,
            package: 1,
            layer,
            expert,
            representation: RepresentationKey {
                layout: 1,
                kernel_abi: 1,
                quant: 1,
            },
            pool: MemoryPoolId(0),
        }
    }

    /// #56 dispatch validation: the submitted target must be the registered
    /// target for its device and must accept the action kind.
    fn check_target(&self, kind: ActionKind) -> Result<(), String> {
        let registered = self
            .registry
            .get(self.target.device)
            .ok_or_else(|| "execution target not registered".to_string())?;
        if *registered != self.target {
            return Err("registered execution target mismatch".into());
        }
        if !registered.accepts(kind) {
            return Err(format!("execution target {kind:?} is not accepted"));
        }
        Ok(())
    }

    /// Reserve destination memory for the exact planned representation and
    /// start (or join) the loader lifecycle. Returns the `load_id` to
    /// publish on the ticket's success, or `None` when the physical store
    /// holds a freshly evicted key the view still considers resident (the
    /// V1 view has no eviction path; the engine LRU is the real budget) —
    /// the caller still issues the physical load and the ticket, just
    /// without a manager lifecycle for this key.
    fn request_load(
        &mut self,
        session: SessionId,
        layer: u32,
        expert: u32,
    ) -> Result<Option<LoadId>, String> {
        let key = self.key(layer, expert);
        let bytes = self
            .plan
            .planned(layer as usize, expert as usize)
            .map(|p| p.slot_bytes())
            .unwrap_or(0);
        match self
            .residency
            .begin_load(key, bytes, session)
            .map_err(|e| format!("residency begin_load: {e:?}"))?
        {
            LoadDisposition::Started { load_id } => Ok(Some(load_id)),
            LoadDisposition::AlreadyResident { .. } | LoadDisposition::Joined { .. } => Ok(None),
        }
    }

    /// Publish a successful physical load into the view (waiter leases are
    /// released immediately; the resumed forward re-acquires through the
    /// physical store).
    fn publish(&mut self, load_id: LoadId) -> Result<(), String> {
        let disposition = self
            .residency
            .complete_load(LoadCompletion {
                load_id,
                result: LoadResult::Success,
            })
            .map_err(|e| format!("residency publish: {e:?}"))?;
        if let CompletionDisposition::Published { wakes, .. } = disposition {
            for wake in wakes {
                if let Ok(lease) = wake.result {
                    let _ = self.residency.release(lease);
                }
            }
        }
        Ok(())
    }

    fn state(&self, layer: u32, expert: u32) -> ResidencyState {
        self.residency.state(&self.key(layer, expert))
    }
}

/// Submit one action through the scheduler; when the worker reports cold
/// routed experts, run the #52 contract for each (reserve -> async native
/// record load ticket -> publish) and resubmit the SAME action until it
/// completes. The sequence (this thread) blocks on completions; the
/// scheduler thread never waits on I/O.
fn step_op(
    handle: &RuntimeHandle,
    session: SessionId,
    bridge: &mut ResidencyBridge,
    completed: &CompletionLog,
    payload: Vec<u8>,
) -> Result<Option<u32>, String> {
    loop {
        let kind = match payload[0] {
            OP_PREFILL | OP_DECODE => ActionKind::Accelerator,
            OP_LOAD => ActionKind::Io,
            other => return Err(format!("unknown scheduled Qwen op {other}")),
        };
        bridge.check_target(kind)?;
        let outcome = submit_and_complete(handle, session, kind, payload.clone(), completed)?;
        match outcome {
            SchedOutcome::Token(token) => return Ok(token),
            SchedOutcome::NeedExperts { layer, experts } => {
                for &ei in &experts {
                    let load_id = bridge.request_load(session, layer as u32, ei)?;
                    match submit_and_complete(
                        handle,
                        session,
                        ActionKind::Io,
                        encode_op(OP_LOAD, layer, ei as usize),
                        completed,
                    )? {
                        SchedOutcome::Token(_) => {}
                        other => {
                            return Err(format!("load action returned {other:?}"));
                        }
                    }
                    if let Some(load_id) = load_id {
                        bridge.publish(load_id)?;
                    }
                }
                // Resume: resubmit the same action; the worker resumes the
                // stashed forward cursor (same token/pos).
            }
        }
    }
}

/// Single-session decode through the bounded scheduler runtime.
fn run_session(
    handle: &RuntimeHandle,
    plan: &Plan,
    completed: &CompletionLog,
    prompt: &[u32],
    max_new: usize,
) -> Result<Vec<u32>, String> {
    let session = match request(handle, RuntimeRequest::CreateSession)? {
        RuntimeReply::SessionCreated(session) => session,
        other => return Err(format!("unexpected session reply: {other:?}")),
    };
    let mut bridge = ResidencyBridge::new(plan)?;
    let result = run_session_inner(handle, session, &mut bridge, completed, prompt, max_new);
    let _ = request(handle, RuntimeRequest::Finish { session })?;
    result
}

/// The per-session op loop (split out so seam tests can drive the exact
/// production sequence and then inspect the bridge).
fn run_session_inner(
    handle: &RuntimeHandle,
    session: SessionId,
    bridge: &mut ResidencyBridge,
    completed: &CompletionLog,
    prompt: &[u32],
    max_new: usize,
) -> Result<Vec<u32>, String> {
    for (pos, &token) in prompt.iter().enumerate() {
        step_op(
            handle,
            session,
            bridge,
            completed,
            encode_op(OP_PREFILL, token as usize, pos),
        )?;
    }
    let mut output = Vec::with_capacity(max_new);
    let mut last = *prompt.last().unwrap_or(&0);
    for pos in prompt.len()..prompt.len() + max_new {
        let next = step_op(
            handle,
            session,
            bridge,
            completed,
            encode_op(OP_DECODE, last as usize, pos),
        )?
        .ok_or_else(|| "decode completion had no token".to_string())?;
        output.push(next);
        last = next;
    }
    Ok(output)
}

/// Decode a `.coli` package through the bounded scheduler runtime
/// (QWEN_SCHED=1). The plan is resolved once from the validated package; the
/// worker owns the model + MetalIO; the scheduler thread never blocks on I/O.
pub fn run_greedy_scheduled(package_dir: &Path, prompt: &[u32], max_new: usize) -> Result<Vec<u32>, String> {
    let cfg = crate::load_cfg(&package_dir.join("config.json"))?;
    let coli = crate::colisource::ColiSource::open(package_dir)?;
    let plan = Arc::new(Plan::resolve(coli.pkg_ref(), cfg.layers, cfg.experts)?);
    let completed = Arc::new(CompletionLog {
        state: Mutex::new(Completed::default()),
        ready: Condvar::new(),
    });
    let worker = QwenExecutor::new(coli, cfg, Arc::clone(&plan), Arc::clone(&completed));
    let shared = Arc::new(Mutex::new(worker));
    let executors = ExecutorSet::new(
        Box::new(SharedExecutor(Arc::clone(&shared))),
        Box::new(ClosedExecutor),
        Box::new(SharedExecutor(shared)),
    );
    let (handle, runtime) = SchedulerRuntime::spawn(
        Budget::default(),
        RuntimeConfig { command_capacity: 64 },
        executors,
    )
    .map_err(|error| format!("scheduler startup failed: {error:?}"))?;
    let result = run_session(&handle, &plan, &completed, prompt, max_new);
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
    use logan_core::sched::ids::Ticket;
    use logan_core::sched::RuntimeConfig;

    // --- fixtures -----------------------------------------------------------

    fn tiny_plan() -> Plan {
        // 2 layers x 2 experts; slot bytes 300 each.
        let size = [(0, 100), (0, 100), (0, 100)];
        Plan {
            layers: vec![
                vec![
                    crate::plan::PlannedExpert { shard_id: 0, regions: size, dims: [(1, 1); 3] },
                    crate::plan::PlannedExpert { shard_id: 1, regions: size, dims: [(1, 1); 3] },
                ],
                vec![
                    crate::plan::PlannedExpert { shard_id: 2, regions: size, dims: [(1, 1); 3] },
                    crate::plan::PlannedExpert { shard_id: 3, regions: size, dims: [(1, 1); 3] },
                ],
            ],
            max_slot_bytes: 300,
        }
    }

    /// Scripted executor (seam test only): forwards every dispatch to the
    /// test's responder thread; never touches MetalIO. SyncSender is Clone,
    /// so one scripted executor can serve both the IO and accelerator lanes
    /// (a single shared worker, exactly like production).
    #[derive(Clone)]
    struct ScriptedExecutor(SyncSender<DispatchRequest>);

    impl BoundedExecutor for ScriptedExecutor {
        fn try_submit(&mut self, request: DispatchRequest) -> SubmitResult {
            match self.0.try_send(request) {
                Ok(()) => SubmitResult::Accepted,
                Err(std::sync::mpsc::TrySendError::Full(request)) => SubmitResult::Full(request),
                Err(std::sync::mpsc::TrySendError::Disconnected(request)) => {
                    SubmitResult::Closed(request)
                }
            }
        }

        fn try_cancel(&mut self, _ticket: Ticket) -> bool {
            false
        }

        fn shutdown(&mut self) {}
    }

    fn scripted_pair() -> (ScriptedExecutor, std::sync::mpsc::Receiver<DispatchRequest>) {
        let (sender, receiver) = std::sync::mpsc::sync_channel(16);
        (ScriptedExecutor(sender), receiver)
    }

    // --- unit tests ---------------------------------------------------------

    /// Sessions are scheduler-owned (opaque ids); even bridge unit tests get
    /// them from a real runtime, exactly like production.
    fn test_session() -> (RuntimeHandle, SchedulerRuntime, SessionId) {
        let executors = ExecutorSet::new(
            Box::new(ClosedExecutor),
            Box::new(ClosedExecutor),
            Box::new(ClosedExecutor),
        );
        let (handle, runtime) = SchedulerRuntime::spawn(
            Budget::default(),
            RuntimeConfig { command_capacity: 8 },
            executors,
        )
        .unwrap();
        let session = match request(&handle, RuntimeRequest::CreateSession).unwrap() {
            RuntimeReply::SessionCreated(session) => session,
            other => panic!("unexpected session reply: {other:?}"),
        };
        (handle, runtime, session)
    }

    #[test]
    fn action_encoding_round_trips() {
        assert_eq!(decode_op(&encode_op(OP_DECODE, 123, 456)).unwrap(), (OP_DECODE, 123, 456));
        assert_eq!(decode_op(&encode_op(OP_LOAD, 7, 9)).unwrap(), (OP_LOAD, 7, 9));
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

    #[test]
    fn bridge_reserve_publish_round_trip() {
        let (handle, runtime, session) = test_session();
        let plan = tiny_plan();
        let mut bridge = ResidencyBridge::new(&plan).unwrap();
        assert_eq!(bridge.state(0, 1), ResidencyState::Absent);
        // Reserve the exact planned representation.
        let load_id = bridge.request_load(session, 0, 1).unwrap().expect("started load");
        assert_eq!(bridge.state(0, 1), ResidencyState::Loading { load_id });
        // Publish on successful completion.
        bridge.publish(load_id).unwrap();
        assert!(matches!(bridge.state(0, 1), ResidencyState::Resident { .. }));
        let stats = bridge.residency.pool_stats(MemoryPoolId(0)).unwrap();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.resident_bytes, 300);
        let _ = request(&handle, RuntimeRequest::Shutdown { mode: logan_core::sched::ShutdownMode::Cancel });
        runtime.join().unwrap();
    }

    #[test]
    fn bridge_evicted_key_returns_physical_only_load() {
        let (handle, runtime, session) = test_session();
        let plan = tiny_plan();
        let mut bridge = ResidencyBridge::new(&plan).unwrap();
        // First load publishes Resident.
        let load_id = bridge.request_load(session, 0, 1).unwrap().unwrap();
        bridge.publish(load_id).unwrap();
        // The physical LRU evicts the key; the view still says Resident
        // (V1 view has no eviction path). A re-reported cold expert must
        // NOT deadlock the driver: physical-only load, no lifecycle.
        assert!(bridge.request_load(session, 0, 1).unwrap().is_none());
        assert!(matches!(bridge.state(0, 1), ResidencyState::Resident { .. }));
        let _ = request(&handle, RuntimeRequest::Shutdown { mode: logan_core::sched::ShutdownMode::Cancel });
        runtime.join().unwrap();
    }

    #[test]
    fn check_target_validates_planned_dispatch() {
        let plan = tiny_plan();
        let bridge = ResidencyBridge::new(&plan).unwrap();
        assert!(bridge.check_target(ActionKind::Accelerator).is_ok());
        assert!(bridge.check_target(ActionKind::Io).is_ok());
        assert!(bridge.check_target(ActionKind::Cpu).is_err());
    }

    // --- sim: tickets, bridge, resume through the REAL runtime --------------

    #[test]
    fn cold_expert_round_trips_ticket_publish_resume() {
        let plan = Arc::new(tiny_plan());
        let completed = Arc::new(CompletionLog {
            state: Mutex::new(Completed::default()),
            ready: Condvar::new(),
        });
        // One bounded worker queue shared by the IO + accelerator lanes,
        // like production; the test's responder thread plays the worker.
        let (io, io_rx) = scripted_pair();
        let accel = io.clone();
        let executors = ExecutorSet::new(Box::new(io), Box::new(ClosedExecutor), Box::new(accel));
        let (handle, runtime) =
            SchedulerRuntime::spawn(Budget::default(), RuntimeConfig { command_capacity: 16 }, executors).unwrap();

        // Responder thread: scripted worker. First decode block reports cold
        // expert (0,1); the load succeeds; the resubmitted decode emits 42.
        let completed_worker = Arc::clone(&completed);
        let responder = thread::spawn(move || {
            let mut decode_calls = 0u32;
            loop {
                let dispatch = match io_rx.recv() {
                    Ok(d) => d,
                    Err(_) => return, // sender dropped: shutdown
                };
                let outcome = match decode_op(&dispatch.action.payload).unwrap() {
                    (OP_PREFILL, _, _) => SchedOutcome::Token(None),
                    (OP_DECODE, _, _) => {
                        decode_calls += 1;
                        if decode_calls == 1 {
                            SchedOutcome::NeedExperts { layer: 0, experts: vec![1] }
                        } else {
                            SchedOutcome::Token(Some(42))
                        }
                    }
                    (OP_LOAD, 0, 1) => SchedOutcome::Token(None),
                    (kind, a, b) => panic!("unexpected scripted action ({kind},{a},{b})"),
                };
                completed_worker.push(dispatch.ticket, Ok(outcome));
                let _ = dispatch.completion.try_complete(dispatch.ticket, Outcome::Ok);
            }
        });

        let session = match request(&handle, RuntimeRequest::CreateSession).unwrap() {
            RuntimeReply::SessionCreated(session) => session,
            other => panic!("unexpected session reply: {other:?}"),
        };
        let mut bridge = ResidencyBridge::new(&plan).unwrap();
        let output = run_session_inner(&handle, session, &mut bridge, &completed, &[5], 1).unwrap();
        assert_eq!(output, vec![42]);

        // The cold expert went Absent -> Loading -> Resident through the
        // ticket lifecycle, and the view saw exactly one load.
        assert!(matches!(bridge.state(0, 1), ResidencyState::Resident { .. }));
        let stats = bridge.residency.pool_stats(MemoryPoolId(0)).unwrap();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.loads_started, 1);
        assert_eq!(stats.resident_bytes, 300);
        // The token the resumed forward emitted came from the resubmitted
        // decode action, not the blocked one (covered by the script above).

        let _ = request(&handle, RuntimeRequest::Shutdown { mode: logan_core::sched::ShutdownMode::Drain });
        runtime.join().unwrap();
        responder.join().unwrap();
    }

    #[test]
    fn scheduled_wrapper_is_token_identical_on_tiny_fixture() {
        // The tiny fixture is safetensors-only (experts preloaded, no
        // MetalIO), so the scheduled wrapper must reproduce the canonical
        // token stream exactly — the extraction/block-seam byte-identity
        // guarantee, enforced in CI. Cold loads are covered by the sim test
        // above and the real-model gate.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/qwen4_moe_tiny");
        let cfg = crate::load_cfg(&dir.join("config.json")).unwrap();
        let st = crate::StFile::open(&dir.join("model.safetensors")).unwrap();
        let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
        let max_new = 8usize;
        let canonical = crate::run_greedy_with(crate::Model::load(&st, &cfg).unwrap(), cfg.clone(), &prompt, max_new);
        let mut sched = crate::Model::load(&st, &cfg).unwrap();
        sched.enable_sched_mode();
        let mut out = Vec::new();
        let mut last = *prompt.last().unwrap();
        for pos in prompt.len()..prompt.len() + max_new {
            match sched.forward_scheduled(last as usize, pos) {
                SchedForward::Logits(logits) => {
                    last = greedy_next(&logits).unwrap();
                    out.push(last);
                }
                SchedForward::NeedExperts { .. } => {
                    panic!("tiny fixture (preloaded experts) must never report cold experts")
                }
            }
        }
        assert_eq!(out, canonical, "scheduled wrapper diverged from canonical");
    }
}