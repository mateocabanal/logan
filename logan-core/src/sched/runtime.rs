//! Minimal threaded runtime shell for issue #45.
//!
//! `SchedulerRuntime` owns one `SchedulerCore` on one thread. Frontends and
//! executor workers communicate through one bounded MPSC command/completion
//! channel. The runtime blocks in `recv` when idle and drains immediately
//! available commands without polling or sleeping.
//!
//! Executors must make `try_submit` and `try_cancel` nonblocking. A full
//! executor queue returns ownership of the `DispatchRequest` to the runtime;
//! the runtime retains it as backpressure and retries only after an explicit
//! `ExecutorReady` command.

use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::thread::{self, JoinHandle};

use super::core::{Action, ActionKind, Budget, Effect, Outcome, SchedError, SchedulerCore};
use super::ids::{SessionId, Ticket};

/// Runtime configuration. Both the inbound command path and executor-facing
/// completion path use this bounded capacity.
#[derive(Debug, Clone, Copy)]
pub struct RuntimeConfig {
    pub command_capacity: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            command_capacity: 256,
        }
    }
}

/// Whether shutdown cancels immediately or waits for accepted work to report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownMode {
    /// Revoke logical work, issue best-effort executor cancels, then exit.
    Cancel,
    /// Stop accepting new work and consume completions until no live ticket
    /// remains. Blocked sessions are then finished deterministically.
    Drain,
}

/// A response returned through a per-request, capacity-one reply channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeReply {
    SessionCreated(SessionId),
    Submitted {
        ticket: Ticket,
        disposition: SubmitDisposition,
    },
    Effects(Vec<Effect>),
    Ok,
    Error(String),
    ShutdownComplete,
}

/// Result of trying to place an action on its backend queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitDisposition {
    Accepted,
    Backpressured,
}

/// An action retained by the runtime when the selected executor is full.
#[derive(Debug)]
pub struct DispatchRequest {
    pub ticket: Ticket,
    pub action: Action,
    pub completion: CompletionSink,
}

/// Nonblocking completion path handed to executor implementations.
#[derive(Clone, Debug)]
pub struct CompletionSink {
    sender: SyncSender<RuntimeCommand>,
}

impl CompletionSink {
    /// Report a completion without ever waiting for scheduler capacity.
    /// A full result is returned to the executor, which retains responsibility
    /// for retrying that completion according to its own bounded policy.
    pub fn try_complete(
        &self,
        ticket: Ticket,
        outcome: Outcome,
    ) -> Result<(), CompletionSendError> {
        self.sender
            .try_send(RuntimeCommand::Complete {
                ticket,
                outcome,
                reply: None,
            })
            .map_err(|error| match error {
                TrySendError::Full(_) => CompletionSendError::Full,
                TrySendError::Disconnected(_) => CompletionSendError::Closed,
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionSendError {
    Full,
    Closed,
}

/// Nonblocking executor contract. Implementations own their worker threads,
/// native handles, and backend resources; the scheduler only submits opaque
/// requests and receives typed completions.
pub trait BoundedExecutor: Send + 'static {
    fn try_submit(&mut self, request: DispatchRequest) -> SubmitResult;
    fn try_cancel(&mut self, ticket: Ticket) -> bool;
    fn shutdown(&mut self);
}

#[derive(Debug)]
pub enum SubmitResult {
    Accepted,
    Full(DispatchRequest),
    Closed(DispatchRequest),
}

/// The three V1 executor lanes. Each lane has independent bounded admission,
/// so saturation in one lane cannot make the scheduler thread wait on it.
pub struct ExecutorSet {
    io: Box<dyn BoundedExecutor>,
    cpu: Box<dyn BoundedExecutor>,
    accelerator: Box<dyn BoundedExecutor>,
}

impl ExecutorSet {
    pub fn new(
        io: Box<dyn BoundedExecutor>,
        cpu: Box<dyn BoundedExecutor>,
        accelerator: Box<dyn BoundedExecutor>,
    ) -> Self {
        Self {
            io,
            cpu,
            accelerator,
        }
    }

    fn lane_mut(&mut self, kind: ActionKind) -> &mut dyn BoundedExecutor {
        match kind {
            ActionKind::Io => &mut *self.io,
            ActionKind::Cpu => &mut *self.cpu,
            ActionKind::Accelerator => &mut *self.accelerator,
        }
    }

    fn try_submit(&mut self, request: DispatchRequest) -> SubmitResult {
        let kind = request.action.kind;
        self.lane_mut(kind).try_submit(request)
    }

    fn try_cancel(&mut self, ticket: Ticket) {
        // A ticket is owned by exactly one lane, but V0 deliberately keeps
        // lane lookup opaque. Three nonblocking probes are safe and avoid
        // leaking backend identity into SchedulerCore.
        if self.io.try_cancel(ticket) {
            return;
        }
        if self.cpu.try_cancel(ticket) {
            return;
        }
        let _ = self.accelerator.try_cancel(ticket);
    }

    fn shutdown(&mut self) {
        self.io.shutdown();
        self.cpu.shutdown();
        self.accelerator.shutdown();
    }
}

/// Frontend handle. `try_request` never blocks while enqueueing; callers may
/// wait on the returned response receiver after the command is accepted.
#[derive(Clone, Debug)]
pub struct RuntimeHandle {
    sender: SyncSender<RuntimeCommand>,
}

impl RuntimeHandle {
    pub fn try_request(
        &self,
        request: RuntimeRequest,
    ) -> Result<Receiver<RuntimeReply>, RequestSendError> {
        let (reply, receiver) = mpsc::sync_channel(1);
        self.sender
            .try_send(request.into_command(ReplyPort(Some(reply))))
            .map_err(|error| match error {
                TrySendError::Full(_) => RequestSendError::Full,
                TrySendError::Disconnected(_) => RequestSendError::Closed,
            })?;
        Ok(receiver)
    }

    /// Fire-and-forget notification for executor capacity or completion.
    pub fn try_notify(&self, request: RuntimeRequest) -> Result<(), RequestSendError> {
        self.sender
            .try_send(request.into_command(ReplyPort::none()))
            .map_err(|error| match error {
                TrySendError::Full(_) => RequestSendError::Full,
                TrySendError::Disconnected(_) => RequestSendError::Closed,
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestSendError {
    Full,
    Closed,
}

/// Frontend-visible commands. RuntimeRequest deliberately has no reply
/// channel; `try_request` supplies one without making the scheduler own a
/// blocking response operation.
#[derive(Debug)]
pub enum RuntimeRequest {
    CreateSession,
    Submit { session: SessionId, action: Action },
    Block { session: SessionId },
    Wake { session: SessionId },
    Complete { ticket: Ticket, outcome: Outcome },
    Cancel { session: SessionId },
    Finish { session: SessionId },
    Fail { session: SessionId, error: String },
    ExecutorReady { kind: ActionKind },
    Shutdown { mode: ShutdownMode },
}

struct ReplyPort(Option<SyncSender<RuntimeReply>>);

impl ReplyPort {
    fn none() -> Self {
        Self(None)
    }

    fn send(&self, reply: RuntimeReply) {
        if let Some(sender) = &self.0 {
            // A dropped/slow frontend must never stall the scheduler owner.
            let _ = sender.try_send(reply);
        }
    }
}

enum RuntimeCommand {
    CreateSession {
        reply: ReplyPort,
    },
    Submit {
        session: SessionId,
        action: Action,
        reply: ReplyPort,
    },
    Block {
        session: SessionId,
        reply: ReplyPort,
    },
    Wake {
        session: SessionId,
        reply: ReplyPort,
    },
    Complete {
        ticket: Ticket,
        outcome: Outcome,
        reply: Option<ReplyPort>,
    },
    Cancel {
        session: SessionId,
        reply: ReplyPort,
    },
    Finish {
        session: SessionId,
        reply: ReplyPort,
    },
    Fail {
        session: SessionId,
        error: String,
        reply: ReplyPort,
    },
    ExecutorReady {
        kind: ActionKind,
    },
    Shutdown {
        mode: ShutdownMode,
        reply: ReplyPort,
    },
}

impl RuntimeRequest {
    fn into_command(self, reply: ReplyPort) -> RuntimeCommand {
        match self {
            Self::CreateSession => RuntimeCommand::CreateSession { reply },
            Self::Submit { session, action } => RuntimeCommand::Submit {
                session,
                action,
                reply,
            },
            Self::Block { session } => RuntimeCommand::Block { session, reply },
            Self::Wake { session } => RuntimeCommand::Wake { session, reply },
            Self::Complete { ticket, outcome } => RuntimeCommand::Complete {
                ticket,
                outcome,
                reply: reply.0.map(|sender| ReplyPort(Some(sender))),
            },
            Self::Cancel { session } => RuntimeCommand::Cancel { session, reply },
            Self::Finish { session } => RuntimeCommand::Finish { session, reply },
            Self::Fail { session, error } => RuntimeCommand::Fail {
                session,
                error,
                reply,
            },
            Self::ExecutorReady { kind } => RuntimeCommand::ExecutorReady { kind },
            Self::Shutdown { mode } => RuntimeCommand::Shutdown { mode, reply },
        }
    }
}

/// Join handle for the single scheduler owner thread.
pub struct SchedulerRuntime {
    join: Option<JoinHandle<()>>,
    shutdown_sender: SyncSender<RuntimeCommand>,
}

impl SchedulerRuntime {
    pub fn spawn(
        budget: Budget,
        config: RuntimeConfig,
        executors: ExecutorSet,
    ) -> Result<(RuntimeHandle, Self), RuntimeConfigError> {
        if config.command_capacity == 0 {
            return Err(RuntimeConfigError::ZeroCapacity);
        }
        let (sender, receiver) = mpsc::sync_channel(config.command_capacity);
        let handle = RuntimeHandle {
            sender: sender.clone(),
        };
        let completion = CompletionSink {
            sender: sender.clone(),
        };
        let join = thread::Builder::new()
            .name("logan-scheduler".into())
            .spawn(move || RuntimeLoop::new(receiver, completion, budget, executors).run())
            .expect("scheduler thread must be creatable");
        Ok((
            handle,
            Self {
                join: Some(join),
                shutdown_sender: sender,
            },
        ))
    }

    pub fn join(mut self) -> thread::Result<()> {
        self.join.take().expect("runtime join called once").join()
    }
}

impl Drop for SchedulerRuntime {
    fn drop(&mut self) {
        if self.join.is_some() {
            let _ = self.shutdown_sender.try_send(RuntimeCommand::Shutdown {
                mode: ShutdownMode::Cancel,
                reply: ReplyPort::none(),
            });
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeConfigError {
    ZeroCapacity,
}

struct RuntimeLoop {
    receiver: Receiver<RuntimeCommand>,
    completion: CompletionSink,
    core: SchedulerCore,
    executors: ExecutorSet,
    pending: VecDeque<DispatchRequest>,
    draining: bool,
    shutdown_reply: Option<ReplyPort>,
}

impl RuntimeLoop {
    fn new(
        receiver: Receiver<RuntimeCommand>,
        completion: CompletionSink,
        budget: Budget,
        executors: ExecutorSet,
    ) -> Self {
        Self {
            receiver,
            completion,
            core: SchedulerCore::new(budget),
            executors,
            pending: VecDeque::new(),
            draining: false,
            shutdown_reply: None,
        }
    }

    fn run(mut self) {
        while let Ok(command) = self.receiver.recv() {
            if self.handle(command) {
                return;
            }
            // Process the burst already in the bounded queue before blocking
            // again. There is no periodic polling or sleep path.
            loop {
                match self.receiver.try_recv() {
                    Ok(command) => {
                        if self.handle(command) {
                            return;
                        }
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        self.shutdown(ShutdownMode::Cancel, ReplyPort::none());
                        return;
                    }
                }
            }
            if self.draining && self.is_quiescent() {
                let reply = self.shutdown_reply.take().unwrap_or_else(ReplyPort::none);
                self.shutdown(ShutdownMode::Drain, reply);
                return;
            }
        }
        self.shutdown(ShutdownMode::Cancel, ReplyPort::none());
    }

    /// Return true to terminate the owner thread.
    fn handle(&mut self, command: RuntimeCommand) -> bool {
        match command {
            RuntimeCommand::CreateSession { reply } => {
                if self.draining {
                    reply.send(RuntimeReply::Error("runtime is draining".into()));
                } else {
                    reply.send(RuntimeReply::SessionCreated(self.core.create_session()));
                }
            }
            RuntimeCommand::Submit {
                session,
                action,
                reply,
            } => {
                if self.draining {
                    reply.send(RuntimeReply::Error("runtime is draining".into()));
                } else {
                    self.submit(session, action, reply);
                }
            }
            RuntimeCommand::Block { session, reply } => {
                let result = self.core.block(session);
                self.respond_effects(result, reply)
            }
            RuntimeCommand::Wake { session, reply } => {
                let result = self.core.wake(session);
                self.respond_effects(result, reply)
            }
            RuntimeCommand::Complete {
                ticket,
                outcome,
                reply,
            } => {
                let effects = self.core.complete(ticket, outcome);
                if let Ok(effects) = &effects {
                    self.apply_effects(effects);
                }
                if let Some(reply) = reply {
                    self.respond_effects(effects, reply);
                }
            }
            RuntimeCommand::Cancel { session, reply } => {
                let effects = self.core.cancel(session);
                if let Ok(effects) = &effects {
                    self.remove_cancelled(effects);
                    self.apply_effects(effects);
                }
                self.respond_effects(effects, reply);
            }
            RuntimeCommand::Finish { session, reply } => {
                let effects = self.core.finish(session);
                if let Ok(effects) = &effects {
                    self.remove_cancelled(effects);
                    self.apply_effects(effects);
                }
                self.respond_effects(effects, reply);
            }
            RuntimeCommand::Fail {
                session,
                error,
                reply,
            } => {
                let effects = self.core.fail(session, error);
                if let Ok(effects) = &effects {
                    self.remove_cancelled(effects);
                    self.apply_effects(effects);
                }
                self.respond_effects(effects, reply);
            }
            RuntimeCommand::ExecutorReady { kind } => self.retry_pending(kind),
            RuntimeCommand::Shutdown { mode, reply } => {
                if mode == ShutdownMode::Drain && !self.is_quiescent() {
                    if self.draining {
                        reply.send(RuntimeReply::Error("runtime is already draining".into()));
                    } else {
                        self.draining = true;
                        self.shutdown_reply = Some(reply);
                    }
                } else {
                    self.shutdown(mode, reply);
                    return true;
                }
            }
        }
        false
    }

    fn submit(&mut self, session: SessionId, action: Action, reply: ReplyPort) {
        match self.core.submit(session, action) {
            Ok((ticket, Effect::Dispatch { ticket: _, action })) => {
                let request = DispatchRequest {
                    ticket,
                    action,
                    completion: self.completion.clone(),
                };
                match self.executors.try_submit(request) {
                    SubmitResult::Accepted => reply.send(RuntimeReply::Submitted {
                        ticket,
                        disposition: SubmitDisposition::Accepted,
                    }),
                    SubmitResult::Full(request) => {
                        self.pending.push_back(request);
                        reply.send(RuntimeReply::Submitted {
                            ticket,
                            disposition: SubmitDisposition::Backpressured,
                        });
                    }
                    SubmitResult::Closed(request) => {
                        let effects = self.core.complete(
                            request.ticket,
                            Outcome::Err("executor closed before dispatch".into()),
                        );
                        if let Ok(effects) = &effects {
                            self.remove_cancelled(effects);
                            self.apply_effects(effects);
                        }
                        reply.send(RuntimeReply::Error("executor closed".into()));
                    }
                }
            }
            Ok((_ticket, effect)) => reply.send(RuntimeReply::Error(format!(
                "unexpected scheduler effect: {effect:?}"
            ))),
            Err(error) => reply.send(RuntimeReply::Error(error.to_string())),
        }
    }

    fn retry_pending(&mut self, kind: ActionKind) {
        let mut index = 0;
        while index < self.pending.len() {
            if self.pending[index].action.kind != kind {
                index += 1;
                continue;
            }
            let request = self.pending.remove(index).expect("index checked");
            match self.executors.try_submit(request) {
                SubmitResult::Accepted => {}
                SubmitResult::Full(request) => {
                    self.pending.insert(index, request);
                    break;
                }
                SubmitResult::Closed(request) => {
                    let effects = self.core.complete(
                        request.ticket,
                        Outcome::Err("executor closed before retry".into()),
                    );
                    if let Ok(effects) = &effects {
                        self.remove_cancelled(effects);
                        self.apply_effects(effects);
                    }
                }
            }
        }
    }

    fn respond_effects(&self, result: Result<Vec<Effect>, SchedError>, reply: ReplyPort) {
        match result {
            Ok(effects) => reply.send(if effects.is_empty() {
                RuntimeReply::Ok
            } else {
                RuntimeReply::Effects(effects)
            }),
            Err(error) => reply.send(RuntimeReply::Error(error.to_string())),
        }
    }

    fn apply_effects(&mut self, effects: &[Effect]) {
        for effect in effects {
            if let Effect::Cancel { ticket } = effect {
                self.executors.try_cancel(*ticket);
            }
        }
    }

    fn remove_cancelled(&mut self, effects: &[Effect]) {
        for effect in effects {
            if let Effect::Cancel { ticket } = effect {
                self.pending.retain(|request| request.ticket != *ticket);
            }
        }
    }

    fn is_quiescent(&self) -> bool {
        self.pending.is_empty() && self.core.total_inflight() == 0
    }

    fn shutdown(&mut self, mode: ShutdownMode, reply: ReplyPort) {
        for session in self.core.session_ids() {
            if mode == ShutdownMode::Cancel {
                if let Ok(effects) = self.core.cancel(session) {
                    self.remove_cancelled(&effects);
                    self.apply_effects(&effects);
                }
            }
            if let Ok(effects) = self.core.finish(session) {
                self.apply_effects(&effects);
            }
        }
        self.pending.clear();
        self.executors.shutdown();
        reply.send(RuntimeReply::ShutdownComplete);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct FakeExecutor {
        state: Arc<Mutex<FakeState>>,
    }

    #[derive(Default)]
    struct FakeState {
        full: bool,
        accepted: Vec<Ticket>,
        cancelled: Vec<Ticket>,
        shutdown: bool,
    }

    impl BoundedExecutor for FakeExecutor {
        fn try_submit(&mut self, request: DispatchRequest) -> SubmitResult {
            let mut state = self.state.lock().unwrap();
            if state.full {
                SubmitResult::Full(request)
            } else {
                state.accepted.push(request.ticket);
                SubmitResult::Accepted
            }
        }

        fn try_cancel(&mut self, ticket: Ticket) -> bool {
            self.state.lock().unwrap().cancelled.push(ticket);
            true
        }

        fn shutdown(&mut self) {
            self.state.lock().unwrap().shutdown = true;
        }
    }

    fn executors(state: &Arc<Mutex<FakeState>>) -> ExecutorSet {
        ExecutorSet::new(
            Box::new(FakeExecutor {
                state: Arc::clone(state),
            }),
            Box::new(FakeExecutor {
                state: Arc::clone(state),
            }),
            Box::new(FakeExecutor {
                state: Arc::clone(state),
            }),
        )
    }

    fn recv(receiver: Receiver<RuntimeReply>) -> RuntimeReply {
        receiver.recv().unwrap()
    }

    #[test]
    fn idle_runtime_blocks_and_request_round_trips() {
        let state = Arc::new(Mutex::new(FakeState::default()));
        let (handle, runtime) = SchedulerRuntime::spawn(
            Budget::default(),
            RuntimeConfig {
                command_capacity: 4,
            },
            executors(&state),
        )
        .unwrap();
        let session = match recv(handle.try_request(RuntimeRequest::CreateSession).unwrap()) {
            RuntimeReply::SessionCreated(session) => session,
            other => panic!("unexpected reply: {other:?}"),
        };
        assert_eq!(session.0, 0);
        let shutdown = handle
            .try_request(RuntimeRequest::Shutdown {
                mode: ShutdownMode::Cancel,
            })
            .unwrap();
        assert_eq!(recv(shutdown), RuntimeReply::ShutdownComplete);
        runtime.join().unwrap();
        assert!(state.lock().unwrap().shutdown);
    }

    #[test]
    fn full_executor_is_backpressure_not_scheduler_blocking() {
        let state = Arc::new(Mutex::new(FakeState {
            full: true,
            ..FakeState::default()
        }));
        let (handle, runtime) = SchedulerRuntime::spawn(
            Budget::default(),
            RuntimeConfig::default(),
            executors(&state),
        )
        .unwrap();
        let session = match recv(handle.try_request(RuntimeRequest::CreateSession).unwrap()) {
            RuntimeReply::SessionCreated(session) => session,
            other => panic!("unexpected reply: {other:?}"),
        };
        let submitted = recv(
            handle
                .try_request(RuntimeRequest::Submit {
                    session,
                    action: Action {
                        kind: ActionKind::Io,
                        payload: vec![7],
                    },
                })
                .unwrap(),
        );
        let ticket = match submitted {
            RuntimeReply::Submitted {
                ticket,
                disposition: SubmitDisposition::Backpressured,
            } => ticket,
            other => panic!("unexpected reply: {other:?}"),
        };
        assert_eq!(ticket.0, 0);
        let shutdown = recv(
            handle
                .try_request(RuntimeRequest::Cancel { session })
                .unwrap(),
        );
        assert!(matches!(shutdown, RuntimeReply::Effects(_)));
        let done = recv(
            handle
                .try_request(RuntimeRequest::Shutdown {
                    mode: ShutdownMode::Cancel,
                })
                .unwrap(),
        );
        assert_eq!(done, RuntimeReply::ShutdownComplete);
        runtime.join().unwrap();
        assert_eq!(state.lock().unwrap().cancelled, vec![ticket]);
    }

    #[test]
    fn completion_wakes_blocked_session_and_shutdown_drains() {
        let state = Arc::new(Mutex::new(FakeState::default()));
        let (handle, runtime) = SchedulerRuntime::spawn(
            Budget::default(),
            RuntimeConfig::default(),
            executors(&state),
        )
        .unwrap();
        let session = match recv(handle.try_request(RuntimeRequest::CreateSession).unwrap()) {
            RuntimeReply::SessionCreated(session) => session,
            other => panic!("unexpected reply: {other:?}"),
        };
        let ticket = match recv(
            handle
                .try_request(RuntimeRequest::Submit {
                    session,
                    action: Action {
                        kind: ActionKind::Cpu,
                        payload: vec![],
                    },
                })
                .unwrap(),
        ) {
            RuntimeReply::Submitted { ticket, .. } => ticket,
            other => panic!("unexpected reply: {other:?}"),
        };
        assert_eq!(
            recv(
                handle
                    .try_request(RuntimeRequest::Block { session })
                    .unwrap()
            ),
            RuntimeReply::Ok
        );
        let shutdown = handle
            .try_request(RuntimeRequest::Shutdown {
                mode: ShutdownMode::Drain,
            })
            .unwrap();
        // Drain is waiting for the completion, not spinning or exiting early.
        assert!(shutdown.try_recv().is_err());
        let completion = handle
            .try_request(RuntimeRequest::Complete {
                ticket,
                outcome: Outcome::Ok,
            })
            .unwrap();
        assert!(
            matches!(recv(completion), RuntimeReply::Effects(effects) if effects.iter().any(|effect| matches!(effect, Effect::Wake { session: id } if *id == session)))
        );
        assert_eq!(recv(shutdown), RuntimeReply::ShutdownComplete);
        runtime.join().unwrap();
    }
}
