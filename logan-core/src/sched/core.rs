//! Pure, single-owner scheduler state machine for issue #43/#44.
//!
//! The core decides lifecycle and dependency transitions. It does not perform
//! file I/O, execute model kernels, wait on devices, create threads, or use a
//! Rust async executor. Drivers execute returned effects and call back with
//! typed ticket completions.
//!
//! ## Ownership invariants
//!
//! * A session is present in the ready queue at most once.
//! * `step` removes queue ownership; only `requeue` can return it.
//! * Finished and failed sessions are never runnable or queued.
//! * A ticket is consumed at most once. Late, duplicate, or cancelled
//!   completions are harmless no-ops.
//! * The scheduler never performs the work represented by an `Effect`.

use std::collections::{HashMap, HashSet, VecDeque};

use super::ids::{Generation, SessionId, Ticket};

/// Backend class selected by an engine-owned action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionKind {
    Io,
    Cpu,
    Accelerator,
}

/// Engine-owned opaque action payload.
///
/// `logan-core` carries this value without interpreting it. Model crates and
/// the runtime driver agree on the payload schema at their boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Action {
    pub kind: ActionKind,
    pub payload: Vec<u8>,
}

/// Result reported by a backend for a submitted ticket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Ok,
    Err(String),
}

/// Work or lifecycle intent returned to the driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    /// Execute an opaque action outside the scheduler owner.
    Dispatch { ticket: Ticket, action: Action },
    /// Best-effort cancellation of an already-submitted action.
    Cancel { ticket: Ticket },
    /// A blocked session became runnable after its dependencies completed.
    Wake { session: SessionId },
    /// A session reached successful terminal state.
    Finished { session: SessionId },
    /// A session reached failed terminal state.
    Failed { session: SessionId, error: String },
}

/// V0 scheduler policy.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// Maximum number of live action tickets owned by one session.
    pub max_inflight_per_session: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            max_inflight_per_session: 64,
        }
    }
}

/// Session lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionState {
    Runnable,
    Blocked,
    Cancelling,
    Finished,
    Failed,
}

/// Scheduler-owned session record.
#[derive(Debug, Clone)]
pub struct Session {
    pub id: SessionId,
    pub state: SessionState,
    pub inflight: Vec<Ticket>,
    in_ready: bool,
}

impl Session {
    fn new(id: SessionId) -> Self {
        Self {
            id,
            state: SessionState::Runnable,
            inflight: Vec::new(),
            in_ready: true,
        }
    }
}

/// One scheduler dispatch quantum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Run(SessionId),
    Idle,
}

/// Errors for invalid session lifecycle operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedError {
    UnknownSession(SessionId),
    IllegalTransition {
        session: SessionId,
        from: SessionState,
    },
    NotRunnable(SessionId),
    InflightLimit(SessionId),
}

impl std::fmt::Display for SchedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Deterministic scheduler core. All mutation is intended to happen on one
/// scheduler owner; no method performs external work.
#[derive(Debug)]
pub struct SchedulerCore {
    budget: Budget,
    sessions: HashMap<SessionId, Session>,
    ready: VecDeque<SessionId>,
    next_session: u64,
    next_ticket: u64,
    /// Ticket ownership and its reserved resource generation. Generation is
    /// carried now so residency can become generation-safe in #47 without
    /// changing the completion correlation shape.
    inflight: HashMap<Ticket, (SessionId, Generation)>,
}

impl SchedulerCore {
    pub fn new(budget: Budget) -> Self {
        Self {
            budget,
            sessions: HashMap::new(),
            ready: VecDeque::new(),
            next_session: 0,
            next_ticket: 0,
            inflight: HashMap::new(),
        }
    }

    fn alloc_session_id(&mut self) -> SessionId {
        let id = SessionId(self.next_session);
        self.next_session += 1;
        id
    }

    fn alloc_ticket(&mut self) -> Ticket {
        let ticket = Ticket(self.next_ticket);
        self.next_ticket += 1;
        ticket
    }

    /// Remove every queue occurrence for `session` and clear its membership
    /// bit. This is defensive as well as normal bookkeeping: a lifecycle
    /// transition remains invariant-safe even if called before `step`.
    fn remove_from_ready(&mut self, session: SessionId) {
        self.ready.retain(|queued| *queued != session);
        if let Some(record) = self.sessions.get_mut(&session) {
            record.in_ready = false;
        }
    }

    fn enqueue_if_runnable(&mut self, session: SessionId) {
        let Some(record) = self.sessions.get_mut(&session) else {
            return;
        };
        if record.state == SessionState::Runnable && !record.in_ready {
            record.in_ready = true;
            self.ready.push_back(session);
        }
    }

    /// Create a runnable session. Creation itself is the only implicit queue
    /// insertion; running sessions must be explicitly requeued by the driver.
    pub fn create_session(&mut self) -> SessionId {
        let id = self.alloc_session_id();
        self.sessions.insert(id, Session::new(id));
        self.ready.push_back(id);
        id
    }

    pub fn session_state(&self, session: SessionId) -> Option<SessionState> {
        self.sessions.get(&session).map(|record| record.state)
    }

    pub fn inflight_count(&self, session: SessionId) -> Option<usize> {
        self.sessions
            .get(&session)
            .map(|record| record.inflight.len())
    }

    pub fn ready_len(&self) -> usize {
        self.ready.len()
    }

    /// Return all known sessions in deterministic ID order. The runtime uses
    /// this query during shutdown; lifecycle mutation remains in this core.
    pub fn session_ids(&self) -> Vec<SessionId> {
        let mut ids: Vec<_> = self.sessions.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    pub fn total_inflight(&self) -> usize {
        self.inflight.len()
    }

    /// Move a runnable session to Blocked. The engine calls this after it has
    /// submitted the dependencies represented by its live tickets.
    pub fn block(&mut self, session: SessionId) -> Result<Vec<Effect>, SchedError> {
        let state = self
            .sessions
            .get(&session)
            .ok_or(SchedError::UnknownSession(session))?
            .state;
        match state {
            SessionState::Runnable => {
                self.remove_from_ready(session);
                self.sessions
                    .get_mut(&session)
                    .expect("session checked above")
                    .state = SessionState::Blocked;
                Ok(Vec::new())
            }
            SessionState::Blocked | SessionState::Cancelling => Ok(Vec::new()),
            SessionState::Finished | SessionState::Failed => Err(SchedError::IllegalTransition {
                from: state,
                session,
            }),
        }
    }

    /// Wake a blocked session. Ordinary wake cannot resurrect cancellation or
    /// terminal state; those transitions require explicit lifecycle calls.
    pub fn wake(&mut self, session: SessionId) -> Result<Vec<Effect>, SchedError> {
        let state = self
            .sessions
            .get(&session)
            .ok_or(SchedError::UnknownSession(session))?
            .state;
        match state {
            SessionState::Blocked => {
                self.sessions
                    .get_mut(&session)
                    .expect("session checked above")
                    .state = SessionState::Runnable;
                self.enqueue_if_runnable(session);
                Ok(vec![Effect::Wake { session }])
            }
            SessionState::Runnable | SessionState::Cancelling => Ok(Vec::new()),
            SessionState::Finished | SessionState::Failed => Err(SchedError::IllegalTransition {
                from: state,
                session,
            }),
        }
    }

    /// Submit one opaque action and return the driver's dispatch intent.
    pub fn submit(
        &mut self,
        session: SessionId,
        action: Action,
    ) -> Result<(Ticket, Effect), SchedError> {
        let record = self
            .sessions
            .get(&session)
            .ok_or(SchedError::UnknownSession(session))?;
        if record.state != SessionState::Runnable {
            return Err(SchedError::NotRunnable(session));
        }
        if record.inflight.len() >= self.budget.max_inflight_per_session {
            return Err(SchedError::InflightLimit(session));
        }

        let ticket = self.alloc_ticket();
        let generation = Generation(ticket.0);
        self.sessions
            .get_mut(&session)
            .expect("session checked above")
            .inflight
            .push(ticket);
        self.inflight.insert(ticket, (session, generation));
        Ok((ticket, Effect::Dispatch { ticket, action }))
    }

    /// Complete a ticket exactly once. A successful completion wakes a blocked
    /// session only when it was that session's final dependency. An error
    /// terminalizes the session and cancels any remaining owned tickets.
    pub fn complete(
        &mut self,
        ticket: Ticket,
        outcome: Outcome,
    ) -> Result<Vec<Effect>, SchedError> {
        let Some((session, _generation)) = self.inflight.remove(&ticket) else {
            // Duplicate, unknown, cancelled, or terminally-stale completion.
            return Ok(Vec::new());
        };
        let record = self
            .sessions
            .get_mut(&session)
            .ok_or(SchedError::UnknownSession(session))?;
        record.inflight.retain(|owned| *owned != ticket);

        match outcome {
            Outcome::Ok => {
                let should_wake =
                    record.state == SessionState::Blocked && record.inflight.is_empty();
                if should_wake {
                    record.state = SessionState::Runnable;
                    record.in_ready = true;
                    self.ready.push_back(session);
                    Ok(vec![Effect::Wake { session }])
                } else {
                    Ok(Vec::new())
                }
            }
            Outcome::Err(error) => self.terminalize(session, SessionState::Failed, Some(error)),
        }
    }

    /// Transition a session to cancellation and revoke all of its live ticket
    /// ownership. The driver may still receive late backend callbacks; those
    /// callbacks are stale no-ops.
    pub fn cancel(&mut self, session: SessionId) -> Result<Vec<Effect>, SchedError> {
        let state = self
            .sessions
            .get(&session)
            .ok_or(SchedError::UnknownSession(session))?
            .state;
        match state {
            SessionState::Finished | SessionState::Failed => Err(SchedError::IllegalTransition {
                from: state,
                session,
            }),
            SessionState::Cancelling => Ok(Vec::new()),
            SessionState::Runnable | SessionState::Blocked => {
                self.remove_from_ready(session);
                let tickets = std::mem::take(
                    &mut self
                        .sessions
                        .get_mut(&session)
                        .expect("session checked above")
                        .inflight,
                );
                self.sessions
                    .get_mut(&session)
                    .expect("session checked above")
                    .state = SessionState::Cancelling;
                Ok(tickets
                    .into_iter()
                    .map(|ticket| {
                        self.inflight.remove(&ticket);
                        Effect::Cancel { ticket }
                    })
                    .collect())
            }
        }
    }

    /// Mark successful terminal state. If cleanup is still outstanding, the
    /// returned effects cancel it before reporting `Finished`.
    pub fn finish(&mut self, session: SessionId) -> Result<Vec<Effect>, SchedError> {
        let state = self
            .sessions
            .get(&session)
            .ok_or(SchedError::UnknownSession(session))?
            .state;
        match state {
            SessionState::Finished => Ok(Vec::new()),
            SessionState::Failed => Err(SchedError::IllegalTransition {
                from: state,
                session,
            }),
            _ => self.terminalize(session, SessionState::Finished, None),
        }
    }

    /// Mark failed terminal state. Duplicate failure is idempotent.
    pub fn fail(&mut self, session: SessionId, error: String) -> Result<Vec<Effect>, SchedError> {
        let state = self
            .sessions
            .get(&session)
            .ok_or(SchedError::UnknownSession(session))?
            .state;
        match state {
            SessionState::Failed => Ok(Vec::new()),
            SessionState::Finished => Err(SchedError::IllegalTransition {
                from: state,
                session,
            }),
            _ => self.terminalize(session, SessionState::Failed, Some(error)),
        }
    }

    /// Shared terminal transition. Removing ticket ownership before emitting
    /// effects makes all later completions stale by construction.
    fn terminalize(
        &mut self,
        session: SessionId,
        state: SessionState,
        error: Option<String>,
    ) -> Result<Vec<Effect>, SchedError> {
        self.remove_from_ready(session);
        let tickets = std::mem::take(
            &mut self
                .sessions
                .get_mut(&session)
                .ok_or(SchedError::UnknownSession(session))?
                .inflight,
        );
        self.sessions
            .get_mut(&session)
            .expect("session checked above")
            .state = state;

        let mut effects = Vec::with_capacity(tickets.len() + 1);
        for ticket in tickets {
            self.inflight.remove(&ticket);
            effects.push(Effect::Cancel { ticket });
        }
        match (state, error) {
            (SessionState::Finished, None) => effects.push(Effect::Finished { session }),
            (SessionState::Failed, Some(error)) => effects.push(Effect::Failed { session, error }),
            _ => unreachable!("terminal state and effect must agree"),
        }
        Ok(effects)
    }

    /// Return one runnable session. This method never requeues automatically:
    /// the driver runs one bounded engine quantum and then calls `requeue`,
    /// `block`, `finish`, or `fail` according to the engine result.
    pub fn step(&mut self) -> Step {
        while let Some(session) = self.ready.pop_front() {
            let Some(record) = self.sessions.get_mut(&session) else {
                continue;
            };
            if record.state != SessionState::Runnable || !record.in_ready {
                continue;
            }
            record.in_ready = false;
            return Step::Run(session);
        }
        Step::Idle
    }

    /// Return a still-runnable session to the back of the FIFO queue. Duplicate
    /// calls are harmless and cannot create duplicate queue membership.
    pub fn requeue(&mut self, session: SessionId) {
        self.enqueue_if_runnable(session);
    }

    /// Debug-only invariant checker used by deterministic tests and future
    /// simulator/replay code (#46).
    #[cfg(debug_assertions)]
    pub fn assert_invariants(&self) {
        let mut seen = HashSet::new();
        for &session in &self.ready {
            assert!(
                seen.insert(session),
                "duplicate ready membership: {session}"
            );
            let record = self
                .sessions
                .get(&session)
                .expect("ready queue references a live session");
            assert!(record.in_ready, "queue/member bit mismatch for {session}");
            assert_eq!(record.state, SessionState::Runnable);
        }
        for (&session, record) in &self.sessions {
            assert_eq!(record.in_ready, seen.contains(&session));
            if matches!(record.state, SessionState::Finished | SessionState::Failed) {
                assert!(!record.in_ready);
            }
            for ticket in &record.inflight {
                assert_eq!(
                    self.inflight.get(ticket).map(|(owner, _)| *owner),
                    Some(session)
                );
            }
        }
        for (ticket, (session, _generation)) in &self.inflight {
            assert!(
                self.sessions
                    .get(session)
                    .is_some_and(|record| record.inflight.contains(ticket))
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(kind: ActionKind) -> Action {
        Action {
            kind,
            payload: vec![1, 2, 3],
        }
    }

    fn submit(core: &mut SchedulerCore, session: SessionId, kind: ActionKind) -> Ticket {
        core.submit(session, action(kind)).unwrap().0
    }

    #[test]
    fn block_and_wake() {
        let mut core = SchedulerCore::new(Budget::default());
        let session = core.create_session();
        assert_eq!(core.step(), Step::Run(session));
        core.block(session).unwrap();
        assert_eq!(core.session_state(session), Some(SessionState::Blocked));
        assert_eq!(core.step(), Step::Idle);
        assert_eq!(core.wake(session).unwrap(), vec![Effect::Wake { session }]);
        assert_eq!(core.step(), Step::Run(session));
        core.assert_invariants();
    }

    #[test]
    fn block_before_step_removes_queue_membership() {
        let mut core = SchedulerCore::new(Budget::default());
        let session = core.create_session();
        core.block(session).unwrap();
        assert_eq!(core.ready_len(), 0);
        assert_eq!(core.step(), Step::Idle);
        core.wake(session).unwrap();
        assert_eq!(core.step(), Step::Run(session));
        core.assert_invariants();
    }

    #[test]
    fn cancel_while_blocked() {
        let mut core = SchedulerCore::new(Budget::default());
        let session = core.create_session();
        core.block(session).unwrap();
        assert!(core.cancel(session).unwrap().is_empty());
        assert_eq!(core.session_state(session), Some(SessionState::Cancelling));
        assert_eq!(core.wake(session).unwrap(), Vec::<Effect>::new());
        assert_eq!(core.step(), Step::Idle);
        core.finish(session).unwrap();
        core.assert_invariants();
    }

    #[test]
    fn cancel_with_inflight_action_makes_completion_stale() {
        let mut core = SchedulerCore::new(Budget::default());
        let session = core.create_session();
        let ticket = submit(&mut core, session, ActionKind::Io);
        assert_eq!(
            core.cancel(session).unwrap(),
            vec![Effect::Cancel { ticket }]
        );
        assert!(core.complete(ticket, Outcome::Ok).unwrap().is_empty());
        assert_eq!(core.session_state(session), Some(SessionState::Cancelling));
        core.finish(session).unwrap();
        core.assert_invariants();
    }

    #[test]
    fn submit_requires_runnable() {
        let mut core = SchedulerCore::new(Budget::default());
        let session = core.create_session();
        core.block(session).unwrap();
        assert_eq!(
            core.submit(session, action(ActionKind::Io)),
            Err(SchedError::NotRunnable(session))
        );
    }

    #[test]
    fn finish_and_failure_are_terminal_and_idempotent() {
        let mut core = SchedulerCore::new(Budget::default());
        let finished = core.create_session();
        core.finish(finished).unwrap();
        assert_eq!(core.finish(finished).unwrap(), Vec::<Effect>::new());
        assert_eq!(
            core.wake(finished),
            Err(SchedError::IllegalTransition {
                session: finished,
                from: SessionState::Finished,
            })
        );

        let failed = core.create_session();
        core.fail(failed, "boom".into()).unwrap();
        assert_eq!(
            core.fail(failed, "ignored".into()).unwrap(),
            Vec::<Effect>::new()
        );
        assert_eq!(core.step(), Step::Idle);
        core.assert_invariants();
    }

    #[test]
    fn out_of_order_completions_wake_only_on_last_dependency() {
        let mut core = SchedulerCore::new(Budget::default());
        let session = core.create_session();
        let first = submit(&mut core, session, ActionKind::Io);
        let second = submit(&mut core, session, ActionKind::Cpu);
        core.block(session).unwrap();
        assert!(core.complete(second, Outcome::Ok).unwrap().is_empty());
        assert_eq!(core.session_state(session), Some(SessionState::Blocked));
        assert_eq!(
            core.complete(first, Outcome::Ok).unwrap(),
            vec![Effect::Wake { session }]
        );
        core.assert_invariants();
    }

    #[test]
    fn duplicate_and_stale_tickets_are_noops() {
        let mut core = SchedulerCore::new(Budget::default());
        let session = core.create_session();
        let ticket = submit(&mut core, session, ActionKind::Accelerator);
        core.complete(ticket, Outcome::Ok).unwrap();
        assert!(core.complete(ticket, Outcome::Ok).unwrap().is_empty());

        let other = core.create_session();
        let stale = submit(&mut core, other, ActionKind::Io);
        core.finish(other).unwrap();
        assert!(core.complete(stale, Outcome::Ok).unwrap().is_empty());
        assert_eq!(core.session_state(other), Some(SessionState::Finished));
        core.assert_invariants();
    }

    #[test]
    fn action_failure_terminalizes_and_cancels_other_tickets() {
        let mut core = SchedulerCore::new(Budget::default());
        let session = core.create_session();
        let first = submit(&mut core, session, ActionKind::Io);
        let second = submit(&mut core, session, ActionKind::Cpu);
        let effects = core
            .complete(first, Outcome::Err("io error".into()))
            .unwrap();
        assert!(effects.contains(&Effect::Cancel { ticket: second }));
        assert!(effects.contains(&Effect::Failed {
            session,
            error: "io error".into()
        }));
        assert!(core.complete(second, Outcome::Ok).unwrap().is_empty());
        core.assert_invariants();
    }

    #[test]
    fn ready_queue_dedupe() {
        let mut core = SchedulerCore::new(Budget::default());
        let session = core.create_session();
        core.requeue(session);
        core.requeue(session);
        assert_eq!(core.ready_len(), 1);
        assert_eq!(core.step(), Step::Run(session));
        assert_eq!(core.step(), Step::Idle);
        core.requeue(session);
        core.assert_invariants();
    }

    #[test]
    fn bounded_round_robin_is_driver_requeue_contract() {
        let mut core = SchedulerCore::new(Budget::default());
        let a = core.create_session();
        let b = core.create_session();
        for expected in [a, b, a, b, a, b] {
            assert_eq!(core.step(), Step::Run(expected));
            core.requeue(expected); // hot session: always asks for another quantum
            core.requeue(expected); // duplicate acknowledgement stays harmless
            core.assert_invariants();
        }
        assert_eq!(core.ready_len(), 2);
    }

    #[test]
    fn inflight_limit_backpressure() {
        let mut core = SchedulerCore::new(Budget {
            max_inflight_per_session: 2,
        });
        let session = core.create_session();
        submit(&mut core, session, ActionKind::Io);
        submit(&mut core, session, ActionKind::Io);
        assert_eq!(
            core.submit(session, action(ActionKind::Io)),
            Err(SchedError::InflightLimit(session))
        );
    }

    #[test]
    fn stale_ready_entries_are_skipped() {
        let mut core = SchedulerCore::new(Budget::default());
        let session = core.create_session();
        core.finish(session).unwrap();
        assert_eq!(core.step(), Step::Idle);
        core.assert_invariants();
    }
}
