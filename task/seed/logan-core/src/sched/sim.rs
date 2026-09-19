//! Deterministic fake runtime, trace, and replay support for issue #46.
//!
//! This module drives the real `SchedulerCore` with explicit scripted
//! operations. It has no threads, clocks, sleeps, I/O, devices, or executor;
//! the same operation stream produces the same decisions and effects.

use super::core::{Action, Budget, Effect, Outcome, SchedError, SchedulerCore, SessionState, Step};
use super::ids::{SessionId, Ticket};

/// One deterministic operation accepted by the simulator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimOp {
    CreateSession,
    Submit { session: SessionId, action: Action },
    Step,
    Requeue { session: SessionId },
    Block { session: SessionId },
    Wake { session: SessionId },
    Complete { ticket: Ticket, outcome: Outcome },
    Cancel { session: SessionId },
    Finish { session: SessionId },
    Fail { session: SessionId, error: String },
}

/// Deterministic result of one simulated operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimOutcome {
    SessionCreated(SessionId),
    Submitted { ticket: Ticket, effect: Effect },
    Step(Step),
    Effects(Vec<Effect>),
    Noop,
    Error(SchedError),
}

/// One trace record. `seq` is a logical sequence number, never wall-clock
/// time, so it is stable across machines and replay runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceEvent {
    pub seq: u64,
    pub op: SimOp,
    pub outcome: SimOutcome,
}

/// A pure simulator around the production scheduler core.
#[derive(Debug)]
pub struct SchedulerSimulator {
    core: SchedulerCore,
    trace: Vec<TraceEvent>,
    next_seq: u64,
}

impl SchedulerSimulator {
    pub fn new(budget: Budget) -> Self {
        Self {
            core: SchedulerCore::new(budget),
            trace: Vec::new(),
            next_seq: 0,
        }
    }

    /// Apply and record one operation, including benign stale/no-op outcomes.
    pub fn apply(&mut self, op: SimOp) -> SimOutcome {
        let outcome = self.execute(&op);
        self.trace.push(TraceEvent {
            seq: self.next_seq,
            op,
            outcome: outcome.clone(),
        });
        self.next_seq += 1;
        #[cfg(debug_assertions)]
        self.core.assert_invariants();
        outcome
    }

    fn execute(&mut self, op: &SimOp) -> SimOutcome {
        match op {
            SimOp::CreateSession => SimOutcome::SessionCreated(self.core.create_session()),
            SimOp::Submit { session, action } => match self.core.submit(*session, action.clone()) {
                Ok((ticket, effect)) => SimOutcome::Submitted { ticket, effect },
                Err(error) => SimOutcome::Error(error),
            },
            SimOp::Step => SimOutcome::Step(self.core.step()),
            SimOp::Requeue { session } => {
                self.core.requeue(*session);
                SimOutcome::Noop
            }
            SimOp::Block { session } => {
                let result = self.core.block(*session);
                self.effects(result)
            }
            SimOp::Wake { session } => {
                let result = self.core.wake(*session);
                self.effects(result)
            }
            SimOp::Complete { ticket, outcome } => {
                let result = self.core.complete(*ticket, outcome.clone());
                self.effects(result)
            }
            SimOp::Cancel { session } => {
                let result = self.core.cancel(*session);
                self.effects(result)
            }
            SimOp::Finish { session } => {
                let result = self.core.finish(*session);
                self.effects(result)
            }
            SimOp::Fail { session, error } => {
                let result = self.core.fail(*session, error.clone());
                self.effects(result)
            }
        }
    }

    fn effects(&self, result: Result<Vec<Effect>, SchedError>) -> SimOutcome {
        match result {
            Ok(effects) if effects.is_empty() => SimOutcome::Noop,
            Ok(effects) => SimOutcome::Effects(effects),
            Err(error) => SimOutcome::Error(error),
        }
    }

    pub fn trace(&self) -> &[TraceEvent] {
        &self.trace
    }

    pub fn session_state(&self, session: SessionId) -> Option<SessionState> {
        self.core.session_state(session)
    }

    pub fn ready_len(&self) -> usize {
        self.core.ready_len()
    }

    pub fn total_inflight(&self) -> usize {
        self.core.total_inflight()
    }

    /// Replay an existing operation trace and compare every logical decision,
    /// including allocated IDs, effects, errors, and event sequence numbers.
    pub fn replay(budget: Budget, expected: &[TraceEvent]) -> Result<(), ReplayError> {
        let mut simulator = Self::new(budget);
        for expected_event in expected {
            let actual_outcome = simulator.apply(expected_event.op.clone());
            let actual_event = simulator.trace.last().expect("apply records an event");
            if actual_event != expected_event {
                return Err(ReplayError {
                    seq: expected_event.seq,
                    expected: expected_event.clone(),
                    actual: actual_event.clone(),
                });
            }
            // Keep this binding explicit: it documents that replay compares
            // the returned decision as well as the stored event.
            debug_assert_eq!(actual_outcome, expected_event.outcome);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayError {
    pub seq: u64,
    pub expected: TraceEvent,
    pub actual: TraceEvent,
}

/// A compact scripted completion source useful for adversarial deterministic
/// tests. Operations are consumed in caller-specified order.
#[derive(Debug, Default)]
pub struct FakeCompletionScript {
    completions: Vec<(Ticket, Outcome)>,
}

impl FakeCompletionScript {
    pub fn new(completions: Vec<(Ticket, Outcome)>) -> Self {
        Self { completions }
    }

    pub fn next(&mut self) -> Option<(Ticket, Outcome)> {
        if self.completions.is_empty() {
            None
        } else {
            Some(self.completions.remove(0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched::core::ActionKind;

    fn action(kind: ActionKind, byte: u8) -> Action {
        Action {
            kind,
            payload: vec![byte],
        }
    }

    fn created(simulator: &mut SchedulerSimulator) -> SessionId {
        match simulator.apply(SimOp::CreateSession) {
            SimOutcome::SessionCreated(session) => session,
            other => panic!("unexpected result: {other:?}"),
        }
    }

    fn submitted(
        simulator: &mut SchedulerSimulator,
        session: SessionId,
        kind: ActionKind,
        byte: u8,
    ) -> Ticket {
        match simulator.apply(SimOp::Submit {
            session,
            action: action(kind, byte),
        }) {
            SimOutcome::Submitted { ticket, .. } => ticket,
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn same_script_replays_identically() {
        let mut simulator = SchedulerSimulator::new(Budget::default());
        let a = created(&mut simulator);
        let b = created(&mut simulator);
        let ta = submitted(&mut simulator, a, ActionKind::Io, 1);
        let tb = submitted(&mut simulator, b, ActionKind::Cpu, 2);
        let operations = [
            SimOp::Step,
            SimOp::Requeue { session: a },
            SimOp::Step,
            SimOp::Requeue { session: b },
            SimOp::Complete {
                ticket: tb,
                outcome: Outcome::Ok,
            },
            SimOp::Complete {
                ticket: ta,
                outcome: Outcome::Err("late failure".into()),
            },
            SimOp::Step,
            SimOp::Cancel { session: a },
            SimOp::Finish { session: a },
        ];
        for operation in operations {
            simulator.apply(operation);
        }
        SchedulerSimulator::replay(Budget::default(), simulator.trace()).unwrap();
    }

    #[test]
    fn adversarial_completion_and_cancellation_are_stale_safe() {
        let mut simulator = SchedulerSimulator::new(Budget::default());
        let session = created(&mut simulator);
        let first = submitted(&mut simulator, session, ActionKind::Io, 1);
        simulator.apply(SimOp::Block { session });
        simulator.apply(SimOp::Cancel { session });
        assert_eq!(
            simulator.apply(SimOp::Complete {
                ticket: first,
                outcome: Outcome::Ok,
            }),
            SimOutcome::Noop
        );
        assert_eq!(
            simulator.session_state(session),
            Some(SessionState::Cancelling)
        );
        simulator.apply(SimOp::Finish { session });
        assert_eq!(simulator.ready_len(), 0);
        assert_eq!(simulator.total_inflight(), 0);
    }

    #[test]
    fn out_of_order_shared_dependencies_wake_once() {
        let mut simulator = SchedulerSimulator::new(Budget::default());
        let session = created(&mut simulator);
        let first = submitted(&mut simulator, session, ActionKind::Cpu, 1);
        let second = submitted(&mut simulator, session, ActionKind::Accelerator, 2);
        simulator.apply(SimOp::Block { session });
        assert_eq!(
            simulator.apply(SimOp::Complete {
                ticket: second,
                outcome: Outcome::Ok,
            }),
            SimOutcome::Noop
        );
        let wake = simulator.apply(SimOp::Complete {
            ticket: first,
            outcome: Outcome::Ok,
        });
        assert!(
            matches!(wake, SimOutcome::Effects(effects) if effects == vec![Effect::Wake { session }])
        );
        assert_eq!(simulator.ready_len(), 1);
        assert_eq!(
            simulator.apply(SimOp::Complete {
                ticket: first,
                outcome: Outcome::Ok,
            }),
            SimOutcome::Noop
        );
        assert_eq!(simulator.ready_len(), 1);
    }

    #[test]
    fn hot_session_cannot_bypass_fifo_in_scripted_quanta() {
        let mut simulator = SchedulerSimulator::new(Budget::default());
        let a = created(&mut simulator);
        let b = created(&mut simulator);
        for expected in [a, b, a, b, a, b] {
            assert_eq!(
                simulator.apply(SimOp::Step),
                SimOutcome::Step(Step::Run(expected))
            );
            simulator.apply(SimOp::Requeue { session: expected });
            simulator.apply(SimOp::Requeue { session: expected });
        }
        assert_eq!(simulator.ready_len(), 2);
    }

    #[test]
    fn completion_script_preserves_explicit_order() {
        let mut script = FakeCompletionScript::new(vec![
            (Ticket(4), Outcome::Ok),
            (Ticket(2), Outcome::Err("boom".into())),
        ]);
        assert_eq!(script.next(), Some((Ticket(4), Outcome::Ok)));
        assert_eq!(
            script.next(),
            Some((Ticket(2), Outcome::Err("boom".into())))
        );
        assert_eq!(script.next(), None);
    }
}
