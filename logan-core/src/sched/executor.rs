//! Dedicated persistent CPU compute executor for issue #51.
//!
//! `CpuExecutor` owns a fixed-size pool of persistent worker threads and a
//! bounded job queue. Submission is nonblocking: a saturated queue returns
//! the job (`SubmitOutcome::Full`) so the scheduler thread never waits on
//! compute capacity. The executor implements the runtime's `BoundedExecutor`
//! contract and can therefore serve as the CPU lane of `ExecutorSet` without
//! any runtime-thread changes.
//!
//! Completion contract:
//! * jobs submitted with a `CompletionSink` (the runtime path) report exactly
//!   once through the sink, retrying with a bounded policy when the command
//!   channel is full;
//! * jobs that retain residency leases (`Job::retained`) additionally land a
//!   `Completed` record whose `released` leases the scheduler owner must
//!   return to the `ResidencyManager`. The lease is kept alive for the whole
//!   job (including handler failure) and is handed back exactly once — even
//!   when the job was cancelled before dispatch.
//!
//! There is no affinity/NUMA/P-vs-E policy here; `CpuConfig` keeps worker
//! count and queue size explicit so such policy can be layered on later
//! without changing the submission contract.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use super::core::{Action, Outcome};
use super::ids::Ticket;
use super::residency::Lease;
use super::runtime::{
    BoundedExecutor, CompletionSendError, CompletionSink, DispatchRequest, SubmitResult,
};

/// Worker-pool configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuConfig {
    /// Number of persistent worker threads.
    pub workers: usize,
    /// Maximum number of queued (not running) jobs.
    pub queue_capacity: usize,
}

impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            workers: thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            queue_capacity: 64,
        }
    }
}

/// Invalid `CpuConfig` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutorConfigError {
    ZeroWorkers,
    ZeroQueueCapacity,
}

/// One unit of engine-owned CPU work.
#[derive(Debug)]
pub struct Job {
    pub ticket: Ticket,
    pub action: Action,
    /// Residency leases retained for the duration of this job. The executor
    /// keeps them alive while the job runs and returns them in `Completed`.
    pub retained: Vec<Lease>,
    /// Runtime completion path. `None` for scheduler-owner-driven jobs,
    /// which are completed through the `Completed` queue instead.
    pub completion: Option<CompletionSink>,
}

impl Job {
    /// Rebuild a runtime `DispatchRequest` for backpressure return. Only jobs
    /// created by `BoundedExecutor::try_submit` route here; they always carry
    /// a sink.
    fn into_request(self) -> DispatchRequest {
        DispatchRequest {
            ticket: self.ticket,
            action: self.action,
            completion: self
                .completion
                .expect("runtime jobs always carry a completion sink"),
        }
    }
}

/// A job that finished (or was cancelled before dispatch) exactly once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completed {
    pub ticket: Ticket,
    pub outcome: Outcome,
    /// Leases retained by the job; the scheduler owner must release each one
    /// through the residency manager.
    pub released: Vec<Lease>,
}

/// Nonblocking result of a job submission.
#[derive(Debug)]
pub enum SubmitOutcome {
    Accepted,
    /// Queue at capacity; the caller owns the job and may retry later.
    Full(Job),
    /// Executor has shut down; the job was accepted by nothing.
    Closed(Job),
}

/// Fixed-size persistent CPU worker pool with a bounded job queue.
pub struct CpuExecutor {
    queue: Arc<Mutex<VecDeque<Job>>>,
    queue_room: Arc<Condvar>,
    shutting_down: Arc<AtomicBool>,
    completed: Arc<Mutex<VecDeque<Completed>>>,
    queue_capacity: usize,
    workers: Vec<JoinHandle<()>>,
}

impl CpuExecutor {
    pub fn new(
        config: CpuConfig,
        handler: Arc<dyn Fn(&Action) -> Outcome + Send + Sync>,
    ) -> Result<Self, ExecutorConfigError> {
        if config.workers == 0 {
            return Err(ExecutorConfigError::ZeroWorkers);
        }
        if config.queue_capacity == 0 {
            return Err(ExecutorConfigError::ZeroQueueCapacity);
        }
        let queue = Arc::new(Mutex::new(VecDeque::new()));
        let queue_room = Arc::new(Condvar::new());
        let shutting_down = Arc::new(AtomicBool::new(false));
        let completed = Arc::new(Mutex::new(VecDeque::new()));
        let pending_sink = Arc::new(Mutex::new(VecDeque::new()));
        let mut workers = Vec::with_capacity(config.workers);
        for index in 0..config.workers {
            let queue = Arc::clone(&queue);
            let queue_room = Arc::clone(&queue_room);
            let shutting_down = Arc::clone(&shutting_down);
            let completed = Arc::clone(&completed);
            let pending_sink = Arc::clone(&pending_sink);
            let handler = Arc::clone(&handler);
            workers.push(
                thread::Builder::new()
                    .name(format!("logan-cpu-worker-{index}"))
                    .spawn(move || {
                        worker_loop(
                            queue,
                            queue_room,
                            shutting_down,
                            completed,
                            pending_sink,
                            handler,
                        )
                    })
                    .expect("worker thread must be creatable"),
            );
        }
        Ok(Self {
            queue,
            queue_room,
            shutting_down,
            completed,
            queue_capacity: config.queue_capacity,
            workers,
        })
    }

    /// Number of persistent worker threads.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Number of queued (not running) jobs.
    pub fn queued_len(&self) -> usize {
        self.queue.lock().expect("queue lock").len()
    }

    /// Submit one job without ever blocking. A full queue returns the job.
    pub fn try_submit_job(&self, job: Job) -> SubmitOutcome {
        if self.shutting_down.load(Ordering::Acquire) {
            return SubmitOutcome::Closed(job);
        }
        let mut queue = self.queue.lock().expect("queue lock");
        if queue.len() >= self.queue_capacity {
            return SubmitOutcome::Full(job);
        }
        queue.push_back(job);
        drop(queue);
        self.queue_room.notify_one();
        SubmitOutcome::Accepted
    }

    /// Cancel one queued job and return its retained leases immediately. A
    /// job already claimed by a worker is never preempted: it runs to
    /// completion and reports normally, so this returns `false` for it
    /// (matching the runtime's best-effort lane contract). A plain queued
    /// job without retained leases is dropped silently — its ticket was
    /// already revoked by the scheduler, so nothing needs reporting.
    pub fn cancel_ticket(&self, ticket: Ticket) -> bool {
        let mut queue = self.queue.lock().expect("queue lock");
        let Some(index) = queue.iter().position(|job| job.ticket == ticket) else {
            return false;
        };
        let job = queue.remove(index).expect("position checked");
        // Hand retained resources back exactly once, even when the ticket is
        // already stale in the scheduler (the owner releases, not completes).
        if !job.retained.is_empty() {
            self.completed
                .lock()
                .expect("completed lock")
                .push_back(Completed {
                    ticket,
                    outcome: Outcome::Err("cancelled before dispatch".into()),
                    released: job.retained,
                });
        }
        true
    }

    /// Drain every finished job in completion order. The scheduler owner uses
    /// this to collect `released` leases; tickets are delivered exactly once.
    pub fn drain_completed(&self) -> Vec<Completed> {
        self.completed
            .lock()
            .expect("completed lock")
            .drain(..)
            .collect()
    }

    /// Stop accepting work and join all workers. Idle workers wake and exit;
    /// a running job finishes before its worker exits.
    pub fn shutdown(&mut self) {
        self.shutting_down.store(true, Ordering::Release);
        self.queue_room.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl Drop for CpuExecutor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl BoundedExecutor for CpuExecutor {
    fn try_submit(&mut self, request: DispatchRequest) -> SubmitResult {
        match self.try_submit_job(Job {
            ticket: request.ticket,
            action: request.action,
            retained: Vec::new(),
            completion: Some(request.completion),
        }) {
            SubmitOutcome::Accepted => SubmitResult::Accepted,
            SubmitOutcome::Full(job) => SubmitResult::Full(job.into_request()),
            SubmitOutcome::Closed(job) => SubmitResult::Closed(job.into_request()),
        }
    }

    fn try_cancel(&mut self, ticket: Ticket) -> bool {
        self.cancel_ticket(ticket)
    }

    fn shutdown(&mut self) {
        self.shutdown();
    }
}

fn worker_loop(
    queue: Arc<Mutex<VecDeque<Job>>>,
    queue_room: Arc<Condvar>,
    shutting_down: Arc<AtomicBool>,
    completed: Arc<Mutex<VecDeque<Completed>>>,
    pending_sink: Arc<Mutex<VecDeque<(CompletionSink, Ticket, Outcome)>>>,
    handler: Arc<dyn Fn(&Action) -> Outcome + Send + Sync>,
) {
    loop {
        retry_parked(&pending_sink);
        let job = {
            let mut queue = queue.lock().expect("queue lock");
            loop {
                if let Some(job) = queue.pop_front() {
                    break job;
                }
                if shutting_down.load(Ordering::Acquire) {
                    return;
                }
                queue = queue_room.wait(queue).expect("queue condvar");
            }
        };
        let outcome = handler(&job.action);
        // Record exactly once: jobs without a runtime sink have no other
        // completion path, and retained leases must always come back. Sink
        // jobs (the runtime path) report through the sink instead, so their
        // records never pile up undrained.
        if job.completion.is_none() || !job.retained.is_empty() {
            completed.lock().expect("completed lock").push_back(Completed {
                ticket: job.ticket,
                outcome: outcome.clone(),
                released: job.retained,
            });
        }
        if let Some(sink) = &job.completion {
            report_to_sink(&pending_sink, sink, job.ticket, outcome);
        }
    }
}

/// Bounded retry policy for the runtime command channel: a few yields, then
/// park the report for a later worker. A persistently full scheduler never
/// loses a completion this way, and the executor never blocks on it.
// ponytail: parked reports retry on the next worker iteration; replace with a
// dedicated completion queue when #52 wires transfer tickets into the runtime.
fn report_to_sink(
    pending: &Mutex<VecDeque<(CompletionSink, Ticket, Outcome)>>,
    sink: &CompletionSink,
    ticket: Ticket,
    outcome: Outcome,
) {
    let mut attempts = 0;
    loop {
        match sink.try_complete(ticket, outcome.clone()) {
            Ok(()) => return,
            Err(CompletionSendError::Full) if attempts < 8 => {
                attempts += 1;
                thread::yield_now();
            }
            Err(CompletionSendError::Full) => {
                pending
                    .lock()
                    .expect("pending sink lock")
                    .push_back((sink.clone(), ticket, outcome));
                return;
            }
            Err(CompletionSendError::Closed) => return,
        }
    }
}

fn retry_parked(pending: &Mutex<VecDeque<(CompletionSink, Ticket, Outcome)>>) {
    let mut retried = Vec::new();
    {
        let mut pending = pending.lock().expect("pending sink lock");
        while let Some(entry) = pending.pop_front() {
            retried.push(entry);
        }
    }
    for (sink, ticket, outcome) in retried {
        report_to_sink(pending, &sink, ticket, outcome);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched::core::{ActionKind, Budget, Effect, SchedulerCore};
    use crate::sched::ids::SessionId;
    use crate::sched::residency::{
        CompletionDisposition, DeviceId, ExpertKey, LoadCompletion, LoadDisposition, LoadResult,
        MemoryPoolId, RepresentationKey, ResidencyManager,
    };
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::time::{Duration, Instant};

    fn ok_handler() -> Arc<dyn Fn(&Action) -> Outcome + Send + Sync> {
        Arc::new(|_: &Action| Outcome::Ok)
    }

    fn job(ticket: Ticket, payload: u8) -> Job {
        Job {
            ticket,
            action: Action {
                kind: ActionKind::Cpu,
                payload: vec![payload],
            },
            retained: Vec::new(),
            completion: None,
        }
    }

    fn wait_until(condition: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !condition() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(condition(), "timed out waiting for condition");
    }

    fn wait_for_completions(executor: &CpuExecutor, count: usize) -> Vec<Completed> {
        let mut all = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while all.len() < count && Instant::now() < deadline {
            all.extend(executor.drain_completed());
            if all.len() < count {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        assert_eq!(all.len(), count, "timed out waiting for {count} completions");
        all
    }

    #[test]
    fn default_worker_count_is_machine_derived_and_config_is_validated() {
        let expected = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        assert_eq!(CpuConfig::default().workers, expected);
        assert!(CpuConfig::default().queue_capacity > 0);
        let handler = ok_handler();
        assert!(matches!(
            CpuExecutor::new(
                CpuConfig {
                    workers: 0,
                    queue_capacity: 4,
                },
                handler.clone(),
            ),
            Err(ExecutorConfigError::ZeroWorkers)
        ));
        assert!(matches!(
            CpuExecutor::new(
                CpuConfig {
                    workers: 1,
                    queue_capacity: 0,
                },
                handler,
            ),
            Err(ExecutorConfigError::ZeroQueueCapacity)
        ));
    }

    #[test]
    fn single_worker_runs_every_job_and_reports_each_ticket_once() {
        let mut executor = CpuExecutor::new(
            CpuConfig {
                workers: 1,
                queue_capacity: 8,
            },
            ok_handler(),
        )
        .unwrap();
        assert_eq!(executor.worker_count(), 1);
        for payload in 0..5u8 {
            assert!(matches!(
                executor.try_submit_job(job(Ticket(payload.into()), payload)),
                SubmitOutcome::Accepted
            ));
        }
        let completed = wait_for_completions(&executor, 5);
        let tickets: Vec<_> = completed.iter().map(|record| record.ticket).collect();
        assert_eq!(tickets, vec![Ticket(0), Ticket(1), Ticket(2), Ticket(3), Ticket(4)]);
        assert!(completed.iter().all(|record| record.outcome == Outcome::Ok));
        assert!(completed.iter().all(|record| record.released.is_empty()));
        // Every ticket completes exactly once: a second drain is empty.
        assert!(executor.drain_completed().is_empty());
        executor.shutdown();
    }

    #[test]
    fn saturation_returns_backpressure_and_never_blocks_submitter() {
        // Workers park inside the handler until the gate opens, so the queue
        // stays full and submissions are deterministic.
        let gate = Arc::new(AtomicBool::new(false));
        let started = Arc::new(AtomicUsize::new(0));
        let handler = {
            let gate = Arc::clone(&gate);
            let started = Arc::clone(&started);
            Arc::new(move |_: &Action| {
                started.fetch_add(1, AtomicOrdering::SeqCst);
                while !gate.load(AtomicOrdering::SeqCst) {
                    std::hint::spin_loop();
                }
                Outcome::Ok
            })
        };
        let mut executor = CpuExecutor::new(
            CpuConfig {
                workers: 2,
                queue_capacity: 2,
            },
            handler,
        )
        .unwrap();
        for payload in 0..2u8 {
            assert!(matches!(
                executor.try_submit_job(job(Ticket(payload.into()), payload)),
                SubmitOutcome::Accepted
            ));
        }
        wait_until(|| started.load(AtomicOrdering::SeqCst) == 2);
        // Two more fill the queue exactly.
        for payload in 2..4u8 {
            assert!(matches!(
                executor.try_submit_job(job(Ticket(payload.into()), payload)),
                SubmitOutcome::Accepted
            ));
        }
        assert_eq!(executor.queued_len(), 2);
        // The fifth submission is backpressure: the job comes straight back.
        let fifth = job(Ticket(9), 9);
        match executor.try_submit_job(fifth) {
            SubmitOutcome::Full(returned) => assert_eq!(returned.ticket, Ticket(9)),
            other => panic!("expected backpressure, got {other:?}"),
        }
        gate.store(true, AtomicOrdering::SeqCst);
        let completed = wait_for_completions(&executor, 4);
        let mut tickets: Vec<_> = completed.iter().map(|record| record.ticket).collect();
        tickets.sort_unstable();
        assert_eq!(tickets, vec![Ticket(0), Ticket(1), Ticket(2), Ticket(3)]);
        assert_eq!(executor.queued_len(), 0);
        executor.shutdown();
    }

    #[test]
    fn cancel_removes_queued_job_and_ignores_running_one() {
        let gate = Arc::new(AtomicBool::new(false));
        let started = Arc::new(AtomicUsize::new(0));
        let handler = {
            let gate = Arc::clone(&gate);
            let started = Arc::clone(&started);
            Arc::new(move |_: &Action| {
                started.fetch_add(1, AtomicOrdering::SeqCst);
                while !gate.load(AtomicOrdering::SeqCst) {
                    std::hint::spin_loop();
                }
                Outcome::Ok
            })
        };
        let mut executor = CpuExecutor::new(
            CpuConfig {
                workers: 2,
                queue_capacity: 4,
            },
            handler,
        )
        .unwrap();
        for payload in 0..4u8 {
            assert!(matches!(
                executor.try_submit_job(job(Ticket(payload.into()), payload)),
                SubmitOutcome::Accepted
            ));
        }
        wait_until(|| started.load(AtomicOrdering::SeqCst) == 2);
        // Tickets 2 and 3 are still queued; tickets 0 and 1 are running.
        assert!(executor.cancel_ticket(Ticket(2)));
        assert!(executor.cancel_ticket(Ticket(3)));
        assert!(!executor.cancel_ticket(Ticket(2))); // idempotent: gone already
        assert!(!executor.cancel_ticket(Ticket(0))); // running: not preemptable
        assert_eq!(executor.queued_len(), 0);
        gate.store(true, AtomicOrdering::SeqCst);
        // Only the running jobs report; the cancelled ones never run.
        let completed = wait_for_completions(&executor, 2);
        assert!(completed.iter().all(|record| {
            record.ticket == Ticket(0) || record.ticket == Ticket(1)
        }));
        assert_eq!(
            executor.drain_completed().len(),
            0,
            "each ticket completes exactly once"
        );
        executor.shutdown();
    }

    #[test]
    fn failure_is_reported_exactly_once_with_error_outcome() {
        let handler: Arc<dyn Fn(&Action) -> Outcome + Send + Sync> = Arc::new(|action: &Action| {
            if action.payload == [1] {
                Outcome::Err("boom".into())
            } else {
                Outcome::Ok
            }
        });
        let mut executor = CpuExecutor::new(
            CpuConfig {
                workers: 1,
                queue_capacity: 4,
            },
            handler,
        )
        .unwrap();
        assert!(matches!(
            executor.try_submit_job(job(Ticket(1), 1)),
            SubmitOutcome::Accepted
        ));
        assert!(matches!(
            executor.try_submit_job(job(Ticket(2), 2)),
            SubmitOutcome::Accepted
        ));
        let completed = wait_for_completions(&executor, 2);
        assert!(completed.contains(&Completed {
            ticket: Ticket(1),
            outcome: Outcome::Err("boom".into()),
            released: Vec::new(),
        }));
        assert!(completed.contains(&Completed {
            ticket: Ticket(2),
            outcome: Outcome::Ok,
            released: Vec::new(),
        }));
        assert!(executor.drain_completed().is_empty());
        executor.shutdown();
    }

    #[test]
    fn submit_after_shutdown_is_closed_and_workers_are_joined() {
        let mut executor = CpuExecutor::new(
            CpuConfig {
                workers: 2,
                queue_capacity: 4,
            },
            ok_handler(),
        )
        .unwrap();
        executor.shutdown();
        match executor.try_submit_job(job(Ticket(5), 5)) {
            SubmitOutcome::Closed(returned) => assert_eq!(returned.ticket, Ticket(5)),
            other => panic!("expected closed, got {other:?}"),
        }
        // Idempotent: a second shutdown still joins and returns cleanly.
        executor.shutdown();
    }

    #[test]
    fn scheduler_submit_completion_releases_residency_lease() {
        // Residency: one resident expert, one lease in hand.
        let pool = MemoryPoolId(0);
        let mut residency = ResidencyManager::new();
        residency.add_pool(pool, 100).unwrap();
        residency.add_device(DeviceId(7), vec![pool]).unwrap();
        let expert = ExpertKey {
            model: 1,
            package: 2,
            layer: 3,
            expert: 4,
            representation: RepresentationKey {
                layout: 1,
                kernel_abi: 7,
                quant: 8,
            },
            pool,
        };
        let load = match residency.begin_load(expert.clone(), 40, SessionId(1)).unwrap() {
            LoadDisposition::Started { load_id } => load_id,
            other => panic!("unexpected {other:?}"),
        };
        let lease = match residency
            .complete_load(LoadCompletion {
                load_id: load,
                result: LoadResult::Success,
            })
            .unwrap()
        {
            CompletionDisposition::Published { wakes, .. } => wakes[0].result.clone().unwrap(),
            other => panic!("unexpected {other:?}"),
        };
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 40);

        // Scheduler submits an opaque CPU action; dispatch lands on the
        // executor with the lease retained by the job.
        let mut core = SchedulerCore::new(Budget::default());
        let session = core.create_session();
        let (ticket, action) = match core
            .submit(
                session,
                Action {
                    kind: ActionKind::Cpu,
                    payload: vec![1],
                },
            )
            .unwrap()
        {
            (ticket, Effect::Dispatch { action, .. }) => (ticket, action),
            other => panic!("unexpected {other:?}"),
        };
        core.block(session).unwrap();
        let mut executor = CpuExecutor::new(
            CpuConfig {
                workers: 1,
                queue_capacity: 4,
            },
            ok_handler(),
        )
        .unwrap();
        assert!(matches!(
            executor.try_submit_job(Job {
                ticket,
                action,
                retained: vec![lease],
                completion: None,
            }),
            SubmitOutcome::Accepted
        ));

        // Ticket completion: reported exactly once, scheduler wakes.
        let completed = wait_for_completions(&executor, 1);
        assert_eq!(completed[0].ticket, ticket);
        assert_eq!(completed[0].outcome, Outcome::Ok);
        assert_eq!(completed[0].released.len(), 1);
        let effects = core.complete(ticket, Outcome::Ok).unwrap();
        assert!(effects.iter().any(|effect| {
            matches!(effect, Effect::Wake { session: id } if *id == session)
        }));
        assert_eq!(core.total_inflight(), 0);

        // Lease release on the scheduler side: unpinned, evictable, whole.
        residency.release(completed[0].released[0].clone()).unwrap();
        assert_eq!(residency.pool_stats(pool).unwrap().pinned_bytes, 0);
        assert_eq!(residency.evict(pool).unwrap(), Some(expert));
        residency.assert_invariants();
        executor.shutdown();
    }
}