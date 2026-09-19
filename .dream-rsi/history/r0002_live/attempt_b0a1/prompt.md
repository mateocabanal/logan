You must read every historical proposal before proposing or implementing a new solution.



Variables (`/Users/mateo/CODE/logan/.dream-rsi/work/r0002/b0a1`, `/Users/mateo/CODE/logan/.dream-rsi/history`, `/Users/mateo/CODE/logan/.dream-rsi/history/baseline`, `logan-core/src/lib.rs`, `/Users/mateo/CODE/logan/task/problem.md`) are filled in
by the calling system. `/Users/mateo/CODE/logan/.dream-rsi/work/r0002/b0a1` is your own attempt directory -- exclude it when scanning sibling
`attempt_*/` dirs.

## 1. Read the complete history first

Before proposing anything, read every `proposal.md` under sibling `attempt_*/` dirs, `/Users/mateo/CODE/logan/.dream-rsi/history`, and
`/Users/mateo/CODE/logan/.dream-rsi/history/baseline` in full -- not a sample, not just recent cycles or the current branch. For each, read its
matching `eval/score.json` (and `error.txt` if it failed). Trust the measured result over what the
proposal claims about itself.

## 2. Learn from both successes and failures

For every past attempt, note the mechanism and how it did. For failures, figure out *why*: a flawed core
idea, or a good idea let down by a bug, bad parameters, or an implementation slip? Don't repeat the
former. The latter is worth retrying -- but only once you've actually located the bug in the code (not
just guessed from the proposal), and only with a specific fix in hand.

## 3. Don't converge into a local optimum

Look at the shape of what's been tried. If most attempts cluster around small variations of one mechanism
with flattening returns, that's a local optimum -- resist proposing another small tweak there.
Deliberately favor a structurally different mechanism or an untried combination over a safer marginal
refinement. Exploration diversity matters as much as the next incremental gain.

## 4. Propose and implement

The new idea must be a genuinely new mechanism, a new combination of previously-successful pieces, or a
targeted fix to a specific bug found in step 2 -- never a repeat or rename of something already tried.
Implement it in `logan-core/src/lib.rs`. Don't claim it compiles, is correct, or beats SOTA until it's actually
evaluated.

## Files

Write only `/Users/mateo/CODE/logan/.dream-rsi/work/r0002/b0a1/proposal.md` (mechanism, evidence from history, why it's not a repeat, expected
benefit/risk) and `/Users/mateo/CODE/logan/.dream-rsi/work/r0002/b0a1/logan-core/src/lib.rs`. Everything else is read-only.

## Problem statement

# Task: Make Logan's runtime subsystems model-agnostic

The candidate workspace is a source-only snapshot of Logan's current working tree. The objective is an end-to-end refactor: move reusable causal state, page/COW transactions, RAM and SSD longest-prefix caching, versioned snapshot framing, residency/resources, speculative lifecycle, and model-neutral telemetry into shared runtime code while keeping forward graphs, attention/router/state meaning, tokenization, and drafter conditioning in engines.

`logan-core` is the shared runtime anchor (`logan-core/src/lib.rs` is the declared eval program), but attempts may edit the relevant workspace crates: `logan-core`, `logan-ir`, `logan-metal`, `logan-ane`, `logan-llama`, `logan-qwen4`, `logan-chat`, `logand`, and `logan-compiler`. Do not edit the scorer, this problem file, fixtures, generated artifacts, or unrelated pre-existing work.

Required outcome:
- generic state regions equivalent to AppendOnly, Ring, MutableFixed, SparsePaged, and Opaque;
- generation-safe checkpoint/commit/rollback with COW/page reclamation and cancellation isolation;
- one shared, model-neutral RAM -> SSD prefix cache with longest reusable token-prefix lookup;
- stable versioned snapshot container with bounded lengths, checksums, atomic publication, and strict model/runtime/tokenizer/template/schema/prefix identity rejection;
- codec boundary where Llama/MiniCPM and Qwen4Exp own state meaning but core owns lifecycle/storage;
- dense Llama/MiniCPM RAM and cross-process SSD reuse with uncached suffix forwarding and cold/restored equivalence;
- Qwen4Exp migration without losing heterogeneous GDN/QSA/PLE state;
- generic speculative transaction control, resources/residency, and telemetry; no second Qwen-only mechanism.

Correctness is non-negotiable. The fixed scorer, outside this seed, runs `cargo test --workspace --all-targets`; the seed baseline is 477 passed, 0 failed, 3 ignored. A candidate is invalid if it fails to compile, fails any test, drops below the baseline pass count, or increases ignored tests. The score is 1000 for that correctness floor plus up to 100 architecture points for observable shared-runtime surfaces. Do not game the scorer with deleted/weakened tests, string-only stubs, or cached/fake output. Do not run or modify the scorer from an attempt.

Existing evidence to preserve: the current tree has Qwen-specific `.lpfx` prefix code and token-identity gates, while the dense MiniCPM/Llama path has model/session tests but no equivalent generic prefix-cache adoption. The repository is intentionally dirty with active MiniCPM5, ANE, daemon, compiler, and protocol work; preserve and reconcile it.

The final implementation must remain buildable and update focused tests/docs. Run only targeted checks while iterating; the fixed scorer owns the full workspace validation.

## Evaluation

Do not run the scoring program yourself unless the task problem statement above tells you to; the runtime
evaluates your workspace with a fixed, read-only scoring protocol and writes `eval/score.json`. Your job
is the implementation and the proposal.

## Note:

Never execute pkill, kill, killall, or terminate unrelated processes.
