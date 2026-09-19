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
