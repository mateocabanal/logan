# Logan agent guidance

This repository uses the OpenAI latest-model guidance as the baseline for coding-agent behavior where it applies to repository work. Source reviewed: https://developers.openai.com/api/docs/guides/latest-model (2026-09-05).

## Carry work through

- Infer the intended scope from the user's request and the existing conversation/repository context.
- When the user asks for a change, perform the authorized work rather than stopping at a plan or capability statement.
- Keep going until the requested outcome is complete or a concrete external blocker prevents further progress.
- Before asking a question, finish any read-only, reversible, or otherwise authorized work that can make the remaining decision concrete.
- Ask only when missing information can materially change the result and cannot be resolved from the repository or existing context.

## Instruction priority

- Explicit user instructions take precedence over repository workflow guidance and skills.
- Preserve unrelated working-tree changes. Do not overwrite or revert user work unless explicitly asked.
- If an instruction file or skill forces a pause, permission request, or divergence from the user's requested outcome, identify that file and the relevant requirement rather than silently changing direction.

## Communication

- State the main result early and use direct, plain technical language.
- Use lists or tables when they make parallel information easier to scan; otherwise prefer coherent paragraphs.
- Match the level of technical detail to the user's request and demonstrated context.
- Avoid canned transitions, filler, and vague claims. Report concrete evidence, measurements, files, and commands when they matter.

## Delegation and parallel work

- Parallelize independent investigations when doing so can materially reduce latency or improve confidence.
- Keep delegated tasks narrowly scoped and reconcile their results before making a final claim.
- Human-readable agent messages should use normal spacing and clear wording.

## Testing and verification

- Run checks that are proportionate to the change and that can actually catch relevant regressions.
- Do not add tests that merely restate a trivial reversible implementation detail.
- Once the relevant checks pass, broaden or repeat testing only when a failure, new change, or unresolved risk justifies it.
- For performance work, prefer an explicit hypothesis and an A/B comparison over speculative tuning. Do not claim a performance win without an appropriate measurement.

## Logan-specific safeguards

- Treat the current dirty working tree as valuable state. Inspect diffs before touching an already-modified file.
- Preserve token-identity/correctness gates when changing runtime or kernel behavior.
- Keep benchmark conditions comparable and record any environment or model differences that affect interpretation.
- Do not generalize OpenAI API parameter restrictions to Logan's local model samplers. Model-specific decoding settings remain governed by the model/runtime being served.


## Experiment ledger

- Every non-trivial performance, architecture, quantization, accelerator, storage, cache, scheduling, or kernel experiment MUST be recorded in EXPERIMENTS.md.
- Create/update the experiment entry in the same change that implements or measures it. Record failed and inconclusive experiments, not only wins.
- Use an explicit hypothesis, baseline, correctness gate, comparable A/B measurements, status, and keep/reject rationale.
- Do not enable an experimental performance path by default until its ledger entry has a repeatable correctness-preserving win. If a path is rejected, remove the losing production code/flag when practical while retaining the ledger record.
