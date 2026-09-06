# OpenAI latest-model guidance applicability

Reviewed against https://developers.openai.com/api/docs/guides/latest-model on 2026-09-05.

Logan is a local inference compiler/runtime. It does not currently issue OpenAI API requests, so the GPT-6 Astra transport and request-parameter migration rules are not runtime requirements for Logan's Qwen4 path. The behavioral guidance for model harnesses and coding agents is still useful and is applied where appropriate.

## Applied

| Guidance area | Logan status |
| --- | --- |
| Infer intent and carry authorized work through instead of stopping at a plan | Applied in root `AGENTS.md` and the default `logan-chat` system prompt |
| Ask focused questions only when missing information can materially change the result | Applied in `AGENTS.md` and the default chat prompt |
| Finish reversible/read-only authorized work before requesting approval | Applied in `AGENTS.md` and the default chat prompt |
| Make user-instruction priority explicit | Applied in `AGENTS.md` and the default chat prompt |
| Keep writing direct and calibrated to the user's technical context | Applied in `AGENTS.md` and the default chat prompt |
| Delegate/parallelize when useful | Applied as repository agent guidance in `AGENTS.md` |
| Calibrate tests to the change and avoid redundant broad verification | Applied in `AGENTS.md` and the default chat prompt |
| Audit instruction files that can influence agent behavior | Root `AGENTS.md` is now the explicit repository instruction surface |

## Not applicable to the current local runtime

These items describe GPT-6 Astra requests to OpenAI and must not be blindly imposed on Logan's local Qwen4 sampler:

- setting `model` to `gpt-6-astra`;
- using the OpenAI Responses API for tool calling;
- removing `temperature`, `top_p`, `top_logprobs`, or Chat Completions `logprobs`;
- representing reasoning effort with OpenAI's `reasoning.effort` / `reasoning_effort` fields;
- using OpenAI `configuration_update` input items;
- replacing OpenAI `prompt_cache_retention` with `prompt_cache_options.ttl`;
- OpenAI service-tier and EU data-residency constraints;
- OpenAI async-tool-call and mid-turn-steering wire protocols.

Logan's `temperature`, `top_p`, and `top_k` settings are local decoding controls for Qwen4 and remain valid. Their defaults should follow the model's own inference guidance and measured output quality, not the Astra API parameter surface.

## If Logan adds an OpenAI-backed provider

An OpenAI provider should be implemented as a separate transport/configuration layer rather than by changing local sampler semantics. For GPT-6 Astra that provider should:

1. use the Responses API for tool calling;
2. omit unsupported sampling/logprob parameters;
3. use a supported reasoning effort (`low` rather than `none`/`minimal` when migrating from those values);
4. preserve request-level reasoning settings when changing effort mid-conversation and use compatible configuration-update items where supported;
5. use the current prompt-cache options and service-tier compatibility rules;
6. preserve tool `call_id` values across async tool execution and results;
7. keep model/provider-specific validation isolated from Qwen/local generation settings.

Re-check the upstream guide before implementing that provider because the API surface can change independently of Logan.
