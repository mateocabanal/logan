# Logan Chat

`logan-chat` is the interactive Ratatui client for Logan's Qwen4 / Qwen3.8-Flash-Next runtime.

It is intentionally a runtime test surface, not a wrapper around the one-shot CLI:

- one `Model` stays loaded for the life of the chat;
- completed causal state is reused directly across turns;
- persistent `.lpfx` state is used for startup/shared-prefix recovery;
- the first system-message boundary is persisted as a semantic checkpoint;
- validated Logan performance paths are enabled by default and remain opt-out through their existing `QWEN_*` environment variables;
- generation is streamed into the TUI token-by-token;
- the default system prompt follows the applicable OpenAI latest-model harness guidance: infer intent, act on authorized requests, ask only when missing information materially changes the result, communicate directly, and calibrate verification to the task;
- `--system TEXT` or `/system TEXT` replaces that default when a model- or workflow-specific prompt is desired;
- runtime counters are read directly from the model/Metal backend, not parsed from stderr.

## Build

```bash
cargo build --release -p logan-chat
```

## Background daemon + web dashboard

`logand` is Logan's resident control plane. It can stay running with no model loaded, discovers Logan packages under `~/models`, and loads/unloads a model without restarting the service.

```bash
cargo run -p logan-chat --bin logand
# dashboard: http://127.0.0.1:11435/
```

The dashboard exposes live prompt/decode throughput, TTFT, context use, expert residency and hit rate, Metal/MetalIO timing, and current fast-path flags. Prompt caching is explicitly tiered:

- **hot / RAM** — a bounded LRU of exact `QwenStateSnapshot` causal prefixes plus boundary logits; exact prompt hits can resume with zero prompt-token replay;
- **cold / SSD** — immutable persistent `.lpfx` snapshots under Logan's prefix-cache directory;
- **active state** — the causal state for the currently loaded conversation, reported separately from the reusable RAM cache.

The RAM prefix-cache budget defaults to 512 MiB. Override it with `LOGAN_HOT_PREFIX_CACHE_BYTES` or `LOGAN_HOT_PREFIX_CACHE_MB`; set the byte budget to `0` to disable the RAM tier.

For an oMLX-style login daemon on macOS:

```bash
./tools/install-logand.sh
```

This builds `logand`, copies it into `~/Library/Application Support/Logan/bin`, and registers `dev.logan.logand` as a per-user LaunchAgent. `./tools/uninstall-logand.sh` removes the service and installed binary while leaving prompt caches and logs intact.

By default the dashboard listens only on `127.0.0.1`. Use `logand --host ADDRESS --port PORT --model-dir PATH` for an explicit alternative.

## OpenAI-compatible API

`logand` exposes OpenAI-compatible text inference endpoints on the same port:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`

Both generation APIs support ordinary JSON responses and SSE streaming. Chat Completions supports `developer`, `system`, `user`, and `assistant` text messages, `max_completion_tokens` / legacy `max_tokens`, `temperature`, `top_p`, the Logan `top_k` extension, and `stream_options.include_usage`.

```bash
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "logan",
    "messages": [{"role":"user","content":"Say hello."}],
    "stream": true,
    "max_completion_tokens": 64
  }'
```

The Responses API accepts string input or text message arrays and supports `previous_response_id` for daemon-local continuation:

```bash
curl http://127.0.0.1:11435/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "logan",
    "input": "Say hello.",
    "max_output_tokens": 64
  }'
```

Protocol requests are stateless with respect to the active dashboard/TUI conversation, but reuse the resident model and the same RAM → SSD prefix-cache hierarchy. Responses include an `x_logan` extension with detailed prompt/decode, cache, expert-residency, Metal, MetalIO, and per-phase timing metrics. The dashboard's **Chat** page uses the streaming Chat Completions endpoint directly and displays those metrics alongside the conversation.

Current compatibility is text-only. Multimodal content, tools/function calls, custom stop sequences, background Responses, and structured/JSON output formats are rejected explicitly instead of being ignored. `n` is currently limited to 1.

## Run TUI

```bash
./target/release/logan-chat \
  ~/models/Qwen3.8-Flash-Next-FP8.Apple8.coli
```

Example with explicit sampling:

```bash
./target/release/logan-chat \
  ~/models/Qwen3.8-Flash-Next-FP8.Apple8.coli \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 40 \
  --max-new 256
```

Use `--greedy` for deterministic greedy decoding.

## Keys

| Key | Action |
| --- | --- |
| Enter | Send |
| Ctrl+J / Ctrl+Enter | Newline |
| Esc | Cancel after the current token |
| Up / Down | Prompt history |
| PgUp / PgDn | Scroll conversation |
| Tab | Toggle runtime stats |
| F1 | Help |
| Ctrl+W | Delete previous word |
| Ctrl+U | Clear prompt editor |
| Ctrl+C | Quit |

## Commands

- `/clear` — new session with the same system prompt
- `/system TEXT` — new session and replace the system prompt
- `/max N` — maximum response tokens
- `/temp F` — temperature
- `/top-p F` — nucleus sampling probability
- `/top-k N` — top-k limit (`0` means no top-k limit)
- `/repeat F` — repeat penalty
- `/greedy` — temperature 0 + top-k 1
- `/stats` — toggle runtime stats
- `/save [FILE]` — write a readable transcript
- `/quit` — quit

## Runtime dashboard

The side panel exposes per-turn and live counters including:

- prompt latency, TTFT, generation rate, wall time;
- live-state tokens reused and SSD-prefix tokens restored;
- SSD restore/write time and on-disk prefix-cache footprint;
- context usage;
- routed-expert LRU occupancy, hits, misses, hit rate and evictions;
- Metal encode/submit/wait/kernel time;
- fused MoE calls and fused expert count;
- MetalIO loads, bytes, waits, failures, outstanding requests and average latency;
- GDN, attention, hyper-connection, head, routed I/O, shared-expert and GPU MoE phase timing;
- active performance paths;
- process peak RSS;
- current sampling parameters and last token id.

## Performance A/B

Validated fast paths are the default. Opt out before launch for targeted A/Bs, for example:

```bash
QWEN_PREFIX_CACHE=0 ./target/release/logan-chat MODEL.coli
QWEN_SHARED_IO_OVERLAP=0 ./target/release/logan-chat MODEL.coli
QWEN_ATTN_METAL=0 ./target/release/logan-chat MODEL.coli
QWEN_BNNS_BF16=0 ./target/release/logan-chat MODEL.coli
```

The TUI uses the package's copied `tokenizer.json` and Qwen ChatML markers (`<|im_start|>` / `<|im_end|>`).
