# Logan Chat

`logan-chat` is the interactive Ratatui client for Logan's Qwen4 / Qwen3.8-Flash-Next runtime.

It is intentionally a runtime test surface, not a wrapper around the one-shot CLI:

- one `Model` stays loaded for the life of the chat;
- completed causal state is reused directly across turns;
- persistent `.lpfx` state is used for startup/shared-prefix recovery;
- the first system-message boundary is persisted as a semantic checkpoint;
- validated Logan performance paths are enabled by default and remain opt-out through their existing `QWEN_*` environment variables;
- generation is streamed into the TUI token-by-token;
- runtime counters are read directly from the model/Metal backend, not parsed from stderr.

## Build

```bash
cargo build --release -p logan-chat
```

## Run

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
