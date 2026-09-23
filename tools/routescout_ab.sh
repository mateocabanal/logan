#!/usr/bin/env bash
# Paired A/B driver for the RouteScout real-model prefetch qualification.
#
# Baseline  (B): stock decode, no prediction flags.
# Candidate (C): online predictor + speculative prefetch enabled.
#
# Identical model, prompt, sampler, token count, and flags apart from the
# predictor switches. Alternating B/C/C/B order cancels monotonic drift
# (thermal, page-cache warm-up) that a single-block A/B would absorb.
#
# Two details that make the measurement valid:
#
#   * The recorded metric is the binary's steady-state decode figure, not its
#     total. Total includes model load and prompt prefill as a large constant,
#     which dilutes a real per-token difference toward zero.
#   * TOKENS must exceed the predictor's confidence warmup or the gate blocks
#     every prefetch and the candidate arm is inert. `prepare_route_prediction`
#     requires `pairs >= QWEN_ROUTE_PREDICT_MIN_SAMPLES` (default 16) per layer,
#     so >= 18 forwards are needed before any speculation is issued. The gate is
#     left at its default so the run exercises the shipped policy.
set -u

MODEL="${MODEL:-$HOME/models/Qwen3.6-35B-A3B-MLX-oQ4-FP16}"
PROMPT="${PROMPT:-1 2 3 4 5 6 7 8}"
TOKENS="${TOKENS:-24}"
PAIRS="${PAIRS:-6}"
OUT="${OUT:-.perf_runs/routescout/EXP-020-prefetch-ab}"
BIN="${BIN:-./target/release/logan-qwen4}"

mkdir -p "$OUT"
: > "$OUT/runs.tsv"
printf 'index\tarm\tdecode_ms_per_token\tprefill_ms\ttotal_ms\tgenerated\tmetal\tfallback\tprefetch_loads\tpairs\tpredicted\n' >> "$OUT/runs.tsv"

run_one() {
    local index="$1" arm="$2"
    local log="$OUT/run-${index}-${arm}.log"

    if [ "$arm" = "B" ]; then
        LOGAN_PROFILE=1 QWEN_PROMPT="$PROMPT" QWEN_MAX_NEW="$TOKENS" QWEN_TOKEN_TIMING=1 \
            "$BIN" "$MODEL" > "$log" 2>&1
    else
        LOGAN_PROFILE=1 QWEN_PROMPT="$PROMPT" QWEN_MAX_NEW="$TOKENS" QWEN_TOKEN_TIMING=1 \
            QWEN_ROUTE_PREDICT=1 QWEN_ROUTE_PREDICT_PREFETCH=1 \
            QWEN_ROUTE_SPEC_CACHE=64 QWEN_ROUTE_PREDICT_BUDGET=8 \
            "$BIN" "$MODEL" > "$log" 2>&1
    fi

    local decode prefill total gen metal fallback loads pairs predicted
    decode=$(grep -o 'decode=[0-9.]* ms/tok' "$log" | tail -1 | sed 's/decode=//;s/ ms\/tok//')
    prefill=$(grep -o 'prefill=[0-9.]* ms' "$log" | tail -1 | sed 's/prefill=//;s/ ms//')
    total=$(grep -o 'total=[0-9.]* ms' "$log" | tail -1 | sed 's/total=//;s/ ms//')
    gen=$(grep -o 'generated: \[[^]]*\]' "$log" | tail -1 | sed 's/generated: //')
    local affine
    affine=$(grep -o 'logan mlx-affine: metal=[0-9]* fallback=[0-9]*' "$log" | tail -1)
    metal=$(printf '%s' "$affine" | sed 's/.*metal=//;s/ .*//')
    fallback=$(printf '%s' "$affine" | sed 's/.*fallback=//')
    loads=$(grep -o 'mio loads=[0-9]*' "$log" | tail -1 | sed 's/mio loads=//')
    pairs=$(grep -o 'pairs=[0-9]* precision' "$log" | tail -1 | sed 's/pairs=//;s/ precision//')
    predicted=$(grep -o 'predicted=[0-9]*' "$log" | tail -1 | sed 's/predicted=//')

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$index" "$arm" "${decode:-NA}" "${prefill:-NA}" "${total:-NA}" "${gen:-NA}" \
        "${metal:-NA}" "${fallback:-NA}" "${loads:-0}" "${pairs:-0}" "${predicted:-0}" \
        >> "$OUT/runs.tsv"
    echo "[$index/$arm] decode=${decode:-NA} ms/tok prefill=${prefill:-NA} loads=${loads:-0} pairs=${pairs:-0}"
}

index=0
for pair in $(seq 1 "$PAIRS"); do
    index=$((index + 1)); run_one "$index" B
    index=$((index + 1)); run_one "$index" C
    index=$((index + 1)); run_one "$index" C
    index=$((index + 1)); run_one "$index" B
done

echo "wrote $OUT/runs.tsv"
