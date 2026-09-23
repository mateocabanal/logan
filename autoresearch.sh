#!/usr/bin/env bash
#
# Canonical benchmark entrypoint for Qwen3.6-35B-A3B decode throughput.
#
# WHAT IS MEASURED
#   Steady-state decode tok/s on the real Qwen3.6-35B-A3B checkpoint with routed
#   experts streamed from the SSD (LOGAN_EXPERT_NOCACHE=1) — the configuration
#   this repository's RouteScout / SSD-streaming work targets.
#
# TWO ARMS
#   greedy  real chat-templated prompt, argmax decode. The trajectory the
#           existing correctness gates pin.
#   sample  same prompt, seeded temperature-1.0 multinomial sampling over the
#           full vocabulary — a realistic non-greedy trajectory.
#
#   The sampled arm is deterministic *because the logits are*: a fixed seed
#   replays the identical trajectory as long as the model computes the same
#   numbers. A change that alters logits can shift that trajectory; that is a
#   numerics change, not a harness failure, and `*_trajectory_sha` surfaces it.
#
# TWO MEASUREMENT DECISIONS THAT MATTER
#   1. Arms run INTERLEAVED (greedy, sample, greedy, sample, ...). A plain
#      "all greedy, then all sample" schedule biases whichever arm runs second,
#      because host thermal/cache state drifts across a multi-minute run. An
#      earlier version of this harness measured the second arm ~25% slower even
#      though both arms decoded the IDENTICAL trajectory — pure position effect.
#   2. Times are POOLED. Every observed decode step across all repeats of an arm
#      goes into one sample, and the median of that sample is the metric. A
#      24-token decode has wide step-to-step variance, so an arm-median over one
#      run is fragile; pooling across repeats is robust to one arm catching a
#      stall.
#
# NO NETWORK. No clock dependence. Fixed prompt, seed, token count, repeats.
#
# Usage: bash autoresearch.sh [--repeats N] [--tokens N] [--arm greedy|sample|both]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODEL_DIR="${LOGAN_BENCH_MODEL:-/Users/mateo/models/Qwen3.6-35B-A3B-MLX-oQ4-FP16}"
TOKENS="${LOGAN_BENCH_TOKENS:-24}"
REPEATS="${LOGAN_BENCH_REPEATS:-3}"
PROMPT="${LOGAN_BENCH_PROMPT:-Explain why memory safety matters in systems programming and how Rust achieves it without a garbage collector.}"
SEED="${LOGAN_BENCH_SEED:-20260923}"
TEMP="${LOGAN_BENCH_TEMP:-1.0}"
ARM="both"

while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARM="$2"; shift 2 ;;
    --tokens) TOKENS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ ! -d "$MODEL_DIR" ]; then
  echo "benchmark model directory not found: $MODEL_DIR" >&2
  exit 1
fi
case "$ARM" in greedy|sample|both) ;; *) echo "bad --arm: $ARM" >&2; exit 2 ;; esac

echo "== building decode_bench (release) =="
cargo build --release -p logan-qwen4 --example decode_bench --offline >/dev/null

BIN="target/release/examples/decode_bench"
[ -x "$BIN" ] || { echo "decode_bench missing after build" >&2; exit 1; }

OUT_DIR=".perf_runs/autoresearch"
mkdir -p "$OUT_DIR"

# Median / p25 / count of a whitespace-separated number stream on stdin.
pooled_stats() {
  python3 -c '
import sys
v = sorted(float(x) for x in sys.stdin.read().split() if x)
if not v:
    sys.exit(1)
n = len(v)
med = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
print(f"{med:.3f} {v[n // 4]:.3f} {n}")
'
}

# One measured run: appends its step times to the pooled file and its token ids
# to the trajectory file.
run_once() {
  local mode="$1" i="$2"
  local log="$OUT_DIR/${mode}-${i}.log"
  env \
    LOGAN_EXPERT_NOCACHE=1 \
    BENCH_TEMP="$TEMP" \
    BENCH_TOP_P=1.0 \
    BENCH_TOP_K=0 \
    BENCH_SEED="$SEED" \
    "$BIN" "$MODEL_DIR" "$TOKENS" "$mode" "$PROMPT" \
    >"$log" 2>"$OUT_DIR/${mode}-${i}.err"

  # A missing metric line is a harness failure, not a slow run.
  grep -q '^BENCH step_ms=' "$log" || {
    echo "run $i of arm '$mode' emitted no step metrics; see $log and $OUT_DIR/${mode}-${i}.err" >&2
    exit 1
  }
  sed -n 's/^BENCH step_ms=//p' "$log" | tr ',' '\n' >>"$OUT_DIR/steps-${mode}.txt"
  sed -n 's/^BENCH ids=//p' "$log" >>"$OUT_DIR/ids-${mode}.txt"
  echo "  $mode run $i: $(sed -n 's/^BENCH decode_tok_s=//p' "$log") tok/s"
}

# Pooled median/p25 for one arm, plus a trajectory-stability assertion.
report_arm() {
  local mode="$1"
  local steps_file="$OUT_DIR/steps-${mode}.txt"
  local ids_file="$OUT_DIR/ids-${mode}.txt"

  local stats median p25 count
  stats="$(pooled_stats <"$steps_file")"
  median="$(printf '%s' "$stats" | cut -d' ' -f1)"
  p25="$(printf '%s' "$stats" | cut -d' ' -f2)"
  count="$(printf '%s' "$stats" | cut -d' ' -f3)"

  # More than one distinct id line means the repeats disagreed; that is a
  # numerics change and must not be averaged over silently.
  local distinct
  distinct="$(sort -u "$ids_file" | wc -l | tr -d ' ')"
  if [ "$distinct" != "1" ]; then
    echo "arm '$mode' produced $distinct distinct trajectories across $REPEATS repeats" >&2
    exit 1
  fi
  local sha tok_s
  sha="$(head -n 1 "$ids_file" | shasum -a 256 | cut -d' ' -f1)"
  tok_s="$(python3 -c "print(f'{1000.0/float(\"$median\"):.4f}')")"

  echo "RESULT $mode median_ms=$median p25_ms=$p25 pooled_steps=$count tok_s=$tok_s trajectory_sha=$sha"
  printf '%s %s %s\n' "$mode" "$tok_s" "$median" >>"$OUT_DIR/summary.tsv"
}

rm -f "$OUT_DIR"/steps-*.txt "$OUT_DIR"/ids-*.txt
: >"$OUT_DIR/steps-greedy.txt"
: >"$OUT_DIR/ids-greedy.txt"
: >"$OUT_DIR/steps-sample.txt"
: >"$OUT_DIR/ids-sample.txt"
: >"$OUT_DIR/summary.tsv"

case "$ARM" in
  greedy)
    echo "== arm: greedy (repeats=$REPEATS tokens=$TOKENS) =="
    for i in $(seq 1 "$REPEATS"); do run_once greedy "$i"; done
    ;;
  sample)
    echo "== arm: sample (temp=$TEMP seed=$SEED repeats=$REPEATS tokens=$TOKENS) =="
    for i in $(seq 1 "$REPEATS"); do run_once sample "$i"; done
    ;;
  both)
    echo "== arms interleaved greedy/sample x$REPEATS (tokens=$TOKENS temp=$TEMP) =="
    for i in $(seq 1 "$REPEATS"); do
      run_once greedy "$i"
      run_once sample "$i"
    done
    ;;
esac

[ -s "$OUT_DIR/steps-greedy.txt" ] && report_arm greedy
[ -s "$OUT_DIR/steps-sample.txt" ] && report_arm sample

# ---------------------------------------------------------------------------
# Emit metrics. Primary is the pooled median decode rate across both arms.
# ---------------------------------------------------------------------------
G_TOK="$(awk '$1=="greedy"{print $2}' "$OUT_DIR/summary.tsv")"
S_TOK="$(awk '$1=="sample"{print $2}' "$OUT_DIR/summary.tsv")"

if [ -n "$G_TOK" ] && [ -n "$S_TOK" ]; then
  PRIMARY="$(python3 -c "print(f'{(float(\"$G_TOK\")+float(\"$S_TOK\"))/2:.4f}')")"
elif [ -n "$G_TOK" ]; then
  PRIMARY="$G_TOK"
else
  PRIMARY="$S_TOK"
fi

printf 'METRIC tok_per_sec=%s\n' "$PRIMARY"
[ -n "$G_TOK" ] && printf 'METRIC greedy_tok_per_sec=%s\n' "$G_TOK"
[ -n "$S_TOK" ] && printf 'METRIC sample_tok_per_sec=%s\n' "$S_TOK"
printf 'METRIC decode_tokens=%s\n' "$TOKENS"
printf 'METRIC repeats=%s\n' "$REPEATS"

# How far the sampled trajectory drifts from the greedy one in rate. A large gap
# means the two arms exercise different expert routes.
if [ -n "$G_TOK" ] && [ -n "$S_TOK" ]; then
  printf 'METRIC arm_rate_gap=%s\n' \
    "$(python3 -c "print(f'{abs(float(\"$G_TOK\")-float(\"$S_TOK\")):.4f}')")"
fi

[ -s "$OUT_DIR/ids-greedy.txt" ] && printf 'METRIC greedy_trajectory_sha=%s\n' \
  "$(head -n 1 "$OUT_DIR/ids-greedy.txt" | shasum -a 256 | cut -d' ' -f1)"
[ -s "$OUT_DIR/ids-sample.txt" ] && printf 'METRIC sample_trajectory_sha=%s\n' \
  "$(head -n 1 "$OUT_DIR/ids-sample.txt" | shasum -a 256 | cut -d' ' -f1)"

exit 0
