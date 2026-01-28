#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-pytorch}"
RUN_CMD="${RUN_CMD:-python train_main.py}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/lufnet_baseline}"
USE_CPU="${USE_CPU:-1}"
RTOL="${RTOL:-0.0}"
ATOL="${ATOL:-0.0}"

mkdir -p "$OUTPUT_DIR"

if [[ "$USE_CPU" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

echo "Running baseline..."
conda run -n "$ENV_NAME" bash -lc "$RUN_CMD" | tee "$OUTPUT_DIR/run1.log"

echo "Running refactor..."
conda run -n "$ENV_NAME" bash -lc "$RUN_CMD" | tee "$OUTPUT_DIR/run2.log"

echo "Comparing logs..."
conda run -n "$ENV_NAME" python compare_runs.py \
  --log-a "$OUTPUT_DIR/run1.log" \
  --log-b "$OUTPUT_DIR/run2.log"

if [[ -n "${CKPT_A:-}" && -n "${CKPT_B:-}" ]]; then
  echo "Comparing checkpoints..."
  conda run -n "$ENV_NAME" python compare_runs.py \
    --ckpt-a "$CKPT_A" \
    --ckpt-b "$CKPT_B" \
    --rtol "$RTOL" \
    --atol "$ATOL"
fi
