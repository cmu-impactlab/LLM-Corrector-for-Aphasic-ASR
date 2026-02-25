#!/usr/bin/env bash
set -euo pipefail

RANDOM_DATASET="dataset/aprocsa1944a-random.jsonl"
EXHAUSTIVE_DATASET="dataset/aprocsa1944a-exhaustive.jsonl"

COMMON_ARGS=(
  --model_name unsloth/Qwen2.5-14B
  --output_dir outputs
  --max_seq_length 4096
  --per_device_train_batch_size 2
  --gradient_accumulation_steps 8
  --learning_rate 2e-4
  --weight_decay 0.01
  --warmup_steps 100
  --max_steps 3000
  --logging_steps 10
  --save_steps 500
  --bf16
)

run_job() {
  local dataset=$1
  local name=$2
  if command -v accelerate >/dev/null 2>&1; then
    accelerate launch --num_processes 2 --mixed_precision bf16 \
      train_qwen.py --dataset "$dataset" --run_name "$name" "${COMMON_ARGS[@]}"
  elif command -v torchrun >/dev/null 2>&1; then
    torchrun --nproc_per_node=2 train_qwen.py --dataset "$dataset" --run_name "$name" "${COMMON_ARGS[@]}"
  else
    echo "[info] Running single-process fallback (no accelerate/torchrun found)"
    python3 train_qwen.py --dataset "$dataset" --run_name "$name" "${COMMON_ARGS[@]}"
  fi
}

run_job "$RANDOM_DATASET" random --no_4bit --bf16
run_job "$EXHAUSTIVE_DATASET" exhaustive --no_4bit --bf16

echo "All runs complete. Outputs in ./outputs/qwen2p5-14b-{random,exhaustive}"
