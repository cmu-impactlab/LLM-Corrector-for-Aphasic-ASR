# Qwen2.5 Fine‑Tuning and Inference Guide (Unsloth + TRL)

This guide explains how to set up the environment, prepare data, train LoRA adapters for `unsloth/Qwen2.5-14B`, generate outputs using the trained adapters, and evaluate WER. It’s Markdown (no LaTeX).

## Overview
- Base model: `unsloth/Qwen2.5-14B`
- Training: Supervised fine‑tuning (SFT) via TRL, LoRA/QLoRA from Unsloth
- Multi‑GPU: Recommended with `accelerate` (DDP)
- Key scripts:
  - `train_qwen.py` — main CLI for training
  - `scripts/generate_asr_response.py` — run inference with a trained adapter
  - `wer-score.py` — compute WER for ASR/random/exhaustive against GT
  - `scripts/bootstrap_venv.sh` — quick local venv bootstrap

Outputs are written under `outputs/qwen2p5-14b-<run_name>`.

## Prerequisites
- Python 3.10+
- NVIDIA GPU(s) with a recent CUDA driver
- Disk space for checkpoints and optional merged weights

### Environment setup
Option A (scripted venv):
- `bash scripts/bootstrap_venv.sh`
- `source .venv/bin/activate`

Option B (manual):
- Create and activate venv: `python3 -m venv .venv && source .venv/bin/activate`
- Install core deps:
  - `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`
  - `pip install unsloth transformers==4.55.4 trl==0.22.2 peft bitsandbytes datasets accelerate sentencepiece protobuf huggingface_hub tensorboard`
- For evaluation: `pip install jiwer`

### Accelerate config
- A 2‑GPU config is provided in `accelerate_config.yaml`:
  - `distributed_type: MULTI_GPU`
  - `num_processes: 2`
  - `mixed_precision: bf16`

Run with: `accelerate launch --config_file accelerate_config.yaml train_qwen.py ...`

## Datasets
`train_qwen.py` supports:
1) Pre‑formatted JSONL with a `text` field (each example already a single prompt+response string ending with EOS).
2) JSONL with `instruction`, `input`, `output` — automatically formatted into an Alpaca‑style `text` with EOS appended.

Provided examples (see `dataset/`):
- `aprocsa1944a-random.jsonl`
- `aprocsa1944a-exhaustive.jsonl`

If your JSONL has only `text`, the script ensures it ends with the tokenizer EOS token; otherwise it raises if required columns are missing.

## Training
Common launch (2 GPUs, bf16):

```
accelerate launch --config_file accelerate_config.yaml \
  train_qwen.py \
  --dataset dataset/aprocsa1944a-random.jsonl \
  --run_name random \
  --no_4bit --bf16 \
  --max_steps 3000
```

Notes:
- `--no_4bit` disables 4‑bit quantized loading. Omit it to try QLoRA (4‑bit) if your environment supports bitsandbytes correctly.
- For two datasets back‑to‑back, see `run_two.sh` or the timed scripts in `scripts/`.

### Key CLI arguments (train_qwen.py)
- `--dataset` (str, required): Path to JSONL.
- `--run_name` (str, required): Tag to name outputs, e.g. `random`.
- `--model_name` (str, default `unsloth/Qwen2.5-14B`): Base model.
- `--output_dir` (str, default `outputs`): Base dir for artifacts.
- `--max_seq_length` (int, default 4096): Context length for tokenizer/model.
- `--per_device_train_batch_size` (int, default 2): Microbatch per GPU.
- `--gradient_accumulation_steps` (int, default 4): Accumulation steps.
- `--learning_rate` (float, default 2e-4)
- `--weight_decay` (float, default 0.01)
- `--warmup_steps` (int, default 50)
- `--max_steps` (int, default 2000): Total steps; set 0 to use epochs.
- `--num_train_epochs` (float, default 0.0): Use if not stepping.
- `--logging_steps` (int, default 10)
- `--save_steps` (int, default 200)
- `--save_total_limit` (int, default 3)
- `--dataloader_num_workers` (int, default 4)
- `--seed` (int, default 3407)
- `--bf16` (flag): Enable bfloat16 when available.
- `--merge_adapter` (flag): Try saving a merged full model to `-merged`.
- `--packing` (flag): Enable TRL sequence packing.
- `--no_4bit` (flag): Disable 4‑bit loading.
- `--report_to` (`none|tensorboard|wandb`, default `tensorboard`)
- `--device_map` (str|JSON, default None): For single‑process sharding. Leave empty when using DDP via accelerate.

Outputs per run:
- Adapters and tokenizer in `outputs/qwen2p5-14b-<run_name>/`
- Logs in `<out>/logs`
- `run_summary.json`, `log_history.json`
- If `--merge_adapter`: merged model under `<out>-merged/`

### Multi‑GPU vs Single‑GPU
- Multi‑GPU via `accelerate launch` (recommended for training).
- Single‑GPU fallback: `python3 train_qwen.py ...` works but will be slower and you may need to reduce batch size.

## Inference (generate transcripts)
Use the fine‑tuned LoRA adapter with the provided prompt template and an ASR text file.

```
python3 scripts/generate_asr_response.py \
  --input "aprocsa1944a Azure.txt" \
  --prompt_module format.py \
  --adapter_dir outputs/qwen2p5-14b-random \
  --base_model unsloth/Qwen2.5-14B \
  --max_seq_length 4096 \
  --max_new_tokens 1800
```

This writes a file like:
`outputs/qwen2p5-14b-random/aprocsa1944a Azure_response.txt`

Flags:
- `--do_sample` to enable sampling (default is greedy)
- `--temperature`, `--top_p` to adjust sampling behavior

## Evaluation (WER)
Install: `pip install jiwer`

Compute WER vs ground truth (uses jiwer with punctuation removal and standardization on both sides):

```
python3 wer-score.py \
  --gt "aprocsa1944a GT.txt" \
  --asr "aprocsa1944a Azure.txt"
```

By default it tries to find model outputs under:
- `outputs/qwen2p5-14b-random/..._response.txt`
- `outputs/qwen2p5-14b-exhaustive/..._response.txt`

Override paths with `--random` and `--exhaustive` if needed. Results are displayed and saved to `outputs/wer_scores.json`.

## Tips & Troubleshooting
- Bitsandbytes/4‑bit issues: try `--no_4bit` (the default commands use it) or ensure a compatible CUDA/toolchain.
- CUDA OOM: reduce `--per_device_train_batch_size` or increase `--gradient_accumulation_steps`.
- Long context: keep `--max_seq_length` aligned with memory budget; consider `--packing` for short examples.
- DDP hangs: re‑run `accelerate config` or use the provided `accelerate_config.yaml`.
- Hugging Face downloads: set `HF_HUB_ENABLE_HF_TRANSFER=1` for faster throughput.

## Quick recipes
- Train both datasets (2 GPUs): `./run_two.sh`

## Project layout
- `train_qwen.py` — SFT LoRA training (Unsloth + TRL)
- `scripts/` — helpers for environment, generation, timed runs
- `dataset/` — JSONL training data
- `format.py` — prompt template for inference
- `wer-score.py` — WER evaluation
- `accelerate_config.yaml` — 2‑GPU bf16 config
- `outputs/` — training artifacts and generated results

