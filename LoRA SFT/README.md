Fine-tune Qwen2.5-14B (Unsloth + TRL)

This workspace contains a CLI training script that fine-tunes `unsloth/Qwen2.5-14B` with LoRA on your JSONL dataset, plus generation and WER evaluation helpers.

Quick links:
- User Guide (installation, datasets, training, inference, WER): `docs/USER_GUIDE.md`

Files:
- `train_qwen.py` — CLI script for SFT fine-tuning with LoRA.
- `scripts/generate_asr_response.py` — generate transcript from ASR text with a trained adapter.
- `wer-score.py` — WER computation using jiwer.
- `run_two.sh` — helper to launch both runs (random, exhaustive) on 2 GPUs.
- `dataset/aprocsa1944a-random.jsonl` — provided dataset.
- `dataset/aprocsa1944a-exhaustive.jsonl` — provided dataset.

Dependencies (minimal):
- Python 3.10+
- CUDA GPU(s)
- `pip install unsloth trl transformers peft bitsandbytes datasets accelerate`
- For WER: `pip install jiwer`

Recommended launcher (2 GPUs, bf16):

  accelerate launch --config_file accelerate_config.yaml \
    train_qwen.py \
      --dataset dataset/aprocsa1944a-random.jsonl \
      --run_name random \
      --no_4bit --bf16 \
      --max_steps 3000

Run both datasets:

  chmod +x run_two.sh
  ./run_two.sh

Outputs:
- Adapters and tokenizer saved to `outputs/qwen2p5-14b-<run_name>`.
- Optional merged full model saved to `outputs/qwen2p5-14b-<run_name>-merged` when `--merge_adapter` is used.

Notes:
- If your dataset already has a `text` field, it is used directly. Otherwise the script formats from `instruction`, `input`, and `output` columns into Alpaca-style text.
- For full model merging, ensure sufficient VRAM and disk; otherwise omit `--merge_adapter` (recommended).
- The older `fine-tune.py` is a Colab-style notebook snippet and is not intended to be executed directly here.
