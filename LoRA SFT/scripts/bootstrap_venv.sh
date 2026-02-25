#!/usr/bin/env bash
set -euo pipefail

PY=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}

echo "[bootstrap] Creating venv at ${VENV_DIR}"
${PY} -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "[bootstrap] Upgrading pip/setuptools/wheel"
pip install -U pip setuptools wheel

echo "[bootstrap] Installing PyTorch + CUDA (using cu121 index). Adjust if needed."
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision torchaudio

echo "[bootstrap] Installing training deps (Unsloth, TRL, etc.)"
pip install unsloth transformers==4.55.4 trl==0.22.2 peft bitsandbytes \
  datasets accelerate sentencepiece protobuf huggingface_hub tensorboard

echo "[bootstrap] Verifying CUDA availability"
python - << 'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
PY

echo "[bootstrap] Done. Activate with: source ${VENV_DIR}/bin/activate"

