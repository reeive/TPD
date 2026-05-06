#!/usr/bin/env bash
# Demo: ViT-B/16 + VPT TTA (run_tta_vpt.py)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export DATA_ROOT="${DATA_ROOT:-./data}"
export MODEL_ROOT="${MODEL_ROOT:-./checkpoints}"

cd "$VIT_ROOT"
export PYTHONPATH="$VIT_ROOT:${PYTHONPATH:-}"

python run_tta_vpt.py \
  --data-dir "$DATA_ROOT/imagenet-r" \
  --model_root "$MODEL_ROOT" \
  --dataset imagenet-r \
  --backbone sup_vitb16_224 \
  --init_head \
  --engine tpd \
  --lr 5e-4 \
  --tta_steps 3 \
  --state_dim 16 \
  --window_size 10 \
  --basis_window 16 \
  --selection_p 0.1 \
  "${@}"
