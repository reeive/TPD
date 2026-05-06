# TPD: Test-Time Prompt-Agnostic Decomposition



## Overview

This repository contains a demo consisting of the standalone `tpd` controller and a ViT-B/16 + VPT runner.
The complete codebase will be released upon acceptance.

## Install

```bash
pip install -r tpd/requirements.txt
export PYTHONPATH="${PWD}/tpd:${PWD}/vit:${PYTHONPATH}"
```

Dependencies follow the original project (PyTorch, timm, torchvision, PyYAML, etc.);
see `tpd/requirements.txt` for the pinned list.

## Quickstart (ViT-B/16 + VPT on ImageNet-R)

Place ImageNet-R under `./data/imagenet-r` (class subfolders) and ViT `.npz`
checkpoints under `./checkpoints` (see `vit/src/tta/_common.py` for expected
filenames), then:

```bash
export DATA_ROOT=./data MODEL_ROOT=./checkpoints
export PYTHONPATH="${PWD}/tpd:${PWD}/vit:${PYTHONPATH}"

python vit/run_tta_vpt.py \
  --data-dir "${DATA_ROOT}/imagenet-r" \
  --model_root "${MODEL_ROOT}" \
  --dataset imagenet-r \
  --backbone sup_vitb16_224 \
  --init_head \
  --engine tpd \
  --lr 5e-4 \
  --tta_steps 3 \
  --state_dim 16 \
  --window_size 10 \
  --basis_window 16 \
  --selection_p 0.1
```

Or use the helper script (defaults: `DATA_ROOT=./data`, `MODEL_ROOT=./checkpoints`):

```bash
bash vit/scripts/run_tta_vpt.sh
```
