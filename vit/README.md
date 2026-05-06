# ViT + VPT demo runner

> **Demo subset.** This folder is the minimal **ViT-B/16 + visual prompt
> tuning (VPT)** entry point bundled for peer review. Other baselines (E2VPT,
> VFPT, BPT) and CLIP pipelines are **not** wired as separate runners here; the
> full codebase will be published upon acceptance. The `src/` tree is kept
> intact so imports remain valid.

## Run

From the **repository root** (`tpd-release/`):

```bash
export PYTHONPATH="${PWD}/tpd:${PWD}/vit:${PYTHONPATH}"
export DATA_ROOT=./data MODEL_ROOT=./checkpoints
bash vit/scripts/run_tta_vpt.sh
```

Or run `python vit/run_tta_vpt.py -h` and pass `--data-dir`, `--model_root`,
`--backbone sup_vitb16_224`, and `--engine tpd` as needed.

## Data and checkpoints

- Dataset: ImageFolder layout, e.g. `./data/imagenet-r/<class_name>/*.JPEG`.
- Weights: place backbone `.npz` files under `./checkpoints` (names in
  `src/tta/_common.py` / `BACKBONE_REGISTRY`).

See the top-level `README.md` for installation and a full quickstart.
