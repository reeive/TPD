#!/usr/bin/env python3
"""Collect ViT+VPT + TPD trajectory on ImageNet-R for Fig 1(a).

Uses model code from ``vit/`` (``run_tta_vpt``) and the standalone ``tpd`` package.

Default evaluation matches full IN-R runs: **cumulative top-1 accuracy**
``acc = 100 * total_correct / total_seen`` (same as ``vit/run_tta_vpt.py``), so the
last ``acc`` after 30k samples is comparable to the reported ~49.37% full pass.

Example::

    conda run -n vpt --no-capture-output python tpd/example.py --gpu 0 \\
        --batch-size 64 --max-samples 0 \\
        --out ./output/motivation_vpt_tpd_real.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_TPD_PKG = Path(__file__).resolve().parent
if str(_TPD_PKG) not in sys.path:
    sys.path.insert(0, str(_TPD_PKG))

import torch

from tpd import TPD

from vit_fig1a_common import (
    build_namespace,
    confident_entropy_loss,
    load_vpt_imagenet_r,
    running_mean_accuracy_pct,
    save_records,
    sliding_window_acc,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="ViT+VPT + TPD → motivation JSON")
    parser.add_argument("--data-dir", default="./data/imagenet-r")
    parser.add_argument("--model-root", default="./checkpoints")
    parser.add_argument(
        "--out",
        default=str(_TPD_PKG.parent / "motivation_vpt_tpd_real.json"),
        help="Output JSON (plot_motivation_tpd_intro.py reads motivation_vpt_tpd_real.json)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max images to process (0 = full ImageNet-R, typically 30k).",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Align with vit/run_tta_vpt online TTA (default 64).",
    )
    parser.add_argument(
        "--acc-mode",
        choices=("running", "sliding"),
        default="running",
        help="running: cumulative mean acc (same as full eval); sliding: windowed acc.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Sliding window size when --acc-mode sliding.",
    )
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--tta-steps", type=int, default=3)
    parser.add_argument("--selection-p", type=float, default=0.1)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=10, help="TPD Koopman window W")
    parser.add_argument("--basis-window", type=int, default=16, help="SVD basis window B")
    parser.add_argument("--anchor-lambda", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    ns = build_namespace(
        data_dir=args.data_dir,
        model_root=args.model_root,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    limit = args.max_samples if args.max_samples > 0 else None
    model, loader = load_vpt_imagenet_r(ns, device)
    model.eval()

    controller = TPD(
        model,
        lr=args.lr,
        r=args.state_dim,
        W=args.window_size,
        B=args.basis_window,
        K=args.tta_steps,
        anchor_lambda=args.anchor_lambda,
        device=device,
    )
    init_vec = controller.prompt.vector().detach().clone()

    def loss_fn(logits):
        return confident_entropy_loss(logits, args.selection_p)

    correct_history: list[int] = []
    records: list[dict] = []
    total_correct = 0
    total_seen = 0
    t0 = time.time()

    for images, labels in loader:
        if limit is not None and total_seen >= limit:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if limit is not None:
            rem = limit - total_seen
            if rem <= 0:
                break
            if rem < labels.shape[0]:
                images = images[:rem]
                labels = labels[:rem]

        preds, info = controller.adapt_and_predict(images, loss_fn=loss_fn)

        if torch.cuda.is_available() and (len(records) + 1) % 128 == 0:
            torch.cuda.empty_cache()

        batch_ok = (preds == labels).int()
        for j in range(labels.shape[0]):
            correct_history.append(int(batch_ok[j].item()))

        batch_correct = int((preds == labels).sum().item())
        total_correct += batch_correct
        total_seen += labels.numel()

        if args.acc_mode == "running":
            acc = running_mean_accuracy_pct(total_correct, total_seen)
        else:
            acc = sliding_window_acc(correct_history, args.window)

        drift = float(
            (controller.prompt.vector() - init_vec.to(controller.prompt.vector().device)).norm().item()
        )

        step = total_seen
        if info is not None:
            rec = {
                "step": step,
                "acc": round(float(acc), 4),
                "rho": round(float(info.rho), 4),
                "drift": round(drift, 4),
                "F1": round(float(info.f1), 6),
                "F2": round(float(info.f2), 6),
                "F3": round(float(info.f3), 6),
                "svd_spectrum": [round(float(x), 6) for x in info.svd_spectrum],
            }
        else:
            rec = {
                "step": step,
                "acc": round(float(acc), 4),
                "rho": 0.0,
                "drift": round(drift, 4),
                "F1": 0.0,
                "F2": 0.0,
                "F3": 0.0,
                "svd_spectrum": [0.0] * 20,
            }
        records.append(rec)

        if len(records) <= 3 or len(records) % 50 == 0:
            print(
                f"[{step:5d}] acc={acc:.2f}% rho={rec['rho']:.4f} drift={rec['drift']:.4f}",
                flush=True,
            )

    elapsed = time.time() - t0
    final_acc = running_mean_accuracy_pct(total_correct, total_seen)
    print(
        f"Done {len(records)} batches ({total_seen} images) in {elapsed:.1f}s | "
        f"final cumulative acc={final_acc:.2f}% (same metric as run_tta_vpt.py)"
    )
    save_records(args.out, records)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
