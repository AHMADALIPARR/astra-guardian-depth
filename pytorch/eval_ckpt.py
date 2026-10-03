# SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
# Copyright (C) 2026 SnapKitty Collective
"""Evaluate a saved checkpoint's per-pass rotation accuracy.

Usage: python eval_ckpt.py checkpoints/final.pt [--r-eval 8]
"""

import argparse
import sys

sys.path.insert(0, ".")

import torch

from astra import LoopedTransformer
from astra.tasks import RotateTask
from train import pass_accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--r-eval", type=int, default=8)
    args = ap.parse_args()

    task = RotateTask(n_bits=8)
    model = LoopedTransformer(
        vocab_size=task.vocab_size, dim=64, n_heads=4, n_pre=1, inject="residual"
    )
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    print(f"loaded {args.ckpt}; parameters: {model.count_parameters()}")
    accs = pass_accuracy(model, task, r_eval=args.r_eval)
    print(f"{'pass':>5}  {'exact':>6}  {'bit':>6}")
    for k, (exact, bit) in accs.items():
        print(f"{k:>5}  {exact:>6.3f}  {bit:>6.3f}")


if __name__ == "__main__":
    main()
