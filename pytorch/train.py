# SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
# Copyright (C) 2026 SnapKitty Collective
"""Depth-beats-parameters demo: the loop learns an iterated function.

Task: bit rotation. The shared block learns ONE job -- rotate right by one --
and looping it k times computes rotate-by-k. Per-pass supervision trains each
iteration directly; evaluation checks every pass, including passes BEYOND the
training loop count (extrapolation of the learned iterative rule).

Same parameters throughout. Depth is compute, not weights.

Usage: python train.py [--steps 2000] [--r-train 4]
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from astra import LoopedTransformer
from astra.tasks import RotateTask, rotate_right
from astra import monitor as mon


def pass_accuracy(model, task, r_eval, n=2000, seed=0):
    """Per-pass accuracy vs rotate-by-k target, for k=1..r_eval.

    Returns {k: (exact_match_acc, bit_acc)}.
    """
    g = torch.Generator().manual_seed(seed)
    model.eval()
    stats = {}
    with torch.no_grad():
        for _ in range(n // 256):
            x, bits = task.sample(256, generator=g)
            _, trace = model(x, r=r_eval, record=True)
            for k in range(1, r_eval + 1):
                # trace.readouts[k-1]: (batch, positions) argmax
                readout = torch.tensor(trace.readouts[k - 1])[:, : task.n_bits]
                target = rotate_right(bits, k)
                exact = (readout == target).all(dim=1)
                s = stats.setdefault(k, [0, 0, 0])
                s[0] += exact.sum().item()
                s[1] += (readout == target).sum().item()
                s[2] += target.numel()
    out = {}
    for k, (exact_ok, bits_ok, bits_total) in stats.items():
        n_samples = bits_total // task.n_bits
        out[k] = (exact_ok / n_samples, bits_ok / bits_total)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--r-train", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    task = RotateTask(n_bits=8)
    model = LoopedTransformer(
        vocab_size=task.vocab_size, dim=64, n_heads=4, n_pre=1, inject="residual"
    )
    print(f"parameters: {model.count_parameters()} "
          f"(blocks: {model.looped_block_instances()} = 1 prelude + 1 shared loop)")

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(args.seed + 1)
    model.train()
    t0 = time.time()
    import os
    os.makedirs("checkpoints", exist_ok=True)
    for step in range(1, args.steps + 1):
        x, bits = task.sample(512, generator=g)
        _, trace = model(x, r=args.r_train, record=True)
        loss = 0.0
        for k, pl in enumerate(trace.pass_logits, start=1):
            pk = pl[:, : task.n_bits, :]
            target = rotate_right(bits, k)
            loss = loss + F.cross_entropy(
                pk.reshape(-1, task.vocab_size), target.reshape(-1)) / args.r_train
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  step {step:5d}  loss {loss.item():.4f}", flush=True)
            torch.save(model.state_dict(), f"checkpoints/ckpt_{step}.pt")
    torch.save(model.state_dict(), "checkpoints/final.pt")
    print(f"trained {args.steps} steps in {time.time()-t0:.1f}s (CPU)")

    print("\neval: pass-k accuracy (same weights, r_eval=8)")
    print("pass k: did the loop correctly apply k rotations?")
    print(f"{'pass':>5}  {'exact':>6}  {'bit':>6}")
    accs = pass_accuracy(model, task, r_eval=8)
    for k, (exact, bit) in accs.items():
        tag = " (beyond training)" if k > args.r_train else ""
        print(f"{k:>5}  {exact:>6.3f}  {bit:>6.3f}{tag}")

    # audit one example's silent loop
    model.eval()
    x, bits = task.sample(1, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        _, trace = model(x, r=8, record=True)
    print("\n" + mon.render_audit(mon.audit(trace)))
    print(f"input      : {bits[0].tolist()}")
    for k in (1, 4, 8):
        print(f"pass {k} readout: {trace.readouts[k-1][0][:task.n_bits]}  "
              f"target: {rotate_right(bits, k)[0].tolist()}")


if __name__ == "__main__":
    main()
