# SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
# Copyright (C) 2026 SnapKitty Collective
"""Benchmark: looped transformer vs Mamba-architecture SSM.

Same task family (rotate-right-by-k, k=1..4), matched parameter budgets
(~107k looped vs ~102k Mamba), same training budget (2000 steps, matched
total training triples: the looped model sees 512*4 pass-targets/step, Mamba
sees 2048*1 targets/step). Each architecture trains the way it trains best:
the looped model with per-pass supervision (its native mode), Mamba with
k-token conditioning (its native mode).

The looped baseline numbers below are from the train.py run (per-pass eval);
this script trains Mamba and prints the side-by-side table.

Usage: python benchmark_mamba.py [--steps 2000]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from astra import MambaBaseline
from astra.tasks import rotate_right

SEP = 2
K_TOK = {1: 3, 2: 4, 3: 5, 4: 6}  # k -> token id
N_BITS = 8


def sample_batch(n, generator=None):
    x = torch.randint(0, 2, (n, N_BITS), generator=generator)
    k = torch.randint(1, 5, (n,), generator=generator)
    k_tok = torch.tensor([K_TOK[ki] for ki in k.tolist()]).unsqueeze(1)
    sep = torch.full((n, 1), SEP, dtype=torch.long)
    inputs = torch.cat([k_tok, sep, x], dim=1)  # (n, 10)
    targets = torch.stack([rotate_right(x[i : i + 1], int(k[i]))[0]
                           for i in range(n)])
    return inputs, targets


def evaluate(model, n=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    model.eval()
    out = {}
    with torch.no_grad():
        for k in (1, 2, 3, 4):
            exact_ok, bit_ok, bit_tot, n_done = 0, 0, 0, 0
            for _ in range(n // 256):
                x = torch.randint(0, 2, (256, N_BITS), generator=g)
                k_tok = torch.full((256, 1), K_TOK[k], dtype=torch.long)
                sep = torch.full((256, 1), SEP, dtype=torch.long)
                inputs = torch.cat([k_tok, sep, x], dim=1)
                logits = model(inputs)[:, 2 : 2 + N_BITS, :]
                pred = logits.argmax(dim=-1)
                target = rotate_right(x, k)
                exact_ok += (pred == target).all(dim=1).sum().item()
                bit_ok += (pred == target).sum().item()
                bit_tot += target.numel()
                n_done += x.shape[0]
            # n_done is the actual evaluated count (n // 256 batches);
            # dividing by n would understate exact-match.
            out[k] = (exact_ok / n_done, bit_ok / bit_tot)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = MambaBaseline(vocab_size=7, d_model=64, n_blocks=3,
                          d_state=16, dt_rank=8)
    print(f"mamba parameters: {model.count_parameters()}")

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(args.seed + 1)
    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        inputs, targets = sample_batch(2048, generator=g)
        logits = model(inputs)[:, 2 : 2 + N_BITS, :]
        loss = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  step {step:5d}  loss {loss.item():.4f}", flush=True)
    print(f"trained {args.steps} steps in {time.time()-t0:.1f}s (CPU)")

    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "mamba_baseline.pt")
    torch.save({"state_dict": model.state_dict(), "steps": args.steps,
                "seed": args.seed, "params": model.count_parameters()}, ckpt)
    print(f"saved checkpoint to {ckpt}")

    # Looped-transformer baseline (train.py, per-pass eval, same task family).
    looped = {1: (0.492, 0.937), 2: (0.262, 0.875),
              3: (0.131, 0.816), 4: (0.064, 0.750)}

    print("\nbenchmark: rotate-right-by-k accuracy (exact / bit)")
    print(f"{'k':>3}  {'looped':>14}  {'mamba':>14}")
    res = evaluate(model)
    for k in (1, 2, 3, 4):
        le, lb = looped[k]
        me, mb = res[k]
        print(f"{k:>3}  {le:.3f} / {lb:.3f}  {me:.3f} / {mb:.3f}")


if __name__ == "__main__":
    main()
