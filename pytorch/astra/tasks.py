# SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
# Copyright (C) 2026 SnapKitty Collective
"""Synthetic algorithmic tasks for probing recurrent depth.

These tasks need multi-step computation, so they separate models by *effective
depth* rather than parameter count: the same shared block looped more times
should do better. Token vocab: 0 -> "0", 1 -> "1", 2 -> <sep>.
"""

import torch


class ParityTask:
    """XOR of all bits. Input: [b1..bn, SEP], target: parity bit.

    Note: genuinely hard to optimize from scratch through a shared loop
    (training stalls at chance). Kept as a task definition; the demo in
    train.py uses RefineTask, which trains reliably and shows the same
    depth-vs-parameters effect.
    """

    SEP = 2

    def __init__(self, n_bits: int = 8):
        self.n_bits = n_bits
        self.vocab_size = 3

    def sample(self, n: int, generator=None):
        bits = torch.randint(0, 2, (n, self.n_bits), generator=generator)
        target = (bits.sum(dim=1) % 2).long()
        sep = torch.full((n, 1), self.SEP, dtype=torch.long)
        inputs = torch.cat([bits, sep], dim=1)
        return inputs, target


class RefineTask:
    """Iterative denoising: recover clean bits from a corrupted copy.

    WARNING: with i.i.d. random clean bits this task is IMPOSSIBLE -- a flipped
    bit is statistically indistinguishable from a clean one, so the Bayes-optimal
    predictor is the identity (accuracy = 1 - flips/n_bits). A model that learns
    to copy its input here is optimal, not broken. Use ChainRefineTask for a
    task where iterative refinement actually helps.
    """

    SEP = 2

    def __init__(self, n_bits: int = 16, flips: int = 4):
        self.n_bits = n_bits
        self.flips = flips
        self.vocab_size = 3

    def sample(self, n: int, generator=None):
        clean = torch.randint(0, 2, (n, self.n_bits), generator=generator)
        noisy = clean.clone()
        # distinct flip positions per row (no replacement)
        flip_pos = torch.rand(n, self.n_bits, generator=generator).argsort(dim=1)[:, : self.flips]
        noisy.scatter_(1, flip_pos, 1 - noisy.gather(1, flip_pos))
        sep = torch.full((n, 1), self.SEP, dtype=torch.long)
        inputs = torch.cat([noisy, sep], dim=1)
        return inputs, clean


class ChainRefineTask:
    """Denoise a corrupted Markov chain -- a genuinely iterative task.

    Clean bits form a random walk: each bit equals the previous one with
    probability `stay_p`. Random flips corrupt `noise_flips` positions.
    Recovering the chain is like belief propagation: each recurrent pass can
    propagate information one step further along the chain, so MORE passes
    (same weights) should give better reconstructions -- depth as compute.

    Unlike i.i.d. bits, the chain has redundancy (adjacent agreement), so the
    task is solvable and iteration helps.

    Measured (2026-10-02, dim 64, 1 prelude layer, r_train=3, 2000 steps,
    deep supervision): bit-acc 0.824--0.826 flat across r=1..8, vs 0.8125 for
    the identity baseline. The loop learns a weak denoiser but extra passes
    don't help -- a negative result, kept so the ledger is honest. The
    depth-beats-parameters demo in train.py uses RotateTask instead.
    """

    SEP = 2

    def __init__(self, n_bits: int = 16, stay_p: float = 0.85, noise_flips: int = 3):
        self.n_bits = n_bits
        self.stay_p = stay_p
        self.noise_flips = noise_flips
        self.vocab_size = 3

    def sample(self, n: int, generator=None):
        clean = torch.empty(n, self.n_bits, dtype=torch.long)
        clean[:, 0] = torch.randint(0, 2, (n,), generator=generator)
        stay = torch.rand(n, self.n_bits - 1, generator=generator) < self.stay_p
        for i in range(1, self.n_bits):
            prev = clean[:, i - 1]
            clean[:, i] = torch.where(stay[:, i - 1], prev, 1 - prev)
        noisy = clean.clone()
        flip_pos = torch.rand(n, self.n_bits, generator=generator).argsort(dim=1)[:, : self.noise_flips]
        noisy.scatter_(1, flip_pos, 1 - noisy.gather(1, flip_pos))
        sep = torch.full((n, 1), self.SEP, dtype=torch.long)
        inputs = torch.cat([noisy, sep], dim=1)
        return inputs, clean


def rotate_left(bits: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate bit-vectors left by k positions (kept for the unit test)."""
    k = k % bits.shape[1]
    return torch.cat([bits[:, k:], bits[:, :k]], dim=1)


def rotate_right(bits: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate bit-vectors right by k positions.

    Causal-compatible: position i takes the value previously at i-1, so a
    causally-masked transformer can implement one rotation per pass by
    attending one step back.
    """
    k = k % bits.shape[1]
    if k == 0:
        return bits
    return torch.cat([bits[:, -k:], bits[:, :-k]], dim=1)


class RotateTask:
    """Learn an iterated function: pass k must output rotate-right-by-k(x).

    The shared block has one job -- rotate right by one -- and looping it k
    times computes rotate-by-k. Rotation is to the RIGHT so a causally-masked
    transformer can implement it (position i attends to i-1). Per-pass
    supervision trains each iteration directly.
    Input: [x (n bits), SEP]; pass-k target: rotate_right(x, k).
    """

    SEP = 2

    def __init__(self, n_bits: int = 8):
        self.n_bits = n_bits
        self.vocab_size = 3

    def sample(self, n: int, generator=None):
        x = torch.randint(0, 2, (n, self.n_bits), generator=generator)
        sep = torch.full((n, 1), self.SEP, dtype=torch.long)
        inputs = torch.cat([x, sep], dim=1)
        return inputs, x  # targets are derived per-pass via rotate_right


class ModAddTask:
    """(a + b) mod p. Input: [a_toks.., SEP, b_toks.., SEP], target: digit class."""

    SEP = 2

    def __init__(self, p: int = 7):
        self.p = p
        self.vocab_size = 3  # digits are binarized below; kept tiny on purpose

    def sample(self, n: int, generator=None):
        a = torch.randint(0, self.p, (n,), generator=generator)
        b = torch.randint(0, self.p, (n,), generator=generator)
        target = ((a + b) % self.p).long()

        def bits(x):
            # 3-bit binary encoding of values 0..6
            return torch.stack(
                [(x >> 2) & 1, (x >> 1) & 1, x & 1], dim=1
            ).long()

        sep = torch.full((n, 1), self.SEP, dtype=torch.long)
        inputs = torch.cat([bits(a), sep, bits(b), sep], dim=1)
        return inputs, target
