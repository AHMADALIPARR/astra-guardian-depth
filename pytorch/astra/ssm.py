# SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
# Copyright (C) 2026 SnapKitty Collective
"""Selective state-space model baseline in the Mamba architecture (Gu & Dao 2023).

PyTorch-native and CPU-executable. This is NOT the official `mamba-ssm`
package (which needs CUDA to build); it implements the same selective SSM
mechanism: input-dependent delta/B/C, zero-order-hold discretization, and a
recurrent scan. Used to benchmark the looped transformer against a standard
SSM at matched parameter count on the iterated-rotation task family.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import RMSNorm


class SelectiveSSM(nn.Module):
    """Mamba-style selective SSM: h' = A h + B x with input-dependent coeffs."""

    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int = 8):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        # Structured A: S4D-style, strictly negative -> stable.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # A = -exp(A_log)
        self.D = nn.Parameter(torch.ones(d_inner))  # skip connection

        # Selectivity: dt, B, C all depend on the input.
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        # Initialize dt in (0.001, 0.1) via inverse-softplus (Mamba convention).
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_inner) -> (B, T, d_inner)."""
        b, t, _ = x.shape
        dt_r, B_, C_ = torch.split(
            self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_r))  # (B, T, d_inner)
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative

        # Zero-order-hold discretization.
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # dB = ((dA - 1) / A) * B  (elementwise; A < 0 so no singularity)
        dB = (dA - 1) / A.unsqueeze(0).unsqueeze(0) * B_.unsqueeze(2)

        h = torch.zeros(b, self.d_inner, self.d_state,
                        device=x.device, dtype=x.dtype)
        ys = []
        for i in range(t):
            h = dA[:, i] * h + dB[:, i] * x[:, i, :, None]
            ys.append((h * C_[:, i, None, :]).sum(dim=-1))
        y = torch.stack(ys, dim=1)
        return y + self.D * x


class MambaBlock(nn.Module):
    """One Mamba block: norm -> in-proj -> causal dw-conv -> SSM -> gate -> out."""

    def __init__(self, d_model: int, d_inner: int = None,
                 d_state: int = 16, dt_rank: int = 8):
        super().__init__()
        d_inner = d_inner or 2 * d_model
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=4, padding=3,
                              groups=d_inner, bias=True)
        self.ssm = SelectiveSSM(d_inner, d_state, dt_rank)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        xz = self.in_proj(h)
        x_, z = xz.chunk(2, dim=-1)
        # Causal depthwise conv (trim the right padding).
        x_c = self.conv(x_.transpose(1, 2))[:, :, : x.shape[1]].transpose(1, 2)
        y = self.ssm(F.silu(x_c))
        return x + self.out_proj(y * F.silu(z))


class MambaBaseline(nn.Module):
    """Standard (non-looped) SSM baseline: embed -> N MambaBlocks -> coda."""

    def __init__(self, vocab_size: int, d_model: int = 64, n_blocks: int = 3,
                 d_state: int = 16, dt_rank: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(
            *[MambaBlock(d_model, d_state=d_state, dt_rank=dt_rank)
              for _ in range(n_blocks)]
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.blocks(self.embed(ids))))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
