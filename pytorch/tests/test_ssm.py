# SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
# Copyright (C) 2026 SnapKitty Collective
"""Unit tests for the Mamba-architecture selective SSM baseline."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from astra.ssm import MambaBlock, MambaBaseline


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        check.failed += 1
check.failed = 0

torch.manual_seed(0)

# --- shape ------------------------------------------------------------------
m = MambaBlock(d_model=32, d_state=16)
x = torch.randn(4, 10, 32)
y = m(x)
check("block output shape", y.shape == (4, 10, 32))

seq = MambaBaseline(vocab_size=7, d_model=32, n_blocks=2, d_state=16)
logits = seq(torch.randint(0, 7, (4, 10)))
check("sequence model logits shape", logits.shape == (4, 10, 7))

# --- determinism --------------------------------------------------------------
m.eval()
with torch.no_grad():
    a = m(x)
    b = m(x)
check("deterministic forward", torch.equal(a, b))

# --- gradients reach the SSM parameters ---------------------------------------
m.train()
out = m(x).sum()
out.backward()
check("grad reaches A_log",
      m.ssm.A_log.grad is not None and m.ssm.A_log.grad.abs().sum() > 0)
check("grad reaches dt_proj", m.ssm.dt_proj.weight.grad is not None)
check("grad reaches in_proj", m.in_proj.weight.grad is not None)

# --- finiteness -----------------------------------------------------------------
check("output finite", torch.isfinite(y).all().item())
check("logits finite", torch.isfinite(logits).all().item())

# --- stable A parameterization ---------------------------------------------------
with torch.no_grad():
    A = -torch.exp(m.ssm.A_log)
check("A strictly negative (stable)", bool((A < 0).all()))

# --- selectivity: input-dependent B, C, dt change the output --------------------
m2 = MambaBlock(d_model=32, d_state=16)
m2.eval()
with torch.no_grad():
    x1 = torch.randn(2, 8, 32)
    x2 = x1.clone()
    x2[:, 3, :] += 5.0  # perturb one timestep
    y1, y2 = m2(x1), m2(x2)
check("input-dependent dynamics (perturbation propagates)",
      not torch.allclose(y1, y2, atol=1e-6))

print(f"\n{check.failed} failures")
sys.exit(1 if check.failed else 0)
