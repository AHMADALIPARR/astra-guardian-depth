# SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
# Copyright (C) 2026 SnapKitty Collective
"""Unit tests for the looped transformer. Run: python -m pytest tests/ -q
(or: python tests/test_looped.py for a dependency-free run)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from astra import LoopedTransformer, TransformerBlock
from astra.tasks import ParityTask, RefineTask, ChainRefineTask, RotateTask, rotate_left, rotate_right
from astra import monitor as mon


def make(vocab=3, **kw):
    torch.manual_seed(0)
    return LoopedTransformer(vocab_size=vocab, dim=32, n_heads=4, n_pre=2, **kw)


def test_shapes():
    m = make()
    x = torch.randint(0, 3, (2, 9))
    for r in (1, 4):
        out = m(x, r=r)
        assert out.shape == (2, 9, 3), out.shape


def test_single_shared_block():
    # n_pre prelude blocks + exactly ONE looped block: depth comes from reuse.
    m = make()
    assert m.looped_block_instances() == 3  # 2 prelude + 1 shared


def test_more_loops_changes_output_same_params():
    m = make().eval()
    x = torch.randint(0, 3, (2, 9))
    with torch.no_grad():
        a = m(x, r=1)
        b = m(x, r=6)
    assert not torch.allclose(a, b), "loop passes should change the computation"
    n1 = m.count_parameters()
    m(x, r=12)
    assert m.count_parameters() == n1, "parameters must not grow with r"


def test_prompt_injection_matters():
    m = make().eval()
    x = torch.randint(0, 3, (2, 9))
    with torch.no_grad():
        base = m(x, r=4)
        with torch.no_grad():
            saved = m.prompt_proj.weight.clone()
            m.prompt_proj.weight.zero_()
            cut = m(x, r=4)
            m.prompt_proj.weight.copy_(saved)
    assert not torch.allclose(base, cut), "prompt re-injection must affect the loop"


def test_deterministic():
    m = make().eval()
    x = torch.randint(0, 3, (2, 9))
    with torch.no_grad():
        assert torch.equal(m(x, r=3), m(x, r=3))


def test_gradients_flow_to_shared_block():
    m = make()
    x = torch.randint(0, 3, (2, 9))
    y = torch.randint(0, 2, (2,))
    loss = F.cross_entropy(m(x, r=3)[:, -1, :], y)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
    assert m.loop.attn.qkv.weight.grad.abs().sum() > 0


def test_trace_records_every_pass():
    m = make().eval()
    x = torch.randint(0, 3, (1, 9))
    with torch.no_grad():
        _, trace = m(x, r=5, record=True)
    assert trace.passes == 5
    assert len(trace.delta_norms) == 5
    assert len(trace.prompt_drift) == 5
    assert len(trace.entropies) == 5
    assert len(trace.readouts) == 5
    a = mon.audit(trace)
    assert set(a) == {"convergence", "anchoring", "decision"}


def test_parity_task_shapes():
    t = ParityTask(n_bits=8)
    x, y = t.sample(16)
    assert x.shape == (16, 9) and y.shape == (16,)
    assert set(y.tolist()) <= {0, 1}
    assert ((x[:, :8].sum(dim=1) % 2) == y).all()


def test_refine_task_shapes():
    t = RefineTask(n_bits=16, flips=4)
    x, clean = t.sample(16)
    assert x.shape == (16, 17) and clean.shape == (16, 16)
    # exactly `flips` positions differ between noisy input and clean target
    diffs = (x[:, :16] != clean).sum(dim=1)
    assert (diffs == 4).all(), diffs


def test_chain_refine_task_shapes():
    t = ChainRefineTask(n_bits=16, stay_p=0.85, noise_flips=3)
    x, clean = t.sample(2000)
    assert x.shape == (2000, 17) and clean.shape == (2000, 16)
    diffs = (x[:, :16] != clean).sum(dim=1)
    assert (diffs == 3).all(), diffs
    # chain structure: adjacent bits agree ~stay_p of the time
    agree = (clean[:, 1:] == clean[:, :-1]).float().mean().item()
    assert 0.80 < agree < 0.90, agree


def test_rotate_left():
    b = torch.tensor([[1, 0, 1, 1, 0, 0, 0, 0]])
    assert rotate_left(b, 1).tolist() == [[0, 1, 1, 0, 0, 0, 0, 1]]
    assert rotate_left(b, 8).tolist() == b.tolist()
    assert rotate_left(b, 9).tolist() == rotate_left(b, 1).tolist()


def test_rotate_task_shapes():
    t = RotateTask(n_bits=8)
    x, bits = t.sample(16)
    assert x.shape == (16, 9) and bits.shape == (16, 8)
    assert (x[:, :8] == bits).all()
    assert (x[:, 8] == 2).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} tests passed")
