# SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
# Copyright (C) 2026 SnapKitty Collective
"""Internal monitoring tools for latent (hidden) reasoning.

Because the recurrent loop reasons silently in continuous activations rather
than in readable text, auditing *how* the model reached a conclusion needs
specialized probes. This module reads the LoopTrace recorded during
forward(record=True) and reports:

  * convergence: does ||h_{k+1} - h_k|| shrink (deliberation settling) or
    explode (unstable loop)?
  * prompt anchoring: does the hidden state drift away from the prompt
    representation despite re-injection?
  * decision dynamics: per-pass coda readouts + entropy -- when did the model
    "make up its mind", and did it waver?

These are behavioral probes over activations, not proofs of intent. They make
the silent loop legible; they do not make it transparent.
"""

from .model import LoopTrace


def convergence_report(trace: LoopTrace) -> dict:
    deltas = trace.delta_norms
    return {
        "passes": trace.passes,
        "delta_first": deltas[0] if deltas else None,
        "delta_last": deltas[-1] if deltas else None,
        "delta_ratio_last_first": (deltas[-1] / (deltas[0] + 1e-12)) if deltas else None,
        "settling": bool(deltas and deltas[-1] < deltas[0]),
        "max_delta": max(deltas) if deltas else None,
    }


def anchoring_report(trace: LoopTrace, drift_warn: float = 0.5) -> dict:
    drift = trace.prompt_drift
    return {
        "drift_first": drift[0] if drift else None,
        "drift_last": drift[-1] if drift else None,
        "drift_growth": (drift[-1] - drift[0]) if drift else None,
        "anchor_warning": bool(drift and drift[-1] > drift_warn),
    }


def decision_report(trace: LoopTrace) -> dict:
    """When did the per-pass readout lock onto its final answer?"""
    readouts = trace.readouts
    if not readouts:
        return {"locked_at_pass": None, "wavered": False}
    final = readouts[-1]
    locked_at = None
    for i, ro in enumerate(readouts):
        if ro == final and all(r == final for r in readouts[i:]):
            locked_at = i + 1  # 1-based pass index
            break
    wavered = any(ro != final for ro in readouts)
    return {
        "locked_at_pass": locked_at,
        "wavered": wavered,
        "final_readout": final,
        "entropy_first": trace.entropies[0] if trace.entropies else None,
        "entropy_last": trace.entropies[-1] if trace.entropies else None,
    }


def audit(trace: LoopTrace) -> dict:
    """Full audit of one recorded forward pass."""
    return {
        "convergence": convergence_report(trace),
        "anchoring": anchoring_report(trace),
        "decision": decision_report(trace),
    }


def render_audit(a: dict) -> str:
    c, an, d = a["convergence"], a["anchoring"], a["decision"]
    lines = [
        "latent-loop audit:",
        f"  passes: {c['passes']}",
        f"  convergence: delta {c['delta_first']:.4f} -> {c['delta_last']:.4f} "
        f"({'settling' if c['settling'] else 'NOT settling'})",
        f"  prompt drift: {an['drift_first']:.4f} -> {an['drift_last']:.4f} "
        f"({'ANCHOR WARNING' if an['anchor_warning'] else 'anchored'})",
        f"  decision: locked at pass {d['locked_at_pass']} "
        f"({'wavered' if d['wavered'] else 'steady'})",
        f"  entropy: {d['entropy_first']:.3f} -> {d['entropy_last']:.3f}",
    ]
    return "\n".join(lines)
