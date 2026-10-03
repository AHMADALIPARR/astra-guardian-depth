# SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
# Copyright (C) 2026 SnapKitty Collective
"""Astra agentic harness: context persistence, triage, sandboxing, benchmarks.

Three systemic frameworks for long-horizon autonomous agents:

1. context.py  - Non-lossy context persistence (Codex Harness): background
                 vector index over raw logs/requirements + structured rolling
                 notes. No summarization, no dropping.
2. triage.py   - Proactive disambiguation & autonomous triage: consequence
                 classifier routes low-consequence decisions to automatic
                 defaults and high-consequence forks to async clarifying
                 questions, without blocking parallel subtasks.
3. sandbox.py  - Perimeter verification & sandboxing: path containment,
                 command allowlisting, and execution limits. Real enforcement,
                 not advisories.
4. benchmark.py - The benchmark harness: executes real tasks against the
                 three frameworks and reports measured numbers.
"""

from . import context, triage, sandbox, benchmark

__all__ = ["context", "triage", "sandbox", "benchmark"]
