# SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
# Copyright (C) 2026 SnapKitty Collective
"""Unit tests for the Astra agentic harness (dependency-free runner)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from harness.context import CodexHarness, VectorIndex, RollingNotes
from harness.triage import Action, Consequence, TriageEngine
from harness.sandbox import Sandbox, PerimeterViolation
from harness.benchmark import bench_triage, bench_sandbox


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        check.failed += 1
check.failed = 0


# --- context: non-lossy retrieval -------------------------------------------
h = CodexHarness()
h.ingest_log("a", "ERROR db trace: conn.py:88 TimeoutError POOL_SIZE=32 exhausted")
h.ingest_log("b", "INFO api heartbeat ok")
h.notes.note_env("POOL_SIZE", "32")
h.notes.note_developer("db pool exhausted at 2026-10-02 deploy")

hits = h.query("what is POOL_SIZE", k=1)
check("retrieval finds exact doc", hits[0][0] == "a")
check("verbatim text preserved", "POOL_SIZE=32" in hits[0][2])
check("env recall exact", h.notes.recall_env("POOL_SIZE") == "32")
check("developer note verbatim",
      h.notes.recall_notes("pool")[0] == "db pool exhausted at 2026-10-02 deploy")
try:
    h.notes.recall_env("MISSING")
    check("unknown env raises (no guessing)", False)
except KeyError:
    check("unknown env raises (no guessing)", True)

# --- triage ------------------------------------------------------------------
eng = TriageEngine()
s, _ = eng.submit(Action(verb="rename", target="/tmp/x.tmp", reversible=True),
                  default="snake_case")
check("low-consequence auto-resolves", s == "resolved")
s, qid = eng.submit(Action(verb="delete", target="/data/prod", reversible=False))
check("high-consequence parks as question", s == "parked")
check("question text names the action",
      "delete" in eng.pending[qid].question)
q = eng.answer(qid, "no")
check("answer resolves parked question",
      q.answer == "no" and not eng.pending_questions())
check("triage benchmark accuracy 1.0", bench_triage()["accuracy"] == 1.0)
check("triage zero false negatives",
      bench_triage()["false_negatives"] == 0)

# --- sandbox ------------------------------------------------------------------
sb = Sandbox("/tmp/astra_harness_test", allowed_commands=("echo",))
sb.write("ok.txt", b"fine")
check("contained write/read", sb.read("ok.txt") == b"fine")
for bad, fn in [("traversal", lambda: sb.write("../x", b"")),
                ("absolute", lambda: sb.read("/etc/hostname")),
                ("blocked cmd", lambda: sb.run(["rm", "x"]))]:
    try:
        fn()
        check(f"sandbox blocks {bad}", False)
    except PerimeterViolation:
        check(f"sandbox blocks {bad}", True)
r = sb.run(["echo", "hi"])
check("allowlisted command runs", r.returncode == 0 and b"hi" in r.stdout)
res = bench_sandbox()
check("sandbox bench blocks all attacks", res["attack_block_rate"] == 1.0)
check("sandbox bench allows legit ops", res["legit_allow_rate"] == 1.0)

print(f"\n{check.failed} failures")
sys.exit(1 if check.failed else 0)
