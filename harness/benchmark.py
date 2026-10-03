# SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
# Copyright (C) 2026 SnapKitty Collective
"""Benchmark harness: true (executed, measured) benchmarks for the frameworks.

Each benchmark runs the real implementation against generated tasks and
reports measured numbers. No mocks, no stubs.

  bench_context_retrieval  - CodexHarness vs a lossy (truncated) baseline on
                             exact fact retrieval from synthetic execution logs.
  bench_triage             - ConsequenceClassifier on a labeled action set.
  bench_sandbox            - Sandbox enforcement: legit ops succeed, escape
                             attempts raise PerimeterViolation.

Run: python -m harness.benchmark  (from the repo root)
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.context import CodexHarness, VectorIndex
from harness.triage import Action, Consequence, TriageEngine
from harness.sandbox import Sandbox, PerimeterViolation


# ---------------------------------------------------------------------------
# 1. Context retrieval: non-lossy vs lossy
# ---------------------------------------------------------------------------

def _gen_logs(n_docs=1500, seed=0):
    rng = random.Random(seed)
    services = ["api", "worker", "scheduler", "db", "cache"]
    levels = ["INFO", "WARN", "ERROR"]
    docs, facts = {}, []
    for i in range(n_docs):
        svc = rng.choice(services)
        lvl = rng.choice(levels)
        if lvl == "ERROR" and rng.random() < 0.3:
            # plant a precise, retrievable fact
            var = f"CFG_{rng.randint(1000, 9999)}"
            val = f"{rng.randint(10000, 99999)}"
            file = f"svc_{svc}/mod_{rng.randint(1, 40)}.py"
            line = rng.randint(1, 900)
            text = (f"{lvl} {svc} trace {i}: {file}:{line} in handle_request "
                    f"ValueError: {var}={val} failed validation")
            facts.append((f"what is {var}", var, val, f"doc{i}"))
        else:
            text = (f"{lvl} {svc} trace {i}: heartbeat ok "
                    f"latency_ms={rng.randint(1, 900)}")
        docs[f"doc{i}"] = text
    return docs, facts


def bench_context_retrieval(n_docs=1500, seed=0):
    docs, facts = _gen_logs(n_docs, seed)
    t0 = time.time()

    # Non-lossy harness: index everything.
    harness = CodexHarness()
    for doc_id, text in docs.items():
        harness.ingest_log(doc_id, text)

    # Lossy baseline: keep every 5th doc (simulates aggressive summarization).
    lossy = VectorIndex()
    for doc_id, text in docs.items():
        if int(doc_id[3:]) % 5 == 0:
            lossy.add(doc_id, text)

    def score(index):
        r1 = r5 = exact = 0
        for q, var, val, want in facts:
            hits = index.search(q, k=5)
            ids = [d for d, _ in hits]
            if ids and ids[0] == want:
                r1 += 1
            if want in ids:
                r5 += 1
            if ids and val in index.get(ids[0]):
                exact += 1
        n = len(facts)
        return {"recall@1": r1 / n, "recall@5": r5 / n, "exact_value": exact / n,
                "n_facts": n, "n_docs": len(docs)}

    full = score(harness.index)
    lossy_scores = score(lossy)
    return {
        "non_lossy": full,
        "lossy_baseline": lossy_scores,
        "index_docs": len(harness.index),
        "seconds": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# 2. Triage classification
# ---------------------------------------------------------------------------

TRIAGE_CASES = [
    # (verb, target, reversible, params, expected)
    ("rename", "/tmp/scratch_01.tmp", True, {}, Consequence.LOW),
    ("write", "/tmp/notes.txt", True, {}, Consequence.LOW),
    ("read", "/tmp/cache/data.json", True, {}, Consequence.LOW),
    ("rename", "report_final_v2.md", True, {}, Consequence.LOW),
    ("write", "./local/output.log", True, {}, Consequence.LOW),
    ("set", "tmp_binding", True, {}, Consequence.LOW),
    ("copy", "/tmp/a.tmp to /tmp/b.tmp", True, {}, Consequence.LOW),
    ("read", "config/local.yaml", True, {}, Consequence.LOW),
    ("write", "scratch/experiment_3.txt", True, {}, Consequence.LOW),
    ("rename", "tmp_upload.bin", True, {}, Consequence.LOW),
    ("list", "/tmp", True, {}, Consequence.LOW),
    ("write", "cache/session.tmp", True, {}, Consequence.LOW),
    ("delete", "/tmp/old_cache.tmp", True, {}, Consequence.LOW),
    ("read", "./README.md", True, {}, Consequence.LOW),
    ("set", "temp_var", True, {}, Consequence.LOW),
    ("write", "/tmp/debug.out", True, {}, Consequence.LOW),
    ("rename", "draft.tmp", True, {}, Consequence.LOW),
    ("copy", "a.tmp", True, {}, Consequence.LOW),
    ("delete", "/data/prod/users", False, {}, Consequence.HIGH),
    ("drop", "database prod_ledger", False, {}, Consequence.HIGH),
    ("deploy", "prod/api", False, {}, Consequence.HIGH),
    ("migrate", "database schema v42", False, {}, Consequence.HIGH),
    ("overwrite", "/etc/nginx.conf", True, {}, Consequence.HIGH),
    ("chmod", "/root/.ssh", True, {}, Consequence.HIGH),
    ("write", "/etc/passwd", True, {}, Consequence.HIGH),
    ("read", "~/.ssh/id_rsa", True, {}, Consequence.HIGH),
    ("publish", "release v2.0", False, {}, Consequence.HIGH),
    ("delete", "production backups", False, {}, Consequence.HIGH),
    ("write", "config", True, {"api_token": "x"}, Consequence.HIGH),
    ("set", "db_password", True, {"db_password": "secret"}, Consequence.HIGH),
    ("format", "/dev/sda", False, {}, Consequence.HIGH),
    ("shutdown", "prod cluster", False, {}, Consequence.HIGH),
    ("delete", "customer records", False, {}, Consequence.HIGH),
    ("deploy", "staging", True, {}, Consequence.HIGH),
    ("migrate", "prod database", True, {}, Consequence.HIGH),
    ("read", "secret vault", True, {}, Consequence.HIGH),
]

LOW_DEFAULTS = {"naming": "snake_case", "tmp_dir": "/tmp", "retries": 3}


def bench_triage():
    eng = TriageEngine()
    tp = tn = fp = fn = 0
    auto = 0
    for verb, target, rev, params, expected in TRIAGE_CASES:
        a = Action(verb=verb, target=target, reversible=rev, params=params)
        got = eng.classifier.classify(a)
        if got == Consequence.HIGH and expected == Consequence.HIGH:
            tp += 1
        elif got == Consequence.LOW and expected == Consequence.LOW:
            tn += 1
        elif got == Consequence.HIGH:
            fp += 1
        else:
            fn += 1
        status, _ = eng.submit(a, default="auto-default")
        if status == "resolved":
            auto += 1
    n = len(TRIAGE_CASES)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    return {
        "accuracy": (tp + tn) / n,
        "precision_high": prec,
        "recall_high": rec,
        "auto_resolved": auto,
        "parked": len(eng.pending_questions()),
        "n_cases": n,
        "false_negatives": fn,  # dangerous: HIGH misclassified as LOW
    }


# ---------------------------------------------------------------------------
# 3. Sandbox enforcement
# ---------------------------------------------------------------------------

def bench_sandbox(root="/tmp/astra_sandbox_bench"):
    import shutil
    shutil.rmtree(root, ignore_errors=True)
    sb = Sandbox(root, allowed_commands=("echo", "true"))

    legit_ok = legit_n = 0
    # legitimate: write + read inside root
    try:
        legit_n += 1
        sb.write("work/out.txt", b"hello")
        assert sb.read("work/out.txt") == b"hello"
        legit_ok += 1
    except Exception:
        pass
    # legitimate: allowlisted command
    try:
        legit_n += 1
        r = sb.run(["echo", "hi"], timeout=10)
        assert r.returncode == 0 and b"hi" in r.stdout
        legit_ok += 1
    except Exception:
        pass

    attacks = [
        ("traversal", lambda: sb.write("../../evil.txt", b"x")),
        ("absolute", lambda: sb.write("/etc/evil.txt", b"x")),
        ("read_outside", lambda: sb.read("/etc/hostname")),
        ("blocked_cmd", lambda: sb.run(["rm", "-rf", "/"], timeout=10)),
        ("blocked_cmd2", lambda: sb.run(["curl", "http://x"], timeout=10)),
    ]
    # symlink escape: link inside root pointing outside
    try:
        os.symlink("/etc", os.path.join(root, "link_out"))
        attacks.append(("symlink_escape",
                        lambda: sb.read("link_out/hostname")))
    except OSError:
        pass

    blocked = 0
    for name, fn in attacks:
        try:
            fn()
        except PerimeterViolation:
            blocked += 1
        except Exception:
            pass  # wrong exception type still counts as not-blocked-cleanly
    shutil.rmtree(root, ignore_errors=True)
    return {
        "legit_allow_rate": legit_ok / legit_n,
        "attack_block_rate": blocked / len(attacks),
        "n_attacks": len(attacks),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "context_retrieval": bench_context_retrieval,
    "triage": bench_triage,
    "sandbox": bench_sandbox,
}


def main():
    results = {}
    for name, fn in BENCHMARKS.items():
        t0 = time.time()
        try:
            results[name] = {"status": "ok", "seconds": round(time.time() - t0, 2),
                             **fn()}
        except Exception as e:  # noqa: BLE001 - benchmark must report, not crash
            results[name] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(results, indent=2))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "benchmark_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
