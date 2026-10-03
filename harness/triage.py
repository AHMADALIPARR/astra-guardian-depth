# SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
# Copyright (C) 2026 SnapKitty Collective
"""Proactive disambiguation & autonomous triage.

Agents fail two ways: they recklessly guess missing parameters (crashes) or
halt on every trivial ambiguity (paralysis). This module implements a
consequence classifier:

* LOW consequence  -> resolve automatically with sensible operational
                     defaults; the agent keeps moving.
* HIGH consequence -> destructive, irreversible, cross-boundary, or
                     secret-touching actions park as async clarifying
                     questions; the agent continues parallel non-blocking
                     subtasks while waiting.

Classification is rule-based and deterministic (auditable), not a vibe.
"""

from dataclasses import dataclass, field
from enum import Enum


class Consequence(Enum):
    LOW = "low"
    HIGH = "high"


@dataclass
class Action:
    """A proposed agent action awaiting triage."""
    verb: str            # e.g. "write", "delete", "deploy", "rename"
    target: str          # what it acts on
    params: dict = field(default_factory=dict)
    reversible: bool = True


# Verbs that are destructive or hard to undo by nature.
HIGH_VERBS = frozenset({
    "delete", "remove", "drop", "destroy", "overwrite", "truncate",
    "deploy", "migrate", "chmod", "chown", "format", "shutdown",
    "reboot", "publish", "release",
})

# Targets that are security boundaries or shared/persistent state.
HIGH_TARGET_HINTS = (
    "prod", "production", "/etc/", "/root", ".ssh", "secret", "token",
    "password", "credential", "database", "migration",
)

# Low-consequence markers: temp/scratch/local naming and bindings.
LOW_TARGET_HINTS = ("tmp", "temp", "scratch", "cache", ".bak")


class ConsequenceClassifier:
    """Deterministic consequence classification."""

    def classify(self, action: Action) -> Consequence:
        verb = action.verb.lower()
        target = action.target.lower()

        # Explicitly marked irreversible -> high, no exceptions.
        if not action.reversible:
            return Consequence.HIGH
        # Destructive verb on anything not obviously scratch -> high.
        if verb in HIGH_VERBS and not any(h in target for h in LOW_TARGET_HINTS):
            return Consequence.HIGH
        # Security boundary or shared persistent state -> high.
        if any(h in target for h in HIGH_TARGET_HINTS):
            return Consequence.HIGH
        # Secrets in params -> high.
        if any("secret" in k.lower() or "token" in k.lower() or "password" in k.lower()
               for k in action.params):
            return Consequence.HIGH
        return Consequence.LOW


@dataclass
class Question:
    qid: int
    action: Action
    question: str
    answer: str = None


class TriageEngine:
    """Routes actions; parks high-consequence ones as async questions."""

    def __init__(self):
        self.classifier = ConsequenceClassifier()
        self.resolved = []      # [(action, resolution)] auto-resolved
        self.pending = {}       # qid -> Question
        self._next_qid = 1

    def submit(self, action: Action, default=None):
        """Triage one action. Returns ('resolved', detail) or ('parked', qid)."""
        if self.classifier.classify(action) == Consequence.LOW:
            resolution = default if default is not None else "default-applied"
            self.resolved.append((action, resolution))
            return ("resolved", resolution)
        qid = self._next_qid
        self._next_qid += 1
        q = Question(
            qid=qid,
            action=action,
            question=(f"High-consequence action needs confirmation: "
                      f"{action.verb} {action.target} "
                      f"(reversible={action.reversible}). Proceed?"),
        )
        self.pending[qid] = q
        return ("parked", qid)

    def answer(self, qid: int, answer: str):
        """Resolve a parked question; returns the Question."""
        q = self.pending.pop(qid)
        q.answer = answer
        return q

    def pending_questions(self) -> list:
        return list(self.pending.values())
