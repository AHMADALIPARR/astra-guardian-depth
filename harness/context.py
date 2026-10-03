# SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
# Copyright (C) 2026 SnapKitty Collective
"""Non-lossy context persistence (Codex Harness).

Standard agents compress long histories with lossy summarization: stack
traces get truncated, environment variables get paraphrased, exact values
get lost. This module decouples working memory from raw storage:

* VectorIndex: every raw log line / requirement is embedded (hashed
  bag-of-words, deterministic, no external dependencies) and kept fully
  searchable in a background vector space. Nothing is dropped.
* RollingNotes: active working memory holds EXACT developer notes and
  environment key-values. Recall returns the original string, never a
  summary.

Retrieval is cosine similarity over L2-normalized hashed embeddings.
"""

import hashlib
import math
import re
from dataclasses import dataclass, field


def _hash_token(token: str, dim: int) -> int:
    return int(hashlib.sha256(token.encode()).hexdigest(), 16) % dim


def tokenize(text: str) -> list:
    """Split on non-alphanumerics so `CFG_1=42` yields `cfg_1` and `42`."""
    return re.findall(r"[a-z0-9_]+", text.lower())


def embed(text: str, dim: int = 256) -> list:
    """Deterministic hashed bag-of-words embedding, L2-normalized."""
    vec = [0.0] * dim
    for token in tokenize(text):
        vec[_hash_token(token, dim)] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorIndex:
    """Background vector space over raw, unmodified documents.

    Exact-token TF-IDF (inverted index, no hashing): rare tokens — the exact
    variable names and values we need to retrieve — dominate the similarity;
    ubiquitous log boilerplate is downweighted. No hash collisions, no
    information loss. Document vectors are (re)computed from current document
    frequencies on first search after new adds.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim  # kept for API compatibility; unused (exact tokens)
        self.docs = {}    # doc_id -> original text (verbatim, never summarized)
        self.counts = {}  # doc_id -> {token: term count}
        self.df = {}      # token -> document frequency
        self._vecs = None  # cached TF-IDF vectors
        self._vocab = None

    def add(self, doc_id: str, text: str):
        self.docs[doc_id] = text
        counts = {}
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
        self.counts[doc_id] = counts
        for tok in counts:
            self.df[tok] = self.df.get(tok, 0) + 1
        self._vecs = None  # invalidate cache
        self._vocab = None

    def __len__(self):
        return len(self.docs)

    def _ensure(self):
        if self._vecs is None:
            self._vocab = sorted(self.df)
            pos = {t: i for i, t in enumerate(self._vocab)}
            n = len(self.docs)
            idf = {t: math.log((n + 1) / (df + 1)) + 1.0
                   for t, df in self.df.items()}
            vecs = {}
            for doc_id, counts in self.counts.items():
                v = {}
                for tok, c in counts.items():
                    v[pos[tok]] = c * idf[tok]
                norm = math.sqrt(sum(x * x for x in v.values()))
                vecs[doc_id] = ({i: x / norm for i, x in v.items()}
                                if norm else v)
            self._vecs = vecs
            self._idf = idf
            self._pos = pos

    def search(self, query: str, k: int = 5) -> list:
        """Return [(doc_id, score)] top-k by TF-IDF cosine similarity."""
        self._ensure()
        q = {}
        for token in tokenize(query):
            if token in self._pos:
                i = self._pos[token]
                q[i] = q.get(i, 0) + self._idf[token]
        norm = math.sqrt(sum(x * x for x in q.values()))
        if norm:
            q = {i: x / norm for i, x in q.items()}
        scored = []
        for doc_id, dv in self._vecs.items():
            s = sum(qv * dv.get(i, 0.0) for i, qv in q.items())
            scored.append((doc_id, s))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def get(self, doc_id: str) -> str:
        """Exact original text. This is the non-lossy guarantee."""
        return self.docs[doc_id]


@dataclass
class RollingNotes:
    """Structured working memory: exact notes, keyed environment facts."""
    developer_notes: list = field(default_factory=list)
    env: dict = field(default_factory=dict)

    def note_developer(self, text: str):
        """Store a developer note verbatim."""
        self.developer_notes.append(text)

    def note_env(self, key: str, value: str):
        """Store an environment fact exactly."""
        self.env[key] = value

    def recall_env(self, key: str) -> str:
        """Exact value, KeyError if never recorded (no guessing)."""
        return self.env[key]

    def recall_notes(self, keyword: str) -> list:
        """All developer notes containing the keyword, verbatim."""
        kw = keyword.lower()
        return [n for n in self.developer_notes if kw in n.lower()]


class CodexHarness:
    """The full persistence harness: vector index + rolling notes."""

    def __init__(self, dim: int = 1024):
        self.index = VectorIndex(dim=dim)
        self.notes = RollingNotes()

    def ingest_log(self, doc_id: str, text: str):
        self.index.add(doc_id, text)

    def query(self, q: str, k: int = 5) -> list:
        """Top-k (doc_id, score, verbatim_text)."""
        return [(doc_id, s, self.index.get(doc_id))
                for doc_id, s in self.index.search(q, k)]
