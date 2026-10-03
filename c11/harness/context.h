// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
// Non-lossy context persistence (Codex Harness) in C11.
//
// Exact-token TF-IDF vector index over raw documents + structured rolling
// notes. No hashing, no summarization. Recall returns verbatim text.

#ifndef ASTRA_CONTEXT_H
#define ASTRA_CONTEXT_H

#include <stddef.h>

#define ASTRA_MAX_DOCS 512
#define ASTRA_MAX_TOKENS 512
#define ASTRA_MAX_TOKEN_LEN 64
#define ASTRA_MAX_TEXT_LEN 1024

typedef struct {
    char id[64];
    char text[ASTRA_MAX_TEXT_LEN];
    // token -> count (simple linear map; vocab is small)
    char tokens[ASTRA_MAX_TOKENS][ASTRA_MAX_TOKEN_LEN];
    int counts[ASTRA_MAX_TOKENS];
    int n_tokens;
} AstraDoc;

typedef struct {
    AstraDoc docs[ASTRA_MAX_DOCS];
    int n_docs;
    // global document frequency: token -> #docs containing it
    char vocab[ASTRA_MAX_TOKENS][ASTRA_MAX_TOKEN_LEN];
    int df[ASTRA_MAX_TOKENS];
    int n_vocab;
} VectorIndex;

typedef struct {
    char key[64];
    char value[ASTRA_MAX_TEXT_LEN];
} EnvEntry;

typedef struct {
    char notes[256][ASTRA_MAX_TEXT_LEN];
    int n_notes;
    EnvEntry env[256];
    int n_env;
} RollingNotes;

typedef struct {
    VectorIndex index;
    RollingNotes notes;
} CodexHarness;

void vector_index_init(VectorIndex *vi);
int vector_index_add(VectorIndex *vi, const char *doc_id, const char *text);
// top-k search; out_ids and out_scores must hold k entries. Returns hits.
int vector_index_search(VectorIndex *vi, const char *query, int k,
                        char out_ids[][64], float *out_scores);
const char *vector_index_get(VectorIndex *vi, const char *doc_id);

void rolling_notes_init(RollingNotes *rn);
void rolling_notes_developer(RollingNotes *rn, const char *text);
void rolling_notes_env(RollingNotes *rn, const char *key, const char *value);
// Returns NULL if key was never recorded (no guessing).
const char *rolling_notes_recall_env(RollingNotes *rn, const char *key);

void codex_harness_init(CodexHarness *h);
void codex_harness_ingest(CodexHarness *h, const char *doc_id, const char *text);

#endif
