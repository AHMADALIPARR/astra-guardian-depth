// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective

#include "context.h"
#include <ctype.h>
#include <math.h>
#include <string.h>

// Tokenize into [a-z0-9_]+ sequences, lowercase.
static int tokenize(const char *text, char toks[][ASTRA_MAX_TOKEN_LEN], int max) {
    int n = 0, len = 0;
    char cur[ASTRA_MAX_TOKEN_LEN];
    for (const char *p = text; ; p++) {
        int c = *p ? tolower((unsigned char)*p) : 0;
        int is_tok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
        if (is_tok && len < ASTRA_MAX_TOKEN_LEN - 1) {
            cur[len++] = (char)c;
        } else {
            if (len > 0 && n < max) {
                cur[len] = '\0';
                strcpy(toks[n++], cur);
                len = 0;
            }
            if (!*p) break;
        }
    }
    return n;
}

static int vocab_id(VectorIndex *vi, const char *tok) {
    for (int i = 0; i < vi->n_vocab; i++)
        if (strcmp(vi->vocab[i], tok) == 0) return i;
    return -1;
}

void vector_index_init(VectorIndex *vi) {
    vi->n_docs = 0;
    vi->n_vocab = 0;
}

int vector_index_add(VectorIndex *vi, const char *doc_id, const char *text) {
    if (vi->n_docs >= ASTRA_MAX_DOCS) return -1;
    AstraDoc *d = &vi->docs[vi->n_docs];
    strncpy(d->id, doc_id, sizeof(d->id) - 1);
    strncpy(d->text, text, sizeof(d->text) - 1);
    d->n_tokens = 0;

    char toks[ASTRA_MAX_TOKENS][ASTRA_MAX_TOKEN_LEN];
    int ntok = tokenize(text, toks, ASTRA_MAX_TOKENS);
    // per-doc token counts (dedupe within doc for df)
    char seen[ASTRA_MAX_TOKENS][ASTRA_MAX_TOKEN_LEN];
    int n_seen = 0;
    for (int i = 0; i < ntok; i++) {
        // add to doc counts
        int found = -1;
        for (int j = 0; j < d->n_tokens; j++)
            if (strcmp(d->tokens[j], toks[i]) == 0) { found = j; break; }
        if (found >= 0) {
            d->counts[found]++;
        } else if (d->n_tokens < ASTRA_MAX_TOKENS) {
            strcpy(d->tokens[d->n_tokens], toks[i]);
            d->counts[d->n_tokens] = 1;
            d->n_tokens++;
        }
        // global df (once per doc)
        int s = 0;
        for (int j = 0; j < n_seen; j++)
            if (strcmp(seen[j], toks[i]) == 0) { s = 1; break; }
        if (!s) {
            if (n_seen < ASTRA_MAX_TOKENS) strcpy(seen[n_seen++], toks[i]);
            int vid = vocab_id(vi, toks[i]);
            if (vid >= 0) {
                vi->df[vid]++;
            } else if (vi->n_vocab < ASTRA_MAX_TOKENS) {
                strcpy(vi->vocab[vi->n_vocab], toks[i]);
                vi->df[vi->n_vocab] = 1;
                vi->n_vocab++;
            }
        }
    }
    vi->n_docs++;
    return 0;
}

static double idf(VectorIndex *vi, const char *tok) {
    int vid = vocab_id(vi, tok);
    int df = (vid >= 0) ? vi->df[vid] : 0;
    return log((double)(vi->n_docs + 1) / (double)(df + 1)) + 1.0;
}

// TF-IDF cosine between query tokens and a doc. Returns score.
static double score_doc(VectorIndex *vi, AstraDoc *d,
                        char qtoks[][ASTRA_MAX_TOKEN_LEN], int nq) {
    // query vector (sparse): qid -> tf*idf; doc vector similarly.
    // Compute dot and norms directly.
    double dot = 0.0, qnorm2 = 0.0, dnorm2 = 0.0;
    // query norm
    for (int i = 0; i < nq; i++) {
        // dedupe query tokens for tf
        int dup = 0;
        for (int j = 0; j < i; j++)
            if (strcmp(qtoks[j], qtoks[i]) == 0) { dup = 1; break; }
        if (dup) continue;
        int tf = 0;
        for (int j = 0; j < nq; j++)
            if (strcmp(qtoks[j], qtoks[i]) == 0) tf++;
        double w = tf * idf(vi, qtoks[i]);
        qnorm2 += w * w;
        // doc tf
        int dtf = 0;
        for (int j = 0; j < d->n_tokens; j++)
            if (strcmp(d->tokens[j], qtoks[i]) == 0) { dtf = d->counts[j]; break; }
        if (dtf) dot += w * (dtf * idf(vi, qtoks[i]));
    }
    for (int j = 0; j < d->n_tokens; j++) {
        double w = d->counts[j] * idf(vi, d->tokens[j]);
        dnorm2 += w * w;
    }
    if (qnorm2 == 0 || dnorm2 == 0) return 0.0;
    return dot / (sqrt(qnorm2) * sqrt(dnorm2));
}

int vector_index_search(VectorIndex *vi, const char *query, int k,
                        char out_ids[][64], float *out_scores) {
    char qtoks[ASTRA_MAX_TOKENS][ASTRA_MAX_TOKEN_LEN];
    int nq = tokenize(query, qtoks, ASTRA_MAX_TOKENS);
    // score all docs, simple insertion into top-k
    int n_out = 0;
    for (int i = 0; i < vi->n_docs; i++) {
        double s = score_doc(vi, &vi->docs[i], qtoks, nq);
        // insert sorted descending
        int pos = n_out;
        while (pos > 0 && out_scores[pos - 1] < (float)s) pos--;
        if (pos < k) {
            if (n_out < k) n_out++;
            for (int j = n_out - 1; j > pos; j--) {
                out_scores[j] = out_scores[j - 1];
                strcpy(out_ids[j], out_ids[j - 1]);
            }
            out_scores[pos] = (float)s;
            strcpy(out_ids[pos], vi->docs[i].id);
        }
    }
    return n_out;
}

const char *vector_index_get(VectorIndex *vi, const char *doc_id) {
    for (int i = 0; i < vi->n_docs; i++)
        if (strcmp(vi->docs[i].id, doc_id) == 0) return vi->docs[i].text;
    return NULL;
}

void rolling_notes_init(RollingNotes *rn) {
    rn->n_notes = 0;
    rn->n_env = 0;
}

void rolling_notes_developer(RollingNotes *rn, const char *text) {
    if (rn->n_notes < 256) {
        strncpy(rn->notes[rn->n_notes++], text, ASTRA_MAX_TEXT_LEN - 1);
    }
}

void rolling_notes_env(RollingNotes *rn, const char *key, const char *value) {
    for (int i = 0; i < rn->n_env; i++) {
        if (strcmp(rn->env[i].key, key) == 0) {
            strncpy(rn->env[i].value, value, ASTRA_MAX_TEXT_LEN - 1);
            return;
        }
    }
    if (rn->n_env < 256) {
        strncpy(rn->env[rn->n_env].key, key, sizeof(rn->env[0].key) - 1);
        strncpy(rn->env[rn->n_env].value, value, ASTRA_MAX_TEXT_LEN - 1);
        rn->n_env++;
    }
}

const char *rolling_notes_recall_env(RollingNotes *rn, const char *key) {
    for (int i = 0; i < rn->n_env; i++)
        if (strcmp(rn->env[i].key, key) == 0) return rn->env[i].value;
    return NULL; // never recorded: no guessing
}

void codex_harness_init(CodexHarness *h) {
    vector_index_init(&h->index);
    rolling_notes_init(&h->notes);
}

void codex_harness_ingest(CodexHarness *h, const char *doc_id, const char *text) {
    vector_index_add(&h->index, doc_id, text);
}
