// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
// C11 Looped Transformer forward.

#include "looped.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static float randf(unsigned long long *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((*s >> 33) / (double)(1ULL << 31)) - 1.0f;
}

static void rmsnorm(const float *x, const float *w, float *y, int dim) {
    float ms = 0;
    for (int i = 0; i < dim; i++) ms += x[i]*x[i];
    float inv = 1.0f / sqrtf(ms/dim + 1e-5f);
    for (int i = 0; i < dim; i++) y[i] = x[i]*inv*w[i];
}

static float silu(float x) { return x/(1.0f+expf(-x)); }

// Causal multi-head attention, single sequence
static void attention(const float *x, TransformerBlock *b, float *out, int T) {
    float qkv[LT_MAX_T][3*LT_D_MODEL];
    for (int t = 0; t < T; t++) {
        for (int o = 0; o < 3*LT_D_MODEL; o++) {
            float s = b->qkv_b[o];
            for (int i = 0; i < LT_D_MODEL; i++)
                s += b->qkv_w[o][i] * x[t*LT_D_MODEL+i];
            qkv[t][o] = s;
        }
    }
    // Per head
    float attn_out[LT_MAX_T][LT_D_MODEL];
    memset(attn_out, 0, sizeof(attn_out));
    for (int hd = 0; hd < LT_N_HEADS; hd++) {
        for (int t = 0; t < T; t++) {
            float scores[LT_MAX_T];
            float mx = -1e30f;
            for (int s = 0; s <= t; s++) {
                float sc = 0;
                for (int i = 0; i < LT_D_HEAD; i++) {
                    int qi = hd*LT_D_HEAD+i;
                    int ki = hd*LT_D_HEAD+i;
                    sc += qkv[t][qi] * qkv[s][LT_D_MODEL+ki];
                }
                sc /= sqrtf((float)LT_D_HEAD);
                scores[s] = sc;
                if (sc > mx) mx = sc;
            }
            float sum = 0;
            for (int s = 0; s <= t; s++) {
                scores[s] = expf(scores[s]-mx);
                sum += scores[s];
            }
            for (int i = 0; i < LT_D_HEAD; i++) {
                float v = 0;
                for (int s = 0; s <= t; s++) {
                    int vi = hd*LT_D_HEAD+i;
                    v += (scores[s]/sum) * qkv[s][2*LT_D_MODEL+vi];
                }
                attn_out[t][hd*LT_D_HEAD+i] = v;
            }
        }
    }
    for (int t = 0; t < T; t++) {
        for (int o = 0; o < LT_D_MODEL; o++) {
            float s = b->out_b[o];
            for (int i = 0; i < LT_D_MODEL; i++)
                s += b->out_w[o][i] * attn_out[t][i];
            out[t*LT_D_MODEL+o] = s;
        }
    }
}

static void transformer_block(const float *h_in, TransformerBlock *b,
                              float *h_out, int T) {
    float tmp[LT_MAX_T][LT_D_MODEL];
    float n1[LT_MAX_T][LT_D_MODEL];
    float attn[LT_MAX_T][LT_D_MODEL];
    float n2[LT_MAX_T][LT_D_MODEL];
    // Attn block
    for (int t = 0; t < T; t++)
        rmsnorm(h_in+t*LT_D_MODEL, b->attn_norm_w, n1[t], LT_D_MODEL);
    attention((float*)n1, b, (float*)attn, T);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < LT_D_MODEL; i++)
            tmp[t][i] = h_in[t*LT_D_MODEL+i] + attn[t][i];
    // MLP block (SwiGLU)
    for (int t = 0; t < T; t++)
        rmsnorm(tmp[t], b->mlp_norm_w, n2[t], LT_D_MODEL);
    for (int t = 0; t < T; t++) {
        float gate[LT_D_FF], up[LT_D_FF];
        for (int o = 0; o < LT_D_FF; o++) {
            float sg = 0, su = 0;
            for (int i = 0; i < LT_D_MODEL; i++) {
                sg += b->gate_w[o][i] * n2[t][i];
                su += b->up_w[o][i] * n2[t][i];
            }
            gate[o] = silu(sg);
            up[o] = su;
        }
        for (int o = 0; o < LT_D_MODEL; o++) {
            float s = 0;
            for (int i = 0; i < LT_D_FF; i++)
                s += b->down_w[o][i] * gate[i] * up[i];
            h_out[t*LT_D_MODEL+o] = tmp[t][o] + s;
        }
    }
}

void looped_init(LoopedModel *m, unsigned long long seed) {
    unsigned long long s = seed ? seed : 0x9999;
    float *p = (float*)m;
    size_t n = sizeof(LoopedModel)/sizeof(float);
    for (size_t i = 0; i < n; i++) p[i] = randf(&s) * 0.1f;
    // norm weights to 1
    for (int i = 0; i < LT_D_MODEL; i++) {
        m->prelude.attn_norm_w[i] = 1;
        m->prelude.mlp_norm_w[i] = 1;
        m->shared.attn_norm_w[i] = 1;
        m->shared.mlp_norm_w[i] = 1;
        m->final_norm_w[i] = 1;
    }
}

int looped_param_count(void) { return (int)(sizeof(LoopedModel)/sizeof(float)); }
void looped_zero_grad(LoopedModel *g) { memset(g, 0, sizeof(LoopedModel)); }

void looped_forward(LoopedModel *m, const int *ids, int T,
                    float *logits, LoopedCache *c) {
    float h[LT_MAX_T][LT_D_MODEL];
    // Embed + pos
    for (int t = 0; t < T; t++)
        for (int i = 0; i < LT_D_MODEL; i++)
            h[t][i] = m->embed[ids[t]][i] + m->pos_emb[t][i];
    // Prelude
    transformer_block((float*)h, &m->prelude, (float*)c->h[0], T);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < LT_D_MODEL; i++)
            c->h0[t][i] = c->h[0][t][i];
    // Loop r times with re-injection
    for (int r = 0; r < LT_R_LOOP; r++) {
        float hin[LT_MAX_T][LT_D_MODEL];
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < LT_D_MODEL; i++) {
                float inj = 0;
                for (int j = 0; j < LT_D_MODEL; j++)
                    inj += m->inject_w[i][j] * c->h0[t][j];
                hin[t][i] = c->h[r][t][i] + inj;
            }
        }
        transformer_block((float*)hin, &m->shared, (float*)c->h[r+1], T);
    }
    // Coda
    float normed[LT_MAX_T][LT_D_MODEL];
    for (int t = 0; t < T; t++)
        rmsnorm(c->h[LT_R_LOOP][t], m->final_norm_w, normed[t], LT_D_MODEL);
    for (int t = 0; t < T; t++)
        for (int v = 0; v < LT_VOCAB; v++) {
            float s = 0;
            for (int i = 0; i < LT_D_MODEL; i++)
                s += m->head_w[v][i] * normed[t][i];
            logits[t*LT_VOCAB+v] = s;
        }
    c->T = T;
}

// Backward is substantial; placeholder for now - full impl needed
void looped_backward(LoopedModel *m, LoopedModel *g, LoopedCache *c,
                     const int *ids, const float *dlogits) {
    // TODO: full backprop through attention, MLP, loop
    (void)m; (void)g; (void)c; (void)ids; (void)dlogits;
}
