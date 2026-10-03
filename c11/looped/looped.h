// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
// C11 Looped Transformer: prelude + shared block looped r times.
// For k-conditioned rotation task, same protocol as C11 Mamba.

#ifndef ASTRA_LOOPED_H
#define ASTRA_LOOPED_H

#define LT_D_MODEL 64
#define LT_N_HEADS 4
#define LT_D_HEAD (LT_D_MODEL / LT_N_HEADS)
#define LT_D_FF 128
#define LT_VOCAB 7
#define LT_MAX_T 128
#define LT_R_LOOP 4

typedef struct {
    // Attention
    float qkv_w[3*LT_D_MODEL][LT_D_MODEL];
    float qkv_b[3*LT_D_MODEL];
    float out_w[LT_D_MODEL][LT_D_MODEL];
    float out_b[LT_D_MODEL];
    float attn_norm_w[LT_D_MODEL];
    // MLP (SwiGLU: gate, up, down)
    float gate_w[LT_D_FF][LT_D_MODEL];
    float up_w[LT_D_FF][LT_D_MODEL];
    float down_w[LT_D_MODEL][LT_D_FF];
    float mlp_norm_w[LT_D_MODEL];
} TransformerBlock;

typedef struct {
    float embed[LT_VOCAB][LT_D_MODEL];
    float pos_emb[LT_MAX_T][LT_D_MODEL];
    TransformerBlock prelude;
    TransformerBlock shared; // looped
    float inject_w[LT_D_MODEL][LT_D_MODEL]; // h0 re-injection proj
    float final_norm_w[LT_D_MODEL];
    float head_w[LT_VOCAB][LT_D_MODEL];
} LoopedModel;

// Cached intermediates for a single transformer block (for backward)
typedef struct {
    float n1[LT_MAX_T][LT_D_MODEL];
    float qkv[LT_MAX_T][3*LT_D_MODEL];
    float attn_out[LT_MAX_T][LT_D_MODEL];
    float tmp[LT_MAX_T][LT_D_MODEL];
    float n2[LT_MAX_T][LT_D_MODEL];
    float gate_pre[LT_MAX_T][LT_D_FF];
    float gate[LT_MAX_T][LT_D_FF];
    float up[LT_MAX_T][LT_D_FF];
} BlockCache;

typedef struct {
    int T;
    float h0[LT_MAX_T][LT_D_MODEL];
    float h[LT_R_LOOP+1][LT_MAX_T][LT_D_MODEL]; // h[0]=prelude out, h[r]=final
    BlockCache prelude_cache;
    BlockCache loop_cache[LT_R_LOOP];
    float normed_final[LT_MAX_T][LT_D_MODEL];
} LoopedCache;

void looped_init(LoopedModel *m, unsigned long long seed);
int looped_param_count(void);
void looped_forward(LoopedModel *m, const int *ids, int T, float *logits, LoopedCache *c);
void looped_zero_grad(LoopedModel *g);
void looped_backward(LoopedModel *m, LoopedModel *g, LoopedCache *c,
                     const int *ids, const float *dlogits);

#endif
