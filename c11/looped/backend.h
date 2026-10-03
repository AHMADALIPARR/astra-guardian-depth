// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
#ifndef ASTRA_LOOPED_BACKEND_H
#define ASTRA_LOOPED_BACKEND_H

#ifndef LT_D_MODEL
#define LT_D_MODEL 64
#endif
#ifndef LT_N_HEADS
#define LT_N_HEADS 4
#endif
#define LT_D_HEAD (LT_D_MODEL / LT_N_HEADS)
#ifndef LT_D_FF
#define LT_D_FF 128
#endif
#ifndef LT_VOCAB
#define LT_VOCAB 7
#endif
#ifndef LT_MAX_T
#define LT_MAX_T 128
#endif
#ifndef LT_R_LOOP
#define LT_R_LOOP 4
#endif
#define LT_EPS 1e-5

_Static_assert(LT_D_MODEL > 0 && LT_N_HEADS > 0, "positive attention dimensions required");
_Static_assert(LT_D_MODEL % LT_N_HEADS == 0, "heads must divide model width");
_Static_assert(LT_D_FF > 0 && LT_VOCAB > 0 && LT_MAX_T > 0 && LT_R_LOOP > 0,
               "positive model dimensions required");

/* Exactly the parameter layout of c11/looped at commit 5ad2721. */
typedef struct {
    float qkv_w[3 * LT_D_MODEL][LT_D_MODEL];
    float qkv_b[3 * LT_D_MODEL];
    float out_w[LT_D_MODEL][LT_D_MODEL];
    float out_b[LT_D_MODEL];
    float attn_norm_w[LT_D_MODEL];
    float gate_w[LT_D_FF][LT_D_MODEL];
    float up_w[LT_D_FF][LT_D_MODEL];
    float down_w[LT_D_MODEL][LT_D_FF];
    float mlp_norm_w[LT_D_MODEL];
} TransformerBlock;

typedef struct {
    float embed[LT_VOCAB][LT_D_MODEL];
    float pos_emb[LT_MAX_T][LT_D_MODEL];
    TransformerBlock prelude;
    TransformerBlock shared;
    float inject_w[LT_D_MODEL][LT_D_MODEL];
    float final_norm_w[LT_D_MODEL];
    float head_w[LT_VOCAB][LT_D_MODEL];
} LoopedModel;

/* Gradient tensors mirror the parameters one for one. No second model copy
 * per recurrence: every use of the shared block accumulates into g.shared. */
typedef TransformerBlock TransformerBlockGrad;
typedef LoopedModel LoopedGrad;

/* Intermediate gradients use the SAME layout as intermediate values. */
typedef struct {
    double input[LT_MAX_T][LT_D_MODEL];
    double norm1[LT_MAX_T][LT_D_MODEL];
    double qkv[LT_MAX_T][3 * LT_D_MODEL];
    double probability[LT_N_HEADS][LT_MAX_T][LT_MAX_T];
    double context[LT_MAX_T][LT_D_MODEL];
    double middle[LT_MAX_T][LT_D_MODEL];
    double norm2[LT_MAX_T][LT_D_MODEL];
    double gate[LT_MAX_T][LT_D_FF];
    double up[LT_MAX_T][LT_D_FF];
    double product[LT_MAX_T][LT_D_FF];
    double output[LT_MAX_T][LT_D_MODEL];
} BlockTensors;
typedef BlockTensors BlockTensorGrad;

typedef struct {
    int T, rounds, valid;
    const LoopedModel *owner;
    int ids[LT_MAX_T];
    /* 0 is the prelude; r+1 is the activation tape for recurrent pass r. */
    BlockTensors block[LT_R_LOOP + 1];
    double normed_final[LT_MAX_T][LT_D_MODEL];
    double logits[LT_MAX_T][LT_VOCAB];
} LoopedCache;

enum { LT_OK = 0, LT_INVALID = -1, LT_NOMEM = -2, LT_NONFINITE = -3 };

void looped_init(LoopedModel *m, unsigned long long seed);
int looped_param_count(void);
void looped_zero_grad(LoopedGrad *g);
int looped_forward(const LoopedModel *m, const int *ids, int T,
                    float *logits, LoopedCache *cache);
int looped_forward_r(const LoopedModel *m, const int *ids, int T, int rounds,
                      float *logits, LoopedCache *cache);
/* Accumulates into g. Cache/model must match; weights must not change between
 * forward and backward. g must not alias m. Cache and workspace are reentrant. */
int looped_backward(const LoopedModel *m, LoopedGrad *g, const LoopedCache *c,
                     const int *ids, const float *dlogits);
int looped_cross_entropy(const LoopedCache *c, const int *targets,
                          double *loss, float *dlogits);
#endif
