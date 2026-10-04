// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
// Mamba SSM in C11: forward (saves intermediates) + backward.
// No Python.

#ifndef ASTRA_SSM_H
#define ASTRA_SSM_H

#define SSM_D_MODEL 64
#define SSM_N_BLOCKS 3
#define SSM_D_STATE 16
#define SSM_DT_RANK 8
#define SSM_D_INNER (2 * SSM_D_MODEL)
#define SSM_VOCAB 7
#define SSM_CONV_K 4
#define SSM_MAX_T 128

typedef struct {
    float A_log[SSM_D_INNER][SSM_D_STATE];
    float D[SSM_D_INNER];
    float x_proj_w[SSM_DT_RANK + 2*SSM_D_STATE][SSM_D_INNER];
    float dt_proj_w[SSM_D_INNER][SSM_DT_RANK];
    float dt_proj_b[SSM_D_INNER];
    float in_proj_w[2*SSM_D_INNER][SSM_D_MODEL];
    float conv_w[SSM_D_INNER][SSM_CONV_K];
    float conv_b[SSM_D_INNER];
    float out_proj_w[SSM_D_MODEL][SSM_D_INNER];
    float norm_w[SSM_D_MODEL];
} MambaBlockParams;

typedef struct {
    float embed[SSM_VOCAB][SSM_D_MODEL];
    MambaBlockParams blocks[SSM_N_BLOCKS];
    float final_norm_w[SSM_D_MODEL];
    float head_w[SSM_VOCAB][SSM_D_MODEL];
} MambaModel;

// Forward cache: all intermediates needed for backward.
typedef struct {
    int T;
    // Per-block intermediates
    float h_in[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_MODEL];
    float normed[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_MODEL];
    float proj[SSM_N_BLOCKS][SSM_MAX_T][2*SSM_D_INNER];
    float xc[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER];      // after conv+silu
    float ssm_y[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER];
    float gated[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER];
    // SSM per-timestep params (for BPTT)
    float ssm_dt[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER];
    float ssm_B[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER][SSM_D_STATE];
    float ssm_C[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER][SSM_D_STATE];
    float ssm_dA[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER][SSM_D_STATE];
    float ssm_dB[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER][SSM_D_STATE];
    float ssm_h[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER][SSM_D_STATE];
    // Final
    float h_final[SSM_MAX_T][SSM_D_MODEL];
    float normed_final[SSM_MAX_T][SSM_D_MODEL];
} ForwardCache;

void mamba_init(MambaModel *m, unsigned long long seed);
int mamba_param_count(void);
void mamba_forward(MambaModel *m, const int *ids, int T,
                   float *logits, ForwardCache *cache);
// Backward: dlogits (T,VOCAB) -> grads. Returns nothing (grads in g).
void mamba_backward(MambaModel *m, MambaModel *g, ForwardCache *cache,
                    const int *ids, const float *dlogits);
void mamba_zero_grad(MambaModel *g);

#endif
