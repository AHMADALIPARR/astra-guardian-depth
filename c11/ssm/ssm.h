// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
// Mamba SSM in C11: selective scan with input-dependent parameters.
// Forward + manual backward. No Python, no PyTorch.

#ifndef ASTRA_SSM_H
#define ASTRA_SSM_H

// Model config (matches Python: 102,144 params)
#define SSM_D_MODEL 64
#define SSM_N_BLOCKS 3
#define SSM_D_STATE 16
#define SSM_DT_RANK 8
#define SSM_D_INNER (2 * SSM_D_MODEL)  // 128
#define SSM_VOCAB 7
#define SSM_CONV_K 4

typedef struct {
    // SelectiveSSM params
    float A_log[SSM_D_INNER][SSM_D_STATE];  // A = -exp(A_log)
    float D[SSM_D_INNER];
    float x_proj_w[SSM_DT_RANK + 2*SSM_D_STATE][SSM_D_INNER]; // (out, in)
    float dt_proj_w[SSM_D_INNER][SSM_DT_RANK];
    float dt_proj_b[SSM_D_INNER];
    // MambaBlock params
    float in_proj_w[2*SSM_D_INNER][SSM_D_MODEL];
    float conv_w[SSM_D_INNER][SSM_CONV_K];   // depthwise
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

// Forward cache for backward pass
typedef struct {
    // per-block, per-timestep intermediates (allocated by forward)
    float *block_out[SSM_N_BLOCKS];  // (T, D_MODEL) each
    float *ssm_out[SSM_N_BLOCKS];    // (T, D_INNER) each
    float *conv_out[SSM_N_BLOCKS];   // (T, D_INNER) each
    // ... (full cache defined in ssm.c)
    void *extra;
} ForwardCache;

void mamba_init(MambaModel *m, unsigned long long seed);
int mamba_param_count(void);
// Forward: ids (T) -> logits (T, VOCAB). Cache optional (NULL = no cache).
void mamba_forward(MambaModel *m, const int *ids, int T,
                   float *logits, ForwardCache *cache);

#endif
