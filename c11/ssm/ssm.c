// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
// C11 Mamba forward: saves all intermediates for backward.

#include "ssm.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static float randf(unsigned long long *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((*s >> 33) / (double)(1ULL << 31)) - 1.0f;
}

static void linear_fwd(const float *W, const float *x, float *y, int out_d, int in_d) {
    for (int o = 0; o < out_d; o++) {
        float s = 0;
        for (int i = 0; i < in_d; i++) s += W[o*in_d + i] * x[i];
        y[o] = s;
    }
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

static void rmsnorm_fwd(const float *x, const float *w, float *y, int dim) {
    float ms = 0;
    for (int i = 0; i < dim; i++) ms += x[i]*x[i];
    float inv = 1.0f / sqrtf(ms / dim + 1e-5f);
    for (int i = 0; i < dim; i++) y[i] = x[i] * inv * w[i];
}

void mamba_init(MambaModel *m, unsigned long long seed) {
    unsigned long long s = seed ? seed : 0x12345678;
    float *p = (float*)m;
    size_t n = sizeof(MambaModel)/sizeof(float);
    for (size_t i = 0; i < n; i++) p[i] = randf(&s) * 0.1f;
    for (int b = 0; b < SSM_N_BLOCKS; b++)
        for (int i = 0; i < SSM_D_INNER; i++)
            for (int j = 0; j < SSM_D_STATE; j++)
                m->blocks[b].A_log[i][j] = logf((float)(j+1));
}

int mamba_param_count(void) { return (int)(sizeof(MambaModel)/sizeof(float)); }

void mamba_zero_grad(MambaModel *g) { memset(g, 0, sizeof(MambaModel)); }

void mamba_forward(MambaModel *m, const int *ids, int T,
                   float *logits, ForwardCache *c) {
    c->T = T;
    float h[SSM_MAX_T][SSM_D_MODEL];

    for (int t = 0; t < T; t++)
        memcpy(h[t], m->embed[ids[t]], SSM_D_MODEL*sizeof(float));

    for (int b = 0; b < SSM_N_BLOCKS; b++) {
        MambaBlockParams *bp = &m->blocks[b];
        // save input
        for (int t = 0; t < T; t++) memcpy(c->h_in[b][t], h[t], SSM_D_MODEL*sizeof(float));
        // norm
        for (int t = 0; t < T; t++)
            rmsnorm_fwd(h[t], bp->norm_w, c->normed[b][t], SSM_D_MODEL);
        // in_proj
        for (int t = 0; t < T; t++)
            linear_fwd((float*)bp->in_proj_w, c->normed[b][t],
                       c->proj[b][t], 2*SSM_D_INNER, SSM_D_MODEL);
        // conv + silu
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < SSM_D_INNER; i++) {
                float s = bp->conv_b[i];
                for (int k = 0; k < SSM_CONV_K; k++) {
                    int tt = t - k;
                    if (tt >= 0) s += bp->conv_w[i][k] * c->proj[b][tt][i];
                }
                c->xc[b][t][i] = silu(s);
            }
        }
        // selective SSM
        for (int i = 0; i < SSM_D_INNER; i++) {
            float hs[SSM_D_STATE] = {0};
            float A[SSM_D_STATE];
            for (int j = 0; j < SSM_D_STATE; j++) A[j] = -expf(bp->A_log[i][j]);
            for (int t = 0; t < T; t++) {
                float px[SSM_DT_RANK + 2*SSM_D_STATE];
                for (int o = 0; o < SSM_DT_RANK + 2*SSM_D_STATE; o++) {
                    float s = 0;
                    for (int j = 0; j < SSM_D_INNER; j++)
                        s += bp->x_proj_w[o][j] * c->xc[b][t][j];
                    px[o] = s;
                }
                float *dt_r = px, *B_ = px+SSM_DT_RANK, *C_ = B_+SSM_D_STATE;
                float dt = bp->dt_proj_b[i];
                for (int j = 0; j < SSM_DT_RANK; j++) dt += bp->dt_proj_w[i][j]*dt_r[j];
                dt = logf(1.0f + expf(dt));
                float xv = c->xc[b][t][i];
                c->ssm_dt[b][t][i] = dt;
                float y = 0;
                for (int j = 0; j < SSM_D_STATE; j++) {
                    float dA = expf(dt * A[j]);
                    float dB = (dA - 1.0f)/A[j] * B_[j];
                    hs[j] = dA*hs[j] + dB*xv;
                    y += hs[j]*C_[j];
                    c->ssm_B[b][t][i][j] = B_[j];
                    c->ssm_C[b][t][i][j] = C_[j];
                    c->ssm_dA[b][t][i][j] = dA;
                    c->ssm_dB[b][t][i][j] = dB;
                    c->ssm_h[b][t][i][j] = hs[j];
                }
                c->ssm_y[b][t][i] = y + bp->D[i]*xv;
            }
        }
        // gate + out_proj + residual
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < SSM_D_INNER; i++) {
                float z = c->proj[b][t][SSM_D_INNER + i];
                c->gated[b][t][i] = c->ssm_y[b][t][i] * silu(z);
            }
            float out[SSM_D_MODEL];
            linear_fwd((float*)bp->out_proj_w, c->gated[b][t], out, SSM_D_MODEL, SSM_D_INNER);
            for (int i = 0; i < SSM_D_MODEL; i++) h[t][i] += out[i];
        }
    }

    for (int t = 0; t < T; t++) memcpy(c->h_final[t], h[t], SSM_D_MODEL*sizeof(float));
    for (int t = 0; t < T; t++)
        rmsnorm_fwd(h[t], m->final_norm_w, c->normed_final[t], SSM_D_MODEL);
    for (int t = 0; t < T; t++)
        linear_fwd((float*)m->head_w, c->normed_final[t],
                   logits + t*SSM_VOCAB, SSM_VOCAB, SSM_D_MODEL);
}
