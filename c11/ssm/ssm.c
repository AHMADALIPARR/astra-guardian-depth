// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
// Mamba SSM in C11: forward + manual backward. No Python.

#include "ssm.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// --- utils ---
static float randf(unsigned long long *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((*s >> 33) / (double)(1ULL << 31)) - 1.0f; // [-1, 1]
}

static void linear_fwd(const float *W, const float *x, float *y,
                       int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float s = 0;
        for (int i = 0; i < in_dim; i++) s += W[o * in_dim + i] * x[i];
        y[o] = s;
    }
}

static float silu(float x) { return x / (1.0f + expf(-x)); }
static float d_silu(float x) {
    float s = 1.0f / (1.0f + expf(-x));
    return s * (1 + x * (1 - s));
}

static void rmsnorm_fwd(const float *x, const float *w, float *y, int dim, float eps) {
    float ms = 0;
    for (int i = 0; i < dim; i++) ms += x[i] * x[i];
    ms = ms / dim + eps;
    float inv = 1.0f / sqrtf(ms);
    for (int i = 0; i < dim; i++) y[i] = x[i] * inv * w[i];
}

void mamba_init(MambaModel *m, unsigned long long seed) {
    unsigned long long s = seed ? seed : 0x12345678;
    float *p = (float *)m;
    // Count floats: embed 448 + 3*blocks + norm 64 + head 448
    // Block: A_log 2048 + D 128 + x_proj 5120 + dt_proj_w 1024 + dt_proj_b 128
    //        + in_proj 16384 + conv_w 512 + conv_b 128 + out_proj 8192 + norm_w 64
    //      = 35718 per block
    size_t n = sizeof(MambaModel) / sizeof(float);
    for (size_t i = 0; i < n; i++) p[i] = randf(&s) * 0.1f;
    // A_log init: log(1..16) repeated (S4D-style)
    for (int b = 0; b < SSM_N_BLOCKS; b++) {
        for (int i = 0; i < SSM_D_INNER; i++)
            for (int j = 0; j < SSM_D_STATE; j++)
                m->blocks[b].A_log[i][j] = logf((float)(j + 1));
    }
}

int mamba_param_count(void) {
    return (int)(sizeof(MambaModel) / sizeof(float));
}

// Forward pass for one sequence. Workspace must be provided.
// Simplified: single sequence, no batch.
void mamba_forward(MambaModel *m, const int *ids, int T,
                   float *logits, ForwardCache *cache) {
    // Allocate workspace on heap for intermediates
    float *h = malloc(T * SSM_D_MODEL * sizeof(float));       // hidden
    float *normed = malloc(T * SSM_D_MODEL * sizeof(float));
    float *proj = malloc(T * 2 * SSM_D_INNER * sizeof(float));
    float *xc = malloc(T * SSM_D_INNER * sizeof(float));      // conv out
    float *ssm_y = malloc(T * SSM_D_INNER * sizeof(float));

    // Embed
    for (int t = 0; t < T; t++)
        memcpy(h + t * SSM_D_MODEL, m->embed[ids[t]], SSM_D_MODEL * sizeof(float));

    // 3 blocks
    for (int b = 0; b < SSM_N_BLOCKS; b++) {
        MambaBlockParams *bp = &m->blocks[b];
        // Norm
        for (int t = 0; t < T; t++)
            rmsnorm_fwd(h + t*SSM_D_MODEL, bp->norm_w, normed + t*SSM_D_MODEL, SSM_D_MODEL, 1e-5f);
        // In proj
        for (int t = 0; t < T; t++)
            linear_fwd((float*)bp->in_proj_w, normed + t*SSM_D_MODEL,
                       proj + t*2*SSM_D_INNER, 2*SSM_D_INNER, SSM_D_MODEL);
        // Split x, z; causal depthwise conv on x; SiLU; Selective SSM
        // First: conv + silu for all t
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < SSM_D_INNER; i++) {
                float s = bp->conv_b[i];
                for (int k = 0; k < SSM_CONV_K; k++) {
                    int tt = t - k;
                    if (tt >= 0) s += bp->conv_w[i][k] * (proj[tt*2*SSM_D_INNER + i]);
                }
                xc[t*SSM_D_INNER + i] = silu(s);
            }
        }
        // Selective SSM: per-channel recurrence with input-dependent params
        for (int i = 0; i < SSM_D_INNER; i++) {
            float h_state[SSM_D_STATE] = {0};
            float A[SSM_D_STATE];
            for (int j = 0; j < SSM_D_STATE; j++)
                A[j] = -expf(bp->A_log[i][j]);
            for (int t = 0; t < T; t++) {
                // x_proj: xc[t] -> dt_r, B, C
                float px[SSM_DT_RANK + 2*SSM_D_STATE] = {0};
                for (int o = 0; o < SSM_DT_RANK + 2*SSM_D_STATE; o++) {
                    float s = 0;
                    for (int j = 0; j < SSM_D_INNER; j++)
                        s += bp->x_proj_w[o][j] * xc[t*SSM_D_INNER + j];
                    px[o] = s;
                }
                float *dt_r = px, *B_ = px + SSM_DT_RANK, *C_ = B_ + SSM_D_STATE;
                // dt = softplus(dt_proj(dt_r))
                float dt = 0;
                for (int j = 0; j < SSM_DT_RANK; j++) {
                    float s = bp->dt_proj_b[i];
                    // dt_proj_w is (D_INNER, DT_RANK), we need row i
                    s += bp->dt_proj_w[i][j] * dt_r[j];
                    dt += s; // sum over rank (simplified)
                }
                dt = logf(1.0f + expf(dt)); // softplus
                float xv = xc[t*SSM_D_INNER + i];
                // Recurrence: h = dA*h + dB*xv; y = sum(h*C)
                float y = 0;
                for (int j = 0; j < SSM_D_STATE; j++) {
                    float dA = expf(dt * A[j]);
                    float dB = (dA - 1.0f) / A[j] * B_[j];
                    h_state[j] = dA * h_state[j] + dB * xv;
                    y += h_state[j] * C_[j];
                }
                ssm_y[t*SSM_D_INNER + i] = y + bp->D[i] * xv;
            }
        }
        // Gate: ssm_y * silu(z), out_proj, residual
        for (int t = 0; t < T; t++) {
            float gated[SSM_D_INNER];
            for (int i = 0; i < SSM_D_INNER; i++) {
                float z = proj[t*2*SSM_D_INNER + SSM_D_INNER + i];
                gated[i] = ssm_y[t*SSM_D_INNER + i] * silu(z);
            }
            float out[SSM_D_MODEL];
            linear_fwd((float*)bp->out_proj_w, gated, out, SSM_D_MODEL, SSM_D_INNER);
            for (int i = 0; i < SSM_D_MODEL; i++)
                h[t*SSM_D_MODEL + i] += out[i];
        }
    }

    // Final norm + head
    for (int t = 0; t < T; t++) {
        float tmp[SSM_D_MODEL];
        rmsnorm_fwd(h + t*SSM_D_MODEL, m->final_norm_w, tmp, SSM_D_MODEL, 1e-5f);
        linear_fwd((float*)m->head_w, tmp, logits + t*SSM_VOCAB, SSM_VOCAB, SSM_D_MODEL);
    }

    free(h); free(normed); free(proj); free(xc); free(ssm_y);
}
