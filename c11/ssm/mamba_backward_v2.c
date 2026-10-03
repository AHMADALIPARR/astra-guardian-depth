// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
#include "internal.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    double dh[SSM_MAX_T][SSM_D_MODEL];
    double dx[SSM_MAX_T][SSM_D_INNER];
    double dy[SSM_MAX_T][SSM_D_INNER];
    double dp[SSM_MAX_T][2 * SSM_D_INNER];
    double dpx[SSM_MAX_T][SSM_P];
} BackwardWork;

int mamba_backward(const MambaModel *m, MambaModel *g, const ForwardCache *c,
                   const int *ids, const float *dlogits) {
    if (!m || !g || m == g || !c || !c->valid || c->owner != m ||
        !ids || !dlogits || c->T < 1 || c->T > SSM_MAX_T) return MAMBA_INVALID;
    int T = c->T;
    for (int t = 0; t < T; ++t) {
        if (ids[t] != c->ids[t]) return MAMBA_INVALID;
        for (int v = 0; v < SSM_VOCAB; ++v)
            if (!isfinite(dlogits[t*SSM_VOCAB+v])) return MAMBA_NONFINITE;
    }
    BackwardWork *w = calloc(1, sizeof *w);
    if (!w) return MAMBA_NOMEM;
    for (int t = 0; t < T; ++t) {
        double dn[SSM_D_MODEL] = {0};
        for (int v = 0; v < SSM_VOCAB; ++v) {
            double d = dlogits[t*SSM_VOCAB+v];
            for (int j = 0; j < SSM_D_MODEL; ++j) {
                g->head_w[v][j] += (float)(d * c->normed_final[t][j]);
                dn[j] += d * m->head_w[v][j];
            }
        }
        norm_backward(c->h[SSM_N_BLOCKS][t], m->final_norm_w, dn,
                      w->dh[t], g->final_norm_w);
    }
    for (int b = SSM_N_BLOCKS - 1; b >= 0; --b) {
        const MambaBlockParams *p = &m->blocks[b];
        MambaBlockParams *gp = &g->blocks[b];
        memset(w->dx, 0, sizeof w->dx);
        memset(w->dp, 0, sizeof w->dp);
        memset(w->dpx, 0, sizeof w->dpx);
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < SSM_D_INNER; ++i) {
                double dg = 0;
                for (int o = 0; o < SSM_D_MODEL; ++o) {
                    gp->out_proj_w[o][i] += (float)(w->dh[t][o] * c->gated[b][t][i]);
                    dg += p->out_proj_w[o][i] * w->dh[t][o];
                }
                double z = c->proj[b][t][SSM_D_INNER+i];
                w->dy[t][i] = dg * silu(z);
                w->dp[t][SSM_D_INNER+i] = dg * c->ssm_y[b][t][i] * dsilu(z);
            }
        }
        /* BPTT. dpx is shared across channels because B/C/dt_r are shared. */
        for (int i = 0; i < SSM_D_INNER; ++i) {
            double carry[SSM_D_STATE] = {0}, A[SSM_D_STATE];
            for (int j = 0; j < SSM_D_STATE; ++j) A[j] = -exp(p->A_log[i][j]);
            for (int t = T - 1; t >= 0; --t) {
                double dy = w->dy[t][i], x = c->xc[b][t][i];
                double dt = c->dt[b][t][i], ddt = 0;
                gp->D[i] += (float)(dy * x);
                w->dx[t][i] += dy * p->D[i];
                for (int j = 0; j < SSM_D_STATE; ++j) {
                    double rho, q, qa;
                    discretize(A[j], dt, &rho, &q, &qa);
                    double B = c->px[b][t][SSM_DT_RANK+j];
                    double C = c->px[b][t][SSM_DT_RANK+SSM_D_STATE+j];
                    double prev = t ? c->state[b][t-1][i][j] : 0;
                    double gh = carry[j] + dy * C;
                    w->dpx[t][SSM_DT_RANK+SSM_D_STATE+j] += dy * c->state[b][t][i][j];
                    w->dpx[t][SSM_DT_RANK+j] += gh * q * x;
                    w->dx[t][i] += gh * q * B;
                    ddt += gh * rho * (prev * A[j] + B * x);
                    gp->A_log[i][j] += (float)(gh * (prev * dt * rho + B * x * qa) * A[j]);
                    carry[j] = gh * rho;
                }
                double draw = ddt * sigmoid(c->dt_raw[b][t][i]);
                gp->dt_proj_b[i] += (float)draw;
                for (int r = 0; r < SSM_DT_RANK; ++r) {
                    gp->dt_proj_w[i][r] += (float)(draw * c->px[b][t][r]);
                    w->dpx[t][r] += draw * p->dt_proj_w[i][r];
                }
            }
        }
        for (int t = 0; t < T; ++t) {
            for (int o = 0; o < SSM_P; ++o) {
                for (int i = 0; i < SSM_D_INNER; ++i) {
                    gp->x_proj_w[o][i] += (float)(w->dpx[t][o] * c->xc[b][t][i]);
                    w->dx[t][i] += w->dpx[t][o] * p->x_proj_w[o][i];
                }
            }
        }
        /* Finish ALL convolution input gradients before in-projection BPTT. */
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < SSM_D_INNER; ++i) {
                double d = w->dx[t][i] * dsilu(c->conv[b][t][i]);
                gp->conv_b[i] += (float)d;
                for (int k = 0; k < SSM_CONV_K && k <= t; ++k) {
                    gp->conv_w[i][k] += (float)(d * c->proj[b][t-k][i]);
                    w->dp[t-k][i] += d * p->conv_w[i][k];
                }
            }
        }
        for (int t = 0; t < T; ++t) {
            double dn[SSM_D_MODEL] = {0};
            for (int o = 0; o < 2 * SSM_D_INNER; ++o) {
                for (int j = 0; j < SSM_D_MODEL; ++j) {
                    gp->in_proj_w[o][j] += (float)(w->dp[t][o] * c->normed[b][t][j]);
                    dn[j] += w->dp[t][o] * p->in_proj_w[o][j];
                }
            }
            /* dh already contains the residual branch's identity gradient. */
            norm_backward(c->h[b][t], p->norm_w, dn, w->dh[t], gp->norm_w);
        }
    }
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < SSM_D_MODEL; ++j)
            g->embed[ids[t]][j] += (float)w->dh[t][j];
    free(w);
    return MAMBA_OK;
}
