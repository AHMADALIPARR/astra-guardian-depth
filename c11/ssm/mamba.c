// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SSM_D_MODEL
#define SSM_D_MODEL 64
#endif
#ifndef SSM_N_BLOCKS
#define SSM_N_BLOCKS 3
#endif
#ifndef SSM_D_STATE
#define SSM_D_STATE 16
#endif
#ifndef SSM_DT_RANK
#define SSM_DT_RANK 8
#endif
#ifndef SSM_D_INNER
#define SSM_D_INNER (2 * SSM_D_MODEL)
#endif
#ifndef SSM_VOCAB
#define SSM_VOCAB 7
#endif
#ifndef SSM_CONV_K
#define SSM_CONV_K 4
#endif
#ifndef SSM_MAX_T
#define SSM_MAX_T 128
#endif

#define SSM_P (SSM_DT_RANK + 2 * SSM_D_STATE)
#define SSM_EPS 1e-5

typedef struct {
    float A_log[SSM_D_INNER][SSM_D_STATE];
    float D[SSM_D_INNER];
    float x_proj_w[SSM_P][SSM_D_INNER];
    float dt_proj_w[SSM_D_INNER][SSM_DT_RANK];
    float dt_proj_b[SSM_D_INNER];
    float in_proj_w[2 * SSM_D_INNER][SSM_D_MODEL];
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

/* Allocate this large object on the heap. */
typedef struct {
    int T, valid;
    const MambaModel *owner;
    int ids[SSM_MAX_T];

    double h[SSM_N_BLOCKS + 1][SSM_MAX_T][SSM_D_MODEL];
    double normed[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_MODEL];
    double proj[SSM_N_BLOCKS][SSM_MAX_T][2 * SSM_D_INNER];
    double conv[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER];
    double xc[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER];
    double px[SSM_N_BLOCKS][SSM_MAX_T][SSM_P];
    double dt_raw[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER];
    double dt[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER];
    double state[SSM_N_BLOCKS][SSM_MAX_T]
                [SSM_D_INNER][SSM_D_STATE];
    double ssm_y[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER];
    double gated[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER];
    double normed_final[SSM_MAX_T][SSM_D_MODEL];
    double logits[SSM_MAX_T][SSM_VOCAB];
} ForwardCache;

enum {
    MAMBA_OK = 0,
    MAMBA_INVALID = -1,
    MAMBA_NOMEM = -2,
    MAMBA_NONFINITE = -3
};

static inline double sigmoid(double x)
{
    double e = exp(-fabs(x));
    return x >= 0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
}

static inline double softplus(double x)
{
    return fmax(x, 0) + log1p(exp(-fabs(x)));
}

static inline double silu(double x)
{
    return x * sigmoid(x);
}

static inline double dsilu(double x)
{
    double s = sigmoid(x);
    return s + x * s * (1.0 - s);
}

/*
 * rho = exp(a * dt)
 * q   = (exp(a * dt) - 1) / a
 * qa  = derivative of q with respect to a
 *
 * The series handles cancellation and the removable pole at a = 0.
 */
static inline void discretize(
    double a, double dt,
    double *rho, double *q, double *qa)
{
    double z = a * dt;
    *rho = exp(z);

    if (fabs(z) < 1e-4) {
        *q = dt * (
            1 + z * (
                0.5 + z * (
                    1.0 / 6 + z * (
                        1.0 / 24 + z * (
                            1.0 / 120 + z / 720)))));

        *qa = dt * dt * (
            0.5 + z * (
                1.0 / 3 + z * (
                    1.0 / 8 + z * (
                        1.0 / 30 + z / 144))));
    } else {
        *q = expm1(z) / a;
        *qa = (z * *rho - expm1(z)) / (a * a);
    }
}

static inline double rms_inv(const double *x)
{
    double sum = 0;
    for (int i = 0; i < SSM_D_MODEL; ++i)
        sum += x[i] * x[i];

    return 1.0 / sqrt(sum / SSM_D_MODEL + SSM_EPS);
}

static inline void norm_forward(
    const double *x, const float *w, double *y)
{
    double r = rms_inv(x);
    for (int i = 0; i < SSM_D_MODEL; ++i)
        y[i] = x[i] * w[i] * r;
}

static inline void norm_backward(
    const double *x, const float *w,
    const double *dy, double *dx, float *dw)
{
    double r = rms_inv(x), dot = 0;

    for (int i = 0; i < SSM_D_MODEL; ++i)
        dot += dy[i] * w[i] * x[i];

    for (int i = 0; i < SSM_D_MODEL; ++i) {
        dw[i] += (float)(dy[i] * x[i] * r);
        dx[i] += r * dy[i] * w[i]
               - x[i] * r * r * r * dot / SSM_D_MODEL;
    }
}

static double uniform(unsigned long long *s)
{
    *s = *s * 6364136223846793005ULL
       + 1442695040888963407ULL;
    return (double)(*s >> 11) * 0x1p-53;
}

#define RANDOMIZE(A, SCALE) do {                                 \
    unsigned char *bytes = (unsigned char *)&(A);                \
    for (size_t k_ = 0; k_ < sizeof(A); k_ += sizeof(float)) {    \
        float v_ = (float)((2 * uniform(&seed) - 1) * (SCALE));  \
        memcpy(bytes + k_, &v_, sizeof v_);                      \
    }                                                           \
} while (0)

void mamba_init(MambaModel *m, unsigned long long seed)
{
    if (!m)
        return;

    memset(m, 0, sizeof *m);
    RANDOMIZE(m->embed, 0.5);
    RANDOMIZE(m->head_w, 1 / sqrt((double)SSM_D_MODEL));

    for (int j = 0; j < SSM_D_MODEL; ++j)
        m->final_norm_w[j] = 1;

    for (int b = 0; b < SSM_N_BLOCKS; ++b) {
        MambaBlockParams *p = &m->blocks[b];

        RANDOMIZE(p->in_proj_w, 1 / sqrt((double)SSM_D_MODEL));
        RANDOMIZE(p->out_proj_w, 1 / sqrt((double)SSM_D_INNER));
        RANDOMIZE(p->x_proj_w, 1 / sqrt((double)SSM_D_INNER));
        RANDOMIZE(p->dt_proj_w, 1 / sqrt((double)SSM_DT_RANK));
        RANDOMIZE(p->conv_w, 1 / sqrt((double)SSM_CONV_K));
        RANDOMIZE(p->conv_b, 0.1);

        for (int j = 0; j < SSM_D_MODEL; ++j)
            p->norm_w[j] = 1;

        for (int i = 0; i < SSM_D_INNER; ++i) {
            p->D[i] = 1;

            double dt = exp(
                log(0.001) + uniform(&seed) * log(100.0));

            p->dt_proj_b[i] = (float)log(expm1(dt));

            for (int j = 0; j < SSM_D_STATE; ++j)
                p->A_log[i][j] = (float)log(j + 1.0);
        }
    }
}

int mamba_param_count(void)
{
    return 2 * SSM_VOCAB * SSM_D_MODEL + SSM_D_MODEL
         + SSM_N_BLOCKS * (
             SSM_D_INNER * SSM_D_STATE
             + SSM_D_INNER
             + SSM_P * SSM_D_INNER
             + SSM_D_INNER * SSM_DT_RANK
             + SSM_D_INNER
             + 3 * SSM_D_INNER * SSM_D_MODEL
             + SSM_D_INNER * SSM_CONV_K
             + SSM_D_INNER
             + SSM_D_MODEL);
}

void mamba_zero_grad(MambaModel *g)
{
    if (g)
        memset(g, 0, sizeof *g);
}

/* Cache may be NULL for inference. Each call starts with zero SSM state. */
int mamba_forward(
    const MambaModel *m, const int *ids, int T,
    float *logits, ForwardCache *cache)
{
    if (cache)
        cache->valid = 0;

    if (!m || !ids || !logits || T < 1 || T > SSM_MAX_T)
        return MAMBA_INVALID;

    for (int t = 0; t < T; ++t)
        if (ids[t] < 0 || ids[t] >= SSM_VOCAB)
            return MAMBA_INVALID;

    ForwardCache *c = cache ? cache : malloc(sizeof *c);
    if (!c)
        return MAMBA_NOMEM;

    c->valid = 0;
    c->T = T;
    c->owner = m;

    for (int t = 0; t < T; ++t) {
        c->ids[t] = ids[t];
        for (int j = 0; j < SSM_D_MODEL; ++j)
            c->h[0][t][j] = m->embed[ids[t]][j];
    }

    for (int b = 0; b < SSM_N_BLOCKS; ++b) {
        const MambaBlockParams *p = &m->blocks[b];
        double A[SSM_D_INNER][SSM_D_STATE];

        for (int i = 0; i < SSM_D_INNER; ++i)
            for (int j = 0; j < SSM_D_STATE; ++j)
                A[i][j] = -exp(p->A_log[i][j]);

        for (int t = 0; t < T; ++t) {
            norm_forward(
                c->h[b][t], p->norm_w, c->normed[b][t]);

            for (int o = 0; o < 2 * SSM_D_INNER; ++o) {
                double sum = 0;
                for (int j = 0; j < SSM_D_MODEL; ++j)
                    sum += p->in_proj_w[o][j]
                         * c->normed[b][t][j];
                c->proj[b][t][o] = sum;
            }

            for (int i = 0; i < SSM_D_INNER; ++i) {
                double sum = p->conv_b[i];

                /* Existing C11 convention: kernel k multiplies t-k. */
                for (int k = 0; k < SSM_CONV_K && k <= t; ++k)
                    sum += p->conv_w[i][k] * c->proj[b][t-k][i];

                c->conv[b][t][i] = sum;
                c->xc[b][t][i] = silu(sum);
            }

            /* Shared input-dependent dt-rank, B, and C projections. */
            for (int o = 0; o < SSM_P; ++o) {
                double sum = 0;
                for (int i = 0; i < SSM_D_INNER; ++i)
                    sum += p->x_proj_w[o][i] * c->xc[b][t][i];
                c->px[b][t][o] = sum;
            }

            for (int i = 0; i < SSM_D_INNER; ++i) {
                double raw = p->dt_proj_b[i];

                /* Bias is added exactly once. */
                for (int r = 0; r < SSM_DT_RANK; ++r)
                    raw += p->dt_proj_w[i][r] * c->px[b][t][r];

                double dt = softplus(raw);
                double x = c->xc[b][t][i];

                c->dt_raw[b][t][i] = raw;
                c->dt[b][t][i] = dt;

                double y = p->D[i] * x;

                for (int j = 0; j < SSM_D_STATE; ++j) {
                    double rho, q, qa;
                    discretize(A[i][j], dt, &rho, &q, &qa);

                    double prev =
                        t ? c->state[b][t-1][i][j] : 0;

                    double B = c->px[b][t][SSM_DT_RANK+j];
                    double C =
                        c->px[b][t][SSM_DT_RANK+SSM_D_STATE+j];

                    double h = rho * prev + q * B * x;
                    c->state[b][t][i][j] = h;
                    y += h * C;
                }

                c->ssm_y[b][t][i] = y;
                c->gated[b][t][i] =
                    y * silu(c->proj[b][t][SSM_D_INNER+i]);
            }

            for (int o = 0; o < SSM_D_MODEL; ++o) {
                double sum = c->h[b][t][o];

                for (int i = 0; i < SSM_D_INNER; ++i)
                    sum += p->out_proj_w[o][i]
                         * c->gated[b][t][i];

                c->h[b+1][t][o] = sum;
            }
        }
    }

    int status = MAMBA_OK;

    for (int t = 0; t < T; ++t) {
        norm_forward(
            c->h[SSM_N_BLOCKS][t],
            m->final_norm_w,
            c->normed_final[t]);

        for (int v = 0; v < SSM_VOCAB; ++v) {
            double sum = 0;

            for (int j = 0; j < SSM_D_MODEL; ++j)
                sum += m->head_w[v][j] * c->normed_final[t][j];

            c->logits[t][v] = sum;

            if (!isfinite(sum) || fabs(sum) > FLT_MAX)
                status = MAMBA_NONFINITE;

            logits[t * SSM_VOCAB + v] = (float)sum;
        }
    }

    c->valid = status == MAMBA_OK;

    if (!cache)
        free(c);

    return status;
}

/* Mean cross-entropy. Target -1 is ignored. */
int mamba_cross_entropy(
    const ForwardCache *c, const int *targets,
    double *loss, float *dlogits)
{
    if (!c || !c->valid || c->T < 1 || c->T > SSM_MAX_T ||
        !targets || !loss || !dlogits)
        return MAMBA_INVALID;

    int count = 0;

    for (int t = 0; t < c->T; ++t) {
        if (targets[t] < -1 || targets[t] >= SSM_VOCAB)
            return MAMBA_INVALID;
        count += targets[t] >= 0;
    }

    *loss = 0;
    memset(
        dlogits, 0,
        (size_t)c->T * SSM_VOCAB * sizeof *dlogits);

    if (!count)
        return MAMBA_OK;

    for (int t = 0; t < c->T; ++t) {
        if (targets[t] == -1)
            continue;

        double mx = c->logits[t][0], sum = 0;

        for (int v = 1; v < SSM_VOCAB; ++v)
            mx = fmax(mx, c->logits[t][v]);

        for (int v = 0; v < SSM_VOCAB; ++v)
            sum += exp(c->logits[t][v] - mx);

        *loss +=
            (mx - c->logits[t][targets[t]] + log(sum)) / count;

        for (int v = 0; v < SSM_VOCAB; ++v) {
            double p = exp(c->logits[t][v] - mx) / sum;
            dlogits[t * SSM_VOCAB + v] =
                (float)((p - (v == targets[t])) / count);
        }
    }

    return MAMBA_OK;
}

typedef struct {
    double dh[SSM_MAX_T][SSM_D_MODEL];
    double dx[SSM_MAX_T][SSM_D_INNER];
    double dy[SSM_MAX_T][SSM_D_INNER];
    double dp[SSM_MAX_T][2 * SSM_D_INNER];
    double dpx[SSM_MAX_T][SSM_P];
} BackwardWork;

/*
 * Accumulates into g.
 * Model parameters MUST remain unchanged since the cached forward.
 * Call mamba_zero_grad(g) before starting a new gradient accumulation.
 */
int mamba_backward(
    const MambaModel *m, MambaModel *g, const ForwardCache *c,
    const int *ids, const float *dlogits)
{
    if (!m || !g || m == g || !c || !c->valid ||
        c->owner != m || !ids || !dlogits ||
        c->T < 1 || c->T > SSM_MAX_T)
        return MAMBA_INVALID;

    int T = c->T;

    for (int t = 0; t < T; ++t) {
        if (ids[t] != c->ids[t])
            return MAMBA_INVALID;

        for (int v = 0; v < SSM_VOCAB; ++v)
            if (!isfinite(dlogits[t * SSM_VOCAB + v]))
                return MAMBA_NONFINITE;
    }

    BackwardWork *w = calloc(1, sizeof *w);
    if (!w)
        return MAMBA_NOMEM;

    /* Output head and final RMSNorm. */
    for (int t = 0; t < T; ++t) {
        double dn[SSM_D_MODEL] = {0};

        for (int v = 0; v < SSM_VOCAB; ++v) {
            double d = dlogits[t * SSM_VOCAB + v];

            for (int j = 0; j < SSM_D_MODEL; ++j) {
                g->head_w[v][j] +=
                    (float)(d * c->normed_final[t][j]);
                dn[j] += d * m->head_w[v][j];
            }
        }

        norm_backward(
            c->h[SSM_N_BLOCKS][t],
            m->final_norm_w, dn,
            w->dh[t], g->final_norm_w);
    }

    for (int b = SSM_N_BLOCKS - 1; b >= 0; --b) {
        const MambaBlockParams *p = &m->blocks[b];
        MambaBlockParams *gp = &g->blocks[b];

        memset(w->dx, 0, sizeof w->dx);
        memset(w->dp, 0, sizeof w->dp);
        memset(w->dpx, 0, sizeof w->dpx);

        /* Output projection and SiLU gate. */
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < SSM_D_INNER; ++i) {
                double dg = 0;

                for (int o = 0; o < SSM_D_MODEL; ++o) {
                    gp->out_proj_w[o][i] +=
                        (float)(w->dh[t][o] * c->gated[b][t][i]);

                    dg += p->out_proj_w[o][i] * w->dh[t][o];
                }

                double z = c->proj[b][t][SSM_D_INNER+i];

                w->dy[t][i] = dg * silu(z);
                w->dp[t][SSM_D_INNER+i] =
                    dg * c->ssm_y[b][t][i] * dsilu(z);
            }
        }

        /*
         * Reverse-time selective SSM.
         *
         * h = rho * previous + q * B * x
         * y = sum(h * C) + D * x
         *
         * B, C and dt-rank projections are shared across channels,
         * so their derivatives accumulate into shared dpx rows.
         */
        for (int i = 0; i < SSM_D_INNER; ++i) {
            double carry[SSM_D_STATE] = {0};
            double A[SSM_D_STATE];

            for (int j = 0; j < SSM_D_STATE; ++j)
                A[j] = -exp(p->A_log[i][j]);

            for (int t = T - 1; t >= 0; --t) {
                double dy = w->dy[t][i];
                double x = c->xc[b][t][i];
                double dt = c->dt[b][t][i];
                double ddt = 0;

                gp->D[i] += (float)(dy * x);
                w->dx[t][i] += dy * p->D[i];

                for (int j = 0; j < SSM_D_STATE; ++j) {
                    double rho, q, qa;
                    discretize(A[j], dt, &rho, &q, &qa);

                    double B = c->px[b][t][SSM_DT_RANK+j];
                    double C =
                        c->px[b][t][SSM_DT_RANK+SSM_D_STATE+j];

                    double prev =
                        t ? c->state[b][t-1][i][j] : 0;

                    double gh = carry[j] + dy * C;

                    /* Readout and input-dependent B projection. */
                    w->dpx[t][SSM_DT_RANK+SSM_D_STATE+j] +=
                        dy * c->state[b][t][i][j];

                    w->dpx[t][SSM_DT_RANK+j] += gh * q * x;

                    /* Direct input path through the recurrence. */
                    w->dx[t][i] += gh * q * B;

                    /* d(q)/d(dt) = rho. */
                    ddt += gh * rho * (prev * A[j] + B * x);

                    /* d(a)/d(A_log) = a. */
                    gp->A_log[i][j] +=
                        (float)(
                            gh * (prev * dt * rho + B * x * qa)
                            * A[j]);

                    carry[j] = gh * rho;
                }

                /* Delta softplus and low-rank delta projection. */
                double draw =
                    ddt * sigmoid(c->dt_raw[b][t][i]);

                gp->dt_proj_b[i] += (float)draw;

                for (int r = 0; r < SSM_DT_RANK; ++r) {
                    gp->dt_proj_w[i][r] +=
                        (float)(draw * c->px[b][t][r]);

                    w->dpx[t][r] += draw * p->dt_proj_w[i][r];
                }
            }
        }

        /* Backpropagate all three input-dependent projection slices. */
        for (int t = 0; t < T; ++t) {
            for (int o = 0; o < SSM_P; ++o) {
                for (int i = 0; i < SSM_D_INNER; ++i) {
                    gp->x_proj_w[o][i] +=
                        (float)(w->dpx[t][o] * c->xc[b][t][i]);

                    w->dx[t][i] +=
                        w->dpx[t][o] * p->x_proj_w[o][i];
                }
            }
        }

        /*
         * SiLU and causal convolution. Finish every convolution
         * contribution before differentiating the input projection.
         */
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < SSM_D_INNER; ++i) {
                double d =
                    w->dx[t][i] * dsilu(c->conv[b][t][i]);

                gp->conv_b[i] += (float)d;

                for (int k = 0; k < SSM_CONV_K && k <= t; ++k) {
                    gp->conv_w[i][k] +=
                        (float)(d * c->proj[b][t-k][i]);

                    w->dp[t-k][i] += d * p->conv_w[i][k];
                }
            }
        }

        /* Input projection, RMSNorm and residual identity path. */
        for (int t = 0; t < T; ++t) {
            double dn[SSM_D_MODEL] = {0};

            for (int o = 0; o < 2 * SSM_D_INNER; ++o) {
                for (int j = 0; j < SSM_D_MODEL; ++j) {
                    gp->in_proj_w[o][j] +=
                        (float)(
                            w->dp[t][o] * c->normed[b][t][j]);

                    dn[j] += w->dp[t][o] * p->in_proj_w[o][j];
                }
            }

            /* dh already holds the residual identity gradient. */
            norm_backward(
                c->h[b][t], p->norm_w,
                dn, w->dh[t], gp->norm_w);
        }
    }

    /* Repeated token IDs accumulate into the same embedding row. */
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < SSM_D_MODEL; ++j)
            g->embed[ids[t]][j] += (float)w->dh[t][j];

    free(w);
    return MAMBA_OK;
}

#ifdef MAMBA_DEMO

static unsigned long long rng = 7;

static int random_int(int n)
{
    rng = rng * 6364136223846793005ULL
        + 1442695040888963407ULL;
    return (int)((rng >> 32) % (unsigned)n);
}

int main(int argc, char **argv)
{
    long steps = 200;

    if (argc > 2) {
        fprintf(stderr, "usage: %s [steps]\n", argv[0]);
        return 1;
    }

    if (argc == 2) {
        char *end;
        errno = 0;
        steps = strtol(argv[1], &end, 10);

        if (errno || *end || steps < 1 || steps > 1000000)
            return 1;
    }

    if (SSM_VOCAB < 7 || SSM_MAX_T < 10)
        return 1;

    MambaModel *m = calloc(1, sizeof *m);
    MambaModel *g = calloc(1, sizeof *g);
    ForwardCache *cache = calloc(1, sizeof *cache);

    size_t n = (size_t)mamba_param_count();
    double *first = calloc(n, sizeof *first);
    double *second = calloc(n, sizeof *second);

    if (!m || !g || !cache || !first || !second ||
        sizeof *m != n * sizeof(float)) {
        free(m);
        free(g);
        free(cache);
        free(first);
        free(second);
        return 1;
    }

    mamba_init(m, 123);
    printf("parameters: %d\n", mamba_param_count());

    float logits[10 * SSM_VOCAB];
    float dlogits[10 * SSM_VOCAB];
    double average = 0, beta1 = 1, beta2 = 1;
    int status = 0;

    for (long step = 1; step <= steps; ++step) {
        int ids[10], targets[10], bits[8];
        int k = 1 + random_int(4);

        /* [rotation-token, separator, eight input bits] */
        ids[0] = k + 2;
        ids[1] = 2;
        targets[0] = targets[1] = -1;

        for (int t = 0; t < 8; ++t)
            ids[t+2] = bits[t] = random_int(2);

        for (int t = 0; t < 8; ++t)
            targets[t+2] = bits[(t-k+8) % 8];

        double loss;

        if (mamba_forward(m, ids, 10, logits, cache) != MAMBA_OK ||
            mamba_cross_entropy(
                cache, targets, &loss, dlogits) != MAMBA_OK) {
            status = 1;
            break;
        }

        mamba_zero_grad(g);

        if (mamba_backward(m, g, cache, ids, dlogits) != MAMBA_OK) {
            status = 1;
            break;
        }

        beta1 *= 0.9;
        beta2 *= 0.999;

        /* Adam; byte access avoids indexing across struct subobjects. */
        for (size_t i = 0; i < n; ++i) {
            float p, d;

            memcpy(
                &p, (unsigned char *)m + i * sizeof(float),
                sizeof p);
            memcpy(
                &d, (unsigned char *)g + i * sizeof(float),
                sizeof d);

            if (!isfinite(d)) {
                status = 1;
                break;
            }

            first[i] = 0.9 * first[i] + 0.1 * d;
            second[i] =
                0.999 * second[i] + 0.001 * (double)d * d;

            p -= (float)(
                0.003 * (first[i] / (1 - beta1)) /
                (sqrt(second[i] / (1 - beta2)) + 1e-8));

            memcpy(
                (unsigned char *)m + i * sizeof(float),
                &p, sizeof p);
        }

        if (status)
            break;

        average += loss;

        if (step % 50 == 0 || step == steps) {
            long count = step % 50 ? step % 50 : 50;
            printf(
                "step %ld mean_loss %.6f\n",
                step, average / count);
            average = 0;
        }
    }

    free(second);
    free(first);
    free(cache);
    free(g);
    free(m);
    return status;
}
#endif
