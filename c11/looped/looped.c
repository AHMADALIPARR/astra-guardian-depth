// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

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

_Static_assert(
    LT_D_MODEL > 0 && LT_N_HEADS > 0,
    "positive attention dimensions required");
_Static_assert(
    LT_D_MODEL % LT_N_HEADS == 0,
    "heads must divide model width");
_Static_assert(
    LT_D_FF > 0 && LT_VOCAB > 0 &&
    LT_MAX_T > 0 && LT_R_LOOP > 0,
    "positive model dimensions required");

/* Parameter layout matches the committed C11 transformer. */
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

/* Every parameter has an identically shaped gradient tensor. */
typedef TransformerBlock TransformerBlockGrad;
typedef LoopedModel LoopedGrad;

/* Values and their gradients have identical tensor layouts. */
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

/* Allocate on the heap: this contains all recurrent activation tapes. */
typedef struct {
    int T, rounds, valid;
    const LoopedModel *owner;
    int ids[LT_MAX_T];

    /* block[0] = prelude; block[r+1] = shared execution r. */
    BlockTensors block[LT_R_LOOP + 1];

    double normed_final[LT_MAX_T][LT_D_MODEL];
    double logits[LT_MAX_T][LT_VOCAB];
} LoopedCache;

enum {
    LT_OK = 0,
    LT_INVALID = -1,
    LT_NOMEM = -2,
    LT_NONFINITE = -3
};

static double sigmoid(double x)
{
    double e = exp(-fabs(x));
    return x >= 0 ? 1 / (1 + e) : e / (1 + e);
}

static double silu(double x)
{
    return x * sigmoid(x);
}

static double dsilu(double x)
{
    double s = sigmoid(x);
    return s + x * s * (1 - s);
}

static double rms_inverse(const double *x)
{
    double sum = 0;

    for (int i = 0; i < LT_D_MODEL; ++i)
        sum += x[i] * x[i];

    return 1 / sqrt(sum / LT_D_MODEL + LT_EPS);
}

static void norm_forward(
    const double *x,
    const float *w,
    double *y)
{
    double inv = rms_inverse(x);

    for (int i = 0; i < LT_D_MODEL; ++i)
        y[i] = x[i] * w[i] * inv;
}

static void norm_backward(
    const double *x,
    const float *w,
    const double *dy,
    double *dx,
    float *dw)
{
    double inv = rms_inverse(x);
    double dot = 0;

    for (int i = 0; i < LT_D_MODEL; ++i)
        dot += dy[i] * w[i] * x[i];

    for (int i = 0; i < LT_D_MODEL; ++i) {
        dw[i] += (float)(dy[i] * x[i] * inv);

        dx[i] += dy[i] * w[i] * inv
               - x[i] * dot * inv * inv * inv / LT_D_MODEL;
    }
}

static void linear_forward(
    int out,
    int in,
    const float w[out][in],
    const float *bias,
    const double *x,
    double *y)
{
    for (int o = 0; o < out; ++o) {
        double sum = bias ? bias[o] : 0;

        for (int i = 0; i < in; ++i)
            sum += w[o][i] * x[i];

        y[o] = sum;
    }
}

static void linear_backward(
    int out,
    int in,
    const float w[out][in],
    const double *x,
    const double *dy,
    double *dx,
    float dw[out][in],
    float *db)
{
    for (int o = 0; o < out; ++o) {
        if (db)
            db[o] += (float)dy[o];

        for (int i = 0; i < in; ++i) {
            dw[o][i] += (float)(dy[o] * x[i]);
            dx[i] += w[o][i] * dy[o];
        }
    }
}

static void block_forward(
    const TransformerBlock *p,
    BlockTensors *a,
    int T)
{
    memset(a->probability, 0, sizeof a->probability);
    memset(a->context, 0, sizeof a->context);

    for (int t = 0; t < T; ++t) {
        norm_forward(
            a->input[t], p->attn_norm_w, a->norm1[t]);

        linear_forward(
            3 * LT_D_MODEL, LT_D_MODEL,
            p->qkv_w, p->qkv_b,
            a->norm1[t], a->qkv[t]);
    }

    double scale = 1 / sqrt((double)LT_D_HEAD);

    for (int head = 0; head < LT_N_HEADS; ++head) {
        int base = head * LT_D_HEAD;

        for (int t = 0; t < T; ++t) {
            double *prob = a->probability[head][t];
            double mx = -INFINITY;
            double sum = 0;

            for (int s = 0; s <= t; ++s) {
                double dot = 0;

                for (int j = 0; j < LT_D_HEAD; ++j)
                    dot += a->qkv[t][base+j]
                         * a->qkv[s][LT_D_MODEL+base+j];

                prob[s] = dot * scale;
                mx = fmax(mx, prob[s]);
            }

            for (int s = 0; s <= t; ++s) {
                prob[s] = exp(prob[s] - mx);
                sum += prob[s];
            }

            for (int s = 0; s <= t; ++s)
                prob[s] /= sum;

            for (int j = 0; j < LT_D_HEAD; ++j)
                for (int s = 0; s <= t; ++s)
                    a->context[t][base+j] +=
                        prob[s]
                        * a->qkv[s][2 * LT_D_MODEL+base+j];
        }
    }

    for (int t = 0; t < T; ++t) {
        linear_forward(
            LT_D_MODEL, LT_D_MODEL,
            p->out_w, p->out_b,
            a->context[t], a->middle[t]);

        for (int i = 0; i < LT_D_MODEL; ++i)
            a->middle[t][i] += a->input[t][i];

        norm_forward(
            a->middle[t], p->mlp_norm_w, a->norm2[t]);

        linear_forward(
            LT_D_FF, LT_D_MODEL,
            p->gate_w, NULL,
            a->norm2[t], a->gate[t]);

        linear_forward(
            LT_D_FF, LT_D_MODEL,
            p->up_w, NULL,
            a->norm2[t], a->up[t]);

        for (int j = 0; j < LT_D_FF; ++j)
            a->product[t][j] =
                silu(a->gate[t][j]) * a->up[t][j];

        linear_forward(
            LT_D_MODEL, LT_D_FF,
            p->down_w, NULL,
            a->product[t], a->output[t]);

        for (int i = 0; i < LT_D_MODEL; ++i)
            a->output[t][i] += a->middle[t][i];
    }
}

/*
 * d mirrors a.
 * Caller zeros d, supplies d.output, and receives d.input.
 * Parameter gradients accumulate into g.
 */
static void block_backward(
    const TransformerBlock *p,
    TransformerBlockGrad *g,
    const BlockTensors *a,
    BlockTensorGrad *d,
    int T)
{
    /* SwiGLU, output residual, and second normalization. */
    for (int t = 0; t < T; ++t) {
        memcpy(
            d->middle[t], d->output[t],
            sizeof d->middle[t]);

        linear_backward(
            LT_D_MODEL, LT_D_FF,
            p->down_w,
            a->product[t], d->output[t],
            d->product[t], g->down_w, NULL);

        for (int j = 0; j < LT_D_FF; ++j) {
            d->gate[t][j] =
                d->product[t][j]
                * a->up[t][j]
                * dsilu(a->gate[t][j]);

            d->up[t][j] =
                d->product[t][j] * silu(a->gate[t][j]);
        }

        linear_backward(
            LT_D_FF, LT_D_MODEL,
            p->gate_w,
            a->norm2[t], d->gate[t],
            d->norm2[t], g->gate_w, NULL);

        linear_backward(
            LT_D_FF, LT_D_MODEL,
            p->up_w,
            a->norm2[t], d->up[t],
            d->norm2[t], g->up_w, NULL);

        norm_backward(
            a->middle[t], p->mlp_norm_w,
            d->norm2[t], d->middle[t],
            g->mlp_norm_w);

        /* Identity branch around attention. */
        memcpy(
            d->input[t], d->middle[t],
            sizeof d->input[t]);

        linear_backward(
            LT_D_MODEL, LT_D_MODEL,
            p->out_w,
            a->context[t], d->middle[t],
            d->context[t], g->out_w, g->out_b);
    }

    double scale = 1 / sqrt((double)LT_D_HEAD);

    /* Causal multi-head attention backward. */
    for (int head = 0; head < LT_N_HEADS; ++head) {
        int base = head * LT_D_HEAD;

        for (int t = 0; t < T; ++t) {
            double dot = 0;
            const double *prob = a->probability[head][t];
            double *dp = d->probability[head][t];

            /*
             * context[t] = sum_s probability[t,s] * value[s]
             * dP = dContext dot V
             * dV += P * dContext
             */
            for (int s = 0; s <= t; ++s) {
                for (int j = 0; j < LT_D_HEAD; ++j) {
                    dp[s] +=
                        d->context[t][base+j]
                        * a->qkv[s][2 * LT_D_MODEL+base+j];

                    d->qkv[s][2 * LT_D_MODEL+base+j] +=
                        prob[s] * d->context[t][base+j];
                }

                dot += prob[s] * dp[s];
            }

            /*
             * Softmax vector-Jacobian product:
             * dScore = P * (dP - sum(P*dP)).
             *
             * Include attention scale when propagating to Q/K.
             * Masked future positions are never visited.
             */
            for (int s = 0; s <= t; ++s) {
                double ds = prob[s] * (dp[s] - dot) * scale;

                for (int j = 0; j < LT_D_HEAD; ++j) {
                    d->qkv[t][base+j] +=
                        ds * a->qkv[s][LT_D_MODEL+base+j];

                    d->qkv[s][LT_D_MODEL+base+j] +=
                        ds * a->qkv[t][base+j];
                }
            }
        }
    }

    /* QKV projection and first normalization. */
    for (int t = 0; t < T; ++t) {
        linear_backward(
            3 * LT_D_MODEL, LT_D_MODEL,
            p->qkv_w,
            a->norm1[t], d->qkv[t],
            d->norm1[t], g->qkv_w, g->qkv_b);

        norm_backward(
            a->input[t], p->attn_norm_w,
            d->norm1[t], d->input[t],
            g->attn_norm_w);
    }
}

static double uniform(unsigned long long *s)
{
    *s = *s * 6364136223846793005ULL
       + 1442695040888963407ULL;

    return (double)(*s >> 11) * 0x1p-53;
}

#define RANDOMIZE(A, SCALE) do {                              \
    unsigned char *bytes_ = (unsigned char *)&(A);            \
    for (size_t i_ = 0; i_ < sizeof(A); i_ += sizeof(float)) { \
        float v_ = (float)(                                  \
            (2 * uniform(&seed) - 1) * (SCALE));              \
        memcpy(bytes_ + i_, &v_, sizeof v_);                  \
    }                                                        \
} while (0)

void looped_init(LoopedModel *m, unsigned long long seed)
{
    if (!m)
        return;

    memset(m, 0, sizeof *m);

    RANDOMIZE(m->embed, 0.3);
    RANDOMIZE(m->pos_emb, 0.05);
    RANDOMIZE(m->head_w, 1 / sqrt((double)LT_D_MODEL));
    RANDOMIZE(m->inject_w, 0.1 / sqrt((double)LT_D_MODEL));

    TransformerBlock *blocks[2] = {
        &m->prelude, &m->shared
    };

    for (int b = 0; b < 2; ++b) {
        TransformerBlock *p = blocks[b];

        RANDOMIZE(p->qkv_w, 1 / sqrt((double)LT_D_MODEL));
        RANDOMIZE(p->out_w, 1 / sqrt((double)LT_D_MODEL));
        RANDOMIZE(p->gate_w, 1 / sqrt((double)LT_D_MODEL));
        RANDOMIZE(p->up_w, 1 / sqrt((double)LT_D_MODEL));
        RANDOMIZE(p->down_w, 1 / sqrt((double)LT_D_FF));

        for (int i = 0; i < LT_D_MODEL; ++i) {
            p->attn_norm_w[i] = 1;
            p->mlp_norm_w[i] = 1;
        }
    }

    for (int i = 0; i < LT_D_MODEL; ++i)
        m->final_norm_w[i] = 1;
}

int looped_param_count(void)
{
    return
        2 * LT_VOCAB * LT_D_MODEL
        + LT_MAX_T * LT_D_MODEL
        + LT_D_MODEL * LT_D_MODEL
        + LT_D_MODEL
        + 2 * (
            4 * LT_D_MODEL * LT_D_MODEL
            + 6 * LT_D_MODEL
            + 3 * LT_D_FF * LT_D_MODEL);
}

void looped_zero_grad(LoopedGrad *g)
{
    if (g)
        memset(g, 0, sizeof *g);
}

int looped_forward_r(
    const LoopedModel *m,
    const int *ids,
    int T,
    int rounds,
    float *logits,
    LoopedCache *cache)
{
    if (cache)
        cache->valid = 0;

    if (!m || !ids || !logits ||
        T < 1 || T > LT_MAX_T ||
        rounds < 1 || rounds > LT_R_LOOP)
        return LT_INVALID;

    for (int t = 0; t < T; ++t)
        if (ids[t] < 0 || ids[t] >= LT_VOCAB)
            return LT_INVALID;

    LoopedCache *c = cache ? cache : malloc(sizeof *c);
    if (!c)
        return LT_NOMEM;

    c->T = T;
    c->rounds = rounds;
    c->owner = m;
    c->valid = 0;

    for (int t = 0; t < T; ++t) {
        c->ids[t] = ids[t];

        for (int j = 0; j < LT_D_MODEL; ++j)
            c->block[0].input[t][j] =
                (double)m->embed[ids[t]][j]
                + m->pos_emb[t][j];
    }

    block_forward(&m->prelude, &c->block[0], T);

    for (int r = 0; r < rounds; ++r) {
        for (int t = 0; t < T; ++t) {
            linear_forward(
                LT_D_MODEL, LT_D_MODEL,
                m->inject_w, NULL,
                c->block[0].output[t],
                c->block[r+1].input[t]);

            for (int j = 0; j < LT_D_MODEL; ++j)
                c->block[r+1].input[t][j] +=
                    c->block[r].output[t][j];
        }

        block_forward(
            &m->shared, &c->block[r+1], T);
    }

    int status = LT_OK;

    for (int t = 0; t < T; ++t) {
        norm_forward(
            c->block[rounds].output[t],
            m->final_norm_w,
            c->normed_final[t]);

        linear_forward(
            LT_VOCAB, LT_D_MODEL,
            m->head_w, NULL,
            c->normed_final[t],
            c->logits[t]);

        for (int v = 0; v < LT_VOCAB; ++v) {
            double x = c->logits[t][v];

            if (!isfinite(x) || fabs(x) > FLT_MAX)
                status = LT_NONFINITE;

            logits[t * LT_VOCAB + v] = (float)x;
        }
    }

    c->valid = status == LT_OK;

    if (!cache)
        free(c);

    return status;
}

int looped_forward(
    const LoopedModel *m,
    const int *ids,
    int T,
    float *logits,
    LoopedCache *cache)
{
    return looped_forward_r(
        m, ids, T, LT_R_LOOP, logits, cache);
}

typedef struct {
    BlockTensorGrad block;
    double upstream[LT_MAX_T][LT_D_MODEL];
    double prompt[LT_MAX_T][LT_D_MODEL];
} BackwardWorkspace;

/*
 * Accumulates parameter gradients into g.
 * Model weights must remain unchanged since the cached forward.
 */
int looped_backward(
    const LoopedModel *m,
    LoopedGrad *g,
    const LoopedCache *c,
    const int *ids,
    const float *dlogits)
{
    if (!m || !g ||
        (const void *)m == (const void *)g ||
        !c || !c->valid || c->owner != m ||
        !ids || !dlogits ||
        c->T < 1 || c->T > LT_MAX_T ||
        c->rounds < 1 || c->rounds > LT_R_LOOP)
        return LT_INVALID;

    int T = c->T;

    for (int t = 0; t < T; ++t) {
        if (ids[t] != c->ids[t])
            return LT_INVALID;

        for (int v = 0; v < LT_VOCAB; ++v)
            if (!isfinite(dlogits[t * LT_VOCAB + v]))
                return LT_NONFINITE;
    }

    BackwardWorkspace *w = calloc(1, sizeof *w);
    if (!w)
        return LT_NOMEM;

    /* Head and final RMSNorm. */
    for (int t = 0; t < T; ++t) {
        double dn[LT_D_MODEL] = {0};
        double dy[LT_VOCAB];

        for (int v = 0; v < LT_VOCAB; ++v)
            dy[v] = dlogits[t * LT_VOCAB + v];

        linear_backward(
            LT_VOCAB, LT_D_MODEL,
            m->head_w,
            c->normed_final[t], dy,
            dn, g->head_w, NULL);

        norm_backward(
            c->block[c->rounds].output[t],
            m->final_norm_w,
            dn, w->upstream[t],
            g->final_norm_w);
    }

    /* Reverse through the distinct activation tape for every pass. */
    for (int r = c->rounds; r >= 1; --r) {
        memset(&w->block, 0, sizeof w->block);

        for (int t = 0; t < T; ++t)
            memcpy(
                w->block.output[t],
                w->upstream[t],
                sizeof w->upstream[t]);

        /* Every pass adds to the SAME shared gradient tensors. */
        block_backward(
            &m->shared,
            &g->shared,
            &c->block[r],
            &w->block,
            T);

        for (int t = 0; t < T; ++t) {
            /*
             * hin = hprevious + Winject * h0
             *
             * dW       += dhin outer h0
             * dh0      += transpose(Winject) * dhin
             * dhprev    = dhin
             */
            linear_backward(
                LT_D_MODEL, LT_D_MODEL,
                m->inject_w,
                c->block[0].output[t],
                w->block.input[t],
                w->prompt[t],
                g->inject_w,
                NULL);

            memcpy(
                w->upstream[t],
                w->block.input[t],
                sizeof w->upstream[t]);
        }
    }

    /*
     * The prelude receives both:
     * 1. the ordinary recurrent-path gradient;
     * 2. the sum of all h0 reinjection gradients.
     */
    memset(&w->block, 0, sizeof w->block);

    for (int t = 0; t < T; ++t)
        for (int j = 0; j < LT_D_MODEL; ++j)
            w->block.output[t][j] =
                w->upstream[t][j] + w->prompt[t][j];

    block_backward(
        &m->prelude,
        &g->prelude,
        &c->block[0],
        &w->block,
        T);

    /* Repeated token IDs accumulate into their embedding rows. */
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < LT_D_MODEL; ++j) {
            float d = (float)w->block.input[t][j];

            g->embed[ids[t]][j] += d;
            g->pos_emb[t][j] += d;
        }
    }

    free(w);
    return LT_OK;
}

/* Mean cross-entropy; target -1 is ignored. */
int looped_cross_entropy(
    const LoopedCache *c,
    const int *targets,
    double *loss,
    float *dlogits)
{
    if (!c || !c->valid ||
        c->T < 1 || c->T > LT_MAX_T ||
        !targets || !loss || !dlogits)
        return LT_INVALID;

    int count = 0;

    for (int t = 0; t < c->T; ++t) {
        if (targets[t] < -1 || targets[t] >= LT_VOCAB)
            return LT_INVALID;

        count += targets[t] >= 0;
    }

    *loss = 0;

    memset(
        dlogits, 0,
        (size_t)c->T * LT_VOCAB * sizeof *dlogits);

    if (!count)
        return LT_OK;

    for (int t = 0; t < c->T; ++t) {
        if (targets[t] < 0)
            continue;

        double mx = c->logits[t][0];
        double sum = 0;

        for (int v = 1; v < LT_VOCAB; ++v)
            mx = fmax(mx, c->logits[t][v]);

        for (int v = 0; v < LT_VOCAB; ++v)
            sum += exp(c->logits[t][v] - mx);

        *loss +=
            (mx - c->logits[t][targets[t]] + log(sum))
            / count;

        for (int v = 0; v < LT_VOCAB; ++v) {
            double p = exp(c->logits[t][v] - mx) / sum;

            dlogits[t * LT_VOCAB + v] =
                (float)((p - (v == targets[t])) / count);
        }
    }

    return LT_OK;
}
