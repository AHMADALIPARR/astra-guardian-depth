// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
// Controlled Mamba experiment: 5 seeds, balanced k, matched exposure.
// Tracks per-k loss, visible/hidden accuracy, gradient norm.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SSM_D_MODEL 64
#define SSM_N_BLOCKS 3
#define SSM_D_STATE 16
#define SSM_DT_RANK 8
#define SSM_D_INNER (2 * SSM_D_MODEL)
#define SSM_VOCAB 7
#define SSM_CONV_K 4
#define SSM_MAX_T 128
#define SSM_P (SSM_DT_RANK + 2 * SSM_D_STATE)

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
    double state[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER][SSM_D_STATE];
    double ssm_y[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER];
    double gated[SSM_N_BLOCKS][SSM_MAX_T][SSM_D_INNER];
    double normed_final[SSM_MAX_T][SSM_D_MODEL];
    double logits[SSM_MAX_T][SSM_VOCAB];
} ForwardCache;

enum { MAMBA_OK = 0 };

void mamba_init(MambaModel *m, unsigned long long seed);
int mamba_param_count(void);
void mamba_zero_grad(MambaModel *g);
int mamba_forward(const MambaModel *m, const int *ids, int T, float *logits, ForwardCache *cache);
int mamba_cross_entropy(const ForwardCache *c, const int *targets, double *loss, float *dlogits);
int mamba_backward(const MambaModel *m, MambaModel *g, const ForwardCache *c, const int *ids, const float *dlogits);

static unsigned long long rng;
static int rnd(int n) {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (int)((rng >> 32) % (unsigned)n);
}

// Fixed eval set: 200 examples per k, generated with seed 999
#define EVAL_PER_K 200
static int eval_ids[4][EVAL_PER_K][10];
static int eval_bits[4][EVAL_PER_K][8];

static void gen_eval_set(void) {
    rng = 999;
    for (int k = 1; k <= 4; k++) {
        for (int ex = 0; ex < EVAL_PER_K; ex++) {
            eval_ids[k-1][ex][0] = k + 2;
            eval_ids[k-1][ex][1] = 2;
            for (int t = 0; t < 8; t++) {
                int b = rnd(2);
                eval_bits[k-1][ex][t] = b;
                eval_ids[k-1][ex][2+t] = b;
            }
        }
    }
}

// Evaluate: per-k visible accuracy, hidden accuracy, loss
static void evaluate(MambaModel *m, ForwardCache *cache,
                     double *vis_acc, double *hid_acc, double *loss_per_k) {
    float logits[10 * SSM_VOCAB];
    for (int k = 1; k <= 4; k++) {
        int vis_ok = 0, vis_n = 0, hid_ok = 0, hid_n = 0;
        double loss_sum = 0;
        for (int ex = 0; ex < EVAL_PER_K; ex++) {
            int *ids = eval_ids[k-1][ex];
            int *bits = eval_bits[k-1][ex];
            if (mamba_forward(m, ids, 10, logits, NULL) != MAMBA_OK) continue;
            for (int pos = 0; pos < 8; pos++) {
                int need = (pos - k + 40) % 8;
                int visible = (2 + need) <= (2 + pos);
                int want = bits[need];
                int pred = 0;
                float best = logits[(2+pos)*SSM_VOCAB];
                for (int v = 1; v < SSM_VOCAB; v++)
                    if (logits[(2+pos)*SSM_VOCAB+v] > best) {
                        best = logits[(2+pos)*SSM_VOCAB+v]; pred = v;
                    }
                if (visible) { vis_n++; if (pred == want) vis_ok++; }
                else { hid_n++; if (pred == want) hid_ok++; }
            }
            // loss
            int targets[10] = {-1, -1, 0,0,0,0,0,0,0,0};
            for (int t = 0; t < 8; t++) targets[2+t] = bits[(t-k+8)%8];
            ForwardCache *c2 = cache; // reuse
            double loss;
            float dl[10*SSM_VOCAB];
            if (mamba_forward(m, ids, 10, logits, c2) == MAMBA_OK &&
                mamba_cross_entropy(c2, targets, &loss, dl) == MAMBA_OK)
                loss_sum += loss;
        }
        vis_acc[k-1] = vis_n ? (double)vis_ok/vis_n : 0;
        hid_acc[k-1] = hid_n ? (double)hid_ok/hid_n : 0;
        loss_per_k[k-1] = loss_sum / EVAL_PER_K;
    }
}

int main(void) {
    gen_eval_set();
    int seeds[5] = {100, 200, 300, 400, 500};
    int batch = 32; // 8 per k, balanced
    int steps = 2000;

    printf("seed,step,examples,k,loss,vis_acc,hid_acc,grad_norm\n");

    for (int s = 0; s < 5; s++) {
        rng = seeds[s];
        MambaModel *m = calloc(1, sizeof *m);
        MambaModel *g = calloc(1, sizeof *g);
        MambaModel *gacc = calloc(1, sizeof *gacc); // accumulated grads
        ForwardCache *cache = calloc(1, sizeof *cache);
        size_t n = (size_t)mamba_param_count();
        double *first = calloc(n, sizeof *first);
        double *second = calloc(n, sizeof *second);
        mamba_init(m, seeds[s]);

        float logits[10 * SSM_VOCAB], dlogits[10 * SSM_VOCAB];
        double beta1 = 1, beta2 = 1;
        long examples = 0;

        for (long step = 1; step <= steps; step++) {
            mamba_zero_grad(gacc);
            double gnorm2 = 0;
            // Balanced batch: 8 examples per k
            for (int bi = 0; bi < batch; bi++) {
                int k = 1 + (bi % 4); // balanced
                int ids[10], targets[10], bits[8];
                ids[0] = k + 2; ids[1] = 2;
                targets[0] = targets[1] = -1;
                for (int t = 0; t < 8; t++) ids[t+2] = bits[t] = rnd(2);
                for (int t = 0; t < 8; t++) targets[t+2] = bits[(t-k+8)%8];
                double loss;
                mamba_forward(m, ids, 10, logits, cache);
                mamba_cross_entropy(cache, targets, &loss, dlogits);
                mamba_zero_grad(g);
                mamba_backward(m, g, cache, ids, dlogits);
                // accumulate
                float *ga = (float*)gacc, *gi = (float*)g;
                for (size_t i = 0; i < n; i++) ga[i] += gi[i] / batch;
                examples++;
            }
            // grad norm
            float *ga = (float*)gacc;
            for (size_t i = 0; i < n; i++) gnorm2 += (double)ga[i]*ga[i];
            double gnorm = sqrt(gnorm2);

            // Adam update
            beta1 *= 0.9; beta2 *= 0.999;
            float *mp = (float*)m;
            for (size_t i = 0; i < n; i++) {
                float d = ga[i];
                first[i] = 0.9*first[i] + 0.1*d;
                second[i] = 0.999*second[i] + 0.001*(double)d*d;
                mp[i] -= (float)(0.003 * (first[i]/(1-beta1)) / (sqrt(second[i]/(1-beta2)) + 1e-8));
            }

            if (step % 500 == 0 || step == 1) {
                double va[4], ha[4], lp[4];
                evaluate(m, cache, va, ha, lp);
                for (int k = 1; k <= 4; k++)
                    printf("%d,%ld,%ld,%d,%.4f,%.3f,%.3f,%.2f\n",
                           seeds[s], step, examples, k, lp[k-1], va[k-1], ha[k-1], gnorm);
                fflush(stdout);
            }
        }
        free(m); free(g); free(gacc); free(cache); free(first); free(second);
    }
    return 0;
}
