// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
// C11 Mamba evaluation: per-k exact/bit accuracy after training.
// Uses the user's verbatim mamba.c (no MAMBA_DEMO).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Include the user's implementation directly (declarations)
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

enum { MAMBA_OK = 0, MAMBA_INVALID = -1, MAMBA_NOMEM = -2, MAMBA_NONFINITE = -3 };

void mamba_init(MambaModel *m, unsigned long long seed);
int mamba_param_count(void);
void mamba_zero_grad(MambaModel *g);
int mamba_forward(const MambaModel *m, const int *ids, int T, float *logits, ForwardCache *cache);
int mamba_cross_entropy(const ForwardCache *c, const int *targets, double *loss, float *dlogits);
int mamba_backward(const MambaModel *m, MambaModel *g, const ForwardCache *c, const int *ids, const float *dlogits);

static unsigned long long rng = 99;
static int rnd(int n) {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (int)((rng >> 32) % (unsigned)n);
}

int main(void) {
    MambaModel *m = calloc(1, sizeof *m);
    MambaModel *g = calloc(1, sizeof *g);
    ForwardCache *cache = calloc(1, sizeof *cache);
    size_t n = (size_t)mamba_param_count();
    double *first = calloc(n, sizeof *first);
    double *second = calloc(n, sizeof *second);
    mamba_init(m, 123);

    float logits[10 * SSM_VOCAB], dlogits[10 * SSM_VOCAB];
    double beta1 = 1, beta2 = 1;

    // Train 2000 steps
    for (long step = 1; step <= 2000; step++) {
        int ids[10], targets[10], bits[8];
        int k = 1 + rnd(4);
        ids[0] = k + 2; ids[1] = 2;
        targets[0] = targets[1] = -1;
        for (int t = 0; t < 8; t++) ids[t+2] = bits[t] = rnd(2);
        for (int t = 0; t < 8; t++) targets[t+2] = bits[(t-k+8)%8];
        double loss;
        if (mamba_forward(m, ids, 10, logits, cache) != MAMBA_OK) return 1;
        if (mamba_cross_entropy(cache, targets, &loss, dlogits) != MAMBA_OK) return 1;
        mamba_zero_grad(g);
        if (mamba_backward(m, g, cache, ids, dlogits) != MAMBA_OK) return 1;
        beta1 *= 0.9; beta2 *= 0.999;
        for (size_t i = 0; i < n; i++) {
            float p, d;
            memcpy(&p, (unsigned char*)m + i*sizeof(float), sizeof p);
            memcpy(&d, (unsigned char*)g + i*sizeof(float), sizeof d);
            first[i] = 0.9*first[i] + 0.1*d;
            second[i] = 0.999*second[i] + 0.001*(double)d*d;
            p -= (float)(0.003 * (first[i]/(1-beta1)) / (sqrt(second[i]/(1-beta2)) + 1e-8));
            memcpy((unsigned char*)m + i*sizeof(float), &p, sizeof p);
        }
    }
    printf("trained 2000 steps\n");

    // Evaluate per-k
    printf("\nbenchmark: rotate-right-by-k accuracy (exact / bit)\n");
    printf("k    mamba\n");
    for (int k = 1; k <= 4; k++) {
        int exact_ok = 0, bit_ok = 0, nex = 500;
        for (int ex = 0; ex < nex; ex++) {
            int ids[10], bits[8];
            ids[0] = k + 2; ids[1] = 2;
            for (int t = 0; t < 8; t++) ids[t+2] = bits[t] = rnd(2);
            if (mamba_forward(m, ids, 10, logits, NULL) != MAMBA_OK) return 1;
            int exact = 1;
            for (int t = 0; t < 8; t++) {
                int want = bits[(t-k+8)%8];
                int pred = 0;
                float best = logits[(t+2)*SSM_VOCAB];
                for (int v = 1; v < SSM_VOCAB; v++)
                    if (logits[(t+2)*SSM_VOCAB+v] > best) {
                        best = logits[(t+2)*SSM_VOCAB+v]; pred = v;
                    }
                if (pred == want) bit_ok++;
                else exact = 0;
            }
            if (exact) exact_ok++;
        }
        printf("%d    %.3f / %.3f\n", k, (double)exact_ok/nex, (double)bit_ok/(nex*8));
    }

    free(m); free(g); free(cache); free(first); free(second);
    return 0;
}
