// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
// Gradient check: numerical vs analytical for C11 Mamba.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../ssm/ssm.h"

static float loss_fn(MambaModel *m, const int *ids, const int *tgt, int T,
                     ForwardCache *c) {
    float *logits = malloc(T * SSM_VOCAB * sizeof(float));
    mamba_forward(m, ids, T, logits, c);
    float loss = 0;
    for (int t = 0; t < T; t++) {
        float mx = logits[t*SSM_VOCAB];
        for (int i = 1; i < SSM_VOCAB; i++)
            if (logits[t*SSM_VOCAB+i] > mx) mx = logits[t*SSM_VOCAB+i];
        float sum = 0;
        for (int i = 0; i < SSM_VOCAB; i++) sum += expf(logits[t*SSM_VOCAB+i]-mx);
        loss -= logits[t*SSM_VOCAB+tgt[t]] - mx - logf(sum);
    }
    free(logits);
    return loss / T;
}

int main(void) {
    MambaModel *m = calloc(1, sizeof(MambaModel));
    MambaModel *g = calloc(1, sizeof(MambaModel));
    ForwardCache *c = calloc(1, sizeof(ForwardCache));
    mamba_init(m, 123);

    int T = 4;
    int ids[4] = {1, 6, 0, 1};
    int tgt[4] = {0, 1, 0, 1};

    // Analytical gradients
    float *logits = malloc(T * SSM_VOCAB * sizeof(float));
    mamba_forward(m, ids, T, logits, c);
    float *dlogits = malloc(T * SSM_VOCAB * sizeof(float));
    for (int t = 0; t < T; t++) {
        float mx = logits[t*SSM_VOCAB];
        for (int i = 1; i < SSM_VOCAB; i++)
            if (logits[t*SSM_VOCAB+i] > mx) mx = logits[t*SSM_VOCAB+i];
        float sum = 0;
        for (int i = 0; i < SSM_VOCAB; i++) sum += expf(logits[t*SSM_VOCAB+i]-mx);
        for (int i = 0; i < SSM_VOCAB; i++) {
            float p = expf(logits[t*SSM_VOCAB+i]-mx)/sum;
            dlogits[t*SSM_VOCAB+i] = (p - (i==tgt[t])) / T;
        }
    }
    mamba_zero_grad(g);
    mamba_backward(m, g, c, ids, dlogits);

    // Numerical gradient for a few params
    float eps = 1e-3f;
    float *mp = (float*)m, *gp = (float*)g;
    int n = mamba_param_count();
    int ncheck = 10, pass = 0;
    // Check specific indices: embed[0][0], head_w[0][0], D[0], A_log[0][0], etc.
    int idxs[10] = {0, 1, 64, 1000, 5000, 10000, 20000, 50000, 80000, 100000};
    for (int k = 0; k < ncheck; k++) {
        int i = idxs[k] % n;
        float orig = mp[i];
        mp[i] = orig + eps;
        float lp = loss_fn(m, ids, tgt, T, c);
        mp[i] = orig - eps;
        float lm = loss_fn(m, ids, tgt, T, c);
        mp[i] = orig;
        float num = (lp - lm) / (2*eps);
        float ana = gp[i];
        float rel = fabsf(num - ana) / (fabsf(num) + fabsf(ana) + 1e-8f);
        int ok = rel < 0.05f || (fabsf(num) < 1e-4f && fabsf(ana) < 1e-4f);
        printf("param[%d]: num=%.6f ana=%.6f rel=%.4f %s\n",
               i, num, ana, rel, ok ? "PASS" : "FAIL");
        if (ok) pass++;
    }
    printf("\n%d/%d gradient checks passed\n", pass, ncheck);
    free(logits); free(dlogits); free(m); free(g); free(c);
    return (pass == ncheck) ? 0 : 1;
}
