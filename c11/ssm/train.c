// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
// C11 Mamba training: rotation task, Adam. No Python.
//
// Usage: ./train_mamba [--steps N]
// Task: rotate-right-by-k (k=1..4), same as Python benchmark.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "../ssm/ssm.h"

#define N_BITS 8
#define K_TOK(k) (k)      // 1..4
#define SEP 5
#define VOCAB_PAD 6

static unsigned long long rng_state = 0x12345;
static int randint(int n) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (int)((rng_state >> 33) % (unsigned)n);
}

// Adam state
typedef struct {
    float *m, *v;
    int t, n;
} Adam;

static void adam_init(Adam *a, int n) {
    a->n = n; a->t = 0;
    a->m = calloc(n, sizeof(float));
    a->v = calloc(n, sizeof(float));
}

static void adam_step(Adam *a, float *p, float *g, float lr) {
    a->t++;
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float bc1 = 1 - powf(b1, (float)a->t);
    float bc2 = 1 - powf(b2, (float)a->t);
    for (int i = 0; i < a->n; i++) {
        a->m[i] = b1*a->m[i] + (1-b1)*g[i];
        a->v[i] = b2*a->v[i] + (1-b2)*g[i]*g[i];
        p[i] -= lr * (a->m[i]/bc1) / (sqrtf(a->v[i]/bc2) + eps);
    }
}

int main(int argc, char **argv) {
    int steps = 2000;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--steps") == 0 && i+1 < argc)
            steps = atoi(argv[++i]);

    MambaModel *m = calloc(1, sizeof(MambaModel));
    MambaModel *g = calloc(1, sizeof(MambaModel));
    ForwardCache *c = calloc(1, sizeof(ForwardCache));
    mamba_init(m, 0xABCD);
    printf("mamba parameters: %d\n", mamba_param_count());
    fflush(stdout);

    Adam opt;
    adam_init(&opt, mamba_param_count());

    int T = 2 + N_BITS; // [k_tok, sep, 8 bits]
    float *logits = malloc(T * SSM_VOCAB * sizeof(float));
    float *dlogits = malloc(T * SSM_VOCAB * sizeof(float));

    clock_t t0 = clock();
    for (int step = 1; step <= steps; step++) {
        // Sample: random k in 1..4, random 8 bits
        int k = 1 + randint(4);
        int bits[N_BITS];
        for (int i = 0; i < N_BITS; i++) bits[i] = randint(2);
        int ids[2+N_BITS];
        ids[0] = K_TOK(k); ids[1] = SEP;
        for (int i = 0; i < N_BITS; i++) ids[2+i] = bits[i];
        // Targets: rotate right by k (only for the 8 bit positions)
        int targets[2+N_BITS];
        targets[0] = -1; targets[1] = -1; // don't compute loss on k_tok/sep
        for (int i = 0; i < N_BITS; i++) {
            int src = (i - k + N_BITS*10) % N_BITS;
            targets[2+i] = bits[src];
        }

        // Forward
        mamba_forward(m, ids, T, logits, c);

        // Loss + dlogits (only on bit positions 2..T-1)
        float loss = 0;
        for (int t = 0; t < T; t++) {
            for (int v = 0; v < SSM_VOCAB; v++) dlogits[t*SSM_VOCAB+v] = 0;
            if (targets[t] < 0) continue;
            float mx = logits[t*SSM_VOCAB];
            for (int i = 1; i < SSM_VOCAB; i++)
                if (logits[t*SSM_VOCAB+i] > mx) mx = logits[t*SSM_VOCAB+i];
            float sum = 0;
            for (int i = 0; i < SSM_VOCAB; i++) sum += expf(logits[t*SSM_VOCAB+i]-mx);
            loss -= logits[t*SSM_VOCAB+targets[t]] - mx - logf(sum);
            for (int i = 0; i < SSM_VOCAB; i++) {
                float p = expf(logits[t*SSM_VOCAB+i]-mx)/sum;
                dlogits[t*SSM_VOCAB+i] = (p - (i==targets[t])) / N_BITS;
            }
        }
        loss /= N_BITS;

        // Backward
        mamba_zero_grad(g);
        mamba_backward(m, g, c, ids, dlogits);

        // Adam step
        adam_step(&opt, (float*)m, (float*)g, 3e-3f);

        if (step % 500 == 0) {
            printf("  step %5d  loss %.4f\n", step, loss);
            fflush(stdout);
        }
    }
    double secs = (double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("trained %d steps in %.1fs (C11, CPU)\n", steps, secs);

    // Evaluate
    printf("\nbenchmark: rotate-right-by-k accuracy (exact / bit)\n");
    for (int k = 1; k <= 4; k++) {
        int exact_ok = 0, bit_ok = 0, n = 0;
        for (int ex = 0; ex < 200; ex++) {
            int bits[N_BITS];
            for (int i = 0; i < N_BITS; i++) bits[i] = randint(2);
            int ids[2+N_BITS];
            ids[0] = K_TOK(k); ids[1] = SEP;
            for (int i = 0; i < N_BITS; i++) ids[2+i] = bits[i];
            mamba_forward(m, ids, T, logits, c);
            int exact = 1;
            for (int i = 0; i < N_BITS; i++) {
                int src = (i - k + N_BITS*10) % N_BITS;
                int pred = 0;
                float best = logits[(2+i)*SSM_VOCAB];
                for (int v = 1; v < SSM_VOCAB; v++)
                    if (logits[(2+i)*SSM_VOCAB+v] > best) {
                        best = logits[(2+i)*SSM_VOCAB+v]; pred = v;
                    }
                if (pred != bits[src]) exact = 0;
                else bit_ok++;
            }
            if (exact) exact_ok++;
            n++;
        }
        printf("k=%d  %.3f / %.3f\n", k, (float)exact_ok/n, (float)bit_ok/(n*N_BITS));
    }

    free(logits); free(dlogits); free(m); free(g); free(c);
    free(opt.m); free(opt.v);
    return 0;
}
