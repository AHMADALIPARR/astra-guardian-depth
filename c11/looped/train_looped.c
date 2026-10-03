// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Balanced-k gradient accumulation: ./train-looped [steps [batch [seed]]]
#include "looped.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long random_word(unsigned long long *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return *state >> 32;
}
static void example(int bits, int k, int *ids, int *targets) {
    ids[0] = k+2; ids[1] = 2;
    targets[0] = targets[1] = -1;
    for (int i = 0; i < 8; ++i) {
        ids[i+2] = (bits >> i) & 1;
        targets[i+2] = (bits >> ((i-k+8)%8)) & 1;
    }
}
static int evaluate(const LoopedModel *m, LoopedCache *c) {
    int ids[10], targets[10];
    float logits[10*LT_VOCAB];
    /* Exhaust all 256 bit strings for each k; evaluation consumes no train RNG. */
    for (int k = 1; k <= 4; ++k) {
        int correct[8] = {0}, exact = 0;
        for (int bits = 0; bits < 256; ++bits) {
            example(bits, k, ids, targets);
            if (looped_forward(m, ids, 10, logits, c) != LT_OK) return 1;
            int all = 1;
            for (int i = 0; i < 8; ++i) {
                int pred = 0;
                for (int v = 1; v < LT_VOCAB; ++v)
                    if (logits[(i+2)*LT_VOCAB+v] > logits[(i+2)*LT_VOCAB+pred]) pred = v;
                int ok = pred == targets[i+2];
                correct[i] += ok; all &= ok;
            }
            exact += all;
        }
        int hidden = 0, visible = 0;
        for (int i = 0; i < 8; ++i) { if (i < k) hidden += correct[i]; else visible += correct[i]; }
        printf("k=%d exact=%.4f bit=%.4f visible=%.4f hidden=%.4f positions=",
               k, exact/256.0, (hidden+visible)/2048.0,
               visible/(256.0*(8-k)), hidden/(256.0*k));
        for (int i = 0; i < 8; ++i) printf(" %.4f", correct[i]/256.0);
        putchar('\n');
    }
    return 0;
}
static int positive(const char *text, long *out) {
    char *end;
    errno = 0;
    *out = strtol(text, &end, 10);
    return !errno && end != text && !*end && *out > 0 && *out <= 1000000;
}
static int positive_lr(const char *text, double *out) {
    char *end;
    errno = 0;
    *out = strtod(text, &end);
    return !errno && end != text && !*end &&
           isfinite(*out) && *out > 0;
}
int main(int argc, char **argv) {
    long steps = 500, batch = 32, seed = 100;
    double lr = 0.003;
    if (argc > 5 || (argc > 1 && !positive(argv[1], &steps)) ||
        (argc > 2 && !positive(argv[2], &batch)) ||
        (argc > 3 && !positive(argv[3], &seed)) ||
        (argc > 4 && !positive_lr(argv[4], &lr)) || batch % 4 || batch > 4096 ||
        LT_MAX_T < 10 || LT_VOCAB < 7) {
        fprintf(stderr, "usage: %s [steps [batch-multiple-of-4 [seed [lr]]]]\n", argv[0]); return 1;
    }
    LoopedModel *m = calloc(1, sizeof *m);
    LoopedGrad *g = calloc(1, sizeof *g);
    LoopedCache *c = calloc(1, sizeof *c);
    size_t n = (size_t)looped_param_count();
    double *first = calloc(n, sizeof *first), *second = calloc(n, sizeof *second);
    if (!m || !g || !c || !first || !second || sizeof *m != n*sizeof(float)) {
        free(m); free(g); free(c); free(first); free(second); return 1;
    }
    unsigned long long rng = (unsigned long long)seed;
    looped_init(m, rng);
    int status = 0, ids[10], targets[10];
    float logits[10*LT_VOCAB], dy[10*LT_VOCAB];
    double beta1 = 1, beta2 = 1;
    printf("parameters=%zu batch=%ld seed=%ld\n", n, batch, seed);
    for (long step = 1; step <= steps; ++step) {
        double loss_sum = 0;
        looped_zero_grad(g);
        for (long e = 0; e < batch; ++e) {
            double loss;
            example((int)(random_word(&rng) % 256), 1+(int)(e%4), ids, targets);
            if (looped_forward(m, ids, 10, logits, c) != LT_OK ||
                looped_cross_entropy(c, targets, &loss, dy) != LT_OK ||
                looped_backward(m, g, c, ids, dy) != LT_OK) { status = 1; goto done; }
            loss_sum += loss;
        }
        beta1 *= 0.9; beta2 *= 0.999;
        double norm2 = 0;
        for (size_t i = 0; i < n; ++i) {
            float gradient;
            memcpy(&gradient, (unsigned char *)g + i*sizeof(float), sizeof gradient);
            if (!isfinite(gradient)) { status = 1; goto done; }
            double d = gradient / (double)batch;
            norm2 += d*d;
        }
        /* No implicit clipping: report the gradient norm for comparison. */
        for (size_t i = 0; i < n; ++i) {
            float parameter, gradient;
            memcpy(&parameter, (unsigned char *)m + i*sizeof(float), sizeof parameter);
            memcpy(&gradient, (unsigned char *)g + i*sizeof(float), sizeof gradient);
            double d = gradient / (double)batch;
            first[i] = 0.9*first[i] + 0.1*d;
            second[i] = 0.999*second[i] + 0.001*d*d;
            parameter -= (float)(lr*(first[i]/(1-beta1))/(sqrt(second[i]/(1-beta2))+1e-8));
            memcpy((unsigned char *)m + i*sizeof(float), &parameter, sizeof parameter);
        }
        if (step == 1 || step % 100 == 0 || step == steps) {
            printf("step=%ld examples=%ld loss=%.6f grad_norm=%.6f\n",
                   step, step*batch, loss_sum/batch, sqrt(norm2));
            fflush(stdout);
        }
    }
    status = evaluate(m, c);
done:
    free(second); free(first); free(c); free(g); free(m);
    return status;
}
