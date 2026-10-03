// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
#include "looped.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

int reference_evaluate(const void *weights, size_t size, const int *ids,
                         int T, float *logits);
int main(void) {
    LoopedModel *m = malloc(sizeof *m);
    LoopedCache *c = malloc(sizeof *c);
    if (!m || !c) { free(m); free(c); return 1; }
    float actual[10*LT_VOCAB], expected[10*LT_VOCAB];
    int ids[10] = {1, 6, 0, 1, 3, 2, 1, 0, 4, 0};
    double largest = 0;
    int lengths[3] = {1, 5, 10};
    for (int seed = 1; seed <= 3; ++seed) {
        looped_init(m, (unsigned long long)seed);
        for (int n = 0; n < 3; ++n) {
            int T = lengths[n];
            if (looped_forward(m, ids, T, actual, c) != LT_OK ||
                reference_evaluate(m, sizeof *m, ids, T, expected)) { free(m); free(c); return 1; }
            for (int i = 0; i < T*LT_VOCAB; ++i) {
                double error = fabs((double)actual[i] - expected[i]);
                largest = fmax(largest, error);
                if (!isfinite(error) || error > 2e-5) {
                    fprintf(stderr, "reference mismatch seed=%d T=%d index=%d error=%g\n", seed, T, i, error);
                    free(m); free(c); return 1;
                }
            }
        }
    }
    printf("PASS committed-forward parity: 9 cases, maximum logit difference %.3g\n", largest);
    free(c); free(m); return 0;
}
