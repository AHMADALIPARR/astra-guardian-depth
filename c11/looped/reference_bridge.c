// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Test-only bridge to the untouched committed forward pass, compiled in its
// own translation unit so the incompatible cache definitions never mix.
#define looped_init reference_init
#define looped_param_count reference_count
#define looped_zero_grad reference_zero
#define looped_forward reference_forward
#define looped_backward reference_backward
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
/* Original forward lacks T validation; this test only calls positive T. */
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include "../looped/looped.c"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

int reference_evaluate(const void *weights, size_t size, const int *ids,
                         int T, float *logits) {
    if (size != sizeof(LoopedModel) || T < 1 || T > LT_MAX_T) return -1;
    LoopedModel *m = malloc(sizeof *m);
    LoopedCache *c = calloc(1, sizeof *c);
    if (!m || !c) { free(m); free(c); return -1; }
    memcpy(m, weights, sizeof *m);
    reference_forward(m, ids, T, logits, c);
    free(c); free(m);
    return 0;
}
