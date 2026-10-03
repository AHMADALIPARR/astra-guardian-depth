// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
#include "looped.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
static LoopedModel *m, *g, *saved;
static LoopedCache *c;
static int ids[LT_MAX_T], checks;
static float logits[LT_MAX_T * LT_VOCAB], dy[LT_MAX_T * LT_VOCAB];
static double max_error;
static float get(const LoopedModel *p, size_t off) {
    float v; memcpy(&v, (const unsigned char *)p + off, sizeof v); return v;
}
static void put(LoopedModel *p, size_t off, float v) {
    memcpy((unsigned char *)p + off, &v, sizeof v);
}
static double objective(int T, int rounds) {
    CHECK(looped_forward_r(m, ids, T, rounds, logits, c) == LT_OK);
    double result = 0;
    for (int t = 0; t < T; ++t)
        for (int v = 0; v < LT_VOCAB; ++v) result += c->logits[t][v] * dy[t*LT_VOCAB+v];
    return result;
}
static void analytic(int T, int rounds) {
    (void)objective(T, rounds);
    looped_zero_grad(g);
    CHECK(looped_backward(m, g, c, ids, dy) == LT_OK);
}
static void check_one(size_t offset, int T, int rounds) {
    float orig = get(m, offset), plus = orig + 0.0001f, minus = orig - 0.0001f;
    put(m, offset, plus); double lp = objective(T, rounds);
    put(m, offset, minus); double lm = objective(T, rounds);
    put(m, offset, orig);
    double numerical = (lp - lm) / ((double)plus - minus), actual = get(g, offset);
    double error = fabs(numerical - actual);
    double tolerance = 2e-8 + 5e-4 * fmax(fabs(numerical), fabs(actual));
    if (!isfinite(actual) || error > tolerance) {
        fprintf(stderr, "FAIL gradient byte=%zu T=%d r=%d numeric=%.12g actual=%.12g error=%g tolerance=%g\n",
                offset, T, rounds, numerical, actual, error, tolerance);
        exit(1);
    }
    max_error = fmax(max_error, error);
    ++checks;
}
static void group(const char *name, size_t start, size_t bytes, int T, int rounds) {
    size_t selected = start;
    double largest = -1;
    for (size_t i = start; i < start + bytes; i += sizeof(float)) {
        double value = fabs(get(g, i));
        if (value > largest) { largest = value; selected = i; }
    }
    CHECK(largest > 1e-8);
    check_one(selected, T, rounds);
    printf("PASS %-15s nonzero gradient %.6g\n", name, largest);
}
#define GROUP(F) group(#F, offsetof(LoopedModel, F), sizeof m->F, 5, LT_R_LOOP)
#define BG(F) group(#F, base + offsetof(TransformerBlock, F), sizeof m->shared.F, 5, LT_R_LOOP)
static void all_groups(void) {
    analytic(5, LT_R_LOOP);
    GROUP(embed); GROUP(pos_emb); GROUP(inject_w); GROUP(final_norm_w); GROUP(head_w);
    for (int b = 0; b < 2; ++b) {
        size_t base = b ? offsetof(LoopedModel, shared) : offsetof(LoopedModel, prelude);
        BG(out_w); BG(out_b); BG(attn_norm_w); BG(gate_w); BG(up_w); BG(down_w); BG(mlp_norm_w);
        size_t off = base + offsetof(TransformerBlock, qkv_w);
        size_t bytes = LT_D_MODEL * LT_D_MODEL * sizeof(float);
        group("query weights", off, bytes, 5, LT_R_LOOP);
        group("key weights", off + bytes, bytes, 5, LT_R_LOOP);
        group("value weights", off + 2*bytes, bytes, 5, LT_R_LOOP);
        off = base + offsetof(TransformerBlock, qkv_b);
        bytes = LT_D_MODEL * sizeof(float);
        group("query bias", off, bytes, 5, LT_R_LOOP);
        group("value bias", off + 2*bytes, bytes, 5, LT_R_LOOP);
        /* Adding a constant key bias shifts every row equally: zero derivative. */
        for (int j = 0; j < LT_D_MODEL; ++j) {
            CHECK(fabs(get(g, off + bytes + j*sizeof(float))) < 2e-7);
            check_one(off + bytes + j*sizeof(float), 5, LT_R_LOOP);
        }
    }
}
static void attention_and_causality(void) {
    CHECK(looped_forward(m, ids, 5, logits, c) == LT_OK);
    for (int r = 0; r <= LT_R_LOOP; ++r)
        for (int h = 0; h < LT_N_HEADS; ++h)
            for (int t = 0; t < 5; ++t) {
                double sum = 0;
                for (int s = 0; s < 5; ++s) {
                    double p = c->block[r].probability[h][t][s];
                    CHECK(isfinite(p) && p >= 0);
                    if (s > t) CHECK(p == 0);
                    sum += p;
                }
                CHECK(fabs(sum - 1) < 1e-12);
            }
    float prefix[4 * LT_VOCAB]; memcpy(prefix, logits, sizeof prefix);
    int old = ids[4]; ids[4] = (old + 1) % LT_VOCAB;
    CHECK(looped_forward(m, ids, 5, logits, c) == LT_OK);
    CHECK(memcmp(prefix, logits, sizeof prefix) == 0);
    ids[4] = old;
    /* A loss on an early token cannot produce gradients in future positions. */
    float upstream[5 * LT_VOCAB] = {0}; upstream[LT_VOCAB + 1] = 1;
    CHECK(looped_forward(m, ids, 5, logits, c) == LT_OK);
    looped_zero_grad(g);
    CHECK(looped_backward(m, g, c, ids, upstream) == LT_OK);
    for (int t = 2; t < LT_MAX_T; ++t)
        for (int j = 0; j < LT_D_MODEL; ++j) CHECK(g->pos_emb[t][j] == 0);
    puts("PASS causal probabilities, prefix invariance, and zero future gradients");
}
static void accumulation_and_validation(void) {
    analytic(5, LT_R_LOOP); memcpy(saved, g, sizeof *g);
    CHECK(looped_backward(m, g, c, ids, dy) == LT_OK);
    for (size_t i = 0; i < sizeof *g; i += sizeof(float))
        CHECK(fabs(get(g, i) - 2*get(saved, i)) < 2e-5);
    CHECK(looped_backward(m, m, c, ids, dy) == LT_INVALID);
    CHECK(looped_backward(saved, g, c, ids, dy) == LT_INVALID);
    int old = ids[0]; ids[0] = (old + 1) % LT_VOCAB;
    CHECK(looped_backward(m, g, c, ids, dy) == LT_INVALID); ids[0] = old;
    float old_dy = dy[0]; dy[0] = NAN;
    CHECK(looped_backward(m, g, c, ids, dy) == LT_NONFINITE); dy[0] = old_dy;
    CHECK(looped_forward(m, ids, 0, logits, c) == LT_INVALID);
    CHECK(!c->valid && looped_backward(m, g, c, ids, dy) == LT_INVALID);
    CHECK(looped_forward(m, ids, LT_MAX_T + 1, logits, c) == LT_INVALID);
    CHECK(looped_forward_r(m, ids, 5, 0, logits, c) == LT_INVALID);
    CHECK(looped_forward_r(m, ids, 5, LT_R_LOOP+1, logits, c) == LT_INVALID);
    ids[0] = LT_VOCAB;
    CHECK(looped_forward(m, ids, 5, logits, c) == LT_INVALID); ids[0] = old;
    CHECK(looped_forward(m, ids, 1, logits, NULL) == LT_OK);
    CHECK(looped_forward(m, ids, LT_MAX_T, logits, c) == LT_OK);
    looped_zero_grad(g);
    CHECK(looped_backward(m, g, c, ids, dy) == LT_OK);
    puts("PASS accumulation, ownership, invalid inputs, inference, maximum length");
}
static void loss_and_learning(void) {
    int targets[5] = {-1, -1, 1, 0, 1};
    double first, last;
    CHECK(looped_forward(m, ids, 5, logits, c) == LT_OK);
    CHECK(looped_cross_entropy(c, targets, &first, dy) == LT_OK);
    for (int t = 0; t < 5; ++t) {
        double sum = 0;
        for (int v = 0; v < LT_VOCAB; ++v) sum += dy[t*LT_VOCAB+v];
        CHECK(fabs(sum) < 1e-7);
        int v = t % LT_VOCAB;
        double orig = c->logits[t][v], plus, minus;
        float expected = dy[t*LT_VOCAB+v], scratch[5*LT_VOCAB];
        c->logits[t][v] = orig + 1e-5;
        CHECK(looped_cross_entropy(c, targets, &plus, scratch) == LT_OK);
        c->logits[t][v] = orig - 1e-5;
        CHECK(looped_cross_entropy(c, targets, &minus, scratch) == LT_OK);
        c->logits[t][v] = orig;
        CHECK(fabs((plus-minus)/2e-5 - expected) < 1e-7);
    }
    for (int step = 0; step < 70; ++step) {
        CHECK(looped_forward(m, ids, 5, logits, c) == LT_OK);
        CHECK(looped_cross_entropy(c, targets, &last, dy) == LT_OK);
        looped_zero_grad(g);
        CHECK(looped_backward(m, g, c, ids, dy) == LT_OK);
        for (size_t i = 0; i < sizeof *m; i += sizeof(float))
            put(m, i, get(m, i) - 0.005f * get(g, i));
    }
    CHECK(looped_forward(m, ids, 5, logits, c) == LT_OK);
    CHECK(looped_cross_entropy(c, targets, &last, dy) == LT_OK);
    CHECK(last < 0.7 * first);
    printf("PASS SGD loss %.6f -> %.6f\n", first, last);
    for (int t = 0; t < 5; ++t) targets[t] = -1;
    CHECK(looped_cross_entropy(c, targets, &last, dy) == LT_OK && last == 0);
    for (int i = 0; i < 5*LT_VOCAB; ++i) CHECK(dy[i] == 0);
}

int main(void) {
    m = calloc(1, sizeof *m); g = calloc(1, sizeof *g);
    saved = calloc(1, sizeof *saved); c = calloc(1, sizeof *c);
    CHECK(m && g && saved && c);
    CHECK(sizeof *m == (size_t)looped_param_count() * sizeof(float));
    looped_init(m, 7123);
    for (int t = 0; t < LT_MAX_T; ++t) {
        ids[t] = (t*t + 1) % LT_VOCAB;
        for (int v = 0; v < LT_VOCAB; ++v) dy[t*LT_VOCAB+v] = (float)(sin(t*2.7 + v*1.3) / 5);
    }
    all_groups();
    if (looped_param_count() < 2000) {
        int configs[3][2] = {{5, 1}, {5, LT_R_LOOP}, {1, LT_R_LOOP}};
        for (int k = 0; k < 3; ++k) {
            analytic(configs[k][0], configs[k][1]);
            for (size_t i = 0; i < sizeof *m; i += sizeof(float))
                check_one(i, configs[k][0], configs[k][1]);
        }
        /* Zero projection still needs a nonzero dW_inject and the h[0] path. */
        memset(m->inject_w, 0, sizeof m->inject_w);
        analytic(5, LT_R_LOOP);
        GROUP(inject_w);
        GROUP(embed);
        looped_init(m, 7123);
    }
    attention_and_causality();
    accumulation_and_validation();
    loss_and_learning();
    printf("PASS %d finite-difference checks, %d parameters, maximum error %.3g\n",
           checks, looped_param_count(), max_error);
    free(c); free(saved); free(g); free(m);
    return 0;
}
