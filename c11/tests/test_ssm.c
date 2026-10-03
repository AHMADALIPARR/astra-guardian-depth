// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
// C11 SSM forward test.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../ssm/ssm.h"

int main(void) {
    MambaModel *m = calloc(1, sizeof(MambaModel));
    mamba_init(m, 42);
    printf("params: %d\n", mamba_param_count());

    int ids[10] = {1, 6, 0, 1, 0, 1, 0, 1, 0, 1};
    float *logits = malloc(10 * SSM_VOCAB * sizeof(float));
    mamba_forward(m, ids, 10, logits, NULL);

    int finite = 1;
    for (int i = 0; i < 10 * SSM_VOCAB; i++)
        if (!isfinite(logits[i])) finite = 0;
    printf("forward finite: %s\n", finite ? "PASS" : "FAIL");
    printf("logits[0]:");
    for (int i = 0; i < SSM_VOCAB; i++) printf(" %.3f", logits[i]);
    printf("\n");

    free(logits); free(m);
    return finite ? 0 : 1;
}
