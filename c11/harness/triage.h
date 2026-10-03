// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
// Proactive disambiguation & autonomous triage in C11.

#ifndef ASTRA_TRIAGE_H
#define ASTRA_TRIAGE_H

typedef enum { CONSEQUENCE_LOW = 0, CONSEQUENCE_HIGH = 1 } Consequence;

typedef struct {
    char verb[32];
    char target[256];
    int reversible; // 0 or 1
    // params as key=value pairs for secret detection
    char params[8][128];
    int n_params;
} Action;

Consequence classify_action(const Action *a);

#endif
