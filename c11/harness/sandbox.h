// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
// Perimeter verification & sandboxing in C11.

#ifndef ASTRA_SANDBOX_H
#define ASTRA_SANDBOX_H

#include <stddef.h>

#define ASTRA_SANDBOX_MAX_CMDS 32
#define ASTRA_SANDBOX_PATH_LEN 1024

typedef struct {
    char root[ASTRA_SANDBOX_PATH_LEN]; // realpath at init
    char allowed[ASTRA_SANDBOX_MAX_CMDS][64];
    int n_allowed;
} Sandbox;

// Returns 0 on success, -1 on violation.
int sandbox_init(Sandbox *sb, const char *root,
                 const char *allowed[], int n_allowed);
// Resolve path against root; writes resolved into out. 0=ok, -1=escape.
int sandbox_resolve(Sandbox *sb, const char *path, char *out, size_t out_n);
// 0 if argv[0] basename is allowlisted, -1 otherwise.
int sandbox_check_command(Sandbox *sb, const char *argv0);

#endif
