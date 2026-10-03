// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective

#define _POSIX_C_SOURCE 200809L
#include "sandbox.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

// Always-null-terminating copy.
static void safe_copy(char *dst, const char *src, size_t n) {
    if (n == 0) return;
    strncpy(dst, src, n - 1);
    dst[n - 1] = '\0';
}

int sandbox_init(Sandbox *sb, const char *root,
                 const char *allowed[], int n_allowed) {
    char rp[ASTRA_SANDBOX_PATH_LEN];
    if (!realpath(root, rp)) {
        // create then resolve
        mkdir(root, 0755);
        if (!realpath(root, rp)) return -1;
    }
    strncpy(sb->root, rp, sizeof(sb->root) - 1);
    sb->n_allowed = 0;
    for (int i = 0; i < n_allowed && i < ASTRA_SANDBOX_MAX_CMDS; i++) {
        strncpy(sb->allowed[sb->n_allowed++], allowed[i], 63);
    }
    return 0;
}

int sandbox_resolve(Sandbox *sb, const char *path, char *out, size_t out_n) {
    char joined[ASTRA_SANDBOX_PATH_LEN * 2];
    if (path[0] == '/') {
        // absolute: must already be inside root
        strncpy(joined, path, sizeof(joined) - 1);
    } else {
        snprintf(joined, sizeof(joined), "%s/%s", sb->root, path);
    }
    // Resolve symlinks lexically where possible; use realpath on the
    // existing prefix. Simplest correct: realpath the full path if it
    // exists, else realpath the parent.
    char rp[ASTRA_SANDBOX_PATH_LEN];
    char *tmp = malloc(strlen(joined) + 1);
    if (!tmp) return -1;
    strcpy(tmp, joined);
    // try full path first
    int ok = (realpath(joined, rp) != NULL);
    if (!ok) {
        // walk up to nearest existing ancestor
        char *slash = strrchr(tmp, '/');
        while (slash && !ok) {
            *slash = '\0';
            ok = (realpath(tmp, rp) != NULL);
            if (!ok) slash = strrchr(tmp, '/');
        }
        if (!ok) { free(tmp); return -1; }
        // append the remainder
        const char *rest = joined + strlen(tmp);
        snprintf(rp + strlen(rp), sizeof(rp) - strlen(rp), "%s", rest);
    }
    free(tmp);
    size_t rlen = strlen(sb->root);
    if (strncmp(rp, sb->root, rlen) != 0 ||
        (rp[rlen] != '\0' && rp[rlen] != '/')) {
        return -1; // escapes root
    }
    strncpy(out, rp, out_n - 1);
    out[out_n - 1] = '\0';
    return 0;
}

int sandbox_check_command(Sandbox *sb, const char *argv0) {
    const char *base = strrchr(argv0, '/');
    base = base ? base + 1 : argv0;
    for (int i = 0; i < sb->n_allowed; i++)
        if (strcmp(sb->allowed[i], base) == 0) return 0;
    return -1;
}
