// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective

#include "triage.h"
#include <ctype.h>
#include <string.h>

static void lower_copy(const char *src, char *dst, size_t n) {
    size_t i;
    for (i = 0; i + 1 < n && src[i]; i++) dst[i] = (char)tolower((unsigned char)src[i]);
    dst[i] = '\0';
}

static int contains(const char *hay, const char *needle) {
    return strstr(hay, needle) != NULL;
}

static const char *HIGH_VERBS[] = {
    "delete", "remove", "drop", "destroy", "overwrite", "truncate",
    "deploy", "migrate", "chmod", "chown", "format", "shutdown",
    "reboot", "publish", "release", NULL
};

static const char *HIGH_HINTS[] = {
    "prod", "production", "/etc/", "/root", ".ssh", "secret", "token",
    "password", "credential", "database", "migration", NULL
};

static const char *LOW_HINTS[] = { "tmp", "temp", "scratch", "cache", ".bak", NULL };

static int in_list(const char *word, const char *const *list) {
    for (int i = 0; list[i]; i++)
        if (strcmp(word, list[i]) == 0) return 1;
    return 0;
}

static int any_hint(const char *target, const char *const *hints) {
    for (int i = 0; hints[i]; i++)
        if (contains(target, hints[i])) return 1;
    return 0;
}

Consequence classify_action(const Action *a) {
    char verb[32], target[256];
    lower_copy(a->verb, verb, sizeof(verb));
    lower_copy(a->target, target, sizeof(target));

    if (!a->reversible) return CONSEQUENCE_HIGH;
    if (in_list(verb, HIGH_VERBS) && !any_hint(target, LOW_HINTS))
        return CONSEQUENCE_HIGH;
    if (any_hint(target, HIGH_HINTS)) return CONSEQUENCE_HIGH;
    for (int i = 0; i < a->n_params; i++) {
        char p[128];
        lower_copy(a->params[i], p, sizeof(p));
        if (contains(p, "secret") || contains(p, "token") || contains(p, "password"))
            return CONSEQUENCE_HIGH;
    }
    return CONSEQUENCE_LOW;
}
