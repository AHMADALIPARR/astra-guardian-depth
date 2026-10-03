// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
// C11 harness tests. Build: cc -std=c11 -O2 -o test_harness test_harness.c
//   ../harness/context.c ../harness/triage.c ../harness/sandbox.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../harness/context.h"
#include "../harness/triage.h"
#include "../harness/sandbox.h"

static int failures = 0;
#define CHECK(name, cond) do { \
    printf("%s %s\n", (cond) ? "PASS" : "FAIL", name); \
    if (!(cond)) failures++; \
} while (0)

int main(void) {
    // --- context --- (heap-allocated: the index is too large for stack)
    CodexHarness *h = calloc(1, sizeof(CodexHarness));
    codex_harness_init(h);
    codex_harness_ingest(h, "a", "ERROR db trace: conn.py:88 TimeoutError POOL_SIZE=32 exhausted");
    codex_harness_ingest(h, "b", "INFO api heartbeat ok");
    rolling_notes_env(&h->notes, "POOL_SIZE", "32");
    rolling_notes_developer(&h->notes, "db pool exhausted at deploy");

    char ids[5][64];
    float scores[5];
    int n = vector_index_search(&h->index, "what is POOL_SIZE", 1, ids, scores);
    CHECK("retrieval finds exact doc", n == 1 && strcmp(ids[0], "a") == 0);
    const char *txt = vector_index_get(&h->index, "a");
    CHECK("verbatim text preserved", txt && strstr(txt, "POOL_SIZE=32"));
    CHECK("env recall exact",
          rolling_notes_recall_env(&h->notes, "POOL_SIZE") &&
          strcmp(rolling_notes_recall_env(&h->notes, "POOL_SIZE"), "32") == 0);
    CHECK("unknown env returns NULL (no guessing)",
          rolling_notes_recall_env(&h->notes, "MISSING") == NULL);

    // --- triage ---
    Action a1 = {.verb = "rename", .target = "/tmp/x.tmp", .reversible = 1, .n_params = 0};
    Action a2 = {.verb = "delete", .target = "/data/prod", .reversible = 0, .n_params = 0};
    Action a3 = {.verb = "read", .target = "~/.ssh/id_rsa", .reversible = 1, .n_params = 0};
    CHECK("low-consequence -> LOW", classify_action(&a1) == CONSEQUENCE_LOW);
    CHECK("irreversible -> HIGH", classify_action(&a2) == CONSEQUENCE_HIGH);
    CHECK("ssh key -> HIGH", classify_action(&a3) == CONSEQUENCE_HIGH);

    // --- sandbox ---
    Sandbox sb;
    const char *allowed[] = {"echo"};
    CHECK("sandbox init", sandbox_init(&sb, "/tmp/astra_c11_test", allowed, 1) == 0);
    char out[1024];
    CHECK("contained path resolves", sandbox_resolve(&sb, "work/out.txt", out, sizeof(out)) == 0);
    CHECK("traversal blocked", sandbox_resolve(&sb, "../../evil", out, sizeof(out)) != 0);
    CHECK("absolute outside blocked",
          sandbox_resolve(&sb, "/etc/hostname", out, sizeof(out)) != 0);
    CHECK("allowlisted cmd ok", sandbox_check_command(&sb, "echo") == 0);
    CHECK("blocked cmd rejected", sandbox_check_command(&sb, "rm") != 0);

    printf("\n%d failures\n", failures);
    free(h);
    return failures ? 1 : 0;
}
