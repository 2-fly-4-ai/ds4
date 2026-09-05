// Timing-only barrier comparison: identical prompt AND continuation token IDs.
// Does not assert that the historically racy implementation has correct logits.
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
static void ck(int ok, const char *msg) { if (!ok) { fprintf(stderr, "%s\n", msg); exit(2); } }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
int main(int argc, char **argv) {
    ck(argc == 5, "usage: model token-source prefix label");
    int prefix = atoi(argv[3]), warm = 64, count = 256, ctx = prefix + warm + count + 1;
    ck(prefix > 0 && prefix <= 32768, "prefix range");
    FILE *f = fopen(argv[2], "rb"); ck(f != NULL, "open corpus");
    fseek(f, 0, SEEK_END); long size = ftell(f); rewind(f);
    ck(size > 0, "corpus size"); char *text = malloc(size + 1); ck(text != NULL, "alloc");
    ck(fread(text, 1, size, f) == (size_t)size, "read"); fclose(f); text[size] = 0;
    ds4_engine_options opt = {.model_path=argv[1], .backend=DS4_BACKEND_METAL,
        .context_size=ctx, .warm_weights=true, .power_percent=100};
    ds4_engine *e = NULL; ck(ds4_engine_open(&e, &opt) == 0, "engine");
    ds4_tokens all = {0}; ds4_tokenize_text(e, text, &all); free(text);
    ck(all.len >= ctx, "not enough corpus tokens");
    uint64_t hash = 1469598103934665603ULL;
    for (int i=0; i<prefix+warm+count; i++) { hash ^= (uint32_t)all.v[i]; hash *= 1099511628211ULL; }
    ds4_tokens prompt = {.v=all.v, .len=prefix, .cap=prefix};
    ds4_session *s = NULL; ck(ds4_session_create(&s, e, ctx) == 0, "session");
    char err[512] = {0}; double t = now();
    ck(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0, err);
    double pp = now() - t;
    for (int i=0; i<warm; i++) ck(ds4_session_eval(s, all.v[prefix+i], err, sizeof(err)) == 0, err);
    t = now();
    for (int i=0; i<count; i++) ck(ds4_session_eval(s, all.v[prefix+warm+i], err, sizeof(err)) == 0, err);
    double seconds = now() - t;
    printf("label=%s prefix=%d warm=%d tokens=%d seconds=%.9f tps=%.6f prefill_seconds=%.9f input_hash=%016llx\n",
        argv[4], prefix, warm, count, seconds, count/seconds, pp, (unsigned long long)hash);
    ds4_session_free(s); ds4_tokens_free(&all); ds4_engine_close(e); return 0;
}
