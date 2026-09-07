/* Model-free grow/reuse oracle adapted from antirez/ds4 c0a6119.
 * Include the core so a tiny synthetic CPU cache exercises the real serializer. */
#include "../ds4.c"
#include <assert.h>

static void test_snapshot_bytes(void) {
    const ds4_shape saved_shape = g_ds4_shape;
    const uint32_t saved_ratio = g_ds4_compress_ratios[0];
    g_ds4_shape = DS4_SHAPE_FLASH;
    g_ds4_shape.n_layer = 1;
    g_ds4_shape.n_head_dim = 4;
    g_ds4_shape.n_vocab = 8;
    g_ds4_compress_ratios[0] = 0;
    ds4_engine e = {.backend = DS4_BACKEND_CPU};
    ds4_session *s = calloc(1, sizeof(*s));
    assert(s);
    s->engine = &e;
    s->ctx_size = 8;
    s->prefill_cap = 1;
    s->checkpoint_valid = true;
    s->logits = calloc(DS4_N_VOCAB, sizeof(float));
    assert(s->logits);
    kv_cache_init(&s->cpu_cache, 8, 0);
    for (int i = 0; i < 3; i++) ds4_tokens_push(&s->checkpoint, i);
    /* Every byte of the final float is nonzero, including the last byte. */
    const uint32_t bits = UINT32_C(0x3f9e0652);
    for (int i = 0; i < 3 * 4; i++)
        memcpy(s->cpu_cache.layer[0].raw_kv + i, &bits, sizeof(bits));
    ds4_session_snapshot snapshot = {0};
    const int lengths[] = {1, 3, 2, 3};
    for (size_t i = 0; i < sizeof(lengths) / sizeof(*lengths); i++) {
        s->checkpoint.len = lengths[i];
        s->cpu_cache.layer[0].n_raw = lengths[i];
        char err[192] = {0};
        FILE *fp = tmpfile();
        assert(fp);
        assert(ds4_session_save_payload(s, fp, err, sizeof(err)) == 0);
        const long bytes = ftell(fp);
        assert(bytes > 0);
        assert(ds4_session_save_snapshot(s, &snapshot, err, sizeof(err)) == 0);
        assert(snapshot.len == (uint64_t)bytes);
        rewind(fp);
        for (long j = 0; j < bytes; j++) {
            const int expected = fgetc(fp);
            if (expected != snapshot.ptr[j])
                fprintf(stderr, "snapshot byte %ld/%ld: expected=%d actual=%d\n",
                        j, bytes, expected, snapshot.ptr[j]);
            assert(expected == snapshot.ptr[j]);
        }
        assert(fgetc(fp) == EOF);
        fclose(fp);
    }
    ds4_session_snapshot_free(&snapshot);
    ds4_session_free(s);
    g_ds4_shape = saved_shape;
    g_ds4_compress_ratios[0] = saved_ratio;
}

int main(void) { test_snapshot_bytes(); puts("snapshot bytes: PASS"); return 0; }
