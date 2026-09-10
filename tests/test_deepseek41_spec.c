#include "ds4.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

static void test_profile(void) {
    uint32_t p[16] = {0};
    ds4_test_deepseek41_profile(p);
    assert(p[0] == 43);
    assert(p[1] == 3);
    assert(p[2] == 5120);
    assert(p[3] == 129280);
    assert(p[4] == 64);
    assert(p[5] == 512);
    assert(p[6] == 1280);
    assert(p[7] == 1024);
    assert(p[8] == 384);
    assert(p[9] == 6);
    assert(p[10] == 2304);
    assert(p[11] == 32);
    assert(p[12] == 512);
    assert(p[13] == 2048);
    assert(p[14] == 8);
    assert(p[15] == 99092);
}

static void test_compress_ratios(void) {
    for (uint32_t il = 0; il < 43; il++) {
        const uint32_t want = il < 2 || il >= 40 ? 0 : (il < 20 ? 2 : 1);
        assert(ds4_test_deepseek41_compress_ratio(il) == want);
    }
}

static void test_shared_sources(void) {
    static const int32_t kv[40] = {
        -1, -1,
         2,  2,  2,  2,  2,  2,
         8,  8,  8,  8,  8,  8,
        14, 14, 14, 14, 14, 14,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    };
    static const int32_t index[40] = {
        -1, -1,
         2,  2,  2,  2,  2,  2,
         8,  8,  8,  8,  8,  8,
        14, 14, 14, 14, 14, 14,
        20, 20, 20, 20,
        24, 24, 24, 24,
        28, 28, 28, 28,
        32, 32, 32, 32,
        36, 36, 36, 36,
    };
    for (uint32_t il = 0; il < 40; il++) {
        assert(ds4_test_deepseek41_kv_source(il) == kv[il]);
        assert(ds4_test_deepseek41_index_source(il) == index[il]);
        assert(ds4_test_deepseek41_owns_kv(il) == (kv[il] == (int32_t)il));
        assert(ds4_test_deepseek41_owns_index(il) == (index[il] == (int32_t)il));
    }
    for (uint32_t il = 40; il < 43; il++) {
        assert(ds4_test_deepseek41_kv_source(il) == -1);
        assert(ds4_test_deepseek41_index_source(il) == -1);
        assert(!ds4_test_deepseek41_owns_kv(il));
        assert(!ds4_test_deepseek41_owns_index(il));
    }
}

static void test_reasoning_effort(void) {
    assert(ds4_deepseek41_reasoning_effort_text(DS4_THINK_NONE) == NULL);
    assert(strstr(ds4_deepseek41_reasoning_effort_text(DS4_THINK_LOW),
                  "Reasoning Effort: 50 ") != NULL);
    assert(strstr(ds4_deepseek41_reasoning_effort_text(DS4_THINK_HIGH),
                  "Reasoning Effort: 75 ") != NULL);
    assert(strstr(ds4_deepseek41_reasoning_effort_text(DS4_THINK_MAX),
                  "Reasoning Effort: 100 ") != NULL);
}

static void test_previous_pre_mix_hc_transition(void) {
    const float residual[6] = {1, 2, 3, 10, 20, 30};
    const float incoming_pre[2] = {0.25f, 0.75f};
    const float post[2] = {0.5f, 2.0f};
    const float comb[4] = {1, 0, 0, 1};
    const float sublayer[3] = {4, 5, 6};
    const float want_collapsed[3] = {7.75f, 15.5f, 23.25f};
    const float want_next[6] = {3, 4.5f, 6, 18, 30, 42};
    float collapsed[3] = {0};
    float next[6] = {0};

    ds4_test_deepseek41_hc_transition(collapsed, next, residual,
                                      incoming_pre, post, comb, sublayer,
                                      3, 2);
    for (int i = 0; i < 3; i++) {
        assert(fabsf(collapsed[i] - want_collapsed[i]) < 1e-6f);
    }
    for (int i = 0; i < 6; i++) {
        assert(fabsf(next[i] - want_next[i]) < 1e-6f);
    }
}

static void test_ratio2_per_dimension_softmax_pool(void) {
    const float values[6] = {2, 10, -4, 8, 20, 6};
    const float scores[6] = {0, 2, -1, 0, 0, 1};
    float out[3] = {0};
    float want[3];
    for (int d = 0; d < 3; d++) {
        const float m = scores[d] > scores[3 + d] ? scores[d] : scores[3 + d];
        const float a = expf(scores[d] - m);
        const float b = expf(scores[3 + d] - m);
        want[d] = (a * values[d] + b * values[3 + d]) / (a + b);
    }
    ds4_test_deepseek41_compressor_pool(out, values, scores, 3, 2);
    for (int i = 0; i < 3; i++) assert(fabsf(out[i] - want[i]) < 1e-6f);
}

static void test_candidate_block_selection(void) {
    enum { N = 2049 * 8 + 3 };
    float scores[N];
    bool mask[N];
    for (uint32_t i = 0; i < N; i++) scores[i] = -(float)i;

    /* There are 2,050 blocks, so exactly two lose.  Block 2049 is the newest
     * partial block and must remain pinned even though its values rank last. */
    ds4_test_deepseek41_candidate_mask(mask, scores, N, N);
    uint32_t kept_blocks = 0;
    for (uint32_t b = 0; b < 2050; b++) {
        const bool kept = mask[b * 8];
        if (kept) kept_blocks++;
        for (uint32_t c = b * 8; c < N && c < (b + 1u) * 8u; c++) {
            assert(mask[c] == kept);
        }
    }
    assert(kept_blocks == 2048);
    assert(mask[2049 * 8]);
    assert(!mask[2047 * 8]);
    assert(!mask[2048 * 8]);

    /* Unreachable tail blocks are never kept, even if their stored values are
     * large; the newest reachable partial block is the one that gets pinned. */
    const uint32_t visible = 17;
    for (uint32_t i = visible; i < N; i++) scores[i] = 1000000.0f;
    ds4_test_deepseek41_candidate_mask(mask, scores, N, visible);
    for (uint32_t i = 24; i < N; i++) assert(!mask[i]);
    assert(mask[16]);
}

int main(void) {
    test_profile();
    test_compress_ratios();
    test_shared_sources();
    test_reasoning_effort();
    test_previous_pre_mix_hc_transition();
    test_ratio2_per_dimension_softmax_pool();
    test_candidate_block_selection();
    puts("deepseek41 spec tests: ok");
    return 0;
}
