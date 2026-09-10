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

int main(void) {
    test_profile();
    test_compress_ratios();
    test_shared_sources();
    test_reasoning_effort();
    test_previous_pre_mix_hc_transition();
    puts("deepseek41 spec tests: ok");
    return 0;
}
