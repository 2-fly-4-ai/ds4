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

static void test_engram_hash(void) {
    static const uint32_t primes[24] = {
        16000057, 16000079, 16000081, 16000097,
        16000121, 16000129, 16000133, 16000183,
        16000189, 16000207, 16000211, 16000253,
        16000277, 16000289, 16000307, 16000321,
        16000339, 16000381, 16000393, 16000399,
        16000403, 16000409, 16000447, 16000463,
    };
    static const uint64_t offsets[24] = {
        0, 16000057, 32000136, 48000217,
        64000314, 80000435, 96000564, 112000697,
        128000880, 144001069, 160001276, 176001487,
        192001740, 208002017, 224002306, 240002613,
        256002934, 272003273, 288003654, 304004047,
        320004446, 336004849, 352005258, 368005705,
    };
    static const uint64_t multipliers[4] = {
        76632096046245ull, 4839876093313ull,
        35959672319349ull, 73987337458391ull,
    };
    static const int32_t tokens[4] = {99091, 123, 45678, 2};
    static const uint64_t want[24] = {
        9940267, 17246986, 44062512, 59128963,
        70205933, 86825630, 104559419, 126322004,
        143303863, 153696633, 162723703, 182925277,
        205669089, 216562958, 228306068, 253536672,
        258157901, 275147698, 300999601, 309805922,
        322447513, 351565973, 361386815, 378039377,
    };
    uint64_t got[24] = {0};
    ds4_test_deepseek41_engram_hash(got, tokens, primes, offsets, multipliers);
    assert(memcmp(got, want, sizeof(want)) == 0);

    /* Exercise the negative-remainder normalization independently of valid
     * token-map values; DEAD is removed before production hashing. */
    const int32_t signed_tokens[4] = {-1, 0, 0, 0};
    const uint32_t tiny_primes[24] = {
        3,5,7,11,13,17,19,23, 29,31,37,41,43,47,53,59,
        61,67,71,73,79,83,89,97,
    };
    const uint64_t zero_offsets[24] = {0};
    const uint64_t one_multipliers[4] = {1, 0, 0, 0};
    ds4_test_deepseek41_engram_hash(got, signed_tokens, tiny_primes,
                                    zero_offsets, one_multipliers);
    for (uint32_t i = 0; i < 24; i++) assert(got[i] == tiny_primes[i] - 1u);
}

static void test_engram_gate(void) {
    const float h[4] = {1, 2, -2, 1};
    const float key[4] = {3, 4, 4, -3};
    const float value[2] = {0.5f, -0.25f};
    const float q_weight[4] = {1, 1, 1, 1};
    const float k_weight[4] = {1, 1, 1, 1};
    const float want[4] = {
        1.38243587f, 1.80878207f, -1.88243587f, 0.94121793f,
    };
    float got[4] = {0};
    ds4_test_deepseek41_engram_inject(got, h, key, value,
                                      q_weight, k_weight, 2, 2, 1.0e-6f);
    for (uint32_t i = 0; i < 4; i++) {
        assert(fabsf(got[i] - want[i]) < 2.0e-6f);
    }
}

int main(void) {
    test_profile();
    test_compress_ratios();
    test_shared_sources();
    test_reasoning_effort();
    test_previous_pre_mix_hc_transition();
    test_ratio2_per_dimension_softmax_pool();
    test_candidate_block_selection();
    test_engram_hash();
    test_engram_gate();
    puts("deepseek41 spec tests: ok");
    return 0;
}
