/* No weights/GPU required: legacy tuning must not broaden model visibility. */
#include "../ds4.c"
#include <assert.h>
int main(int argc, char **argv) {
    assert(argc == 2);
    if (!strcmp(argv[1], "default")) unsetenv("DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD");
    else setenv("DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD", argv[1], 1);
    ds4_gpu_graph g = {0};
    const uint32_t model_topk = DS4_N_INDEXER_TOP_K;
    assert(model_topk == 512u);
    uint32_t threshold = metal_graph_decode_indexer_sparse_threshold(&g);
    assert(threshold <= model_topk);
    /* Both predicates are required by scalar decode. At 513 rows it must
     * select 512, regardless of a legacy 1024/2048/4096 override. */
    for (uint32_t count = 0; count <= 4097; count++)
        assert((count > threshold && count > model_topk) == (count > model_topk));
    printf("INDEXER_CONTRACT_PASS setting=%s threshold=%u topk=%u\n", argv[1], threshold, model_topk);
    return 0;
}
