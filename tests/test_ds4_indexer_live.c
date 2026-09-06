/* Exercise the model-defined sparse boundary through real Metal/SSD decode. */
#include "../ds4.c"
#include <assert.h>

int main(int argc, char **argv) {
    assert(argc == 3);
    bool streaming = !strcmp(argv[2], "ssd");
    assert(streaming || !strcmp(argv[2], "metal"));
    ds4_engine_options options = {
        .model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100, .warm_weights = true,
        .ssd_streaming = streaming,
        .ssd_streaming_cache_bytes = 8ull << 30
    };
    ds4_engine *engine = NULL;
    assert(ds4_engine_open(&engine, &options) == 0);
    ds4_tokens seed = {0}, prompt = {0};
    ds4_tokenize_text(engine, "A reliable cache preserves the model's selected memory entries.\n", &seed);
    assert(seed.len > 0);
    for (int i = 0; i < 2052; i++) token_vec_push(&prompt, seed.v[i % seed.len]);
    ds4_session *session = NULL;
    char error[512] = {0};
    assert(ds4_session_create(&session, engine, 4096) == 0);
    if (ds4_session_sync(session, &prompt, error, sizeof(error)) != 0) {
        fprintf(stderr, "%s\n", error); return 2;
    }
    for (int step = 0; step < 16; step++) {
        int token = ds4_session_argmax(session);
        if (ds4_session_eval(session, token, error, sizeof(error)) != 0) {
            fprintf(stderr, "%s\n", error); return 2;
        }
        for (uint32_t i = 0; i < DS4_N_VOCAB; i++) assert(isfinite(session->logits[i]));
        ds4_gpu_graph *graph = &session->graph;
        assert(metal_graph_decode_indexer_sparse_threshold(graph) == DS4_N_INDEXER_TOP_K);
        uint32_t rows = graph->layer_n_index_comp[DS4_N_LAYER - 1];
        assert(rows > DS4_N_INDEXER_TOP_K && rows < 1024);
        int32_t selected[512];
        assert(ds4_gpu_tensor_read(metal_graph_comp_selected(graph), 0,
                                   selected, sizeof(selected)) != 0);
        for (uint32_t i = 0; i < DS4_N_INDEXER_TOP_K; i++) {
            assert(selected[i] >= 0 && (uint32_t)selected[i] < rows);
            for (uint32_t j = 0; j < i; j++) assert(selected[i] != selected[j]);
        }
        printf("INDEXER_LIVE_STEP mode=%s step=%d rows=%u token=%d selected=512\n",
               argv[2], step, rows, token);
        fflush(stdout);
    }
    ds4_session_free(session);
    ds4_tokens_free(&seed); ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    printf("INDEXER_LIVE_PASS mode=%s\n", argv[2]);
    return 0;
}
