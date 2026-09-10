extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (n_rot > head_dim || !cuda_tensor_has_elems2(x, n_tok, head_dim, sizeof(float))) return 0;
    if (n_tok == 0u || head_dim == 0u) return 1;
    const uint32_t n_nope = head_dim - n_rot;
    if (n_nope == 0) return 1;
    const uint32_t groups = (n_nope + 63u) / 64u;
    fp8_kv_quantize_kernel<<<dim3(n_tok, groups), 64>>>((float *)x->ptr, n_tok, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize launch");
}

static int v41_cache_quantize_launch(ds4_gpu_tensor *x, uint32_t n_tok,
                                     uint32_t head_dim, uint32_t mode) {
    const uint32_t group_size = mode == 1u ? 16u : 32u;
    if (!cuda_tensor_has_elems2(x, n_tok, head_dim, sizeof(float)) ||
        n_tok == 0u || head_dim == 0u || mode > 3u ||
        (head_dim % group_size) != 0u) return 0;
    v41_cache_quantize_kernel<<<dim3(n_tok, head_dim / group_size), 32>>>(
        (float *)x->ptr, n_tok, head_dim, mode);
    return cuda_ok(cudaGetLastError(), "V4.1 cache quantize launch");
}
extern "C" int ds4_gpu_v41_window_kv_quantize_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim) {
    return v41_cache_quantize_launch(x, n_tok, head_dim, 0u);
}
extern "C" int ds4_gpu_v41_compressed_kv_quantize_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim) {
    return v41_cache_quantize_launch(x, n_tok, head_dim, 1u);
}
extern "C" int ds4_gpu_v41_round_bf16_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t width) {
    return v41_cache_quantize_launch(x, n_tok, width, 2u);
}
extern "C" int ds4_gpu_v41_indexer_qat_tensor(
        ds4_gpu_tensor *x, uint32_t n_rows, uint32_t head_dim) {
    return v41_cache_quantize_launch(x, n_rows, head_dim, 3u);
}
