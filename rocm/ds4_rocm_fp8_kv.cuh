// DS4 ROCm FP8 KV quantization and raw-cache store kernels.
//
// This file is included from ds4_cuda.cu (same translation unit) so these
// kernels can reuse the backend's existing device helpers without HIP device
// linking or behavior changes.

__global__ static void fp8_kv_quantize_kernel(float *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t row = blockIdx.x;
    const uint32_t grp = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t off = grp * 64u;
    if (row >= n_tok || off >= n_nope) return;
    float *xr = x + (uint64_t)row * head_dim;
    __shared__ float scratch[64];
    float v = 0.0f;
    if (tid < 64u && off + tid < n_nope) v = xr[off + tid];
    scratch[tid] = (tid < 64u && off + tid < n_nope) ? fabsf(v) : 0.0f;
    __syncthreads();
    for (uint32_t stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    const float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
    if (tid < 64u && off + tid < n_nope) {
        const float q = dsv4_e4m3fn_dequant_dev(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
        xr[off + tid] = q;
    }
}

__device__ static float v41_round_bf16_dev(float value) {
    union { float f; uint32_t u; } bits;
    bits.f = value;
    if ((bits.u & 0x7f800000u) != 0x7f800000u) {
        bits.u += 0x7fffu + ((bits.u >> 16u) & 1u);
    }
    bits.u &= 0xffff0000u;
    return bits.f;
}

__global__ static void v41_cache_quantize_kernel(
        float *x, uint32_t n_tok, uint32_t head_dim, uint32_t mode) {
    const uint32_t row = blockIdx.x;
    const uint32_t group = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (row >= n_tok || tid >= 32u) return;
    float *xr = x + (uint64_t)row * head_dim;
    if (mode == 2u) {
        const uint32_t i = group * 32u + tid;
        if (i < head_dim) xr[i] = v41_round_bf16_dev(xr[i]);
        return;
    }
    const uint32_t group_size = mode == 1u ? 16u : 32u;
    const uint32_t off = group * group_size;
    if (off >= head_dim) return;
    __shared__ float scratch[32];
    float v = 0.0f;
    if (tid < group_size) v = v41_round_bf16_dev(xr[off + tid]);
    scratch[tid] = tid < group_size ? fabsf(v) : 0.0f;
    __syncthreads();
    for (uint32_t stride = group_size >> 1u; stride != 0u; stride >>= 1u) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    if (tid < group_size) {
        float out;
        if (mode == 0u) {
            const float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
            out = dsv4_e4m3fn_dequant_dev(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
        } else if (mode == 1u) {
            const float scale = dsv4_e4m3fn_dequant_dev(fmaxf(scratch[0], 0.01171875f) / 6.0f);
            out = dsv4_e2m1fn_dequant_dev(fminf(6.0f, fmaxf(-6.0f, v / scale))) * scale;
        } else {
            const float scale = exp2f(ceilf(log2f(
                fmaxf(scratch[0], 7.052966104933725e-38f) / 6.0f)));
            out = dsv4_e2m1fn_dequant_dev(
                fminf(6.0f, fmaxf(-6.0f, v / scale))) * scale;
        }
        xr[off + tid] = v41_round_bf16_dev(out);
    }
}

__global__ static void store_raw_kv_batch_kernel(float *raw, const float *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t t = gid / head_dim;
    uint32_t row = (pos0 + t) % raw_cap;
    const uint16_t hb = f32_to_f16_bits_hip_round(kv[(uint64_t)t * head_dim + d]);
    raw[(uint64_t)row * head_dim + d] = f16_bits_to_f32(hb);
}
