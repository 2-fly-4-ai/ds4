/* 35B-A3B GDN geometry. Keep the working 27B specialization unchanged. */
#include <metal_stdlib>
using namespace metal;

constant uint QWEN35_GDN_V_HEADS = 32u;
constant uint QWEN35_GDN_K_HEADS = 16u;
constant uint QWEN35_GDN_HEAD_DIM = 128u;
constant uint QWEN35_GDN_CONV_K = 4u;

struct ds4_qwen35_gdn_conv_args {
    uint32_t n_channels;
    uint32_t layer;
    uint32_t n_tok;
    uint32_t save_steps;
};


struct ds4_qwen35_gdn_core_args {
    uint32_t layer;
    uint32_t n_tok;
    uint32_t save_steps;
};


kernel void kernel_qwen35_gdn_conv(
        constant ds4_qwen35_gdn_conv_args & args,
        device const float * qkv,
        device const float * conv_w,
        device float * conv,
        device float * mixed,
        device float * conv_steps,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_channels) return;
    device float *h = conv + ((uint)args.layer * args.n_channels + gid) * QWEN35_GDN_CONV_K;
    const device float *w = conv_w + gid * QWEN35_GDN_CONV_K;
    const uint ntok = args.n_tok == 0u ? 1u : args.n_tok;
    const uint layer_off = ((uint)args.layer * args.n_channels + gid) * QWEN35_GDN_CONV_K;
    const uint step_stride = 40u * args.n_channels * QWEN35_GDN_CONV_K;
    for (uint t = 0; t < ntok; t++) {
        const float x = qkv[t * args.n_channels + gid];
        h[0] = h[1];
        h[1] = h[2];
        h[2] = h[3];
        h[3] = x;
        float acc = w[0]*h[0] + w[1]*h[1] + w[2]*h[2] + w[3]*h[3];
        mixed[t * args.n_channels + gid] = acc / (1.0f + exp(-acc));
        if (args.save_steps != 0u && ntok > 1u) {
            device float *hs = conv_steps + t * step_stride + layer_off;
            hs[0] = h[0]; hs[1] = h[1]; hs[2] = h[2]; hs[3] = h[3];
        }

    }
}



kernel void kernel_qwen35_gdn_core(
        constant ds4_qwen35_gdn_core_args & args,
        device const float * mixed,
        device const float * z,
        device const float * alpha,
        device const float * beta,
        device const float * A_log,
        device const float * dt_bias,
        device const float * snorm,
        device float * state,
        device float * core,
        device float * state_steps,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint  lid   [[thread_index_in_threadgroup]]) {

    const uint vh = tgpig.x;
    const uint j = lid;
    if (vh >= QWEN35_GDN_V_HEADS || j >= QWEN35_GDN_HEAD_DIM) return;

    const uint kh = vh % QWEN35_GDN_K_HEADS;
    threadgroup float tq[QWEN35_GDN_HEAD_DIM];
    threadgroup float tk[QWEN35_GDN_HEAD_DIM];
    threadgroup float toh[QWEN35_GDN_HEAD_DIM];
    const uint ntok = args.n_tok == 0u ? 1u : args.n_tok;
    const uint sg = lid / 32u;

    for (uint t = 0; t < ntok; t++) {
        const device float *mixed_t = mixed + t * 8192u;
        const device float *z_t = z + t * 4096u;
        const device float *alpha_t = alpha + t * 32u;
        const device float *beta_t = beta + t * 32u;
        device float *core_t = core + t * 4096u;

        const device float *qsrc = mixed_t + kh * QWEN35_GDN_HEAD_DIM;
        const device float *ksrc = mixed_t + QWEN35_GDN_K_HEADS * QWEN35_GDN_HEAD_DIM + kh * QWEN35_GDN_HEAD_DIM;
        tq[j] = qsrc[j];
        tk[j] = ksrc[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float qn = tq[j] * tq[j];
        float kn = tk[j] * tk[j];
        qn += simd_shuffle_xor(qn, 1u);
        qn += simd_shuffle_xor(qn, 2u);
        qn += simd_shuffle_xor(qn, 4u);
        qn += simd_shuffle_xor(qn, 8u);
        qn += simd_shuffle_xor(qn, 16u);
        kn += simd_shuffle_xor(kn, 1u);
        kn += simd_shuffle_xor(kn, 2u);
        kn += simd_shuffle_xor(kn, 4u);
        kn += simd_shuffle_xor(kn, 8u);
        kn += simd_shuffle_xor(kn, 16u);
        threadgroup float qn_sg[4];
        threadgroup float kn_sg[4];
        if ((lid & 31u) == 0u) {
            qn_sg[sg] = qn;
            kn_sg[sg] = kn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        qn = qn_sg[0] + qn_sg[1] + qn_sg[2] + qn_sg[3];
        kn = kn_sg[0] + kn_sg[1] + kn_sg[2] + kn_sg[3];
        qn = 1.0f / max(sqrt(qn), 1e-6f);
        kn = 1.0f / max(sqrt(kn), 1e-6f);
        tq[j] *= qn;
        tk[j] *= kn;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float vj = mixed_t[4096u + vh * QWEN35_GDN_HEAD_DIM + j];
        float g = alpha_t[vh] + dt_bias[vh];
        if (g > 20.0f) {
        } else if (g < -20.0f) {
            g = exp(g);
        } else {
            g = log(1.0f + exp(g));
        }
        const float decay = exp(g * A_log[vh]);
        const float b = 1.0f / (1.0f + exp(-beta_t[vh]));

        device float *Sj = state + ((uint)args.layer * QWEN35_GDN_V_HEADS + vh) *
                           QWEN35_GDN_HEAD_DIM * QWEN35_GDN_HEAD_DIM + j * QWEN35_GDN_HEAD_DIM;
        float kv = 0.0f;
        for (uint i = 0; i < QWEN35_GDN_HEAD_DIM; i += 4u) {
            float4 s = *(device float4 *)(Sj + i);
            s *= decay;
            kv += s.x * tk[i] + s.y * tk[i + 1] + s.z * tk[i + 2] + s.w * tk[i + 3];
            *(device float4 *)(Sj + i) = s;
        }
        const float delta = (vj - kv) * b;
        float oh = 0.0f;
        const float qscale = 1.0f / sqrt((float)QWEN35_GDN_HEAD_DIM);
        for (uint i = 0; i < QWEN35_GDN_HEAD_DIM; i += 4u) {
            float4 s = *(device float4 *)(Sj + i);
            s.x += tk[i] * delta;
            s.y += tk[i + 1] * delta;
            s.z += tk[i + 2] * delta;
            s.w += tk[i + 3] * delta;
            *(device float4 *)(Sj + i) = s;
            oh += s.x * tq[i] + s.y * tq[i + 1] + s.z * tq[i + 2] + s.w * tq[i + 3];
        }
        oh *= qscale;
        toh[j] = oh;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float ss = toh[j] * toh[j];
        ss += simd_shuffle_xor(ss, 1u);
        ss += simd_shuffle_xor(ss, 2u);
        ss += simd_shuffle_xor(ss, 4u);
        ss += simd_shuffle_xor(ss, 8u);
        ss += simd_shuffle_xor(ss, 16u);
        threadgroup float ss_sg[4];
        if ((lid & 31u) == 0u) ss_sg[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ss = (ss_sg[0] + ss_sg[1] + ss_sg[2] + ss_sg[3]) / (float)QWEN35_GDN_HEAD_DIM;
        const float nscale = 1.0f / sqrt(ss + 1e-6f);
        const float zj = z_t[vh * QWEN35_GDN_HEAD_DIM + j];
        core_t[vh * QWEN35_GDN_HEAD_DIM + j] = oh * nscale * snorm[j] * (zj / (1.0f + exp(-zj)));
        if (args.save_steps != 0u && ntok > 1u) {
            const uint state_off = ((uint)args.layer * QWEN35_GDN_V_HEADS + vh) *
                                   QWEN35_GDN_HEAD_DIM * QWEN35_GDN_HEAD_DIM + j * QWEN35_GDN_HEAD_DIM;
            const uint step_stride = 40u * QWEN35_GDN_V_HEADS * QWEN35_GDN_HEAD_DIM * QWEN35_GDN_HEAD_DIM;
            device float *dst = state_steps + t * step_stride + state_off;
            for (uint i = 0; i < QWEN35_GDN_HEAD_DIM; i += 4u)
                *(device float4 *)(dst + i) = *(device float4 *)(Sj + i);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

    }
}
