// activations.h — silu, sigmoid, softplus
// Copied from ggml/src/ggml-cuda/unary.cu
#pragma once

// --- silu --- unary.cuh:94-96
static __device__ __forceinline__ float op_silu(float x) {
    return x / (1.0f + expf(-x));
}

// --- sigmoid --- unary.cu:48-50
static __device__ __forceinline__ float op_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// --- softplus --- unary.cu:88-90
static __device__ __forceinline__ float op_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

// --- gelu (tanh approximation, default) --- unary.cuh:98-103
// Matches ggml_cuda_op_gelu_single — used by Gemma/Phi/Bert etc.
static __device__ __forceinline__ float op_gelu(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

// --- gelu_erf (exact GELU) --- unary.cu:24-28
// x * 0.5 * (1 + erf(x / sqrt(2)))
static __device__ __forceinline__ float op_gelu_erf(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210485f;
    return 0.5f * x * (1.0f + erff(x * SQRT_2_INV));
}

// --- gelu_quick --- unary.cu:30-34
// x * sigmoid(1.702 * x) — approximation used by some Phi variants
static __device__ __forceinline__ float op_gelu_quick(float x) {
    const float GELU_QUICK_COEF = -1.702f;
    return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

// --- relu² (squared ReLU) --- used by some Phi variants
static __device__ __forceinline__ float op_relu2(float x) {
    float y = x > 0.0f ? x : 0.0f;
    return y * y;
}

// --- ALiBi slope --- ported from ggml-cuda/common.cuh::get_alibi_slope (line 906-916).
// Used by BLOOM / Falcon / JAIS / MPT in attention to inject position bias as `slope * (kv_pos - q_pos)`.
// max_bias is hparams.f_max_alibi_bias; n_head_log2 = next_power_of_2_below(n_head);
// m0 = 2^(-8/n_head_log2); m1 = 2^(-4/n_head_log2).
static __device__ __forceinline__ float get_alibi_slope(
        const float max_bias, const unsigned int h, const unsigned int n_head_log2,
        const float m0, const float m1) {
    if (max_bias <= 0.0f) return 1.0f;
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? (int)h + 1 : 2 * (int)(h - n_head_log2) + 1;
    return powf(base, (float)exph);
}
