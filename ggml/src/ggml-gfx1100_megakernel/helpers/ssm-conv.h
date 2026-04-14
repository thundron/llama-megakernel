// ssm-conv.h — 1D causal convolution for SSM/DeltaNet
// Copied from ggml/src/ggml-cuda/ssm-conv.cu lines 4-47
// Used by DeltaNet layers for conv1d preprocessing of QKV.
#pragma once

#include "hip-shim.h"
#include "activations.h"

// --- ssm_conv_f32 device function --- ssm-conv.cu:4-47
// Single-token path (n_t=1): convolve one new input with conv_kernel-1 history values.
// conv_state: ring buffer [d_inner, d_conv] per channel
// conv_weight: [d_conv, d_inner] (row per channel, d_conv elements)
// apply_silu: optionally fuse SiLU activation on output
template <bool apply_silu, int d_conv>
static __device__ void ssm_conv_1token(
        const float * __restrict__ new_input,  // [d_inner] — new QKV projection for this token
        const float * __restrict__ conv_weight, // [d_conv * d_inner] — row-major: weight[ch * d_conv + t]
        float       * __restrict__ conv_state,  // [d_inner * d_conv] — ring buffer: state[ch * d_conv + t]
        float       * __restrict__ output,      // [d_inner] — convolution output
        const int d_inner,
        const int tid,
        const int block_size) {

    for (int ch = tid; ch < d_inner; ch += block_size) {
        int buf_base = ch * d_conv;

        // Shift history left by 1, insert new input at end
        // (hardcoded for d_conv=4, the common case for Qwen3.5)
#pragma unroll
        for (int t = 0; t < d_conv - 1; t++) {
            conv_state[buf_base + t] = conv_state[buf_base + t + 1];
        }
        conv_state[buf_base + d_conv - 1] = new_input[ch];

        // Convolve: output[ch] = sum_t(state[ch,t] * weight[ch,t])
        float sum = 0.0f;
#pragma unroll
        for (int t = 0; t < d_conv; t++) {
            sum += conv_state[buf_base + t] * conv_weight[ch * d_conv + t];
        }

        if constexpr (apply_silu) {
            output[ch] = op_silu(sum);
        } else {
            output[ch] = sum;
        }
    }
}
