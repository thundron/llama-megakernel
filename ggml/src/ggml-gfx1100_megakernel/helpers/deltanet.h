// deltanet.h — Gated Delta Net recurrence (single-token autoregressive path)
// Copied from ggml/src/ggml-cuda/gated_delta_net.cu lines 1-146
// Used by DeltaNet layers for state update + attention output.
#pragma once

#include "hip-shim.h"
#include "warp-reduce.h"

// --- gated_delta_net_cuda (non-KDA, single-token) --- gated_delta_net.cu:78-105
// Processes one token per call. Each warp owns one column of the state matrix.
// state layout: transposed — state[col * S_v + row] = S[row][col]
//
// GDA (gate-dependent attention): g is scalar per head
//   g_val = exp(g)
//   kv_col = sum_i(S[i][col] * k[i])
//   delta_col = (v[col] - g_val * kv_col) * beta
//   S[i][col] = g_val * S[i][col] + k[i] * delta_col
//   attn[col] = sum_i(S[i][col] * q[i]) * scale
//
// Parameters (all per-head, already extracted by the megakernel dispatch):
//   q, k: [S_k] — L2-normalized, scaled query and key
//   v:    [S_v] — value
//   g:    scalar — gate (= softplus(alpha + dt_bias) * ssm_a, then exp'd)
//   beta: scalar — sigmoid(beta_proj)
//   state: [S_v * S_v] — persistent recurrent state (transposed layout)
//   output: [S_v] — attention output for this head
//   S_v: state dimension (= key_dim = value_dim, typically 128)
//   scale: 1/sqrt(S_v)
template <int S_v>
static __device__ void deltanet_recurrence_gda(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float   g_val_log,  // gate BEFORE exp (will be exp'd here)
        const float   beta_val,   // sigmoid(beta_proj)
        float       * __restrict__ state,  // [S_v * S_v] transposed
        float       * __restrict__ output, // [S_v]
        const float   scale) {

    // Each warp owns one column (col = warp index within block)
    const int lane = threadIdx.x % WARP_SIZE;
    const int col  = threadIdx.x / WARP_SIZE; // which column this warp handles

    if (col >= S_v) return;

    constexpr int warp_size = WARP_SIZE < S_v ? WARP_SIZE : S_v;
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;

    // Load state column into registers
    float s_shard[rows_per_lane];
    float * col_state = state + col * S_v;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r] = (i < S_v) ? col_state[i] : 0.0f;
    }

    const float g_val = expf(g_val_log);

    // Cache k and q in registers
    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        k_reg[r] = (i < S_v) ? k[i] : 0.0f;
        q_reg[r] = (i < S_v) ? q[i] : 0.0f;
    }

    // kv[col] = sum_i(S[i][col] * k[i])
    float kv_shard = 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        kv_shard += s_shard[r] * k_reg[r];
    }
    float kv_col = warp_reduce_sum<warp_size>(kv_shard);

    // delta[col] = (v[col] - g * kv[col]) * beta
    float delta_col = (v[col] - g_val * kv_col) * beta_val;

    // Fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
    // attn[col] = sum_i(S[i][col] * q[i])
    float attn_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
        attn_partial += s_shard[r] * q_reg[r];
    }
    float attn_col = warp_reduce_sum<warp_size>(attn_partial);

    if (lane == 0) {
        output[col] = attn_col * scale;
    }

    // Write state back (transposed layout)
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        if (i < S_v) {
            col_state[i] = s_shard[r];
        }
    }
}

// --- KDA variant (key-dependent attention) --- gated_delta_net.cu:106-135
// g is per-element [S_v] instead of scalar
template <int S_v>
static __device__ void deltanet_recurrence_kda(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g_log, // [S_v] gate per element (BEFORE exp)
        const float   beta_val,
        float       * __restrict__ state,
        float       * __restrict__ output,
        const float   scale) {

    const int lane = threadIdx.x % WARP_SIZE;
    const int col  = threadIdx.x / WARP_SIZE;

    if (col >= S_v) return;

    constexpr int warp_size = WARP_SIZE < S_v ? WARP_SIZE : S_v;
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;

    float s_shard[rows_per_lane];
    float * col_state = state + col * S_v;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r] = (i < S_v) ? col_state[i] : 0.0f;
    }

    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        k_reg[r] = (i < S_v) ? k[i] : 0.0f;
        q_reg[r] = (i < S_v) ? q[i] : 0.0f;
    }

    // kv[col] = sum_i(exp(g[i]) * S[i][col] * k[i])
    float kv_shard = 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        float gi = (i < S_v) ? expf(g_log[i]) : 0.0f;
        kv_shard += gi * s_shard[r] * k_reg[r];
    }
    float kv_col = warp_reduce_sum<warp_size>(kv_shard);

    float delta_col = (v[col] - kv_col) * beta_val;

    float attn_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        float gi = (i < S_v) ? expf(g_log[i]) : 0.0f;
        s_shard[r] = gi * s_shard[r] + k_reg[r] * delta_col;
        attn_partial += s_shard[r] * q_reg[r];
    }
    float attn_col = warp_reduce_sum<warp_size>(attn_partial);

    if (lane == 0) {
        output[col] = attn_col * scale;
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        if (i < S_v) {
            col_state[i] = s_shard[r];
        }
    }
}
