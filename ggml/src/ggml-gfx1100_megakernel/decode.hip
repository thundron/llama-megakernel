// decode.hip — Decode kernels for gfx1100 single-token decode
//
// Host-orchestrated: each phase is a separate kernel launch.
// No grid sync, no persistent kernel. Same pattern as prefill.hip.
// Model constants baked at compile time.
// All math ported from ggml-cuda baseline.

#include "helpers/hip-shim.h"
#include "helpers/warp-reduce.h"
#include "helpers/quant-types.h"
#include "helpers/iq-tables.h"
#include "helpers/vec-dot.h"
#include "helpers/rope.h"
#include "helpers/matvec.h"
#include "helpers/activations.h"
#include "helpers/dequant.h"
#include "helpers/block-reduce.h"
#include "helpers/rope.h"
#include "helpers/l2-norm.h"
#include "helpers/ssm-conv.h"
#include "helpers/deltanet.h"
#include "helpers/attention.h"
#include "helpers/attention-tile.h"
#ifdef GGML_USE_WMMA_FATTN
#include "helpers/attention-wmma.h"
#endif
#include "helpers/mem-utils.h"
#include "helpers/embed-dequant.h"
#include "helpers/mmq-quantize.h"

// =============================================================================
// Model constants (filled by template at compile time)
// =============================================================================

#ifndef HIDDEN_SIZE
#define HIDDEN_SIZE 2048
#endif
#ifndef INTERMEDIATE_SIZE
#define INTERMEDIATE_SIZE 8192
#endif
#ifndef NUM_LAYERS
#define NUM_LAYERS 16
#endif
#ifndef VOCAB_SIZE
#define VOCAB_SIZE 128256
#endif
#ifndef NORM_EPS
#define NORM_EPS 1e-5f
#endif
#ifndef NORM_ADD_ONE
#define NORM_ADD_ONE 0
#endif

#ifndef FA_N_Q_HEADS
#define FA_N_Q_HEADS 32
#endif
#ifndef FA_N_KV_HEADS
#define FA_N_KV_HEADS 8
#endif
#ifndef FA_HEAD_DIM
#define FA_HEAD_DIM 64
#endif
#ifndef FA_ROPE_THETA
#define FA_ROPE_THETA 500000.0f
#endif
#ifndef FA_ROPE_DIM
#define FA_ROPE_DIM 64
#endif
#ifndef FA_HAS_GATED_ATTN
#define FA_HAS_GATED_ATTN 0
#endif

#ifndef DN_N_HEADS
#define DN_N_HEADS 0
#endif
#ifndef DN_N_K_HEADS
#define DN_N_K_HEADS 0
#endif
#ifndef DN_KEY_DIM
#define DN_KEY_DIM 0
#endif
#ifndef DN_VALUE_DIM
#define DN_VALUE_DIM 0
#endif
#ifndef DN_CONV_KERNEL
#define DN_CONV_KERNEL 0
#endif

constexpr int FA_GQA_RATIO  = FA_N_Q_HEADS / (FA_N_KV_HEADS > 0 ? FA_N_KV_HEADS : 1);
constexpr int FA_Q_SIZE     = FA_N_Q_HEADS * FA_HEAD_DIM;
constexpr int FA_KV_SIZE    = FA_N_KV_HEADS * FA_HEAD_DIM;
constexpr int FA_QPROJ_SIZE = FA_HAS_GATED_ATTN ? FA_Q_SIZE * 2 : FA_Q_SIZE;
constexpr int DN_QK_SIZE    = DN_N_K_HEADS * DN_KEY_DIM;
constexpr int DN_V_SIZE     = DN_N_HEADS * DN_VALUE_DIM;
constexpr int DN_CONV_CH    = DN_QK_SIZE * 2 + DN_V_SIZE;

// =============================================================================
// Write decode params from host-uploaded GPU buffer to batch_token_ids
// Used to decouple token_id from H2D memcpy (enables graph reuse)
// =============================================================================

extern "C" __global__ void eval_write_decode_params(
        const int * __restrict__ d_params,
        int       * __restrict__ batch_token_ids) {
    if (threadIdx.x == 0) batch_token_ids[0] = d_params[0];
}

// =============================================================================
// Embedding: Q6_K dequant (from baseline convert.cu)
// =============================================================================

extern "C" __global__ void eval_embed_q6k(
        const void * __restrict__ embed_weight,
        const long long            embed_stride,
        float      * __restrict__ hidden,
        const int token_id) {
    const int nb = HIDDEN_SIZE / QK_K;
    const int b  = blockIdx.x;
    if (b >= nb) return;
    const block_q6_K * bq6 = (const block_q6_K *)((const char *)embed_weight + (long long)token_id * embed_stride);
    if (threadIdx.x < 64) {
        dequantize_block_q6_K_device(bq6, hidden, b, threadIdx.x);
    }
}

// Q4_K embedding dequant (32 threads per block, from baseline convert.cu)
extern "C" __global__ void eval_embed_q4k(
        const void * __restrict__ embed_weight,
        const long long            embed_stride,
        float      * __restrict__ hidden,
        const int token_id) {
    const int nb = HIDDEN_SIZE / QK_K;
    const int b  = blockIdx.x;
    if (b >= nb) return;
    const block_q4_K * bq4 = (const block_q4_K *)((const char *)embed_weight + (long long)token_id * embed_stride);
    if (threadIdx.x < 32) {
        dequantize_block_q4_K_device(bq4, hidden, b, threadIdx.x);
    }
}

// =============================================================================
// Embedding dequant for ALL types — macro to avoid repetition
// Pattern: cast row pointer to block type, call dequantize_block_TYPE_device
// =============================================================================

// K-quant embed: 64 threads, QK=256
#define EVAL_EMBED_KQUANT(name, block_type, dequant_fn, threads) \
extern "C" __global__ void eval_embed_##name( \
        const void * __restrict__ embed_weight, const long long embed_stride, \
        float * __restrict__ hidden, const int token_id) { \
    const int nb = HIDDEN_SIZE / QK_K; \
    const int b  = blockIdx.x; \
    if (b >= nb) return; \
    const block_type * row = (const block_type *)((const char *)embed_weight + (long long)token_id * embed_stride); \
    if (threadIdx.x < threads) dequant_fn(row, hidden, b, threadIdx.x); \
}

// Small-block embed (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/IQ4_NL) — DIRECT PORT of baseline
// ggml-cuda/getrows.cu k_get_rows<qk, qr, dequantize_qX> for single-token gather.
// Baseline launch: dim3(GET_ROWS_BLOCK_SIZE=256,1,1) block × dim3(1, ceildiv(H, 2*256), 1).
// Each thread iter handles 2 elements (i00, i00+y_offset). y_offset = qr==1 ? 1 : qk/2.
// Properly handles HIDDEN_SIZE % qk != 0 (per-element loop with `i00 < ne00` guard).
// Caller must invoke with block=(256,1,1) grid=(1, ceildiv(HIDDEN_SIZE,2*256), 1).
#define EVAL_EMBED_GETROWS(name, qk_const, qr_const, dequant_fn) \
extern "C" __global__ void eval_embed_##name( \
        const void * __restrict__ embed_weight, const long long embed_stride, \
        float * __restrict__ hidden, const int token_id) { \
    const int qk = qk_const; \
    const int qr = qr_const; \
    const int y_offset = qr == 1 ? 1 : qk / 2; \
    const void * src0_row = (const char *)embed_weight + (long long)token_id * embed_stride; \
    for (int i00 = 2 * (blockIdx.y * blockDim.x + threadIdx.x); \
         i00 < HIDDEN_SIZE; \
         i00 += gridDim.y * blockDim.x * 2) { \
        const long long ib   = i00 / qk; \
        const int iqs  = (i00 % qk) / qr; \
        const int iybs = i00 - i00 % qk; \
        float2 v; \
        dequant_fn(src0_row, ib, iqs, v); \
        hidden[iybs + iqs + 0]        = v.x; \
        hidden[iybs + iqs + y_offset] = v.y; \
    } \
}

// Legacy (broken pre-session, kept for K-quant variants only). Will be replaced
// when K-quant get_rows is needed for non-256-aligned H.
#define EVAL_EMBED_SMALL(name, block_type, dequant_fn) \
extern "C" __global__ void eval_embed_##name( \
        const void * __restrict__ embed_weight, const long long embed_stride, \
        float * __restrict__ hidden, const int token_id) { \
    const int nb = HIDDEN_SIZE / QK_K; \
    const int b  = blockIdx.x; \
    if (b >= nb) return; \
    const void * row = (const char *)embed_weight + (long long)token_id * embed_stride; \
    dequant_fn(row, hidden, b, threadIdx.x); \
}

// Direct ports of baseline get_rows for SMALL types (constants from baseline ggml-common.h):
// QK4_0=32, QR4_0=2; QK4_1=32, QR4_1=2; QK5_0=32, QR5_0=2; QK5_1=32, QR5_1=2; QK8_0=32, QR8_0=1
EVAL_EMBED_GETROWS(q4_0, 32, 2, baseline_dequantize_q4_0)
EVAL_EMBED_GETROWS(q4_1, 32, 2, baseline_dequantize_q4_1)
EVAL_EMBED_GETROWS(q5_0, 32, 2, baseline_dequantize_q5_0)
EVAL_EMBED_GETROWS(q5_1, 32, 2, baseline_dequantize_q5_1)
EVAL_EMBED_GETROWS(q8_0, 32, 1, baseline_dequantize_q8_0)
EVAL_EMBED_KQUANT(q2k,  block_q2_K,  dequantize_block_q2_K_device, 64)
EVAL_EMBED_KQUANT(q3k,  block_q3_K,  dequantize_block_q3_K_device, 64)
EVAL_EMBED_KQUANT(q5k,  block_q5_K,  dequantize_block_q5_K_device, 64)
// IQ types: 32 threads each
EVAL_EMBED_KQUANT(iq2_xxs, block_iq2_xxs, dequantize_block_iq2_xxs_device, 32)
EVAL_EMBED_KQUANT(iq2_xs,  block_iq2_xs,  dequantize_block_iq2_xs_device,  32)
EVAL_EMBED_KQUANT(iq2_s,   block_iq2_s,   dequantize_block_iq2_s_device,   32)
EVAL_EMBED_KQUANT(iq3_xxs, block_iq3_xxs, dequantize_block_iq3_xxs_device, 32)
EVAL_EMBED_KQUANT(iq3_s,   block_iq3_s,   dequantize_block_iq3_s_device,   32)
EVAL_EMBED_KQUANT(iq1_s,   block_iq1_s,   dequantize_block_iq1_s_device,   32)
EVAL_EMBED_KQUANT(iq1_m,   block_iq1_m,   dequantize_block_iq1_m_device,   32)
EVAL_EMBED_SMALL(iq4_nl,  void, dequantize_block_iq4_nl_device)
EVAL_EMBED_KQUANT(iq4_xs, block_iq4_xs, dequantize_block_iq4_xs_device,   32)
EVAL_EMBED_SMALL(mxfp4,   void, dequantize_block_mxfp4_device)

#undef EVAL_EMBED_KQUANT
#undef EVAL_EMBED_SMALL

// NVFP4 embedding (QK=64, different superblock size)
extern "C" __global__ void eval_embed_nvfp4(
        const void * __restrict__ embed_weight, const long long embed_stride,
        float * __restrict__ hidden, const int token_id) {
    const int nb = HIDDEN_SIZE / 512;  // NVFP4: 8 blocks of 64 = 512 per superblock
    const int b  = blockIdx.x;
    if (b >= nb) return;
    const void * row = (const char *)embed_weight + (long long)token_id * embed_stride;
    dequantize_block_nvfp4_device(row, hidden, b, threadIdx.x);
}

// F32 embedding (direct copy)
extern "C" __global__ void eval_embed_f32(
        const void * __restrict__ embed_weight, const long long embed_stride,
        float * __restrict__ hidden, const int token_id) {
    const float * row = (const float *)((const char *)embed_weight + (long long)token_id * embed_stride);
    dequantize_block_f32_device(row, hidden, 0, threadIdx.x + blockIdx.x * blockDim.x, HIDDEN_SIZE);
}

// F16 embedding (half → float)
extern "C" __global__ void eval_embed_f16(
        const void * __restrict__ embed_weight, const long long embed_stride,
        float * __restrict__ hidden, const int token_id) {
    const __half * row = (const __half *)((const char *)embed_weight + (long long)token_id * embed_stride);
    dequantize_block_f16_device(row, hidden, 0, threadIdx.x + blockIdx.x * blockDim.x, HIDDEN_SIZE);
}

// BF16 embedding (bf16 → float)
extern "C" __global__ void eval_embed_bf16(
        const void * __restrict__ embed_weight, const long long embed_stride,
        float * __restrict__ hidden, const int token_id) {
    const uint16_t * row = (const uint16_t *)((const char *)embed_weight + (long long)token_id * embed_stride);
    dequantize_block_bf16_device(row, hidden, 0, threadIdx.x + blockIdx.x * blockDim.x, HIDDEN_SIZE);
}

// =============================================================================
// RMSNorm: input → norm_out (f32), saves residual, quantizes to Q8_1
// From baseline norm.cu rms_norm_f32 (do_multiply=true)
// =============================================================================

// LayerNorm — ported from ggml-cuda/norm.cu::norm_f32
// Computes (x - mean) / sqrt(var + eps), then optional `*w + b` (w,b may be NULL).
// Matches baseline build_norm(LLM_NORM) + ggml_mul(mw) + ggml_add(mb).
// Single-row variant (decode): gridDim.x=1, blockDim.x=512.
extern "C" __global__ void eval_layernorm(
        const float * __restrict__ input,
        const float * __restrict__ weight,   // may be NULL
        const float * __restrict__ bias,     // may be NULL
        float       * __restrict__ norm_out,
        float       * __restrict__ residual, // copy of input for post-norm residual add
        const int n,
        const float eps) {
    const int tid = threadIdx.x;
    __shared__ float2 smem2[32]; // up to 32 warps

    // Pass 1: mean + mean-of-squares + residual copy
    float2 ms = make_float2(0.0f, 0.0f);
    for (int i = tid; i < n; i += blockDim.x) {
        float v = input[i];
        if (residual) residual[i] = v;
        ms.x += v;
        ms.y += v * v;
    }
    ms.x = warp_reduce_sum(ms.x);
    ms.y = warp_reduce_sum(ms.y);
    int wid = tid / WARP_SIZE, lid = tid % WARP_SIZE;
    if (lid == 0) smem2[wid] = ms;
    __syncthreads();
    if (wid == 0) {
        float2 s = (lid < (int)(blockDim.x / WARP_SIZE)) ? smem2[lid] : make_float2(0.0f, 0.0f);
        s.x = warp_reduce_sum(s.x);
        s.y = warp_reduce_sum(s.y);
        if (lid == 0) {
            const float inv_n = 1.0f / (float)n;
            const float mean  = s.x * inv_n;
            const float var   = s.y * inv_n - mean * mean;
            smem2[0].x = mean;
            smem2[0].y = rsqrtf(var + eps);
        }
    }
    __syncthreads();
    const float mean    = smem2[0].x;
    const float inv_std = smem2[0].y;

    // Pass 2: normalize + optional weight/bias fold
    for (int i = tid; i < n; i += blockDim.x) {
        float y = (input[i] - mean) * inv_std;
        if (weight) y *= weight[i];
        if (bias)   y += bias[i];
        norm_out[i] = y;
    }
}

// L2 norm (per-row) — ported from ggml-cuda/norm.cu::l2_norm_f32.
// Computes x / sqrt(sum(x²) + eps). Used by some Qwen3 variants on conv output.
extern "C" __global__ void eval_l2norm(
        const float * __restrict__ input,
        float       * __restrict__ output,
        const int n,
        const float eps) {
    const int tid = threadIdx.x;
    __shared__ float smem[32];

    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = input[i];
        ss += v * v;
    }
    ss = warp_reduce_sum(ss);
    int wid = tid / WARP_SIZE, lid = tid % WARP_SIZE;
    if (lid == 0) smem[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        float s = (lid < (int)(blockDim.x / WARP_SIZE)) ? smem[lid] : 0.0f;
        s = warp_reduce_sum(s);
        if (lid == 0) smem[0] = rsqrtf(fmaxf(s, eps * eps));
    }
    __syncthreads();
    float rstd = smem[0];

    for (int i = tid; i < n; i += blockDim.x) {
        output[i] = input[i] * rstd;
    }
}

// Sigmoid (element-wise) — ported from ggml-cuda/unary.cu::op_sigmoid
extern "C" __global__ void eval_sigmoid(
        const float * __restrict__ input,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = op_sigmoid(input[i]);
}

// Softplus (element-wise) — ported from ggml-cuda/unary.cu::op_softplus
extern "C" __global__ void eval_softplus(
        const float * __restrict__ input,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = op_softplus(input[i]);
}

// Tanh (element-wise) — ported from ggml-cuda/unary.cu::op_tanh
extern "C" __global__ void eval_tanh(
        const float * __restrict__ input,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = tanhf(input[i]);
}

// Negate (element-wise)
extern "C" __global__ void eval_neg(
        const float * __restrict__ input,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = -input[i];
}

// Exp (element-wise) — ported from baseline ggml_exp
extern "C" __global__ void eval_exp(
        const float * __restrict__ input,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = expf(input[i]);
}

// ReLU (element-wise) — ported from ggml-cuda/unary.cu::op_relu
extern "C" __global__ void eval_relu(
        const float * __restrict__ input,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = fmaxf(input[i], 0.0f);
}

// Square (element-wise) — ported from ggml_sqr
extern "C" __global__ void eval_sqr(
        const float * __restrict__ input,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = input[i] * input[i];
}

// dst[i] = a[i] * (1.0 - b[i]) — RWKV6Qwen2: k = k * (1 - w)
// Ported from baseline rwkv6-base.cpp line 129: ggml_mul(ggml_sub(one, w), k)
extern "C" __global__ void eval_mul_one_minus(
        float       * __restrict__ dst,    // [N] output (can alias a)
        const float * __restrict__ a,      // [N] input (k)
        const float * __restrict__ b,      // [N] subtracted (w)
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = a[i] * (1.0f - b[i]);
}

// dst[i] = src[i / repeat_count] — expand [n_head] → [d_inner] by repeating each value head_dim times
// Mamba2: dt is [n_head], scan kernel expects [d_inner] where d_inner = n_head * head_dim
extern "C" __global__ void eval_repeat_interleave(
        float       * __restrict__ dst,     // [d_inner]
        const float * __restrict__ src,     // [n_head]
        const int head_dim,
        const int d_inner) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_inner) dst[i] = src[i / head_dim];
}

// Per-head dot product + scale + add — RWKV7 r*k additive correction
// For each head h: out[h*hs..h*hs+hs-1] += sum(r[h*hs..]*k[h*hs..])*rk[h*hs..]*v[h*hs..]
// Ported from baseline rwkv7.cpp: r_k correction (lines 154-160)
extern "C" __global__ void eval_rwkv7_rk_correction(
        float       * __restrict__ output,  // [H] WKV output — modified in-place
        const float * __restrict__ r,       // [H] receptance
        const float * __restrict__ k,       // [H] key
        const float * __restrict__ v,       // [H] value
        const float * __restrict__ r_k,     // [H] r_k weight
        const int head_size,
        const int n_head) {
    int h = blockIdx.x; // one block per head
    int tid = threadIdx.x;
    if (h >= n_head) return;

    const float * rh = r + h * head_size;
    const float * kh = k + h * head_size;

    // Step 1: weighted dot product sum(r * k * r_k) for this head
    const float * rk_h = r_k + h * head_size;
    __shared__ float smem[32];
    float dot = 0.0f;
    for (int i = tid; i < head_size; i += blockDim.x)
        dot += rh[i] * kh[i] * rk_h[i];
    dot = warp_reduce_sum(dot);
    int wid = tid / 32, lid = tid % 32;
    if (lid == 0) smem[wid] = dot;
    __syncthreads();
    if (wid == 0) {
        float v2 = (lid < (int)(blockDim.x / 32)) ? smem[lid] : 0.0f;
        v2 = warp_reduce_sum(v2);
        if (lid == 0) smem[0] = v2;
    }
    __syncthreads();
    float rk_dot = smem[0]; // scalar: sum(r[h]*k[h])

    // Step 2: output[h*hs + i] += rk_dot * v[h*hs + i]
    // (r_k weighting was already included in the dot product)
    const float * vb = v + h * head_size;
    float * out = output + h * head_size;
    for (int i = tid; i < head_size; i += blockDim.x)
        out[i] += rk_dot * vb[i];
}

// SiLU (element-wise, standalone without mul) — ported from ggml-cuda/unary.cu::op_silu
extern "C" __global__ void eval_silu(
        const float * __restrict__ input,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = input[i] / (1.0f + expf(-input[i]));
}

// Sub (element-wise): dst = a - b — baseline ggml_sub
extern "C" __global__ void eval_sub(
        const float * __restrict__ a,
        const float * __restrict__ b,
        float       * __restrict__ dst,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = a[i] - b[i];
}

// Mul-add (element-wise): dst = a * b + c — for lerp computation
extern "C" __global__ void eval_muladd(
        const float * __restrict__ a,
        const float * __restrict__ b,
        const float * __restrict__ c,
        float       * __restrict__ dst,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = a[i] * b[i] + c[i];
}

// Weighted accumulate: dst[i] += w * src[i].
// MoE: accumulate weight * expert_output across selected experts.
extern "C" __global__ void eval_axpy(
        float       * __restrict__ dst,
        const float * __restrict__ src,
        const float w,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] += w * src[i];
}

// Sum reduction over a row → scalar (for MoE weight normalization).
// out_sum = sum_{i=0..N-1}(in[i]). Single-block, 1024 threads.
extern "C" __global__ void eval_sum_row(
        const float * __restrict__ in,
        float       * __restrict__ out_sum,
        const int N) {
    const int tid = threadIdx.x;
    __shared__ float smem[32];
    float s = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) s += in[i];
    s = warp_reduce_sum(s);
    int lane = tid % WARP_SIZE, wid = tid / WARP_SIZE;
    if (lane == 0) smem[wid] = s;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < (int)(blockDim.x / WARP_SIZE)) ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) *out_sum = v;
    }
}

// SSM conv1d single-step (Mamba decode) — adapted from ggml-cuda/ssm-conv.cu::ssm_conv_f32 for n_t=1.
// state[d_inner * d_conv] is updated in-place: shifted left by 1, new x_in placed at position d_conv-1.
// output[i] = silu(sum_{j=0..d_conv-1}(state[i*d_conv + j] * w[i*d_conv + j])) when apply_silu=1
// Standalone RoPE NEOX — apply rotation in-place to a float array.
// Ported from baseline rope.cu rope_neox pattern. NEOX pairing: (x[i], x[i+n/2])
// Grid: (1,1,1), Block: (n_dim/2, 1, 1) or larger
// For DS2 MLA, T5 relative bias, and any arch that needs RoPE on arbitrary slices
extern "C" __global__ void eval_rope_neox_inplace(
        float       * __restrict__ data,     // [n_dim] in-place
        const int position,
        const float rope_theta,
        const int n_dim) {
    const int i = threadIdx.x;
    if (i >= n_dim / 2) return;
    float theta = (float)position * powf(rope_theta, -(float)(i * 2) / (float)n_dim);
    float cv = cosf(theta), sv = sinf(theta);
    float x0 = data[i];
    float x1 = data[i + n_dim / 2];
    data[i]            = x0 * cv - x1 * sv;
    data[i + n_dim / 2] = x0 * sv + x1 * cv;
}

// Shape conventions match baseline: w is [d_conv, d_inner] row-major (w_block[i * stride_w + j]).
extern "C" __global__ void eval_ssm_conv_step(
        const float * __restrict__ x_in,        // [d_inner] new input column
        float       * __restrict__ state,       // [d_inner * d_conv] persistent state, updated
        const float * __restrict__ w,           // [d_inner * d_conv] conv weights
        float       * __restrict__ y_out,       // [d_inner] output
        const int d_inner,
        const int d_conv,
        const int apply_silu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_inner) return;

    // Shift state left by 1 (drop oldest), append new x_in[i] at last slot.
    float * s = state + i * d_conv;
    for (int j = 0; j < d_conv - 1; j++) s[j] = s[j+1];
    s[d_conv - 1] = x_in[i];

    // Convolution: dot(state, weights) for this channel
    const float * wi = w + i * d_conv;
    float sumf = 0.0f;
    for (int j = 0; j < d_conv; j++) sumf += s[j] * wi[j];
    y_out[i] = apply_silu ? op_silu(sumf) : sumf;
}

// SSM selective-scan single-step — adapted from ggml-cuda/ssm-scan.cu for n_t=1.
// Mamba selective state: h = exp(dt * A) * h + dt * x * B; y = sum_n(h * C); y += D * x
//   x:    [d_inner]
//   dt:   [d_inner]            (delta — selectivity factor)
//   A:    [d_inner * d_state]  (state transition matrix, log-stored: A_actual = -exp(A))
//   B:    [d_state]            (input projection)
//   C:    [d_state]            (output projection)
//   D:    [d_inner]            (skip connection)
//   h:    [d_inner * d_state]  (recurrent state, updated in place)
//   y:    [d_inner]            output
// Each thread handles one d_inner channel; iterates d_state for inner sum.
extern "C" __global__ void eval_ssm_scan_step(
        const float * __restrict__ x,
        const float * __restrict__ dt,
        const float * __restrict__ A,
        const float * __restrict__ B,
        const float * __restrict__ C,
        const float * __restrict__ D,
        float       * __restrict__ h,
        float       * __restrict__ y,
        const int d_inner,
        const int d_state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_inner) return;

    const float xi  = x[i];
    // Apply softplus to dt — baseline ssm-scan.cu line 83-87
    float dti = dt[i];
    if (dti <= 20.0f) dti = log1pf(expf(dti));
    const float * Ai = A + i * d_state;
    float * hi = h + i * d_state;

    float yi = 0.0f;
    for (int n = 0; n < d_state; n++) {
        // baseline: A is stored as log; effective transition = -exp(A)
        const float a_eff = -expf(Ai[n]);
        // Discretize: dA = exp(dt * a_eff); dB = dt * B[n] * x[i]
        const float dA = expf(dti * a_eff);
        const float dBx = dti * B[n] * xi;
        // Recurrence: h_new = dA * h_prev + dBx
        float hin = dA * hi[n] + dBx;
        hi[n] = hin;
        yi += hin * C[n];
    }
    if (D) yi += D[i] * xi;
    y[i] = yi;
}

// =============================================================================
// RWKV WKV6 single-step — ported from ggml-cuda/wkv.cu::rwkv_wkv_f32
// Specialized for decode: B=1, T=1. One block per head, blockDim.x = head_size.
// Grid: (n_head, 1, 1), Block: (head_size, 1, 1)
// state: [n_head, head_size, head_size] — persistent per-layer
// k, v, r, tf(first), td(decay): [C] where C = n_head * head_size
// y: [C] output
// =============================================================================
extern "C" __global__ void eval_rwkv_wkv6_step(
        const float * __restrict__ k,     // [C]
        const float * __restrict__ v,     // [C]
        const float * __restrict__ r,     // [C]
        const float * __restrict__ tf,    // [C] time_first weight
        const float * __restrict__ td,    // [C] time_decay (already exp(-exp(w)))
        float       * __restrict__ state, // [n_head, head_size, head_size] in/out
        float       * __restrict__ y,     // [C] output
        const int head_size,
        const int n_head) {
    const int tid = threadIdx.x;
    const int head_i = blockIdx.x;
    if (head_i >= n_head) return;
    if (tid >= head_size) return;

    const int C = n_head * head_size;
    const int t = head_i * head_size + tid; // index into k,v,r,tf,td,y

    // Load state row: state[head_i][tid][0..head_size-1]
    float * s = state + head_i * head_size * head_size + tid * head_size;

    // Baseline wkv.cu lines 29-59 (inner loop for one timestep)
    const float _k_tid = k[t];
    const float _r_tid = r[t];
    const float _tf_tid = tf[head_i * head_size + tid]; // time_first is per-head
    const float _td_tid = td[t];
    const float _v = v[t];

    // Load k, r, tf, td into shared memory for cross-thread access
    extern __shared__ float smem[];
    float * _k_shared  = smem;
    float * _r_shared  = _k_shared + head_size;
    float * _tf_shared = _r_shared + head_size;
    float * _td_shared = _tf_shared + head_size;

    _k_shared[tid]  = _k_tid;
    _r_shared[tid]  = _r_tid;
    _tf_shared[tid] = _tf_tid;
    _td_shared[tid] = _td_tid;
    __syncthreads();

    float y_val = 0;
    for (int j = 0; j < head_size; j++) {
        float kv = _k_shared[j] * _v;
        y_val += _r_shared[j] * (_tf_shared[j] * kv + s[j]);
        s[j] = s[j] * _td_shared[j] + kv;
    }
    y[t] = y_val;
}

// =============================================================================
// RWKV WKV7 single-step — ported from ggml-cuda/wkv.cu::rwkv_wkv7_f32
// Same structure as WKV6 but with a,b params instead of tf(first).
// Grid: (n_head, 1, 1), Block: (head_size, 1, 1)
// =============================================================================
extern "C" __global__ void eval_rwkv_wkv7_step(
        const float * __restrict__ r,     // [C]
        const float * __restrict__ w,     // [C] decay
        const float * __restrict__ k,     // [C]
        const float * __restrict__ v,     // [C]
        const float * __restrict__ a,     // [C]
        const float * __restrict__ b_vec, // [C]
        float       * __restrict__ state, // [n_head, head_size, head_size] in/out
        float       * __restrict__ y,     // [C] output
        const int head_size,
        const int n_head) {
    const int tid = threadIdx.x;
    const int head_i = blockIdx.x;
    if (head_i >= n_head) return;
    if (tid >= head_size) return;

    const int t = head_i * head_size + tid;

    float * s = state + head_i * head_size * head_size + tid * head_size;

    extern __shared__ float smem7[];
    float * _r_s = smem7;
    float * _w_s = _r_s + head_size;
    float * _k_s = _w_s + head_size;
    float * _a_s = _k_s + head_size;
    float * _b_s = _a_s + head_size;

    _r_s[tid] = r[t];
    _w_s[tid] = w[t];
    _k_s[tid] = k[t];
    _a_s[tid] = a[t];
    _b_s[tid] = b_vec[t];
    __syncthreads();

    // Compute sa = dot(a, state[tid])
    float sa = 0;
    for (int j = 0; j < head_size; j++) {
        sa += _a_s[j] * s[j];
    }

    const float _v = v[t];
    float y_val = 0;
    for (int j = 0; j < head_size; j++) {
        float kv = _k_s[j] * _v;
        s[j] = s[j] * _w_s[j] + kv + sa * _b_s[j];
        y_val += s[j] * _r_s[j];
    }
    y[t] = y_val;
}

// Concat along dim0 — ported from ggml-cuda/concat.cu::concat_f32_dim0.
// dst[blockIdx.z * ne0*ne1 + blockIdx.y * ne0 + nidx] = x or y depending on nidx<ne00
extern "C" __global__ void eval_concat_dim0(
        const float * __restrict__ x,
        const float * __restrict__ y,
        float       * __restrict__ dst,
        const int ne0,
        const int ne00) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) return;
    int offset_dst = nidx + blockIdx.y * ne0 + blockIdx.z * ne0 * gridDim.y;
    if (nidx < ne00) {
        int offset_src = nidx + blockIdx.y * ne00 + blockIdx.z * ne00 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src = (nidx - ne00) + blockIdx.y * (ne0 - ne00) + blockIdx.z * (ne0 - ne00) * gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

// Concat along dim1 — ported from ggml-cuda/concat.cu::concat_f32_dim1.
extern "C" __global__ void eval_concat_dim1(
        const float * __restrict__ x,
        const float * __restrict__ y,
        float       * __restrict__ dst,
        const int ne0,
        const int ne01) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) return;
    int offset_dst = nidx + blockIdx.y * ne0 + blockIdx.z * ne0 * gridDim.y;
    if (blockIdx.y < (unsigned)ne01) {
        int offset_src = nidx + blockIdx.y * ne0 + blockIdx.z * ne0 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src = nidx + (blockIdx.y - ne01) * ne0 + blockIdx.z * ne0 * (gridDim.y - ne01);
        dst[offset_dst] = y[offset_src];
    }
}

// Repeat along dim0 — broadcast x across ne0 / ne00 copies.
// baseline: ggml_repeat — ne0 = ne00 * factor.
extern "C" __global__ void eval_repeat_dim0(
        const float * __restrict__ x,
        float       * __restrict__ dst,
        const int ne00,
        const int ne0) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) return;
    int row = blockIdx.y;
    dst[row * ne0 + nidx] = x[row * ne00 + (nidx % ne00)];
}

// Bitonic argsort (descending) — ported from ggml-cuda/argsort.cu k_argsort_f32_i32.
// Returns indices of the sorted values in descending order. Used for MoE top-k expert routing.
// grid=(nrows,1,1), block=(ncols_pad,1,1), shared = ncols_pad * sizeof(int).
// ncols_pad must be next power of 2 ≥ ncols.
extern "C" __global__ void eval_argsort_desc(
        const float * __restrict__ x,
        int         * __restrict__ dst,
        const int ncols,
        const int ncols_pad) {
    int col = threadIdx.x;
    int row = blockIdx.x;
    if (col >= ncols_pad) return;

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];
    dst_row[col] = col;
    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (x_row[dst_row[col]] < x_row[dst_row[ixj]]))) {
                        int tmp = dst_row[col]; dst_row[col] = dst_row[ixj]; dst_row[ixj] = tmp;
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (x_row[dst_row[col]] > x_row[dst_row[ixj]]))) {
                        int tmp = dst_row[col]; dst_row[col] = dst_row[ixj]; dst_row[ixj] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }
    if (col < ncols) dst[row * ncols + col] = dst_row[col];
}

// Softmax over a row — simplified from ggml-cuda/softmax.cu::soft_max_f32.
// No mask / ALiBi / scale — pass `scale_val` = 1.0 for plain softmax.
// grid=(nrows,1,1), block=(1024,1,1) typical. Shared = max(WARP_SIZE, blockDim.x) * sizeof(float).
// Used for MoE expert gating + attention softmax outside of flash attention.
extern "C" __global__ void eval_softmax_row(
        const float * __restrict__ x,
        float       * __restrict__ dst,
        const int ncols,
        const float scale_val) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nwarps = blockDim.x / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int wid  = tid / WARP_SIZE;

    extern __shared__ float smem_sm[];  // size: nwarps floats
    const float * src = x   + row * ncols;
    float       * dst_row = dst + row * ncols;

    // Pass 1: max
    float vmax = -INFINITY;
    for (int c = tid; c < ncols; c += blockDim.x) {
        float v = src[c] * scale_val;
        if (v > vmax) vmax = v;
    }
    // warp reduce max
    for (int o = WARP_SIZE/2; o > 0; o >>= 1) {
        float other = __shfl_xor(vmax, o);
        if (other > vmax) vmax = other;
    }
    if (lane == 0) smem_sm[wid] = vmax;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < nwarps) ? smem_sm[lane] : -INFINITY;
        for (int o = WARP_SIZE/2; o > 0; o >>= 1) {
            float other = __shfl_xor(v, o);
            if (other > v) v = other;
        }
        if (lane == 0) smem_sm[0] = v;
    }
    __syncthreads();
    vmax = smem_sm[0];

    // Pass 2: exp and sum
    float sum = 0.0f;
    for (int c = tid; c < ncols; c += blockDim.x) {
        float e = expf(src[c] * scale_val - vmax);
        dst_row[c] = e;
        sum += e;
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) smem_sm[wid] = sum;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < nwarps) ? smem_sm[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem_sm[0] = v;
    }
    __syncthreads();
    const float inv_sum = 1.0f / smem_sm[0];

    // Pass 3: normalize
    for (int c = tid; c < ncols; c += blockDim.x) {
        dst_row[c] *= inv_sum;
    }
}

extern "C" __global__ void eval_rmsnorm_q8(
        const float * __restrict__ input,
        const float * __restrict__ weight,
        float       * __restrict__ norm_out,
        float       * __restrict__ residual,
        const int n) {

    const int tid = threadIdx.x;
    __shared__ float smem[32];

    constexpr int MAX_PER_THREAD = 16;
    float local_vals[MAX_PER_THREAD];
    int count = 0;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = input[i];
        local_vals[count++] = v;
        residual[i] = v;
        ss += v * v;
    }

    ss = warp_reduce_sum(ss);
    int wid = tid / WARP_SIZE, lid = tid % WARP_SIZE;
    if (lid == 0) smem[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        float s = (lid < (int)(blockDim.x / WARP_SIZE)) ? smem[lid] : 0.0f;
        s = warp_reduce_sum(s);
        if (lid == 0) smem[0] = rsqrtf(s / (float)n + NORM_EPS);
    }
    __syncthreads();
    float rstd = smem[0];

    count = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        float w = weight[i];
#if NORM_ADD_ONE
        w = 1.0f + w;
#endif
        norm_out[i] = local_vals[count++] * rstd * w;
    }
}

// Fused RMSNorm + Q8_1 quantize: norm_out AND q8_act in ONE kernel launch.
extern "C" __global__ void eval_rmsnorm_q8_quantize(
        const float * __restrict__ input,
        const float * __restrict__ weight,
        float       * __restrict__ norm_out,
        float       * __restrict__ residual,
        block_q8_1  * __restrict__ q8_out,
        const int n) {

    const int tid = threadIdx.x;
    __shared__ float smem[32];

    constexpr int MAX_PER_THREAD = 16;
    float local_vals[MAX_PER_THREAD];
    int count = 0;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = input[i];
        local_vals[count++] = v;
        residual[i] = v;
        ss += v * v;
    }

    ss = warp_reduce_sum(ss);
    int wid = tid / WARP_SIZE, lid = tid % WARP_SIZE;
    if (lid == 0) smem[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        float s = (lid < (int)(blockDim.x / WARP_SIZE)) ? smem[lid] : 0.0f;
        s = warp_reduce_sum(s);
        if (lid == 0) smem[0] = rsqrtf(s / (float)n + NORM_EPS);
    }
    __syncthreads();
    float rstd = smem[0];

    count = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        float w = weight[i];
#if NORM_ADD_ONE
        w = 1.0f + w;
#endif
        float normed = local_vals[count++] * rstd * w;
        norm_out[i] = normed;

        int ib = i / QK8_1;
        int iqs = i % QK8_1;
        float amax = fabsf(normed);
        float sum_val = normed;
        amax = warp_reduce_max<QK8_1>(amax);
        sum_val = warp_reduce_sum<QK8_1>(sum_val);
        float d = amax / 127.0f;
        q8_out[ib].qs[iqs] = (amax == 0.0f) ? 0 : (int8_t)roundf(normed / d);
        if (iqs == 0) q8_out[ib].ds = make_half2(d, sum_val);
    }
}

// Fused post-norm + residual add: output = rmsnorm(input, weight) + residual
// Used for Gemma2/3/4 post-attention-norm and post-FFN-norm paths.
// Eliminates 2 kernel launches per post-norm (rmsnorm + add_residual → 1 launch).
extern "C" __global__ void eval_rmsnorm_add(
        const float * __restrict__ input,
        const float * __restrict__ weight,
        const float * __restrict__ residual,
        float       * __restrict__ output,
        const int n) {
    const int tid = threadIdx.x;
    __shared__ float smem[32];

    constexpr int MAX_PER_THREAD = 16;
    float local_vals[MAX_PER_THREAD];
    int count = 0;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = input[i];
        local_vals[count++] = v;
        ss += v * v;
    }
    ss = warp_reduce_sum(ss);
    int wid = tid / WARP_SIZE, lid = tid % WARP_SIZE;
    if (lid == 0) smem[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        float s = (lid < (int)(blockDim.x / WARP_SIZE)) ? smem[lid] : 0.0f;
        s = warp_reduce_sum(s);
        if (lid == 0) smem[0] = rsqrtf(s / (float)n + NORM_EPS);
    }
    __syncthreads();
    float rstd = smem[0];

    count = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        float w = weight[i];
#if NORM_ADD_ONE
        w = 1.0f + w;
#endif
        output[i] = local_vals[count++] * rstd * w + residual[i];
    }
}

// =============================================================================
// Q8_1 quantization: f32 → Q8_1 (from baseline quantize.cu)
// =============================================================================

extern "C" __global__ void eval_quantize_q8(
        const float * __restrict__ input,
        block_q8_1  * __restrict__ output,
        const int n) {
    int n_blocks = n / QK8_1;
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= n_blocks * QK8_1) return;

    int ib  = flat / QK8_1;
    int iqs = flat % QK8_1;
    float val = input[flat];
    float amax = fabsf(val), sum = val;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    float d = amax / 127.0f;
    output[ib].qs[iqs] = (amax == 0.0f) ? 0 : (int8_t)roundf(val / d);
    if (iqs == 0) output[ib].ds = make_half2(d, sum);
}

// =============================================================================
// Matvec: Q4_K weights × Q8_1 input → f32 output
// From baseline mmvq.cu mul_mat_vec_q
// =============================================================================

// EXACT port from baseline mmvq.cu mul_mat_vec_q.
// Launch with: grid=(out_dim,1,1), block=(32,4,1) — 2D thread block!
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q4k(
        const void       * __restrict__ weight,
        const long long                  weight_stride_bytes,
        const void       * __restrict__ q8_input,
        float            * __restrict__ output,
        const int in_dim,
        const int out_dim) {

    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q4_K));
    int row = blockIdx.x;
    if (row >= out_dim) return;
    mmvq_q4k_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
}

// Matvec + residual
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q4k_residual(
        const void       * __restrict__ weight,
        const long long                  weight_stride_bytes,
        const void       * __restrict__ q8_input,
        const float      * __restrict__ residual,
        float            * __restrict__ output,
        const int in_dim,
        const int out_dim) {

    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q4_K));
    int row = blockIdx.x;
    if (row >= out_dim) return;
    mmvq_q4k_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
    // Add residual — only lane 0 of warp 0 wrote the result
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        output[row] += residual[row];
    }
}

// =============================================================================
// Fused gate+up matvec — uses mmvq_generic_row with gate_weight parameter.
// Matches baseline's has_fusion=true path: computes BOTH up and gate projections
// in one kernel, sharing Q8 input data. Applies GLU activation at output.
// This replaces the separate eval_fused_gate_up_silu/gelu kernels.
// =============================================================================

// Macro to define fused gate+up kernels for each quant type
#define DEFINE_FUSED_MATVEC_GLU(name, block_type, qk, qi, vdr, vec_dot_fn, nwarps) \
extern "C" __global__ void __launch_bounds__(32*nwarps, 1) eval_matvec_glu_##name( \
        const void * __restrict__ up_weight, \
        const long long up_stride_bytes, \
        const void * __restrict__ gate_weight, \
        const long long gate_stride_bytes, \
        const void * __restrict__ q8_input, \
        float * __restrict__ output, \
        const int in_dim, const int out_dim, \
        const int glu_op) { \
    const int stride_row = (int)(up_stride_bytes / sizeof(block_type)); \
    int row = blockIdx.x; \
    if (row >= out_dim) return; \
    mmvq_generic_row<qk, qi, vdr, vec_dot_fn, nwarps>( \
        up_weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row, \
        gate_weight, glu_op); \
}

DEFINE_FUSED_MATVEC_GLU(q4k,  block_q4_K, QK_K, QI4_K, 2, vec_dot_q4_K_q8_1, 8)
DEFINE_FUSED_MATVEC_GLU(q6k,  block_q6_K, QK_K, QI6_K, 1, vec_dot_q6_K_q8_1, 8)
DEFINE_FUSED_MATVEC_GLU(q4_0, block_q4_0, 32, QI4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1, 8)
DEFINE_FUSED_MATVEC_GLU(q8_0, block_q8_0, 32, QI8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1, 8)

#undef DEFINE_FUSED_MATVEC_GLU

// =============================================================================
// P0 matvec variants — all types from baseline mmvq.cu
// Pattern: stride_row = weight_stride_bytes / sizeof(block_TYPE), call mmvq_TYPE_row
// =============================================================================

// --- Q4_0 ---
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q4_0(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ q8_input, float * __restrict__ output,
        const int in_dim, const int out_dim) {
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q4_0));
    int row = blockIdx.x; if (row >= out_dim) return;
    mmvq_q4_0_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
}
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q4_0_residual(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ q8_input, const float * __restrict__ residual,
        float * __restrict__ output, const int in_dim, const int out_dim) {
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q4_0));
    int row = blockIdx.x; if (row >= out_dim) return;
    mmvq_q4_0_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
    if (threadIdx.x == 0 && threadIdx.y == 0) output[row] += residual[row];
}

// --- Q4_1 ---
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q4_1(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ q8_input, float * __restrict__ output,
        const int in_dim, const int out_dim) {
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q4_1));
    int row = blockIdx.x; if (row >= out_dim) return;
    mmvq_q4_1_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
}
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q4_1_residual(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ q8_input, const float * __restrict__ residual,
        float * __restrict__ output, const int in_dim, const int out_dim) {
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q4_1));
    int row = blockIdx.x; if (row >= out_dim) return;
    mmvq_q4_1_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
    if (threadIdx.x == 0 && threadIdx.y == 0) output[row] += residual[row];
}

// --- Q5_0 ---
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q5_0(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ q8_input, float * __restrict__ output,
        const int in_dim, const int out_dim) {
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q5_0));
    int row = blockIdx.x; if (row >= out_dim) return;
    mmvq_q5_0_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
}
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q5_0_residual(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ q8_input, const float * __restrict__ residual,
        float * __restrict__ output, const int in_dim, const int out_dim) {
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q5_0));
    int row = blockIdx.x; if (row >= out_dim) return;
    mmvq_q5_0_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
    if (threadIdx.x == 0 && threadIdx.y == 0) output[row] += residual[row];
}

// --- Q5_1 ---
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q5_1(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ q8_input, float * __restrict__ output,
        const int in_dim, const int out_dim) {
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q5_1));
    int row = blockIdx.x; if (row >= out_dim) return;
    mmvq_q5_1_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
}
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q5_1_residual(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ q8_input, const float * __restrict__ residual,
        float * __restrict__ output, const int in_dim, const int out_dim) {
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q5_1));
    int row = blockIdx.x; if (row >= out_dim) return;
    mmvq_q5_1_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
    if (threadIdx.x == 0 && threadIdx.y == 0) output[row] += residual[row];
}

// --- Q8_0 ---
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q8_0(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ q8_input, float * __restrict__ output,
        const int in_dim, const int out_dim) {
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q8_0));
    int row = blockIdx.x; if (row >= out_dim) return;
    mmvq_q8_0_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
}
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q8_0_residual(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ q8_input, const float * __restrict__ residual,
        float * __restrict__ output, const int in_dim, const int out_dim) {
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q8_0));
    int row = blockIdx.x; if (row >= out_dim) return;
    mmvq_q8_0_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
    if (threadIdx.x == 0 && threadIdx.y == 0) output[row] += residual[row];
}

// --- Q2_K ---
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q2k(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ q8_input, float * __restrict__ output,
        const int in_dim, const int out_dim) {
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q2_K));
    int row = blockIdx.x; if (row >= out_dim) return;
    mmvq_q2k_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
}
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q2k_residual(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ q8_input, const float * __restrict__ residual,
        float * __restrict__ output, const int in_dim, const int out_dim) {
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q2_K));
    int row = blockIdx.x; if (row >= out_dim) return;
    mmvq_q2k_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
    if (threadIdx.x == 0 && threadIdx.y == 0) output[row] += residual[row];
}

// --- Q3_K ---
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q3k(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ q8_input, float * __restrict__ output,
        const int in_dim, const int out_dim) {
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q3_K));
    int row = blockIdx.x; if (row >= out_dim) return;
    mmvq_q3k_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
}
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q3k_residual(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ q8_input, const float * __restrict__ residual,
        float * __restrict__ output, const int in_dim, const int out_dim) {
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q3_K));
    int row = blockIdx.x; if (row >= out_dim) return;
    mmvq_q3k_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
    if (threadIdx.x == 0 && threadIdx.y == 0) output[row] += residual[row];
}

// --- Q5_K ---
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q5k(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ q8_input, float * __restrict__ output,
        const int in_dim, const int out_dim) {
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q5_K));
    int row = blockIdx.x; if (row >= out_dim) return;
    mmvq_q5k_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
}
extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q5k_residual(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ q8_input, const float * __restrict__ residual,
        float * __restrict__ output, const int in_dim, const int out_dim) {
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q5_K));
    int row = blockIdx.x; if (row >= out_dim) return;
    mmvq_q5k_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
    if (threadIdx.x == 0 && threadIdx.y == 0) output[row] += residual[row];
}

// =============================================================================
// IQ + MXFP4 + NVFP4 matvec kernels — using macro to avoid 22x repetition
// =============================================================================

#define EVAL_MATVEC_PAIR(name, block_type, row_fn) \
extern "C" __global__ void eval_matvec_##name( \
        const void * __restrict__ weight, const long long weight_stride_bytes, \
        const void * __restrict__ q8_input, float * __restrict__ output, \
        const int in_dim, const int out_dim) { \
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_type)); \
    int row = blockIdx.x; if (row >= out_dim) return; \
    row_fn(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row); \
} \
extern "C" __global__ void eval_matvec_##name##_residual( \
        const void * __restrict__ weight, const long long weight_stride_bytes, \
        const void * __restrict__ q8_input, const float * __restrict__ residual, \
        float * __restrict__ output, const int in_dim, const int out_dim) { \
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_type)); \
    int row = blockIdx.x; if (row >= out_dim) return; \
    row_fn(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row); \
    if (threadIdx.x == 0 && threadIdx.y == 0) output[row] += residual[row]; \
}

EVAL_MATVEC_PAIR(iq2_xxs, block_iq2_xxs, mmvq_iq2_xxs_row)
EVAL_MATVEC_PAIR(iq2_xs,  block_iq2_xs,  mmvq_iq2_xs_row)
EVAL_MATVEC_PAIR(iq2_s,   block_iq2_s,   mmvq_iq2_s_row)
EVAL_MATVEC_PAIR(iq3_xxs, block_iq3_xxs, mmvq_iq3_xxs_row)
EVAL_MATVEC_PAIR(iq3_s,   block_iq3_s,   mmvq_iq3_s_row)
EVAL_MATVEC_PAIR(iq1_s,   block_iq1_s,   mmvq_iq1_s_row)
EVAL_MATVEC_PAIR(iq1_m,   block_iq1_m,   mmvq_iq1_m_row)
EVAL_MATVEC_PAIR(iq4_nl,  block_iq4_nl,  mmvq_iq4_nl_row)
EVAL_MATVEC_PAIR(iq4_xs,  block_iq4_xs,  mmvq_iq4_xs_row)
EVAL_MATVEC_PAIR(mxfp4,   block_mxfp4,   mmvq_mxfp4_row)
EVAL_MATVEC_PAIR(nvfp4,   block_nvfp4,   mmvq_nvfp4_row)

#undef EVAL_MATVEC_PAIR

// =============================================================================
// Q6_K matvec variants (same pattern, different vec_dot + block size)
// =============================================================================

extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q6k(
        const void       * __restrict__ weight,
        const long long                  weight_stride_bytes,
        const void       * __restrict__ q8_input,
        float            * __restrict__ output,
        const int in_dim,
        const int out_dim) {

    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q6_K));
    int row = blockIdx.x;
    if (row >= out_dim) return;
    mmvq_q6k_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
}

extern "C" __global__ void __launch_bounds__(128, 1) eval_matvec_q6k_residual(
        const void       * __restrict__ weight,
        const long long                  weight_stride_bytes,
        const void       * __restrict__ q8_input,
        const float      * __restrict__ residual,
        float            * __restrict__ output,
        const int in_dim,
        const int out_dim) {

    const int stride_row = (int)(weight_stride_bytes / sizeof(block_q6_K));
    int row = blockIdx.x;
    if (row >= out_dim) return;
    mmvq_q6k_row(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        output[row] += residual[row];
    }
}

// =============================================================================
// 8-warp matvec variants for RDNA3 (gfx1100) — baseline nwarps=8 for ncols_dst=1
// Launch with: grid=(out_dim,1,1), block=(32,8,1) — 256 threads
// Only for types that benefit on RDNA3 (MMVQ_PARAMETERS_RDNA3_0 table)
// =============================================================================

#define EVAL_MATVEC_8W_PAIR(name, block_type, row_fn) \
extern "C" __global__ void __launch_bounds__(256, 1) eval_matvec_##name##_8w( \
        const void * __restrict__ weight, const long long weight_stride_bytes, \
        const void * __restrict__ q8_input, float * __restrict__ output, \
        const int in_dim, const int out_dim) { \
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_type)); \
    int row = blockIdx.x; if (row >= out_dim) return; \
    row_fn(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row); \
} \
extern "C" __global__ void eval_matvec_##name##_8w_residual( \
        const void * __restrict__ weight, const long long weight_stride_bytes, \
        const void * __restrict__ q8_input, const float * __restrict__ residual, \
        float * __restrict__ output, const int in_dim, const int out_dim) { \
    const int stride_row = (int)(weight_stride_bytes / sizeof(block_type)); \
    int row = blockIdx.x; if (row >= out_dim) return; \
    row_fn(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row); \
    if (threadIdx.x == 0 && threadIdx.y == 0) output[row] += residual[row]; \
}

EVAL_MATVEC_8W_PAIR(q4_0,   block_q4_0,   mmvq_q4_0_row_8w)
EVAL_MATVEC_8W_PAIR(q4_1,   block_q4_1,   mmvq_q4_1_row_8w)
EVAL_MATVEC_8W_PAIR(q5_0,   block_q5_0,   mmvq_q5_0_row_8w)
EVAL_MATVEC_8W_PAIR(q5_1,   block_q5_1,   mmvq_q5_1_row_8w)
EVAL_MATVEC_8W_PAIR(q8_0,   block_q8_0,   mmvq_q8_0_row_8w)
EVAL_MATVEC_8W_PAIR(iq4_nl, block_iq4_nl, mmvq_iq4_nl_row_8w)
EVAL_MATVEC_8W_PAIR(q4k,    block_q4_K,   mmvq_q4k_row_8w)
EVAL_MATVEC_8W_PAIR(q6k,    block_q6_K,   mmvq_q6k_row_8w)

#undef EVAL_MATVEC_8W_PAIR

// =============================================================================
// Residual add: output = a + b
// =============================================================================

extern "C" __global__ void eval_add_residual(
        const float * __restrict__ a,
        const float * __restrict__ b,
        float       * __restrict__ output,
        const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = a[i] + b[i];
}

// Element-wise multiply — baseline binbcast.cu op_mul: return a * b
// Used for LoRA/per-tensor scales: if (wo_s) cur *= wo_s
extern "C" __global__ void eval_elementwise_mul(
        const float * __restrict__ a,
        const float * __restrict__ b,
        float       * __restrict__ output,
        const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = a[i] * b[i];
}

// SiLU(gate) * up — element-wise (from prefill.hip prompt_silu_mul)
extern "C" __global__ void eval_silu_mul(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = op_silu(gate[i]) * up[i];
}

// GELU (tanh approx) gate — baseline LLM_FFN_GELU used by Gemma/Phi/Bert
extern "C" __global__ void eval_gelu_mul(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = op_gelu(gate[i]) * up[i];
}

// GELU-erf (exact) gate — for archs using erf variant
extern "C" __global__ void eval_gelu_erf_mul(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = op_gelu_erf(gate[i]) * up[i];
}

// GELU-quick gate
extern "C" __global__ void eval_gelu_quick_mul(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = op_gelu_quick(gate[i]) * up[i];
}

// Standalone GELU (tanh-approx) — ported from baseline ggml-cuda/unary.cu
// BERT sequential FFN: up → gelu(up) → down (no gate)
extern "C" __global__ void eval_gelu(
        const float * __restrict__ input,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = op_gelu(input[i]);
}

// Standalone GELU-erf (exact) variant
extern "C" __global__ void eval_gelu_erf(
        const float * __restrict__ input,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = op_gelu_erf(input[i]);
}

// =============================================================================
// GroupNorm — ported from baseline ggml-cuda/norm.cu::group_norm_f32
// Used by WavTokenizer PosNet layers. One block per group.
// output = (x - mean) / sqrt(var + eps) — bare normalize, no weight/bias
// Weight and bias applied separately if needed.
// =============================================================================
extern "C" __global__ void eval_group_norm(
        const float * __restrict__ x,
        float       * __restrict__ dst,
        const int group_size,
        const int ne_elements,
        const float eps) {
    const int start = blockIdx.x * group_size + threadIdx.x;
    const int end   = min(blockIdx.x * group_size + group_size, ne_elements);
    __shared__ float smem[32];

    // Pass 1: compute mean
    float tmp = 0.0f;
    for (int j = start; j < end; j += blockDim.x) {
        tmp += x[j];
    }
    tmp = warp_reduce_sum(tmp);
    int wid = threadIdx.x / WARP_SIZE, lid = threadIdx.x % WARP_SIZE;
    if (lid == 0) smem[wid] = tmp;
    __syncthreads();
    if (wid == 0) {
        float v = (lid < (int)(blockDim.x / WARP_SIZE)) ? smem[lid] : 0;
        v = warp_reduce_sum(v);
        if (lid == 0) smem[0] = v / (float)group_size; // mean
    }
    __syncthreads();
    const float mean = smem[0];

    // Pass 2: compute variance and center
    tmp = 0.0f;
    for (int j = start; j < end; j += blockDim.x) {
        float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }
    tmp = warp_reduce_sum(tmp);
    if (lid == 0) smem[wid] = tmp;
    __syncthreads();
    if (wid == 0) {
        float v = (lid < (int)(blockDim.x / WARP_SIZE)) ? smem[lid] : 0;
        v = warp_reduce_sum(v);
        if (lid == 0) smem[0] = rsqrtf(v / (float)group_size + eps);
    }
    __syncthreads();
    const float scale = smem[0];

    // Pass 3: normalize
    for (int j = start; j < end; j += blockDim.x) {
        dst[j] *= scale;
    }
}

// =============================================================================
// Im2col 1D — ported from baseline ggml-cuda/im2col.cu::im2col_kernel
// Rearranges input for conv_1d: output is [IC*KW, OW] ready for GEMM with kernel
// Grid: (ceil(IC*KW/256), OW, N), Block: (256, 1, 1)
// =============================================================================
extern "C" __global__ void eval_im2col_1d(
        const float * __restrict__ src,    // [N, IC, IW] input
        float       * __restrict__ dst,    // [N, OW, IC*KW] im2col output
        const int IC,                      // input channels
        const int IW,                      // input width (time dimension)
        const int OW,                      // output width
        const int KW,                      // kernel width
        const int s0,                      // stride
        const int p0,                      // padding
        const int d0) {                    // dilation
    const int ikw_ic = blockIdx.x * blockDim.x + threadIdx.x;
    const int ow = blockIdx.y;
    const int n = blockIdx.z;

    const int IC_KW = IC * KW;
    if (ikw_ic >= IC_KW) return;

    const int ikw = ikw_ic / IC;
    const int ic  = ikw_ic % IC;

    const int iw = ow * s0 + ikw * d0 - p0;

    float val = 0.0f;
    if (iw >= 0 && iw < IW) {
        val = src[n * IC * IW + ic * IW + iw];
    }
    dst[n * OW * IC_KW + ow * IC_KW + ikw_ic] = val;
}

// Swish activation: out = x * sigmoid(x) — used by WavTokenizer PosNet
extern "C" __global__ void eval_swish(
        const float * __restrict__ input,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

// =============================================================================
// Depthwise Conv1d with half-padding — ported from baseline conv2d-dw.cu
// Each channel independently convolved with its own [kernel_size] filter.
// Grid: (ceil(total/256), 1, 1), Block: (256, 1, 1)
// total = channels * OW, OW = IW (same padding)
// =============================================================================
extern "C" __global__ void eval_conv1d_dw_ph(
        const float * __restrict__ input,    // [channels, IW]
        const float * __restrict__ kernel,   // [channels, 1, KW]
        const float * __restrict__ bias,     // [channels] or NULL
        float       * __restrict__ output,   // [channels, OW]
        const int channels,
        const int IW,                        // input width (= time)
        const int KW,                        // kernel width
        const int total) {                   // channels * IW
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int c = idx / IW;      // channel index
    int ow = idx % IW;     // output time position (OW = IW for same padding)
    int p = KW / 2;        // half padding

    float sum = 0.0f;
    for (int kw = 0; kw < KW; kw++) {
        int iw = ow + kw - p;
        if (iw >= 0 && iw < IW) {
            sum += input[c * IW + iw] * kernel[c * KW + kw];
        }
    }
    if (bias) sum += bias[c];
    output[c * IW + ow] = sum;
}

// ReLU² gate — squared ReLU
extern "C" __global__ void eval_relu2_mul(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = op_relu2(gate[i]) * up[i];
}

// Scalar scale: output[i] = input[i] * scale (constant float passed via launch args)
// baseline: ggml_scale — used by Gemma (input*sqrt(H), Q * f_attention_scale)
extern "C" __global__ void eval_scale_scalar(
        const float * __restrict__ input,
        float       * __restrict__ output,
        const float scale,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = input[i] * scale;
}

// Logit soft-cap: tanh(x / cap) * cap — baseline Gemma2 final logit softcap
// cf. src/models/gemma3.cpp:142-146
extern "C" __global__ void eval_softcap(
        float * __restrict__ logits,
        const float cap,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = logits[i];
        logits[i] = tanhf(x / cap) * cap;
    }
}

// =============================================================================
// QK Norm + RoPE + KV cache write (single token, all KV heads)
// From baseline rope.cu rope_norm + norm.cu rms_norm_f32
// =============================================================================

extern "C" __global__ void eval_qk_norm_rope_kv_write(
        float        * __restrict__ q_proj,     // [FA_QPROJ_SIZE] f32
        float        * __restrict__ kv_proj,    // [FA_KV_SIZE*2] f32
        const float  * __restrict__ q_norm_w,   // [FA_HEAD_DIM] or NULL
        const float  * __restrict__ k_norm_w,   // [FA_HEAD_DIM] or NULL
        __half       * __restrict__ k_cache,    // [n_kv_heads, max_seq, head_dim]
        __half       * __restrict__ v_cache,
        const float  * __restrict__ freq_factors, // [FA_ROPE_DIM/2] or NULL — from rope_freqs.weight
        const int position,
        const int max_seq_len,
        const float rope_theta_override,        // 0 = use FA_ROPE_THETA, non-zero = override (for iSWA per-layer theta)
        const int * __restrict__ d_decode_params) { // [3] GPU-resident {token_id, position, kv_len} or nullptr

    // GPU-resident position override for graph reuse — if d_decode_params is non-null,
    // read position from GPU buffer instead of scalar arg (which is baked into the graph)
    const int actual_position = d_decode_params ? d_decode_params[1] : position;

    // One warp per head. Process all Q heads and KV heads.
    int idx = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    int lid = threadIdx.x % WARP_SIZE;

    int total_q  = FA_N_Q_HEADS;
    int total_kv = FA_N_KV_HEADS;
    // Per-layer theta: >0 = override, 0 = use FA_ROPE_THETA, <0 = skip RoPE entirely (Cohere2 non-SWA)
    const float theta_base_val = (rope_theta_override > 0.0f) ? rope_theta_override : FA_ROPE_THETA;
    const bool skip_rope = (rope_theta_override < 0.0f);

    // Q heads: optional RMSNorm + RoPE in-place
    // q_norm_w is null for models without per-head QK norm (e.g. Llama 3.2)
    if (idx < total_q) {
        int head = idx;
        float * qh = q_proj + head * (FA_HAS_GATED_ATTN ? FA_HEAD_DIM * 2 : FA_HEAD_DIM);

        // QK RMSNorm — baseline line 86: ggml_rms_norm (NO weight multiply, just normalize)
        // Condition: q_norm_w non-null signals use_kq_norm is set for this model
        if (q_norm_w) {
            float ss = 0;
            for (int i = lid; i < FA_HEAD_DIM; i += WARP_SIZE) ss += qh[i] * qh[i];
            ss = warp_reduce_sum(ss);
            float sc = rsqrtf(ss / (float)FA_HEAD_DIM + NORM_EPS);
            sc = __shfl_sync(0xffffffff, sc, 0, WARP_SIZE);
            for (int i = lid; i < FA_HEAD_DIM; i += WARP_SIZE) {
                qh[i] = qh[i] * sc;  // bare norm, no weight
            }
        }

        // RoPE — pairing depends on baseline arch's rope_type:
        //   ROPE_NONE (0): no RoPE (ALiBi, GPT-2, StarCoder) — skip rotation
        //   ROPE_NORM (1, Llama): pairs (qh[i], qh[i+1]) for i in [0,2,4,...]
        //   ROPE_NEOX (2, Qwen/StableLM/GPT-NeoX/Phi/Gemma/...): pairs (qh[i], qh[i + n_dims/2])
        // Compile-time #if (ROPE_TYPE baked into .hsaco via model_config_<hash>.h)
        // baseline rope.cu rope_norm vs rope_neox (lines 44-180)
#if ROPE_TYPE == 0 /* ROPE_NONE — no rotation, ALiBi/GPT2/StarCoder */
        // No RoPE: Q values pass through unchanged
#elif ROPE_TYPE == 3 /* ROPE_MULTI (Qwen2VL/Qwen3VL) — per-section position components */
        if (!skip_rope) {
            const int sect_dims = ROPE_SECTION_0 + ROPE_SECTION_1 + ROPE_SECTION_2 + ROPE_SECTION_3;
            for (int i = lid; i < FA_ROPE_DIM / 2; i += WARP_SIZE) {
                // Pick position by section — for text-only, all sections use `actual_position`
                int sector = i % sect_dims;
                int p = actual_position; // text-only: all sections same position
                // For vision-language: p = pos_quad[section_index], but we only have scalar position

                float theta_base = (float)p * powf(theta_base_val, -(float)(i*2) / (float)FA_ROPE_DIM);
                float ff = freq_factors ? freq_factors[i] : 1.0f;
                float cv, sv;
                rope_yarn<true>(theta_base / ff, ROPE_FREQ_SCALE,
                               rope_corr_dims{YARN_CORR_DIM0, YARN_CORR_DIM1},
                               i*2, YARN_EXT_FACTOR, YARN_ATTN_FACTOR, cv, sv);
                // NeoX pairing: (i, i + n_dims/2)
                float x0 = qh[i];
                float x1 = qh[i + FA_ROPE_DIM / 2];
                qh[i]                    = x0 * cv - x1 * sv;
                qh[i + FA_ROPE_DIM / 2]  = x0 * sv + x1 * cv;
            }
        }
#elif ROPE_TYPE == 2 /* ROPE_NEOX */
        if (!skip_rope) {
        for (int i = lid; i < FA_ROPE_DIM / 2; i += WARP_SIZE) {
            float theta_base = (float)actual_position * powf(theta_base_val, -(float)(i*2) / (float)FA_ROPE_DIM);
            float ff = freq_factors ? freq_factors[i] : 1.0f;
            float cv, sv;
            rope_yarn<true>(theta_base / ff, ROPE_FREQ_SCALE,
                           rope_corr_dims{YARN_CORR_DIM0, YARN_CORR_DIM1},
                           i*2, YARN_EXT_FACTOR, YARN_ATTN_FACTOR, cv, sv);
            float x0 = qh[i];
            float x1 = qh[i + FA_ROPE_DIM / 2];
            qh[i]                    = x0 * cv - x1 * sv;
            qh[i + FA_ROPE_DIM / 2]  = x0 * sv + x1 * cv;
        }
        }
#else /* ROPE_NORM (default for Llama-like) */
        if (!skip_rope) {
        for (int i = lid * 2; i < FA_ROPE_DIM; i += WARP_SIZE * 2) {
            float theta_base = (float)actual_position * powf(theta_base_val, -(float)i / (float)FA_ROPE_DIM);
            float ff = freq_factors ? freq_factors[i / 2] : 1.0f;
            float cv, sv;
            rope_yarn<true>(theta_base / ff, ROPE_FREQ_SCALE,
                           rope_corr_dims{YARN_CORR_DIM0, YARN_CORR_DIM1},
                           i / 2, YARN_EXT_FACTOR, YARN_ATTN_FACTOR, cv, sv);
            float x0 = qh[i], x1 = qh[i + 1];
            qh[i]     = x0 * cv - x1 * sv;
            qh[i + 1] = x0 * sv + x1 * cv;
        }
        }
#endif
    }

    // KV heads: RMSNorm + RoPE + cache write
    int kidx = idx - total_q;
    if (idx >= total_q && kidx < total_kv) {
        int head = kidx;
        float * kh = kv_proj + head * FA_HEAD_DIM;
        float * vh = kv_proj + FA_KV_SIZE + head * FA_HEAD_DIM;
        __half * kc = k_cache + head * max_seq_len * FA_HEAD_DIM + actual_position * FA_HEAD_DIM;
        __half * vc = v_cache + head * max_seq_len * FA_HEAD_DIM + actual_position * FA_HEAD_DIM;

        // K QK RMSNorm — baseline line 87: bare ggml_rms_norm, no weight
        if (k_norm_w) {
            float ss = 0;
            for (int i = lid; i < FA_HEAD_DIM; i += WARP_SIZE) ss += kh[i] * kh[i];
            ss = warp_reduce_sum(ss);
            float sc = rsqrtf(ss / (float)FA_HEAD_DIM + NORM_EPS);
            sc = __shfl_sync(0xffffffff, sc, 0, WARP_SIZE);
            for (int i = lid; i < FA_HEAD_DIM; i += WARP_SIZE) {
                kh[i] = kh[i] * sc;  // bare norm, no weight
            }
        }

        // K RoPE — same pairing as Q, gated on ROPE_TYPE
#if ROPE_TYPE == 0 /* ROPE_NONE */
        // No RoPE: K values pass through unchanged
#elif ROPE_TYPE == 3 /* ROPE_MULTI */
        if (!skip_rope) {
            const int sect_dims = ROPE_SECTION_0 + ROPE_SECTION_1 + ROPE_SECTION_2 + ROPE_SECTION_3;
            for (int i = lid; i < FA_ROPE_DIM / 2; i += WARP_SIZE) {
                int p = actual_position;
                float theta_base = (float)p * powf(theta_base_val, -(float)(i*2) / (float)FA_ROPE_DIM);
                float ff = freq_factors ? freq_factors[i] : 1.0f;
                float cv, sv;
                rope_yarn<true>(theta_base / ff, ROPE_FREQ_SCALE,
                               rope_corr_dims{YARN_CORR_DIM0, YARN_CORR_DIM1},
                               i*2, YARN_EXT_FACTOR, YARN_ATTN_FACTOR, cv, sv);
                float x0 = kh[i];
                float x1 = kh[i + FA_ROPE_DIM / 2];
                kh[i]                    = x0 * cv - x1 * sv;
                kh[i + FA_ROPE_DIM / 2]  = x0 * sv + x1 * cv;
            }
        }
#elif ROPE_TYPE == 2 /* ROPE_NEOX */
        if (!skip_rope) {
        for (int i = lid; i < FA_ROPE_DIM / 2; i += WARP_SIZE) {
            float theta_base = (float)actual_position * powf(theta_base_val, -(float)(i*2) / (float)FA_ROPE_DIM);
            float ff = freq_factors ? freq_factors[i] : 1.0f;
            float cv, sv;
            rope_yarn<true>(theta_base / ff, ROPE_FREQ_SCALE,
                           rope_corr_dims{YARN_CORR_DIM0, YARN_CORR_DIM1},
                           i*2, YARN_EXT_FACTOR, YARN_ATTN_FACTOR, cv, sv);
            float x0 = kh[i];
            float x1 = kh[i + FA_ROPE_DIM / 2];
            kh[i]                    = x0 * cv - x1 * sv;
            kh[i + FA_ROPE_DIM / 2]  = x0 * sv + x1 * cv;
        }
        }
#else /* ROPE_NORM */
        if (!skip_rope) {
        for (int i = lid * 2; i < FA_ROPE_DIM; i += WARP_SIZE * 2) {
            float theta_base = (float)actual_position * powf(theta_base_val, -(float)i / (float)FA_ROPE_DIM);
            float ff = freq_factors ? freq_factors[i / 2] : 1.0f;
            float cv, sv;
            rope_yarn<true>(theta_base / ff, ROPE_FREQ_SCALE,
                           rope_corr_dims{YARN_CORR_DIM0, YARN_CORR_DIM1},
                           i / 2, YARN_EXT_FACTOR, YARN_ATTN_FACTOR, cv, sv);
            float x0 = kh[i], x1 = kh[i + 1];
            kh[i]     = x0 * cv - x1 * sv;
            kh[i + 1] = x0 * sv + x1 * cv;
        }
        }
#endif

        for (int i = lid; i < FA_HEAD_DIM; i += WARP_SIZE) {
            kc[i] = __float2half(kh[i]);
            vc[i] = __float2half(vh[i]);
        }
    }
}

// =============================================================================
// Attention decode: single token, all Q heads
// From baseline fattn-vec.cuh (online softmax)
// =============================================================================

extern "C" __global__ void __launch_bounds__(128, 1)
eval_attention_decode(
        const float  * __restrict__ q_proj,     // [FA_QPROJ_SIZE]
        const __half * __restrict__ k_cache,
        const __half * __restrict__ v_cache,
        float        * __restrict__ attn_out,   // [FA_Q_SIZE]
        const int kv_len,
        const int max_seq_len,
        const float alibi_max_bias,             // 0 = no ALiBi; >0 = ALiBi enabled
        const float alibi_m0,
        const float alibi_m1,
        const int   alibi_n_head_log2,
        const int   cur_pos,                    // current token position
        const float * __restrict__ rel_pos_bias, // [n_head, kv_len] T5 relative position bias or nullptr
        const float attn_softcap,               // Gemma2/3/4: tanh(score/cap)*cap (0 = disabled)
        const int * __restrict__ d_decode_params) { // [3] GPU-resident {token_id, position, kv_len} or nullptr

    // GPU-resident overrides for graph reuse
    const int actual_kv_len  = d_decode_params ? d_decode_params[2] : kv_len;
    const int actual_cur_pos = d_decode_params ? d_decode_params[1] : cur_pos;

    // One block per Q head.  Thread block: dim3(32, 4, 1) = 128 threads, 4 warps.
    int qh = blockIdx.x;
    if (qh >= FA_N_Q_HEADS) return;

    int kvh = qh / FA_GQA_RATIO;
    const float * q = q_proj + qh * (FA_HAS_GATED_ATTN ? FA_HEAD_DIM * 2 : FA_HEAD_DIM);
    const __half * kc = k_cache + kvh * max_seq_len * FA_HEAD_DIM;
    const __half * vc = v_cache + kvh * max_seq_len * FA_HEAD_DIM;
    float * out = attn_out + qh * FA_HEAD_DIM;

    // Strides in bytes (matching baseline nb11/nb21 convention)
    const int k_stride_bytes = FA_HEAD_DIM * sizeof(__half);
    const int v_stride_bytes = FA_HEAD_DIM * sizeof(__half);

    // ALiBi: compute per-head slope — baseline fattn-vec.cuh line 106
    float slope = 0.0f;
    if (alibi_max_bias > 0.0f) {
        slope = get_alibi_slope(alibi_max_bias, (unsigned)qh, (unsigned)alibi_n_head_log2, alibi_m0, alibi_m1);
    }

    // T5 relative position bias: point to this head's slice [actual_kv_len]
    const float * head_rel_bias = rel_pos_bias ? (rel_pos_bias + qh * actual_kv_len) : nullptr;

    // Use compile-time attention scale if set, otherwise 1/sqrt(d)
#if defined(HAS_CUSTOM_ATTN_SCALE) && HAS_CUSTOM_ATTN_SCALE
    const float attn_scale = FA_ATTENTION_SCALE;
#else
    const float attn_scale = 1.0f / sqrtf((float)FA_HEAD_DIM);
#endif
    // Match baseline's ne11 = GGML_PAD(kv_used, 256) for the attention loop bound.
    // Per-head cache pointers (kc, vc) already use max_seq_len for stride above.
    // The inner function uses this only for ne11 (loop bound) — nb12/nb22 strides
    // derived from ne11 are declared but never used.
    const int n_kv_padded = ((actual_kv_len + 255) & ~255);
    const int attn_n_kv = (n_kv_padded < max_seq_len) ? n_kv_padded : max_seq_len;
    flash_attn_vec_f16<FA_HEAD_DIM>(q, kc, vc, out, actual_kv_len, attn_n_kv,
        attn_scale, k_stride_bytes, v_stride_bytes,
        slope, actual_cur_pos, head_rel_bias, attn_softcap);

#if FA_HAS_GATED_ATTN
    // Gated attention: need all 128 threads, use flat tid
    const int flat_tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const float * gate = q + FA_HEAD_DIM;
    for (int i = flat_tid; i < FA_HEAD_DIM; i += 128) {
        out[i] *= op_sigmoid(gate[i]);
    }
#endif
}

// =============================================================================
// Attention decode (parallel blocks): splits KV range across gridDim.y blocks.
// For models with few Q heads (e.g. Gemma-2 has 8), single-block attention
// underutilizes the GPU (~8% occupancy on 96 CU). Parallel blocks fix this
// by launching parallel_blocks * n_q_heads total blocks.
//
// Grid: (n_q_heads, parallel_blocks, 1)
// Each block processes kv_len/parallel_blocks positions.
// A separate reduction kernel (eval_attention_combine) merges partial results.
// =============================================================================
extern "C" __global__ void __launch_bounds__(128, 1)
eval_attention_decode_pb(
        const float  * __restrict__ q_proj,
        const __half * __restrict__ k_cache,
        const __half * __restrict__ v_cache,
        float        * __restrict__ attn_out,    // [n_q_heads * D] final output (only used when pb==1)
        float        * __restrict__ dst_partial,  // [n_q_heads * pb * D] partial VKQ per block
        float        * __restrict__ dst_meta,     // [n_q_heads * pb * 2] (KQ_max, KQ_sum) per block
        const int kv_len,
        const int max_seq_len,
        const float alibi_max_bias,
        const float alibi_m0,
        const float alibi_m1,
        const int   alibi_n_head_log2,
        const int   cur_pos,
        const float * __restrict__ rel_pos_bias,
        const float attn_softcap,
        const int * __restrict__ d_decode_params,
        const int parallel_blocks) {

    const int actual_kv_len  = d_decode_params ? d_decode_params[2] : kv_len;
    const int actual_cur_pos = d_decode_params ? d_decode_params[1] : cur_pos;

    int qh = blockIdx.x;
    int pb_idx = blockIdx.y;  // which parallel block we are
    if (qh >= FA_N_Q_HEADS) return;

    int kvh = qh / FA_GQA_RATIO;
    const float * q = q_proj + qh * (FA_HAS_GATED_ATTN ? FA_HEAD_DIM * 2 : FA_HEAD_DIM);
    const __half * kc = k_cache + kvh * max_seq_len * FA_HEAD_DIM;
    const __half * vc = v_cache + kvh * max_seq_len * FA_HEAD_DIM;

    const int k_stride_bytes = FA_HEAD_DIM * sizeof(__half);
    const int v_stride_bytes = FA_HEAD_DIM * sizeof(__half);

    float slope = 0.0f;
    if (alibi_max_bias > 0.0f) {
        slope = get_alibi_slope(alibi_max_bias, (unsigned)qh, (unsigned)alibi_n_head_log2, alibi_m0, alibi_m1);
    }
    const float * head_rel_bias = rel_pos_bias ? (rel_pos_bias + qh * actual_kv_len) : nullptr;

#if defined(HAS_CUSTOM_ATTN_SCALE) && HAS_CUSTOM_ATTN_SCALE
    const float attn_scale = FA_ATTENTION_SCALE;
#else
    const float attn_scale = 1.0f / sqrtf((float)FA_HEAD_DIM);
#endif

    // Match baseline's ne11 = GGML_PAD(kv_used, 256) for the attention loop bound.
    // Per-head cache pointers (kc, vc) already use max_seq_len for stride above.
    const int n_kv_padded = ((actual_kv_len + 255) & ~255);
    const int attn_n_kv = (n_kv_padded < max_seq_len) ? n_kv_padded : max_seq_len;

    // Per-head scratch: partial[pb_idx * D] and meta[pb_idx * 2]
    float * head_partial = dst_partial + qh * parallel_blocks * FA_HEAD_DIM;
    float * head_meta    = dst_meta    + qh * parallel_blocks * 2;

    if (parallel_blocks == 1) {
        // Single block: write directly to attn_out (same as eval_attention_decode)
        float * out = attn_out + qh * FA_HEAD_DIM;
        flash_attn_vec_f16<FA_HEAD_DIM>(q, kc, vc, out, actual_kv_len, attn_n_kv,
            attn_scale, k_stride_bytes, v_stride_bytes,
            slope, actual_cur_pos, head_rel_bias, attn_softcap,
            1, 0, nullptr, nullptr);
    } else {
        // Multi-block: write partial results + metadata
        flash_attn_vec_f16<FA_HEAD_DIM>(q, kc, vc, nullptr, actual_kv_len, attn_n_kv,
            attn_scale, k_stride_bytes, v_stride_bytes,
            slope, actual_cur_pos, head_rel_bias, attn_softcap,
            parallel_blocks, pb_idx, head_partial, head_meta);
    }

#if FA_HAS_GATED_ATTN
    // Gated attention only works with single block (apply gate after combine for multi-block)
    if (parallel_blocks == 1) {
        float * out = attn_out + qh * FA_HEAD_DIM;
        const int flat_tid = threadIdx.y * WARP_SIZE + threadIdx.x;
        const float * gate = q + FA_HEAD_DIM;
        for (int i = flat_tid; i < FA_HEAD_DIM; i += 128) {
            out[i] *= op_sigmoid(gate[i]);
        }
    }
#endif
}

// =============================================================================
// Attention combine: merge partial results from parallel blocks.
// Grid: (n_q_heads, 1, 1), Block: (D, 1, 1) or (256, 1, 1) for D>256.
// Ported from baseline fattn-common.cuh flash_attn_combine_results.
// =============================================================================
extern "C" __global__ void eval_attention_combine(
        const float * __restrict__ dst_partial,  // [n_q_heads * pb * D]
        const float * __restrict__ dst_meta,     // [n_q_heads * pb * 2]
        float       * __restrict__ attn_out,     // [n_q_heads * D]
        const int parallel_blocks) {

    const int qh = blockIdx.x;
    const int tid = threadIdx.x;

    const float * head_partial = dst_partial + qh * parallel_blocks * FA_HEAD_DIM;
    const float * head_meta    = dst_meta    + qh * parallel_blocks * 2;
    float * out = attn_out + qh * FA_HEAD_DIM;

    if (tid >= FA_HEAD_DIM) return;

    // Find global max across all blocks
    float kqmax = head_meta[0];  // block 0's KQ_max
    for (int b = 1; b < parallel_blocks; b++) {
        kqmax = fmaxf(kqmax, head_meta[b * 2]);
    }

    // Weighted sum of partial outputs using online softmax correction
    float numerator = 0.0f;
    float denominator = 0.0f;
    for (int b = 0; b < parallel_blocks; b++) {
        float block_max = head_meta[b * 2];
        float block_sum = head_meta[b * 2 + 1];
        float scale = expf(block_max - kqmax);
        numerator   += scale * head_partial[b * FA_HEAD_DIM + tid];
        denominator += scale * block_sum;
    }

    out[tid] = numerator / denominator;
}

// =============================================================================
// Attention decode (TILE): single token, all Q heads
// From baseline fattn-tile.cuh — tiled KQ/VKQ with shared memory
// Handles arbitrary D (including D=96) without padding-induced divergence.
// Thread block: dim3(32, 1, 1) = 32 threads, 1 warp (nwarps=1 for ncols=1 decode)
// =============================================================================

// Helper: select nbatch_K based on D (from RDNA config table in fattn-tile.cuh)
// These match ggml_cuda_fattn_tile_get_config_amd_rdna for ncols=2 (smallest config)
// Constraints: nbatch_K % 8 == 0, DV % nbatch_K == 0 || (nbatch_K*2/3 divides DV)
// For known head sizes, use tuned values. For unknown sizes, use D itself
// (safe since D is always a multiple of 8 for transformer models).
template <int D> struct tile_nbatch_K              { static constexpr int value = D; };
template <> struct tile_nbatch_K< 40>              { static constexpr int value =  40; };
template <> struct tile_nbatch_K< 64>              { static constexpr int value =  64; };
template <> struct tile_nbatch_K< 72>              { static constexpr int value =  72; };
template <> struct tile_nbatch_K< 80>              { static constexpr int value =  40; };
template <> struct tile_nbatch_K< 96>              { static constexpr int value =  48; };
template <> struct tile_nbatch_K<112>              { static constexpr int value =  56; };
template <> struct tile_nbatch_K<128>              { static constexpr int value =  64; };
template <> struct tile_nbatch_K<256>              { static constexpr int value =  64; };

extern "C" __global__ void __launch_bounds__(32, 4)
eval_attention_decode_tile(
        const float  * __restrict__ q_proj,     // [FA_QPROJ_SIZE]
        const __half * __restrict__ k_cache,
        const __half * __restrict__ v_cache,
        float        * __restrict__ attn_out,   // [FA_Q_SIZE]
        const int kv_len,
        const int max_seq_len,
        const float alibi_max_bias,             // 0 = no ALiBi; >0 = ALiBi enabled
        const float alibi_m0,
        const float alibi_m1,
        const int   alibi_n_head_log2,
        const int   cur_pos,                    // current token position
        const float * __restrict__ rel_pos_bias, // [n_head, kv_len] T5 relative position bias or nullptr
        const float attn_softcap,               // Gemma2/3/4: tanh(score/cap)*cap (0 = disabled)
        const int * __restrict__ d_decode_params) { // [3] GPU-resident {token_id, position, kv_len} or nullptr

    // GPU-resident overrides for graph reuse
    const int actual_kv_len  = d_decode_params ? d_decode_params[2] : kv_len;
    const int actual_cur_pos = d_decode_params ? d_decode_params[1] : cur_pos;

    // One block per Q head.  Thread block: dim3(32, 1, 1) = 32 threads, 1 warp.
    int qh = blockIdx.x;
    if (qh >= FA_N_Q_HEADS) return;

    int kvh = qh / FA_GQA_RATIO;
    const float * q = q_proj + qh * (FA_HAS_GATED_ATTN ? FA_HEAD_DIM * 2 : FA_HEAD_DIM);
    const __half * kc = k_cache + kvh * max_seq_len * FA_HEAD_DIM;
    const __half * vc = v_cache + kvh * max_seq_len * FA_HEAD_DIM;
    float * out = attn_out + qh * FA_HEAD_DIM;

    const int k_stride_bytes = FA_HEAD_DIM * sizeof(__half);
    const int v_stride_bytes = FA_HEAD_DIM * sizeof(__half);

    // ALiBi slope
    float slope = 0.0f;
    if (alibi_max_bias > 0.0f) {
        slope = get_alibi_slope(alibi_max_bias, (unsigned)qh, (unsigned)alibi_n_head_log2, alibi_m0, alibi_m1);
    }

    // T5 relative position bias
    const float * head_rel_bias = rel_pos_bias ? (rel_pos_bias + qh * actual_kv_len) : nullptr;

    // Attention scale
#if defined(HAS_CUSTOM_ATTN_SCALE) && HAS_CUSTOM_ATTN_SCALE
    const float attn_scale = FA_ATTENTION_SCALE;
#else
    const float attn_scale = 1.0f / sqrtf((float)FA_HEAD_DIM);
#endif

    // RDNA decode config: nwarps=1, nbatch_fa=WARP_SIZE=32, nbatch_K from config table
    constexpr int nbatch_fa = WARP_SIZE; // 32
    constexpr int nbatch_K  = tile_nbatch_K<FA_HEAD_DIM>::value;

    flash_attn_tile_f16<FA_HEAD_DIM, nbatch_fa, nbatch_K>(
        q, kc, vc, out, actual_kv_len,
        attn_scale, k_stride_bytes, v_stride_bytes,
        slope, actual_cur_pos, head_rel_bias, attn_softcap);

#if FA_HAS_GATED_ATTN
    // Gated attention: nwarps=1, 32 threads available
    const float * gate = q + FA_HEAD_DIM;
    for (int i = threadIdx.x; i < FA_HEAD_DIM; i += WARP_SIZE) {
        out[i] *= op_sigmoid(gate[i]);
    }
#endif
}

// =============================================================================
// Attention decode (WMMA): single token, all Q heads
// From baseline fattn-wmma-f16.cu — uses rocWMMA 16x16 matrix fragments
// for KQ dot products and VKQ accumulation.
// Thread block: dim3(32, 4, 1) = 128 threads, 4 warps
// Requires D divisible by 16. Uses ncols=16 (padded, only column 0 valid).
// =============================================================================

#ifdef GGML_USE_WMMA_FATTN

extern "C" __global__ void __launch_bounds__(128, 1)
eval_attention_decode_wmma(
        const float  * __restrict__ q_proj,     // [FA_QPROJ_SIZE]
        const __half * __restrict__ k_cache,
        const __half * __restrict__ v_cache,
        float        * __restrict__ attn_out,   // [FA_Q_SIZE]
        const int kv_len,
        const int max_seq_len,
        const float alibi_max_bias,             // 0 = no ALiBi; >0 = ALiBi enabled
        const float alibi_m0,
        const float alibi_m1,
        const int   alibi_n_head_log2,
        const int   cur_pos,                    // current token position
        const float * __restrict__ rel_pos_bias, // [n_head, kv_len] T5 relative position bias or nullptr
        const float attn_softcap,               // Gemma2/3/4: tanh(score/cap)*cap (0 = disabled)
        const int * __restrict__ d_decode_params) { // [3] GPU-resident {token_id, position, kv_len} or nullptr

    // GPU-resident overrides for graph reuse
    const int actual_kv_len  = d_decode_params ? d_decode_params[2] : kv_len;
    const int actual_cur_pos = d_decode_params ? d_decode_params[1] : cur_pos;

    // One block per Q head.  Thread block: dim3(32, 4, 1) = 128 threads, 4 warps.
    int qh = blockIdx.x;
    if (qh >= FA_N_Q_HEADS) return;

    int kvh = qh / FA_GQA_RATIO;
    const float * q = q_proj + qh * (FA_HAS_GATED_ATTN ? FA_HEAD_DIM * 2 : FA_HEAD_DIM);
    const __half * kc = k_cache + kvh * max_seq_len * FA_HEAD_DIM;
    const __half * vc = v_cache + kvh * max_seq_len * FA_HEAD_DIM;
    float * out = attn_out + qh * FA_HEAD_DIM;

    // ALiBi slope
    float slope = 0.0f;
    if (alibi_max_bias > 0.0f) {
        slope = get_alibi_slope(alibi_max_bias, (unsigned)qh, (unsigned)alibi_n_head_log2, alibi_m0, alibi_m1);
    }

    // Attention scale
#if defined(HAS_CUSTOM_ATTN_SCALE) && HAS_CUSTOM_ATTN_SCALE
    const float attn_scale = FA_ATTENTION_SCALE;
#else
    const float attn_scale = 1.0f / sqrtf((float)FA_HEAD_DIM);
#endif

    // Dispatch softcap at compile time
#if defined(HAS_ATTN_SOFTCAP) && HAS_ATTN_SOFTCAP
    flash_attn_wmma_f16<FA_HEAD_DIM, true>(
        q, kc, vc, out, actual_kv_len, attn_scale, slope, actual_cur_pos, ATTN_SOFTCAP_VAL);
#else
    flash_attn_wmma_f16<FA_HEAD_DIM, false>(
        q, kc, vc, out, actual_kv_len, attn_scale, slope, actual_cur_pos, 0.0f);
#endif

#if FA_HAS_GATED_ATTN
    // Gated attention: need all 128 threads, use flat tid
    const int flat_tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const float * gate = q + FA_HEAD_DIM;
    for (int i = flat_tid; i < FA_HEAD_DIM; i += 128) {
        out[i] *= op_sigmoid(gate[i]);
    }
#endif
}

#endif // GGML_USE_WMMA_FATTN

// =============================================================================
// Cross-attention: T5 decoder cross-attention uses eval_attention_decode directly
// with encoder K/V converted from f32→f16 via gemm_f32_to_f16 kernel.
// See forward/t5-dec.cpp for the implementation.
// No separate cross-attention kernel needed — eval_attention_decode handles
// any f16 K/V buffer regardless of whether it's a KV cache or flat encoder output.
// =============================================================================
// eval_cross_attention removed — T5 decoder cross-attention uses eval_attention_decode
// with encoder K/V converted from f32→f16 via gemm_f32_to_f16. See forward/t5-dec.cpp.

// =============================================================================
// F16/BF16/F32 matvec — from baseline mmvf.cu mul_mat_vec_f
// Stripped: T-specific, type_acc=float, ncols_dst=1, block_size=256, no fusion
// Input is F32 activation (NOT Q8_1), output is F32
// Grid: (nrows, 1, 1), Block: (256, 1, 1)
// =============================================================================

// --- F16 matvec --- mmvf.cu lines 155-186 (float accumulation path)
extern "C" __global__ void __launch_bounds__(256, 1) eval_matvec_f16(
        const void       * __restrict__ weight,
        const long long                  weight_stride_bytes,
        const void       * __restrict__ f32_input,   // F32 activation
        float            * __restrict__ output,
        const int in_dim,
        const int out_dim) {

    const int row = blockIdx.x;
    if (row >= out_dim) return;
    constexpr int block_size = 256;
    const int tid = threadIdx.x;

    // x = row of F16 weight, y = F32 activation
    const __half2 * x2 = (const __half2 *)((const char *)weight + (long long)row * weight_stride_bytes);
    const float2 * y2 = (const float2 *)f32_input;
    const int ncols2 = in_dim / 2;

    float sumf = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 tmpx = __half22float2(x2[col2]);
        const float2 tmpy = y2[col2];
        sumf += tmpx.x * tmpy.x + tmpx.y * tmpy.y;
    }

    // Warp reduce + inter-warp reduce (from baseline mmvf.cu lines 298-330)
    sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    __shared__ float buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) buf_iw[tid] = 0.0f;
    __syncthreads();
    buf_iw[tid / WARP_SIZE] = sumf;
    __syncthreads();
    if (tid < WARP_SIZE) {
        sumf = buf_iw[tid];
        sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    }
    if (tid == 0) output[row] = sumf;
}

extern "C" __global__ void __launch_bounds__(256, 1) eval_matvec_f16_residual(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ f32_input, const float * __restrict__ residual,
        float * __restrict__ output, const int in_dim, const int out_dim) {
    const int row = blockIdx.x;
    if (row >= out_dim) return;
    constexpr int block_size = 256;
    const int tid = threadIdx.x;
    const __half2 * x2 = (const __half2 *)((const char *)weight + (long long)row * weight_stride_bytes);
    const float2 * y2 = (const float2 *)f32_input;
    const int ncols2 = in_dim / 2;
    float sumf = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 tmpx = __half22float2(x2[col2]);
        const float2 tmpy = y2[col2];
        sumf += tmpx.x * tmpy.x + tmpx.y * tmpy.y;
    }
    sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    __shared__ float buf_iw_r[WARP_SIZE];
    if (tid < WARP_SIZE) buf_iw_r[tid] = 0.0f;
    __syncthreads();
    buf_iw_r[tid / WARP_SIZE] = sumf;
    __syncthreads();
    if (tid < WARP_SIZE) {
        sumf = buf_iw_r[tid];
        sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    }
    if (tid == 0) output[row] = sumf + residual[row];
}

// --- BF16 matvec --- mmvf.cu lines 230-296 (HIP path: manual cast)
extern "C" __global__ void __launch_bounds__(256, 1) eval_matvec_bf16(
        const void       * __restrict__ weight,
        const long long                  weight_stride_bytes,
        const void       * __restrict__ f32_input,
        float            * __restrict__ output,
        const int in_dim,
        const int out_dim) {

    const int row = blockIdx.x;
    if (row >= out_dim) return;
    constexpr int block_size = 256;
    const int tid = threadIdx.x;

    // BF16 on HIP: read as int (2x bf16 packed), manually convert to float
    const int * x2 = (const int *)((const char *)weight + (long long)row * weight_stride_bytes);
    const float2 * y2 = (const float2 *)f32_input;
    const int ncols2 = in_dim / 2;

    float sumf = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const int tmpx = x2[col2];
        const float2 tmpy = y2[col2];
        // BF16→F32: shift left 16 bits (BF16 is top 16 bits of F32)
        const float tmpx0 = __uint_as_float(((uint32_t)(uint16_t)tmpx) << 16);
        const float tmpx1 = __uint_as_float(((uint32_t)(uint16_t)(tmpx >> 16)) << 16);
        sumf += tmpx0 * tmpy.x + tmpx1 * tmpy.y;
    }

    sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    __shared__ float buf_iw_b[WARP_SIZE];
    if (tid < WARP_SIZE) buf_iw_b[tid] = 0.0f;
    __syncthreads();
    buf_iw_b[tid / WARP_SIZE] = sumf;
    __syncthreads();
    if (tid < WARP_SIZE) {
        sumf = buf_iw_b[tid];
        sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    }
    if (tid == 0) output[row] = sumf;
}

extern "C" __global__ void __launch_bounds__(256, 1) eval_matvec_bf16_residual(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ f32_input, const float * __restrict__ residual,
        float * __restrict__ output, const int in_dim, const int out_dim) {
    const int row = blockIdx.x;
    if (row >= out_dim) return;
    constexpr int block_size = 256;
    const int tid = threadIdx.x;
    const int * x2 = (const int *)((const char *)weight + (long long)row * weight_stride_bytes);
    const float2 * y2 = (const float2 *)f32_input;
    const int ncols2 = in_dim / 2;
    float sumf = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const int tmpx = x2[col2];
        const float2 tmpy = y2[col2];
        const float tmpx0 = __uint_as_float(((uint32_t)(uint16_t)tmpx) << 16);
        const float tmpx1 = __uint_as_float(((uint32_t)(uint16_t)(tmpx >> 16)) << 16);
        sumf += tmpx0 * tmpy.x + tmpx1 * tmpy.y;
    }
    sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    __shared__ float buf_iw_br[WARP_SIZE];
    if (tid < WARP_SIZE) buf_iw_br[tid] = 0.0f;
    __syncthreads();
    buf_iw_br[tid / WARP_SIZE] = sumf;
    __syncthreads();
    if (tid < WARP_SIZE) {
        sumf = buf_iw_br[tid];
        sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    }
    if (tid == 0) output[row] = sumf + residual[row];
}

// --- F32 matvec --- mmvf.cu lines 117-153 (F32 path)
extern "C" __global__ void __launch_bounds__(256, 1) eval_matvec_f32(
        const void       * __restrict__ weight,
        const long long                  weight_stride_bytes,
        const void       * __restrict__ f32_input,
        float            * __restrict__ output,
        const int in_dim,
        const int out_dim) {

    const int row = blockIdx.x;
    if (row >= out_dim) return;
    constexpr int block_size = 256;
    const int tid = threadIdx.x;

    const float2 * x2 = (const float2 *)((const char *)weight + (long long)row * weight_stride_bytes);
    const float2 * y2 = (const float2 *)f32_input;
    const int ncols2 = in_dim / 2;

    float sumf = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 tmpx = x2[col2];
        const float2 tmpy = y2[col2];
        sumf += tmpx.x * tmpy.x + tmpx.y * tmpy.y;
    }

    sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    __shared__ float buf_iw_f[WARP_SIZE];
    if (tid < WARP_SIZE) buf_iw_f[tid] = 0.0f;
    __syncthreads();
    buf_iw_f[tid / WARP_SIZE] = sumf;
    __syncthreads();
    if (tid < WARP_SIZE) {
        sumf = buf_iw_f[tid];
        sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    }
    if (tid == 0) output[row] = sumf;
}

extern "C" __global__ void __launch_bounds__(256, 1) eval_matvec_f32_residual(
        const void * __restrict__ weight, const long long weight_stride_bytes,
        const void * __restrict__ f32_input, const float * __restrict__ residual,
        float * __restrict__ output, const int in_dim, const int out_dim) {
    const int row = blockIdx.x;
    if (row >= out_dim) return;
    constexpr int block_size = 256;
    const int tid = threadIdx.x;
    const float2 * x2 = (const float2 *)((const char *)weight + (long long)row * weight_stride_bytes);
    const float2 * y2 = (const float2 *)f32_input;
    const int ncols2 = in_dim / 2;
    float sumf = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 tmpx = x2[col2];
        const float2 tmpy = y2[col2];
        sumf += tmpx.x * tmpy.x + tmpx.y * tmpy.y;
    }
    sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    __shared__ float buf_iw_fr[WARP_SIZE];
    if (tid < WARP_SIZE) buf_iw_fr[tid] = 0.0f;
    __syncthreads();
    buf_iw_fr[tid / WARP_SIZE] = sumf;
    __syncthreads();
    if (tid < WARP_SIZE) {
        sumf = buf_iw_fr[tid];
        sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    }
    if (tid == 0) output[row] = sumf + residual[row];
}

// =============================================================================
// Final norm (from baseline norm.cu)
// =============================================================================

extern "C" __global__ void eval_final_norm(
        const float * __restrict__ input,
        const float * __restrict__ weight,
        float       * __restrict__ output,
        const int n) {

    const int tid = threadIdx.x;
    __shared__ float smem[32];

    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) ss += input[i] * input[i];

    ss = warp_reduce_sum(ss);
    int wid = tid / WARP_SIZE, lid = tid % WARP_SIZE;
    if (lid == 0) smem[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        float s = (lid < (int)(blockDim.x / WARP_SIZE)) ? smem[lid] : 0.0f;
        s = warp_reduce_sum(s);
        if (lid == 0) smem[0] = rsqrtf(s / (float)n + NORM_EPS);
    }
    __syncthreads();
    float rstd = smem[0];

    for (int i = tid; i < n; i += blockDim.x) {
        float w = weight[i];
#if NORM_ADD_ONE
        w = 1.0f + w;
#endif
        output[i] = input[i] * rstd * w;
    }
}

// =============================================================================
// LM head uses generic eval_matvec_* kernels via pick_matvec dispatch.
// Dedicated eval_lm_head / eval_lm_head_q6k removed — they had swapped param order.

// =============================================================================
// DeltaNet kernels
// =============================================================================

// Conv1d + SiLU for one DeltaNet head (from baseline ssm-conv.cu)
// One block per head. conv_buf = [conv_ch * conv_kernel] ring buffer.
extern "C" __global__ void eval_dn_conv1d_silu(
        const float * __restrict__ qkv_proj,     // [total_conv_ch] f32
        const float * __restrict__ conv_weight,   // [total_conv_ch * conv_kernel] f32
        float       * __restrict__ conv_buf,      // [total_conv_ch * conv_kernel] per-layer ring buffer
        float       * __restrict__ q_out,         // [key_dim] output
        float       * __restrict__ k_out,         // [key_dim] output
        float       * __restrict__ v_out,         // [value_dim] output
        const int num_v_heads,
        const int num_k_heads,
        const int key_dim,
        const int value_dim,
        const int conv_kernel,
        const int total_conv_ch) {

    const int h = blockIdx.x;
    if (h >= num_v_heads) return;
    const int kh = h % num_k_heads;
    const bool is_first = (h < num_k_heads);

    // Channel offsets: [Q: n_k_heads*key_dim | K: n_k_heads*key_dim | V: n_heads*value_dim]
    const int q_off = kh * key_dim;
    const int k_off = num_k_heads * key_dim + kh * key_dim;
    const int v_off = 2 * num_k_heads * key_dim + h * value_dim;

    // V channels
    for (int c = threadIdx.x; c < value_dim; c += blockDim.x) {
        int ch = v_off + c;
        int buf = ch * conv_kernel;
        for (int t = 0; t < conv_kernel - 1; t++) conv_buf[buf + t] = conv_buf[buf + t + 1];
        conv_buf[buf + conv_kernel - 1] = qkv_proj[ch];
        float co = 0;
        for (int t = 0; t < conv_kernel; t++) co += conv_buf[buf + t] * conv_weight[ch * conv_kernel + t];
        v_out[h * value_dim + c] = op_silu(co);
    }

    // Q channels
    for (int c = threadIdx.x; c < key_dim; c += blockDim.x) {
        int ch = q_off + c;
        int buf = ch * conv_kernel;
        if (is_first) {
            for (int t = 0; t < conv_kernel - 1; t++) conv_buf[buf + t] = conv_buf[buf + t + 1];
            conv_buf[buf + conv_kernel - 1] = qkv_proj[ch];
        }
        float co = 0;
        for (int t = 0; t < conv_kernel; t++) co += conv_buf[buf + t] * conv_weight[ch * conv_kernel + t];
        q_out[h * key_dim + c] = op_silu(co);
    }

    // K channels
    for (int c = threadIdx.x; c < key_dim; c += blockDim.x) {
        int ch = k_off + c;
        int buf = ch * conv_kernel;
        if (is_first) {
            for (int t = 0; t < conv_kernel - 1; t++) conv_buf[buf + t] = conv_buf[buf + t + 1];
            conv_buf[buf + conv_kernel - 1] = qkv_proj[ch];
        }
        float co = 0;
        for (int t = 0; t < conv_kernel; t++) co += conv_buf[buf + t] * conv_weight[ch * conv_kernel + t];
        k_out[h * key_dim + c] = op_silu(co);
    }
}

// L2 normalize Q and K (per-head, from baseline norm.cu l2_norm)
extern "C" __global__ void eval_dn_l2_norm(
        float * __restrict__ data,     // [n_heads * dim] in-place normalize
        const int n_heads,
        const int dim,
        const float eps) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    float * head = data + h * dim;
    __shared__ float smem[32];

    float sq = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) sq += head[i] * head[i];
    sq = warp_reduce_sum(sq);
    int wid = threadIdx.x / WARP_SIZE, lid = threadIdx.x % WARP_SIZE;
    if (lid == 0) smem[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        float s = (lid < (int)(blockDim.x / WARP_SIZE)) ? smem[lid] : 0;
        s = warp_reduce_sum(s);
        if (lid == 0) smem[0] = rsqrtf(fmaxf(s, eps * eps));
    }
    __syncthreads();
    float scale = smem[0];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) head[i] *= scale;
}

// DeltaNet recurrence + gated norm (one block per head, from baseline gated_delta_net.cu GDA path)
extern "C" __global__ void eval_dn_recurrence(
        const float * __restrict__ q,          // [n_heads * key_dim]
        const float * __restrict__ k,          // [n_heads * key_dim]
        const float * __restrict__ v,          // [n_heads * value_dim]
        const float * __restrict__ beta_proj,  // [n_heads]
        const float * __restrict__ alpha_proj, // [n_heads]
        const float * __restrict__ ssm_a,      // [n_heads] f32 weight
        const float * __restrict__ dt_bias,    // [n_heads] f32 weight
        const float * __restrict__ z_proj,     // [n_heads * value_dim]
        const float * __restrict__ norm_w,     // [value_dim]
        float       * __restrict__ state,      // [n_heads * key_dim * value_dim]
        float       * __restrict__ output,     // [n_heads * value_dim]
        const int n_heads,
        const int key_dim,
        const int value_dim,
        const float norm_eps) {

    int h = blockIdx.x;
    if (h >= n_heads) return;

    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE, lid = tid % WARP_SIZE;
    const int NUM_WARPS_DN = blockDim.x / WARP_SIZE;

    // Beta = sigmoid(beta_proj), gate = exp(softplus(alpha + dt_bias) * ssm_a)
    __shared__ float s_beta, s_gate;
    if (tid == 0) {
        s_beta = op_sigmoid(beta_proj[h]);
        float a_val = ssm_a[h];
        float dt_b = dt_bias[h];
        float x = alpha_proj[h] + dt_b;
        float sp = op_softplus(x);
        s_gate = expf(sp * a_val);
    }
    __syncthreads();

    float beta = s_beta, g_val = s_gate;
    float scale = rsqrtf((float)key_dim);  // baseline: 1/sqrt(S_k), NOT 1/sqrt(S_v)
    float * head_state = state + h * key_dim * value_dim;
    float * out_head = output + h * value_dim;
    const float * qh = q + h * key_dim;
    const float * kh = k + h * key_dim;
    const float * vh = v + h * value_dim;

    // State update + output (from gated_delta_net_cuda GDA path)
    const int j_per_warp = value_dim / NUM_WARPS_DN;
    const int i_per_lane = key_dim / WARP_SIZE;

    for (int jj = 0; jj < j_per_warp; jj++) {
        int j = wid * j_per_warp + jj;
        if (j >= value_dim) break;

        float kv_shard = 0;
        for (int ii = 0; ii < i_per_lane; ii++) {
            int i = lid + ii * WARP_SIZE;
            if (i < key_dim) kv_shard += head_state[j * key_dim + i] * kh[i];
        }
        float kv = warp_reduce_sum(kv_shard);

        float delta = (vh[j] - g_val * kv) * beta;

        float attn_shard = 0;
        for (int ii = 0; ii < i_per_lane; ii++) {
            int i = lid + ii * WARP_SIZE;
            if (i < key_dim) {
                head_state[j * key_dim + i] = g_val * head_state[j * key_dim + i] + kh[i] * delta;
                attn_shard += head_state[j * key_dim + i] * qh[i];
            }
        }
        float attn = warp_reduce_sum(attn_shard);
        if (lid == 0) out_head[j] = attn * scale;
    }
    __syncthreads();

    // Gated RMSNorm: norm(output) * silu(z)
    __shared__ float s_gnorm[32];
    float sq = 0;
    for (int i = tid; i < value_dim; i += blockDim.x) sq += out_head[i] * out_head[i];
    sq = warp_reduce_sum(sq);
    if (lid == 0) s_gnorm[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        float v2 = (lid < NUM_WARPS_DN) ? s_gnorm[lid] : 0;
        v2 = warp_reduce_sum(v2);
        if (lid == 0) s_gnorm[0] = rsqrtf(v2 / (float)value_dim + norm_eps);
    }
    __syncthreads();
    float rstd = s_gnorm[0];
    for (int i = tid; i < value_dim; i += blockDim.x) {
        float normed = out_head[i] * rstd * norm_w[i];
        float zv = z_proj[h * value_dim + i];
        out_head[i] = normed * op_silu(zv);
    }
}

// =============================================================================
// MMQ Q8_1 batch quantization: f32 → block_q8_1_mmq
// From baseline quantize.cu lines 175-271, dispatched per ds_layout.
// Launch with: grid=(ne1, block_num_y, ne2*ne3), block=(CUDA_QUANTIZE_BLOCK_SIZE_MMQ,1,1)
//   where block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ)
// =============================================================================

// D4 layout: 1 float32 scale per 32 values, no partial sums.
// Used by: Q5_0, Q8_0, Q3_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, MXFP4, NVFP4
extern "C" __global__ void eval_quantize_mmq_q8_1_d4(
        const float   * __restrict__ x,
        const int32_t * __restrict__ ids,
        void          * __restrict__ vy,
        const int64_t ne00,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t ne0,
        const int     ne1,
        const int     ne2) {
    quantize_mmq_q8_1_impl<MMQ_Q8_1_DS_LAYOUT_D4>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}

// DS4 layout: 1 half2(scale,sum) per 32 values.
// Used by: Q4_0, Q4_1, Q5_1, Q4_K, Q5_K, IQ1_S
extern "C" __global__ void eval_quantize_mmq_q8_1_ds4(
        const float   * __restrict__ x,
        const int32_t * __restrict__ ids,
        void          * __restrict__ vy,
        const int64_t ne00,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t ne0,
        const int     ne1,
        const int     ne2) {
    quantize_mmq_q8_1_impl<MMQ_Q8_1_DS_LAYOUT_DS4>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}

// D2S6 layout: 1 half scale per 64 values + 1 half sum per 16 values (first 96).
// Used by: Q2_K
extern "C" __global__ void eval_quantize_mmq_q8_1_d2s6(
        const float   * __restrict__ x,
        const int32_t * __restrict__ ids,
        void          * __restrict__ vy,
        const int64_t ne00,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t ne0,
        const int     ne1,
        const int     ne2) {
    quantize_mmq_q8_1_impl<MMQ_Q8_1_DS_LAYOUT_D2S6>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}


// =============================================================================
// GPU Sampling Kernels — logits → token_id entirely on device
// Ported from baseline src/llama-sampling.cpp sampling pipeline
// =============================================================================

// Temperature scaling: logits[i] /= temperature (in-place)
extern "C" __global__ void eval_sample_temperature(
        float * __restrict__ logits, const int V, const float inv_temp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) logits[i] *= inv_temp;
}

// Repetition penalty: for tokens in penalty_list, scale logits
// Baseline: logit > 0 ? logit / penalty : logit * penalty
extern "C" __global__ void eval_sample_rep_penalty(
        float * __restrict__ logits,
        const int * __restrict__ penalty_tokens, const int n_penalty,
        const float penalty) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_penalty) return;
    int tok = penalty_tokens[i];
    float l = logits[tok];
    logits[tok] = (l > 0.0f) ? l / penalty : l * penalty;
}

// Argmax (greedy) — single-block reduction, returns index of max logit
extern "C" __global__ void eval_sample_argmax(
        const float * __restrict__ logits, int * __restrict__ out_token, const int V) {
    int tid = threadIdx.x;
    __shared__ float s_val[256];
    __shared__ int   s_idx[256];

    float best = -1e30f;
    int   bidx = 0;
    for (int i = tid; i < V; i += blockDim.x) {
        if (logits[i] > best) { best = logits[i]; bidx = i; }
    }
    s_val[tid] = best;
    s_idx[tid] = bidx;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_val[tid + s] > s_val[tid]) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) *out_token = s_idx[0];
}

// Top-K + Softmax + Top-P + Categorical sampling — fused single-block kernel
// Uses the existing argsort for top-K, then does softmax + cumsum + sample in one pass.
// For simplicity, this kernel handles V up to 256K with a single block of 256 threads.
// The sorted indices and probs are computed in shared memory.
extern "C" __global__ void eval_sample_top_k_p(
        float * __restrict__ logits,       // [V] — modified in-place (temp-scaled, penalty-applied)
        int   * __restrict__ out_token,    // [1] output token ID
        const int V,
        const int top_k,      // 0 = use all V
        const float top_p,    // 1.0 = disabled
        const float random_val) {
    // Single-block implementation: thread 0 does sequential scan over V
    // This is not performance-optimal but is correct and avoids complex parallel prefix.
    if (threadIdx.x != 0) return;

    int k = (top_k > 0 && top_k < V) ? top_k : V;

    // Find top-K by partial selection (O(V*K) but K is small, typically 40-100)
    // For production: use radix select. For correctness: simple scan.
    float max_val = -1e30f;
    for (int i = 0; i < V; i++) if (logits[i] > max_val) max_val = logits[i];

    // Softmax numerator: exp(logit - max)
    float sum_exp = 0.0f;
    for (int i = 0; i < V; i++) {
        float e = expf(logits[i] - max_val);
        logits[i] = e; // temporarily store exp values
        sum_exp += e;
    }

    // Normalize to probabilities
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < V; i++) logits[i] *= inv_sum;

    // Sort by probability (descending) using simple selection for top-K
    // Then scan cumulative sum until top_p threshold
    float cum = 0.0f;
    float r = random_val;
    int selected = 0;

    // Greedy scan: find tokens in decreasing probability order
    // Mark selected tokens by setting their prob to -1 after processing
    for (int step = 0; step < k; step++) {
        // Find current max
        float best_p = -1.0f;
        int best_i = 0;
        for (int i = 0; i < V; i++) {
            if (logits[i] > best_p) { best_p = logits[i]; best_i = i; }
        }
        if (best_p <= 0.0f) break;

        cum += best_p;

        // Renormalize: at this point we have accumulated cum probability
        // Check if random falls within [cum - best_p, cum)
        if (r < cum) {
            selected = best_i;
            break;
        }

        logits[best_i] = -1.0f; // mark as used
        selected = best_i; // fallback if we exhaust K

        // Top-P cutoff
        if (cum >= top_p) break;
    }

    *out_token = selected;
}

// =============================================================================
// Multimodal preprocessing kernels
// =============================================================================

// Image patch extraction: resize + normalize + extract patches + linear projection
// Baseline vision.cpp: bilinear resize → normalize(mean,std) → split into patches → proj
// Grid: (n_patches_x * n_patches_y), block: (patch_size * patch_size * 3 / threads_per_block)
extern "C" __global__ void eval_image_patches(
        const float * __restrict__ image,  // [H, W, 3] normalized f32 RGB
        float       * __restrict__ patches, // [n_patches, patch_dim] output
        const int H, const int W,
        const int patch_size,              // typically 14 or 16
        const int n_patches_x, const int n_patches_y) {
    int patch_idx = blockIdx.x;
    int px = patch_idx % n_patches_x;
    int py = patch_idx / n_patches_x;
    int tid = threadIdx.x;

    int patch_dim = patch_size * patch_size * 3;
    float * out = patches + (int64_t)patch_idx * patch_dim;

    // Extract patch pixels: image[py*ps+j, px*ps+i, c]
    for (int i = tid; i < patch_dim; i += blockDim.x) {
        int c = i / (patch_size * patch_size);
        int rem = i % (patch_size * patch_size);
        int pj = rem / patch_size;
        int pi = rem % patch_size;
        int img_y = py * patch_size + pj;
        int img_x = px * patch_size + pi;
        if (img_y < H && img_x < W) {
            out[i] = image[(img_y * W + img_x) * 3 + c];
        } else {
            out[i] = 0.0f; // zero-pad if image doesn't divide evenly
        }
    }
}

// =============================================================================
// Mel spectrogram: brute-force DFT + mel filterbank on GPU
// No hipfft dependency. N=400-512 is small enough for O(N^2) DFT.
// Ported from baseline whisper.cpp log_mel_spectrogram
// =============================================================================

// Phase 1: STFT power spectrum via brute-force DFT
// Grid: (n_frames, 1, 1), Block: (256, 1, 1)
// Each block processes one frame. Threads split frequency bins.
// Hann window computed inline: w[n] = 0.5 * (1 - cos(2*pi*n/(N-1)))
// Output: power[frame * (n_fft/2+1) + k] = |X[k]|^2
extern "C" __global__ void eval_stft_power(
        const float * __restrict__ audio,       // [n_samples] input audio
        float       * __restrict__ power,       // [n_frames, n_fft/2+1] output power spectrum
        const int n_samples,
        const int n_fft,
        const int hop_length) {
    const int frame = blockIdx.x;
    const int n_fft_half = n_fft / 2 + 1;
    const int frame_start = frame * hop_length;
    const float two_pi = 6.283185307179586f;
    const float inv_n_minus_1 = 1.0f / (float)(n_fft - 1);

    for (int k = threadIdx.x; k < n_fft_half; k += blockDim.x) {
        float re = 0.0f, im = 0.0f;
        float angle_k = two_pi * (float)k / (float)n_fft;

        for (int n = 0; n < n_fft; n++) {
            int sample_idx = frame_start + n;
            float x = (sample_idx < n_samples) ? audio[sample_idx] : 0.0f;
            // Hann window
            float w = 0.5f * (1.0f - cosf(two_pi * (float)n * inv_n_minus_1));
            float xw = x * w;
            float phase = angle_k * (float)n;
            re += xw * cosf(phase);
            im -= xw * sinf(phase);
        }

        power[frame * n_fft_half + k] = re * re + im * im;
    }
}

// Phase 2: Apply mel filterbank to power spectrum
// Grid: (n_frames, n_mels, 1), Block: (256, 1, 1)
// mel_out[mel * n_frames + frame] = sum_k(filter[mel * n_fft_half + k] * power[frame * n_fft_half + k])
extern "C" __global__ void eval_mel_filterbank(
        const float * __restrict__ power,        // [n_frames, n_fft/2+1]
        const float * __restrict__ mel_filters,  // [n_mels, n_fft/2+1] precomputed filterbank (on GPU)
        float       * __restrict__ mel_out,      // [n_mels, n_frames] output
        const int n_frames,
        const int n_fft_half,
        const int n_mels) {
    const int frame = blockIdx.x;
    const int mel = blockIdx.y;

    // Each thread sums over a range of frequency bins
    float sum = 0.0f;
    for (int k = threadIdx.x; k < n_fft_half; k += blockDim.x) {
        sum += mel_filters[mel * n_fft_half + k] * power[frame * n_fft_half + k];
    }

    // Warp reduction
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    // Inter-warp reduction via shared memory
    __shared__ float sdata[8]; // up to 8 warps (256 threads)
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) sdata[warp_id] = sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        for (int w = 0; w < n_warps; w++) total += sdata[w];
        // Clamp and log — baseline: log(max(val, 1e-10))
        total = logf(fmaxf(total, 1e-10f));
        mel_out[mel * n_frames + frame] = total;
    }
}

// Phase 3: Normalize mel spectrogram — baseline: clamp to max-8, shift by (max+4)/4
// Grid: ((n+255)/256, 1, 1), Block: (256, 1, 1)
// Must be called AFTER a separate max-reduction pass writes max_val to mel_max[0].
extern "C" __global__ void eval_mel_normalize(
        float       * __restrict__ mel,         // [n_mels * n_frames] in/out
        const float * __restrict__ mel_max,     // [1] max value (from reduction)
        const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float mmax = mel_max[0];
    float val = mel[i];
    val = fmaxf(val, mmax - 8.0f);
    mel[i] = (val + 4.0f) / 4.0f;
}

// =============================================================================
// Fused MoE matvec — reads expert index from GPU-resident sorted indices
// No hipMemcpy D2H needed. One kernel launch per selected expert per projection.
// =============================================================================

// Fused MoE weighted accumulate — reads routing weight from GPU-resident probs array.
// dst[i] += probs[sorted_ids[expert_rank]] * src[i]
// Grid: ((H+255)/256, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void eval_moe_weighted_add(
        float       * __restrict__ dst,          // [H] accumulator
        const float * __restrict__ src,          // [H] expert FFN output
        const float * __restrict__ probs,        // [n_expert] routing probabilities (GPU, already normalized)
        const int   * __restrict__ sorted_ids,   // [n_expert] sorted expert indices (GPU)
        const int expert_rank,                   // which top-K slot (0, 1, ...)
        const int H) {
    __shared__ float s_weight;
    if (threadIdx.x == 0) {
        int eid = sorted_ids[expert_rank];
        s_weight = probs[eid];
    }
    __syncthreads();
    float w = s_weight;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) {
        dst[i] += w * src[i];
    }
}

// Qwen2MoE shared expert gate: dst[i] *= sigmoid_val
// Applies a scalar gate (from GPU memory) element-wise to a vector.
// sigmoid_val is at gate_ptr[0], already passed through sigmoid on GPU.
// Grid: ((H+255)/256, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void eval_moe_gate_mul(
        float       * __restrict__ dst,          // [H] vector to scale
        const float * __restrict__ gate_ptr,     // [1] scalar gate value on GPU
        const int H) {
    __shared__ float s_gate;
    if (threadIdx.x == 0) s_gate = gate_ptr[0];
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) {
        dst[i] *= s_gate;
    }
}

// Chameleon image token suppression — sets logits[start..start+count-1] = -FLT_MAX on GPU.
// No host-side std::vector, no hipMemcpy. Grid: ((count+255)/256), Block: (256)
extern "C" __global__ void eval_chameleon_suppress(
        float * __restrict__ logits,
        const int start,
        const int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        logits[start + i] = -3.402823466e+38f; // -FLT_MAX
    }
}

// T5 relative position bias — compute on GPU, no D2H.
// For each head h and key position j: bias[h * kv_len + j] = table[h * n_bkts + bucket(j - cur_pos)]
// Grid: (n_heads, 1, 1), Block: (256, 1, 1)
// Ported from baseline: llama_relative_position_bucket (decoder: bidirectional=false)
extern "C" __global__ void eval_t5_rel_bias_compute(
        const float * __restrict__ bias_table,   // [n_heads, n_bkts] on GPU
        float       * __restrict__ bias_out,     // [n_heads, kv_len] output
        const int n_bkts,
        const int kv_len,
        const int cur_pos) {
    const int head = blockIdx.x;

    for (int j = threadIdx.x; j < kv_len; j += blockDim.x) {
        int rel = j - cur_pos; // relative position (negative for past tokens)

        // Baseline: llama_relative_position_bucket (bidirectional=false)
        // Unidirectional: only negative relative positions
        int abs_rel = (rel > 0) ? 0 : -rel; // clamp positive to 0

        int bucket;
        int half_bkts = n_bkts / 2; // max_exact
        if (abs_rel < half_bkts) {
            bucket = abs_rel;
        } else {
            // Logarithmic bucketing: bucket = half_bkts + log(abs_rel/half_bkts) / log(max_dist/half_bkts) * (half_bkts-1)
            // max_distance = half_bkts * 2 = n_bkts (for decoder)
            float max_dist = (float)(half_bkts * 2);
            float log_ratio = logf((float)abs_rel / (float)half_bkts)
                            / logf(max_dist / (float)half_bkts);
            bucket = half_bkts + (int)(log_ratio * (float)(half_bkts - 1));
            if (bucket >= n_bkts) bucket = n_bkts - 1;
        }

        bias_out[head * kv_len + j] = bias_table[head * n_bkts + bucket];
    }
}

// =============================================================================
// Fused MoE matvec for ALL quantized types — reads expert ID from GPU memory
// Grid: (out_dim, 1, 1), Block: (32, 4, 1) for all quantized types
// No hipMemcpy D2H needed.
// =============================================================================

#define DEFINE_MOE_MATVEC(name, block_type, mmvq_row_fn) \
extern "C" __global__ void eval_moe_matvec_##name( \
        const void  * __restrict__ weight_base, \
        const long long expert_stride, \
        const long long row_stride_bytes, \
        const void  * __restrict__ q8_input, \
        float       * __restrict__ output, \
        const int   * __restrict__ sorted_ids, \
        const int expert_rank, \
        const int in_dim, \
        const int out_dim) { \
    __shared__ int s_eid; \
    if (threadIdx.x == 0 && threadIdx.y == 0) s_eid = sorted_ids[expert_rank]; \
    __syncthreads(); \
    const void * weight = (const char *)weight_base + (long long)s_eid * expert_stride; \
    const int stride_row = (int)(row_stride_bytes / sizeof(block_type)); \
    int row = blockIdx.x; \
    if (row >= out_dim) return; \
    mmvq_row_fn(weight, q8_input, output, (uint32_t)in_dim, (uint32_t)stride_row, row); \
}

// K-quant types (QK=256)
DEFINE_MOE_MATVEC(q4k, block_q4_K, mmvq_q4k_row)
DEFINE_MOE_MATVEC(q6k, block_q6_K, mmvq_q6k_row)
DEFINE_MOE_MATVEC(q5k, block_q5_K, mmvq_q5k_row)
DEFINE_MOE_MATVEC(q3k, block_q3_K, mmvq_q3k_row)
DEFINE_MOE_MATVEC(q2k, block_q2_K, mmvq_q2k_row)

// Small-block types (QK=32)
DEFINE_MOE_MATVEC(q4_0, block_q4_0, mmvq_q4_0_row)
DEFINE_MOE_MATVEC(q4_1, block_q4_1, mmvq_q4_1_row)
DEFINE_MOE_MATVEC(q5_0, block_q5_0, mmvq_q5_0_row)
DEFINE_MOE_MATVEC(q5_1, block_q5_1, mmvq_q5_1_row)
DEFINE_MOE_MATVEC(q8_0, block_q8_0, mmvq_q8_0_row)

// IQ types (QK=256)
DEFINE_MOE_MATVEC(iq2_xxs, block_iq2_xxs, mmvq_iq2_xxs_row)
DEFINE_MOE_MATVEC(iq2_xs,  block_iq2_xs,  mmvq_iq2_xs_row)
DEFINE_MOE_MATVEC(iq2_s,   block_iq2_s,   mmvq_iq2_s_row)
DEFINE_MOE_MATVEC(iq3_xxs, block_iq3_xxs, mmvq_iq3_xxs_row)
DEFINE_MOE_MATVEC(iq3_s,   block_iq3_s,   mmvq_iq3_s_row)
DEFINE_MOE_MATVEC(iq1_s,   block_iq1_s,   mmvq_iq1_s_row)
DEFINE_MOE_MATVEC(iq1_m,   block_iq1_m,   mmvq_iq1_m_row)
DEFINE_MOE_MATVEC(iq4_nl,  block_iq4_nl,  mmvq_iq4_nl_row)
DEFINE_MOE_MATVEC(iq4_xs,  block_iq4_xs,  mmvq_iq4_xs_row)

// MXFP4 (QK=32)
DEFINE_MOE_MATVEC(mxfp4, block_mxfp4, mmvq_mxfp4_row)

// NVFP4 (QK=64) — uses same macro since mmvq_nvfp4_row handles the different QK internally
DEFINE_MOE_MATVEC(nvfp4, block_nvfp4, mmvq_nvfp4_row)

#undef DEFINE_MOE_MATVEC

// =============================================================================
// Fused gate+up+silu matvec — replaces 3 kernel launches with 1
// Each block handles 1 output row, computing BOTH gate and up dot products
// against the SAME q8_act input (loaded into shared memory once).
// Then applies silu(gate) * up and writes the result.
// Grid: (out_dim, 1, 1), Block: (32, 4, 1) = 128 threads
// Shared memory: n_q8_blocks * sizeof(block_q8_1) for q8 input caching
// =============================================================================

#define DEFINE_FUSED_GATE_UP_SILU(name, block_type, QK_T, QI_T, VDR_T, vec_dot_fn) \
extern "C" __global__ void eval_fused_gate_up_silu_##name( \
        const void  * __restrict__ gate_weight, \
        const long long gate_stride_bytes, \
        const void  * __restrict__ up_weight, \
        const long long up_stride_bytes, \
        const void  * __restrict__ q8_input, \
        float       * __restrict__ output, \
        const int in_dim, \
        const int out_dim) { \
    /* Shared memory: q8_act copy for fast repeated access */ \
    extern __shared__ float smem[]; \
    block_q8_1 * sq8 = (block_q8_1 *)smem; \
    const int n_q8_bytes = (in_dim / QK8_1) * (int)sizeof(block_q8_1); \
    /* Cooperatively load q8_input into shared memory (as int32 words) */ \
    const int n_words = n_q8_bytes / 4; \
    for (int i = (int)(threadIdx.y * WARP_SIZE + threadIdx.x); i < n_words; \
         i += (int)(blockDim.x * blockDim.y)) { \
        ((int *)sq8)[i] = ((const int *)q8_input)[i]; \
    } \
    __syncthreads(); \
    \
    int row = blockIdx.x; \
    if (row >= out_dim) return; \
    \
    /* Accumulation loop — same structure as mmvq_generic_row */ \
    constexpr int qk  = QK_T; \
    constexpr int qi  = QI_T; \
    constexpr int vdr = VDR_T; \
    constexpr int nwarps = 4; \
    constexpr int warp_size = WARP_SIZE; \
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi; \
    const int tid = warp_size * threadIdx.y + threadIdx.x; \
    const int blocks_per_row_x = in_dim / qk; \
    const int gate_stride_row = (int)(gate_stride_bytes / sizeof(block_type)); \
    const int up_stride_row   = (int)(up_stride_bytes   / sizeof(block_type)); \
    const int kbx_gate = row * gate_stride_row; \
    const int kbx_up   = row * up_stride_row; \
    \
    float gate_sum = 0.0f; \
    float up_sum   = 0.0f; \
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) { \
        const int kby = kbx * (qk / QK8_1); \
        const int kqs = vdr * (tid % (qi/vdr)); \
        gate_sum += vec_dot_fn(gate_weight, sq8 + kby, kbx_gate + kbx, kqs); \
        up_sum   += vec_dot_fn(up_weight,   sq8 + kby, kbx_up   + kbx, kqs); \
    } \
    \
    /* Inter-warp reduction for both sums via shared memory */ \
    /* Re-use smem after q8 load is no longer needed — but q8 data is still */ \
    /* in smem and referenced above, so use a separate region after it. */ \
    /* Instead, use __shared__ arrays with fixed size. */ \
    __shared__ float tmp_gate[3][warp_size]; \
    __shared__ float tmp_up[3][warp_size]; \
    if (threadIdx.y > 0) { \
        tmp_gate[threadIdx.y - 1][threadIdx.x] = gate_sum; \
        tmp_up[threadIdx.y - 1][threadIdx.x]   = up_sum; \
    } \
    __syncthreads(); \
    if (threadIdx.y > 0) return; \
    for (int l = 0; l < nwarps - 1; ++l) { \
        gate_sum += tmp_gate[l][threadIdx.x]; \
        up_sum   += tmp_up[l][threadIdx.x]; \
    } \
    gate_sum = warp_reduce_sum<warp_size>(gate_sum); \
    up_sum   = warp_reduce_sum<warp_size>(up_sum); \
    \
    if (threadIdx.x == 0) { \
        float silu_gate = gate_sum / (1.0f + expf(-gate_sum)); \
        output[row] = silu_gate * up_sum; \
    } \
}

// GELU-tanh variant: same structure, different activation
// gelu_tanh(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
#define DEFINE_FUSED_GATE_UP_GELU(name, block_type, QK_T, QI_T, VDR_T, vec_dot_fn) \
extern "C" __global__ void eval_fused_gate_up_gelu_##name( \
        const void  * __restrict__ gate_weight, \
        const long long gate_stride_bytes, \
        const void  * __restrict__ up_weight, \
        const long long up_stride_bytes, \
        const void  * __restrict__ q8_input, \
        float       * __restrict__ output, \
        const int in_dim, \
        const int out_dim) { \
    extern __shared__ float smem[]; \
    block_q8_1 * sq8 = (block_q8_1 *)smem; \
    const int n_q8_bytes = (in_dim / QK8_1) * (int)sizeof(block_q8_1); \
    const int n_words = n_q8_bytes / 4; \
    for (int i = (int)(threadIdx.y * WARP_SIZE + threadIdx.x); i < n_words; \
         i += (int)(blockDim.x * blockDim.y)) { \
        ((int *)sq8)[i] = ((const int *)q8_input)[i]; \
    } \
    __syncthreads(); \
    int row = blockIdx.x; \
    if (row >= out_dim) return; \
    constexpr int qk  = QK_T; \
    constexpr int qi  = QI_T; \
    constexpr int vdr = VDR_T; \
    constexpr int nwarps = 4; \
    constexpr int warp_size = WARP_SIZE; \
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi; \
    const int tid = warp_size * threadIdx.y + threadIdx.x; \
    const int blocks_per_row_x = in_dim / qk; \
    const int gate_stride_row = (int)(gate_stride_bytes / sizeof(block_type)); \
    const int up_stride_row   = (int)(up_stride_bytes   / sizeof(block_type)); \
    const int kbx_gate = row * gate_stride_row; \
    const int kbx_up   = row * up_stride_row; \
    float gate_sum = 0.0f; \
    float up_sum   = 0.0f; \
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) { \
        const int kby = kbx * (qk / QK8_1); \
        const int kqs = vdr * (tid % (qi/vdr)); \
        gate_sum += vec_dot_fn(gate_weight, sq8 + kby, kbx_gate + kbx, kqs); \
        up_sum   += vec_dot_fn(up_weight,   sq8 + kby, kbx_up   + kbx, kqs); \
    } \
    __shared__ float tmp_gate[3][warp_size]; \
    __shared__ float tmp_up[3][warp_size]; \
    if (threadIdx.y > 0) { \
        tmp_gate[threadIdx.y - 1][threadIdx.x] = gate_sum; \
        tmp_up[threadIdx.y - 1][threadIdx.x]   = up_sum; \
    } \
    __syncthreads(); \
    if (threadIdx.y > 0) return; \
    for (int l = 0; l < nwarps - 1; ++l) { \
        gate_sum += tmp_gate[l][threadIdx.x]; \
        up_sum   += tmp_up[l][threadIdx.x]; \
    } \
    gate_sum = warp_reduce_sum<warp_size>(gate_sum); \
    up_sum   = warp_reduce_sum<warp_size>(up_sum); \
    if (threadIdx.x == 0) { \
        /* GELU-tanh: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))) */ \
        const float GELU_COEF = 0.044715f; \
        const float SQRT_2_PI = 0.7978845608028654f; \
        float gelu_gate = 0.5f * gate_sum * (1.0f + tanhf(SQRT_2_PI * (gate_sum + GELU_COEF * gate_sum * gate_sum * gate_sum))); \
        output[row] = gelu_gate * up_sum; \
    } \
}

// K-quant types (QK=256) — VDR values from baseline vecdotq.cuh
#ifndef VDR_Q4_K_Q8_1_MMVQ
#define VDR_Q4_K_Q8_1_MMVQ 2
#endif
#ifndef VDR_Q6_K_Q8_1_MMVQ
#define VDR_Q6_K_Q8_1_MMVQ 1
#endif
DEFINE_FUSED_GATE_UP_SILU(q4k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1)
DEFINE_FUSED_GATE_UP_SILU(q6k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1)
DEFINE_FUSED_GATE_UP_SILU(q5k, block_q5_K, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1)
DEFINE_FUSED_GATE_UP_SILU(q3k, block_q3_K, QK_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1)
DEFINE_FUSED_GATE_UP_SILU(q2k, block_q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)

// Small-block types (QK=32)
DEFINE_FUSED_GATE_UP_SILU(q4_0, block_q4_0, 32, QI4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1)
DEFINE_FUSED_GATE_UP_SILU(q4_1, block_q4_1, 32, QI4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1)
DEFINE_FUSED_GATE_UP_SILU(q5_0, block_q5_0, 32, QI5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1)
DEFINE_FUSED_GATE_UP_SILU(q5_1, block_q5_1, 32, QI5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1)
DEFINE_FUSED_GATE_UP_SILU(q8_0, block_q8_0, 32, QI8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1)

// IQ types (QK=256)
DEFINE_FUSED_GATE_UP_SILU(iq2_xxs, block_iq2_xxs, QK_K, QI2_XXS, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1)
DEFINE_FUSED_GATE_UP_SILU(iq2_xs,  block_iq2_xs,  QK_K, QI2_XS,  VDR_IQ2_XS_Q8_1_MMVQ,  vec_dot_iq2_xs_q8_1)
DEFINE_FUSED_GATE_UP_SILU(iq2_s,   block_iq2_s,   QK_K, QI2_S,   VDR_IQ2_S_Q8_1_MMVQ,   vec_dot_iq2_s_q8_1)
DEFINE_FUSED_GATE_UP_SILU(iq3_xxs, block_iq3_xxs, QK_K, QI3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1)
DEFINE_FUSED_GATE_UP_SILU(iq3_s,   block_iq3_s,   QK_K, QI3_S,   VDR_IQ3_S_Q8_1_MMVQ,   vec_dot_iq3_s_q8_1)
DEFINE_FUSED_GATE_UP_SILU(iq1_s,   block_iq1_s,   QK_K, QI1_S,   VDR_IQ1_S_Q8_1_MMVQ,   vec_dot_iq1_s_q8_1)
DEFINE_FUSED_GATE_UP_SILU(iq1_m,   block_iq1_m,   QK_K, QI1_M,   VDR_IQ1_M_Q8_1_MMVQ,   vec_dot_iq1_m_q8_1)
DEFINE_FUSED_GATE_UP_SILU(iq4_nl,  block_iq4_nl,  32,   QI4_NL,  VDR_IQ4_NL_Q8_1_MMVQ,  vec_dot_iq4_nl_q8_1)
DEFINE_FUSED_GATE_UP_SILU(iq4_xs,  block_iq4_xs,  QK_K, QI4_XS,  VDR_IQ4_XS_Q8_1_MMVQ,  vec_dot_iq4_xs_q8_1)

// MXFP4 (QK=32)
DEFINE_FUSED_GATE_UP_SILU(mxfp4, block_mxfp4, 32, QI_MXFP4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1)

// NVFP4 (QK=64) — wrap vec_dot to match signature
DEFINE_FUSED_GATE_UP_SILU(nvfp4, block_nvfp4, QK_NVFP4, QI_NVFP4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1_wrap)

#undef DEFINE_FUSED_GATE_UP_SILU

// GELU-tanh variants (Gemma-2/3/4) — same types as SiLU
DEFINE_FUSED_GATE_UP_GELU(q4k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1)
DEFINE_FUSED_GATE_UP_GELU(q6k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1)
DEFINE_FUSED_GATE_UP_GELU(q5k, block_q5_K, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1)
DEFINE_FUSED_GATE_UP_GELU(q3k, block_q3_K, QK_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1)
DEFINE_FUSED_GATE_UP_GELU(q2k, block_q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)
DEFINE_FUSED_GATE_UP_GELU(q4_0, block_q4_0, 32, QI4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1)
DEFINE_FUSED_GATE_UP_GELU(q8_0, block_q8_0, 32, QI8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1)

#undef DEFINE_FUSED_GATE_UP_GELU

// =============================================================================
// Fusion 1: Fused QKV projection — 3 matvec launches → 1
// Each block handles 1 output row across Q, K, or V projection.
// All blocks share the same q8_act input via shared memory.
// Grid: (q_out_dim + kv_out_dim + kv_out_dim, 1, 1)
// Block: (32, 4, 1) = 128 threads (same as regular matvec)
// blockIdx.x < q_out_dim → Q projection
// blockIdx.x < q_out_dim + kv_out_dim → K projection
// else → V projection
// =============================================================================

#define DEFINE_FUSED_QKV_MATVEC(name, block_type, QK_T, QI_T, VDR_T, vec_dot_fn) \
extern "C" __global__ void eval_fused_qkv_matvec_##name( \
        const void  * __restrict__ q_weight, \
        const long long q_stride, \
        const void  * __restrict__ k_weight, \
        const long long k_stride, \
        const void  * __restrict__ v_weight, \
        const long long v_stride, \
        const void  * __restrict__ q8_input, \
        float       * __restrict__ q_output, \
        float       * __restrict__ k_output, \
        float       * __restrict__ v_output, \
        const int in_dim, \
        const int q_out_dim, \
        const int kv_out_dim) { \
    /* Shared memory: q8_act copy for fast repeated access */ \
    extern __shared__ float smem[]; \
    block_q8_1 * sq8 = (block_q8_1 *)smem; \
    const int n_q8_bytes = (in_dim / QK8_1) * (int)sizeof(block_q8_1); \
    const int n_words = n_q8_bytes / 4; \
    for (int i = (int)(threadIdx.y * WARP_SIZE + threadIdx.x); i < n_words; \
         i += (int)(blockDim.x * blockDim.y)) { \
        ((int *)sq8)[i] = ((const int *)q8_input)[i]; \
    } \
    __syncthreads(); \
    \
    /* Determine which projection and row */ \
    int bid = blockIdx.x; \
    const void * weight; \
    long long stride_bytes; \
    float * output; \
    int row, out_dim; \
    if (bid < q_out_dim) { \
        weight = q_weight; stride_bytes = q_stride; \
        output = q_output; row = bid; out_dim = q_out_dim; \
    } else if (bid < q_out_dim + kv_out_dim) { \
        weight = k_weight; stride_bytes = k_stride; \
        output = k_output; row = bid - q_out_dim; out_dim = kv_out_dim; \
    } else { \
        weight = v_weight; stride_bytes = v_stride; \
        output = v_output; row = bid - q_out_dim - kv_out_dim; out_dim = kv_out_dim; \
    } \
    if (row >= out_dim) return; \
    \
    /* Accumulation loop — same structure as fused gate+up */ \
    constexpr int qk  = QK_T; \
    constexpr int qi  = QI_T; \
    constexpr int vdr = VDR_T; \
    constexpr int nwarps = 4; \
    constexpr int warp_size = WARP_SIZE; \
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi; \
    const int tid = warp_size * threadIdx.y + threadIdx.x; \
    const int blocks_per_row_x = in_dim / qk; \
    const int stride_row = (int)(stride_bytes / sizeof(block_type)); \
    const int kbx_base = row * stride_row; \
    \
    float dot_sum = 0.0f; \
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) { \
        const int kby = kbx * (qk / QK8_1); \
        const int kqs = vdr * (tid % (qi/vdr)); \
        dot_sum += vec_dot_fn(weight, sq8 + kby, kbx_base + kbx, kqs); \
    } \
    \
    /* Inter-warp reduction */ \
    __shared__ float tmp_reduce[3][warp_size]; \
    if (threadIdx.y > 0) { \
        tmp_reduce[threadIdx.y - 1][threadIdx.x] = dot_sum; \
    } \
    __syncthreads(); \
    if (threadIdx.y > 0) return; \
    for (int l = 0; l < nwarps - 1; ++l) { \
        dot_sum += tmp_reduce[l][threadIdx.x]; \
    } \
    dot_sum = warp_reduce_sum<warp_size>(dot_sum); \
    \
    if (threadIdx.x == 0) { \
        output[row] = dot_sum; \
    } \
}

// K-quant types
DEFINE_FUSED_QKV_MATVEC(q4k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1)
DEFINE_FUSED_QKV_MATVEC(q6k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1)
DEFINE_FUSED_QKV_MATVEC(q5k, block_q5_K, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1)
DEFINE_FUSED_QKV_MATVEC(q3k, block_q3_K, QK_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1)
DEFINE_FUSED_QKV_MATVEC(q2k, block_q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)
// Small-block types
DEFINE_FUSED_QKV_MATVEC(q4_0, block_q4_0, 32, QI4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1)
DEFINE_FUSED_QKV_MATVEC(q4_1, block_q4_1, 32, QI4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1)
DEFINE_FUSED_QKV_MATVEC(q5_0, block_q5_0, 32, QI5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1)
DEFINE_FUSED_QKV_MATVEC(q5_1, block_q5_1, 32, QI5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1)
DEFINE_FUSED_QKV_MATVEC(q8_0, block_q8_0, 32, QI8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1)

#undef DEFINE_FUSED_QKV_MATVEC

// =============================================================================
// Fusion 2: Fused quantize + matvec — eliminates q8_act global memory round-trip
// Phase 1: All threads cooperatively quantize f32 input to Q8_1 in shared memory
// Phase 2: Each block computes its matvec row dot product against shared Q8_1
// Grid: (out_dim, 1, 1), Block: (32, 4, 1) = 128 threads
// =============================================================================

#define DEFINE_FUSED_QUANTIZE_MATVEC(name, block_type, QK_T, QI_T, VDR_T, vec_dot_fn) \
extern "C" __global__ void eval_quantize_matvec_##name( \
        const float * __restrict__ f32_input, \
        const void  * __restrict__ weight, \
        const long long weight_stride, \
        float       * __restrict__ output, \
        const int in_dim, \
        const int out_dim) { \
    /* Shared memory: Q8_1 quantized input */ \
    extern __shared__ float smem[]; \
    block_q8_1 * sq8 = (block_q8_1 *)smem; \
    const int n_q8_blocks = in_dim / QK8_1; \
    const int n_threads = blockDim.x * blockDim.y; \
    const int flat_tid = threadIdx.y * blockDim.x + threadIdx.x; \
    \
    /* Phase 1: Quantize f32 → Q8_1 in shared memory */ \
    /* Each Q8_1 block = 32 consecutive elements, needs warp-level max+sum */ \
    /* With 128 threads and QK8_1=32, each warp handles one Q8_1 block at a time */ \
    for (int ib = flat_tid / QK8_1; ib < n_q8_blocks; ib += n_threads / QK8_1) { \
        int iqs = flat_tid % QK8_1; \
        int idx = ib * QK8_1 + iqs; \
        float val = (idx < in_dim) ? f32_input[idx] : 0.0f; \
        float amax = fabsf(val); \
        float sum_val = val; \
        amax = warp_reduce_max<QK8_1>(amax); \
        sum_val = warp_reduce_sum<QK8_1>(sum_val); \
        float d = amax / 127.0f; \
        sq8[ib].qs[iqs] = (amax == 0.0f) ? 0 : (int8_t)roundf(val / d); \
        if (iqs == 0) sq8[ib].ds = make_half2(d, sum_val); \
    } \
    __syncthreads(); \
    \
    /* Phase 2: Matvec dot product — same as regular matvec */ \
    int row = blockIdx.x; \
    if (row >= out_dim) return; \
    \
    constexpr int qk  = QK_T; \
    constexpr int qi  = QI_T; \
    constexpr int vdr = VDR_T; \
    constexpr int nwarps = 4; \
    constexpr int warp_size = WARP_SIZE; \
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi; \
    const int tid = warp_size * threadIdx.y + threadIdx.x; \
    const int blocks_per_row_x = in_dim / qk; \
    const int stride_row = (int)(weight_stride / sizeof(block_type)); \
    const int kbx_base = row * stride_row; \
    \
    float dot_sum = 0.0f; \
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) { \
        const int kby = kbx * (qk / QK8_1); \
        const int kqs = vdr * (tid % (qi/vdr)); \
        dot_sum += vec_dot_fn(weight, sq8 + kby, kbx_base + kbx, kqs); \
    } \
    \
    __shared__ float tmp_reduce[3][warp_size]; \
    if (threadIdx.y > 0) { \
        tmp_reduce[threadIdx.y - 1][threadIdx.x] = dot_sum; \
    } \
    __syncthreads(); \
    if (threadIdx.y > 0) return; \
    for (int l = 0; l < nwarps - 1; ++l) { \
        dot_sum += tmp_reduce[l][threadIdx.x]; \
    } \
    dot_sum = warp_reduce_sum<warp_size>(dot_sum); \
    \
    if (threadIdx.x == 0) { \
        output[row] = dot_sum; \
    } \
} \
\
/* Residual-adding variant: output = matvec(quantize(input)) + residual */ \
extern "C" __global__ void eval_quantize_matvec_residual_##name( \
        const float * __restrict__ f32_input, \
        const void  * __restrict__ weight, \
        const long long weight_stride, \
        const float * __restrict__ residual, \
        float       * __restrict__ output, \
        const int in_dim, \
        const int out_dim) { \
    extern __shared__ float smem[]; \
    block_q8_1 * sq8 = (block_q8_1 *)smem; \
    const int n_q8_blocks = in_dim / QK8_1; \
    const int n_threads = blockDim.x * blockDim.y; \
    const int flat_tid = threadIdx.y * blockDim.x + threadIdx.x; \
    \
    for (int ib = flat_tid / QK8_1; ib < n_q8_blocks; ib += n_threads / QK8_1) { \
        int iqs = flat_tid % QK8_1; \
        int idx = ib * QK8_1 + iqs; \
        float val = (idx < in_dim) ? f32_input[idx] : 0.0f; \
        float amax = fabsf(val); \
        float sum_val = val; \
        amax = warp_reduce_max<QK8_1>(amax); \
        sum_val = warp_reduce_sum<QK8_1>(sum_val); \
        float d = amax / 127.0f; \
        sq8[ib].qs[iqs] = (amax == 0.0f) ? 0 : (int8_t)roundf(val / d); \
        if (iqs == 0) sq8[ib].ds = make_half2(d, sum_val); \
    } \
    __syncthreads(); \
    \
    int row = blockIdx.x; \
    if (row >= out_dim) return; \
    \
    constexpr int qk  = QK_T; \
    constexpr int qi  = QI_T; \
    constexpr int vdr = VDR_T; \
    constexpr int nwarps = 4; \
    constexpr int warp_size = WARP_SIZE; \
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi; \
    const int tid = warp_size * threadIdx.y + threadIdx.x; \
    const int blocks_per_row_x = in_dim / qk; \
    const int stride_row = (int)(weight_stride / sizeof(block_type)); \
    const int kbx_base = row * stride_row; \
    \
    float dot_sum = 0.0f; \
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) { \
        const int kby = kbx * (qk / QK8_1); \
        const int kqs = vdr * (tid % (qi/vdr)); \
        dot_sum += vec_dot_fn(weight, sq8 + kby, kbx_base + kbx, kqs); \
    } \
    \
    __shared__ float tmp_reduce[3][warp_size]; \
    if (threadIdx.y > 0) { \
        tmp_reduce[threadIdx.y - 1][threadIdx.x] = dot_sum; \
    } \
    __syncthreads(); \
    if (threadIdx.y > 0) return; \
    for (int l = 0; l < nwarps - 1; ++l) { \
        dot_sum += tmp_reduce[l][threadIdx.x]; \
    } \
    dot_sum = warp_reduce_sum<warp_size>(dot_sum); \
    \
    if (threadIdx.x == 0) { \
        output[row] = dot_sum + residual[row]; \
    } \
}

// K-quant types
DEFINE_FUSED_QUANTIZE_MATVEC(q4k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1)
DEFINE_FUSED_QUANTIZE_MATVEC(q6k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1)
DEFINE_FUSED_QUANTIZE_MATVEC(q5k, block_q5_K, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1)
DEFINE_FUSED_QUANTIZE_MATVEC(q3k, block_q3_K, QK_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1)
DEFINE_FUSED_QUANTIZE_MATVEC(q2k, block_q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)
// Small-block types
DEFINE_FUSED_QUANTIZE_MATVEC(q4_0, block_q4_0, 32, QI4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1)
DEFINE_FUSED_QUANTIZE_MATVEC(q4_1, block_q4_1, 32, QI4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1)
DEFINE_FUSED_QUANTIZE_MATVEC(q5_0, block_q5_0, 32, QI5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1)
DEFINE_FUSED_QUANTIZE_MATVEC(q5_1, block_q5_1, 32, QI5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1)
DEFINE_FUSED_QUANTIZE_MATVEC(q8_0, block_q8_0, 32, QI8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1)

#undef DEFINE_FUSED_QUANTIZE_MATVEC

// =============================================================================
// Fusion 3: Fused consecutive norms — post-attn rmsnorm+add + pre-FFN rmsnorm+q8
// Eliminates 1 kernel launch + 2 global memory round-trips per layer.
// Used when BOTH post-attn-norm AND pre-FFN-norm exist (Gemma-2/3/4).
// Grid: (1, 1, 1), Block: (norm_threads, 1, 1)
//
// Steps:
// 1. Read hidden (O-proj output), compute rmsnorm with post_norm_w
// 2. Add old residual → write new hidden + save as new residual
// 3. Read new hidden, compute rmsnorm with ffn_norm_w
// 4. Write norm_out + quantize to q8_out
// =============================================================================

extern "C" __global__ void eval_rmsnorm_add_rmsnorm_q8(
        float       * __restrict__ hidden,
        const float * __restrict__ post_norm_w,
        const float * __restrict__ ffn_norm_w,
        float       * __restrict__ residual,
        float       * __restrict__ norm_out,
        block_q8_1  * __restrict__ q8_out,
        const int n) {

    const int tid = threadIdx.x;
    __shared__ float smem[32];

    // === Pass 1: post-attn rmsnorm + residual add ===
    constexpr int MAX_PER_THREAD = 16;
    float local_vals[MAX_PER_THREAD];
    int count = 0;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = hidden[i];
        local_vals[count++] = v;
        ss += v * v;
    }

    ss = warp_reduce_sum(ss);
    int wid = tid / WARP_SIZE, lid = tid % WARP_SIZE;
    if (lid == 0) smem[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        float s = (lid < (int)(blockDim.x / WARP_SIZE)) ? smem[lid] : 0.0f;
        s = warp_reduce_sum(s);
        if (lid == 0) smem[0] = rsqrtf(s / (float)n + NORM_EPS);
    }
    __syncthreads();
    float rstd1 = smem[0];

    float local_hidden[MAX_PER_THREAD];
    count = 0;
    float ss2 = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float w = post_norm_w[i];
#if NORM_ADD_ONE
        w = 1.0f + w;
#endif
        float normed = local_vals[count] * rstd1 * w;
        float new_h = normed + residual[i];
        hidden[i] = new_h;
        residual[i] = new_h;
        local_hidden[count] = new_h;
        ss2 += new_h * new_h;
        count++;
    }

    // === Pass 2: pre-FFN rmsnorm + Q8_1 quantize ===
    ss2 = warp_reduce_sum(ss2);
    if (lid == 0) smem[wid] = ss2;
    __syncthreads();
    if (wid == 0) {
        float s = (lid < (int)(blockDim.x / WARP_SIZE)) ? smem[lid] : 0.0f;
        s = warp_reduce_sum(s);
        if (lid == 0) smem[0] = rsqrtf(s / (float)n + NORM_EPS);
    }
    __syncthreads();
    float rstd2 = smem[0];

    count = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        float w = ffn_norm_w[i];
#if NORM_ADD_ONE
        w = 1.0f + w;
#endif
        float normed = local_hidden[count++] * rstd2 * w;
        norm_out[i] = normed;

        int ib = i / QK8_1;
        int iqs = i % QK8_1;
        float amax = fabsf(normed);
        float sum_val = normed;
        amax = warp_reduce_max<QK8_1>(amax);
        sum_val = warp_reduce_sum<QK8_1>(sum_val);
        float d = amax / 127.0f;
        q8_out[ib].qs[iqs] = (amax == 0.0f) ? 0 : (int8_t)roundf(normed / d);
        if (iqs == 0) q8_out[ib].ds = make_half2(d, sum_val);
    }
}

// =============================================================================
// Fused gate+up+silu for float types (F16, BF16, F32)
// These use f32 input (not Q8_1), different dot product algorithm.
// Grid: (out_dim, 1, 1), Block: (256, 1, 1)
// =============================================================================

extern "C" __global__ void __launch_bounds__(256, 1) eval_fused_gate_up_silu_f16(
        const void  * __restrict__ gate_weight,
        const long long gate_stride_bytes,
        const void  * __restrict__ up_weight,
        const long long up_stride_bytes,
        const void  * __restrict__ f32_input,
        float       * __restrict__ output,
        const int in_dim,
        const int out_dim) {
    const int row = blockIdx.x;
    if (row >= out_dim) return;
    constexpr int block_size = 256;
    const int tid = threadIdx.x;

    const __half2 * gx2 = (const __half2 *)((const char *)gate_weight + (long long)row * gate_stride_bytes);
    const __half2 * ux2 = (const __half2 *)((const char *)up_weight   + (long long)row * up_stride_bytes);
    const float2  * y2  = (const float2 *)f32_input;
    const int ncols2 = in_dim / 2;

    float gate_sum = 0.0f, up_sum = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 gy = __half22float2(gx2[col2]);
        const float2 uy = __half22float2(ux2[col2]);
        const float2 yy = y2[col2];
        gate_sum += gy.x * yy.x + gy.y * yy.y;
        up_sum   += uy.x * yy.x + uy.y * yy.y;
    }

    gate_sum = warp_reduce_sum<WARP_SIZE>(gate_sum);
    up_sum   = warp_reduce_sum<WARP_SIZE>(up_sum);
    __shared__ float buf_gate[WARP_SIZE];
    __shared__ float buf_up[WARP_SIZE];
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if (lane == 0) { buf_gate[warp] = gate_sum; buf_up[warp] = up_sum; }
    __syncthreads();
    if (tid < WARP_SIZE) {
        gate_sum = buf_gate[tid];
        up_sum   = buf_up[tid];
        gate_sum = warp_reduce_sum<WARP_SIZE>(gate_sum);
        up_sum   = warp_reduce_sum<WARP_SIZE>(up_sum);
    }
    if (tid == 0) {
        float silu_gate = gate_sum / (1.0f + expf(-gate_sum));
        output[row] = silu_gate * up_sum;
    }
}

extern "C" __global__ void __launch_bounds__(256, 1) eval_fused_gate_up_silu_bf16(
        const void  * __restrict__ gate_weight,
        const long long gate_stride_bytes,
        const void  * __restrict__ up_weight,
        const long long up_stride_bytes,
        const void  * __restrict__ f32_input,
        float       * __restrict__ output,
        const int in_dim,
        const int out_dim) {
    const int row = blockIdx.x;
    if (row >= out_dim) return;
    constexpr int block_size = 256;
    const int tid = threadIdx.x;

    const int * gx2 = (const int *)((const char *)gate_weight + (long long)row * gate_stride_bytes);
    const int * ux2 = (const int *)((const char *)up_weight   + (long long)row * up_stride_bytes);
    const float2 * y2 = (const float2 *)f32_input;
    const int ncols2 = in_dim / 2;

    float gate_sum = 0.0f, up_sum = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const int gx = gx2[col2];
        const int ux = ux2[col2];
        const float2 yy = y2[col2];
        const float gx0 = __uint_as_float(((uint32_t)(uint16_t)gx) << 16);
        const float gx1 = __uint_as_float(((uint32_t)(uint16_t)(gx >> 16)) << 16);
        const float ux0 = __uint_as_float(((uint32_t)(uint16_t)ux) << 16);
        const float ux1 = __uint_as_float(((uint32_t)(uint16_t)(ux >> 16)) << 16);
        gate_sum += gx0 * yy.x + gx1 * yy.y;
        up_sum   += ux0 * yy.x + ux1 * yy.y;
    }

    gate_sum = warp_reduce_sum<WARP_SIZE>(gate_sum);
    up_sum   = warp_reduce_sum<WARP_SIZE>(up_sum);
    __shared__ float buf_gate_b[WARP_SIZE];
    __shared__ float buf_up_b[WARP_SIZE];
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if (lane == 0) { buf_gate_b[warp] = gate_sum; buf_up_b[warp] = up_sum; }
    __syncthreads();
    if (tid < WARP_SIZE) {
        gate_sum = buf_gate_b[tid];
        up_sum   = buf_up_b[tid];
        gate_sum = warp_reduce_sum<WARP_SIZE>(gate_sum);
        up_sum   = warp_reduce_sum<WARP_SIZE>(up_sum);
    }
    if (tid == 0) {
        float silu_gate = gate_sum / (1.0f + expf(-gate_sum));
        output[row] = silu_gate * up_sum;
    }
}

extern "C" __global__ void __launch_bounds__(256, 1) eval_fused_gate_up_silu_f32(
        const void  * __restrict__ gate_weight,
        const long long gate_stride_bytes,
        const void  * __restrict__ up_weight,
        const long long up_stride_bytes,
        const void  * __restrict__ f32_input,
        float       * __restrict__ output,
        const int in_dim,
        const int out_dim) {
    const int row = blockIdx.x;
    if (row >= out_dim) return;
    constexpr int block_size = 256;
    const int tid = threadIdx.x;

    const float2 * gx2 = (const float2 *)((const char *)gate_weight + (long long)row * gate_stride_bytes);
    const float2 * ux2 = (const float2 *)((const char *)up_weight   + (long long)row * up_stride_bytes);
    const float2 * y2  = (const float2 *)f32_input;
    const int ncols2 = in_dim / 2;

    float gate_sum = 0.0f, up_sum = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 gx = gx2[col2];
        const float2 ux = ux2[col2];
        const float2 yy = y2[col2];
        gate_sum += gx.x * yy.x + gx.y * yy.y;
        up_sum   += ux.x * yy.x + ux.y * yy.y;
    }

    gate_sum = warp_reduce_sum<WARP_SIZE>(gate_sum);
    up_sum   = warp_reduce_sum<WARP_SIZE>(up_sum);
    __shared__ float buf_gate_f[WARP_SIZE];
    __shared__ float buf_up_f[WARP_SIZE];
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if (lane == 0) { buf_gate_f[warp] = gate_sum; buf_up_f[warp] = up_sum; }
    __syncthreads();
    if (tid < WARP_SIZE) {
        gate_sum = buf_gate_f[tid];
        up_sum   = buf_up_f[tid];
        gate_sum = warp_reduce_sum<WARP_SIZE>(gate_sum);
        up_sum   = warp_reduce_sum<WARP_SIZE>(up_sum);
    }
    if (tid == 0) {
        float silu_gate = gate_sum / (1.0f + expf(-gate_sum));
        output[row] = silu_gate * up_sum;
    }
}

// =============================================================================
// Fused MoE matvec for float types (F16, BF16, F32)
// These use f32 input (not Q8_1), different algorithm than quantized types.
// Grid: (out_dim, 1, 1), Block: (256, 1, 1)
// =============================================================================

// F16 MoE matvec — reads expert ID from GPU, computes dot product with f32 input
extern "C" __global__ void eval_moe_matvec_f16(
        const void  * __restrict__ weight_base,
        const long long expert_stride,
        const long long row_stride_bytes,
        const void  * __restrict__ f32_input,     // f32 activation (not q8)
        float       * __restrict__ output,
        const int   * __restrict__ sorted_ids,
        const int expert_rank,
        const int in_dim,
        const int out_dim) {
    __shared__ int s_eid;
    if (threadIdx.x == 0) s_eid = sorted_ids[expert_rank];
    __syncthreads();

    const int row = blockIdx.x;
    if (row >= out_dim) return;

    const __half2 * x2 = (const __half2 *)((const char *)weight_base + (long long)s_eid * expert_stride
                                            + (long long)row * row_stride_bytes);
    const float2 * y2 = (const float2 *)f32_input;
    const int ncols2 = in_dim / 2;

    float sumf = 0.0f;
    for (int col2 = threadIdx.x; col2 < ncols2; col2 += blockDim.x) {
        const float2 tmpx = __half22float2(x2[col2]);
        const float2 tmpy = y2[col2];
        sumf += tmpx.x * tmpy.x + tmpx.y * tmpy.y;
    }
    sumf = warp_reduce_sum<WARP_SIZE>(sumf);

    __shared__ float buf_iw[WARP_SIZE];
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) buf_iw[warp] = sumf;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        int nw = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        for (int w = 0; w < nw; w++) total += buf_iw[w];
        output[row] = total;
    }
}

// BF16 MoE matvec
extern "C" __global__ void eval_moe_matvec_bf16(
        const void  * __restrict__ weight_base,
        const long long expert_stride,
        const long long row_stride_bytes,
        const void  * __restrict__ f32_input,
        float       * __restrict__ output,
        const int   * __restrict__ sorted_ids,
        const int expert_rank,
        const int in_dim,
        const int out_dim) {
    __shared__ int s_eid;
    if (threadIdx.x == 0) s_eid = sorted_ids[expert_rank];
    __syncthreads();

    const int row = blockIdx.x;
    if (row >= out_dim) return;

    const int * x2 = (const int *)((const char *)weight_base + (long long)s_eid * expert_stride
                                    + (long long)row * row_stride_bytes);
    const float2 * y2 = (const float2 *)f32_input;
    const int ncols2 = in_dim / 2;

    float sumf = 0.0f;
    for (int col2 = threadIdx.x; col2 < ncols2; col2 += blockDim.x) {
        const int tmpx = x2[col2];
        const float2 tmpy = y2[col2];
        const float tmpx0 = __uint_as_float(((uint32_t)(uint16_t)tmpx) << 16);
        const float tmpx1 = __uint_as_float(((uint32_t)(uint16_t)(tmpx >> 16)) << 16);
        sumf += tmpx0 * tmpy.x + tmpx1 * tmpy.y;
    }
    sumf = warp_reduce_sum<WARP_SIZE>(sumf);

    __shared__ float buf_iw[WARP_SIZE];
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) buf_iw[warp] = sumf;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        int nw = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        for (int w = 0; w < nw; w++) total += buf_iw[w];
        output[row] = total;
    }
}

// F32 MoE matvec
extern "C" __global__ void eval_moe_matvec_f32(
        const void  * __restrict__ weight_base,
        const long long expert_stride,
        const long long row_stride_bytes,
        const void  * __restrict__ f32_input,
        float       * __restrict__ output,
        const int   * __restrict__ sorted_ids,
        const int expert_rank,
        const int in_dim,
        const int out_dim) {
    __shared__ int s_eid;
    if (threadIdx.x == 0) s_eid = sorted_ids[expert_rank];
    __syncthreads();

    const int row = blockIdx.x;
    if (row >= out_dim) return;

    const float2 * x2 = (const float2 *)((const char *)weight_base + (long long)s_eid * expert_stride
                                           + (long long)row * row_stride_bytes);
    const float2 * y2 = (const float2 *)f32_input;
    const int ncols2 = in_dim / 2;

    float sumf = 0.0f;
    for (int col2 = threadIdx.x; col2 < ncols2; col2 += blockDim.x) {
        const float2 tmpx = x2[col2];
        const float2 tmpy = y2[col2];
        sumf += tmpx.x * tmpy.x + tmpx.y * tmpy.y;
    }
    sumf = warp_reduce_sum<WARP_SIZE>(sumf);

    __shared__ float buf_iw[WARP_SIZE];
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) buf_iw[warp] = sumf;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        int nw = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        for (int w = 0; w < nw; w++) total += buf_iw[w];
        output[row] = total;
    }
}

// MoE weight normalization — normalizes the top-K routing weights in-place on GPU
// Reads probs[sorted_ids[0..n_used-1]], normalizes, writes back.
// Single block, single thread (n_used is small, typically 2-8)
extern "C" __global__ void eval_moe_normalize_weights(
        float       * __restrict__ probs,        // [n_expert] routing probs — modified in-place
        const int   * __restrict__ sorted_ids,   // [n_expert] sorted indices
        const int n_used,
        const int do_normalize,                  // 1 = normalize, 0 = skip
        const float w_scale) {                   // weight scale multiplier
    if (threadIdx.x != 0) return;

    if (do_normalize) {
        float wsum = 0.0f;
        for (int i = 0; i < n_used; i++) wsum += probs[sorted_ids[i]];
        if (wsum < 6.103515625e-5f) wsum = 6.103515625e-5f;
        for (int i = 0; i < n_used; i++) probs[sorted_ids[i]] /= wsum;
    }

    if (w_scale != 0.0f && w_scale != 1.0f) {
        for (int i = 0; i < n_used; i++) probs[sorted_ids[i]] *= w_scale;
    }
}

// Fill integer buffer with sequential values: dst[i] = start + i
// Used for pos_embd position generation on GPU (no host std::vector needed)
// Grid: ((n+255)/256, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void eval_fill_positions(
        int * __restrict__ dst,
        const int start,
        const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = start + i;
}

// Max reduction for mel normalization — writes max value to dst[0]
// Grid: (1, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void eval_max_reduce(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        const int n) {
    __shared__ float sdata[256];
    float maxv = -3.402823466e+38f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        maxv = fmaxf(maxv, src[i]);
    }
    sdata[threadIdx.x] = maxv;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) dst[0] = sdata[0];
}

// =============================================================================
// Batched MoE kernels — group tokens by expert for prompt-path dispatch
// =============================================================================

// Batch MoE weight normalization — normalizes top-K routing weights for S tokens in parallel
// probs: [S, n_expert] routing probs (in-place), sorted_ids: [S, n_expert] sorted expert indices
// Grid: (S, 1, 1), Block: (1, 1, 1) — one block per token, single-thread (n_used is small)
extern "C" __global__ void eval_moe_batch_normalize_weights(
        float       * __restrict__ probs,         // [S * n_expert] routing probs — modified in-place
        const int   * __restrict__ sorted_ids,    // [S * n_expert] sorted expert indices
        const int n_expert,                       // number of experts
        const int n_used,                         // top-K experts per token
        const int do_normalize,                   // 1 = normalize, 0 = skip
        const float w_scale) {                    // weight scale multiplier
    if (threadIdx.x != 0) return;
    const int t = blockIdx.x; // token index
    float * p = probs + t * n_expert;
    const int * ids = sorted_ids + t * n_expert;

    if (do_normalize) {
        float wsum = 0.0f;
        for (int i = 0; i < n_used; i++) wsum += p[ids[i]];
        if (wsum < 6.103515625e-5f) wsum = 6.103515625e-5f;
        for (int i = 0; i < n_used; i++) p[ids[i]] /= wsum;
    }

    if (w_scale != 0.0f && w_scale != 1.0f) {
        for (int i = 0; i < n_used; i++) p[ids[i]] *= w_scale;
    }
}

// Group tokens by expert for batched MoE dispatch.
// Reads sorted_ids[S, n_expert] (top n_used entries per row are the selected experts).
// Outputs:
//   expert_counts[n_expert] — how many tokens assigned to each expert
//   token_map[total_assignments * 2] — (token_idx, expert_rank) pairs grouped by expert
//     where total_assignments = S * n_used
// Grid: (1, 1, 1), Block: (1, 1, 1) — single thread, S*n_used is small
extern "C" __global__ void eval_moe_group_tokens(
        const int * __restrict__ sorted_ids,    // [S, n_expert] sorted expert IDs per token
        int       * __restrict__ expert_counts, // [n_expert] output: count per expert
        int       * __restrict__ token_map,     // [S * n_used * 2] output: (token_idx, rank) pairs
        const int S,                            // number of tokens in batch
        const int n_expert,                     // total number of experts (stride for sorted_ids)
        const int n_used) {                     // top-K experts per token
    if (threadIdx.x != 0) return;

    // Zero counts
    for (int e = 0; e < n_expert; e++) expert_counts[e] = 0;

    // Count tokens per expert
    for (int t = 0; t < S; t++) {
        for (int k = 0; k < n_used; k++) {
            int eid = sorted_ids[t * n_expert + k]; // top-K are first n_used in sorted order
            if (eid >= 0 && eid < n_expert) expert_counts[eid]++;
        }
    }

    // Compute exclusive prefix sums for offsets
    int offsets[256]; // max 256 experts
    offsets[0] = 0;
    for (int e = 1; e < n_expert; e++) offsets[e] = offsets[e-1] + expert_counts[e-1];

    // Fill token_map: for each (token, rank), place at the expert's offset
    int pos[256];
    for (int e = 0; e < n_expert; e++) pos[e] = offsets[e];

    for (int t = 0; t < S; t++) {
        for (int k = 0; k < n_used; k++) {
            int eid = sorted_ids[t * n_expert + k];
            if (eid >= 0 && eid < n_expert) {
                int idx = pos[eid]++;
                token_map[idx * 2 + 0] = t;  // token index
                token_map[idx * 2 + 1] = k;  // expert rank (for weight lookup)
            }
        }
    }
}

// Gather scattered token rows into a contiguous buffer for batched GEMM.
// For each assignment in the map, copies src[token_idx * row_stride .. + dim-1]
// into dst[i * dim .. + dim-1] where i is the position in the map.
// Grid: (n_tok, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void eval_moe_gather(
        const float * __restrict__ src,         // [S, dim] source (batch_norm)
        float       * __restrict__ dst,         // [n_tok, dim] destination (contiguous)
        const int   * __restrict__ token_map,   // [total * 2] (token_idx, rank) pairs
        const int map_offset,                   // offset into token_map for this expert
        const int dim) {                        // row dimension (H)
    const int i = blockIdx.x;  // which assigned token (0..n_tok-1)
    const int token_idx = token_map[(map_offset + i) * 2 + 0];

    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        dst[i * dim + j] = src[token_idx * dim + j];
    }
}

// Scatter expert FFN output back to per-token accumulators with routing weight.
// For each assignment, reads the routing weight from probs[sorted_ids[token * n_expert + rank]]
// and does: dst[token_idx * dim + j] += weight * src[i * dim + j]
// Grid: (n_tok, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void eval_moe_scatter_weighted_add(
        float       * __restrict__ dst,         // [S, dim] accumulator (batch_hidden)
        const float * __restrict__ src,         // [n_tok, dim] expert output
        const float * __restrict__ probs,       // [S, n_expert] routing probabilities
        const int   * __restrict__ sorted_ids,  // [S, n_expert] sorted expert indices
        const int   * __restrict__ token_map,   // [total * 2] (token_idx, rank) pairs
        const int map_offset,                   // offset into token_map for this expert
        const int n_expert,                     // stride for probs/sorted_ids rows
        const int dim) {                        // row dimension (H)
    const int i = blockIdx.x;  // which assigned token (0..n_tok-1)
    const int token_idx = token_map[(map_offset + i) * 2 + 0];
    const int rank      = token_map[(map_offset + i) * 2 + 1];

    // Read routing weight: probs[token_idx][sorted_ids[token_idx][rank]]
    __shared__ float s_weight;
    if (threadIdx.x == 0) {
        int eid = sorted_ids[token_idx * n_expert + rank];
        s_weight = probs[token_idx * n_expert + eid];
    }
    __syncthreads();
    float w = s_weight;

    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        atomicAdd(&dst[token_idx * dim + j], w * src[i * dim + j]);
    }
}
