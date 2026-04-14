// prefill.hip — Custom kernels for prompt prefill on gfx1100
//
// Host-orchestrated: rocBLAS GEMM for matrix multiplications,
// these kernels for everything rocBLAS can't do.
//
// Launched between rocBLAS calls by the host orchestrator in gfx1100-megakernel.cpp.
// Math ported from ggml-cuda. Model constants baked at compile time.

#include "helpers/hip-shim.h"
#include "helpers/warp-reduce.h"
#include "helpers/rope.h"
#include "helpers/quant-types.h"
#include "helpers/iq-tables.h"
#include "helpers/vec-dot.h"
#include "helpers/activations.h"
#include "helpers/dequant.h"
#include "helpers/embed-dequant.h"
#include "helpers/deltanet.h"
#include "helpers/rope.h"
#include "helpers/mem-utils.h"
#include "helpers/mmq.h"

// =============================================================================
// Model constants (same as decode.hip, filled by template)
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
// Batch Embedding: token_ids[S] → output[S, HIDDEN_SIZE] (all quant types)
//
// Grid: (nb_superblocks, S)  —  blockIdx.y = token index, blockIdx.x = superblock
// Each kernel mirrors its eval_embed_* counterpart but handles S tokens at once.
// =============================================================================

// K-quant batch embed: 64 (or 32) threads per block, QK_K=256 elements per superblock
// block_type: typed row pointer type; dequant_fn: dequantize_block_TYPE_device
// threads argument is documentation only — actual blockDim.x set by host
#define PROMPT_EMBED_KQUANT(name, block_type, dequant_fn) \
extern "C" __global__ void prompt_embed_##name( \
        const int     * __restrict__ token_ids, \
        const void    * __restrict__ embed_weight, \
        const long long              embed_stride, \
        float         * __restrict__ output, \
        const int S) { \
    int s = blockIdx.y; \
    if (s >= S) return; \
    int b = blockIdx.x; \
    const int nb = HIDDEN_SIZE / QK_K; \
    if (b >= nb) return; \
    const block_type * row = (const block_type *)((const char *)embed_weight + (long long)token_ids[s] * embed_stride); \
    float * out = output + (long long)s * HIDDEN_SIZE; \
    dequant_fn(row, out, b, threadIdx.x); \
}

// Small-block batch embed: 32 threads, 32 elements per block (QK4_0/QK5_0/QK8_0 = 32)
// dequant_fn takes (vx, y, block_idx, tid) where vx is the row start
// Grid: (H/32, S, 1) — one block per 32-element quant block
#define PROMPT_EMBED_SMALL(name, dequant_fn) \
extern "C" __global__ void prompt_embed_##name( \
        const int     * __restrict__ token_ids, \
        const void    * __restrict__ embed_weight, \
        const long long              embed_stride, \
        float         * __restrict__ output, \
        const int S) { \
    int s = blockIdx.y; \
    if (s >= S) return; \
    int b = blockIdx.x; \
    const int nb = HIDDEN_SIZE / 32; \
    if (b >= nb) return; \
    const void * row = (const char *)embed_weight + (long long)token_ids[s] * embed_stride; \
    float * out = output + (long long)s * HIDDEN_SIZE; \
    dequant_fn(row, out, b, threadIdx.x); \
}

// Q6_K (64 threads, retained from original implementation)
PROMPT_EMBED_KQUANT(q6k,     block_q6_K,   dequantize_block_q6_K_device)
// Q4_K (32 threads)
PROMPT_EMBED_KQUANT(q4k,     block_q4_K,   dequantize_block_q4_K_device)
// Q5_K (64 threads)
PROMPT_EMBED_KQUANT(q5k,     block_q5_K,   dequantize_block_q5_K_device)
// Q2_K (64 threads)
PROMPT_EMBED_KQUANT(q2k,     block_q2_K,   dequantize_block_q2_K_device)
// Q3_K (64 threads)
PROMPT_EMBED_KQUANT(q3k,     block_q3_K,   dequantize_block_q3_K_device)
// Small-block types (32 threads, 8 small blocks per QK_K superblock)
PROMPT_EMBED_SMALL(q4_0,     dequantize_block_q4_0_device)
PROMPT_EMBED_SMALL(q4_1,     dequantize_block_q4_1_device)
PROMPT_EMBED_SMALL(q5_0,     dequantize_block_q5_0_device)
PROMPT_EMBED_SMALL(q5_1,     dequantize_block_q5_1_device)
PROMPT_EMBED_SMALL(q8_0,     dequantize_block_q8_0_device)
PROMPT_EMBED_SMALL(iq4_nl,   dequantize_block_iq4_nl_device)
PROMPT_EMBED_SMALL(mxfp4,    dequantize_block_mxfp4_device)
// IQ K-quant types (32 threads)
PROMPT_EMBED_KQUANT(iq2_xxs, block_iq2_xxs, dequantize_block_iq2_xxs_device)
PROMPT_EMBED_KQUANT(iq2_xs,  block_iq2_xs,  dequantize_block_iq2_xs_device)
PROMPT_EMBED_KQUANT(iq2_s,   block_iq2_s,   dequantize_block_iq2_s_device)
PROMPT_EMBED_KQUANT(iq3_xxs, block_iq3_xxs, dequantize_block_iq3_xxs_device)
PROMPT_EMBED_KQUANT(iq3_s,   block_iq3_s,   dequantize_block_iq3_s_device)
PROMPT_EMBED_KQUANT(iq1_s,   block_iq1_s,   dequantize_block_iq1_s_device)
PROMPT_EMBED_KQUANT(iq1_m,   block_iq1_m,   dequantize_block_iq1_m_device)
PROMPT_EMBED_KQUANT(iq4_xs,  block_iq4_xs,  dequantize_block_iq4_xs_device)

#undef PROMPT_EMBED_KQUANT
#undef PROMPT_EMBED_SMALL

// NVFP4 batch embed (QK=64, 512 elements per superblock — different superblock size)
extern "C" __global__ void prompt_embed_nvfp4(
        const int     * __restrict__ token_ids,
        const void    * __restrict__ embed_weight,
        const long long              embed_stride,
        float         * __restrict__ output,
        const int S) {
    int s = blockIdx.y;
    if (s >= S) return;
    int b = blockIdx.x;
    const int nb = HIDDEN_SIZE / 512;  // NVFP4: 512 elements per superblock
    if (b >= nb) return;
    const void * row = (const char *)embed_weight + (long long)token_ids[s] * embed_stride;
    float * out = output + (long long)s * HIDDEN_SIZE;
    dequantize_block_nvfp4_device(row, out, b, threadIdx.x);
}

// F32 batch embed (flat thread grid: blockDim.x=256, blockIdx.x covers H elements)
extern "C" __global__ void prompt_embed_f32(
        const int     * __restrict__ token_ids,
        const void    * __restrict__ embed_weight,
        const long long              embed_stride,
        float         * __restrict__ output,
        const int S) {
    int s = blockIdx.y;
    if (s >= S) return;
    const float * row = (const float *)((const char *)embed_weight + (long long)token_ids[s] * embed_stride);
    float * out = output + (long long)s * HIDDEN_SIZE;
    dequantize_block_f32_device(row, out, 0, threadIdx.x + blockIdx.x * blockDim.x, HIDDEN_SIZE);
}

// F16 batch embed
extern "C" __global__ void prompt_embed_f16(
        const int     * __restrict__ token_ids,
        const void    * __restrict__ embed_weight,
        const long long              embed_stride,
        float         * __restrict__ output,
        const int S) {
    int s = blockIdx.y;
    if (s >= S) return;
    const __half * row = (const __half *)((const char *)embed_weight + (long long)token_ids[s] * embed_stride);
    float * out = output + (long long)s * HIDDEN_SIZE;
    dequantize_block_f16_device(row, out, 0, threadIdx.x + blockIdx.x * blockDim.x, HIDDEN_SIZE);
}

// BF16 batch embed
extern "C" __global__ void prompt_embed_bf16(
        const int     * __restrict__ token_ids,
        const void    * __restrict__ embed_weight,
        const long long              embed_stride,
        float         * __restrict__ output,
        const int S) {
    int s = blockIdx.y;
    if (s >= S) return;
    const uint16_t * row = (const uint16_t *)((const char *)embed_weight + (long long)token_ids[s] * embed_stride);
    float * out = output + (long long)s * HIDDEN_SIZE;
    dequantize_block_bf16_device(row, out, 0, threadIdx.x + blockIdx.x * blockDim.x, HIDDEN_SIZE);
}

// =============================================================================
// Batched RMSNorm: f32 in → f32 out + f32 residual copy
// =============================================================================

extern "C" __global__ void prompt_rmsnorm(
        const float * __restrict__ input,    // [S, D]
        const float * __restrict__ weight,   // [D]
        float       * __restrict__ output,   // [S, D]
        float       * __restrict__ residual, // [S, D]
        const int S, const int D) {
    int s = blockIdx.x;
    if (s >= S) return;
    int tid = threadIdx.x, wid = tid / 32, lid = tid % 32;
    __shared__ float smem[32];

    const float * row_in = input + s * D;
    float * row_out = output + s * D;
    float * row_res = residual + s * D;

    float sq = 0;
    for (int i = tid; i < D; i += blockDim.x) {
        float v = row_in[i];
        row_res[i] = v;
        sq += v * v;
    }
    sq = warp_reduce_sum(sq);
    if (lid == 0) smem[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        float v = (lid < (int)(blockDim.x / 32)) ? smem[lid] : 0;
        v = warp_reduce_sum(v);
        if (lid == 0) smem[0] = rsqrtf(v / (float)D + NORM_EPS);
    }
    __syncthreads();
    float rstd = smem[0];
    for (int i = tid; i < D; i += blockDim.x) {
        float w = weight[i];
#if NORM_ADD_ONE
        w = 1.0f + w;
#endif
        row_out[i] = row_in[i] * rstd * w;
    }
}

// Batch LayerNorm — ported from baseline ggml-cuda/norm.cu::norm_f32
// BERT uses LayerNorm (not RMSNorm): output = (x - mean) / sqrt(var + eps) * weight + bias
// Grid: (S, 1, 1), block: (256 or 1024, 1, 1)
// Each block processes one token's hidden state.
extern "C" __global__ void prompt_layernorm(
        const float * __restrict__ input,    // [S, D]
        const float * __restrict__ weight,   // [D] or NULL
        const float * __restrict__ bias,     // [D] or NULL
        float       * __restrict__ output,   // [S, D]
        const int D,
        const float eps) {
    int s = blockIdx.x;
    int tid = threadIdx.x, wid = tid / 32, lid = tid % 32;
    __shared__ float2 smem2[32]; // up to 32 warps

    const float * row_in = input + (int64_t)s * D;
    float * row_out = output + (int64_t)s * D;

    // Pass 1: accumulate sum and sum-of-squares (float2)
    float2 ms = make_float2(0.0f, 0.0f);
    for (int i = tid; i < D; i += blockDim.x) {
        float v = row_in[i];
        ms.x += v;
        ms.y += v * v;
    }
    // Warp reduce
    ms.x = warp_reduce_sum(ms.x);
    ms.y = warp_reduce_sum(ms.y);
    if (lid == 0) smem2[wid] = ms;
    __syncthreads();
    if (wid == 0) {
        float2 v = (lid < (int)(blockDim.x / 32)) ? smem2[lid] : make_float2(0.0f, 0.0f);
        v.x = warp_reduce_sum(v.x);
        v.y = warp_reduce_sum(v.y);
        if (lid == 0) {
            const float inv_n = 1.0f / (float)D;
            const float mean = v.x * inv_n;
            const float var = v.y * inv_n - mean * mean;
            smem2[0].x = mean;
            smem2[0].y = rsqrtf(var + eps);
        }
    }
    __syncthreads();
    const float mean = smem2[0].x;
    const float inv_std = smem2[0].y;

    // Pass 2: normalize + optional weight/bias
    for (int i = tid; i < D; i += blockDim.x) {
        float y = (row_in[i] - mean) * inv_std;
        if (weight) y *= weight[i];
        if (bias) y += bias[i];
        row_out[i] = y;
    }
}

// Per-head LayerNorm for BERT QK norm (Jina-BERT-V2, NomicBERT)
// Applies LayerNorm independently to each head's head_dim elements.
// Input: [total_groups, head_dim] where total_groups = S * n_head
// Grid: (total_groups, 1, 1), block: (min(head_dim, 256), 1, 1)
// Ported from baseline norm.cu — per-group normalization
extern "C" __global__ void prompt_per_head_layernorm(
        const float * __restrict__ input,    // [total_groups, head_dim]
        const float * __restrict__ weight,   // [head_dim] or NULL
        const float * __restrict__ bias,     // [head_dim] or NULL
        float       * __restrict__ output,   // [total_groups, head_dim]
        const int head_dim,
        const float eps) {
    int g = blockIdx.x; // group index (s * n_head + head)
    int tid = threadIdx.x, wid = tid / 32, lid = tid % 32;
    __shared__ float2 smem2[32];

    const float * row_in = input + (int64_t)g * head_dim;
    float * row_out = output + (int64_t)g * head_dim;

    float2 ms = make_float2(0.0f, 0.0f);
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = row_in[i];
        ms.x += v;
        ms.y += v * v;
    }
    ms.x = warp_reduce_sum(ms.x);
    ms.y = warp_reduce_sum(ms.y);
    if (lid == 0) smem2[wid] = ms;
    __syncthreads();
    if (wid == 0) {
        float2 v = (lid < (int)(blockDim.x / 32)) ? smem2[lid] : make_float2(0.0f, 0.0f);
        v.x = warp_reduce_sum(v.x);
        v.y = warp_reduce_sum(v.y);
        if (lid == 0) {
            const float inv_n = 1.0f / (float)head_dim;
            const float mean = v.x * inv_n;
            const float var = v.y * inv_n - mean * mean;
            smem2[0].x = mean;
            smem2[0].y = rsqrtf(var + eps);
        }
    }
    __syncthreads();
    const float mean = smem2[0].x;
    const float inv_std = smem2[0].y;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        float y = (row_in[i] - mean) * inv_std;
        if (weight) y *= weight[i];
        if (bias) y += bias[i];
        row_out[i] = y;
    }
}

// Batch NeoX-style RoPE for encoder tokens (ModernBERT, NomicBERT)
// Ported from baseline modern-bert.cpp lines 50-60: ggml_rope_ext on Q and K
// Applies NeoX RoPE with full YaRN params to data[s * dim + i] for s in [0, S)
// Grid: (S, 1, 1), block: (min(n_dims/2, 256), 1, 1)
extern "C" __global__ void prompt_rope_neox_inplace(
        float       * __restrict__ data,     // [S, dim] — modified in-place
        const int dim,                       // total dim per token
        const int n_dims,                    // RoPE dim
        const float theta_base,
        const int start_pos) {               // first position index
    int s = blockIdx.x;
    int i = threadIdx.x;
    if (i >= n_dims / 2) return;

    int pos = start_pos + s;
    float * row = data + (int64_t)s * dim;

    float theta_extrap = (float)pos * powf(theta_base, -(float)(i * 2) / (float)n_dims);
    // Apply rope_yarn with compile-time YaRN params (same as fused QK-norm-RoPE)
    float cv, sv;
    rope_yarn<true>(theta_extrap, ROPE_FREQ_SCALE,
                   rope_corr_dims{YARN_CORR_DIM0, YARN_CORR_DIM1},
                   i*2, YARN_EXT_FACTOR, YARN_ATTN_FACTOR, cv, sv);
    float x0 = row[i];
    float x1 = row[i + n_dims / 2];
    row[i]              = x0 * cv - x1 * sv;
    row[i + n_dims / 2] = x0 * sv + x1 * cv;
}

// =============================================================================
// Residual add: output = a + b
// =============================================================================

// Broadcast bias add: output[s*dim + i] += bias[i] for all s in [0, S)
// Ported from baseline binbcast.cu op_add with broadcast dim.
// Grid: ((S*dim+255)/256), Block: 256
extern "C" __global__ void prompt_add_bias(
        float       * __restrict__ data,   // [S, dim] — in-place
        const float * __restrict__ bias,   // [dim]
        const int dim, const int total) {  // total = S * dim
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) data[i] += bias[i % dim];
}

// Broadcast element-wise multiply: data[s*dim + i] *= scale[i] for all s
// Ported from baseline binbcast.cu op_mul with broadcast dim.
extern "C" __global__ void prompt_elementwise_mul(
        float       * __restrict__ data,   // [S, dim] — in-place
        const float * __restrict__ scale,  // [dim]
        const int dim, const int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) data[i] *= scale[i % dim];
}

// Position embedding lookup + add — ported from baseline ggml_get_rows + ggml_add
// For BERT: output[s*D + i] += pos_embd[positions[s]*D + i]
// Grid: (ceil(D/256), S, 1), Block: (256, 1, 1)
// Handles arbitrary position arrays (not just sequential 0,1,2,...)
extern "C" __global__ void prompt_add_pos_embd(
        float       * __restrict__ output,       // [S, D] f32 token embeddings (in-place add)
        const float * __restrict__ pos_embd,     // [max_pos, D] f32 position embedding table
        const int   * __restrict__ positions,    // [S] position indices
        const int D,                             // hidden dimension
        const int S) {                           // number of tokens
    int s = blockIdx.y;
    if (s >= S) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;
    int pos = positions[s];
    output[s * D + i] += pos_embd[(int64_t)pos * D + i];
}

extern "C" __global__ void prompt_add_residual(
        const float * __restrict__ a,
        const float * __restrict__ b,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = a[i] + b[i];
}

// =============================================================================
// SiLU(gate) * up
// =============================================================================

extern "C" __global__ void prompt_silu_mul(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        float       * __restrict__ output,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = op_silu(gate[i]) * up[i];
}

// =============================================================================
// QK Norm + RoPE + KV cache write (batched, all positions)
// One warp per (position, head) pair.
// =============================================================================

extern "C" __global__ void prompt_qk_norm_rope(
        float        * __restrict__ q,        // [S, FA_QPROJ_SIZE]
        float        * __restrict__ k,        // [S, FA_KV_SIZE]
        const float  * __restrict__ v,        // [S, FA_KV_SIZE]
        const float  * __restrict__ q_norm_w, // [FA_HEAD_DIM] or NULL
        const float  * __restrict__ k_norm_w, // [FA_HEAD_DIM] or NULL
        __half       * __restrict__ k_cache,  // [n_kv_heads, max_seq, head_dim]
        __half       * __restrict__ v_cache,
        const float  * __restrict__ freq_factors, // [FA_ROPE_DIM/2] or NULL — baseline rope.cu line 62
        const int S, const int max_seq,
        const int start_pos,
        const float rope_theta_override) { // 0 = use FA_ROPE_THETA, non-zero = override (for iSWA per-layer theta), <0 = skip RoPE

    int idx = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    // Per-layer theta: >0 = override, 0 = compile-time default, <0 = skip RoPE (Cohere2 non-SWA)
    const float theta_base_val = (rope_theta_override > 0.0f) ? rope_theta_override : FA_ROPE_THETA;
    const bool skip_rope = (rope_theta_override < 0.0f);
    int total_q = S * FA_N_Q_HEADS;
    int total_k = S * FA_N_KV_HEADS;

    // Q heads: norm + rope (in-place)
    if (idx < total_q) {
        int local_pos = idx / FA_N_Q_HEADS;
        int head      = idx % FA_N_Q_HEADS;
        int abs_pos   = start_pos + local_pos; // absolute sequence position for RoPE
        float * qh = q + local_pos * FA_QPROJ_SIZE + head * (FA_HAS_GATED_ATTN ? FA_HEAD_DIM * 2 : FA_HEAD_DIM);

        // QK RMSNorm — baseline line 86: bare ggml_rms_norm, no weight multiply
        if (q_norm_w) {
            float ss = 0;
            for (int i = lid; i < FA_HEAD_DIM; i += 32) ss += qh[i] * qh[i];
            ss = warp_reduce_sum(ss);
            float sc = rsqrtf(ss / (float)FA_HEAD_DIM + NORM_EPS);
            sc = __shfl_sync(0xffffffff, sc, 0, 32);
            for (int i = lid; i < FA_HEAD_DIM; i += 32) {
                qh[i] = qh[i] * sc;  // bare norm, no weight
            }
        }

        // RoPE — dispatched by ROPE_TYPE macro, with YaRN support via rope_yarn()
#if ROPE_TYPE == 0 /* ROPE_NONE — ALiBi, GPT-2, T5 */
        // No rotation applied
#elif ROPE_TYPE == 3 /* ROPE_MULTI */
        if (!skip_rope) {
        for (int i = lid; i < FA_ROPE_DIM / 2; i += WARP_SIZE) {
            float theta_base = (float)abs_pos * powf(theta_base_val, -(float)(i*2) / (float)FA_ROPE_DIM);
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
#elif ROPE_TYPE == 2 /* ROPE_NEOX */
        if (!skip_rope) {
        for (int i = lid; i < FA_ROPE_DIM / 2; i += WARP_SIZE) {
            float theta_base = (float)abs_pos * powf(theta_base_val, -(float)(i*2) / (float)FA_ROPE_DIM);
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
            float theta_base = (float)abs_pos * powf(theta_base_val, -(float)i / (float)FA_ROPE_DIM);
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

    // K heads: norm + rope + cache write
    int kidx = idx - total_q;
    if (idx >= total_q && kidx < total_k) {
        int local_pos = kidx / FA_N_KV_HEADS;
        int head      = kidx % FA_N_KV_HEADS;
        int abs_pos   = start_pos + local_pos; // absolute sequence position for RoPE + cache slot
        float * kh = k + local_pos * FA_KV_SIZE + head * FA_HEAD_DIM;
        const float * vh = v + local_pos * FA_KV_SIZE + head * FA_HEAD_DIM;
        __half * kc = k_cache + head * max_seq * FA_HEAD_DIM + abs_pos * FA_HEAD_DIM;
        __half * vc = v_cache + head * max_seq * FA_HEAD_DIM + abs_pos * FA_HEAD_DIM;

        // K QK RMSNorm — baseline line 87: bare ggml_rms_norm, no weight
        if (k_norm_w) {
            float ss = 0;
            for (int i = lid; i < FA_HEAD_DIM; i += 32) ss += kh[i] * kh[i];
            ss = warp_reduce_sum(ss);
            float sc = rsqrtf(ss / (float)FA_HEAD_DIM + NORM_EPS);
            sc = __shfl_sync(0xffffffff, sc, 0, 32);
            for (int i = lid; i < FA_HEAD_DIM; i += 32) {
                kh[i] = kh[i] * sc;
            }
        }

        // K RoPE — dispatched by ROPE_TYPE, with skip_rope guard
#if ROPE_TYPE == 0 /* ROPE_NONE */
        // No rotation
#elif ROPE_TYPE == 3 /* ROPE_MULTI */
        if (!skip_rope) {
        for (int i = lid; i < FA_ROPE_DIM / 2; i += WARP_SIZE) {
            float theta_base = (float)abs_pos * powf(theta_base_val, -(float)(i*2) / (float)FA_ROPE_DIM);
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
            float theta_base = (float)abs_pos * powf(theta_base_val, -(float)(i*2) / (float)FA_ROPE_DIM);
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
            float theta_base = (float)abs_pos * powf(theta_base_val, -(float)i / (float)FA_ROPE_DIM);
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

        // Write to cache at absolute position slot
        for (int i = lid; i < FA_HEAD_DIM; i += 32) {
            kc[i] = __float2half(kh[i]);
            vc[i] = __float2half(vh[i]);
        }
    }
}

// =============================================================================
// Causal attention (f32 Q/K/V, f32 accumulation, f32 output)
// One warp per (position, q_head) pair. Online softmax.
// =============================================================================

// Batch causal attention — reads from KV CACHE (f16), not batch buffers.
// This handles both initial prompt (start_pos=0) and multi-turn (start_pos>0).
// Q is f32 from batch projection, K/V are f16 from KV cache at [0..start_pos+S-1].
// Ported from baseline fattn-vec.cuh: iterates over ALL kv_len positions with causal mask.
// Grid: (ceil(S * FA_N_Q_HEADS / warps_per_block)), block: (512 = 16 warps)
extern "C" __global__ void prompt_causal_attn(
        const float  * __restrict__ q,       // [S, FA_QPROJ_SIZE] f32 batch Q
        const __half * __restrict__ k_cache,  // [n_kv_heads, max_seq, head_dim] f16 KV cache
        const __half * __restrict__ v_cache,  // [n_kv_heads, max_seq, head_dim] f16 KV cache
        float        * __restrict__ out,      // [S, FA_Q_SIZE]
        const int S,
        const int start_pos,    // first position of this batch in the sequence
        const int max_seq_len) {

    int idx = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    if (idx >= S * FA_N_Q_HEADS) return;

    int token_in_batch = idx / FA_N_Q_HEADS;  // 0..S-1
    int abs_pos = start_pos + token_in_batch;  // absolute position in sequence
    int qh  = idx % FA_N_Q_HEADS;
    int kvh = qh / FA_GQA_RATIO;

#if defined(HAS_CUSTOM_ATTN_SCALE) && HAS_CUSTOM_ATTN_SCALE
    float scale = FA_ATTENTION_SCALE;
#else
    float scale = 1.0f / sqrtf((float)FA_HEAD_DIM);
#endif
    constexpr int EPL = FA_HEAD_DIM / 32;

    const float * qv = q + token_in_batch * FA_QPROJ_SIZE + qh * (FA_HAS_GATED_ATTN ? FA_HEAD_DIM * 2 : FA_HEAD_DIM);
    float * ov = out + token_in_batch * FA_Q_SIZE + qh * FA_HEAD_DIM;

    float ql[EPL];
    for (int e = 0; e < EPL; e++) ql[e] = qv[lid * EPL + e];

    float oa[EPL] = {};
    float mx = -1e30f, se = 0;

    // Iterate over ALL K/V positions 0..abs_pos (causal mask: attend to self and earlier)
    // K/V are in the KV cache as f16, layout: [kv_head, max_seq, head_dim]
    const __half * k_head = k_cache + kvh * max_seq_len * FA_HEAD_DIM;
    const __half * v_head = v_cache + kvh * max_seq_len * FA_HEAD_DIM;

    for (int kp = 0; kp <= abs_pos; kp++) {
        const __half * kv = k_head + kp * FA_HEAD_DIM;
        const __half * vv = v_head + kp * FA_HEAD_DIM;

        float sc = 0;
        for (int e = 0; e < EPL; e++)
            sc += ql[e] * __half2float(kv[lid * EPL + e]);
        sc = warp_reduce_sum(sc) * scale;
        sc = __shfl_sync(0xffffffff, sc, 0, 32);

#if defined(HAS_ATTN_SOFTCAP) && HAS_ATTN_SOFTCAP
        sc = tanhf(sc / ATTN_SOFTCAP_VAL) * ATTN_SOFTCAP_VAL;
#endif

        float om = mx;
        mx = fmaxf(mx, sc);
        float ed = expf(om - mx);
        se = se * ed + expf(sc - mx);
        float wt = expf(sc - mx);
        for (int e = 0; e < EPL; e++)
            oa[e] = oa[e] * ed + wt * __half2float(vv[lid * EPL + e]);
    }

    float rs = 1.0f / se;
    for (int e = 0; e < EPL; e++) {
        int i = lid * EPL + e;
#if FA_HAS_GATED_ATTN
        float gate = op_sigmoid(qv[FA_HEAD_DIM + i]);
        ov[i] = oa[e] * rs * gate;
#else
        ov[i] = oa[e] * rs;
#endif
    }
}

// =============================================================================
// Bidirectional attention (encoder) — same as prompt_causal_attn but ALL
// positions attend to ALL positions (no causal mask).
// Used by: T5 encoder, BERT, embedding models.
// Ported from prompt_causal_attn above with kp <= pos → kp < S.
// =============================================================================
extern "C" __global__ void prompt_bidirectional_attn(
        const float * __restrict__ q,     // [S, FA_QPROJ_SIZE]
        const float * __restrict__ k,     // [S, FA_KV_SIZE]
        const float * __restrict__ v,     // [S, FA_KV_SIZE]
        float       * __restrict__ out,   // [S, FA_Q_SIZE]
        const int S) {

    int idx = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    if (idx >= S * FA_N_Q_HEADS) return;

    int pos = idx / FA_N_Q_HEADS;
    int qh  = idx % FA_N_Q_HEADS;
    int kvh = qh / FA_GQA_RATIO;

#if defined(HAS_CUSTOM_ATTN_SCALE) && HAS_CUSTOM_ATTN_SCALE
    float scale = FA_ATTENTION_SCALE;
#else
    float scale = 1.0f / sqrtf((float)FA_HEAD_DIM);
#endif
    constexpr int EPL = FA_HEAD_DIM / 32;

    const float * qv = q + pos * FA_QPROJ_SIZE + qh * (FA_HAS_GATED_ATTN ? FA_HEAD_DIM * 2 : FA_HEAD_DIM);
    float * ov = out + pos * FA_Q_SIZE + qh * FA_HEAD_DIM;

    float ql[EPL];
    for (int e = 0; e < EPL; e++) ql[e] = qv[lid * EPL + e];

    float oa[EPL] = {};
    float mx = -1e30f, se = 0;

    // Bidirectional: attend to ALL positions (no causal mask)
    for (int kp = 0; kp < S; kp++) {
        const float * kv = k + kp * FA_KV_SIZE + kvh * FA_HEAD_DIM;
        const float * vv = v + kp * FA_KV_SIZE + kvh * FA_HEAD_DIM;

        float sc = 0;
        for (int e = 0; e < EPL; e++) sc += ql[e] * kv[lid * EPL + e];
        sc = warp_reduce_sum(sc) * scale;
        sc = __shfl_sync(0xffffffff, sc, 0, 32);

#if defined(HAS_ATTN_SOFTCAP) && HAS_ATTN_SOFTCAP
        sc = tanhf(sc / ATTN_SOFTCAP_VAL) * ATTN_SOFTCAP_VAL;
#endif

        float om = mx;
        mx = fmaxf(mx, sc);
        float ed = expf(om - mx);
        se = se * ed + expf(sc - mx);
        float wt = expf(sc - mx);
        for (int e = 0; e < EPL; e++)
            oa[e] = oa[e] * ed + wt * vv[lid * EPL + e];
    }

    float rs = 1.0f / se;
    for (int e = 0; e < EPL; e++) {
        int i = lid * EPL + e;
        ov[i] = oa[e] * rs;
    }
}

// =============================================================================
// DeltaNet recurrence (batched: processes all tokens for one head)
// One block per head. State-in-registers across token loop.
// Ported from gated_delta_net_cuda (GDA path).
// =============================================================================

#if DN_N_HEADS > 0
extern "C" __global__ void __launch_bounds__(512, 1)
prompt_deltanet_recurrence(
        const float  * __restrict__ qkv_proj,    // [S, DN_CONV_CH]
        const float  * __restrict__ z_proj,      // [S, DN_V_SIZE]
        const float  * __restrict__ beta_proj,   // [S, DN_N_HEADS]
        const float  * __restrict__ alpha_proj,  // [S, DN_N_HEADS]
        const float  * __restrict__ conv_w,      // [DN_CONV_CH * DN_CONV_KERNEL]
        const float  * __restrict__ ssm_a,       // [DN_N_HEADS]
        const float  * __restrict__ dt_bias,     // [DN_N_HEADS]
        const float  * __restrict__ norm_w,      // [DN_VALUE_DIM]
        float        * __restrict__ state,       // [DN_N_HEADS, DN_KEY_DIM, DN_VALUE_DIM]
        float        * __restrict__ conv_buf,    // [DN_CONV_CH * DN_CONV_KERNEL]
        float        * __restrict__ output,      // [S, DN_V_SIZE]
        const int S) {

    const int h = blockIdx.x;
    if (h >= DN_N_HEADS) return;
    const int kh = h % DN_N_K_HEADS;
    const int tid = threadIdx.x;
    const int wid = tid / 32, lid = tid % 32;
    constexpr int NWARPS = 512 / 32;

    const float a_val = ssm_a[h];
    const float dt_b  = dt_bias[h];

    __shared__ float s_q[DN_KEY_DIM], s_k[DN_KEY_DIM], s_v[DN_VALUE_DIM];
    __shared__ float s_beta, s_decay;
    __shared__ float s_gnorm[NWARPS];

    float * my_state = state + h * DN_KEY_DIM * DN_VALUE_DIM;

    // Channel offsets
    const int q_off = kh * DN_KEY_DIM;
    const int k_off = DN_QK_SIZE + kh * DN_KEY_DIM;
    const int v_off = 2 * DN_QK_SIZE + h * DN_VALUE_DIM;
    const bool is_first = (h < DN_N_K_HEADS);

    // Load state into registers
    constexpr int CPW = DN_VALUE_DIM / NWARPS;
    constexpr int RPL = DN_KEY_DIM / 32;
    float sreg[CPW * RPL];

    for (int jj = 0; jj < CPW; jj++) {
        int j = wid * CPW + jj;
        for (int ii = 0; ii < RPL; ii++)
            sreg[jj * RPL + ii] = my_state[j * DN_KEY_DIM + lid + ii * 32];
    }

    for (int t = 0; t < S; t++) {
        // Conv1d + SiLU for Q
        for (int c = tid; c < DN_KEY_DIM; c += 512) {
            int ch = q_off + c;
            int buf = ch * DN_CONV_KERNEL;
            if (is_first) {
                for (int tt = 0; tt < DN_CONV_KERNEL - 1; tt++)
                    conv_buf[buf + tt] = conv_buf[buf + tt + 1];
                conv_buf[buf + DN_CONV_KERNEL - 1] = qkv_proj[t * DN_CONV_CH + ch];
            }
            float co = 0;
            for (int tt = 0; tt < DN_CONV_KERNEL; tt++)
                co += conv_buf[buf + tt] * conv_w[ch * DN_CONV_KERNEL + tt];
            s_q[c] = op_silu(co);
        }
        // Conv1d + SiLU for K
        for (int c = tid; c < DN_KEY_DIM; c += 512) {
            int ch = k_off + c;
            int buf = ch * DN_CONV_KERNEL;
            if (is_first) {
                for (int tt = 0; tt < DN_CONV_KERNEL - 1; tt++)
                    conv_buf[buf + tt] = conv_buf[buf + tt + 1];
                conv_buf[buf + DN_CONV_KERNEL - 1] = qkv_proj[t * DN_CONV_CH + ch];
            }
            float co = 0;
            for (int tt = 0; tt < DN_CONV_KERNEL; tt++)
                co += conv_buf[buf + tt] * conv_w[ch * DN_CONV_KERNEL + tt];
            s_k[c] = op_silu(co);
        }
        // Conv1d + SiLU for V
        for (int c = tid; c < DN_VALUE_DIM; c += 512) {
            int ch = v_off + c;
            int buf = ch * DN_CONV_KERNEL;
            for (int tt = 0; tt < DN_CONV_KERNEL - 1; tt++)
                conv_buf[buf + tt] = conv_buf[buf + tt + 1];
            conv_buf[buf + DN_CONV_KERNEL - 1] = qkv_proj[t * DN_CONV_CH + ch];
            float co = 0;
            for (int tt = 0; tt < DN_CONV_KERNEL; tt++)
                co += conv_buf[buf + tt] * conv_w[ch * DN_CONV_KERNEL + tt];
            s_v[c] = op_silu(co);
        }
        __syncthreads();

        // L2 normalize Q, K
        if (wid == 0) {
            float sq = 0;
            for (int i = lid; i < DN_KEY_DIM; i += 32) sq += s_q[i] * s_q[i];
            sq = warp_reduce_sum(sq);
            float n = rsqrtf(fmaxf(sq, NORM_EPS * NORM_EPS));
            n = __shfl_sync(0xffffffff, n, 0, 32);
            for (int i = lid; i < DN_KEY_DIM; i += 32) s_q[i] *= n;
        }
        if (wid == 1) {
            float sq = 0;
            for (int i = lid; i < DN_KEY_DIM; i += 32) sq += s_k[i] * s_k[i];
            sq = warp_reduce_sum(sq);
            float n = rsqrtf(fmaxf(sq, NORM_EPS * NORM_EPS));
            n = __shfl_sync(0xffffffff, n, 0, 32);
            for (int i = lid; i < DN_KEY_DIM; i += 32) s_k[i] *= n;
        }
        __syncthreads();

        // Beta / gate
        if (tid == 0) {
            s_beta = op_sigmoid(beta_proj[t * DN_N_HEADS + h]);
            float x = alpha_proj[t * DN_N_HEADS + h] + dt_b;
            float sp = op_softplus(x);
            s_decay = expf(sp * a_val);
        }
        __syncthreads();
        float beta = s_beta, decay = s_decay;
        float scale = rsqrtf((float)DN_VALUE_DIM);

        float * out_h = output + t * DN_V_SIZE + h * DN_VALUE_DIM;

        // State-in-registers recurrence (matching gated_delta_net_cuda GDA)
        for (int jj = 0; jj < CPW; jj++) {
            int j = wid * CPW + jj;
            float kv = 0;
            for (int ii = 0; ii < RPL; ii++)
                kv += sreg[jj * RPL + ii] * s_k[lid + ii * 32];
            kv = warp_reduce_sum(kv);
            kv = __shfl_sync(0xffffffff, kv, 0, 32);

            float delta = (s_v[j] - decay * kv) * beta;

            float attn = 0;
            for (int ii = 0; ii < RPL; ii++) {
                sreg[jj * RPL + ii] = decay * sreg[jj * RPL + ii] + s_k[lid + ii * 32] * delta;
                attn += sreg[jj * RPL + ii] * s_q[lid + ii * 32];
            }
            attn = warp_reduce_sum(attn);
            if (lid == 0) out_h[j] = attn * scale;
        }
        __syncthreads();

        // Gated RMSNorm → output
        const float * z_h = z_proj + t * DN_V_SIZE + h * DN_VALUE_DIM;
        float sq2 = 0;
        for (int i = tid; i < DN_VALUE_DIM; i += 512) sq2 += out_h[i] * out_h[i];
        sq2 = warp_reduce_sum(sq2);
        if (lid == 0) s_gnorm[wid] = sq2;
        __syncthreads();
        if (wid == 0) {
            float v = (lid < NWARPS) ? s_gnorm[lid] : 0;
            v = warp_reduce_sum(v);
            if (lid == 0) s_gnorm[0] = rsqrtf(v / (float)DN_VALUE_DIM + NORM_EPS);
        }
        __syncthreads();
        float rstd = s_gnorm[0];
        for (int i = tid; i < DN_VALUE_DIM; i += 512) {
            float normed = out_h[i] * rstd * norm_w[i];
            out_h[i] = normed * op_silu(z_h[i]);
        }
        __syncthreads();
    }

    // Write state back
    for (int jj = 0; jj < CPW; jj++) {
        int j = wid * CPW + jj;
        for (int ii = 0; ii < RPL; ii++)
            my_state[j * DN_KEY_DIM + lid + ii * 32] = sreg[jj * RPL + ii];
    }
}
#endif // DN_N_HEADS > 0

// =============================================================================
// Final norm (last token only → logits)
// =============================================================================

extern "C" __global__ void prompt_final_norm(
        const float * __restrict__ hidden,   // [S, HIDDEN_SIZE]
        const float * __restrict__ weight,   // [HIDDEN_SIZE]
        float       * __restrict__ normed,   // [HIDDEN_SIZE] (last token only)
        const int S) {
    int tid = threadIdx.x, wid = tid / 32, lid = tid % 32;
    __shared__ float smem[32];

    const float * row = hidden + (S - 1) * HIDDEN_SIZE;
    float sq = 0;
    for (int i = tid; i < HIDDEN_SIZE; i += blockDim.x) {
        float v = row[i];
        sq += v * v;
    }
    sq = warp_reduce_sum(sq);
    if (lid == 0) smem[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        float v = (lid < (int)(blockDim.x / 32)) ? smem[lid] : 0;
        v = warp_reduce_sum(v);
        if (lid == 0) smem[0] = rsqrtf(v / (float)HIDDEN_SIZE + NORM_EPS);
    }
    __syncthreads();
    float rstd = smem[0];
    for (int i = tid; i < HIDDEN_SIZE; i += blockDim.x) {
        float w = weight[i];
#if NORM_ADD_ONE
        w = 1.0f + w;
#endif
        normed[i] = row[i] * rstd * w;
    }
}

// =============================================================================
// LM head: normed → logits (Q4_K weights × f32 hidden, warp-per-row)
// =============================================================================

extern "C" __global__ void prompt_lm_head(
        const float * __restrict__ normed,     // [HIDDEN_SIZE]
        const void  * __restrict__ weight,     // [VOCAB_SIZE, HIDDEN_SIZE] Q4_K
        const long long             stride,
        float       * __restrict__ logits,     // [VOCAB_SIZE]
        float       * __restrict__ block_maxv, // [gridDim.x] scratch for argmax
        int         * __restrict__ block_maxi) {

    // Each block does a slice of vocab rows. Each warp does one row.
    int wid = threadIdx.x / 32, lid = threadIdx.x % 32;
    int nwarps = blockDim.x / 32;
    int rpb = (VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int rs = blockIdx.x * rpb, re = min(rs + rpb, VOCAB_SIZE);

    // Load normed hidden into shared (f32)
    __shared__ float s_hidden[HIDDEN_SIZE];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += blockDim.x) s_hidden[i] = normed[i];
    __syncthreads();

    // Quantize to Q8_1 in shared memory
    __shared__ block_q8_1 s_q8[HIDDEN_SIZE / QK8_1];
    {
        int n_q8 = HIDDEN_SIZE / QK8_1;
        for (int ib = threadIdx.x / QK8_1; ib < n_q8; ib += blockDim.x / QK8_1) {
            int iqs = threadIdx.x % QK8_1;
            float val = s_hidden[ib * QK8_1 + iqs];
            float amax = fabsf(val), sum = val;
            amax = warp_reduce_max<QK8_1>(amax);
            sum  = warp_reduce_sum<QK8_1>(sum);
            float d = amax / 127.0f;
            s_q8[ib].qs[iqs] = (amax == 0.0f) ? 0 : (int8_t)roundf(val / d);
            if (iqs == 0) s_q8[ib].ds = make_half2(d, sum);
        }
    }
    __syncthreads();

    float lm = -1e30f;
    int li = -1;
    for (int m = rs + wid; m < re; m += nwarps) {
        const void * w_row = (const char *)weight + (long long)m * stride;
        float sum = 0;
        for (int iqs = lid; iqs < HIDDEN_SIZE / QK8_1; iqs += 32) {
            sum += vec_dot_q4_K_q8_1(w_row, s_q8, iqs / QI4_K, iqs % QI4_K);
        }
        sum = warp_reduce_sum(sum);
        if (lid == 0) {
            logits[m] = sum;
            if (sum > lm) { lm = sum; li = m; }
        }
    }

    // Per-block argmax reduce
    lm = __shfl_sync(0xffffffff, lm, 0, 32);
    li = __shfl_sync(0xffffffff, li, 0, 32);
    __shared__ float wm[32];
    __shared__ int wi[32];
    if (lid == 0) { wm[wid] = lm; wi[wid] = li; }
    __syncthreads();
    if (wid == 0) {
        float mv = (lid < nwarps) ? wm[lid] : -1e30f;
        int mi = (lid < nwarps) ? wi[lid] : -1;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_down_sync(0xffffffffull, mv, o);
            int oi = __shfl_down_sync(0xffffffffull, mi, o);
            if (ov > mv) { mv = ov; mi = oi; }
        }
        if (lid == 0) { block_maxv[blockIdx.x] = mv; block_maxi[blockIdx.x] = mi; }
    }
}

// Final reduce across blocks
extern "C" __global__ void prompt_lm_reduce(
        const float * __restrict__ bmv,
        const int   * __restrict__ bmi,
        int         * __restrict__ output_token,
        const int nb) {
    int tid = threadIdx.x;
    float best = -1e30f;
    int bi = -1;
    for (int i = tid; i < nb; i += blockDim.x) {
        float v = bmv[i];
        if (v > best) { best = v; bi = bmi[i]; }
    }
    __shared__ float sv[256];
    __shared__ int si[256];
    sv[tid] = best;
    si[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) *output_token = si[0];
}

// =============================================================================
// MMQ kernel template instantiations — RDNA3/AMD_WMMA path
// mmq.cuh conventional tiling kernel: mul_mat_q<type, mmq_x, need_check>
// Instantiate mmq_x=32 and mmq_x=64 for each type (both need_check variants).
//
// Each instantiation gets an extern "C" wrapper so the host can load it via
// hipModuleGetFunction with a predictable unmangled symbol name.
// Naming: prompt_mmq_<NAME>_<mmq_x>[_check]
// =============================================================================

#define MMQ_KERNEL_PARAMS \
    const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst, \
    float * __restrict__ dst, float * __restrict__ tmp_fixup, \
    const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst, \
    const int channel_ratio, const int nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst, \
    const int sample_ratio, const int nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst, \
    const int ncols_max

#define MMQ_KERNEL_ARGS \
    x, y, ids_dst, dst, tmp_fixup, \
    ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, \
    channel_ratio, nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst, \
    sample_ratio, nsamples_y, stride_sample_x, stride_sample_y, stride_sample_dst, \
    ncols_max

#define MMQ_WRAP(NAME, TYPE, MMQ_X, NC) \
    extern "C" __global__ __launch_bounds__(256, 2) \
    void prompt_mmq_##NAME##_##MMQ_X##NC(MMQ_KERNEL_PARAMS) { \
        mul_mat_q<TYPE, MMQ_X, (sizeof(#NC) > 1)>(MMQ_KERNEL_ARGS); \
    }

// Emit 4 wrappers per type: {32,64} x {nocheck, check}
#define MMQ_INST(NAME, TYPE) \
    MMQ_WRAP(NAME, TYPE, 32,      ) \
    MMQ_WRAP(NAME, TYPE, 32, _check) \
    MMQ_WRAP(NAME, TYPE, 64,      ) \
    MMQ_WRAP(NAME, TYPE, 64, _check)

MMQ_INST(q4_0,    GGML_TYPE_Q4_0)
MMQ_INST(q4_1,    GGML_TYPE_Q4_1)
MMQ_INST(q5_0,    GGML_TYPE_Q5_0)
MMQ_INST(q5_1,    GGML_TYPE_Q5_1)
MMQ_INST(q8_0,    GGML_TYPE_Q8_0)
MMQ_INST(mxfp4,   GGML_TYPE_MXFP4)
MMQ_INST(nvfp4,   GGML_TYPE_NVFP4)
MMQ_INST(q2k,     GGML_TYPE_Q2_K)
MMQ_INST(q3k,     GGML_TYPE_Q3_K)
MMQ_INST(q4k,     GGML_TYPE_Q4_K)
MMQ_INST(q5k,     GGML_TYPE_Q5_K)
MMQ_INST(q6k,     GGML_TYPE_Q6_K)
MMQ_INST(iq2_xxs, GGML_TYPE_IQ2_XXS)
MMQ_INST(iq2_xs,  GGML_TYPE_IQ2_XS)
MMQ_INST(iq2_s,   GGML_TYPE_IQ2_S)
MMQ_INST(iq3_xxs, GGML_TYPE_IQ3_XXS)
MMQ_INST(iq3_s,   GGML_TYPE_IQ3_S)
MMQ_INST(iq1_s,   GGML_TYPE_IQ1_S)
MMQ_INST(iq4_nl,  GGML_TYPE_IQ4_NL)
MMQ_INST(iq4_xs,  GGML_TYPE_IQ4_XS)

#undef MMQ_KERNEL_PARAMS
#undef MMQ_KERNEL_ARGS
#undef MMQ_WRAP
#undef MMQ_INST

// =============================================================================
// F32→F16 conversion kernel for rocBLAS GEMM input path
// Pattern from rocblas-gemm.h comments (baseline convert.cu)
// =============================================================================

extern "C" __global__ void gemm_f32_to_f16(
        const float * __restrict__ x, __half * __restrict__ y, int64_t k) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) y[i] = __float2half(x[i]);
}

// =============================================================================
// Weight dequantize-to-F16 kernels for rocBLAS GEMM weight path
// Ported from baseline convert.cu — same math, __half output instead of float.
//
// Signature: (const void * vx, __half * y, int64_t k)
//   k = total number of output elements (out_dim * in_dim)
//   blockIdx.x = superblock index
//   Threads per block: 32 for small-block types, 64 for K-quants
//
// Grid: k / QK_K for K-quant types, (k + 255) / 256 for small-block types
//
// Helper from convert.cu:194-201 (used by Q4_K, Q5_K dequant kernels below)
// =============================================================================

static __device__ __forceinline__ void dequant_get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// =============================================================================
// These match baseline convert.cu dequantize_row_*_cuda launch patterns.
// =============================================================================

// --- Q4_0: 32 threads, grid = (k+255)/256 ---
extern "C" __global__ void prompt_dequant_q4_0_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const int64_t i = blockIdx.x;
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= k/32) return;
    __half * y = yy + 256*i + 32*ir + 4*il;
    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;
    const uint8_t * q = x->qs + 4*il;
    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = __float2half(d * (q[l] & 0xF) + dm);
        y[l+16] = __float2half(d * (q[l] >>  4) + dm);
    }
}

// --- Q4_1: 32 threads, grid = (k+255)/256 ---
extern "C" __global__ void prompt_dequant_q4_1_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const int64_t i = blockIdx.x;
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= k/32) return;
    __half * y = yy + 256*i + 32*ir + 4*il;
    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);
    const uint8_t * q = x->qs + 4*il;
    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = __float2half(d.x * (q[l] & 0xF) + d.y);
        y[l+16] = __float2half(d.x * (q[l] >>  4) + d.y);
    }
}

// --- Q5_0: 32 threads, grid = (k+255)/256 ---
extern "C" __global__ void prompt_dequant_q5_0_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const int64_t i = blockIdx.x;
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= k/32) return;
    __half * y = yy + 256*i + 32*ir + 4*il;
    const block_q5_0 * x = (const block_q5_0 *)vx + ib;
    const float d = __half2float(x->d);
    uint32_t qh;
    memcpy(&qh, x->qh, sizeof(qh));
    for (int l = 0; l < 4; ++l) {
        const int iqs = 4*il + l;
        const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;
        y[l +  0] = __float2half(((x->qs[iqs] & 0xf) | xh_0) * d - 16.0f * d);
        y[l + 16] = __float2half(((x->qs[iqs] >>   4) | xh_1) * d - 16.0f * d);
    }
}

// --- Q5_1: 32 threads, grid = (k+255)/256 ---
extern "C" __global__ void prompt_dequant_q5_1_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const int64_t i = blockIdx.x;
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= k/32) return;
    __half * y = yy + 256*i + 32*ir + 4*il;
    const block_q5_1 * x = (const block_q5_1 *)vx + ib;
    const float2 dm = __half22float2(x->dm);
    uint32_t qh;
    memcpy(&qh, x->qh, sizeof(qh));
    for (int l = 0; l < 4; ++l) {
        const int iqs = 4*il + l;
        const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;
        y[l +  0] = __float2half(((x->qs[iqs] & 0xf) | xh_0) * dm.x + dm.y);
        y[l + 16] = __float2half(((x->qs[iqs] >>   4) | xh_1) * dm.x + dm.y);
    }
}

// --- Q8_0: ported from baseline convert.cu dequantize_block_q8_0_f16 ---
// Grid: (k + NE_ALIGN - 1) / NE_ALIGN, Block: 32 (WARP_SIZE)
// Uses shared memory to load raw block data, then dequant to half2.
#define CUDA_Q8_0_NE_ALIGN 2048
extern "C" __global__ void prompt_dequant_q8_0_f16(const void * __restrict__ vx, __half * __restrict__ y, int64_t k) {
    constexpr int nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;
    const int * x0 = ((const int *) vx) + blockIdx.x * nint;
    __half2 * y2 = (__half2 *) (y + i0);

    __shared__ int vals[nint];

#pragma unroll
    for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
        if (i0*sizeof(block_q8_0)/QK8_0 + sizeof(int)*(ix0 + threadIdx.x) >= (uint64_t)k*sizeof(block_q8_0)/QK8_0) {
            break;
        }
        const int ix = ix0 + threadIdx.x;
        vals[ix] = x0[ix];
    }

    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2*WARP_SIZE) {
        if (i0 + iy + 2*threadIdx.x >= k) {
            return;
        }

        const __half * b0 = ((const __half *) vals) + (sizeof(block_q8_0)/sizeof(__half)) * ((iy + 2*threadIdx.x)/QK8_0);
        const __half    d = *b0;
        const char2  qs = ((const char2 *) (b0 + 1))[threadIdx.x % (QK8_0/2)];

        y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __halves2half2(d, d));
    }
}
#undef CUDA_Q8_0_NE_ALIGN

// --- Q2_K: 64 threads, grid = k/QK_K ---
extern "C" __global__ void prompt_dequant_q2k_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const int64_t i = blockIdx.x;
    const block_q2_K * x = (const block_q2_K *)vx;
    const int64_t tid = threadIdx.x;
    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;
    const uint8_t q = x[i].qs[32*n + l];
    __half * y = yy + i*QK_K + 128*n;
    float dall = __low2float(x[i].dm);
    float dmin = __high2float(x[i].dm);
    y[l+ 0] = __float2half(dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4));
    y[l+32] = __float2half(dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4));
    y[l+64] = __float2half(dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4));
    y[l+96] = __float2half(dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4));
}

// --- Q3_K: 64 threads, grid = k/QK_K ---
extern "C" __global__ void prompt_dequant_q3k_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const int64_t i = blockIdx.x;
    const block_q3_K * x = (const block_q3_K *)vx;
    const int64_t r = threadIdx.x/4;
    const int64_t tid = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(threadIdx.x%4);
    const int64_t n = tid / 4;
    const int64_t j = tid - 4*n;
    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;
    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    float d_all = __half2float(x[i].d);
    float dl = d_all * (us - 32);
    __half * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;
    for (int l = l0; l < l0+4; ++l) y[l] = __float2half(dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
}

// --- Q4_K: 32 threads, grid = k/QK_K ---
extern "C" __global__ void prompt_dequant_q4k_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const block_q4_K * x = (const block_q4_K *)vx;
    const int64_t i = blockIdx.x;
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;
    __half * y = yy + i*QK_K + 64*il + n*ir;
    const float dall = __low2float(x[i].dm);
    const float dmin = __high2float(x[i].dm);
    const uint8_t * q = x[i].qs + 32*il + n*ir;
    uint8_t sc, m2;
    dequant_get_scale_min_k4(is + 0, x[i].scales, sc, m2);
    const float d1 = dall * sc; const float m1 = dmin * m2;
    dequant_get_scale_min_k4(is + 1, x[i].scales, sc, m2);
    const float d2 = dall * sc; const float mm2 = dmin * m2;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = __float2half(d1 * (q[l] & 0xF) - m1);
        y[l +32] = __float2half(d2 * (q[l] >>  4) - mm2);
    }
}

// --- Q5_K: 64 threads, grid = k/QK_K ---
extern "C" __global__ void prompt_dequant_q5k_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const block_q5_K * x = (const block_q5_K *)vx;
    const int64_t i = blockIdx.x;
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/16;
    const int64_t ir  = tid%16;
    const int64_t is  = 2*il;
    __half * y = yy + i*QK_K + 64*il + 2*ir;
    const float dall = __low2float(x[i].dm);
    const float dmin = __high2float(x[i].dm);
    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;
    uint8_t sc, m2;
    dequant_get_scale_min_k4(is + 0, x[i].scales, sc, m2);
    const float d1 = dall * sc; const float m1 = dmin * m2;
    dequant_get_scale_min_k4(is + 1, x[i].scales, sc, m2);
    const float d2 = dall * sc; const float mm2 = dmin * m2;
    uint8_t hm = 1 << (2*il);
    y[ 0] = __float2half(d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1);
    y[ 1] = __float2half(d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[32] = __float2half(d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - mm2);
    y[33] = __float2half(d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - mm2);
}

// --- Q6_K: 64 threads, grid = k/QK_K ---
extern "C" __global__ void prompt_dequant_q6k_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const block_q6_K * x = (const block_q6_K *)vx;
    const int64_t i = blockIdx.x;
    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/32;
    const int64_t il  = tid - 32*ip;
    const int64_t is  = 8*ip + il/16;
    __half * y = yy + i*QK_K + 128*ip + il;
    const float d = __half2float(x[i].d);
    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;
    y[ 0] = __float2half(d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    y[32] = __float2half(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    y[64] = __float2half(d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    y[96] = __float2half(d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32));
}

// --- IQ2_XXS: 32 threads, grid = k/QK_K ---
extern "C" __global__ void prompt_dequant_iq2_xxs_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const int64_t i = blockIdx.x;
    const block_iq2_xxs * x = (const block_iq2_xxs *)vx;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8;
    const int64_t ib = tid%8;
    __half * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t * aux8 = (const uint8_t *)q2;
    const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = __half2float(x[i].d) * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = __float2half(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}

// --- IQ2_XS: 32 threads, grid = k/QK_K ---
extern "C" __global__ void prompt_dequant_iq2_xs_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const int64_t i = blockIdx.x;
    const block_iq2_xs * x = (const block_iq2_xs *)vx;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8;
    const int64_t ib = tid%8;
    __half * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d = __half2float(x[i].d) * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = __float2half(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}

// --- IQ2_S: 32 threads, grid = k/QK_K ---
extern "C" __global__ void prompt_dequant_iq2_s_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const int64_t i = blockIdx.x;
    const block_iq2_s * x = (const block_iq2_s *)vx;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8;
    const int64_t ib = tid%8;
    __half * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    const float d = __half2float(x[i].d) * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
    for (int j = 0; j < 8; ++j) y[j] = __float2half(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}

// --- IQ3_XXS: 32 threads, grid = k/QK_K ---
extern "C" __global__ void prompt_dequant_iq3_xxs_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const int64_t i = blockIdx.x;
    const block_iq3_xxs * x = (const block_iq3_xxs *)vx;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8;
    const int64_t ib = tid%8;
    __half * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * q3 = x[i].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = __half2float(x[i].d) * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = __float2half(d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f));
        y[j+4] = __float2half(d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f));
    }
}

// --- IQ3_S: 32 threads, grid = k/QK_K ---
extern "C" __global__ void prompt_dequant_iq3_s_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const int64_t i = blockIdx.x;
    const block_iq3_s * x = (const block_iq3_s *)vx;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8;
    const int64_t ib = tid%8;
    __half * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * qs = x[i].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    const float d = __half2float(x[i].d) * (1 + 2*((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[i].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = __float2half(d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f));
        y[j+4] = __float2half(d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f));
    }
}

// --- IQ1_S: 32 threads, grid = k/QK_K ---
extern "C" __global__ void prompt_dequant_iq1_s_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const int64_t i = blockIdx.x;
    const block_iq1_s * x = (const block_iq1_s *)vx;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8;
    const int64_t ib = tid%8;
    __half * y = yy + i*QK_K + 32*ib + 8*il;
    const float delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = __half2float(x[i].d) * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) y[j] = __float2half(d * (q[j] + delta));
}

// --- IQ1_M: 32 threads, grid = k/QK_K ---
extern "C" __global__ void prompt_dequant_iq1_m_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const int64_t i = blockIdx.x;
    const block_iq1_m * x = (const block_iq1_m *)vx;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8;
    const int64_t ib = tid%8;
    __half * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * sc = (const uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2*ib + il/2;
    const float d = __half2float(scale.f16) * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    const float delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) y[j] = __float2half(d * (q[j] + delta));
}

// --- IQ4_NL: 32 threads, grid = (k + QK_K - 1) / QK_K ---
extern "C" __global__ void prompt_dequant_iq4_nl_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const int64_t i = blockIdx.x;
    const block_iq4_nl * x = (const block_iq4_nl *)vx + i*(QK_K/QK4_NL);
    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8;
    const int64_t ib = tid%8;
    __half * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t * q4 = x[ib].qs + 4*il;
    const float d = __half2float(x[ib].d);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = __float2half(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j+16] = __float2half(d * kvalues_iq4nl[q4[j] >>  4]);
    }
}

// --- IQ4_XS: 32 threads, grid = (k + QK_K - 1) / QK_K ---
extern "C" __global__ void prompt_dequant_iq4_xs_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const int64_t i = blockIdx.x;
    const block_iq4_xs * x = (const block_iq4_xs *)vx;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8;
    const int64_t ib = tid%8;
    __half * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t * q4 = x[i].qs + 16*ib + 4*il;
    const float d = __half2float(x[i].d) * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = __float2half(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j+16] = __float2half(d * kvalues_iq4nl[q4[j] >>  4]);
    }
}

// --- MXFP4: 32 threads, grid = (k + QK_K - 1) / QK_K ---
extern "C" __global__ void prompt_dequant_mxfp4_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const int64_t i = blockIdx.x;
    const block_mxfp4 * x = (const block_mxfp4 *)vx + i*(QK_K/QK_MXFP4);
    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8;
    const int64_t ib = tid%8;
    __half * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t * q4 = x[ib].qs + 4*il;
    const float d = ggml_cuda_e8m0_to_fp32(x[ib].e);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = __float2half(d * kvalues_mxfp4[q4[j] & 0xf] * 0.5f);
        y[j+16] = __float2half(d * kvalues_mxfp4[q4[j] >>  4] * 0.5f);
    }
}

// --- NVFP4: 32 threads, grid = k / QK_NVFP4 ---
extern "C" __global__ void prompt_dequant_nvfp4_f16(const void * __restrict__ vx, __half * __restrict__ yy, int64_t k) {
    const int64_t i = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t base = i * QK_NVFP4;
    if (base >= k) return;
    const block_nvfp4 * x = (const block_nvfp4 *)vx;
    const block_nvfp4 & xb = x[i];
    const int sub = tid / (QK_NVFP4_SUB / 2);
    const int j = tid % (QK_NVFP4_SUB / 2);
    const float d = ggml_cuda_ue4m3_to_fp32(xb.d[sub]);
    const uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];
    const int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    const int64_t y1 = y0 + QK_NVFP4_SUB / 2;
    yy[y0] = __float2half(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = __float2half(d * kvalues_mxfp4[q >> 4]);
}

// --- F32→F16: 256 threads, grid = (k + 255) / 256 ---
extern "C" __global__ void prompt_dequant_f32_f16(const void * __restrict__ vx, __half * __restrict__ y, int64_t k) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) y[i] = __float2half(((const float *)vx)[i]);
}

// --- BF16→F16: 256 threads, grid = (k + 255) / 256 ---
extern "C" __global__ void prompt_dequant_bf16_f16(const void * __restrict__ vx, __half * __restrict__ y, int64_t k) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) {
        uint32_t bits = (uint32_t)((const uint16_t *)vx)[i] << 16;
        float val;
        __builtin_memcpy(&val, &bits, sizeof(float));
        y[i] = __float2half(val);
    }
}
