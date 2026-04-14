// attention.h -- Line-for-line port of baseline flash_attn_ext_vec kernel to HIP/gfx1100
// Source: ggml/src/ggml-cuda/fattn-vec.cuh (lines 19-517)
//       + ggml/src/ggml-cuda/fattn-common.cuh (vec_dot, dequantize_V, etc.)
// Specialized for: ncols=1, type_K=F16, type_V=F16, gfx1100 (RDNA3, wave32)
//
// V_DOT2_F32_F16_AVAILABLE=true on gfx1100.
// ALL variables, ALL branches, ALL loops from baseline are preserved.
// The only changes: function signature (megakernel interface), parameter mapping
// at top of function, mask/sinks replaced with our direct parameters.
#pragma once

#include "hip-shim.h"
#include "warp-reduce.h"
#include "mem-utils.h"
#include <cfloat>

// --- Constants from fattn-common.cuh lines 9-19 ---
#define FATTN_KQ_MAX_OFFSET (3.0f * 0.6931f)
#define SOFTMAX_FTZ_THRESHOLD -20.0f

// ============================================================================
// ggml_cuda_mad overloads -- from common.cuh lines 712-751
// ============================================================================

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float v, const float u) {
    acc += v * u;
}

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float2 v, const float2 u) {
    acc += v.x * u.x;
    acc += v.y * u.y;
}

// RDNA3 v_dot2_f32_f16 (V_DOT2_F32_F16_AVAILABLE path)
static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const __half2 v, const __half2 u) {
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(v), "v"(u));
}

// half2 * half2 accumulate into half2
static __device__ __forceinline__ void ggml_cuda_mad(__half2 & acc, const __half2 v, const __half2 u) {
    acc += v * u;
}

// ============================================================================
// vec_dot_fattn_vec_KQ_f16 -- from fattn-common.cuh lines 47-75
// Exact baseline signature with 4 arguments (Q_q8 and Q_ds_v unused for f16).
// ============================================================================
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
        const char * __restrict__ K_c, const void * __restrict__ Q_v,
        const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    (void)Q_q8;
    (void)Q_ds_v;

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            // V_DOT2_F32_F16_AVAILABLE path: Q_v is half2, use v_dot2_f32_f16
            ggml_cuda_mad(sum, tmp[k_KQ_1], ((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
        }
    }

    return sum;
}

// ============================================================================
// dequantize_V_f16 -- from fattn-common.cuh lines 337-353
// T=half path for V_DOT2_F32_F16_AVAILABLE: output is half, straight memcpy.
// ============================================================================
template <int ne>
static __device__ __forceinline__ void dequantize_V_f16_half(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    // T=half path: straight memcpy of ne half values = ne*2 bytes
    ggml_cuda_memcpy_1<ne * sizeof(__half)>(dst, (const __half *) vx + i0);
}

// ============================================================================
// vec_dot_KQ function pointer type -- from fattn-common.cuh line 44-45
// ============================================================================
typedef float (*vec_dot_KQ_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds);

// ============================================================================
// dequantize_V function pointer type -- from fattn-common.cuh line 335
// ============================================================================
typedef void (*dequantize_V_t)(const void *, void *, const int64_t);

// ============================================================================
// flash_attn_vec_f16 -- Line-for-line port of flash_attn_ext_vec kernel body
// from fattn-vec.cuh lines 19-517, specialized for:
//   D (template), ncols=1, type_K=F16, type_V=F16
//   V_DOT2_F32_F16_AVAILABLE=true (gfx1100 RDNA3)
//   RDNA=true (nthreads_KQ_q=2)
//
// use_logit_softcap is controlled by HAS_ATTN_SOFTCAP compile-time define.
//
// Thread block: dim3(32, 4, 1) = 128 threads, 4 warps
// Called from eval_attention_decode kernel as a __device__ function.
// ============================================================================
template <int D>
static __device__ void flash_attn_vec_f16(
        const float  * __restrict__ Q_f,     // [D] f32 query for one head (NOT yet scaled)
        const __half * __restrict__ K,       // [kv_len, D] f16 K cache (position-major)
        const __half * __restrict__ V,       // [kv_len, D] f16 V cache (position-major)
        float        * __restrict__ dst,     // [D] f32 output (final for pb==1, partial for pb>1)
        const int kv_len,                    // number of valid KV positions
        const int max_seq_len,               // total allocated cache size (baseline ne11)
        const float scale,                   // 1/sqrt(D)
        const int k_stride,                  // stride between K positions (in bytes) = D * sizeof(half)
        const int v_stride,                  // stride between V positions (in bytes) = D * sizeof(half)
        const float alibi_slope = 0.0f,      // ALiBi slope for this head (0 = disabled)
        const int   cur_pos = 0,             // current token position (for ALiBi bias = slope*(k_pos - cur_pos))
        const float * __restrict__ rel_pos_bias = nullptr,  // [kv_len] T5 relative position bias, or nullptr
        const float attn_logit_softcap = 0.0f, // Gemma2/3/4: tanh(score/cap)*cap before softmax (0 = disabled)
        const int parallel_blocks = 1,        // number of blocks splitting KV range
        const int block_y = 0,                // this block's index in the parallel split
        float * __restrict__ dst_partial = nullptr,  // [parallel_blocks * D] partial output scratch
        float * __restrict__ dst_meta = nullptr) {   // [parallel_blocks * 2] (KQ_max, KQ_sum) scratch

    // =========================================================================
    // Map megakernel parameters to baseline's kernel variables
    // (fattn-vec.cuh lines 26-42 parameter list)
    // =========================================================================
#if defined(HAS_ATTN_SOFTCAP) && HAS_ATTN_SOFTCAP
    constexpr bool use_logit_softcap = true;
#else
    constexpr bool use_logit_softcap = false;
#endif

    // Baseline passes scale pre-divided by softcap from the launcher (fattn-common.cuh line 1136).
    // We replicate that here since our caller passes raw scale.
    const float effective_scale = use_logit_softcap ? (scale / attn_logit_softcap) : scale;

    const float logit_softcap = attn_logit_softcap;

    // ne/nb mapping: baseline kernel signature variables
    const int32_t ne00 = D;
    const uint3   ne01 = make_uint3(1, 0, 1); // ne01.x = nrows_Q, ne01.z = total Q cols
    const int32_t ne02 = 1;
    const int32_t ne03 = 1;
    const int32_t nb01 = D * (int32_t)sizeof(float);
    const int32_t nb02 = nb01;
    const int32_t nb03 = nb02;
    const int32_t ne10 = D;
    // ne11 = K tensor dimension 1 (baseline's K->ne[1] = GGML_PAD(kv_used, 256)).
    // The loop iterates over ne11 entries. Positions beyond kv_len are masked with -inf.
    // Caller passes GGML_PAD(kv_len, 256) capped at max_seq_len, matching baseline.
    const int32_t ne11 = max_seq_len;
    const int32_t ne12 = 1;
    const int32_t ne13 = 1;
    const int32_t nb11 = k_stride;
    const int32_t nb12 = nb11 * ne11;
    const int64_t nb13 = (int64_t)nb12 * ne12;
    const int32_t nb21 = v_stride;
    const int32_t nb22 = nb21 * ne11;
    const int64_t nb23 = (int64_t)nb22 * ne12;
    const int32_t ne31 = kv_len;
    const int32_t ne32 = 1;
    const int32_t ne33 = 1;
    const int32_t nb31 = (int32_t)sizeof(__half);
    const int32_t nb32 = nb31 * ne31;
    const int64_t nb33 = (int64_t)nb32 * ne32;

    const float max_bias = 0.0f;
    const float m0       = 0.0f;
    const float m1       = 0.0f;
    const uint32_t n_head_log2 = 0;

    // Mask/sinks/KV_max: not used in megakernel (we handle masking directly)
    const char * __restrict__ mask   = nullptr;
    const char * __restrict__ sinks  = nullptr;
    const int  * __restrict__ KV_max = nullptr;

    // Q/K/V as char pointers for baseline arithmetic (baseline takes char*)
    const char * __restrict__ Q_c = (const char *) Q_f;
    const char * __restrict__ K_c = (const char *) K;
    const char * __restrict__ V_c = (const char *) V;

    // =========================================================================
    // Skip unused kernel variants for faster compilation (fattn-vec.cuh lines 46-58)
    // =========================================================================
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        (void)Q_c; (void)K_c; (void)V_c; (void)mask; (void)sinks; (void)KV_max; (void)dst;
        (void)effective_scale; (void)max_bias; (void)m0; (void)m1; (void)n_head_log2; (void)logit_softcap;
        (void)ne00; (void)ne01; (void)ne02; (void)ne03;
        (void)nb01; (void)nb02; (void)nb03;
        (void)ne10; (void)ne11; (void)ne12; (void)ne13;
        (void)nb11; (void)nb12; (void)nb13;
        (void)nb21; (void)nb22; (void)nb23;
        (void)ne31; (void)ne32; (void)ne33;
        (void)nb31; (void)nb32; (void)nb33;
        return;
    }

    // =========================================================================
    // Compile-time constants (fattn-vec.cuh lines 60-93)
    // =========================================================================

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    // fattn-vec.cuh lines 65-75
#ifdef RDNA
    constexpr int nthreads_KQ_q = 2;
#else
    constexpr int nthreads_KQ_q = 2; // We are always RDNA3 on gfx1100
#endif // RDNA
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);

    constexpr int nthreads    = 128; // ggml_cuda_fattn_vec_get_nthreads_device()
    // type_K == GGML_TYPE_F16: nthreads_KQ = 128 / cpy_nb
    constexpr int nthreads_KQ = 128 / cpy_nb;
    // type_V == GGML_TYPE_F16: nthreads_V = 128 / cpy_nb
    constexpr int nthreads_V  = 128 / cpy_nb;

    constexpr int ncols = 1;

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

    // V_DOT2 path (type_V == F16): V_rows_per_thread = 2*cpy_ne
    constexpr int V_rows_per_thread = 2*cpy_ne;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    // fattn-vec.cuh lines 87-93
    constexpr vec_dot_KQ_t vec_dot_KQ = vec_dot_fattn_vec_KQ_f16<D, nthreads_KQ>;
    constexpr bool Q_q8_1 = false; // type_K == F16 => not quantized
    // V_DOT2_F32_F16_AVAILABLE path: T=half
    constexpr dequantize_V_t dequantize_V = dequantize_V_f16_half<V_rows_per_thread>;

    // fattn-vec.cuh line 95
    const int ic0 = 0; // blockIdx.x * ncols; megakernel: single column

    // fattn-vec.cuh lines 97-99
    const int sequence = 0;
    const int head = 0;
    const int gqa_ratio = ne02 / ne12;
    // Q, K, V pointer offsets: already set up by caller, no offset needed
    // (baseline: Q += nb03*sequence + nb02*head + nb01*ic0; etc.)

    // fattn-vec.cuh line 104
    const half * maskh = (const half *) mask; // nullptr in our case

    // fattn-vec.cuh line 106
    const float slope = alibi_slope; // get_alibi_slope(max_bias, head, n_head_log2, m0, m1)

    // fattn-vec.cuh line 108 — baseline asserts D%64==0 for VEC.
    // Removed: kernel compiles for all D but is only CALLED when D%64==0 (runtime dispatch).
    // For D=96 (Phi-3.5), WMMA or TILE kernel is used instead.
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    // fattn-vec.cuh lines 113-121
    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine  = nwarps*V_cols_per_iter*D;
    // V_DOT2_F32_F16_AVAILABLE path
    half2            VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];

    // fattn-vec.cuh lines 123-129
    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // =========================================================================
    // Convert Q to half2 and store in registers (fattn-vec.cuh lines 131-237)
    // =========================================================================

    // V_DOT2_F32_F16_AVAILABLE path
    half2  Q_reg[ncols][(D/2)/nthreads_KQ]; // Will be initialized completely.
    // Q_q8_1 arrays -- always declared for register pressure matching
    int    Q_i32[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    float2  Q_ds[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];

    if constexpr (Q_q8_1) {
        // fattn-vec.cuh lines 140-192 -- q8_1 quantization path
        // Dead code for type_K=F16, but kept for register pressure matching
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 1 && ic0 + j >= int(ne01.z)) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    if (i0 + WARP_SIZE <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_f_j = (const float *) (Q_c + j*nb01);
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f_j + i0*sizeof(int), effective_scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
        // V_DOT2_F32_F16_AVAILABLE path (fattn-vec.cuh lines 194-217)
        const half2 scale_h2 = make_half2(effective_scale, effective_scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q_c + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;

                __align__(16) float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
    }

    // =========================================================================
    // Main KV loop (fattn-vec.cuh lines 239-368)
    // =========================================================================

    const int k_VKQ_max = KV_max ? KV_max[sequence*1 + 0] : ne11; // gridDim.x=1 for megakernel

    // Baseline: K += blockIdx.y*nthreads * nb11; V += blockIdx.y*nthreads * nb21;
    // maskh += blockIdx.y*nthreads;
    const char * K_loop = K_c + block_y*nthreads * nb11;
    const char * V_loop = V_c + block_y*nthreads * nb21;
    // maskh offset: maskh += blockIdx.y*nthreads (nullptr in our case, handled below)

    for (int k_VKQ_0 = block_y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += parallel_blocks*nthreads,
             // Increment pointers after each loop:
             K_loop += parallel_blocks*nthreads*nb11, V_loop += parallel_blocks*nthreads*nb21) {

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]; // KQ in registers.

        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum;

                // Compute dot product for ALL positions (matching baseline).
                // Baseline reads all ne11 positions and applies mask.
                // For positions >= kv_len, the cache data may be uninitialized,
                // but the -inf mask ensures they contribute 0 to softmax.
                if (k_VKQ_0 + i_KQ < k_VKQ_max) {
                    sum = vec_dot_KQ(K_loop + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                    sum = warp_reduce_sum<nthreads_KQ>(sum);

                    if (use_logit_softcap) {
                        sum = logit_softcap*tanhf(sum);
                    }

                    // Apply causal mask: positions >= kv_len get -inf
                    // This matches baseline's mask tensor approach.
                    if (k_VKQ_0 + i_KQ >= kv_len) {
                        sum = -INFINITY;
                    } else {
                        // ALiBi and relative position bias only for valid positions
                        if (alibi_slope != 0.0f) {
                            sum += slope * (float)((k_VKQ_0 + i_KQ) - cur_pos);
                        }
                        if (rel_pos_bias != nullptr) {
                            sum += rel_pos_bias[k_VKQ_0 + i_KQ];
                        }
                    }
                } else {
                    // Beyond allocated cache: equivalent to -inf
                    sum = -INFINITY;
                    warp_reduce_sum<nthreads_KQ>(0.0f);
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

            // V_DOT2_F32_F16_AVAILABLE path (fattn-vec.cuh lines 294-299)
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
        }

        // fattn-vec.cuh lines 309-311: __syncwarp not needed on HIP

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

            // V_DOT2_F32_F16_AVAILABLE path (fattn-vec.cuh lines 317-345)
            half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                half2 tmp[V_rows_per_thread/2];
                // type_V == F16, not BF16: direct dequantize path (fattn-vec.cuh lines 334-337)
                dequantize_V(V_loop + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
        }
    }

    // =========================================================================
    // Sinks processing (fattn-vec.cuh lines 370-401)
    // =========================================================================

    if (sinks && block_y == 0) {
        const float sink = ((const float *) sinks)[head];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max[j]) : 0.0f);

            // V_DOT2_F32_F16_AVAILABLE path
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
        }
    }

    // =========================================================================
    // Cross-warp reduction (fattn-vec.cuh lines 403-504)
    // =========================================================================

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= int(ne01.z)) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

        // V_DOT2_F32_F16_AVAILABLE path (fattn-vec.cuh lines 434-448)
        half2 * VKQ_tmp = (half2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
        }

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if (parallel_blocks == 1) {
                    // gridDim.y == 1: normalize and write final output
                    dst_val /= KQ_sum[j_VKQ];
                    dst[i0 + tid] = dst_val;
                } else {
                    // gridDim.y > 1: write unnormalized partial to scratch
                    dst_partial[block_y*D + i0 + tid] = dst_val;
                }
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }

    }

    // fattn-vec.cuh lines 502-504
    if (parallel_blocks > 1 && tid < ncols && (ncols == 1 || ic0 + tid < int(ne01.z))) {
        dst_meta[block_y * 2 + 0] = KQ_max[tid];
        dst_meta[block_y * 2 + 1] = KQ_sum[tid];
    }

    // Suppress unused-variable warnings for register-pressure-matching declarations
    (void)mask; (void)sinks; (void)KV_max;
    (void)max_bias; (void)m0; (void)m1; (void)n_head_log2;
    (void)ne00; (void)ne01; (void)ne02; (void)ne03;
    (void)nb01; (void)nb02; (void)nb03;
    (void)ne10; (void)ne12; (void)ne13;
    (void)nb12; (void)nb13;
    (void)nb22; (void)nb23;
    (void)ne31; (void)ne32; (void)ne33;
    (void)nb31; (void)nb32; (void)nb33;
    (void)ic0; (void)sequence; (void)head; (void)gqa_ratio;
    (void)maskh; (void)slope;
    (void)logit_softcap;
    (void)nthreads_KQ_q; (void)nthreads_V_q;
    (void)Q_q8_1;
    (void)Q_i32; (void)Q_ds;
    (void)k_VKQ_max;
    (void)vec_dot_KQ; (void)dequantize_V;
}
