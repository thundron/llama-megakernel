// attention-tile.h -- Port of baseline TILE flash attention kernel to HIP/gfx1100
// Source: ggml/src/ggml-cuda/fattn-tile.cuh
// Specialized for: decode (ncols=1, nwarps=1), type_K=F16, type_V=F16
// Target: gfx1100 (RDNA3, wave32, FAST_FP16_AVAILABLE, V_DOT2_F32_F16_AVAILABLE)
//
// Uses the exact same tiling, shared memory layout, and accumulation order as
// the baseline TILE kernel to produce bit-identical results for any D (including
// D=96 for Phi-3.5 where the VEC kernel diverges due to padding).
//
// Key simplification from baseline:
//   ncols=1, nwarps=1 => cpw=1, np=1
//   No cross-warp reduction needed.
//   nbatch_fa = WARP_SIZE (1 KQ position per thread)
#pragma once

#include "hip-shim.h"
#include "warp-reduce.h"
#include "mem-utils.h"
#include <cfloat>

// --- Constants from fattn-common.cuh ---
#ifndef FATTN_KQ_MAX_OFFSET
#define FATTN_KQ_MAX_OFFSET (3.0f * 0.6931f)
#endif

// ============================================================================
// ggml_cuda_unroll -- recursive compile-time unroll (from common.cuh:394-409)
// ============================================================================
template <int n>
struct tile_cuda_unroll {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        tile_cuda_unroll<n - 1>{}(f, args...);
    }
};

template <>
struct tile_cuda_unroll<1> {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};

// ============================================================================
// tile_load_tile_h2 -- loads K or V tile into shared memory
// Exact port of fattn-tile.cuh flash_attn_tile_load_tile (half2 -> half2 variant)
// Template params:
//   I = number of KV positions to load
//   J = number of dimension elements (in half units, must be % 8 == 0)
//   J_padding = extra padding in half2 units per row
// ============================================================================
template<int warp_size, int nwarps, int I, int J, int J_padding, bool oob_check>
static __device__ __forceinline__ void tile_load_tile_h2(
        const __half2 * const __restrict__ KV, __half2 * const __restrict__ tile_KV,
        const int stride_KV, const int i_sup) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes(); // 16
    constexpr int cpy_ne = cpy_nb / 4;                    // 4

    auto load = [&] __device__ (const int n) {
        const int stride_j = warp_size >> n;

        if (stride_j == 0) {
            return;
        }

        const int j0_start = stride_j == warp_size ? 0 : ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (2*stride_j);
        const int j0_stop  =                             ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (1*stride_j);
        const int stride_i = warp_size / stride_j;

        if (j0_start == j0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_j == warp_size ? 0 : threadIdx.x / stride_j);

            if (i0 + nwarps*stride_i <= I || i < I) {
#pragma unroll
                for (int j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    const int j = j0*cpy_ne + (stride_j == warp_size ? threadIdx.x : threadIdx.x % stride_j)*cpy_ne;

                    const __align__(16) __half2 zero[cpy_ne] = {{0.0f, 0.0f}};
                    ggml_cuda_memcpy_1<cpy_nb>(
                        tile_KV + i*(J/2 + J_padding) + j,
                        !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);
                }
            }
        }
    };
    static_assert(J % 8 == 0, "bad J");
    static_assert((J/2) % cpy_ne == 0, "bad J");
    tile_cuda_unroll<7>{}(load);
}

// ============================================================================
// tile_iter_KQ -- KQ matrix multiplication for one K-dimension tile
// Exact port of fattn-tile.cuh flash_attn_tile_iter_KQ, FAST_FP16_AVAILABLE path
// Specialized: ncols=1, nwarps=1, cpw=1, np=1
// ============================================================================
template <int warp_size, int DKQ, int nbatch_fa, int nbatch_K, bool oob_check>
static __device__ __forceinline__ void tile_iter_KQ(
        __half2  * const Q_tmp,
        const __half2 * const __restrict__ K_h2,
        __half2  * const KV_tmp,
        const int stride_K2,
        const int k_VKQ_0,
        const int k_VKQ_sup,
        const int k_KQ_0,
        float * KQ_acc) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes(); // 16
    constexpr int cpy_ne = cpy_nb / 4;                    // 4
    constexpr int nwarps = 1;

    // np=1: each thread handles nbatch_fa/warp_size KQ positions
    tile_load_tile_h2<warp_size, nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check>
        (K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2, KV_tmp, stride_K2, k_VKQ_sup);
    __syncthreads();

    // FAST_FP16_AVAILABLE path: half2 dot products
    static_assert((nbatch_K/2) % cpy_ne == 0, "bad nbatch_K");
#pragma unroll
    for (int k_KQ_1 = 0; k_KQ_1 < nbatch_K/2; k_KQ_1 += cpy_ne) {
        __align__(16) __half2 K_k[nbatch_fa/warp_size][cpy_ne];
        __align__(16) __half2 Q_k[cpy_ne]; // cpw=1

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += warp_size) {
            const int i_KQ = i_KQ_0 + threadIdx.x;
            ggml_cuda_memcpy_1<cpy_nb>(&K_k[i_KQ_0/warp_size], &KV_tmp[i_KQ*(nbatch_K/2 + cpy_ne) + k_KQ_1]);
        }

        // Load Q fragment (same for all threads in this warp since cpw=1, np=1)
        ggml_cuda_memcpy_1<cpy_nb>(&Q_k, &Q_tmp[k_KQ_0/2 + k_KQ_1]);

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += warp_size) {
#pragma unroll
            for (int k = 0; k < cpy_ne; ++k) {
                // v_dot2_f32_f16: accumulate K_k * Q_k into f32
                float & acc = KQ_acc[i_KQ_0/warp_size];
                const __half2 & kv = K_k[i_KQ_0/warp_size][k];
                const __half2 & qv = Q_k[k];
                asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(kv), "v"(qv));
            }
        }
    }

    if (k_KQ_0 + nbatch_K < DKQ) {
        __syncthreads(); // Sync not needed on last iteration.
    }
}

// ============================================================================
// tile_iter -- single iteration of main loop over up to nbatch_fa KV tokens
// Exact port of fattn-tile.cuh flash_attn_tile_iter
// Specialized: ncols=1, nwarps=1, cpw=1, np=1, FAST_FP16_AVAILABLE
// ============================================================================
template <int warp_size, int DKQ, int DV, int nbatch_fa, int nbatch_K,
    bool use_logit_softcap, bool oob_check>
static __device__ __forceinline__ void tile_iter(
        __half2 * const Q_tmp,
        const __half2 * const __restrict__ K_h2,
        const __half2 * const __restrict__ V_h2,
        const float logit_softcap,
        __half    * const KQ,
        __half2   * const KV_tmp,
        const int stride_K2,
        const int stride_V2,
        float * const KQ_max,
        float * const KQ_sum,
        __half2 * const VKQ,
        const int k_VKQ_0,
        const int k_VKQ_max,
        // ALiBi & rel pos bias for decode
        const float alibi_slope,
        const int   cur_pos,
        const float * __restrict__ rel_pos_bias) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes(); // 16
    constexpr int cpy_ne = cpy_nb / 4;                    // 4
    constexpr int nwarps = 1;

    constexpr int DVp = (DV + 2*warp_size - 1) & ~(2*warp_size - 1); // DV padded to multiple of 2*warp_size

    // cpw=1, KQ_cs=1 (single element per column chunk)
    constexpr int KQ_cs = 1;

    const int k_VKQ_sup = k_VKQ_max - k_VKQ_0;

    float KQ_max_new = KQ_max[0];

    // With np=1: nbatch_fa/warp_size KQ positions per thread
    float KQ_acc[nbatch_fa/warp_size] = {0.0f};

    // KQ = K @ Q matrix multiplication:
    constexpr int nbatch_K_last = DKQ % nbatch_K;
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < DKQ - nbatch_K_last; k_KQ_0 += nbatch_K) {
        tile_iter_KQ<warp_size, DKQ, nbatch_fa, nbatch_K, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }
    if (nbatch_K_last > 0) {
        constexpr int k_KQ_0 = DKQ - nbatch_K_last;
        tile_iter_KQ<warp_size, DKQ, nbatch_fa, nbatch_K_last, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }

    // Apply logit softcap, ALiBi, rel_pos_bias, update KQ_max:
#pragma unroll
    for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += warp_size) {
        const int i_KQ = i_KQ_0 + threadIdx.x;

        if (use_logit_softcap) {
            KQ_acc[i_KQ_0/warp_size] = logit_softcap * tanhf(KQ_acc[i_KQ_0/warp_size]);
        }

        if (!oob_check || i_KQ < k_VKQ_sup) {
            // ALiBi: add slope * (kv_position - q_position)
            if (alibi_slope != 0.0f) {
                KQ_acc[i_KQ_0/warp_size] += alibi_slope * (float)((k_VKQ_0 + i_KQ) - cur_pos);
            }
            // T5 relative position bias
            if (rel_pos_bias != nullptr) {
                KQ_acc[i_KQ_0/warp_size] += rel_pos_bias[k_VKQ_0 + i_KQ];
            }

            KQ_max_new = fmaxf(KQ_max_new, KQ_acc[i_KQ_0/warp_size] + FATTN_KQ_MAX_OFFSET);
        }
    }

    KQ_max_new = warp_reduce_max<warp_size>(KQ_max_new);

    // np=1: no cross-warp sharing needed, syncthreads to prepare for KQ write
    __syncthreads();

    // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
    {
        __half tmp[nbatch_fa/warp_size];

        const float KQ_max_scale = expf(KQ_max[0] - KQ_max_new);
        KQ_max[0] = KQ_max_new;

        float KQ_sum_add = 0.0f;
#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += warp_size) {
            const float val = !oob_check || i0 + threadIdx.x < static_cast<uint32_t>(k_VKQ_sup) ?
                expf(KQ_acc[i0/warp_size] - KQ_max[0]) : 0.0f;
            KQ_sum_add += val;
            tmp[i0/warp_size] = (__half)val;
        }
        KQ_sum[0] = KQ_sum[0]*KQ_max_scale + KQ_sum_add;

        const __half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
        for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
            VKQ[i0/warp_size] *= KQ_max_scale_h2;
        }

        // Write KQ values to shared memory
#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += warp_size) {
            const int i = i0 + threadIdx.x;
            KQ[i] = tmp[i0/warp_size];
        }
    }

    // VKQ = V @ KQ matrix multiplication:
    // FAST_FP16_AVAILABLE path
    static_assert(DV <= DKQ, "bad DV");
    static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K*2/3) == 0), "bad nbatch_K");
    constexpr int nbatch_V = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K*2/3) * nbatch_fa / DV;
    static_assert(nbatch_fa % nbatch_V == 0, "bad nbatch_V");
#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
        tile_load_tile_h2<warp_size, nwarps, nbatch_V, DV, 0, oob_check>
            (V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0);
        __syncthreads();

        // np=1: iterate one position at a time
#pragma unroll
        for (int k1 = 0; k1 < nbatch_V; ++k1) {
            __align__(16) __half2 V_k[(DVp/2)/warp_size];

            constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(&V_k[i0/warp_size], &KV_tmp[k1*(DV/2) + i0 + threadIdx.x*cpy_ne_D]);
            }

            // Load KQ weight for this position
            __half2 KQ_k = __half2half2(KQ[k0 + k1]);

#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[i0/warp_size] += V_k[i0/warp_size] * KQ_k;
            }
        }

        __syncthreads();
    }
}

// ============================================================================
// flash_attn_tile_f16 -- Full TILE attention kernel for decode (ncols=1)
// Template params: D = head dimension (compile-time), nbatch_fa, nbatch_K
// from the RDNA config table in fattn-tile.cuh.
//
// Uses nwarps=1 (single warp) for decode since ncols=1.
// This gives np=1, cpw=1, nbatch_fa/warp_size positions per thread.
//
// For RDNA decode configs (nbatch_fa = WARP_SIZE = 32):
//   D=64:  nbatch_K=64
//   D=96:  nbatch_K=48
//   D=128: nbatch_K=64
//   D=256: nbatch_K=64
//
// Thread block: dim3(WARP_SIZE, 1, 1) = dim3(32, 1, 1) = 32 threads
// Grid: (n_q_heads, 1, 1)
// ============================================================================
template <int D, int NBATCH_FA, int NBATCH_K>
static __device__ void flash_attn_tile_f16(
        const float  * __restrict__ Q_f,      // [D] f32 query for one head (NOT yet scaled)
        const __half * __restrict__ K,        // [kv_len, D] f16 K cache (position-major per head)
        const __half * __restrict__ V,        // [kv_len, D] f16 V cache (position-major per head)
        float        * __restrict__ dst,      // [D] f32 output
        const int kv_len,                     // number of valid KV positions
        const float scale,                    // 1/sqrt(D)
        const int k_stride,                   // stride between K positions (in bytes) = D * sizeof(half)
        const int v_stride,                   // stride between V positions (in bytes) = D * sizeof(half)
        const float alibi_slope = 0.0f,       // ALiBi slope for this head (0 = disabled)
        const int   cur_pos = 0,              // current token position
        const float * __restrict__ rel_pos_bias = nullptr,
        const float attn_logit_softcap = 0.0f) {

    constexpr int warp_size = WARP_SIZE; // 32
    constexpr int nwarps    = 1;
    constexpr int nbatch_fa = NBATCH_FA;
    constexpr int nbatch_K  = NBATCH_K;

    static_assert(nbatch_fa == warp_size, "For nwarps=1 decode, nbatch_fa must equal warp_size");

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes(); // 16
    constexpr int cpy_ne = cpy_nb / 4;                    // 4

    constexpr int DKQp = (D + 2*warp_size - 1) & ~(2*warp_size - 1); // D padded to multiple of 64
    constexpr int DVp  = DKQp; // DKQ == DV for our use case

    // Shared memory layout (FAST_FP16_AVAILABLE path):
    //   Q_tmp:  [D/2] half2
    //   KV_tmp: [nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-D] half2
    //   KQ:     [nbatch_fa] half
    __shared__ __half2 Q_tmp[D/2];
    __shared__ __half2 KV_tmp[nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp - D];
    __shared__ __half  KQ[nbatch_fa];

    // VKQ accumulators in registers: half2[(DVp/2)/warp_size]
    __align__(16) __half2 VKQ[(DVp/2)/warp_size] = {{0.0f, 0.0f}};

    float KQ_max = -FLT_MAX/2.0f;
    float KQ_sum = 0.0f;

    // Determine logit softcap usage from compile-time macro
    // Baseline divides scale by softcap before Q loading so that
    // KQ_acc = Q*K^T / (sqrt(D)*softcap), then applies softcap*tanh(KQ_acc).
#if defined(HAS_ATTN_SOFTCAP) && HAS_ATTN_SOFTCAP
    constexpr bool use_logit_softcap = true;
    const float logit_softcap_val = ATTN_SOFTCAP_VAL;
    const float effective_scale = scale / ATTN_SOFTCAP_VAL;
#else
    constexpr bool use_logit_softcap = false;
    const float logit_softcap_val = 0.0f;
    const float effective_scale = scale;
#endif

    // Load Q data, scale, convert to FP16:
    // nwarps=1, np=1: single warp loads all Q elements
    {
        constexpr int cpy_ne_D = cpy_ne < DKQp/warp_size ? cpy_ne : DKQp/warp_size;

#pragma unroll
        for (int i0 = 0; i0 < DKQp; i0 += warp_size*cpy_ne_D) {
            if (i0 + warp_size*cpy_ne_D <= D || i0 + threadIdx.x*cpy_ne_D < D) {
                __align__(16) float tmp_f[cpy_ne_D] = {0.0f};
                ggml_cuda_memcpy_1<sizeof(tmp_f)>
                    (tmp_f, &Q_f[i0 + threadIdx.x*cpy_ne_D]);

#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    tmp_f[i1] *= effective_scale;
                }

                __align__(16) __half2 tmp_h2[cpy_ne_D/2];
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; i1 += 2) {
                    tmp_h2[i1/2] = __halves2half2(__float2half(tmp_f[i1 + 0]), __float2half(tmp_f[i1 + 1]));
                }
                ggml_cuda_memcpy_1<sizeof(tmp_h2)>(
                    &Q_tmp[i0/2 + threadIdx.x*(cpy_ne_D/2)],
                    tmp_h2);
            }
        }
    }

    __syncthreads();

    // Strides in half2 units
    const int stride_K2 = k_stride / sizeof(__half2); // D/2
    const int stride_V2 = v_stride / sizeof(__half2); // D/2

    const __half2 * K_h2 = (const __half2 *) K;
    const __half2 * V_h2 = (const __half2 *) V;

    // Main loop over KV cache:
    const int k_VKQ_max = kv_len;
    {
        int k_VKQ_0 = 0;
        while (k_VKQ_0 < k_VKQ_max - nbatch_fa) {
            constexpr bool oob_check = false;
            tile_iter<warp_size, D, D, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>
                (Q_tmp, K_h2, V_h2, logit_softcap_val,
                 KQ, KV_tmp, stride_K2, stride_V2,
                 &KQ_max, &KQ_sum, VKQ, k_VKQ_0, k_VKQ_max,
                 alibi_slope, cur_pos, rel_pos_bias);
            k_VKQ_0 += nbatch_fa;
        }
        if (k_VKQ_0 < k_VKQ_max) {
            constexpr bool oob_check = true;
            tile_iter<warp_size, D, D, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>
                (Q_tmp, K_h2, V_h2, logit_softcap_val,
                 KQ, KV_tmp, stride_K2, stride_V2,
                 &KQ_max, &KQ_sum, VKQ, k_VKQ_0, k_VKQ_max,
                 alibi_slope, cur_pos, rel_pos_bias);
        }
    }

    KQ_sum = warp_reduce_sum<warp_size>(KQ_sum);

    // np=1: no cross-warp reduction needed. Write back results directly.
    {
        const float norm = 1.0f / KQ_sum;

        constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
            __align__(16) float2 tmp[cpy_ne_D];
#pragma unroll
            for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = __half22float2(VKQ[i0/warp_size + i1]);
                tmp[i1].x *= norm;
                tmp[i1].y *= norm;
            }
            if (i0 + warp_size*cpy_ne_D <= D/2 || i0 + threadIdx.x*cpy_ne_D < D/2) {
                ggml_cuda_memcpy_1<sizeof(tmp)>(&dst[2*i0 + threadIdx.x*(2*cpy_ne_D)], tmp);
            }
        }
    }
}
