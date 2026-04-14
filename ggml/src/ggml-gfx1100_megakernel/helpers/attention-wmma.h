// attention-wmma.h -- Faithful port of baseline WMMA flash attention kernel
// Source: ggml/src/ggml-cuda/fattn-wmma-f16.cu (lines 24-507)
// Specialized for: decode (ncols=16 padded, 1 actual query), type_K=F16, type_V=F16
// Target: gfx1100 (RDNA3, wave32, 16x16 WMMA fragments via rocWMMA)
//
// This is a line-by-line port of baseline's flash_attn_ext_f16 kernel.
// Every variable, every branch, every loop from baseline is preserved.
// Only the following adaptations are made:
//   1. Function signature uses our megakernel interface (Q_f, K, V, dst, kv_len, etc.)
//   2. Parameter mapping at the top (ne11 = kv_len, etc.)
//   3. Mask/sinks tensor access replaced with parameter-based approach
//   4. blockIdx.y/gridDim.y = 1 (single block per head in megakernel)
//   5. rocwmma namespace alias and HIP-specific half type
#pragma once

#ifdef GGML_USE_WMMA_FATTN

#include "hip-shim.h"
#include "warp-reduce.h"
#include <rocwmma/rocwmma.hpp>
#include <cfloat>
#include <type_traits>

namespace wmma = rocwmma;

// --- Constants from fattn-common.cuh ---
#ifndef FATTN_KQ_STRIDE
#define FATTN_KQ_STRIDE 256
#endif
#ifndef FATTN_KQ_MAX_OFFSET
#define FATTN_KQ_MAX_OFFSET (3.0f * 0.6931f)
#endif
#ifndef SOFTMAX_FTZ_THRESHOLD
#define SOFTMAX_FTZ_THRESHOLD (-20.0f)
#endif
#ifndef HALF_MAX_HALF
#define HALF_MAX_HALF __float2half(65504.0f/2)
#endif

// ============================================================================
// Compile-time helpers (from fattn-wmma-f16.cu lines 509-521)
// ============================================================================

static constexpr int wmma_get_max_power_of_2(int x) {
    return x % 2 == 0 ? 2 * wmma_get_max_power_of_2(x / 2) : 1;
}

static constexpr int wmma_get_VKQ_stride(int D, int nwarps, int frag_m) {
    return (wmma_get_max_power_of_2(D / frag_m) < nwarps ?
            wmma_get_max_power_of_2(D / frag_m) : nwarps) * frag_m;
}

// ============================================================================
// HIP shims for half2 intrinsics used in half2 code paths
// ============================================================================
static __device__ __forceinline__ half2 mega_h2exp(half2 x) {
    return __halves2half2(hexp(__low2half(x)), hexp(__high2half(x)));
}

static __device__ __forceinline__ half mega_ggml_cuda_hmax(const half a, const half b) {
    return __hmax(a, b);
}

static __device__ __forceinline__ half2 mega_ggml_cuda_hmax2(const half2 a, const half2 b) {
    return __halves2half2(__hmax(__low2half(a), __low2half(b)), __hmax(__high2half(a), __high2half(b)));
}

// warp_reduce_max for half (needed for half2 branch)
template<int width = WARP_SIZE>
static __device__ __forceinline__ half warp_reduce_max_h(half x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = mega_ggml_cuda_hmax(x, __shfl(x, __lane_id() ^ offset, width));
    }
    return x;
}

// warp_reduce_sum for half2 (needed for half2 branch)
template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_reduce_sum_h2(half2 x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        half2 other;
        other = __halves2half2(
            __shfl(__low2half(x),  __lane_id() ^ offset, width),
            __shfl(__high2half(x), __lane_id() ^ offset, width));
        x = __hadd2(x, other);
    }
    return x;
}

// ============================================================================
// flash_attn_wmma_f16_impl -- Line-by-line port of baseline flash_attn_ext_f16
//
// Template params match baseline: D, ncols, nwarps, VKQ_stride, KQ_acc_t,
//   use_logit_softcap.
// Thread block: dim3(32, 4, 1) = 128 threads, 4 warps
// Grid: 1 block per Q head (blockIdx.y = 0, gridDim.y = 1)
// ============================================================================
template <int D, int ncols, int nwarps, int VKQ_stride, typename KQ_acc_t, bool use_logit_softcap>
static __device__ void flash_attn_wmma_f16_impl(
        const float  * __restrict__ Q_f,      // [D] f32 query for one head
        const half   * __restrict__ K_h,      // [kv_len, D] f16 K cache (position-major)
        const half   * __restrict__ V_h,      // [kv_len, D] f16 V cache (position-major)
        float        * __restrict__ dst,      // [D] f32 output
        const int kv_len,                     // number of valid KV positions
        const float scale,                    // 1/sqrt(D)
        const float alibi_slope,              // ALiBi slope for this head (0 = disabled)
        const int   cur_pos,                  // current token position (for ALiBi)
        const float logit_softcap) {          // softcap value (0 = disabled)

    // --- Baseline line 48-52: skip unused kernel variants ---
    // (Not needed in megakernel — we only instantiate what we use)

    // --- Baseline line 56: warp_size ---
    constexpr int warp_size = WARP_SIZE;  // 32 on gfx1100

    // --- Baseline line 58: ic0 ---
    // Baseline: const int ic0 = ncols * blockIdx.x;
    // In baseline, blockIdx.x indexes which ncols-wide column block to process.
    // In megakernel, blockIdx.x is the Q head index, NOT the column index.
    // We always process the single query at column 0, so ic0 = 0.
    const int ic0 = 0;

    // --- Baseline lines 60-64: static asserts and frag sizes ---
    static_assert(D <= FATTN_KQ_STRIDE, "D must be <= FATTN_KQ_STRIDE.");
    static_assert(ncols == 8 || ncols % 16 == 0, "ncols must be 8 or a multiple of 16.");
    constexpr int frag_m = ncols == 8 ? 32 : 16;
    constexpr int frag_n = ncols == 8 ?  8 : 16;
    static_assert(D % frag_m == 0, "If ncols == 8 then D % frag_m must be 0.");

    // --- Baseline lines 66-77: fragment types ---
    // On HIP we use half (not _Float16) for rocwmma compatibility
    typedef wmma::fragment<wmma::matrix_a,    frag_m, frag_n, 16, half, wmma::row_major> frag_a_K;
    typedef wmma::fragment<wmma::matrix_a,    frag_m, frag_n, 16, half, wmma::col_major> frag_a_V;
    typedef wmma::fragment<wmma::matrix_b,    frag_m, frag_n, 16, half, wmma::col_major> frag_b;
    typedef wmma::fragment<wmma::accumulator, frag_m, frag_n, 16, KQ_acc_t>              frag_c_KQ;
    typedef wmma::fragment<wmma::accumulator, frag_m, frag_n, 16, half>                  frag_c_VKQ;

    // --- Baseline lines 79-86: compile-time constants ---
    constexpr int KQ_stride_tc = nwarps * frag_m; // Number of KQ rows calculated in parallel.
    constexpr int VKQ_ratio = KQ_stride_tc / VKQ_stride; // Number of parallel VKQ accumulators needed to keep all warps busy.
    static_assert(VKQ_ratio <= nwarps, "VKQ_ratio must be <= nwarps.");

    // Pad internal representation of KQ, KQV to reduce shared memory bank conflicts:
    constexpr int D_padded = D + 8;
    constexpr int kqs_padded = FATTN_KQ_STRIDE + 8;
    constexpr int kqar = sizeof(KQ_acc_t) / sizeof(half);

    // --- Baseline lines 88-99: pointer setup ---
    // Megakernel mapping: sequence=0, head resolved externally
    const int sequence = 0;
    const int head     = 0;  // head mapping done outside
    const int gqa_ratio = 1; // GQA resolved outside

    // Q_f, K_h, V_h are already offset to the correct head by the caller.
    // Baseline: const float * Q_f = (const float *)(Q + nb03*sequence + nb02*head + nb01*ic0);
    // Baseline: const half  * K_h = (const half *)(K + nb13*sequence + nb12*(head/gqa_ratio));
    // Baseline: const half  * V_h = (const half *)(V + nb13*sequence + nb12*(head/gqa_ratio));

    // Mask and sinks: baseline lines 94-96
    // In megakernel we don't have mask/sinks tensors — we compute mask inline.
    // But we declare the pointers for register pressure matching.
    const half  * maskh  = nullptr;
    const half2 * mask2  = nullptr;
    const float * sinksf = nullptr;

    // ne01.z = number of valid Q columns. For decode, this is 1.
    const int ne01_z = 1;

    // ne11 = baseline's K->ne[1] = GGML_PAD(kv_used, 256).
    // This sets k_VKQ_max (the loop bound). Masking still uses kv_len directly.
    // For WMMA with FATTN_KQ_STRIDE=256, this produces the same iteration count
    // as ne11=kv_len, but aligns with baseline's value.
    const int ne11 = ((kv_len + 255) & ~255);

    // nb31 = mask row stride in bytes. Not used (mask==nullptr) but declared.
    const int nb31 = 0;

    const int stride_Q  = D; // baseline: nb01 / sizeof(float)
    const int stride_KV = D; // baseline: nb11 / sizeof(half)

    // --- Baseline lines 102-106: ALiBi / softcap variables ---
    const float slopef = alibi_slope;
    const half  slopeh = __float2half(slopef);
    const half2 slope2 = make_half2(slopef, slopef);

    const half2 logit_softcap_2 = make_half2(logit_softcap, logit_softcap);

    // --- Baseline line 108: Q fragments ---
    frag_b Q_b[D / 16][ncols / frag_n];

    // --- Baseline lines 110-115: shared memory ---
    // A single buffer for temporarily holding tiles of KQ and VKQ parts:
    constexpr int mem_KQ = ncols * kqs_padded * kqar;
    constexpr int mem_VKQ_parts = VKQ_ratio * ncols * D_padded;
    __shared__ half KQ[mem_KQ >= mem_VKQ_parts ? mem_KQ : mem_VKQ_parts];
    float * KQ_f  = (float *) KQ;
    half2 * KQ2   = (half2 *) KQ;

    // --- Baseline lines 117-124: float accumulators ---
    float    KQ_rowsum_f[ncols / nwarps] = {0.0f};
    float       KQ_max_f[ncols / nwarps];
    float KQ_max_scale_f[ncols / nwarps] = {0.0f};

#pragma unroll
    for (int j = 0; j < ncols / nwarps; ++j) {
        KQ_max_f[j] = -FLT_MAX / 2.0f;
    }

    // --- Baseline lines 126-133: half2 accumulators ---
    half2    KQ_rowsum_h2[ncols / nwarps] = {{0.0f, 0.0f}};
    half2       KQ_max_h2[ncols / nwarps];
    half2 KQ_max_scale_h2[ncols / nwarps] = {{0.0f, 0.0f}};

#pragma unroll
    for (int j = 0; j < ncols / nwarps; ++j) {
        KQ_max_h2[j] = make_half2(-__half2float(HALF_MAX_HALF), -__half2float(HALF_MAX_HALF));
    }

    // --- Baseline lines 135-136: VKQ accumulator ---
    __shared__ half VKQ[ncols * D_padded]; // Accumulator for final VKQ slice.
    half2 * VKQ2 = (half2 *) VKQ;

    // --- Baseline lines 138-148: _Float16 / half pointer aliases ---
    // On HIP < 6.5 these are identity aliases.
    const half * K_h_f16  = K_h;
    const half * V_h_f16  = V_h;
    half       * KQ_f16   = KQ;
    half       * VKQ_f16  = VKQ;

    // --- Baseline lines 150-161: Initialize VKQ to zero ---
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < D / 2; i0 += warp_size) {
            const int i = i0 + threadIdx.x;
            if (i0 + warp_size > D / 2 && i >= D / 2) {
                break;
            }
            VKQ2[j * (D_padded / 2) + i] = make_half2(0.0f, 0.0f);
        }
    }

    // --- Baseline lines 163-175: Convert Q to half and apply scale, store in KQ ---
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < D; i0 += warp_size) {
            const int i = i0 + threadIdx.x;
            if (i0 + warp_size > D && i >= D) {
                break;
            }
            // Baseline: KQ[j*D_padded + i] = ic0 + j < ne01.z ? Q_f[j*stride_Q + i] * scale : 0.0f;
            // For megakernel decode: ic0=0, ne01_z=1, so column 0 gets Q data, rest get 0.
            KQ[j * D_padded + i] = ic0 + j < ne01_z ? Q_f[j * stride_Q + i] * scale : 0.0f;
        }
    }

    __syncthreads();

    // --- Baseline lines 179-186: Load Q into tensor core fragments ---
#pragma unroll
    for (int i0 = 0; i0 < D; i0 += 16) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += frag_n) {
            wmma::load_matrix_sync(Q_b[i0 / 16][j0 / frag_n], KQ_f16 + j0 * D_padded + i0, D_padded);
        }
    }

    __syncthreads();

    // --- Baseline line 191: k_VKQ_max ---
    // Baseline: const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    const int k_VKQ_max = ne11;

    // --- Baseline lines 192-404: Main KV loop ---
    // Baseline: for (int k_VKQ_0 = blockIdx.y*FATTN_KQ_STRIDE; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*FATTN_KQ_STRIDE)
    // Megakernel: blockIdx.y=0, gridDim.y=1
    for (int k_VKQ_0 = 0; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += FATTN_KQ_STRIDE) {
        // --- Baseline lines 194-214: Calculate tile of KQ ---
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE; i_KQ_0 += KQ_stride_tc) {
            frag_c_KQ KQ_c[ncols / frag_n];
#pragma unroll
            for (int j = 0; j < ncols / frag_n; ++j) {
                wmma::fill_fragment(KQ_c[j], static_cast<KQ_acc_t>(0.0f));
            }
#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D; k_KQ_0 += 16) {
                frag_a_K K_a;
                wmma::load_matrix_sync(K_a, K_h_f16 + int64_t(k_VKQ_0 + i_KQ_0 + frag_m * threadIdx.y) * stride_KV + k_KQ_0, stride_KV);
#pragma unroll
                for (int j = 0; j < ncols / frag_n; ++j) {
                    wmma::mma_sync(KQ_c[j], K_a, Q_b[k_KQ_0 / 16][j], KQ_c[j]);
                }
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                wmma::store_matrix_sync(
                    (KQ_acc_t *) KQ + j0 * kqs_padded + i_KQ_0 + frag_m * threadIdx.y,
                    KQ_c[j0 / frag_n], kqs_padded, wmma::mem_col_major);
            }
        }

        __syncthreads();

        // --- Baseline lines 220-321: Calculate softmax for each KQ column ---
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (std::is_same<KQ_acc_t, float>::value) {
                // --- Baseline lines 224-271: float path ---
                float KQ_f_tmp[FATTN_KQ_STRIDE / warp_size];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += warp_size) {
                    const int k = k0 + threadIdx.x;

                    KQ_f_tmp[k0 / warp_size] = KQ_f[j * kqs_padded + k];

                    if (use_logit_softcap) {
                        KQ_f_tmp[k0 / warp_size] = logit_softcap * tanhf(KQ_f_tmp[k0 / warp_size]);
                    }
                }

                float KQ_max_new = KQ_max_f[j0 / nwarps];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += warp_size) {
                    const int k = k0 + threadIdx.x;

                    // Baseline line 242-243:
                    // KQ_f_tmp[k0/warp_size] += mask && ic0 + j < int(ne01.z) ?
                    //     __half2float(slopeh*maskh[j*(nb31/sizeof(half)) + k_VKQ_0 + k]) : 0.0f;
                    //
                    // Megakernel: no mask tensor. For causal decode, positions >= kv_len
                    // are masked out (set to -inf). ALiBi is applied as slope*(pos - cur_pos).
                    // Padded Q columns (ic0+j >= ne01_z) also get -inf.
                    if (k_VKQ_0 + k >= kv_len || ic0 + j >= ne01_z) {
                        KQ_f_tmp[k0 / warp_size] = -INFINITY;
                    } else if (slopef != 0.0f) {
                        KQ_f_tmp[k0 / warp_size] += slopef * (float)((k_VKQ_0 + k) - cur_pos);
                    }

                    KQ_max_new = fmaxf(KQ_max_new, KQ_f_tmp[k0 / warp_size] + FATTN_KQ_MAX_OFFSET);
                }
                KQ_max_new = warp_reduce_max<warp_size>(KQ_max_new);

                const float diff = KQ_max_f[j0 / nwarps] - KQ_max_new;
                KQ_max_scale_f[j0 / nwarps] = expf(diff);
                if (diff <= SOFTMAX_FTZ_THRESHOLD) {
                    KQ_max_scale_f[j0 / nwarps] = 0.0f;
                }
                KQ_max_f[j0 / nwarps] = KQ_max_new;

                float KQ_rowsum_add = 0.0f;
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += warp_size) {
                    const int k = k0 + threadIdx.x;

                    const float diff = KQ_f_tmp[k0 / warp_size] - KQ_max_f[j0 / nwarps];
                    KQ_f_tmp[k0 / warp_size] = expf(diff);
                    if (diff <= SOFTMAX_FTZ_THRESHOLD) {
                        KQ_f_tmp[k0 / warp_size] = 0.0f;
                    }
                    KQ_rowsum_add += KQ_f_tmp[k0 / warp_size];
                    KQ[j * (kqar * kqs_padded) + k] = KQ_f_tmp[k0 / warp_size];
                }
                KQ_rowsum_add = warp_reduce_sum<warp_size>(KQ_rowsum_add);

                // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
                KQ_rowsum_f[j0 / nwarps] = KQ_max_scale_f[j0 / nwarps] * KQ_rowsum_f[j0 / nwarps] + KQ_rowsum_add;
            } else {
                // --- Baseline lines 272-321: half2 path ---
                half2 KQ2_tmp[FATTN_KQ_STRIDE / (2 * warp_size)];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE / 2; k0 += warp_size) {
                    const int k = k0 + threadIdx.x;

                    KQ2_tmp[k0 / warp_size] = KQ2[j * (kqs_padded / 2) + k];

                    if (use_logit_softcap) {
                        // There is no dedicated tangens hyperbolicus function for half2.
                        KQ2_tmp[k0 / warp_size] = mega_h2exp(KQ2_tmp[k0 / warp_size] * make_half2(2.0f, 2.0f));
                        KQ2_tmp[k0 / warp_size] = (KQ2_tmp[k0 / warp_size] - make_half2(1.0f, 1.0f))
                                                 / (KQ2_tmp[k0 / warp_size] + make_half2(1.0f, 1.0f));

                        KQ2_tmp[k0 / warp_size] *= logit_softcap_2;
                    }
                }

                half2 KQ_max_new = KQ_max_h2[j0 / nwarps];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE / 2; k0 += warp_size) {
                    const int k = k0 + threadIdx.x;

                    // Baseline line 295:
                    // KQ2_tmp[k0/warp_size] += mask && ic0 + j < int(ne01.z) ?
                    //     slope2*mask2[(j*ne11 + k_VKQ_0)/2 + k] : make_half2(0.0f, 0.0f);
                    //
                    // Megakernel: no mask tensor. Apply masking inline for half2 path.
                    // Each half2 element covers 2 KV positions: (k_VKQ_0 + 2*k) and (k_VKQ_0 + 2*k + 1).
                    if (ic0 + j >= ne01_z) {
                        KQ2_tmp[k0 / warp_size] = make_half2(-__half2float(HALF_MAX_HALF), -__half2float(HALF_MAX_HALF));
                    } else {
                        const int pos_lo = k_VKQ_0 + 2 * k;
                        const int pos_hi = pos_lo + 1;
                        half lo = __low2half(KQ2_tmp[k0 / warp_size]);
                        half hi = __high2half(KQ2_tmp[k0 / warp_size]);
                        if (pos_lo >= kv_len) {
                            lo = __float2half(-__half2float(HALF_MAX_HALF));
                        } else if (slopef != 0.0f) {
                            lo = __hadd(lo, __float2half(slopef * (float)(pos_lo - cur_pos)));
                        }
                        if (pos_hi >= kv_len) {
                            hi = __float2half(-__half2float(HALF_MAX_HALF));
                        } else if (slopef != 0.0f) {
                            hi = __hadd(hi, __float2half(slopef * (float)(pos_hi - cur_pos)));
                        }
                        KQ2_tmp[k0 / warp_size] = __halves2half2(lo, hi);
                    }

                    KQ_max_new = mega_ggml_cuda_hmax2(KQ_max_new, KQ2_tmp[k0 / warp_size]);
                }
                KQ_max_new = __halves2half2(
                    warp_reduce_max_h<warp_size>(mega_ggml_cuda_hmax(__low2half(KQ_max_new), __high2half(KQ_max_new))),
                    warp_reduce_max_h<warp_size>(mega_ggml_cuda_hmax(__low2half(KQ_max_new), __high2half(KQ_max_new))));
                const half2 diff = KQ_max_h2[j0 / nwarps] - KQ_max_new;
                KQ_max_scale_h2[j0 / nwarps] = mega_h2exp(diff);
                // Baseline line 301-302: ftz_mask via __hgt2_mask
                // Approximate: if both halves are below threshold, zero out
                if (__half2float(__low2half(diff)) <= SOFTMAX_FTZ_THRESHOLD &&
                    __half2float(__high2half(diff)) <= SOFTMAX_FTZ_THRESHOLD) {
                    KQ_max_scale_h2[j0 / nwarps] = make_half2(0.0f, 0.0f);
                }
                KQ_max_h2[j0 / nwarps] = KQ_max_new;

                half2 KQ_rowsum_add = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE / 2; k0 += warp_size) {
                    const int k = k0 + threadIdx.x;

                    const half2 diff = KQ2_tmp[k0 / warp_size] - KQ_max_h2[j0 / nwarps];
                    KQ2_tmp[k0 / warp_size] = mega_h2exp(diff);
                    // ftz_mask approximation
                    if (__half2float(__low2half(diff)) <= SOFTMAX_FTZ_THRESHOLD &&
                        __half2float(__high2half(diff)) <= SOFTMAX_FTZ_THRESHOLD) {
                        KQ2_tmp[k0 / warp_size] = make_half2(0.0f, 0.0f);
                    }
                    KQ_rowsum_add += KQ2_tmp[k0 / warp_size];
                    KQ2[j * (kqs_padded / 2) + k] = KQ2_tmp[k0 / warp_size];
                }
                KQ_rowsum_add = warp_reduce_sum_h2<warp_size>(KQ_rowsum_add);

                // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
                KQ_rowsum_h2[j0 / nwarps] = KQ_max_scale_h2[j0 / nwarps] * KQ_rowsum_h2[j0 / nwarps] + KQ_rowsum_add;
            }
        }

        __syncthreads();

        // --- Baseline lines 326-337: Load KQ into WMMA fragments ---
        frag_b KQ_b[FATTN_KQ_STRIDE / (VKQ_ratio * 16)][ncols / frag_n];
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += frag_n) {
#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += VKQ_ratio * 16) {
                const int k = k0 + (threadIdx.y % VKQ_ratio) * 16;
                wmma::load_matrix_sync(
                    KQ_b[k0 / (VKQ_ratio * 16)][j0 / frag_n],
                    KQ_f16 + j0 * (kqar * kqs_padded) + k,
                    kqar * kqs_padded);
            }
        }

        // --- Baseline lines 339-358: V accumulation via WMMA ---
        frag_c_VKQ VKQ_c[D / VKQ_stride][ncols / frag_n];
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D; i_VKQ_0 += VKQ_stride) {
#pragma unroll
            for (int j = 0; j < ncols / frag_n; ++j) {
                wmma::fill_fragment(VKQ_c[i_VKQ_0 / VKQ_stride][j], static_cast<half>(0.0f));
            }

#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += VKQ_ratio * 16) {
                const int k = k0 + (threadIdx.y % VKQ_ratio) * 16;

                frag_a_V v_a;
                wmma::load_matrix_sync(v_a, V_h_f16 + int64_t(k_VKQ_0 + k) * stride_KV + i_VKQ_0 + frag_m * (threadIdx.y / VKQ_ratio), stride_KV);
#pragma unroll
                for (int j = 0; j < ncols / frag_n; ++j) {
                    wmma::mma_sync(VKQ_c[i_VKQ_0 / VKQ_stride][j], v_a, KQ_b[k0 / (VKQ_ratio * 16)][j], VKQ_c[i_VKQ_0 / VKQ_stride][j]);
                }
            }
        }

        __syncthreads();

        // --- Baseline lines 362-372: Store VKQ_c to shared memory ---
        const int offset_k = (threadIdx.y % VKQ_ratio) * (ncols * D_padded);
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += VKQ_stride) {
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                wmma::store_matrix_sync(
                    KQ_f16 + offset_k + j0 * D_padded + i_KQ_0 + frag_m * (threadIdx.y / VKQ_ratio),
                    VKQ_c[i_KQ_0 / VKQ_stride][j0 / frag_n],
                    D_padded, wmma::mem_col_major);
            }
        }

        __syncthreads();

        // --- Baseline lines 376-401: Accumulate VKQ_c parts, scale by softmax ---
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            half2 VKQ_scale;
            if (std::is_same<KQ_acc_t, float>::value) {
                VKQ_scale = make_half2(KQ_max_scale_f[j0 / nwarps], KQ_max_scale_f[j0 / nwarps]);
            } else {
                VKQ_scale = KQ_max_scale_h2[j0 / nwarps];
            }

#pragma unroll
            for (int i0 = 0; i0 < D / 2; i0 += warp_size) {
                const int i = i0 + threadIdx.x;
                if (i0 + warp_size > D / 2 && i >= D / 2) {
                    break;
                }

                half2 VKQ_add = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int l = 0; l < VKQ_ratio; ++l) {
                    VKQ_add += KQ2[l * (ncols * D_padded / 2) + j * (D_padded / 2) + i];
                }
                VKQ2[j * (D_padded / 2) + i] = VKQ_scale * VKQ2[j * (D_padded / 2) + i] + VKQ_add;
            }
        }

        __syncthreads();
    }

    // --- Baseline lines 406-452: Apply attention sinks ---
    // Baseline: if (sinksf && blockIdx.y == 0)
    // Megakernel: sinksf is always nullptr, but we keep the full branch for
    // register pressure matching and future extensibility.
    if (sinksf && blockIdx.y == 0) {
        const float sinkf = sinksf[head];
        const half  sinkh = __float2half(sinkf);

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (std::is_same<KQ_acc_t, float>::value) {
                float kqmax_new = fmaxf(KQ_max_f[j0 / nwarps], sinkf);

                const float KQ_max_scale = expf(KQ_max_f[j0 / nwarps] - kqmax_new);
                KQ_max_f[j0 / nwarps] = kqmax_new;

                KQ_rowsum_f[j0 / nwarps] = KQ_rowsum_f[j0 / nwarps] * KQ_max_scale + expf(sinkf - KQ_max_f[j0 / nwarps]);

                const half2 scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
                for (int i0 = 0; i0 < D / 2; i0 += warp_size) {
                    const int i = i0 + threadIdx.x;
                    if (i0 + warp_size > D / 2 && i >= D / 2) break;
                    VKQ2[j * (D_padded / 2) + i] *= scale_h2;
                }
            } else {
                half kqmax_old = __low2half(KQ_max_h2[j0 / nwarps]);
                half kqmax_new = __float2half(fmaxf(__half2float(kqmax_old), __half2float(sinkh)));
                KQ_max_h2[j0 / nwarps] = __halves2half2(kqmax_new, kqmax_new);

                const half  KQ_max_scale_h = hexp(__hsub(kqmax_old, kqmax_new));
                const half2 KQ_max_scale   = __halves2half2(KQ_max_scale_h, KQ_max_scale_h);

                KQ_rowsum_h2[j0 / nwarps] = KQ_rowsum_h2[j0 / nwarps] * KQ_max_scale;
                const half val = hexp(__hsub(sinkh, kqmax_new));
                KQ_rowsum_h2[j0 / nwarps].x = __hadd(KQ_rowsum_h2[j0 / nwarps].x, val);

#pragma unroll
                for (int i0 = 0; i0 < D / 2; i0 += warp_size) {
                    const int i = i0 + threadIdx.x;
                    if (i0 + warp_size > D / 2 && i >= D / 2) break;
                    VKQ2[j * (D_padded / 2) + i] *= KQ_max_scale;
                }
            }
        }

        __syncthreads();
    }

    // --- Baseline lines 453-494: Write output ---
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j_VKQ = j0 + threadIdx.y;
        if (ic0 + j_VKQ >= ne01_z) {
            return;
        }

        float KQ_rowsum_j;
        if (std::is_same<KQ_acc_t, float>::value) {
            KQ_rowsum_j = KQ_rowsum_f[j0 / nwarps];
        } else {
            KQ_rowsum_j = __low2float(KQ_rowsum_h2[j0 / nwarps]) + __high2float(KQ_rowsum_h2[j0 / nwarps]);
        }

        // Baseline: const int j_dst_unrolled = ((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y;
        // Megakernel: single block, output goes directly to dst[i].

#pragma unroll
        for (int i0 = 0; i0 < D; i0 += warp_size) {
            const int i = i0 + threadIdx.x;
            if (i0 + warp_size > D && i >= D) {
                break;
            }
            float dst_val = __half2float(VKQ[j_VKQ * D_padded + i]);
            // Baseline: if (gridDim.y == 1) { dst_val /= KQ_rowsum_j; }
            // Megakernel: always single block (gridDim.y == 1)
            dst_val /= KQ_rowsum_j;
            dst[i] = dst_val;
        }

        // Baseline lines 482-493: dst_meta for multi-block reduction
        // Megakernel: gridDim.y == 1, so this is skipped (baseline: if (gridDim.y == 1 || threadIdx.x != 0) continue;)
    }

    // Suppress unused-variable warnings for dead-code declarations
    (void)ic0; (void)sequence; (void)head; (void)gqa_ratio; (void)stride_Q;
    (void)slopeh; (void)slope2; (void)logit_softcap_2;
    (void)maskh; (void)mask2; (void)sinksf; (void)nb31; (void)ne11;
    (void)VKQ_f16; (void)ne01_z;
}

// ============================================================================
// Wrapper: keeps the original 2-template-parameter interface for decode.hip
// but internally dispatches to the 6-template-parameter impl that matches
// baseline's template structure.
// ============================================================================
template <int D, bool use_logit_softcap>
static __device__ void flash_attn_wmma_f16(
        const float  * __restrict__ Q_f,
        const half   * __restrict__ K_h,
        const half   * __restrict__ V_h,
        float        * __restrict__ dst,
        const int kv_len,
        const float scale,
        const float alibi_slope,
        const int   cur_pos,
        const float logit_softcap) {

    constexpr int ncols     = 16;
    constexpr int nwarps    = 4;
    constexpr int frag_m    = 16;
    constexpr int VKQ_stride_val = wmma_get_VKQ_stride(D, nwarps, frag_m);
    using KQ_acc_t = float;

    flash_attn_wmma_f16_impl<D, ncols, nwarps, VKQ_stride_val, KQ_acc_t, use_logit_softcap>(
        Q_f, K_h, V_h, dst, kv_len, scale, alibi_slope, cur_pos, logit_softcap);
}

#endif // GGML_USE_WMMA_FATTN
