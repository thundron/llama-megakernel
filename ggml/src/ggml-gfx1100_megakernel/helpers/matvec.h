// matvec.h — Quantized matrix-vector multiply for ALL types
// EXACT PORT from ggml/src/ggml-cuda/mmvq.cu mul_mat_vec_q (lines 391-589)
// Stripped to: ncols_dst=1, has_fusion=false, no ids, single channel/sample.
// Uses 2D thread block (warp_size, nwarps) matching baseline EXACTLY.
// Generic table (nwarps=4) for compatibility + RDNA3 table (nwarps=8) for selected types.
#pragma once

#include "hip-shim.h"
#include "warp-reduce.h"
#include "quant-types.h"
#include "vec-dot.h"
#include "activations.h"

// ============================================================================
// Generic matvec row template — from baseline mmvq.cu mul_mat_vec_q
// Parameterized on qk, qi, vdr, vec_dot function.
// GENERIC table: nwarps=4, rows_per_block=1 (works on all AMD GPUs).
// ============================================================================

typedef float (*vec_dot_fn_t)(const void *, const block_q8_1 *, const int &, const int &);

// GLU operation types (matching baseline's ggml_glu_op)
enum megakernel_glu_op {
    MK_GLU_NONE    = -1,
    MK_GLU_SWIGLU  = 0,  // result *= silu(gate)
    MK_GLU_GEGLU   = 1,  // result *= gelu(gate)
};

// Matvec template — exact port of baseline mmvq.cu mul_mat_vec_q.
// Matches baseline's register/shared memory layout INCLUDING tmp_gate and
// tmp_shared_gate, which are always declared (affecting register pressure
// and instruction scheduling). This produces bit-exact results.
//
// When gate_weight != nullptr: fused gate+up+activation in one kernel
// (reads Q8 input once, computes both projections, applies GLU at output).
template <int QK_T, int QI_T, int VDR_T, vec_dot_fn_t VEC_DOT, int NWARPS_T = 4>
__device__ __forceinline__ void mmvq_generic_row(
        const void       * __restrict__ vx,
        const void       * __restrict__ vy,
        float            * __restrict__ dst,
        const uint32_t ncols_x,
        const uint32_t stride_row_x,
        const int row0,
        const void       * __restrict__ gate_weight = nullptr,
        const int glu_op = MK_GLU_NONE) {

    constexpr int qk  = QK_T;
    constexpr int qi  = QI_T;
    constexpr int vdr = VDR_T;
    constexpr int nwarps = NWARPS_T;
    constexpr int rows_per_cuda_block = 1;
    constexpr int warp_size = WARP_SIZE;
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;

    const int tid = warp_size * threadIdx.y + threadIdx.x;
    const int blocks_per_row_x = ncols_x / qk;

    // Partial sums — always declare both to match baseline register layout.
    // tmp_gate is used when gate_weight != nullptr (fused gate+up mode).
    float tmp[rows_per_cuda_block] = {0.0f};
    float tmp_gate[rows_per_cuda_block] = {0.0f};

    const bool use_gate = (gate_weight != nullptr);

    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int kbx_offset = row0 * stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp[i] += VEC_DOT(vx, &y[kby], kbx_offset + i * stride_row_x + kbx, kqs);
            if (use_gate) {
                tmp_gate[i] += VEC_DOT(gate_weight, &y[kby], kbx_offset + i * stride_row_x + kbx, kqs);
            }
        }
    }

    // Shared memory for inter-warp reduction — both always declared
    __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][1][rows_per_cuda_block][warp_size];
    __shared__ float tmp_shared_gate[nwarps - 1 > 0 ? nwarps - 1 : 1][1][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp_shared[threadIdx.y - 1][0][i][threadIdx.x] = tmp[i];
            if (use_gate) {
                tmp_shared_gate[threadIdx.y - 1][0][i][threadIdx.x] = tmp_gate[i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
        for (int l = 0; l < nwarps - 1; ++l) {
            tmp[i] += tmp_shared[l][0][i][threadIdx.x];
            if (use_gate) {
                tmp_gate[i] += tmp_shared_gate[l][0][i][threadIdx.x];
            }
        }
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if (use_gate) {
            tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
        }
    }

    if (threadIdx.x < rows_per_cuda_block) {
        float result = tmp[threadIdx.x];
        if (use_gate) {
            float gate_value = tmp_gate[threadIdx.x];
            if (glu_op == MK_GLU_SWIGLU) {
                result *= op_silu(gate_value);
            } else if (glu_op == MK_GLU_GEGLU) {
                result *= op_gelu(gate_value);
            }
        }
        dst[row0 + threadIdx.x] = result;
    }
}

// ============================================================================
// Type-specific instantiations (aliases for readability)
// ============================================================================

// Small-block types (QK=32)
#define MMVQ_ROW_SMALL(name, qi, vdr, vec_dot_fn) \
    __device__ __forceinline__ void name( \
        const void * vx, const void * vy, float * dst, \
        uint32_t ncols_x, uint32_t stride_row_x, int row0) { \
        mmvq_generic_row<32, qi, vdr, vec_dot_fn>(vx, vy, dst, ncols_x, stride_row_x, row0); \
    }

// K-quant types (QK=256)
#define MMVQ_ROW_KQUANT(name, qi, vdr, vec_dot_fn) \
    __device__ __forceinline__ void name( \
        const void * vx, const void * vy, float * dst, \
        uint32_t ncols_x, uint32_t stride_row_x, int row0) { \
        mmvq_generic_row<QK_K, qi, vdr, vec_dot_fn>(vx, vy, dst, ncols_x, stride_row_x, row0); \
    }

MMVQ_ROW_SMALL(mmvq_q4_0_row, QI4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1)
MMVQ_ROW_SMALL(mmvq_q4_1_row, QI4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1)
MMVQ_ROW_SMALL(mmvq_q5_0_row, QI5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1)
MMVQ_ROW_SMALL(mmvq_q5_1_row, QI5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1)
MMVQ_ROW_SMALL(mmvq_q8_0_row, QI8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1)
MMVQ_ROW_KQUANT(mmvq_q2k_row, QI2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)
MMVQ_ROW_KQUANT(mmvq_q3k_row, QI3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1)
MMVQ_ROW_KQUANT(mmvq_q5k_row, QI5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1)

// IQ types (QK=256, various qi/vdr)
MMVQ_ROW_KQUANT(mmvq_iq2_xxs_row, QI2_XXS, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1)
MMVQ_ROW_KQUANT(mmvq_iq2_xs_row,  QI2_XS,  VDR_IQ2_XS_Q8_1_MMVQ,  vec_dot_iq2_xs_q8_1)
MMVQ_ROW_KQUANT(mmvq_iq2_s_row,   QI2_S,   VDR_IQ2_S_Q8_1_MMVQ,   vec_dot_iq2_s_q8_1)
MMVQ_ROW_KQUANT(mmvq_iq3_xxs_row, QI3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1)
MMVQ_ROW_KQUANT(mmvq_iq3_s_row,   QI3_S,   VDR_IQ3_S_Q8_1_MMVQ,   vec_dot_iq3_s_q8_1)
MMVQ_ROW_KQUANT(mmvq_iq1_s_row,   QI1_S,   VDR_IQ1_S_Q8_1_MMVQ,   vec_dot_iq1_s_q8_1)
MMVQ_ROW_KQUANT(mmvq_iq1_m_row,   QI1_M,   VDR_IQ1_M_Q8_1_MMVQ,   vec_dot_iq1_m_q8_1)
MMVQ_ROW_KQUANT(mmvq_iq4_xs_row,  QI4_XS,  VDR_IQ4_XS_Q8_1_MMVQ,  vec_dot_iq4_xs_q8_1)

// IQ4_NL (QK=32)
MMVQ_ROW_SMALL(mmvq_iq4_nl_row, QI4_NL, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1)

// ============================================================================
// 8-warp variants for RDNA3 (gfx1100) — baseline uses nwarps=8 for ncols_dst=1 decode
// Only for types that benefit on RDNA3 (from baseline's MMVQ_PARAMETERS_RDNA3_0 table):
// Q4_K, Q6_K, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL
// ============================================================================

#define MMVQ_ROW_SMALL_8W(name, qi, vdr, vec_dot_fn) \
    __device__ __forceinline__ void name( \
        const void * vx, const void * vy, float * dst, \
        uint32_t ncols_x, uint32_t stride_row_x, int row0) { \
        mmvq_generic_row<32, qi, vdr, vec_dot_fn, 8>(vx, vy, dst, ncols_x, stride_row_x, row0); \
    }

#define MMVQ_ROW_KQUANT_8W(name, qi, vdr, vec_dot_fn) \
    __device__ __forceinline__ void name( \
        const void * vx, const void * vy, float * dst, \
        uint32_t ncols_x, uint32_t stride_row_x, int row0) { \
        mmvq_generic_row<QK_K, qi, vdr, vec_dot_fn, 8>(vx, vy, dst, ncols_x, stride_row_x, row0); \
    }

MMVQ_ROW_SMALL_8W(mmvq_q4_0_row_8w,  QI4_0,  VDR_Q4_0_Q8_1_MMVQ,  vec_dot_q4_0_q8_1)
MMVQ_ROW_SMALL_8W(mmvq_q4_1_row_8w,  QI4_1,  VDR_Q4_1_Q8_1_MMVQ,  vec_dot_q4_1_q8_1)
MMVQ_ROW_SMALL_8W(mmvq_q5_0_row_8w,  QI5_0,  VDR_Q5_0_Q8_1_MMVQ,  vec_dot_q5_0_q8_1)
MMVQ_ROW_SMALL_8W(mmvq_q5_1_row_8w,  QI5_1,  VDR_Q5_1_Q8_1_MMVQ,  vec_dot_q5_1_q8_1)
MMVQ_ROW_SMALL_8W(mmvq_q8_0_row_8w,  QI8_0,  VDR_Q8_0_Q8_1_MMVQ,  vec_dot_q8_0_q8_1)
MMVQ_ROW_SMALL_8W(mmvq_iq4_nl_row_8w, QI4_NL, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1)
MMVQ_ROW_KQUANT_8W(mmvq_q4k_row_8w,  QI4_K,  2, vec_dot_q4_K_q8_1)
MMVQ_ROW_KQUANT_8W(mmvq_q6k_row_8w,  QI6_K,  1, vec_dot_q6_K_q8_1)

// MXFP4 (QK=32) + NVFP4 (QK=64)
MMVQ_ROW_SMALL(mmvq_mxfp4_row, QI_MXFP4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1)

// NVFP4 needs QK=64 specialization — wrap to match vec_dot_fn_t signature
static __device__ __forceinline__ float vec_dot_nvfp4_q8_1_wrap(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return vec_dot_nvfp4_q8_1(vbq, bq8_1, kbx, iqs);
}
__device__ __forceinline__ void mmvq_nvfp4_row(
    const void * vx, const void * vy, float * dst,
    uint32_t ncols_x, uint32_t stride_row_x, int row0) {
    mmvq_generic_row<QK_NVFP4, QI_NVFP4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1_wrap>(
        vx, vy, dst, ncols_x, stride_row_x, row0);
}

// ============================================================================
// Legacy Q4_K/Q6_K row functions (original implementations kept for compatibility)
// ============================================================================

// Baseline constants for Q4_K, ncols_dst=1
// Sources: ggml-common.h, common.cuh, vecdotq.cuh, mmvq.cu
//
// From ggml-common.h:
//   QK_K = 256
//   QR4_K = 2
//   QI4_K = QK_K / (4 * QR4_K) = 256 / (4 * 2) = 32
//   QK8_1 = 32
//   QI8_1 = QK8_1 / 4 = 8
//
// From vecdotq.cuh:
//   VDR_Q4_K_Q8_1_MMVQ = 2
//
// From mmvq.cu (GENERIC table, ncols_dst=1):
//   nwarps = 4
//   rows_per_cuda_block = 1
//
// Derived (mmvq.cu line 412):
//   blocks_per_iter = vdr * nwarps * warp_size / qi = 2 * 4 * 32 / 32 = 8
//
// Thread mapping (128 threads, tid 0..127):
//   kbx  = tid / (qi/vdr) = tid / 16   -> initial block index
//   kqs  = vdr * (tid % (qi/vdr)) = 2 * (tid % 16)  -> quant sub-index
constexpr int MMVQ_QK   = QK_K;                   // 256
constexpr int MMVQ_QI   = QI4_K;                  // 32  (= QK_K / (4*QR4_K))
constexpr int MMVQ_VDR  = 2;                      // VDR_Q4_K_Q8_1_MMVQ
constexpr int MMVQ_NWARPS = 4;
constexpr int MMVQ_ROWS_PER_BLOCK = 1;
constexpr int MMVQ_WARP_SIZE = WARP_SIZE;          // 32

// Exact port of mul_mat_vec_q from mmvq.cu lines 391-589
// Simplified: type=Q4_K, ncols_dst=1, rows_per_cuda_block=1, has_fusion=false, no ids
// Thread block: dim3(WARP_SIZE, MMVQ_NWARPS, 1) = dim3(32, 4, 1) = 128 threads
// Grid: (nrows, 1, 1) — one block per output row
//
// IMPORTANT: launch with dim3(WARP_SIZE, MMVQ_NWARPS, 1) block size!
__device__ __forceinline__ void mmvq_q4k_row(
        const void       * __restrict__ vx,
        const void       * __restrict__ vy,
        float            * __restrict__ dst,
        const uint32_t ncols_x,
        const uint32_t stride_row_x,
        const int row0) {

    // mmvq.cu lines 399-412: constexpr setup
    constexpr int qk  = MMVQ_QK;                  // 256
    constexpr int qi  = MMVQ_QI;                   // 32
    constexpr int vdr = MMVQ_VDR;                  // 2
    constexpr int nwarps = MMVQ_NWARPS;            // 4
    constexpr int rows_per_cuda_block = MMVQ_ROWS_PER_BLOCK; // 1
    constexpr int warp_size = MMVQ_WARP_SIZE;      // 32
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi; // 2*4*32/32 = 8

    // mmvq.cu line 409
    const int tid = warp_size * threadIdx.y + threadIdx.x;
    // mmvq.cu line 411
    const int blocks_per_row_x = ncols_x / qk;

    // mmvq.cu line 475: partial sum for each thread
    float tmp[rows_per_cuda_block] = {0.0f};

    // mmvq.cu line 478-479: y pointer and kbx_offset
    // (simplified: no sample/channel offsets since single channel/sample)
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int kbx_offset = row0 * stride_row_x;

    // mmvq.cu lines 481-501: main accumulation loop
    // With qi=32, vdr=2: kbx starts at tid/16, kqs = 2*(tid%16), stride = 8
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);       // y block index aligned with kbx
        const int kqs = vdr * (tid % (qi/vdr));    // x block quant index

#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp[i] += vec_dot_q4_K_q8_1(
                vx, &y[kby], kbx_offset + i * stride_row_x + kbx, kqs);
        }
    }

    // mmvq.cu line 503: shared memory for inter-warp reduction
    __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][1][rows_per_cuda_block][warp_size];

    // mmvq.cu lines 511-524: warps 1-3 write to shared memory
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp_shared[threadIdx.y - 1][0][i][threadIdx.x] = tmp[i];
        }
    }
    // mmvq.cu line 525
    __syncthreads();
    // mmvq.cu lines 526-528: warps 1-3 exit
    if (threadIdx.y > 0) return;

    // mmvq.cu lines 533-546: warp 0 accumulates partial sums + warp reduce
#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
        for (int l = 0; l < nwarps - 1; ++l) {
            tmp[i] += tmp_shared[l][0][i][threadIdx.x];
        }
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    // mmvq.cu line 554+582: write result (no fusion)
    if (threadIdx.x < rows_per_cuda_block) {
        dst[row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

// ============================================================================
// Q6_K variant — same structure, different constants and vec_dot function
// From baseline: qi=32, vdr=1, blocks_per_iter = 1*4*32/32 = 4
// ============================================================================

constexpr int MMVQ_Q6K_VDR = 1;  // VDR_Q6_K_Q8_1_MMVQ

__device__ __forceinline__ void mmvq_q6k_row(
        const void       * __restrict__ vx,
        const void       * __restrict__ vy,
        float            * __restrict__ dst,
        const uint32_t ncols_x,
        const uint32_t stride_row_x,
        const int row0) {

    constexpr int qk  = MMVQ_QK;               // 256
    constexpr int qi  = MMVQ_QI;               // 32 (same as Q4_K)
    constexpr int vdr = MMVQ_Q6K_VDR;          // 1 (different from Q4_K which is 2)
    constexpr int nwarps = MMVQ_NWARPS;         // 4
    constexpr int rows_per_cuda_block = MMVQ_ROWS_PER_BLOCK; // 1
    constexpr int warp_size = MMVQ_WARP_SIZE;   // 32
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi; // 1*4*32/32 = 4

    const int tid = warp_size * threadIdx.y + threadIdx.x;
    const int blocks_per_row_x = ncols_x / qk;

    float tmp[rows_per_cuda_block] = {0.0f};

    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int kbx_offset = row0 * stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp[i] += vec_dot_q6_K_q8_1(
                vx, &y[kby], kbx_offset + i * stride_row_x + kbx, kqs);
        }
    }

    __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][1][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp_shared[threadIdx.y - 1][0][i][threadIdx.x] = tmp[i];
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

    for (int i = 0; i < rows_per_cuda_block; ++i) {
        for (int l = 0; l < nwarps - 1; ++l) {
            tmp[i] += tmp_shared[l][0][i][threadIdx.x];
        }
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    if (threadIdx.x < rows_per_cuda_block) {
        dst[row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}
