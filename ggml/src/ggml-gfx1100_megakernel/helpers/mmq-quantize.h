#pragma once
// Port of ggml/src/ggml-cuda/quantize.cu lines 175-271 — MMQ Q8_1 batch quantization
// Struct/enum from ggml/src/ggml-cuda/mmq.cuh lines 22-96
// For use with MMQ tiled GEMM kernels
#include "hip-shim.h"
#include "quant-types.h"

// From ggml/src/ggml-cuda/quantize.cuh line 9
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

// =============================================================================
// MMQ Q8_1 ds layout enum — from mmq.cuh lines 22-26
// =============================================================================
#ifndef MMQ_Q8_1_DS_LAYOUT_DEFINED
#define MMQ_Q8_1_DS_LAYOUT_DEFINED

enum mmq_q8_1_ds_layout {
    MMQ_Q8_1_DS_LAYOUT_D4,
    MMQ_Q8_1_DS_LAYOUT_DS4,
    MMQ_Q8_1_DS_LAYOUT_D2S6,
};

// =============================================================================
// MMQ Q8_1 block struct — from mmq.cuh lines 28-47
// The y float data is converted to a data layout that can simply be copied to
// shared memory as a contiguous block.
// The y float data is first grouped as blocks of 128 values.
// These blocks are then treated as individual data values and transposed.
//
// To avoid shared memory bank conflicts each block is padded with 16 bytes.
// This padding is also used to store block scales/partial sums.
// The scales multiplied with the quantized data are equal to the unquantized values.
// The partial sums are obtained by summing up a subgroup of the contained values
//     (prior to quantization) and are only needed for performance reasons.
//
// The exact data stored depends on the x data type.
// =============================================================================
struct block_q8_1_mmq {
    union {
        float d4[4];    // 1 32 bit scale per 32 values, stored as d0,d1,d2,d3
        __half2 ds4[4]; // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored as d0,s0,d1,s1,d2,s2,d3,s3
        __half  d2s6[8]; // 1 16 bit scale per 64 values + 1 16 bit partial sum per 16 values for the first 96 values,
                         //     stored as d0,d1,s1,s2,s3,s4,s5
    };
    int8_t qs[4*QK8_1]; // 128 values quantized to 8 bit each
};

static_assert(sizeof(block_q8_1_mmq) == 4*QK8_1 + 4*sizeof(__half2), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4*sizeof(block_q8_1),        "Unexpected block_q8_1_mmq size");

#endif // MMQ_Q8_1_DS_LAYOUT_DEFINED

// =============================================================================
// quantize_mmq_q8_1 kernel template — from quantize.cu lines 175-271
// Template parameter selects scale/sum storage layout.
// Each thread processes 4 consecutive float values.
// =============================================================================
template <mmq_q8_1_ds_layout ds_layout>
static __device__ void quantize_mmq_q8_1_impl(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        const float d = 1.0f / d_inv;

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}
