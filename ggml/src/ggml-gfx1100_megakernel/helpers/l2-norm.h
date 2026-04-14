// l2-norm.h — L2 normalization
// Copied from ggml/src/ggml-cuda/norm.cu lines 239-271
// Used by DeltaNet layers for Q/K normalization.
#pragma once

#include "hip-shim.h"
#include "block-reduce.h"

// scale = rsqrt(max(sum(x^2), eps^2)), dst[i] = scale * x[i]
// smem: caller-provided shared memory, at least [blockDim.x / WARP_SIZE] floats
static __device__ void l2_norm_f32_device(
        const float * __restrict__ x,
        float       * __restrict__ dst,
        float       * __restrict__ smem,
        const int ncols,
        const float eps) {

    const int tid = threadIdx.x;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += blockDim.x) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM>(tmp, smem);

    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += blockDim.x) {
        dst[col] = scale * x[col];
    }
}
