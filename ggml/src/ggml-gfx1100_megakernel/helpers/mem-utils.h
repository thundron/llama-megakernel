// mem-utils.h — aligned memory copy helpers
// Copied from ggml/src/ggml-cuda/common.cuh lines 352-362, 759-788
#pragma once

#include "hip-shim.h"

// --- ggml_cuda_get_max_cpy_bytes --- common.cuh:352-357
// HIP always supports 16-byte aligned copies
static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
    return 16;
}

// --- ggml_cuda_memcpy_1 --- common.cuh:759-788
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        }
    }
}
