// dequant.h — bf16/f16 conversion helpers
// Copied from ggml-cuda common patterns
#pragma once

#include "hip-shim.h"

// bf16 ↔ f32
typedef uint16_t bf16_t;

static __device__ __forceinline__ float bf16_to_f32(bf16_t v) {
    uint32_t x = (uint32_t)v << 16;
    float f;
    __builtin_memcpy(&f, &x, 4);
    return f;
}

static __device__ __forceinline__ bf16_t f32_to_bf16(float f) {
    uint32_t x;
    __builtin_memcpy(&x, &f, 4);
    return (bf16_t)(x >> 16);
}
