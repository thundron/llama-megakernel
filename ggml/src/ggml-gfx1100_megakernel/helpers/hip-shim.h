// hip-shim.h — CUDA→HIP compatibility macros
// From ggml/src/ggml-cuda/vendors/hip.h (lines 37-39)
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>   // must come BEFORE __shfl_sync macros (ROCm defines bf16 shuffle overloads internally)

// CUDA sync shuffles → HIP (drop mask parameter)
// These macros must be defined AFTER all HIP headers that define __shfl_sync overloads.
#define __shfl_sync(mask, var, laneMask, width)     __shfl(var, laneMask, width)
#define __shfl_up_sync(mask, var, laneMask, width)   __shfl_up(var, laneMask, width)
#define __shfl_xor_sync(mask, var, laneMask, width)  __shfl_xor(var, laneMask, width)

// make_half2 → HIP equivalent
#define make_half2(a, b) __halves2half2(__float2half(a), __float2half(b))

// gfx1100 (RDNA3) is wave32
#define WARP_SIZE 32
