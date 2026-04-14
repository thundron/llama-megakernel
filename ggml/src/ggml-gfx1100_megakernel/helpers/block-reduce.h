// block-reduce.h — block-level reduction (used by l2-norm, rms-norm)
// Copied from ggml/src/ggml-cuda/common.cuh lines 540-618
#pragma once

#include "hip-shim.h"
#include "warp-reduce.h"

// --- block_reduce_method enum --- common.cuh:540-543
enum class block_reduce_method {
    MAX,
    SUM,
};

// --- Type trait helper ---
template <typename T>
inline constexpr bool ggml_cuda_dependent_false_v = false;

template <typename T, typename... Ts>
inline constexpr bool is_any = (std::is_same_v<T, Ts> || ...);

// --- block_reduce_policy --- common.cuh:545-596
template <block_reduce_method method, typename T>
struct block_reduce_policy;

template <typename T>
struct block_reduce_policy<block_reduce_method::SUM, T> {
    static __device__ T reduce(T val) {
        return warp_reduce_sum(val);
    }
    static __device__ T sentinel() {
        if constexpr (std::is_same_v<T, float>) {
            return 0.0f;
        } else if constexpr (std::is_same_v<T, float2>) {
            return make_float2(0.0f, 0.0f);
        } else if constexpr (std::is_same_v<T, int>) {
            return 0;
        } else {
            static_assert(ggml_cuda_dependent_false_v<T>, "unsupported type");
            return T{};
        }
    }
};

template <typename T>
struct block_reduce_policy<block_reduce_method::MAX, T> {
    static __device__ T reduce(T val) {
        return warp_reduce_max(val);
    }
    static __device__ T sentinel() {
        if constexpr (std::is_same_v<T, float>) {
            return -INFINITY;
        } else {
            static_assert(ggml_cuda_dependent_false_v<T>, "unsupported type");
            return T{};
        }
    }
};

// --- block_reduce --- common.cuh:598-618
template <block_reduce_method reduce_method_t, const unsigned int block_size_template = 0, typename T>
static __device__ T block_reduce(T val, T * shared_vals) {
    val = block_reduce_policy<reduce_method_t, T>::reduce(val);
    const unsigned int block_size = block_size_template == 0 ? blockDim.x : block_size_template;
    if (block_size > WARP_SIZE) {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        __syncthreads();
        val = block_reduce_policy<reduce_method_t, T>::sentinel();
        if (lane_id < (static_cast<int>(block_size) / WARP_SIZE)) {
            val = shared_vals[lane_id];
        }
        return block_reduce_policy<reduce_method_t, T>::reduce(val);
    }
    return val;
}
