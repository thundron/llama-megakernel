// shared/batch-ops.h — batch operation helpers shared between prompt and encoder paths
#pragma once
#include "../gfx1100-internal.h"

void launch_mmq_quantize(hipFunction_t fn, const float * input, void * output,
                          int hidden_size, int n_tokens, hipStream_t stream);

void launch_mmq_kernel(hipFunction_t fn, const void * weight, long long weight_stride,
                        const void * q8_input, float * output, int in_dim, int out_dim,
                        int n_tokens, size_t shared_mem, hipStream_t stream);

void batch_projection(
    int weight_type, const void * weight, long long weight_stride,
    const float * f32_input, const void * q8_input,
    float * output, int in_dim, int out_dim, int n_tokens, hipStream_t stream);

void batch_projection_residual(
    int weight_type, const void * weight, long long weight_stride,
    const float * f32_input, const void * q8_input,
    const float * residual, float * output, int in_dim, int out_dim, int n_tokens,
    hipStream_t stream);
