#ifndef GGML_TURBOQ_H
#define GGML_TURBOQ_H

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Rotation functions
GGML_API void ggml_turboq_rotate_forward(float * y, const float * x, int64_t dim, uint64_t seed);
GGML_API void ggml_turboq_rotate_backward(float * x, const float * y, int64_t dim, uint64_t seed);

// Quantize functions
GGML_API void quantize_row_tbq3_0(const float * x, void * y, int64_t n);
GGML_API void quantize_row_tbq4_0(const float * x, void * y, int64_t n);

// Reference versions
GGML_API void quantize_row_tbq3_0_ref(const float * x, block_tbq3_0 * y, int64_t k);
GGML_API void quantize_row_tbq4_0_ref(const float * x, block_tbq4_0 * y, int64_t k);

#ifdef __cplusplus
}
#endif

#endif
