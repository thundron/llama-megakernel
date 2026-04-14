/*
 * TurboQuant (TBQ) implementation
 * From PR #21089: https://github.com/ggml-org/llama.cpp/pull/21089
 */

#include "ggml-turboq.h"
#include "ggml.h"
#include "ggml-turboq-tables.h"
#include "ggml-impl.h"

#include <float.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Deterministic random number generator
// ============================================================================

static inline uint64_t tbq_rand_next(uint64_t *state) {
    *state = (*state * 6364136223846793005ULL + 1442695040888963407ULL);
    return *state;
}

static inline float tbq_rand_uniform(uint64_t *state) {
    return (float)(tbq_rand_next(state) % 1000000) / 1000000.0f;
}

// ============================================================================
// Rotation matrix using Givens rotations
// ============================================================================

static void tbq_generate_rotation_givens(float *Q, uint64_t seed) {
    const int dim = 128;
    
    // Start with identity matrix
    for (int i = 0; i < dim * dim; i++) {
        Q[i] = (i % (dim + 1) == 0) ? 1.0f : 0.0f;
    }
    
    uint64_t state = seed;
    
    // Apply random Givens rotations
    for (int iter = 0; iter < dim * dim / 2; iter++) {
        int i = tbq_rand_next(&state) % dim;
        int j = tbq_rand_next(&state) % dim;
        if (i == j) continue;
        
        float theta = tbq_rand_uniform(&state) * 2.0f * (float)M_PI;
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);
        
        float *row_i = Q + i * dim;
        float *row_j = Q + j * dim;
        
        float *new_row_i = (float *)malloc(dim * sizeof(float));
        float *new_row_j = (float *)malloc(dim * sizeof(float));
        
        for (int k = 0; k < dim; k++) {
            new_row_i[k] = cos_t * row_i[k] - sin_t * row_j[k];
            new_row_j[k] = sin_t * row_i[k] + cos_t * row_j[k];
        }
        
        for (int k = 0; k < dim; k++) {
            row_i[k] = new_row_i[k];
            row_j[k] = new_row_j[k];
        }
        
        free(new_row_i);
        free(new_row_j);
    }
}

// Global rotation matrix
static float *tbq_rotation_matrix = NULL;
static int tbq_rotation_initialized = 0;

static void tbq_init_rotation(void) {
    if (!tbq_rotation_initialized) {
        tbq_rotation_matrix = (float *)malloc(128 * 128 * sizeof(float));
        tbq_generate_rotation_givens(tbq_rotation_matrix, 0x517cc1b727220a95ULL);
        tbq_rotation_initialized = 1;
    }
}

// Forward rotation: y = Q @ x
void ggml_turboq_rotate_forward(float * y, const float * x, int64_t dim, uint64_t seed) {
    (void)seed; // unused - we use global rotation matrix
    tbq_init_rotation();
    
    for (int i = 0; i < dim; i++) {
        y[i] = 0.0f;
        for (int j = 0; j < dim; j++) {
            y[i] += tbq_rotation_matrix[i * dim + j] * x[j];
        }
    }
}

// Backward rotation: x = Q^T @ y  
void ggml_turboq_rotate_backward(float * x, const float * y, int64_t dim, uint64_t seed) {
    (void)seed;
    tbq_init_rotation();
    
    for (int i = 0; i < dim; i++) {
        x[i] = 0.0f;
        for (int j = 0; j < dim; j++) {
            x[i] += tbq_rotation_matrix[j * dim + i] * y[j];
        }
    }
}

// ============================================================================
// Quantize functions
// ============================================================================

static inline int tbq3_find_nearest(float v) {
    int best_idx = 0;
    float best_diff = FLT_MAX;
    for (int i = 0; i < 8; i++) {
        float diff = fabsf(v - tbq3_codebook[i]);
        if (diff < best_diff) {
            best_diff = diff;
            best_idx = i;
        }
    }
    return best_idx;
}

static inline int tbq4_find_nearest(float v) {
    int best_idx = 0;
    float best_diff = FLT_MAX;
    for (int i = 0; i < 16; i++) {
        float diff = fabsf(v - tbq4_codebook[i]);
        if (diff < best_diff) {
            best_diff = diff;
            best_idx = i;
        }
    }
    return best_idx;
}

void quantize_row_tbq3_0(const float * x, void * y, int64_t n) {
    block_tbq3_0 * block = (block_tbq3_0 *)y;
    int64_t nb = n / 256;
    
    float *rotated = (float *)malloc(128 * sizeof(float));
    
    for (int64_t i = 0; i < nb; i++) {
        const float *x_block = x + i * 256;
        
        float norm = 0.0f;
        for (int64_t j = 0; j < 256; j++) {
            norm += x_block[j] * x_block[j];
        }
        norm = sqrtf(norm);
        if (norm < 1e-8f) norm = 1e-8f;
        
        // Rotate and quantize first half
        for (int64_t j = 0; j < 128; j++) {
            rotated[j] = x_block[j] / norm;
        }
        ggml_turboq_rotate_forward(rotated, rotated, 128, 0);
        
        int32_t indices[256];
        for (int64_t j = 0; j < 128; j++) {
            indices[j] = tbq3_find_nearest(rotated[j]);
        }
        
        // Rotate and quantize second half
        for (int64_t j = 0; j < 128; j++) {
            rotated[j] = x_block[128 + j] / norm;
        }
        ggml_turboq_rotate_forward(rotated, rotated, 128, 0);
        
        for (int64_t j = 0; j < 128; j++) {
            indices[128 + j] = tbq3_find_nearest(rotated[j]);
        }
        
        // Pack 3-bit indices
        for (int k = 0; k < 256; k += 8) {
            int idx = k / 8;
            block->qs[idx * 3 + 0] = indices[k] | (indices[k + 1] << 3) | (indices[k + 2] >> 1);
            block->qs[idx * 3 + 1] = (indices[k + 2] << 7) | (indices[k + 3] << 4) | (indices[k + 4] >> 2);
            block->qs[idx * 3 + 2] = (indices[k + 4] << 6) | (indices[k + 5] << 3) | (indices[k + 6] >> 3);
            block->qs[idx * 3 + 3] = (indices[k + 6] << 5) | indices[k + 7];
        }
        
        block->d = GGML_FP32_TO_FP16(norm);
        block++;
    }
    
    free(rotated);
    
    // Handle remainder
    if (n % 256 > 0) {
        block->d = GGML_FP32_TO_FP16(0.0f);
        memset(block->qs, 0, sizeof(block->qs));
    }
}

void quantize_row_tbq4_0(const float * x, void * y, int64_t n) {
    block_tbq4_0 * block = (block_tbq4_0 *)y;
    int64_t nb = n / 256;
    
    float *rotated = (float *)malloc(128 * sizeof(float));
    
    for (int64_t i = 0; i < nb; i++) {
        const float *x_block = x + i * 256;
        
        float norm = 0.0f;
        for (int64_t j = 0; j < 256; j++) {
            norm += x_block[j] * x_block[j];
        }
        norm = sqrtf(norm);
        if (norm < 1e-8f) norm = 1e-8f;
        
        for (int64_t j = 0; j < 128; j++) {
            rotated[j] = x_block[j] / norm;
        }
        ggml_turboq_rotate_forward(rotated, rotated, 128, 0);
        
        int32_t indices[256];
        for (int64_t j = 0; j < 128; j++) {
            indices[j] = tbq4_find_nearest(rotated[j]);
        }
        
        for (int64_t j = 0; j < 128; j++) {
            rotated[j] = x_block[128 + j] / norm;
        }
        ggml_turboq_rotate_forward(rotated, rotated, 128, 0);
        
        for (int64_t j = 0; j < 128; j++) {
            indices[128 + j] = tbq4_find_nearest(rotated[j]);
        }
        
        // Pack 4-bit indices
        for (int j = 0; j < 256; j += 2) {
            block->qs[j / 2] = indices[j] | (indices[j + 1] << 4);
        }
        
        block->d = GGML_FP32_TO_FP16(norm);
        block++;
    }
    
    free(rotated);
    
    if (n % 256 > 0) {
        block->d = GGML_FP32_TO_FP16(0.0f);
        memset(block->qs, 0, sizeof(block->qs));
    }
}

// Reference versions
void quantize_row_tbq3_0_ref(const float * x, block_tbq3_0 * y, int64_t n) {
    quantize_row_tbq3_0(x, (void *)y, n);
}

void quantize_row_tbq4_0_ref(const float * x, block_tbq4_0 * y, int64_t n) {
    quantize_row_tbq4_0(x, (void *)y, n);
}
