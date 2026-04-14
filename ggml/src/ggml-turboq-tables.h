#ifndef GGML_TURBOQ_TABLES_H
#define GGML_TURBOQ_TABLES_H

// Codebook values for TurboQuant
// These are derived from the optimal quantization points for Gaussian distributions

// 3-bit codebook: 8 values
extern const float tbq3_codebook[8];

// 4-bit codebook: 16 values
extern const float tbq4_codebook[16];

#endif
