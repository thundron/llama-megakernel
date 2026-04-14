// quant-types.h — ALL quantization block struct definitions
// Copied VERBATIM from ggml/src/ggml-common.h, adapted ggml_half → __half for HIP
#pragma once

#include "hip-shim.h"

// ============================================================================
// ggml_type enum — device-side copy from ggml.h:389-430
// Only the quant types needed by MMQ kernels.
// ============================================================================
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    GGML_TYPE_MXFP4   = 39,
    GGML_TYPE_NVFP4   = 40,
    GGML_TYPE_COUNT,
};

// ============================================================================
// Constants from ggml-common.h:89-167
// ============================================================================

#define QK_K 256
#define K_SCALE_SIZE 12

#define QK1_0 128
#define QI1_0 (QK1_0 / 32)
#define QR1_0 1

#define QK4_0 32
#define QI4_0 (QK4_0 / (4 * QR4_0))
#define QR4_0 2

#define QK4_1 32
#define QI4_1 (QK4_1 / (4 * QR4_1))
#define QR4_1 2

#define QK_MXFP4 32
#define QI_MXFP4 (QK_MXFP4 / (4 * QR_MXFP4))
#define QR_MXFP4 2

#define QK_NVFP4 64
#define QK_NVFP4_SUB 16
#define QI_NVFP4 (QK_NVFP4 / (4 * QR_NVFP4))
#define QR_NVFP4 2

#define QK5_0 32
#define QI5_0 (QK5_0 / (4 * QR5_0))
#define QR5_0 2

#define QK5_1 32
#define QI5_1 (QK5_1 / (4 * QR5_1))
#define QR5_1 2

#define QK8_0 32
#define QI8_0 (QK8_0 / (4 * QR8_0))
#define QR8_0 1

#define QK8_1 32
#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QR8_1 1

#define QI2_K (QK_K / (4*QR2_K))
#define QR2_K 4

#define QI3_K (QK_K / (4*QR3_K))
#define QR3_K 4

#define QI4_K (QK_K / (4*QR4_K))
#define QR4_K 2

#define QI5_K (QK_K / (4*QR5_K))
#define QR5_K 2

#define QI6_K (QK_K / (4*QR6_K))
#define QR6_K 2

#define QI2_XXS (QK_K / (4*QR2_XXS))
#define QR2_XXS 4

#define QI2_XS (QK_K / (4*QR2_XS))
#define QR2_XS 4

#define QI2_S (QK_K / (4*QR2_S))
#define QR2_S 4

#define QI3_XXS (QK_K / (4*QR3_XXS))
#define QR3_XXS 4

#define QI3_S (QK_K / (4*QR3_S))
#define QR3_S 4

#define QI1_S (QK_K / (4*QR1_S))
#define QR1_S 8

#define QI1_M (QK_K / (4*QR1_M))
#define QR1_M 8

#define QK4_NL 32
#define QI4_NL (QK4_NL / (4*QR4_NL))
#define QR4_NL 2

#define QI4_XS (QK_K / (4*QR4_XS))
#define QR4_XS 2

// ============================================================================
// IQ special constants from ggml-common.h:1121-1122
// ============================================================================

#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f

#define IQ3S_N_SCALE (QK_K/64)

// ============================================================================
// Block structs — from ggml-common.h:184-450
// ============================================================================

// --- block_q4_0 --- ggml-common.h:184-189
struct block_q4_0 {
    __half   d;            // delta
    uint8_t  qs[QK4_0/2]; // nibbles / quants
};
static_assert(sizeof(block_q4_0) == sizeof(__half) + QK4_0/2, "wrong q4_0 block size");

// --- block_q4_1 --- ggml-common.h:191-202
struct block_q4_1 {
    __half2  dm;           // d and m packed as half2
    uint8_t  qs[QK4_1/2]; // nibbles / quants
};
static_assert(sizeof(block_q4_1) == sizeof(__half2) + QK4_1/2, "wrong q4_1 block size");

// --- block_mxfp4 --- ggml-common.h:204-209
struct block_mxfp4 {
    uint8_t e;              // E8M0 exponent
    uint8_t qs[QK_MXFP4/2];
};
static_assert(sizeof(block_mxfp4) == sizeof(uint8_t) + QK_MXFP4/2, "wrong mxfp4 block size");

// --- block_nvfp4 --- ggml-common.h:211-217
struct block_nvfp4 {
    uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; // UE4M3 scales (one per 16-element sub-block)
    uint8_t qs[QK_NVFP4/2];           // packed 4-bit E2M1 values
};
static_assert(sizeof(block_nvfp4) == sizeof(uint8_t)*(QK_NVFP4/QK_NVFP4_SUB) + QK_NVFP4/2, "wrong nvfp4 block size");

// --- block_q5_0 --- ggml-common.h:219-225
struct block_q5_0 {
    __half   d;            // delta
    uint8_t  qh[4];       // 5-th bit of quants
    uint8_t  qs[QK5_0/2]; // nibbles / quants
};
static_assert(sizeof(block_q5_0) == sizeof(__half) + sizeof(uint32_t) + QK5_0/2, "wrong q5_0 block size");

// --- block_q5_1 --- ggml-common.h:227-239
struct block_q5_1 {
    __half2  dm;           // d and m packed as half2
    uint8_t  qh[4];       // 5-th bit of quants
    uint8_t  qs[QK5_1/2]; // nibbles / quants
};
static_assert(sizeof(block_q5_1) == sizeof(__half2) + sizeof(uint32_t) + QK5_1/2, "wrong q5_1 block size");

// --- block_q8_0 --- ggml-common.h:241-246
struct block_q8_0 {
    __half  d;         // delta
    int8_t  qs[QK8_0]; // quants
};
static_assert(sizeof(block_q8_0) == sizeof(__half) + QK8_0, "wrong q8_0 block size");

// --- block_q8_1 --- ggml-common.h:248-259
struct block_q8_1 {
    __half2 ds;          // d = scale, s = d * sum(qs[i])
    int8_t  qs[QK8_1];  // quants
};
static_assert(sizeof(block_q8_1) == sizeof(__half2) + QK8_1, "wrong q8_1 block size");

// --- block_q2_K --- ggml-common.h:288-299
struct block_q2_K {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    __half2 dm;               // super-block scale + min packed
};
static_assert(sizeof(block_q2_K) == sizeof(__half2) + QK_K/16 + QK_K/4, "wrong q2_K block size");

// --- block_q3_K --- ggml-common.h:305-311
struct block_q3_K {
    uint8_t hmask[QK_K/8]; // quants - high bit
    uint8_t qs[QK_K/4];    // quants - low 2 bits
    uint8_t scales[12];    // scales, quantized with 6 bits
    __half  d;             // super-block scale
};
static_assert(sizeof(block_q3_K) == sizeof(__half) + QK_K/4 + QK_K/8 + 12, "wrong q3_K block size");

// --- block_q4_K --- ggml-common.h:317-328
struct block_q4_K {
    __half2  dm;                    // d and dmin (super-block scales)
    uint8_t  scales[K_SCALE_SIZE];  // scales and mins, quantized with 6 bits
    uint8_t  qs[QK_K/2];           // 4-bit quants
};
static_assert(sizeof(block_q4_K) == sizeof(__half2) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size");

// --- block_q5_K --- ggml-common.h:334-346
struct block_q5_K {
    __half2  dm;                    // d and dmin (super-block scales)
    uint8_t  scales[K_SCALE_SIZE];  // scales and mins, quantized with 6 bits
    uint8_t  qh[QK_K/8];           // quants, high bit
    uint8_t  qs[QK_K/2];           // quants, low 4 bits
};
static_assert(sizeof(block_q5_K) == sizeof(__half2) + K_SCALE_SIZE + QK_K/2 + QK_K/8, "wrong q5_K block size");

// --- block_q6_K --- ggml-common.h:352-358
struct block_q6_K {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    __half  d;               // super-block scale
};
static_assert(sizeof(block_q6_K) == sizeof(__half) + QK_K/16 + 3*QK_K/4, "wrong q6_K block size");

// --- block_iq2_xxs --- ggml-common.h:371-375
struct block_iq2_xxs {
    __half   d;
    uint16_t qs[QK_K/8];
};
static_assert(sizeof(block_iq2_xxs) == sizeof(__half) + QK_K/8*sizeof(uint16_t), "wrong iq2_xxs block size");

// --- block_iq2_xs --- ggml-common.h:378-383
struct block_iq2_xs {
    __half   d;
    uint16_t qs[QK_K/8];
    uint8_t  scales[QK_K/32];
};
static_assert(sizeof(block_iq2_xs) == sizeof(__half) + QK_K/8*sizeof(uint16_t) + QK_K/32, "wrong iq2_xs block size");

// --- block_iq2_s --- ggml-common.h:386-392
struct block_iq2_s {
    __half   d;
    uint8_t  qs[QK_K/4];
    uint8_t  qh[QK_K/32];
    uint8_t  scales[QK_K/32];
};
static_assert(sizeof(block_iq2_s) == sizeof(__half) + QK_K/4 + QK_K/16, "wrong iq2_s block size");

// --- block_iq3_xxs --- ggml-common.h:397-401
struct block_iq3_xxs {
    __half   d;
    uint8_t  qs[3*QK_K/8];
};
static_assert(sizeof(block_iq3_xxs) == sizeof(__half) + 3*(QK_K/8), "wrong iq3_xxs block size");

// --- block_iq3_s --- ggml-common.h:404-412
struct block_iq3_s {
    __half   d;
    uint8_t  qs[QK_K/4];
    uint8_t  qh[QK_K/32];
    uint8_t  signs[QK_K/8];
    uint8_t  scales[IQ3S_N_SCALE];
};
static_assert(sizeof(block_iq3_s) == sizeof(__half) + 13*(QK_K/32) + IQ3S_N_SCALE, "wrong iq3_s block size");

// --- block_iq1_s --- ggml-common.h:415-420
struct block_iq1_s {
    __half   d;
    uint8_t  qs[QK_K/8];
    uint16_t qh[QK_K/32];
};
static_assert(sizeof(block_iq1_s) == sizeof(__half) + QK_K/8 + QK_K/16, "wrong iq1_s block size");

// --- block_iq1_m --- ggml-common.h:423-428
struct block_iq1_m {
    uint8_t  qs[QK_K/8];      // grid index, low 8 bits
    uint8_t  qh[QK_K/16];     // grid index, high 3 bits + grid shift bit
    uint8_t  scales[QK_K/32]; // 3-bit block scales
};
static_assert(sizeof(block_iq1_m) == QK_K/8 + QK_K/16 + QK_K/32, "wrong iq1_m block size");

// --- iq1m_scale_t --- ggml-common.h:431-434
typedef union {
    __half   f16;
    uint16_t u16;
} iq1m_scale_t;

// --- block_iq4_nl --- ggml-common.h:437-442
struct block_iq4_nl {
    __half   d;
    uint8_t  qs[QK4_NL/2];
};
static_assert(sizeof(block_iq4_nl) == sizeof(__half) + QK4_NL/2, "wrong iq4_nl block size");

// --- block_iq4_xs --- ggml-common.h:444-450
struct block_iq4_xs {
    __half   d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K/64];
    uint8_t  qs[QK_K/2];
};
static_assert(sizeof(block_iq4_xs) == sizeof(__half) + sizeof(uint16_t) + QK_K/64 + QK_K/2, "wrong iq4_xs block size");
