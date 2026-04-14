// embed-dequant.h — ALL embedding dequantization for token lookup
// Ported VERBATIM from ggml/src/ggml-cuda/convert.cu + dequantize.cuh
#pragma once

#include "hip-shim.h"
#include "quant-types.h"

// ============================================================================
// Baseline dequantize functions — VERBATIM from ggml/src/ggml-cuda/dequantize.cuh.
// Signature: (vx, ib, iqs, &v) — v.x/v.y filled with two adjacent (qr==1) or
// stride-qk/2 (qr==2) dequantized values. Used by k_get_rows port for
// Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 single-row embedding gather.
// ============================================================================

static __device__ __forceinline__ void baseline_dequantize_q4_0(const void * vx, const long long ib, const int iqs, float2 & v) {
    const block_q4_0 * x = (const block_q4_0 *) vx;
    const float d = __half2float(x[ib].d);
    const int vui = x[ib].qs[iqs];
    v.x = vui & 0xF;
    v.y = vui >> 4;
    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void baseline_dequantize_q4_1(const void * vx, const long long ib, const int iqs, float2 & v) {
    const block_q4_1 * x = (const block_q4_1 *) vx;
    const float dx = __half2float(x[ib].dm.x);
    const float dy = __half2float(x[ib].dm.y);
    const int vui = x[ib].qs[iqs];
    v.x = vui & 0xF;
    v.y = vui >> 4;
    v.x = (v.x * dx) + dy;
    v.y = (v.y * dx) + dy;
}

static __device__ __forceinline__ void baseline_dequantize_q5_0(const void * vx, const long long ib, const int iqs, float2 & v) {
    const block_q5_0 * x = (const block_q5_0 *) vx;
    const float d = __half2float(x[ib].d);
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;
    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);
    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void baseline_dequantize_q5_1(const void * vx, const long long ib, const int iqs, float2 & v) {
    const block_q5_1 * x = (const block_q5_1 *) vx;
    const float dx = __half2float(x[ib].dm.x);
    const float dy = __half2float(x[ib].dm.y);
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;
    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);
    v.x = (v.x * dx) + dy;
    v.y = (v.y * dx) + dy;
}

static __device__ __forceinline__ void baseline_dequantize_q8_0(const void * vx, const long long ib, const int iqs, float2 & v) {
    const block_q8_0 * x = (const block_q8_0 *) vx;
    const float d = __half2float(x[ib].d);
    v.x = x[ib].qs[iqs + 0] * d;
    v.y = x[ib].qs[iqs + 1] * d;
}

// ============================================================================
// Helper: get_scale_min_k4 from convert.cu:194-201 (used by Q4_K, Q5_K)
// ============================================================================
static __device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// ============================================================================
// Small-block types (QK=32) — from dequantize.cuh + convert.cu
// Each function: 32 threads dequantize one superblock (256 elements = 8 blocks)
// ============================================================================

// --- Q4_0 --- convert.cu:83-109
static __device__ void dequantize_block_q4_0_device(
        const void * __restrict__ vx, float * __restrict__ y,
        const int block_idx, const int tid) {  // tid: 0..31
    const int il  = tid / 8;
    const int ir  = tid % 8;
    const int ib = 8 * block_idx + ir;
    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    float * out = y + block_idx * QK_K + 32 * ir + 4 * il;
    const float d = __half2float(x->d);
    const float dm = -8 * d;
    const uint8_t * q = x->qs + 4 * il;
    for (int l = 0; l < 4; ++l) {
        out[l +  0] = d * (q[l] & 0xF) + dm;
        out[l + 16] = d * (q[l] >>  4) + dm;
    }
}

// --- Q4_1 --- convert.cu:112-136
static __device__ void dequantize_block_q4_1_device(
        const void * __restrict__ vx, float * __restrict__ y,
        const int block_idx, const int tid) {
    const int il  = tid / 8;
    const int ir  = tid % 8;
    const int ib = 8 * block_idx + ir;
    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    float * out = y + block_idx * QK_K + 32 * ir + 4 * il;
    const float2 d = __half22float2(x->dm);
    const uint8_t * q = x->qs + 4 * il;
    for (int l = 0; l < 4; ++l) {
        out[l +  0] = d.x * (q[l] & 0xF) + d.y;
        out[l + 16] = d.x * (q[l] >>  4) + d.y;
    }
}

// --- Q5_0 --- dequantize.cuh:31-47 + convert.cu pattern
static __device__ void dequantize_block_q5_0_device(
        const void * __restrict__ vx, float * __restrict__ y,
        const int block_idx, const int tid) {
    const int il  = tid / 8;
    const int ir  = tid % 8;
    const int ib = 8 * block_idx + ir;
    const block_q5_0 * x = (const block_q5_0 *)vx + ib;
    float * out = y + block_idx * QK_K + 32 * ir + 4 * il;
    const float d = __half2float(x->d);
    uint32_t qh;
    memcpy(&qh, x->qh, sizeof(qh));
    for (int l = 0; l < 4; ++l) {
        const int iqs = 4 * il + l;
        const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;
        out[l +  0] = ((x->qs[iqs] & 0xf) | xh_0) * d - 16.0f * d;
        out[l + 16] = ((x->qs[iqs] >>   4) | xh_1) * d - 16.0f * d;
    }
}

// --- Q5_1 --- dequantize.cuh:49-65 + convert.cu pattern
static __device__ void dequantize_block_q5_1_device(
        const void * __restrict__ vx, float * __restrict__ y,
        const int block_idx, const int tid) {
    const int il  = tid / 8;
    const int ir  = tid % 8;
    const int ib = 8 * block_idx + ir;
    const block_q5_1 * x = (const block_q5_1 *)vx + ib;
    float * out = y + block_idx * QK_K + 32 * ir + 4 * il;
    const float2 dm = __half22float2(x->dm);
    uint32_t qh;
    memcpy(&qh, x->qh, sizeof(qh));
    for (int l = 0; l < 4; ++l) {
        const int iqs = 4 * il + l;
        const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;
        out[l +  0] = ((x->qs[iqs] & 0xf) | xh_0) * dm.x + dm.y;
        out[l + 16] = ((x->qs[iqs] >>   4) | xh_1) * dm.x + dm.y;
    }
}

// --- Q8_0 --- dequantize.cuh:67-77
static __device__ void dequantize_block_q8_0_device(
        const void * __restrict__ vx, float * __restrict__ y,
        const int block_idx, const int tid) {
    const int il  = tid / 8;
    const int ir  = tid % 8;
    const int ib = 8 * block_idx + ir;
    const block_q8_0 * x = (const block_q8_0 *)vx + ib;
    float * out = y + block_idx * QK_K + 32 * ir + 4 * il;
    const float d = __half2float(x->d);
    for (int l = 0; l < 4; ++l) {
        out[l] = x->qs[4 * il + l] * d;
    }
    // Q8_0 has 32 elements per block, only fills 4 per thread (il=0..3, 4 each = 16)
    // But 32 threads * 4 = 128 values for 8 blocks = 256 elements. Correct.
}

// ============================================================================
// K-quant types (QK=256) — from convert.cu
// ============================================================================

// --- Q2_K --- convert.cu:142-160, 64 threads
static __device__ void dequantize_block_q2_K_device(
        const block_q2_K * __restrict__ x, float * __restrict__ y,
        const int block_idx, const int tid) {  // tid: 0..63
    const int n   = tid / 32;
    const int l   = tid - 32 * n;
    const int is  = 8 * n + l / 16;
    const uint8_t q = x[block_idx].qs[32 * n + l];
    float * out = y + block_idx * QK_K + 128 * n;
    const float dall = __low2float(x[block_idx].dm);
    const float dmin = __high2float(x[block_idx].dm);
    out[l +  0] = dall * (x[block_idx].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[block_idx].scales[is+0] >> 4);
    out[l + 32] = dall * (x[block_idx].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[block_idx].scales[is+2] >> 4);
    out[l + 64] = dall * (x[block_idx].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[block_idx].scales[is+4] >> 4);
    out[l + 96] = dall * (x[block_idx].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[block_idx].scales[is+6] >> 4);
}

// --- Q3_K --- convert.cu:163-191, 64 threads
static __device__ void dequantize_block_q3_K_device(
        const block_q3_K * __restrict__ x, float * __restrict__ y,
        const int block_idx, const int tid) {
    const int r = tid / 4;
    const int t = r / 2;
    const int is0 = r % 2;
    const int l0 = 16 * is0 + 4 * (tid % 4);
    const int n = t / 4;
    const int j = t - 4 * n;
    uint8_t m = 1 << (4 * n + j);
    int is = 8 * n + 2 * j + is0;
    int shift = 2 * j;
    int8_t us = is <  4 ? (x[block_idx].scales[is-0] & 0xF) | (((x[block_idx].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[block_idx].scales[is-0] & 0xF) | (((x[block_idx].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[block_idx].scales[is-8] >>  4) | (((x[block_idx].scales[is+0] >> 4) & 3) << 4) :
                          (x[block_idx].scales[is-8] >>  4) | (((x[block_idx].scales[is-4] >> 6) & 3) << 4);
    float d_all = __half2float(x[block_idx].d);
    float dl = d_all * (us - 32);
    float * out = y + block_idx * QK_K + 128 * n + 32 * j;
    const uint8_t * q = x[block_idx].qs + 32 * n;
    const uint8_t * hm = x[block_idx].hmask;
    for (int l = l0; l < l0 + 4; ++l) out[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
}

// --- Q4_K --- convert.cu:203-232, 32 threads (already had this)
static __device__ void dequantize_block_q4_K_device(
        const block_q4_K * __restrict__ x, float * __restrict__ y,
        const int block_idx, const int tid) {
    const int il = tid / 8;
    const int ir = tid % 8;
    const int is = 2 * il;
    const int n  = 4;
    float * out = y + block_idx * QK_K + 64 * il + n * ir;
    const float dall = __low2float(x[block_idx].dm);
    const float dmin = __high2float(x[block_idx].dm);
    const uint8_t * q = x[block_idx].qs + 32 * il + n * ir;
    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[block_idx].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[block_idx].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        out[l +  0] = d1 * (q[l] & 0xF) - m1;
        out[l + 32] = d2 * (q[l] >>  4) - m2;
    }
}

// --- Q5_K --- convert.cu:235-266, 64 threads
static __device__ void dequantize_block_q5_K_device(
        const block_q5_K * __restrict__ x, float * __restrict__ y,
        const int block_idx, const int tid) {
    const int il  = tid / 16;
    const int ir  = tid % 16;
    const int is  = 2 * il;
    float * out = y + block_idx * QK_K + 64 * il + 2 * ir;
    const float dall = __low2float(x[block_idx].dm);
    const float dmin = __high2float(x[block_idx].dm);
    const uint8_t * ql = x[block_idx].qs + 32 * il + 2 * ir;
    const uint8_t * qh = x[block_idx].qh + 2 * ir;
    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[block_idx].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[block_idx].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    uint8_t hm = 1 << (2 * il);
    out[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    out[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    out[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    out[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
}

// --- Q6_K --- convert.cu:269-292, 64 threads (already had this)
static __device__ void dequantize_block_q6_K_device(
        const block_q6_K * __restrict__ x, float * __restrict__ y,
        const int block_idx, const int tid) {
    const int ip = tid / 32;
    const int il = tid - 32 * ip;
    const int is = 8 * ip + il / 16;
    float * out = y + block_idx * QK_K + 128 * ip + il;
    const float d = __half2float(x[block_idx].d);
    const uint8_t * ql = x[block_idx].ql + 64 * ip + il;
    const uint8_t   qh = x[block_idx].qh[32 * ip + il];
    const int8_t  * sc = x[block_idx].scales + is;
    out[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    out[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    out[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    out[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

// ============================================================================
// IQ types — from convert.cu:294-469, all use 32 threads
// ============================================================================

// --- IQ2_XXS --- convert.cu:295-310
static __device__ void dequantize_block_iq2_xxs_device(
        const block_iq2_xxs * __restrict__ x, float * __restrict__ y,
        const int block_idx, const int tid) {
    const int il = tid / 8;
    const int ib = tid % 8;
    float * out = y + block_idx * QK_K + 32 * ib + 8 * il;
    const uint16_t * q2 = x[block_idx].qs + 4 * ib;
    const uint8_t * aux8 = (const uint8_t *)q2;
    const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = __half2float(x[block_idx].d) * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) out[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

// --- IQ2_XS --- convert.cu:314-327
static __device__ void dequantize_block_iq2_xs_device(
        const block_iq2_xs * __restrict__ x, float * __restrict__ y,
        const int block_idx, const int tid) {
    const int il = tid / 8;
    const int ib = tid % 8;
    float * out = y + block_idx * QK_K + 32 * ib + 8 * il;
    const uint16_t * q2 = x[block_idx].qs + 4 * ib;
    const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d = __half2float(x[block_idx].d) * (0.5f + ((x[block_idx].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) out[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

// --- IQ2_S --- convert.cu:331-343
static __device__ void dequantize_block_iq2_s_device(
        const block_iq2_s * __restrict__ x, float * __restrict__ y,
        const int block_idx, const int tid) {
    const int il = tid / 8;
    const int ib = tid % 8;
    float * out = y + block_idx * QK_K + 32 * ib + 8 * il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[block_idx].qs[4*ib+il] | ((x[block_idx].qh[ib] << (8-2*il)) & 0x300)));
    const float d = __half2float(x[block_idx].d) * (0.5f + ((x[block_idx].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[block_idx].qs[QK_K/8+4*ib+il];
    for (int j = 0; j < 8; ++j) out[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

// --- IQ3_XXS --- convert.cu:345-366
static __device__ void dequantize_block_iq3_xxs_device(
        const block_iq3_xxs * __restrict__ x, float * __restrict__ y,
        const int block_idx, const int tid) {
    const int il = tid / 8;
    const int ib = tid % 8;
    float * out = y + block_idx * QK_K + 32 * ib + 8 * il;
    const uint8_t * q3 = x[block_idx].qs + 8 * ib;
    const uint16_t * gas = (const uint16_t *)(x[block_idx].qs + QK_K/4) + 2*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = __half2float(x[block_idx].d) * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        out[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        out[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

// --- IQ3_S --- convert.cu:369-388
static __device__ void dequantize_block_iq3_s_device(
        const block_iq3_s * __restrict__ x, float * __restrict__ y,
        const int block_idx, const int tid) {
    const int il = tid / 8;
    const int ib = tid % 8;
    float * out = y + block_idx * QK_K + 32 * ib + 8 * il;
    const uint8_t * qs = x[block_idx].qs + 8 * ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[block_idx].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[block_idx].qh[ib] << (7-2*il)) & 256)));
    const float d = __half2float(x[block_idx].d) * (1 + 2*((x[block_idx].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[block_idx].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        out[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        out[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

// --- IQ1_S --- convert.cu:391-404
static __device__ void dequantize_block_iq1_s_device(
        const block_iq1_s * __restrict__ x, float * __restrict__ y,
        const int block_idx, const int tid) {
    const int il = tid / 8;
    const int ib = tid % 8;
    float * out = y + block_idx * QK_K + 32 * ib + 8 * il;
    const float delta = x[block_idx].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = __half2float(x[block_idx].d) * (2*((x[block_idx].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[block_idx].qs[4*ib+il] | (((x[block_idx].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) out[j] = d * (q[j] + delta);
}

// --- IQ1_M --- convert.cu:412-433
static __device__ void dequantize_block_iq1_m_device(
        const block_iq1_m * __restrict__ x, float * __restrict__ y,
        const int block_idx, const int tid) {
    const int il = tid / 8;
    const int ib = tid % 8;
    float * out = y + block_idx * QK_K + 32 * ib + 8 * il;
    const uint16_t * sc = (const uint16_t *)x[block_idx].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int ib16 = 2*ib + il/2;
    const float d = __half2float(scale.f16) * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    const float delta = x[block_idx].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[block_idx].qs[4*ib+il] | (((x[block_idx].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) out[j] = d * (q[j] + delta);
}

// --- IQ4_NL --- convert.cu:437-452
static __device__ void dequantize_block_iq4_nl_device(
        const void * __restrict__ vx, float * __restrict__ y,
        const int block_idx, const int tid) {
    const block_iq4_nl * x = (const block_iq4_nl *)vx + block_idx * (QK_K / QK4_NL);
    const int il = tid / 8;
    const int ib = tid % 8;
    float * out = y + block_idx * QK_K + 32 * ib + 4 * il;
    const uint8_t * q4 = x[ib].qs + 4 * il;
    const float d = __half2float(x[ib].d);
    for (int j = 0; j < 4; ++j) {
        out[j +  0] = d * kvalues_iq4nl[q4[j] & 0xf];
        out[j + 16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

// --- IQ4_XS --- convert.cu:455-469
static __device__ void dequantize_block_iq4_xs_device(
        const block_iq4_xs * __restrict__ x, float * __restrict__ y,
        const int block_idx, const int tid) {
    const int il = tid / 8;
    const int ib = tid % 8;
    float * out = y + block_idx * QK_K + 32 * ib + 4 * il;
    const uint8_t * q4 = x[block_idx].qs + 16 * ib + 4 * il;
    const float d = __half2float(x[block_idx].d) * ((((x[block_idx].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[block_idx].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        out[j +  0] = d * kvalues_iq4nl[q4[j] & 0xf];
        out[j + 16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

// --- MXFP4 --- convert.cu:472-487
static __device__ void dequantize_block_mxfp4_device(
        const void * __restrict__ vx, float * __restrict__ y,
        const int block_idx, const int tid) {
    const block_mxfp4 * x = (const block_mxfp4 *)vx + block_idx * (QK_K / QK_MXFP4);
    const int il = tid / 8;
    const int ib = tid % 8;
    float * out = y + block_idx * QK_K + 32 * ib + 4 * il;
    const uint8_t * q4 = x[ib].qs + 4 * il;
    const float d = ggml_cuda_e8m0_to_fp32(x[ib].e);
    for (int j = 0; j < 4; ++j) {
        out[j +  0] = d * kvalues_mxfp4[q4[j] & 0xf] * 0.5f;
        out[j + 16] = d * kvalues_mxfp4[q4[j] >>  4] * 0.5f;
    }
}

// ============================================================================
// NVFP4 embedding — QK=64, different block layout from QK=256 types
// From baseline convert.cu — uses kvalues_mxfp4 + ue4m3 scales
// ============================================================================
static __device__ void dequantize_block_nvfp4_device(
        const void * __restrict__ vx, float * __restrict__ y,
        const int block_idx, const int tid) {
    // NVFP4: QK=64, 8 blocks per 512 elements
    // With 32 threads processing 8 sub-blocks of 64 each = 512 elements per superblock
    const int il = tid / 8;
    const int ib = tid % 8;
    const block_nvfp4 * x = (const block_nvfp4 *)vx + block_idx * (512 / QK_NVFP4) + ib;
    float * out = y + block_idx * 512 + QK_NVFP4 * ib + 4 * il;
    const uint8_t * q4 = x->qs + 4 * il;
    const float d = ggml_cuda_ue4m3_to_fp32(x->d[il / (QK_NVFP4_SUB/4)]);
    for (int j = 0; j < 4; ++j) {
        out[j +  0] = d * kvalues_mxfp4[q4[j] & 0xf];
        out[j + 32] = d * kvalues_mxfp4[q4[j] >>  4];
    }
}

// ============================================================================
// Float-type embeddings — F16, BF16, F32 (just copy/convert row to f32)
// ============================================================================

// F32 → F32: direct copy
static __device__ void dequantize_block_f32_device(
        const float * __restrict__ x, float * __restrict__ y,
        const int row_offset, const int tid, const int n) {
    for (int i = tid; i < n; i += blockDim.x) {
        y[i] = x[row_offset + i];
    }
}

// F16 → F32: convert half to float
static __device__ void dequantize_block_f16_device(
        const __half * __restrict__ x, float * __restrict__ y,
        const int row_offset, const int tid, const int n) {
    for (int i = tid; i < n; i += blockDim.x) {
        y[i] = __half2float(x[row_offset + i]);
    }
}

// BF16 → F32: shift left 16 bits
static __device__ void dequantize_block_bf16_device(
        const uint16_t * __restrict__ x, float * __restrict__ y,
        const int row_offset, const int tid, const int n) {
    for (int i = tid; i < n; i += blockDim.x) {
        uint32_t bits = (uint32_t)x[row_offset + i] << 16;
        float val;
        memcpy(&val, &bits, sizeof(float));
        y[i] = val;
    }
}
