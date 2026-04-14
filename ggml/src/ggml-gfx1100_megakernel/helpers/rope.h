// rope.h — Rotary Position Embedding (RoPE)
// Copied from ggml/src/ggml-cuda/rope.cu lines 6-112
// Only rope_norm (adjacent pairing) — used by Qwen3/3.5 attention layers.
#pragma once

#include "hip-shim.h"

// --- rope_corr_dims --- rope.cu:6-8
struct rope_corr_dims {
    float v[2];
};

// --- rope_yarn_ramp --- rope.cu:15-18
static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// --- rope_yarn --- rope.cu:22-41
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

// --- rope_norm device function --- rope.cu:43-112
// Adjacent pairing: (x[0],x[1]), (x[2],x[3]), ...
// For use as __device__ function within a megakernel (not standalone kernel).
template <bool forward, bool has_ff>
static __device__ void rope_norm_apply(
        const float * __restrict__ x,
        float       * __restrict__ dst,
        const int            ne00,
        const int            n_dims,
        const int            i0,         // dimension index (must be even)
        const int32_t        pos,        // position for this token
        const float          freq_scale,
        const float          ext_factor,
        const float          attn_factor,
        const rope_corr_dims corr_dims,
        const float          theta_scale,
        const float *        freq_factors) {

    if (i0 >= ne00 || i0 >= n_dims) {
        // Beyond RoPE dims: pass through
        if (i0 < ne00) {
            dst[i0 + 0] = x[i0 + 0];
            dst[i0 + 1] = x[i0 + 1];
        }
        return;
    }

    const float theta_base = pos * powf(theta_scale, i0 / 2.0f);
    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;
    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[i0 + 0];
    const float x1 = x[i0 + 1];

    dst[i0 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[i0 + 1] = x0 * sin_theta + x1 * cos_theta;
}

// --- rope_neox device function --- ported from rope.cu:115-180
// NeoX pairing: (x[i], x[i + n_dims/2])
// Used by GPT-NeoX, StableLM, Phi, Qwen, etc. (arch's rope_type == ROPE_NEOX)
template <bool forward, bool has_ff>
static __device__ void rope_neox_apply(
        const float * __restrict__ x,
        float       * __restrict__ dst,
        const int            ne00,
        const int            n_dims,
        const int            i0,         // even dimension index
        const int32_t        pos,
        const float          freq_scale,
        const float          ext_factor,
        const float          attn_factor,
        const rope_corr_dims corr_dims,
        const float          theta_scale,
        const float *        freq_factors) {

    if (i0 >= n_dims) {
        // Beyond RoPE dims: pass through — matches baseline rope.cu:159-164
        if (i0 < ne00) {
            dst[i0/2 + 0]        = x[i0/2 + 0];
            dst[i0/2 + n_dims/2] = x[i0/2 + n_dims/2]; // NOT a typo: NeoX layout
        }
        return;
    }

    const float theta_base  = pos * powf(theta_scale, i0 / 2.0f);
    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta, sin_theta;
    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[i0/2 + 0];
    const float x1 = x[i0/2 + n_dims/2];

    dst[i0/2 + 0]          = x0 * cos_theta - x1 * sin_theta;
    dst[i0/2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

// --- rope_multi (M-RoPE) device function --- ported from rope.cu:183-266
// Multi-dimensional RoPE: different "sections" of head dim use different position
// components (t/h/w/extra). Used by Qwen2VL / Qwen3VL.
// sections[0..3] partition [0, n_dims/2) and each interval uses pos[i2 + k*ne2].
struct mrope_sections_gfx { int v[4]; };

template <bool forward, bool has_ff>
static __device__ void rope_multi_apply(
        const float * __restrict__ x,
        float       * __restrict__ dst,
        const int            ne00,
        const int            n_dims,
        const int            i0,
        const int32_t *      pos_quad,   // [4] per-section position components
        const float          freq_scale,
        const float          ext_factor,
        const float          attn_factor,
        const rope_corr_dims corr_dims,
        const float          theta_scale,
        const float *        freq_factors,
        const mrope_sections_gfx sections) {

    if (i0 >= n_dims) {
        if (i0 < ne00) {
            dst[i0/2 + 0]        = x[i0/2 + 0];
            dst[i0/2 + n_dims/2] = x[i0/2 + n_dims/2];
        }
        return;
    }

    // Baseline rope.cu:221-239: pick pos component by section
    const int sect_dims   = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w       = sections.v[1] + sections.v[0];
    const int sec_e       = sections.v[2] + sec_w;
    const int sector      = (i0 / 2) % sect_dims;
    int32_t p = pos_quad[0];
    if (sector >= sections.v[0] && sector < sec_w) {
        p = pos_quad[1];
    } else if (sector >= sec_w && sector < sec_e) {
        p = pos_quad[2];
    } else if (sector >= sec_e) {
        p = pos_quad[3];
    }

    const float theta_base  = p * powf(theta_scale, i0 / 2.0f);
    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta, sin_theta;
    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[i0/2 + 0];
    const float x1 = x[i0/2 + n_dims/2];

    dst[i0/2 + 0]          = x0 * cos_theta - x1 * sin_theta;
    dst[i0/2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}
