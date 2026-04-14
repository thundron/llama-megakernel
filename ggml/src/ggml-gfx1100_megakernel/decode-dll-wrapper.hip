// decode-dll-wrapper.hip — Host-side DLL wrapper for fast kernel dispatch
//
// This file is compiled with hipcc --shared to produce a per-model .dll/.so
// containing ALL kernels from decode.hip as embedded device code.
//
// The DLL exports two functions:
//   gfx1100_get_kernel(name)  — resolve kernel name → host function pointer
//   gfx1100_dll_launch(fn, grid, block, shm, stream, args) — fast launch
//
// Why this is faster than .hsaco:
//   .hsaco path:  hipModuleLaunchKernel → module lookup → dispatch  (~2.1µs)
//   .dll path:    hipLaunchKernel → direct dispatch                 (~1.5µs)
//
// The ~0.6µs savings per launch * ~80 launches/token = ~50µs/token savings.
// For Gemma-2 at 7ms/token, that's ~0.7% — compounding with other overhead.

#include "decode.hip"
#include <cstring>

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C" __attribute__((visibility("default")))
#endif

// ============================================================================
// Kernel lookup table — maps name strings to host-side __global__ addresses
// ============================================================================

struct _kentry { const char * name; const void * fn; };

static const _kentry _ktable[] = {
    // --- Embedding kernels (all quant types) ---
    {"eval_embed_q4_0",            (const void *)eval_embed_q4_0},
    {"eval_embed_q4_1",            (const void *)eval_embed_q4_1},
    {"eval_embed_q5_0",            (const void *)eval_embed_q5_0},
    {"eval_embed_q5_1",            (const void *)eval_embed_q5_1},
    {"eval_embed_q8_0",            (const void *)eval_embed_q8_0},
    {"eval_embed_q2k",             (const void *)eval_embed_q2k},
    {"eval_embed_q3k",             (const void *)eval_embed_q3k},
    {"eval_embed_q4k",             (const void *)eval_embed_q4k},
    {"eval_embed_q5k",             (const void *)eval_embed_q5k},
    {"eval_embed_q6k",             (const void *)eval_embed_q6k},
    {"eval_embed_iq2_xxs",         (const void *)eval_embed_iq2_xxs},
    {"eval_embed_iq2_xs",          (const void *)eval_embed_iq2_xs},
    {"eval_embed_iq2_s",           (const void *)eval_embed_iq2_s},
    {"eval_embed_iq3_xxs",         (const void *)eval_embed_iq3_xxs},
    {"eval_embed_iq3_s",           (const void *)eval_embed_iq3_s},
    {"eval_embed_iq1_s",           (const void *)eval_embed_iq1_s},
    {"eval_embed_iq1_m",           (const void *)eval_embed_iq1_m},
    {"eval_embed_iq4_nl",          (const void *)eval_embed_iq4_nl},
    {"eval_embed_iq4_xs",          (const void *)eval_embed_iq4_xs},
    {"eval_embed_mxfp4",           (const void *)eval_embed_mxfp4},
    {"eval_embed_nvfp4",           (const void *)eval_embed_nvfp4},
    {"eval_embed_f32",             (const void *)eval_embed_f32},
    {"eval_embed_f16",             (const void *)eval_embed_f16},
    {"eval_embed_bf16",            (const void *)eval_embed_bf16},

    // --- Norm + quantize ---
    {"eval_rmsnorm_q8",            (const void *)eval_rmsnorm_q8},
    {"eval_rmsnorm_q8_quantize",   (const void *)eval_rmsnorm_q8_quantize},
    {"eval_rmsnorm_add",           (const void *)eval_rmsnorm_add},
    {"eval_rmsnorm_add_rmsnorm_q8",(const void *)eval_rmsnorm_add_rmsnorm_q8},
    {"eval_quantize_q8",           (const void *)eval_quantize_q8},
    {"eval_layernorm",             (const void *)eval_layernorm},
    {"eval_l2norm",                (const void *)eval_l2norm},
    {"eval_final_norm",            (const void *)eval_final_norm},

    // --- Matvec kernels (all quant types, regular + residual) ---
    {"eval_matvec_q4_0",           (const void *)eval_matvec_q4_0},
    {"eval_matvec_q4_0_residual",  (const void *)eval_matvec_q4_0_residual},
    {"eval_matvec_q4_1",           (const void *)eval_matvec_q4_1},
    {"eval_matvec_q4_1_residual",  (const void *)eval_matvec_q4_1_residual},
    {"eval_matvec_q5_0",           (const void *)eval_matvec_q5_0},
    {"eval_matvec_q5_0_residual",  (const void *)eval_matvec_q5_0_residual},
    {"eval_matvec_q5_1",           (const void *)eval_matvec_q5_1},
    {"eval_matvec_q5_1_residual",  (const void *)eval_matvec_q5_1_residual},
    {"eval_matvec_q8_0",           (const void *)eval_matvec_q8_0},
    {"eval_matvec_q8_0_residual",  (const void *)eval_matvec_q8_0_residual},
    {"eval_matvec_q2k",            (const void *)eval_matvec_q2k},
    {"eval_matvec_q2k_residual",   (const void *)eval_matvec_q2k_residual},
    {"eval_matvec_q3k",            (const void *)eval_matvec_q3k},
    {"eval_matvec_q3k_residual",   (const void *)eval_matvec_q3k_residual},
    {"eval_matvec_q4k",            (const void *)eval_matvec_q4k},
    {"eval_matvec_q4k_residual",   (const void *)eval_matvec_q4k_residual},
    {"eval_matvec_q5k",            (const void *)eval_matvec_q5k},
    {"eval_matvec_q5k_residual",   (const void *)eval_matvec_q5k_residual},
    {"eval_matvec_q6k",            (const void *)eval_matvec_q6k},
    {"eval_matvec_q6k_residual",   (const void *)eval_matvec_q6k_residual},
    {"eval_matvec_iq2_xxs",        (const void *)eval_matvec_iq2_xxs},
    {"eval_matvec_iq2_xxs_residual",(const void *)eval_matvec_iq2_xxs_residual},
    {"eval_matvec_iq2_xs",         (const void *)eval_matvec_iq2_xs},
    {"eval_matvec_iq2_xs_residual",(const void *)eval_matvec_iq2_xs_residual},
    {"eval_matvec_iq2_s",          (const void *)eval_matvec_iq2_s},
    {"eval_matvec_iq2_s_residual", (const void *)eval_matvec_iq2_s_residual},
    {"eval_matvec_iq3_xxs",        (const void *)eval_matvec_iq3_xxs},
    {"eval_matvec_iq3_xxs_residual",(const void *)eval_matvec_iq3_xxs_residual},
    {"eval_matvec_iq3_s",          (const void *)eval_matvec_iq3_s},
    {"eval_matvec_iq3_s_residual", (const void *)eval_matvec_iq3_s_residual},
    {"eval_matvec_iq1_s",          (const void *)eval_matvec_iq1_s},
    {"eval_matvec_iq1_s_residual", (const void *)eval_matvec_iq1_s_residual},
    {"eval_matvec_iq1_m",          (const void *)eval_matvec_iq1_m},
    {"eval_matvec_iq1_m_residual", (const void *)eval_matvec_iq1_m_residual},
    {"eval_matvec_iq4_nl",         (const void *)eval_matvec_iq4_nl},
    {"eval_matvec_iq4_nl_residual",(const void *)eval_matvec_iq4_nl_residual},
    {"eval_matvec_iq4_xs",         (const void *)eval_matvec_iq4_xs},
    {"eval_matvec_iq4_xs_residual",(const void *)eval_matvec_iq4_xs_residual},
    {"eval_matvec_mxfp4",          (const void *)eval_matvec_mxfp4},
    {"eval_matvec_mxfp4_residual", (const void *)eval_matvec_mxfp4_residual},
    {"eval_matvec_nvfp4",          (const void *)eval_matvec_nvfp4},
    {"eval_matvec_nvfp4_residual", (const void *)eval_matvec_nvfp4_residual},
    {"eval_matvec_f16",            (const void *)eval_matvec_f16},
    {"eval_matvec_f16_residual",   (const void *)eval_matvec_f16_residual},
    {"eval_matvec_bf16",           (const void *)eval_matvec_bf16},
    {"eval_matvec_bf16_residual",  (const void *)eval_matvec_bf16_residual},
    {"eval_matvec_f32",            (const void *)eval_matvec_f32},
    {"eval_matvec_f32_residual",   (const void *)eval_matvec_f32_residual},

    // --- 8-warp matvec variants (RDNA3) ---
    {"eval_matvec_q4_0_8w",            (const void *)eval_matvec_q4_0_8w},
    {"eval_matvec_q4_0_8w_residual",   (const void *)eval_matvec_q4_0_8w_residual},
    {"eval_matvec_q4_1_8w",            (const void *)eval_matvec_q4_1_8w},
    {"eval_matvec_q4_1_8w_residual",   (const void *)eval_matvec_q4_1_8w_residual},
    {"eval_matvec_q5_0_8w",            (const void *)eval_matvec_q5_0_8w},
    {"eval_matvec_q5_0_8w_residual",   (const void *)eval_matvec_q5_0_8w_residual},
    {"eval_matvec_q5_1_8w",            (const void *)eval_matvec_q5_1_8w},
    {"eval_matvec_q5_1_8w_residual",   (const void *)eval_matvec_q5_1_8w_residual},
    {"eval_matvec_q8_0_8w",            (const void *)eval_matvec_q8_0_8w},
    {"eval_matvec_q8_0_8w_residual",   (const void *)eval_matvec_q8_0_8w_residual},
    {"eval_matvec_iq4_nl_8w",          (const void *)eval_matvec_iq4_nl_8w},
    {"eval_matvec_iq4_nl_8w_residual", (const void *)eval_matvec_iq4_nl_8w_residual},
    {"eval_matvec_q4k_8w",             (const void *)eval_matvec_q4k_8w},
    {"eval_matvec_q4k_8w_residual",    (const void *)eval_matvec_q4k_8w_residual},
    {"eval_matvec_q6k_8w",             (const void *)eval_matvec_q6k_8w},
    {"eval_matvec_q6k_8w_residual",    (const void *)eval_matvec_q6k_8w_residual},

    // --- Element-wise ops ---
    {"eval_add_residual",          (const void *)eval_add_residual},
    {"eval_elementwise_mul",       (const void *)eval_elementwise_mul},
    {"eval_silu_mul",              (const void *)eval_silu_mul},
    {"eval_gelu_mul",              (const void *)eval_gelu_mul},
    {"eval_gelu",                  (const void *)eval_gelu},
    {"eval_gelu_erf",              (const void *)eval_gelu_erf},
    {"eval_gelu_erf_mul",          (const void *)eval_gelu_erf_mul},
    {"eval_gelu_quick_mul",        (const void *)eval_gelu_quick_mul},
    {"eval_relu2_mul",             (const void *)eval_relu2_mul},
    {"eval_scale_scalar",          (const void *)eval_scale_scalar},
    {"eval_softcap",               (const void *)eval_softcap},
    {"eval_silu",                  (const void *)eval_silu},
    {"eval_sub",                   (const void *)eval_sub},
    {"eval_muladd",                (const void *)eval_muladd},
    {"eval_axpy",                  (const void *)eval_axpy},
    {"eval_sum_row",               (const void *)eval_sum_row},
    {"eval_sigmoid",               (const void *)eval_sigmoid},
    {"eval_softplus",              (const void *)eval_softplus},
    {"eval_tanh",                  (const void *)eval_tanh},
    {"eval_neg",                   (const void *)eval_neg},
    {"eval_exp",                   (const void *)eval_exp},
    {"eval_relu",                  (const void *)eval_relu},
    {"eval_sqr",                   (const void *)eval_sqr},
    {"eval_mul_one_minus",         (const void *)eval_mul_one_minus},
    {"eval_repeat_interleave",     (const void *)eval_repeat_interleave},

    // --- Attention + RoPE ---
    {"eval_qk_norm_rope_kv_write", (const void *)eval_qk_norm_rope_kv_write},
    // Fused gate+up matvec (baseline has_fusion path)
    {"eval_matvec_glu_q4k",        (const void *)eval_matvec_glu_q4k},
    {"eval_matvec_glu_q6k",        (const void *)eval_matvec_glu_q6k},
    {"eval_matvec_glu_q4_0",       (const void *)eval_matvec_glu_q4_0},
    {"eval_matvec_glu_q8_0",       (const void *)eval_matvec_glu_q8_0},

    {"eval_attention_decode",      (const void *)eval_attention_decode},
    {"eval_attention_decode_pb",   (const void *)eval_attention_decode_pb},
    {"eval_attention_combine",     (const void *)eval_attention_combine},
    {"eval_attention_decode_tile", (const void *)eval_attention_decode_tile},
    {"eval_attention_decode_wmma", (const void *)eval_attention_decode_wmma},
    {"eval_rope_neox_inplace",     (const void *)eval_rope_neox_inplace},
    {"eval_write_decode_params",   (const void *)eval_write_decode_params},

    // --- SSM / RWKV / DeltaNet ---
    {"eval_ssm_conv_step",         (const void *)eval_ssm_conv_step},
    {"eval_ssm_scan_step",         (const void *)eval_ssm_scan_step},
    {"eval_rwkv_wkv6_step",        (const void *)eval_rwkv_wkv6_step},
    {"eval_rwkv_wkv7_step",        (const void *)eval_rwkv_wkv7_step},
    {"eval_rwkv7_rk_correction",   (const void *)eval_rwkv7_rk_correction},
    {"eval_dn_conv1d_silu",        (const void *)eval_dn_conv1d_silu},
    {"eval_dn_l2_norm",            (const void *)eval_dn_l2_norm},
    {"eval_dn_recurrence",         (const void *)eval_dn_recurrence},

    // --- Sorting / Softmax ---
    {"eval_argsort_desc",          (const void *)eval_argsort_desc},
    {"eval_softmax_row",           (const void *)eval_softmax_row},
    {"eval_concat_dim0",           (const void *)eval_concat_dim0},
    {"eval_concat_dim1",           (const void *)eval_concat_dim1},
    {"eval_repeat_dim0",           (const void *)eval_repeat_dim0},

    // --- MoE ---
    {"eval_moe_matvec_q4k",        (const void *)eval_moe_matvec_q4k},
    {"eval_moe_matvec_q6k",        (const void *)eval_moe_matvec_q6k},
    {"eval_moe_matvec_q5k",        (const void *)eval_moe_matvec_q5k},
    {"eval_moe_matvec_q3k",        (const void *)eval_moe_matvec_q3k},
    {"eval_moe_matvec_q2k",        (const void *)eval_moe_matvec_q2k},
    {"eval_moe_matvec_q4_0",       (const void *)eval_moe_matvec_q4_0},
    {"eval_moe_matvec_q4_1",       (const void *)eval_moe_matvec_q4_1},
    {"eval_moe_matvec_q5_0",       (const void *)eval_moe_matvec_q5_0},
    {"eval_moe_matvec_q5_1",       (const void *)eval_moe_matvec_q5_1},
    {"eval_moe_matvec_q8_0",       (const void *)eval_moe_matvec_q8_0},
    {"eval_moe_matvec_iq2_xxs",    (const void *)eval_moe_matvec_iq2_xxs},
    {"eval_moe_matvec_iq2_xs",     (const void *)eval_moe_matvec_iq2_xs},
    {"eval_moe_matvec_iq2_s",      (const void *)eval_moe_matvec_iq2_s},
    {"eval_moe_matvec_iq3_xxs",    (const void *)eval_moe_matvec_iq3_xxs},
    {"eval_moe_matvec_iq3_s",      (const void *)eval_moe_matvec_iq3_s},
    {"eval_moe_matvec_iq1_s",      (const void *)eval_moe_matvec_iq1_s},
    {"eval_moe_matvec_iq1_m",      (const void *)eval_moe_matvec_iq1_m},
    {"eval_moe_matvec_iq4_nl",     (const void *)eval_moe_matvec_iq4_nl},
    {"eval_moe_matvec_iq4_xs",     (const void *)eval_moe_matvec_iq4_xs},
    {"eval_moe_matvec_mxfp4",      (const void *)eval_moe_matvec_mxfp4},
    {"eval_moe_matvec_nvfp4",      (const void *)eval_moe_matvec_nvfp4},
    {"eval_moe_matvec_f16",        (const void *)eval_moe_matvec_f16},
    {"eval_moe_matvec_bf16",       (const void *)eval_moe_matvec_bf16},
    {"eval_moe_matvec_f32",        (const void *)eval_moe_matvec_f32},
    {"eval_moe_weighted_add",      (const void *)eval_moe_weighted_add},
    {"eval_moe_normalize_weights", (const void *)eval_moe_normalize_weights},
    {"eval_moe_gate_mul",          (const void *)eval_moe_gate_mul},
    {"eval_moe_batch_normalize_weights", (const void *)eval_moe_batch_normalize_weights},
    {"eval_moe_group_tokens",      (const void *)eval_moe_group_tokens},
    {"eval_moe_gather",            (const void *)eval_moe_gather},
    {"eval_moe_scatter_weighted_add",(const void *)eval_moe_scatter_weighted_add},

    // --- Fused gate+up+SiLU (all quant types) ---
    {"eval_fused_gate_up_silu_q4_0",   (const void *)eval_fused_gate_up_silu_q4_0},
    {"eval_fused_gate_up_silu_q4_1",   (const void *)eval_fused_gate_up_silu_q4_1},
    {"eval_fused_gate_up_silu_q5_0",   (const void *)eval_fused_gate_up_silu_q5_0},
    {"eval_fused_gate_up_silu_q5_1",   (const void *)eval_fused_gate_up_silu_q5_1},
    {"eval_fused_gate_up_silu_q8_0",   (const void *)eval_fused_gate_up_silu_q8_0},
    {"eval_fused_gate_up_silu_q2k",    (const void *)eval_fused_gate_up_silu_q2k},
    {"eval_fused_gate_up_silu_q3k",    (const void *)eval_fused_gate_up_silu_q3k},
    {"eval_fused_gate_up_silu_q4k",    (const void *)eval_fused_gate_up_silu_q4k},
    {"eval_fused_gate_up_silu_q5k",    (const void *)eval_fused_gate_up_silu_q5k},
    {"eval_fused_gate_up_silu_q6k",    (const void *)eval_fused_gate_up_silu_q6k},
    {"eval_fused_gate_up_silu_iq2_xxs",(const void *)eval_fused_gate_up_silu_iq2_xxs},
    {"eval_fused_gate_up_silu_iq2_xs", (const void *)eval_fused_gate_up_silu_iq2_xs},
    {"eval_fused_gate_up_silu_iq2_s",  (const void *)eval_fused_gate_up_silu_iq2_s},
    {"eval_fused_gate_up_silu_iq3_xxs",(const void *)eval_fused_gate_up_silu_iq3_xxs},
    {"eval_fused_gate_up_silu_iq3_s",  (const void *)eval_fused_gate_up_silu_iq3_s},
    {"eval_fused_gate_up_silu_iq1_s",  (const void *)eval_fused_gate_up_silu_iq1_s},
    {"eval_fused_gate_up_silu_iq1_m",  (const void *)eval_fused_gate_up_silu_iq1_m},
    {"eval_fused_gate_up_silu_iq4_nl", (const void *)eval_fused_gate_up_silu_iq4_nl},
    {"eval_fused_gate_up_silu_iq4_xs", (const void *)eval_fused_gate_up_silu_iq4_xs},
    {"eval_fused_gate_up_silu_mxfp4",  (const void *)eval_fused_gate_up_silu_mxfp4},
    {"eval_fused_gate_up_silu_nvfp4",  (const void *)eval_fused_gate_up_silu_nvfp4},
    {"eval_fused_gate_up_silu_f16",    (const void *)eval_fused_gate_up_silu_f16},
    {"eval_fused_gate_up_silu_bf16",   (const void *)eval_fused_gate_up_silu_bf16},
    {"eval_fused_gate_up_silu_f32",    (const void *)eval_fused_gate_up_silu_f32},

    // --- Fused gate+up+GELU (tanh approx) ---
    {"eval_fused_gate_up_gelu_q4k",    (const void *)eval_fused_gate_up_gelu_q4k},
    {"eval_fused_gate_up_gelu_q6k",    (const void *)eval_fused_gate_up_gelu_q6k},
    {"eval_fused_gate_up_gelu_q5k",    (const void *)eval_fused_gate_up_gelu_q5k},
    {"eval_fused_gate_up_gelu_q3k",    (const void *)eval_fused_gate_up_gelu_q3k},
    {"eval_fused_gate_up_gelu_q2k",    (const void *)eval_fused_gate_up_gelu_q2k},
    {"eval_fused_gate_up_gelu_q4_0",   (const void *)eval_fused_gate_up_gelu_q4_0},
    {"eval_fused_gate_up_gelu_q8_0",   (const void *)eval_fused_gate_up_gelu_q8_0},

    // --- Fused QKV projection ---
    {"eval_fused_qkv_matvec_q4k",  (const void *)eval_fused_qkv_matvec_q4k},
    {"eval_fused_qkv_matvec_q6k",  (const void *)eval_fused_qkv_matvec_q6k},
    {"eval_fused_qkv_matvec_q5k",  (const void *)eval_fused_qkv_matvec_q5k},
    {"eval_fused_qkv_matvec_q3k",  (const void *)eval_fused_qkv_matvec_q3k},
    {"eval_fused_qkv_matvec_q2k",  (const void *)eval_fused_qkv_matvec_q2k},
    {"eval_fused_qkv_matvec_q4_0", (const void *)eval_fused_qkv_matvec_q4_0},
    {"eval_fused_qkv_matvec_q4_1", (const void *)eval_fused_qkv_matvec_q4_1},
    {"eval_fused_qkv_matvec_q5_0", (const void *)eval_fused_qkv_matvec_q5_0},
    {"eval_fused_qkv_matvec_q5_1", (const void *)eval_fused_qkv_matvec_q5_1},
    {"eval_fused_qkv_matvec_q8_0", (const void *)eval_fused_qkv_matvec_q8_0},

    // --- Fused quantize+matvec ---
    {"eval_quantize_matvec_q4k",   (const void *)eval_quantize_matvec_q4k},
    {"eval_quantize_matvec_q6k",   (const void *)eval_quantize_matvec_q6k},
    {"eval_quantize_matvec_q5k",   (const void *)eval_quantize_matvec_q5k},
    {"eval_quantize_matvec_q3k",   (const void *)eval_quantize_matvec_q3k},
    {"eval_quantize_matvec_q2k",   (const void *)eval_quantize_matvec_q2k},
    {"eval_quantize_matvec_q4_0",  (const void *)eval_quantize_matvec_q4_0},
    {"eval_quantize_matvec_q4_1",  (const void *)eval_quantize_matvec_q4_1},
    {"eval_quantize_matvec_q5_0",  (const void *)eval_quantize_matvec_q5_0},
    {"eval_quantize_matvec_q5_1",  (const void *)eval_quantize_matvec_q5_1},
    {"eval_quantize_matvec_q8_0",  (const void *)eval_quantize_matvec_q8_0},

    // --- Fused quantize+matvec+residual ---
    {"eval_quantize_matvec_residual_q4k",  (const void *)eval_quantize_matvec_residual_q4k},
    {"eval_quantize_matvec_residual_q6k",  (const void *)eval_quantize_matvec_residual_q6k},
    {"eval_quantize_matvec_residual_q5k",  (const void *)eval_quantize_matvec_residual_q5k},
    {"eval_quantize_matvec_residual_q3k",  (const void *)eval_quantize_matvec_residual_q3k},
    {"eval_quantize_matvec_residual_q2k",  (const void *)eval_quantize_matvec_residual_q2k},
    {"eval_quantize_matvec_residual_q4_0", (const void *)eval_quantize_matvec_residual_q4_0},
    {"eval_quantize_matvec_residual_q4_1", (const void *)eval_quantize_matvec_residual_q4_1},
    {"eval_quantize_matvec_residual_q5_0", (const void *)eval_quantize_matvec_residual_q5_0},
    {"eval_quantize_matvec_residual_q5_1", (const void *)eval_quantize_matvec_residual_q5_1},
    {"eval_quantize_matvec_residual_q8_0", (const void *)eval_quantize_matvec_residual_q8_0},

    // --- MMQ quantize ---
    {"eval_quantize_mmq_q8_1_d4",  (const void *)eval_quantize_mmq_q8_1_d4},
    {"eval_quantize_mmq_q8_1_ds4", (const void *)eval_quantize_mmq_q8_1_ds4},
    {"eval_quantize_mmq_q8_1_d2s6",(const void *)eval_quantize_mmq_q8_1_d2s6},

    // --- Sampling ---
    {"eval_sample_temperature",    (const void *)eval_sample_temperature},
    {"eval_sample_rep_penalty",    (const void *)eval_sample_rep_penalty},
    {"eval_sample_argmax",         (const void *)eval_sample_argmax},
    {"eval_sample_top_k_p",        (const void *)eval_sample_top_k_p},

    // --- Audio / Image / Special ---
    {"eval_image_patches",         (const void *)eval_image_patches},
    {"eval_stft_power",            (const void *)eval_stft_power},
    {"eval_mel_filterbank",        (const void *)eval_mel_filterbank},
    {"eval_mel_normalize",         (const void *)eval_mel_normalize},
    {"eval_max_reduce",            (const void *)eval_max_reduce},
    {"eval_chameleon_suppress",    (const void *)eval_chameleon_suppress},
    {"eval_t5_rel_bias_compute",   (const void *)eval_t5_rel_bias_compute},
    {"eval_fill_positions",        (const void *)eval_fill_positions},

    // --- WavTokenizer ---
    {"eval_group_norm",            (const void *)eval_group_norm},
    {"eval_im2col_1d",             (const void *)eval_im2col_1d},
    {"eval_swish",                 (const void *)eval_swish},
    {"eval_conv1d_dw_ph",          (const void *)eval_conv1d_dw_ph},

    {nullptr, nullptr}  // sentinel
};

// ============================================================================
// Exported functions — loaded by init.cpp via LoadLibrary/dlopen
// ============================================================================

// Resolve kernel name → host-side __global__ function pointer.
// Called once per kernel during init. Linear scan is fine for ~200 entries.
DLLEXPORT const void * gfx1100_get_kernel(const char * name) {
    for (int i = 0; _ktable[i].name; i++) {
        if (strcmp(_ktable[i].name, name) == 0) return _ktable[i].fn;
    }
    return nullptr;
}

// Launch kernel via hipLaunchKernel — the fast path.
// hipLaunchKernel uses pre-registered device code embedded in this DLL,
// avoiding the module lookup overhead of hipModuleLaunchKernel.
DLLEXPORT hipError_t gfx1100_dll_launch(
    const void * fn,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shm, hipStream_t stream, void ** args) {
    return hipLaunchKernel(fn, dim3(gx, gy, gz), dim3(bx, by, bz),
                           args, shm, stream);
}
