// gfx1100-internal.h — shared structs, globals, utilities
// Auto-extracted from gfx1100-megakernel.cpp
#pragma once

#include "gfx1100-megakernel.h"
#include "helpers/rocblas-gemm.h"
#include "arch_ids.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <functional>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

#define HIP_ASSERT(expr) do { \
    hipError_t _e = (expr); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "gfx1100-megakernel: HIP error at %s:%d: %s\n", \
                __FILE__, __LINE__, hipGetErrorString(_e)); \
        return -1; \
    } \
} while(0)

// ============================================================================
// Model config — passed by the caller (extracted from GGUF)
// ============================================================================

struct gfx1100_layer_weights {
    // Weight matrices — slots match fill_attention/fill_deltanet layout
    const void * ptrs[16];      // weight data pointers (GPU)
    long long    strides[16];   // row stride in bytes
    int          types[16];     // ggml_type per slot

    // --- All optional tensors from baseline llama.cpp ---
    // Attention biases — baseline: if (bq) Qcur = ggml_add(Qcur, bq)
    const void * bias_q;        // bq [qproj_size] f32 or NULL
    const void * bias_k;        // bk [kv_size] f32 or NULL
    const void * bias_v;        // bv [kv_size] f32 or NULL
    const void * bias_o;        // bo [hidden_size] f32 or NULL — baseline: build_attn adds bo

    // LoRA / per-tensor scales — baseline: build_lora_mm(w, cur, w_s) multiplies by w_s
    const void * scale_q;       // wq_s or NULL
    const void * scale_k;       // wk_s or NULL
    const void * scale_v;       // wv_s or NULL
    const void * scale_o;       // wo_s or NULL — baseline: if (wo_s) cur *= wo_s

    // FFN biases — baseline: build_ffn adds these after each projection
    const void * ffn_gate_bias; // ffn_gate_b or NULL
    const void * ffn_up_bias;   // ffn_up_b or NULL
    const void * ffn_down_bias; // ffn_down_b or NULL

    // FFN scales — baseline: build_ffn multiplies by these
    const void * ffn_gate_scale; // ffn_gate_s or NULL
    const void * ffn_up_scale;   // ffn_up_s or NULL
    const void * ffn_down_scale; // ffn_down_s or NULL

    // MoE detection — baseline: if (ffn_gate_inp != NULL) use MoE path
    const void * ffn_gate_inp;   // NULL for standard FFN, non-NULL for MoE

    // MoE expert weights — baseline: build_moe_ffn tensors
    // These are 3D: [out_dim, n_expert, in_dim]. To get expert i's weight:
    //   ptr = base + i * expert_stride_bytes
    // where expert_stride_bytes = out_dim * row_stride_bytes
    const void * ffn_gate_exps;      // [n_ff, n_expert, n_embd] or NULL
    long long    ffn_gate_exps_stride; // row stride in bytes
    int          ffn_gate_exps_type;   // ggml_type
    const void * ffn_up_exps;        // [n_ff, n_expert, n_embd] or NULL
    long long    ffn_up_exps_stride;
    int          ffn_up_exps_type;
    const void * ffn_down_exps;      // [n_embd, n_expert, n_ff] or NULL
    long long    ffn_down_exps_stride;
    int          ffn_down_exps_type;

    // MoE routing parameters
    long long    ffn_gate_inp_stride; // row stride for router weight
    int          ffn_gate_inp_type;   // ggml_type for router weight

    // MoE gating function type — baseline: llama_expert_gating_func_type
    int          moe_gating_op;      // 1=softmax, 2=sigmoid, 3=softmax_weight
    float        moe_w_scale;        // hparams.expert_weights_scale
    int          moe_norm_w;         // normalize expert weights to sum=1

    // Norm biases — needed for LayerNorm models (BERT, BLOOM, Falcon, GPT2, Phi, etc.)
    // For RMSNorm models these are NULL.
    const void * attn_norm_bias; // attn_norm_b or NULL — baseline: build_norm(LLM_NORM_LAYER) adds bias
    const void * ffn_norm_bias;  // ffn_norm_b or NULL

    // Gemma2/3/4 post-norms — baseline: build_gemma2
    // Applied AFTER attention/FFN output, BEFORE residual add
    const void * attn_post_norm;  // [n_embd] RMSNorm after attention output (Gemma2+ only)
    const void * ffn_post_norm;   // [n_embd] RMSNorm after FFN output (Gemma2+ only)

    // BitNet sub-norms — baseline: build_bitnet
    // RMSNorm applied BETWEEN attention/FFN output and the output projection
    const void * attn_sub_norm;  // [n_embd] RMSNorm before wo projection (BitNet only)
    const void * ffn_sub_norm;   // [n_ff] RMSNorm before ffn_down projection (BitNet only)

    // CogVLM visual expert weights — alternate weight set for image tokens
    const void * visexp_wqkv;        // [n_embd, n_embd + 2*n_embd_gqa] fused QKV for visual expert
    long long    visexp_wqkv_stride;
    int          visexp_wqkv_type;
    const void * visexp_wo;          // [n_embd, n_embd] output projection for visual expert
    long long    visexp_wo_stride;
    int          visexp_wo_type;
    const void * visexp_ffn_gate;    // [n_ff, n_embd] FFN gate for visual expert
    long long    visexp_ffn_gate_stride;
    int          visexp_ffn_gate_type;
    const void * visexp_ffn_down;    // [n_embd, n_ff] FFN down for visual expert
    long long    visexp_ffn_down_stride;
    int          visexp_ffn_down_type;
    const void * visexp_ffn_up;      // [n_ff, n_embd] FFN up for visual expert
    long long    visexp_ffn_up_stride;
    int          visexp_ffn_up_type;

    // Qwen2MoE shared expert — baseline: ffn_gate_inp_shexp, ffn_gate/up/down_shexp
    const void * ffn_gate_inp_shexp;   // shared expert gate or NULL
    long long    ffn_gate_inp_shexp_stride;
    int          ffn_gate_inp_shexp_type;
    const void * ffn_gate_shexp;       // [n_ff, n_embd] shared expert gate weight
    long long    ffn_gate_shexp_stride;
    int          ffn_gate_shexp_type;
    const void * ffn_up_shexp;         // [n_ff, n_embd] shared expert up weight
    long long    ffn_up_shexp_stride;
    int          ffn_up_shexp_type;
    const void * ffn_down_shexp;       // [n_embd, n_ff] shared expert down weight
    long long    ffn_down_shexp_stride;
    int          ffn_down_shexp_type;

    // DeepSeek2 MLA layer weights — baseline: deepseek2.cpp MLA path
    const void * wq_a;            // [q_lora_rank, n_embd] Q LoRA down-projection
    long long    wq_a_stride;
    int          wq_a_type;
    const void * attn_q_a_norm;   // [q_lora_rank] Q LoRA norm weight
    const void * wq_b;            // [n_head * n_embd_head_k, q_lora_rank] Q LoRA up-projection
    long long    wq_b_stride;
    int          wq_b_type;
    const void * wkv_a_mqa;       // [kv_lora_rank + n_embd_head_qk_rope, n_embd] KV compressed projection
    long long    wkv_a_mqa_stride;
    int          wkv_a_mqa_type;
    const void * attn_kv_a_norm;  // [kv_lora_rank] KV compressed norm weight
    const void * wk_b;            // [n_embd_head_qk_nope, kv_lora_rank, n_head] K absorption matrix
    long long    wk_b_stride;
    int          wk_b_type;
    const void * wv_b;            // [n_embd_head_v, kv_lora_rank, n_head] V decompression matrix
    long long    wv_b_stride;
    int          wv_b_type;
    const void * wkv_b;           // [n_head*(nope+v_dim), kv_lora_rank] non-MLA path: full KV up-proj
    long long    wkv_b_stride;
    int          wkv_b_type;

    // RWKV layer weights — baseline: build_rwkv6_time_mix + build_rwkv6_channel_mix
    // Time-mix (recurrent attention)
    const void * time_mix_lerp_x;     // [n_embd] lerp base
    const void * time_mix_lerp_fused; // [n_embd, 5] fused lerp (w,k,v,r,g) or NULL
    const void * time_mix_lerp_w;     // [n_embd] or NULL (if fused)
    const void * time_mix_lerp_k;     // [n_embd] or NULL
    const void * time_mix_lerp_v;     // [n_embd] or NULL
    const void * time_mix_lerp_r;     // [n_embd] or NULL
    const void * time_mix_lerp_g;     // [n_embd] or NULL
    const void * time_mix_w1;         // [n_embd, lora_size*5] lora down
    const void * time_mix_w2;         // [lora_size, n_embd, 5] lora up
    long long    time_mix_w1_stride;
    long long    time_mix_w2_stride;
    int          time_mix_w1_type;
    int          time_mix_w2_type;
    const void * time_mix_receptance;  // [n_embd, n_embd]
    long long    time_mix_receptance_stride;
    int          time_mix_receptance_type;
    const void * time_mix_key;         // [n_embd, n_embd]
    long long    time_mix_key_stride;
    int          time_mix_key_type;
    const void * time_mix_value;       // [n_embd, n_embd]
    long long    time_mix_value_stride;
    int          time_mix_value_type;
    const void * time_mix_gate;        // [n_embd, n_embd]
    long long    time_mix_gate_stride;
    int          time_mix_gate_type;
    const void * time_mix_output;      // [n_embd, n_embd]
    long long    time_mix_output_stride;
    int          time_mix_output_type;
    const void * time_mix_first;       // [n_embd] time_first (WKV6 only)
    const void * time_mix_decay;       // [n_embd] base decay
    const void * time_mix_decay_w1;    // [n_embd, lora_size] decay lora down
    const void * time_mix_decay_w2;    // [lora_size, n_embd] decay lora up
    long long    time_mix_decay_w1_stride;
    long long    time_mix_decay_w2_stride;
    int          time_mix_decay_w1_type;
    int          time_mix_decay_w2_type;
    const void * time_mix_ln;          // [n_embd] group norm weight
    const void * time_mix_ln_b;        // [n_embd] group norm bias
    const void * time_mix_receptance_b; // optional bias
    const void * time_mix_key_b;        // optional bias
    const void * time_mix_value_b;      // optional bias

    // RWKV7-specific weights — baseline: build_rwkv7_time_mix
    const void * time_mix_w0;           // [n_embd] decay bias (RWKV7 only)
    const void * time_mix_a0;           // [n_embd] a gate bias
    const void * time_mix_a1;           // [lora_size, n_embd] a gate LoRA down
    const void * time_mix_a2;           // [n_embd, lora_size] a gate LoRA up
    long long    time_mix_a1_stride;
    long long    time_mix_a2_stride;
    int          time_mix_a1_type;
    int          time_mix_a2_type;
    const void * time_mix_v0;           // [n_embd] v_first gate bias
    const void * time_mix_v1;           // [lora_size, n_embd] v_first gate LoRA down
    const void * time_mix_v2;           // [n_embd, lora_size] v_first gate LoRA up
    long long    time_mix_v1_stride;
    long long    time_mix_v2_stride;
    int          time_mix_v1_type;
    int          time_mix_v2_type;
    const void * time_mix_g1;           // [lora_size, n_embd] output gate LoRA down (optional)
    const void * time_mix_g2;           // [n_embd, lora_size] output gate LoRA up (optional)
    long long    time_mix_g1_stride;
    long long    time_mix_g2_stride;
    int          time_mix_g1_type;
    int          time_mix_g2_type;
    const void * time_mix_k_k;          // [n_embd] kk scaling for L2 norm key
    const void * time_mix_k_a;          // [n_embd] k-to-a mixing for key modification
    const void * time_mix_r_k;          // [n_embd] r*k additive output weight

    // Channel-mix (FFN replacement)
    const void * channel_mix_lerp_k;   // [n_embd]
    const void * channel_mix_lerp_r;   // [n_embd]
    const void * channel_mix_receptance; // [n_embd, n_embd]
    long long    channel_mix_receptance_stride;
    int          channel_mix_receptance_type;
    const void * channel_mix_key;       // [n_embd, n_embd]
    long long    channel_mix_key_stride;
    int          channel_mix_key_type;
    const void * channel_mix_value;     // [n_embd, n_embd]
    long long    channel_mix_value_stride;
    int          channel_mix_value_type;

    // T5 encoder layer weights — baseline: build_t5_enc
    // These are separate from the decoder weights (wq, wk, etc.)
    const void * wq_enc;              // [n_embd_k_gqa, n_embd]
    long long    wq_enc_stride;
    int          wq_enc_type;
    const void * wk_enc;              // [n_embd_k_gqa, n_embd]
    long long    wk_enc_stride;
    int          wk_enc_type;
    const void * wv_enc;              // [n_embd_v_gqa, n_embd]
    long long    wv_enc_stride;
    int          wv_enc_type;
    const void * wo_enc;              // [n_embd, n_embd_v_gqa]
    long long    wo_enc_stride;
    int          wo_enc_type;
    const void * attn_norm_enc;       // [n_embd] encoder attention RMSNorm weight
    const void * ffn_norm_enc;        // [n_embd] encoder FFN RMSNorm weight
    const void * ffn_gate_enc;        // [n_ff, n_embd] encoder FFN gate (Flan-T5) or NULL
    long long    ffn_gate_enc_stride;
    int          ffn_gate_enc_type;
    const void * ffn_down_enc;        // [n_embd, n_ff] encoder FFN down
    long long    ffn_down_enc_stride;
    int          ffn_down_enc_type;
    const void * ffn_up_enc;          // [n_ff, n_embd] encoder FFN up
    long long    ffn_up_enc_stride;
    int          ffn_up_enc_type;

    // T5 relative position bias — baseline: attn_rel_b, attn_rel_b_enc
    // Shape: [n_head, n_rel_attn_bkts]. Only present on layer 0 (shared to all layers).
    const void * attn_rel_b;          // decoder self-attention relative bias or NULL
    const void * attn_rel_b_enc;      // encoder self-attention relative bias or NULL

    // T5 cross-attention layer weights — baseline: build_t5_dec cross-attention
    const void * attn_norm_cross;     // [n_embd] cross-attention RMSNorm weight
    const void * wq_cross;            // [n_embd_k_gqa, n_embd]
    long long    wq_cross_stride;
    int          wq_cross_type;
    const void * wk_cross;            // [n_embd_k_gqa, n_embd]
    long long    wk_cross_stride;
    int          wk_cross_type;
    const void * wv_cross;            // [n_embd_v_gqa, n_embd]
    long long    wv_cross_stride;
    int          wv_cross_type;
    const void * wo_cross;            // [n_embd, n_embd_v_gqa]
    long long    wo_cross_stride;
    int          wo_cross_type;

    // BERT post-norm layer weights — baseline: build_bert
    // Post-attention LayerNorm (applied after residual add, not before)
    const void * attn_out_norm;       // [n_embd] or NULL
    const void * attn_out_norm_b;     // [n_embd] bias or NULL
    // Post-FFN LayerNorm
    const void * layer_out_norm;      // [n_embd] or NULL
    const void * layer_out_norm_b;    // [n_embd] bias or NULL
    // Jina-BERT-V2 extra norm
    const void * attn_norm_2;         // [n_embd] or NULL
    const void * attn_norm_2_b;       // [n_embd] bias or NULL
    // QK normalization (BERT variants)
    const void * attn_q_norm;         // [n_embd_head] or NULL — for BERT QK norm
    const void * attn_q_norm_b;       // [n_embd_head] bias or NULL
    const void * attn_k_norm;         // [n_embd_head] or NULL
    const void * attn_k_norm_b;       // [n_embd_head] bias or NULL

    // Mamba/SSM layer weights — baseline: build_mamba_layer
    const void * ssm_in;          // [2*d_inner, n_embd] input projection (x + z)
    long long    ssm_in_stride;
    int          ssm_in_type;
    const void * ssm_conv1d;      // [d_conv, d_inner] 1D convolution weights
    const void * ssm_conv1d_b;    // [d_inner] conv bias
    const void * ssm_x;           // [dt_rank + 2*d_state, d_inner] x projection
    long long    ssm_x_stride;
    int          ssm_x_type;
    const void * ssm_dt;          // [d_inner, dt_rank] dt projection
    long long    ssm_dt_stride;
    int          ssm_dt_type;
    const void * ssm_dt_b;        // [d_inner] dt bias
    const void * ssm_a;           // [d_inner, d_state] SSM A matrix
    const void * ssm_d;           // [d_inner] D parameter
    const void * ssm_out;         // [n_embd, d_inner] output projection
    long long    ssm_out_stride;
    int          ssm_out_type;

    // FalconMamba dt/B/C normalization — baseline: ssm_dt_b_c_rms flag
    const void * ssm_dt_norm;     // [d_inner] RMSNorm on dt (FalconMamba) or NULL
    const void * ssm_b_norm;      // [d_state] RMSNorm on B or NULL
    const void * ssm_c_norm;      // [d_state] RMSNorm on C or NULL

    // Mamba2-specific weights
    const void * ssm_norm;        // [d_inner] grouped RMSNorm after skip-gating (Mamba2 only, or NULL)
};

// WavTokenizer PosNet layer weights — baseline: llama_layer_posnet (6 layers, hardcoded)
// Layers 0,1,3,4: ResNet blocks (norm1→swish→conv1→norm2→swish→conv2→residual)
// Layer 2: Self-attention (attn_norm→QKV conv→softmax→O conv→residual)
// Layer 5: Final GroupNorm only
struct gfx1100_posnet_layer {
    // ResNet block weights (layers 0,1,3,4)
    const void * norm1;       // [channels] GroupNorm weight or NULL
    const void * norm1_b;     // [channels] GroupNorm bias or NULL
    const void * conv1;       // [channels, channels, kernel_size] Conv1d weight
    long long    conv1_stride;
    int          conv1_type;
    int          conv1_kernel_size; // kernel width (ne[0] of weight tensor)
    const void * conv1_b;     // [channels] Conv1d bias
    const void * norm2;       // [channels] GroupNorm weight
    const void * norm2_b;     // [channels] GroupNorm bias
    const void * conv2;       // [channels, channels, kernel_size] Conv1d weight
    long long    conv2_stride;
    int          conv2_type;
    int          conv2_kernel_size;
    const void * conv2_b;     // [channels] Conv1d bias

    // Attention block weights (layer 2 only)
    const void * attn_norm;   // [channels] GroupNorm weight
    const void * attn_norm_b; // [channels] GroupNorm bias
    const void * attn_q;      // [channels, channels, 1] Conv1d Q
    long long    attn_q_stride;
    int          attn_q_type;
    const void * attn_q_b;    // [channels] Q bias
    const void * attn_k;      // [channels, channels, 1] Conv1d K
    long long    attn_k_stride;
    int          attn_k_type;
    const void * attn_k_b;    // [channels] K bias
    const void * attn_v;      // [channels, channels, 1] Conv1d V
    long long    attn_v_stride;
    int          attn_v_type;
    const void * attn_v_b;    // [channels] V bias
    const void * attn_o;      // [channels, channels, 1] Conv1d output
    long long    attn_o_stride;
    int          attn_o_type;
    const void * attn_o_b;    // [channels] output bias
};

// WavTokenizer ConvNext layer weights — baseline: llama_layer_convnext
// DepthwiseConv1d → LayerNorm → FFN(GELU, sequential) → gamma → residual
struct gfx1100_convnext_layer {
    const void * dw;          // [channels, 1, kernel_size] depthwise conv weight
    long long    dw_stride;
    int          dw_type;
    int          dw_kernel_size; // kernel width (ne[0] of dw weight)
    const void * dw_b;        // [channels] depthwise conv bias
    const void * norm;        // [channels] LayerNorm weight
    const void * norm_b;      // [channels] LayerNorm bias
    const void * pw1;         // [n_ff, channels] pointwise conv1 (FFN up)
    long long    pw1_stride;
    int          pw1_type;
    const void * pw1_b;       // [n_ff] bias
    const void * pw2;         // [channels, n_ff] pointwise conv2 (FFN down)
    long long    pw2_stride;
    int          pw2_type;
    const void * pw2_b;       // [channels] bias
    const void * gamma;       // [channels] per-channel scaling
};

struct gfx1100_model_config {
    // --- Architecture identification (for compile-time specialization) ---
    // Mirrors src/llama-arch.h::LLM_ARCH_* — see arch_ids.h for the
    // gfx1100 enum that maps 1:1 to the baseline IDs.
    int arch_id;                // ARCH_LLAMA, ARCH_QWEN2, ..., see arch_ids.h

    // --- Capability bits (for compile-time #if DCE in .hip sources) ---
    // Derived at init from llama_model hparams + tensor presence.
    // These are OR'd across all layers: "does the model use this feature".
    // Each bit participates in the .hsaco hash → specialized build per model.
    int has_qk_norm;            // attn_q_norm / attn_k_norm present (Qwen3/Qwen35/Llama4/...)
    int has_bias_q;             // bq present on any layer
    int has_bias_k;             // bk
    int has_bias_v;             // bv
    int has_bias_o;             // bo
    int has_scale_q;            // wq_s (BitNet / LoRA)
    int has_scale_k;            // wk_s
    int has_scale_v;            // wv_s
    int has_scale_o;            // wo_s
    int has_bias_ffn_gate;      // ffn_gate_b
    int has_bias_ffn_up;        // ffn_up_b
    int has_bias_ffn_down;      // ffn_down_b
    int has_scale_ffn_gate;     // ffn_gate_s
    int has_scale_ffn_up;       // ffn_up_s
    int has_scale_ffn_down;     // ffn_down_s
    int rope_type;              // ROPE_NONE / NORMAL / NEOX / MULTI / YARN — mirrors llama_rope_type
    int has_rope_freq_factors;  // rope_freq_factors != NULL (Llama 3 long RoPE)
    int has_moe;                // ffn_gate_inp present on any layer
    int moe_n_experts;          // hparams.n_expert
    int moe_n_experts_used;     // hparams.n_expert_used
    int has_ssm;                // Mamba/Mamba2/Jamba/Nemotron-H state space layers
    int has_dn;                 // Qwen35 DeltaNet layers
    int attn_scale_type;        // ATTN_SCALE_DEFAULT (1/sqrt(d)) / SOFTCAP / CUSTOM
    float attn_softcap_val;     // hparams.f_attn_logit_softcapping when attn_soft_cap=true
    int has_final_logit_softcap;// hparams.f_final_logit_softcapping > 0 (Gemma2)
    int has_embed_scale;        // Gemma: scale embeddings by sqrt(n_embd) after lookup
    float f_logit_scale;        // Cohere2/Command-R/Granite/Grok: scale logits after LM head (0 = no scale)
    float f_residual_scale;     // Granite: scale attn/FFN output before residual add (0 = no scale)
    int cogvlm_is_image;        // CogVLM: 1 = current token is image (use visual expert weights)
    int has_swin_norm;          // Chameleon: post-norm instead of pre-norm (norm after attn/FFN output)
    int chameleon_img_token_start; // Chameleon: first image token ID (logits suppressed for 4..8195)
    int chameleon_img_token_count; // Chameleon: number of image tokens to suppress (8192)
    float final_logit_softcap_val;
    int norm_type;              // NORM_RMS / NORM_L2 / NORM_LAYER
    int act_type;               // ACT_SILU / ACT_GELU / ACT_GELU_APPROX / ACT_RELU / ACT_RELU2
    int pooling_type;           // LLAMA_POOLING_TYPE_* for embedding models
    int has_swa;                // hparams.swa_type != LLAMA_SWA_TYPE_NONE
    int swa_type;               // LLAMA_SWA_TYPE_*
    int n_swa;                  // sliding window size
    int has_alibi;              // hparams.use_alibi
    float alibi_max_bias;       // hparams.f_max_alibi_bias (typically 8.0 for BLOOM/Falcon)
    float alibi_m0;             // powf(2.0f, -max_bias / n_head_log2)
    float alibi_m1;             // powf(2.0f, -max_bias/2.0f / n_head_log2)
    int   alibi_n_head_log2;    // 1 << floor(log2(n_head))

    // Parallel attn+FFN — baseline: hparams.use_par_res (GPT-NeoX, Falcon 40B)
    // x = x + attn(ln1(x)) + ffn(ln2(x)) instead of sequential residuals
    int use_par_res;

    // Joint QKV projection — baseline: some archs use wqkv instead of separate wq/wk/wv
    int has_wqkv;               // model.layers[il].wqkv != NULL

    // Geometry
    int hidden_size;
    int intermediate_size;
    int vocab_size;
    int n_layers;
    int layer_types[128]; // 0=attention, 1=deltanet, 2=ssm, 3=rwkv
    int layer_use_swa[128]; // per-layer: 1=sliding window, 0=full context (for iSWA models)

    // Attention
    int fa_n_q_heads;
    int fa_n_kv_heads;
    int fa_head_dim;
    float fa_rope_theta;
    float fa_rope_theta_swa;  // SWA-layer RoPE theta for iSWA models (Gemma3/4, Cohere2), 0 = same as fa_rope_theta
    int skip_rope_on_global_layers; // Cohere2: global (non-SWA) layers skip RoPE entirely (baseline: !is_swa → no rope)
    int per_layer_n_q_heads[128];  // OpenELM: per-layer Q head count (0 = use compile-time FA_N_Q_HEADS)
    int per_layer_n_kv_heads[128]; // OpenELM: per-layer KV head count (0 = use compile-time FA_N_KV_HEADS)
    int vision_patch_size;         // Vision models: patch size for image preprocessing (0 = default 14)
    const void * rope_freq_factors_per_layer[128]; // Phi3 SU-RoPE: per-layer freq factors (NULL = use global)
    int use_shared_norm_ffn;       // Cohere2: FFN reads from attn_norm output, not post-attention. Residual = FFN + inpL + attn_out
    int fa_rope_dim;
    int fa_has_gated_attn;
    float fa_attention_scale;   // baseline: hparams.f_attention_scale (0 = use 1/sqrt(d))
    int fa_use_kq_norm;         // baseline: hparams.use_kq_norm (Llama 4 only)

    // DeltaNet
    int dn_n_heads;
    int dn_n_k_heads;
    int dn_key_dim;
    int dn_value_dim;
    int dn_conv_kernel;

    // SSM (Mamba/Mamba2)
    int ssm_d_conv;             // hparams.ssm_d_conv
    int ssm_d_inner;            // hparams.ssm_d_inner
    int ssm_d_state;            // hparams.ssm_d_state
    int ssm_dt_rank;            // hparams.ssm_dt_rank
    int ssm_n_group;            // hparams.ssm_n_group

    // RWKV
    int wkv_head_size;          // hparams.wkv_head_size (typically 64)
    int rwkv_lora_size;         // hparams.time_mix_extra_dim (typically 32 or 64)

    // DeepSeek2 MLA
    int mla_kv_lora_rank;       // hparams.n_lora_kv (typically 512)
    int mla_q_lora_rank;        // hparams.n_lora_q (typically 1536)
    int mla_n_embd_head_qk_rope; // hparams.n_rot() for MLA rope portion
    int mla_n_layer_dense_lead; // hparams.n_layer_dense_lead

    // RoPE (YaRN / scaling)
    float rope_freq_scale;      // hparams.rope_freq_scale_train
    float rope_attn_factor;     // hparams.rope_attn_factor
    float yarn_ext_factor;      // hparams.yarn_ext_factor
    float yarn_attn_factor;     // hparams.yarn_attn_factor
    float yarn_beta_fast;       // hparams.yarn_beta_fast
    float yarn_beta_slow;       // hparams.yarn_beta_slow
    int   n_ctx_orig_yarn;      // hparams.n_ctx_orig_yarn
    int   rope_sections[4];     // hparams.rope_sections (Multi-RoPE)

    // Norm
    float norm_eps;
    int norm_add_one;

    // Weights
    gfx1100_layer_weights layers[128];
    const void * embed_weight;
    long long    embed_stride;
    int          embed_type;   // ggml_type (14=Q6_K, 12=Q4_K, etc.)
    const void * final_norm_weight;
    const void * final_norm_bias;   // for LayerNorm models (BERT, RWKV) or NULL
    const void * tok_norm_weight;   // RWKV: model.tok_norm (LayerNorm on input embedding)
    const void * tok_norm_bias;     // RWKV: model.tok_norm_b
    const void * lm_head_weight;
    long long    lm_head_stride;
    int          lm_head_type;   // ggml_type for LM head weight
    const void * rope_freq_factors; // per-dim RoPE frequency factors (rope_freqs.weight) or NULL

    // T5 encoder model-level weights
    const void * output_norm_enc;   // [n_embd] final encoder RMSNorm weight or NULL
    int          dec_n_layer;       // hparams.dec_n_layer (decoder layer count, 0 if not encoder-decoder)
    int          n_rel_attn_bkts;   // hparams.n_rel_attn_bkts (T5 relative position bias buckets)

    // BERT model-level weights
    const void * pos_embd;          // [max_seq, n_embd] absolute position embeddings or NULL
    long long    pos_embd_stride;
    int          pos_embd_type;
    const void * type_embd;         // [n_type, n_embd] token type embeddings or NULL (always row 0)
    long long    type_embd_stride;
    int          type_embd_type;

    // WavTokenizer — convolutional audio decoder weights
    const void * wav_conv1d;        // [out_channels, in_channels, kernel_size] initial conv1d
    long long    wav_conv1d_stride;
    int          wav_conv1d_type;
    int          wav_conv1d_kernel_size; // kernel width (ne[0] of weight tensor)
    const void * wav_conv1d_b;      // [out_channels] initial conv1d bias
    const void * wav_output_b;      // [n_embd] output projection bias (model.output_b)
    gfx1100_posnet_layer   posnet_layers[6];  // 6 hardcoded PosNet layers
    gfx1100_convnext_layer convnext_layers[32]; // up to 32 ConvNext layers
    int          n_convnext_layers;  // hparams.convnext.n_layer
    int          wav_posnet_n_groups; // number of GroupNorm groups (typically 32)

    // Audio preprocessing (Whisper, Qwen3-audio, etc.)
    // Mel filterbank coefficients loaded from model GGUF (encoder.mel_filters)
    const float * mel_filters;     // [n_mels, n_fft/2+1] on GPU, or NULL if no audio
    int audio_n_fft;               // 400 for Whisper, 0 if no audio
    int audio_hop_length;          // 160 for Whisper
    int audio_n_mels;              // 80 or 128 for Whisper

    // Encoder output buffer — used by T5 decoder cross-attention
    float *      encoder_output;    // [n_enc_tokens, n_embd] encoder hidden states (GPU)
    int          n_enc_tokens;      // number of encoder output tokens

    // KV cache (position-major) — one pair per attention layer
    void * k_cache_ptrs[128];
    void * v_cache_ptrs[128];
    int    kv_stride;       // elements per position per head
    int    max_seq_len;
    int    kv_type;         // 0=F32, 1=F16 (default F16)

    // ========================================================================
    // VALIDATE-step fields: additional GGUF signals informing the composition system
    // ========================================================================
    // NOTE: These MUST be at the end to preserve ABI compatibility with callers
    // that use an older `gfx1100_model_config` struct (e.g., test harness).
    // All default 0 (= not set / feature absent) — populated by model loader when present.
    // When the caller's struct is smaller, memcpy reads past the caller's buffer
    // but only for these trailing fields — older code just leaves them zero-initialized.

    int n_ctx_train;            // {arch}.context_length — max KV cache / attention loop bound
    int attn_causal;             // {arch}.attention.causal (default 1 = causal)
    float attn_clamp_kqv;        // {arch}.attention.clamp_kqv (StarCoder2) — QKV clamping magnitude
    int swa_pattern;             // {arch}.full_attention_interval — iSWA layer pattern
    int shared_kv_layers;        // {arch}.attention.shared_kv_layers (SmolLM3 cross-layer KV reuse)
    int rescale_every_n;         // {arch}.rescale_every_n_layers (RWKV periodic rescaling)
    float embed_scale_val;       // {arch}.embedding_scale explicit value (when not Gemma sqrt(H))
    int has_rope_sections;       // {arch}.rope.dimension_sections present (Qwen2VL/3VL)
    int fa_rope_dim_swa;         // SWA-specific RoPE dim (0 = same as fa_rope_dim)
    int fa_value_dim;            // explicit value_length (0 = same as key/head_dim)

    // MoE advanced params
    int moe_expert_ff_len;       // {arch}.expert_feed_forward_length
    int moe_shared_count;        // {arch}.expert_shared_count (Qwen2MoE, DeepSeek2)
    int moe_shared_ff_len;       // {arch}.expert_shared_feed_forward_length
    int moe_gating_func;         // {arch}.expert_gating_func (0=softmax, 1=sigmoid, ...)
    int moe_every_n_layers;      // {arch}.moe_every_n_layers (MoE frequency)
    int moe_interleave_step;     // {arch}.interleave_moe_layer_step
    int moe_group_count;         // {arch}.expert_group_count (DeepSeek3)
    int moe_group_used_count;    // {arch}.expert_group_used_count
    float router_softcap;        // {arch}.router_logit_softcapping

    // Tensor-existence-derived capabilities (set by loader scan)
    int has_post_attn_norm;      // any layer has post_attention_norm.weight
    int has_post_ffn_norm;       // any layer has post_ffw_norm.weight
    int has_fused_qkv;           // any layer has attn_qkv.weight (alias for has_wqkv)
    int has_gated_ffn;           // any layer has both ffn_gate and ffn_up (gated activation)
    int has_cross_attn;          // T5 decoder: cross_attn_q.weight present
    int has_encoder;             // T5/BERT: enc.blk.N.attn_q.weight present
};

// ============================================================================
// Compiled kernel state
// ============================================================================

struct gfx1100_compiled {
    hipModule_t   eval_module;
    // Eval kernels (separate launches per phase)
    // Embedding dequant per type
    hipFunction_t eval_embed_q4_0;
    hipFunction_t eval_embed_q4_1;
    hipFunction_t eval_embed_q5_0;
    hipFunction_t eval_embed_q5_1;
    hipFunction_t eval_embed_q8_0;
    hipFunction_t eval_embed_q2k;
    hipFunction_t eval_embed_q3k;
    hipFunction_t eval_embed_q4k;
    hipFunction_t eval_embed_q5k;
    hipFunction_t eval_embed_q6k;
    hipFunction_t eval_embed_iq2_xxs;
    hipFunction_t eval_embed_iq2_xs;
    hipFunction_t eval_embed_iq2_s;
    hipFunction_t eval_embed_iq3_xxs;
    hipFunction_t eval_embed_iq3_s;
    hipFunction_t eval_embed_iq1_s;
    hipFunction_t eval_embed_iq1_m;
    hipFunction_t eval_embed_iq4_nl;
    hipFunction_t eval_embed_iq4_xs;
    hipFunction_t eval_embed_mxfp4;
    hipFunction_t eval_embed_nvfp4;
    hipFunction_t eval_embed_f32;
    hipFunction_t eval_embed_f16;
    hipFunction_t eval_embed_bf16;
    hipFunction_t eval_rmsnorm_q8;
    hipFunction_t eval_rmsnorm_q8_quantize; // fused: norm + residual + Q8 quantize in 1 launch
    hipFunction_t eval_quantize_q8;
    // Matvec kernels per quant type (from baseline mmvq.cu)
    hipFunction_t eval_matvec_q4_0;
    hipFunction_t eval_matvec_q4_0_residual;
    hipFunction_t eval_matvec_q4_1;
    hipFunction_t eval_matvec_q4_1_residual;
    hipFunction_t eval_matvec_q5_0;
    hipFunction_t eval_matvec_q5_0_residual;
    hipFunction_t eval_matvec_q5_1;
    hipFunction_t eval_matvec_q5_1_residual;
    hipFunction_t eval_matvec_q8_0;
    hipFunction_t eval_matvec_q8_0_residual;
    hipFunction_t eval_matvec_q2k;
    hipFunction_t eval_matvec_q2k_residual;
    hipFunction_t eval_matvec_q3k;
    hipFunction_t eval_matvec_q3k_residual;
    hipFunction_t eval_matvec_q4k;
    hipFunction_t eval_matvec_q4k_residual;
    hipFunction_t eval_matvec_q5k;
    hipFunction_t eval_matvec_q5k_residual;
    hipFunction_t eval_matvec_q6k;
    hipFunction_t eval_matvec_q6k_residual;
    // IQ types
    hipFunction_t eval_matvec_iq2_xxs;
    hipFunction_t eval_matvec_iq2_xxs_residual;
    hipFunction_t eval_matvec_iq2_xs;
    hipFunction_t eval_matvec_iq2_xs_residual;
    hipFunction_t eval_matvec_iq2_s;
    hipFunction_t eval_matvec_iq2_s_residual;
    hipFunction_t eval_matvec_iq3_xxs;
    hipFunction_t eval_matvec_iq3_xxs_residual;
    hipFunction_t eval_matvec_iq3_s;
    hipFunction_t eval_matvec_iq3_s_residual;
    hipFunction_t eval_matvec_iq1_s;
    hipFunction_t eval_matvec_iq1_s_residual;
    hipFunction_t eval_matvec_iq1_m;
    hipFunction_t eval_matvec_iq1_m_residual;
    hipFunction_t eval_matvec_iq4_nl;
    hipFunction_t eval_matvec_iq4_nl_residual;
    hipFunction_t eval_matvec_iq4_xs;
    hipFunction_t eval_matvec_iq4_xs_residual;
    hipFunction_t eval_matvec_mxfp4;
    hipFunction_t eval_matvec_mxfp4_residual;
    hipFunction_t eval_matvec_nvfp4;
    hipFunction_t eval_matvec_nvfp4_residual;
    // 8-warp matvec variants for RDNA3 (gfx1100) — nwarps=8, block=(32,8,1)=256 threads
    // Only for types that benefit on RDNA3 (baseline MMVQ_PARAMETERS_RDNA3_0 table)
    hipFunction_t eval_matvec_q4_0_8w;
    hipFunction_t eval_matvec_q4_0_8w_residual;
    hipFunction_t eval_matvec_q4_1_8w;
    hipFunction_t eval_matvec_q4_1_8w_residual;
    hipFunction_t eval_matvec_q5_0_8w;
    hipFunction_t eval_matvec_q5_0_8w_residual;
    hipFunction_t eval_matvec_q5_1_8w;
    hipFunction_t eval_matvec_q5_1_8w_residual;
    hipFunction_t eval_matvec_q8_0_8w;
    hipFunction_t eval_matvec_q8_0_8w_residual;
    hipFunction_t eval_matvec_iq4_nl_8w;
    hipFunction_t eval_matvec_iq4_nl_8w_residual;
    hipFunction_t eval_matvec_q4k_8w;
    hipFunction_t eval_matvec_q4k_8w_residual;
    hipFunction_t eval_matvec_q6k_8w;
    hipFunction_t eval_matvec_q6k_8w_residual;
    hipFunction_t eval_silu_mul;
    hipFunction_t eval_gelu_mul;         // Gemma/Bert/Phi (tanh-approx GELU)
    hipFunction_t eval_gelu;             // standalone GELU (BERT sequential FFN)
    hipFunction_t eval_gelu_erf;         // standalone exact GELU variant
    hipFunction_t eval_group_norm;       // GroupNorm (WavTokenizer PosNet)
    hipFunction_t eval_im2col_1d;        // Im2col for 1D convolution
    hipFunction_t eval_swish;            // Swish = x * sigmoid(x) (WavTokenizer PosNet)
    hipFunction_t eval_conv1d_dw_ph;     // Depthwise Conv1d with half-padding (WavTokenizer ConvNext)
    hipFunction_t eval_gelu_erf_mul;     // exact GELU variant
    hipFunction_t eval_gelu_quick_mul;   // GELU-quick variant
    hipFunction_t eval_relu2_mul;        // squared ReLU (some Phi)
    hipFunction_t eval_scale_scalar;     // constant scalar multiply
    hipFunction_t eval_softcap;          // tanh(x/cap)*cap (Gemma2 logits)
    hipFunction_t eval_layernorm;        // LLM_NORM (mean-centered) — BERT/Falcon/MPT/GPT2/NeoX/Bloom/Phi2
    hipFunction_t eval_l2norm;           // general L2 norm (not DeltaNet-specific)
    hipFunction_t eval_sigmoid;          // element-wise sigmoid
    hipFunction_t eval_softplus;         // element-wise softplus
    hipFunction_t eval_argsort_desc;     // bitonic top-k argsort (desc) — MoE routing
    hipFunction_t eval_softmax_row;      // general softmax per row — MoE gate / non-flash attention
    hipFunction_t eval_concat_dim0;      // ggml_concat along dim 0
    hipFunction_t eval_concat_dim1;      // ggml_concat along dim 1
    hipFunction_t eval_repeat_dim0;      // ggml_repeat broadcasting dim 0
    hipFunction_t eval_ssm_conv_step;    // Mamba/Mamba2 conv1d single-token step
    hipFunction_t eval_ssm_scan_step;    // Mamba/Mamba2 selective scan single-token step
    hipFunction_t eval_rwkv_wkv6_step;   // RWKV6 WKV single-token step
    hipFunction_t eval_rwkv_wkv7_step;   // RWKV7 WKV single-token step
    // Standalone RoPE kernel (for DS2 MLA, T5 relative bias, etc.)
    hipFunction_t eval_rope_neox_inplace;
    // Element-wise ops for RWKV/MoE/misc
    hipFunction_t eval_tanh;
    hipFunction_t eval_neg;
    hipFunction_t eval_exp;
    hipFunction_t eval_relu;
    hipFunction_t eval_sqr;
    hipFunction_t eval_mul_one_minus;       // dst = a * (1-b) — RWKV6Qwen2 k*(1-w)
    hipFunction_t eval_repeat_interleave;   // expand [n] → [n*repeat] — Mamba2 dt expansion
    hipFunction_t eval_rwkv7_rk_correction; // per-head r*k dot + scale + add — RWKV7
    hipFunction_t eval_sample_temperature;  // GPU sampling: temperature scaling
    hipFunction_t eval_sample_rep_penalty;  // GPU sampling: repetition penalty
    hipFunction_t eval_sample_argmax;       // GPU sampling: greedy argmax
    hipFunction_t eval_sample_top_k_p;      // GPU sampling: top-K + softmax + top-P + categorical
    // Fused MoE matvec — reads expert ID from GPU, zero D2H
    // K-quant types
    hipFunction_t eval_moe_matvec_q4k;
    hipFunction_t eval_moe_matvec_q6k;
    hipFunction_t eval_moe_matvec_q5k;
    hipFunction_t eval_moe_matvec_q3k;
    hipFunction_t eval_moe_matvec_q2k;
    // Small-block types
    hipFunction_t eval_moe_matvec_q4_0;
    hipFunction_t eval_moe_matvec_q4_1;
    hipFunction_t eval_moe_matvec_q5_0;
    hipFunction_t eval_moe_matvec_q5_1;
    hipFunction_t eval_moe_matvec_q8_0;
    // IQ types
    hipFunction_t eval_moe_matvec_iq2_xxs;
    hipFunction_t eval_moe_matvec_iq2_xs;
    hipFunction_t eval_moe_matvec_iq2_s;
    hipFunction_t eval_moe_matvec_iq3_xxs;
    hipFunction_t eval_moe_matvec_iq3_s;
    hipFunction_t eval_moe_matvec_iq1_s;
    hipFunction_t eval_moe_matvec_iq1_m;
    hipFunction_t eval_moe_matvec_iq4_nl;
    hipFunction_t eval_moe_matvec_iq4_xs;
    // MXFP4/NVFP4
    hipFunction_t eval_moe_matvec_mxfp4;
    hipFunction_t eval_moe_matvec_nvfp4;
    // Float types (use f32 input, not q8)
    hipFunction_t eval_moe_matvec_f16;
    hipFunction_t eval_moe_matvec_bf16;
    hipFunction_t eval_moe_matvec_f32;
    // MoE utility
    hipFunction_t eval_moe_weighted_add;    // dst += weight * src (weight from GPU)
    hipFunction_t eval_moe_normalize_weights; // normalize top-K routing weights on GPU
    hipFunction_t eval_moe_gate_mul;        // Qwen2MoE shared expert: dst *= gate scalar (from GPU)
    hipFunction_t eval_moe_batch_normalize_weights; // batch normalize routing weights for S tokens
    hipFunction_t eval_moe_group_tokens;    // group tokens by expert for batched dispatch
    hipFunction_t eval_moe_gather;          // gather scattered token rows into contiguous buffer
    hipFunction_t eval_moe_scatter_weighted_add; // scatter expert output with routing weight
    // Fused gate+up+silu matvec — replaces 3 kernel launches (gate matvec + up matvec + silu_mul) with 1
    // Quantized types: block=(32,4,1), shared memory = (in_dim/QK8_1)*sizeof(block_q8_1)
    hipFunction_t eval_fused_gate_up_silu_q4_0;
    hipFunction_t eval_fused_gate_up_silu_q4_1;
    hipFunction_t eval_fused_gate_up_silu_q5_0;
    hipFunction_t eval_fused_gate_up_silu_q5_1;
    hipFunction_t eval_fused_gate_up_silu_q8_0;
    hipFunction_t eval_fused_gate_up_silu_q2k;
    hipFunction_t eval_fused_gate_up_silu_q3k;
    hipFunction_t eval_fused_gate_up_silu_q4k;
    hipFunction_t eval_fused_gate_up_silu_q5k;
    hipFunction_t eval_fused_gate_up_silu_q6k;
    hipFunction_t eval_fused_gate_up_silu_iq2_xxs;
    hipFunction_t eval_fused_gate_up_silu_iq2_xs;
    hipFunction_t eval_fused_gate_up_silu_iq2_s;
    hipFunction_t eval_fused_gate_up_silu_iq3_xxs;
    hipFunction_t eval_fused_gate_up_silu_iq3_s;
    hipFunction_t eval_fused_gate_up_silu_iq1_s;
    hipFunction_t eval_fused_gate_up_silu_iq1_m;
    hipFunction_t eval_fused_gate_up_silu_iq4_nl;
    hipFunction_t eval_fused_gate_up_silu_iq4_xs;
    hipFunction_t eval_fused_gate_up_silu_mxfp4;
    hipFunction_t eval_fused_gate_up_silu_nvfp4;
    // Float types: block=(256,1,1), no shared memory needed for q8
    hipFunction_t eval_fused_gate_up_silu_f16;
    hipFunction_t eval_fused_gate_up_silu_bf16;
    hipFunction_t eval_fused_gate_up_silu_f32;
    // Fused gate+up+GELU (tanh-approx) variants
    hipFunction_t eval_fused_gate_up_gelu_q4k;
    hipFunction_t eval_fused_gate_up_gelu_q6k;
    hipFunction_t eval_fused_gate_up_gelu_q5k;
    hipFunction_t eval_fused_gate_up_gelu_q3k;
    hipFunction_t eval_fused_gate_up_gelu_q2k;
    hipFunction_t eval_fused_gate_up_gelu_q4_0;
    hipFunction_t eval_fused_gate_up_gelu_q8_0;
    // Fused QKV projection — 3 matvec launches → 1 (shared-memory q8_act reuse)
    hipFunction_t eval_fused_qkv_matvec_q4k;
    hipFunction_t eval_fused_qkv_matvec_q6k;
    hipFunction_t eval_fused_qkv_matvec_q5k;
    hipFunction_t eval_fused_qkv_matvec_q3k;
    hipFunction_t eval_fused_qkv_matvec_q2k;
    hipFunction_t eval_fused_qkv_matvec_q4_0;
    hipFunction_t eval_fused_qkv_matvec_q4_1;
    hipFunction_t eval_fused_qkv_matvec_q5_0;
    hipFunction_t eval_fused_qkv_matvec_q5_1;
    hipFunction_t eval_fused_qkv_matvec_q8_0;
    // Fused quantize+matvec — eliminates q8_act global memory round-trip
    hipFunction_t eval_quantize_matvec_q4k;
    hipFunction_t eval_quantize_matvec_q6k;
    hipFunction_t eval_quantize_matvec_q5k;
    hipFunction_t eval_quantize_matvec_q3k;
    hipFunction_t eval_quantize_matvec_q2k;
    hipFunction_t eval_quantize_matvec_q4_0;
    hipFunction_t eval_quantize_matvec_q4_1;
    hipFunction_t eval_quantize_matvec_q5_0;
    hipFunction_t eval_quantize_matvec_q5_1;
    hipFunction_t eval_quantize_matvec_q8_0;
    // Fused quantize+matvec+residual variants
    hipFunction_t eval_quantize_matvec_residual_q4k;
    hipFunction_t eval_quantize_matvec_residual_q6k;
    hipFunction_t eval_quantize_matvec_residual_q5k;
    hipFunction_t eval_quantize_matvec_residual_q3k;
    hipFunction_t eval_quantize_matvec_residual_q2k;
    hipFunction_t eval_quantize_matvec_residual_q4_0;
    hipFunction_t eval_quantize_matvec_residual_q4_1;
    hipFunction_t eval_quantize_matvec_residual_q5_0;
    hipFunction_t eval_quantize_matvec_residual_q5_1;
    hipFunction_t eval_quantize_matvec_residual_q8_0;
    // Fused consecutive norms — post-attn rmsnorm+add + pre-FFN rmsnorm+q8 (Gemma-2/3/4)
    hipFunction_t eval_rmsnorm_add_rmsnorm_q8;
    hipFunction_t eval_image_patches;       // multimodal: extract image patches
    hipFunction_t eval_stft_power;          // mel spectrogram phase 1: brute-force DFT → power spectrum
    hipFunction_t eval_mel_filterbank;      // mel spectrogram phase 2: apply mel filterbank + log
    hipFunction_t eval_mel_normalize;       // mel spectrogram phase 3: normalize to [-1, 1] range
    hipFunction_t eval_rmsnorm_add;         // fused: output = rmsnorm(input, weight) + residual (post-norm path)
    hipFunction_t eval_chameleon_suppress;  // set logits[start..start+count-1] = -FLT_MAX on GPU
    hipFunction_t eval_t5_rel_bias_compute; // compute T5 relative position bias on GPU
    hipFunction_t eval_fill_positions;      // fill int buffer with start, start+1, ..., start+n-1
    hipFunction_t eval_max_reduce;          // max reduction for mel normalization
    hipFunction_t eval_silu;             // standalone silu (not fused with mul)
    hipFunction_t eval_sub;
    hipFunction_t eval_muladd;
    hipFunction_t eval_axpy;             // dst += w * src (MoE expert accumulate)
    hipFunction_t eval_sum_row;          // sum reduction → scalar (MoE weight norm)
    // F16/BF16/F32 matvec (non-quantized, from baseline mmvf.cu)
    hipFunction_t eval_matvec_f16;
    hipFunction_t eval_matvec_f16_residual;
    hipFunction_t eval_matvec_bf16;
    hipFunction_t eval_matvec_bf16_residual;
    hipFunction_t eval_matvec_f32;
    hipFunction_t eval_matvec_f32_residual;
    hipFunction_t eval_add_residual;
    hipFunction_t eval_elementwise_mul;
    hipFunction_t eval_write_decode_params;
    hipFunction_t eval_qk_norm_rope_kv_write;
    // Fused gate+up matvec (baseline has_fusion=true path)
    hipFunction_t eval_matvec_glu_q4k;
    hipFunction_t eval_matvec_glu_q6k;
    hipFunction_t eval_matvec_glu_q4_0;
    hipFunction_t eval_matvec_glu_q8_0;

    hipFunction_t eval_attention_decode;
    hipFunction_t eval_attention_decode_pb;      // parallel-blocks variant
    hipFunction_t eval_attention_combine;        // reduction kernel for pb>1
    hipFunction_t eval_attention_decode_tile;
    hipFunction_t eval_attention_decode_wmma;
    hipFunction_t eval_final_norm;
    // LM head uses generic pick_matvec dispatch — no dedicated kernels needed
    // DeltaNet
    hipFunction_t eval_dn_conv1d_silu;
    hipFunction_t eval_dn_l2_norm;
    hipFunction_t eval_dn_recurrence;

    hipModule_t   prompt_module;
    // Batch embedding kernels — one per quant type (mirrors eval_embed_* but batch)
    hipFunction_t prompt_embed_q4_0;
    hipFunction_t prompt_embed_q4_1;
    hipFunction_t prompt_embed_q5_0;
    hipFunction_t prompt_embed_q5_1;
    hipFunction_t prompt_embed_q8_0;
    hipFunction_t prompt_embed_q2k;
    hipFunction_t prompt_embed_q3k;
    hipFunction_t prompt_embed_q4k;
    hipFunction_t prompt_embed_q5k;
    hipFunction_t prompt_embed_q6k;
    hipFunction_t prompt_embed_iq2_xxs;
    hipFunction_t prompt_embed_iq2_xs;
    hipFunction_t prompt_embed_iq2_s;
    hipFunction_t prompt_embed_iq3_xxs;
    hipFunction_t prompt_embed_iq3_s;
    hipFunction_t prompt_embed_iq1_s;
    hipFunction_t prompt_embed_iq1_m;
    hipFunction_t prompt_embed_iq4_nl;
    hipFunction_t prompt_embed_iq4_xs;
    hipFunction_t prompt_embed_mxfp4;
    hipFunction_t prompt_embed_nvfp4;
    hipFunction_t prompt_embed_f32;
    hipFunction_t prompt_embed_f16;
    hipFunction_t prompt_embed_bf16;
    hipFunction_t prompt_rmsnorm;
    hipFunction_t prompt_layernorm;  // BERT post-norm: (x-mean)/sqrt(var+eps)*w+b
    hipFunction_t prompt_per_head_layernorm; // Jina-BERT QK norm: per-head LayerNorm on [S*n_head, head_dim]
    hipFunction_t prompt_rope_neox_inplace;  // ModernBERT/NomicBERT: batch NeoX RoPE on Q/K [S, dim]
    hipFunction_t prompt_add_pos_embd; // BERT absolute position embeddings: out += pos_embd[pos[s]]
    hipFunction_t prompt_add_residual;
    hipFunction_t prompt_silu_mul;
    hipFunction_t prompt_add_bias;
    hipFunction_t prompt_elementwise_mul;
    hipFunction_t prompt_qk_norm_rope;
    hipFunction_t prompt_causal_attn;
    hipFunction_t prompt_bidirectional_attn;
    hipFunction_t prompt_deltanet;
    hipFunction_t prompt_final_norm;
    hipFunction_t prompt_lm_head;
    hipFunction_t prompt_lm_reduce;

    // MMQ kernel handles: prompt_mmq[type_idx][mmq_x_idx][need_check]
    // type_idx: index into mmq_type_table (0..19), mmq_x_idx: 0=32 1=64, need_check: 0/1
    hipFunction_t prompt_mmq[20][2][2];

    // MMQ Q8_1 batch quantize kernels (from decode.hip, 3 layouts)
    hipFunction_t eval_quantize_mmq_q8_1_d4;
    hipFunction_t eval_quantize_mmq_q8_1_ds4;
    hipFunction_t eval_quantize_mmq_q8_1_d2s6;

    // F32→F16 conversion kernel for rocBLAS input
    hipFunction_t gemm_f32_to_f16;

    // Dequant-to-F16 kernels for rocBLAS weight path (from prefill.hip)
    hipFunction_t dequant_f16_q4_0;
    hipFunction_t dequant_f16_q4_1;
    hipFunction_t dequant_f16_q5_0;
    hipFunction_t dequant_f16_q5_1;
    hipFunction_t dequant_f16_q8_0;
    hipFunction_t dequant_f16_q2k;
    hipFunction_t dequant_f16_q3k;
    hipFunction_t dequant_f16_q4k;
    hipFunction_t dequant_f16_q5k;
    hipFunction_t dequant_f16_q6k;
    hipFunction_t dequant_f16_iq2_xxs;
    hipFunction_t dequant_f16_iq2_xs;
    hipFunction_t dequant_f16_iq2_s;
    hipFunction_t dequant_f16_iq3_xxs;
    hipFunction_t dequant_f16_iq3_s;
    hipFunction_t dequant_f16_iq1_s;
    hipFunction_t dequant_f16_iq1_m;
    hipFunction_t dequant_f16_iq4_nl;
    hipFunction_t dequant_f16_iq4_xs;
    hipFunction_t dequant_f16_mxfp4;
    hipFunction_t dequant_f16_nvfp4;
    hipFunction_t dequant_f16_f32;
    hipFunction_t dequant_f16_bf16;

    bool valid;
};

// ============================================================================
// Scratch buffers (VRAM)
// ============================================================================

struct gfx1100_buffers {
    float * hidden;
    float * residual;
    float * norm_out;
    void  * q8_act;         // block_q8_1 array
    float * proj_scratch;
    float * kv_scratch;
    float * attn_out;
    float * mlp_inter;
    float * logits;
    float * z_scratch;
    float * beta_scratch;
    float * alpha_scratch;

    // DeltaNet persistent state
    float * dn_states;
    float * conv_bufs;

    // Mamba/SSM persistent state
    float * ssm_conv_states; // [n_layers, d_inner, d_conv-1] conv shift register
    float * ssm_scan_states; // [n_layers, d_inner, d_state] selective scan state
    float * ssm_xz;          // [2 * d_inner] scratch for x+z after ssm_in projection
    float * ssm_x_db;        // [dt_rank + 2*d_state] scratch for dt+B+C
    float * ssm_dt;           // [d_inner] scratch for dt after projection

    // RWKV persistent state
    float * rwkv_att_shift;  // [n_layers, n_embd] previous token's att_norm output
    float * rwkv_ffn_shift;  // [n_layers, n_embd] previous token's ffn_norm output
    float * rwkv_wkv_state;  // [n_layers, n_head, head_size, head_size] WKV recurrent state
    // RWKV7 v_first — layer 0's v output, shared across all layers (cross-layer residual)
    float * rwkv7_v_first;    // [n_embd] or NULL (allocated only for RWKV7)
    bool    rwkv7_v_first_set; // true after first layer writes v_first

    // RWKV scratch buffers
    float * rwkv_sx;          // [n_embd] x_prev - cur
    float * rwkv_xxx;         // [n_embd * 5] lerp intermediate (or lora_size * 5)
    float * rwkv_xw;          // [n_embd] mixed x for decay
    float * rwkv_xk;          // [n_embd]
    float * rwkv_xv;          // [n_embd]
    float * rwkv_xr;          // [n_embd]
    float * rwkv_xg;          // [n_embd]

    // KV cache pointer arrays (device)
    void ** d_k_cache_ptrs;
    void ** d_v_cache_ptrs;

    // Grid sync
    unsigned int * barrier_counter;
    unsigned int * barrier_gen;

    // Layer weights (device)
    gfx1100_layer_weights * d_layer_weights;

    // LM head scratch
    float * lm_block_maxv;
    int   * lm_block_maxi;
    int   * moe_sorted_ids;   // [256] GPU buffer for MoE sorted expert indices (avoids mlp_inter aliasing)
    float * moe_probs;        // [256] GPU buffer for MoE routing probabilities (avoids proj_scratch aliasing)

    // Batched MoE buffers (prompt path — group-by-expert dispatch)
    float * batch_moe_probs;    // [max_batch * n_expert] router probs for all tokens (softmax in-place)
    int   * batch_moe_sorted;   // [max_batch * n_expert] sorted expert IDs for all tokens
    int   * moe_expert_counts;  // [n_expert] token count per expert
    int   * moe_token_map;      // [max_batch * n_used * 2] grouped (token_idx, rank) pairs
    int   * output_token;

    // Batch buffers for prompt processing (allocated for max_seq_len tokens)
    float * batch_hidden;      // [max_batch * hidden_size]
    float * batch_norm;        // [max_batch * hidden_size]
    float * batch_residual;    // [max_batch * hidden_size]
    void  * batch_q8_mmq;      // [max_batch * hidden_size * sizeof(block_q8_1_mmq) / QK8_1]
    float * batch_proj;        // [max_batch * max(qproj_size, dn_conv_ch)]
    float * batch_kv;          // [max_batch * 2 * kv_size]
    float * batch_attn_out;    // [max_batch * q_size]
    float * batch_mlp;         // [max_batch * intermediate_size]
    int   * batch_token_ids;   // [max_batch] token IDs on device
    int     max_batch;         // allocated capacity

    // Multi-position logits buffer (for speculative decoding verification)
    // When eval_prompt is called with all_logits=true, logits for every position
    // are written here: batch_logits[pos * vocab_size + tok] = logit
    float * batch_logits;      // [max_batch * vocab_size] or NULL
    int     batch_logits_cap;  // max_batch for which batch_logits is allocated

    // T5 relative position bias scratch (precomputed per decode step)
    float * d_rel_pos_bias;    // [n_head * max_seq_len] or nullptr

    // Audio preprocessing scratch (Whisper mel spectrogram)
    float * d_stft_power;      // [max_audio_frames * (n_fft/2+1)] or nullptr
    float * d_mel_scratch;     // [1] scalar scratch for max_reduce
    int     max_audio_frames;  // capacity of d_stft_power

    // Parallel-block attention scratch
    float * attn_partial;     // [n_q_heads * parallel_blocks * head_dim] partial VKQ
    float * attn_meta;        // [n_q_heads * parallel_blocks * 2] (KQ_max, KQ_sum)
    int     attn_parallel_blocks; // computed at init based on n_q_heads and GPU CU count

    int * d_decode_params;    // [3] GPU-resident: {token_id, position, kv_len} for graph reuse

    hipStream_t stream;
    bool allocated;
};

// ============================================================================
// Global state (single model at a time)
// ============================================================================

extern gfx1100_model_config g_config;
extern gfx1100_compiled     g_compiled;
extern gfx1100_buffers      g_bufs;
extern bool                  g_initialized;
extern rocblas_gemm_state    g_rocblas;

// Utility functions (defined in gfx1100-init.cpp)
bool should_use_mmq(int type, int batch_size);
int get_mmq_q8_1_layout(int weight_type);
int mmq_type_index(int ggml_type);
size_t mmq_shared_mem_size(int ggml_type, int mmq_x);
int gfx1100_init(const gfx1100_model_config * cfg);

// ============================================================================
// DLL-based kernel dispatch — faster than hipModuleLaunchKernel (.hsaco)
//
// DLL mode (default):  hipcc --shared → .dll → hipLaunchKernel via DLL (~1.5µs/launch)
// HSACO mode (fallback): hipcc --genco → .hsaco → hipModuleLaunchKernel (~2.1µs/launch)
//
// The forward files call hipModuleLaunchKernel as before. The macro below
// redirects all calls to gfx1100_dispatch which picks the fast or fallback
// path based on g_dll_mode (set once at init, never changes).
//
// Override: GFX1100_HSACO=1 to force .hsaco fallback.
// ============================================================================

// DLL launch function — exported by decode-dll-wrapper.hip
typedef hipError_t (*gfx1100_dll_launch_fn)(
    const void * fn,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shm, hipStream_t stream, void ** args);

// DLL kernel resolver — exported by decode-dll-wrapper.hip
typedef const void * (*gfx1100_get_kernel_fn)(const char * name);

// Set in gfx1100_init() — constant after init
extern bool                   g_dll_mode;
extern gfx1100_dll_launch_fn  g_dll_launch;

// Unified dispatch — all forward code goes through this.
// In DLL mode: calls hipLaunchKernel inside the DLL (fast path, ~1.5µs)
// In HSACO mode: calls hipModuleLaunchKernel directly (fallback, ~2.1µs)
//
// The branch on g_dll_mode is perfectly predicted (set once, never changes)
// and the function is small enough to be fully inlined by the compiler.
static inline hipError_t gfx1100_dispatch(
    hipFunction_t fn,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shm, hipStream_t stream, void ** args, void * /*extra*/) {
    if (g_dll_mode) {
        return g_dll_launch((const void *)fn, gx, gy, gz, bx, by, bz, shm, stream, args);
    }
    return hipModuleLaunchKernel(fn, gx, gy, gz, bx, by, bz, shm, stream, args, nullptr);
}

// Redirect all hipModuleLaunchKernel calls to our fast dispatch.
// Safe: gfx1100-internal.h is only included by megakernel code,
// and init.cpp uses the real hipModuleLaunchKernel only in the HSACO
// fallback path (via the inline function above, before the macro takes effect).
#define hipModuleLaunchKernel gfx1100_dispatch

// Self-contained model loading — reads GGUF file, populates config, maps tensors to GPU, calls init.
// This is the "full megakernel" entry point — no external caller needed.
// Ported from baseline src/llama-model-loader.cpp and src/llama-model.cpp.
int gfx1100_load_model(const char * model_path, int n_ctx, int n_batch);

// Multimodal preprocessing — image/audio → embeddings on GPU
// These are the preprocessing steps BEFORE the model's encoder layers.
// After preprocessing, the embeddings can be passed to gfx1100_eval_prompt as token embeddings.

// Image preprocessing: raw pixels → patch embeddings on GPU
// Ported from baseline src/llama-vision.cpp
// Input: RGB image as float array [height, width, 3] on GPU
// Output: patch embeddings [n_patches, embed_dim] on GPU, ready for encoder
int gfx1100_preprocess_image(
    const float * image_rgb,  // [H, W, 3] f32 on GPU
    int height, int width,
    float * patch_embeddings, // [n_patches, embed_dim] output on GPU
    int * n_patches);         // output: number of patches

// Audio preprocessing: raw waveform → mel spectrogram embeddings on GPU
// Ported from baseline src/llama-audio.cpp
// Input: raw audio samples [n_samples] f32 on GPU
// Output: mel features [n_frames, n_mels] on GPU, ready for encoder
int gfx1100_preprocess_audio(
    const float * audio_samples, // [n_samples] f32 on GPU
    int n_samples,
    int sample_rate,
    float * mel_features,        // [n_frames, n_mels] output on GPU
    int * n_frames);

// MMQ type table (defined in gfx1100-init.cpp)
struct mmq_type_entry { int ggml_type; const char * name; };
extern const mmq_type_entry mmq_type_table[];
extern const int MMQ_NUM_TYPES;


// Shared inline helpers (extracted to shared/ directory)
#include "shared/matvec-dispatch.h"
#include "shared/embed-dispatch.h"
#include "shared/ssm-step.h"

// Forward declarations for forward functions (defined in forward/*.cpp)
int forward_decode_llama_family(int token_id, int position, float * logits_out);
int forward_decode_mamba(int token_id, int position, float * logits_out);
int forward_decode_rwkv6(int token_id, int position, float * logits_out);
int forward_decode_mamba2(int token_id, int position, float * logits_out);
int forward_decode_rwkv7(int token_id, int position, float * logits_out);
int forward_decode_bitnet(int token_id, int position, float * logits_out);
int forward_decode_cogvlm(int token_id, int position, float * logits_out);
int forward_decode_wavtokenizer(const int * tokens, int n_tokens, float * embd_out);
int forward_encode_t5(const int * tokens, int n_tokens, float * embd_enc_out);
int forward_decode_t5(int token_id, int position, float * logits_out);
int forward_encode_bert(const int * tokens, int n_tokens, float * embd_out);
int forward_decode_deepseek2_mla(int token_id, int position, float * logits_out);
int gfx1100_eval_prompt(const int * token_ids, int n_tokens, int start_pos, float * logits_out);
int gfx1100_eval_decode(int token_id, int position, float * logits_out);

// Speculative decoding support
// Evaluate N tokens and produce logits at EVERY position (not just last)
// logits_out: [n_tokens * vocab_size] f32, row-major (position-major)
// Returns 0 on success. Used for target model verification in speculative decoding.
int gfx1100_eval_verify(const int * token_ids, int n_tokens, int start_pos, float * all_logits_out);

// KV cache rollback: invalidate all KV entries from position p0 onward
// Used after rejecting draft tokens in speculative decoding.
int gfx1100_kv_cache_rollback(int p0);
int gfx1100_kv_cache_seq_shift(int p0, int p1, int delta);
int gfx1100_kv_cache_seq_rm(int p0, int p1, int kv_len);
int gfx1100_kv_cache_seq_cp(int p0, int p1, int dest_offset);

// GPU sampling — logits → token_id entirely on device
// Ported from baseline src/llama-sampling.cpp sampling pipeline
// Batched multi-token decode — process multiple tokens through the full pipeline
// Each token can have a different position (for multi-user serving or continuation).
// Uses the batch prompt path internally with per-token positions.
int gfx1100_eval_decode_batch(
    const int * token_ids,   // [n_tokens] token IDs
    const int * positions,   // [n_tokens] absolute positions per token
    int n_tokens,
    float * logits_out);     // [n_tokens * vocab_size] or NULL (last token only if NULL)

int gfx1100_sample_token(
    float temperature,       // 0 = greedy, >0 = scale logits by 1/temp
    int   top_k,             // 0 = disabled, >0 = keep only top K logits
    float top_p,             // 1.0 = disabled, <1.0 = nucleus sampling cutoff
    float repetition_penalty,// 1.0 = disabled, >1.0 = penalize repeated tokens
    const int * penalty_tokens, // [n_penalty] token IDs to penalize (GPU pointer, or NULL)
    int n_penalty,           // number of penalty tokens
    float random_val,        // uniform random in [0,1) for categorical sampling
    int * out_token);        // output: selected token ID (GPU pointer)
