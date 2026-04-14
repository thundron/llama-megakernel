// test-megakernel-e2e.cpp — End-to-end test for gfx1100 megakernel
//
// Loads a GGUF model, extracts weight pointers, initializes megakernel,
// runs baseline vs megakernel, compares.
//
// Usage: test-megakernel-e2e <model.gguf>

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

// Internal headers — struct layout only, no method calls
#include "llama-model.h"
#include "llama-hparams.h"
#include "llama-arch.h"
#include "../arch_ids.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>
#include <chrono>

// Must match gfx1100-internal.h EXACTLY — ABI boundary
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
    const void * ffn_gate_exps;      // [n_ff, n_expert, n_embd] or NULL
    long long    ffn_gate_exps_stride;
    int          ffn_gate_exps_type;
    const void * ffn_up_exps;        // [n_ff, n_expert, n_embd] or NULL
    long long    ffn_up_exps_stride;
    int          ffn_up_exps_type;
    const void * ffn_down_exps;      // [n_embd, n_expert, n_ff] or NULL
    long long    ffn_down_exps_stride;
    int          ffn_down_exps_type;

    // MoE routing parameters
    long long    ffn_gate_inp_stride;
    int          ffn_gate_inp_type;

    // MoE gating function type
    int          moe_gating_op;
    float        moe_w_scale;
    int          moe_norm_w;

    // Norm biases
    const void * attn_norm_bias;
    const void * ffn_norm_bias;

    // Gemma2/3/4 post-norms
    const void * attn_post_norm;
    const void * ffn_post_norm;

    // BitNet sub-norms
    const void * attn_sub_norm;
    const void * ffn_sub_norm;

    // CogVLM visual expert weights
    const void * visexp_wqkv;
    long long    visexp_wqkv_stride;
    int          visexp_wqkv_type;
    const void * visexp_wo;
    long long    visexp_wo_stride;
    int          visexp_wo_type;
    const void * visexp_ffn_gate;
    long long    visexp_ffn_gate_stride;
    int          visexp_ffn_gate_type;
    const void * visexp_ffn_down;
    long long    visexp_ffn_down_stride;
    int          visexp_ffn_down_type;
    const void * visexp_ffn_up;
    long long    visexp_ffn_up_stride;
    int          visexp_ffn_up_type;

    // Qwen2MoE shared expert
    const void * ffn_gate_inp_shexp;
    long long    ffn_gate_inp_shexp_stride;
    int          ffn_gate_inp_shexp_type;
    const void * ffn_gate_shexp;
    long long    ffn_gate_shexp_stride;
    int          ffn_gate_shexp_type;
    const void * ffn_up_shexp;
    long long    ffn_up_shexp_stride;
    int          ffn_up_shexp_type;
    const void * ffn_down_shexp;
    long long    ffn_down_shexp_stride;
    int          ffn_down_shexp_type;

    // DeepSeek2 MLA layer weights
    const void * wq_a;
    long long    wq_a_stride;
    int          wq_a_type;
    const void * attn_q_a_norm;
    const void * wq_b;
    long long    wq_b_stride;
    int          wq_b_type;
    const void * wkv_a_mqa;
    long long    wkv_a_mqa_stride;
    int          wkv_a_mqa_type;
    const void * attn_kv_a_norm;
    const void * wk_b;
    long long    wk_b_stride;
    int          wk_b_type;
    const void * wv_b;
    long long    wv_b_stride;
    int          wv_b_type;
    const void * wkv_b;
    long long    wkv_b_stride;
    int          wkv_b_type;

    // RWKV layer weights — time-mix
    const void * time_mix_lerp_x;
    const void * time_mix_lerp_fused;
    const void * time_mix_lerp_w;
    const void * time_mix_lerp_k;
    const void * time_mix_lerp_v;
    const void * time_mix_lerp_r;
    const void * time_mix_lerp_g;
    const void * time_mix_w1;
    const void * time_mix_w2;
    long long    time_mix_w1_stride;
    long long    time_mix_w2_stride;
    int          time_mix_w1_type;
    int          time_mix_w2_type;
    const void * time_mix_receptance;
    long long    time_mix_receptance_stride;
    int          time_mix_receptance_type;
    const void * time_mix_key;
    long long    time_mix_key_stride;
    int          time_mix_key_type;
    const void * time_mix_value;
    long long    time_mix_value_stride;
    int          time_mix_value_type;
    const void * time_mix_gate;
    long long    time_mix_gate_stride;
    int          time_mix_gate_type;
    const void * time_mix_output;
    long long    time_mix_output_stride;
    int          time_mix_output_type;
    const void * time_mix_first;
    const void * time_mix_decay;
    const void * time_mix_decay_w1;
    const void * time_mix_decay_w2;
    long long    time_mix_decay_w1_stride;
    long long    time_mix_decay_w2_stride;
    int          time_mix_decay_w1_type;
    int          time_mix_decay_w2_type;
    const void * time_mix_ln;
    const void * time_mix_ln_b;
    const void * time_mix_receptance_b;
    const void * time_mix_key_b;
    const void * time_mix_value_b;

    // RWKV7-specific weights
    const void * time_mix_w0;
    const void * time_mix_a0;
    const void * time_mix_a1;
    const void * time_mix_a2;
    long long    time_mix_a1_stride;
    long long    time_mix_a2_stride;
    int          time_mix_a1_type;
    int          time_mix_a2_type;
    const void * time_mix_v0;
    const void * time_mix_v1;
    const void * time_mix_v2;
    long long    time_mix_v1_stride;
    long long    time_mix_v2_stride;
    int          time_mix_v1_type;
    int          time_mix_v2_type;
    const void * time_mix_g1;
    const void * time_mix_g2;
    long long    time_mix_g1_stride;
    long long    time_mix_g2_stride;
    int          time_mix_g1_type;
    int          time_mix_g2_type;
    const void * time_mix_k_k;
    const void * time_mix_k_a;
    const void * time_mix_r_k;

    // Channel-mix (FFN replacement)
    const void * channel_mix_lerp_k;
    const void * channel_mix_lerp_r;
    const void * channel_mix_receptance;
    long long    channel_mix_receptance_stride;
    int          channel_mix_receptance_type;
    const void * channel_mix_key;
    long long    channel_mix_key_stride;
    int          channel_mix_key_type;
    const void * channel_mix_value;
    long long    channel_mix_value_stride;
    int          channel_mix_value_type;

    // T5 encoder layer weights
    const void * wq_enc;
    long long    wq_enc_stride;
    int          wq_enc_type;
    const void * wk_enc;
    long long    wk_enc_stride;
    int          wk_enc_type;
    const void * wv_enc;
    long long    wv_enc_stride;
    int          wv_enc_type;
    const void * wo_enc;
    long long    wo_enc_stride;
    int          wo_enc_type;
    const void * attn_norm_enc;
    const void * ffn_norm_enc;
    const void * ffn_gate_enc;
    long long    ffn_gate_enc_stride;
    int          ffn_gate_enc_type;
    const void * ffn_down_enc;
    long long    ffn_down_enc_stride;
    int          ffn_down_enc_type;
    const void * ffn_up_enc;
    long long    ffn_up_enc_stride;
    int          ffn_up_enc_type;

    // T5 relative position bias
    const void * attn_rel_b;
    const void * attn_rel_b_enc;

    // T5 cross-attention layer weights
    const void * attn_norm_cross;
    const void * wq_cross;
    long long    wq_cross_stride;
    int          wq_cross_type;
    const void * wk_cross;
    long long    wk_cross_stride;
    int          wk_cross_type;
    const void * wv_cross;
    long long    wv_cross_stride;
    int          wv_cross_type;
    const void * wo_cross;
    long long    wo_cross_stride;
    int          wo_cross_type;

    // BERT post-norm layer weights
    const void * attn_out_norm;
    const void * attn_out_norm_b;
    const void * layer_out_norm;
    const void * layer_out_norm_b;
    const void * attn_norm_2;
    const void * attn_norm_2_b;
    const void * attn_q_norm;
    const void * attn_q_norm_b;
    const void * attn_k_norm;
    const void * attn_k_norm_b;

    // Mamba/SSM layer weights
    const void * ssm_in;
    long long    ssm_in_stride;
    int          ssm_in_type;
    const void * ssm_conv1d;
    const void * ssm_conv1d_b;
    const void * ssm_x;
    long long    ssm_x_stride;
    int          ssm_x_type;
    const void * ssm_dt;
    long long    ssm_dt_stride;
    int          ssm_dt_type;
    const void * ssm_dt_b;
    const void * ssm_a;
    const void * ssm_d;
    const void * ssm_out;
    long long    ssm_out_stride;
    int          ssm_out_type;

    // FalconMamba dt/B/C normalization
    const void * ssm_dt_norm;
    const void * ssm_b_norm;
    const void * ssm_c_norm;

    // Mamba2-specific weights
    const void * ssm_norm;
};

// WavTokenizer PosNet layer weights
struct gfx1100_posnet_layer {
    const void * norm1;
    const void * norm1_b;
    const void * conv1;
    long long    conv1_stride;
    int          conv1_type;
    int          conv1_kernel_size;
    const void * conv1_b;
    const void * norm2;
    const void * norm2_b;
    const void * conv2;
    long long    conv2_stride;
    int          conv2_type;
    int          conv2_kernel_size;
    const void * conv2_b;

    const void * attn_norm;
    const void * attn_norm_b;
    const void * attn_q;
    long long    attn_q_stride;
    int          attn_q_type;
    const void * attn_q_b;
    const void * attn_k;
    long long    attn_k_stride;
    int          attn_k_type;
    const void * attn_k_b;
    const void * attn_v;
    long long    attn_v_stride;
    int          attn_v_type;
    const void * attn_v_b;
    const void * attn_o;
    long long    attn_o_stride;
    int          attn_o_type;
    const void * attn_o_b;
};

// WavTokenizer ConvNext layer weights
struct gfx1100_convnext_layer {
    const void * dw;
    long long    dw_stride;
    int          dw_type;
    int          dw_kernel_size;
    const void * dw_b;
    const void * norm;
    const void * norm_b;
    const void * pw1;
    long long    pw1_stride;
    int          pw1_type;
    const void * pw1_b;
    const void * pw2;
    long long    pw2_stride;
    int          pw2_type;
    const void * pw2_b;
    const void * gamma;
};

struct gfx1100_model_config {
    // --- Architecture identification ---
    int arch_id;

    // --- Capability bits ---
    int has_qk_norm;
    int has_bias_q;
    int has_bias_k;
    int has_bias_v;
    int has_bias_o;
    int has_scale_q;
    int has_scale_k;
    int has_scale_v;
    int has_scale_o;
    int has_bias_ffn_gate;
    int has_bias_ffn_up;
    int has_bias_ffn_down;
    int has_scale_ffn_gate;
    int has_scale_ffn_up;
    int has_scale_ffn_down;
    int rope_type;
    int has_rope_freq_factors;
    int has_moe;
    int moe_n_experts;
    int moe_n_experts_used;
    int has_ssm;
    int has_dn;
    int attn_scale_type;
    float attn_softcap_val;
    int has_final_logit_softcap;
    int has_embed_scale;
    float f_logit_scale;
    float f_residual_scale;
    int cogvlm_is_image;
    int has_swin_norm;
    int chameleon_img_token_start;
    int chameleon_img_token_count;
    float final_logit_softcap_val;
    int norm_type;
    int act_type;
    int pooling_type;
    int has_swa;
    int swa_type;
    int n_swa;
    int has_alibi;
    float alibi_max_bias;
    float alibi_m0;
    float alibi_m1;
    int   alibi_n_head_log2;

    // Parallel attn+FFN
    int use_par_res;

    // Joint QKV projection
    int has_wqkv;

    // Geometry
    int hidden_size;
    int intermediate_size;
    int vocab_size;
    int n_layers;
    int layer_types[128]; // 0=attention, 1=deltanet, 2=ssm, 3=rwkv
    int layer_use_swa[128];

    // Attention
    int fa_n_q_heads;
    int fa_n_kv_heads;
    int fa_head_dim;
    float fa_rope_theta;
    float fa_rope_theta_swa;
    int skip_rope_on_global_layers;
    int per_layer_n_q_heads[128];
    int per_layer_n_kv_heads[128];
    int vision_patch_size;
    const void * rope_freq_factors_per_layer[128];
    int use_shared_norm_ffn;
    int fa_rope_dim;
    int fa_has_gated_attn;
    float fa_attention_scale;
    int fa_use_kq_norm;

    // DeltaNet
    int dn_n_heads;
    int dn_n_k_heads;
    int dn_key_dim;
    int dn_value_dim;
    int dn_conv_kernel;

    // SSM (Mamba/Mamba2)
    int ssm_d_conv;
    int ssm_d_inner;
    int ssm_d_state;
    int ssm_dt_rank;
    int ssm_n_group;

    // RWKV
    int wkv_head_size;
    int rwkv_lora_size;

    // DeepSeek2 MLA
    int mla_kv_lora_rank;
    int mla_q_lora_rank;
    int mla_n_embd_head_qk_rope;
    int mla_n_layer_dense_lead;

    // RoPE (YaRN / scaling)
    float rope_freq_scale;
    float rope_attn_factor;
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    int   n_ctx_orig_yarn;
    int   rope_sections[4];

    // Norm
    float norm_eps;
    int norm_add_one;

    // Weights
    gfx1100_layer_weights layers[128];
    const void * embed_weight;
    long long    embed_stride;
    int          embed_type;
    const void * final_norm_weight;
    const void * final_norm_bias;
    const void * tok_norm_weight;
    const void * tok_norm_bias;
    const void * lm_head_weight;
    long long    lm_head_stride;
    int          lm_head_type;
    const void * rope_freq_factors;

    // T5 encoder model-level weights
    const void * output_norm_enc;
    int          dec_n_layer;
    int          n_rel_attn_bkts;

    // BERT model-level weights
    const void * pos_embd;
    long long    pos_embd_stride;
    int          pos_embd_type;
    const void * type_embd;
    long long    type_embd_stride;
    int          type_embd_type;

    // WavTokenizer
    const void * wav_conv1d;
    long long    wav_conv1d_stride;
    int          wav_conv1d_type;
    int          wav_conv1d_kernel_size;
    const void * wav_conv1d_b;
    const void * wav_output_b;
    gfx1100_posnet_layer   posnet_layers[6];
    gfx1100_convnext_layer convnext_layers[32];
    int          n_convnext_layers;
    int          wav_posnet_n_groups;

    // Audio preprocessing
    const float * mel_filters;
    int audio_n_fft;
    int audio_hop_length;
    int audio_n_mels;

    // Encoder output buffer
    float *      encoder_output;
    int          n_enc_tokens;

    // KV cache
    void * k_cache_ptrs[128];
    void * v_cache_ptrs[128];
    int    kv_stride;
    int    max_seq_len;
    int    kv_type;
};

typedef int (*fn_init_t)(const gfx1100_model_config *);
typedef int (*fn_decode_t)(int token_id, int position, float * logits_out);
typedef int (*fn_is_available_t)(void);
typedef int (*fn_is_ready_t)(void);

static const void * td(const ggml_tensor * t) { return t ? t->data : nullptr; }
static long long ts(const ggml_tensor * t) { return t ? (long long)t->nb[1] : 0; }
static int       tt(const ggml_tensor * t) { return t ? (int)t->type : 0; }

// Map baseline enum llm_arch (src/llama-arch.h) → gfx1100 ARCH_* (arch_ids.h).
// When baseline appends a new arch, add it here too.
static int map_arch_id(llm_arch a) {
    switch (a) {
        case LLM_ARCH_CLIP:             return ARCH_CLIP;
        case LLM_ARCH_LLAMA:            return ARCH_LLAMA;
        case LLM_ARCH_LLAMA4:           return ARCH_LLAMA4;
        case LLM_ARCH_DECI:             return ARCH_DECI;
        case LLM_ARCH_FALCON:           return ARCH_FALCON;
        case LLM_ARCH_BAICHUAN:         return ARCH_BAICHUAN;
        case LLM_ARCH_GROK:             return ARCH_GROK;
        case LLM_ARCH_GPT2:             return ARCH_GPT2;
        case LLM_ARCH_GPTJ:             return ARCH_GPTJ;
        case LLM_ARCH_GPTNEOX:          return ARCH_GPTNEOX;
        case LLM_ARCH_MPT:              return ARCH_MPT;
        case LLM_ARCH_STARCODER:        return ARCH_STARCODER;
        case LLM_ARCH_REFACT:           return ARCH_REFACT;
        case LLM_ARCH_BERT:             return ARCH_BERT;
        case LLM_ARCH_MODERN_BERT:      return ARCH_MODERN_BERT;
        case LLM_ARCH_NOMIC_BERT:       return ARCH_NOMIC_BERT;
        case LLM_ARCH_NOMIC_BERT_MOE:   return ARCH_NOMIC_BERT_MOE;
        case LLM_ARCH_NEO_BERT:         return ARCH_NEO_BERT;
        case LLM_ARCH_JINA_BERT_V2:     return ARCH_JINA_BERT_V2;
        case LLM_ARCH_JINA_BERT_V3:     return ARCH_JINA_BERT_V3;
        case LLM_ARCH_EUROBERT:         return ARCH_EUROBERT;
        case LLM_ARCH_BLOOM:            return ARCH_BLOOM;
        case LLM_ARCH_STABLELM:         return ARCH_STABLELM;
        case LLM_ARCH_QWEN:             return ARCH_QWEN;
        case LLM_ARCH_QWEN2:            return ARCH_QWEN2;
        case LLM_ARCH_QWEN2MOE:         return ARCH_QWEN2MOE;
        case LLM_ARCH_QWEN2VL:          return ARCH_QWEN2VL;
        case LLM_ARCH_QWEN3:            return ARCH_QWEN3;
        case LLM_ARCH_QWEN3MOE:         return ARCH_QWEN3MOE;
        case LLM_ARCH_QWEN3NEXT:        return ARCH_QWEN3NEXT;
        case LLM_ARCH_QWEN3VL:          return ARCH_QWEN3VL;
        case LLM_ARCH_QWEN3VLMOE:       return ARCH_QWEN3VLMOE;
        case LLM_ARCH_QWEN35:           return ARCH_QWEN35;
        case LLM_ARCH_QWEN35MOE:        return ARCH_QWEN35MOE;
        case LLM_ARCH_PHI2:             return ARCH_PHI2;
        case LLM_ARCH_PHI3:             return ARCH_PHI3;
        case LLM_ARCH_PHIMOE:           return ARCH_PHIMOE;
        case LLM_ARCH_PLAMO:            return ARCH_PLAMO;
        case LLM_ARCH_PLAMO2:           return ARCH_PLAMO2;
        case LLM_ARCH_PLAMO3:           return ARCH_PLAMO3;
        case LLM_ARCH_CODESHELL:        return ARCH_CODESHELL;
        case LLM_ARCH_ORION:            return ARCH_ORION;
        case LLM_ARCH_INTERNLM2:        return ARCH_INTERNLM2;
        case LLM_ARCH_MINICPM:          return ARCH_MINICPM;
        case LLM_ARCH_MINICPM3:         return ARCH_MINICPM3;
        case LLM_ARCH_GEMMA:            return ARCH_GEMMA;
        case LLM_ARCH_GEMMA2:           return ARCH_GEMMA2;
        case LLM_ARCH_GEMMA3:           return ARCH_GEMMA3;
        case LLM_ARCH_GEMMA3N:          return ARCH_GEMMA3N;
        case LLM_ARCH_GEMMA4:           return ARCH_GEMMA4;
        case LLM_ARCH_GEMMA_EMBEDDING:  return ARCH_GEMMA_EMBEDDING;
        case LLM_ARCH_STARCODER2:       return ARCH_STARCODER2;
        case LLM_ARCH_MAMBA:            return ARCH_MAMBA;
        case LLM_ARCH_MAMBA2:           return ARCH_MAMBA2;
        case LLM_ARCH_JAMBA:            return ARCH_JAMBA;
        case LLM_ARCH_FALCON_H1:        return ARCH_FALCON_H1;
        case LLM_ARCH_XVERSE:           return ARCH_XVERSE;
        case LLM_ARCH_COMMAND_R:        return ARCH_COMMAND_R;
        case LLM_ARCH_COHERE2:          return ARCH_COHERE2;
        case LLM_ARCH_DBRX:             return ARCH_DBRX;
        case LLM_ARCH_OLMO:             return ARCH_OLMO;
        case LLM_ARCH_OLMO2:            return ARCH_OLMO2;
        case LLM_ARCH_OLMOE:            return ARCH_OLMOE;
        case LLM_ARCH_OPENELM:          return ARCH_OPENELM;
        case LLM_ARCH_ARCTIC:           return ARCH_ARCTIC;
        case LLM_ARCH_DEEPSEEK:         return ARCH_DEEPSEEK;
        case LLM_ARCH_DEEPSEEK2:        return ARCH_DEEPSEEK2;
        case LLM_ARCH_DEEPSEEK2OCR:     return ARCH_DEEPSEEK2OCR;
        case LLM_ARCH_CHATGLM:          return ARCH_CHATGLM;
        case LLM_ARCH_GLM4:             return ARCH_GLM4;
        case LLM_ARCH_GLM4_MOE:         return ARCH_GLM4_MOE;
        case LLM_ARCH_GLM_DSA:          return ARCH_GLM_DSA;
        case LLM_ARCH_BITNET:           return ARCH_BITNET;
        case LLM_ARCH_T5:               return ARCH_T5;
        case LLM_ARCH_T5ENCODER:        return ARCH_T5ENCODER;
        case LLM_ARCH_JAIS:             return ARCH_JAIS;
        case LLM_ARCH_JAIS2:            return ARCH_JAIS2;
        case LLM_ARCH_NEMOTRON:         return ARCH_NEMOTRON;
        case LLM_ARCH_NEMOTRON_H:       return ARCH_NEMOTRON_H;
        case LLM_ARCH_NEMOTRON_H_MOE:   return ARCH_NEMOTRON_H_MOE;
        case LLM_ARCH_EXAONE:           return ARCH_EXAONE;
        case LLM_ARCH_EXAONE4:          return ARCH_EXAONE4;
        case LLM_ARCH_EXAONE_MOE:       return ARCH_EXAONE_MOE;
        case LLM_ARCH_RWKV6:            return ARCH_RWKV6;
        case LLM_ARCH_RWKV6QWEN2:       return ARCH_RWKV6QWEN2;
        case LLM_ARCH_RWKV7:            return ARCH_RWKV7;
        case LLM_ARCH_ARWKV7:           return ARCH_ARWKV7;
        case LLM_ARCH_GRANITE:          return ARCH_GRANITE;
        case LLM_ARCH_GRANITE_MOE:      return ARCH_GRANITE_MOE;
        case LLM_ARCH_GRANITE_HYBRID:   return ARCH_GRANITE_HYBRID;
        case LLM_ARCH_CHAMELEON:        return ARCH_CHAMELEON;
        case LLM_ARCH_WAVTOKENIZER_DEC: return ARCH_WAVTOKENIZER_DEC;
        case LLM_ARCH_PLM:              return ARCH_PLM;
        case LLM_ARCH_BAILINGMOE:       return ARCH_BAILINGMOE;
        case LLM_ARCH_BAILINGMOE2:      return ARCH_BAILINGMOE2;
        case LLM_ARCH_DOTS1:            return ARCH_DOTS1;
        case LLM_ARCH_ARCEE:            return ARCH_ARCEE;
        case LLM_ARCH_AFMOE:            return ARCH_AFMOE;
        case LLM_ARCH_ERNIE4_5:         return ARCH_ERNIE4_5;
        case LLM_ARCH_ERNIE4_5_MOE:     return ARCH_ERNIE4_5_MOE;
        case LLM_ARCH_HUNYUAN_MOE:      return ARCH_HUNYUAN_MOE;
        case LLM_ARCH_HUNYUAN_DENSE:    return ARCH_HUNYUAN_DENSE;
        case LLM_ARCH_SMOLLM3:          return ARCH_SMOLLM3;
        case LLM_ARCH_OPENAI_MOE:       return ARCH_OPENAI_MOE;
        case LLM_ARCH_LFM2:             return ARCH_LFM2;
        case LLM_ARCH_LFM2MOE:          return ARCH_LFM2MOE;
        case LLM_ARCH_DREAM:            return ARCH_DREAM;
        case LLM_ARCH_SMALLTHINKER:     return ARCH_SMALLTHINKER;
        case LLM_ARCH_LLADA:            return ARCH_LLADA;
        case LLM_ARCH_LLADA_MOE:        return ARCH_LLADA_MOE;
        case LLM_ARCH_SEED_OSS:         return ARCH_SEED_OSS;
        case LLM_ARCH_GROVEMOE:         return ARCH_GROVEMOE;
        case LLM_ARCH_APERTUS:          return ARCH_APERTUS;
        case LLM_ARCH_MINIMAX_M2:       return ARCH_MINIMAX_M2;
        case LLM_ARCH_COGVLM:           return ARCH_COGVLM;
        case LLM_ARCH_RND1:             return ARCH_RND1;
        case LLM_ARCH_PANGU_EMBED:      return ARCH_PANGU_EMBED;
        case LLM_ARCH_MISTRAL3:         return ARCH_MISTRAL3;
        case LLM_ARCH_MISTRAL4:         return ARCH_MISTRAL4;
        case LLM_ARCH_PADDLEOCR:        return ARCH_PADDLEOCR;
        case LLM_ARCH_MIMO2:            return ARCH_MIMO2;
        case LLM_ARCH_STEP35:           return ARCH_STEP35;
        case LLM_ARCH_LLAMA_EMBED:      return ARCH_LLAMA_EMBED;
        case LLM_ARCH_MAINCODER:        return ARCH_MAINCODER;
        case LLM_ARCH_KIMI_LINEAR:      return ARCH_KIMI_LINEAR;
        case LLM_ARCH_UNKNOWN:          return ARCH_UNKNOWN;
    }
    return ARCH_UNKNOWN;
}

// Map llama_rope_type (include/llama.h) → ROPE_* (arch_ids.h).
static int map_rope_type(llama_rope_type r) {
    switch (r) {
        case LLAMA_ROPE_TYPE_NONE:   return ROPE_NONE;
        case LLAMA_ROPE_TYPE_NORM:   return ROPE_NORM;
        case LLAMA_ROPE_TYPE_NEOX:   return ROPE_NEOX;
        case LLAMA_ROPE_TYPE_MROPE:  return ROPE_MROPE;
        case LLAMA_ROPE_TYPE_IMROPE: return ROPE_IMROPE;
        case LLAMA_ROPE_TYPE_VISION: return ROPE_VISION;
    }
    return ROPE_NONE;
}

static void fill_attention(gfx1100_layer_weights & lw, const llama_layer & L,
                           const llama_hparams & hp) {
    memset(&lw, 0, sizeof(lw));

    // Core weight matrix slots
    lw.ptrs[0] = td(L.attn_norm);     lw.strides[0] = ts(L.attn_norm);     lw.types[0] = tt(L.attn_norm);
    lw.ptrs[1] = td(L.wq);            lw.strides[1] = ts(L.wq);            lw.types[1] = tt(L.wq);
    lw.ptrs[2] = td(L.wk);            lw.strides[2] = ts(L.wk);            lw.types[2] = tt(L.wk);
    lw.ptrs[3] = td(L.wv);            lw.strides[3] = ts(L.wv);            lw.types[3] = tt(L.wv);
    lw.ptrs[4] = td(L.attn_q_norm);   lw.strides[4] = ts(L.attn_q_norm);   lw.types[4] = tt(L.attn_q_norm);
    lw.ptrs[5] = td(L.attn_k_norm);   lw.strides[5] = ts(L.attn_k_norm);   lw.types[5] = tt(L.attn_k_norm);
    lw.ptrs[6] = td(L.wo);            lw.strides[6] = ts(L.wo);            lw.types[6] = tt(L.wo);
    lw.ptrs[7] = td(L.ffn_norm);      lw.strides[7] = ts(L.ffn_norm);      lw.types[7] = tt(L.ffn_norm);
    // Fused QKV (Phi3, InternLM2, Falcon, GPT-NeoX, etc.)
    if (!lw.ptrs[1] && L.wqkv) {
        const char * base = (const char *)td(L.wqkv);
        long long stride = ts(L.wqkv);
        int q_rows = hp.n_head_arr[0] * hp.n_embd_head_k_full;
        int kv_rows = hp.n_head_kv_arr[0] * hp.n_embd_head_k_full;
        lw.ptrs[1] = base;
        lw.strides[1] = stride;
        lw.types[1] = tt(L.wqkv);
        lw.ptrs[2] = base + (long long)q_rows * stride;
        lw.strides[2] = stride;
        lw.types[2] = tt(L.wqkv);
        lw.ptrs[3] = base + (long long)(q_rows + kv_rows) * stride;
        lw.strides[3] = stride;
        lw.types[3] = tt(L.wqkv);
    }

    lw.ptrs[8] = td(L.ffn_gate);      lw.strides[8] = ts(L.ffn_gate);      lw.types[8] = tt(L.ffn_gate);
    lw.ptrs[9] = td(L.ffn_up);        lw.strides[9] = ts(L.ffn_up);        lw.types[9] = tt(L.ffn_up);
    lw.ptrs[10] = td(L.ffn_down);     lw.strides[10] = ts(L.ffn_down);     lw.types[10] = tt(L.ffn_down);
    lw.ptrs[11] = td(L.ffn_norm);     lw.strides[11] = ts(L.ffn_norm);     lw.types[11] = tt(L.ffn_norm);
    lw.ptrs[12] = td(L.ffn_post_norm);lw.strides[12] = ts(L.ffn_post_norm);lw.types[12] = tt(L.ffn_post_norm);

    // Attention biases
    lw.bias_q = td(L.bq);
    lw.bias_k = td(L.bk);
    lw.bias_v = td(L.bv);
    lw.bias_o = td(L.bo);

    // LoRA / per-tensor scales
    lw.scale_q = td(L.wq_s);
    lw.scale_k = td(L.wk_s);
    lw.scale_v = td(L.wv_s);
    lw.scale_o = td(L.wo_s);

    // FFN biases
    lw.ffn_gate_bias  = td(L.ffn_gate_b);
    lw.ffn_up_bias    = td(L.ffn_up_b);
    lw.ffn_down_bias  = td(L.ffn_down_b);

    // FFN scales
    lw.ffn_gate_scale = td(L.ffn_gate_s);
    lw.ffn_up_scale   = td(L.ffn_up_s);
    lw.ffn_down_scale = td(L.ffn_down_s);

    // MoE detection + expert weights
    lw.ffn_gate_inp        = td(L.ffn_gate_inp);
    lw.ffn_gate_inp_stride = ts(L.ffn_gate_inp);
    lw.ffn_gate_inp_type   = tt(L.ffn_gate_inp);

    lw.ffn_gate_exps        = td(L.ffn_gate_exps);
    lw.ffn_gate_exps_stride = ts(L.ffn_gate_exps);
    lw.ffn_gate_exps_type   = tt(L.ffn_gate_exps);
    lw.ffn_up_exps          = td(L.ffn_up_exps);
    lw.ffn_up_exps_stride   = ts(L.ffn_up_exps);
    lw.ffn_up_exps_type     = tt(L.ffn_up_exps);
    lw.ffn_down_exps        = td(L.ffn_down_exps);
    lw.ffn_down_exps_stride = ts(L.ffn_down_exps);
    lw.ffn_down_exps_type   = tt(L.ffn_down_exps);

    // MoE gating parameters
    lw.moe_gating_op = (int)hp.expert_gating_func;
    lw.moe_w_scale   = hp.expert_weights_scale;
    lw.moe_norm_w    = hp.expert_weights_norm ? 1 : 0;

    // Norm biases (LayerNorm models)
    lw.attn_norm_bias = td(L.attn_norm_b);
    lw.ffn_norm_bias  = td(L.ffn_norm_b);

    // Gemma2/3/4 post-norms
    lw.attn_post_norm = td(L.attn_post_norm);
    lw.ffn_post_norm  = td(L.ffn_post_norm);

    // BitNet sub-norms
    lw.attn_sub_norm = td(L.attn_sub_norm);
    lw.ffn_sub_norm  = td(L.ffn_sub_norm);

    // CogVLM visual expert weights
    lw.visexp_wqkv        = td(L.visexp_attn_wqkv);
    lw.visexp_wqkv_stride = ts(L.visexp_attn_wqkv);
    lw.visexp_wqkv_type   = tt(L.visexp_attn_wqkv);
    lw.visexp_wo           = td(L.visexp_attn_wo);
    lw.visexp_wo_stride    = ts(L.visexp_attn_wo);
    lw.visexp_wo_type      = tt(L.visexp_attn_wo);
    lw.visexp_ffn_gate        = td(L.visexp_ffn_gate);
    lw.visexp_ffn_gate_stride = ts(L.visexp_ffn_gate);
    lw.visexp_ffn_gate_type   = tt(L.visexp_ffn_gate);
    lw.visexp_ffn_down        = td(L.visexp_ffn_down);
    lw.visexp_ffn_down_stride = ts(L.visexp_ffn_down);
    lw.visexp_ffn_down_type   = tt(L.visexp_ffn_down);
    lw.visexp_ffn_up          = td(L.visexp_ffn_up);
    lw.visexp_ffn_up_stride   = ts(L.visexp_ffn_up);
    lw.visexp_ffn_up_type     = tt(L.visexp_ffn_up);

    // Shared expert (Qwen2MoE)
    lw.ffn_gate_inp_shexp        = td(L.ffn_gate_inp_shexp);
    lw.ffn_gate_inp_shexp_stride = ts(L.ffn_gate_inp_shexp);
    lw.ffn_gate_inp_shexp_type   = tt(L.ffn_gate_inp_shexp);
    lw.ffn_gate_shexp        = td(L.ffn_gate_shexp);
    lw.ffn_gate_shexp_stride = ts(L.ffn_gate_shexp);
    lw.ffn_gate_shexp_type   = tt(L.ffn_gate_shexp);
    lw.ffn_up_shexp          = td(L.ffn_up_shexp);
    lw.ffn_up_shexp_stride   = ts(L.ffn_up_shexp);
    lw.ffn_up_shexp_type     = tt(L.ffn_up_shexp);
    lw.ffn_down_shexp        = td(L.ffn_down_shexp);
    lw.ffn_down_shexp_stride = ts(L.ffn_down_shexp);
    lw.ffn_down_shexp_type   = tt(L.ffn_down_shexp);

    // DeepSeek2 MLA
    lw.wq_a           = td(L.wq_a);
    lw.wq_a_stride    = ts(L.wq_a);
    lw.wq_a_type      = tt(L.wq_a);
    lw.attn_q_a_norm  = td(L.attn_q_a_norm);
    lw.wq_b           = td(L.wq_b);
    lw.wq_b_stride    = ts(L.wq_b);
    lw.wq_b_type      = tt(L.wq_b);
    lw.wkv_a_mqa       = td(L.wkv_a_mqa);
    lw.wkv_a_mqa_stride = ts(L.wkv_a_mqa);
    lw.wkv_a_mqa_type   = tt(L.wkv_a_mqa);
    lw.attn_kv_a_norm = td(L.attn_kv_a_norm);
    lw.wk_b           = td(L.wk_b);
    lw.wk_b_stride    = ts(L.wk_b);
    lw.wk_b_type      = tt(L.wk_b);
    lw.wv_b           = td(L.wv_b);
    lw.wv_b_stride    = ts(L.wv_b);
    lw.wv_b_type      = tt(L.wv_b);
    lw.wkv_b          = td(L.wkv_b);
    lw.wkv_b_stride   = ts(L.wkv_b);
    lw.wkv_b_type     = tt(L.wkv_b);

    // T5 encoder layer weights
    lw.wq_enc           = td(L.wq_enc);
    lw.wq_enc_stride    = ts(L.wq_enc);
    lw.wq_enc_type      = tt(L.wq_enc);
    lw.wk_enc           = td(L.wk_enc);
    lw.wk_enc_stride    = ts(L.wk_enc);
    lw.wk_enc_type      = tt(L.wk_enc);
    lw.wv_enc           = td(L.wv_enc);
    lw.wv_enc_stride    = ts(L.wv_enc);
    lw.wv_enc_type      = tt(L.wv_enc);
    lw.wo_enc           = td(L.wo_enc);
    lw.wo_enc_stride    = ts(L.wo_enc);
    lw.wo_enc_type      = tt(L.wo_enc);
    lw.attn_norm_enc    = td(L.attn_norm_enc);
    lw.ffn_norm_enc     = td(L.ffn_norm_enc);
    lw.ffn_gate_enc        = td(L.ffn_gate_enc);
    lw.ffn_gate_enc_stride = ts(L.ffn_gate_enc);
    lw.ffn_gate_enc_type   = tt(L.ffn_gate_enc);
    lw.ffn_down_enc        = td(L.ffn_down_enc);
    lw.ffn_down_enc_stride = ts(L.ffn_down_enc);
    lw.ffn_down_enc_type   = tt(L.ffn_down_enc);
    lw.ffn_up_enc          = td(L.ffn_up_enc);
    lw.ffn_up_enc_stride   = ts(L.ffn_up_enc);
    lw.ffn_up_enc_type     = tt(L.ffn_up_enc);

    // T5 relative position bias
    lw.attn_rel_b     = td(L.attn_rel_b);
    lw.attn_rel_b_enc = td(L.attn_rel_b_enc);

    // T5 cross-attention
    lw.attn_norm_cross    = td(L.attn_norm_cross);
    lw.wq_cross           = td(L.wq_cross);
    lw.wq_cross_stride    = ts(L.wq_cross);
    lw.wq_cross_type      = tt(L.wq_cross);
    lw.wk_cross           = td(L.wk_cross);
    lw.wk_cross_stride    = ts(L.wk_cross);
    lw.wk_cross_type      = tt(L.wk_cross);
    lw.wv_cross           = td(L.wv_cross);
    lw.wv_cross_stride    = ts(L.wv_cross);
    lw.wv_cross_type      = tt(L.wv_cross);
    lw.wo_cross           = td(L.wo_cross);
    lw.wo_cross_stride    = ts(L.wo_cross);
    lw.wo_cross_type      = tt(L.wo_cross);

    // BERT post-norm weights
    lw.attn_out_norm   = td(L.attn_out_norm);
    lw.attn_out_norm_b = td(L.attn_out_norm_b);
    lw.layer_out_norm   = td(L.layer_out_norm);
    lw.layer_out_norm_b = td(L.layer_out_norm_b);
    lw.attn_norm_2      = td(L.attn_norm_2);
    lw.attn_norm_2_b    = td(L.attn_norm_2_b);
    lw.attn_q_norm      = td(L.attn_q_norm);
    lw.attn_q_norm_b    = td(L.attn_q_norm_b);
    lw.attn_k_norm      = td(L.attn_k_norm);
    lw.attn_k_norm_b    = td(L.attn_k_norm_b);
}

static void fill_deltanet(gfx1100_layer_weights & lw, const llama_layer & L) {
    memset(&lw, 0, sizeof(lw));
    lw.ptrs[0] = td(L.attn_norm);     lw.strides[0] = ts(L.attn_norm);     lw.types[0] = tt(L.attn_norm);
    auto * qkv = L.wqkv ? L.wqkv : L.ssm_in;
    lw.ptrs[1] = td(qkv);             lw.strides[1] = ts(qkv);             lw.types[1] = tt(qkv);
    lw.ptrs[2] = td(L.wqkv_gate);     lw.strides[2] = ts(L.wqkv_gate);     lw.types[2] = tt(L.wqkv_gate);
    auto * beta = L.ssm_beta ? L.ssm_beta : L.ssm_beta_alpha;
    lw.ptrs[3] = td(beta);            lw.strides[3] = ts(beta);            lw.types[3] = tt(beta);
    lw.ptrs[4] = td(L.ssm_alpha);     lw.strides[4] = ts(L.ssm_alpha);     lw.types[4] = tt(L.ssm_alpha);
    lw.ptrs[5] = td(L.ssm_conv1d);    lw.strides[5] = ts(L.ssm_conv1d);    lw.types[5] = tt(L.ssm_conv1d);
    lw.ptrs[6] = td(L.ssm_a);         lw.strides[6] = ts(L.ssm_a);         lw.types[6] = tt(L.ssm_a);
    lw.ptrs[7] = td(L.ssm_dt);        lw.strides[7] = ts(L.ssm_dt);        lw.types[7] = tt(L.ssm_dt);
    lw.ptrs[8] = td(L.ssm_norm);      lw.strides[8] = ts(L.ssm_norm);      lw.types[8] = tt(L.ssm_norm);
    lw.ptrs[9] = td(L.ssm_out);       lw.strides[9] = ts(L.ssm_out);       lw.types[9] = tt(L.ssm_out);
    lw.ptrs[10] = td(L.attn_post_norm); lw.strides[10] = ts(L.attn_post_norm); lw.types[10] = tt(L.attn_post_norm);
    lw.ptrs[11] = td(L.ffn_gate);     lw.strides[11] = ts(L.ffn_gate);      lw.types[11] = tt(L.ffn_gate);
    lw.ptrs[12] = td(L.ffn_up);       lw.strides[12] = ts(L.ffn_up);        lw.types[12] = tt(L.ffn_up);
    lw.ptrs[13] = td(L.ffn_down);     lw.strides[13] = ts(L.ffn_down);      lw.types[13] = tt(L.ffn_down);
}

static void fill_ssm(gfx1100_layer_weights & lw, const llama_layer & L) {
    memset(&lw, 0, sizeof(lw));

    // ptrs[0] = attn_norm (layer norm before SSM block)
    lw.ptrs[0] = td(L.attn_norm);     lw.strides[0] = ts(L.attn_norm);     lw.types[0] = tt(L.attn_norm);

    // FFN slots (used by hybrid SSM+FFN models like Jamba)
    lw.ptrs[7] = td(L.ffn_norm);      lw.strides[7] = ts(L.ffn_norm);      lw.types[7] = tt(L.ffn_norm);
    lw.ptrs[8] = td(L.ffn_gate);      lw.strides[8] = ts(L.ffn_gate);      lw.types[8] = tt(L.ffn_gate);
    lw.ptrs[9] = td(L.ffn_up);        lw.strides[9] = ts(L.ffn_up);        lw.types[9] = tt(L.ffn_up);
    lw.ptrs[10] = td(L.ffn_down);     lw.strides[10] = ts(L.ffn_down);     lw.types[10] = tt(L.ffn_down);

    // Mamba/SSM weights
    lw.ssm_in        = td(L.ssm_in);
    lw.ssm_in_stride = ts(L.ssm_in);
    lw.ssm_in_type   = tt(L.ssm_in);
    lw.ssm_conv1d    = td(L.ssm_conv1d);
    lw.ssm_conv1d_b  = td(L.ssm_conv1d_b);
    lw.ssm_x         = td(L.ssm_x);
    lw.ssm_x_stride  = ts(L.ssm_x);
    lw.ssm_x_type    = tt(L.ssm_x);
    lw.ssm_dt        = td(L.ssm_dt);
    lw.ssm_dt_stride = ts(L.ssm_dt);
    lw.ssm_dt_type   = tt(L.ssm_dt);
    lw.ssm_dt_b      = td(L.ssm_dt_b);
    lw.ssm_a         = td(L.ssm_a);
    lw.ssm_d         = td(L.ssm_d);
    lw.ssm_out        = td(L.ssm_out);
    lw.ssm_out_stride = ts(L.ssm_out);
    lw.ssm_out_type   = tt(L.ssm_out);

    // FalconMamba dt/B/C normalization
    lw.ssm_dt_norm = td(L.ssm_dt_norm);
    lw.ssm_b_norm  = td(L.ssm_b_norm);
    lw.ssm_c_norm  = td(L.ssm_c_norm);

    // Mamba2 grouped RMSNorm
    lw.ssm_norm = td(L.ssm_norm);

    // Norm biases (for LayerNorm models)
    lw.attn_norm_bias = td(L.attn_norm_b);
    lw.ffn_norm_bias  = td(L.ffn_norm_b);

    // MoE for hybrid models (Jamba uses MoE in SSM layers)
    lw.ffn_gate_inp        = td(L.ffn_gate_inp);
    lw.ffn_gate_inp_stride = ts(L.ffn_gate_inp);
    lw.ffn_gate_inp_type   = tt(L.ffn_gate_inp);
    lw.ffn_gate_exps        = td(L.ffn_gate_exps);
    lw.ffn_gate_exps_stride = ts(L.ffn_gate_exps);
    lw.ffn_gate_exps_type   = tt(L.ffn_gate_exps);
    lw.ffn_up_exps          = td(L.ffn_up_exps);
    lw.ffn_up_exps_stride   = ts(L.ffn_up_exps);
    lw.ffn_up_exps_type     = tt(L.ffn_up_exps);
    lw.ffn_down_exps        = td(L.ffn_down_exps);
    lw.ffn_down_exps_stride = ts(L.ffn_down_exps);
    lw.ffn_down_exps_type   = tt(L.ffn_down_exps);
}

static void fill_rwkv(gfx1100_layer_weights & lw, const llama_layer & L) {
    memset(&lw, 0, sizeof(lw));

    // ptrs[0] = attn_norm (layer norm before time-mix)
    lw.ptrs[0] = td(L.attn_norm);     lw.strides[0] = ts(L.attn_norm);     lw.types[0] = tt(L.attn_norm);

    // FFN/channel-mix norm + weights
    lw.ptrs[7] = td(L.ffn_norm);      lw.strides[7] = ts(L.ffn_norm);      lw.types[7] = tt(L.ffn_norm);
    lw.ptrs[8] = td(L.ffn_gate);      lw.strides[8] = ts(L.ffn_gate);      lw.types[8] = tt(L.ffn_gate);
    lw.ptrs[9] = td(L.ffn_up);        lw.strides[9] = ts(L.ffn_up);        lw.types[9] = tt(L.ffn_up);
    lw.ptrs[10] = td(L.ffn_down);     lw.strides[10] = ts(L.ffn_down);     lw.types[10] = tt(L.ffn_down);

    // Norm biases (RWKV uses LayerNorm)
    lw.attn_norm_bias = td(L.attn_norm_b);
    lw.ffn_norm_bias  = td(L.ffn_norm_b);

    // Time-mix lerp weights
    lw.time_mix_lerp_x     = td(L.time_mix_lerp_x);
    lw.time_mix_lerp_fused = td(L.time_mix_lerp_fused);
    lw.time_mix_lerp_w     = td(L.time_mix_lerp_w);
    lw.time_mix_lerp_k     = td(L.time_mix_lerp_k);
    lw.time_mix_lerp_v     = td(L.time_mix_lerp_v);
    lw.time_mix_lerp_r     = td(L.time_mix_lerp_r);
    lw.time_mix_lerp_g     = td(L.time_mix_lerp_g);

    // Time-mix LoRA
    lw.time_mix_w1        = td(L.time_mix_w1);
    lw.time_mix_w2        = td(L.time_mix_w2);
    lw.time_mix_w1_stride = ts(L.time_mix_w1);
    lw.time_mix_w2_stride = ts(L.time_mix_w2);
    lw.time_mix_w1_type   = tt(L.time_mix_w1);
    lw.time_mix_w2_type   = tt(L.time_mix_w2);

    // Time-mix projections
    lw.time_mix_receptance        = td(L.time_mix_receptance);
    lw.time_mix_receptance_stride = ts(L.time_mix_receptance);
    lw.time_mix_receptance_type   = tt(L.time_mix_receptance);
    lw.time_mix_key               = td(L.time_mix_key);
    lw.time_mix_key_stride        = ts(L.time_mix_key);
    lw.time_mix_key_type          = tt(L.time_mix_key);
    lw.time_mix_value             = td(L.time_mix_value);
    lw.time_mix_value_stride      = ts(L.time_mix_value);
    lw.time_mix_value_type        = tt(L.time_mix_value);
    lw.time_mix_gate              = td(L.time_mix_gate);
    lw.time_mix_gate_stride       = ts(L.time_mix_gate);
    lw.time_mix_gate_type         = tt(L.time_mix_gate);
    lw.time_mix_output            = td(L.time_mix_output);
    lw.time_mix_output_stride     = ts(L.time_mix_output);
    lw.time_mix_output_type       = tt(L.time_mix_output);

    // Time-mix decay + first
    lw.time_mix_first      = td(L.time_mix_first);
    lw.time_mix_decay      = td(L.time_mix_decay);
    lw.time_mix_decay_w1        = td(L.time_mix_decay_w1);
    lw.time_mix_decay_w2        = td(L.time_mix_decay_w2);
    lw.time_mix_decay_w1_stride = ts(L.time_mix_decay_w1);
    lw.time_mix_decay_w2_stride = ts(L.time_mix_decay_w2);
    lw.time_mix_decay_w1_type   = tt(L.time_mix_decay_w1);
    lw.time_mix_decay_w2_type   = tt(L.time_mix_decay_w2);

    // Time-mix group norm
    lw.time_mix_ln   = td(L.time_mix_ln);
    lw.time_mix_ln_b = td(L.time_mix_ln_b);

    // Optional biases
    lw.time_mix_receptance_b = td(L.time_mix_receptance_b);
    lw.time_mix_key_b        = td(L.time_mix_key_b);
    lw.time_mix_value_b      = td(L.time_mix_value_b);

    // RWKV7-specific weights
    lw.time_mix_w0        = td(L.time_mix_w0);
    lw.time_mix_a0        = td(L.time_mix_a0);
    lw.time_mix_a1        = td(L.time_mix_a1);
    lw.time_mix_a2        = td(L.time_mix_a2);
    lw.time_mix_a1_stride = ts(L.time_mix_a1);
    lw.time_mix_a2_stride = ts(L.time_mix_a2);
    lw.time_mix_a1_type   = tt(L.time_mix_a1);
    lw.time_mix_a2_type   = tt(L.time_mix_a2);
    lw.time_mix_v0        = td(L.time_mix_v0);
    lw.time_mix_v1        = td(L.time_mix_v1);
    lw.time_mix_v2        = td(L.time_mix_v2);
    lw.time_mix_v1_stride = ts(L.time_mix_v1);
    lw.time_mix_v2_stride = ts(L.time_mix_v2);
    lw.time_mix_v1_type   = tt(L.time_mix_v1);
    lw.time_mix_v2_type   = tt(L.time_mix_v2);
    lw.time_mix_g1        = td(L.time_mix_g1);
    lw.time_mix_g2        = td(L.time_mix_g2);
    lw.time_mix_g1_stride = ts(L.time_mix_g1);
    lw.time_mix_g2_stride = ts(L.time_mix_g2);
    lw.time_mix_g1_type   = tt(L.time_mix_g1);
    lw.time_mix_g2_type   = tt(L.time_mix_g2);
    lw.time_mix_k_k       = td(L.time_mix_k_k);
    lw.time_mix_k_a       = td(L.time_mix_k_a);
    lw.time_mix_r_k       = td(L.time_mix_r_k);

    // Channel-mix weights
    lw.channel_mix_lerp_k = td(L.channel_mix_lerp_k);
    lw.channel_mix_lerp_r = td(L.channel_mix_lerp_r);
    lw.channel_mix_receptance        = td(L.channel_mix_receptance);
    lw.channel_mix_receptance_stride = ts(L.channel_mix_receptance);
    lw.channel_mix_receptance_type   = tt(L.channel_mix_receptance);
    lw.channel_mix_key               = td(L.channel_mix_key);
    lw.channel_mix_key_stride        = ts(L.channel_mix_key);
    lw.channel_mix_key_type          = tt(L.channel_mix_key);
    lw.channel_mix_value             = td(L.channel_mix_value);
    lw.channel_mix_value_stride      = ts(L.channel_mix_value);
    lw.channel_mix_value_type        = tt(L.channel_mix_value);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];

    // ---- Load model ----
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    assert(model && "Failed to load model");

    // Read hparams directly from struct fields (no method calls)
    const auto & hp = model->hparams;
    int n_layer = hp.n_layer;
    int n_embd  = hp.n_embd;
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // ---- Build megakernel config ----
    gfx1100_model_config cfg = {};

    // Architecture + RoPE ids (compile-time specialization driver)
    cfg.arch_id   = map_arch_id(model->arch);
    cfg.rope_type = map_rope_type((llama_rope_type)hp.rope_type);

    cfg.hidden_size       = n_embd;
    cfg.intermediate_size = hp.n_ff_arr[0];
    cfg.vocab_size        = n_vocab;
    cfg.n_layers          = n_layer;
    cfg.norm_eps          = hp.f_norm_rms_eps;
    cfg.norm_add_one      = 0;
    cfg.fa_n_q_heads      = hp.n_head_arr[0];
    cfg.fa_n_kv_heads     = hp.n_head_kv_arr[0];
    cfg.fa_head_dim       = hp.n_embd_head_k_full;
    cfg.fa_rope_theta     = hp.rope_freq_base_train;
    cfg.fa_rope_dim       = hp.n_embd_head_k_full;
    cfg.fa_has_gated_attn = 0;
    cfg.fa_attention_scale = hp.f_attention_scale; // baseline line 27: 0 means 1/sqrt(d)
    cfg.fa_use_kq_norm = hp.use_kq_norm ? 1 : 0;  // baseline line 84
    cfg.dn_n_heads        = hp.ssm_dt_rank;
    cfg.dn_n_k_heads      = hp.ssm_n_group;
    cfg.dn_key_dim        = hp.ssm_d_state;
    cfg.dn_value_dim      = (hp.ssm_dt_rank > 0) ? (int)(hp.ssm_d_inner / hp.ssm_dt_rank) : 0;
    cfg.dn_conv_kernel    = hp.ssm_d_conv;
    // Default 8192 to support iSWA testing beyond n_swa=4096.
    // Override with 5th argument: test-megakernel-e2e <model> [baseline] [unused] [n_gen] [max_seq]
    cfg.max_seq_len       = argc >= 6 ? atoi(argv[5]) : 8192;
    // NOTE: max_seq_len affects SU-RoPE freq factor selection (rope_long vs rope_short).
    // Baseline test-baseline-logits uses n_ctx=2048. For fair comparison, the freq factor
    // selection below uses max_seq_len to decide, matching baseline's get_rope_factors() logic.
    cfg.kv_stride         = hp.n_embd_head_k_full;

    // SSM params — direct copy from baseline hparams
    cfg.ssm_d_conv  = hp.ssm_d_conv;
    cfg.ssm_d_inner = hp.ssm_d_inner;
    cfg.ssm_d_state = hp.ssm_d_state;
    cfg.ssm_dt_rank = hp.ssm_dt_rank;
    cfg.ssm_n_group = hp.ssm_n_group;

    // RoPE scaling / YaRN — resolve ext_factor like baseline (llama-context.cpp:96-97)
    // Raw hparams value is -1.0 (unset sentinel); baseline resolves to 1.0 for YARN, 0.0 otherwise.
    cfg.rope_freq_scale   = hp.rope_freq_scale_train;
    cfg.rope_attn_factor  = hp.rope_attn_factor;
    {
        float ext = hp.yarn_ext_factor;
        if (ext < 0.0f) {
            ext = (hp.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_YARN) ? 1.0f : 0.0f;
        }
        cfg.yarn_ext_factor = ext;
    }
    // Resolve yarn_attn_factor like baseline (llama-context.cpp:100-137).
    // For LONGROPE (ext_factor=0): yaf = 1.0 * rope_attn_factor
    // For YARN (ext_factor!=0): yaf = get_mscale(...) * rope_attn_factor
    // Missing the rope_attn_factor multiplication caused Phi-3.5 to be 4/100 wrong.
    {
        float yaf = hp.yarn_attn_factor;  // raw hparams (typically 1.0)
        if (cfg.yarn_ext_factor != 0.0f) {
            float factor = 1.0f / cfg.rope_freq_scale;
            if (factor > 1.0f) {
                yaf = (0.1f * logf(factor) + 1.0f);  // get_mscale with default m=1
            }
        }
        yaf *= hp.rope_attn_factor;  // THE FIX: Phi-3.5 has rope_attn_factor=1.19
        cfg.yarn_attn_factor = yaf;
    }
    cfg.yarn_beta_fast    = hp.yarn_beta_fast;
    cfg.yarn_beta_slow    = hp.yarn_beta_slow;
    cfg.n_ctx_orig_yarn   = (int)hp.n_ctx_orig_yarn;
    for (int i = 0; i < 4; i++) cfg.rope_sections[i] = hp.rope_sections[i];

    // Attention scale type (baseline: f_attn_logit_softcapping for Gemma2, f_attention_scale for Granite)
    if (hp.attn_soft_cap) {
        cfg.attn_scale_type = ATTN_SCALE_SOFTCAP;
        cfg.attn_softcap_val = hp.f_attn_logit_softcapping;
    } else if (hp.f_attention_scale != 0.0f) {
        cfg.attn_scale_type = ATTN_SCALE_CUSTOM;
    } else {
        cfg.attn_scale_type = ATTN_SCALE_DEFAULT;
    }
    switch (model->arch) {
        case LLM_ARCH_GEMMA2: case LLM_ARCH_GEMMA3: case LLM_ARCH_GEMMA3N:
        case LLM_ARCH_GEMMA4:
            if (hp.f_final_logit_softcapping > 0.0f) {
                cfg.has_final_logit_softcap   = 1;
                cfg.final_logit_softcap_val   = hp.f_final_logit_softcapping;
            }
            break;
        default: break;
    }

    // MoE — baseline: hparams.n_expert, hparams.n_expert_used
    cfg.moe_n_experts      = (int)hp.n_expert;
    cfg.moe_n_experts_used = (int)hp.n_expert_used;

    // SWA (sliding window) — baseline: hparams.swa_type / n_swa
    cfg.swa_type = (int)hp.swa_type;
    cfg.n_swa    = (int)hp.n_swa;
    cfg.has_swa  = (hp.swa_type != LLAMA_SWA_TYPE_NONE) ? 1 : 0;

    // ALiBi — baseline: hparams.use_alibi
    cfg.has_alibi = hp.use_alibi ? 1 : 0;
    if (hp.use_alibi) {
        cfg.alibi_max_bias = hp.f_max_alibi_bias;
        int n_head = (int)hp.n_head_arr[0];
        int n_head_log2 = 1;
        while (n_head_log2 * 2 <= n_head) n_head_log2 *= 2;
        cfg.alibi_n_head_log2 = n_head_log2;
        cfg.alibi_m0 = powf(2.0f, -hp.f_max_alibi_bias / (float)n_head_log2);
        cfg.alibi_m1 = powf(2.0f, -hp.f_max_alibi_bias / 2.0f / (float)n_head_log2);
    }

    // Parallel attn+FFN — baseline: hparams.use_par_res
    cfg.use_par_res = hp.use_par_res ? 1 : 0;

    // Pooling — baseline: hparams.pooling_type
    cfg.pooling_type = (int)hp.pooling_type;

    // Logit/residual/embedding scales — Granite, Cohere, Gemma, Grok
    cfg.f_logit_scale    = hp.f_logit_scale;
    cfg.f_residual_scale = hp.f_residual_scale;

    // Chameleon image token suppression
    cfg.has_swin_norm = hp.swin_norm ? 1 : 0;

    // SWA-layer RoPE theta
    cfg.fa_rope_theta_swa = hp.rope_freq_base_train_swa;

    // RWKV params
    cfg.wkv_head_size  = (int)hp.wkv_head_size;
    cfg.rwkv_lora_size = (int)hp.time_mix_extra_dim;

    // DeepSeek2 MLA params
    cfg.mla_kv_lora_rank       = (int)hp.n_lora_kv;
    cfg.mla_q_lora_rank        = (int)hp.n_lora_q;
    cfg.mla_n_embd_head_qk_rope = (int)hp.n_rot_full;
    cfg.mla_n_layer_dense_lead = (int)hp.n_layer_dense_lead;

    // T5 encoder-decoder params
    cfg.dec_n_layer     = (int)hp.dec_n_layer;
    cfg.n_rel_attn_bkts = (int)hp.n_rel_attn_bkts;

    // Norm type: default RMS. LayerNorm-using archs are mapped explicitly.
    // (BERT family, Falcon, MPT, GPT2/NeoX, Bloom, StarCoder, Phi2, OpenELM, etc.)
    switch (model->arch) {
        case LLM_ARCH_BERT: case LLM_ARCH_MODERN_BERT: case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE: case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_JINA_BERT_V2: case LLM_ARCH_JINA_BERT_V3: case LLM_ARCH_EUROBERT:
        case LLM_ARCH_FALCON: case LLM_ARCH_MPT: case LLM_ARCH_GPT2:
        case LLM_ARCH_GPTJ: case LLM_ARCH_GPTNEOX: case LLM_ARCH_BLOOM:
        case LLM_ARCH_STARCODER: case LLM_ARCH_STARCODER2:
        case LLM_ARCH_PHI2: case LLM_ARCH_OPENELM: case LLM_ARCH_JAIS:
        case LLM_ARCH_JAIS2: case LLM_ARCH_REFACT: case LLM_ARCH_CODESHELL:
            cfg.norm_type = NORM_LAYER; cfg.norm_eps = hp.f_norm_eps; break;
        case LLM_ARCH_QWEN3: case LLM_ARCH_QWEN35: case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN35MOE: case LLM_ARCH_QWEN3NEXT: case LLM_ARCH_QWEN3VL:
        case LLM_ARCH_QWEN3VLMOE:
            cfg.norm_type = NORM_RMS; break;  // Qwen3 uses RMS; L2 is only on conv output
        default:
            cfg.norm_type = NORM_RMS; break;
    }

    // Activation type — mostly SiLU (SwiGLU). GELU for Gemma/Phi/Bert. Overrides below.
    switch (model->arch) {
        case LLM_ARCH_GEMMA: case LLM_ARCH_GEMMA2: case LLM_ARCH_GEMMA3:
        case LLM_ARCH_GEMMA3N: case LLM_ARCH_GEMMA4: case LLM_ARCH_GEMMA_EMBEDDING:
            cfg.act_type = ACT_GELU_TANH; break;
        case LLM_ARCH_PHI2:
            cfg.act_type = ACT_GELU; break;
        case LLM_ARCH_BERT: case LLM_ARCH_MODERN_BERT: case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE: case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_JINA_BERT_V2: case LLM_ARCH_JINA_BERT_V3: case LLM_ARCH_EUROBERT:
            cfg.act_type = ACT_GELU; break;
        case LLM_ARCH_BLOOM: case LLM_ARCH_FALCON: case LLM_ARCH_MPT:
        case LLM_ARCH_GPT2: case LLM_ARCH_GPTNEOX: case LLM_ARCH_STARCODER:
            cfg.act_type = ACT_GELU; break;
        default:
            cfg.act_type = ACT_SILU; break;
    }

    for (int il = 0; il < n_layer; il++) {
        const llama_layer & L = model->layers[il];
        bool recurrent = hp.recurrent_layer_arr[il];

        // Classify layer type: 0=attention, 1=deltanet, 2=ssm, 3=rwkv
        if (!recurrent) {
            cfg.layer_types[il] = 0; // attention
            fill_attention(cfg.layers[il], L, hp);
        } else if (L.time_mix_key || L.time_mix_receptance) {
            cfg.layer_types[il] = 3; // rwkv
            fill_rwkv(cfg.layers[il], L);
        } else if (L.ssm_in || L.ssm_conv1d) {
            cfg.layer_types[il] = 2; // ssm (Mamba/Mamba2)
            fill_ssm(cfg.layers[il], L);
        } else {
            cfg.layer_types[il] = 1; // deltanet
            fill_deltanet(cfg.layers[il], L);
        }

        // SWA per-layer flag
        cfg.layer_use_swa[il] = hp.swa_layers[il] ? 1 : 0;

        // Per-layer head counts (OpenELM)
        cfg.per_layer_n_q_heads[il]  = (int)hp.n_head_arr[il];
        cfg.per_layer_n_kv_heads[il] = (int)hp.n_head_kv_arr[il];

        // Per-layer RoPE freq factors (Phi3 SU-RoPE)
        // Logic mirrors llama_model::get_rope_factors():
        //   rope_freqs > rope_long (if n_ctx > n_ctx_orig_yarn) > rope_short
        if (L.rope_freqs) {
            cfg.rope_freq_factors_per_layer[il] = td(L.rope_freqs);
        } else if ((int)cfg.max_seq_len > (int)hp.n_ctx_orig_yarn) {
            cfg.rope_freq_factors_per_layer[il] = td(L.rope_long);
        } else {
            cfg.rope_freq_factors_per_layer[il] = td(L.rope_short);
        }

        // Detect gated attention
        if (!recurrent && L.wq) {
            int64_t q_out = L.wq->ne[1];
            if (q_out > (int64_t)(hp.n_head_arr[il] * hp.n_embd_head_k_full))
                cfg.fa_has_gated_attn = 1;
        }

        // Detect joint QKV
        if (L.wqkv) cfg.has_wqkv = 1;

        // Derive capability bits (OR across all layers)
        if (L.attn_q_norm || L.attn_k_norm) cfg.has_qk_norm = 1;
        if (L.bq) cfg.has_bias_q = 1;
        if (L.bk) cfg.has_bias_k = 1;
        if (L.bv) cfg.has_bias_v = 1;
        if (L.bo) cfg.has_bias_o = 1;
        if (L.wq_s) cfg.has_scale_q = 1;
        if (L.wk_s) cfg.has_scale_k = 1;
        if (L.wv_s) cfg.has_scale_v = 1;
        if (L.wo_s) cfg.has_scale_o = 1;
        if (L.ffn_gate_b) cfg.has_bias_ffn_gate = 1;
        if (L.ffn_up_b)   cfg.has_bias_ffn_up   = 1;
        if (L.ffn_down_b) cfg.has_bias_ffn_down = 1;
        if (L.ffn_gate_s) cfg.has_scale_ffn_gate = 1;
        if (L.ffn_up_s)   cfg.has_scale_ffn_up   = 1;
        if (L.ffn_down_s) cfg.has_scale_ffn_down = 1;
        if (L.ffn_gate_inp) cfg.has_moe = 1;
        if (L.ssm_in || L.ssm_conv1d || L.ssm_a) cfg.has_ssm = 1;
        if (recurrent && !L.ssm_a && !L.time_mix_key) cfg.has_dn = 1;
    }
    cfg.has_rope_freq_factors = (n_layer > 0 && (model->layers[0].rope_freqs || model->layers[0].rope_long || model->layers[0].rope_short)) ? 1 : 0;

    // NOTE: Gemma's original PyTorch uses (1 + weight) in RMSNorm, but
    // convert_hf_to_gguf.py already bakes the +1 into the GGUF weights
    // (see Gemma2Model.modify_tensors / Gemma3Model.norm_shift).
    // norm_add_one stays 0 for all models — the offset is pre-applied.

    cfg.embed_weight      = td(model->tok_embd);
    cfg.embed_stride      = ts(model->tok_embd);
    cfg.embed_type        = model->tok_embd ? model->tok_embd->type : 0;
    cfg.final_norm_weight = td(model->output_norm);
    cfg.final_norm_bias   = td(model->output_norm_b);
    cfg.tok_norm_weight   = td(model->tok_norm);
    cfg.tok_norm_bias     = td(model->tok_norm_b);
    cfg.lm_head_weight    = td(model->output);
    cfg.lm_head_stride    = ts(model->output);
    cfg.lm_head_type      = model->output ? (int)model->output->type : 0;
    // RoPE frequency factors — per-dimension multipliers (Llama 3: rope_freqs, Phi-3.5: rope_long/rope_short)
    if (n_layer > 0) {
        if (model->layers[0].rope_freqs) {
            cfg.rope_freq_factors = td(model->layers[0].rope_freqs);
        } else if ((int)cfg.max_seq_len > (int)hp.n_ctx_orig_yarn) {
            cfg.rope_freq_factors = td(model->layers[0].rope_long);
        } else {
            cfg.rope_freq_factors = td(model->layers[0].rope_short);
        }
    } else {
        cfg.rope_freq_factors = nullptr;
    }

    // T5 encoder final norm
    cfg.output_norm_enc = td(model->output_norm_enc);

    // BERT position + type embeddings
    cfg.pos_embd        = td(model->pos_embd);
    cfg.pos_embd_stride = ts(model->pos_embd);
    cfg.pos_embd_type   = tt(model->pos_embd);
    cfg.type_embd        = td(model->type_embd);
    cfg.type_embd_stride = ts(model->type_embd);
    cfg.type_embd_type   = tt(model->type_embd);

    // Gemma embed scale
    switch (model->arch) {
        case LLM_ARCH_GEMMA: case LLM_ARCH_GEMMA2: case LLM_ARCH_GEMMA3:
        case LLM_ARCH_GEMMA3N: case LLM_ARCH_GEMMA4: case LLM_ARCH_GEMMA_EMBEDDING:
            cfg.has_embed_scale = 1; break;
        default: break;
    }

    // WavTokenizer output bias
    cfg.wav_output_b = td(model->output_b);

    // Print config
    int n_attn = 0, n_dn = 0, n_ssm = 0, n_rwkv = 0;
    for (int i = 0; i < n_layer; i++) {
        switch (cfg.layer_types[i]) {
            case 0: n_attn++; break;
            case 1: n_dn++; break;
            case 2: n_ssm++; break;
            case 3: n_rwkv++; break;
        }
    }
    fprintf(stderr, "=== Model: %s ===\n", model_path);
    fprintf(stderr, "  layers=%d (attn=%d, dn=%d, ssm=%d, rwkv=%d), H=%d, FF=%d, V=%d\n",
            n_layer, n_attn, n_dn, n_ssm, n_rwkv, n_embd, cfg.intermediate_size, n_vocab);
    fprintf(stderr, "  attn: q=%d kv=%d dim=%d rope=%.0f gated=%d\n",
            cfg.fa_n_q_heads, cfg.fa_n_kv_heads, cfg.fa_head_dim, cfg.fa_rope_theta, cfg.fa_has_gated_attn);
    if (n_dn > 0)
        fprintf(stderr, "  dn: heads=%d k=%d key=%d val=%d conv=%d\n",
                cfg.dn_n_heads, cfg.dn_n_k_heads, cfg.dn_key_dim, cfg.dn_value_dim, cfg.dn_conv_kernel);
    if (n_ssm > 0)
        fprintf(stderr, "  ssm: d_conv=%d d_inner=%d d_state=%d dt_rank=%d n_group=%d\n",
                cfg.ssm_d_conv, cfg.ssm_d_inner, cfg.ssm_d_state, cfg.ssm_dt_rank, cfg.ssm_n_group);
    if (n_rwkv > 0)
        fprintf(stderr, "  rwkv: wkv_head=%d lora=%d\n",
                cfg.wkv_head_size, cfg.rwkv_lora_size);

    // Verify weights — ptrs[0] (attn_norm) must exist on all layers.
    // ptrs[1] (wq) only exists on attention/deltanet layers, not SSM/RWKV.
    for (int il = 0; il < n_layer; il++) {
        assert(cfg.layers[il].ptrs[0] != nullptr);
        if (cfg.layer_types[il] == 0 || cfg.layer_types[il] == 1) {
            assert(cfg.layers[il].ptrs[1] != nullptr);
        }
    }
    assert(cfg.embed_weight && cfg.lm_head_weight && cfg.final_norm_weight);
    fprintf(stderr, "  embed=%p (type=%d) lm_head=%p (type=%d) final_norm=%p\n",
            cfg.embed_weight, cfg.embed_type, cfg.lm_head_weight, cfg.lm_head_type, cfg.final_norm_weight);
    fprintf(stderr, "  L0: attn_norm=%p wq=%p(%d) ffn_gate=%p(%d) ffn_down=%p(%d)\n",
            cfg.layers[0].ptrs[0], cfg.layers[0].ptrs[1], cfg.layers[0].types[1],
            cfg.layers[0].ptrs[8], cfg.layers[0].types[8], cfg.layers[0].ptrs[10], cfg.layers[0].types[10]);
    fprintf(stderr, "  lm_head tied to embed? %s\n",
            cfg.lm_head_weight == cfg.embed_weight ? "YES" : "no");

    // ---- Allocate KV cache (f16, position-major) ----
    {
        size_t kv_bytes_per_layer = (size_t)cfg.fa_n_kv_heads * cfg.max_seq_len * cfg.fa_head_dim * 2; // f16
        int attn_count = 0;
        for (int il = 0; il < n_layer; il++) {
            if (cfg.layer_types[il] == 0) {
                hipMalloc(&cfg.k_cache_ptrs[attn_count], kv_bytes_per_layer);
                hipMalloc(&cfg.v_cache_ptrs[attn_count], kv_bytes_per_layer);
                hipMemset(cfg.k_cache_ptrs[attn_count], 0, kv_bytes_per_layer);
                hipMemset(cfg.v_cache_ptrs[attn_count], 0, kv_bytes_per_layer);
                attn_count++;
            }
        }
        fprintf(stderr, "  KV cache: %d layers, %.1f MiB each\n",
                attn_count, kv_bytes_per_layer / (1024.0 * 1024.0));
    }

    // ---- Find megakernel backend ----
    fn_init_t mk_init = nullptr;
    fn_decode_t mk_decode = nullptr;
    fn_is_ready_t mk_ready = nullptr;

    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * avail = (fn_is_available_t)ggml_backend_reg_get_proc_address(reg, "gfx1100_is_available");
        if (avail && avail()) {
            mk_init   = (fn_init_t)ggml_backend_reg_get_proc_address(reg, "gfx1100_init");
            mk_decode = (fn_decode_t)ggml_backend_reg_get_proc_address(reg, "gfx1100_eval_decode");
            mk_ready  = (fn_is_ready_t)ggml_backend_reg_get_proc_address(reg, "gfx1100_is_ready");
            break;
        }
    }
    assert(mk_init && mk_decode && "gfx1100 backend not found");

    // ---- Init megakernel ----
    fprintf(stderr, "Compiling megakernel (first run may take ~30s)...\n");
    int rc = mk_init(&cfg);
    assert(rc == 0 && "megakernel init failed");
    assert(mk_ready());
    fprintf(stderr, "Megakernel ready.\n");

    // ---- Mode: compare against baseline logits if file provided ----
    // argv[2]: if it's a readable file → baseline comparison mode. If numeric → n_gen. If empty → skip.
    const char * baseline_path = nullptr;
    if (argc >= 3 && argv[2][0] != '\0') {
        // Check if it looks like a file path (not a plain number)
        FILE * probe = fopen(argv[2], "rb");
        if (probe) { fclose(probe); baseline_path = argv[2]; }
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (baseline_path) {
        // ---- COMPARISON MODE: load baseline logits, run megakernel, compare ----
        FILE * fin = fopen(baseline_path, "rb");
        if (!fin) { fprintf(stderr, "Cannot open baseline file: %s\n", baseline_path); return 1; }

        int bl_n_tokens, bl_n_vocab;
        fread(&bl_n_tokens, sizeof(int), 1, fin);
        fread(&bl_n_vocab, sizeof(int), 1, fin);
        assert(bl_n_vocab == n_vocab);

        fprintf(stderr, "\n=== Comparing %d tokens against baseline ===\n", bl_n_tokens);

        std::vector<float> mk_logits(n_vocab);
        std::vector<float> bl_logits(n_vocab);
        int mismatches = 0;
        float worst_abs = 0, worst_rel = 0;

        auto t0 = std::chrono::high_resolution_clock::now();

        for (int pos = 0; pos < bl_n_tokens; pos++) {
            // Read baseline logits + tokens
            fread(bl_logits.data(), sizeof(float), n_vocab, fin);
            int bl_token_in, bl_argmax;
            fread(&bl_token_in, sizeof(int), 1, fin);
            fread(&bl_argmax, sizeof(int), 1, fin);

            // Run megakernel with same input token
            rc = mk_decode(bl_token_in, pos, mk_logits.data());
            assert(rc == 0);

            // Compare
            int mk_argmax = 0; float mk_max = mk_logits[0];
            float max_abs = 0, max_rel = 0;
            int nans = 0, n_identical = 0;
            double diff_sum = 0, diff_sq = 0;
            for (int i = 0; i < n_vocab; i++) {
                if (mk_logits[i] != mk_logits[i]) nans++;
                if (mk_logits[i] > mk_max) { mk_max = mk_logits[i]; mk_argmax = i; }
                float d = fabsf(mk_logits[i] - bl_logits[i]);
                if (d == 0.0f) n_identical++;
                diff_sum += d; diff_sq += (double)d * d;
                if (d > max_abs) max_abs = d;
                float a = fabsf(bl_logits[i]);
                if (a > 1.0f) { float r = d / a; if (r > max_rel) max_rel = r; }
            }
            if (max_abs > worst_abs) worst_abs = max_abs;
            if (max_rel > worst_rel) worst_rel = max_rel;

            bool match = (mk_argmax == bl_argmax);
            if (!match) mismatches++;

            // Accumulate per-position variance bucket stats
            // Buckets: exact, <1e-6, <1e-4, <1e-2, <0.1, <1.0, >=1.0
            int bucket = (max_abs == 0) ? 0 : (max_abs < 1e-6f) ? 1 : (max_abs < 1e-4f) ? 2 :
                         (max_abs < 1e-2f) ? 3 : (max_abs < 0.1f) ? 4 : (max_abs < 1.0f) ? 5 : 6;
            static int buckets[7] = {};
            buckets[bucket]++;
            // Track divergence curve (sample every 100 positions)
            static float divergence_samples[100] = {};
            int sample_idx = pos * 100 / bl_n_tokens;
            if (sample_idx < 100) divergence_samples[sample_idx] = max_abs;

            float pct_identical = 100.0f * n_identical / n_vocab;
            float mean_diff = (float)(diff_sum / n_vocab);
            float rms_diff = sqrtf((float)(diff_sq / n_vocab));

            char mk_buf[64] = {}, bl_buf[64] = {};
            llama_token_to_piece(vocab, mk_argmax, mk_buf, sizeof(mk_buf) - 1, 0, true);
            llama_token_to_piece(vocab, bl_argmax, bl_buf, sizeof(bl_buf) - 1, 0, true);

            // Compact output: only show details for mismatches and every 100th position
            if (!match || pos % 100 == 0 || pos < 5) {
                fprintf(stderr, "  [%d] in=%d  mk=%d \"%s\"  bl=%d \"%s\"  max_abs=%.4f identical=%.1f%% mean_diff=%.2e rms_diff=%.2e %s%s\n",
                        pos, bl_token_in, mk_argmax, mk_buf, bl_argmax, bl_buf,
                        max_abs, pct_identical, mean_diff, rms_diff,
                        match ? "OK" : "MISMATCH", nans ? " NaN!" : "");
            }

            // Final position: print bucket summary
            if (pos == bl_n_tokens - 1) {
                fprintf(stderr, "\n=== Variance Buckets (per-position max_abs) ===\n");
                const char * labels[] = {"exact (0)", "<1e-6", "<1e-4", "<1e-2", "<0.1", "<1.0", ">=1.0"};
                for (int b = 0; b < 7; b++) {
                    if (buckets[b] > 0)
                        fprintf(stderr, "  %-10s: %d positions (%.1f%%)\n",
                                labels[b], buckets[b], 100.0f * buckets[b] / bl_n_tokens);
                }
                // Divergence curve (10 samples across the sequence)
                fprintf(stderr, "\n=== Divergence Curve (max_abs at sampled positions) ===\n  ");
                int n_samples = bl_n_tokens < 100 ? bl_n_tokens : 10;
                int step = 100 / n_samples;
                for (int s = 0; s < n_samples; s++) {
                    int si = s * step;
                    if (si < 100) fprintf(stderr, "pos~%d:%.3f ", si * bl_n_tokens / 100, divergence_samples[si]);
                }
                fprintf(stderr, "\n");
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        fclose(fin);

        fprintf(stderr, "\n=== Results ===\n");
        fprintf(stderr, "  Tokens: %d, Mismatches: %d/%d\n", bl_n_tokens, mismatches, bl_n_tokens);
        fprintf(stderr, "  Worst abs diff: %.4f\n", worst_abs);
        fprintf(stderr, "  Worst rel diff: %.4f\n", worst_rel);
        fprintf(stderr, "  Speed: %.1f tok/s\n", bl_n_tokens * 1000.0 / ms);

        llama_model_free(model);
        llama_backend_free();

        if (mismatches > bl_n_tokens / 2) {
            fprintf(stderr, "FAIL: >50%% argmax mismatches\n");
            return 1;
        }
        return 0;
    }

    // ---- STANDALONE MODE: generate and print text ----
    std::vector<float> mk_logits(n_vocab);
    int cur_token = llama_vocab_bos(vocab);
    if (cur_token < 0 || cur_token >= n_vocab) cur_token = 1;
    // n_gen: check argv[3] first, then argv[2] if no baseline path was given (standalone shortcut)
    int n_gen = 2000;
    if (argc >= 4 && argv[3][0] != '\0') {
        n_gen = atoi(argv[3]);
    } else if (argc >= 3 && !baseline_path && argv[2][0] != '\0') {
        // Standalone: ./test-megakernel-e2e model 5000
        int v = atoi(argv[2]);
        if (v > 0) n_gen = v;
    }

    fprintf(stderr, "\n=== Megakernel decode (standalone, %d tokens) ===\n", n_gen);
    auto t0 = std::chrono::high_resolution_clock::now();

    // Warmup: first token includes JIT overhead, measure separately
    rc = mk_decode(cur_token, 0, mk_logits.data());
    assert(rc == 0);
    { int argmax = 0; float maxv = mk_logits[0];
      for (int i = 0; i < n_vocab; i++) if (mk_logits[i] > maxv) { maxv = mk_logits[i]; argmax = i; }
      char buf[64] = {};
      llama_token_to_piece(vocab, argmax, buf, sizeof(buf) - 1, 0, true);
      fprintf(stderr, "  [warmup] %d → %d \"%s\" (%.2f)\n", cur_token, argmax, buf, maxv);
      cur_token = argmax;
    }

    auto t_gen_start = std::chrono::high_resolution_clock::now();
    int n_nans = 0;
    for (int pos = 1; pos < n_gen; pos++) {
        rc = mk_decode(cur_token, pos, mk_logits.data());
        assert(rc == 0);

        int argmax = 0; float maxv = mk_logits[0];
        for (int i = 0; i < n_vocab; i++) {
            if (mk_logits[i] != mk_logits[i]) n_nans++;
            if (mk_logits[i] > maxv) { maxv = mk_logits[i]; argmax = i; }
        }
        // Print first 5 and last 5 tokens
        if (pos < 6 || pos >= n_gen - 5) {
            char buf[64] = {};
            llama_token_to_piece(vocab, argmax, buf, sizeof(buf) - 1, 0, true);
            fprintf(stderr, "  [%d] %d → %d \"%s\" (%.2f)\n", pos, cur_token, argmax, buf, maxv);
        } else if (pos == 6) {
            fprintf(stderr, "  ... (%d tokens) ...\n", n_gen - 11);
        }
        cur_token = argmax;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_total = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_gen   = std::chrono::duration<double, std::milli>(t1 - t_gen_start).count();
    int    gen_count = n_gen - 1; // exclude warmup
    fprintf(stderr, "\n=== Benchmark Results ===\n");
    fprintf(stderr, "  Total:   %d tokens in %.1f ms (%.1f tok/s incl. warmup)\n", n_gen, ms_total, n_gen * 1000.0 / ms_total);
    fprintf(stderr, "  Decode:  %d tokens in %.1f ms (%.1f tok/s sustained)\n", gen_count, ms_gen, gen_count * 1000.0 / ms_gen);
    if (n_nans > 0) fprintf(stderr, "  WARNING: %d NaN logits detected!\n", n_nans);

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
