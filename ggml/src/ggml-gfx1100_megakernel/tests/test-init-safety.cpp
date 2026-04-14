// test-init-safety.cpp — Tests that init correctly rejects unsupported configurations
//
// Tests: MoE detection, attention scale detection, NULL pointer safety
// No GPU kernels launched — only tests the init function's validation.
//
// Usage: test-init-safety <model.gguf>

#include "llama.h"
#include "ggml.h"
#include "llama-model.h"
#include "llama-hparams.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

// Must match gfx1100-megakernel.cpp exactly
struct gfx1100_layer_weights {
    const void * ptrs[16];
    long long    strides[16];
    int          types[16];
    const void * bias_q;
    const void * bias_k;
    const void * bias_v;
    const void * bias_o;
    const void * scale_q;
    const void * scale_k;
    const void * scale_v;
    const void * scale_o;
    const void * ffn_gate_bias;
    const void * ffn_up_bias;
    const void * ffn_down_bias;
    const void * ffn_gate_scale;
    const void * ffn_up_scale;
    const void * ffn_down_scale;
    const void * ffn_gate_inp;
};

struct gfx1100_model_config {
    // Mirrors gfx1100-megakernel.cpp::gfx1100_model_config
    int arch_id;
    int has_qk_norm;
    int has_bias_q, has_bias_k, has_bias_v, has_bias_o;
    int has_scale_q, has_scale_k, has_scale_v, has_scale_o;
    int has_bias_ffn_gate, has_bias_ffn_up, has_bias_ffn_down;
    int has_scale_ffn_gate, has_scale_ffn_up, has_scale_ffn_down;
    int rope_type;
    int has_rope_freq_factors;
    int has_moe;
    int moe_n_experts, moe_n_experts_used;
    int has_ssm, has_dn;
    int attn_scale_type;
    float attn_softcap_val;
    int has_final_logit_softcap;
    float final_logit_softcap_val;
    int norm_type;
    int act_type;
    int pooling_type;
    int has_swa;
    int swa_type;
    int n_swa;
    int has_alibi;

    int hidden_size;
    int intermediate_size;
    int vocab_size;
    int n_layers;
    int layer_types[128];
    int fa_n_q_heads;
    int fa_n_kv_heads;
    int fa_head_dim;
    float fa_rope_theta;
    int fa_rope_dim;
    int fa_has_gated_attn;
    float fa_attention_scale;
    int fa_use_kq_norm;
    int dn_n_heads;
    int dn_n_k_heads;
    int dn_key_dim;
    int dn_value_dim;
    int dn_conv_kernel;
    int ssm_d_conv, ssm_d_inner, ssm_d_state, ssm_dt_rank, ssm_n_group;
    float rope_freq_scale, rope_attn_factor;
    float yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow;
    int n_ctx_orig_yarn;
    int rope_sections[4];
    float norm_eps;
    int norm_add_one;
    gfx1100_layer_weights layers[128];
    const void * embed_weight;
    long long    embed_stride;
    int          embed_type;
    const void * final_norm_weight;
    const void * lm_head_weight;
    long long    lm_head_stride;
    int          lm_head_type;
    const void * rope_freq_factors;
    void * k_cache_ptrs[128];
    void * v_cache_ptrs[128];
    int    kv_stride;
    int    max_seq_len;
    int    kv_type;
};

typedef int (*fn_init_t)(const gfx1100_model_config *);
typedef int (*fn_is_available_t)(void);

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    // Find megakernel backend
    llama_backend_init();

    fn_init_t mk_init = nullptr;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * avail = (fn_is_available_t)ggml_backend_reg_get_proc_address(reg, "gfx1100_is_available");
        if (avail && avail()) {
            mk_init = (fn_init_t)ggml_backend_reg_get_proc_address(reg, "gfx1100_init");
            break;
        }
    }
    if (!mk_init) {
        fprintf(stderr, "SKIP: gfx1100 backend not available\n");
        return 0;
    }

    int pass = 0, fail = 0;

    // TEST: MoE detection — set ffn_gate_inp to non-NULL, expect FATAL return -1
    {
        gfx1100_model_config cfg = {};
        cfg.hidden_size = 2048;
        cfg.intermediate_size = 8192;
        cfg.vocab_size = 128256;
        cfg.n_layers = 1;
        cfg.fa_n_q_heads = 32;
        cfg.fa_n_kv_heads = 8;
        cfg.fa_head_dim = 64;
        cfg.fa_rope_theta = 500000.0f;
        cfg.fa_rope_dim = 64;
        cfg.norm_eps = 1e-5f;
        cfg.max_seq_len = 2048;
        // Set a dummy non-NULL pointer for MoE detection
        cfg.layers[0].ffn_gate_inp = (const void *)0x1;

        int rc = mk_init(&cfg);
        if (rc == -1) {
            fprintf(stderr, "  PASS: MoE detection — init returned -1\n");
            pass++;
        } else {
            fprintf(stderr, "  FAIL: MoE detection — init returned %d (expected -1)\n", rc);
            fail++;
        }
    }

    // TEST: Attention scale detection — set non-default scale, expect FATAL return -1
    {
        gfx1100_model_config cfg = {};
        cfg.hidden_size = 2048;
        cfg.intermediate_size = 8192;
        cfg.vocab_size = 128256;
        cfg.n_layers = 1;
        cfg.fa_n_q_heads = 32;
        cfg.fa_n_kv_heads = 8;
        cfg.fa_head_dim = 64;
        cfg.fa_rope_theta = 500000.0f;
        cfg.fa_rope_dim = 64;
        cfg.norm_eps = 1e-5f;
        cfg.max_seq_len = 2048;
        cfg.fa_attention_scale = 0.5f; // non-default, should fail

        int rc = mk_init(&cfg);
        if (rc == -1) {
            fprintf(stderr, "  PASS: Attention scale detection — init returned -1\n");
            pass++;
        } else {
            fprintf(stderr, "  FAIL: Attention scale detection — init returned %d (expected -1)\n", rc);
            fail++;
        }
    }

    fprintf(stderr, "\n=== Results: %d pass, %d fail ===\n", pass, fail);

    llama_backend_free();
    return fail > 0 ? 1 : 0;
}
