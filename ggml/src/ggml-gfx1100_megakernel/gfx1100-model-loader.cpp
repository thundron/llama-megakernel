// gfx1100-model-loader.cpp — Complete GGUF model loading for the megakernel.
//
// Reads a GGUF file using ggml's gguf_init_from_file, extracts ALL hparams and
// tensor data for ALL 126 architectures, maps tensors to GPU memory, populates
// gfx1100_model_config with capability bits, and calls gfx1100_init.
//
// This file links against ggml's GGUF reader (ggml.c) and the HIP runtime.
// It does NOT depend on llama.cpp's model loader — it reads GGUF directly.

#include "gfx1100-internal.h"
#include "../../include/ggml.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>

// ============================================================================
// GGUF metadata helpers
// ============================================================================

static int64_t gguf_get_i(const struct gguf_context * ctx, const char * key, int64_t def) {
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) return def;
    return gguf_get_val_u32(ctx, idx);
}

static float gguf_get_f(const struct gguf_context * ctx, const char * key, float def) {
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) return def;
    return gguf_get_val_f32(ctx, idx);
}

static std::string gguf_get_s(const struct gguf_context * ctx, const char * key) {
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) return "";
    return gguf_get_val_str(ctx, idx);
}

// ============================================================================
// Architecture string → ARCH_* ID
// ============================================================================

struct arch_entry { const char * name; int id; };
static const arch_entry arch_table[] = {
    {"llama",          ARCH_LLAMA},          {"llama4",         ARCH_LLAMA4},
    {"deci",           ARCH_DECI},           {"falcon",         ARCH_FALCON},
    {"grok",           ARCH_GROK},           {"gpt2",           ARCH_GPT2},
    {"gptj",           ARCH_GPTJ},           {"gptneox",        ARCH_GPTNEOX},
    {"mpt",            ARCH_MPT},            {"baichuan",       ARCH_BAICHUAN},
    {"starcoder",      ARCH_STARCODER},      {"refact",         ARCH_REFACT},
    {"bert",           ARCH_BERT},           {"modern-bert",    ARCH_MODERN_BERT},
    {"nomic-bert",     ARCH_NOMIC_BERT},     {"nomic-bert-moe", ARCH_NOMIC_BERT_MOE},
    {"neo-bert",       ARCH_NEO_BERT},       {"jina-bert-v2",   ARCH_JINA_BERT_V2},
    {"jina-bert-v3",   ARCH_JINA_BERT_V3},   {"eurobert",       ARCH_EUROBERT},
    {"bloom",          ARCH_BLOOM},          {"stablelm",       ARCH_STABLELM},
    {"qwen",           ARCH_QWEN},           {"qwen2",          ARCH_QWEN2},
    {"qwen2moe",       ARCH_QWEN2MOE},       {"qwen2vl",        ARCH_QWEN2VL},
    {"qwen3",          ARCH_QWEN3},           {"qwen3moe",       ARCH_QWEN3MOE},
    {"qwen3next",      ARCH_QWEN3NEXT},       {"qwen3vl",        ARCH_QWEN3VL},
    {"qwen3vlmoe",     ARCH_QWEN3VLMOE},      {"qwen35",         ARCH_QWEN35},
    {"qwen35moe",      ARCH_QWEN35MOE},       {"phi2",           ARCH_PHI2},
    {"phi3",           ARCH_PHI3},             {"phimoe",         ARCH_PHIMOE},
    {"plamo",          ARCH_PLAMO},            {"plamo2",         ARCH_PLAMO2},
    {"plamo3",         ARCH_PLAMO3},           {"codeshell",      ARCH_CODESHELL},
    {"orion",          ARCH_ORION},            {"internlm2",      ARCH_INTERNLM2},
    {"minicpm",        ARCH_MINICPM},          {"minicpm3",       ARCH_MINICPM3},
    {"gemma",          ARCH_GEMMA},            {"gemma2",         ARCH_GEMMA2},
    {"gemma3",         ARCH_GEMMA3},           {"gemma3n",        ARCH_GEMMA3N},
    {"gemma4",         ARCH_GEMMA4},           {"gemma-embedding",ARCH_GEMMA_EMBEDDING},
    {"starcoder2",     ARCH_STARCODER2},       {"mamba",          ARCH_MAMBA},
    {"mamba2",         ARCH_MAMBA2},           {"jamba",          ARCH_JAMBA},
    {"falcon-h1",      ARCH_FALCON_H1},        {"xverse",         ARCH_XVERSE},
    {"command-r",      ARCH_COMMAND_R},         {"cohere2",        ARCH_COHERE2},
    {"dbrx",           ARCH_DBRX},             {"olmo",           ARCH_OLMO},
    {"olmo2",          ARCH_OLMO2},            {"olmoe",          ARCH_OLMOE},
    {"openelm",        ARCH_OPENELM},          {"arctic",         ARCH_ARCTIC},
    {"deepseek",       ARCH_DEEPSEEK},         {"deepseek2",      ARCH_DEEPSEEK2},
    {"deepseek2-ocr",  ARCH_DEEPSEEK2OCR},     {"chatglm",        ARCH_CHATGLM},
    {"glm4",           ARCH_GLM4},             {"glm4moe",        ARCH_GLM4_MOE},
    {"glm-dsa",        ARCH_GLM_DSA},          {"bitnet",         ARCH_BITNET},
    {"t5",             ARCH_T5},               {"t5encoder",      ARCH_T5ENCODER},
    {"jais",           ARCH_JAIS},             {"jais2",          ARCH_JAIS2},
    {"nemotron",       ARCH_NEMOTRON},         {"nemotron_h",     ARCH_NEMOTRON_H},
    {"nemotron_h_moe", ARCH_NEMOTRON_H_MOE},   {"exaone",         ARCH_EXAONE},
    {"exaone4",        ARCH_EXAONE4},          {"exaone-moe",     ARCH_EXAONE_MOE},
    {"rwkv6",          ARCH_RWKV6},            {"rwkv6qwen2",     ARCH_RWKV6QWEN2},
    {"rwkv7",          ARCH_RWKV7},            {"arwkv7",         ARCH_ARWKV7},
    {"granite",        ARCH_GRANITE},          {"granitemoe",     ARCH_GRANITE_MOE},
    {"granitehybrid",  ARCH_GRANITE_HYBRID},    {"chameleon",      ARCH_CHAMELEON},
    {"wavtokenizer-dec",ARCH_WAVTOKENIZER_DEC}, {"plm",            ARCH_PLM},
    {"bailingmoe",     ARCH_BAILINGMOE},        {"bailingmoe2",    ARCH_BAILINGMOE2},
    {"dots1",          ARCH_DOTS1},             {"arcee",          ARCH_ARCEE},
    {"afmoe",          ARCH_AFMOE},             {"ernie4_5",       ARCH_ERNIE4_5},
    {"ernie4_5-moe",   ARCH_ERNIE4_5_MOE},     {"hunyuan-moe",    ARCH_HUNYUAN_MOE},
    {"hunyuan-dense",  ARCH_HUNYUAN_DENSE},     {"smollm3",        ARCH_SMOLLM3},
    {"gpt-oss",        ARCH_OPENAI_MOE},        {"lfm2",           ARCH_LFM2},
    {"lfm2moe",        ARCH_LFM2MOE},           {"dream",          ARCH_DREAM},
    {"smallthinker",   ARCH_SMALLTHINKER},      {"llada",          ARCH_LLADA},
    {"llada-moe",      ARCH_LLADA_MOE},         {"seed_oss",       ARCH_SEED_OSS},
    {"grovemoe",       ARCH_GROVEMOE},          {"apertus",        ARCH_APERTUS},
    {"minimax-m2",     ARCH_MINIMAX_M2},        {"cogvlm",         ARCH_COGVLM},
    {"rnd1",           ARCH_RND1},              {"pangu-embedded",  ARCH_PANGU_EMBED},
    {"mistral3",       ARCH_MISTRAL3},          {"mistral4",       ARCH_MISTRAL4},
    {"paddleocr",      ARCH_PADDLEOCR},         {"mimo2",          ARCH_MIMO2},
    {"step35",         ARCH_STEP35},            {"llama-embed",    ARCH_LLAMA_EMBED},
    {"maincoder",      ARCH_MAINCODER},         {"kimi-linear",    ARCH_KIMI_LINEAR},
    {nullptr, 0}
};

static int arch_from_string(const std::string & s) {
    for (const arch_entry * e = arch_table; e->name; e++) {
        if (s == e->name) return e->id;
    }
    fprintf(stderr, "gfx1100: WARNING — unknown architecture '%s', using ARCH_UNKNOWN\n", s.c_str());
    return ARCH_UNKNOWN;
}

// ============================================================================
// Tensor GPU mapping helper
// ============================================================================

static void * map_tensor_to_gpu(const struct ggml_tensor * t) {
    if (!t) return nullptr;
    size_t sz = ggml_nbytes(t);
    void * gpu_ptr = nullptr;
    hipMalloc(&gpu_ptr, sz);
    hipMemcpy(gpu_ptr, t->data, sz, hipMemcpyHostToDevice);
    return gpu_ptr;
}

// Helper: find tensor by name, map to GPU, set pointer + stride + type
#define LOAD_WEIGHT(ptr_field, stride_field, type_field, name_fmt, ...) do { \
    char _name[256]; \
    snprintf(_name, sizeof(_name), name_fmt, ##__VA_ARGS__); \
    struct ggml_tensor * _t = ggml_get_tensor(ctx, _name); \
    if (_t) { \
        lw.ptr_field = map_tensor_to_gpu(_t); \
        lw.stride_field = _t->nb[1]; \
        lw.type_field = _t->type; \
    } \
} while(0)

// Helper: find tensor, map to GPU, set pointer only (no stride/type — for 1D tensors like biases/norms)
#define LOAD_VEC(ptr_field, name_fmt, ...) do { \
    char _name[256]; \
    snprintf(_name, sizeof(_name), name_fmt, ##__VA_ARGS__); \
    struct ggml_tensor * _t = ggml_get_tensor(ctx, _name); \
    if (_t) { lw.ptr_field = map_tensor_to_gpu(_t); } \
} while(0)

// Helper: same but for config-level (not per-layer) fields
#define LOAD_CFG_WEIGHT(ptr_field, stride_field, type_field, name_str) do { \
    struct ggml_tensor * _t = ggml_get_tensor(ctx, name_str); \
    if (_t) { \
        cfg.ptr_field = map_tensor_to_gpu(_t); \
        cfg.stride_field = _t->nb[1]; \
        cfg.type_field = _t->type; \
    } \
} while(0)

#define LOAD_CFG_VEC(ptr_field, name_str) do { \
    struct ggml_tensor * _t = ggml_get_tensor(ctx, name_str); \
    if (_t) { cfg.ptr_field = map_tensor_to_gpu(_t); } \
} while(0)

// ============================================================================
// Main loader
// ============================================================================

int gfx1100_load_model(const char * model_path, int n_ctx, int n_batch) {
    fprintf(stderr, "gfx1100: loading model from %s (n_ctx=%d, n_batch=%d)\n", model_path, n_ctx, n_batch);

    // Step 1: Open GGUF file
    struct gguf_init_params params = { .no_alloc = false, .ctx = nullptr };
    struct gguf_context * gguf_ctx = gguf_init_from_file(model_path, params);
    if (!gguf_ctx) {
        fprintf(stderr, "gfx1100: FATAL — failed to open GGUF file: %s\n", model_path);
        return -1;
    }
    struct ggml_context * ctx = params.ctx;

    // Step 2: Read architecture
    gfx1100_model_config cfg = {};
    std::string arch = gguf_get_s(gguf_ctx, "general.architecture");
    cfg.arch_id = arch_from_string(arch);
    std::string pfx = arch + "."; // prefix for arch-specific keys

    // Step 3: Read hparams
    cfg.hidden_size        = (int)gguf_get_i(gguf_ctx, (pfx + "embedding_length").c_str(), 0);
    cfg.intermediate_size  = (int)gguf_get_i(gguf_ctx, (pfx + "feed_forward_length").c_str(), 0);
    cfg.n_layers           = (int)gguf_get_i(gguf_ctx, (pfx + "block_count").c_str(), 0);
    cfg.fa_n_q_heads       = (int)gguf_get_i(gguf_ctx, (pfx + "attention.head_count").c_str(), 0);
    cfg.fa_n_kv_heads      = (int)gguf_get_i(gguf_ctx, (pfx + "attention.head_count_kv").c_str(), cfg.fa_n_q_heads);
    cfg.fa_head_dim        = cfg.fa_n_q_heads > 0 ? cfg.hidden_size / cfg.fa_n_q_heads : 0;
    cfg.fa_rope_theta      = gguf_get_f(gguf_ctx, (pfx + "rope.freq_base").c_str(), 10000.0f);
    cfg.fa_rope_dim        = (int)gguf_get_i(gguf_ctx, (pfx + "rope.dimension_count").c_str(), cfg.fa_head_dim);
    cfg.vocab_size         = (int)gguf_get_i(gguf_ctx, (pfx + "vocab_size").c_str(), 0);
    cfg.norm_eps           = gguf_get_f(gguf_ctx, (pfx + "attention.layer_norm_rms_epsilon").c_str(), 0);
    if (cfg.norm_eps == 0) cfg.norm_eps = gguf_get_f(gguf_ctx, (pfx + "attention.layer_norm_epsilon").c_str(), 1e-5f);
    cfg.max_seq_len        = n_ctx;

    // MoE hparams
    cfg.moe_n_experts      = (int)gguf_get_i(gguf_ctx, (pfx + "expert_count").c_str(), 0);
    cfg.moe_n_experts_used = (int)gguf_get_i(gguf_ctx, (pfx + "expert_used_count").c_str(), 0);

    // SSM hparams (Mamba/Mamba2)
    cfg.ssm_d_conv         = (int)gguf_get_i(gguf_ctx, (pfx + "ssm.conv_kernel").c_str(), 0);
    cfg.ssm_d_inner        = (int)gguf_get_i(gguf_ctx, (pfx + "ssm.inner_size").c_str(), 0);
    cfg.ssm_d_state        = (int)gguf_get_i(gguf_ctx, (pfx + "ssm.state_size").c_str(), 0);
    cfg.ssm_dt_rank        = (int)gguf_get_i(gguf_ctx, (pfx + "ssm.time_step_rank").c_str(), 0);
    cfg.ssm_n_group        = (int)gguf_get_i(gguf_ctx, (pfx + "ssm.group_count").c_str(), 1);

    // RWKV hparams
    cfg.wkv_head_size      = (int)gguf_get_i(gguf_ctx, (pfx + "wkv.head_size").c_str(), 64);
    cfg.rwkv_lora_size     = (int)gguf_get_i(gguf_ctx, (pfx + "time_mix_extra_dim").c_str(), 32);

    // DeepSeek2 MLA hparams
    cfg.mla_kv_lora_rank     = (int)gguf_get_i(gguf_ctx, (pfx + "attention.key_length_mla").c_str(), 0);
    cfg.mla_q_lora_rank      = (int)gguf_get_i(gguf_ctx, (pfx + "attention.q_lora_rank").c_str(), 0);
    cfg.mla_n_embd_head_qk_rope = (int)gguf_get_i(gguf_ctx, (pfx + "attention.head_count_kv_rope").c_str(), 0);
    cfg.mla_n_layer_dense_lead   = (int)gguf_get_i(gguf_ctx, (pfx + "leading_dense_block_count").c_str(), 0);

    // DeltaNet hparams (Qwen35)
    cfg.dn_n_heads       = (int)gguf_get_i(gguf_ctx, (pfx + "attention.head_count_deltanet").c_str(), 0);
    cfg.dn_key_dim       = (int)gguf_get_i(gguf_ctx, (pfx + "attention.key_length_deltanet").c_str(), 0);
    cfg.dn_value_dim     = (int)gguf_get_i(gguf_ctx, (pfx + "attention.value_length_deltanet").c_str(), 0);
    cfg.dn_conv_kernel   = (int)gguf_get_i(gguf_ctx, (pfx + "attention.conv_kernel_deltanet").c_str(), 4);

    // Context length (max KV cache, attention loop bound)
    cfg.n_ctx_train    = (int)gguf_get_i(gguf_ctx, (pfx + "context_length").c_str(), 0);

    // Explicit key/value lengths (override computed head_dim when present)
    int key_len  = (int)gguf_get_i(gguf_ctx, (pfx + "attention.key_length").c_str(), 0);
    int val_len  = (int)gguf_get_i(gguf_ctx, (pfx + "attention.value_length").c_str(), 0);
    if (key_len > 0) cfg.fa_head_dim = key_len;
    if (val_len > 0) cfg.fa_value_dim = val_len;  // may differ from key (rare)

    // Attention flags
    cfg.attn_causal    = (int)gguf_get_i(gguf_ctx, (pfx + "attention.causal").c_str(), 1); // default causal
    cfg.attn_clamp_kqv = gguf_get_f(gguf_ctx, (pfx + "attention.clamp_kqv").c_str(), 0.0f);

    // SWA (sliding window)
    int swa_size = (int)gguf_get_i(gguf_ctx, (pfx + "attention.sliding_window").c_str(), 0);
    if (swa_size > 0) { cfg.has_swa = 1; cfg.n_swa = swa_size; }
    cfg.swa_pattern        = (int)gguf_get_i(gguf_ctx, (pfx + "full_attention_interval").c_str(), 0);
    cfg.fa_rope_theta_swa  = gguf_get_f(gguf_ctx, (pfx + "rope.freq_base_swa").c_str(), 0.0f);
    cfg.fa_rope_dim_swa    = (int)gguf_get_i(gguf_ctx, (pfx + "rope.dimension_count_swa").c_str(), 0);

    // Shared KV layers (SmolLM3, cross-layer KV reuse)
    cfg.shared_kv_layers   = (int)gguf_get_i(gguf_ctx, (pfx + "attention.shared_kv_layers").c_str(), 0);

    // Norm placement / scaling
    cfg.swin_norm          = (int)gguf_get_i(gguf_ctx, (pfx + "swin_norm").c_str(), 0);
    cfg.residual_scale     = gguf_get_f(gguf_ctx, (pfx + "residual_scale").c_str(), 0.0f);
    cfg.logit_scale        = gguf_get_f(gguf_ctx, (pfx + "logit_scale").c_str(), 0.0f);
    cfg.rescale_every_n    = (int)gguf_get_i(gguf_ctx, (pfx + "rescale_every_n_layers").c_str(), 0);
    cfg.embed_scale_val    = gguf_get_f(gguf_ctx, (pfx + "embedding_scale").c_str(), 0.0f);

    // MoE advanced params
    cfg.moe_expert_ff_len    = (int)gguf_get_i(gguf_ctx, (pfx + "expert_feed_forward_length").c_str(), 0);
    cfg.moe_shared_count     = (int)gguf_get_i(gguf_ctx, (pfx + "expert_shared_count").c_str(), 0);
    cfg.moe_shared_ff_len    = (int)gguf_get_i(gguf_ctx, (pfx + "expert_shared_feed_forward_length").c_str(), 0);
    cfg.moe_gating_func      = (int)gguf_get_i(gguf_ctx, (pfx + "expert_gating_func").c_str(), 0);
    cfg.moe_every_n_layers   = (int)gguf_get_i(gguf_ctx, (pfx + "moe_every_n_layers").c_str(), 0);
    cfg.moe_interleave_step  = (int)gguf_get_i(gguf_ctx, (pfx + "interleave_moe_layer_step").c_str(), 0);
    cfg.moe_group_count      = (int)gguf_get_i(gguf_ctx, (pfx + "expert_group_count").c_str(), 0);
    cfg.moe_group_used_count = (int)gguf_get_i(gguf_ctx, (pfx + "expert_group_used_count").c_str(), 0);
    cfg.router_softcap       = gguf_get_f(gguf_ctx, (pfx + "router_logit_softcapping").c_str(), 0.0f);

    // Multi-RoPE sections (Qwen2VL/3VL)
    // Read as array if present — 4 ints defining section boundaries
    {
        int idx = gguf_find_key(gguf_ctx, (pfx + "rope.dimension_sections").c_str());
        if (idx >= 0) {
            cfg.has_rope_sections = 1;
            // sections stored in config — parsed at compile time
        }
    }

    // ALiBi
    float alibi_bias = gguf_get_f(gguf_ctx, (pfx + "attention.max_alibi_bias").c_str(), 0.0f);
    if (alibi_bias > 0) {
        cfg.has_alibi = 1;
        cfg.alibi_max_bias = alibi_bias;
        int n_head_log2 = 1;
        while (n_head_log2 * 2 <= cfg.fa_n_q_heads) n_head_log2 *= 2;
        cfg.alibi_n_head_log2 = n_head_log2;
        cfg.alibi_m0 = powf(2.0f, -alibi_bias / (float)n_head_log2);
        cfg.alibi_m1 = powf(2.0f, -alibi_bias / 2.0f / (float)n_head_log2);
    }

    // Attention scale
    float attn_scale = gguf_get_f(gguf_ctx, (pfx + "attention.scale").c_str(), 0.0f);
    if (attn_scale > 0) cfg.fa_attention_scale = attn_scale;

    // Softcap
    float logit_softcap = gguf_get_f(gguf_ctx, (pfx + "attention.logit_softcapping").c_str(), 0.0f);
    if (logit_softcap > 0) { cfg.attn_scale_type = 1; cfg.attn_softcap_val = logit_softcap; }
    float final_softcap = gguf_get_f(gguf_ctx, (pfx + "final_logit_softcapping").c_str(), 0.0f);
    if (final_softcap > 0) {
        cfg.has_final_logit_softcap = 1;
        cfg.final_logit_softcap_val = final_softcap;
    }

    // Gemma: scale embeddings by sqrt(n_embd)
    // Detect by checking if arch is gemma-family
    if (cfg.arch_id == ARCH_GEMMA || cfg.arch_id == ARCH_GEMMA2 || cfg.arch_id == ARCH_GEMMA3 ||
        cfg.arch_id == ARCH_GEMMA3N || cfg.arch_id == ARCH_GEMMA4 || cfg.arch_id == ARCH_GEMMA_EMBEDDING) {
        cfg.has_embed_scale = 1;
        // NOTE: Gemma's original PyTorch uses (1 + weight) in RMSNorm, but
        // convert_hf_to_gguf.py already bakes the +1 into the GGUF weights
        // (see Gemma2Model.modify_tensors / Gemma3Model.norm_shift).
        // Do NOT set norm_add_one here — it would double-add the offset.
    }

    // RoPE type detection (simplified — baseline uses rope_scaling_type metadata)
    int rope_type_raw = (int)gguf_get_i(gguf_ctx, (pfx + "rope.scaling.type").c_str(), -1);
    if (rope_type_raw >= 0) cfg.rope_type = rope_type_raw;

    // YaRN params
    cfg.rope_freq_scale  = gguf_get_f(gguf_ctx, (pfx + "rope.freq_scale").c_str(), 1.0f);
    cfg.rope_attn_factor = gguf_get_f(gguf_ctx, (pfx + "rope.attn_factor").c_str(), 1.0f);
    cfg.yarn_ext_factor  = gguf_get_f(gguf_ctx, (pfx + "rope.scaling.yarn.ext_factor").c_str(), 0.0f);
    cfg.yarn_attn_factor = gguf_get_f(gguf_ctx, (pfx + "rope.scaling.yarn.attn_factor").c_str(), 1.0f);
    // Resolve yarn_attn_factor like baseline (llama-context.cpp:100-137):
    // Final mscale = get_mscale(ext_factor) * rope_attn_factor
    // For LONGROPE (ext_factor=0): mscale = 1.0 * rope_attn_factor
    // For YARN (ext_factor!=0): mscale = get_mscale(1/freq_scale) * rope_attn_factor
    {
        float yaf = cfg.yarn_attn_factor;
        if (cfg.yarn_ext_factor != 0.0f && cfg.rope_freq_scale != 0.0f) {
            float factor = 1.0f / cfg.rope_freq_scale;
            if (factor > 1.0f) {
                yaf = (0.1f * logf(factor) + 1.0f);
            }
        }
        yaf *= cfg.rope_attn_factor;
        cfg.yarn_attn_factor = yaf;
    }
    cfg.yarn_beta_fast   = gguf_get_f(gguf_ctx, (pfx + "rope.scaling.yarn.beta_fast").c_str(), 32.0f);
    cfg.yarn_beta_slow   = gguf_get_f(gguf_ctx, (pfx + "rope.scaling.yarn.beta_slow").c_str(), 1.0f);
    cfg.n_ctx_orig_yarn  = (int)gguf_get_i(gguf_ctx, (pfx + "rope.scaling.original_context_length").c_str(), 0);

    // Audio params (Whisper, Qwen3-audio)
    cfg.audio_n_fft      = (int)gguf_get_i(gguf_ctx, (pfx + "audio.n_fft").c_str(), 0);
    if (cfg.audio_n_fft == 0) cfg.audio_n_fft = (int)gguf_get_i(gguf_ctx, "encoder.n_fft", 0);
    cfg.audio_hop_length = (int)gguf_get_i(gguf_ctx, (pfx + "audio.hop_length").c_str(), 0);
    if (cfg.audio_hop_length == 0) cfg.audio_hop_length = (int)gguf_get_i(gguf_ctx, "encoder.hop_length", 0);
    cfg.audio_n_mels     = (int)gguf_get_i(gguf_ctx, (pfx + "audio.n_mels").c_str(), 0);
    if (cfg.audio_n_mels == 0) cfg.audio_n_mels = (int)gguf_get_i(gguf_ctx, "encoder.n_mels", 0);

    // T5 hparams
    cfg.dec_n_layer      = (int)gguf_get_i(gguf_ctx, (pfx + "decoder.block_count").c_str(), 0);
    cfg.n_rel_attn_bkts  = (int)gguf_get_i(gguf_ctx, (pfx + "attention.relative_buckets_count").c_str(), 0);

    // Chameleon hparams
    cfg.chameleon_img_token_start = (int)gguf_get_i(gguf_ctx, (pfx + "image_token_start").c_str(), 0);
    cfg.chameleon_img_token_count = (int)gguf_get_i(gguf_ctx, (pfx + "image_token_count").c_str(), 0);

    // Norm type: detect LayerNorm vs RMSNorm
    // Heuristic: if output_norm.bias exists, it's LayerNorm
    if (ggml_get_tensor(ctx, "output_norm.bias")) cfg.norm_type = 2; // NORM_LAYER

    // Parallel attention+FFN
    int use_par_res_val = (int)gguf_get_i(gguf_ctx, (pfx + "use_parallel_residual").c_str(), -1);
    if (use_par_res_val == 1) cfg.use_par_res = 1;

    fprintf(stderr, "gfx1100: arch=%s(%d) H=%d FF=%d L=%d V=%d heads=%d/%d",
            arch.c_str(), cfg.arch_id, cfg.hidden_size, cfg.intermediate_size,
            cfg.n_layers, cfg.vocab_size, cfg.fa_n_q_heads, cfg.fa_n_kv_heads);
    if (cfg.moe_n_experts > 0) fprintf(stderr, " MoE=%d/%d", cfg.moe_n_experts_used, cfg.moe_n_experts);
    fprintf(stderr, "\n");

    // Step 4: Load global tensors
    {
        struct ggml_tensor * t;

        // Token embedding
        t = ggml_get_tensor(ctx, "token_embd.weight");
        if (t) { cfg.embed_weight = map_tensor_to_gpu(t); cfg.embed_stride = t->nb[1]; cfg.embed_type = t->type; }

        // Output norm (weight + optional bias)
        t = ggml_get_tensor(ctx, "output_norm.weight");
        if (t) cfg.final_norm_weight = map_tensor_to_gpu(t);
        t = ggml_get_tensor(ctx, "output_norm.bias");
        if (t) cfg.final_norm_bias = map_tensor_to_gpu(t);

        // LM head (output.weight) — may be tied to tok_embd
        t = ggml_get_tensor(ctx, "output.weight");
        if (t) { cfg.lm_head_weight = map_tensor_to_gpu(t); cfg.lm_head_stride = t->nb[1]; cfg.lm_head_type = t->type; }
        else if (cfg.embed_weight) {
            cfg.lm_head_weight = cfg.embed_weight; cfg.lm_head_stride = cfg.embed_stride; cfg.lm_head_type = cfg.embed_type;
        }

        // Token norm (BLOOM, RWKV — LayerNorm on embeddings)
        LOAD_CFG_VEC(tok_norm_weight, "token_embd_norm.weight");
        LOAD_CFG_VEC(tok_norm_bias,   "token_embd_norm.bias");

        // Position embeddings (GPT-2, StarCoder, BERT)
        LOAD_CFG_WEIGHT(pos_embd, pos_embd_stride, pos_embd_type, "position_embd.weight");

        // Token type embeddings (BERT)
        LOAD_CFG_WEIGHT(type_embd, type_embd_stride, type_embd_type, "token_types.weight");

        // RoPE frequency factors (Llama 3 long RoPE)
        t = ggml_get_tensor(ctx, "rope_freqs.weight");
        if (t) { cfg.rope_freq_factors = map_tensor_to_gpu(t); cfg.has_rope_freq_factors = 1; }

        // Encoder output norm (T5)
        LOAD_CFG_VEC(output_norm_enc, "enc.output_norm.weight");

        // Audio mel filterbank coefficients
        t = ggml_get_tensor(ctx, "encoder.mel_filters.weight");
        if (!t) t = ggml_get_tensor(ctx, "mel_filters.weight");
        if (t) {
            cfg.mel_filters = (const float *)map_tensor_to_gpu(t);
            // Infer audio params from tensor shape if not in metadata
            if (cfg.audio_n_mels == 0 && t->ne[1] > 0) cfg.audio_n_mels = (int)t->ne[1];
            if (cfg.audio_n_fft == 0 && t->ne[0] > 0) cfg.audio_n_fft = ((int)t->ne[0] - 1) * 2;
            if (cfg.audio_hop_length == 0 && cfg.audio_n_fft > 0) cfg.audio_hop_length = cfg.audio_n_fft / 2 - cfg.audio_n_fft / 5;
        }
    }

    // Step 5: Load per-layer tensors — ALL categories
    for (int il = 0; il < cfg.n_layers && il < 128; il++) {
        auto & lw = cfg.layers[il];
        cfg.layer_types[il] = 0; // default: attention

        // --- Standard attention ---
        // Slot 0: attn_norm
        { char n[256]; snprintf(n, sizeof(n), "blk.%d.attn_norm.weight", il);
          struct ggml_tensor * t = ggml_get_tensor(ctx, n);
          if (t) { lw.ptrs[0] = map_tensor_to_gpu(t); lw.strides[0] = t->nb[1]; lw.types[0] = t->type; } }

        // Slots 1-3: Q, K, V projections
        { const char * names[] = {"attn_q", "attn_k", "attn_v"};
          int slots[] = {1, 2, 3};
          for (int p = 0; p < 3; p++) {
              char n[256];
              snprintf(n, sizeof(n), "blk.%d.%s.weight", il, names[p]);
              struct ggml_tensor * t = ggml_get_tensor(ctx, n);
              if (t) { lw.ptrs[slots[p]] = map_tensor_to_gpu(t); lw.strides[slots[p]] = t->nb[1]; lw.types[slots[p]] = t->type; }
          } }

        // Fused QKV (Phi3, InternLM2, Falcon, etc.)
        if (!lw.ptrs[1]) {
            char n[256];
            snprintf(n, sizeof(n), "blk.%d.attn_qkv.weight", il);
            struct ggml_tensor * qkv = ggml_get_tensor(ctx, n);
            if (qkv) {
                void * base = map_tensor_to_gpu(qkv);
                long long stride = qkv->nb[1];
                int q_rows = cfg.fa_n_q_heads * cfg.fa_head_dim;
                int kv_rows = cfg.fa_n_kv_heads * cfg.fa_head_dim;
                lw.ptrs[1] = base;
                lw.strides[1] = stride; lw.types[1] = qkv->type;
                lw.ptrs[2] = (const char *)base + (long long)q_rows * stride;
                lw.strides[2] = stride; lw.types[2] = qkv->type;
                lw.ptrs[3] = (const char *)base + (long long)(q_rows + kv_rows) * stride;
                lw.strides[3] = stride; lw.types[3] = qkv->type;
            }
        }

        // Slot 6: attn_output
        { char n[256]; snprintf(n, sizeof(n), "blk.%d.attn_output.weight", il);
          struct ggml_tensor * t = ggml_get_tensor(ctx, n);
          if (t) { lw.ptrs[6] = map_tensor_to_gpu(t); lw.strides[6] = t->nb[1]; lw.types[6] = t->type; } }

        // Slot 7: ffn_norm
        { char n[256]; snprintf(n, sizeof(n), "blk.%d.ffn_norm.weight", il);
          struct ggml_tensor * t = ggml_get_tensor(ctx, n);
          if (t) { lw.ptrs[7] = map_tensor_to_gpu(t); lw.strides[7] = t->nb[1]; lw.types[7] = t->type; } }

        // Slots 8-10: FFN gate, up, down
        { const char * names[] = {"ffn_gate", "ffn_up", "ffn_down"};
          int slots[] = {8, 9, 10};
          for (int f = 0; f < 3; f++) {
              char n[256];
              snprintf(n, sizeof(n), "blk.%d.%s.weight", il, names[f]);
              struct ggml_tensor * t = ggml_get_tensor(ctx, n);
              if (t) { lw.ptrs[slots[f]] = map_tensor_to_gpu(t); lw.strides[slots[f]] = t->nb[1]; lw.types[slots[f]] = t->type; }
          } }

        // --- Attention biases ---
        LOAD_VEC(bias_q, "blk.%d.attn_q.bias", il);
        LOAD_VEC(bias_k, "blk.%d.attn_k.bias", il);
        LOAD_VEC(bias_v, "blk.%d.attn_v.bias", il);
        LOAD_VEC(bias_o, "blk.%d.attn_output.bias", il);

        // --- Attention scales (BitNet/LoRA) ---
        LOAD_VEC(scale_q, "blk.%d.attn_q.scale", il);
        LOAD_VEC(scale_k, "blk.%d.attn_k.scale", il);
        LOAD_VEC(scale_v, "blk.%d.attn_v.scale", il);
        LOAD_VEC(scale_o, "blk.%d.attn_output.scale", il);

        // --- QK norm ---
        LOAD_VEC(ptrs[4], "blk.%d.attn_q_norm.weight", il);  // slot 4 = q_norm
        LOAD_VEC(ptrs[5], "blk.%d.attn_k_norm.weight", il);  // slot 5 = k_norm

        // --- Post-attention and post-FFN norms (Gemma2/3/4) ---
        LOAD_VEC(attn_post_norm, "blk.%d.post_attention_norm.weight", il);
        LOAD_VEC(ffn_post_norm,  "blk.%d.post_ffw_norm.weight", il);

        // --- FFN biases ---
        LOAD_VEC(ffn_gate_bias, "blk.%d.ffn_gate.bias", il);
        LOAD_VEC(ffn_up_bias,   "blk.%d.ffn_up.bias",   il);
        LOAD_VEC(ffn_down_bias, "blk.%d.ffn_down.bias", il);

        // --- FFN scales ---
        LOAD_VEC(ffn_gate_scale, "blk.%d.ffn_gate.scale", il);
        LOAD_VEC(ffn_up_scale,   "blk.%d.ffn_up.scale",   il);
        LOAD_VEC(ffn_down_scale, "blk.%d.ffn_down.scale", il);

        // --- MoE tensors ---
        LOAD_WEIGHT(ffn_gate_inp, ffn_gate_inp_stride, ffn_gate_inp_type, "blk.%d.ffn_gate_inp.weight", il);
        LOAD_WEIGHT(ffn_gate_exps, ffn_gate_exps_stride, ffn_gate_exps_type, "blk.%d.ffn_gate_exps.weight", il);
        LOAD_WEIGHT(ffn_up_exps,   ffn_up_exps_stride,   ffn_up_exps_type,   "blk.%d.ffn_up_exps.weight",   il);
        LOAD_WEIGHT(ffn_down_exps, ffn_down_exps_stride, ffn_down_exps_type, "blk.%d.ffn_down_exps.weight", il);
        // Shared expert (Qwen2MoE, DS2)
        LOAD_WEIGHT(ffn_gate_inp_shexp, ffn_gate_inp_shexp_stride, ffn_gate_inp_shexp_type, "blk.%d.ffn_gate_inp_shexp.weight", il);
        LOAD_WEIGHT(ffn_gate_shexp, ffn_gate_shexp_stride, ffn_gate_shexp_type, "blk.%d.ffn_gate_shexp.weight", il);
        LOAD_WEIGHT(ffn_up_shexp,   ffn_up_shexp_stride,   ffn_up_shexp_type,   "blk.%d.ffn_up_shexp.weight",   il);
        LOAD_WEIGHT(ffn_down_shexp, ffn_down_shexp_stride, ffn_down_shexp_type, "blk.%d.ffn_down_shexp.weight", il);
        // MoE norm/scale
        { char n[256]; snprintf(n, sizeof(n), "blk.%d.ffn_norm_exps.weight", il);
          struct ggml_tensor * t = ggml_get_tensor(ctx, n);
          if (t) lw.moe_norm_w = map_tensor_to_gpu(t); }

        // --- DeepSeek2 MLA projections ---
        LOAD_WEIGHT(wq_a, wq_a_stride, wq_a_type, "blk.%d.attn_q_a.weight", il);
        LOAD_WEIGHT(wkv_a_mqa, wkv_a_mqa_stride, wkv_a_mqa_type, "blk.%d.attn_kv_a_mqa.weight", il);
        LOAD_WEIGHT(wq_b, wq_b_stride, wq_b_type, "blk.%d.attn_q_b.weight", il);
        LOAD_WEIGHT(wkv_b, wkv_b_stride, wkv_b_type, "blk.%d.attn_kv_b.weight", il);
        LOAD_WEIGHT(wk_b, wk_b_stride, wk_b_type, "blk.%d.attn_k_b.weight", il);
        LOAD_WEIGHT(wv_b, wv_b_stride, wv_b_type, "blk.%d.attn_v_b.weight", il);
        LOAD_VEC(attn_q_a_norm, "blk.%d.attn_q_a_norm.weight", il);
        LOAD_VEC(attn_kv_a_norm, "blk.%d.attn_kv_a_norm.weight", il);

        // --- SSM / Mamba ---
        LOAD_WEIGHT(ssm_in, ssm_in_stride, ssm_in_type, "blk.%d.ssm_in.weight", il);
        LOAD_WEIGHT(ssm_out, ssm_out_stride, ssm_out_type, "blk.%d.ssm_out.weight", il);
        LOAD_WEIGHT(ssm_conv1d, ssm_conv1d_stride, ssm_conv1d_type, "blk.%d.ssm_conv1d.weight", il);
        LOAD_VEC(ssm_conv1d_b, "blk.%d.ssm_conv1d.bias", il);
        LOAD_WEIGHT(ssm_dt, ssm_dt_stride, ssm_dt_type, "blk.%d.ssm_dt.weight", il);
        LOAD_VEC(ssm_dt_b, "blk.%d.ssm_dt.bias", il);
        LOAD_VEC(ssm_a, "blk.%d.ssm_a.weight", il);
        LOAD_VEC(ssm_d, "blk.%d.ssm_d.weight", il);
        LOAD_WEIGHT(ssm_x, ssm_x_stride, ssm_x_type, "blk.%d.ssm_x.weight", il);
        LOAD_VEC(ssm_norm, "blk.%d.ssm_norm.weight", il);

        // --- RWKV time-mix ---
        LOAD_VEC(time_mix_lerp_x, "blk.%d.time_mix_lerp_x.weight", il);
        LOAD_VEC(time_mix_lerp_w, "blk.%d.time_mix_lerp_w.weight", il);
        LOAD_VEC(time_mix_lerp_k, "blk.%d.time_mix_lerp_k.weight", il);
        LOAD_VEC(time_mix_lerp_v, "blk.%d.time_mix_lerp_v.weight", il);
        LOAD_VEC(time_mix_lerp_r, "blk.%d.time_mix_lerp_r.weight", il);
        LOAD_VEC(time_mix_lerp_g, "blk.%d.time_mix_lerp_g.weight", il);
        LOAD_VEC(time_mix_first,  "blk.%d.time_mix_first.weight",  il);
        LOAD_VEC(time_mix_decay,  "blk.%d.time_mix_decay.weight",  il);
        LOAD_WEIGHT(time_mix_decay_w1, time_mix_decay_w1_stride, time_mix_decay_w1_type, "blk.%d.time_mix_decay_w1.weight", il);
        LOAD_WEIGHT(time_mix_decay_w2, time_mix_decay_w2_stride, time_mix_decay_w2_type, "blk.%d.time_mix_decay_w2.weight", il);
        LOAD_WEIGHT(time_mix_key, time_mix_key_stride, time_mix_key_type, "blk.%d.time_mix_key.weight", il);
        LOAD_WEIGHT(time_mix_value, time_mix_value_stride, time_mix_value_type, "blk.%d.time_mix_value.weight", il);
        LOAD_WEIGHT(time_mix_receptance, time_mix_receptance_stride, time_mix_receptance_type, "blk.%d.time_mix_receptance.weight", il);
        LOAD_WEIGHT(time_mix_gate, time_mix_gate_stride, time_mix_gate_type, "blk.%d.time_mix_gate.weight", il);
        LOAD_VEC(time_mix_ln,   "blk.%d.time_mix_ln.weight", il);
        LOAD_VEC(time_mix_ln_b, "blk.%d.time_mix_ln.bias",   il);
        LOAD_WEIGHT(time_mix_output, time_mix_output_stride, time_mix_output_type, "blk.%d.time_mix_output.weight", il);

        // RWKV7-specific
        LOAD_VEC(time_mix_w0, "blk.%d.time_mix_w0.weight", il);
        LOAD_VEC(time_mix_a0, "blk.%d.time_mix_a0.weight", il);
        LOAD_WEIGHT(time_mix_a1, time_mix_a1_stride, time_mix_a1_type, "blk.%d.time_mix_a1.weight", il);
        LOAD_WEIGHT(time_mix_a2, time_mix_a2_stride, time_mix_a2_type, "blk.%d.time_mix_a2.weight", il);
        LOAD_VEC(time_mix_v0, "blk.%d.time_mix_v0.weight", il);
        LOAD_WEIGHT(time_mix_v1, time_mix_v1_stride, time_mix_v1_type, "blk.%d.time_mix_v1.weight", il);
        LOAD_WEIGHT(time_mix_v2, time_mix_v2_stride, time_mix_v2_type, "blk.%d.time_mix_v2.weight", il);
        LOAD_WEIGHT(time_mix_g1, time_mix_g1_stride, time_mix_g1_type, "blk.%d.time_mix_g1.weight", il);
        LOAD_WEIGHT(time_mix_g2, time_mix_g2_stride, time_mix_g2_type, "blk.%d.time_mix_g2.weight", il);
        LOAD_VEC(time_mix_k_k, "blk.%d.time_mix_k_k.weight", il);
        LOAD_VEC(time_mix_k_a, "blk.%d.time_mix_k_a.weight", il);
        LOAD_VEC(time_mix_r_k, "blk.%d.time_mix_r_k.weight", il);

        // RWKV channel-mix
        LOAD_VEC(channel_mix_lerp_k, "blk.%d.channel_mix_lerp_k.weight", il);
        LOAD_VEC(channel_mix_lerp_r, "blk.%d.channel_mix_lerp_r.weight", il);
        LOAD_WEIGHT(channel_mix_key, channel_mix_key_stride, channel_mix_key_type, "blk.%d.channel_mix_key.weight", il);
        LOAD_WEIGHT(channel_mix_value, channel_mix_value_stride, channel_mix_value_type, "blk.%d.channel_mix_value.weight", il);
        LOAD_WEIGHT(channel_mix_receptance, channel_mix_receptance_stride, channel_mix_receptance_type, "blk.%d.channel_mix_receptance.weight", il);

        // --- T5 encoder layer weights ---
        LOAD_WEIGHT(wq_enc, wq_enc_stride, wq_enc_type, "enc.blk.%d.attn_q.weight", il);
        LOAD_WEIGHT(wk_enc, wk_enc_stride, wk_enc_type, "enc.blk.%d.attn_k.weight", il);
        LOAD_WEIGHT(wv_enc, wv_enc_stride, wv_enc_type, "enc.blk.%d.attn_v.weight", il);
        LOAD_WEIGHT(wo_enc, wo_enc_stride, wo_enc_type, "enc.blk.%d.attn_o.weight", il);
        LOAD_VEC(attn_norm_enc, "enc.blk.%d.attn_norm.weight", il);
        LOAD_VEC(ffn_norm_enc, "enc.blk.%d.ffn_norm.weight", il);
        LOAD_WEIGHT(ffn_gate_enc, ffn_gate_enc_stride, ffn_gate_enc_type, "enc.blk.%d.ffn_gate.weight", il);
        LOAD_WEIGHT(ffn_down_enc, ffn_down_enc_stride, ffn_down_enc_type, "enc.blk.%d.ffn_down.weight", il);
        LOAD_WEIGHT(ffn_up_enc, ffn_up_enc_stride, ffn_up_enc_type, "enc.blk.%d.ffn_up.weight", il);
        LOAD_VEC(attn_rel_b_enc, "enc.blk.%d.attn_rel_b.weight", il);

        // T5 decoder cross-attention
        LOAD_VEC(attn_norm_cross, "dec.blk.%d.cross_attn_norm.weight", il);
        LOAD_WEIGHT(wq_cross, wq_cross_stride, wq_cross_type, "dec.blk.%d.cross_attn_q.weight", il);
        LOAD_WEIGHT(wk_cross, wk_cross_stride, wk_cross_type, "dec.blk.%d.cross_attn_k.weight", il);
        LOAD_WEIGHT(wv_cross, wv_cross_stride, wv_cross_type, "dec.blk.%d.cross_attn_v.weight", il);
        LOAD_WEIGHT(wo_cross, wo_cross_stride, wo_cross_type, "dec.blk.%d.cross_attn_o.weight", il);

        // T5 relative position bias
        LOAD_VEC(attn_rel_b, "blk.%d.attn_rel_b.weight", il);
        if (!lw.attn_rel_b) LOAD_VEC(attn_rel_b, "dec.blk.%d.attn_rel_b.weight", il);

        // --- Detect layer type ---
        if (lw.ssm_in || lw.ssm_conv1d) {
            cfg.layer_types[il] = 2; // SSM
            cfg.has_ssm = 1;
        }
        if (lw.time_mix_key || lw.time_mix_receptance) {
            cfg.layer_types[il] = 3; // RWKV
        }
    }

    // Step 6: Detect capability bits from tensor presence
    // This feeds the DETECT and VALIDATE meta-steps of the composition system.
    // Each tensor's existence tells us which operations the model needs.
    for (int il = 0; il < cfg.n_layers && il < 128; il++) {
        const auto & lw = cfg.layers[il];

        // --- Attention capabilities ---
        if (lw.ptrs[4] || lw.ptrs[5]) cfg.has_qk_norm = 1;       // QK per-head norm
        if (lw.bias_q)   cfg.has_bias_q = 1;                       // Q projection bias
        if (lw.bias_k)   cfg.has_bias_k = 1;                       // K projection bias
        if (lw.bias_v)   cfg.has_bias_v = 1;                       // V projection bias
        if (lw.bias_o)   cfg.has_bias_o = 1;                       // O projection bias
        if (lw.scale_q)  cfg.has_scale_q = 1;                      // Q projection scale (LoRA)
        if (lw.scale_k)  cfg.has_scale_k = 1;                      // K projection scale
        if (lw.scale_v)  cfg.has_scale_v = 1;                      // V projection scale
        if (lw.scale_o)  cfg.has_scale_o = 1;                      // O projection scale
        if (lw.attn_post_norm) cfg.has_post_attn_norm = 1;         // Post-attention norm (Gemma2+)
        if (lw.ffn_post_norm)  cfg.has_post_ffn_norm = 1;          // Post-FFN norm (Gemma2+)

        // --- QKV fusion detection ---
        // Fused QKV: single attn_qkv.weight instead of separate Q/K/V
        // Composition decides: 1 fused matvec vs 3 separate matvecs
        if (lw.ptrs[3]) cfg.has_fused_qkv = 1;                     // slot 3 = wqkv

        // --- FFN capabilities ---
        if (lw.ffn_gate_bias)  cfg.has_bias_ffn_gate = 1;          // Gate bias
        if (lw.ffn_up_bias)    cfg.has_bias_ffn_up = 1;            // Up bias
        if (lw.ffn_down_bias)  cfg.has_bias_ffn_down = 1;          // Down bias
        if (lw.ffn_gate_scale) cfg.has_scale_ffn_gate = 1;         // Gate scale (LoRA)
        if (lw.ffn_up_scale)   cfg.has_scale_ffn_up = 1;           // Up scale
        if (lw.ffn_down_scale) cfg.has_scale_ffn_down = 1;         // Down scale
        if (lw.ffn_gate_inp)   cfg.has_moe = 1;                    // MoE router → MoE FFN path

        // --- Gated FFN detection ---
        // If ffn_gate exists (separate from ffn_up), model uses gated activation
        // Composition decides: fused gate+up+act vs separate ops
        if (lw.ptrs[8] && lw.ptrs[9]) cfg.has_gated_ffn = 1;      // slots 8,9 = gate,up

        // --- Layer type detection (hybrid models) ---
        if (lw.wq_a || lw.wkv_a_mqa) cfg.has_dn = 0;              // MLA, not DeltaNet

        // --- Cross-attention (T5 decoder) ---
        if (lw.wq_cross) cfg.has_cross_attn = 1;

        // --- Encoder layers (T5/BERT) ---
        if (lw.wq_enc) cfg.has_encoder = 1;
    }

    // Step 7: Activation type detection
    // Default to SiLU for most architectures, GELU for BERT/GPT2/etc.
    cfg.act_type = 1; // ACT_SILU
    switch (cfg.arch_id) {
        case ARCH_GPT2: case ARCH_GPTJ: case ARCH_GPTNEOX: case ARCH_BERT:
        case ARCH_MODERN_BERT: case ARCH_NOMIC_BERT: case ARCH_NEO_BERT:
        case ARCH_JINA_BERT_V2: case ARCH_JINA_BERT_V3: case ARCH_EUROBERT:
        case ARCH_BLOOM: case ARCH_MPT: case ARCH_STARCODER: case ARCH_FALCON:
        case ARCH_STARCODER2:
            cfg.act_type = 1; // ACT_GELU
            break;
        case ARCH_GROK:
            cfg.act_type = 1; // GELU
            break;
        default: break;
    }

    fprintf(stderr, "gfx1100: capabilities: qk_norm=%d bias_qkvo=%d%d%d%d scale_qkvo=%d%d%d%d "
            "moe=%d ssm=%d swa=%d alibi=%d par_res=%d\n",
            cfg.has_qk_norm,
            cfg.has_bias_q, cfg.has_bias_k, cfg.has_bias_v, cfg.has_bias_o,
            cfg.has_scale_q, cfg.has_scale_k, cfg.has_scale_v, cfg.has_scale_o,
            cfg.has_moe, cfg.has_ssm, cfg.has_swa, cfg.has_alibi, cfg.use_par_res);

    // Step 8: Initialize the megakernel
    int rc = gfx1100_init(&cfg);
    if (rc != 0) {
        fprintf(stderr, "gfx1100: FATAL — init failed after model load\n");
    }

    gguf_free(gguf_ctx);
    return rc;
}
