// comp-detect.h — DETECT meta-step: systematic capability detection
//
// Scans the model config to build a complete capability bitfield. Each layer
// is inspected for which tensors exist, and the union of features across all
// layers determines the "capabilities" the model needs.
//
// This is the second half of DETECT (the first half is reading GGUF keys).
// Together they give the composition system complete knowledge of the model's
// operational requirements BEFORE any kernel composition decisions are made.
//
// Philosophy: tensor existence is the source of truth. If a model has
// `blk.N.attn_q.bias`, it needs bias-add after Q projection — regardless of
// what the GGUF metadata says. Always trust the tensors over the metadata.
#pragma once

#include "../gfx1100-internal.h"
#include <cstdio>

// ============================================================================
// Per-layer capability bitfield (fits in a single uint64_t for fast ORing)
// ============================================================================
enum comp_layer_cap : uint64_t {
    // Norm capabilities
    CAP_ATTN_NORM          = 1ULL << 0,   // pre-attention norm (always present)
    CAP_POST_ATTN_NORM     = 1ULL << 1,   // Gemma2/3/4: norm after attention output
    CAP_FFN_NORM           = 1ULL << 2,   // pre-FFN norm
    CAP_POST_FFN_NORM      = 1ULL << 3,   // Gemma2/3/4: norm after FFN output
    CAP_QK_NORM            = 1ULL << 4,   // per-head Q/K L2 norm (Qwen3, Llama4)

    // QKV projection capabilities
    CAP_WQ                 = 1ULL << 5,   // separate Q weight
    CAP_WK                 = 1ULL << 6,   // separate K weight
    CAP_WV                 = 1ULL << 7,   // separate V weight
    CAP_WQKV               = 1ULL << 8,   // fused QKV weight (Phi3, some others)
    CAP_BIAS_Q             = 1ULL << 9,
    CAP_BIAS_K             = 1ULL << 10,
    CAP_BIAS_V             = 1ULL << 11,
    CAP_SCALE_Q            = 1ULL << 12,  // LoRA/BitNet scale
    CAP_SCALE_K            = 1ULL << 13,
    CAP_SCALE_V            = 1ULL << 14,

    // Attention output
    CAP_WO                 = 1ULL << 15,  // O projection
    CAP_BIAS_O             = 1ULL << 16,
    CAP_SCALE_O            = 1ULL << 17,
    CAP_GATED_ATTN         = 1ULL << 18,  // gated attention output

    // FFN capabilities
    CAP_FFN_GATE           = 1ULL << 19,  // gate projection (gated activation)
    CAP_FFN_UP             = 1ULL << 20,  // up projection
    CAP_FFN_DOWN           = 1ULL << 21,  // down projection
    CAP_FFN_GATE_BIAS      = 1ULL << 22,
    CAP_FFN_UP_BIAS        = 1ULL << 23,
    CAP_FFN_DOWN_BIAS      = 1ULL << 24,
    CAP_FFN_GATE_SCALE     = 1ULL << 25,
    CAP_FFN_UP_SCALE       = 1ULL << 26,
    CAP_FFN_DOWN_SCALE     = 1ULL << 27,

    // MoE routing
    CAP_MOE_ROUTER         = 1ULL << 28,  // ffn_gate_inp (expert routing)
    CAP_MOE_EXPERTS        = 1ULL << 29,  // expert weights
    CAP_MOE_SHARED_EXPERT  = 1ULL << 30,  // shared expert FFN

    // Hybrid layer types
    CAP_SSM                = 1ULL << 31,  // Mamba/Mamba2 state space
    CAP_RWKV_TIME_MIX      = 1ULL << 32,  // RWKV time-mix
    CAP_RWKV_CHANNEL_MIX   = 1ULL << 33,  // RWKV channel-mix
    CAP_MLA                = 1ULL << 34,  // DeepSeek2 MLA (compressed KV)
    CAP_DELTANET           = 1ULL << 35,  // Qwen35 DeltaNet

    // T5 / encoder-decoder
    CAP_CROSS_ATTN         = 1ULL << 36,  // cross-attention (T5 decoder)
    CAP_ENCODER            = 1ULL << 37,  // encoder layer (T5, BERT)
    CAP_REL_POS_BIAS       = 1ULL << 38,  // relative position bias (T5)

    // Convolutional
    CAP_SSM_CONV1D         = 1ULL << 39,  // SSM conv1d
    CAP_AUDIO_MEL          = 1ULL << 40,  // audio mel filterbank

    // --- Extended DETECT bits (all-tensor scan) ---
    // LayerNorm biases (BERT, BLOOM, Falcon, GPT-2, Phi, etc.)
    CAP_ATTN_NORM_BIAS     = 1ULL << 41,  // attn_norm_bias
    CAP_FFN_NORM_BIAS      = 1ULL << 42,  // ffn_norm_bias
    CAP_QK_NORM_BIAS       = 1ULL << 43,  // attn_q_norm_b / attn_k_norm_b

    // BitNet sub-norms (RMSNorm between attn/FFN core and output projection)
    CAP_ATTN_SUB_NORM      = 1ULL << 44,
    CAP_FFN_SUB_NORM       = 1ULL << 45,

    // DeepSeek2 MLA detail
    CAP_MLA_Q_NORM         = 1ULL << 46,  // attn_q_a_norm (Q LoRA norm)
    CAP_MLA_KV_NORM        = 1ULL << 47,  // attn_kv_a_norm (KV compressed norm)
    CAP_MLA_ABSORBED       = 1ULL << 48,  // wk_b / wv_b (absorbed decode path)
    CAP_MLA_UNCOMPRESSED   = 1ULL << 49,  // wkv_b (full KV up-proj for prefill)

    // BERT family post-norms
    CAP_BERT_POST_NORM     = 1ULL << 50,  // attn_out_norm OR layer_out_norm
    CAP_BERT_ATTN_NORM_2   = 1ULL << 51,  // Jina-BERT-V2 extra norm

    // SSM variants
    CAP_SSM_DTBC_NORM      = 1ULL << 52,  // FalconMamba dt/B/C RMSNorm
    CAP_SSM_MAMBA2_NORM    = 1ULL << 53,  // Mamba2 grouped RMSNorm (ssm_norm)

    // CogVLM visual expert
    CAP_VISEXP_ATTN        = 1ULL << 54,  // visexp_wqkv / visexp_wo
    CAP_VISEXP_FFN         = 1ULL << 55,  // visexp_ffn_*

    // RWKV variants
    CAP_RWKV_GATED_TIME_MIX = 1ULL << 56, // time_mix_gate present
    CAP_RWKV_FUSED_LERP     = 1ULL << 57, // time_mix_lerp_fused (vs. separate lerps)
    CAP_RWKV7               = 1ULL << 58, // RWKV7-specific a/v/g/k_k/k_a/r_k weights

    // T5 variants
    CAP_T5_GATED_FFN        = 1ULL << 59, // ffn_gate_enc present (Flan-T5)
};

// ============================================================================
// Complete model capability report
// ============================================================================
struct comp_capabilities {
    uint64_t global_caps;            // OR of all layers
    uint64_t per_layer_caps[128];    // per-layer detail
    int n_layers;

    // Consistency flags (set if all non-trivial layers share this capability)
    bool qkv_fusion_consistent;      // all attention layers either fused or all separate
    bool gated_ffn_consistent;       // all FFN layers either gated or all ungated
    bool bias_pattern_consistent;    // bias pattern same across all layers
    bool moe_layer_pattern;          // MoE at every layer vs alternating

    // Attention layer indices (0-based)
    int n_attn_layers;
    int n_ssm_layers;
    int n_rwkv_layers;
    int n_dn_layers;
    int n_moe_layers;

    // Weight type homogeneity — can we use a single matvec kernel variant?
    bool weight_types_uniform[16];   // per-slot: all layers use the same type
    int  weight_types_first[16];     // per-slot: the type of slot in layer 0

    // Derived decisions for COMPOSE
    bool can_fuse_qkv_globally;      // safe to compose a fused-QKV plan
    bool can_fuse_gate_up_globally;  // safe to compose fused gate+up+act
    bool can_use_single_attn_kernel; // all attention layers same config
};

// ============================================================================
// Detect capabilities by scanning all layer weights
// ============================================================================
static void comp_detect(const gfx1100_model_config & cfg,
                        comp_capabilities & caps) {
    memset(&caps, 0, sizeof(caps));
    caps.n_layers = cfg.n_layers;

    // Initialize type-first to -1 (unknown) for uniformity tracking
    for (int s = 0; s < 16; s++) {
        caps.weight_types_uniform[s] = true;
        caps.weight_types_first[s] = -1;
    }

    // Scan each layer's weights
    for (int il = 0; il < cfg.n_layers; il++) {
        const gfx1100_layer_weights & lw = cfg.layers[il];

        // DEBUG: dump ptrs for the first layer to verify the struct layout
        if (il == 0 && getenv("GFX1100_COMPOSITION_DIAG")) {
            fprintf(stderr, "  DEBUG L0 ptrs:");
            for (int s = 0; s < 11; s++) {
                fprintf(stderr, " [%d]=%s", s, lw.ptrs[s] ? "Y" : "-");
            }
            fprintf(stderr, "\n  DEBUG L0 extras: attn_post_norm=%s ffn_post_norm=%s bias_q=%s ffn_gate_inp=%s\n",
                    lw.attn_post_norm ? "Y" : "-",
                    lw.ffn_post_norm ? "Y" : "-",
                    lw.bias_q ? "Y" : "-",
                    lw.ffn_gate_inp ? "Y" : "-");
        }

        uint64_t c = 0;

        // --- Norms (slot 0 = attn_norm, slot 7 = ffn_norm) ---
        if (lw.ptrs[0])          c |= CAP_ATTN_NORM;
        if (lw.ptrs[7])          c |= CAP_FFN_NORM;
        if (lw.attn_post_norm)   c |= CAP_POST_ATTN_NORM;
        if (lw.ffn_post_norm)    c |= CAP_POST_FFN_NORM;
        if (lw.ptrs[4] || lw.ptrs[5]) c |= CAP_QK_NORM;

        // --- QKV projections (slots 1=wq, 2=wk, 3=wv, wqkv is separate) ---
        if (lw.ptrs[1])  c |= CAP_WQ;
        if (lw.ptrs[2])  c |= CAP_WK;
        if (lw.ptrs[3])  c |= CAP_WV;
        // Check for fused wqkv (stored in specific slot pattern — test harness sets it)
        // For detection: if wqkv is set, Q/K/V separate slots may not be
        // This is resolved by the test harness / config loader

        // --- Attention biases/scales ---
        if (lw.bias_q)  c |= CAP_BIAS_Q;
        if (lw.bias_k)  c |= CAP_BIAS_K;
        if (lw.bias_v)  c |= CAP_BIAS_V;
        if (lw.bias_o)  c |= CAP_BIAS_O;
        if (lw.scale_q) c |= CAP_SCALE_Q;
        if (lw.scale_k) c |= CAP_SCALE_K;
        if (lw.scale_v) c |= CAP_SCALE_V;
        if (lw.scale_o) c |= CAP_SCALE_O;

        // --- Attention output (slot 6 = wo) ---
        if (lw.ptrs[6]) c |= CAP_WO;

        // --- FFN (slots 8=gate, 9=up, 10=down) ---
        if (lw.ptrs[8])  c |= CAP_FFN_GATE;
        if (lw.ptrs[9])  c |= CAP_FFN_UP;
        if (lw.ptrs[10]) c |= CAP_FFN_DOWN;
        if (lw.ffn_gate_bias)  c |= CAP_FFN_GATE_BIAS;
        if (lw.ffn_up_bias)    c |= CAP_FFN_UP_BIAS;
        if (lw.ffn_down_bias)  c |= CAP_FFN_DOWN_BIAS;
        if (lw.ffn_gate_scale) c |= CAP_FFN_GATE_SCALE;
        if (lw.ffn_up_scale)   c |= CAP_FFN_UP_SCALE;
        if (lw.ffn_down_scale) c |= CAP_FFN_DOWN_SCALE;

        // --- MoE ---
        if (lw.ffn_gate_inp)       c |= CAP_MOE_ROUTER;
        if (lw.ffn_gate_inp_shexp) c |= CAP_MOE_SHARED_EXPERT;

        // --- Hybrid layer types ---
        if (lw.ssm_in || lw.ssm_a) c |= CAP_SSM;
        if (lw.ssm_conv1d)         c |= CAP_SSM_CONV1D;
        if (lw.time_mix_key)       c |= CAP_RWKV_TIME_MIX;
        if (lw.channel_mix_key)    c |= CAP_RWKV_CHANNEL_MIX;
        if (lw.wq_a || lw.wkv_a_mqa) c |= CAP_MLA;
        // DeltaNet has specific DN tensor patterns — detect from hparams
        if (cfg.has_dn && cfg.layer_types[il] == 1) c |= CAP_DELTANET;

        // --- T5 / encoder-decoder ---
        if (lw.wq_cross) c |= CAP_CROSS_ATTN;
        if (lw.wq_enc)   c |= CAP_ENCODER;
        if (lw.attn_rel_b || lw.attn_rel_b_enc) c |= CAP_REL_POS_BIAS;
        if (lw.ffn_gate_enc) c |= CAP_T5_GATED_FFN;

        // --- Extended detection (all-tensor scan) ---
        // LayerNorm biases — BERT, BLOOM, Falcon, GPT-2, Phi, etc.
        if (lw.attn_norm_bias) c |= CAP_ATTN_NORM_BIAS;
        if (lw.ffn_norm_bias)  c |= CAP_FFN_NORM_BIAS;
        if (lw.attn_q_norm_b || lw.attn_k_norm_b) c |= CAP_QK_NORM_BIAS;

        // BitNet sub-norms (between core op and output projection)
        if (lw.attn_sub_norm) c |= CAP_ATTN_SUB_NORM;
        if (lw.ffn_sub_norm)  c |= CAP_FFN_SUB_NORM;

        // DeepSeek2 MLA detail
        if (lw.attn_q_a_norm)  c |= CAP_MLA_Q_NORM;
        if (lw.attn_kv_a_norm) c |= CAP_MLA_KV_NORM;
        if (lw.wk_b || lw.wv_b) c |= CAP_MLA_ABSORBED;
        if (lw.wkv_b)          c |= CAP_MLA_UNCOMPRESSED;

        // BERT post-norms + Jina-BERT extra norm
        if (lw.attn_out_norm || lw.layer_out_norm) c |= CAP_BERT_POST_NORM;
        if (lw.attn_norm_2) c |= CAP_BERT_ATTN_NORM_2;

        // SSM variants
        if (lw.ssm_dt_norm || lw.ssm_b_norm || lw.ssm_c_norm) c |= CAP_SSM_DTBC_NORM;
        if (lw.ssm_norm) c |= CAP_SSM_MAMBA2_NORM;

        // CogVLM visual expert (image-token path)
        if (lw.visexp_wqkv || lw.visexp_wo) c |= CAP_VISEXP_ATTN;
        if (lw.visexp_ffn_gate || lw.visexp_ffn_down || lw.visexp_ffn_up) c |= CAP_VISEXP_FFN;

        // RWKV detail
        if (lw.time_mix_gate)        c |= CAP_RWKV_GATED_TIME_MIX;
        if (lw.time_mix_lerp_fused)  c |= CAP_RWKV_FUSED_LERP;
        // RWKV7 tensors (any one implies the variant)
        if (lw.time_mix_w0 || lw.time_mix_a0 || lw.time_mix_a1 || lw.time_mix_a2 ||
            lw.time_mix_v0 || lw.time_mix_v1 || lw.time_mix_v2 ||
            lw.time_mix_g1 || lw.time_mix_g2 ||
            lw.time_mix_k_k || lw.time_mix_k_a || lw.time_mix_r_k) {
            c |= CAP_RWKV7;
        }

        // --- Record per-layer ---
        caps.per_layer_caps[il] = c;
        caps.global_caps |= c;

        // --- Count layer types ---
        if (c & CAP_SSM)           caps.n_ssm_layers++;
        else if (c & CAP_RWKV_TIME_MIX) caps.n_rwkv_layers++;
        else if (c & CAP_DELTANET)      caps.n_dn_layers++;
        else                             caps.n_attn_layers++;
        if (c & CAP_MOE_ROUTER)    caps.n_moe_layers++;

        // --- Track weight type uniformity ---
        for (int s = 0; s < 16; s++) {
            if (!lw.ptrs[s]) continue;
            if (caps.weight_types_first[s] == -1) {
                caps.weight_types_first[s] = lw.types[s];
            } else if (caps.weight_types_first[s] != lw.types[s]) {
                caps.weight_types_uniform[s] = false;
            }
        }
    }

    // ========================================================================
    // Cross-layer consistency analysis
    // ========================================================================

    // QKV fusion consistency: either ALL attention layers have fused QKV,
    // or ALL have separate Q/K/V. Mixed = can't compose a uniform plan.
    bool any_wqkv = false;
    bool any_sep_qkv = false;
    for (int il = 0; il < cfg.n_layers; il++) {
        uint64_t c = caps.per_layer_caps[il];
        if ((c & (CAP_WQ | CAP_WK | CAP_WV)) == (CAP_WQ | CAP_WK | CAP_WV)) {
            any_sep_qkv = true;
        }
        if (c & CAP_WQKV) any_wqkv = true;
    }
    caps.qkv_fusion_consistent = !(any_wqkv && any_sep_qkv);

    // Gated FFN consistency: either all layers have gate+up or none
    bool any_gated = false;
    bool any_ungated_ffn = false;
    for (int il = 0; il < cfg.n_layers; il++) {
        uint64_t c = caps.per_layer_caps[il];
        // Skip non-FFN layers (SSM, RWKV don't have standard FFN)
        if (c & (CAP_SSM | CAP_RWKV_TIME_MIX | CAP_DELTANET)) continue;
        bool has_gate_up = (c & CAP_FFN_GATE) && (c & CAP_FFN_UP);
        bool has_up_only = !(c & CAP_FFN_GATE) && (c & CAP_FFN_UP);
        if (has_gate_up) any_gated = true;
        if (has_up_only) any_ungated_ffn = true;
    }
    caps.gated_ffn_consistent = !(any_gated && any_ungated_ffn);

    // Bias pattern consistency: all attention layers use same bias combo
    uint64_t bias_mask = CAP_BIAS_Q | CAP_BIAS_K | CAP_BIAS_V | CAP_BIAS_O;
    uint64_t first_bias_pattern = ~0ULL;  // sentinel
    caps.bias_pattern_consistent = true;
    for (int il = 0; il < cfg.n_layers; il++) {
        uint64_t c = caps.per_layer_caps[il];
        if (c & (CAP_SSM | CAP_RWKV_TIME_MIX | CAP_DELTANET)) continue;
        uint64_t this_pattern = c & bias_mask;
        if (first_bias_pattern == ~0ULL) {
            first_bias_pattern = this_pattern;
        } else if (first_bias_pattern != this_pattern) {
            caps.bias_pattern_consistent = false;
        }
    }

    // MoE pattern: uniform (every layer) or sparse (alternating)
    caps.moe_layer_pattern = (caps.n_moe_layers == caps.n_attn_layers);

    // ========================================================================
    // Derive COMPOSE decisions
    // ========================================================================

    // Fused QKV is possible if:
    //   - All attention layers use separate Q/K/V (not wqkv)
    //   - All Q/K/V share the same weight type (slots 1, 2, 3 uniform)
    //   - No biases or scales on Q/K/V (would need extra post-steps)
    //   - No gated attention
    caps.can_fuse_qkv_globally =
        caps.qkv_fusion_consistent &&
        any_sep_qkv && !any_wqkv &&
        caps.weight_types_uniform[1] && caps.weight_types_uniform[2] && caps.weight_types_uniform[3] &&
        caps.weight_types_first[1] == caps.weight_types_first[2] &&
        caps.weight_types_first[1] == caps.weight_types_first[3] &&
        !(caps.global_caps & (CAP_BIAS_Q | CAP_BIAS_K | CAP_BIAS_V)) &&
        !(caps.global_caps & (CAP_SCALE_Q | CAP_SCALE_K | CAP_SCALE_V)) &&
        !(caps.global_caps & CAP_GATED_ATTN);

    // Fused gate+up+activation possible if:
    //   - All FFN layers have gate+up
    //   - gate and up share same weight type
    //   - No biases or scales on gate/up
    caps.can_fuse_gate_up_globally =
        caps.gated_ffn_consistent && any_gated &&
        caps.weight_types_uniform[8] && caps.weight_types_uniform[9] &&
        caps.weight_types_first[8] == caps.weight_types_first[9] &&
        !(caps.global_caps & (CAP_FFN_GATE_BIAS | CAP_FFN_UP_BIAS)) &&
        !(caps.global_caps & (CAP_FFN_GATE_SCALE | CAP_FFN_UP_SCALE));

    // Single attention kernel usable if:
    //   - All attention layers have same QK norm pattern
    //   - All layers have same SWA config (either all SWA or all full)
    caps.can_use_single_attn_kernel =
        (caps.n_ssm_layers == 0) &&
        (caps.n_rwkv_layers == 0) &&
        (caps.n_dn_layers == 0);
}

// ============================================================================
// Print capability report for debugging
// ============================================================================
static void comp_print_capabilities(const comp_capabilities & caps) {
    fprintf(stderr, "gfx1100 DETECT: capability analysis\n");
    fprintf(stderr, "  layers: %d total, %d attn, %d ssm, %d rwkv, %d dn, %d moe\n",
            caps.n_layers, caps.n_attn_layers, caps.n_ssm_layers,
            caps.n_rwkv_layers, caps.n_dn_layers, caps.n_moe_layers);

    fprintf(stderr, "  global capabilities: 0x%016llx\n",
            (unsigned long long)caps.global_caps);

    // Decode bits
    auto show = [&](uint64_t bit, const char * name) {
        if (caps.global_caps & bit) fprintf(stderr, "    %s\n", name);
    };
    show(CAP_POST_ATTN_NORM, "post_attn_norm");
    show(CAP_POST_FFN_NORM, "post_ffn_norm");
    show(CAP_QK_NORM, "qk_norm");
    show(CAP_WQKV, "fused_wqkv");
    show(CAP_BIAS_Q | CAP_BIAS_K | CAP_BIAS_V, "qkv_biases");
    show(CAP_BIAS_O, "o_bias");
    show(CAP_SCALE_Q | CAP_SCALE_K | CAP_SCALE_V, "qkv_scales (LoRA/BitNet)");
    show(CAP_GATED_ATTN, "gated_attention");
    show(CAP_FFN_GATE, "ffn_gate (gated activation)");
    show(CAP_FFN_GATE_BIAS | CAP_FFN_UP_BIAS | CAP_FFN_DOWN_BIAS, "ffn_biases");
    show(CAP_MOE_ROUTER, "moe_router");
    show(CAP_MOE_SHARED_EXPERT, "moe_shared_expert");
    show(CAP_SSM, "ssm_layers");
    show(CAP_RWKV_TIME_MIX, "rwkv_time_mix");
    show(CAP_MLA, "mla (deepseek2)");
    show(CAP_DELTANET, "deltanet");
    show(CAP_CROSS_ATTN, "cross_attention");
    show(CAP_ENCODER, "encoder_layers");
    show(CAP_REL_POS_BIAS, "relative_pos_bias");
    show(CAP_T5_GATED_FFN, "t5_gated_ffn (Flan-T5)");

    // --- Extended detection (all-tensor scan) ---
    show(CAP_ATTN_NORM_BIAS, "attn_norm_bias (LayerNorm)");
    show(CAP_FFN_NORM_BIAS,  "ffn_norm_bias (LayerNorm)");
    show(CAP_QK_NORM_BIAS,   "qk_norm_bias");
    show(CAP_ATTN_SUB_NORM,  "attn_sub_norm (BitNet)");
    show(CAP_FFN_SUB_NORM,   "ffn_sub_norm (BitNet)");
    show(CAP_MLA_Q_NORM,     "mla_q_a_norm");
    show(CAP_MLA_KV_NORM,    "mla_kv_a_norm");
    show(CAP_MLA_ABSORBED,   "mla_absorbed (wk_b/wv_b — decode path)");
    show(CAP_MLA_UNCOMPRESSED,"mla_uncompressed (wkv_b — prefill path)");
    show(CAP_BERT_POST_NORM,  "bert_post_norm");
    show(CAP_BERT_ATTN_NORM_2,"bert_attn_norm_2 (Jina-BERT-V2)");
    show(CAP_SSM_DTBC_NORM,   "ssm_dtbc_norm (FalconMamba)");
    show(CAP_SSM_MAMBA2_NORM, "ssm_mamba2_norm");
    show(CAP_VISEXP_ATTN,     "cogvlm_visexp_attn");
    show(CAP_VISEXP_FFN,      "cogvlm_visexp_ffn");
    show(CAP_RWKV_GATED_TIME_MIX, "rwkv_gated_time_mix");
    show(CAP_RWKV_FUSED_LERP,     "rwkv_fused_lerp");
    show(CAP_RWKV7,               "rwkv7_variant");

    fprintf(stderr, "  consistency: qkv_fusion=%s gated_ffn=%s bias_pattern=%s\n",
            caps.qkv_fusion_consistent ? "yes" : "no",
            caps.gated_ffn_consistent  ? "yes" : "no",
            caps.bias_pattern_consistent ? "yes" : "no");

    fprintf(stderr, "  compose decisions: can_fuse_qkv=%s can_fuse_gate_up=%s can_single_attn=%s\n",
            caps.can_fuse_qkv_globally     ? "YES" : "no",
            caps.can_fuse_gate_up_globally ? "YES" : "no",
            caps.can_use_single_attn_kernel? "YES" : "no");

    // Weight type summary
    const char * slot_names[16] = {
        "attn_norm", "wq", "wk", "wv", "q_norm", "k_norm", "wo", "ffn_norm",
        "ffn_gate", "ffn_up", "ffn_down", "?", "?", "?", "?", "?"
    };
    fprintf(stderr, "  weight types (uniform across layers): ");
    for (int s = 0; s < 11; s++) {
        if (caps.weight_types_first[s] < 0) continue;
        fprintf(stderr, "%s=%d%s ", slot_names[s], caps.weight_types_first[s],
                caps.weight_types_uniform[s] ? "" : "*");
    }
    fprintf(stderr, "\n  (* = NOT uniform, different types per layer)\n");
}
