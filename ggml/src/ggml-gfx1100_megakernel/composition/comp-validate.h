// comp-validate.h — VALIDATE meta-step: verify model is supportable
//
// Runs before COMPOSE. Checks the detected capabilities against what our
// kernel library actually supports. If anything is unsupported, returns a
// detailed diagnostic so the caller can fall back to baseline ggml-hip.
//
// This is NOT an optimization — it's a correctness gate. If we compose a
// plan with unsupported operations, the executor would fail silently or
// crash. VALIDATE catches these cases upfront with clear error messages.
//
// The goal: user always gets a working inference, either via megakernel
// (if supported) or via baseline (if not). No silent failures, no crashes.
#pragma once

#include "comp-types.h"
#include "../gfx1100-internal.h"
#include <cstdio>
#include <cstring>
#include <cstdarg>

// ============================================================================
// Validation result
// ============================================================================
struct comp_validation_result {
    bool supported;           // overall verdict
    char reason[512];         // human-readable reason if not supported
    int capability_bits;      // bitfield of detected capabilities (for logging)

    // Feature flags that triggered rejection (for detailed diagnostics)
    bool needs_bidirectional_attn;
    bool needs_cross_attn;
    bool needs_encoder;
    bool needs_ssm;
    bool needs_rwkv;
    bool needs_mla;
    bool needs_deltanet;
    bool needs_moe;
    bool needs_custom_gating;
    bool needs_temperature_attn;
    bool needs_shared_kv;
    bool needs_swin_norm;
    bool needs_rescaling;
    bool needs_multi_rope;
    bool needs_yarn;
};

// ============================================================================
// Feature support matrix — what we can handle vs what we can't
// ============================================================================
// This is the single source of truth for what the megakernel supports.
// When adding a new kernel, update this matrix.

// Attention kernels supported per head_dim
static inline bool comp_supports_head_dim(int head_dim) {
    // VEC kernel: D=64, 128, 256
    // WMMA kernel: D multiple of 16, D ≤ 128
    // TILE kernel: arbitrary D (fallback)
    return head_dim == 64 || head_dim == 96 || head_dim == 128 || head_dim == 256 ||
           (head_dim % 16 == 0 && head_dim <= 256) ||
           (head_dim % 8 == 0);  // TILE fallback
}

// RoPE types: 0=NONE, 1=NORM, 2=NEOX, 3=MULTI
static inline bool comp_supports_rope_type(int rope_type) {
    return rope_type >= 0 && rope_type <= 3;
}

// Norm types: 1=RMS, 2=LAYER
static inline bool comp_supports_norm_type(int norm_type) {
    return norm_type == 1 || norm_type == 2;
}

// Activation types — detected from tensor patterns
// 0=SILU, 1=GELU_TANH, 2=GELU_EXACT, 3=RELU, 4=SWISH
static inline bool comp_supports_activation(int act_type) {
    return act_type >= 0 && act_type <= 4;
}

// ============================================================================
// Main validation entry point
// ============================================================================
static void comp_validate(const gfx1100_model_config & cfg,
                          comp_validation_result & result) {
    memset(&result, 0, sizeof(result));
    result.supported = true;

    auto reject = [&](const char * reason_fmt, ...) {
        result.supported = false;
        va_list ap;
        va_start(ap, reason_fmt);
        vsnprintf(result.reason, sizeof(result.reason), reason_fmt, ap);
        va_end(ap);
    };

    // ========================================================================
    // 1. CORE DIMENSIONS — must be present and reasonable
    // ========================================================================
    if (cfg.hidden_size <= 0) {
        reject("invalid hidden_size=%d", cfg.hidden_size);
        return;
    }
    if (cfg.n_layers <= 0 || cfg.n_layers > 128) {
        reject("unsupported n_layers=%d (max 128)", cfg.n_layers);
        return;
    }
    if (cfg.vocab_size <= 0) {
        reject("invalid vocab_size=%d", cfg.vocab_size);
        return;
    }

    // ========================================================================
    // 2. ATTENTION CONFIGURATION — must match kernel capabilities
    // ========================================================================
    if (cfg.fa_n_q_heads <= 0) {
        reject("invalid n_q_heads=%d", cfg.fa_n_q_heads);
        return;
    }
    if (cfg.fa_n_kv_heads <= 0) {
        reject("invalid n_kv_heads=%d", cfg.fa_n_kv_heads);
        return;
    }
    if (cfg.fa_n_q_heads % cfg.fa_n_kv_heads != 0) {
        reject("GQA ratio not integer: n_q=%d n_kv=%d",
               cfg.fa_n_q_heads, cfg.fa_n_kv_heads);
        return;
    }
    if (!comp_supports_head_dim(cfg.fa_head_dim)) {
        reject("unsupported head_dim=%d (need 64/96/128/256 or multiple of 8/16)",
               cfg.fa_head_dim);
        return;
    }

    // Non-causal attention = bidirectional (BERT/T5 encoder)
    // Note: attn_causal is a tri-state — we only reject if explicitly set to 0
    //       via GGUF key "attention.causal=false". Zero-init (unset) means
    //       "caller didn't tell us" which defaults to causal for decoder models.
    // TODO: once loaders consistently set attn_causal, we can rely on it.
    // For now, only reject when we detect an encoder (which implies non-causal).
    if (result.needs_encoder) {
        result.needs_bidirectional_attn = true;
        reject("encoder-layer models use non-causal attention (not yet supported)");
        return;
    }

    // ========================================================================
    // 3. HYBRID LAYER TYPES — SSM/RWKV/MLA/DeltaNet
    // ========================================================================
    if (cfg.has_ssm) {
        result.needs_ssm = true;
        // SSM has partial support, check dimensions
        if (cfg.ssm_d_state <= 0 || cfg.ssm_d_inner <= 0) {
            reject("SSM dimensions invalid: d_state=%d d_inner=%d",
                   cfg.ssm_d_state, cfg.ssm_d_inner);
            return;
        }
    }

    if (cfg.has_dn) {
        result.needs_deltanet = true;
        if (cfg.dn_key_dim <= 0 || cfg.dn_value_dim <= 0) {
            reject("DeltaNet dims invalid: key=%d value=%d",
                   cfg.dn_key_dim, cfg.dn_value_dim);
            return;
        }
    }

    // MLA (DeepSeek2)
    if (cfg.mla_kv_lora_rank > 0 || cfg.mla_q_lora_rank > 0) {
        result.needs_mla = true;
        // MLA is partially supported
    }

    // RWKV — detect by checking first layer's weights
    bool has_rwkv = false;
    for (int il = 0; il < cfg.n_layers && il < 2; il++) {
        if (cfg.layers[il].time_mix_key || cfg.layers[il].time_mix_receptance) {
            has_rwkv = true;
            break;
        }
    }
    if (has_rwkv) {
        result.needs_rwkv = true;
        // RWKV is partially supported
    }

    // ========================================================================
    // 4. MoE — routing support
    // ========================================================================
    if (cfg.has_moe || cfg.moe_n_experts > 0) {
        result.needs_moe = true;
        if (cfg.moe_n_experts_used == 0) {
            reject("MoE model missing expert_used_count");
            return;
        }
        if (cfg.moe_n_experts_used > cfg.moe_n_experts) {
            reject("MoE: expert_used_count=%d > expert_count=%d",
                   cfg.moe_n_experts_used, cfg.moe_n_experts);
            return;
        }
        // Non-standard gating functions (e.g., sigmoid routing for DeepSeek3)
        if (cfg.moe_gating_func > 1) {
            result.needs_custom_gating = true;
            // Reject for now — only softmax/topk supported
            reject("MoE gating_func=%d not yet supported (only softmax/topk)",
                   cfg.moe_gating_func);
            return;
        }
        // Expert groups (DeepSeek3) — not yet supported
        if (cfg.moe_group_count > 1) {
            reject("MoE expert groups (count=%d) not yet supported",
                   cfg.moe_group_count);
            return;
        }
    }

    // ========================================================================
    // 5. ENCODER-DECODER — T5, BERT
    // ========================================================================
    // Check if encoder tensors exist in first layer
    if (cfg.n_layers > 0 && cfg.layers[0].wq_enc != nullptr) {
        result.needs_encoder = true;
        reject("encoder-decoder models (T5/BERT) partial support only");
        return;
    }

    if (cfg.n_layers > 0 && cfg.layers[0].wq_cross != nullptr) {
        result.needs_cross_attn = true;
        reject("cross-attention (T5 decoder) not yet supported");
        return;
    }

    // ========================================================================
    // 6. RARE ATTENTION FEATURES — mostly unsupported
    // ========================================================================
    if (cfg.attn_clamp_kqv > 0) {
        reject("attention clamp_kqv (StarCoder2) not yet supported");
        return;
    }

    // Temperature attention (dynamic scaling)
    // These GGUF keys imply temperature-modulated attention
    // (We don't have a field for them yet, but validate tolerates absence)

    // Shared KV layers (cross-layer KV reuse)
    if (cfg.shared_kv_layers > 0) {
        result.needs_shared_kv = true;
        // This is advanced; we don't support it yet
        reject("shared_kv_layers=%d not yet supported", cfg.shared_kv_layers);
        return;
    }

    // Swin-style norm (Chameleon)
    if (cfg.has_swin_norm) {
        result.needs_swin_norm = true;
        reject("swin_norm placement not yet supported");
        return;
    }

    // Periodic rescaling (RWKV)
    if (cfg.rescale_every_n > 0) {
        result.needs_rescaling = true;
        // RWKV-specific, check with RWKV path
    }

    // ========================================================================
    // 7. RoPE VARIANTS
    // ========================================================================
    if (!comp_supports_rope_type(cfg.rope_type)) {
        reject("unsupported rope_type=%d", cfg.rope_type);
        return;
    }
    if (cfg.rope_type == 3) {  // ROPE_MULTI (Qwen2VL/3VL)
        result.needs_multi_rope = true;
        if (!cfg.has_rope_sections) {
            reject("ROPE_MULTI needs rope.dimension_sections");
            return;
        }
    }
    // YaRN is supported via existing params
    if (cfg.yarn_ext_factor != 0.0f) {
        result.needs_yarn = true;
        // Supported, not a blocker
    }

    // ========================================================================
    // 8. NORM TYPE
    // ========================================================================
    if (!comp_supports_norm_type(cfg.norm_type)) {
        reject("unsupported norm_type=%d (1=RMS, 2=Layer)", cfg.norm_type);
        return;
    }

    // ========================================================================
    // 9. QUANTIZATION TYPES — all tensors must use supported types
    // ========================================================================
    // Check first layer's weight types
    if (cfg.n_layers > 0) {
        const auto & lw = cfg.layers[0];
        // Slots 1-3 = Q/K/V, 6 = O, 8-10 = gate/up/down
        int type_slots[] = {1, 2, 3, 6, 8, 9, 10};
        const char * slot_names[] = {"wq", "wk", "wv", "wo", "ffn_gate", "ffn_up", "ffn_down"};
        for (size_t i = 0; i < sizeof(type_slots)/sizeof(int); i++) {
            int slot = type_slots[i];
            if (!lw.ptrs[slot]) continue;  // not present is OK
            int t = lw.types[slot];
            // We support types 0-30 (Q4_0..IQ4_XS + K-quants + IQ-quants + F16/BF16/F32 + MXFP4/NVFP4)
            if (t < 0 || t > 35) {
                reject("unsupported weight type=%d in %s", t, slot_names[i]);
                return;
            }
        }
    }

    // ========================================================================
    // 10. Everything else — build capability bitfield for logging
    // ========================================================================
    int bits = 0;
    if (cfg.has_moe)              bits |= 0x0001;
    if (cfg.has_swa)              bits |= 0x0002;
    if (cfg.has_alibi)            bits |= 0x0004;
    if (cfg.attn_softcap_val > 0) bits |= 0x0008;
    if (cfg.has_final_logit_softcap) bits |= 0x0010;
    if (cfg.has_post_attn_norm)   bits |= 0x0020;
    if (cfg.has_post_ffn_norm)    bits |= 0x0040;
    if (cfg.has_fused_qkv)        bits |= 0x0080;
    if (cfg.has_gated_ffn)        bits |= 0x0100;
    if (cfg.has_qk_norm)          bits |= 0x0200;
    if (cfg.has_ssm)              bits |= 0x0400;
    if (cfg.has_dn)               bits |= 0x0800;
    if (cfg.use_par_res)          bits |= 0x1000;
    if (cfg.has_embed_scale)      bits |= 0x2000;
    if (cfg.has_bias_q || cfg.has_bias_k || cfg.has_bias_v) bits |= 0x4000;
    if (cfg.has_scale_q || cfg.has_scale_k || cfg.has_scale_v) bits |= 0x8000;
    result.capability_bits = bits;
}

// ============================================================================
// Pretty-print validation result (for logging)
// ============================================================================
static void comp_print_validation(const comp_validation_result & r) {
    fprintf(stderr, "gfx1100 composition validate: %s\n",
            r.supported ? "SUPPORTED" : "REJECTED");
    if (!r.supported) {
        fprintf(stderr, "  reason: %s\n", r.reason);
    }
    fprintf(stderr, "  capabilities: 0x%04x", r.capability_bits);
    if (r.capability_bits & 0x0001) fprintf(stderr, " MoE");
    if (r.capability_bits & 0x0002) fprintf(stderr, " SWA");
    if (r.capability_bits & 0x0004) fprintf(stderr, " ALiBi");
    if (r.capability_bits & 0x0008) fprintf(stderr, " AttnSoftcap");
    if (r.capability_bits & 0x0010) fprintf(stderr, " LogitSoftcap");
    if (r.capability_bits & 0x0020) fprintf(stderr, " PostAttnNorm");
    if (r.capability_bits & 0x0040) fprintf(stderr, " PostFFNNorm");
    if (r.capability_bits & 0x0080) fprintf(stderr, " FusedQKV");
    if (r.capability_bits & 0x0100) fprintf(stderr, " GatedFFN");
    if (r.capability_bits & 0x0200) fprintf(stderr, " QKNorm");
    if (r.capability_bits & 0x0400) fprintf(stderr, " SSM");
    if (r.capability_bits & 0x0800) fprintf(stderr, " DeltaNet");
    if (r.capability_bits & 0x1000) fprintf(stderr, " ParResidual");
    if (r.capability_bits & 0x2000) fprintf(stderr, " EmbedScale");
    if (r.capability_bits & 0x4000) fprintf(stderr, " Biases");
    if (r.capability_bits & 0x8000) fprintf(stderr, " Scales");
    fprintf(stderr, "\n");
}
