// comp-compose.h — Composer: inspects model config → produces execution composition
//
// The composer runs ONCE at init time. It reads the model's capabilities
// and generates the optimal kernel sequence. No if/else at runtime.
//
// PoC scope: Llama-family dense models only.
#pragma once

#include "comp-types.h"
#include "../gfx1100-internal.h"

// ============================================================================
// Composer: builds a composition from model config + compiled kernels
// ============================================================================
// ============================================================================
// Composer coverage (2026-04-16):
//
// SUPPORTED (composition path is chosen when GFX1100_COMPOSITION=1):
//   - Llama family (Llama 2/3, Mistral, Qwen2/2.5) — dense, SILU, gated FFN
//   - Gemma 2/3/4 — dense, GELU_TANH, gated FFN, post-attn/post-FFN norms, iSWA
//     (per-layer RoPE theta switching handled by executor)
//
// UNSUPPORTED (composer sets plan.n_layer = -1 and dispatch falls back to
// the forward/*.cpp tree, which correctly handles these today):
//   - MoE models (Mixtral etc.)            — no MoE router/expert ops yet
//   - SSM / Mamba / Mamba2                 — hybrid layer path not modeled
//   - RWKV6 / RWKV7                        — time-mix/channel-mix not modeled
//   - DeltaNet (Qwen35)                    — not modeled
//   - DeepSeek2 MLA                        — absorbed/uncompressed not modeled
//   - T5 / BERT encoder+decoder            — cross-attention not modeled
//   - BitNet                               — sub-norm in output path not modeled
//   - CogVLM                               — visual-expert weight switching NM
//   - Phi-3.5 wqkv (fused)                 — fused QKV single-weight path NM
//   - Cohere2 skip_rope_on_global_layers   — RoPE-skip kernel not present
//   - Non-SILU / non-GELU_TANH activation  — other fused kernels absent
//   - Non-gated FFN (GPT-2, OPT, Falcon)   — ungated fused kernel absent
//
// Fallback is safe: the forward path is the tested production path.
// ============================================================================
static void compose(const gfx1100_model_config & cfg,
                    const gfx1100_compiled & k,
                    const gfx1100_buffers & b,
                    gfx1100_composition & plan) {
    memset(&plan, 0, sizeof(plan));

    // Early-bail guards for cases the composer doesn't model. Each bail
    // returns `plan.n_layer = -1` to signal fallback to forward/*.cpp.
    // Additive — never reject a model that was previously composable.
    auto bail = [&](const char * reason) {
        fprintf(stderr, "gfx1100 COMPOSE: unsupported (%s) — falling back to forward/*.cpp\n",
                reason);
        plan.n_layer = -1;
    };
    if (cfg.has_moe)                         { bail("MoE router/experts");       return; }
    if (cfg.has_ssm)                         { bail("SSM hybrid layers");        return; }
    if (cfg.has_dn)                          { bail("DeltaNet");                 return; }
    if (cfg.has_swin_norm)                   { bail("swin_norm (Chameleon)");    return; }
    if (cfg.use_par_res)                     { bail("parallel residual");        return; }
    if (cfg.skip_rope_on_global_layers)      { bail("skip_rope_on_global (Cohere2)"); return; }
    if (cfg.has_wqkv)                        { bail("fused wqkv (Phi-3)");       return; }
    // RWKV/DeltaNet/MLA/T5/BERT layers show up as non-0 layer_types entries
    for (int i = 0; i < cfg.n_layers; i++) {
        if (cfg.layer_types[i] != 0) { bail("non-attention layer type (RWKV/SSM/DN)"); return; }
    }
    // Encoder-decoder T5 models
    if (cfg.dec_n_layer > 0 || cfg.n_rel_attn_bkts > 0) {
        bail("T5 encoder-decoder"); return;
    }

    int H  = cfg.hidden_size;
    int FF = cfg.intermediate_size;
    int V  = cfg.vocab_size;
    int q_size  = cfg.fa_n_q_heads * cfg.fa_head_dim;
    int kv_size = cfg.fa_n_kv_heads * cfg.fa_head_dim;
    int n_q     = cfg.fa_n_q_heads;

    plan.n_layers    = cfg.n_layers;
    plan.hidden_size = H;
    plan.vocab_size  = V;

    // Fill buffer table
    plan.buffers[BUF_HIDDEN]       = b.hidden;
    plan.buffers[BUF_RESIDUAL]     = b.residual;
    plan.buffers[BUF_NORM_OUT]     = b.norm_out;
    plan.buffers[BUF_Q8_ACT]       = b.q8_act;
    plan.buffers[BUF_PROJ_SCRATCH] = b.proj_scratch;
    plan.buffers[BUF_MLP_INTER]    = b.mlp_inter;
    plan.buffers[BUF_ATTN_OUT]     = b.attn_out;
    plan.buffers[BUF_KV_SCRATCH]   = b.kv_scratch;
    plan.buffers[BUF_LOGITS]       = b.logits;

    // ================================================================
    // PRE-LAYER STEPS
    // ================================================================
    auto & pre = plan.pre;
    int np = 0;

    // Step 0: Embedding lookup
    {
        comp_step & s = pre[np++];
        s.op = OP_EMBED_LOOKUP;
        // Kernel and dims resolved by executor based on embed_type
    }

    // Step 1: Embed scale (Gemma only)
    if (cfg.has_embed_scale) {
        comp_step & s = pre[np++];
        s.op = OP_EMBED_SCALE;
    }

    plan.n_pre = np;

    // ================================================================
    // PER-LAYER STEPS — detected from model capabilities
    // ================================================================
    auto & layer = plan.layer;
    int nl = 0;

    // --- Phase 1: Pre-attention norm + Q8 quantize ---
    {
        comp_step & s = layer[nl++];
        s.op = OP_RMSNORM_Q8_QUANTIZE;
        s.kernel = k.eval_rmsnorm_q8_quantize;
        int norm_threads = (H < 1024) ? 256 : 1024;
        s.grid_x = 1; s.grid_y = 1; s.grid_z = 1;
        s.block_x = norm_threads; s.block_y = 1; s.block_z = 1;
        s.weight_slot = 0;  // ptrs[0] = attn_norm
        s.dim_in = H;
    }

    // --- Phase 2: QKV projection ---
    // Detect: can we use fused QKV?
    // Conditions: separate Q/K/V weights, same type, no biases/scales
    bool can_fuse_qkv = (cfg.layers[0].ptrs[1] && cfg.layers[0].ptrs[2] && cfg.layers[0].ptrs[3]
                         && cfg.layers[0].types[1] == cfg.layers[0].types[2]
                         && cfg.layers[0].types[1] == cfg.layers[0].types[3]
                         && !cfg.has_bias_q && !cfg.has_bias_k && !cfg.has_bias_v
                         && !cfg.has_scale_q && !cfg.has_scale_k && !cfg.has_scale_v
                         && !cfg.fa_has_gated_attn);

    if (can_fuse_qkv) {
        comp_step & s = layer[nl++];
        s.op = OP_FUSED_QKV;
        // Kernel resolved by executor based on weight type
        int total_rows = q_size + 2 * kv_size;
        s.grid_x = total_rows; s.grid_y = 1; s.grid_z = 1;
        s.block_x = 32; s.block_y = 4; s.block_z = 1;
        s.shared_mem = (H / 32) * 36;  // Q8_1 in shared memory
        s.weight_slot = 1;   // ptrs[1] = wq
        s.weight_slot_2 = 2; // ptrs[2] = wk
        s.weight_slot_3 = 3; // ptrs[3] = wv
        s.dim_in = H;
        s.dim_out = q_size;
        s.dim_out_2 = kv_size;
    } else {
        // 3 separate matvecs
        for (int p = 0; p < 3; p++) {
            comp_step & s = layer[nl++];
            s.op = OP_MATVEC;
            s.grid_x = (p == 0) ? q_size : kv_size;
            s.grid_y = 1; s.grid_z = 1;
            s.block_x = 32; s.block_y = 4; s.block_z = 1;
            s.weight_slot = 1 + p;  // ptrs[1]=wq, ptrs[2]=wk, ptrs[3]=wv
            s.dim_in = H;
            s.dim_out = (p == 0) ? q_size : kv_size;
        }
    }

    // --- Phase 3: RoPE + KV write ---
    {
        comp_step & s = layer[nl++];
        s.op = OP_ROPE_KV_WRITE;
        s.kernel = k.eval_qk_norm_rope_kv_write;
        s.grid_x = n_q; s.grid_y = 1; s.grid_z = 1;
        s.block_x = 32; s.block_y = 4; s.block_z = 1;
    }

    // --- Phase 4: Attention decode ---
    {
        comp_step & s = layer[nl++];
        // Auto-select kernel based on head dim
        if (k.eval_attention_decode_wmma && cfg.fa_head_dim % 16 == 0 && cfg.fa_head_dim <= 128) {
            s.op = OP_ATTN_DECODE_WMMA;
            s.kernel = k.eval_attention_decode_wmma;
            s.block_x = 32; s.block_y = 4; s.block_z = 1;
        } else if (cfg.fa_head_dim % 64 != 0 && k.eval_attention_decode_tile) {
            s.op = OP_ATTN_DECODE_TILE;
            s.kernel = k.eval_attention_decode_tile;
            s.block_x = 32; s.block_y = 1; s.block_z = 1;
        } else {
            s.op = OP_ATTN_DECODE_VEC;
            s.kernel = k.eval_attention_decode;
            s.block_x = 32; s.block_y = 4; s.block_z = 1;
        }
        s.grid_x = n_q; s.grid_y = 1; s.grid_z = 1;
    }

    // --- Phase 5: O projection + residual ---
    bool has_post_attn_norm = (cfg.layers[0].attn_post_norm != nullptr);

    if (has_post_attn_norm) {
        // quantize → matvec → post-norm+residual (3 steps)
        { comp_step & s = layer[nl++]; s.op = OP_QUANTIZE_Q8; s.dim_in = q_size; }
        { comp_step & s = layer[nl++]; s.op = OP_MATVEC; s.weight_slot = 6; s.dim_in = q_size; s.dim_out = H; }
        { comp_step & s = layer[nl++]; s.op = OP_RMSNORM_ADD; s.dim_in = H; }
    } else {
        // quantize → fused matvec+residual (2 steps)
        { comp_step & s = layer[nl++]; s.op = OP_QUANTIZE_Q8; s.dim_in = q_size; }
        { comp_step & s = layer[nl++]; s.op = OP_MATVEC_RESIDUAL; s.weight_slot = 6; s.dim_in = q_size; s.dim_out = H; }
    }

    // --- Phase 6: Pre-FFN norm + Q8 quantize ---
    // Can we fuse consecutive norms? (post-attn-norm + pre-FFN-norm)
    bool has_post_ffn_norm = (cfg.layers[0].ffn_post_norm != nullptr);
    // Consecutive norm fusion only when both post-attn AND pre-FFN norms exist
    // and the previous step was OP_RMSNORM_ADD
    // For the PoC, just use the regular fused norm+quantize
    {
        comp_step & s = layer[nl++];
        s.op = OP_RMSNORM_Q8_QUANTIZE;
        s.kernel = k.eval_rmsnorm_q8_quantize;
        int norm_threads = (H < 1024) ? 256 : 1024;
        s.grid_x = 1; s.grid_y = 1; s.grid_z = 1;
        s.block_x = norm_threads; s.block_y = 1; s.block_z = 1;
        s.weight_slot = 7;  // ptrs[7] = ffn_norm
        s.dim_in = H;
    }

    // --- Phase 7: FFN body ---
    // Detect activation type and fusion opportunity
    bool has_gate = (cfg.layers[0].ptrs[8] != nullptr);  // gate weight exists
    bool same_ffn_type = has_gate && (cfg.layers[0].types[8] == cfg.layers[0].types[9]);
    bool no_ffn_extras = (!cfg.has_bias_ffn_gate && !cfg.has_bias_ffn_up
                          && !cfg.has_scale_ffn_gate && !cfg.has_scale_ffn_up);

    if (has_gate && same_ffn_type && no_ffn_extras && cfg.act_type == 1 /* ACT_SILU */) {
        comp_step & s = layer[nl++];
        s.op = OP_FUSED_GATE_UP_SILU;
        s.weight_slot = 8;   // gate
        s.weight_slot_2 = 9; // up
        s.dim_in = H;
        s.dim_out = FF;
        s.grid_x = FF; s.grid_y = 1; s.grid_z = 1;
        s.block_x = 32; s.block_y = 4; s.block_z = 1;
        s.shared_mem = (H / 32) * 36;
    } else if (has_gate && same_ffn_type && no_ffn_extras && cfg.act_type == 3 /* ACT_GELU_TANH */) {
        comp_step & s = layer[nl++];
        s.op = OP_FUSED_GATE_UP_GELU;
        s.weight_slot = 8;
        s.weight_slot_2 = 9;
        s.dim_in = H;
        s.dim_out = FF;
        s.grid_x = FF; s.grid_y = 1; s.grid_z = 1;
        s.block_x = 32; s.block_y = 4; s.block_z = 1;
        s.shared_mem = (H / 32) * 36;
    } else {
        // Fallback: separate gate + up + activation (3 steps)
        // Not implementing for PoC — would need OP_SILU_MUL etc.
        // For now, the composer rejects and falls back to if/else path
        plan.n_layer = -1;  // signal: composition not available
        return;
    }

    // --- Phase 8: Down projection + residual ---
    if (has_post_ffn_norm) {
        { comp_step & s = layer[nl++]; s.op = OP_QUANTIZE_Q8; s.dim_in = FF; }
        { comp_step & s = layer[nl++]; s.op = OP_MATVEC; s.weight_slot = 10; s.dim_in = FF; s.dim_out = H; }
        { comp_step & s = layer[nl++]; s.op = OP_RMSNORM_ADD; s.dim_in = H; }
    } else {
        { comp_step & s = layer[nl++]; s.op = OP_QUANTIZE_Q8; s.dim_in = FF; }
        { comp_step & s = layer[nl++]; s.op = OP_MATVEC_RESIDUAL; s.weight_slot = 10; s.dim_in = FF; s.dim_out = H; }
    }

    plan.n_layer = nl;

    // ================================================================
    // POST-LAYER STEPS
    // ================================================================
    auto & post = plan.post;
    int npost = 0;

    // Final norm
    { comp_step & s = post[npost++]; s.op = OP_FINAL_NORM; }
    // Quantize for LM head
    { comp_step & s = post[npost++]; s.op = OP_QUANTIZE_Q8; s.dim_in = H; }
    // LM head
    { comp_step & s = post[npost++]; s.op = OP_LM_HEAD; }
    // Final logit softcap (Gemma only)
    if (cfg.has_final_logit_softcap && cfg.final_logit_softcap_val > 0) {
        comp_step & s = post[npost++];
        s.op = OP_SOFTCAP;
    }

    plan.n_post = npost;
}

// ============================================================================
// Print composition — for debugging and verification
// ============================================================================
static const char * comp_op_name(comp_op op) {
    switch (op) {
        case OP_RMSNORM_Q8_QUANTIZE:    return "rmsnorm_q8_quantize";
        case OP_RMSNORM_ADD:            return "rmsnorm_add";
        case OP_RMSNORM_ADD_RMSNORM_Q8: return "rmsnorm_add_rmsnorm_q8";
        case OP_MATVEC:                 return "matvec";
        case OP_MATVEC_RESIDUAL:        return "matvec_residual";
        case OP_FUSED_QKV:              return "fused_qkv";
        case OP_FUSED_GATE_UP_SILU:     return "fused_gate_up_silu";
        case OP_FUSED_GATE_UP_GELU:     return "fused_gate_up_gelu";
        case OP_QUANTIZE_Q8:            return "quantize_q8";
        case OP_ROPE_KV_WRITE:          return "rope_kv_write";
        case OP_ATTN_DECODE_VEC:        return "attn_decode_vec";
        case OP_ATTN_DECODE_TILE:       return "attn_decode_tile";
        case OP_ATTN_DECODE_WMMA:       return "attn_decode_wmma";
        case OP_EMBED_LOOKUP:           return "embed_lookup";
        case OP_EMBED_SCALE:            return "embed_scale";
        case OP_FINAL_NORM:             return "final_norm";
        case OP_LM_HEAD:                return "lm_head";
        case OP_SOFTCAP:                return "softcap";
        case OP_NOP:                    return "nop";
        default:                        return "???";
    }
}

static void print_composition(const gfx1100_composition & plan) {
    fprintf(stderr, "gfx1100 COMPOSITION (%d layers):\n", plan.n_layers);
    fprintf(stderr, "  PRE (%d steps):\n", plan.n_pre);
    for (int i = 0; i < plan.n_pre; i++)
        fprintf(stderr, "    [%d] %s\n", i, comp_op_name(plan.pre[i].op));
    fprintf(stderr, "  LAYER (%d steps):\n", plan.n_layer);
    for (int i = 0; i < plan.n_layer; i++)
        fprintf(stderr, "    [%d] %s (w=%d, %dx%dx%d, %dx%dx%d)\n", i,
                comp_op_name(plan.layer[i].op), plan.layer[i].weight_slot,
                plan.layer[i].grid_x, plan.layer[i].grid_y, plan.layer[i].grid_z,
                plan.layer[i].block_x, plan.layer[i].block_y, plan.layer[i].block_z);
    fprintf(stderr, "  POST (%d steps):\n", plan.n_post);
    for (int i = 0; i < plan.n_post; i++)
        fprintf(stderr, "    [%d] %s\n", i, comp_op_name(plan.post[i].op));
    int total = plan.n_pre + plan.n_layer * plan.n_layers + plan.n_post;
    fprintf(stderr, "  TOTAL: %d launches per token\n", total);
}
