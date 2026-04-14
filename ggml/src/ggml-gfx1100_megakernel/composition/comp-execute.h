// comp-execute.h — Executor: runs a composed composition as a flat kernel launch loop
//
// The executor takes a composition and launches kernels sequentially.
// No if/else branching — the composition IS the dispatch table.
//
// For hipGraph integration: the executor's loop is what gets captured.
// Capture once, replay forever.
//
// PoC scope: Llama-family dense decode only.
// Does NOT handle: MoE, SSM, RWKV, DeltaNet, T5, BERT, parallel attn+FFN,
//                  swin_norm, Cohere shared norm, gated attention.
//                  These fall back to the if/else forward path.
#pragma once

#include "comp-types.h"
#include "../gfx1100-internal.h"

// ============================================================================
// Buffer context — resolved once per call, used by all steps
// ============================================================================
struct comp_exec_ctx {
    hipStream_t s;

    // GPU buffers (don't change between tokens)
    float * hidden;
    float * residual;
    float * norm_out;
    void  * q8_act;
    float * proj_scratch;
    float * mlp_inter;
    float * attn_out;
    float * kv_scratch;
    float * logits;
    int   * batch_token_ids;

    // Model config (constant after init)
    const gfx1100_model_config * cfg;
    const gfx1100_compiled     * k;

    // Per-token values
    int token_id;
    int position;
    int kv_len;
};

// ============================================================================
// Kernel picker helpers — same logic as llama-family.cpp but standalone
// ============================================================================

static hipFunction_t comp_pick_embed(const gfx1100_compiled & k, int type) {
    switch (type) {
        case 10: return k.eval_embed_q2k;
        case 11: return k.eval_embed_q3k;
        case 12: return k.eval_embed_q4k;
        case 13: return k.eval_embed_q5k;
        case 14: return k.eval_embed_q6k;
        case  8: return k.eval_embed_q8_0;
        case  2: return k.eval_embed_q4_0;
        case  3: return k.eval_embed_q4_1;
        case  6: return k.eval_embed_q5_0;
        case  7: return k.eval_embed_q5_1;
        default: return k.eval_embed_q4k;
    }
}

static int comp_embed_blocks(int H, int type) {
    switch (type) {
        case 10: case 11: case 12: case 13: case 14:
            return H / 256;  // K-quant: QK_K=256
        default:
            return H / 32;   // small block: QK=32
    }
}

static int comp_embed_threads(int type) {
    switch (type) {
        case 10: case 13: case 14: return 64;  // Q2K, Q5K, Q6K
        case 12: return 32;                      // Q4K
        default: return 32;
    }
}

static bool comp_is_float(int type) {
    return type == 0 || type == 1 || type == 30;
}

static hipFunction_t comp_pick_matvec(const gfx1100_compiled & k, int type) {
    switch (type) {
        case  0: return k.eval_matvec_f32;
        case  1: return k.eval_matvec_f16;
        case 30: return k.eval_matvec_bf16;
        case  2: return k.eval_matvec_q4_0;
        case  3: return k.eval_matvec_q4_1;
        case  6: return k.eval_matvec_q5_0;
        case  7: return k.eval_matvec_q5_1;
        case  8: return k.eval_matvec_q8_0;
        case 10: return k.eval_matvec_q2k;
        case 11: return k.eval_matvec_q3k;
        case 12: return k.eval_matvec_q4k;
        case 13: return k.eval_matvec_q5k;
        case 14: return k.eval_matvec_q6k;
        default: return k.eval_matvec_q4k;
    }
}

static hipFunction_t comp_pick_matvec_res(const gfx1100_compiled & k, int type) {
    switch (type) {
        case  0: return k.eval_matvec_f32_residual;
        case  1: return k.eval_matvec_f16_residual;
        case 30: return k.eval_matvec_bf16_residual;
        case  2: return k.eval_matvec_q4_0_residual;
        case  3: return k.eval_matvec_q4_1_residual;
        case  6: return k.eval_matvec_q5_0_residual;
        case  7: return k.eval_matvec_q5_1_residual;
        case  8: return k.eval_matvec_q8_0_residual;
        case 10: return k.eval_matvec_q2k_residual;
        case 11: return k.eval_matvec_q3k_residual;
        case 12: return k.eval_matvec_q4k_residual;
        case 13: return k.eval_matvec_q5k_residual;
        case 14: return k.eval_matvec_q6k_residual;
        default: return k.eval_matvec_q4k_residual;
    }
}

static hipFunction_t comp_pick_fused_qkv(const gfx1100_compiled & k, int type) {
    switch (type) {
        case  2: return k.eval_fused_qkv_matvec_q4_0;
        case  3: return k.eval_fused_qkv_matvec_q4_1;
        case  6: return k.eval_fused_qkv_matvec_q5_0;
        case  7: return k.eval_fused_qkv_matvec_q5_1;
        case  8: return k.eval_fused_qkv_matvec_q8_0;
        case 10: return k.eval_fused_qkv_matvec_q2k;
        case 11: return k.eval_fused_qkv_matvec_q3k;
        case 12: return k.eval_fused_qkv_matvec_q4k;
        case 13: return k.eval_fused_qkv_matvec_q5k;
        case 14: return k.eval_fused_qkv_matvec_q6k;
        default: return nullptr;
    }
}

static hipFunction_t comp_pick_fused_gate_up_silu(const gfx1100_compiled & k, int type) {
    switch (type) {
        case 12: return k.eval_fused_gate_up_silu_q4k;
        case 14: return k.eval_fused_gate_up_silu_q6k;
        case  2: return k.eval_fused_gate_up_silu_q4_0;
        case  3: return k.eval_fused_gate_up_silu_q4_1;
        case  6: return k.eval_fused_gate_up_silu_q5_0;
        case  7: return k.eval_fused_gate_up_silu_q5_1;
        case  8: return k.eval_fused_gate_up_silu_q8_0;
        case 10: return k.eval_fused_gate_up_silu_q2k;
        case 11: return k.eval_fused_gate_up_silu_q3k;
        case 13: return k.eval_fused_gate_up_silu_q5k;
        default: return nullptr;
    }
}

static hipFunction_t comp_pick_fused_gate_up_gelu(const gfx1100_compiled & k, int type) {
    switch (type) {
        case 12: return k.eval_fused_gate_up_gelu_q4k;
        case 14: return k.eval_fused_gate_up_gelu_q6k;
        case  2: return k.eval_fused_gate_up_gelu_q4_0;
        case  8: return k.eval_fused_gate_up_gelu_q8_0;
        default: return nullptr;
    }
}

// ============================================================================
// Step executors — one function per operation type
// ============================================================================

static void exec_rmsnorm_q8_quantize(const comp_step & step, int il, comp_exec_ctx & ctx) {
    const void * w = ctx.cfg->layers[il].ptrs[step.weight_slot];
    int n = ctx.cfg->hidden_size;
    void * args[] = { (void *)&ctx.hidden, (void *)&w, (void *)&ctx.norm_out,
                      (void *)&ctx.residual, (void *)&ctx.q8_act, (void *)&n };
    hipModuleLaunchKernel(step.kernel, 1, 1, 1,
                         step.block_x, 1, 1, 0, ctx.s, args, nullptr);
}

static void exec_matvec(const comp_step & step, int il, comp_exec_ctx & ctx,
                        float * output) {
    int type = ctx.cfg->layers[il].types[step.weight_slot];
    hipFunction_t fn = comp_pick_matvec(*ctx.k, type);
    const void * w = ctx.cfg->layers[il].ptrs[step.weight_slot];
    long long st = ctx.cfg->layers[il].strides[step.weight_slot];
    bool is_f = comp_is_float(type);
    const void * input = is_f ? (const void *)ctx.norm_out : (const void *)ctx.q8_act;
    int in_dim = step.dim_in, out_dim = step.dim_out;
    void * args[] = { (void *)&w, (void *)&st, (void *)&input,
                      (void *)&output, (void *)&in_dim, (void *)&out_dim };
    if (is_f)
        hipModuleLaunchKernel(fn, out_dim, 1, 1, 256, 1, 1, 0, ctx.s, args, nullptr);
    else
        hipModuleLaunchKernel(fn, out_dim, 1, 1, 32, 4, 1, 0, ctx.s, args, nullptr);
}

static void exec_matvec_residual(const comp_step & step, int il, comp_exec_ctx & ctx) {
    int type = ctx.cfg->layers[il].types[step.weight_slot];
    hipFunction_t fn = comp_pick_matvec_res(*ctx.k, type);
    const void * w = ctx.cfg->layers[il].ptrs[step.weight_slot];
    long long st = ctx.cfg->layers[il].strides[step.weight_slot];
    bool is_f = comp_is_float(type);
    const void * input = is_f ? (const void *)ctx.norm_out : (const void *)ctx.q8_act;
    int in_dim = step.dim_in, out_dim = step.dim_out;
    void * args[] = { (void *)&w, (void *)&st, (void *)&input,
                      (void *)&ctx.residual, (void *)&ctx.hidden,
                      (void *)&in_dim, (void *)&out_dim };
    if (is_f)
        hipModuleLaunchKernel(fn, out_dim, 1, 1, 256, 1, 1, 0, ctx.s, args, nullptr);
    else
        hipModuleLaunchKernel(fn, out_dim, 1, 1, 32, 4, 1, 0, ctx.s, args, nullptr);
}

static void exec_fused_qkv(const comp_step & step, int il, comp_exec_ctx & ctx) {
    int type = ctx.cfg->layers[il].types[step.weight_slot];
    hipFunction_t fn = comp_pick_fused_qkv(*ctx.k, type);
    if (!fn) {
        // Fallback: 3 separate matvecs
        comp_step q_step = step; q_step.dim_out = step.dim_out; q_step.op = OP_MATVEC;
        exec_matvec(q_step, il, ctx, ctx.proj_scratch);
        comp_step k_step = step; k_step.weight_slot = step.weight_slot_2;
        k_step.dim_out = step.dim_out_2; k_step.op = OP_MATVEC;
        exec_matvec(k_step, il, ctx, ctx.kv_scratch);
        comp_step v_step = step; v_step.weight_slot = step.weight_slot_3;
        v_step.dim_out = step.dim_out_2; v_step.op = OP_MATVEC;
        exec_matvec(v_step, il, ctx, ctx.kv_scratch + step.dim_out_2);
        return;
    }
    const void * wq = ctx.cfg->layers[il].ptrs[step.weight_slot];
    long long sq = ctx.cfg->layers[il].strides[step.weight_slot];
    const void * wk = ctx.cfg->layers[il].ptrs[step.weight_slot_2];
    long long sk = ctx.cfg->layers[il].strides[step.weight_slot_2];
    const void * wv = ctx.cfg->layers[il].ptrs[step.weight_slot_3];
    long long sv = ctx.cfg->layers[il].strides[step.weight_slot_3];
    const void * input = (const void *)ctx.q8_act;
    int q_out = step.dim_out;
    int kv_out = step.dim_out_2;
    int total = q_out + 2 * kv_out;
    int in_dim = step.dim_in;
    void * args[] = { (void *)&wq, (void *)&sq, (void *)&wk, (void *)&sk,
                      (void *)&wv, (void *)&sv, (void *)&input,
                      (void *)&ctx.proj_scratch, (void *)&ctx.kv_scratch,
                      (void *)&in_dim, (void *)&q_out, (void *)&kv_out };
    hipModuleLaunchKernel(fn, total, 1, 1, 32, 4, 1, step.shared_mem, ctx.s, args, nullptr);
}

static void exec_quantize_q8(const comp_step & step, int il, comp_exec_ctx & ctx,
                             float * input) {
    int n = step.dim_in;
    int blocks = (n + 511) / 512;
    void * args[] = { (void *)&input, (void *)&ctx.q8_act, (void *)&n };
    hipModuleLaunchKernel(ctx.k->eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, ctx.s, args, nullptr);
}

static void exec_rope_kv_write(const comp_step & step, int il, comp_exec_ctx & ctx) {
    // Count attention layers before this one to get the right KV cache slot
    int attn_idx = 0;
    for (int i = 0; i < il; i++)
        if (ctx.cfg->layer_types[i] == 0) attn_idx++;

    float theta = ctx.cfg->fa_rope_theta;
    if (ctx.cfg->fa_rope_theta_swa > 0 && ctx.cfg->layer_use_swa[il])
        theta = ctx.cfg->fa_rope_theta_swa;
    const void * freq_factors = ctx.cfg->rope_freq_factors_per_layer[il]
                                ? ctx.cfg->rope_freq_factors_per_layer[il]
                                : ctx.cfg->rope_freq_factors;
    const void * q_norm_w = ctx.cfg->layers[il].ptrs[4];
    const void * k_norm_w = ctx.cfg->layers[il].ptrs[5];
    void * args[] = {
        (void *)&ctx.proj_scratch, (void *)&ctx.kv_scratch,
        (void *)&ctx.attn_out,
        (void *)&ctx.cfg->k_cache_ptrs[attn_idx], (void *)&ctx.cfg->v_cache_ptrs[attn_idx],
        (void *)&ctx.position, (void *)&ctx.cfg->max_seq_len,
        (void *)&theta, (void *)&freq_factors,
        (void *)&q_norm_w, (void *)&k_norm_w
    };
    hipModuleLaunchKernel(step.kernel, step.grid_x, 1, 1,
                         step.block_x, step.block_y, 1, 0, ctx.s, args, nullptr);
}

static void exec_attention(const comp_step & step, int il, comp_exec_ctx & ctx) {
    int attn_idx = 0;
    for (int i = 0; i < il; i++)
        if (ctx.cfg->layer_types[i] == 0) attn_idx++;

    float alibi_mb = ctx.cfg->alibi_max_bias;
    float alibi_m0 = ctx.cfg->alibi_m0;
    float alibi_m1 = ctx.cfg->alibi_m1;
    int   alibi_nhl = ctx.cfg->alibi_n_head_log2;
    float softcap = ctx.cfg->attn_softcap_val;
    const float * rel_bias = nullptr;
    void * args[] = {
        (void *)&ctx.proj_scratch,
        (void *)&ctx.cfg->k_cache_ptrs[attn_idx], (void *)&ctx.cfg->v_cache_ptrs[attn_idx],
        (void *)&ctx.attn_out,
        (void *)&ctx.kv_len, (void *)&ctx.cfg->max_seq_len,
        (void *)&alibi_mb, (void *)&alibi_m0, (void *)&alibi_m1,
        (void *)&alibi_nhl, (void *)&ctx.position,
        (void *)&rel_bias, (void *)&softcap
    };
    hipModuleLaunchKernel(step.kernel, step.grid_x, 1, 1,
                         step.block_x, step.block_y, 1, 0, ctx.s, args, nullptr);
}

static void exec_rmsnorm_add(const comp_step & step, int il, comp_exec_ctx & ctx) {
    // Determine which post-norm weight to use
    const void * post_w = ctx.cfg->layers[il].attn_post_norm;
    if (!post_w) post_w = ctx.cfg->layers[il].ffn_post_norm;
    int n = ctx.cfg->hidden_size;
    int norm_threads = (n < 1024) ? 256 : 1024;
    void * args[] = { (void *)&ctx.hidden, (void *)&post_w,
                      (void *)&ctx.residual, (void *)&ctx.hidden, (void *)&n };
    hipModuleLaunchKernel(ctx.k->eval_rmsnorm_add, 1, 1, 1,
                         norm_threads, 1, 1, 0, ctx.s, args, nullptr);
}

static void exec_fused_gate_up(const comp_step & step, int il, comp_exec_ctx & ctx) {
    int type = ctx.cfg->layers[il].types[step.weight_slot];
    hipFunction_t fn = (step.op == OP_FUSED_GATE_UP_SILU)
                       ? comp_pick_fused_gate_up_silu(*ctx.k, type)
                       : comp_pick_fused_gate_up_gelu(*ctx.k, type);
    if (!fn) return;  // shouldn't happen if composer did its job
    const void * gw = ctx.cfg->layers[il].ptrs[step.weight_slot];
    long long gs = ctx.cfg->layers[il].strides[step.weight_slot];
    const void * uw = ctx.cfg->layers[il].ptrs[step.weight_slot_2];
    long long us = ctx.cfg->layers[il].strides[step.weight_slot_2];
    bool is_f = comp_is_float(type);
    const void * input = is_f ? (const void *)ctx.norm_out : (const void *)ctx.q8_act;
    int in_dim = step.dim_in, out_dim = step.dim_out;
    void * args[] = { (void *)&gw, (void *)&gs, (void *)&uw, (void *)&us,
                      (void *)&input, (void *)&ctx.mlp_inter,
                      (void *)&in_dim, (void *)&out_dim };
    hipModuleLaunchKernel(fn, out_dim, 1, 1, 32, 4, 1, step.shared_mem, ctx.s, args, nullptr);
}

// ============================================================================
// Main executor: runs the full composition for one decode token
// ============================================================================
static int execute_composition(const gfx1100_composition & plan,
                        const gfx1100_model_config & cfg,
                        const gfx1100_compiled & k,
                        gfx1100_buffers & b,
                        int token_id, int position, float * logits_out) {

    int H  = cfg.hidden_size;
    int FF = cfg.intermediate_size;
    int V  = cfg.vocab_size;
    int q_size  = cfg.fa_n_q_heads * cfg.fa_head_dim;

    // Build execution context
    comp_exec_ctx ctx;
    ctx.s = b.stream;
    ctx.hidden = b.hidden;
    ctx.residual = b.residual;
    ctx.norm_out = b.norm_out;
    ctx.q8_act = b.q8_act;
    ctx.proj_scratch = b.proj_scratch;
    ctx.mlp_inter = b.mlp_inter;
    ctx.attn_out = b.attn_out;
    ctx.kv_scratch = b.kv_scratch;
    ctx.logits = b.logits;
    ctx.batch_token_ids = b.batch_token_ids;
    ctx.cfg = &cfg;
    ctx.k = &k;
    ctx.token_id = token_id;
    ctx.position = position;
    ctx.kv_len = position + 1;

    // ================================================================
    // PRE-LAYER
    // ================================================================
    for (int i = 0; i < plan.n_pre; i++) {
        const comp_step & step = plan.pre[i];
        switch (step.op) {
        case OP_EMBED_LOOKUP: {
            hipMemcpyAsync(b.batch_token_ids, &token_id, sizeof(int), hipMemcpyHostToDevice, ctx.s);
            hipFunction_t fn = comp_pick_embed(k, cfg.embed_type);
            int nb_x = comp_embed_blocks(H, cfg.embed_type);
            int threads = comp_embed_threads(cfg.embed_type);
            const void * ew = cfg.embed_weight;
            long long es = cfg.embed_stride;
            void * args[] = { (void *)&b.batch_token_ids, (void *)&ew,
                              (void *)&es, (void *)&ctx.hidden };
            hipModuleLaunchKernel(fn, nb_x, 1, 1, threads, 1, 1, 0, ctx.s, args, nullptr);
            break;
        }
        case OP_EMBED_SCALE: {
            float scale = sqrtf((float)H);
            int n = H;
            void * args[] = { (void *)&ctx.hidden, (void *)&ctx.hidden, (void *)&scale, (void *)&n };
            hipModuleLaunchKernel(k.eval_scale_scalar, (n+255)/256, 1, 1, 256, 1, 1, 0, ctx.s, args, nullptr);
            break;
        }
        default: break;
        }
    }

    // ================================================================
    // LAYER LOOP — flat iteration, no branching
    // ================================================================
    for (int il = 0; il < plan.n_layers; il++) {
        for (int i = 0; i < plan.n_layer; i++) {
            const comp_step & step = plan.layer[i];
            switch (step.op) {
            case OP_RMSNORM_Q8_QUANTIZE:
                exec_rmsnorm_q8_quantize(step, il, ctx);
                break;
            case OP_FUSED_QKV:
                exec_fused_qkv(step, il, ctx);
                break;
            case OP_MATVEC:
                exec_matvec(step, il, ctx, ctx.hidden);
                break;
            case OP_MATVEC_RESIDUAL:
                exec_matvec_residual(step, il, ctx);
                break;
            case OP_QUANTIZE_Q8:
                // Determine input buffer from context
                if (step.dim_in == q_size)
                    exec_quantize_q8(step, il, ctx, ctx.attn_out);
                else if (step.dim_in == FF)
                    exec_quantize_q8(step, il, ctx, ctx.mlp_inter);
                else
                    exec_quantize_q8(step, il, ctx, ctx.norm_out);
                break;
            case OP_ROPE_KV_WRITE:
                exec_rope_kv_write(step, il, ctx);
                break;
            case OP_ATTN_DECODE_VEC:
            case OP_ATTN_DECODE_TILE:
            case OP_ATTN_DECODE_WMMA:
                exec_attention(step, il, ctx);
                break;
            case OP_RMSNORM_ADD:
                exec_rmsnorm_add(step, il, ctx);
                break;
            case OP_FUSED_GATE_UP_SILU:
            case OP_FUSED_GATE_UP_GELU:
                exec_fused_gate_up(step, il, ctx);
                break;
            case OP_NOP:
                break;
            default:
                fprintf(stderr, "gfx1100 composition executor: unhandled op %d at layer %d step %d\n",
                        step.op, il, i);
                return -1;
            }
        }
    }

    // ================================================================
    // POST-LAYER
    // ================================================================
    for (int i = 0; i < plan.n_post; i++) {
        const comp_step & step = plan.post[i];
        switch (step.op) {
        case OP_FINAL_NORM: {
            const void * w = cfg.final_norm_weight;
            int n = H;
            int nt = (H < 1024) ? 256 : 1024;
            // Final norm: no residual needed, reuse norm_out as dummy
            void * args[] = { (void *)&ctx.hidden, (void *)&w, (void *)&ctx.norm_out,
                              (void *)&ctx.norm_out, (void *)&n };
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, nt, 1, 1, 0, ctx.s, args, nullptr);
            break;
        }
        case OP_QUANTIZE_Q8: {
            int n = H;
            int blocks = (n + 511) / 512;
            void * args[] = { (void *)&ctx.norm_out, (void *)&ctx.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, ctx.s, args, nullptr);
            break;
        }
        case OP_LM_HEAD: {
            const void * w = cfg.lm_head_weight;
            long long st = cfg.lm_head_stride;
            bool is_f = comp_is_float(cfg.lm_head_type);
            const void * input = is_f ? (const void *)ctx.norm_out : (const void *)ctx.q8_act;
            hipFunction_t fn = comp_pick_matvec(k, cfg.lm_head_type);
            void * args[] = { (void *)&w, (void *)&st, (void *)&input,
                              (void *)&ctx.logits, (void *)&H, (void *)&V };
            if (is_f)
                hipModuleLaunchKernel(fn, V, 1, 1, 256, 1, 1, 0, ctx.s, args, nullptr);
            else
                hipModuleLaunchKernel(fn, V, 1, 1, 32, 4, 1, 0, ctx.s, args, nullptr);
            break;
        }
        case OP_SOFTCAP: {
            float cap = cfg.final_logit_softcap_val;
            int n = V;
            void * args[] = { (void *)&ctx.logits, (void *)&cap, (void *)&n };
            hipModuleLaunchKernel(k.eval_softcap, (n+255)/256, 1, 1, 256, 1, 1, 0, ctx.s, args, nullptr);
            break;
        }
        default: break;
        }
    }

    // Copy logits to host
    if (logits_out) {
        hipMemcpyAsync(logits_out, ctx.logits, V * sizeof(float), hipMemcpyDeviceToHost, ctx.s);
    }
    hipStreamSynchronize(ctx.s);

    return 0;
}
