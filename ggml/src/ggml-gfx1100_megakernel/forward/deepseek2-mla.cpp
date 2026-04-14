// forward/deepseek2-mla.cpp — DeepSeek2 MLA attention forward
#include "../gfx1100-internal.h"

int forward_decode_deepseek2_mla(int token_id, int position, float * logits_out) {
    // Ported from baseline src/models/deepseek2.cpp lines 52-275 (MLA path)
    auto & c = g_config;
    auto & b = g_bufs;
    auto & k = g_compiled;
    auto   s = b.stream;
    int    H = c.hidden_size;
    int    V = c.vocab_size;
    int    FF = c.intermediate_size;
    const int norm_threads = (H < 1024) ? 256 : 1024;

    int n_head          = c.fa_n_q_heads;
    int kv_lora_rank    = c.mla_kv_lora_rank > 0 ? c.mla_kv_lora_rank : 512;
    int n_embd_head_k   = c.fa_head_dim;
    int n_embd_head_qk_rope = c.mla_n_embd_head_qk_rope > 0 ? c.mla_n_embd_head_qk_rope : 64;
    int n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;
    int q_lora_rank     = c.mla_q_lora_rank > 0 ? c.mla_q_lora_rank : 1536;
    int n_layer_dense_lead = c.mla_n_layer_dense_lead > 0 ? c.mla_n_layer_dense_lead : 1;

    // Type-aware matvec: float types use (256,1,1); quantized w/ 8-warp use (32,8,1); fallback (32,4,1)
    auto is_float_type = [](int type) { return type == 0 || type == 1 || type == 30; };
    auto pick_matvec_8w = [&](int type) -> hipFunction_t {
        switch (type) {
            case  2: return k.eval_matvec_q4_0_8w;
            case  3: return k.eval_matvec_q4_1_8w;
            case  6: return k.eval_matvec_q5_0_8w;
            case  7: return k.eval_matvec_q5_1_8w;
            case  8: return k.eval_matvec_q8_0_8w;
            case 12: return k.eval_matvec_q4k_8w;
            case 14: return k.eval_matvec_q6k_8w;
            case 20: return k.eval_matvec_iq4_nl_8w;
            default: return nullptr;
        }
    };
    auto pick_matvec_8w_res = [&](int type) -> hipFunction_t {
        switch (type) {
            case  2: return k.eval_matvec_q4_0_8w_residual;
            case  3: return k.eval_matvec_q4_1_8w_residual;
            case  6: return k.eval_matvec_q5_0_8w_residual;
            case  7: return k.eval_matvec_q5_1_8w_residual;
            case  8: return k.eval_matvec_q8_0_8w_residual;
            case 12: return k.eval_matvec_q4k_8w_residual;
            case 14: return k.eval_matvec_q6k_8w_residual;
            case 20: return k.eval_matvec_iq4_nl_8w_residual;
            default: return nullptr;
        }
    };
    auto launch_mv = [&](int type, const void * w, long long st, float * src_f32, float * output, int in_dim, int out_dim) {
        bool is_f = is_float_type(type);
        const void * input = is_f ? (const void *)src_f32 : (const void *)b.q8_act;
        void * args[] = { (void *)&w, (void *)&st, (void *)&input, (void *)&output, (void *)&in_dim, (void *)&out_dim };
        if (is_f) {
            hipModuleLaunchKernel(pick_matvec(type), out_dim, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        } else {
            hipFunction_t fn8w = pick_matvec_8w(type);
            if (fn8w) hipModuleLaunchKernel(fn8w, out_dim, 1, 1, 32, 8, 1, 0, s, args, nullptr);
            else      hipModuleLaunchKernel(pick_matvec(type), out_dim, 1, 1, 32, 4, 1, 0, s, args, nullptr);
        }
    };
    auto launch_mv_res = [&](int type, const void * w, long long st, float * src_f32, float * residual, float * output, int in_dim, int out_dim) {
        bool is_f = is_float_type(type);
        const void * input = is_f ? (const void *)src_f32 : (const void *)b.q8_act;
        void * args[] = { (void *)&w, (void *)&st, (void *)&input, (void *)&residual, (void *)&output, (void *)&in_dim, (void *)&out_dim };
        if (is_f) {
            hipModuleLaunchKernel(pick_matvec_res(type), out_dim, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        } else {
            hipFunction_t fn8w = pick_matvec_8w_res(type);
            if (fn8w) hipModuleLaunchKernel(fn8w, out_dim, 1, 1, 32, 8, 1, 0, s, args, nullptr);
            else      hipModuleLaunchKernel(pick_matvec_res(type), out_dim, 1, 1, 32, 4, 1, 0, s, args, nullptr);
        }
    };

    auto launch_add = [&](float * dst, const void * bias, int n) {
        void * ba[] = { (void *)&dst, (void *)&bias, (void *)&dst, (void *)&n };
        hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, s, ba, nullptr);
    };
    auto quant_matvec = [&](int type, const void * w, long long st, float * input, float * output,
                             int in_dim, int out_dim) {
        quant_and_launch_matvec(type, w, st, input, output, in_dim, out_dim, s);
    };
    auto f32_matvec = [&](const void * w, long long st, float * input, float * output,
                           int in_dim, int out_dim) {
        void * args[] = { (void *)&w, (void *)&st, (void *)&input, (void *)&output, &in_dim, &out_dim };
        hipModuleLaunchKernel(k.eval_matvec_f32, out_dim, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    };

    // Write per-token decode parameters to GPU-resident buffer.
    {
        int h_params[2] = { token_id, position };
        hipMemcpyAsync(b.d_decode_params, h_params, 2 * sizeof(int),
                       hipMemcpyHostToDevice, s);
    }

    // Embedding — full type dispatch via shared launch_embed
    if (launch_embed(token_id, b.hidden, s) != 0) return -1;

    for (int il = 0; il < c.n_layers; il++) {
        const gfx1100_layer_weights & lw = c.layers[il];

        // RMSNorm — baseline line 56
        {
            const void * nw = lw.ptrs[0]; int n = H;
            void * args[] = { (void *)&b.hidden, (void *)&nw, (void *)&b.norm_out, (void *)&b.residual, (void *)&n };
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        }

        // Q LoRA: q = wq_a @ cur → norm → wq_b @ q — baseline lines 97-104
        quant_matvec(lw.wq_a_type, lw.wq_a, lw.wq_a_stride, b.norm_out, b.proj_scratch, H, q_lora_rank);
        {
            const void * nw = lw.attn_q_a_norm; int n = q_lora_rank;
            int nt = (q_lora_rank < 1024) ? 256 : 1024;
            void * args[] = { (void *)&b.proj_scratch, (void *)&nw, (void *)&b.proj_scratch,
                              (void *)&b.kv_scratch, (void *)&n };
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, nt, 1, 1, 0, s, args, nullptr);
        }
        int q_total = n_head * n_embd_head_k;
        quant_matvec(lw.wq_b_type, lw.wq_b, lw.wq_b_stride, b.proj_scratch, b.attn_out, q_lora_rank, q_total);

        // KV compressed: kv_cmpr_pe = wkv_a_mqa @ cur — baseline line 121
        int kv_cmpr_total = kv_lora_rank + n_embd_head_qk_rope;
        quant_matvec(lw.wkv_a_mqa_type, lw.wkv_a_mqa, lw.wkv_a_mqa_stride, b.norm_out, b.kv_scratch, H, kv_cmpr_total);

        // RMSNorm on kv_cmpr — baseline line 145
        {
            const void * nw = lw.attn_kv_a_norm; int n = kv_lora_rank;
            int nt = (kv_lora_rank < 1024) ? 256 : 1024;
            void * args[] = { (void *)&b.kv_scratch, (void *)&nw, (void *)&b.kv_scratch,
                              (void *)&b.mlp_inter, (void *)&n };
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, nt, 1, 1, 0, s, args, nullptr);
        }

        // RoPE on q_pe and k_pe — baseline lines 137-143
        // q_pe is at attn_out[h * n_embd_head_k + nope_dim] per head, size rope_dim
        // k_pe is at kv_scratch[kv_lora_rank], size rope_dim (single KV head)
        {
            // Per-head q_pe RoPE
            for (int h = 0; h < n_head; h++) {
                float * q_pe = b.attn_out + h * n_embd_head_k + n_embd_head_qk_nope;
                int pos = position;
                float theta = c.fa_rope_theta;
                int nd = n_embd_head_qk_rope;
                void * args[] = { (void *)&q_pe, (void *)&pos, (void *)&theta, (void *)&nd };
                hipModuleLaunchKernel(k.eval_rope_neox_inplace, 1, 1, 1, nd/2, 1, 1, 0, s, args, nullptr);
            }
            // k_pe RoPE (single KV head)
            {
                float * k_pe = b.kv_scratch + kv_lora_rank;
                int pos = position;
                float theta = c.fa_rope_theta;
                int nd = n_embd_head_qk_rope;
                void * args[] = { (void *)&k_pe, (void *)&pos, (void *)&theta, (void *)&nd };
                hipModuleLaunchKernel(k.eval_rope_neox_inplace, 1, 1, 1, nd/2, 1, 1, 0, s, args, nullptr);
            }
        }

        // Absorption: per-head q_nope_absorbed = wk_b @ q_nope — baseline line 154
        for (int h = 0; h < n_head; h++) {
            float * q_h = b.attn_out + h * n_embd_head_k;
            const char * wk_b_h = (const char *)lw.wk_b + (long long)h * kv_lora_rank * lw.wk_b_stride;
            float * absorbed_h = b.proj_scratch + h * (kv_lora_rank + n_embd_head_qk_rope);
            f32_matvec(wk_b_h, lw.wk_b_stride, q_h, absorbed_h, n_embd_head_qk_nope, kv_lora_rank);
            hipMemcpyAsync(absorbed_h + kv_lora_rank, q_h + n_embd_head_qk_nope,
                           n_embd_head_qk_rope * sizeof(float), hipMemcpyDeviceToDevice, s);
        }

        // Attention: Q = proj_scratch, K = kv_scratch[0..kv_cmpr_total], V = kv_scratch[0..kv_lora_rank]
        // MLA converts to MQA (1 KV head). head_dim = kv_lora_rank + rope_dim.
        // The baked-in .hsaco attention kernel has FA_HEAD_DIM set per model.
        // For MLA, FA_HEAD_DIM = kv_lora_rank + rope_dim = 576 (DS2) or similar.
        // If the .hsaco was compiled for this model, it should match.
        // Write K to KV cache position, then run attention
        {
            // Write K to cache: k_cache[0, position, :] = kv_scratch[0..kv_cmpr_total]
            // Write V to cache: v_cache[0, position, :] = kv_scratch[0..kv_lora_rank]
            // (1 KV head for MLA absorption path)
            if (c.k_cache_ptrs && c.k_cache_ptrs[il]) {
                int attn_head_dim = kv_lora_rank + n_embd_head_qk_rope;
                __half * kc = (__half *)c.k_cache_ptrs[il] + (long long)position * attn_head_dim;
                __half * vc = (__half *)c.v_cache_ptrs[il] + (long long)position * kv_lora_rank;
                // Convert f32 K to f16 and write to cache — using gemm_f32_to_f16 from prefill.hip
                {
                    float * k_f32 = b.kv_scratch; // K = [kv_lora_rank + rope_dim] in f32
                    int64_t k_elems = (int64_t)attn_head_dim;
                    void * args[] = { (void *)&k_f32, (void *)&kc, (void *)&k_elems };
                    hipModuleLaunchKernel(k.gemm_f32_to_f16, (attn_head_dim+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }
                // Convert f32 V to f16 — V = kv_cmpr = kv_scratch[0..kv_lora_rank-1]
                {
                    float * v_f32 = b.kv_scratch;
                    int64_t v_elems = (int64_t)kv_lora_rank;
                    void * args[] = { (void *)&v_f32, (void *)&vc, (void *)&v_elems };
                    hipModuleLaunchKernel(k.gemm_f32_to_f16, (kv_lora_rank+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }
            }

            // Run attention with Q from proj_scratch
            int kv_len = position + 1;
            float alibi_mb = c.alibi_max_bias;
            float alibi_m0v = c.alibi_m0, alibi_m1v = c.alibi_m1;
            int alibi_nhl = c.alibi_n_head_log2, cur_pos = position;
            const float * rel_bias = nullptr;
            float softcap = 0.0f;  // no T5 relative position bias for DeepSeek2
            const int * d_params_nullptr = nullptr;
            void * args[] = { (void *)&b.proj_scratch,
                              (void *)&c.k_cache_ptrs[il], (void *)&c.v_cache_ptrs[il],
                              (void *)&b.attn_out,
                              (void *)&kv_len, (void *)&c.max_seq_len,
                              (void *)&alibi_mb, (void *)&alibi_m0v,
                              (void *)&alibi_m1v, (void *)&alibi_nhl, (void *)&cur_pos,
                              (void *)&rel_bias,
                              (void *)&softcap, (void *)&d_params_nullptr };
            hipModuleLaunchKernel(k.eval_attention_decode, n_head, 1, 1, 32, 4, 1, 0, s, args, nullptr);
        }

        // V decompression: wv_b @ attn_out (per head) — done in build_attn via wv_b param
        // For our megakernel: separate matvec after attention
        if (lw.wv_b) {
            // attn_out has [kv_lora_rank per head * n_head]
            // wv_b: [n_embd_head_v, kv_lora_rank, n_head] decompresses V
            // For decode, per-head: [n_embd_head_v] = wv_b_h @ attn_out_h[kv_lora_rank]
            for (int h = 0; h < n_head; h++) {
                float * attn_h = b.attn_out + h * kv_lora_rank; // MQA: V dim is kv_lora_rank
                const char * wv_b_h = (const char *)lw.wv_b + (long long)h * n_embd_head_k * lw.wv_b_stride;
                float * out_h = b.proj_scratch + h * n_embd_head_k; // decompressed V
                f32_matvec(wv_b_h, lw.wv_b_stride, attn_h, out_h, kv_lora_rank, n_embd_head_k);
            }
            // proj_scratch now has decompressed attention output [n_head * n_embd_head_k = H]
            hipMemcpyAsync(b.attn_out, b.proj_scratch, H * sizeof(float), hipMemcpyDeviceToDevice, s);
        }

        // O projection + residual — baseline line 184-186
        {
            int n = H, q8blocks = (n+511)/512;
            void * q8args[] = { (void *)&b.attn_out, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
            launch_mv_res(lw.types[6], lw.ptrs[6], lw.strides[6], b.attn_out, b.residual, b.hidden, n, H);
        }

        // FFN — baseline lines 233-275
        {
            const void * nw = lw.ptrs[7]; int n = H;
            void * args[] = { (void *)&b.hidden, (void *)&nw, (void *)&b.norm_out, (void *)&b.residual, (void *)&n };
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        }
        {
            int n = H, blocks = (n+511)/512;
            void * q8args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
        }

        if (il < n_layer_dense_lead) {
            // Dense FFN: gate/up/silu_mul/down — baseline line 237
            launch_mv(lw.types[8], lw.ptrs[8], lw.strides[8], b.norm_out, b.mlp_inter, H, FF);
            launch_mv(lw.types[9], lw.ptrs[9], lw.strides[9], b.norm_out, b.proj_scratch, H, FF);
            {
                int n = FF;
                void * args[] = { (void *)&b.mlp_inter, (void *)&b.proj_scratch, (void *)&b.mlp_inter, (void *)&n };
                hipModuleLaunchKernel(k.eval_silu_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }
            {
                int n = FF, q8blocks = (n+511)/512;
                void * q8args[] = { (void *)&b.mlp_inter, (void *)&b.q8_act, (void *)&n };
                hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
                launch_mv_res(lw.types[10], lw.ptrs[10], lw.strides[10], b.mlp_inter, b.residual, b.hidden, FF, H);
            }
        } else if (lw.ffn_gate_inp) {
            // MoE FFN — reuse inline MoE dispatch from forward_decode_llama_family
            // Same routing logic: softmax → argsort → per-expert matvec → weighted accum
            const int n_expert = c.moe_n_experts;
            const int n_used = c.moe_n_experts_used;

            // Router
            {
                void * inp = (void *)b.q8_act;
                void * args[] = { (void *)&lw.ffn_gate_inp, (void *)&lw.ffn_gate_inp_stride,
                                  &inp, (void *)&b.proj_scratch, (void *)&H, (void *)&n_expert };
                hipModuleLaunchKernel(pick_matvec(lw.ffn_gate_inp_type), n_expert, 1, 1, 32, 4, 1, 0, s, args, nullptr);
            }
            // Softmax
            {
                float sv = 1.0f; int n = n_expert;
                void * args[] = { (void *)&b.proj_scratch, (void *)&b.proj_scratch, (void *)&n, (void *)&sv };
                hipModuleLaunchKernel(k.eval_softmax_row, 1, 1, 1, 256, 1, 1,
                                     (256/32) * sizeof(float), s, args, nullptr);
            }
            // Argsort — kernel: eval_argsort_desc(x, dst, ncols, ncols_pad) — bitonic sort
            {
                int n = n_expert;
                int npad = 1;
                while (npad < n) npad *= 2;
                void * inp = (void *)b.proj_scratch; void * out = (void *)b.mlp_inter;
                void * args[] = { &inp, &out, &n, &npad };
                hipModuleLaunchKernel(k.eval_argsort_desc, 1, 1, 1, npad, 1, 1,
                                     npad * sizeof(int), s, args, nullptr);
            }
            // Copy sorted indices and probs to dedicated MoE buffers (D2D, no host stall)
            hipMemcpyAsync(b.moe_sorted_ids, b.mlp_inter, n_expert * sizeof(int), hipMemcpyDeviceToDevice, s);
            hipMemcpyAsync(b.moe_probs, b.proj_scratch, n_expert * sizeof(float), hipMemcpyDeviceToDevice, s);

            // Normalize MoE weights on GPU — zero D2H
            {
                int do_norm = lw.moe_norm_w ? 1 : 0;
                float ws = lw.moe_w_scale;
                void * nargs[] = { (void *)&b.moe_probs, (void *)&b.moe_sorted_ids,
                                   (void *)&n_used, (void *)&do_norm, (void *)&ws };
                hipModuleLaunchKernel(k.eval_moe_normalize_weights, 1, 1, 1, 1, 1, 1, 0, s, nargs, nullptr);
            }

            // Zero the output accumulator
            hipMemsetAsync(b.hidden, 0, H * sizeof(float), s);

            // Expert strides (in bytes)
            long long gate_exp_stride = (long long)FF * lw.ffn_gate_exps_stride;
            long long up_exp_stride   = (long long)FF * lw.ffn_up_exps_stride;
            long long down_exp_stride = (long long)H  * lw.ffn_down_exps_stride;

            auto pick_moe_mv = [&](int type) -> hipFunction_t {
                switch (type) {
                    case  0: return k.eval_moe_matvec_f32;
                    case  1: return k.eval_moe_matvec_f16;
                    case 30: return k.eval_moe_matvec_bf16;
                    case  2: return k.eval_moe_matvec_q4_0;
                    case  3: return k.eval_moe_matvec_q4_1;
                    case  6: return k.eval_moe_matvec_q5_0;
                    case  7: return k.eval_moe_matvec_q5_1;
                    case  8: return k.eval_moe_matvec_q8_0;
                    case 10: return k.eval_moe_matvec_q2k;
                    case 11: return k.eval_moe_matvec_q3k;
                    case 12: return k.eval_moe_matvec_q4k;
                    case 13: return k.eval_moe_matvec_q5k;
                    case 14: return k.eval_moe_matvec_q6k;
                    case 16: return k.eval_moe_matvec_iq2_xxs;
                    case 17: return k.eval_moe_matvec_iq2_xs;
                    case 18: return k.eval_moe_matvec_iq3_xxs;
                    case 19: return k.eval_moe_matvec_iq1_s;
                    case 20: return k.eval_moe_matvec_iq4_nl;
                    case 21: return k.eval_moe_matvec_iq3_s;
                    case 22: return k.eval_moe_matvec_iq2_s;
                    case 23: return k.eval_moe_matvec_iq4_xs;
                    case 29: return k.eval_moe_matvec_iq1_m;
                    case 39: return k.eval_moe_matvec_mxfp4;
                    case 40: return k.eval_moe_matvec_nvfp4;
                    default:
                        fprintf(stderr, "gfx1100: unsupported MoE matvec type %d\n", type);
                        return k.eval_moe_matvec_q4k;
                }
            };

            auto is_moe_float_type = [](int type) -> bool {
                return type == 0 || type == 1 || type == 30;
            };

            for (int ei = 0; ei < n_used; ei++) {
                // Re-quantize norm_out for this expert's gate/up projections
                // (q8_act gets overwritten by down-proj quantize on each iteration)
                {
                    int n = H, q8blocks = (n + 511) / 512;
                    void * q8args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
                    hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
                }

                // Gate: fused MoE matvec — reads expert ID from moe_sorted_ids on GPU
                {
                    hipFunction_t fn = pick_moe_mv(lw.ffn_gate_exps_type);
                    bool is_float = is_moe_float_type(lw.ffn_gate_exps_type);
                    void * input = is_float ? (void *)b.norm_out : (void *)b.q8_act;
                    void * args[] = { (void *)&lw.ffn_gate_exps, (void *)&gate_exp_stride,
                                      (void *)&lw.ffn_gate_exps_stride, (void *)&input,
                                      (void *)&b.mlp_inter, (void *)&b.moe_sorted_ids,
                                      (void *)&ei, (void *)&H, (void *)&FF };
                    if (is_float) hipModuleLaunchKernel(fn, FF, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                    else hipModuleLaunchKernel(fn, FF, 1, 1, 32, 4, 1, 0, s, args, nullptr);
                }

                // Up: fused MoE matvec
                {
                    hipFunction_t fn = pick_moe_mv(lw.ffn_up_exps_type);
                    bool is_float = is_moe_float_type(lw.ffn_up_exps_type);
                    void * input = is_float ? (void *)b.norm_out : (void *)b.q8_act;
                    void * args[] = { (void *)&lw.ffn_up_exps, (void *)&up_exp_stride,
                                      (void *)&lw.ffn_up_exps_stride, (void *)&input,
                                      (void *)&b.proj_scratch, (void *)&b.moe_sorted_ids,
                                      (void *)&ei, (void *)&H, (void *)&FF };
                    if (is_float) hipModuleLaunchKernel(fn, FF, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                    else hipModuleLaunchKernel(fn, FF, 1, 1, 32, 4, 1, 0, s, args, nullptr);
                }

                // SiLU(gate) * up
                {
                    int n = FF;
                    void * args[] = { (void *)&b.mlp_inter, (void *)&b.proj_scratch, (void *)&b.mlp_inter, (void *)&n };
                    hipModuleLaunchKernel(k.eval_silu_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }

                // Quantize activation for down projection
                {
                    int n = FF, q8blocks = (n+511)/512;
                    void * q8args[] = { (void *)&b.mlp_inter, (void *)&b.q8_act, (void *)&n };
                    hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
                }

                // Down: fused MoE matvec
                {
                    hipFunction_t fn = pick_moe_mv(lw.ffn_down_exps_type);
                    bool is_float = is_moe_float_type(lw.ffn_down_exps_type);
                    void * input = is_float ? (void *)b.mlp_inter : (void *)b.q8_act;
                    void * args[] = { (void *)&lw.ffn_down_exps, (void *)&down_exp_stride,
                                      (void *)&lw.ffn_down_exps_stride, (void *)&input,
                                      (void *)&b.proj_scratch, (void *)&b.moe_sorted_ids,
                                      (void *)&ei, (void *)&FF, (void *)&H };
                    if (is_float) hipModuleLaunchKernel(fn, H, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                    else hipModuleLaunchKernel(fn, H, 1, 1, 32, 4, 1, 0, s, args, nullptr);
                }

                // Weighted accumulate — weight read from GPU moe_probs buffer
                {
                    int n = H;
                    void * args[] = { (void *)&b.hidden, (void *)&b.proj_scratch,
                                      (void *)&b.moe_probs, (void *)&b.moe_sorted_ids,
                                      (void *)&ei, (void *)&n };
                    hipModuleLaunchKernel(k.eval_moe_weighted_add, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }
            }
            // DeepSeek2 shared expert FFN — baseline deepseek2.cpp lines 260-272
            // Unlike Qwen2MoE, DS2 shared expert has NO gating mechanism — it's a plain SwiGLU FFN
            // added directly to the MoE output.
            if (lw.ffn_gate_shexp && lw.ffn_up_shexp && lw.ffn_down_shexp) {
                // Re-quantize norm_out for shared expert projections
                {
                    int n = H, q8blocks = (n + 511) / 512;
                    void * q8args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
                    hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
                }
                // Gate: [FF] = ffn_gate_shexp @ norm_out
                launch_mv(lw.ffn_gate_shexp_type, lw.ffn_gate_shexp, lw.ffn_gate_shexp_stride,
                          b.norm_out, b.mlp_inter, H, FF);
                // Up: [FF] = ffn_up_shexp @ norm_out
                launch_mv(lw.ffn_up_shexp_type, lw.ffn_up_shexp, lw.ffn_up_shexp_stride,
                          b.norm_out, b.proj_scratch, H, FF);
                // SiLU(gate) * up
                {
                    int n = FF;
                    void * args[] = { (void *)&b.mlp_inter, (void *)&b.proj_scratch, (void *)&b.mlp_inter, (void *)&n };
                    hipModuleLaunchKernel(k.eval_silu_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }
                // Down: [H] = ffn_down_shexp @ mlp_inter
                {
                    int n = FF, q8blocks = (n+511)/512;
                    void * q8args[] = { (void *)&b.mlp_inter, (void *)&b.q8_act, (void *)&n };
                    hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
                }
                launch_mv(lw.ffn_down_shexp_type, lw.ffn_down_shexp, lw.ffn_down_shexp_stride,
                          b.mlp_inter, b.proj_scratch, FF, H);
                // hidden += shared_expert_out (no gating for DS2, unlike Qwen2MoE)
                {
                    int n = H;
                    void * args[] = { (void *)&b.hidden, (void *)&b.proj_scratch, (void *)&b.hidden, (void *)&n };
                    hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }
            }
            launch_add(b.hidden, (const void *)b.residual, H);
        }
    }

    // Final norm + LM head
    {
        int n = H;
        void * args[] = { (void *)&b.hidden, (void *)&c.final_norm_weight, (void *)&b.norm_out, (void *)&n };
        hipModuleLaunchKernel(k.eval_final_norm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
    }
    {
        int n = H, blocks = (n+511)/512;
        void * q8args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
        hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
    }
    {
        launch_mv(c.lm_head_type, c.lm_head_weight, c.lm_head_stride, b.norm_out, b.logits, H, V);
    }
    if (logits_out) {
        hipMemcpyAsync(logits_out, b.logits, V * sizeof(float), hipMemcpyDeviceToHost, s);
    }
    hipStreamSynchronize(s);
    return 0;
}

// ----------------------------------------------------------------------------
// forward_decode_rwkv6 — ported from baseline src/models/rwkv6-base.cpp
//
// RWKV6 recurrent language model decode. Per layer:
//   LayerNorm → time_mix (recurrent, replaces attention):
//     lerp mixing → R,K,V,G projections → decay computation → WKV6 step →
//     group norm → gate → output projection
//   LayerNorm → channel_mix (replaces FFN):
//     lerp mixing → receptance(sigmoid) * value(key^2) → output
//   Residual add
//
// Stateful: x_prev [n_layers, n_embd] and WKV state [n_layers, n_head, head_size, head_size]
// ----------------------------------------------------------------------------
