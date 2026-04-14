// forward/cogvlm.cpp — CogVLM decoder forward (fused QKV + visual expert)
// Ported from baseline src/models/cogvlm.cpp
//
// Key differences from standard Llama:
//   1. Fused wqkv: single [3*n_embd, n_embd] matrix → slice into Q||K||V
//   2. Dual weight sets: text tokens use wqkv/wo/ffn_*, image tokens use visexp_*
//   3. No QKV biases, no LoRA scales, no output bias
//   4. Simple RoPE (no YaRN, no freq_factors)
//   5. No MoE, no QK norm, no attention scale override
//
// For decode: batch-level text vs image decision. The megakernel processes
// one token at a time. The caller sets a flag or the config determines
// which weight set to use. For now: always use text weights (ptrs[1..10]).
// Visual expert support requires the caller to swap weight pointers.
#include "../gfx1100-internal.h"

int forward_decode_cogvlm(int token_id, int position, float * logits_out) {
    auto & c = g_config;
    auto & b = g_bufs;
    auto & k = g_compiled;
    auto   s = b.stream;
    int    H = c.hidden_size;
    int    V = c.vocab_size;
    int    FF = c.intermediate_size;
    const int norm_threads = (H < 1024) ? 256 : 1024;
    int head_dim = c.fa_head_dim;
    int n_head = c.fa_n_q_heads;
    int n_kv_head = c.fa_n_kv_heads;
    int qproj_size = n_head * head_dim;
    int kv_size = n_kv_head * head_dim;

    // Write per-token decode parameters to GPU-resident buffer.
    {
        int h_params[2] = { token_id, position };
        hipMemcpyAsync(b.d_decode_params, h_params, 2 * sizeof(int),
                       hipMemcpyHostToDevice, s);
    }

    // Embedding
    if (launch_embed(token_id, b.hidden, s) != 0) return -1;

    for (int il = 0; il < c.n_layers; il++) {
        const gfx1100_layer_weights & lw = c.layers[il];

        // Weight selection: text vs visual expert
        // Baseline cogvlm.cpp lines 22-46: batch-level decision based on ubatch.token != NULL
        // For megakernel: always text path. Visual expert would use visexp_* weights.
        // The weight pointers are in the standard ptrs[] slots for text.
        // For fused QKV: ptrs[1] = wqkv [3*n_embd, n_embd], strides[1], types[1]
        // For visual expert: lw.visexp_wqkv, etc.
        // Baseline cogvlm.cpp lines 22-27: text if ubatch.token != NULL, image otherwise
        // For megakernel: caller sets cogvlm_is_image in config before eval_decode
        bool is_text = (c.cogvlm_is_image == 0);

        const void * wqkv_ptr    = is_text ? lw.ptrs[1]      : lw.visexp_wqkv;
        long long    wqkv_stride = is_text ? lw.strides[1]    : lw.visexp_wqkv_stride;
        int          wqkv_type   = is_text ? lw.types[1]      : lw.visexp_wqkv_type;
        const void * wo_ptr      = is_text ? lw.ptrs[6]       : lw.visexp_wo;
        long long    wo_stride   = is_text ? lw.strides[6]    : lw.visexp_wo_stride;
        int          wo_type     = is_text ? lw.types[6]      : lw.visexp_wo_type;
        const void * gate_ptr    = is_text ? lw.ptrs[8]       : lw.visexp_ffn_gate;
        long long    gate_stride = is_text ? lw.strides[8]    : lw.visexp_ffn_gate_stride;
        int          gate_type   = is_text ? lw.types[8]      : lw.visexp_ffn_gate_type;
        const void * up_ptr      = is_text ? lw.ptrs[9]       : lw.visexp_ffn_up;
        long long    up_stride   = is_text ? lw.strides[9]    : lw.visexp_ffn_up_stride;
        int          up_type     = is_text ? lw.types[9]      : lw.visexp_ffn_up_type;
        const void * down_ptr    = is_text ? lw.ptrs[10]      : lw.visexp_ffn_down;
        long long    down_stride = is_text ? lw.strides[10]   : lw.visexp_ffn_down_stride;
        int          down_type   = is_text ? lw.types[10]     : lw.visexp_ffn_down_type;

        // ===== Attention norm (shared between text and visual expert) =====
        // Baseline cogvlm.cpp line 49: build_norm(inpL, attn_norm, NULL, LLM_NORM_RMS)
        {
            const void * nw = lw.ptrs[0]; int n = H;
            void * args[] = { (void *)&b.hidden, (void *)&nw, (void *)&b.norm_out,
                              (void *)&b.residual, (void *)&n };
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        }

        // ===== Fused QKV projection =====
        // Baseline cogvlm.cpp line 53: qkv = wqkv * cur
        // Output: [3*n_embd] = Q||K||V concatenated
        // For quantized weights: quantize input first, then matvec
        {
            int n = H, blocks = (n+511)/512;
            void * q8args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
        }

        int qkv_dim = qproj_size + 2 * kv_size;  // 3*n_embd for standard MHA
        launch_matvec_typed(wqkv_type, wqkv_ptr, wqkv_stride,
                            b.proj_scratch, H, qkv_dim, s);

        // Slice QKV from proj_scratch:
        // proj_scratch[0..qproj_size-1] = Q
        // proj_scratch[qproj_size..qproj_size+kv_size-1] = K
        // proj_scratch[qproj_size+kv_size..qkv_dim-1] = V
        // Copy K and V to kv_scratch for the QK-norm-RoPE kernel
        hipMemcpyAsync(b.kv_scratch, b.proj_scratch + qproj_size,
                       kv_size * sizeof(float), hipMemcpyDeviceToDevice, s);
        hipMemcpyAsync(b.kv_scratch + kv_size, b.proj_scratch + qproj_size + kv_size,
                       kv_size * sizeof(float), hipMemcpyDeviceToDevice, s);

        // ===== QK norm + RoPE + KV cache write =====
        // Baseline cogvlm.cpp lines 63-64: ggml_rope (simple, no YaRN)
        // Our kernel handles ROPE_TYPE at compile time
        {
            const void * q_nw = lw.ptrs[4]; const void * k_nw = lw.ptrs[5];
            void * kc = c.k_cache_ptrs[il]; void * vc = c.v_cache_ptrs[il];
            const void * ff = nullptr; // no freq_factors for CogVLM
            int total_heads = n_head + n_kv_head;
            int blocks = (total_heads + 15) / 16;
            float theta_ovr = 0.0f;
            const int * d_params_nullptr = nullptr;
            void * args[] = { (void *)&b.proj_scratch, (void *)&b.kv_scratch,
                              (void *)&q_nw, (void *)&k_nw,
                              (void *)&kc, (void *)&vc, (void *)&ff,
                              (void *)&position, (void *)&c.max_seq_len, (void *)&theta_ovr,
                              (void *)&d_params_nullptr };
            hipModuleLaunchKernel(k.eval_qk_norm_rope_kv_write, blocks, 1, 1, 512, 1, 1, 0, s, args, nullptr);
        }

        // ===== Attention decode =====
        // Baseline cogvlm.cpp line 67: build_attn with scale=1/sqrt(head_dim), wo=NULL
        {
            int kv_len = position + 1;
            float alibi_mb = 0, alibi_m0v = 0, alibi_m1v = 0;
            int alibi_nhl = 0;
            int cur_pos = position;
            const float * rel_bias = nullptr;
            float softcap = 0.0f;
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

        // ===== O projection + residual =====
        // Baseline cogvlm.cpp line 67: wo inside build_attn (but we do it manually)
        {
            int n = qproj_size, q8blocks = (n+511)/512;
            void * q8args[] = { (void *)&b.attn_out, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
        }
        launch_matvec_res_typed(wo_type, wo_ptr, wo_stride,
                                 b.residual, b.hidden, qproj_size, H, s);

        // ===== FFN norm (shared between text and visual expert) =====
        // Baseline cogvlm.cpp line 77: build_norm(ffn_inp, ffn_norm, NULL, LLM_NORM_RMS)
        {
            const void * nw = lw.ptrs[7]; int n = H;
            void * args[] = { (void *)&b.hidden, (void *)&nw, (void *)&b.norm_out,
                              (void *)&b.residual, (void *)&n };
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        }
        {
            int n = H, blocks = (n+511)/512;
            void * q8args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
        }

        // ===== FFN: SiLU-gated (parallel) =====
        // Baseline cogvlm.cpp lines 79-85: build_ffn with LLM_FFN_SILU, LLM_FFN_PAR
        // Gate
        launch_matvec_typed(gate_type, gate_ptr, gate_stride,
                            b.mlp_inter, H, FF, s);
        // Up
        launch_matvec_typed(up_type, up_ptr, up_stride,
                            b.proj_scratch, H, FF, s);
        // SiLU(gate) * up
        {
            int n = FF;
            void * args[] = { (void *)&b.mlp_inter, (void *)&b.proj_scratch,
                              (void *)&b.mlp_inter, (void *)&n };
            hipModuleLaunchKernel(k.eval_silu_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
        // Down + residual
        {
            int n = FF, q8blocks = (n+511)/512;
            void * q8args[] = { (void *)&b.mlp_inter, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
        }
        launch_matvec_res_typed(down_type, down_ptr, down_stride,
                                 b.residual, b.hidden, FF, H, s);
    }

    // Final norm
    {
        int n = H;
        void * args[] = { (void *)&b.hidden, (void *)&c.final_norm_weight,
                          (void *)&b.norm_out, (void *)&n };
        hipModuleLaunchKernel(k.eval_final_norm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
    }
    {
        int n = H, blocks = (n + 511) / 512;
        void * q8args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
        hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
    }
    launch_matvec_typed(c.lm_head_type, c.lm_head_weight, c.lm_head_stride,
                         b.logits, H, V, s);
    if (logits_out) {
        hipMemcpyAsync(logits_out, b.logits, V * sizeof(float), hipMemcpyDeviceToHost, s);
    }
    hipStreamSynchronize(s);
    return 0;
}
