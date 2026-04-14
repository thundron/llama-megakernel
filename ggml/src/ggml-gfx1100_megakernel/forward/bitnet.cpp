// forward/bitnet.cpp — BitNet 1.58-bit forward
// Ported from baseline src/models/bitnet.cpp
//
// Nearly identical to Llama except:
//   1. attn_sub_norm (RMSNorm) applied BETWEEN attention output and wo projection
//   2. ffn_sub_norm (RMSNorm) applied BETWEEN SiLU-gated output and ffn_down projection
//   3. wo and ffn_down are done OUTSIDE build_attn/build_ffn
//   4. LM head reuses tok_embd (weight-tied)
//   5. Uses TQ1_0/TQ2_0 ternary quant types for weights
//
// Note: TQ1_0/TQ2_0 ternary matvec kernels don't exist in baseline ggml-cuda.
// This forward function uses the standard pick_matvec dispatch which will work
// if/when ternary matvec kernels are added to decode.hip.
#include "../gfx1100-internal.h"

int forward_decode_bitnet(int token_id, int position, float * logits_out) {
    auto & c = g_config;
    auto & b = g_bufs;
    auto & k = g_compiled;
    auto   s = b.stream;
    int    H = c.hidden_size;
    int    V = c.vocab_size;
    int    FF = c.intermediate_size;
    const int norm_threads = (H < 1024) ? 256 : 1024;

    // Write per-token decode parameters to GPU-resident buffer.
    // Embedding reads token_id from d_decode_params[0], RoPE/attention read position from [1].
    {
        int h_params[2] = { token_id, position };
        hipMemcpyAsync(b.d_decode_params, h_params, 2 * sizeof(int),
                       hipMemcpyHostToDevice, s);
    }

    // Embedding
    if (launch_embed(token_id, b.hidden, s) != 0) return -1;

    for (int il = 0; il < c.n_layers; il++) {
        const gfx1100_layer_weights & lw = c.layers[il];

        // ===== Attention block =====

        // RMSNorm — baseline bitnet.cpp line 24
        {
            const void * nw = lw.ptrs[0]; int n = H;
            void * args[] = { (void *)&b.hidden, (void *)&nw, (void *)&b.norm_out,
                              (void *)&b.residual, (void *)&n };
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        }

        // Quantize
        {
            int n = H, blocks = (n+511)/512;
            void * q8args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
        }

        // Q, K, V projections — baseline lines 32-53 (with optional wq_s/wk_s/wv_s scales)
        int qproj_size = c.fa_n_q_heads * c.fa_head_dim;
        int kv_size = c.fa_n_kv_heads * c.fa_head_dim;

        // Q
        launch_matvec_typed(lw.types[1], lw.ptrs[1], lw.strides[1],
                            b.proj_scratch, H, qproj_size, s);
        if (lw.scale_q) {
            // BitNet per-tensor scale: scalar multiply
            int n = qproj_size;
            void * args[] = { (void *)&b.proj_scratch, (void *)&lw.scale_q, (void *)&b.proj_scratch, (void *)&n };
            hipModuleLaunchKernel(k.eval_elementwise_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
        if (lw.bias_q) {
            int n = qproj_size;
            void * args[] = { (void *)&b.proj_scratch, (void *)&lw.bias_q, (void *)&b.proj_scratch, (void *)&n };
            hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }

        // K
        launch_matvec_typed(lw.types[2], lw.ptrs[2], lw.strides[2],
                            b.kv_scratch, H, kv_size, s);
        if (lw.scale_k) {
            int n = kv_size;
            void * args[] = { (void *)&b.kv_scratch, (void *)&lw.scale_k, (void *)&b.kv_scratch, (void *)&n };
            hipModuleLaunchKernel(k.eval_elementwise_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }

        // V
        {
            float * v_out = b.kv_scratch + kv_size;
            launch_matvec_typed(lw.types[3], lw.ptrs[3], lw.strides[3],
                                v_out, H, kv_size, s);
            if (lw.scale_v) {
                int n = kv_size;
                void * args[] = { (void *)&v_out, (void *)&lw.scale_v, (void *)&v_out, (void *)&n };
                hipModuleLaunchKernel(k.eval_elementwise_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }
        }

        // QK norm + RoPE + KV cache write
        {
            const void * q_nw = lw.ptrs[4]; const void * k_nw = lw.ptrs[5];
            void * kc = c.k_cache_ptrs[il]; void * vc = c.v_cache_ptrs[il];
            const void * ff = c.rope_freq_factors;
            int total_heads = c.fa_n_q_heads + c.fa_n_kv_heads;
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

        // Attention decode
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
            hipModuleLaunchKernel(k.eval_attention_decode, c.fa_n_q_heads, 1, 1, 32, 4, 1, 0, s, args, nullptr);
        }

        // ===== BitNet attn_sub_norm: RMSNorm BEFORE wo projection =====
        // Baseline bitnet.cpp lines 79-82
        if (lw.attn_sub_norm) {
            int n = qproj_size;
            float eps = c.norm_eps;
            // Inline RMSNorm on attn_out [qproj_size]
            // Use eval_rmsnorm_q8 with norm_out=attn_out (in-place via temp)
            // Actually, we need a bare RMSNorm (no q8 output, no residual copy)
            // For BitNet: the sub_norm just normalizes, no weight multiplication (bare norm)
            // Wait — baseline does build_norm which DOES use the weight.
            // Let me re-read: build_norm(cur, attn_sub_norm, NULL, LLM_NORM_RMS)
            // So it IS weighted: y = x / rms(x) * weight
            // Use the eval_rmsnorm_q8 pattern but without the q8/residual output
            // Since we can't easily do this with eval_rmsnorm_q8, use a two-step:
            // 1. Copy attn_out to proj_scratch
            // 2. Run eval_rmsnorm_q8 with input=proj_scratch, norm_out=attn_out, residual=NULL-ish
            hipMemcpyAsync(b.proj_scratch, b.attn_out, qproj_size * sizeof(float), hipMemcpyDeviceToDevice, s);
            void * args[] = { (void *)&b.proj_scratch, (void *)&lw.attn_sub_norm,
                              (void *)&b.attn_out, (void *)&b.proj_scratch, (void *)&n };
            int nt = (n < 1024) ? 256 : 1024;
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, nt, 1, 1, 0, s, args, nullptr);
        }

        // wo projection + scale + residual — baseline bitnet.cpp lines 84-88
        // wo is done OUTSIDE build_attn
        {
            int n = qproj_size, q8blocks = (n+511)/512;
            void * q8args[] = { (void *)&b.attn_out, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);

            // wo matvec → hidden
            launch_matvec_typed(lw.types[6], lw.ptrs[6], lw.strides[6],
                                b.hidden, qproj_size, H, s);
            if (lw.scale_o) {
                int n2 = H;
                void * args[] = { (void *)&b.hidden, (void *)&lw.scale_o, (void *)&b.hidden, (void *)&n2 };
                hipModuleLaunchKernel(k.eval_elementwise_mul, (n2+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }
            if (lw.bias_o) {
                int n2 = H;
                void * args[] = { (void *)&b.hidden, (void *)&lw.bias_o, (void *)&b.hidden, (void *)&n2 };
                hipModuleLaunchKernel(k.eval_add_residual, (n2+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }
            // Residual add
            {
                int n2 = H;
                void * args[] = { (void *)&b.hidden, (void *)&b.residual, (void *)&b.hidden, (void *)&n2 };
                hipModuleLaunchKernel(k.eval_add_residual, (n2+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }
        }

        // ===== FFN block =====

        // FFN norm — baseline bitnet.cpp line 100
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

        // Gate + Up projections — baseline bitnet.cpp lines 105-111
        // Gate
        launch_matvec_typed(lw.types[8], lw.ptrs[8], lw.strides[8],
                            b.mlp_inter, H, FF, s);
        if (lw.ffn_gate_scale) {
            int n = FF;
            void * args[] = { (void *)&b.mlp_inter, (void *)&lw.ffn_gate_scale, (void *)&b.mlp_inter, (void *)&n };
            hipModuleLaunchKernel(k.eval_elementwise_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
        // Up
        launch_matvec_typed(lw.types[9], lw.ptrs[9], lw.strides[9],
                            b.proj_scratch, H, FF, s);
        if (lw.ffn_up_scale) {
            int n = FF;
            void * args[] = { (void *)&b.proj_scratch, (void *)&lw.ffn_up_scale, (void *)&b.proj_scratch, (void *)&n };
            hipModuleLaunchKernel(k.eval_elementwise_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }

        // SiLU(gate) * up — baseline: LLM_FFN_SILU + LLM_FFN_PAR
        {
            int n = FF;
            void * args[] = { (void *)&b.mlp_inter, (void *)&b.proj_scratch,
                              (void *)&b.mlp_inter, (void *)&n };
            hipModuleLaunchKernel(k.eval_silu_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }

        // ===== BitNet ffn_sub_norm: RMSNorm BEFORE ffn_down projection =====
        // Baseline bitnet.cpp lines 113-116
        // ffn_sub_norm has shape [n_ff], normalizes the FF-dimensional intermediate
        if (lw.ffn_sub_norm) {
            int n = FF;
            hipMemcpyAsync(b.proj_scratch, b.mlp_inter, FF * sizeof(float), hipMemcpyDeviceToDevice, s);
            void * args[] = { (void *)&b.proj_scratch, (void *)&lw.ffn_sub_norm,
                              (void *)&b.mlp_inter, (void *)&b.proj_scratch, (void *)&n };
            int nt = (n < 1024) ? 256 : 1024;
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, nt, 1, 1, 0, s, args, nullptr);
        }

        // ffn_down projection + scale + residual — baseline bitnet.cpp lines 118-121
        {
            int n = FF, q8blocks = (n+511)/512;
            void * q8args[] = { (void *)&b.mlp_inter, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
        }
        // Down projection → attn_out (temp buffer)
        launch_matvec_typed(lw.types[10], lw.ptrs[10], lw.strides[10],
                            b.attn_out, FF, H, s);
        if (lw.ffn_down_scale) {
            int n = H;
            void * args[] = { (void *)&b.attn_out, (void *)&lw.ffn_down_scale, (void *)&b.attn_out, (void *)&n };
            hipModuleLaunchKernel(k.eval_elementwise_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
        // Residual: hidden = down_out + ffn_inp(residual)
        {
            int n = H;
            void * args[] = { (void *)&b.attn_out, (void *)&b.residual, (void *)&b.hidden, (void *)&n };
            hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
    }

    // Final norm
    {
        int n = H;
        void * args[] = { (void *)&b.hidden, (void *)&c.final_norm_weight,
                          (void *)&b.norm_out, (void *)&n };
        hipModuleLaunchKernel(k.eval_final_norm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
    }
    // Quantize for LM head
    {
        int n = H, blocks = (n + 511) / 512;
        void * q8args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
        hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
    }
    // LM head — uses tok_embd (weight-tied), stored as lm_head_weight in config
    launch_matvec_typed(c.lm_head_type, c.lm_head_weight, c.lm_head_stride,
                         b.logits, H, V, s);
    if (logits_out) {
        hipMemcpyAsync(logits_out, b.logits, V * sizeof(float), hipMemcpyDeviceToHost, s);
    }
    hipStreamSynchronize(s);
    return 0;
}
