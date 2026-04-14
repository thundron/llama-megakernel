// forward/t5-dec.cpp — T5 decoder forward (self-attention + cross-attention)
// Ported from baseline src/models/t5-dec.cpp (build_t5_dec)
#include "../gfx1100-internal.h"
#include "../shared/batch-ops.h"

int forward_decode_t5(int token_id, int position, float * logits_out) {
    auto & c = g_config;
    auto & b = g_bufs;
    auto & k = g_compiled;
    auto   s = b.stream;
    int    H = c.hidden_size;
    int    V = c.vocab_size;
    int    FF = c.intermediate_size;
    const int norm_threads = (H < 1024) ? 256 : 1024;

    auto launch_add = [&](float * dst, const void * bias, int n) {
        void * ba[] = { (void *)&dst, (void *)&bias, (void *)&dst, (void *)&n };
        hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, s, ba, nullptr);
    };

    // Type-aware matvec: float types use (256,1,1) block and f32 input; quantized use (32,4,1) + q8_act
    auto is_float_type = [](int type) { return type == 0 || type == 1 || type == 30; };
    auto launch_mv = [&](int type, const void * w, long long st, float * src_f32, float * output, int in_dim, int out_dim) {
        bool is_f = is_float_type(type);
        const void * input = is_f ? (const void *)src_f32 : (const void *)b.q8_act;
        void * args[] = { (void *)&w, (void *)&st, (void *)&input, (void *)&output, (void *)&in_dim, (void *)&out_dim };
        hipModuleLaunchKernel(pick_matvec(type), out_dim, 1, 1, is_f ? 256 : 32, is_f ? 1 : 4, 1, 0, s, args, nullptr);
    };
    auto launch_mv_res = [&](int type, const void * w, long long st, float * src_f32, float * residual, float * output, int in_dim, int out_dim) {
        bool is_f = is_float_type(type);
        const void * input = is_f ? (const void *)src_f32 : (const void *)b.q8_act;
        void * args[] = { (void *)&w, (void *)&st, (void *)&input, (void *)&residual, (void *)&output, (void *)&in_dim, (void *)&out_dim };
        hipModuleLaunchKernel(pick_matvec_res(type), out_dim, 1, 1, is_f ? 256 : 32, is_f ? 1 : 4, 1, 0, s, args, nullptr);
    };

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

        // Phase 1: Self-attention (causal, with KV cache)
        // RMSNorm — baseline t5-dec.cpp line 30
        {
            const void * nw = lw.ptrs[0]; int n = H;
            void * args[] = { (void *)&b.hidden, (void *)&nw, (void *)&b.norm_out, (void *)&b.residual, (void *)&n };
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        }
        // Quantize
        {
            int n = H, blocks = (n+511)/512;
            void * q8args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
        }
        // Q, K, V projections — baseline lines 37-48
        int qproj_size = c.fa_n_q_heads * c.fa_head_dim;
        int kv_size = c.fa_n_kv_heads * c.fa_head_dim;
        launch_mv(lw.types[1], lw.ptrs[1], lw.strides[1], b.norm_out, b.proj_scratch, H, qproj_size);
        launch_mv(lw.types[2], lw.ptrs[2], lw.strides[2], b.norm_out, b.kv_scratch, H, kv_size);
        launch_mv(lw.types[3], lw.ptrs[3], lw.strides[3], b.norm_out, b.kv_scratch + kv_size, H, kv_size);

        // QK norm + RoPE (T5 uses ROPE_NONE) + KV cache write
        // T5 has relative position bias instead of RoPE — handled by ROPE_TYPE=0 (skip rotation)
        {
            int attn_idx = il; // for T5, attn layer index = layer index
            const void * q_nw = lw.ptrs[4]; const void * k_nw = lw.ptrs[5];
            void * kc = c.k_cache_ptrs[attn_idx]; void * vc = c.v_cache_ptrs[attn_idx];
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

        // Self-attention decode — T5 uses relative position bias added to QK^T scores
        // Baseline: ggml_add(kq, kq_b) before softmax, scale=1.0 (not 1/sqrt(d))
        // Position bias is precomputed from attn_rel_b (layer 0 fallback, unidirectional buckets)
        {
            int kv_len = position + 1;
            float alibi_mb = 0; float alibi_m0v = 0, alibi_m1v = 0;
            int alibi_nhl = 0, cur_pos = position;

            // Compute T5 decoder relative position bias — unidirectional buckets
            // Baseline: build_inp_pos_bucket_dec with bidirectional=false
            float * d_rel_bias_buf = b.d_rel_pos_bias;  // preallocated [n_head, max_seq_len]
            const float * rel_bias_ptr = nullptr;
            float softcap = 0.0f;
            if (c.n_rel_attn_bkts > 0 && d_rel_bias_buf) {
                // Get bias table — fallback to layer 0 if current layer doesn't have one
                const void * bias_table = lw.attn_rel_b ? lw.attn_rel_b : c.layers[0].attn_rel_b;
                if (bias_table) {
                    int n_head_v = c.fa_n_q_heads;
                    int n_bkts = c.n_rel_attn_bkts;

                    // Compute bucket indices and gather bias values entirely on GPU
                    {
                        void * args[] = { (void *)&bias_table, (void *)&d_rel_bias_buf,
                                          (void *)&n_bkts, (void *)&kv_len, (void *)&cur_pos };
                        hipModuleLaunchKernel(k.eval_t5_rel_bias_compute,
                                             n_head_v, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                    }
                    rel_bias_ptr = d_rel_bias_buf;
                }
            }

            const int * d_params_nullptr = nullptr;
            void * args[] = { (void *)&b.proj_scratch,
                              (void *)&c.k_cache_ptrs[il], (void *)&c.v_cache_ptrs[il],
                              (void *)&b.attn_out,
                              (void *)&kv_len, (void *)&c.max_seq_len,
                              (void *)&alibi_mb, (void *)&alibi_m0v,
                              (void *)&alibi_m1v, (void *)&alibi_nhl, (void *)&cur_pos,
                              (void *)&rel_bias_ptr,
                              (void *)&softcap, (void *)&d_params_nullptr };
            hipModuleLaunchKernel(k.eval_attention_decode, c.fa_n_q_heads, 1, 1, 32, 4, 1, 0, s, args, nullptr);
        }

        // O projection + residual — baseline line 53-58
        {
            int n = qproj_size, q8blocks = (n+511)/512;
            void * q8args[] = { (void *)&b.attn_out, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
            launch_mv_res(lw.types[6], lw.ptrs[6], lw.strides[6], b.attn_out, b.residual, b.hidden, n, H);
        }

        // Phase 2: Cross-attention — baseline build_t5_dec lines ~80-114
        // Q from decoder hidden state, K/V from encoder output
        // No position bias, no causal mask, scale=1.0
        if (lw.attn_norm_cross && c.encoder_output && c.n_enc_tokens > 0) {
            // RMSNorm on cross_inp
            {
                const void * nw = lw.attn_norm_cross; int n = H;
                void * args[] = { (void *)&b.hidden, (void *)&nw, (void *)&b.norm_out, (void *)&b.residual, (void *)&n };
                hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
            }
            {
                int n = H, blocks = (n+511)/512;
                void * q8args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
                hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
            }
            // Q from decoder (single token)
            quant_and_launch_matvec(lw.wq_cross_type, lw.wq_cross, lw.wq_cross_stride,
                                     b.norm_out, b.proj_scratch, H, qproj_size, s);
            // K from encoder output (n_enc_tokens tokens)
            // We need to project all encoder tokens through wk_cross.
            // Since this is a batch operation on encoder tokens, we use the batch buffers.
            int n_enc = c.n_enc_tokens;
            {
                // Quantize encoder output for K projection
                launch_mmq_quantize(k.eval_quantize_mmq_q8_1_d4, c.encoder_output,
                                    b.batch_q8_mmq, H, n_enc, s);
                // K = wk_cross * encoder_output → batch_kv [n_enc, kv_size]
                batch_projection(lw.wk_cross_type, lw.wk_cross, lw.wk_cross_stride,
                                 c.encoder_output, b.batch_q8_mmq,
                                 b.batch_kv, H, kv_size, n_enc, s);
                // V = wv_cross * encoder_output → batch_kv + kv_size [n_enc, kv_size]
                batch_projection(lw.wv_cross_type, lw.wv_cross, lw.wv_cross_stride,
                                 c.encoder_output, b.batch_q8_mmq,
                                 b.batch_kv + (size_t)n_enc * kv_size, H, kv_size, n_enc, s);
            }
            // Cross-attention: Q [1, n_head, head_dim] attends to K/V [n_enc, n_kv_head, head_dim]
            // This is bidirectional (decoder token attends to all encoder tokens)
            // No position bias, scale=1.0
            // We can reuse eval_attention_decode but with encoder K/V instead of KV cache.
            // Actually we need a cross-attention kernel. For now, compute on the host-side
            // pattern: Q*K^T → softmax → *V, implemented via bidirectional attention with S=1 query
            // and S_enc keys.
            //
            // Use prompt_bidirectional_attn with S=1 query tokens attending to n_enc key tokens.
            // But that kernel expects Q/K/V all with the same S dimension.
            // Instead, do a simple cross-attention: for each head,
            //   attn_score[j] = sum_d Q[d] * K[j,d] for j in 0..n_enc-1
            //   attn_weight = softmax(attn_score * scale)
            //   output[d] = sum_j attn_weight[j] * V[j,d]
            //
            // This is the same as eval_attention_decode but reading from a flat buffer
            // instead of the KV cache. We can write K/V to a temporary KV cache-like layout
            // and call eval_attention_decode.
            //
            // Simpler: write cross K/V to temporary positions in the KV cache and call attention.
            // But that would corrupt the self-attention KV cache.
            //
            // Cleanest approach: use the already-ported prompt_bidirectional_attn kernel.
            // It needs Q [S_q, q_size], K [S_k, kv_size], V [S_k, kv_size] but assumes S_q == S_k.
            //
            // For cross-attention with a single decoder token, we can either:
            // 1. Write a new cross-attention kernel
            // 2. Use eval_attention_decode with a temporary "cache" buffer
            //
            // Option 2 is simpler and correct: write the encoder K/V into a flat buffer laid out
            // like the KV cache (position-major, head-interleaved), then call eval_attention_decode
            // with kv_len = n_enc.
            {
                // eval_attention_decode expects f16 K/V cache. Convert batch_kv f32 → f16.
                // batch_kv is [n_enc, kv_size] f32 for K, then [n_enc, kv_size] f32 for V
                // Convert both to f16 in-place (reuse batch_attn_out as temp f16 buffer)
                float * cross_k_f32 = b.batch_kv;
                float * cross_v_f32 = b.batch_kv + (size_t)n_enc * kv_size;
                // Use the latter half of batch_kv area as f16 storage (f16 is half the size)
                // We need n_enc * kv_size * sizeof(half) * 2 bytes for f16 K and V
                // batch_proj has enough space: [max_batch * max_proj]
                __half * cross_k_f16 = (__half *)b.batch_proj;
                __half * cross_v_f16 = cross_k_f16 + (size_t)n_enc * kv_size;

                // Convert K f32 → f16
                {
                    int N = n_enc * kv_size;
                    float * src = cross_k_f32;
                    __half * dst = cross_k_f16;
                    void * args[] = { (void *)&src, (void *)&dst, (void *)&N };
                    hipModuleLaunchKernel(k.gemm_f32_to_f16, (N + 255) / 256, 1, 1, 256, 1, 1,
                                         0, s, args, nullptr);
                }
                // Convert V f32 → f16
                {
                    int N = n_enc * kv_size;
                    float * src = cross_v_f32;
                    __half * dst = cross_v_f16;
                    void * args[] = { (void *)&src, (void *)&dst, (void *)&N };
                    hipModuleLaunchKernel(k.gemm_f32_to_f16, (N + 255) / 256, 1, 1, 256, 1, 1,
                                         0, s, args, nullptr);
                }

                // Now call eval_attention_decode with f16 K/V
                // Layout: [n_enc, n_kv_head * head_dim] f16, position-major
                // max_seq = n_enc (stride between heads = n_enc * head_dim)
                // For cross-attention: kv_len = n_enc (attends to all encoder tokens)
                float alibi_mb = 0; float alibi_m0v = 0, alibi_m1v = 0;
                int alibi_nhl = 0;
                int cross_kv_len = n_enc;
                int cross_cur_pos = n_enc - 1;
                const float * cross_rel_bias = nullptr;
                float softcap = 0.0f;  // no position bias for cross-attention
                int cross_max_seq = n_enc;
                const int * d_params_nullptr = nullptr;
                void * args[] = { (void *)&b.proj_scratch,
                                  (void *)&cross_k_f16, (void *)&cross_v_f16,
                                  (void *)&b.attn_out,
                                  (void *)&cross_kv_len, (void *)&cross_max_seq,
                                  (void *)&alibi_mb, (void *)&alibi_m0v,
                                  (void *)&alibi_m1v, (void *)&alibi_nhl, (void *)&cross_cur_pos,
                                  (void *)&cross_rel_bias,
                                  (void *)&softcap, (void *)&d_params_nullptr };
                hipModuleLaunchKernel(k.eval_attention_decode, c.fa_n_q_heads, 1, 1, 32, 4, 1, 0, s, args, nullptr);
            }
            // O projection (wo_cross) + residual
            {
                int n = qproj_size, q8blocks = (n+511)/512;
                void * q8args[] = { (void *)&b.attn_out, (void *)&b.q8_act, (void *)&n };
                hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
                launch_matvec_res_typed(lw.wo_cross_type, lw.wo_cross, lw.wo_cross_stride,
                                         b.residual, b.hidden, qproj_size, H, s);
            }
        }

        // Phase 3: FFN — baseline lines 122-138
        // RMSNorm
        {
            int post_norm_idx = 7;
            const void * nw = lw.ptrs[post_norm_idx]; int n = H;
            void * args[] = { (void *)&b.hidden, (void *)&nw, (void *)&b.norm_out, (void *)&b.residual, (void *)&n };
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        }
        {
            int n = H, blocks = (n+511)/512;
            void * q8args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
        }
        // T5 FFN: relu (no gate) or gelu-gated — baseline lines 130-137
        // For T5 without gate: up → relu → down
        // For flan-T5 with gate: gate + up → gelu_mul → down
        {
            int gate_idx = 8, up_idx = 9, down_idx = 10;
            // Up projection
            launch_mv(lw.types[up_idx], lw.ptrs[up_idx], lw.strides[up_idx], b.norm_out, b.proj_scratch, H, FF);
            if (lw.ptrs[gate_idx]) {
                // Gated: gate projection + gelu_mul
                launch_mv(lw.types[gate_idx], lw.ptrs[gate_idx], lw.strides[gate_idx], b.norm_out, b.mlp_inter, H, FF);
                // gelu(gate) * up — baseline uses LLM_FFN_GELU + LLM_FFN_PAR
                {
                    int n = FF;
                    void * args[] = { (void *)&b.mlp_inter, (void *)&b.proj_scratch,
                                      (void *)&b.mlp_inter, (void *)&n };
                    hipModuleLaunchKernel(k.eval_gelu_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }
            } else {
                // No gate: relu(up)
                {
                    int n = FF;
                    void * args[] = { (void *)&b.proj_scratch, (void *)&b.proj_scratch, (void *)&n };
                    hipModuleLaunchKernel(k.eval_relu, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }
                hipMemcpyAsync(b.mlp_inter, b.proj_scratch, FF * sizeof(float), hipMemcpyDeviceToDevice, s);
            }
            // Down + residual
            {
                int n = FF, q8blocks = (n+511)/512;
                void * q8args[] = { (void *)&b.mlp_inter, (void *)&b.q8_act, (void *)&n };
                hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
                launch_mv_res(lw.types[down_idx], lw.ptrs[down_idx], lw.strides[down_idx], b.mlp_inter, b.residual, b.hidden, FF, H);
            }
        }
    }

    // Final norm
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
    launch_mv(c.lm_head_type, c.lm_head_weight, c.lm_head_stride, b.norm_out, b.logits, H, V);
    if (logits_out) {
        hipMemcpyAsync(logits_out, b.logits, V * sizeof(float), hipMemcpyDeviceToHost, s);
    }
    hipStreamSynchronize(s);
    return 0;
}

// ----------------------------------------------------------------------------
// forward_decode_deepseek2_mla — ported from baseline src/models/deepseek2.cpp
// Multi-head Latent Attention: compressed KV with LoRA decomposition.
// ----------------------------------------------------------------------------
