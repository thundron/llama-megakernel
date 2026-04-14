// encoder/bert.cpp — BERT encoder forward (bidirectional attention)
// Ported from baseline src/models/bert.cpp (build_bert)
//
// BERT architecture (post-norm, LayerNorm, bidirectional):
//   Embedding: tok_embd + type_embd[0] + pos_embd[pos] → LayerNorm(tok_norm)
//   Per layer:
//     1. QKV projection (with biases) → bidirectional attention (scale=1/sqrt(d)) →
//        O projection + bo → residual add → LayerNorm(attn_out_norm + attn_out_norm_b)
//     2. FFN: up+up_b → GELU → down+down_b (or SiLU-gated, GEGLU, MoE variants) →
//        residual add → LayerNorm(layer_out_norm + layer_out_norm_b)
//   Output: encoder embeddings (no LM head, no logits)
//
// Also covers: ModernBERT, NomicBERT, NomicBERT-MoE, JinaBERT-V2/V3, EuroBERT, NeoBERT
//
// Key differences from T5 encoder:
//   - LayerNorm (not RMSNorm), post-norm (not pre-norm)
//   - QKV biases always present
//   - Attention scale = 1/sqrt(d_k) (standard, not T5's 1.0)
//   - Optional absolute position embeddings or RoPE (variant-dependent)
//   - Token type embeddings
//   - Embedding LayerNorm before first layer
#include "../gfx1100-internal.h"
#include "../shared/batch-ops.h"

int forward_encode_bert(const int * tokens, int n_tokens, float * embd_out) {
    auto & c = g_config;
    auto & b = g_bufs;
    auto & k = g_compiled;
    hipStream_t stream = b.stream;

    int H = c.hidden_size;
    int FF = c.intermediate_size;
    int S = n_tokens;
    int n_head = c.fa_n_q_heads;
    int n_kv_head = c.fa_n_kv_heads;
    int head_dim = c.fa_head_dim;

    if (S > b.max_batch) {
        fprintf(stderr, "gfx1100: BERT encoder — n_tokens=%d exceeds max_batch=%d\n", S, b.max_batch);
        return -1;
    }

    // =====================================================================
    // Phase 0: Embedding — baseline build_bert lines 1-30
    // tok_embd + type_embd[0] + pos_embd[pos] → LayerNorm(tok_norm)
    // =====================================================================

    // Copy token IDs to device
    hipMemcpyAsync(b.batch_token_ids, tokens, S * sizeof(int), hipMemcpyHostToDevice, stream);

    // Token embedding lookup → batch_hidden [S, H]
    {
        hipFunction_t embed_fn = nullptr;
        int embed_threads = 256;
        switch (c.embed_type) {
            case 12: embed_fn = k.prompt_embed_q4k;  embed_threads = 32; break;
            case 14: embed_fn = k.prompt_embed_q6k;  embed_threads = 64; break;
            case  2: embed_fn = k.prompt_embed_q4_0;  embed_threads = 256; break;
            case  3: embed_fn = k.prompt_embed_q4_1;  embed_threads = 256; break;
            case  6: embed_fn = k.prompt_embed_q5_0;  embed_threads = 256; break;
            case  7: embed_fn = k.prompt_embed_q5_1;  embed_threads = 256; break;
            case  8: embed_fn = k.prompt_embed_q8_0;  embed_threads = 256; break;
            case 10: embed_fn = k.prompt_embed_q2k;  embed_threads = 64; break;
            case 11: embed_fn = k.prompt_embed_q3k;  embed_threads = 64; break;
            case 13: embed_fn = k.prompt_embed_q5k;  embed_threads = 64; break;
            case  0: embed_fn = k.prompt_embed_f32;  embed_threads = 256; break;
            case  1: embed_fn = k.prompt_embed_f16;  embed_threads = 256; break;
            case 30: embed_fn = k.prompt_embed_bf16; embed_threads = 256; break;
            default:
                fprintf(stderr, "gfx1100: BERT encoder — unsupported embed type %d\n", c.embed_type);
                return -1;
        }
        int nb_x = H / 256;
        if (nb_x < 1) nb_x = 1;
        const void * ew = c.embed_weight;
        long long es = c.embed_stride;
        // Kernel: prompt_embed_*(token_ids, embed_weight, embed_stride, output, S)
        void * args[] = { (void *)&b.batch_token_ids, (void *)&ew, (void *)&es,
                          (void *)&b.batch_hidden, (void *)&S };
        hipModuleLaunchKernel(embed_fn, nb_x, S, 1, embed_threads, 1, 1, 0, stream, args, nullptr);
    }

    // Token type embedding: add type_embd row 0 to all tokens
    // Baseline: inpL += ggml_get_rows(type_embd, [0, 0, ..., 0])
    // All tokens get type 0 (Sentence A)
    if (c.type_embd) {
        // Add type_embd[0] (which is just a [H] f32 vector) to each token
        const void * type_row0 = c.type_embd;
        int N = S;
        // Reuse prompt_add_bias kernel: adds bias[i] to each row
        void * args[] = { (void *)&b.batch_hidden, (void *)&type_row0, (void *)&H, (void *)&N };
        hipModuleLaunchKernel(k.prompt_add_bias, (H + 255) / 256, S, 1, 256, 1, 1, 0, stream, args, nullptr);
    }

    // Absolute position embeddings: batch_hidden[s] += pos_embd[s]
    // Baseline: inpL = ggml_add(inpL, ggml_get_rows(pos_embd, inp_pos))
    // For simple sequential positions, pos[s] = s
    if (c.pos_embd && k.prompt_add_pos_embd) {
        // Use proper position embedding kernel: output[s*H + i] += pos_embd[positions[s]*H + i]
        // For BERT, positions are sequential: positions[s] = s
        std::vector<int> h_pos(S);
        for (int i = 0; i < S; i++) h_pos[i] = i;
        hipMemcpyAsync(b.batch_token_ids, h_pos.data(), S * sizeof(int), hipMemcpyHostToDevice, stream);

        int D = H;
        void * args[] = { (void *)&b.batch_hidden, (void *)&c.pos_embd,
                          (void *)&b.batch_token_ids, (void *)&D, (void *)&S };
        hipModuleLaunchKernel(k.prompt_add_pos_embd, (H + 255) / 256, S, 1, 256, 1, 1,
                              0, stream, args, nullptr);
    }

    // Embedding LayerNorm: baseline build_norm(inpL, tok_norm, tok_norm_b, LLM_NORM, 0)
    if (c.tok_norm_weight) {
        const void * w = c.tok_norm_weight;
        const void * bias = c.tok_norm_bias;
        float eps = c.norm_eps;
        int D = H;
        void * args[] = { (void *)&b.batch_hidden, (void *)&w, (void *)&bias,
                          (void *)&b.batch_hidden, (void *)&D, (void *)&eps };
        int threads = (H < 1024) ? 256 : 1024;
        hipModuleLaunchKernel(k.prompt_layernorm, S, 1, 1, threads, 1, 1, 0, stream, args, nullptr);
    }

    // =====================================================================
    // Phase 1: Layer loop — ported from baseline build_bert
    // =====================================================================
    for (int il = 0; il < c.n_layers; il++) {
        const gfx1100_layer_weights & lw = c.layers[il];

        // Save input for residual (pre-norm: residual = input BEFORE attention)
        // BERT is post-norm: residual add happens before LayerNorm
        hipMemcpyAsync(b.batch_residual, b.batch_hidden, (size_t)S * H * sizeof(float),
                       hipMemcpyDeviceToDevice, stream);

        // --- QKV projections with biases ---
        // Baseline: Q = wq*cur + bq, K = wk*cur + bk, V = wv*cur + bv
        int q_size = n_head * head_dim;
        int kv_size = n_kv_head * head_dim;

        // Quantize for projections
        launch_mmq_quantize(k.eval_quantize_mmq_q8_1_d4, b.batch_hidden,
                            b.batch_q8_mmq, H, S, stream);

        // Q → batch_proj [S, q_size]
        batch_projection(lw.types[1], lw.ptrs[1], lw.strides[1],
                         b.batch_hidden, b.batch_q8_mmq,
                         b.batch_proj, H, q_size, S, stream);
        // Add Q bias
        if (lw.bias_q) {
            int N = S;
            void * args[] = { (void *)&b.batch_proj, (void *)&lw.bias_q, (void *)&q_size, (void *)&N };
            hipModuleLaunchKernel(k.prompt_add_bias, (q_size + 255) / 256, S, 1, 256, 1, 1, 0, stream, args, nullptr);
        }

        // K → batch_kv [S, kv_size]
        batch_projection(lw.types[2], lw.ptrs[2], lw.strides[2],
                         b.batch_hidden, b.batch_q8_mmq,
                         b.batch_kv, H, kv_size, S, stream);
        if (lw.bias_k) {
            int N = S;
            void * args[] = { (void *)&b.batch_kv, (void *)&lw.bias_k, (void *)&kv_size, (void *)&N };
            hipModuleLaunchKernel(k.prompt_add_bias, (kv_size + 255) / 256, S, 1, 256, 1, 1, 0, stream, args, nullptr);
        }

        // V → batch_kv + kv_size [S, kv_size]
        float * v_buf = b.batch_kv + (size_t)S * kv_size;
        batch_projection(lw.types[3], lw.ptrs[3], lw.strides[3],
                         b.batch_hidden, b.batch_q8_mmq,
                         v_buf, H, kv_size, S, stream);
        if (lw.bias_v) {
            int N = S;
            void * args[] = { (void *)&v_buf, (void *)&lw.bias_v, (void *)&kv_size, (void *)&N };
            hipModuleLaunchKernel(k.prompt_add_bias, (kv_size + 255) / 256, S, 1, 256, 1, 1, 0, stream, args, nullptr);
        }

        // Encoder RoPE (ModernBERT, NomicBERT) — ported from modern-bert.cpp lines 50-60
        // Applied to Q and K BEFORE QK norm. Positions are sequential: 0, 1, ..., S-1
        if (c.rope_type != 0 && !c.pos_embd && k.prompt_rope_neox_inplace) {
            // Q RoPE: batch_proj [S, q_size] — apply NeoX RoPE
            int q_dim = q_size; // n_head * head_dim
            int rope_dim = c.fa_rope_dim > 0 ? c.fa_rope_dim : head_dim;
            int threads = (rope_dim / 2 < 256) ? rope_dim / 2 : 256;
            float theta = c.fa_rope_theta;
            int sp = 0; // encoder: positions start at 0
            void * q_args[] = { (void *)&b.batch_proj, (void *)&q_dim, (void *)&rope_dim,
                                (void *)&theta, (void *)&sp };
            hipModuleLaunchKernel(k.prompt_rope_neox_inplace, S, 1, 1, threads, 1, 1,
                                 0, stream, q_args, nullptr);
            // K RoPE: batch_kv [S, kv_size]
            int k_dim = kv_size;
            void * k_args[] = { (void *)&b.batch_kv, (void *)&k_dim, (void *)&rope_dim,
                                (void *)&theta, (void *)&sp };
            hipModuleLaunchKernel(k.prompt_rope_neox_inplace, S, 1, 1, threads, 1, 1,
                                 0, stream, k_args, nullptr);
        }

        // QK norm (Jina-BERT-V2, NomicBERT) — per-head LayerNorm on Q and K
        // Baseline jina-bert-v2.cpp: if (attn_q_norm) Q = layernorm(Q, per_head)
        if (lw.attn_q_norm && k.prompt_per_head_layernorm) {
            int total_groups = S * n_head;
            int hd = head_dim;
            float eps = c.norm_eps;
            const void * qnw = lw.attn_q_norm;
            const void * qnb = lw.attn_q_norm_b;
            void * args[] = { (void *)&b.batch_proj, (void *)&qnw, (void *)&qnb,
                              (void *)&b.batch_proj, (void *)&hd, (void *)&eps };
            int threads = (hd < 256) ? hd : 256;
            hipModuleLaunchKernel(k.prompt_per_head_layernorm, total_groups, 1, 1,
                                 threads, 1, 1, 0, stream, args, nullptr);
        }
        if (lw.attn_k_norm && k.prompt_per_head_layernorm) {
            int total_groups = S * n_kv_head;
            int hd = head_dim;
            float eps = c.norm_eps;
            const void * knw = lw.attn_k_norm;
            const void * knb = lw.attn_k_norm_b;
            void * args[] = { (void *)&b.batch_kv, (void *)&knw, (void *)&knb,
                              (void *)&b.batch_kv, (void *)&hd, (void *)&eps };
            int threads = (hd < 256) ? hd : 256;
            hipModuleLaunchKernel(k.prompt_per_head_layernorm, total_groups, 1, 1,
                                 threads, 1, 1, 0, stream, args, nullptr);
        }

        // --- Bidirectional attention (no causal mask, scale=1/sqrt(d)) ---
        {
            float * Q = b.batch_proj;
            float * K = b.batch_kv;
            float * V = v_buf;
            float * out = b.batch_attn_out;
            float scale = 1.0f / sqrtf((float)head_dim);  // BERT: standard 1/sqrt(d) scaling
            const float * kq_b = nullptr;  // no position bias for BERT (uses embeddings/RoPE/ALiBi)
            int nh = n_head;
            int nkv = n_kv_head;
            int hd = head_dim;
            int seq = S;
            void * args[] = { (void *)&Q, (void *)&K, (void *)&V, (void *)&out,
                              (void *)&kq_b, (void *)&scale,
                              (void *)&nh, (void *)&nkv, (void *)&hd, (void *)&seq };
            hipModuleLaunchKernel(k.prompt_bidirectional_attn,
                                 n_head, S, 1, 256, 1, 1, 0, stream, args, nullptr);
        }

        // --- O projection + bias ---
        launch_mmq_quantize(k.eval_quantize_mmq_q8_1_d4, b.batch_attn_out,
                            b.batch_q8_mmq, q_size, S, stream);
        batch_projection(lw.types[6], lw.ptrs[6], lw.strides[6],
                         b.batch_attn_out, b.batch_q8_mmq,
                         b.batch_norm, H, H, S, stream);
        if (lw.bias_o) {
            int N = S;
            void * args[] = { (void *)&b.batch_norm, (void *)&lw.bias_o, (void *)&H, (void *)&N };
            hipModuleLaunchKernel(k.prompt_add_bias, (H + 255) / 256, S, 1, 256, 1, 1, 0, stream, args, nullptr);
        }

        // --- Residual add: cur = O_proj_out + input ---
        {
            int N = S * H;
            void * args[] = { (void *)&b.batch_norm, (void *)&b.batch_residual,
                              (void *)&b.batch_hidden, (void *)&N };
            hipModuleLaunchKernel(k.prompt_add_residual, (N + 255) / 256, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
        }

        // --- Post-attention LayerNorm ---
        // Baseline: build_norm(cur, attn_out_norm, attn_out_norm_b, LLM_NORM, il)
        if (lw.attn_out_norm) {
            const void * w = lw.attn_out_norm;
            const void * bias = lw.attn_out_norm_b;
            float eps = c.norm_eps;
            int D = H;
            void * args[] = { (void *)&b.batch_hidden, (void *)&w, (void *)&bias,
                              (void *)&b.batch_hidden, (void *)&D, (void *)&eps };
            int threads = (H < 1024) ? 256 : 1024;
            hipModuleLaunchKernel(k.prompt_layernorm, S, 1, 1, threads, 1, 1, 0, stream, args, nullptr);
        }

        // Jina-BERT-V2/V3: extra LayerNorm between attention and FFN
        // Baseline jina-bert-v2.cpp: build_norm(cur, attn_norm_2, attn_norm_2_b, LLM_NORM)
        if (lw.attn_norm_2) {
            const void * w = lw.attn_norm_2;
            const void * bias = lw.attn_norm_2_b;
            float eps = c.norm_eps;
            int D = H;
            void * args[] = { (void *)&b.batch_hidden, (void *)&w, (void *)&bias,
                              (void *)&b.batch_hidden, (void *)&D, (void *)&eps };
            int threads = (H < 1024) ? 256 : 1024;
            hipModuleLaunchKernel(k.prompt_layernorm, S, 1, 1, threads, 1, 1, 0, stream, args, nullptr);
        }

        // Save for FFN residual
        hipMemcpyAsync(b.batch_residual, b.batch_hidden, (size_t)S * H * sizeof(float),
                       hipMemcpyDeviceToDevice, stream);

        // --- FFN ---
        // Baseline: four variants depending on arch
        launch_mmq_quantize(k.eval_quantize_mmq_q8_1_d4, b.batch_hidden,
                            b.batch_q8_mmq, H, S, stream);

        // Check which FFN variant: gate present → SiLU-gated (Nomic) or GEGLU (Jina)
        // No gate → sequential GELU (classic BERT)
        if (lw.ptrs[8]) {
            // Has gate weight → parallel FFN (LLM_FFN_PAR)
            // Dispatch by activation type
            // Gate → batch_proj [S, FF]
            batch_projection(lw.types[8], lw.ptrs[8], lw.strides[8],
                             b.batch_hidden, b.batch_q8_mmq,
                             b.batch_proj, H, FF, S, stream);
            // Up → batch_mlp [S, FF]
            batch_projection(lw.types[9], lw.ptrs[9], lw.strides[9],
                             b.batch_hidden, b.batch_q8_mmq,
                             b.batch_mlp, H, FF, S, stream);

            // Apply gate bias if present
            if (lw.ffn_gate_bias) {
                int N = S;
                void * args[] = { (void *)&b.batch_proj, (void *)&lw.ffn_gate_bias, (void *)&FF, (void *)&N };
                hipModuleLaunchKernel(k.prompt_add_bias, (FF + 255) / 256, S, 1, 256, 1, 1, 0, stream, args, nullptr);
            }
            if (lw.ffn_up_bias) {
                int N = S;
                void * args[] = { (void *)&b.batch_mlp, (void *)&lw.ffn_up_bias, (void *)&FF, (void *)&N };
                hipModuleLaunchKernel(k.prompt_add_bias, (FF + 255) / 256, S, 1, 256, 1, 1, 0, stream, args, nullptr);
            }

            // Activation: SiLU(gate)*up or GELU(gate)*up depending on act_type
            {
                int N = S * FF;
                void * args[] = { (void *)&b.batch_proj, (void *)&b.batch_mlp,
                                  (void *)&b.batch_proj, (void *)&N };
                hipFunction_t act_fn;
                switch (c.act_type) {
                    case 0: act_fn = k.prompt_silu_mul; break;  // ACT_SILU (Nomic-BERT)
                    case 1: // ACT_GELU
                    default: act_fn = k.eval_gelu_mul; break;   // GELU (BERT, Jina-V2 GEGLU)
                }
                hipModuleLaunchKernel(act_fn, (N + 255) / 256, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
            }
        } else {
            // No gate → sequential FFN (LLM_FFN_SEQ): up → GELU → (result in batch_proj)
            batch_projection(lw.types[9], lw.ptrs[9], lw.strides[9],
                             b.batch_hidden, b.batch_q8_mmq,
                             b.batch_proj, H, FF, S, stream);
            if (lw.ffn_up_bias) {
                int N = S;
                void * args[] = { (void *)&b.batch_proj, (void *)&lw.ffn_up_bias, (void *)&FF, (void *)&N };
                hipModuleLaunchKernel(k.prompt_add_bias, (FF + 255) / 256, S, 1, 256, 1, 1, 0, stream, args, nullptr);
            }
            // Standalone GELU activation in-place
            // Baseline: LLM_FFN_GELU (tanh-approx) for classic BERT
            // eval_gelu: out[i] = gelu(input[i])
            {
                int N = S * FF;
                void * args[] = { (void *)&b.batch_proj, (void *)&b.batch_proj, (void *)&N };
                hipModuleLaunchKernel(k.eval_gelu, (N + 255) / 256, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
            }
        }

        // Down projection + residual
        launch_mmq_quantize(k.eval_quantize_mmq_q8_1_d4, b.batch_proj,
                            b.batch_q8_mmq, FF, S, stream);
        batch_projection(lw.types[10], lw.ptrs[10], lw.strides[10],
                         b.batch_proj, b.batch_q8_mmq,
                         b.batch_norm, FF, H, S, stream);
        if (lw.ffn_down_bias) {
            int N = S;
            void * args[] = { (void *)&b.batch_norm, (void *)&lw.ffn_down_bias, (void *)&H, (void *)&N };
            hipModuleLaunchKernel(k.prompt_add_bias, (H + 255) / 256, S, 1, 256, 1, 1, 0, stream, args, nullptr);
        }

        // Residual add: cur = ffn_out + ffn_inp
        {
            int N = S * H;
            void * args[] = { (void *)&b.batch_norm, (void *)&b.batch_residual,
                              (void *)&b.batch_hidden, (void *)&N };
            hipModuleLaunchKernel(k.prompt_add_residual, (N + 255) / 256, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
        }

        // --- Post-FFN LayerNorm ---
        // Baseline: build_norm(cur, layer_out_norm, layer_out_norm_b, LLM_NORM, il)
        if (lw.layer_out_norm) {
            const void * w = lw.layer_out_norm;
            const void * bias = lw.layer_out_norm_b;
            float eps = c.norm_eps;
            int D = H;
            void * args[] = { (void *)&b.batch_hidden, (void *)&w, (void *)&bias,
                              (void *)&b.batch_hidden, (void *)&D, (void *)&eps };
            int threads = (H < 1024) ? 256 : 1024;
            hipModuleLaunchKernel(k.prompt_layernorm, S, 1, 1, threads, 1, 1, 0, stream, args, nullptr);
        }
    }

    // =====================================================================
    // Phase 2: Output — embeddings only, no LM head
    // Baseline: res->t_embd = cur (no final norm, already normed per-layer)
    // =====================================================================
    if (embd_out) {
        hipMemcpyAsync(embd_out, b.batch_hidden, (size_t)S * H * sizeof(float),
                       hipMemcpyDeviceToHost, stream);
    }

    hipStreamSynchronize(stream);
    return 0;
}
