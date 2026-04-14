// encoder/t5-enc.cpp — T5 encoder forward (bidirectional attention)
// Ported from baseline src/models/t5-enc.cpp (build_t5_enc)
//
// Per layer:
//   1. RMSNorm (attn_norm_enc) → QKV projections (wq/wk/wv_enc) →
//      bidirectional attention (no causal mask, relative position bias, scale=1.0) →
//      O projection (wo_enc) → residual
//   2. RMSNorm (ffn_norm_enc) →
//      FFN: ReLU (T5) or GELU-gated (Flan-T5) → residual
// Final: RMSNorm (output_norm_enc) → output encoder embeddings (no LM head)
//
// Key differences from decoder:
//   - No KV cache — recomputes everything each pass
//   - Bidirectional attention — all tokens see all tokens
//   - Uses separate encoder weight tensors (wq_enc, etc.)
//   - Attention scale = 1.0 (not 1/sqrt(d_k)) — scale baked into weights
//   - Output: embeddings only, no LM head / logits
//   - Relative position bias from layer 0's attn_rel_b_enc (shared across all layers)
#include "../gfx1100-internal.h"
#include "../shared/batch-ops.h"
#include "../shared/t5-pos-bias.h"

int forward_encode_t5(const int * tokens, int n_tokens, float * embd_enc_out) {
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
    int n_bkts = c.n_rel_attn_bkts;

    if (S > b.max_batch) {
        fprintf(stderr, "gfx1100: T5 encoder — n_tokens=%d exceeds max_batch=%d\n", S, b.max_batch);
        return -1;
    }

    // =====================================================================
    // Phase 0: Batch embedding — same as prompt path
    // =====================================================================
    // Copy token IDs to device
    hipMemcpyAsync(b.batch_token_ids, tokens, S * sizeof(int), hipMemcpyHostToDevice, stream);

    // Batch embed: output → batch_hidden [S, H]
    {
        // Use prompt batch embed kernel for all S tokens
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
                fprintf(stderr, "gfx1100: T5 encoder — unsupported embed type %d\n", c.embed_type);
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

    // =====================================================================
    // Phase 1: Compute relative position bias buckets (CPU, once for all layers)
    // Baseline: build_inp_pos_bucket_enc → [S, S] I32 bucket indices
    // Then: get_rows(attn_rel_b_enc, bucket_indices) → [n_head, S, S] bias
    // =====================================================================
    // For now, compute position buckets on CPU and upload to GPU.
    // The bias is gathered from attn_rel_b_enc[layer 0] using the bucket indices.
    // This is a [n_head, S*S] lookup that we do once before the layer loop.
    //
    // Position bucket: for encoder, bidirectional=true.
    // pos_bucket[i*S + j] = relative_position_bucket(pos[j] - pos[i], n_bkts, true)
    // For a simple sequence, pos[i] = i.
    float * h_rel_bias = nullptr;
    float * d_rel_bias = nullptr;
    if (n_bkts > 0 && c.layers[0].attn_rel_b_enc) {
        // Compute bucket indices on CPU
        std::vector<int> buckets(S * S);
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < S; j++) {
                buckets[i * S + j] = relative_position_bucket(j - i, n_bkts, true);
            }
        }

        // Gather from bias table: attn_rel_b_enc is [n_head, n_bkts] f32
        // For each head h and positions (i,j): bias[h*S*S + i*S + j] = table[h*n_bkts + bucket[i*S+j]]
        h_rel_bias = (float *)malloc(n_head * S * S * sizeof(float));
        const float * table = (const float *)c.layers[0].attn_rel_b_enc;

        // We need to read the table from GPU to CPU first
        std::vector<float> h_table(n_head * n_bkts);
        hipMemcpy(h_table.data(), table, h_table.size() * sizeof(float), hipMemcpyDeviceToHost);

        for (int h = 0; h < n_head; h++) {
            for (int i = 0; i < S; i++) {
                for (int j = 0; j < S; j++) {
                    h_rel_bias[h * S * S + i * S + j] = h_table[h * n_bkts + buckets[i * S + j]];
                }
            }
        }

        // Upload bias to GPU
        hipMalloc(&d_rel_bias, n_head * S * S * sizeof(float));
        hipMemcpyAsync(d_rel_bias, h_rel_bias, n_head * S * S * sizeof(float),
                       hipMemcpyHostToDevice, stream);
        free(h_rel_bias);
    }

    // =====================================================================
    // Phase 2: Layer loop — ported from baseline build_t5_enc
    // =====================================================================
    for (int il = 0; il < c.n_layers; il++) {
        const gfx1100_layer_weights & lw = c.layers[il];

        // Skip layers that don't have encoder weights
        if (!lw.attn_norm_enc) continue;

        // --- Pre-attention RMSNorm ---
        // Baseline: build_norm(inpL, attn_norm_enc, NULL, LLM_NORM_RMS, il)
        // Input: batch_hidden [S, H], Output: batch_norm [S, H], Residual: batch_residual [S, H]
        // Kernel: prompt_rmsnorm(input, weight, output, residual, S, D)
        {
            const void * w = lw.attn_norm_enc;
            int D = H;
            void * args[] = { (void *)&b.batch_hidden, (void *)&w, (void *)&b.batch_norm,
                              (void *)&b.batch_residual, (void *)&S, (void *)&D };
            int threads = (H < 1024) ? 256 : 1024;
            hipModuleLaunchKernel(k.prompt_rmsnorm, S, 1, 1, threads, 1, 1, 0, stream, args, nullptr);
        }

        // --- Quantize normalized tokens for projections ---
        // MMQ Q8_1 quantize: batch_norm [S, H] → batch_q8_mmq
        launch_mmq_quantize(k.eval_quantize_mmq_q8_1_d4, b.batch_norm,
                            b.batch_q8_mmq, H, S, stream);

        // --- QKV projections ---
        // Baseline: Q = wq_enc * cur, K = wk_enc * cur, V = wv_enc * cur (no biases)
        int q_size = n_head * head_dim;
        int kv_size = n_kv_head * head_dim;

        // Q → batch_proj [S, q_size]
        batch_projection(lw.wq_enc_type, lw.wq_enc, lw.wq_enc_stride,
                         b.batch_norm, b.batch_q8_mmq,
                         b.batch_proj, H, q_size, S, stream);

        // K → batch_kv [S, kv_size] (first half)
        batch_projection(lw.wk_enc_type, lw.wk_enc, lw.wk_enc_stride,
                         b.batch_norm, b.batch_q8_mmq,
                         b.batch_kv, H, kv_size, S, stream);

        // V → batch_kv + kv_size [S, kv_size] (second half)
        batch_projection(lw.wv_enc_type, lw.wv_enc, lw.wv_enc_stride,
                         b.batch_norm, b.batch_q8_mmq,
                         b.batch_kv + (size_t)S * kv_size, H, kv_size, S, stream);

        // --- Bidirectional attention with relative position bias ---
        // Baseline: build_attn with kq_scale=1.0f, kq_b=relative_bias, bidirectional mask
        // Uses prompt_bidirectional_attn kernel (already ported)
        //
        // Kernel expects: Q [S, n_head, head_dim], K [S, n_kv_head, head_dim],
        //                 V [S, n_kv_head, head_dim], kq_b [n_head, S, S] or NULL
        // Output: attn_out [S, n_head * head_dim]
        {
            float * Q = b.batch_proj;
            float * K = b.batch_kv;
            float * V = b.batch_kv + (size_t)S * kv_size;
            float * out = b.batch_attn_out;
            float scale = 1.0f;  // T5: no 1/sqrt(d) scaling
            int nh = n_head;
            int nkv = n_kv_head;
            int hd = head_dim;
            int seq = S;
            void * args[] = { (void *)&Q, (void *)&K, (void *)&V, (void *)&out,
                              (void *)&d_rel_bias, (void *)&scale,
                              (void *)&nh, (void *)&nkv, (void *)&hd, (void *)&seq };
            hipModuleLaunchKernel(k.prompt_bidirectional_attn,
                                 n_head, S, 1, 256, 1, 1, 0, stream, args, nullptr);
        }

        // --- O projection + residual ---
        // Baseline: wo_enc * attn_out + residual → hidden
        // Quantize attn_out for O projection
        launch_mmq_quantize(k.eval_quantize_mmq_q8_1_d4, b.batch_attn_out,
                            b.batch_q8_mmq, q_size, S, stream);

        // O proj: batch_attn_out [S, q_size] → batch_hidden [S, H], then add residual
        // Residual is in batch_residual (saved by prompt_rmsnorm above)
        batch_projection_residual(lw.wo_enc_type, lw.wo_enc, lw.wo_enc_stride,
                                  b.batch_attn_out, b.batch_q8_mmq,
                                  b.batch_residual, b.batch_hidden, q_size, H, S, stream);
        // Now batch_hidden = wo * attn_out + residual

        // --- Pre-FFN RMSNorm ---
        // Baseline: build_norm(ffn_inp, ffn_norm_enc, NULL, LLM_NORM_RMS, il)
        // Kernel: prompt_rmsnorm(input, weight, output, residual, S, D)
        {
            const void * w = lw.ffn_norm_enc;
            int D = H;
            void * args[] = { (void *)&b.batch_hidden, (void *)&w, (void *)&b.batch_norm,
                              (void *)&b.batch_residual, (void *)&S, (void *)&D };
            int threads = (H < 1024) ? 256 : 1024;
            hipModuleLaunchKernel(k.prompt_rmsnorm, S, 1, 1, threads, 1, 1, 0, stream, args, nullptr);
        }

        // Re-quantize for FFN projections
        launch_mmq_quantize(k.eval_quantize_mmq_q8_1_d4, b.batch_norm,
                            b.batch_q8_mmq, H, S, stream);

        // --- FFN ---
        // Baseline: two variants based on presence of ffn_gate_enc
        if (lw.ffn_gate_enc) {
            // Flan-T5: gate + up → GELU(gate) * up → down
            // Baseline: LLM_FFN_GELU, LLM_FFN_PAR

            // Gate: [S, FF]
            batch_projection(lw.ffn_gate_enc_type, lw.ffn_gate_enc, lw.ffn_gate_enc_stride,
                             b.batch_norm, b.batch_q8_mmq,
                             b.batch_proj, H, FF, S, stream);

            // Up: [S, FF]
            batch_projection(lw.ffn_up_enc_type, lw.ffn_up_enc, lw.ffn_up_enc_stride,
                             b.batch_norm, b.batch_q8_mmq,
                             b.batch_mlp, H, FF, S, stream);

            // GELU(gate) * up → batch_proj
            // eval_gelu_mul is element-wise: out[i] = gelu(gate[i]) * up[i]
            // Works for any count, not just single-token
            {
                int N = S * FF;
                void * args[] = { (void *)&b.batch_proj, (void *)&b.batch_mlp,
                                  (void *)&b.batch_proj, (void *)&N };
                hipModuleLaunchKernel(k.eval_gelu_mul, (N + 255) / 256, 1, 1, 256, 1, 1,
                                     0, stream, args, nullptr);
            }
        } else {
            // T5: up → ReLU → (in batch_proj)
            // Baseline: LLM_FFN_RELU, LLM_FFN_SEQ

            // Up: [S, FF]
            batch_projection(lw.ffn_up_enc_type, lw.ffn_up_enc, lw.ffn_up_enc_stride,
                             b.batch_norm, b.batch_q8_mmq,
                             b.batch_proj, H, FF, S, stream);

            // ReLU in-place
            {
                int N = S * FF;
                void * args[] = { (void *)&b.batch_proj, (void *)&b.batch_proj, (void *)&N };
                hipModuleLaunchKernel(k.eval_relu, (N + 255) / 256, 1, 1, 256, 1, 1,
                                     0, stream, args, nullptr);
            }
        }

        // Down: [S, H] + residual
        launch_mmq_quantize(k.eval_quantize_mmq_q8_1_d4, b.batch_proj,
                            b.batch_q8_mmq, FF, S, stream);

        // Residual is in batch_residual (saved by pre-FFN prompt_rmsnorm)
        batch_projection_residual(lw.ffn_down_enc_type, lw.ffn_down_enc, lw.ffn_down_enc_stride,
                                  b.batch_proj, b.batch_q8_mmq,
                                  b.batch_residual, b.batch_hidden, FF, H, S, stream);
    }

    // =====================================================================
    // Phase 3: Final RMSNorm → output encoder embeddings
    // Baseline: build_norm(cur, output_norm_enc, NULL, LLM_NORM_RMS, -1)
    // =====================================================================
    if (c.output_norm_enc) {
        const void * w = c.output_norm_enc;
        int D = H;
        // For final norm, residual is not needed — write batch_hidden in-place
        // Kernel: prompt_rmsnorm(input, weight, output, residual, S, D)
        void * args[] = { (void *)&b.batch_hidden, (void *)&w, (void *)&b.batch_hidden,
                          (void *)&b.batch_residual, (void *)&S, (void *)&D };
        int threads = (H < 1024) ? 256 : 1024;
        hipModuleLaunchKernel(k.prompt_rmsnorm, S, 1, 1, threads, 1, 1, 0, stream, args, nullptr);
    }

    // Copy encoder output to host (or to encoder_output buffer for decoder cross-attention)
    if (embd_enc_out) {
        hipMemcpyAsync(embd_enc_out, b.batch_hidden, (size_t)S * H * sizeof(float),
                       hipMemcpyDeviceToHost, stream);
    }

    // Also store on GPU for decoder cross-attention
    if (!c.encoder_output) {
        hipMalloc((void **)&c.encoder_output, (size_t)S * H * sizeof(float));
    }
    hipMemcpyAsync(c.encoder_output, b.batch_hidden, (size_t)S * H * sizeof(float),
                   hipMemcpyDeviceToDevice, stream);
    c.n_enc_tokens = S;

    // Free temporary relative position bias
    if (d_rel_bias) {
        hipFree(d_rel_bias);
    }

    hipStreamSynchronize(stream);
    return 0;
}
