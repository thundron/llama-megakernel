// forward/prompt.cpp — Batch prefill forward
#include "../gfx1100-internal.h"
#include "../shared/batch-ops.h"

int gfx1100_eval_prompt(const int * token_ids, int n_tokens, int start_pos, float * logits_out) {
    if (!g_initialized || !g_compiled.valid) return -1;

    // Single token: use optimized decode path
    if (n_tokens == 1) {
        return gfx1100_eval_decode(token_ids[0], start_pos, logits_out);
    }

    // Batch path — handles any n_tokens >= 2, any start_pos.
    //
    // Arch dispatch: the current batch pipeline (MMQ quantize → rocBLAS GEMM →
    // RoPE → causal attention → SwiGLU FFN) mirrors baseline's src/models/llama.cpp
    // and covers Llama-family dense transformers. Archs with different layer
    // structure (MoE FFN, Mamba SSM, RWKV, T5 enc-dec, DeepSeek MLA, Gemma
    // softcap, Phi parallel FFN) must implement their own forward_prompt_*.
    {
        const int arch = g_config.arch_id;
        bool llama_family = false;
        switch (arch) {
            case ARCH_LLAMA: case ARCH_LLAMA_EMBED: case ARCH_MISTRAL3: case ARCH_MISTRAL4:
            case ARCH_QWEN:  case ARCH_QWEN2: case ARCH_QWEN2VL:
            case ARCH_QWEN3: case ARCH_QWEN3VL: case ARCH_QWEN35:
            case ARCH_ARCEE: case ARCH_BAICHUAN: case ARCH_CODESHELL:
            case ARCH_INTERNLM2: case ARCH_MINICPM: case ARCH_OLMO: case ARCH_OLMO2:
            case ARCH_ORION:  case ARCH_REFACT:   case ARCH_STABLELM:
            case ARCH_XVERSE: case ARCH_EXAONE:   case ARCH_EXAONE4:
            case ARCH_SMOLLM3: case ARCH_DREAM:   case ARCH_DECI:
            case ARCH_GRANITE:   case ARCH_APERTUS:
            case ARCH_ERNIE4_5:  case ARCH_GLM4:
            case ARCH_SEED_OSS:  case ARCH_HUNYUAN_DENSE:
            case ARCH_PANGU_EMBED: case ARCH_MAINCODER:
                llama_family = true; break;
            // Encoder-only architectures — handled separately below
            case ARCH_T5ENCODER:
            case ARCH_BERT: case ARCH_MODERN_BERT:
            case ARCH_NOMIC_BERT: case ARCH_NOMIC_BERT_MOE:
            case ARCH_JINA_BERT_V2: case ARCH_JINA_BERT_V3:
            case ARCH_EUROBERT: case ARCH_NEO_BERT:
            case ARCH_GEMMA_EMBEDDING:
            case ARCH_WAVTOKENIZER_DEC:
                break; // encoder-only / audio decoder, not llama_family
            // Everything else — route through batch llama_family path
            // MoE layers use batch attention + per-token MoE FFN
            // SSM/RWKV layers processed recurrently per-token within the batch layer loop
            default:
                llama_family = true; break;
        }
        if (!llama_family) {
            // Encoder-only or special architectures
            switch (arch) {
                case ARCH_T5ENCODER:
                    return forward_encode_t5(token_ids, n_tokens, logits_out);
                case ARCH_BERT: case ARCH_MODERN_BERT:
                case ARCH_NOMIC_BERT: case ARCH_NOMIC_BERT_MOE:
                case ARCH_JINA_BERT_V2: case ARCH_JINA_BERT_V3:
                case ARCH_EUROBERT: case ARCH_NEO_BERT:
                case ARCH_GEMMA_EMBEDDING:
                    return forward_encode_bert(token_ids, n_tokens, logits_out);
                case ARCH_WAVTOKENIZER_DEC:
                    return forward_decode_wavtokenizer(token_ids, n_tokens, logits_out);
                default:
                    break;
            }
            // Unknown encoder arch — error
            fprintf(stderr, "gfx1100: unsupported encoder arch %d in prompt path\n", arch);
            return -1;
        }
    }

    auto & c = g_config;

    // Parallel attn+FFN (GPT-NeoX, Falcon, GPT-J) handled in the batch path via
    // use_par_res flag in Phase 5 (FFN norm reads from original input) and Phase 7 (three-way add).

    auto & b = g_bufs;
    auto & k = g_compiled;
    auto   s = b.stream;
    int    H = c.hidden_size;
    int    FF = c.intermediate_size;
    int    V = c.vocab_size;
    int    S = n_tokens;

    // If batch exceeds max_batch, process in chunks (not sequential fallback)
    if (S > b.max_batch) {
        int processed = 0;
        while (processed < S) {
            int chunk = (S - processed > b.max_batch) ? b.max_batch : (S - processed);
            float * out = (processed + chunk == S) ? logits_out : nullptr;
            int rc = gfx1100_eval_prompt(token_ids + processed, chunk, start_pos + processed, out);
            if (rc != 0) return rc;
            processed += chunk;
        }
        return 0;
    }

    // Copy token IDs to device
    hipMemcpyAsync(b.batch_token_ids, token_ids, S * sizeof(int), hipMemcpyHostToDevice, s);

    // ---- Batch embedding: token_ids → batch_hidden[S, H] ----
    // Grid: (nb_x, S) where nb_x depends on type; all kernels share same arg layout.
    {
        const void * embed_w = c.embed_weight;
        long long embed_st = c.embed_stride;
        void * args[] = { (void *)&b.batch_token_ids, (void *)&embed_w,
                          (void *)&embed_st, (void *)&b.batch_hidden, (void *)&S };

        hipFunction_t embed_fn = nullptr;
        int embed_threads = 0;
        int embed_nb_x = 0;  // blocks along x (superblocks or flat chunks)
        switch (c.embed_type) {
            // Small-block types: 32 threads, 32 elements per block → nb = H/32 blocks
            case  2: embed_fn = k.prompt_embed_q4_0;    embed_threads = 32;  embed_nb_x = H / 32; break;
            case  3: embed_fn = k.prompt_embed_q4_1;    embed_threads = 32;  embed_nb_x = H / 32; break;
            case  6: embed_fn = k.prompt_embed_q5_0;    embed_threads = 32;  embed_nb_x = H / 32; break;
            case  7: embed_fn = k.prompt_embed_q5_1;    embed_threads = 32;  embed_nb_x = H / 32; break;
            case  8: embed_fn = k.prompt_embed_q8_0;    embed_threads = 32;  embed_nb_x = H / 32; break;
            // K-quant types: 64 threads, QK_K=256
            case 10: embed_fn = k.prompt_embed_q2k;     embed_threads = 64;  embed_nb_x = H / 256; break;
            case 11: embed_fn = k.prompt_embed_q3k;     embed_threads = 64;  embed_nb_x = H / 256; break;
            case 12: embed_fn = k.prompt_embed_q4k;     embed_threads = 32;  embed_nb_x = H / 256; break;
            case 13: embed_fn = k.prompt_embed_q5k;     embed_threads = 64;  embed_nb_x = H / 256; break;
            case 14: embed_fn = k.prompt_embed_q6k;     embed_threads = 64;  embed_nb_x = H / 256; break;
            // IQ types: 32 threads, QK_K=256
            case 16: embed_fn = k.prompt_embed_iq2_xxs; embed_threads = 32;  embed_nb_x = H / 256; break;
            case 17: embed_fn = k.prompt_embed_iq2_xs;  embed_threads = 32;  embed_nb_x = H / 256; break;
            case 18: embed_fn = k.prompt_embed_iq3_xxs; embed_threads = 32;  embed_nb_x = H / 256; break;
            case 19: embed_fn = k.prompt_embed_iq1_s;   embed_threads = 32;  embed_nb_x = H / 256; break;
            case 20: embed_fn = k.prompt_embed_iq4_nl;  embed_threads = 32;  embed_nb_x = H / 32; break;
            case 21: embed_fn = k.prompt_embed_iq3_s;   embed_threads = 32;  embed_nb_x = H / 256; break;
            case 22: embed_fn = k.prompt_embed_iq2_s;   embed_threads = 32;  embed_nb_x = H / 256; break;
            case 23: embed_fn = k.prompt_embed_iq4_xs;  embed_threads = 32;  embed_nb_x = H / 256; break;
            case 29: embed_fn = k.prompt_embed_iq1_m;   embed_threads = 32;  embed_nb_x = H / 256; break;
            case 39: embed_fn = k.prompt_embed_mxfp4;   embed_threads = 32;  embed_nb_x = H / 32; break;
            // NVFP4: 512 elements per superblock
            case 40: embed_fn = k.prompt_embed_nvfp4;   embed_threads = 32;  embed_nb_x = H / 512; break;
            // Float types: flat thread grid along x
            case  0: embed_fn = k.prompt_embed_f32;     embed_threads = 256; embed_nb_x = (H + 255) / 256; break;
            case  1: embed_fn = k.prompt_embed_f16;     embed_threads = 256; embed_nb_x = (H + 255) / 256; break;
            case 30: embed_fn = k.prompt_embed_bf16;    embed_threads = 256; embed_nb_x = (H + 255) / 256; break;
            default:
                fprintf(stderr, "gfx1100: FATAL — unsupported batch embed type %d\n", c.embed_type);
                return -1;
        }
        // Grid is 2D: (embed_nb_x, S) — blockIdx.y = token index
        hipModuleLaunchKernel(embed_fn, embed_nb_x, S, 1, embed_threads, 1, 1, 0, s, args, nullptr);
    }

    // Gemma embedding scale: multiply embeddings by sqrt(n_embd)
    if (c.has_embed_scale) {
        float scale = sqrtf((float)H);
        int N = S * H;
        void * args[] = { (void *)&b.batch_hidden, (void *)&b.batch_hidden, (void *)&scale, (void *)&N };
        hipModuleLaunchKernel(k.eval_scale_scalar, (N+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    }

    // Position embeddings (GPT-2, StarCoder, GPT-J, CodeShell)
    // Baseline gpt2.cpp line 20: inpL = ggml_add(inpL, ggml_get_rows(pos_embd, inp_pos))
    if (c.pos_embd && k.prompt_add_pos_embd) {
        // Fill position IDs on GPU: start_pos, start_pos+1, ..., start_pos+S-1
        {
            void * args[] = { (void *)&b.batch_token_ids, (void *)&start_pos, (void *)&S };
            hipModuleLaunchKernel(k.eval_fill_positions, (S+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
        int D = H;
        void * args[] = { (void *)&b.batch_hidden, (void *)&c.pos_embd,
                          (void *)&b.batch_token_ids, (void *)&D, (void *)&S };
        hipModuleLaunchKernel(k.prompt_add_pos_embd, (H + 255) / 256, S, 1, 256, 1, 1, 0, s, args, nullptr);
    }

    // Token norm (BLOOM — LayerNorm on embeddings before first layer)
    // Baseline bloom.cpp line 18: build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM)
    if (c.tok_norm_weight && k.prompt_layernorm) {
        const void * w = c.tok_norm_weight;
        const void * bias = c.tok_norm_bias;
        float eps = c.norm_eps;
        int D = H;
        void * args[] = { (void *)&b.batch_hidden, (void *)&w, (void *)&bias,
                          (void *)&b.batch_hidden, (void *)&D, (void *)&eps };
        int threads = (H < 1024) ? 256 : 1024;
        hipModuleLaunchKernel(k.prompt_layernorm, S, 1, 1, threads, 1, 1, 0, s, args, nullptr);
    }

    // ---- Layer loop ----
    int attn_idx = 0;
    int dn_idx = 0;

    for (int il = 0; il < c.n_layers; il++) {
        const gfx1100_layer_weights & lw = c.layers[il];

        // Phase 1: Batch RMSNorm → batch_norm + batch_residual
        {
            int D = H;
            int norm_threads = (H < 1024) ? 256 : 1024;
            if (c.has_swin_norm) {
                // Chameleon post-norm: skip pre-attn norm, pass input directly
                hipMemcpyAsync(b.batch_norm, b.batch_hidden, (size_t)S * H * sizeof(float), hipMemcpyDeviceToDevice, s);
                hipMemcpyAsync(b.batch_residual, b.batch_hidden, (size_t)S * H * sizeof(float), hipMemcpyDeviceToDevice, s);
            } else if (c.norm_type == 2 && k.prompt_layernorm) {
                // LayerNorm models (BLOOM, GPT2, Falcon, MPT, Phi2, etc.)
                const void * norm_w = lw.ptrs[0];
                const void * norm_b = lw.attn_norm_bias;
                float eps = c.norm_eps;
                // prompt_layernorm(input, weight, bias, output, D, eps) — grid: (S,1,1)
                // Note: prompt_layernorm doesn't write residual, do it separately
                void * args[] = { (void *)&b.batch_hidden, (void *)&norm_w, (void *)&norm_b,
                                  (void *)&b.batch_norm, (void *)&D, (void *)&eps };
                hipModuleLaunchKernel(k.prompt_layernorm, S, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
                hipMemcpyAsync(b.batch_residual, b.batch_hidden, (size_t)S * H * sizeof(float), hipMemcpyDeviceToDevice, s);
            } else {
                // RMSNorm (default — Llama, Qwen, Gemma, etc.)
                const void * norm_w = lw.ptrs[0];
                void * args[] = { (void *)&b.batch_hidden, (void *)&norm_w,
                                  (void *)&b.batch_norm, (void *)&b.batch_residual, (void *)&S, (void *)&D };
                hipModuleLaunchKernel(k.prompt_rmsnorm, S, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
            }
        }

        // Phase 1b: Batch quantize for MMQ (pick layout based on first weight type this layer uses)
        int first_weight_type = lw.types[1];
        int layout = get_mmq_q8_1_layout(first_weight_type);
        {
            hipFunction_t q8_fn = (layout == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                  (layout == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                                  k.eval_quantize_mmq_q8_1_d2s6;
            launch_mmq_quantize(q8_fn, b.batch_norm, b.batch_q8_mmq, H, S, s);
        }

        // Batch bias/scale helpers — ported from baseline binbcast.cu op_add/op_mul with broadcast
        auto batch_add_bias = [&](float * data, const void * bias, int dim) {
            if (!bias) return;
            int total = S * dim;
            void * ba[] = { (void *)&data, (void *)&bias, (void *)&dim, (void *)&total };
            hipModuleLaunchKernel(k.prompt_add_bias, (total+255)/256, 1, 1, 256, 1, 1, 0, s, ba, nullptr);
        };
        auto batch_mul_scale = [&](float * data, const void * scale, int dim) {
            if (!scale) return;
            int total = S * dim;
            void * ma[] = { (void *)&data, (void *)&scale, (void *)&dim, (void *)&total };
            hipModuleLaunchKernel(k.prompt_elementwise_mul, (total+255)/256, 1, 1, 256, 1, 1, 0, s, ma, nullptr);
        };

        if (c.layer_types[il] == 0) {
            // ---- Attention layer ----
            int qproj_size = c.fa_n_q_heads * c.fa_head_dim;
            if (c.fa_has_gated_attn) qproj_size *= 2;
            int kv_size = c.fa_n_kv_heads * c.fa_head_dim;
            int q_size = c.fa_n_q_heads * c.fa_head_dim;

            // Q projection — baseline line 46: Qcur = build_lora_mm(wq, cur, wq_s); if (bq) Qcur += bq
            batch_projection(lw.types[1], lw.ptrs[1], lw.strides[1],
                             b.batch_norm, b.batch_q8_mmq,
                             b.batch_proj, H, qproj_size, S, s);
            batch_mul_scale(b.batch_proj, lw.scale_q, qproj_size);
            batch_add_bias(b.batch_proj, lw.bias_q, qproj_size);

            // K projection — baseline line 52
            if (lw.types[2] != first_weight_type) {
                int kl = get_mmq_q8_1_layout(lw.types[2]);
                hipFunction_t kq8 = (kl == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                    (kl == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                                k.eval_quantize_mmq_q8_1_d2s6;
                launch_mmq_quantize(kq8, b.batch_norm, b.batch_q8_mmq, H, S, s);
            }
            batch_projection(lw.types[2], lw.ptrs[2], lw.strides[2],
                             b.batch_norm, b.batch_q8_mmq,
                             b.batch_kv, H, kv_size, S, s);
            batch_mul_scale(b.batch_kv, lw.scale_k, kv_size);
            batch_add_bias(b.batch_kv, lw.bias_k, kv_size);

            // V projection — baseline line 58
            if (lw.types[3] != lw.types[2]) {
                int vl = get_mmq_q8_1_layout(lw.types[3]);
                hipFunction_t vq8 = (vl == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                    (vl == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                                k.eval_quantize_mmq_q8_1_d2s6;
                launch_mmq_quantize(vq8, b.batch_norm, b.batch_q8_mmq, H, S, s);
            }
            float * v_out = b.batch_kv + (size_t)S * kv_size;
            batch_projection(lw.types[3], lw.ptrs[3], lw.strides[3],
                             b.batch_norm, b.batch_q8_mmq,
                             v_out, H, kv_size, S, s);
            batch_mul_scale(v_out, lw.scale_v, kv_size);
            batch_add_bias(v_out, lw.bias_v, kv_size);

            // QK Norm + RoPE + KV cache write (batched)
            {
                const void * q_nw = lw.ptrs[4];
                const void * k_nw = lw.ptrs[5];
                void * kc = c.k_cache_ptrs[attn_idx];
                void * vc = c.v_cache_ptrs[attn_idx];
                int max_seq = c.max_seq_len;
                int total_heads = S * (c.fa_n_q_heads + c.fa_n_kv_heads);
                int warps_per_block = 512 / 32; // 16 warps
                int nblocks = (total_heads + warps_per_block - 1) / warps_per_block;
                const void * ff = c.rope_freq_factors;  // per-dim freq factors or NULL
                // Per-layer theta override for iSWA models
                float theta_ovr = (c.layer_use_swa[il] && c.fa_rope_theta_swa > 0)
                                  ? c.fa_rope_theta_swa : 0.0f;
                void * args[] = { (void *)&b.batch_proj, (void *)&b.batch_kv,
                                  (void *)&v_out,
                                  (void *)&q_nw, (void *)&k_nw,
                                  (void *)&kc, (void *)&vc, (void *)&ff,
                                  (void *)&S, (void *)&max_seq,
                                  (void *)&start_pos, (void *)&theta_ovr };
                hipModuleLaunchKernel(k.prompt_qk_norm_rope, nblocks, 1, 1, 512, 1, 1, 0, s, args, nullptr);
            }

            // Causal attention — reads from KV CACHE (f16), supports multi-turn
            // K/V have already been written to cache by prompt_qk_norm_rope above
            {
                int total_q_heads = S * c.fa_n_q_heads;
                int warps_per_block = 512 / 32;
                int nblocks = (total_q_heads + warps_per_block - 1) / warps_per_block;
                void * kc = c.k_cache_ptrs[attn_idx];
                void * vc = c.v_cache_ptrs[attn_idx];
                int max_seq = c.max_seq_len;
                void * args[] = { (void *)&b.batch_proj, (void *)&kc, (void *)&vc,
                                  (void *)&b.batch_attn_out, (void *)&S,
                                  (void *)&start_pos, (void *)&max_seq };
                hipModuleLaunchKernel(k.prompt_causal_attn, nblocks, 1, 1, 512, 1, 1, 0, s, args, nullptr);
            }

            // O projection + residual → batch_hidden
            // Re-quantize attn_out for MMQ
            {
                int o_type = lw.types[6];
                int ol = get_mmq_q8_1_layout(o_type);
                hipFunction_t oq8 = (ol == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                    (ol == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                                k.eval_quantize_mmq_q8_1_d2s6;
                launch_mmq_quantize(oq8, b.batch_attn_out, b.batch_q8_mmq, q_size, S, s);

                if (lw.attn_post_norm || c.has_swin_norm || c.f_residual_scale != 0.0f ||
                    lw.scale_o || lw.bias_o) {
                    // Non-fused: matvec → bo → wo_s → post_norm → residual_scale → residual
                    batch_projection(o_type, lw.ptrs[6], lw.strides[6],
                                     b.batch_attn_out, b.batch_q8_mmq,
                                     b.batch_hidden, q_size, H, S, s);
                    batch_add_bias(b.batch_hidden, lw.bias_o, H);
                    batch_mul_scale(b.batch_hidden, lw.scale_o, H);
                    // Gemma2/3/4 post-attn norm or Chameleon swin_norm
                    if (lw.attn_post_norm || c.has_swin_norm) {
                        const void * pnw = lw.attn_post_norm ? lw.attn_post_norm :
                                           (c.has_swin_norm ? lw.ptrs[0] : nullptr);
                        if (pnw) {
                            float eps = c.norm_eps;
                            int D = H;
                            void * na[] = { (void *)&b.batch_hidden, (void *)&pnw,
                                            (void *)&b.batch_hidden, (void *)&b.batch_hidden, (void *)&S, (void *)&D };
                            int nt = (H < 1024) ? 256 : 1024;
                            hipModuleLaunchKernel(k.prompt_rmsnorm, S, 1, 1, nt, 1, 1, 0, s, na, nullptr);
                        }
                    }
                    // Granite residual scale
                    if (c.f_residual_scale != 0.0f && c.f_residual_scale != 1.0f) {
                        float rs = c.f_residual_scale;
                        int N = S * H;
                        void * sa[] = { (void *)&b.batch_hidden, (void *)&b.batch_hidden, (void *)&rs, (void *)&N };
                        hipModuleLaunchKernel(k.eval_scale_scalar, (N+255)/256, 1, 1, 256, 1, 1, 0, s, sa, nullptr);
                    }
                    // Add residual
                    int N = S * H;
                    void * ra[] = { (void *)&b.batch_hidden, (void *)&b.batch_residual,
                                    (void *)&b.batch_hidden, (void *)&N };
                    hipModuleLaunchKernel(k.prompt_add_residual, (N+255)/256, 1, 1, 256, 1, 1, 0, s, ra, nullptr);
                } else {
                    // Fused path
                    batch_projection_residual(o_type, lw.ptrs[6], lw.strides[6],
                                              b.batch_attn_out, b.batch_q8_mmq,
                                              b.batch_residual, b.batch_hidden,
                                              q_size, H, S, s);
                }
            }

            attn_idx++;

        } else {
            // ---- DeltaNet layer (batch) ----
            int dn_v_size = c.dn_n_heads * c.dn_value_dim;
            int dn_qk_size = c.dn_n_k_heads * c.dn_key_dim;
            int dn_conv_ch = dn_qk_size * 2 + dn_v_size;

            // QKV projection → batch_proj[S, dn_conv_ch]
            batch_projection(lw.types[1], lw.ptrs[1], lw.strides[1],
                             b.batch_norm, b.batch_q8_mmq,
                             b.batch_proj, H, dn_conv_ch, S, s);

            // Z projection → batch_attn_out[S, dn_v_size] (reuse)
            if (lw.types[2] != first_weight_type) {
                int zl = get_mmq_q8_1_layout(lw.types[2]);
                hipFunction_t zq8 = (zl == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                    (zl == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                                k.eval_quantize_mmq_q8_1_d2s6;
                launch_mmq_quantize(zq8, b.batch_norm, b.batch_q8_mmq, H, S, s);
            }
            batch_projection(lw.types[2], lw.ptrs[2], lw.strides[2],
                             b.batch_norm, b.batch_q8_mmq,
                             b.batch_attn_out, H, dn_v_size, S, s);

            // Beta projection → batch_kv[S, dn_n_heads] (reuse batch_kv for small output)
            if (lw.types[3] != lw.types[2]) {
                int bl = get_mmq_q8_1_layout(lw.types[3]);
                hipFunction_t bq8 = (bl == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                    (bl == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                                k.eval_quantize_mmq_q8_1_d2s6;
                launch_mmq_quantize(bq8, b.batch_norm, b.batch_q8_mmq, H, S, s);
            }
            batch_projection(lw.types[3], lw.ptrs[3], lw.strides[3],
                             b.batch_norm, b.batch_q8_mmq,
                             b.batch_kv, H, c.dn_n_heads, S, s);

            // Alpha projection → batch_kv + offset[S, dn_n_heads]
            if (lw.types[4] != lw.types[3]) {
                int al = get_mmq_q8_1_layout(lw.types[4]);
                hipFunction_t aq8 = (al == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                    (al == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                                k.eval_quantize_mmq_q8_1_d2s6;
                launch_mmq_quantize(aq8, b.batch_norm, b.batch_q8_mmq, H, S, s);
            }
            float * alpha_out = b.batch_kv + (size_t)S * c.dn_n_heads;
            batch_projection(lw.types[4], lw.ptrs[4], lw.strides[4],
                             b.batch_norm, b.batch_q8_mmq,
                             alpha_out, H, c.dn_n_heads, S, s);

            // DeltaNet recurrence (batched: processes all S tokens per head)
            // The prompt_deltanet_recurrence kernel handles conv1d + L2 norm + recurrence + gated norm
            {
                float * layer_state = b.dn_states + (long long)dn_idx * c.dn_n_heads * c.dn_key_dim * c.dn_value_dim;
                float * layer_conv = b.conv_bufs + (long long)dn_idx * dn_conv_ch * c.dn_conv_kernel;
                const void * conv_w = lw.ptrs[5];
                const void * ssm_a = lw.ptrs[6];
                const void * dt_bias = lw.ptrs[7];
                const void * norm_w = lw.ptrs[8];
                // batch_proj = QKV[S, dn_conv_ch], batch_attn_out = Z[S, dn_v_size]
                // batch_kv = Beta[S, dn_n_heads], alpha_out = Alpha[S, dn_n_heads]
                // Output → batch_mlp (reuse, [S, dn_v_size])
                void * args[] = {
                    (void *)&b.batch_proj,        // qkv_proj [S, DN_CONV_CH]
                    (void *)&b.batch_attn_out,    // z_proj [S, DN_V_SIZE]
                    (void *)&b.batch_kv,          // beta_proj [S, DN_N_HEADS]
                    (void *)&alpha_out,            // alpha_proj [S, DN_N_HEADS]
                    (void *)&conv_w,               // conv weight
                    (void *)&ssm_a,                // SSM A
                    (void *)&dt_bias,              // dt bias
                    (void *)&norm_w,               // norm weight
                    (void *)&layer_state,          // persistent state
                    (void *)&layer_conv,           // conv buffer
                    (void *)&b.batch_mlp,          // output [S, DN_V_SIZE]
                    (void *)&S,
                };
                hipModuleLaunchKernel(k.prompt_deltanet, c.dn_n_heads, 1, 1, 512, 1, 1, 0, s, args, nullptr);
            }

            // O projection + residual → batch_hidden
            {
                int o_type = lw.types[9];
                int ol = get_mmq_q8_1_layout(o_type);
                hipFunction_t oq8 = (ol == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                    (ol == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                                k.eval_quantize_mmq_q8_1_d2s6;
                launch_mmq_quantize(oq8, b.batch_mlp, b.batch_q8_mmq, dn_v_size, S, s);

                batch_projection_residual(o_type, lw.ptrs[9], lw.strides[9],
                                          b.batch_mlp, b.batch_q8_mmq,
                                          b.batch_residual, b.batch_hidden,
                                          dn_v_size, H, S, s);
            }

            dn_idx++;
        }

        // Phase 5: Pre-FFN norm → batch_norm + batch_residual (or skip for swin_norm)
        {
            int D = H;
            int norm_threads = (H < 1024) ? 256 : 1024;
            if (c.has_swin_norm) {
                // Chameleon post-norm: skip pre-FFN norm
                hipMemcpyAsync(b.batch_norm, b.batch_hidden, (size_t)S * H * sizeof(float), hipMemcpyDeviceToDevice, s);
                hipMemcpyAsync(b.batch_residual, b.batch_hidden, (size_t)S * H * sizeof(float), hipMemcpyDeviceToDevice, s);
            } else if (c.norm_type == 2 && k.prompt_layernorm) {
                int post_norm_idx = (c.layer_types[il] == 0) ? 7 : 10;
                const void * norm_w = lw.ptrs[post_norm_idx];
                const void * norm_b = lw.ffn_norm_bias;
                float eps = c.norm_eps;
                void * args[] = { (void *)&b.batch_hidden, (void *)&norm_w, (void *)&norm_b,
                                  (void *)&b.batch_norm, (void *)&D, (void *)&eps };
                hipModuleLaunchKernel(k.prompt_layernorm, S, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
                hipMemcpyAsync(b.batch_residual, b.batch_hidden, (size_t)S * H * sizeof(float), hipMemcpyDeviceToDevice, s);
            } else {
                int post_norm_idx = (c.layer_types[il] == 0) ? 7 : 10;
                const void * norm_w = lw.ptrs[post_norm_idx];
                void * args[] = { (void *)&b.batch_hidden, (void *)&norm_w,
                                  (void *)&b.batch_norm, (void *)&b.batch_residual, (void *)&S, (void *)&D };
                hipModuleLaunchKernel(k.prompt_rmsnorm, S, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
            }
        }

        // MoE check: if this layer has a MoE router (ffn_gate_inp), use batched
        // group-by-expert dispatch. Route all S tokens, group by expert, then run
        // batched GEMM per expert instead of per-token sequential MoE.
        if (lw.ffn_gate_inp) {
            int ne = c.moe_n_experts;
            int n_used = c.moe_n_experts_used;

            // Step 1: Batch router — project all S tokens through gate_inp → [S, n_expert]
            // Quantize batch_norm for router weight type
            {
                int rl = get_mmq_q8_1_layout(lw.ffn_gate_inp_type);
                hipFunction_t rq8 = (rl == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                    (rl == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                                k.eval_quantize_mmq_q8_1_d2s6;
                launch_mmq_quantize(rq8, b.batch_norm, b.batch_q8_mmq, H, S, s);
            }
            batch_projection(lw.ffn_gate_inp_type, lw.ffn_gate_inp, lw.ffn_gate_inp_stride,
                             b.batch_norm, b.batch_q8_mmq,
                             b.batch_moe_probs, H, ne, S, s);

            // Step 2: Batch softmax/sigmoid — S rows of n_expert columns each
            if (lw.moe_gating_op == 2) { // sigmoid
                int total_ne = S * ne;
                void * args[] = { (void *)&b.batch_moe_probs, (void *)&b.batch_moe_probs, (void *)&total_ne };
                hipModuleLaunchKernel(k.eval_sigmoid, (total_ne+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            } else { // softmax — one block per row
                float sv = 1.0f;
                void * args[] = { (void *)&b.batch_moe_probs, (void *)&b.batch_moe_probs, (void *)&ne, (void *)&sv };
                hipModuleLaunchKernel(k.eval_softmax_row, S, 1, 1, 256, 1, 1,
                                     (256/32) * sizeof(float), s, args, nullptr);
            }

            // Step 3: Batch argsort — S rows, each block sorts n_expert elements
            {
                int npad = 1;
                while (npad < ne) npad *= 2;
                void * args[] = { (void *)&b.batch_moe_probs, (void *)&b.batch_moe_sorted, (void *)&ne, (void *)&npad };
                hipModuleLaunchKernel(k.eval_argsort_desc, S, 1, 1, npad, 1, 1,
                                     npad * sizeof(int), s, args, nullptr);
            }

            // Step 4: Batch normalize routing weights
            {
                int do_norm = lw.moe_norm_w ? 1 : 0;
                float ws = lw.moe_w_scale;
                void * nargs[] = { (void *)&b.batch_moe_probs, (void *)&b.batch_moe_sorted,
                                   (void *)&ne, (void *)&n_used, (void *)&do_norm, (void *)&ws };
                hipModuleLaunchKernel(k.eval_moe_batch_normalize_weights, S, 1, 1, 1, 1, 1, 0, s, nargs, nullptr);
            }

            // Step 5: Group tokens by expert on GPU
            {
                void * args[] = { (void *)&b.batch_moe_sorted, (void *)&b.moe_expert_counts,
                                  (void *)&b.moe_token_map, (void *)&S, (void *)&ne, (void *)&n_used };
                hipModuleLaunchKernel(k.eval_moe_group_tokens, 1, 1, 1, 1, 1, 1, 0, s, args, nullptr);
            }

            // Read expert counts to host for the per-expert loop
            int h_expert_counts[256];
            hipMemcpyAsync(h_expert_counts, b.moe_expert_counts, ne * sizeof(int), hipMemcpyDeviceToHost, s);
            hipStreamSynchronize(s);

            // Zero the MoE output accumulator — batch_hidden is repurposed
            hipMemsetAsync(b.batch_hidden, 0, (size_t)S * H * sizeof(float), s);

            // Expert strides
            long long gate_exp_stride = (long long)FF * lw.ffn_gate_exps_stride;
            long long up_exp_stride   = (long long)FF * lw.ffn_up_exps_stride;
            long long down_exp_stride = (long long)H  * lw.ffn_down_exps_stride;

            // Step 6: Per-expert batched FFN — iterate over experts, run batched GEMM
            int map_offset = 0;
            for (int eid = 0; eid < ne; eid++) {
                int n_tok = h_expert_counts[eid];
                if (n_tok == 0) continue;

                // Compute expert weight pointers
                const void * gate_w = (const char *)lw.ffn_gate_exps + (long long)eid * gate_exp_stride;
                const void * up_w   = (const char *)lw.ffn_up_exps   + (long long)eid * up_exp_stride;
                const void * down_w = (const char *)lw.ffn_down_exps + (long long)eid * down_exp_stride;

                // Gather: copy assigned tokens' norm_out into batch_attn_out [n_tok, H] contiguously
                {
                    int mo = map_offset;
                    void * args[] = { (void *)&b.batch_norm, (void *)&b.batch_attn_out,
                                      (void *)&b.moe_token_map, (void *)&mo, (void *)&H };
                    hipModuleLaunchKernel(k.eval_moe_gather, n_tok, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }

                // Quantize gathered input for MMQ
                {
                    int gl = get_mmq_q8_1_layout(lw.ffn_gate_exps_type);
                    hipFunction_t gq8 = (gl == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                        (gl == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                                    k.eval_quantize_mmq_q8_1_d2s6;
                    launch_mmq_quantize(gq8, b.batch_attn_out, b.batch_q8_mmq, H, n_tok, s);
                }

                // Gate GEMM: [FF, H] x [H, n_tok] → batch_mlp [n_tok, FF]
                batch_projection(lw.ffn_gate_exps_type, gate_w, lw.ffn_gate_exps_stride,
                                 b.batch_attn_out, b.batch_q8_mmq,
                                 b.batch_mlp, H, FF, n_tok, s);

                // Up GEMM: [FF, H] x [H, n_tok] → batch_proj [n_tok, FF]
                // Re-quantize if up weight type differs from gate
                if (lw.ffn_up_exps_type != lw.ffn_gate_exps_type) {
                    int ul = get_mmq_q8_1_layout(lw.ffn_up_exps_type);
                    hipFunction_t uq8 = (ul == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                        (ul == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                                    k.eval_quantize_mmq_q8_1_d2s6;
                    launch_mmq_quantize(uq8, b.batch_attn_out, b.batch_q8_mmq, H, n_tok, s);
                }
                batch_projection(lw.ffn_up_exps_type, up_w, lw.ffn_up_exps_stride,
                                 b.batch_attn_out, b.batch_q8_mmq,
                                 b.batch_proj, H, FF, n_tok, s);

                // Activation: act_fn(gate) * up → batch_mlp [n_tok * FF]
                {
                    int N = n_tok * FF;
                    void * args[] = { (void *)&b.batch_mlp, (void *)&b.batch_proj,
                                      (void *)&b.batch_mlp, (void *)&N };
                    hipFunction_t act_fn;
                    switch (c.act_type) {
                        case ACT_GELU:       act_fn = k.eval_gelu_erf_mul; break;
                        case ACT_GELU_TANH:  act_fn = k.eval_gelu_mul; break;
                        case ACT_GELU_QUICK: act_fn = k.eval_gelu_quick_mul; break;
                        case ACT_RELU2:      act_fn = k.eval_relu2_mul; break;
                        case ACT_SILU:
                        default:             act_fn = k.eval_silu_mul; break;
                    }
                    hipModuleLaunchKernel(act_fn, (N+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }

                // Quantize activation output for down projection
                {
                    int dl = get_mmq_q8_1_layout(lw.ffn_down_exps_type);
                    hipFunction_t dq8 = (dl == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                        (dl == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                                    k.eval_quantize_mmq_q8_1_d2s6;
                    launch_mmq_quantize(dq8, b.batch_mlp, b.batch_q8_mmq, FF, n_tok, s);
                }

                // Down GEMM: [H, FF] x [FF, n_tok] → batch_proj [n_tok, H]
                batch_projection(lw.ffn_down_exps_type, down_w, lw.ffn_down_exps_stride,
                                 b.batch_mlp, b.batch_q8_mmq,
                                 b.batch_proj, FF, H, n_tok, s);

                // Scatter-weighted-add: accumulate into batch_hidden with routing weights
                {
                    int mo = map_offset;
                    void * args[] = { (void *)&b.batch_hidden, (void *)&b.batch_proj,
                                      (void *)&b.batch_moe_probs, (void *)&b.batch_moe_sorted,
                                      (void *)&b.moe_token_map, (void *)&mo,
                                      (void *)&ne, (void *)&H };
                    hipModuleLaunchKernel(k.eval_moe_scatter_weighted_add, n_tok, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }

                map_offset += n_tok;
            }

            // Qwen2MoE shared expert — batch version
            if (lw.ffn_gate_inp_shexp && lw.ffn_gate_shexp && lw.ffn_up_shexp && lw.ffn_down_shexp) {
                // Shared expert gate: sigmoid(gate_inp_shexp @ batch_norm) → [S, 1] scalars
                // Use batch_projection to get [S, 1] output into batch_kv (just S floats)
                {
                    int rl = get_mmq_q8_1_layout(lw.ffn_gate_inp_shexp_type);
                    hipFunction_t rq8 = (rl == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                        (rl == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                                    k.eval_quantize_mmq_q8_1_d2s6;
                    launch_mmq_quantize(rq8, b.batch_norm, b.batch_q8_mmq, H, S, s);
                }
                float * shexp_gates = b.batch_kv;  // reuse batch_kv for [S] gate scalars
                batch_projection(lw.ffn_gate_inp_shexp_type, lw.ffn_gate_inp_shexp,
                                 lw.ffn_gate_inp_shexp_stride,
                                 b.batch_norm, b.batch_q8_mmq,
                                 shexp_gates, H, 1, S, s);
                // Sigmoid on [S] gate values
                {
                    void * args[] = { (void *)&shexp_gates, (void *)&shexp_gates, (void *)&S };
                    hipModuleLaunchKernel(k.eval_sigmoid, (S+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }

                // Shared expert SwiGLU FFN — batch version
                // Gate projection: [FF, H] x [H, S] → batch_mlp [S, FF]
                {
                    int gl = get_mmq_q8_1_layout(lw.ffn_gate_shexp_type);
                    hipFunction_t gq8 = (gl == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                        (gl == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                                    k.eval_quantize_mmq_q8_1_d2s6;
                    launch_mmq_quantize(gq8, b.batch_norm, b.batch_q8_mmq, H, S, s);
                }
                batch_projection(lw.ffn_gate_shexp_type, lw.ffn_gate_shexp, lw.ffn_gate_shexp_stride,
                                 b.batch_norm, b.batch_q8_mmq,
                                 b.batch_mlp, H, FF, S, s);

                // Up projection: [FF, H] x [H, S] → batch_proj [S, FF]
                if (lw.ffn_up_shexp_type != lw.ffn_gate_shexp_type) {
                    int ul = get_mmq_q8_1_layout(lw.ffn_up_shexp_type);
                    hipFunction_t uq8 = (ul == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                        (ul == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                                    k.eval_quantize_mmq_q8_1_d2s6;
                    launch_mmq_quantize(uq8, b.batch_norm, b.batch_q8_mmq, H, S, s);
                }
                batch_projection(lw.ffn_up_shexp_type, lw.ffn_up_shexp, lw.ffn_up_shexp_stride,
                                 b.batch_norm, b.batch_q8_mmq,
                                 b.batch_proj, H, FF, S, s);

                // Activation: silu(gate) * up → batch_mlp [S * FF]
                {
                    int N = S * FF;
                    void * args[] = { (void *)&b.batch_mlp, (void *)&b.batch_proj,
                                      (void *)&b.batch_mlp, (void *)&N };
                    hipModuleLaunchKernel(k.eval_silu_mul, (N+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }

                // Down projection: [H, FF] x [FF, S] → batch_proj [S, H]
                {
                    int dl = get_mmq_q8_1_layout(lw.ffn_down_shexp_type);
                    hipFunction_t dq8 = (dl == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                        (dl == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                                    k.eval_quantize_mmq_q8_1_d2s6;
                    launch_mmq_quantize(dq8, b.batch_mlp, b.batch_q8_mmq, FF, S, s);
                }
                batch_projection(lw.ffn_down_shexp_type, lw.ffn_down_shexp, lw.ffn_down_shexp_stride,
                                 b.batch_mlp, b.batch_q8_mmq,
                                 b.batch_proj, FF, H, S, s);

                // Apply per-token gate: batch_proj[t] *= shexp_gates[t], then accumulate
                for (int t = 0; t < S; t++) {
                    float * tok_out = b.batch_proj + (size_t)t * H;
                    float * tok_gate = shexp_gates + t;
                    int n = H;
                    void * args[] = { (void *)&tok_out, (void *)&tok_gate, (void *)&n };
                    hipModuleLaunchKernel(k.eval_moe_gate_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }
                // Accumulate shared expert output into batch_hidden
                {
                    int N = S * H;
                    void * args[] = { (void *)&b.batch_hidden, (void *)&b.batch_proj,
                                      (void *)&b.batch_hidden, (void *)&N };
                    hipModuleLaunchKernel(k.prompt_add_residual, (N+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }
            }

            // Add residual: batch_hidden += batch_residual
            {
                int N = S * H;
                void * args[] = { (void *)&b.batch_hidden, (void *)&b.batch_residual,
                                  (void *)&b.batch_hidden, (void *)&N };
                hipModuleLaunchKernel(k.prompt_add_residual, (N+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }
            continue; // skip the standard batch FFN below
        }

        // Phase 5b: Batch quantize for MLP
        {
            int gate_idx = (c.layer_types[il] == 0) ? 8 : 11;
            int gate_type = lw.types[gate_idx];
            int gl = get_mmq_q8_1_layout(gate_type);
            hipFunction_t gq8 = (gl == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                (gl == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                            k.eval_quantize_mmq_q8_1_d2s6;
            launch_mmq_quantize(gq8, b.batch_norm, b.batch_q8_mmq, H, S, s);
        }

        // Phase 6: Gate + Up + SiLU
        {
            int gate_idx = (c.layer_types[il] == 0) ? 8 : 11;
            int up_idx = gate_idx + 1;

            // Gate projection — baseline line 116: ffn_gate + ffn_gate_b + ffn_gate_s
            batch_projection(lw.types[gate_idx], lw.ptrs[gate_idx], lw.strides[gate_idx],
                             b.batch_norm, b.batch_q8_mmq,
                             b.batch_mlp, H, FF, S, s);
            batch_add_bias(b.batch_mlp, lw.ffn_gate_bias, FF);
            batch_mul_scale(b.batch_mlp, lw.ffn_gate_scale, FF);

            // Up projection — baseline line 115: ffn_up + ffn_up_b + ffn_up_s
            if (lw.types[up_idx] != lw.types[gate_idx]) {
                int ul = get_mmq_q8_1_layout(lw.types[up_idx]);
                hipFunction_t uq8 = (ul == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                    (ul == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                                k.eval_quantize_mmq_q8_1_d2s6;
                launch_mmq_quantize(uq8, b.batch_norm, b.batch_q8_mmq, H, S, s);
            }
            batch_projection(lw.types[up_idx], lw.ptrs[up_idx], lw.strides[up_idx],
                             b.batch_norm, b.batch_q8_mmq,
                             b.batch_proj, H, FF, S, s);
            batch_add_bias(b.batch_proj, lw.ffn_up_bias, FF);
            batch_mul_scale(b.batch_proj, lw.ffn_up_scale, FF);

            // SiLU(gate) * up → batch_mlp
            {
                int N = S * FF;
                void * args[] = { (void *)&b.batch_mlp, (void *)&b.batch_proj,
                                  (void *)&b.batch_mlp, (void *)&N };
                // Dispatch activation by act_type — not always SiLU
                hipFunction_t act_fn;
                switch (c.act_type) {
                    case ACT_GELU:       act_fn = k.eval_gelu_erf_mul; break;
                    case ACT_GELU_TANH:  act_fn = k.eval_gelu_mul; break;
                    case ACT_GELU_QUICK: act_fn = k.eval_gelu_quick_mul; break;
                    case ACT_RELU2:      act_fn = k.eval_relu2_mul; break;
                    case ACT_SILU:
                    default:             act_fn = k.prompt_silu_mul; break;
                }
                hipModuleLaunchKernel(act_fn, (N + 255) / 256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }
        }

        // Phase 7: Down projection + residual → batch_hidden
        {
            int down_idx = (c.layer_types[il] == 0) ? 10 : 13;
            int down_type = lw.types[down_idx];
            int dl = get_mmq_q8_1_layout(down_type);
            hipFunction_t dq8 = (dl == 0) ? k.eval_quantize_mmq_q8_1_d4 :
                                (dl == 1) ? k.eval_quantize_mmq_q8_1_ds4 :
                                            k.eval_quantize_mmq_q8_1_d2s6;
            launch_mmq_quantize(dq8, b.batch_mlp, b.batch_q8_mmq, FF, S, s);

            if (lw.ffn_post_norm || c.has_swin_norm || c.f_residual_scale != 0.0f ||
                lw.ffn_down_scale || lw.ffn_down_bias) {
                // Non-fused: down → bias → scale → post_norm → residual_scale → residual
                batch_projection(down_type, lw.ptrs[down_idx], lw.strides[down_idx],
                                 b.batch_mlp, b.batch_q8_mmq,
                                 b.batch_hidden, FF, H, S, s);
                batch_add_bias(b.batch_hidden, lw.ffn_down_bias, H);
                batch_mul_scale(b.batch_hidden, lw.ffn_down_scale, H);
                // Post-FFN norm (Gemma2/3/4 or Chameleon swin_norm)
                if (lw.ffn_post_norm || c.has_swin_norm) {
                    const void * pfnw = lw.ffn_post_norm ? lw.ffn_post_norm :
                                        (c.has_swin_norm ? lw.ptrs[7] : nullptr);
                    if (pfnw) {
                        int D = H;
                        void * na[] = { (void *)&b.batch_hidden, (void *)&pfnw,
                                        (void *)&b.batch_hidden, (void *)&b.batch_hidden, (void *)&S, (void *)&D };
                        int nt = (H < 1024) ? 256 : 1024;
                        hipModuleLaunchKernel(k.prompt_rmsnorm, S, 1, 1, nt, 1, 1, 0, s, na, nullptr);
                    }
                }
                // Granite residual scale
                if (c.f_residual_scale != 0.0f && c.f_residual_scale != 1.0f) {
                    float rs = c.f_residual_scale;
                    int N2 = S * H;
                    void * sa[] = { (void *)&b.batch_hidden, (void *)&b.batch_hidden, (void *)&rs, (void *)&N2 };
                    hipModuleLaunchKernel(k.eval_scale_scalar, (N2+255)/256, 1, 1, 256, 1, 1, 0, s, sa, nullptr);
                }
                int N = S * H;
                void * ra[] = { (void *)&b.batch_hidden, (void *)&b.batch_residual,
                                (void *)&b.batch_hidden, (void *)&N };
                hipModuleLaunchKernel(k.prompt_add_residual, (N+255)/256, 1, 1, 256, 1, 1, 0, s, ra, nullptr);
            } else {
                // Fused path
                batch_projection_residual(down_type, lw.ptrs[down_idx], lw.strides[down_idx],
                                          b.batch_mlp, b.batch_q8_mmq,
                                          b.batch_residual, b.batch_hidden,
                                          FF, H, S, s);
            }
        }
    }

    // ---- Final norm (last token only) → norm_out ----
    // Baseline norm.cu: block_size = (ncols < 1024) ? 256 : 1024
    {
        int norm_threads = (H < 1024) ? 256 : 1024;
        if (c.norm_type == 2 && k.prompt_layernorm) {
            // LayerNorm models (BLOOM, GPT2, Falcon, MPT, Phi2, etc.)
            // prompt_layernorm processes rows: pass pointer to last token
            float * last_token = b.batch_hidden + (size_t)(S - 1) * H;
            const void * w = c.final_norm_weight;
            const void * bias = c.final_norm_bias;
            float eps = c.norm_eps;
            int D = H;
            void * args[] = { (void *)&last_token, (void *)&w, (void *)&bias,
                              (void *)&b.norm_out, (void *)&D, (void *)&eps };
            hipModuleLaunchKernel(k.prompt_layernorm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        } else {
            // RMSNorm (default — Llama, Qwen, Gemma, etc.)
            void * args[] = { (void *)&b.batch_hidden, (void *)&c.final_norm_weight,
                              (void *)&b.norm_out, (void *)&S };
            hipModuleLaunchKernel(k.prompt_final_norm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        }
    }

    // ---- LM head (single token from norm_out, use existing eval path) ----
    // Quantize norm_out for matvec
    {
        int n = H;
        int blocks = (n + 511) / 512;
        void * args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
        hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, args, nullptr);
    }

    // LM head matvec (type-dispatched, single token)
    {
        auto is_float_type = [](int type) -> bool {
            return type == 0 || type == 1 || type == 30;
        };
        auto pick_matvec = [&](int type) -> hipFunction_t {
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
                case 16: return k.eval_matvec_iq2_xxs;
                case 17: return k.eval_matvec_iq2_xs;
                case 18: return k.eval_matvec_iq3_xxs;
                case 19: return k.eval_matvec_iq1_s;
                case 20: return k.eval_matvec_iq4_nl;
                case 21: return k.eval_matvec_iq3_s;
                case 22: return k.eval_matvec_iq2_s;
                case 23: return k.eval_matvec_iq4_xs;
                case 29: return k.eval_matvec_iq1_m;
                case 39: return k.eval_matvec_mxfp4;
                case 40: return k.eval_matvec_nvfp4;
                default: return k.eval_matvec_q4k;
            }
        };
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
        bool is_f = is_float_type(c.lm_head_type);
        const void * input = is_f ? (const void *)b.norm_out : (const void *)b.q8_act;
        const void * w = c.lm_head_weight;
        long long st = c.lm_head_stride;
        void * args[] = { (void *)&w, (void *)&st, (void *)&input,
                          (void *)&b.logits, (void *)&H, (void *)&V };
        if (is_f) {
            hipModuleLaunchKernel(pick_matvec(c.lm_head_type), V, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        } else {
            hipFunction_t fn8w = pick_matvec_8w(c.lm_head_type);
            if (fn8w) {
                hipModuleLaunchKernel(fn8w, V, 1, 1, 32, 8, 1, 0, s, args, nullptr);
            } else {
                hipModuleLaunchKernel(pick_matvec(c.lm_head_type), V, 1, 1, 32, 4, 1, 0, s, args, nullptr);
            }
        }
    }

    // Logit scale (Cohere2, Command-R, Granite, Grok)
    if (c.f_logit_scale != 0.0f && c.f_logit_scale != 1.0f) {
        float ls = c.f_logit_scale;
        int n = V;
        void * args[] = { (void *)&b.logits, (void *)&b.logits, (void *)&ls, (void *)&n };
        hipModuleLaunchKernel(k.eval_scale_scalar, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    }

    // Final logit softcap (Gemma2)
    if (c.has_final_logit_softcap && c.final_logit_softcap_val > 0.0f) {
        float cap = c.final_logit_softcap_val;
        float inv_cap = 1.0f / cap;
        int n = V;
        {
            void * args[] = { (void *)&b.logits, (void *)&b.logits, (void *)&inv_cap, (void *)&n };
            hipModuleLaunchKernel(k.eval_scale_scalar, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
        {
            void * args[] = { (void *)&b.logits, (void *)&b.logits, (void *)&n };
            hipModuleLaunchKernel(k.eval_tanh, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
        {
            void * args[] = { (void *)&b.logits, (void *)&b.logits, (void *)&cap, (void *)&n };
            hipModuleLaunchKernel(k.eval_scale_scalar, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
    }

    // Chameleon image token suppression
    if (c.chameleon_img_token_count > 0) {
        int start = c.chameleon_img_token_start;
        int count = c.chameleon_img_token_count;
        if (start + count <= V) {
            void * args[] = { (void *)&b.logits, (void *)&start, (void *)&count };
            hipModuleLaunchKernel(k.eval_chameleon_suppress, (count+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
    }

    // Copy logits to host
    if (logits_out) {
        hipMemcpyAsync(logits_out, b.logits, V * sizeof(float),
                       hipMemcpyDeviceToHost, s);
    }
    hipStreamSynchronize(s);

    return 0;
}
