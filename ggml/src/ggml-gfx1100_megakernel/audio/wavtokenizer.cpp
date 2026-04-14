// audio/wavtokenizer.cpp — WavTokenizer audio decoder forward
// Ported from baseline src/models/wavtokenizer-dec.cpp
//
// Convolutional audio decoder (NOT autoregressive):
//   1. Token embedding → Conv1d (initial projection)
//   2. PosNet: 6 heterogeneous layers (ResNet + attention + GroupNorm)
//   3. Token LayerNorm
//   4. ConvNext blocks: DepthwiseConv1d → LayerNorm → FFN(GELU) → gamma → residual
//   5. Output: LayerNorm → linear → waveform embeddings
//
// Conv1d = im2col + GEMM (baseline decomposition). GroupNorm ported from norm.cu.
#include "../gfx1100-internal.h"
#include "../shared/batch-ops.h"

// Helper: launch GroupNorm on a 1D signal [channels, time]
// Baseline: group_norm_f32 with group_size = channels/n_groups * time
static void launch_group_norm(const float * input, float * output,
                               int channels, int time_len, int n_groups, float eps,
                               hipStream_t stream) {
    auto & k = g_compiled;
    if (!k.eval_group_norm) return;
    int group_size = (channels / n_groups) * time_len;
    int ne_elements = channels * time_len;
    void * args[] = { (void *)&input, (void *)&output,
                      (void *)&group_size, (void *)&ne_elements, (void *)&eps };
    int threads = (group_size < 1024) ? 32 : 1024;
    hipModuleLaunchKernel(k.eval_group_norm, n_groups, 1, 1, threads, 1, 1,
                          32 * sizeof(float), stream, args, nullptr);
}

// Helper: launch Swish activation in-place: x = x * sigmoid(x)
static void launch_swish(float * data, int N, hipStream_t stream) {
    auto & k = g_compiled;
    if (!k.eval_swish) return;
    void * args[] = { (void *)&data, (void *)&data, (void *)&N };
    hipModuleLaunchKernel(k.eval_swish, (N + 255) / 256, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}

// Helper: GroupNorm + weight + bias (fused: normalize then apply affine)
static void launch_group_norm_affine(const float * input, float * output,
                                      const void * weight, const void * bias,
                                      int channels, int time_len, int n_groups, float eps,
                                      hipStream_t stream) {
    auto & k = g_compiled;
    // Step 1: bare GroupNorm
    launch_group_norm(input, output, channels, time_len, n_groups, eps, stream);
    // Step 2: multiply by weight (per-channel broadcast), add bias (per-channel broadcast)
    // Weight is [channels], data is [channels * time] — need broadcast: data[i] *= weight[i % channels]
    // prompt_elementwise_mul does exactly this: data[i] *= scale[i % dim]
    if (weight) {
        int total = channels * time_len;
        int dim = channels;
        void * args[] = { (void *)&output, (void *)&weight, (void *)&dim, (void *)&total };
        hipModuleLaunchKernel(k.prompt_elementwise_mul, (total + 255) / 256, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
    }
    // prompt_add_bias does: data[s*dim + i] += bias[i] — per-channel broadcast add
    if (bias) {
        int N = time_len;  // number of "rows" (time steps)
        int dim = channels;
        void * args[] = { (void *)&output, (void *)&bias, (void *)&dim, (void *)&N };
        hipModuleLaunchKernel(k.prompt_add_bias, (dim + 255) / 256, N, 1, 256, 1, 1, 0, stream, args, nullptr);
    }
}

// Helper: launch Conv1d with half-padding via im2col + GEMM
// input [S, IC], weight [OC, IC, KW], bias [OC] or NULL
// output [S, OC] (same time dimension due to half-padding)
static int launch_conv1d_ph(
        const float * input, int IC, int IW,
        const void * weight, long long weight_stride, int weight_type,
        const void * bias, int OC, int KW,
        float * output, float * im2col_buf,
        hipStream_t stream) {
    auto & k = g_compiled;
    auto & b = g_bufs;
    int p0 = KW / 2, s0 = 1, d0 = 1;
    int OW = IW;
    if (!k.eval_im2col_1d) return -1;

    // Im2col
    int IC_KW = IC * KW;
    {
        void * args[] = { (void *)&input, (void *)&im2col_buf,
                          (void *)&IC, (void *)&IW, (void *)&OW, (void *)&KW,
                          (void *)&s0, (void *)&p0, (void *)&d0 };
        hipModuleLaunchKernel(k.eval_im2col_1d, (IC_KW + 255) / 256, OW, 1,
                              256, 1, 1, 0, stream, args, nullptr);
    }

    // GEMM: weight [OC, IC*KW] × im2col [IC*KW, OW] → output [OC, OW]
    launch_mmq_quantize(k.eval_quantize_mmq_q8_1_d4, im2col_buf,
                        b.batch_q8_mmq, IC_KW, OW, stream);
    batch_projection(weight_type, weight, weight_stride,
                     im2col_buf, b.batch_q8_mmq, output, IC_KW, OC, OW, stream);

    if (bias) {
        int N = OW;
        void * args[] = { (void *)&output, (void *)&bias, (void *)&OC, (void *)&N };
        hipModuleLaunchKernel(k.prompt_add_bias, (OC + 255) / 256, OW, 1, 256, 1, 1,
                              0, stream, args, nullptr);
    }
    return 0;
}

int forward_decode_wavtokenizer(const int * tokens, int n_tokens, float * embd_out) {
    auto & c = g_config;
    auto & b = g_bufs;
    auto & k = g_compiled;
    hipStream_t stream = b.stream;

    int H = c.hidden_size;
    int S = n_tokens;
    int n_groups = c.wav_posnet_n_groups > 0 ? c.wav_posnet_n_groups : 32;
    float eps = c.norm_eps;

    if (S > b.max_batch) {
        fprintf(stderr, "gfx1100: WavTokenizer — n_tokens=%d exceeds max_batch=%d\n", S, b.max_batch);
        return -1;
    }

    // =====================================================================
    // Phase 1: Token embedding → batch_hidden [S, H]
    // =====================================================================
    hipMemcpyAsync(b.batch_token_ids, tokens, S * sizeof(int), hipMemcpyHostToDevice, stream);
    {
        hipFunction_t embed_fn = nullptr;
        int embed_threads = 256;
        switch (c.embed_type) {
            case 0:  embed_fn = k.prompt_embed_f32;  break;
            case 1:  embed_fn = k.prompt_embed_f16;  break;
            case 30: embed_fn = k.prompt_embed_bf16; break;
            case 12: embed_fn = k.prompt_embed_q4k;  embed_threads = 32; break;
            case 14: embed_fn = k.prompt_embed_q6k;  embed_threads = 64; break;
            default:
                fprintf(stderr, "gfx1100: WavTokenizer — unsupported embed type %d\n", c.embed_type);
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
    // Phase 2: Initial Conv1d — baseline line 11
    // cur = transpose → conv_1d_ph(conv1d, cur) → transpose
    // Baseline: ggml_conv_1d_ph(model.conv1d, cur) + ggml_add(bias)
    // This transforms [n_embd, S] to [out_channels, S]
    if (c.wav_conv1d && k.eval_im2col_1d) {
        int KW = c.wav_conv1d_kernel_size > 0 ? c.wav_conv1d_kernel_size : 7;
        int rc = launch_conv1d_ph(b.batch_hidden, H, S,
                                  c.wav_conv1d, c.wav_conv1d_stride, c.wav_conv1d_type,
                                  c.wav_conv1d_b, H, KW,  // OC = H (same channels)
                                  b.batch_norm, b.batch_proj, stream);
        if (rc == 0) {
            hipMemcpyAsync(b.batch_hidden, b.batch_norm, (size_t)S * H * sizeof(float),
                           hipMemcpyDeviceToDevice, stream);
        }
    }

    // =====================================================================
    // Phase 3: PosNet — 6 hardcoded layers
    // Baseline wavtokenizer-dec.cpp lines 16-73
    // =====================================================================
    for (int il = 0; il < 6; il++) {
        const gfx1100_posnet_layer & pl = c.posnet_layers[il];

        if (il == 0 || il == 1 || il == 3 || il == 4) {
            // ResNet block: norm1 → swish → conv1 → norm2 → swish → conv2 → residual
            if (!pl.norm1) continue; // skip if weights not loaded

            // Save residual
            hipMemcpyAsync(b.batch_residual, b.batch_hidden, (size_t)S * H * sizeof(float),
                           hipMemcpyDeviceToDevice, stream);

            // GroupNorm1 + weight + bias
            launch_group_norm_affine(b.batch_hidden, b.batch_norm, pl.norm1, pl.norm1_b,
                                     H, S, n_groups, eps, stream);

            // Swish
            launch_swish(b.batch_norm, S * H, stream);

            // Conv1d #1 — via im2col + GEMM
            // Baseline: ggml_conv_1d_ph(conv1, cur) with kernel_size from weight ne[0]
            // Our layout: [S, H] row-major. Conv1d operates per-channel along time.
            // For same-padding conv1d: OW = IW = S
            if (k.eval_im2col_1d && pl.conv1) {
                // Determine kernel size from weight stride — typical PosNet uses KW=3
                // For now assume KW=3 (baseline wavtokenizer PosNet conv)
                int KW = pl.conv1_kernel_size > 0 ? pl.conv1_kernel_size : 3;
                int rc = launch_conv1d_ph(b.batch_norm, H, S,
                                          pl.conv1, pl.conv1_stride, pl.conv1_type,
                                          pl.conv1_b, H, KW,
                                          b.batch_mlp,      // output
                                          b.batch_proj,      // im2col scratch
                                          stream);
                if (rc == 0) {
                    // conv output in batch_mlp, copy back to batch_norm
                    hipMemcpyAsync(b.batch_norm, b.batch_mlp, (size_t)S * H * sizeof(float),
                                   hipMemcpyDeviceToDevice, stream);
                }
            }

            // GroupNorm2 + weight + bias
            if (pl.norm2) {
                launch_group_norm_affine(b.batch_norm, b.batch_norm, pl.norm2, pl.norm2_b,
                                         H, S, n_groups, eps, stream);
            }

            // Swish
            launch_swish(b.batch_norm, S * H, stream);

            // Conv1d #2 — same as #1
            if (k.eval_im2col_1d && pl.conv2) {
                int KW = pl.conv2_kernel_size > 0 ? pl.conv2_kernel_size : 3;
                int rc = launch_conv1d_ph(b.batch_norm, H, S,
                                          pl.conv2, pl.conv2_stride, pl.conv2_type,
                                          pl.conv2_b, H, KW,
                                          b.batch_mlp, b.batch_proj, stream);
                if (rc == 0) {
                    hipMemcpyAsync(b.batch_norm, b.batch_mlp, (size_t)S * H * sizeof(float),
                                   hipMemcpyDeviceToDevice, stream);
                }
            }

            // Residual add
            {
                int N = S * H;
                void * args[] = { (void *)&b.batch_norm, (void *)&b.batch_residual,
                                  (void *)&b.batch_hidden, (void *)&N };
                hipModuleLaunchKernel(k.prompt_add_residual, (N + 255) / 256, 1, 1, 256, 1, 1,
                                     0, stream, args, nullptr);
            }
        } else if (il == 2) {
            // Self-attention block: GroupNorm → Q/K/V conv1d → softmax → O conv → residual
            if (!pl.attn_norm) continue;

            hipMemcpyAsync(b.batch_residual, b.batch_hidden, (size_t)S * H * sizeof(float),
                           hipMemcpyDeviceToDevice, stream);

            // GroupNorm
            launch_group_norm_affine(b.batch_hidden, b.batch_norm, pl.attn_norm, pl.attn_norm_b,
                                     H, S, n_groups, eps, stream);

            // Q, K, V projections via 1x1 Conv1d (= pointwise = simple matvec per time step)
            // For 1x1 conv: no im2col needed, just batch matmul
            launch_mmq_quantize(k.eval_quantize_mmq_q8_1_d4, b.batch_norm,
                                b.batch_q8_mmq, H, S, stream);

            // Q → batch_proj [S, H]
            if (pl.attn_q)
                batch_projection(pl.attn_q_type, pl.attn_q, pl.attn_q_stride,
                                 b.batch_norm, b.batch_q8_mmq, b.batch_proj, H, H, S, stream);
            // K → batch_kv [S, H]
            if (pl.attn_k)
                batch_projection(pl.attn_k_type, pl.attn_k, pl.attn_k_stride,
                                 b.batch_norm, b.batch_q8_mmq, b.batch_kv, H, H, S, stream);
            // V → batch_attn_out [S, H]
            if (pl.attn_v)
                batch_projection(pl.attn_v_type, pl.attn_v, pl.attn_v_stride,
                                 b.batch_norm, b.batch_q8_mmq, b.batch_attn_out, H, H, S, stream);

            // Add biases
            if (pl.attn_q_b) {
                int N = S;
                void * args[] = { (void *)&b.batch_proj, (void *)&pl.attn_q_b, (void *)&H, (void *)&N };
                hipModuleLaunchKernel(k.prompt_add_bias, (H+255)/256, S, 1, 256, 1, 1, 0, stream, args, nullptr);
            }
            if (pl.attn_k_b) {
                int N = S;
                void * args[] = { (void *)&b.batch_kv, (void *)&pl.attn_k_b, (void *)&H, (void *)&N };
                hipModuleLaunchKernel(k.prompt_add_bias, (H+255)/256, S, 1, 256, 1, 1, 0, stream, args, nullptr);
            }
            if (pl.attn_v_b) {
                int N = S;
                void * args[] = { (void *)&b.batch_attn_out, (void *)&pl.attn_v_b, (void *)&H, (void *)&N };
                hipModuleLaunchKernel(k.prompt_add_bias, (H+255)/256, S, 1, 256, 1, 1, 0, stream, args, nullptr);
            }

            // Bidirectional attention (all tokens see all tokens)
            // Use prompt_bidirectional_attn kernel
            if (k.prompt_bidirectional_attn && pl.attn_q) {
                float scale = 1.0f / sqrtf((float)H);  // treating all channels as one head
                int nh = 1, nkv = 1, hd = H, seq = S;
                const float * kq_b = nullptr;
                void * args[] = { (void *)&b.batch_proj, (void *)&b.batch_kv,
                                  (void *)&b.batch_attn_out, (void *)&b.batch_mlp,
                                  (void *)&kq_b, (void *)&scale,
                                  (void *)&nh, (void *)&nkv, (void *)&hd, (void *)&seq };
                hipModuleLaunchKernel(k.prompt_bidirectional_attn,
                                     1, S, 1, 256, 1, 1, 0, stream, args, nullptr);
            }

            // O projection
            if (pl.attn_o) {
                launch_mmq_quantize(k.eval_quantize_mmq_q8_1_d4, b.batch_mlp,
                                    b.batch_q8_mmq, H, S, stream);
                batch_projection(pl.attn_o_type, pl.attn_o, pl.attn_o_stride,
                                 b.batch_mlp, b.batch_q8_mmq, b.batch_norm, H, H, S, stream);
                if (pl.attn_o_b) {
                    int N = S;
                    void * args[] = { (void *)&b.batch_norm, (void *)&pl.attn_o_b, (void *)&H, (void *)&N };
                    hipModuleLaunchKernel(k.prompt_add_bias, (H+255)/256, S, 1, 256, 1, 1, 0, stream, args, nullptr);
                }
            }

            // Residual
            {
                int N = S * H;
                void * args[] = { (void *)&b.batch_norm, (void *)&b.batch_residual,
                                  (void *)&b.batch_hidden, (void *)&N };
                hipModuleLaunchKernel(k.prompt_add_residual, (N + 255) / 256, 1, 1, 256, 1, 1,
                                     0, stream, args, nullptr);
            }
        } else if (il == 5) {
            // Final GroupNorm only
            if (pl.norm1) {
                launch_group_norm_affine(b.batch_hidden, b.batch_hidden, pl.norm1, pl.norm1_b,
                                         H, S, n_groups, eps, stream);
            }
        }
    }

    // =====================================================================
    // Phase 4: Token LayerNorm — baseline line 96
    // =====================================================================
    if (c.tok_norm_weight && k.prompt_layernorm) {
        const void * w = c.tok_norm_weight;
        const void * bias = c.tok_norm_bias;
        int D = H;
        void * args[] = { (void *)&b.batch_hidden, (void *)&w, (void *)&bias,
                          (void *)&b.batch_hidden, (void *)&D, (void *)&eps };
        int threads = (H < 1024) ? 256 : 1024;
        hipModuleLaunchKernel(k.prompt_layernorm, S, 1, 1, threads, 1, 1, 0, stream, args, nullptr);
    }

    // =====================================================================
    // Phase 5: ConvNext blocks — baseline lines 99-131
    // DepthwiseConv1d → transpose → LayerNorm → FFN(GELU, seq) → gamma → residual
    // =====================================================================
    for (int il = 0; il < c.n_convnext_layers; il++) {
        const gfx1100_convnext_layer & cl = c.convnext_layers[il];
        if (!cl.norm) continue;

        hipMemcpyAsync(b.batch_residual, b.batch_hidden, (size_t)S * H * sizeof(float),
                       hipMemcpyDeviceToDevice, stream);

        // Depthwise Conv1d — baseline: ggml_conv_1d_dw_ph → im2col + mul_mat
        // Depthwise: each channel independently convolved with its own [kernel_size] filter
        // Weight: [channels, 1, kernel_size]. For depthwise, we process each channel
        // independently. With our im2col kernel (IC=1 per channel), this becomes
        // S independent dot products of length KW per channel.
        //
        // Simplified approach: for small kernels (7-15), use the existing eval_ssm_conv_step
        // pattern (shift register + convolve) per channel. But that's for single-step.
        // For batch: im2col with IC=1 per channel, then element-wise multiply-accumulate.
        //
        // For now: use im2col with IC=1 for each channel independently.
        // This is correct but slow (H separate im2col + dot product launches).
        // A dedicated depthwise conv1d kernel would be much faster.
        if (k.eval_conv1d_dw_ph && cl.dw) {
            // Depthwise Conv1d — ported from baseline conv2d_dw_kernel adapted for 1D
            // Weight: [channels, 1, KW]. Each channel independently convolved.
            int KW = cl.dw_kernel_size > 0 ? cl.dw_kernel_size : 7;
            int total = H * S;
            const float * dw_w = (const float *)cl.dw;
            const float * dw_b = (const float *)cl.dw_b;
            void * args[] = { (void *)&b.batch_hidden, (void *)&dw_w, (void *)&dw_b,
                              (void *)&b.batch_norm,
                              (void *)&H, (void *)&S, (void *)&KW, (void *)&total };
            hipModuleLaunchKernel(k.eval_conv1d_dw_ph, (total + 255) / 256, 1, 1,
                                  256, 1, 1, 0, stream, args, nullptr);
        } else {
            hipMemcpyAsync(b.batch_norm, b.batch_hidden, (size_t)S * H * sizeof(float),
                           hipMemcpyDeviceToDevice, stream);
        }

        // Add dw bias only if the kernel didn't already do it
        // (eval_conv1d_dw_ph adds bias internally; fallback memcpy path does not)
        if (cl.dw_b && !(k.eval_conv1d_dw_ph && cl.dw)) {
            int N = S;
            void * args[] = { (void *)&b.batch_norm, (void *)&cl.dw_b, (void *)&H, (void *)&N };
            hipModuleLaunchKernel(k.prompt_add_bias, (H+255)/256, S, 1, 256, 1, 1, 0, stream, args, nullptr);
        }

        // LayerNorm (per time step)
        {
            const void * w = cl.norm;
            const void * bias = cl.norm_b;
            int D = H;
            void * args[] = { (void *)&b.batch_norm, (void *)&w, (void *)&bias,
                              (void *)&b.batch_norm, (void *)&D, (void *)&eps };
            int threads = (H < 1024) ? 256 : 1024;
            hipModuleLaunchKernel(k.prompt_layernorm, S, 1, 1, threads, 1, 1, 0, stream, args, nullptr);
        }

        // FFN: pw1 (up) → GELU → pw2 (down)
        int FF = c.intermediate_size > 0 ? c.intermediate_size : H * 4;
        launch_mmq_quantize(k.eval_quantize_mmq_q8_1_d4, b.batch_norm,
                            b.batch_q8_mmq, H, S, stream);
        // pw1 (up): [S, H] → [S, FF]
        if (cl.pw1) {
            batch_projection(cl.pw1_type, cl.pw1, cl.pw1_stride,
                             b.batch_norm, b.batch_q8_mmq, b.batch_proj, H, FF, S, stream);
            if (cl.pw1_b) {
                int N = S;
                void * args[] = { (void *)&b.batch_proj, (void *)&cl.pw1_b, (void *)&FF, (void *)&N };
                hipModuleLaunchKernel(k.prompt_add_bias, (FF+255)/256, S, 1, 256, 1, 1, 0, stream, args, nullptr);
            }
        }

        // GELU activation (standalone, sequential FFN)
        {
            int N = S * FF;
            void * args[] = { (void *)&b.batch_proj, (void *)&b.batch_proj, (void *)&N };
            hipModuleLaunchKernel(k.eval_gelu, (N + 255) / 256, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
        }

        // pw2 (down): [S, FF] → [S, H]
        if (cl.pw2) {
            launch_mmq_quantize(k.eval_quantize_mmq_q8_1_d4, b.batch_proj,
                                b.batch_q8_mmq, FF, S, stream);
            batch_projection(cl.pw2_type, cl.pw2, cl.pw2_stride,
                             b.batch_proj, b.batch_q8_mmq, b.batch_norm, FF, H, S, stream);
            if (cl.pw2_b) {
                int N = S;
                void * args[] = { (void *)&b.batch_norm, (void *)&cl.pw2_b, (void *)&H, (void *)&N };
                hipModuleLaunchKernel(k.prompt_add_bias, (H+255)/256, S, 1, 256, 1, 1, 0, stream, args, nullptr);
            }
        }

        // Gamma scaling: element-wise multiply by per-channel gamma
        if (cl.gamma) {
            // gamma is [H], broadcast across time
            // For [S, H] layout: multiply each row element by gamma[col]
            // Use prompt_elementwise_mul or scale pattern
            // Since gamma is per-channel, we need broadcast multiply
            // For now: element-wise mul treats gamma as [H] repeated S times
            // This is incorrect for non-[S,H] layout but works for our row-major data
            for (int t = 0; t < S; t++) {
                float * row = b.batch_norm + t * H;
                int N = H;
                void * args[] = { (void *)&row, (void *)&cl.gamma, (void *)&row, (void *)&N };
                hipModuleLaunchKernel(k.eval_elementwise_mul, (N+255)/256, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
            }
        }

        // Residual add
        {
            int N = S * H;
            void * args[] = { (void *)&b.batch_norm, (void *)&b.batch_residual,
                              (void *)&b.batch_hidden, (void *)&N };
            hipModuleLaunchKernel(k.prompt_add_residual, (N + 255) / 256, 1, 1, 256, 1, 1,
                                 0, stream, args, nullptr);
        }
    }

    // =====================================================================
    // Phase 6: Final LayerNorm + output projection
    // Baseline lines 133-148
    // =====================================================================
    // Final LayerNorm
    if (c.final_norm_weight && k.prompt_layernorm) {
        const void * w = c.final_norm_weight;
        const void * bias = c.final_norm_bias;
        int D = H;
        void * args[] = { (void *)&b.batch_hidden, (void *)&w, (void *)&bias,
                          (void *)&b.batch_hidden, (void *)&D, (void *)&eps };
        int threads = (H < 1024) ? 256 : 1024;
        hipModuleLaunchKernel(k.prompt_layernorm, S, 1, 1, threads, 1, 1, 0, stream, args, nullptr);
    }

    // Output projection: model.output * cur + output_b → waveform embeddings
    // WavTokenizer produces t_embd (embeddings), NOT t_logits
    if (c.lm_head_weight) {
        int V = c.vocab_size;  // output dim (waveform embedding dim)
        launch_mmq_quantize(k.eval_quantize_mmq_q8_1_d4, b.batch_hidden,
                            b.batch_q8_mmq, H, S, stream);
        batch_projection(c.lm_head_type, c.lm_head_weight, c.lm_head_stride,
                         b.batch_hidden, b.batch_q8_mmq, b.batch_proj, H, V, S, stream);
        if (c.wav_output_b) {
            int N = S;
            void * args[] = { (void *)&b.batch_proj, (void *)&c.wav_output_b, (void *)&V, (void *)&N };
            hipModuleLaunchKernel(k.prompt_add_bias, (V+255)/256, S, 1, 256, 1, 1, 0, stream, args, nullptr);
        }
        // Copy output embeddings
        if (embd_out) {
            hipMemcpyAsync(embd_out, b.batch_proj, (size_t)S * V * sizeof(float),
                           hipMemcpyDeviceToHost, stream);
        }
    } else {
        // No output projection — return hidden states directly
        if (embd_out) {
            hipMemcpyAsync(embd_out, b.batch_hidden, (size_t)S * H * sizeof(float),
                           hipMemcpyDeviceToHost, stream);
        }
    }

    hipStreamSynchronize(stream);
    return 0;
}
