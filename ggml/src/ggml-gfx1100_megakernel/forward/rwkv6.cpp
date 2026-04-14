// forward/rwkv6.cpp — RWKV6 recurrence forward
#include "../gfx1100-internal.h"

int forward_decode_rwkv6(int token_id, int position, float * logits_out) {
    auto & c = g_config;
    auto & b = g_bufs;
    auto & k = g_compiled;
    auto   s = b.stream;
    int    H = c.hidden_size;
    int    V = c.vocab_size;
    const int norm_threads = (H < 1024) ? 256 : 1024;
    // RWKV geometry
    int head_size = c.wkv_head_size > 0 ? c.wkv_head_size : 64;
    int n_head = H / head_size;

    // Helper lambdas for element-wise ops on GPU buffers
    auto ew_launch = [&](hipFunction_t fn, int n) {
        hipModuleLaunchKernel(fn, (n+255)/256, 1, 1, 256, 1, 1, 0, s, nullptr, nullptr);
    };
    // sub: dst = a - b
    auto launch_sub = [&](const float * a, const float * b_ptr, float * dst, int n) {
        void * args[] = { (void *)&a, (void *)&b_ptr, (void *)&dst, (void *)&n };
        hipModuleLaunchKernel(k.eval_sub, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    };
    // muladd: dst = a * b + c
    auto launch_muladd = [&](const float * a, const float * b_ptr, const float * c_ptr, float * dst, int n) {
        void * args[] = { (void *)&a, (void *)&b_ptr, (void *)&c_ptr, (void *)&dst, (void *)&n };
        hipModuleLaunchKernel(k.eval_muladd, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    };
    // unary in-place: dst = fn(dst)
    auto launch_unary = [&](hipFunction_t fn, float * buf, int n) {
        void * args[] = { (void *)&buf, (void *)&buf, (void *)&n };
        hipModuleLaunchKernel(fn, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    };
    auto launch_mul = [&](float * dst, const void * scale, int n) {
        void * ma[] = { (void *)&dst, (void *)&scale, (void *)&dst, (void *)&n };
        hipModuleLaunchKernel(k.eval_elementwise_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, ma, nullptr);
    };
    auto launch_add = [&](float * dst, const void * bias, int n) {
        void * ba[] = { (void *)&dst, (void *)&bias, (void *)&dst, (void *)&n };
        hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, s, ba, nullptr);
    };

    // Embedding — full type dispatch via shared launch_embed
    if (launch_embed(token_id, b.hidden, s) != 0) return -1;

    // Initial LayerNorm (tok_norm) — baseline rwkv6.cpp line 11
    // build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, 0)
    if (c.tok_norm_weight) {
        const void * nw = c.tok_norm_weight;
        const void * nb = c.tok_norm_bias;
        float eps = c.norm_eps;
        int n = H;
        void * args[] = { (void *)&b.hidden, (void *)&nw, (void *)&nb,
                          (void *)&b.hidden, (void *)&b.residual, (void *)&n, (void *)&eps };
        hipModuleLaunchKernel(k.eval_layernorm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
    }

    for (int il = 0; il < c.n_layers; il++) {
        const gfx1100_layer_weights & lw = c.layers[il];

        // Phase 1: LayerNorm → att_norm — baseline rwkv6.cpp line 32
        {
            const void * nw = lw.ptrs[0]; // attn_norm weight
            const void * nb = lw.attn_norm_bias;
            float eps = c.norm_eps;
            int n = H;
            void * args[] = { (void *)&b.hidden, (void *)&nw, (void *)&nb,
                              (void *)&b.norm_out, (void *)&b.residual, (void *)&n, (void *)&eps };
            hipModuleLaunchKernel(k.eval_layernorm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        }
        // b.norm_out = att_norm, b.residual = inpL (for later residual add)

        // Phase 2: Time-mix (RWKV recurrence) — baseline rwkv6-base.cpp build_rwkv6_time_mix
        // Complete implementation for decode (n_tokens=1)
        {
            float * att_shift = b.rwkv_att_shift + (long long)il * H;
            float * cur = b.norm_out; // att_norm output [H]

            // sx = x_prev - cur — baseline line 52
            launch_sub(att_shift, cur, b.rwkv_sx, H);

            // Update att_shift state for next token
            hipMemcpyAsync(att_shift, cur, H * sizeof(float), hipMemcpyDeviceToDevice, s);

            // --- LoRA mixing bottleneck: baseline lines 57-91 ---
            // Step 1: xxx_base = lerp_x * sx + cur  [H]
            launch_muladd((const float *)lw.time_mix_lerp_x, b.rwkv_sx, cur, b.proj_scratch, H);

            // Step 2: xxx_lora = w1 @ xxx_base  [lora_size*5]
            // w1 is [lora_size*5, H], typically F32 (small matrix)
            // Use F32 matvec for the LoRA bottleneck
            {
                int lora5 = lw.time_mix_w1_stride > 0 ?
                    (int)(lw.time_mix_w1_stride / sizeof(float)) : 0; // approximate out_dim
                // For F32 weights: use eval_matvec_f32 directly
                const void * w = lw.time_mix_w1;
                long long st = lw.time_mix_w1_stride;
                void * inp = (void *)b.proj_scratch;
                void * out = (void *)b.rwkv_xxx;
                int in_dim = H;
                // out_dim = number of rows in w1 = lora_size * 5
                // We can infer from stride: each row is H floats, stride = H * sizeof(float)
                // The tensor shape is [H, lora_size*5], so out_dim rows
                // lora_size from config (rwkv_lora_size = hparams.time_mix_extra_dim)
                int lora_s = c.rwkv_lora_size > 0 ? c.rwkv_lora_size : 32; int out_dim = lora_s * 5;
                void * args[] = { &w, &st, &inp, &out, &in_dim, &out_dim };
                hipModuleLaunchKernel(k.eval_matvec_f32, out_dim, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }

            // Step 3: tanh(xxx_lora) — baseline line 59
            {
                int n = (c.rwkv_lora_size > 0 ? c.rwkv_lora_size : 32) * 5;
                launch_unary(k.eval_tanh, b.rwkv_xxx, n);
            }

            // Step 4: w2 up-projection — 5 separate matvecs from 5 slices of rwkv_xxx
            // w2 shape is [lora_size, H, 5] → 5 matrices of [H, lora_size]
            // For decode: each produces [H] output from [lora_size] input
            // Output goes to rwkv_xw, rwkv_xk, rwkv_xv, rwkv_xr, rwkv_xg
            {
                int lora_size = c.rwkv_lora_size > 0 ? c.rwkv_lora_size : 32;
                float * xxx_slices[5] = { b.rwkv_xxx, b.rwkv_xxx + lora_size,
                    b.rwkv_xxx + 2*lora_size, b.rwkv_xxx + 3*lora_size, b.rwkv_xxx + 4*lora_size };
                float * outputs[5] = { b.rwkv_xw, b.rwkv_xk, b.rwkv_xv, b.rwkv_xr, b.rwkv_xg };

                for (int vi = 0; vi < 5; vi++) {
                    // w2 slice i: base + i * lora_size * H * sizeof(float)
                    const char * w2_slice = (const char *)lw.time_mix_w2 +
                        (long long)vi * lora_size * lw.time_mix_w2_stride;
                    void * w = (void *)w2_slice;
                    long long st = lw.time_mix_w2_stride;
                    void * inp = (void *)xxx_slices[vi];
                    void * out = (void *)outputs[vi];
                    int in_dim = lora_size, out_dim = H;
                    void * args[] = { &w, &st, &inp, &out, &in_dim, &out_dim };
                    hipModuleLaunchKernel(k.eval_matvec_f32, out_dim, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }
            }

            // Step 5: Final lerp per component — baseline lines 86-90
            // x_i = (xxx_i + lerp_i) * sx + cur  for i in {w,k,v,r,g}
            if (lw.time_mix_lerp_fused) {
                // Fused path (RWKV6Qwen2): single [n_embd, 5] tensor
                // lerp_fused[:, i] is at offset i * H floats
                float * outputs[5] = { b.rwkv_xw, b.rwkv_xk, b.rwkv_xv, b.rwkv_xr, b.rwkv_xg };
                for (int vi = 0; vi < 5; vi++) {
                    const void * lerp_slice = (const char *)lw.time_mix_lerp_fused + (long long)vi * H * sizeof(float);
                    // xxx_i += lerp_fused[:, vi]
                    launch_add(outputs[vi], lerp_slice, H);
                    // xxx_i *= sx
                    launch_mul(outputs[vi], (const void *)b.rwkv_sx, H);
                    // xxx_i += cur
                    launch_add(outputs[vi], (const void *)cur, H);
                }
            } else {
                // Non-fused path (standard RWKV6): 5 individual lerp tensors
                struct { float * xxx; const void * lerp; float * out; } lerp_ops[5] = {
                    { b.rwkv_xw, lw.time_mix_lerp_w, b.rwkv_xw },
                    { b.rwkv_xk, lw.time_mix_lerp_k, b.rwkv_xk },
                    { b.rwkv_xv, lw.time_mix_lerp_v, b.rwkv_xv },
                    { b.rwkv_xr, lw.time_mix_lerp_r, b.rwkv_xr },
                    { b.rwkv_xg, lw.time_mix_lerp_g, b.rwkv_xg },
                };
                for (int vi = 0; vi < 5; vi++) {
                    if (!lerp_ops[vi].lerp) continue;
                    // xxx_i += lerp_i
                    launch_add(lerp_ops[vi].out, lerp_ops[vi].lerp, H);
                    // xxx_i *= sx
                    launch_mul(lerp_ops[vi].out, (const void *)b.rwkv_sx, H);
                    // xxx_i += cur
                    launch_add(lerp_ops[vi].out, (const void *)cur, H);
                }
            }
            // Now rwkv_xw..rwkv_xg have the 5 mixed inputs

            // --- R, K, V, G projections — baseline lines 92-109 ---
            // Each is a matvec: [H, H] @ [H] → [H]
            // r = receptance @ xr
            auto quant_and_matvec = [&](const void * w, long long st, int type,
                                         float * input, float * output, int dim) {
                quant_and_launch_matvec(type, w, st, input, output, dim, dim, s);
            };

            quant_and_matvec(lw.time_mix_receptance, lw.time_mix_receptance_stride,
                             lw.time_mix_receptance_type, b.rwkv_xr, b.proj_scratch, H);
            quant_and_matvec(lw.time_mix_key, lw.time_mix_key_stride,
                             lw.time_mix_key_type, b.rwkv_xk, b.kv_scratch, H);
            quant_and_matvec(lw.time_mix_value, lw.time_mix_value_stride,
                             lw.time_mix_value_type, b.rwkv_xv, b.mlp_inter, H);
            quant_and_matvec(lw.time_mix_gate, lw.time_mix_gate_stride,
                             lw.time_mix_gate_type, b.rwkv_xg, b.attn_out, H);

            // g = silu(gate) for standard RWKV6, sigmoid(gate) for QRWKV (RWKV6Qwen2)
            // Baseline line 108: is_qrwkv ? sigmoid(g) : silu(g)
            // Detect QRWKV by time_mix_first == nullptr
            if (lw.time_mix_first) {
                launch_unary(k.eval_silu, b.attn_out, H);
            } else {
                // QRWKV: sigmoid gate
                int n = H;
                void * args[] = { (void *)&b.attn_out, (void *)&b.attn_out, (void *)&n };
                hipModuleLaunchKernel(k.eval_sigmoid, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }

            // Add biases if present — baseline lines 95-103
            if (lw.time_mix_receptance_b) launch_add(b.proj_scratch, lw.time_mix_receptance_b, H);
            if (lw.time_mix_key_b)        launch_add(b.kv_scratch, lw.time_mix_key_b, H);
            if (lw.time_mix_value_b)      launch_add(b.mlp_inter, lw.time_mix_value_b, H);

            // --- Decay computation — baseline lines 122-127 ---
            // w = exp(-exp(decay + decay_w2 @ tanh(decay_w1 @ xw)))
            // Step a: tmp = decay_w1 @ xw → [lora_size]
            // Step b: tmp = tanh(tmp)
            // Step c: tmp = decay_w2 @ tmp → [H]
            // Step d: tmp += decay
            // Step e: tmp = exp(tmp)
            // Step f: tmp = -tmp
            // Step g: w = exp(tmp)
            {
                // decay_w1 @ xw → use rwkv_xxx as scratch (lora_size fits)
                const void * dw1 = lw.time_mix_decay_w1;
                long long dw1_st = lw.time_mix_decay_w1_stride;
                int lora_size = c.rwkv_lora_size > 0 ? c.rwkv_lora_size : 32;
                void * inp = (void *)b.rwkv_xw;
                void * tmp_lora = (void *)b.rwkv_xxx; // reuse
                void * args1[] = { (void *)&dw1, &dw1_st, &inp, &tmp_lora, &H, &lora_size };
                hipModuleLaunchKernel(k.eval_matvec_f32, lora_size, 1, 1, 256, 1, 1, 0, s, args1, nullptr);

                // tanh
                launch_unary(k.eval_tanh, (float *)tmp_lora, lora_size);

                // decay_w2 @ tmp → rwkv_xw (reuse as decay output [H])
                const void * dw2 = lw.time_mix_decay_w2;
                long long dw2_st = lw.time_mix_decay_w2_stride;
                void * out_decay = (void *)b.rwkv_xw;
                void * args2[] = { (void *)&dw2, &dw2_st, &tmp_lora, &out_decay, &lora_size, &H };
                hipModuleLaunchKernel(k.eval_matvec_f32, H, 1, 1, 256, 1, 1, 0, s, args2, nullptr);

                // w += decay (base decay tensor)
                launch_add(b.rwkv_xw, lw.time_mix_decay, H);

                // w = exp(-exp(w)) — baseline line 126
                launch_unary(k.eval_exp, b.rwkv_xw, H);
                launch_unary(k.eval_neg, b.rwkv_xw, H);
                launch_unary(k.eval_exp, b.rwkv_xw, H);
            }
            // b.rwkv_xw now has the decay weights w [H]

            // QRWKV: k = k * (1 - w) — baseline rwkv6-base.cpp line 129
            if (!lw.time_mix_first) {
                int n = H;
                void * args[] = { (void *)&b.kv_scratch, (void *)&b.kv_scratch, (void *)&b.rwkv_xw, (void *)&n };
                hipModuleLaunchKernel(k.eval_mul_one_minus, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }

            // --- WKV6 recurrence — baseline line 139 ---
            // y = rwkv_wkv6(k, v, r, time_first, w, state)
            // k = kv_scratch [H], v = mlp_inter [H], r = proj_scratch [H]
            // w = rwkv_xw [H], time_first = lw.time_mix_first [H]
            // state = rwkv_wkv_state + il * n_head * head_size * head_size
            {
                float * wkv_state = b.rwkv_wkv_state + (long long)il * n_head * head_size * head_size;
                float * k_buf = b.kv_scratch;
                float * v_buf = b.mlp_inter;
                float * r_buf = b.proj_scratch;
                // QRWKV: time_mix_first is NULL — use zero-filled proj_scratch as substitute
                // (WKV6 kernel reads tf but for QRWKV the gated_linear_attn equivalent has tf=0)
                float * tf;
                if (lw.time_mix_first) {
                    tf = (float *)lw.time_mix_first;
                } else {
                    tf = b.proj_scratch; // reuse as temp — fill with zeros
                    hipMemsetAsync(tf, 0, H * sizeof(float), s);
                }
                float * td = b.rwkv_xw; // decay
                float * y_buf = b.hidden; // output (will be overwritten, that's OK since
                                          // we saved residual in b.residual)
                int hs = head_size, nh = n_head;
                size_t smem = 4 * head_size * sizeof(float);
                void * args[] = { &k_buf, &v_buf, &r_buf, &tf, &td, &wkv_state, &y_buf, &hs, &nh };
                hipModuleLaunchKernel(k.eval_rwkv_wkv6_step, n_head, 1, 1, head_size, 1, 1,
                                      smem, s, args, nullptr);
            }
            // b.hidden now has WKV6 output y [H]

            // --- Group norm — baseline lines 149-156 ---
            // Baseline: reshape to [head_dim, n_head], ggml_norm per head, reshape back, mul+add
            // For decode: per-head layernorm on head_size elements
            // Launch one eval_layernorm per head, each normalizing head_size elements
            if (lw.time_mix_ln) {
                float eps = 64e-5f; // baseline uses 64e-5 for RWKV group norm
                int nt = (head_size < 1024) ? 256 : 1024;
                for (int h = 0; h < n_head; h++) {
                    float * h_in  = b.hidden + h * head_size;
                    const float * h_w = (const float *)lw.time_mix_ln + h * head_size;
                    const float * h_b = (const float *)lw.time_mix_ln_b + h * head_size;
                    float * h_out = b.hidden + h * head_size; // in-place
                    float * h_res = b.proj_scratch + h * head_size; // unused residual
                    int n = head_size;
                    void * args[] = { (void *)&h_in, (void *)&h_w, (void *)&h_b,
                                      (void *)&h_out, (void *)&h_res, (void *)&n, (void *)&eps };
                    hipModuleLaunchKernel(k.eval_layernorm, 1, 1, 1, nt, 1, 1, 0, s, args, nullptr);
                }
            }

            // --- Gate multiply — baseline line 160: cur = cur * g ---
            launch_mul(b.hidden, (const void *)b.attn_out, H);

            // --- Output projection — baseline line 161: cur = output @ y ---
            {
                int n = H, blocks = (n+511)/512;
                void * q8args[] = { (void *)&b.hidden, (void *)&b.q8_act, (void *)&n };
                hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
            }
            quant_and_matvec(lw.time_mix_output, lw.time_mix_output_stride,
                             lw.time_mix_output_type, b.hidden, b.hidden, H);
        }

        // Residual: hidden = time_mix_output + residual(inpL)
        launch_add(b.hidden, (const void *)b.residual, H);

        // Phase 3: Channel-mix (FFN replacement) — baseline rwkv6-base.cpp build_rwkv6_channel_mix
        // LayerNorm
        {
            int post_norm_idx = 7; // attn_norm_2
            const void * nw = lw.ptrs[post_norm_idx];
            const void * nb = lw.ffn_norm_bias;
            float eps = c.norm_eps;
            int n = H;
            void * args[] = { (void *)&b.hidden, (void *)&nw, (void *)&nb,
                              (void *)&b.norm_out, (void *)&b.residual, (void *)&n, (void *)&eps };
            hipModuleLaunchKernel(k.eval_layernorm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        }
        // Channel mix: baseline rwkv6-base.cpp::build_rwkv6_channel_mix lines 9-28
        // r = sigmoid(receptance @ xr), k = relu²(key @ xk), cur = r * (value @ k)
        {
            float * ffn_shift = b.rwkv_ffn_shift + (long long)il * H;
            float * cur = b.norm_out;

            // sx = ffn_shift - cur — baseline line 13
            launch_sub(ffn_shift, cur, b.rwkv_sx, H);
            // Update ffn_shift state
            hipMemcpyAsync(ffn_shift, cur, H * sizeof(float), hipMemcpyDeviceToDevice, s);

            // xk = lerp_k * sx + cur — baseline line 17
            launch_muladd((const float *)lw.channel_mix_lerp_k, b.rwkv_sx, cur, b.proj_scratch, H);
            // xr = lerp_r * sx + cur — baseline line 18
            launch_muladd((const float *)lw.channel_mix_lerp_r, b.rwkv_sx, cur, b.kv_scratch, H);

            // r = sigmoid(receptance @ xr) — baseline line 20
            quant_and_launch_matvec(lw.channel_mix_receptance_type,
                                     lw.channel_mix_receptance, lw.channel_mix_receptance_stride,
                                     b.kv_scratch, b.attn_out, H, H, s);
            launch_unary(k.eval_sigmoid, b.attn_out, H);
            // b.attn_out = r [H]

            // k = sqr(relu(key @ xk)) — baseline line 21
            quant_and_launch_matvec(lw.channel_mix_key_type,
                                     lw.channel_mix_key, lw.channel_mix_key_stride,
                                     b.proj_scratch, b.mlp_inter, H, H, s);
            launch_unary(k.eval_relu, b.mlp_inter, H);
            launch_unary(k.eval_sqr, b.mlp_inter, H);
            // b.mlp_inter = k = relu²(key @ xk) [H]

            // cur = r * (value @ k) — baseline line 22
            quant_and_launch_matvec(lw.channel_mix_value_type,
                                     lw.channel_mix_value, lw.channel_mix_value_stride,
                                     b.mlp_inter, b.hidden, H, H, s);
            // hidden = value @ k, now multiply by r
            launch_mul(b.hidden, (const void *)b.attn_out, H);
            // b.hidden = r * (value @ k) = channel_mix output
        }

        // Residual
        launch_add(b.hidden, (const void *)b.residual, H);
    }

    // Final LayerNorm — baseline rwkv6.cpp line 83
    // build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1)
    {
        const void * nw = c.final_norm_weight;
        const void * nb = c.final_norm_bias; // output_norm_b for LayerNorm models
        float eps = c.norm_eps;
        int n = H;
        void * args[] = { (void *)&b.hidden, (void *)&nw, (void *)&nb,
                          (void *)&b.norm_out, (void *)&b.residual, (void *)&n, (void *)&eps };
        hipModuleLaunchKernel(k.eval_layernorm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
    }

    // LM head — same as other archs
    {
        int n = H; int blocks = (n+511)/512;
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

// ----------------------------------------------------------------------------
// forward_decode_mamba
//
// Mamba/Mamba2 SSM decode — ported from baseline src/models/mamba-base.cpp::build_mamba_layer.
// Per-layer: norm → ssm_in proj → conv1d step → silu → ssm_x proj → dt proj →
//            ssm_scan step → D skip connection → silu(z)*y gate → ssm_out proj → residual.
// Stateful: conv states [n_layers, d_inner, d_conv-1] and scan states [n_layers, d_inner, d_state].
// ----------------------------------------------------------------------------
