// forward/rwkv7.cpp — RWKV7 recurrence forward
// Ported from baseline src/models/rwkv7-base.cpp (build_rwkv7_time_mix + build_rwkv7_channel_mix)
//
// Key differences from RWKV6:
//   - Single fused lerp (not w1/tanh/w2 + individual lerps)
//   - Decay: exp(sigmoid(w) * -0.606531) instead of exp(-exp(w))
//   - v_first cross-layer residual (layer 0's v blended into all layers)
//   - L2-normalized key kk for WKV7 a/b parameters
//   - Key modification via a gate: k = k + k_a*(a - 1)
//   - WKV7 recurrence: state = state*w + k*v + dot(a,state)*b (state feedback)
//   - Post-WKV r*k additive output correction
//   - Optional gating: g2 * sigmoid(g1 * xg)
//   - Simpler channel_mix: no receptance gate
#include "../gfx1100-internal.h"

int forward_decode_rwkv7(int token_id, int position, float * logits_out) {
    auto & c = g_config;
    auto & b = g_bufs;
    auto & k = g_compiled;
    auto   s = b.stream;
    int    H = c.hidden_size;
    int    V = c.vocab_size;
    const int norm_threads = (H < 1024) ? 256 : 1024;
    int head_size = c.wkv_head_size > 0 ? c.wkv_head_size : 64;
    int n_head = H / head_size;
    int lora_size = c.rwkv_lora_size;

    // Helper lambdas
    auto launch_add = [&](float * dst, const void * bias, int n) {
        void * ba[] = { (void *)&dst, (void *)&bias, (void *)&dst, (void *)&n };
        hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, s, ba, nullptr);
    };
    auto launch_sub = [&](const float * a, const float * bp, float * dst, int n) {
        void * args[] = { (void *)&a, (void *)&bp, (void *)&dst, (void *)&n };
        hipModuleLaunchKernel(k.eval_sub, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    };
    auto launch_mul = [&](float * dst, const void * w, int n) {
        void * args[] = { (void *)&dst, (void *)&w, (void *)&dst, (void *)&n };
        hipModuleLaunchKernel(k.eval_elementwise_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    };
    auto launch_muladd = [&](const float * a, const float * bp, const float * cp, float * dst, int n) {
        void * args[] = { (void *)&a, (void *)&bp, (void *)&cp, (void *)&dst, (void *)&n };
        hipModuleLaunchKernel(k.eval_muladd, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    };

    // Embedding
    if (launch_embed(token_id, b.hidden, s) != 0) return -1;

    // Token norm (LayerNorm on embedding) — baseline: build_norm(tok_norm)
    if (c.tok_norm_weight) {
        const void * w = c.tok_norm_weight;
        const void * bias = c.tok_norm_bias;
        int n = H;
        float eps = c.norm_eps;
        void * args[] = { (void *)&b.hidden, (void *)&w, (void *)&bias,
                          (void *)&b.hidden, (void *)&b.hidden, (void *)&n, (void *)&eps };
        hipModuleLaunchKernel(k.eval_layernorm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
    }

    // Reset v_first flag for this generation step
    b.rwkv7_v_first_set = false;

    for (int il = 0; il < c.n_layers; il++) {
        const gfx1100_layer_weights & lw = c.layers[il];

        // ===== Attention norm =====
        // RWKV7: LayerNorm (LLM_NORM). ARWKV7: RMSNorm (LLM_NORM_RMS) with optional bias.
        // Ported from baseline rwkv7.cpp line 24 and arwkv7.cpp line 27.
        if (c.norm_type == 2) {
            // LayerNorm (standard RWKV7)
            const void * nw = lw.ptrs[0];
            const void * nb = lw.attn_norm_bias;
            int n = H;
            float eps = c.norm_eps;
            void * args[] = { (void *)&b.hidden, (void *)&nw, (void *)&nb,
                              (void *)&b.norm_out, (void *)&b.residual, (void *)&n, (void *)&eps };
            hipModuleLaunchKernel(k.eval_layernorm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        } else {
            // RMSNorm (ARWKV7)
            const void * nw = lw.ptrs[0];
            int n = H;
            void * args[] = { (void *)&b.hidden, (void *)&nw,
                              (void *)&b.norm_out, (void *)&b.residual, (void *)&n };
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        }
        // b.norm_out = normalized, b.residual = input copy

        // ===== Time-mix: RWKV7 fused lerp =====
        // Baseline: sx = x_prev - cur
        float * att_shift = b.rwkv_att_shift + (long long)il * H;
        launch_sub(att_shift, b.norm_out, b.rwkv_sx, H);
        // Save current as next x_prev
        hipMemcpyAsync(att_shift, b.norm_out, H * sizeof(float), hipMemcpyDeviceToDevice, s);

        // RWKV7: xxx = lerp_fused * sx + cur (single fused lerp for 5-6 components)
        // lerp_fused is [n_embd, 5 or 6] — each column is one lerp weight
        // For decode (single token): multiply sx element-wise by each lerp column, add cur
        // Components: xr, xw, xk, xv, xa, (xg optional)
        // We compute each component separately since we don't have batch matmul
        // xr = lerp_fused[:,0] * sx + cur
        // xw = lerp_fused[:,1] * sx + cur
        // etc.

        // For simplicity, store all 5 mixed vectors in rwkv_xxx buffer
        // rwkv_xxx is [n_embd * 5] (or 6 with gating)
        // Each component: xxx_i = lerp_fused_col_i * sx + cur
        {
            const float * fused = (const float *)lw.time_mix_lerp_fused;
            int n_comp = lw.time_mix_g1 ? 6 : 5;
            for (int comp = 0; comp < n_comp; comp++) {
                float * dst = b.rwkv_xxx + comp * H;
                const float * lerp_col = fused + comp * H;  // column-major: fused[i + comp*H]
                // dst = lerp_col * sx + cur
                launch_muladd(lerp_col, b.rwkv_sx, b.norm_out, dst, H);
            }
        }
        float * xr = b.rwkv_xxx + 0 * H;
        float * xw = b.rwkv_xxx + 1 * H;
        float * xk = b.rwkv_xxx + 2 * H;
        float * xv = b.rwkv_xxx + 3 * H;
        float * xa = b.rwkv_xxx + 4 * H;
        float * xg = lw.time_mix_g1 ? (b.rwkv_xxx + 5 * H) : nullptr;

        // ===== r = receptance(xr) =====
        quant_and_launch_matvec(lw.time_mix_receptance_type, lw.time_mix_receptance,
                                 lw.time_mix_receptance_stride, xr, b.rwkv_xr, H, H, s);

        // ===== w (decay) = exp(sigmoid(W2(tanh(W1(xw))) + w0) * -0.606531) =====
        // Step 1: W1(xw) → rwkv_xw [lora_size]
        quant_and_launch_matvec(lw.time_mix_decay_w1_type, lw.time_mix_decay_w1,
                                 lw.time_mix_decay_w1_stride, xw, b.rwkv_xw, H, lora_size, s);
        // Step 2: tanh in-place
        {
            int n = lora_size;
            void * args[] = { (void *)&b.rwkv_xw, (void *)&b.rwkv_xw, (void *)&n };
            hipModuleLaunchKernel(k.eval_tanh, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
        // Step 3: W2(tanh result) → rwkv_xw becomes [H]
        // Reuse proj_scratch as temp for W2 output
        quant_and_launch_matvec(lw.time_mix_decay_w2_type, lw.time_mix_decay_w2,
                                 lw.time_mix_decay_w2_stride, b.rwkv_xw, b.proj_scratch, lora_size, H, s);
        // Step 4: + w0 bias
        if (lw.time_mix_w0) {
            launch_add(b.proj_scratch, lw.time_mix_w0, H);
        }
        // Step 5: exp(sigmoid(x) * -0.606531)
        // sigmoid in-place, then scale by -0.606531, then exp
        {
            int n = H;
            void * args[] = { (void *)&b.proj_scratch, (void *)&b.proj_scratch, (void *)&n };
            hipModuleLaunchKernel(k.eval_sigmoid, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
        // Scale by -0.606531 and exp — need a fused kernel or two ops
        // For now: scale by -0.606531 (use eval_scale_scalar) then exp
        {
            int n = H;
            float scale = -0.606531f;
            void * args[] = { (void *)&b.proj_scratch, (void *)&b.proj_scratch, (void *)&scale, (void *)&n };
            hipModuleLaunchKernel(k.eval_scale_scalar, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
        {
            int n = H;
            void * args[] = { (void *)&b.proj_scratch, (void *)&b.proj_scratch, (void *)&n };
            hipModuleLaunchKernel(k.eval_exp, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
        // proj_scratch now holds the decay w [H]

        // ===== k = key(xk) =====
        quant_and_launch_matvec(lw.time_mix_key_type, lw.time_mix_key,
                                 lw.time_mix_key_stride, xk, b.rwkv_xk, H, H, s);

        // ===== v = value(xv) =====
        quant_and_launch_matvec(lw.time_mix_value_type, lw.time_mix_value,
                                 lw.time_mix_value_stride, xv, b.rwkv_xv, H, H, s);

        // ===== v_first cross-layer residual =====
        // Baseline: if first layer, save v. Else blend with v_first via learned gate.
        if (!b.rwkv7_v_first_set) {
            // Layer 0: save v as v_first
            hipMemcpyAsync(b.rwkv7_v_first, b.rwkv_xv, H * sizeof(float), hipMemcpyDeviceToDevice, s);
            b.rwkv7_v_first_set = true;
        } else if (lw.time_mix_v1 && lw.time_mix_v2) {
            // Other layers: v = v + sigmoid(V2(V1(xv)) + v0) * (v_first - v)
            // Step 1: V1(xv) → temp [lora_size]
            quant_and_launch_matvec(lw.time_mix_v1_type, lw.time_mix_v1,
                                     lw.time_mix_v1_stride, xv, b.rwkv_xg, H, lora_size, s);
            // Step 2: V2(temp) → attn_out [H]
            quant_and_launch_matvec(lw.time_mix_v2_type, lw.time_mix_v2,
                                     lw.time_mix_v2_stride, b.rwkv_xg, b.attn_out, lora_size, H, s);
            // Step 3: + v0
            if (lw.time_mix_v0) launch_add(b.attn_out, lw.time_mix_v0, H);
            // Step 4: sigmoid
            {
                int n = H;
                void * args[] = { (void *)&b.attn_out, (void *)&b.attn_out, (void *)&n };
                hipModuleLaunchKernel(k.eval_sigmoid, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }
            // Step 5: v = v + gate * (v_first - v) = v*(1-gate) + v_first*gate
            // Compute (v_first - v) → mlp_inter
            launch_sub(b.rwkv7_v_first, b.rwkv_xv, b.mlp_inter, H);
            // gate * (v_first - v) → mlp_inter
            launch_mul(b.mlp_inter, b.attn_out, H);
            // v += result
            launch_add(b.rwkv_xv, b.mlp_inter, H);
        }

        // ===== a gate: a = sigmoid(A2(A1(xa)) + a0) =====
        float * a_gate = b.attn_out; // reuse buffer
        if (lw.time_mix_a1 && lw.time_mix_a2) {
            quant_and_launch_matvec(lw.time_mix_a1_type, lw.time_mix_a1,
                                     lw.time_mix_a1_stride, xa, b.rwkv_xg, H, lora_size, s);
            quant_and_launch_matvec(lw.time_mix_a2_type, lw.time_mix_a2,
                                     lw.time_mix_a2_stride, b.rwkv_xg, a_gate, lora_size, H, s);
            if (lw.time_mix_a0) launch_add(a_gate, lw.time_mix_a0, H);
            {
                int n = H;
                void * args[] = { (void *)&a_gate, (void *)&a_gate, (void *)&n };
                hipModuleLaunchKernel(k.eval_sigmoid, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }
        }

        // ===== kk = L2_norm(k * k_k) =====
        // k_k element-wise multiply, then per-head L2 norm
        float * kk_buf = b.mlp_inter; // reuse
        if (lw.time_mix_k_k) {
            hipMemcpyAsync(kk_buf, b.rwkv_xk, H * sizeof(float), hipMemcpyDeviceToDevice, s);
            launch_mul(kk_buf, lw.time_mix_k_k, H);
            // Per-head L2 norm: each head has head_size elements
            for (int h = 0; h < n_head; h++) {
                float * head_ptr = kk_buf + h * head_size;
                int n = head_size;
                void * args[] = { (void *)&head_ptr, (void *)&head_ptr, (void *)&n };
                int nt = (n < 256) ? 32 : 256;
                hipModuleLaunchKernel(k.eval_l2norm, 1, 1, 1, nt, 1, 1, 0, s, args, nullptr);
            }
        }

        // ===== Key modification: k = k + k_a*(a - 1) =====
        if (lw.time_mix_k_a) {
            // ka = k * k_a
            float * ka_buf = b.ssm_xz; // reuse scratch
            hipMemcpyAsync(ka_buf, b.rwkv_xk, H * sizeof(float), hipMemcpyDeviceToDevice, s);
            launch_mul(ka_buf, lw.time_mix_k_a, H);
            // k = k + a*ka - ka = k + ka*(a-1)
            // Compute a*ka → temp
            float * temp = b.ssm_x_db; // reuse scratch
            hipMemcpyAsync(temp, ka_buf, H * sizeof(float), hipMemcpyDeviceToDevice, s);
            launch_mul(temp, a_gate, H);
            // k = k + temp - ka = k + a*ka - ka
            launch_add(b.rwkv_xk, temp, H);
            launch_sub(b.rwkv_xk, ka_buf, b.rwkv_xk, H);
        }

        // ===== WKV7 recurrence =====
        // Inputs: r, w, k, v, -kk, kk*a, state
        // Output: cur [H], updated state
        {
            float * wkv_state = b.rwkv_wkv_state + (long long)il * n_head * head_size * head_size;
            float * r_ptr = b.rwkv_xr;
            float * w_ptr = b.proj_scratch;  // decay
            float * k_ptr = b.rwkv_xk;      // modified key
            float * v_ptr = b.rwkv_xv;      // value (with v_first residual)
            // -kk and kk*a: compute on the fly
            // negate kk → ssm_dt (reuse)
            float * neg_kk = b.ssm_dt;
            {
                int n = H;
                void * args[] = { (void *)&kk_buf, (void *)&neg_kk, (void *)&n };
                hipModuleLaunchKernel(k.eval_neg, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }
            // kk * a → ssm_xz (reuse)
            float * kk_a = b.ssm_xz;
            hipMemcpyAsync(kk_a, kk_buf, H * sizeof(float), hipMemcpyDeviceToDevice, s);
            launch_mul(kk_a, a_gate, H);

            int hs = head_size, nh = n_head;
            void * args[] = { (void *)&r_ptr, (void *)&w_ptr, (void *)&k_ptr,
                              (void *)&v_ptr, (void *)&neg_kk, (void *)&kk_a,
                              (void *)&wkv_state, (void *)&b.norm_out, // output into norm_out
                              (void *)&hs, (void *)&nh };
            // WKV7 uses extern __shared__ float smem7[] with 5 arrays of head_size floats
            size_t wkv7_smem = 5 * head_size * sizeof(float);
            hipModuleLaunchKernel(k.eval_rwkv_wkv7_step, n_head, 1, 1, head_size, 1, 1,
                                 wkv7_smem, s, args, nullptr);
        }
        // norm_out now has WKV7 output [H]

        // ===== Group norm =====
        if (lw.time_mix_ln) {
            for (int h = 0; h < n_head; h++) {
                float * head_ptr = b.norm_out + h * head_size;
                const float * ln_w = (const float *)lw.time_mix_ln + h * head_size;
                const float * ln_b = (const float *)lw.time_mix_ln_b + h * head_size;
                int n = head_size;
                float eps = 64e-5f;  // baseline uses 64e-5 for RWKV group norm
                void * args[] = { (void *)&head_ptr, (void *)&ln_w, (void *)&ln_b,
                                  (void *)&head_ptr, (void *)&head_ptr, (void *)&n, (void *)&eps };
                int nt = (n < 256) ? 32 : 256;
                hipModuleLaunchKernel(k.eval_layernorm, 1, 1, 1, nt, 1, 1, 0, s, args, nullptr);
            }
        }

        // ===== r*k additive output correction =====
        // Baseline: rk = sum_rows(k * r * r_k); cur += v * rk
        if (lw.time_mix_r_k) {
            // r*k additive correction — ported from baseline rwkv7-base.cpp lines 127-129
            // Per-head: rk_dot = sum(r[h]*k[h]*r_k[h]); output[h] += v[h] * rk_dot
            int hs = head_size, nh = n_head;
            const void * rk_w = lw.time_mix_r_k;
            size_t smem = (hs < 256 ? 256 : hs) / 32 * sizeof(float); // for warp reduce
            void * args[] = { (void *)&b.norm_out, (void *)&b.rwkv_xr, (void *)&b.rwkv_xk,
                              (void *)&b.rwkv_xv, (void *)&rk_w, (void *)&hs, (void *)&nh };
            int threads = (hs < 256) ? hs : 256;
            hipModuleLaunchKernel(k.eval_rwkv7_rk_correction, nh, 1, 1, threads, 1, 1,
                                 smem, s, args, nullptr);
        }

        // ===== Optional gating =====
        if (xg && lw.time_mix_g1 && lw.time_mix_g2) {
            // g = G2(sigmoid(G1(xg)))
            quant_and_launch_matvec(lw.time_mix_g1_type, lw.time_mix_g1,
                                     lw.time_mix_g1_stride, xg, b.rwkv_xg, H, lora_size, s);
            {
                int n = lora_size;
                void * args[] = { (void *)&b.rwkv_xg, (void *)&b.rwkv_xg, (void *)&n };
                hipModuleLaunchKernel(k.eval_sigmoid, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }
            quant_and_launch_matvec(lw.time_mix_g2_type, lw.time_mix_g2,
                                     lw.time_mix_g2_stride, b.rwkv_xg, b.attn_out, lora_size, H, s);
            // cur *= g
            launch_mul(b.norm_out, b.attn_out, H);
        }

        // ===== Output projection + residual =====
        {
            int n = H, q8blocks = (n+511)/512;
            void * q8args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
        }
        launch_matvec_res_typed(lw.time_mix_output_type, lw.time_mix_output,
                                 lw.time_mix_output_stride, b.residual, b.hidden, H, H, s);

        // ===== FFN norm =====
        // RWKV7: LayerNorm. ARWKV7: RMSNorm (no bias on FFN norm per baseline arwkv7.cpp line 53).
        {
            int post_norm_idx = 7;
            const void * nw = lw.ptrs[post_norm_idx];
            int n = H;
            float eps = c.norm_eps;
            if (c.norm_type == 2) {
                // LayerNorm (RWKV7)
                const void * nb = lw.ffn_norm_bias;
                void * args[] = { (void *)&b.hidden, (void *)&nw, (void *)&nb,
                                  (void *)&b.norm_out, (void *)&b.residual, (void *)&n, (void *)&eps };
                hipModuleLaunchKernel(k.eval_layernorm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
            } else {
                // RMSNorm (ARWKV7)
                void * args[] = { (void *)&b.hidden, (void *)&nw,
                                  (void *)&b.norm_out, (void *)&b.residual, (void *)&n };
                hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
            }
        }

        // ===== FFN dispatch =====
        // ARWKV7: standard SiLU-gated FFN (gate+up+silu_mul+down) — baseline arwkv7.cpp lines 52-66
        // RWKV7: channel_mix (sqr(relu(K(xk))), then V(k)) — baseline rwkv7.cpp lines 73-88
        if (c.norm_type != 2 && lw.ptrs[8]) {
            // ARWKV7: standard FFN — gate=ptrs[8], up=ptrs[9], down=ptrs[10]
            int FF = c.intermediate_size;
            {
                int n = H, q8blocks = (n+511)/512;
                void * q8args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
                hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
            }
            // Gate: [FF]
            launch_matvec_typed(lw.types[8], lw.ptrs[8], lw.strides[8], b.mlp_inter, H, FF, s);
            // Up: [FF]
            launch_matvec_typed(lw.types[9], lw.ptrs[9], lw.strides[9], b.proj_scratch, H, FF, s);
            // SiLU(gate) * up
            {
                int n = FF;
                void * args[] = { (void *)&b.mlp_inter, (void *)&b.proj_scratch, (void *)&b.mlp_inter, (void *)&n };
                hipModuleLaunchKernel(k.eval_silu_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }
            // Down + residual
            {
                int n = FF, q8blocks = (n+511)/512;
                void * q8args[] = { (void *)&b.mlp_inter, (void *)&b.q8_act, (void *)&n };
                hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
            }
            launch_matvec_res_typed(lw.types[10], lw.ptrs[10], lw.strides[10],
                                     b.residual, b.hidden, FF, H, s);
        } else {
        // ===== Channel-mix (RWKV7 simplified — no receptance) =====
        // Baseline: k = sqr(relu(K(lerp_k*sx + cur))); cur = V(k)
        float * ffn_shift = b.rwkv_ffn_shift + (long long)il * H;
        launch_sub(ffn_shift, b.norm_out, b.rwkv_sx, H);
        hipMemcpyAsync(ffn_shift, b.norm_out, H * sizeof(float), hipMemcpyDeviceToDevice, s);

        // xk = lerp_k * sx + cur
        if (lw.channel_mix_lerp_k) {
            launch_muladd((const float *)lw.channel_mix_lerp_k, b.rwkv_sx, b.norm_out, b.rwkv_xk, H);
        } else {
            hipMemcpyAsync(b.rwkv_xk, b.norm_out, H * sizeof(float), hipMemcpyDeviceToDevice, s);
        }

        // k = sqr(relu(K(xk)))
        quant_and_launch_matvec(lw.channel_mix_key_type, lw.channel_mix_key,
                                 lw.channel_mix_key_stride, b.rwkv_xk, b.proj_scratch, H, H, s);
        {
            int n = H;
            void * args[] = { (void *)&b.proj_scratch, (void *)&b.proj_scratch, (void *)&n };
            hipModuleLaunchKernel(k.eval_relu, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
        {
            int n = H;
            void * args[] = { (void *)&b.proj_scratch, (void *)&b.proj_scratch, (void *)&n };
            hipModuleLaunchKernel(k.eval_sqr, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }

        // cur = V(k) + residual
        {
            int n = H, q8blocks = (n+511)/512;
            void * q8args[] = { (void *)&b.proj_scratch, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
        }
        launch_matvec_res_typed(lw.channel_mix_value_type, lw.channel_mix_value,
                                 lw.channel_mix_value_stride, b.residual, b.hidden, H, H, s);
        } // end else (RWKV7 channel_mix)
    }

    // Final norm — RWKV7: LayerNorm, ARWKV7: RMSNorm (with bias per baseline arwkv7.cpp line 75)
    if (c.norm_type == 2) {
        const void * nw = c.final_norm_weight;
        const void * nb = c.final_norm_bias;
        float eps = c.norm_eps;
        int n = H;
        void * args[] = { (void *)&b.hidden, (void *)&nw, (void *)&nb,
                          (void *)&b.norm_out, (void *)&b.residual, (void *)&n, (void *)&eps };
        hipModuleLaunchKernel(k.eval_layernorm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
    } else {
        const void * nw = c.final_norm_weight;
        int n = H;
        void * args[] = { (void *)&b.hidden, (void *)&nw,
                          (void *)&b.norm_out, (void *)&b.residual, (void *)&n };
        hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
    }
    {
        int n = H, blocks = (n + 511) / 512;
        void * args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
        hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, args, nullptr);
    }
    launch_matvec_typed(c.lm_head_type, c.lm_head_weight, c.lm_head_stride,
                         b.logits, H, V, s);
    if (logits_out) {
        hipMemcpyAsync(logits_out, b.logits, V * sizeof(float), hipMemcpyDeviceToHost, s);
    }
    hipStreamSynchronize(s);
    return 0;
}
