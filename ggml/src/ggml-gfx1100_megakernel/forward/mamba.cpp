// forward/mamba.cpp — Mamba/SSM forward
#include "../gfx1100-internal.h"

int forward_decode_mamba(int token_id, int position, float * logits_out) {
    auto & c = g_config;
    auto & b = g_bufs;
    auto & k = g_compiled;
    auto   s = b.stream;
    int    H = c.hidden_size;
    int    V = c.vocab_size;
    int    d_inner = c.ssm_d_inner;
    int    d_state = c.ssm_d_state;
    int    d_conv  = c.ssm_d_conv;
    int    dt_rank = c.ssm_dt_rank;

    const int norm_threads = (H < 1024) ? 256 : 1024;

    // Embedding — full type dispatch via shared launch_embed
    if (launch_embed(token_id, b.hidden, s) != 0) return -1;

    // Layer loop
    for (int il = 0; il < c.n_layers; il++) {
        const gfx1100_layer_weights & lw = c.layers[il];

        // Phase 1: RMSNorm — baseline mamba.cpp line 16: build_norm(inpL, attn_norm, NULL, LLM_NORM_RMS)
        {
            const void * norm_w = lw.ptrs[0];
            int n = H;
            void * args[] = { (void *)&b.hidden, (void *)&norm_w, (void *)&b.norm_out,
                              (void *)&b.residual, (void *)&n };
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        }

        // Phase 2: Input projection — baseline line 45: xz = ssm_in @ cur → [2*d_inner]
        // Quantize norm_out first
        {
            int n = H;
            int blocks = (n + 511) / 512;
            void * args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, args, nullptr);
        }
        // ssm_in matvec: [2*d_inner, H] @ q8_act[H] → ssm_xz[2*d_inner]
        launch_matvec_typed(lw.ssm_in_type, lw.ssm_in, lw.ssm_in_stride,
                            b.ssm_xz, H, 2 * d_inner, s);
        // ssm_xz now has x[0..d_inner-1] and z[d_inner..2*d_inner-1]

        // Phase 3: Conv1d step — baseline lines 53-81
        // Shift conv state, append x, convolve, add bias, silu
        // Conv state for layer il: b.ssm_conv_states + il * d_inner * (d_conv-1)
        // Kernel: eval_ssm_conv_step(x_in, state, w, y_out, d_inner, d_conv, apply_silu)
        {
            float * conv_state = b.ssm_conv_states + (long long)il * d_inner * (d_conv - 1);
            float * x_in = b.ssm_xz; // first d_inner elements
            const void * conv_w = lw.ssm_conv1d;   // [d_conv, d_inner]
            int n = d_inner;
            int dc = d_conv;
            int apply_silu_val = 0; // apply silu separately after bias
            void * args[] = { (void *)&x_in, (void *)&conv_state, (void *)&conv_w,
                              (void *)&x_in, (void *)&n, (void *)&dc, (void *)&apply_silu_val };
            hipModuleLaunchKernel(k.eval_ssm_conv_step, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
        // Add conv bias — baseline: ggml_add(conv_out, conv1d_bias)
        {
            const void * conv_b = lw.ssm_conv1d_b;
            int n = d_inner;
            float * x_in = b.ssm_xz;
            void * ba[] = { (void *)&x_in, (void *)&conv_b, (void *)&x_in, (void *)&n };
            hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, s, ba, nullptr);
        }
        // Apply SiLU — baseline: ggml_silu(conv_out) for Mamba1
        {
            float * x_in = b.ssm_xz;
            int n = d_inner;
            void * sa[] = { (void *)&x_in, (void *)&x_in, (void *)&n };
            hipModuleLaunchKernel(k.eval_silu, (n+255)/256, 1, 1, 256, 1, 1, 0, s, sa, nullptr);
        }
        // x_in (= ssm_xz[0..d_inner-1]) now has the conv1d output with silu applied

        // Phase 4: x projection — baseline line 86: x_db = ssm_x @ x → [dt_rank + 2*d_state]
        // quant_and_launch_matvec handles quantization internally
        quant_and_launch_matvec(lw.ssm_x_type, lw.ssm_x, lw.ssm_x_stride,
                                 b.ssm_xz, b.ssm_x_db, d_inner, dt_rank + 2 * d_state, s);
        // ssm_x_db has: dt[0..dt_rank-1], B[dt_rank..dt_rank+d_state-1], C[dt_rank+d_state..]

        // Phase 5: dt projection — baseline line 103-105: dt = ssm_dt @ dt + dt_bias → [d_inner]
        quant_and_launch_matvec(lw.ssm_dt_type, lw.ssm_dt, lw.ssm_dt_stride,
                                 b.ssm_x_db, b.ssm_dt, dt_rank, d_inner, s);
        // Add dt bias
        {
            const void * bias = lw.ssm_dt_b;
            int n = d_inner;
            void * dst = (void *)b.ssm_dt;
            void * args[] = { (void *)&dst, (void *)&bias, (void *)&dst, (void *)&n };
            hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }

        // FalconMamba dt/B/C RMSNorm — baseline mamba-base.cpp lines 96-101
        // Applied when ssm_dt_norm, ssm_b_norm, ssm_c_norm weights are present
        if (lw.ssm_dt_norm) {
            int n = d_inner;
            void * args[] = { (void *)&b.ssm_dt, (void *)&lw.ssm_dt_norm,
                              (void *)&b.ssm_dt, (void *)&b.ssm_dt, (void *)&n };
            int nt = (n < 1024) ? 256 : 1024;
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, nt, 1, 1, 0, s, args, nullptr);
        }
        if (lw.ssm_b_norm) {
            float * B_ptr = b.ssm_x_db + dt_rank;
            int n = d_state;
            void * args[] = { (void *)&B_ptr, (void *)&lw.ssm_b_norm,
                              (void *)&B_ptr, (void *)&B_ptr, (void *)&n };
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
        if (lw.ssm_c_norm) {
            float * C_ptr = b.ssm_x_db + dt_rank + d_state;
            int n = d_state;
            void * args[] = { (void *)&C_ptr, (void *)&lw.ssm_c_norm,
                              (void *)&C_ptr, (void *)&C_ptr, (void *)&n };
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }

        // Phase 6: SSM scan step — baseline lines 110-131
        // s' = exp(A * softplus(dt)) * s + B * x; y = C * s' + D * x
        // State: b.ssm_scan_states + il * d_inner * d_state
        {
            float * scan_state = b.ssm_scan_states + (long long)il * d_inner * d_state;
            float * x_in = b.ssm_xz;                    // x after conv1d+silu [d_inner]
            float * dt_buf = b.ssm_dt;                   // dt after projection [d_inner]
            const void * A = lw.ssm_a;                   // [d_inner, d_state]
            float * B = b.ssm_x_db + dt_rank;            // B from x_db [d_state]
            float * C = b.ssm_x_db + dt_rank + d_state;  // C from x_db [d_state]
            const void * D_param = lw.ssm_d;              // [d_inner]
            float * y_out = b.proj_scratch;               // output [d_inner]
            int di = d_inner, ds = d_state;
            // Kernel: eval_ssm_scan_step(x, dt, A, B, C, D, h, y, d_inner, d_state)
            void * args[] = { (void *)&x_in, (void *)&dt_buf,
                              (void *)&A, (void *)&B, (void *)&C, (void *)&D_param,
                              (void *)&scan_state, (void *)&y_out, (void *)&di, (void *)&ds };
            hipModuleLaunchKernel(k.eval_ssm_scan_step, (di+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }

        // Phase 7: Output gate — baseline line 137: y = swiglu_split(z, y) = silu(z) * y
        // z is ssm_xz[d_inner..2*d_inner-1], y is proj_scratch[0..d_inner-1]
        {
            float * z_ptr = b.ssm_xz + d_inner;
            float * y_ptr = b.proj_scratch;
            int n = d_inner;
            // silu_mul: out = silu(a) * b → we need silu(z) * y
            // eval_silu_mul does: out = silu(gate) * up
            // So: gate=z, up=y, out=y
            void * args[] = { (void *)&z_ptr, (void *)&y_ptr, (void *)&y_ptr, (void *)&n };
            hipModuleLaunchKernel(k.eval_silu_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }

        // Phase 8: Output projection + residual — baseline line 140: cur = ssm_out @ y + residual
        {
            int n = d_inner;
            int q8blocks = (n + 511) / 512;
            void * q8args[] = { (void *)&b.proj_scratch, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
        }
        launch_matvec_res_typed(lw.ssm_out_type, lw.ssm_out, lw.ssm_out_stride,
                                 b.residual, b.hidden, d_inner, H, s);
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
        int n = H;
        int blocks = (n + 511) / 512;
        void * args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
        hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, args, nullptr);
    }

    // LM head matvec — proper type dispatch
    launch_matvec_typed(c.lm_head_type, c.lm_head_weight, c.lm_head_stride,
                         b.logits, H, V, s);

    // Copy logits
    if (logits_out) {
        hipMemcpyAsync(logits_out, b.logits, V * sizeof(float),
                       hipMemcpyDeviceToHost, s);
    }
    hipStreamSynchronize(s);
    return 0;
}

// ----------------------------------------------------------------------------
// forward_decode_mamba2 — Mamba2 (SSD parameterization)
// Ported from baseline src/models/mamba-base.cpp::build_mamba2_layer
//
// Key differences from Mamba1:
//   - Input projection splits into z, xBC (fused), dt (not x, z)
//   - Conv1d operates on wider xBC tensor (d_inner + 2*n_group*d_state)
//   - After conv+silu, xBC is split into x, B, C by slicing (no ssm_x projection)
//   - dt comes directly from input projection (n_head values, no up-projection)
//   - Multi-head SSM: n_head = ssm_dt_rank, head_dim = d_inner / n_head
//   - Optional grouped RMS norm after skip-gating
// Same kernels: eval_ssm_conv_step, eval_ssm_scan_step (just different widths)
int forward_decode_mamba2(int token_id, int position, float * logits_out) {
    auto & c = g_config;
    auto & b = g_bufs;
    auto & k = g_compiled;
    auto   s = b.stream;
    int    H = c.hidden_size;
    int    V = c.vocab_size;
    int    d_inner = c.ssm_d_inner;
    int    d_state = c.ssm_d_state;
    int    d_conv  = c.ssm_d_conv;
    int    n_head  = c.ssm_dt_rank;   // Mamba2: n_head = dt_rank
    int    n_group = c.ssm_n_group;
    // Mamba2 conv operates on xBC: d_inner + 2 * n_group * d_state
    int    xBC_dim = d_inner + 2 * n_group * d_state;
    // Input projection: 2*d_inner + 2*n_group*d_state + n_head
    int    proj_dim = 2 * d_inner + 2 * n_group * d_state + n_head;

    const int norm_threads = (H < 1024) ? 256 : 1024;

    // Embedding
    if (launch_embed(token_id, b.hidden, s) != 0) return -1;

    for (int il = 0; il < c.n_layers; il++) {
        const gfx1100_layer_weights & lw = c.layers[il];

        // RMSNorm — baseline: build_norm(inpL, attn_norm, NULL, LLM_NORM_RMS)
        {
            const void * norm_w = lw.ptrs[0]; int n = H;
            void * args[] = { (void *)&b.hidden, (void *)&norm_w, (void *)&b.norm_out,
                              (void *)&b.residual, (void *)&n };
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        }

        // Input projection → [proj_dim] = [2*d_inner + 2*n_group*d_state + n_head]
        // Baseline: cur = ssm_in @ x → then split into z, xBC, dt
        quant_and_launch_matvec(lw.ssm_in_type, lw.ssm_in, lw.ssm_in_stride,
                                 b.norm_out, b.proj_scratch, H, proj_dim, s);

        // Split: proj_scratch[0..d_inner-1] = z (gate)
        //        proj_scratch[d_inner..d_inner+xBC_dim-1] = xBC (conv input)
        //        proj_scratch[d_inner+xBC_dim..proj_dim-1] = dt [n_head]
        float * z_ptr = b.proj_scratch;
        float * xBC_ptr = b.proj_scratch + d_inner;
        float * dt_ptr = b.proj_scratch + d_inner + xBC_dim;

        // Copy xBC to ssm_xz for conv (conv modifies in-place)
        hipMemcpyAsync(b.ssm_xz, xBC_ptr, xBC_dim * sizeof(float), hipMemcpyDeviceToDevice, s);

        // Conv1d step on xBC — wider than Mamba1 (d_inner + 2*n_group*d_state)
        // Conv state for layer il: b.ssm_conv_states + il * xBC_dim * (d_conv-1)
        // Kernel: eval_ssm_conv_step(x_in, state, w, y_out, d_inner, d_conv, apply_silu)
        {
            float * conv_state = b.ssm_conv_states + (long long)il * xBC_dim * (d_conv - 1);
            float * x_in = b.ssm_xz;
            const void * conv_w = lw.ssm_conv1d;
            int n = xBC_dim, dc = d_conv;
            int apply_silu_val = 0; // apply silu separately after bias
            void * args[] = { (void *)&x_in, (void *)&conv_state, (void *)&conv_w,
                              (void *)&x_in, (void *)&n, (void *)&dc, (void *)&apply_silu_val };
            hipModuleLaunchKernel(k.eval_ssm_conv_step, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
        // Add conv bias — baseline: ggml_add(conv_out, conv1d_bias)
        {
            const void * conv_b = lw.ssm_conv1d_b;
            float * x_in = b.ssm_xz;
            int n = xBC_dim;
            void * ba[] = { (void *)&x_in, (void *)&conv_b, (void *)&x_in, (void *)&n };
            hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, s, ba, nullptr);
        }
        // Apply SiLU — baseline: ggml_silu(conv_out)
        {
            float * x_in = b.ssm_xz;
            int n = xBC_dim;
            void * sa[] = { (void *)&x_in, (void *)&x_in, (void *)&n };
            hipModuleLaunchKernel(k.eval_silu, (n+255)/256, 1, 1, 256, 1, 1, 0, s, sa, nullptr);
        }

        // After conv+silu, split xBC → x, B, C
        // ssm_xz[0..d_inner-1] = x
        // ssm_xz[d_inner..d_inner+n_group*d_state-1] = B
        // ssm_xz[d_inner+n_group*d_state..xBC_dim-1] = C
        float * x_post_conv = b.ssm_xz;
        float * B_ptr = b.ssm_xz + d_inner;
        float * C_ptr = b.ssm_xz + d_inner + n_group * d_state;

        // dt: add bias directly to the n_head values (no up-projection)
        {
            const void * bias = lw.ssm_dt_b;
            int n = n_head;
            void * args[] = { (void *)&dt_ptr, (void *)&bias, (void *)&dt_ptr, (void *)&n };
            hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }

        // SSM scan step — Mamba2 multi-head
        // dt is [n_head] but the scan kernel expects [d_inner]. Expand dt by repeating
        // each head's value head_dim times into b.ssm_dt.
        {
            int head_dim = d_inner / n_head;
            int di = d_inner;
            void * args[] = { (void *)&b.ssm_dt, (void *)&dt_ptr, (void *)&head_dim, (void *)&di };
            hipModuleLaunchKernel(k.eval_repeat_interleave, (di+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
        {
            float * scan_state = b.ssm_scan_states + (long long)il * d_inner * d_state;
            float * y_out = b.mlp_inter;
            float * dt_expanded = b.ssm_dt;
            int di = d_inner, ds = d_state;
            const void * A = lw.ssm_a;
            const void * D_param = lw.ssm_d;
            // Kernel: eval_ssm_scan_step(x, dt, A, B, C, D, h, y, d_inner, d_state)
            void * args[] = { (void *)&x_post_conv, (void *)&dt_expanded,
                              (void *)&A, (void *)&B_ptr, (void *)&C_ptr, (void *)&D_param,
                              (void *)&scan_state, (void *)&y_out, (void *)&di, (void *)&ds };
            hipModuleLaunchKernel(k.eval_ssm_scan_step, (di+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }

        // Output gate: silu(z) * y — baseline: swiglu_split(z, y)
        {
            float * y = b.mlp_inter;
            int n = d_inner;
            void * args[] = { (void *)&z_ptr, (void *)&y, (void *)&y, (void *)&n };
            hipModuleLaunchKernel(k.eval_silu_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }

        // Optional grouped RMS norm — baseline: if (ssm_norm) group_norm(y)
        if (lw.ssm_norm) {
            // Grouped RMSNorm: reshape [d_inner] as [d_inner/n_group, n_group]
            // Norm each group independently
            int group_size = d_inner / n_group;
            for (int g = 0; g < n_group; g++) {
                float * group_ptr = b.mlp_inter + g * group_size;
                const float * norm_w = (const float *)lw.ssm_norm + g * group_size;
                int n = group_size;
                float eps = c.norm_eps;
                // Inline RMSNorm for each group
                void * args[] = { (void *)&group_ptr, (void *)&norm_w, (void *)&group_ptr,
                                  (void *)&group_ptr, (void *)&n };
                int nt = (n < 1024) ? 256 : 1024;
                hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, nt, 1, 1, 0, s, args, nullptr);
                // Note: eval_rmsnorm_q8 writes residual=input, norm_out=normalized
                // We reuse the same buffer so this normalizes in-place via the residual copy
            }
        }

        // Output projection + residual
        {
            int n = d_inner, q8blocks = (n + 511) / 512;
            void * q8args[] = { (void *)&b.mlp_inter, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
        }
        launch_matvec_res_typed(lw.ssm_out_type, lw.ssm_out, lw.ssm_out_stride,
                                 b.residual, b.hidden, d_inner, H, s);
    }

    // Final norm + LM head (same as Mamba1)
    {
        int n = H;
        void * args[] = { (void *)&b.hidden, (void *)&c.final_norm_weight,
                          (void *)&b.norm_out, (void *)&n };
        hipModuleLaunchKernel(k.eval_final_norm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
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

// ----------------------------------------------------------------------------
// forward_decode_llama_family
//
// Dense transformer decode (Llama/Qwen2/Qwen3/Mistral/etc.). Covers archs
// whose per-layer structure matches baseline src/models/llama.cpp::build_llama:
//   RMSNorm → {Q,K,V proj (+bias)(+scale)} → QK norm (opt) → RoPE → KV write →
//   flash-attn decode → O proj (+bo)(+wo_s) → residual → RMSNorm →
//   {gate,up proj (+bias)(+scale)} → SiLU-mul → down proj → residual.
// Also supports per-layer DeltaNet (Qwen35) via layer_types[il] == 1.
//
// This is the existing hand-ported path that matches baseline 10/10 argmax
// on Llama 3.2 1B Q4_K_M. Arch-specific paths (MoE, SSM, RWKV, MLA, Gemma
// softcap, Phi parallel FFN, etc.) live in their own forward_decode_* fn.
