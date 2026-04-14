// shared/dispatch.cpp — shared matvec/embed/SSM dispatch functions
#include "../gfx1100-internal.h"


bool is_float_type(int type) { return type == 0 || type == 1 || type == 30; }

hipFunction_t pick_matvec(int type) {
    auto & k = g_compiled;
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
}

hipFunction_t pick_matvec_res(int type) {
    auto & k = g_compiled;
    switch (type) {
        case  0: return k.eval_matvec_f32_residual;
        case  1: return k.eval_matvec_f16_residual;
        case 30: return k.eval_matvec_bf16_residual;
        case  2: return k.eval_matvec_q4_0_residual;
        case  3: return k.eval_matvec_q4_1_residual;
        case  6: return k.eval_matvec_q5_0_residual;
        case  7: return k.eval_matvec_q5_1_residual;
        case  8: return k.eval_matvec_q8_0_residual;
        case 10: return k.eval_matvec_q2k_residual;
        case 11: return k.eval_matvec_q3k_residual;
        case 12: return k.eval_matvec_q4k_residual;
        case 13: return k.eval_matvec_q5k_residual;
        case 14: return k.eval_matvec_q6k_residual;
        case 16: return k.eval_matvec_iq2_xxs_residual;
        case 17: return k.eval_matvec_iq2_xs_residual;
        case 18: return k.eval_matvec_iq3_xxs_residual;
        case 19: return k.eval_matvec_iq1_s_residual;
        case 20: return k.eval_matvec_iq4_nl_residual;
        case 21: return k.eval_matvec_iq3_s_residual;
        case 22: return k.eval_matvec_iq2_s_residual;
        case 23: return k.eval_matvec_iq4_xs_residual;
        case 29: return k.eval_matvec_iq1_m_residual;
        case 39: return k.eval_matvec_mxfp4_residual;
        case 40: return k.eval_matvec_nvfp4_residual;
        default: return k.eval_matvec_q4k_residual;
    }
}

// 8-warp variants — returns non-null for types that benefit on RDNA3
// 8-warp variants — returns non-null for types that benefit on RDNA3
static hipFunction_t pick_matvec_8w(int type) {
    auto & k = g_compiled;
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
}

static hipFunction_t pick_matvec_8w_res(int type) {
    auto & k = g_compiled;
    switch (type) {
        case  2: return k.eval_matvec_q4_0_8w_residual;
        case  3: return k.eval_matvec_q4_1_8w_residual;
        case  6: return k.eval_matvec_q5_0_8w_residual;
        case  7: return k.eval_matvec_q5_1_8w_residual;
        case  8: return k.eval_matvec_q8_0_8w_residual;
        case 12: return k.eval_matvec_q4k_8w_residual;
        case 14: return k.eval_matvec_q6k_8w_residual;
        case 20: return k.eval_matvec_iq4_nl_8w_residual;
        default: return nullptr;
    }
}

void launch_matvec_typed(int type, const void * w, long long st,
                                 float * output, int in_dim, int out_dim,
                                 hipStream_t stream) {
    bool is_f = is_float_type(type);
    const void * input = is_f ? (const void *)g_bufs.norm_out : (const void *)g_bufs.q8_act;
    void * args[] = { (void *)&w, (void *)&st, (void *)&input,
                      (void *)&output, (void *)&in_dim, (void *)&out_dim };
    if (is_f) {
        hipModuleLaunchKernel(pick_matvec(type), out_dim, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
    } else {
        hipFunction_t fn8w = pick_matvec_8w(type);
        if (fn8w) {
            hipModuleLaunchKernel(fn8w, out_dim, 1, 1, 32, 8, 1, 0, stream, args, nullptr);
        } else {
            hipModuleLaunchKernel(pick_matvec(type), out_dim, 1, 1, 32, 4, 1, 0, stream, args, nullptr);
        }
    }
}

void launch_matvec_res_typed(int type, const void * w, long long st,
                                     float * residual, float * output, int in_dim, int out_dim,
                                     hipStream_t stream) {
    bool is_f = is_float_type(type);
    const void * input = is_f ? (const void *)g_bufs.norm_out : (const void *)g_bufs.q8_act;
    void * args[] = { (void *)&w, (void *)&st, (void *)&input,
                      (void *)&residual, (void *)&output, (void *)&in_dim, (void *)&out_dim };
    if (is_f) {
        hipModuleLaunchKernel(pick_matvec_res(type), out_dim, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
    } else {
        hipFunction_t fn8w = pick_matvec_8w_res(type);
        if (fn8w) {
            hipModuleLaunchKernel(fn8w, out_dim, 1, 1, 32, 8, 1, 0, stream, args, nullptr);
        } else {
            hipModuleLaunchKernel(pick_matvec_res(type), out_dim, 1, 1, 32, 4, 1, 0, stream, args, nullptr);
        }
    }
}

void quant_and_launch_matvec(int type, const void * w, long long st,
                                     float * input_f32, float * output,
                                     int in_dim, int out_dim, hipStream_t stream) {
    // Quantize input
    int n = in_dim, blocks = (n + 511) / 512;
    void * q8args[] = { (void *)&input_f32, (void *)&g_bufs.q8_act, (void *)&n };
    hipModuleLaunchKernel(g_compiled.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, stream, q8args, nullptr);
    // Launch matvec (uses q8_act as input for quantized, input_f32 via norm_out for float)
    // For simplicity, always use q8_act since we just quantized
    bool is_f = is_float_type(type);
    const void * mv_input = is_f ? (const void *)input_f32 : (const void *)g_bufs.q8_act;
    void * args[] = { (void *)&w, (void *)&st, (void *)&mv_input,
                      (void *)&output, (void *)&in_dim, (void *)&out_dim };
    if (is_f) {
        hipModuleLaunchKernel(pick_matvec(type), out_dim, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
    } else {
        hipFunction_t fn8w = pick_matvec_8w(type);
        if (fn8w) {
            hipModuleLaunchKernel(fn8w, out_dim, 1, 1, 32, 8, 1, 0, stream, args, nullptr);
        } else {
            hipModuleLaunchKernel(pick_matvec(type), out_dim, 1, 1, 32, 4, 1, 0, stream, args, nullptr);
        }
    }
}

int launch_embed(int token_id, float * output, hipStream_t stream) {
    auto & c = g_config;
    auto & k = g_compiled;
    auto & b = g_bufs;
    int H = c.hidden_size;
    const void * embed_ptr = c.embed_weight;
    long long stride = c.embed_stride;
    // Write token_id to d_decode_params[0] so the embedding kernel can read it.
    // For forward files with hipGraph, d_decode_params is already written before
    // this call (with both token_id and position); this is a harmless re-write.
    hipMemcpyAsync(b.d_decode_params, &token_id, sizeof(int),
                   hipMemcpyHostToDevice, stream);
    // Embedding kernels read token_id from d_decode_params[0] (GPU-resident pointer)
    // for hipGraph reuse — stable pointer that doesn't change between tokens.
    void * args[] = { (void *)&embed_ptr, (void *)&stride, (void *)&output, (void *)&b.d_decode_params };

    hipFunction_t embed_fn; int embed_threads;
    switch (c.embed_type) {
        case  2: embed_fn = k.eval_embed_q4_0;    embed_threads = 256; break;
        case  3: embed_fn = k.eval_embed_q4_1;    embed_threads = 256; break;
        case  6: embed_fn = k.eval_embed_q5_0;    embed_threads = 256; break;
        case  7: embed_fn = k.eval_embed_q5_1;    embed_threads = 256; break;
        case  8: embed_fn = k.eval_embed_q8_0;    embed_threads = 256; break;
        case 10: embed_fn = k.eval_embed_q2k;     embed_threads = 64; break;
        case 11: embed_fn = k.eval_embed_q3k;     embed_threads = 64; break;
        case 12: embed_fn = k.eval_embed_q4k;     embed_threads = 32; break;
        case 13: embed_fn = k.eval_embed_q5k;     embed_threads = 64; break;
        case 14: embed_fn = k.eval_embed_q6k;     embed_threads = 64; break;
        case 16: embed_fn = k.eval_embed_iq2_xxs; embed_threads = 32; break;
        case 17: embed_fn = k.eval_embed_iq2_xs;  embed_threads = 32; break;
        case 18: embed_fn = k.eval_embed_iq3_xxs; embed_threads = 32; break;
        case 19: embed_fn = k.eval_embed_iq1_s;   embed_threads = 32; break;
        case 20: embed_fn = k.eval_embed_iq4_nl;  embed_threads = 32; break;
        case 21: embed_fn = k.eval_embed_iq3_s;   embed_threads = 32; break;
        case 22: embed_fn = k.eval_embed_iq2_s;   embed_threads = 32; break;
        case 23: embed_fn = k.eval_embed_iq4_xs;  embed_threads = 32; break;
        case 29: embed_fn = k.eval_embed_iq1_m;   embed_threads = 32; break;
        case 39: embed_fn = k.eval_embed_mxfp4;   embed_threads = 32; break;
        case 40: embed_fn = k.eval_embed_nvfp4;   embed_threads = 32; break;
        case  0: embed_fn = k.eval_embed_f32;     embed_threads = 256; break;
        case  1: embed_fn = k.eval_embed_f16;     embed_threads = 256; break;
        case 30: embed_fn = k.eval_embed_bf16;    embed_threads = 256; break;
        default:
            fprintf(stderr, "gfx1100: FATAL — unsupported embed type %d\n", c.embed_type);
            return -1;
    }
    const bool is_small_q = (c.embed_type == 2 || c.embed_type == 3 ||
                              c.embed_type == 6 || c.embed_type == 7 || c.embed_type == 8);
    if (c.embed_type == 0 || c.embed_type == 1 || c.embed_type == 30) {
        int grid = (H + embed_threads - 1) / embed_threads;
        hipModuleLaunchKernel(embed_fn, grid, 1, 1, embed_threads, 1, 1, 0, stream, args, nullptr);
    } else if (is_small_q) {
        int grid_y = (H + 2 * embed_threads - 1) / (2 * embed_threads);
        hipModuleLaunchKernel(embed_fn, 1, grid_y, 1, embed_threads, 1, 1, 0, stream, args, nullptr);
    } else {
        int nb = H / 256;
        hipModuleLaunchKernel(embed_fn, nb > 0 ? nb : 1, 1, 1, embed_threads, 1, 1, 0, stream, args, nullptr);
    }
    return 0;
}

void ssm_layer_step(int il, hipStream_t stream) {
    auto & c = g_config;
    auto & b = g_bufs;
    auto & k = g_compiled;
    int H = c.hidden_size;
    int d_inner = c.ssm_d_inner, d_state = c.ssm_d_state;
    int d_conv = c.ssm_d_conv, dt_rank = c.ssm_dt_rank;
    const gfx1100_layer_weights & lw = c.layers[il];

    // ssm_in @ cur → [2*d_inner]
    quant_and_launch_matvec(lw.ssm_in_type, lw.ssm_in, lw.ssm_in_stride,
                             b.norm_out, b.ssm_xz, H, 2 * d_inner, stream);
    // Conv1d step — kernel: eval_ssm_conv_step(x_in, state, w, y_out, d_inner, d_conv, apply_silu)
    {
        // For Mamba2 hybrids, conv state width is d_inner + 2*n_group*d_state
        int conv_w = d_inner;
        if (c.ssm_n_group > 0) conv_w = d_inner + 2 * c.ssm_n_group * d_state;
        float * cs = b.ssm_conv_states + (long long)il * conv_w * (d_conv - 1);
        float * x = b.ssm_xz;
        const void * cw = lw.ssm_conv1d;
        int n = d_inner, dc = d_conv;
        int apply_silu_val = 0; // apply silu separately after bias
        void * args[] = { (void *)&x, (void *)&cs, (void *)&cw, (void *)&x, (void *)&n, (void *)&dc, (void *)&apply_silu_val };
        hipModuleLaunchKernel(k.eval_ssm_conv_step, (n+255)/256, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
    }
    // Add conv bias — baseline: ggml_add(conv_out, conv1d_bias)
    {
        const void * cb2 = lw.ssm_conv1d_b;
        int n = d_inner;
        void * ba[] = { (void *)&b.ssm_xz, (void *)&cb2, (void *)&b.ssm_xz, (void *)&n };
        hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, stream, ba, nullptr);
    }
    // Apply SiLU
    {
        int n = d_inner;
        void * sa[] = { (void *)&b.ssm_xz, (void *)&b.ssm_xz, (void *)&n };
        hipModuleLaunchKernel(k.eval_silu, (n+255)/256, 1, 1, 256, 1, 1, 0, stream, sa, nullptr);
    }
    // ssm_x @ x → [dt_rank + 2*d_state]
    quant_and_launch_matvec(lw.ssm_x_type, lw.ssm_x, lw.ssm_x_stride,
                             b.ssm_xz, b.ssm_x_db, d_inner, dt_rank + 2 * d_state, stream);
    // dt = ssm_dt @ dt + bias
    quant_and_launch_matvec(lw.ssm_dt_type, lw.ssm_dt, lw.ssm_dt_stride,
                             b.ssm_x_db, b.ssm_dt, dt_rank, d_inner, stream);
    {
        const void * bias = lw.ssm_dt_b; int n = d_inner;
        void * args[] = { (void *)&b.ssm_dt, (void *)&bias, (void *)&b.ssm_dt, (void *)&n };
        hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
    }
    // SSM scan step
    {
        float * ss = b.ssm_scan_states + (long long)il * d_inner * d_state;
        float * x = b.ssm_xz, * dt2 = b.ssm_dt;
        const void * A = lw.ssm_a;
        float * B2 = b.ssm_x_db + dt_rank, * C2 = b.ssm_x_db + dt_rank + d_state;
        const void * D2 = lw.ssm_d;
        float * y = b.proj_scratch;
        int di = d_inner, ds = d_state;
        // Kernel: eval_ssm_scan_step(x, dt, A, B, C, D, h, y, d_inner, d_state)
        void * args[] = { (void *)&x, (void *)&dt2, (void *)&A,
                          (void *)&B2, (void *)&C2, (void *)&D2, (void *)&ss, (void *)&y, (void *)&di, (void *)&ds };
        hipModuleLaunchKernel(k.eval_ssm_scan_step, (di+255)/256, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
    }
    // silu(z) * y
    {
        float * z = b.ssm_xz + d_inner, * y = b.proj_scratch; int n = d_inner;
        void * args[] = { (void *)&z, (void *)&y, (void *)&y, (void *)&n };
        hipModuleLaunchKernel(k.eval_silu_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
    }
    // ssm_out @ y + residual → hidden
    {
        int n = d_inner, q8blocks = (n+511)/512;
        void * q8args[] = { (void *)&b.proj_scratch, (void *)&b.q8_act, (void *)&n };
        hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, stream, q8args, nullptr);
    }
    launch_matvec_res_typed(lw.ssm_out_type, lw.ssm_out, lw.ssm_out_stride,
                             b.residual, b.hidden, d_inner, H, stream);
}
