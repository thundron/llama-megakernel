// forward/llama-family.cpp — Dense transformer forward (~95 archs)
#include "../gfx1100-internal.h"

// ---- Per-phase GPU profiling (GFX1100_PROFILE env var) ----
// Phase indices for timing accumulation
enum gfx1100_profile_phase {
    PROF_EMBED    = 0,
    PROF_NORM     = 1,  // pre-attn norm + quantize (accumulated per layer)
    PROF_QKV_PROJ = 2,  // Q/K/V projections
    PROF_ROPE_KV  = 3,  // RoPE + KV cache write
    PROF_ATTN     = 4,  // flash attention decode
    PROF_O_PROJ   = 5,  // output projection + post-attn-norm + residual
    PROF_FFN_NORM = 6,  // pre-FFN norm + quantize
    PROF_FFN_PROJ = 7,  // gate + up + activation + quantize + down
    PROF_FFN_RES  = 8,  // post-FFN-norm + residual
    PROF_LM_HEAD  = 9,  // final norm + LM head projection
    PROF_N_PHASES = 10,
};

static const char * gfx1100_profile_phase_names[] = {
    "embed", "norm", "qkv_proj", "rope_kv", "attn",
    "o_proj", "ffn_norm", "ffn_proj", "ffn_res", "lm_head",
};

int forward_decode_llama_family(int token_id, int position, float * logits_out) {
    auto & c   = g_config;
    auto & b   = g_bufs;
    auto & k   = g_compiled;
    auto   s   = b.stream;
    int    H   = c.hidden_size;
    int    FF  = c.intermediate_size;
    int    V   = c.vocab_size;

    // ---- Profiling setup ----
    static bool profile_enabled = (getenv("GFX1100_PROFILE") != nullptr);
    // Per-layer: 9 event-record points (boundaries between phases inside the loop)
    // Plus 4 for embed (before/after) and lm_head (before/after)
    // Layout: ev_embed[0]=before_embed, ev_embed[1]=after_embed
    //         ev_layer[il*9 + 0..8] = boundaries within layer il
    //         ev_lm[0]=before_lm_head, ev_lm[1]=after_lm_head
    hipEvent_t ev_embed[2] = {};
    hipEvent_t ev_lm[2] = {};
    std::vector<hipEvent_t> ev_layer;  // [n_layers * 9]
    float prof_ms[PROF_N_PHASES] = {};

    if (profile_enabled) {
        hipEventCreate(&ev_embed[0]);
        hipEventCreate(&ev_embed[1]);
        hipEventCreate(&ev_lm[0]);
        hipEventCreate(&ev_lm[1]);
        ev_layer.resize(c.n_layers * 9);
        for (int i = 0; i < c.n_layers * 9; i++) {
            hipEventCreate(&ev_layer[i]);
        }
    }

    // Profiling helper macros — record event on stream only when profiling
    #define PROF_RECORD_EMBED(idx) do { if (profile_enabled) hipEventRecord(ev_embed[idx], s); } while(0)
    #define PROF_RECORD_LM(idx)    do { if (profile_enabled) hipEventRecord(ev_lm[idx], s); } while(0)
    #define PROF_RECORD_LAYER(il, boundary) do { if (profile_enabled) hipEventRecord(ev_layer[(il)*9 + (boundary)], s); } while(0)

    // Baseline norm.cu thread dispatch: 256 for ncols<1024, 1024 for ncols>=1024
    const int norm_threads = (H < 1024) ? 256 : 1024;

    // ---- Per-phase debug dump (GFX1100_DUMP env var) ----
    // Dumps intermediate GPU buffers to binary files for comparison with baseline.
    // Only active for layer 0, token 0 (first decode after init).
    // Files: dump_mk_{phase}_{position}.bin — each contains n floats.
    static bool dump_enabled = (getenv("GFX1100_DUMP") != nullptr);
    static int dump_target_pos = getenv("GFX1100_DUMP_POS") ? atoi(getenv("GFX1100_DUMP_POS")) : 0;
    static int dump_token_count = 0;
    auto dump_gpu = [&](const char * tag, const float * d_ptr, int n) {
        if (!dump_enabled || dump_token_count != dump_target_pos) return;
        hipStreamSynchronize(s);
        std::vector<float> h(n);
        hipMemcpy(h.data(), d_ptr, n * sizeof(float), hipMemcpyDeviceToHost);
        char path[256];
        snprintf(path, sizeof(path), "dump_mk_%s_%d.bin", tag, position);
        FILE * f = fopen(path, "wb");
        if (f) { fwrite(h.data(), sizeof(float), n, f); fclose(f); }
        // Print first 8 values + stats
        double sum = 0, sumsq = 0;
        for (int i = 0; i < n; i++) { sum += h[i]; sumsq += h[i]*h[i]; }
        fprintf(stderr, "  DUMP %s [%d]: first8=[%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f] mean=%.6f rms=%.6f\n",
                tag, n, h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7],
                sum/n, sqrt(sumsq/n));
    };
    auto dump_gpu_half = [&](const char * tag, const __half * d_ptr, int n) {
        if (!dump_enabled || dump_token_count != dump_target_pos) return;
        hipStreamSynchronize(s);
        std::vector<__half> hh(n);
        std::vector<float> hf(n);
        hipMemcpy(hh.data(), d_ptr, n * sizeof(__half), hipMemcpyDeviceToHost);
        for (int i = 0; i < n; i++) hf[i] = __half2float(hh[i]);
        char path[256];
        snprintf(path, sizeof(path), "dump_mk_%s_%d.bin", tag, position);
        FILE * f = fopen(path, "wb");
        if (f) { fwrite(hf.data(), sizeof(float), n, f); fclose(f); }
        fprintf(stderr, "  DUMP %s [%d]: first4=[%.6f %.6f %.6f %.6f]\n",
                tag, n, hf[0], hf[1], hf[2], hf[3]);
    };

    // ---- hipGraph capture/replay for true graph reuse ----
    // GFX1100_GRAPH=1: capture the forward pass once, then replay with GPU-resident params.
    // d_decode_params[3] = {token_id, position, kv_len} written via H2D OUTSIDE the graph.
    // Kernels read from d_decode_params pointer (baked into graph) instead of scalar args.
    static bool use_graph = (getenv("GFX1100_GRAPH") != nullptr);
    static hipGraphExec_t g_graph_exec = nullptr;
    static int g_graph_warmup = 0;
    static const int GRAPH_WARMUP_TOKENS = 2;

    // Disable graph for models with SWA (sliding window attention) — the K/V cache pointer
    // offsets change each token and would be baked into the graph incorrectly.
    // Also disable for models with pos_embd (GPT-2/StarCoder) — position embedding offset
    // would be baked in. Also disable when profiling (event records are not graph-compatible).
    if (use_graph && (c.n_swa > 0 || c.pos_embd || profile_enabled)) {
        static bool warned = false;
        if (!warned) {
            const char * reason = c.n_swa > 0 ? "SWA model" :
                                  c.pos_embd ? "pos_embd model" : "profiling enabled";
            fprintf(stderr, "gfx1100-megakernel: hipGraph disabled (%s)\n", reason);
            warned = true;
        }
        use_graph = false;
    }

    // For graph path: pointer to GPU-resident decode params (non-graph: nullptr)
    // d_params_ptr: nullptr during warmup + non-graph, b.d_decode_params during capture/replay
    // Must be nullptr until d_decode_params is actually written (after warmup)
    const int * d_params_ptr = (use_graph && g_graph_warmup >= GRAPH_WARMUP_TOKENS) ? b.d_decode_params : nullptr;

    // Helper: norm dispatch — calls eval_rmsnorm_q8 or eval_layernorm based on norm_type
    // Baseline norm.cu: rms_norm_f32_cuda / norm_f32_cuda (LayerNorm)
    auto launch_norm = [&](const float * input, const void * norm_w, const void * norm_b,
                           float * norm_out, float * residual) {
        if (c.norm_type == 2) { // NORM_LAYER (arch_ids.h)
            // eval_layernorm(input, weight, bias, norm_out, residual, n, eps)
            int n = H; float eps = c.norm_eps;
            void * args[] = { (void *)&input, (void *)&norm_w, (void *)&norm_b,
                              (void *)&norm_out, (void *)&residual, (void *)&n, (void *)&eps };
            hipModuleLaunchKernel(k.eval_layernorm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        } else {
            // eval_rmsnorm_q8(input, weight, norm_out, residual, n)
            int n = H;
            void * args[] = { (void *)&input, (void *)&norm_w, (void *)&norm_out,
                              (void *)&residual, (void *)&n };
            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        }
    };

    // Helper: fused norm + Q8 quantize in one kernel launch (RMSNorm only)
    // Replaces the 2-launch pattern: launch_norm + eval_quantize_q8
    auto launch_norm_q8 = [&](const float * input, const void * norm_w, const void * norm_b,
                              float * norm_out, float * residual, void * q8_out) {
        if (c.norm_type == 2) {
            // LayerNorm: can't fuse yet, fall back to 2-launch
            int n = H; float eps = c.norm_eps;
            void * args[] = { (void *)&input, (void *)&norm_w, (void *)&norm_b,
                              (void *)&norm_out, (void *)&residual, (void *)&n, (void *)&eps };
            hipModuleLaunchKernel(k.eval_layernorm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
            int blocks = (n + 511) / 512;
            void * q8args[] = { (void *)&norm_out, (void *)&q8_out, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
        } else {
            // RMSNorm: fused single-pass kernel
            int n = H;
            void * args[] = { (void *)&input, (void *)&norm_w, (void *)&norm_out,
                              (void *)&residual, (void *)&q8_out, (void *)&n };
            hipModuleLaunchKernel(k.eval_rmsnorm_q8_quantize, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        }
    };

    // Helper: element-wise add — baseline: ggml_add(a, b)
    auto launch_add = [&](float * dst, const void * bias, int n) {
        void * ba[] = { (void *)&dst, (void *)&bias, (void *)&dst, (void *)&n };
        hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, s, ba, nullptr);
    };
    // Helper: element-wise multiply — baseline: ggml_mul(cur, scale)
    auto launch_mul = [&](float * dst, const void * scale, int n) {
        void * ma[] = { (void *)&dst, (void *)&scale, (void *)&dst, (void *)&n };
        hipModuleLaunchKernel(k.eval_elementwise_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, ma, nullptr);
    };

    // Helper: is type non-quantized? (F16/BF16/F32 use f32 input, block=(256,1,1))
    auto is_float_type = [](int type) -> bool {
        return type == 0 || type == 1 || type == 30; // F32=0, F16=1, BF16=30
    };

    // Helper: pick fused gate+up+silu kernel by ggml_type
    // Returns nullptr if no fused kernel for this type
    auto pick_fused_gate_up_silu = [&](int type) -> hipFunction_t {
        switch (type) {
            case  0: return k.eval_fused_gate_up_silu_f32;
            case  1: return k.eval_fused_gate_up_silu_f16;
            case 30: return k.eval_fused_gate_up_silu_bf16;
            case  2: return k.eval_fused_gate_up_silu_q4_0;
            case  3: return k.eval_fused_gate_up_silu_q4_1;
            case  6: return k.eval_fused_gate_up_silu_q5_0;
            case  7: return k.eval_fused_gate_up_silu_q5_1;
            case  8: return k.eval_fused_gate_up_silu_q8_0;
            case 10: return k.eval_fused_gate_up_silu_q2k;
            case 11: return k.eval_fused_gate_up_silu_q3k;
            case 12: return k.eval_fused_gate_up_silu_q4k;
            case 13: return k.eval_fused_gate_up_silu_q5k;
            case 14: return k.eval_fused_gate_up_silu_q6k;
            case 16: return k.eval_fused_gate_up_silu_iq2_xxs;
            case 17: return k.eval_fused_gate_up_silu_iq2_xs;
            case 18: return k.eval_fused_gate_up_silu_iq3_xxs;
            case 19: return k.eval_fused_gate_up_silu_iq1_s;
            case 20: return k.eval_fused_gate_up_silu_iq4_nl;
            case 21: return k.eval_fused_gate_up_silu_iq3_s;
            case 22: return k.eval_fused_gate_up_silu_iq2_s;
            case 23: return k.eval_fused_gate_up_silu_iq4_xs;
            case 29: return k.eval_fused_gate_up_silu_iq1_m;
            case 39: return k.eval_fused_gate_up_silu_mxfp4;
            case 40: return k.eval_fused_gate_up_silu_nvfp4;
            default: return (hipFunction_t)nullptr;
        }
    };

    auto pick_fused_gate_up_gelu = [&](int type) -> hipFunction_t {
        switch (type) {
            case  2: return k.eval_fused_gate_up_gelu_q4_0;
            case  8: return k.eval_fused_gate_up_gelu_q8_0;
            case 10: return k.eval_fused_gate_up_gelu_q2k;
            case 11: return k.eval_fused_gate_up_gelu_q3k;
            case 12: return k.eval_fused_gate_up_gelu_q4k;
            case 13: return k.eval_fused_gate_up_gelu_q5k;
            case 14: return k.eval_fused_gate_up_gelu_q6k;
            default: return (hipFunction_t)nullptr;
        }
    };

    // Helper: pick fused QKV matvec kernel by ggml_type (quantized only)
    auto pick_fused_qkv_matvec = [&](int type) -> hipFunction_t {
        switch (type) {
            case  2: return k.eval_fused_qkv_matvec_q4_0;
            case  3: return k.eval_fused_qkv_matvec_q4_1;
            case  6: return k.eval_fused_qkv_matvec_q5_0;
            case  7: return k.eval_fused_qkv_matvec_q5_1;
            case  8: return k.eval_fused_qkv_matvec_q8_0;
            case 10: return k.eval_fused_qkv_matvec_q2k;
            case 11: return k.eval_fused_qkv_matvec_q3k;
            case 12: return k.eval_fused_qkv_matvec_q4k;
            case 13: return k.eval_fused_qkv_matvec_q5k;
            case 14: return k.eval_fused_qkv_matvec_q6k;
            default: return (hipFunction_t)nullptr;
        }
    };

    // Helper: pick fused quantize+matvec kernel by ggml_type (quantized only)
    auto pick_quantize_matvec = [&](int type) -> hipFunction_t {
        switch (type) {
            case  2: return k.eval_quantize_matvec_q4_0;
            case  3: return k.eval_quantize_matvec_q4_1;
            case  6: return k.eval_quantize_matvec_q5_0;
            case  7: return k.eval_quantize_matvec_q5_1;
            case  8: return k.eval_quantize_matvec_q8_0;
            case 10: return k.eval_quantize_matvec_q2k;
            case 11: return k.eval_quantize_matvec_q3k;
            case 12: return k.eval_quantize_matvec_q4k;
            case 13: return k.eval_quantize_matvec_q5k;
            case 14: return k.eval_quantize_matvec_q6k;
            default: return (hipFunction_t)nullptr;
        }
    };

    // Helper: pick fused quantize+matvec+residual kernel by ggml_type
    auto pick_quantize_matvec_res = [&](int type) -> hipFunction_t {
        switch (type) {
            case  2: return k.eval_quantize_matvec_residual_q4_0;
            case  3: return k.eval_quantize_matvec_residual_q4_1;
            case  6: return k.eval_quantize_matvec_residual_q5_0;
            case  7: return k.eval_quantize_matvec_residual_q5_1;
            case  8: return k.eval_quantize_matvec_residual_q8_0;
            case 10: return k.eval_quantize_matvec_residual_q2k;
            case 11: return k.eval_quantize_matvec_residual_q3k;
            case 12: return k.eval_quantize_matvec_residual_q4k;
            case 13: return k.eval_quantize_matvec_residual_q5k;
            case 14: return k.eval_quantize_matvec_residual_q6k;
            default: return (hipFunction_t)nullptr;
        }
    };

    // Helper: pick matvec kernel + block dims by ggml_type
    // Quantized: block=(32,4,1), input=q8_act
    // Float:     block=(256,1,1), input=norm_out (f32)
    auto pick_matvec = [&](int type) -> hipFunction_t {
        switch (type) {
            case  0: return k.eval_matvec_f32;        // GGML_TYPE_F32
            case  1: return k.eval_matvec_f16;        // GGML_TYPE_F16
            case 30: return k.eval_matvec_bf16;       // GGML_TYPE_BF16
            case  2: return k.eval_matvec_q4_0;       // GGML_TYPE_Q4_0
            case  3: return k.eval_matvec_q4_1;       // GGML_TYPE_Q4_1
            case  6: return k.eval_matvec_q5_0;       // GGML_TYPE_Q5_0
            case  7: return k.eval_matvec_q5_1;       // GGML_TYPE_Q5_1
            case  8: return k.eval_matvec_q8_0;       // GGML_TYPE_Q8_0
            case 10: return k.eval_matvec_q2k;        // GGML_TYPE_Q2_K
            case 11: return k.eval_matvec_q3k;        // GGML_TYPE_Q3_K
            case 12: return k.eval_matvec_q4k;        // GGML_TYPE_Q4_K
            case 13: return k.eval_matvec_q5k;        // GGML_TYPE_Q5_K
            case 14: return k.eval_matvec_q6k;        // GGML_TYPE_Q6_K
            case 16: return k.eval_matvec_iq2_xxs;    // GGML_TYPE_IQ2_XXS
            case 17: return k.eval_matvec_iq2_xs;     // GGML_TYPE_IQ2_XS
            case 18: return k.eval_matvec_iq3_xxs;    // GGML_TYPE_IQ3_XXS
            case 19: return k.eval_matvec_iq1_s;      // GGML_TYPE_IQ1_S
            case 20: return k.eval_matvec_iq4_nl;     // GGML_TYPE_IQ4_NL
            case 21: return k.eval_matvec_iq3_s;      // GGML_TYPE_IQ3_S
            case 22: return k.eval_matvec_iq2_s;      // GGML_TYPE_IQ2_S
            case 23: return k.eval_matvec_iq4_xs;     // GGML_TYPE_IQ4_XS
            case 29: return k.eval_matvec_iq1_m;      // GGML_TYPE_IQ1_M
            case 39: return k.eval_matvec_mxfp4;      // GGML_TYPE_MXFP4
            case 40: return k.eval_matvec_nvfp4;      // GGML_TYPE_NVFP4
            default:
                fprintf(stderr, "gfx1100: unsupported matvec type %d\n", type);
                return k.eval_matvec_q4k;
        }
    };
    auto pick_matvec_res = [&](int type) -> hipFunction_t {
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
            default:
                fprintf(stderr, "gfx1100: unsupported matvec_res type %d\n", type);
                return k.eval_matvec_q4k_residual;
        }
    };

    // Helper: pick 8-warp matvec kernel for RDNA3-optimized types, nullptr for others
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
    auto pick_matvec_8w_res = [&](int type) -> hipFunction_t {
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
    };

    // Helper: launch matvec with correct block dims and input buffer
    // Quantized types with 8-warp: block=(32,8,1)=256 threads
    // Quantized types fallback:    block=(32,4,1)=128 threads
    // Float types:                 block=(256,1,1)=256 threads
    // Match baseline RDNA3 nwarps: use 8-warp variants when available (Q4_K, Q6_K, Q4_0, etc.)
    // Baseline mmvq.cu MMVQ_PARAMETERS_RDNA3_0 table uses nwarps=8 for these types.
    // Using the same nwarps produces identical float accumulation order → 0 argmax mismatches.
    auto launch_matvec = [&](int type, const void * w, long long st,
                             float * output, int in_dim, int out_dim) {
        bool is_f = is_float_type(type);
        const void * input = is_f ? (const void *)b.norm_out : (const void *)b.q8_act;
        void * args[] = { (void *)&w, (void *)&st, (void *)&input,
                          (void *)&output, (void *)&in_dim, (void *)&out_dim };
        if (is_f) {
            hipModuleLaunchKernel(pick_matvec(type), out_dim, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        } else {
            hipFunction_t fn8w = pick_matvec_8w(type);
            if (fn8w) {
                hipModuleLaunchKernel(fn8w, out_dim, 1, 1, 32, 8, 1, 0, s, args, nullptr);
            } else {
                hipModuleLaunchKernel(pick_matvec(type), out_dim, 1, 1, 32, 4, 1, 0, s, args, nullptr);
            }
        }
    };
    auto launch_matvec_res = [&](int type, const void * w, long long st,
                                  float * residual, float * output, int in_dim, int out_dim) {
        bool is_f = is_float_type(type);
        const void * input = is_f ? (const void *)b.norm_out : (const void *)b.q8_act;
        void * args[] = { (void *)&w, (void *)&st, (void *)&input,
                          (void *)&residual, (void *)&output, (void *)&in_dim, (void *)&out_dim };
        if (is_f) {
            hipModuleLaunchKernel(pick_matvec_res(type), out_dim, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        } else {
            hipFunction_t fn8w = pick_matvec_8w_res(type);
            if (fn8w) {
                hipModuleLaunchKernel(fn8w, out_dim, 1, 1, 32, 8, 1, 0, s, args, nullptr);
            } else {
                hipModuleLaunchKernel(pick_matvec_res(type), out_dim, 1, 1, 32, 4, 1, 0, s, args, nullptr);
            }
        }
    };

    // ---- Embedding ----
    PROF_RECORD_EMBED(0);
    // Embedding table copied to GPU during init (no CPU barrier).
    // Kernel reads directly from GPU-resident table using token_id.
    {
        const void * embed_ptr = c.embed_weight;  // GPU pointer (copied during init if needed)
        long long stride = c.embed_stride;
        int tok = token_id;
        int nb_embed = H / 256;
        void * args[] = { (void *)&embed_ptr, (void *)&stride,
                          (void *)&b.hidden, (void *)&tok };
        // Type-based embedding dispatch — all 20 types
        hipFunction_t embed_fn; int embed_threads;
        switch (c.embed_type) {
            // SMALL types — baseline k_get_rows pattern, 256 threads, 2D grid.
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
            case  0: embed_fn = k.eval_embed_f32;     embed_threads = 256; break; // F32
            case  1: embed_fn = k.eval_embed_f16;     embed_threads = 256; break; // F16
            case 30: embed_fn = k.eval_embed_bf16;    embed_threads = 256; break; // BF16
            default:
                fprintf(stderr, "gfx1100: FATAL — unsupported embed type %d\n", c.embed_type);
                return -1; // ABORT — do NOT silently fallback
        }
        // Float types: flat thread grid (k_get_rows_float pattern, ne00/block step).
        // SMALL Q-quants (Q4_0..Q8_0): baseline k_get_rows pattern grid=(1, ceildiv(H,2*256), 1).
        // K-quants and IQ-quants: per-superblock (256-element) pattern, kept from existing code.
        // NVFP4: per-superblock (512-element).
        const bool is_small_q = (c.embed_type == 2 || c.embed_type == 3 ||
                                  c.embed_type == 6 || c.embed_type == 7 ||
                                  c.embed_type == 8);
        if (c.embed_type == 0 || c.embed_type == 1 || c.embed_type == 30) {
            int grid = (H + embed_threads - 1) / embed_threads;
            hipModuleLaunchKernel(embed_fn, grid, 1, 1, embed_threads, 1, 1, 0, s, args, nullptr);
        } else if (c.embed_type == 40) {
            int nb_nvfp4 = H / 512; // NVFP4: 512 elements per superblock
            hipModuleLaunchKernel(embed_fn, nb_nvfp4, 1, 1, embed_threads, 1, 1, 0, s, args, nullptr);
        } else if (is_small_q) {
            // Baseline k_get_rows: block=256, grid=(1, ceildiv(H, 2*block), 1)
            const int grid_y = (H + 2 * embed_threads - 1) / (2 * embed_threads);
            hipModuleLaunchKernel(embed_fn, 1, grid_y, 1, embed_threads, 1, 1, 0, s, args, nullptr);
        } else {
            hipModuleLaunchKernel(embed_fn, nb_embed, 1, 1, embed_threads, 1, 1, 0, s, args, nullptr);
        }
    }

    // Gemma embedding scale: multiply embeddings by sqrt(n_embd)
    // Baseline gemma2-iswa.cpp line 11: inpL = ggml_scale(inpL, sqrtf(n_embd))
    if (c.has_embed_scale) {
        float scale = sqrtf((float)H);
        int n = H;
        void * args[] = { (void *)&b.hidden, (void *)&b.hidden, (void *)&scale, (void *)&n };
        hipModuleLaunchKernel(k.eval_scale_scalar, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    }

    // GPT-2/StarCoder/GPTJ: add learned absolute position embeddings
    // Baseline gpt2.cpp line 20: inpL = ggml_add(inpL, ggml_get_rows(pos_embd, inp_pos))
    if (c.pos_embd) {
        // For single-token decode: hidden += pos_embd[position]
        const float * pos_row = (const float *)c.pos_embd + (long long)position * H;
        void * args[] = { (void *)&b.hidden, (void *)&pos_row, (void *)&b.hidden, (void *)&H };
        hipModuleLaunchKernel(k.eval_add_residual, (H+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    }

    // Token norm (BLOOM, RWKV — LayerNorm on embeddings before first layer)
    // Baseline bloom.cpp line 18: build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM)
    if (c.tok_norm_weight) {
        int n = H;
        float eps = c.norm_eps;
        const void * tw = c.tok_norm_weight;
        const void * tb = c.tok_norm_bias;
        void * args[] = { (void *)&b.hidden, (void *)&tw, (void *)&tb,
                          (void *)&b.hidden, (void *)&b.hidden, (void *)&n, (void *)&eps };
        hipModuleLaunchKernel(k.eval_layernorm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
    }

    PROF_RECORD_EMBED(1);
    dump_gpu("embed", b.hidden, H);

    // ---- hipGraph: replay captured graph if available ----
    if (use_graph && g_graph_exec && g_graph_warmup > GRAPH_WARMUP_TOKENS) {
        // Token 4+: write new params to GPU, then replay captured graph
        int h_params[3] = { token_id, position, position + 1 };
        hipMemcpyAsync(b.d_decode_params, h_params, 3 * sizeof(int),
                       hipMemcpyHostToDevice, s);
        hipGraphLaunch(g_graph_exec, s);

        // Copy logits to host (outside graph)
        if (logits_out) {
            hipMemcpyAsync(logits_out, b.logits, V * sizeof(float),
                           hipMemcpyDeviceToHost, s);
        }
        hipStreamSynchronize(s);
        return 0;
    }

    // ---- hipGraph: warmup or capture ----
    bool capturing_graph = false;
    if (use_graph) {
        g_graph_warmup++;
        if (g_graph_warmup > GRAPH_WARMUP_TOKENS) {
            // First capture: write params to GPU, then begin capture
            int h_params[3] = { token_id, position, position + 1 };
            hipMemcpyAsync(b.d_decode_params, h_params, 3 * sizeof(int),
                           hipMemcpyHostToDevice, s);
            hipStreamSynchronize(s);  // ensure H2D completes before capture
            hipStreamBeginCapture(s, hipStreamCaptureModeGlobal);
            capturing_graph = true;

            // Launch eval_write_decode_params inside capture
            {
                int * dp = b.d_decode_params;
                int * bt = b.batch_token_ids;
                void * args[] = { (void *)&dp, (void *)&bt };
                hipModuleLaunchKernel(k.eval_write_decode_params, 1, 1, 1, 32, 1, 1, 0, s, args, nullptr);
            }
        }
    }

    // ---- Layer loop (host-orchestrated) ----
    int attn_idx = 0;
    int dn_idx   = 0;

    int debug_max_layers = c.n_layers;
    bool skip_pre_attn_norm = false;  // set when post-FFN norm fuses with next layer's pre-attn norm
    for (int il = 0; il < debug_max_layers; il++) {
        const gfx1100_layer_weights & lw = c.layers[il];

        PROF_RECORD_LAYER(il, 0);  // boundary 0: start of layer (before norm)

        // Phase 1: Norm → norm_out + residual
        // Dispatches to RMSNorm or LayerNorm based on model's norm_type
        // Chameleon swin_norm: skip pre-norm, apply after attention/FFN output instead
        if (skip_pre_attn_norm) {
            // Already done by previous layer's fused post-FFN norm + pre-attn norm
            skip_pre_attn_norm = false;
        } else if (c.has_swin_norm) {
            // Post-norm mode: no pre-attention norm, input goes directly to QKV
            hipMemcpyAsync(b.norm_out, b.hidden, H * sizeof(float), hipMemcpyDeviceToDevice, s);
            hipMemcpyAsync(b.residual, b.hidden, H * sizeof(float), hipMemcpyDeviceToDevice, s);
            int n = H, blocks = (n + 511) / 512;
            void * args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, args, nullptr);
        } else {
            // Fused: norm + residual copy + Q8 quantize in ONE kernel launch
            launch_norm_q8(b.hidden, lw.ptrs[0], lw.attn_norm_bias, b.norm_out, b.residual, b.q8_act);
        }

        PROF_RECORD_LAYER(il, 1);  // boundary 1: after norm+quantize, before QKV proj
        if (il == 0) {
            char tag1[64], tag2[64], tag3[64];
            snprintf(tag1, sizeof(tag1), "L%d_norm_out", il);
            snprintf(tag2, sizeof(tag2), "L%d_residual", il);
            snprintf(tag3, sizeof(tag3), "L%d_hidden_in", il);
            dump_gpu(tag1, b.norm_out, H);
            dump_gpu(tag2, b.residual, H);
            dump_gpu(tag3, b.hidden, H);
        }

        if (c.layer_types[il] == 2) {
            // ---- SSM/Mamba layer (hybrid: Jamba, Falcon-H1, etc.) ----
            // Baseline jamba.cpp: SSM replaces attention only. FFN still runs after.
            ssm_layer_step(il, s);
        } else if (c.layer_types[il] == 0) {
            // ---- Attention layer ----
            // Cohere2: save norm_out for FFN (FFN reads from pre-attention norm, not post-attention)
            // Baseline cohere2-iswa.cpp line 31: ffn_inp = cur (cur = attn_norm output)
            if (c.use_shared_norm_ffn) {
                hipMemcpyAsync(b.mlp_inter, b.norm_out, H * sizeof(float), hipMemcpyDeviceToDevice, s);
            }
            // Per-layer head count override for OpenELM (different n_head per layer)
            int layer_n_q  = (c.per_layer_n_q_heads[il] > 0)  ? c.per_layer_n_q_heads[il]  : c.fa_n_q_heads;
            int layer_n_kv = (c.per_layer_n_kv_heads[il] > 0) ? c.per_layer_n_kv_heads[il] : c.fa_n_kv_heads;
            int qproj_size = layer_n_q * c.fa_head_dim;
            if (c.fa_has_gated_attn) qproj_size *= 2;
            int kv_size = layer_n_kv * c.fa_head_dim;

            // QKV projections — try fused 3-in-1 kernel when conditions allow
            // Conditions: same quant type for Q/K/V, no per-projection biases/scales,
            // quantized (not float), no gated attention
            {
                bool same_type = (lw.types[1] == lw.types[2] && lw.types[2] == lw.types[3]);
                bool no_bias_scale = (!lw.bias_q && !lw.bias_k && !lw.bias_v &&
                                      !lw.scale_q && !lw.scale_k && !lw.scale_v);
                bool is_quant = !is_float_type(lw.types[1]);
                hipFunction_t fused_qkv = (same_type && no_bias_scale && is_quant && !c.fa_has_gated_attn)
                                          ? pick_fused_qkv_matvec(lw.types[1])
                                          : (hipFunction_t)nullptr;
                if (fused_qkv) {
                    // Fused QKV: 1 launch instead of 3
                    int smem_size = (H / 32) * 36; // QK8_1=32, sizeof(block_q8_1)=36
                    int total_rows = qproj_size + kv_size + kv_size;
                    void * args[] = {
                        (void *)&lw.ptrs[1], (void *)&lw.strides[1],
                        (void *)&lw.ptrs[2], (void *)&lw.strides[2],
                        (void *)&lw.ptrs[3], (void *)&lw.strides[3],
                        (void *)&b.q8_act,
                        (void *)&b.proj_scratch, (void *)&b.kv_scratch,
                        (void *)&b.kv_scratch, // v_output offset added below
                        (void *)&H, (void *)&qproj_size, (void *)&kv_size
                    };
                    // V output goes to kv_scratch + kv_size
                    float * v_out = b.kv_scratch + kv_size;
                    args[9] = (void *)&v_out;
                    hipModuleLaunchKernel(fused_qkv, total_rows, 1, 1, 32, 4, 1, smem_size, s, args, nullptr);
                } else {
                    // Fallback: 3 separate matvec launches
                    // Q projection — baseline line 46
                    launch_matvec(lw.types[1], lw.ptrs[1], lw.strides[1], b.proj_scratch, H, qproj_size);
                    if (lw.scale_q) launch_mul(b.proj_scratch, lw.scale_q, qproj_size);
                    if (lw.bias_q)  launch_add(b.proj_scratch, lw.bias_q, qproj_size);

                    // K projection — baseline line 52
                    launch_matvec(lw.types[2], lw.ptrs[2], lw.strides[2], b.kv_scratch, H, kv_size);
                    if (lw.scale_k) launch_mul(b.kv_scratch, lw.scale_k, kv_size);
                    if (lw.bias_k)  launch_add(b.kv_scratch, lw.bias_k, kv_size);

                    // V projection — baseline line 58
                    launch_matvec(lw.types[3], lw.ptrs[3], lw.strides[3], b.kv_scratch + kv_size, H, kv_size);
                    if (lw.scale_v) launch_mul(b.kv_scratch + kv_size, lw.scale_v, kv_size);
                    if (lw.bias_v)  launch_add(b.kv_scratch + kv_size, lw.bias_v, kv_size);
                }
            }

            PROF_RECORD_LAYER(il, 2);  // boundary 2: after QKV proj, before rope_kv
            if (il == 0) {
                int qsz = c.fa_n_q_heads * c.fa_head_dim;
                int kvsz = c.fa_n_kv_heads * c.fa_head_dim;
                dump_gpu("L0_q_proj", b.proj_scratch, qsz);
                dump_gpu("L0_k_proj", b.kv_scratch, kvsz);
                dump_gpu("L0_v_proj", b.kv_scratch + kvsz, kvsz);
            }

            // QK Norm + RoPE + KV cache write
            {
                const void * q_nw = lw.ptrs[4];
                const void * k_nw = lw.ptrs[5];
                void * kc = c.k_cache_ptrs[attn_idx];
                void * vc = c.v_cache_ptrs[attn_idx];
                int total_heads = layer_n_q + layer_n_kv;
                int blocks = (total_heads + 15) / 16;
                // Per-layer freq factors for Phi3 SU-RoPE (baseline phi3.cpp line 34)
                // If per-layer array is set and has a non-NULL entry for this layer, use it
                const void * ff = (c.rope_freq_factors_per_layer[il])
                                  ? c.rope_freq_factors_per_layer[il]
                                  : c.rope_freq_factors;  // fallback to global
                // Per-layer theta override for iSWA models
                // SWA layers: use SWA theta if set (Gemma3/4)
                // Non-SWA layers: use global theta (default), OR skip RoPE entirely (Cohere2)
                float theta_ovr = 0.0f;
                if (c.layer_use_swa[il] && c.fa_rope_theta_swa > 0) {
                    theta_ovr = c.fa_rope_theta_swa;  // SWA layer with different theta
                } else if (!c.layer_use_swa[il] && c.skip_rope_on_global_layers) {
                    theta_ovr = -1.0f;  // Cohere2: global layers skip RoPE entirely
                }
                void * args[] = { (void *)&b.proj_scratch, (void *)&b.kv_scratch,
                                  (void *)&q_nw, (void *)&k_nw,
                                  (void *)&kc, (void *)&vc, (void *)&ff,
                                  (void *)&position, (void *)&c.max_seq_len, (void *)&theta_ovr,
                                  (void *)&d_params_ptr };
                hipModuleLaunchKernel(k.eval_qk_norm_rope_kv_write, blocks, 1, 1, 512, 1, 1, 0, s, args, nullptr);
            }

            PROF_RECORD_LAYER(il, 3);  // boundary 3: after rope_kv, before attn

            // Attention decode — baseline fattn-vec.cuh, with ALiBi support
            {
                int kv_len = position + 1;
                // SWA: for iSWA layers, limit attention to last n_swa positions
                int kv_cache_offset = 0;
                if (c.layer_use_swa[il] && c.n_swa > 0 && kv_len > c.n_swa) {
                    kv_cache_offset = kv_len - c.n_swa;
                    kv_len = c.n_swa;
                }
                float alibi_mb = c.alibi_max_bias;
                float alibi_m0v = c.alibi_m0;
                float alibi_m1v = c.alibi_m1;
                int   alibi_nhl = c.alibi_n_head_log2;
                int   cur_pos = position;
                const float * rel_bias = nullptr;  // no T5 relative position bias for Llama-family
                float softcap = c.attn_softcap_val; // Gemma2/3/4: tanh(score/cap)*cap (0 = disabled)
                // SWA: offset K/V cache pointers to skip positions before the window
                // Cache layout: [kv_head, max_seq_len, head_dim] f16
                // Offset by kv_cache_offset positions = kv_cache_offset * head_dim * sizeof(half) bytes
                const char * k_ptr = (const char *)c.k_cache_ptrs[attn_idx]
                    + (size_t)kv_cache_offset * c.fa_head_dim * 2;  // 2 bytes per f16 element
                const char * v_ptr = (const char *)c.v_cache_ptrs[attn_idx]
                    + (size_t)kv_cache_offset * c.fa_head_dim * 2;
                void * args[] = { (void *)&b.proj_scratch,
                                  (void *)&k_ptr,
                                  (void *)&v_ptr,
                                  (void *)&b.attn_out,
                                  (void *)&kv_len, (void *)&c.max_seq_len,
                                  (void *)&alibi_mb, (void *)&alibi_m0v,
                                  (void *)&alibi_m1v, (void *)&alibi_nhl,
                                  (void *)&cur_pos, (void *)&rel_bias,
                                  (void *)&softcap, (void *)&d_params_ptr };
                // Attention kernel selection — override with GFX1100_ATTN env var:
                //   "wmma" = force WMMA (rocWMMA tensor cores)
                //   "tile" = force TILE (scalar, arbitrary D)
                //   "vec"  = force VEC (vectorized, D%2==0)
                //   unset  = auto (WMMA > TILE > VEC based on D and availability)
                static const char * attn_override = getenv("GFX1100_ATTN");
                bool use_wmma = false, use_tile = false;
                if (attn_override) {
                    if (strcmp(attn_override, "wmma") == 0) use_wmma = (k.eval_attention_decode_wmma != nullptr);
                    else if (strcmp(attn_override, "tile") == 0) use_tile = (k.eval_attention_decode_tile != nullptr);
                    // "vec" = neither wmma nor tile → falls through to VEC
                } else {
                    // Auto: WMMA for D<=128 (fastest on RDNA3 with rocWMMA), VEC for D>128, TILE as fallback
                    use_wmma = (k.eval_attention_decode_wmma && c.fa_head_dim % 16 == 0 && c.fa_head_dim <= 128);
                    use_tile = (!use_wmma && c.fa_head_dim % 64 != 0 && k.eval_attention_decode_tile);
                }

                if (use_wmma) {
                    hipModuleLaunchKernel(k.eval_attention_decode_wmma, c.fa_n_q_heads, 1, 1, 32, 4, 1, 0, s, args, nullptr);
                } else if (use_tile) {
                    hipModuleLaunchKernel(k.eval_attention_decode_tile, c.fa_n_q_heads, 1, 1, 32, 1, 1, 0, s, args, nullptr);
                } else {
                    // Use parallel-block VEC attention for better GPU occupancy
                    int pb = b.attn_parallel_blocks;
                    void * pb_args[] = { (void *)&b.proj_scratch,
                                         (void *)&k_ptr,
                                         (void *)&v_ptr,
                                         (void *)&b.attn_out,
                                         (void *)&b.attn_partial,
                                         (void *)&b.attn_meta,
                                         (void *)&kv_len, (void *)&c.max_seq_len,
                                         (void *)&alibi_mb, (void *)&alibi_m0v,
                                         (void *)&alibi_m1v, (void *)&alibi_nhl,
                                         (void *)&cur_pos, (void *)&rel_bias,
                                         (void *)&softcap, (void *)&d_params_ptr,
                                         (void *)&pb };
                    hipModuleLaunchKernel(k.eval_attention_decode_pb,
                        c.fa_n_q_heads, pb, 1,   // grid: heads * parallel_blocks
                        32, 4, 1,                 // block: 128 threads
                        0, s, pb_args, nullptr);
                    // If multi-block, run reduction kernel
                    if (pb > 1) {
                        int combine_threads = c.fa_head_dim <= 256 ? c.fa_head_dim : 256;
                        void * comb_args[] = { (void *)&b.attn_partial,
                                               (void *)&b.attn_meta,
                                               (void *)&b.attn_out,
                                               (void *)&pb };
                        hipModuleLaunchKernel(k.eval_attention_combine,
                            c.fa_n_q_heads, 1, 1,
                            combine_threads, 1, 1,
                            0, s, comb_args, nullptr);
                    }
                }
            }

            PROF_RECORD_LAYER(il, 4);  // boundary 4: after attn, before o_proj
            if (il == 0) dump_gpu("L0_attn_out", b.attn_out, c.fa_n_q_heads * c.fa_head_dim);

            // Dump attention output for every layer when bisecting divergence
            if (dump_enabled && dump_token_count == dump_target_pos) {
                int attn_size = c.fa_n_q_heads * c.fa_head_dim;
                hipStreamSynchronize(s);
                std::vector<float> h(attn_size);
                hipMemcpy(h.data(), b.attn_out, attn_size * sizeof(float), hipMemcpyDeviceToHost);
                double sum = 0, sumsq = 0;
                for (int i = 0; i < attn_size; i++) { sum += h[i]; sumsq += h[i]*h[i]; }
                fprintf(stderr, "  ATTN_OUT L%d: mean=%.8f rms=%.8f first4=[%.6f %.6f %.6f %.6f]\n",
                        il, sum/attn_size, sqrt(sumsq/attn_size), h[0], h[1], h[2], h[3]);
                char path[256];
                snprintf(path, sizeof(path), "dump_mk_L%d_attn_%d.bin", il, position);
                FILE * f = fopen(path, "wb");
                if (f) { fwrite(h.data(), sizeof(float), attn_size, f); fclose(f); }
            }

            // O projection — baseline: cur = wo × attn_out (+ bo) (* wo_s)
            // For sequential: + residual. For parallel: write to separate buffer.
            // Try fused quantize+matvec to eliminate q8_act global round-trip.
            {
                int n = c.fa_n_q_heads * c.fa_head_dim;
                bool is_o_quant = !is_float_type(lw.types[6]);
                bool no_o_extras = (!lw.bias_o && !lw.scale_o);
                // Disabled: fused quantize+matvec is 9-16x slower than separate kernels (shared memory Q8 overhead)
                hipFunction_t fused_qm = (hipFunction_t)nullptr;
                hipFunction_t fused_qmr = (hipFunction_t)nullptr;

                if (c.use_par_res || c.use_shared_norm_ffn) {
                    // Parallel attn+FFN or shared-norm FFN (Cohere2):
                    // O proj → attn_out (reuse buffer), NO residual add yet
                    if (fused_qm) {
                        int smem_size = (n / 32) * 36;
                        void * args[] = { (void *)&b.attn_out, (void *)&lw.ptrs[6], (void *)&lw.strides[6],
                                          (void *)&b.attn_out, (void *)&n, (void *)&H };
                        hipModuleLaunchKernel(fused_qm, H, 1, 1, 32, 4, 1, smem_size, s, args, nullptr);
                    } else {
                        int q8blocks = (n + 511) / 512;
                        void * q8args[] = { (void *)&b.attn_out, (void *)&b.q8_act, (void *)&n };
                        hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
                        launch_matvec(lw.types[6], lw.ptrs[6], lw.strides[6], b.attn_out, n, H);
                    }
                    if (lw.bias_o)  launch_add(b.attn_out, lw.bias_o, H);
                    if (lw.scale_o) launch_mul(b.attn_out, lw.scale_o, H);
                } else if (c.has_swin_norm || lw.attn_post_norm || lw.scale_o || lw.bias_o) {
                    // Sequential non-fused: matvec → bo → wo_s → post_norm/swin_norm → residual
                    if (fused_qm) {
                        int smem_size = (n / 32) * 36;
                        void * args[] = { (void *)&b.attn_out, (void *)&lw.ptrs[6], (void *)&lw.strides[6],
                                          (void *)&b.hidden, (void *)&n, (void *)&H };
                        hipModuleLaunchKernel(fused_qm, H, 1, 1, 32, 4, 1, smem_size, s, args, nullptr);
                    } else {
                        int q8blocks = (n + 511) / 512;
                        void * q8args[] = { (void *)&b.attn_out, (void *)&b.q8_act, (void *)&n };
                        hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
                        launch_matvec(lw.types[6], lw.ptrs[6], lw.strides[6], b.hidden, n, H);
                    }
                    if (lw.bias_o)  launch_add(b.hidden, lw.bias_o, H);
                    if (lw.scale_o) launch_mul(b.hidden, lw.scale_o, H);
                    // Post-attention norm: Gemma2/3/4 (attn_post_norm) or Chameleon (swin_norm)
                    const void * post_attn_w = lw.attn_post_norm ? lw.attn_post_norm :
                                               (c.has_swin_norm ? lw.ptrs[0] : nullptr);
                    // Granite residual scale
                    if (c.f_residual_scale != 0.0f && c.f_residual_scale != 1.0f) {
                        float rs = c.f_residual_scale;
                        int n2 = H;
                        void * args[] = { (void *)&b.hidden, (void *)&b.hidden, (void *)&rs, (void *)&n2 };
                        hipModuleLaunchKernel(k.eval_scale_scalar, (n2+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                    }
                    if (post_attn_w) {
                        // Fusion 3: if post-attn norm is followed by pre-FFN norm (Gemma-2/3/4),
                        // fuse both norms + residual add + q8 quantize into 1 kernel (2 launches → 1)
                        // Conditions: RMSNorm, not parallel, not swin, not shared_norm, standard FFN norm
                        int post_norm_idx = (c.layer_types[il] == 0) ? 7 : 10;
                        bool can_fuse_norms = !getenv("GFX1100_NOFUSE") && (c.norm_type != 2) &&  // not LayerNorm
                                              !c.use_par_res && !c.use_shared_norm_ffn && !c.has_swin_norm &&
                                              lw.ptrs[post_norm_idx] != nullptr &&  // FFN norm weight exists
                                              !lw.ffn_norm_bias &&  // no FFN norm bias
                                              (c.f_residual_scale == 0.0f || c.f_residual_scale == 1.0f);
                        if (can_fuse_norms) {
                            // Split into 4 separate kernel launches matching baseline's operations:
                            // 1. post-attn rmsnorm → proj_scratch (matches baseline's rms_norm_f32<N,true,false>)
                            // 2. add residual → hidden (matches baseline's fused add inside rms_norm_f32<N,true,true>)
                            // 3. update residual = hidden
                            // 4. pre-FFN rmsnorm + q8 (matches baseline's rms_norm_f32<N,true,false>)
                            int hn = H;
                            if (il == 0) {
                                dump_gpu("L0_fused_in_hidden", b.hidden, H);
                                dump_gpu("L0_fused_in_residual", b.residual, H);
                            }
                            // Step 1: post-attn rmsnorm(hidden, weight) → proj_scratch
                            {
                                void * a[] = { (void *)&b.hidden, (void *)&post_attn_w,
                                               (void *)&b.proj_scratch, (void *)&b.hidden, (void *)&hn };
                                hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, norm_threads, 1, 1, 0, s, a, nullptr);
                            }
                            // Step 2: hidden = proj_scratch + residual
                            {
                                void * a[] = { (void *)&b.proj_scratch, (void *)&b.residual, (void *)&b.hidden, (void *)&hn };
                                hipModuleLaunchKernel(k.eval_add_residual, (hn+255)/256, 1, 1, 256, 1, 1, 0, s, a, nullptr);
                            }
                            // Step 3: residual = hidden (for next layer/FFN residual)
                            hipMemcpyAsync(b.residual, b.hidden, H * sizeof(float), hipMemcpyDeviceToDevice, s);
                            // Step 4: pre-FFN rmsnorm(hidden, ffn_norm_w) → norm_out + q8_act
                            {
                                const void * ffn_nw = lw.ptrs[post_norm_idx];
                                launch_norm_q8(b.hidden, ffn_nw, lw.ffn_norm_bias,
                                               b.norm_out, b.residual, b.q8_act);
                            }
                            if (il == 0) {
                                dump_gpu("L0_fused_out_norm", b.norm_out, H);
                                dump_gpu("L0_fused_out_hidden", b.hidden, H);
                            }

                            // Pre-FFN norm is done — skip to FFN
                            attn_idx++;
                            PROF_RECORD_LAYER(il, 5);
                            goto ffn_norm_done;
                        } else {
                            // Split into 2 kernels to match baseline's separate rmsnorm + add:
                            // 1. rmsnorm(hidden, weight) → proj_scratch (norm output)
                            // 2. add(proj_scratch, residual) → hidden
                            int hn = H;
                            // eval_rmsnorm_q8 writes: norm_out = input*rstd*weight, residual_out = input
                            // We use proj_scratch as norm output, hidden as dummy residual target
                            void * norm_args[] = { (void *)&b.hidden, (void *)&post_attn_w,
                                                   (void *)&b.proj_scratch, (void *)&b.hidden, (void *)&hn };
                            hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, norm_threads, 1, 1, 0, s, norm_args, nullptr);
                            // Now add residual: hidden = proj_scratch + residual
                            // proj_scratch has the normed output, residual has the original input
                            void * add_args[] = { (void *)&b.proj_scratch, (void *)&b.residual, (void *)&b.hidden, (void *)&hn };
                            hipModuleLaunchKernel(k.eval_add_residual, (hn+255)/256, 1, 1, 256, 1, 1, 0, s, add_args, nullptr);
                            // Update residual for next norm/layer (matches baseline's sa_out becoming the new residual)
                            hipMemcpyAsync(b.residual, b.hidden, H * sizeof(float), hipMemcpyDeviceToDevice, s);
                        }
                    } else {
                        launch_add(b.hidden, (const void *)b.residual, H);
                    }
                } else {
                    // Sequential: try fused quantize+matvec+residual (1 launch instead of 3)
                    if (fused_qmr) {
                        int smem_size = (n / 32) * 36;
                        void * args[] = { (void *)&b.attn_out, (void *)&lw.ptrs[6], (void *)&lw.strides[6],
                                          (void *)&b.residual, (void *)&b.hidden, (void *)&n, (void *)&H };
                        hipModuleLaunchKernel(fused_qmr, H, 1, 1, 32, 4, 1, smem_size, s, args, nullptr);
                    } else {
                        // Fallback: quantize + matvec_res (2 launches)
                        int q8blocks = (n + 511) / 512;
                        void * q8args[] = { (void *)&b.attn_out, (void *)&b.q8_act, (void *)&n };
                        hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
                        launch_matvec_res(lw.types[6], lw.ptrs[6], lw.strides[6], b.residual, b.hidden, n, H);
                    }
                }
            }

            attn_idx++;

        } else {
            // ---- DeltaNet layer ----
            int dn_v_size = c.dn_n_heads * c.dn_value_dim;
            int dn_qk_size = c.dn_n_k_heads * c.dn_key_dim;
            int dn_conv_ch = dn_qk_size * 2 + dn_v_size;

            // QKV projection → proj_scratch
            launch_matvec(lw.types[1], lw.ptrs[1], lw.strides[1], b.proj_scratch, H, dn_conv_ch);
            // Z projection → z_scratch
            launch_matvec(lw.types[2], lw.ptrs[2], lw.strides[2], b.z_scratch, H, dn_v_size);
            // Beta projection → beta_scratch
            launch_matvec(lw.types[3], lw.ptrs[3], lw.strides[3], b.beta_scratch, H, c.dn_n_heads);
            // Alpha projection → alpha_scratch
            launch_matvec(lw.types[4], lw.ptrs[4], lw.strides[4], b.alpha_scratch, H, c.dn_n_heads);

            // Conv1d + SiLU → q_buf, k_buf, v_buf in attn_out (reuse buffer)
            {
                float * layer_conv = b.conv_bufs + (long long)dn_idx * dn_conv_ch * c.dn_conv_kernel;
                const void * conv_w = lw.ptrs[5]; // f32 conv1d weight
                float * q_buf = b.attn_out;
                float * k_buf = b.attn_out + c.dn_n_heads * c.dn_key_dim;
                float * v_buf = b.attn_out + c.dn_n_heads * c.dn_key_dim * 2;
                int nv = c.dn_n_heads, nk = c.dn_n_k_heads, kd = c.dn_key_dim, vd = c.dn_value_dim;
                int ck = c.dn_conv_kernel, cc = dn_conv_ch;
                void * args[] = { (void *)&b.proj_scratch, (void *)&conv_w, (void *)&layer_conv,
                                  (void *)&q_buf, (void *)&k_buf, (void *)&v_buf,
                                  (void *)&nv, (void *)&nk, (void *)&kd, (void *)&vd, (void *)&ck, (void *)&cc };
                hipModuleLaunchKernel(k.eval_dn_conv1d_silu, nv, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }

            // L2 normalize Q and K
            {
                float * q_buf = b.attn_out;
                float * k_buf = b.attn_out + c.dn_n_heads * c.dn_key_dim;
                int nv = c.dn_n_heads, kd = c.dn_key_dim;
                float eps = c.norm_eps;
                void * q_args[] = { (void *)&q_buf, (void *)&nv, (void *)&kd, (void *)&eps };
                hipModuleLaunchKernel(k.eval_dn_l2_norm, nv, 1, 1, 256, 1, 1, 0, s, q_args, nullptr);
                // K: n_k_heads, but each V head maps to a K head via cyclic
                // Actually need to normalize per V-head's K (which shares with other V heads)
                // For simplicity: normalize all n_heads * key_dim
                void * k_args[] = { (void *)&k_buf, (void *)&nv, (void *)&kd, (void *)&eps };
                hipModuleLaunchKernel(k.eval_dn_l2_norm, nv, 1, 1, 256, 1, 1, 0, s, k_args, nullptr);
            }

            // DeltaNet recurrence + gated norm
            {
                float * q_buf = b.attn_out;
                float * k_buf = b.attn_out + c.dn_n_heads * c.dn_key_dim;
                float * v_buf = b.attn_out + c.dn_n_heads * c.dn_key_dim * 2;
                float * layer_state = b.dn_states + (long long)dn_idx * c.dn_n_heads * c.dn_key_dim * c.dn_value_dim;
                const void * ssm_a = lw.ptrs[6];
                const void * dt_bias = lw.ptrs[7];
                const void * norm_w = lw.ptrs[8];
                int nh = c.dn_n_heads, kd = c.dn_key_dim, vd = c.dn_value_dim;
                float eps = c.norm_eps;
                // Output goes to proj_scratch (reuse)
                void * args[] = { (void *)&q_buf, (void *)&k_buf, (void *)&v_buf,
                                  (void *)&b.beta_scratch, (void *)&b.alpha_scratch,
                                  (void *)&ssm_a, (void *)&dt_bias,
                                  (void *)&b.z_scratch, (void *)&norm_w,
                                  (void *)&layer_state, (void *)&b.proj_scratch,
                                  (void *)&nh, (void *)&kd, (void *)&vd, (void *)&eps };
                hipModuleLaunchKernel(k.eval_dn_recurrence, nh, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }

            // O projection + residual
            {
                int n = dn_v_size;
                int q8blocks = (n + 511) / 512;
                void * q8args[] = { (void *)&b.proj_scratch, (void *)&b.q8_act, (void *)&n };
                hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);

                const void * w = lw.ptrs[9]; long long st = lw.strides[9];
                launch_matvec_res(lw.types[9], lw.ptrs[9], lw.strides[9], b.residual, b.hidden, dn_v_size, H);
            }

            dn_idx++;
        }

        PROF_RECORD_LAYER(il, 5);  // boundary 5: after o_proj, before ffn_norm
        if (il == 0) {
            dump_gpu("L0_after_oproj", b.hidden, H);
            dump_gpu("L0_residual_pre_ffn", b.residual, H);
        }

        // Phase 5: Pre-FFN norm (or skip for swin_norm/shared_norm modes)
        // For parallel attn+FFN: norm reads from ORIGINAL input (b.residual), not b.hidden
        // NOTE: if Fusion 3 already ran (via goto ffn_norm_done), this entire block is skipped.
        if (c.use_shared_norm_ffn) {
            // Cohere2: FFN reads from pre-attention norm output (saved in mlp_inter)
            hipMemcpyAsync(b.norm_out, b.mlp_inter, H * sizeof(float), hipMemcpyDeviceToDevice, s);
            int n = H, blocks = (n + 511) / 512;
            void * q8args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
        } else if (c.has_swin_norm) {
            // Chameleon post-norm: skip pre-FFN norm
            hipMemcpyAsync(b.norm_out, b.hidden, H * sizeof(float), hipMemcpyDeviceToDevice, s);
            hipMemcpyAsync(b.residual, b.hidden, H * sizeof(float), hipMemcpyDeviceToDevice, s);
            int n = H, blocks = (n + 511) / 512;
            void * q8args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
            hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
        } else {
            // Fused: FFN norm + residual + Q8 quantize in ONE kernel launch
            int post_norm_idx = (c.layer_types[il] == 0) ? 7 : 10;
            float * norm_input = c.use_par_res ? b.residual : b.hidden;
            float * norm_residual = c.use_par_res ? b.hidden : b.residual;
            launch_norm_q8(norm_input, lw.ptrs[post_norm_idx], lw.ffn_norm_bias,
                           b.norm_out, norm_residual, b.q8_act);
        }

        ffn_norm_done:
        PROF_RECORD_LAYER(il, 6);  // boundary 6: after ffn_norm+quantize, before ffn_proj
        if (il == 0) dump_gpu("L0_ffn_norm", b.norm_out, H);

        // Phase 6: FFN — baseline: if (ffn_gate_inp == nullptr) standard FFN else MoE
        if (lw.ffn_gate_inp) {
            // ================================================================
            // MoE FFN — port of baseline build_moe_ffn (llama-graph.cpp:1268-1622)
            // For n_tokens=1 (decode), simplified to single-vector ops.
            //
            // Flow (Qwen2MoE pattern, most common):
            //   1. logits = gate_inp @ cur     [n_expert]
            //   2. probs = softmax(logits)     [n_expert]
            //   3. selected = argsort_topk(probs, n_used)  [n_used]
            //   4. weights = probs[selected]   [n_used]
            //   5. For each selected expert i:
            //      gate_i = gate_exps[expert_id] @ cur   [n_ff]
            //      up_i   = up_exps[expert_id] @ cur     [n_ff]
            //      act_i  = silu(gate_i) * up_i          [n_ff]
            //      down_i = down_exps[expert_id] @ act_i [n_embd]
            //      hidden += weights[i] * down_i
            // ================================================================
            const int n_expert = c.moe_n_experts;
            const int n_used   = c.moe_n_experts_used;

            // Step 1: Router logits — baseline line 1299: logits = gate_inp @ cur
            // gate_inp is [n_expert, n_embd], cur is q8_act [n_embd]
            // Output: proj_scratch [n_expert] (reuse buffer, n_expert << max_proj)
            launch_matvec(lw.ffn_gate_inp_type, lw.ffn_gate_inp,
                          lw.ffn_gate_inp_stride, b.proj_scratch, H, n_expert);

            // Step 2: Gating — baseline lines 1311-1326
            // Most MoE: softmax. Some: sigmoid. Dispatch on moe_gating_op.
            {
                float scale_val = 1.0f;
                int n = n_expert;
                void * args[] = { (void *)&b.proj_scratch, (void *)&b.proj_scratch, (void *)&n, (void *)&scale_val };
                if (lw.moe_gating_op == 2) {
                    // sigmoid — baseline line 1318
                    hipModuleLaunchKernel(k.eval_sigmoid, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                } else {
                    // softmax (default) — baseline line 1314
                    // eval_softmax_row uses extern __shared__ — need (256/32)*sizeof(float) = 32 bytes
                    hipModuleLaunchKernel(k.eval_softmax_row, 1, 1, 1, 256, 1, 1,
                                         (256/32) * sizeof(float), s, args, nullptr);
                }
            }
            // proj_scratch[0..n_expert-1] now contains routing probabilities

            // Step 3: Top-k selection — baseline line 1374: argsort_top_k
            // eval_argsort_desc outputs sorted indices into mlp_inter (reuse as int buffer)
            // Kernel: eval_argsort_desc(x, dst, ncols, ncols_pad) — bitonic sort needs power-of-2 block
            {
                int n = n_expert;
                int npad = 1;
                while (npad < n) npad *= 2;
                void * inp = (void *)b.proj_scratch;
                void * out = (void *)b.mlp_inter; // reuse as int32 output
                void * args[] = { &inp, &out, &n, &npad };
                hipModuleLaunchKernel(k.eval_argsort_desc, 1, 1, 1, npad, 1, 1,
                                     npad * sizeof(int), s, args, nullptr);
            }
            // mlp_inter[0..n_expert-1] has sorted indices (as int32 reinterpreted in float buffer)

            // Copy sorted indices and probs to dedicated MoE buffers (avoids buffer aliasing)
            hipMemcpyAsync(b.moe_sorted_ids, b.mlp_inter, n_expert * sizeof(int), hipMemcpyDeviceToDevice, s);
            hipMemcpyAsync(b.moe_probs, b.proj_scratch, n_expert * sizeof(float), hipMemcpyDeviceToDevice, s);

            // Normalize MoE weights on GPU — zero D2H
            {
                int do_norm = lw.moe_norm_w ? 1 : 0;
                float ws = lw.moe_w_scale;
                void * nargs[] = { (void *)&b.moe_probs, (void *)&b.moe_sorted_ids,
                                   (void *)&n_used, (void *)&do_norm, (void *)&ws };
                hipModuleLaunchKernel(k.eval_moe_normalize_weights, 1, 1, 1, 1, 1, 1, 0, s, nargs, nullptr);
            }

            // Zero the output accumulator
            hipMemsetAsync(b.hidden, 0, H * sizeof(float), s);

            // Per-expert FFN — fully on GPU, no D2H, no hipStreamSynchronize
            long long gate_exp_stride = (long long)FF * lw.ffn_gate_exps_stride;
            long long up_exp_stride   = (long long)FF * lw.ffn_up_exps_stride;
            long long down_exp_stride = (long long)H  * lw.ffn_down_exps_stride;

            auto pick_moe_mv = [&](int type) -> hipFunction_t {
                switch (type) {
                    case  0: return k.eval_moe_matvec_f32;
                    case  1: return k.eval_moe_matvec_f16;
                    case 30: return k.eval_moe_matvec_bf16;
                    case  2: return k.eval_moe_matvec_q4_0;
                    case  3: return k.eval_moe_matvec_q4_1;
                    case  6: return k.eval_moe_matvec_q5_0;
                    case  7: return k.eval_moe_matvec_q5_1;
                    case  8: return k.eval_moe_matvec_q8_0;
                    case 10: return k.eval_moe_matvec_q2k;
                    case 11: return k.eval_moe_matvec_q3k;
                    case 12: return k.eval_moe_matvec_q4k;
                    case 13: return k.eval_moe_matvec_q5k;
                    case 14: return k.eval_moe_matvec_q6k;
                    case 16: return k.eval_moe_matvec_iq2_xxs;
                    case 17: return k.eval_moe_matvec_iq2_xs;
                    case 18: return k.eval_moe_matvec_iq3_xxs;
                    case 19: return k.eval_moe_matvec_iq1_s;
                    case 20: return k.eval_moe_matvec_iq4_nl;
                    case 21: return k.eval_moe_matvec_iq3_s;
                    case 22: return k.eval_moe_matvec_iq2_s;
                    case 23: return k.eval_moe_matvec_iq4_xs;
                    case 29: return k.eval_moe_matvec_iq1_m;
                    case 39: return k.eval_moe_matvec_mxfp4;
                    case 40: return k.eval_moe_matvec_nvfp4;
                    default:
                        fprintf(stderr, "gfx1100: unsupported MoE matvec type %d\n", type);
                        return k.eval_moe_matvec_q4k;
                }
            };

            auto is_moe_float_type = [](int type) -> bool {
                return type == 0 || type == 1 || type == 30;
            };

            for (int ei = 0; ei < n_used; ei++) {
                // Re-quantize norm_out for this expert (skip first — q8_act still valid from Phase 5b)
                if (ei > 0) {
                    int n = H, q8blocks = (n + 511) / 512;
                    void * q8args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
                    hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
                }

                // Gate: fused MoE matvec — reads expert ID from moe_sorted_ids on GPU
                {
                    hipFunction_t fn = pick_moe_mv(lw.ffn_gate_exps_type);
                    bool is_float = is_moe_float_type(lw.ffn_gate_exps_type);
                    void * input = is_float ? (void *)b.norm_out : (void *)b.q8_act;
                    void * args[] = { (void *)&lw.ffn_gate_exps, (void *)&gate_exp_stride,
                                      (void *)&lw.ffn_gate_exps_stride, (void *)&input,
                                      (void *)&b.mlp_inter, (void *)&b.moe_sorted_ids,
                                      (void *)&ei, (void *)&H, (void *)&FF };
                    if (is_float)
                        hipModuleLaunchKernel(fn, FF, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                    else
                        hipModuleLaunchKernel(fn, FF, 1, 1, 32, 4, 1, 0, s, args, nullptr);
                }

                // Up: fused MoE matvec
                {
                    hipFunction_t fn = pick_moe_mv(lw.ffn_up_exps_type);
                    bool is_float = is_moe_float_type(lw.ffn_up_exps_type);
                    void * input = is_float ? (void *)b.norm_out : (void *)b.q8_act;
                    void * args[] = { (void *)&lw.ffn_up_exps, (void *)&up_exp_stride,
                                      (void *)&lw.ffn_up_exps_stride, (void *)&input,
                                      (void *)&b.proj_scratch, (void *)&b.moe_sorted_ids,
                                      (void *)&ei, (void *)&H, (void *)&FF };
                    if (is_float)
                        hipModuleLaunchKernel(fn, FF, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                    else
                        hipModuleLaunchKernel(fn, FF, 1, 1, 32, 4, 1, 0, s, args, nullptr);
                }

                // Activation: gate_fn(gate) * up
                {
                    int n = FF;
                    void * args[] = { (void *)&b.mlp_inter, (void *)&b.proj_scratch,
                                      (void *)&b.mlp_inter, (void *)&n };
                    hipFunction_t act_fn;
                    switch (c.act_type) {
                        case ACT_GELU:       act_fn = k.eval_gelu_erf_mul; break;
                        case ACT_GELU_TANH:  act_fn = k.eval_gelu_mul; break;
                        case ACT_GELU_QUICK: act_fn = k.eval_gelu_quick_mul; break;
                        case ACT_RELU2:      act_fn = k.eval_relu2_mul; break;
                        case ACT_SILU:
                        default:             act_fn = k.eval_silu_mul; break;
                    }
                    hipModuleLaunchKernel(act_fn, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }

                // Quantize activation for down projection
                {
                    int n = FF, q8blocks = (n + 511) / 512;
                    void * q8args[] = { (void *)&b.mlp_inter, (void *)&b.q8_act, (void *)&n };
                    hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
                }

                // Down: fused MoE matvec — output to proj_scratch
                {
                    hipFunction_t fn = pick_moe_mv(lw.ffn_down_exps_type);
                    bool is_float = is_moe_float_type(lw.ffn_down_exps_type);
                    void * input = is_float ? (void *)b.mlp_inter : (void *)b.q8_act;
                    void * args[] = { (void *)&lw.ffn_down_exps, (void *)&down_exp_stride,
                                      (void *)&lw.ffn_down_exps_stride, (void *)&input,
                                      (void *)&b.proj_scratch, (void *)&b.moe_sorted_ids,
                                      (void *)&ei, (void *)&FF, (void *)&H };
                    if (is_float)
                        hipModuleLaunchKernel(fn, H, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                    else
                        hipModuleLaunchKernel(fn, H, 1, 1, 32, 4, 1, 0, s, args, nullptr);
                }

                // Weighted accumulate: hidden += probs[sorted_ids[ei]] * proj_scratch
                {
                    int n = H;
                    void * args[] = { (void *)&b.hidden, (void *)&b.proj_scratch,
                                      (void *)&b.moe_probs, (void *)&b.moe_sorted_ids,
                                      (void *)&ei, (void *)&n };
                    hipModuleLaunchKernel(k.eval_moe_weighted_add, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }
            }

            // Qwen2MoE shared expert — baseline qwen2moe.cpp lines 102-126
            // gate = sigmoid(ffn_gate_inp_shexp @ cur)
            // ffn = down_shexp @ (silu(gate_shexp @ cur) * (up_shexp @ cur))
            // hidden += ffn * gate
            if (lw.ffn_gate_inp_shexp) {
                // Shared expert gate: sigmoid(ffn_gate_inp_shexp @ norm_out) → scalar gate
                quant_and_launch_matvec(lw.ffn_gate_inp_shexp_type, lw.ffn_gate_inp_shexp,
                                         lw.ffn_gate_inp_shexp_stride, b.norm_out,
                                         b.proj_scratch, H, 1, s);
                // sigmoid on the gate scalar
                {
                    int n = 1;
                    void * args[] = { (void *)&b.proj_scratch, (void *)&b.proj_scratch, (void *)&n };
                    hipModuleLaunchKernel(k.eval_sigmoid, 1, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }
                // Save gate scalar to attn_out[0] before FFN overwrites proj_scratch
                hipMemcpyAsync(b.attn_out, b.proj_scratch, sizeof(float), hipMemcpyDeviceToDevice, s);
                float * d_shexp_gate = b.attn_out;

                // Shared expert FFN: SwiGLU (gate_shexp, up_shexp, down_shexp)
                if (lw.ffn_gate_shexp && lw.ffn_up_shexp && lw.ffn_down_shexp) {
                    // Gate projection
                    launch_matvec(lw.ffn_gate_shexp_type, lw.ffn_gate_shexp, lw.ffn_gate_shexp_stride,
                                  b.mlp_inter, H, FF);
                    // Up projection
                    launch_matvec(lw.ffn_up_shexp_type, lw.ffn_up_shexp, lw.ffn_up_shexp_stride,
                                  b.proj_scratch, H, FF);
                    // SiLU(gate) * up
                    {
                        int n = FF;
                        void * args[] = { (void *)&b.mlp_inter, (void *)&b.proj_scratch,
                                          (void *)&b.mlp_inter, (void *)&n };
                        hipModuleLaunchKernel(k.eval_silu_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                    }
                    // Quantize + down projection
                    {
                        int n = FF, q8blocks = (n+511)/512;
                        void * q8args[] = { (void *)&b.mlp_inter, (void *)&b.q8_act, (void *)&n };
                        hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
                    }
                    launch_matvec(lw.ffn_down_shexp_type, lw.ffn_down_shexp, lw.ffn_down_shexp_stride,
                                  b.proj_scratch, FF, H);
                    // Apply gate: proj_scratch *= d_shexp_gate[0] (both on GPU)
                    {
                        int n = H;
                        void * args[] = { (void *)&b.proj_scratch, (void *)&d_shexp_gate, (void *)&n };
                        hipModuleLaunchKernel(k.eval_moe_gate_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                    }
                    // hidden += proj_scratch
                    launch_add(b.hidden, (const void *)b.proj_scratch, H);
                }
            }

            PROF_RECORD_LAYER(il, 7);  // boundary 7 (MoE): after expert FFN, before residual add

            // Add residual: hidden += residual — baseline line 127: cur = ggml_add(cur, ffn_inp)
            launch_add(b.hidden, (const void *)b.residual, H);

            PROF_RECORD_LAYER(il, 8);  // boundary 8 (MoE): end of layer

            // Skip standard FFN path
            goto layer_done;
        }

        // Standard FFN: gate + up + silu_mul + down — baseline lines 114-119
        {
            int gate_idx = (c.layer_types[il] == 0) ? 8 : 11;
            int up_idx   = gate_idx + 1;
            int in_dim = H, out_dim = FF;

            if (!lw.ptrs[gate_idx] && c.act_type == ACT_SILU) {
                // Fused SwiGLU (Phi3, ChatGLM): single up matrix produces 2*FF,
                // split into gate[0:FF] and up[FF:2*FF], then silu(gate) * up.
                // Baseline phi3.cpp line 100: LLM_FFN_SWIGLU, ffn_gate=NULL
                int fused_dim = 2 * FF;
                launch_matvec(lw.types[up_idx], lw.ptrs[up_idx], lw.strides[up_idx], b.proj_scratch, in_dim, fused_dim);
                if (lw.ffn_up_bias) {
                    const void * ub = lw.ffn_up_bias;
                    int n = fused_dim;
                    void * ba[] = { (void *)&b.proj_scratch, (void *)&ub, (void *)&b.proj_scratch, (void *)&n };
                    hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, s, ba, nullptr);
                }
                {
                    float * gate_half = b.proj_scratch;
                    float * up_half = b.proj_scratch + FF;
                    int n = FF;
                    void * args[] = { (void *)&gate_half, (void *)&up_half, (void *)&b.mlp_inter, (void *)&n };
                    hipModuleLaunchKernel(k.eval_silu_mul, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }
            } else if (!lw.ptrs[gate_idx]) {
                // Ungated sequential FFN (Falcon, BLOOM, GPT2, StarCoder, MPT, Phi2, etc.)
                // Baseline: up[H→FF] → activation(alone) → down[FF→H]
                // No gate matrix, no gate*up multiply — just activation applied element-wise
                launch_matvec(lw.types[up_idx], lw.ptrs[up_idx], lw.strides[up_idx], b.mlp_inter, in_dim, out_dim);
                if (lw.ffn_up_bias) {
                    const void * ub = lw.ffn_up_bias;
                    int n = out_dim;
                    void * ba[] = { (void *)&b.mlp_inter, (void *)&ub, (void *)&b.mlp_inter, (void *)&n };
                    hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, s, ba, nullptr);
                }
                if (lw.ffn_up_scale) launch_mul(b.mlp_inter, lw.ffn_up_scale, FF);
                // Apply activation alone (not gated) — dispatch by act_type
                {
                    int n = FF;
                    hipFunction_t act_fn;
                    switch (c.act_type) {
                        case ACT_GELU:       act_fn = k.eval_gelu_erf; break;
                        case ACT_GELU_TANH:  act_fn = k.eval_gelu; break;
                        case ACT_GELU_QUICK: act_fn = k.eval_gelu_erf; break; // standalone GELU-quick (reuse exact GELU for ungated path)
                        case ACT_RELU2:      act_fn = k.eval_relu; break; // squared relu handled separately
                        case ACT_SILU:
                        default:             act_fn = k.eval_silu; break;
                    }
                    void * args[] = { (void *)&b.mlp_inter, (void *)&b.mlp_inter, (void *)&n };
                    hipModuleLaunchKernel(act_fn, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                    // For squared ReLU: apply sqr after relu
                    if (c.act_type == ACT_RELU2) {
                        void * sq_args[] = { (void *)&b.mlp_inter, (void *)&b.mlp_inter, (void *)&n };
                        hipModuleLaunchKernel(k.eval_sqr, (n+255)/256, 1, 1, 256, 1, 1, 0, s, sq_args, nullptr);
                    }
                }
            } else {
            // Try fused gate+up+activation: 1 kernel launch instead of 3
            // Conditions: SiLU or GELU_TANH activation, same quant type for gate/up, no bias/scale
            bool same_type = (lw.types[gate_idx] == lw.types[up_idx]);
            bool no_bias_scale = (!lw.ffn_gate_bias && !lw.ffn_up_bias
                                  && !lw.ffn_gate_scale && !lw.ffn_up_scale);
            bool use_fused_silu = (c.act_type == ACT_SILU && same_type && no_bias_scale);
            bool use_fused_gelu = (c.act_type == ACT_GELU_TANH && same_type && no_bias_scale);
            hipFunction_t fused_fn = use_fused_silu ? pick_fused_gate_up_silu(lw.types[gate_idx])
                                   : use_fused_gelu ? pick_fused_gate_up_gelu(lw.types[gate_idx])
                                   : (hipFunction_t)nullptr;
            // Try fused matvec GLU (baseline has_fusion path): 1 kernel for gate+up+activation
            // Uses mmvq_generic_row with gate_weight — matching baseline's register layout exactly.
            hipFunction_t glu_fn = nullptr;
            int glu_op = -1;
            if (use_fused_silu || use_fused_gelu) {
                glu_op = use_fused_silu ? 0 : 1;  // MK_GLU_SWIGLU=0, MK_GLU_GEGLU=1
                switch (lw.types[gate_idx]) {
                    case 12: glu_fn = k.eval_matvec_glu_q4k; break;
                    case 14: glu_fn = k.eval_matvec_glu_q6k; break;
                    case  2: glu_fn = k.eval_matvec_glu_q4_0; break;
                    case  8: glu_fn = k.eval_matvec_glu_q8_0; break;
                }
            }
            if (glu_fn) {
                void * input = (void *)b.q8_act;
                void * args[] = {
                    (void *)&lw.ptrs[up_idx],   (void *)&lw.strides[up_idx],
                    (void *)&lw.ptrs[gate_idx], (void *)&lw.strides[gate_idx],
                    (void *)&input, (void *)&b.mlp_inter, (void *)&H, (void *)&FF,
                    (void *)&glu_op
                };
                hipModuleLaunchKernel(glu_fn, FF, 1, 1, 32, 8, 1, 0, s, args, nullptr);
            } else if (fused_fn) {
                // Fallback: old fused gate+up kernel (for types without GLU matvec)
                bool is_f = is_float_type(lw.types[gate_idx]);
                void * input = is_f ? (void *)b.norm_out : (void *)b.q8_act;
                void * args[] = {
                    (void *)&lw.ptrs[gate_idx], (void *)&lw.strides[gate_idx],
                    (void *)&lw.ptrs[up_idx],   (void *)&lw.strides[up_idx],
                    (void *)&input, (void *)&b.mlp_inter, (void *)&H, (void *)&FF
                };
                if (is_f) {
                    hipModuleLaunchKernel(fused_fn, FF, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                } else {
                    int smem_size = (H / 32) * 36;
                    hipModuleLaunchKernel(fused_fn, FF, 1, 1, 32, 4, 1, smem_size, s, args, nullptr);
                }
            } else {
            // Fallback: 3-launch gate + up + activation*mul
            // Gate projection — baseline: build_ffn arg ffn_gate + ffn_gate_b + ffn_gate_s
            launch_matvec(lw.types[gate_idx], lw.ptrs[gate_idx], lw.strides[gate_idx], b.mlp_inter, in_dim, out_dim);
            if (lw.ffn_gate_bias) {
                const void * gb = lw.ffn_gate_bias;
                int n = out_dim;
                void * ba[] = { (void *)&b.mlp_inter, (void *)&gb, (void *)&b.mlp_inter, (void *)&n };
                hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, s, ba, nullptr);
            }
            if (lw.ffn_gate_scale) launch_mul(b.mlp_inter, lw.ffn_gate_scale, FF);

            // Up projection
            launch_matvec(lw.types[up_idx], lw.ptrs[up_idx], lw.strides[up_idx], b.proj_scratch, in_dim, out_dim);
            if (lw.ffn_up_bias) {
                const void * ub = lw.ffn_up_bias;
                int n = out_dim;
                void * ba[] = { (void *)&b.proj_scratch, (void *)&ub, (void *)&b.proj_scratch, (void *)&n };
                hipModuleLaunchKernel(k.eval_add_residual, (n+255)/256, 1, 1, 256, 1, 1, 0, s, ba, nullptr);
            }
            if (lw.ffn_up_scale) launch_mul(b.proj_scratch, lw.ffn_up_scale, FF);

            // Activation(gate) * up → mlp_inter
            // Dispatch by act_type — baseline uses SiLU for Llama, GELU for Gemma2/3/4, etc.
            {
                int n = FF;
                void * args[] = { (void *)&b.mlp_inter, (void *)&b.proj_scratch,
                                  (void *)&b.mlp_inter, (void *)&n };
                hipFunction_t act_fn;
                switch (c.act_type) {
                    case ACT_GELU:       act_fn = k.eval_gelu_erf_mul; break;  // exact GELU
                    case ACT_GELU_TANH:  act_fn = k.eval_gelu_mul; break;      // tanh-approx (Gemma)
                    case ACT_GELU_QUICK: act_fn = k.eval_gelu_quick_mul; break; // quick GELU
                    case ACT_RELU2:      act_fn = k.eval_relu2_mul; break;     // squared ReLU
                    case ACT_SILU:                                              // SiLU (default)
                    default:             act_fn = k.eval_silu_mul; break;
                }
                hipModuleLaunchKernel(act_fn, (n + 255) / 256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
            }
            } // end else fallback (3-launch gate+up path)
            } // end else (standard gate+up path — fused or fallback)
        }

        PROF_RECORD_LAYER(il, 7);  // boundary 7: after ffn_proj (gate+up+act), before ffn_res (down+residual)
        if (il == 0) dump_gpu("L0_ffn_out", b.mlp_inter, FF);

        // Phase 7: Down projection + residual
        // Try fused quantize+matvec to eliminate q8_act global round-trip.
        {
            int down_idx = (c.layer_types[il] == 0) ? 10 : 13;
            bool is_down_quant = !is_float_type(lw.types[down_idx]);
            bool no_down_extras = (!lw.ffn_down_bias && !lw.ffn_down_scale);
            // Disabled: fused quantize+matvec is 9-16x slower than separate kernels
            hipFunction_t fused_qm_down = (hipFunction_t)nullptr;
            hipFunction_t fused_qmr_down = (hipFunction_t)nullptr;

            if (c.use_par_res || c.use_shared_norm_ffn) {
                // Parallel attn+FFN or shared-norm FFN:
                //   hidden = ffn_out + attn_out + inpL (three-way add)
                if (fused_qm_down) {
                    int smem_size = (FF / 32) * 36;
                    int n = FF;
                    void * args[] = { (void *)&b.mlp_inter, (void *)&lw.ptrs[down_idx], (void *)&lw.strides[down_idx],
                                      (void *)&b.hidden, (void *)&n, (void *)&H };
                    hipModuleLaunchKernel(fused_qm_down, H, 1, 1, 32, 4, 1, smem_size, s, args, nullptr);
                } else {
                    int n = FF, q8blocks = (n + 511) / 512;
                    void * q8args[] = { (void *)&b.mlp_inter, (void *)&b.q8_act, (void *)&n };
                    hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
                    launch_matvec(lw.types[down_idx], lw.ptrs[down_idx], lw.strides[down_idx], b.hidden, FF, H);
                }
                if (lw.ffn_down_bias)  launch_add(b.hidden, lw.ffn_down_bias, H);
                if (lw.ffn_down_scale) launch_mul(b.hidden, lw.ffn_down_scale, H);
                // Three-way combine
                launch_add(b.hidden, (const void *)b.attn_out, H);
                launch_add(b.hidden, (const void *)b.residual, H);
            } else if (c.has_swin_norm || lw.ffn_post_norm || lw.ffn_down_scale || lw.ffn_down_bias) {
                // Sequential non-fused: matvec → bias → scale → post_norm/swin_norm → residual
                if (fused_qm_down) {
                    int smem_size = (FF / 32) * 36;
                    int n = FF;
                    void * args[] = { (void *)&b.mlp_inter, (void *)&lw.ptrs[down_idx], (void *)&lw.strides[down_idx],
                                      (void *)&b.hidden, (void *)&n, (void *)&H };
                    hipModuleLaunchKernel(fused_qm_down, H, 1, 1, 32, 4, 1, smem_size, s, args, nullptr);
                } else {
                    int n = FF, q8blocks = (n + 511) / 512;
                    void * q8args[] = { (void *)&b.mlp_inter, (void *)&b.q8_act, (void *)&n };
                    hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
                    launch_matvec(lw.types[down_idx], lw.ptrs[down_idx], lw.strides[down_idx], b.hidden, FF, H);
                }
                if (lw.ffn_down_bias)  launch_add(b.hidden, lw.ffn_down_bias, H);
                if (lw.ffn_down_scale) launch_mul(b.hidden, lw.ffn_down_scale, H);
                // Post-FFN norm: Gemma2/3/4 (ffn_post_norm) or Chameleon (swin_norm uses ptrs[7])
                const void * post_ffn_w = lw.ffn_post_norm ? lw.ffn_post_norm :
                                          (c.has_swin_norm ? lw.ptrs[7] : nullptr);
                // Granite residual scale
                if (c.f_residual_scale != 0.0f && c.f_residual_scale != 1.0f) {
                    float rs = c.f_residual_scale;
                    int n2 = H;
                    void * args[] = { (void *)&b.hidden, (void *)&b.hidden, (void *)&rs, (void *)&n2 };
                    hipModuleLaunchKernel(k.eval_scale_scalar, (n2+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
                }
                if (post_ffn_w) {
                    // Post-FFN norm: split into eval_rmsnorm_q8 + eval_add_residual
                    // to match baseline's rms_norm_f32<N, true, true> register pressure.
                    // Baseline fuses {RMS_NORM, MUL, ADD} into one kernel.
                    // Our split produces byte-exact results (verified for post-attn path).
                    {
                        int hn = H;
                        // Step 1: rmsnorm(hidden, post_ffn_w) → proj_scratch
                        void * norm_args[] = { (void *)&b.hidden, (void *)&post_ffn_w,
                                               (void *)&b.proj_scratch, (void *)&b.hidden, (void *)&hn };
                        hipModuleLaunchKernel(k.eval_rmsnorm_q8, 1, 1, 1, norm_threads, 1, 1, 0, s, norm_args, nullptr);
                        // Step 2: hidden = proj_scratch + residual
                        void * add_args[] = { (void *)&b.proj_scratch, (void *)&b.residual, (void *)&b.hidden, (void *)&hn };
                        hipModuleLaunchKernel(k.eval_add_residual, (hn+255)/256, 1, 1, 256, 1, 1, 0, s, add_args, nullptr);
                        // Step 3: update residual for next layer
                        hipMemcpyAsync(b.residual, b.hidden, H * sizeof(float), hipMemcpyDeviceToDevice, s);
                    }
                } else {
                    launch_add(b.hidden, (const void *)b.residual, H);
                }
            } else {
                // Sequential: try fused quantize+matvec+residual (1 launch instead of 3)
                if (fused_qmr_down) {
                    int smem_size = (FF / 32) * 36;
                    int n = FF;
                    void * args[] = { (void *)&b.mlp_inter, (void *)&lw.ptrs[down_idx], (void *)&lw.strides[down_idx],
                                      (void *)&b.residual, (void *)&b.hidden, (void *)&n, (void *)&H };
                    hipModuleLaunchKernel(fused_qmr_down, H, 1, 1, 32, 4, 1, smem_size, s, args, nullptr);
                } else {
                    // Fallback: quantize + matvec_res (2 launches)
                    int n = FF, q8blocks = (n + 511) / 512;
                    void * q8args[] = { (void *)&b.mlp_inter, (void *)&b.q8_act, (void *)&n };
                    hipModuleLaunchKernel(k.eval_quantize_q8, q8blocks, 1, 1, 512, 1, 1, 0, s, q8args, nullptr);
                    launch_matvec_res(lw.types[down_idx], lw.ptrs[down_idx], lw.strides[down_idx], b.residual, b.hidden, FF, H);
                }
            }
        }

        PROF_RECORD_LAYER(il, 8);  // boundary 8: end of layer (after ffn_res)
        if (il == 0) dump_gpu("L0_end", b.hidden, H);

        // Dump hidden state after every layer for divergence bisection
        if (dump_enabled && dump_token_count == dump_target_pos) {
            hipStreamSynchronize(s);
            std::vector<float> h(H);
            hipMemcpy(h.data(), b.hidden, H * sizeof(float), hipMemcpyDeviceToHost);
            double sum = 0, sumsq = 0;
            for (int i = 0; i < H; i++) { sum += h[i]; sumsq += h[i]*h[i]; }
            fprintf(stderr, "  LAYER_END L%d: mean=%.8f rms=%.8f first4=[%.6f %.6f %.6f %.6f]\n",
                    il, sum/H, sqrt(sumsq/H), h[0], h[1], h[2], h[3]);
            char path[256];
            snprintf(path, sizeof(path), "dump_mk_L%d_end_%d.bin", il, position);
            FILE * f = fopen(path, "wb");
            if (f) { fwrite(h.data(), sizeof(float), H, f); fclose(f); }
        }

        layer_done:;
    }

    // ---- Final norm ----
    PROF_RECORD_LM(0);
    // Baseline norm.cu: block_size = (ncols < 1024) ? 256 : 1024
    {
        int n = H;
        int norm_threads = (H < 1024) ? 256 : 1024;
        if (c.norm_type == 2) {
            // LayerNorm final norm (BLOOM, GPT2, Falcon, etc.)
            float eps = c.norm_eps;
            void * args[] = { (void *)&b.hidden, (void *)&c.final_norm_weight,
                              (void *)&c.final_norm_bias, (void *)&b.norm_out,
                              (void *)&b.hidden, (void *)&n, (void *)&eps };
            hipModuleLaunchKernel(k.eval_layernorm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        } else {
            void * args[] = { (void *)&b.hidden, (void *)&c.final_norm_weight,
                              (void *)&b.norm_out, (void *)&n };
            hipModuleLaunchKernel(k.eval_final_norm, 1, 1, 1, norm_threads, 1, 1, 0, s, args, nullptr);
        }
    }

    // ---- Quantize for LM head ----
    {
        int n = H;
        int blocks = (n + 511) / 512;
        void * args[] = { (void *)&b.norm_out, (void *)&b.q8_act, (void *)&n };
        hipModuleLaunchKernel(k.eval_quantize_q8, blocks, 1, 1, 512, 1, 1, 0, s, args, nullptr);
    }

    // ---- LM head (matvec over vocab — type-dispatched, handles F16/quantized) ----
    launch_matvec(c.lm_head_type, c.lm_head_weight, c.lm_head_stride, b.logits, H, V);

    // ---- Logit scale (Cohere2, Command-R, Granite, Grok) ----
    // Baseline: ggml_scale(cur, f_logit_scale) or ggml_scale(cur, 1/f_logit_scale)
    // The caller sets f_logit_scale to the appropriate value (or reciprocal for Granite)
    if (c.f_logit_scale != 0.0f && c.f_logit_scale != 1.0f) {
        float ls = c.f_logit_scale;
        int n = V;
        void * args[] = { (void *)&b.logits, (void *)&b.logits, (void *)&ls, (void *)&n };
        hipModuleLaunchKernel(k.eval_scale_scalar, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    }

    // ---- Final logit softcap (Gemma2) ----
    // Baseline gemma2-iswa.cpp lines 120-122: tanh(logits/cap) * cap
    // Ported from baseline: ggml_scale(1/cap) → ggml_tanh → ggml_scale(cap)
    if (c.has_final_logit_softcap && c.final_logit_softcap_val > 0.0f) {
        // Fused: logits[i] = tanh(logits[i] / cap) * cap — one kernel instead of three
        float cap = c.final_logit_softcap_val;
        int n = V;
        void * args[] = { (void *)&b.logits, (void *)&cap, (void *)&n };
        hipModuleLaunchKernel(k.eval_softcap, (n+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    }

    // Chameleon: suppress image token logits — baseline chameleon.cpp lines 161-172
    // Set logits[4..4+8192-1] to -FLT_MAX so image tokens can never be sampled
    if (c.chameleon_img_token_count > 0) {
        int start = c.chameleon_img_token_start;
        int count = c.chameleon_img_token_count;
        if (start + count <= V) {
            void * args[] = { (void *)&b.logits, (void *)&start, (void *)&count };
            hipModuleLaunchKernel(k.eval_chameleon_suppress, (count+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
        }
    }

    PROF_RECORD_LM(1);

    // ---- hipGraph: end capture and instantiate ----
    if (capturing_graph) {
        hipGraph_t graph = nullptr;
        hipStreamEndCapture(s, &graph);
        if (graph) {
            hipGraphInstantiate(&g_graph_exec, graph, nullptr, nullptr, 0);
            hipGraphDestroy(graph);
            // Launch the newly instantiated graph
            hipGraphLaunch(g_graph_exec, s);
            fprintf(stderr, "gfx1100-megakernel: hipGraph captured and instantiated (token %d)\n", position);
        } else {
            fprintf(stderr, "gfx1100-megakernel: WARNING — hipGraph capture failed, falling back to non-graph\n");
            use_graph = false;
        }
    }

    // Copy logits to host
    if (logits_out) {
        hipMemcpyAsync(logits_out, b.logits, V * sizeof(float),
                       hipMemcpyDeviceToHost, s);
    }
    hipStreamSynchronize(s);

    // ---- Profiling: compute and print per-phase breakdown ----
    if (profile_enabled) {
        // Embed phase: ev_embed[0] → ev_embed[1]
        float ms_tmp = 0.0f;
        hipEventElapsedTime(&ms_tmp, ev_embed[0], ev_embed[1]);
        prof_ms[PROF_EMBED] = ms_tmp;

        // Per-layer phases: accumulate across all layers
        // Layer boundary layout (9 boundaries per layer):
        //   0 = start of layer (before norm)
        //   1 = after norm+quantize
        //   2 = after QKV proj
        //   3 = after rope_kv
        //   4 = after attn
        //   5 = after o_proj
        //   6 = after ffn_norm+quantize
        //   7 = after ffn_proj (gate+up+act)
        //   8 = after ffn_res (down+residual) = end of layer
        for (int il = 0; il < debug_max_layers; il++) {
            int base = il * 9;
            float dt = 0.0f;

            // norm:     boundary 0 → 1
            hipEventElapsedTime(&dt, ev_layer[base + 0], ev_layer[base + 1]);
            prof_ms[PROF_NORM] += dt;

            // qkv_proj: boundary 1 → 2
            hipEventElapsedTime(&dt, ev_layer[base + 1], ev_layer[base + 2]);
            prof_ms[PROF_QKV_PROJ] += dt;

            // rope_kv:  boundary 2 → 3
            hipEventElapsedTime(&dt, ev_layer[base + 2], ev_layer[base + 3]);
            prof_ms[PROF_ROPE_KV] += dt;

            // attn:     boundary 3 → 4
            hipEventElapsedTime(&dt, ev_layer[base + 3], ev_layer[base + 4]);
            prof_ms[PROF_ATTN] += dt;

            // o_proj:   boundary 4 → 5
            hipEventElapsedTime(&dt, ev_layer[base + 4], ev_layer[base + 5]);
            prof_ms[PROF_O_PROJ] += dt;

            // ffn_norm: boundary 5 → 6
            hipEventElapsedTime(&dt, ev_layer[base + 5], ev_layer[base + 6]);
            prof_ms[PROF_FFN_NORM] += dt;

            // ffn_proj: boundary 6 → 7
            hipEventElapsedTime(&dt, ev_layer[base + 6], ev_layer[base + 7]);
            prof_ms[PROF_FFN_PROJ] += dt;

            // ffn_res:  boundary 7 → 8
            hipEventElapsedTime(&dt, ev_layer[base + 7], ev_layer[base + 8]);
            prof_ms[PROF_FFN_RES] += dt;
        }

        // LM head: ev_lm[0] → ev_lm[1]
        hipEventElapsedTime(&ms_tmp, ev_lm[0], ev_lm[1]);
        prof_ms[PROF_LM_HEAD] = ms_tmp;

        // Compute total
        float total = 0.0f;
        for (int i = 0; i < PROF_N_PHASES; i++) total += prof_ms[i];

        // Print breakdown
        fprintf(stderr, "\ngfx1100 PROFILE (arch=%d, %dL, H=%d, FF=%d, 1 token):\n",
                c.arch_id, c.n_layers, H, FF);
        for (int i = 0; i < PROF_N_PHASES; i++) {
            float pct = (total > 0.0f) ? (prof_ms[i] / total * 100.0f) : 0.0f;
            fprintf(stderr, "  %-10s %7.2f ms (%5.1f%%)\n", gfx1100_profile_phase_names[i], prof_ms[i], pct);
        }
        fprintf(stderr, "  %-10s %7.2f ms\n", "TOTAL", total);

        // Cleanup events
        hipEventDestroy(ev_embed[0]);
        hipEventDestroy(ev_embed[1]);
        hipEventDestroy(ev_lm[0]);
        hipEventDestroy(ev_lm[1]);
        for (int i = 0; i < c.n_layers * 9; i++) {
            hipEventDestroy(ev_layer[i]);
        }
    }

    #undef PROF_RECORD_EMBED
    #undef PROF_RECORD_LM
    #undef PROF_RECORD_LAYER

    if (dump_enabled) dump_token_count++;
    return 0;
}
