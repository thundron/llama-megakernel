// test-kernel-prompt-ops.cpp — Unit tests for prompt batch kernels
//
// Tests: prompt_add_residual, prompt_silu_mul, prompt_rmsnorm, prompt_final_norm,
//        prompt_embed_f32, prompt_embed_f16, prompt_embed_bf16
//
// All synthetic. Uses HIDDEN_SIZE=2048 (Llama 1B .hsaco default) where needed.
//
// Usage: test-kernel-prompt-ops
// Requires: prefill.hip_*.hsaco in ~/.cache/gfx1100-megakernel/

#include "test-harness.h"

static constexpr int HIDDEN = 2048;  // must match HIDDEN_SIZE in .hsaco
static constexpr float NORM_EPS = 1e-5f;

static float cpu_silu(float x) { return x / (1.0f + expf(-x)); }

// ---------------------------------------------------------------------------
// Test: prompt_add_residual
// Grid: ((N+255)/256,1,1), Block: (256,1,1)
// Signature: (const float* a, const float* b, float* output, int N)
// ---------------------------------------------------------------------------
static void test_add_residual(hipModule_t mod) {
    const char * name = "prompt_add_residual";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int N = 4096;  // e.g. seq_len=2 * hidden=2048
    std::vector<float> h_a(N), h_b(N), h_cpu(N), h_gpu(N);
    for (int i = 0; i < N; i++) { h_a[i] = sinf((float)i * 0.1f); h_b[i] = cosf((float)i * 0.07f); h_cpu[i] = h_a[i] + h_b[i]; }

    GpuBuf g_a(N * 4), g_b(N * 4), g_out(N * 4);
    g_a.upload(h_a.data(), N * 4); g_b.upload(h_b.data(), N * 4);

    void * ap = g_a.ptr, * bp = g_b.ptr, * op = g_out.ptr; int n = N;
    void * args[] = { &ap, &bp, &op, &n };
    if (!launch_kernel(fn, (N + 255) / 256, 1, 1, 256, 1, 1, args)) { test_fail(name, "launch failed"); return; }

    g_out.download(h_gpu.data(), N * 4);
    CompareResult r = compare_float(h_gpu.data(), h_cpu.data(), N);
    if (r.nan_count > 0 || r.max_abs > 1e-5f) { test_fail(name, "nan=%d max_abs=%.2e", r.nan_count, (double)r.max_abs); return; }
    test_pass(name, "max_abs=%.2e", (double)r.max_abs);
}

// ---------------------------------------------------------------------------
// Test: prompt_silu_mul
// ---------------------------------------------------------------------------
static void test_silu_mul(hipModule_t mod) {
    const char * name = "prompt_silu_mul";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int N = 4096;
    std::vector<float> h_g(N), h_u(N), h_cpu(N), h_gpu(N);
    for (int i = 0; i < N; i++) { h_g[i] = sinf((float)i * 0.05f) * 3.0f; h_u[i] = cosf((float)i * 0.03f) * 2.0f; h_cpu[i] = cpu_silu(h_g[i]) * h_u[i]; }

    GpuBuf g_g(N * 4), g_u(N * 4), g_out(N * 4);
    g_g.upload(h_g.data(), N * 4); g_u.upload(h_u.data(), N * 4);

    void * gp = g_g.ptr, * up = g_u.ptr, * op = g_out.ptr; int n = N;
    void * args[] = { &gp, &up, &op, &n };
    if (!launch_kernel(fn, (N + 255) / 256, 1, 1, 256, 1, 1, args)) { test_fail(name, "launch failed"); return; }

    g_out.download(h_gpu.data(), N * 4);
    CompareResult r = compare_float(h_gpu.data(), h_cpu.data(), N);
    if (r.nan_count > 0 || r.max_rel > 1e-4f) { test_fail(name, "nan=%d max_rel=%.2e", r.nan_count, (double)r.max_rel); return; }
    test_pass(name, "max_abs=%.2e, max_rel=%.2e", (double)r.max_abs, (double)r.max_rel);
}

// ---------------------------------------------------------------------------
// Test: prompt_rmsnorm
// Grid: (S, 1, 1), Block: (256, 1, 1)
// Signature: (input[S,D], weight[D], output[S,D], residual[S,D], S, D)
// ---------------------------------------------------------------------------
static void test_rmsnorm(hipModule_t mod) {
    const char * name = "prompt_rmsnorm";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int S = 4, D = HIDDEN;
    std::vector<float> h_in(S * D), h_w(D), h_cpu_out(S * D), h_cpu_res(S * D);

    for (int i = 0; i < S * D; i++) h_in[i] = sinf((float)i * 0.01f) * 2.0f;
    for (int i = 0; i < D; i++) h_w[i] = 0.5f + 0.5f * cosf((float)i * 0.02f);

    // CPU reference
    for (int s = 0; s < S; s++) {
        const float * row = h_in.data() + s * D;
        float * res = h_cpu_res.data() + s * D;
        float * out = h_cpu_out.data() + s * D;
        double sq = 0;
        for (int i = 0; i < D; i++) { res[i] = row[i]; sq += (double)row[i] * (double)row[i]; }
        float rstd = 1.0f / sqrtf((float)(sq / D) + NORM_EPS);
        for (int i = 0; i < D; i++) out[i] = row[i] * rstd * h_w[i];
    }

    GpuBuf g_in(S * D * 4), g_w(D * 4), g_out(S * D * 4), g_res(S * D * 4);
    g_in.upload(h_in.data(), S * D * 4); g_w.upload(h_w.data(), D * 4);

    void * ip = g_in.ptr, * wp = g_w.ptr, * op = g_out.ptr, * rp = g_res.ptr;
    int sv = S, dv = D;
    void * args[] = { &ip, &wp, &op, &rp, &sv, &dv };
    if (!launch_kernel(fn, (unsigned)S, 1, 1, 256, 1, 1, args)) { test_fail(name, "launch failed"); return; }

    std::vector<float> h_gpu_out(S * D), h_gpu_res(S * D);
    g_out.download(h_gpu_out.data(), S * D * 4);
    g_res.download(h_gpu_res.data(), S * D * 4);

    CompareResult r_out = compare_float(h_gpu_out.data(), h_cpu_out.data(), S * D);
    CompareResult r_res = compare_float(h_gpu_res.data(), h_cpu_res.data(), S * D);

    if (r_out.nan_count > 0) { test_fail(name, "%d NaN in output", r_out.nan_count); return; }
    if (r_res.max_abs > 1e-5f) { test_fail(name, "residual mismatch: max_abs=%.2e", (double)r_res.max_abs); return; }
    if (r_out.max_rel > 1e-3f) { test_fail(name, "output max_rel=%.2e", (double)r_out.max_rel); return; }
    test_pass(name, "out_rel=%.2e, res_exact=%.2e", (double)r_out.max_rel, (double)r_res.max_abs);
}

// ---------------------------------------------------------------------------
// Test: prompt_final_norm
// Grid: (1, 1, 1), Block: (256, 1, 1)
// Signature: (hidden[S, HIDDEN_SIZE], weight[HIDDEN_SIZE], normed[HIDDEN_SIZE], S)
// Uses HIDDEN_SIZE compile-time constant (processes last token only)
// ---------------------------------------------------------------------------
static void test_final_norm(hipModule_t mod) {
    const char * name = "prompt_final_norm";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int S = 3;
    std::vector<float> h_hidden(S * HIDDEN), h_w(HIDDEN), h_cpu(HIDDEN);

    for (int i = 0; i < S * HIDDEN; i++) h_hidden[i] = sinf((float)i * 0.005f) * 2.0f;
    for (int i = 0; i < HIDDEN; i++) h_w[i] = 0.5f + 0.5f * cosf((float)i * 0.02f);

    // CPU: process last row only
    const float * last = h_hidden.data() + (S - 1) * HIDDEN;
    double sq = 0;
    for (int i = 0; i < HIDDEN; i++) sq += (double)last[i] * (double)last[i];
    float rstd = 1.0f / sqrtf((float)(sq / HIDDEN) + NORM_EPS);
    for (int i = 0; i < HIDDEN; i++) h_cpu[i] = last[i] * rstd * h_w[i];

    GpuBuf g_h(S * HIDDEN * 4), g_w(HIDDEN * 4), g_out(HIDDEN * 4);
    g_h.upload(h_hidden.data(), S * HIDDEN * 4); g_w.upload(h_w.data(), HIDDEN * 4);

    void * hp = g_h.ptr, * wp = g_w.ptr, * op = g_out.ptr; int sv = S;
    void * args[] = { &hp, &wp, &op, &sv };
    if (!launch_kernel(fn, 1, 1, 1, 256, 1, 1, args)) { test_fail(name, "launch failed"); return; }

    std::vector<float> h_gpu(HIDDEN);
    g_out.download(h_gpu.data(), HIDDEN * 4);
    CompareResult r = compare_float(h_gpu.data(), h_cpu.data(), HIDDEN);

    if (r.nan_count > 0) { test_fail(name, "%d NaN", r.nan_count); return; }
    if (r.max_rel > 1e-3f) { test_fail(name, "max_rel=%.2e", (double)r.max_rel); return; }
    test_pass(name, "max_abs=%.2e, max_rel=%.2e", (double)r.max_abs, (double)r.max_rel);
}

// ---------------------------------------------------------------------------
// Test: prompt_embed_f32
// Grid: dim3((HIDDEN+255)/256, S, 1), Block: (256, 1, 1)
// Signature: (token_ids[S], embed_weight, embed_stride, output[S,HIDDEN], S)
// ---------------------------------------------------------------------------
static void test_embed_f32(hipModule_t mod) {
    const char * name = "prompt_embed_f32";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int S = 3, V = 4;
    const long long stride = (long long)HIDDEN * sizeof(float);

    std::vector<float> h_table(V * HIDDEN);
    for (int v = 0; v < V; v++)
        for (int d = 0; d < HIDDEN; d++)
            h_table[v * HIDDEN + d] = sinf((float)(v * HIDDEN + d) * 0.01f);

    int h_tokens[] = { 0, 2, 1 };
    std::vector<float> h_cpu(S * HIDDEN);
    for (int s = 0; s < S; s++)
        for (int d = 0; d < HIDDEN; d++)
            h_cpu[s * HIDDEN + d] = h_table[h_tokens[s] * HIDDEN + d];

    GpuBuf g_tok(S * sizeof(int));
    GpuBuf g_tbl(V * HIDDEN * sizeof(float));
    GpuBuf g_out(S * HIDDEN * sizeof(float));
    g_tok.upload(h_tokens, S * sizeof(int));
    g_tbl.upload(h_table.data(), V * HIDDEN * sizeof(float));

    void * tp = g_tok.ptr, * tb = g_tbl.ptr, * op = g_out.ptr;
    long long s_val = stride; int sv = S;
    void * args[] = { &tp, &tb, &s_val, &op, &sv };
    unsigned gx = (HIDDEN + 255) / 256;
    if (!launch_kernel(fn, gx, (unsigned)S, 1, 256, 1, 1, args)) { test_fail(name, "launch failed"); return; }

    std::vector<float> h_gpu(S * HIDDEN);
    g_out.download(h_gpu.data(), S * HIDDEN * sizeof(float));
    CompareResult r = compare_float(h_gpu.data(), h_cpu.data(), S * HIDDEN);

    if (r.nan_count > 0 || r.max_abs > 1e-6f) { test_fail(name, "nan=%d max_abs=%.2e", r.nan_count, (double)r.max_abs); return; }
    test_pass(name, "exact F32 copy, max_abs=%.2e", (double)r.max_abs);
}

// ---------------------------------------------------------------------------
// Test: prompt_embed_f16
// ---------------------------------------------------------------------------
static void test_embed_f16(hipModule_t mod) {
    const char * name = "prompt_embed_f16";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int S = 2, V = 4;
    const long long stride = (long long)HIDDEN * sizeof(uint16_t);

    std::vector<uint16_t> h_table(V * HIDDEN);
    for (int v = 0; v < V; v++)
        for (int d = 0; d < HIDDEN; d++)
            h_table[v * HIDDEN + d] = f32_to_f16(sinf((float)(v * HIDDEN + d) * 0.01f));

    int h_tokens[] = { 1, 3 };
    std::vector<float> h_cpu(S * HIDDEN);
    for (int s = 0; s < S; s++)
        for (int d = 0; d < HIDDEN; d++)
            h_cpu[s * HIDDEN + d] = f16_to_f32(h_table[h_tokens[s] * HIDDEN + d]);

    GpuBuf g_tok(S * sizeof(int));
    GpuBuf g_tbl(V * HIDDEN * sizeof(uint16_t));
    GpuBuf g_out(S * HIDDEN * sizeof(float));
    g_tok.upload(h_tokens, S * sizeof(int));
    g_tbl.upload(h_table.data(), V * HIDDEN * sizeof(uint16_t));

    void * tp = g_tok.ptr, * tb = g_tbl.ptr, * op = g_out.ptr;
    long long s_val = stride; int sv = S;
    void * args[] = { &tp, &tb, &s_val, &op, &sv };
    unsigned gx = (HIDDEN + 255) / 256;
    if (!launch_kernel(fn, gx, (unsigned)S, 1, 256, 1, 1, args)) { test_fail(name, "launch failed"); return; }

    std::vector<float> h_gpu(S * HIDDEN);
    g_out.download(h_gpu.data(), S * HIDDEN * sizeof(float));
    CompareResult r = compare_float(h_gpu.data(), h_cpu.data(), S * HIDDEN);

    if (r.nan_count > 0 || r.max_abs > 1e-3f) { test_fail(name, "nan=%d max_abs=%.2e", r.nan_count, (double)r.max_abs); return; }
    test_pass(name, "max_abs=%.2e", (double)r.max_abs);
}

int main() {
    std::string hsaco_path = find_hsaco("prefill.hip_", 64);
    if (hsaco_path.empty()) { fprintf(stderr, "FAIL: prefill.hip not found\n"); return 1; }
    fprintf(stderr, "Using hsaco: %s\n\n", hsaco_path.c_str());

    hipModule_t mod = nullptr;
    if (hipModuleLoad(&mod, hsaco_path.c_str()) != hipSuccess) {
        fprintf(stderr, "FAIL: hipModuleLoad\n"); return 1;
    }

    test_add_residual(mod);
    test_silu_mul(mod);
    test_rmsnorm(mod);
    test_final_norm(mod);
    test_embed_f32(mod);
    test_embed_f16(mod);

    hipModuleUnload(mod);
    return test_summary();
}
