// test-kernel-simple-ops.cpp — Unit tests for eval_add_residual, eval_silu_mul, eval_final_norm
//
// Purely synthetic — no model required. Creates known f32 arrays on host,
// launches kernels via HIP module API, compares GPU output against CPU reference.
//
// Usage: test-kernel-simple-ops
// Requires: decode.hip_*.hsaco in ~/.cache/gfx1100-megakernel/

#include "test-harness.h"

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------
static float cpu_silu(float x) {
    return x / (1.0f + expf(-x));
}

static void cpu_add_residual(const float * a, const float * b, float * out, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

static void cpu_silu_mul(const float * gate, const float * up, float * out, int n) {
    for (int i = 0; i < n; i++) out[i] = cpu_silu(gate[i]) * up[i];
}

static void cpu_final_norm(const float * input, const float * weight, float * out,
                           int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)input[i] * (double)input[i];
    float rstd = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = input[i] * rstd * weight[i];
}

// ---------------------------------------------------------------------------
// Test: eval_add_residual
// Grid: (n+255)/256, Block: 256
// Signature: (const float* a, const float* b, float* output, int n)
// ---------------------------------------------------------------------------
static void test_add_residual(hipModule_t mod) {
    const char * name = "eval_add_residual";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found in .hsaco"); return; }

    const int n = 2048;
    std::vector<float> h_a(n), h_b(n), h_out(n), h_cpu(n);

    for (int i = 0; i < n; i++) {
        h_a[i] = sinf((float)i * 0.1f) * 5.0f;
        h_b[i] = cosf((float)i * 0.07f) * 3.0f;
    }
    cpu_add_residual(h_a.data(), h_b.data(), h_cpu.data(), n);

    GpuBuf g_a(n * sizeof(float)), g_b(n * sizeof(float)), g_out(n * sizeof(float));
    g_a.upload(h_a.data(), n * sizeof(float));
    g_b.upload(h_b.data(), n * sizeof(float));

    void * a_ptr = g_a.ptr, * b_ptr = g_b.ptr, * o_ptr = g_out.ptr;
    int n_val = n;
    void * args[] = { &a_ptr, &b_ptr, &o_ptr, &n_val };
    unsigned grid = (n + 255) / 256;

    if (!launch_kernel(fn, grid, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }

    g_out.download(h_out.data(), n * sizeof(float));
    CompareResult r = compare_float(h_out.data(), h_cpu.data(), n);

    if (r.nan_count > 0) { test_fail(name, "%d NaN values", r.nan_count); return; }
    if (r.max_abs > 1e-5f) { test_fail(name, "max_abs=%.2e > 1e-5", (double)r.max_abs); return; }
    test_pass(name, "max_abs=%.2e, max_rel=%.2e", (double)r.max_abs, (double)r.max_rel);
}

// ---------------------------------------------------------------------------
// Test: eval_silu_mul
// Grid: (n+255)/256, Block: 256
// Signature: (const float* gate, const float* up, float* output, int n)
// ---------------------------------------------------------------------------
static void test_silu_mul(hipModule_t mod) {
    const char * name = "eval_silu_mul";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found in .hsaco"); return; }

    const int n = 8192;
    std::vector<float> h_gate(n), h_up(n), h_out(n), h_cpu(n);

    for (int i = 0; i < n; i++) {
        h_gate[i] = sinf((float)i * 0.05f) * 4.0f;  // range [-4, 4]
        h_up[i]   = cosf((float)i * 0.03f) * 2.0f;   // range [-2, 2]
    }
    cpu_silu_mul(h_gate.data(), h_up.data(), h_cpu.data(), n);

    GpuBuf g_gate(n * sizeof(float)), g_up(n * sizeof(float)), g_out(n * sizeof(float));
    g_gate.upload(h_gate.data(), n * sizeof(float));
    g_up.upload(h_up.data(), n * sizeof(float));

    void * gate_ptr = g_gate.ptr, * up_ptr = g_up.ptr, * out_ptr = g_out.ptr;
    int n_val = n;
    void * args[] = { &gate_ptr, &up_ptr, &out_ptr, &n_val };
    unsigned grid = (n + 255) / 256;

    if (!launch_kernel(fn, grid, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }

    g_out.download(h_out.data(), n * sizeof(float));
    CompareResult r = compare_float(h_out.data(), h_cpu.data(), n);

    if (r.nan_count > 0) { test_fail(name, "%d NaN values", r.nan_count); return; }
    // SiLU uses expf — allow slightly larger error from GPU exp implementation
    if (r.max_rel > 1e-4f) { test_fail(name, "max_rel=%.2e > 1e-4", (double)r.max_rel); return; }
    test_pass(name, "max_abs=%.2e, max_rel=%.2e", (double)r.max_abs, (double)r.max_rel);
}

// ---------------------------------------------------------------------------
// Test: eval_final_norm
// Grid: 1, Block: 256  (single block, shared-memory reduction)
// Signature: (const float* input, const float* weight, float* output, int n)
// NOTE: Uses compile-time NORM_EPS (default 1e-5f)
// ---------------------------------------------------------------------------
static void test_final_norm(hipModule_t mod) {
    const char * name = "eval_final_norm";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found in .hsaco"); return; }

    // Use HIDDEN_SIZE=2048 to match default Llama 1B .hsaco
    // The kernel takes n as a parameter but uses blockDim.x for the reduction,
    // so n must not exceed what 256 threads can handle (they loop: i += blockDim.x)
    const int n = 2048;
    const float eps = 1e-5f;  // must match NORM_EPS baked into .hsaco

    std::vector<float> h_input(n), h_weight(n), h_out(n), h_cpu(n);

    for (int i = 0; i < n; i++) {
        h_input[i]  = sinf((float)i * 0.1f) * 2.0f;
        h_weight[i] = 0.5f + 0.5f * cosf((float)i * 0.02f);  // positive weights ~[0,1]
    }
    cpu_final_norm(h_input.data(), h_weight.data(), h_cpu.data(), n, eps);

    GpuBuf g_in(n * sizeof(float)), g_w(n * sizeof(float)), g_out(n * sizeof(float));
    g_in.upload(h_input.data(), n * sizeof(float));
    g_w.upload(h_weight.data(), n * sizeof(float));

    void * in_ptr = g_in.ptr, * w_ptr = g_w.ptr, * out_ptr = g_out.ptr;
    int n_val = n;
    void * args[] = { &in_ptr, &w_ptr, &out_ptr, &n_val };

    // Single block, 256 threads
    if (!launch_kernel(fn, 1, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }

    g_out.download(h_out.data(), n * sizeof(float));
    CompareResult r = compare_float(h_out.data(), h_cpu.data(), n);

    if (r.nan_count > 0) { test_fail(name, "%d NaN values", r.nan_count); return; }
    if (r.max_rel > 1e-3f) { test_fail(name, "max_rel=%.2e > 1e-3", (double)r.max_rel); return; }
    test_pass(name, "max_abs=%.2e, max_rel=%.2e", (double)r.max_abs, (double)r.max_rel);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    // final_norm depends on HIDDEN_SIZE=2048 → FA_HEAD_DIM=64 .hsaco
    std::string hsaco_path = find_hsaco("decode.hip_", 64);
    if (hsaco_path.empty()) {
        fprintf(stderr, "FAIL: decode.hip_*.hsaco with FA_HEAD_DIM=64 not found\n");
        return 1;
    }
    fprintf(stderr, "Using hsaco: %s\n\n", hsaco_path.c_str());

    hipModule_t mod = nullptr;
    hipError_t e = hipModuleLoad(&mod, hsaco_path.c_str());
    if (e != hipSuccess) {
        fprintf(stderr, "FAIL: hipModuleLoad — %s\n", hipGetErrorString(e));
        return 1;
    }

    test_add_residual(mod);
    test_silu_mul(mod);
    test_final_norm(mod);

    hipModuleUnload(mod);
    return test_summary();
}
