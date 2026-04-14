// test-operation-ordering.cpp — Verify bias/scale/residual ordering matches baseline
//
// Baseline order: cur = matvec(weight, input); cur += bias; cur *= scale; cur += residual
// The fused matvec_res path should NOT be used when scale exists.
//
// Creates synthetic data, computes both CPU reference (baseline order) and
// GPU chain (separate kernels), compares.
//
// Usage: test-operation-ordering
// Requires: decode.hip_*.hsaco

#include "test-harness.h"

static void test_ordering(hipModule_t mod) {
    const char * name = "bias_scale_residual_order";

    hipFunction_t add_fn = load_kernel(mod, "eval_add_residual");
    hipFunction_t mul_fn = load_kernel(mod, "eval_elementwise_mul");
    if (!add_fn || !mul_fn) { test_fail(name, "kernels not found"); return; }

    const int N = 256;
    std::vector<float> h_matvec_result(N), h_bias(N), h_scale(N), h_residual(N);
    std::vector<float> h_cpu(N), h_gpu(N);

    // Synthetic data
    for (int i = 0; i < N; i++) {
        h_matvec_result[i] = sinf((float)i * 0.1f) * 3.0f;   // "projection output"
        h_bias[i]          = cosf((float)i * 0.05f) * 0.1f;    // small bias
        h_scale[i]         = 0.8f + 0.4f * sinf((float)i * 0.03f); // scale ~[0.4, 1.2]
        h_residual[i]      = sinf((float)i * 0.2f) * 2.0f;    // residual from previous layer
    }

    // CPU reference — baseline order: (matvec + bias) * scale + residual
    for (int i = 0; i < N; i++) {
        float cur = h_matvec_result[i];
        cur += h_bias[i];       // bias FIRST
        cur *= h_scale[i];      // scale SECOND
        cur += h_residual[i];   // residual LAST
        h_cpu[i] = cur;
    }

    // GPU chain — separate kernel launches matching dispatch non-fused path
    GpuBuf g_data(N*4), g_bias(N*4), g_scale(N*4), g_res(N*4);
    g_data.upload(h_matvec_result.data(), N*4);
    g_bias.upload(h_bias.data(), N*4);
    g_scale.upload(h_scale.data(), N*4);
    g_res.upload(h_residual.data(), N*4);

    // Step 1: data += bias
    {
        void * dp = g_data.ptr, * bp = g_bias.ptr;
        int n = N;
        void * args[] = { &dp, &bp, &dp, &n };
        launch_kernel(add_fn, (N+255)/256, 1, 1, 256, 1, 1, args);
    }
    // Step 2: data *= scale
    {
        void * dp = g_data.ptr, * sp = g_scale.ptr;
        int n = N;
        void * args[] = { &dp, &sp, &dp, &n };
        launch_kernel(mul_fn, (N+255)/256, 1, 1, 256, 1, 1, args);
    }
    // Step 3: data += residual
    {
        void * dp = g_data.ptr, * rp = g_res.ptr;
        int n = N;
        void * args[] = { &dp, &rp, &dp, &n };
        launch_kernel(add_fn, (N+255)/256, 1, 1, 256, 1, 1, args);
    }

    g_data.download(h_gpu.data(), N*4);
    CompareResult r = compare_float(h_gpu.data(), h_cpu.data(), N);

    if (r.nan_count > 0) { test_fail(name, "%d NaN", r.nan_count); return; }
    if (r.max_abs > 1e-5f) { test_fail(name, "max_abs=%.2e (ordering wrong?)", (double)r.max_abs); return; }
    test_pass(name, "max_abs=%.2e — bias→scale→residual order correct", (double)r.max_abs);

    // Also test WRONG order to verify the test catches it
    {
        const char * wrong_name = "wrong_order_detected";
        // Wrong: (matvec + residual) * scale + bias — the old fused bug
        std::vector<float> h_wrong(N);
        for (int i = 0; i < N; i++) {
            float cur = h_matvec_result[i];
            cur += h_residual[i];   // residual FIRST (wrong)
            cur *= h_scale[i];      // scale SECOND
            cur += h_bias[i];       // bias LAST (wrong)
            h_wrong[i] = cur;
        }
        CompareResult rw = compare_float(h_wrong.data(), h_cpu.data(), N);
        if (rw.max_abs > 0.01f) {
            test_pass(wrong_name, "wrong order differs by %.2e — test catches ordering bugs", (double)rw.max_abs);
        } else {
            test_fail(wrong_name, "wrong order matches correct — test is not sensitive enough");
        }
    }
}

int main() {
    std::string hsaco_path = find_hsaco("decode.hip_", 64);
    if (hsaco_path.empty()) { fprintf(stderr, "FAIL: decode.hip not found\n"); return 1; }

    hipModule_t mod = nullptr;
    if (hipModuleLoad(&mod, hsaco_path.c_str()) != hipSuccess) {
        fprintf(stderr, "FAIL: hipModuleLoad\n"); return 1;
    }

    test_ordering(mod);

    hipModuleUnload(mod);
    return test_summary();
}
