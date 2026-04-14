// test-kernel-matvec-float.cpp — Unit tests for eval_matvec_f16, _bf16, _f32 + residual
//
// Purely synthetic. Creates known weight matrices in F16/BF16/F32 format,
// known f32 input, runs GPU matvec, compares against CPU dot product.
//
// Usage: test-kernel-matvec-float
// Requires: decode.hip_*.hsaco in ~/.cache/gfx1100-megakernel/

#include "test-harness.h"

// ---------------------------------------------------------------------------
// CPU matvec reference for F16 weights × F32 input
// ---------------------------------------------------------------------------
static void cpu_matvec_f16(const uint16_t * weight, long long stride_bytes,
                           const float * input, float * output,
                           int in_dim, int out_dim) {
    for (int row = 0; row < out_dim; row++) {
        const uint16_t * w = (const uint16_t *)((const char *)weight + (long long)row * stride_bytes);
        double acc = 0.0;
        for (int i = 0; i < in_dim; i++) acc += (double)f16_to_f32(w[i]) * (double)input[i];
        output[row] = (float)acc;
    }
}

// ---------------------------------------------------------------------------
// CPU matvec reference for BF16 weights × F32 input
// ---------------------------------------------------------------------------
static void cpu_matvec_bf16(const uint16_t * weight, long long stride_bytes,
                            const float * input, float * output,
                            int in_dim, int out_dim) {
    for (int row = 0; row < out_dim; row++) {
        const uint16_t * w = (const uint16_t *)((const char *)weight + (long long)row * stride_bytes);
        double acc = 0.0;
        for (int i = 0; i < in_dim; i++) acc += (double)bf16_to_f32(w[i]) * (double)input[i];
        output[row] = (float)acc;
    }
}

// ---------------------------------------------------------------------------
// CPU matvec reference for F32 weights × F32 input
// ---------------------------------------------------------------------------
static void cpu_matvec_f32(const float * weight, long long stride_bytes,
                           const float * input, float * output,
                           int in_dim, int out_dim) {
    for (int row = 0; row < out_dim; row++) {
        const float * w = (const float *)((const char *)weight + (long long)row * stride_bytes);
        double acc = 0.0;
        for (int i = 0; i < in_dim; i++) acc += (double)w[i] * (double)input[i];
        output[row] = (float)acc;
    }
}

// ---------------------------------------------------------------------------
// Test helper: run a float-type matvec and check results
// ---------------------------------------------------------------------------
static void run_matvec_test(hipModule_t mod, const char * kernel_name,
                            int elem_size, // 2 for f16/bf16, 4 for f32
                            void (* cpu_ref)(const void *, long long, const float *, float *, int, int),
                            bool is_residual) {
    hipFunction_t fn = load_kernel(mod, kernel_name);
    if (!fn) { test_fail(kernel_name, "kernel not found"); return; }

    const int in_dim  = 512;   // small but exercises reduction
    const int out_dim = 64;
    const long long stride = (long long)in_dim * elem_size;
    const size_t weight_bytes = (size_t)out_dim * stride;

    // Create host data
    std::vector<uint8_t> h_weight(weight_bytes);
    std::vector<float>   h_input(in_dim);
    std::vector<float>   h_residual(out_dim);
    std::vector<float>   h_gpu_out(out_dim);
    std::vector<float>   h_cpu_out(out_dim);

    // Synthetic input
    for (int i = 0; i < in_dim; i++) h_input[i] = sinf((float)i * 0.1f);
    for (int i = 0; i < out_dim; i++) h_residual[i] = cosf((float)i * 0.3f) * 2.0f;

    // Synthetic weight matrix — fill with known values then cast to target type
    if (elem_size == 2 && (strstr(kernel_name, "bf16") != nullptr)) {
        // BF16
        uint16_t * w = (uint16_t *)h_weight.data();
        for (int r = 0; r < out_dim; r++)
            for (int c = 0; c < in_dim; c++)
                w[r * in_dim + c] = f32_to_bf16(sinf((float)(r * in_dim + c) * 0.01f) * 0.5f);
    } else if (elem_size == 2) {
        // F16
        uint16_t * w = (uint16_t *)h_weight.data();
        for (int r = 0; r < out_dim; r++)
            for (int c = 0; c < in_dim; c++)
                w[r * in_dim + c] = f32_to_f16(sinf((float)(r * in_dim + c) * 0.01f) * 0.5f);
    } else {
        // F32
        float * w = (float *)h_weight.data();
        for (int r = 0; r < out_dim; r++)
            for (int c = 0; c < in_dim; c++)
                w[r * in_dim + c] = sinf((float)(r * in_dim + c) * 0.01f) * 0.5f;
    }

    // CPU reference
    cpu_ref(h_weight.data(), stride, h_input.data(), h_cpu_out.data(), in_dim, out_dim);
    if (is_residual) {
        for (int i = 0; i < out_dim; i++) h_cpu_out[i] += h_residual[i];
    }

    // GPU buffers
    GpuBuf g_weight(weight_bytes);
    GpuBuf g_input(in_dim * sizeof(float));
    GpuBuf g_residual(out_dim * sizeof(float));
    GpuBuf g_output(out_dim * sizeof(float));

    g_weight.upload(h_weight.data(), weight_bytes);
    g_input.upload(h_input.data(), in_dim * sizeof(float));
    g_residual.upload(h_residual.data(), out_dim * sizeof(float));

    // Launch
    void * w_ptr = g_weight.ptr;
    long long s_val = stride;
    void * in_ptr = g_input.ptr;
    void * res_ptr = g_residual.ptr;
    void * out_ptr = g_output.ptr;
    int in_val = in_dim, out_val = out_dim;

    bool ok;
    if (is_residual) {
        void * args[] = { &w_ptr, &s_val, &in_ptr, &res_ptr, &out_ptr, &in_val, &out_val };
        ok = launch_kernel(fn, (unsigned)out_dim, 1, 1, 256, 1, 1, args);
    } else {
        void * args[] = { &w_ptr, &s_val, &in_ptr, &out_ptr, &in_val, &out_val };
        ok = launch_kernel(fn, (unsigned)out_dim, 1, 1, 256, 1, 1, args);
    }
    if (!ok) { test_fail(kernel_name, "launch failed"); return; }

    g_output.download(h_gpu_out.data(), out_dim * sizeof(float));
    CompareResult r = compare_float(h_gpu_out.data(), h_cpu_out.data(), out_dim);

    if (r.nan_count > 0) { test_fail(kernel_name, "%d NaN values", r.nan_count); return; }
    // F16/BF16 precision: allow ~1% relative error
    float tol = (elem_size == 2) ? 0.02f : 1e-4f;
    if (r.max_rel > tol) {
        test_fail(kernel_name, "max_rel=%.2e > %.2e", (double)r.max_rel, (double)tol);
        return;
    }
    test_pass(kernel_name, "max_abs=%.2e, max_rel=%.2e", (double)r.max_abs, (double)r.max_rel);
}

// ---------------------------------------------------------------------------
// Wrapper CPU ref functions (cast void* back to typed pointer)
// ---------------------------------------------------------------------------
static void ref_f16(const void * w, long long s, const float * in, float * out, int id, int od) {
    cpu_matvec_f16((const uint16_t *)w, s, in, out, id, od);
}
static void ref_bf16(const void * w, long long s, const float * in, float * out, int id, int od) {
    cpu_matvec_bf16((const uint16_t *)w, s, in, out, id, od);
}
static void ref_f32(const void * w, long long s, const float * in, float * out, int id, int od) {
    cpu_matvec_f32((const float *)w, s, in, out, id, od);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::string hsaco_path = find_hsaco("decode.hip_", 64);
    if (hsaco_path.empty()) {
        fprintf(stderr, "FAIL: decode.hip_*.hsaco not found\n");
        return 1;
    }
    fprintf(stderr, "Using hsaco: %s\n\n", hsaco_path.c_str());

    hipModule_t mod = nullptr;
    hipError_t e = hipModuleLoad(&mod, hsaco_path.c_str());
    if (e != hipSuccess) {
        fprintf(stderr, "FAIL: hipModuleLoad — %s\n", hipGetErrorString(e));
        return 1;
    }

    // F16 matvec + residual
    run_matvec_test(mod, "eval_matvec_f16",          2, ref_f16,  false);
    run_matvec_test(mod, "eval_matvec_f16_residual",  2, ref_f16,  true);

    // BF16 matvec + residual
    run_matvec_test(mod, "eval_matvec_bf16",          2, ref_bf16, false);
    run_matvec_test(mod, "eval_matvec_bf16_residual",  2, ref_bf16, true);

    // F32 matvec + residual
    run_matvec_test(mod, "eval_matvec_f32",          4, ref_f32,  false);
    run_matvec_test(mod, "eval_matvec_f32_residual",  4, ref_f32,  true);

    hipModuleUnload(mod);
    return test_summary();
}
