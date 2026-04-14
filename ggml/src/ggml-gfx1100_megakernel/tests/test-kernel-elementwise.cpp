// test-kernel-elementwise.cpp — Unit tests for element-wise kernels
// Tests: eval_tanh, eval_neg, eval_exp, eval_relu, eval_sqr, eval_silu,
//        eval_sub, eval_muladd, eval_sigmoid, eval_gelu, eval_gelu_erf
// Requires: decode.hip_*.hsaco in ~/.cache/gfx1100-megakernel/
#include "test-harness.h"
#include <cmath>
#include <cstdlib>

// CPU reference functions
static float ref_tanh(float x)    { return tanhf(x); }
static float ref_neg(float x)     { return -x; }
static float ref_exp(float x)     { return expf(x); }
static float ref_relu(float x)    { return x > 0 ? x : 0; }
static float ref_sqr(float x)     { return x * x; }
static float ref_silu(float x)    { return x / (1.0f + expf(-x)); }
static float ref_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float ref_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
static float ref_gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x / 1.4142135624f));
}

static const int N = 1024;

struct unary_test {
    const char * kernel_name;
    float (*ref_fn)(float);
    float tolerance;
};

static void test_unary(hipModule_t mod, const unary_test & t) {
    hipFunction_t fn = load_kernel(mod, t.kernel_name);
    if (!fn) { test_fail(t.kernel_name, "kernel not found"); return; }

    // Generate random input
    std::vector<float> h_in(N), h_out(N), h_ref(N);
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(rand() % 2000 - 1000) / 200.0f;  // range [-5, 5]
        h_ref[i] = t.ref_fn(h_in[i]);
    }

    GpuBuf d_in(N * 4), d_out(N * 4);
    hipMemcpy(d_in.ptr, h_in.data(), N * 4, hipMemcpyHostToDevice);

    int n = N;
    void * args[] = { &d_in.ptr, &d_out.ptr, &n };
    if (!launch_kernel(fn, (N + 255) / 256, 1, 1, 256, 1, 1, args)) {
        test_fail(t.kernel_name, "launch failed"); return;
    }
    hipDeviceSynchronize();

    hipMemcpy(h_out.data(), d_out.ptr, N * 4, hipMemcpyDeviceToHost);

    float max_err = 0;
    for (int i = 0; i < N; i++) {
        float err = fabsf(h_out[i] - h_ref[i]);
        if (h_ref[i] != 0) err /= fabsf(h_ref[i]) + 1e-8f;  // relative error
        if (err > max_err) max_err = err;
    }
    if (max_err < t.tolerance) {
        test_pass(t.kernel_name, "max_err=%.6f", max_err);
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "max relative error %.6f > tolerance %.6f", max_err, t.tolerance);
        test_fail(t.kernel_name, msg);
    }
}

static void test_sub(hipModule_t mod) {
    const char * name = "eval_sub";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    std::vector<float> h_a(N), h_b(N), h_out(N), h_ref(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(rand() % 1000) / 100.0f;
        h_b[i] = (float)(rand() % 1000) / 100.0f;
        h_ref[i] = h_a[i] - h_b[i];
    }
    GpuBuf d_a(N * 4), d_b(N * 4), d_out(N * 4);
    hipMemcpy(d_a.ptr, h_a.data(), N * 4, hipMemcpyHostToDevice);
    hipMemcpy(d_b.ptr, h_b.data(), N * 4, hipMemcpyHostToDevice);

    int n = N;
    void * args[] = { &d_a.ptr, &d_b.ptr, &d_out.ptr, &n };
    if (!launch_kernel(fn, (N + 255) / 256, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }
    hipDeviceSynchronize();
    hipMemcpy(h_out.data(), d_out.ptr, N * 4, hipMemcpyDeviceToHost);

    float max_err = 0;
    for (int i = 0; i < N; i++) {
        float err = fabsf(h_out[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }
    if (max_err < 1e-5f) test_pass(name, "max_err=%.6f", max_err);
    else {
        char msg[128]; snprintf(msg, sizeof(msg), "max error %.8f", max_err);
        test_fail(name, msg);
    }
}

static void test_muladd(hipModule_t mod) {
    const char * name = "eval_muladd";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    std::vector<float> h_a(N), h_b(N), h_c(N), h_out(N), h_ref(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(rand() % 1000) / 200.0f;
        h_b[i] = (float)(rand() % 1000) / 200.0f;
        h_c[i] = (float)(rand() % 1000) / 200.0f;
        h_ref[i] = h_a[i] * h_b[i] + h_c[i];
    }
    GpuBuf d_a(N * 4), d_b(N * 4), d_c(N * 4), d_out(N * 4);
    hipMemcpy(d_a.ptr, h_a.data(), N * 4, hipMemcpyHostToDevice);
    hipMemcpy(d_b.ptr, h_b.data(), N * 4, hipMemcpyHostToDevice);
    hipMemcpy(d_c.ptr, h_c.data(), N * 4, hipMemcpyHostToDevice);

    int n = N;
    void * args[] = { &d_a.ptr, &d_b.ptr, &d_c.ptr, &d_out.ptr, &n };
    if (!launch_kernel(fn, (N + 255) / 256, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }
    hipDeviceSynchronize();
    hipMemcpy(h_out.data(), d_out.ptr, N * 4, hipMemcpyDeviceToHost);

    float max_err = 0;
    for (int i = 0; i < N; i++) {
        float err = fabsf(h_out[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }
    if (max_err < 1e-4f) test_pass(name, "max_err=%.6f", max_err);
    else {
        char msg[128]; snprintf(msg, sizeof(msg), "max error %.8f", max_err);
        test_fail(name, msg);
    }
}

int main() {
    hipInit(0);
    hipSetDevice(0);

    std::string hsaco_path = find_hsaco("decode.hip_", 64);
    if (hsaco_path.empty()) {
        fprintf(stderr, "FAIL: decode.hip_*.hsaco not found\n");
        return 1;
    }
    hipModule_t mod;
    if (hipModuleLoad(&mod, hsaco_path.c_str()) != hipSuccess) {
        fprintf(stderr, "FAIL: cannot load %s\n", hsaco_path.c_str());
        return 1;
    }

    srand(42);

    // Unary element-wise tests
    unary_test tests[] = {
        { "eval_tanh",    ref_tanh,    1e-5f },
        { "eval_neg",     ref_neg,     1e-7f },
        { "eval_exp",     ref_exp,     1e-5f },
        { "eval_relu",    ref_relu,    1e-7f },
        { "eval_sqr",     ref_sqr,     1e-5f },
        { "eval_sigmoid", ref_sigmoid, 1e-5f },
        { "eval_gelu",    ref_gelu,    1e-4f },
        { "eval_gelu_erf",ref_gelu_erf,1e-4f },
    };
    for (auto & t : tests) test_unary(mod, t);

    // Binary/ternary tests
    test_sub(mod);
    test_muladd(mod);

    hipModuleUnload(mod);
    return test_summary();
}
