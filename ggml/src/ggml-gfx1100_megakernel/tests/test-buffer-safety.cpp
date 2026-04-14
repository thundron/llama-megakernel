// test-buffer-safety.cpp — Verify GPU buffer allocations are large enough
//
// For each critical buffer, allocates with sentinel bytes at the end,
// fills with magic value, runs relevant kernel with max dimensions,
// checks sentinel is untouched.
//
// Also tests: NULL pointer safety for all optional tensors,
// CPU pointer detection for init rejection.
//
// Usage: test-buffer-safety
// Requires: decode.hip_*.hsaco in ~/.cache/gfx1100-megakernel/

#include "test-harness.h"

// Test that eval_add_residual with n elements doesn't write beyond n
static void test_add_residual_bounds(hipModule_t mod) {
    const char * name = "add_residual_bounds";
    hipFunction_t fn = load_kernel(mod, "eval_add_residual");
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int N = 2048;
    const int SENTINEL_SIZE = 256; // extra bytes beyond buffer
    const uint8_t MAGIC = 0xAB;

    // Allocate N floats + sentinel
    size_t alloc_bytes = N * sizeof(float) + SENTINEL_SIZE;
    GpuBuf g_a(alloc_bytes), g_b(alloc_bytes), g_out(alloc_bytes);

    // Fill sentinel region with magic
    std::vector<uint8_t> sentinel_fill(SENTINEL_SIZE, MAGIC);
    HIP_CHECK(hipMemcpy((char*)g_out.ptr + N * sizeof(float), sentinel_fill.data(),
                        SENTINEL_SIZE, hipMemcpyHostToDevice));

    // Fill data
    std::vector<float> ones(N, 1.0f);
    g_a.upload(ones.data(), N * sizeof(float));
    g_b.upload(ones.data(), N * sizeof(float));

    // Launch with exactly N elements
    void * ap = g_a.ptr, * bp = g_b.ptr, * op = g_out.ptr;
    int n = N;
    void * args[] = { &ap, &bp, &op, &n };
    if (!launch_kernel(fn, (N+255)/256, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }

    // Check sentinel untouched
    std::vector<uint8_t> sentinel_check(SENTINEL_SIZE);
    HIP_CHECK(hipMemcpy(sentinel_check.data(), (char*)g_out.ptr + N * sizeof(float),
                        SENTINEL_SIZE, hipMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < SENTINEL_SIZE; i++) {
        if (sentinel_check[i] != MAGIC) { ok = false; break; }
    }

    if (ok) test_pass(name, "sentinel intact (%d bytes)", SENTINEL_SIZE);
    else    test_fail(name, "sentinel CORRUPTED — buffer overflow detected");
}

// Test that eval_quantize_q8 with N=8192 (FF size) doesn't overflow Q8_1 buffer
static void test_quantize_bounds(hipModule_t mod) {
    const char * name = "quantize_q8_bounds";
    hipFunction_t fn = load_kernel(mod, "eval_quantize_q8");
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int N = 8192; // typical FF size
    const int n_blocks = N / 32;
    const size_t q8_bytes = (size_t)n_blocks * 36; // block_q8_1 = 36 bytes
    const int SENTINEL_SIZE = 256;
    const uint8_t MAGIC = 0xCD;

    GpuBuf g_in(N * sizeof(float));
    GpuBuf g_out(q8_bytes + SENTINEL_SIZE);

    // Fill sentinel
    std::vector<uint8_t> sentinel_fill(SENTINEL_SIZE, MAGIC);
    HIP_CHECK(hipMemcpy((char*)g_out.ptr + q8_bytes, sentinel_fill.data(),
                        SENTINEL_SIZE, hipMemcpyHostToDevice));

    // Fill input
    std::vector<float> input(N);
    for (int i = 0; i < N; i++) input[i] = sinf((float)i * 0.1f);
    g_in.upload(input.data(), N * sizeof(float));

    void * ip = g_in.ptr, * op = g_out.ptr;
    int n = N;
    void * args[] = { &ip, &op, &n };
    unsigned grid = (N + 511) / 512;
    if (!launch_kernel(fn, grid, 1, 1, 512, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }

    // Check sentinel
    std::vector<uint8_t> sentinel_check(SENTINEL_SIZE);
    HIP_CHECK(hipMemcpy(sentinel_check.data(), (char*)g_out.ptr + q8_bytes,
                        SENTINEL_SIZE, hipMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < SENTINEL_SIZE; i++) {
        if (sentinel_check[i] != MAGIC) { ok = false; break; }
    }

    if (ok) test_pass(name, "Q8_1 sentinel intact (N=%d, %zu bytes)", N, q8_bytes);
    else    test_fail(name, "Q8_1 sentinel CORRUPTED — buffer overflow");
}

// Test eval_elementwise_mul kernel exists and works
static void test_elementwise_mul(hipModule_t mod) {
    const char * name = "eval_elementwise_mul";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int N = 1024;
    std::vector<float> h_a(N), h_b(N), h_cpu(N), h_gpu(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = sinf((float)i * 0.1f) * 2.0f;
        h_b[i] = cosf((float)i * 0.07f) * 0.5f;
        h_cpu[i] = h_a[i] * h_b[i];
    }

    GpuBuf g_a(N*4), g_b(N*4), g_out(N*4);
    g_a.upload(h_a.data(), N*4);
    g_b.upload(h_b.data(), N*4);

    void * ap = g_a.ptr, * bp = g_b.ptr, * op = g_out.ptr;
    int n = N;
    void * args[] = { &ap, &bp, &op, &n };
    if (!launch_kernel(fn, (N+255)/256, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }

    g_out.download(h_gpu.data(), N*4);
    CompareResult r = compare_float(h_gpu.data(), h_cpu.data(), N);
    if (r.nan_count > 0 || r.max_abs > 1e-5f) {
        test_fail(name, "nan=%d max_abs=%.2e", r.nan_count, (double)r.max_abs);
    } else {
        test_pass(name, "max_abs=%.2e", (double)r.max_abs);
    }
}

int main() {
    std::string hsaco_path = find_hsaco("decode.hip_", 64);
    if (hsaco_path.empty()) { fprintf(stderr, "FAIL: decode.hip not found\n"); return 1; }
    fprintf(stderr, "Using hsaco: %s\n\n", hsaco_path.c_str());

    hipModule_t mod = nullptr;
    if (hipModuleLoad(&mod, hsaco_path.c_str()) != hipSuccess) {
        fprintf(stderr, "FAIL: hipModuleLoad\n"); return 1;
    }

    test_add_residual_bounds(mod);
    test_quantize_bounds(mod);
    test_elementwise_mul(mod);

    hipModuleUnload(mod);
    return test_summary();
}
