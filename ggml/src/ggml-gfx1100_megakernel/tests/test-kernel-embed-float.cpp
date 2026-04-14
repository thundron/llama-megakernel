// test-kernel-embed-float.cpp — Unit tests for eval_embed_f32, _f16, _bf16
//
// Purely synthetic. Creates fake embedding table rows, runs GPU embed kernel,
// compares against CPU dequant.
//
// NOTE: Uses HIDDEN_SIZE=2048 (Llama 1B .hsaco default)
//
// Usage: test-kernel-embed-float
// Requires: decode.hip_*.hsaco in ~/.cache/gfx1100-megakernel/

#include "test-harness.h"

static constexpr int HIDDEN = 2048;  // must match HIDDEN_SIZE in .hsaco

// ---------------------------------------------------------------------------
// Test: eval_embed_f32
// Grid: (HIDDEN+255)/256, Block: 256
// Signature: (const void* embed_weight, long long embed_stride, float* hidden, int token_id)
// ---------------------------------------------------------------------------
static void test_embed_f32(hipModule_t mod) {
    const char * name = "eval_embed_f32";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    // Create fake embedding table: 4 rows × HIDDEN floats
    const int n_vocab = 4;
    const long long stride = (long long)HIDDEN * sizeof(float);
    std::vector<float> h_table(n_vocab * HIDDEN);
    for (int v = 0; v < n_vocab; v++)
        for (int d = 0; d < HIDDEN; d++)
            h_table[v * HIDDEN + d] = sinf((float)(v * HIDDEN + d) * 0.01f);

    const int token_id = 2;  // test row 2
    std::vector<float> h_cpu(HIDDEN);
    for (int d = 0; d < HIDDEN; d++) h_cpu[d] = h_table[token_id * HIDDEN + d];

    GpuBuf g_table(n_vocab * HIDDEN * sizeof(float));
    GpuBuf g_hidden(HIDDEN * sizeof(float));
    g_table.upload(h_table.data(), n_vocab * HIDDEN * sizeof(float));

    void * tbl_ptr = g_table.ptr;
    long long s_val = stride;
    void * hid_ptr = g_hidden.ptr;
    int tok_val = token_id;
    void * args[] = { &tbl_ptr, &s_val, &hid_ptr, &tok_val };

    unsigned grid = (HIDDEN + 255) / 256;
    if (!launch_kernel(fn, grid, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }

    std::vector<float> h_gpu(HIDDEN);
    g_hidden.download(h_gpu.data(), HIDDEN * sizeof(float));
    CompareResult r = compare_float(h_gpu.data(), h_cpu.data(), HIDDEN);

    if (r.nan_count > 0) { test_fail(name, "%d NaN", r.nan_count); return; }
    if (r.max_abs > 1e-6f) { test_fail(name, "max_abs=%.2e > 1e-6", (double)r.max_abs); return; }
    test_pass(name, "max_abs=%.2e (exact F32 copy)", (double)r.max_abs);
}

// ---------------------------------------------------------------------------
// Test: eval_embed_f16
// ---------------------------------------------------------------------------
static void test_embed_f16(hipModule_t mod) {
    const char * name = "eval_embed_f16";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int n_vocab = 4;
    const long long stride = (long long)HIDDEN * sizeof(uint16_t);
    std::vector<uint16_t> h_table(n_vocab * HIDDEN);
    for (int v = 0; v < n_vocab; v++)
        for (int d = 0; d < HIDDEN; d++)
            h_table[v * HIDDEN + d] = f32_to_f16(sinf((float)(v * HIDDEN + d) * 0.01f));

    const int token_id = 1;
    std::vector<float> h_cpu(HIDDEN);
    for (int d = 0; d < HIDDEN; d++) h_cpu[d] = f16_to_f32(h_table[token_id * HIDDEN + d]);

    GpuBuf g_table(n_vocab * HIDDEN * sizeof(uint16_t));
    GpuBuf g_hidden(HIDDEN * sizeof(float));
    g_table.upload(h_table.data(), n_vocab * HIDDEN * sizeof(uint16_t));

    void * tbl_ptr = g_table.ptr;
    long long s_val = stride;
    void * hid_ptr = g_hidden.ptr;
    int tok_val = token_id;
    void * args[] = { &tbl_ptr, &s_val, &hid_ptr, &tok_val };

    unsigned grid = (HIDDEN + 255) / 256;
    if (!launch_kernel(fn, grid, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }

    std::vector<float> h_gpu(HIDDEN);
    g_hidden.download(h_gpu.data(), HIDDEN * sizeof(float));
    CompareResult r = compare_float(h_gpu.data(), h_cpu.data(), HIDDEN);

    if (r.nan_count > 0) { test_fail(name, "%d NaN", r.nan_count); return; }
    if (r.max_abs > 1e-3f) { test_fail(name, "max_abs=%.2e > 1e-3", (double)r.max_abs); return; }
    test_pass(name, "max_abs=%.2e", (double)r.max_abs);
}

// ---------------------------------------------------------------------------
// Test: eval_embed_bf16
// ---------------------------------------------------------------------------
static void test_embed_bf16(hipModule_t mod) {
    const char * name = "eval_embed_bf16";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int n_vocab = 4;
    const long long stride = (long long)HIDDEN * sizeof(uint16_t);
    std::vector<uint16_t> h_table(n_vocab * HIDDEN);
    for (int v = 0; v < n_vocab; v++)
        for (int d = 0; d < HIDDEN; d++)
            h_table[v * HIDDEN + d] = f32_to_bf16(sinf((float)(v * HIDDEN + d) * 0.01f));

    const int token_id = 3;
    std::vector<float> h_cpu(HIDDEN);
    for (int d = 0; d < HIDDEN; d++) h_cpu[d] = bf16_to_f32(h_table[token_id * HIDDEN + d]);

    GpuBuf g_table(n_vocab * HIDDEN * sizeof(uint16_t));
    GpuBuf g_hidden(HIDDEN * sizeof(float));
    g_table.upload(h_table.data(), n_vocab * HIDDEN * sizeof(uint16_t));

    void * tbl_ptr = g_table.ptr;
    long long s_val = stride;
    void * hid_ptr = g_hidden.ptr;
    int tok_val = token_id;
    void * args[] = { &tbl_ptr, &s_val, &hid_ptr, &tok_val };

    unsigned grid = (HIDDEN + 255) / 256;
    if (!launch_kernel(fn, grid, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }

    std::vector<float> h_gpu(HIDDEN);
    g_hidden.download(h_gpu.data(), HIDDEN * sizeof(float));
    CompareResult r = compare_float(h_gpu.data(), h_cpu.data(), HIDDEN);

    if (r.nan_count > 0) { test_fail(name, "%d NaN", r.nan_count); return; }
    // BF16 has less precision than F16
    if (r.max_abs > 0.01f) { test_fail(name, "max_abs=%.2e > 0.01", (double)r.max_abs); return; }
    test_pass(name, "max_abs=%.2e", (double)r.max_abs);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    // Needs HIDDEN_SIZE=2048 → FA_HEAD_DIM=64 .hsaco (Llama 1B)
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

    test_embed_f32(mod);
    test_embed_f16(mod);
    test_embed_bf16(mod);

    hipModuleUnload(mod);
    return test_summary();
}
