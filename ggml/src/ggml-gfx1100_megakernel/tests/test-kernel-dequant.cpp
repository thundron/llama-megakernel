// test-kernel-dequant.cpp — Unit tests for all prompt_dequant_*_f16 kernels
//
// Synthetic. For each quant type: create random f32 data, quantize with ggml,
// run GPU dequant-to-F16 kernel, compare F16→F32 output against ggml's to_float.
//
// Tests standard types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
// Also tests: F32→F16, BF16→F16
//
// Usage: test-kernel-dequant
// Requires: prefill.hip_*.hsaco in ~/.cache/gfx1100-megakernel/

#include "test-harness.h"
#include "ggml-quants.h"

// ---------------------------------------------------------------------------
// Dequant type descriptor
// ---------------------------------------------------------------------------
struct DequantDesc {
    const char * kernel_name;   // e.g. "prompt_dequant_q4k_f16"
    ggml_type    type;
    int          qk;            // group size: 32 or 256
    unsigned     block_threads; // threadblock size for the kernel
    bool         grid_per_superblock; // true = grid=(k/qk), false = grid=(k+threads-1)/threads
    int          ne_align;      // 0 = use default, >0 = grid=(k+ne_align-1)/ne_align (Q8_0 uses 2048)
};

static const DequantDesc DEQUANT_TYPES[] = {
    // Standard types — grid = k / qk superblocks
    { "prompt_dequant_q4_0_f16",  GGML_TYPE_Q4_0,  32,  32, true,  0    },
    { "prompt_dequant_q4_1_f16",  GGML_TYPE_Q4_1,  32,  32, true,  0    },
    { "prompt_dequant_q5_0_f16",  GGML_TYPE_Q5_0,  32,  32, true,  0    },
    { "prompt_dequant_q5_1_f16",  GGML_TYPE_Q5_1,  32,  32, true,  0    },
    // Q8_0 uses baseline convert.cu approach: grid = (k + 2048 - 1) / 2048, block = 32
    { "prompt_dequant_q8_0_f16",  GGML_TYPE_Q8_0,  32,  32, false, 2048 },
    // K-quant types — grid = k / QK_K superblocks
    { "prompt_dequant_q2k_f16",   GGML_TYPE_Q2_K,  256, 64, true,  0    },
    { "prompt_dequant_q3k_f16",   GGML_TYPE_Q3_K,  256, 64, true,  0    },
    { "prompt_dequant_q4k_f16",   GGML_TYPE_Q4_K,  256, 32, true,  0    },
    { "prompt_dequant_q5k_f16",   GGML_TYPE_Q5_K,  256, 64, true,  0    },
    { "prompt_dequant_q6k_f16",   GGML_TYPE_Q6_K,  256, 64, true,  0    },
};
static constexpr int N_DEQUANT = sizeof(DEQUANT_TYPES) / sizeof(DEQUANT_TYPES[0]);

// ---------------------------------------------------------------------------
// Test a single dequant kernel
// ---------------------------------------------------------------------------
static void test_dequant(hipModule_t mod, const DequantDesc & desc) {
    hipFunction_t fn = load_kernel(mod, desc.kernel_name);
    if (!fn) { test_fail(desc.kernel_name, "kernel not found in prefill.hip"); return; }

    // k = number of float elements to dequant (must be multiple of qk AND ne_align if set)
    const int k = 2048;
    if (k % desc.qk != 0) {
        test_fail(desc.kernel_name, "k=%d not divisible by qk=%d", k, desc.qk);
        return;
    }

    // 1. Create random f32 data and quantize with ggml
    std::vector<float> h_f32(k);
    for (int i = 0; i < k; i++) h_f32[i] = sinf((float)i * 0.05f) * 2.0f;

    const size_t qbytes = ggml_row_size(desc.type, k);
    std::vector<uint8_t> h_quant(qbytes);
    ggml_quantize_chunk(desc.type, h_f32.data(), h_quant.data(), 0, 1, k, nullptr);

    // 2. CPU reference: ggml dequant to f32
    std::vector<float> h_cpu_f32(k);
    {
        const ggml_type_traits * tt = ggml_get_type_traits(desc.type);
        tt->to_float(h_quant.data(), h_cpu_f32.data(), k);
    }

    // 3. GPU: upload quant data, run dequant kernel, get F16 output
    GpuBuf g_quant(qbytes);
    GpuBuf g_f16(k * sizeof(uint16_t));  // __half output
    g_quant.upload(h_quant.data(), qbytes);

    void * vx_ptr = g_quant.ptr;
    void * yy_ptr = g_f16.ptr;
    int64_t k_val = k;
    void * args[] = { &vx_ptr, &yy_ptr, &k_val };

    unsigned grid;
    if (desc.ne_align > 0) {
        grid = (k + desc.ne_align - 1) / desc.ne_align;
    } else if (desc.grid_per_superblock) {
        grid = k / desc.qk;
    } else {
        grid = (k + desc.block_threads - 1) / desc.block_threads;
    }

    if (!launch_kernel(fn, grid, 1, 1, desc.block_threads, 1, 1, args)) {
        test_fail(desc.kernel_name, "launch failed"); return;
    }

    // 4. Download F16 output, convert to f32, compare
    std::vector<uint16_t> h_f16(k);
    g_f16.download(h_f16.data(), k * sizeof(uint16_t));

    std::vector<float> h_gpu_f32(k);
    for (int i = 0; i < k; i++) h_gpu_f32[i] = f16_to_f32(h_f16[i]);

    CompareResult r = compare_float(h_gpu_f32.data(), h_cpu_f32.data(), k);

    if (r.nan_count > 0) { test_fail(desc.kernel_name, "%d NaN", r.nan_count); return; }
    // F16 output adds truncation error on top of quantization error
    // Allow 0.01 abs (F16 precision) + 5% relative
    if (r.max_rel > 0.05f && r.max_abs > 0.01f) {
        test_fail(desc.kernel_name, "max_abs=%.4f max_rel=%.4f",
                  (double)r.max_abs, (double)r.max_rel);
        return;
    }
    test_pass(desc.kernel_name, "max_abs=%.2e, max_rel=%.4f",
              (double)r.max_abs, (double)r.max_rel);
}

// ---------------------------------------------------------------------------
// Test: prompt_dequant_f32_f16 and prompt_dequant_bf16_f16
// These are simple element-wise conversions, not block-structured
// Grid: (k + threads - 1) / threads, Block: 256
// Signature: (const void* vx, __half* y, int64_t k)
// ---------------------------------------------------------------------------
static void test_dequant_f32(hipModule_t mod) {
    const char * name = "prompt_dequant_f32_f16";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int k = 1024;
    std::vector<float> h_f32(k);
    for (int i = 0; i < k; i++) h_f32[i] = sinf((float)i * 0.1f) * 5.0f;

    GpuBuf g_in(k * sizeof(float));
    GpuBuf g_out(k * sizeof(uint16_t));
    g_in.upload(h_f32.data(), k * sizeof(float));

    void * in_ptr = g_in.ptr, * out_ptr = g_out.ptr;
    int64_t k_val = k;
    void * args[] = { &in_ptr, &out_ptr, &k_val };
    unsigned grid = (k + 255) / 256;

    if (!launch_kernel(fn, grid, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }

    std::vector<uint16_t> h_f16(k);
    g_out.download(h_f16.data(), k * sizeof(uint16_t));

    // Check: f16(gpu) ≈ f16(cpu)
    int err = 0;
    float max_diff = 0;
    for (int i = 0; i < k; i++) {
        float gpu_val = f16_to_f32(h_f16[i]);
        float cpu_val = f16_to_f32(f32_to_f16(h_f32[i]));
        float diff = fabsf(gpu_val - cpu_val);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-3f) err++;
    }
    if (err > 0) { test_fail(name, "%d errors, max_diff=%.4f", err, (double)max_diff); return; }
    test_pass(name, "max_diff=%.2e", (double)max_diff);
}

static void test_dequant_bf16(hipModule_t mod) {
    const char * name = "prompt_dequant_bf16_f16";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int k = 1024;
    std::vector<uint16_t> h_bf16(k);
    for (int i = 0; i < k; i++) h_bf16[i] = f32_to_bf16(sinf((float)i * 0.1f) * 5.0f);

    GpuBuf g_in(k * sizeof(uint16_t));
    GpuBuf g_out(k * sizeof(uint16_t));
    g_in.upload(h_bf16.data(), k * sizeof(uint16_t));

    void * in_ptr = g_in.ptr, * out_ptr = g_out.ptr;
    int64_t k_val = k;
    void * args[] = { &in_ptr, &out_ptr, &k_val };
    unsigned grid = (k + 255) / 256;

    if (!launch_kernel(fn, grid, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }

    std::vector<uint16_t> h_f16(k);
    g_out.download(h_f16.data(), k * sizeof(uint16_t));

    int err = 0;
    float max_diff = 0;
    for (int i = 0; i < k; i++) {
        float gpu_val = f16_to_f32(h_f16[i]);
        float cpu_val = bf16_to_f32(h_bf16[i]);  // bf16→f32 reference
        float diff = fabsf(gpu_val - cpu_val);
        if (diff > max_diff) max_diff = diff;
        // BF16→F16 loses precision in different mantissa bits
        if (diff > 0.05f) err++;
    }
    if (err > 0) { test_fail(name, "%d errors, max_diff=%.4f", err, (double)max_diff); return; }
    test_pass(name, "max_diff=%.2e", (double)max_diff);
}

int main() {
    std::string hsaco_path = find_hsaco("prefill.hip_", 64);
    if (hsaco_path.empty()) {
        fprintf(stderr, "FAIL: prefill.hip_*.hsaco not found\n");
        return 1;
    }
    fprintf(stderr, "Using hsaco: %s\n\n", hsaco_path.c_str());

    hipModule_t mod = nullptr;
    if (hipModuleLoad(&mod, hsaco_path.c_str()) != hipSuccess) {
        fprintf(stderr, "FAIL: hipModuleLoad\n"); return 1;
    }

    // Standard quant dequant tests
    for (int i = 0; i < N_DEQUANT; i++) {
        test_dequant(mod, DEQUANT_TYPES[i]);
    }

    // F32/BF16 → F16 tests
    test_dequant_f32(mod);
    test_dequant_bf16(mod);

    hipModuleUnload(mod);
    return test_summary();
}
