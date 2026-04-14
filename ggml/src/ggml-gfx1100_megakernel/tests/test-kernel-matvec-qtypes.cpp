// test-kernel-matvec-qtypes.cpp — Unit tests for all quantized eval_matvec_* kernels
//
// Purely synthetic. Uses ggml's quantize_row_* to create weight data,
// GPU eval_quantize_q8 to quantize input, then runs each matvec kernel
// and compares against CPU dequant + dot product reference.
//
// Tests: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
// Each type tested both normal and _residual variants.
//
// Usage: test-kernel-matvec-qtypes
// Requires: decode.hip_*.hsaco in ~/.cache/gfx1100-megakernel/

#include "test-harness.h"
#include "ggml-quants.h"

// ---------------------------------------------------------------------------
// Quant type descriptor — maps ggml type to kernel name, block size, etc.
// ---------------------------------------------------------------------------
struct QuantTypeDesc {
    const char *   kernel_name;     // e.g. "eval_matvec_q4k"
    ggml_type      type;
    size_t         block_size;      // sizeof(block_q4_K) etc.
    int            qk;              // quantization group size (32 or 256)
};

// All standard quant types to test
static const QuantTypeDesc QUANT_TYPES[] = {
    { "eval_matvec_q4_0",  GGML_TYPE_Q4_0,  sizeof(block_q4_0),  32  },
    { "eval_matvec_q4_1",  GGML_TYPE_Q4_1,  sizeof(block_q4_1),  32  },
    { "eval_matvec_q5_0",  GGML_TYPE_Q5_0,  sizeof(block_q5_0),  32  },
    { "eval_matvec_q5_1",  GGML_TYPE_Q5_1,  sizeof(block_q5_1),  32  },
    { "eval_matvec_q8_0",  GGML_TYPE_Q8_0,  sizeof(block_q8_0),  32  },
    { "eval_matvec_q2k",   GGML_TYPE_Q2_K,  sizeof(block_q2_K),  256 },
    { "eval_matvec_q3k",   GGML_TYPE_Q3_K,  sizeof(block_q3_K),  256 },
    { "eval_matvec_q4k",   GGML_TYPE_Q4_K,  sizeof(block_q4_K),  256 },
    { "eval_matvec_q5k",   GGML_TYPE_Q5_K,  sizeof(block_q5_K),  256 },
    { "eval_matvec_q6k",   GGML_TYPE_Q6_K,  sizeof(block_q6_K),  256 },
    // IQ types
    { "eval_matvec_iq2_xxs", GGML_TYPE_IQ2_XXS, sizeof(block_iq2_xxs), 256 },
    { "eval_matvec_iq2_xs",  GGML_TYPE_IQ2_XS,  sizeof(block_iq2_xs),  256 },
    { "eval_matvec_iq2_s",   GGML_TYPE_IQ2_S,   sizeof(block_iq2_s),   256 },
    { "eval_matvec_iq3_xxs", GGML_TYPE_IQ3_XXS, sizeof(block_iq3_xxs), 256 },
    { "eval_matvec_iq3_s",   GGML_TYPE_IQ3_S,   sizeof(block_iq3_s),   256 },
    { "eval_matvec_iq1_s",   GGML_TYPE_IQ1_S,   sizeof(block_iq1_s),   256 },
    { "eval_matvec_iq1_m",   GGML_TYPE_IQ1_M,   sizeof(block_iq1_m),   256 },
    { "eval_matvec_iq4_nl",  GGML_TYPE_IQ4_NL,  sizeof(block_iq4_nl),  256 },
    { "eval_matvec_iq4_xs",  GGML_TYPE_IQ4_XS,  sizeof(block_iq4_xs),  256 },
    // FP4 types
    { "eval_matvec_mxfp4",   GGML_TYPE_MXFP4,   sizeof(block_mxfp4),   256 },
    { "eval_matvec_nvfp4",   GGML_TYPE_NVFP4,   sizeof(block_nvfp4),   256 },
};
static constexpr int N_TYPES = sizeof(QUANT_TYPES) / sizeof(QUANT_TYPES[0]);

// ---------------------------------------------------------------------------
// Test a single quant type matvec (normal + residual)
// ---------------------------------------------------------------------------
static void test_matvec_qtype(hipModule_t mod, hipFunction_t quantize_fn,
                              const QuantTypeDesc & desc) {
    // --- Setup dimensions ---
    // in_dim must be divisible by qk (quantization group size)
    // Use 512 for qk=32, 512 for qk=256 (512/256=2 blocks per row)
    const int in_dim  = 512;
    const int out_dim = 32;

    // Use ggml_row_size for correct byte calculation (handles super-blocks like IQ4_NL)
    const size_t row_bytes = ggml_row_size(desc.type, in_dim);
    const long long stride = (long long)row_bytes;
    const size_t weight_total = (size_t)out_dim * row_bytes;

    // --- Create synthetic input ---
    std::vector<float> h_input(in_dim);
    for (int i = 0; i < in_dim; i++) h_input[i] = sinf((float)i * 0.1f);

    // --- Create synthetic weight matrix ---
    // 1. Generate f32 weights
    // 2. Quantize each row using ggml
    std::vector<float> h_weight_f32(out_dim * in_dim);
    for (int r = 0; r < out_dim; r++)
        for (int c = 0; c < in_dim; c++)
            h_weight_f32[r * in_dim + c] = cosf((float)(r * in_dim + c) * 0.007f) * 0.5f;

    // IQ types require an importance matrix (imatrix)
    std::vector<float> h_imatrix(in_dim, 1.0f);  // uniform importance
    const float * imatrix_ptr = ggml_quantize_requires_imatrix(desc.type) ? h_imatrix.data() : nullptr;

    std::vector<uint8_t> h_weight_q(weight_total);
    for (int r = 0; r < out_dim; r++) {
        ggml_quantize_chunk(desc.type,
                            h_weight_f32.data() + r * in_dim,
                            h_weight_q.data() + r * row_bytes,
                            0, 1, in_dim, imatrix_ptr);
    }

    // --- CPU reference: dequant + dot ---
    std::vector<float> h_cpu_out(out_dim);
    {
        std::vector<float> row_f32(in_dim);
        const ggml_type_traits * tt = ggml_get_type_traits(desc.type);
        for (int r = 0; r < out_dim; r++) {
            tt->to_float(h_weight_q.data() + r * row_bytes, row_f32.data(), in_dim);
            double acc = 0.0;
            for (int c = 0; c < in_dim; c++) acc += (double)row_f32[c] * (double)h_input[c];
            h_cpu_out[r] = (float)acc;
        }
    }

    // --- GPU: quantize input to Q8_1, then run matvec ---
    const int n_q8_blocks = in_dim / 32;
    const size_t q8_bytes = (size_t)n_q8_blocks * sizeof(block_q8_1_host);

    GpuBuf g_input_f32(in_dim * sizeof(float));
    GpuBuf g_q8_input(q8_bytes);
    GpuBuf g_weight(weight_total);
    GpuBuf g_output(out_dim * sizeof(float));

    g_input_f32.upload(h_input.data(), in_dim * sizeof(float));
    g_weight.upload(h_weight_q.data(), weight_total);

    // Step 1: GPU quantize f32 → Q8_1
    {
        void * in_ptr = g_input_f32.ptr;
        void * q8_ptr = g_q8_input.ptr;
        int n_val = in_dim;
        void * args[] = { &in_ptr, &q8_ptr, &n_val };
        unsigned grid = (in_dim + 511) / 512;
        if (!launch_kernel(quantize_fn, grid, 1, 1, 512, 1, 1, args)) {
            test_fail(desc.kernel_name, "quantize launch failed"); return;
        }
    }

    // Step 2: Run matvec
    hipFunction_t matvec_fn = load_kernel(mod, desc.kernel_name);
    if (!matvec_fn) {
        test_fail(desc.kernel_name, "kernel not found"); return;
    }

    {
        void * w_ptr = g_weight.ptr;
        long long s_val = stride;
        void * q8_ptr = g_q8_input.ptr;
        void * o_ptr = g_output.ptr;
        int in_val = in_dim, out_val = out_dim;
        void * args[] = { &w_ptr, &s_val, &q8_ptr, &o_ptr, &in_val, &out_val };
        if (!launch_kernel(matvec_fn, (unsigned)out_dim, 1, 1, 32, 4, 1, args)) {
            test_fail(desc.kernel_name, "matvec launch failed"); return;
        }
    }

    std::vector<float> h_gpu_out(out_dim);
    g_output.download(h_gpu_out.data(), out_dim * sizeof(float));
    CompareResult r = compare_float(h_gpu_out.data(), h_cpu_out.data(), out_dim);

    if (r.nan_count > 0) { test_fail(desc.kernel_name, "%d NaN", r.nan_count); return; }
    // Double quantization (weight + input) → allow up to 10% relative error
    // IQ1_S/IQ1_M (~1.5 bpw) and NVFP4 (~4 bpw with block scaling) are extremely lossy — allow 25%
    float tol = (desc.type == GGML_TYPE_IQ1_S || desc.type == GGML_TYPE_IQ1_M ||
                 desc.type == GGML_TYPE_NVFP4) ? 0.25f : 0.10f;
    if (r.max_rel > tol) {
        test_fail(desc.kernel_name, "max_rel=%.4f > %.2f", (double)r.max_rel, (double)tol);

        // Print first 4 for debugging
        fprintf(stderr, "  GPU:  ");
        for (int i = 0; i < 4; i++) fprintf(stderr, "%.4f ", h_gpu_out[i]);
        fprintf(stderr, "\n  CPU:  ");
        for (int i = 0; i < 4; i++) fprintf(stderr, "%.4f ", h_cpu_out[i]);
        fprintf(stderr, "\n");
        return;
    }
    test_pass(desc.kernel_name, "max_abs=%.2e, max_rel=%.4f",
              (double)r.max_abs, (double)r.max_rel);

    // --- Test residual variant ---
    char res_name[128];
    snprintf(res_name, sizeof(res_name), "%s_residual", desc.kernel_name);
    hipFunction_t res_fn = load_kernel(mod, res_name);
    if (!res_fn) {
        test_fail(res_name, "kernel not found"); return;
    }

    std::vector<float> h_residual(out_dim);
    for (int i = 0; i < out_dim; i++) h_residual[i] = (float)i * 0.1f - 1.5f;

    GpuBuf g_residual(out_dim * sizeof(float));
    GpuBuf g_output2(out_dim * sizeof(float));
    g_residual.upload(h_residual.data(), out_dim * sizeof(float));

    {
        void * w_ptr = g_weight.ptr;
        long long s_val = stride;
        void * q8_ptr = g_q8_input.ptr;
        void * res_ptr = g_residual.ptr;
        void * o_ptr = g_output2.ptr;
        int in_val = in_dim, out_val = out_dim;
        void * args[] = { &w_ptr, &s_val, &q8_ptr, &res_ptr, &o_ptr, &in_val, &out_val };
        if (!launch_kernel(res_fn, (unsigned)out_dim, 1, 1, 32, 4, 1, args)) {
            test_fail(res_name, "launch failed"); return;
        }
    }

    std::vector<float> h_gpu_res(out_dim);
    g_output2.download(h_gpu_res.data(), out_dim * sizeof(float));

    // CPU reference for residual: matvec_output + residual
    std::vector<float> h_cpu_res(out_dim);
    for (int i = 0; i < out_dim; i++) h_cpu_res[i] = h_gpu_out[i] + h_residual[i];

    CompareResult r2 = compare_float(h_gpu_res.data(), h_cpu_res.data(), out_dim);
    if (r2.nan_count > 0) { test_fail(res_name, "%d NaN", r2.nan_count); return; }
    if (r2.max_abs > 1e-4f) {
        test_fail(res_name, "max_abs=%.2e > 1e-4 (residual add error)", (double)r2.max_abs);
        return;
    }
    test_pass(res_name, "max_abs=%.2e (residual correct)", (double)r2.max_abs);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    // Init IQ quantization lookup tables (needed for IQ2/IQ3/IQ1/IQ4 types)
    ggml_quantize_init(GGML_TYPE_IQ2_XXS);
    ggml_quantize_init(GGML_TYPE_IQ2_XS);
    ggml_quantize_init(GGML_TYPE_IQ2_S);
    ggml_quantize_init(GGML_TYPE_IQ3_XXS);
    ggml_quantize_init(GGML_TYPE_IQ3_S);
    ggml_quantize_init(GGML_TYPE_IQ1_S);
    ggml_quantize_init(GGML_TYPE_IQ1_M);

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

    // Load the quantize kernel (shared by all matvec tests)
    hipFunction_t quantize_fn = load_kernel(mod, "eval_quantize_q8");
    if (!quantize_fn) {
        fprintf(stderr, "FAIL: eval_quantize_q8 not found\n");
        hipModuleUnload(mod);
        return 1;
    }

    for (int i = 0; i < N_TYPES; i++) {
        test_matvec_qtype(mod, quantize_fn, QUANT_TYPES[i]);
    }

    hipModuleUnload(mod);
    return test_summary();
}
