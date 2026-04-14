// test-kernel-lmhead.cpp — Unit tests for eval_lm_head (Q4_K) and eval_lm_head_q6k
//
// Synthetic. Creates quantized weight matrix (small vocab), Q8_1 input via GPU,
// runs LM head kernel, compares against CPU dequant + dot product.
//
// Usage: test-kernel-lmhead
// Requires: decode.hip_*.hsaco in ~/.cache/gfx1100-megakernel/

#include "test-harness.h"
#include "ggml-quants.h"

// ---------------------------------------------------------------------------
// Test a single LM head variant
// ---------------------------------------------------------------------------
static void test_lm_head(hipModule_t mod, hipFunction_t quantize_fn,
                         const char * kernel_name, ggml_type qtype,
                         size_t block_size, int qk) {
    hipFunction_t fn = load_kernel(mod, kernel_name);
    if (!fn) { test_fail(kernel_name, "kernel not found"); return; }

    const int in_dim    = 512;
    const int vocab     = 32;   // small test vocab
    const int blocks_per_row = in_dim / qk;
    const size_t row_bytes = (size_t)blocks_per_row * block_size;
    const long long stride = (long long)row_bytes;

    // Synthetic f32 input
    std::vector<float> h_input(in_dim);
    for (int i = 0; i < in_dim; i++) h_input[i] = sinf((float)i * 0.1f) * 0.5f;

    // Create quantized weight: [vocab, in_dim]
    std::vector<float> h_wf32(vocab * in_dim);
    for (int r = 0; r < vocab; r++)
        for (int c = 0; c < in_dim; c++)
            h_wf32[r * in_dim + c] = cosf((float)(r * in_dim + c) * 0.005f) * 0.3f;

    std::vector<uint8_t> h_wq(vocab * row_bytes);
    for (int r = 0; r < vocab; r++) {
        ggml_quantize_chunk(qtype,
                            h_wf32.data() + r * in_dim,
                            h_wq.data() + r * row_bytes,
                            0, 1, in_dim, nullptr);
    }

    // CPU reference
    std::vector<float> h_cpu(vocab);
    {
        std::vector<float> row_f32(in_dim);
        const ggml_type_traits * tt = ggml_get_type_traits(qtype);
        for (int r = 0; r < vocab; r++) {
            tt->to_float(h_wq.data() + r * row_bytes, row_f32.data(), in_dim);
            double acc = 0.0;
            for (int c = 0; c < in_dim; c++) acc += (double)row_f32[c] * (double)h_input[c];
            h_cpu[r] = (float)acc;
        }
    }

    // GPU
    const int n_q8_blocks = in_dim / 32;
    const size_t q8_bytes = (size_t)n_q8_blocks * sizeof(block_q8_1_host);

    GpuBuf g_input(in_dim * sizeof(float));
    GpuBuf g_q8(q8_bytes);
    GpuBuf g_weight(vocab * row_bytes);
    GpuBuf g_logits(vocab * sizeof(float));

    g_input.upload(h_input.data(), in_dim * sizeof(float));
    g_weight.upload(h_wq.data(), vocab * row_bytes);

    // Quantize input to Q8_1
    {
        void * in_ptr = g_input.ptr, * q8_ptr = g_q8.ptr;
        int n_val = in_dim;
        void * args[] = { &in_ptr, &q8_ptr, &n_val };
        if (!launch_kernel(quantize_fn, (in_dim + 511) / 512, 1, 1, 512, 1, 1, args)) {
            test_fail(kernel_name, "quantize failed"); return;
        }
    }

    // Launch LM head
    // Signature: (q8_input, weight, weight_stride_bytes, logits, in_dim, vocab_size)
    {
        void * q8_ptr = g_q8.ptr, * w_ptr = g_weight.ptr, * l_ptr = g_logits.ptr;
        long long s_val = stride;
        int in_val = in_dim, v_val = vocab;
        void * args[] = { &q8_ptr, &w_ptr, &s_val, &l_ptr, &in_val, &v_val };
        if (!launch_kernel(fn, (unsigned)vocab, 1, 1, 32, 4, 1, args)) {
            test_fail(kernel_name, "launch failed"); return;
        }
    }

    std::vector<float> h_gpu(vocab);
    g_logits.download(h_gpu.data(), vocab * sizeof(float));
    CompareResult r = compare_float(h_gpu.data(), h_cpu.data(), vocab);

    if (r.nan_count > 0) { test_fail(kernel_name, "%d NaN", r.nan_count); return; }
    if (r.max_rel > 0.10f) {
        test_fail(kernel_name, "max_rel=%.4f > 10%%", (double)r.max_rel);
        return;
    }
    test_pass(kernel_name, "max_abs=%.2e, max_rel=%.4f", (double)r.max_abs, (double)r.max_rel);
}

int main() {
    std::string hsaco_path = find_hsaco("decode.hip_", 64);
    if (hsaco_path.empty()) { fprintf(stderr, "FAIL: decode.hip not found\n"); return 1; }
    fprintf(stderr, "Using hsaco: %s\n\n", hsaco_path.c_str());

    hipModule_t mod = nullptr;
    if (hipModuleLoad(&mod, hsaco_path.c_str()) != hipSuccess) {
        fprintf(stderr, "FAIL: hipModuleLoad\n"); return 1;
    }
    hipFunction_t qfn = load_kernel(mod, "eval_quantize_q8");
    if (!qfn) { fprintf(stderr, "FAIL: quantize kernel\n"); return 1; }

    test_lm_head(mod, qfn, "eval_lm_head",     GGML_TYPE_Q4_K, sizeof(block_q4_K), 256);
    test_lm_head(mod, qfn, "eval_lm_head_q6k", GGML_TYPE_Q6_K, sizeof(block_q6_K), 256);

    hipModuleUnload(mod);
    return test_summary();
}
