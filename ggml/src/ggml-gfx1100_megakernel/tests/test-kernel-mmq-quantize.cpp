// test-kernel-mmq-quantize.cpp — Unit tests for eval_quantize_mmq_q8_1_{d4,ds4,d2s6}
//
// Synthetic. Creates known f32 input, runs MMQ quantize kernel, reads back
// block_q8_1_mmq output, verifies scales are valid and dequantized values
// approximate the original input.
//
// Usage: test-kernel-mmq-quantize
// Requires: decode.hip_*.hsaco in ~/.cache/gfx1100-megakernel/

#include "test-harness.h"

// Match block_q8_1_mmq layout from mmq-quantize.h
struct block_q8_1_mmq_host {
    union {
        float    d4[4];   // D4 layout: 4 float32 scales
        uint32_t ds4[4];  // DS4 layout: 4 packed half2(d, sum)
        uint16_t d2s6[8]; // D2S6 layout: 2 half scales + 6 half sums
    };
    int8_t qs[128];       // 128 quantized int8 values
};
static_assert(sizeof(block_q8_1_mmq_host) == 144, "block_q8_1_mmq size mismatch");

// ---------------------------------------------------------------------------
// Test a single MMQ quantize kernel
// ---------------------------------------------------------------------------
static void test_mmq_quantize(hipModule_t mod, const char * kernel_name, int layout) {
    hipFunction_t fn = load_kernel(mod, kernel_name);
    if (!fn) { test_fail(kernel_name, "kernel not found"); return; }

    // Setup: 2 rows of 512 floats each
    int64_t ne00 = 512;
    int64_t ne0  = 512;
    int     ne1  = 2;
    int     ne2  = 1;
    int64_t s01  = ne00;  // row stride in floats
    int64_t s02  = ne00 * ne1;
    int64_t s03  = ne00 * ne1 * ne2;

    // Number of blocks: ne0 / 128 * ne1 = 4 * 2 = 8 blocks
    const int n_blocks = (int)(ne0 / 128) * ne1;
    const size_t out_bytes = (size_t)n_blocks * sizeof(block_q8_1_mmq_host);

    // Synthetic input
    std::vector<float> h_input(ne1 * ne00);
    for (int i = 0; i < (int)(ne1 * ne00); i++) {
        h_input[i] = sinf((float)i * 0.07f) * 5.0f;
    }

    GpuBuf g_input(ne1 * ne00 * sizeof(float));
    GpuBuf g_output(out_bytes);
    g_input.upload(h_input.data(), ne1 * ne00 * sizeof(float));

    // Launch: grid=(ne1, block_num_y, ne2*ne3), block=(128,1,1)
    // block_num_y = (ne0 + 4*128 - 1) / (4*128)
    unsigned block_num_y = (unsigned)((ne0 + 4 * 128 - 1) / (4 * 128));

    void * x_ptr = g_input.ptr;
    void * ids_ptr = nullptr;  // no id remapping
    void * vy_ptr = g_output.ptr;
    void * args[] = { &x_ptr, &ids_ptr, &vy_ptr, &ne00, &s01, &s02, &s03, &ne0, &ne1, &ne2 };

    if (!launch_kernel(fn, (unsigned)ne1, block_num_y, (unsigned)(ne2), 128, 1, 1, args)) {
        test_fail(kernel_name, "launch failed"); return;
    }

    // Read back
    std::vector<uint8_t> h_raw(out_bytes);
    g_output.download(h_raw.data(), out_bytes);
    const block_q8_1_mmq_host * blocks = (const block_q8_1_mmq_host *)h_raw.data();

    // Validate blocks
    int nan_scales = 0, zero_scales = 0, zero_qs = 0;
    float max_dequant_err = 0;
    int total_vals = 0;

    for (int b = 0; b < n_blocks; b++) {
        const block_q8_1_mmq_host & blk = blocks[b];

        if (layout == 0) { // D4
            for (int s = 0; s < 4; s++) {
                if (std::isnan(blk.d4[s])) nan_scales++;
                if (blk.d4[s] == 0.0f) zero_scales++;
            }
            // Dequant and compare
            // Block b covers input elements: need to figure out which row and offset
            // blocks are stored as: [block_within_row * ne1 + row]
            // For ne0=512: 4 blocks per row. block_within_row = b / ne1, row = b % ne1
            int row = b % ne1;
            int bwr = b / ne1;
            int offset = bwr * 128;

            for (int q = 0; q < 128; q++) {
                float d = blk.d4[q / 32];
                float dequant = d * (float)blk.qs[q];
                float orig = h_input[row * ne00 + offset + q];
                float err = fabsf(dequant - orig);
                if (err > max_dequant_err) max_dequant_err = err;
                total_vals++;
                if (blk.qs[q] == 0 && fabsf(orig) > 0.01f) zero_qs++;
            }
        } else {
            // DS4/D2S6: just check non-zero
            bool any_nonzero = false;
            for (int q = 0; q < 128; q++) if (blk.qs[q] != 0) any_nonzero = true;
            if (!any_nonzero) zero_qs++;
        }
    }

    if (nan_scales > 0) { test_fail(kernel_name, "%d NaN scales", nan_scales); return; }
    if (zero_qs > n_blocks / 2) { test_fail(kernel_name, "too many zero blocks: %d/%d", zero_qs, n_blocks); return; }

    if (layout == 0) {
        // D4: check dequant error (Q8_1 should be within 1% for reasonable inputs)
        float max_input = 0;
        for (int i = 0; i < (int)(ne1 * ne00); i++) max_input = fmaxf(max_input, fabsf(h_input[i]));
        float rel_err = max_dequant_err / (max_input + 1e-6f);
        if (rel_err > 0.02f) { test_fail(kernel_name, "dequant rel_err=%.4f > 2%%", (double)rel_err); return; }
        test_pass(kernel_name, "dequant_err=%.2e (%.2f%% of max), %d blocks",
                  (double)max_dequant_err, (double)(rel_err * 100), n_blocks);
    } else {
        test_pass(kernel_name, "%d blocks, qs non-zero", n_blocks);
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

    test_mmq_quantize(mod, "eval_quantize_mmq_q8_1_d4",   0);
    test_mmq_quantize(mod, "eval_quantize_mmq_q8_1_ds4",  1);
    test_mmq_quantize(mod, "eval_quantize_mmq_q8_1_d2s6", 2);

    hipModuleUnload(mod);
    return test_summary();
}
