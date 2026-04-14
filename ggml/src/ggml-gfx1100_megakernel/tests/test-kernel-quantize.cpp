// test-kernel-quantize.cpp — Unit test for eval_quantize_q8 GPU kernel
//
// Creates a known f32 input array (sinf pattern), copies to GPU,
// launches eval_quantize_q8 via HIP module API, and compares Q8_1
// output against a CPU reference quantization.
//
// No model required — purely synthetic input.
//
// Usage: test-kernel-quantize
// The pre-compiled .hsaco must exist at:
//   ~/.cache/gfx1100-megakernel/decode.hip_*.hsaco

#include "ggml.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// HIP error check macro
// ---------------------------------------------------------------------------
#define HIP_CHECK(call)                                                    \
    do {                                                                   \
        hipError_t _e = (call);                                            \
        if (_e != hipSuccess) {                                            \
            fprintf(stderr, "HIP error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, hipGetErrorString(_e));             \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// Q8_1 block layout on host (matches ggml-common.h block_q8_1)
// ds[0] = f16(d),  ds[1] = f16(raw_sum)
// qs[0..31] = quantized int8 values
// sizeof = 4 + 32 = 36 bytes
// ---------------------------------------------------------------------------
#define QK8_1 32

struct block_q8_1_host {
    uint16_t ds[2];      // ds[0] = f16(d), ds[1] = f16(sum)
    int8_t   qs[QK8_1]; // quantized int8 values
};
static_assert(sizeof(block_q8_1_host) == 36, "block_q8_1_host must be 36 bytes");

// ---------------------------------------------------------------------------
// Find the eval .hsaco in ~/.cache/gfx1100-megakernel/
// Returns path to the first matching decode.hip_*.hsaco, or empty string.
// ---------------------------------------------------------------------------
static std::string find_hsaco() {
    const char * home = getenv("USERPROFILE");
    if (!home) home = getenv("HOME");
    if (!home) {
        fprintf(stderr, "Cannot determine home directory (USERPROFILE/HOME not set)\n");
        return "";
    }
    fs::path cache_dir = fs::path(home) / ".cache" / "gfx1100-megakernel";
    if (!fs::is_directory(cache_dir)) {
        fprintf(stderr, "Cache directory not found: %s\n", cache_dir.string().c_str());
        return "";
    }
    for (const auto & entry : fs::directory_iterator(cache_dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        // Match: decode.hip_*.hsaco
        if (name.size() > 4 &&
            name.substr(name.size() - 6) == ".hsaco" &&
            name.rfind("decode.hip_", 0) == 0) {
            return entry.path().string();
        }
    }
    return "";
}

// ---------------------------------------------------------------------------
// CPU reference Q8_1 quantization
// For each group of 32 floats: amax, d = amax/127, quantize each to int8
// ds.x = d (scale), ds.y = raw sum of the input values
// ---------------------------------------------------------------------------
static void cpu_quantize_q8(
        const float          * input,
        int                    n_blocks,
        std::vector<float>   & out_d,
        std::vector<float>   & out_sum,
        std::vector<std::vector<int8_t>> & out_qs) {
    out_d.resize(n_blocks);
    out_sum.resize(n_blocks);
    out_qs.resize(n_blocks, std::vector<int8_t>(QK8_1));

    for (int ib = 0; ib < n_blocks; ib++) {
        float amax = 0.0f;
        float sum  = 0.0f;
        for (int j = 0; j < QK8_1; j++) {
            float v = input[ib * QK8_1 + j];
            amax = fmaxf(amax, fabsf(v));
            sum += v;
        }
        float d = amax / 127.0f;
        out_d[ib]   = d;
        out_sum[ib] = sum;
        for (int j = 0; j < QK8_1; j++) {
            float v = input[ib * QK8_1 + j];
            out_qs[ib][j] = (amax == 0.0f) ? 0 : (int8_t)roundf(v / d);
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int /*argc*/, char ** /*argv*/) {
    // ---- Find .hsaco ----
    std::string hsaco_path = find_hsaco();
    if (hsaco_path.empty()) {
        fprintf(stderr, "FAIL: decode.hip_*.hsaco not found in ~/.cache/gfx1100-megakernel/\n");
        return 1;
    }
    fprintf(stderr, "Using hsaco: %s\n", hsaco_path.c_str());

    // ---- Build known host input ----
    const int n         = 2048;
    const int n_blocks  = n / QK8_1;  // 64 blocks
    const size_t q8_bytes = (size_t)n_blocks * sizeof(block_q8_1_host); // 64 * 36 = 2304

    std::vector<float> host_input(n);
    for (int i = 0; i < n; i++) {
        host_input[i] = sinf((float)i * 0.1f) * 10.0f;
    }

    fprintf(stderr, "Input: n=%d, n_blocks=%d, q8_bytes=%zu\n", n, n_blocks, q8_bytes);

    // ---- Allocate GPU buffers ----
    float * gpu_input  = nullptr;
    void  * gpu_output = nullptr;

    HIP_CHECK(hipMalloc(&gpu_input,  (size_t)n * sizeof(float)));
    HIP_CHECK(hipMalloc(&gpu_output, q8_bytes));

    HIP_CHECK(hipMemcpy(gpu_input, host_input.data(),
                        (size_t)n * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(gpu_output, 0, q8_bytes));

    // ---- Load HIP module and get kernel function ----
    hipModule_t   hip_mod    = nullptr;
    hipFunction_t quantize_fn = nullptr;

    {
        hipError_t e = hipModuleLoad(&hip_mod, hsaco_path.c_str());
        if (e != hipSuccess) {
            fprintf(stderr, "FAIL: hipModuleLoad(%s) — %s\n",
                    hsaco_path.c_str(), hipGetErrorString(e));
            hipFree(gpu_input);
            hipFree(gpu_output);
            return 1;
        }
    }
    {
        hipError_t e = hipModuleGetFunction(&quantize_fn, hip_mod, "eval_quantize_q8");
        if (e != hipSuccess) {
            fprintf(stderr, "FAIL: hipModuleGetFunction(eval_quantize_q8) — %s\n",
                    hipGetErrorString(e));
            hipModuleUnload(hip_mod);
            hipFree(gpu_input);
            hipFree(gpu_output);
            return 1;
        }
    }

    // ---- Launch kernel ----
    // Grid: (n + 511) / 512 blocks, Block: 512 threads
    // Signature: eval_quantize_q8(const float * input, block_q8_1 * output, const int n)
    {
        const unsigned grid_x = ((unsigned)n + 511u) / 512u;
        const void * args[] = {
            &gpu_input,   // const float * input
            &gpu_output,  // block_q8_1 * output
            &n            // const int n
        };
        hipError_t e = hipModuleLaunchKernel(
            quantize_fn,
            grid_x, 1, 1,  // grid
            512, 1, 1,      // block: 512 threads (16 warps of 32)
            0, nullptr,
            (void **)args, nullptr);
        if (e != hipSuccess) {
            fprintf(stderr, "FAIL: hipModuleLaunchKernel — %s\n", hipGetErrorString(e));
            hipModuleUnload(hip_mod);
            hipFree(gpu_input);
            hipFree(gpu_output);
            return 1;
        }
        HIP_CHECK(hipDeviceSynchronize());
    }

    // ---- Copy Q8_1 output back to host ----
    std::vector<uint8_t> raw_output(q8_bytes);
    HIP_CHECK(hipMemcpy(raw_output.data(), gpu_output, q8_bytes, hipMemcpyDeviceToHost));

    const block_q8_1_host * gpu_blocks =
        reinterpret_cast<const block_q8_1_host *>(raw_output.data());

    // ---- Print first 4 GPU blocks ----
    fprintf(stderr, "\n[GPU Q8_1 output — first 4 blocks]\n");
    for (int ib = 0; ib < 4 && ib < n_blocks; ib++) {
        float d   = ggml_fp16_to_fp32((ggml_fp16_t)gpu_blocks[ib].ds[0]);
        float s   = ggml_fp16_to_fp32((ggml_fp16_t)gpu_blocks[ib].ds[1]);
        fprintf(stderr, "  block[%d]: d=%.6f  sum=%.4f  qs[0..3]=%d %d %d %d\n",
                ib, (double)d, (double)s,
                gpu_blocks[ib].qs[0], gpu_blocks[ib].qs[1],
                gpu_blocks[ib].qs[2], gpu_blocks[ib].qs[3]);
    }

    // ---- CPU reference ----
    std::vector<float> cpu_d, cpu_sum;
    std::vector<std::vector<int8_t>> cpu_qs;
    cpu_quantize_q8(host_input.data(), n_blocks, cpu_d, cpu_sum, cpu_qs);

    fprintf(stderr, "\n[CPU reference — first 4 blocks]\n");
    for (int ib = 0; ib < 4 && ib < n_blocks; ib++) {
        fprintf(stderr, "  block[%d]: d=%.6f  sum=%.4f  qs[0..3]=%d %d %d %d\n",
                ib, (double)cpu_d[ib], (double)cpu_sum[ib],
                cpu_qs[ib][0], cpu_qs[ib][1], cpu_qs[ib][2], cpu_qs[ib][3]);
    }

    // ---- Compare GPU vs CPU ----
    float max_diff_d   = 0.0f;
    float max_diff_sum = 0.0f;
    int   qs_mismatch  = 0;
    int   d_fail       = 0;
    int   sum_fail     = 0;

    for (int ib = 0; ib < n_blocks; ib++) {
        float gpu_d_val   = ggml_fp16_to_fp32((ggml_fp16_t)gpu_blocks[ib].ds[0]);
        float gpu_sum_val = ggml_fp16_to_fp32((ggml_fp16_t)gpu_blocks[ib].ds[1]);

        float diff_d   = fabsf(gpu_d_val   - cpu_d[ib]);
        float diff_sum = fabsf(gpu_sum_val - cpu_sum[ib]);

        if (diff_d   > max_diff_d)   max_diff_d   = diff_d;
        if (diff_sum > max_diff_sum) max_diff_sum = diff_sum;
        if (diff_d   > 1e-3f) d_fail++;
        if (diff_sum > 1e-1f) sum_fail++;

        for (int j = 0; j < QK8_1; j++) {
            if (gpu_blocks[ib].qs[j] != cpu_qs[ib][j]) {
                qs_mismatch++;
                if (qs_mismatch <= 4) {
                    fprintf(stderr,
                            "  qs mismatch at block[%d][%d]: gpu=%d cpu=%d\n",
                            ib, j, (int)gpu_blocks[ib].qs[j], (int)cpu_qs[ib][j]);
                }
            }
        }
    }

    fprintf(stderr, "\n[Comparison GPU vs CPU]\n");
    fprintf(stderr, "  d   max_abs_diff = %.6e  (blocks failing >1e-3: %d/%d)\n",
            (double)max_diff_d,   d_fail,   n_blocks);
    fprintf(stderr, "  sum max_abs_diff = %.6e  (blocks failing >1e-1: %d/%d)\n",
            (double)max_diff_sum, sum_fail, n_blocks);
    fprintf(stderr, "  qs mismatches (exact): %d / %d\n",
            qs_mismatch, n_blocks * QK8_1);

    // ---- Cleanup ----
    hipFree(gpu_input);
    hipFree(gpu_output);
    hipModuleUnload(hip_mod);

    // ---- PASS / FAIL ----
    bool pass = true;

    if (d_fail > 0) {
        fprintf(stderr, "FAIL: %d blocks have d error > 1e-3 (max=%.6e)\n",
                d_fail, (double)max_diff_d);
        pass = false;
    }
    if (sum_fail > 0) {
        fprintf(stderr, "FAIL: %d blocks have sum error > 1e-1 (max=%.6e)\n",
                sum_fail, (double)max_diff_sum);
        pass = false;
    }
    if (qs_mismatch > 0) {
        fprintf(stderr, "FAIL: %d qs values do not match CPU reference exactly\n",
                qs_mismatch);
        pass = false;
    }

    if (pass) {
        printf("PASS: eval_quantize_q8 GPU output matches CPU reference "
               "(d_max=%.2e, sum_max=%.2e, qs exact)\n",
               (double)max_diff_d, (double)max_diff_sum);
        return 0;
    }
    return 1;
}
