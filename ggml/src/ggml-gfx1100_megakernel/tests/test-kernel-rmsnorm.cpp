// test-kernel-rmsnorm.cpp — Unit test for eval_rmsnorm_q8 GPU kernel
//
// Loads a GGUF model, extracts layer 0 attn_norm weight (GPU pointer),
// creates a known f32 input on GPU, launches eval_rmsnorm_q8 via HIP
// module API, and compares output against a CPU reference RMSNorm.
//
// Usage: test-kernel-rmsnorm <model.gguf>
// The pre-compiled .hsaco must exist at:
//   ~/.cache/gfx1100-megakernel/decode.hip_*.hsaco

#include "llama.h"
#include "ggml.h"

// Internal struct layout only — no method calls
#include "llama-model.h"
#include "llama-hparams.h"

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
// Q8_1 block layout (matches quant-types.h / ggml-common.h)
// sizeof = 4 (half2) + 32 (int8) = 36 bytes
// ---------------------------------------------------------------------------
#define QK8_1 32

struct block_q8_1_cpu {
    uint32_t ds;      // packed half2: d = scale, s = d * sum(qs)
    int8_t   qs[QK8_1];
};
static_assert(sizeof(block_q8_1_cpu) == 4 + QK8_1, "block_q8_1 size must be 36");

// ---------------------------------------------------------------------------
// Find the eval .hsaco in ~/.cache/gfx1100-megakernel/
// Returns the path to the first matching decode.hip_*.hsaco, or empty string.
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
// CPU reference RMSNorm
// scale = 1 / sqrt(mean(x^2) + eps)
// output[i] = scale * input[i] * weight[i]
// ---------------------------------------------------------------------------
static void cpu_rmsnorm(
        const float * input,
        const float * weight,
        float       * output,
        int           n,
        float         eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) sum_sq += input[i] * input[i];
    float scale = 1.0f / sqrtf(sum_sq / (float)n + eps);
    for (int i = 0; i < n; i++) output[i] = scale * input[i] * weight[i];
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];

    // ---- Find .hsaco ----
    std::string hsaco_path = find_hsaco();
    if (hsaco_path.empty()) {
        fprintf(stderr, "FAIL: decode.hip_*.hsaco not found in ~/.cache/gfx1100-megakernel/\n");
        return 1;
    }
    fprintf(stderr, "Using hsaco: %s\n", hsaco_path.c_str());

    // ---- Load model (GPU layers) ----
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "FAIL: could not load model: %s\n", model_path);
        return 1;
    }

    // ---- Get layer 0 attn_norm weight (lives on GPU for GPU-loaded models) ----
    if (model->layers.empty()) {
        fprintf(stderr, "FAIL: model has no layers\n");
        llama_model_free(model);
        return 1;
    }
    const ggml_tensor * attn_norm = model->layers[0].attn_norm;
    if (!attn_norm || !attn_norm->data) {
        fprintf(stderr, "FAIL: layers[0].attn_norm is null or has no data pointer\n");
        llama_model_free(model);
        return 1;
    }

    const int n = (int)attn_norm->ne[0];  // number of elements = HIDDEN_SIZE
    const float eps = model->hparams.f_norm_rms_eps;

    fprintf(stderr, "attn_norm: n=%d, eps=%e, type=%s\n",
            n, (double)eps, ggml_type_name(attn_norm->type));

    // Norm weights must be f32 for this test
    if (attn_norm->type != GGML_TYPE_F32) {
        fprintf(stderr, "FAIL: attn_norm type is %s, expected F32\n",
                ggml_type_name(attn_norm->type));
        llama_model_free(model);
        return 1;
    }

    // attn_norm->data is a GPU pointer (norm weights are on GPU when n_gpu_layers=-1)
    const float * weight_gpu = (const float *)attn_norm->data;

    // ---- Build known host input: input[i] = sinf(i * 0.1f) ----
    std::vector<float> host_input(n);
    for (int i = 0; i < n; i++) host_input[i] = sinf((float)i * 0.1f);

    // ---- Allocate GPU buffers ----
    float        * gpu_input    = nullptr;
    float        * gpu_norm_out = nullptr;
    void         * gpu_q8_out   = nullptr;  // block_q8_1 array
    float        * gpu_residual = nullptr;

    const int n_q8_blocks   = n / QK8_1;
    const size_t q8_bytes   = (size_t)n_q8_blocks * sizeof(block_q8_1_cpu);

    HIP_CHECK(hipMalloc(&gpu_input,    (size_t)n * sizeof(float)));
    HIP_CHECK(hipMalloc(&gpu_norm_out, (size_t)n * sizeof(float)));
    HIP_CHECK(hipMalloc(&gpu_q8_out,   q8_bytes));
    HIP_CHECK(hipMalloc(&gpu_residual, (size_t)n * sizeof(float)));

    // Copy host input to GPU
    HIP_CHECK(hipMemcpy(gpu_input, host_input.data(),
                        (size_t)n * sizeof(float), hipMemcpyHostToDevice));

    // Zero output buffers
    HIP_CHECK(hipMemset(gpu_norm_out, 0, (size_t)n * sizeof(float)));
    HIP_CHECK(hipMemset(gpu_q8_out,   0, q8_bytes));
    HIP_CHECK(hipMemset(gpu_residual, 0, (size_t)n * sizeof(float)));

    // ---- Load HIP module and get kernel function ----
    hipModule_t   hip_mod    = nullptr;
    hipFunction_t rmsnorm_fn = nullptr;

    {
        hipError_t e = hipModuleLoad(&hip_mod, hsaco_path.c_str());
        if (e != hipSuccess) {
            fprintf(stderr, "FAIL: hipModuleLoad(%s) — %s\n",
                    hsaco_path.c_str(), hipGetErrorString(e));
            hipFree(gpu_input);
            hipFree(gpu_norm_out);
            hipFree(gpu_q8_out);
            hipFree(gpu_residual);
            llama_model_free(model);
            return 1;
        }
    }
    {
        hipError_t e = hipModuleGetFunction(&rmsnorm_fn, hip_mod, "eval_rmsnorm_q8");
        if (e != hipSuccess) {
            fprintf(stderr, "FAIL: hipModuleGetFunction(eval_rmsnorm_q8) — %s\n",
                    hipGetErrorString(e));
            hipModuleUnload(hip_mod);
            hipFree(gpu_input);
            hipFree(gpu_norm_out);
            hipFree(gpu_q8_out);
            hipFree(gpu_residual);
            llama_model_free(model);
            return 1;
        }
    }

    // ---- Launch kernel ----
    // Grid: 1 block, Block: 512 threads
    // Signature: eval_rmsnorm_q8(input, weight, norm_out, q8_out, residual, n)
    {
        const void * args[] = {
            &gpu_input,    // const float * input
            &weight_gpu,   // const float * weight
            &gpu_norm_out, // float * norm_out
            &gpu_q8_out,   // block_q8_1 * q8_out
            &gpu_residual, // float * residual
            &n             // const int n
        };
        hipError_t e = hipModuleLaunchKernel(
            rmsnorm_fn,
            1, 1, 1,    // grid: 1 block
            512, 1, 1,  // block: 512 threads
            0, nullptr,
            (void **)args, nullptr);
        if (e != hipSuccess) {
            fprintf(stderr, "FAIL: hipModuleLaunchKernel — %s\n", hipGetErrorString(e));
            hipModuleUnload(hip_mod);
            hipFree(gpu_input);
            hipFree(gpu_norm_out);
            hipFree(gpu_q8_out);
            hipFree(gpu_residual);
            llama_model_free(model);
            return 1;
        }
        HIP_CHECK(hipDeviceSynchronize());
    }

    // ---- Copy GPU outputs back to host ----
    std::vector<float> gpu_norm_result(n);
    std::vector<float> gpu_residual_result(n);

    HIP_CHECK(hipMemcpy(gpu_norm_result.data(), gpu_norm_out,
                        (size_t)n * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gpu_residual_result.data(), gpu_residual,
                        (size_t)n * sizeof(float), hipMemcpyDeviceToHost));

    // ---- Print first 8 values from GPU kernel ----
    fprintf(stderr, "\n[GPU norm_out — first 8 values]\n");
    for (int i = 0; i < 8 && i < n; i++) {
        fprintf(stderr, "  [%d] = %f\n", i, gpu_norm_result[i]);
    }

    // Non-zero / NaN counts for norm_out
    int gpu_nonzero = 0, gpu_nans = 0;
    for (int i = 0; i < n; i++) {
        float v = gpu_norm_result[i];
        if (v != v) gpu_nans++;
        else if (v != 0.0f) gpu_nonzero++;
    }
    fprintf(stderr, "  non-zero: %d/%d\n", gpu_nonzero, n);
    if (gpu_nans > 0) fprintf(stderr, "  WARNING: %d NaN values in norm_out\n", gpu_nans);

    // ---- CPU reference ----
    // Fetch the norm weight from GPU so we can compute CPU reference
    std::vector<float> host_weight(n);
    HIP_CHECK(hipMemcpy(host_weight.data(), weight_gpu,
                        (size_t)n * sizeof(float), hipMemcpyDeviceToHost));

    std::vector<float> cpu_norm_result(n);
    cpu_rmsnorm(host_input.data(), host_weight.data(), cpu_norm_result.data(), n, eps);

    fprintf(stderr, "\n[CPU reference norm_out — first 8 values]\n");
    for (int i = 0; i < 8 && i < n; i++) {
        fprintf(stderr, "  [%d] = %f\n", i, cpu_norm_result[i]);
    }

    // ---- Compare norm_out: GPU vs CPU ----
    float max_diff_norm = 0.0f;
    int   mismatch_norm = 0;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(gpu_norm_result[i] - cpu_norm_result[i]);
        if (diff > max_diff_norm) max_diff_norm = diff;
        if (diff > 1e-4f) mismatch_norm++;
    }

    fprintf(stderr, "\n[Comparison norm_out GPU vs CPU]\n");
    fprintf(stderr, "  max_abs_diff = %e\n", (double)max_diff_norm);
    fprintf(stderr, "  values with diff > 1e-4: %d/%d\n", mismatch_norm, n);

    // ---- Verify residual == input (exact copy) ----
    int residual_mismatch = 0;
    for (int i = 0; i < n; i++) {
        if (gpu_residual_result[i] != host_input[i]) {
            residual_mismatch++;
        }
    }
    fprintf(stderr, "\n[Residual == Input check]\n");
    fprintf(stderr, "  mismatches (exact): %d/%d\n", residual_mismatch, n);

    // ---- Cleanup GPU buffers ----
    hipFree(gpu_input);
    hipFree(gpu_norm_out);
    hipFree(gpu_q8_out);
    hipFree(gpu_residual);
    hipModuleUnload(hip_mod);
    llama_model_free(model);
    llama_backend_free();

    // ---- PASS / FAIL ----
    bool pass = true;

    if (gpu_nans > 0) {
        fprintf(stderr, "FAIL: norm_out contains %d NaN values\n", gpu_nans);
        pass = false;
    }
    if (gpu_nonzero == 0) {
        fprintf(stderr, "FAIL: norm_out is all zeros (kernel did not run?)\n");
        pass = false;
    }
    if (max_diff_norm > 1e-4f) {
        fprintf(stderr, "FAIL: norm_out max_abs_diff %.6e exceeds threshold 1e-4\n",
                (double)max_diff_norm);
        pass = false;
    }
    if (residual_mismatch > 0) {
        fprintf(stderr, "FAIL: residual != input (%d/%d elements differ)\n",
                residual_mismatch, n);
        pass = false;
    }

    if (pass) {
        printf("PASS: eval_rmsnorm_q8 GPU output matches CPU reference "
               "(norm max_abs_diff=%.2e, residual exact, non-zero=%d/%d)\n",
               (double)max_diff_norm, gpu_nonzero, n);
        return 0;
    }
    return 1;
}
