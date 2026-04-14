// test-kernel-embed.cpp — Unit test for eval_embed_q6k GPU kernel
//
// Loads a GGUF model, extracts the embedding weight GPU pointer,
// launches eval_embed_q6k via HIP module API, and compares output
// against a CPU reference dequantization.
//
// Usage: test-kernel-embed <model.gguf>
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
// Q6_K block layout (matches ggml-common.h, quant-types.h)
// sizeof = 2 + 16 + 64 + 128 = 210 bytes
// ---------------------------------------------------------------------------
#define QK_K 256

struct block_q6_K_cpu {
    uint8_t  ql[128];   // lower 4 bits: QK_K/2
    uint8_t  qh[64];    // upper 2 bits: QK_K/4
    int8_t   scales[16];// per-16-element scales: QK_K/16
    uint16_t d;         // f16 super-block scale
};
static_assert(sizeof(block_q6_K_cpu) == 210, "block_q6_K size must be 210");

// ---------------------------------------------------------------------------
// CPU Q6_K dequantization — mirrors dequantize_block_q6_K in convert.cu
// Produces QK_K floats per super-block, for `nb_blocks` blocks.
// ---------------------------------------------------------------------------
static void cpu_dequant_q6k(const block_q6_K_cpu * blocks, int nb_blocks, float * out) {
    for (int b = 0; b < nb_blocks; b++) {
        const block_q6_K_cpu & blk = blocks[b];
        float d = ggml_fp16_to_fp32((ggml_fp16_t)blk.d);

        // 64 virtual threads per block, each produces 4 values
        for (int tid = 0; tid < 64; tid++) {
            const int ip = tid / 32;       // 0 or 1
            const int il = tid - 32 * ip;  // 0..31
            const int is = 8 * ip + il / 16;

            float * y = out + b * QK_K + 128 * ip + il;

            const uint8_t * ql = blk.ql + 64 * ip + il;
            const uint8_t   qh = blk.qh[32 * ip + il];
            const int8_t  * sc = blk.scales + is;

            y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
            y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
            y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
            y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
        }
    }
}

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

    // ---- Get embedding tensor info ----
    const ggml_tensor * tok_embd = model->tok_embd;
    if (!tok_embd || !tok_embd->data) {
        fprintf(stderr, "FAIL: tok_embd is null or has no GPU data\n");
        llama_model_free(model);
        return 1;
    }

    const int hidden_size = (int)tok_embd->ne[0]; // columns = embedding dim
    const long long embed_stride = (long long)tok_embd->nb[1]; // bytes per row
    const int nb_blocks = hidden_size / QK_K;

    fprintf(stderr, "tok_embd: hidden_size=%d, embed_stride=%lld, nb_blocks=%d\n",
            hidden_size, embed_stride, nb_blocks);

    // Sanity check: for Q6_K the stride should be nb_blocks * 210
    {
        int expected_stride = nb_blocks * (int)sizeof(block_q6_K_cpu);
        if (embed_stride != (long long)expected_stride) {
            fprintf(stderr, "WARNING: expected stride %d but got %lld "
                    "(type may not be Q6_K or HIDDEN_SIZE mismatch)\n",
                    expected_stride, embed_stride);
        }
    }

    const void * embed_gpu = tok_embd->data;

    // Test token id = 1 (BOS for most models)
    const int token_id = 1;

    // ---- Allocate GPU output buffer ----
    float * gpu_out = nullptr;
    HIP_CHECK(hipMalloc(&gpu_out, (size_t)hidden_size * sizeof(float)));
    HIP_CHECK(hipMemset(gpu_out, 0, (size_t)hidden_size * sizeof(float)));

    // ---- Load HIP module and get kernel function ----
    hipModule_t   hip_mod = nullptr;
    hipFunction_t embed_fn = nullptr;

    {
        hipError_t e = hipModuleLoad(&hip_mod, hsaco_path.c_str());
        if (e != hipSuccess) {
            fprintf(stderr, "FAIL: hipModuleLoad(%s) — %s\n",
                    hsaco_path.c_str(), hipGetErrorString(e));
            hipFree(gpu_out);
            llama_model_free(model);
            return 1;
        }
    }
    {
        hipError_t e = hipModuleGetFunction(&embed_fn, hip_mod, "eval_embed_q6k");
        if (e != hipSuccess) {
            fprintf(stderr, "FAIL: hipModuleGetFunction(eval_embed_q6k) — %s\n",
                    hipGetErrorString(e));
            hipModuleUnload(hip_mod);
            hipFree(gpu_out);
            llama_model_free(model);
            return 1;
        }
    }

    // ---- Copy embedding row to GPU (tok_embd lives on CPU!) ----
    void * gpu_embed_staging = nullptr;
    HIP_CHECK(hipMalloc(&gpu_embed_staging, embed_stride));
    {
        const char * cpu_row = (const char *)embed_gpu + (long long)token_id * embed_stride;
        HIP_CHECK(hipMemcpy(gpu_embed_staging, cpu_row, embed_stride, hipMemcpyHostToDevice));
    }

    // ---- Launch kernel ----
    // Grid: nb_blocks = HIDDEN_SIZE/QK_K, Block: 64 threads
    // token_id=0 because we already offset the row
    {
        int zero_token = 0;
        const void * args[] = {
            &gpu_embed_staging,  // const void * embed_weight (GPU copy of single row)
            &embed_stride,       // const long long embed_stride
            &gpu_out,            // float * hidden
            &zero_token          // const int token_id (0 — row already offset)
        };
        hipError_t e = hipModuleLaunchKernel(
            embed_fn,
            (unsigned)nb_blocks, 1, 1,
            64, 1, 1,
            0, nullptr,
            (void **)args, nullptr);
        if (e != hipSuccess) {
            fprintf(stderr, "FAIL: hipModuleLaunchKernel — %s\n", hipGetErrorString(e));
            hipFree(gpu_embed_staging);
            hipModuleUnload(hip_mod);
            hipFree(gpu_out);
            llama_model_free(model);
            return 1;
        }
        HIP_CHECK(hipDeviceSynchronize());
    }
    hipFree(gpu_embed_staging);

    // ---- Copy GPU output to host ----
    std::vector<float> gpu_result(hidden_size);
    HIP_CHECK(hipMemcpy(gpu_result.data(), gpu_out, (size_t)hidden_size * sizeof(float), hipMemcpyDeviceToHost));
    hipFree(gpu_out);

    // ---- Print first 8 values from GPU kernel ----
    fprintf(stderr, "\n[GPU kernel output — first 8 values]\n");
    for (int i = 0; i < 8 && i < hidden_size; i++) {
        fprintf(stderr, "  [%d] = %f\n", i, gpu_result[i]);
    }

    // Non-zero / NaN count
    int gpu_nonzero = 0, gpu_nans = 0;
    for (int i = 0; i < hidden_size; i++) {
        float v = gpu_result[i];
        if (v != v) gpu_nans++;
        else if (v != 0.0f) gpu_nonzero++;
    }
    fprintf(stderr, "  non-zero: %d/%d\n", gpu_nonzero, hidden_size);
    if (gpu_nans > 0) fprintf(stderr, "  WARNING: %d NaN values\n", gpu_nans);

    // ---- CPU reference dequant ----
    // tok_embd is on CPU — direct memcpy, no HIP needed
    std::vector<uint8_t> raw_row(embed_stride);
    memcpy(raw_row.data(),
           (const char *)embed_gpu + (long long)token_id * embed_stride,
           embed_stride);

    const block_q6_K_cpu * blocks = (const block_q6_K_cpu *)raw_row.data();
    std::vector<float> cpu_result(hidden_size, 0.0f);
    cpu_dequant_q6k(blocks, nb_blocks, cpu_result.data());

    fprintf(stderr, "\n[CPU reference output — first 8 values]\n");
    for (int i = 0; i < 8 && i < hidden_size; i++) {
        fprintf(stderr, "  [%d] = %f\n", i, cpu_result[i]);
    }

    // ---- Compare GPU vs CPU ----
    float max_abs_diff = 0.0f;
    int mismatch_count = 0;
    for (int i = 0; i < hidden_size; i++) {
        float diff = fabsf(gpu_result[i] - cpu_result[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
        if (diff > 1e-4f) mismatch_count++;
    }

    fprintf(stderr, "\n[Comparison GPU vs CPU]\n");
    fprintf(stderr, "  max_abs_diff = %e\n", (double)max_abs_diff);
    fprintf(stderr, "  values with diff > 1e-4: %d/%d\n", mismatch_count, hidden_size);

    // ---- Determine PASS/FAIL ----
    bool pass = true;

    if (gpu_nans > 0) {
        fprintf(stderr, "FAIL: GPU output contains NaN\n");
        pass = false;
    }
    if (gpu_nonzero == 0) {
        fprintf(stderr, "FAIL: GPU output is all zeros (kernel did not run?)\n");
        pass = false;
    }
    if (max_abs_diff > 1e-3f) {
        fprintf(stderr, "FAIL: max_abs_diff %.6e exceeds threshold 1e-3\n", (double)max_abs_diff);
        pass = false;
    }

    hipModuleUnload(hip_mod);
    llama_model_free(model);
    llama_backend_free();

    if (pass) {
        printf("PASS: eval_embed_q6k GPU output matches CPU reference "
               "(max_abs_diff=%.2e, non-zero=%d/%d)\n",
               (double)max_abs_diff, gpu_nonzero, hidden_size);
        return 0;
    }
    return 1;
}
