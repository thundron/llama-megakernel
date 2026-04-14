// test-kernel-matvec.cpp — Unit test for eval_matvec_q4k GPU kernel
//
// Loads a GGUF model (Llama 1B recommended), extracts layer 0 wq weight
// (Q4_K on GPU), creates synthetic f32 input, quantizes to Q8_1 via
// eval_quantize_q8 kernel, then runs eval_matvec_q4k and compares output
// against a CPU reference built from ggml's own dequantization.
//
// Usage: test-kernel-matvec <model.gguf>
// The pre-compiled .hsaco must exist at:
//   ~/.cache/gfx1100-megakernel/decode.hip_*.hsaco

#include "llama.h"
#include "ggml.h"
#include "ggml-quants.h"

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
    uint32_t ds;         // packed half2: d = scale, s = d * sum(qs)
    int8_t   qs[QK8_1];
};
static_assert(sizeof(block_q8_1_cpu) == 4 + QK8_1, "block_q8_1 size must be 36");

// ---------------------------------------------------------------------------
// Q4_K block layout (matches ggml-common.h)
// sizeof = 4 (half2) + 12 (scales) + 128 (qs) = 144 bytes
// ---------------------------------------------------------------------------
#define QK_K       256
#define K_SCALE_SIZE 12

struct block_q4_K_cpu {
    uint32_t dm;                    // packed half2: d (super-scale) and dmin
    uint8_t  scales[K_SCALE_SIZE];  // scales and mins, 6-bit each
    uint8_t  qs[QK_K / 2];         // 4-bit quants
};
static_assert(sizeof(block_q4_K_cpu) == 4 + K_SCALE_SIZE + QK_K / 2, "block_q4_K size must be 144");

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
        if (name.size() > 6 &&
            name.substr(name.size() - 6) == ".hsaco" &&
            name.rfind("decode.hip_", 0) == 0) {
            return entry.path().string();
        }
    }
    return "";
}

// ---------------------------------------------------------------------------
// Extract 6-bit sub-scale and sub-min for Q4_K sub-block j (0..7).
// Directly mirrors ggml-quants.c:get_scale_min_k4:
//
//   if (j < 4):  d = q[j] & 63;     m = q[j+4] & 63;
//   else:         d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
//                 m = (q[j+4] >>  4) | ((q[j  ] >> 6) << 4);
// ---------------------------------------------------------------------------
static inline void get_scale_min_k4_cpu(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j]   & 63;
        *m = q[j+4] & 63;
    } else {
        *d = (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>    4) | ((q[j  ] >> 6) << 4);
    }
}

// ---------------------------------------------------------------------------
// CPU Q4_K dequantization for one super-block into `out[QK_K]` floats.
// Mirrors ggml-quants.c:dequantize_row_q4_K loop structure exactly:
//   for j in {0, 64, 128, 192}:   (4 groups of 64)
//     is = j/64 * 2
//     get_scale_min_k4(is+0) → sc0,m0 → d1=d*sc0, m1=min*m0
//     get_scale_min_k4(is+1) → sc1,m1 → d2=d*sc1, m2=min*m1
//     for l in 0..31: out[j+l]    = d1 * (q[j/2+l] & 0xf) - m1
//     for l in 0..31: out[j+32+l] = d2 * (q[j/2+l] >> 4)  - m2
// ---------------------------------------------------------------------------
static void cpu_dequant_q4k_block(const block_q4_K_cpu & blk, float * out) {
    uint16_t hd    = (uint16_t)(blk.dm & 0xffff);
    uint16_t hdmin = (uint16_t)(blk.dm >> 16);
    float d   = ggml_fp16_to_fp32((ggml_fp16_t)hd);
    float dmin = ggml_fp16_to_fp32((ggml_fp16_t)hdmin);

    const uint8_t * q  = blk.qs;
    float         * y  = out;
    int is = 0;
    uint8_t sc, m;

    for (int j = 0; j < QK_K; j += 64) {
        get_scale_min_k4_cpu(is + 0, blk.scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        get_scale_min_k4_cpu(is + 1, blk.scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;

        for (int l = 0; l < 32; l++) y[l]    = d1 * (q[l] & 0x0F) - m1;
        for (int l = 0; l < 32; l++) y[l+32] = d2 * (q[l] >>    4) - m2;

        q  += 32;
        y  += 64;
        is += 2;
    }
}

// ---------------------------------------------------------------------------
// CPU reference matvec: Q4_K weight matrix × f32 input → f32 output
// For each output row: dequant Q4_K row, dot product with input.
// weight_bytes: raw Q4_K bytes of the full matrix (fetched from GPU)
// nb1: row stride in bytes (= in_dim / QK_K * sizeof(block_q4_K_cpu))
// ---------------------------------------------------------------------------
static void cpu_matvec_q4k(
        const uint8_t * weight_bytes,
        long long       nb1,
        const float   * input,
        float         * output,
        int             in_dim,
        int             out_dim) {
    const int blocks_per_row = in_dim / QK_K;
    std::vector<float> row_f32(in_dim);

    for (int row = 0; row < out_dim; row++) {
        const block_q4_K_cpu * row_blocks =
            (const block_q4_K_cpu *)((const char *)weight_bytes + (long long)row * nb1);

        // Dequant entire row
        for (int b = 0; b < blocks_per_row; b++) {
            cpu_dequant_q4k_block(row_blocks[b], row_f32.data() + b * QK_K);
        }

        // Dot product with f32 input
        double acc = 0.0;
        for (int i = 0; i < in_dim; i++) {
            acc += (double)row_f32[i] * (double)input[i];
        }
        output[row] = (float)acc;
    }
}

// ---------------------------------------------------------------------------
// Helper: load a kernel function from the HIP module
// ---------------------------------------------------------------------------
static hipFunction_t load_kernel(hipModule_t mod, const char * name) {
    hipFunction_t fn = nullptr;
    hipError_t e = hipModuleGetFunction(&fn, mod, name);
    if (e != hipSuccess) {
        fprintf(stderr, "FAIL: hipModuleGetFunction(%s) — %s\n", name, hipGetErrorString(e));
        return nullptr;
    }
    return fn;
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

    // ---- Load model (all layers to GPU) ----
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "FAIL: could not load model: %s\n", model_path);
        return 1;
    }

    // ---- Get layer 0 wq weight ----
    if (model->layers.empty()) {
        fprintf(stderr, "FAIL: model has no layers\n");
        llama_model_free(model);
        return 1;
    }
    // Test ffn_down (8192 -> 2048) to catch bugs with larger input dims
    const ggml_tensor * wq = model->layers[0].ffn_down;
    if (!wq || !wq->data) {
        fprintf(stderr, "FAIL: layers[0].ffn_down is null or has no data pointer\n");
        llama_model_free(model);
        return 1;
    }

    if (wq->type != GGML_TYPE_Q4_K) {
        fprintf(stderr, "FAIL: layers[0].ffn_down type is %s, expected Q4_K\n",
                ggml_type_name(wq->type));
        llama_model_free(model);
        return 1;
    }

    const int in_dim  = (int)wq->ne[0];
    const int out_dim = (int)wq->ne[1];
    const long long nb1 = (long long)wq->nb[1];

    fprintf(stderr, "ffn_down: type=Q4_K, in_dim=%d, out_dim=%d, nb1=%lld\n",
            in_dim, out_dim, nb1);

    // Validate stride: should be (in_dim / QK_K) * sizeof(block_q4_K_cpu) = 8 * 144 = 1152
    {
        long long expected_nb1 = (long long)(in_dim / QK_K) * (long long)sizeof(block_q4_K_cpu);
        if (nb1 != expected_nb1) {
            fprintf(stderr, "WARNING: expected nb1=%lld but got %lld\n", expected_nb1, nb1);
        }
    }

    const int stride_row = (int)(nb1 / sizeof(block_q4_K_cpu));
    fprintf(stderr, "  stride_row (Q4_K blocks per row) = %d\n", stride_row);

    // wq->data is a GPU pointer (weight was loaded to GPU)
    const void * weight_gpu = wq->data;

    // ---- Build synthetic f32 input on host: input[i] = sinf(i * 0.1f) ----
    std::vector<float> host_input(in_dim);
    for (int i = 0; i < in_dim; i++) {
        host_input[i] = sinf((float)i * 0.1f);
    }

    // ---- Allocate GPU buffers ----
    float       * gpu_input   = nullptr;
    void        * gpu_q8_in   = nullptr;  // block_q8_1 array
    float       * gpu_output  = nullptr;

    const int n_q8_blocks  = in_dim / QK8_1;
    const size_t q8_bytes  = (size_t)n_q8_blocks * sizeof(block_q8_1_cpu);
    const size_t out_bytes = (size_t)out_dim * sizeof(float);

    HIP_CHECK(hipMalloc(&gpu_input,  (size_t)in_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&gpu_q8_in,  q8_bytes));
    HIP_CHECK(hipMalloc(&gpu_output, out_bytes));

    HIP_CHECK(hipMemcpy(gpu_input, host_input.data(),
                        (size_t)in_dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(gpu_q8_in,  0, q8_bytes));
    HIP_CHECK(hipMemset(gpu_output, 0, out_bytes));

    // ---- Load HIP module and kernels ----
    hipModule_t   hip_mod     = nullptr;
    hipFunction_t quantize_fn = nullptr;
    hipFunction_t matvec_fn   = nullptr;

    {
        hipError_t e = hipModuleLoad(&hip_mod, hsaco_path.c_str());
        if (e != hipSuccess) {
            fprintf(stderr, "FAIL: hipModuleLoad(%s) — %s\n",
                    hsaco_path.c_str(), hipGetErrorString(e));
            hipFree(gpu_input);
            hipFree(gpu_q8_in);
            hipFree(gpu_output);
            llama_model_free(model);
            return 1;
        }
    }

    quantize_fn = load_kernel(hip_mod, "eval_quantize_q8");
    matvec_fn   = load_kernel(hip_mod, "eval_matvec_q4k");

    if (!quantize_fn || !matvec_fn) {
        hipModuleUnload(hip_mod);
        hipFree(gpu_input);
        hipFree(gpu_q8_in);
        hipFree(gpu_output);
        llama_model_free(model);
        return 1;
    }

    // ---- Step 1: Quantize f32 input to Q8_1 on GPU ----
    // eval_quantize_q8(const float * input, block_q8_1 * output, const int n)
    // Grid: (n + 511) / 512 blocks, Block: 512 threads
    {
        const unsigned grid_x = ((unsigned)in_dim + 511u) / 512u;
        const void * args[] = {
            &gpu_input,   // const float * input
            &gpu_q8_in,   // block_q8_1 * output
            &in_dim       // const int n
        };
        hipError_t e = hipModuleLaunchKernel(
            quantize_fn,
            grid_x, 1, 1,
            512, 1, 1,
            0, nullptr,
            (void **)args, nullptr);
        if (e != hipSuccess) {
            fprintf(stderr, "FAIL: hipModuleLaunchKernel(eval_quantize_q8) — %s\n",
                    hipGetErrorString(e));
            hipModuleUnload(hip_mod);
            hipFree(gpu_input);
            hipFree(gpu_q8_in);
            hipFree(gpu_output);
            llama_model_free(model);
            return 1;
        }
        HIP_CHECK(hipDeviceSynchronize());
    }

    // ---- Step 2: Run eval_matvec_q4k ----
    // eval_matvec_q4k(weight, weight_stride_bytes, q8_input, output, in_dim, out_dim)
    // Grid: out_dim blocks, Block: 128 threads
    {
        const void * weight_ptr = weight_gpu;
        const long long stride  = nb1;
        const void * q8_ptr     = gpu_q8_in;
        const void * args[] = {
            &weight_ptr,  // const void * weight
            &stride,      // const long long weight_stride_bytes
            &q8_ptr,      // const block_q8_1 * q8_input
            &gpu_output,  // float * output
            &in_dim,      // const int in_dim
            &out_dim      // const int out_dim
        };
        hipError_t e = hipModuleLaunchKernel(
            matvec_fn,
            (unsigned)out_dim, 1, 1,
            32, 4, 1,  // 2D block: (warp_size=32, nwarps=4)
            0, nullptr,
            (void **)args, nullptr);
        if (e != hipSuccess) {
            fprintf(stderr, "FAIL: hipModuleLaunchKernel(eval_matvec_q4k) — %s\n",
                    hipGetErrorString(e));
            hipModuleUnload(hip_mod);
            hipFree(gpu_input);
            hipFree(gpu_q8_in);
            hipFree(gpu_output);
            llama_model_free(model);
            return 1;
        }
        HIP_CHECK(hipDeviceSynchronize());
    }

    // ---- Copy GPU output to host ----
    std::vector<float> gpu_result(out_dim);
    HIP_CHECK(hipMemcpy(gpu_result.data(), gpu_output, out_bytes, hipMemcpyDeviceToHost));

    // ---- Print first 8 GPU output values ----
    fprintf(stderr, "\n[GPU output — first 8 values]\n");
    for (int i = 0; i < 8 && i < out_dim; i++) {
        fprintf(stderr, "  output[%d] = %f\n", i, gpu_result[i]);
    }

    // Non-zero / NaN stats
    int gpu_nonzero = 0, gpu_nans = 0;
    float gpu_min = 1e38f, gpu_max = -1e38f;
    for (int i = 0; i < out_dim; i++) {
        float v = gpu_result[i];
        if (v != v) {
            gpu_nans++;
        } else {
            if (v != 0.0f) gpu_nonzero++;
            if (v < gpu_min) gpu_min = v;
            if (v > gpu_max) gpu_max = v;
        }
    }
    fprintf(stderr, "  non-zero: %d/%d, NaNs: %d\n", gpu_nonzero, out_dim, gpu_nans);
    fprintf(stderr, "  magnitude: min=%.6f  max=%.6f\n", (double)gpu_min, (double)gpu_max);

    // ---- CPU reference ----
    // Fetch raw Q4_K weight bytes from GPU
    const size_t weight_bytes_total = (size_t)out_dim * (size_t)nb1;
    std::vector<uint8_t> host_weight_raw(weight_bytes_total);
    HIP_CHECK(hipMemcpy(host_weight_raw.data(), weight_gpu,
                        weight_bytes_total, hipMemcpyDeviceToHost));

    // Use ggml's own dequant for authoritative CPU reference
    std::vector<float> cpu_result(out_dim);
    {
        std::vector<float> row_f32(in_dim);
        for (int row = 0; row < out_dim; row++) {
            const void * row_data = host_weight_raw.data() + (size_t)row * nb1;
            dequantize_row_q4_K((const block_q4_K *)row_data, row_f32.data(), in_dim);
            double acc = 0.0;
            for (int i = 0; i < in_dim; i++) acc += (double)row_f32[i] * (double)host_input[i];
            cpu_result[row] = (float)acc;
        }
    }

    fprintf(stderr, "\n[CPU reference — first 8 values]\n");
    for (int i = 0; i < 8 && i < out_dim; i++) {
        fprintf(stderr, "  output[%d] = %f\n", i, cpu_result[i]);
    }

    // ---- Compare GPU vs CPU ----
    float max_abs_diff  = 0.0f;
    float max_rel_diff  = 0.0f;  // relative to |cpu| when cpu != 0
    int   large_diff    = 0;
    float tol           = 0.5f;  // Q4_K quantization + Q8_1 re-quantization: expect ~1% error

    for (int i = 0; i < out_dim; i++) {
        float diff = fabsf(gpu_result[i] - cpu_result[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
        float cpu_abs = fabsf(cpu_result[i]);
        if (cpu_abs > 1e-6f) {
            float rel = diff / cpu_abs;
            if (rel > max_rel_diff) max_rel_diff = rel;
        }
        if (diff > tol) large_diff++;
    }

    fprintf(stderr, "\n[Comparison GPU vs CPU]\n");
    fprintf(stderr, "  max_abs_diff = %.6e\n", (double)max_abs_diff);
    fprintf(stderr, "  max_rel_diff = %.6e\n", (double)max_rel_diff);
    fprintf(stderr, "  values with abs_diff > %.2f: %d/%d\n",
            (double)tol, large_diff, out_dim);

    // ---- Cleanup ----
    hipFree(gpu_input);
    hipFree(gpu_q8_in);
    hipFree(gpu_output);
    hipModuleUnload(hip_mod);
    llama_model_free(model);
    llama_backend_free();

    // ---- PASS / FAIL ----
    bool pass = true;

    if (gpu_nans > 0) {
        fprintf(stderr, "FAIL: GPU output contains %d NaN values\n", gpu_nans);
        pass = false;
    }
    if (gpu_nonzero == 0) {
        fprintf(stderr, "FAIL: GPU output is all zeros (kernel did not run?)\n");
        pass = false;
    }
    // All values identical → likely a bug (e.g. wrong row index)
    if (gpu_max - gpu_min < 1e-6f && out_dim > 1) {
        fprintf(stderr, "FAIL: GPU output range is ~0 (all values identical?)\n");
        pass = false;
    }
    // Relative error: Q4_K × Q8_1 pipeline introduces two quantization steps.
    // ~5% relative error is a reasonable upper bound.
    if (max_rel_diff > 0.05f) {
        fprintf(stderr, "FAIL: max relative diff %.6e exceeds 5%% threshold\n",
                (double)max_rel_diff);
        pass = false;
    }

    if (pass) {
        printf("PASS: eval_matvec_q4k GPU output matches CPU reference "
               "(max_abs=%.2e, max_rel=%.2e, non-zero=%d/%d)\n",
               (double)max_abs_diff, (double)max_rel_diff, gpu_nonzero, out_dim);
        return 0;
    }
    return 1;
}
