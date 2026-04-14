// test-kernel-qknorm.cpp — Unit test for eval_qk_norm_rope_kv_write GPU kernel
//
// Loads a Llama 1B model, creates synthetic Q and KV projections on GPU,
// launches eval_qk_norm_rope_kv_write via HIP module API, and validates:
//   1. Q proj (after RMSNorm + RoPE) matches CPU reference within 1e-3
//   2. K cache at position 5 has non-zero values (KV was written)
//   3. V cache at position 5 has non-zero values (V was written)
//   4. No NaN values anywhere
//
// Usage: test-kernel-qknorm <model.gguf>
// The pre-compiled .hsaco must exist at:
//   ~/.cache/gfx1100-megakernel/decode.hip_*.hsaco

#include "llama.h"
#include "ggml.h"

// Internal struct layout only — no method calls
#include "llama-model.h"
#include "llama-hparams.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

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
// Model constants for Llama 1B
// These MUST match what the .hsaco was compiled with.
// ---------------------------------------------------------------------------
static const int FA_N_Q_HEADS  = 32;
static const int FA_N_KV_HEADS = 8;
static const int FA_HEAD_DIM   = 64;
static const int FA_ROPE_DIM   = 64;
static const float FA_ROPE_THETA = 500000.0f;
static const float NORM_EPS      = 1e-5f;
// FA_HAS_GATED_ATTN = 0 → Q stride is FA_HEAD_DIM (not 2*FA_HEAD_DIM)

// Derived sizes
static const int FA_Q_SIZE    = FA_N_Q_HEADS * FA_HEAD_DIM;   // 32*64 = 2048
static const int FA_QPROJ_SIZE = FA_Q_SIZE;                   // no gated attn
static const int FA_KV_SIZE   = FA_N_KV_HEADS * FA_HEAD_DIM;  // 8*64 = 512
// kv_proj: [FA_KV_SIZE*2] = 1024 floats (K then V)

// ---------------------------------------------------------------------------
// Find the eval .hsaco in ~/.cache/gfx1100-megakernel/
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
// CPU reference: RMSNorm for one head
// scale = rsqrt(mean(x^2) + eps), output[i] = x[i] * scale * weight[i]
// ---------------------------------------------------------------------------
static void cpu_rmsnorm_head(
        float       * x,       // in-place
        const float * weight,
        int           n,
        float         eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float sc = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) x[i] = x[i] * sc * weight[i];
}

// ---------------------------------------------------------------------------
// CPU reference: RoPE (adjacent pairing) for one head
// Matches decode.hip rope section: adjacent pairs (i, i+1) for i in 0,2,4,...
// theta[i] = position * ROPE_THETA^(-i / ROPE_DIM)
// ---------------------------------------------------------------------------
static void cpu_rope_head(float * x, int position) {
    for (int i = 0; i < FA_ROPE_DIM; i += 2) {
        float theta = (float)position * powf(FA_ROPE_THETA, -(float)i / (float)FA_ROPE_DIM);
        float cv = cosf(theta), sv = sinf(theta);
        float x0 = x[i], x1 = x[i + 1];
        x[i]     = x0 * cv - x1 * sv;
        x[i + 1] = x0 * sv + x1 * cv;
    }
}

// ---------------------------------------------------------------------------
// Helper: load a kernel function from HIP module
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

    if (model->layers.empty()) {
        fprintf(stderr, "FAIL: model has no layers\n");
        llama_model_free(model);
        return 1;
    }

    // ---- Get Q norm weight (GPU pointer) ----
    const ggml_tensor * q_norm_t = model->layers[0].attn_q_norm;
    if (!q_norm_t || !q_norm_t->data) {
        fprintf(stderr, "FAIL: layers[0].attn_q_norm is null or has no data\n");
        llama_model_free(model);
        return 1;
    }
    if (q_norm_t->type != GGML_TYPE_F32) {
        fprintf(stderr, "FAIL: attn_q_norm type is %s, expected F32\n",
                ggml_type_name(q_norm_t->type));
        llama_model_free(model);
        return 1;
    }
    if ((int)q_norm_t->ne[0] != FA_HEAD_DIM) {
        fprintf(stderr, "FAIL: attn_q_norm ne[0]=%lld, expected %d\n",
                (long long)q_norm_t->ne[0], FA_HEAD_DIM);
        llama_model_free(model);
        return 1;
    }
    const float * q_norm_gpu = (const float *)q_norm_t->data;
    fprintf(stderr, "attn_q_norm: ne[0]=%lld, type=F32, data=%p (GPU)\n",
            (long long)q_norm_t->ne[0], (const void *)q_norm_gpu);

    // ---- Get K norm weight (GPU pointer) ----
    const ggml_tensor * k_norm_t = model->layers[0].attn_k_norm;
    if (!k_norm_t || !k_norm_t->data) {
        fprintf(stderr, "FAIL: layers[0].attn_k_norm is null or has no data\n");
        llama_model_free(model);
        return 1;
    }
    if (k_norm_t->type != GGML_TYPE_F32) {
        fprintf(stderr, "FAIL: attn_k_norm type is %s, expected F32\n",
                ggml_type_name(k_norm_t->type));
        llama_model_free(model);
        return 1;
    }
    if ((int)k_norm_t->ne[0] != FA_HEAD_DIM) {
        fprintf(stderr, "FAIL: attn_k_norm ne[0]=%lld, expected %d\n",
                (long long)k_norm_t->ne[0], FA_HEAD_DIM);
        llama_model_free(model);
        return 1;
    }
    const float * k_norm_gpu = (const float *)k_norm_t->data;
    fprintf(stderr, "attn_k_norm: ne[0]=%lld, type=F32, data=%p (GPU)\n",
            (long long)k_norm_t->ne[0], (const void *)k_norm_gpu);

    // ---- Build synthetic host inputs ----
    // Q proj: FA_QPROJ_SIZE = 2048 floats, sinf pattern
    // KV proj: FA_KV_SIZE*2 = 1024 floats, cosf pattern
    std::vector<float> host_q(FA_QPROJ_SIZE);
    std::vector<float> host_kv(FA_KV_SIZE * 2);

    for (int i = 0; i < FA_QPROJ_SIZE; i++) {
        host_q[i] = sinf((float)i * 0.1f);
    }
    for (int i = 0; i < FA_KV_SIZE * 2; i++) {
        host_kv[i] = cosf((float)i * 0.1f);
    }

    // ---- Fetch norm weights from GPU for CPU reference ----
    std::vector<float> host_q_norm(FA_HEAD_DIM);
    std::vector<float> host_k_norm(FA_HEAD_DIM);
    HIP_CHECK(hipMemcpy(host_q_norm.data(), q_norm_gpu,
                        (size_t)FA_HEAD_DIM * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_k_norm.data(), k_norm_gpu,
                        (size_t)FA_HEAD_DIM * sizeof(float), hipMemcpyDeviceToHost));

    fprintf(stderr, "q_norm_w[0..3]: %f %f %f %f\n",
            host_q_norm[0], host_q_norm[1], host_q_norm[2], host_q_norm[3]);
    fprintf(stderr, "k_norm_w[0..3]: %f %f %f %f\n",
            host_k_norm[0], host_k_norm[1], host_k_norm[2], host_k_norm[3]);

    // ---- Allocate GPU buffers ----
    const int position    = 5;
    const int max_seq_len = 2048;

    float  * gpu_q  = nullptr;
    float  * gpu_kv = nullptr;
    __half * gpu_k_cache = nullptr;
    __half * gpu_v_cache = nullptr;

    // KV cache: [n_kv_heads * max_seq_len * head_dim] half floats each
    const size_t kv_cache_elems = (size_t)FA_N_KV_HEADS * max_seq_len * FA_HEAD_DIM;
    const size_t kv_cache_bytes = kv_cache_elems * sizeof(__half);

    HIP_CHECK(hipMalloc(&gpu_q,       (size_t)FA_QPROJ_SIZE * sizeof(float)));
    HIP_CHECK(hipMalloc(&gpu_kv,      (size_t)FA_KV_SIZE * 2 * sizeof(float)));
    HIP_CHECK(hipMalloc(&gpu_k_cache, kv_cache_bytes));
    HIP_CHECK(hipMalloc(&gpu_v_cache, kv_cache_bytes));

    // Copy synthetic inputs to GPU
    HIP_CHECK(hipMemcpy(gpu_q,  host_q.data(),
                        (size_t)FA_QPROJ_SIZE * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(gpu_kv, host_kv.data(),
                        (size_t)FA_KV_SIZE * 2 * sizeof(float), hipMemcpyHostToDevice));

    // Zero the KV cache so we can verify position 5 is written
    HIP_CHECK(hipMemset(gpu_k_cache, 0, kv_cache_bytes));
    HIP_CHECK(hipMemset(gpu_v_cache, 0, kv_cache_bytes));

    // ---- Load HIP module and kernel ----
    hipModule_t   hip_mod  = nullptr;
    hipFunction_t kernel_fn = nullptr;

    {
        hipError_t e = hipModuleLoad(&hip_mod, hsaco_path.c_str());
        if (e != hipSuccess) {
            fprintf(stderr, "FAIL: hipModuleLoad(%s) — %s\n",
                    hsaco_path.c_str(), hipGetErrorString(e));
            hipFree(gpu_q);
            hipFree(gpu_kv);
            hipFree(gpu_k_cache);
            hipFree(gpu_v_cache);
            llama_model_free(model);
            return 1;
        }
    }

    kernel_fn = load_kernel(hip_mod, "eval_qk_norm_rope_kv_write");
    if (!kernel_fn) {
        hipModuleUnload(hip_mod);
        hipFree(gpu_q);
        hipFree(gpu_kv);
        hipFree(gpu_k_cache);
        hipFree(gpu_v_cache);
        llama_model_free(model);
        return 1;
    }

    // ---- Launch kernel ----
    // Kernel signature:
    //   eval_qk_norm_rope_kv_write(q_proj, kv_proj, q_norm_w, k_norm_w,
    //                               k_cache, v_cache, position, max_seq_len)
    // Block: 512 threads (= 16 warps, one warp per head slot)
    // Grid: (total_heads + 15) / 16 where total_heads = FA_N_Q_HEADS + FA_N_KV_HEADS = 40
    //   → ceil(40/16) = 3 blocks
    {
        const int total_heads = FA_N_Q_HEADS + FA_N_KV_HEADS;  // 40
        const unsigned grid_x = ((unsigned)total_heads + 15u) / 16u;  // 3

        const float * freq_factors_null = nullptr;
        float rope_theta_ovr = 0.0f;
        const int * d_params_nullptr = nullptr;
        const void * args[] = {
            &gpu_q,             // float * q_proj
            &gpu_kv,            // float * kv_proj
            &q_norm_gpu,        // const float * q_norm_w (GPU pointer)
            &k_norm_gpu,        // const float * k_norm_w (GPU pointer)
            &gpu_k_cache,       // __half * k_cache
            &gpu_v_cache,       // __half * v_cache
            &freq_factors_null, // const float * freq_factors (NULL)
            &position,          // const int position
            &max_seq_len,       // const int max_seq_len
            &rope_theta_ovr,    // const float rope_theta_override
            &d_params_nullptr   // const int * d_decode_params (NULL)
        };

        fprintf(stderr, "Launching eval_qk_norm_rope_kv_write: grid=(%u,1,1), block=(512,1,1)\n",
                grid_x);

        hipError_t e = hipModuleLaunchKernel(
            kernel_fn,
            grid_x, 1, 1,   // grid
            512, 1, 1,       // block: 512 threads = 16 warps
            0, nullptr,
            (void **)args, nullptr);
        if (e != hipSuccess) {
            fprintf(stderr, "FAIL: hipModuleLaunchKernel — %s\n", hipGetErrorString(e));
            hipModuleUnload(hip_mod);
            hipFree(gpu_q);
            hipFree(gpu_kv);
            hipFree(gpu_k_cache);
            hipFree(gpu_v_cache);
            llama_model_free(model);
            return 1;
        }
        HIP_CHECK(hipDeviceSynchronize());
        fprintf(stderr, "Kernel launched and synchronized\n");
    }

    // ---- Copy results back to host ----
    std::vector<float> result_q(FA_QPROJ_SIZE);
    HIP_CHECK(hipMemcpy(result_q.data(), gpu_q,
                        (size_t)FA_QPROJ_SIZE * sizeof(float), hipMemcpyDeviceToHost));

    // KV cache: only need position 5 slot for each head
    // k_cache layout: [head, seq_pos, dim] = head * max_seq_len * head_dim + pos * head_dim
    std::vector<uint16_t> result_k_cache(kv_cache_elems);
    std::vector<uint16_t> result_v_cache(kv_cache_elems);
    HIP_CHECK(hipMemcpy(result_k_cache.data(), gpu_k_cache, kv_cache_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(result_v_cache.data(), gpu_v_cache, kv_cache_bytes, hipMemcpyDeviceToHost));

    // ---- Print first 8 Q values ----
    fprintf(stderr, "\n[GPU Q proj — first 8 values after norm+rope]\n");
    for (int i = 0; i < 8 && i < FA_QPROJ_SIZE; i++) {
        fprintf(stderr, "  q[%d] = %f\n", i, result_q[i]);
    }

    // ---- Check Q for NaN and non-zero ----
    int q_nans = 0, q_nonzero = 0;
    for (int i = 0; i < FA_QPROJ_SIZE; i++) {
        float v = result_q[i];
        if (v != v) q_nans++;
        else if (v != 0.0f) q_nonzero++;
    }
    fprintf(stderr, "Q proj: non-zero=%d/%d, NaNs=%d\n", q_nonzero, FA_QPROJ_SIZE, q_nans);

    // ---- Check KV cache at position 5 ----
    // For each KV head, position 5 occupies: head * max_seq_len * FA_HEAD_DIM + 5 * FA_HEAD_DIM
    int k_nonzero_at_pos5 = 0;
    int v_nonzero_at_pos5 = 0;
    for (int head = 0; head < FA_N_KV_HEADS; head++) {
        int slot = head * max_seq_len * FA_HEAD_DIM + position * FA_HEAD_DIM;
        for (int d = 0; d < FA_HEAD_DIM; d++) {
            if (result_k_cache[slot + d] != 0) k_nonzero_at_pos5++;
            if (result_v_cache[slot + d] != 0) v_nonzero_at_pos5++;
        }
    }
    fprintf(stderr, "\n[KV cache at position %d]\n", position);
    fprintf(stderr, "  K cache non-zero elements: %d / %d\n",
            k_nonzero_at_pos5, FA_N_KV_HEADS * FA_HEAD_DIM);
    fprintf(stderr, "  V cache non-zero elements: %d / %d\n",
            v_nonzero_at_pos5, FA_N_KV_HEADS * FA_HEAD_DIM);

    // Print first KV head, position 5 values (as float)
    {
        int slot = 0 * max_seq_len * FA_HEAD_DIM + position * FA_HEAD_DIM;
        fprintf(stderr, "  K cache head=0, pos=%d, first 4 values: ", position);
        for (int d = 0; d < 4; d++) {
            uint16_t h = result_k_cache[slot + d];
            float f;
            memcpy(&f, &h, 2);
            // __half to float manually via bit pattern
            // Use a simple union conversion that works without __half on host
            // Actually: use ggml_fp16_to_fp32 for correctness
            f = ggml_fp16_to_fp32((ggml_fp16_t)result_k_cache[slot + d]);
            fprintf(stderr, "%f ", f);
        }
        fprintf(stderr, "\n");

        fprintf(stderr, "  V cache head=0, pos=%d, first 4 values: ", position);
        for (int d = 0; d < 4; d++) {
            float f = ggml_fp16_to_fp32((ggml_fp16_t)result_v_cache[slot + d]);
            fprintf(stderr, "%f ", f);
        }
        fprintf(stderr, "\n");
    }

    // Verify position 0 is untouched (all zeros) — kernel only writes position 5
    int k_nonzero_at_pos0 = 0, v_nonzero_at_pos0 = 0;
    for (int head = 0; head < FA_N_KV_HEADS; head++) {
        int slot = head * max_seq_len * FA_HEAD_DIM + 0 * FA_HEAD_DIM;
        for (int d = 0; d < FA_HEAD_DIM; d++) {
            if (result_k_cache[slot + d] != 0) k_nonzero_at_pos0++;
            if (result_v_cache[slot + d] != 0) v_nonzero_at_pos0++;
        }
    }
    fprintf(stderr, "  K cache non-zero at position 0 (should be 0): %d\n", k_nonzero_at_pos0);
    fprintf(stderr, "  V cache non-zero at position 0 (should be 0): %d\n", v_nonzero_at_pos0);

    // ---- CPU reference ----
    // Apply RMSNorm then RoPE to each Q head
    std::vector<float> cpu_q(host_q);  // start from same synthetic input

    fprintf(stderr, "\n[Computing CPU reference]\n");
    for (int head = 0; head < FA_N_Q_HEADS; head++) {
        float * qh = cpu_q.data() + head * FA_HEAD_DIM;
        cpu_rmsnorm_head(qh, host_q_norm.data(), FA_HEAD_DIM, NORM_EPS);
        cpu_rope_head(qh, position);
    }

    fprintf(stderr, "\n[CPU Q proj — first 8 values after norm+rope]\n");
    for (int i = 0; i < 8 && i < FA_QPROJ_SIZE; i++) {
        fprintf(stderr, "  cpu_q[%d] = %f\n", i, cpu_q[i]);
    }

    // ---- Compare GPU Q vs CPU Q ----
    float max_diff_q = 0.0f;
    int   mismatch_q = 0;
    for (int i = 0; i < FA_QPROJ_SIZE; i++) {
        float diff = fabsf(result_q[i] - cpu_q[i]);
        if (diff > max_diff_q) max_diff_q = diff;
        if (diff > 1e-3f) mismatch_q++;
    }
    fprintf(stderr, "\n[Comparison GPU Q vs CPU Q]\n");
    fprintf(stderr, "  max_abs_diff = %e\n", (double)max_diff_q);
    fprintf(stderr, "  values with diff > 1e-3: %d / %d\n", mismatch_q, FA_QPROJ_SIZE);

    // Print first 8 diffs
    fprintf(stderr, "  First 8 element diffs:\n");
    for (int i = 0; i < 8 && i < FA_QPROJ_SIZE; i++) {
        fprintf(stderr, "    [%d] gpu=%f  cpu=%f  diff=%e\n",
                i, result_q[i], cpu_q[i], (double)fabsf(result_q[i] - cpu_q[i]));
    }

    // ---- Cleanup ----
    hipFree(gpu_q);
    hipFree(gpu_kv);
    hipFree(gpu_k_cache);
    hipFree(gpu_v_cache);
    hipModuleUnload(hip_mod);
    llama_model_free(model);
    llama_backend_free();

    // ---- PASS / FAIL ----
    bool pass = true;

    if (q_nans > 0) {
        fprintf(stderr, "\nFAIL: Q proj contains %d NaN values — likely NaN source!\n", q_nans);
        pass = false;
    }
    if (q_nonzero == 0) {
        fprintf(stderr, "FAIL: Q proj is all zeros (kernel did not run?)\n");
        pass = false;
    }
    if (max_diff_q > 1e-3f) {
        fprintf(stderr, "FAIL: Q proj max_abs_diff %.6e exceeds threshold 1e-3\n",
                (double)max_diff_q);
        pass = false;
    }
    if (mismatch_q > 0) {
        fprintf(stderr, "FAIL: %d / %d Q elements differ by more than 1e-3\n",
                mismatch_q, FA_QPROJ_SIZE);
        pass = false;
    }
    if (k_nonzero_at_pos5 == 0) {
        fprintf(stderr, "FAIL: K cache at position %d is all zeros\n", position);
        pass = false;
    }
    if (v_nonzero_at_pos5 == 0) {
        fprintf(stderr, "FAIL: V cache at position %d is all zeros\n", position);
        pass = false;
    }
    if (k_nonzero_at_pos0 > 0) {
        fprintf(stderr, "FAIL: K cache at position 0 was written (should be untouched)\n");
        pass = false;
    }
    if (v_nonzero_at_pos0 > 0) {
        fprintf(stderr, "FAIL: V cache at position 0 was written (should be untouched)\n");
        pass = false;
    }

    if (pass) {
        printf("PASS: eval_qk_norm_rope_kv_write GPU output matches CPU reference "
               "(Q max_abs_diff=%.2e, K/V cache written at pos %d)\n",
               (double)max_diff_q, position);
        return 0;
    }
    return 1;
}
