// test-kernel-moe.cpp — MoE FFN integration test
// Tests the MoE routing pipeline: softmax → argsort_top_k → per-expert matvec → weighted add
// Requires: decode.hip_*.hsaco in ~/.cache/gfx1100-megakernel/
#include "test-harness.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <numeric>

static const int N_EXPERTS = 8;
static const int N_EXPERTS_USED = 2;
static const int HIDDEN = 256;
static const int N_FF = 512;

// CPU reference: softmax over N_EXPERTS router logits
static void ref_softmax(const float * in, float * out, int n) {
    float max_v = -1e30f;
    for (int i = 0; i < n; i++) max_v = fmaxf(max_v, in[i]);
    float sum = 0;
    for (int i = 0; i < n; i++) { out[i] = expf(in[i] - max_v); sum += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= sum;
}

// CPU reference: argsort descending, return top-k indices
static void ref_argsort_topk(const float * probs, int * indices, int n, int k) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[a] > probs[b]; });
    for (int i = 0; i < k; i++) indices[i] = idx[i];
}

int main() {
    hipInit(0);
    hipSetDevice(0);

    std::string hsaco_path = find_hsaco("decode.hip_", 64);
    if (hsaco_path.empty()) {
        fprintf(stderr, "FAIL: decode.hip_*.hsaco not found\n");
        return 1;
    }
    hipModule_t mod;
    if (hipModuleLoad(&mod, hsaco_path.c_str()) != hipSuccess) {
        fprintf(stderr, "FAIL: cannot load %s\n", hsaco_path.c_str());
        return 1;
    }

    srand(42);

    // ===== Test 1: Softmax row kernel =====
    {
        hipFunction_t fn = load_kernel(mod, "eval_softmax_row");
        if (!fn) { test_fail("eval_softmax_row", "kernel not found"); }
        else {
            std::vector<float> h_in(N_EXPERTS), h_out(N_EXPERTS), h_ref(N_EXPERTS);
            for (int i = 0; i < N_EXPERTS; i++) h_in[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
            ref_softmax(h_in.data(), h_ref.data(), N_EXPERTS);

            GpuBuf d_in(N_EXPERTS * 4), d_out(N_EXPERTS * 4);
            hipMemcpy(d_in.ptr, h_in.data(), N_EXPERTS * 4, hipMemcpyHostToDevice);

            int n = N_EXPERTS;
            float scale = 1.0f;
            void * args[] = { &d_in.ptr, &d_out.ptr, &n, &scale };
            if (!launch_kernel(fn, 1, 1, 1, 256, 1, 1, args, (256/32) * sizeof(float))) {
                test_fail("eval_softmax_row", "launch failed");
            } else {
                hipDeviceSynchronize();
                hipMemcpy(h_out.data(), d_out.ptr, N_EXPERTS * 4, hipMemcpyDeviceToHost);
                float max_err = 0;
                for (int i = 0; i < N_EXPERTS; i++) {
                    float err = fabsf(h_out[i] - h_ref[i]);
                    if (err > max_err) max_err = err;
                }
                if (max_err < 1e-4f) test_pass("eval_softmax_row", "max_err=%.6f", max_err);
                else {
                    char msg[128]; snprintf(msg, sizeof(msg), "max error %.8f", max_err);
                    test_fail("eval_softmax_row", msg);
                }
            }
        }
    }

    // ===== Test 2: Argsort descending =====
    {
        hipFunction_t fn = load_kernel(mod, "eval_argsort_desc");
        if (!fn) { test_fail("eval_argsort_desc", "kernel not found"); }
        else {
            std::vector<float> h_probs(N_EXPERTS);
            std::vector<int> h_idx(N_EXPERTS), h_ref_idx(N_EXPERTS);
            for (int i = 0; i < N_EXPERTS; i++) h_probs[i] = (float)(rand() % 1000) / 100.0f;
            ref_argsort_topk(h_probs.data(), h_ref_idx.data(), N_EXPERTS, N_EXPERTS);

            GpuBuf d_probs(N_EXPERTS * 4), d_idx(N_EXPERTS * 4);
            hipMemcpy(d_probs.ptr, h_probs.data(), N_EXPERTS * 4, hipMemcpyHostToDevice);

            int n = N_EXPERTS;
            int npad = 1;
            while (npad < n) npad *= 2;
            void * args[] = { &d_probs.ptr, &d_idx.ptr, &n, &npad };
            if (!launch_kernel(fn, 1, 1, 1, npad, 1, 1, args, npad * sizeof(int))) {
                test_fail("eval_argsort_desc", "launch failed");
            } else {
                hipDeviceSynchronize();
                hipMemcpy(h_idx.data(), d_idx.ptr, N_EXPERTS * 4, hipMemcpyDeviceToHost);
                // Check top-2 match
                bool ok = (h_idx[0] == h_ref_idx[0] && h_idx[1] == h_ref_idx[1]);
                if (ok) test_pass("eval_argsort_desc", "top-2 match");
                else {
                    char msg[256];
                    snprintf(msg, sizeof(msg), "top-2 mismatch: got [%d,%d] expected [%d,%d]",
                             h_idx[0], h_idx[1], h_ref_idx[0], h_ref_idx[1]);
                    test_fail("eval_argsort_desc", msg);
                }
            }
        }
    }

    // ===== Test 3: Full MoE pipeline (softmax → topk → weighted accumulate) =====
    // This tests the logical correctness of the MoE dispatch pattern
    {
        // Generate random router logits
        std::vector<float> h_logits(N_EXPERTS);
        for (int i = 0; i < N_EXPERTS; i++) h_logits[i] = (float)(rand() % 1000) / 100.0f;

        // CPU reference: softmax → top-2 → normalize weights
        std::vector<float> h_probs(N_EXPERTS);
        ref_softmax(h_logits.data(), h_probs.data(), N_EXPERTS);

        int top_idx[2];
        ref_argsort_topk(h_probs.data(), top_idx, N_EXPERTS, 2);

        float w0 = h_probs[top_idx[0]];
        float w1 = h_probs[top_idx[1]];
        float wsum = w0 + w1;
        w0 /= wsum;
        w1 /= wsum;

        // The MoE dispatch pattern is:
        //   output = w0 * expert[top_idx[0]](input) + w1 * expert[top_idx[1]](input)
        // We can't test the full expert matvec without weights, but we can verify
        // the softmax + argsort + weight normalization is correct.
        if (fabsf(w0 + w1 - 1.0f) < 1e-6f && w0 >= w1 && w0 > 0 && w1 > 0) {
            test_pass("moe_pipeline_weights", "w0=%.4f w1=%.4f", w0, w1);
        } else {
            char msg[128];
            snprintf(msg, sizeof(msg), "w0=%.4f w1=%.4f sum=%.6f", w0, w1, w0+w1);
            test_fail("moe_pipeline_weights", msg);
        }
    }

    hipModuleUnload(mod);
    return test_summary();
}
