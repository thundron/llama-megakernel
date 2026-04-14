// test-kernel-attention-d64.cpp — Unit test for eval_attention_decode
//
// Synthetic flash attention test with FA_HEAD_DIM=64 (matching Llama 1B .hsaco).
// Creates small Q, K cache, V cache; launches attention kernel; compares against
// CPU reference (standard scaled dot-product attention with softmax).
//
// Setup: 2 Q heads (both mapping to KV head 0 via GQA), kv_len=8, max_seq=16, D=64.
// This is deliberately small to be safe even if something is wrong.
//
// Usage: test-kernel-attention-d64
// Requires: decode.hip_*.hsaco compiled for FA_HEAD_DIM=64 (Llama 1B defaults)

#include "test-harness.h"

// ---------------------------------------------------------------------------
// Constants matching Llama 1B .hsaco defaults
// ---------------------------------------------------------------------------
static constexpr int D              = 64;   // FA_HEAD_DIM
static constexpr int N_Q_HEADS      = 32;   // FA_N_Q_HEADS (baked, but we only test 2)
static constexpr int N_KV_HEADS     = 8;    // FA_N_KV_HEADS
static constexpr int GQA_RATIO      = N_Q_HEADS / N_KV_HEADS;  // 4
static constexpr int TEST_Q_HEADS   = 2;    // only launch 2 blocks
static constexpr int KV_LEN         = 128;  // must be >= nthreads=128 to avoid dilution from zero-V positions
static constexpr int MAX_SEQ        = 128;  // max_seq_len buffer size

// ---------------------------------------------------------------------------
// CPU reference: standard scaled dot-product attention
// ---------------------------------------------------------------------------
static void cpu_attention_decode(
        const float * q_proj,     // [N_Q_HEADS * D] (we only use TEST_Q_HEADS)
        const uint16_t * k_cache, // [N_KV_HEADS, MAX_SEQ, D] f16
        const uint16_t * v_cache, // [N_KV_HEADS, MAX_SEQ, D] f16
        float * output,           // [N_Q_HEADS * D]
        int kv_len) {

    const float scale = 1.0f / sqrtf((float)D);

    for (int qh = 0; qh < TEST_Q_HEADS; qh++) {
        int kvh = qh / GQA_RATIO;  // both heads 0,1 → kvh=0

        const float * q = q_proj + qh * D;
        const uint16_t * kc = k_cache + kvh * MAX_SEQ * D;
        const uint16_t * vc = v_cache + kvh * MAX_SEQ * D;
        float * out = output + qh * D;

        // Compute QK scores
        std::vector<float> qk(kv_len);
        float qk_max = -1e30f;
        for (int p = 0; p < kv_len; p++) {
            double dot = 0.0;
            for (int d = 0; d < D; d++) {
                dot += (double)q[d] * (double)f16_to_f32(kc[p * D + d]);
            }
            qk[p] = (float)dot * scale;
            if (qk[p] > qk_max) qk_max = qk[p];
        }

        // Softmax
        std::vector<float> attn(kv_len);
        float sum = 0.0f;
        for (int p = 0; p < kv_len; p++) {
            attn[p] = expf(qk[p] - qk_max);
            sum += attn[p];
        }
        for (int p = 0; p < kv_len; p++) attn[p] /= sum;

        // Weighted sum of V
        for (int d = 0; d < D; d++) {
            double acc = 0.0;
            for (int p = 0; p < kv_len; p++) {
                acc += (double)attn[p] * (double)f16_to_f32(vc[p * D + d]);
            }
            out[d] = (float)acc;
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    // Must use .hsaco compiled for FA_HEAD_DIM=64 (Llama 1B)
    std::string hsaco_path = find_hsaco("decode.hip_", 64);
    if (hsaco_path.empty()) {
        fprintf(stderr, "FAIL: decode.hip_*.hsaco with FA_HEAD_DIM=64 not found\n");
        return 1;
    }
    fprintf(stderr, "Using hsaco: %s\n\n", hsaco_path.c_str());

    hipModule_t mod = nullptr;
    hipError_t e = hipModuleLoad(&mod, hsaco_path.c_str());
    if (e != hipSuccess) {
        fprintf(stderr, "FAIL: hipModuleLoad — %s\n", hipGetErrorString(e));
        return 1;
    }

    hipFunction_t fn = load_kernel(mod, "eval_attention_decode");
    if (!fn) {
        test_fail("eval_attention_decode", "kernel not found");
        hipModuleUnload(mod);
        return test_summary();
    }

    // --- Build synthetic data ---
    // Q projection: [N_Q_HEADS * D] f32 (only fill first TEST_Q_HEADS heads)
    const int q_elems = N_Q_HEADS * D;
    std::vector<float> h_q(q_elems, 0.0f);
    for (int qh = 0; qh < TEST_Q_HEADS; qh++) {
        for (int d = 0; d < D; d++) {
            // Strong Q signal — unit-magnitude, distinct per head
            h_q[qh * D + d] = (d < D/2) ? 1.0f : -1.0f;
            h_q[qh * D + d] *= (1.0f + 0.1f * sinf((float)(qh * D + d)));
        }
    }

    // K cache: [N_KV_HEADS, MAX_SEQ, D] f16 (only fill kvh=0, positions 0..KV_LEN-1)
    const int kv_elems = N_KV_HEADS * MAX_SEQ * D;
    std::vector<uint16_t> h_k(kv_elems, 0);
    std::vector<uint16_t> h_v(kv_elems, 0);
    for (int p = 0; p < KV_LEN; p++) {
        for (int d = 0; d < D; d++) {
            // Varied K: some positions align with Q, creating peaked softmax
            float k_val = cosf((float)(p * 17 + d) * 0.2f) * 0.5f;
            h_k[p * D + d] = f32_to_f16(k_val);
            // V with unit magnitude to produce non-tiny outputs
            float v_val = sinf((float)(p * D + d) * 0.1f) * 2.0f;
            h_v[p * D + d] = f32_to_f16(v_val);
        }
    }

    // CPU reference
    const int out_elems = N_Q_HEADS * D;
    std::vector<float> h_cpu_out(out_elems, 0.0f);
    cpu_attention_decode(h_q.data(), h_k.data(), h_v.data(), h_cpu_out.data(), KV_LEN);

    // --- GPU buffers ---
    GpuBuf g_q(q_elems * sizeof(float));
    GpuBuf g_k(kv_elems * sizeof(uint16_t));
    GpuBuf g_v(kv_elems * sizeof(uint16_t));
    GpuBuf g_out(out_elems * sizeof(float));

    g_q.upload(h_q.data(), q_elems * sizeof(float));
    g_k.upload(h_k.data(), kv_elems * sizeof(uint16_t));
    g_v.upload(h_v.data(), kv_elems * sizeof(uint16_t));

    // --- Launch kernel ---
    // Grid: (TEST_Q_HEADS, 1, 1) — one block per Q head
    // Block: dim3(32, 4, 1) = 128 threads
    void * q_ptr = g_q.ptr;
    void * k_ptr = g_k.ptr;
    void * v_ptr = g_v.ptr;
    void * o_ptr = g_out.ptr;
    int kv_len_val = KV_LEN;
    int max_seq_val = MAX_SEQ;

    float alibi_mb = 0, alibi_m0 = 0, alibi_m1 = 0;
    int alibi_nhl = 0, cur_pos = KV_LEN - 1;
    const float * rel_bias = nullptr;
    float softcap = 0.0f;
    const int * d_params_nullptr = nullptr;
    void * args[] = { &q_ptr, &k_ptr, &v_ptr, &o_ptr, &kv_len_val, &max_seq_val,
                      &alibi_mb, &alibi_m0, &alibi_m1, &alibi_nhl, &cur_pos, &rel_bias,
                      &softcap, &d_params_nullptr };

    if (!launch_kernel(fn, (unsigned)TEST_Q_HEADS, 1, 1, 32, 4, 1, args)) {
        test_fail("eval_attention_decode", "launch failed");
        hipModuleUnload(mod);
        return test_summary();
    }

    // --- Read back and compare ---
    std::vector<float> h_gpu_out(out_elems, 0.0f);
    g_out.download(h_gpu_out.data(), out_elems * sizeof(float));

    // Print first head output
    fprintf(stderr, "[Head 0 output — first 8 dims]\n");
    fprintf(stderr, "  GPU:  ");
    for (int d = 0; d < 8; d++) fprintf(stderr, "%8.5f ", h_gpu_out[d]);
    fprintf(stderr, "\n  CPU:  ");
    for (int d = 0; d < 8; d++) fprintf(stderr, "%8.5f ", h_cpu_out[d]);
    fprintf(stderr, "\n\n");

    // Compare both heads
    for (int qh = 0; qh < TEST_Q_HEADS; qh++) {
        char name[64];
        snprintf(name, sizeof(name), "attn_decode_head_%d", qh);

        CompareResult r = compare_float(
            h_gpu_out.data() + qh * D,
            h_cpu_out.data() + qh * D, D);

        if (r.nan_count > 0) {
            test_fail(name, "%d NaN values", r.nan_count);
            continue;
        }
        // VKQ accumulation in half2 over 128 positions: significant precision loss
        // compared to CPU double-precision reference. Allow 10% relative OR 0.01 absolute.
        if (r.max_rel > 0.10f && r.max_abs > 0.01f) {
            test_fail(name, "max_rel=%.4f max_abs=%.4f", (double)r.max_rel, (double)r.max_abs);
            continue;
        }
        test_pass(name, "max_abs=%.2e, max_rel=%.4f", (double)r.max_abs, (double)r.max_rel);
    }

    // --- Test 2: kv_len=1 (the edge case that caused the 10-unit logit shift bug) ---
    // With bounds check, position 0 should get 100% attention weight.
    // Without bounds check, 127 zero-V positions dilute the output.
    {
        const int KV1 = 1, MAX1 = 128;
        const int kv1_elems = N_KV_HEADS * MAX1 * D;
        std::vector<float> h_q1(q_elems, 0.0f);
        std::vector<uint16_t> h_k1(kv1_elems, 0);
        std::vector<uint16_t> h_v1(kv1_elems, 0);

        // Q = all 1.0, K[0] = all 0.5, V[0] = all 0.3
        for (int d = 0; d < D; d++) h_q1[d] = 1.0f;
        for (int d = 0; d < D; d++) h_k1[d] = f32_to_f16(0.5f);
        for (int d = 0; d < D; d++) h_v1[d] = f32_to_f16(0.3f);

        // CPU reference: with 1 position, output = V[0] exactly
        std::vector<float> h_cpu1(out_elems, 0.0f);
        for (int d = 0; d < D; d++) h_cpu1[d] = 0.3f; // V[0] value

        GpuBuf g_q1(q_elems * sizeof(float));
        GpuBuf g_k1(kv1_elems * sizeof(uint16_t));
        GpuBuf g_v1(kv1_elems * sizeof(uint16_t));
        GpuBuf g_out1(out_elems * sizeof(float));
        g_q1.upload(h_q1.data(), q_elems * sizeof(float));
        g_k1.upload(h_k1.data(), kv1_elems * sizeof(uint16_t));
        g_v1.upload(h_v1.data(), kv1_elems * sizeof(uint16_t));

        void * q1p = g_q1.ptr, * k1p = g_k1.ptr, * v1p = g_v1.ptr, * o1p = g_out1.ptr;
        int kv1_val = KV1, max1_val = MAX1;
        void * args1[] = { &q1p, &k1p, &v1p, &o1p, &kv1_val, &max1_val };
        if (launch_kernel(fn, 1, 1, 1, 32, 4, 1, args1)) {
            std::vector<float> h_gpu1(out_elems, 0.0f);
            g_out1.download(h_gpu1.data(), out_elems * sizeof(float));

            CompareResult r = compare_float(h_gpu1.data(), h_cpu1.data(), D);
            if (r.nan_count > 0) {
                test_fail("attn_kv1_bounds", "%d NaN", r.nan_count);
            } else if (r.max_abs > 0.01f) {
                // >0.01 means dilution — the bounds check is broken
                test_fail("attn_kv1_bounds", "max_abs=%.4f (dilution detected)", (double)r.max_abs);
            } else {
                test_pass("attn_kv1_bounds", "max_abs=%.4f (no dilution)", (double)r.max_abs);
            }
        }
    }

    hipModuleUnload(mod);
    return test_summary();
}
