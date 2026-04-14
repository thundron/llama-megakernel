// test-kernel-deltanet.cpp — Unit tests for DeltaNet eval kernels
//
// Tests: eval_dn_l2_norm, eval_dn_conv1d_silu, eval_dn_recurrence
// Purely synthetic with small dimensions. CPU reference follows exact GPU algorithm.
//
// Setup: n_heads=2, key_dim=32, value_dim=32, conv_kernel=4
//
// Usage: test-kernel-deltanet
// Requires: decode.hip_*.hsaco in ~/.cache/gfx1100-megakernel/

#include "test-harness.h"

// CPU activation functions matching GPU
static float cpu_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float cpu_silu(float x) { return x / (1.0f + expf(-x)); }
static float cpu_softplus(float x) { return (x > 20.0f) ? x : logf(1.0f + expf(x)); }

static constexpr int N_HEADS    = 2;
static constexpr int N_K_HEADS  = 2;
static constexpr int KEY_DIM    = 32;
static constexpr int VALUE_DIM  = 32;
static constexpr int CONV_K     = 4;
static constexpr int TOTAL_CH   = 2 * N_K_HEADS * KEY_DIM + N_HEADS * VALUE_DIM; // Q + K + V

// ---------------------------------------------------------------------------
// Test: eval_dn_l2_norm
// Grid: (n_heads, 1, 1), Block: (256, 1, 1)
// Signature: (float* data, int n_heads, int dim, float eps)
// In-place L2 normalization per head
// ---------------------------------------------------------------------------
static void test_l2_norm(hipModule_t mod) {
    const char * name = "eval_dn_l2_norm";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int dim = KEY_DIM;
    const int total = N_HEADS * dim;
    const float eps = 1e-5f;

    std::vector<float> h_data(total), h_cpu(total);
    for (int i = 0; i < total; i++) {
        h_data[i] = sinf((float)i * 0.3f) * 2.0f;
        h_cpu[i] = h_data[i];
    }

    // CPU reference
    for (int h = 0; h < N_HEADS; h++) {
        float * head = h_cpu.data() + h * dim;
        float sq = 0;
        for (int i = 0; i < dim; i++) sq += head[i] * head[i];
        float scale = 1.0f / sqrtf(fmaxf(sq, eps * eps));
        for (int i = 0; i < dim; i++) head[i] *= scale;
    }

    GpuBuf g_data(total * sizeof(float));
    g_data.upload(h_data.data(), total * sizeof(float));

    void * d_ptr = g_data.ptr;
    int nh = N_HEADS, dm = dim;
    float ep = eps;
    void * args[] = { &d_ptr, &nh, &dm, &ep };

    if (!launch_kernel(fn, (unsigned)N_HEADS, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }

    std::vector<float> h_out(total);
    g_data.download(h_out.data(), total * sizeof(float));
    CompareResult r = compare_float(h_out.data(), h_cpu.data(), total);

    if (r.nan_count > 0) { test_fail(name, "%d NaN", r.nan_count); return; }
    if (r.max_rel > 1e-3f) { test_fail(name, "max_rel=%.2e", (double)r.max_rel); return; }
    test_pass(name, "max_abs=%.2e, max_rel=%.2e", (double)r.max_abs, (double)r.max_rel);
}

// ---------------------------------------------------------------------------
// Test: eval_dn_conv1d_silu
// Grid: (num_v_heads, 1, 1), Block: (256, 1, 1)
// Signature: (qkv_proj, conv_weight, conv_buf, q_out, k_out, v_out,
//             num_v_heads, num_k_heads, key_dim, value_dim, conv_kernel, total_conv_ch)
// ---------------------------------------------------------------------------
static void test_conv1d_silu(hipModule_t mod) {
    const char * name = "eval_dn_conv1d_silu";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int total_ch = TOTAL_CH;

    // Inputs
    std::vector<float> h_proj(total_ch), h_conv_w(total_ch * CONV_K);
    std::vector<float> h_conv_buf(total_ch * CONV_K, 0.0f);

    for (int i = 0; i < total_ch; i++) h_proj[i] = sinf((float)i * 0.1f);
    for (int i = 0; i < total_ch * CONV_K; i++) h_conv_w[i] = cosf((float)i * 0.05f) * 0.1f;

    // CPU reference
    std::vector<float> cpu_buf(total_ch * CONV_K, 0.0f);
    std::vector<float> cpu_q(N_HEADS * KEY_DIM, 0.0f);
    std::vector<float> cpu_k(N_HEADS * KEY_DIM, 0.0f);
    std::vector<float> cpu_v(N_HEADS * VALUE_DIM, 0.0f);

    for (int h = 0; h < N_HEADS; h++) {
        int kh = h % N_K_HEADS;
        bool is_first = (h < N_K_HEADS);

        int q_off = kh * KEY_DIM;
        int k_off = N_K_HEADS * KEY_DIM + kh * KEY_DIM;
        int v_off = 2 * N_K_HEADS * KEY_DIM + h * VALUE_DIM;

        // V channels
        for (int c = 0; c < VALUE_DIM; c++) {
            int ch = v_off + c;
            int buf = ch * CONV_K;
            for (int t = 0; t < CONV_K - 1; t++) cpu_buf[buf + t] = cpu_buf[buf + t + 1];
            cpu_buf[buf + CONV_K - 1] = h_proj[ch];
            float co = 0;
            for (int t = 0; t < CONV_K; t++) co += cpu_buf[buf + t] * h_conv_w[ch * CONV_K + t];
            cpu_v[h * VALUE_DIM + c] = cpu_silu(co);
        }

        // Q channels
        for (int c = 0; c < KEY_DIM; c++) {
            int ch = q_off + c;
            int buf = ch * CONV_K;
            if (is_first) {
                for (int t = 0; t < CONV_K - 1; t++) cpu_buf[buf + t] = cpu_buf[buf + t + 1];
                cpu_buf[buf + CONV_K - 1] = h_proj[ch];
            }
            float co = 0;
            for (int t = 0; t < CONV_K; t++) co += cpu_buf[buf + t] * h_conv_w[ch * CONV_K + t];
            cpu_q[h * KEY_DIM + c] = cpu_silu(co);
        }

        // K channels
        for (int c = 0; c < KEY_DIM; c++) {
            int ch = k_off + c;
            int buf = ch * CONV_K;
            if (is_first) {
                for (int t = 0; t < CONV_K - 1; t++) cpu_buf[buf + t] = cpu_buf[buf + t + 1];
                cpu_buf[buf + CONV_K - 1] = h_proj[ch];
            }
            float co = 0;
            for (int t = 0; t < CONV_K; t++) co += cpu_buf[buf + t] * h_conv_w[ch * CONV_K + t];
            cpu_k[h * KEY_DIM + c] = cpu_silu(co);
        }
    }

    // GPU
    GpuBuf g_proj(total_ch * sizeof(float));
    GpuBuf g_cw(total_ch * CONV_K * sizeof(float));
    GpuBuf g_cb(total_ch * CONV_K * sizeof(float));
    GpuBuf g_q(N_HEADS * KEY_DIM * sizeof(float));
    GpuBuf g_k(N_HEADS * KEY_DIM * sizeof(float));
    GpuBuf g_v(N_HEADS * VALUE_DIM * sizeof(float));

    g_proj.upload(h_proj.data(), total_ch * sizeof(float));
    g_cw.upload(h_conv_w.data(), total_ch * CONV_K * sizeof(float));
    // conv_buf starts zeroed (GpuBuf constructor memsets)

    void * p_ptr = g_proj.ptr, * cw_ptr = g_cw.ptr, * cb_ptr = g_cb.ptr;
    void * q_ptr = g_q.ptr, * k_ptr = g_k.ptr, * v_ptr = g_v.ptr;
    int nvh = N_HEADS, nkh = N_K_HEADS, kd = KEY_DIM, vd = VALUE_DIM;
    int ck = CONV_K, tc = total_ch;
    void * args[] = { &p_ptr, &cw_ptr, &cb_ptr, &q_ptr, &k_ptr, &v_ptr,
                      &nvh, &nkh, &kd, &vd, &ck, &tc };

    if (!launch_kernel(fn, (unsigned)N_HEADS, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }

    // Compare Q, K, V separately
    std::vector<float> h_gq(N_HEADS * KEY_DIM), h_gk(N_HEADS * KEY_DIM), h_gv(N_HEADS * VALUE_DIM);
    g_q.download(h_gq.data(), N_HEADS * KEY_DIM * sizeof(float));
    g_k.download(h_gk.data(), N_HEADS * KEY_DIM * sizeof(float));
    g_v.download(h_gv.data(), N_HEADS * VALUE_DIM * sizeof(float));

    CompareResult rq = compare_float(h_gq.data(), cpu_q.data(), N_HEADS * KEY_DIM);
    CompareResult rk = compare_float(h_gk.data(), cpu_k.data(), N_HEADS * KEY_DIM);
    CompareResult rv = compare_float(h_gv.data(), cpu_v.data(), N_HEADS * VALUE_DIM);

    bool ok = true;
    if (rq.nan_count || rk.nan_count || rv.nan_count) {
        test_fail(name, "NaN: Q=%d K=%d V=%d", rq.nan_count, rk.nan_count, rv.nan_count);
        ok = false;
    }
    float max_rel = fmaxf(rq.max_rel, fmaxf(rk.max_rel, rv.max_rel));
    if (max_rel > 1e-4f) {
        test_fail(name, "max_rel: Q=%.2e K=%.2e V=%.2e", (double)rq.max_rel, (double)rk.max_rel, (double)rv.max_rel);
        ok = false;
    }
    if (ok) test_pass(name, "Q_rel=%.2e K_rel=%.2e V_rel=%.2e",
                      (double)rq.max_rel, (double)rk.max_rel, (double)rv.max_rel);
}

// ---------------------------------------------------------------------------
// Test: eval_dn_recurrence
// Grid: (n_heads, 1, 1), Block: (256, 1, 1)
// Signature: (q, k, v, beta_proj, alpha_proj, ssm_a, dt_bias, z_proj, norm_w,
//             state, output, n_heads, key_dim, value_dim, norm_eps)
// ---------------------------------------------------------------------------
static void test_recurrence(hipModule_t mod) {
    const char * name = "eval_dn_recurrence";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int kd = KEY_DIM, vd = VALUE_DIM;
    const float norm_eps = 1e-5f;

    // Inputs
    std::vector<float> h_q(N_HEADS * kd), h_k(N_HEADS * kd), h_v(N_HEADS * vd);
    std::vector<float> h_beta(N_HEADS), h_alpha(N_HEADS), h_ssm_a(N_HEADS), h_dt_bias(N_HEADS);
    std::vector<float> h_z(N_HEADS * vd), h_nw(vd);
    std::vector<float> h_state(N_HEADS * kd * vd, 0.0f);

    for (int i = 0; i < N_HEADS * kd; i++) { h_q[i] = sinf((float)i * 0.1f) * 0.3f; h_k[i] = cosf((float)i * 0.07f) * 0.2f; }
    for (int i = 0; i < N_HEADS * vd; i++) { h_v[i] = sinf((float)i * 0.05f) * 0.4f; h_z[i] = cosf((float)i * 0.03f); }
    for (int i = 0; i < vd; i++) h_nw[i] = 0.5f + 0.5f * sinf((float)i * 0.1f);
    for (int h = 0; h < N_HEADS; h++) {
        h_beta[h]  = sinf((float)h * 1.5f) * 0.5f;
        h_alpha[h] = cosf((float)h * 2.0f) * 0.3f;
        h_ssm_a[h] = -expf(sinf((float)h * 0.7f));
        h_dt_bias[h] = 0.1f * (float)h;
    }

    // CPU reference
    std::vector<float> cpu_state(h_state);
    std::vector<float> cpu_out(N_HEADS * vd, 0.0f);

    for (int h = 0; h < N_HEADS; h++) {
        float beta = cpu_sigmoid(h_beta[h]);
        float x = h_alpha[h] + h_dt_bias[h];
        float sp = cpu_softplus(x);
        float g_val = expf(sp * h_ssm_a[h]);
        float scale = 1.0f / sqrtf((float)kd);

        float * hs = cpu_state.data() + h * kd * vd;
        const float * qh = h_q.data() + h * kd;
        const float * kh = h_k.data() + h * kd;
        const float * vh = h_v.data() + h * vd;

        for (int j = 0; j < vd; j++) {
            // kv = sum_i(state[j,i] * k[i])
            float kv = 0;
            for (int i = 0; i < kd; i++) kv += hs[j * kd + i] * kh[i];
            float delta = (vh[j] - g_val * kv) * beta;
            // Update state and compute attention
            float attn = 0;
            for (int i = 0; i < kd; i++) {
                hs[j * kd + i] = g_val * hs[j * kd + i] + kh[i] * delta;
                attn += hs[j * kd + i] * qh[i];
            }
            cpu_out[h * vd + j] = attn * scale;
        }

        // Gated RMSNorm
        float sq = 0;
        for (int i = 0; i < vd; i++) sq += cpu_out[h * vd + i] * cpu_out[h * vd + i];
        float rstd = 1.0f / sqrtf(sq / (float)vd + norm_eps);
        for (int i = 0; i < vd; i++) {
            float normed = cpu_out[h * vd + i] * rstd * h_nw[i];
            float zv = h_z[h * vd + i];
            cpu_out[h * vd + i] = normed * cpu_silu(zv);
        }
    }

    // GPU
    GpuBuf g_q(N_HEADS * kd * sizeof(float));
    GpuBuf g_k(N_HEADS * kd * sizeof(float));
    GpuBuf g_v(N_HEADS * vd * sizeof(float));
    GpuBuf g_beta(N_HEADS * sizeof(float));
    GpuBuf g_alpha(N_HEADS * sizeof(float));
    GpuBuf g_ssm(N_HEADS * sizeof(float));
    GpuBuf g_dtb(N_HEADS * sizeof(float));
    GpuBuf g_z(N_HEADS * vd * sizeof(float));
    GpuBuf g_nw(vd * sizeof(float));
    GpuBuf g_state(N_HEADS * kd * vd * sizeof(float));
    GpuBuf g_out(N_HEADS * vd * sizeof(float));

    g_q.upload(h_q.data(), N_HEADS * kd * sizeof(float));
    g_k.upload(h_k.data(), N_HEADS * kd * sizeof(float));
    g_v.upload(h_v.data(), N_HEADS * vd * sizeof(float));
    g_beta.upload(h_beta.data(), N_HEADS * sizeof(float));
    g_alpha.upload(h_alpha.data(), N_HEADS * sizeof(float));
    g_ssm.upload(h_ssm_a.data(), N_HEADS * sizeof(float));
    g_dtb.upload(h_dt_bias.data(), N_HEADS * sizeof(float));
    g_z.upload(h_z.data(), N_HEADS * vd * sizeof(float));
    g_nw.upload(h_nw.data(), vd * sizeof(float));
    g_state.upload(h_state.data(), N_HEADS * kd * vd * sizeof(float));

    void * qp = g_q.ptr, * kp = g_k.ptr, * vp = g_v.ptr;
    void * bp = g_beta.ptr, * ap = g_alpha.ptr, * sp = g_ssm.ptr, * dp = g_dtb.ptr;
    void * zp = g_z.ptr, * np = g_nw.ptr, * stp = g_state.ptr, * op = g_out.ptr;
    int nh = N_HEADS, kdd = kd, vdd = vd;
    float ne = norm_eps;
    void * args[] = { &qp, &kp, &vp, &bp, &ap, &sp, &dp, &zp, &np, &stp, &op,
                      &nh, &kdd, &vdd, &ne };

    if (!launch_kernel(fn, (unsigned)N_HEADS, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }

    std::vector<float> h_gpu_out(N_HEADS * vd);
    g_out.download(h_gpu_out.data(), N_HEADS * vd * sizeof(float));

    CompareResult r = compare_float(h_gpu_out.data(), cpu_out.data(), N_HEADS * vd);

    if (r.nan_count > 0) {
        test_fail(name, "%d NaN values", r.nan_count);
        fprintf(stderr, "  GPU first 8: ");
        for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", h_gpu_out[i]);
        fprintf(stderr, "\n  CPU first 8: ");
        for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", cpu_out[i]);
        fprintf(stderr, "\n");
        return;
    }
    // Recurrence has cascading errors; allow wider tolerance
    if (r.max_rel > 0.05f) {
        test_fail(name, "max_rel=%.4f > 5%%", (double)r.max_rel);
        return;
    }
    test_pass(name, "max_abs=%.2e, max_rel=%.4f", (double)r.max_abs, (double)r.max_rel);

    // Also check state was updated correctly
    std::vector<float> h_gpu_state(N_HEADS * kd * vd);
    g_state.download(h_gpu_state.data(), N_HEADS * kd * vd * sizeof(float));
    CompareResult rs = compare_float(h_gpu_state.data(), cpu_state.data(), N_HEADS * kd * vd);
    if (rs.nan_count > 0) {
        test_fail("dn_recurrence_state", "%d NaN in state", rs.nan_count);
    } else if (rs.max_rel > 0.05f) {
        test_fail("dn_recurrence_state", "max_rel=%.4f", (double)rs.max_rel);
    } else {
        test_pass("dn_recurrence_state", "max_abs=%.2e, max_rel=%.4f",
                  (double)rs.max_abs, (double)rs.max_rel);
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

    test_l2_norm(mod);
    test_conv1d_silu(mod);
    test_recurrence(mod);

    hipModuleUnload(mod);
    return test_summary();
}
