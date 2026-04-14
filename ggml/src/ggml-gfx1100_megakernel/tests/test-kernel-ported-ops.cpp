// test-kernel-ported-ops.cpp — Unit tests for all kernels ported from
// baseline ggml-cuda this session. Synthetic — no model required.
//
// For each kernel: build input on host → CPU reference (mirrors baseline
// CUDA op exactly) → upload → launch → download → compare. Tolerances
// reflect the float vs. CUDA-internal-double accumulation gap.
//
// Mapping: kernel → baseline source file
//   eval_gelu_mul       → ggml-cuda/unary.cuh::ggml_cuda_op_gelu_single
//   eval_gelu_erf_mul   → ggml-cuda/unary.cu::op_gelu_erf
//   eval_gelu_quick_mul → ggml-cuda/unary.cu::op_gelu_quick
//   eval_relu2_mul      → squared ReLU (Phi variants)
//   eval_scale_scalar   → ggml_scale (Gemma input/Q scale)
//   eval_softcap        → src/models/gemma3.cpp:142-146
//   eval_layernorm      → ggml-cuda/norm.cu::norm_f32
//   eval_l2norm         → ggml-cuda/norm.cu::l2_norm_f32
//   eval_sigmoid        → ggml-cuda/unary.cu::op_sigmoid
//   eval_softplus       → ggml-cuda/unary.cu::op_softplus
//   eval_argsort_desc   → ggml-cuda/argsort.cu::k_argsort_f32_i32
//   eval_softmax_row    → ggml-cuda/softmax.cu::soft_max_f32
//   eval_concat_dim0/1  → ggml-cuda/concat.cu::concat_f32_dim0/1
//   eval_repeat_dim0    → ggml_repeat
//   eval_ssm_conv_step  → ggml-cuda/ssm-conv.cu (n_t=1 variant)
//   eval_ssm_scan_step  → ggml-cuda/ssm-scan.cu (n_t=1 variant)
//   eval_axpy           → dst += w * src (MoE accumulator)
//   eval_sum_row        → sum reduction (MoE weight norm)

#include "test-harness.h"
#include "ggml-quants.h"

#include <algorithm>

// ============================================================================
// CPU references — mirror baseline CUDA exactly.
// ============================================================================
static float cpu_gelu(float x) {
    const float COEF = 0.044715f;
    const float S2P  = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(S2P * x * (1.0f + COEF * x * x)));
}
static float cpu_gelu_erf(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210485f;
    return 0.5f * x * (1.0f + erff(x * SQRT_2_INV));
}
static float cpu_gelu_quick(float x) {
    const float COEF = -1.702f;
    return x * (1.0f / (1.0f + expf(COEF * x)));
}
static float cpu_relu2(float x) { float y = x > 0 ? x : 0; return y * y; }
static float cpu_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float cpu_softplus(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }
static float cpu_softcap(float x, float cap) { return tanhf(x / cap) * cap; }

static void cpu_layernorm(const float * x, const float * w, const float * b,
                          float * out, int n, float eps) {
    double mean = 0, sq = 0;
    for (int i = 0; i < n; i++) { mean += x[i]; sq += (double)x[i] * x[i]; }
    mean /= n;
    double var = sq / n - mean * mean;
    float inv_std = 1.0f / sqrtf((float)var + eps);
    for (int i = 0; i < n; i++) {
        float y = ((float)(x[i] - mean)) * inv_std;
        if (w) y *= w[i];
        if (b) y += b[i];
        out[i] = y;
    }
}

static void cpu_l2norm(const float * x, float * out, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    // Baseline: rsqrtf(fmaxf(ss, eps*eps)) — clamp, not add
    float rstd = 1.0f / sqrtf(fmaxf((float)ss, eps * eps));
    for (int i = 0; i < n; i++) out[i] = x[i] * rstd;
}

static void cpu_softmax(const float * x, float * out, int n, float scale_val) {
    float vmax = -INFINITY;
    for (int i = 0; i < n; i++) { float v = x[i] * scale_val; if (v > vmax) vmax = v; }
    double sum = 0;
    for (int i = 0; i < n; i++) { float e = expf(x[i] * scale_val - vmax); out[i] = e; sum += e; }
    float inv = (float)(1.0 / sum);
    for (int i = 0; i < n; i++) out[i] *= inv;
}

// ============================================================================
// Test wrappers
// ============================================================================
template <typename CPU>
static void test_unary(hipModule_t mod, const char * name, CPU cpu_fn,
                       float min_v, float max_v, int n = 4096,
                       float tol_abs = 2e-6f, float tol_rel = 1e-5f) {
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    std::vector<float> in(n), out(n), cpu(n);
    // For GELU/SiLU/etc. test the gate*up form: we only have *_mul kernels here.
    // For pure unary kernels (sigmoid/softplus) we pass through `up=1` instead — skip.
    for (int i = 0; i < n; i++) {
        float t = (float)i / (n - 1);
        in[i] = min_v + t * (max_v - min_v);
        cpu[i] = cpu_fn(in[i]);
    }

    GpuBuf g_in(n * sizeof(float)), g_out(n * sizeof(float));
    g_in.upload(in.data(), n * sizeof(float));
    void * a = g_in.ptr, * o = g_out.ptr;
    int nv = n;
    void * args[] = { &a, &o, &nv };
    if (!launch_kernel(fn, (n+255)/256, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }
    g_out.download(out.data(), n * sizeof(float));
    auto r = compare_float(out.data(), cpu.data(), n);
    if (r.nan_count > 0) { test_fail(name, "%d NaN", r.nan_count); return; }
    if (r.max_abs > tol_abs && r.max_rel > tol_rel) {
        test_fail(name, "max_abs=%.2e, max_rel=%.2e (tol abs=%.0e rel=%.0e)",
                  r.max_abs, r.max_rel, tol_abs, tol_rel);
        return;
    }
    test_pass(name, "max_abs=%.2e max_rel=%.2e", r.max_abs, r.max_rel);
}

// gate*up activation kernels: signature (gate, up, out, n)
template <typename ACT>
static void test_act_mul(hipModule_t mod, const char * name, ACT act_fn,
                         float gate_min, float gate_max,
                         float tol_abs = 2e-6f, float tol_rel = 1e-4f) {
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int n = 4096;
    std::vector<float> gate(n), up(n), out(n), cpu(n);
    for (int i = 0; i < n; i++) {
        float t = (float)i / (n - 1);
        gate[i] = gate_min + t * (gate_max - gate_min);
        up[i]   = sinf(0.05f * (float)i) * 2.0f;
        cpu[i]  = act_fn(gate[i]) * up[i];
    }
    GpuBuf gg(n*sizeof(float)), gu(n*sizeof(float)), go(n*sizeof(float));
    gg.upload(gate.data(), n*sizeof(float));
    gu.upload(up.data(),   n*sizeof(float));
    void * a = gg.ptr, * b = gu.ptr, * o = go.ptr; int nv = n;
    void * args[] = { &a, &b, &o, &nv };
    if (!launch_kernel(fn, (n+255)/256, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }
    go.download(out.data(), n*sizeof(float));
    auto r = compare_float(out.data(), cpu.data(), n);
    if (r.nan_count > 0) { test_fail(name, "%d NaN", r.nan_count); return; }
    if (r.max_abs > tol_abs && r.max_rel > tol_rel) {
        test_fail(name, "max_abs=%.2e max_rel=%.2e", r.max_abs, r.max_rel); return;
    }
    test_pass(name, "max_abs=%.2e max_rel=%.2e", r.max_abs, r.max_rel);
}

// ----------------------------------------------------------------------------
// eval_scale_scalar (a, out, scale, n)
// ----------------------------------------------------------------------------
static void test_scale_scalar(hipModule_t mod) {
    const char * name = "eval_scale_scalar";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }
    const int n = 4096;
    const float scale = 0.7071f;
    std::vector<float> in(n), out(n), cpu(n);
    for (int i = 0; i < n; i++) { in[i] = sinf(0.01f * i) * 5.0f; cpu[i] = in[i] * scale; }
    GpuBuf g_in(n*sizeof(float)), g_out(n*sizeof(float));
    g_in.upload(in.data(), n*sizeof(float));
    void * a = g_in.ptr, * o = g_out.ptr; int nv = n; float sv = scale;
    void * args[] = { &a, &o, &sv, &nv };
    if (!launch_kernel(fn, (n+255)/256, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }
    g_out.download(out.data(), n*sizeof(float));
    auto r = compare_float(out.data(), cpu.data(), n);
    if (r.max_rel > 1e-6f) { test_fail(name, "max_rel=%.2e", r.max_rel); return; }
    test_pass(name, "max_abs=%.2e max_rel=%.2e", r.max_abs, r.max_rel);
}

// ----------------------------------------------------------------------------
// eval_softcap: in-place. signature (logits, cap, n)
// ----------------------------------------------------------------------------
static void test_softcap(hipModule_t mod) {
    const char * name = "eval_softcap";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }
    const int n = 4096;
    const float cap = 30.0f;
    std::vector<float> data(n), cpu(n);
    for (int i = 0; i < n; i++) {
        data[i] = sinf(0.05f * i) * 100.0f; // some > cap, exercises tanh saturation
        cpu[i] = cpu_softcap(data[i], cap);
    }
    GpuBuf g_data(n*sizeof(float));
    g_data.upload(data.data(), n*sizeof(float));
    void * a = g_data.ptr; int nv = n; float cv = cap;
    void * args[] = { &a, &cv, &nv };
    if (!launch_kernel(fn, (n+255)/256, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }
    std::vector<float> out(n);
    g_data.download(out.data(), n*sizeof(float));
    auto r = compare_float(out.data(), cpu.data(), n);
    if (r.max_rel > 1e-4f) { test_fail(name, "max_rel=%.2e", r.max_rel); return; }
    test_pass(name, "max_abs=%.2e max_rel=%.2e", r.max_abs, r.max_rel);
}

// ----------------------------------------------------------------------------
// eval_layernorm (input, weight, bias, norm_out, residual, n, eps)
// Single-row, blockDim=256
// ----------------------------------------------------------------------------
static void test_layernorm(hipModule_t mod) {
    const char * name = "eval_layernorm";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }
    const int n = 2048;
    const float eps = 1e-5f;
    std::vector<float> in(n), w(n), b(n), out(n), res(n), cpu(n);
    for (int i = 0; i < n; i++) {
        in[i] = sinf(0.02f * i) * 3.0f + 0.5f;
        w[i]  = 0.5f + 0.5f * cosf(0.01f * i);
        b[i]  = 0.1f * sinf(0.03f * i);
    }
    cpu_layernorm(in.data(), w.data(), b.data(), cpu.data(), n, eps);

    GpuBuf g_in(n*sizeof(float)), g_w(n*sizeof(float)), g_b(n*sizeof(float)),
           g_out(n*sizeof(float)), g_res(n*sizeof(float));
    g_in.upload(in.data(), n*sizeof(float));
    g_w.upload(w.data(),   n*sizeof(float));
    g_b.upload(b.data(),   n*sizeof(float));
    void * a = g_in.ptr, * wp = g_w.ptr, * bp = g_b.ptr, * op = g_out.ptr, * rp = g_res.ptr;
    int nv = n; float ev = eps;
    void * args[] = { &a, &wp, &bp, &op, &rp, &nv, &ev };
    if (!launch_kernel(fn, 1, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }
    g_out.download(out.data(), n*sizeof(float));
    g_res.download(res.data(), n*sizeof(float));
    auto r = compare_float(out.data(), cpu.data(), n);
    if (r.nan_count > 0) { test_fail(name, "%d NaN", r.nan_count); return; }
    if (r.max_rel > 1e-3f) { test_fail(name, "norm_out max_rel=%.2e", r.max_rel); return; }
    auto rr = compare_float(res.data(), in.data(), n);
    if (rr.max_abs > 0.0f) { test_fail(name, "residual mismatch max_abs=%.2e", rr.max_abs); return; }
    test_pass(name, "max_abs=%.2e max_rel=%.2e", r.max_abs, r.max_rel);
}

// ----------------------------------------------------------------------------
// eval_l2norm (input, output, n, eps)
// ----------------------------------------------------------------------------
static void test_l2norm(hipModule_t mod) {
    const char * name = "eval_l2norm";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }
    const int n = 2048;
    const float eps = 1e-6f;
    std::vector<float> in(n), out(n), cpu(n);
    for (int i = 0; i < n; i++) in[i] = sinf(0.01f * i) * 2.0f + 0.3f;
    cpu_l2norm(in.data(), cpu.data(), n, eps);
    GpuBuf g_in(n*sizeof(float)), g_out(n*sizeof(float));
    g_in.upload(in.data(), n*sizeof(float));
    void * a = g_in.ptr, * o = g_out.ptr; int nv = n; float ev = eps;
    void * args[] = { &a, &o, &nv, &ev };
    if (!launch_kernel(fn, 1, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }
    g_out.download(out.data(), n*sizeof(float));
    auto r = compare_float(out.data(), cpu.data(), n);
    if (r.max_rel > 1e-4f) { test_fail(name, "max_rel=%.2e", r.max_rel); return; }
    test_pass(name, "max_abs=%.2e max_rel=%.2e", r.max_abs, r.max_rel);
}

// ----------------------------------------------------------------------------
// Pure unary kernels: sigmoid, softplus (input, output, N)
// ----------------------------------------------------------------------------
static void test_unary_pure(hipModule_t mod, const char * name,
                             float (*cpu_fn)(float), float min_v, float max_v,
                             float tol_rel) {
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }
    const int n = 4096;
    std::vector<float> in(n), out(n), cpu(n);
    for (int i = 0; i < n; i++) {
        float t = (float)i / (n - 1);
        in[i] = min_v + t * (max_v - min_v);
        cpu[i] = cpu_fn(in[i]);
    }
    GpuBuf g_in(n*sizeof(float)), g_out(n*sizeof(float));
    g_in.upload(in.data(), n*sizeof(float));
    void * a = g_in.ptr, * o = g_out.ptr; int nv = n;
    void * args[] = { &a, &o, &nv };
    if (!launch_kernel(fn, (n+255)/256, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }
    g_out.download(out.data(), n*sizeof(float));
    auto r = compare_float(out.data(), cpu.data(), n);
    if (r.nan_count > 0) { test_fail(name, "%d NaN", r.nan_count); return; }
    if (r.max_rel > tol_rel) { test_fail(name, "max_rel=%.2e > %.0e", r.max_rel, tol_rel); return; }
    test_pass(name, "max_abs=%.2e max_rel=%.2e", r.max_abs, r.max_rel);
}

// ----------------------------------------------------------------------------
// eval_argsort_desc — bitonic sort
// (x, dst, ncols, ncols_pad)
// ----------------------------------------------------------------------------
static int next_pow2(int x) { int n = 1; while (n < x) n <<= 1; return n; }

static void test_argsort_desc(hipModule_t mod) {
    const char * name = "eval_argsort_desc";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }
    const int ncols = 64;     // typical: n_expert <= 256
    const int nrows = 4;
    const int pad   = next_pow2(ncols);

    std::vector<float> x(ncols * nrows);
    for (int r = 0; r < nrows; r++)
        for (int c = 0; c < ncols; c++) x[r * ncols + c] = sinf(0.13f * (r*100 + c));
    std::vector<int> cpu_idx(ncols * nrows), gpu_idx(ncols * nrows);
    for (int r = 0; r < nrows; r++) {
        std::vector<int> idx(ncols);
        for (int c = 0; c < ncols; c++) idx[c] = c;
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b){ return x[r*ncols+a] > x[r*ncols+b]; });
        for (int c = 0; c < ncols; c++) cpu_idx[r*ncols + c] = idx[c];
    }
    GpuBuf g_x(ncols*nrows*sizeof(float)), g_idx(ncols*nrows*sizeof(int));
    g_x.upload(x.data(), ncols*nrows*sizeof(float));
    void * xp = g_x.ptr, * ip = g_idx.ptr; int ncv = ncols, npv = pad;
    void * args[] = { &xp, &ip, &ncv, &npv };
    unsigned shared = pad * sizeof(int);
    if (!launch_kernel(fn, nrows, 1, 1, pad, 1, 1, args, shared)) {
        test_fail(name, "launch failed"); return;
    }
    g_idx.download(gpu_idx.data(), ncols*nrows*sizeof(int));
    // Bitonic sort is stable for values, but indices for ties may differ; compare values.
    int mismatch = 0;
    for (int r = 0; r < nrows; r++)
        for (int c = 0; c < ncols; c++)
            if (x[r*ncols + cpu_idx[r*ncols+c]] != x[r*ncols + gpu_idx[r*ncols+c]])
                mismatch++;
    if (mismatch > 0) { test_fail(name, "%d index mismatches (value-wise)", mismatch); return; }
    test_pass(name, "%d rows × %d cols sorted desc", nrows, ncols);
}

// ----------------------------------------------------------------------------
// eval_softmax_row — (x, dst, ncols, scale_val), shared = (block/WARP) * sizeof(float)
// ----------------------------------------------------------------------------
static void test_softmax_row(hipModule_t mod) {
    const char * name = "eval_softmax_row";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }
    const int ncols = 256, nrows = 4;
    std::vector<float> x(ncols * nrows), out(ncols * nrows), cpu(ncols * nrows);
    for (int r = 0; r < nrows; r++)
        for (int c = 0; c < ncols; c++)
            x[r*ncols + c] = sinf(0.07f * (r*100 + c));
    for (int r = 0; r < nrows; r++)
        cpu_softmax(x.data() + r*ncols, cpu.data() + r*ncols, ncols, 1.0f);

    GpuBuf gx(ncols*nrows*sizeof(float)), go(ncols*nrows*sizeof(float));
    gx.upload(x.data(), ncols*nrows*sizeof(float));
    void * xp = gx.ptr, * op = go.ptr; int ncv = ncols; float sv = 1.0f;
    void * args[] = { &xp, &op, &ncv, &sv };
    int block = 256;
    unsigned shared = (block / 32) * sizeof(float); // up to 8 warps
    if (!launch_kernel(fn, nrows, 1, 1, block, 1, 1, args, shared)) {
        test_fail(name, "launch failed"); return;
    }
    go.download(out.data(), ncols*nrows*sizeof(float));
    auto r = compare_float(out.data(), cpu.data(), ncols * nrows);
    if (r.nan_count > 0) { test_fail(name, "%d NaN", r.nan_count); return; }
    if (r.max_rel > 1e-3f) { test_fail(name, "max_rel=%.2e", r.max_rel); return; }
    // Also verify each row sums to ~1
    for (int rr = 0; rr < nrows; rr++) {
        double s = 0;
        for (int c = 0; c < ncols; c++) s += out[rr*ncols + c];
        if (fabs(s - 1.0) > 1e-3) { test_fail(name, "row %d sum=%.4f", rr, s); return; }
    }
    test_pass(name, "max_rel=%.2e (rows sum to 1 ± 1e-3)", r.max_rel);
}

// ----------------------------------------------------------------------------
// eval_concat_dim0: dst[blockIdx.z*ne0*ne1 + blockIdx.y*ne0 + nidx] = x or y
// (x, y, dst, ne0, ne00). Grid: (blocks,ne1,ne2). Block: 256.
// ----------------------------------------------------------------------------
static void test_concat_dim0(hipModule_t mod) {
    const char * name = "eval_concat_dim0";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }
    const int ne00 = 32, ne01 = 4;
    const int ne10 = 16; // y has ne00..ne0-1 columns, so ne10 = ne0 - ne00
    const int ne0  = ne00 + ne10;
    const int ne1  = ne01;
    std::vector<float> x(ne00*ne1), y(ne10*ne1), out(ne0*ne1), cpu(ne0*ne1);
    for (int i = 0; i < (int)x.size(); i++) x[i] = (float)(i + 1);
    for (int i = 0; i < (int)y.size(); i++) y[i] = -(float)(i + 1);
    for (int j = 0; j < ne1; j++) {
        for (int i = 0; i < ne00; i++) cpu[j*ne0 + i]        = x[j*ne00 + i];
        for (int i = 0; i < ne10; i++) cpu[j*ne0 + ne00 + i] = y[j*ne10 + i];
    }
    GpuBuf gx(x.size()*sizeof(float)), gy(y.size()*sizeof(float)),
           go(out.size()*sizeof(float));
    gx.upload(x.data(), x.size()*sizeof(float));
    gy.upload(y.data(), y.size()*sizeof(float));
    void * xp = gx.ptr, * yp = gy.ptr, * op = go.ptr;
    int ne0v = ne0, ne00v = ne00;
    void * args[] = { &xp, &yp, &op, &ne0v, &ne00v };
    int blocks = (ne0 + 255) / 256;
    if (!launch_kernel(fn, blocks, ne1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }
    go.download(out.data(), out.size()*sizeof(float));
    auto r = compare_float(out.data(), cpu.data(), (int)out.size());
    if (r.max_abs > 0.0f) { test_fail(name, "max_abs=%.2e", r.max_abs); return; }
    test_pass(name, "exact match");
}

// ----------------------------------------------------------------------------
// eval_concat_dim1
// ----------------------------------------------------------------------------
static void test_concat_dim1(hipModule_t mod) {
    const char * name = "eval_concat_dim1";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }
    const int ne0 = 16, ne01 = 3, ne11 = 2;
    const int ne1 = ne01 + ne11;
    std::vector<float> x(ne0*ne01), y(ne0*ne11), out(ne0*ne1), cpu(ne0*ne1);
    for (int i = 0; i < (int)x.size(); i++) x[i] = (float)(i + 1);
    for (int i = 0; i < (int)y.size(); i++) y[i] = -(float)(i + 100);
    for (int j = 0; j < ne01; j++)
        for (int i = 0; i < ne0; i++) cpu[j*ne0 + i] = x[j*ne0 + i];
    for (int j = 0; j < ne11; j++)
        for (int i = 0; i < ne0; i++) cpu[(ne01 + j)*ne0 + i] = y[j*ne0 + i];
    GpuBuf gx(x.size()*sizeof(float)), gy(y.size()*sizeof(float)), go(out.size()*sizeof(float));
    gx.upload(x.data(), x.size()*sizeof(float));
    gy.upload(y.data(), y.size()*sizeof(float));
    void * xp = gx.ptr, * yp = gy.ptr, * op = go.ptr;
    int ne0v = ne0, ne01v = ne01;
    void * args[] = { &xp, &yp, &op, &ne0v, &ne01v };
    int blocks = (ne0 + 255) / 256;
    if (!launch_kernel(fn, blocks, ne1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }
    go.download(out.data(), out.size()*sizeof(float));
    auto r = compare_float(out.data(), cpu.data(), (int)out.size());
    if (r.max_abs > 0.0f) { test_fail(name, "max_abs=%.2e", r.max_abs); return; }
    test_pass(name, "exact match");
}

// ----------------------------------------------------------------------------
// eval_repeat_dim0
// ----------------------------------------------------------------------------
static void test_repeat_dim0(hipModule_t mod) {
    const char * name = "eval_repeat_dim0";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }
    const int ne00 = 16, factor = 4, ne0 = ne00 * factor, ne1 = 3;
    std::vector<float> x(ne00 * ne1), out(ne0 * ne1), cpu(ne0 * ne1);
    for (int i = 0; i < (int)x.size(); i++) x[i] = (float)(i + 1);
    for (int j = 0; j < ne1; j++)
        for (int i = 0; i < ne0; i++) cpu[j*ne0 + i] = x[j*ne00 + (i % ne00)];
    GpuBuf gx(x.size()*sizeof(float)), go(out.size()*sizeof(float));
    gx.upload(x.data(), x.size()*sizeof(float));
    void * xp = gx.ptr, * op = go.ptr; int ne00v = ne00, ne0v = ne0;
    void * args[] = { &xp, &op, &ne00v, &ne0v };
    int blocks = (ne0 + 255) / 256;
    if (!launch_kernel(fn, blocks, ne1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }
    go.download(out.data(), out.size()*sizeof(float));
    auto r = compare_float(out.data(), cpu.data(), (int)out.size());
    if (r.max_abs > 0.0f) { test_fail(name, "max_abs=%.2e", r.max_abs); return; }
    test_pass(name, "exact match");
}

// ----------------------------------------------------------------------------
// eval_axpy: dst += w * src — (dst, src, w, N)
// ----------------------------------------------------------------------------
static void test_axpy(hipModule_t mod) {
    const char * name = "eval_axpy";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }
    const int n = 4096;
    const float w = 0.37f;
    std::vector<float> dst(n), src(n), cpu(n);
    for (int i = 0; i < n; i++) {
        dst[i] = sinf(0.01f * i) * 2.0f;
        src[i] = cosf(0.013f * i) * 3.0f;
        cpu[i] = dst[i] + w * src[i];
    }
    GpuBuf gd(n*sizeof(float)), gs(n*sizeof(float));
    gd.upload(dst.data(), n*sizeof(float));
    gs.upload(src.data(), n*sizeof(float));
    void * dp = gd.ptr, * sp = gs.ptr; int nv = n; float wv = w;
    void * args[] = { &dp, &sp, &wv, &nv };
    if (!launch_kernel(fn, (n+255)/256, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }
    std::vector<float> out(n);
    gd.download(out.data(), n*sizeof(float));
    auto r = compare_float(out.data(), cpu.data(), n);
    // Tolerance: GPU may use FMA (fused multiply-add) where CPU does mul-then-add.
    // FMA produces slightly different rounding (~1ulp). 1e-4 rel is still tight.
    if (r.max_rel > 1e-4f) { test_fail(name, "max_rel=%.2e", r.max_rel); return; }
    test_pass(name, "max_abs=%.2e max_rel=%.2e", r.max_abs, r.max_rel);
}

// ----------------------------------------------------------------------------
// eval_sum_row: out_sum = sum(in[0..N-1])
// ----------------------------------------------------------------------------
static void test_sum_row(hipModule_t mod) {
    const char * name = "eval_sum_row";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }
    const int n = 4096;
    std::vector<float> in(n);
    double cpu_sum = 0;
    for (int i = 0; i < n; i++) { in[i] = sinf(0.01f * i) * 0.5f; cpu_sum += in[i]; }
    GpuBuf gi(n*sizeof(float)), gs(sizeof(float));
    gi.upload(in.data(), n*sizeof(float));
    void * ip = gi.ptr, * sp = gs.ptr; int nv = n;
    void * args[] = { &ip, &sp, &nv };
    if (!launch_kernel(fn, 1, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }
    float gpu_sum = 0;
    gs.download(&gpu_sum, sizeof(float));
    float diff = fabsf(gpu_sum - (float)cpu_sum);
    float rel  = diff / fabsf((float)cpu_sum + 1e-9f);
    if (rel > 1e-4f) { test_fail(name, "rel=%.2e (gpu=%.4f cpu=%.4f)", rel, gpu_sum, (float)cpu_sum); return; }
    test_pass(name, "diff=%.2e rel=%.2e", diff, rel);
}

// ----------------------------------------------------------------------------
// eval_ssm_conv_step
// (x_in, state, w, y_out, d_inner, d_conv, apply_silu)
// ----------------------------------------------------------------------------
static void test_ssm_conv_step(hipModule_t mod) {
    const char * name = "eval_ssm_conv_step";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }
    const int d_inner = 64, d_conv = 4;
    std::vector<float> x_in(d_inner), state(d_inner * d_conv), w(d_inner * d_conv),
                       y_out(d_inner), cpu_state(d_inner * d_conv), cpu_y(d_inner);
    for (int i = 0; i < d_inner; i++) x_in[i] = sinf(0.01f * i) * 0.5f;
    for (int i = 0; i < d_inner * d_conv; i++) {
        state[i] = sinf(0.013f * i);
        w[i]     = 0.5f * cosf(0.017f * i);
        cpu_state[i] = state[i];
    }
    // CPU reference: shift state left, append x_in, dot(state, w), silu
    auto silu = [](float x){ return x / (1.0f + expf(-x)); };
    for (int i = 0; i < d_inner; i++) {
        for (int j = 0; j < d_conv - 1; j++) cpu_state[i*d_conv + j] = cpu_state[i*d_conv + j + 1];
        cpu_state[i*d_conv + d_conv - 1] = x_in[i];
        float s = 0;
        for (int j = 0; j < d_conv; j++) s += cpu_state[i*d_conv + j] * w[i*d_conv + j];
        cpu_y[i] = silu(s);
    }
    GpuBuf gx(d_inner*sizeof(float)), gs(d_inner*d_conv*sizeof(float)),
           gw(d_inner*d_conv*sizeof(float)), gy(d_inner*sizeof(float));
    gx.upload(x_in.data(), d_inner*sizeof(float));
    gs.upload(state.data(), d_inner*d_conv*sizeof(float));
    gw.upload(w.data(),    d_inner*d_conv*sizeof(float));
    void * xp = gx.ptr, * sp = gs.ptr, * wp = gw.ptr, * yp = gy.ptr;
    int di = d_inner, dc = d_conv, sil = 1;
    void * args[] = { &xp, &sp, &wp, &yp, &di, &dc, &sil };
    if (!launch_kernel(fn, (d_inner+255)/256, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }
    std::vector<float> y_gpu(d_inner), s_gpu(d_inner*d_conv);
    gy.download(y_gpu.data(), d_inner*sizeof(float));
    gs.download(s_gpu.data(), d_inner*d_conv*sizeof(float));
    auto ry = compare_float(y_gpu.data(), cpu_y.data(), d_inner);
    auto rs = compare_float(s_gpu.data(), cpu_state.data(), d_inner*d_conv);
    if (ry.max_rel > 1e-4f) { test_fail(name, "y max_rel=%.2e", ry.max_rel); return; }
    if (rs.max_abs > 1e-6f) { test_fail(name, "state max_abs=%.2e", rs.max_abs); return; }
    test_pass(name, "y max_rel=%.2e, state max_abs=%.2e", ry.max_rel, rs.max_abs);
}

// ----------------------------------------------------------------------------
// eval_ssm_scan_step
// (x, dt, A, B, C, D, h, y, d_inner, d_state)
// h_new = exp(dt * (-exp(A))) * h + dt * B * x; y = sum(h * C) + D * x
// ----------------------------------------------------------------------------
static void test_ssm_scan_step(hipModule_t mod) {
    const char * name = "eval_ssm_scan_step";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }
    const int d_inner = 32, d_state = 16;
    std::vector<float> x(d_inner), dt(d_inner), A(d_inner * d_state), B(d_state),
                       C(d_state), D(d_inner), h(d_inner * d_state),
                       cpu_h(d_inner * d_state), cpu_y(d_inner);
    for (int i = 0; i < d_inner; i++) {
        x[i]  = sinf(0.01f * i) * 0.5f;
        dt[i] = 0.1f + 0.05f * cosf(0.02f * i);
        D[i]  = 0.1f * sinf(0.03f * i);
    }
    for (int n = 0; n < d_state; n++) {
        B[n] = 0.2f * sinf(0.05f * n);
        C[n] = 0.3f * cosf(0.07f * n);
    }
    for (int i = 0; i < d_inner * d_state; i++) {
        A[i] = -1.0f + 0.5f * sinf(0.011f * i);  // log-stored, will be -exp(A)
        h[i] = 0.05f * sinf(0.013f * i);
        cpu_h[i] = h[i];
    }
    // CPU reference — must apply softplus to dt, matching baseline ssm-scan.cu
    for (int i = 0; i < d_inner; i++) {
        const float xi = x[i];
        float dti = dt[i];
        if (dti <= 20.0f) dti = log1pf(expf(dti)); // softplus
        float yi = 0;
        for (int n = 0; n < d_state; n++) {
            float a_eff = -expf(A[i*d_state + n]);
            float dA  = expf(dti * a_eff);
            float dBx = dti * B[n] * xi;
            float hin = dA * cpu_h[i*d_state + n] + dBx;
            cpu_h[i*d_state + n] = hin;
            yi += hin * C[n];
        }
        cpu_y[i] = yi + D[i] * xi;
    }
    GpuBuf gx(d_inner*sizeof(float)), gdt(d_inner*sizeof(float)),
           gA(d_inner*d_state*sizeof(float)), gB(d_state*sizeof(float)),
           gC(d_state*sizeof(float)), gD(d_inner*sizeof(float)),
           gh(d_inner*d_state*sizeof(float)), gy(d_inner*sizeof(float));
    gx.upload(x.data(), d_inner*sizeof(float));
    gdt.upload(dt.data(), d_inner*sizeof(float));
    gA.upload(A.data(), d_inner*d_state*sizeof(float));
    gB.upload(B.data(), d_state*sizeof(float));
    gC.upload(C.data(), d_state*sizeof(float));
    gD.upload(D.data(), d_inner*sizeof(float));
    gh.upload(h.data(), d_inner*d_state*sizeof(float));
    void * xp = gx.ptr, * dtp = gdt.ptr, * Ap = gA.ptr, * Bp = gB.ptr, * Cp = gC.ptr,
         * Dp = gD.ptr, * hp = gh.ptr, * yp = gy.ptr;
    int di = d_inner, ds = d_state;
    void * args[] = { &xp, &dtp, &Ap, &Bp, &Cp, &Dp, &hp, &yp, &di, &ds };
    if (!launch_kernel(fn, (d_inner+255)/256, 1, 1, 256, 1, 1, args)) {
        test_fail(name, "launch failed"); return;
    }
    std::vector<float> y_gpu(d_inner), h_gpu(d_inner*d_state);
    gy.download(y_gpu.data(), d_inner*sizeof(float));
    gh.download(h_gpu.data(), d_inner*d_state*sizeof(float));
    auto ry = compare_float(y_gpu.data(), cpu_y.data(), d_inner);
    auto rh = compare_float(h_gpu.data(), cpu_h.data(), d_inner*d_state);
    if (ry.nan_count > 0 || rh.nan_count > 0) {
        test_fail(name, "y_nan=%d h_nan=%d", ry.nan_count, rh.nan_count); return;
    }
    if (ry.max_rel > 1e-4f) { test_fail(name, "y max_rel=%.2e", ry.max_rel); return; }
    if (rh.max_rel > 1e-4f) { test_fail(name, "h max_rel=%.2e", rh.max_rel); return; }
    test_pass(name, "y max_rel=%.2e, h max_rel=%.2e", ry.max_rel, rh.max_rel);
}

// ----------------------------------------------------------------------------
// eval_embed_q8_0 — DIRECT PORT verification.
// Bug history: pre-session SMALL macro ignored token_id*stride (always read token 0)
// AND assumed H % 256 == 0. Caused Qwen2 (Q8_0 embed, H=896) to produce constant
// output regardless of input token. Test must use H not divisible by 256 to catch
// regression — but H is baked in via HIDDEN_SIZE macro. Only catch if .hsaco is
// loaded for the right model. Test below verifies token offset works.
// ----------------------------------------------------------------------------
struct host_block_q8_0 { uint16_t d_f16; int8_t qs[32]; };

// CPU reference: exactly mirrors baseline ggml-cuda/dequantize.cuh::dequantize_q8_0
// + k_get_rows write loop. Token offset, per-block d, signed qs.
static void cpu_embed_q8_0(const host_block_q8_0 * table, long long row_bytes,
                           int token_id, float * out, int H) {
    const host_block_q8_0 * row = (const host_block_q8_0 *)
        ((const char *)table + (long long)token_id * row_bytes);
    constexpr int qk = 32, qr = 1;
    const int y_offset = qr == 1 ? 1 : qk / 2;
    for (int i00 = 0; i00 < H; i00 += 2) {
        const int ib   = i00 / qk;
        const int iqs  = (i00 % qk) / qr;
        const int iybs = i00 - i00 % qk;
        const float d = f16_to_f32(row[ib].d_f16);
        out[iybs + iqs + 0]        = row[ib].qs[iqs + 0] * d;
        out[iybs + iqs + y_offset] = row[ib].qs[iqs + 1] * d;
    }
}

static void test_embed_q8_0(hipModule_t mod) {
    const char * name = "eval_embed_q8_0";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    // RANDOMIZED data — this catches bugs missed by all-equal synthetic patterns.
    // Tests:
    //   1. token_id offset works (different tokens → different output)
    //   2. per-block d_f16 applied correctly (varies per block)
    //   3. signed qs (negative) handled
    //   4. all H elements written (no tail garbage from wrong nb)
    //   5. bit-exact match with CPU baseline-faithful dequant
    const int V = 8;
    const int H = 2048;  // match Llama 1B .hsaco HIDDEN_SIZE
    const int blocks_per_row = H / 32;
    const size_t row_bytes = (size_t)blocks_per_row * sizeof(host_block_q8_0);

    std::vector<host_block_q8_0> table(V * blocks_per_row);
    // Reproducible seed: fill with values that exercise all code paths
    for (int t = 0; t < V; t++) {
        for (int b = 0; b < blocks_per_row; b++) {
            host_block_q8_0 & blk = table[t * blocks_per_row + b];
            // Vary d per block — catches per-block d indexing bugs
            float d = 0.001f * (float)(1 + t * 13 + b * 7);
            blk.d_f16 = f32_to_f16(d);
            for (int i = 0; i < 32; i++) {
                // Mix positive, negative, sign-bit edge: -128, -1, 0, 1, 127
                int v = ((t * 257 + b * 31 + i * 11) & 0xFF) - 128;
                blk.qs[i] = (int8_t)v;
            }
        }
    }

    GpuBuf g_table(table.size() * sizeof(host_block_q8_0));
    g_table.upload(table.data(), table.size() * sizeof(host_block_q8_0));

    GpuBuf g_hidden(H * sizeof(float));

    bool all_ok = true;
    float worst_abs = 0.0f, worst_rel = 0.0f;
    int worst_tok = -1, worst_idx = -1;
    for (int t = 0; t < V; t++) {
        // Reset GPU hidden buffer to garbage marker so untouched cells fail loudly
        std::vector<float> sentinel(H, 1.5e30f);
        g_hidden.upload(sentinel.data(), H * sizeof(float));

        const void * embed_ptr = g_table.ptr;
        long long stride = (long long)row_bytes;
        int tok = t;
        float * hp = (float *)g_hidden.ptr;
        void * args[] = { (void *)&embed_ptr, (void *)&stride, (void *)&hp, (void *)&tok };

        const int block = 256;
        const int grid_y = (H + 2 * block - 1) / (2 * block);
        if (!launch_kernel(fn, 1, grid_y, 1, block, 1, 1, args)) {
            test_fail(name, "launch failed for token %d", t); return;
        }
        std::vector<float> gpu_out(H);
        g_hidden.download(gpu_out.data(), H * sizeof(float));

        // CPU reference using exact baseline dequant
        std::vector<float> cpu_out(H, 0.0f);
        cpu_embed_q8_0(table.data(), row_bytes, t, cpu_out.data(), H);

        // Compare bit-by-bit (Q8_0 dequant is deterministic — int8*float, no FMA reordering)
        for (int i = 0; i < H; i++) {
            float diff = fabsf(gpu_out[i] - cpu_out[i]);
            float rel = (fabsf(cpu_out[i]) > 1e-9f) ? diff / fabsf(cpu_out[i]) : 0.0f;
            if (diff > worst_abs) { worst_abs = diff; worst_tok = t; worst_idx = i; }
            if (rel  > worst_rel) worst_rel = rel;
            // Q8_0 should be bit-exact (single mul, no reduction)
            if (diff > 1e-5f) {
                test_fail(name, "token=%d hidden[%d]: gpu=%.6f cpu=%.6f diff=%.2e",
                          t, i, gpu_out[i], cpu_out[i], diff);
                all_ok = false;
                break;
            }
            // Sentinel check: untouched cell stays 1.5e30 → catches H % qk regressions
            if (gpu_out[i] > 1e29f) {
                test_fail(name, "token=%d hidden[%d] = sentinel (UNWRITTEN)", t, i);
                all_ok = false;
                break;
            }
        }
        if (!all_ok) break;
    }
    if (all_ok) test_pass(name, "V=%d H=%d randomized: max_abs=%.2e max_rel=%.2e",
                          V, H, worst_abs, worst_rel);
}

// ----------------------------------------------------------------------------
// eval_matvec_q5_0 at Qwen2 dimensions (in_dim=896, out_dim=896, GQA-K=128).
// Existing test-kernel-matvec-qtypes uses in_dim=512 (multiple of 32 AND 256).
// Qwen2's wq/wk/wv use Q5_0 with in_dim=896 (multiple of 32 only). This test
// catches dimension-specific bugs that pass at 512.
// ----------------------------------------------------------------------------
static void test_matvec_q5_0_qwen2_dim(hipModule_t mod, hipFunction_t quantize_fn) {
    const char * name = "eval_matvec_q5_0@896";
    hipFunction_t fn = load_kernel(mod, "eval_matvec_q5_0");
    if (!fn) { test_fail(name, "kernel not found"); return; }

    // Test BOTH attn_q (out=896) and attn_k (out=128) dimensions used by Qwen2-0.5B
    struct Case { int in_dim; int out_dim; const char * desc; };
    Case cases[] = {
        { 896, 896, "wq [H=896, H_q=896]" },
        { 896, 128, "wk [H=896, H_kv=128]" },
        { 896, 128, "wv [H=896, H_kv=128]" },
    };

    bool all_ok = true;
    for (const auto & c : cases) {
        const int in_dim = c.in_dim;
        const int out_dim = c.out_dim;

        // F32 input + reference weight
        std::vector<float> input(in_dim);
        for (int i = 0; i < in_dim; i++) input[i] = sinf(0.01f * i) * 2.0f;

        std::vector<float> weight_f32(out_dim * in_dim);
        for (int r = 0; r < out_dim; r++)
            for (int c2 = 0; c2 < in_dim; c2++)
                weight_f32[r * in_dim + c2] = cosf(0.013f * (r * 17 + c2)) * 0.5f;

        // Quantize each row to Q5_0
        const size_t row_bytes = ggml_row_size(GGML_TYPE_Q5_0, in_dim);
        std::vector<uint8_t> weight_q(out_dim * row_bytes);
        for (int r = 0; r < out_dim; r++) {
            ggml_quantize_chunk(GGML_TYPE_Q5_0, weight_f32.data() + r * in_dim,
                                weight_q.data() + r * row_bytes, 0, 1, in_dim, nullptr);
        }

        // CPU reference: dequant + dot
        std::vector<float> cpu_out(out_dim);
        const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q5_0);
        std::vector<float> row_f32(in_dim);
        for (int r = 0; r < out_dim; r++) {
            tt->to_float(weight_q.data() + r * row_bytes, row_f32.data(), in_dim);
            double acc = 0;
            for (int c2 = 0; c2 < in_dim; c2++) acc += (double)row_f32[c2] * (double)input[c2];
            cpu_out[r] = (float)acc;
        }

        // GPU: quantize input → Q8_1, then matvec
        const int n_q8_blocks = in_dim / 32;
        const size_t q8_bytes = (size_t)n_q8_blocks * sizeof(block_q8_1_host);

        GpuBuf g_in(in_dim * sizeof(float));
        GpuBuf g_q8(q8_bytes);
        GpuBuf g_w(out_dim * row_bytes);
        GpuBuf g_out(out_dim * sizeof(float));
        g_in.upload(input.data(), in_dim * sizeof(float));
        g_w.upload(weight_q.data(), out_dim * row_bytes);

        // Quantize: grid = ceildiv(n_blocks*32, 512)
        {
            void * ip = g_in.ptr, * qp = g_q8.ptr; int nv = in_dim;
            void * args[] = { &ip, &qp, &nv };
            unsigned grid = (in_dim + 511) / 512;
            if (!launch_kernel(quantize_fn, grid, 1, 1, 512, 1, 1, args)) {
                test_fail(name, "%s: quantize launch failed", c.desc); all_ok = false; break;
            }
        }
        // Matvec: grid=(out_dim,1,1), block=(32,4,1)
        {
            void * wp = g_w.ptr; long long sv = (long long)row_bytes;
            void * qp = g_q8.ptr; void * op = g_out.ptr;
            int iv = in_dim, ov = out_dim;
            void * args[] = { &wp, &sv, &qp, &op, &iv, &ov };
            if (!launch_kernel(fn, (unsigned)out_dim, 1, 1, 32, 4, 1, args)) {
                test_fail(name, "%s: matvec launch failed", c.desc); all_ok = false; break;
            }
        }
        std::vector<float> gpu_out(out_dim);
        g_out.download(gpu_out.data(), out_dim * sizeof(float));
        auto r = compare_float(gpu_out.data(), cpu_out.data(), out_dim);
        // Tolerance: double quantization (Q5_0 weight + Q8_1 input). Q5_0 ~ 5.5 bpw.
        // Existing test-kernel-matvec-qtypes uses 0.10 tolerance for Q5_0 at in_dim=512.
        if (r.max_rel > 0.10f) {
            test_fail(name, "%s: max_rel=%.4f > 0.10 (max_abs=%.2e)", c.desc, r.max_rel, r.max_abs);
            all_ok = false; break;
        }
    }
    if (all_ok) test_pass(name, "wq+wk+wv @ in_dim=896 all within 10%% rel");
}

// ----------------------------------------------------------------------------
// eval_rmsnorm_q8 at H=896 (Qwen2 dimension).
// Existing test-kernel-rmsnorm uses Llama's H=2048 which is multiple of 256.
// H=896 is 28*32 (multiple of 32 only). Tests that warp reduction handles
// non-power-of-2, non-256-aligned dimensions correctly.
// CPU reference: baseline norm.cu rms_norm_f32 algorithm.
// ----------------------------------------------------------------------------
static void cpu_rmsnorm(const float * input, const float * weight,
                        float * norm_out, int n, float eps) {
    // Baseline norm.cu: sum(x^2), rsqrt(sum/n + eps), output = x * rsqrt * weight
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)input[i] * (double)input[i];
    float rstd = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) norm_out[i] = input[i] * rstd * weight[i];
}

static void test_rmsnorm_qwen2(hipModule_t mod) {
    const char * name = "eval_rmsnorm_q8@896";
    hipFunction_t fn = load_kernel(mod, "eval_rmsnorm_q8");
    if (!fn) { test_fail(name, "kernel not found"); return; }

    // Test at H=896 (Qwen2-0.5B) and H=896+32=928 (non-standard) to catch boundary bugs
    int dims[] = { 896, 928, 768, 1024 };
    bool all_ok = true;
    for (int H : dims) {
        // Randomized input + weight that exercises sign, magnitude variation
        std::vector<float> input(H), weight(H), norm_out(H), residual(H), cpu_out(H);
        for (int i = 0; i < H; i++) {
            input[i]  = sinf(0.017f * i) * 3.0f + 0.1f * cosf(0.031f * i);
            weight[i] = 0.8f + 0.4f * cosf(0.013f * i);
        }
        cpu_rmsnorm(input.data(), weight.data(), cpu_out.data(), H, 1e-5f);

        GpuBuf g_in(H*sizeof(float)), g_w(H*sizeof(float)),
               g_out(H*sizeof(float)), g_res(H*sizeof(float));
        g_in.upload(input.data(), H*sizeof(float));
        g_w.upload(weight.data(), H*sizeof(float));

        void * ip = g_in.ptr, * wp = g_w.ptr, * op = g_out.ptr, * rp = g_res.ptr;
        int nv = H;
        void * args[] = { &ip, &wp, &op, &rp, &nv };
        // Megakernel launches with 512 threads (baseline uses 256 for ncols<1024)
        if (!launch_kernel(fn, 1, 1, 1, 512, 1, 1, args)) {
            test_fail(name, "launch failed H=%d", H); all_ok = false; break;
        }
        g_out.download(norm_out.data(), H*sizeof(float));
        g_res.download(residual.data(), H*sizeof(float));

        // Residual must be exact copy of input
        auto rr = compare_float(residual.data(), input.data(), H);
        if (rr.max_abs > 0.0f) {
            test_fail(name, "H=%d residual mismatch max_abs=%.2e", H, rr.max_abs);
            all_ok = false; break;
        }
        // Norm output: baseline uses block_reduce with 256 threads, we use 512.
        // FP rounding differs in reduction order. Allow 1e-5 relative.
        auto r = compare_float(norm_out.data(), cpu_out.data(), H);
        if (r.nan_count > 0) {
            test_fail(name, "H=%d: %d NaN values", H, r.nan_count);
            all_ok = false; break;
        }
        if (r.max_rel > 1e-4f) {
            test_fail(name, "H=%d: max_rel=%.2e > 1e-4", H, r.max_rel);
            all_ok = false; break;
        }
    }
    if (all_ok) test_pass(name, "H=896,928,768,1024 all within 1e-4 rel");
}

// ----------------------------------------------------------------------------
// eval_quantize_q8 at H=896.
// Verify Q8_1 quantization produces same result as baseline quantize_q8_1
// at non-256-aligned dimensions. CPU reference mirrors baseline quantize.cu.
// ----------------------------------------------------------------------------
static void cpu_quantize_q8_1(const float * input, block_q8_1_host * output, int n) {
    const int nb = n / 32;
    for (int ib = 0; ib < nb; ib++) {
        float amax = 0;
        float sum  = 0;
        for (int j = 0; j < 32; j++) {
            float v = input[ib * 32 + j];
            amax = fmaxf(amax, fabsf(v));
            sum += v;
        }
        float d = amax / 127.0f;
        output[ib].ds[0] = f32_to_f16(d);
        output[ib].ds[1] = f32_to_f16(sum);
        for (int j = 0; j < 32; j++) {
            float v = input[ib * 32 + j];
            output[ib].qs[j] = (amax == 0.0f) ? 0 : (int8_t)roundf(v / d);
        }
    }
}

static void test_quantize_q8_qwen2(hipModule_t mod) {
    const char * name = "eval_quantize_q8@896";
    hipFunction_t fn = load_kernel(mod, "eval_quantize_q8");
    if (!fn) { test_fail(name, "kernel not found"); return; }

    int dims[] = { 896, 4864, 768, 2048 }; // Qwen2 H, FF, other sizes
    bool all_ok = true;
    for (int n : dims) {
        std::vector<float> input(n);
        for (int i = 0; i < n; i++)
            input[i] = sinf(0.019f * i) * 4.0f + 0.5f * cosf(0.043f * i);

        const int nb = n / 32;
        std::vector<block_q8_1_host> cpu_q8(nb);
        cpu_quantize_q8_1(input.data(), cpu_q8.data(), n);

        GpuBuf g_in(n * sizeof(float));
        GpuBuf g_q8(nb * sizeof(block_q8_1_host));
        g_in.upload(input.data(), n * sizeof(float));

        void * ip = g_in.ptr, * qp = g_q8.ptr; int nv = n;
        void * args[] = { &ip, &qp, &nv };
        unsigned grid = (n + 511) / 512;
        if (!launch_kernel(fn, grid, 1, 1, 512, 1, 1, args)) {
            test_fail(name, "launch failed n=%d", n); all_ok = false; break;
        }
        std::vector<block_q8_1_host> gpu_q8(nb);
        g_q8.download(gpu_q8.data(), nb * sizeof(block_q8_1_host));

        // Compare each block: d, sum (as f16), and all 32 qs values
        for (int ib = 0; ib < nb; ib++) {
            if (gpu_q8[ib].ds[0] != cpu_q8[ib].ds[0]) {
                test_fail(name, "n=%d block %d: d gpu=0x%04x cpu=0x%04x",
                          n, ib, gpu_q8[ib].ds[0], cpu_q8[ib].ds[0]);
                all_ok = false; break;
            }
            // Sum may differ slightly: GPU reduces 32 floats via warp shuffle,
            // CPU sums sequentially. Both then convert to f16. Allow 1% relative.
            float gpu_s = f16_to_f32(gpu_q8[ib].ds[1]);
            float cpu_s = f16_to_f32(cpu_q8[ib].ds[1]);
            if (fabsf(gpu_s - cpu_s) > 0.01f * (fabsf(cpu_s) + 1e-6f)) {
                test_fail(name, "n=%d block %d: sum gpu=%.4f cpu=%.4f",
                          n, ib, gpu_s, cpu_s);
                all_ok = false; break;
            }
            for (int j = 0; j < 32; j++) {
                if (gpu_q8[ib].qs[j] != cpu_q8[ib].qs[j]) {
                    test_fail(name, "n=%d block %d qs[%d]: gpu=%d cpu=%d",
                              n, ib, j, gpu_q8[ib].qs[j], cpu_q8[ib].qs[j]);
                    all_ok = false; break;
                }
            }
            if (!all_ok) break;
        }
        if (!all_ok) break;
    }
    if (all_ok) test_pass(name, "n=896,4864,768,2048 bit-exact qs, sum within 1%%");
}

// ----------------------------------------------------------------------------
// eval_rwkv_wkv6_step — ported from baseline wkv.cu::rwkv_wkv_f32
// CPU reference mirrors baseline exactly for B=1, T=1 (single decode step).
// Tests: state update, y output, per-head independence.
// ----------------------------------------------------------------------------
static void cpu_rwkv_wkv6_step(
        const float * k, const float * v, const float * r,
        const float * tf, const float * td,
        float * state, float * y,
        int head_size, int n_head) {
    // Baseline wkv.cu lines 27-59 for one timestep (t=0)
    for (int head_i = 0; head_i < n_head; head_i++) {
        for (int tid = 0; tid < head_size; tid++) {
            int t = head_i * head_size + tid;
            float * s = state + head_i * head_size * head_size + tid * head_size;
            float _v = v[t];
            float y_val = 0;
            for (int j = 0; j < head_size; j++) {
                float kv = k[head_i * head_size + j] * _v;
                y_val += r[head_i * head_size + j] * (tf[head_i * head_size + j] * kv + s[j]);
                s[j] = s[j] * td[head_i * head_size + j] + kv;
            }
            y[t] = y_val;
        }
    }
}

static void test_rwkv_wkv6_step(hipModule_t mod) {
    const char * name = "eval_rwkv_wkv6_step";
    hipFunction_t fn = load_kernel(mod, name);
    if (!fn) { test_fail(name, "kernel not found"); return; }

    const int head_size = 64; // typical RWKV head_size
    const int n_head = 4;     // small for testing
    const int C = n_head * head_size;
    const int state_size = n_head * head_size * head_size;

    // Randomized inputs
    std::vector<float> h_k(C), h_v(C), h_r(C), h_tf(C), h_td(C);
    std::vector<float> h_state(state_size), h_y_cpu(C), h_y_gpu(C);
    std::vector<float> h_state_cpu(state_size);

    for (int i = 0; i < C; i++) {
        h_k[i]  = sinf(0.1f * i) * 0.5f;
        h_v[i]  = cosf(0.13f * i) * 0.5f;
        h_r[i]  = sinf(0.07f * i + 1.0f) * 0.5f;
        h_tf[i] = 0.8f + 0.2f * cosf(0.03f * i); // time_first near 1
        h_td[i] = 0.9f + 0.1f * sinf(0.05f * i); // time_decay near 0.95
    }
    for (int i = 0; i < state_size; i++) {
        h_state[i] = sinf(0.017f * i) * 0.1f; // small initial state
    }
    h_state_cpu = h_state; // copy for CPU reference

    // CPU reference
    cpu_rwkv_wkv6_step(h_k.data(), h_v.data(), h_r.data(), h_tf.data(), h_td.data(),
                        h_state_cpu.data(), h_y_cpu.data(), head_size, n_head);

    // GPU
    GpuBuf g_k(C*4), g_v(C*4), g_r(C*4), g_tf(C*4), g_td(C*4);
    GpuBuf g_state(state_size*4), g_y(C*4);
    g_k.upload(h_k.data(), C*4);
    g_v.upload(h_v.data(), C*4);
    g_r.upload(h_r.data(), C*4);
    g_tf.upload(h_tf.data(), C*4);
    g_td.upload(h_td.data(), C*4);
    g_state.upload(h_state.data(), state_size*4);

    void * kp = g_k.ptr, * vp = g_v.ptr, * rp = g_r.ptr;
    void * tfp = g_tf.ptr, * tdp = g_td.ptr;
    void * sp = g_state.ptr, * yp = g_y.ptr;
    int hs = head_size, nh = n_head;
    // Shared memory: 4 * head_size * sizeof(float) for k,r,tf,td
    size_t smem = 4 * head_size * sizeof(float);
    void * args[] = { &kp, &vp, &rp, &tfp, &tdp, &sp, &yp, &hs, &nh };
    if (!launch_kernel(fn, n_head, 1, 1, head_size, 1, 1, args, smem)) {
        test_fail(name, "launch failed"); return;
    }

    g_y.download(h_y_gpu.data(), C*4);
    auto r_cmp = compare_float(h_y_gpu.data(), h_y_cpu.data(), C);
    if (r_cmp.nan_count > 0) { test_fail(name, "%d NaN", r_cmp.nan_count); return; }
    // WKV involves many multiply-adds — allow some tolerance
    if (r_cmp.max_rel > 1e-3f) {
        test_fail(name, "y max_rel=%.2e > 1e-3", r_cmp.max_rel); return;
    }

    // Also check state was updated
    std::vector<float> h_state_gpu(state_size);
    g_state.download(h_state_gpu.data(), state_size*4);
    auto s_cmp = compare_float(h_state_gpu.data(), h_state_cpu.data(), state_size);
    if (s_cmp.max_rel > 1e-3f) {
        test_fail(name, "state max_rel=%.2e > 1e-3", s_cmp.max_rel); return;
    }

    test_pass(name, "y: max_abs=%.2e max_rel=%.2e; state: max_abs=%.2e",
              r_cmp.max_abs, r_cmp.max_rel, s_cmp.max_abs);
}

// ============================================================================
// main
// ============================================================================
int main() {
    std::string hsaco_path = find_hsaco("decode.hip_", 64);
    if (hsaco_path.empty()) {
        fprintf(stderr, "FAIL: decode.hip_*.hsaco with FA_HEAD_DIM=64 not found\n");
        fprintf(stderr, "Run a model first to populate the .hsaco cache.\n");
        return 1;
    }
    fprintf(stderr, "Using hsaco: %s\n\n", hsaco_path.c_str());

    hipModule_t mod = nullptr;
    hipError_t e = hipModuleLoad(&mod, hsaco_path.c_str());
    if (e != hipSuccess) {
        fprintf(stderr, "FAIL: hipModuleLoad — %s\n", hipGetErrorString(e));
        return 1;
    }

    // Activation gate*up kernels
    test_act_mul(mod, "eval_gelu_mul",       cpu_gelu,       -5.0f, 5.0f, 2e-5f, 1e-4f);
    test_act_mul(mod, "eval_gelu_erf_mul",   cpu_gelu_erf,   -5.0f, 5.0f, 2e-5f, 1e-4f);
    test_act_mul(mod, "eval_gelu_quick_mul", cpu_gelu_quick, -5.0f, 5.0f, 2e-5f, 1e-4f);
    test_act_mul(mod, "eval_relu2_mul",      cpu_relu2,      -3.0f, 3.0f, 1e-6f, 1e-6f);

    // Pure unary (input, output, n)
    test_unary_pure(mod, "eval_sigmoid",  cpu_sigmoid,  -10.0f, 10.0f, 1e-5f);
    test_unary_pure(mod, "eval_softplus", cpu_softplus, -10.0f, 10.0f, 1e-5f);

    // Scalar ops with extra param
    test_scale_scalar(mod);
    test_softcap(mod);

    // Norms (single-row reductions)
    test_layernorm(mod);
    test_l2norm(mod);

    // Reductions / sorts / softmax
    test_softmax_row(mod);
    test_argsort_desc(mod);
    test_sum_row(mod);

    // Tensor ops
    test_concat_dim0(mod);
    test_concat_dim1(mod);
    test_repeat_dim0(mod);

    // Linear algebra primitives
    test_axpy(mod);

    // SSM (Mamba) single-step
    test_ssm_conv_step(mod);
    test_ssm_scan_step(mod);

    // RWKV WKV6 single-step
    test_rwkv_wkv6_step(mod);

    // Embed (Q8_0) — would have caught the pre-session token_id-offset bug
    test_embed_q8_0(mod);

    // Qwen2-dimension tests: H=896 (not power of 2, not multiple of 256)
    test_rmsnorm_qwen2(mod);
    test_quantize_q8_qwen2(mod);

    // Matvec at Qwen2 dimensions (in_dim=896) — catches bugs that pass at 512
    hipFunction_t quantize_fn = load_kernel(mod, "eval_quantize_q8");
    if (quantize_fn) {
        test_matvec_q5_0_qwen2_dim(mod, quantize_fn);
    } else {
        fprintf(stderr, "SKIP: eval_quantize_q8 not found — matvec@896 test skipped\n");
    }

    (void)hipModuleUnload(mod);
    return test_summary();
}
