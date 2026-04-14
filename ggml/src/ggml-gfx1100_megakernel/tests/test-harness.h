// test-harness.h — Shared utilities for gfx1100 megakernel unit tests
//
// Provides: HIP error checking, .hsaco discovery, kernel loading, comparison.
// All tests are SYNTHETIC — no model loading required.

#pragma once

#include "ggml.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>
#include <filesystem>
#include <functional>

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
// Find eval/prompt .hsaco in ~/.cache/gfx1100-megakernel/
// If head_dim > 0, only return an .hsaco whose model_config_*.h has the matching FA_HEAD_DIM.
// This prevents tests from accidentally using a .hsaco compiled for a different model.
// ---------------------------------------------------------------------------
static std::string find_hsaco(const char * prefix = "decode.hip_", int head_dim = 0) {
    const char * home = getenv("USERPROFILE");
    if (!home) home = getenv("HOME");
    if (!home) {
        fprintf(stderr, "Cannot determine home directory\n");
        return "";
    }
    fs::path cache_dir = fs::path(home) / ".cache" / "gfx1100-megakernel";
    if (!fs::is_directory(cache_dir)) {
        fprintf(stderr, "Cache directory not found: %s\n", cache_dir.string().c_str());
        return "";
    }
    // Collect all candidates, then pick newest. Avoids picking a stale .hsaco
    // when several were compiled with different (arch, caps) hashes for the
    // same head_dim — the most recent one carries the latest kernel set.
    std::string best_match;
    fs::file_time_type best_match_mtime{};
    std::string best_fallback;
    fs::file_time_type best_fallback_mtime{};

    for (const auto & entry : fs::directory_iterator(cache_dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.size() <= 6) continue;
        if (name.substr(name.size() - 6) != ".hsaco") continue;
        if (name.rfind(prefix, 0) != 0) continue;

        auto mtime = fs::last_write_time(entry);

        if (head_dim <= 0) {
            if (best_fallback.empty() || mtime > best_fallback_mtime) {
                best_fallback = entry.path().string();
                best_fallback_mtime = mtime;
            }
            continue;
        }

        // Check matching FA_HEAD_DIM in companion model_config_*.h
        std::string hash = name.substr(strlen(prefix), name.size() - strlen(prefix) - 6);
        fs::path config_path = cache_dir / ("model_config_" + hash + ".h");
        bool dim_match = false;
        if (fs::exists(config_path)) {
            FILE * f = fopen(config_path.string().c_str(), "r");
            if (f) {
                char line[256];
                while (fgets(line, sizeof(line), f)) {
                    int dim_val = 0;
                    if (sscanf(line, "#define FA_HEAD_DIM %d", &dim_val) == 1) {
                        if (dim_val == head_dim) { dim_match = true; break; }
                    }
                }
                fclose(f);
            }
        }
        if (dim_match) {
            if (best_match.empty() || mtime > best_match_mtime) {
                best_match = entry.path().string();
                best_match_mtime = mtime;
            }
        } else {
            if (best_fallback.empty() || mtime > best_fallback_mtime) {
                best_fallback = entry.path().string();
                best_fallback_mtime = mtime;
            }
        }
    }
    return best_match.empty() ? best_fallback : best_match;
}

// ---------------------------------------------------------------------------
// Load HIP module + get kernel function
// ---------------------------------------------------------------------------
static hipFunction_t load_kernel(hipModule_t mod, const char * name) {
    hipFunction_t fn = nullptr;
    hipError_t e = hipModuleGetFunction(&fn, mod, name);
    if (e != hipSuccess) {
        fprintf(stderr, "WARN: hipModuleGetFunction(%s) — %s\n", name, hipGetErrorString(e));
        return nullptr;
    }
    return fn;
}

// ---------------------------------------------------------------------------
// Q8_1 block layout (matches ggml-common.h block_q8_1)
// ---------------------------------------------------------------------------
#define QK8_1 32

struct block_q8_1_host {
    uint16_t ds[2];      // ds[0] = f16(d), ds[1] = f16(sum)
    int8_t   qs[QK8_1];
};
static_assert(sizeof(block_q8_1_host) == 36, "block_q8_1_host must be 36 bytes");

// ---------------------------------------------------------------------------
// CPU reference: f16 <-> f32 via ggml
// ---------------------------------------------------------------------------
static inline float f16_to_f32(uint16_t h) {
    return ggml_fp16_to_fp32((ggml_fp16_t)h);
}
static inline uint16_t f32_to_f16(float f) {
    return (uint16_t)ggml_fp32_to_fp16(f);
}

// ---------------------------------------------------------------------------
// CPU reference: bf16 <-> f32
// ---------------------------------------------------------------------------
static inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}
static inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

// ---------------------------------------------------------------------------
// GPU buffer RAII helper
// ---------------------------------------------------------------------------
struct GpuBuf {
    void * ptr = nullptr;
    size_t size = 0;

    GpuBuf() = default;
    explicit GpuBuf(size_t n) : size(n) {
        HIP_CHECK(hipMalloc(&ptr, n));
        HIP_CHECK(hipMemset(ptr, 0, n));
    }
    ~GpuBuf() { if (ptr) (void)hipFree(ptr); }

    GpuBuf(const GpuBuf &) = delete;
    GpuBuf & operator=(const GpuBuf &) = delete;
    GpuBuf(GpuBuf && o) noexcept : ptr(o.ptr), size(o.size) { o.ptr = nullptr; }

    void upload(const void * data, size_t n) {
        HIP_CHECK(hipMemcpy(ptr, data, n, hipMemcpyHostToDevice));
    }
    void download(void * data, size_t n) const {
        HIP_CHECK(hipMemcpy(data, ptr, n, hipMemcpyDeviceToHost));
    }

    template<typename T> T * as() { return (T *)ptr; }
    template<typename T> const T * as() const { return (const T *)ptr; }
};

// ---------------------------------------------------------------------------
// Test result tracking
// ---------------------------------------------------------------------------
struct TestResult {
    std::string name;
    bool        pass;
    std::string detail;
};

static std::vector<TestResult> g_results;

static void test_pass(const char * name, const char * fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_results.push_back({name, true, buf});
    fprintf(stderr, "  PASS: %s — %s\n", name, buf);
}

static void test_fail(const char * name, const char * fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_results.push_back({name, false, buf});
    fprintf(stderr, "  FAIL: %s — %s\n", name, buf);
}

static int test_summary() {
    int pass = 0, fail = 0;
    for (const auto & r : g_results) {
        if (r.pass) pass++; else fail++;
    }
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "  %d passed, %d failed, %d total\n", pass, fail, pass + fail);
    fprintf(stderr, "========================================\n");

    if (fail > 0) {
        fprintf(stderr, "\nFailed tests:\n");
        for (const auto & r : g_results) {
            if (!r.pass) fprintf(stderr, "  FAIL: %s — %s\n", r.name.c_str(), r.detail.c_str());
        }
    }
    return fail > 0 ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Compare float arrays: returns max_abs_diff, max_rel_diff, nan_count
// ---------------------------------------------------------------------------
struct CompareResult {
    float max_abs  = 0.0f;
    float max_rel  = 0.0f;
    int   nan_count = 0;
    int   zero_count = 0;
    int   n = 0;
};

static CompareResult compare_float(const float * gpu, const float * cpu, int n) {
    CompareResult r;
    r.n = n;
    for (int i = 0; i < n; i++) {
        if (std::isnan(gpu[i])) { r.nan_count++; continue; }
        if (gpu[i] == 0.0f && cpu[i] == 0.0f) { r.zero_count++; continue; }
        float diff = fabsf(gpu[i] - cpu[i]);
        if (diff > r.max_abs) r.max_abs = diff;
        float cpu_abs = fabsf(cpu[i]);
        if (cpu_abs > 1e-6f) {
            float rel = diff / cpu_abs;
            if (rel > r.max_rel) r.max_rel = rel;
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// Launch kernel helper (reduces boilerplate)
// ---------------------------------------------------------------------------
static bool launch_kernel(hipFunction_t fn, unsigned gx, unsigned gy, unsigned gz,
                          unsigned bx, unsigned by, unsigned bz,
                          void ** args, unsigned shared_mem = 0) {
    hipError_t e = hipModuleLaunchKernel(fn, gx, gy, gz, bx, by, bz,
                                          shared_mem, nullptr, args, nullptr);
    if (e != hipSuccess) {
        fprintf(stderr, "hipModuleLaunchKernel failed: %s\n", hipGetErrorString(e));
        return false;
    }
    HIP_CHECK(hipDeviceSynchronize());
    return true;
}
