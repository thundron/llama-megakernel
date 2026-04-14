// test-grid-sync.hip — Measure grid sync cost on gfx1100
//
// Tests three approaches:
//   1. hipLaunchCooperativeKernel with cooperative_groups::grid_group::sync()
//   2. Hand-rolled atomic barrier (Luce-style GridSync)
//   3. Two separate kernel launches (baseline comparison)
//
// Each approach does: Phase A (write pattern) → sync → Phase B (read + verify)
// Repeated 1000 times inside the kernel to amortize launch overhead.
//
// Build: hipcc --offload-arch=gfx1100 -O3 -std=c++17 test-grid-sync.hip -o test-grid-sync
// Run:   ./test-grid-sync

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <chrono>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "HIP error %d at %s:%d: %s\n", e, __FILE__, __LINE__, hipGetErrorString(e)); exit(1); } } while(0)

// ============================================================================
// Approach 1: cooperative_groups grid sync
// ============================================================================

__global__ void kernel_coop_grid_sync(float * __restrict__ buf, int N, int n_iters) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int iter = 0; iter < n_iters; iter++) {
        // Phase A: write
        for (int i = tid; i < N; i += stride) {
            buf[i] = (float)(iter * N + i);
        }

        grid.sync();

        // Phase B: read from a different offset (cross-block dependency)
        for (int i = tid; i < N; i += stride) {
            int j = (i + N/2) % N;
            buf[i] += buf[j] * 0.001f;
        }

        grid.sync();
    }
}

// ============================================================================
// Approach 2: atomic barrier (Luce-style)
// ============================================================================

struct AtomicGridSync {
    unsigned int *counter;
    unsigned int *generation;
    unsigned int nblocks;
    unsigned int local_gen;

    __device__ void sync() {
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int my_gen = local_gen;
            __threadfence();
            unsigned int arrived = atomicAdd(counter, 1);
            if (arrived == nblocks - 1) {
                *counter = 0;
                __threadfence();
                atomicAdd(generation, 1);
            } else {
                volatile unsigned int *vgen = (volatile unsigned int *)generation;
                while (*vgen <= my_gen) { }
            }
            local_gen = my_gen + 1;
        }
        __syncthreads();
    }
};

__global__ void kernel_atomic_grid_sync(
        float * __restrict__ buf, int N, int n_iters,
        unsigned int * __restrict__ barrier_counter,
        unsigned int * __restrict__ barrier_generation) {

    AtomicGridSync grid;
    grid.counter = barrier_counter;
    grid.generation = barrier_generation;
    grid.nblocks = gridDim.x;
    grid.local_gen = 0;

    // Initialize barrier
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *barrier_counter = 0;
        *barrier_generation = 0;
    }
    grid.sync();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int iter = 0; iter < n_iters; iter++) {
        // Phase A: write
        for (int i = tid; i < N; i += stride) {
            buf[i] = (float)(iter * N + i);
        }

        grid.sync();

        // Phase B: read cross-block
        for (int i = tid; i < N; i += stride) {
            int j = (i + N/2) % N;
            buf[i] += buf[j] * 0.001f;
        }

        grid.sync();
    }
}

// ============================================================================
// Approach 3: two separate kernels (baseline)
// ============================================================================

__global__ void kernel_phase_a(float * __restrict__ buf, int N, int iter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        buf[i] = (float)(iter * N + i);
    }
}

__global__ void kernel_phase_b(float * __restrict__ buf, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        int j = (i + N/2) % N;
        buf[i] += buf[j] * 0.001f;
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    const int N = 8192;           // buffer size (simulates hidden state)
    const int n_iters = 1000;     // iterations inside kernel
    const int block_size = 256;

    // Query device for cooperative launch support
    int dev = 0;
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, dev));
    printf("Device: %s\n", props.name);
    printf("CUs: %d, Max threads/block: %d\n", props.multiProcessorCount, props.maxThreadsPerBlock);

    int coop_supported = 0;
    HIP_CHECK(hipDeviceGetAttribute(&coop_supported, hipDeviceAttributeCooperativeLaunch, dev));
    printf("Cooperative launch supported: %s\n", coop_supported ? "YES" : "NO");

    // Determine max blocks for cooperative launch
    int max_blocks_coop = 0;
    if (coop_supported) {
        HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_coop, kernel_coop_grid_sync, block_size, 0));
        max_blocks_coop *= props.multiProcessorCount;
        printf("Max cooperative blocks: %d (per SM: %d, SMs: %d)\n",
               max_blocks_coop, max_blocks_coop / props.multiProcessorCount,
               props.multiProcessorCount);
    }

    // Allocate
    float *d_buf;
    unsigned int *d_counter, *d_generation;
    HIP_CHECK(hipMalloc(&d_buf, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_counter, sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_generation, sizeof(unsigned int)));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Test with multiple block counts
    int block_counts[] = {48, 96, 192};
    int n_configs = 3;

    for (int ci = 0; ci < n_configs; ci++) {
        int n_blocks = block_counts[ci];
        printf("\n====== %d blocks x %d threads = %d total threads ======\n",
               n_blocks, block_size, n_blocks * block_size);

    HIP_CHECK(hipMemset(d_buf, 0, N * sizeof(float)));
    HIP_CHECK(hipMemset(d_counter, 0, sizeof(unsigned int)));
    HIP_CHECK(hipMemset(d_generation, 0, sizeof(unsigned int)));

    // Warmup
    kernel_phase_a<<<n_blocks, block_size, 0, stream>>>(d_buf, N, 0);
    kernel_phase_b<<<n_blocks, block_size, 0, stream>>>(d_buf, N);
    HIP_CHECK(hipStreamSynchronize(stream));

    // ---- Test 1: Cooperative groups grid sync ----
    if (coop_supported) {
        HIP_CHECK(hipMemset(d_buf, 0, N * sizeof(float)));
        HIP_CHECK(hipStreamSynchronize(stream));

        void *args[] = { &d_buf, (void*)&N, (void*)&n_iters };

        auto t0 = std::chrono::high_resolution_clock::now();
        HIP_CHECK(hipLaunchCooperativeKernel(
            (void*)kernel_coop_grid_sync,
            dim3(n_blocks), dim3(block_size), args, 0, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double us_per_sync = ms * 1000.0 / (n_iters * 2);  // 2 syncs per iter
        printf("1. Cooperative grid sync:  %.2f ms total, %.3f us/sync (%d syncs)\n",
               ms, us_per_sync, n_iters * 2);
    } else {
        printf("1. Cooperative grid sync:  SKIPPED (not supported)\n");
    }

    // ---- Test 2: Atomic barrier ----
    {
        HIP_CHECK(hipMemset(d_buf, 0, N * sizeof(float)));
        HIP_CHECK(hipMemset(d_counter, 0, sizeof(unsigned int)));
        HIP_CHECK(hipMemset(d_generation, 0, sizeof(unsigned int)));
        HIP_CHECK(hipStreamSynchronize(stream));

        auto t0 = std::chrono::high_resolution_clock::now();
        kernel_atomic_grid_sync<<<n_blocks, block_size, 0, stream>>>(
            d_buf, N, n_iters, d_counter, d_generation);
        HIP_CHECK(hipStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double us_per_sync = ms * 1000.0 / (n_iters * 2);
        printf("2. Atomic barrier sync:   %.2f ms total, %.3f us/sync (%d syncs)\n",
               ms, us_per_sync, n_iters * 2);
    }

    // ---- Test 3: Two separate kernel launches ----
    {
        HIP_CHECK(hipMemset(d_buf, 0, N * sizeof(float)));
        HIP_CHECK(hipStreamSynchronize(stream));

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < n_iters; iter++) {
            kernel_phase_a<<<n_blocks, block_size, 0, stream>>>(d_buf, N, iter);
            kernel_phase_b<<<n_blocks, block_size, 0, stream>>>(d_buf, N);
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double us_per_launch = ms * 1000.0 / (n_iters * 2);
        printf("3. Separate launches:     %.2f ms total, %.3f us/launch (%d launches)\n",
               ms, us_per_launch, n_iters * 2);
    }

    printf("\nConclusion: if grid sync < separate launches, persistent kernel is viable.\n");

    } // end block_counts loop

    (void)hipFree(d_buf);
    (void)hipFree(d_counter);
    (void)hipFree(d_generation);
    (void)hipStreamDestroy(stream);
    return 0;
}
