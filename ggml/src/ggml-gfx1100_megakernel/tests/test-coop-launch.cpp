// Minimal test: does hipModuleLaunchCooperativeKernel work on gfx1100?
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cassert>

int main() {
    hipModule_t mod;
    hipFunction_t fn;
    hipError_t e;

    // Try multiple paths
    const char * paths[] = {
        "C:\\Users\\thund\\.cache\\gfx1100-megakernel\\test_coop.hsaco",
        "/tmp/test_coop.hsaco",
        nullptr
    };
    for (int p = 0; paths[p]; p++) {
        e = hipModuleLoad(&mod, paths[p]);
        if (e == hipSuccess) { fprintf(stderr, "Loaded: %s\n", paths[p]); break; }
    }
    if (e != hipSuccess) { fprintf(stderr, "ModuleLoad: %s\n", hipGetErrorString(e)); return 1; }

    e = hipModuleGetFunction(&fn, mod, "test_coop");
    if (e != hipSuccess) { fprintf(stderr, "GetFunction: %s\n", hipGetErrorString(e)); return 1; }

    // Query max blocks
    int blocks_per_sm = 0;
    e = hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, fn, 256, 0);
    fprintf(stderr, "Occupancy: %d blocks per SM (err=%s)\n", blocks_per_sm, hipGetErrorString(e));

    int n_sm = 0;
    hipDeviceGetAttribute(&n_sm, hipDeviceAttributeMultiprocessorCount, 0);
    fprintf(stderr, "Device: %d SMs\n", n_sm);

    // Allocate output
    float * d_out;
    hipMalloc(&d_out, 3 * sizeof(float));
    hipMemset(d_out, 0, 3 * sizeof(float));

    void * args[] = { (void *)&d_out };

    // Try cooperative launch with 2 blocks
    fprintf(stderr, "Launching cooperative kernel with 2 blocks, 256 threads...\n");
    e = hipModuleLaunchCooperativeKernel(fn, 2, 1, 1, 256, 1, 1, 0, 0, args);
    fprintf(stderr, "Launch result: %s\n", hipGetErrorString(e));

    if (e == hipSuccess) {
        hipDeviceSynchronize();
        float h[3] = {};
        hipMemcpy(h, d_out, 3 * sizeof(float), hipMemcpyDeviceToHost);
        fprintf(stderr, "Results: [%.1f, %.1f, %.1f] (expected [1.0, 2.0, 3.0])\n", h[0], h[1], h[2]);
        if (h[0] == 1.0f && h[1] == 2.0f && h[2] == 3.0f) {
            fprintf(stderr, "PASS: cooperative groups work on gfx1100\n");
        } else {
            fprintf(stderr, "FAIL: wrong results\n");
        }
    }

    hipFree(d_out);
    hipModuleUnload(mod);
    return (e == hipSuccess) ? 0 : 1;
}
