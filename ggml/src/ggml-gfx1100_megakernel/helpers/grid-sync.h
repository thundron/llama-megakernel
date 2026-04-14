// grid-sync.h — Grid-level barrier for persistent kernel
// Adapted from luce-megakernel kernel.cu lines 115-139 for HIP/gfx1100
#pragma once

#include "hip-shim.h"

struct GridSync {
    unsigned int * counter;
    unsigned int * generation;
    unsigned int   nblocks;
    unsigned int   local_gen;

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
                volatile unsigned int * vgen = (volatile unsigned int *)generation;
                while (*vgen <= my_gen) {}
            }
            local_gen = my_gen + 1;
        }
        __syncthreads();
    }
};
