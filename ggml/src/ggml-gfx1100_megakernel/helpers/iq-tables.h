// iq-tables.h — IQ quantization lookup tables for device code
// Includes tables from ggml-common.h via GGML_TABLE_BEGIN mechanism.
// Tables: iq2xxs_grid, iq2xs_grid, iq2s_grid, iq3xxs_grid, iq3s_grid,
//         iq1s_grid_gpu, ksigns_iq2xs, kmask_iq2xs, kvalues_iq4nl, kvalues_mxfp4
#pragma once

// Skip DECL section (we have our own block structs in quant-types.h)
// Only emit IMPL section (lookup tables as static const __device__ arrays)
#ifndef GGML_COMMON_IMPL
#define GGML_COMMON_DECL          // prevent block struct redefinitions
#define GGML_COMMON_IMPL_CUDA     // select CUDA/HIP table emission path
#include "../../ggml-common.h"
#undef GGML_COMMON_DECL
#endif
