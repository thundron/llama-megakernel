#pragma once
// rocBLAS GEMM wrapper — dequantizes quantized weights to F16, then calls hipblasGemmEx
// Pattern from baseline ggml-cuda.cu ggml_cuda_op_mul_mat_cublas (lines 1457-1515)
// Macro mappings from baseline vendors/hip.h
//
// This is a HOST-SIDE C++ header (not a device kernel).
// gfx1100 (RDNA3) uses the F16-input, F32-compute, F32-output GEMM path.
//
// Architecture note: this megakernel backend JIT-compiles .hip files with hipcc,
// loads them as hipModule_t, and launches kernels via hipModuleLaunchKernel.
// Device kernels (F32->F16 conversion, weight dequant) must live in the JIT source.
// This header provides only the HOST-SIDE hipblasGemmEx wrapper + temp buffer mgmt.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ggml type enum values — from ggml/include/ggml.h:390-420
// We only need the integer values for the switch table.
#define ROCBLAS_GEMM_TYPE_F32     0
#define ROCBLAS_GEMM_TYPE_F16     1
#define ROCBLAS_GEMM_TYPE_Q4_0    2
#define ROCBLAS_GEMM_TYPE_Q4_1    3
#define ROCBLAS_GEMM_TYPE_Q5_0    6
#define ROCBLAS_GEMM_TYPE_Q5_1    7
#define ROCBLAS_GEMM_TYPE_Q8_0    8
#define ROCBLAS_GEMM_TYPE_Q2_K   10
#define ROCBLAS_GEMM_TYPE_Q3_K   11
#define ROCBLAS_GEMM_TYPE_Q4_K   12
#define ROCBLAS_GEMM_TYPE_Q5_K   13
#define ROCBLAS_GEMM_TYPE_Q6_K   14
#define ROCBLAS_GEMM_TYPE_IQ2_XXS 16
#define ROCBLAS_GEMM_TYPE_IQ2_XS  17
#define ROCBLAS_GEMM_TYPE_IQ3_XXS 18
#define ROCBLAS_GEMM_TYPE_IQ1_S   19
#define ROCBLAS_GEMM_TYPE_IQ4_NL  20
#define ROCBLAS_GEMM_TYPE_IQ3_S   21
#define ROCBLAS_GEMM_TYPE_IQ2_S   22
#define ROCBLAS_GEMM_TYPE_IQ4_XS  23
#define ROCBLAS_GEMM_TYPE_IQ1_M   29
#define ROCBLAS_GEMM_TYPE_BF16    30

// ============================================================================
// rocblas_gemm_state — persistent handle + temp buffers
// Pattern: baseline ggml_backend_cuda_context owns cublas_handle + pool allocators
// We simplify to fixed-size temp buffers (no pool, megakernel knows max dims).
// ============================================================================

struct rocblas_gemm_state {
    hipblasHandle_t handle;
    void * weight_f16_temp;   // [max_weight_elements * sizeof(half)]
    void * input_f16_temp;    // [max_input_elements * sizeof(half)]
    size_t weight_temp_bytes;
    size_t input_temp_bytes;
    bool initialized;
};

// ============================================================================
// Device kernel source for JIT compilation (to be included in the .hip file)
//
// The following kernel must be compiled into the .hsaco module alongside the
// other megakernel device code. It provides F32→F16 conversion for the
// activation input to hipblasGemmEx. Add this to your decode.hip source:
//
//   extern "C" __global__
//   void gemm_f32_to_f16(const float * __restrict__ x, __half * __restrict__ y, int64_t k) {
//       const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
//       if (i < k) {
//           y[i] = __float2half(x[i]);
//       }
//   }
//
// Weight dequant-to-F16 kernels should follow the same pattern as the existing
// embed-dequant.h functions but outputting __half instead of float.
// See baseline convert.cu ggml_get_to_fp16_cuda() (lines 712-766) for the
// full type dispatch table.
// ============================================================================

// ============================================================================
// rocblas_gemm_init — create hipBLAS handle, allocate temp buffers
// Pattern: baseline ggml_backend_cuda_context constructor (cublas handle creation)
// ============================================================================

static inline int rocblas_gemm_init(rocblas_gemm_state * state, int max_in_dim, int max_out_dim, int max_batch) {
    memset(state, 0, sizeof(*state));

    hipblasStatus_t st = hipblasCreate(&state->handle);
    if (st != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "rocblas-gemm: hipblasCreate failed: %d\n", (int)st);
        return -1;
    }

    // Weight temp: max_out_dim * max_in_dim elements of half
    // Pattern: baseline lines 1459-1465 (src0_as_f16.alloc(row_diff*ne00))
    state->weight_temp_bytes = (size_t)max_out_dim * max_in_dim * sizeof(half);
    hipError_t err = hipMalloc(&state->weight_f16_temp, state->weight_temp_bytes);
    if (err != hipSuccess) {
        fprintf(stderr, "rocblas-gemm: hipMalloc weight_f16_temp (%zu bytes) failed: %s\n",
                state->weight_temp_bytes, hipGetErrorString(err));
        hipblasDestroy(state->handle);
        return -1;
    }

    // Input temp: max_batch * max_in_dim elements of half
    // Pattern: baseline lines 1469-1475 (src1_as_f16.alloc(src1_ncols*ne10))
    state->input_temp_bytes = (size_t)max_batch * max_in_dim * sizeof(half);
    err = hipMalloc(&state->input_f16_temp, state->input_temp_bytes);
    if (err != hipSuccess) {
        fprintf(stderr, "rocblas-gemm: hipMalloc input_f16_temp (%zu bytes) failed: %s\n",
                state->input_temp_bytes, hipGetErrorString(err));
        hipFree(state->weight_f16_temp);
        hipblasDestroy(state->handle);
        return -1;
    }

    state->initialized = true;
    return 0;
}

// ============================================================================
// rocblas_gemm_launch_f32_to_f16 — launch the JIT-compiled F32→F16 kernel
// Pattern: baseline convert.cu convert_unary kernel (CUDA_DEQUANTIZE_BLOCK_SIZE=256)
// The kernel function handle comes from hipModuleGetFunction on the .hsaco module.
// ============================================================================

static inline int rocblas_gemm_launch_f32_to_f16(
        hipFunction_t  kernel_f32_to_f16,
        const float *  f32_input,
        half *         f16_output,
        int64_t        n_elements,
        hipStream_t    stream) {

    const int block_size = 256;  // CUDA_DEQUANTIZE_BLOCK_SIZE from baseline convert.cuh:4
    const int grid_size = (int)((n_elements + block_size - 1) / block_size);

    void * args[] = {
        (void *)&f32_input,
        (void *)&f16_output,
        (void *)&n_elements,
    };

    hipError_t e = hipModuleLaunchKernel(
        kernel_f32_to_f16,
        grid_size, 1, 1,       // grid
        block_size, 1, 1,      // block
        0,                     // shared mem
        stream,
        args, nullptr);

    if (e != hipSuccess) {
        fprintf(stderr, "rocblas-gemm: gemm_f32_to_f16 launch failed: %s\n", hipGetErrorString(e));
        return -1;
    }
    return 0;
}

// ============================================================================
// rocblas_gemm_launch_dequant_f16 — launch a JIT-compiled weight dequant-to-F16 kernel
// Pattern: baseline convert.cu dequantize_row_TYPE_cuda functions
// The kernel function handle comes from hipModuleGetFunction on the .hsaco module.
//
// This is a generic launcher for dequant kernels with the signature:
//   __global__ void dequant_TYPE_to_f16(const void * weight, __half * out, int64_t n_elements)
//
// grid_size and block_size are caller-computed to match the kernel's expected
// launch configuration (varies per quant type — see dequant_f16_launch_params).
// ============================================================================

static inline int rocblas_gemm_launch_dequant_f16(
        hipFunction_t  dequant_kernel,
        const void *   weight_data,
        half *         f16_output,
        int64_t        n_elements,
        int            grid_size,
        int            block_size,
        hipStream_t    stream) {

    void * args[] = {
        (void *)&weight_data,
        (void *)&f16_output,
        (void *)&n_elements,
    };

    hipError_t e = hipModuleLaunchKernel(
        dequant_kernel,
        grid_size, 1, 1,
        block_size, 1, 1,
        0,
        stream,
        args, nullptr);

    if (e != hipSuccess) {
        fprintf(stderr, "rocblas-gemm: dequant_f16 launch failed: %s\n", hipGetErrorString(e));
        return -1;
    }
    return 0;
}

// ============================================================================
// rocblas_gemm_exec — dequant weight → F16, convert input → F16, call hipblasGemmEx
//
// Pattern: baseline ggml_cuda_op_mul_mat_cublas lines 1457-1515
// Specifically the RDNA3 / CDNA path (lines 1483-1497):
//   - F16 inputs (weight + activation)
//   - F32 compute (HIPBLAS_COMPUTE_32F)
//   - F32 output (HIPBLAS_R_32F)
//   - hipblasGemmEx with HIPBLAS_OP_T, HIPBLAS_OP_N
//
// Matrix layout (matching baseline):
//   weight: [out_dim x in_dim] row-major = [in_dim x out_dim] col-major
//   input:  [batch_size x in_dim] row-major = [in_dim x batch_size] col-major
//   output: [batch_size x out_dim] row-major = [out_dim x batch_size] col-major
//
// GEMM: output = weight^T * input
//   (out_dim x batch_size) = (out_dim x in_dim) * (in_dim x batch_size)
//
// hipblasGemmEx args (mapped from baseline lines 1490-1497):
//   transa = HIPBLAS_OP_T   (transpose weight: stored as [out_dim, in_dim] contiguous rows)
//   transb = HIPBLAS_OP_N   (input as-is)
//   m = out_dim             (= row_diff in baseline)
//   n = batch_size          (= src1_ncols in baseline)
//   k = in_dim              (= ne10 in baseline)
//   alpha = 1.0f
//   A = weight_f16          (leading dim = in_dim = ne00)
//   B = input_f16           (leading dim = in_dim = ne10)
//   beta = 0.0f
//   C = f32_output          (leading dim = out_dim = ldc)
//   computeType = HIPBLAS_COMPUTE_32F
//   algo = HIPBLAS_GEMM_DEFAULT
//
// Parameters:
//   kernel_f32_to_f16: JIT-compiled F32→F16 kernel handle (from hipModuleGetFunction)
//   dequant_kernel: JIT-compiled weight dequant-to-F16 kernel handle, or nullptr if weight is F16
//   dequant_grid_size: number of blocks to launch for dequant kernel (caller-computed per quant type)
//   dequant_block_size: threads per block for dequant kernel launch (32 for K-quants, 256 for simple)
//   weight: device pointer to quantized (or F16) weight data
//   weight_type: ggml_type enum value (integer)
//   n_weight_elements: total number of output elements after dequantization (out_dim * in_dim)
//   f32_input: device pointer to F32 activations [batch_size x in_dim]
//   in_dim: inner dimension (K in GEMM)
//   out_dim: output dimension (M in GEMM)
//   batch_size: number of input vectors (N in GEMM)
//   f32_output: device pointer to F32 output [batch_size x out_dim]
//   stream: HIP stream to execute on
// ============================================================================

static inline int rocblas_gemm_exec(
        rocblas_gemm_state * state,
        hipFunction_t   kernel_f32_to_f16,
        hipFunction_t   dequant_kernel,
        int             dequant_grid_size,
        int             dequant_block_size,
        const void *    weight,
        int             weight_type,
        int64_t         n_weight_elements,
        const float *   f32_input,
        int             in_dim,
        int             out_dim,
        int             batch_size,
        float *         f32_output,
        hipStream_t     stream) {

    if (!state || !state->initialized) {
        fprintf(stderr, "rocblas-gemm: state not initialized\n");
        return -1;
    }

    // --- Step 1: Dequantize weight to F16 ---
    // Pattern: baseline lines 1459-1467
    //   if (src0->type != GGML_TYPE_F16) {
    //       to_fp16_cuda(src0_dd_i, src0_as_f16.get(), row_diff*ne00, stream);
    //   }
    //   const half * src0_ptr = src0->type == GGML_TYPE_F16 ? (const half *)src0_dd_i : src0_as_f16.get();

    const half * weight_f16_ptr;

    if (weight_type == ROCBLAS_GEMM_TYPE_F16) {
        // F16 weight: use directly, no conversion needed
        weight_f16_ptr = (const half *)weight;
    } else {
        // Quantized or other type: dequantize to F16 temp buffer
        if (!dequant_kernel) {
            fprintf(stderr, "rocblas-gemm: no dequant kernel for weight_type %d\n", weight_type);
            return -1;
        }

        // Validate temp buffer size
        size_t needed = (size_t)n_weight_elements * sizeof(half);
        if (needed > state->weight_temp_bytes) {
            fprintf(stderr, "rocblas-gemm: weight temp buffer too small (%zu needed, %zu available)\n",
                    needed, state->weight_temp_bytes);
            return -1;
        }

        int rc = rocblas_gemm_launch_dequant_f16(
            dequant_kernel,
            weight,
            (half *)state->weight_f16_temp,
            n_weight_elements,
            dequant_grid_size,
            dequant_block_size,
            stream);
        if (rc != 0) return rc;

        weight_f16_ptr = (const half *)state->weight_f16_temp;
    }

    // --- Step 2: Convert F32 input to F16 ---
    // Pattern: baseline lines 1469-1477
    //   if (src1->type != GGML_TYPE_F16) {
    //       to_fp16_cuda(src1_ddf_i, src1_as_f16.get(), src1_ncols*ne10, stream);
    //   }

    int64_t n_input_elements = (int64_t)batch_size * in_dim;
    size_t input_needed = (size_t)n_input_elements * sizeof(half);
    if (input_needed > state->input_temp_bytes) {
        fprintf(stderr, "rocblas-gemm: input temp buffer too small (%zu needed, %zu available)\n",
                input_needed, state->input_temp_bytes);
        return -1;
    }

    if (!kernel_f32_to_f16) {
        fprintf(stderr, "rocblas-gemm: kernel_f32_to_f16 handle is null\n");
        return -1;
    }

    {
        int rc = rocblas_gemm_launch_f32_to_f16(
            kernel_f32_to_f16,
            f32_input,
            (half *)state->input_f16_temp,
            n_input_elements,
            stream);
        if (rc != 0) return rc;
    }

    const half * input_f16_ptr = (const half *)state->input_f16_temp;

    // --- Step 3: Call hipblasGemmEx ---
    // Pattern: baseline lines 1483-1497 (CDNA / RDNA3 path)
    //   cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
    //       row_diff, src1_ncols, ne10,
    //       &alpha, src0_ptr,  CUDA_R_16F, ne00,
    //               src1_ptr,  CUDA_R_16F, ne10,
    //       &beta,  dst_dd_i,  CUDA_R_32F, ldc,
    //       CUBLAS_COMPUTE_32F,
    //       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    //
    // Mapped via vendors/hip.h:
    //   cublasGemmEx      -> hipblasGemmEx       (line 45)
    //   CUBLAS_OP_T       -> HIPBLAS_OP_T         (line 21)
    //   CUBLAS_OP_N       -> HIPBLAS_OP_N         (line 20)
    //   CUDA_R_16F        -> HIPBLAS_R_16F        (line 24)
    //   CUDA_R_32F        -> HIPBLAS_R_32F        (line 26)
    //   CUBLAS_COMPUTE_32F-> HIPBLAS_COMPUTE_32F  (line 162, HIP >= 6.5)
    //                        HIPBLAS_R_32F         (line 168, HIP < 6.5)
    //   CUBLAS_GEMM_DEFAULT_TENSOR_OP -> HIPBLAS_GEMM_DEFAULT (line 19)

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    hipblasStatus_t st = hipblasSetStream(state->handle, stream);
    if (st != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "rocblas-gemm: hipblasSetStream failed: %d\n", (int)st);
        return -1;
    }

    // HIP >= 6.5 uses hipblasComputeType_t; older uses hipblasDatatype_t
    // vendors/hip.h lines 160-172 handle this mapping.
    // We use the HIP >= 6.5 path since ROCm 7.1 is the target.
#if HIP_VERSION >= 60500000
    hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F;
#else
    hipblasDatatype_t    compute_type = HIPBLAS_R_32F;
#endif

    st = hipblasGemmEx(
        state->handle,
        HIPBLAS_OP_T,                  // transa: transpose weight
        HIPBLAS_OP_N,                  // transb: input as-is
        out_dim,                       // m = row_diff
        batch_size,                    // n = src1_ncols
        in_dim,                        // k = ne10
        &alpha,
        weight_f16_ptr, HIPBLAS_R_16F, in_dim,      // A = weight, lda = ne00
        input_f16_ptr,  HIPBLAS_R_16F, in_dim,      // B = input,  ldb = ne10
        &beta,
        f32_output,     HIPBLAS_R_32F, out_dim,      // C = output, ldc = out_dim
        compute_type,
        HIPBLAS_GEMM_DEFAULT);

    if (st != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "rocblas-gemm: hipblasGemmEx failed: %d\n", (int)st);
        return -1;
    }

    return 0;
}

// ============================================================================
// rocblas_gemm_free — destroy handle, free temp buffers
// Pattern: baseline ggml_backend_cuda_context destructor
// ============================================================================

static inline void rocblas_gemm_free(rocblas_gemm_state * state) {
    if (!state || !state->initialized) {
        return;
    }

    if (state->weight_f16_temp) {
        hipFree(state->weight_f16_temp);
        state->weight_f16_temp = nullptr;
    }
    if (state->input_f16_temp) {
        hipFree(state->input_f16_temp);
        state->input_f16_temp = nullptr;
    }

    hipblasDestroy(state->handle);
    state->handle = nullptr;
    state->initialized = false;
}
