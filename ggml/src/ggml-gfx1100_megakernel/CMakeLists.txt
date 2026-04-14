ggml_add_backend_library(ggml-gfx1100_megakernel
    gfx1100-megakernel.cpp
    gfx1100-megakernel.h
    gfx1100-internal.h
    gfx1100-init.cpp
    # Shared host-side helpers
    shared/dispatch.cpp
    shared/batch-ops.cpp
    # Decode forward functions
    forward/llama-family.cpp
    forward/mamba.cpp
    forward/rwkv6.cpp
    forward/t5-dec.cpp
    forward/deepseek2-mla.cpp
    forward/rwkv7.cpp
    forward/prompt.cpp
    # Encoder forward functions
    encoder/t5-enc.cpp
    encoder/bert.cpp
    # Audio decoder
    audio/wavtokenizer.cpp
    # Special arch forward functions
    forward/bitnet.cpp
    forward/cogvlm.cpp
)

target_link_libraries(ggml-gfx1100_megakernel PRIVATE ggml-base)
target_include_directories(ggml-gfx1100_megakernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# HIP runtime
find_package(hip QUIET)
if(hip_FOUND)
    target_link_libraries(ggml-gfx1100_megakernel PRIVATE hip::host)
else()
    find_library(HIP_LIB amdhip64 PATHS "$ENV{ROCM_PATH}/lib" "$ENV{HIP_PATH}/lib" "C:/Program Files/AMD/ROCm/7.1/lib")
    if(HIP_LIB)
        target_link_libraries(ggml-gfx1100_megakernel PRIVATE ${HIP_LIB})
    endif()
    target_include_directories(ggml-gfx1100_megakernel PRIVATE
        "$ENV{ROCM_PATH}/include"
        "C:/Program Files/AMD/ROCm/7.1/include"
    )
endif()

# hipBLAS / rocBLAS — needed for GEMM fallback path (rocblas-gemm.h)
# Pattern: baseline ggml-cuda links cuBLAS; we link the HIP equivalents.
# vendors/hip.h maps cuBLAS → hipBLAS at compile time; hipBLAS calls rocBLAS internally.
find_package(hipblas QUIET)
find_package(rocblas QUIET)
if(hipblas_FOUND AND rocblas_FOUND)
    target_link_libraries(ggml-gfx1100_megakernel PRIVATE roc::hipblas roc::rocblas)
else()
    # Fallback: find libraries manually (Windows ROCm installs)
    find_library(HIPBLAS_LIB hipblas PATHS "$ENV{ROCM_PATH}/lib" "C:/Program Files/AMD/ROCm/7.1/lib")
    find_library(ROCBLAS_LIB rocblas PATHS "$ENV{ROCM_PATH}/lib" "C:/Program Files/AMD/ROCm/7.1/lib")
    if(HIPBLAS_LIB AND ROCBLAS_LIB)
        target_link_libraries(ggml-gfx1100_megakernel PRIVATE ${HIPBLAS_LIB} ${ROCBLAS_LIB})
    else()
        message(WARNING "hipblas/rocblas not found — GEMM fallback (rocblas-gemm.h) will not link")
    endif()
    target_include_directories(ggml-gfx1100_megakernel PRIVATE
        "$ENV{ROCM_PATH}/include"
        "C:/Program Files/AMD/ROCm/7.1/include"
    )
endif()

# Test: exactly two kernels
add_executable(test-two-kernels tests/test-two-kernels.cpp)
add_test(NAME test-two-kernels COMMAND test-two-kernels)

# Test: cooperative groups on gfx1100 (compiled with hipcc directly)
find_program(HIPCC_COMPILER hipcc PATHS "C:/Program Files/AMD/ROCm/7.1/bin" "$ENV{ROCM_PATH}/bin")
if(HIPCC_COMPILER)
    add_custom_command(
        OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-coop-launch${CMAKE_EXECUTABLE_SUFFIX}"
        COMMAND "${HIPCC_COMPILER}" --offload-arch=gfx1100 -O2
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test-coop-launch.cpp"
            -o "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-coop-launch${CMAKE_EXECUTABLE_SUFFIX}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test-coop-launch.cpp"
        COMMENT "Building test-coop-launch with hipcc"
        VERBATIM)
    add_custom_target(test-coop-launch DEPENDS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-coop-launch${CMAKE_EXECUTABLE_SUFFIX}")
endif()

# ---------------------------------------------------------------------------
# Helper: add a GPU kernel unit test with HIP runtime linking
# ---------------------------------------------------------------------------
function(add_kernel_test TARGET_NAME SOURCE_FILE)
    cmake_parse_arguments(ARG "NEEDS_LLAMA" "" "" ${ARGN})

    add_executable(${TARGET_NAME} tests/${SOURCE_FILE})

    if(ARG_NEEDS_LLAMA)
        target_link_libraries(${TARGET_NAME} PRIVATE llama ggml)
        target_include_directories(${TARGET_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/ggml/include
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CMAKE_SOURCE_DIR}/common
        )
    else()
        target_link_libraries(${TARGET_NAME} PRIVATE ggml)
        target_include_directories(${TARGET_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/include
            ${CMAKE_SOURCE_DIR}/ggml/src
        )
    endif()

    # HIP runtime for hipModuleLoad / hipModuleLaunchKernel
    if(hip_FOUND)
        target_link_libraries(${TARGET_NAME} PRIVATE hip::host)
    else()
        if(HIP_LIB)
            target_link_libraries(${TARGET_NAME} PRIVATE ${HIP_LIB})
        endif()
        target_include_directories(${TARGET_NAME} PRIVATE
            "$ENV{ROCM_PATH}/include"
            "C:/Program Files/AMD/ROCm/7.1/include"
        )
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Original tests (model-based)
# ---------------------------------------------------------------------------
add_kernel_test(test-megakernel-e2e   test-megakernel-e2e.cpp   NEEDS_LLAMA)
add_kernel_test(test-kernel-embed     test-kernel-embed.cpp     NEEDS_LLAMA)
add_kernel_test(test-kernel-rmsnorm   test-kernel-rmsnorm.cpp   NEEDS_LLAMA)
add_kernel_test(test-kernel-matvec    test-kernel-matvec.cpp    NEEDS_LLAMA)
add_kernel_test(test-kernel-qknorm    test-kernel-qknorm.cpp    NEEDS_LLAMA)
add_kernel_test(test-kernel-quantize  test-kernel-quantize.cpp)

# ---------------------------------------------------------------------------
# NEW: Synthetic unit tests (no model required)
# Run these BEFORE any e2e tests to catch kernel bugs safely.
# ---------------------------------------------------------------------------

# Simple element-wise ops: add_residual, silu_mul, final_norm
add_kernel_test(test-kernel-simple-ops    test-kernel-simple-ops.cpp)

# Newly-ported kernels (this session): GELU variants, scale_scalar, softcap,
# layernorm, l2norm, sigmoid, softplus, argsort_desc, softmax_row, concat,
# repeat, axpy, sum_row, ssm_conv_step, ssm_scan_step.
# Each test uses a CPU reference that mirrors the baseline ggml-cuda op exactly.
add_kernel_test(test-kernel-ported-ops    test-kernel-ported-ops.cpp  NEEDS_LLAMA)

# Float-type matvec: f16, bf16, f32 + residual variants
add_kernel_test(test-kernel-matvec-float  test-kernel-matvec-float.cpp)

# Attention decode with FA_HEAD_DIM=64 (matches Llama 1B .hsaco)
add_kernel_test(test-kernel-attention-d64 test-kernel-attention-d64.cpp)

# Float-type embeddings: f32, f16, bf16
add_kernel_test(test-kernel-embed-float   test-kernel-embed-float.cpp)

# All quantized matvec types: Q4_0..Q6_K (uses ggml quantize/dequantize as reference)
add_kernel_test(test-kernel-matvec-qtypes test-kernel-matvec-qtypes.cpp  NEEDS_LLAMA)

# LM head: Q4_K and Q6_K vocab projection
# test-kernel-lmhead removed — eval_lm_head/q6k dead code, LM head uses generic pick_matvec

# Init safety tests (MoE detection, attention scale, etc.) — no GPU kernels launched
add_kernel_test(test-init-safety  test-init-safety.cpp  NEEDS_LLAMA)

# Buffer safety tests — sentinel overflow checks, elementwise_mul kernel test
add_kernel_test(test-buffer-safety  test-buffer-safety.cpp)

# Operation ordering test — verify bias→scale→residual matches baseline
add_kernel_test(test-operation-ordering  test-operation-ordering.cpp)

# NULL pointer safety — verify NULL optional tensors don't crash kernels
add_kernel_test(test-null-safety  test-null-safety.cpp)

# Element-wise kernels: tanh, neg, exp, relu, sqr, sigmoid, gelu, sub, muladd
add_kernel_test(test-kernel-elementwise  test-kernel-elementwise.cpp)

# MoE FFN integration: softmax, argsort, routing pipeline
add_kernel_test(test-kernel-moe          test-kernel-moe.cpp)

# All prompt dequant kernels: Q4_0..Q6_K, F32, BF16 → F16
add_kernel_test(test-kernel-dequant       test-kernel-dequant.cpp       NEEDS_LLAMA)

# DeltaNet: conv1d_silu, l2_norm, recurrence
add_kernel_test(test-kernel-deltanet      test-kernel-deltanet.cpp)

# Prompt batch ops: rmsnorm, add_residual, silu_mul, final_norm, embed
add_kernel_test(test-kernel-prompt-ops    test-kernel-prompt-ops.cpp)

# MMQ Q8_1 batch quantization: d4, ds4, d2s6 layouts
add_kernel_test(test-kernel-mmq-quantize  test-kernel-mmq-quantize.cpp)

# Baseline logit dumper (uses standard llama_decode, no megakernel)
add_executable(test-baseline-logits tests/test-baseline-logits.cpp)
target_link_libraries(test-baseline-logits PRIVATE llama ggml)
target_include_directories(test-baseline-logits PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
)

# Baseline per-layer dump (uses eval callback to intercept intermediates)
add_executable(test-baseline-dump tests/test-baseline-dump.cpp)
target_link_libraries(test-baseline-dump PRIVATE llama ggml)
target_include_directories(test-baseline-dump PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
)
