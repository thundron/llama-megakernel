#pragma once
#include "hip-shim.h"
#include "mem-utils.h"
#include <type_traits>
// Port of ggml/src/ggml-cuda/mma.cuh — RDNA3/AMD_WMMA paths ONLY
// All other architectures (Volta, Turing, Ampere, Blackwell, CDNA/MFMA, RDNA4) removed.
#define AMD_WMMA_AVAILABLE 1
#define RDNA3 1

// HIP bfloat16 support — baseline vendors/hip.h lines 245-246
// hip_bf16.h already included via hip-shim.h (must be before __shfl_sync macros)
typedef __hip_bfloat16  nv_bfloat16;
typedef __hip_bfloat162 nv_bfloat162;

// mma.cuh line 19: common.cuh include replaced by hip-shim.h + mem-utils.h above.

namespace ggml_cuda_mma {

    // mma.cuh lines 74-83: data_layout enum
    enum data_layout {
        DATA_LAYOUT_I_MAJOR           =  0,
        DATA_LAYOUT_J_MAJOR           = 10,
        DATA_LAYOUT_I_MAJOR_MIRRORED  = 20,
        DATA_LAYOUT_J_MAJOR_MIRRORED  = 30,
    };

    // mma.cuh lines 89-92
    static constexpr bool is_i_major(const data_layout dl) {
        return dl == DATA_LAYOUT_I_MAJOR ||
               dl == DATA_LAYOUT_I_MAJOR_MIRRORED;
    }

    // mma.cuh lines 94-100 — RDNA3 path only, made unconditional
    static constexpr __device__ data_layout get_input_data_layout() {
        return DATA_LAYOUT_I_MAJOR_MIRRORED;
    }

    // Forward declarations for cross-references
    template <int I_, int J_, typename T, data_layout ds_=DATA_LAYOUT_I_MAJOR>
    struct tile {};

    // ========================================================================
    // tile<I, J, T, DATA_LAYOUT_I_MAJOR> — AMD_WMMA path
    // mma.cuh lines 106, 187-230 (AMD_WMMA_AVAILABLE block)
    // ========================================================================
    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

        // mma.cuh line 188
        static constexpr int ne = I * J / 32;
        T x[ne] = {0};

        // mma.cuh lines 191-196
        static constexpr __device__ bool supported() {
            if (I == 16 && J == 16) return true;
            if (I == 16 && J == 8) return true;
            if (I == 16 && J == 4) return true;
            return false;
        }

        // mma.cuh lines 198-204
        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (supported()) {
                return threadIdx.x % 16;
            } else {
                // NO_DEVICE_CODE placeholder — should never be reached on gfx1100
                return -1;
            }
        }

        // mma.cuh lines 207-230 — RDNA3 path within AMD_WMMA
        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 16 && J == 16) {
                // mma.cuh lines 209-216: RDNA3 path
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int>) {
                    // matrix C
                    return 2 * l + (threadIdx.x / 16);
                } else {
                    // matrix A&B
                    return l;
                }
            } else if constexpr (I == 16 && J == 8) {
                // mma.cuh line 223 — RDNA3 uses same formula as RDNA4 here
                return ne * (threadIdx.x / 16) + l;
            } else if constexpr (I == 16 && J == 4) {
                // mma.cuh lines 224-226
                return ne * (threadIdx.x / 16) + l;
            } else {
                return -1;
            }
        }
    };

    // ========================================================================
    // tile<I, J, half2, DATA_LAYOUT_I_MAJOR> — AMD_WMMA path
    // mma.cuh lines 281, 316-341
    // ========================================================================
    template <int I_, int J_>
    struct tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

        // mma.cuh line 317
        static constexpr int ne = I * J / 32;
        half2 x[ne] = {{0.0f, 0.0f}};

        // mma.cuh lines 320-322
        static constexpr __device__ bool supported() {
            if (I == 16 && J == 8) return true;
            return false;
        }

        // mma.cuh lines 325-330
        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 16 && J == 8) {
                return threadIdx.x % 16;
            } else {
                return -1;
            }
        }

        // mma.cuh lines 334-341
        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 16 && J == 8) {
                return ne * (threadIdx.x / 16) + l;
            } else {
                return -1;
            }
        }
    };

    // ========================================================================
    // tile<I, J, nv_bfloat162, DATA_LAYOUT_I_MAJOR> — AMD_WMMA path
    // mma.cuh lines 413-486 (AMD_WMMA_AVAILABLE block, lines 419-433)
    // ========================================================================
    template <int I_, int J_>
    struct tile<I_, J_, nv_bfloat162, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

        // mma.cuh line 420
        static constexpr int ne = tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::ne;
        nv_bfloat162 x[ne] = {{0.0f, 0.0f}};

        // mma.cuh lines 423-425
        static constexpr __device__ bool supported() {
            return tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::supported();
        }

        // mma.cuh lines 427-429
        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::get_i(l);
        }

        // mma.cuh lines 431-433
        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::get_j(l);
        }
    };

    // ========================================================================
    // tile<I, J, T, DATA_LAYOUT_J_MAJOR> — arch-independent
    // mma.cuh lines 488-508
    // ========================================================================
    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_J_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_J_MAJOR;

        static constexpr int ne = tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::ne;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::supported();
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_j(l);
        }

        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_i(l);
        }
    };

    // ========================================================================
    // tile<I, J, T, DATA_LAYOUT_I_MAJOR_MIRRORED> — RDNA3 specific
    // mma.cuh lines 510-545
    // ========================================================================
    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR_MIRRORED;

        // mma.cuh line 517: RDNA3
        static constexpr int         ne = I * J / 32 * 2;

        T x[ne] = {0};

        // mma.cuh lines 521-526
        static constexpr __device__ bool supported() {
            if (I == 16 && J == 16) return true;
            if (I == 16 && J == 8)  return true;
            if (I == 16 && J == 4)  return true;
            return false;
        }

        // mma.cuh lines 528-534
        static __device__ __forceinline__ int get_i(const int /*l*/) {
            if constexpr (supported()) {
                return threadIdx.x % 16;
            } else {
                return -1;
            }
        }

        // mma.cuh lines 537-544
        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (supported()) {
                return l;
            } else {
                return -1;
            }
        }
    };

    // ========================================================================
    // tile<I, J, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> — RDNA3 path
    // mma.cuh lines 547-596 (RDNA3 block, lines 552-567)
    // ========================================================================
    template <int I_, int J_>
    struct tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR_MIRRORED;

        // mma.cuh line 553: RDNA3
        static constexpr int         ne = tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::ne;

        half2 x[ne] = {{0.0f, 0.0f}};

        // mma.cuh lines 557-559
        static constexpr __device__ bool supported() {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::supported();
        }

        // mma.cuh lines 561-563
        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_i(l);
        }

        // mma.cuh lines 565-567
        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_j(l);
        }
    };

    // ========================================================================
    // tile<I, J, nv_bfloat162, DATA_LAYOUT_I_MAJOR_MIRRORED>
    // mma.cuh lines 598-618
    // ========================================================================
    template <int I_, int J_>
    struct tile<I_, J_, nv_bfloat162, DATA_LAYOUT_I_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR_MIRRORED;
        // mma.cuh line 603
        static constexpr int         ne = tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::ne;

        nv_bfloat162 x[ne] = {{0.0f, 0.0f}};

        // mma.cuh lines 607-609
        static constexpr __device__ bool supported() {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::supported();
        }

        // mma.cuh lines 611-613
        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_i(l);
        }

        // mma.cuh lines 615-617
        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_j(l);
        }
    };

    // ========================================================================
    // tile<I, J, half2, DATA_LAYOUT_J_MAJOR_MIRRORED>
    // mma.cuh lines 620-651
    // ========================================================================
    template <int I_, int J_>
    struct tile<I_, J_, half2, DATA_LAYOUT_J_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_J_MAJOR_MIRRORED;
        static constexpr int         ne = I * J / (WARP_SIZE/4);

        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  4) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 4) {
                return ((l / 2) * 4) + (threadIdx.x % 4);
            } else {
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 4) {
                return ((threadIdx.x / 16) * 2) + (l % 2);
            } else {
                return -1;
            }
        }
    };

    // ========================================================================
    // get_half2() — AMD_WMMA path
    // mma.cuh lines 671-680 (AMD_WMMA_AVAILABLE || AMD_MFMA_AVAILABLE block)
    // ========================================================================
    template <int I, int J>
    static __device__ __forceinline__ tile<I, J/2, half2> get_half2(const tile<I, J, float> & tile_float) {
        tile<I, J/2, half2> ret;
#pragma unroll
        for (int l0 = 0; l0 < tile_float.ne; l0 += 2) {
            ret.x[l0/2] = make_half2(tile_float.x[l0 + 0], tile_float.x[l0 + 1]);
        }
        return ret;
    }

    // mma.cuh lines 682-685: get_transposed — not supported on RDNA3 (stub)
    static __device__ __forceinline__ tile<8, 8, half2> get_transposed(const tile<16, 4, half2> & t) {
        (void)t;
        // NO_DEVICE_CODE — not reachable on gfx1100
        return tile<8, 8, half2>{};
    }

    // ========================================================================
    // load_generic() — AMD_WMMA path
    // mma.cuh lines 717-755 (AMD_WMMA_AVAILABLE block, lines 728-748)
    // ========================================================================
    template <int I, int J, typename T, data_layout dl>
    static __device__ __forceinline__ void load_generic(tile<I, J, T, dl> & t, const T * __restrict__ xs0, const int stride) {
        // mma.cuh lines 729-748: AMD_WMMA path
        // All wmma layout has contiguous data when i-major.
        if constexpr (is_i_major(dl)) {
            // the data must be aligned to 16 bytes when bigger than ggml_cuda_get_max_cpy_bytes()
            constexpr int aligned_copy_bytes = ggml_cuda_get_max_cpy_bytes();
            if constexpr (sizeof(t.x) > aligned_copy_bytes) {
                static_assert(sizeof(t.x) % aligned_copy_bytes == 0, "bad type size");
                constexpr int aligned_copy_count = sizeof(t.x)/aligned_copy_bytes;
#pragma unroll
                for (int i = 0; i < aligned_copy_count; ++i) {
                    ggml_cuda_memcpy_1<aligned_copy_bytes>(t.x + t.ne/aligned_copy_count*i, xs0 + t.get_i(0) * stride + t.get_j(t.ne/aligned_copy_count*i));
                }
            } else {
                ggml_cuda_memcpy_1<sizeof(t.x)>(t.x, xs0 + t.get_i(0) * stride + t.get_j(0));
            }
        } else {
#pragma unroll
            for (int l = 0; l < t.ne; ++l) {
                t.x[l] = xs0[t.get_i(l)*stride + t.get_j(l)];
            }
        }
    }

    // ========================================================================
    // load_ldmatrix overloads — non-TURING paths that dispatch to load_generic
    // ========================================================================

    // mma.cuh lines 758-769: tile<8,8,T> — no TURING, falls to load_generic
    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 8, T> & t, const T * __restrict__ xs0, const int stride) {
        load_generic(t, xs0, stride);
    }

    // mma.cuh lines 771-788: tile<16,4,T> — non-Turing, non-Volta path
    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 4, T> & t, const T * __restrict__ xs0, const int stride) {
        load_generic(t, xs0, stride);
    }

    // mma.cuh lines 790-813: tile<16,8,T,dl> — non-Turing, non-Volta path
    template <typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 8, T, dl> & t, const T * __restrict__ xs0, const int stride) {
        load_generic(t, xs0, stride);
    }

    // mma.cuh lines 815-818: tile<8,4, half2, I_MAJOR_MIRRORED>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> & t, const half2 * __restrict__ xs0, const int stride) {
        ggml_cuda_memcpy_1<4*sizeof(half2)>(t.x, xs0 + t.get_i(0)*stride);
    }

    // mma.cuh lines 820-826: tile<8,4, half2, J_MAJOR_MIRRORED>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED> & t, const half2 * __restrict__ xs0, const int stride) {
#pragma unroll
        for (int l0 = 0; l0 < t.ne; l0 += 2) {
            ggml_cuda_memcpy_1<2*sizeof(half2)>(t.x + l0, xs0 + t.get_i(l0)*stride + t.get_j(l0));
        }
    }

    // Note: load_ldmatrix_trans (mma.cuh lines 838-851) is TURING-only, omitted for RDNA3.

    // ========================================================================
    // mma() overloads — RDNA3 WMMA intrinsics
    // ========================================================================

    // mma.cuh lines 1105-1163: mma(tile<16,16,float>, tile<16,8,half2>, tile<16,8,half2>) — RDNA3 WMMA FP16->FP32
    template <data_layout dl_ab, data_layout dl_d>
    static __device__ __forceinline__ void mma(
            tile<16, 16, float, dl_d> & D, const tile<16, 8, half2, dl_ab> & A, const tile<16, 8, half2, dl_ab> & B) {
        // mma.cuh lines 1142-1148: RDNA3 path
        using halfx16_t = __attribute__((ext_vector_type(16))) _Float16;
        using floatx8_t = __attribute__((ext_vector_type(8))) float;
        floatx8_t& acc_frag = reinterpret_cast<floatx8_t&>(D.x[0]);
        const halfx16_t& a_frag = reinterpret_cast<const halfx16_t&>(A.x[0]);
        const halfx16_t& b_frag = reinterpret_cast<const halfx16_t&>(B.x[0]);
        acc_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, acc_frag);
    }

    // mma.cuh lines 1166-1211: mma(tile<16,16,float>, tile<16,8,nv_bfloat162>, tile<16,8,nv_bfloat162>) — RDNA3 WMMA BF16->FP32
    template <data_layout dl_ab, data_layout dl_d>
    static __device__ __forceinline__ void mma(
            tile<16, 16, float, dl_d> & D, const tile<16, 8, nv_bfloat162, dl_ab> & A, const tile<16, 8, nv_bfloat162, dl_ab> & B) {
        // mma.cuh lines 1177-1183: RDNA3 path
        using bf16x16_t = __attribute__((ext_vector_type(16))) __bf16;
        using floatx8_t = __attribute__((ext_vector_type(8))) float;
        floatx8_t& acc_frag = reinterpret_cast<floatx8_t&>(D.x[0]);
        const bf16x16_t& a_frag = reinterpret_cast<const bf16x16_t&>(A.x[0]);
        const bf16x16_t& b_frag = reinterpret_cast<const bf16x16_t&>(B.x[0]);
        acc_frag = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, acc_frag);
    }

    // mma.cuh lines 1214-1292: mma(tile<16,16,int>, tile<16,8,int>, tile<16,8,int>) — RDNA3 WMMA INT8
    template <data_layout dl_d, data_layout dl_ab>
    static __device__ __forceinline__ void mma(
            tile<16, 16, int, dl_d> & D, const tile<16, 8, int, dl_ab> & A, const tile<16, 8, int, dl_ab> & B) {
        // mma.cuh lines 1236-1291: AMD_WMMA path, RDNA3 sub-path (lines 1264-1286)
        using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
        int32x8_t * acc = (int32x8_t *) D.x;

        using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
        int32x4_t * a_vec = (int32x4_t *) A.x;
        int32x4_t * b_vec = (int32x4_t *) B.x;

        // mma.cuh lines 1269-1275
        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true,
            a_vec[0],
            true,
            b_vec[0],
            acc[0],
            true
        );

        // mma.cuh lines 1278-1285
        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true,
            a_vec[1],
            true,
            b_vec[1],
            acc[0],
            true
        );
    }

    // mma.cuh lines 1370-1408: mma(tile<16,16,int>, tile<16,4,int>, tile<16,4,int>) — RDNA3 WMMA INT8 (single k-step)
    template <data_layout dl_d, data_layout dl_ab>
    static __device__ __forceinline__ void mma(
            tile<16, 16, int, dl_d> & D, const tile<16, 4, int, dl_ab> & A, const tile<16, 4, int, dl_ab> & B) {
        // mma.cuh lines 1373-1408: AMD_WMMA path, RDNA3 sub-path (lines 1389-1401)
        using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
        int32x8_t * acc = (int32x8_t *) D.x;

        using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
        int32x4_t * a_vec = (int32x4_t *) A.x;
        int32x4_t * b_vec = (int32x4_t *) B.x;

        // mma.cuh lines 1394-1401
        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true,
            a_vec[0],
            true,
            b_vec[0],
            acc[0],
            false
        );
    }

    // mma.cuh lines 1321-1328: Generic 32-row mma via two 16-row mma calls
    template <typename T1, typename T2, int J, int K>
    static __device__ __forceinline__ void mma(
            tile<32, J, T1> & D, const tile<32, K, T2> & A, const tile<J, K, T2> & B) {
        tile      <16, J, T1> * D16 = reinterpret_cast<      tile<16, J, T1> *>(&D);
        const tile<16, K, T2> * A16 = reinterpret_cast<const tile<16, K, T2> *>(&A);
        mma(D16[0], A16[0], B);
        mma(D16[1], A16[1], B);
    }

} // namespace ggml_cuda_mma
