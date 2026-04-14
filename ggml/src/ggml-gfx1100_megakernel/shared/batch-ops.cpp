// shared/batch-ops.cpp — batch operation helpers
#include "batch-ops.h"

void launch_mmq_quantize(hipFunction_t fn, const float * input, void * output,
                                int in_dim, int batch_size, hipStream_t stream) {
    int64_t ne00 = in_dim;
    int64_t s01  = in_dim;
    int64_t s02  = 0;
    int64_t s03  = 0;
    int64_t ne0  = in_dim;
    int     ne1  = batch_size;
    int     ne2  = 1;
    const int32_t * ids = nullptr; // no row remapping
    int block_num_y = ((int)ne0 + 4 * 128 - 1) / (4 * 128);
    void * args[] = {
        (void *)&input, (void *)&ids, (void *)&output,
        (void *)&ne00, (void *)&s01, (void *)&s02, (void *)&s03,
        (void *)&ne0, (void *)&ne1, (void *)&ne2,
    };
    hipModuleLaunchKernel(fn, ne1, block_num_y, 1, 128, 1, 1, 0, stream, args, nullptr);
}

// Helper: launch MMQ matmul kernel
// Grid: (nty, ntx, 1) where nty = ceil(out_dim/128), ntx = ceil(batch/mmq_x)
// Block: (32, 8, 1) = warp_size * nwarps
void launch_mmq_kernel(hipFunction_t fn, const void * weight, long long weight_stride,
                              const void * q8_input, float * output,
                              int in_dim, int out_dim, int batch_size, int mmq_x,
                              size_t shared_mem, hipStream_t stream) {
    const int mmq_y = 128;
    int nty = (out_dim + mmq_y - 1) / mmq_y;
    int ntx = (batch_size + mmq_x - 1) / mmq_x;

    // mul_mat_q kernel parameters (from mmq.h signature):
    //   x, y, ids_dst, dst, tmp_fixup,
    //   ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst,
    //   channel_ratio, nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst,
    //   sample_ratio, nsamples_y, stride_sample_x, stride_sample_y, stride_sample_dst,
    //   ncols_max
    const char * x = (const char *)weight;
    const int * y = (const int *)q8_input;
    const int32_t * ids_dst = nullptr;  // no row remapping
    float * dst = output;
    float * tmp_fixup = nullptr;        // no fixup (conventional tiling)

    // ncols_x = number of columns in weight = in_dim (in quant blocks: in_dim/qk, but kernel uses raw)
    // nrows_x = out_dim
    // ncols_dst = batch_size (columns in output)
    // stride_row_x = weight_stride in bytes / type_size — but MMQ x is byte pointer, stride = bytes per row
    // For MMQ kernel, stride_row_x = number of qk-blocks per row = in_dim / qk (kernel divides by qk internally)
    // Actually: looking at the kernel, offset_x = it*mmq_y*stride_row_x, and x is a char*
    // stride_row_x should be the byte stride between rows of weight = weight_stride
    int ncols_x = in_dim;
    int nrows_x = out_dim;
    int ncols_dst_v = batch_size;
    int stride_row_x = (int)weight_stride; // byte stride per row
    // ncols_y = batch_size (number of Q8_1 columns)
    // For Q8_1 MMQ: each column is sizeof(block_q8_1_mmq) * (in_dim / 128) bytes
    // The y pointer layout: y[col * (in_dim / ne_block) * sz + block_offset]
    // ne_block = 4*QK8_1 = 128, sz = sizeof(block_q8_1_mmq)/sizeof(int) = 144/4 = 36
    int ncols_y = batch_size;
    int stride_col_dst = out_dim; // output is column-major: [out_dim x batch_size]

    // Channel/sample params: single channel, single sample (no MoE)
    int channel_ratio = 1;
    int nchannels_y = 1;
    int stride_channel_x = 0;
    int stride_channel_y = 0;
    int stride_channel_dst = 0;
    int sample_ratio = 1;
    int nsamples_y = 1;
    int stride_sample_x = 0;
    int stride_sample_y = 0;
    int stride_sample_dst = 0;
    int ncols_max = batch_size;

    void * args[] = {
        (void *)&x, (void *)&y, (void *)&ids_dst, (void *)&dst, (void *)&tmp_fixup,
        (void *)&ncols_x, (void *)&nrows_x, (void *)&ncols_dst_v, (void *)&stride_row_x,
        (void *)&ncols_y, (void *)&stride_col_dst,
        (void *)&channel_ratio, (void *)&nchannels_y,
        (void *)&stride_channel_x, (void *)&stride_channel_y, (void *)&stride_channel_dst,
        (void *)&sample_ratio, (void *)&nsamples_y,
        (void *)&stride_sample_x, (void *)&stride_sample_y, (void *)&stride_sample_dst,
        (void *)&ncols_max,
    };
    hipModuleLaunchKernel(fn, nty, ntx, 1, 32, 8, 1, shared_mem, stream, args, nullptr);
}

// Helper: perform batch projection via MMQ or rocBLAS
// weight: [out_dim, in_dim] quantized weight
// input: batch_norm [S, in_dim] f32 or batch_q8_mmq [S, ...] Q8_1
// output: [S, out_dim] f32
void batch_projection(
        int weight_type, const void * weight, long long weight_stride,
        const float * f32_input, const void * q8_input,
        float * output, int in_dim, int out_dim, int batch_size,
        hipStream_t stream) {

    auto & k = g_compiled;

    if (should_use_mmq(weight_type, batch_size)) {
        // MMQ path
        int ti = mmq_type_index(weight_type);
        if (ti < 0) {
            fprintf(stderr, "gfx1100: unsupported MMQ type %d, falling back to rocBLAS\n", weight_type);
            goto rocblas_path;
        }
        {
            int mmq_x = (batch_size <= 32) ? 32 : 64;
            int xi = (mmq_x == 32) ? 0 : 1;
            bool need_check = (out_dim % 128 != 0);
            int ci = need_check ? 1 : 0;
            hipFunction_t fn = k.prompt_mmq[ti][xi][ci];
            if (!fn) {
                fprintf(stderr, "gfx1100: MMQ kernel not loaded for type %d mmq_x=%d check=%d\n",
                        weight_type, mmq_x, ci);
                goto rocblas_path;
            }
            size_t shmem = mmq_shared_mem_size(weight_type, mmq_x);
            launch_mmq_kernel(fn, weight, weight_stride, q8_input, output,
                              in_dim, out_dim, batch_size, mmq_x, shmem, stream);
        }
        return;
    }

rocblas_path:
    if (g_rocblas.initialized) {
        // rocBLAS path: dequant weight → F16, convert input → F16, hipblasGemmEx
        // Select dequant-to-F16 kernel and compute grid/block launch params per quant type.
        // Grid/block patterns match baseline convert.cu dequantize_row_*_cuda launchers.
        hipFunction_t dequant_fn = nullptr;
        int dequant_bs = 0;
        int dequant_grid = 0;
        int64_t n_weight_elems = (int64_t)out_dim * in_dim;

        switch (weight_type) {
            // --- Small-block types (QK=32): 32 threads, grid = (k+255)/256 ---
            case  2: dequant_fn = k.dequant_f16_q4_0;  dequant_bs = 32; dequant_grid = (int)((n_weight_elems + 255) / 256); break; // Q4_0
            case  3: dequant_fn = k.dequant_f16_q4_1;  dequant_bs = 32; dequant_grid = (int)((n_weight_elems + 255) / 256); break; // Q4_1
            case  6: dequant_fn = k.dequant_f16_q5_0;  dequant_bs = 32; dequant_grid = (int)((n_weight_elems + 255) / 256); break; // Q5_0
            case  7: dequant_fn = k.dequant_f16_q5_1;  dequant_bs = 32; dequant_grid = (int)((n_weight_elems + 255) / 256); break; // Q5_1
            case  8: dequant_fn = k.dequant_f16_q8_0;  dequant_bs = 32; dequant_grid = (int)((n_weight_elems + 255) / 256); break; // Q8_0
            // --- K-quants (QK=256): 64 threads for Q2/Q3/Q5/Q6, 32 for Q4_K ---
            case 10: dequant_fn = k.dequant_f16_q2k;   dequant_bs = 64; dequant_grid = (int)(n_weight_elems / 256); break; // Q2_K
            case 11: dequant_fn = k.dequant_f16_q3k;   dequant_bs = 64; dequant_grid = (int)(n_weight_elems / 256); break; // Q3_K
            case 12: dequant_fn = k.dequant_f16_q4k;   dequant_bs = 32; dequant_grid = (int)(n_weight_elems / 256); break; // Q4_K
            case 13: dequant_fn = k.dequant_f16_q5k;   dequant_bs = 64; dequant_grid = (int)(n_weight_elems / 256); break; // Q5_K
            case 14: dequant_fn = k.dequant_f16_q6k;   dequant_bs = 64; dequant_grid = (int)(n_weight_elems / 256); break; // Q6_K
            // --- IQ types: 32 threads, grid = k/QK_K ---
            case 16: dequant_fn = k.dequant_f16_iq2_xxs; dequant_bs = 32; dequant_grid = (int)(n_weight_elems / 256); break; // IQ2_XXS
            case 17: dequant_fn = k.dequant_f16_iq2_xs;  dequant_bs = 32; dequant_grid = (int)(n_weight_elems / 256); break; // IQ2_XS
            case 22: dequant_fn = k.dequant_f16_iq2_s;   dequant_bs = 32; dequant_grid = (int)(n_weight_elems / 256); break; // IQ2_S
            case 18: dequant_fn = k.dequant_f16_iq3_xxs; dequant_bs = 32; dequant_grid = (int)(n_weight_elems / 256); break; // IQ3_XXS
            case 21: dequant_fn = k.dequant_f16_iq3_s;   dequant_bs = 32; dequant_grid = (int)(n_weight_elems / 256); break; // IQ3_S
            case 19: dequant_fn = k.dequant_f16_iq1_s;   dequant_bs = 32; dequant_grid = (int)(n_weight_elems / 256); break; // IQ1_S
            case 29: dequant_fn = k.dequant_f16_iq1_m;   dequant_bs = 32; dequant_grid = (int)(n_weight_elems / 256); break; // IQ1_M
            case 20: dequant_fn = k.dequant_f16_iq4_nl;  dequant_bs = 32; dequant_grid = (int)((n_weight_elems + 255) / 256); break; // IQ4_NL
            case 23: dequant_fn = k.dequant_f16_iq4_xs;  dequant_bs = 32; dequant_grid = (int)((n_weight_elems + 255) / 256); break; // IQ4_XS
            // --- MXFP4/NVFP4: 32 threads ---
            case 39: dequant_fn = k.dequant_f16_mxfp4;   dequant_bs = 32; dequant_grid = (int)((n_weight_elems + 255) / 256); break; // MXFP4
            case 40: dequant_fn = k.dequant_f16_nvfp4;   dequant_bs = 32; dequant_grid = (int)(n_weight_elems / 64);          break; // NVFP4
            // --- Float types: 256 threads, simple element-wise ---
            case  0: dequant_fn = k.dequant_f16_f32;     dequant_bs = 256; dequant_grid = (int)((n_weight_elems + 255) / 256); break; // F32
            case 30: dequant_fn = k.dequant_f16_bf16;    dequant_bs = 256; dequant_grid = (int)((n_weight_elems + 255) / 256); break; // BF16
            case  1: dequant_fn = nullptr; break; // F16: no dequant needed, used directly
            default:
                fprintf(stderr, "gfx1100: no dequant-to-F16 kernel for type %d\n", weight_type);
                break;
        }

        rocblas_gemm_exec(&g_rocblas, k.gemm_f32_to_f16, dequant_fn, dequant_grid, dequant_bs,
                          weight, weight_type, n_weight_elems,
                          f32_input, in_dim, out_dim, batch_size, output, stream);
    } else {
        fprintf(stderr, "gfx1100: ERROR — no rocBLAS and MMQ failed for type %d\n", weight_type);
    }
}

// Helper: launch batch projection with residual add
// output[i] = proj_output[i] + residual[i]
void batch_projection_residual(
        int weight_type, const void * weight, long long weight_stride,
        const float * f32_input, const void * q8_input,
        const float * residual, float * output, int in_dim, int out_dim, int batch_size,
        hipStream_t stream) {

    // Project into output first
    batch_projection(weight_type, weight, weight_stride, f32_input, q8_input,
                     output, in_dim, out_dim, batch_size, stream);

    // Then add residual: output += residual (element-wise, S*out_dim elements)
    auto & k = g_compiled;
    int N = batch_size * out_dim;
    void * args[] = { (void *)&output, (void *)&residual, (void *)&output, (void *)&N };
    hipModuleLaunchKernel(k.prompt_add_residual, (N + 255) / 256, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}
