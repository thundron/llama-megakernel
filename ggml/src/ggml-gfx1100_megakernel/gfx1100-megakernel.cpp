// gfx1100-megakernel.cpp — dispatch switch + exports + backend registration (slim)
// All forward functions are in forward/*.cpp
// Init/compile/load is in gfx1100-init.cpp
// Structs/globals are in gfx1100-internal.h
#include "gfx1100-internal.h"
#include "ggml-backend-impl.h"

int gfx1100_eval_decode(int token_id, int position, float * logits_out) {
    assert(g_initialized && g_compiled.valid);
    const int arch = g_config.arch_id;

    switch (arch) {
        // =================================================================
        // Dense transformer: sequential RMSNorm/LayerNorm → QKV → RoPE →
        // attention → O proj → residual → norm → FFN → residual.
        // forward_decode_llama_family handles:
        //   - RMSNorm OR LayerNorm (dispatched by norm_type)
        //   - NORM/NEOX/MROPE RoPE (compile-time #if ROPE_TYPE)
        //   - QKV biases + LoRA scales (runtime NULL checks)
        //   - QK norm (compile-time HAS_QK_NORM)
        //   - NORM_ADD_ONE (Gemma, compile-time)
        //   - Custom attention scale (compile-time)
        //   - Gated attention (compile-time FA_HAS_GATED_ATTN)
        //   - All activation types via ACT_TYPE
        //   - Per-layer DeltaNet (layer_types[il]==1)
        // =================================================================

        // --- RMSNorm + ROPE_NORM archs ---
        case ARCH_LLAMA: case ARCH_LLAMA_EMBED: case ARCH_LLAMA4:
        case ARCH_MISTRAL3: case ARCH_MISTRAL4:
        case ARCH_QWEN35: // hybrid: per-layer DN; attention layers are NORM-RoPE
        case ARCH_BAICHUAN:
        case ARCH_INTERNLM2: case ARCH_MINICPM:
        case ARCH_DECI: case ARCH_DREAM:
        case ARCH_APERTUS:
        case ARCH_ERNIE4_5: case ARCH_GLM4:
        case ARCH_SEED_OSS: case ARCH_HUNYUAN_DENSE:
        case ARCH_PANGU_EMBED: case ARCH_MAINCODER:
        case ARCH_SMOLLM3: case ARCH_EXAONE:
        case ARCH_GLM_DSA: case ARCH_LLADA:
        case ARCH_PLAMO: case ARCH_PLM:
        case ARCH_REFACT:
        case ARCH_XVERSE:
        case ARCH_CHATGLM:
        case ARCH_OPENELM:
            // WARNING: OpenELM uses per-layer n_head (different head counts per layer).
            // The megakernel uses compile-time fixed FA_N_Q_HEADS/FA_N_KV_HEADS.
            // OpenELM will use layer 0's head count for ALL layers — wrong for layers with different counts.
            // Per-layer head counts handled via per_layer_n_q_heads[]/per_layer_n_kv_heads[].
        case ARCH_GEMMA: // uses NORM_ADD_ONE + custom scale, handled by compile flags
            return forward_decode_llama_family(token_id, position, logits_out);

        // --- RMSNorm + ROPE_NEOX archs ---
        case ARCH_QWEN: case ARCH_QWEN2: case ARCH_QWEN2VL:
        case ARCH_QWEN3: case ARCH_QWEN3VL:
        case ARCH_OLMO2: case ARCH_STABLELM:
            return forward_decode_llama_family(token_id, position, logits_out);

        // --- LayerNorm archs (sequential attn+FFN, no ALiBi, no parallel) ---
        // These use eval_layernorm via norm_type dispatch in launch_norm()
        case ARCH_ORION: case ARCH_COMMAND_R:
        case ARCH_CODESHELL:
        case ARCH_NEMOTRON: case ARCH_OLMO:
        case ARCH_STARCODER2:
        case ARCH_CHAMELEON: case ARCH_GRANITE:
        case ARCH_ARCEE: // uses ReLU² but ACT_TYPE handles it
            return forward_decode_llama_family(token_id, position, logits_out);

        // =================================================================
        // Architectures that need features beyond forward_decode_llama_family.
        // Each will get its own forward_decode_* function.
        // Until ported: FATAL with instruction message.
        // =================================================================

        // --- MoE archs (standard attn + MoE FFN, no SWA/SSM) ---
        // These use the same attention pattern as Llama but with MoE FFN.
        // forward_decode_llama_family now handles MoE via the ffn_gate_inp dispatch.
        case ARCH_QWEN2MOE: case ARCH_QWEN3MOE:
        case ARCH_GROK: case ARCH_DBRX:
        case ARCH_OLMOE:
        case ARCH_DEEPSEEK:
        case ARCH_AFMOE: case ARCH_DOTS1: case ARCH_GROVEMOE:
        case ARCH_GLM4_MOE:
        case ARCH_BAILINGMOE: case ARCH_BAILINGMOE2:
        case ARCH_HUNYUAN_MOE: case ARCH_MINIMAX_M2:
        case ARCH_ERNIE4_5_MOE: case ARCH_LLADA_MOE:
        case ARCH_RND1: case ARCH_EXAONE_MOE:
        case ARCH_PHIMOE:
            return forward_decode_llama_family(token_id, position, logits_out);

        // --- MoE + SWA archs (forward_decode_llama_family handles both) ---
        // Per-layer SWA via layer_use_swa[], MoE via ffn_gate_inp dispatch
        case ARCH_QWEN3NEXT: case ARCH_QWEN35MOE: case ARCH_QWEN3VLMOE:
        case ARCH_GRANITE_MOE:
        case ARCH_OPENAI_MOE: case ARCH_SMALLTHINKER:
        case ARCH_MIMO2: case ARCH_STEP35:
        case ARCH_JAIS2:
        case ARCH_ARCTIC:
            return forward_decode_llama_family(token_id, position, logits_out);

        // --- Sliding window archs (iSWA: some layers full context, some windowed) ---
        // Basic SWA (clamp kv_len) is now in the attention launch.
        // All SWA features implemented: per-layer SWA flag, KV cache offset,
        // softcap (compile-time ATTN_SOFTCAP_VAL), post-attn/FFN norms
        case ARCH_GEMMA2: case ARCH_GEMMA3: case ARCH_GEMMA3N:
        case ARCH_GEMMA4: case ARCH_COHERE2:
        case ARCH_EXAONE4: case ARCH_PLAMO3:
            return forward_decode_llama_family(token_id, position, logits_out);
        case ARCH_LFM2: case ARCH_LFM2MOE: // SWA + SSM hybrid
            // SSM layers via layer_types[il]==2 (ssm_layer_step)
            // SWA layers via layer_use_swa[il] with KV cache offset
            // Both handled by forward_decode_llama_family
            return forward_decode_llama_family(token_id, position, logits_out);

        // --- Pure SSM/Mamba archs ---
        case ARCH_MAMBA:
            return forward_decode_mamba(token_id, position, logits_out);
        case ARCH_MAMBA2:
        case ARCH_PLAMO2:
            return forward_decode_mamba2(token_id, position, logits_out);

        // --- Hybrid SSM+attention archs ---
        // SSM layers handled by ssm_layer_step() via layer_types[il]==2 dispatch
        case ARCH_JAMBA: case ARCH_FALCON_H1:
        case ARCH_NEMOTRON_H: case ARCH_NEMOTRON_H_MOE:
        case ARCH_GRANITE_HYBRID: case ARCH_KIMI_LINEAR:
            return forward_decode_llama_family(token_id, position, logits_out);

        // --- RWKV archs ---
        case ARCH_RWKV6: case ARCH_RWKV6QWEN2:
            return forward_decode_rwkv6(token_id, position, logits_out);
        case ARCH_RWKV7: case ARCH_ARWKV7:
            return forward_decode_rwkv7(token_id, position, logits_out);

        // --- ALiBi archs (LayerNorm + ALiBi slopes in attention, no RoPE) ---
        // ALiBi is now handled in eval_attention_decode via alibi_max_bias params.
        // These archs use LayerNorm (dispatched by norm_type) + ALiBi (no RoPE).
        // Note: BLOOM and FALCON may use parallel attn+FFN for some variants.
        case ARCH_BLOOM: case ARCH_JAIS:
        case ARCH_MPT:
            return forward_decode_llama_family(token_id, position, logits_out);

        // --- ALiBi + parallel attn+FFN ---
        // use_par_res flag handles the parallel residual pattern in forward_decode_llama_family
        case ARCH_FALCON: case ARCH_GPTNEOX:
            return forward_decode_llama_family(token_id, position, logits_out);

        // --- Legacy no-RoPE archs (ROPE_TYPE==0 compiles to skip rotation) ---
        // Positional info comes from learned embeddings added to input (GPT2/GPTJ/StarCoder)
        // or ALiBi (handled above). The QK-norm-RoPE kernel with ROPE_TYPE=0 just writes
        // Q/K to cache without rotation.
        // Note: GPT2 uses LayerNorm; GPTJ uses parallel attn+FFN
        case ARCH_GPT2: case ARCH_STARCODER:
            return forward_decode_llama_family(token_id, position, logits_out);
        case ARCH_GPTJ:
            return forward_decode_llama_family(token_id, position, logits_out);

        // --- Encoder-decoder ---
        case ARCH_T5:
            return forward_decode_t5(token_id, position, logits_out);
        case ARCH_T5ENCODER:
        case ARCH_BERT: case ARCH_MODERN_BERT:
        case ARCH_NOMIC_BERT: case ARCH_NOMIC_BERT_MOE:
        case ARCH_JINA_BERT_V2: case ARCH_JINA_BERT_V3:
        case ARCH_EUROBERT: case ARCH_NEO_BERT:
        case ARCH_GEMMA_EMBEDDING:
            fprintf(stderr, "gfx1100: arch %d is encoder-only — use gfx1100_eval_prompt() for encoding\n", arch);
            return -1;

        // --- DeepSeek2 MLA (and MiniCPM3 which uses the same MLA architecture) ---
        case ARCH_DEEPSEEK2: case ARCH_DEEPSEEK2OCR:
        case ARCH_MINICPM3: // MiniCPM3 uses MLA: wq_a/wq_b/wkv_a/wkv_b per baseline minicpm3.cpp
            return forward_decode_deepseek2_mla(token_id, position, logits_out);

        // --- Special archs ---
        case ARCH_BITNET:
            return forward_decode_bitnet(token_id, position, logits_out);
        case ARCH_PHI2:
            // Phi2 uses parallel attn+FFN (same as GPT-NeoX pattern) + LayerNorm + GELU
            // The parallel pattern is handled by use_par_res flag in forward_decode_llama_family
            return forward_decode_llama_family(token_id, position, logits_out);
        case ARCH_PHI3:
            // Phi3 may use MoE — check at runtime via ffn_gate_inp
            return forward_decode_llama_family(token_id, position, logits_out);
        case ARCH_COGVLM:
            // CogVLM decoder: fused QKV + dual weight sets (text/visual expert)
            return forward_decode_cogvlm(token_id, position, logits_out);
        case ARCH_PADDLEOCR:
            // PaddleOCR: identical to Qwen2VL decoder with optional biases
            return forward_decode_llama_family(token_id, position, logits_out);
        case ARCH_WAVTOKENIZER_DEC:
            // WavTokenizer: convolutional audio decoder (not autoregressive)
            // Single-token decode doesn't apply — use eval_prompt path
            fprintf(stderr, "gfx1100: WavTokenizer is a batch audio decoder — use gfx1100_eval_prompt()\n");
            return -1;

        default:
            fprintf(stderr,
                "gfx1100: FATAL — arch %d has no decode dispatcher yet. "
                "Port the baseline src/models/<arch>.cpp forward graph and add a "
                "case to gfx1100_eval_decode's switch.\n", arch);
            return -1;
    }
}

// ============================================================================
// Query functions
// ============================================================================

int gfx1100_is_available(void) {
    int n = 0;
    hipGetDeviceCount(&n);
    for (int i = 0; i < n; i++) {
        hipDeviceProp_t p;
        if (hipGetDeviceProperties(&p, i) == hipSuccess && strstr(p.gcnArchName, "gfx1100"))
            return 1;
    }
    return 0;
}

int gfx1100_is_ready(void) {
    return g_initialized && g_compiled.valid ? 1 : 0;
}

int gfx1100_copy_logits(float * host, int n_vocab) {
    if (!g_initialized) return -1;
    hipMemcpyAsync(host, g_bufs.logits, n_vocab * sizeof(float),
                   hipMemcpyDeviceToHost, g_bufs.stream);
    hipStreamSynchronize(g_bufs.stream);
    return 0;
}

// ============================================================================
// Prompt eval: batch GEMM via MMQ + rocBLAS engines
// Small batches (<= 8 tokens) fall back to sequential decode.
// ============================================================================

// Helper: launch MMQ batch quantize kernel (from decode.hip)
// grid=(ne1, ceil(ne0/(4*128)), ne2*ne3), block=(128,1,1)
// ne00=ne0=in_dim (padded to 4*128 alignment), ne1=batch_size, ne2=1
// s01=in_dim (stride in floats between rows), s02=s03=0
static void launch_mmq_quantize(hipFunction_t fn, const float * input, void * output,
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
static void launch_mmq_kernel(hipFunction_t fn, const void * weight, long long weight_stride,
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
static void batch_projection(
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
static void batch_projection_residual(
        int weight_type, const void * weight, long long weight_stride,
        const float * f32_input, const void * q8_input,
        float * residual, float * output, int in_dim, int out_dim, int batch_size,
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

// ============================================================================
// Speculative decoding support
// ============================================================================

int gfx1100_eval_verify(const int * token_ids, int n_tokens, int start_pos, float * all_logits_out) {
    if (!g_initialized || !g_compiled.valid) return -1;

    // Speculative verification: evaluate n_tokens and produce logits at EVERY position.
    // This is used by the target model to verify draft tokens.
    //
    // Implementation: for each token, run sequential decode and copy logits to the
    // appropriate row of all_logits_out. This is the simplest correct approach.
    // A batch implementation would be faster but requires the batch path to support
    // multi-position logit output.
    //
    // Baseline does this via a single llama_decode call with batch.logits[i]=true for all i.
    // Our megakernel's prompt path only outputs logits for the last token.
    // For now: sequential decode for each token, copying logits at each step.
    int V = g_config.vocab_size;
    for (int i = 0; i < n_tokens; i++) {
        int pos = start_pos + i;
        float * logits_row = all_logits_out + (size_t)i * V;
        int rc = gfx1100_eval_decode(token_ids[i], pos, logits_row);
        if (rc != 0) {
            fprintf(stderr, "gfx1100: verify failed at token %d (pos=%d)\n", i, pos);
            return rc;
        }
    }
    return 0;
}

int gfx1100_eval_decode_batch(
    const int * token_ids, const int * positions, int n_tokens, float * logits_out) {
    if (!g_initialized || n_tokens <= 0) return -1;

    // Single token: use the optimized decode path
    if (n_tokens == 1) {
        return gfx1100_eval_decode(token_ids[0], positions[0], logits_out);
    }

    // Multi-token: use the batch prompt path with start_pos = positions[0]
    // This handles sequential positions [start_pos, start_pos + n_tokens - 1]
    // For the common case of continuation decode, positions are sequential.
    int start_pos = positions[0];

    // Verify positions are sequential (single-sequence continuation)
    bool sequential = true;
    for (int i = 1; i < n_tokens; i++) {
        if (positions[i] != start_pos + i) { sequential = false; break; }
    }

    if (sequential) {
        // Use the batch prompt path directly — it handles start_pos > 0 now
        return gfx1100_eval_prompt(token_ids, n_tokens, start_pos, logits_out);
    }

    // Non-sequential positions (multi-user or speculative): fall back to per-token decode
    // Each token gets its own decode call with its own position
    for (int i = 0; i < n_tokens; i++) {
        float * out = (i == n_tokens - 1) ? logits_out : nullptr;
        int rc = gfx1100_eval_decode(token_ids[i], positions[i], out);
        if (rc != 0) return rc;
    }
    return 0;
}

int gfx1100_sample_token(
    float temperature, int top_k, float top_p, float repetition_penalty,
    const int * penalty_tokens, int n_penalty, float random_val, int * out_token) {
    if (!g_initialized) return -1;
    auto & b = g_bufs;
    auto & k = g_compiled;
    auto   s = b.stream;
    int    V = g_config.vocab_size;

    // Step 1: Temperature scaling (skip if temp == 0 for greedy)
    if (temperature > 0.0f && temperature != 1.0f) {
        float inv_t = 1.0f / temperature;
        void * args[] = { (void *)&b.logits, (void *)&V, (void *)&inv_t };
        hipModuleLaunchKernel(k.eval_sample_temperature, (V+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    }

    // Step 2: Repetition penalty
    if (repetition_penalty != 1.0f && penalty_tokens && n_penalty > 0) {
        void * args[] = { (void *)&b.logits, (void *)&penalty_tokens, (void *)&n_penalty, (void *)&repetition_penalty };
        hipModuleLaunchKernel(k.eval_sample_rep_penalty, (n_penalty+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    }

    // Step 3: Greedy (temperature == 0) or stochastic sampling
    if (temperature <= 0.0f) {
        // Greedy: argmax
        void * args[] = { (void *)&b.logits, (void *)&out_token, (void *)&V };
        hipModuleLaunchKernel(k.eval_sample_argmax, 1, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    } else {
        // Top-K + Top-P + categorical
        void * args[] = { (void *)&b.logits, (void *)&out_token, (void *)&V,
                          (void *)&top_k, (void *)&top_p, (void *)&random_val };
        hipModuleLaunchKernel(k.eval_sample_top_k_p, 1, 1, 1, 1, 1, 1, 0, s, args, nullptr);
    }

    hipStreamSynchronize(s);
    return 0;
}

int gfx1100_preprocess_image(
    const float * image_rgb, int height, int width,
    float * patch_embeddings, int * n_patches) {
    if (!g_initialized) return -1;
    // Baseline vision: patch_size is model-dependent (typically 14 for ViT, 16 for CLIP)
    // For now: use 14 as default. The model's patch_size should be in the config.
    int patch_size = (g_config.vision_patch_size > 0) ? g_config.vision_patch_size : 14;
    int npx = width / patch_size;
    int npy = height / patch_size;
    int total_patches = npx * npy;
    *n_patches = total_patches;

    int patch_dim = patch_size * patch_size * 3;
    int threads = (patch_dim < 256) ? patch_dim : 256;
    void * args[] = { (void *)&image_rgb, (void *)&patch_embeddings,
                      (void *)&height, (void *)&width, (void *)&patch_size,
                      (void *)&npx, (void *)&npy };
    hipModuleLaunchKernel(g_compiled.eval_image_patches, total_patches, 1, 1,
                         threads, 1, 1, 0, g_bufs.stream, args, nullptr);
    hipStreamSynchronize(g_bufs.stream);
    return 0;
}

int gfx1100_preprocess_audio(
    const float * audio_samples, int n_samples, int sample_rate,
    float * mel_features, int * n_frames) {
    if (!g_initialized) return -1;
    (void)sample_rate; // sample_rate is implicit in n_fft/hop from model

    auto & c = g_config;
    auto & b = g_bufs;
    auto & k = g_compiled;
    auto   s = b.stream;

    // Audio params from model (populated during init from GGUF)
    int n_fft  = c.audio_n_fft;
    int hop    = c.audio_hop_length;
    int n_mels = c.audio_n_mels;

    if (n_fft <= 0 || hop <= 0 || n_mels <= 0 || !c.mel_filters) {
        fprintf(stderr, "gfx1100: audio preprocessing not configured (no mel filters in model)\n");
        return -1;
    }

    int frames = n_samples / hop + 1;
    int n_fft_half = n_fft / 2 + 1;
    *n_frames = frames;

    // Allocate STFT scratch if needed
    if (frames > b.max_audio_frames || !b.d_stft_power) {
        if (b.d_stft_power) hipFree(b.d_stft_power);
        hipMalloc(&b.d_stft_power, (size_t)frames * n_fft_half * sizeof(float));
        b.max_audio_frames = frames;
        if (!b.d_mel_scratch) hipMalloc(&b.d_mel_scratch, sizeof(float));
    }

    // Phase 1: STFT power spectrum via brute-force DFT
    // Grid: (frames, 1, 1), Block: (256, 1, 1)
    {
        void * args[] = { (void *)&audio_samples, (void *)&b.d_stft_power,
                          (void *)&n_samples, (void *)&n_fft, (void *)&hop };
        hipModuleLaunchKernel(k.eval_stft_power, frames, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    }

    // Phase 2: Mel filterbank + log
    // Grid: (frames, n_mels, 1), Block: (256, 1, 1)
    {
        void * args[] = { (void *)&b.d_stft_power, (void *)&c.mel_filters,
                          (void *)&mel_features,
                          (void *)&frames, (void *)&n_fft_half, (void *)&n_mels };
        hipModuleLaunchKernel(k.eval_mel_filterbank, frames, n_mels, 1, 256, 1, 1, 0, s, args, nullptr);
    }

    // Phase 3: Normalize — find max, then clamp to max-8 and scale
    int total = n_mels * frames;
    {
        void * args[] = { (void *)&mel_features, (void *)&b.d_mel_scratch, (void *)&total };
        hipModuleLaunchKernel(k.eval_max_reduce, 1, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    }
    {
        void * args[] = { (void *)&mel_features, (void *)&b.d_mel_scratch, (void *)&total };
        hipModuleLaunchKernel(k.eval_mel_normalize, (total+255)/256, 1, 1, 256, 1, 1, 0, s, args, nullptr);
    }

    hipStreamSynchronize(s);
    return 0;
}

int gfx1100_kv_cache_rollback(int p0) {
    if (!g_initialized) return -1;
    // No GPU op needed — attention kernel only reads positions 0..kv_len-1.
    // Caller tracks current position.
    (void)p0;
    return 0;
}

// KV cache sequence shift — ported from baseline llama_kv_cache_seq_add
// Shifts KV cache data within [p0, p1) by delta positions.
// Used for context window sliding: after removing old tokens, shift remaining tokens left.
// All on GPU — hipMemcpy device-to-device within the cache.
int gfx1100_kv_cache_seq_shift(int p0, int p1, int delta) {
    if (!g_initialized) return -1;
    auto & c = g_config;
    auto & b = g_bufs;

    int hd = c.fa_head_dim;
    int max_seq = c.max_seq_len;
    int n_kv = c.fa_n_kv_heads;
    size_t row_bytes = hd * sizeof(__half);

    // For each KV head, shift positions [p0, p1) by delta within the cache
    int n_attn = 0;
    for (int i = 0; i < c.n_layers; i++) if (c.layer_types[i] == 0) n_attn++;

    for (int il = 0; il < n_attn; il++) {
        __half * kc = (__half *)c.k_cache_ptrs[il];
        __half * vc = (__half *)c.v_cache_ptrs[il];
        for (int h = 0; h < n_kv; h++) {
            __half * k_head = kc + h * max_seq * hd;
            __half * v_head = vc + h * max_seq * hd;
            // Copy [p0, p1) to [p0+delta, p1+delta) — or the reverse direction
            int dst_start = p0 + delta;
            int src_start = p0;
            int count = p1 - p0;
            if (dst_start >= 0 && dst_start + count <= max_seq) {
                hipMemcpyAsync(k_head + dst_start * hd, k_head + src_start * hd,
                               count * row_bytes, hipMemcpyDeviceToDevice, b.stream);
                hipMemcpyAsync(v_head + dst_start * hd, v_head + src_start * hd,
                               count * row_bytes, hipMemcpyDeviceToDevice, b.stream);
            }
        }
    }
    hipStreamSynchronize(b.stream);
    return 0;
}

// KV cache sequence remove — ported from baseline llama_kv_cache_seq_rm
// Removes positions [p0, p1) by shifting [p1, end) left by (p1-p0).
int gfx1100_kv_cache_seq_rm(int p0, int p1, int kv_len) {
    if (!g_initialized) return -1;
    // Shift [p1, kv_len) left by (p1-p0) positions
    return gfx1100_kv_cache_seq_shift(p1, kv_len, -(p1 - p0));
}

// KV cache sequence copy — copy positions [p0, p1) from current cache to dest_offset
// Used for beam search: duplicate a sequence's KV cache for a new beam.
int gfx1100_kv_cache_seq_cp(int p0, int p1, int dest_offset) {
    if (!g_initialized) return -1;
    auto & c = g_config;
    auto & b = g_bufs;

    int hd = c.fa_head_dim;
    int max_seq = c.max_seq_len;
    int n_kv = c.fa_n_kv_heads;
    size_t row_bytes = hd * sizeof(__half);

    int n_attn = 0;
    for (int i = 0; i < c.n_layers; i++) if (c.layer_types[i] == 0) n_attn++;

    for (int il = 0; il < n_attn; il++) {
        __half * kc = (__half *)c.k_cache_ptrs[il];
        __half * vc = (__half *)c.v_cache_ptrs[il];
        for (int h = 0; h < n_kv; h++) {
            __half * k_head = kc + h * max_seq * hd;
            __half * v_head = vc + h * max_seq * hd;
            int count = p1 - p0;
            hipMemcpyAsync(k_head + dest_offset * hd, k_head + p0 * hd,
                           count * row_bytes, hipMemcpyDeviceToDevice, b.stream);
            hipMemcpyAsync(v_head + dest_offset * hd, v_head + p0 * hd,
                           count * row_bytes, hipMemcpyDeviceToDevice, b.stream);
        }
    }
    hipStreamSynchronize(b.stream);
    return 0;
}

// =============================================================================
// ggml backend registration — exposes megakernel functions via proc_address
// =============================================================================

static const char * gfx1100_mk_reg_get_name(ggml_backend_reg_t) {
    return "gfx1100-megakernel";
}

static size_t gfx1100_mk_reg_get_device_count(ggml_backend_reg_t) {
    return 0; // megakernel doesn't expose devices via the standard ggml backend API
}

static ggml_backend_dev_t gfx1100_mk_reg_get_device(ggml_backend_reg_t, size_t) {
    return nullptr;
}

static void * gfx1100_mk_reg_get_proc_address(ggml_backend_reg_t, const char * name) {
    if (strcmp(name, "gfx1100_is_available")       == 0) return (void *)(intptr_t)gfx1100_is_available;
    if (strcmp(name, "gfx1100_is_ready")           == 0) return (void *)(intptr_t)gfx1100_is_ready;
    if (strcmp(name, "gfx1100_init")               == 0) return (void *)(intptr_t)gfx1100_init;
    if (strcmp(name, "gfx1100_eval_decode")         == 0) return (void *)(intptr_t)gfx1100_eval_decode;
    if (strcmp(name, "gfx1100_eval_prompt")         == 0) return (void *)(intptr_t)gfx1100_eval_prompt;
    // gfx1100_load_model is in gfx1100-model-loader.cpp (standalone GGUF loader, not linked into shared lib)
    if (strcmp(name, "gfx1100_preprocess_image")    == 0) return (void *)(intptr_t)gfx1100_preprocess_image;
    if (strcmp(name, "gfx1100_preprocess_audio")    == 0) return (void *)(intptr_t)gfx1100_preprocess_audio;
    if (strcmp(name, "gfx1100_kv_cache_seq_shift")  == 0) return (void *)(intptr_t)gfx1100_kv_cache_seq_shift;
    if (strcmp(name, "gfx1100_kv_cache_seq_rm")     == 0) return (void *)(intptr_t)gfx1100_kv_cache_seq_rm;
    if (strcmp(name, "gfx1100_kv_cache_seq_cp")     == 0) return (void *)(intptr_t)gfx1100_kv_cache_seq_cp;
    if (strcmp(name, "gfx1100_kv_cache_rollback")   == 0) return (void *)(intptr_t)gfx1100_kv_cache_rollback;
    if (strcmp(name, "gfx1100_sample_token")        == 0) return (void *)(intptr_t)gfx1100_sample_token;
    return nullptr;
}

ggml_backend_reg_t ggml_backend_gfx1100_megakernel_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ {
            /* .get_name         = */ gfx1100_mk_reg_get_name,
            /* .get_device_count = */ gfx1100_mk_reg_get_device_count,
            /* .get_device       = */ gfx1100_mk_reg_get_device,
            /* .get_proc_address = */ gfx1100_mk_reg_get_proc_address,
        },
        /* .context     = */ nullptr,
    };
    return &reg;
}

