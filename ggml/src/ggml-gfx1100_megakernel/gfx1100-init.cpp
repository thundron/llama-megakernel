// gfx1100-init.cpp — compile, load kernels, allocate buffers, populate config
//
// Undo the hipModuleLaunchKernel redirect for this file — we need the real
// API for the HSACO fallback path in gfx1100_dispatch (defined inline in the
// header, before the macro). This file never calls hipModuleLaunchKernel
// directly, so the undef is purely defensive.
#include "gfx1100-internal.h"
#undef hipModuleLaunchKernel

// Composition meta-steps: DETECT (tensor capability scan) + VALIDATE (support check)
// Always run during gfx1100_init. Env vars control printing only:
//   GFX1100_COMPOSITION_DIAG=1 — print diagnostics and continue as normal
//   GFX1100_DIAG_ONLY=1        — print diagnostics and exit(0) before GPU init
#include "composition/comp-types.h"
#include "composition/comp-detect.h"
#include "composition/comp-validate.h"
#include "composition/comp-tune.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// Global definitions (declared extern in gfx1100-internal.h)
gfx1100_model_config g_config = {};
gfx1100_compiled     g_compiled = {};
gfx1100_buffers      g_bufs = {};
bool                  g_initialized = false;
std::string           g_rocm_device_lib_path;  // auto-detected from hipcc location
rocblas_gemm_state    g_rocblas = {};

// Composition meta-step 1+2 results. Populated unconditionally in gfx1100_init.
// Downstream code may read these globals to drive compile flags, dispatch
// decisions, buffer sizing, etc. Nothing consumes them yet — this is pure
// signal-making; wiring is added meta-step by meta-step.
comp_capabilities        g_comp_caps = {};
comp_validation_result   g_comp_validation = {};
comp_tuning              g_comp_tuning = {};

// DLL dispatch globals (declared extern in gfx1100-internal.h)
bool                   g_dll_mode = false;
gfx1100_dll_launch_fn  g_dll_launch = nullptr;
static void *          g_dll_handle = nullptr;
static gfx1100_get_kernel_fn g_get_kernel = nullptr;

// ============================================================================
// MMQ decision logic (from baseline mmq.cu lines 265-369, RDNA3 path)
// ============================================================================

// RDNA3: use MMQ for almost everything except Q2_K/Q6_K at large batch
bool should_use_mmq(int type, int batch_size) {
    switch (type) {
        case 10: return batch_size <= 128; // Q2_K
        case 14: return batch_size <= 128; // Q6_K
        default: return true;
    }
}

// Maps weight type → Q8_1 MMQ quantize layout (host-side mirror of mmq_get_q8_1_ds_layout)
// Returns 0=D4, 1=DS4, 2=D2S6
int get_mmq_q8_1_layout(int weight_type) {
    switch (weight_type) {
        case  2:                   // Q4_0
        case  3:                   // Q4_1
            return 1; // DS4
        case  6:                   // Q5_0
        case  8:                   // Q8_0
        case 39:                   // MXFP4
        case 40:                   // NVFP4
            return 0; // D4
        case  7:                   // Q5_1
            return 1; // DS4
        case 10:                   // Q2_K
            return 2; // D2S6
        case 11:                   // Q3_K
            return 0; // D4
        case 12: case 13:         // Q4_K, Q5_K
            return 1; // DS4
        case 14:                   // Q6_K
        case 16: case 17: case 22: // IQ2_XXS, IQ2_XS, IQ2_S
        case 18: case 21:         // IQ3_XXS, IQ3_S
        case 20: case 23:         // IQ4_NL, IQ4_XS
            return 0; // D4
        case 19:                   // IQ1_S
            return 1; // DS4
        default: return 0;
    }
}

// MMQ type table: maps ggml_type integer to table index (0..19)
// Parallel array of short names for kernel symbol lookup.
const mmq_type_entry mmq_type_table[] = {
    {  2, "q4_0"    },  //  0
    {  3, "q4_1"    },  //  1
    {  6, "q5_0"    },  //  2
    {  7, "q5_1"    },  //  3
    {  8, "q8_0"    },  //  4
    { 39, "mxfp4"   },  //  5
    { 40, "nvfp4"   },  //  6
    { 10, "q2k"     },  //  7
    { 11, "q3k"     },  //  8
    { 12, "q4k"     },  //  9
    { 13, "q5k"     },  // 10
    { 14, "q6k"     },  // 11
    { 16, "iq2_xxs" },  // 12
    { 17, "iq2_xs"  },  // 13
    { 22, "iq2_s"   },  // 14
    { 18, "iq3_xxs" },  // 15
    { 21, "iq3_s"   },  // 16
    { 19, "iq1_s"   },  // 17
    { 20, "iq4_nl"  },  // 18
    { 23, "iq4_xs"  },  // 19
};
const int MMQ_NUM_TYPES = sizeof(mmq_type_table) / sizeof(mmq_type_table[0]);

// Find MMQ type table index for a given ggml_type, or -1 if not supported
int mmq_type_index(int ggml_type) {
    for (int i = 0; i < MMQ_NUM_TYPES; i++) {
        if (mmq_type_table[i].ggml_type == ggml_type) return i;
    }
    return -1;
}

// Shared memory size for MMQ kernel (host-side calculation)
// From mmq.h mmq_get_nbytes_shared — MMA formula
// mmq_y=128, warp_size=32, nwarps=8 (RDNA3 constants)
size_t mmq_shared_mem_size(int ggml_type, int mmq_x) {
    // We need mmq_get_mma_tile_x_k(type) but that's a device constexpr.
    // Mirror the switch from mmq-tiles.h lines 220-241 on the host.
    int tile_x_k;
    switch (ggml_type) {
        // Types that use Q8_0 tile layout (tile_x_k = value from MMQ_MMA_TILE_X_K_Q8_0)
        // From mmq-tiles.h: MMQ_MMA_TILE_X_K_Q8_0 = (QK8_0*4 + 8*sizeof(float)) / sizeof(int) = (128+32)/4 = 40
        case  2: tile_x_k = 40; break; // Q4_0
        case  6: tile_x_k = 40; break; // Q5_0
        case  8: tile_x_k = 40; break; // Q8_0
        case 16: tile_x_k = 40; break; // IQ2_XXS
        case 18: tile_x_k = 40; break; // IQ3_XXS
        case 21: tile_x_k = 40; break; // IQ3_S
        case 19: tile_x_k = 40; break; // IQ1_S
        case 23: tile_x_k = 40; break; // IQ4_XS
        case 20: tile_x_k = 40; break; // IQ4_NL
        // Types that use Q8_1 tile layout (tile_x_k = value from MMQ_MMA_TILE_X_K_Q8_1)
        // MMQ_MMA_TILE_X_K_Q8_1 = (QK8_1*4 + 8*sizeof(half2)) / sizeof(int) = (128+32)/4 = 40
        case  3: tile_x_k = 40; break; // Q4_1
        case  7: tile_x_k = 40; break; // Q5_1
        case 39: tile_x_k = 40; break; // MXFP4
        case 12: tile_x_k = 40; break; // Q4_K
        case 13: tile_x_k = 40; break; // Q5_K
        // NVFP4: MMQ_MMA_TILE_X_K_NVFP4 — different structure
        // From mmq-tiles.h: MMQ_MMA_TILE_X_K_NVFP4 = specific value
        case 40: tile_x_k = 40; break; // NVFP4 (same 40 — simplified)
        // Q2_K: MMQ_MMA_TILE_X_K_Q2_K
        case 10: tile_x_k = 40; break; // Q2_K
        // Q3_K: MMQ_MMA_TILE_X_K_Q3_K
        case 11: tile_x_k = 40; break; // Q3_K
        // Q6_K: MMQ_MMA_TILE_X_K_Q6_K
        case 14: tile_x_k = 40; break; // Q6_K
        // IQ2_XS, IQ2_S: use Q3_K tile
        case 17: tile_x_k = 40; break; // IQ2_XS
        case 22: tile_x_k = 40; break; // IQ2_S
        default: tile_x_k = 40; break;
    }
    const int mmq_y = 128;
    const int warp_size = 32;
    const int nwarps = 8;
    // block_q8_1_mmq size = 4*QK8_1 + 4*sizeof(half2) = 128 + 16 = 144 bytes
    const int block_q8_1_mmq_size = 144;
    // Formula from mmq.h mmq_get_nbytes_shared:
    //   nbs_ids = mmq_x * sizeof(int)
    //   nbs_x = mmq_y * tile_x_k * sizeof(int)
    //   nbs_y = mmq_x * block_q8_1_mmq_size
    //   total = nbs_ids + nbs_x + GGML_PAD(nbs_y, nwarps*warp_size*sizeof(int))
    size_t nbs_ids = mmq_x * sizeof(int);
    size_t nbs_x = (size_t)mmq_y * tile_x_k * sizeof(int);
    size_t nbs_y = (size_t)mmq_x * block_q8_1_mmq_size;
    size_t pad_unit = (size_t)nwarps * warp_size * sizeof(int);
    size_t nbs_y_padded = ((nbs_y + pad_unit - 1) / pad_unit) * pad_unit;
    return nbs_ids + nbs_x + nbs_y_padded;
}

// ============================================================================
// Hipcc compilation + caching
// ============================================================================

static std::string get_cache_dir() {
    const char * home = getenv("HOME");
    if (!home) home = getenv("USERPROFILE");
    if (!home) home = "/tmp";
    std::string dir = std::string(home) + "/.cache/gfx1100-megakernel";
    fs::create_directories(dir);
    return dir;
}

static std::string compute_hash(const std::string & flags) {
    // Simple hash of compile flags → deterministic filename
    size_t h = std::hash<std::string>{}(flags);
    char buf[32];
    snprintf(buf, sizeof(buf), "%016zx", h);
    return std::string(buf);
}

static std::string build_compile_flags(const gfx1100_model_config & cfg) {
    std::ostringstream ss;
    // --- Architecture + capability bits (drive #if specialization in .hip) ---
    ss << " -DARCH_ID=" << cfg.arch_id;
    ss << " -DROPE_TYPE=" << cfg.rope_type;
    ss << " -DNORM_TYPE=" << cfg.norm_type;
    ss << " -DACT_TYPE=" << cfg.act_type;
    ss << " -DATTN_SCALE_TYPE=" << cfg.attn_scale_type;
    ss << " -DHAS_QK_NORM=" << cfg.has_qk_norm;
    ss << " -DHAS_BIAS_Q=" << cfg.has_bias_q;
    ss << " -DHAS_BIAS_K=" << cfg.has_bias_k;
    ss << " -DHAS_BIAS_V=" << cfg.has_bias_v;
    ss << " -DHAS_BIAS_O=" << cfg.has_bias_o;
    ss << " -DHAS_SCALE_Q=" << cfg.has_scale_q;
    ss << " -DHAS_SCALE_K=" << cfg.has_scale_k;
    ss << " -DHAS_SCALE_V=" << cfg.has_scale_v;
    ss << " -DHAS_SCALE_O=" << cfg.has_scale_o;
    ss << " -DHAS_BIAS_FFN_GATE=" << cfg.has_bias_ffn_gate;
    ss << " -DHAS_BIAS_FFN_UP=" << cfg.has_bias_ffn_up;
    ss << " -DHAS_BIAS_FFN_DOWN=" << cfg.has_bias_ffn_down;
    ss << " -DHAS_SCALE_FFN_GATE=" << cfg.has_scale_ffn_gate;
    ss << " -DHAS_SCALE_FFN_UP=" << cfg.has_scale_ffn_up;
    ss << " -DHAS_SCALE_FFN_DOWN=" << cfg.has_scale_ffn_down;
    ss << " -DHAS_ROPE_FREQ_FACTORS=" << cfg.has_rope_freq_factors;
    ss << " -DHAS_MOE=" << cfg.has_moe;
    ss << " -DMOE_N_EXPERTS=" << cfg.moe_n_experts;
    ss << " -DMOE_N_EXPERTS_USED=" << cfg.moe_n_experts_used;
    ss << " -DHAS_SSM=" << cfg.has_ssm;
    ss << " -DHAS_DN=" << cfg.has_dn;
    ss << " -DHAS_FINAL_LOGIT_SOFTCAP=" << cfg.has_final_logit_softcap;
    ss << " -DHAS_SWA=" << cfg.has_swa;
    ss << " -DSWA_TYPE=" << cfg.swa_type;
    ss << " -DN_SWA=" << cfg.n_swa;
    ss << " -DHAS_ALIBI=" << cfg.has_alibi;
    ss << " -DPOOLING_TYPE=" << cfg.pooling_type;
    ss << " -DATTN_SOFTCAP_VAL=" << cfg.attn_softcap_val << "f";
    ss << " -DFINAL_LOGIT_SOFTCAP_VAL=" << cfg.final_logit_softcap_val << "f";

    // --- Geometry / per-model numerics ---
    ss << " -DHIDDEN_SIZE=" << cfg.hidden_size;
    ss << " -DINTERMEDIATE_SIZE=" << cfg.intermediate_size;
    ss << " -DVOCAB_SIZE=" << cfg.vocab_size;
    ss << " -DNUM_LAYERS=" << cfg.n_layers;
    ss << " -DNORM_EPS=" << cfg.norm_eps << "f";
    ss << " -DNORM_ADD_ONE=" << cfg.norm_add_one;
    ss << " -DFA_N_Q_HEADS=" << cfg.fa_n_q_heads;
    ss << " -DFA_N_KV_HEADS=" << cfg.fa_n_kv_heads;
    ss << " -DFA_HEAD_DIM=" << cfg.fa_head_dim;
    ss << " -DFA_ROPE_THETA=" << cfg.fa_rope_theta << "f";
    ss << " -DFA_ROPE_DIM=" << cfg.fa_rope_dim;
    float swa_theta = cfg.fa_rope_theta_swa > 0 ? cfg.fa_rope_theta_swa : cfg.fa_rope_theta;
    ss << " -DFA_ROPE_THETA_SWA=" << swa_theta << "f";
    ss << " -DFA_HAS_GATED_ATTN=" << cfg.fa_has_gated_attn;
    ss << " -DFA_USE_KQ_NORM=" << cfg.fa_use_kq_norm;
    ss << " -DFA_ATTENTION_SCALE=" << cfg.fa_attention_scale << "f";
    ss << " -DDN_N_HEADS=" << cfg.dn_n_heads;
    ss << " -DDN_N_K_HEADS=" << cfg.dn_n_k_heads;
    ss << " -DDN_KEY_DIM=" << cfg.dn_key_dim;
    ss << " -DDN_VALUE_DIM=" << cfg.dn_value_dim;
    ss << " -DDN_CONV_KERNEL=" << cfg.dn_conv_kernel;
    ss << " -DSSM_D_CONV="  << cfg.ssm_d_conv;
    ss << " -DSSM_D_INNER=" << cfg.ssm_d_inner;
    ss << " -DSSM_D_STATE=" << cfg.ssm_d_state;
    ss << " -DSSM_DT_RANK=" << cfg.ssm_dt_rank;
    ss << " -DSSM_N_GROUP=" << cfg.ssm_n_group;
    ss << " -DROPE_FREQ_SCALE="  << cfg.rope_freq_scale << "f";
    ss << " -DROPE_ATTN_FACTOR=" << cfg.rope_attn_factor << "f";
    ss << " -DYARN_EXT_FACTOR="  << cfg.yarn_ext_factor << "f";
    ss << " -DYARN_ATTN_FACTOR=" << cfg.yarn_attn_factor << "f";
    ss << " -DYARN_BETA_FAST="   << cfg.yarn_beta_fast  << "f";
    ss << " -DYARN_BETA_SLOW="   << cfg.yarn_beta_slow  << "f";
    ss << " -DN_CTX_ORIG_YARN="  << cfg.n_ctx_orig_yarn;
    // Precompute YaRN corr_dims from beta_fast/beta_slow — baseline ggml_rope_yarn_corr_dims
    {
        int n_dims = cfg.fa_rope_dim;
        int n_ctx  = cfg.n_ctx_orig_yarn > 0 ? cfg.n_ctx_orig_yarn : 4096;
        float base = cfg.fa_rope_theta > 0 ? cfg.fa_rope_theta : 10000.0f;
        auto corr_dim = [](int nd, int nc, float nr, float b) -> float {
            return nd * logf(nc / (nr * 2 * 3.14159265358979f)) / (2 * logf(b));
        };
        float cd0 = 0.0f, cd1 = 0.0f;
        if (n_dims > 0 && cfg.yarn_ext_factor != 0.0f) {
            cd0 = fmaxf(0.0f, floorf(corr_dim(n_dims, n_ctx, cfg.yarn_beta_fast, base)));
            cd1 = fminf((float)(n_dims - 1), ceilf(corr_dim(n_dims, n_ctx, cfg.yarn_beta_slow, base)));
        }
        ss << " -DYARN_CORR_DIM0=" << cd0 << "f";
        ss << " -DYARN_CORR_DIM1=" << cd1 << "f";
    }
    ss << " -DROPE_SECTION_0=" << cfg.rope_sections[0];
    ss << " -DROPE_SECTION_1=" << cfg.rope_sections[1];
    ss << " -DROPE_SECTION_2=" << cfg.rope_sections[2];
    ss << " -DROPE_SECTION_3=" << cfg.rope_sections[3];
    ss << " -DNUM_CUS=96 -DBLOCK_SIZE=512";
    ss << " -DGGML_USE_WMMA_FATTN";

    // Layer type array
    ss << " -DLAYER_TYPES_INIT=\"{";
    for (int i = 0; i < cfg.n_layers; i++) {
        if (i > 0) ss << ",";
        ss << cfg.layer_types[i];
    }
    ss << "}\"";

    return ss.str();
}

static int compile_kernel(const char * src_name, const gfx1100_model_config & cfg,
                          const std::string & cache_dir, const std::string & hash,
                          hipModule_t * out_module) {
    std::string hsaco_path = cache_dir + "/" + std::string(src_name) + "_" + hash + ".hsaco";

    if (fs::exists(hsaco_path)) {
        // Check if source is newer than cached .hsaco (invalidate stale cache)
        bool stale = false;
        {
            auto hsaco_time = fs::last_write_time(hsaco_path);
            const char * env_src = getenv("GFX1100_MEGAKERNEL_SRC");
            std::string check_dir = env_src ? env_src : "ggml/src/ggml-gfx1100_megakernel";
            std::string src_check = check_dir + "/" + src_name;
            if (fs::exists(src_check) && fs::last_write_time(src_check) > hsaco_time) {
                stale = true;
            }
            std::string helpers_dir = check_dir + "/helpers";
            if (fs::is_directory(helpers_dir)) {
                for (const auto & entry : fs::directory_iterator(helpers_dir)) {
                    if (entry.is_regular_file() && fs::last_write_time(entry) > hsaco_time) {
                        stale = true; break;
                    }
                }
            }
        }
        if (!stale) {
            hipError_t e = hipModuleLoad(out_module, hsaco_path.c_str());
            if (e == hipSuccess) {
                fprintf(stderr, "gfx1100-megakernel: loaded cached %s\n", hsaco_path.c_str());
                return 0;
            }
        } else {
            fprintf(stderr, "gfx1100-megakernel: source newer than cache, recompiling %s\n", src_name);
        }
    }

    // Find source directory
    std::string src_dir;
    const char * env_src = getenv("GFX1100_MEGAKERNEL_SRC");
    if (env_src) {
        src_dir = env_src;
    } else {
        const char * candidates[] = {
            "ggml/src/ggml-gfx1100_megakernel",
            "../ggml/src/ggml-gfx1100_megakernel",
            "../../ggml/src/ggml-gfx1100_megakernel",
            "../../../ggml/src/ggml-gfx1100_megakernel",
            ".",
        };
        for (auto c : candidates) {
            if (fs::exists(std::string(c) + "/" + src_name)) { src_dir = c; break; }
        }
    }
    assert(!src_dir.empty() && "Cannot find megakernel source directory");

    std::string src_path = src_dir + "/" + src_name;
    assert(fs::exists(src_path) && "Source file not found");

    // Write model constants to a temp header (avoids shell quoting issues).
    // Matches build_compile_flags() — keep in sync.
    std::string config_header = cache_dir + "/model_config_" + hash + ".h";
    {
        std::ofstream hf(config_header);
        assert(hf.is_open());
        hf << "#pragma once\n";
        // Arch + capability bits
        hf << "#define ARCH_ID " << cfg.arch_id << "\n";
        hf << "#define ROPE_TYPE " << cfg.rope_type << "\n";
        hf << "#define NORM_TYPE " << cfg.norm_type << "\n";
        hf << "#define ACT_TYPE " << cfg.act_type << "\n";
        hf << "#define ATTN_SCALE_TYPE " << cfg.attn_scale_type << "\n";
        hf << "#define HAS_QK_NORM " << cfg.has_qk_norm << "\n";
        hf << "#define HAS_BIAS_Q " << cfg.has_bias_q << "\n";
        hf << "#define HAS_BIAS_K " << cfg.has_bias_k << "\n";
        hf << "#define HAS_BIAS_V " << cfg.has_bias_v << "\n";
        hf << "#define HAS_BIAS_O " << cfg.has_bias_o << "\n";
        hf << "#define HAS_SCALE_Q " << cfg.has_scale_q << "\n";
        hf << "#define HAS_SCALE_K " << cfg.has_scale_k << "\n";
        hf << "#define HAS_SCALE_V " << cfg.has_scale_v << "\n";
        hf << "#define HAS_SCALE_O " << cfg.has_scale_o << "\n";
        hf << "#define HAS_BIAS_FFN_GATE " << cfg.has_bias_ffn_gate << "\n";
        hf << "#define HAS_BIAS_FFN_UP " << cfg.has_bias_ffn_up << "\n";
        hf << "#define HAS_BIAS_FFN_DOWN " << cfg.has_bias_ffn_down << "\n";
        hf << "#define HAS_SCALE_FFN_GATE " << cfg.has_scale_ffn_gate << "\n";
        hf << "#define HAS_SCALE_FFN_UP " << cfg.has_scale_ffn_up << "\n";
        hf << "#define HAS_SCALE_FFN_DOWN " << cfg.has_scale_ffn_down << "\n";
        hf << "#define HAS_ROPE_FREQ_FACTORS " << cfg.has_rope_freq_factors << "\n";
        hf << "#define HAS_MOE " << cfg.has_moe << "\n";
        hf << "#define MOE_N_EXPERTS " << cfg.moe_n_experts << "\n";
        hf << "#define MOE_N_EXPERTS_USED " << cfg.moe_n_experts_used << "\n";
        hf << "#define HAS_SSM " << cfg.has_ssm << "\n";
        hf << "#define HAS_DN " << cfg.has_dn << "\n";
        hf << "#define HAS_FINAL_LOGIT_SOFTCAP " << cfg.has_final_logit_softcap << "\n";
        hf << "#define HAS_SWA " << cfg.has_swa << "\n";
        hf << "#define SWA_TYPE " << cfg.swa_type << "\n";
        hf << "#define N_SWA " << cfg.n_swa << "\n";
        hf << "#define HAS_ALIBI " << cfg.has_alibi << "\n";
        hf << "#define POOLING_TYPE " << cfg.pooling_type << "\n";
        char cap_buf[32];
        snprintf(cap_buf, sizeof(cap_buf), "%.6ef", cfg.attn_softcap_val);
        hf << "#define ATTN_SOFTCAP_VAL " << cap_buf << "\n";
        hf << "#define HAS_ATTN_SOFTCAP " << (cfg.attn_softcap_val > 0 ? 1 : 0) << "\n";
        snprintf(cap_buf, sizeof(cap_buf), "%.6ef", cfg.final_logit_softcap_val);
        hf << "#define FINAL_LOGIT_SOFTCAP_VAL " << cap_buf << "\n";

        // Geometry
        hf << "#define HIDDEN_SIZE " << cfg.hidden_size << "\n";
        hf << "#define INTERMEDIATE_SIZE " << cfg.intermediate_size << "\n";
        hf << "#define VOCAB_SIZE " << cfg.vocab_size << "\n";
        hf << "#define NUM_LAYERS " << cfg.n_layers << "\n";
        char eps_buf[32]; snprintf(eps_buf, sizeof(eps_buf), "%.10ef", cfg.norm_eps);
        hf << "#define NORM_EPS " << eps_buf << "\n";
        hf << "#define NORM_ADD_ONE " << cfg.norm_add_one << "\n";
        hf << "#define FA_N_Q_HEADS " << cfg.fa_n_q_heads << "\n";
        hf << "#define FA_N_KV_HEADS " << cfg.fa_n_kv_heads << "\n";
        hf << "#define FA_HEAD_DIM " << cfg.fa_head_dim << "\n";
        char theta_buf[32]; snprintf(theta_buf, sizeof(theta_buf), "%.1ff", cfg.fa_rope_theta);
        hf << "#define FA_ROPE_THETA " << theta_buf << "\n";
        {
            float swa_th = cfg.fa_rope_theta_swa > 0 ? cfg.fa_rope_theta_swa : cfg.fa_rope_theta;
            char swa_buf[32]; snprintf(swa_buf, sizeof(swa_buf), "%.1ff", swa_th);
            hf << "#define FA_ROPE_THETA_SWA " << swa_buf << "\n";
        }
        hf << "#define FA_ROPE_DIM " << cfg.fa_rope_dim << "\n";
        hf << "#define FA_HAS_GATED_ATTN " << cfg.fa_has_gated_attn << "\n";
        hf << "#define FA_USE_KQ_NORM " << cfg.fa_use_kq_norm << "\n";
        char scale_buf[32];
        snprintf(scale_buf, sizeof(scale_buf), "%.10ef", cfg.fa_attention_scale);
        hf << "#define FA_ATTENTION_SCALE " << scale_buf << "\n";
        hf << "#define HAS_CUSTOM_ATTN_SCALE " << (cfg.fa_attention_scale != 0.0f ? 1 : 0) << "\n";
        hf << "#define DN_N_HEADS " << cfg.dn_n_heads << "\n";
        hf << "#define DN_N_K_HEADS " << cfg.dn_n_k_heads << "\n";
        hf << "#define DN_KEY_DIM " << cfg.dn_key_dim << "\n";
        hf << "#define DN_VALUE_DIM " << cfg.dn_value_dim << "\n";
        hf << "#define DN_CONV_KERNEL " << cfg.dn_conv_kernel << "\n";
        hf << "#define SSM_D_CONV " << cfg.ssm_d_conv << "\n";
        hf << "#define SSM_D_INNER " << cfg.ssm_d_inner << "\n";
        hf << "#define SSM_D_STATE " << cfg.ssm_d_state << "\n";
        hf << "#define SSM_DT_RANK " << cfg.ssm_dt_rank << "\n";
        hf << "#define SSM_N_GROUP " << cfg.ssm_n_group << "\n";
        char ropescale_buf[32];
        snprintf(ropescale_buf, sizeof(ropescale_buf), "%.10ef", cfg.rope_freq_scale);
        hf << "#define ROPE_FREQ_SCALE " << ropescale_buf << "\n";
        snprintf(ropescale_buf, sizeof(ropescale_buf), "%.10ef", cfg.rope_attn_factor);
        hf << "#define ROPE_ATTN_FACTOR " << ropescale_buf << "\n";
        snprintf(ropescale_buf, sizeof(ropescale_buf), "%.10ef", cfg.yarn_ext_factor);
        hf << "#define YARN_EXT_FACTOR " << ropescale_buf << "\n";
        snprintf(ropescale_buf, sizeof(ropescale_buf), "%.10ef", cfg.yarn_attn_factor);
        hf << "#define YARN_ATTN_FACTOR " << ropescale_buf << "\n";
        snprintf(ropescale_buf, sizeof(ropescale_buf), "%.10ef", cfg.yarn_beta_fast);
        hf << "#define YARN_BETA_FAST " << ropescale_buf << "\n";
        snprintf(ropescale_buf, sizeof(ropescale_buf), "%.10ef", cfg.yarn_beta_slow);
        hf << "#define YARN_BETA_SLOW " << ropescale_buf << "\n";
        hf << "#define N_CTX_ORIG_YARN " << cfg.n_ctx_orig_yarn << "\n";
        // YaRN corr_dims — precomputed from beta_fast/beta_slow (same formula as command-line path)
        {
            int n_dims = cfg.fa_rope_dim;
            int n_ctx  = cfg.n_ctx_orig_yarn > 0 ? cfg.n_ctx_orig_yarn : 4096;
            float base = cfg.fa_rope_theta > 0 ? cfg.fa_rope_theta : 10000.0f;
            auto corr_dim_fn = [](int nd, int nc, float nr, float b) -> float {
                return nd * logf(nc / (nr * 2 * 3.14159265358979f)) / (2 * logf(b));
            };
            float cd0 = 0.0f, cd1 = 0.0f;
            if (n_dims > 0 && cfg.yarn_ext_factor != 0.0f) {
                cd0 = fmaxf(0.0f, floorf(corr_dim_fn(n_dims, n_ctx, cfg.yarn_beta_fast, base)));
                cd1 = fminf((float)(n_dims - 1), ceilf(corr_dim_fn(n_dims, n_ctx, cfg.yarn_beta_slow, base)));
            }
            snprintf(ropescale_buf, sizeof(ropescale_buf), "%.10ef", cd0);
            hf << "#define YARN_CORR_DIM0 " << ropescale_buf << "\n";
            snprintf(ropescale_buf, sizeof(ropescale_buf), "%.10ef", cd1);
            hf << "#define YARN_CORR_DIM1 " << ropescale_buf << "\n";
        }
        hf << "#define ROPE_SECTION_0 " << cfg.rope_sections[0] << "\n";
        hf << "#define ROPE_SECTION_1 " << cfg.rope_sections[1] << "\n";
        hf << "#define ROPE_SECTION_2 " << cfg.rope_sections[2] << "\n";
        hf << "#define ROPE_SECTION_3 " << cfg.rope_sections[3] << "\n";
        hf << "#define NUM_CUS 96\n";
        hf << "#define BLOCK_SIZE 512\n";
        hf << "#define GGML_USE_WMMA_FATTN 1\n";
        hf << "#define LAYER_TYPES_INIT {";
        for (int i = 0; i < cfg.n_layers; i++) {
            if (i > 0) hf << ",";
            hf << cfg.layer_types[i];
        }
        hf << "}\n";
    }

    // Find hipcc — check pip-installed ROCm SDK first, then standard install paths
    std::string hipcc = "hipcc";
    g_rocm_device_lib_path.clear();  // reset for this compilation
    for (const char * p : {
        "C:/rocm-7.2.1/Lib/site-packages/_rocm_sdk_devel/bin/hipcc.exe",
        "C:/Program Files/AMD/ROCm/7.1/bin/hipcc.exe",
        "C:/Program Files/AMD/ROCm/6.3/bin/hipcc.exe",
        "C:/Program Files/AMD/ROCm/6.2/bin/hipcc.exe",
    }) {
        if (fs::exists(p)) { hipcc = p; break; }
    }
    // Auto-detect device lib path from hipcc location
    // pip install: .../bin/hipcc.exe → .../_rocm_sdk_core/lib/llvm/amdgcn/bitcode
    // system install: .../bin/hipcc.exe → .../amdgcn/bitcode (found automatically)
    {
        fs::path hipcc_dir = fs::path(hipcc).parent_path();
        // pip-installed ROCm: _rocm_sdk_devel/bin → _rocm_sdk_core/lib/llvm/amdgcn/bitcode
        fs::path pip_bitcode = hipcc_dir / ".." / ".." / "_rocm_sdk_core" / "lib" / "llvm" / "amdgcn" / "bitcode";
        // System ROCm: ROCm/7.x/bin → ROCm/7.x/amdgcn/bitcode
        fs::path sys_bitcode = hipcc_dir / ".." / "amdgcn" / "bitcode";
        if (fs::exists(pip_bitcode / "hip.bc")) {
            g_rocm_device_lib_path = pip_bitcode.string();
        } else if (fs::exists(sys_bitcode / "hip.bc")) {
            g_rocm_device_lib_path = sys_bitcode.string();
        }
    }

    // Normalize paths to forward slashes
    auto norm = [](std::string s) {
        for (auto & c : s) if (c == '\\') c = '/';
        return s;
    };

    // On Windows, system() with nested quotes is broken.
    // Write a .bat/.sh script to avoid quoting hell.
    std::string script_path = cache_dir + "/compile_" + hash;
#ifdef _WIN32
    script_path += ".bat";
    {
        std::ofstream sf(script_path);
        assert(sf.is_open());
        sf << "@echo off\n";
        sf << "\"" << norm(hipcc) << "\" --genco --offload-arch=gfx1100 -O3 --std=c++17";
        sf << " -Wno-unused-result";
        if (!g_rocm_device_lib_path.empty()) sf << " --rocm-device-lib-path=\"" << norm(g_rocm_device_lib_path) << "\"";
        sf << " -I\"" << norm(src_dir) << "\"";
        sf << " -I\"C:/Users/thund/development/rocwmma/library/include\"";
        sf << " -DGGML_USE_WMMA_FATTN";
        sf << " -include \"" << norm(config_header) << "\"";
        sf << " \"" << norm(src_path) << "\"";
        sf << " -o \"" << norm(hsaco_path) << "\"\n";
    }
    std::string cmd = "\"" + norm(script_path) + "\"";
#else
    script_path += ".sh";
    {
        std::ofstream sf(script_path);
        assert(sf.is_open());
        sf << "#!/bin/sh\n";
        sf << "'" << norm(hipcc) << "' --genco --offload-arch=gfx1100 -O3 --std=c++17";
        sf << " -Wno-unused-result";
        if (!g_rocm_device_lib_path.empty()) sf << " --rocm-device-lib-path='" << norm(g_rocm_device_lib_path) << "'";
        sf << " -I'" << norm(src_dir) << "'";
        sf << " -I'C:/Users/thund/development/rocwmma/library/include'";
        sf << " -DGGML_USE_WMMA_FATTN";
        sf << " -include '" << norm(config_header) << "'";
        sf << " '" << norm(src_path) << "'";
        sf << " -o '" << norm(hsaco_path) << "'\n";
    }
    std::string cmd = "sh '" + norm(script_path) + "'";
#endif

    fprintf(stderr, "gfx1100-megakernel: compiling %s...\n", src_name);
    fprintf(stderr, "  script: %s\n", script_path.c_str());
    int rc = system(cmd.c_str());
    if (rc != 0) {
        fprintf(stderr, "gfx1100-megakernel: hipcc failed (exit %d)\n", rc);
        return -1;
    }

    hipError_t e = hipModuleLoad(out_module, hsaco_path.c_str());
    if (e != hipSuccess) {
        fprintf(stderr, "gfx1100-megakernel: hipModuleLoad failed: %s\n", hipGetErrorString(e));
        return -1;
    }

    fprintf(stderr, "gfx1100-megakernel: compiled %s (cached: %s)\n", src_name, hsaco_path.c_str());
    return 0;
}

// ============================================================================
// DLL compilation — produces a shared library with fast hipLaunchKernel path
// ============================================================================

static int compile_dll(const gfx1100_model_config & cfg,
                       const std::string & cache_dir, const std::string & hash) {
    std::string dll_ext;
#ifdef _WIN32
    dll_ext = ".dll";
#else
    dll_ext = ".so";
#endif
    std::string dll_path = cache_dir + "/decode_" + hash + dll_ext;

    // Check cache
    if (fs::exists(dll_path)) {
        // Check if source is newer than cached DLL (invalidate stale cache)
        bool stale = false;
        auto dll_time = fs::last_write_time(dll_path);
        const char * env_src = getenv("GFX1100_MEGAKERNEL_SRC");
        std::string check_dir = env_src ? env_src : "ggml/src/ggml-gfx1100_megakernel";
        // Check both decode.hip and decode-dll-wrapper.hip
        for (const char * fn : {"decode.hip", "decode-dll-wrapper.hip"}) {
            std::string src_check = check_dir + "/" + fn;
            if (fs::exists(src_check) && fs::last_write_time(src_check) > dll_time) {
                stale = true; break;
            }
        }
        // Check helpers/
        std::string helpers_dir = check_dir + "/helpers";
        if (!stale && fs::is_directory(helpers_dir)) {
            for (const auto & entry : fs::directory_iterator(helpers_dir)) {
                if (entry.is_regular_file() && fs::last_write_time(entry) > dll_time) {
                    stale = true; break;
                }
            }
        }
        if (!stale) {
            fprintf(stderr, "gfx1100-megakernel: loading cached DLL %s\n", dll_path.c_str());
            goto load_dll;
        }
        fprintf(stderr, "gfx1100-megakernel: source newer than cache, recompiling DLL\n");
    }

    {
        // Find source directory
        std::string src_dir;
        const char * env_src2 = getenv("GFX1100_MEGAKERNEL_SRC");
        if (env_src2) {
            src_dir = env_src2;
        } else {
            const char * candidates[] = {
                "ggml/src/ggml-gfx1100_megakernel",
                "../ggml/src/ggml-gfx1100_megakernel",
                "../../ggml/src/ggml-gfx1100_megakernel",
                "../../../ggml/src/ggml-gfx1100_megakernel",
                ".",
            };
            for (auto c : candidates) {
                if (fs::exists(std::string(c) + "/decode-dll-wrapper.hip")) { src_dir = c; break; }
            }
        }
        assert(!src_dir.empty() && "Cannot find megakernel source directory");

        std::string wrapper_path = src_dir + "/decode-dll-wrapper.hip";
        assert(fs::exists(wrapper_path) && "decode-dll-wrapper.hip not found");

        // Write model config header (reuse the same format as .hsaco path)
        std::string config_header = cache_dir + "/model_config_" + hash + ".h";
        // The config header is already written by build_compile_flags / compile_kernel
        // but in case we're called first, generate it here too
        if (!fs::exists(config_header)) {
            // Use the same header generation as compile_kernel
            // (This code path is hit when DLL mode is used without .hsaco fallback)
            std::ofstream hf(config_header);
            assert(hf.is_open());
            hf << "#pragma once\n";
            hf << "#define ARCH_ID " << cfg.arch_id << "\n";
            hf << "#define ROPE_TYPE " << cfg.rope_type << "\n";
            hf << "#define NORM_TYPE " << cfg.norm_type << "\n";
            hf << "#define ACT_TYPE " << cfg.act_type << "\n";
            hf << "#define ATTN_SCALE_TYPE " << cfg.attn_scale_type << "\n";
            hf << "#define HAS_QK_NORM " << cfg.has_qk_norm << "\n";
            hf << "#define HAS_BIAS_Q " << cfg.has_bias_q << "\n";
            hf << "#define HAS_BIAS_K " << cfg.has_bias_k << "\n";
            hf << "#define HAS_BIAS_V " << cfg.has_bias_v << "\n";
            hf << "#define HAS_BIAS_O " << cfg.has_bias_o << "\n";
            hf << "#define HAS_SCALE_Q " << cfg.has_scale_q << "\n";
            hf << "#define HAS_SCALE_K " << cfg.has_scale_k << "\n";
            hf << "#define HAS_SCALE_V " << cfg.has_scale_v << "\n";
            hf << "#define HAS_SCALE_O " << cfg.has_scale_o << "\n";
            hf << "#define HAS_BIAS_FFN_GATE " << cfg.has_bias_ffn_gate << "\n";
            hf << "#define HAS_BIAS_FFN_UP " << cfg.has_bias_ffn_up << "\n";
            hf << "#define HAS_BIAS_FFN_DOWN " << cfg.has_bias_ffn_down << "\n";
            hf << "#define HAS_SCALE_FFN_GATE " << cfg.has_scale_ffn_gate << "\n";
            hf << "#define HAS_SCALE_FFN_UP " << cfg.has_scale_ffn_up << "\n";
            hf << "#define HAS_SCALE_FFN_DOWN " << cfg.has_scale_ffn_down << "\n";
            hf << "#define HAS_ROPE_FREQ_FACTORS " << cfg.has_rope_freq_factors << "\n";
            hf << "#define HAS_MOE " << cfg.has_moe << "\n";
            hf << "#define MOE_N_EXPERTS " << cfg.moe_n_experts << "\n";
            hf << "#define MOE_N_EXPERTS_USED " << cfg.moe_n_experts_used << "\n";
            hf << "#define HAS_SSM " << cfg.has_ssm << "\n";
            hf << "#define HAS_DN " << cfg.has_dn << "\n";
            hf << "#define HAS_FINAL_LOGIT_SOFTCAP " << cfg.has_final_logit_softcap << "\n";
            hf << "#define HAS_SWA " << cfg.has_swa << "\n";
            hf << "#define SWA_TYPE " << cfg.swa_type << "\n";
            hf << "#define N_SWA " << cfg.n_swa << "\n";
            hf << "#define HAS_ALIBI " << cfg.has_alibi << "\n";
            hf << "#define POOLING_TYPE " << cfg.pooling_type << "\n";
            char cap_buf[32];
            snprintf(cap_buf, sizeof(cap_buf), "%.6ef", cfg.attn_softcap_val);
            hf << "#define ATTN_SOFTCAP_VAL " << cap_buf << "\n";
            hf << "#define HAS_ATTN_SOFTCAP " << (cfg.attn_softcap_val > 0 ? 1 : 0) << "\n";
            snprintf(cap_buf, sizeof(cap_buf), "%.6ef", cfg.final_logit_softcap_val);
            hf << "#define FINAL_LOGIT_SOFTCAP_VAL " << cap_buf << "\n";
            hf << "#define HIDDEN_SIZE " << cfg.hidden_size << "\n";
            hf << "#define INTERMEDIATE_SIZE " << cfg.intermediate_size << "\n";
            hf << "#define VOCAB_SIZE " << cfg.vocab_size << "\n";
            hf << "#define NUM_LAYERS " << cfg.n_layers << "\n";
            char eps_buf[32]; snprintf(eps_buf, sizeof(eps_buf), "%.10ef", cfg.norm_eps);
            hf << "#define NORM_EPS " << eps_buf << "\n";
            hf << "#define NORM_ADD_ONE " << cfg.norm_add_one << "\n";
            hf << "#define FA_N_Q_HEADS " << cfg.fa_n_q_heads << "\n";
            hf << "#define FA_N_KV_HEADS " << cfg.fa_n_kv_heads << "\n";
            hf << "#define FA_HEAD_DIM " << cfg.fa_head_dim << "\n";
            char theta_buf[32]; snprintf(theta_buf, sizeof(theta_buf), "%.1ff", cfg.fa_rope_theta);
            hf << "#define FA_ROPE_THETA " << theta_buf << "\n";
            {
                float swa_th = cfg.fa_rope_theta_swa > 0 ? cfg.fa_rope_theta_swa : cfg.fa_rope_theta;
                char swa_buf[32]; snprintf(swa_buf, sizeof(swa_buf), "%.1ff", swa_th);
                hf << "#define FA_ROPE_THETA_SWA " << swa_buf << "\n";
            }
            hf << "#define FA_ROPE_DIM " << cfg.fa_rope_dim << "\n";
            hf << "#define FA_HAS_GATED_ATTN " << cfg.fa_has_gated_attn << "\n";
            hf << "#define FA_USE_KQ_NORM " << cfg.fa_use_kq_norm << "\n";
            char scale_buf[32];
            snprintf(scale_buf, sizeof(scale_buf), "%.10ef", cfg.fa_attention_scale);
            hf << "#define FA_ATTENTION_SCALE " << scale_buf << "\n";
            hf << "#define HAS_CUSTOM_ATTN_SCALE " << (cfg.fa_attention_scale != 0.0f ? 1 : 0) << "\n";
            hf << "#define DN_N_HEADS " << cfg.dn_n_heads << "\n";
            hf << "#define DN_N_K_HEADS " << cfg.dn_n_k_heads << "\n";
            hf << "#define DN_KEY_DIM " << cfg.dn_key_dim << "\n";
            hf << "#define DN_VALUE_DIM " << cfg.dn_value_dim << "\n";
            hf << "#define DN_CONV_KERNEL " << cfg.dn_conv_kernel << "\n";
            hf << "#define SSM_D_CONV " << cfg.ssm_d_conv << "\n";
            hf << "#define SSM_D_INNER " << cfg.ssm_d_inner << "\n";
            hf << "#define SSM_D_STATE " << cfg.ssm_d_state << "\n";
            hf << "#define SSM_DT_RANK " << cfg.ssm_dt_rank << "\n";
            hf << "#define SSM_N_GROUP " << cfg.ssm_n_group << "\n";
            char ropescale_buf[32];
            snprintf(ropescale_buf, sizeof(ropescale_buf), "%.10ef", cfg.rope_freq_scale);
            hf << "#define ROPE_FREQ_SCALE " << ropescale_buf << "\n";
            snprintf(ropescale_buf, sizeof(ropescale_buf), "%.10ef", cfg.rope_attn_factor);
            hf << "#define ROPE_ATTN_FACTOR " << ropescale_buf << "\n";
            snprintf(ropescale_buf, sizeof(ropescale_buf), "%.10ef", cfg.yarn_ext_factor);
            hf << "#define YARN_EXT_FACTOR " << ropescale_buf << "\n";
            snprintf(ropescale_buf, sizeof(ropescale_buf), "%.10ef", cfg.yarn_attn_factor);
            hf << "#define YARN_ATTN_FACTOR " << ropescale_buf << "\n";
            snprintf(ropescale_buf, sizeof(ropescale_buf), "%.10ef", cfg.yarn_beta_fast);
            hf << "#define YARN_BETA_FAST " << ropescale_buf << "\n";
            snprintf(ropescale_buf, sizeof(ropescale_buf), "%.10ef", cfg.yarn_beta_slow);
            hf << "#define YARN_BETA_SLOW " << ropescale_buf << "\n";
            hf << "#define N_CTX_ORIG_YARN " << cfg.n_ctx_orig_yarn << "\n";
            {
                int n_dims = cfg.fa_rope_dim;
                int n_ctx  = cfg.n_ctx_orig_yarn > 0 ? cfg.n_ctx_orig_yarn : 4096;
                float base = cfg.fa_rope_theta > 0 ? cfg.fa_rope_theta : 10000.0f;
                auto corr_dim_fn = [](int nd, int nc, float nr, float b) -> float {
                    return nd * logf(nc / (nr * 2 * 3.14159265358979f)) / (2 * logf(b));
                };
                float cd0 = 0.0f, cd1 = 0.0f;
                if (n_dims > 0 && cfg.yarn_ext_factor != 0.0f) {
                    cd0 = fmaxf(0.0f, floorf(corr_dim_fn(n_dims, n_ctx, cfg.yarn_beta_fast, base)));
                    cd1 = fminf((float)(n_dims - 1), ceilf(corr_dim_fn(n_dims, n_ctx, cfg.yarn_beta_slow, base)));
                }
                snprintf(ropescale_buf, sizeof(ropescale_buf), "%.10ef", cd0);
                hf << "#define YARN_CORR_DIM0 " << ropescale_buf << "\n";
                snprintf(ropescale_buf, sizeof(ropescale_buf), "%.10ef", cd1);
                hf << "#define YARN_CORR_DIM1 " << ropescale_buf << "\n";
            }
            hf << "#define ROPE_SECTION_0 " << cfg.rope_sections[0] << "\n";
            hf << "#define ROPE_SECTION_1 " << cfg.rope_sections[1] << "\n";
            hf << "#define ROPE_SECTION_2 " << cfg.rope_sections[2] << "\n";
            hf << "#define ROPE_SECTION_3 " << cfg.rope_sections[3] << "\n";
            hf << "#define NUM_CUS 96\n";
            hf << "#define BLOCK_SIZE 512\n";
            hf << "#define GGML_USE_WMMA_FATTN 1\n";
            hf << "#define LAYER_TYPES_INIT {";
            for (int i = 0; i < cfg.n_layers; i++) {
                if (i > 0) hf << ",";
                hf << cfg.layer_types[i];
            }
            hf << "}\n";
        }

        // Find hipcc — same search order as compile_kernel
        std::string hipcc = "hipcc";
        for (const char * p : {
            "C:/rocm-7.2.1/Lib/site-packages/_rocm_sdk_devel/bin/hipcc.exe",
            "C:/Program Files/AMD/ROCm/7.1/bin/hipcc.exe",
            "C:/Program Files/AMD/ROCm/6.3/bin/hipcc.exe",
            "C:/Program Files/AMD/ROCm/6.2/bin/hipcc.exe",
        }) {
            if (fs::exists(p)) { hipcc = p; break; }
        }

        auto norm = [](std::string s) {
            for (auto & c : s) if (c == '\\') c = '/';
            return s;
        };

        // Write compile script — uses --shared instead of --genco
        std::string script_path = cache_dir + "/compile_dll_" + hash;
#ifdef _WIN32
        script_path += ".bat";
        {
            std::ofstream sf(script_path);
            assert(sf.is_open());
            sf << "@echo off\n";
            sf << "\"" << norm(hipcc) << "\" --shared --offload-arch=gfx1100 -O3 --std=c++17";
            sf << " -Wno-unused-result";
            if (!g_rocm_device_lib_path.empty()) sf << " --rocm-device-lib-path=\"" << norm(g_rocm_device_lib_path) << "\"";
            sf << " -I\"" << norm(src_dir) << "\"";
            sf << " -I\"C:/Users/thund/development/rocwmma/library/include\"";
            sf << " -DGGML_USE_WMMA_FATTN";
            sf << " -include \"" << norm(config_header) << "\"";
            sf << " \"" << norm(wrapper_path) << "\"";
            sf << " -o \"" << norm(dll_path) << "\"\n";
        }
        std::string cmd = "\"" + norm(script_path) + "\"";
#else
        script_path += ".sh";
        {
            std::ofstream sf(script_path);
            assert(sf.is_open());
            sf << "#!/bin/sh\n";
            sf << "'" << norm(hipcc) << "' --shared --offload-arch=gfx1100 -O3 --std=c++17";
            sf << " -Wno-unused-result";
            sf << " -fPIC";
            if (!g_rocm_device_lib_path.empty()) sf << " --rocm-device-lib-path='" << norm(g_rocm_device_lib_path) << "'";
            sf << " -I'" << norm(src_dir) << "'";
            sf << " -I'C:/Users/thund/development/rocwmma/library/include'";
            sf << " -DGGML_USE_WMMA_FATTN";
            sf << " -include '" << norm(config_header) << "'";
            sf << " '" << norm(wrapper_path) << "'";
            sf << " -o '" << norm(dll_path) << "'\n";
        }
        std::string cmd = "sh '" + norm(script_path) + "'";
#endif

        fprintf(stderr, "gfx1100-megakernel: compiling DLL...\n");
        fprintf(stderr, "  script: %s\n", script_path.c_str());
        int rc = system(cmd.c_str());
        if (rc != 0) {
            fprintf(stderr, "gfx1100-megakernel: DLL compilation failed (exit %d)\n", rc);
            return -1;
        }
        fprintf(stderr, "gfx1100-megakernel: compiled DLL (cached: %s)\n", dll_path.c_str());
    }

load_dll:
    // Load the DLL
#ifdef _WIN32
    HMODULE hmod = LoadLibraryA(dll_path.c_str());
    if (!hmod) {
        fprintf(stderr, "gfx1100-megakernel: LoadLibrary failed for %s (error %lu)\n",
                dll_path.c_str(), GetLastError());
        return -1;
    }
    g_dll_handle = (void *)hmod;
    g_get_kernel = (gfx1100_get_kernel_fn)GetProcAddress(hmod, "gfx1100_get_kernel");
    g_dll_launch = (gfx1100_dll_launch_fn)GetProcAddress(hmod, "gfx1100_dll_launch");
#else
    void * dl = dlopen(dll_path.c_str(), RTLD_NOW);
    if (!dl) {
        fprintf(stderr, "gfx1100-megakernel: dlopen failed for %s: %s\n",
                dll_path.c_str(), dlerror());
        return -1;
    }
    g_dll_handle = dl;
    g_get_kernel = (gfx1100_get_kernel_fn)dlsym(dl, "gfx1100_get_kernel");
    g_dll_launch = (gfx1100_dll_launch_fn)dlsym(dl, "gfx1100_dll_launch");
#endif
    if (!g_get_kernel || !g_dll_launch) {
        fprintf(stderr, "gfx1100-megakernel: FATAL — DLL missing exports "
                "(get_kernel=%p, dll_launch=%p)\n",
                (void *)g_get_kernel, (void *)g_dll_launch);
        return -1;
    }
    fprintf(stderr, "gfx1100-megakernel: DLL loaded, fast dispatch active\n");
    return 0;
}

// Helper: load a kernel from the DLL by name, storing as hipFunction_t.
// The actual value is a const void* (host __global__ fn pointer), but we
// store it in hipFunction_t for compatibility with existing struct layout.
// gfx1100_dispatch() casts it back to const void* when in DLL mode.
static hipFunction_t dll_get_fn(const char * name) {
    const void * fn = g_get_kernel(name);
    return reinterpret_cast<hipFunction_t>(const_cast<void *>(fn));
}

// ============================================================================
// Init: compile kernels + allocate buffers
// ============================================================================

int gfx1100_init(const gfx1100_model_config * cfg) {
    assert(cfg != nullptr);

    // ABI-safe copy: callers may link against an older `gfx1100_model_config`
    // layout that ends at `kv_type` (before the VALIDATE-step trailing fields).
    // Using `memcpy(..., sizeof(g_config))` here would read past the caller's
    // buffer and fill our trailing fields with stack garbage. Instead:
    //   1. Zero the entire struct so unset trailing fields stay well-defined.
    //   2. Copy only up to the end of the legacy layout (`kv_type`).
    // When we introduce a versioned ABI, this truncation window can grow.
    memset(&g_config, 0, sizeof(g_config));
    static constexpr size_t legacy_size =
        offsetof(gfx1100_model_config, kv_type) + sizeof(g_config.kv_type);
    memcpy(&g_config, cfg, legacy_size);

    // Struct ABI check for composition diagnostics
    if (getenv("GFX1100_COMPOSITION_DIAG") || getenv("GFX1100_DIAG_ONLY")) {
        fprintf(stderr, "\n=== Struct ABI ===\n");
        fprintf(stderr, "  DLL view:  sizeof(gfx1100_model_config) = %zu\n", sizeof(gfx1100_model_config));
        fprintf(stderr, "  DLL view:  sizeof(gfx1100_layer_weights) = %zu\n", sizeof(gfx1100_layer_weights));
        fprintf(stderr, "  offsetof(layers) = %zu\n", offsetof(gfx1100_model_config, layers));
        fprintf(stderr, "  legacy-copy size = %zu (from caller)\n",
                offsetof(gfx1100_model_config, kv_type) + sizeof(g_config.kv_type));
        fprintf(stderr, "  L0 ptrs[0]=%p ptrs[1]=%p ptrs[2]=%p ptrs[3]=%p\n",
                g_config.layers[0].ptrs[0], g_config.layers[0].ptrs[1],
                g_config.layers[0].ptrs[2], g_config.layers[0].ptrs[3]);
    }

    // === Composition meta-steps 1+2: DETECT + VALIDATE ===
    // Always run. Results cached in g_comp_caps / g_comp_validation for any
    // downstream code that wants to consume them. Nothing gates on them yet —
    // behavior is unchanged from before. Env vars only control printing:
    //   GFX1100_COMPOSITION_DIAG=1 — print capabilities + validation to stderr
    //   GFX1100_DIAG_ONLY=1        — print and exit(0) before GPU init
    const bool diag      = getenv("GFX1100_COMPOSITION_DIAG") || getenv("GFX1100_DIAG_ONLY");
    const bool diag_only = getenv("GFX1100_DIAG_ONLY") != nullptr;

    comp_detect(g_config, g_comp_caps);
    comp_validate(g_config, g_comp_validation);
    comp_tune(g_config, g_comp_caps, g_comp_tuning);
    if (diag) {
        comp_print_capabilities(g_comp_caps);
        comp_print_validation(g_comp_validation);
        comp_print_tuning(g_comp_tuning);
    }
    if (diag_only) {
        fprintf(stderr, "gfx1100: GFX1100_DIAG_ONLY set — exiting before GPU init\n");
        fflush(stderr);
        exit(0);
    }

    // Auto-populate layer_use_swa[] if n_swa > 0 but no layers are marked
    // Most iSWA models alternate: even layers = SWA, odd = global (Gemma2/3/4, Cohere2, etc.)
    if (g_config.n_swa > 0) {
        bool has_any_swa = false;
        for (int i = 0; i < g_config.n_layers; i++) {
            if (g_config.layer_use_swa[i]) { has_any_swa = true; break; }
        }
        if (!has_any_swa) {
            // Fallback: alternating SWA pattern (even=SWA, odd=global)
            // This matches Gemma2/3/4, Cohere2, Exaone4, Plamo3 default patterns
            for (int i = 0; i < g_config.n_layers; i++) {
                g_config.layer_use_swa[i] = (i % 2 == 0) ? 1 : 0;
            }
            fprintf(stderr, "gfx1100: auto-populated layer_use_swa[] with alternating pattern (n_swa=%d)\n",
                    g_config.n_swa);
        }
    }

    std::string flags = build_compile_flags(g_config);
    std::string cache_dir = get_cache_dir();
    std::string hash = compute_hash(flags);

    // Auto-detect ROCm device library path (needed for pip-installed ROCm SDK)
    // Must run before compile_dll/compile_kernel which generate JIT compile scripts
    {
        for (const char * p : {
            "C:/rocm-7.2.1/Lib/site-packages/_rocm_sdk_core/lib/llvm/amdgcn/bitcode/hip.bc",
            "C:/Program Files/AMD/ROCm/7.1/amdgcn/bitcode/hip.bc",
        }) {
            if (fs::exists(p)) {
                g_rocm_device_lib_path = fs::path(p).parent_path().string();
                break;
            }
        }
        if (!g_rocm_device_lib_path.empty()) {
            fprintf(stderr, "gfx1100: device lib path: %s\n", g_rocm_device_lib_path.c_str());
        }
    }

    // Choose dispatch mode: DLL (fast, default) or HSACO (fallback)
    bool force_hsaco = (getenv("GFX1100_HSACO") != nullptr);
    g_dll_mode = false;

    if (!force_hsaco) {
        // Try DLL compilation first — faster kernel dispatch
        int dll_rc = compile_dll(g_config, cache_dir, hash);
        if (dll_rc == 0) {
            g_dll_mode = true;
            fprintf(stderr, "gfx1100-megakernel: using DLL dispatch (~1.5µs/launch)\n");
        } else {
            fprintf(stderr, "gfx1100-megakernel: DLL compilation failed, falling back to HSACO\n");
        }
    }

    if (!g_dll_mode) {
        // HSACO path: compile device code object, load via hipModule API
        if (compile_kernel("decode.hip", g_config, cache_dir, hash, &g_compiled.eval_module) != 0) return -1;
        fprintf(stderr, "gfx1100-megakernel: using HSACO dispatch (~2.1µs/launch)\n");
    }

    // In DLL mode, em is null (no .hsaco compiled for decode).
    // In HSACO mode, em is the loaded module.
    // Prompt module always uses .hsaco regardless of mode.
    hipModule_t em = g_compiled.eval_module;

    // Unified kernel loading — DLL for decode module, HSACO for prompt module.
    // The check "mod == em" works because:
    //   DLL mode:   em = null, decode LOAD_FNs pass em (null), prompt LOAD_FNs pass prompt_module (valid)
    //   HSACO mode: em = valid module, all LOAD_FNs use the appropriate module
    auto load_fn = [em](hipFunction_t * fn, hipModule_t mod, const char * name) {
        if (g_dll_mode && mod == em) {
            *fn = dll_get_fn(name);
            if (!*fn) {
                fprintf(stderr, "gfx1100: FATAL — kernel '%s' not found in DLL\n", name);
                return hipErrorNotFound;
            }
            return hipSuccess;
        }
        hipError_t e = hipModuleGetFunction(fn, mod, name);
        if (e != hipSuccess) {
            fprintf(stderr, "gfx1100: FATAL — kernel '%s' not found: %s\n", name, hipGetErrorString(e));
            *fn = nullptr;
        }
        return e;
    };
    int load_errors = 0;
    #define LOAD_FN(fn_field, mod, name) do { if (load_fn(&fn_field, mod, name) != hipSuccess) load_errors++; } while(0)
    // OPT_FN: optional kernel loading — silent on failure (MoE, WavTokenizer, fused, etc.)
    #define OPT_FN(fn_field, mod, name) do { \
        if (g_dll_mode && (mod) == em) { fn_field = dll_get_fn(name); } \
        else { hipModuleGetFunction(&(fn_field), (mod), name); } \
    } while(0)
    // Embedding kernel handles — all types
    LOAD_FN(g_compiled.eval_embed_q4_0,            em, "eval_embed_q4_0");
    LOAD_FN(g_compiled.eval_embed_q4_1,            em, "eval_embed_q4_1");
    LOAD_FN(g_compiled.eval_embed_q5_0,            em, "eval_embed_q5_0");
    LOAD_FN(g_compiled.eval_embed_q5_1,            em, "eval_embed_q5_1");
    LOAD_FN(g_compiled.eval_embed_q8_0,            em, "eval_embed_q8_0");
    LOAD_FN(g_compiled.eval_embed_q2k,             em, "eval_embed_q2k");
    LOAD_FN(g_compiled.eval_embed_q3k,             em, "eval_embed_q3k");
    LOAD_FN(g_compiled.eval_embed_q4k,             em, "eval_embed_q4k");
    LOAD_FN(g_compiled.eval_embed_q5k,             em, "eval_embed_q5k");
    LOAD_FN(g_compiled.eval_embed_q6k,             em, "eval_embed_q6k");
    LOAD_FN(g_compiled.eval_embed_iq2_xxs,         em, "eval_embed_iq2_xxs");
    LOAD_FN(g_compiled.eval_embed_iq2_xs,          em, "eval_embed_iq2_xs");
    LOAD_FN(g_compiled.eval_embed_iq2_s,           em, "eval_embed_iq2_s");
    LOAD_FN(g_compiled.eval_embed_iq3_xxs,         em, "eval_embed_iq3_xxs");
    LOAD_FN(g_compiled.eval_embed_iq3_s,           em, "eval_embed_iq3_s");
    LOAD_FN(g_compiled.eval_embed_iq1_s,           em, "eval_embed_iq1_s");
    LOAD_FN(g_compiled.eval_embed_iq1_m,           em, "eval_embed_iq1_m");
    LOAD_FN(g_compiled.eval_embed_iq4_nl,          em, "eval_embed_iq4_nl");
    LOAD_FN(g_compiled.eval_embed_iq4_xs,          em, "eval_embed_iq4_xs");
    LOAD_FN(g_compiled.eval_embed_mxfp4,           em, "eval_embed_mxfp4");
    LOAD_FN(g_compiled.eval_embed_nvfp4,          em, "eval_embed_nvfp4");
    LOAD_FN(g_compiled.eval_embed_f32,            em, "eval_embed_f32");
    LOAD_FN(g_compiled.eval_embed_f16,            em, "eval_embed_f16");
    LOAD_FN(g_compiled.eval_embed_bf16,           em, "eval_embed_bf16");
    LOAD_FN(g_compiled.eval_rmsnorm_q8,           em, "eval_rmsnorm_q8");
    LOAD_FN(g_compiled.eval_rmsnorm_q8_quantize,  em, "eval_rmsnorm_q8_quantize");
    LOAD_FN(g_compiled.eval_quantize_q8,          em, "eval_quantize_q8");
    // P0 matvec kernel handles (all types from baseline mmvq.cu)
    LOAD_FN(g_compiled.eval_matvec_q4_0,           em, "eval_matvec_q4_0");
    LOAD_FN(g_compiled.eval_matvec_q4_0_residual,  em, "eval_matvec_q4_0_residual");
    LOAD_FN(g_compiled.eval_matvec_q4_1,           em, "eval_matvec_q4_1");
    LOAD_FN(g_compiled.eval_matvec_q4_1_residual,  em, "eval_matvec_q4_1_residual");
    LOAD_FN(g_compiled.eval_matvec_q5_0,           em, "eval_matvec_q5_0");
    LOAD_FN(g_compiled.eval_matvec_q5_0_residual,  em, "eval_matvec_q5_0_residual");
    LOAD_FN(g_compiled.eval_matvec_q5_1,           em, "eval_matvec_q5_1");
    LOAD_FN(g_compiled.eval_matvec_q5_1_residual,  em, "eval_matvec_q5_1_residual");
    LOAD_FN(g_compiled.eval_matvec_q8_0,           em, "eval_matvec_q8_0");
    LOAD_FN(g_compiled.eval_matvec_q8_0_residual,  em, "eval_matvec_q8_0_residual");
    LOAD_FN(g_compiled.eval_matvec_q2k,            em, "eval_matvec_q2k");
    LOAD_FN(g_compiled.eval_matvec_q2k_residual,   em, "eval_matvec_q2k_residual");
    LOAD_FN(g_compiled.eval_matvec_q3k,            em, "eval_matvec_q3k");
    LOAD_FN(g_compiled.eval_matvec_q3k_residual,   em, "eval_matvec_q3k_residual");
    LOAD_FN(g_compiled.eval_matvec_q4k,            em, "eval_matvec_q4k");
    LOAD_FN(g_compiled.eval_matvec_q4k_residual,   em, "eval_matvec_q4k_residual");
    LOAD_FN(g_compiled.eval_matvec_q5k,            em, "eval_matvec_q5k");
    LOAD_FN(g_compiled.eval_matvec_q5k_residual,   em, "eval_matvec_q5k_residual");
    LOAD_FN(g_compiled.eval_matvec_q6k,            em, "eval_matvec_q6k");
    LOAD_FN(g_compiled.eval_matvec_q6k_residual,   em, "eval_matvec_q6k_residual");
    // IQ type kernel handles
    LOAD_FN(g_compiled.eval_matvec_iq2_xxs,          em, "eval_matvec_iq2_xxs");
    LOAD_FN(g_compiled.eval_matvec_iq2_xxs_residual, em, "eval_matvec_iq2_xxs_residual");
    LOAD_FN(g_compiled.eval_matvec_iq2_xs,           em, "eval_matvec_iq2_xs");
    LOAD_FN(g_compiled.eval_matvec_iq2_xs_residual,  em, "eval_matvec_iq2_xs_residual");
    LOAD_FN(g_compiled.eval_matvec_iq2_s,            em, "eval_matvec_iq2_s");
    LOAD_FN(g_compiled.eval_matvec_iq2_s_residual,   em, "eval_matvec_iq2_s_residual");
    LOAD_FN(g_compiled.eval_matvec_iq3_xxs,          em, "eval_matvec_iq3_xxs");
    LOAD_FN(g_compiled.eval_matvec_iq3_xxs_residual, em, "eval_matvec_iq3_xxs_residual");
    LOAD_FN(g_compiled.eval_matvec_iq3_s,            em, "eval_matvec_iq3_s");
    LOAD_FN(g_compiled.eval_matvec_iq3_s_residual,   em, "eval_matvec_iq3_s_residual");
    LOAD_FN(g_compiled.eval_matvec_iq1_s,            em, "eval_matvec_iq1_s");
    LOAD_FN(g_compiled.eval_matvec_iq1_s_residual,   em, "eval_matvec_iq1_s_residual");
    LOAD_FN(g_compiled.eval_matvec_iq1_m,            em, "eval_matvec_iq1_m");
    LOAD_FN(g_compiled.eval_matvec_iq1_m_residual,   em, "eval_matvec_iq1_m_residual");
    LOAD_FN(g_compiled.eval_matvec_iq4_nl,           em, "eval_matvec_iq4_nl");
    LOAD_FN(g_compiled.eval_matvec_iq4_nl_residual,  em, "eval_matvec_iq4_nl_residual");
    LOAD_FN(g_compiled.eval_matvec_iq4_xs,           em, "eval_matvec_iq4_xs");
    LOAD_FN(g_compiled.eval_matvec_iq4_xs_residual,  em, "eval_matvec_iq4_xs_residual");
    LOAD_FN(g_compiled.eval_matvec_mxfp4,            em, "eval_matvec_mxfp4");
    LOAD_FN(g_compiled.eval_matvec_mxfp4_residual,   em, "eval_matvec_mxfp4_residual");
    LOAD_FN(g_compiled.eval_matvec_nvfp4,            em, "eval_matvec_nvfp4");
    LOAD_FN(g_compiled.eval_matvec_nvfp4_residual,   em, "eval_matvec_nvfp4_residual");
    // 8-warp matvec variants for RDNA3 (gfx1100) — nwarps=8, block=(32,8,1)
    LOAD_FN(g_compiled.eval_matvec_q4_0_8w,           em, "eval_matvec_q4_0_8w");
    LOAD_FN(g_compiled.eval_matvec_q4_0_8w_residual,  em, "eval_matvec_q4_0_8w_residual");
    LOAD_FN(g_compiled.eval_matvec_q4_1_8w,           em, "eval_matvec_q4_1_8w");
    LOAD_FN(g_compiled.eval_matvec_q4_1_8w_residual,  em, "eval_matvec_q4_1_8w_residual");
    LOAD_FN(g_compiled.eval_matvec_q5_0_8w,           em, "eval_matvec_q5_0_8w");
    LOAD_FN(g_compiled.eval_matvec_q5_0_8w_residual,  em, "eval_matvec_q5_0_8w_residual");
    LOAD_FN(g_compiled.eval_matvec_q5_1_8w,           em, "eval_matvec_q5_1_8w");
    LOAD_FN(g_compiled.eval_matvec_q5_1_8w_residual,  em, "eval_matvec_q5_1_8w_residual");
    LOAD_FN(g_compiled.eval_matvec_q8_0_8w,           em, "eval_matvec_q8_0_8w");
    LOAD_FN(g_compiled.eval_matvec_q8_0_8w_residual,  em, "eval_matvec_q8_0_8w_residual");
    LOAD_FN(g_compiled.eval_matvec_iq4_nl_8w,         em, "eval_matvec_iq4_nl_8w");
    LOAD_FN(g_compiled.eval_matvec_iq4_nl_8w_residual, em, "eval_matvec_iq4_nl_8w_residual");
    LOAD_FN(g_compiled.eval_matvec_q4k_8w,            em, "eval_matvec_q4k_8w");
    LOAD_FN(g_compiled.eval_matvec_q4k_8w_residual,   em, "eval_matvec_q4k_8w_residual");
    LOAD_FN(g_compiled.eval_matvec_q6k_8w,            em, "eval_matvec_q6k_8w");
    LOAD_FN(g_compiled.eval_matvec_q6k_8w_residual,   em, "eval_matvec_q6k_8w_residual");
    // eval_matvec_gate_up_silu removed — replaced by separate gate + up + silu_mul launches
    LOAD_FN(g_compiled.eval_add_residual,         em, "eval_add_residual");
    LOAD_FN(g_compiled.eval_elementwise_mul,      em, "eval_elementwise_mul");
    LOAD_FN(g_compiled.eval_silu_mul,            em, "eval_silu_mul");
    LOAD_FN(g_compiled.eval_gelu_mul,            em, "eval_gelu_mul");
    LOAD_FN(g_compiled.eval_gelu,                em, "eval_gelu");
    LOAD_FN(g_compiled.eval_gelu_erf,            em, "eval_gelu_erf");
    // WavTokenizer kernels — may not exist in all .hsaco
    OPT_FN(g_compiled.eval_group_norm,  em, "eval_group_norm");
    OPT_FN(g_compiled.eval_im2col_1d,   em, "eval_im2col_1d");
    OPT_FN(g_compiled.eval_swish,       em, "eval_swish");
    OPT_FN(g_compiled.eval_conv1d_dw_ph, em, "eval_conv1d_dw_ph");
    LOAD_FN(g_compiled.eval_gelu_erf_mul,        em, "eval_gelu_erf_mul");
    LOAD_FN(g_compiled.eval_gelu_quick_mul,      em, "eval_gelu_quick_mul");
    LOAD_FN(g_compiled.eval_relu2_mul,           em, "eval_relu2_mul");
    LOAD_FN(g_compiled.eval_scale_scalar,        em, "eval_scale_scalar");
    LOAD_FN(g_compiled.eval_softcap,             em, "eval_softcap");
    LOAD_FN(g_compiled.eval_layernorm,           em, "eval_layernorm");
    LOAD_FN(g_compiled.eval_l2norm,              em, "eval_l2norm");
    LOAD_FN(g_compiled.eval_sigmoid,             em, "eval_sigmoid");
    LOAD_FN(g_compiled.eval_softplus,            em, "eval_softplus");
    LOAD_FN(g_compiled.eval_argsort_desc,        em, "eval_argsort_desc");
    LOAD_FN(g_compiled.eval_softmax_row,         em, "eval_softmax_row");
    LOAD_FN(g_compiled.eval_concat_dim0,         em, "eval_concat_dim0");
    LOAD_FN(g_compiled.eval_concat_dim1,         em, "eval_concat_dim1");
    LOAD_FN(g_compiled.eval_repeat_dim0,         em, "eval_repeat_dim0");
    LOAD_FN(g_compiled.eval_ssm_conv_step,       em, "eval_ssm_conv_step");
    LOAD_FN(g_compiled.eval_ssm_scan_step,       em, "eval_ssm_scan_step");
    LOAD_FN(g_compiled.eval_rwkv_wkv6_step,    em, "eval_rwkv_wkv6_step");
    LOAD_FN(g_compiled.eval_rwkv_wkv7_step,    em, "eval_rwkv_wkv7_step");
    LOAD_FN(g_compiled.eval_rope_neox_inplace, em, "eval_rope_neox_inplace");
    LOAD_FN(g_compiled.eval_tanh,              em, "eval_tanh");
    LOAD_FN(g_compiled.eval_neg,               em, "eval_neg");
    LOAD_FN(g_compiled.eval_exp,               em, "eval_exp");
    LOAD_FN(g_compiled.eval_relu,              em, "eval_relu");
    LOAD_FN(g_compiled.eval_sqr,               em, "eval_sqr");
    LOAD_FN(g_compiled.eval_mul_one_minus,      em, "eval_mul_one_minus");
    LOAD_FN(g_compiled.eval_repeat_interleave,  em, "eval_repeat_interleave");
    LOAD_FN(g_compiled.eval_rwkv7_rk_correction, em, "eval_rwkv7_rk_correction");
    LOAD_FN(g_compiled.eval_sample_temperature,  em, "eval_sample_temperature");
    LOAD_FN(g_compiled.eval_sample_rep_penalty,  em, "eval_sample_rep_penalty");
    LOAD_FN(g_compiled.eval_sample_argmax,       em, "eval_sample_argmax");
    LOAD_FN(g_compiled.eval_sample_top_k_p,      em, "eval_sample_top_k_p");
    // MoE matvec — K-quant types
    OPT_FN(g_compiled.eval_moe_matvec_q4k, em, "eval_moe_matvec_q4k");
    OPT_FN(g_compiled.eval_moe_matvec_q6k, em, "eval_moe_matvec_q6k");
    OPT_FN(g_compiled.eval_moe_matvec_q5k, em, "eval_moe_matvec_q5k");
    OPT_FN(g_compiled.eval_moe_matvec_q3k, em, "eval_moe_matvec_q3k");
    OPT_FN(g_compiled.eval_moe_matvec_q2k, em, "eval_moe_matvec_q2k");
    // MoE matvec — small-block types
    OPT_FN(g_compiled.eval_moe_matvec_q4_0, em, "eval_moe_matvec_q4_0");
    OPT_FN(g_compiled.eval_moe_matvec_q4_1, em, "eval_moe_matvec_q4_1");
    OPT_FN(g_compiled.eval_moe_matvec_q5_0, em, "eval_moe_matvec_q5_0");
    OPT_FN(g_compiled.eval_moe_matvec_q5_1, em, "eval_moe_matvec_q5_1");
    OPT_FN(g_compiled.eval_moe_matvec_q8_0, em, "eval_moe_matvec_q8_0");
    // MoE matvec — IQ types
    OPT_FN(g_compiled.eval_moe_matvec_iq2_xxs, em, "eval_moe_matvec_iq2_xxs");
    OPT_FN(g_compiled.eval_moe_matvec_iq2_xs, em, "eval_moe_matvec_iq2_xs");
    OPT_FN(g_compiled.eval_moe_matvec_iq2_s, em, "eval_moe_matvec_iq2_s");
    OPT_FN(g_compiled.eval_moe_matvec_iq3_xxs, em, "eval_moe_matvec_iq3_xxs");
    OPT_FN(g_compiled.eval_moe_matvec_iq3_s, em, "eval_moe_matvec_iq3_s");
    OPT_FN(g_compiled.eval_moe_matvec_iq1_s, em, "eval_moe_matvec_iq1_s");
    OPT_FN(g_compiled.eval_moe_matvec_iq1_m, em, "eval_moe_matvec_iq1_m");
    OPT_FN(g_compiled.eval_moe_matvec_iq4_nl, em, "eval_moe_matvec_iq4_nl");
    OPT_FN(g_compiled.eval_moe_matvec_iq4_xs, em, "eval_moe_matvec_iq4_xs");
    // MoE matvec — MXFP4/NVFP4
    OPT_FN(g_compiled.eval_moe_matvec_mxfp4, em, "eval_moe_matvec_mxfp4");
    OPT_FN(g_compiled.eval_moe_matvec_nvfp4, em, "eval_moe_matvec_nvfp4");
    // MoE matvec — float types
    OPT_FN(g_compiled.eval_moe_matvec_f16, em, "eval_moe_matvec_f16");
    OPT_FN(g_compiled.eval_moe_matvec_bf16, em, "eval_moe_matvec_bf16");
    OPT_FN(g_compiled.eval_moe_matvec_f32, em, "eval_moe_matvec_f32");
    // MoE utility
    OPT_FN(g_compiled.eval_moe_weighted_add, em, "eval_moe_weighted_add");
    OPT_FN(g_compiled.eval_moe_normalize_weights, em, "eval_moe_normalize_weights");
    OPT_FN(g_compiled.eval_moe_gate_mul, em, "eval_moe_gate_mul");
    OPT_FN(g_compiled.eval_moe_batch_normalize_weights, em, "eval_moe_batch_normalize_weights");
    OPT_FN(g_compiled.eval_moe_group_tokens, em, "eval_moe_group_tokens");
    OPT_FN(g_compiled.eval_moe_gather, em, "eval_moe_gather");
    OPT_FN(g_compiled.eval_moe_scatter_weighted_add, em, "eval_moe_scatter_weighted_add");
    // Fused gate+up+silu matvec — all quant types
    OPT_FN(g_compiled.eval_fused_gate_up_silu_q4_0,   em, "eval_fused_gate_up_silu_q4_0");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_q4_1,   em, "eval_fused_gate_up_silu_q4_1");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_q5_0,   em, "eval_fused_gate_up_silu_q5_0");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_q5_1,   em, "eval_fused_gate_up_silu_q5_1");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_q8_0,   em, "eval_fused_gate_up_silu_q8_0");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_q2k,    em, "eval_fused_gate_up_silu_q2k");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_q3k,    em, "eval_fused_gate_up_silu_q3k");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_q4k,    em, "eval_fused_gate_up_silu_q4k");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_q5k,    em, "eval_fused_gate_up_silu_q5k");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_q6k,    em, "eval_fused_gate_up_silu_q6k");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_iq2_xxs, em, "eval_fused_gate_up_silu_iq2_xxs");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_iq2_xs,  em, "eval_fused_gate_up_silu_iq2_xs");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_iq2_s,   em, "eval_fused_gate_up_silu_iq2_s");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_iq3_xxs, em, "eval_fused_gate_up_silu_iq3_xxs");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_iq3_s,   em, "eval_fused_gate_up_silu_iq3_s");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_iq1_s,   em, "eval_fused_gate_up_silu_iq1_s");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_iq1_m,   em, "eval_fused_gate_up_silu_iq1_m");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_iq4_nl,  em, "eval_fused_gate_up_silu_iq4_nl");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_iq4_xs,  em, "eval_fused_gate_up_silu_iq4_xs");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_mxfp4,   em, "eval_fused_gate_up_silu_mxfp4");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_nvfp4,   em, "eval_fused_gate_up_silu_nvfp4");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_f16,     em, "eval_fused_gate_up_silu_f16");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_bf16,    em, "eval_fused_gate_up_silu_bf16");
    OPT_FN(g_compiled.eval_fused_gate_up_silu_f32,     em, "eval_fused_gate_up_silu_f32");
    // Fused gate+up+GELU (tanh-approx) variants
    OPT_FN(g_compiled.eval_fused_gate_up_gelu_q4k,    em, "eval_fused_gate_up_gelu_q4k");
    OPT_FN(g_compiled.eval_fused_gate_up_gelu_q6k,    em, "eval_fused_gate_up_gelu_q6k");
    OPT_FN(g_compiled.eval_fused_gate_up_gelu_q5k,    em, "eval_fused_gate_up_gelu_q5k");
    OPT_FN(g_compiled.eval_fused_gate_up_gelu_q3k,    em, "eval_fused_gate_up_gelu_q3k");
    OPT_FN(g_compiled.eval_fused_gate_up_gelu_q2k,    em, "eval_fused_gate_up_gelu_q2k");
    OPT_FN(g_compiled.eval_fused_gate_up_gelu_q4_0,   em, "eval_fused_gate_up_gelu_q4_0");
    OPT_FN(g_compiled.eval_fused_gate_up_gelu_q8_0,   em, "eval_fused_gate_up_gelu_q8_0");
    // Fused QKV projection — 3 matvec launches → 1
    OPT_FN(g_compiled.eval_fused_qkv_matvec_q4k,    em, "eval_fused_qkv_matvec_q4k");
    OPT_FN(g_compiled.eval_fused_qkv_matvec_q6k,    em, "eval_fused_qkv_matvec_q6k");
    OPT_FN(g_compiled.eval_fused_qkv_matvec_q5k,    em, "eval_fused_qkv_matvec_q5k");
    OPT_FN(g_compiled.eval_fused_qkv_matvec_q3k,    em, "eval_fused_qkv_matvec_q3k");
    OPT_FN(g_compiled.eval_fused_qkv_matvec_q2k,    em, "eval_fused_qkv_matvec_q2k");
    OPT_FN(g_compiled.eval_fused_qkv_matvec_q4_0,   em, "eval_fused_qkv_matvec_q4_0");
    OPT_FN(g_compiled.eval_fused_qkv_matvec_q4_1,   em, "eval_fused_qkv_matvec_q4_1");
    OPT_FN(g_compiled.eval_fused_qkv_matvec_q5_0,   em, "eval_fused_qkv_matvec_q5_0");
    OPT_FN(g_compiled.eval_fused_qkv_matvec_q5_1,   em, "eval_fused_qkv_matvec_q5_1");
    OPT_FN(g_compiled.eval_fused_qkv_matvec_q8_0,   em, "eval_fused_qkv_matvec_q8_0");
    // Fused quantize+matvec — eliminates q8_act global memory round-trip
    OPT_FN(g_compiled.eval_quantize_matvec_q4k,     em, "eval_quantize_matvec_q4k");
    OPT_FN(g_compiled.eval_quantize_matvec_q6k,     em, "eval_quantize_matvec_q6k");
    OPT_FN(g_compiled.eval_quantize_matvec_q5k,     em, "eval_quantize_matvec_q5k");
    OPT_FN(g_compiled.eval_quantize_matvec_q3k,     em, "eval_quantize_matvec_q3k");
    OPT_FN(g_compiled.eval_quantize_matvec_q2k,     em, "eval_quantize_matvec_q2k");
    OPT_FN(g_compiled.eval_quantize_matvec_q4_0,    em, "eval_quantize_matvec_q4_0");
    OPT_FN(g_compiled.eval_quantize_matvec_q4_1,    em, "eval_quantize_matvec_q4_1");
    OPT_FN(g_compiled.eval_quantize_matvec_q5_0,    em, "eval_quantize_matvec_q5_0");
    OPT_FN(g_compiled.eval_quantize_matvec_q5_1,    em, "eval_quantize_matvec_q5_1");
    OPT_FN(g_compiled.eval_quantize_matvec_q8_0,    em, "eval_quantize_matvec_q8_0");
    // Fused quantize+matvec+residual variants
    OPT_FN(g_compiled.eval_quantize_matvec_residual_q4k,  em, "eval_quantize_matvec_residual_q4k");
    OPT_FN(g_compiled.eval_quantize_matvec_residual_q6k,  em, "eval_quantize_matvec_residual_q6k");
    OPT_FN(g_compiled.eval_quantize_matvec_residual_q5k,  em, "eval_quantize_matvec_residual_q5k");
    OPT_FN(g_compiled.eval_quantize_matvec_residual_q3k,  em, "eval_quantize_matvec_residual_q3k");
    OPT_FN(g_compiled.eval_quantize_matvec_residual_q2k,  em, "eval_quantize_matvec_residual_q2k");
    OPT_FN(g_compiled.eval_quantize_matvec_residual_q4_0, em, "eval_quantize_matvec_residual_q4_0");
    OPT_FN(g_compiled.eval_quantize_matvec_residual_q4_1, em, "eval_quantize_matvec_residual_q4_1");
    OPT_FN(g_compiled.eval_quantize_matvec_residual_q5_0, em, "eval_quantize_matvec_residual_q5_0");
    OPT_FN(g_compiled.eval_quantize_matvec_residual_q5_1, em, "eval_quantize_matvec_residual_q5_1");
    OPT_FN(g_compiled.eval_quantize_matvec_residual_q8_0, em, "eval_quantize_matvec_residual_q8_0");
    // Fused consecutive norms (Gemma-2/3/4)
    OPT_FN(g_compiled.eval_rmsnorm_add_rmsnorm_q8,  em, "eval_rmsnorm_add_rmsnorm_q8");
    // Multimodal
    OPT_FN(g_compiled.eval_image_patches, em, "eval_image_patches");
    OPT_FN(g_compiled.eval_stft_power, em, "eval_stft_power");
    OPT_FN(g_compiled.eval_mel_filterbank, em, "eval_mel_filterbank");
    OPT_FN(g_compiled.eval_mel_normalize, em, "eval_mel_normalize");
    OPT_FN(g_compiled.eval_rmsnorm_add, em, "eval_rmsnorm_add");
    OPT_FN(g_compiled.eval_chameleon_suppress, em, "eval_chameleon_suppress");
    OPT_FN(g_compiled.eval_t5_rel_bias_compute, em, "eval_t5_rel_bias_compute");
    OPT_FN(g_compiled.eval_fill_positions, em, "eval_fill_positions");
    OPT_FN(g_compiled.eval_max_reduce, em, "eval_max_reduce");
    LOAD_FN(g_compiled.eval_silu,              em, "eval_silu");
    LOAD_FN(g_compiled.eval_sub,               em, "eval_sub");
    LOAD_FN(g_compiled.eval_muladd,            em, "eval_muladd");
    LOAD_FN(g_compiled.eval_axpy,                em, "eval_axpy");
    LOAD_FN(g_compiled.eval_sum_row,             em, "eval_sum_row");
    LOAD_FN(g_compiled.eval_matvec_f16,          em, "eval_matvec_f16");
    LOAD_FN(g_compiled.eval_matvec_f16_residual, em, "eval_matvec_f16_residual");
    LOAD_FN(g_compiled.eval_matvec_bf16,         em, "eval_matvec_bf16");
    LOAD_FN(g_compiled.eval_matvec_bf16_residual,em, "eval_matvec_bf16_residual");
    LOAD_FN(g_compiled.eval_matvec_f32,          em, "eval_matvec_f32");
    LOAD_FN(g_compiled.eval_matvec_f32_residual, em, "eval_matvec_f32_residual");
    LOAD_FN(g_compiled.eval_write_decode_params,    em, "eval_write_decode_params");
    LOAD_FN(g_compiled.eval_qk_norm_rope_kv_write, em, "eval_qk_norm_rope_kv_write");
    // Fused gate+up matvec (baseline has_fusion path)
    OPT_FN(g_compiled.eval_matvec_glu_q4k,  em, "eval_matvec_glu_q4k");
    OPT_FN(g_compiled.eval_matvec_glu_q6k,  em, "eval_matvec_glu_q6k");
    OPT_FN(g_compiled.eval_matvec_glu_q4_0, em, "eval_matvec_glu_q4_0");
    OPT_FN(g_compiled.eval_matvec_glu_q8_0, em, "eval_matvec_glu_q8_0");
    LOAD_FN(g_compiled.eval_attention_decode,      em, "eval_attention_decode");
    LOAD_FN(g_compiled.eval_attention_decode_pb,   em, "eval_attention_decode_pb");
    LOAD_FN(g_compiled.eval_attention_combine,     em, "eval_attention_combine");
    LOAD_FN(g_compiled.eval_attention_decode_tile,  em, "eval_attention_decode_tile");
    // WMMA attention — optional, only present when compiled with rocWMMA
    OPT_FN(g_compiled.eval_attention_decode_wmma,  em, "eval_attention_decode_wmma");
    LOAD_FN(g_compiled.eval_final_norm,           em, "eval_final_norm");
    // eval_lm_head / eval_lm_head_q6k removed — LM head uses generic pick_matvec
    LOAD_FN(g_compiled.eval_dn_conv1d_silu,       em, "eval_dn_conv1d_silu");
    LOAD_FN(g_compiled.eval_dn_l2_norm,           em, "eval_dn_l2_norm");
    LOAD_FN(g_compiled.eval_dn_recurrence,        em, "eval_dn_recurrence");
    assert(g_compiled.eval_embed_q6k && "eval_embed_q6k not found");

    // Compile prefill kernels
    if (compile_kernel("prefill.hip", g_config, cache_dir, hash, &g_compiled.prompt_module) != 0) return -1;
    // Batch embed kernels — all quant types
    LOAD_FN(g_compiled.prompt_embed_q4_0,   g_compiled.prompt_module, "prompt_embed_q4_0");
    LOAD_FN(g_compiled.prompt_embed_q4_1,   g_compiled.prompt_module, "prompt_embed_q4_1");
    LOAD_FN(g_compiled.prompt_embed_q5_0,   g_compiled.prompt_module, "prompt_embed_q5_0");
    LOAD_FN(g_compiled.prompt_embed_q5_1,   g_compiled.prompt_module, "prompt_embed_q5_1");
    LOAD_FN(g_compiled.prompt_embed_q8_0,   g_compiled.prompt_module, "prompt_embed_q8_0");
    LOAD_FN(g_compiled.prompt_embed_q2k,    g_compiled.prompt_module, "prompt_embed_q2k");
    LOAD_FN(g_compiled.prompt_embed_q3k,    g_compiled.prompt_module, "prompt_embed_q3k");
    LOAD_FN(g_compiled.prompt_embed_q4k,    g_compiled.prompt_module, "prompt_embed_q4k");
    LOAD_FN(g_compiled.prompt_embed_q5k,    g_compiled.prompt_module, "prompt_embed_q5k");
    LOAD_FN(g_compiled.prompt_embed_q6k,    g_compiled.prompt_module, "prompt_embed_q6k");
    LOAD_FN(g_compiled.prompt_embed_iq2_xxs, g_compiled.prompt_module, "prompt_embed_iq2_xxs");
    LOAD_FN(g_compiled.prompt_embed_iq2_xs,  g_compiled.prompt_module, "prompt_embed_iq2_xs");
    LOAD_FN(g_compiled.prompt_embed_iq2_s,   g_compiled.prompt_module, "prompt_embed_iq2_s");
    LOAD_FN(g_compiled.prompt_embed_iq3_xxs, g_compiled.prompt_module, "prompt_embed_iq3_xxs");
    LOAD_FN(g_compiled.prompt_embed_iq3_s,   g_compiled.prompt_module, "prompt_embed_iq3_s");
    LOAD_FN(g_compiled.prompt_embed_iq1_s,   g_compiled.prompt_module, "prompt_embed_iq1_s");
    LOAD_FN(g_compiled.prompt_embed_iq1_m,   g_compiled.prompt_module, "prompt_embed_iq1_m");
    LOAD_FN(g_compiled.prompt_embed_iq4_nl,  g_compiled.prompt_module, "prompt_embed_iq4_nl");
    LOAD_FN(g_compiled.prompt_embed_iq4_xs,  g_compiled.prompt_module, "prompt_embed_iq4_xs");
    LOAD_FN(g_compiled.prompt_embed_mxfp4,   g_compiled.prompt_module, "prompt_embed_mxfp4");
    LOAD_FN(g_compiled.prompt_embed_nvfp4,   g_compiled.prompt_module, "prompt_embed_nvfp4");
    LOAD_FN(g_compiled.prompt_embed_f32,     g_compiled.prompt_module, "prompt_embed_f32");
    LOAD_FN(g_compiled.prompt_embed_f16,     g_compiled.prompt_module, "prompt_embed_f16");
    LOAD_FN(g_compiled.prompt_embed_bf16,    g_compiled.prompt_module, "prompt_embed_bf16");
    LOAD_FN(g_compiled.prompt_rmsnorm,     g_compiled.prompt_module, "prompt_rmsnorm");
    // prompt_layernorm may not exist in all .hsaco — only for BERT/LayerNorm archs
    OPT_FN(g_compiled.prompt_layernorm, g_compiled.prompt_module, "prompt_layernorm");
    OPT_FN(g_compiled.prompt_per_head_layernorm, g_compiled.prompt_module, "prompt_per_head_layernorm");
    OPT_FN(g_compiled.prompt_rope_neox_inplace, g_compiled.prompt_module, "prompt_rope_neox_inplace");
    OPT_FN(g_compiled.prompt_add_pos_embd, g_compiled.prompt_module, "prompt_add_pos_embd");
    LOAD_FN(g_compiled.prompt_add_residual, g_compiled.prompt_module, "prompt_add_residual");
    LOAD_FN(g_compiled.prompt_silu_mul,    g_compiled.prompt_module, "prompt_silu_mul");
    LOAD_FN(g_compiled.prompt_add_bias,          g_compiled.prompt_module, "prompt_add_bias");
    LOAD_FN(g_compiled.prompt_elementwise_mul,   g_compiled.prompt_module, "prompt_elementwise_mul");
    LOAD_FN(g_compiled.prompt_qk_norm_rope,      g_compiled.prompt_module, "prompt_qk_norm_rope");
    LOAD_FN(g_compiled.prompt_causal_attn, g_compiled.prompt_module, "prompt_causal_attn");
    LOAD_FN(g_compiled.prompt_bidirectional_attn, g_compiled.prompt_module, "prompt_bidirectional_attn");
    LOAD_FN(g_compiled.prompt_final_norm,  g_compiled.prompt_module, "prompt_final_norm");
    LOAD_FN(g_compiled.prompt_lm_head,     g_compiled.prompt_module, "prompt_lm_head");
    LOAD_FN(g_compiled.prompt_lm_reduce,   g_compiled.prompt_module, "prompt_lm_reduce");
    if (cfg->dn_n_heads > 0) {
        LOAD_FN(g_compiled.prompt_deltanet, g_compiled.prompt_module, "prompt_deltanet_recurrence");
    }

    // Load MMQ kernel handles from prompt module
    // Symbol names: prompt_mmq_<name>_<mmq_x>[_check]
    {
        auto & pm = g_compiled.prompt_module;
        memset(g_compiled.prompt_mmq, 0, sizeof(g_compiled.prompt_mmq));
        for (int ti = 0; ti < MMQ_NUM_TYPES; ti++) {
            const char * name = mmq_type_table[ti].name;
            for (int xi = 0; xi < 2; xi++) {
                int mmq_x = (xi == 0) ? 32 : 64;
                for (int ci = 0; ci < 2; ci++) {
                    char sym[128];
                    if (ci == 0) {
                        snprintf(sym, sizeof(sym), "prompt_mmq_%s_%d", name, mmq_x);
                    } else {
                        snprintf(sym, sizeof(sym), "prompt_mmq_%s_%d_check", name, mmq_x);
                    }
                    LOAD_FN(g_compiled.prompt_mmq[ti][xi][ci], pm, sym);
                }
            }
        }
    }

    // Load MMQ Q8_1 batch quantize kernels from decode module (they live in decode.hip)
    LOAD_FN(g_compiled.eval_quantize_mmq_q8_1_d4,   em, "eval_quantize_mmq_q8_1_d4");
    LOAD_FN(g_compiled.eval_quantize_mmq_q8_1_ds4,  em, "eval_quantize_mmq_q8_1_ds4");
    LOAD_FN(g_compiled.eval_quantize_mmq_q8_1_d2s6, em, "eval_quantize_mmq_q8_1_d2s6");

    // Load F32→F16 conversion kernel from prompt module (for rocBLAS input path)
    LOAD_FN(g_compiled.gemm_f32_to_f16, g_compiled.prompt_module, "gemm_f32_to_f16");

    // Load dequant-to-F16 kernels from prompt module (for rocBLAS weight path)
    {
        auto & pm = g_compiled.prompt_module;
        LOAD_FN(g_compiled.dequant_f16_q4_0,    pm, "prompt_dequant_q4_0_f16");
        LOAD_FN(g_compiled.dequant_f16_q4_1,    pm, "prompt_dequant_q4_1_f16");
        LOAD_FN(g_compiled.dequant_f16_q5_0,    pm, "prompt_dequant_q5_0_f16");
        LOAD_FN(g_compiled.dequant_f16_q5_1,    pm, "prompt_dequant_q5_1_f16");
        LOAD_FN(g_compiled.dequant_f16_q8_0,    pm, "prompt_dequant_q8_0_f16");
        LOAD_FN(g_compiled.dequant_f16_q2k,     pm, "prompt_dequant_q2k_f16");
        LOAD_FN(g_compiled.dequant_f16_q3k,     pm, "prompt_dequant_q3k_f16");
        LOAD_FN(g_compiled.dequant_f16_q4k,     pm, "prompt_dequant_q4k_f16");
        LOAD_FN(g_compiled.dequant_f16_q5k,     pm, "prompt_dequant_q5k_f16");
        LOAD_FN(g_compiled.dequant_f16_q6k,     pm, "prompt_dequant_q6k_f16");
        LOAD_FN(g_compiled.dequant_f16_iq2_xxs, pm, "prompt_dequant_iq2_xxs_f16");
        LOAD_FN(g_compiled.dequant_f16_iq2_xs,  pm, "prompt_dequant_iq2_xs_f16");
        LOAD_FN(g_compiled.dequant_f16_iq2_s,   pm, "prompt_dequant_iq2_s_f16");
        LOAD_FN(g_compiled.dequant_f16_iq3_xxs, pm, "prompt_dequant_iq3_xxs_f16");
        LOAD_FN(g_compiled.dequant_f16_iq3_s,   pm, "prompt_dequant_iq3_s_f16");
        LOAD_FN(g_compiled.dequant_f16_iq1_s,   pm, "prompt_dequant_iq1_s_f16");
        LOAD_FN(g_compiled.dequant_f16_iq1_m,   pm, "prompt_dequant_iq1_m_f16");
        LOAD_FN(g_compiled.dequant_f16_iq4_nl,  pm, "prompt_dequant_iq4_nl_f16");
        LOAD_FN(g_compiled.dequant_f16_iq4_xs,  pm, "prompt_dequant_iq4_xs_f16");
        LOAD_FN(g_compiled.dequant_f16_mxfp4,   pm, "prompt_dequant_mxfp4_f16");
        LOAD_FN(g_compiled.dequant_f16_nvfp4,   pm, "prompt_dequant_nvfp4_f16");
        LOAD_FN(g_compiled.dequant_f16_f32,     pm, "prompt_dequant_f32_f16");
        LOAD_FN(g_compiled.dequant_f16_bf16,    pm, "prompt_dequant_bf16_f16");
    }

    #undef LOAD_FN
    #undef OPT_FN
    if (load_errors > 0) {
        fprintf(stderr, "gfx1100: FATAL — %d kernel(s) failed to load\n", load_errors);
        return -1;
    }
    g_compiled.valid = true;

    // Allocate buffers
    auto & b = g_bufs;
    hipStreamCreate(&b.stream);

    int H = cfg->hidden_size;
    int FF = cfg->intermediate_size;
    int V = cfg->vocab_size;
    int fa_q_size = cfg->fa_n_q_heads * cfg->fa_head_dim;
    int fa_kv_size = cfg->fa_n_kv_heads * cfg->fa_head_dim;
    int fa_qproj = cfg->fa_has_gated_attn ? fa_q_size * 2 : fa_q_size;
    int dn_qk = cfg->dn_n_k_heads * cfg->dn_key_dim;
    int dn_v  = cfg->dn_n_heads * cfg->dn_value_dim;
    int dn_conv_ch = dn_qk * 2 + dn_v;
    int max_proj = (fa_qproj > dn_conv_ch) ? fa_qproj : dn_conv_ch;
    // proj_scratch is reused for FFN up projection (FF outputs) — must hold max(max_proj, 2*FF)
    // 2*FF needed for Phi3/ChatGLM fused SwiGLU where up produces 2*FF values
    int max_scratch = (max_proj > 2 * FF) ? max_proj : 2 * FF;
    // q8_act must hold max(H, FF, 2*FF for SwiGLU, ssm_d_inner for Mamba)
    int max_dim = H;
    if (FF > max_dim) max_dim = FF;
    if (2 * FF > max_dim) max_dim = 2 * FF;
    if (cfg->ssm_d_inner > max_dim) max_dim = cfg->ssm_d_inner;
    int q8_size = (max_dim / 32 + 1) * 36; // Q8_1 blocks

    // --- VRAM budget check ---
    // Sum all planned allocations (activation buffers, batch buffers, arch-specific state).
    // Does NOT include model weights (already on GPU) or KV cache (managed by llama.cpp).
    {
        size_t budget = 0;

        // Core decode activation buffers (lines below)
        budget += (size_t)H * 4;              // hidden
        budget += (size_t)H * 4;              // residual
        budget += (size_t)H * 4;              // norm_out
        budget += (size_t)q8_size;            // q8_act
        budget += (size_t)max_scratch * 4;    // proj_scratch
        budget += (size_t)fa_kv_size * 2 * 4; // kv_scratch
        budget += (size_t)fa_q_size * 4;      // attn_out
        budget += (size_t)FF * 4;             // mlp_inter
        budget += (size_t)V * 4;              // logits

        // DeltaNet buffers (conditional)
        if (cfg->dn_n_heads > 0) {
            budget += (size_t)dn_v * 4;                        // z_scratch
            budget += (size_t)cfg->dn_n_heads * 4;             // beta_scratch
            budget += (size_t)cfg->dn_n_heads * 4;             // alpha_scratch
            int n_dn = 0;
            for (int i = 0; i < cfg->n_layers; i++) if (cfg->layer_types[i] == 1) n_dn++;
            budget += (size_t)n_dn * cfg->dn_n_heads * cfg->dn_key_dim * cfg->dn_value_dim * 4; // dn_states
            budget += (size_t)n_dn * dn_conv_ch * cfg->dn_conv_kernel * 4;                       // conv_bufs
        }

        // RWKV buffers (conditional)
        if (cfg->wkv_head_size > 0) {
            int hs = cfg->wkv_head_size;
            int nh = H / hs;
            int nl = cfg->n_layers;
            int lora = cfg->rwkv_lora_size > 0 ? cfg->rwkv_lora_size : 64;
            budget += (size_t)nl * H * sizeof(float);          // rwkv_att_shift
            budget += (size_t)nl * H * sizeof(float);          // rwkv_ffn_shift
            budget += (size_t)nl * nh * hs * hs * sizeof(float); // rwkv_wkv_state
            budget += (size_t)H * sizeof(float);               // rwkv_sx
            budget += (size_t)H * 6 * sizeof(float);           // rwkv_xxx
            budget += (size_t)(H > lora * 5 ? H : lora * 5) * sizeof(float); // rwkv_xw
            budget += (size_t)H * sizeof(float);               // rwkv_xk
            budget += (size_t)H * sizeof(float);               // rwkv_xv
            budget += (size_t)H * sizeof(float);               // rwkv_xr
            budget += (size_t)H * sizeof(float);               // rwkv_xg
            budget += (size_t)H * sizeof(float);               // rwkv7_v_first
        }

        // SSM/Mamba buffers (conditional)
        if (cfg->ssm_d_inner > 0) {
            int di = cfg->ssm_d_inner, ds = cfg->ssm_d_state, dc = cfg->ssm_d_conv;
            int nl = cfg->n_layers;
            int n_ssm = 0;
            for (int i = 0; i < nl; i++) if (cfg->layer_types[i] == 2) n_ssm++;
            if (n_ssm == 0) n_ssm = nl;
            int conv_width = di;
            if (cfg->ssm_n_group > 0) conv_width = di + 2 * cfg->ssm_n_group * ds;
            int xz_size_budget = (2 * di > conv_width) ? 2 * di : conv_width;
            budget += (size_t)n_ssm * conv_width * (dc - 1) * sizeof(float); // ssm_conv_states
            budget += (size_t)n_ssm * di * ds * sizeof(float);               // ssm_scan_states
            budget += (size_t)xz_size_budget * sizeof(float);                // ssm_xz
            budget += (size_t)(cfg->ssm_dt_rank + 2 * ds) * sizeof(float);  // ssm_x_db
            budget += (size_t)di * sizeof(float);                            // ssm_dt
        }

        // Misc fixed-size buffers
        budget += sizeof(unsigned int);  // barrier_counter
        budget += sizeof(unsigned int);  // barrier_gen

        // T5 relative position bias (conditional)
        if (cfg->n_rel_attn_bkts > 0) {
            budget += (size_t)cfg->fa_n_q_heads * cfg->max_seq_len * sizeof(float);
        }

        // Layer weights array + KV cache pointer arrays
        budget += (size_t)cfg->n_layers * sizeof(gfx1100_layer_weights);
        int n_attn_budget = 0;
        for (int i = 0; i < cfg->n_layers; i++) if (cfg->layer_types[i] == 0) n_attn_budget++;
        budget += (size_t)n_attn_budget * sizeof(void *); // d_k_cache_ptrs
        budget += (size_t)n_attn_budget * sizeof(void *); // d_v_cache_ptrs

        // LM head scratch
        budget += 512 * sizeof(float);  // lm_block_maxv
        budget += 512 * sizeof(int);    // lm_block_maxi
        budget += sizeof(int);          // output_token

        // Batch buffers for prompt processing
        {
            int S = cfg->max_seq_len;
            int max_q8_dim_b = H;
            if (FF > max_q8_dim_b) max_q8_dim_b = FF;
            if (fa_q_size > max_q8_dim_b) max_q8_dim_b = fa_q_size;
            size_t q8_mmq_per_tok_b = ((size_t)max_q8_dim_b / 128 + 1) * 144;
            int batch_proj_dim_b = max_proj > FF ? max_proj : FF;
            int batch_attn_dim_b = fa_q_size;
            if (dn_v > batch_attn_dim_b) batch_attn_dim_b = dn_v;
            if (H > batch_attn_dim_b) batch_attn_dim_b = H;

            budget += (size_t)S * H * sizeof(float);                    // batch_hidden
            budget += (size_t)S * H * sizeof(float);                    // batch_norm
            budget += (size_t)S * H * sizeof(float);                    // batch_residual
            budget += (size_t)S * q8_mmq_per_tok_b;                     // batch_q8_mmq
            budget += (size_t)S * batch_proj_dim_b * sizeof(float);     // batch_proj
            budget += (size_t)S * fa_kv_size * 2 * sizeof(float);       // batch_kv
            budget += (size_t)S * batch_attn_dim_b * sizeof(float);     // batch_attn_out
            budget += (size_t)S * FF * sizeof(float);                   // batch_mlp
            budget += (size_t)S * sizeof(int);                          // batch_token_ids
        }

        // MoE buffers (decode path — single token)
        budget += 256 * sizeof(int);    // moe_sorted_ids
        budget += 256 * sizeof(float);  // moe_probs

        // Batched MoE buffers (prompt path — group-by-expert dispatch)
        if (cfg->has_moe) {
            int moe_ne = cfg->moe_n_experts > 0 ? cfg->moe_n_experts : 8;
            int moe_nu = cfg->moe_n_experts_used > 0 ? cfg->moe_n_experts_used : 2;
            int moe_S = cfg->max_seq_len;
            budget += (size_t)moe_S * moe_ne * sizeof(float);     // batch_moe_probs
            budget += (size_t)moe_S * moe_ne * sizeof(int);       // batch_moe_sorted
            budget += (size_t)moe_ne * sizeof(int);                // moe_expert_counts
            budget += (size_t)moe_S * moe_nu * 2 * sizeof(int);  // moe_token_map
        }

        // Query available VRAM
        size_t free_mem = 0, total_mem = 0;
        hipMemGetInfo(&free_mem, &total_mem);

        const size_t safety_margin = (size_t)256 * 1048576; // 256 MB for .hsaco code, driver overhead, etc.

        if (budget + safety_margin > free_mem) {
            fprintf(stderr, "gfx1100: VRAM budget EXCEEDED — cannot allocate buffers\n");
            fprintf(stderr, "  planned allocation:  %zu MB\n", budget / 1048576);
            fprintf(stderr, "  safety margin:       %zu MB (hsaco + driver overhead)\n", safety_margin / 1048576);
            fprintf(stderr, "  total needed:        %zu MB\n", (budget + safety_margin) / 1048576);
            fprintf(stderr, "  free VRAM:           %zu MB\n", free_mem / 1048576);
            fprintf(stderr, "  total VRAM:          %zu MB\n", total_mem / 1048576);
            fprintf(stderr, "  shortfall:           %zu MB\n", (budget + safety_margin - free_mem) / 1048576);
            return -1;
        }

        fprintf(stderr, "gfx1100: VRAM budget OK — %zu MB planned + %zu MB margin < %zu MB free\n",
                budget / 1048576, safety_margin / 1048576, free_mem / 1048576);
    }

    hipMalloc(&b.hidden,       H * 4);
    hipMalloc(&b.residual,     H * 4);
    hipMalloc(&b.norm_out,     H * 4);
    hipMalloc(&b.q8_act,       q8_size);
    hipMalloc(&b.proj_scratch, max_scratch * 4);
    hipMalloc(&b.kv_scratch,   fa_kv_size * 2 * 4);
    hipMalloc(&b.attn_out,     fa_q_size * 4);
    hipMalloc(&b.mlp_inter,    FF * 4);
    hipMalloc(&b.logits,       V * 4);

    // Ensure embedding table is on GPU — ggml may place tok_embd on CPU.
    // Baseline relies on ggml scheduler to copy; we bypass it, so copy once here.
    {
        hipPointerAttribute_t attr = {};
        hipError_t pe = hipPointerGetAttributes(&attr, (void *)cfg->embed_weight);
        bool on_device = (pe == hipSuccess && (attr.type == hipMemoryTypeDevice || attr.type == hipMemoryTypeUnified));
        if (!on_device && cfg->embed_weight) {
            size_t embed_bytes = (size_t)cfg->vocab_size * cfg->embed_stride;
            void * gpu_embed = nullptr;
            hipMalloc(&gpu_embed, embed_bytes);
            hipMemcpy(gpu_embed, cfg->embed_weight, embed_bytes, hipMemcpyHostToDevice);
            g_config.embed_weight = gpu_embed;
            fprintf(stderr, "gfx1100-megakernel: copied embedding table to GPU (%.1f MiB)\n",
                    embed_bytes / (1024.0 * 1024.0));
        }
    }

    // Ensure rope_freq_factors is on GPU (same check)
    if (cfg->rope_freq_factors) {
        hipPointerAttribute_t attr = {};
        hipError_t pe = hipPointerGetAttributes(&attr, (void *)cfg->rope_freq_factors);
        bool on_device = (pe == hipSuccess && (attr.type == hipMemoryTypeDevice || attr.type == hipMemoryTypeUnified));
        if (!on_device) {
            int n_factors = cfg->fa_rope_dim / 2;
            float * gpu_ff = nullptr;
            hipMalloc(&gpu_ff, n_factors * sizeof(float));
            hipMemcpy(gpu_ff, cfg->rope_freq_factors, n_factors * sizeof(float), hipMemcpyHostToDevice);
            g_config.rope_freq_factors = gpu_ff;
        }
    }

    // Ensure model-level pointers are on GPU
    {
        auto check_gpu = [](const void * ptr, const char * name) {
            if (!ptr) return;
            hipPointerAttribute_t attr = {};
            hipError_t pe = hipPointerGetAttributes(&attr, (void *)ptr);
            bool on_device = (pe == hipSuccess &&
                (attr.type == hipMemoryTypeDevice || attr.type == hipMemoryTypeUnified));
            if (!on_device) {
                fprintf(stderr, "gfx1100-megakernel: FATAL — %s is on CPU (ptr=%p). "
                        "All weights must be on GPU.\n", name, ptr);
            }
        };
        check_gpu(g_config.final_norm_weight, "final_norm_weight");
        check_gpu(g_config.lm_head_weight, "lm_head_weight");
        if (g_config.output_norm_enc) check_gpu(g_config.output_norm_enc, "output_norm_enc");
        if (g_config.pos_embd)        check_gpu(g_config.pos_embd, "pos_embd");
        if (g_config.type_embd)       check_gpu(g_config.type_embd, "type_embd");

        // For weight-tied models: lm_head == embed (both originally CPU).
        // embed_weight was copied to GPU above, but lm_head_weight still points to old CPU ptr.
        if (g_config.lm_head_weight) {
            hipPointerAttribute_t attr = {};
            hipError_t pe = hipPointerGetAttributes(&attr, (void *)g_config.lm_head_weight);
            bool on_device = (pe == hipSuccess &&
                (attr.type == hipMemoryTypeDevice || attr.type == hipMemoryTypeUnified));
            if (!on_device) {
                // Weight-tied: copy entire LM head to GPU
                size_t lm_bytes = (size_t)cfg->vocab_size * cfg->lm_head_stride;
                void * gpu_lm = nullptr;
                hipMalloc(&gpu_lm, lm_bytes);
                hipMemcpy(gpu_lm, g_config.lm_head_weight, lm_bytes, hipMemcpyHostToDevice);
                g_config.lm_head_weight = gpu_lm;
                fprintf(stderr, "gfx1100-megakernel: copied lm_head_weight to GPU (%.1f MiB)\n",
                        lm_bytes / (1024.0 * 1024.0));
            }
        }

        // Verify all per-layer weight matrices are on GPU (should always be with ngl=-1)
        for (int il = 0; il < cfg->n_layers; il++) {
            for (int s = 0; s < 16; s++) {
                if (g_config.layers[il].ptrs[s]) {
                    hipPointerAttribute_t attr = {};
                    hipError_t pe = hipPointerGetAttributes(&attr, (void *)g_config.layers[il].ptrs[s]);
                    bool on_device = (pe == hipSuccess &&
                        (attr.type == hipMemoryTypeDevice || attr.type == hipMemoryTypeUnified));
                    if (!on_device) {
                        fprintf(stderr, "gfx1100-megakernel: FATAL — layers[%d].ptrs[%d] is on CPU. "
                                "Model must be fully offloaded to GPU.\n", il, s);
                        return -1;
                    }
                }
            }
        }
    }

    // Ensure all per-layer optional tensors (biases/scales) are on GPU.
    // These may be on CPU due to mmap fallback or small tensor placement.
    {
        int qproj_size = cfg->fa_has_gated_attn ? fa_q_size * 2 : fa_q_size;
        int kv_size_bias = fa_kv_size;
        for (int il = 0; il < cfg->n_layers; il++) {
            auto ensure_gpu = [](const void *& ptr, size_t n_bytes, const char * name, int layer) {
                if (!ptr) return;
                hipPointerAttribute_t attr = {};
                hipError_t pe = hipPointerGetAttributes(&attr, (void *)ptr);
                bool on_device = (pe == hipSuccess &&
                    (attr.type == hipMemoryTypeDevice || attr.type == hipMemoryTypeUnified));
                if (!on_device) {
                    void * gpu_copy = nullptr;
                    hipMalloc(&gpu_copy, n_bytes);
                    hipMemcpy(gpu_copy, ptr, n_bytes, hipMemcpyHostToDevice);
                    ptr = gpu_copy;
                    fprintf(stderr, "gfx1100-megakernel: copied %s[%d] to GPU (%zu bytes)\n",
                            name, layer, n_bytes);
                }
            };
            ensure_gpu(g_config.layers[il].bias_q, qproj_size * sizeof(float),   "bias_q", il);
            ensure_gpu(g_config.layers[il].bias_k, kv_size_bias * sizeof(float),  "bias_k", il);
            ensure_gpu(g_config.layers[il].bias_v, kv_size_bias * sizeof(float),  "bias_v", il);
            ensure_gpu(g_config.layers[il].bias_o, H * sizeof(float),             "bias_o", il);
            ensure_gpu(g_config.layers[il].scale_q, qproj_size * sizeof(float),   "scale_q", il);
            ensure_gpu(g_config.layers[il].scale_k, kv_size_bias * sizeof(float), "scale_k", il);
            ensure_gpu(g_config.layers[il].scale_v, kv_size_bias * sizeof(float), "scale_v", il);
            ensure_gpu(g_config.layers[il].scale_o, H * sizeof(float),            "scale_o", il);
            ensure_gpu(g_config.layers[il].ffn_gate_bias, FF * sizeof(float),     "ffn_gate_b", il);
            ensure_gpu(g_config.layers[il].ffn_up_bias,   FF * sizeof(float),     "ffn_up_b", il);
            ensure_gpu(g_config.layers[il].ffn_down_bias, H * sizeof(float),      "ffn_down_b", il);
            ensure_gpu(g_config.layers[il].ffn_gate_scale, FF * sizeof(float),    "ffn_gate_s", il);
            ensure_gpu(g_config.layers[il].ffn_up_scale,   FF * sizeof(float),    "ffn_up_s", il);
            ensure_gpu(g_config.layers[il].ffn_down_scale, H * sizeof(float),     "ffn_down_s", il);

            // T5 encoder weights — baseline: build_t5_enc (only present for T5/Flan-T5)
            ensure_gpu(g_config.layers[il].attn_norm_enc,  H * sizeof(float),     "attn_norm_enc", il);
            ensure_gpu(g_config.layers[il].ffn_norm_enc,   H * sizeof(float),     "ffn_norm_enc", il);

            // T5 relative position bias — [n_head, n_rel_attn_bkts] f32
            // Only layer 0 has it; all other layers fall back to layer 0's table.
            if (cfg->n_rel_attn_bkts > 0) {
                int rel_b_elems = cfg->fa_n_q_heads * cfg->n_rel_attn_bkts;
                ensure_gpu(g_config.layers[il].attn_rel_b,     rel_b_elems * sizeof(float), "attn_rel_b", il);
                ensure_gpu(g_config.layers[il].attn_rel_b_enc, rel_b_elems * sizeof(float), "attn_rel_b_enc", il);
            }

            // T5 cross-attention weights — baseline: build_t5_dec
            ensure_gpu(g_config.layers[il].attn_norm_cross, H * sizeof(float),    "attn_norm_cross", il);

            // BERT post-norm weights — baseline: build_bert
            ensure_gpu(g_config.layers[il].attn_out_norm,   H * sizeof(float),    "attn_out_norm", il);
            ensure_gpu(g_config.layers[il].attn_out_norm_b, H * sizeof(float),    "attn_out_norm_b", il);
            ensure_gpu(g_config.layers[il].layer_out_norm,  H * sizeof(float),    "layer_out_norm", il);
            ensure_gpu(g_config.layers[il].layer_out_norm_b,H * sizeof(float),    "layer_out_norm_b", il);
            ensure_gpu(g_config.layers[il].attn_norm_2,     H * sizeof(float),    "attn_norm_2", il);
            ensure_gpu(g_config.layers[il].attn_norm_2_b,   H * sizeof(float),    "attn_norm_2_b", il);
        }
    }

    if (cfg->dn_n_heads > 0) {
        hipMalloc(&b.z_scratch,     dn_v * 4);
        hipMalloc(&b.beta_scratch,  cfg->dn_n_heads * 4);
        hipMalloc(&b.alpha_scratch, cfg->dn_n_heads * 4);

        int n_dn = 0;
        for (int i = 0; i < cfg->n_layers; i++) if (cfg->layer_types[i] == 1) n_dn++;
        size_t state_bytes = (size_t)n_dn * cfg->dn_n_heads * cfg->dn_key_dim * cfg->dn_value_dim * 4;
        size_t conv_bytes  = (size_t)n_dn * dn_conv_ch * cfg->dn_conv_kernel * 4;
        hipMalloc(&b.dn_states, state_bytes);
        hipMalloc(&b.conv_bufs, conv_bytes);
        hipMemset(b.dn_states, 0, state_bytes);
        hipMemset(b.conv_bufs, 0, conv_bytes);
    }

    // RWKV persistent state + scratch buffers
    if (cfg->wkv_head_size > 0) {
        int hs = cfg->wkv_head_size;
        int nh = H / hs;
        int nl = cfg->n_layers;
        hipMalloc(&b.rwkv_att_shift,  (size_t)nl * H * sizeof(float));
        hipMalloc(&b.rwkv_ffn_shift,  (size_t)nl * H * sizeof(float));
        hipMalloc(&b.rwkv_wkv_state,  (size_t)nl * nh * hs * hs * sizeof(float));
        hipMemset(b.rwkv_att_shift, 0, (size_t)nl * H * sizeof(float));
        hipMemset(b.rwkv_ffn_shift, 0, (size_t)nl * H * sizeof(float));
        hipMemset(b.rwkv_wkv_state, 0, (size_t)nl * nh * hs * hs * sizeof(float));
        // Scratch buffers
        int lora = cfg->rwkv_lora_size > 0 ? cfg->rwkv_lora_size : 64;
        hipMalloc(&b.rwkv_sx,   H * sizeof(float));
        hipMalloc(&b.rwkv_xxx,  H * 6 * sizeof(float));  // up to 6 components (r,w,k,v,a,g)
        hipMalloc(&b.rwkv_xw,   (H > lora * 5 ? H : lora * 5) * sizeof(float));
        hipMalloc(&b.rwkv_xk,   H * sizeof(float));
        hipMalloc(&b.rwkv_xv,   H * sizeof(float));
        hipMalloc(&b.rwkv_xr,   H * sizeof(float));
        hipMalloc(&b.rwkv_xg,   H * sizeof(float));
        hipMalloc(&b.rwkv7_v_first, H * sizeof(float));
        b.rwkv7_v_first_set = false;
        fprintf(stderr, "gfx1100-megakernel: allocated RWKV buffers (hs=%d, nh=%d)\n", hs, nh);
    }

    // SSM/Mamba persistent state + scratch buffers
    if (cfg->ssm_d_inner > 0) {
        int di = cfg->ssm_d_inner, ds = cfg->ssm_d_state, dc = cfg->ssm_d_conv;
        int nl = cfg->n_layers;
        int n_ssm = 0;
        for (int i = 0; i < nl; i++) if (cfg->layer_types[i] == 2) n_ssm++;
        if (n_ssm == 0) n_ssm = nl;  // pure Mamba: all layers are SSM
        // For Mamba2: conv state width is wider (d_inner + 2*n_group*d_state)
        int conv_width = di;
        if (cfg->ssm_n_group > 0) conv_width = di + 2 * cfg->ssm_n_group * ds;
        hipMalloc(&b.ssm_conv_states, (size_t)n_ssm * conv_width * (dc - 1) * sizeof(float));
        hipMalloc(&b.ssm_scan_states, (size_t)n_ssm * di * ds * sizeof(float));
        // ssm_xz: Mamba1 uses 2*d_inner (x + z), Mamba2 copies xBC_dim = conv_width into it
        int xz_size = (2 * di > conv_width) ? 2 * di : conv_width;
        hipMalloc(&b.ssm_xz,          xz_size * sizeof(float));
        hipMalloc(&b.ssm_x_db,        (cfg->ssm_dt_rank + 2 * ds) * sizeof(float));
        hipMalloc(&b.ssm_dt,          di * sizeof(float));
        hipMemset(b.ssm_conv_states, 0, (size_t)n_ssm * conv_width * (dc - 1) * sizeof(float));
        hipMemset(b.ssm_scan_states, 0, (size_t)n_ssm * di * ds * sizeof(float));
        fprintf(stderr, "gfx1100-megakernel: allocated SSM buffers (d_inner=%d, d_state=%d, n_ssm=%d)\n",
                di, ds, n_ssm);
    }

    hipMalloc(&b.barrier_counter, sizeof(unsigned int));
    hipMalloc(&b.barrier_gen,     sizeof(unsigned int));

    // T5 relative position bias scratch buffer
    if (cfg->n_rel_attn_bkts > 0) {
        size_t rel_bias_bytes = (size_t)cfg->fa_n_q_heads * cfg->max_seq_len * sizeof(float);
        hipMalloc(&b.d_rel_pos_bias, rel_bias_bytes);
        fprintf(stderr, "gfx1100-megakernel: allocated T5 rel_pos_bias buffer (%.1f KiB)\n",
                rel_bias_bytes / 1024.0);
    }

    // Copy layer weights to device
    hipMalloc(&b.d_layer_weights, cfg->n_layers * sizeof(gfx1100_layer_weights));
    hipMemcpy(b.d_layer_weights, cfg->layers, cfg->n_layers * sizeof(gfx1100_layer_weights), hipMemcpyHostToDevice);

    // Copy KV cache pointer arrays to device
    int n_attn = 0;
    for (int i = 0; i < cfg->n_layers; i++) if (cfg->layer_types[i] == 0) n_attn++;
    hipMalloc(&b.d_k_cache_ptrs, n_attn * sizeof(void *));
    hipMalloc(&b.d_v_cache_ptrs, n_attn * sizeof(void *));
    hipMemcpy(b.d_k_cache_ptrs, cfg->k_cache_ptrs, n_attn * sizeof(void *), hipMemcpyHostToDevice);
    hipMemcpy(b.d_v_cache_ptrs, cfg->v_cache_ptrs, n_attn * sizeof(void *), hipMemcpyHostToDevice);

    // LM head scratch
    hipMalloc(&b.lm_block_maxv, 512 * sizeof(float));
    hipMalloc(&b.lm_block_maxi, 512 * sizeof(int));
    hipMalloc(&b.output_token,  sizeof(int));

    // Batch buffers for prompt processing
    {
        int S = cfg->max_seq_len;
        b.max_batch = S;
        // block_q8_1_mmq is 144 bytes per 128 values (4*QK8_1 + 16 metadata)
        // Must hold max(H, FF, fa_q_size) per token — FFN down quantize uses FF as in_dim
        int max_q8_dim = H;
        if (FF > max_q8_dim) max_q8_dim = FF;
        if (fa_q_size > max_q8_dim) max_q8_dim = fa_q_size;
        size_t q8_mmq_per_tok = ((size_t)max_q8_dim / 128 + 1) * 144;

        // batch_proj must hold max(max_proj, FF) — FFN up writes FF elements per token
        int batch_proj_dim = max_proj > FF ? max_proj : FF;

        // batch_attn_out must hold max(fa_q_size, dn_v_size, H) — DeltaNet Z and MoE gather reuse this buffer
        int batch_attn_dim = fa_q_size;
        if (dn_v > batch_attn_dim) batch_attn_dim = dn_v;
        if (H > batch_attn_dim) batch_attn_dim = H;

        hipMalloc(&b.batch_hidden,    (size_t)S * H * sizeof(float));
        hipMalloc(&b.batch_norm,      (size_t)S * H * sizeof(float));
        hipMalloc(&b.batch_residual,  (size_t)S * H * sizeof(float));
        hipMalloc(&b.batch_q8_mmq,    (size_t)S * q8_mmq_per_tok);
        hipMalloc(&b.batch_proj,      (size_t)S * batch_proj_dim * sizeof(float));
        hipMalloc(&b.batch_kv,        (size_t)S * fa_kv_size * 2 * sizeof(float));
        hipMalloc(&b.batch_attn_out,  (size_t)S * batch_attn_dim * sizeof(float));
        hipMalloc(&b.batch_mlp,       (size_t)S * FF * sizeof(float));
        hipMalloc(&b.batch_token_ids, (size_t)S * sizeof(int));
        fprintf(stderr, "gfx1100-megakernel: allocated batch buffers for max_batch=%d\n", S);
    }

    // GPU-resident decode params for hipGraph reuse
    hipMalloc(&b.d_decode_params, 3 * sizeof(int));

    // Parallel-block attention scratch — driven by TUNE meta-step (g_comp_tuning).
    // All sizing logic + env-var override live in composition/comp-tune.h.
    {
        const int pb = g_comp_tuning.attn_parallel_blocks;
        const int n_q = cfg->fa_n_q_heads;
        const int D   = cfg->fa_head_dim;
        b.attn_parallel_blocks = pb;
        if (pb > 1) {
            HIP_ASSERT(hipMalloc(&b.attn_partial, (size_t)n_q * pb * D * sizeof(float)));
            HIP_ASSERT(hipMalloc(&b.attn_meta,    (size_t)n_q * pb * 2 * sizeof(float)));
            fprintf(stderr, "gfx1100: parallel attention: %d blocks/head (%d heads * %d pb = %d total blocks)\n",
                    pb, n_q, pb, n_q * pb);
        } else {
            b.attn_partial = nullptr;
            b.attn_meta = nullptr;
        }
    }

    // MoE dedicated buffers — separate from mlp_inter/proj_scratch to avoid aliasing
    hipMalloc(&b.moe_sorted_ids, 256 * sizeof(int));
    hipMalloc(&b.moe_probs,      256 * sizeof(float));

    // Batched MoE buffers (prompt path — group-by-expert dispatch)
    if (cfg->has_moe && b.max_batch > 0) {
        int moe_S  = b.max_batch;
        int moe_ne = cfg->moe_n_experts > 0 ? cfg->moe_n_experts : 8;
        int moe_nu = cfg->moe_n_experts_used > 0 ? cfg->moe_n_experts_used : 2;
        hipMalloc(&b.batch_moe_probs,   (size_t)moe_S * moe_ne * sizeof(float));
        hipMalloc(&b.batch_moe_sorted,  (size_t)moe_S * moe_ne * sizeof(int));
        hipMalloc(&b.moe_expert_counts, (size_t)moe_ne * sizeof(int));
        hipMalloc(&b.moe_token_map,     (size_t)moe_S * moe_nu * 2 * sizeof(int));
    } else {
        b.batch_moe_probs = nullptr;
        b.batch_moe_sorted = nullptr;
        b.moe_expert_counts = nullptr;
        b.moe_token_map = nullptr;
    }

    // --- VRAM post-allocation summary ---
    {
        size_t free_mem = 0, total_mem = 0;
        hipMemGetInfo(&free_mem, &total_mem);
        size_t allocated = total_mem - free_mem;
        fprintf(stderr, "gfx1100: VRAM budget: %zu MB allocated / %zu MB free / %zu MB total\n",
                allocated / 1048576, free_mem / 1048576, total_mem / 1048576);
    }

    // Initialize rocBLAS state for prompt GEMM fallback
    {
        int max_in = (H > FF) ? H : FF;
        int max_out_dim = (V > FF) ? V : FF;
        max_out_dim = (max_out_dim > fa_qproj) ? max_out_dim : fa_qproj;
        int max_batch = cfg->max_seq_len;
        if (rocblas_gemm_init(&g_rocblas, max_in, max_out_dim, max_batch) != 0) {
            fprintf(stderr, "gfx1100-megakernel: WARNING — rocBLAS init failed, prompt will use MMQ only\n");
        }
    }

    // YaRN RoPE — handled by compile-time ROPE_THETA, rope freq_factors, and the
    // RoPE kernel which already supports ext_factor/attn_factor/freq_scale via the
    // fused QK-norm-RoPE kernel parameters. No longer FATAL.

    // Custom attention scale — handled by compile-time FA_ATTN_SCALE flag.
    // The attention kernel uses 1/sqrt(d) by default; custom scale overrides it.
    // No longer FATAL — the .hsaco is compiled with the correct scale for this model.
    if (cfg->fa_attention_scale != 0.0f) {
        float expected = 1.0f / sqrtf((float)cfg->fa_head_dim);
        if (fabsf(cfg->fa_attention_scale - expected) > 1e-6f) {
            fprintf(stderr, "gfx1100-megakernel: custom attention scale %.6f (default would be %.6f)\n",
                    cfg->fa_attention_scale, expected);
        }
    }
    // DeepSeek2 YaRN mscale: the caller should compute
    //   mscale = attn_factor * (1 + 0.1 * log_mul * log(1/freq_scale))
    //   fa_attention_scale = mscale^2 / sqrt(head_dim)
    // and set it in cfg->fa_attention_scale before calling init.
    // The .hsaco is compiled with this scale baked in.
    // MoE detection — MoE FFN dispatch is now supported in forward_decode_llama_family.
    // No longer FATAL. Log info for debugging.
    {
        int moe_layers = 0;
        for (int il = 0; il < cfg->n_layers; il++)
            if (g_config.layers[il].ffn_gate_inp) moe_layers++;
        if (moe_layers > 0)
            fprintf(stderr, "gfx1100-megakernel: %d/%d layers use MoE FFN dispatch\n", moe_layers, cfg->n_layers);
    }

    // WavTokenizer validation: warn if conv weights present but kernel sizes unset
    if (cfg->wav_conv1d && cfg->wav_conv1d_kernel_size == 0) {
        fprintf(stderr, "gfx1100-megakernel: WARNING — wav_conv1d weight present but "
                "wav_conv1d_kernel_size not set. Set it to weight->ne[0] for correct conv1d.\n");
    }
    for (int il = 0; il < 6; il++) {
        const auto & pl = cfg->posnet_layers[il];
        if (pl.conv1 && pl.conv1_kernel_size == 0) {
            fprintf(stderr, "gfx1100-megakernel: WARNING — posnet layer %d conv1 weight present "
                    "but conv1_kernel_size not set. Set it to weight->ne[0].\n", il);
        }
        if (pl.conv2 && pl.conv2_kernel_size == 0) {
            fprintf(stderr, "gfx1100-megakernel: WARNING — posnet layer %d conv2 weight present "
                    "but conv2_kernel_size not set. Set it to weight->ne[0].\n", il);
        }
    }
    for (int il = 0; il < cfg->n_convnext_layers; il++) {
        const auto & cl = cfg->convnext_layers[il];
        if (cl.dw && cl.dw_kernel_size == 0) {
            fprintf(stderr, "gfx1100-megakernel: WARNING — convnext layer %d dw weight present "
                    "but dw_kernel_size not set. Set it to weight->ne[0].\n", il);
        }
    }

    b.allocated = true;
    g_initialized = true;

    fprintf(stderr, "gfx1100-megakernel: initialized (%d layers, H=%d, FF=%d, V=%d)\n",
            cfg->n_layers, H, FF, V);
    return 0;
}

// ============================================================================
// Eval decode: single token → logits
// ============================================================================

// Helper to launch a kernel with error checking
#define LAUNCH(fn, grid, block, ...) do { \
    void * _args[] = { __VA_ARGS__ }; \
    hipError_t _e = hipModuleLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, b.stream, _args, nullptr); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "gfx1100-megakernel: launch failed: %s\n", hipGetErrorString(_e)); \
        return -1; \
    } \
} while(0)

// Pick matvec kernel by ggml_type (replaces stride-based Q4K/Q6K hack)
// GGML_TYPE_Q4_K = 12, GGML_TYPE_Q6_K = 14
// TODO: add more types as kernels are ported from baseline vecdotq.cuh

// ----------------------------------------------------------------------------
// forward_encode_t5 — ported from baseline src/models/t5-enc.cpp
// Encoder: bidirectional attention (no causal mask) over input tokens.
// Output: embd_enc[n_embd, n_tokens] used by decoder's cross-attention.
// ----------------------------------------------------------------------------
// T5 encoder processes ALL input tokens at once (like prefill but bidirectional).
// Output: embd_enc buffer [n_embd, n_tokens] used by decoder's cross-attention.
// Ported from baseline src/models/t5-enc.cpp
//
// Per layer:
//   RMSNorm → bidirectional self-attention (no causal mask, + relative position bias) → residual
//   RMSNorm → FFN (relu or gelu-gated) → residual
//
// The bidirectional attention is the same as causal attention but without masking
// future positions. For our megakernel, this means:
//   - For small n_tokens: use the prompt MMQ path without causal mask
//   - The attention kernel needs a non-causal variant
//
// For now: use the prompt path (prompt_causal_attn) which HAS causal masking.
// This is WRONG for encoder — it masks future tokens that should be visible.
// Need: eval_attention_bidirectional kernel (task #235)
