// comp-types.h — Execution composition data structures for the megakernel
//
// The composition replaces the if/else forward dispatch with a pre-composed
// sequence of kernel launches. The composer builds the composition at init time
// by inspecting model capabilities. The executor runs it as a flat loop.
//
// PoC scope: Llama 1B decode only.
#pragma once

#include <hip/hip_runtime.h>

// Maximum steps per layer (generous for the PoC)
#define COMP_MAX_LAYER_STEPS  32
#define COMP_MAX_PRE_STEPS     8
#define COMP_MAX_POST_STEPS    8

// ============================================================================
// Buffer slots — indices into a buffer table, NOT raw pointers.
// The executor resolves these to actual GPU pointers at launch time.
// ============================================================================
enum comp_buf {
    BUF_HIDDEN      = 0,  // b.hidden
    BUF_RESIDUAL    = 1,  // b.residual
    BUF_NORM_OUT    = 2,  // b.norm_out
    BUF_Q8_ACT      = 3,  // b.q8_act
    BUF_PROJ_SCRATCH = 4, // b.proj_scratch
    BUF_MLP_INTER   = 5,  // b.mlp_inter
    BUF_ATTN_OUT    = 6,  // b.attn_out
    BUF_KV_SCRATCH  = 7,  // b.kv_scratch
    BUF_LOGITS      = 8,  // b.logits
    BUF_NONE        = -1,
};

// ============================================================================
// Operation types — the vocabulary of composition steps
// ============================================================================
enum comp_op {
    // Norms
    OP_RMSNORM_Q8_QUANTIZE,   // fused: input → norm_out + residual + q8_act
    OP_RMSNORM_ADD,            // fused: rmsnorm(input) + residual → output (post-norm)
    OP_RMSNORM_ADD_RMSNORM_Q8, // fused consecutive: post-norm + pre-norm in one kernel

    // Projections
    OP_MATVEC,                 // weight @ q8/f32 → output
    OP_MATVEC_RESIDUAL,        // weight @ q8/f32 + residual → output
    OP_FUSED_QKV,              // 3 matvecs in 1: Q+K+V from same q8_act
    OP_FUSED_GATE_UP_SILU,     // silu(gate @ input) * (up @ input)
    OP_FUSED_GATE_UP_GELU,     // gelu(gate @ input) * (up @ input)

    // Quantize
    OP_QUANTIZE_Q8,            // f32 → Q8_1

    // Attention
    OP_ROPE_KV_WRITE,          // RoPE + KV cache write
    OP_ATTN_DECODE_VEC,        // flash attention VEC kernel
    OP_ATTN_DECODE_TILE,       // flash attention TILE kernel
    OP_ATTN_DECODE_WMMA,       // flash attention WMMA kernel

    // Embedding / LM head
    OP_EMBED_LOOKUP,           // token → hidden
    OP_EMBED_SCALE,            // hidden *= sqrt(H)
    OP_FINAL_NORM,             // final RMSNorm (no residual)
    OP_LM_HEAD,                // hidden → logits
    OP_SOFTCAP,                // tanh(x/cap) * cap

    // Misc
    OP_NOP,                    // no-op (placeholder for alignment)
};

// ============================================================================
// Composition step — one kernel launch
// ============================================================================
struct comp_step {
    comp_op op;

    // Which kernel to launch (resolved during composition compilation)
    hipFunction_t kernel;

    // Grid and block dims
    int grid_x, grid_y, grid_z;
    int block_x, block_y, block_z;
    int shared_mem;

    // Weight reference: which layer weight slot to use
    // -1 = global weight (embed, lm_head, final_norm)
    // 0+ = layer index (resolved per-layer during execution)
    int weight_slot;       // slot index in gfx1100_layer_weights (0-15 for ptrs/strides/types)
    int weight_slot_2;     // second weight (for fused QKV: K weight, for fused gate+up: up weight)
    int weight_slot_3;     // third weight (for fused QKV: V weight)

    // Dimensions (baked at composition time from model config)
    int dim_in;
    int dim_out;
    int dim_out_2;         // for fused QKV: kv_size
};

// ============================================================================
// Execution composition
// ============================================================================
struct gfx1100_composition {
    // Pre-layer steps (embedding, scaling)
    comp_step pre[COMP_MAX_PRE_STEPS];
    int n_pre;

    // Per-layer steps (same sequence for every layer of the same type)
    comp_step layer[COMP_MAX_LAYER_STEPS];
    int n_layer;

    // Post-layer steps (final norm, LM head, softcap)
    comp_step post[COMP_MAX_POST_STEPS];
    int n_post;

    // Buffer table — maps comp_buf enum to actual GPU pointers
    void * buffers[16];

    // Model config snapshot (needed by executor for layer weight lookup)
    int n_layers;
    int hidden_size;
    int vocab_size;
};
