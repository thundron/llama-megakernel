// comp-tune.h — TUNE meta-step: per-model launch / tuning decisions
//
// Centralizes tuning decisions that today are scattered across gfx1100-init.cpp
// and forward/*.cpp. Driven by g_comp_caps (DETECT output) and model geometry.
// A single audited struct (comp_tuning) is the downstream consumer.
//
// Scope today:
//   - attn_parallel_blocks (moved here from inline in gfx1100_init)
//   - hooks for future block-dim / occupancy tuning
//   - hooks for future bottleneck detection (attn-heavy vs FFN-heavy)
//
// Philosophy: every tuned value must come with a one-liner WHY in the struct
// so that future regression debugging can find the rationale.
#pragma once

#include "comp-detect.h"
#include "../gfx1100-internal.h"

// ============================================================================
// Tuning decisions produced by TUNE. Populated once per model at init.
// ============================================================================
struct comp_tuning {
    // Attention parallel-block factor: split each attention head across
    // multiple blocks when n_q < n_cu/2 to fill the GPU. Computed so that
    // n_q * pb >= target_blocks where target_blocks ~ n_cu.
    int attn_parallel_blocks;       // 1..16; 1 = no split
    const char * attn_pb_reason;    // human-readable explanation

    // Bottleneck classification (filled in when detection lands).
    // Hints which meta-steps benefit from extra tuning effort per model.
    enum bottleneck_kind {
        BOTTLENECK_UNKNOWN  = 0,
        BOTTLENECK_ATTN,            // short sequences, many heads, D=64..256
        BOTTLENECK_FFN,              // large FF, gated, compute-heavy
        BOTTLENECK_EMBED_LMHEAD,     // tiny model, weight-load dominated
        BOTTLENECK_KV_BANDWIDTH,     // long context, KV reads dominate
    };
    int bottleneck;                  // bottleneck_kind
    const char * bottleneck_reason;

    // Reserved for future block-dim tuning. Each kernel family gets a
    // suggested block size; 0 = "use kernel default".
    int block_matvec_q4k;
    int block_rmsnorm;
    int block_attn_decode;
};

// ============================================================================
// TUNE: compute tuning decisions from caps + cfg
// ============================================================================
static void comp_tune(const gfx1100_model_config & cfg,
                      const comp_capabilities & caps,
                      comp_tuning & t) {
    memset(&t, 0, sizeof(t));

    // --- Attention parallel-block count ---
    // gfx1100 = RDNA3 = 96 CUs. Each attention block occupies 1 CU (128 threads).
    // If n_q_heads < n_cu/2 we split each head across pb blocks to fill the GPU.
    // Above that threshold, one block per head gives good occupancy already.
    const int n_cu = 96;
    const int n_q  = cfg.fa_n_q_heads;
    if (n_q <= 0) {
        t.attn_parallel_blocks = 1;
        t.attn_pb_reason = "fa_n_q_heads<=0 (non-attention model?)";
    } else if (n_q >= n_cu / 2) {
        t.attn_parallel_blocks = 1;
        t.attn_pb_reason = "n_q >= n_cu/2 (48) — single block per head has good occupancy";
    } else {
        int pb = (n_cu + n_q - 1) / n_q;
        if (pb < 1) pb = 1;
        if (pb > 16) pb = 16;
        t.attn_parallel_blocks = pb;
        t.attn_pb_reason = "ceil(n_cu / n_q) to fill GPU when heads are scarce";
    }

    // Env-var override for debugging
    if (const char * pb_env = getenv("GFX1100_PB")) {
        t.attn_parallel_blocks = atoi(pb_env);
        t.attn_pb_reason = "GFX1100_PB env var override";
    }

    // --- Bottleneck classification (heuristic, not benchmarked) ---
    // Use capability + geometry signals to tag the likely hot spot so future
    // tuning effort can prioritize. Not a performance decision by itself.
    const int H    = cfg.hidden_size;
    const int FF   = cfg.intermediate_size;
    const int V    = cfg.vocab_size;

    if (H > 0 && V > 0 && V > 16 * H) {
        // tiny hidden relative to vocab → LM head dominates
        t.bottleneck = comp_tuning::BOTTLENECK_EMBED_LMHEAD;
        t.bottleneck_reason = "V/H ratio suggests LM head is the dominant launch";
    } else if (FF > 4 * H && (caps.global_caps & CAP_FFN_GATE)) {
        t.bottleneck = comp_tuning::BOTTLENECK_FFN;
        t.bottleneck_reason = "large gated FFN relative to hidden size";
    } else if (n_q >= 32 && cfg.fa_head_dim <= 128) {
        t.bottleneck = comp_tuning::BOTTLENECK_ATTN;
        t.bottleneck_reason = "many heads with small head_dim — attention-bound";
    } else if (cfg.max_seq_len > 4096) {
        t.bottleneck = comp_tuning::BOTTLENECK_KV_BANDWIDTH;
        t.bottleneck_reason = "long context — KV-bandwidth bound";
    } else {
        t.bottleneck = comp_tuning::BOTTLENECK_UNKNOWN;
        t.bottleneck_reason = "no heuristic match";
    }
}

// ============================================================================
// Print TUNE decisions
// ============================================================================
static void comp_print_tuning(const comp_tuning & t) {
    fprintf(stderr, "gfx1100 TUNE:\n");
    fprintf(stderr, "  attn_parallel_blocks = %d  (%s)\n",
            t.attn_parallel_blocks,
            t.attn_pb_reason ? t.attn_pb_reason : "");
    const char * bn = "?";
    switch (t.bottleneck) {
        case comp_tuning::BOTTLENECK_UNKNOWN:       bn = "unknown"; break;
        case comp_tuning::BOTTLENECK_ATTN:          bn = "attention"; break;
        case comp_tuning::BOTTLENECK_FFN:           bn = "ffn"; break;
        case comp_tuning::BOTTLENECK_EMBED_LMHEAD:  bn = "embed/lm-head"; break;
        case comp_tuning::BOTTLENECK_KV_BANDWIDTH:  bn = "kv-bandwidth"; break;
    }
    fprintf(stderr, "  bottleneck = %s  (%s)\n",
            bn, t.bottleneck_reason ? t.bottleneck_reason : "");
}
