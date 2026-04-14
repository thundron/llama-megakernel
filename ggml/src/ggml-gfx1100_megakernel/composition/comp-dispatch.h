// comp-dispatch.h — Top-level composition-based dispatch for decode
//
// Pipeline: DETECT → VALIDATE → COMPOSE → EXECUTE
//
// DETECT runs a systematic tensor-existence scan to build a complete
// capability bitfield. VALIDATE rejects unsupported models with clear
// diagnostics before we waste cycles on COMPOSE. If anything is unsupported,
// we fall back to the traditional if/else forward dispatch which has broader
// architecture coverage.
//
// Enable composition dispatch with GFX1100_COMPOSITION=1 env var.
#pragma once

#include "comp-types.h"
#include "comp-detect.h"
#include "comp-validate.h"
#include "comp-compose.h"
#include "comp-optimize.h"
#include "comp-execute.h"

// Global composition state — initialized once, reused for all tokens
static gfx1100_composition g_comp = {};
static comp_capabilities g_comp_caps = {};
static comp_validation_result g_comp_validation = {};
static bool g_comp_initialized = false;
static bool g_comp_available = false;
static bool g_comp_enabled = false;

// Initialize the composition system. Call once after gfx1100_init.
// Runs: DETECT → VALIDATE → COMPOSE → (cached executor state).
static void comp_init(const gfx1100_model_config & cfg,
                      const gfx1100_compiled & k,
                      const gfx1100_buffers & b) {
    g_comp_enabled = (getenv("GFX1100_COMPOSITION") != nullptr);
    if (!g_comp_enabled) return;

    // STEP 1: DETECT — systematic tensor-existence capability scan
    comp_detect(cfg, g_comp_caps);
    comp_print_capabilities(g_comp_caps);

    // STEP 2: VALIDATE — reject unsupported models upfront
    comp_validate(cfg, g_comp_validation);
    comp_print_validation(g_comp_validation);

    if (!g_comp_validation.supported) {
        fprintf(stderr, "gfx1100 composition: model unsupported, "
                "falling back to if/else dispatch\n");
        g_comp_available = false;
        g_comp_initialized = true;
        return;
    }

    // STEP 3: COMPOSE — build optimal kernel sequence
    compose(cfg, k, b, g_comp);

    if (g_comp.n_layer > 0) {
        // STEP 3b: Auto-fusion optimizer post-pass (safe rewrites + anti-pattern audit)
        comp_optimize_stats opt_stats;
        comp_optimize(g_comp, opt_stats);
        comp_print_optimize_stats(opt_stats);

        g_comp_available = true;
        print_composition(g_comp);
    } else {
        fprintf(stderr, "gfx1100 composition: compose failed for arch %d "
                "despite passing validation — falling back to if/else dispatch\n",
                cfg.arch_id);
        g_comp_available = false;
    }
    g_comp_initialized = true;
}

// Try composition-based decode. Returns 0 if composition handled it, -1 if fallback needed.
static int comp_decode(int token_id, int position, float * logits_out) {
    if (!g_comp_enabled || !g_comp_available) return -1;

    return execute_composition(g_comp, g_config, g_compiled, g_bufs,
                               token_id, position, logits_out);
}
