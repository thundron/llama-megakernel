// comp-optimize.h — Auto-fusion optimizer (COMPOSE meta-step, post-pass)
//
// Runs AFTER the baseline composer has emitted ops. Scans the step list for
// patterns that can be rewritten to fewer, fused kernels. Also enforces
// anti-patterns — fusions that were measured to be slower than the
// decomposed form (e.g. QUANTIZE_Q8 + MATVEC → shared-memory Q8 quantization
// was benchmarked at 9-16x SLOWER than separate quantize then matvec).
//
// Today, the baseline composer already emits the fused forms inline. This
// optimizer exists so:
//   1. Newly added patterns have a single obvious home.
//   2. Anti-patterns are enforced by the name of the function that forbids them.
//   3. Diagnostic printing shows what fusions were considered.
//
// Philosophy: never regress. If a rewrite can't be proven safe, leave the
// steps alone. The composer already produced correct code.
#pragma once

#include "comp-types.h"
#include <cstdio>

// ============================================================================
// Fusion catalog — documented patterns the optimizer knows about.
// Each pattern has a fingerprint (ordered op sequence) and a result op.
// If the result op is OP_NOP, the pattern is a DOCUMENTED ANTI-PATTERN and
// must never be applied (here for future optimizers to know about).
// ============================================================================

// Anti-patterns — NEVER apply these. Documented for posterity.
struct comp_anti_pattern {
    const char * name;
    comp_op      a;
    comp_op      b;
    const char * reason;
};

static const comp_anti_pattern COMP_ANTI_PATTERNS[] = {
    {
        "quantize_matvec_shared_q8",
        OP_QUANTIZE_Q8, OP_MATVEC,
        "fusing Q8 quantization into matvec shared memory measured 9-16x "
        "SLOWER on gfx1100 (RX 7900 XTX). The cost of the quantization pass "
        "in shared memory exceeds the L2 cache benefit of skipping the "
        "separate quantize launch.",
    },
};
static constexpr int COMP_N_ANTI_PATTERNS =
    sizeof(COMP_ANTI_PATTERNS) / sizeof(COMP_ANTI_PATTERNS[0]);

// ============================================================================
// Optimizer statistics — populated per run for diagnostics.
// ============================================================================
struct comp_optimize_stats {
    int n_patterns_seen;        // how many 2-op windows were inspected
    int n_rewrites_applied;     // how many pairs were fused into 1
    int n_anti_patterns_seen;   // how many anti-pattern pairs we *refused* to fuse
    int n_rewrites_skipped;     // fusible in theory, skipped (kernel missing, safety)
};

// ============================================================================
// Pattern rewriters
// ============================================================================

// Scan a contiguous step list for anti-pattern adjacencies. Does NOT modify
// the list — it only counts. An anti-pattern that appears means the composer
// already chose the decomposed form (correct); seeing one should be rare.
static int comp_count_anti_patterns(const comp_step * steps, int n) {
    int count = 0;
    for (int i = 0; i + 1 < n; i++) {
        for (int k = 0; k < COMP_N_ANTI_PATTERNS; k++) {
            if (steps[i].op == COMP_ANTI_PATTERNS[k].a &&
                steps[i + 1].op == COMP_ANTI_PATTERNS[k].b) {
                count++;
            }
        }
    }
    return count;
}

// ============================================================================
// Top-level optimizer entry point.
// ============================================================================
// Applies safe rewrites to `plan`. Current behavior: no rewrites (the composer
// already emits fused forms inline). This function exists so new fusion
// patterns have an obvious, audited home — any future rewriter added here
// must (a) have a unit test, (b) respect the anti-pattern list.
static void comp_optimize(gfx1100_composition & plan,
                          comp_optimize_stats & stats) {
    memset(&stats, 0, sizeof(stats));

    // Count anti-patterns in each sublist (for diagnostics).
    stats.n_anti_patterns_seen += comp_count_anti_patterns(plan.pre, plan.n_pre);
    stats.n_anti_patterns_seen += comp_count_anti_patterns(plan.layer, plan.n_layer);
    stats.n_anti_patterns_seen += comp_count_anti_patterns(plan.post, plan.n_post);

    // Count 2-op windows inspected (for diagnostics).
    auto window_count = [](int n) { return n > 0 ? n - 1 : 0; };
    stats.n_patterns_seen =
        window_count(plan.n_pre) +
        window_count(plan.n_layer) +
        window_count(plan.n_post);

    // Future rewriters:
    //   - MATVEC + RMSNORM_ADD → MATVEC_POSTNORM_RESIDUAL (kernel not yet written)
    //   - OP_EMBED_LOOKUP + OP_EMBED_SCALE + norm → fused embed pipeline
    //   - ROPE_KV_WRITE + ATTN_DECODE_* → ROPE_ATTN_FUSED (kernel not yet written)
    //
    // When adding a rewriter: increment n_rewrites_applied on success,
    // n_rewrites_skipped if the pattern matched but the fused kernel is
    // unavailable at this link (for debugability).
}

// ============================================================================
// Print optimizer stats
// ============================================================================
static void comp_print_optimize_stats(const comp_optimize_stats & s) {
    fprintf(stderr, "gfx1100 COMPOSE (optimize): "
                    "patterns_seen=%d rewrites=%d skipped=%d anti_patterns=%d\n",
            s.n_patterns_seen, s.n_rewrites_applied,
            s.n_rewrites_skipped, s.n_anti_patterns_seen);
    if (s.n_anti_patterns_seen > 0) {
        fprintf(stderr, "  NOTE: anti-patterns adjacent in plan — composer correctly "
                        "used the decomposed form; this is benign.\n");
    }
}
