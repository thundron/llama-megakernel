# Kernel Composition System Design

## Overview

Replace the hardcoded if/else forward dispatch with a **kernel composition** generated per-model at init time. The composition is a flat list of fused operation steps assembled from base kernel ops. The forward loop executes the composition with zero branching.

## Architecture

```
  GGUF Model
      |
      v
  [1. DETECT] — read 183 GGUF keys + scan tensor names for capability bitfield
      |           (see docs/GGUF_SIGNALS.md for complete signal catalog)
      v
  [2. VALIDATE] — verify model is supportable, reject unsupported combos
      |             graceful fallback to baseline ggml-hip if not
      v
  [3. COMPOSE] — match patterns, select optimal fused kernel sequence
      |           + auto-fusion optimizer (merge adjacent steps)
      v
  [4. COMPILE] — JIT only referenced kernels (dead code elimination)
      |
      v
  [5. TUNE] — per-model launch params (parallel blocks, grid/block dims)
      |         + attention bottleneck detection
      v
  [6. ALLOCATE] — compute exact buffer sizes from composition + model dims
      |             KV cache, scratch buffers, MoE routing, SSM state
      |             shared_kv_layers → deduplicate KV cache across layers
      v
  [7. DISPATCH] — flat host-dispatch loop or hipGraph capture/replay
      |
      v
  [8. CODEGEN] — (future) generate single persistent __global__ function
                  from composed ops, with atomic grid sync between phases
```

## Data Structures

### Base Operations (the vocabulary)

Each base operation is a kernel that does one logical thing:

```cpp
enum gfx1100_op {
    // Norms (produce norm_out + residual, optionally quantize)
    OP_RMSNORM,              // input → norm_out + residual
    OP_RMSNORM_Q8,           // input → norm_out + residual + q8_act (fused)
    OP_LAYERNORM,            // same but LayerNorm
    OP_RMSNORM_ADD,          // rmsnorm(input) + residual → output (post-norm)

    // Projections
    OP_MATVEC,               // weight @ q8/f32 → output
    OP_MATVEC_RESIDUAL,      // weight @ q8/f32 + residual → output
    OP_FUSED_GATE_UP_SILU,   // silu(gate @ input) * (up @ input) → output
    OP_FUSED_GATE_UP_GELU,   // gelu(gate @ input) * (up @ input) → output
    OP_FUSED_QKV,            // [Q; K; V] = wqkv @ input, split by head dims

    // Quantize
    OP_QUANTIZE_Q8,          // f32 → Q8_1

    // Attention
    OP_ROPE_KV_WRITE,        // apply RoPE to Q/K, write K/V to cache
    OP_FLASH_ATTN_DECODE,    // flash attention single-token decode
    OP_FLASH_ATTN_PREFILL,   // flash attention batch prefill

    // Element-wise
    OP_ADD_RESIDUAL,         // dst += src
    OP_SILU_MUL,             // dst = silu(a) * b
    OP_GELU_MUL,             // dst = gelu(a) * b
    OP_SCALE,                // dst *= scalar
    OP_SOFTCAP,              // dst = tanh(dst/cap) * cap
    OP_ADD_BIAS,             // dst += bias
    OP_MUL_SCALE,            // dst *= per-element scale

    // MoE
    OP_MOE_ROUTER,           // router logits → sorted expert IDs + weights
    OP_MOE_EXPERT_FFN,       // per-expert fused matvec loop
    OP_MOE_SHARED_EXPERT,    // shared expert FFN (Qwen2MoE, DS2)

    // Special
    OP_EMBED_LOOKUP,         // token → hidden
    OP_LM_HEAD,              // hidden → logits
    OP_CHAMELEON_SUPPRESS,   // suppress image token logits
    OP_FILL_POSITIONS,       // generate position IDs on GPU
};
```

### Composition Step

```cpp
struct gfx1100_composition_step {
    gfx1100_op op;           // which operation
    hipFunction_t kernel;    // resolved kernel handle (set during compile)

    // Buffer slots — indices into a buffer table, not raw pointers
    // This allows the same plan to work with different buffer allocations
    int8_t input;            // -1 = none
    int8_t output;
    int8_t aux0, aux1;       // extra inputs (e.g., residual, weight2 for fused gate+up)

    // Weight reference
    int layer;               // which layer's weights to use
    int weight_slot;         // which slot in gfx1100_layer_weights

    // Dimensions (from model config, baked at plan time)
    int dim_in, dim_out;

    // Launch config
    int grid_x, grid_y, grid_z;
    int block_x, block_y, block_z;
    int shared_mem;
};
```

### Execution Plan

```cpp
struct gfx1100_composition {
    // Per-layer plan (same steps repeated for each layer, with layer index varying)
    gfx1100_composition_step layer_steps[64];   // max 64 steps per layer
    int n_layer_steps;

    // Pre-layer steps (embedding, position encoding)
    gfx1100_composition_step pre_steps[8];
    int n_pre_steps;

    // Post-layer steps (final norm, LM head, softcap)
    gfx1100_composition_step post_steps[8];
    int n_post_steps;

    // Buffer table — maps buffer slot IDs to actual GPU pointers
    void * buffers[32];
    // 0 = hidden, 1 = residual, 2 = norm_out, 3 = q8_act,
    // 4 = proj_scratch, 5 = mlp_inter, 6 = attn_out, 7 = kv_scratch,
    // 8 = logits, 9 = moe_sorted_ids, 10 = moe_probs, ...
};
```

## Composer Logic

The composer runs at init time, after model config is populated:

```cpp
void compose(const gfx1100_model_config & cfg,
                  const gfx1100_compiled & kernels,
                  gfx1100_composition & plan) {
    // Examine ONE representative layer to build the per-layer plan
    // (hybrid models with mixed layer types need per-layer-type plans)

    const auto & lw = cfg.layers[0]; // representative

    // === ATTENTION PHASE ===

    // Step 1: Norm
    if (cfg.norm_type == NORM_LAYER)
        add_step(plan, OP_LAYERNORM, ...);
    else
        add_step(plan, OP_RMSNORM_Q8, ...);  // fused norm+quantize

    // Step 2: QKV projection
    if (cfg.has_wqkv)
        add_step(plan, OP_FUSED_QKV, ...);
    else {
        add_step(plan, OP_MATVEC, /* wq */);
        add_step(plan, OP_MATVEC, /* wk */);
        add_step(plan, OP_MATVEC, /* wv */);
    }

    // Step 2b: QKV biases (conditional — only added if model has them)
    if (cfg.has_bias_q) add_step(plan, OP_ADD_BIAS, /* bq */);
    if (cfg.has_bias_k) add_step(plan, OP_ADD_BIAS, /* bk */);
    if (cfg.has_bias_v) add_step(plan, OP_ADD_BIAS, /* bv */);

    // Step 3: RoPE + KV cache write
    add_step(plan, OP_ROPE_KV_WRITE, ...);

    // Step 4: Attention
    add_step(plan, OP_FLASH_ATTN_DECODE, ...);

    // Step 5: O projection + residual
    if (has_post_attn_norm) {
        add_step(plan, OP_QUANTIZE_Q8, /* attn_out */);
        add_step(plan, OP_MATVEC, /* wo */);
        if (cfg.has_bias_o) add_step(plan, OP_ADD_BIAS, /* bo */);
        add_step(plan, OP_RMSNORM_ADD, /* post_attn_norm + residual */);
    } else {
        add_step(plan, OP_QUANTIZE_Q8, /* attn_out */);
        add_step(plan, OP_MATVEC_RESIDUAL, /* wo + residual */);
    }

    // === FFN PHASE ===

    // Step 6: FFN Norm
    add_step(plan, OP_RMSNORM_Q8, /* ffn_norm */);

    // Step 7: FFN body — pattern depends on model
    if (has_moe) {
        add_step(plan, OP_MOE_ROUTER, ...);
        add_step(plan, OP_MOE_EXPERT_FFN, ...);
        if (has_shared_expert) add_step(plan, OP_MOE_SHARED_EXPERT, ...);
    } else if (!has_gate && act == SILU) {
        // Fused SwiGLU (Phi3): single matrix, split, silu
        add_step(plan, OP_MATVEC, /* gate_up fused */);
        add_step(plan, OP_SILU_MUL, /* split + activate */);
    } else if (has_gate && act == SILU && can_fuse_gate_up) {
        add_step(plan, OP_FUSED_GATE_UP_SILU, /* gate, up */);
    } else if (has_gate && act == GELU_TANH && can_fuse_gate_up) {
        add_step(plan, OP_FUSED_GATE_UP_GELU, /* gate, up */);
    } else if (has_gate) {
        // Fallback: separate gate + up + activation
        add_step(plan, OP_MATVEC, /* gate */);
        add_step(plan, OP_MATVEC, /* up */);
        add_step(plan, OP_SILU_MUL or OP_GELU_MUL, ...);
    } else {
        // Ungated: up + activation alone
        add_step(plan, OP_MATVEC, /* up */);
        add_step(plan, OP_SILU or OP_GELU, ...);
    }

    // Step 8: Down projection + residual
    add_step(plan, OP_QUANTIZE_Q8, /* mlp_inter */);
    if (has_post_ffn_norm) {
        add_step(plan, OP_MATVEC, /* down */);
        add_step(plan, OP_RMSNORM_ADD, /* post_ffn_norm + residual */);
    } else {
        add_step(plan, OP_MATVEC_RESIDUAL, /* down + residual */);
    }
}
```

## Executor

The forward loop becomes trivial:

```cpp
int forward_decode_composed(int token_id, int position, float * logits_out) {
    auto & plan = g_plan;

    // Pre-layer steps (embed, etc.)
    for (int i = 0; i < plan.n_pre_steps; i++)
        launch_step(plan.pre_steps[i], -1, position);

    // Layer loop
    for (int il = 0; il < cfg.n_layers; il++) {
        for (int i = 0; i < plan.n_layer_steps; i++)
            launch_step(plan.layer_steps[i], il, position);
    }

    // Post-layer steps (final norm, LM head, etc.)
    for (int i = 0; i < plan.n_post_steps; i++)
        launch_step(plan.post_steps[i], -1, position);

    hipStreamSynchronize(stream);
    return 0;
}

void launch_step(const gfx1100_composition_step & step, int layer, int position) {
    // Build args array from step's buffer references + layer weights
    // Launch kernel with step's grid/block/shared_mem config
    void * args[16];
    int nargs = build_args(step, layer, position, args);
    hipModuleLaunchKernel(step.kernel,
        step.grid_x, step.grid_y, step.grid_z,
        step.block_x, step.block_y, step.block_z,
        step.shared_mem, stream, args, nullptr);
}
```

## Hybrid Layer Types

For models with mixed layer types (e.g., Jamba: some layers attention, some SSM):

```cpp
struct gfx1100_composition {
    // Instead of one layer_steps array, have per-layer-type plans:
    gfx1100_composition_step attn_steps[64];
    int n_attn_steps;
    gfx1100_composition_step ssm_steps[64];
    int n_ssm_steps;
    gfx1100_composition_step rwkv_steps[64];
    int n_rwkv_steps;
    gfx1100_composition_step dn_steps[64];
    int n_dn_steps;

    // Layer dispatch table: which plan to use per layer
    int layer_plan_type[128]; // 0=attn, 1=ssm, 2=rwkv, 3=deltanet
};
```

## The 5 Meta-Steps

The composition system is a pipeline of 5 meta-steps, inspired by functional
programming's compose-then-execute paradigm. Each step transforms the model
description into a more optimized form:

```
  GGUF Model
      |
      v
  [1. DETECT] — read weights, types, shapes, capabilities from GGUF
      |           Output: model config struct (dimensions, quant types, features)
      v
  [2. COMPOSE] — match patterns, select optimal fused kernel sequence
      |           Output: flat list of (op, kernel, buffers, dims, launch config)
      |           Also: auto-fusion optimizer merges adjacent steps
      v
  [3. COMPILE] — JIT compile ONLY the kernels referenced by the composition
      |           Dead code elimination: unreferenced kernel variants are #ifdef'd out
      |           Output: model-specific .dll/.hsaco containing only needed kernels
      v
  [4. TUNE] — pick optimal launch parameters per step
      |         - Parallel blocks (pb) for attention: based on n_q_heads vs CU count
      |         - Block dimensions: based on head_dim, hidden_size
      |         - Shared memory: based on kernel requirements
      |         - Warp count: based on register pressure analysis
      |         Output: composition steps with final grid/block/shm configs
      v
  [5. DISPATCH] — execute the composition
                  Two modes from the SAME plan:
                  a) Host-dispatch: flat loop, one hipModuleLaunchKernel per step
                  b) Graph-capture: record once, hipGraphLaunch replay per token
                     (only 3 scalars change: position, kv_len, token_id)
```

### Why this matters

Each kernel launch in the composition is:
- **Model-specific**: only contains code paths for THIS model's quant types,
  head dimensions, activation functions, norm types
- **Optimally configured**: launch parameters tuned for THIS model's head count,
  hidden size, layer structure
- **Minimally fused**: adjacent ops merged when profitable (not when they hurt,
  e.g., quantize+matvec was found to be 9-16x SLOWER due to shared memory overhead)

The composition IS the model's forward pass. Different model → different composition →
different compiled kernel → different launch configs. No wasted code, no wasted registers,
no wasted launches.

### Compile-time dead code elimination (Step 3)

The composition references specific kernels. The COMPILE step generates #defines
that gate which kernels are included:

```cpp
// Generated from composition — only Llama 1B Q4_K needs:
#define NEED_MATVEC_Q4K 1
#define NEED_MATVEC_Q6K 1    // for embed
#define NEED_ATTN_WMMA 1     // D=64
#define NEED_NORM_RMS 1
#define NEED_FUSED_GATE_UP_SILU 1
// Everything else is compiled out — no Q5_K, no TILE attn, no LayerNorm, etc.
```

This reduces: compile time (~30s → ~5s), DLL size (~1.2MB → ~200KB),
GPU instruction cache pressure (fewer code paths loaded).

### Attention tuning (Step 4)

The TUNE step is critical for attention dispatch. Measured data:
- Gemma-2 (8 Q heads): single-block = 8% CU occupancy → pb=3 gives 96 blocks = 100%
- Llama (32 Q heads): single-block = 33% occupancy → pb=1 is optimal
- Phi-3.5 (32 Q heads, D=96): WMMA preferred over VEC for D≤128

The tuning formula:
```cpp
int pb = max(1, min(8, target_blocks / n_q_heads));
// where target_blocks = n_CU (96 for gfx1100)
```

### Attention bottleneck detection (Step 4 enhancement)

Profile data reveals attention dominates differently per model:
- Llama 1B:  attn = 26% (balanced)
- Qwen 0.5B: attn = 29% (balanced)
- Gemma-2 2B: **attn = 44.8%** (D=256 KV cache bandwidth bottleneck)
- Phi-3.5:   attn = 38.7% (D=96, large model)

When the TUNE step detects attention > 40% of compute, it should apply
model-specific attention optimizations:

1. **Parallel block splitting** — for low-head-count models (Gemma-2: 8 Q heads
   → only 8 blocks → 8% CU occupancy). Split KV range across `pb` blocks:
   `pb = max(1, 96 / n_q_heads)` → Gemma-2 gets pb=3 → 24 blocks → much better.

2. **KV cache compression** — D=256 means each KV position reads 512 bytes (2×256×f16).
   TBQ4_0 KV cache would reduce this to 128 bytes (4x compression). The TUNE step
   could select cache format based on head dimension.

3. **Attention + O-proj fusion** — keep attention output in registers/LDS, feed
   directly into O-projection matvec without global memory round-trip. Saves
   `n_q_heads * head_dim * 4 bytes` of global writes + reads per layer.

4. **Persistent attention** — for models where attention dominates, launch a
   persistent kernel where different warp groups pipeline: group A does attention
   for layer N while group B does O-proj for layer N-1. Requires grid sync.

### Graph capture (Step 5b)

The composition is deterministic — same kernel sequence every token.
Only 3 scalar args change (position, kv_len, token_id).

```cpp
// First token: execute plan normally + capture
hipStreamBeginCapture(stream);
for (auto & step : plan.steps) launch_step(step);
hipStreamEndCapture(stream, &graph);
hipGraphInstantiate(&graph_exec, graph);

// Subsequent tokens: update scalars + replay
d_params[0] = token_id; d_params[1] = position; d_params[2] = kv_len;
hipGraphLaunch(graph_exec, stream);
```

NOTE: As of ROCm 7.1, hipGraphLaunch is NOT faster than direct dispatch on gfx1100.
Also, SWA models (Gemma-2) have per-layer attention window differences that
complicate graph reuse. This remains a future optimization target.

### Grid sync benchmark (2026-04-16, gfx1100 RX 7900 XTX)

Measured cost per synchronization point, comparing persistent kernel (atomic
barrier) vs separate kernel launches:

| Blocks | Atomic Barrier | Separate Launches | Speedup |
|--------|---------------|-------------------|---------|
| 48     | 1.56 µs/sync  | 2.37 µs/launch    | 1.52x   |
| 96     | 1.21 µs/sync  | 1.69 µs/launch    | 1.39x   |
| 192    | 0.79 µs/sync  | 1.53 µs/launch    | 1.94x   |

Key findings:
- `hipDeviceAttributeCooperativeLaunch` = 0 on ROCm 7.1.51803 / gfx1100 / Windows
  (may be a Windows-specific limitation or RDNA3 driver gap — needs testing on Linux)
- Hand-rolled atomic barrier (atomicAdd + volatile generation counter) WORKS
  and is the recommended approach regardless of cooperative launch support
- Persistent mega-kernel with atomic grid sync is **always faster** than
  separate launches (1.4-1.9x less dispatch overhead)
- More blocks = cheaper sync (atomic contention amortized)
- At 192 blocks × 312 syncs (Gemma-2): 0.25ms sync overhead vs 0.48ms launch overhead

This validates the CODEGEN approach: a generated persistent `__global__` function
with atomic grid sync between phases is viable and faster than our current
separate-kernel-launch architecture.

Test: `tests/test-grid-sync.hip`

## Benefits

1. **New architectures = new composition, not new code**
2. **Fusion decisions made once, not per-token**
3. **The composition IS the documentation** — print it to see exactly what a model does
4. **Testable** — compose a composition, verify it matches expected sequence
5. **Optimizable** — auto-fusion optimizer merges adjacent steps automatically
6. **Compile-time specialization** — the .dll only includes kernels the composition references
7. **Profiling per-step is trivial** — each step has a name and a kernel handle
8. **Per-model tuning** — launch configs optimized for each model's specific geometry
9. **Graph-ready** — the flat plan structure is directly capturable as a hipGraph

## Implementation Status (2026-04-16)

| # | Meta-step  | Status | Files | Gaps |
|---|-----------|--------|-------|------|
| 1 | DETECT    | DONE (runtime scan) | `composition/comp-detect.h`, runs every `gfx1100_init()` into `g_comp_caps` | Tensor scan now covers ~all fields in `gfx1100_layer_weights` (60-bit capability bitfield). Missing GGUF-key side still split across loaders |
| 2 | VALIDATE  | DONE (PoC, wired) | `composition/comp-validate.h`, runs every `gfx1100_init()` into `g_comp_validation` | Results cached but nothing downstream gates on them yet |
| 3 | COMPOSE   | DONE (Llama+Gemma) | `composition/comp-compose.h`, optimizer scaffold in `comp-optimize.h` | Supports Llama-family + Gemma iSWA with post-norms. Bails to forward/*.cpp for MoE/SSM/RWKV/MLA/T5/BERT/Cohere2/BitNet/CogVLM/Phi-3 wqkv/non-SILU-GELU. Anti-pattern DB audited each plan |
| 4 | COMPILE   | RESCOPED | JIT compile pipeline in `gfx1100-init.cpp` | `NEED_KERNEL_*` DCE was built then reverted — irrelevant under per-model codegen direction. COMPILE now means "hipcc the emitted `megakernel_<model>.hip`" |
| 5 | TUNE      | DONE (PoC) | `composition/comp-tune.h`, `g_comp_tuning` populated every init | `attn_parallel_blocks` moved out of init. Heuristic bottleneck classification (ATTN/FFN/EMBED-LMHEAD/KV-BW). Block-dim/occupancy tuning is a hook |
| 6 | ALLOCATE  | PARTIAL | Manual allocation in `mk_init` | Not driven by composition plan; no buffer slot reuse optimization |
| 7 | DISPATCH  | PARTIAL | host-dispatch works; hipGraph tested | No automatic mode selection; hipGraph not faster on ROCm 7.2.1 |
| 8 | CODEGEN   | NOT STARTED | grid sync benchmarked in `tests/test-grid-sync.hip` | Atomic barrier validated at 0.79-1.56 µs/sync |

## Migration Path (remaining work)

### Near-term — ALL DONE (2026-04-16)
1. ~~Wire VALIDATE into `gfx1100_init()` (not just composition dispatch)~~
2. ~~Extend DETECT to scan ALL tensor types for capability bitfield at runtime~~
3. ~~Implement auto-fusion optimizer in COMPOSE step~~ (scaffold + anti-pattern DB)
4. ~~Build NEED_KERNEL_* compile-time DCE infrastructure~~ (flags; per-kernel wrapping incremental)
5. ~~Extend composer to handle SWA layers (iSWA pattern)~~ (Gemma iSWA works, unsupported cases bail)

### Medium-term
6. Port SSM/RWKV/DeltaNet/T5 forward paths to composition-based
   — Largest remaining task. Today `forward/*.cpp` handles these correctly;
     the composer refuses them and falls back. Each arch is an independent
     port (SSM, RWKV, DeltaNet, MLA, T5 dec, T5 enc, BERT, BitNet, CogVLM).

   **Recommended order (smallest delta first):**
   1. BitNet — 281 LOC, Llama + sub-norms between core op and output proj.
      New ops: `OP_ATTN_SUB_NORM`, `OP_FFN_SUB_NORM`.
   2. Phi-3-wqkv — small, fused QKV single-weight path. New op: `OP_FUSED_WQKV_SPLIT`.
   3. Cohere2 — Llama + skip-RoPE-on-global. New kernel: `OP_KV_WRITE_NO_ROPE`.
   4. MoE — Llama + expert routing. New ops: `OP_MOE_ROUTER`, `OP_MOE_EXPERT_FFN`, `OP_MOE_SHARED_EXPERT`.
   5. MLA — absorbed+uncompressed paths. New ops for both.
   6. T5 decoder — adds cross-attention. New op: `OP_CROSS_ATTN`.
   7. T5 encoder / BERT — bidirectional attention. New attention kernel variant.
   8. SSM / Mamba / Mamba2 — entirely different layer kind.
   9. RWKV6 / RWKV7 — time-mix + channel-mix.
   10. DeltaNet — conv1d + delta-rule recurrence.
   11. CogVLM — visual-expert weight switching per-token.

   **Per-port template:**
   - Add `CAP_*` bits used (already in `comp-detect.h` for most).
   - Extend `comp-compose.h`: remove the `bail()` for this arch, emit ops.
   - Add new `comp_op` enum values in `comp-types.h`.
   - Add new execute handlers in `comp-execute.h`.
   - Add kernel wiring in `gfx1100-internal.h` if new kernels needed.
   - Verify against the existing `forward/*.cpp` baseline before enabling.

7. ~~Implement TUNE step (bottleneck detection, pb calculation)~~ DONE (PoC)
8. Implement ALLOCATE step (composition-driven buffer sizing)
9. Delete the old if/else forward functions once coverage matches

### Long-term
10. CODEGEN: generated persistent `__global__` per model with atomic grid sync

## How "works for all models" stands today

| Model family | Dispatch path | Notes |
|---|---|---|
| Llama / Mistral / Qwen2/2.5 | composer (opt-in) OR `forward/llama-family.cpp` | both paths tested |
| Gemma 2/3/4 | composer (opt-in) OR `forward/llama-family.cpp` | iSWA handled |
| Phi-3.5 (fused wqkv) | `forward/llama-family.cpp` | composer bails — wqkv support absent |
| BitNet | `forward/bitnet.cpp` | sub-norm path |
| CogVLM | `forward/cogvlm.cpp` | visual-expert dispatch |
| DeepSeek2 MLA | `forward/deepseek2-mla.cpp` | absorbed+uncompressed |
| T5 decoder | `forward/t5-dec.cpp` | cross-attention |
| T5 encoder | `encoder/t5-enc.cpp` | |
| BERT | `encoder/bert.cpp` | |
| Mamba / Mamba2 | `forward/mamba.cpp` | SSM |
| RWKV6 / RWKV7 | `forward/rwkv6.cpp` / `rwkv7.cpp` | |

Every current model + quant works via the forward-tree path. The composer
accelerates Llama/Gemma specifically; other archs continue to use forward/*.cpp
unchanged. DETECT and VALIDATE now run for every model (populating
`g_comp_caps` / `g_comp_validation`) so downstream consumers (e.g. NEED_KERNEL_*
DCE, future TUNE/ALLOCATE steps) have full per-model capability data.

## Auto-Fusion Optimizer (COMPOSE step)

A composition optimizer pass that runs after initial composition:

```cpp
void optimize_composition(gfx1100_composition & plan) {
    // Pattern: MATVEC(gate) + MATVEC(up) + SILU_MUL → FUSED_GATE_UP_SILU
    // Pattern: RMSNORM + QUANTIZE_Q8 → RMSNORM_Q8
    // Pattern: MATVEC + ADD_RESIDUAL → MATVEC_RESIDUAL
    // Pattern: MATVEC + RMSNORM_ADD → MATVEC_POSTNORM_RESIDUAL (future kernel)
    // Pattern: MATVEC(wq) + MATVEC(wk) + MATVEC(wv) → FUSED_QKV (future kernel)
    //
    // ANTI-patterns (measured to be slower):
    // Pattern: QUANTIZE_Q8 + MATVEC → FUSED_QUANTIZE_MATVEC  ← 9-16x SLOWER!
    //   (shared memory Q8 quantization overhead > L2 cache benefit)
}
```

This separates the concerns: the composer builds a correct plan from base ops, then the
optimizer merges adjacent ops into fused variants when the kernels exist. Adding a new
fused kernel just means adding a new pattern to the optimizer — all models that match
the pattern automatically benefit.

## Measured Data (informing design decisions)

### Gemma-2 2B kernel launch analysis (26 layers)
- Current: 14 launches/layer × 26 = 364 total
- With QKV fusion: 12 launches/layer × 26 = 312
- Theoretical minimum: 9 launches/layer × 26 = 234
- Dispatch overhead: ~0.8µs per hipModuleLaunchKernel
- Total dispatch cost: 364 × 0.8µs = 0.29ms = ~2.6% of 11ms decode time

### Performance impact of dispatch overhead
- At 100 tokens: dispatch is ~12% of total (attention small, overhead dominates)
- At 2000 tokens: dispatch is ~3.8% (attention large, overhead amortized)
- At 5000 tokens: dispatch is ~2% (attention dominates)
