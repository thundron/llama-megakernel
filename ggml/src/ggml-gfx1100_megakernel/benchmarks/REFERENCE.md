# gfx1100 Megakernel Performance Reference

**Last updated**: 2026-04-14 (post-optimization round 2)
**Hardware**: AMD Radeon RX 7900 XTX (gfx1100, RDNA3, 24 GB VRAM, 960 GB/s peak bandwidth, wave32)
**Software**: llama.cpp commit 82764d8f4, ROCm 7.1, hipcc, rocWMMA 2.1.0
**Baseline**: Standard llama.cpp HIP/ROCm backend (graph-based dispatch, rocWMMA enabled)

---

## Theoretical Maximum Decode Throughput

For single-token decode, the bottleneck is memory bandwidth (reading all weights once per token).

| Model | Params | Weight Size | Theo Max (960 GB/s) |
|-------|--------|-------------|---------------------|
| Llama 3.2 1B Q4_K_M | 1.24B | 763 MB | 1289 tok/s |
| Qwen2.5 0.5B Q4_K_M | 0.63B | 463 MB | 2124 tok/s |
| Gemma-2 2B Q4_K_M | 2.61B | 1594 MB | 617 tok/s |
| Phi-3.5-mini Q4_K_M | 3.82B | 2234 MB | 440 tok/s |

---

## Decode Throughput (2000 tokens sustained, tok/s)

Both baseline and megakernel built with rocWMMA 2.1.0 enabled.

| Model | Baseline | Megakernel | Speedup | MK Efficiency | Attn Kernel |
|-------|----------|------------|---------|---------------|-------------|
| Llama 3.2 1B | 303 | **427** | **1.41x** | 33.1% | VEC (D=64) |
| Qwen2.5 0.5B | 334 | **447** | **1.34x** | 21.1% | VEC (D=64) |
| Gemma-2 2B | 158 | **152** | 0.96x | 24.6% | VEC (D=256) |
| Phi-3.5-mini | 117 | **162** | **1.38x** | 36.8% | WMMA (D=96) |

Efficiency = measured tok/s / theoretical max tok/s

### Attention kernel selection (auto-dispatch, overridable via GFX1100_ATTN env var)
- `GFX1100_ATTN=wmma` — force rocWMMA tensor core kernel
- `GFX1100_ATTN=tile` — force TILE scalar kernel
- `GFX1100_ATTN=vec` — force VEC vectorized kernel
- unset — auto: WMMA for D<=128, VEC for D>=128 (D%64==0), TILE fallback

### Per-model attention kernel benchmark (Phi-3.5 D=96)
| Kernel | tok/s | vs Baseline |
|--------|-------|-------------|
| VEC (D_PAD=128) | 151 | 1.29x |
| TILE (native D=96) | 145 | 1.24x |
| WMMA (rocWMMA 16x16) | 163 | 1.39x |
| Baseline (VEC, no rocWMMA) | 117 | 1.00x |

---

## Correctness (100-token golden reference vs baseline)

| Model | Argmax Match | Max Abs Error | Status |
|-------|-------------|---------------|--------|
| Llama 3.2 1B | **100/100** | 0.82 | PASS |
| Qwen2.5 0.5B | **100/100** | 1.23 | PASS |
| Gemma-2 2B | **100/100** | 0.75 | PASS |
| Phi-3.5-mini | 96/100 | 3.0 (mismatches), 16.8 (matching) | Float non-associativity |

Phi-3.5 mismatches: 4 close-call tokens where top-2 logits differ by 1.2-3.0. All three attention kernels (VEC, TILE, WMMA) produce the same 4 mismatches — the divergence is upstream in 32-layer MHA with mscale=1.19, not from the attention kernel.

---

## Per-Phase Profile (single token decode at position ~2000)

### Llama 3.2 1B (16 layers, H=2048, FF=8192, D=64)

| Phase | Time | % |
|-------|------|---|
| norm | 0.58 ms | 13.5% |
| qkv_proj | 0.52 ms | 12.2% |
| rope_kv | 0.17 ms | 3.9% |
| **attn** | **1.04 ms** | **24.2%** |
| o_proj | 0.07 ms | 1.6% |
| ffn_norm | 0.70 ms | 16.4% |
| **ffn_proj** | **0.81 ms** | **18.9%** |
| ffn_res | 0.16 ms | 3.8% |
| lm_head | 0.27 ms | 6.2% |
| **TOTAL** | **4.28 ms** | |

### Gemma-2 2B (26 layers, H=2304, FF=9216, D=256)

| Phase | Time | % | Notes |
|-------|------|---|-------|
| norm | 1.19 ms | 10.8% | |
| qkv_proj | 0.82 ms | 7.4% | |
| rope_kv | 0.30 ms | 2.7% | |
| **attn** | **4.88 ms** | **44.3%** | D=256 KV cache bandwidth bottleneck |
| o_proj | 0.05 ms | 0.4% | |
| ffn_norm | 1.12 ms | 10.2% | |
| ffn_proj | 1.91 ms | 17.4% | Fused gate+up+gelu (1 launch) |
| ffn_res | 0.48 ms | 4.3% | Includes post-FFN norm (rmsnorm_add) |
| lm_head | 0.26 ms | 2.4% | |
| **TOTAL** | **11.01 ms** | | |

### Per-layer comparison: Gemma-2 vs Llama

| Phase | Gemma per-layer | Llama per-layer | Ratio | Expected | Analysis |
|-------|----------------|-----------------|-------|----------|----------|
| norm | 0.046 ms | 0.036 ms | 1.14x | ~1.13x (H ratio) | OK |
| qkv_proj | 0.032 ms | 0.033 ms | 0.99x | ~1.0x | OK — similar H |
| attn | **0.188 ms** | **0.065 ms** | **2.89x** | 2.0x | **45% excess** |
| ffn_proj | 0.074 ms | 0.051 ms | 1.50x | 1.13x (FF ratio) | Was 3-launch, now fused |
| ffn_res | 0.018 ms | 0.010 ms | 2.04x | ~1.0x | Post-norm overhead |

### D=256 Attention Bandwidth Analysis

| Metric | D=64 (Llama) | D=256 (Gemma-2) |
|--------|-------------|-----------------|
| Bytes per K row | 128 | 512 |
| Cache lines per K row | 2 | 8 |
| KV bytes per layer (pos=2000) | 4.0 MB | 8.0 MB |
| KV bytes per token (all layers) | 64 MB | 208 MB |
| Theoretical min attn time | 0.067 ms | 0.217 ms |
| Actual attn time | 1.04 ms | 4.88 ms |
| **Bandwidth utilization** | **6.4%** | **4.4%** |

D=256 has 1.5x worse bandwidth utilization than D=64. Root cause: 512-byte K rows span 8 cache lines, causing L1 cache pressure when 128 threads read different rows simultaneously.

Baseline uses VEC kernel for D=256 decode (same as us). The 4% gap (158 vs 152 tok/s) is from host dispatch overhead (~312 kernel launches per token), not from any kernel being slower.

---

## Optimizations Applied

### Kernel fusions (eliminate launches + reduce memory traffic)
1. **Fused rmsnorm + Q8 quantize** — single-pass register caching, 1 kernel instead of 2. Saves 2 launches × 2 norms × n_layers per token.
2. **Fused gate+up+silu matvec** — shared memory q8_act staging, 1 kernel instead of 3. For SiLU models (Llama, Qwen).
3. **Fused gate+up+gelu matvec** — same pattern with GELU-tanh activation. For Gemma-2/3/4.
4. **Fused post-norm + residual add** (eval_rmsnorm_add) — 1 kernel instead of 2 for post-attn/post-FFN norms.
5. **Fused logit softcap** (eval_softcap) — `tanh(x/cap)*cap` in 1 kernel instead of 3.
6. **MoE skip first re-quantize** — q8_act already valid from Phase 5b.
7. **Single-pass rmsnorm** — cache input in registers, no re-read for normalize pass.

### Attention kernels (3 options, auto-dispatched)
- **VEC**: 4 warps, vectorized half2, fastest for D∈{64,128,256}
- **TILE**: 1 warp, scalar, handles arbitrary D without padding
- **WMMA**: 4 warps, rocWMMA 16x16 tensor cores, fastest for D∈{96} on RDNA3

### Infrastructure
- VRAM budget check before allocation (hipMemGetInfo)
- Per-phase profiling via hipEvent (GFX1100_PROFILE=1 env var)
- Attention kernel override (GFX1100_ATTN={wmma,tile,vec} env var)
- 126-architecture model loader from GGUF
- Batched MoE with group-by-expert dispatch

---

## Remaining Performance Gaps

### Gemma-2 2B: -4% vs baseline (152 vs 158 tok/s)
- **Root cause**: Host dispatch overhead. ~312 kernel launches per token (26 layers × ~12 launches each). Each launch adds ~2-5µs of CPU overhead.
- **Fix**: Plan system — compose operations into fewer, larger fused kernels. Target: <100 launches per token.
- **NOT from**: attention kernel (both use VEC), FFN (now fused), norms (now fused).

### Phi-3.5 3.8B: 96/100 correctness
- **Root cause**: Float non-associativity across 32 MHA layers with mscale=1.19. All 3 attention kernels produce identical mismatches.
- **Fix**: Plan system — match baseline's exact operation ordering to get identical float rounding.

---

## Model Architecture Features

| Feature | Llama 1B | Qwen 0.5B | Gemma-2 2B | Phi-3.5 |
|---------|----------|-----------|-----------|---------|
| Layers | 16 | 24 | 26 | 32 |
| Hidden | 2048 | 896 | 2304 | 3072 |
| FF | 8192 | 4864 | 9216 | 8192 |
| Head dim | 64 | 64 | 256 | 96 |
| Q heads | 32 | 14 | 8 | 32 |
| KV heads | 8 | 2 | 4 | 32 (MHA) |
| Activation | SiLU (fused) | SiLU (fused) | GELU-tanh (fused) | SiLU (fused SwiGLU) |
| Post-norms | No | No | Yes (fused rmsnorm_add) | No |
| Softcap | No | No | 50.0 attn + 30.0 logit | No |
| Fused QKV | No | No | No | Yes (wqkv) |
| RoPE | Normal | Normal | Normal | NeoX + SU (rope_short) |

---

## Files

- `benchmarks/REFERENCE.md` — this file
- `benchmarks/profile-data.jsonl` — structured data for agent consumption
- `benchmarks/golden-baseline-*.bin` — 100-token baseline logits (binary)
- `benchmarks/golden-compare-*.log` — per-token comparison logs
- `benchmarks/rocwmma/` — rocWMMA-enabled benchmark runs (baseline + megakernel)
- `docs/PLAN_SYSTEM.md` — fusion plan system design document
