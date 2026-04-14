# gfx1100 Megakernel Test Matrix

Complete coverage: 1 small model per (quant_type x head_dim x layer_type).

## Dimension A: Quantization Types (21 from baseline mmvq.cu)

All types need working: matvec, embedding, LM head, gate+up+silu.

| ID | Type | QK | qi | vdr | Block Bytes | Priority |
|----|------|----|----|-----|-------------|----------|
| 2 | Q4_0 | 32 | 4 | 2 | 18 | P0 |
| 3 | Q4_1 | 32 | 4 | 2 | 20 | P1 |
| 6 | Q5_0 | 32 | 4 | 2 | 22 | P1 |
| 7 | Q5_1 | 32 | 4 | 2 | 24 | P1 |
| 8 | Q8_0 | 32 | 8 | 2 | 34 | P0 |
| 10 | Q2_K | 256 | 16 | 1 | 84 | P0 |
| 11 | Q3_K | 256 | 16 | 1 | 110 | P0 |
| 12 | Q4_K | 256 | 32 | 2 | 144 | DONE |
| 13 | Q5_K | 256 | 32 | 2 | 176 | P0 |
| 14 | Q6_K | 256 | 32 | 1 | 210 | DONE |
| 16 | IQ2_XXS | 256 | 16 | 2 | 66 | P2 |
| 17 | IQ2_XS | 256 | 16 | 2 | 98 | P2 |
| 22 | IQ2_S | 256 | 16 | 2 | 100 | P2 |
| 18 | IQ3_XXS | 256 | 16 | 2 | 98 | P2 |
| 21 | IQ3_S | 256 | 16 | 2 | 152 | P2 |
| 19 | IQ1_S | 256 | 8 | 1 | 66 | P3 |
| 29 | IQ1_M | 256 | 8 | 1 | 64 | P3 |
| 20 | IQ4_NL | 32 | 4 | 2 | 18 | P1 |
| 23 | IQ4_XS | 256 | 32 | 4 | 162 | P1 |
| 39 | MXFP4 | 32 | 4 | 2 | 17 | P3 |
| 40 | NVFP4 | 64 | 8 | 4 | 36 | P3 |

## Dimension B: Head Dimensions (actually used by models)

| D | VEC | Smallest Test Model | Params | Q4_K Size |
|---|-----|---------------------|--------|-----------|
| 64 | Y | Llama 3.2 1B | 1B | ~600MB |
| 128 | Y | Qwen2 0.5B | 0.5B | ~300MB |
| 128 | Y | Qwen3.5 0.8B (hybrid) | 0.8B | ~480MB |
| 256 | Y | Gemma 2B | 2B | ~1.2GB |

Note: D=80/96/112 not used by any real models. Our VEC kernel covers D={64,128,256}.

## Dimension C: Layer Types

| Type | Test Model |
|------|------------|
| Pure Attention | Qwen 2.5 0.5B, Llama 3.2 1B |
| DeltaNet Hybrid (attn + recurrent) | Carnice 9B (24 DN + 8 attn) |

## Test Matrix: Models to Download

One base model per head dimension. Quantize each to all quant types.

### Primary test model (covers most quants): Llama 3.2 1B (D=64)

Use `llama-quantize` to produce one GGUF per quant type:

```bash
# Base: Llama-3.2-1B-Instruct (F16 GGUF from HuggingFace)
# Size: ~2.5GB F16, ~0.7GB Q4_K_M
for Q in Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K_S Q4_K_M Q5_K_M Q6_K \
         IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ1_S IQ1_M IQ4_NL IQ4_XS; do
  ./llama-quantize Llama-3.2-1B-Instruct-F16.gguf Llama-3.2-1B-${Q}.gguf ${Q}
done
```

### Secondary test models (other head dimensions):

| Model | D | Params | Q4_K Size | Purpose |
|-------|---|--------|-----------|---------|
| Qwen2-0.5B | 128 | 0.5B | ~300MB | D=128 pure attention |
| Qwen3.5-0.8B | 128 | 0.8B | ~480MB | D=128 + DeltaNet hybrid |
| Gemma-2B | 256 | 2B | ~1.2GB | D=256 pure attention |

Only need Q4_K_M for secondary models (testing D, not quant type).

## Test Script: test-megakernel-matrix.sh

```bash
#!/bin/bash
# Run complete test matrix
# Usage: ./test-megakernel-matrix.sh <models_dir>

MODELS_DIR=${1:-.}
PASS=0
FAIL=0
SKIP=0

echo "=== gfx1100 Megakernel Test Matrix ==="

# Dimension A: All quant types on Llama 1B (D=64)
for Q in Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K_S Q4_K_M Q5_K_M Q6_K \
         IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ1_S IQ1_M IQ4_NL IQ4_XS; do
  MODEL="${MODELS_DIR}/Llama-3.2-1B-${Q}.gguf"
  if [ ! -f "$MODEL" ]; then
    echo "SKIP: $MODEL (not found)"
    SKIP=$((SKIP+1))
    continue
  fi
  echo -n "TEST: Llama-1B ${Q} ... "
  ./test-megakernel-e2e "$MODEL" 2>"/tmp/megakernel-${Q}.log"
  if [ $? -eq 0 ]; then
    echo "PASS"
    PASS=$((PASS+1))
  else
    echo "FAIL (see /tmp/megakernel-${Q}.log)"
    FAIL=$((FAIL+1))
  fi
done

# Dimension B: Head dimensions (Q4_K_M only)
for MODEL_NAME in SmolLM2-135M StableLM-2-1.6B Phi-3.5-mini Gemma-2-2B Carnice-9B; do
  MODEL="${MODELS_DIR}/${MODEL_NAME}-Q4_K_M.gguf"
  if [ ! -f "$MODEL" ]; then
    echo "SKIP: $MODEL (not found)"
    SKIP=$((SKIP+1))
    continue
  fi
  echo -n "TEST: ${MODEL_NAME} Q4_K_M ... "
  ./test-megakernel-e2e "$MODEL" 2>"/tmp/megakernel-${MODEL_NAME}.log"
  if [ $? -eq 0 ]; then
    echo "PASS"
    PASS=$((PASS+1))
  else
    echo "FAIL (see /tmp/megakernel-${MODEL_NAME}.log)"
    FAIL=$((FAIL+1))
  fi
done

echo ""
echo "=== Results: ${PASS} PASS, ${FAIL} FAIL, ${SKIP} SKIP ==="
[ $FAIL -eq 0 ] && exit 0 || exit 1
```

## Acceptance Criteria

For each test:
1. Model loads without error
2. Megakernel compiles (hipcc JIT) without error
3. 10 tokens generated without NaN
4. All logits non-zero (no dead outputs)
5. Greedy argmax produces non-zero tokens (not all token_id=0)
6. Performance: faster than baseline llama.cpp HIP backend

## Expected Total Tests

- 19 quant types x Llama 1B = 19 tests (2 already pass: Q4_K, Q6_K)
- 5 additional models x Q4_K = 5 tests
- **Total: 24 tests** (2 pass, 22 to implement)

## Current Status

| Test | Status |
|------|--------|
| Llama 1B Q4_K_M (D=64) | PASS (318 tok/s) |
| Llama 1B Q6_K (D=64) | PASS |
| Carnice 9B Q4_K_M (D=256, DeltaNet) | FAIL (DN outputs zeros) |
| All others | NOT IMPLEMENTED |
