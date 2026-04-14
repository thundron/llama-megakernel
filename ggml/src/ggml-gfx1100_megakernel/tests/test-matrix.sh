#!/bin/bash
# test-matrix.sh — Run the full decode test matrix
#
# For each model: generate baseline logits, then compare megakernel output.
# Reports PASS/FAIL per model with tok/s.
#
# Usage: ./test-matrix.sh <build_dir> <models_dir>
# Example: ./test-matrix.sh ../../build /c/Users/thund/.cache/huggingface/hub

set -e
BUILD=${1:?Usage: test-matrix.sh <build_dir> <models_dir>}
MODELS=${2:?Usage: test-matrix.sh <build_dir> <models_dir>}
LOGDIR="${BUILD}/test-matrix-logs"
mkdir -p "$LOGDIR"

BL="${BUILD}/bin/test-baseline-logits.exe"
MK="${BUILD}/bin/test-megakernel-e2e.exe"

PASS=0
FAIL=0
SKIP=0

echo "=== gfx1100 Megakernel Test Matrix ==="
echo ""

run_test() {
    local NAME="$1"
    local MODEL="$2"
    local NTOK="${3:-10}"

    if [ ! -f "$MODEL" ]; then
        echo "SKIP: $NAME (model not found: $MODEL)"
        SKIP=$((SKIP+1))
        return
    fi

    echo -n "TEST: $NAME ... "

    # Step 1: baseline logits
    "$BL" "$MODEL" "$LOGDIR/${NAME}-baseline.bin" "$NTOK" \
        2>"$LOGDIR/${NAME}-baseline.log"
    if [ $? -ne 0 ]; then
        echo "FAIL (baseline crashed)"
        FAIL=$((FAIL+1))
        return
    fi

    # Step 2: megakernel comparison
    "$MK" "$MODEL" "$LOGDIR/${NAME}-baseline.bin" \
        2>"$LOGDIR/${NAME}-megakernel.log"
    RC=$?

    if [ $RC -eq 0 ]; then
        SPEED=$(grep "Speed:" "$LOGDIR/${NAME}-megakernel.log" | head -1 | grep -o '[0-9.]*' | head -1)
        echo "PASS (${SPEED} tok/s)"
        PASS=$((PASS+1))
    else
        echo "FAIL (see $LOGDIR/${NAME}-megakernel.log)"
        grep "MISMATCH\|FAIL\|error" "$LOGDIR/${NAME}-megakernel.log" | head -3
        FAIL=$((FAIL+1))
    fi
}

# Find models in HuggingFace cache
HF="$MODELS"
find_model() {
    find "$HF" -name "$1" 2>/dev/null | head -1
}

# --- Primary: Llama 3.2 1B Q4_K_M (D=64, pure attention) ---
LLAMA=$(find_model "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
run_test "llama-1b-q4km" "$LLAMA"

# --- D=128: Qwen2-0.5B Q4_K_M ---
QWEN=$(find_model "qwen2-0_5b-instruct-q4_k_m.gguf")
run_test "qwen2-0.5b-q4km" "$QWEN"

# --- D=256: Carnice 9B Q4_K_M (DeltaNet hybrid) ---
CARNICE9=$(find_model "Carnice-9b-Q4_K_M.gguf")
run_test "carnice-9b-q4km" "$CARNICE9"

# --- D=256: Carnice 27B Q4_K_M ---
CARNICE27=$(find_model "Carnice-27b-Q4_K_M.gguf")
run_test "carnice-27b-q4km" "$CARNICE27"

# --- Quantization matrix: Llama 1B in all quant types ---
# These require pre-quantized models in models_dir
for Q in Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K_S Q4_K_M Q5_K_M Q6_K \
         IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ1_S IQ1_M IQ4_NL IQ4_XS; do
    MODEL=$(find_model "Llama-3.2-1B-${Q}.gguf" 2>/dev/null)
    if [ -z "$MODEL" ]; then
        MODEL=$(find_model "Llama-3.2-1B-Instruct-${Q}.gguf" 2>/dev/null)
    fi
    run_test "llama-1b-${Q}" "${MODEL:-NOT_FOUND}" 5
done

echo ""
echo "=== Results: ${PASS} PASS, ${FAIL} FAIL, ${SKIP} SKIP ==="
[ $FAIL -eq 0 ] && exit 0 || exit 1
