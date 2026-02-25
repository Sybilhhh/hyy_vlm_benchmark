#!/bin/bash
# ============================================================================
# Run inference across all training checkpoints for each benchmark.
#
# For each checkpoint-N/dit.safetensors found in TRAIN_OUTPUT_DIR, runs
# infer_unified.py against every benchmark defined below.
#
# Supports multi-GPU data parallelism: set NUM_GPUS > 1 to distribute
# CSV rows across GPUs (each GPU loads a full pipeline independently).
#
# Usage:
#   bash scripts/run_inference.sh                  # single GPU
#   NUM_GPUS=4 bash scripts/run_inference.sh       # 4 GPUs
# ============================================================================

set -euo pipefail

# ---- Training output directory (contains checkpoint-*/) ----
TRAIN_OUTPUT_DIR="outputs/prop_run5_0225_0350"

# ---- Model config (same one used for training) ----
CONF_PATH="configs/k5_unified.yaml"

# ---- Inference output root ----

INFER_OUTPUT_DIR="outputs/prop_run5_0225_0350/benchmarks"

# ---- Inference hyperparams ----
NUM_STEPS=50
SEED=42
TIME_LENGTH=5

# ---- Hardware ----
NUM_GPUS=4

# ---- Flags ----
EXTRA_FLAGS=""
# Uncomment to use NF4 quantized Qwen (saves ~10GB VRAM)
# EXTRA_FLAGS="${EXTRA_FLAGS} --quantized_qwen"
# Uncomment to expand prompts with Qwen before generation
# EXTRA_FLAGS="${EXTRA_FLAGS} --expand_prompts"
# Uncomment to limit samples per benchmark (for quick testing)
# EXTRA_FLAGS="${EXTRA_FLAGS} --max_samples 1"

# ============================================================================
# Benchmark definitions
#
# Each benchmark is a pair: CSV_PATH|DATA_ROOT
# - CSV_PATH: path to the benchmark CSV (task is read from the 'task' column)
# - DATA_ROOT: root directory for relative paths in CSV (use "" if absolute)
# ============================================================================
BENCHMARKS=(
    "/scratch/dyvm6xra/dyvm6xrauserzhefan/DataGeneration/TestData/Benchmark/propagation/prop_benchmark_ood_test_head20.csv|/scratch/dyvm6xra/dyvm6xrauserzhefan/DataGeneration/TestData"
)

# ============================================================================
# 默认按 step 从大到小推理（最新 checkpoint 优先）
CHECKPOINTS=$(find "${TRAIN_OUTPUT_DIR}" -name "dit.safetensors" -path "*/checkpoint-*/dit.safetensors" | sort -t'-' -k2 -nr)

if [ -z "${CHECKPOINTS}" ]; then
    echo "No checkpoints found in ${TRAIN_OUTPUT_DIR}/checkpoint-*/"
    exit 1
fi

echo "=========================================="
echo " Kandinsky-5 Inference Runner"
echo "=========================================="
echo " Config:       ${CONF_PATH}"
echo " Train dir:    ${TRAIN_OUTPUT_DIR}"
echo " Output dir:   ${INFER_OUTPUT_DIR}"
echo " Num steps:    ${NUM_STEPS}"
echo " Seed:         ${SEED}"
echo " GPUs:         ${NUM_GPUS}"
echo " Benchmarks:   ${#BENCHMARKS[@]}"
echo " Checkpoints:"
for ckpt in ${CHECKPOINTS}; do
    step=$(echo "${ckpt}" | grep -oP 'checkpoint-\K[0-9]+')
    echo "   - checkpoint-${step}"
done
echo "=========================================="

if [ "${NUM_GPUS}" -gt 1 ]; then
    LAUNCH_CMD="torchrun --nproc_per_node=${NUM_GPUS}"
else
    LAUNCH_CMD="python"
fi

for CKPT_PATH in ${CHECKPOINTS}; do
    STEP=$(echo "${CKPT_PATH}" | grep -oP 'checkpoint-\K[0-9]+')

    for BENCH_ENTRY in "${BENCHMARKS[@]}"; do
        IFS='|' read -r CSV_PATH DATA_ROOT <<< "${BENCH_ENTRY}"
        BENCH_NAME=$(basename "$(dirname "${CSV_PATH}")")

        OUTPUT_DIR="${INFER_OUTPUT_DIR}/checkpoint-${STEP}/${BENCH_NAME}"
        mkdir -p "${OUTPUT_DIR}"

        echo ""
        echo ">>> checkpoint-${STEP} | ${BENCH_NAME}"
        echo "    dit:    ${CKPT_PATH}"
        echo "    csv:    ${CSV_PATH}"
        echo "    output: ${OUTPUT_DIR}"
        echo "    gpus:   ${NUM_GPUS}"

        DATA_ROOT_ARG=""
        if [ -n "${DATA_ROOT}" ]; then
            DATA_ROOT_ARG="--data_root ${DATA_ROOT}"
        fi

        ${LAUNCH_CMD} infer_unified.py \
            --conf_path "${CONF_PATH}" \
            --dit_checkpoint "${CKPT_PATH}" \
            --csv_path "${CSV_PATH}" \
            ${DATA_ROOT_ARG} \
            --output_dir "${OUTPUT_DIR}" \
            --num_steps ${NUM_STEPS} \
            --seed ${SEED} \
            --time_length ${TIME_LENGTH} \
            ${EXTRA_FLAGS}
    done
done

echo ""
echo "=========================================="
echo " All inference complete."
echo " Results in: ${INFER_OUTPUT_DIR}"
echo "=========================================="
