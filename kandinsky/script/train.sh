#!/bin/bash
# ============================================================================
# Kandinsky-5 Unified Training (t2v / t2i / ti2i / tv2v)
#
# Loads from an existing T2V Lite checkpoint (visual_cond=True, 33ch)
# or any other checkpoint, with automatic channel expansion if needed.
#
# Usage:
#   bash scripts/train_unified.sh                        # single GPU
#   accelerate launch --num_processes 8 train_unified.py  # multi-GPU (manual)
# ============================================================================

set -euo pipefail

# ---- Paths (edit these) ----
CONF_PATH="configs/k5_unified.yaml"
DIT_CHECKPOINT="/scratch/dyvm6xra/dyvm6xrauser13/models/kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s/model/kandinsky5lite_t2v_sft_5s.safetensors"
CSV_PATH="/scratch/dyvm6xra/dyvm6xrauserzhefan/DataGeneration/TestData/ditto_500/data_info_200_prop.csv"
DATA_ROOT="/scratch/dyvm6xra/dyvm6xrauserzhefan/DataGeneration/TestData"                    # set if CSV paths are relative to a root
OUTPUT_DIR="outputs/prop_run5_0225_0350"

# ---- Training hyperparams ----
LEARNING_RATE=1e-5
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION=1
MAX_TRAIN_STEPS=100000
MAX_FRAMES=81                   # 5s video at 24fps / temporal_compression=4 + 1
NUM_WORKERS=4

# ---- Scheduler ----
LR_SCHEDULER="constant_with_warmup"
LR_WARMUP_STEPS=1000

# ---- Checkpointing ----
CHECKPOINTING_STEPS=100
LOG_INTERVAL=10

# ---- Hardware ----
MIXED_PRECISION="bf16"
NUM_GPUS=8

# ---- Flags ----
EXTRA_FLAGS=""
EXTRA_FLAGS="${EXTRA_FLAGS} --gradient_checkpointing"
# Uncomment to use NF4 quantized Qwen (saves ~10GB VRAM)
# EXTRA_FLAGS="${EXTRA_FLAGS} --quantized_qwen"

# ---- Benchmark during training (set to "" to disable) ----
# Run benchmark inference every N steps (must be a multiple of CHECKPOINTING_STEPS)
BENCHMARK_EVERY="500"
# BENCHMARK_EVERY="1000"
BENCHMARK_NUM_STEPS=50
BENCHMARK_MAX_SAMPLES=2

# Benchmark CSVs and data roots (parallel arrays; task is read from CSV 'task' column)
BENCHMARK_CSVS=(
    #"/scratch/dyvm6xra/dyvm6xrauserzhefan/DataGeneration/TestData/Benchmark/propagation/metadata.csv"
    "/scratch/dyvm6xra/dyvm6xrauserzhefan/DataGeneration/TestData/ditto_500/prop_ood_test.csv"
)
BENCHMARK_DATA_ROOTS=(
    "/scratch/dyvm6xra/dyvm6xrauserzhefan/DataGeneration/TestData"
)

BENCHMARK_ARGS=""
if [ -n "${BENCHMARK_EVERY}" ]; then
    BENCHMARK_ARGS="--benchmark_every_n_steps ${BENCHMARK_EVERY}"
    BENCHMARK_ARGS="${BENCHMARK_ARGS} --benchmark_csv ${BENCHMARK_CSVS[*]}"
    BENCHMARK_ARGS="${BENCHMARK_ARGS} --benchmark_data_root ${BENCHMARK_DATA_ROOTS[*]}"
    BENCHMARK_ARGS="${BENCHMARK_ARGS} --benchmark_num_steps ${BENCHMARK_NUM_STEPS}"
    if [ -n "${BENCHMARK_MAX_SAMPLES:-}" ]; then
        BENCHMARK_ARGS="${BENCHMARK_ARGS} --benchmark_max_samples ${BENCHMARK_MAX_SAMPLES}"
    fi
fi

# ---- Build data_root arg ----
DATA_ROOT_ARG=""
if [ -n "${DATA_ROOT}" ]; then
    DATA_ROOT_ARG="--data_root ${DATA_ROOT}"
fi

# ---- Resume (leave empty to train from scratch) ----
#RESUME_FROM="/home/dyvm6xra/.../outputs/prop_run/checkpoint-6"
# To resume from a specific checkpoint:
# RESUME_FROM="/path/to/outputs/prop_run/checkpoint-6"
# To resume from latest checkpoint in OUTPUT_DIR:
# RESUME_FROM="latest"
#RESUME_FROM="/home/dyvm6xra/dyvm6xrauser04/yuyang/algorithm/202602_proporgation/kandinsky-5/outputs/prop_run3/checkpoint-1400"

RESUME_FROM=""
RESUME_ARG=""
if [ -n "${RESUME_FROM}" ]; then
    RESUME_ARG="--resume_from_checkpoint ${RESUME_FROM}"
fi

# ============================================================================
echo "=========================================="
echo " Kandinsky-5 Unified Training"
echo "=========================================="
echo " Config:          ${CONF_PATH}"
echo " DiT Checkpoint:  ${DIT_CHECKPOINT}"
echo " CSV:             ${CSV_PATH}"
echo " Output:          ${OUTPUT_DIR}"
echo " GPUs:            ${NUM_GPUS}"
echo " Batch size:      ${TRAIN_BATCH_SIZE} x ${GRADIENT_ACCUMULATION} accum"
echo " LR:              ${LEARNING_RATE}"
echo " Max steps:       ${MAX_TRAIN_STEPS}"
echo " Channel expand:  enabled (auto zero-pad if ckpt mismatches)"
echo "=========================================="

ACC_CONFIG="acc_config/accelerate_config.yaml"

if [ "${NUM_GPUS}" -gt 1 ]; then
    CMD="accelerate launch --config_file ${ACC_CONFIG}"
else
    CMD="python"
fi

${CMD} train_unified.py \
    --conf_path "${CONF_PATH}" \
    --dit_checkpoint "${DIT_CHECKPOINT}" \
    --csv_path ${CSV_PATH} \
    ${DATA_ROOT_ARG} \
    --output_dir "${OUTPUT_DIR}" \
    --learning_rate ${LEARNING_RATE} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --max_train_steps ${MAX_TRAIN_STEPS} \
    --max_frames ${MAX_FRAMES} \
    --dataloader_num_workers ${NUM_WORKERS} \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_warmup_steps ${LR_WARMUP_STEPS} \
    --checkpointing_steps ${CHECKPOINTING_STEPS} \
    --log_interval ${LOG_INTERVAL} \
    --mixed_precision ${MIXED_PRECISION} \
    --allow_channel_expansion \
    ${RESUME_ARG} \
    ${EXTRA_FLAGS} \
    ${BENCHMARK_ARGS}
