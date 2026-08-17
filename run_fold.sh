#!/usr/bin/env bash
# Train one canonical dataset fold without sharing prepared data or feature cache.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 DATASET FOLD [RUN_NAME]" >&2
  echo "DATASET must be WOS-150-H2, RCV1-103-H3, or Eurlex-4k" >&2
  exit 2
fi

DATASET=$1
FOLD=$2
RUN_NAME=${3:-"${DATASET}-fold-${FOLD}"}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATASETS_DIR=${DATASETS_DIR:-"${SCRIPT_DIR}/../datasets"}
PREPARED_DATA_DIR=${PREPARED_DATA_DIR:-"${SCRIPT_DIR}/resource/prepared-datasets"}
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/models/${RUN_NAME}"}
CACHE_DIR=${CACHE_DIR:-"${SCRIPT_DIR}/.cache/${DATASET}/fold_${FOLD}"}
PER_GPU_TRAIN_BATCH_SIZE=${PER_GPU_TRAIN_BATCH_SIZE:-12}
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-bert-base-uncased}
EXPORT_RANKINGS=${EXPORT_RANKINGS:-1}
FORCE_PREPARE=${FORCE_PREPARE:-0}
NUM_TRAINING_STEPS=${NUM_TRAINING_STEPS:-96000}
SAVE_STEPS=${SAVE_STEPS:-3000}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NUM_WARMUP_STEPS=${NUM_WARMUP_STEPS:-500}
LEARNING_RATE=${LEARNING_RATE:-3e-5}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0}
RANDOM_PROB=${RANDOM_PROB:-0}
KEEP_PROB=${KEEP_PROB:-0}
SEED=${SEED:-42}
# Transformers 2.x resolves the shortcut through a retired S3 URL. Prefer the
# Hub-staged local checkpoint when this RunPod layout is available.
LOCAL_BERT_DIR=/workspace/flaviossf/pretrained/bert-base-uncased
if [[ "$MODEL_NAME_OR_PATH" == "bert-base-uncased" && -f "$LOCAL_BERT_DIR/config.json" && -f "$LOCAL_BERT_DIR/pytorch_model.bin" ]]; then
  MODEL_NAME_OR_PATH=$LOCAL_BERT_DIR
fi

case "$DATASET" in
  WOS-150-H2)
    MAX_SOURCE_LENGTH=509
    MAX_TARGET_LENGTH=3
    LABEL_CPT_STEPS=300
    EXTRA_LABEL_FLAGS=(--label_cpt_not_incr_mask_ratio)
    LABEL_INIT_FLAGS=(--random_label_init --label_cpt_steps "$LABEL_CPT_STEPS" --label_cpt_use_bce)
    ;;
  RCV1-103-H3)
    MAX_SOURCE_LENGTH=492
    MAX_TARGET_LENGTH=5
    LABEL_CPT_STEPS=100
    EXTRA_LABEL_FLAGS=()
    LABEL_INIT_FLAGS=(--random_label_init --label_cpt_steps "$LABEL_CPT_STEPS" --label_cpt_use_bce)
    ;;
  Eurlex-4k)
    MAX_SOURCE_LENGTH=510
    MAX_TARGET_LENGTH=2
    EXTRA_LABEL_FLAGS=()
    LABEL_INIT_FLAGS=()
    ;;
  *)
    echo "Unsupported dataset: $DATASET" >&2
    exit 2
    ;;
esac

DATASET_DIR="${DATASETS_DIR}/${DATASET}"
[[ -d "$DATASET_DIR" ]] || { echo "Dataset directory not found: $DATASET_DIR" >&2; exit 1; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Output already exists: $OUTPUT_DIR (choose a new RUN_NAME)" >&2; exit 1; }
mkdir -p "$CACHE_DIR"

COMMAND=(
  "$PYTHON_BIN" "$SCRIPT_DIR/run.py"
  --dataset-dir "$DATASET_DIR"
  --dataset-name "$DATASET"
  --fold "$FOLD"
  --prepared-data-dir "$PREPARED_DATA_DIR"
  --output_dir "$OUTPUT_DIR"
  --model_type bert
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --do_lower_case
  --max_source_seq_length "$MAX_SOURCE_LENGTH"
  --max_target_seq_length "$MAX_TARGET_LENGTH"
  --per_gpu_train_batch_size "$PER_GPU_TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --label_smoothing "$LABEL_SMOOTHING"
  --learning_rate "$LEARNING_RATE"
  --num_warmup_steps "$NUM_WARMUP_STEPS"
  --num_training_steps "$NUM_TRAINING_STEPS"
  --cache_dir "$CACHE_DIR"
  --save_steps "$SAVE_STEPS"
  --random_prob "$RANDOM_PROB"
  --keep_prob "$KEEP_PROB"
  --soft_label
  --soft_label_hier_real
  --seed "$SEED"
)
COMMAND+=("${LABEL_INIT_FLAGS[@]}")
if [[ "$FORCE_PREPARE" == "1" ]]; then
  COMMAND+=(--force-prepare)
elif [[ "$FORCE_PREPARE" != "0" ]]; then
  echo "FORCE_PREPARE must be 0 or 1" >&2
  exit 2
fi
COMMAND+=("${EXTRA_LABEL_FLAGS[@]}")
if [[ "$EXPORT_RANKINGS" == "1" ]]; then
  COMMAND+=(--export-rankings --ranking-cutoffs 1 5 10)
fi
if [[ "${HBGL_WANDB:-0}" == "1" ]]; then
  COMMAND+=(--wandb)
fi

printf 'Running %q ' "${COMMAND[@]}"
printf '\n'
exec "${COMMAND[@]}"
