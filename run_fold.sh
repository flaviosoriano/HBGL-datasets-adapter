#!/usr/bin/env bash
# Train one canonical dataset fold without sharing prepared data or feature cache.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 DATASET FOLD [RUN_NAME]" >&2
  echo "DATASET must be WOS-150-H2 or RCV1-103-H3" >&2
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

case "$DATASET" in
  WOS-150-H2)
    MAX_SOURCE_LENGTH=509
    MAX_TARGET_LENGTH=3
    LABEL_CPT_STEPS=300
    EXTRA_LABEL_FLAGS=(--label_cpt_not_incr_mask_ratio)
    ;;
  RCV1-103-H3)
    MAX_SOURCE_LENGTH=492
    MAX_TARGET_LENGTH=5
    LABEL_CPT_STEPS=100
    EXTRA_LABEL_FLAGS=()
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
  python3 "$SCRIPT_DIR/run.py"
  --dataset-dir "$DATASET_DIR"
  --dataset-name "$DATASET"
  --fold "$FOLD"
  --prepared-data-dir "$PREPARED_DATA_DIR"
  --output_dir "$OUTPUT_DIR"
  --model_type bert
  --model_name_or_path bert-base-uncased
  --do_lower_case
  --max_source_seq_length "$MAX_SOURCE_LENGTH"
  --max_target_seq_length "$MAX_TARGET_LENGTH"
  --per_gpu_train_batch_size 12
  --gradient_accumulation_steps 1
  --label_smoothing 0
  --learning_rate 3e-5
  --num_warmup_steps 500
  --num_training_steps 96000
  --cache_dir "$CACHE_DIR"
  --save_steps 3000
  --random_prob 0
  --keep_prob 0
  --soft_label
  --seed 42
  --random_label_init
  --label_cpt_steps "$LABEL_CPT_STEPS"
  --label_cpt_use_bce
)
COMMAND+=("${EXTRA_LABEL_FLAGS[@]}")
if [[ "${HBGL_WANDB:-0}" == "1" ]]; then
  COMMAND+=(--wandb)
fi

printf 'Running %q ' "${COMMAND[@]}"
printf '\n'
exec "${COMMAND[@]}"
