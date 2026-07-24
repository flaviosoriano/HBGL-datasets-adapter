#!/usr/bin/env bash
# Run inside the HGBL tmux pane. Downloads only when WOS is not already staged,
# then performs a short fold-0 training smoke test for VRAM observation.
set -euo pipefail

ROOT=/workspace/flaviossf
CONDA_DIR="$ROOT/miniconda3"
HBGL_ENV="$CONDA_DIR/envs/hbgl"
REPO_DIR="$ROOT/HBGL"
DATASETS_DIR="$ROOT/datasets"
DATASET=WOS-150-H2
FOLD=0
HF_DATASET_REPO=${HF_DATASET_REPO:-LBD-UFMG/WOS-150-H2}
BATCH_SIZE=${BATCH_SIZE:-12}
TRAINING_STEPS=${TRAINING_STEPS:-12}
RUN_ID=${RUN_ID:-"wos-fold0-vram-smoke-b${BATCH_SIZE}-$(date -u +%Y%m%dT%H%M%SZ)"}
DATASET_DIR="$DATASETS_DIR/$DATASET"
PREPARED_DATA_DIR="$REPO_DIR/resource/prepared-datasets"
OUTPUT_DIR="$REPO_DIR/models/$RUN_ID"
CACHE_DIR="$REPO_DIR/.cache/$DATASET/fold_${FOLD}/$RUN_ID"
PRETRAINED_DIR="$ROOT/pretrained/bert-base-uncased"
LOG_DIR="$ROOT/logs/$RUN_ID"

[[ -x "$HBGL_ENV/bin/python" ]] || { echo "Missing HBGL environment: $HBGL_ENV" >&2; exit 1; }
[[ -d "$REPO_DIR/.git" ]] || { echo "Missing HBGL repository: $REPO_DIR" >&2; exit 1; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Refusing to overwrite existing output: $OUTPUT_DIR" >&2; exit 1; }

if [[ ! -f "$DATASET_DIR/samples.pkl" || ! -f "$DATASET_DIR/fold_0/train.pkl" || ! -f "$DATASET_DIR/fold_0/val.pkl" || ! -f "$DATASET_DIR/fold_0/test.pkl" ]]; then
    : "${HF_TOKEN:?HF_TOKEN must be present in this tmux pane to stage $DATASET.}"
    export HF_HOME="$ROOT/.cache/huggingface"
    mkdir -p "$DATASET_DIR"
    HF_DATASET_REPO="$HF_DATASET_REPO" DATASET_DIR="$DATASET_DIR" \
        "$CONDA_DIR/bin/conda" run --no-capture-output --prefix "$HBGL_ENV" python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_DATASET_REPO"],
    repo_type="dataset",
    local_dir=os.environ["DATASET_DIR"],
    local_dir_use_symlinks=False,
    token=os.environ["HF_TOKEN"],
)
PY
fi

for path in samples.pkl fold_0/train.pkl fold_0/val.pkl fold_0/test.pkl; do
    [[ -f "$DATASET_DIR/$path" ]] || { echo "Dataset artifact missing after staging: $DATASET_DIR/$path" >&2; exit 1; }
done

# Source validation and dataset preparation happen before the expensive model run.
"$CONDA_DIR/bin/conda" run --no-capture-output --prefix "$HBGL_ENV" \
    python "$REPO_DIR/dataset_adapter.py" validate \
    --dataset-dir "$DATASET_DIR" --dataset-name "$DATASET"

# Transformers 2.x maps bert-base-uncased to a retired S3 URL.  Pre-stage the
# exact public checkpoint from the Hub and make the legacy loader use that path.
if [[ ! -f "$PRETRAINED_DIR/config.json" || ! -f "$PRETRAINED_DIR/pytorch_model.bin" || ! -f "$PRETRAINED_DIR/vocab.txt" ]]; then
    export HF_HOME="$ROOT/.cache/huggingface"
    PRETRAINED_DIR="$PRETRAINED_DIR" "$CONDA_DIR/bin/conda" run --no-capture-output --prefix "$HBGL_ENV" python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bert-base-uncased",
    local_dir=os.environ["PRETRAINED_DIR"],
    local_dir_use_symlinks=False,
    allow_patterns=["config.json", "pytorch_model.bin", "vocab.txt"],
)
PY
fi

mkdir -p "$LOG_DIR" "$CACHE_DIR"
printf 'run_id=%s\ndataset=%s\nfold=%s\nbatch_size=%s\ntraining_steps=%s\nrepo_commit=%s\n' \
    "$RUN_ID" "$DATASET" "$FOLD" "$BATCH_SIZE" "$TRAINING_STEPS" \
    "$(git -C "$REPO_DIR" rev-parse HEAD)" > "$LOG_DIR/RUN_INFO"

"$CONDA_DIR/bin/conda" run --no-capture-output --prefix "$HBGL_ENV" \
    python "$REPO_DIR/run.py" \
    --dataset-dir "$DATASET_DIR" \
    --dataset-name "$DATASET" \
    --fold "$FOLD" \
    --prepared-data-dir "$PREPARED_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_type bert \
    --model_name_or_path "$PRETRAINED_DIR" \
    --do_lower_case \
    --max_source_seq_length 509 \
    --max_target_seq_length 3 \
    --per_gpu_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps 1 \
    --label_smoothing 0 \
    --learning_rate 3e-5 \
    --num_warmup_steps 0 \
    --num_training_steps "$TRAINING_STEPS" \
    --cache_dir "$CACHE_DIR" \
    --save_steps 1000 \
    --random_prob 0 \
    --keep_prob 0 \
    --soft_label \
    --seed 42 \
    --random_label_init \
    --label_cpt_steps 300 \
    --label_cpt_use_bce \
    --label_cpt_not_incr_mask_ratio \
    2>&1 | tee "$LOG_DIR/train.log"
