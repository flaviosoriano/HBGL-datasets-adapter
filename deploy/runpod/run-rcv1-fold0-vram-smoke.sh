#!/usr/bin/env bash
# Run inside the HGBL tmux pane. Stage canonical RCV1 if needed, then build a
# deliberately small, taxonomy-faithful fold-0 subset for a quick VRAM test.
set -euo pipefail

ROOT=/workspace/flaviossf
CONDA_DIR="$ROOT/miniconda3"
HBGL_ENV="$CONDA_DIR/envs/hbgl"
REPO_DIR="$ROOT/HBGL"
DATASETS_DIR="$ROOT/datasets"
DATASET=RCV1-103-H3
FOLD=0
HF_DATASET_REPO=${HF_DATASET_REPO:-LBD-UFMG/RCV1-103-H3}
BATCH_SIZE=${BATCH_SIZE:-12}
TRAINING_STEPS=${TRAINING_STEPS:-12}
LABEL_CPT_STEPS=${LABEL_CPT_STEPS:-100}
TRAIN_LIMIT=${TRAIN_LIMIT:-256}
VALID_LIMIT=${VALID_LIMIT:-128}
TEST_LIMIT=${TEST_LIMIT:-128}
RUN_ID=${RUN_ID:-"rcv1-fold0-vram-smoke-b${BATCH_SIZE}-$(date -u +%Y%m%dT%H%M%SZ)"}
DATASET_DIR="$DATASETS_DIR/$DATASET"
SMOKE_PREPARED_DIR="$ROOT/smoke-prepared/$DATASET/fold_${FOLD}/$RUN_ID"
OUTPUT_DIR="$REPO_DIR/models/$RUN_ID"
CACHE_DIR="$REPO_DIR/.cache/$DATASET/fold_${FOLD}/$RUN_ID"
PRETRAINED_DIR="$ROOT/pretrained/bert-base-uncased"
LOG_DIR="$ROOT/logs/$RUN_ID"

[[ -x "$HBGL_ENV/bin/python" ]] || { echo "Missing HBGL environment: $HBGL_ENV" >&2; exit 1; }
[[ -d "$REPO_DIR/.git" ]] || { echo "Missing HBGL repository: $REPO_DIR" >&2; exit 1; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Refusing to overwrite existing output: $OUTPUT_DIR" >&2; exit 1; }
[[ ! -e "$SMOKE_PREPARED_DIR" ]] || { echo "Refusing to overwrite prepared smoke data: $SMOKE_PREPARED_DIR" >&2; exit 1; }

if [[ ! -f "$DATASET_DIR/samples.pkl" || ! -f "$DATASET_DIR/fold_0/train.pkl" || ! -f "$DATASET_DIR/fold_0/val.pkl" || ! -f "$DATASET_DIR/fold_0/test.pkl" ]]; then
    : "${HF_TOKEN:?HF_TOKEN must be present in this tmux pane to stage $DATASET.}"
    export HF_HOME="$ROOT/.cache/huggingface"
    mkdir -p "$DATASET_DIR"
    HF_DATASET_REPO="$HF_DATASET_REPO" DATASET_DIR="$DATASET_DIR" \
        "$CONDA_DIR/bin/conda" run --no-capture-output --prefix "$HBGL_ENV" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["HF_DATASET_REPO"], repo_type="dataset",
    local_dir=os.environ["DATASET_DIR"], local_dir_use_symlinks=False,
    token=os.environ["HF_TOKEN"],
)
PY
fi

for path in samples.pkl fold_0/train.pkl fold_0/val.pkl fold_0/test.pkl; do
    [[ -f "$DATASET_DIR/$path" ]] || { echo "Dataset artifact missing after staging: $DATASET_DIR/$path" >&2; exit 1; }
done

# Build only the examples required to exercise training/evaluation, while
# retaining the complete official RCV1 label taxonomy and stable label IDs.
REPO_DIR="$REPO_DIR" DATASET_DIR="$DATASET_DIR" SMOKE_PREPARED_DIR="$SMOKE_PREPARED_DIR" \
TRAIN_LIMIT="$TRAIN_LIMIT" VALID_LIMIT="$VALID_LIMIT" TEST_LIMIT="$TEST_LIMIT" \
    PYTHONPATH="$REPO_DIR" "$CONDA_DIR/bin/conda" run --no-capture-output --prefix "$HBGL_ENV" python - <<'PY'
import json
import os
import pickle
from pathlib import Path
import dataset_adapter as adapter

repo = Path(os.environ['REPO_DIR'])
dataset_dir = Path(os.environ['DATASET_DIR'])
target = Path(os.environ['SMOKE_PREPARED_DIR'])
limits = {'train': int(os.environ['TRAIN_LIMIT']), 'val': int(os.environ['VALID_LIMIT']), 'test': int(os.environ['TEST_LIMIT'])}
samples = adapter.load_samples(dataset_dir)
splits = adapter.load_fold_ids(dataset_dir, 0)
adapter.validate_dataset(samples, splits, 'RCV1-103-H3', 0)
code_taxonomy = adapter.load_code_taxonomy(repo / 'data/rcv1/rcv1.taxonomy')
code_to_label = json.loads((repo / 'data/rcv1/rcv1_topic_codes.json').read_text(encoding='utf-8'))
taxonomy, id_to_label, fallbacks = adapter.build_rcv1_taxonomy(samples, code_taxonomy, code_to_label)
depths = adapter.compute_depths(taxonomy)
label_map = {label: f'[A_{label_id}]' for label_id, label in id_to_label.items()}
target.mkdir(parents=True, exist_ok=False)
for split, limit in limits.items():
    adapter._write_jsonl(
        target / f'{split}.jsonl',
        adapter._make_rows(samples, splits[split][:limit], split, id_to_label, label_map, depths),
    )
with (target / 'label_map.pkl').open('wb') as handle:
    pickle.dump(label_map, handle, protocol=4)
adapter._write_taxonomy(target / 'label_taxonomy.tsv', taxonomy)
(target / 'manifest.json').write_text(json.dumps({
    'kind': 'vram-smoke-subset', 'dataset': 'RCV1-103-H3', 'fold': 0,
    'counts': limits, 'labels': len(label_map), 'max_depth': max(depths.values()),
    'taxonomy_fallbacks': fallbacks,
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print('Prepared RCV1 VRAM subset:', limits, 'labels=', len(label_map))
PY

# Avoid retired Transformers 2.x S3 checkpoint URLs by using a local Hub copy.
if [[ ! -f "$PRETRAINED_DIR/config.json" || ! -f "$PRETRAINED_DIR/pytorch_model.bin" || ! -f "$PRETRAINED_DIR/vocab.txt" ]]; then
    export HF_HOME="$ROOT/.cache/huggingface"
    PRETRAINED_DIR="$PRETRAINED_DIR" "$CONDA_DIR/bin/conda" run --no-capture-output --prefix "$HBGL_ENV" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id='bert-base-uncased', local_dir=os.environ['PRETRAINED_DIR'], local_dir_use_symlinks=False, allow_patterns=['config.json', 'pytorch_model.bin', 'vocab.txt'])
PY
fi

mkdir -p "$LOG_DIR" "$CACHE_DIR"
printf 'run_id=%s\ndataset=%s\nfold=%s\nbatch_size=%s\ntraining_steps=%s\nlabel_cpt_steps=%s\nsubset=%s/%s/%s\nrepo_commit=%s\n' \
    "$RUN_ID" "$DATASET" "$FOLD" "$BATCH_SIZE" "$TRAINING_STEPS" "$LABEL_CPT_STEPS" "$TRAIN_LIMIT" "$VALID_LIMIT" "$TEST_LIMIT" \
    "$(git -C "$REPO_DIR" rev-parse HEAD)" > "$LOG_DIR/RUN_INFO"

"$CONDA_DIR/bin/conda" run --no-capture-output --prefix "$HBGL_ENV" \
    python "$REPO_DIR/run.py" \
    --train_file "$SMOKE_PREPARED_DIR/train.jsonl" \
    --valid_file "$SMOKE_PREPARED_DIR/val.jsonl" \
    --test_file "$SMOKE_PREPARED_DIR/test.jsonl" \
    --add_vocab_file "$SMOKE_PREPARED_DIR/label_map.pkl" \
    --label_cpt "$SMOKE_PREPARED_DIR/label_taxonomy.tsv" \
    --output_dir "$OUTPUT_DIR" \
    --model_type bert \
    --model_name_or_path "$PRETRAINED_DIR" \
    --do_lower_case \
    --max_source_seq_length 492 \
    --max_target_seq_length 5 \
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
    --label_cpt_steps "$LABEL_CPT_STEPS" \
    --label_cpt_use_bce \
    2>&1 | tee "$LOG_DIR/train.log"
