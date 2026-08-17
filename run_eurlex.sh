#!/usr/bin/env bash
# Canonical Eurlex-4K runner. Usage: bash run_eurlex.sh FOLD [RUN_NAME]
set -euo pipefail

FOLD=${1:?usage: $0 FOLD [RUN_NAME]}
RUN_NAME=${2:-"Eurlex-4k-fold-${FOLD}"}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
EURLEX_CONFIG=${EURLEX_CONFIG:-"$SCRIPT_DIR/configs/eurlex-4k.yaml"}

# Load the versioned YAML when present. Explicitly exported shell variables
# take precedence; YAML values fill only unset variables.
if [[ -f "$EURLEX_CONFIG" ]]; then
  CONFIG_EXPORTS=$("$PYTHON_BIN" - "$EURLEX_CONFIG" <<'PYLOAD'
import os
import shlex
import sys

try:
    import yaml
except ImportError as error:
    raise SystemExit(
        "Eurlex config requires PyYAML in the config-loader Python environment"
    ) from error

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}
if not isinstance(config, dict):
    raise SystemExit("Eurlex YAML root must be a mapping")

def scalar(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    return str(value)

values = {}
environment = config.get("environment", {})
if not isinstance(environment, dict):
    raise SystemExit("Eurlex YAML environment must be a mapping")
values.update(environment)

dataset = config.get("dataset", {})
if isinstance(dataset, dict):
    if "data_root" in dataset:
        values["DATASETS_DIR"] = dataset["data_root"]
    if "prepared_data_dir" in dataset:
        values["PREPARED_DATA_DIR"] = dataset["prepared_data_dir"]

runtime = config.get("runtime", {})
if isinstance(runtime, dict):
    if "python" in runtime:
        values["PYTHON_BIN"] = runtime["python"]
    if "model_name_or_path" in runtime:
        values["MODEL_NAME_OR_PATH"] = runtime["model_name_or_path"]
    if "path_prefix" in runtime:
        values["PATH_PREFIX"] = runtime["path_prefix"]

training = config.get("training", {})
training_env = {
    "per_gpu_train_batch_size": "PER_GPU_TRAIN_BATCH_SIZE",
    "gradient_accumulation_steps": "GRADIENT_ACCUMULATION_STEPS",
    "num_training_steps": "NUM_TRAINING_STEPS",
    "save_steps": "SAVE_STEPS",
    "num_warmup_steps": "NUM_WARMUP_STEPS",
    "learning_rate": "LEARNING_RATE",
    "label_smoothing": "LABEL_SMOOTHING",
    "random_prob": "RANDOM_PROB",
    "keep_prob": "KEEP_PROB",
    "seed": "SEED",
}
if isinstance(training, dict):
    for yaml_key, env_key in training_env.items():
        if yaml_key in training:
            values[env_key] = training[yaml_key]

ranking = config.get("ranking", {})
if isinstance(ranking, dict) and "export" in ranking:
    values["EXPORT_RANKINGS"] = ranking["export"]

for key, value in values.items():
    if not isinstance(key, str) or not key or value is None:
        continue
    if key == "PATH_PREFIX":
        if "PATH_PREFIX" not in os.environ:
            print("export PATH_PREFIX={}".format(shlex.quote(scalar(value))))
            print("export PATH=$PATH_PREFIX:$PATH")
        continue
    if key not in os.environ:
        print("export {}={}".format(key, shlex.quote(scalar(value))))
PYLOAD
  )
  eval "$CONFIG_EXPORTS"
fi

# Eurlex has 3,956 label tokens and long documents; start conservatively.
PER_GPU_TRAIN_BATCH_SIZE=${PER_GPU_TRAIN_BATCH_SIZE:-8}
export PER_GPU_TRAIN_BATCH_SIZE
exec bash "$SCRIPT_DIR/run_fold.sh" Eurlex-4k "$FOLD" "$RUN_NAME"
