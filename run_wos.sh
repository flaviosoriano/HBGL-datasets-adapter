#!/usr/bin/env bash
# Canonical Web of Science runner. Usage: bash run_wos.sh FOLD [RUN_NAME]
set -euo pipefail

FOLD=${1:?usage: $0 FOLD [RUN_NAME]}
RUN_NAME=${2:-"WOS-150-H2-fold-${FOLD}"}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# WOS fold-0 GPU smoke tests on the RTX A6000 completed safely at batch 32.
PER_GPU_TRAIN_BATCH_SIZE=${PER_GPU_TRAIN_BATCH_SIZE:-32}
export PER_GPU_TRAIN_BATCH_SIZE
exec bash "$SCRIPT_DIR/run_fold.sh" WOS-150-H2 "$FOLD" "$RUN_NAME"
