#!/usr/bin/env bash
# Canonical RCV1 runner. Usage: bash run_rcv1.sh FOLD [RUN_NAME]
set -euo pipefail

FOLD=${1:?usage: $0 FOLD [RUN_NAME]}
RUN_NAME=${2:-"RCV1-103-H3-fold-${FOLD}"}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# RCV1 fold-0 GPU smoke tests on the RTX A6000 completed safely at batch 32.
PER_GPU_TRAIN_BATCH_SIZE=${PER_GPU_TRAIN_BATCH_SIZE:-32}
export PER_GPU_TRAIN_BATCH_SIZE
exec bash "$SCRIPT_DIR/run_fold.sh" RCV1-103-H3 "$FOLD" "$RUN_NAME"
