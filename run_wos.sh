#!/usr/bin/env bash
# Canonical Web of Science runner. Usage: bash run_wos.sh FOLD [RUN_NAME]
set -euo pipefail

FOLD=${1:?usage: $0 FOLD [RUN_NAME]}
RUN_NAME=${2:-"WOS-150-H2-fold-${FOLD}"}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/run_fold.sh" WOS-150-H2 "$FOLD" "$RUN_NAME"
