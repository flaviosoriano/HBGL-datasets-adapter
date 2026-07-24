#!/usr/bin/env bash
# Bootstrap fallback for a Pod-local Volume Disk.
# Base image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
# Runtime created below: Python 3.8.18 + torch 1.8.2/cu111 + transformers 2.10.0.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

ROOT=/workspace/flaviossf
CONDA_DIR="$ROOT/miniconda3"
HBGL_ENV="$CONDA_DIR/envs/hbgl"
REPO_DIR="$ROOT/HBGL"
DATASETS_DIR="$ROOT/datasets"
HBGL_REPO=https://github.com/flaviosoriano/HBGL-datasets-adapter.git
HBGL_REVISION=3aa7ecf2e3249aacd095171a068f1449c4c6b99c
MINICONDA_NAME=Miniconda3-py38_23.11.0-2-Linux-x86_64.sh
MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_NAME"
MINICONDA_SHA256=cb908ddbd603d789d94076ea4dd3f8517b15866719e007725dca778a8dfab823

apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl git openssh-server tmux ninja-build build-essential
rm -rf /var/lib/apt/lists/*

# RunPod supplies this environment variable when an SSH public key is configured.
if [[ -n "${SSH_PUBLIC_KEY:-}" ]]; then
    install -d -m 700 /root/.ssh
    printf '%s\n' "$SSH_PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi
mkdir -p /run/sshd
pgrep -x sshd >/dev/null || /usr/sbin/sshd

mkdir -p "$ROOT"
if [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
    installer=$(mktemp --suffix=.sh /tmp/hbgl-miniconda-XXXXXX)
    curl --fail --location --retry 5 --output "$installer" "$MINICONDA_URL"
    echo "$MINICONDA_SHA256  $installer" | sha256sum --check --status
    bash "$installer" -b -p "$CONDA_DIR"
    rm -f "$installer"
fi

RUNTIME_MARKER="$ROOT/.hbgl-runtime-legacy-v2.ready"
if [[ ! -f "$RUNTIME_MARKER" ]]; then
    if [[ ! -x "$HBGL_ENV/bin/python" ]]; then
        "$CONDA_DIR/bin/conda" create --yes --prefix "$HBGL_ENV" python=3.8.18 pip
    fi
    "$CONDA_DIR/bin/conda" run --prefix "$HBGL_ENV" python -m pip install --no-cache-dir 'pip<24.1'
    "$CONDA_DIR/bin/conda" run --prefix "$HBGL_ENV" python -m pip install --no-cache-dir \
        'torch==1.8.2' \
        --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
    "$CONDA_DIR/bin/conda" run --prefix "$HBGL_ENV" python -m pip install --no-cache-dir \
        'numpy==1.19.5' \
        'boto3==1.17.112' \
        'requests==2.25.1' \
        'tqdm==4.64.1' \
        'wandb==0.10.33' \
        'sentencepiece==0.1.99' \
        'transformers==2.10.0' \
        'tokenizers==0.7.0' \
        'sacremoses==0.0.53' \
        'filelock==3.0.12' \
        'huggingface_hub==0.20.3' \
        'regex==2021.11.10' \
        'protobuf==3.20.3'
    "$CONDA_DIR/bin/conda" run --prefix "$HBGL_ENV" python -m pip check
    touch "$RUNTIME_MARKER"
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
    git clone --depth 1 "$HBGL_REPO" "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --depth 1 origin "$HBGL_REVISION"
git -C "$REPO_DIR" checkout --detach "$HBGL_REVISION"
[[ "$(git -C "$REPO_DIR" rev-parse HEAD)" == "$HBGL_REVISION" ]]

"$CONDA_DIR/bin/conda" run --prefix "$HBGL_ENV" python - <<'PY'
import torch
import transformers

assert torch.__version__.startswith("1.8.2"), torch.__version__
assert torch.version.cuda == "11.1", torch.version.cuda
assert transformers.__version__ == "2.10.0", transformers.__version__
assert torch.cuda.is_available(), "CUDA is unavailable inside the HBGL environment"
capability = torch.cuda.get_device_capability()
assert capability in {(8, 0), (8, 6)}, (
    f"HBGL legacy runtime only permits Ampere GPUs (sm80/sm86); got sm{capability[0]}{capability[1]}"
)
print({
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": capability,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
})
PY

cat > "$ROOT/hbgl-runtime.env" <<EOF
export HBGL_ROOT=$ROOT
export HBGL_ENV=$HBGL_ENV
export HBGL_REPO_DIR=$REPO_DIR
export DATASETS_DIR=$DATASETS_DIR
EOF
chmod 600 "$ROOT/hbgl-runtime.env"

# The runtime intentionally does not download datasets or start a training run.
# Each fold Pod stages its own canonical datasets below DATASETS_DIR, then launches
# from its own tmux session, e.g.:
# source /workspace/flaviossf/hbgl-runtime.env
# tmux new-session -A -s hbgl
# /workspace/flaviossf/miniconda3/bin/conda run -p "$HBGL_ENV" \
#   bash "$REPO_DIR/run_fold.sh" RCV1-103-H3 0 hbgl-rcv1-fold-0
mkdir -p "$DATASETS_DIR"
tmux has-session -t hbgl 2>/dev/null || tmux new-session -d -s hbgl
printf 'HBGL legacy runtime is ready at %s; revision=%s; datasets=%s\n' \
    "$REPO_DIR" "$HBGL_REVISION" "$DATASETS_DIR"
exec sleep infinity
