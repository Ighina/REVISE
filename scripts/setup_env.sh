#!/usr/bin/env bash
# Create the Python environment for the pipeline.
#
#   scripts/setup_env.sh            # venv in ./.venv, CPU-only PyTorch
#   CUDA=1 scripts/setup_env.sh     # the default PyTorch wheel (CUDA)
#   CONDA=1 scripts/setup_env.sh    # a conda environment named "revise" instead
#
# Python 3.10 or newer is required.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

if [ "${CONDA:-0}" = "1" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda create -y -n revise python=3.11
  conda activate revise
  ACTIVATE="conda activate revise"
else
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  ACTIVATE="source .venv/bin/activate"
fi

pip install --upgrade pip
if [ "${CUDA:-0}" = "1" ]; then
  pip install torch
else
  pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install -r requirements.txt
pip install -e .

echo
echo "done.  activate the environment with:  $ACTIVATE"
echo "then check the installation with:      scripts/run_smoke.sh"
