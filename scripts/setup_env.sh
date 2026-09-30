#!/usr/bin/env bash
# Create the `revise` conda environment with a CPU-only PyTorch build.
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n revise python=3.11
conda activate revise
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r "$(dirname "$0")/../requirements.txt"
pip install -e "$(dirname "$0")/.."
echo "done. activate with: conda activate revise"
