#!/usr/bin/env bash
# Full pipeline on a tiny model to validate the code path (minutes on CPU).
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate revise
export REVISE_SCRATCH=${REVISE_SCRATCH:-/scratch/users/iacopog/revise}
python -m revise.cli all -c configs/smoke.yaml
