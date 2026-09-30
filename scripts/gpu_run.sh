#!/usr/bin/env bash
# Run a sequence of pipeline stages on a GPU claimed through gpuq.
# Usage: scripts/gpu_run.sh <config> <stage> [stage ...]   (env: GPUQ_ARGS, e.g. "--devices 3 -m 20", HOURS)
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=$1; shift
STAGES="$*"
HOURS=${HOURS:-12}
RUN=$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['run_name'])")
LOG=${REVISE_SCRATCH:-/scratch/users/iacopog/revise}/runs/$RUN; mkdir -p "$LOG"
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
INNER="source $CONDA_SH && conda activate revise && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 REVISE_SCRATCH=${REVISE_SCRATCH:-/scratch/users/iacopog/revise} REVISE_DEVICE=cuda REVISE_BATCH_SIZE=${BATCH:-32} && for st in $STAGES; do echo \"[\$(date)] gpu stage \$st\"; python -m revise.cli \$st -c $CFG >> $LOG/gpu_stage_\$st.log 2>&1; done && echo \"[\$(date)] GPU STAGES DONE: $STAGES\""
echo "[$(date)] submitting $CFG stages: $STAGES"
gpuq submit ${GPUQ_ARGS:---queue -m 20} -t "$HOURS" --name "revise-$RUN" -- bash -c "$INNER"
