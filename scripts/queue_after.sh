#!/usr/bin/env bash
# Wait for a PID to exit, then run the given steering configs and refresh the report.
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate revise
export REVISE_SCRATCH=${REVISE_SCRATCH:-/scratch/users/iacopog/revise}
WAIT_PID=$1; shift
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
LOG=$REVISE_SCRATCH/runs/musique2hop_qwen7b
for CFG in "$@"; do
  echo "[$(date)] steer $CFG"
  REVISE_NUM_THREADS=64 python -m revise.cli steer -c "$CFG" >> "$LOG/stage_steer_large.log" 2>&1
done
python -m revise.cli report -c configs/musique_2hop_qwen7b.yaml >> "$LOG/stage_report3.log" 2>&1
echo "[$(date)] LARGE-ALPHA STEERING DONE"
