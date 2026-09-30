#!/usr/bin/env bash
# Re-run the causal stages (steer, patch, adaptive) + report on an existing run.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate revise
export REVISE_SCRATCH=${REVISE_SCRATCH:-/scratch/users/iacopog/revise}
CFG=${1:-configs/musique_2hop_qwen7b.yaml}
RUN=$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['run_name'])")
LOG=$REVISE_SCRATCH/runs/$RUN
echo "[$(date)] causal rerun start"
REVISE_NUM_THREADS=${THREADS:-64} python -m revise.cli steer    -c "$CFG" >> "$LOG/stage_steer2.log" 2>&1
echo "[$(date)] steer done"
REVISE_NUM_THREADS=${THREADS:-64} python -m revise.cli patch    -c "$CFG" >> "$LOG/stage_patch2.log" 2>&1
echo "[$(date)] patch done"
python -m revise.cli adaptive -c "$CFG" >> "$LOG/stage_adaptive2.log" 2>&1
python -m revise.cli report   -c "$CFG" >> "$LOG/stage_report2.log" 2>&1
echo "[$(date)] CAUSAL RERUN DONE -> $LOG/report/summary.md"
