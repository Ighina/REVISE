#!/usr/bin/env bash
# Collect + probe a transfer dataset (single regime) and score the source run's directions on it.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate revise
export REVISE_SCRATCH=${REVISE_SCRATCH:-/scratch/users/iacopog/revise}
CFG=${1:-configs/twowiki_qwen7b.yaml}
SRC=${2:-configs/musique_2hop_qwen7b.yaml}
RUN=$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['run_name'])")
LOG=$REVISE_SCRATCH/runs/$RUN; mkdir -p "$LOG"
echo "[$(date)] start $CFG (transfer from $SRC)"
REVISE_NUM_THREADS=${THREADS:-64} python -m revise.cli screen  -c "$CFG" >> "$LOG/stage_screen.log" 2>&1
echo "[$(date)] screen done"
REVISE_NUM_THREADS=${THREADS:-64} python -m revise.cli collect -c "$CFG" >> "$LOG/stage_collect.log" 2>&1
echo "[$(date)] collect done"
python -m revise.cli features -c "$CFG" >> "$LOG/stage_features.log" 2>&1
python -m revise.cli probes   -c "$CFG" --n-jobs 16 >> "$LOG/stage_probes.log" 2>&1
python -m revise.cli transfer -c "$CFG" --transfer-from "$SRC" >> "$LOG/stage_transfer.log" 2>&1
python -m revise.cli adaptive -c "$CFG" >> "$LOG/stage_adaptive.log" 2>&1
python -m revise.cli report   -c "$CFG" >> "$LOG/stage_report.log" 2>&1
echo "[$(date)] ALL DONE -> $LOG/report/summary.md"
