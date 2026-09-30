#!/usr/bin/env bash
# Phase-1 experiment: the two stateless regimes are collected as two concurrent
# processes (THREADS_EACH threads each), then the remaining stages run
# sequentially.  Resumable: re-run to continue.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate revise
export REVISE_SCRATCH=${REVISE_SCRATCH:-/scratch/users/iacopog/revise}
CFG=${1:-configs/musique_2hop_qwen7b.yaml}
THREADS_EACH=${THREADS_EACH:-40}   # per collector process; the machine is shared, so no NUMA pinning
RUN=$(python -c "import yaml,sys; print(yaml.safe_load(open('$CFG'))['run_name'])")
LOG=$REVISE_SCRATCH/runs/$RUN; mkdir -p "$LOG"
echo "[$(date)] start $CFG -> $LOG"
python -m revise.cli screen -c "$CFG" >> "$LOG/stage_screen.log" 2>&1
echo "[$(date)] screen done"
REVISE_NUM_THREADS=$THREADS_EACH python -m revise.cli collect -c "$CFG" -r evidence_only      >> "$LOG/stage_collect_evidence_only.log" 2>&1 &
P0=$!
REVISE_NUM_THREADS=$THREADS_EACH python -m revise.cli collect -c "$CFG" -r evidence_augmented >> "$LOG/stage_collect_evidence_augmented.log" 2>&1 &
P1=$!
wait $P0; wait $P1
echo "[$(date)] collect done"
python -m revise.cli conversational -c "$CFG" >> "$LOG/stage_conversational.log" 2>&1
echo "[$(date)] conversational done"
python -m revise.cli features -c "$CFG" >> "$LOG/stage_features.log" 2>&1
python -m revise.cli probes   -c "$CFG" --n-jobs 16 >> "$LOG/stage_probes.log" 2>&1
python -m revise.cli transfer -c "$CFG" >> "$LOG/stage_transfer.log" 2>&1
echo "[$(date)] probes done"
python -m revise.cli steer -c "$CFG" >> "$LOG/stage_steer.log" 2>&1
python -m revise.cli patch -c "$CFG" >> "$LOG/stage_patch.log" 2>&1
python -m revise.cli adaptive -c "$CFG" >> "$LOG/stage_adaptive.log" 2>&1
python -m revise.cli report   -c "$CFG" >> "$LOG/stage_report.log" 2>&1
echo "[$(date)] ALL DONE -> $LOG/report/summary.md"
