#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate revise
export REVISE_SCRATCH=/scratch/users/iacopog/revise HF_HUB_OFFLINE=1 REVISE_DEVICE=cpu REVISE_NUM_THREADS=48
until grep -q "PHASE 2 CHAIN DONE" $REVISE_SCRATCH/runs/phase2_chain.log; do sleep 120; done
L=$REVISE_SCRATCH/runs/musiquefull_qwen7b
echo "[$(date)] re-probing document split with the fixed split"
python -m revise.cli probes -c configs/musique_full_qwen7b.yaml --splits document --n-jobs 16 >> $L/refresh_probes_document.log 2>&1
python -m revise.cli transfer -c configs/musique_full_qwen7b.yaml --transfer-from configs/musique_2hop_qwen7b.yaml >> $L/refresh_transfer.log 2>&1
python -m revise.cli adaptive -c configs/musique_full_qwen7b.yaml >> $L/refresh_adaptive.log 2>&1
python -m revise.cli report -c configs/musique_full_qwen7b.yaml >> $L/refresh_report.log 2>&1
echo "[$(date)] MULTIHOP REFRESH DONE"
