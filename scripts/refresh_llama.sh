#!/usr/bin/env bash
# After the chain moves past Llama, rebuild Llama features/probes/adaptive/report on the topped-up data (CPU).
set -uo pipefail
cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate revise
export REVISE_SCRATCH=/scratch/users/iacopog/revise HF_HUB_OFFLINE=1 REVISE_DEVICE=cpu REVISE_NUM_THREADS=32
LOG=$REVISE_SCRATCH/runs/musique2hop_llama8b
until grep -q "===== configs/musique_2hop_mistral7b.yaml" $REVISE_SCRATCH/runs/phase2_chain.log; do sleep 120; done
for st in features probes adaptive report; do echo "[$(date)] refresh $st"; python -m revise.cli $st -c configs/musique_2hop_llama8b.yaml >> $LOG/refresh_stage_$st.log 2>&1; done
echo "[$(date)] LLAMA REFRESH DONE"
