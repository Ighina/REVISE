#!/usr/bin/env bash
# Resume the Llama/Mistral replication after the 2026-10-02 pause (GPU 3 shared with other jobs -> OOM).
# 1. redo the ablate stage that OOM'd for the Mistral BM25-distractor run, then its last CPU stages;
# 2. continue chain_replication.sh with the runs not yet done (collection is resumable, so the
#    partially collected Mistral BM25-full run picks up where it stopped).
#   usage: setsid nohup scripts/resume_replication.sh >> $REVISE_SCRATCH/runs/replication_chain.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate revise
export REVISE_SCRATCH=${REVISE_SCRATCH:-/scratch/users/iacopog/revise}
export GPUQ_ARGS=${GPUQ_ARGS:--g 1 -m 40} BATCH=${BATCH:-16} TOKEN_BUDGET=${TOKEN_BUDGET:-32000}
C=configs
echo "[$(date)] ===== RESUME: ablate for $C/musique_2hop_mistral7b_bm25_noise.yaml"
HOURS=2 scripts/gpu_run.sh $C/musique_2hop_mistral7b_bm25_noise.yaml ablate || echo "[$(date)] WARNING ablate failed again"
for st in adaptive report; do
  REVISE_DEVICE=cpu REVISE_NUM_THREADS=48 python -m revise.cli $st -c $C/musique_2hop_mistral7b_bm25_noise.yaml \
    >> "$REVISE_SCRATCH/runs/musique2hop_mistral7b_bm25_noise/cpu_stage_$st.log" 2>&1 || echo "[$(date)] WARNING cpu $st failed"
done
exec scripts/chain_replication.sh \
  $C/musique_2hop_mistral7b_bm25_full.yaml $C/musique_2hop_mistral7b_variants.yaml \
  $C/hotpotqa_llama8b.yaml $C/ragbench_llama8b.yaml $C/synthetic_llama8b.yaml \
  $C/hotpotqa_mistral7b.yaml $C/ragbench_mistral7b.yaml $C/synthetic_mistral7b.yaml
