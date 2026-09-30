#!/usr/bin/env bash
# wait for job A, run the low-dose dm-steer follow-up on Qwen, then the Phase-2 chain
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate revise
export REVISE_SCRATCH=/scratch/users/iacopog/revise
while kill -0 "$1" 2>/dev/null; do sleep 60; done
HOURS=2 BATCH=32 GPUQ_ARGS="--devices 3 -m 20" scripts/gpu_run.sh configs/musique_2hop_qwen7b_dmsteer_small.yaml ablate report
exec scripts/chain_phase2.sh 0
