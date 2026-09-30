#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -q "===== configs/musique_full_qwen7b.yaml" /scratch/users/iacopog/revise/runs/phase2_chain.log; do sleep 120; done
HOURS=2 BATCH=32 GPUQ_ARGS="--devices 3 -m 20" scripts/gpu_run.sh configs/musique_2hop_mistral7b_dmsteer_small.yaml ablate report
