#!/usr/bin/env bash
# Top up the RAGBench collection with a small token budget, then continue the Phase 3 chain.
set -uo pipefail
cd "$(dirname "$0")/.."
export REVISE_SCRATCH=/scratch/users/iacopog/revise
HOURS=3 BATCH=8 TOKEN_BUDGET=12000 GPUQ_ARGS="--devices 3 -m 20" scripts/gpu_run.sh configs/ragbench_qwen7b.yaml collect
TOKEN_BUDGET=16000 BATCH=16 exec scripts/chain_phase3.sh configs/ragbench_qwen7b.yaml configs/synthetic_qwen7b.yaml
