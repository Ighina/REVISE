#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate revise
export REVISE_SCRATCH=/scratch/users/iacopog/revise
HOURS=3 BATCH=32 GPUQ_ARGS="--devices 3 -m 20" scripts/gpu_run.sh configs/ragbench_qwen7b.yaml judge
for st in features probes adaptive report; do echo "[$(date)] cpu $st"; REVISE_DEVICE=cpu REVISE_NUM_THREADS=48 python -m revise.cli $st -c configs/ragbench_qwen7b.yaml >> $REVISE_SCRATCH/runs/ragbench_qwen7b/cpu_stage_${st}_judged.log 2>&1; done
REVISE_DEVICE=cpu python -m revise.analysis.groups configs/ragbench_qwen7b.yaml question >> $REVISE_SCRATCH/runs/ragbench_qwen7b/groups_judged.log 2>&1
echo "[$(date)] RAGBENCH JUDGE DONE"
