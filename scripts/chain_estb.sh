#!/usr/bin/env bash
# ESTB runs: GPU stages (screen [variants] collect, patch ablate) and CPU analysis per config.
#   usage: scripts/chain_estb.sh <config> [<config> ...]   (env GPU_PRE="screen variants collect" to add the variants stage)
set -uo pipefail
cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate revise
export REVISE_SCRATCH=${REVISE_SCRATCH:-/scratch/users/iacopog/revise}
export GPUQ_ARGS=${GPUQ_ARGS:---devices 3 -m 20} BATCH=${BATCH:-16} TOKEN_BUDGET=${TOKEN_BUDGET:-16000}
SRC=configs/musique_2hop_qwen7b.yaml
cpu() { local cfg=$1; shift; for st in "$@"; do echo "[$(date)] cpu $cfg $st"; REVISE_DEVICE=cpu REVISE_NUM_THREADS=48 python -m revise.cli $st -c "$cfg" ${EXTRA:-} >> "$REVISE_SCRATCH/runs/$(python -c "import yaml;print(yaml.safe_load(open('$cfg'))['run_name'])")/cpu_stage_$st.log" 2>&1 || echo "[$(date)] WARNING cpu stage $st failed for $cfg"; done; }
gpu() { local cfg=$1; shift; HOURS=${HOURS:-6} scripts/gpu_run.sh "$cfg" "$@" || echo "[$(date)] WARNING gpu stages $* failed for $cfg"; }
for CFG in "$@"; do
  echo "[$(date)] ===== $CFG"
  RUN=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['run_name'])"); mkdir -p "$REVISE_SCRATCH/runs/$RUN"
  gpu "$CFG" ${GPU_PRE:-screen collect}
  cpu "$CFG" features probes
  EXTRA="--transfer-from $SRC" cpu "$CFG" transfer; EXTRA=""
  gpu "$CFG" patch ablate
  cpu "$CFG" adaptive report
done
echo "[$(date)] ESTB CHAIN DONE: $*"
