#!/usr/bin/env bash
# Sequential GPU/CPU chain for the Phase-2 experiments (one GPU job at a time).
#   arg1: PID to wait for before starting (previous GPU job driver), or 0
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate revise
export REVISE_SCRATCH=${REVISE_SCRATCH:-/scratch/users/iacopog/revise} HF_HUB_OFFLINE=1
export GPUQ_ARGS=${GPUQ_ARGS:---devices 3 -m 20} BATCH=${BATCH:-32}
WAIT_PID=${1:-0}
while [ "$WAIT_PID" != "0" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
SRC=configs/musique_2hop_qwen7b.yaml
cpu() { local cfg=$1; shift; for st in "$@"; do echo "[$(date)] cpu $cfg $st"; REVISE_DEVICE=cpu REVISE_NUM_THREADS=64 python -m revise.cli $st -c "$cfg" ${EXTRA:-} >> "$REVISE_SCRATCH/runs/$(python -c "import yaml;print(yaml.safe_load(open('$cfg'))['run_name'])")/cpu_stage_$st.log" 2>&1 || echo "[$(date)] WARNING cpu stage $st failed for $cfg"; done; }
gpu() { local cfg=$1; shift; HOURS=${HOURS:-6} scripts/gpu_run.sh "$cfg" "$@" || echo "[$(date)] WARNING gpu stages $* failed for $cfg"; }

for CFG in configs/musique_2hop_llama8b.yaml configs/musique_2hop_mistral7b.yaml; do
  echo "[$(date)] ===== $CFG"
  gpu "$CFG" screen collect
  cpu "$CFG" features probes
  EXTRA="--transfer-from $SRC" cpu "$CFG" transfer; EXTRA=""
  gpu "$CFG" steer patch ablate
  cpu "$CFG" adaptive report
done
CFG=configs/musique_full_qwen7b.yaml
echo "[$(date)] ===== $CFG"
gpu "$CFG" screen collect
cpu "$CFG" features probes
EXTRA="--transfer-from $SRC" cpu "$CFG" transfer; EXTRA=""
gpu "$CFG" patch ablate
cpu "$CFG" adaptive report
echo "[$(date)] PHASE 2 CHAIN DONE"
