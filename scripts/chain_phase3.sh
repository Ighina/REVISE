#!/usr/bin/env bash
# Phase 3 (review response): HotpotQA -> RAGBench -> synthetic, one GPU job at a time; CPU analysis in between.
set -uo pipefail
cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate revise
export REVISE_SCRATCH=${REVISE_SCRATCH:-/scratch/users/iacopog/revise} HF_HUB_OFFLINE=0
export GPUQ_ARGS=${GPUQ_ARGS:---devices 3 -m 20} BATCH=${BATCH:-32}
SRC=configs/musique_2hop_qwen7b.yaml
cpu() { local cfg=$1; shift; for st in "$@"; do echo "[$(date)] cpu $cfg $st"; REVISE_DEVICE=cpu REVISE_NUM_THREADS=48 python -m revise.cli $st -c "$cfg" ${EXTRA:-} >> "$REVISE_SCRATCH/runs/$(python -c "import yaml;print(yaml.safe_load(open('$cfg'))['run_name'])")/cpu_stage_$st.log" 2>&1 || echo "[$(date)] WARNING cpu stage $st failed for $cfg"; done; }
gpu() { local cfg=$1; shift; HOURS=${HOURS:-6} scripts/gpu_run.sh "$cfg" "$@" || echo "[$(date)] WARNING gpu stages $* failed for $cfg"; }
for CFG in configs/hotpotqa_qwen7b.yaml configs/ragbench_qwen7b.yaml configs/synthetic_qwen7b.yaml; do
  echo "[$(date)] ===== $CFG"
  RUN=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['run_name'])"); mkdir -p "$REVISE_SCRATCH/runs/$RUN"
  # dataset download/caching happens on CPU first (needs network; the GPU job runs offline)
  REVISE_DEVICE=cpu python -c "
import yaml, revise
from revise.data.loaders import load_questions
d = yaml.safe_load(open('$CFG'))['data']
kw = dict(split=d['split'], limit=d.get('pool_limit'), hops=d.get('hops')); kw.update(d.get('loader_kwargs') or {})
if d.get('types'): kw['types'] = d['types']
print('cached', len(load_questions(d['dataset'], **kw)), 'questions')" >> "$REVISE_SCRATCH/runs/$RUN/cpu_stage_prepare.log" 2>&1
  gpu "$CFG" screen collect
  cpu "$CFG" features probes
  EXTRA="--transfer-from $SRC" cpu "$CFG" transfer; EXTRA=""
  gpu "$CFG" patch ablate
  cpu "$CFG" adaptive report
done
echo "[$(date)] PHASE 3 CHAIN DONE"
