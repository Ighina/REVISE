#!/usr/bin/env bash
# Llama / Mistral replication on the external and ESTB sources: same stages as chain_phase3.sh / chain_estb.sh,
# with the transfer source set to the same model's MuSiQue 2-hop run and the GPU stages chosen per config
# (variants configs add `variants`, RAGBench adds the Qwen `judge`).  Qwen's retrieval and paraphrase caches are
# copied in first (both are resumable by qid, so only questions Qwen never saw are retrieved / paraphrased).
#   usage: scripts/chain_replication.sh <config> [<config> ...]
set -uo pipefail
cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate revise
export REVISE_SCRATCH=${REVISE_SCRATCH:-/scratch/users/iacopog/revise}
export GPUQ_ARGS=${GPUQ_ARGS:--g 1 -m 40} BATCH=${BATCH:-16} TOKEN_BUDGET=${TOKEN_BUDGET:-32000}
R=$REVISE_SCRATCH/runs
cpu() { local cfg=$1; shift; for st in "$@"; do echo "[$(date)] cpu $cfg $st"; REVISE_DEVICE=cpu REVISE_NUM_THREADS=48 python -m revise.cli $st -c "$cfg" ${EXTRA:-} >> "$R/$RUN/cpu_stage_$st.log" 2>&1 || echo "[$(date)] WARNING cpu stage $st failed for $cfg"; done; }
gpu() { local cfg=$1; shift; HOURS=${HOURS:-6} scripts/gpu_run.sh "$cfg" "$@" || echo "[$(date)] WARNING gpu stages $* failed for $cfg"; }
for CFG in "$@"; do
  echo "[$(date)] ===== $CFG"
  RUN=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['run_name'])"); mkdir -p "$R/$RUN"
  MODEL=$(echo "$RUN" | grep -oE "llama8b|mistral7b")
  SRC=configs/musique_2hop_$MODEL.yaml
  QWEN=${RUN/$MODEL/qwen7b}
  for f in retrieved_e5.jsonl retrieved_bm25.jsonl variants.jsonl; do
    [ -f "$R/$QWEN/$f" ] && [ ! -f "$R/$RUN/$f" ] && cp "$R/$QWEN/$f" "$R/$RUN/$f" && echo "[$(date)] seeded $f from $QWEN"
  done
  STAGES="screen collect"
  grep -q "^variants:" "$CFG" && STAGES="screen variants collect"
  grep -q "dataset: ragbench" "$CFG" && STAGES="screen collect judge"
  gpu "$CFG" $STAGES
  cpu "$CFG" features probes
  EXTRA="--transfer-from $SRC" cpu "$CFG" transfer; EXTRA=""
  gpu "$CFG" patch ablate
  cpu "$CFG" adaptive report
  case "$CFG" in *ragbench*|*hotpotqa*) REVISE_DEVICE=cpu python -m revise.analysis.groups "$CFG" question >> "$R/$RUN/groups.log" 2>&1 ;; esac
done
echo "[$(date)] REPLICATION CHAIN DONE: $*"
