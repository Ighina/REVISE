# Shared helpers for the replication scripts.  Source it; do not run it.
#
# Every script that sources this file runs from the repository root, writes its
# outputs under $REVISE_SCRATCH (default: ./scratch) and logs one file per
# pipeline stage to $REVISE_SCRATCH/runs/<run_name>/stage_<stage>.log.
#
# Environment variables honoured by the pipeline:
#   REVISE_SCRATCH       where weights, datasets, activations and runs live
#   REVISE_DEVICE        cpu | cuda   (default: cuda when available)
#   REVISE_BATCH_SIZE    overrides model.batch_size   (32 is a good GPU value)
#   REVISE_NUM_THREADS   CPU threads for the forward passes
#   REVISE_TOKEN_BUDGET  max prompt tokens per batch (default 48000)
# and by these scripts:
#   PYTHON               interpreter to use (default: python)
#   N_JOBS               parallel probe fits (default: 8)
#   KEEP_GOING=1         carry on after a failing stage instead of stopping
#   DRY_RUN=1            print the commands a script would run, run nothing

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
export REVISE_SCRATCH=${REVISE_SCRATCH:-$REPO_ROOT/scratch}
PYTHON=${PYTHON:-python}
N_JOBS=${N_JOBS:-8}

# Main-text model configs, and the source run whose directions are transferred.
MAIN_CONFIGS=(configs/musique_2hop_qwen7b.yaml configs/musique_2hop_llama8b.yaml configs/musique_2hop_mistral7b.yaml)
SRC_CONFIG=configs/musique_2hop_qwen7b.yaml

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

die() { log "ERROR: $*"; exit 1; }

# run_name_of <config> -- the run directory name a config writes to.
run_name_of() { "$PYTHON" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['run_name'])" "$1"; }

# cfg_get <config> <dotted.key> -- a value from the config *after* defaults are merged.
cfg_get() {
  "$PYTHON" - "$1" "$2" <<'PY'
import sys
from revise.experiment import load_config
value = load_config(sys.argv[1])
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

# stage <config> <stage> [extra cli args...] -- run one pipeline stage.
# Stages are idempotent and resumable, so re-running a script continues it.
stage() {
  local cfg=$1 st=$2; shift 2
  [ -f "$cfg" ] || die "no such config: $cfg"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "    $PYTHON -m revise.cli $st -c $cfg ${*:-}"
    return 0
  fi
  local run logdir
  run=$(run_name_of "$cfg")
  logdir="$REVISE_SCRATCH/runs/$run"; mkdir -p "$logdir"
  log "  $run: $st ${*:-}"
  if "$PYTHON" -m revise.cli "$st" -c "$cfg" "$@" >>"$logdir/stage_$st.log" 2>&1; then
    return 0
  fi
  log "  FAILED: stage $st of $cfg -- last lines of $logdir/stage_$st.log:"
  tail -n 15 "$logdir/stage_$st.log" | sed 's/^/      /'
  [ "${KEEP_GOING:-0}" = "1" ] || exit 1
}

# analysis <module> [args...] -- run one analysis module (CPU only, prints its own summary).
analysis() {
  local mod=$1; shift
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "    $PYTHON -m $mod $*"
    return 0
  fi
  log "  analysis: $mod $*"
  if ! "$PYTHON" -m "$mod" "$@"; then
    log "  FAILED: $mod $*"
    [ "${KEEP_GOING:-0}" = "1" ] || exit 1
  fi
}

# the stages that build a run's probes from scratch, for one config
collect_and_probe() {
  local cfg=$1
  stage "$cfg" screen
  stage "$cfg" collect
  if [ "$(cfg_get "$cfg" conversational.enabled)" = "True" ]; then
    stage "$cfg" conversational
  fi
  stage "$cfg" features
  stage "$cfg" probes --n-jobs "$N_JOBS"
}
