#!/usr/bin/env bash
# Validate the installation end to end on Qwen2.5-0.5B: every stage of the
# pipeline on 24 questions, about ten minutes on CPU.  Run this first -- it
# downloads the tiny model and MuSiQue, and exercises the same code path as the
# paper runs.
#
# Usage: scripts/run_smoke.sh [config]            (default: configs/smoke.yaml)
source "$(dirname "$0")/lib.sh"

CFG=${1:-configs/smoke.yaml}
log "smoke test: $CFG (scratch: $REVISE_SCRATCH)"
for st in screen collect conversational features probes transfer steer patch adaptive report; do
  stage "$CFG" "$st"
done
log "smoke test done -> $REVISE_SCRATCH/runs/$(run_name_of "$CFG")/report/summary.md"
