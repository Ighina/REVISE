#!/usr/bin/env bash
# The adaptive-retrieval appendix: every learned direction used as a stopping
# rule ("retrieve again unless the state says the evidence is sufficient") and
# as an answer verifier, against the full-context, oracle, behavioural and
# next-token-entropy baselines.
#
# CPU only -- it replays the behaviour and activations already collected, so it
# needs no forward passes.  The `adaptive` stage writes both adaptive/results.json
# (the headline policies) and adaptive/directions.json (every direction, with the
# accuracy / rounds trade-off curves of the appendix figure).
#
# Requires: the runs of scripts/run_in_domain_results.sh, and for the remaining
# columns those of run_transfer_results.sh and run_external_results.sh.
#
# Usage: scripts/run_adaptive_results.sh [config ...]
source "$(dirname "$0")/lib.sh"

CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=("${MAIN_CONFIGS[@]}" configs/hotpotqa_qwen7b.yaml \
  configs/twowiki_qwen7b.yaml configs/synthetic_qwen7b.yaml)

for CFG in "${CONFIGS[@]}"; do
  log "=== adaptive retrieval: $CFG"
  stage "$CFG" adaptive
done

log "adaptive results done"
