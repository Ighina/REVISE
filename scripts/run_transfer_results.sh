#!/usr/bin/env bash
# The transfer appendix: do the MuSiQue 2-hop directions still separate the
# transitions on another dataset and at other hop counts, without retraining?
#
#   twowiki_qwen7b      2WikiMultiHopQA (compositional and inference questions)
#   musique_full_qwen7b MuSiQue 3-4 hop, which also adds leave-one-out chains
#
# Both runs are also rows of the split-generalisation and causal tables, and
# the 3-4 hop run is the "Qwen 3--4 hop" row of Table 4.  The stateless ->
# conversational transfer of the same appendix comes from the Qwen MuSiQue run
# itself (its `transfer` stage, in run_in_domain_results.sh).
#
# Requires: scripts/run_in_domain_results.sh (the source directions).
#
# Usage: scripts/run_transfer_results.sh [config ...]
source "$(dirname "$0")/lib.sh"

CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(configs/twowiki_qwen7b.yaml configs/musique_full_qwen7b.yaml)

for CFG in "${CONFIGS[@]}"; do
  log "=== transfer target: $CFG"
  collect_and_probe "$CFG"
  stage "$CFG" transfer --transfer-from "$SRC_CONFIG"
  stage "$CFG" adaptive
  stage "$CFG" report
done

log "transfer results done (the causal rows of these runs come from scripts/run_causal_results.sh)"
