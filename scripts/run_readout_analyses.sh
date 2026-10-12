#!/usr/bin/env bash
# The "what do the probes read?" appendix, all CPU-only re-analyses of the
# activations collected by scripts/run_in_domain_results.sh:
#
#   representation_ablation  the same tasks from h(C_{k-1}), h(C_k), the update
#                            Delta h, or both states concatenated
#   behaviour_control        sufficiency probes that control for what the model
#                            actually output: an output-only classifier, and the
#                            probe stratified and refitted within the contexts
#                            where the model abstains or answers
#   verbal_vs_internal       the model's verbalised INSUFFICIENT judgement
#                            against the state probe at the same anchor,
#                            including the paragraph-count-matched contexts
#
# Requires: scripts/run_in_domain_results.sh.
#
# Usage: scripts/run_readout_analyses.sh [config ...]
source "$(dirname "$0")/lib.sh"

CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=("${MAIN_CONFIGS[@]}")

log "=== state versus update representations"
analysis revise.analysis.representation_ablation "${CONFIGS[@]}"
log "=== output-controlled sufficiency probes"
analysis revise.analysis.behaviour_control "${CONFIGS[@]}"
log "=== verbalised versus internal sufficiency"
analysis revise.analysis.verbal_vs_internal "${CONFIGS[@]}"

log "read-out analyses done"
