#!/usr/bin/env bash
# Table 2 (main text): held-out AUROC of the probe on paired updates for the
# three models on MuSiQue 2-hop, with the nuisance-classifier and
# gain-beyond-nuisance columns and question-level bootstrap intervals.
#
# The same runs produce: the AUROC-by-layer and cross-task transfer figures,
# the split-generalisation and behaviour appendix tables, the closed-book
# appendix numbers, and the directions every later script transfers or
# intervenes on.  Run this script before any of the others.
#
# Stages: screen -> collect -> (conversational, Qwen only) -> features ->
#         probes -> transfer -> closedbook, then the bootstrap intervals.
#
# Cost: the forward passes dominate (about 90k prompts per model); see the
# Runtime section of the README.  Every stage is resumable -- re-run to continue.
#
# Usage: scripts/run_in_domain_results.sh [config ...]   (default: the three models)
source "$(dirname "$0")/lib.sh"

CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=("${MAIN_CONFIGS[@]}")

for CFG in "${CONFIGS[@]}"; do
  log "=== in-domain results: $CFG"
  collect_and_probe "$CFG"
  stage "$CFG" transfer            # cross-task + stateless -> conversational direction transfer
  stage "$CFG" closedbook          # closed-book accuracy of the selected questions (appendix)
  stage "$CFG" report
done

log "=== question-level bootstrap confidence intervals (Table 2)"
analysis revise.analysis.bootstrap "${CONFIGS[@]}"

log "in-domain results done; rebuild the tables with scripts/make_paper_artifacts.sh"
