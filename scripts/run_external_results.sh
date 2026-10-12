#!/usr/bin/env bash
# Table 5 (main text): the external benchmarks for Qwen2.5-7B -- HotpotQA
# (independent multi-hop), RAGBench (naturally retrieved evidence over five
# domains, mostly single-hop) and the synthetic controlled benchmark.
#
# Each dataset gets within-dataset probes, the MuSiQue 2-hop directions applied
# without retraining, and the patching and projection-out tests; the per-group
# AUROCs of the appendix are computed last.  RAGBench answers are scored by the
# LLM judge, which is why it has an extra `judge` stage.
#
# Requires: scripts/run_in_domain_results.sh (the MuSiQue directions to transfer).
#
# Usage: scripts/run_external_results.sh [config ...]
source "$(dirname "$0")/lib.sh"

CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(configs/hotpotqa_qwen7b.yaml configs/ragbench_qwen7b.yaml configs/synthetic_qwen7b.yaml)

# the grouping split each dataset's appendix table is reported on
group_split_for() {
  case "$1" in
    *hotpotqa*) echo "document" ;;
    *) echo "question" ;;
  esac
}

for CFG in "${CONFIGS[@]}"; do
  log "=== external benchmark: $CFG"
  stage "$CFG" screen
  stage "$CFG" collect
  if [ "$(cfg_get "$CFG" data.dataset)" = "ragbench" ]; then
    stage "$CFG" judge           # free-form answers: score them with the LLM judge
  fi
  stage "$CFG" features
  stage "$CFG" probes --n-jobs "$N_JOBS"
  stage "$CFG" transfer --transfer-from "$SRC_CONFIG"
  stage "$CFG" patch
  stage "$CFG" ablate
  stage "$CFG" adaptive
  stage "$CFG" report
  analysis revise.analysis.groups "$CFG" "$(group_split_for "$CFG")"
done

log "external results done"
