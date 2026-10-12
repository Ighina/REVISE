#!/usr/bin/env bash
# The Evidence State Transition Benchmark (ESTB) appendix: the same transitions
# with the evidence coming from somewhere other than the benchmark's own gold
# and distractor paragraphs.
#
#   *_e5_noise / *_bm25_noise  benchmark gold kept, distractors retrieved from
#                              the wiki-18 corpus by E5 or BM25
#   *_e5_full  / *_bm25_full   gold *and* distractors retrieved, so a question
#                              only enters when retrieval found its gold hops
#   *_variants                 LLM paraphrase and partial variants of the gold
#                              paragraphs, plus independent cross-dataset sources
#
# The retrieval configs retrieve inside their `screen` stage, so the corpus and
# index caches must exist first:
#   BM25: python scripts/build_retrieval_index.py       (hours, CPU, one-off)
#   E5:   python scripts/check_e5_retrieval.py          (downloads + concatenates
#                                                        the FAISS index parts)
# Both caches live under $REVISE_SCRATCH/retrieval and are shared by every run.
#
# Requires: scripts/run_in_domain_results.sh (the MuSiQue directions to transfer).
#
# Usage: scripts/run_estb.sh [config ...]        (default: the Qwen ESTB cells)
source "$(dirname "$0")/lib.sh"

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(configs/musique_2hop_qwen7b_e5_noise.yaml configs/musique_2hop_qwen7b_bm25_noise.yaml
           configs/musique_2hop_qwen7b_e5_full.yaml configs/musique_2hop_qwen7b_bm25_full.yaml
           configs/musique_2hop_qwen7b_variants.yaml)
fi

for CFG in "${CONFIGS[@]}"; do
  log "=== ESTB cell: $CFG"
  stage "$CFG" screen
  case "$CFG" in
    *variants*) stage "$CFG" variants ;;   # generate the paraphrase / partial / independent sources first
  esac
  stage "$CFG" collect
  stage "$CFG" features
  stage "$CFG" probes --n-jobs "$N_JOBS"
  stage "$CFG" transfer --transfer-from "$SRC_CONFIG"
  stage "$CFG" patch
  stage "$CFG" ablate
  stage "$CFG" report
done

log "=== ESTB cell table"
analysis revise.analysis.estb "${CONFIGS[@]}"

log "ESTB done"
