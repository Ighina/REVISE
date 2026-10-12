#!/usr/bin/env bash
# Table 4 and Figure 1 (main text): the causal tests.
#
#   patch   the insufficient prompt's anchor residual is patched from the
#           sufficient prompt -- whole residual, the component along or
#           orthogonal to the probe direction, random and PCA subspaces, the
#           gold answer's unembedding direction (Table 4, left)
#   ablate  projection-out on sufficient prompts: the component along the
#           difference-of-means or logistic direction is set to its
#           insufficient-state value (Table 4, right), plus the subspace-rank
#           appendix figure
#   ablate  low-dose difference-of-means steering, four layers from the best
#           sufficiency layer (Figure 1, dose-response)
#   steer   additive steering with the logistic direction, and the large-alpha /
#           all-position controls for Qwen (appendix)
#
# Requires: scripts/run_in_domain_results.sh (the directions being intervened on).
#
# Usage: scripts/run_causal_results.sh
source "$(dirname "$0")/lib.sh"

# Table 4: the three models plus the 3-4 hop run (which run_transfer_results.sh collects).
CAUSAL_CONFIGS=("${MAIN_CONFIGS[@]}" configs/musique_full_qwen7b.yaml)
for CFG in "${CAUSAL_CONFIGS[@]}"; do
  log "=== patching and projection-out: $CFG"
  stage "$CFG" patch
  stage "$CFG" ablate
done

log "=== Figure 1: low-dose difference-of-means steering"
for MODEL in qwen7b llama8b mistral7b; do
  stage "configs/musique_2hop_${MODEL}_dmsteer_small.yaml" ablate
done

log "=== additive steering with the logistic direction (appendix)"
for CFG in "${MAIN_CONFIGS[@]}"; do
  stage "$CFG" steer
done

log "=== steering controls: large alphas, and steering every position (appendix, Qwen)"
stage configs/musique_2hop_qwen7b_steer_large.yaml steer
stage configs/musique_2hop_qwen7b_steer_allpos.yaml steer

log "causal results done"
