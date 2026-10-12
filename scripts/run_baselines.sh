#!/usr/bin/env bash
# Table 3 (main text): the baselines that see the same test increments as the
# probe -- TF-IDF and DeBERTa text classifiers, the model's own verbalised
# judgement (it answers rather than saying INSUFFICIENT), and its output
# confidence -- plus the uptake-given-an-answer appendix table.
#
# Requires: scripts/run_in_domain_results.sh (the probes and their test splits).
#
# The text classifiers are trained on the reduced budget used in the main text
# (one epoch, 256 tokens, 8k training increments); the full-budget run for the
# appendix is added for Qwen only, as in the paper.
#
# Usage: scripts/run_baselines.sh
source "$(dirname "$0")/lib.sh"

for MODEL in qwen7b llama8b mistral7b; do
  log "=== text baselines: $MODEL"
  stage "configs/musique_2hop_${MODEL}_textbaseline.yaml" text_baseline
done

log "=== full-budget text baseline (appendix, Qwen only)"
stage configs/musique_2hop_qwen7b_textbaseline_full.yaml text_baseline

log "=== the model's own verbalised judgement and output confidence"
analysis revise.analysis.self_verbal_baseline "${MAIN_CONFIGS[@]}"

log "baselines done"
