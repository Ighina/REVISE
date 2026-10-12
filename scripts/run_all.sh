#!/usr/bin/env bash
# Replicate every experiment in the paper, in dependency order, then rebuild the
# tables and figures.  This is weeks of GPU time; in practice run the individual
# scripts (each is resumable and each says which table it produces).
#
# Usage: scripts/run_all.sh
source "$(dirname "$0")/lib.sh"

log "##### 1/8 main-text probe results (Table 2)"
scripts/run_in_domain_results.sh
log "##### 2/8 baselines (Table 3)"
scripts/run_baselines.sh
log "##### 3/8 transfer targets: 2Wiki and MuSiQue 3-4 hop (appendix, and a Table 4 row)"
scripts/run_transfer_results.sh
log "##### 4/8 causal tests (Table 4, Figure 1)"
scripts/run_causal_results.sh
log "##### 5/8 external benchmarks (Table 5)"
scripts/run_external_results.sh
log "##### 6/8 read-out analyses (appendix)"
scripts/run_readout_analyses.sh
log "##### 7/8 adaptive retrieval (appendix)"
scripts/run_adaptive_results.sh
log "##### 8/8 ESTB: retrieved and rewritten evidence (appendix)"
scripts/run_estb.sh
log "##### tables and figures"
scripts/make_paper_artifacts.sh
