#!/usr/bin/env bash
# Rebuild the paper's LaTeX tables and figures from the finished runs.
#
# Writes to $REVISE_PAPER_DIR (default: ./paper_artifacts), which gets
# tables/*.tex, figures/*.pdf and tables/numbers.json -- the consolidated
# dictionary of every number quoted in the text.
#
# A builder whose runs are missing prints a warning and is skipped, so this is
# also the quickest way to see how far a replication has got.
#
# Usage: scripts/make_paper_artifacts.sh
source "$(dirname "$0")/lib.sh"

export REVISE_PAPER_DIR=${REVISE_PAPER_DIR:-$REPO_ROOT/paper_artifacts}

builders=(
  make_tables.py            # Tables 2, 4, 5 and the splits, behaviour and group appendix tables
  make_baselines_table.py   # Table 3 and the uptake-given-an-answer appendix table
  make_state_tables.py      # the read-out appendix tables
  make_figures.py           # Figure 1 and the appendix figures
  make_adaptive_tables.py   # the adaptive-retrieval appendix
  make_estb_table.py        # the ESTB appendix
)
for b in "${builders[@]}"; do
  log "=== $b"
  "$PYTHON" "scripts/paper/$b" || log "  WARNING: $b failed (its runs are probably incomplete)"
done

log "paper artefacts in $REVISE_PAPER_DIR"
