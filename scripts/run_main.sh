#!/usr/bin/env bash
# Phase-1 experiment (MuSiQue 2-hop, Qwen2.5-7B-Instruct). Every stage is resumable;
# re-run this script after an interruption and it continues where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate revise
export REVISE_SCRATCH=${REVISE_SCRATCH:-/scratch/users/iacopog/revise}
CFG=${1:-configs/musique_2hop_qwen7b.yaml}
python -m revise.cli screen        -c "$CFG"
python -m revise.cli collect       -c "$CFG"
python -m revise.cli conversational -c "$CFG"
python -m revise.cli features      -c "$CFG"
python -m revise.cli probes        -c "$CFG" --n-jobs 16
python -m revise.cli transfer      -c "$CFG"
python -m revise.cli steer         -c "$CFG"
python -m revise.cli patch         -c "$CFG"
python -m revise.cli adaptive      -c "$CFG"
python -m revise.cli report        -c "$CFG"
