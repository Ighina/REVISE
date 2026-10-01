#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 60; done
exec scripts/chain_estb.sh configs/musique_2hop_qwen7b_bm25_noise.yaml configs/musique_2hop_qwen7b_bm25_full.yaml
