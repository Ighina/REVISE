#!/usr/bin/env bash
# Reuse already-downloaded model snapshots from a shared (read-only) HF cache
# without re-downloading: creates a real model directory in the project cache
# whose blobs/refs/snapshots are symlinks into the shared cache (huggingface_hub
# can then write its own metadata such as trees/ next to them).
# Usage: scripts/link_shared_cache.sh [SHARED_HF_HOME] [MODEL_DIR ...]
set -euo pipefail
SHARED=${1:-/scratch/datasets/.cache/huggingface}
shift || true
DEST=${REVISE_HF_HOME:-${REVISE_SCRATCH:-/scratch/users/iacopog/revise}/hf_cache}/hub
mkdir -p "$DEST"
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(models--Qwen--Qwen2.5-7B-Instruct models--Qwen--Qwen2.5-1.5B-Instruct models--meta-llama--Llama-3.1-8B-Instruct)
fi
for m in "${MODELS[@]}"; do
  if [ ! -d "$SHARED/hub/$m" ]; then echo "skip $m (not in shared cache)"; continue; fi
  if [ -L "$DEST/$m" ]; then rm "$DEST/$m"; fi          # replace an old whole-dir symlink
  if [ -d "$DEST/$m" ] && [ ! -L "$DEST/$m/snapshots" ]; then echo "skip $m (real copy present)"; continue; fi
  mkdir -p "$DEST/$m"
  for sub in blobs refs snapshots; do
    [ -e "$SHARED/hub/$m/$sub" ] && ln -sfn "$SHARED/hub/$m/$sub" "$DEST/$m/$sub"
  done
  echo "linked $m"
done
