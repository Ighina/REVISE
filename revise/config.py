"""Global paths and environment configuration.

Everything heavy (model weights, datasets, activations, run outputs) lives
under ``REVISE_SCRATCH`` (default ``/scratch/users/iacopog/revise``), never in
the working directory.  This module must be imported before ``transformers``
or ``datasets`` so that ``HF_HOME`` is honoured.
"""
from __future__ import annotations

import os
from pathlib import Path

SCRATCH_ROOT = Path(os.environ.get("REVISE_SCRATCH", "/scratch/users/iacopog/revise"))
HF_CACHE = Path(os.environ.get("REVISE_HF_HOME", SCRATCH_ROOT / "hf_cache"))
DATA_DIR = SCRATCH_ROOT / "data"
RUNS_DIR = SCRATCH_ROOT / "runs"
ARTIFACTS_DIR = SCRATCH_ROOT / "artifacts"

for _d in (SCRATCH_ROOT, HF_CACHE, DATA_DIR, RUNS_DIR, ARTIFACTS_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# Route every Hugging Face cache to scratch (weights, datasets, hub metadata).
# The user's login shell may already export a shared HF_HOME; we override it on
# purpose so that everything downloaded by this project lands on scratch.  Set
# REVISE_KEEP_HF_ENV=1 to respect a pre-existing HF_HOME instead.
if not os.environ.get("REVISE_KEEP_HF_ENV"):
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE / "datasets")
    os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


def run_dir(run_name: str) -> Path:
    d = RUNS_DIR / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d
