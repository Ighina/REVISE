"""Shared paths, run names and formatting helpers for the paper-artefact scripts.

The scripts in this directory rebuild the paper's LaTeX tables and figures from
the JSON artefacts of finished runs.  They read ``$REVISE_SCRATCH/runs`` and
write to ``$REVISE_PAPER_DIR`` (default: ``paper_artifacts/`` next to this
checkout), which gets a ``tables/`` and a ``figures/`` subdirectory.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# allow `python scripts/paper/make_tables.py` from a checkout that is not installed
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from revise import config  # noqa: E402  (must follow the sys.path tweak)

RUNS = config.RUNS_DIR
PAPER_DIR = Path(os.environ.get("REVISE_PAPER_DIR") or config.REPO_ROOT / "paper_artifacts").expanduser()
TABLES = PAPER_DIR / "tables"
FIGURES = PAPER_DIR / "figures"

# The three main-text models, and the further runs used by the appendix tables.
MODELS = [("Qwen2.5-7B", "musique2hop_qwen7b"), ("Llama-3.1-8B", "musique2hop_llama8b"), ("Mistral-7B", "musique2hop_mistral7b")]
EXTRA = [("Qwen 3--4 hop", "musiquefull_qwen7b"), ("Qwen 2Wiki", "twowiki_qwen7b")]
CONFIG_OF = {
    "musique2hop_qwen7b": "configs/musique_2hop_qwen7b.yaml",
    "musique2hop_llama8b": "configs/musique_2hop_llama8b.yaml",
    "musique2hop_mistral7b": "configs/musique_2hop_mistral7b.yaml",
    "musiquefull_qwen7b": "configs/musique_full_qwen7b.yaml",
    "twowiki_qwen7b": "configs/twowiki_qwen7b.yaml",
}


def read_json(run: str, rel: str):
    """Artefact ``rel`` of run ``run``, or None when that stage has not been run."""
    p = RUNS / run / rel
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def write_table(name: str, lines: list[str]) -> Path:
    """Write a LaTeX table body (no float environment) to ``tables/<name>.tex``."""
    TABLES.mkdir(parents=True, exist_ok=True)
    p = TABLES / f"{name}.tex"
    p.write_text("\n".join(lines) + "\n")
    print(f"wrote {p}")
    return p


def figure_path(name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES / f"{name}.pdf"


def f3(x) -> str:
    return "--" if x is None else f"{x:.3f}"


def f2(x) -> str:
    return "--" if x is None else f"{x:.2f}"
