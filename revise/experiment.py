"""Experiment configuration + run directory layout."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from revise import config
from revise.data.schema import Condition, Increment, Question, read_jsonl

DEFAULTS: Dict[str, Any] = {
    "run_name": "smoke",
    "model": {
        "name": "Qwen/Qwen2.5-7B-Instruct", "dtype": "bfloat16", "quantize": None, "num_threads": None,
        "device": "auto", "batch_size": 8, "max_new_tokens": 16, "capture_layers": None, "use_chat_template": True,
    },
    "data": {
        "dataset": "musique", "split": "train", "hops": [2], "pool_limit": 4000, "n_questions": 2000,
        "select": "wrong_closed_book", "seed": 0, "n_random_perms": 2, "include_full": True, "include_loo": True,
        "types": None,
    },
    "regimes": ["evidence_only", "evidence_augmented"],
    "conversational": {
        "enabled": True, "base_regime": "evidence_only",
        "kinds": ["decisive", "gold_nondecisive", "distractor", "redundant", "false", "answer_ctrl",
                  "post_distractor", "post_redundant", "post_false", "loo_decisive"],
    },
    "eval": {"correctness": "lenient"},
    "analysis": {
        "layers": None, "splits": ["question", "document", "answer", "relation"], "test_frac": 0.3,
        "probe_C": 0.5, "n_passage_svd": 16, "seed": 0,
    },
    "steer": {"regime": "evidence_only", "directions": [{"task": "state_sufficiency"}, {"task": "uptake"}], "layers": None,
              "alphas": [-2, -1, 0, 1, 2, 4], "split": "document", "max_questions": 100, "positions": "anchor_onward",
              "control_random": True},
    "patch": {"regime": "evidence_only", "tasks": ["state_sufficiency", "uptake"], "modes": ["single", "multi"], "layers": None,
              "split": "document", "max_questions": 100},
    "adaptive": {"regime": "evidence_only", "split": "document", "layer": None, "target_precision": 0.97},
    "ablate": {"regime": "evidence_only", "split": "document", "max_questions": 150, "multi_layer": True,
               "experiments": ["projout", "dm_steer", "subspace"], "projout_tasks": ["state_sufficiency", "uptake"],
               "projout_sets": ["sufficient", "false", "insufficient"], "dm_alphas": [-4, -2, 2, 4, 8],
               "ranks": [1, 4, 16, 64, 256], "pca_train": 1500, "positions": "anchor_onward"},
    "text_baseline": {"regime": "evidence_only", "splits": ["question", "document"], "tasks": ["sufficiency", "sufficiency_matched", "uptake"],
                      "encoder": "microsoft/deberta-v3-base", "epochs": 2, "max_len": 512, "lr": 2e-5, "batch_size": 16},
}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    return _merge(DEFAULTS, user)


class Run:
    """Paths inside a run directory."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dir = config.run_dir(cfg["run_name"])
        (self.dir / "behavior").mkdir(exist_ok=True)
        (self.dir / "acts").mkdir(exist_ok=True)
        with open(self.dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    # files
    @property
    def screen_path(self) -> Path: return self.dir / "screen.jsonl"
    @property
    def questions_path(self) -> Path: return self.dir / "questions.jsonl"
    @property
    def conditions_path(self) -> Path: return self.dir / "conditions.jsonl"
    @property
    def increments_path(self) -> Path: return self.dir / "increments.jsonl"
    def behavior_path(self, regime: str) -> Path: return self.dir / "behavior" / f"{regime}.jsonl"
    def acts_dir(self, regime: str) -> Path: return self.dir / "acts" / regime
    def features_dir(self, regime: str) -> Path:
        d = self.dir / "features" / regime; d.mkdir(parents=True, exist_ok=True); return d
    def probes_dir(self, regime: str) -> Path:
        d = self.dir / "probes" / regime; d.mkdir(parents=True, exist_ok=True); return d
    def sub(self, name: str) -> Path:
        d = self.dir / name; d.mkdir(parents=True, exist_ok=True); return d

    # loaders
    def questions(self) -> List[Question]:
        return [Question.from_json(d) for d in read_jsonl(self.questions_path)]

    def conditions(self) -> List[Condition]:
        return [Condition.from_json(d) for d in read_jsonl(self.conditions_path)]

    def increments(self) -> List[Increment]:
        return [Increment.from_json(d) for d in read_jsonl(self.increments_path)]

    def behavior(self, regime: str, judged: bool = True) -> Dict[str, dict]:
        """Behaviour rows keyed by condition key.  When an LLM-judged copy exists
        (``<regime>.judged.jsonl``, see ``run/judge.py``) it takes precedence."""
        p = self.behavior_path(regime)
        pj = p.with_suffix(".judged.jsonl")
        if judged and pj.exists():
            rows = {r["key"]: r for r in read_jsonl(pj)}
            if p.exists():   # rows not yet judged fall back to lenient scoring
                for r in read_jsonl(p):
                    rows.setdefault(r["key"], r)
            return rows
        if not p.exists():
            return {}
        return {r["key"]: r for r in read_jsonl(p)}

    def save_json(self, rel: str, obj: Any) -> Path:
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(obj, f, indent=2, default=float)
        return p
