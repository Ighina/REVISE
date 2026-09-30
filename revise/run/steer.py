"""Causal validation 1: additive steering along a learned direction.

For each direction (default: the *state* sufficiency probe and the *update*
uptake probe) we add ``alpha * unit * v`` at layer ``l`` (anchor position
onward) where ``unit`` is the separation between two state groups along
``v`` measured on training questions:

  state_sufficiency : mean(v.h | sufficient) - mean(v.h | insufficient)
  other tasks       : mean(v.h | sufficient & correct) - mean(v.h | sufficient & not correct)

so alpha = 1 moves an activation by one "sufficient minus insufficient" gap.
We then measure, on held-out questions:

  sufficient      -> correct-uptake rate            (should go up with +alpha)
  insufficient    -> hallucinated-answer rate       (should NOT go up)
  false           -> false-answer following rate    (should NOT go up)
  answer_ctrl     -> copying the injected answer    (should NOT go up)
  distractor_only -> hallucinated-answer rate       (should NOT go up)

A random direction with the same L2 magnitude is run as a control.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from revise.analysis.features import load_features
from revise.analysis.probes import load_direction
from revise.analysis.splits import make_split
from revise.data.schema import Condition
from revise.eval import clean_generation, is_correct, is_insufficient, normalize_answer
from revise.experiment import Run
from revise.model.backend import HFBackend, Intervention
from revise.prompts import build_stateless
from revise.run.collect import get_backend

log = logging.getLogger(__name__)

EVAL_SETS = {
    "sufficient": lambda c: c.name == "sufficient_min" and c.order_tag == "canonical",
    "insufficient": lambda c: c.name in ("gold_single", "chain_prefix", "loo") and c.order_tag == "canonical",
    "false": lambda c: c.name == "penultimate_plus_false" and c.order_tag == "canonical",
    "answer_ctrl": lambda c: c.name == "penultimate_plus_answer_ctrl" and c.order_tag == "canonical",
    "distractor_only": lambda c: c.name == "distractors_only",
}


def _pick_eval_conditions(run: Run, split_mode: str, cfg: dict, max_questions: int) -> Dict[str, List[Condition]]:
    qs = run.questions()
    assign = make_split(qs, split_mode, cfg["analysis"]["test_frac"], cfg["analysis"]["seed"])
    test_q = [q.qid for q in qs if assign[q.qid] == "test"][:max_questions]
    keep = set(test_q)
    sets: Dict[str, List[Condition]] = defaultdict(list)
    seen = set()
    for c in run.conditions():
        if c.qid not in keep:
            continue
        for name, sel in EVAL_SETS.items():
            if sel(c) and (name, c.qid) not in seen:
                sets[name].append(c); seen.add((name, c.qid))
                break
    return sets


def state_gap(run: Run, regime: str, task: str, split_mode: str, layer: int, v: np.ndarray, cfg: dict) -> dict:
    """Separation along ``v`` between two state groups on training questions."""
    X, meta = load_features(run, regime, "states")
    assign = make_split(run.questions(), split_mode, cfg["analysis"]["test_frac"], cfg["analysis"]["seed"])
    tr = np.array([assign.get(m["qid"], "train") == "train" for m in meta])
    s = np.asarray(X[:, layer]).astype(np.float32) @ v
    suff = np.array([m["sufficient"] for m in meta]); fal = np.array([m["has_false"] for m in meta])
    corr = np.array([m["correct"] for m in meta])
    if task.startswith("state_sufficiency") or task.startswith("sufficiency"):
        pos, neg = tr & suff & ~fal, tr & ~suff & ~fal
    else:
        pos, neg = tr & suff & corr, tr & suff & ~corr
    gap = float(s[pos].mean() - s[neg].mean()) if pos.any() and neg.any() else 0.0
    norm = float(np.linalg.norm(np.asarray(X[tr][:, layer]).astype(np.float32), axis=1).mean()) if tr.any() else 0.0
    return {"gap": gap, "pos_mean": float(s[pos].mean()), "neg_mean": float(s[neg].mean()), "residual_norm": norm}


def _score(rows: List[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "correct": float(np.mean([r["correct"] for r in rows])) if n else None,
        "insufficient": float(np.mean([r["insufficient"] for r in rows])) if n else None,
        "answered": float(np.mean([not r["insufficient"] for r in rows])) if n else None,
        "false_follow": float(np.mean([r["false_follow"] for r in rows if r["false_follow"] is not None]))
        if any(r["false_follow"] is not None for r in rows) else None,
    }


def run_steer(cfg: dict, backend: Optional[HFBackend] = None) -> dict:
    run = Run(cfg)
    sc = cfg["steer"]
    regime, mode = sc["regime"], sc["split"]
    dir_specs = sc.get("directions") or [{"task": sc.get("task", "state_sufficiency")}]
    qs = {q.qid: q for q in run.questions()}
    sets = _pick_eval_conditions(run, mode, cfg, sc["max_questions"])
    backend = backend or get_backend(cfg)
    rng = np.random.default_rng(cfg["analysis"]["seed"])
    results = {"config": sc, "units": {}, "runs": []}
    import os
    bs = int(os.environ.get("REVISE_BATCH_SIZE") or cfg["model"]["batch_size"])
    for spec in dir_specs:
        task = spec["task"]
        for layer in (spec.get("layers") or sc.get("layers") or [None]):
            layer, v, proj_std = load_direction(run, regime, task, mode, layer)
            g = state_gap(run, regime, task, mode, layer, v, cfg)
            unit = abs(g["gap"]) if abs(g["gap"]) > 1e-4 else proj_std
            sign = 1.0 if g["gap"] >= 0 else -1.0   # +alpha always points towards the "sufficient/correct" side
            results["units"][f"{task}/L{layer}"] = {**g, "unit": unit, "proj_std": proj_std}
            log.info("steer %s L%d: unit=%.3f (gap %.3f, residual norm %.1f)", task, layer, unit, g["gap"], g["residual_norm"])
            directions = {"probe": sign * v}
            if sc.get("control_random", True):
                r = rng.standard_normal(v.shape).astype(np.float32); r /= np.linalg.norm(r)
                directions["random"] = r
            for dname, d in directions.items():
                for alpha in sc["alphas"]:
                    vec = torch.tensor(d * float(alpha) * unit)
                    iv = [Intervention("steer", layer, vec, positions=sc.get("positions", "anchor_onward"))] if alpha != 0 else []
                    for set_name, conds in sets.items():
                        rows = []
                        for i in tqdm(range(0, len(conds), bs), desc=f"steer {task} L{layer} {dname} a={alpha} {set_name}", leave=False):
                            batch = conds[i:i + bs]
                            msgs = [build_stateless(regime, qs[c.qid].question, c.paragraphs) for c in batch]
                            out = backend.run_messages(msgs, interventions=iv, capture=False)
                            for j, c in enumerate(batch):
                                q = qs[c.qid]
                                pred = clean_generation(out.generated[j])
                                fa = next((p.meta.get("false_answer") for p in c.paragraphs if p.role == "false"), None)
                                rows.append({
                                    "qid": c.qid, "cid": c.cid, "pred": pred,
                                    "correct": is_correct(pred, q.all_answers, cfg["eval"]["correctness"]),
                                    "insufficient": is_insufficient(pred),
                                    "false_follow": (normalize_answer(fa) in normalize_answer(pred)) if fa else None,
                                })
                        res = {"task": task, "layer": layer, "direction": dname, "alpha": alpha, "l2": float(abs(alpha) * unit),
                               "set": set_name, **_score(rows)}
                        results["runs"].append(res)
                        log.info("steer %s L%d %-6s a=%+.2f %-15s correct=%.2f answered=%.2f false=%s", task, layer, dname, alpha,
                                 set_name, res["correct"] or 0, res["answered"] or 0, res["false_follow"])
                    run.save_json(sc.get("out", "steer/results.json"), results)
    run.save_json(sc.get("out", "steer/results.json"), results)
    return results
