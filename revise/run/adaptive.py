"""Downstream use: adaptive-retrieval simulation from stored activations.

Each held-out question is walked along its canonical evidence trajectory
(distractors -> +g_1 -> ... -> minimal sufficient -> +distractor).  A policy
reads the stage's *state* probe score s = v·h (pre-generation) and stops when
it exceeds a threshold calibrated on training questions; we report answer
accuracy at the stopping stage and the number of retrieval rounds, against
behavioural (stop at first non-INSUFFICIENT), entropy, oracle and full-context
baselines.  No new forward passes are needed.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List

import numpy as np

from revise.analysis.features import load_features
from revise.analysis.probes import load_direction
from revise.analysis.splits import make_split
from revise.experiment import Run

log = logging.getLogger(__name__)


def _trajectories(run: Run) -> Dict[str, List[str]]:
    """qid -> ordered condition keys along the canonical chain."""
    incs = run.increments()
    chain: Dict[str, List[str]] = {}
    nxt = {}
    for i in incs:
        if i.order_tag == "canonical" and i.kind in ("gold_nondecisive", "decisive"):
            nxt[(i.qid, i.from_cid)] = i.to_cid
    post = {i.qid: i.to_cid for i in incs if i.kind == "post_distractor"}
    cd = {i.qid: i.to_cid for i in incs if i.kind == "distractor_from_none"}
    for qid in {i.qid for i in incs}:
        seq = []
        if qid in cd:
            seq.append(cd[qid])
        cur = "C0"
        while (qid, cur) in nxt:
            cur = nxt[(qid, cur)]; seq.append(cur)
        if qid in post:
            seq.append(post[qid])
        chain[qid] = [f"{qid}::{c}" for c in seq]
    return chain


def _threshold_at_precision(s: np.ndarray, y: np.ndarray, target: float) -> float:
    order = np.argsort(-s)
    s, y = s[order], y[order]
    tp = np.cumsum(y); k = np.arange(1, len(y) + 1)
    prec = tp / k
    ok = np.where(prec >= target)[0]
    return float(s[ok[-1]]) if len(ok) else float(s[0]) + 1e-6


def run_adaptive(cfg: dict) -> dict:
    run = Run(cfg)
    ac = cfg["adaptive"]
    regime, mode = ac["regime"], ac["split"]
    X, meta = load_features(run, regime, "states")
    key2i = {m["key"]: i for i, m in enumerate(meta)}
    layer, v, _ = load_direction(run, regime, "state_sufficiency", mode, ac.get("layer"))
    assign = make_split(run.questions(), mode, cfg["analysis"]["test_frac"], cfg["analysis"]["seed"])
    chains = _trajectories(run)
    scores = np.asarray(X[:, layer]).astype(np.float32) @ v
    ent = np.array([m["entropy"] for m in meta])

    def stage_rows(qids):
        rows = []
        for q in qids:
            for k, key in enumerate(chains.get(q, [])):
                if key in key2i:
                    rows.append((q, k, key2i[key]))
        return rows

    train_q = [q for q in chains if assign.get(q) == "train"]
    test_q = [q for q in chains if assign.get(q) == "test"]
    tr = stage_rows(train_q)
    y_tr = np.array([meta[i]["sufficient"] for _, _, i in tr], dtype=int)
    thr = _threshold_at_precision(np.array([scores[i] for _, _, i in tr]), y_tr, ac["target_precision"])
    ent_thr = _threshold_at_precision(-np.array([ent[i] for _, _, i in tr]), y_tr, ac["target_precision"])

    policies = {
        "probe": lambda i: scores[i] >= thr,
        "behavioral": lambda i: not meta[i]["insufficient"],
        "entropy": lambda i: -ent[i] >= ent_thr,
        "oracle": lambda i: bool(meta[i]["sufficient"]),
        "full": lambda i: False,
    }
    out = {"layer": layer, "threshold": thr, "n_test": len(test_q), "policies": {}}
    for name, stop in policies.items():
        acc, rounds, toks, stopped_early = [], [], [], []
        for q in test_q:
            keys = [key2i[k] for k in chains[q] if k in key2i]
            if not keys:
                continue
            k_stop = next((k for k, i in enumerate(keys) if stop(i)), len(keys) - 1)
            i = keys[k_stop]
            acc.append(meta[i]["correct"]); rounds.append(k_stop + 1); toks.append(meta[i]["prompt_tokens"])
            stopped_early.append(k_stop < len(keys) - 1)
        out["policies"][name] = {"accuracy": float(np.mean(acc)), "mean_rounds": float(np.mean(rounds)),
                                 "mean_prompt_tokens": float(np.mean(toks)), "n": len(acc)}
        log.info("adaptive %-10s acc=%.3f rounds=%.2f tokens=%.0f", name, out["policies"][name]["accuracy"],
                 out["policies"][name]["mean_rounds"], out["policies"][name]["mean_prompt_tokens"])
    run.save_json("adaptive/results.json", out)
    return out
