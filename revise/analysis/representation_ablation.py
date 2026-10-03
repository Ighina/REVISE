"""Which representation of an increment carries the readout?

For every increment C_{k-1} -> C_k the update probes use Delta h = h(C_k) - h(C_{k-1}).
This module re-fits the same probes (same increments, labels, splits, C, layer selection by
grouped inner CV on training questions) on four representations of the *same* increments:

  from    h(C_{k-1})              the state before the new evidence
  to      h(C_k)                  the state after it
  delta   h(C_k) - h(C_{k-1})     the paired update (the paper's probes)
  concat  [h(C_{k-1}); h(C_k)]    both states; a linear probe on it can represent delta, to, from

and reports held-out AUROC for each, plus two diagnostics for the sufficiency tasks:
how much of the *to*-state AUROC is the state-sufficiency signal (fraction of negatives whose
C_k is itself sufficient), and the AUROC of the stored *state-sufficiency* probe applied to h(C_k).

  python -m revise.analysis.representation_ablation configs/musique_2hop_qwen7b.yaml [...]
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Dict, List

import numpy as np
from joblib import Parallel, delayed

from revise.analysis.features import load_features
from revise.analysis.probes import DELTA_TASKS, _layer_eval, _select
from revise.analysis.splits import make_split
from revise.experiment import Run, load_config

log = logging.getLogger(__name__)
REPS = ("from", "to", "delta", "concat")
TASKS = ("sufficiency", "sufficiency_matched", "uptake", "correction", "stability", "revision")


def _rep(rep: str, S: np.ndarray, X: np.ndarray, ia: np.ndarray, ib: np.ndarray, rows: np.ndarray) -> np.ndarray:
    if rep == "from":
        return S[ia[rows]]
    if rep == "to":
        return S[ib[rows]]
    if rep == "delta":
        return X[rows]
    return np.concatenate([S[ia[rows]], S[ib[rows]]], axis=1)


def _one_layer(l, S_l, X_l, ia, ib, idx, y, tr, te, groups, C, rep):
    A = _rep(rep, S_l, X_l, ia, ib, idx[tr]).astype(np.float32)
    B = _rep(rep, S_l, X_l, ia, ib, idx[te]).astype(np.float32)
    r = _layer_eval(A, y[tr], groups[tr], B, y[te], C, 3)
    return {"layer": int(l), "cv_auc": r["cv_auc"], "test_auc": r["test_auc"], "dm_test_auc": r["dm_test_auc"]}


def run_ablation(cfg: dict, regime: str = "evidence_only", mode: str = "document", layers=None,
                 tasks=TASKS, n_jobs: int = 28) -> dict:
    run = Run(cfg)
    a = cfg["analysis"]
    X, dmeta = load_features(run, regime, "deltas")
    S, smeta = load_features(run, regime, "states")
    k2i = {m["key"]: i for i, m in enumerate(smeta)}
    ia = np.array([k2i[f"{m['qid']}::{m['from_cid']}"] for m in dmeta])
    ib = np.array([k2i[f"{m['qid']}::{m['to_cid']}"] for m in dmeta])
    L = X.shape[1]
    layers = list(layers or range(1, L))     # all blocks, as the update probes
    log.info("loading %d layers of %s", len(layers), run.cfg["run_name"])
    Xs = np.asarray(X[:, layers]); Ss = np.asarray(S[:, layers])
    assign = make_split(run.questions(), mode, a["test_frac"], a["seed"])
    out: Dict[str, dict] = {"run": cfg["run_name"], "mode": mode, "layers": layers, "tasks": {}}
    for task in tasks:
        idx, y = _select(dmeta, task, DELTA_TASKS)
        sp = np.array([assign.get(dmeta[i]["qid"], "train") for i in idx])
        tr, te = np.where(sp == "train")[0], np.where(sp == "test")[0]
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        groups = np.array([dmeta[i]["qid"] for i in idx])
        res = {"n_train": int(len(tr)), "n_test": int(len(te)), "pos_rate": float(y.mean()), "reps": {}}
        if task.startswith("sufficiency"):
            neg = idx[y == 0]
            res["neg_frac_to_sufficient"] = float(np.mean([dmeta[i]["to_sufficient"] for i in neg]))
        for rep in REPS:
            per = Parallel(n_jobs=n_jobs)(delayed(_one_layer)(l, Ss[:, j], Xs[:, j], ia, ib, idx, y, tr, te, groups, a["probe_C"], rep)
                                          for j, l in enumerate(layers))
            best = max(per, key=lambda r: -1 if r["cv_auc"] is None else r["cv_auc"])
            res["reps"][rep] = {"best_layer": best["layer"], "test_auc": best["test_auc"], "dm_test_auc": best["dm_test_auc"],
                                "by_layer": per}
            log.info("[%s] %-20s %-6s layer %2d  test AUROC %.3f", cfg["run_name"], task, rep, best["layer"], best["test_auc"] or float("nan"))
        out["tasks"][task] = res
    # the stored state-sufficiency probe, applied to h(C_k) of the delta-task test increments
    try:
        d = run.probes_dir(regime) / "state_sufficiency" / mode
        m = json.load(open(d / "metrics.json")); li = m["layers"].index(m["best_layer"])
        w = np.load(d / "coef_scaled.npy")[li]
        for task in ("sufficiency", "sufficiency_matched"):
            idx, y = _select(dmeta, task, DELTA_TASKS)
            te = np.array([assign.get(dmeta[i]["qid"], "train") == "test" for i in idx])
            s = np.asarray(S[ib[idx[te]], m["best_layer"]]).astype(np.float32) @ w
            from sklearn.metrics import roc_auc_score
            out["tasks"][task]["state_probe_on_to_auc"] = float(roc_auc_score(y[te], s))
    except FileNotFoundError:
        pass
    run.save_json(f"probes/{regime}/representation_ablation_{mode}.json", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for c in sys.argv[1:]:
        run_ablation(load_config(c))
