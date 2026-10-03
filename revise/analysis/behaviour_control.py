"""Is the sufficiency readout just the prompted output decision (answer vs INSUFFICIENT)?

The anchor is the position that emits the first answer token, and the prompt makes
"output INSUFFICIENT" the alternative to answering, so a probe for sufficiency could be
reading the model's *planned output* rather than an evidence-matching computation.
This module holds the output fixed:

  * output-only baseline: AUROC of the indicator "model answers at C_k" (and, for updates,
    "abstained at C_{k-1} and answers at C_k") for the sufficiency label;
  * stratified test: the standard probe (trained on all increments) scored separately on test
    increments where the model abstains at C_k and where it answers at C_k;
  * within-stratum training: probes trained *and* tested only on increments where the model
    abstains at C_k (resp. answers), i.e. "evidence complete but the model still says
    INSUFFICIENT" against controls after which it also says INSUFFICIENT;
  * the same three analyses for the state-sufficiency probe on raw states h(C).

Representations: h(C_k) ("to") and Delta h ("delta"), document split, layer chosen by grouped
inner CV on training questions over every second block.

  python -m revise.analysis.behaviour_control configs/musique_2hop_qwen7b.yaml [...]
"""
from __future__ import annotations

import logging
import sys
from typing import Dict

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from revise.analysis.features import load_features
from revise.analysis.probes import DELTA_TASKS, STATE_TASKS, _fit_lr, _layer_eval, _select, _Std
from revise.analysis.splits import make_split
from revise.experiment import Run, load_config

log = logging.getLogger(__name__)


def _auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 and min(y.sum(), len(y) - y.sum()) >= 5 else None


def _best_layer(F, layers, y, tr, te, groups, C, n_jobs):
    per = Parallel(n_jobs=n_jobs)(delayed(_layer_eval)(F[:, j][tr].astype(np.float32), y[tr], groups[tr],
                                                       F[:, j][te].astype(np.float32), y[te], C, 3) for j in range(len(layers)))
    j = max(range(len(per)), key=lambda k: -1 if per[k]["cv_auc"] is None else per[k]["cv_auc"])
    return j, per[j]


def _scores(F_j, y, tr, te, C):
    std = _Std(F_j[tr].astype(np.float32))
    clf = _fit_lr(std(F_j[tr].astype(np.float32)), y[tr], C)
    return clf.decision_function(std(F_j[te].astype(np.float32)))


def _analyse(name, F, layers, y, answers_to, answers_from, tr, te, groups, C, n_jobs) -> Dict:
    """F: [n, n_layers, d]; answers_to/from: bool arrays (model gives a non-INSUFFICIENT answer)."""
    out = {"n_test": int(len(te)), "pos_rate": float(y.mean())}
    out["output_only_auc"] = _auc(y[te], answers_to[te].astype(float))
    if answers_from is not None:
        out["output_change_auc"] = _auc(y[te], (answers_to[te] & ~answers_from[te]).astype(float))
    j, best = _best_layer(F, layers, y, tr, te, groups, C, n_jobs)
    s = _scores(F[:, j], y, tr, te, C)
    out["layer"] = int(layers[j]); out["all_auc"] = _auc(y[te], s)
    for stratum, mask in (("abstains", ~answers_to), ("answers", answers_to)):
        m_te = mask[te]
        out[f"stratified_{stratum}_auc"] = _auc(y[te][m_te], s[m_te])
        out[f"n_test_{stratum}"] = int(m_te.sum()); out[f"pos_test_{stratum}"] = int(y[te][m_te].sum())
        tr_s, te_s = tr[mask[tr]], te[mask[te]]
        if len(np.unique(y[tr_s])) == 2 and len(np.unique(y[te_s])) == 2 and min(y[tr_s].sum(), len(tr_s) - y[tr_s].sum()) >= 20:
            js, bs = _best_layer(F, layers, y, tr_s, te_s, groups, C, n_jobs)
            out[f"within_{stratum}_auc"] = bs["test_auc"]; out[f"within_{stratum}_layer"] = int(layers[js])
            out[f"n_train_{stratum}"] = int(len(tr_s)); out[f"pos_train_{stratum}"] = int(y[tr_s].sum())
    log.info("%s: %s", name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()})
    return out


def run_control(cfg: dict, regime: str = "evidence_only", mode: str = "document", n_jobs: int = 14) -> dict:
    run = Run(cfg); a = cfg["analysis"]; C = a["probe_C"]
    X, dmeta = load_features(run, regime, "deltas")
    S, smeta = load_features(run, regime, "states")
    k2i = {m["key"]: i for i, m in enumerate(smeta)}
    ib = np.array([k2i[f"{m['qid']}::{m['to_cid']}"] for m in dmeta])
    layers = list(range(2, X.shape[1], 2))
    Xs, Ss = np.asarray(X[:, layers]), np.asarray(S[:, layers])
    assign = make_split(run.questions(), mode, a["test_frac"], a["seed"])
    out = {"run": cfg["run_name"], "mode": mode, "updates": {}, "states": {}}
    for task in ("sufficiency", "sufficiency_matched"):
        idx, y = _select(dmeta, task, DELTA_TASKS)
        sp = np.array([assign.get(dmeta[i]["qid"], "train") for i in idx])
        tr, te = np.where(sp == "train")[0], np.where(sp == "test")[0]
        groups = np.array([dmeta[i]["qid"] for i in idx])
        ans_to = np.array([dmeta[i]["state_to"] != "insufficient" for i in idx])
        ans_from = np.array([dmeta[i]["state_from"] != "insufficient" for i in idx])
        out["updates"][task] = {
            "delta": _analyse(f"{cfg['run_name']} {task} delta", Xs[idx], layers, y, ans_to, ans_from, tr, te, groups, C, n_jobs),
            "to": _analyse(f"{cfg['run_name']} {task} to", Ss[ib[idx]], layers, y, ans_to, ans_from, tr, te, groups, C, n_jobs)}
    idx, y = _select(smeta, "state_sufficiency", STATE_TASKS)
    sp = np.array([assign.get(smeta[i]["qid"], "train") for i in idx])
    tr, te = np.where(sp == "train")[0], np.where(sp == "test")[0]
    groups = np.array([smeta[i]["qid"] for i in idx])
    ans = np.array([not smeta[i]["insufficient"] for i in idx])
    out["states"]["state_sufficiency"] = _analyse(f"{cfg['run_name']} state_sufficiency", Ss[idx], layers, y, ans, None, tr, te, groups, C, n_jobs)
    run.save_json(f"probes/{regime}/behaviour_control_{mode}.json", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for c in sys.argv[1:]:
        run_control(load_config(c))
