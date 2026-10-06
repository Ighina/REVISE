"""Self-verbalisation and output-confidence baselines for the increment tasks.

The prompt asks the model to output INSUFFICIENT when the evidence does not determine the
answer, so its own output is a verbalised sufficiency judgement.  For every update task we
score, on exactly the test increments of the probes and text baselines (document split):

  verbalised_answers       the model answers (does not say INSUFFICIENT) at C_k
  verbalised_switch        it said INSUFFICIENT at C_{k-1} and answers at C_k
  confidence_top_prob      top-token probability at the anchor of C_k
  confidence_neg_entropy   negated next-token entropy at the anchor of C_k
  confidence_d_*           change of those two quantities over the increment

  python -m revise.analysis.self_verbal_baseline configs/musique_2hop_qwen7b.yaml [...]
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from revise.analysis.features import load_features
from revise.analysis.probes import DELTA_TASKS, _select
from revise.analysis.splits import make_split
from revise.experiment import Run, load_config

TASKS = ("sufficiency", "sufficiency_matched", "uptake", "correction", "stability", "revision")


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def _uptake_given_answer(r, regime, mode, a, dm, idx_all, assign, sig, ans_to, y):
    """Uptake among the decisive increments the model *answers*: correct vs wrong answer.
    The verbalised signal is constant here, so this is where internal state and confidence
    can be compared.  Scores: the stored uptake probe (trained on all decisive increments),
    a probe refitted on answered increments only (stored best layer), and output confidence."""
    import json
    from revise.analysis.probes import _fit_lr, _Std
    X, _ = load_features(r, regime, "deltas")
    d = r.probes_dir(regime) / "uptake" / mode
    m = json.load(open(d / "metrics.json")); L = m["best_layer"]
    w = np.load(d / "coef_scaled.npy")[m["layers"].index(L)]
    sp = np.array([assign.get(dm[i]["qid"], "train") for i in idx_all])
    yy = np.array([int(dm[i]["state_to"] == "correct") for i in idx_all])
    ans = np.array([dm[i]["state_to"] != "insufficient" for i in idx_all])
    tr, te = (sp == "train") & ans, (sp == "test") & ans
    H = np.asarray(X[idx_all, L]).astype(np.float32)
    s_stored = H @ w
    std = _Std(H[tr]); clf = _fit_lr(std(H[tr]), yy[tr], a["probe_C"]); s_within = clf.decision_function(std(H))
    te_mask = ans_to.astype(bool)           # test increments, answered
    return {"n_test_answered": int(te.sum()), "pos_rate": float(yy[te].mean()),
            "stored_probe_auc": _auc(yy[te], s_stored[te]), "within_answered_probe_auc": _auc(yy[te], s_within[te]),
            "confidence_top_prob": _auc(y[te_mask], sig["confidence_top_prob"][te_mask]),
            "confidence_neg_entropy": _auc(y[te_mask], sig["confidence_neg_entropy"][te_mask]),
            "confidence_d_top_prob": _auc(y[te_mask], sig["confidence_d_top_prob"][te_mask])}


def run_baseline(cfg: dict, regime: str = "evidence_only", mode: str = "document") -> dict:
    r = Run(cfg); a = cfg["analysis"]
    _, dm = load_features(r, regime, "deltas")
    _, sm = load_features(r, regime, "states")
    k2i = {m["key"]: i for i, m in enumerate(sm)}
    assign = make_split(r.questions(), mode, a["test_frac"], a["seed"])
    out = {"run": cfg["run_name"], "mode": mode, "tasks": {}}
    for task in TASKS:
        idx, y = _select(dm, task, DELTA_TASKS)
        te = np.array([assign.get(dm[i]["qid"], "train") == "test" for i in idx])
        idx, y = idx[te], y[te]
        rows = [dm[i] for i in idx]
        to = [sm[k2i[f"{m['qid']}::{m['to_cid']}"]] for m in rows]
        fr = [sm[k2i[f"{m['qid']}::{m['from_cid']}"]] for m in rows]
        ans_to = np.array([m["state_to"] != "insufficient" for m in rows], float)
        ans_from = np.array([m["state_from"] != "insufficient" for m in rows], float)
        sig = {
            "verbalised_answers": ans_to,
            "verbalised_switch": ans_to * (1 - ans_from),
            "confidence_top_prob": np.array([t["top_prob"] for t in to], float),
            "confidence_neg_entropy": -np.array([t["entropy"] for t in to], float),
            "confidence_d_top_prob": np.array([t["top_prob"] - f["top_prob"] for t, f in zip(to, fr)], float),
            "confidence_d_neg_entropy": -np.array([t["entropy"] - f["entropy"] for t, f in zip(to, fr)], float),
        }
        res = {"n_test": int(len(y)), "pos_rate": float(y.mean()),
               "answer_rate_pos": float(ans_to[y == 1].mean()), "answer_rate_neg": float(ans_to[y == 0].mean())}
        res.update({k: _auc(y, s) for k, s in sig.items()})
        if task == "uptake":
            res["given_answer"] = _uptake_given_answer(r, regime, mode, a, dm, idx_all=_select(dm, task, DELTA_TASKS)[0], assign=assign, sig=sig, ans_to=ans_to, y=y)
        out["tasks"][task] = res
        print(cfg["run_name"], task, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
    r.save_json(f"probes/{regime}/self_verbal_{mode}.json", out)
    return out


if __name__ == "__main__":
    for c in sys.argv[1:]:
        run_baseline(load_config(c))
