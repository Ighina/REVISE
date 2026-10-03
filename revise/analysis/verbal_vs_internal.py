"""Verbalised versus internal sufficiency judgements.

The prompt asks the model to output INSUFFICIENT when the evidence does not determine the
answer, so its output is itself a sufficiency judgement ("answers" = judged sufficient).
On contexts that are sufficient by construction an INSUFFICIENT output is a verbalised false
negative.  This module compares that judgement with the state-sufficiency probe read at the
answer-start anchor of the *same* forward pass, on held-out contexts (document split, contexts
with falsified evidence excluded as in the probe task):

  * verbalised judgement: TPR (sufficient contexts answered), FPR (insufficient contexts answered);
  * probe at a threshold calibrated on *training* contexts to the verbalised FPR measured on
    training contexts (same false-alarm rate), and at 0.95 training precision;
  * the probe's recall on the sufficient contexts the model wrongly called INSUFFICIENT, overall
    and per condition family, and its false-alarm rate on insufficient contexts the model
    (correctly) called INSUFFICIENT.

  python -m revise.analysis.verbal_vs_internal configs/musique_2hop_qwen7b.yaml [...]
"""
from __future__ import annotations

import json
import logging
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from revise.analysis.features import load_features
from revise.analysis.probes import STATE_TASKS, _fit_lr, _select, _Std
from revise.analysis.splits import make_split
from revise.experiment import Run, load_config

log = logging.getLogger(__name__)


def _thr_at_fpr(s_neg: np.ndarray, fpr: float) -> float:
    return float(np.quantile(s_neg, 1 - fpr)) if fpr > 0 else float(s_neg.max()) + 1e-6


def _thr_at_precision(s: np.ndarray, y: np.ndarray, target: float) -> float:
    o = np.argsort(-s); s, y = s[o], y[o]
    prec = np.cumsum(y) / np.arange(1, len(y) + 1)
    ok = np.where(prec >= target)[0]
    return float(s[ok[-1]]) if len(ok) else float(s[0]) + 1e-6


def run_compare(cfg: dict, regime: str = "evidence_only", mode: str = "document") -> dict:
    run = Run(cfg); a = cfg["analysis"]
    S, sm = load_features(run, regime, "states")
    m = json.load(open(run.probes_dir(regime) / "state_sufficiency" / mode / "metrics.json"))
    layer = m["best_layer"]
    idx, y = _select(sm, "state_sufficiency", STATE_TASKS)
    assign = make_split(run.questions(), mode, a["test_frac"], a["seed"])
    sp = np.array([assign.get(sm[i]["qid"], "train") for i in idx])
    tr, te = sp == "train", sp == "test"
    H = np.asarray(S[idx, layer]).astype(np.float32)
    std = _Std(H[tr]); clf = _fit_lr(std(H[tr]), y[tr], a["probe_C"])
    s = clf.decision_function(std(H))
    ans = np.array([not sm[i]["insufficient"] for i in idx])
    names = np.array([sm[i]["name"] for i in idx])
    out = {"run": cfg["run_name"], "layer": layer, "n_test": int(te.sum())}
    # verbalised judgement
    v_tpr_tr, v_fpr_tr = ans[tr & (y == 1)].mean(), ans[tr & (y == 0)].mean()
    out["verbal"] = {"tpr": float(ans[te & (y == 1)].mean()), "fpr": float(ans[te & (y == 0)].mean())}
    out["verbal"]["auc"] = float(roc_auc_score(y[te], ans[te].astype(float)))
    out["probe_auc"] = float(roc_auc_score(y[te], s[te]))
    thrs = {"matched_fpr": _thr_at_fpr(s[tr & (y == 0)], v_fpr_tr), "precision_0.95": _thr_at_precision(s[tr], y[tr], 0.95)}
    wrong_abst = te & (y == 1) & ~ans          # sufficient, model said INSUFFICIENT
    right_abst = te & (y == 0) & ~ans          # insufficient, model said INSUFFICIENT
    out["n_sufficient_abstained"] = int(wrong_abst.sum()); out["n_insufficient_abstained"] = int(right_abst.sum())
    out["n_sufficient"] = int((te & (y == 1)).sum()); out["n_insufficient"] = int((te & (y == 0)).sum())
    for k, t in thrs.items():
        p = s >= t
        r = {"tpr": float(p[te & (y == 1)].mean()), "fpr": float(p[te & (y == 0)].mean()),
             "recall_on_wrong_abstentions": float(p[wrong_abst].mean()),
             "false_alarm_on_correct_abstentions": float(p[right_abst].mean()),
             "precision": float(y[te][p[te]].mean()) if p[te].any() else None, "by_family": {}}
        for fam in sorted(set(names[te & (y == 1)])):
            mk = wrong_abst & (names == fam)
            if mk.sum():
                r["by_family"][fam] = {"n_abstained": int(mk.sum()), "abstain_rate": float((~ans[te & (y == 1) & (names == fam)]).mean()),
                                       "probe_recall": float(p[mk].mean())}
        out[k] = r
    # graded: mean probe score by (sufficient, verbal) cell, in units of the training class-mean gap
    gap = s[tr & (y == 1)].mean() - s[tr & (y == 0)].mean(); base = s[tr & (y == 0)].mean()
    out["mean_score_gap_units"] = {f"{'suff' if yy else 'insuff'}_{'answered' if aa else 'abstained'}":
                                   float(((s[te & (y == yy) & (ans == aa)] - base) / gap).mean())
                                   for yy in (1, 0) for aa in (True, False) if (te & (y == yy) & (ans == aa)).any()}
    # paragraph-count-matched version: every context with >= 3 paragraphs is sufficient by construction,
    # so count / length alone separates part of the task.  Restrict to two-paragraph contexts
    # (minimal sufficient vs distractors only, penultimate + distractor / duplicate / answer-string)
    # and retrain the probe on them only.
    npar = np.array([sm[i]["n_paragraphs"] for i in idx])
    two = npar == 2
    std2 = _Std(H[tr & two]); clf2 = _fit_lr(std2(H[tr & two]), y[tr & two], a["probe_C"])
    s2 = clf2.decision_function(std2(H))
    t2, e2 = tr & two, te & two
    cm = {"n_test": int(e2.sum()), "n_sufficient": int((e2 & (y == 1)).sum()), "n_insufficient": int((e2 & (y == 0)).sum()),
          "insufficient_families": sorted(set(names[e2 & (y == 0)])),
          "verbal_auc": float(roc_auc_score(y[e2], ans[e2].astype(float))),
          "verbal_tpr": float(ans[e2 & (y == 1)].mean()), "verbal_fpr": float(ans[e2 & (y == 0)].mean()),
          "probe_auc": float(roc_auc_score(y[e2], s2[e2])), "probe_all_contexts_auc_on_two": float(roc_auc_score(y[e2], s[e2]))}
    wa2, ra2 = e2 & (y == 1) & ~ans, e2 & (y == 0) & ~ans
    cm["n_sufficient_abstained"] = int(wa2.sum())
    cm["stratified_abstains_auc"] = float(roc_auc_score(y[e2 & ~ans], s2[e2 & ~ans]))
    for k, t in {"matched_fpr": _thr_at_fpr(s2[t2 & (y == 0)], ans[t2 & (y == 0)].mean()),
                 "precision_0.95": _thr_at_precision(s2[t2], y[t2], 0.95)}.items():
        p = s2 >= t
        cm[k] = {"tpr": float(p[e2 & (y == 1)].mean()), "fpr": float(p[e2 & (y == 0)].mean()),
                 "recall_on_wrong_abstentions": float(p[wa2].mean()), "false_alarm_on_correct_abstentions": float(p[ra2].mean())}
    g2 = s2[t2 & (y == 1)].mean() - s2[t2 & (y == 0)].mean(); b2 = s2[t2 & (y == 0)].mean()
    cm["mean_score_gap_units"] = {f"{'suff' if yy else 'insuff'}_{'answered' if aa else 'abstained'}":
                                  float(((s2[e2 & (y == yy) & (ans == aa)] - b2) / g2).mean()) for yy in (1, 0) for aa in (True, False)}
    out["count_matched"] = cm
    log.info("  count-matched (2 paragraphs): %s", json.dumps(cm))
    log.info("%s %s", cfg["run_name"], json.dumps({k: v for k, v in out.items() if k not in ("matched_fpr", "precision_0.95", "count_matched")}))
    log.info("  matched FPR: %s", json.dumps({k: v for k, v in out["matched_fpr"].items() if k != "by_family"}))
    log.info("  by family: %s", json.dumps(out["matched_fpr"]["by_family"]))
    run.save_json(f"probes/{regime}/verbal_vs_internal_{mode}.json", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for c in sys.argv[1:]:
        run_compare(load_config(c))
