"""Held-out probe AUROC broken down by question group (closed-book status, domain, question type).

Uses the stored best-layer probe weights, so no refitting; groups come from
screen.jsonl (closed-book correctness) and Question.extra/qtype.
"""
from __future__ import annotations

import json
import sys
from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score

from revise.analysis.features import load_features
from revise.analysis.probes import DELTA_TASKS, STATE_TASKS, _select
from revise.analysis.splits import make_split
from revise.data.schema import read_jsonl
from revise.experiment import Run, load_config


def group_auc(run: Run, regime: str, task: str, mode: str) -> Dict:
    which = "deltas" if task in DELTA_TASKS else "states"
    defs = DELTA_TASKS if which == "deltas" else STATE_TASKS
    X, meta = load_features(run, regime, which)
    d = run.probes_dir(regime) / task / mode
    m = json.load(open(d / "metrics.json")); layer = m["best_layer"]; li = m["layers"].index(layer)
    coef = np.load(d / "coef_scaled.npy")[li]
    idx, y = _select(meta, task, defs)
    assign = make_split(run.questions(), mode, run.cfg["analysis"]["test_frac"], run.cfg["analysis"]["seed"])
    te = np.array([assign.get(meta[i]["qid"], "train") == "test" for i in idx]); idx, y = idx[te], y[te]
    s = np.asarray(X[idx, layer]).astype(np.float32) @ coef
    qs = {q.qid: q for q in run.questions()}
    cb = {r["qid"]: bool(r["correct"]) for r in read_jsonl(run.screen_path)} if run.screen_path.exists() else {}
    groups = {
        "closed_book": [("correct" if cb.get(meta[i]["qid"]) else "incorrect") for i in idx],
        "domain": [(qs[meta[i]["qid"]].extra or {}).get("domain", "-") for i in idx],
        "qtype": [qs[meta[i]["qid"]].qtype.split(":")[0] for i in idx],
        "n_hops": [str(qs[meta[i]["qid"]].n_hops) for i in idx],
    }
    out = {"task": task, "split": mode, "layer": layer, "overall": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None, "groups": {}}
    for gname, labels in groups.items():
        labels = np.array(labels); res = {}
        for g in sorted(set(labels)):
            mk = labels == g
            res[g] = {"auc": float(roc_auc_score(y[mk], s[mk])) if mk.sum() >= 20 and len(np.unique(y[mk])) == 2 else None, "n": int(mk.sum()), "pos_rate": float(y[mk].mean())}
        if len(res) > 1:
            out["groups"][gname] = res
    return out


if __name__ == "__main__":
    cfg = load_config(sys.argv[1]); run = Run(cfg); regime = "evidence_only"
    mode = sys.argv[2] if len(sys.argv) > 2 else "document"
    res = {}
    for task in ["sufficiency", "sufficiency_matched", "uptake", "correction"]:
        if not (run.probes_dir(regime) / task / mode / "metrics.json").exists():
            continue
        r = group_auc(run, regime, task, mode); res[task] = r
        print(f"{task:20s} overall {r['overall']:.3f}")
        for gname, gr in r["groups"].items():
            print(f"   by {gname}: " + ", ".join(f"{g}={v['auc']:.3f} (n={v['n']})" if v['auc'] is not None else f"{g}=n/a (n={v['n']})" for g, v in gr.items()))
    run.save_json(f"probes/group_auc_{mode}.json", res)
