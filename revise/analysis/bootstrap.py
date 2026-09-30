"""Question-level bootstrap confidence intervals for held-out probe AUROC.

Uses the saved per-layer probe weights (coef/sd) so no refitting is needed:
the decision score is X @ coef_scaled up to an additive constant.
"""
from __future__ import annotations

import json
import sys
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

from revise.analysis.features import load_features
from revise.analysis.probes import DELTA_TASKS, STATE_TASKS, _select
from revise.analysis.splits import make_split
from revise.experiment import Run, load_config


def bootstrap_auc(run: Run, regime: str, task: str, mode: str, n_boot: int = 1000, seed: int = 0) -> Dict:
    which = "deltas" if task in DELTA_TASKS else "states"
    defs = DELTA_TASKS if which == "deltas" else STATE_TASKS
    X, meta = load_features(run, regime, which)
    d = run.probes_dir(regime) / task / mode
    m = json.load(open(d / "metrics.json"))
    layer = m["best_layer"]; li = m["layers"].index(layer)
    coef = np.load(d / "coef_scaled.npy")[li]
    idx, y = _select(meta, task, defs)
    assign = make_split(run.questions(), mode, run.cfg["analysis"]["test_frac"], run.cfg["analysis"]["seed"])
    te = np.array([assign.get(meta[i]["qid"], "train") == "test" for i in idx])
    idx, y = idx[te], y[te]
    s = np.asarray(X[idx, layer]).astype(np.float32) @ coef
    qids = np.array([meta[i]["qid"] for i in idx])
    uq = np.unique(qids); by_q = {q: np.where(qids == q)[0] for q in uq}
    rng = np.random.default_rng(seed); aucs = []
    for _ in range(n_boot):
        samp = rng.choice(uq, len(uq), replace=True)
        ii = np.concatenate([by_q[q] for q in samp])
        if len(np.unique(y[ii])) < 2:
            continue
        aucs.append(roc_auc_score(y[ii], s[ii]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return {"task": task, "split": mode, "layer": layer, "auc": float(roc_auc_score(y, s)), "ci95": [float(lo), float(hi)],
            "n_test": int(len(y)), "n_test_questions": int(len(uq)), "pos_rate": float(y.mean())}


if __name__ == "__main__":
    out = {}
    for cfg_path in sys.argv[1:]:
        cfg = load_config(cfg_path); run = Run(cfg); name = cfg["run_name"]; out[name] = {}
        for task in ["sufficiency", "sufficiency_matched", "uptake", "correction", "stability", "state_sufficiency"]:
            for mode in cfg["analysis"]["splits"]:
                if not (run.probes_dir("evidence_only") / task / mode / "metrics.json").exists():
                    continue
                r = bootstrap_auc(run, "evidence_only", task, mode)
                out[name][f"{task}/{mode}"] = r
                print(f"{name:24s} {task:20s} {mode:9s} L{r['layer']:<2d} AUROC {r['auc']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] n={r['n_test']} q={r['n_test_questions']}", flush=True)
        run.save_json("probes/bootstrap_ci.json", out[name])
    json.dump(out, open("/scratch/users/iacopog/revise/artifacts/bootstrap_ci_all.json", "w"), indent=2)
