"""Evidence State Transition Benchmark (ESTB): cell-wise evaluation across runs.

The benchmark is the union of the per-axis runs (benchmark trajectories, LLM /
independent-source variants, BM25 / E5 retrieved evidence, RAGBench domains,
HotpotQA question types, synthetic controlled trajectories).  Every run already
stores paired-update features with increment kinds (quality states) and
question metadata (domain, hops, retrieval source); this module scores each
*cell* with (i) the MuSiQue-2-hop directions applied without retraining and
(ii) the run's own held-out probe, for the sufficiency and uptake tasks.

Cell keys: run, quality state (increment kind), and one grouping axis
(domain | n_hops | qtype | retrieval label of the added paragraph).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

from revise.analysis.features import DECISIVE_KINDS, load_features
from revise.analysis.probes import DELTA_TASKS, _select
from revise.analysis.splits import make_split
from revise.experiment import Run, load_config

QUALITY = {  # increment kind -> ESTB quality state
    "distractor_from_none": "C1 irrelevant", "distractor": "C1 irrelevant", "post_distractor": "C4 complete+irrelevant",
    "gold_nondecisive": "C2 partial (hop)", "partial": "C2 partial (masked)", "loo_decisive": "C3 complete",
    "decisive": "C3 complete", "decisive_paraphrase": "C3 complete (paraphrase)", "decisive_independent": "C3 complete (independent)",
    "redundant": "C2+redundant", "post_redundant": "C4 complete+redundant", "post_redundant_paraphrase": "C4 complete+paraphrase",
    "post_redundant_independent": "C4 complete+independent", "false": "C5 contradictory (synthetic)", "false_variant": "C5 contradictory",
    "post_false": "C5 complete+contradictory", "answer_ctrl": "C6 answer string, no relation", "full_context": "C4 full context",
}


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 and len(y) >= 20 else None


def cell_table(run: Run, src: Run, regime: str = "evidence_only", mode: str = "question") -> List[dict]:
    X, meta = load_features(run, regime, "deltas")
    qs = {q.qid: q for q in run.questions()}
    assign = make_split(run.questions(), mode, run.cfg["analysis"]["test_frac"], run.cfg["analysis"]["seed"])
    rows = []
    for task in ("sufficiency", "uptake"):
        d_src = src.probes_dir(regime) / task / "document"
        if not (d_src / "metrics.json").exists():
            continue
        m_src = json.load(open(d_src / "metrics.json")); layer_src = m_src["best_layer"]
        v_src = np.load(d_src / "directions.npy")[m_src["layers"].index(layer_src)]
        d_own = run.probes_dir(regime) / task / mode
        own = None
        if (d_own / "metrics.json").exists():
            m_own = json.load(open(d_own / "metrics.json")); layer_own = m_own["best_layer"]
            own = (layer_own, np.load(d_own / "coef_scaled.npy")[m_own["layers"].index(layer_own)])
        idx, y = _select(meta, task, DELTA_TASKS)
        te = np.array([assign.get(meta[i]["qid"], "train") == "test" for i in idx])
        if v_src.shape[0] != X.shape[2]:
            v_src = None
        s_src = (np.asarray(X[idx, layer_src]).astype(np.float32) @ v_src) if v_src is not None else None
        s_own = (np.asarray(X[idx[te], own[0]]).astype(np.float32) @ own[1]) if own else None
        groups: Dict[str, List[str]] = {
            "quality": [QUALITY.get(meta[i]["kind"], meta[i]["kind"]) for i in idx],
            "domain": [(qs[meta[i]["qid"]].extra or {}).get("domain", "wikipedia") for i in idx],
            "n_hops": [str(qs[meta[i]["qid"]].n_hops) for i in idx],
            "retrieval": [((qs[meta[i]["qid"]].extra or {}).get("retrieval") or {}).get("method", "benchmark") for i in idx],
        }
        rows.append({"run": run.cfg["run_name"], "task": task, "axis": "all", "cell": "all", "n": int(len(idx)), "pos_rate": float(y.mean()),
                     "auc_musique_dir": _auc(y, s_src) if s_src is not None else None, "auc_own_probe": _auc(y[te], s_own) if own else None})
        for axis, labels in groups.items():
            labels = np.array(labels)
            for g in sorted(set(labels)):
                mk = labels == g
                if task == "sufficiency" and axis == "quality":
                    # one-vs-rest: how well is this quality state separated from the decisive state?
                    pos = np.array([meta[i]["kind"] in DECISIVE_KINDS for i in idx])
                    sel = mk | pos; yy = pos[sel].astype(int)
                    rows.append({"run": run.cfg["run_name"], "task": task, "axis": axis, "cell": g, "n": int(mk.sum()), "pos_rate": None,
                                 "auc_musique_dir": _auc(yy, s_src[sel]) if s_src is not None and not pos[mk].all() else None,
                                 "auc_own_probe": _auc(yy[te[sel]], s_own[(mk | pos)[te]]) if own and not pos[mk].all() else None})
                else:
                    rows.append({"run": run.cfg["run_name"], "task": task, "axis": axis, "cell": g, "n": int(mk.sum()), "pos_rate": float(y[mk].mean()),
                                 "auc_musique_dir": _auc(y[mk], s_src[mk]) if s_src is not None else None,
                                 "auc_own_probe": _auc(y[mk & te], s_own[mk[te]]) if own else None})
    return rows


if __name__ == "__main__":
    src = Run(load_config("configs/musique_2hop_qwen7b.yaml"))
    all_rows = []
    for cfg_path in sys.argv[1:]:
        cfg = load_config(cfg_path); run = Run(cfg)
        mode = "question" if "document" not in cfg["analysis"]["splits"] or cfg["data"]["dataset"] in ("ragbench", "synthetic") else "document"
        try:
            rows = cell_table(run, src, mode=mode)
        except FileNotFoundError as e:
            print("skip", cfg_path, e); continue
        all_rows += rows
        for r in rows:
            if r["axis"] in ("all", "quality") or r["n"] >= 200:
                print(f"{r['run']:28s} {r['task']:11s} {r['axis']:9s} {str(r['cell']):32s} n={r['n']:6d} musique-dir={('%.3f' % r['auc_musique_dir']) if r['auc_musique_dir'] is not None else '  -  '} own={('%.3f' % r['auc_own_probe']) if r['auc_own_probe'] is not None else '  -  '}")
    json.dump(all_rows, open("/scratch/users/iacopog/revise/artifacts/estb_cells.json", "w"), indent=1)
