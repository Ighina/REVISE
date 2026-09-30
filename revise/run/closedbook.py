"""Item 5: does the evidence-state readout transfer to closed-book uncertainty?

Stage ``closedbook``: re-run the closed-book prompt for the screened pool with
activation capture (cheap: ~80-token prompts), then score every stored
direction (state_sufficiency, state_correct, state_insufficient_flag, and the
delta-probe directions) on the closed-book anchor activations for two targets:
closed-book correctness and closed-book abstention (INSUFFICIENT).  A probe
trained directly on closed-book activations (question-disjoint split) gives
the ceiling.
"""
from __future__ import annotations

import json
import logging
import os
import random
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from revise.data.loaders import load_questions
from revise.data.schema import append_jsonl, read_jsonl
from revise.experiment import Run
from revise.prompts import build_stateless
from revise.run.collect import _batches, _behavior_row, get_backend
from revise.store import ActivationStore

log = logging.getLogger(__name__)
REGIME = "closed_book"


def _maxn(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def run_closedbook(cfg: dict, backend=None) -> dict:
    run = Run(cfg)
    d = cfg["data"]
    pool = load_questions(d["dataset"], split=d["split"], hops=d.get("hops"), limit=d.get("pool_limit"),
                          **({"types": d["types"]} if d.get("types") and d["dataset"] == "2wiki" else {}))
    screened = {r["qid"] for r in read_jsonl(run.screen_path)} if run.screen_path.exists() else set()
    pool = [q for q in pool if q.qid in screened] or pool
    store = ActivationStore(run.acts_dir(REGIME))
    beh = run.behavior(REGIME)
    todo = [q for q in pool if f"{q.qid}::closed_book" not in beh or f"{q.qid}::closed_book" not in store]
    mode = cfg["eval"]["correctness"]
    bs = int(os.environ.get("REVISE_BATCH_SIZE") or cfg["model"]["batch_size"])
    if todo:
        backend = backend or get_backend(cfg)
        for batch in tqdm(_batches(todo, bs, lambda q: len(q.question)), desc="closedbook"):
            out = backend.run_messages([build_stateless("closed_book", q.question, []) for q in batch],
                                       gold_answers=[q.answer for q in batch], capture=True)
            keys = [f"{q.qid}::closed_book" for q in batch]
            store.append(keys, out.anchor_hidden)
            for i, q in enumerate(batch):
                if keys[i] not in beh:
                    append_jsonl(run.behavior_path(REGIME), _behavior_row(keys[i], q, out, i, mode, {"regime": REGIME}))
        store.flush()
    beh = run.behavior(REGIME)
    keys = [k for k in beh if k in store]
    X = store.get_many(keys).astype(np.float32)          # [n, nL, d]
    y_corr = np.array([beh[k]["correct"] for k in keys], dtype=int)
    y_abst = np.array([beh[k]["insufficient"] for k in keys], dtype=int)
    qids = np.array([beh[k]["qid"] for k in keys])
    log.info("closed-book pool: %d questions, correct %.3f, abstain %.3f", len(keys), y_corr.mean(), y_abst.mean())
    res: Dict[str, dict] = {"n": len(keys), "correct_rate": float(y_corr.mean()), "abstain_rate": float(y_abst.mean()), "directions": {}}
    regime = cfg["adaptive"]["regime"]; mode_split = cfg["adaptive"]["split"]
    for task in ("state_sufficiency", "state_correct", "state_insufficient_flag", "sufficiency", "uptake", "correction"):
        dpath = run.probes_dir(regime) / task / mode_split
        if not (dpath / "metrics.json").exists():
            continue
        with open(dpath / "metrics.json") as f:
            m = json.load(f)
        D = np.load(dpath / "directions.npy")
        entry = {"layers": m["layers"], "src_best_layer": m["best_layer"], "auc_correct": [], "auc_abstain": []}
        for li, l in enumerate(m["layers"]):
            s = X[:, l] @ D[li]
            entry["auc_correct"].append(float(roc_auc_score(y_corr, s)) if len(np.unique(y_corr)) == 2 else None)
            entry["auc_abstain"].append(float(roc_auc_score(y_abst, s)) if len(np.unique(y_abst)) == 2 else None)
        bi = m["layers"].index(m["best_layer"])
        entry["auc_correct_at_src_best"] = entry["auc_correct"][bi]; entry["auc_abstain_at_src_best"] = entry["auc_abstain"][bi]
        res["directions"][task] = entry
        log.info("closed-book transfer %-24s L%d: AUROC correct=%.3f abstain=%.3f (best any layer: correct=%.3f)", task, m["best_layer"],
                 entry["auc_correct_at_src_best"] or 0, entry["auc_abstain_at_src_best"] or 0,
                 _maxn(entry["auc_correct"]) or 0)
    # ceiling: probe trained on closed-book activations, question-disjoint split
    rng = random.Random(0); uq = sorted(set(qids)); rng.shuffle(uq); test_q = set(uq[: int(0.3 * len(uq))])
    te = np.array([q in test_q for q in qids]); tr = ~te
    ceiling = {"auc_correct": [], "auc_abstain": []}
    for l in range(X.shape[1]):
        mu, sd = X[tr, l].mean(0), X[tr, l].std(0) + 1e-6
        for name, y in (("auc_correct", y_corr), ("auc_abstain", y_abst)):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                ceiling[name].append(None); continue
            clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, tol=1e-3).fit((X[tr, l] - mu) / sd, y[tr])
            ceiling[name].append(float(roc_auc_score(y[te], clf.decision_function((X[te, l] - mu) / sd))))
    res["closed_book_probe_ceiling"] = ceiling
    log.info("closed-book own-probe ceiling: correct max AUROC %.3f, abstain max AUROC %.3f",
             _maxn(ceiling["auc_correct"]) or 0, _maxn(ceiling["auc_abstain"]) or 0)
    run.save_json("closedbook/results.json", res)
    return res
