"""Text-only baselines for the increment tasks.

Answers the reviewer question "would a text classifier on the evidence do as
well as the activation probe?".  Same increments, same labels, same
question-grouped splits as ``probes.py``.

  tfidf    : TF-IDF (question + added paragraph + context titles) -> logistic regression
  encoder  : fine-tuned cross-encoder (default DeBERTa-v3-base) on
             "question [SEP] added paragraph [SEP] previous context" (GPU recommended)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from revise.analysis.features import load_features
from revise.analysis.probes import DELTA_TASKS, _select
from revise.analysis.splits import make_split
from revise.experiment import Run

log = logging.getLogger(__name__)


def _texts(run: Run, meta: List[dict], idx: np.ndarray, with_context: bool) -> List[str]:
    qs = {q.qid: q for q in run.questions()}
    conds = {c.key: c for c in run.conditions()} if with_context else {}
    out = []
    for i in idx:
        m = meta[i]; q = qs[m["qid"]]
        t = f"{q.question} [SEP] {m.get('added_text', '')}"
        if with_context:
            prev = conds.get(f"{m['qid']}::{m['from_cid']}")
            if prev is not None:
                t += " [SEP] " + " ".join(f"{p.title}: {p.text}" for p in prev.paragraphs)
        out.append(t)
    return out


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def _encoder_scores(train_texts, y_tr, test_texts, cfg) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tb = cfg["text_baseline"]
    device = os.environ.get("REVISE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(tb["encoder"])
    model = AutoModelForSequenceClassification.from_pretrained(tb["encoder"], num_labels=2, dtype=torch.float32).to(device)
    pos_w = float((len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=tb["lr"], weight_decay=0.01)
    enc = tok(train_texts, truncation=True, max_length=tb["max_len"], padding=True, return_tensors="pt")
    ds = list(zip(enc["input_ids"], enc["attention_mask"], torch.tensor(y_tr)))
    dl = DataLoader(ds, batch_size=tb["batch_size"], shuffle=True)
    steps = tb["epochs"] * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    model.train()
    for ep in range(tb["epochs"]):
        tot = 0.0
        for ids, am, yy in dl:
            out = model(input_ids=ids.to(device), attention_mask=am.to(device))
            loss = loss_fn(out.logits, yy.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
            tot += float(loss)
        log.info("  encoder epoch %d loss %.4f", ep + 1, tot / len(dl))
    model.eval(); scores = []
    with torch.no_grad():
        for b in range(0, len(test_texts), 64):
            e = tok(test_texts[b:b + 64], truncation=True, max_length=tb["max_len"], padding=True, return_tensors="pt").to(device)
            lg = model(**e).logits.float()
            scores.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
    del model; torch.cuda.empty_cache() if device == "cuda" else None
    return np.concatenate(scores)


def run_text_baseline(run: Run, cfg: dict) -> dict:
    tb = cfg["text_baseline"]
    regime = tb["regime"]
    X, meta = load_features(run, regime, "deltas")
    qs = run.questions()
    results: Dict[str, dict] = {}
    probe_summary = {}
    ps = run.probes_dir(regime) / "summary.json"
    if ps.exists():
        with open(ps) as f:
            probe_summary = json.load(f)
    for task in tb["tasks"]:
        idx, y = _select(meta, task, DELTA_TASKS)
        qid = np.array([meta[i]["qid"] for i in idx])
        for mode in tb["splits"]:
            assign = make_split(qs, mode, cfg["analysis"]["test_frac"], cfg["analysis"]["seed"])
            is_tr = np.array([assign.get(q, "train") == "train" for q in qid])
            tr, te = np.where(is_tr)[0], np.where(~is_tr)[0]
            if len(te) < 10 or len(np.unique(y[te])) < 2:
                continue
            sub = tb.get("train_subsample")
            if sub and len(tr) > sub:   # keep all positives when they are the minority
                rng = np.random.default_rng(cfg["analysis"]["seed"])
                pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
                n_pos = min(len(pos), sub // 2); n_neg = min(len(neg), sub - n_pos)
                tr = np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]))
            entry = {"task": task, "split": mode, "n_train": int(len(tr)), "n_test": int(len(te))}
            for with_ctx in (False, True):
                tag = "q+added+context" if with_ctx else "q+added"
                tt = _texts(run, meta, idx[tr], with_ctx); ttest = _texts(run, meta, idx[te], with_ctx)
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000).fit(tt)
                clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(vec.transform(tt), y[tr])
                s = clf.decision_function(vec.transform(ttest))
                entry[f"tfidf[{tag}]_auc"] = _auc(y[te], s)
                if tb.get("encoder"):
                    try:
                        s = _encoder_scores(tt, y[tr], ttest, cfg)
                        entry[f"encoder[{tag}]_auc"] = _auc(y[te], s)
                    except Exception as e:  # pragma: no cover
                        log.warning("encoder baseline failed (%s): %s", tag, e)
                        entry[f"encoder[{tag}]_auc"] = None
            pe = probe_summary.get(f"deltas/{task}/{mode}")
            if pe:
                entry["probe_auc"] = pe["best_test_auc"]; entry["probe_layer"] = pe["best_layer"]
            results[f"{task}/{mode}"] = entry
            log.info("text baseline %-20s %-9s %s", task, mode, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in entry.items() if k.endswith("_auc")}))
            run.save_json(tb.get("out", "text_baseline/results.json"), results)
    return results
