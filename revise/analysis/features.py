"""Turn collected behaviour + activations into probe-ready datasets.

Outputs under ``features/<regime>/``:
  deltas.npy        [n_inc, n_layers+1, d]  paired update  Δh = h(C_k) - h(C_{k-1})
  deltas_meta.jsonl one row per increment with labels + nuisance covariates
  states.npy        [n_cond, n_layers+1, d] raw anchor activation per condition
  states_meta.jsonl one row per condition
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from revise.data.schema import read_jsonl, write_jsonl
from revise.eval import answers_equivalent, normalize_answer
from revise.experiment import Run
from revise.store import ActivationStore

log = logging.getLogger(__name__)

DECISIVE_KINDS = ("decisive", "loo_decisive", "decisive_paraphrase", "decisive_independent")


def _relevance(question: str, texts: List[str]) -> List[float]:
    if not texts:
        return []
    vec = TfidfVectorizer(stop_words="english").fit([question] + texts)
    q = vec.transform([question]); t = vec.transform(texts)
    return cosine_similarity(q, t)[0].tolist()


def build_features(run: Run, regime: str, cfg: dict) -> None:
    layers = None   # always keep every layer so that layer ids stay absolute across all artifacts
    qs = {q.qid: q for q in run.questions()}
    conds = {c.key: c for c in run.conditions()}
    incs = run.increments()
    conv = regime == "conversational"
    base_regime = cfg["conversational"]["base_regime"] if conv else regime
    beh_to = run.behavior(regime)
    beh_from = run.behavior(base_regime)
    closed_book = {r["qid"]: bool(r["correct"]) for r in read_jsonl(run.screen_path)} if run.screen_path.exists() else {}
    store_to = ActivationStore(run.acts_dir(regime))
    store_from = ActivationStore(run.acts_dir(base_regime))
    out_dir = run.features_dir(regime)

    # ---------------- states (stateless regimes only) -----------------------
    if not conv:
        keys = [k for k in conds if k in beh_to and k in store_to]
        X = store_to.get_many(keys, layers)
        np.save(out_dir / "states.npy", X)
        rows = []
        for k in keys:
            c, b, q = conds[k], beh_to[k], qs[conds[k].qid]
            rows.append({
                "key": k, "qid": c.qid, "cid": c.cid, "name": c.name, "sufficient": c.sufficient,
                "n_gold": c.n_gold, "n_distractor": c.n_distractor, "has_false": c.has_false,
                "has_answer_ctrl": c.has_answer_ctrl, "state": b["state"], "pred": b["pred"], "correct": b["correct"],
                "insufficient": b["insufficient"], "prompt_tokens": b["prompt_tokens"], "entropy": b["entropy"],
                "top_prob": b["top_prob"], "gold_logprob": b["gold_logprob"], "answer_in_context": b["answer_in_context"],
                "n_paragraphs": len(c.paragraphs), "n_hops": q.n_hops, "order_tag": c.order_tag,
            })
        write_jsonl(out_dir / "states_meta.jsonl", rows)
        log.info("[%s] states: %d conditions", regime, len(keys))

    # ---------------- deltas ---------------------------------------------------
    rows, mats = [], []
    for inc in incs:
        fk = f"{inc.qid}::{inc.from_cid}"
        tk = f"{inc.qid}::{inc.iid}" if conv else f"{inc.qid}::{inc.to_cid}"
        if fk not in beh_from or fk not in store_from or tk not in beh_to or tk not in store_to:
            continue
        q = qs[inc.qid]
        c_from, c_to = conds[fk], conds[f"{inc.qid}::{inc.to_cid}"]
        b_from, b_to = beh_from[fk], beh_to[tk]
        added = [p for p in c_to.paragraphs if p.pid in set(inc.added_pids)]
        added_text = " ".join(p.title + " " + p.text for p in added)
        rel = _relevance(q.question, [p.title + " " + p.text for p in added])
        ans_norm = [normalize_answer(a) for a in q.all_answers if normalize_answer(a)]
        in_added = any(a in normalize_answer(added_text) for a in ans_norm)
        h_from = store_from.get(fk, layers).astype(np.float32)
        h_to = store_to.get(tk, layers).astype(np.float32)
        mats.append((h_to - h_from).astype(np.float16))
        rows.append({
            "iid": inc.iid, "qid": inc.qid, "from_cid": inc.from_cid, "to_cid": inc.to_cid, "kind": inc.kind,
            "control_group": inc.control_group, "order_tag": inc.order_tag,
            "becomes_sufficient": inc.becomes_sufficient, "from_sufficient": inc.from_sufficient,
            "to_sufficient": inc.to_sufficient, "is_decisive": inc.kind in DECISIVE_KINDS,
            "state_from": b_from["state"], "state_to": b_to["state"], "pred_from": b_from["pred"], "pred_to": b_to["pred"],
            "revision": not answers_equivalent(b_from["pred"], b_to["pred"]),
            "correction": b_from["state"] != "correct" and b_to["state"] == "correct",
            # nuisance covariates
            "d_tokens": b_to["prompt_tokens"] - b_from["prompt_tokens"], "to_tokens": b_to["prompt_tokens"],
            "d_gold_lp": (b_to["gold_logprob"] or 0.0) - (b_from["gold_logprob"] or 0.0),
            "d_entropy": b_to["entropy"] - b_from["entropy"], "d_top_prob": b_to["top_prob"] - b_from["top_prob"],
            "answer_in_added": in_added, "answer_in_to": b_to["answer_in_context"], "answer_in_from": b_from["answer_in_context"],
            "relevance": max(rel) if rel else 0.0, "n_added": len(added), "added_wc": len(added_text.split()),
            "inserted_pos_rel": (c_to.inserted_position or 0) / max(1, len(c_to.paragraphs) - 1),
            "added_text": added_text[:2000], "n_hops": q.n_hops,
            "false_answer": next((p.meta.get("false_answer") for p in added if p.role == "false"), None),
            "closed_book_correct": closed_book.get(inc.qid), "domain": (q.extra or {}).get("domain"),
        })
    if mats:
        np.save(out_dir / "deltas.npy", np.stack(mats, axis=0))
    else:
        log.warning("[%s] no increments with both endpoints collected", regime)
    write_jsonl(out_dir / "deltas_meta.jsonl", rows)
    log.info("[%s] deltas: %d increments", regime, len(rows))


def load_features(run: Run, regime: str, which: str = "deltas"):
    d = run.features_dir(regime)
    X = np.load(d / f"{which}.npy", mmap_mode="r")
    meta = list(read_jsonl(d / f"{which}_meta.jsonl"))
    return X, meta
