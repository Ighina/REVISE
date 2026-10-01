"""Model-facing stages: closed-book screening, stateless collection over all
evidence conditions, and the conversational-revision ablation.

All stages are resumable: rows already present in ``behavior/<regime>.jsonl``
and the activation store are skipped.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from revise.data.loaders import load_questions
from revise.data.schema import Condition, Increment, Question, append_jsonl, read_jsonl, write_jsonl
from revise.data.trajectories import build_all
from revise.eval import answer_state, clean_generation, exact_match, is_insufficient, max_f1, normalize_answer
from revise.experiment import Run
from revise.model.backend import HFBackend
from revise.prompts import build_conversational, build_stateless
from revise.store import ActivationStore

log = logging.getLogger(__name__)


def get_backend(cfg: dict) -> HFBackend:
    m = cfg["model"]
    import os
    num_threads = int(os.environ["REVISE_NUM_THREADS"]) if os.environ.get("REVISE_NUM_THREADS") else m.get("num_threads")
    device = os.environ.get("REVISE_DEVICE") or m.get("device", "auto")
    return HFBackend(m["name"], dtype=m["dtype"], quantize=m.get("quantize"), num_threads=num_threads, device=device,
                     max_new_tokens=m["max_new_tokens"], use_chat_template=m.get("use_chat_template", True),
                     capture_layers=m.get("capture_layers"))


def _answer_in_text(q: Question, text: str) -> bool:
    t = normalize_answer(text)
    return any(normalize_answer(a) and normalize_answer(a) in t for a in q.all_answers)


def _behavior_row(key: str, q: Question, out, i: int, mode: str, extra: dict) -> dict:
    pred = clean_generation(out.generated[i])
    row = {
        "key": key, "qid": q.qid, "generated": out.generated[i], "pred": pred,
        "state": answer_state(pred, q.all_answers, mode), "insufficient": is_insufficient(pred),
        "em": exact_match(pred, q.all_answers), "f1": max_f1(pred, q.all_answers),
        "gold_logprob": out.gold_logprob[i], "gold_first_logprob": out.gold_first_logprob[i],
        "entropy": out.anchor_entropy[i], "top_prob": out.anchor_top_prob[i], "top_token": out.anchor_top_token[i],
        "prompt_tokens": out.prompt_len[i],
    }
    row["correct"] = row["state"] == "correct"
    row.update(extra)
    return row


def _batches(items: List, batch_size: int, length_key) -> List[List]:
    import os
    if os.environ.get("REVISE_BATCH_SIZE"):
        batch_size = int(os.environ["REVISE_BATCH_SIZE"])
    items = sorted(items, key=length_key)
    # Token budget per batch (chars/4 ~ tokens): long full-context prompts get smaller
    # batches so that attention/KV memory stays bounded on GPU.  Default 48k tokens.
    budget = int(os.environ.get("REVISE_TOKEN_BUDGET") or 48000)
    batches, cur = [], []
    for it in items:
        est = max(1, length_key(it) // 4 + 64)
        if cur and (len(cur) >= batch_size or (len(cur) + 1) * est > budget):
            batches.append(cur); cur = []
        cur.append(it)
    if cur:
        batches.append(cur)
    return batches


# ---------------------------------------------------------------------------
# 1. closed-book screening + trajectory construction
# ---------------------------------------------------------------------------

def run_screen(cfg: dict, backend: Optional[HFBackend] = None) -> Run:
    run = Run(cfg)
    d = cfg["data"]
    pool = load_questions(d["dataset"], split=d["split"], hops=d.get("hops"), limit=d.get("pool_limit"),
                          **({"types": d["types"]} if d.get("types") else {}), **(d.get("loader_kwargs") or {}))
    rng = random.Random(d["seed"])
    rng.shuffle(pool)
    log.info("screening pool of %d questions (closed book)", len(pool))
    mode = cfg["eval"]["correctness"]
    done = {r["qid"] for r in read_jsonl(run.screen_path)} if run.screen_path.exists() else set()
    todo = [q for q in pool if q.qid not in done]
    if todo:
        backend = backend or get_backend(cfg)
        for batch in tqdm(_batches(todo, cfg["model"]["batch_size"], lambda q: len(q.question)), desc="screen"):
            msgs = [build_stateless("closed_book", q.question, []) for q in batch]
            out = backend.run_messages(msgs, gold_answers=[q.answer for q in batch], capture=False)
            for i, q in enumerate(batch):
                append_jsonl(run.screen_path, _behavior_row(f"{q.qid}::closed_book", q, out, i, mode, {"regime": "closed_book"}))
    screen = {r["qid"]: r for r in read_jsonl(run.screen_path)}
    if d.get("select", "wrong_closed_book") == "wrong_closed_book":
        selected = [q for q in pool if not screen[q.qid]["correct"]]
    else:
        selected = list(pool)
    selected = selected[: d["n_questions"]]
    log.info("selected %d / %d questions (closed-book accuracy %.3f)", len(selected), len(pool),
             np.mean([screen[q.qid]["correct"] for q in pool]))
    rcfg = d.get("retrieval")
    if rcfg:   # ESTB retrieval-source axis: swap benchmark distractors (and optionally gold) for retrieved passages
        from revise.data.retrieval import apply_retrieved, retrieve_for_questions
        rpath = run.dir / f"retrieved_{rcfg['method']}.jsonl"
        retrieve_for_questions(selected, rcfg["method"], rcfg.get("k", 20), rpath)
        rows = {r["qid"]: r for r in read_jsonl(rpath)}
        swapped = [apply_retrieved(q, rows[q.qid], rcfg.get("mode", "noise"), rcfg.get("n_distractors", 18)) for q in selected if q.qid in rows]
        selected = [q for q in swapped if q is not None]
        log.info("retrieval (%s, mode=%s): %d questions usable", rcfg["method"], rcfg.get("mode", "noise"), len(selected))
    write_jsonl(run.questions_path, (q.to_json() for q in selected))
    conds, incs = build_all(selected, seed=d["seed"], n_random_perms=d["n_random_perms"],
                            include_full=d["include_full"], include_loo=d["include_loo"])
    write_jsonl(run.conditions_path, (c.to_json() for c in conds))
    write_jsonl(run.increments_path, (i.to_json() for i in incs))
    log.info("built %d conditions and %d increments", len(conds), len(incs))
    return run


# ---------------------------------------------------------------------------
# 2. stateless collection
# ---------------------------------------------------------------------------

def run_collect(cfg: dict, regime: str, backend: Optional[HFBackend] = None, max_conditions: Optional[int] = None) -> None:
    run = Run(cfg)
    qs = {q.qid: q for q in run.questions()}
    conds = run.conditions()
    mode = cfg["eval"]["correctness"]
    store = ActivationStore(run.acts_dir(regime))
    beh = run.behavior(regime)
    todo = [c for c in conds if c.key not in beh or c.key not in store]
    if max_conditions:
        todo = todo[:max_conditions]
    log.info("[%s] %d conditions to run (%d already done)", regime, len(todo), len(conds) - len(todo))
    if not todo:
        return
    backend = backend or get_backend(cfg)
    t0 = time.time(); n_tok = 0
    batches = _batches(todo, cfg["model"]["batch_size"], lambda c: sum(len(p.text) for p in c.paragraphs))
    for batch in tqdm(batches, desc=f"collect[{regime}]"):
        msgs = [build_stateless(regime, qs[c.qid].question, c.paragraphs) for c in batch]
        out = backend.run_messages(msgs, gold_answers=[qs[c.qid].answer for c in batch], capture=True)
        keys = [c.key for c in batch]
        store.append(keys, out.anchor_hidden)
        for i, c in enumerate(batch):
            q = qs[c.qid]
            ctx = " ".join(p.title + " " + p.text for p in c.paragraphs)
            if c.key not in beh:
                append_jsonl(run.behavior_path(regime), _behavior_row(c.key, q, out, i, mode, {
                    "regime": regime, "cid": c.cid, "name": c.name, "sufficient": c.sufficient,
                    "n_paragraphs": len(c.paragraphs), "answer_in_context": _answer_in_text(q, ctx),
                    "context_chars": len(ctx),
                }))
        n_tok += int(out.timing["prompt_tokens"])
    store.flush()
    dt = time.time() - t0
    log.info("[%s] done: %d conditions, %.1f s (%.1f prompt tok/s)", regime, len(todo), dt, n_tok / max(dt, 1e-6))


# ---------------------------------------------------------------------------
# 3. conversational revision ablation
# ---------------------------------------------------------------------------

def run_conversational(cfg: dict, backend: Optional[HFBackend] = None, max_increments: Optional[int] = None) -> None:
    run = Run(cfg)
    cc = cfg["conversational"]
    base = cc["base_regime"]
    regime = "conversational"
    qs = {q.qid: q for q in run.questions()}
    conds = {c.key: c for c in run.conditions()}
    incs = [i for i in run.increments() if i.kind in set(cc["kinds"])]
    base_beh = run.behavior(base)
    mode = cfg["eval"]["correctness"]
    store = ActivationStore(run.acts_dir(regime))
    beh = run.behavior(regime)
    todo = []
    for inc in incs:
        key = f"{inc.qid}::{inc.iid}"
        fk = f"{inc.qid}::{inc.from_cid}"
        if fk not in base_beh:
            continue   # need the stateless answer A_k first
        if key in beh and key in store:
            continue
        todo.append(inc)
    if max_increments:
        todo = todo[:max_increments]
    log.info("[conversational] %d increments to run", len(todo))
    if not todo:
        return
    backend = backend or get_backend(cfg)
    batches = _batches(todo, cfg["model"]["batch_size"],
                       lambda i: sum(len(p.text) for p in conds[f"{i.qid}::{i.to_cid}"].paragraphs))
    for batch in tqdm(batches, desc="collect[conversational]"):
        msgs, golds = [], []
        for inc in batch:
            q = qs[inc.qid]
            frm = conds[f"{inc.qid}::{inc.from_cid}"]
            to = conds[f"{inc.qid}::{inc.to_cid}"]
            new = [p for p in to.paragraphs if p.pid in set(inc.added_pids)]
            prev_answer = base_beh[f"{inc.qid}::{inc.from_cid}"]["pred"] or "INSUFFICIENT"
            msgs.append(build_conversational(q.question, frm.paragraphs, prev_answer, new, base_regime=base))
            golds.append(q.answer)
        out = backend.run_messages(msgs, gold_answers=golds, capture=True)
        keys = [f"{inc.qid}::{inc.iid}" for inc in batch]
        store.append(keys, out.anchor_hidden)
        for i, inc in enumerate(batch):
            q = qs[inc.qid]
            to = conds[f"{inc.qid}::{inc.to_cid}"]
            if keys[i] not in beh:
                ctx = " ".join(p.title + " " + p.text for p in to.paragraphs)
                append_jsonl(run.behavior_path(regime), _behavior_row(keys[i], q, out, i, mode, {
                    "regime": regime, "iid": inc.iid, "cid": inc.to_cid, "from_cid": inc.from_cid, "kind": inc.kind,
                    "name": to.name, "sufficient": to.sufficient, "n_paragraphs": len(to.paragraphs),
                    "answer_in_context": _answer_in_text(q, ctx), "context_chars": len(ctx),
                    "prev_pred": base_beh[f"{inc.qid}::{inc.from_cid}"]["pred"],
                }))
    store.flush()
