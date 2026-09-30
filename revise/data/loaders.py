"""Dataset loaders producing ``Question`` objects with ordered gold hops.

Supported:
  * MuSiQue (answerable subset) via the Hub mirror ``dgslibisey/MuSiQue``
    (official format: paragraphs w/ is_supporting, question_decomposition).
  * 2WikiMultiHopQA via the parquet files in ``xanhho/2WikiMultihopQA``
    (official format: context, supporting_facts, evidences, type).

Both loaders cache a normalised JSONL copy under ``DATA_DIR`` so that later
stages never touch the Hub again.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from revise import config
from revise.data.schema import Paragraph, Question, read_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# MuSiQue
# ---------------------------------------------------------------------------

_REL_RE = re.compile(r">>\s*(.+)$")


def _musique_relation(sub_q: str) -> str:
    m = _REL_RE.search(sub_q)
    if m:
        return m.group(1).strip().lower()
    return "nl:" + " ".join(sub_q.lower().split()[:3])   # natural-language sub-question


def _musique_to_question(r: dict) -> Optional[Question]:
    pars = {p["idx"]: p for p in r["paragraphs"]}
    decomp = r["question_decomposition"]
    gold: List[Paragraph] = []
    seen = set()
    for hop, d in enumerate(decomp):
        idx = d["paragraph_support_idx"]
        if idx is None or idx not in pars or idx in seen:
            return None   # malformed; skip
        seen.add(idx)
        p = pars[idx]
        gold.append(Paragraph(pid=f"g{hop}", title=p["title"], text=p["paragraph_text"], role="gold", hop=hop))
    distractors = [
        Paragraph(pid=f"d{i}", title=p["title"], text=p["paragraph_text"], role="distractor")
        for i, p in enumerate(sorted((p for p in r["paragraphs"] if not p["is_supporting"]), key=lambda p: p["idx"]))
    ]
    n_hops = len(gold)
    if n_hops < 2:
        return None
    return Question(
        qid=r["id"],
        dataset="musique",
        question=r["question"].strip(),
        answer=r["answer"].strip(),
        aliases=[a for a in (r.get("answer_aliases") or []) if a and a != r["answer"]],
        gold=gold,
        distractors=distractors,
        n_hops=n_hops,
        decomposition=[{"question": d["question"], "answer": d["answer"]} for d in decomp],
        relation_types=[_musique_relation(d["question"]) for d in decomp],
        qtype=r["id"].split("__")[0],
        bridge_entities=[d["answer"] for d in decomp[:-1]],
    )


def load_musique(split: str = "train", limit: Optional[int] = None, hops: Optional[List[int]] = None,
                 force: bool = False) -> List[Question]:
    cache = config.DATA_DIR / f"musique_{split}.jsonl"
    if not cache.exists() or force:
        from datasets import load_dataset
        ds = load_dataset("dgslibisey/MuSiQue", split=split)
        qs = []
        for r in ds:
            if not r.get("answerable", True):
                continue
            q = _musique_to_question(r)
            if q is not None:
                qs.append(q)
        write_jsonl(cache, (q.to_json() for q in qs))
    out = [Question.from_json(d) for d in read_jsonl(cache)]
    if hops:
        out = [q for q in out if q.n_hops in hops]
    if limit:
        out = out[:limit]
    return out


# ---------------------------------------------------------------------------
# 2WikiMultiHopQA
# ---------------------------------------------------------------------------

def _norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def _twowiki_to_question(r: dict) -> Optional[Question]:
    context = json.loads(r["context"]) if isinstance(r["context"], str) else r["context"]
    sup = json.loads(r["supporting_facts"]) if isinstance(r["supporting_facts"], str) else r["supporting_facts"]
    evid = json.loads(r["evidences"]) if isinstance(r["evidences"], str) else r["evidences"]
    ctx = {}
    for title, sents in context:
        ctx[title] = " ".join(s.strip() for s in sents)
    sup_titles: List[str] = []
    for title, _ in sup:
        if title not in sup_titles and title in ctx:
            sup_titles.append(title)
    if len(sup_titles) < 2:
        return None
    # Order gold paragraphs following the evidence chain (subject of each triple).
    ordered: List[str] = []
    for subj, rel, obj in evid:
        ns = _norm_title(subj)
        for t in sup_titles:
            nt = _norm_title(t)
            if t not in ordered and (ns == nt or ns in nt or nt in ns):
                ordered.append(t)
                break
    for t in sup_titles:
        if t not in ordered:
            ordered.append(t)
    gold = [Paragraph(pid=f"g{h}", title=t, text=ctx[t], role="gold", hop=h) for h, t in enumerate(ordered)]
    distractors = [Paragraph(pid=f"d{i}", title=t, text=txt, role="distractor")
                   for i, (t, txt) in enumerate(ctx.items()) if t not in sup_titles]
    rels = [e[1].strip().lower() for e in evid] if evid else []
    decomposition = [{"question": f"{e[0]} >> {e[1]}", "answer": e[2]} for e in evid]
    bridge = [e[2] for e in evid[:-1]] if evid else []
    return Question(
        qid=r["_id"], dataset="2wiki", question=r["question"].strip(), answer=r["answer"].strip(), aliases=[],
        gold=gold, distractors=distractors, n_hops=len(gold), decomposition=decomposition,
        relation_types=rels, qtype=r["type"], bridge_entities=bridge,
    )


def load_twowiki(split: str = "train", limit: Optional[int] = None, hops: Optional[List[int]] = None,
                 types: Optional[List[str]] = None, force: bool = False) -> List[Question]:
    cache = config.DATA_DIR / f"twowiki_{split}.jsonl"
    if not cache.exists() or force:
        import pandas as pd
        df = pd.read_parquet(f"hf://datasets/xanhho/2WikiMultihopQA/{split}.parquet")
        qs = []
        for r in df.to_dict("records"):
            q = _twowiki_to_question(r)
            if q is not None:
                qs.append(q)
        write_jsonl(cache, (q.to_json() for q in qs))
    out = [Question.from_json(d) for d in read_jsonl(cache)]
    if hops:
        out = [q for q in out if q.n_hops in hops]
    if types:
        out = [q for q in out if q.qtype in types]
    if limit:
        out = out[:limit]
    return out


# ---------------------------------------------------------------------------
# HotpotQA (distractor setting: 2 supporting paragraphs + 8 distractors)
# ---------------------------------------------------------------------------

def _hotpot_to_question(r: dict) -> Optional[Question]:
    ctx = r["context"]
    titles, sents = ctx["title"], ctx["sentences"]
    text_by_title = {t: " ".join(s.strip() for s in ss) for t, ss in zip(titles, sents)}
    sf = r["supporting_facts"]
    # order supporting titles by their first supporting sentence index, then by appearance
    first = {}
    for t, sid in zip(sf["title"], sf["sent_id"]):
        if t in text_by_title:
            first[t] = min(first.get(t, 10 ** 6), int(sid))
    sup_titles = sorted(first, key=lambda t: (first[t], titles.index(t)))
    if len(sup_titles) != 2:
        return None
    # for bridge questions the paragraph whose title appears in the question is hop 0
    ql = r["question"].lower()
    if r["type"] == "bridge":
        sup_titles.sort(key=lambda t: 0 if t.lower() in ql else 1)
    gold = [Paragraph(pid=f"g{h}", title=t, text=text_by_title[t], role="gold", hop=h) for h, t in enumerate(sup_titles)]
    distractors = [Paragraph(pid=f"d{i}", title=t, text=text_by_title[t], role="distractor")
                   for i, t in enumerate(titles) if t not in sup_titles]
    # sub-answers: HotpotQA has no decomposition; the bridge entity is the second title for bridge questions
    decomposition = [{"question": f"{sup_titles[0]} >> bridge", "answer": sup_titles[1]}, {"question": f"{sup_titles[1]} >> answer", "answer": r["answer"]}] \
        if r["type"] == "bridge" else [{"question": f"{t} >> comparison", "answer": r["answer"]} for t in sup_titles]
    return Question(
        qid=r["id"], dataset="hotpotqa", question=r["question"].strip(), answer=r["answer"].strip(), aliases=[],
        gold=gold, distractors=distractors, n_hops=2, decomposition=decomposition,
        relation_types=[r["type"], r["type"]], qtype=f"{r['type']}:{r['level']}", bridge_entities=[sup_titles[1]] if r["type"] == "bridge" else [],
        extra={"domain": "wikipedia"},
    )


def load_hotpotqa(split: str = "train", limit: Optional[int] = None, hops: Optional[List[int]] = None,
                  types: Optional[List[str]] = None, force: bool = False) -> List[Question]:
    cache = config.DATA_DIR / f"hotpotqa_{split}.jsonl"
    if not cache.exists() or force:
        from datasets import load_dataset
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, trust_remote_code=False)
        qs = []
        for r in ds:
            q = _hotpot_to_question(r)
            if q is not None:
                qs.append(q)
        write_jsonl(cache, (q.to_json() for q in qs))
    out = [Question.from_json(d) for d in read_jsonl(cache)]
    if types:
        out = [q for q in out if q.qtype.split(":")[0] in types]
    if limit:
        out = out[:limit]
    return out


# ---------------------------------------------------------------------------
# RAGBench (naturally retrieved documents with annotated relevant sentences)
# ---------------------------------------------------------------------------

RAGBENCH_SUBSETS = ["covidqa", "cuad", "delucionqa", "emanual", "expertqa", "finqa", "hagrid", "hotpotqa", "msmarco", "pubmedqa", "tatqa", "techqa"]
RAGBENCH_DOMAIN = {"covidqa": "biomedical", "pubmedqa": "biomedical", "cuad": "legal", "techqa": "technical", "emanual": "technical",
                   "finqa": "finance", "tatqa": "finance", "delucionqa": "customer_support", "msmarco": "general", "hagrid": "general",
                   "expertqa": "general", "hotpotqa": "wikipedia"}


def _ragbench_to_question(r: dict, subset: str) -> Optional[Question]:
    docs = r.get("documents") or []
    rel_keys = set(r.get("all_relevant_sentence_keys") or [])
    doc_sents = r.get("documents_sentences") or []
    # a document is a gold unit if any of its sentences is annotated relevant; sentence keys look like "0a", "1c" (doc index + letter)
    gold_idx = []
    for di, sents in enumerate(doc_sents):
        keys = {s[0] for s in sents if isinstance(s, (list, tuple)) and len(s) == 2}
        if keys & rel_keys:
            gold_idx.append(di)
    if not gold_idx or len(gold_idx) > 3 or len(gold_idx) >= len(docs):
        return None
    answer = (r.get("response") or "").strip()
    if not answer:
        return None
    gold = [Paragraph(pid=f"g{h}", title=f"document {di + 1}", text=docs[di].strip(), role="gold", hop=h) for h, di in enumerate(gold_idx)]
    distractors = [Paragraph(pid=f"d{i}", title=f"document {di + 1}", text=docs[di].strip(), role="distractor")
                   for i, di in enumerate(d for d in range(len(docs)) if d not in gold_idx)]
    return Question(
        qid=f"{subset}:{r['id']}", dataset="ragbench", question=r["question"].strip(), answer=answer, aliases=[],
        gold=gold, distractors=distractors, n_hops=len(gold),
        decomposition=[{"question": f"{subset} >> relevant document {h}", "answer": answer} for h in range(len(gold))],
        relation_types=[subset] * len(gold), qtype=subset, bridge_entities=[],
        extra={"domain": RAGBENCH_DOMAIN.get(subset, "general"), "subset": subset, "long_form_reference": len(answer.split()) > 6},
    )


def load_ragbench(split: str = "train", limit: Optional[int] = None, hops: Optional[List[int]] = None,
                  types: Optional[List[str]] = None, force: bool = False, per_subset: Optional[int] = None) -> List[Question]:
    subsets = types or RAGBENCH_SUBSETS
    out: List[Question] = []
    for subset in subsets:
        cache = config.DATA_DIR / f"ragbench_{subset}_{split}.jsonl"
        if not cache.exists() or force:
            from datasets import load_dataset
            ds = load_dataset("rungalileo/ragbench", subset, split=split)
            qs = [q for q in (_ragbench_to_question(r, subset) for r in ds) if q is not None]
            write_jsonl(cache, (q.to_json() for q in qs))
        qs = [Question.from_json(d) for d in read_jsonl(cache)]
        if hops:
            qs = [q for q in qs if q.n_hops in hops]
        out.extend(qs[:per_subset] if per_subset else qs)
    if limit:
        out = out[:limit]
    return out


# ---------------------------------------------------------------------------
# Synthetic controlled benchmark
# ---------------------------------------------------------------------------

def load_synthetic(split: str = "train", limit: Optional[int] = None, hops: Optional[List[int]] = None,
                   force: bool = False, n_questions: int = 3000, seed: int = 0) -> List[Question]:
    from revise.data.synthetic import generate
    cache = config.DATA_DIR / f"synthetic_{split}_{n_questions}_{seed}.jsonl"
    if not cache.exists() or force:
        qs = generate(n_questions=n_questions, seed=seed if split == "train" else seed + 1000)
        write_jsonl(cache, (q.to_json() for q in qs))
    out = [Question.from_json(d) for d in read_jsonl(cache)]
    if hops:
        out = [q for q in out if q.n_hops in hops]
    if limit:
        out = out[:limit]
    return out


LOADERS = {"musique": load_musique, "2wiki": load_twowiki, "hotpotqa": load_hotpotqa, "ragbench": load_ragbench, "synthetic": load_synthetic}


def load_questions(name: str, **kw) -> List[Question]:
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(LOADERS)}")
    return LOADERS[name](**kw)
