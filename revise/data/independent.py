"""ESTB linguistic-form axis: independently authored evidence.

MuSiQue, HotpotQA and 2WikiMultiHopQA all draw their paragraphs from Wikipedia
but from different dumps, different paragraph boundaries and, often, different
editors' revisions.  For a gold hop paragraph with Wikipedia title T and
sub-answer a, a paragraph from *another* dataset whose title normalises to T and
which contains a is an independently written statement of the same relation.
Such paragraphs are attached as the ``independent`` variant of the hop and give
``decisive_independent`` / ``post_redundant_independent`` increments.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

from revise import config
from revise.data.schema import Paragraph, Question, read_jsonl
from revise.eval import normalize_answer

log = logging.getLogger(__name__)

_CACHE_FILES = {"musique": ["musique_train.jsonl", "musique_validation.jsonl"],
                "hotpotqa": ["hotpotqa_train.jsonl", "hotpotqa_validation.jsonl"],
                "2wiki": ["twowiki_train.jsonl", "twowiki_dev.jsonl"]}


def _norm_title(t: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*$", "", t.strip().lower())


def _contains(text: str, s: str) -> bool:
    ns = normalize_answer(s)
    return bool(ns) and re.search(r"\b" + re.escape(ns) + r"\b", normalize_answer(text)) is not None


def _ngram_overlap(a: str, b: str, n: int = 4) -> float:
    wa, wb = normalize_answer(a).split(), normalize_answer(b).split()
    ga = {tuple(wa[i:i + n]) for i in range(max(0, len(wa) - n + 1))}
    gb = {tuple(wb[i:i + n]) for i in range(max(0, len(wb) - n + 1))}
    return len(ga & gb) / max(1, len(gb))


def build_title_index(exclude_dataset: str) -> Dict[str, List[Tuple[str, str, str]]]:
    """title -> [(dataset, title, text)] over the cached questions of the *other* datasets (gold and distractor paragraphs)."""
    idx: Dict[str, List[Tuple[str, str, str]]] = {}
    for ds, files in _CACHE_FILES.items():
        if ds == exclude_dataset:
            continue
        for fn in files:
            p = config.DATA_DIR / fn
            if not p.exists():
                continue
            for r in read_jsonl(p):
                for par in r["gold"] + r["distractors"]:
                    key = _norm_title(par["title"])
                    bucket = idx.setdefault(key, [])
                    if all(par["text"] != b[2] for b in bucket):
                        bucket.append((ds, par["title"], par["text"]))
    log.info("independent-source index: %d titles from datasets other than %s", len(idx), exclude_dataset)
    return idx


def independent_variants(questions: List[Question]) -> int:
    """Attach an ``independent`` variant for every gold hop that has an independently written
    paragraph (same title, contains the sub-answer, textually different) in another dataset."""
    if not questions:
        return 0
    idx = build_title_index(questions[0].dataset)
    n = 0
    for q in questions:
        ex = dict(q.extra or {}); cur = dict(ex.get("variants", {}))
        for g in q.gold:
            keep = q.decomposition[g.hop]["answer"] if g.hop < len(q.decomposition) else q.answer
            # same title, contains the sub-answer, and textually different enough (< 60% shared 4-grams) to count
            # as independently written rather than another copy of the same Wikipedia revision
            cands = [c for c in idx.get(_norm_title(g.title), []) if _contains(c[2], keep) and _ngram_overlap(g.text, c[2]) < 0.6]
            if not cands:
                continue
            cands.sort(key=lambda c: _ngram_overlap(g.text, c[2]))
            ds, title, text = cands[0]
            cur.setdefault(f"g{g.hop}", {})["independent"] = Paragraph(pid=f"i{g.hop}", title=title, text=text, role="independent", hop=g.hop,
                                                                      meta={"source_dataset": ds}).__dict__
            n += 1
        if cur:
            ex["variants"] = cur; q.extra = ex
    log.info("independent variants attached to %d hops across %d questions", n, len(questions))
    return n
