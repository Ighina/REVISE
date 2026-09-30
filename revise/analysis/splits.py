"""Train/test splits that are disjoint by question, document, answer entity or
relation type (RESEARCH_PLAN.md, "Training split")."""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Dict, List, Sequence

from revise.data.schema import Question
from revise.eval import normalize_answer

log = logging.getLogger(__name__)

SPLIT_MODES = ("question", "document", "answer", "relation")


class _UF:
    def __init__(self):
        self.p: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def group_ids(questions: Sequence[Question], mode: str) -> Dict[str, str]:
    """Map qid -> group id such that train/test must not share a group."""
    if mode == "question":
        return {q.qid: q.qid for q in questions}
    if mode == "answer":
        return {q.qid: "ans:" + normalize_answer(q.answer) for q in questions}
    if mode == "relation":
        return {q.qid: "rel:" + "|".join(q.relation_types) if q.relation_types else "type:" + q.qtype for q in questions}
    if mode == "document":
        uf = _UF()
        for q in questions:
            titles = ["doc:" + t.strip().lower() for t in q.doc_titles]
            uf.union("q:" + q.qid, titles[0])
            for t in titles[1:]:
                uf.union(titles[0], t)
        return {q.qid: uf.find("q:" + q.qid) for q in questions}
    raise ValueError(mode)


def make_split(questions: Sequence[Question], mode: str, test_frac: float = 0.3, seed: int = 0) -> Dict[str, str]:
    """Return qid -> 'train' | 'test'."""
    groups = group_ids(questions, mode)
    by_group: Dict[str, List[str]] = defaultdict(list)
    for qid, g in groups.items():
        by_group[g].append(qid)
    order = sorted(by_group)
    random.Random(f"{seed}:{mode}").shuffle(order)
    n_test_target = int(round(test_frac * len(questions)))
    assign: Dict[str, str] = {}
    n_test = 0
    for g in order:
        # A giant connected component (multi-hop questions sharing many documents) would
        # swallow the whole test side; keep any group larger than the test target in train.
        if len(by_group[g]) > n_test_target:
            for qid in by_group[g]:
                assign[qid] = "train"
            continue
        if n_test < n_test_target:
            for qid in by_group[g]:
                assign[qid] = "test"
            n_test += len(by_group[g])
        else:
            for qid in by_group[g]:
                assign[qid] = "train"
    if n_test == 0 and len(by_group) > 1:   # every group oversized: give the smallest one to test
        g = min(order, key=lambda g: len(by_group[g]))
        for qid in by_group[g]:
            assign[qid] = "test"
        n_test = len(by_group[g])
    n_tr = sum(v == "train" for v in assign.values())
    if n_tr == 0 or n_test == 0:
        log.warning("split mode %r produced an empty side (train=%d, test=%d, groups=%d)", mode, n_tr, n_test, len(by_group))
    return assign


def check_disjoint(questions: Sequence[Question], assign: Dict[str, str], mode: str) -> bool:
    groups = group_ids(questions, mode)
    seen: Dict[str, str] = {}
    for q in questions:
        g = groups[q.qid]
        if g in seen and seen[g] != assign[q.qid]:
            return False
        seen[g] = assign[q.qid]
    return True
