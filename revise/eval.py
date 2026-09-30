"""Answer normalisation, correctness and answer-equivalence utilities."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List

INSUFFICIENT_TOKEN = "INSUFFICIENT"

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def clean_generation(text: str) -> str:
    """Strip the generated continuation down to a short answer string."""
    t = text.strip()
    # keep only the first line / sentence-ish chunk
    t = t.split("\n")[0].strip()
    for prefix in ("answer:", "final answer:", "the answer is", "answer is"):
        if t.lower().startswith(prefix):
            t = t[len(prefix):].strip()
    t = t.strip().strip('"').strip("'").rstrip(".").strip()
    return t


def is_insufficient(pred: str) -> bool:
    p = pred.strip().upper()
    return p.startswith(INSUFFICIENT_TOKEN) or p.rstrip(".") == INSUFFICIENT_TOKEN or \
        ("INSUFFICIENT" in p and len(p.split()) <= 4)


def exact_match(pred: str, golds: Iterable[str]) -> bool:
    np_ = normalize_answer(pred)
    return any(np_ == normalize_answer(g) for g in golds if g)


def f1_score(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def max_f1(pred: str, golds: Iterable[str]) -> float:
    return max((f1_score(pred, g) for g in golds if g), default=0.0)


def is_correct(pred: str, golds: List[str], mode: str = "lenient") -> bool:
    """``strict`` = exact match on any alias.  ``lenient`` additionally accepts a
    prediction that contains the gold answer and is at most 3x its length."""
    if is_insufficient(pred):
        return False
    if exact_match(pred, golds):
        return True
    if mode == "lenient":
        np_ = normalize_answer(pred)
        for g in golds:
            ng = normalize_answer(g)
            if ng and re.search(r"\b" + re.escape(ng) + r"\b", np_) and len(np_.split()) <= 3 * max(1, len(ng.split())):
                return True
            # long-form reference (e.g. RAGBench responses): accept a short prediction that the
            # reference contains, or a yes/no/number that matches the reference's leading token
            if len(ng.split()) > 6 and np_:
                if len(np_.split()) >= 2 and re.search(r"\b" + re.escape(np_) + r"\b", ng):
                    return True
                head = ng.split()[0]
                if np_.split()[0] in ("yes", "no", "maybe") and head == np_.split()[0]:
                    return True
                nums = re.findall(r"-?\d+(?:\.\d+)?", ng)
                if re.fullmatch(r"-?\d+(?:\.\d+)?%?", np_.replace(",", "")) and np_.rstrip("%").replace(",", "") in nums:
                    return True
    return False


def answers_equivalent(a: str, b: str, f1_threshold: float = 0.8) -> bool:
    """Semantic-ish invariance test between two model answers."""
    ia, ib = is_insufficient(a), is_insufficient(b)
    if ia or ib:
        return ia == ib
    na, nb = normalize_answer(a), normalize_answer(b)
    if na == nb:
        return True
    return f1_score(a, b) >= f1_threshold


def answer_state(pred: str, golds: List[str], mode: str = "lenient") -> str:
    """One of: correct | insufficient | wrong."""
    if is_insufficient(pred):
        return "insufficient"
    return "correct" if is_correct(pred, golds, mode) else "wrong"
