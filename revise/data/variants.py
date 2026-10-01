"""ESTB linguistic-form axis for benchmark paragraphs: LLM paraphrases and partial variants.

For every gold hop paragraph of a question we generate, with the local model:

  paraphrase   same facts, different wording; must still contain the hop's sub-answer string
  partial      the same paragraph rewritten so that the hop's sub-answer (and its aliases) is
               withheld ("the name is not given here") while subject and relation stay

Results are stored as ``Question.extra["variants"]`` in a ``variants.jsonl`` file
inside the run directory, where ``trajectories.build_trajectories`` picks them up
(``decisive_paraphrase``, ``partial``, ``post_redundant_paraphrase`` increments).
Generation is validated: a paraphrase is kept only if it contains the sub-answer
and shares < 60% of its 4-grams with the original; a partial only if it does NOT
contain the sub-answer (nor the final answer) and mentions the subject.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from revise.data.schema import Paragraph, Question, append_jsonl, read_jsonl
from revise.eval import normalize_answer

log = logging.getLogger(__name__)

PARA_PROMPT = (
    "Rewrite the following encyclopedia paragraph in completely different wording and sentence structure, "
    "keeping every fact exactly the same. You must keep the exact string \"{keep}\" verbatim. "
    "Output only the rewritten paragraph.\n\nParagraph:\n{text}"
)
PARTIAL_PROMPT = (
    "Rewrite the following encyclopedia paragraph so that it still talks about \"{subject}\" and the same topic, "
    "but withholds the specific fact \"{keep}\": do not mention \"{keep}\" or any equivalent name or value anywhere; "
    "instead say that the detail is not recorded here. Keep the rest of the content. Output only the rewritten paragraph.\n\nParagraph:\n{text}"
)


def _ngrams(s: str, n: int = 4) -> set:
    w = normalize_answer(s).split()
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def _contains(text: str, s: str) -> bool:
    ns = normalize_answer(s)
    return bool(ns) and re.search(r"\b" + re.escape(ns) + r"\b", normalize_answer(text)) is not None


def _valid_paraphrase(orig: str, new: str, keep: str) -> bool:
    if not _contains(new, keep) or len(new.split()) < 0.5 * len(orig.split()) or len(new.split()) > 2.0 * len(orig.split()):
        return False
    a, b = _ngrams(orig), _ngrams(new)
    overlap = len(a & b) / max(1, len(b))
    return overlap < 0.6


def _valid_partial(new: str, keep: str, answer: str, aliases: List[str], subject: str) -> bool:
    if any(_contains(new, s) for s in [keep, answer] + list(aliases) if s):
        return False
    return len(new.split()) >= 15 and (_contains(new, subject) or _contains(new, subject.split()[0]))


def generate_variants(questions: List[Question], backend, out_path: Path, max_new_tokens: int = 220, batch_size: int = 16) -> Dict[str, dict]:
    """Resumable: one JSONL row per question {qid, variants: {g0: {paraphrase, partial}, ...}, stats}."""
    done = {r["qid"]: r for r in read_jsonl(out_path)} if out_path.exists() else {}
    todo = [q for q in questions if q.qid not in done]
    log.info("variants: %d questions to do (%d done)", len(todo), len(done))
    jobs = []   # (q, hop, kind, prompt)
    for q in todo:
        for g in q.gold:
            keep = q.decomposition[g.hop]["answer"] if g.hop < len(q.decomposition) else q.answer
            subject = q.decomposition[g.hop]["question"].split(">>")[0].strip() if g.hop < len(q.decomposition) and ">>" in q.decomposition[g.hop]["question"] else g.title
            jobs.append((q, g, "paraphrase", PARA_PROMPT.format(keep=keep, text=g.text), keep, subject))
            jobs.append((q, g, "partial", PARTIAL_PROMPT.format(keep=keep, subject=subject, text=g.text), keep, subject))
    results: Dict[str, Dict[str, dict]] = {}
    old_max = backend.max_new_tokens; backend.max_new_tokens = max_new_tokens
    try:
        for b in range(0, len(jobs), batch_size):
            batch = jobs[b:b + batch_size]
            out = backend.run_messages([[{"role": "user", "content": j[3]}] for j in batch], capture=False)
            for (q, g, kind, _, keep, subject), text in zip(batch, out.generated):
                text = text.strip().split("\n\n")[0].strip() if kind == "partial" else text.strip()
                ok = _valid_paraphrase(g.text, text, keep) if kind == "paraphrase" else _valid_partial(text, keep, q.answer, q.aliases, subject)
                if ok:
                    pid = ("p" if kind == "paraphrase" else "q") + str(g.hop)
                    results.setdefault(q.qid, {}).setdefault(f"g{g.hop}", {})[kind] = Paragraph(
                        pid=pid, title=g.title, text=text, role=kind, hop=g.hop, meta={"generated_by": backend.model_name}).__dict__
            if (b // batch_size) % 20 == 0:
                log.info("variants: %d/%d generations", min(b + batch_size, len(jobs)), len(jobs))
    finally:
        backend.max_new_tokens = old_max
    for q in todo:
        v = results.get(q.qid, {})
        append_jsonl(out_path, {"qid": q.qid, "variants": v, "n_paraphrase": sum("paraphrase" in d for d in v.values()),
                                "n_partial": sum("partial" in d for d in v.values())})
    return {r["qid"]: r for r in read_jsonl(out_path)}


def attach_variants(questions: List[Question], rows: Dict[str, dict]) -> int:
    """Put generated variants into ``q.extra['variants']`` (merging with existing ones)."""
    n = 0
    for q in questions:
        r = rows.get(q.qid)
        if not r or not r["variants"]:
            continue
        ex = dict(q.extra or {}); cur = dict(ex.get("variants", {}))
        for gk, d in r["variants"].items():
            cur.setdefault(gk, {}).update(d)
        ex["variants"] = cur; q.extra = ex; n += 1
    return n
