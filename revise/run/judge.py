"""LLM-judge scoring of model answers against long-form references (RAGBench).

Writes ``behavior/<regime>.judged.jsonl``: a copy of the behaviour rows with
``correct`` / ``state`` replaced by the judge's verdict (abstentions are kept as
``insufficient``).  ``Run.behavior`` prefers the judged file when it exists, so
features, probes and reports pick it up without further changes.
"""
from __future__ import annotations

import logging
import os
from typing import List

from tqdm import tqdm

from revise.data.schema import append_jsonl
from revise.experiment import Run
from revise.run.collect import get_backend

log = logging.getLogger(__name__)

JUDGE_PROMPT = (
    "You are grading a short answer against a reference answer.\n\n"
    "Question: {question}\n\nReference answer: {reference}\n\nCandidate answer: {candidate}\n\n"
    "Does the candidate answer state the same final answer as the reference (ignoring wording, extra detail and explanations)? "
    "Reply with exactly one word: yes or no."
)


def run_judge(cfg: dict, regime: str = "evidence_only", backend=None) -> None:
    run = Run(cfg)
    qs = {q.qid: q for q in run.questions()}
    beh = run.behavior(regime, judged=False)
    out_path = run.behavior_path(regime).with_suffix(".judged.jsonl")
    done = set(r["key"] for r in run.behavior(regime, judged=True)) if out_path.exists() else set()
    todo = [r for k, r in beh.items() if k not in done]
    log.info("[judge] %d rows to grade (%d done)", len(todo), len(done))
    if not todo:
        return
    backend = backend or get_backend(cfg)
    bs = int(os.environ.get("REVISE_BATCH_SIZE") or cfg["model"]["batch_size"])
    old = backend.max_new_tokens; backend.max_new_tokens = 3
    try:
        for b in tqdm(range(0, len(todo), bs), desc="judge"):
            batch = todo[b:b + bs]
            msgs: List[list] = []
            for r in batch:
                q = qs[r["qid"]]
                msgs.append([{"role": "user", "content": JUDGE_PROMPT.format(question=q.question, reference=q.answer[:1500], candidate=r["pred"][:300] or "(no answer)")}])
            out = backend.run_messages(msgs, capture=False)
            for r, g in zip(batch, out.generated):
                verdict = g.strip().lower().startswith("yes")
                row = dict(r)
                if r["insufficient"]:
                    row["correct"], row["state"] = False, "insufficient"
                else:
                    row["correct"], row["state"] = bool(verdict), ("correct" if verdict else "wrong")
                row["judge_raw"], row["correct_lenient"] = g.strip(), r["correct"]
                append_jsonl(out_path, row)
    finally:
        backend.max_new_tokens = old
    log.info("[judge] wrote %s", out_path)
