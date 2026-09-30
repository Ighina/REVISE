"""Prompt templates for every instruction regime.

All stateless regimes share the same skeleton so that the *only* difference
between evidence stages is the evidence block, and the activation anchor (the
last prompt token, immediately before the first answer token) is comparable
across contexts of different length.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from revise.data.schema import Paragraph

REGIMES = ("evidence_only", "evidence_augmented", "closed_book", "conversational")

_INSTR = {
    "evidence_only": (
        "Use only the supplied evidence to answer the question. Do not rely on any other knowledge. "
        "If the evidence is insufficient to determine the answer, output exactly INSUFFICIENT. "
        "Respond with only the short answer, no explanation."
    ),
    "evidence_augmented": (
        "Use the supplied evidence together with your own knowledge to answer the question. "
        "If you cannot determine the answer, output exactly INSUFFICIENT. "
        "Respond with only the short answer, no explanation."
    ),
    "closed_book": (
        "Answer the question from your own knowledge. If you do not know the answer, output exactly INSUFFICIENT. "
        "Respond with only the short answer, no explanation."
    ),
}

_CONV_INSTR = (
    "Answer the question again using all of the evidence supplied so far. "
    "Change your answer only if the total evidence warrants it. "
    "If the evidence is insufficient to determine the answer, output exactly INSUFFICIENT. "
    "Respond with only the short answer, no explanation."
)


def render_evidence(paragraphs: Sequence[Paragraph]) -> str:
    if not paragraphs:
        return "(no evidence supplied)"
    return "\n\n".join(f"[{i + 1}] {p.render()}" for i, p in enumerate(paragraphs))


def build_stateless(regime: str, question: str, paragraphs: Sequence[Paragraph]) -> List[dict]:
    """Return chat ``messages`` for a stateless (fresh forward pass) regime."""
    if regime not in _INSTR:
        raise ValueError(regime)
    if regime == "closed_book":
        user = f"{_INSTR[regime]}\n\nQuestion:\n{question}\n\nAnswer:"
    else:
        user = f"{_INSTR[regime]}\n\nEvidence:\n{render_evidence(paragraphs)}\n\nQuestion:\n{question}\n\nAnswer:"
    return [{"role": "user", "content": user}]


def build_conversational(question: str, prev_paragraphs: Sequence[Paragraph], prev_answer: str,
                         new_paragraphs: Sequence[Paragraph], base_regime: str = "evidence_only") -> List[dict]:
    """Two-turn conversational revision prompt (secondary protocol).

    Turn 1 is the stateless prompt for ``C_k`` with the model's own answer
    ``A_k`` inserted as the assistant turn; turn 2 supplies only the new
    evidence ``E_{k+1}`` and asks the model to answer again.
    """
    msgs = build_stateless(base_regime, question, prev_paragraphs)
    msgs.append({"role": "assistant", "content": prev_answer})
    msgs.append({
        "role": "user",
        "content": (
            f"Your previous answer was: {prev_answer}\n\n"
            f"New evidence:\n{render_evidence(new_paragraphs)}\n\n"
            f"{_CONV_INSTR}\n\nQuestion:\n{question}\n\nAnswer:"
        ),
    })
    return msgs


def messages_to_raw(messages: List[dict]) -> str:
    """Fallback rendering when a tokenizer has no chat template."""
    parts = []
    for m in messages:
        tag = {"user": "User", "assistant": "Assistant", "system": "System"}[m["role"]]
        parts.append(f"{tag}: {m['content']}")
    return "\n\n".join(parts) + "\n\nAssistant:"
