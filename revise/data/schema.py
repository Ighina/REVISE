"""Core data structures shared across the pipeline.

Terminology (mirrors RESEARCH_PLAN.md):

* ``Question``   -- a multi-hop question with its ordered gold supporting
                    paragraphs (one per reasoning hop) and matched distractors.
* ``Condition``  -- one evidence context ``C`` shown to the model (a specific
                    ordered list of paragraphs).  ``sufficient`` is True iff
                    every gold hop paragraph is present un-falsified.
* ``Increment``  -- an ordered pair of conditions ``C_{k-1} -> C_k`` whose
                    textual difference is a single paragraph (or a small set),
                    labelled by *what kind* of evidence was added.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional


@dataclass
class Paragraph:
    pid: str                 # unique id within the question, e.g. "g0", "d3", "f1", "a0"
    title: str
    text: str
    role: str                # gold | distractor | false | answer_ctrl | redundant
    hop: Optional[int] = None  # reasoning-hop index for gold / false paragraphs
    meta: dict = field(default_factory=dict)  # e.g. {"false_answer": ..., "true_answer": ...}

    def render(self) -> str:
        return f"Title: {self.title}\n{self.text}"


@dataclass
class Question:
    qid: str
    dataset: str
    question: str
    answer: str
    aliases: List[str]
    gold: List[Paragraph]            # ordered by reasoning hop (canonical order)
    distractors: List[Paragraph]
    n_hops: int
    decomposition: List[dict] = field(default_factory=list)   # [{question, answer}]
    relation_types: List[str] = field(default_factory=list)   # e.g. ["performer", "spouse"]
    qtype: str = ""                                            # dataset-specific type/template
    bridge_entities: List[str] = field(default_factory=list)  # intermediate hop answers
    extra: dict = field(default_factory=dict)

    @property
    def doc_titles(self) -> List[str]:
        return [p.title for p in self.gold]

    @property
    def all_answers(self) -> List[str]:
        return [self.answer] + [a for a in self.aliases if a]

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Question":
        d = dict(d)
        d["gold"] = [Paragraph(**p) for p in d["gold"]]
        d["distractors"] = [Paragraph(**p) for p in d["distractors"]]
        return cls(**d)


@dataclass
class Condition:
    cid: str                     # unique within question, e.g. "C_g0", "C_g0g1", "C_g0+D"
    qid: str
    name: str                    # family name (see trajectories.py)
    paragraphs: List[Paragraph]  # in presentation order
    sufficient: bool
    gold_hops: List[int]         # hop indices present (true, unfalsified gold)
    n_gold: int
    n_distractor: int
    has_false: bool
    has_answer_ctrl: bool
    has_redundant: bool
    order_tag: str = "canonical"  # canonical | reverse | permN | loo | single | none
    inserted_position: Optional[int] = None  # position of newly inserted paragraph (for increments)

    @property
    def key(self) -> str:
        return f"{self.qid}::{self.cid}"

    def to_json(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Condition":
        d = dict(d)
        d["paragraphs"] = [Paragraph(**p) for p in d["paragraphs"]]
        return cls(**d)


@dataclass
class Increment:
    iid: str
    qid: str
    from_cid: str
    to_cid: str
    kind: str          # decisive | gold_nondecisive | distractor | redundant | false |
                       # answer_ctrl | post_distractor | post_redundant | full_context | loo_decisive
    becomes_sufficient: bool
    from_sufficient: bool
    to_sufficient: bool
    added_pids: List[str]
    order_tag: str = "canonical"
    control_group: str = ""   # shared id linking a decisive increment with its matched controls

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Increment":
        return cls(**d)


# --------------------------------------------------------------------------
# JSONL helpers
# --------------------------------------------------------------------------

def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
