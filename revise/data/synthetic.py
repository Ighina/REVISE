"""Synthetic controlled evidence benchmark (review point 12, "clean causal trajectory").

Fictional entities (so closed-book failure holds by construction), 1--3-hop
compositional questions over a small relation schema, and for every gold fact
five template families (linguistic form axis), a paraphrase, a *partial*
variant (subject and relation mentioned, value withheld) and a *contradictory*
variant (same subject/relation, different value).  Distractors are near-misses
(same relation, other subject; same subject, other relation) plus unrelated
filler paragraphs, so retrieval-like lexical overlap is present.

Everything is produced as ``Question`` objects with ``extra["variants"]`` so the
standard trajectory builder emits the usual states plus paraphrase/partial
increments (see ``trajectories.py``).
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from revise.data.schema import Paragraph, Question

_SYL = ["ka", "ro", "vi", "len", "mor", "tha", "zu", "bel", "dri", "os", "ne", "quil", "fa", "tor", "ime", "sha", "gur", "el", "pin", "wex"]


def _name(rng: random.Random, n_syl: int = 3, cap: bool = True) -> str:
    s = "".join(rng.choice(_SYL) for _ in range(n_syl))
    return s.capitalize() if cap else s


def _person(rng): return f"{_name(rng, 2)} {_name(rng, 3)}"
def _place(rng): return _name(rng, 3)
def _org(rng): return f"{_name(rng, 2)} {rng.choice(['Industries', 'Collective', 'Institute', 'Works', 'Guild', 'Laboratories'])}"
def _award(rng): return f"the {_name(rng, 2)} {rng.choice(['Prize', 'Medal', 'Award', 'Laurel'])}"
def _year(rng): return str(rng.randint(1802, 2019))

# relation: (subject type, object type, question phrase, template families)
RELATIONS: Dict[str, Tuple[str, str, str, List[str]]] = {
    "founder": ("org", "person", "the founder of {S}", [
        "{S} was founded by {O}.",
        "{O} is credited with establishing {S}.",
        "The organisation known as {S} owes its existence to its founder, {O}.",
        "Historians agree that {O} set up {S}.",
        "{S}: an organisation created by {O}.",
    ]),
    "birthplace": ("person", "place", "the birthplace of {S}", [
        "{S} was born in {O}.",
        "{O} is the birthplace of {S}.",
        "Born in the town of {O}, {S} spent an unremarkable childhood there.",
        "{S}'s birth was registered in {O}.",
        "The native town of {S} is {O}.",
    ]),
    "capital": ("place", "place", "the capital of the region containing {S}", [
        "{S} lies in a region whose capital is {O}.",
        "The regional capital for {S} is {O}.",
        "{S} belongs to the province administered from {O}.",
        "Administratively, {S} answers to the capital {O}.",
        "{O} serves as the capital of the region that includes {S}.",
    ]),
    "spouse": ("person", "person", "the spouse of {S}", [
        "{S} is married to {O}.",
        "{O} and {S} married in a small ceremony.",
        "The spouse of {S} is {O}.",
        "{S} lives with {O}, whom {S} married years ago.",
        "Records list {O} as the husband or wife of {S}.",
    ]),
    "award": ("person", "award", "the award received by {S}", [
        "{S} received {O}.",
        "{O} was given to {S}.",
        "Among {S}'s honours is {O}.",
        "{S} was the recipient of {O}.",
        "The jury awarded {O} to {S}.",
    ]),
    "employer": ("person", "org", "the employer of {S}", [
        "{S} works for {O}.",
        "{O} employs {S}.",
        "{S} has been on the staff of {O} for years.",
        "The employer of {S} is {O}.",
        "{S} joined {O} as a researcher.",
    ]),
    "headquarters": ("org", "place", "the headquarters location of {S}", [
        "{S} is headquartered in {O}.",
        "The head office of {S} is in {O}.",
        "{S} runs its operations from {O}.",
        "{O} hosts the headquarters of {S}.",
        "{S}'s main offices are located in {O}.",
    ]),
}
_GEN = {"person": _person, "place": _place, "org": _org, "award": _award}

# hop chains: each next relation's subject type must equal the previous object type
CHAINS = {
    1: [["birthplace"], ["founder"], ["award"], ["headquarters"], ["spouse"], ["employer"]],
    2: [["founder", "birthplace"], ["employer", "headquarters"], ["spouse", "award"], ["founder", "spouse"], ["employer", "founder"]],
    3: [["founder", "birthplace", "capital"], ["employer", "founder", "birthplace"], ["spouse", "employer", "headquarters"]],
}
PARTIAL = "{S} is associated with a {REL}, but the relevant name is not recorded in this source."
FILLER = [
    "The local council met on {Y} to discuss the harvest.", "A small museum in {P} keeps a collection of pressed flowers.",
    "{N} published a pamphlet on bridge maintenance in {Y}.", "The river near {P} floods roughly every decade.",
    "{N} is remembered for a lengthy correspondence about weather.", "Nothing of note happened in {P} in {Y}.",
]


def _filler(rng: random.Random, n: int = 2) -> str:
    return " ".join(rng.choice(FILLER).format(Y=_year(rng), P=_place(rng), N=_person(rng)) for _ in range(n))


def _paragraph(rng: random.Random, rel: str, S: str, O: str, family: int, pid: str, role: str, hop: int, meta=None) -> Paragraph:
    tpl = RELATIONS[rel][3][family]
    sent = tpl.format(S=S, O=O)
    body = [sent, _filler(rng, 1)]
    rng.shuffle(body)
    return Paragraph(pid=pid, title=S, text=" ".join([_filler(rng, 1)] + body), role=role, hop=hop, meta=meta or {})


def make_question(rng: random.Random, chain: List[str], qid: str, entity_pool: Dict[str, List[str]]) -> Question:
    """Build one question along ``chain`` (list of relations, hop order)."""
    n = len(chain)
    subj_type = RELATIONS[chain[0]][0]
    entities = [_GEN[subj_type](rng)]
    for rel in chain:
        entities.append(_GEN[RELATIONS[rel][1]](rng))
    # question text by composition: innermost first
    phrase = entities[0]
    for rel in chain:
        phrase = RELATIONS[rel][2].format(S=phrase)
    question = f"What is {phrase}?" if RELATIONS[chain[-1]][1] != "person" else f"Who is {phrase}?"
    gold, variants, decomposition, rels = [], {}, [], []
    for h, rel in enumerate(chain):
        S, O = entities[h], entities[h + 1]
        fam = rng.randrange(5)
        gold.append(_paragraph(rng, rel, S, O, fam, f"g{h}", "gold", h))
        fam2 = rng.choice([f for f in range(5) if f != fam])
        false_obj = _GEN[RELATIONS[rel][1]](rng)
        variants[f"g{h}"] = {
            "paraphrase": _paragraph(rng, rel, S, O, fam2, f"p{h}", "paraphrase", h),
            "partial": Paragraph(pid=f"q{h}", title=S, text=f"{_filler(rng, 1)} {PARTIAL.format(S=S, REL=rel)} {_filler(rng, 1)}", role="partial", hop=h),
            "contradictory": _paragraph(rng, rel, S, false_obj, fam2, f"c{h}", "false", h, meta={"false_answer": false_obj, "true_answer": O, "natural": False}),
        }
        decomposition.append({"question": f"{S} >> {rel}", "answer": O})
        rels.append(rel)
        entity_pool.setdefault(rel, []).append(O)
    # distractors: near-miss (same relation, other subject), (same subject, other relation), unrelated filler
    distractors = []
    for i in range(6):
        rel = rng.choice(list(RELATIONS)); S2 = _GEN[RELATIONS[rel][0]](rng); O2 = _GEN[RELATIONS[rel][1]](rng)
        distractors.append(_paragraph(rng, rel, S2, O2, rng.randrange(5), f"d{i}", "distractor", None))
    for i in range(6, 9):   # same subject as a hop, different relation
        h = rng.randrange(n); S = entities[h]
        other = [r for r in RELATIONS if r != chain[h] and RELATIONS[r][0] == RELATIONS[chain[h]][0]]
        rel = rng.choice(other) if other else rng.choice(list(RELATIONS))
        distractors.append(_paragraph(rng, rel, S, _GEN[RELATIONS[rel][1]](rng), rng.randrange(5), f"d{i}", "distractor", None))
    for i in range(9, 12):
        distractors.append(Paragraph(pid=f"d{i}", title=_place(rng), text=_filler(rng, 3), role="distractor"))
    return Question(
        qid=qid, dataset="synthetic", question=question, answer=entities[-1], aliases=[], gold=gold, distractors=distractors,
        n_hops=n, decomposition=decomposition, relation_types=rels, qtype=f"{n}hop:" + "-".join(chain),
        bridge_entities=entities[1:-1], extra={"variants": {k: {kk: vv.__dict__ for kk, vv in v.items()} for k, v in variants.items()}, "domain": "synthetic"},
    )


def generate(n_questions: int = 1500, seed: int = 0, hops=(1, 2, 3)) -> List[Question]:
    rng = random.Random(seed)
    pool: Dict[str, List[str]] = {}
    qs = []
    for i in range(n_questions):
        n = hops[i % len(hops)]
        chain = rng.choice(CHAINS[n])
        qs.append(make_question(rng, chain, f"syn{n}_{i:05d}", pool))
    return qs
