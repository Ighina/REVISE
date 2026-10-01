"""Construct evidence conditions and paired increments for a question.

For a question with gold hop paragraphs g_0..g_{n-1} we build (names follow
RESEARCH_PLAN.md):

  C0           no evidence
  CD           matched distractors only (n of them, length-matched to golds)
  chain(o)     for each hop order o (canonical, reverse, random perms):
               prefixes o[:1], o[:2], ..., o[:n]  -> the last is minimally
               sufficient (C_12 for two hops)
  LOO          for n>=3: all golds but one (insufficient) and its completion
  C_ALL+D      sufficient + distractor        C_ALL+R  sufficient + redundant
  C_F          full available document context (all paragraphs, shuffled)

and from every *penultimate* state S (n-1 golds present) the matched
increments

  S -> S+g_last        decisive           (positive)
  S -> S+D             distractor         (matched length)
  S -> S+R             redundant          (duplicate of an already-present gold)
  S -> S+F             false              (final hop with its answer swapped)
  S -> S+A             answer_ctrl        (answer string inside an irrelevant paragraph)

New paragraphs are inserted at a *random position* of the previous
presentation order so that increments are textually minimal while the
absolute position of the decisive paragraph is randomised.
"""
from __future__ import annotations

import random
import re
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from revise.data.schema import Condition, Increment, Paragraph, Question


def _wc(p: Paragraph) -> int:
    return len(p.text.split())


def _cid(paragraphs: Sequence[Paragraph]) -> str:
    return "C0" if not paragraphs else "C[" + "+".join(p.pid for p in paragraphs) + "]"


def _insert(prev: Sequence[Paragraph], new: Sequence[Paragraph], rng: random.Random) -> Tuple[List[Paragraph], int]:
    pos = rng.randint(0, len(prev))
    out = list(prev[:pos]) + list(new) + list(prev[pos:])
    return out, pos


class DistractorPool:
    """Hands out unused distractors, length-matched to a target paragraph."""

    def __init__(self, distractors: Sequence[Paragraph], rng: random.Random):
        self.avail = list(distractors)
        rng.shuffle(self.avail)
        self.rng = rng

    def take_matched(self, target: Paragraph) -> Optional[Paragraph]:
        if not self.avail:
            return None
        tw = _wc(target)
        best = min(self.avail, key=lambda p: abs(_wc(p) - tw))
        self.avail.remove(best)
        return best

    def take_any(self) -> Optional[Paragraph]:
        return self.avail.pop() if self.avail else None


def _replace_ci(text: str, needle: str, repl: str) -> Tuple[str, int]:
    pat = re.compile(re.escape(needle), flags=re.IGNORECASE)
    new, n = pat.subn(lambda _m: repl, text)   # function replacement: no backslash/group processing of ``repl``
    return new, n


def falsify(q: Question, hop: int, false_answer: str) -> Optional[Paragraph]:
    """Return a copy of gold hop ``hop`` with its sub-answer replaced by ``false_answer``."""
    p = q.gold[hop]
    sub_ans = q.decomposition[hop]["answer"] if hop < len(q.decomposition) else q.answer
    for cand in [sub_ans] + ([q.answer] + q.aliases if hop == q.n_hops - 1 else []):
        if not cand:
            continue
        text, n = _replace_ci(p.text, cand, false_answer)
        if n > 0:
            title, _ = _replace_ci(p.title, cand, false_answer)
            return Paragraph(pid=f"f{hop}", title=title, text=text, role="false", hop=hop,
                             meta={"false_answer": false_answer, "true_answer": cand})
    return None


def answer_control(q: Question, base: Paragraph) -> Paragraph:
    """Distractor paragraph carrying the *answer string* in an irrelevant context."""
    text = f"{base.text} (Not to be confused with {q.answer}.)"
    return Paragraph(pid=f"a_{base.pid}", title=base.title, text=text, role="answer_ctrl")


def redundant_copy(p: Paragraph) -> Paragraph:
    return Paragraph(pid=f"r{p.hop}", title=p.title, text=p.text, role="redundant", hop=p.hop)


def _variants(q: Question) -> Dict[str, Dict[str, Paragraph]]:
    """Optional per-hop variants stored in ``q.extra['variants']`` as
    ``{"g0": {"paraphrase": {...}, "partial": {...}, "contradictory": {...}}, ...}``."""
    out: Dict[str, Dict[str, Paragraph]] = {}
    for gk, d in (q.extra or {}).get("variants", {}).items():
        out[gk] = {name: (p if isinstance(p, Paragraph) else Paragraph(**p)) for name, p in d.items()}
    return out


def build_trajectories(q: Question, false_answer_pool: Dict[str, List[str]], seed: int = 0,
                       n_random_perms: int = 2, include_full: bool = True,
                       include_loo: bool = True) -> Tuple[List[Condition], List[Increment]]:
    rng = random.Random(f"{seed}:{q.qid}")
    n = q.n_hops
    conds: Dict[str, Condition] = {}
    incs: List[Increment] = []

    def add_cond(name: str, pars: Sequence[Paragraph], order_tag: str, inserted: Optional[int] = None) -> Condition:
        cid = _cid(pars)
        if cid in conds:
            return conds[cid]
        gold_hops = sorted({p.hop for p in pars if p.role in ("gold", "paraphrase", "independent")})   # variants supply the hop too
        false_hops = {p.hop for p in pars if p.role == "false"}
        c = Condition(
            cid=cid, qid=q.qid, name=name, paragraphs=list(pars),
            sufficient=(len(gold_hops) == n),
            gold_hops=gold_hops, n_gold=len(gold_hops),
            n_distractor=sum(p.role == "distractor" for p in pars),
            has_false=bool(false_hops), has_answer_ctrl=any(p.role == "answer_ctrl" for p in pars),
            has_redundant=any(p.role == "redundant" for p in pars),
            order_tag=order_tag, inserted_position=inserted,
        )
        conds[cid] = c
        return c

    def add_inc(kind: str, frm: Condition, to: Condition, added: Sequence[Paragraph], order_tag: str, group: str):
        iid = f"{frm.cid}->{to.cid}"
        if any(i.iid == iid for i in incs):
            return
        incs.append(Increment(
            iid=iid, qid=q.qid, from_cid=frm.cid, to_cid=to.cid, kind=kind,
            becomes_sufficient=(to.sufficient and not frm.sufficient),
            from_sufficient=frm.sufficient, to_sufficient=to.sufficient,
            added_pids=[p.pid for p in added], order_tag=order_tag, control_group=group,
        ))

    pool = DistractorPool(q.distractors, rng)
    c0 = add_cond("none", [], "none")

    # --- distractors only ---------------------------------------------------
    cd_pars = [d for d in (pool.take_matched(g) for g in q.gold) if d is not None]
    if cd_pars:
        cd = add_cond("distractors_only", cd_pars, "none")
        add_inc("distractor_from_none", c0, cd, cd_pars, "none", f"{q.qid}|C0")

    # --- false-answer candidates per hop --------------------------------------
    def false_for_hop(hop: int) -> Optional[Paragraph]:
        rel = q.relation_types[hop] if hop < len(q.relation_types) else ""
        true_sub = (q.decomposition[hop]["answer"] if hop < len(q.decomposition) else q.answer).lower()
        cands = [a for a in false_answer_pool.get(rel, []) if a.lower() != true_sub and a.lower() != q.answer.lower()]
        if not cands:
            cands = [a for a in false_answer_pool.get("*", []) if a.lower() != true_sub and a.lower() != q.answer.lower()]
        if not cands:
            return None
        return falsify(q, hop, rng.choice(cands))

    # --- hop orders -----------------------------------------------------------
    orders: List[Tuple[str, List[int]]] = [("canonical", list(range(n))), ("reverse", list(range(n))[::-1])]
    if n >= 3:
        seen = {tuple(o) for _, o in orders}
        tries = 0
        while len(orders) < 2 + n_random_perms and tries < 50:
            perm = list(range(n)); rng.shuffle(perm); tries += 1
            if tuple(perm) not in seen:
                seen.add(tuple(perm)); orders.append((f"perm{len(orders) - 1}", perm))

    def build_controls(S: Condition, last_hop: int, order_tag: str):
        """Matched controls from penultimate state S (all golds but ``last_hop``)."""
        group = f"{q.qid}|{S.cid}"
        g_last = q.gold[last_hop]
        # distractor (length-matched to the decisive paragraph)
        d = pool.take_matched(g_last)
        if d is not None:
            pars, pos = _insert(S.paragraphs, [d], rng)
            add_inc("distractor", S, add_cond("penultimate_plus_distractor", pars, order_tag, pos), [d], order_tag, group)
        # redundant: duplicate of a gold already present
        present = [p for p in S.paragraphs if p.role == "gold"]
        if present:
            r = redundant_copy(rng.choice(present))
            pars, pos = _insert(S.paragraphs, [r], rng)
            add_inc("redundant", S, add_cond("penultimate_plus_redundant", pars, order_tag, pos), [r], order_tag, group)
        # false: decisive hop with its answer swapped
        f = false_for_hop(last_hop)
        if f is not None:
            pars, pos = _insert(S.paragraphs, [f], rng)
            add_inc("false", S, add_cond("penultimate_plus_false", pars, order_tag, pos), [f], order_tag, group)
        # answer-string control: answer inside an irrelevant paragraph
        base = pool.take_any()
        if base is not None:
            a = answer_control(q, base)
            pars, pos = _insert(S.paragraphs, [a], rng)
            add_inc("answer_ctrl", S, add_cond("penultimate_plus_answer_ctrl", pars, order_tag, pos), [a], order_tag, group)
        # linguistic-form / quality variants of the decisive hop, when the question carries them
        v = _variants(q).get(f"g{last_hop}", {})
        if "paraphrase" in v:      # same fact, different wording: crosses sufficiency
            pars, pos = _insert(S.paragraphs, [v["paraphrase"]], rng)
            add_inc("decisive_paraphrase", S, add_cond("sufficient_min_paraphrase", pars, order_tag, pos), [v["paraphrase"]], order_tag, group)
        if "independent" in v:     # independently authored statement of the same hop: crosses sufficiency
            pars, pos = _insert(S.paragraphs, [v["independent"]], rng)
            add_inc("decisive_independent", S, add_cond("sufficient_min_independent", pars, order_tag, pos), [v["independent"]], order_tag, group)
        if "partial" in v:         # subject and relation mentioned, value withheld: stays insufficient
            pars, pos = _insert(S.paragraphs, [v["partial"]], rng)
            add_inc("partial", S, add_cond("penultimate_plus_partial", pars, order_tag, pos), [v["partial"]], order_tag, group)
        if "contradictory" in v:   # pre-built false variant (e.g. naturally retrieved contradiction)
            pars, pos = _insert(S.paragraphs, [v["contradictory"]], rng)
            add_inc("false_variant", S, add_cond("penultimate_plus_false_variant", pars, order_tag, pos), [v["contradictory"]], order_tag, group)

    sufficient_conds: List[Condition] = []
    for order_tag, order in orders:
        prev = c0
        prev_pars: List[Paragraph] = []
        for k, hop in enumerate(order):
            g = q.gold[hop]
            pars, pos = _insert(prev_pars, [g], rng)
            is_last = (k == n - 1)
            name = "sufficient_min" if is_last else ("gold_single" if k == 0 else "chain_prefix")
            cur = add_cond(name, pars, order_tag, pos)
            kind = "decisive" if is_last else "gold_nondecisive"
            add_inc(kind, prev, cur, [g], order_tag, f"{q.qid}|{prev.cid}")
            if is_last:
                build_controls(prev, hop, order_tag)
                sufficient_conds.append(cur)
            prev, prev_pars = cur, pars

    # --- leave-one-out (n >= 3) ---------------------------------------------
    if include_loo and n >= 3:
        for hop in range(n):
            others = [g for g in q.gold if g.hop != hop]
            rng.shuffle(others)
            loo = add_cond("loo", others, "loo")
            pars, pos = _insert(others, [q.gold[hop]], rng)
            full = add_cond("sufficient_min", pars, "loo", pos)
            add_inc("loo_decisive", loo, full, [q.gold[hop]], "loo", f"{q.qid}|{loo.cid}")
            build_controls(loo, hop, "loo")

    # --- post-sufficiency increments ------------------------------------------
    if sufficient_conds:
        S = sufficient_conds[0]   # canonical-order minimal sufficient context
        group = f"{q.qid}|{S.cid}"
        d = pool.take_matched(q.gold[-1]) or pool.take_any()
        if d is not None:
            pars, pos = _insert(S.paragraphs, [d], rng)
            add_inc("post_distractor", S, add_cond("sufficient_plus_distractor", pars, "canonical", pos), [d], "canonical", group)
        gv = _variants(q)
        for kind, inc_kind, cname in (("paraphrase", "post_redundant_paraphrase", "sufficient_plus_paraphrase"),
                                      ("independent", "post_redundant_independent", "sufficient_plus_independent")):
            for gk, v in gv.items():    # paraphrastic / independent-source redundancy after sufficiency
                if kind in v:
                    pars, pos = _insert(S.paragraphs, [v[kind]], rng)
                    add_inc(inc_kind, S, add_cond(cname, pars, "canonical", pos), [v[kind]], "canonical", group)
                    break
        r = redundant_copy(rng.choice(q.gold))
        pars, pos = _insert(S.paragraphs, [r], rng)
        add_inc("post_redundant", S, add_cond("sufficient_plus_redundant", pars, "canonical", pos), [r], "canonical", group)
        f = false_for_hop(n - 1)
        if f is not None:   # conflicting false evidence on top of sufficient evidence
            pars, pos = _insert(S.paragraphs, [f], rng)
            add_inc("post_false", S, add_cond("sufficient_plus_false", pars, "canonical", pos), [f], "canonical", group)
        if include_full:
            allp = list(q.gold) + list(q.distractors)
            rng.shuffle(allp)
            cf = add_cond("full", allp, "shuffled")
            add_inc("full_context", S, cf, [p for p in allp if p.role != "gold"], "shuffled", group)

    return list(conds.values()), incs


def build_false_answer_pool(questions: Sequence[Question]) -> Dict[str, List[str]]:
    """Map relation type -> list of sub-answers seen for that relation (plus '*' for all)."""
    pool: Dict[str, List[str]] = {"*": []}
    for q in questions:
        for hop, d in enumerate(q.decomposition):
            rel = q.relation_types[hop] if hop < len(q.relation_types) else ""
            a = d["answer"].strip()
            if not a:
                continue
            pool.setdefault(rel, []).append(a)
            pool["*"].append(a)
    return {k: sorted(set(v)) for k, v in pool.items()}


def build_all(questions: Sequence[Question], seed: int = 0, **kw) -> Tuple[List[Condition], List[Increment]]:
    pool = build_false_answer_pool(questions)
    conds: List[Condition] = []
    incs: List[Increment] = []
    for q in questions:
        c, i = build_trajectories(q, pool, seed=seed, **kw)
        conds.extend(c); incs.extend(i)
    return conds, incs
