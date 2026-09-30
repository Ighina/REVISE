from collections import Counter

from revise.data.schema import Paragraph, Question
from revise.data.trajectories import build_all, build_trajectories


def _q(qid: str, n_hops: int, answer: str = "Alpha City") -> Question:
    gold = [Paragraph(pid=f"g{h}", title=f"{qid}-G{h}", text=f"Gold hop {h}. Sub answer is Sub{h}." + (f" Final answer is {answer}." if h == n_hops - 1 else ""), role="gold", hop=h)
            for h in range(n_hops)]
    dis = [Paragraph(pid=f"d{i}", title=f"{qid}-D{i}", text=("word " * (5 + 3 * i)).strip(), role="distractor") for i in range(20)]
    return Question(qid=qid, dataset="synthetic", question=f"Q {qid}?", answer=answer, aliases=[], gold=gold, distractors=dis,
                    n_hops=n_hops, decomposition=[{"question": f"#{h} >> rel{h}", "answer": (f"Sub{h}" if h < n_hops - 1 else answer)} for h in range(n_hops)],
                    relation_types=[f"rel{(h + sum(map(ord, qid))) % 3}" for h in range(n_hops)], qtype=f"{n_hops}hop", bridge_entities=[f"Sub{h}" for h in range(n_hops - 1)])


def test_two_hop_structure():
    qs = [_q("a", 2, "Alpha City"), _q("b", 2, "Beta Town"), _q("c", 2, "Gamma Ville")]
    conds, incs = build_all(qs, seed=1)
    kinds = Counter(i.kind for i in incs if i.qid == "a")
    assert kinds["decisive"] == 2          # canonical + reverse
    assert kinds["distractor"] == 2 and kinds["redundant"] == 2 and kinds["answer_ctrl"] == 2
    assert kinds["false"] == 2            # answer string present in the paragraph -> falsifiable
    for i in incs:
        assert i.becomes_sufficient == (i.kind in ("decisive", "loo_decisive"))
        if i.kind in ("distractor", "redundant", "false", "answer_ctrl"):
            assert not i.to_sufficient
    # every decisive increment shares its control group with >= 3 matched controls
    groups = Counter((i.control_group, i.kind) for i in incs)
    for i in incs:
        if i.kind == "decisive":
            assert sum(groups[(i.control_group, k)] for k in ("distractor", "redundant", "false", "answer_ctrl")) >= 3
    by_key = {c.key: c for c in conds}
    for i in incs:
        frm, to = by_key[f"{i.qid}::{i.from_cid}"], by_key[f"{i.qid}::{i.to_cid}"]
        assert len(to.paragraphs) == len(frm.paragraphs) + len(i.added_pids) or i.kind == "full_context"
        assert set(p.pid for p in frm.paragraphs) <= set(p.pid for p in to.paragraphs) or i.kind == "full_context"
    fc = [c for c in conds if c.has_false][0]
    fp = [p for p in fc.paragraphs if p.role == "false"][0]
    assert fp.meta["false_answer"].lower() in fp.text.lower() and fp.meta["true_answer"].lower() not in fp.text.lower()


def test_three_hop_has_loo_and_perms():
    qs = [_q("a", 3), _q("b", 3, "Beta Town")]
    conds, incs = build_all(qs, seed=0, n_random_perms=2)
    kinds = Counter(i.kind for i in incs if i.qid == "a")
    tags = {i.order_tag for i in incs if i.qid == "a"}
    assert {"canonical", "reverse", "perm1", "perm2", "loo"} <= tags
    # leave-one-out states coincide with chain prefixes for some orders and are de-duplicated,
    # so only the *union* of decisive completions is guaranteed
    assert kinds["decisive"] + kinds["loo_decisive"] >= 4
    assert all(len(c.paragraphs) == 2 for c in conds if c.name == "loo")
    assert all(len(c.paragraphs) == 3 and c.sufficient for c in conds if c.name == "sufficient_min")


def test_deterministic():
    q = _q("z", 2)
    pool = {"*": ["Other Place"], "rel1": ["Other Place"]}
    a = build_trajectories(q, pool, seed=3)
    b = build_trajectories(q, pool, seed=3)
    assert [c.cid for c in a[0]] == [c.cid for c in b[0]]
    assert [i.iid for i in a[1]] == [i.iid for i in b[1]]
