from revise.data.retrieval import apply_retrieved, label_passages
from tests.test_trajectories import _q


def test_label_and_apply():
    q = _q("r1", 2, answer="Alpha City")
    q.gold[0].title, q.gold[1].title = "Pinos Laboratories", "Torzu Oselwex"
    q.decomposition[0]["answer"], q.bridge_entities = "Torzu Oselwex", ["Torzu Oselwex"]
    passages = [
        {"pid": "a", "title": "Pinos Laboratories", "text": "Pinos Laboratories was founded by Torzu Oselwex.", "score": 9.0, "rank": 0},
        {"pid": "b", "title": "Pinos Laboratories", "text": "The company makes glass.", "score": 8.0, "rank": 1},
        {"pid": "c", "title": "Alpha City FC", "text": "Alpha City FC is a football club.", "score": 7.0, "rank": 2},
        {"pid": "d", "title": "Torzu Oselwex", "text": "Torzu Oselwex was born in Alpha City.", "score": 6.0, "rank": 3},
        {"pid": "e", "title": "Weather", "text": "It rained.", "score": 1.0, "rank": 4},
        {"pid": "f", "title": "Rivers", "text": "Rivers flow.", "score": 0.5, "rank": 5},
    ]
    lab = label_passages(q, passages)
    assert [p["label"] for p in lab] == ["retrieved_gold", "same_page", "near_miss", "retrieved_gold", "irrelevant", "irrelevant"]
    row = {"qid": q.qid, "method": "bm25", "passages": lab}
    q_noise = apply_retrieved(q, row, "noise")
    assert [p.title for p in q_noise.gold] == ["Pinos Laboratories", "Torzu Oselwex"] and len(q_noise.distractors) == 4
    assert q_noise.distractors[0].meta["retrieval_label"] == "same_page"
    q_full = apply_retrieved(q, row, "full")
    assert q_full is not None and q_full.gold[1].text.startswith("Torzu Oselwex was born")
    lab2 = [p for p in lab if p["pid"] != "d"]
    assert apply_retrieved(q, {"qid": q.qid, "method": "bm25", "passages": lab2}, "full") is None
