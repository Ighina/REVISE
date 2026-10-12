"""Check the E5 dense retriever: load the index and report gold-hop recall on MuSiQue.

Concatenates the FAISS index parts on first use (slow), runs the multi-query
retrieval of ``revise.data.retrieval`` on the first 200 questions of the
MuSiQue run and reports how many gold hops per question are recovered, which is
the number quoted for the retrieved-evidence conditions.
"""
import collections
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a checkout without installing

from revise.data.retrieval import E5Retriever, WikiCorpus, label_passages, merge_hits, question_queries  # noqa: E402
from revise.experiment import Run, load_config  # noqa: E402

N_QUESTIONS = 200
K = 30


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    qs = Run(load_config("configs/musique_2hop_qwen7b.yaml")).questions()[:N_QUESTIONS]
    t = time.time(); corpus = WikiCorpus(); print(f"corpus {len(corpus)} passages {time.time() - t:.0f}s", flush=True)
    t = time.time(); retriever = E5Retriever(corpus, device="cpu"); print(f"e5 index loaded {time.time() - t:.0f}s", flush=True)

    qlists = [question_queries(q) for q in qs]
    flat = [x for ql in qlists for x in ql]
    t = time.time(); hits = retriever.search(flat, k=K)
    print(f"search {len(flat)} queries (multi-query) {time.time() - t:.0f}s", flush=True)

    found, any_gold, pos, mix = collections.Counter(), 0, 0, {}
    for q, ql in zip(qs, qlists):
        merged = merge_hits(hits[pos:pos + len(ql)], K); pos += len(ql)
        labelled = label_passages(q, [{**corpus.passage(i), "score": s, "rank": k} for k, (i, s, _) in enumerate(merged)])
        gold_hops = {p["hop"] for p in labelled if p["label"] == "retrieved_gold"}
        found[len(gold_hops)] += 1; any_gold += bool(gold_hops)
        mix = dict(collections.Counter(p["label"] for p in labelled))
    print("gold hops found per question:", dict(found), "| any gold:", any_gold / len(qs))
    print("label mix on the last question:", mix)


if __name__ == "__main__":
    main()
