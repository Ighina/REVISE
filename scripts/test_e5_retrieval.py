"""Concatenate the E5 FAISS index parts (first use), run E5 retrieval on MuSiQue questions, report gold recall."""
import logging, time, collections
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
from revise.experiment import load_config, Run
from revise.data.retrieval import WikiCorpus, E5Retriever, label_passages
cfg = load_config("configs/musique_2hop_qwen7b.yaml"); qs = Run(cfg).questions()[:200]
t = time.time(); c = WikiCorpus(); print("corpus", len(c), "%.0fs" % (time.time() - t), flush=True)
t = time.time(); r = E5Retriever(c, device="cpu"); print("e5 index loaded %.0fs" % (time.time() - t), flush=True)
t = time.time(); hits = r.search([q.question for q in qs], k=30); print("search 200 queries %.0fs" % (time.time() - t), flush=True)
rec = collections.Counter(); any_gold = 0
for q, h in zip(qs, hits):
    lab = label_passages(q, [{**c.passage(i), "score": s, "rank": k} for k, (i, s) in enumerate(h)])
    hops = {p["hop"] for p in lab if p["label"] == "retrieved_gold"}
    rec[len(hops)] += 1; any_gold += bool(hops)
    cnt = collections.Counter(p["label"] for p in lab)
print("gold hops found per question (of 2):", dict(rec), "| any gold:", any_gold / len(qs))
print("label mix on last question:", dict(cnt))
