"""Concatenate the E5 FAISS index parts (first use), run E5 retrieval on MuSiQue questions, report gold recall."""
import logging, time, collections
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
from revise.experiment import load_config, Run
from revise.data.retrieval import WikiCorpus, E5Retriever, label_passages
cfg = load_config("configs/musique_2hop_qwen7b.yaml"); qs = Run(cfg).questions()[:200]
t = time.time(); c = WikiCorpus(); print("corpus", len(c), "%.0fs" % (time.time() - t), flush=True)
t = time.time(); r = E5Retriever(c, device="cpu"); print("e5 index loaded %.0fs" % (time.time() - t), flush=True)
from revise.data.retrieval import question_queries, merge_hits
qlists = [question_queries(q) for q in qs]; flat = [x for ql in qlists for x in ql]
t = time.time(); hits = r.search(flat, k=30); print("search %d queries (multi-query) %.0fs" % (len(flat), time.time() - t), flush=True)
rec = collections.Counter(); any_gold = 0; pos = 0
for q, ql in zip(qs, qlists):
    per = hits[pos:pos + len(ql)]; pos += len(ql); h = merge_hits(per, 30)
    lab = label_passages(q, [{**c.passage(i), "score": s, "rank": k} for k, (i, s, _) in enumerate(h)])
    hops = {p["hop"] for p in lab if p["label"] == "retrieved_gold"}
    rec[len(hops)] += 1; any_gold += bool(hops)
    cnt = collections.Counter(p["label"] for p in lab)
print("gold hops found per question (of 2):", dict(rec), "| any gold:", any_gold / len(qs))
print("label mix on last question:", dict(cnt))
