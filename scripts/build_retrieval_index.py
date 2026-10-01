"""Parse the wiki-18 corpus into the npz cache and build the BM25 index (CPU, long-running)."""
import logging, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
from revise.data.retrieval import WikiCorpus, BM25Retriever
t = time.time(); c = WikiCorpus(); print("corpus ready", len(c), "%.0fs" % (time.time() - t), flush=True)
t = time.time(); r = BM25Retriever(c); print("bm25 ready %.0fs" % (time.time() - t), flush=True)
hits = r.search(["Who is the spouse of the Green performer?", "What is the capital of the region containing Nuevo Laredo?"], k=5)
for h in hits: print([(c.passage(i)["title"], round(s, 1)) for i, s in h])
