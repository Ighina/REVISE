"""Parse the wiki-18 corpus into the npz cache and build the BM25 index.

One-off preparation for the retrieved-evidence (ESTB) runs: CPU only, a few
hours, and the caches it writes under ``$REVISE_SCRATCH/retrieval`` are reused
by every later run.  Prints the top hits for two MuSiQue questions as a check.
"""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a checkout without installing

from revise.data.retrieval import BM25Retriever, WikiCorpus  # noqa: E402

QUERIES = ["Who is the spouse of the Green performer?", "What is the capital of the region containing Nuevo Laredo?"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t = time.time(); corpus = WikiCorpus(); print(f"corpus ready {len(corpus)} passages {time.time() - t:.0f}s", flush=True)
    t = time.time(); retriever = BM25Retriever(corpus); print(f"bm25 ready {time.time() - t:.0f}s", flush=True)
    for hits in retriever.search(QUERIES, k=5):
        print([(corpus.passage(i)["title"], round(s, 1)) for i, s in hits])


if __name__ == "__main__":
    main()
