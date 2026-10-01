"""Retrieval layer for the Evidence State Transition Benchmark (retrieval-source axis).

Retrieves passages from the wiki-18 corpus (FlashRAG's 2018 Wikipedia dump,
~21M 100-word passages) with BM25 (``bm25s``) or dense E5 (FAISS flat index
from ``PeterJinGo/wiki-18-e5-index`` + ``intfloat/e5-base-v2`` queries), labels
each retrieved passage against the question's gold hops, and can replace a
question's benchmark-provided distractors (or its gold paragraphs) with
naturally retrieved ones.

Passage labels
--------------
  retrieved_gold   title matches a gold hop page AND the hop's sub-answer occurs in the passage
  same_page        title matches a gold hop page but the sub-answer is absent (partial evidence)
  near_miss        passage mentions a bridge entity / the question subject, or the final answer,
                   without being a gold page (answer-string-without-relation in the wild)
  irrelevant       everything else
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from revise import config
from revise.data.schema import Paragraph, Question, append_jsonl, read_jsonl
from revise.eval import normalize_answer

log = logging.getLogger(__name__)
RETRIEVAL_DIR = config.SCRATCH_ROOT / "retrieval"
RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)


def _hub_file(repo: str, name: str) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, repo_type="dataset", local_files_only=True)) / name


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------

class WikiCorpus:
    """wiki-18 passages as parallel arrays; cached as .npy after first parse."""

    def __init__(self, max_passages: Optional[int] = None):
        cache = RETRIEVAL_DIR / "wiki18_corpus.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            self.ids, self.titles, self.texts = z["ids"], z["titles"], z["texts"]
        else:
            src = _hub_file("PeterJinGo/wiki-18-corpus", "wiki-18.jsonl.gz")
            ids, titles, texts = [], [], []
            for n, line in enumerate(self._iter_lines(src)):
                if max_passages and n >= max_passages:
                    break
                r = json.loads(line)
                title, text = self._split(r)
                ids.append(str(r.get("id", n))); titles.append(title); texts.append(text)
                if n % 2_000_000 == 0:
                    log.info("corpus parse: %d passages", n)
            self.ids, self.titles, self.texts = np.array(ids, dtype=object), np.array(titles, dtype=object), np.array(texts, dtype=object)
            if not max_passages:
                np.savez(cache, ids=self.ids, titles=self.titles, texts=self.texts)
        log.info("wiki-18 corpus: %d passages", len(self.ids))

    @staticmethod
    def _iter_lines(src: Path) -> Iterable[str]:
        """The Hub file is a gzipped tar containing one JSONL member (FlashRAG dump);
        fall back to plain gzipped JSONL."""
        import io, tarfile
        with open(src, "rb") as fh:
            head = fh.read(2)
        with gzip.open(src, "rb") as g:
            probe = g.read(512)
        if b"ustar" in probe:
            with tarfile.open(src, mode="r|gz") as tar:
                for m in tar:
                    if m.isfile():
                        f = tar.extractfile(m)
                        for raw in f:          # streaming member: iterate raw lines
                            line = raw.decode("utf-8", errors="replace")
                            if line.strip():
                                yield line
        else:
            with gzip.open(src, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield line

    @staticmethod
    def _split(r: dict) -> Tuple[str, str]:
        if "contents" in r:                      # FlashRAG format: '"Title"\ntext'
            c = r["contents"]
            if "\n" in c:
                t, x = c.split("\n", 1)
                return t.strip().strip('"'), x.strip()
            return "", c.strip()
        return str(r.get("title", "")).strip(), str(r.get("text", "")).strip()

    def __len__(self):
        return len(self.ids)

    def passage(self, i: int) -> dict:
        return {"pid": self.ids[i], "title": self.titles[i], "text": self.texts[i]}


# ---------------------------------------------------------------------------
# retrievers
# ---------------------------------------------------------------------------

class BM25Retriever:
    def __init__(self, corpus: WikiCorpus):
        import bm25s
        self.corpus = corpus
        idx_dir = RETRIEVAL_DIR / "bm25s_wiki18"
        if idx_dir.exists():
            self.bm25 = bm25s.BM25.load(str(idx_dir))
        else:
            import Stemmer
            self.stemmer = Stemmer.Stemmer("english")
            docs = [f"{t} {x}" for t, x in zip(corpus.titles, corpus.texts)]
            log.info("tokenising %d passages for BM25", len(docs))
            tokens = bm25s.tokenize(docs, stopwords="en", stemmer=self.stemmer, show_progress=False)
            self.bm25 = bm25s.BM25()
            self.bm25.index(tokens, show_progress=False)
            self.bm25.save(str(idx_dir))
        import Stemmer
        self.stemmer = Stemmer.Stemmer("english")

    def search(self, queries: Sequence[str], k: int = 20) -> List[List[Tuple[int, float]]]:
        import bm25s
        toks = bm25s.tokenize(list(queries), stopwords="en", stemmer=self.stemmer, show_progress=False)
        res, scores = self.bm25.retrieve(toks, k=k, show_progress=False, n_threads=max(1, (os.cpu_count() or 8) // 4))
        return [[(int(i), float(s)) for i, s in zip(res[q], scores[q])] for q in range(len(queries))]


class E5Retriever:
    def __init__(self, corpus: WikiCorpus, device: Optional[str] = None):
        import faiss, torch
        from transformers import AutoModel, AutoTokenizer
        self.corpus = corpus
        idx = RETRIEVAL_DIR / "e5_Flat.index"
        if not idx.exists():                     # the Hub repo ships the index split in two parts
            parts = [_hub_file("PeterJinGo/wiki-18-e5-index", p) for p in ("part_aa", "part_ab")]
            with open(idx, "wb") as out:
                for p in parts:
                    with open(p, "rb") as f:
                        while True:
                            chunk = f.read(1 << 26)
                            if not chunk:
                                break
                            out.write(chunk)
        log.info("loading FAISS index (this maps ~65 GB)")
        self.index = faiss.read_index(str(idx), faiss.IO_FLAG_MMAP)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
        self.enc = AutoModel.from_pretrained("intfloat/e5-base-v2").to(self.device).eval()

    def _embed(self, queries: Sequence[str]) -> np.ndarray:
        import torch
        out = []
        for b in range(0, len(queries), 64):
            batch = ["query: " + q for q in queries[b:b + 64]]
            enc = self.tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            with torch.no_grad():
                h = self.enc(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1)
                e = torch.nn.functional.normalize(e, dim=-1)
            out.append(e.float().cpu().numpy())
        return np.concatenate(out)

    def search(self, queries: Sequence[str], k: int = 20) -> List[List[Tuple[int, float]]]:
        q = self._embed(queries)
        scores, ids = self.index.search(q, k)
        return [[(int(i), float(s)) for i, s in zip(ids[r], scores[r]) if i >= 0] for r in range(len(queries))]


# ---------------------------------------------------------------------------
# labelling and application
# ---------------------------------------------------------------------------

def _norm_title(t: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*$", "", t.strip().lower())


def _contains(text: str, s: str) -> bool:
    ns = normalize_answer(s)
    return bool(ns) and re.search(r"\b" + re.escape(ns) + r"\b", normalize_answer(text)) is not None


def label_passages(q: Question, passages: List[dict]) -> List[dict]:
    gold_titles = {_norm_title(g.title): g.hop for g in q.gold}
    sub_answers = {g.hop: (q.decomposition[g.hop]["answer"] if g.hop < len(q.decomposition) else q.answer) for g in q.gold}
    entities = [e for e in q.bridge_entities if e] + [q.answer]
    out = []
    for p in passages:
        nt = _norm_title(p["title"]); body = f"{p['title']} {p['text']}"
        label, hop = "irrelevant", None
        if nt in gold_titles:
            hop = gold_titles[nt]
            label = "retrieved_gold" if _contains(body, sub_answers.get(hop, "")) else "same_page"
        elif any(_contains(body, e) for e in entities):
            label = "near_miss"
        out.append({**p, "label": label, "hop": hop})
    return out


def retrieve_for_questions(questions: Sequence[Question], method: str, k: int, out_path: Path,
                           corpus: Optional[WikiCorpus] = None) -> None:
    """Write ``{qid, method, passages: [...]}`` rows; resumable."""
    done = {r["qid"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [q for q in questions if q.qid not in done]
    if not todo:
        return
    corpus = corpus or WikiCorpus()
    retr = BM25Retriever(corpus) if method == "bm25" else E5Retriever(corpus)
    for b in range(0, len(todo), 256):
        batch = todo[b:b + 256]
        hits = retr.search([q.question for q in batch], k=k)
        for q, h in zip(batch, hits):
            passages = [{**corpus.passage(i), "score": s, "rank": r} for r, (i, s) in enumerate(h)]
            append_jsonl(out_path, {"qid": q.qid, "method": method, "passages": label_passages(q, passages)})
        log.info("retrieved %d/%d", min(b + 256, len(todo)), len(todo))


def apply_retrieved(q: Question, row: dict, mode: str = "noise", n_distractors: int = 18) -> Optional[Question]:
    """Return a copy of ``q`` whose distractors (mode ``noise``) or distractors *and* gold
    paragraphs (mode ``full``) come from retrieval.  ``full`` requires a retrieved_gold
    passage for every hop, else None."""
    ps = row["passages"]
    gold = list(q.gold)
    if mode == "full":
        found = {}
        for p in ps:
            if p["label"] == "retrieved_gold" and p["hop"] not in found:
                found[p["hop"]] = p
        if len(found) < q.n_hops:
            return None
        gold = [Paragraph(pid=f"g{h}", title=found[h]["title"], text=found[h]["text"], role="gold", hop=h,
                          meta={"source": row["method"], "rank": found[h]["rank"]}) for h in range(q.n_hops)]
    noise = [p for p in ps if p["label"] in ("near_miss", "same_page", "irrelevant")]
    noise.sort(key=lambda p: ({"same_page": 0, "near_miss": 1, "irrelevant": 2}[p["label"]], p["rank"]))
    distractors = [Paragraph(pid=f"d{i}", title=p["title"], text=p["text"], role="distractor",
                             meta={"source": row["method"], "rank": p["rank"], "retrieval_label": p["label"]})
                   for i, p in enumerate(noise[:n_distractors])]
    if len(distractors) < max(2, q.n_hops):
        return None
    q2 = Question(**{**q.__dict__, "gold": gold, "distractors": distractors})
    q2.extra = {**(q.extra or {}), "retrieval": {"method": row["method"], "mode": mode}}
    return q2
