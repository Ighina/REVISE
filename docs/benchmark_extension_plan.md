# Benchmark extension plan (response to `first_revision.md`)

The review's central objection is external validity: the evidence trajectories are researcher-built from gold paragraphs of Wikipedia multi-hop benchmarks, so the claims about "RAG evidence uptake" outrun the evidence. It asks for (5) HotpotQA, (6) a single-hop setting, (7) naturally retrieved noisy evidence, (8–9) RAGBench with domain shift, (10) closed-book-correct questions, (11) stronger redundancy manipulations, and (12) a purpose-built Evidence State Transition Benchmark. This document sets out what is added, in what order, and how point 12 can be assembled from what already exists.

## 1. Queue of additional benchmarks

All additions reuse the existing pipeline (`screen → collect → features → probes → transfer → ablate/patch → report`); each needs only a loader that yields `Question` objects with ordered gold units and distractors, plus a config. Runs use the GPU for collection (about one GPU-hour per 2,000 questions) and CPU for analysis.

| # | benchmark | role for the review | loader unit | status |
|---|---|---|---|---|
| 1 | **HotpotQA** (distractor setting, 2 supporting paragraphs + 8 distractors; bridge and comparison types) | independent multi-hop replication (point 5); the transfer chain MuSiQue → 2Wiki → HotpotQA | gold paragraph per supporting title, ordered by first supporting-fact sentence | `revise/data/loaders.py::load_hotpotqa` |
| 2 | **RAGBench** (12 sub-datasets, 5 domains: biomedical `pubmedqa`/`covidqa`, legal `cuad`, technical `techqa`/`emanual`, finance `finqa`/`tatqa`, customer support `delucionqa`, plus general `msmarco`/`hagrid`/`expertqa`/`hotpotqa`) | realistic, mostly single-hop RAG with naturally retrieved documents and annotated relevance (points 6, 7, 8, 9) | a "gold unit" is a retrieved document containing at least one annotated relevant sentence; the other retrieved documents are the natural distractors | `load_ragbench` |
| 3 | **Synthetic controlled benchmark** | clean causal trajectories with full control of quantity, quality and linguistic form; fictional entities so closed-book failure holds by construction (point 12, partial) | templated 1–3-hop facts with paraphrase families, contradictory variants, partial (sub-relation) variants | `revise/data/synthetic.py` |
| 4 | **Evidence State Transition Benchmark (ESTB)** | the benchmark of point 12, assembled from 1–3 plus retrieval (section 2) | union of the above with new axes | `revise/data/estb.py` |

Cheap additions that the queue also covers because they are configuration changes, not new code:

* **Closed-book-correct questions (point 10).** `data.select: all` keeps the whole screened pool and records closed-book status per question, so probes and increments can be split into the review's four groups (closed-book correct/incorrect × sufficient/distractor). Runs with `select: all` are included for MuSiQue 2-hop and HotpotQA.
* **Single-hop evidence integration (point 6).** RAGBench sub-datasets are single-hop; in addition HotpotQA's `comparison` questions contain two independent facts that can be presented as one-hop increments. NQ/TriviaQA are not added at this stage because RAGBench already supplies single-hop questions with natural retrieval; they remain an optional later addition.

### Answer scoring on RAGBench

RAGBench stores a long-form reference `response` rather than a short gold answer. Correctness (needed for uptake, correction and stability labels) is therefore scored two ways: (i) token-F1 against the reference response with a threshold, and (ii) an LLM judge (the same model in a yes/no prompt: "Does the candidate answer agree with the reference?") on GPU. Sufficiency labels do not depend on this: they come from the annotated relevant sentences. Sub-datasets whose references are short (`pubmedqa` yes/no/maybe, `finqa`/`tatqa` numeric, `covidqa` spans) are scored with the existing normalised match.

## 2. Point 12: can the Evidence State Transition Benchmark be assembled from what we have?

Yes. Every axis the review lists maps to an existing resource or to one that is already on this machine.

| axis (review) | states | source | already available |
|---|---|---|---|
| evidence quality | C0 none, C1 irrelevant, C2 partial, C3 complete, C4 complete + redundant, C5 complete + contradictory, C6 answer string without supporting relation | the current trajectory builder produces exactly these seven (none, distractors-only, chain prefixes, minimal sufficient, sufficient + duplicate, sufficient + falsified hop, answer-string hatnote) for MuSiQue, 2Wiki, HotpotQA | yes |
| evidence quantity | 1 → 2 → 3 → 5 documents at fixed quality | chain prefixes plus post-sufficiency distractor additions; a `quantity` sweep adds 0/1/3 distractors to each quality state | yes (config) |
| linguistic form | verbatim → paraphrase → independently authored | verbatim: current duplicates. Paraphrase: generate with the local 7B model (or a stronger local model) and keep only paraphrases that preserve the sub-answer string and pass a round-trip check. Independently authored: HotpotQA and 2Wiki paragraphs for the *same* Wikipedia entity are written by different editors at different times; for entities that appear as gold in both datasets (title overlap) the two paragraphs are independent sources of the same relation. Synthetic templates give a third, fully controlled form | partly (paraphrase and cross-dataset alignment to be built) |
| retrieval source | gold → BM25 → dense → hybrid | the `wiki-18` passage corpus (18M Wikipedia passages, `PeterJinGo/wiki-18-corpus`, about 5 GB) and its E5 dense index (`PeterJinGo/wiki-18-e5-index`) are public; the shared cache on this host only holds empty stubs of them, so both must be downloaded to scratch (7 TB free). BM25 via `rank_bm25`/Pyserini over the same corpus. For every MuSiQue/2Wiki/HotpotQA question we retrieve top-k passages, label each passage gold/near-miss by title and sub-answer match, and build trajectories whose distractors are *retrieved* rather than benchmark-provided | public, to download; retrieval code to be written |
| domain | Wikipedia → biomedical → legal → technical → finance | RAGBench sub-datasets by domain | yes |
| question type | multi-hop bridge, comparison, single-hop | HotpotQA types, RAGBench | yes |
| closed-book status | correct vs incorrect | `select: all` | yes |

Two constraints follow from the assessment:

1. **The contradictory state (C5) needs a source for the false fact.** The current falsification swaps the sub-answer for another question's sub-answer of the same relation; for retrieved evidence, contradiction can also be *natural* (a retrieved passage that states a different value), which the passage labelling step detects when a passage contains a same-typed entity in the answer slot. Both kinds are kept, labelled `false_synthetic` and `false_natural`.
2. **Partial evidence (C2) is naturally defined for multi-hop questions (one hop present) but not for single-hop ones.** For RAGBench and synthetic single-hop facts, "partial" is a document that mentions the subject and the relation type but not the value (built by masking the value sentence), or a sub-relation.

### Construction plan

1. **Loaders** for HotpotQA and RAGBench producing `Question` objects (gold units, distractors, answer/aliases, sub-answers, relation labels, domain tag, closed-book status).
2. **Retrieval layer** (`revise/data/retrieval.py`): BM25 and E5 over `wiki-18`; for each question, top-20 passages with gold/near-miss/irrelevant labels; a trajectory family `retrieved` that draws distractors and near-misses from retrieval instead of from the benchmark pool.
3. **Paraphrase layer**: paraphrase every gold paragraph with the local model under a constraint that the sub-answer string is preserved; a second variant that also drops the answer sentence (partial). Store as new paragraph roles `paraphrase`, `partial`.
4. **Cross-dataset independent sources**: align HotpotQA/2Wiki/MuSiQue gold titles; for aligned entities emit `independent_redundant` increments (same relation, different author).
5. **Synthetic generator**: fictional entities, 1–3 hops, five template families per relation, contradictions, partials; used both as its own run and as the fully controlled sub-corpus of ESTB.
6. **Assembly**: a manifest that samples questions across datasets and domains, enumerates the seven quality states × quantity levels × linguistic form × retrieval source available for each question, and writes standard `conditions.jsonl`/`increments.jsonl` so every existing analysis runs unchanged. Balanced increments per cell; document-, answer- and relation-disjoint splits as before.
7. **Evaluation protocol**: train directions on MuSiQue only and evaluate on every ESTB cell (the "train on Wikipedia trajectories, evaluate on biomedical RAG without retraining" test the review asks for), then within-ESTB probes as the ceiling.

Estimated cost: HotpotQA and RAGBench collection about one GPU-hour each per model; retrieval over wiki-18 about two CPU-hours for BM25 plus index loading for E5; paraphrasing 2,000 questions × 2 paragraphs about half a GPU-hour; ESTB collection one to two GPU-hours per model. Total on the order of ten GPU-hours, comparable to the whole Phase 2.

### Order of execution

1. HotpotQA (Qwen) with `select: all`.
2. RAGBench (Qwen), all sub-datasets, sufficiency and judge-scored uptake.
3. Synthetic benchmark (Qwen).
4. Retrieval layer + retrieved-evidence trajectories for MuSiQue and HotpotQA.
5. Paraphrase and independent-source layers.
6. ESTB assembly and the MuSiQue-trained → ESTB evaluation; replicate on Llama and Mistral for the ESTB run only.
