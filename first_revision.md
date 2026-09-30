Yes. **The datasets are appropriate for the paper’s immediate question, but they are also one of the less novel parts of the experimental design.** More importantly, the choice of datasets creates a few threats to the paper's strongest claims about *evidence uptake* and *generalization*.

## 1. What the paper actually uses

The core setup is:

* **MuSiQue** as the primary benchmark, with 2–4-hop questions.
* **2WikiMultiHopQA** as the main cross-dataset transfer set.
* Three 7–8B instruction models: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3.
* Questions are filtered to cases where the models are **incorrect in the closed-book condition**, then evidence trajectories are constructed around those questions.
* The authors create controlled evidence conditions: distractor, individual reasoning hops, decisive hop, redundant support, falsified evidence, and answer-string-in-irrelevant-passage. 

The authors then report transfer to 2Wiki and 3–4-hop MuSiQue, including the notable degradation of uptake to about **0.61 AUROC** on document/answer-disjoint splits for 3–4-hop questions. 

So the dataset selection is not arbitrary. **MuSiQue is actually a sensible choice for constructing ordered evidence trajectories**, because its questions have explicit multi-hop structure.

---

# 2. Why MuSiQue is a good choice

I would give the authors credit here.

MuSiQue is much better suited than ordinary NQ/TriviaQA-style datasets for their central experiment because the paper needs something like:

$$
C_0 \rightarrow C_1 \rightarrow C_2 \rightarrow C_{\mathrm{sufficient}}
$$

where each increment has a semantic interpretation.

A 2–4-hop question naturally provides:

> hop 1 → hop 2 → hop 3 → hop 4.

That makes it possible to ask whether the residual state changes when the **missing piece of the reasoning chain** is supplied.

This is exactly the kind of controlled intervention the paper needs.

The broader RAG literature continues to use MuSiQue alongside HotpotQA and 2WikiMultiHopQA for multi-hop retrieval evaluation, so it is also a conventional and defensible benchmark rather than an obscure choice. ([DOI][1])

---

# 3. But there is a serious problem: MuSiQue is not really a natural RAG dataset

This is the first criticism I would put into an ACL review.

The paper is making claims about:

> **retrieval-augmented language models and evidence uptake**

but its core evidence trajectories are **researcher-constructed from benchmark gold evidence**, rather than naturally retrieved evidence.

That is useful experimentally—but it means the paper is closer to:

> **controlled evidence integration in multi-hop QA**

than to ordinary RAG.

This distinction matters.

A real RAG pipeline has:

$$
q \rightarrow \text{retriever} \rightarrow
\{d_1,\ldots,d_k\} \rightarrow LLM.
$$

The paper instead largely studies:

$$
q \rightarrow
\text{carefully selected evidence trajectory}
\rightarrow LLM.
$$

The latter is *much better for causal identification*, but weaker for ecological validity.

That is actually a reasonable tradeoff—but the paper should acknowledge it more explicitly.

---

# 4. 2WikiMultiHopQA is useful, but not a genuinely independent type of evidence

The paper's cross-dataset validation on 2Wiki is valuable. It obtains roughly:

* **0.956 sufficiency AUROC**
* **0.791 uptake AUROC**

for the Qwen transfer experiment. 

But 2WikiMultiHopQA is still another **Wikipedia-derived multi-hop QA benchmark**.

So the cross-dataset transfer answers:

> “Does this representation generalize from one multi-hop Wikipedia-style benchmark to another?”

It does **not** really answer:

> “Does this representation generalize to different RAG domains, document distributions, retrieval noise, or user questions?”

This is particularly important because the paper emphasizes transferability as one of its contributions.

---

# 5. The most interesting missing dataset: HotpotQA

I think **HotpotQA should almost certainly have been included**.

The authors use MuSiQue and 2Wiki, but HotpotQA is arguably the most obvious third benchmark for this particular experiment.

Why?

Because HotpotQA gives another multi-hop setting with supporting documents and distractors. The broader multi-hop literature routinely evaluates HotpotQA, MuSiQue and 2Wiki together. ([DOI][1])

More importantly, HotpotQA would let the authors test whether the discovered “sufficiency” direction is genuinely about:

> **evidence becoming sufficient**

rather than about the particular structure of MuSiQue.

I would therefore have liked to see:

**MuSiQue → 2Wiki → HotpotQA**

rather than only:

**MuSiQue → 2Wiki.**

---

# 6. More importantly: they should have included a *single-hop* dataset

This is probably the biggest experimental omission.

The paper is trying to distinguish:

> **sufficiency**

from

> **uptake**

and argues that the latter is a deeper, internally represented state.

But almost all of the central experiments involve **multi-hop reasoning**.

That leaves an important ambiguity:

### Is “uptake” really evidence integration?

Or is it partly:

> “the model has completed a multi-step reasoning pattern”?

A single-hop RAG dataset could help separate those possibilities.

For example:

$$
\text{question} + \text{one relevant document}
$$

versus

$$
\text{question} + \text{one irrelevant document}.
$$

If the uptake representation survives there, that is much stronger evidence that it represents **evidence integration itself**, rather than multi-hop completion.

A benchmark such as **Natural Questions** would provide a useful complementary setting. Current RAG evaluations routinely combine NQ with HotpotQA, MuSiQue and 2Wiki. ([NeurIPS Proceedings][2])

---

# 7. An even better choice: a dataset with *retrieval noise naturally present*

The paper's controlled distractors are useful, but there is a methodological concern.

The authors report that distractor conditions lead to extremely high abstention—roughly 94–97%—which demonstrates that the models can distinguish their constructed sufficient/insufficient conditions. 

But the distractors are **experimentally designed**.

Real retrieval produces:

* semantically similar near-misses,
* partially relevant passages,
* duplicate facts,
* conflicting sources,
* incomplete chains,
* irrelevant but lexically similar passages,
* documents containing the answer string but not the answer relation.

The paper does include several of these conditions, which is good. But a benchmark built from **actual retrieval outputs** would make the evidence-uptake claim much stronger.

For example, RAG-QA-style benchmark constructions explicitly combine golden documents with retrieved noise documents from a large Wikipedia pool. ([GitHub][3])

That would let the authors test whether the same latent state appears under **naturally generated retrieval distributions**, rather than only their hand-designed trajectories.

---

# 8. RAGBench would have been an interesting complementary dataset

**RAGBench** is particularly relevant because it is explicitly designed for RAG evaluation and contains around **100K examples across five industry domains**, with labels concerning the quality and grounding of RAG outputs. ([arXiv][4])

I would *not* replace MuSiQue with RAGBench.

Instead:

### MuSiQue

for controlled mechanistic experiments.

### RAGBench

for external validity.

That would give the paper a nice division:

> **controlled laboratory setting → realistic RAG setting**

The authors could ask whether a probe learned from controlled evidence trajectories predicts uptake/grounding in naturally occurring RAG contexts.

That would be substantially more convincing than another Wikipedia benchmark alone.

---

# 9. A domain-specific dataset would have been even more valuable

The current datasets are overwhelmingly general-domain Wikipedia QA.

This creates a potential confound:

> perhaps the probe is learning a generic “Wikipedia multi-hop answerability” state.

A much stronger experiment would involve a domain with genuinely external knowledge—for example:

* biomedical RAG,
* legal RAG,
* technical/manual RAG,
* financial/document QA.

RAGBench explicitly contains industry-specific domains, making this kind of evaluation possible. ([arXiv][4])

A particularly interesting test would be:

**train probe on Wikipedia evidence trajectories → evaluate on biomedical/technical RAG**

without retraining the probe.

If uptake remained detectable, that would dramatically strengthen the claim that the paper found a **general evidence-integration state** rather than a benchmark-specific feature.

---

# 10. There is another issue: filtering only closed-book failures

This is already acknowledged as a limitation by the paper, and I think it is important.

The authors select questions where the model is **wrong without evidence**. 

This is sensible if the goal is to study evidence correcting a model.

But it creates a selection effect:

$$
D = \{q: \text{model fails closed-book}\}.
$$

Therefore the probe is learned on:

> questions where external evidence is particularly necessary.

That is not the same as:

$$
D_{\mathrm{all}} =
\{\text{questions the model may encounter in RAG}\}.
$$

This could inflate the apparent regularity of the sufficiency state.

The authors should ideally have included **four groups**:

| Closed-book | Evidence   | Purpose                        |
| ----------- | ---------- | ------------------------------ |
| Correct     | Sufficient | ordinary evidence confirmation |
| Correct     | Distractor | test unnecessary retrieval     |
| Incorrect   | Sufficient | evidence correction            |
| Incorrect   | Distractor | genuine insufficiency          |

That would allow them to distinguish:

> **evidence sufficiency**

from

> **need for evidence**

from

> **correction of an existing belief**.

The current setup is much more heavily weighted toward the last two.

---

# 11. The redundancy manipulation is also too weak

The authors themselves acknowledge this limitation: redundant evidence is essentially **verbatim duplication**. 

This is important for the dataset question.

If the paper wants to argue that the residual state represents evidence sufficiency rather than superficial textual properties, stronger redundancy tests would be:

### A. Paraphrastic redundancy

Same fact, completely different wording.

### B. Independent-source redundancy

Two independently written passages supporting the same relation.

### C. Contradictory redundancy

Two sources supporting opposite relations.

### D. Partial redundancy

A second document supplies only part of the already-known relation.

These would make the “uptake” construct much more convincing.

---

# 12. The most interesting dataset the authors could have created themselves

Ironically, I think the paper's biggest opportunity was **not another existing benchmark**.

They already have the ingredients for a much more novel benchmark.

I would construct an **Evidence State Transition Benchmark** with controlled states such as:

$$
C_0 = \text{no evidence}
$$

$$
C_1 = \text{irrelevant evidence}
$$

$$
C_2 = \text{partial evidence}
$$

$$
C_3 = \text{complete evidence}
$$

$$
C_4 = \text{complete + redundant}
$$

$$
C_5 = \text{complete + contradictory}
$$

$$
C_6 = \text{answer string but no supporting relation}.
$$

Then independently vary:

### Evidence quantity

1 → 2 → 3 → 5 documents

### Evidence quality

irrelevant → partial → sufficient → contradictory

### Linguistic form

verbatim → paraphrase → independently authored

### Retrieval source

gold → BM25 → dense retrieval → hybrid retrieval

### Domain

Wikipedia → biomedical → legal → technical.

That would turn the dataset itself into a **general-purpose causal testbed for evidence integration**.

The present paper is halfway there, but stops at benchmark-derived multi-hop trajectories.

---

# 13. There is actually a very relevant 2026 precedent

A July 2026 ACL paper, **S2G-RAG**, explicitly introduces a controller that predicts whether the current evidence is sufficient for answering and identifies the missing information for the next retrieval step. It evaluates on TriviaQA, HotpotQA and 2WikiMultiHopQA. ([ACL Anthology][5])

This is interesting for the attached paper because it demonstrates that **evidence sufficiency itself is now becoming an explicit RAG control variable**, rather than merely a latent interpretability target.

So the attached paper's choice of MuSiQue/2Wiki is increasingly easy to motivate as:

> good datasets for studying the phenomenon,

but less easy to motivate as:

> novel datasets for establishing that the phenomenon matters to RAG.

---

# 14. What I would have wanted experimentally

If I were reviewing this for ACL, my preferred dataset matrix would be:

| Dataset                            | Role                                 |
| ---------------------------------- | ------------------------------------ |
| **MuSiQue**                        | Main controlled multi-hop trajectory |
| **HotpotQA**                       | Independent multi-hop replication    |
| **2WikiMultiHopQA**                | Cross-dataset transfer               |
| **NQ / TriviaQA**                  | Single-hop evidence integration      |
| **RAGBench**                       | Realistic domain-specific RAG        |
| **Naturally retrieved corpus**     | Retrieval-noise validity             |
| **Synthetic controlled benchmark** | Clean causal trajectory              |

That would test six different axes:

$$
\text{hop complexity}
\times
\text{retrieval realism}
\times
\text{domain}
\times
\text{question type}
\times
\text{evidence quality}
\times
\text{source independence}.
$$

The current paper mostly tests the first and partially the fifth.

---

# 15. Does this weaken the paper?

**Somewhat, but not fatally.**

I would distinguish two questions.

### Are the chosen datasets appropriate for the claimed *mechanistic experiment*?

**Yes.**

MuSiQue is actually quite well suited to the controlled incremental evidence design. The use of 2Wiki and 3–4-hop questions provides meaningful transfer tests.

### Are they sufficient to support the broader claim about *RAG evidence integration in general*?

**Not quite.**

The main weaknesses are:

1. predominantly Wikipedia-style multi-hop QA;
2. limited single-hop evaluation;
3. artificially constructed rather than naturally retrieved evidence;
4. closed-book-failure selection;
5. weak redundancy manipulation;
6. no genuinely domain-shifted RAG evaluation.

And the paper itself already acknowledges several of these limitations, especially the closed-book selection, 3–4-hop uptake degradation, small uptake test sets, and verbatim redundancy. 

### My revised reviewer view

I would **not penalize the paper heavily for choosing MuSiQue/2Wiki**. They are sensible experimental substrates.

I **would penalize it for not making the scope of the dataset evidence sufficiently explicit**. In particular, the paper should be careful not to generalize:

> “we have identified the internal evidence-uptake state of RAG LLMs”

when the evidence is more accurately:

> “we identify a highly decodable state associated with evidence uptake in controlled multi-hop QA trajectories.”

That distinction becomes especially important given the paper's own finding that uptake falls to about **0.61 AUROC** on the harder 3–4-hop document/answer splits. 

**If I were revising the ACL review, I would make dataset design one of the main weaknesses, but not a reason by itself to reject.** The strongest requested addition would be **HotpotQA + a single-hop RAG dataset + naturally retrieved/noisy evidence**, rather than simply adding another multi-hop benchmark.

[1]: https://doi.org/10.1145/3789506?utm_source=chatgpt.com "Retrieval-Augmented Generation for Multi-Hop Question Answering Based on Structured Planning | ACM Transactions on Knowledge Discovery from Data"
[2]: https://proceedings.neurips.cc/paper_files/paper/2025/file/daafafb5cf0229f0d1e01db3d72c2942-Paper-Conference.pdf?utm_source=chatgpt.com "Cooperative Retrieval-Augmented Generation for"
[3]: https://github.com/AQ-MedAI/RagQALeaderboard?utm_source=chatgpt.com "GitHub - AQ-MedAI/RagQALeaderboard: RAG-QA Leaderboard · GitHub"
[4]: https://arxiv.org/abs/2407.11005?utm_source=chatgpt.com "RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems"
[5]: https://aclanthology.org/2026.acl-long.1185/?utm_source=chatgpt.com "S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA - ACL Anthology"

