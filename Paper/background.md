# Background: concepts and their implementation in REVISE

This document explains, in detail, the concepts used in the REVISE experiments and how each one is implemented in the codebase (`revise/`). It is written for a reader who knows machine learning and language models but has not followed the project, and it is meant to be read alongside the paper draft in `Paper/`.

The project ran in three phases. Phase 1–2 (MuSiQue, 2WikiMultiHopQA, three model families) established the readout and causal results. Phase 3, the response to the first review (`first_revision.md`), added three external sources (HotpotQA, RAGBench, a synthetic benchmark), the **Evidence State Transition Benchmark (ESTB)** with retrieved and rewritten evidence, per-group analyses, LLM-judge scoring, an extended adaptive-retrieval study and the replication of the external and ESTB runs on Llama and Mistral. Sections 2.4, 3.4, 4.6, 5.8, 7 and 8 describe the Phase 3 additions.

---

## 1. The research question and its decomposition

The plan (`RESEARCH_PLAN.md`) asks whether, as retrieved evidence becomes sufficient to answer a question, an LLM undergoes an internal state transition that predicts successful knowledge accommodation before it generates an answer. The plan itself warns that "answer update" is not one thing, and separates four objects:

| Object | Meaning | How it is labelled here |
|---|---|---|
| Evidence sufficiency | the context now contains enough to answer | from the construction: all gold hops present |
| Evidence uptake | the model has internally incorporated the decisive fact | behavioural: the decisive increment makes the answer correct |
| Answer revision | the output changes | behavioural: answers differ |
| Correctness transition | wrong to correct | behavioural |

The experiments therefore never train a single "wrong-to-correct" probe. They train separate probes for sufficiency, uptake, correction, stability and revision, and ask whether one direction predicts all of them (it does not).

A second question, orthogonal to the first, runs through the whole project: **is a direction that *reads* a state also the direction that *controls* it?** Section 6 explains why the answer turned out to be no.

---

## 2. Data: multi-hop QA as a source of controlled evidence

### 2.1 Why multi-hop datasets

A two-hop question such as "Who is the spouse of the performer of *Green*?" requires two facts: *Green* was performed by Steve Hillage (hop 1), and Steve Hillage's spouse is Miquette Giraudy (hop 2). MuSiQue supplies, for every question, the gold paragraph for each hop, a decomposition with the sub-answer of each hop and the relation used (`performer`, `spouse`), and eighteen distractor paragraphs. This gives a principled notion of "sufficient": a context is sufficient if and only if it contains every gold hop paragraph. 2WikiMultiHopQA supplies the same in a slightly different format (evidence triples instead of decompositions) and is used as an out-of-distribution transfer set. Phase 3 added three sources to test whether the findings depend on how MuSiQue was built (section 2.4).

### 2.2 Loaders (`revise/data/loaders.py`)

Each dataset is converted into a `Question` object (`revise/data/schema.py`) with: the ordered gold paragraphs (`hop = 0, 1, ...` in reasoning order, taken from the decomposition order for MuSiQue and from the evidence-chain order for 2Wiki), the distractors, the answer and its aliases, the per-hop sub-answers (`decomposition`), the relation types (the text after `>>` in a MuSiQue sub-question, or the relation of a 2Wiki triple) and the bridge entities. A normalised JSONL copy is cached under the scratch data directory so later stages never touch the Hub.

### 2.3 Question selection

For each model, 4,000 questions are screened closed-book (question only, no evidence). Only questions the model gets wrong are kept (2,000 per model). This guarantees that every evidence increment concerns knowledge the model needs, so a correct answer after evidence is attributable to the evidence. Closed-book accuracy was 1.4% (Qwen), 4.0% (Llama) and 1.3% (3–4-hop), so the filter removes little but matters conceptually.

The review asked whether the readout is merely a "the model needs evidence" signal. The external runs therefore use `data.select: all`, which keeps closed-book-correct questions too and records each question's closed-book status in `screen.jsonl`, so every result can be split by closed-book status (section 5.8).

### 2.4 External sources (Phase 3)

| source | what it adds | gold unit | loader |
|---|---|---|---|
| **HotpotQA** (distractor setting) | an independently built multi-hop dataset; *bridge* and *comparison* questions | the two supporting-fact paragraphs, ordered by first supporting sentence; for bridge questions the paragraph whose title appears in the question is hop 0 and the other title is the bridge entity | `load_hotpotqa` |
| **RAGBench** (12 sub-datasets) | naturally retrieved documents from real RAG pipelines, mostly single-hop, five domains (biomedical `pubmedqa`/`covidqa`, legal `cuad`, technical `techqa`/`emanual`, finance `finqa`/`tatqa`, customer support `delucionqa`, general `msmarco`/`hagrid`/`expertqa`, Wikipedia `hotpotqa`) | a retrieved document with at least one annotated relevant sentence (`all_relevant_sentence_keys`); the other retrieved documents of the same example are the *natural* distractors; questions with one or two gold documents are kept | `load_ragbench` |
| **Synthetic** (`revise/data/synthetic.py`) | fictional entities, so closed-book failure holds by construction; full control of quality and linguistic form | templated facts over seven relations (founder, birthplace, capital, spouse, award, employer, headquarters), 1–3 hops, five template families per relation; each fact also has a paraphrase, a *partial* (subject and relation, value withheld) and a *contradictory* (different value) variant; distractors are near misses (same relation, other subject; same subject, other relation) plus filler | `load_synthetic` |

HotpotQA has no decomposition, so the sub-answer of hop 0 of a bridge question is the bridge title, and comparison questions have two independent facts that both lead to the answer. RAGBench gives a long-form reference `response` instead of a short answer, which changes how correctness is scored (section 4.6). Its "hops" are the number of gold documents, not reasoning steps.

---

## 3. Evidence trajectories (`revise/data/trajectories.py`)

### 3.1 Conditions

A **condition** is one ordered list of paragraphs shown to the model. For gold hops g₀…gₙ₋₁ the builder creates:

* `C0` (no evidence) and `CD` (n distractors, each length-matched in word count to one gold paragraph).
* **Prefix chains** in the canonical hop order, the reverse order and, for n ≥ 3, random permutations: `C[g0]`, `C[g0+g1]`, …; the last prefix is the *minimal sufficient* context. When a new paragraph is added it is inserted at a **uniformly random position** in the previous presentation order. This is the key design choice: increments are textually minimal (one paragraph added), yet the absolute position of the decisive paragraph is randomised, so no direction can encode recency or position.
* **Leave-one-out** contexts for n ≥ 3 (all golds but one) and their completions. For two-hop questions these coincide with the chain prefixes and are de-duplicated.
* **Post-sufficiency** contexts: sufficient + distractor, sufficient + a verbatim duplicate of a gold paragraph ("redundant"), sufficient + a conflicting falsified final hop, and the full 20-paragraph context shuffled.

Conditions are identified by their ordered paragraph ids (`C[g1+d3+g0]`), so identical contexts produced by different routes are stored once.

### 3.2 Matched increments

An **increment** is an ordered pair of conditions differing by the added paragraph(s). The central methodological idea of the plan is that a decisive increment must be compared with *matched controls from the same starting state*, otherwise a probe can succeed by detecting length, the answer string or passage relevance. From every penultimate state S (all hops but the last one added) the builder emits:

| increment | what is added | crosses sufficiency |
|---|---|---|
| `decisive` | the missing gold hop | yes |
| `distractor` | a distractor length-matched to that gold hop | no |
| `redundant` | a duplicate of a gold paragraph already in S | no |
| `false` | the missing hop with its sub-answer replaced by another question's sub-answer of the same relation type | no |
| `answer_ctrl` | a distractor with "(Not to be confused with {answer}.)" appended, so the answer string is present in an irrelevant context | no |

All five share a `control_group` id. Additional kinds: `gold_nondecisive` (adding a hop that does not complete the evidence), `distractor_from_none`, `post_distractor`, `post_redundant`, `post_false`, `full_context`, `loo_decisive`.

**Falsification** (`falsify`) replaces, case-insensitively, the hop's sub-answer in the paragraph text and title with a false answer sampled from the pool of sub-answers seen for the same relation type across the dataset, so the false paragraph is type-plausible. The true and false strings are stored in the paragraph's `meta` so that "false-answer following" can be measured later.

### 3.3 Why keep the whole trajectory

Stopping when the answer first becomes correct would condition the dataset on the model's behaviour and remove the transitions the plan cares about (correct→correct under redundant evidence, correct→wrong under distraction, wrong→same wrong under ignored evidence). Every question is therefore run through every condition.

---

### 3.4 Linguistic-form and quality variants (Phase 3)

The builder also accepts optional per-hop *variants* stored in `Question.extra["variants"]` (`{"g0": {"paraphrase": ..., "partial": ..., "independent": ...}}`). When present, it emits extra increments from the same penultimate state S and after sufficiency:

| increment | what is added | crosses sufficiency |
|---|---|---|
| `decisive_paraphrase` | the missing hop, reworded | yes |
| `decisive_independent` | the missing hop as written in another dataset | yes |
| `partial` | the missing hop rewritten to withhold its value | no |
| `post_redundant_paraphrase` / `post_redundant_independent` | a reworded or independently written copy of a hop already present, after sufficiency | no |

`DECISIVE_KINDS` includes the paraphrase and independent kinds, so they are positives for sufficiency and uptake. Variants come from the synthetic generator (by construction) or, for MuSiQue, from the `variants` stage (section 7.4).

---

## 4. Prompts, the anchor position and the model backend

### 4.1 Prompts (`revise/prompts.py`)

Every stateless stage uses one template so that the only difference between stages is the evidence block:

```
Use only the supplied evidence to answer the question. Do not rely on any other knowledge.
If the evidence is insufficient to determine the answer, output exactly INSUFFICIENT.
Respond with only the short answer, no explanation.

Evidence:
[1] Title: ...
...

Question:
...

Answer:
```

The message is wrapped in the model's chat template. Two instruction regimes exist: **evidence-only** (above) and **evidence-augmented** ("together with your own knowledge"), the latter closer to real RAG but introducing context–memory arbitration. The **conversational-revision** protocol (secondary, an ablation) is a two-turn chat: the stateless prompt for Cₖ, the model's own stateless answer Aₖ as the assistant turn, and a second user turn "Your previous answer was Aₖ. New evidence: Eₖ₊₁. Answer the question again. Change your answer only if the total evidence warrants it."

### 4.2 The anchor

Activations are read at the **last prompt token**, the position whose next-token distribution produces the first answer token (for Qwen this is the newline after `<|im_start|>assistant`). Reading here has two advantages: the position is comparable across contexts of different length, and everything read is strictly *pre-generation*, so a probe cannot be reading the answer back off the generated tokens.

### 4.3 Backend (`revise/model/backend.py`)

The backend is Hugging Face Transformers on CPU or CUDA with forward hooks; llama.cpp was rejected because steering and patching need read/write access to every layer. Per batch of prompts it does the following.

1. Left-pads the prompts so the anchor is the same index for every row, and appends the gold answer tokens *after* the anchor. Because attention is causal, the anchor's activations are unaffected, but the same forward pass yields the teacher-forced log-probability of the gold answer, a covariate used later.
2. Registers hooks on every decoder block (and the embedding layer) that copy the anchor row of the block output; index l in the stored array is the residual stream after block l (index 0 is the embeddings, and the last index is before the final norm).
3. Keeps logits only for the anchor and the appended answer positions (`logits_to_keep`), which is what makes long prompts fit on a shared GPU.
4. Crops the KV cache back to the prompt and decodes greedily up to 16 tokens, stopping at end-of-sequence or a newline.

The same hook mechanism implements interventions (`Intervention` objects): `steer` adds a vector at the anchor and every generated position (or at all positions), `patch` replaces the anchor residual during the prefill, and `project` sets the component along a unit vector to a target value. Correctness of the hooks was verified against `output_hidden_states` and by checking that a patched anchor is read back exactly.

### 4.4 Batching on GPU

Batches are formed by sorting conditions by length and capping each batch by a token budget (`REVISE_TOKEN_BUDGET`, default 48k tokens), so full-context prompts automatically get small batches. This was added after a CUDA out-of-memory on the longest prompts.

### 4.5 Answer scoring (`revise/eval.py`)

Predictions are normalised (lower-case, punctuation and articles removed). A prediction is *correct* if it exactly matches any alias, or (lenient mode, used throughout) contains the gold answer and is at most three times its length. `INSUFFICIENT` in any short form counts as abstention. Two answers are *equivalent* (no revision) if their normalised forms match or their token F1 is at least 0.8.

### 4.6 LLM-judge scoring and the fixed auxiliary model (Phase 3)

RAGBench references are long-form answers, so exact match or containment underestimates correctness. The `judge` stage (`revise/run/judge.py`) asks an instruction-tuned model, in a yes/no prompt, whether the candidate states the same final answer as the reference. It writes `behavior/<regime>.judged.jsonl`, keeping abstentions as `insufficient`, and `Run.behavior` prefers the judged file whenever it exists, so features, probes and reports use the judged labels without other changes. On Qwen the judge lowered RAGBench uptake from 0.754 to 0.685: lenient containment had been crediting partial overlaps. Sufficiency labels do not depend on this, because they come from the annotated relevant sentences.

The judge and the paraphrase generator (section 7.4) are *auxiliary* models. For the Llama/Mistral replication they must not be the model under study, or labels and evidence text would differ between families. A config key `aux_model` (used by `aux_be()` in `revise/cli.py`) fixes both to Qwen2.5-7B-Instruct for every family.

---

## 5. Readout: features, labels, splits and probes

### 5.1 Features (`revise/analysis/features.py`)

For each increment with both endpoints collected, Δh = h(Cₖ) − h(Cₖ₋₁) is stored for all layers (float16). The metadata row records the increment kind, whether it becomes sufficient, the model's answer state before and after (`correct`, `wrong`, `insufficient`), whether the answer was revised, and the **nuisance covariates**: change in prompt tokens and total tokens, change in gold log-probability, change in next-token entropy and top probability, whether the answer string occurs in the added text, in the full new context, or in the previous context, TF-IDF cosine relevance of the added paragraph to the question, number and word count of added paragraphs, relative insertion position, and the added text itself (for a TF-IDF-SVD embedding fitted on training data). Raw states h(C) are stored separately for *state probes*.

### 5.2 Tasks (`revise/analysis/probes.py`)

| task | positives | negatives |
|---|---|---|
| sufficiency | decisive and leave-one-out completions | every other increment kind |
| sufficiency (matched) | decisive | only the four matched controls from the same penultimate state |
| uptake | decisive increments after which the answer is correct | decisive increments after which it is not |
| correction | increments from a wrong or abstaining state that end correct | same start, not correct after |
| stability | increments from a correct state that stay correct | correct → not correct |
| revision | answer changed | not changed |
| state sufficiency / state correct / abstention flag | raw activation of a condition, labelled by sufficiency, correctness, INSUFFICIENT |

### 5.3 Splits (`revise/analysis/splits.py`)

Four groupings, each guaranteeing that train and test share nothing of that type: question id; **document** (union-find over questions sharing any gold Wikipedia title, then whole components assigned to one side; a component larger than the test target is forced to train, which was necessary for 3–4-hop questions where most questions share a document); **answer entity** (normalised answer); **relation** (the tuple of relation types). Test fraction 0.3. On RAGBench the documents are pipeline-specific and the connected components of the document split span almost the whole pool (2,999 train questions and 1 test question), so RAGBench analyses use the relation split, whose groups are the twelve sub-datasets and so test domain transfer, or the question split. The synthetic benchmark has no shared documents and uses the question and relation splits.

### 5.4 Probe training and evaluation

Per task, per split, per layer: features are standardised on the training split and an L2-regularised logistic regression (C = 0.5, balanced classes) is fitted. The reported layer is chosen by 3-fold **grouped** cross-validation on training questions only (never on the test set), and the number reported is the held-out AUROC at that layer. The direction is the weight vector divided by the feature standard deviations, normalised. The **difference-of-means** direction (class-mean difference, normalised) is fitted alongside and evaluated by projection. 95% confidence intervals come from a question-level bootstrap of the test set (`revise/analysis/bootstrap.py`).

### 5.5 Nuisance analysis

Three quantities answer "is the probe reading something trivial?":

* **nuisance-only AUROC**: logistic regression on the covariates alone (standardised, plus the 16-dim passage embedding);
* **residualised probe**: ridge-regress Δh on the covariates (training data), probe the residual;
* **gain beyond nuisance**: AUROC of a classifier on [covariates, out-of-fold probe score] minus the nuisance-only AUROC. Out-of-fold scores (5-fold, grouped by question) prevent leakage.

### 5.6 Transfer

`apply_directions` scores a stored direction on another feature set without retraining: stateless → conversational (Δ defined as h_conv(Cₖ→Cₖ₊₁) − h_stateless(Cₖ)), MuSiQue → 2Wiki, and 2-hop → 3–4-hop. Cross-model transfer of raw vectors is impossible (hidden sizes differ), so model-family transfer is replication of the whole procedure. `transfer_matrix` reports the AUROC of each task's direction on every other task and the cosine between directions.

In Phase 3 the MuSiQue 2-hop directions are also applied to HotpotQA, RAGBench, the synthetic data and every ESTB run (`revise.cli transfer --transfer-from <source config>`). The ESTB runs sample from the same MuSiQue pool as the source run (818 of 1,200 questions overlap with the source's *training* split), so `revise/analysis/estb.py` drops those questions before scoring a source direction. Without this, transfer is inflated: on the Llama E5-distractor run the unfiltered uptake transfer is 0.955, while Qwen's filtered figures are 0.76–0.82. For the replication, the source is always the same family's MuSiQue run.

### 5.7 Text-only baselines (`revise/analysis/text_baseline.py`)

The reviewer question "would a text classifier do as well?" is answered by training, on exactly the same increments, labels and splits, a TF-IDF logistic regression and a fine-tuned DeBERTa-v3-base cross-encoder, each on "question [SEP] added paragraph" and on that plus the previous context. Training positives are kept and negatives subsampled in the reduced-budget version.

### 5.8 Per-group analysis (`revise/analysis/groups.py`)

The stored best-layer probe of a run, trained once on all questions, is scored separately on test increments grouped by closed-book status (from `screen.jsonl`), question type (HotpotQA bridge/comparison), domain (RAGBench), and number of hops or gold documents. No refitting is involved. This answers two review points. First, uptake is not a multi-hop artefact: single-hop RAGBench gives 0.70, HotpotQA comparison questions 0.79, single-hop synthetic 0.99. Second, the readout is not a "need for evidence" signal: HotpotQA sufficiency is 0.978 on closed-book-incorrect and 0.984 on closed-book-correct questions.

### 5.9 What the probes read: state vs update, evidence vs output (added 2026-10-03)

Three analyses answer the reviewer question "how much of the 0.95 is the *change* of state?" and the alternative that "sufficiency" is just the model's planned INSUFFICIENT output.

* **Representation ablation** (`revise/analysis/representation_ablation.py`). Every update probe is refitted on h(C_{k-1}), h(C_k), Δh and [h(C_{k-1}); h(C_k)] for the same increments, labels and splits.
  * h(C_k) gives 0.93–0.94 against 0.95 for Δh on sufficiency, and 0.90 against 0.91 on matched.
  * h(C_{k-1}) gives exactly 0.733, the AUROC of an oracle that knows only the starting stage, and exactly 0.500 on matched, where the starting state is shared.
  * The concatenation equals Δh.
  * For uptake, correction, stability and revision, h(C_k) is as good as or better than Δh.
  * The stored state-sufficiency probe applied to h(C_k) gives only 0.70 on the full update task, because 27% of the negatives end in an already sufficient context.
  * Conclusion: the information is in the resulting state. The paired update is a controlled way to read it, not its source.
* **Output held fixed** (`revise/analysis/behaviour_control.py`). Sufficiency probes are scored, and retrained, only on increments after which the model still says INSUFFICIENT, giving Δh 0.935–0.949. The output indicator alone gives 0.65–0.76. So the readout is not the planned output.
* **Verbalised vs internal sufficiency** (`revise/analysis/verbal_vs_internal.py`). The model's answer/INSUFFICIENT output is a sufficiency judgement, and it rejects 33–43% of sufficient contexts.
  * The state-sufficiency probe's 0.99 is partly paragraph count: every context with three or more paragraphs is sufficient.
  * Count-matched (two-paragraph contexts, probe retrained) it is 0.976–0.986, against 0.71–0.77 for the verbalised judgement, and 0.967–0.982 among contexts the model rejects.
  * At 0.95 precision the probe recovers 76–81% of wrongly rejected contexts with 3–4% false alarms.
  * Wrongly rejected contexts score 0.80–0.87 on a 0 (insufficient) to 1 (sufficient) scale, against 1.03 for answered ones, so the internal state is graded and the verbal decision is a noisier thresholded read-out.
* Tables come from `Paper/figures/make_state_tables.py`: `rep_ablation.tex`, `behaviour_control.tex`, `verbal_internal.tex`. The write-up is in Appendix C (behaviour), Appendix D (state vs update) and the readout subsection "What the probes read".

### 5.10 Self-verbalisation and confidence baselines (added 2026-10-06)

`revise/analysis/self_verbal_baseline.py` scores, on exactly the test increments of the probes and text baselines (document split), the model's own output as a classifier: "answers rather than INSUFFICIENT at C_k", the switch from INSUFFICIENT to an answer, and output confidence (top-token probability, negative entropy, and their change over the increment). `Paper/figures/make_baselines_table.py` builds `tables/baselines.tex` (text classifiers + verbalised + confidence + probe) and `tables/uptake_given_answer.tex`.

* Sufficiency: verbalised 0.65–0.69 (matched 0.67–0.73); confidence below 0.5 (INSUFFICIENT is a confident output); probe 0.95 / 0.91.
* Uptake: the "answers" indicator alone gives 0.78, the same as the probe, because 45–55% of non-uptake decisive increments are abstentions. Among answered decisive increments the probe separates correct from wrong at 0.63–0.68 vs 0.63–0.64 for confidence; combined with the decision to answer, 0.84–0.86 vs 0.84. So the uptake probe largely reads the decision to answer and the correctness-specific part is modest. The paper now says this plainly (§5.2, Appendix C "Uptake and the decision to answer", Limitations).
* Correction and stability are predicted by "answers" at 0.88–0.90 / 0.80–0.90, above the probes, which is one reason they moved to an appendix.

Paper layout from 2026-10-06: the ACL-style sources live in `Paper_acl/` (times font and acl.sty untouched) and are zipped as `new_revision.zip`; `Paper/` keeps the generation scripts. Main-text order after revision-2: intro, related work, setup (data, trajectories, tasks table, splits, baselines), method (probes, nuisance, generalisation, causal tests), readout + baselines, causal (trimmed), generalisation, conclusion, limitations; discussion dropped; layer depth, projections by kind, correction/stability/revision, subspace rank and adaptive retrieval moved to appendices.

---

## 6. Causal validation: readout versus mechanism

Linear separability shows that a state is *readable*. It does not show that the direction is what the network *uses*. Three families of intervention test this (`revise/run/steer.py`, `revise/run/patch.py`, `revise/run/ablate.py`), all on held-out questions of the document split.

### 6.1 Steering

Add α·u·v at layer l at the anchor and every generated position. The unit u is the separation between the sufficient-state mean and the insufficient-state mean of vᵀh on training conditions, so α = 1 moves an activation by one "sufficient minus insufficient" gap. A random unit direction with the same ℓ₂ magnitude is the control. Evaluation sets: minimal sufficient contexts (should gain correct answers), penultimate contexts (should not start answering), distractor-only contexts (should not hallucinate), penultimate + false final hop (should not follow the false answer), penultimate + answer-string control (should not copy the answer). The plan's own criterion is applied: a direction that makes the model accept any passage is a context-compliance direction, not accommodation.

An early version scaled α by the standard deviation of *increment* projections (about 0.12 on a residual norm of 61) and, unsurprisingly, showed nothing; this is documented in the paper appendix because it is the kind of implementation artefact reviewers rightly ask about.

### 6.2 Projection-out

Set vᵀh to its insufficient-state mean at every layer from the best layer onward (anchor and generated positions), on *sufficient* prompts. If v carries the sufficiency signal causally, the model should start abstaining. This is a stronger test than addition because it removes rather than adds.

### 6.3 Activation patching and subspace rank

Run the insufficient prompt Cₙ₋₁ with its anchor residual replaced, at every layer from the best layer to the last (single-layer patching was tried first and found inert), by:

* `full`: the sufficient run's residual h_suff;
* `parallel`: h_ins + proj_v(Δ), only the component of the update along the probe direction;
* `orthogonal`: h_ins + Δ − proj_v(Δ);
* `random`: the projection of Δ on a random direction;
* `distractor` / `other_q`: the projection along v of a matched distractor update or of another question's update (controls for "any projection along v");
* `pcaK`: h_ins + P_K Δ with P_K the top-K principal components of training decisive updates at that layer, K ∈ {1, 4, 16, 64, 256}, plus a random 256-dimensional subspace;
* `unembed_add` / `unembed_remove`: only the component along the gold answer's first-token unembedding direction, or everything but it.

Patching at the anchor at all later layers is equivalent to replacing that position's entire residual trajectory, which later positions then attend to during decoding.

### 6.4 What the tests showed, and why

The logistic directions were inert under all three interventions in all three models; the full anchor residual was causally sufficient and its effect lay entirely in the orthogonal complement; the causal subspace was high-rank and model-specific (dominated by the answer's unembedding direction on Llama, by top principal components on Mistral). The one causally active direction was the **difference of means**, which projection-out showed to be an abstention gate and steering showed to be indiscriminate. The interpretation offered in the paper: a discriminative probe minimises within-class variance, so it down-weights the high-variance "answering" axis that carries the mean difference; the state is readable along many directions, but the network's own computation runs through a few high-variance ones plus high-rank answer content.

In Phase 3 the patching and projection-out tests were repeated on HotpotQA, RAGBench, the synthetic data and every ESTB run (`patch: {modes: [multi]}`, `ablate: {experiments: [projout]}`, 150 held-out questions each). The dissociation reproduced everywhere: the full patch makes the insufficient prompt answer, the probe-direction component does nothing, and projecting out v_dm silences answering. On synthetic data the patched model answers 89% of the time but is never correct, because a fictional answer can only be read off evidence tokens the insufficient prompt lacks. The first Llama replication run (E5-retrieved distractors) shows the same pattern: 0.14 → 0.67 answered with the full patch, 0.15 with the probe component, 0.03 after projecting out v_dm.

---

## 7. The Evidence State Transition Benchmark (ESTB)

The review asked for a benchmark that varies evidence along several axes at once. ESTB is not a new dataset file but a *labelling scheme and evaluation protocol* applied to a family of runs that all produce the standard `conditions.jsonl` / `increments.jsonl`, so every existing analysis runs on them unchanged. The paper's appendix "The Evidence State Transition Benchmark" is the reference description; this section explains the implementation.

### 7.1 What it measures

The unit stays the paired update of one increment. ESTB labels each increment with a **quality state** (`QUALITY` in `revise/analysis/estb.py`):

| state | meaning | increment kinds |
|---|---|---|
| C0 | none | start of every trajectory |
| C1 | irrelevant | `distractor_from_none`, `distractor` |
| C2 | partial | `gold_nondecisive` (another hop missing), `partial` (value withheld), `redundant` from a penultimate state |
| C3 | complete | `decisive`, `loo_decisive`, `decisive_paraphrase`, `decisive_independent` |
| C4 | complete + redundant | `post_distractor`, `post_redundant`, `post_redundant_paraphrase`, `post_redundant_independent`, `full_context` |
| C5 | contradictory | `false` (substituted value, from the penultimate state), `post_false` |
| C6 | answer string, no relation | `answer_ctrl` |

It also records the other axes: retrieval source, linguistic form, domain, question type, hop count and closed-book status. A *cell* is the set of increments of one run sharing one value on one axis. Per cell, `cell_table` reports four things:

1. the run's own held-out probe AUROC (is the state decodable here at all?);
2. the AUROC of the MuSiQue direction applied without retraining, on questions the source never trained on (is it the *same* state?);
3. for each quality state, the AUROC separating its increments from the decisive ones (does this state look like completion?);
4. the patching and projection-out answered rates (does the causal profile carry over?).

### 7.2 Axes and the runs that realise them

Evidence quality is present in every run. Retrieval source comes from the four retrieval runs (`musique_2hop_<model>_{e5,bm25}_{noise,full}`, section 7.3). Linguistic form comes from the variants run (`musique_2hop_<model>_variants`, section 7.4) and the synthetic benchmark. Domain, question type, hop count and closed-book status come from the HotpotQA, RAGBench and synthetic runs and are scored by `groups.py`. The retrieval and variants runs use 1,200 MuSiQue 2-hop questions with `n_random_perms: 0`.

### 7.3 Retrieval-source axis (`revise/data/retrieval.py`)

* **Corpus.** wiki-18 (FlashRAG's December-2018 Wikipedia, 21M 100-word passages, `PeterJinGo/wiki-18-corpus`), parsed once into `retrieval/wiki18_corpus.npz` on scratch.
* **BM25.** `bm25s` over title + text with English stop-words and the Snowball stemmer; the index is built once and saved (`retrieval/bm25s_wiki18`).
* **E5.** The published FAISS flat inner-product index of `e5-base-v2` passage embeddings (`PeterJinGo/wiki-18-e5-index`, two parts concatenated into `e5_Flat.index`, memory-mapped, about 65 GB). Queries use the `query:` prefix, mean pooling and ℓ₂ normalisation. FAISS threads are capped at 32 because OpenBLAS fails with 256.
* **Multi-query retrieval.** A single question-level query rarely finds both hops. Each question therefore issues the question plus one query per hop, built from MuSiQue's decomposition with earlier sub-answers substituted for `#k`. The hit lists are merged round-robin by rank, de-duplicated and cut at k = 30. This imitates an iterative RAG system that solves earlier hops correctly, so gold coverage is an upper bound.
* **Labels.** `retrieved_gold` (gold page title and the hop's sub-answer present), `same_page` (gold page, sub-answer absent: partial evidence in the wild), `near_miss` (mentions a bridge entity or the final answer, not a gold page: a natural answer-string control), `irrelevant`. For E5: 8% gold, 14% same page, 29% near miss, 49% irrelevant. At least one gold passage is found for 85.5% of questions, both hops for 33.5% (BM25: 68.1% and 18.2%).
* **Modes.** `noise`: benchmark gold kept, every distractor replaced by the 18 hardest retrieved non-gold passages (same page, then near miss, then irrelevant, each by rank), from which the builder draws length-matched distractors as usual. `full`: each hop's gold is also replaced by its top `retrieved_gold` passage. Only questions with both hops retrieved are usable (402 for E5, 219 for BM25 on Qwen; 405 for E5 on Llama).
* Retrieval runs inside `collect` and writes `retrieved_<method>.jsonl` per run, resumably by question id. The replication chain seeds each new run with Qwen's file, so only questions Qwen never saw are retrieved again.

### 7.4 Linguistic-form axis (`revise/data/variants.py`, `revise/data/independent.py`)

* **Paraphrase / partial** (`variants` stage, GPU). For every gold hop, Qwen2.5-7B generates (greedy, ≤220 tokens):
  * a *paraphrase* that must keep the sub-answer verbatim;
  * a *partial* rewrite that keeps the subject and topic but withholds the value ("the detail is not recorded here").

  Validation:
  * A paraphrase must contain the sub-answer, be 0.5–2× the original length and share under 60% of its word 4-grams with the original.
  * A partial must contain neither the sub-answer, nor the final answer, nor any alias, must have at least 15 words, and must mention the subject.

  Yield: 2,180 of 2,400 hops (90.8%) get a paraphrase, 1,411 (58.8%) a partial. Stored in `variants.jsonl`, resumable, and reused across families.
* **Independently authored** (CPU). MuSiQue, HotpotQA and 2Wiki take paragraphs from different Wikipedia dumps and revisions. A title index over the *other* datasets' paragraphs finds, for a gold hop with title T and sub-answer a, a paragraph with the same normalised title that contains a and shares under 60% of its 4-grams with the MuSiQue text (least-overlapping one kept). This yields 228 hops (144 from HotpotQA, 84 from 2Wiki).
* After attaching variants, the stage rebuilds conditions and increments, so `collect` picks up the new kinds (section 3.4).

### 7.5 Results in brief (Qwen; `Paper/tables/estb.tex`, `estb_quality.tex`)

* **Sufficiency** stays at 0.91–0.95 within run under every evidence source. The MuSiQue direction transfers at 0.91–0.94 (sufficiency) and 0.76–0.82 (uptake) on unseen questions.
* **Retrieved distractors are harder controls than benchmark ones.** The transferred direction separates C1 from decisive increments at 0.906 (E5) against 0.945, and the run's own probe recovers most of the gap.
* **Wording and authorship do not matter.** Paraphrased and independently written decisive paragraphs score like the originals (0.929 and 0.918 vs 0.931 AUROC; mean projections 0.098, 0.096, 0.101).
* **Contradictory and masked-partial evidence are the hardest states** (0.76 and 0.81 own probe; 0.73 and 0.74 transferred). Partial evidence in the hop sense is trivial (1.00).
* **The causal profile reproduces in every cell.**

### 7.6 Model-family replication (in progress)

`scripts/chain_replication.sh` runs the 8 external/ESTB configs for Llama-3.1-8B and Mistral-7B (`configs/*_llama8b*.yaml`, `*_mistral7b*.yaml`, 16 runs). Per run, it chooses the GPU stages from the config (`screen variants collect` for variants runs, `screen collect judge` for RAGBench), uses the same family's MuSiQue run as the transfer source, and seeds Qwen's retrieval and paraphrase caches. The log is `runs/replication_chain.log`. `Paper/figures/make_estb_table.py` and `estb.py` still name the Qwen runs only and need a model argument once the runs finish.

---

## 8. Downstream uses

### 8.1 Adaptive retrieval (`revise/run/adaptive.py`)

The simulation walks each held-out question along its canonical trajectory, one retrieval round per stage: distractors → hop 1 → minimal sufficient → + distractor. Every stage's activations and answers are stored, so no new forward passes are needed. `run_adaptive` keeps the original protocol: stop when the state-sufficiency score exceeds a threshold at 0.97 training precision.

`run_adaptive_directions` (Phase 3) turns *every* learned direction into a stopping rule, writing `adaptive/directions.json`:

* **Signals.** State probes read h(C_k) at round k: sufficiency (logistic and v_dm), correctness, abstention flag. Update probes read the Δh of the increment that produced the round's context: sufficiency (logistic and v_dm), matched, uptake, correction, stability, revision. `_incoming_deltas` maps each stage to that increment.
* **Calibration.** The threshold *and sign* are chosen on training questions to maximise accuracy at the stopping stage (ties: fewer rounds). The other sign's reading is also reported, which gives stability's "stop when the answer changed" rule.
* **Baselines.** Never stop; oracle; behavioural (first non-INSUFFICIENT answer, which needs one generation per round); entropy threshold.
* **Statistics.** Bootstrap CI of accuracy, a paired bootstrap of the difference to the behavioural rule, and test-set threshold sweeps (accuracy vs mean rounds) for the trade-off figure.
* **Verification.** At the stopping round chosen by the state-sufficiency rule (and by the oracle), the AUROC of every signal for the correctness of the answer given there, plus selective accuracy on the top half. Output confidence (top-token probability, negative entropy) is the baseline.

Findings (`Paper/tables/adaptive.tex`, `adaptive_verify.tex`, `figures/adaptive_curves.pdf`):

* **Only sufficiency works as a stopping rule.** The update-sufficiency v_dm rule is the best learned rule on all three MuSiQue models (0.478 / 0.468 / 0.336) and matches the oracle (0.475 / 0.470 / 0.336) in about three rounds. The logistic sufficiency probes tie the behavioural rule on MuSiQue without any generation and beat it on HotpotQA (0.73 vs 0.65).
* **Uptake, correction and stability are poor stopping rules.** Stability read as "stop when the answer changed" never fires, and revision is the worst learned rule.
* **Uptake (or correction) is the best learned verifier once retrieval stops** in every setting: 0.672 vs 0.621 for top-token probability on MuSiQue Qwen, 0.847 vs 0.705 on 2Wiki. Keeping the top half by uptake gives higher accuracy than keeping the top half by output confidence in five of six settings.
* **Division of labour.** Sufficiency decides when to stop; uptake decides whether to trust the answer.
* **Uninformative sources.** 3–4-hop MuSiQue (no post-sufficiency stage) and RAGBench (short trajectories) cannot separate the policies.

### 8.2 Closed-book transfer (`revise/run/closedbook.py`)

Capture the closed-book activations of the screened pool and score every stored direction for predicting closed-book correctness and abstention, with a probe trained on closed-book data as the ceiling.

---

## 9. Engineering notes that affect interpretation

* **Numerical noise.** Everything runs in bfloat16. Greedy generations are stable across batch compositions, but gold log-probabilities can differ by up to about 0.5 nats between batch layouts, so that covariate is coarse.
* **Resumability.** Every stage is idempotent: behaviour rows and activation shards already present are skipped, which is why runs could be interrupted, topped up and continued.
* **Compute.** One 2-hop model (35k conditions) takes about 22 hours on 64 CPU cores or about one hour on a shared H200. GPUs on this host are shared through `gpuq` with a weekly budget; the whole set of causal and baseline experiments used about 7 GPU-hours. Analysis stages (features, probes, transfer, adaptive, report) run on CPU.
* **GPU queue.** `gpuq submit --devices N` only stacks onto a card the user already holds and is refused otherwise. A card of one's own is `-g 1 -m <GB>` (`--queue` and `-t` are now ignored). Long chains are started with `setsid nohup ... &`, not inside tmux: a deleted tmux session once killed a running chain.
* **Chains.** `scripts/chain_phase3.sh` (HotpotQA, RAGBench, synthetic), `scripts/chain_estb.sh` (retrieval and variants runs) and `scripts/chain_replication.sh` (Llama/Mistral) run GPU stages through `scripts/gpu_run.sh` and the CPU analysis stages in between. A failed stage is logged as `WARNING` and the chain continues.
* **Phase 3 compute.** Qwen external and ESTB runs took 0.3–1.4 GPU-hours each. E5 runs are dominated by loading and searching the 65 GB index. The Llama runs are faster because Qwen's retrieval cache is reused.
* **Where things are.** Runs live under `/scratch/users/iacopog/revise/runs/<run_name>/` with `report/summary.md`, `probes/`, `patch/`, `ablate/`, `steer/`, `text_baseline/`, `closedbook/`, `adaptive/` (`results.json`, `directions.json`), and for ESTB runs `retrieved_<method>.jsonl` and `variants.jsonl`. The retrieval corpus and indices are under `/scratch/users/iacopog/revise/retrieval/`. The paper's tables and figures are regenerated from these artifacts by `Paper/figures/make_tables.py`, `make_figures.py`, `make_estb_table.py` and `make_adaptive_tables.py`, so no number in a table is typed by hand.
* **Paper layout.** Each supplementary experiment has its own appendix file in `Paper/sections/appendices/` (behaviour, splits, augmented regime, transfer, conversational, 3–4 hop, 2Wiki, closed book, logistic steering, v_dm steering, single-layer patching, adaptive retrieval, per-group results, ESTB, compute).
