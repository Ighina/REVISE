# Background: concepts and their implementation in REVISE

This document explains, in detail, the concepts used in the REVISE experiments and how each one is implemented in the codebase (`revise/`). It is written for a reader who knows machine learning and language models but has not followed the project, and it is meant to be read alongside the paper draft in `Paper/`.

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

A two-hop question such as "Who is the spouse of the performer of *Green*?" requires two facts: *Green* was performed by Steve Hillage (hop 1), and Steve Hillage's spouse is Miquette Giraudy (hop 2). MuSiQue supplies, for every question, the gold paragraph for each hop, a decomposition with the sub-answer of each hop and the relation used (`performer`, `spouse`), and eighteen distractor paragraphs. This gives a principled notion of "sufficient": a context is sufficient if and only if it contains every gold hop paragraph. 2WikiMultiHopQA supplies the same in a slightly different format (evidence triples instead of decompositions) and is used as an out-of-distribution transfer set.

### 2.2 Loaders (`revise/data/loaders.py`)

Each dataset is converted into a `Question` object (`revise/data/schema.py`) with: the ordered gold paragraphs (`hop = 0, 1, ...` in reasoning order, taken from the decomposition order for MuSiQue and from the evidence-chain order for 2Wiki), the distractors, the answer and its aliases, the per-hop sub-answers (`decomposition`), the relation types (the text after `>>` in a MuSiQue sub-question, or the relation of a 2Wiki triple) and the bridge entities. A normalised JSONL copy is cached under the scratch data directory so later stages never touch the Hub.

### 2.3 Question selection

For each model, 4,000 questions are screened closed-book (question only, no evidence). Only questions the model gets wrong are kept (2,000 per model). This guarantees that every evidence increment concerns knowledge the model needs, so a correct answer after evidence is attributable to the evidence. Closed-book accuracy was 1.4% (Qwen), 4.0% (Llama) and 1.3% (3–4-hop), so the filter removes little but matters conceptually.

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

Four groupings, each guaranteeing that train and test share nothing of that type: question id; **document** (union-find over questions sharing any gold Wikipedia title, then whole components assigned to one side; a component larger than the test target is forced to train, which was necessary for 3–4-hop questions where most questions share a document); **answer entity** (normalised answer); **relation** (the tuple of relation types). Test fraction 0.3.

### 5.4 Probe training and evaluation

Per task, per split, per layer: features are standardised on the training split and an L2-regularised logistic regression (C = 0.5, balanced classes) is fitted. The reported layer is chosen by 3-fold **grouped** cross-validation on training questions only (never on the test set), and the number reported is the held-out AUROC at that layer. The direction is the weight vector divided by the feature standard deviations, normalised. The **difference-of-means** direction (class-mean difference, normalised) is fitted alongside and evaluated by projection. 95% confidence intervals come from a question-level bootstrap of the test set (`revise/analysis/bootstrap.py`).

### 5.5 Nuisance analysis

Three quantities answer "is the probe reading something trivial?":

* **nuisance-only AUROC**: logistic regression on the covariates alone (standardised, plus the 16-dim passage embedding);
* **residualised probe**: ridge-regress Δh on the covariates (training data), probe the residual;
* **gain beyond nuisance**: AUROC of a classifier on [covariates, out-of-fold probe score] minus the nuisance-only AUROC. Out-of-fold scores (5-fold, grouped by question) prevent leakage.

### 5.6 Transfer

`apply_directions` scores a stored direction on another feature set without retraining: stateless → conversational (Δ defined as h_conv(Cₖ→Cₖ₊₁) − h_stateless(Cₖ)), MuSiQue → 2Wiki, and 2-hop → 3–4-hop. Cross-model transfer of raw vectors is impossible (hidden sizes differ), so model-family transfer is replication of the whole procedure. `transfer_matrix` reports the AUROC of each task's direction on every other task and the cosine between directions.

### 5.7 Text-only baselines (`revise/analysis/text_baseline.py`)

The reviewer question "would a text classifier do as well?" is answered by training, on exactly the same increments, labels and splits, a TF-IDF logistic regression and a fine-tuned DeBERTa-v3-base cross-encoder, each on "question [SEP] added paragraph" and on that plus the previous context. Training positives are kept and negatives subsampled in the reduced-budget version.

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

---

## 7. Downstream uses

* **Adaptive retrieval** (`revise/run/adaptive.py`): walk each held-out question along its canonical trajectory (distractors → hop 1 → hop 2 → + distractor), stop when the pre-generation state-sufficiency score exceeds a threshold calibrated on training questions at a target precision, and compare accuracy and rounds with "stop when the model does not say INSUFFICIENT", an entropy threshold, an oracle and full context. No new forward passes are needed since every stage's activations and answers are stored.
* **Closed-book transfer** (`revise/run/closedbook.py`): capture the closed-book activations of the screened pool and score every stored direction for predicting closed-book correctness and abstention, against a probe trained on closed-book data as the ceiling.

---

## 8. Engineering notes that affect interpretation

* **Numerical noise.** Everything runs in bfloat16. Greedy generations are stable across batch compositions, but gold log-probabilities can differ by up to about 0.5 nats between batch layouts, so that covariate is coarse.
* **Resumability.** Every stage is idempotent: behaviour rows and activation shards already present are skipped, which is why runs could be interrupted, topped up and continued.
* **Compute.** One 2-hop model (35k conditions) takes about 22 hours on 64 CPU cores or about one hour on a shared H200. GPUs on this host are shared through `gpuq` with a weekly budget; the whole set of causal and baseline experiments used about 7 GPU-hours. Analysis stages (features, probes, transfer, adaptive, report) run on CPU.
* **Where things are.** Runs live under `/scratch/users/iacopog/revise/runs/<run_name>/` with `report/summary.md`, `probes/`, `patch/`, `ablate/`, `steer/`, `text_baseline/`, `closedbook/`. The paper's tables and figures are regenerated from these artifacts by `Paper/figures/make_tables.py` and `Paper/figures/make_figures.py`, so no number in the paper is typed by hand.
