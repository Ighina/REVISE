Yes—this is a substantially cleaner idea. The progressive evidence trajectory gives you **paired within-question interventions** and a natural transition point, but the strongest target is probably not a generic “answer-update direction”; it is an **evidence-sufficiency or knowledge-accommodation signal** that emerges when newly supplied evidence resolves what the model was missing.

The main risk is that a shared direction merely encodes **“the model is now following contextual rather than parametric knowledge.”** Existing mechanistic work already finds separable contextual and parametric knowledge dynamics, residual-stream conflict signals, and context-following directions, with recent results suggesting that such directions transfer poorly across tasks. [alphaxiv](https://www.alphaxiv.org/abs/2609.00753)

## Recommended framing

A sharper research question is:

> **As evidence becomes sufficient to answer a question, does an LLM undergo a shared internal state transition that predicts successful knowledge accommodation before the answer is generated?**

That gives you three related but distinct objects:

| Signal | Meaning | Potential use |
|---|---|---|
| Evidence sufficiency | The supplied context now contains enough information | Decide whether to retrieve again |
| Evidence uptake | The model has internally incorporated the decisive fact | Detect ignored evidence |
| Answer revision | The output changes after new evidence | Behavioral consequence |
| Correctness transition | The answer changes from wrong to correct | Supervised target used to identify useful internal changes |

The hypothesis should be that **sufficiency and uptake precede and predict revision**, not that every answer revision has one universal direction.

## Avoid stopping early

I would not literally stop when the answer first becomes correct. Run the entire evidence trajectory for every question, including stages after the first correct answer.

Otherwise, you condition the dataset on the model’s behavior and lose important transitions:

- Wrong \(\rightarrow\) correct: desired accommodation.
- Wrong \(\rightarrow\) different wrong: evidence-induced movement without successful integration.
- Correct \(\rightarrow\) correct: reinforcing or redundant evidence.
- Correct \(\rightarrow\) wrong: distraction or harmful updating.
- Correct \(\rightarrow\) different correct formulation: semantic invariance.
- Wrong \(\rightarrow\) same wrong: ignored evidence or entrenched parametric knowledge.

The last two stages after correctness are particularly useful for distinguishing **“I have enough evidence”** from **“I happened to produce the right answer.”**

## Prompt design

I would **not** use “Given additional evidence \(X\), would you change your answer?” as the primary condition. That introduces several artifacts:

- It explicitly tells the model that revision is expected.
- It makes the previous answer salient, creating anchoring or choice-supportive bias.
- The model may answer the metacognitive question without recomputing the QA task.
- It turns the setup into conversational belief revision rather than ordinary RAG.
- Different models may interpret “would you change?” as a social or instruction-following task.

Recent work finds that merely exposing models to their previous choices can increase commitment, while contradictory advice can produce a separate over-updating bias. [nature](https://www.nature.com/articles/s42256-026-01217-9)

### Primary protocol

Use **stateless recomputation** at every evidence stage:

```text
Use the supplied evidence to answer the question.
If the evidence is insufficient, output INSUFFICIENT.

Evidence:
{cumulative_evidence_k}

Question:
{question}

Answer:
```

Each stage gets a fresh forward pass with the same template:

\[
C_0 \subset C_1 \subset \cdots \subset C_K.
\]

Read activations at the fixed final prompt token immediately before answer generation. This gives you comparable positions despite contexts having different lengths.

However, include at least two instruction regimes:

1. **Evidence-only:** “Answer using only the supplied evidence.”
2. **Evidence-augmented:** “Use the supplied evidence and your existing knowledge.”

The first isolates context sufficiency but may exaggerate context-following. The second is closer to realistic RAG but introduces context–memory arbitration. Existing studies show that the interaction between retrieved and parametric knowledge can vary substantially with the task and instruction. [arxiv](https://arxiv.org/html/2506.06485v3)

### Secondary protocol

Then use a conversational revision condition:

```text
Your previous answer was: {A_k}

New evidence:
{E_{k+1}}

Answer the question again. Change your answer only if the total evidence warrants it.
```

This should be an ablation answering:

> Is the internal transition intrinsic to evidence integration, or does it appear only when the model is explicitly asked to reconsider?

If the direction appears in stateless recomputation and transfers to conversational revision, the claim becomes considerably stronger.

## Dataset choice

**MuSiQue is probably the best starting point.** It has 2–4-hop questions, gold supporting paragraphs, explicit question decompositions, distractors, and answerable/unanswerable contrast pairs created by removing a necessary supporting paragraph. [arxiv](https://arxiv.org/html/2108.00573v3)

That structure gives you a principled progression:

\[
\text{distractors}
\rightarrow
\text{first required hop}
\rightarrow
\text{additional required hops}
\rightarrow
\text{minimal sufficient evidence}
\rightarrow
\text{redundant supporting evidence}.
\]

The minimally different answerable/unanswerable pairs are especially useful because you can examine whether the model has an internal **context-sufficiency transition** when the missing support is restored. [aclanthology](https://aclanthology.org/2023.emnlp-main.220.pdf)

Other options have different benefits:

| Dataset | Advantages | Disadvantages |
|---|---|---|
| **MuSiQue** | Gold decompositions, 2–4 hops, controlled unanswerability | Multi-hop reasoning may mix evidence uptake with reasoning completion |
| **2WikiMultiHopQA** | Explicit structured evidence triples and reasoning paths | Template artifacts and entity-heavy answers |
| **HotpotQA** | Large, easy tooling, sentence-level supporting facts | Mostly two-hop; known shortcut concerns |
| **QASPER** | Specialist questions, full-document grounding, human evidence annotations | Answers are heterogeneous and often long-form |
| **Custom atomic QA** | Precise control over evidence strength and sufficiency | Lower ecological validity |

HotpotQA provides sentence-level supporting facts across Wikipedia documents, while 2WikiMultiHopQA additionally provides structured subject–relation–object evidence paths. QASPER is attractive for reducing contamination because its questions target information in full NLP papers and include human-selected supporting evidence, although its answer formats are less controlled. [hotpotqa.github](https://hotpotqa.github.io/)

My preference would be:

- **Phase 1:** 2WikiMultiHopQA or a filtered MuSiQue two-hop subset.
- **Phase 2:** Full MuSiQue with 2–4 hops.
- **Phase 3:** QASPER or another specialist corpus for domain transfer.

## Evidence trajectories

Do not equate “more tokens” with “stronger evidence.” Define semantically meaningful evidence stages.

For a two-hop question requiring \(e_1\) and \(e_2\), construct:

- \(C_0\): no evidence.
- \(C_D\): matched distractors only.
- \(C_1\): \(e_1\) only.
- \(C_2\): \(e_2\) only.
- \(C_{12}\): \(e_1+e_2\), minimally sufficient.
- \(C_{12D}\): sufficient evidence plus distractors.
- \(C_{12R}\): sufficient evidence plus redundant support.
- \(C_F\): full available document context.

Run both evidence orders:

\[
e_1\rightarrow e_2
\qquad\text{and}\qquad
e_2\rightarrow e_1.
\]

Also randomize paragraph and sentence order. Otherwise, a direction can encode recency, context length, or position rather than sufficiency.

For 3–4-hop questions, sampling every permutation is expensive. Use:

- Canonical reasoning order.
- Reverse order.
- Several random permutations.
- Leave-one-support-out contexts.
- Minimal sufficient versus full evidence.

The leave-one-out condition is especially powerful: the text difference can be very small while answerability changes.

## Activation construction

At layer \(l\), let

\[
h^{(l)}_{i,k}
\]

be the activation for question \(i\) after evidence stage \(k\), measured at a fixed answer-start anchor. Define the paired update:

\[
\Delta h^{(l)}_{i,k}
=
h^{(l)}_{i,k}-h^{(l)}_{i,k-1}.
\]

You can then label each increment along several axes:

\[
y^{\text{sufficient}}_{i,k}
=
\mathbf{1}[C_k\text{ is sufficient and }C_{k-1}\text{ is not}],
\]

\[
y^{\text{revision}}_{i,k}
=
\mathbf{1}[A_{i,k}\not\equiv A_{i,k-1}],
\]

\[
y^{\text{correction}}_{i,k}
=
\mathbf{1}[A_{i,k-1}\text{ wrong}, A_{i,k}\text{ correct}].
\]

Do not initially average all wrong-to-correct deltas. Train separate linear models for:

1. Sufficient versus still-insufficient evidence increments.
2. Evidence incorporated versus behaviorally ignored.
3. Wrong-to-correct versus wrong-to-wrong updates.
4. Stable-correct versus destabilized-correct updates.

If one direction predicts all four in the expected ordering, that is much stronger evidence for a shared accommodation process.

## Critical confound

The raw paired difference

\[
h(Q,C_k)-h(Q,C_{k-1})
\]

will inevitably contain information about:

- Added tokens.
- Context length.
- New entities.
- Answer-string occurrence.
- Passage relevance.
- Context position.
- Change in the output distribution.
- General semantic content.

A successful classifier could therefore detect that the answer text has appeared in the prompt rather than a metacognitive state transition.

You need matched increments:

| Positive increment | Matched control |
|---|---|
| Add decisive gold evidence | Add equally long relevant distractor |
| Add final missing hop | Add a redundant already-known hop |
| Add evidence containing answer string | Add answer string in an irrelevant context |
| Add correct evidence | Add plausible false evidence |
| Add first support | Add same support after the answer is already correct |
| Add evidence causing correction | Add identical evidence when the model ignores it |

This lets you estimate something closer to

\[
v_l =
\mathbb{E}[\Delta h_l\mid\text{successful accommodation}]
-
\mathbb{E}[\Delta h_l\mid\text{matched textual change without accommodation}].
\]

An even better approach is to regress out nuisance variables—token count, answer-string presence, evidence relevance, logit change, and passage embeddings—and learn the direction from the residual activation update.

## What direction might emerge?

There are at least four possibilities:

### Answer-content direction

The activation moves toward the representation or unembedding direction of the correct answer. This is expected but not especially novel.

**Test:** Does the direction transfer across disjoint answer entities and answer types?

### Context-following direction

The model switches from parametric recall to contextual extraction. This is plausible because previous studies find distinguishable context and memory pathways, and context-faithfulness steering methods already exploit related directions. [papers.nips](https://papers.nips.cc/paper_files/paper/2025/file/94f7c80260eea8d52a5c0ec7b1637c04-Paper-Conference.pdf)

**Test:** Does it appear when new evidence agrees with parametric knowledge, not only during conflict?

### Sufficiency direction

The model represents whether the currently supplied evidence supports a determinate answer, independently of which answer it is.

**Test:** Can it detect minimally sufficient contexts on unseen questions before decoding?

### Accommodation direction

The model has successfully integrated new evidence into its answer computation.

**Test:** Does projecting out or steering this direction causally alter whether valid new evidence updates the answer, while leaving irrelevant evidence ineffective?

The **sufficiency** and **accommodation** interpretations are the valuable ones. A context-following direction would still be useful, but the novelty would be lower because conflict and context-reliance signals are already detectable in residual activations. [research.ed.ac](https://www.research.ed.ac.uk/en/publications/analysing-the-residual-stream-of-language-models-under-knowledge-/)

## Causal validation

After finding a candidate \(v_l\), test more than linear separability.

### Steering

Add or subtract \(v_l\) after presenting partial evidence:

\[
h'_l=h_l+\alpha v_l.
\]

Ask whether positive steering:

- Increases correct uptake only when sufficient evidence exists.
- Avoids hallucinating an answer when evidence remains insufficient.
- Does not cause indiscriminate context copying.
- Does not increase following of false evidence.

If steering makes the model accept any passage, you found a **context-compliance direction**, not accommodation.

### Activation patching

Patch the post-evidence activation from a sufficient run into the matched insufficient run. Then patch only the component parallel to \(v_l\):

\[
h_{\text{patched}}
=
h_{\text{insufficient}}
+
\operatorname{proj}_{v_l}
\left(
h_{\text{sufficient}}-h_{\text{insufficient}}
\right).
\]

Compare this with:

- Orthogonal-component patching.
- Random-direction patching.
- Same-answer, different-question patching.
- Same-question, distractor-update patching.

### Pre-generation detection

The strongest result would be that the direction predicts whether evidence has been successfully integrated **before the first answer token**. If it is only detectable after generating the corrected answer, it may simply reflect the changed output.

## Training split

Split by more than question ID:

- Disjoint documents.
- Disjoint answer entities.
- Disjoint relation types where possible.
- Disjoint question templates.
- Separate datasets for transfer evaluation.

This is particularly important for 2WikiMultiHopQA because its structured construction could permit relation- or template-specific shortcuts.

A strict hierarchy would be:

1. Within-dataset, unseen questions.
2. Unseen answers and entities.
3. Unseen relation or reasoning types.
4. MuSiQue \(\rightarrow\) 2Wiki transfer.
5. Wikipedia QA \(\rightarrow\) QASPER transfer.
6. Model-family transfer.

Recent evidence that source-authority directions transfer weakly across tasks means that failure at levels 4–6 is a realistic outcome, not merely an implementation defect. [alphaxiv](https://www.alphaxiv.org/abs/2609.00753)

## Downstream purpose

There is one important conceptual issue: if you need to reveal ground-truth evidence until the answer becomes correct, the resulting method is not yet a standalone UQ estimator. It is an **adaptive retrieval or evidence-sufficiency detector**.

The natural deployment loop is:

1. Retrieve initial passages.
2. Run the model and read the accommodation/sufficiency score.
3. If insufficient, retrieve additional passages.
4. Stop when the internal signal indicates that evidence has been integrated.
5. Answer or abstain.

This is arguably more useful and defensible than generic UQ. The score predicts **whether more retrieval is needed**, not merely whether the final answer “feels uncertain.”

A later extension can ask whether the learned direction transfers to closed-book QA:

\[
s(x)=v_l^\top h_l(x).
\]

If the signal predicts errors even without external evidence, it becomes a broader uncertainty estimator. But that should be a downstream discovery, not the initial promised result.

## Best initial experiment

I would start with:

- One open-weight 7B–9B model.
- MuSiQue two-hop or 2WikiMultiHopQA.
- Approximately 2,000 questions the model answers incorrectly without context.
- Stateless fresh prompts at every evidence stage.
- Both gold-evidence orders.
- Gold, distractor, redundant, and false-evidence increments.
- Activations at answer-start across all layers.
- Four transition labels rather than only wrong-to-correct.
- Document-, entity-, and answer-disjoint evaluation.

The primary test would be:

> Does a linear direction learned from paired activation updates distinguish a final missing-support increment from matched non-decisive increments, and does it predict subsequent correction beyond answer logit change, answer-string presence, passage relevance, and context length?

That is a crisp, falsifiable result. If successful, the next step is causal steering and adaptive retrieval—not immediately claiming a universal uncertainty vector.

PLEASE WHILE IMPLEMENTING THIS DO NOT USE THE CURRENT WORKING DIRECTORY FOR DOWNLOADING WEIGHTS, DATA and SAVE ARTIFICATS: USE /scratch/users/iacopog INSTEAD.
ALSO, DO CREATE A VIRTUAL ENVIRONMENT CALLED revise WHERE YOU DOWNLOAD ALL THE REQUIRED PYTHON PACKAGES REQUIRED AND USED THAT THROUGHOUT
