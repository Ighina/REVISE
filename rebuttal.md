# Rebuttal — Submission 2364

Limit: 5000 characters per response. Triples of numbers are Qwen / Llama / Mistral. Everything marked **[new]** was computed for this rebuttal from the stored activations and outputs (`revise/analysis/rebuttal.py`, document split, held-out questions); no new model runs were needed.

<!-- BEGIN C9nT -->
## Response to Reviewer C9nT

Thank you for the careful reading. We answer your three questions and the point on the causal tests.

**1. Why study sufficiency and uptake together?** We agree they are different things, and the paper treats them as two tasks with two probes. Sufficiency is a property of the context, labelled by construction. Uptake is a property of the model's behaviour, defined only on increments where the evidence has just become complete. The results confirm the distinction: the two directions are nearly orthogonal (cosine 0.02–0.03), the sufficiency direction predicts uptake at only 0.54–0.60, sufficiency peaks at mid-depth and uptake near the last layers, and a text classifier can predict sufficiency (0.89–0.92) but not uptake (at most 0.60). We study them together because they are the two consecutive decisions of a RAG loop: is the evidence complete, so retrieval can stop, and will the model use it, so the answer can be trusted. The gap between them is large: on complete evidence the models answer correctly only 36–47% of the time. We will state this in the introduction instead of leaving it to Table 1.

**2. Have you used the scores to guide retrieval or abstention?** Yes, in Appendix R, which the main text cites only in the conclusion. We will move a summary into the main text.
- Stopping retrieval: the update-sufficiency score matches the oracle on all three models (accuracy 0.478 / 0.468 / 0.336 against 0.475 / 0.470 / 0.336 for the oracle and 0.404 / 0.385 / 0.243 without stopping). It is read before any generation, whereas the rule "stop at the first answer that is not INSUFFICIENT" needs a generation per round. On HotpotQA the sufficiency probes reach 0.73 against 0.65 for that rule.
- Trusting the answer: keeping the half of the answers ranked highest by the uptake score gives higher accuracy than ranking by output probability in five of six settings (0.789 against 0.751 on MuSiQue, 0.957 against 0.870 on 2Wiki).
- Abstention **[new]**: with a threshold calibrated on held-out folds, the sufficiency probe accepts 47–59% of the complete contexts the model wrongly refuses, at about 1% false alarms and 0.90–0.93 precision. The model's own answers, read as a sufficiency verdict, have 0.40–0.50 precision.

These are simulations on stored trajectories and the gains over the behavioural rule are modest on MuSiQue; we say so in the appendix and will keep saying so. On the interventions: the null result concerns the probe directions only and does not limit their use as monitors. The mean-difference direction has a large effect (projecting it out takes answering from 0.72 to 0.01).

**3. What does the probe offer beyond DeBERTa?** For sufficiency, not higher accuracy than a well-trained text classifier, and we will say this more plainly. Table 3 uses a reduced training budget (0.89–0.92); with the full budget DeBERTa with context reaches 0.97 on Qwen, above the probe's 0.95. We will add that row. Sufficiency is largely a property of the input, as the paper says. What the probe adds is different:
- It tells us what this model has registered. Its natural comparison is the model's own judgement, which is the signal RAG systems use today: 0.98 against 0.71–0.77 for the verbalised verdict, and **[new]** against 0.78–0.85 for the model's continuous refusal log-odds (paired difference +0.13 to +0.21, 95% intervals above +0.11). Among contexts the model refuses, the probe still separates complete from incomplete evidence at 0.97–0.98. A text classifier cannot show that a model has registered evidence it then refuses to use.
- It costs a dot product on a forward pass that is run anyway, with no second model.
- For uptake the input is not enough: text classifiers stay at 0.47–0.60 (0.63 with full budget) against 0.77–0.78.

**4. Size of the causal evaluation, effect sizes and intervals.** Each test uses 150 held-out questions per condition set and is repeated on three models, on 3–4-hop questions, on three external benchmarks and on retrieved evidence, with the same outcome each time. The effects are far outside sampling noise: with n=150 the 95% interval of any reported rate is at most ±0.08, against changes such as 0.13→0.71 (full patch) and 0.72→0.01 (projection), while the null conditions stay within 0.01 of the baseline (0.13 against 0.13). We will add Wilson intervals to Tables 4, 5 and 15 and paired differences against the random-direction control. We will also state the conclusion as "no effect under the interventions we tested" rather than as a general claim.
<!-- END C9nT -->

<!-- BEGIN AcFv -->
## Response to Reviewer AcFv

Thank you for the review. Your summary is accurate, including its last sentence, and we start there.

**1. The signal reflects that the last expected piece has arrived, not a judgement of reliability.** We agree, and the paper reports it: a falsified final paragraph projects second only to the true one, and Appendix C says the direction encodes "the final slot has been filled" rather than truth. Partial and contradictory paragraphs are the hardest controls (0.81 and 0.76). This is currently in an appendix. We will state it in the main text and in the Limitations: our label is completeness of the gold evidence, not semantic sufficiency or reliability.

**2. What "sufficient" means.** Sufficiency could be defined relative to the model's knowledge and reasoning. That is why the paper splits the question in two.
- Sufficiency (Section 3) is defined without reference to the model: a context is sufficient exactly when it contains every gold paragraph of a compositional question, and MuSiQue is built so that every hop is needed. The section says it is "defined by the construction and not by the model's behaviour".
- The model-relative part, whether this model with its knowledge and reasoning turns complete evidence into a correct answer, is the second task, uptake, measured on behaviour.

Model knowledge is controlled: the main runs keep only questions the model fails without evidence, and on HotpotQA, where known questions are kept, the readout is the same for both groups (0.984 and 0.978). The distance between the two notions is one of our results: on complete evidence the models answer correctly 36–47% of the time and say INSUFFICIENT 30–36% of the time. The conclusion you quote uses the word without restating the definition. We will write "the gold evidence is complete" in the abstract, introduction and conclusion.

**3. Relation to probes that decode confidence.** The paper tests this directly, and the sufficiency signal is not confidence:
- Output confidence detects the decisive increment at 0.41–0.45 AUROC, below chance, because INSUFFICIENT is a confident output (Table 3). The verbalised verdict reaches 0.65–0.69, and the model's continuous refusal log-odds, added for this rebuttal, 0.69–0.75. The probe reaches 0.95.
- With the output held fixed, on increments after which the model still says INSUFFICIENT, the probe stays at 0.94 (Table 11).
- After the changes in entropy, top probability and gold-answer log-probability are regressed out of the activations, the probe stays at 0.94–0.96.
- The sufficiency direction is nearly orthogonal to the direction that predicts correctness (cosine 0.02–0.03) and predicts correctness at only 0.54–0.60.

Work on confidence and truthfulness probes predicts whether an answer will be right. Our target is a property of the evidence, tested against matched controls that start from the same context, and compared on the same inputs with the model's verbalised verdict and with a text classifier. We know of no earlier study that does this; if you have specific papers in mind we will gladly discuss them. For uptake we agree with you, and the paper says so: much of that readout is the model's decision to answer, and the part specific to correctness is modest (Section 5.1, Table 12). After Reviewer X5jE's comments we will further weaken that claim and drop "best single predictor" from the abstract.

**4. Value to the community.** Beyond the readout itself: (i) models register complete evidence and still refuse to answer a third of the time, and a linear readout recovers a large share of those cases; (ii) a negative result with controls, that directions with 0.95 AUROC have no effect when steered, patched or projected out, while a different direction gates abstention; (iii) partial transfer from Wikipedia to RAGBench domains, a warning against deploying such detectors across domains; (iv) a use of the scores for stopping retrieval (Appendix R) and a released benchmark.

**5. Readability.** We will define the two tasks with a worked example trajectory on page 1, give one takeaway sentence per results paragraph, and move secondary numbers to tables.

**6. Soundness.** The concerns above are about framing and novelty. We did not find a methodological issue in the review (labels, splits, controls, statistics). If there is one behind the Soundness score, we would be grateful to hear it so that we can respond.
<!-- END AcFv -->

<!-- BEGIN X5jE -->
## Response to Reviewer X5jE

Thank you for checking the details. You are right on W3 and largely on W1, W2 and W4. We ran the analyses you asked for (new; document split, test questions).

**W1. Uptake versus endpoint correctness.** Correct: on decisive increments the label is correctness after completion. Positives by answer before the decisive paragraph, over 4000 decisive increments per model:
- from INSUFFICIENT: 1439 / 1339 / 859
- from a wrong answer: 103 / 128 / 192
- already correct: 311 / 390 / 395 (17 / 21 / 27% of positives)

Two stricter labels, same layer:
- Newly correct only (increments with a correct earlier answer removed): stored probe 0.774 / 0.797 / 0.761 (original 0.775 / 0.783 / 0.771); refitted 0.820 / 0.820 / 0.778.
- Counterfactual dependence (also removing positives for which the model is correct after any matched control added to the same penultimate context: distractor, duplicate or answer string; 1356 / 1322 / 937 positives remain): stored 0.772 / 0.794 / 0.765; refitted 0.824 / 0.824 / 0.786.

The conclusions hold. We will adopt the strict label and report the transition table.

**W2. Which directions were intervened on.** The text is ambiguous and Section 4 is wrong on one point. In all causal tests the sufficiency directions are state directions, fitted on h(C) for sufficient against insufficient contexts: v_lr is that logistic probe and v_dm that mean difference. The update-sufficiency direction of Table 2 was not intervened on; its cosine with the state logistic direction is 0.23–0.26. The uptake direction is the update one.
- Projection-out (Table 4, right): v_lr and v_dm, every layer from the best layer on, anchor and generated positions. Identical schedule, 0.72 against 0.01: this is the matched comparison.
- Patching: the per-layer v_lr component of the update, same layers, anchor only.
- Steering v_lr (App. N): best layer only, anchor onward, up to 22 l2.
- Steering v_dm (App. O): four layers from the best, |alpha| <= 1.

The steering schedules differ. We will add this mapping as a table, steer v_lr under the four-layer schedule, and run the update-sufficiency directions through projection and patching. We will state the result as a null under the additive, projection and component-patching interventions tested at the anchor, not as "causally unused".

**W3. Calibration and numbers.** Your arithmetic is right. Held-out precision at the training threshold is 0.81 / 0.80 / 0.83: the threshold was set on in-sample scores of the contexts the probe was trained on. "At 0.95 precision" will be removed. With the threshold set on out-of-fold scores (5 folds grouped by question), held-out precision is 0.90 / 0.91 / 0.93, TPR 0.65 / 0.69 / 0.73, FPR 1.5 / 1.5 / 1.2%; the probe then recovers 47 / 52 / 59% of wrongly refused complete contexts with about 1% false alarms. Still below nominal, so we will report held-out precision at both operating points (the verbalised verdict has precision 0.49 / 0.50 / 0.40).
- 0.82 is the question-split value; the cross-task figure uses the document split (uptake to correction 0.77 / 0.77 / 0.76, the reverse 0.68 / 0.71 / 0.70). We will use the document split throughout.
- 2Wiki uptake is 0.894. The 0.91 in Section 7 is the correction value (0.914), entered by mistake.
- RAGBench: "five" is its own domain taxonomy, while Table 18 groups its sub-datasets six ways. We will use one taxonomy and list the mapping.

**W4. Continuous output baselines and paired intervals.** We added the exact refusal log-odds at the anchor, log p/(1-p) of the first refusal token.
- Sufficiency, count-matched contexts: 0.84 / 0.85 / 0.78 (binary verdict 0.77 / 0.77 / 0.71) against 0.976 / 0.981 / 0.986 for the probe; paired differences +0.13 [0.12, 0.15], +0.13 [0.12, 0.15], +0.21 [0.19, 0.22]. Among refused contexts: 0.72 / 0.74 / 0.68 against 0.97 / 0.97 / 0.98. On increments: 0.74 / 0.75 / 0.69 against 0.95. The sufficiency advantage holds.
- Uptake: it does not. Refusal log-odds reach 0.795 / 0.790 / 0.826 against 0.775 / 0.783 / 0.771 for the probe: -0.020 [-0.053, 0.016], -0.007 [-0.041, 0.027], -0.055 [-0.087, -0.023], so the output is ahead on Mistral. Among answered increments, probe minus confidence is +0.022 [-0.032, 0.077], +0.019 [-0.043, 0.084], -0.009 [-0.065, 0.047]; the refitted probe and the combination with the decision to answer also have intervals that include zero.

We will remove "best single predictor of correctness" from the abstract and introduction. The uptake claim becomes: readable from activations, not from text (at most 0.63), and not shown to exceed the model's own output.

**Other points.** We will separate current-state sufficiency (state probe) from the transition into sufficiency (update probe) in the main text; Appendix E shows the information is in the resulting state (0.93–0.94 against 0.95). The Limitations will add that the label is gold-evidence completeness rather than semantic sufficiency, and the calibration gap.
<!-- END X5jE -->

---

# Not part of the reviewer responses

## Draft confidential note to the AC about Review AcFv

Use the venue's review-issue form or confidential comment to the AC/SAC, not the public response. Keep the response to the reviewer above as it is.

> We would like to flag two points about the review by Reviewer AcFv, and we ask only that they be weighed in the meta-review.
>
> 1. The scores do not follow from the review text. Soundness is 1.5 and Overall is 1 ("has to be fully redone, or is not relevant to the *ACL community"). The review names no methodological flaw: both weaknesses concern framing and novelty, and the same review says "the authors conduct a detailed evaluation" and "the results are clearly presented", and rates Excitement 2 ("might resonate with others in the community"). The other two reviewers rate Soundness 3 and 2.5 and Overall 2.5 and 3; one of them checked the appendix tables numerically.
> 2. The two weaknesses are not specific and are addressed in the submission. (a) "What does sufficient mean": Section 3 defines sufficiency by construction (all gold paragraphs present) and states that it is not defined by the model's behaviour; the model-relative part is the separate uptake task. (b) "Contributes little beyond probes decoding confidence": no reference is given, and Table 3 reports output confidence as a baseline on the same task (0.41–0.45 AUROC against 0.95 for the probe), with further controls that hold the output fixed (Table 11).
>
> We have answered the reviewer's points in the author response and have accepted the one we think is right, that the label measures completeness of gold evidence rather than semantic sufficiency.
