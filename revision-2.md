# Rewrite
## Rationale

The current paper appears good for AI reviewers, but from the standpoint of a human reader it is difficult to read, the paper appears:

- Not focused enough: the main contribution and rationale for the paper should be "do internal states show that evidence is enough and will be used by LLMs? And do internal states are better at predicting this than baselines (classifiers based on text and self verbalization)?" (I will do further work on the related work section to justify this based on human cognition literature). The paper, however, currently reads as a mixture of experiments where this overarching narrative and main reason to conduct such as experiments get lost in many small technical elements.

- The writing often looks artificial: lately there has been a lot of effort put into ensuring that papers are not just completely AI slop and if the language sounds too artificial there is the risk that human reviewers / AC might decide that the paper has not undergone the relevant human revision.

- Concepts are often assumed or explained in a way that puts a lot of burden on the reader: simplicity is what shows that someone is mastering the subject, we do not need to explain things in such a way that people feel frustrated to read our paper.

## Fixes

Based on the above problems, there is a number of fixes that can be done.

### Focus
For the focus, the main points that need to be implemented are:

- Rewriting: abstract, introduction and conclusions need to be re-written to really show that the focus is in fact a question about knowing if there are internal states related to the sufficiency and uptake of newly retrieved information, just as suggested by the title. This means that we need to really focus the writing on this specific problem. The idea of causality is interesting but a corollary to the main question: it will still be included in main text and the finding of the causal v_dm direction also it is interesting, but the abstract does not need to mention this in more than one sentence, while also the introduction should include this as a corollary contribution, where the main motivation remains the exploration of internal states in a context of evidence which might or might not be relevant is incrementally retrieved.

- Rearraging: it follows from above that many of the information currently contained in the main text are not strictly necessary and in fact might confuse a reader and make them lose focus about the main point of enquiry. As such, we need to move many elements that are not strictly required for the central arguements into appendix. The central argument, following the existing sections and the order of importance will be as follow:
1) We find the directions related to the sufficiency and uptake of new evidence (linear probes).
2) We find that these internal directions are more effective than text baselines (text classifiers) and self-verbalization baselines (INSUFFICIENCY output).
3) We find that these directions are not causal. The causal direction we can find is v_dm, the difference of means of sufficient and insufficient states, which acts like an abstention mechanism.
4) We find that the directions found are recoverable across models families and that are partly transferable across a variety of datasets.
Therefore, we have a number of parts in our current draft that we will need to move to appendix for sure:
-- The analysis of the layer depth where the information is recovered from as in beginning of section 5 and especially figure 1: the analysis of the layer depth is interesting but not strictly necessary for the main argument, and it can be moved to appendix. The main text can simply mention that we find that the information is recoverable from multiple layers, and that we report results for best layer (or as we actually did in fact, check the codebase).
-- Correction and stability are interesting things to predict, but they are not directly the sufficiency and uptake information we are interested in: all things related to these can be moved to appendix showing how we have in fact tried to find dimensions for other related concepts and how they are related to the main ones we explored.
-- All the transfer results in section 5 are redundant: a single section about transferability and generalising results to other dataset should be section 7 later on.
-- Figure 2 should be moved to appendix, potentially in an existing appendix about more in details results, but in case there is no similar enough we can create a new appendix briefly describing how different increments project on the sufficiency direction.
-- Section 5.1 is too verbose and generally redundant: we should move the "state vs update" directly in appendix (there should already be an appendix which was going into details of this) and then we can also move the INSUFFICIENCY paragraph (evidence vs output) in a much more compressed form to be included inside section 5.2 about baselines (see later in the Adding part).
-- The explanation of the intervention we conducted at the start of section 6 should really be something that is already explained in the methodology section.
-- The paragraph starting "The causal content is high-dimensional and model-specific." can be moved to appendix together with figure 3 it refers to.
-- The use for adaptive retrieval as a practical application of the things we investigate also should just be mentioned as something that we show in the relative appendix (probably in the conclusions, with a sentence like "our results help us understanding the internal mechanisms of LLMs better, but they might also have implications for concrete tasks like adaptive retrieval, see appendix X for details").
-- The whole Discussion section can be omitted all together: this can be brought up just shall the reviewers ask for explanation about why the found direction are not causal. As previously mentioned the current focus should be on reading this dimensions anyway rather than proving that they are causal or not causal.

- Adding: since we want to show that the internal dimension predict as well as beyond text and self-verbalization baselines, we need to add a paragraph in section 5.2 about the INSUFFICIENCY baseline, which is the output of the model when it is asked to self-verbalize whether it has enough information to answer the question. To do so, however, we need to perform an additional experiment where we simply get the AUROC results of using INSUFFICIENCY to separate sufficiency and uptake. The paragraph about INSUFFICIENCY at the moment goes on and on about different ways of showing how INSUFFICIENCY is not the only reason for which the model predicts sufficiency and uptake, but it currently uses roundabout arguments, such as showing that "On held-out contexts that contain every gold paragraph, the model says INSUFFICIENT 33\% (Qwen), 40\% (Llama) and 43\% 492 (Mistral) of the time (Table 6).". Actually the INSUFFICIENT output used as a classifier should also exist already, but you should check that in fact covers the same data as the ones shown in text-baselines and probe results (also you might want to re-run for the sufficiency (matched)). Also, look up if a comparable baseline can be reported for uptake.

### writing
- In the paper there are currently comments in parenthesis from my collaborator where she found explained things unclear: find those and address, also make sure to clarify any other potentially confusing points.
- The tasks that we explore, sufficiency and uptake, and the increments we use to explore these tasks could be better summarised in a table clearly highlighting:
1) what tasks we are investigating: i) sufficiency, ii) uptake
2) what increments we are using to explore these tasks: i) decisive, ii) non-decisive, iii) irrelevant (in all its forms)
These tables can be minimal so as not to take too much space in the main text and then be complete in appendix