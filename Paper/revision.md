# Revision notes: readability pass

This pass rewrites the paper for a human reader who has not followed the project. The model for tone and level of explanation is the abstract and introduction the author rewrote in `my-revision/`. No result, number or claim was changed. Every number in the revised prose was checked against the existing text and tables.

## Files not revised in this pass

`sections/setup.tex` and `sections/method.tex` were **not edited**. Claude Code's permission classifier blocked the command that would have read them, so they could not be revised safely. They still contain the old text.

This affects the example the author gave. The Data subsection of `setup.tex` still lacks HotpotQA and RAGBench. Those datasets are now described as a bulleted list at the start of Section 7, External validity. A ready-to-paste Data subsection is given at the end of this file. Once it goes into `setup.tex`, the opening list of Section 7 can shrink to one sentence that points back to it. The model citations added to `references.bib` (Qwen2.5, Llama 3, Mistral 7B) should be cited where `setup.tex` introduces the models.

## Structure and front matter

- **Abstract and introduction.** Replaced with the author's rewritten versions from `my-revision/`. Only clear typos were fixed:
  - "consisting in" became "consisting of"; "futher" became "further"; "transfering" became "transferring".
  - "two additional models family" became "two additional model families"; "LLMs abstention" became "LLM abstention".
  - A duplicated "release ... which we release" was fixed.
  - The benchmark is now named (Evidence State Transition Benchmark, ESTB) in the last contribution.
  - HotpotQA and RAGBench are cited there, and the section has a `sec:intro` label.
- **Title and author.** Taken from `my-revision/main.tex`: "To Revise or not to Revise? ..." and the author's name.
- **Section order.** Related Work now follows the Introduction, as in `my-revision/`. It is still an empty placeholder. Setup still comes before Method. `my-revision/` swaps them, but that order could not be checked without reading both files.
- **Template.** The article template was kept. `my-revision/` uses the ACL style in review mode, which is anonymous and not meant for arXiv.
- **Bibliography.** Started from `my-revision/references.bib`, which adds the RAG, knowledge-conflict, SpARE, activation-steering and confidence references. Added entries:
  - datasets and retrieval: HotpotQA, RAGBench, FlashRAG, DPR (the wiki-18 corpus), E5, BM25, `bm25s`;
  - models: DeBERTaV3, Qwen2.5, Llama 3, Mistral 7B;
  - methods: linear probes, Marks and Tegmark's comparison of mean-difference and logistic directions, and activation patching (ROME).

## Section 5: Readout results

- A new opening paragraph says what every number is: held-out AUROC, best layer chosen on training questions. It also explains the document split in one sentence.
- **Matched controls** are now spelled out where they are first used: the same starting context plus a length-matched distractor, a duplicate, a falsified copy, or an answer-string paragraph. Before, the reader had to find them in an appendix.
- **Nuisance covariates** and residualising are explained in words. The table caption now says what "nuis." and "gain" mean.
- The paragraph on Figure 3 now says what a projection is, and names the increment kinds in plain words.
- **Uptake** and **correction** are defined in the paragraph that reports them.
- The transfer paragraph says what each of the three target settings is. One example is the conversational protocol, where the model sees its previous answer.
- "What the probes read" was reorganised into shorter paragraphs. Each one states the alternative explanation, the test and the conclusion. "Anchor" is explained again where the output confound depends on it. The count-matched comparison is described as an operation: retrain on two-paragraph contexts only.
- The text-baseline subsection now names the three classifiers and cites DeBERTaV3. The table caption explains "q+added" and "+context".

## Section 6: Causal results

- A new opening explains the three interventions in a short list before any result uses them:
  - steering adds a direction to the residual stream;
  - patching replaces the insufficient prompt's anchor residual with the sufficient one's;
  - projection-out sets the component along a direction to its insufficient-state mean.
- The opening also defines the insufficient and sufficient prompts, and both kinds of direction, with citations.
- The causal table caption now explains every column. Before, labels such as $\parallel v$, rand-256, PC1, PC256 and "unemb." were undefined.
- $\ell_2$ units, the steering strength $\alpha$, principal components and the unembedding direction are explained where they first appear.
- "By the criterion we set in advance" now states the criterion. The undefined term "accommodation" was replaced by its meaning: integrating evidence better.
- The loose final paragraph on adaptive retrieval got a heading. It now explains what a stopping rule and the behavioural baseline are.

## Section 7: External validity

- The three external sources are now a bulleted list, each with its citation, in the style the author asked for. Each entry says in prose why the dataset was built and what it measures. A separate sentence says how the paper uses it.
- The ESTB paragraph now motivates the retrieval axis with one sentence on real RAG systems. It cites FlashRAG and the wiki-18 corpus, and introduces BM25 and E5 as lexical and dense retrievers with citations.
- Long sentences were split, and "per-group breakdown" and "need for evidence" were explained.

## Discussion and conclusion

- The discussion no longer relies on "lever" and "accommodation" without definitions. Each paragraph now states the idea in plain words.
- The explanation of why logistic probes miss the causal direction was rewritten without jargon. A logistic probe down-weights directions with high within-class variance, and the causal shift lies along such a direction.
- The validation advice became an explicit list of three controls.
- The conclusion was rewritten as short sentences that summarise the findings.

## Appendices

- **Evidence conditions (A).** Rewritten as explained prose: what a condition is, what a chain is, why insertion positions are random, and what each matched control is. A new table describes every increment kind name used in tables and figures, with its sufficiency label.
- **Probing details (B).**
  - Activation capture is explained in words: batching, teacher-forced gold log-probability, cropping the cached keys and values.
  - The probe direction's normalisation and the residualised and "gain" analyses are spelled out. The steering unit is related to $\alpha$.
  - Linear probes are cited.
- **Splits.** The document split is explained: questions that share a page are grouped, and each group goes to one side.
- **2WikiMultiHopQA.** Now has a citation, a sentence on why the dataset was built and what it contains, and why we use it.
- **Closed-book transfer.** A motivating sentence was added.
- **Results by group.** A configuration key (`select: all`) that had leaked into the text was replaced by its meaning.
- **ESTB.**
  - Removed a phrase ("the review's evidence-quality axis") that referred to a reviewer report the reader has never seen.
  - Added citations for Qwen2.5, wiki-18, FlashRAG, BM25, `bm25s` and E5. The two retrievers are described as lexical and dense.

## Build

- `Paper/tables/` was missing from the working tree, although `main.tex` needs it. It was restored from the last commit.
- The paper compiles with no undefined references or citations. The remaining overfull boxes all come from result tables and predate this pass: there were 8 before and there are 6 now.
- The page count rose from 30 to 34, because terms are now explained where they are used.

## Upload archive

`Paper.zip` sits at the repository root, next to the Paper folder. It contains only what arXiv needs to compile the paper:
- `main.tex`, `references.bib` and the generated `main.bbl`;
- `sections/`, the table `.tex` files and the figure PDFs.

These were left out on purpose:
- `background.md`, because arXiv publishes all source files and it holds internal notes, server paths and references to the review;
- this file, the figure-generation scripts, the JSON data files, PNG duplicates of the figures, `main.pdf` and LaTeX build files.

The archive was test-compiled on its own. Re-create it after `setup.tex` and `method.tex` are revised.

## Proposed Data subsection for setup.tex

This text is built only from the project notes and the appendices. Check the counts against the current `setup.tex` before pasting it in.

```latex
\subsection{Data}
\label{sec:data}
Our method needs questions whose answer depends on several separate pieces of evidence, together with annotations saying which paragraphs contain them. With such annotations, a context is \emph{sufficient} exactly when it contains every gold paragraph, so sufficiency can be defined from the construction rather than from the model's behaviour. Multi-hop question-answering datasets provide this. We use the following sources.
\begin{itemize}
  \item \textbf{\musique{}} \cite{trivedi2022musique} was built by composing single-hop questions into multi-hop ones, keeping only compositions in which every hop is needed, so that models cannot answer through shortcuts. Each question comes with one gold Wikipedia paragraph per hop, a decomposition into sub-questions with their sub-answers and relation types, and 18 distractor paragraphs. We use its 2-hop questions for all main experiments and its 3--4-hop questions for a separate run (\cref{app:multihop}).
  \item \textbf{\twowiki{}} \cite{ho2020constructing} was built from Wikipedia and Wikidata to evaluate multi-hop reasoning with explicit reasoning paths, and gives the evidence triples that lead to each answer. We use it to test whether directions learned on \musique{} transfer to an independently constructed dataset (\cref{app:twowiki}).
  \item \textbf{HotpotQA} \cite{yang2018hotpotqa} was crowdsourced to test multi-hop reasoning over pairs of Wikipedia paragraphs, with annotated supporting sentences. It contains bridge questions and comparison questions, whose two facts are independent lookups. We use 2{,}000 questions from its distractor setting as an external benchmark (\cref{sec:external}).
  \item \textbf{RAGBench} \cite{friel2024ragbench} was built to evaluate RAG systems on realistic inputs. It collects twelve question-answering datasets from biomedical, finance, technical, customer-support and general domains, with the documents a real retriever returned for each question and annotations of the sentences that support the answer. We use 3{,}000 of its mostly single-hop questions to test the method on naturally retrieved evidence and outside Wikipedia (\cref{sec:external}).
  \item \textbf{A synthetic benchmark}, which we generate, covers fictional entities, so no answer can be known closed-book. Every fact also comes in paraphrased, partial and contradictory versions. We use it to control the quality and wording of the evidence (\cref{sec:external}).
  \item \textbf{wiki-18} \cite{karpukhin2020dense}, a 21M-passage Wikipedia corpus distributed with FlashRAG \cite{jin2025flashrag}, is the retrieval corpus for the retrieval runs of our benchmark, with BM25 \cite{robertson2009bm25} and E5 \cite{wang2022e5} as retrievers (\cref{app:estb}).
\end{itemize}
For each model we screen 4{,}000 \musique{} questions closed-book, i.e.\ without any evidence, and keep 2{,}000 that the model answers incorrectly. Every question therefore concerns knowledge the model lacks, and a correct answer after evidence is added can be attributed to that evidence.
```

If this subsection goes in, the bulleted list at the start of Section 7 can become one sentence pointing back to `sec:data`.

## ACL version (`ACL-Paper/`, `ACL-Paper.zip`)

`ACL-Paper/` is a copy of the revised paper set in the ACL style from `my-revision/`.

- **Preamble.** Taken verbatim from `my-revision/main.tex`, and checked line by line to be identical. `acl.sty` and `acl_natbib.bst` are byte-identical copies.
  - No font, margin, spacing or style package was added or changed. The body is Times at 11pt, review mode, and anonymous ("Anonymous ACL submission").
  - The arXiv version's geometry, numeric natbib and plainnat style were not carried over, because acl.sty sets margins, citations and the bibliography style itself.
- **Limitations.** ACL requires a dedicated Limitations section after the conclusion. The Limitations paragraph was moved out of the Discussion into an unnumbered `Limitations` section, `sections/limitations.tex`. It does not count toward the page limit.
- **Floats.** Wide tables and multi-panel figures use the full-width `table*` and `figure*` environments. Small figures fill one column. Two tables have narrower column padding; no font size was changed.
- **Build.** It compiles with no overflowing boxes and no undefined references, and all fonts are embedded.
- **Page limit: still a desk-rejection risk.** The main text runs to page 11, and the conclusion ends on page 11. ACL long papers allow 8 pages of content, so about 3 pages must be cut or moved to the appendices before submission.
- `setup.tex` and `method.tex` are the same unrevised files as in `Paper/`.
