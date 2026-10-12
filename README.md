# The Evidence Within

Code for *The Evidence Within: Evidence Sufficiency and Uptake in the Residual
Stream of Retrieval-Augmented LLMs*.

The question the paper asks is whether a retrieval-augmented model's internal
state, read **before** it generates anything, already says that the evidence in
its context is sufficient and that it is about to use it — and whether that
internal read-out beats the baselines a practitioner would reach for first: a
text classifier over the retrieved passage, the model's own verbalised
judgement, and its output confidence.

The pipeline builds semantically-defined **evidence trajectories** for
multi-hop questions, runs one forward pass per evidence stage, records the
residual stream at the answer-start anchor for every layer, trains linear
probes on **paired activation updates** `Δh = h(C_k) − h(C_{k−1})` against
matched textual controls, and then validates the directions causally
(activation patching, projection-out, steering) and tests how far they
transfer.

```
quick start
  scripts/setup_env.sh          # environment
  scripts/run_smoke.sh          # 10-minute end-to-end check on Qwen2.5-0.5B
  scripts/run_in_domain_results.sh   # Table 2, and the runs everything else builds on
  scripts/make_paper_artifacts.sh    # rebuild the tables and figures
```

## Contents

- [Setup](#setup)
- [Replicating the paper](#replicating-the-paper)
- [The pipeline](#the-pipeline)
- [Evidence conditions and increments](#evidence-conditions-and-increments)
- [Layout](#layout)
- [Runtime and hardware](#runtime-and-hardware)
- [Notes and assumptions](#notes-and-assumptions)

## Setup

```bash
scripts/setup_env.sh                 # venv in ./.venv with CPU-only PyTorch
CUDA=1 scripts/setup_env.sh          # ... or the CUDA wheel
CONDA=1 scripts/setup_env.sh         # ... or a conda env named "revise"
source .venv/bin/activate

export REVISE_SCRATCH=/path/with/space   # optional; default ./scratch
python -m pytest -q                      # unit tests, ~3 s
python -m pytest -q -m slow              # ... plus the real-backend test (downloads a 0.5B model)
scripts/run_smoke.sh                     # whole pipeline on Qwen2.5-0.5B, ~10 min
```

Everything heavy — model weights, datasets, activations, run outputs — is
written under `REVISE_SCRATCH`, never into the repository. The activation store
dominates: all layers at one token position in float16 is about **7 GB per
regime** for a 2,000-question MuSiQue run, so roughly 20 GB for the Qwen run
(two regimes plus the conversational protocol) and 7–10 GB for each
single-regime run, on top of ~15 GB of weights per model.
`revise.config` also points every Hugging Face cache at
`$REVISE_SCRATCH/hf_cache`; set `REVISE_KEEP_HF_ENV=1` to keep an existing
`HF_HOME` instead.

Llama-3.1-8B is a gated model: accept its licence on the Hub and log in with
`huggingface-cli login` before running the Llama configs.

The retrieved-evidence (ESTB) runs need three extra packages,
`pip install -e ".[retrieval]"`; nothing else does.

### Environment variables

| variable | effect |
|---|---|
| `REVISE_SCRATCH` | where weights, data, activations and runs live (default `./scratch`) |
| `REVISE_DEVICE` | `cpu` or `cuda`; default is CUDA when a GPU is visible |
| `REVISE_BATCH_SIZE` | overrides `model.batch_size`; 32 is a good GPU value, batching barely helps on CPU |
| `REVISE_NUM_THREADS` | CPU threads for the forward passes (the configs assume a 64-core host) |
| `REVISE_TOKEN_BUDGET` | maximum prompt tokens per batch, default 48000; lower it if a GPU runs out of memory |
| `PYTHON`, `N_JOBS`, `KEEP_GOING`, `DRY_RUN` | read by the scripts below: interpreter, parallel probe fits, carry on after a failure, print commands without running them |

## Replicating the paper

One script per result. Each one documents the table or figure it produces,
which scripts must have run first, and the stages it runs; each is **resumable**
— every stage skips the work already in its run directory, so re-run a script
after an interruption and it continues. `DRY_RUN=1` prints the plan instead of
running it.

| script | produces | needs first |
|---|---|---|
| `run_smoke.sh` | nothing; validates the installation on a 0.5B model | — |
| `run_in_domain_results.sh` | **Table 2** (probe AUROC with nuisance and gain columns, bootstrap intervals) for the three models on MuSiQue 2-hop; also the AUROC-by-layer and cross-task transfer figures, the split-generalisation, behaviour and closed-book appendix tables | — |
| `run_baselines.sh` | **Table 3** (TF-IDF and DeBERTa text classifiers, verbalised judgement, output confidence) and the uptake-given-an-answer appendix table | in-domain |
| `run_causal_results.sh` | **Table 4** (patching, projection-out) and **Figure 1** (steering dose-response), plus the steering appendices | in-domain, transfer |
| `run_external_results.sh` | **Table 5** (HotpotQA, RAGBench, synthetic) and the per-group appendix table | in-domain |
| `run_transfer_results.sh` | the transfer appendix: 2WikiMultiHopQA and MuSiQue 3–4 hop | in-domain |
| `run_readout_analyses.sh` | the read-out appendix: state vs update representations, output-controlled probes, verbalised vs internal sufficiency | in-domain |
| `run_adaptive_results.sh` | the adaptive-retrieval appendix (every direction as a stopping rule and as a verifier) | the runs it is given |
| `run_estb.sh` | the ESTB appendix: retrieved (BM25 / E5) and LLM-rewritten evidence | in-domain |
| `export_estb_csv.py` | the evidence trajectories as CSV, for sharing or for use outside this pipeline | — |
| `make_paper_artifacts.sh` | all the LaTeX tables and PDF figures, plus `numbers.json` | whichever runs exist |
| `run_all.sh` | all of the above in dependency order (weeks of GPU time) | — |

So the shortest path to the main result is

```bash
scripts/run_in_domain_results.sh     # the long one: ~90k forward passes per model
scripts/run_baselines.sh
scripts/make_paper_artifacts.sh      # -> paper_artifacts/tables/probes_models.tex
```

`make_paper_artifacts.sh` writes to `$REVISE_PAPER_DIR` (default
`./paper_artifacts`): one `.tex` file per table, the figures as PDF, and
`tables/numbers.json`, a single dictionary holding every number quoted in the
paper. A builder whose runs are missing warns and is skipped, so running it on
a partial replication shows how far things have got.

Scripts take configs as arguments, so a single model or dataset is easy to
isolate:

```bash
scripts/run_in_domain_results.sh configs/musique_2hop_qwen7b.yaml
DRY_RUN=1 scripts/run_causal_results.sh          # what would run
KEEP_GOING=1 scripts/run_external_results.sh     # don't stop at the first failure
```

### Retrieval caches

`run_estb.sh` retrieves evidence from the wiki-18 corpus inside its `screen`
stage, so the caches have to exist first (both are one-off, CPU-only, and
shared by every later run):

```bash
python scripts/build_retrieval_index.py   # parse the corpus, build the BM25 index (hours)
python scripts/check_e5_retrieval.py      # fetch and concatenate the E5 FAISS index; reports gold recall
```

### Exporting the benchmark as CSV

The ESTB is constructed rather than downloaded, so it can also be written out as
plain data and used without this pipeline:

```bash
python scripts/export_estb_csv.py                                   # synthetic, 500 questions
python scripts/export_estb_csv.py --dataset musique --hops 2 --n-questions 200
```

This writes two files to `$REVISE_SCRATCH/estb_csv` (override with `--out-dir`):

| file | one row per | key columns |
|---|---|---|
| `estb_conditions.csv` | evidence context | `sufficient`, `condition_name`, `n_gold`, `has_false`, `paragraph_roles`, `context` (the evidence block exactly as the prompt renders it) |
| `estb_increments.csv` | transition `C_{k−1} → C_k` | `kind`, `is_decisive`, `becomes_sufficient`, `from_sufficient`/`to_sufficient`, `control_group`, `added_text` |

Both repeat the question, answer and hop count on every row, so each file stands
alone. The `synthetic` dataset is generated locally and needs neither network
access nor a model; the benchmark datasets are fetched from the Hub.

The labels in these files are the ones the construction determines —
`sufficient`, `kind`, `becomes_sufficient`, `is_decisive`. The **behavioural**
labels (uptake, correction, stability, `state_from`/`state_to`) describe what a
specific model did with the evidence, so they exist only after `collect` has run
for that model; the exporter does not invent them. Note also that `sufficient`
is true for `sufficient_plus_false`, where every gold hop is present and a
*conflicting* paragraph has been added on top.

## The pipeline

Every experiment is a `(config, stage)` pair:

```bash
python -m revise.cli <stage> -c configs/<config>.yaml
```

Stages are idempotent and resumable; `all` runs the usual sequence.

| stage | what it does | output under `$REVISE_SCRATCH/runs/<run_name>/` |
|---|---|---|
| `screen` | closed-book pass over the question pool, keep the questions the model gets wrong, build the conditions and increments | `screen.jsonl`, `questions.jsonl`, `conditions.jsonl`, `increments.jsonl` |
| `collect` | one forward pass per condition and regime; anchor activations for all layers | `behavior/<regime>.jsonl`, `acts/<regime>/` |
| `conversational` | the second protocol: "Your previous answer was A_k … new evidence …" | `behavior/conversational.jsonl`, `acts/conversational/` |
| `features` | `Δh` per increment, transition labels, nuisance covariates, raw states per condition | `features/<regime>/` |
| `probes` | per-layer logistic probes for every task, nuisance-only / residualised / gain-beyond-nuisance, cross-task transfer matrix | `probes/<regime>/<task>/<split>/` |
| `transfer` | stateless → conversational transfer; `--transfer-from <config>` for another dataset or model | `probes/transfer_*.json` |
| `steer` | additive `±α·v` steering on held-out contexts, with a random-direction control | `steer/results.json` |
| `patch` | patch the sufficient run's anchor into the insufficient run: full, ∥v, ⊥v, random, distractor, other question | `patch/results.json` |
| `ablate` | projection-out, low-dose difference-of-means steering, subspace rank | `ablate/results.json` |
| `adaptive` | stop-retrieving-when-sufficient simulation against behavioural / entropy / oracle / full baselines | `adaptive/results.json`, `adaptive/directions.json` |
| `text_baseline` | TF-IDF and DeBERTa classifiers on the same increments | `text_baseline/results.json` |
| `closedbook` | closed-book accuracy of the selected questions | `closedbook/results.json` |
| `judge` | LLM-judged correctness, for free-form answers (RAGBench) | `behavior/<regime>.judged.jsonl` |
| `variants` | LLM paraphrase / partial variants and independent cross-dataset sources (ESTB) | `variants.jsonl` |
| `retrieve` | BM25 / E5 retrieval for the question set | `retrieved_<method>.jsonl` |
| `report` | markdown summary and AUROC-by-layer plots | `report/summary.md`, `report/*.png` |

A few analyses are modules rather than stages, because they re-read a finished
run's features: `revise.analysis.bootstrap` (question-level confidence
intervals), `representation_ablation`, `behaviour_control`,
`verbal_vs_internal`, `self_verbal_baseline`, `groups` and `estb`. All take
config paths as arguments, and the replication scripts call them in the right
places.

### Configs

`configs/` holds one YAML per experiment; `revise/experiment.py::DEFAULTS` has
the full default configuration and a config only overrides what it changes.
Several configs share a `run_name` on purpose — a follow-up such as
`musique_2hop_qwen7b_dmsteer_small.yaml` writes a differently named artefact
(`ablate/results_dmsteer_small.json`) into the *same* run directory, reusing
its activations.

```
smoke.yaml                                 tiny end-to-end check
musique_2hop_{qwen7b,llama8b,mistral7b}    the main runs (Table 2)
musique_full_qwen7b, twowiki_qwen7b        3-4 hop and 2Wiki transfer targets
hotpotqa_*, ragbench_*, synthetic_*        external benchmarks (Table 5)
*_textbaseline[_full]                      text classifiers (Table 3)
*_dmsteer_small, *_steer_large/_allpos     steering follow-ups (Figure 1)
*_{e5,bm25}_{noise,full}, *_variants       ESTB evidence sources
```

The external-benchmark and ESTB configs also exist for Llama and Mistral (for
example `configs/ragbench_llama8b.yaml`). The paper reports those cells for
Qwen, which is what the scripts run by default; pass the other configs as
arguments to run them as well.

## Evidence conditions and increments

For gold hops `g_0 … g_{n−1}` (the canonical reasoning order from the dataset's
decomposition) and length-matched distractors, `trajectories.py` builds

* `C0`, `CD` (matched distractors only);
* prefix chains in canonical, reverse and random hop orders (`C_1 → C_12` …),
  the last prefix being *minimally sufficient*;
* leave-one-out contexts (n ≥ 3) and their completion;
* `C_ALL+D`, `C_ALL+R` (duplicate support), `C_ALL+F` (conflicting false
  evidence), `C_F` (the full 20-paragraph context);

and from every penultimate state `S` the **matched increments**

| increment | kind | crosses sufficiency |
|---|---|---|
| `S → S+g_last` | `decisive` | yes |
| `S → S+D` (length-matched distractor) | `distractor` | no |
| `S → S+R` (duplicate of a present gold) | `redundant` | no |
| `S → S+F` (final hop with the sub-answer swapped for a same-relation entity) | `false` | no |
| `S → S+A` (distractor carrying the answer string in a "not to be confused with" hatnote) | `answer_ctrl` | no |

New paragraphs are inserted at a random position, so increments are textually
minimal while the decisive paragraph's position is randomised. Each increment
also carries the model's behavioural transition
(`state_from/state_to ∈ {correct, wrong, insufficient}`), which gives the probe
tasks: **sufficiency**, **uptake** (decisive and used vs decisive and ignored),
**correction** (wrong→correct vs wrong→wrong), **stability** (correct→correct
vs correct→wrong) and **revision**.

The nuisance covariates regressed out and compared against are: Δtokens,
context length, Δ gold-answer log-probability, Δ entropy, Δ top-probability,
presence of the answer string (in the added text and in the full context),
TF-IDF relevance of the added passage, insertion position, and a TF-IDF-SVD
embedding of the added passage.

## Layout

```
revise/
  config.py              scratch and cache paths (everything under REVISE_SCRATCH)
  experiment.py          config defaults, merging, and the run directory layout
  cli.py                 python -m revise.cli <stage> -c configs/<config>.yaml
  prompts.py             evidence-only, evidence-augmented, closed-book, conversational templates
  eval.py                answer normalisation, lenient / strict correctness, abstention detection
  store.py               sharded float16 activation store
  data/loaders.py        MuSiQue, 2Wiki, HotpotQA, RAGBench -> Question (ordered gold hops, distractors)
  data/synthetic.py      templated controlled trajectories with fictional entities
  data/trajectories.py   evidence conditions and labelled increments
  data/retrieval.py      wiki-18 corpus, BM25 and E5 retrieval, passage labelling
  data/variants.py       LLM paraphrase / partial variants
  data/independent.py    the same fact from another dataset's paragraphs
  model/backend.py       HF backend: anchor activations, gold log-prob, greedy decoding, steer/patch hooks
  run/collect.py         closed-book screening, stateless collection, conversational protocol
  run/{steer,patch,ablate}.py   the causal interventions
  run/adaptive.py        adaptive-retrieval simulation
  run/{judge,closedbook}.py     LLM-judged correctness, closed-book accuracy
  analysis/features.py   Δh datasets, transition labels, nuisance covariates
  analysis/splits.py     question / document / answer-entity / relation-disjoint splits
  analysis/probes.py     per-layer probes, nuisance analyses, cross-task and cross-run transfer
  analysis/bootstrap.py  question-level bootstrap intervals
  analysis/*.py          the read-out, baseline, group and ESTB analyses
  analysis/report.py     per-run markdown summary and plots
configs/                 one YAML per experiment
scripts/                 setup, the replication scripts, and paper/ (tables and figures)
tests/                   trajectory invariants, splits, evaluation, planted-direction probe, tiny backend
```

## Runtime and hardware

The cost is dominated by the forward passes, each a prefill of a 300–700-token
prompt. As reported in the paper's compute appendix:

* collection for one 2-hop model (≈35k conditions) takes **about one hour on an
  H200**, or about 22 hours on 64 CPU cores at ~300 prompt tokens/s;
* the MuSiQue 3–4 hop run took 1.3 GPU-hours;
* all the causal experiments and text baselines together used about 7 GPU-hours;
* probing runs on CPU in one to two hours per model.

Both devices work — the backend selects CUDA when it is visible, and
`REVISE_DEVICE=cpu|cuda` overrides that. On a GPU set `REVISE_BATCH_SIZE=32`
(batching barely helps on CPU) and lower `REVISE_TOKEN_BUDGET` if memory is
tight. On CPU, set `REVISE_NUM_THREADS` to the number of *physical* cores of
one socket, since hyper-threads do not help GEMM throughput, and
`model.quantize: int8` cuts memory about 4× at some accuracy cost.

For a cheaper first pass, reduce `data.n_questions`: the probes are trained on
increments, so even 300 questions give several thousand labelled examples.
Every stage is deterministic given the seed (question sampling, insertion
positions, false-answer choice, splits) and resumable.

The stages that need no model are `features`, `probes`, `transfer`, `adaptive`,
`report`, every analysis module, and the TF-IDF half of `text_baseline`.

## Notes and assumptions

* Correctness is a lenient match (exact match on any alias, or the alias
  contained in a prediction at most 3× its length); `eval.correctness: strict`
  gives exact match, and `judge` replaces both with an LLM judgement.
* Activations are taken at the last prompt token, after the chat template's
  assistant header — the position that emits the first answer token — so the
  state probes are genuinely *pre-generation*.
* Layer index `l` in every artefact is the residual stream after decoder block
  `l` (0 = embeddings), before the final norm.
* Redundant evidence is a verbatim duplicate of a gold paragraph already in the
  context, since MuSiQue and 2Wiki have no paraphrased supports; the ESTB
  variants runs supply LLM paraphrases instead.
* `sufficient` is False for a `false` condition: the true gold hop is absent.
* `aux_model` in a config fixes one model for judging and for generating
  variants, so those labels are identical across the model families studied.
