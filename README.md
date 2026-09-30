# REVISE

Code for the research plan in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md):

> As evidence becomes sufficient to answer a question, does an LLM undergo a
> shared internal state transition that predicts successful knowledge
> accommodation *before* the answer is generated?

The pipeline builds semantically-defined **evidence trajectories** for
multi-hop questions (MuSiQue / 2WikiMultiHopQA), runs a stateless forward pass
per evidence stage, records the residual stream at the answer-start anchor for
every layer, and then trains linear probes on **paired activation updates**
`Δh = h(C_k) − h(C_{k−1})` against matched textual controls, followed by
causal validation (steering, activation patching) and an adaptive-retrieval
simulation.

Everything runs on **CPU**: the backend is Hugging Face Transformers in
bfloat16 with forward hooks (llama.cpp would be faster but offers no clean
per-layer read/write access from Python, which steering and patching need).
On this machine (2×64-core AMD EPYC 9555 with AVX-512 bf16, 755 GB RAM) a 7B
model is workable; see *Runtime* below.

## Layout

```
revise/
  config.py            scratch/cache paths (everything under /scratch/users/iacopog/revise)
  data/loaders.py      MuSiQue + 2Wiki -> Question (ordered gold hops, distractors, decomposition)
  data/trajectories.py evidence conditions + labelled increments (decisive / distractor / redundant / false / answer-string)
  prompts.py           evidence-only, evidence-augmented, closed-book, conversational-revision templates
  model/backend.py     CPU HF backend: anchor activations (all layers), gold log-prob, greedy decoding, steer/patch hooks
  store.py             sharded float16 activation store
  run/collect.py       closed-book screening, stateless collection, conversational ablation (all resumable)
  analysis/features.py Δh datasets + transition labels + nuisance covariates
  analysis/splits.py   question / document / answer-entity / relation-disjoint splits
  analysis/probes.py   per-layer probes, nuisance/residual analyses, cross-task + cross-regime/dataset transfer
  run/steer.py         additive steering with random-direction control and false/answer-string/distractor sets
  run/patch.py         anchor patching: full / parallel / orthogonal / random / distractor / other-question
  run/adaptive.py      adaptive-retrieval simulation from the pre-generation state probe
  analysis/report.py   markdown summary + AUROC-by-layer plots
  cli.py               `python -m revise.cli <stage> -c configs/<cfg>.yaml`
configs/               smoke (0.5B), musique_2hop_qwen7b, musique_full_qwen7b, twowiki_qwen7b, musique_2hop_llama8b
scripts/               setup_env.sh, link_shared_cache.sh, run_smoke.sh, run_main.sh
tests/                 unit tests (trajectory invariants, splits, eval, planted-direction probe, tiny backend)
```

## Setup

```bash
scripts/setup_env.sh                       # conda env "revise" (python 3.11, CPU torch)
conda activate revise
export REVISE_SCRATCH=/scratch/users/iacopog/revise   # default; weights, data, runs all live here
scripts/link_shared_cache.sh               # optional: symlink already-downloaded 7B/8B weights instead of re-downloading
python -m pytest -q                        # 11 tests, ~10 s
```

`revise.config` forces `HF_HOME` to `$REVISE_SCRATCH/hf_cache` (set
`REVISE_KEEP_HF_ENV=1` to keep a pre-existing `HF_HOME`).

## Running

```bash
scripts/run_smoke.sh                                   # whole pipeline on Qwen2.5-0.5B, ~10 min
scripts/run_main.sh configs/musique_2hop_qwen7b.yaml   # Phase 1 (resumable; re-run to continue)
```

Stages (each is idempotent and resumable):

| stage | what it does | output (under `runs/<run_name>/`) |
|---|---|---|
| `screen` | closed-book pass over the question pool; keep questions the model gets wrong; build conditions + increments | `screen.jsonl`, `questions.jsonl`, `conditions.jsonl`, `increments.jsonl` |
| `collect` | stateless forward pass for every condition and regime; anchor activations for all layers | `behavior/<regime>.jsonl`, `acts/<regime>/` |
| `conversational` | secondary protocol: "Your previous answer was A_k … new evidence …" | `behavior/conversational.jsonl`, `acts/conversational/` |
| `features` | Δh per increment, transition labels, nuisance covariates; raw states per condition | `features/<regime>/` |
| `probes` | per-layer LR probes for the four transition tasks (+ revision, + state probes), nuisance-only / residualised / gain-beyond-nuisance, cross-task transfer matrix | `probes/<regime>/<task>/<split>/` |
| `transfer` | stateless → conversational direction transfer; `--transfer-from <other cfg>` for dataset/model transfer | `probes/transfer_*.json` |
| `steer` | ±α·v steering on held-out sufficient / insufficient / false / answer-string / distractor contexts, with random-direction control | `steer/results.json` |
| `patch` | patch sufficient-run anchor into insufficient run: full, ∥v, ⊥v, random dir, distractor update, other question | `patch/results.json` |
| `adaptive` | stop-retrieving-when-sufficient simulation vs behavioural / entropy / oracle / full baselines | `adaptive/results.json` |
| `report` | markdown tables + plots | `report/summary.md`, `report/*.png` |

## Evidence conditions and increments

For gold hops `g_0 … g_{n−1}` (canonical reasoning order from the dataset's
decomposition) and length-matched distractors, `trajectories.py` builds

* `C0`, `CD` (matched distractors only)
* prefix chains in canonical, reverse and random hop orders (`C_1 → C_12` …), the last prefix being *minimally sufficient*
* leave-one-out contexts (n ≥ 3) and their completion
* `C_ALL+D`, `C_ALL+R` (duplicate support), `C_ALL+F` (conflicting false evidence), `C_F` (full 20-paragraph context)

and from every penultimate state `S` the **matched increments**

| increment | kind | crosses sufficiency |
|---|---|---|
| `S → S+g_last` | `decisive` | yes |
| `S → S+D` (length-matched distractor) | `distractor` | no |
| `S → S+R` (duplicate of a present gold) | `redundant` | no |
| `S → S+F` (final hop with the sub-answer swapped for a same-relation entity) | `false` | no |
| `S → S+A` (distractor carrying the answer string in a "not to be confused with" hatnote) | `answer_ctrl` | no |

New paragraphs are inserted at a random position so that increments are
textually minimal while decisive-paragraph position is randomised.  Each
increment additionally carries the behavioural transition of the model
(`state_from/state_to ∈ {correct, wrong, insufficient}`), giving the four
probe tasks of the plan: **sufficiency**, **uptake** (decisive & correct vs
decisive & ignored), **correction** (wrong→correct vs wrong→wrong) and
**stability** (correct→correct vs correct→wrong), plus **revision**.

Nuisance covariates regressed out / compared against: Δtokens, context
length, Δ gold-answer log-prob, Δ entropy, Δ top-prob, answer-string presence
(in added text / in full context), TF-IDF relevance of the added passage,
insertion position, passage TF-IDF-SVD embedding.

## Runtime on CPU

Per-condition cost is dominated by the prefill of a ~300–700-token prompt.
Measured throughput is logged by `collect` (`prompt tok/s`).  Rough budget for
Phase 1 (2,000 questions × ~17 conditions × 2 regimes + ~10 conversational
increments each ≈ 90k forward passes):

* Qwen2.5-0.5B: the whole smoke pipeline takes ~10 minutes.
* Qwen2.5-7B bf16, 64 threads (measured on this machine): model load 4 s,
  prefill ≈ 300–350 prompt tokens/s, **≈ 1.3 s per condition** (batch 1 or 8
  are nearly identical on CPU).  Full Phase 1 (~90k passes) ≈ 30–35 h of wall
  clock; it is resumable, so it can be run in chunks.  Reduce
  `data.n_questions` (e.g. 500 → ~8 h) for a first pass; probes are trained on
  Δh so even 300 questions give several thousand labelled increments.

Set `model.num_threads` to the number of *physical* cores of one socket
(64 here); hyper-threads do not help GEMM throughput.  `model.quantize: int8`
(dynamic int8 for `nn.Linear`) cuts memory ~4× at some accuracy cost.

## Notes / assumptions

* Correctness uses a lenient match (exact match on any alias, or the alias
  contained in a prediction ≤ 3× its length); `eval.correctness: strict` for EM.
* Activations are stored at the last prompt token (after the chat template's
  assistant header), i.e. the position that emits the first answer token, so
  the state probes are genuinely *pre-generation*.
* Layer index `l` in every artifact = residual stream after decoder block `l`
  (0 = embeddings), before the final norm.
* Redundant evidence is a verbatim duplicate of a present gold paragraph
  (MuSiQue/2Wiki have no paraphrased supports).  Replace `redundant_copy` in
  `trajectories.py` with a paraphrase source if desired.
* `sufficient` for a `false` condition is False (the true gold hop is absent).

## GPU runs

The backend auto-selects CUDA when available (`model.device: auto`; override
with `REVISE_DEVICE=cpu|cuda`).  `REVISE_BATCH_SIZE` overrides the batch size
at run time (32 is a good GPU default; batching barely helps on CPU).  The
`revise` env carries a CUDA 12.8 build of PyTorch that also runs on CPU.

This host shares GPUs through `gpuq` (rolling 7-day GPU-hour budget, jobs
capped at 48 h, stacking on a card you already hold is allowed with `-m`).
`scripts/gpu_run.sh <config> <stage> [stage ...]` submits a sequence of
pipeline stages as one gpuq job with the env activated and `HF_HUB_OFFLINE=1`
(every model/dataset used is cached under scratch; Llama-3.1 is gated and
must be loaded offline):

```bash
HOURS=6 BATCH=32 GPUQ_ARGS="--queue -m 20" scripts/gpu_run.sh configs/musique_2hop_llama8b.yaml screen collect features
GPUQ_ARGS="--devices 3 -m 20" scripts/gpu_run.sh configs/musique_2hop_qwen7b.yaml ablate      # stack on your own card
```

CPU-only stages (no model forward passes): `features`, `probes`, `transfer`,
`adaptive`, `report`, and the TF-IDF part of `text_baseline`.
