"""Causal validation 2: activation patching at the anchor position.

For every decisive increment (C_{n-1} -> C_n) of a held-out question we run
the *insufficient* prompt C_{n-1} while replacing the anchor residual with

  full        h_suff
  parallel    h_ins + proj_v(h_suff - h_ins)
  orthogonal  h_ins + (Δ - proj_v Δ)
  random      h_ins + proj_r Δ                    (random unit r)
  distractor  h_ins + proj_v(h_{ins+D} - h_ins)   (matched distractor update)
  other_q     h_ins + proj_v(Δ' )                 (Δ' from a different question)

either at a single layer (``mode: single``) or at every layer from ``l``
onward (``mode: multi``, using that task's per-layer direction v_l), and
record whether the answer becomes correct / stops being INSUFFICIENT.
"""
from __future__ import annotations

import json
import logging
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from revise.analysis.probes import load_direction
from revise.analysis.splits import make_split
from revise.eval import clean_generation, is_correct, is_insufficient
from revise.experiment import Run
from revise.model.backend import HFBackend, Intervention
from revise.prompts import build_stateless
from revise.run.collect import get_backend
from revise.store import ActivationStore

log = logging.getLogger(__name__)
VARIANTS = ("none", "full", "parallel", "orthogonal", "random", "distractor", "other_q")


def _all_directions(run: Run, regime: str, task: str, mode: str):
    d = run.probes_dir(regime) / task / mode
    with open(d / "metrics.json") as f:
        m = json.load(f)
    return m["layers"], np.load(d / "directions.npy"), m["best_layer"]


def run_patch(cfg: dict, backend: Optional[HFBackend] = None) -> dict:
    run = Run(cfg)
    pc = cfg["patch"]
    regime, mode = pc["regime"], pc["split"]
    tasks = pc.get("tasks") or [pc.get("task", "state_sufficiency")]
    modes = pc.get("modes") or ["single", "multi"]
    qs = {q.qid: q for q in run.questions()}
    conds = {c.key: c for c in run.conditions()}
    store = ActivationStore(run.acts_dir(regime))
    assign = make_split(run.questions(), mode, cfg["analysis"]["test_frac"], cfg["analysis"]["seed"])
    incs = [i for i in run.increments() if i.kind == "decisive" and assign.get(i.qid) == "test"
            and f"{i.qid}::{i.from_cid}" in store and f"{i.qid}::{i.to_cid}" in store]
    by_q: Dict[str, object] = {}
    for i in incs:
        if i.qid not in by_q or i.order_tag == "canonical":
            by_q[i.qid] = i
    incs = list(by_q.values())[: pc["max_questions"]]
    dist = {i.control_group: i for i in run.increments() if i.kind == "distractor"}
    backend = backend or get_backend(cfg)
    n_layers_total = store.shape_tail[0]
    rng = random.Random(cfg["analysis"]["seed"])
    nrng = np.random.default_rng(cfg["analysis"]["seed"])
    results = {"config": pc, "runs": []}
    import os
    bs = int(os.environ.get("REVISE_BATCH_SIZE") or cfg["model"]["batch_size"])
    for task in tasks:
        layer_ids, dirs, best_layer = _all_directions(run, regime, task, mode)
        for pmode in modes:
            for start in (pc.get("layers") or [best_layer]):
                layers = [start] if pmode == "single" else [l for l in layer_ids if l >= start]
                V = {l: dirs[layer_ids.index(l)] for l in layers}
                R = {l: (lambda x: x / np.linalg.norm(x))(nrng.standard_normal(dirs.shape[1]).astype(np.float32)) for l in layers}
                per_variant: Dict[str, List[dict]] = {k: [] for k in VARIANTS}
                for b0 in tqdm(range(0, len(incs), bs), desc=f"patch {task} {pmode} L{start}"):
                    batch = incs[b0:b0 + bs]
                    msgs, golds = [], []
                    vecs: Dict[str, Dict[int, list]] = {k: {l: [] for l in layers} for k in VARIANTS}
                    for inc in batch:
                        q = qs[inc.qid]
                        H_ins = store.get(f"{inc.qid}::{inc.from_cid}", layers).astype(np.float32)
                        H_suf = store.get(f"{inc.qid}::{inc.to_cid}", layers).astype(np.float32)
                        d = dist.get(inc.control_group)
                        H_d = store.get(f"{inc.qid}::{d.to_cid}", layers).astype(np.float32) if d is not None and f"{inc.qid}::{d.to_cid}" in store else None
                        other = rng.choice([o for o in incs if o.qid != inc.qid]) if len(incs) > 1 else None
                        H_o = (store.get(f"{other.qid}::{other.to_cid}", layers).astype(np.float32) -
                               store.get(f"{other.qid}::{other.from_cid}", layers).astype(np.float32)) if other is not None else None
                        for li, l in enumerate(layers):
                            h_ins, h_suf, v, r = H_ins[li], H_suf[li], V[l], R[l]
                            delta = h_suf - h_ins
                            par = (delta @ v) * v
                            vecs["none"][l].append(h_ins); vecs["full"][l].append(h_suf)
                            vecs["parallel"][l].append(h_ins + par); vecs["orthogonal"][l].append(h_ins + delta - par)
                            vecs["random"][l].append(h_ins + (delta @ r) * r)
                            vecs["distractor"][l].append(h_ins + ((H_d[li] - h_ins) @ v) * v if H_d is not None else None)
                            vecs["other_q"][l].append(h_ins + (H_o[li] @ v) * v if H_o is not None else None)
                        msgs.append(build_stateless(regime, q.question, conds[f"{inc.qid}::{inc.from_cid}"].paragraphs))
                        golds.append(q)
                    for variant in VARIANTS:
                        keep = [j for j in range(len(batch)) if vecs[variant][layers[0]][j] is not None]
                        if not keep:
                            continue
                        iv = [] if variant == "none" else [
                            Intervention("patch", l, torch.tensor(np.stack([vecs[variant][l][j] for j in keep]))) for l in layers]
                        out = backend.run_messages([msgs[j] for j in keep], interventions=iv, capture=False)
                        for jj, j in enumerate(keep):
                            pred = clean_generation(out.generated[jj])
                            per_variant[variant].append({
                                "qid": golds[j].qid, "pred": pred,
                                "correct": is_correct(pred, golds[j].all_answers, cfg["eval"]["correctness"]),
                                "insufficient": is_insufficient(pred),
                            })
                for variant, rows in per_variant.items():
                    if not rows:
                        continue
                    res = {"task": task, "mode": pmode, "layer": start, "n_layers_patched": len(layers), "variant": variant,
                           "n": len(rows), "correct": float(np.mean([r["correct"] for r in rows])),
                           "answered": float(np.mean([not r["insufficient"] for r in rows]))}
                    results["runs"].append(res)
                    log.info("patch %s %s L%d %-11s correct=%.2f answered=%.2f (n=%d)", task, pmode, start, variant,
                             res["correct"], res["answered"], len(rows))
                run.save_json("patch/results.json", results)
    run.save_json("patch/results.json", results)
    return results
