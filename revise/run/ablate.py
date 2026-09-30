"""Causal validation 3 (follow-ups requested after Phase 1):

  projout   : on *sufficient* prompts, set the component along a direction to
              the insufficient-state mean (or zero), at every layer from l
              onward, anchor position onward.  Does the model start abstaining?
              Controls: random direction; also run on false/insufficient sets.
  dm_steer  : difference-of-means steering at all layers from l onward and all
              positions (the standard representation-engineering recipe).
  subspace  : patch h_ins + P_k(h_suf - h_ins) at every layer from l onward,
              where P_k projects on the top-k principal components of training
              deltas at that layer, for k in ``ranks``.  Gives the rank of the
              causal subspace.  ``unembed`` variants add / remove only the
              component along the gold answer's first-token unembedding.
"""
from __future__ import annotations

import json
import logging
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from revise.analysis.features import load_features
from revise.analysis.probes import load_direction
from revise.analysis.splits import make_split
from revise.eval import clean_generation, is_correct, is_insufficient, normalize_answer
from revise.experiment import Run
from revise.model.backend import HFBackend, Intervention
from revise.prompts import build_stateless
from revise.run.collect import get_backend
from revise.run.steer import EVAL_SETS, _pick_eval_conditions, _score, state_gap
from revise.store import ActivationStore

log = logging.getLogger(__name__)


def _dirs(run: Run, regime: str, task: str, mode: str, kind: str = "lr"):
    d = run.probes_dir(regime) / task / mode
    with open(d / "metrics.json") as f:
        m = json.load(f)
    return m["layers"], np.load(d / ("directions.npy" if kind == "lr" else "dm_directions.npy")), m["best_layer"]


def _eval_rows(backend, cfg, regime, qs, conds, iv, bs, desc):
    rows = []
    for i in tqdm(range(0, len(conds), bs), desc=desc, leave=False):
        batch = conds[i:i + bs]
        msgs = [build_stateless(regime, qs[c.qid].question, c.paragraphs) for c in batch]
        out = backend.run_messages(msgs, interventions=iv, capture=False)
        for j, c in enumerate(batch):
            q = qs[c.qid]; pred = clean_generation(out.generated[j])
            fa = next((p.meta.get("false_answer") for p in c.paragraphs if p.role == "false"), None)
            rows.append({"qid": c.qid, "pred": pred, "correct": is_correct(pred, q.all_answers, cfg["eval"]["correctness"]),
                         "insufficient": is_insufficient(pred),
                         "false_follow": (normalize_answer(fa) in normalize_answer(pred)) if fa else None})
    return rows


def run_ablate(cfg: dict, backend: Optional[HFBackend] = None) -> dict:
    import os
    run = Run(cfg)
    ac = cfg["ablate"]
    regime, mode = ac["regime"], ac["split"]
    bs = int(os.environ.get("REVISE_BATCH_SIZE") or cfg["model"]["batch_size"])
    qs = {q.qid: q for q in run.questions()}
    sets = _pick_eval_conditions(run, mode, cfg, ac["max_questions"])
    backend = backend or get_backend(cfg)
    nrng = np.random.default_rng(cfg["analysis"]["seed"])
    results = {"config": ac, "runs": []}
    experiments = ac.get("experiments", ["projout", "dm_steer", "subspace"])

    # ------------------------------------------------------------ projout
    if "projout" in experiments:
        for task in ac.get("projout_tasks", ["state_sufficiency", "uptake"]):
            for dkind in ("lr", "dm"):
                layer_ids, D, best = _dirs(run, regime, task, mode, dkind)
                layers = [l for l in layer_ids if l >= best] if ac.get("multi_layer", True) else [best]
                # per-layer target = insufficient-state mean along that layer's direction
                targets = {l: state_gap(run, regime, task, mode, l, D[layer_ids.index(l)], cfg)["neg_mean"] for l in layers}
                for dname in ("probe", "random"):
                    for set_name in ac.get("projout_sets", ["sufficient", "false", "insufficient"]):
                        conds = sets[set_name]
                        iv = []
                        for l in layers:
                            v = D[layer_ids.index(l)]
                            if dname == "random":
                                r = nrng.standard_normal(v.shape).astype(np.float32); v = r / np.linalg.norm(r)
                            iv.append(Intervention("project", l, torch.tensor(v), positions=ac.get("positions", "anchor_onward"),
                                                   target=float(targets[l]) if dname == "probe" else 0.0))
                        rows = _eval_rows(backend, cfg, regime, qs, conds, iv, bs, f"projout {task}/{dkind} {dname} {set_name}")
                        res = {"experiment": "projout", "task": task, "direction_kind": dkind, "direction": dname, "layer": best,
                               "n_layers": len(layers), "set": set_name, **_score(rows)}
                        results["runs"].append(res); run.save_json(ac.get("out", "ablate/results.json"), results)
                        log.info("projout %s/%s %-6s L%d+ %-12s correct=%.2f answered=%.2f", task, dkind, dname, best, set_name,
                                 res["correct"] or 0, res["answered"] or 0)

    # ------------------------------------------------------------ dm_steer
    if "dm_steer" in experiments:
        task = "state_sufficiency"
        layer_ids, D, best = _dirs(run, regime, task, mode, "dm")
        layers = [l for l in layer_ids if l >= best][: ac.get("dm_max_layers", 999)]
        units = {l: state_gap(run, regime, task, mode, l, D[layer_ids.index(l)], cfg) for l in layers}
        log.info("dm_steer layers %s, per-layer gap L2: %s", layers, {l: round(units[l]["gap"], 2) for l in layers})
        for dname in ("probe", "random"):
            for alpha in ac.get("dm_alphas", [-4, -2, 2, 4, 8]):
                iv = []
                for l in layers:
                    v = D[layer_ids.index(l)]; g = units[l]["gap"]
                    if dname == "random":
                        r = nrng.standard_normal(v.shape).astype(np.float32); v = r / np.linalg.norm(r)
                    iv.append(Intervention("steer", l, torch.tensor(v * float(alpha) * abs(g) * (1 if g >= 0 else -1)),
                                           positions=ac.get("dm_positions", "all")))
                for set_name, conds in sets.items():
                    rows = _eval_rows(backend, cfg, regime, qs, conds, iv, bs, f"dm_steer {dname} a={alpha} {set_name}")
                    res = {"experiment": "dm_steer", "task": task, "direction": dname, "alpha": alpha, "layer": best,
                           "n_layers": len(layers), "set": set_name, **_score(rows)}
                    results["runs"].append(res); run.save_json(ac.get("out", "ablate/results.json"), results)
                    log.info("dm_steer %-6s a=%+.1f L%d+ %-15s correct=%.2f answered=%.2f false=%s", dname, alpha, best, set_name,
                             res["correct"] or 0, res["answered"] or 0, res["false_follow"])

    # ------------------------------------------------------------ subspace rank
    if "subspace" in experiments:
        store = ActivationStore(run.acts_dir(regime))
        conds_all = {c.key: c for c in run.conditions()}
        assign = make_split(run.questions(), mode, cfg["analysis"]["test_frac"], cfg["analysis"]["seed"])
        dec = [i for i in run.increments() if i.kind == "decisive" and i.order_tag == "canonical"
               and f"{i.qid}::{i.from_cid}" in store and f"{i.qid}::{i.to_cid}" in store]
        train = [i for i in dec if assign.get(i.qid) == "train"]
        test = [i for i in dec if assign.get(i.qid) == "test"][: ac["max_questions"]]
        start = ac.get("subspace_from_layer") or _dirs(run, regime, "state_sufficiency", mode)[2]
        n_layers_total = store.shape_tail[0]
        layers = list(range(start, n_layers_total))
        ranks = ac.get("ranks", [1, 4, 16, 64, 256])
        # PCA of training deltas per layer
        rng = random.Random(0); sub = train[:] ; rng.shuffle(sub); sub = sub[: ac.get("pca_train", 1500)]
        deltas = np.stack([store.get(f"{i.qid}::{i.to_cid}", layers).astype(np.float32) -
                           store.get(f"{i.qid}::{i.from_cid}", layers).astype(np.float32) for i in sub])  # [n, L, d]
        comps: Dict[int, np.ndarray] = {}
        for li, l in enumerate(layers):
            Xl = deltas[:, li] - deltas[:, li].mean(0)
            _, _, Vt = np.linalg.svd(Xl, full_matrices=False)
            comps[l] = Vt[: max(ranks)]
        # answer-token unembedding directions
        W_U = backend.model.get_output_embeddings().weight.detach().float().cpu().numpy()
        variants = ["none", "full"] + [f"pca{k}" for k in ranks] + [f"pca{k}_rand" for k in ranks[-1:]] + ["unembed_add", "unembed_remove"]
        per: Dict[str, List[dict]] = {v: [] for v in variants}
        for b0 in tqdm(range(0, len(test), bs), desc=f"subspace L{start}+"):
            batch = test[b0:b0 + bs]
            msgs = [build_stateless(regime, qs[i.qid].question, conds_all[f"{i.qid}::{i.from_cid}"].paragraphs) for i in batch]
            H_ins = np.stack([store.get(f"{i.qid}::{i.from_cid}", layers).astype(np.float32) for i in batch])
            H_suf = np.stack([store.get(f"{i.qid}::{i.to_cid}", layers).astype(np.float32) for i in batch])
            Dl = H_suf - H_ins
            ans_vecs = []
            for i in batch:
                tok = backend.tok(qs[i.qid].answer, add_special_tokens=False)["input_ids"][0]
                u = W_U[tok]; ans_vecs.append(u / (np.linalg.norm(u) + 1e-9))
            U = np.stack(ans_vecs)  # [B, d]
            for variant in variants:
                iv = []
                if variant != "none":
                    for li, l in enumerate(layers):
                        if variant == "full":
                            vec = H_suf[:, li]
                        elif variant.startswith("pca"):
                            k = int(variant.replace("pca", "").replace("_rand", ""))
                            P = comps[l][:k]
                            if variant.endswith("_rand"):
                                R = nrng.standard_normal((k, P.shape[1])).astype(np.float32); P, _ = np.linalg.qr(R.T); P = P.T
                            vec = H_ins[:, li] + (Dl[:, li] @ P.T) @ P
                        elif variant == "unembed_add":
                            vec = H_ins[:, li] + ((Dl[:, li] * U).sum(1, keepdims=True)) * U
                        elif variant == "unembed_remove":
                            vec = H_suf[:, li] - ((Dl[:, li] * U).sum(1, keepdims=True)) * U
                        iv.append(Intervention("patch", l, torch.tensor(vec)))
                out = backend.run_messages(msgs, interventions=iv, capture=False)
                for j, i in enumerate(batch):
                    pred = clean_generation(out.generated[j])
                    per[variant].append({"correct": is_correct(pred, qs[i.qid].all_answers, cfg["eval"]["correctness"]),
                                         "insufficient": is_insufficient(pred)})
        for variant, rows in per.items():
            res = {"experiment": "subspace", "variant": variant, "layer": start, "n_layers": len(layers), "n": len(rows),
                   "correct": float(np.mean([r["correct"] for r in rows])), "answered": float(np.mean([not r["insufficient"] for r in rows]))}
            results["runs"].append(res)
            log.info("subspace L%d+ %-14s correct=%.3f answered=%.3f", start, variant, res["correct"], res["answered"])
        run.save_json(ac.get("out", "ablate/results.json"), results)
    return results
