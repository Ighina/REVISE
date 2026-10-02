"""Downstream use: adaptive-retrieval simulation from stored activations.

Each held-out question is walked along its canonical evidence trajectory
(distractors -> +g_1 -> ... -> minimal sufficient -> +distractor).  A policy
reads the stage's *state* probe score s = v·h (pre-generation) and stops when
it exceeds a threshold calibrated on training questions; we report answer
accuracy at the stopping stage and the number of retrieval rounds, against
behavioural (stop at first non-INSUFFICIENT), entropy, oracle and full-context
baselines.  No new forward passes are needed.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List

import numpy as np

from revise.analysis.features import load_features
from revise.analysis.probes import load_direction
from revise.analysis.splits import make_split
from revise.experiment import Run

log = logging.getLogger(__name__)


def _trajectories(run: Run) -> Dict[str, List[str]]:
    """qid -> ordered condition keys along the canonical chain."""
    incs = run.increments()
    chain: Dict[str, List[str]] = {}
    nxt = {}
    for i in incs:
        if i.order_tag == "canonical" and i.kind in ("gold_nondecisive", "decisive"):
            nxt[(i.qid, i.from_cid)] = i.to_cid
    post = {i.qid: i.to_cid for i in incs if i.kind == "post_distractor"}
    cd = {i.qid: i.to_cid for i in incs if i.kind == "distractor_from_none"}
    for qid in {i.qid for i in incs}:
        seq = []
        if qid in cd:
            seq.append(cd[qid])
        cur = "C0"
        while (qid, cur) in nxt:
            cur = nxt[(qid, cur)]; seq.append(cur)
        if qid in post:
            seq.append(post[qid])
        chain[qid] = [f"{qid}::{c}" for c in seq]
    return chain


def _threshold_at_precision(s: np.ndarray, y: np.ndarray, target: float) -> float:
    order = np.argsort(-s)
    s, y = s[order], y[order]
    tp = np.cumsum(y); k = np.arange(1, len(y) + 1)
    prec = tp / k
    ok = np.where(prec >= target)[0]
    return float(s[ok[-1]]) if len(ok) else float(s[0]) + 1e-6


def run_adaptive(cfg: dict) -> dict:
    run = Run(cfg)
    ac = cfg["adaptive"]
    regime, mode = ac["regime"], ac["split"]
    X, meta = load_features(run, regime, "states")
    key2i = {m["key"]: i for i, m in enumerate(meta)}
    layer, v, _ = load_direction(run, regime, "state_sufficiency", mode, ac.get("layer"))
    assign = make_split(run.questions(), mode, cfg["analysis"]["test_frac"], cfg["analysis"]["seed"])
    chains = _trajectories(run)
    scores = np.asarray(X[:, layer]).astype(np.float32) @ v
    ent = np.array([m["entropy"] for m in meta])

    def stage_rows(qids):
        rows = []
        for q in qids:
            for k, key in enumerate(chains.get(q, [])):
                if key in key2i:
                    rows.append((q, k, key2i[key]))
        return rows

    train_q = [q for q in chains if assign.get(q) == "train"]
    test_q = [q for q in chains if assign.get(q) == "test"]
    tr = stage_rows(train_q)
    y_tr = np.array([meta[i]["sufficient"] for _, _, i in tr], dtype=int)
    thr = _threshold_at_precision(np.array([scores[i] for _, _, i in tr]), y_tr, ac["target_precision"])
    ent_thr = _threshold_at_precision(-np.array([ent[i] for _, _, i in tr]), y_tr, ac["target_precision"])

    policies = {
        "probe": lambda i: scores[i] >= thr,
        "behavioral": lambda i: not meta[i]["insufficient"],
        "entropy": lambda i: -ent[i] >= ent_thr,
        "oracle": lambda i: bool(meta[i]["sufficient"]),
        "full": lambda i: False,
    }
    out = {"layer": layer, "threshold": thr, "n_test": len(test_q), "policies": {}}
    for name, stop in policies.items():
        acc, rounds, toks, stopped_early = [], [], [], []
        for q in test_q:
            keys = [key2i[k] for k in chains[q] if k in key2i]
            if not keys:
                continue
            k_stop = next((k for k, i in enumerate(keys) if stop(i)), len(keys) - 1)
            i = keys[k_stop]
            acc.append(meta[i]["correct"]); rounds.append(k_stop + 1); toks.append(meta[i]["prompt_tokens"])
            stopped_early.append(k_stop < len(keys) - 1)
        out["policies"][name] = {"accuracy": float(np.mean(acc)), "mean_rounds": float(np.mean(rounds)),
                                 "mean_prompt_tokens": float(np.mean(toks)), "n": len(acc)}
        log.info("adaptive %-10s acc=%.3f rounds=%.2f tokens=%.0f", name, out["policies"][name]["accuracy"],
                 out["policies"][name]["mean_rounds"], out["policies"][name]["mean_prompt_tokens"])
    run.save_json("adaptive/results.json", out)
    try:
        run_adaptive_directions(cfg)
    except Exception as e:   # e.g. runs without delta features / probes, or degenerate trajectories
        log.warning("adaptive directions skipped: %r", e)
    return out


# ---------------------------------------------------------------------------
# every learned direction as a stopping rule
# ---------------------------------------------------------------------------

STATE_RULES = ["state_sufficiency", "state_correct", "state_insufficient_flag"]
DELTA_RULES = ["sufficiency", "sufficiency_matched", "uptake", "correction", "stability", "revision"]


def _incoming_deltas(run: Run, regime: str, chains: Dict[str, List[str]]):
    """For every stage key on the canonical chains, the row of the delta feature matrix for the
    increment that produced it (C0->C_D, C0->C_g0, C_g0->C_g0g1, ..., C_suff->C_suff+d)."""
    Xd, dmeta = load_features(run, regime, "deltas")
    want = {"distractor_from_none", "gold_nondecisive", "decisive", "post_distractor"}
    by_to = {}
    for j, m in enumerate(dmeta):
        if m["kind"] in want and m["order_tag"] in ("canonical", "none", "post"):
            by_to.setdefault(f"{m['qid']}::{m['to_cid']}", j)
    for j, m in enumerate(dmeta):      # post-sufficiency increments may carry another order tag
        if m["kind"] in want:
            by_to.setdefault(f"{m['qid']}::{m['to_cid']}", j)
    return Xd, by_to


def _bootstrap_ci(x: np.ndarray, n: int = 1000, seed: int = 0) -> List[float]:
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def _simulate(keys_by_q: List[List[int]], stop_scores: List[np.ndarray], thr: float, meta, ci: bool = False) -> dict:
    acc, rounds, toks, answered = [], [], [], []
    for keys, sc in zip(keys_by_q, stop_scores):
        hit = np.where(sc >= thr)[0]
        k = int(hit[0]) if len(hit) else len(keys) - 1
        i = keys[k]
        acc.append(meta[i]["correct"]); rounds.append(k + 1); toks.append(meta[i]["prompt_tokens"]); answered.append(not meta[i]["insufficient"])
    out = {"accuracy": float(np.mean(acc)), "mean_rounds": float(np.mean(rounds)), "mean_prompt_tokens": float(np.mean(toks)),
           "answered": float(np.mean(answered)), "n": len(acc)}
    if ci:
        out["accuracy_ci"] = _bootstrap_ci(np.array(acc, dtype=float)); out["_correct"] = [bool(a) for a in acc]
    return out


def run_adaptive_directions(cfg: dict) -> dict:
    """Use each learned direction as the stopping rule of the adaptive-retrieval simulation.

    At round k the policy reads either the state h(C_k) (state probes) or the update the round
    produced, Delta h = h(C_k) - h(C_prev) (delta probes), projects it on the direction and stops
    when the projection crosses a threshold; otherwise it retrieves the next unit.  Threshold *and
    sign* are chosen on training questions to maximise accuracy at the stopping stage (ties: fewer
    rounds), so a direction may be used either way round (e.g. stop when the stability probe says the
    last update *destabilised* the answer, i.e. the answer changed).  A threshold sweep on test
    questions gives each direction's accuracy / rounds trade-off curve.
    """
    run = Run(cfg)
    ac = cfg["adaptive"]
    regime, mode = ac["regime"], ac["split"]
    Xs, smeta = load_features(run, regime, "states")
    key2i = {m["key"]: i for i, m in enumerate(smeta)}
    chains = _trajectories(run)
    Xd, by_to = _incoming_deltas(run, regime, chains)
    assign = make_split(run.questions(), mode, cfg["analysis"]["test_frac"], cfg["analysis"]["seed"])
    qids = {sp: [q for q in chains if assign.get(q) == sp and any(k in key2i for k in chains[q])] for sp in ("train", "test")}

    def stages(q):
        return [k for k in chains[q] if k in key2i and k in by_to]

    rules = {}
    for task in STATE_RULES + DELTA_RULES:
        for kind in ("lr", "dm") if task in ("state_sufficiency", "sufficiency") else ("lr",):
            try:
                layer, v, _ = load_direction(run, regime, task, mode, kind=kind)
            except FileNotFoundError:
                continue
            if task in STATE_RULES:
                f = (lambda layer, v: (lambda ks: np.asarray(Xs[[key2i[k] for k in ks], layer]).astype(np.float32) @ v))(layer, v)
            else:
                f = (lambda layer, v: (lambda ks: np.asarray(Xd[[by_to[k] for k in ks], layer]).astype(np.float32) @ v))(layer, v)
            rules[f"{task}" + ("" if kind == "lr" else "_dm")] = (task, kind, int(layer), f)

    ent = np.array([m["entropy"] for m in smeta])
    base = {"behavioral": lambda ks: np.array([0.0 if smeta[key2i[k]]["insufficient"] else 1.0 for k in ks]),
            "oracle": lambda ks: np.array([1.0 if smeta[key2i[k]]["sufficient"] else 0.0 for k in ks]),
            "full": lambda ks: np.zeros(len(ks)),
            "entropy": lambda ks: -ent[[key2i[k] for k in ks]]}
    out = {"split": mode, "n_train": len(qids["train"]), "n_test": len(qids["test"]), "policies": {}, "curves": {}}
    for name, spec in list(rules.items()) + [(k, None) for k in base]:
        f = spec[3] if spec else base[name]
        sc = {sp: [f(stages(q)) for q in qids[sp]] for sp in ("train", "test")}
        ks = {sp: [[key2i[k] for k in stages(q)] for q in qids[sp]] for sp in ("train", "test")}
        keep = {sp: [j for j, k in enumerate(ks[sp]) if k] for sp in ("train", "test")}
        sc = {sp: [sc[sp][j] for j in keep[sp]] for sp in sc}; ks = {sp: [ks[sp][j] for j in keep[sp]] for sp in ks}
        if name in ("behavioral", "oracle", "full"):
            best = (1.0, 0.5)
        else:
            per_sign = {}
            for sign in (1.0, -1.0):          # stop when sign * score >= t
                tr_sc = [sign * x for x in sc["train"]]
                cands = np.append(np.unique(np.quantile(np.concatenate(tr_sc), np.linspace(0, 1, 201))), np.inf)
                b, b_obj = None, None
                for t in cands:
                    r = _simulate(ks["train"], tr_sc, float(t), smeta)
                    obj = (round(r["accuracy"], 6), -r["mean_rounds"])
                    if b_obj is None or obj > b_obj:
                        b_obj, b = obj, float(t)
                per_sign[sign] = (b_obj, b)
            sign = max(per_sign, key=lambda k: per_sign[k][0])
            best = (sign, per_sign[sign][1])
        sign, thr = best
        res = _simulate(ks["test"], [sign * x for x in sc["test"]], thr, smeta, ci=True)
        res.update({"sign": sign, "threshold": thr})
        if name not in ("behavioral", "oracle", "full"):
            o = -sign   # the other reading of the same direction, also calibrated on training questions
            alt = _simulate(ks["test"], [o * x for x in sc["test"]], per_sign[o][1], smeta, ci=True)
            alt.pop("_correct"); res["other_sign"] = {**alt, "sign": o, "threshold": per_sign[o][1]}
        if spec:
            res.update({"task": spec[0], "kind": spec[1], "layer": spec[2], "signal": "state" if spec[0] in STATE_RULES else "delta"})
        out["policies"][name] = res
        if name not in ("behavioral", "oracle", "full"):
            allt = np.concatenate([sign * x for x in sc["test"]])
            curve = [_simulate(ks["test"], [sign * x for x in sc["test"]], float(t), smeta) for t in np.unique(np.quantile(allt, np.linspace(0, 1, 41)))]
            out["curves"][name] = [(c["mean_rounds"], c["accuracy"]) for c in curve]
        log.info("adaptive[dir] %-24s sign=%+d acc=%.3f rounds=%.2f answered=%.2f", name, int(sign), res["accuracy"], res["mean_rounds"], res["answered"])
    # verification: once retrieval stops, does a direction tell whether the answer given is correct?
    from sklearn.metrics import roc_auc_score
    out["verification"] = {}
    stop_rules = {"oracle": base["oracle"]}
    if "state_sufficiency" in rules:
        stop_rules["state_sufficiency"] = rules["state_sufficiency"][3]
    for sname, sf in stop_rules.items():
        sign, thr = out["policies"][sname]["sign"], out["policies"][sname]["threshold"]
        stop_keys = []
        for q in qids["test"]:
            st = stages(q)
            if not st:
                continue
            hit = np.where(sign * sf(st) >= thr)[0]
            stop_keys.append(st[int(hit[0]) if len(hit) else len(st) - 1])
        y = np.array([smeta[key2i[k]]["correct"] for k in stop_keys], dtype=int)
        ans = np.array([not smeta[key2i[k]]["insufficient"] for k in stop_keys])
        ver = {"n": len(y), "n_answered": int(ans.sum()), "accuracy_answered": float(y[ans].mean()) if ans.any() else None, "auroc": {}, "selective_acc_50": {}}
        sig = {n: spec[3] for n, spec in rules.items()}
        sig["top_prob"] = lambda ks: np.array([smeta[key2i[k]]["top_prob"] for k in ks])
        sig["neg_entropy"] = lambda ks: -ent[[key2i[k] for k in ks]]
        for n, f in sig.items():
            sc = f(stop_keys)
            yy, ss = y[ans], sc[ans]           # among answered stops: is the answer correct?
            if len(np.unique(yy)) == 2:
                ver["auroc"][n] = float(roc_auc_score(yy, ss))
                top = np.argsort(-ss)[: max(1, len(ss) // 2)]
                ver["selective_acc_50"][n] = float(yy[top].mean())
        out["verification"][sname] = ver
        log.info("verification at %s stop: %s", sname, {k: round(v, 3) for k, v in ver["auroc"].items()})

    ref = np.array(out["policies"]["behavioral"]["_correct"], dtype=float)
    rng = np.random.default_rng(1)
    boots = [rng.integers(0, len(ref), len(ref)) for _ in range(1000)]
    for name, r in out["policies"].items():
        x = np.array(r.pop("_correct"), dtype=float)
        if len(x) == len(ref):
            d = [x[b].mean() - ref[b].mean() for b in boots]
            r["diff_vs_behavioral_ci"] = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
    run.save_json("adaptive/directions.json", out)
    return out
