"""Linear probes on paired activation updates (and on raw states).

For every task, split mode and layer we fit a logistic-regression probe and
report held-out AUROC / AP.  At the layer selected by *inner* grouped CV on
the training questions we additionally run the nuisance analyses:

  * nuisance-only baseline (token count, answer-string presence, relevance,
    logit change, passage embedding, ...)
  * residualised probe (nuisance regressed out of Δh before probing)
  * "beyond nuisance": AUROC gain from adding the probe score to the nuisance
    model (probe scores on train are out-of-fold)

and the cross-task transfer matrix (does one direction predict all tasks in
the expected ordering?).
"""
from __future__ import annotations

import json
import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from revise.analysis.features import DECISIVE_KINDS, load_features
from revise.analysis.splits import make_split
from revise.experiment import Run

log = logging.getLogger(__name__)

CONTROL_KINDS = ("distractor", "redundant", "false", "answer_ctrl", "partial", "false_variant")
NONDECISIVE_KINDS = ("gold_nondecisive", "distractor_from_none") + CONTROL_KINDS + \
    ("post_distractor", "post_redundant", "post_false", "post_redundant_paraphrase", "post_redundant_independent", "full_context")
WRONG_STATES = ("wrong", "insufficient")

# task -> (selector, label) over delta-meta rows
DELTA_TASKS: Dict[str, Tuple[Callable[[dict], bool], Callable[[dict], int]]] = {
    # 1. sufficient vs still-insufficient increments (all non-decisive increments as negatives)
    "sufficiency": (lambda m: m["kind"] in DECISIVE_KINDS + NONDECISIVE_KINDS, lambda m: int(m["becomes_sufficient"])),
    # 1b. decisive vs *matched* controls from the same penultimate state
    "sufficiency_matched": (lambda m: m["kind"] in DECISIVE_KINDS + CONTROL_KINDS, lambda m: int(m["becomes_sufficient"])),
    # 2. evidence incorporated vs behaviourally ignored (decisive increments only)
    "uptake": (lambda m: m["kind"] in DECISIVE_KINDS, lambda m: int(m["state_to"] == "correct")),
    # 3. wrong-to-correct vs wrong-to-wrong
    "correction": (lambda m: m["state_from"] in WRONG_STATES, lambda m: int(m["state_to"] == "correct")),
    # 4. stable-correct vs destabilised-correct
    "stability": (lambda m: m["state_from"] == "correct", lambda m: int(m["state_to"] == "correct")),
    # 5. any answer revision
    "revision": (lambda m: True, lambda m: int(m["revision"])),
}
STATE_TASKS: Dict[str, Tuple[Callable[[dict], bool], Callable[[dict], int]]] = {
    "state_sufficiency": (lambda m: not m["has_false"], lambda m: int(m["sufficient"])),
    "state_correct": (lambda m: True, lambda m: int(m["correct"])),
    "state_insufficient_flag": (lambda m: True, lambda m: int(m["insufficient"])),
}
NUISANCE_COLS = ["d_tokens", "to_tokens", "d_gold_lp", "d_entropy", "d_top_prob", "answer_in_added", "answer_in_to",
                 "answer_in_from", "relevance", "n_added", "added_wc", "inserted_pos_rel"]


def _safe_auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def _safe_ap(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, s))


def _fit_lr(X: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    return LogisticRegression(C=C, class_weight="balanced", max_iter=1000, tol=1e-3).fit(X, y)


class _Std:
    def __init__(self, X: np.ndarray):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd


def _nuisance_matrix(meta: List[dict], idx: np.ndarray, svd: Optional[Tuple[TfidfVectorizer, TruncatedSVD]]) -> np.ndarray:
    Z = np.array([[float(meta[i].get(c) or 0.0) for c in NUISANCE_COLS] for i in idx], dtype=np.float32)
    if svd is not None:
        vec, red = svd
        E = red.transform(vec.transform([meta[i].get("added_text", "") for i in idx])).astype(np.float32)
        Z = np.concatenate([Z, E], axis=1)
    return Z


def _fit_svd(meta: List[dict], idx: np.ndarray, k: int) -> Optional[Tuple[TfidfVectorizer, TruncatedSVD]]:
    texts = [meta[i].get("added_text", "") for i in idx]
    if len(texts) < k + 2 or not any(texts):
        return None
    vec = TfidfVectorizer(stop_words="english", min_df=2).fit(texts)
    M = vec.transform(texts)
    if M.shape[1] <= k:
        return None
    return vec, TruncatedSVD(n_components=k, random_state=0).fit(M)


def _layer_eval(Xl_tr, y_tr, g_tr, Xl_te, y_te, C: float, n_inner: int) -> dict:
    std = _Std(Xl_tr)
    A, B = std(Xl_tr), std(Xl_te)
    # inner grouped CV on train for layer selection
    inner = []
    n_groups = len(np.unique(g_tr))
    if n_groups >= n_inner and len(np.unique(y_tr)) == 2:
        for tr, va in GroupKFold(n_splits=n_inner).split(A, y_tr, g_tr):
            if len(np.unique(y_tr[tr])) < 2 or len(np.unique(y_tr[va])) < 2:
                continue
            m = _fit_lr(A[tr], y_tr[tr], C)
            inner.append(roc_auc_score(y_tr[va], m.decision_function(A[va])))
    clf = _fit_lr(A, y_tr, C)
    w = clf.coef_[0] / std.sd
    w_unit = w / (np.linalg.norm(w) + 1e-9)
    s_te = clf.decision_function(B)
    dm = Xl_tr[y_tr == 1].mean(0) - Xl_tr[y_tr == 0].mean(0)
    dm_unit = dm / (np.linalg.norm(dm) + 1e-9)
    return {
        "cv_auc": float(np.mean(inner)) if inner else None,
        "test_auc": _safe_auc(y_te, s_te), "test_ap": _safe_ap(y_te, s_te),
        "dm_test_auc": _safe_auc(y_te, Xl_te @ dm_unit),
        "direction": w_unit.astype(np.float32), "dm_direction": dm_unit.astype(np.float32),
        "proj_std": float((Xl_tr @ w_unit).std()), "intercept": float(clf.intercept_[0]),
        "coef_scaled": (clf.coef_[0] / std.sd).astype(np.float32), "mu": std.mu.astype(np.float32),
    }


def _nuisance_analysis(Xl_tr, y_tr, g_tr, Xl_te, y_te, Z_tr, Z_te, C: float) -> dict:
    out = {}
    zs = _Std(Z_tr); Ztr, Zte = zs(Z_tr), zs(Z_te)
    nuis = _fit_lr(Ztr, y_tr, C)
    out["nuisance_only_auc"] = _safe_auc(y_te, nuis.decision_function(Zte))
    # residualised probe
    ridge = Ridge(alpha=1.0).fit(Ztr, Xl_tr)
    R_tr, R_te = Xl_tr - ridge.predict(Ztr), Xl_te - ridge.predict(Zte)
    rs = _Std(R_tr)
    res = _fit_lr(rs(R_tr), y_tr, C)
    out["residual_probe_auc"] = _safe_auc(y_te, res.decision_function(rs(R_te)))
    w = res.coef_[0] / rs.sd
    out["residual_direction"] = (w / (np.linalg.norm(w) + 1e-9)).astype(np.float32)
    # beyond nuisance: out-of-fold probe scores on train
    xs = _Std(Xl_tr)
    s_tr = np.zeros(len(y_tr), dtype=np.float32)
    n_groups = len(np.unique(g_tr))
    if n_groups >= 5:
        for tr, va in GroupKFold(n_splits=5).split(Xl_tr, y_tr, g_tr):
            if len(np.unique(y_tr[tr])) < 2:
                continue
            s_tr[va] = _fit_lr(xs(Xl_tr[tr]), y_tr[tr], C).decision_function(xs(Xl_tr[va]))
    full = _fit_lr(xs(Xl_tr), y_tr, C)
    s_te = full.decision_function(xs(Xl_te))
    ss = _Std(s_tr[:, None])
    comb = _fit_lr(np.concatenate([Ztr, ss(s_tr[:, None])], 1), y_tr, C)
    out["nuisance_plus_probe_auc"] = _safe_auc(y_te, comb.decision_function(np.concatenate([Zte, ss(s_te[:, None])], 1)))
    if out["nuisance_plus_probe_auc"] is not None and out["nuisance_only_auc"] is not None:
        out["auc_gain_beyond_nuisance"] = out["nuisance_plus_probe_auc"] - out["nuisance_only_auc"]
    return out


def _select(meta: List[dict], task: str, tasks: dict) -> Tuple[np.ndarray, np.ndarray]:
    sel, lab = tasks[task]
    idx = np.array([i for i, m in enumerate(meta) if sel(m)], dtype=int)
    y = np.array([lab(meta[i]) for i in idx], dtype=int)
    return idx, y


def run_probes(run: Run, regime: str, cfg: dict, tasks: Optional[Sequence[str]] = None,
               split_modes: Optional[Sequence[str]] = None, n_jobs: int = 8) -> dict:
    a = cfg["analysis"]
    C = a["probe_C"]
    qs = run.questions()
    split_modes = list(split_modes or a["splits"])
    results: Dict[str, dict] = {}
    for which, task_defs in (("deltas", DELTA_TASKS), ("states", STATE_TASKS)):
        if which == "states" and regime == "conversational":
            continue
        try:
            X, meta = load_features(run, regime, which)
        except FileNotFoundError:
            log.warning("no %s features for %s", which, regime); continue
        n_layers = X.shape[1]
        layers = a.get("layers") or list(range(n_layers))
        for task in (tasks or list(task_defs)):
            if task not in task_defs:
                continue
            idx, y = _select(meta, task, task_defs)
            if len(idx) < 20 or y.sum() < 5 or (len(y) - y.sum()) < 5:
                log.warning("[%s/%s] too few examples (%d, %d pos); skipping", regime, task, len(idx), int(y.sum()))
                continue
            qid = np.array([meta[i]["qid"] for i in idx])
            for mode in split_modes:
                assign = make_split(qs, mode, a["test_frac"], a["seed"])
                is_tr = np.array([assign.get(q, "train") == "train" for q in qid])
                tr, te = np.where(is_tr)[0], np.where(~is_tr)[0]
                if len(te) < 10 or len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
                    log.warning("[%s/%s/%s] degenerate split; skipping", regime, task, mode); continue
                Xs = np.asarray(X[idx])  # [n, nL, d] float16
                per_layer = Parallel(n_jobs=n_jobs)(
                    delayed(_layer_eval)(Xs[tr, l].astype(np.float32), y[tr], qid[tr], Xs[te, l].astype(np.float32), y[te], C, 3)
                    for l in layers)
                cv = [r["cv_auc"] if r["cv_auc"] is not None else -1 for r in per_layer]
                best_i = int(np.argmax(cv))
                best_layer = layers[best_i]
                entry = {
                    "task": task, "split": mode, "which": which, "n_train": int(len(tr)), "n_test": int(len(te)),
                    "pos_rate_train": float(y[tr].mean()), "pos_rate_test": float(y[te].mean()),
                    "layers": layers, "cv_auc": cv, "test_auc": [r["test_auc"] for r in per_layer],
                    "test_ap": [r["test_ap"] for r in per_layer], "dm_test_auc": [r["dm_test_auc"] for r in per_layer],
                    "best_layer": best_layer, "best_test_auc": per_layer[best_i]["test_auc"],
                    "proj_std": [r["proj_std"] for r in per_layer],
                }
                dirs = np.stack([r["direction"] for r in per_layer]); dms = np.stack([r["dm_direction"] for r in per_layer])
                out_dir = run.probes_dir(regime) / task / mode
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / "directions.npy", dirs); np.save(out_dir / "dm_directions.npy", dms)
                np.save(out_dir / "coef_scaled.npy", np.stack([r["coef_scaled"] for r in per_layer]))
                np.save(out_dir / "mu.npy", np.stack([r["mu"] for r in per_layer]))
                if which == "deltas":
                    svd = _fit_svd(meta, idx[tr], a["n_passage_svd"])
                    Z_tr, Z_te = _nuisance_matrix(meta, idx[tr], svd), _nuisance_matrix(meta, idx[te], svd)
                    nz = _nuisance_analysis(Xs[tr, best_layer].astype(np.float32), y[tr], qid[tr],
                                            Xs[te, best_layer].astype(np.float32), y[te], Z_tr, Z_te, C)
                    np.save(out_dir / "residual_direction_best.npy", nz.pop("residual_direction"))
                    entry["nuisance"] = nz
                    entry["kind_projection_means"] = kind_projection_table(Xs[te, best_layer].astype(np.float32), dirs[best_i],
                                                                           [meta[i] for i in idx[te]])
                with open(out_dir / "metrics.json", "w") as f:
                    json.dump(entry, f, indent=2)
                results[f"{which}/{task}/{mode}"] = entry
                log.info("[%s] %-20s %-9s best layer %2d  cv %.3f  test AUROC %.3f  (n=%d/%d)", regime, task, mode,
                         best_layer, cv[best_i], entry["best_test_auc"] or float("nan"), len(tr), len(te))
    # cross-task transfer (deltas), per split
    for mode in split_modes:
        tm = transfer_matrix(run, regime, mode, cfg)
        if tm:
            results[f"transfer/{mode}"] = tm
    # merge into any existing summary so partial re-runs (e.g. one split) keep the other entries
    sp = run.probes_dir(regime) / "summary.json"
    merged = {}
    if sp.exists():
        with open(sp) as f:
            merged = json.load(f)
    merged.update(results)
    with open(sp, "w") as f:
        json.dump(merged, f, indent=2)
    return merged


def kind_projection_table(Xl: np.ndarray, v: np.ndarray, meta: List[dict]) -> dict:
    """Mean projection of test increments on a direction, grouped by increment kind
    and by behavioural transition."""
    s = Xl @ v
    out: Dict[str, dict] = {"by_kind": {}, "by_transition": {}}
    kinds = sorted({m["kind"] for m in meta})
    for k in kinds:
        m = np.array([mm["kind"] == k for mm in meta])
        out["by_kind"][k] = {"mean": float(s[m].mean()), "sd": float(s[m].std()), "n": int(m.sum())}
    for name, sel in {
        "wrong->correct": lambda m: m["state_from"] in WRONG_STATES and m["state_to"] == "correct",
        "wrong->wrong(changed)": lambda m: m["state_from"] in WRONG_STATES and m["state_to"] != "correct" and m["revision"],
        "wrong->wrong(same)": lambda m: m["state_from"] in WRONG_STATES and m["state_to"] != "correct" and not m["revision"],
        "correct->correct": lambda m: m["state_from"] == "correct" and m["state_to"] == "correct",
        "correct->wrong": lambda m: m["state_from"] == "correct" and m["state_to"] != "correct",
    }.items():
        m = np.array([bool(sel(mm)) for mm in meta])
        if m.any():
            out["by_transition"][name] = {"mean": float(s[m].mean()), "sd": float(s[m].std()), "n": int(m.sum())}
    return out


def load_direction(run: Run, regime: str, task: str, mode: str, layer: Optional[int] = None,
                   kind: str = "lr") -> Tuple[int, np.ndarray, float]:
    d = run.probes_dir(regime) / task / mode
    with open(d / "metrics.json") as f:
        m = json.load(f)
    layer = m["best_layer"] if layer is None else layer
    li = m["layers"].index(layer)
    dirs = np.load(d / ("directions.npy" if kind == "lr" else "dm_directions.npy"))
    return layer, dirs[li], m["proj_std"][li]


def transfer_matrix(run: Run, regime: str, mode: str, cfg: dict) -> dict:
    """AUROC of every task's best-layer direction applied to every other task's test set."""
    try:
        X, meta = load_features(run, regime, "deltas")
    except FileNotFoundError:
        return {}
    qs = run.questions()
    assign = make_split(qs, mode, cfg["analysis"]["test_frac"], cfg["analysis"]["seed"])
    avail = [t for t in DELTA_TASKS if (run.probes_dir(regime) / t / mode / "metrics.json").exists()]
    if len(avail) < 2:
        return {}
    dirs = {t: load_direction(run, regime, t, mode) for t in avail}
    out: Dict[str, dict] = {"auc": {}, "cosine": {}, "layers": {t: int(dirs[t][0]) for t in avail}}
    for src in avail:
        layer, v, _ = dirs[src]
        out["auc"][src] = {}
        out["cosine"][src] = {}
        for tgt in avail:
            idx, y = _select(meta, tgt, DELTA_TASKS)
            te = np.array([assign.get(meta[i]["qid"], "train") == "test" for i in idx])
            if te.sum() < 10 or len(np.unique(y[te])) < 2:
                out["auc"][src][tgt] = None
            else:
                s = np.asarray(X[idx[te], layer]).astype(np.float32) @ v
                out["auc"][src][tgt] = _safe_auc(y[te], s)
            # cosine with the target direction *at the source layer*
            try:
                _, vt, _ = load_direction(run, regime, tgt, mode, layer=layer)
                out["cosine"][src][tgt] = float(v @ vt)
            except Exception:
                out["cosine"][src][tgt] = None
    return out


def apply_directions(run_src: Run, regime_src: str, run_tgt: Run, regime_tgt: str, mode: str, cfg: dict,
                     tasks: Optional[Sequence[str]] = None, all_as_test: bool = True) -> dict:
    """Transfer test: directions learned in (run_src, regime_src) scored on
    (run_tgt, regime_tgt) features -- e.g. stateless -> conversational, or
    MuSiQue -> 2Wiki.  Uses every layer's direction."""
    X, meta = load_features(run_tgt, regime_tgt, "deltas")
    out = {}
    if not all_as_test:
        assign = make_split(run_tgt.questions(), mode, cfg["analysis"]["test_frac"], cfg["analysis"]["seed"])
    for task in (tasks or list(DELTA_TASKS)):
        d = run_src.probes_dir(regime_src) / task / mode
        if not (d / "metrics.json").exists():
            continue
        with open(d / "metrics.json") as f:
            m = json.load(f)
        dirs = np.load(d / "directions.npy")
        if dirs.shape[1] != X.shape[2] or len(m["layers"]) > X.shape[1]:
            log.warning("transfer skipped for %s: source directions are %s-dim over %d layers, target activations are %s-dim over %d layers "
                        "(different model family; raw directions are not transferable)", task, dirs.shape[1], len(m["layers"]), X.shape[2], X.shape[1])
            continue
        idx, y = _select(meta, task, DELTA_TASKS)
        if not all_as_test:
            keep = np.array([assign.get(meta[i]["qid"], "train") == "test" for i in idx])
            idx, y = idx[keep], y[keep]
        if len(idx) < 10 or len(np.unique(y)) < 2:
            continue
        aucs = []
        for li, layer in enumerate(m["layers"]):
            s = np.asarray(X[idx, layer]).astype(np.float32) @ dirs[li]
            aucs.append(_safe_auc(y, s))
        best_li = m["layers"].index(m["best_layer"])
        out[task] = {"layers": m["layers"], "auc": aucs, "auc_at_src_best_layer": aucs[best_li],
                     "src_best_layer": m["best_layer"], "n": int(len(idx)), "pos_rate": float(y.mean())}
    return out
