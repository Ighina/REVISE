"""Aggregate metrics into a markdown report + per-layer AUROC plots."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from revise.experiment import Run

log = logging.getLogger(__name__)


def _fmt(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def _table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def build_report(run: Run, regimes: List[str]) -> Path:
    rep = run.sub("report")
    md = [f"# REVISE report: `{run.cfg['run_name']}`", "", f"Model: `{run.cfg['model']['name']}`  ",
          f"Dataset: `{run.cfg['data']['dataset']}` ({run.cfg['data']['split']}, hops={run.cfg['data']['hops']})", ""]
    # behaviour summary
    for regime in regimes:
        beh = run.behavior(regime)
        if not beh:
            continue
        by_name: Dict[str, List[dict]] = {}
        for r in beh.values():
            by_name.setdefault(r.get("name", "?"), []).append(r)
        rows = [[n, str(len(v)), _fmt(np.mean([x["correct"] for x in v])), _fmt(np.mean([x["insufficient"] for x in v]))]
                for n, v in sorted(by_name.items())]
        md += [f"## Behaviour by condition family ({regime})", "", _table(["condition", "n", "correct", "INSUFFICIENT"], rows), ""]
    # probes
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    for regime in regimes:
        p = run.probes_dir(regime) / "summary.json"
        if not p.exists():
            continue
        with open(p) as f:
            summ = json.load(f)
        entries = [v for k, v in summ.items() if not k.startswith("transfer/")]
        if entries:
            rows = []
            for e in entries:
                nz = e.get("nuisance", {})
                rows.append([e["task"], e["split"], str(e["best_layer"]), _fmt(e["best_test_auc"]),
                             _fmt(e["dm_test_auc"][e["layers"].index(e["best_layer"])]),
                             _fmt(nz.get("nuisance_only_auc")), _fmt(nz.get("residual_probe_auc")),
                             _fmt(nz.get("auc_gain_beyond_nuisance")), f"{e['n_train']}/{e['n_test']}", _fmt(e["pos_rate_test"])])
            md += [f"## Probes ({regime})", "",
                   _table(["task", "split", "best L", "AUROC", "diff-means AUROC", "nuisance-only", "residualised", "gain beyond nuisance", "n tr/te", "pos rate"], rows), ""]
            # ordering table on the sufficiency direction
            for e in entries:
                kp = e.get("kind_projection_means")
                if kp and e["task"] in ("sufficiency", "sufficiency_matched"):
                    rows = [[k, _fmt(v["mean"]), _fmt(v["sd"]), str(v["n"])] for k, v in sorted(kp["by_kind"].items(), key=lambda kv: -kv[1]["mean"])]
                    md += [f"### Projection on `{e['task']}` direction (L{e['best_layer']}, split={e['split']}) by increment kind", "",
                           _table(["kind", "mean proj", "sd", "n"], rows), ""]
                    rows = [[k, _fmt(v["mean"]), _fmt(v["sd"]), str(v["n"])] for k, v in kp["by_transition"].items()]
                    md += ["by behavioural transition:", "", _table(["transition", "mean proj", "sd", "n"], rows), ""]
            if plt is not None:
                for split in sorted({e["split"] for e in entries}):
                    fig, ax = plt.subplots(figsize=(7, 4))
                    for e in entries:
                        if e["split"] != split:
                            continue
                        ax.plot(e["layers"], [a if a is not None else np.nan for a in e["test_auc"]], marker="o", ms=3, label=e["task"])
                    ax.axhline(0.5, color="grey", lw=0.8, ls="--"); ax.set_xlabel("layer"); ax.set_ylabel("test AUROC")
                    ax.set_title(f"{regime} / split={split}"); ax.legend(fontsize=7); fig.tight_layout()
                    fig.savefig(rep / f"auroc_by_layer_{regime}_{split}.png", dpi=130); plt.close(fig)
        for k, tm in summ.items():
            if not k.startswith("transfer/") or not tm:
                continue
            tasks = list(tm["auc"])
            rows = [[src + f" (L{tm['layers'][src]})"] + [_fmt(tm["auc"][src].get(t)) for t in tasks] for src in tasks]
            md += [f"## Cross-task transfer AUROC ({regime}, {k.split('/')[1]} split): rows = direction, cols = task", "",
                   _table(["direction \\ task"] + tasks, rows), ""]
            rows = [[src] + [_fmt(tm["cosine"][src].get(t)) for t in tasks] for src in tasks]
            md += ["cosine similarity between directions (at the row direction's layer):", "", _table(["dir"] + tasks, rows), ""]
    for name in ("transfer_conversational.json", "transfer_external.json"):
        p = run.dir / "probes" / name
        if p.exists():
            with open(p) as f:
                tr = json.load(f)
            rows = [[t, str(v["src_best_layer"]), _fmt(v["auc_at_src_best_layer"]), _fmt(max(a for a in v["auc"] if a is not None)), str(v["n"])] for t, v in tr.items()]
            md += [f"## Direction transfer: {name.replace('.json', '')}", "", _table(["task", "src layer", "AUROC @ src best layer", "best AUROC any layer", "n"], rows), ""]
    # steer / patch / adaptive
    for p in sorted((run.dir / "steer").glob("*.json")) if (run.dir / "steer").exists() else []:
        with open(p) as f:
            st = json.load(f)
        rows = [[r.get("task", "?"), str(r["layer"]), r["direction"], f"{r['alpha']:+.2f}", _fmt(r.get("l2")), r["set"], _fmt(r["correct"]), _fmt(r["answered"]), _fmt(r["false_follow"]), str(r["n"])] for r in st["runs"]]
        md += [f"## Steering: {p.stem} (positions={st['config'].get('positions', 'anchor_onward')}; alpha in units of the sufficient-minus-insufficient gap)", "",
               "units: " + json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in st.get("units", {}).items()}), "",
               _table(["task", "layer", "dir", "alpha", "L2 shift", "set", "correct", "answered", "false-follow", "n"], rows), ""]
    p = run.dir / "patch" / "results.json"
    if p.exists():
        with open(p) as f:
            pt = json.load(f)
        rows = [[r.get("task", "?"), r.get("mode", "single"), str(r["layer"]), str(r.get("n_layers_patched", 1)), r["variant"], _fmt(r["correct"]), _fmt(r["answered"]), str(r["n"])] for r in pt["runs"]]
        md += ["## Activation patching (insufficient prompt, patched at anchor)", "", _table(["task", "mode", "from layer", "#layers", "variant", "correct", "answered", "n"], rows), ""]
    p = run.dir / "ablate" / "results.json"
    if p.exists():
        with open(p) as f:
            ab = json.load(f)
        rows = [[r["experiment"], r.get("task", r.get("variant", "")), r.get("direction_kind", ""), r.get("direction", ""),
                 f"{r['alpha']:+.1f}" if "alpha" in r else "", str(r["layer"]) + "+" if r.get("n_layers", 1) > 1 else str(r["layer"]),
                 r.get("set", "insufficient->patched"), _fmt(r["correct"]), _fmt(r["answered"]), _fmt(r.get("false_follow")), str(r["n"])] for r in ab["runs"]]
        md += ["## Ablations: projection-out, difference-of-means multi-layer steering, causal subspace rank", "",
               _table(["exp", "task/variant", "dir kind", "dir", "alpha", "layers", "set", "correct", "answered", "false-follow", "n"], rows), ""]
    p = run.dir / "text_baseline" / "results.json"
    if p.exists():
        with open(p) as f:
            tbr = json.load(f)
        keys = sorted({k for e in tbr.values() for k in e if k.endswith("_auc")})
        rows = [[e["task"], e["split"], _fmt(e.get("probe_auc"))] + [_fmt(e.get(k)) for k in keys] for e in tbr.values()]
        md += ["## Text-only baselines vs activation probe (held-out AUROC)", "", _table(["task", "split", "probe"] + keys, rows), ""]
    p = run.dir / "closedbook" / "results.json"
    if p.exists():
        with open(p) as f:
            cb = json.load(f)
        _mx = lambda xs: max([x for x in xs if x is not None], default=None)
        rows = [[t, str(v["src_best_layer"]), _fmt(v["auc_correct_at_src_best"]), _fmt(_mx(v["auc_correct"])),
                 _fmt(v["auc_abstain_at_src_best"]), _fmt(_mx(v["auc_abstain"]))] for t, v in cb["directions"].items()]
        ce = cb["closed_book_probe_ceiling"]
        rows.append(["(own closed-book probe, ceiling)", "-", "-", _fmt(_mx(ce["auc_correct"])), "-", _fmt(_mx(ce["auc_abstain"]))])
        md += [f"## Closed-book transfer (n={cb['n']}, correct rate {cb['correct_rate']:.3f}, abstain rate {cb['abstain_rate']:.3f})", "",
               _table(["direction", "src layer", "AUROC correct @src", "best any layer", "AUROC abstain @src", "best any layer"], rows), ""]
    p = run.dir / "adaptive" / "results.json"
    if p.exists():
        with open(p) as f:
            ad = json.load(f)
        rows = [[k, _fmt(v["accuracy"]), f"{v['mean_rounds']:.2f}", f"{v['mean_prompt_tokens']:.0f}", str(v["n"])] for k, v in ad["policies"].items()]
        md += [f"## Adaptive retrieval simulation (state probe, layer {ad['layer']})", "", _table(["policy", "accuracy", "rounds", "prompt tokens", "n"], rows), ""]
    out = rep / "summary.md"
    out.write_text("\n".join(md))
    log.info("report written to %s", out)
    return out
