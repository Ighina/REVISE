"""Generate the paper figures from the run artifacts (PDF, matplotlib only)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.environ.get("REVISE_SCRATCH", "/scratch/users/iacopog/revise") + "/runs"
OUT = os.path.dirname(os.path.abspath(__file__))
MODELS = {"Qwen2.5-7B": "musique2hop_qwen7b", "Llama-3.1-8B": "musique2hop_llama8b", "Mistral-7B": "musique2hop_mistral7b"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
COL = {"sufficiency": "#1f77b4", "sufficiency_matched": "#aec7e8", "uptake": "#d62728", "correction": "#ff7f0e", "stability": "#2ca02c", "revision": "#7f7f7f"}


def load(model, rel):
    with open(f"{R}/{MODELS[model]}/{rel}") as f:
        return json.load(f)


def fig_auroc_by_layer():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.75), sharey=True)
    for ax, (m, r) in zip(axes, MODELS.items()):
        s = load(m, "probes/evidence_only/summary.json")
        for t in ["sufficiency", "sufficiency_matched", "uptake", "correction", "stability"]:
            e = s.get(f"deltas/{t}/document")
            if not e: continue
            L = np.array(e["layers"]) / max(e["layers"])
            ax.plot(L, [a if a is not None else np.nan for a in e["test_auc"]], color=COL[t], lw=1.2, label=t.replace("_", " "))
        ax.axhline(0.5, color="k", lw=0.5, ls=":"); ax.set_title(m); ax.set_xlabel("relative depth"); ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("held-out AUROC (document split)"); axes[-1].legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/auroc_by_layer.pdf"); plt.close(fig)


def baseline_rates(model):
    """Unsteered rates on the same held-out evaluation sets (from the stored stateless behaviour)."""
    sys.path.insert(0, os.path.dirname(OUT) + "/..")
    from revise.experiment import load_config, Run
    from revise.run.steer import _pick_eval_conditions
    from revise.eval import normalize_answer
    cfg = load_config(f"configs/{ {'Qwen2.5-7B': 'musique_2hop_qwen7b', 'Llama-3.1-8B': 'musique_2hop_llama8b', 'Mistral-7B': 'musique_2hop_mistral7b'}[model] }.yaml")
    run = Run(cfg); beh = run.behavior("evidence_only")
    sets = _pick_eval_conditions(run, cfg["steer"]["split"], cfg, cfg["steer"]["max_questions"])
    out = {}
    for name, conds in sets.items():
        rows = [beh[c.key] for c in conds if c.key in beh]
        ff = []
        for c in conds:
            fa = next((p.meta.get("false_answer") for p in c.paragraphs if p.role == "false"), None)
            if fa and c.key in beh: ff.append(normalize_answer(fa) in normalize_answer(beh[c.key]["pred"]))
        out[name] = {"correct": np.mean([r["correct"] for r in rows]), "answered": np.mean([not r["insufficient"] for r in rows]),
                     "false_follow": (np.mean(ff) if ff else None)}
    return out


def fig_steer():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.75), sharey=True)
    sets = [("sufficient", "sufficient: correct", "correct", "#2ca02c"), ("sufficient", "sufficient: answered", "answered", "#98df8a"),
            ("insufficient", "insufficient: answered", "answered", "#d62728"), ("distractor_only", "distractors only: answered", "answered", "#ff9896"),
            ("false", "false evidence: followed", "false_follow", "#9467bd")]
    for ax, (m, r) in zip(axes, MODELS.items()):
        a = load(m, "ablate/results_dmsteer_small.json")
        runs = [x for x in a["runs"] if x["direction"] == "probe"]
        alphas = sorted({x["alpha"] for x in runs})
        base = baseline_rates(m)
        for set_name, label, key, c in sets:
            xs, ys = [0.0], [base[set_name][key] if set_name in base and base[set_name].get(key) is not None else np.nan]
            for al in alphas:
                x = next((x for x in runs if x["alpha"] == al and x["set"] == set_name), None)
                if x is not None and x.get(key) is not None: xs.append(al); ys.append(x[key])
            o = np.argsort(xs); ax.plot(np.array(xs)[o], np.array(ys)[o], marker="o", ms=2.5, lw=1, color=c, label=label)
        ax.set_title(m); ax.set_xlabel(r"steering $\alpha$ (gap units)"); ax.set_ylim(0, 1)
    axes[0].set_ylabel("rate"); axes[-1].legend(fontsize=5.5, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{OUT}/steer_dose_response.pdf"); plt.close(fig)


def fig_subspace():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.7), sharey=True)
    order = ["none", "pca1", "pca4", "pca16", "pca64", "pca256", "pca256_rand", "unembed_add", "unembed_remove", "full"]
    lab = {"none": "none", "pca1": "PC1", "pca4": "PC4", "pca16": "PC16", "pca64": "PC64", "pca256": "PC256", "pca256_rand": "rand256",
           "unembed_add": "unemb", "unembed_remove": "full−unemb", "full": "full"}
    for ax, (m, r) in zip(axes, MODELS.items()):
        a = load(m, "ablate/results.json")
        sub = {x["variant"]: x for x in a["runs"] if x["experiment"] == "subspace"}
        xs = np.arange(len(order))
        ax.bar(xs - 0.2, [sub[v]["answered"] for v in order], 0.4, color="#9ecae1", label="answered")
        ax.bar(xs + 0.2, [sub[v]["correct"] for v in order], 0.4, color="#3182bd", label="correct")
        ax.set_xticks(xs); ax.set_xticklabels([lab[v] for v in order], rotation=60, fontsize=6); ax.set_title(m); ax.set_ylim(0, 0.85)
    axes[0].set_ylabel("rate after patching"); axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/subspace_rank.pdf"); plt.close(fig)


def fig_projection_by_kind():
    s = load("Qwen2.5-7B", "probes/evidence_only/summary.json")
    e = s["deltas/sufficiency/document"]; kp = e["kind_projection_means"]["by_kind"]
    items = sorted(kp.items(), key=lambda kv: kv[1]["mean"])
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.barh([k.replace("_", " ") for k, _ in items], [v["mean"] for _, v in items],
            xerr=[v["sd"] / np.sqrt(v["n"]) for _, v in items], color=["#d62728" if k in ("decisive", "loo_decisive") else "#9ecae1" for k, _ in items])
    ax.axvline(0, color="k", lw=0.5); ax.set_xlabel(f"projection on sufficiency dir. (L{e['best_layer']})")
    fig.tight_layout(); fig.savefig(f"{OUT}/projection_by_kind.pdf"); plt.close(fig)


def fig_transfer():
    s = load("Qwen2.5-7B", "probes/evidence_only/summary.json"); tm = s["transfer/document"]
    tasks = list(tm["auc"]); M = np.array([[tm["auc"][a].get(b) or np.nan for b in tasks] for a in tasks])
    labels = [t.replace("_", " ") for t in tasks]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    cmap = plt.get_cmap("viridis"); vmin, vmax = 0.5, 1.0
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            r, g, b, _ = cmap((M[i, j] - vmin) / (vmax - vmin))
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b          # relative luminance of the cell colour
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="black" if lum > 0.5 else "white")
    ax.set_xlabel("evaluated task", labelpad=8); ax.set_ylabel("direction learned for", labelpad=8)
    ax.tick_params(axis="both", length=2, pad=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("AUROC", labelpad=6); cb.ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.3, right=0.98, top=0.98, bottom=0.3)
    fig.savefig(f"{OUT}/transfer_matrix.pdf", bbox_inches="tight", pad_inches=0.05); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_auroc_by_layer, fig_steer, fig_subspace, fig_projection_by_kind, fig_transfer):
        try:
            f(); print("ok", f.__name__)
        except Exception as ex:
            print("FAILED", f.__name__, repr(ex))
