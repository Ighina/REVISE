"""Adaptive-retrieval appendix: every learned direction as a stopping rule and as an answer verifier.

Reads runs/<run>/adaptive/directions.json (written by revise.run.adaptive.run_adaptive_directions)
and writes tables/adaptive.tex, tables/adaptive_verify.tex and figures/adaptive_curves.pdf.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.environ.get("REVISE_SCRATCH", "/scratch/users/iacopog/revise") + "/runs"
OUT = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(os.path.dirname(OUT), "tables")
COLS = [("\\musique{} Qwen", "musique2hop_qwen7b"), ("\\musique{} Llama", "musique2hop_llama8b"), ("\\musique{} Mistral", "musique2hop_mistral7b"),
        ("HotpotQA Qwen", "hotpotqa_qwen7b"), ("\\twowiki{} Qwen", "twowiki_qwen7b"), ("Synthetic Qwen", "synthetic_qwen7b")]
D = {run: json.load(open(f"{R}/{run}/adaptive/directions.json")) for _, run in COLS}

ROWS = [("Baselines", None),
        ("full context (never stop)", "full"), ("oracle (stop when sufficient)", "oracle"),
        ("behavioural (first non-INSUFFICIENT)", "behavioral"), ("next-token entropy", "entropy"),
        ("State probes, $h(C_k)$", None),
        ("sufficiency (logistic)", "state_sufficiency"), ("sufficiency ($\\vdm$)", "state_sufficiency_dm"),
        ("correctness", "state_correct"), ("abstention flag", "state_insufficient_flag"),
        ("Update probes, $\\Delta h$ of the round", None),
        ("sufficiency (logistic)", "sufficiency"), ("sufficiency ($\\vdm$)", "sufficiency_dm"), ("sufficiency matched", "sufficiency_matched"),
        ("uptake", "uptake"), ("correction", "correction"), ("stability", "stability"),
        ("stability, stop when the answer changes", ("stability", -1.0)), ("revision (stop when the answer changes)", ("revision", 1.0))]


def pick(d, key):
    if isinstance(key, tuple):
        name, sign = key
        p = d["policies"].get(name)
        if not p:
            return None
        return p if p["sign"] == sign else {**p["other_sign"], "diff_vs_behavioral_ci": None}
    return d["policies"].get(key)


def cell(p, best, key):
    if p is None:
        return "--"
    s = f"{p['accuracy']:.3f} ({p['mean_rounds']:.2f})"
    if isinstance(key, str) and key not in ("full", "oracle", "behavioral") and p.get("sign") == -1.0:
        s += "$^{-}$"      # the training-optimal reading stops when the projection is *low*
    ci = p.get("diff_vs_behavioral_ci")
    if ci and ci[0] > 0:
        s += "$^{+}$"
    return f"\\textbf{{{s}}}" if best else s


lines = [r"\begin{tabular}{l" + "c" * len(COLS) + "}", r"\toprule",
         "Stopping rule & " + " & ".join(c for c, _ in COLS) + r" \\", r"\midrule"]
learned = [k for _, k in ROWS if k and k not in ("full", "oracle", "behavioral", "entropy")]
for label, key in ROWS:
    if key is None:
        lines.append(f"\\multicolumn{{{len(COLS) + 1}}}{{l}}{{\\emph{{{label}}}}} \\\\")
        continue
    cells = []
    for _, run in COLS:
        d = D[run]; p = pick(d, key)
        top = max(pick(d, k)["accuracy"] for k in learned + ["behavioral", "entropy"] if pick(d, k))
        cells.append(cell(p, p is not None and key not in ("oracle",) and abs(p["accuracy"] - top) < 1e-9, key))
    lines.append(f"\\quad {label} & " + " & ".join(cells) + r" \\")
lines.append(r"\midrule")
lines.append("test questions & " + " & ".join(str(D[run]["n_test"]) for _, run in COLS) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{TAB}/adaptive.tex", "w").write("\n".join(lines))

VROWS = [("top-token probability", "top_prob"), ("$-$next-token entropy", "neg_entropy"),
         ("state: sufficiency", "state_sufficiency"), ("state: correctness", "state_correct"), ("state: abstention flag", "state_insufficient_flag"),
         ("update: sufficiency", "sufficiency"), ("update: sufficiency ($\\vdm$)", "sufficiency_dm"),
         ("update: uptake", "uptake"), ("update: correction", "correction"), ("update: stability", "stability"), ("update: revision", "revision")]
lines = [r"\begin{tabular}{l" + "c" * len(COLS) + "}", r"\toprule",
         "Signal at the stopping round & " + " & ".join(c for c, _ in COLS) + r" \\", r"\midrule"]
for label, key in VROWS:
    cells = []
    for _, run in COLS:
        v = D[run]["verification"]["state_sufficiency"]
        a = v["auroc"].get(key)
        best = a is not None and a == max(v["auroc"].values())
        cells.append("--" if a is None else (f"\\textbf{{{a:.3f}}}" if best else f"{a:.3f}"))
    lines.append(f"{label} & " + " & ".join(cells) + r" \\")
lines.append(r"\midrule")
for label, key in [("accuracy of all answered stops", None), ("accuracy, top half by top-token prob.", "top_prob"), ("accuracy, top half by uptake", "uptake")]:
    cells = []
    for _, run in COLS:
        v = D[run]["verification"]["state_sufficiency"]
        x = v["accuracy_answered"] if key is None else v["selective_acc_50"].get(key)
        cells.append("--" if x is None else f"{x:.3f}")
    lines.append(f"{label} & " + " & ".join(cells) + r" \\")
lines.append("answered stops & " + " & ".join(str(D[run]["verification"]["state_sufficiency"]["n_answered"]) for _, run in COLS) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{TAB}/adaptive_verify.tex", "w").write("\n".join(lines))

# trade-off curves: test-set threshold sweeps for four directions, baselines as points
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
CURVES = [("state_sufficiency", "state sufficiency", "#1f77b4", "-"), ("sufficiency_dm", r"update sufficiency ($v_{dm}$)", "#1f77b4", "--"),
          ("uptake", "update uptake", "#d62728", "-"), ("stability", "update stability", "#2ca02c", "-")]
panels = [("MuSiQue, Qwen2.5-7B", "musique2hop_qwen7b"), ("MuSiQue, Llama-3.1-8B", "musique2hop_llama8b"),
          ("MuSiQue, Mistral-7B", "musique2hop_mistral7b"), ("HotpotQA, Qwen2.5-7B", "hotpotqa_qwen7b")]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
for ax, (title, run) in zip(axes, panels):
    d = D[run]
    for key, label, c, ls in CURVES:
        pts = sorted(set(map(tuple, d["curves"].get(key, []))))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls, lw=1.2, label=label)
    for key, label, m in [("behavioral", "behavioural", "^"), ("oracle", "oracle", "*"), ("full", "full context", "s")]:
        p = d["policies"][key]
        ax.plot(p["mean_rounds"], p["accuracy"], m, color="black", ms=5 if m != "*" else 7, label=label, ls="none")
    ax.set_title(title, fontsize=7.5); ax.set_xlabel("mean retrieval rounds"); ax.set_xlim(0.9, 4.1)
axes[0].set_ylabel("accuracy at stop")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
fig.set_size_inches(7.0, 2.2); fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig(f"{OUT}/adaptive_curves.pdf"); fig.savefig(f"{OUT}/adaptive_curves.png"); plt.close(fig)
print(open(f"{TAB}/adaptive.tex").read()); print(open(f"{TAB}/adaptive_verify.tex").read())
