"""Tables for 'what the probes read': representation ablation (state vs update), output-controlled
sufficiency probes, and verbalised versus internal sufficiency judgements.

Reads runs/<run>/probes/evidence_only/{representation_ablation,behaviour_control,verbal_vs_internal}_document.json
and writes tables/rep_ablation.tex, tables/behaviour_control.tex, tables/verbal_internal.tex.
"""
import json, os

R = os.environ.get("REVISE_SCRATCH", "/scratch/users/iacopog/revise") + "/runs"
TAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tables")
MODELS = [("Qwen2.5-7B", "musique2hop_qwen7b"), ("Llama-3.1-8B", "musique2hop_llama8b"), ("Mistral-7B", "musique2hop_mistral7b")]
J = lambda run, name: json.load(open(f"{R}/{run}/probes/evidence_only/{name}_document.json"))
f3 = lambda x: "--" if x is None else f"{x:.3f}"

# 1. representation ablation
TASKS = [("sufficiency", "Sufficiency"), ("sufficiency_matched", "Sufficiency (matched)"), ("uptake", "Uptake"),
         ("correction", "Correction"), ("stability", "Stability"), ("revision", "Revision")]
REPS = [("from", "$h(C_{k-1})$"), ("to", "$h(C_k)$"), ("delta", "$\\Delta h$"), ("concat", "$[h(C_{k-1});h(C_k)]$")]
lines = [r"\begin{tabular}{ll" + "c" * len(REPS) + "}", r"\toprule",
         "Model & Task & " + " & ".join(l for _, l in REPS) + r" \\", r"\midrule"]
for mi, (mname, run) in enumerate(MODELS):
    d = J(run, "representation_ablation")["tasks"]
    for ti, (t, tl) in enumerate(TASKS):
        if t not in d:
            continue
        vals = [d[t]["reps"][r]["test_auc"] for r, _ in REPS]
        best = max(v for v in vals if v is not None)
        cells = [(f"\\textbf{{{v:.3f}}}" if v is not None and abs(v - best) < 5e-4 else f3(v)) for v in vals]
        lines.append(f"{mname if ti == 0 else ''} & {tl} & " + " & ".join(cells) + r" \\")
    if mi < len(MODELS) - 1:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{TAB}/rep_ablation.tex", "w").write("\n".join(lines))

# 2. output-controlled sufficiency probes
lines = [r"\begin{tabular}{lllcccccc}", r"\toprule",
         r"& & & output & all & \multicolumn{2}{c}{model abstains at $C_k$} & \multicolumn{2}{c}{model answers at $C_k$} \\",
         r"\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"Model & Task & Repr. & only & test & stratified & within & stratified & within \\", r"\midrule"]
for mi, (mname, run) in enumerate(MODELS):
    d = J(run, "behaviour_control")
    rows = [("Sufficiency", "$\\Delta h$", d["updates"]["sufficiency"]["delta"]), ("Sufficiency", "$h(C_k)$", d["updates"]["sufficiency"]["to"]),
            ("Matched", "$\\Delta h$", d["updates"]["sufficiency_matched"]["delta"]), ("Matched", "$h(C_k)$", d["updates"]["sufficiency_matched"]["to"]),
            ("State suff.", "$h(C)$", d["states"]["state_sufficiency"])]
    for ri, (tl, rl, r) in enumerate(rows):
        lines.append(f"{mname if ri == 0 else ''} & {tl} & {rl} & {f3(r['output_only_auc'])} & {f3(r['all_auc'])} & "
                     f"{f3(r.get('stratified_abstains_auc'))} & {f3(r.get('within_abstains_auc'))} & "
                     f"{f3(r.get('stratified_answers_auc'))} & {f3(r.get('within_answers_auc'))} \\\\")
    if mi < len(MODELS) - 1:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{TAB}/behaviour_control.tex", "w").write("\n".join(lines))

# 3. verbalised vs internal sufficiency
FAM = [("sufficient_min", "minimal sufficient"), ("sufficient_plus_distractor", "sufficient + distractor"),
       ("sufficient_plus_redundant", "sufficient + duplicate"), ("full", "full context (20 par.)")]
V = {run: J(run, "verbal_vs_internal") for _, run in MODELS}
lines = [r"\begin{tabular}{l" + "c" * len(MODELS) + "}", r"\toprule", " & " + " & ".join(m for m, _ in MODELS) + r" \\", r"\midrule",
         r"\multicolumn{" + str(len(MODELS) + 1) + r"}{l}{\emph{Held-out contexts}} \\"]
lines.append(r"\quad sufficient / insufficient & " + " & ".join(f"{V[r]['n_sufficient']} / {V[r]['n_insufficient']}" for _, r in MODELS) + r" \\")
lines.append(r"\quad sufficient but verbalised INSUFFICIENT & " + " & ".join(f"{V[r]['n_sufficient_abstained']} ({V[r]['n_sufficient_abstained'] / V[r]['n_sufficient']:.0%})".replace('%', '\\%') for _, r in MODELS) + r" \\")
lines.append(r"\multicolumn{" + str(len(MODELS) + 1) + r"}{l}{\emph{Verbalised judgement (answers = sufficient)}} \\")
lines.append(r"\quad AUROC & " + " & ".join(f3(V[r]["verbal"]["auc"]) for _, r in MODELS) + r" \\")
lines.append(r"\quad TPR / FPR & " + " & ".join(f"{V[r]['verbal']['tpr']:.3f} / {V[r]['verbal']['fpr']:.3f}" for _, r in MODELS) + r" \\")
lines.append(r"\multicolumn{" + str(len(MODELS) + 1) + r"}{l}{\emph{State-sufficiency probe at the same anchor}} \\")
lines.append(r"\quad AUROC & " + " & ".join(f3(V[r]["probe_auc"]) for _, r in MODELS) + r" \\")
for key, lab in (("matched_fpr", "verbalised FPR"), ("precision_0.95", "0.95 precision")):
    lines.append(r"\quad TPR / FPR, threshold at " + lab + " & " + " & ".join(f"{V[r][key]['tpr']:.3f} / {V[r][key]['fpr']:.3f}" for _, r in MODELS) + r" \\")
    lines.append(r"\qquad recall on wrongly abstained sufficient contexts & " + " & ".join(f3(V[r][key]["recall_on_wrong_abstentions"]) for _, r in MODELS) + r" \\")
    lines.append(r"\qquad false alarms on correctly abstained insufficient contexts & " + " & ".join(f3(V[r][key]["false_alarm_on_correct_abstentions"]) for _, r in MODELS) + r" \\")
lines.append(r"\multicolumn{" + str(len(MODELS) + 1) + r"}{l}{\emph{Wrong abstentions by condition: abstain rate; probe recall at 0.95 precision}} \\")
for fam, lab in FAM:
    lines.append(f"\\quad {lab} & " + " & ".join(
        (lambda b: f"{b['abstain_rate']:.2f}; {b['probe_recall']:.2f}" if b else "--")(V[r]["precision_0.95"]["by_family"].get(fam)) for _, r in MODELS) + r" \\")
lines.append(r"\multicolumn{" + str(len(MODELS) + 1) + r"}{l}{\emph{Mean probe score (0 = insufficient mean, 1 = sufficient mean on training contexts)}} \\")
for key, lab in (("suff_answered", "sufficient, answered"), ("suff_abstained", "sufficient, verbalised INSUFFICIENT"),
                 ("insuff_answered", "insufficient, answered"), ("insuff_abstained", "insufficient, verbalised INSUFFICIENT")):
    lines.append(f"\\quad {lab} & " + " & ".join(f"{V[r]['mean_score_gap_units'][key]:.2f}" for _, r in MODELS) + r" \\")
N = str(len(MODELS) + 1)
lines.append(r"\midrule")
lines.append(r"\multicolumn{" + N + r"}{l}{\emph{Two-paragraph contexts only (paragraph count matched; probe retrained on them)}} \\")
lines.append(r"\quad minimal sufficient / insufficient two-paragraph & " + " & ".join(f"{V[r]['count_matched']['n_sufficient']} / {V[r]['count_matched']['n_insufficient']}" for _, r in MODELS) + r" \\")
lines.append(r"\quad minimal sufficient but verbalised INSUFFICIENT & " + " & ".join(str(V[r]['count_matched']['n_sufficient_abstained']) for _, r in MODELS) + r" \\")
lines.append(r"\quad verbalised judgement: AUROC; TPR / FPR & " + " & ".join(f"{V[r]['count_matched']['verbal_auc']:.3f}; {V[r]['count_matched']['verbal_tpr']:.2f} / {V[r]['count_matched']['verbal_fpr']:.2f}" for _, r in MODELS) + r" \\")
lines.append(r"\quad probe AUROC (all two-paragraph contexts) & " + " & ".join(f3(V[r]['count_matched']['probe_auc']) for _, r in MODELS) + r" \\")
lines.append(r"\quad probe AUROC among contexts verbalised INSUFFICIENT & " + " & ".join(f3(V[r]['count_matched']['stratified_abstains_auc']) for _, r in MODELS) + r" \\")
for key, lab in (("matched_fpr", "verbalised FPR"), ("precision_0.95", "0.95 precision")):
    lines.append(r"\quad threshold at " + lab + r": TPR / FPR & " + " & ".join(f"{V[r]['count_matched'][key]['tpr']:.3f} / {V[r]['count_matched'][key]['fpr']:.3f}" for _, r in MODELS) + r" \\")
    lines.append(r"\qquad recall on wrong abstentions; false alarms on correct abstentions & " + " & ".join(f"{V[r]['count_matched'][key]['recall_on_wrong_abstentions']:.3f}; {V[r]['count_matched'][key]['false_alarm_on_correct_abstentions']:.3f}" for _, r in MODELS) + r" \\")
lines.append(r"\quad mean probe score: sufficient answered / sufficient abstained & " + " & ".join(f"{V[r]['count_matched']['mean_score_gap_units']['suff_answered']:.2f} / {V[r]['count_matched']['mean_score_gap_units']['suff_abstained']:.2f}" for _, r in MODELS) + r" \\")
lines.append(r"\quad mean probe score: insufficient answered / insufficient abstained & " + " & ".join(f"{V[r]['count_matched']['mean_score_gap_units']['insuff_answered']:.2f} / {V[r]['count_matched']['mean_score_gap_units']['insuff_abstained']:.2f}" for _, r in MODELS) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{TAB}/verbal_internal.tex", "w").write("\n".join(lines))
for f in ("rep_ablation", "behaviour_control", "verbal_internal"):
    print(open(f"{TAB}/{f}.tex").read())
