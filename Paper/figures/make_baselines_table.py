"""Baselines table for the main text: text classifiers, the model's own verbalised judgement
(answers vs INSUFFICIENT), its output confidence, and the activation probe, on the same test
increments (document split).  Also the uptake comparison restricted to answered increments.

Reads runs/<run>/text_baseline/results.json and probes/evidence_only/self_verbal_document.json.
Writes tables/baselines.tex and tables/uptake_given_answer.tex (in this script's ../tables and,
if present, ../../Paper_acl/tables).
"""
import json, os
from sklearn.metrics import roc_auc_score

R = os.environ.get("REVISE_SCRATCH", "/scratch/users/iacopog/revise") + "/runs"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTS = [os.path.join(os.path.dirname(HERE), "tables")]
alt = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Paper_acl", "tables")
if os.path.isdir(alt):
    OUTS.append(alt)
MODELS = [("Qwen2.5-7B", "musique2hop_qwen7b"), ("Llama-3.1-8B", "musique2hop_llama8b"), ("Mistral-7B", "musique2hop_mistral7b")]
TASKS = [("sufficiency", "sufficiency"), ("sufficiency_matched", "sufficiency (matched)"), ("uptake", "uptake")]
f3 = lambda x: "--" if x is None else f"{x:.3f}"

lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
         r"& & \multicolumn{3}{c}{text classifiers} & \multicolumn{2}{c}{the model's own output} & \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
         r"Model & Task & TF-IDF & DeBERTa (q+added) & DeBERTa (+context) & verbalised & confidence & probe \\", r"\midrule"]
given = []
for mi, (mname, run) in enumerate(MODELS):
    tb = json.load(open(f"{R}/{run}/text_baseline/results.json"))
    sv = json.load(open(f"{R}/{run}/probes/evidence_only/self_verbal_document.json"))["tasks"]
    for ti, (t, tl) in enumerate(TASKS):
        e = tb.get(f"{t}/document", {}); v = sv[t]
        conf = max(v["confidence_top_prob"], v["confidence_neg_entropy"])
        probe = e.get("probe_auc")
        if probe is None:   # matched task has no text baseline on Llama/Mistral: take the probe number from the probe metrics
            m = json.load(open(f"{R}/{run}/probes/evidence_only/{t}/document/metrics.json")); probe = m["best_test_auc"]
        lines.append(f"{mname if ti == 0 else ''} & {tl} & {f3(e.get('tfidf[q+added]_auc'))} & {f3(e.get('encoder[q+added]_auc'))} & "
                     f"{f3(e.get('encoder[q+added+context]_auc'))} & {f3(v['verbalised_answers'])} & {f3(conf)} & {f3(probe)} \\\\")
    if mi < len(MODELS) - 1:
        lines.append(r"\midrule")
    g = sv["uptake"]["given_answer"]
    # combined: answered first, then ranked by the score (positives are all answered)
    a = sv["uptake"]["answer_rate_neg"]
    comb = lambda auc_within: (1 - a) + a * auc_within
    given.append(f"{mname} & {g['n_test_answered']} & {g['pos_rate']:.2f} & {f3(g['confidence_neg_entropy'])} & {f3(g['stored_probe_auc'])} & {f3(g['within_answered_probe_auc'])} & "
                 f"{f3(sv['uptake']['verbalised_answers'])} & {f3(comb(g['confidence_neg_entropy']))} & {f3(comb(g['within_answered_probe_auc']))} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
g_lines = [r"\begin{tabular}{lrccccccc}", r"\toprule",
           r"& & & \multicolumn{3}{c}{answered increments only} & \multicolumn{3}{c}{all decisive increments} \\",
           r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
           r"Model & $n$ & correct & confidence & probe & probe (refit) & verbalised & + confidence & + probe \\", r"\midrule"] + given + [r"\bottomrule", r"\end{tabular}"]
for o in OUTS:
    open(f"{o}/baselines.tex", "w").write("\n".join(lines))
    open(f"{o}/uptake_given_answer.tex", "w").write("\n".join(g_lines))
print("\n".join(lines)); print("\n".join(g_lines))
