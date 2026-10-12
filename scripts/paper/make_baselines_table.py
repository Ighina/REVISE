"""Baselines table for the main text: text classifiers, the model's own verbalised judgement
(answers vs INSUFFICIENT), its output confidence, and the activation probe, on the same test
increments (document split).  Also the uptake comparison restricted to answered increments.

Reads runs/<run>/text_baseline/results.json and probes/evidence_only/self_verbal_document.json,
and writes tables/baselines.tex (main text) and tables/uptake_given_answer.tex (appendix).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import MODELS, RUNS, f3, write_table  # noqa: E402

TASKS = [("sufficiency", "sufficiency"), ("sufficiency_matched", "sufficiency (matched)"), ("uptake", "uptake")]

lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
         r"& & \multicolumn{3}{c}{text classifiers} & \multicolumn{2}{c}{the model's own output} & \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
         r"Model & Task & TF-IDF & DeBERTa (q+added) & DeBERTa (+context) & verbalised & confidence & probe \\", r"\midrule"]
given = []
for mi, (mname, run) in enumerate(MODELS):
    tb = json.load(open(RUNS / run / "text_baseline" / "results.json"))
    sv = json.load(open(RUNS / run / "probes" / "evidence_only" / "self_verbal_document.json"))["tasks"]
    for ti, (t, tl) in enumerate(TASKS):
        e = tb.get(f"{t}/document", {}); v = sv[t]
        conf = max(v["confidence_top_prob"], v["confidence_neg_entropy"])
        probe = e.get("probe_auc")
        if probe is None:   # matched task has no text baseline on Llama/Mistral: take the probe number from the probe metrics
            m = json.load(open(RUNS / run / "probes" / "evidence_only" / t / "document" / "metrics.json")); probe = m["best_test_auc"]
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
write_table("baselines", lines)
write_table("uptake_given_answer", g_lines)
