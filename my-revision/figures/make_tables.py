"""Build the paper's LaTeX tables + a consolidated numbers.json from run artifacts."""
import json, os, glob
import numpy as np

R = os.environ.get("REVISE_SCRATCH", "/scratch/users/iacopog/revise") + "/runs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables")
MODELS = {"Qwen2.5-7B": "musique2hop_qwen7b", "Llama-3.1-8B": "musique2hop_llama8b", "Mistral-7B": "musique2hop_mistral7b"}
EXTRA = {"Qwen 3--4 hop": "musiquefull_qwen7b", "Qwen 2Wiki": "twowiki_qwen7b"}
NUM = {}


def J(run, rel):
    p = f"{R}/{run}/{rel}"
    return json.load(open(p)) if os.path.exists(p) else None


def f3(x): return "--" if x is None else f"{x:.3f}"
def f2(x): return "--" if x is None else f"{x:.2f}"


def ci(run, task, split):
    b = J(run, "probes/bootstrap_ci.json")
    if b and f"{task}/{split}" in b:
        lo, hi = b[f"{task}/{split}"]["ci95"]; return f"[{lo:.2f},{hi:.2f}]"
    return ""


def table_probes():
    tasks = [("sufficiency", "Sufficiency"), ("sufficiency_matched", "Sufficiency (matched)"), ("uptake", "Uptake"),
             ("correction", "Correction"), ("stability", "Stability"), ("state_sufficiency", "State sufficiency")]
    rows = []
    for t, name in tasks:
        cells = []
        for m, run in MODELS.items():
            s = J(run, "probes/evidence_only/summary.json")
            e = s.get(f"deltas/{t}/document") or s.get(f"states/{t}/document")
            nz = e.get("nuisance", {}) if e else {}
            NUM.setdefault("probes", {}).setdefault(m, {})[t] = {"auc": e["best_test_auc"], "layer": e["best_layer"], "n_layers": max(e["layers"]),
                                                                 "nuisance": nz.get("nuisance_only_auc"), "residual": nz.get("residual_probe_auc"), "gain": nz.get("auc_gain_beyond_nuisance"), "ci": ci(run, t, "document")}
            cells.append(f"{e['best_test_auc']:.3f} {ci(run, t, 'document')} & {f2(nz.get('nuisance_only_auc'))} & {f2(nz.get('auc_gain_beyond_nuisance')).replace('-', '$-$') if nz else '--'}")
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    tex = [r"\begin{tabular}{l" + "ccc" * 3 + "}", r"\toprule",
           r"& \multicolumn{3}{c}{Qwen2.5-7B} & \multicolumn{3}{c}{Llama-3.1-8B} & \multicolumn{3}{c}{Mistral-7B} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
           r"Task & AUROC [95\% CI] & nuis. & gain & AUROC [95\% CI] & nuis. & gain & AUROC [95\% CI] & nuis. & gain \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/probes_models.tex", "w").write("\n".join(tex))


def table_splits():
    rows = []
    for m, run in list(MODELS.items()) + list(EXTRA.items()):
        s = J(run, "probes/evidence_only/summary.json")
        if not s: continue
        for t, name in [("sufficiency", "Suff."), ("uptake", "Uptake")]:
            cells = []
            for sp in ["question", "document", "answer", "relation"]:
                e = s.get(f"deltas/{t}/{sp}"); cells.append(f3(e["best_test_auc"]) if e else "--")
                NUM.setdefault("splits", {}).setdefault(m, {}).setdefault(t, {})[sp] = e["best_test_auc"] if e else None
            rows.append(f"{m} & {name} & " + " & ".join(cells) + r" \\")
    tex = [r"\begin{tabular}{llcccc}", r"\toprule", r"Run & Task & question & document & answer & relation \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/splits.tex", "w").write("\n".join(tex))


def table_text_baseline():
    rows = []
    for m, run in MODELS.items():
        for fn, tag in [("text_baseline/results.json", "reduced budget"), ("text_baseline/results_full_budget.json", "full budget")]:
            tb = J(run, fn)
            if not tb: continue
            for key, e in tb.items():
                t, sp = key.split("/")
                if sp != "document": continue
                NUM.setdefault("text", {}).setdefault(m, {})[f"{t}/{tag}"] = e
                rows.append(f"{m} & {tag} & {t.replace('_', ' ')} & {f3(e.get('tfidf[q+added]_auc'))} & {f3(e.get('encoder[q+added]_auc'))} & {f3(e.get('encoder[q+added+context]_auc'))} & {f3(e.get('probe_auc'))} \\\\")
    tex = [r"\begin{tabular}{lllcccc}", r"\toprule", r"Model & Budget & Task & TF-IDF & DeBERTa (q+added) & DeBERTa (+context) & Probe \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/text_baseline.tex", "w").write("\n".join(tex))


def table_causal():
    rows = []
    for m, run in list(MODELS.items()) + [("Qwen 3--4 hop", "musiquefull_qwen7b")]:
        p = J(run, "patch/results.json"); a = J(run, "ablate/results.json")
        if not p or not a: continue
        mm = {x["variant"]: x for x in p["runs"] if x["task"] == "state_sufficiency" and x["mode"] == "multi"}
        sub = {x["variant"]: x for x in a["runs"] if x["experiment"] == "subspace"}
        pj = {(x["direction_kind"], x["direction"]): x for x in a["runs"] if x["experiment"] == "projout" and x["task"] == "state_sufficiency" and x["set"] == "sufficient"}
        NUM.setdefault("causal", {})[m] = {"patch": {k: (v["correct"], v["answered"]) for k, v in mm.items()}, "subspace": {k: (v["correct"], v["answered"]) for k, v in sub.items()},
                                          "projout": {f"{k[0]}/{k[1]}": v["answered"] for k, v in pj.items()}, "layer": mm["none"]["layer"], "n_layers": mm["none"]["n_layers_patched"]}
        c = lambda v: f"{v['correct']:.2f}/{v['answered']:.2f}"
        rows.append(f"{m} & {c(mm['none'])} & {c(mm['full'])} & {c(mm['parallel'])} & {c(mm['orthogonal'])} & {c(sub['pca256_rand'])} & {c(sub['pca1'])} & {c(sub['pca256'])} & {c(sub['unembed_add'])} & {pj[('dm','probe')]['answered']:.2f} & {pj[('dm','random')]['answered']:.2f} & {pj[('lr','probe')]['answered']:.2f} \\\\")
    tex = [r"\begin{tabular}{l" + "c" * 11 + "}", r"\toprule",
           r"& \multicolumn{8}{c}{Patching the insufficient prompt (correct/answered)} & \multicolumn{3}{c}{Projection-out on sufficient (answered)} \\",
           r"\cmidrule(lr){2-9}\cmidrule(lr){10-12}",
           r"Model & none & full & $\parallel v$ & $\perp v$ & rand-256 & PC1 & PC256 & unemb. & $v_{\mathrm{dm}}$ & random & $v_{\mathrm{lr}}$ \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/causal.tex", "w").write("\n".join(tex))


def table_steer():
    rows = []
    for m, run in MODELS.items():
        a = J(run, "ablate/results_dmsteer_small.json")
        rs = [x for x in a["runs"] if x["direction"] == "probe"]
        for al in (-1.0, -0.5, 0.5, 1.0):
            g = {x["set"]: x for x in rs if x["alpha"] == al}
            NUM.setdefault("steer", {}).setdefault(m, {})[al] = {k: (v["correct"], v["answered"], v.get("false_follow")) for k, v in g.items()}
            rows.append(f"{m if al == -1.0 else ''} & ${al:+.1f}$ & {g['sufficient']['correct']:.2f}/{g['sufficient']['answered']:.2f} & {g['insufficient']['answered']:.2f} & {g['distractor_only']['answered']:.2f} & {g['false']['answered']:.2f}/{g['false']['false_follow']:.2f} & {g['answer_ctrl']['answered']:.2f} \\\\")
    tex = [r"\begin{tabular}{llccccc}", r"\toprule", r"Model & $\alpha$ & sufficient corr./ans. & insufficient ans. & distractors ans. & false ans./followed & answer-string ans. \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/steer.tex", "w").write("\n".join(tex))


def table_transfer():
    rows = []
    q = "musique2hop_qwen7b"
    conv = J(q, "probes/transfer_conversational.json"); wiki = J("twowiki_qwen7b", "probes/transfer_external.json"); mh = J("musiquefull_qwen7b", "probes/transfer_external.json")
    for t in ["sufficiency", "sufficiency_matched", "uptake", "correction", "stability", "revision"]:
        cells = [f3(conv[t]["auc_at_src_best_layer"]) if conv and t in conv else "--", f3(wiki[t]["auc_at_src_best_layer"]) if wiki and t in wiki else "--", f3(mh[t]["auc_at_src_best_layer"]) if mh and t in mh else "--"]
        NUM.setdefault("transfer", {})[t] = {"conversational": conv[t]["auc_at_src_best_layer"] if conv and t in conv else None, "2wiki": wiki[t]["auc_at_src_best_layer"] if wiki and t in wiki else None, "multihop": mh[t]["auc_at_src_best_layer"] if mh and t in mh else None}
        rows.append(f"{t.replace('_', ' ')} & " + " & ".join(cells) + r" \\")
    tex = [r"\begin{tabular}{lccc}", r"\toprule", r"Direction (MuSiQue 2-hop, stateless) & $\to$ conversational & $\to$ 2WikiMultiHopQA & $\to$ MuSiQue 3--4 hop \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/transfer.tex", "w").write("\n".join(tex))


def table_behaviour():
    rows = []
    names = [("none", "no evidence"), ("distractors_only", "distractors only"), ("gold_single", "one hop"), ("sufficient_min", "minimal sufficient"),
             ("penultimate_plus_false", "one hop + false final hop"), ("sufficient_plus_distractor", "sufficient + distractor"), ("full", "full context (20 par.)")]
    beh = {m: [json.loads(l) for l in open(f"{R}/{run}/behavior/evidence_only.jsonl")] for m, run in MODELS.items()}
    for key, name in names:
        cells = []
        for m in MODELS:
            b = [x for x in beh[m] if x["name"] == key]
            c, a = np.mean([x["correct"] for x in b]), np.mean([x["insufficient"] for x in b])
            NUM.setdefault("behaviour", {}).setdefault(m, {})[key] = (float(c), float(a))
            cells.append(f"{c:.2f} / {a:.2f}")
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    tex = [r"\begin{tabular}{lccc}", r"\toprule", r"Condition (correct / abstain) & Qwen2.5-7B & Llama-3.1-8B & Mistral-7B \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/behaviour.tex", "w").write("\n".join(tex))


def misc():
    q = "musique2hop_qwen7b"
    NUM["closedbook"] = J(q, "closedbook/results.json")
    NUM["adaptive"] = {m: J(run, "adaptive/results.json") for m, run in MODELS.items()}
    s = J(q, "probes/evidence_only/summary.json")
    NUM["kind_projection"] = s["deltas/sufficiency/document"]["kind_projection_means"]
    NUM["cosine"] = s["transfer/document"]["cosine"]
    NUM["counts"] = {m: {"conditions": len(open(f"{R}/{run}/behavior/evidence_only.jsonl").readlines())} for m, run in MODELS.items()}
    w = J("twowiki_qwen7b", "probes/evidence_only/summary.json")
    NUM["twowiki_probes"] = {t: w[f"deltas/{t}/document"]["best_test_auc"] for t in ["sufficiency", "sufficiency_matched", "uptake", "correction"]}


EXT = [("HotpotQA", "hotpotqa_qwen7b", "document"), ("RAGBench", "ragbench_qwen7b", "question"), ("RAGBench (held-out sub-datasets)", "ragbench_qwen7b", "relation"),
       ("Synthetic", "synthetic_qwen7b", "question"), ("Synthetic (held-out relation chains)", "synthetic_qwen7b", "relation")]


def table_external():
    rows = []
    for name, run, split in EXT:
        s = J(run, "probes/evidence_only/summary.json"); t = J(run, "probes/transfer_external.json")
        p = J(run, "patch/results.json"); a = J(run, "ablate/results.json")
        if not s: continue
        g = lambda task: s.get(f"deltas/{task}/{split}", {}).get("best_test_auc")
        tr = (lambda task: t[task]["auc_at_src_best_layer"] if t and task in t else None)
        cells = [f3(g("sufficiency")), f3(g("sufficiency_matched")), f3(g("uptake")), f3(g("correction")), f3(tr("sufficiency")), f3(tr("uptake"))]
        NUM.setdefault("external", {})[name] = {"sufficiency": g("sufficiency"), "matched": g("sufficiency_matched"), "uptake": g("uptake"), "correction": g("correction"),
                                                "transfer_sufficiency": tr("sufficiency"), "transfer_uptake": tr("uptake")}
        if p and "held-out" not in name:
            mm = {x["variant"]: x for x in p["runs"] if x["task"] == "state_sufficiency" and x["mode"] == "multi"}
            pj = {x["direction"]: x for x in a["runs"] if x["experiment"] == "projout" and x["task"] == "state_sufficiency" and x["direction_kind"] == "dm" and x["set"] == "sufficient"} if a else {}
            cells += [f"{mm['none']['answered']:.2f}", f"{mm['full']['answered']:.2f}", f"{mm['parallel']['answered']:.2f}", f"{mm['orthogonal']['answered']:.2f}",
                      f"{pj['probe']['answered']:.2f}" if pj else "--", f"{pj['random']['answered']:.2f}" if pj else "--"]
            NUM["external"][name]["patch"] = {k: (v["correct"], v["answered"]) for k, v in mm.items()}
        else:
            cells += ["--"] * 6
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    tex = [r"\begin{tabular}{l" + "c" * 12 + "}", r"\toprule",
           r"& \multicolumn{4}{c}{within-dataset probes (AUROC)} & \multicolumn{2}{c}{MuSiQue dir.} & \multicolumn{4}{c}{patched insufficient (answered)} & \multicolumn{2}{c}{proj.-out (answered)} \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-11}\cmidrule(lr){12-13}",
           r"Dataset & suff. & matched & uptake & corr. & suff. & uptake & none & full & $\parallel v$ & $\perp v$ & $\vdm$ & random \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/external.tex", "w").write("\n".join(tex))


def table_groups():
    rows = []
    for name, run, split, gname, order in [("HotpotQA", "hotpotqa_qwen7b", "document", "closed_book", ["incorrect", "correct"]),
                                            ("HotpotQA", "hotpotqa_qwen7b", "document", "qtype", ["bridge", "comparison"]),
                                            ("RAGBench", "ragbench_qwen7b", "question", "domain", ["biomedical", "finance", "technical", "customer_support", "general", "wikipedia"]),
                                            ("RAGBench", "ragbench_qwen7b", "question", "n_hops", ["1", "2"]),
                                            ("Synthetic", "synthetic_qwen7b", "question", "n_hops", ["1", "2", "3"])]:
        g = J(run, f"probes/group_auc_{split}.json")
        if not g: continue
        for grp in order:
            cells = []
            for task in ["sufficiency", "uptake", "correction"]:
                v = g.get(task, {}).get("groups", {}).get(gname, {}).get(grp)
                cells.append(f"{v['auc']:.3f} ({v['n']})" if v and v["auc"] is not None else "--")
            rows.append(f"{name} & {gname.replace('_', ' ')} & {grp.replace('_', ' ')} & " + " & ".join(cells) + r" \\")
    tex = [r"\begin{tabular}{lllccc}", r"\toprule", r"Dataset & grouping & group & sufficiency & uptake & correction \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/groups.tex", "w").write("\n".join(tex))


if __name__ == "__main__":
    for f in (table_probes, table_splits, table_text_baseline, table_causal, table_steer, table_transfer, table_behaviour, table_external, table_groups, misc):
        try:
            f(); print("ok", f.__name__)
        except Exception as ex:
            print("FAILED", f.__name__, repr(ex))
    json.dump(NUM, open(f"{OUT}/numbers.json", "w"), indent=1, default=float)
