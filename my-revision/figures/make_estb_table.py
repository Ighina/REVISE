"""ESTB summary table: per axis cell, within-run probe and MuSiQue-direction transfer (source-unseen questions)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("REVISE_SCRATCH", "/scratch/users/iacopog/revise")
from revise.analysis.estb import cell_table
from revise.experiment import Run, load_config
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables")
RUNS = [("Benchmark gold + benchmark distractors", "musique2hop_qwen7b", "document"),
        ("Gold + E5-retrieved distractors", "musique2hop_qwen7b_e5_noise", "document"),
        ("Gold + BM25-retrieved distractors", "musique2hop_qwen7b_bm25_noise", "document"),
        ("All evidence E5-retrieved", "musique2hop_qwen7b_e5_full", "question"),
        ("All evidence BM25-retrieved", "musique2hop_qwen7b_bm25_full", "question"),
        ("+ LLM paraphrase / partial / independent", "musique2hop_qwen7b_variants", "document")]
src = Run(load_config("configs/musique_2hop_qwen7b.yaml"))
rows_tex, cells = [], {}
cfgmap = {"musique_2hop_qwen7b": "configs/musique_2hop_qwen7b.yaml"}
for name, run_name, split in RUNS:
    cfg = load_config(f"configs/{run_name.replace('musique2hop', 'musique_2hop')}.yaml" if run_name != "musique2hop_qwen7b" else "configs/musique_2hop_qwen7b.yaml")
    run = Run(cfg); s = json.load(open(run.probes_dir("evidence_only") / "summary.json"))
    rows = cell_table(run, src, mode=split); cells[run_name] = rows
    get = lambda task, key: next((r[key] for r in rows if r["task"] == task and r["axis"] == "all"), None)
    own = lambda task: s.get(f"deltas/{task}/{split}", {}).get("best_test_auc")
    p = json.load(open(run.dir / "patch" / "results.json")); mm = {x["variant"]: x for x in p["runs"] if x["task"] == "state_sufficiency" and x["mode"] == "multi"}
    a = json.load(open(run.dir / "ablate" / "results.json")); pj = {x["direction"]: x for x in a["runs"] if x["experiment"] == "projout" and x["direction_kind"] == "dm" and x["set"] == "sufficient" and x["task"] == "state_sufficiency"}
    nq = len(run.questions())
    f = lambda x: "--" if x is None else f"{x:.3f}"
    same = run_name == "musique2hop_qwen7b"
    rows_tex.append(f"{name} & {nq} & {f(own('sufficiency'))} & {f(own('sufficiency_matched'))} & {f(own('uptake'))} & "
                    f"{'(source)' if same else f(get('sufficiency','auc_musique_dir'))} & {'(source)' if same else f(get('uptake','auc_musique_dir'))} & "
                    f"{mm['none']['answered']:.2f} & {mm['full']['answered']:.2f} & {mm['parallel']['answered']:.2f} & {pj['probe']['answered']:.2f} \\\\")
    print(name, rows_tex[-1])
# quality-state cells from the variants run (linguistic form)
v = cells["musique2hop_qwen7b_variants"]
qrows = [r for r in v if r["task"] == "sufficiency" and r["axis"] == "quality" and r["auc_own_probe"] is not None]
tex = [r"\begin{tabular}{lrccccccccc}", r"\toprule",
       r"& & \multicolumn{3}{c}{within-run probe (AUROC)} & \multicolumn{2}{c}{MuSiQue dir.} & \multicolumn{4}{c}{answered rate} \\",
       r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-11}",
       r"Evidence source & $n_q$ & suff. & matched & uptake & suff. & uptake & unpatched & full & $\parallel v$ & proj.-out $\vdm$ \\", r"\midrule"] + rows_tex + [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/estb.tex", "w").write("\n".join(tex))
tex2 = [r"\begin{tabular}{lrcc}", r"\toprule", r"Quality state vs.\ decisive increment & $n$ & own probe & MuSiQue dir. \\", r"\midrule"] + \
       [f"{r['cell']} & {r['n']} & {r['auc_own_probe']:.3f} & {'--' if r['auc_musique_dir'] is None else '%.3f' % r['auc_musique_dir']} \\\\" for r in sorted(qrows, key=lambda r: r['auc_own_probe'])] + [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/estb_quality.tex", "w").write("\n".join(tex2))
json.dump(cells, open(f"{OUT}/estb_cells.json", "w"), indent=1, default=float)
