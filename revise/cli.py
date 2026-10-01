"""Command-line entry point: ``python -m revise.cli <stage> --config configs/x.yaml``."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from revise import config as _cfg  # noqa: F401
from revise.experiment import Run, load_config

STAGES = ["screen", "collect", "conversational", "features", "probes", "transfer", "steer", "patch", "adaptive", "ablate", "text_baseline", "closedbook", "retrieve", "variants", "report", "all"]


def _setup_logging(run: Run) -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(run.dir / "pipeline.log")])
    for noisy in ("transformers", "httpx", "huggingface_hub", "urllib3", "datasets", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="REVISE experiment pipeline")
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--regime", "-r", default=None, help="restrict collect/features/probes to one regime")
    ap.add_argument("--max", type=int, default=None, help="cap number of conditions/increments (debug)")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--transfer-from", default=None, help="config of the run whose directions are applied (transfer)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run = Run(cfg)
    _setup_logging(run)
    log = logging.getLogger("revise")
    regimes = [args.regime] if args.regime else list(cfg["regimes"])
    stage = args.stage
    t0 = time.time()
    backend = None

    def be():
        nonlocal backend
        if backend is None:
            from revise.run.collect import get_backend
            backend = get_backend(cfg)
        return backend

    if stage in ("screen", "all"):
        from revise.run.collect import run_screen
        run_screen(cfg, backend=be())
    if stage in ("collect", "all"):
        from revise.run.collect import run_collect
        for r in regimes:
            run_collect(cfg, r, backend=be(), max_conditions=args.max)
    if stage in ("conversational", "all") and cfg["conversational"]["enabled"]:
        from revise.run.collect import run_conversational
        run_conversational(cfg, backend=be(), max_increments=args.max)
    if stage in ("features", "all"):
        from revise.analysis.features import build_features
        fr = regimes + (["conversational"] if cfg["conversational"]["enabled"] and not args.regime else [])
        for r in fr:
            build_features(run, r, cfg)
    if stage in ("probes", "all"):
        from revise.analysis.probes import run_probes
        pr = regimes + (["conversational"] if cfg["conversational"]["enabled"] and not args.regime else [])
        for r in pr:
            run_probes(run, r, cfg, tasks=args.tasks, split_modes=args.splits, n_jobs=args.n_jobs)
    if stage in ("transfer", "all"):
        from revise.analysis.probes import apply_directions
        base = cfg["conversational"]["base_regime"]
        mode = (args.splits or cfg["analysis"]["splits"])[0]
        if cfg["conversational"]["enabled"] and (run.features_dir("conversational") / "deltas.npy").exists():
            res = apply_directions(run, base, run, "conversational", mode, cfg, all_as_test=False)
            run.save_json("probes/transfer_conversational.json", res)
            log.info("stateless -> conversational transfer: %s", json.dumps({k: v["auc_at_src_best_layer"] for k, v in res.items()}))
        if args.transfer_from:
            src = Run(load_config(args.transfer_from))
            res = apply_directions(src, base, run, base, mode, cfg, all_as_test=True)
            run.save_json("probes/transfer_external.json", res)
            log.info("external transfer from %s: %s", src.dir, json.dumps({k: v["auc_at_src_best_layer"] for k, v in res.items()}))
    if stage in ("steer", "all"):
        from revise.run.steer import run_steer
        run_steer(cfg, backend=be())
    if stage in ("patch", "all"):
        from revise.run.patch import run_patch
        run_patch(cfg, backend=be())
    if stage in ("adaptive", "all"):
        from revise.run.adaptive import run_adaptive
        run_adaptive(cfg)
    if stage == "ablate":
        from revise.run.ablate import run_ablate
        run_ablate(cfg, backend=be())
    if stage == "variants":
        # ESTB linguistic-form axis: LLM paraphrase / partial variants (GPU), cross-dataset independent
        # sources (CPU); then rebuild conditions/increments so that `collect` picks up the new ones.
        from revise.data.variants import attach_variants, generate_variants
        from revise.data.independent import independent_variants
        from revise.data.trajectories import build_all
        from revise.data.schema import write_jsonl
        qs = run.questions()
        vcfg = cfg.get("variants") or {}
        rows = {}
        if vcfg.get("llm", True):
            rows = generate_variants(qs, be(), run.dir / "variants.jsonl", batch_size=int(os.environ.get("REVISE_BATCH_SIZE") or cfg["model"]["batch_size"]))
        n_llm = attach_variants(qs, rows)
        n_ind = independent_variants(qs) if vcfg.get("independent", True) else 0
        write_jsonl(run.questions_path, (q.to_json() for q in qs))
        d = cfg["data"]
        conds, incs = build_all(qs, seed=d["seed"], n_random_perms=d["n_random_perms"], include_full=d["include_full"], include_loo=d["include_loo"])
        write_jsonl(run.conditions_path, (c.to_json() for c in conds)); write_jsonl(run.increments_path, (i.to_json() for i in incs))
        log.info("variants attached: llm=%d independent=%d; rebuilt %d conditions / %d increments", n_llm, n_ind, len(conds), len(incs))
    if stage == "retrieve":
        from revise.data.retrieval import retrieve_for_questions
        rcfg = cfg["data"].get("retrieval") or {"method": "bm25", "k": 20}
        retrieve_for_questions(run.questions(), rcfg["method"], rcfg.get("k", 20), run.dir / f"retrieved_{rcfg['method']}.jsonl")
    if stage == "closedbook":
        from revise.run.closedbook import run_closedbook
        run_closedbook(cfg, backend=be())
    if stage == "text_baseline":
        from revise.analysis.text_baseline import run_text_baseline
        run_text_baseline(run, cfg)
    if stage in ("report", "all"):
        from revise.analysis.report import build_report
        p = build_report(run, list(cfg["regimes"]) + (["conversational"] if cfg["conversational"]["enabled"] else []))
        print(p.read_text())
    log.info("stage %s finished in %.1f s", stage, time.time() - t0)


if __name__ == "__main__":
    main()
