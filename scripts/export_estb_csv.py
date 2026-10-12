"""Export the ESTB evidence trajectories as two CSV files.

The Evidence State Transition Benchmark is *constructed*, not downloaded: a
question with ordered gold hops and matched distractors is turned into evidence
contexts and into labelled transitions between them (see "Evidence conditions
and increments" in the README).  This script runs that construction and writes
it out as data, so the benchmark can be shared and used without this pipeline.

    estb_conditions.csv   one row per evidence context: the paragraphs the model
                          would be shown, and whether they are sufficient to
                          answer the question
    estb_increments.csv   one row per transition C_{k-1} -> C_k: the paragraph
                          that was added, what kind of evidence it is, and
                          whether the context crosses into sufficiency

Both files repeat the question, answer and hop count on every row, so each is
usable on its own.

The labels here are the ones that follow from the construction (`sufficient`,
`kind`, `becomes_sufficient`, `is_decisive`).  The paper's *behavioural* labels
-- uptake, correction, stability and the state_from/state_to transitions --
describe what a particular model did, so they only exist after `collect` has run
for that model; this export deliberately does not invent them.

The synthetic dataset is generated locally and needs neither a network
connection nor a model.  The benchmark datasets (musique, 2wiki, hotpotqa,
ragbench) are downloaded from the Hub on first use; note that `--select` cannot
restrict to the questions a model gets wrong closed-book, since that too needs a
collected run.

Examples
--------
    python scripts/export_estb_csv.py                          # synthetic, 500 questions
    python scripts/export_estb_csv.py --dataset musique --hops 2 --n-questions 200
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a checkout without installing

from revise import config  # noqa: E402
from revise.analysis.features import DECISIVE_KINDS  # noqa: E402
from revise.data.loaders import load_questions  # noqa: E402
from revise.data.schema import Condition, Paragraph, Question  # noqa: E402
from revise.data.trajectories import build_all  # noqa: E402

log = logging.getLogger("export_estb")

CONDITION_COLS = [
    "qid", "dataset", "question", "answer", "n_hops", "qtype", "relation_types",
    "cid", "condition_name", "sufficient", "n_paragraphs", "n_gold", "n_distractor",
    "has_false", "has_answer_ctrl", "has_redundant", "gold_hops_present",
    "order_tag", "inserted_position", "paragraph_pids", "paragraph_roles",
    "paragraph_titles", "context",
]
INCREMENT_COLS = [
    "qid", "dataset", "question", "answer", "n_hops", "qtype", "relation_types",
    "iid", "from_cid", "to_cid", "from_condition_name", "to_condition_name",
    "kind", "is_decisive", "becomes_sufficient", "from_sufficient", "to_sufficient",
    "order_tag", "control_group", "n_added", "added_pids", "added_roles",
    "added_titles", "added_text",
]


def render_context(paragraphs: List[Paragraph]) -> str:
    """The evidence block exactly as the prompt builder renders it."""
    return "\n\n".join(f"[{i + 1}] {p.render()}" for i, p in enumerate(paragraphs))


def question_fields(q: Question) -> dict:
    return {"qid": q.qid, "dataset": q.dataset, "question": q.question, "answer": q.answer,
            "n_hops": q.n_hops, "qtype": q.qtype, "relation_types": "|".join(q.relation_types)}


def condition_row(cond: Condition, q: Question) -> dict:
    ps = [Paragraph(**p) if isinstance(p, dict) else p for p in cond.paragraphs]
    return {
        **question_fields(q),
        "cid": cond.cid, "condition_name": cond.name, "sufficient": int(cond.sufficient),
        "n_paragraphs": len(ps), "n_gold": cond.n_gold, "n_distractor": cond.n_distractor,
        "has_false": int(cond.has_false), "has_answer_ctrl": int(cond.has_answer_ctrl),
        "has_redundant": int(cond.has_redundant),
        "gold_hops_present": "|".join(str(h) for h in cond.gold_hops),
        "order_tag": cond.order_tag,
        "inserted_position": "" if cond.inserted_position is None else cond.inserted_position,
        "paragraph_pids": "|".join(p.pid for p in ps),
        "paragraph_roles": "|".join(p.role for p in ps),
        "paragraph_titles": "|".join(p.title for p in ps),
        "context": render_context(ps),
    }


def increment_row(inc, q: Question, conds: dict) -> dict:
    to_cond = conds[inc.to_cid]
    ps = [Paragraph(**p) if isinstance(p, dict) else p for p in to_cond.paragraphs]
    added = [p for p in ps if p.pid in set(inc.added_pids)]
    return {
        **question_fields(q),
        "iid": inc.iid, "from_cid": inc.from_cid, "to_cid": inc.to_cid,
        "from_condition_name": conds[inc.from_cid].name if inc.from_cid in conds else "",
        "to_condition_name": to_cond.name,
        "kind": inc.kind, "is_decisive": int(inc.kind in DECISIVE_KINDS),
        "becomes_sufficient": int(inc.becomes_sufficient),
        "from_sufficient": int(inc.from_sufficient), "to_sufficient": int(inc.to_sufficient),
        "order_tag": inc.order_tag, "control_group": inc.control_group,
        "n_added": len(added), "added_pids": "|".join(p.pid for p in added),
        "added_roles": "|".join(p.role for p in added),
        "added_titles": "|".join(p.title for p in added),
        "added_text": "\n\n".join(p.render() for p in added),
    }


def write_csv(path: Path, cols: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s (%d rows, %.1f MB)", path, len(rows), path.stat().st_size / 1e6)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "musique", "2wiki", "hotpotqa", "ragbench"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-questions", type=int, default=500, help="questions to export")
    ap.add_argument("--hops", type=int, nargs="*", default=None, help="keep only these hop counts")
    ap.add_argument("--pool-limit", type=int, default=None, help="questions to load before filtering (benchmark datasets)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-random-perms", type=int, default=1, help="random hop orders per question, besides canonical and reverse")
    ap.add_argument("--no-full", action="store_true", help="omit the full-context condition")
    ap.add_argument("--no-loo", action="store_true", help="omit leave-one-out conditions")
    ap.add_argument("--out-dir", default=str(config.SCRATCH_ROOT / "estb_csv"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    kw = {"split": args.split, "hops": args.hops, "limit": args.pool_limit or args.n_questions}
    if args.dataset == "synthetic":
        kw["n_questions"] = max(args.n_questions * 3, 1000)   # the generator spreads over 1-3 hops
        kw["seed"] = args.seed
    qs = load_questions(args.dataset, **kw)[: args.n_questions]
    log.info("%s: %d questions (%d hops seen)", args.dataset, len(qs), len({q.n_hops for q in qs}))

    conds, incs = build_all(qs, seed=args.seed, n_random_perms=args.n_random_perms,
                            include_full=not args.no_full, include_loo=not args.no_loo)
    by_q = {q.qid: q for q in qs}
    by_cid = {c.cid: c for c in conds}
    log.info("built %d conditions and %d increments", len(conds), len(incs))

    out = Path(args.out_dir)
    write_csv(out / "estb_conditions.csv", CONDITION_COLS, [condition_row(c, by_q[c.qid]) for c in conds])
    write_csv(out / "estb_increments.csv", INCREMENT_COLS, [increment_row(i, by_q[i.qid], by_cid) for i in incs])

    kinds = {}
    for i in incs:
        kinds[i.kind] = kinds.get(i.kind, 0) + 1
    log.info("increment kinds: %s", dict(sorted(kinds.items(), key=lambda kv: -kv[1])))
    log.info("sufficient conditions: %d / %d", sum(c.sufficient for c in conds), len(conds))


if __name__ == "__main__":
    main()
