"""Plant a direction in synthetic Δh and check the probe machinery recovers it."""
import json

import numpy as np
import yaml

from revise.analysis.probes import run_probes
from revise.data.schema import write_jsonl
from revise.experiment import Run, load_config
from tests.test_trajectories import _q


def test_probe_recovers_planted_direction(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({"run_name": "pytest_synthetic", "analysis": {"layers": [0, 1], "splits": ["question"], "n_passage_svd": 4}}))
    cfg = load_config(cfg_path)
    run = Run(cfg)
    qs = [_q(f"q{i}", 2, answer=f"Ans{i}") for i in range(60)]
    write_jsonl(run.questions_path, (q.to_json() for q in qs))
    rng = np.random.default_rng(0)
    d, nL = 32, 2
    v = rng.standard_normal(d); v /= np.linalg.norm(v)
    rows, mats = [], []
    kinds = ["decisive", "distractor", "redundant", "false", "answer_ctrl", "gold_nondecisive"]
    for q in qs:
        for k in kinds:
            pos = k == "decisive"
            noisy_flag = bool(rng.random() < (0.7 if pos else 0.3))   # correlated with, but not identical to, the label
            x = rng.standard_normal((nL, d)).astype(np.float32)
            x[1] += (3.0 if pos else 0.0) * v            # planted only at layer 1
            mats.append(x.astype(np.float16))
            rows.append({"iid": f"{q.qid}-{k}", "qid": q.qid, "from_cid": "C[g0]", "to_cid": f"C[{k}]", "kind": k,
                         "control_group": f"{q.qid}|C[g0]", "order_tag": "canonical", "becomes_sufficient": pos,
                         "from_sufficient": False, "to_sufficient": pos, "is_decisive": pos,
                         "state_from": "insufficient", "state_to": "correct" if pos and rng.random() < 0.8 else "wrong",
                         "pred_from": "INSUFFICIENT", "pred_to": "x", "revision": True, "correction": pos,
                         "d_tokens": float(rng.integers(50, 150)), "to_tokens": 300.0, "d_gold_lp": float(rng.standard_normal()),
                         "d_entropy": 0.0, "d_top_prob": 0.0, "answer_in_added": noisy_flag, "answer_in_to": noisy_flag, "answer_in_from": False,
                         "relevance": float(rng.random()), "n_added": 1, "added_wc": 40, "inserted_pos_rel": 0.5,
                         "added_text": " ".join(rng.choice(["alpha", "beta", "gamma", "delta", "eps"], 8)), "n_hops": 2, "false_answer": None})
    fd = run.features_dir("evidence_only")
    np.save(fd / "deltas.npy", np.stack(mats))
    write_jsonl(fd / "deltas_meta.jsonl", rows)
    res = run_probes(run, "evidence_only", cfg, tasks=["sufficiency", "sufficiency_matched"], n_jobs=2)
    e = res["deltas/sufficiency/question"]
    assert e["best_layer"] == 1
    assert e["best_test_auc"] > 0.95
    assert e["test_auc"][0] < 0.75                      # layer 0 carries no signal
    dirs = np.load(run.probes_dir("evidence_only") / "sufficiency" / "question" / "directions.npy")
    assert abs(dirs[1] @ v) > 0.8
    nz = e["nuisance"]
    assert nz["residual_probe_auc"] > 0.85              # signal survives regressing out (noisy) nuisance
    assert nz["auc_gain_beyond_nuisance"] > 0.1
    assert "auc_gain_beyond_nuisance" in nz
