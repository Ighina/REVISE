"""Slow test: exercises the real backend on a small model if weights are cached."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def backend():
    from revise.model.backend import HFBackend
    try:
        return HFBackend("Qwen/Qwen2.5-0.5B-Instruct", num_threads=8, max_new_tokens=6)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"model unavailable: {e}")


def test_batch_matches_single(backend):
    from revise.prompts import build_stateless
    from revise.data.schema import Paragraph
    p = [Paragraph("g0", "Paris", "Paris is the capital of France.", "gold", 0)]
    msgs = [build_stateless("evidence_only", "What is the capital of France?", p),
            build_stateless("evidence_only", "What is the capital of Germany?", p)]
    out = backend.run_messages(msgs, gold_answers=["Paris", "Berlin"])
    assert out.anchor_hidden.shape == (2, backend.n_layers + 1, backend.hidden_size)
    for i, m in enumerate(msgs):
        s = backend.run_messages([m], gold_answers=[["Paris", "Berlin"][i]])
        assert s.generated[0] == out.generated[i]
        assert np.abs(s.anchor_hidden[0].astype(np.float32) - out.anchor_hidden[i].astype(np.float32)).max() < 2.0


def test_patch_and_steer_hooks(backend):
    from revise.model.backend import Intervention
    from revise.prompts import build_stateless
    m = build_stateless("closed_book", "What is the capital of France?", [])
    base = backend.run_messages([m])
    zero = backend.run_messages([m], interventions=[Intervention("steer", 5, torch.zeros(backend.hidden_size))])
    assert zero.generated == base.generated
    src = torch.tensor(base.anchor_hidden[0][5].astype(np.float32))[None]
    patched = backend.run_messages([m], interventions=[Intervention("patch", 5, src)])
    assert np.abs(patched.anchor_hidden[0][5].astype(np.float32) - src[0].numpy()).max() == 0
