from revise.analysis.splits import SPLIT_MODES, check_disjoint, make_split
from tests.test_trajectories import _q


def test_splits_disjoint():
    qs = [_q(f"q{i}", 2, answer=f"Ans{i % 4}") for i in range(40)]
    # make some questions share a gold document
    for i in range(0, 40, 5):
        qs[i].gold[0].title = "SHARED"
    for mode in SPLIT_MODES:
        assign = make_split(qs, mode, test_frac=0.3, seed=0)
        assert set(assign.values()) <= {"train", "test"}
        assert check_disjoint(qs, assign, mode)
        n_test = sum(v == "test" for v in assign.values())
        assert 0 < n_test < 40
