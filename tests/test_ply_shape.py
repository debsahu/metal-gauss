"""bench/ply_shape.py -- the shape columns read back from a ply, with no GPU and no scene."""
import pytest
import torch

from bench.ply_shape import anchor, log_scales_from_ply, shape_of_ply


def _write(path, log_scales):
    from metal_gauss.train import export_ply
    n = log_scales.shape[0]
    export_ply({"means": torch.zeros(n, 3), "log_scales": log_scales,
                "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
                "logit_opac": torch.zeros(n),
                "sh_dc": torch.zeros(n, 1, 3), "sh_rest": torch.zeros(n, 15, 3)}, str(path))


def test_log_scales_are_read_back_without_an_exp_log_round_trip(tmp_path):
    """CATCHES the reader that goes through `io.load_ply`, which EXPONENTIATES. The ply
    stores log scales; exponentiating and re-logging is lossy and, worse, a reader that
    forgot the second half would report exp(scale) as a log scale and every shape column
    would be wrong in a plausible-looking way."""
    ls = torch.log(torch.tensor([[0.001, 0.02, 0.02], [1e-6, 1e-4, 0.02]]))
    f = tmp_path / "a.ply"
    _write(f, ls)
    assert torch.allclose(log_scales_from_ply(str(f)), ls, atol=0, rtol=0)


def test_shape_of_ply_matches_shape_metrics_on_the_same_scales(tmp_path):
    """The tool must not be a second implementation of the gate's own statistic."""
    from metal_gauss.train import shape_metrics
    g = torch.Generator().manual_seed(9)
    ls = torch.log(torch.rand(300, 3, generator=g) * 0.03 + 1e-6)
    f = tmp_path / "b.ply"
    _write(f, ls)
    got, want = shape_of_ply(str(f)), shape_metrics(ls)
    assert got["splats"] == 300
    for k in want:
        assert got[k] == pytest.approx(want[k], rel=1e-6, abs=1e-9), k


def test_anchor_refuses_fewer_than_three_arms():
    """CATCHES an anchor built from an n=2 mean -- the exact defect section 8.2 retracted a
    batch of claims over. A Band-1 cumulative check anchored on two runs inherits it."""
    rows = [{"aspect_p50": 0.5}, {"aspect_p50": 0.6}]
    with pytest.raises(RuntimeError, match="n >= 3"):
        anchor(rows, ["aspect_p50"])


def test_anchor_reports_mean_and_spread(tmp_path):
    rows = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    a = anchor(rows, ["x"])
    assert a["x"]["mean"] == pytest.approx(2.0) and a["x"]["spread"] == pytest.approx(2.0)
