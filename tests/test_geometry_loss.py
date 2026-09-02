"""Geometry terms of the earthbyte/slam Stage 3 recipe, ported from Brush."""
import json
import math
from pathlib import Path

import pytest
import torch

FIX = Path(__file__).parent / "fixtures" / "normals_from_depth_slanted_plane.json"


# ---------------------------------------------------------------- flatten (Task 4)

def test_flatten_loss_is_mean_min_activated_scale_and_only_moves_min_axis():
    from metal_gauss.geometry_loss import flatten_loss
    ls = torch.log(torch.tensor([[0.10, 0.02, 0.05],
                                 [0.03, 0.03, 0.30]])).requires_grad_(True)
    loss = flatten_loss(ls)
    assert loss.item() == pytest.approx((0.02 + 0.03) / 2, rel=1e-6)
    loss.backward()
    g = ls.grad
    assert g[0, 1] != 0 and g[0, 0] == 0 and g[0, 2] == 0      # only the min axis of row 0
    assert g[1, 2] == 0                                        # never the max axis


def test_flatten_loss_uses_activated_scales_not_log_scales():
    """exp() is load-bearing: PlanarGS L_s is a length in metres. Dropping it makes the
    term negative for every sub-metre splat, so 'minimising' it INFLATES the thin axis.
    Here every scale is < 1, so the log-space mean is negative and the correct one is not."""
    from metal_gauss.geometry_loss import flatten_loss
    ls = torch.log(torch.full((4, 3), 0.05))
    assert flatten_loss(ls).item() == pytest.approx(0.05, rel=1e-6)
    assert flatten_loss(ls).item() > 0.0


def test_flatten_loss_gradient_pushes_the_thin_axis_DOWN():
    """Sign check. Gradient descent must SHRINK the smallest axis; a sign error here
    produces fatter splats and would be read as 'flatten does not work on this trainer'."""
    from metal_gauss.geometry_loss import flatten_loss
    ls = torch.log(torch.tensor([[0.10, 0.02, 0.05]])).requires_grad_(True)
    flatten_loss(ls).backward()
    assert ls.grad[0, 1] > 0            # d(loss)/d(log s_min) > 0  ->  descent shrinks it
    # and its magnitude is the activated scale / N, not 1/N
    assert ls.grad[0, 1].item() == pytest.approx(0.02, rel=1e-6)


def test_flatten_loss_is_scale_only_and_ignores_splat_count_scaling():
    """It is a MEAN, not a sum: doubling the splat count must not double the term, or the
    weight would silently depend on --budget."""
    from metal_gauss.geometry_loss import flatten_loss
    ls = torch.log(torch.tensor([[0.10, 0.02, 0.05]]))
    assert flatten_loss(ls.repeat(7, 1)).item() == pytest.approx(flatten_loss(ls).item(),
                                                                 rel=1e-6)


def test_flatten_flag_actually_reaches_the_training_loss(tmp_path):
    """Wiring, not arithmetic. This repo's failure log is full of flags that PARSE and do
    nothing -- LFS's `--train` is a no-op, its `--init=path.ply` is dead, and a harness
    once forwarded --budget so auto_budget() never ran in an 8-scene sweep. A unit test on
    flatten_loss() cannot see any of that.

    Built through the REAL parser so the arms differ in one flag and inherit every default
    from the same place the CLI does -- writing the namespace by hand in the test is the
    exact mistake `_run_report`'s docstring records.
    """
    pytest.importorskip("pycolmap")
    if not torch.backends.mps.is_available():
        pytest.skip("needs MPS")
    import numpy as np
    from PIL import Image
    from metal_gauss.train import build_parser, train

    (tmp_path / "sparse").mkdir(); (tmp_path / "images").mkdir()
    (tmp_path / "sparse" / "cameras.txt").write_text("1 PINHOLE 32 32 32 32 16 16\n")
    rng = np.random.default_rng(0)
    lines = []
    for i in range(3):
        lines.append(f"{i + 1} 1 0 0 0 0 0 {i * 0.3 - 0.3} 1 v{i}.png\n\n")
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(
            tmp_path / "images" / f"v{i}.png")
    (tmp_path / "sparse" / "images.txt").write_text("".join(lines))
    pts = rng.normal(0, 0.4, (60, 3)) + np.array([0, 0, 3.0])
    (tmp_path / "sparse" / "points3D.txt").write_text("".join(
        f"{i + 1} {x} {y} {z} 128 128 128 0.5\n" for i, (x, y, z) in enumerate(pts)))

    def run(w, out):
        a = build_parser().parse_args([
            "--colmap", str(tmp_path / "sparse"), "--images", str(tmp_path / "images"),
            "--steps", "40", "--budget", "400", "--max-resolution", "32",
            "--eval-every", "40", "--eval-split-every", "1000", "--seed", "0",
            "--num-downscales", "0", "--no-grow", "--sh-warmup", "0",
            "--flatten-loss-weight", str(w), "--export", str(out)])
        a.resolution_schedule = max(1, a.steps // 3)
        train(a)
        import plyfile
        v = plyfile.PlyData.read(str(out))["vertex"]
        s = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1))
        return float(np.median(s.min(1)))

    off = run(0.0, tmp_path / "off.ply")
    on = run(50.0, tmp_path / "on.ply")
    assert on < off * 0.9, f"flatten weight did not reach the loss: min-axis p50 {off} -> {on}"
