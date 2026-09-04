"""Task 21: the photometric ceiling probe's machinery, tested against the Brush
forms it claims to reproduce.

Every test here exists because a specific wrong implementation would otherwise
produce a NULL -- "no photometric component to recover" -- which is the answer
the pre-registered decision rule keys on. A fitter that silently does not fit
and a scene with nothing to fix are indistinguishable from the delta alone.
"""

import json
import math

import numpy as np
import pytest
import torch

from bench import lpips_attr as LA


# --------------------------------------------------------------- bilateral grid

def _reference_slice(grid: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """Explicit 8-corner trilinear slice, transcribed from Brush's
    bilagrid_kernels.rs:37-70 (sample_point), :84-93 (corner_weight) and
    :120-146 (interpolate, color_component). Deliberately a slow independent
    transcription -- it is the reference the vectorised implementation is
    checked against, so it must not share code with it.
    """
    C, GL, GH, GW = grid.shape
    assert C == 12
    H, W, _ = rgb.shape
    out = torch.zeros_like(rgb)
    for py in range(H):
        for px in range(W):
            r, g, b = (float(v) for v in rgb[py, px])
            x = px * (GW - 1) / max(W - 1, 1)
            y = py * (GH - 1) / max(H - 1, 1)
            raw_z = (0.299 * r + 0.587 * g + 0.114 * b) * (GL - 1)
            z = min(max(raw_z, 0.0), float(GL - 1))
            x0, y0, z0 = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
            x1, y1, z1 = min(x0 + 1, GW - 1), min(y0 + 1, GH - 1), min(z0 + 1, GL - 1)
            tx, ty, tz = x - math.floor(x), y - math.floor(y), z - math.floor(z)
            coef = []
            for c in range(12):
                v = 0.0
                for corner in range(8):
                    cx = x1 if corner & 1 else x0
                    cy = y1 if corner & 2 else y0
                    cz = z1 if corner & 4 else z0
                    wx = tx if corner & 1 else 1.0 - tx
                    wy = ty if corner & 2 else 1.0 - ty
                    wz = tz if corner & 4 else 1.0 - tz
                    v += float(grid[c, cz, cy, cx]) * wx * wy * wz
                coef.append(v)
            col = [r, g, b, 1.0]
            for row in range(3):
                out[py, px, row] = sum(coef[row * 4 + k] * col[k] for k in range(4))
    return out


def test_bilagrid_identity_leaves_the_image_unchanged():
    """Catches a wrong coefficient layout. Brush's identity sets channels 0, 5
    and 10 -- the diagonal of a ROW-MAJOR 3x4 -- and any other convention
    (column-major, 4x3, channels 0/4/8) makes the identity grid a colour
    scramble instead of a no-op."""
    torch.manual_seed(0)
    rgb = torch.rand(7, 5, 3)
    grid = LA.bilagrid_identity(8, 4, 4)
    out = LA.bilagrid_apply(grid, rgb)
    assert torch.allclose(out, rgb, atol=1e-6), (out - rgb).abs().max()


def test_bilagrid_matches_the_explicit_trilinear_reference():
    """Catches align_corners, padding mode, axis order and the guidance scale.
    grid_sample's conventions differ from Brush's kernel in every one of those,
    and each difference is a silent few-percent change, not an error."""
    torch.manual_seed(1)
    grid = LA.bilagrid_identity(6, 5, 4) + 0.3 * torch.randn(12, 6, 5, 4)
    rgb = torch.rand(9, 11, 3)
    got = LA.bilagrid_apply(grid, rgb)
    ref = _reference_slice(grid, rgb)
    assert torch.allclose(got, ref, atol=2e-5), (got - ref).abs().max()


def test_bilagrid_guidance_is_bt601_luminance_and_not_the_mean():
    """A grid constant in x and y but varying along guidance turns the output
    into a pure readout of the guidance coordinate. BT.601 and the channel mean
    agree on grey and differ everywhere else, so this fixture separates them:
    the colour below is chosen to make the two differ by more than a slice."""
    GL = 8
    grid = torch.zeros(12, GL, 2, 2)
    for z in range(GL):
        # row 0 of the affine outputs a constant equal to z/(GL-1): only the
        # bias column (channel 3) is set, so the output does not depend on rgb.
        grid[3, z] = z / (GL - 1)
    rgb = torch.tensor([[[1.0, 0.0, 0.0]]])          # pure red
    got = float(LA.bilagrid_apply(grid, rgb)[0, 0, 0])
    assert got == pytest.approx(0.299, abs=1e-5), got
    assert abs(got - 1.0 / 3.0) > 0.03, "BT.601 and the channel mean are not separated"


def test_bilagrid_tv_matches_brushs_form():
    """Brush sums the MEAN squared first difference along each of the three
    axes (bilagrid.rs:340-348). Summing the means and meaning the sums differ by
    a constant factor, which silently rescales the only regulariser the fit has."""
    torch.manual_seed(2)
    grid = torch.randn(12, 5, 4, 3)
    dx = (grid[..., 1:] - grid[..., :-1]) ** 2
    dy = (grid[:, :, 1:] - grid[:, :, :-1]) ** 2
    dz = (grid[:, 1:] - grid[:, :-1]) ** 2
    want = dx.mean() + dy.mean() + dz.mean()
    assert LA.bilagrid_tv(grid) == pytest.approx(float(want), rel=1e-6)


# ------------------------------------------------------------------ PPISP forms

def test_crf_identity_raw_params_are_the_identity_curve():
    """The raw ZEROS are NOT the identity here -- softplus(0) = 0.693 gives
    toe = shoulder = 0.993 and gamma = 0.793, a real curve. Initialising a
    'no-op' fitter at raw zeros starts it off identity and makes any recovered
    dLPIPS partly an artefact of the initialisation."""
    x = torch.linspace(0.0, 1.0, 257)[None, :, None].repeat(1, 1, 3)
    p = LA.crf_identity_raw().expand(1, 1, 3, 4)
    y = LA.crf_apply(x, p[0, 0])
    assert torch.allclose(y, x, atol=1e-5), (y - x).abs().max()
    zeros = torch.zeros(3, 4)
    assert not torch.allclose(LA.crf_apply(x, zeros), x, atol=1e-3), \
        "raw zeros must NOT be the identity, or this test proves nothing"


def test_crf_is_monotone_for_random_parameters():
    torch.manual_seed(3)
    x = torch.linspace(0.0, 1.0, 513)[None, :, None].repeat(1, 1, 3)
    for _ in range(20):
        p = torch.randn(3, 4)
        y = LA.crf_apply(x, p)
        assert (y[0, 1:] - y[0, :-1] >= -1e-6).all()


@pytest.mark.parametrize("H,W", [(4, 8), (8, 4)])
def test_vig_uv_is_normalised_by_the_max_dimension_about_the_centre(H, W):
    """ppisp_math.rs:386-397: (px + 0.5 - W/2)/max(W,H). Normalising by W and H
    separately -- the obvious alternative -- makes the falloff elliptical
    instead of radial on a non-square image, which is every image here.

    BOTH ORIENTATIONS ARE REQUIRED and the first version of this test had only
    one. On a landscape image max(W,H) == W, so dividing the x component by W
    instead is BIT-IDENTICAL and the wrong rule passes; only the portrait case
    can see it, and vice versa for y. A rule tested only where it cannot bind
    is not tested.
    """
    m = float(max(H, W))
    uv = LA.vig_uv(H, W)
    assert uv.shape == (H, W, 2)
    assert float(uv[0, 0, 0]) == pytest.approx((0.5 - W * 0.5) / m)
    assert float(uv[0, 0, 1]) == pytest.approx((0.5 - H * 0.5) / m)
    assert float(uv[H - 1, W - 1, 0]) == pytest.approx((W - 0.5 - W * 0.5) / m)
    assert float(uv[H - 1, W - 1, 1]) == pytest.approx((H - 0.5 - H * 0.5) / m)


def test_vig_uv_fixture_can_separate_per_axis_normalisation():
    """The discriminating power of the fixture above, asserted so a future
    single-orientation rewrite cannot quietly re-pin it: on each shape at least
    one axis must disagree with the per-axis rule."""
    for H, W in [(4, 8), (8, 4)]:
        m = float(max(H, W))
        disagree_x = abs((0.5 - W * 0.5) / m - (0.5 - W * 0.5) / W) > 1e-9
        disagree_y = abs((0.5 - H * 0.5) / m - (0.5 - H * 0.5) / H) > 1e-9
        assert disagree_x or disagree_y, (H, W)
    assert abs((0.5 - 8 * 0.5) / 8.0 - (0.5 - 8 * 0.5) / 8.0) < 1e-12


def test_vig_falloff_matches_the_hand_computed_polynomial():
    """1 + a0 r^2 + a1 r^4 + a2 r^6, clamped to [0,1] (ppisp_math.rs:332-347).
    The clamp is load-bearing: without it the fitter can BRIGHTEN, which the
    physical model forbids and which would let it fit content."""
    uv = torch.tensor([[[0.3, 0.4]]])
    p = torch.tensor([[0.0, 0.0, -1.0, 0.5, -0.25]])
    r2 = 0.25
    want = 1.0 + (-1.0) * r2 + 0.5 * r2 ** 2 + (-0.25) * r2 ** 3
    got = LA.vig_falloff(uv, p)
    assert float(got[0, 0, 0]) == pytest.approx(want, abs=1e-6)
    hot = LA.vig_falloff(uv, torch.tensor([[0.0, 0.0, 4.0, 0.0, 0.0]]))
    assert float(hot[0, 0, 0]) == pytest.approx(1.0), "falloff must clamp at 1"


# ------------------------------------------------------------------- the scoring

def test_quantize_reproduces_a_png_round_trip(tmp_path):
    """Every fitted image is scored after a uint8 round trip, because the
    baseline LPIPS was scored off PNGs. A fitter that only wins in float is not
    a gain the pipeline can realise, and rounding-in-numpy that disagrees with
    PIL by one level would move LPIPS in the fourth decimal -- the same order as
    the floor being graded against."""
    from PIL import Image
    torch.manual_seed(4)
    arr = torch.rand(6, 7, 3)
    q = LA.quantize(arr)
    p = tmp_path / "x.png"
    Image.fromarray((arr.clamp(0, 1) * 255).round().to(torch.uint8).numpy()).save(p)
    back = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0)
    assert torch.equal(q, back)


def test_delta_is_exactly_zero_for_an_unchanged_image():
    """C2, mechanically: an identity fitter must return dLPIPS == 0.0 exactly.
    A sign error in the delta turns every loss into a gain and would read as a
    large recoverable component."""
    calls = []

    def fake_metric(a, b):
        calls.append(1)
        return 0.42
    d = LA.delta_lpips(fake_metric, [torch.rand(4, 4, 3)], [None], [torch.rand(4, 4, 3)])
    assert d == [0.0]
    assert len(calls) == 2, "both terms of the delta must be evaluated"


def test_delta_is_positive_when_the_fit_is_closer():
    seq = iter([0.5, 0.2])
    d = LA.delta_lpips(lambda a, b: next(seq), [torch.rand(2, 2, 3)],
                       [torch.rand(2, 2, 3)], [torch.rand(2, 2, 3)])
    assert d == [pytest.approx(0.3)]


# ------------------------------------------------------------------ result files

def test_write_json_refuses_to_overwrite(tmp_path):
    p = tmp_path / "r.json"
    LA.write_json(p, {"kind": LA.KIND, "a": 1})
    with pytest.raises(FileExistsError):
        LA.write_json(p, {"kind": LA.KIND, "a": 2})
    assert json.loads(p.read_text())["a"] == 1


def test_write_json_stamps_the_kind_and_schema(tmp_path):
    p = tmp_path / "r.json"
    LA.write_json(p, {"a": 1})
    d = json.loads(p.read_text())
    assert d["kind"] == LA.KIND and d["schema"] == LA.SCHEMA


def test_read_result_refuses_a_foreign_or_fabricated_file(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"kind": "smoke_fixture", "schema": 1}))
    with pytest.raises(ValueError, match="kind"):
        LA.read_result(p)
    q = tmp_path / "s.json"
    q.write_text(json.dumps({"kind": LA.KIND, "schema": LA.SCHEMA + 99}))
    with pytest.raises(ValueError, match="schema"):
        LA.read_result(q)
