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


# ------------------------------------------------- C1: the fitters actually fit
#
# A fitter that silently does not fit returns dLPIPS ~= 0 and reads as "there is
# nothing for an appearance model to fix". These are the mechanical positive
# controls that make a null result mean something.

def _smooth_field(h, w, seed=0):
    """A low-frequency multiplicative gain field -- the thing an appearance
    model is FOR, and the thing LPIPS is argued to be nearly blind to."""
    torch.manual_seed(seed)
    y = torch.linspace(-1, 1, h)[:, None]
    x = torch.linspace(-1, 1, w)[None, :]
    f = 1.0 + 0.25 * torch.cos(1.7 * x) * torch.cos(1.1 * y) - 0.15 * (x ** 2 + y ** 2)
    return f[..., None].repeat(1, 1, 3) * torch.tensor([1.0, 0.95, 1.08])


def test_affine_fit_recovers_a_known_affine_exactly():
    torch.manual_seed(10)
    render = torch.rand(40, 55, 3)
    m = torch.tensor([[1.1, 0.05, -0.02], [0.0, 0.93, 0.04], [0.03, -0.01, 1.07]])
    b = torch.tensor([0.02, -0.01, 0.03])
    gt = render @ m.T + b
    fit, info = fit_call = LA.fit_affine(render, gt)
    assert info["mse_after"] < 1e-12, info["mse_after"]
    assert torch.allclose(fit, gt, atol=1e-5)
    assert info["mse_after"] < 1e-6 * info["mse_before"]


def test_affine_fit_is_the_global_optimum_not_merely_an_improvement():
    """Catches a wrong normal-equation solve that still improves on identity.
    Perturbing the returned parameters in any direction must make it worse."""
    torch.manual_seed(11)
    render = torch.rand(30, 40, 3)
    gt = (render * 0.8 + 0.1 + 0.05 * torch.rand(30, 40, 3)).clamp(0, 1)
    fit, info = LA.fit_affine(render, gt)
    w = torch.tensor(info["params"]).T                      # [4, 3]
    x1 = torch.cat([render.reshape(-1, 3), torch.ones(30 * 40, 1)], 1)
    base = float(((x1 @ w - gt.reshape(-1, 3)) ** 2).mean())
    for k in range(8):
        torch.manual_seed(100 + k)
        w2 = w + 0.01 * torch.randn_like(w)
        assert float(((x1 @ w2 - gt.reshape(-1, 3)) ** 2).mean()) > base


def test_bilagrid_fit_recovers_a_known_low_frequency_gain_field():
    torch.manual_seed(12)
    render = torch.rand(48, 64, 3)
    gt = (render * _smooth_field(48, 64)).clamp(0, 1)
    fit, info = LA.fit_bilagrid(render, gt, steps=400, lr=0.02, tv_weight=0.0)
    assert info["mse_after"] < 0.1 * info["mse_before"], info


def test_bilagrid_fit_cannot_manufacture_a_delta_when_there_is_nothing_to_fit():
    """render == gt. An identity grid has TV exactly zero, so the only gradient
    is float32 noise in the trilinear sum -- and ADAM NORMALISES BY THE GRADIENT
    MAGNITUDE, so a ~1e-7 gradient still buys a full lr-sized step. Measured
    drift after 50 steps is RMS ~7e-5, real but a fiftieth of a uint8 level.

    So the property that matters is not "the raw fit does not move" -- it does --
    but that it cannot move ACROSS THE SCORING, which quantises.

    BIT-EQUALITY OF THE QUANTISED IMAGES IS THE WRONG ASSERTION and the first
    version of this test used it: a value sitting exactly on a rounding boundary
    flips under an arbitrarily small perturbation, so 29 of 2304 levels move
    while the largest raw drift is 0.08 OF ONE LEVEL. A correct implementation
    fails a bit-equality test, which makes it a check the thing being checked
    cannot satisfy. The bound that is both true and load-bearing is: no value
    moves by half a level, so no value can move by more than one level, and only
    boundary values move at all. A fitter that genuinely drifted -- one dragged
    off identity by its own regulariser -- would move by many levels and is
    still caught.
    """
    torch.manual_seed(13)
    render = torch.rand(24, 32, 3)
    fit, info = LA.fit_bilagrid(render, render, steps=50)
    level = 1.0 / 255.0
    drift = (fit - render).abs().max()
    assert float(drift) < 0.5 * level, float(drift) / level
    # Compared in INTEGER levels: (1/255 - 1/255) in float32 is not 0, so a
    # tolerance in image units here is a tolerance on float32 noise.
    q = ((LA.quantize(fit) - LA.quantize(render)) * 255.0).round().abs()
    assert float(q.max()) <= 1.0, float(q.max())
    assert float((q > 0).float().mean()) < 0.05, float((q > 0).float().mean())


def test_ppisp_fit_recovers_a_known_vignetting_and_crf():
    torch.manual_seed(14)
    renders = [torch.rand(40, 52, 3) for _ in range(3)]
    vig = torch.tensor([[0.02, -0.01, -1.4, 0.3, 0.0]] * 3)
    crf = LA.crf_identity_raw().expand(3, 4).clone()
    crf[:, 2] += 0.4
    # Built by the INDEPENDENT reference, never by the function under test --
    # see _reference_ppisp: constructing the target with apply_ppisp made two
    # whole-stage deletions invisible.
    gts = [_reference_ppisp(r, vig, crf) for r in renders]
    fits, info = LA.fit_ppisp_shared(renders, gts, steps=1200, lr=0.05)
    assert info["mse_after"] < 0.1 * info["mse_before"], info


def test_ppisp_parameters_are_SHARED_and_have_no_per_view_axis():
    """The 80%-of-(b) shape rule compares a SHARED 27-parameter model against a
    per-view one. A (c) that had silently become per-view would be a different
    comparison wearing the same name."""
    torch.manual_seed(15)
    renders = [torch.rand(20, 24, 3) for _ in range(2)]
    gts = [r.clamp(0, 1) for r in renders]
    _, info = LA.fit_ppisp_shared(renders, gts, steps=5)
    assert np.array(info["vignetting"]).shape == (3, 5)
    assert np.array(info["crf_raw"]).shape == (3, 4)
    assert info["n_params"] == 27 and info["n_views_shared_over"] == 2


def test_ppisp_shared_cannot_fit_two_views_that_disagree():
    """Discriminating power for the sharing: two views needing OPPOSITE spatial
    falloff cannot both be corrected by one lens model, while a per-view fitter
    would flatten both. If this ever passes trivially, (c) has stopped being
    shared."""
    torch.manual_seed(16)
    base = torch.rand(40, 52, 3) * 0.6 + 0.2
    uv = LA.vig_uv(40, 52)
    left = LA.vig_falloff(uv, torch.tensor([[-0.35, 0.0, -3.0, 0.0, 0.0]] * 3))
    right = LA.vig_falloff(uv, torch.tensor([[0.35, 0.0, -3.0, 0.0, 0.0]] * 3))
    renders = [base.clone(), base.clone()]
    gts = [(base * left).clamp(0, 1), (base * right).clamp(0, 1)]
    _, shared = LA.fit_ppisp_shared(renders, gts, steps=600, lr=0.05)
    per_view = [LA.fit_ppisp_shared([r], [g], steps=600, lr=0.05)[1]
                for r, g in zip(renders, gts)]
    solo = sum(i["mse_after"] for i in per_view) / 2
    assert solo < 0.25 * shared["mse_after"], (solo, shared["mse_after"])


def test_bilagrid_fit_cannot_reproduce_an_unrelated_target():
    """The fit must be a FUNCTION OF THE RENDER. A fitter that read the ground
    truth directly -- one transposed argument -- would return ~gt and report a
    huge recovered dLPIPS, which is the most attractive way for this probe to
    produce a false BUILD.

    THE IMAGE SIZE IS PART OF THE FIXTURE. A 16x16x8 grid is 2048 cells of 12
    parameters; how much unrelated content it can absorb is set by PIXELS PER
    CELL-BIN, measured here on white noise at steps=400, lr=0.02, tv=0:

        48x64     1.5 px/cell-bin   residual 0.033 of baseline  (fits noise)
        240x320  37.5               0.463
        480x640 150.0               0.490                       (plateau)

    The first version of this test used 48x64 and FAILED for a real reason: at
    1.5 pixels per cell-bin the grid genuinely can fit noise. The production
    images sit at 1350 (1920x1440) and 369 (1008x756) px/cell-bin, deep in the
    plateau -- which is also why the nuisance control C3 has to be run at the
    real resolution and cannot be inferred from a small fixture.
    """
    torch.manual_seed(17)
    render = torch.rand(240, 320, 3)
    unrelated = torch.rand(240, 320, 3)
    fit, info = LA.fit_bilagrid(render, unrelated, steps=400, lr=0.02, tv_weight=0.0)
    assert info["mse_after"] > 0.3 * info["mse_before"], info
    assert info["mse_after"] > 1e-3, "a fitter reading gt directly would reach ~0"


def test_the_same_target_from_two_renders_gives_two_different_fits():
    """Decisive form of the same guard: a fitter that read gt would return the
    SAME image for both."""
    torch.manual_seed(19)
    gt = torch.rand(64, 80, 3)
    a, _ = LA.fit_bilagrid(torch.rand(64, 80, 3), gt, steps=60, lr=0.02)
    b, _ = LA.fit_bilagrid(torch.rand(64, 80, 3), gt, steps=60, lr=0.02)
    assert float(((a - b) ** 2).mean()) > 1e-3


def test_ppisp_fit_cannot_reproduce_an_unrelated_target():
    torch.manual_seed(18)
    renders = [torch.rand(40, 52, 3)]
    unrelated = [torch.rand(40, 52, 3)]
    _, info = LA.fit_ppisp_shared(renders, unrelated, steps=400, lr=0.05)
    assert info["mse_after"] > 0.5 * info["mse_before"], info


def _reference_ppisp(rgb, vig, crf):
    """INDEPENDENT transcription of the per-camera PPISP stages, from
    ppisp_kernels.rs:82-125 and ppisp_math.rs:332-347,360-382. Deliberately does
    not call anything in lpips_attr: the first version of
    `test_ppisp_fit_recovers_a_known_vignetting_and_crf` built its ground truth
    by calling `LA.apply_ppisp` itself, so DELETING A WHOLE STAGE left the test
    green -- the target lost the stage at the same moment the model did. A
    fixture a wrong implementation can also produce is not a fixture.
    """
    h, w, _ = rgb.shape
    m = float(max(w, h))
    ux = (torch.arange(w, dtype=rgb.dtype) + 0.5 - w * 0.5) / m
    uy = (torch.arange(h, dtype=rgb.dtype) + 0.5 - h * 0.5) / m
    out = torch.zeros_like(rgb)
    for c in range(3):
        cx, cy, a0, a1, a2 = (float(v) for v in vig[c])
        dx = ux[None, :] - cx
        dy = uy[:, None] - cy
        r2 = dx * dx + dy * dy
        f = (1.0 + a0 * r2 + a1 * r2 ** 2 + a2 * r2 ** 3).clamp(0.0, 1.0)
        x = (rgb[..., c] * f).clamp(0.0, 1.0)
        t_r, s_r, g_r, c_r = (float(v) for v in crf[c])
        toe = 0.3 + math.log1p(math.exp(-abs(t_r))) + max(t_r, 0.0)
        sho = 0.3 + math.log1p(math.exp(-abs(s_r))) + max(s_r, 0.0)
        gam = 0.1 + math.log1p(math.exp(-abs(g_r))) + max(g_r, 0.0)
        ctr = 1.0 / (1.0 + math.exp(-c_r))
        lerp = toe + ctr * (sho - toe)
        aa = sho * ctr / lerp
        bb = 1.0 - aa
        lo = aa * (x / ctr).clamp_min(1e-12) ** toe
        hi = 1.0 - bb * ((1.0 - x) / (1.0 - ctr)).clamp_min(1e-12) ** sho
        out[..., c] = torch.where(x <= ctr, lo, hi).clamp_min(0.0) ** gam
    return out


def test_apply_ppisp_matches_an_independent_transcription_of_brush():
    torch.manual_seed(20)
    rgb = torch.rand(17, 23, 3)
    vig = torch.tensor([[0.02, -0.01, -1.4, 0.3, 0.0],
                        [-0.03, 0.05, -0.9, 0.0, 0.2],
                        [0.0, 0.0, -2.1, 0.6, -0.3]])
    crf = LA.crf_identity_raw().expand(3, 4).clone()
    crf = crf + torch.tensor([[0.2, -0.3, 0.4, 0.1],
                              [-0.1, 0.2, -0.2, -0.3],
                              [0.3, 0.1, 0.15, 0.05]])
    got = LA.apply_ppisp(rgb, vig, crf)
    ref = _reference_ppisp(rgb, vig, crf)
    assert torch.allclose(got, ref, atol=2e-6), (got - ref).abs().max()


def test_the_ppisp_fixture_separates_the_two_stages():
    """Discriminating power against the two mutants that survived first time:
    the fixture's parameters must make BOTH stages matter, or dropping either is
    invisible."""
    torch.manual_seed(21)
    rgb = torch.rand(17, 23, 3)
    vig = torch.tensor([[0.02, -0.01, -1.4, 0.3, 0.0]] * 3)
    crf = LA.crf_identity_raw().expand(3, 4).clone() + 0.4
    ident_v = torch.zeros(3, 5)
    ident_c = LA.crf_identity_raw().expand(3, 4).clone()
    full = _reference_ppisp(rgb, vig, crf)
    no_vig = _reference_ppisp(rgb, ident_v, crf)
    no_crf = _reference_ppisp(rgb, vig, ident_c)
    assert float(((full - no_vig) ** 2).mean()) > 1e-3
    assert float(((full - no_crf) ** 2).mean()) > 1e-3


# ------------------------------------------------------- the device they run on

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")


@mps
@pytest.mark.parametrize("fitter", ["affine", "bilagrid", "ppisp"])
def test_every_fitter_runs_on_MPS(fitter):
    """The probe runs on MPS and the CPU-only tests could not see that
    `fit_affine` did `.double()` BEFORE `.cpu()`, which raises on MPS -- MPS has
    no float64. It died on the first real view instead, which is exactly how
    research/metal-gauss.md 13.5 records finding the same defect.
    """
    torch.manual_seed(30)
    r = torch.rand(24, 32, 3, device="mps")
    g = (r * 0.9 + 0.05).clamp(0, 1)
    if fitter == "affine":
        fit, info = LA.fit_affine(r, g)
    elif fitter == "bilagrid":
        fit, info = LA.fit_bilagrid(r, g, steps=20)
    else:
        fits, info = LA.fit_ppisp_shared([r], [g], steps=20)
        fit = fits[0]
    assert fit.device.type == "mps", fit.device
    assert fit.shape == r.shape
    assert torch.isfinite(fit).all()
    assert info["mse_after"] <= info["mse_before"] * 1.0001
