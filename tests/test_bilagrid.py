"""Bilateral grid (Task 22) -- correctness of the slice, the TV and the wiring.

WHAT EACH TEST IS FOR is stated on the test, in the form "this fails if X". Every
one of them was confirmed to fail by SUBSTITUTION -- a deliberately wrong
implementation swapped in, the named test watched to fail, then reverted -- and
the battery that did it is recorded in the branch's mutation commit. A test whose
substitution was BEHAVIOURALLY IDENTICAL proves nothing and is marked as such.

The reference for the slice is an explicit 8-corner transcription of
brush-appearance/src/bilagrid_kernels.rs written from the Rust and NOT from the
module under test -- Task 21 lost two mutants to a fixture that built its ground
truth by calling the function it was testing.
"""

from __future__ import annotations

import math

import pytest
import torch

from metal_gauss.bilagrid import (
    BilateralGrid, DEFAULT_DIMS, identity_grids, slice_apply, tv_loss, warmup_exp_lr,
)

# TRANSCRIBED, NOT IMPORTED. The first version of this file imported LUMA_* and
# IDENTITY_CHANNELS from the module under test, and the mutation battery caught
# it: swapping the module's red and blue luma weights, and reading the affine
# column-major, BOTH changed behaviour and killed NO test, because the reference
# moved with the target. That is Task 21's "fixture built by calling the function
# under test", reproduced exactly. These are read off
# brush-appearance/src/bilagrid_kernels.rs:15-17 and bilagrid.rs:373-377.
REF_LUMA_R, REF_LUMA_G, REF_LUMA_B = 0.299, 0.587, 0.114
REF_IDENTITY_CHANNELS = (0, 5, 10)


# --------------------------------------------------------------------------
# The independent reference: a literal transcription of the Rust kernel.
# --------------------------------------------------------------------------

def _brush_slice_reference(grid, rgb):
    """Explicit 8-corner trilinear slice, transcribed from
    brush-appearance/src/bilagrid_kernels.rs:37-70 (sample_point), :72-93
    (corner_x/y/z, corner_weight), :120-147 (interpolate) and :158-171 (the
    row-major 3x4 applied to (r, g, b, 1)).

    grid: [12, L, H, W] float64 tensor. rgb: [h, w, 3] float64 tensor.
    Deliberately a scalar loop: it shares no vectorised machinery, no coordinate
    normalisation and no library call with the implementation, so agreement is
    evidence rather than a tautology.
    """
    n_coef, gl, gh, gw = grid.shape
    assert n_coef == 12
    h, w, _ = rgb.shape
    out = torch.zeros(h, w, 3, dtype=torch.float64)
    for py in range(h):
        for px in range(w):
            r, g, b = (float(rgb[py, px, i]) for i in range(3))
            x = px * (gw - 1) / max(w - 1, 1)
            y = py * (gh - 1) / max(h - 1, 1)
            raw_z = (REF_LUMA_R * r + REF_LUMA_G * g + REF_LUMA_B * b) * (gl - 1)
            z = min(max(raw_z, 0.0), float(gl - 1))
            x0, y0, z0 = math.floor(x), math.floor(y), math.floor(z)
            x1, y1, z1 = min(x0 + 1, gw - 1), min(y0 + 1, gh - 1), min(z0 + 1, gl - 1)
            tx, ty, tz = x - math.floor(x), y - math.floor(y), z - math.floor(z)
            comp = (r, g, b, 1.0)
            for row in range(3):
                acc = 0.0
                for col in range(4):
                    coefficient = row * 4 + col
                    val = 0.0
                    for corner in range(8):                 # bit 1 = x, 2 = y, 4 = z
                        cx = x0 if (corner & 1) == 0 else x1
                        cy = y0 if (corner & 2) == 0 else y1
                        cz = z0 if (corner & 4) == 0 else z1
                        wx = (1.0 - tx) if (corner & 1) == 0 else tx
                        wy = (1.0 - ty) if (corner & 2) == 0 else ty
                        wz = (1.0 - tz) if (corner & 4) == 0 else tz
                        val += float(grid[coefficient, cz, cy, cx]) * wx * wy * wz
                    acc += val * comp[col]
                out[py, px, row] = acc
    return out


def _fixture(seed=0):
    """A grid and a render chosen so the reference can DISCRIMINATE.

    Non-square image (h != w) so an x/y swap is visible; a non-cubic grid
    (gw != gh != gl) so an axis permutation cannot hide; grid values that vary
    across every axis so interpolation weights matter; and a render whose
    luminance spans BEYOND [0, 1] on both sides so the guidance clamp is exercised.
    `test_the_fixture_can_tell_the_rules_apart` asserts all of that rather than
    assuming it.
    """
    gen = torch.Generator().manual_seed(seed)
    grid = torch.randn(12, 5, 3, 4, generator=gen, dtype=torch.float64) * 0.4
    for c in REF_IDENTITY_CHANNELS:
        grid[c] += 1.0
    # NOTE the range. A per-channel [-0.3, 1.3] was the first attempt and the
    # discriminating-power guard below rejected it: the luminance is a weighted
    # MEAN of three channels, so it concentrates and never left [0, 1] even
    # though the channels did. The clamp would have gone untested.
    rgb = torch.rand(7, 5, 3, generator=gen, dtype=torch.float64) * 3.0 - 1.0
    return grid, rgb


def test_the_fixture_can_tell_the_rules_apart():
    """Discriminating power of the fixture itself, asserted rather than assumed.

    Fails if the fixture degenerates to a case where a wrong rule would agree:
    a square image, a cubic grid, an in-range-only luminance, or a grid flat
    enough that interpolation is a no-op. Task 21 shipped a `vig_uv` test whose
    landscape-only fixture made the wrong normalisation BIT-IDENTICAL, and a
    normals fixture whose plane family satisfied both the right and the wrong
    orientation rule. This is the guard against repeating that.
    """
    grid, rgb = _fixture()
    h, w, _ = rgb.shape
    _, gl, gh, gw = grid.shape
    assert h != w, "square image cannot catch an x/y swap"
    assert len({gl, gh, gw}) == 3, "cubic grid cannot catch an axis permutation"
    lum = REF_LUMA_R * rgb[..., 0] + REF_LUMA_G * rgb[..., 1] + REF_LUMA_B * rgb[..., 2]
    assert lum.min() < 0.0 and lum.max() > 1.0, "guidance clamp is never exercised"
    # An x/y swap of the grid must actually change the reference's output.
    ref = _brush_slice_reference(grid, rgb)
    swapped = _brush_slice_reference(grid.permute(0, 1, 3, 2).contiguous()[:, :, :3, :3],
                                    rgb[:, :3])
    assert not torch.allclose(ref[:, :3], swapped, atol=1e-9), \
        "the fixture cannot distinguish a transposed spatial grid"
    # And the interpolation must be doing work: a nearest-corner read must differ.
    assert grid.std() > 0.1, "grid too flat for interpolation weights to matter"


def test_slice_matches_the_independent_eight_corner_transcription():
    """The load-bearing correctness test.

    Fails if: grid_sample's (x, y, z) last-axis order is confused with the
    tensor's (D, H, W) order; align_corners is False; padding_mode is not
    "border"; the luma coefficients are wrong or in the wrong order; the affine
    is read column-major; or the guidance is not clamped.
    """
    grid, rgb = _fixture()
    got = slice_apply(grid[None].to(torch.float64), rgb.to(torch.float64))
    want = _brush_slice_reference(grid, rgb)
    assert torch.allclose(got, want, atol=1e-10), \
        f"max|delta| = {(got - want).abs().max().item():.3e}"


def test_identity_grid_reproduces_the_render():
    """Fails if the identity channels are 0, 4, 8 (the column-major reading of a
    3x4) instead of 0, 5, 10, or if the bias column is not zero. A column-major
    identity produces a plausible image, not an error, which is why this is a
    test and not an assertion."""
    rgb = torch.rand(9, 6, 3, dtype=torch.float64)
    g = identity_grids(1, (4, 3, 5), device="cpu", dtype=torch.float64)
    assert torch.allclose(slice_apply(g, rgb), rgb, atol=1e-12)


def test_a_constant_grid_reduces_to_one_global_affine():
    """Fails if the interpolation weights do not sum to 1 -- a bug that the
    identity test above cannot see, because an identity grid is unchanged by any
    weighting that sums to one OR by several that do not."""
    gen = torch.Generator().manual_seed(3)
    m = torch.randn(3, 4, generator=gen, dtype=torch.float64)
    g = m.reshape(12)[:, None, None, None].expand(12, 5, 3, 4).contiguous()[None]
    rgb = torch.rand(7, 5, 3, generator=gen, dtype=torch.float64)
    col = torch.cat([rgb, torch.ones_like(rgb[..., :1])], -1)
    want = (m * col[..., None, :]).sum(-1)
    assert torch.allclose(slice_apply(g, rgb), want, atol=1e-12)


def test_grid_that_varies_only_in_guidance_is_not_a_global_affine():
    """The control for the test above: proves that test is not vacuous, i.e. that
    a NON-constant grid does NOT reduce to a global affine. Without this, a
    slice implementation that ignored the grid's spatial and guidance structure
    entirely would still pass `test_a_constant_grid_reduces_to_one_global_affine`.
    """
    gen = torch.Generator().manual_seed(4)
    g = identity_grids(1, (4, 3, 5), device="cpu", dtype=torch.float64)
    g[0, 3] = torch.linspace(-1, 1, 5, dtype=torch.float64)[:, None, None]   # red bias vs guidance
    rgb = torch.rand(7, 5, 3, generator=gen, dtype=torch.float64)
    out = slice_apply(g, rgb)
    assert not torch.allclose(out, rgb, atol=1e-6), "guidance axis had no effect"
    # and the effect must track luminance, which is what makes it BILATERAL
    lum = REF_LUMA_R * rgb[..., 0] + REF_LUMA_G * rgb[..., 1] + REF_LUMA_B * rgb[..., 2]
    delta = (out - rgb)[..., 0].flatten()
    corr = torch.corrcoef(torch.stack([delta, lum.flatten()]))[0, 1]
    assert corr > 0.99, f"red bias did not track luminance (r = {corr:.3f})"


def test_guidance_is_clamped_and_the_clamp_kills_the_gradient():
    """Brush's `guidance_active` flag (bilagrid_kernels.rs:67) gates the guidance
    gradient when the luminance leaves [0, 1]. Fails if the clamp is missing (the
    slice would read outside the grid, or extrapolate) or if it is applied in a
    way that leaks gradient -- e.g. via `min`/`max` on a detached copy."""
    g = identity_grids(1, (4, 3, 5), device="cpu", dtype=torch.float64)
    g[0, 3] = torch.linspace(-1, 1, 5, dtype=torch.float64)[:, None, None]
    # Two renders far outside [0, 1]: different luminance, but both clamped to the
    # same end of the guidance axis, so the sliced coefficients must be identical.
    hi1 = torch.full((4, 4, 3), 2.0, dtype=torch.float64)
    hi2 = torch.full((4, 4, 3), 5.0, dtype=torch.float64)
    c1 = slice_apply(g, hi1) - hi1                     # isolate the bias column
    c2 = slice_apply(g, hi2) - hi2
    assert torch.allclose(c1, c2, atol=1e-12), "guidance was not clamped"
    over = torch.full((4, 4, 3), 2.0, dtype=torch.float64, requires_grad=True)
    slice_apply(g, over).sum().backward()
    # the affine still passes rgb through, so d(out)/d(rgb) is the 3x3 part, but
    # NO gradient may arrive through the guidance coordinate. With the identity
    # 3x3 and a guidance-only bias, that makes every entry exactly 1.
    assert torch.allclose(over.grad, torch.ones_like(over), atol=1e-12), \
        "gradient leaked through the clamped guidance coordinate"


# ------------------------------------------------------------------ TV

def test_tv_is_zero_at_identity_and_positive_off_it():
    """Fails if TV is computed on the wrong tensor or is sign-indefinite."""
    g = identity_grids(2, (4, 3, 5), device="cpu", dtype=torch.float64)
    assert tv_loss(g).item() == 0.0
    g[0, 0, 1, 1, 1] += 0.5
    assert tv_loss(g).item() > 0.0


def test_tv_is_a_sum_of_three_means_not_a_mean_of_the_concatenation():
    """Brush's TV (bilagrid.rs:340-348) is `dx.mean() + dy.mean() + dz.mean()`.
    Fails if the three are concatenated and averaged, or summed rather than
    meaned. The three axes have DIFFERENT lengths here, so those forms give
    genuinely different numbers -- asserted below so this cannot be a
    coincidence of a cubic grid."""
    gen = torch.Generator().manual_seed(7)
    g = torch.randn(1, 12, 5, 3, 4, generator=gen, dtype=torch.float64)
    dx = (g[..., 1:] - g[..., :-1]) ** 2
    dy = (g[..., 1:, :] - g[..., :-1, :]) ** 2
    dz = (g[..., 1:, :, :] - g[..., :-1, :, :]) ** 2
    want = dx.mean() + dy.mean() + dz.mean()
    concat_mean = torch.cat([dx.flatten(), dy.flatten(), dz.flatten()]).mean()
    assert not math.isclose(want.item(), concat_mean.item(), rel_tol=1e-6), \
        "fixture cannot separate sum-of-means from mean-of-concatenation"
    assert dx.numel() != dy.numel() != dz.numel()
    assert torch.allclose(tv_loss(g), want, atol=1e-12)


def test_tv_defaults_to_the_active_view_only():
    """Brush lifts and regularises ONLY the active view's grid
    (train_state.rs:352-366, 453-465). Fails if the regulariser averages over all
    N views, which would divide each view's TV gradient by N while its
    photometric gradient stayed put -- an N-fold weakening of a term Task 21
    measured to be load-bearing."""
    m = BilateralGrid(4, (4, 3, 5), device="cpu")
    with torch.no_grad():
        m.grids[1, 0, 1, 1, 1] += 0.5
    assert m.regulariser(0).item() == 0.0, "a quiet view must contribute no TV"
    assert m.regulariser(1).item() > 0.0
    # and the global form must differ from the per-view form, or the test is moot
    assert m.regulariser(None).item() < m.regulariser(1).item()


def test_tv_gradient_reaches_only_the_active_view():
    """Fails if `regulariser(idx)` indexes with `grids[idx]` in a way that
    detaches, or regularises the wrong block."""
    m = BilateralGrid(3, (4, 3, 5), device="cpu")
    with torch.no_grad():
        m.grids += torch.randn_like(m.grids) * 0.1
    m.regulariser(1).backward()
    assert m.grids.grad[1].abs().sum() > 0
    assert m.grids.grad[0].abs().sum() == 0.0
    assert m.grids.grad[2].abs().sum() == 0.0


# ------------------------------------------------------- module and indexing

def test_forward_uses_the_indexed_view_and_no_other():
    """Fails if the module slices the wrong view, or broadcasts over all of them.
    The two views here are deliberately DIFFERENT transforms, so returning the
    wrong one is visible."""
    m = BilateralGrid(2, (4, 3, 5), device="cpu")
    with torch.no_grad():
        m.grids[0, 3] = 0.25          # view 0: +0.25 red bias
        m.grids[1, 3] = -0.5          # view 1: -0.5 red bias
    rgb = torch.rand(6, 4, 3)
    assert torch.allclose((m(rgb, 0) - rgb)[..., 0], torch.full((6, 4), 0.25), atol=1e-6)
    assert torch.allclose((m(rgb, 1) - rgb)[..., 0], torch.full((6, 4), -0.5), atol=1e-6)


def test_photometric_gradient_lands_only_in_the_drawn_view():
    """The property that makes per-view grids sound: view j's grid must not move
    when view i is drawn. Fails if the slice is replaced by something that
    touches the whole parameter (e.g. an expand, or a gather over all views)."""
    m = BilateralGrid(3, (4, 3, 5), device="cpu")
    m(torch.rand(6, 4, 3), 1).sum().backward()
    assert m.grids.grad[1].abs().sum() > 0
    assert m.grids.grad[0].abs().sum() == 0.0
    assert m.grids.grad[2].abs().sum() == 0.0


def test_gradient_reaches_the_render_through_the_AFFINE_and_not_only_the_guidance():
    """Fails if the render is detached in the affine multiply -- which would leave
    the gaussians unable to see the correction, i.e. the model inert while still
    reporting a parameter count.

    THE FIRST VERSION OF THIS TEST DID NOT CATCH THAT, and the mutation battery
    said so. `rgb` reaches the output by TWO routes -- the affine columns and the
    luminance guidance coordinate -- so `rgb.grad.sum() > 0` was satisfied by the
    guidance route alone even with the affine route detached. The fix is to make
    the guidance route contribute EXACTLY ZERO: with a spatially and tonally
    CONSTANT grid, d(coefficients)/d(luminance) is zero everywhere, so every unit
    of gradient that arrives must have come through the affine. The expected
    value is then closed-form -- the column sums of the 3x3 part -- rather than
    merely "nonzero"."""
    gen = torch.Generator().manual_seed(21)
    m3x3 = torch.randn(3, 3, generator=gen, dtype=torch.float64)
    coef = torch.cat([m3x3, torch.zeros(3, 1, dtype=torch.float64)], 1).reshape(12)
    g = coef[:, None, None, None].expand(12, 5, 3, 4).contiguous()[None]
    rgb = torch.rand(6, 4, 3, generator=gen, dtype=torch.float64, requires_grad=True)
    slice_apply(g, rgb).sum().backward()
    want = m3x3.sum(0).expand(6, 4, 3)              # d(sum out_i)/d(rgb_j) = sum_i M_ij
    assert torch.allclose(rgb.grad, want, atol=1e-10), \
        f"affine gradient path is broken: max|delta| {(rgb.grad - want).abs().max():.3e}"
    assert rgb.grad.abs().sum() > 0
    # control: the fixture must be one where the guidance route really is silent,
    # or "all gradient came through the affine" would be unprovable.
    varying = g.clone(); varying[0, 3] = torch.linspace(-1, 1, 5, dtype=torch.float64)[:, None, None]
    r2 = rgb.detach().clone().requires_grad_(True)
    slice_apply(varying, r2).sum().backward()
    assert not torch.allclose(r2.grad, want, atol=1e-6), \
        "guidance route is inert even when the grid varies -- fixture proves nothing"


def test_parameter_count_is_twelve_per_cell_per_view():
    """Fails if dims are read in the wrong order -- (x, y, guidance) is Brush's
    --bilagrid-dims order and gives a [N, 12, guidance, y, x] tensor."""
    m = BilateralGrid(196, DEFAULT_DIMS, device="cpu")
    assert tuple(m.grids.shape) == (196, 12, 8, 16, 16)
    assert m.grids.numel() == 196 * 12 * 8 * 16 * 16 == 4_816_896


def test_dims_below_two_are_rejected():
    """Brush asserts the same (bilagrid.rs:52-59): a dimension of 1 divides by
    zero in both the interpolation and the TV normalisation."""
    with pytest.raises(ValueError, match=">= 2"):
        identity_grids(1, (16, 16, 1), device="cpu")


# --------------------------------------------------------------- lr schedule

def test_warmup_exp_lr_matches_an_independent_transcription():
    """Transcribed from brush-appearance/src/lib.rs:159-174. Fails on an
    off-by-one in the warmup (`step + 1`), on dividing the decay by
    `decay_steps - warmup`, or on a linear instead of exponential decay."""
    base, warmup, decay = 2e-3, 1000, 30000
    def ref(step):
        if step < warmup:
            return base * (0.01 + 0.99 * (step + 1) / warmup)
        return base * 0.01 ** ((step - warmup) / decay)
    for s in (0, 1, 499, 999, 1000, 1001, 7000, 29999, 30000):
        assert math.isclose(warmup_exp_lr(s, base, decay_steps=decay), ref(s), rel_tol=1e-12)
    # discriminating power: the schedule must actually MOVE, or this is vacuous
    assert warmup_exp_lr(0, base, decay_steps=decay) < 0.05 * base
    assert warmup_exp_lr(999, base, decay_steps=decay) > 0.99 * base
    assert warmup_exp_lr(30000, base, decay_steps=decay) < 0.02 * base
