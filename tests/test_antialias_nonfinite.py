"""`--antialias` produced NON-FINITE GRADIENTS on exactly the splats it exists to help.

FOUND 2026-09-06 by an arm, not by a test. The `--antialias` arm of the needle batch
exported a ply carrying 31,158 non-finite `scale_*` values across 10,386 of 500,000 splats
(2.08%) and 14,824 non-finite opacities. Nothing in the run said so -- `shape_metrics` ran
on MPS, where `torch.sort` orders non-finite values differently from CPU, and reported a
plausible `aspect_p50 0.3085`.

THE MECHANISM, measured rather than inferred. The compensation is
`sqrt(clamp(clamp_min(det_before, 0) / det_after, 0, 1))`. `sqrt` has an UNBOUNDED
derivative at 0, and `clamp_min` contributes a ZERO gradient below its bound, so the chain
evaluates `0 * inf = NaN` for every splat with `det_before <= 0`:

    sx (undilated xx)   det_before        scale     d scale / d conic
                    1           50     0.874439     [-1.705e-01, 0, -1.320e-01]
                 0.01          0.5     0.179069     [-8.327e-01, 0, -2.702e-02]
                 1e-6        5e-05     0.001820     [-8.191e+01, 0, -2.747e-04]
                    0            0     0.000000     [nan, nan, nan]
                -1e-6       -5e-05     0.000000     [nan, nan, nan]
                -1e-2        -0.5      0.000000     [nan, nan, nan]

`det_before <= 0` means the UNDILATED projected covariance is degenerate in one direction
-- i.e. the splat is thinner than the 0.3 px blur. **That is a needle.** So the flag
offered as a needle fix poisons the gradients of the needles specifically, and the
population it poisons is then counted as HEALTHY by `needle_frac`, because
`(aspect < 0.1)` is False for NaN.

This is the same algebraic shape as the Brush `fold_min_scale` defect CLAUDE.md records --
an opacity-compensation coefficient of the form `sqrt(det ratio)` whose backward divides by
a determinant that has reached zero -- fixed there by `eccec445`.

THE FIX keeps the value and bounds the gradient: floor the sqrt's argument, and select the
degenerate branch with `torch.where`, which routes NO gradient to it rather than a NaN one.
"""
from __future__ import annotations

import math

import pytest
import torch

BLUR = 0.3


def _conic_from_undilated(sx, sy, bxy=0.0, blur=BLUR):
    """A conic whose UNDILATED covariance is diag(sx, sy) + [[0,bxy],[bxy,0]].

    The functions under test recover `det_before` from the conic, so parameterising the
    fixture this way is the only way to put a chosen `det_before` in front of them."""
    a, b, c = sx + blur, bxy, sy + blur
    det = a * c - b * b
    return torch.tensor([[c / det, -b / det, a / det]], dtype=torch.float64)


@pytest.mark.parametrize("sx", [0.0, -1e-9, -1e-6, -1e-2, -1.0])
def test_the_gradient_is_FINITE_for_a_degenerate_undilated_covariance(sx):
    """THE TEST. det_before <= 0 is a splat thinner than the blur -- a needle -- and its
    gradient must be a number. Before the fix every one of these is NaN."""
    from metal_gauss.metal_backend import antialias_scale
    con = _conic_from_undilated(sx, 50.0).requires_grad_(True)
    antialias_scale(con, BLUR).sum().backward()
    assert torch.isfinite(con.grad).all(), (
        f"det_before = {sx * 50.0:g} gave d(scale)/d(conic) = {con.grad.tolist()}. "
        f"A non-finite gradient here reaches Adam and writes NaN into log_scales and "
        f"logit_opac on the next step.")


@pytest.mark.parametrize("sx", [1.0, 1e-1, 1e-2, 1e-4, 1e-6])
def test_the_VALUE_is_unchanged_wherever_the_old_formula_was_well_defined(sx):
    """The fix must not move any number that was already a number, or every previous
    --antialias result becomes incomparable. The reference here is the OLD expression,
    written out, evaluated without gradients."""
    from metal_gauss.metal_backend import antialias_scale
    con = _conic_from_undilated(sx, 50.0)
    cxx, cxy, cyy = con[:, 0], con[:, 1], con[:, 2]
    det_after = 1.0 / (cxx * cyy - cxy * cxy).clamp_min(1e-12)
    a, b, c = cyy * det_after, -cxy * det_after, cxx * det_after
    det_before = (a - BLUR) * (c - BLUR) - b * b
    old = (det_before.clamp_min(0.0) / det_after).clamp(0.0, 1.0).sqrt()
    assert antialias_scale(con, BLUR).item() == pytest.approx(old.item(), rel=1e-12)


def test_a_degenerate_splat_still_gets_a_compensation_of_zero_not_a_floor():
    """The fix must not smuggle in an opacity floor. A splat with a degenerate undilated
    covariance carries no energy to preserve, so its compensation is 0 -- as before."""
    from metal_gauss.metal_backend import antialias_scale
    assert antialias_scale(_conic_from_undilated(-1e-2, 50.0), BLUR).item() == 0.0


def test_the_gradient_is_BOUNDED_not_merely_finite():
    """`finite` is not enough: 1/(2*sqrt(r)) at r = 1e-45 is finite in f64 and overflows
    f32, and the trainer is f32. The floor caps it at 1/(2*sqrt(floor))."""
    from metal_gauss.metal_backend import antialias_scale, _AA_RATIO_FLOOR
    worst = 0.0
    for sx in (1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 0.0, -1e-9):
        con = _conic_from_undilated(sx, 50.0).requires_grad_(True)
        antialias_scale(con, BLUR).sum().backward()
        worst = max(worst, float(con.grad.abs().max()))
    bound = 1.0 / (2.0 * math.sqrt(_AA_RATIO_FLOOR))
    assert worst < bound * 1e3, f"worst |grad| {worst:.3e} against a floor bound {bound:.3e}"
    assert worst < 3.0e38, "f32 would overflow on this gradient"


@pytest.mark.parametrize("sx", [0.0, -1e-6, -1e-2])
def test_the_TORCH_REFERENCE_path_has_the_same_defect_and_the_same_fix(sx):
    """`torch_ref.project` carries its own copy of the expression, and it is the oracle
    the Metal kernels are gradchecked against. Fixing one and not the other would leave
    the reference disagreeing with the implementation exactly where it matters."""
    from metal_gauss.torch_ref import project
    # One gaussian, axis-aligned, placed so its projected covariance is degenerate in x.
    means = torch.tensor([[0.0, 0.0, 4.0]], dtype=torch.float64, requires_grad=True)
    # Build cov3d directly rather than from quats/scales so det_before is controllable.
    z = 4.0
    fx = fy = 100.0
    # projected cov ~ (f/z)^2 * cov3d[:2,:2]; choose cov3d so that it lands on sx, 50.
    k = (z / fx) ** 2
    cov3d = torch.zeros(1, 3, 3, dtype=torch.float64)
    # EXACTLY zero, not 1e-14. A real covariance cannot have a negative determinant, so
    # the reachable degenerate case in this path is det_before == 0 (rank-deficient in x),
    # and `sqrt` is unbounded exactly there. A tiny positive value instead of zero makes
    # the whole test non-discriminating: the first draft used 1e-14 and PASSED against the
    # unfixed code.
    cov3d[0, 0, 0] = max(sx, 0.0) * k
    cov3d[0, 1, 1] = 50.0 * k
    cov3d[0, 2, 2] = 1e-6
    cov3d = cov3d.requires_grad_(True)
    vm = torch.eye(4, dtype=torch.float64)
    out = project(means, cov3d, vm, fx, fy, 32.0, 32.0, 64, 64, 0.1, 100.0,
                  blur=BLUR, antialias=True)
    out[5].sum().backward()
    assert torch.isfinite(cov3d.grad).all(), (
        f"torch_ref.project opacity_scale gradient is non-finite: {cov3d.grad.tolist()}")
