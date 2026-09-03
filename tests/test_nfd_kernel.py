"""Metal `normals_from_depth`: forward, and the GATHER-form adjoint.

Derivation, masking rules and measured mutant kill-power:
research/normals-from-depth-adjoint.md (verified 31/31 against torch.autograd).

The adjoint is a GATHER, not a scatter: each INPUT pixel pulls from up to three output
roles and its own ray multiplies all three, so it is one thread per input pixel with NO
atomics. That makes this lane deterministic, unlike the rasteriser.
"""
import json
import math
from pathlib import Path

import pytest
import torch

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
FIX = Path(__file__).parent / "fixtures" / "normals_from_depth_slanted_plane.json"

# The grazing-plane fixture's intrinsics. The pixel-centre-vs-integer ray convention shows
# only 1.3e-4 relative error at fx=1000 -- it would slip past a loose bar -- but 7.7e-2
# with THESE. Same class as the n_z-exactly-zero geometry that made an orientation test
# vacuous: the constants are what give the test its teeth.
GRAZING = dict(fx=4.0, fy=5.0, cx=-3.5, cy=2.0)
PROD = dict(fx=1000.0, fy=1000.0, cx=64.0, cy=48.0)


def _torch_ref(depth, fx, fy, cx, cy):
    from metal_gauss.geometry_loss import _normals_from_depth_torch
    return _normals_from_depth_torch(depth, fx, fy, cx, cy)


def _metal(depth, fx, fy, cx, cy):
    from metal_gauss.geometry_loss import normals_from_depth
    return normals_from_depth(depth, fx, fy, cx, cy)


def _bars(got, ref, what):
    """max|d|/max|ref| plus cosine. NOT per-element relative: the three roles cancel at
    some pixels, so a CORRECT f32 implementation reads 1e-4 there while scale-relative
    reads 6e-7 (note section 5)."""
    got, ref = got.detach().cpu(), ref.detach().cpu()      # MPS has no float64
    rel = ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
    cos = torch.nn.functional.cosine_similarity(
        got.flatten().double(), ref.flatten().double(), dim=0).item()
    assert rel <= 1e-5 and cos >= 1 - 1e-6, f"{what}: rel {rel:.3e} cos {cos:.9f}"
    return rel, cos


def _depth(h, w, seed=0, holes=0.0, dev="mps"):
    g = torch.Generator().manual_seed(seed)
    d = (torch.rand(h, w, generator=g) * 2.0 + 0.5)
    if holes:
        d[torch.rand(h, w, generator=g) < holes] = 0.0
    return d.to(dev)


@mps
@pytest.mark.parametrize("h,w", [(2, 2), (5, 2), (2, 5), (6, 8), (48, 64)])
def test_metal_forward_matches_the_torch_reference(h, w):
    d = _depth(h, w, seed=h * w)
    got, ref = _metal(d, **PROD), _torch_ref(d, **PROD)
    assert got.shape == (h, w, 3)
    # The note's prescribed metric, not a hand-picked absolute: f32 op-order differences
    # between Metal and torch are ~2e-6 on unit normals, which is 2e-6 SCALE-RELATIVE and
    # entirely benign. A tighter absolute bound fails on correct code.
    _bars(got, ref, f"forward {h}x{w}")


@mps
@pytest.mark.parametrize("h,w,holes", [(6, 8, 0.0), (12, 10, 0.2), (48, 64, 0.05)])
def test_metal_adjoint_matches_autograd_on_the_reference(h, w, holes):
    """The bar from plan section 4: rel <= 1e-5, cosine >= 1 - 1e-6."""
    d = _depth(h, w, seed=h + w, holes=holes)
    g = torch.randn(h, w, 3, device="mps", generator=torch.Generator("mps").manual_seed(7))
    dr = d.clone().requires_grad_(True)
    (_torch_ref(dr, **PROD) * g).sum().backward()
    dm = d.clone().requires_grad_(True)
    (_metal(dm, **PROD) * g).sum().backward()
    _bars(dm.grad, dr.grad, f"{h}x{w} holes={holes}")


@mps
def test_adjoint_is_exactly_zero_where_the_forward_cannot_see(h=8, w=9):
    """Hard zeros, asserted with ==: any input with D <= 0 (every reader is invalid), and
    the bottom-right corner, which no output pixel reads at all."""
    d = _depth(h, w, seed=3)
    d[2, 3] = 0.0
    d[5, 6] = -1.0
    dm = d.clone().requires_grad_(True)
    g = torch.randn(h, w, 3, device="mps")
    (_metal(dm, **PROD) * g).sum().backward()
    assert dm.grad[2, 3].item() == 0.0 and dm.grad[5, 6].item() == 0.0
    assert dm.grad[h - 1, w - 1].item() == 0.0


@mps
def test_euler_identity_sum_D_times_grad_is_zero():
    """The forward is degree-0 homogeneous in D, so sum(D * dL/dD) == 0 for ANY depth
    graph -- exactly what the (I - n n^T) projection buys. One assertion that catches a
    whole family of errors."""
    d = _depth(16, 20, seed=11)
    dm = d.clone().requires_grad_(True)
    g = torch.randn(16, 20, 3, device="mps")
    (_metal(dm, **PROD) * g).sum().backward()
    s = (d * dm.grad).sum().abs().item()
    scale = (d.abs() * dm.grad.abs()).sum().item()
    assert s <= 1e-5 * max(scale, 1e-12), f"Euler identity violated: {s:.3e} vs {scale:.3e}"


@mps
def test_cotangent_along_n_gives_exactly_zero_gradient():
    """g = n is tangent-free, so the (I - n n^T) projection annihilates it. Without the
    projection this reads 6.4 (note section 7)."""
    d = _depth(10, 12, seed=13)
    n = _metal(d, **PROD).detach()
    dm = d.clone().requires_grad_(True)
    (_metal(dm, **PROD) * n).sum().backward()
    assert dm.grad.abs().max().item() < 1e-6


@mps
def test_ray_convention_uses_INTEGER_indices_not_pixel_centres():
    """`normals_from_depth` uses INTEGER pixel indices; `plane_depth_from_features` uses
    pixel CENTRES. Both match Brush and both are deliberate.

    MEASURED with the grazing-plane fixture's intrinsics the two conventions differ by
    7.7e-2; at fx=1000 they differ by 1.3e-4 and this test would be blind. The intrinsics
    ARE the test."""
    d = _depth(8, 10, seed=17)
    got = _metal(d, **GRAZING)
    ref_int = _torch_ref(d, **GRAZING)
    centred = dict(GRAZING, cx=GRAZING["cx"] - 0.5, cy=GRAZING["cy"] - 0.5)
    ref_centre = _torch_ref(d, **centred)
    _bars(got, ref_int, "grazing intrinsics forward")
    sep = (ref_int - ref_centre).abs().max().item()
    assert sep > 1e-2, f"these intrinsics cannot separate the conventions (sep {sep:.2e})"


@mps
def test_single_output_pixel_isolates_all_three_roles():
    """A 2x2 depth map has exactly one output pixel, so its cotangent reaches exactly the
    three inputs it reads and nothing else. Each role is asserted individually with real
    magnitude, so a dropped role NAMES itself instead of shrinking a norm slightly."""
    d = torch.tensor([[1.0, 1.3], [0.8, 2.0]], device="mps").requires_grad_(True)
    g = torch.zeros(2, 2, 3, device="mps"); g[0, 0] = torch.tensor([0.3, -0.7, 0.2])
    (_metal(d, **GRAZING) * g).sum().backward()
    gr = d.grad
    m = gr.abs().max().item()
    assert gr[1, 1].item() == 0.0, "the corner is read by no output pixel"
    for idx, role in (((0, 0), "A base"), ((0, 1), "B +1 in u"), ((1, 0), "C +1 in v")):
        assert abs(gr[idx].item()) > 1e-3 * m, f"role {role} contributed nothing"


@mps
def test_f32_parity_on_a_production_conditioned_plane():
    """A real tilted plane at fx=1000, the regime the trainer actually runs in."""
    h, w = 40, 52
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                          torch.arange(w, dtype=torch.float32), indexing="ij")
    d = (2.0 + 0.004 * (u - PROD["cx"]) - 0.003 * (v - PROD["cy"])).to("mps")
    g = torch.randn(h, w, 3, device="mps")
    dr = d.clone().requires_grad_(True); (_torch_ref(dr, **PROD) * g).sum().backward()
    dm = d.clone().requires_grad_(True); (_metal(dm, **PROD) * g).sum().backward()
    _bars(dm.grad, dr.grad, "production plane")


@mps
@pytest.mark.parametrize("case", ["slanted_plane", "grazing_plane_positive_nz"])
def test_metal_forward_still_matches_the_cross_language_fixture(case):
    from metal_gauss.geometry_loss import normals_from_depth
    fx_ = json.loads(FIX.read_text())
    c = next(k for k in fx_["cases"] if k["name"] == case)
    K = c["intrinsics"]
    d = torch.tensor(c["depth"], dtype=torch.float32, device="mps")
    want = torch.tensor(c["expected_normal"], dtype=torch.float32, device="mps")
    got = normals_from_depth(d, K["fx"], K["fy"], K["cx"], K["cy"])
    valid = want.norm(dim=-1) > 0.5
    _bars(got[valid], want[valid], f"fixture {case}")
    assert (got[~valid] == 0).all()


@mps
def test_the_metal_path_is_actually_taken_for_mps_float32():
    """Without this every comparison above could be torch-against-torch. The dispatch is
    the thing under test as much as the arithmetic."""
    from metal_gauss import geometry_loss as gl
    calls = []
    real = gl._NormalsFromDepthMetal.apply
    gl._NormalsFromDepthMetal.apply = lambda *a: (calls.append(1), real(*a))[1]
    try:
        gl.normals_from_depth(_depth(6, 6), **PROD)
        gl.normals_from_depth(_depth(6, 6, dev="cpu"), **PROD)          # CPU -> torch
        gl.normals_from_depth(_depth(6, 6).double(), **PROD) if False else None
    finally:
        gl._NormalsFromDepthMetal.apply = real
    assert calls == [1], f"expected exactly one Metal dispatch, got {len(calls)}"


@mps
def test_torch_fallback_still_serves_float64_and_cpu():
    """The cross-language fixture is float64 and the CPU path must keep working; the
    kernel is float32-only."""
    from metal_gauss.geometry_loss import normals_from_depth
    d64 = torch.rand(6, 8, dtype=torch.float64) + 0.5
    out = normals_from_depth(d64, **PROD)
    assert out.dtype == torch.float64 and out.shape == (6, 8, 3)


@mps
def test_overflowing_depth_is_invalid_not_a_spurious_normal():
    """`(|n_d| > 0.5) == valid` holds on 775,946,240 real pixels with zero mismatches, but
    the EXACT relation is `gate == valid AND isfinite(L)`. They separate only when |c|^2
    overflows f32, at depths ~1e12 m, where a `valid`-masked kernel emits a normal the
    downstream gate would have rejected and scores a spurious depth-normal loss of 1.0.

    HONESTY: this test PASSED BEFORE the `isfinite(L)` guard was added, because c/inf
    already yields a zero normal that the gate rejects. It is a forward-looking regression
    pin, not the reproduction of a live bug -- it constrains a future fused loss kernel
    from masking on `valid` where `valid` and the gate could diverge. `depth_img` is a
    weighted mean of splat z so it cannot overflow with finite means; the guard closes the
    question by construction rather than by that argument."""
    from metal_gauss.geometry_loss import normals_from_depth
    d = torch.full((6, 8), 1e12, device="mps")
    d[2, 3] = 2e12                                  # break the plane so c is not zero
    out = normals_from_depth(d, **PROD)
    assert torch.isfinite(out).all(), "overflow leaked a non-finite normal"
    bad = out.norm(dim=-1)
    assert ((bad == 0) | (bad > 0.5)).all(), \
        "emitted a normal the |n| > 0.5 gate would reject: gate and valid have separated"
