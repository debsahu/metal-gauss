"""PGSR plane-aux depth: the fused aux path carrying `[d, n]` where `[z, n]` was.

WHAT THIS FILE IS ABOUT. `plane_features` and `plane_depth_from_features` are already
tested as pure functions in test_geometry_loss.py. Nothing there goes through the
rasterizer, so nothing there can see the three things that actually go wrong when the
depth CHANNEL changes meaning:

  1. the composited offset lands in the wrong aux channel, or on the wrong contract;
  2. the loss kernel divides a plane depth by alpha (mode confusion) -- the loud mutant,
     `rel = 1/alpha - 1`, which is tens of percent at alpha 0.7 and therefore testable;
  3. `gt_depth` is not masked by the returned `valid`, so an unsupervisable pixel is
     scored as an UNCOVERED one instead of leaving the numerator AND the denominator.

Each of those has a test below whose name says what it would catch, and each was confirmed
to fail by substitution before the implementation existed.
"""
import pytest
import torch

from metal_gauss import render

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
pytestmark = mps


def _K(W, H, f=None):
    K = torch.eye(3)
    K[0, 0] = K[1, 1] = f if f is not None else 0.8 * max(W, H)
    K[0, 2], K[1, 2] = W / 2, H / 2
    return K


def _leaves(n=300, seed=11, dev="mps"):
    torch.manual_seed(seed)
    L = dict(m=torch.randn(n, 3) * 0.6 + torch.tensor([0., 0., 4.]), q=torch.randn(n, 4),
             s=torch.rand(n, 3) * 0.10 + 0.03, o=torch.rand(n) * 0.7 + 0.15,
             sh=torch.randn(n, 16, 3) * 0.25)
    return {k: v.to(dev).requires_grad_(True) for k, v in L.items()}


def _wall(n_side=13, z0=3.0, wide=0.30, thin=0.004, span=1.0,
          mvec=(0.4, 0.25, 1.0), dev="mps"):
    """A TILTED planar wall: splat centres ON the plane, thin axis ALONG the plane normal.

    THE FIRST VERSION OF THIS FIXTURE WAS GEOMETRICALLY WRONG and is worth recording,
    because it fails in the direction that looks like a working test. It tilted each splat
    RANDOMLY about a fronto-parallel wall, on the reasoning that "tilt is what PGSR is
    about". But a tilted splat's tangent plane is not the wall: its ray intersection is
    `n_i . p_i / (n_i . r)`, which is correct for THAT splat's plane and simply not 3.0.
    The plane depth read 8.4e-2 off a wall at 3.0 m -- and it was right to.

    The bias PGSR removes is the tilt of the SURFACE relative to the IMAGE PLANE, not of a
    splat relative to its surface. So the surface is tilted here (~25 degrees) and every
    splat agrees with it. Measured on this fixture (probe, 2026-09-03): plane depth is
    exact to 2.4e-6 while alpha-composited centre depth is off by 0.11-0.28 m mean, five
    orders of magnitude apart, at every splat size tried between 0.16 and 0.55.

    Returns `(leaves, n_wall_unit, c)` where the wall is `{p : n_wall . p == c}`.
    """
    m = torch.tensor(mvec, dtype=torch.float32); m = m / m.norm()
    e3 = torch.tensor([0., 0., 1.])
    ax = torch.cross(e3, m, dim=0)
    ang = torch.atan2(ax.norm(), torch.dot(e3, m))
    ax = ax / ax.norm().clamp_min(1e-12)
    q1 = torch.cat([torch.cos(ang / 2)[None], torch.sin(ang / 2) * ax])
    t1 = torch.tensor([1., 0., 0.]) - m * m[0]; t1 = t1 / t1.norm()
    t2 = torch.cross(m, t1, dim=0)
    k = torch.linspace(-span, span, n_side)
    vv, uu = torch.meshgrid(k, k, indexing="ij")
    ctr = torch.tensor([0., 0., float(z0)])
    p = ctr + uu.reshape(-1, 1) * t1 + vv.reshape(-1, 1) * t2
    N = p.shape[0]
    sh = torch.zeros(N, 16, 3); sh[:, 0] = 0.6
    L = dict(m=p, q=q1[None].expand(N, 4).contiguous(),
             s=torch.tensor([wide, wide, thin])[None].expand(N, 3).contiguous(),
             o=torch.full((N,), 0.99), sh=sh)
    return ({kk: v.to(dev).requires_grad_(True) for kk, v in L.items()},
            m.to(dev), float(torch.dot(m, ctr)))


def _wall_truth(n_wall, c, W, H, K, dev="mps"):
    """Analytic ray-plane depth of `_wall`, at PIXEL CENTRES (the plane path's convention)."""
    fx, fy = K[0, 0].item(), K[1, 1].item()
    cx, cy = K[0, 2].item(), K[1, 2].item()
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                          torch.arange(W, dtype=torch.float32), indexing="ij")
    ray = torch.stack([(u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy, torch.ones_like(u)], -1)
    return c / (ray.to(dev) * n_wall).sum(-1)


def _render_aux(L, W, H, vm, K, depth_source):
    """Render through the TRAINER's own aux builder, not a local reimplementation.

    This is deliberate. An earlier draft of this file packed `[n_cam, d]` itself, and every
    forward assertion passed before a single line of plane-aux wiring existed -- it was
    testing `plane_depth_from_features`, which test_geometry_loss.py already covers, and
    was blind to the thing this task changes. Going through `train.geometry_aux` means a
    mutant there (returning z under `plane-aux`, detaching the offset, swapping the two
    list slots) fails these tests.
    """
    from metal_gauss.train import geometry_aux
    aux = geometry_aux(L["m"], L["q"], L["s"], vm.to("mps"), depth_source=depth_source)
    rgb, alpha, info = render(
        L["m"], L["q"], L["s"], L["o"], L["sh"], K, vm, W, H,
        sh_degree=3, backend="metal",
        aux_colors=aux, aux_detach_weights=[True, True])
    return rgb, alpha, info


def _render_plane(L, W, H, vm, K):
    return _render_aux(L, W, H, vm, K, "plane-aux")


def _feat_img(info, alpha):
    """(H,W,5) = n_sum(3) + offset_sum(1) + alpha(1), from the rendered aux maps."""
    n_sum = info["aux"][0]
    off = info["aux"][1][..., :1]
    return torch.cat([n_sum, off, alpha.detach()[..., None]], dim=-1)


# ------------------------------------------------------------------ (a) forward, e2e

def test_plane_depth_through_the_FUSED_AUX_PATH_recovers_the_wall_depth():
    """Would catch: the composited offset landing in the wrong aux channel, the plane
    layout silently taking the multipass path, or a normalisation applied to the offset
    channel. All three leave `depth` finite and plausible and wrong by a scale factor or a
    channel permutation; matching the analytic ray-plane depth to 1e-4 separates them.
    """
    from metal_gauss.geometry_loss import plane_depth_from_features
    from metal_gauss.metal_backend import _use_fused_aux
    W = H = 64
    vm = torch.eye(4)
    K = _K(W, H)
    L, n_wall, c = _wall()
    _, alpha, info = _render_plane(L, W, H, vm, K)
    assert _use_fused_aux([True, True], 2), "the plane layout must take the FUSED path"
    assert info["aux_path"] == "fused"
    depth, _, valid = plane_depth_from_features(
        _feat_img(info, alpha), K[0, 0].item(), K[1, 1].item(),
        K[0, 2].item(), K[1, 2].item())
    truth = _wall_truth(n_wall, c, W, H, K)
    sel = (alpha.detach() > 0.95) & (valid > 0.5)
    assert sel.sum() > 1000, f"fixture must cover the frame, got {sel.sum()}"
    err = (depth[sel] - truth[sel]).abs().max().item()
    assert err < 1e-4, f"plane depth off the analytic wall by {err:.4e}"


def test_plane_depth_beats_centre_depth_on_a_TILTED_wall():
    """THE MECHANISM CHECK. Would catch `plane_features` degenerating to the centre-depth
    channel -- `d` computed as z, or the offset never composited.

    On a surface tilted ~25 degrees to the image plane the alpha-composited CENTRE depth is
    biased (each splat contributes one constant depth over a footprint whose true depth
    ramps, and front-to-back compositing weights the near contributors) while the plane
    depth is exact. A FRONTO-PARALLEL fixture cannot make this assertion at all -- the two
    sources agree there -- which is the trap the first version of this file fell into.

    The `assert e_centre > 1e-3` line is the fixture's own discriminating-power check: if
    centre depth were already exact, the comparison below would be vacuous.
    """
    from metal_gauss.geometry_loss import plane_depth_from_features
    W = H = 64
    vm = torch.eye(4)
    K = _K(W, H)
    L, n_wall, c = _wall()
    _, alpha, info = _render_plane(L, W, H, vm, K)
    plane, _, valid = plane_depth_from_features(
        _feat_img(info, alpha), K[0, 0].item(), K[1, 1].item(),
        K[0, 2].item(), K[1, 2].item())

    # centre depth over the SAME splats, through the SAME builder under `center`
    _, alpha2, info2 = _render_aux(L, W, H, vm, K, "center")
    centre = info2["aux"][1][..., 0] / alpha2.detach().clamp_min(1e-10)

    truth = _wall_truth(n_wall, c, W, H, K)
    sel = (alpha.detach() > 0.95) & (valid > 0.5)
    assert sel.sum() > 1000, f"need a solid interior, got {sel.sum()}"
    e_plane = (plane[sel] - truth[sel]).abs().mean().item()
    e_centre = (centre[sel] - truth[sel]).abs().mean().item()
    assert e_centre > 1e-3, ("the fixture cannot separate the two depth sources: centre "
                             f"depth is already exact ({e_centre:.3e})")
    assert e_plane < 0.01 * e_centre, \
        f"plane depth ({e_plane:.4e}) must beat centre depth ({e_centre:.4e})"


def test_invalid_plane_pixels_are_EXACTLY_zero_in_both_maps():
    """Would catch: an invalid pixel carrying a small non-zero depth. `depth_loss` reads
    0 as 'no prediction'; anything else is a measurement the geometry never made."""
    from metal_gauss.geometry_loss import plane_depth_from_features
    W = H = 48
    vm = torch.eye(4)
    K = _K(W, H)
    L = _leaves(n=40, seed=5)                    # sparse: most of the frame is empty
    _, alpha, info = _render_plane(L, W, H, vm, K)
    depth, normal, valid = plane_depth_from_features(
        _feat_img(info, alpha), K[0, 0].item(), K[1, 1].item(),
        K[0, 2].item(), K[1, 2].item())
    inv = valid < 0.5
    assert inv.sum() > 0, "fixture produced no invalid pixels"
    assert (depth[inv] == 0.0).all()
    assert (normal[inv] == 0.0).all()


# ---------------------------------------------------- (b) the gradient-reach contract

def test_plane_aux_gradient_reaches_quats_and_means_but_NOT_scales_or_opacity():
    """The plane layout's version of `depth_aux_does_not_touch_opacity`.

    `d = n . p` is live in BOTH `n` (through quats) and `means`, so both must move --
    unlike the centre-depth channel, where only means does. `scales` must stay dead (the
    thin-axis choice is a detached argmin) and `logit_opac` must stay dead (the blending
    weights are detached by the fused kernel's contract).

    TEETH: the reference gradients are asserted ABOVE 1e-3 before the zeros are checked. A
    version of this test that only asserted the zeros would pass on a render that produced
    no gradient at all.
    """
    from metal_gauss.geometry_loss import depth_loss, plane_depth_from_features
    W = H = 56
    vm = torch.eye(4)
    K = _K(W, H)
    L, _n_wall, _c = _wall(n_side=9)
    _, alpha, info = _render_plane(L, W, H, vm, K)
    depth, _, valid = plane_depth_from_features(
        _feat_img(info, alpha), K[0, 0].item(), K[1, 1].item(),
        K[0, 2].item(), K[1, 2].item())
    gt = torch.full_like(depth, 3.4) * valid
    depth_loss(depth, gt, "disparity").backward()

    for name in ("q", "m"):
        g = L[name].grad
        amax = 0.0 if g is None else g.abs().max().item()
        assert amax > 1e-3, f"plane depth must reach {name} with real magnitude, got {amax}"
    for name in ("s", "o"):
        g = L[name].grad
        amax = 0.0 if g is None else g.abs().max().item()
        assert amax < 1e-8, f"plane depth must not push {name}, got {amax}"


# ----------------------------------------------------------- (c) the depth_mode uniform

def _exposing_plane(h=40, w=52, dev="mps", seed=3):
    """alpha spread, holes, disagreeing contributors -- and a FINISHED depth map that is
    NOT z/alpha, so mode confusion cannot hide."""
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(h, w, generator=g) * 0.5 + 0.45          # 0.45..0.95, never ~1
    n1 = torch.nn.functional.normalize(torch.randn(h, w, 3, generator=g), dim=-1)
    n2 = torch.nn.functional.normalize(torch.randn(h, w, 3, generator=g), dim=-1)
    t = torch.rand(h, w, 1, generator=g)
    n_sum = a[..., None] * (t * n1 + (1 - t) * n2)
    depth = torch.rand(h, w, generator=g) * 2.0 + 0.6       # a real depth, not z/a
    gt_d = torch.rand(h, w, generator=g) * 3 + 0.5
    gt_d[torch.rand(h, w, generator=g) < 0.12] = 0.0
    gt_n = torch.nn.functional.normalize(torch.randn(h, w, 3, generator=g), dim=-1)
    gt_n[torch.rand(h, w, generator=g) < 0.12] = 0.0
    return [x.to(dev) for x in (n_sum, depth, a, gt_d, gt_n)]


def _torch_chain_given_depth(depth, n_sum, alpha, gt_d, gt_n, K, space="disparity"):
    """The torch reference for depth_mode=1: the depth map is GIVEN, not z/alpha."""
    from metal_gauss.geometry_loss import (depth_loss, depth_normal_loss, normal_loss,
                                           normals_from_depth)
    a = alpha.detach().clamp_min(1e-10)
    ni = n_sum / a[..., None]
    ni = ni / ni.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    nd = normals_from_depth(depth, *K)
    return torch.stack([depth_loss(depth, gt_d, space), normal_loss(ni, gt_n),
                        depth_normal_loss(nd, ni, alpha.detach())])


GRAZING = (4.0, 5.0, -3.5, 2.0)


def test_depth_mode_GIVEN_matches_the_float64_torch_reference():
    """Would catch: the kernel still dividing by alpha in mode 1, or striding the given
    depth map as if it were (H,W,3)."""
    from metal_gauss.geometry_loss import fused_geometry_losses
    n_sum, depth, a, gt_d, gt_n = _exposing_plane()
    got = fused_geometry_losses(depth, n_sum, a, gt_d, gt_n, GRAZING,
                                depth_mode="given").cpu().double()
    ref = _torch_chain_given_depth(*[x.cpu().double() for x in (depth, n_sum, a, gt_d, gt_n)],
                                   GRAZING)
    rel = ((got - ref).abs().max() / ref.abs().max()).item()
    assert rel <= 1e-5, f"rel {rel:.3e}\n got {got.tolist()}\n ref {ref.tolist()}"


def test_the_LOUD_MUTANT_mode_confusion_is_actually_loud():
    """THE MUTANT THIS FILE EXISTS FOR, asserted rather than assumed.

    Running mode 0 (z/alpha) on a plane depth divides by alpha twice, so the depth term's
    input is off by exactly 1/alpha and the relative error is at least 1/alpha_max - 1.
    If that were small, the mode test above would have no teeth and could pass on a kernel
    that ignored the uniform. It is not small: the fixture's alpha tops out below 1, so
    the floor below is a real bound and the measured error is far above it.
    """
    from metal_gauss.geometry_loss import fused_geometry_losses
    n_sum, depth, a, gt_d, gt_n = _exposing_plane()
    good = fused_geometry_losses(depth, n_sum, a, gt_d, gt_n, GRAZING,
                                 depth_mode="given").cpu().double()
    # mode 0 wants an (H,W,3) whose channel 0 is the value it will divide: feed the same
    # depth, so it computes depth/alpha where it should have computed depth.
    z3 = depth[..., None].expand(-1, -1, 3).contiguous()
    bad = fused_geometry_losses(z3, n_sum, a, gt_d, gt_n, GRAZING,
                                depth_mode="alpha").cpu().double()
    floor = (1.0 / a.max().item()) - 1.0
    assert floor > 0.02, f"fixture alpha too close to 1 to bound the mutant: {floor}"
    rel = ((bad[0] - good[0]).abs() / good[0].abs()).item()
    assert rel >= floor, f"mode confusion only moved the depth term by {rel:.3e} (>= {floor:.3e})"


def test_depth_mode_GIVEN_gradients_match_the_torch_chain():
    """Would catch: the Python backward still applying the 1/alpha factor to the depth lane
    in mode 1. That factor lives in `_FusedGeometryLosses.backward`, NOT in the kernel
    (research/metal-gauss.md section 11.4's deferred optimisation left the gather's final
    multiply outside MSL), so it is the half of the mode switch a kernel-only change would
    miss -- silently, because the forward would still look right. The defect it guards is
    a factor of 1/alpha in [1.05, 2.2] over the whole depth lane, i.e. rel ~ 0.5, not 1e-5.

    ## The 1e-5 bar is MISSED at the GRAZING intrinsics, and the miss is recorded, not tuned

    Measured 2026-09-03, both f32 paths against the same f64 CPU reference, 8 fixture
    seeds, GRAZING (fx=4) `given` mode, `rel = max|delta| / max|ref|` on d(depth):

        seed      0        1        2        3        4        5        6        7
        metal   7.6e-6  11.9e-6   8.2e-6  10.4e-6   5.6e-6   5.4e-6   5.9e-6   9.7e-6
        torch   24.0e-6 16.3e-6  12.9e-6   7.5e-6   4.6e-6   3.6e-6  16.3e-6   9.4e-6

    THE TORCH f32 CHAIN MISSES 1e-5 ON 4 OF 8 SEEDS; THE KERNEL MISSES ON 2. The bar is
    below what f32 arithmetic delivers for this expression at fx = 4, where
    `normals_from_depth`'s 1/L amplification is extreme -- exactly the situation section
    11.1 met, and diagnosed the same way (build a better reference, then ask which side is
    wrong). Here the answer is neither: on seed 3, `max|delta|` is 1.7e-8 in absolute
    terms and only TWO pixels of 2080 sit above half of it, so the max-statistic is being
    set by one f32 cancellation. It is not an L1 sign tie -- the smallest residual on the
    fixture is 2.0e-4, five thousand ulps from zero, checked.

    So: 1e-5 is asserted at PROD, where it is met with margin (3.5e-6). At GRAZING the
    assertion is against the f32 floor MEASURED ON THE SAME FIXTURE IN THE SAME TEST -- the
    kernel must be within 2x of what the torch f32 chain itself achieves. That is a
    stronger statement than a fixed constant would be, because a systematically wrong
    adjoint fails it at any intrinsics, and it cannot be met by an implementation that is
    merely close to another wrong one: the reference is f64.
    """
    from metal_gauss.geometry_loss import fused_geometry_losses
    for K, bar in ((PROD, 1e-5), (GRAZING, None)):
        n_sum, depth, a, gt_d, gt_n = _exposing_plane()
        w = torch.tensor([1.0, 0.2, 0.05], device="mps")

        d1 = depth.clone().requires_grad_(True)
        n1 = n_sum.clone().requires_grad_(True)
        (fused_geometry_losses(d1, n1, a, gt_d, gt_n, K, depth_mode="given")
         * w).sum().backward()

        def reference(dtype):
            d = depth.cpu().to(dtype).clone().requires_grad_(True)
            n = n_sum.cpu().to(dtype).clone().requires_grad_(True)
            (_torch_chain_given_depth(d, n, a.cpu().to(dtype), gt_d.cpu().to(dtype),
                                      gt_n.cpu().to(dtype), K)
             * w.cpu().to(dtype)).sum().backward()
            return d.grad.double(), n.grad.double()

        f32, f64 = reference(torch.float32), reference(torch.float64)
        rel = lambda x, y: ((x - y).abs().max() / y.abs().max()).item()

        for i, (got, what) in enumerate(((d1.grad, "d(depth)"), (n1.grad, "d(n_sum)"))):
            got = got.cpu().double()
            r = rel(got, f64[i])
            cos = torch.nn.functional.cosine_similarity(
                got.flatten(), f64[i].flatten(), dim=0).item()
            assert cos >= 1 - 1e-6, f"{what} @ {K}: cos {cos:.9f}"
            if bar is not None:
                assert r <= bar, f"{what} @ PROD: rel {r:.3e} > {bar:.0e}"
            else:
                floor = rel(f32[i], f64[i])
                assert floor > 0, f"{what}: f32 floor came out 0, the probe is broken"
                assert r <= max(bar or 0.0, 2.0 * floor), (
                    f"{what} @ GRAZING: rel {r:.3e} against an f32 floor of {floor:.3e}")


PROD = (1000.0, 1000.0, 64.0, 48.0)


def test_depth_mode_ALPHA_is_bit_identical_to_the_pre_task_call():
    """Regression: the mode branch is a dispatch uniform, so `center` must take exactly
    the path it always did. Compares against the no-argument call, which is the signature
    every pre-Task-19 caller used."""
    from metal_gauss.geometry_loss import fused_geometry_losses
    from tests.test_fused_geom_loss import exposing
    n_sum, z, a, gt_d, gt_n = exposing()
    old = fused_geometry_losses(z, n_sum, a, gt_d, gt_n, GRAZING)
    new = fused_geometry_losses(z, n_sum, a, gt_d, gt_n, GRAZING, depth_mode="alpha")
    assert torch.equal(old, new)


def test_depth_mode_ALPHA_gradients_are_bit_identical_to_the_pre_task_call():
    from metal_gauss.geometry_loss import fused_geometry_losses
    from tests.test_fused_geom_loss import exposing
    n_sum, z, a, gt_d, gt_n = exposing()
    w = torch.tensor([1.0, 0.2, 0.05], device="mps")
    outs = []
    for kw in ({}, {"depth_mode": "alpha"}):
        zc = z.clone().requires_grad_(True); nc = n_sum.clone().requires_grad_(True)
        (fused_geometry_losses(zc, nc, a, gt_d, gt_n, GRAZING, **kw) * w).sum().backward()
        outs.append((zc.grad.clone(), nc.grad.clone()))
    assert torch.equal(outs[0][0], outs[1][0])
    assert torch.equal(outs[0][1], outs[1][1])


# --------------------------------------------------- (d) gt_depth * valid, not pred * valid

def test_invalid_plane_pixels_leave_BOTH_numerator_and_denominator():
    """Brush train.rs:1945 masks the GT (`gt_depth * valid`), not only the prediction.

    Would catch: masking only the prediction. That leaves gt > 0 at an invalid pixel, so
    the pixel stays in the count with a full-magnitude residual -- it is scored as
    UNCOVERED (a reconstruction failure) instead of UNSUPERVISED (no data). The two
    differ by a large constant, and this test separates them by asserting the loss equals
    the mean over the valid pixels ONLY.
    """
    from metal_gauss.geometry_loss import depth_loss
    g = torch.Generator().manual_seed(11)
    H, W = 12, 16
    pred = (torch.rand(H, W, generator=g) * 2 + 0.5).to("mps")
    gt = (torch.rand(H, W, generator=g) * 2 + 0.5).to("mps")
    valid = (torch.rand(H, W, generator=g) > 0.35).float().to("mps")
    pred = pred * valid                       # invalid plane pixels are exactly 0

    got = depth_loss(pred, gt * valid, "disparity")
    sel = valid > 0.5
    want = (1.0 / pred[sel] - 1.0 / gt[sel]).abs().mean()
    assert torch.allclose(got, want, atol=1e-6), f"{got.item()} vs {want.item()}"

    # masking only the prediction: invalid pixels stay in the denominator with a full
    # 1/gt residual. Must be a DIFFERENT, larger number -- if it were not, this test
    # could not tell the two rules apart.
    wrong = depth_loss(pred, gt, "disparity")
    assert wrong.item() > got.item() * 1.05, \
        f"fixture cannot separate the masking rules: {wrong.item()} vs {got.item()}"


# ------------------------------------------------------- (e) the two ray conventions

def test_the_two_ray_conventions_are_kept_DIFFERENT_on_purpose():
    """`plane_depth_from_features` uses pixel CENTRES; `normals_from_depth` uses INTEGER
    indices. Both match Brush. Would catch: someone 'harmonising' them -- which is how a
    cross-language contract silently diverges from its fixture.

    Asserted structurally, by feeding a ramp through each and requiring the results to
    disagree by the half-pixel shift at intrinsics where a half pixel matters.
    """
    from metal_gauss.geometry_loss import normals_from_depth, plane_depth_from_features
    H = W = 9
    fx = fy = 4.0
    cx = cy = 4.5
    n = torch.tensor([0.35, -0.25, -1.0], dtype=torch.float64)
    n = n / n.norm()
    feat = torch.zeros(H, W, 5, dtype=torch.float64)
    feat[..., :3] = n
    feat[..., 3] = -2.5
    feat[..., 4] = 1.0
    depth, _, _ = plane_depth_from_features(feat, fx, fy, cx, cy)
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                          torch.arange(W, dtype=torch.float64), indexing="ij")
    centre = torch.stack([(u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy, torch.ones_like(u)], -1)
    integer = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], -1)
    assert torch.allclose(depth, -2.5 / (centre * n).sum(-1), atol=1e-12)
    assert not torch.allclose(depth, -2.5 / (integer * n).sum(-1), atol=1e-3)

    # normals_from_depth on a plane must return that plane's normal under INTEGER rays.
    d_int = (-2.5 / (integer * n).sum(-1)).float()
    nd = normals_from_depth(d_int, fx, fy, cx, cy)
    inner = nd[1:-2, 1:-2].reshape(-1, 3)
    assert torch.allclose(inner, n.float().expand_as(inner), atol=1e-4), \
        "normals_from_depth must use integer indices"
    d_ctr = (-2.5 / (centre * n).sum(-1)).float()
    nd2 = normals_from_depth(d_ctr, fx, fy, cx, cy)
    inner2 = nd2[1:-2, 1:-2].reshape(-1, 3)
    assert not torch.allclose(inner2, n.float().expand_as(inner2), atol=1e-4), \
        "this configuration cannot tell the two conventions apart"


# ------------------------------------------------------- (f) non-pinhole falls back

def test_plane_aux_falls_back_to_center_on_a_non_pinhole_camera():
    """Brush warns and skips (`warn_plane_depth_needs_pinhole`, train.rs:1416-1418).

    NOTE ON SCOPE, stated because the plan's version of this test cannot fail on this
    trainer as written: metal-gauss has no camera-model concept at all -- dataset.py reads
    `cam.params[0:4]` as (fx, fy, cx, cy) with no check, and the rasterizer is pinhole
    only. There is therefore no state in which a warn-and-fallback could fire. This test
    is against the guard that MAKES it fireable: the COLMAP model name is recorded on the
    Scene, and a model whose params are not (fx, fy, cx, cy) falls back to `center` with a
    warning rather than silently unprojecting with the wrong ray.
    """
    from metal_gauss.train import resolve_depth_source
    assert resolve_depth_source("plane-aux", "PINHOLE") == "plane-aux"
    assert resolve_depth_source("plane-aux", None) == "plane-aux"      # unknown: allow
    assert resolve_depth_source("center", "OPENCV_FISHEYE") == "center"
    with pytest.warns(UserWarning, match="pinhole"):
        assert resolve_depth_source("plane-aux", "OPENCV_FISHEYE") == "center"
    with pytest.warns(UserWarning, match="pinhole"):
        assert resolve_depth_source("plane-aux", "SIMPLE_RADIAL") == "center"


# ------------------------------------------------- (g) the contract reaches the kernel

def test_render_view_requests_the_DETACHED_contract_under_plane_aux():
    """Would catch: plane-aux being wired with live blending weights.

    That is Brush's `plane-fused`, the mode CLAUDE.md bans (opacity p50 collapses ~30% on
    both test scenes) and the shape of this repo's own needle collapse -- a live weight
    path on a depth channel, which read HEALTHIER than baseline on thin-axis, opacity and
    dark fraction while destroying the splats. Asserted at the call, not inferred.
    """
    from unittest.mock import patch
    from metal_gauss import train as T
    W = H = 40
    L, _n, _c = _wall(n_side=6)
    p = {"means": L["m"], "quats": L["q"], "log_scales": torch.log(L["s"]),
         "logit_opac": torch.logit(L["o"].clamp(1e-4, 1 - 1e-4)),
         "sh_dc": L["sh"][:, :1], "sh_rest": L["sh"][:, 1:]}

    class V:
        image = torch.zeros(H, W, 3, dtype=torch.uint8)
        K = _K(W, H)
        viewmat = torch.eye(4)

    seen = {}
    real = T.render

    def spy(*a, **kw):
        seen.update(aux_detach_weights=kw.get("aux_detach_weights"),
                    n_aux=len(kw.get("aux_colors") or []))
        return real(*a, **kw)

    with patch.object(T, "render", spy):
        T.render_view(p, V(), L["m"].shape[0], sh_deg=3, want_geometry=True,
                      depth_source="plane-aux")
    assert seen["n_aux"] == 2, seen
    assert seen["aux_detach_weights"] == [True, True], seen


# ------------------------------------------------------------- (h) end-to-end wiring

def test_train_runs_the_recipe_under_plane_aux_and_logs_a_real_depth_term():
    """Would catch: `--depth-source plane-aux` accepted and then ignored, or a term that
    silently comes out zero / non-finite once the plane depth is the depth map.

    Compares against the same run under `center` and requires the depth term to DIFFER --
    a plane path that quietly fell back to centre depth would log an identical number.
    """
    import math
    from metal_gauss import train as T
    from tests.test_train_recipe import _args, _synthetic_scene

    def run(src):
        return T.train(_args(flatten_loss_weight=1.0, depth_loss_weight=1.0,
                             normal_loss_weight=0.2, depth_normal_weight=0.05,
                             depth_source=src), scene=_synthetic_scene())

    out_p = run("plane-aux")
    out_c = run("center")
    tp, tc = out_p["log"][-1]["terms"], out_c["log"][-1]["terms"]
    for k in ("depth", "normal", "depth_normal", "flatten"):
        assert k in tp and math.isfinite(tp[k]) and tp[k] > 0, f"{k} = {tp.get(k)}"
    assert tp["depth"] != tc["depth"], \
        "plane-aux logged the same depth term as centre depth: it fell back silently"
    assert out_p["resolved"]["depth_source"] == "plane-aux"


def test_geometry_terms_masks_the_GT_by_valid_and_not_only_the_prediction():
    """THE TEST THAT WAS MISSING, and the mutation battery is how that was found.

    The first version of this file checked the gt*valid rule against `depth_loss` on
    synthetic tensors -- one level below the code that has to apply it. Mutant M5
    (`gt_depth = gt_depth`, dropping the mask in `geometry_terms`) SURVIVED the whole
    suite: the end-to-end recipe test could not see it, because on a fronto-parallel
    synthetic wall every plane pixel is valid and the mask is all ones. A rule tested only
    where it cannot bind is not tested.

    So this fixture manufactures invalid pixels -- a band of near-zero `n_sum` whose
    ray-plane denominator falls under `min_denom` -- and asserts the masking is
    IDEMPOTENT: handing in a `gt_depth` that is already `gt * valid` must give the same
    depth term as handing in the raw one. It can only do that if the code applied the mask
    itself. The last two assertions are the fixture's discriminating-power check: the two
    gt variants must differ, and the two loss values must differ, or the idempotence above
    would hold for a code path that never masked anything.
    """
    import argparse
    from metal_gauss.geometry_loss import plane_depth_from_features
    from metal_gauss.train import geometry_terms
    g = torch.Generator().manual_seed(5)
    H, W = 24, 32
    K = torch.tensor([[40.0, 0, W / 2], [0, 40.0, H / 2], [0, 0, 1.0]])

    a = (torch.rand(H, W, generator=g) * 0.3 + 0.7)
    n = torch.nn.functional.normalize(torch.randn(H, W, 3, generator=g), dim=-1)
    n[..., 2] = -n[..., 2].abs()                       # face the camera
    n_sum = a[..., None] * torch.nn.functional.normalize(n, dim=-1)
    n_sum[:, :10] *= 1e-6                              # denominator under min_denom
    off = (a * -2.0)[..., None].expand(H, W, 3).contiguous()
    aux = [n_sum.to("mps"), off.to("mps")]
    alpha = a.to("mps")
    gt_d = (torch.rand(H, W, generator=g) * 2 + 1.0).to("mps")
    gt_n = torch.nn.functional.normalize(
        torch.randn(H, W, 3, generator=g), dim=-1).to("mps")

    feat = torch.cat([aux[0], aux[1][..., :1], alpha[..., None]], -1)
    _, _, valid = plane_depth_from_features(feat, 40.0, 40.0, W / 2, H / 2)
    assert 0.1 < valid.mean().item() < 0.9, \
        f"fixture must have BOTH valid and invalid pixels, got {valid.mean().item():.3f}"

    args = argparse.Namespace(depth_loss_weight=1.0, normal_loss_weight=0.2,
                              depth_normal_weight=0.05, depth_loss_space="disparity",
                              depth_source="plane-aux")
    raw = geometry_terms(args, aux, alpha, K, gt_d, gt_n, None)["depth"]
    pre = geometry_terms(args, aux, alpha, K, gt_d * valid, gt_n, None)["depth"]
    assert torch.allclose(raw, pre, rtol=0, atol=0), \
        (f"geometry_terms did not mask gt_depth by valid: raw {raw.item():.6f} vs "
         f"pre-masked {pre.item():.6f}")

    # discriminating power: the two gt variants, and the two losses, must actually differ
    assert not torch.equal(gt_d, gt_d * valid)
    zero_gt = geometry_terms(args, aux, alpha, K, gt_d * 0 + gt_d, gt_n, None)
    unmasked_ref = geometry_terms(
        argparse.Namespace(**{**vars(args), "depth_source": "center"}),
        aux, alpha, K, gt_d, gt_n, None)["depth"]
    assert not torch.allclose(raw, unmasked_ref), \
        "the fixture cannot separate the plane path from the centre path"
    assert torch.isfinite(zero_gt["depth"])


def test_the_FUSED_LOSS_KERNEL_IS_NOT_USED_when_depth_normal_weight_is_zero():
    """A FINDING, pinned so it cannot drift unnoticed, not a defect being fixed here.

    `geometry_terms`'s fused branch requires ALL THREE weights positive
    (`args.depth_normal_weight > 0`). So R1p -- flatten 1.0, depth 1.0, normal 0.2,
    dn 0.0 -- runs the TORCH loss chain, not the Tier 2 fused kernel.

    That matters for three things at once:

      * R1p is the base arm the plan pre-registers for every Tier 3 comparison, so every
        Tier 3 arm measured with dn = 0 is measured on the torch chain. The A/B is still
        one-variable (both arms take the same branch), but it is NOT the code path Tier 2
        optimised.
      * research/metal-gauss.md section 11.6a's "recipe ON" figures (1.59x / 1.79x) were
        taken with dn = 0.05, i.e. WITH the kernel. They do not describe an R1p run.
      * Task 19's route (ii) -- "fold the ray-plane division into the loss kernel" -- cannot
        help R1p at all, because R1p never calls that kernel. Route (ii) as the plan
        specifies it is a speedup for dn > 0 configurations only.

    Verified by execution rather than by reading: `fused_geometry_losses` is patched to
    raise, and `geometry_terms` is called at dn = 0 and dn = 0.05 with everything else
    identical. Only the second must reach it. A test that merely read the condition could
    not tell whether some other guard also short-circuits.
    """
    import argparse
    from unittest.mock import patch
    from metal_gauss import train as T
    from tests.test_fused_geom_loss import exposing

    n_sum, z, a, gt_d, gt_n = exposing(seed=3)
    K = torch.tensor([[600.0, 0, 26.0], [0, 600.0, 20.0], [0, 0, 1.0]])
    reached = []

    def spy(*args, **kw):
        reached.append(kw.get("depth_mode", "alpha"))
        raise AssertionError("fused kernel reached")

    for dn, expect in ((0.0, False), (0.05, True)):
        args = argparse.Namespace(depth_loss_weight=1.0, normal_loss_weight=0.2,
                                  depth_normal_weight=dn, depth_loss_space="disparity",
                                  depth_source="center")
        reached.clear()
        with patch.object(T, "fused_geometry_losses", spy):
            try:
                T.geometry_terms(args, [n_sum, z], a, K, gt_d, gt_n, None)
            except AssertionError as e:
                if "fused kernel reached" not in str(e):
                    raise
        assert bool(reached) is expect, (
            f"dn={dn}: fused kernel reached={bool(reached)}, expected {expect}")
