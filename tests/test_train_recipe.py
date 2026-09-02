"""End-to-end: 40 steps of the geometry recipe on a synthetic planar scene.

Not a quality test -- it proves the terms are WIRED (non-zero, finite, logged) and that the
run completes. A term that is silently always zero is the failure this catches; the unit
tests in test_geometry_loss.py cannot see it because they never go through `train()`.
"""
import math

import numpy as np
import pytest
import torch

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def _synthetic_scene(n_views=6, W=64, H=48, with_depth=True, with_normal=True,
                     masked=True, mask_value=255):
    """A fronto-parallel textured wall at z=3: exact depth (3.0 everywhere), exact normal
    (0,0,-1). Priors are stored in the same quantized residency the loader produces."""
    from metal_gauss.dataset import Scene, View
    from metal_gauss.prior_io import encode_depth_u16mm, encode_normal_u8
    rng = np.random.default_rng(0)
    views = []
    f = 0.8 * W
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    for i in range(n_views):
        vm = torch.eye(4)
        vm[0, 3] = 0.2 * (i - n_views / 2)
        img = (rng.random((H, W, 3)) * 255).astype(np.uint8)
        mask = None
        if masked:
            m = np.full((H, W), mask_value, np.uint8)
            m[:, :4] = 0                                   # 4 of 64 columns dropped
            mask = torch.from_numpy(m)
        depth = normal = None
        if with_depth:
            depth = torch.from_numpy(encode_depth_u16mm(np.full((H, W), 3.0, np.float32)))
        if with_normal:
            nrm = np.zeros((H, W, 3), np.float32)
            nrm[..., 2] = -1.0
            normal = torch.from_numpy(encode_normal_u8(nrm))
        views.append(View(f"v{i}", torch.from_numpy(img), K, vm,
                          mask=mask, depth=depth, normal=normal))
    pts = rng.random((400, 3)).astype(np.float32)
    pts[:, :2] = (pts[:, :2] - 0.5) * 4
    pts[:, 2] = 3.0
    return Scene(views[:-1], views[-1:], pts, rng.random((400, 3)).astype(np.float32))


def _args(**over):
    """Built through the REAL parser, so the arms inherit every default from the one place
    the CLI does. Hand-writing this namespace is how a sweep once ran with settings other
    than the ones it reported (see `_run_report`'s docstring)."""
    from metal_gauss.train import build_parser
    argv = ["--colmap", "x", "--images", "y", "--steps", "40", "--budget", "2000",
            "--max-resolution", "64", "--eval-every", "40", "--relocate-every", "20",
            "--eval-split-every", "8", "--sh-warmup", "0", "--no-grow",
            "--num-downscales", "0", "--seed", "0", "--densify-weight", "opacity"]
    for k, v in over.items():
        flag = "--" + k.replace("_", "-")
        argv += [flag] if v is True else [flag, str(v)]
    a = build_parser().parse_args(argv)
    a.resolution_schedule = 100
    return a


@mps
def test_recipe_runs_and_logs_every_term():
    from metal_gauss import train as T
    out = T.train(_args(flatten_loss_weight=1.0, depth_loss_weight=1.0,
                        normal_loss_weight=0.2, depth_normal_weight=0.05),
                  scene=_synthetic_scene())
    terms = out["log"][-1]["terms"]
    for k in ("l1", "ssim", "flatten", "depth", "normal", "depth_normal"):
        assert k in terms, f"{k} not logged"
        assert math.isfinite(terms[k]) and terms[k] > 0, f"{k} = {terms.get(k)}"
    assert 0.9 < out["metrics"]["coverage"] < 0.95        # 4 of 64 columns dropped = 93.75%
    assert math.isfinite(out["metrics"]["psnr_masked"])
    assert out["metrics"]["terms"] == terms               # the report carries them too


@mps
def test_geometry_terms_are_absent_when_their_weights_are_zero():
    """A term computed and then multiplied by 0 still costs an aux pass and its backward.
    Zero weights must take the cheap path, and the log must not claim a term that is off."""
    from metal_gauss import train as T
    out = T.train(_args(flatten_loss_weight=1.0), scene=_synthetic_scene())
    terms = out["log"][-1]["terms"]
    assert "flatten" in terms
    for k in ("depth", "normal", "depth_normal"):
        assert k not in terms, k


@mps
@pytest.mark.parametrize("flag,attr", [("depth_loss_weight", "depth"),
                                       ("normal_loss_weight", "normal")])
def test_weight_without_prior_is_a_startup_error(flag, attr):
    """Refuse to run rather than silently train without the supervision it was configured
    for. A run that quietly drops a term looks exactly like one that kept it."""
    from metal_gauss import train as T
    sc = _synthetic_scene(**{f"with_{attr}": False})
    with pytest.raises(RuntimeError, match=rf"{flag.replace('_', '-')}.*no view"):
        T.train(_args(**{flag: 1.0}), scene=sc)


@mps
def test_depth_normal_weight_needs_no_prior_at_all():
    """It compares the render's own depth-derived normals against its own rendered normals,
    so it is the cheapest of the three to switch on and must NOT demand a prior."""
    from metal_gauss import train as T
    out = T.train(_args(depth_normal_weight=0.05),
                  scene=_synthetic_scene(with_depth=False, with_normal=False))
    assert out["log"][-1]["terms"]["depth_normal"] > 0


@mps
def test_depth_term_falls_toward_zero_on_a_scene_whose_prior_is_exact():
    """The wall is at z=3 and the prior says 3.0 everywhere, so the depth term is
    optimisable to ~0. If it were wired to the wrong tensor -- the preprocess's gradient-less
    `depth` output, say -- it would sit at a constant instead of descending."""
    from metal_gauss import train as T
    out = T.train(_args(depth_loss_weight=1.0, steps=200, eval_every=20),
                  scene=_synthetic_scene())
    d = [e["terms"]["depth"] for e in out["log"] if "terms" in e]
    assert len(d) >= 4
    assert d[-1] < d[0], f"depth term did not descend: {d[0]:.4g} -> {d[-1]:.4g}"


@mps
def test_masked_pixels_do_not_supervise_geometry():
    """The mask must gate GT depth/normal BINARILY. Multiplying a metric depth by a
    fractional alpha would invent depths between 0 and the true value -- a smaller depth is
    a different surface, not a less-certain one."""
    from metal_gauss import train as T
    a = T.train(_args(depth_loss_weight=1.0), scene=_synthetic_scene(masked=True))
    b = T.train(_args(depth_loss_weight=1.0), scene=_synthetic_scene(masked=False))
    assert a["log"][-1]["terms"]["depth"] != b["log"][-1]["terms"]["depth"]
    assert math.isfinite(a["log"][-1]["terms"]["depth"])


# ------------------------------------------- the four properties Checkpoint C reads for

def _tiny_render(scales=(0.001, 0.1, 0.1), mean=(0.5, 0.0, 3.0), n=1):
    """One gaussian, rendered exactly as train() renders it, with the geometry aux on."""
    from metal_gauss.train import render_view
    from metal_gauss.dataset import View
    W = H = 32
    f = 0.8 * W
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    img = torch.zeros(H, W, 3, dtype=torch.uint8)
    v = View("t", img, K, torch.eye(4))
    p = {"means": torch.tensor([list(mean)] * n, device="mps").requires_grad_(True),
         "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n, device="mps").requires_grad_(True),
         "log_scales": torch.log(torch.tensor([list(scales)] * n, device="mps")).requires_grad_(True),
         "logit_opac": torch.full((n,), 4.0, device="mps").requires_grad_(True),
         "sh_dc": torch.zeros(n, 1, 3, device="mps").requires_grad_(True),
         "sh_rest": torch.zeros(n, 15, 3, device="mps").requires_grad_(True)}
    rgb, alpha, info = render_view(p, v, n, 3, (0.0, 0.0, 0.0), want_geometry=True)
    return p, v, alpha, info


@mps
def test_geometry_terms_do_not_differentiate_through_alpha():
    """CHECK (1). A depth loss must not be able to buy its error down by fading a splat
    out. `alpha` is the divisor that recovers an attribute from its alpha-weighted
    composite, so if it carried a gradient, reducing coverage would be a descent direction.
    Brush's banned `--depth-source plane-fused` is exactly this failure (opacity p50 -30%).
    """
    from metal_gauss.train import build_parser, geometry_terms
    _, v, alpha, info = _tiny_render()
    a = build_parser().parse_args(["--colmap", "x", "--images", "y",
                                   "--depth-loss-weight", "1.0",
                                   "--depth-normal-weight", "0.05"])
    gt = torch.full_like(alpha, 3.0)
    terms = geometry_terms(a, info["aux"], alpha, v.K, gt_depth=gt)
    for name, t in terms.items():
        g = torch.autograd.grad(t, alpha, allow_unused=True, retain_graph=True)[0]
        assert g is None, f"{name} differentiates through alpha (grad {None if g is None else g.abs().max()})"


@mps
def test_geometry_aux_z_carries_a_gradient_to_means():
    """CHECK (2). The z map must come from torch `means`, not from the preprocess's own
    `depth` output, whose gradient is silently dropped. For an identity viewmat,
    z = means_z broadcast over 3 columns, so d(sum z)/d(means) is exactly (0, 0, 3)."""
    from metal_gauss.train import geometry_aux
    m = torch.randn(5, 3, device="mps").requires_grad_(True)
    q = torch.randn(5, 4, device="mps")
    s = torch.rand(5, 3, device="mps") * 0.1 + 0.01
    _, z = geometry_aux(m, q, s, torch.eye(4))
    z.sum().backward()
    assert m.grad is not None
    assert torch.allclose(m.grad, torch.tensor([[0.0, 0.0, 3.0]], device="mps").expand(5, 3))


@mps
def test_render_view_normals_follow_the_scales_it_renders():
    """CHECK (3). The thin axis must be read off the same scales tensor `render` receives.
    Thin axis x -> camera-facing normal (-1, 0, 0); if the axis order were disturbed the
    map would carry (0, 0, -1) instead."""
    _, _, alpha, info = _tiny_render(scales=(0.001, 0.1, 0.1))
    n = (info["aux"][0] / alpha.clamp_min(1e-10)[..., None])
    covered = alpha > 0.5
    assert covered.any(), "nothing rendered; test is vacuous"
    mean_n = n[covered].mean(0)
    mean_n = mean_n / mean_n.norm()
    assert mean_n[0].item() < -0.9, f"expected the x axis, got {mean_n.tolist()}"


@mps
def test_mask_gates_the_depth_prior_binarily_not_by_scaling_it():
    """CHECK (4). A fractional gate would multiply a METRIC depth by 0.78 and supervise the
    splats toward a surface that is not there. Masks are 0/255 today, so binary and
    fractional agree -- this test uses a 200-valued mask, where they do not, and pins the
    rule before some future float mask source makes it matter for real."""
    from metal_gauss.train import build_parser, geometry_terms
    _, v, alpha, info = _tiny_render()
    a = build_parser().parse_args(["--colmap", "x", "--images", "y",
                                   "--depth-loss-weight", "1.0", "--depth-loss-space", "metric"])
    gt = torch.full_like(alpha, 3.0)
    m01 = torch.full_like(alpha, 200.0 / 255.0)          # 0.784: "keep", not "keep 78%"
    binary = geometry_terms(a, info["aux"], alpha, v.K, gt_depth=gt, keep=(m01 > 0.5))
    frac = geometry_terms(a, info["aux"], alpha, v.K, gt_depth=gt, keep=m01)
    unmasked = geometry_terms(a, info["aux"], alpha, v.K, gt_depth=gt)
    assert binary["depth"].item() == pytest.approx(unmasked["depth"].item(), rel=1e-6)
    assert abs(frac["depth"].item() - unmasked["depth"].item()) > 1e-3


@mps
def test_train_call_site_gates_the_prior_binarily_too():
    """CHECK (4), at the CALL SITE. `geometry_terms` is pinned above, but train() is what
    builds `keep`, and with the usual 0/255 masks `m01` and `(m01 > 0.5)` are the same
    tensor -- so no realistic scene can tell a fractional gate from a binary one, and the
    mutation survives. This scene uses a mask valued 200: still unambiguously "keep", but
    0.784 as a float. Binary must reproduce the 255 run exactly; fractional cannot.

    ONE STEP, deliberately: the PHOTOMETRIC loss weights by the fractional `m01` on
    purpose (Brush semantics), so over many steps the two runs diverge for a legitimate
    reason and the comparison stops isolating the gate. At step 1 the parameters are still
    identical, so the only thing that can move the depth term is how the prior was gated.
    """
    from metal_gauss import train as T
    full = T.train(_args(depth_loss_weight=1.0, steps=1, eval_every=1),
                   scene=_synthetic_scene(mask_value=255))
    partial = T.train(_args(depth_loss_weight=1.0, steps=1, eval_every=1),
                      scene=_synthetic_scene(mask_value=200))
    a = full["log"][-1]["terms"]["depth"]
    b = partial["log"][-1]["terms"]["depth"]
    assert b == pytest.approx(a, rel=1e-6), (
        f"a mask of 200 changed the depth term ({a:.6g} -> {b:.6g}); it is being used as a "
        f"weight (0.784) rather than as a keep/drop decision")
