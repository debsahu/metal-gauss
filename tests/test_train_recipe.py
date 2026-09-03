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
                     masked=True, mask_value=255, seed_z_offset=0.0):
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
    pts[:, 2] = 3.0 + seed_z_offset      # displace the SEED, not the prior
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
def test_depth_term_descends_when_the_seed_starts_at_the_wrong_depth():
    """The depth loss must MOVE GAUSSIANS ALONG Z. Seed at z = 2.9, prior says 3.0.

    DO NOT "SIMPLIFY" THIS BACK TO SEEDING AT THE PRIOR'S DEPTH. The previous version did
    exactly that -- seed z = 3.0, prior 3.0 -- so the depth term began AT ITS OWN FLOOR
    (~0.009) with nothing to descend toward. It passed only because the open
    blending-weight path let the loss reshape splats instead of moving them, so THE TEST
    REQUIRED THE BUG IN ORDER TO PASS, and closing the weight path (Brush ae2ec651's second
    half) broke it. That is the eighth distinct way this project has produced a test result
    that looked like evidence and was not, and the first where the test actively depended
    on the defect it should have caught.

    A CONVERGENCE TEST MUST BE SIZED TO THE OPTIMISER'S ACTUAL REACH, or it measures the
    schedule rather than the mechanism. The default position LR on this 0.40-extent scene
    is 8e-5, so 300 Adam steps command roughly 0.024 m of travel: a 0.3 m displacement
    moves 7% and reads as a broken depth path when nothing is broken. At lr 1e-2 with a
    0.1 m displacement the term falls to 0.37 of its start, which is the mechanism.
    """
    from metal_gauss import train as T
    out = T.train(_args(depth_loss_weight=1.0, steps=300, eval_every=30, lr_means=1e-2),
                  scene=_synthetic_scene(seed_z_offset=-0.1))
    d = [e["terms"]["depth"] for e in out["log"] if "terms" in e]
    assert len(d) >= 4
    assert d[-1] < 0.6 * d[0], f"depth term did not descend: {d[0]:.4g} -> {d[-1]:.4g}"


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


@mps
def test_train_routes_its_photometric_loss_through_photometric_loss():
    """Seam test, and deliberately so: the property under test IS the call, because the
    duplication it replaces was invisible to every value-based test (the two forms agreed
    numerically). A spy that DELEGATES, so the run still has to come out correct."""
    from metal_gauss import train as T
    calls = []
    real = T.photometric_loss

    def spy(*a, **kw):
        calls.append(kw.get("return_terms", False))
        return real(*a, **kw)

    T.photometric_loss = spy
    try:
        out = T.train(_args(steps=3, eval_every=3), scene=_synthetic_scene())
    finally:
        T.photometric_loss = real
    assert len(calls) == 3, f"expected one call per step, got {len(calls)}"
    assert all(calls), "train must ask for the term breakdown, not recompute it"
    assert out["log"][-1]["terms"]["l1"] > 0


@mps
def test_partial_prior_coverage_is_reported_not_silent():
    """The instrument CLAUDE.md needed and did not have: '24 of 276 faces trained
    photometric-only', found long after the fact. The startup check only proves at least
    ONE view carries the prior, so a dataset with 1-of-196 coverage trains almost entirely
    unsupervised and every line of the log looks identical to full coverage."""
    from metal_gauss import train as T
    sc = _synthetic_scene()
    for v in sc.train[2:]:
        v.depth = None                                   # 2 of 5 training views keep depth
    out = T.train(_args(depth_loss_weight=1.0), scene=sc)
    cov = out["metrics"]["term_view_coverage"]
    assert cov["depth"] == [2, len(sc.train)]
    assert out["metrics"]["term_coverage_warning"] is not None


@mps
def test_full_prior_coverage_reports_no_warning():
    from metal_gauss import train as T
    sc = _synthetic_scene()
    out = T.train(_args(depth_loss_weight=1.0, normal_loss_weight=0.2), scene=sc)
    cov = out["metrics"]["term_view_coverage"]
    n = len(sc.train)
    assert cov["depth"] == [n, n] and cov["normal"] == [n, n]
    assert out["metrics"]["term_coverage_warning"] is None


def test_geometry_coverage_warning_fires_only_on_partial_coverage():
    from metal_gauss.train import geometry_coverage_warning
    assert geometry_coverage_warning({"depth": (196, 196)}) is None
    assert geometry_coverage_warning({}) is None
    w = geometry_coverage_warning({"depth": (10, 196), "normal": (196, 196)})
    assert w is not None and "depth" in w and "10/196" in w and "normal" not in w


@mps
def test_every_loss_term_enters_the_total_exactly_once_per_step(monkeypatch):
    """TERM MULTIPLICITY. The general defect; flatten was merely the instance.

    Re-assembling the loss so each half could be logged separately (Fix-up 2) left the
    ORIGINAL `loss = loss + flatten_loss_weight * flatten_loss(...)` in place beside the
    new logged one, so every flatten run in this project trained at 2x the flag -- the
    plan's own probe, this agent's reproduction and the reviewer's all agreed with each
    other because all three carried the same bug. Agreement is not correctness.

    This asserts multiplicity for EVERY term at once, so a future re-assembly cannot
    silently double any of them. It counts CALLS, which catches a duplicated
    `loss = loss + w * term(...)` line -- the shape that actually occurred. It does not
    catch adding an already-computed tensor twice; that needs the trainer's own `loss`
    scalar in the log, which is a change outside the term branches and is deliberately
    deferred (see the report).
    """
    import metal_gauss.train as MT
    from metal_gauss import train as T

    # FORCE THE TORCH LOSS PATH. This test counts calls to depth_loss / normal_loss /
    # depth_normal_loss, and the fused kernel (plan Task 17) calls none of them -- it
    # computes all three in one pass. The fused path's multiplicity is guaranteed
    # structurally (one kernel, one contribution per term) and is pinned numerically by
    # test_fused_geom_loss.py's value-parity test, which would read 2x on a double-add.
    monkeypatch.setenv("MG_TORCH_LOSS", "1")

    steps = 6
    watched = ["photometric_loss", "flatten_loss", "depth_loss", "normal_loss",
               "depth_normal_loss"]
    calls = {k: 0 for k in watched}
    originals = {k: getattr(MT, k) for k in watched}

    def make(name):
        def spy(*a, **kw):
            # Count only calls that can REACH THE TOTAL. Diagnostics -- the depth-normal
            # floor, which is the same term evaluated on the priors -- run under
            # torch.no_grad() and build no graph, so they are not multiplicity.
            if torch.is_grad_enabled():
                calls[name] += 1
            return originals[name](*a, **kw)
        return spy

    for k in watched:
        setattr(MT, k, make(k))
    try:
        T.train(_args(steps=steps, eval_every=steps, flatten_loss_weight=1.0,
                      depth_loss_weight=1.0, normal_loss_weight=0.2,
                      depth_normal_weight=0.05),
                scene=_synthetic_scene())
    finally:
        for k, v in originals.items():
            setattr(MT, k, v)

    for name in watched:
        per_step = calls[name] / steps
        assert per_step == 1.0, (
            f"{name} entered the differentiable graph {per_step} times per step, not once "
            f"-- that term enters the total {per_step}x, so its effective weight is "
            f"{per_step}x the flag")


def test_run_report_uses_the_env_snapshot_it_is_given_not_a_fresh_git_query():
    """PROVENANCE. `_run_report` used to call git at REPORT-WRITE time, so a run that had
    already imported train.py recorded whatever commit existed when it FINISHED. On
    2026-09-02 arm B0a executed c047dad and recorded 727b8ba, a commit made 6 minutes after
    it started -- the mechanism claimed to answer "what code produced this number?" and
    answered "what was checked out when it stopped?".

    The snapshot must be taken at process start and used verbatim."""
    import argparse
    from metal_gauss.train import _run_report
    snap = {"git": "DEADBEEF", "dirty": True, "torch": "x", "platform": "y",
            "machine": "z", "started_at": "2026-01-01T00:00:00Z"}
    r = _run_report(argparse.Namespace(steps=10, budget=7, seed=0), [], 1.0, 7, env=snap)
    assert r["env"]["git"] == "DEADBEEF", "re-queried git instead of using the snapshot"
    assert r["env"]["dirty"] is True
    assert r["env"]["started_at"] == "2026-01-01T00:00:00Z"
    assert "finished_at" in r["env"] and r["env"]["finished_at"] > r["env"]["started_at"]


def test_env_snapshot_reports_a_real_commit_and_a_start_timestamp():
    from metal_gauss.train import _env_snapshot
    s = _env_snapshot()
    assert s["git"] is None or len(s["git"]) >= 7
    assert isinstance(s["dirty"], bool)
    assert s["started_at"].endswith("Z") and "T" in s["started_at"]


@mps
def test_train_snapshots_the_environment_before_it_starts_training():
    """End to end: the report's started_at must predate finished_at, and the hash must be
    the one captured at entry."""
    from metal_gauss import train as T
    out = T.train(_args(steps=2, eval_every=2), scene=_synthetic_scene())
    env = out["env"]
    assert env["started_at"] < env["finished_at"]
    assert env["git"] is None or len(env["git"]) >= 7


def test_shape_metrics_measure_the_IN_PLANE_aspect_not_the_thinness():
    """The instrument that was missing. Flatten legitimately drives the SMALLEST axis down
    -- that is a disc, and thin-axis/opacity/dark all correctly report it as healthy. A
    NEEDLE is the MIDDLE axis collapsing while the largest holds, and nothing in the
    battery could see it: P-GEOM's R1 read healthier than baseline on all three of those
    metrics while smid fell 6.62 -> 1.35 mm at constant smax.

    So the ratio must be smid/smax. Using smin/smax instead would score a perfect disc as a
    needle and this whole instrument would be noise."""
    from metal_gauss.train import shape_metrics
    disc = torch.log(torch.tensor([[0.001, 0.020, 0.020]] * 4))     # flat, NOT a needle
    m = shape_metrics(disc)
    assert m["aspect_p50"] == pytest.approx(1.0, rel=1e-5)
    assert m["needle_frac"] == 0.0
    needle = torch.log(torch.tensor([[0.001, 0.001, 0.020]] * 4))   # smid collapsed
    n = shape_metrics(needle)
    assert n["aspect_p50"] == pytest.approx(0.05, rel=1e-5)
    assert n["needle_frac"] == 1.0
    mixed = torch.log(torch.tensor([[0.001, 0.020, 0.020], [0.001, 0.001, 0.020]]))
    assert shape_metrics(mixed)["needle_frac"] == pytest.approx(0.5)


def test_shape_metrics_are_invariant_to_axis_order():
    """The scales are unordered per splat; the metric must sort."""
    from metal_gauss.train import shape_metrics
    a = shape_metrics(torch.log(torch.tensor([[0.02, 0.001, 0.02]])))
    b = shape_metrics(torch.log(torch.tensor([[0.001, 0.02, 0.02]])))
    assert a["aspect_p50"] == pytest.approx(b["aspect_p50"])


@mps
def test_training_logs_shape_metrics_and_the_depth_normal_floor():
    """Two instruments, both absent when the collapse happened.

    `aspect_p50` / `needle_frac` say WHAT the splats became. `dn_floor` says what
    `depth_normal` could achieve on this data at all: it is the same term evaluated on the
    PRIOR depth and PRIOR normals, so it is the value a perfect render would score. A term
    sitting far above its own floor and climbing is broken, and that should be legible
    during training rather than in a post-mortem."""
    from metal_gauss import train as T
    out = T.train(_args(steps=6, eval_every=6, depth_normal_weight=0.05,
                        depth_loss_weight=1.0, normal_loss_weight=0.2),
                  scene=_synthetic_scene())
    e = out["log"][-1]
    for k in ("aspect_p50", "needle_frac"):
        assert k in e["shape"] and math.isfinite(e["shape"][k]), k
    assert 0.0 <= e["shape"]["needle_frac"] <= 1.0
    assert "dn_floor" in e["terms"], "the depth-normal floor must be logged beside the term"
    assert math.isfinite(e["terms"]["dn_floor"])
    assert out["metrics"]["shape"] == e["shape"]


@mps
def test_depth_normal_floor_is_small_on_a_scene_whose_priors_are_exact():
    """The synthetic wall is fronto-parallel with depth exactly 3.0 and normal exactly
    (0,0,-1), so normals differentiated from the PRIOR depth agree with the PRIOR normals
    almost perfectly. A floor near 1.0 would mean the floor itself is meaningless."""
    from metal_gauss import train as T
    out = T.train(_args(steps=4, eval_every=4, depth_normal_weight=0.05,
                        depth_loss_weight=1.0, normal_loss_weight=0.2),
                  scene=_synthetic_scene())
    assert out["log"][-1]["terms"]["dn_floor"] < 0.05
