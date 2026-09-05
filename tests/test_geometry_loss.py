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


W_FLAT = 50.0          # any w large enough to flip the min-axis gradient sign; see probe C


@pytest.fixture(scope="module")
def flatten_arms(tmp_path_factory):
    """Three one-step training runs at flatten weights 0, W and 2W, sharing one scene.

    WHY ONE STEP. At step 1 the splat state is EXACTLY the seed in all three arms (same
    --seed, --no-grow, same view order), so the photometric terms, the regularisers and
    the rendered view are bit-identical and the ONLY admissible difference between arms
    is `w * flatten`. That makes both the total loss and d(loss)/d(log_scales) exactly
    linear in w, which is a mechanism these probes can check to f32 precision instead of
    an effect size an optimiser can cap.

    The hooks are additive; none of them changes what train() computes.
      make_optimizer -> stash the optimiser, so the named "scales" param group is
          reachable in EVERY arm including w = 0, where flatten_loss is never called.
      flatten_loss   -> record the term, whether it carries grad, and the analytic
          d(flatten)/d(log_scales) for that arm's seed.
      Tensor.backward -> record the total loss and d(loss)/d(log_scales) at step 1.
    """
    pytest.importorskip("pycolmap")
    if not torch.backends.mps.is_available():
        pytest.skip("needs MPS")
    import numpy as np
    from PIL import Image
    import metal_gauss.train as T
    from metal_gauss.train import build_parser, train

    root = tmp_path_factory.mktemp("flatten")
    (root / "sparse").mkdir(); (root / "images").mkdir()
    (root / "sparse" / "cameras.txt").write_text("1 PINHOLE 32 32 32 32 16 16\n")
    rng = np.random.default_rng(0)
    lines = []
    for i in range(3):
        lines.append(f"{i + 1} 1 0 0 0 0 0 {i * 0.3 - 0.3} 1 v{i}.png\n\n")
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(
            root / "images" / f"v{i}.png")
    (root / "sparse" / "images.txt").write_text("".join(lines))
    pts = rng.normal(0, 0.4, (60, 3)) + np.array([0, 0, 3.0])
    (root / "sparse" / "points3D.txt").write_text("".join(
        f"{i + 1} {x} {y} {z} 128 128 128 0.5\n" for i, (x, y, z) in enumerate(pts)))

    def run(w, tag):
        rec = {"w": w, "flatten_calls": 0, "flatten": None, "grad_ls": None,
               "term_requires_grad": None, "dflatten_dls": None}
        real_mo, real_fl, real_bw = T.make_optimizer, T.flatten_loss, torch.Tensor.backward

        def mo(*a, **k):
            o = real_mo(*a, **k)
            g = next(g for g in o.param_groups if g.get("name") == "scales")
            rec["scales_param"], rec["scales_lr"] = g["params"][0], g["lr"]
            return o

        def fl(ls):
            out = real_fl(ls)
            rec["flatten_calls"] += 1
            rec["flatten"] = float(out.detach())
            rec["term_requires_grad"] = bool(out.requires_grad)
            with torch.no_grad():                 # analytic d(flatten)/d(log_scales)
                e = ls.detach().exp()
                am = e.argmin(dim=1, keepdim=True)
                rec["dflatten_dls"] = (torch.zeros_like(e).scatter_(
                    1, am, e.gather(1, am)) / e.shape[0]).cpu().clone()
            return out

        def bw(self, *a, **k):
            r = real_bw(self, *a, **k)
            if "loss" not in rec:                                  # the step-1 backward
                rec["loss"] = float(self.detach())
                sp = rec.get("scales_param")
                rec["grad_ls"] = None if sp is None or sp.grad is None else sp.grad.cpu().clone()
            return r

        T.make_optimizer, T.flatten_loss, torch.Tensor.backward = mo, fl, bw
        try:
            a = build_parser().parse_args([
                "--colmap", str(root / "sparse"), "--images", str(root / "images"),
                "--steps", "1", "--budget", "400", "--max-resolution", "32",
                "--eval-every", "1", "--eval-split-every", "1000", "--seed", "0",
                "--num-downscales", "0", "--no-grow", "--sh-warmup", "0",
                "--flatten-loss-weight", str(w), "--export", str(root / f"{tag}.ply")])
            a.resolution_schedule = 1
            train(a)
        finally:
            T.make_optimizer, T.flatten_loss, torch.Tensor.backward = real_mo, real_fl, real_bw

        import plyfile
        v = plyfile.PlyData.read(str(root / f"{tag}.ply"))["vertex"]
        ls = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)
        rec["ply_log_min"] = ls.min(1)
        rec["ply_minaxis_p50"] = float(np.median(np.exp(ls).min(1)))
        assert "loss" in rec, f"w={w}: no backward was observed; the probe is vacuous"
        return rec

    return {0.0: run(0.0, "w0"), 1.0: run(W_FLAT, "w1"), 2.0: run(2 * W_FLAT, "w2")}


def test_flatten_flag_actually_reaches_the_training_loss(flatten_arms):
    """Wiring, not arithmetic. This repo's failure log is full of flags that PARSE and do
    nothing -- LFS's `--train` is a no-op, its `--init=path.ply` is dead, and a harness
    once forwarded --budget so auto_budget() never ran in an 8-scene sweep. A unit test on
    flatten_loss() cannot see any of that.

    REPLACED 2026-09-04. This test used to run two 40-step arms and assert a >=10% drop in
    the median min axis. That assertion SATURATES: at step 1 Adam's update is exactly
    -lr*sign(g), so every w past the point that flips the min-axis gradient sign gives the
    IDENTICAL result -- MEASURED, w = 50 and w = 100 export min-axis p50 0.278286368 and
    0.278286368, bit-identical, and the previous agent measured w = 1, 50 and 200 identical
    to 0.1 pp at 40 steps. The 10% bar was therefore measuring the optimiser's step budget,
    not flatten's strength, and could not be repaired by raising the weight. It also read
    10.51-10.67% on the lockfile torch against 9.38-9.48% on 2.14.0, i.e. it sat on the
    edge of its own bar for a reason that had nothing to do with flatten.

    What replaces it asserts the mechanism the name claims, at f32 precision: the total
    loss is exactly linear in the flatten weight, because at step 1 nothing else can differ.

    MEASURED on clean code, twice, identical to every digit: rel residual 1.9e-8 (W-0),
    6.4e-8 (2W-W), 1.9e-8 (2W-0) -- f32 rounding of a single addition. An inert term gives
    EXACTLY 0.0, a halved weight gives 0.5 and a sign flip gives 2.0, so the bar below
    separates clean from broken by six orders. Confirmed by substitution against seven
    mutants; this test kills inert, half-weight, sign-flip and double-add, and is BLIND by
    construction to a detached term and to a term spread over all three axes -- both of
    which leave the loss value exactly right. Those two are why
    test_flatten_gradient_reaches_log_scales exists.

    NOTE the earlier record of this probe said the residual was 4.4e-16 at w=10 and 1.8e-15
    at w=50, i.e. f64 roundoff. That does not reproduce and cannot be right: the loss is
    f32, so `fl32(loss + w*flatten) - loss` differs from `w*flatten` by ~eps_f32, which is
    what the 2e-8 - 6e-8 above is. The probe is exact to f32, not to f64.
    """
    a0, a1, a2 = flatten_arms[0.0], flatten_arms[1.0], flatten_arms[2.0]

    # The block must run in the weighted arms and NOT in the w = 0 arm. Without this a
    # mutant that deletes both lines 618-619 would leave `flatten` at None and the
    # arithmetic below would never execute.
    assert a0["flatten_calls"] == 0, "flatten_loss ran at weight 0"
    assert a1["flatten_calls"] == 1 and a2["flatten_calls"] == 1, (
        f"flatten_loss ran {a1['flatten_calls']}/{a2['flatten_calls']} times, expected once "
        "-- a second add would double the term, which has happened in this project before")
    assert a1["flatten"] is not None and a1["flatten"] > 0.0

    # THE NULL, first: the predicted contribution must dominate the loss it is added to,
    # or "the two arms agree" would be a statement about two near-equal numbers.
    # Measured 33.60x.
    pred = W_FLAT * a1["flatten"]
    assert pred > a0["loss"], (
        f"fixture does NOT discriminate: w*flatten {pred:.3e} is not large against "
        f"loss(w=0) {a0['loss']:.3e}, so a missing term would barely move the total")

    for name, hi, lo, k in (("W-0", a1, a0, 1), ("2W-W", a2, a1, 1), ("2W-0", a2, a0, 2)):
        d = hi["loss"] - lo["loss"]
        # IMPOSSIBLE VALUE, not a plausible one: an inert term gives exactly zero.
        assert d != 0.0, f"{name}: d_loss is EXACTLY 0.0 -- the flatten term is INERT"
        rel = abs(d - k * pred) / (k * pred)
        assert rel <= 1e-6, (
            f"{name}: d_loss {d:.9e} != w*flatten {k * pred:.9e} (rel {rel:.3e}); the term "
            "reaches the loss with the wrong weight, wrong sign, or is added twice")


def test_flatten_gradient_reaches_log_scales(flatten_arms):
    """ADDED 2026-09-04 alongside the rewrite above, and it catches a mutant the loss-value
    probe CANNOT see: `flatten_loss(p["log_scales"][:active].detach())`. That leaves the
    loss value exactly right -- the probe above passes clean -- while the term contributes
    no gradient at all, so flatten does nothing to the model.

    At step 1 the photometric gradient is identical across arms, so the difference between
    two arms' d(loss)/d(log_scales) is exactly `dw * d(flatten)/d(log_scales)`, which is
    analytically `exp(s_min)/N` on each splat's argmin lane and EXACTLY zero on the other
    two. Both halves are asserted: the lanes that should move do, elementwise, and the
    lanes that should not stay put.

    MEASURED clean: per-element relative error on the 400 predicted lanes max 9.9e-8 -
    1.5e-7; leakage onto the other 800 lanes <= 1.2e-8 of the on-lane magnitude. Scored
    PER ELEMENT deliberately -- `max|delta| / max|ref|` is the form the Checkpoint F audit
    found could be satisfied by a single degenerate value.

    WHAT THIS CANNOT SEE, stated because the step-1 design makes it structural rather than
    incidental. At the moment flatten_loss is called the splat state is the seed, and the
    seed is EXACTLY isotropic -- measured, per-splat log-scale spread max 0.000e+00 on
    400 of 400 -- so min, max and any other order statistic coincide. Replacing `.min()`
    with `.max()` in flatten_loss is therefore a BEHAVIOURALLY IDENTICAL mutant here and
    this probe passes it; confirmed by substitution. Lane SELECTION is owned by
    test_flatten_loss_is_mean_min_activated_scale_and_only_moves_min_axis and
    test_flatten_loss_gradient_pushes_the_thin_axis_DOWN, which build anisotropic input by
    hand and do kill it. What this probe does own is lane COUNT: spreading the term over
    all three axes (`.min()` -> `.mean()`) leaves the value untouched under an isotropic
    seed, so the loss probe above cannot see it, and this one kills it on the leakage
    assertion. Both substitutions were run.
    """
    a0, a1, a2 = flatten_arms[0.0], flatten_arms[1.0], flatten_arms[2.0]
    assert a1["term_requires_grad"] is True, "the flatten term carries no gradient at all"
    for r in (a0, a1, a2):
        assert r["grad_ls"] is not None, "no d(loss)/d(log_scales) was captured"

    for name, hi, lo, k in (("W-0", a1, a0, 1.0), ("2W-W", a2, a1, 1.0), ("2W-0", a2, a0, 2.0)):
        d = (hi["grad_ls"] - lo["grad_ls"]).double()
        want = (a1["dflatten_dls"] * (k * W_FLAT)).double()
        lane = want.abs() > 0

        # THE NULL. Measured max|want| = 4.9e-2 at k = 1. If this ever falls quiet the
        # comparison below is satisfied by two zeros.
        assert want.abs().max().item() > 1e-3, (
            f"{name}: the predicted flatten gradient is ~0, so this test is vacuous")
        assert int(lane.sum()) == want.shape[0], (
            f"{name}: expected exactly one lane per splat, got {int(lane.sum())} "
            f"of {want.numel()}")

        rel = ((d[lane] - want[lane]).abs() / want[lane].abs()).max().item()
        assert rel <= 1e-5, (
            f"{name}: d(loss)/d(log_scales) does not carry w * d(flatten)/d(log_scales) "
            f"(worst per-element rel {rel:.3e}). A detached term reads 1.0 here.")

        leak = d[~lane].abs().max().item() / want.abs().max().item()
        assert leak <= 1e-6, (
            f"{name}: flatten leaked gradient onto lanes that are not the smallest axis "
            f"({leak:.3e} of the on-lane magnitude)")


def test_flatten_moves_the_min_axis_through_the_optimiser(flatten_arms):
    """The end-to-end half, and the only thing the two probes above do not cover: that the
    gradient survives into the optimiser and actually moves the parameter, in the right
    direction. This is what the deleted 40-step arm was really testing.

    Asserted as a DIRECTION with no magnitude bar, so it cannot saturate the way the 10%
    assertion did. One Adam step is bit-reproducible here (measured: identical to nine
    decimals across two full repeats), so the noise floor is zero and `<` is meaningful.

    MEASURED clean: min-axis p50 0.280929387 (w=0) -> 0.278286368 (w=W), with 62.75% of
    splats lower in log space and a median shift of exactly -2*lr = -1.0e-2, which is
    Adam's first step being -lr*sign(g) on both sides of a flipped sign. That last number
    is also the saturation: it does not depend on w, which is why the old bar could not be
    rescued by raising the weight.
    """
    import numpy as np
    a0, a1 = flatten_arms[0.0], flatten_arms[1.0]
    d = a0["ply_log_min"] - a1["ply_log_min"]          # > 0 where flatten shrank the axis

    assert not np.array_equal(a0["ply_log_min"], a1["ply_log_min"]), (
        "the exported min axis is bit-identical with and without flatten: the gradient "
        "never reached the optimiser")
    assert a1["ply_minaxis_p50"] < a0["ply_minaxis_p50"], (
        f"flatten did not shrink the min axis: p50 {a0['ply_minaxis_p50']:.9f} -> "
        f"{a1['ply_minaxis_p50']:.9f}")
    assert float((d > 0).mean()) > 0.5, (
        f"only {100 * float((d > 0).mean()):.2f}% of splats moved their min axis DOWN; "
        "a sign error inverts this")
    assert d.min() >= 0.0, (
        f"flatten INFLATED the min axis of at least one splat by {-d.min():.3e} in log "
        "space; the term can only ever push it down")


# ---------------------------------------------------- normals_from_depth (Task 6)

def _fixture_case(name):
    fx_ = json.loads(FIX.read_text())
    return fx_, next(k for k in fx_["cases"] if k["name"] == name)


def _rays(h, w, fx, fy, cx, cy):
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float64),
                          torch.arange(w, dtype=torch.float64), indexing="ij")
    return torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], -1)


@pytest.mark.parametrize("case", ["slanted_plane", "grazing_plane_positive_nz"])
def test_normals_from_depth_matches_fixture(case):
    from metal_gauss.geometry_loss import normals_from_depth
    fx_, c = _fixture_case(case)
    K = c["intrinsics"]
    depth = torch.tensor(c["depth"], dtype=torch.float64)
    want = torch.tensor(c["expected_normal"], dtype=torch.float64)
    got = normals_from_depth(depth, K["fx"], K["fy"], K["cx"], K["cy"])
    valid = want.norm(dim=-1) > 0.5
    assert torch.allclose(got[valid], want[valid], atol=fx_["tolerance"]), \
        (got[valid] - want[valid]).abs().max()
    assert (got[~valid] == 0).all()


def test_the_nz_rule_and_the_ray_rule_disagree_on_EVERY_valid_pixel_of_the_grazing_case():
    """THE NULL MODEL. Before trusting a test to discriminate two rules, prove on this data
    that they actually differ -- the repo's previous fixture used a plane family where both
    rules agree, so the test that was supposed to pin the flip could never have failed.

    Here the two candidate orientations are computed side by side from the SAME raw cross
    product, and they must be exact negations on 100% of valid pixels."""
    _, c = _fixture_case("grazing_plane_positive_nz")
    K = c["intrinsics"]
    depth = torch.tensor(c["depth"], dtype=torch.float64)
    h, w = depth.shape
    u = (torch.arange(w, dtype=torch.float64) - K["cx"]) / K["fx"]
    v = (torch.arange(h, dtype=torch.float64) - K["cy"]) / K["fy"]
    P = torch.stack([depth * u[None, :], depth * v[:, None], depth], -1)
    base = P[:-1, :-1]
    raw = torch.cross(P[:-1, 1:] - base, P[1:, :-1] - base, dim=-1)   # cross(du, dv)
    raw = raw / raw.norm(dim=-1, keepdim=True)
    r = _rays(h, w, K["fx"], K["fy"], K["cx"], K["cy"])[:-1, :-1]
    by_ray = torch.where(((raw * r).sum(-1) > 0)[..., None], -raw, raw)
    by_nz = torch.where((raw[..., 2] > 0)[..., None], -raw, raw)
    assert torch.allclose(by_nz, -by_ray), "this fixture cannot discriminate the two rules"
    assert (by_ray[..., 2] > 0).all(), "ray-correct normals must have n_z > 0 here"
    # ...and the shipped function must agree with the RAY arm, not the n_z arm.
    from metal_gauss.geometry_loss import normals_from_depth
    got = normals_from_depth(depth, K["fx"], K["fy"], K["cx"], K["cy"])[:-1, :-1]
    assert torch.allclose(got, by_ray, atol=1e-12)
    assert not torch.allclose(got, by_nz, atol=1e-3)


def test_grazing_plane_discriminates_the_ray_rule_from_the_nz_rule():
    """The fixture's second case exists because the first could not tell the two rules apart.
    Every camera-facing normal here has n_z > 0, so an `n_z <= 0` rule negates ALL of them."""
    from metal_gauss.geometry_loss import normals_from_depth
    _, c = _fixture_case("grazing_plane_positive_nz")
    K = c["intrinsics"]
    depth = torch.tensor(c["depth"], dtype=torch.float64)
    n = normals_from_depth(depth, K["fx"], K["fy"], K["cx"], K["cy"])
    valid = n.norm(dim=-1) > 0.5
    r = _rays(*depth.shape, K["fx"], K["fy"], K["cx"], K["cy"])
    assert ((n * r).sum(-1)[valid] <= -0.2).all()      # camera-facing, per-pixel ray
    assert (n[..., 2][valid] > 0.3).all()              # ...and n_z POSITIVE everywhere
    assert valid.sum() == 35


def test_normals_from_depth_faces_the_camera_on_random_depth_graphs():
    """The invariant is a theorem for ANY depth graph, not just planes:
        dot(cross(dPdu, dPdv), r) = z(u+1,v) z(u,v+1) / (fx fy)
    which is > 0 exactly when validity holds. So the correct orientation is a global
    negation and needs no data-dependent test. 200 random graphs, including a wide,
    off-centre principal point where n_z and the ray diverge hardest."""
    from metal_gauss.geometry_loss import normals_from_depth
    g = torch.Generator().manual_seed(0)
    worst_nz_violations = 0
    for _ in range(200):
        depth = torch.rand(9, 11, generator=g, dtype=torch.float64) * 4.0 + 0.2
        fx, fy, cx, cy = 3.0, 4.0, -6.0, 7.0        # deliberately extreme / off-centre
        n = normals_from_depth(depth, fx, fy, cx, cy)
        valid = n.norm(dim=-1) > 0.5
        r = _rays(9, 11, fx, fy, cx, cy)
        assert ((n * r).sum(-1)[valid] < 0).all()
        assert torch.allclose(n[valid].norm(dim=-1),
                              torch.ones(int(valid.sum()), dtype=torch.float64))
        worst_nz_violations += int((n[..., 2][valid] > 0).sum())
    assert worst_nz_violations > 0, \
        "no sample had n_z > 0, so this test never exercised the rules' disagreement"


def test_normals_from_depth_zeroes_the_border_and_any_nonpositive_contributor():
    from metal_gauss.geometry_loss import normals_from_depth
    depth = torch.full((4, 5), 2.0, dtype=torch.float64)
    depth[1, 1] = 0.0                        # invalid: kills (1,1), (0,1) and (1,0)
    n = normals_from_depth(depth, 2.0, 2.0, 2.0, 1.5)
    assert (n[-1, :] == 0).all() and (n[:, -1] == 0).all()      # forward differences
    for p in [(1, 1), (0, 1), (1, 0)]:
        assert (n[p] == 0).all(), p
    assert (n[2, 2] != 0).any()                                  # untouched neighbour lives


def test_normals_from_depth_is_all_zero_on_a_degenerate_shape():
    from metal_gauss.geometry_loss import normals_from_depth
    assert (normals_from_depth(torch.ones(1, 5, dtype=torch.float64), 1, 1, 0, 0) == 0).all()


# ---------------------------------------------------------------- splat normals

def test_splat_normals_cam_faces_camera_by_ray_not_axis():
    """THE orientation test. The axis rule must give a definite WRONG answer here, not an
    undecidable one.

    An earlier version of this test put the thin axis exactly perpendicular to the optical
    axis so that `n_cam[:, 2] == 0`, reasoning that the n_z rule "cannot decide". That was
    the mistake: an undecidable rule still returns something, and at that geometry its
    tie-break coincided with the ray answer, so the `facing = n_cam[:, 2]` mutant PASSED
    this test. The suite killed it only via the unrelated column-vs-row test. This repo has
    now shipped that shape of non-test three times (a plane family where both rules agree;
    a flip test asserting over an empty array; this).

    So: grazing geometry, borrowed from the fixture's case 2. The splat's normal has
    n_z = +0.707 -- STRICTLY positive, so the axis rule is decidable -- while
    dot(n, p_cam) = -1.414, so it is already facing the camera and must NOT be flipped.
    The axis rule flips it and ends up pointing away; the ray rule leaves it alone.
    """
    from metal_gauss.geometry_loss import splat_normals_cam
    r2 = math.sqrt(0.5)
    quats = torch.tensor([[0.38268343, 0.0, -0.92387953, 0.0]])   # -135 deg about y
    scales = torch.tensor([[0.001, 0.1, 0.1]])                    # thin axis = column 0
    means = torch.tensor([[3.0, 0.0, 1.0]])                       # off-axis, in front
    n = splat_normals_cam(means, quats, scales, torch.eye(4))

    assert n[0, 2].item() > 0.1, "n_z must be decidably POSITIVE or the axis rule is mute"
    assert torch.allclose(n, torch.tensor([[-r2, 0.0, r2]]), atol=1e-6), \
        f"the axis rule would flip this to (+.707, 0, -.707). got {n}"
    assert (n * means).sum() < 0                     # camera-facing, by the per-splat ray


def test_splat_normals_cam_does_not_annihilate_a_perpendicular_splat():
    """`sign()` returns 0 at exactly 0 and multiplying by it deletes the normal. A `>`
    comparison keeps it. A zero normal is not 'no opinion' downstream -- depth_normal_loss
    gates on norm > 0.5, so an annihilated splat silently leaves the loss."""
    from metal_gauss.geometry_loss import splat_normals_cam
    means = torch.tensor([[0.0, 0.0, 5.0]])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    scales = torch.tensor([[0.1, 0.001, 0.1]])       # thin axis = y, exactly perpendicular
    n = splat_normals_cam(means, quats, scales, torch.eye(4))
    assert (n * means).sum() == 0.0                  # the degenerate case, by construction
    assert n.norm().item() == pytest.approx(1.0)
    # Which of the two signs is taken at exactly 0 is arbitrary -- but Brush's
    # `splat_normals` builds its selector as `(facing < 0) * 2 - 1`, so it NEGATES the tie,
    # and this port matches it for cross-implementation parity. Pinned so a future
    # "simplification" to `> 0` is caught rather than silently diverging.
    assert torch.allclose(n, torch.tensor([[0.0, -1.0, 0.0]]))


def test_splat_normals_cam_applies_the_view_rotation():
    """n is a CAMERA-frame vector. Rotating the camera 90 deg about z must rotate it.

    The splat is off-axis so `dot(n_cam, p_cam) = 1`, comfortably off the perpendicular
    tie -- a splat on the optical axis puts every in-plane normal at dot 0, where the
    answer is a tie-break rather than a rotation."""
    from metal_gauss.geometry_loss import splat_normals_cam
    vm = torch.eye(4)
    vm[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    vm[:3, 3] = torch.tensor([0.0, 0.0, 5.0])
    means = torch.tensor([[1.0, 0.0, 0.0]])          # p_cam = (0, 1, 5)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    scales = torch.tensor([[0.001, 0.1, 0.1]])       # thin axis = world x -> cam +y
    n = splat_normals_cam(means, quats, scales, vm)
    assert torch.allclose(n, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6)
    # Skipping the rotation entirely would leave n_cam = (1,0,0) and answer (-1,0,0).
    assert not torch.allclose(n, torch.tensor([[-1.0, 0.0, 0.0]]), atol=1e-6)


def test_splat_normals_cam_picks_the_thinnest_axis_column_of_R():
    from metal_gauss.geometry_loss import splat_normals_cam
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    m = torch.tensor([[0.0, 0.0, 5.0]]).repeat(3, 1)
    s = torch.tensor([[0.001, 0.1, 0.2], [0.2, 0.001, 0.1], [0.2, 0.1, 0.001]])
    n = splat_normals_cam(m, q, s, torch.eye(4))
    assert torch.allclose(n.abs(), torch.eye(3), atol=1e-6)


def test_splat_normals_cam_takes_the_COLUMN_of_R_not_the_row():
    """Every other splat test here uses an identity quaternion, where R is symmetric and
    row == column -- so none of them can tell the two apart. `build_cov3d` forms
    `R * scales[:, None, :]` = `R @ diag(s)`, so scale i scales COLUMN i; indexing the row
    transposes the rotation and points the normal somewhere else entirely.

    The rotation must be chosen with care: for a 90-degree rotation about z, row 0 is
    exactly MINUS column 0, and the camera-facing flip then maps one onto the other, so
    such a test cannot discriminate at all. Here R is the 120-degree rotation about
    (1,1,1) that cycles x->y->z->x, whose column 0 (+y) and row 0 (+z) are ORTHOGONAL, and
    the splat is placed off the perpendicular tie so neither arm lands on a coin flip:
    the column answers +y, the row answers -z."""
    from metal_gauss.geometry_loss import splat_normals_cam
    q = torch.tensor([[0.5, 0.5, 0.5, 0.5]])         # 120 deg about (1,1,1): x->y->z->x
    m = torch.tensor([[0.0, -1.0, 5.0]])
    s = torch.tensor([[0.001, 0.1, 0.2]])            # thin axis index 0
    n = splat_normals_cam(m, q, s, torch.eye(4))
    assert torch.allclose(n, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6), \
        f"expected R's column 0 (+y); the row would give -z. got {n}"


def test_splat_normals_cam_gradient_reaches_quats_not_scales():
    from metal_gauss.geometry_loss import splat_normals_cam
    torch.manual_seed(0)
    q = torch.randn(5, 4, requires_grad=True)
    s = torch.rand(5, 3, requires_grad=True)
    m = torch.randn(5, 3) + torch.tensor([0, 0, 4.0])
    splat_normals_cam(m, q, s, torch.eye(4)).sum().backward()
    assert q.grad.abs().sum() > 0
    assert s.grad is None or s.grad.abs().sum() == 0   # argmin detached, as in Brush


# ---------------------------------------------------------------- the three losses

def test_depth_loss_disparity_counts_uncovered_and_ignores_invalid_gt():
    from metal_gauss.geometry_loss import depth_loss
    gt = torch.tensor([[2.0, 0.0], [4.0, 1.0]])
    pred = torch.tensor([[2.0, 7.0], [0.0, 1.0]])
    # (0,1): gt invalid -> ignored. (1,0): pred uncovered (0) -> full disparity error 1/4.
    assert depth_loss(pred, gt, "disparity").item() == pytest.approx((0 + 0.25 + 0) / 3)
    assert depth_loss(pred, gt, "metric").item() == pytest.approx((0 + 4.0 + 0) / 3)


def test_depth_loss_divides_by_the_VALID_count_not_the_pixel_count():
    """Dividing by H*W makes the term shrink with the fraction of invalid pixels, so a
    sparse LiDAR prior would be silently down-weighted against a dense one."""
    from metal_gauss.geometry_loss import depth_loss
    gt = torch.zeros(4, 4); gt[0, 0] = 2.0
    pred = torch.full((4, 4), 1.0)
    assert depth_loss(pred, gt, "metric").item() == pytest.approx(1.0)      # not 1/16


def test_depth_loss_with_no_valid_pixels_is_zero_not_nan():
    from metal_gauss.geometry_loss import depth_loss
    l = depth_loss(torch.ones(3, 3), torch.zeros(3, 3))
    assert l.item() == 0.0 and torch.isfinite(l)


@pytest.mark.parametrize("space", ["disparity", "metric"])
def test_depth_loss_is_finite_with_inf_and_nan_outside_the_mask(space):
    """`x * 0` is NaN for inf and NaN, so masking MUST substitute before the arithmetic,
    not multiply after it. A single non-finite render pixel would otherwise NaN the whole
    training step -- and Adam writes NaN into the parameters on the first such step.

    BOTH spaces, deliberately. In disparity the `pred > 0` guard happens to re-mask a NaN
    that leaked in (NaN > 0 is False), so a multiply-after implementation passes the
    disparity case and NaNs only in metric -- which is exactly the kind of half-covered
    test that ships a latent defect."""
    from metal_gauss.geometry_loss import depth_loss
    inf, nan = float("inf"), float("nan")
    # A NaN in the GROUND TRUTH matters too: `--prior-resident float32` reads a TIFF
    # straight through, so a corrupt prior arrives here unsanitised. `nan > 0` is False, so
    # the pixel is correctly invalid -- but `nan * 0` is still NaN, and a multiply-after
    # implementation NaNs the run on a single bad prior pixel.
    gt = torch.tensor([[1.0, 0.0, 0.0, nan]])
    pred = torch.tensor([[1.0, inf, nan, 2.0]], requires_grad=True)
    l = depth_loss(pred, gt, space)
    l.backward()
    assert torch.isfinite(l), f"{space}: loss is {l}"
    assert torch.isfinite(pred.grad).all(), f"{space}: grad is {pred.grad}"


def test_depth_loss_uncovered_lane_gradient_is_exactly_zero_NOT_brush_count_nan():
    """DELIBERATE DIVERGENCE FROM BRUSH -- do not "restore parity" by removing it.

    Brush's `Count` mode forms `pred.recip()` and masks afterwards, so an uncovered lane
    (`pred == 0`) goes through `recip(0) = inf` and its VJP comes back NaN. Brush pins that
    in `exclude_numerator_preserves_every_finite_disparity_gradient` and its own comment
    calls it a latent defect, contained only because no gaussian in its pipeline touches an
    uncovered pixel. Ours guards BEFORE the reciprocal, so:

        forward  -- identical to Brush `Count` (the lane scores the full disparity 1/gt),
        backward -- 0, i.e. Brush's `ExcludeNumerator` behaviour, not `Count`'s NaN.

    A single NaN here reaches Adam and writes NaN into every parameter on the first step.
    """
    from metal_gauss.geometry_loss import depth_loss
    gt = torch.tensor([[4.0, 2.0]])
    pred = torch.tensor([[0.0, 2.0]], requires_grad=True)     # lane 0 uncovered
    l = depth_loss(pred, gt, "disparity")
    l.backward()
    assert l.item() == pytest.approx(0.25 / 2)               # full 1/gt, counted in the mean
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[0, 0].item() == 0.0, \
        f"uncovered lane must carry exactly 0 gradient, got {pred.grad[0, 0]}"


def test_normal_loss_l1_over_valid_components():
    from metal_gauss.geometry_loss import normal_loss
    gt = torch.zeros(1, 2, 3); gt[0, 0] = torch.tensor([0, 0, -1.0])       # pixel 1 invalid
    pred = torch.zeros(1, 2, 3)
    pred[0, 0] = torch.tensor([0.1, 0, -1.0]); pred[0, 1] = 5.0
    assert normal_loss(pred, gt).item() == pytest.approx(0.1 / 3)


def test_normal_loss_is_l1_on_components_not_cosine():
    """Brush uses component L1. Cosine would score these two IDENTICALLY (both are 90 deg
    from gt) while L1 separates them, and the gradient shapes differ."""
    from metal_gauss.geometry_loss import normal_loss
    gt = torch.tensor([[[0.0, 0.0, -1.0]]])
    a = normal_loss(torch.tensor([[[1.0, 0.0, 0.0]]]), gt).item()
    b = normal_loss(torch.tensor([[[0.0, -1.0, 0.0]]]), gt).item()
    assert a == pytest.approx(2.0 / 3) and b == pytest.approx(2.0 / 3)
    c = normal_loss(torch.tensor([[[0.6, 0.0, -0.8]]]), gt).item()
    assert c == pytest.approx((0.6 + 0.2) / 3)      # cosine would give 1 - 0.8 = 0.2


def test_depth_normal_loss_gates_on_alpha_and_validity():
    from metal_gauss.geometry_loss import depth_normal_loss
    nd = torch.zeros(1, 3, 3); nr = torch.zeros(1, 3, 3)
    nd[0, 0] = nr[0, 0] = torch.tensor([0, 0, -1.0])              # agree -> 0
    nd[0, 1] = torch.tensor([0, 0, -1.0]); nr[0, 1] = torch.tensor([1.0, 0, 0])  # 90 deg -> 1
    nd[0, 2] = nr[0, 2] = torch.tensor([0, 0, -1.0])              # agree but uncovered
    alpha = torch.tensor([[1.0, 1.0, 0.1]])
    assert depth_normal_loss(nd, nr, alpha).item() == pytest.approx(0.5)


def test_depth_normal_loss_penalises_an_anti_aligned_pair_most():
    """1 - cos, so agreement is 0, perpendicular is 1, and an INVERTED normal is 2. If a
    prior's sign were flipped this term is what would blow up -- it must be able to."""
    from metal_gauss.geometry_loss import depth_normal_loss
    nd = torch.tensor([[[0.0, 0.0, -1.0]]])
    alpha = torch.ones(1, 1)
    assert depth_normal_loss(nd, -nd, alpha).item() == pytest.approx(2.0)
    assert depth_normal_loss(nd, nd, alpha).item() == pytest.approx(0.0)


def test_depth_normal_loss_with_nothing_valid_is_zero_not_nan():
    from metal_gauss.geometry_loss import depth_normal_loss
    l = depth_normal_loss(torch.zeros(2, 2, 3), torch.zeros(2, 2, 3), torch.zeros(2, 2))
    assert l.item() == 0.0 and torch.isfinite(l)


# ------------------------------------------------------- PGSR plane-aux (Task 16 / 10b)

def _plane_feat(H, W, n_cam, d, alpha=1.0):
    """[H,W,5] = n_sum(3) + offset_sum(1) + alpha(1), constant over the frame."""
    f = torch.zeros(H, W, 5, dtype=torch.float64)
    f[..., :3] = torch.tensor(n_cam, dtype=torch.float64)
    f[..., 3] = d
    f[..., 4] = alpha
    return f


def test_plane_depth_recovers_a_fronto_parallel_plane_exactly():
    """The whole point of PGSR: intersect each camera ray with the splat's tangent plane
    instead of taking its centre depth. For a plane at z = 3 with normal (0,0,-1), the
    offset is n.(p) = -3 and every ray must read exactly 3.0 -- unlike centre depth, which
    is constant across a footprint and therefore wrong by +-r*tan(theta) at its ends."""
    from metal_gauss.geometry_loss import plane_depth_from_features
    H = W = 8
    feat = _plane_feat(H, W, (0.0, 0.0, -1.0), -3.0)
    depth, normal, valid = plane_depth_from_features(feat, 4.0, 4.0, W / 2, H / 2)
    assert valid.all()
    assert torch.allclose(depth, torch.full((H, W), 3.0, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(normal[0, 0], torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64))


def test_plane_depth_matches_the_analytic_tilted_plane_and_uses_PIXEL_CENTRE_rays():
    """A tilted plane makes the ray direction matter, so this is where the +0.5 shows.

    Brush uses pixel CENTRES here -- `(u + 0.5 - cx)/fx` -- and INTEGER indices in
    `normals_from_depth`. Both are replicated deliberately: the fixture that pins
    normals_from_depth across languages uses integer indices, and changing either side of a
    cross-language contract to match the other is how they silently diverge. The analytic
    check below fails by ~5% under integer indices at this focal length."""
    from metal_gauss.geometry_loss import plane_depth_from_features
    H = W = 8
    fx = fy = 5.0
    cx, cy = W / 2, H / 2
    n = torch.tensor([0.3, -0.2, -1.0], dtype=torch.float64)
    n = n / n.norm()
    d = -2.5                                            # n . p = d for points on the plane
    feat = _plane_feat(H, W, tuple(n.tolist()), d)
    depth, _, valid = plane_depth_from_features(feat, fx, fy, cx, cy)
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                          torch.arange(W, dtype=torch.float64), indexing="ij")
    ray = torch.stack([(u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy, torch.ones_like(u)], -1)
    want = d / (ray * n).sum(-1)                        # t such that n.(t*ray) = d
    assert valid.all()
    assert torch.allclose(depth, want, atol=1e-12)
    ray_int = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], -1)
    want_int = d / (ray_int * n).sum(-1)
    assert not torch.allclose(depth, want_int, atol=1e-3), \
        "this configuration cannot tell pixel-centre rays from integer ones"


def test_plane_depth_needs_no_alpha_division():
    """Alpha cancels: numerator and denominator are composited with the SAME weights, so
    scaling every channel by a common factor must leave the depth unchanged. Dividing by
    alpha here -- as the centre-depth path must -- would make the depth alpha-dependent."""
    from metal_gauss.geometry_loss import plane_depth_from_features
    H = W = 6
    base = _plane_feat(H, W, (0.2, 0.1, -0.97), -2.0, alpha=1.0)
    scaled = base.clone()
    # 0.6, not 0.37: alpha is ALSO the coverage gate (min_alpha defaults to 0.5), so a
    # factor below it makes the pixel invalid for a reason unrelated to the cancellation.
    scaled[..., :4] *= 0.6                              # partly-covered pixel, same geometry
    scaled[..., 4] = 0.6
    d0, _, v0 = plane_depth_from_features(base, 5.0, 5.0, 3.0, 3.0)
    d1, _, v1 = plane_depth_from_features(scaled, 5.0, 5.0, 3.0, 3.0)
    assert v0.all() and v1.all()
    assert torch.allclose(d0, d1, atol=1e-12)


def test_plane_depth_zeroes_a_pixel_with_any_non_finite_channel():
    """JOINT SANITISATION EXISTS FOR THE BACKWARD, NOT THE FORWARD.

    Sanitise on the joint finite mask, before any division. A NaN in n_x alone would
    otherwise decay into a perfectly plausible axis-aligned plane and be reported VALID.

    Do not "simplify" this to a per-channel mask on the strength of the forward behaviour:
    with a per-channel mask every forward assertion below STILL PASSES, because the
    validity cascade zeroes the bad pixel downstream anyway. Only the gradient check at the
    end can see the difference -- a value masked out AFTER an op reappears as 0 * inf in
    that op's VJP and poisons the map."""
    from metal_gauss.geometry_loss import plane_depth_from_features
    feat = _plane_feat(4, 4, (0.0, 0.0, -1.0), -3.0)
    feat[1, 1, 0] = float("nan")
    feat[2, 2, 3] = float("inf")
    depth, normal, valid = plane_depth_from_features(feat, 4.0, 4.0, 2.0, 2.0)
    for px in ((1, 1), (2, 2)):
        assert valid[px] == 0.0 and depth[px] == 0.0 and (normal[px] == 0).all()
    assert torch.isfinite(depth).all() and torch.isfinite(normal).all()
    assert valid[0, 0] == 1.0

    # THE BACKWARD is why sanitisation must be JOINT and must happen BEFORE any division.
    # The forward alone does not discriminate: with a per-channel mask the NaN survives
    # into depth_raw, and the validity cascade then zeroes the pixel anyway, so every
    # forward assertion above still passes. It is the VJP that breaks -- a value masked
    # out AFTER an op reappears as 0 * inf in that op's gradient and poisons the map.
    f2 = _plane_feat(4, 4, (0.0, 0.0, -1.0), -3.0).requires_grad_(True)
    bad = f2.clone()
    bad = torch.where(torch.zeros_like(bad, dtype=torch.bool), bad, bad)   # keep the graph
    inject = torch.zeros_like(f2)
    inject[1, 1, 0] = float("nan")
    d2, n2, _ = plane_depth_from_features(f2 + inject, 4.0, 4.0, 2.0, 2.0)
    (d2.sum() + n2.sum()).backward()
    assert torch.isfinite(f2.grad).all(), \
        "a non-finite channel poisoned the gradient: sanitise jointly, before the divide"


def test_plane_depth_rejects_grazing_denominators_and_out_of_range_depths():
    from metal_gauss.geometry_loss import plane_depth_from_features
    H = W = 4
    # n = +x, so denom = (u + 0.5 - cx)/fx, which at fx = 50 over a 4-px frame spans only
    # +-0.03. min_denom must exceed that for the gate to bite; at 1e-2 half the frame is
    # still "valid" and this assertion would pass for the wrong reason.
    grazing = _plane_feat(H, W, (1.0, 0.0, 0.0), -3.0)      # n . ray ~ 0 near the axis
    _, _, v = plane_depth_from_features(grazing, 50.0, 50.0, 2.0, 2.0, min_denom=5e-2)
    assert v.sum() == 0
    far = _plane_feat(H, W, (0.0, 0.0, -1.0), -500.0)
    d, _, v2 = plane_depth_from_features(far, 4.0, 4.0, 2.0, 2.0, max_depth=100.0)
    assert v2.sum() == 0 and (d == 0).all()
    near = _plane_feat(H, W, (0.0, 0.0, -1.0), -1e-4)
    _, _, v3 = plane_depth_from_features(near, 4.0, 4.0, 2.0, 2.0, min_depth=1e-3)
    assert v3.sum() == 0


def test_plane_depth_rejects_uncovered_pixels_by_alpha():
    from metal_gauss.geometry_loss import plane_depth_from_features
    feat = _plane_feat(4, 4, (0.0, 0.0, -1.0), -3.0, alpha=0.2)
    _, _, v = plane_depth_from_features(feat, 4.0, 4.0, 2.0, 2.0, min_alpha=0.5)
    assert v.sum() == 0


def test_plane_features_offset_is_the_world_normal_dotted_with_mean_minus_camera():
    """d = n_world . (mean - cam_pos), which is exactly n_cam . p_cam in camera
    coordinates -- a rotation preserves dot products. Both are live so `quats` learns
    through n and `means` through d."""
    from metal_gauss.geometry_loss import plane_features, splat_normals_cam
    torch.manual_seed(0)
    m = torch.randn(6, 3) + torch.tensor([0.0, 0.0, 4.0])
    q = torch.randn(6, 4)
    s = torch.rand(6, 3) * 0.1 + 0.01
    vm = torch.eye(4); vm[:3, 3] = torch.tensor([0.1, -0.2, 0.3])
    n_cam, d = plane_features(m, q, s, vm)
    assert torch.allclose(n_cam, splat_normals_cam(m, q, s, vm))
    p_cam = m @ vm[:3, :3].T + vm[:3, 3]
    assert torch.allclose(d, (n_cam * p_cam).sum(-1), atol=1e-5)


def test_plane_features_gradient_reaches_quats_and_means_but_not_scales():
    from metal_gauss.geometry_loss import plane_features
    torch.manual_seed(1)
    m = (torch.randn(5, 3) + torch.tensor([0.0, 0.0, 4.0])).requires_grad_(True)
    q = torch.randn(5, 4, requires_grad=True)
    s = (torch.rand(5, 3) * 0.1 + 0.01).requires_grad_(True)
    n_cam, d = plane_features(m, q, s, torch.eye(4))
    (n_cam.sum() + d.sum()).backward()
    assert q.grad.abs().sum() > 0 and m.grad.abs().sum() > 0
    assert s.grad is None or s.grad.abs().sum() == 0
