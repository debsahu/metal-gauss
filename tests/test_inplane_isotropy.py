"""The in-plane isotropy barrier: a hinge on log(smax/smid).

WHY IT EXISTS. metal-gauss produces 3-10x more needle-shaped splats than every other
trainer on the same scenes -- needle_frac = frac(smid/smax < 0.1) is 16.6% on
playroom_0821 against Brush's 0.55-4.5% and LFS's 0.15%. Flatten is exonerated: it
collapses smin and leaves smid/smax untouched (0.839 -> 0.839 across three
single-variable pairs), so the slivers exist at flatten 0. Nothing in the trainer's
objective mentions the IN-PLANE aspect ratio at all, and this term is that omission.

WHY THE HINGE ON smid/smax AND NOT THE PUBLISHED ERANK BARRIER. Measured on the real
B0a ply (500,000 splats), shrinking smin ALONE -- which is exactly what flatten does --
moves `max(-log(erank - 1 + eps), 0)` from 1.3936 to 1.5273 (+9.6%) while this hinge is
invariant to 6 decimal places at 0.785225. The erank barrier is a function of all three
axes, so its value and therefore its effective weight depend on how hard flatten is
pulling; this one is orthogonal to flatten BY CONSTRUCTION. That property is pinned by
`test_gradient_on_the_smin_lane_is_exactly_zero` below, which is the whole reason the
term can be added to a recipe that already contains flatten.
"""
from __future__ import annotations

import math

import pytest
import torch


def _ls(rows):
    return torch.log(torch.tensor(rows, dtype=torch.float64))


LOG2 = math.log(2.0)


# ------------------------------------------------------------------ arithmetic

def test_barrier_is_the_hinged_log_ratio_of_the_two_IN_PLANE_axes():
    """The value, from first principles, on rows whose answer can be written by hand.

    CATCHES: any change to which axes the term reads, and any change to the mean/sum
    convention. Row 0 is isotropic in plane (pays 0), row 1 sits exactly on the knee
    (pays 0), row 2 is a 10:1 needle."""
    from metal_gauss.geometry_loss import inplane_isotropy_loss
    ls = _ls([[0.001, 0.020, 0.020],     # smax/smid = 1     -> 0
              [0.001, 0.010, 0.020],     # smax/smid = 2     -> 0 (on the knee)
              [0.001, 0.002, 0.020]])    # smax/smid = 10    -> log(10) - log(2)
    want = (0.0 + 0.0 + (math.log(10.0) - LOG2)) / 3.0
    assert inplane_isotropy_loss(ls, LOG2).item() == pytest.approx(want, rel=1e-12)


def test_a_batch_of_DISCS_pays_EXACTLY_zero():
    """THE HINGE. Without relu the term is log(smax/smid) - log(r0), which is NEGATIVE for
    every disc, so minimising it drives aspect -> 1 on splats that were already fine and
    spends photometric quality doing it. The pre-registered signature of that mutant is
    aspect_p50 -> 1 and PSNR -0.3 dB; this test catches it at the tensor, for free.

    Exactly zero, not approximately: relu(x) for x < 0 is 0.0, and a `softplus` or a
    `clamp_min(eps)` substituted for it would not be."""
    from metal_gauss.geometry_loss import inplane_isotropy_loss
    discs = _ls([[0.0001, 0.020, 0.020], [0.003, 0.011, 0.020], [0.001, 0.020, 0.021]])
    v = inplane_isotropy_loss(discs, LOG2)
    assert v.item() == 0.0
    # ... and it must still be differentiable, with a zero gradient, not a detached 0.
    d = discs.clone().requires_grad_(True)
    inplane_isotropy_loss(d, LOG2).backward()
    assert d.grad is not None and torch.count_nonzero(d.grad) == 0


def test_it_is_DIMENSIONLESS_and_must_not_be_divided_by_a_metric_scale():
    """CATCHES mutant (d): weight divided by metric_scale as flatten's is in Brush.

    flatten is a LENGTH in metres, so a metric normalisation is at least arguable there
    (CLAUDE.md records it as a measurable dilution, +1.1-1.4 deg of thin-axis). This term
    is a log RATIO of two lengths. Scaling a whole scene by k leaves it invariant to float roundoff, so
    a per-scene divisor would make the same flag mean different things on two scenes for
    no reason at all -- the pre-registered signature is 'effect differs between scenes by
    their scale ratio'."""
    from metal_gauss.geometry_loss import inplane_isotropy_loss
    ls = _ls([[0.001, 0.002, 0.020], [0.004, 0.030, 0.410]])
    base = inplane_isotropy_loss(ls, LOG2).item()
    assert base > 0.0, "the fixture must PAY, or invariance under scaling is vacuous"
    for k in (1e-4, 0.37, 1.0, 5.0, 1e4):
        got = inplane_isotropy_loss(ls + math.log(k), LOG2).item()
        # rel, not exact: adding log(k) to all three lanes is not a bit-exact shift of
        # their differences in floating point. A metric-scale divisor -- the mutant this
        # test exists for -- would move the value by a factor of k, which is 1e8 across
        # this sweep, so 1e-12 separates the two by twenty orders of magnitude.
        assert got == pytest.approx(base, rel=1e-12), f"scaling by {k} moved the term"


def test_it_is_a_MEAN_so_the_weight_does_not_depend_on_budget():
    from metal_gauss.geometry_loss import inplane_isotropy_loss
    ls = _ls([[0.001, 0.002, 0.020]])
    assert inplane_isotropy_loss(ls.repeat(13, 1), LOG2).item() == pytest.approx(
        inplane_isotropy_loss(ls, LOG2).item(), rel=1e-12)


def test_the_ratio_argument_moves_the_knee():
    """r0 is the only tuning knob. A term that ignored it would look correct on every
    other test here, because every other test uses r0 = 2."""
    from metal_gauss.geometry_loss import inplane_isotropy_loss
    ls = _ls([[0.001, 0.005, 0.020]])              # smax/smid = 4
    assert inplane_isotropy_loss(ls, math.log(4.0)).item() == 0.0
    assert inplane_isotropy_loss(ls, math.log(8.0)).item() == 0.0
    assert inplane_isotropy_loss(ls, math.log(2.0)).item() == pytest.approx(
        math.log(4.0) - LOG2, rel=1e-12)
    assert inplane_isotropy_loss(ls, 0.0).item() == pytest.approx(math.log(4.0), rel=1e-12)


# ------------------------------------------------------------------ gradients

def test_gradient_on_the_smin_lane_is_exactly_zero():
    """THE NON-INTERFERENCE REQUIREMENT, and the one this term's whole design rests on.

    flatten acts on smin and nothing else. If this barrier touched the smin lane the two
    would compete for one degree of freedom and neither weight could be tuned
    independently -- which is precisely why the erank form was rejected (see the module
    docstring: shrinking smin alone moves the erank barrier +9.6% on the real ply).

    CATCHES mutant (a), the barrier written on smin/smax instead of smid/smax. Its
    pre-registered signature at 30k is 'smin rises, thin-axis worsens, needles unchanged';
    here it is one exactly-nonzero float, at step 0, for nothing.

    EXACT zero, and it must hold for every row whatever the stored axis order, so the
    lane is located per row by argmin rather than assumed to be column 0."""
    from metal_gauss.geometry_loss import inplane_isotropy_loss
    rows = [[0.0010, 0.0020, 0.0200],      # needle, smin first
            [0.0200, 0.0010, 0.0020],      # same splat, axes rotated
            [0.0020, 0.0200, 0.0010],      # and again
            [0.0001, 0.0200, 0.0200]]      # a disc: nothing may move at all
    ls = _ls(rows).requires_grad_(True)
    inplane_isotropy_loss(ls, LOG2).backward()
    smin_lane = ls.detach().argmin(dim=1)
    got = ls.grad[torch.arange(len(rows)), smin_lane]
    assert torch.count_nonzero(got) == 0, (
        f"the barrier leaked gradient onto the smallest axis: {got.tolist()}. That lane "
        f"belongs to flatten; sharing it makes the two weights uninterpretable.")


def test_gradient_pushes_smid_UP_and_smax_DOWN():
    """Sign check. Reversed, the term MANUFACTURES needles while its logged value falls,
    which is the failure mode nothing else in the battery would name."""
    from metal_gauss.geometry_loss import inplane_isotropy_loss
    ls = _ls([[0.001, 0.002, 0.020]]).requires_grad_(True)
    inplane_isotropy_loss(ls, LOG2).backward()
    g = ls.grad[0]
    assert g[1] < 0, "descent must GROW smid"      # d/d(log smid) < 0
    assert g[2] > 0, "descent must SHRINK smax"    # d/d(log smax) > 0
    # log-space hinge: the magnitudes are exactly 1/N, independent of the scales
    assert g[1].item() == pytest.approx(-1.0, rel=1e-12)
    assert g[2].item() == pytest.approx(+1.0, rel=1e-12)


def test_it_reads_the_SORTED_axes_not_the_stored_column_order():
    """CATCHES mutant (b), the barrier applied to unsorted scales. log_scales columns
    carry no ordering convention -- a splat's largest axis is column 0 about a third of
    the time -- so an unsorted term would act correctly on only ~1/3 of splats and its
    pre-registered signature is 'aspect improves on ~1/3 of splats only'.

    Here it is exact: the value and the per-row gradient magnitudes must be invariant
    under every one of the 6 column permutations of every row."""
    from metal_gauss.geometry_loss import inplane_isotropy_loss
    import itertools
    # BOTH rows must be over the knee, or a row contributes a zero gradient and the
    # permutation assertion below is satisfied by a term that does nothing. The first
    # draft of this fixture had a second row at smax/smid = 1.15, which pays nothing.
    base_rows = [[0.0007, 0.0030, 0.0400], [0.0100, 0.0130, 0.0900]]
    for r in base_rows:
        assert max(r) / sorted(r)[1] > 2.0, f"fixture row {r} does not reach the hinge"
    ref = inplane_isotropy_loss(_ls(base_rows), LOG2).item()
    for perm in itertools.permutations(range(3)):
        rows = [[r[i] for i in perm] for r in base_rows]
        ls = _ls(rows).requires_grad_(True)
        v = inplane_isotropy_loss(ls, LOG2)
        assert v.item() == pytest.approx(ref, rel=1e-12), f"value changed under {perm}"
        v.backward()
        # sorted-by-magnitude gradient per row is permutation invariant
        for r in range(len(rows)):
            order = ls.detach()[r].argsort()
            assert ls.grad[r][order].tolist() == pytest.approx(
                [0.0, -0.5, 0.5], abs=1e-12), f"gradient changed under {perm}"


def test_a_disc_and_a_needle_of_the_SAME_volume_are_scored_differently():
    """DISCRIMINATING POWER OF THE FIXTURE ITSELF. A term that looked only at total size,
    or at smin, would score these two identically -- and every other test here would
    still pass, because none of them contains a pair that separates 'thin' from 'thready'.
    That is the exact failure this project has shipped before: a fixture whose family
    made both rules agree."""
    from metal_gauss.geometry_loss import inplane_isotropy_loss, flatten_loss
    disc = _ls([[0.002, 0.020, 0.020]])
    needle = _ls([[0.002, 0.002, 0.200]])          # 10x the extent, same smin
    assert flatten_loss(disc).item() == pytest.approx(flatten_loss(needle).item()), (
        "the fixture is not discriminating: flatten already separates these two, so a "
        "pass here would not be evidence about the new term")
    assert inplane_isotropy_loss(disc, LOG2).item() == 0.0
    assert inplane_isotropy_loss(needle, LOG2).item() > 1.0


# ------------------------------------------------------------------ wiring

def _tiny_colmap_scene(tmp_path):
    """The same 3-view 32px scene `test_flatten_flag_actually_reaches_the_training_loss`
    builds. Duplicated rather than shared because a fixture two wiring tests depend on is
    a fixture that can silently stop testing either of them."""
    import numpy as np
    from PIL import Image
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


def _aspect_p50_after(tmp_path, weight, out, ratio=1.0, steps=60):
    import numpy as np
    import plyfile
    from metal_gauss.train import build_parser, train
    a = build_parser().parse_args([
        "--colmap", str(tmp_path / "sparse"), "--images", str(tmp_path / "images"),
        "--steps", str(steps), "--budget", "400", "--max-resolution", "32",
        "--eval-every", str(steps), "--eval-split-every", "1000", "--seed", "0",
        "--num-downscales", "0", "--no-grow", "--sh-warmup", "0",
        "--inplane-isotropy-weight", str(weight),
        "--inplane-isotropy-ratio", str(ratio), "--export", str(out)])
    a.resolution_schedule = max(1, a.steps // 3)
    train(a)
    v = plyfile.PlyData.read(str(out))["vertex"]
    s = np.sort(np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)), axis=1)
    return float(np.median(s[:, 1] / np.maximum(s[:, 2], 1e-12)))


def test_inplane_flag_actually_reaches_the_training_loss(tmp_path):
    """WIRING, not arithmetic. This repo's log is full of flags that parse and do nothing
    -- LFS's `--train` is a no-op, its `--init=` is dead -- and a unit test on the loss
    function cannot see any of that.

    THE BAR IS SET FROM A MEASURED NULL, not guessed. The flatten wiring test asserts a
    >= 10%% drop and reads 10.51%% on the pinned torch and 9.38%% on the documented one --
    half a point of margin either side, and raising the weight cannot restore it because
    the effect saturates. So this test measures the weight-0 repeat spread first and
    requires the treated arm to beat it by 10x. If the null is not tight the test says so
    instead of quietly resting on it.

    A STEP-1 LOSS-ARITHMETIC PROBE, which is the stronger instrument used for flatten,
    IS NOT AVAILABLE HERE and it is worth saying why: the seed is isotropic, so the hinge
    is inactive at step 1, the term is exactly 0, and Dloss = w * 0 for every w. A probe
    that passes at every weight including a broken one is not a probe."""
    pytest.importorskip("pycolmap")
    if not torch.backends.mps.is_available():
        pytest.skip("needs MPS")
    _tiny_colmap_scene(tmp_path)
    off_a = _aspect_p50_after(tmp_path, 0.0, tmp_path / "off_a.ply")
    off_b = _aspect_p50_after(tmp_path, 0.0, tmp_path / "off_b.ply")
    null = abs(off_a - off_b)
    on = _aspect_p50_after(tmp_path, 50.0, tmp_path / "on.ply")
    effect = on - off_a
    assert effect > 0, (
        f"the barrier made splats LESS isotropic: aspect_p50 {off_a:.6f} -> {on:.6f}")
    assert effect > 10 * max(null, 1e-9), (
        f"aspect_p50 {off_a:.6f} -> {on:.6f} (effect {effect:.6f}) against a weight-0 "
        f"repeat null of {null:.2e}: the flag did not reach the loss, or its effect is "
        f"inside this scene's own noise")


def test_the_ratio_flag_reaches_the_training_loss_too(tmp_path):
    """A weight that works with a ratio that is ignored would pass every other test here:
    all of them would simply be running at r0 = 2. r0 = 1e6 puts every splat under the
    knee, so a correctly wired ratio makes the SAME weight inert."""
    pytest.importorskip("pycolmap")
    if not torch.backends.mps.is_available():
        pytest.skip("needs MPS")
    _tiny_colmap_scene(tmp_path)
    off = _aspect_p50_after(tmp_path, 0.0, tmp_path / "off.ply")
    inert = _aspect_p50_after(tmp_path, 50.0, tmp_path / "inert.ply", ratio=1e6)
    live = _aspect_p50_after(tmp_path, 50.0, tmp_path / "live.ply", ratio=1.0)
    assert abs(inert - off) < 0.1 * abs(live - off), (
        f"r0=1e6 should make the term inert but moved aspect_p50 {off:.6f} -> "
        f"{inert:.6f}, against {live:.6f} at r0=1: the ratio flag is not reaching relu()")


def _step1(tmp_path, weight, tag, ratio=1e-9):
    """Total loss and logged term after exactly ONE step.

    At step 1 the splat state is exactly the seed in every arm, so the photometric terms
    are bit-identical and the only admissible difference in total loss is w * term. This
    is the probe research/metal-gauss.md section 8.0a used to prove flatten is not inert,
    and it resolves to 4.4e-16.

    `ratio` defaults to 1e-9 ON PURPOSE. The seed is isotropic, so at the operational
    r0 = 2 the hinge is shut, the term is exactly 0, and Dloss = w * 0 holds for a
    correctly wired term AND for a completely disconnected one. Pushing the knee below
    every splat makes the term nonzero at step 1 and the probe able to fail."""
    import json
    import re
    from metal_gauss.train import build_parser, train
    rep = tmp_path / f"{tag}.json"
    a = build_parser().parse_args([
        "--colmap", str(tmp_path / "sparse"), "--images", str(tmp_path / "images"),
        "--steps", "1", "--budget", "400", "--max-resolution", "32",
        "--eval-every", "1", "--eval-split-every", "1000", "--seed", "0",
        "--num-downscales", "0", "--no-grow", "--sh-warmup", "0",
        "--inplane-isotropy-weight", str(weight),
        "--inplane-isotropy-ratio", str(ratio), "--report", str(rep)])
    a.resolution_schedule = 1
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train(a)
    m = re.search(r"step\s+1\s+loss\s+([0-9.]+)", buf.getvalue())
    assert m, "the trainer did not print a step-1 loss line:\n" + buf.getvalue()[-800:]
    terms = json.loads(rep.read_text())["log"][0]["terms"]
    return float(m.group(1)), terms.get("inplane")


def test_the_weight_enters_the_total_EXACTLY_ONCE_and_UNDIVIDED(tmp_path):
    """LOSS ARITHMETIC, the strongest instrument available for a loss term.

    CATCHES THREE THINGS AT ONCE.
      * a double-add -- the defect that doubled every flatten run in this project until
        2026-09-02 -- which reads as a residual of exactly w * term, not zero;
      * a term that never reaches `loss` at all, which reads as Dloss = 0;
      * mutant (d), the weight divided by a metric scale as Brush divides flatten's.
        A per-scene divisor makes the residual (1 - 1/scale) * w * term, and P-GEOM's
        scene extent is 1.82, so it would miss by ~45%% of the term. No unit test on the
        loss FUNCTION can see that, because the divisor lives at the call site.

    The residual bound is set by the trainer's own print precision (%.4f on two lines),
    not chosen: 1e-4 is what two roundings can hide."""
    pytest.importorskip("pycolmap")
    if not torch.backends.mps.is_available():
        pytest.skip("needs MPS")
    _tiny_colmap_scene(tmp_path)
    base, _ = _step1(tmp_path, 0.0, "w0")
    for w in (1.0, 10.0):
        loss, term = _step1(tmp_path, w, f"w{w:g}")
        assert term is not None and term > 0.0, (
            f"the term logged {term} at w={w}; with r0=1e-9 every splat is over the knee, "
            f"so a zero here means the probe is vacuous, not that the term is fine")
        residual = (loss - base) - w * term
        assert abs(residual) < 1e-4, (
            f"w={w}: total loss moved {loss - base:.6f} but w*term is {w * term:.6f} "
            f"(residual {residual:.3e}). Ratio {(loss - base) / (w * term):.6f} -- 2.0 is "
            f"a double-add, 0.0 is a flag that never reaches the loss, and 1/scene_scale "
            f"is a metric normalisation this dimensionless term must not have.")
