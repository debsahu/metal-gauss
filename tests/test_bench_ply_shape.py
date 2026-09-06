"""bench/ply_shape.py -- the shape columns read back from a ply, with no GPU and no scene.

Also the file that holds the TWO tools of this name to each other. `bench/ply_shape.py`
calls `train.shape_metrics`; `scripts/ply_shape.py` re-implements it in numpy over a
hand-parsed ply as a deliberate cross-check. Both docstrings say so, and
`test_the_two_ply_shape_tools_agree_on_the_same_ply` below is what makes the two names safe
to keep -- without it they are simply free to drift.
"""
import sys
from pathlib import Path

import pytest
import torch

from bench.ply_shape import COLUMNS, anchor, log_scales_from_ply, shape_of_ply

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ply_shape import cross_check, shape_from_ply  # noqa: E402


def _write(path, log_scales):
    from metal_gauss.train import export_ply
    n = log_scales.shape[0]
    export_ply({"means": torch.zeros(n, 3), "log_scales": log_scales,
                "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
                "logit_opac": torch.zeros(n),
                "sh_dc": torch.zeros(n, 1, 3), "sh_rest": torch.zeros(n, 15, 3)}, str(path))


def test_log_scales_are_read_back_without_an_exp_log_round_trip(tmp_path):
    """CATCHES the reader that goes through `io.load_ply`, which EXPONENTIATES. The ply
    stores log scales; exponentiating and re-logging is lossy and, worse, a reader that
    forgot the second half would report exp(scale) as a log scale and every shape column
    would be wrong in a plausible-looking way."""
    ls = torch.log(torch.tensor([[0.001, 0.02, 0.02], [1e-6, 1e-4, 0.02]]))
    f = tmp_path / "a.ply"
    _write(f, ls)
    assert torch.allclose(log_scales_from_ply(str(f)), ls, atol=0, rtol=0)


def test_shape_of_ply_matches_shape_metrics_on_the_same_scales(tmp_path):
    """The tool must not be a second implementation of the gate's own statistic."""
    from metal_gauss.train import shape_metrics
    g = torch.Generator().manual_seed(9)
    ls = torch.log(torch.rand(300, 3, generator=g) * 0.03 + 1e-6)
    f = tmp_path / "b.ply"
    _write(f, ls)
    got, want = shape_of_ply(str(f)), shape_metrics(ls)
    assert got["splats"] == 300
    for k in want:
        assert got[k] == pytest.approx(want[k], rel=1e-6, abs=1e-9), k


def test_anchor_refuses_fewer_than_three_arms():
    """CATCHES an anchor built from an n=2 mean -- the exact defect section 8.2 retracted a
    batch of claims over. A Band-1 cumulative check anchored on two runs inherits it."""
    rows = [{"aspect_p50": 0.5}, {"aspect_p50": 0.6}]
    with pytest.raises(RuntimeError, match="n >= 3"):
        anchor(rows, ["aspect_p50"])


def test_anchor_reports_mean_and_spread(tmp_path):
    rows = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    a = anchor(rows, ["x"])
    assert a["x"]["mean"] == pytest.approx(2.0) and a["x"]["spread"] == pytest.approx(2.0)


# A ply the two implementations could DISAGREE on if either were wrong, and the fixture's
# discriminating power is asserted rather than assumed (see the first test below).
#
#   * EVEN count, so the median convention is live. `torch.median` takes the lower of the
#     two middle values and `np.median` averages them; an odd count would hide the
#     difference and let a wrong convention pass.
#   * FIVE populations, not three. The obvious three -- hard needles, soft needles, healthy
#     -- make `needle_frac` and `hard_needle_frac` different numbers, which catches a
#     SWAPPED threshold. They do NOT catch a MOVED one, and that was measured rather than
#     reasoned: with populations at aspect 0.004 / 0.04 / >=0.4, mutating
#     `train.HARD_NEEDLE_ASPECT` from 0.01 to 0.02 changed `hard_needle_frac` by exactly
#     nothing and SURVIVED ALL 716 TESTS IN THIS SUITE, this file's first draft included.
#     So two more populations sit just ABOVE each threshold, at 0.015 and 0.15, and
#     `test_the_fixture_can_actually_SEPARATE_the_two_tools` asserts they do their job.
#   * nothing near either threshold, so the counts are not decided by float32 rounding.
_SMAX = 0.025
#            aspect,  count,  what it is
_POPS = [(0.004, 20),   # hard needle, well below 0.01
         (0.015, 15),   # BETWEEN 0.01 and 0.02 -- sees a moved hard-needle threshold
         (0.040, 25),   # soft needle: a needle, not a hard one
         (0.150, 20),   # BETWEEN 0.1 and 0.2 -- sees a moved needle threshold
         (None,  40)]   # healthy, a spread so the medians are not all ties


def _fixture_log_scales() -> torch.Tensor:
    smid = torch.cat([torch.full((n,), _SMAX * a) if a is not None
                      else torch.linspace(0.010, 0.024, n) for a, n in _POPS])
    smax = torch.full((smid.shape[0],), _SMAX)
    return torch.log(torch.stack([smid * 0.3, smid, smax], dim=1))


# The trainer stores float32 and this file recomputes in float64, so ~1e-7 of relative
# disagreement is arithmetic, not a defect. The smallest REAL disagreement is one
# misclassified splat -- 1/120 here, and 2e-6 at the 500,000 splats these tools are used on.
# `scripts/ply_shape.CHECK_TOL` sits at the same 1e-6 for the same reason.
_AGREE_REL = 1e-6


def test_the_fixture_can_actually_SEPARATE_the_two_tools(tmp_path):
    """Assert the discriminating power directly, so a future edit to the fixture cannot
    quietly turn the agreement test into a tautology.

    Three properties have to hold or the agreement below proves nothing:
      1. the two needle columns are different numbers, both strictly inside (0, 1) --
         otherwise a swapped `needle`/`hard_needle` threshold is invisible;
      2. the WRONG median convention gives a MEASURABLY different answer -- otherwise
         `np.median` in place of `torch.median` sails through;
      3. the count is even, which is what makes (2) possible at all.
    """
    ls = _fixture_log_scales()
    n = ls.shape[0]
    assert n % 2 == 0, "an odd count makes the two median conventions agree"
    from metal_gauss.train import shape_metrics
    m = shape_metrics(ls)
    assert 0.0 < m["hard_needle_frac"] < m["needle_frac"] < 1.0, m
    # A MOVED threshold must change the answer, not only a swapped one. Measured against
    # the actual mutant: 0.01 -> 0.02 survived the whole suite before these populations
    # existed. `aspect` here is smid/smax with smax constant, so it is exact.
    aspect = torch.exp(ls).sort(dim=-1).values
    aspect = aspect[:, 1] / aspect[:, 2]
    for col, lo, hi in (("hard_needle_frac", 0.01, 0.02), ("needle_frac", 0.1, 0.2)):
        between = int(((aspect >= lo) & (aspect < hi)).sum())
        assert between > 0, (
            f"no splat with aspect in [{lo}, {hi}): {col} cannot see a threshold moved "
            f"from {lo} to {hi}, which is a mutant this suite has already let through")
    # the wrong convention must be separable by more than the tolerance the test uses
    _write(tmp_path / "f.ply", ls)
    avg = shape_from_ply(tmp_path / "f.ply", "average")
    moved = [k for k in ("aspect_p50", "smid_p50_mm")
             if abs(m[k] - avg[k]) > _AGREE_REL * max(1.0, abs(m[k])) * 100]
    assert moved, ("no column separates `lower` from `average` on this fixture, so the "
                   "agreement test cannot see a median-convention error")


def test_the_two_ply_shape_tools_agree_on_the_same_ply(tmp_path):
    """`bench/ply_shape.py` (calls `train.shape_metrics`, torch, float32) against
    `scripts/ply_shape.py` (numpy re-implementation over a hand-parsed ply, float64), on
    one ply, over every column they share.

    THIS IS WHAT MAKES KEEPING BOTH SAFE. The two exist for opposed reasons -- one must not
    reimplement the statistic, the other must -- and the only thing standing between that
    and two numbers with the same name meaning different things is a test that computes
    both. It would catch a divergence in the field order, the median convention, either
    needle threshold, the mm conversion, or the smid/smax lane choice.
    """
    ls = _fixture_log_scales()
    f = tmp_path / "both.ply"
    _write(f, ls)
    a = shape_of_ply(str(f))                 # via train.shape_metrics
    b = shape_from_ply(f, "lower")           # independent numpy
    assert a["splats"] == b["n_splats"] == ls.shape[0]
    assert set(COLUMNS) <= set(a) and set(COLUMNS) <= set(b)
    for k in COLUMNS:
        assert a[k] == pytest.approx(b[k], rel=_AGREE_REL, abs=0.0), (
            f"{k}: bench={a[k]!r} scripts={b[k]!r}")


def test_the_numpy_tool_is_NaN_BLIND_and_the_cross_check_REFUSES_rather_than_agreeing(
        tmp_path):
    """The ONE place the two legitimately diverge, pinned so nobody discovers it as a
    mystery. `shape_metrics` excludes non-finite rows and reports `nonfinite_frac`;
    `scripts/ply_shape.shape_from_ply` does not, because a cross-check that silently
    adopted the reference's own filtering would stop being independent.

    On a contaminated ply that divergence is real and it is in the direction that flatters
    the trainer: a `nan` scale is never `< 0.1` so it sits in `needle_frac`'s DENOMINATOR
    only, and a `+inf` smax makes `smid/smax` exactly 0, which lands in the NUMERATOR as a
    needle that does not exist.

    The right behaviour is not to fix the numpy side -- it is for `cross_check` to REFUSE
    TO WRITE, which is the loud failure the blind statistic never gave. That is what this
    asserts.
    """
    import json
    ls = _fixture_log_scales().clone()
    ls[3, 1] = float("nan")            # a hard needle, lost to the numerator on both sides
    ls[7, 2] = float("inf")            # a hard needle, INVENTED as a needle by the numpy side
    f = tmp_path / "bad.ply"
    _write(f, ls)

    from metal_gauss.train import shape_metrics
    ref = shape_metrics(ls)
    assert ref["nonfinite_frac"] == pytest.approx(2 / ls.shape[0])
    got = shape_from_ply(f, "lower")
    # the divergence must be REAL, or the refusal below is not evidence of anything
    assert abs(ref["needle_frac"] - got["needle_frac"]) > 1e-4, (ref, got)

    rep = tmp_path / "r.json"
    rep.write_text(json.dumps({"metrics": {"shape": ref}}))
    with pytest.raises(SystemExit) as e:
        cross_check(got, rep, None, f)
    assert "needle_frac" in str(e.value)

    # ... and it does NOT refuse on the clean ply, or the refusal is just a broken check
    clean = tmp_path / "ok.ply"
    _write(clean, _fixture_log_scales())
    rep2 = tmp_path / "r2.json"
    rep2.write_text(json.dumps({"metrics": {"shape": shape_metrics(_fixture_log_scales())}}))
    checked = cross_check(shape_from_ply(clean, "lower"), rep2, None, clean)
    assert {"aspect_p50", "needle_frac"} <= set(checked), checked


def test_the_hard_needle_threshold_is_ONE_number_in_two_files_and_matches_its_derivation():
    """`HARD_NEEDLE_ASPECT` is written out independently in `metal_gauss/train.py` and in
    `scripts/ply_shape.py`, because the second is a deliberate re-implementation of the
    first. Two literals of the same constant drift; this is what stops them.

    It also re-derives the number rather than taking 0.01 on trust, because 0.01 looks
    exactly like a taste threshold and is not one. splat-transform's SOG writer stores the
    quaternion's smallest three components as `255 * (q * 0.5 + 0.5)` in uint8 after
    scaling by +-sqrt(2), so:

        step        = sqrt(2) / 255                 one uint8 level, in component units
        per-comp    = step / 2                      worst-case round-to-nearest
        |dq|        = (step / 2) * sqrt(3)          over three components
        rotation    = 2 * |dq|                      a quaternion perturbation of norm e
                                                    is a rotation of 2e
                    = 0.0096058 rad

    Below that aspect a splat's minor in-plane half-axis is smaller than the rim
    displacement its own quantised orientation produces. The constant is that number
    rounded UP to two significant figures -- so it must sit at or above the derivation and
    comfortably below twice it, which is what pins it to 0.01 rather than to 0.02.
    """
    import math
    from metal_gauss.train import HARD_NEEDLE_ASPECT as trainer
    from ply_shape import HARD_NEEDLE_ASPECT as crosscheck
    assert trainer == crosscheck, (
        f"the trainer says {trainer} and the ply cross-check says {crosscheck}; a column "
        f"named hard_needle_frac would then mean two different things")
    derived = 2.0 * (math.sqrt(2.0) / 255.0 / 2.0) * math.sqrt(3.0)
    assert derived == pytest.approx(0.0096058, abs=5e-8), derived
    assert derived <= trainer < 2.0 * derived, (
        f"{trainer} is not the delivery-derived threshold {derived:.7f} rounded up; it is "
        f"either below the quantisation floor or a different number entirely")
