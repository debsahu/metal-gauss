"""`shape_metrics` was NaN-BLIND on MPS, which is the only device it ever runs on.

FOUND 2026-09-06, by an arm rather than by a test. The `--antialias` arm exported a ply
with 31,158 non-finite `scale_*` values across 10,386 of 500,000 splats (2.08%) and 14,824
non-finite opacities -- and the trainer's own per-eval `shape` line for that run reported
`aspect_p50 0.3085`, a perfectly plausible number. Reading the same tensor on CPU returns
`nan`.

    ls = log([[0.001, 0.006, 0.025]] * 100); ls[0,1] = nan; ls[1,2] = inf
    CPU : aspect_p50 nan   smax_p50_mm nan
    MPS : aspect_p50 0.24  smax_p50_mm 25.0

`torch.sort` orders non-finite values differently on the two backends and `median` then
picks a finite element. Training always computes this on MPS, so EVERY `shape` line in
EVERY metal-gauss log is blind to exactly the contamination that CLAUDE.md's Stage 5 says
poisons a SOG codebook and kills the operator's desktop session.

This is the project's own recurring shape: a check that a condition other than the one
being checked can satisfy. `needle_frac` is worse than blind -- `(aspect < 0.1)` is False
for NaN on both devices, so a contaminated splat is silently counted as a HEALTHY one, and
the metric this whole investigation turns on is biased toward the answer we want.
"""
from __future__ import annotations

import math

import pytest
import torch


def _contaminated(n=100):
    ls = torch.log(torch.tensor([[0.001, 0.006, 0.025]] * n))
    ls[0, 1] = float("nan")
    ls[1, 2] = float("inf")
    ls[2, 0] = float("-inf")
    return ls


def test_the_same_tensor_gives_the_same_answer_on_CPU_and_MPS():
    """THE TEST. Everything else here is detail; this is the defect.

    A metric whose value depends on which backend computed it is not a measurement, and
    the backend it runs on in production is the one that lies."""
    if not torch.backends.mps.is_available():
        pytest.skip("needs MPS -- this test exists to compare the two backends")
    from metal_gauss.train import shape_metrics
    ls = _contaminated()
    cpu = shape_metrics(ls)
    mps = shape_metrics(ls.to("mps"))
    for k in cpu:
        a, b = cpu[k], mps[k]
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=1e-6), (
            f"{k}: CPU {a} vs MPS {b}. Non-finite scales are ordered differently by "
            f"torch.sort on the two backends; the statistic must not inherit that.")


def test_the_nonfinite_fraction_is_REPORTED_not_silently_dropped():
    """Dropping the bad rows without saying so would be the same defect in a new place:
    a clean-looking number over a contaminated population. 3 of 100 rows here."""
    from metal_gauss.train import shape_metrics
    m = shape_metrics(_contaminated())
    assert "nonfinite_frac" in m, "the contamination is not reported at all"
    assert m["nonfinite_frac"] == pytest.approx(0.03), m["nonfinite_frac"]
    clean = shape_metrics(torch.log(torch.tensor([[0.001, 0.006, 0.025]] * 100)))
    assert clean["nonfinite_frac"] == 0.0


def test_the_statistics_are_computed_over_the_FINITE_rows_only():
    """And they must equal what the same rows give with the bad ones never present --
    otherwise the mask is doing something other than excluding them."""
    from metal_gauss.train import shape_metrics
    good = torch.log(torch.tensor([[0.001, 0.006, 0.025]] * 97))
    m = shape_metrics(_contaminated())
    ref = shape_metrics(good)
    for k in ("aspect_p50", "smid_p50_mm", "smax_p50_mm"):
        assert m[k] == pytest.approx(ref[k], rel=1e-9), f"{k}: {m[k]} vs {ref[k]}"


def test_a_nonfinite_splat_is_not_counted_as_a_HEALTHY_one():
    """THE BIAS THIS REMOVES. `(aspect < 0.1)` is False for NaN, so before the fix a
    contaminated splat was counted in the DENOMINATOR of `needle_frac` and never in the
    numerator -- it made the trainer look better at exactly the metric under study.

    Here every finite row IS a needle (aspect 0.001/0.025 = 0.04), so a correct
    `needle_frac` is 1.0 over the finite population. Counting the three bad rows as
    healthy would give 0.97, and that 3 pp is the same order as the entire effect N1 and
    N2 were measured to have."""
    from metal_gauss.train import shape_metrics
    ls = torch.log(torch.tensor([[0.001, 0.001, 0.025]] * 100))
    ls[0, 1] = float("nan"); ls[1, 2] = float("inf"); ls[2, 0] = float("-inf")
    m = shape_metrics(ls)
    assert m["needle_frac"] == pytest.approx(1.0), (
        f"needle_frac {m['needle_frac']} -- non-finite rows are being counted as healthy")
    assert m["nonfinite_frac"] == pytest.approx(0.03)


def test_an_entirely_nonfinite_tensor_does_not_crash_and_says_so():
    """The degenerate case must report 1.0 and NaN statistics, not raise and not return a
    confident 0.0 needle fraction over nothing."""
    from metal_gauss.train import shape_metrics
    ls = torch.full((10, 3), float("nan"))
    m = shape_metrics(ls)
    assert m["nonfinite_frac"] == 1.0
    assert math.isnan(m["aspect_p50"]) and math.isnan(m["smid_p50_mm"])
    assert math.isnan(m["needle_frac"]), (
        "a needle fraction over an empty population must be NaN, not 0.0 -- 0.0 is the "
        "best possible score and would be reported by an arm that measured nothing")
