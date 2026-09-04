"""Task 21 step 5: the decision rule.

The rule's dangerous edge is that a broken fitter and a scene with nothing to
fix both return dLPIPS ~= 0. Every test here is about not confusing them.
"""
import pytest

from bench import lpips_verdict as V


def _res(**fitters):
    return {"stage": "step3_ceiling", "scene": "pgeom", "arm": "R1", "tag": "t",
            "n_views": 25, "baseline": {"lpips_mean": 0.3968, "psnr_mean": 22.6},
            "fitters": {n: {"delta_lpips_mean": d, "delta_psnr_mean": 1.0,
                            "n_params_per_view": 12,
                            "synthetic_control_c1": None if c1 is None else
                            {"passed": c1, "recovered_fraction_mean": 0.95 if c1 else 0.3,
                             "floor": 0.9}}
                        for n, (d, c1) in fitters.items()}}


def test_cut_when_every_usable_fitter_is_under_the_threshold():
    v = V.verdict(_res(affine=(0.004, True), bilagrid_tv10=(0.009, True),
                       ppisp=(0.001, True)))
    assert v["verdict"] == "CUT" and v["ceiling"] == pytest.approx(0.009)


def test_build_when_a_usable_fitter_clears_the_threshold():
    v = V.verdict(_res(affine=(0.004, True), bilagrid_tv10=(0.021, True),
                       ppisp=(0.005, True)))
    assert v["verdict"] == "BUILD" and v["shape"] == "bilagrid"


def test_shape_is_ppisp_when_c_reaches_80_percent_of_b():
    v = V.verdict(_res(affine=(0.004, True), bilagrid_tv10=(0.020, True),
                       ppisp=(0.016, True)))
    assert v["verdict"] == "BUILD" and v["shape"] == "ppisp"


def test_shape_boundary_is_inclusive_at_exactly_80_percent():
    """A rule stated as '>= 80%' must not flip on the boundary it names."""
    assert V.verdict(_res(bilagrid_tv10=(0.020, True),
                          ppisp=(0.016, True)))["shape"] == "ppisp"
    assert V.verdict(_res(bilagrid_tv10=(0.020, True),
                          ppisp=(0.0159, True)))["shape"] == "bilagrid"


def test_a_fitter_that_FAILED_its_c1_control_cannot_produce_a_CUT():
    """THE test. A broken fitter returns dLPIPS ~= 0, which is the same number a
    real null returns. If the only fitters in the rule failed C1 the answer is
    NO VERDICT, never CUT."""
    v = V.verdict(_res(affine=(0.001, False), bilagrid_tv10=(0.002, False),
                       ppisp=(0.000, False)))
    assert v["verdict"] == "NO VERDICT"
    assert "failed" in v["why"] and "CUT" in v["why"]


def test_a_fitter_with_no_c1_control_at_all_is_not_usable():
    v = V.verdict(_res(affine=(0.001, None), bilagrid_tv10=(0.002, None),
                       ppisp=(0.000, None)))
    assert v["verdict"] == "NO VERDICT"


def test_a_failed_fitter_is_excluded_from_the_ceiling_even_when_it_is_the_largest():
    """A broken fitter can also return a LARGE number -- one that reads the
    ground truth, say. It must not be able to carry a BUILD either."""
    v = V.verdict(_res(affine=(0.004, True), bilagrid_tv10=(0.900, False),
                       ppisp=(0.001, True)))
    assert v["verdict"] == "CUT"
    assert v["ceiling"] == pytest.approx(0.004)
    assert "bilagrid_tv10" in v["excluded"]


def test_capacity_overrides_a_build():
    v = V.verdict(_res(affine=(0.004, True), bilagrid_tv10=(0.021, True),
                       ppisp=(0.005, True)), capacity_delta=0.05, capacity_floor=0.001)
    assert v["verdict"] == "DEFER" and v["capacity_grid_ceiling"] == pytest.approx(0.021)


def test_capacity_below_the_grid_ceiling_does_not_defer():
    v = V.verdict(_res(affine=(0.004, True), bilagrid_tv10=(0.021, True),
                       ppisp=(0.005, True)), capacity_delta=0.010)
    assert v["verdict"] == "BUILD"


def test_a_non_rule_fitter_cannot_set_the_ceiling():
    """bilagrid_tv0 is an unregularised upper bound, not a model anyone ships;
    the pre-registration says the rule is applied to (a), (b), (c)."""
    v = V.verdict(_res(affine=(0.004, True), bilagrid_tv10=(0.009, True),
                       ppisp=(0.001, True), bilagrid_tv0=(0.080, True)))
    assert v["verdict"] == "CUT"
    assert v["ceiling_fitter"] == "bilagrid_tv10"
    assert v["rows"]["bilagrid_tv0"]["delta_lpips"] == pytest.approx(0.080)
