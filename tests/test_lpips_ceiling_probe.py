"""Task 21 step 3: the probe's control arithmetic.

C1 exists so that a null result means "there is no photometric component" and
not "the fitter did not run". These tests make sure C1 itself can tell those
apart -- a control that always passes is worse than no control.
"""
import argparse
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_p = Path(__file__).resolve().parents[1] / "scripts" / "lpips_ceiling_probe.py"
_spec = importlib.util.spec_from_file_location("lpips_ceiling_probe", _p)
CP = importlib.util.module_from_spec(_spec)
sys.modules["lpips_ceiling_probe"] = CP
_spec.loader.exec_module(CP)


def _l2_metric(a, b):
    """A stand-in for LPIPS: any monotone image distance exercises the same
    arithmetic, and using a real LPIPS here would test torchvision, not this."""
    return float(((a - b) ** 2).mean().sqrt())


def _args(**kw):
    d = dict(bilagrid_steps=5, ppisp_steps=5)
    d.update(kw)
    return argparse.Namespace(**d)


def test_c1_reports_a_pass_for_a_fitter_that_actually_inverts_the_distortion(monkeypatch):
    torch.manual_seed(0)
    renders = [torch.rand(16, 20, 3) for _ in range(2)]
    monkeypatch.setattr(CP, "run_fitter",
                        lambda name, bad, clean, args: (list(clean), [{}] * len(bad)))
    r = CP.synthetic_control("affine", renders, _l2_metric, _args(), "cpu")
    assert r["recovered_fraction_mean"] == pytest.approx(1.0, abs=1e-6)
    assert r["passed"] is True


def test_c1_reports_a_FAILURE_for_a_fitter_that_does_nothing(monkeypatch):
    """The whole point. A no-op fitter must not be able to pass C1, or every
    null this probe reports is uninterpretable."""
    torch.manual_seed(1)
    renders = [torch.rand(16, 20, 3) for _ in range(2)]
    monkeypatch.setattr(CP, "run_fitter",
                        lambda name, bad, clean, args: (list(bad), [{}] * len(bad)))
    r = CP.synthetic_control("affine", renders, _l2_metric, _args(), "cpu")
    assert r["recovered_fraction_mean"] == pytest.approx(0.0, abs=1e-6)
    assert r["passed"] is False


def test_c1_fails_a_fitter_that_recovers_most_but_not_enough(monkeypatch):
    """The 0.90 floor must bind somewhere between 0 and 1, or it is decoration."""
    torch.manual_seed(2)
    renders = [torch.rand(16, 20, 3) for _ in range(2)]

    def half(name, bad, clean, args):
        return [0.5 * (b + c) for b, c in zip(bad, clean)], [{}] * len(bad)
    monkeypatch.setattr(CP, "run_fitter", half)
    r = CP.synthetic_control("affine", renders, _l2_metric, _args(), "cpu")
    assert 0.4 < r["recovered_fraction_mean"] < 0.7
    assert r["passed"] is False


def test_c1_injects_a_distortion_large_enough_to_measure(monkeypatch):
    """Discriminating power of the injected fields: if a field were near
    identity, `induced` would be ~0, the fraction would be 0/0, and C1 would
    pass or fail on noise."""
    torch.manual_seed(3)
    renders = [torch.rand(64, 80, 3) * 0.8 + 0.1]
    seen = {}

    def cap(name, bad, clean, args):
        seen[name] = float(((bad[0] - clean[0]) ** 2).mean().sqrt())
        return list(clean), [{}]
    monkeypatch.setattr(CP, "run_fitter", cap)
    for name in ("affine", "bilagrid_tv10", "ppisp"):
        CP.synthetic_control(name, renders, _l2_metric, _args(), "cpu")
        assert seen[name] > 0.01, (name, seen[name])


def test_psnr_matches_the_definition():
    a = torch.zeros(4, 4, 3)
    b = torch.full((4, 4, 3), 0.1)
    assert CP.psnr(a, b) == pytest.approx(20.0, abs=1e-4)
