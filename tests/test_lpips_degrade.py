"""Task 21: the LPIPS calibration curve's degradations."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_p = Path(__file__).resolve().parents[1] / "bench" / "lpips_degrade.py"
_spec = importlib.util.spec_from_file_location("lpips_degrade", _p)
DG = importlib.util.module_from_spec(_spec)
sys.modules["lpips_degrade"] = DG
_spec.loader.exec_module(DG)


def test_blur_at_sigma_zero_is_the_identity():
    x = torch.rand(9, 11, 3)
    assert torch.equal(DG.gaussian_blur(x, 0.0), x)


def test_blur_preserves_shape_and_mean_and_reduces_variance():
    """A blur that changed the mean would be a brightness change dressed as a
    blur, and LPIPS would be reading the wrong degradation."""
    torch.manual_seed(0)
    x = torch.rand(40, 52, 3)
    y = DG.gaussian_blur(x, 2.0)
    assert y.shape == x.shape
    assert float(y.mean()) == pytest.approx(float(x.mean()), abs=2e-3)
    assert float(y.var()) < 0.35 * float(x.var())


def test_blur_kernel_is_normalised_so_a_flat_field_is_unchanged():
    flat = torch.full((20, 24, 3), 0.42)
    assert torch.allclose(DG.gaussian_blur(flat, 3.0), flat, atol=1e-6)


def test_blur_is_monotone_in_sigma_on_a_real_signal():
    torch.manual_seed(1)
    x = torch.rand(48, 60, 3)
    v = [float(((DG.gaussian_blur(x, s) - x) ** 2).mean()) for s in (0.5, 1, 2, 4)]
    assert v == sorted(v), v


def test_resample_at_k_1_is_the_identity():
    """DOCUMENTATION, NOT A DISCRIMINATING TEST, and it is labelled so nobody
    counts it as coverage: at k = 1 every step downstream of the guard is
    already the identity (box_downscale returns the image, repeat_interleave(1)
    is a no-op), so deleting the guard changes nothing. The mutant for it was
    dropped from the battery after being proven inert rather than left in to
    inflate a kill count."""
    x = torch.rand(9, 11, 3)
    assert torch.equal(DG.resample(x, 1), x)


def test_resample_restores_the_original_size_exactly():
    """A size change would confound the degradation with LPIPS's own scale
    dependence, which is a separate measurement."""
    for hw in [(48, 60), (49, 61)]:
        x = torch.rand(*hw, 3)
        assert DG.resample(x, 4).shape == x.shape


def test_resample_makes_kxk_blocks_constant():
    torch.manual_seed(2)
    x = torch.rand(40, 60, 3)
    y = DG.resample(x, 4)
    blk = y[:4, :4, 0]
    assert float(blk.max() - blk.min()) < 1e-6
    assert float(blk[0, 0]) == pytest.approx(float(x[:4, :4, 0].mean()), abs=1e-6)


def test_resample_is_monotone_in_the_factor():
    torch.manual_seed(3)
    x = torch.rand(48, 48, 3)
    v = [float(((DG.resample(x, k) - x) ** 2).mean()) for k in (2, 3, 4, 6)]
    assert v == sorted(v), v
