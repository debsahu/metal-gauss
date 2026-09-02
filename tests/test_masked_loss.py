"""Masked photometric loss, masked held-out PSNR, and the `mean coverage` instrument.

The whole point of the coverage line: `mean coverage 100.0%` on a run that supplied
masks means the masks never reached the loss. That is how splat_3840v2_full reported
49.561 dB against a 32.8 dB sibling and burned 47,037 s of GPU (CLAUDE.md Stage 3).
"""
import math

import numpy as np
import pytest
import torch

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")

# SSIM's gaussian window is 11 taps, so a pixel's value depends on the 5 rows either
# side of it. Any mask boundary bleeds that far -- masking cannot be exact at the
# boundary however it is normalised (CLAUDE.md Stage 4). Tests that want an exact
# zero must keep wrecked pixels this far inside the dropped region.
SSIM_RADIUS = 5


@mps
def test_photometric_loss_ignores_dropped_pixels():
    """Wreck a band, drop a strictly larger band around it with >= SSIM_RADIUS margin,
    and the loss must be exactly zero -- no L1 contribution and no SSIM window that
    both sees a wrecked pixel and is centred on a kept one."""
    from metal_gauss.train import _gaussian_kernel, photometric_loss
    torch.manual_seed(0)
    gt = torch.rand(48, 64, 3, device="mps")
    pred = gt.clone()
    pred[16:32] = 1.0 - pred[16:32]                        # wreck a middle band
    mask = torch.ones(48, 64, device="mps")
    mask[16 - SSIM_RADIUS:32 + SSIM_RADIUS] = 0            # drop it, with margin
    k = _gaussian_kernel(device="mps")
    assert photometric_loss(pred, gt, mask, k).item() == pytest.approx(0.0, abs=1e-6)
    assert photometric_loss(pred, gt, None, k).item() > 0.1


@mps
def test_photometric_loss_masks_with_keep_not_drop():
    """The one-character error that costs a run: multiplying by DROP trains ONLY the
    operator and the monopod. Inverting the mask here must make the loss LARGE, not
    small -- so an implementation that used (1 - mask) fails loudly."""
    from metal_gauss.train import _gaussian_kernel, photometric_loss
    torch.manual_seed(0)
    gt = torch.rand(48, 64, 3, device="mps")
    pred = gt.clone()
    pred[16:32] = 1.0 - pred[16:32]
    keep = torch.ones(48, 64, device="mps")
    keep[16 - SSIM_RADIUS:32 + SSIM_RADIUS] = 0
    k = _gaussian_kernel(device="mps")
    assert photometric_loss(pred, gt, keep, k).item() < 1e-6
    assert photometric_loss(pred, gt, 1.0 - keep, k).item() > 0.05


@mps
def test_photometric_loss_all_ones_mask_equals_unmasked():
    from metal_gauss.train import _gaussian_kernel, photometric_loss
    torch.manual_seed(1)
    gt, pred = torch.rand(32, 32, 3, device="mps"), torch.rand(32, 32, 3, device="mps")
    k = _gaussian_kernel(device="mps")
    a = photometric_loss(pred, gt, torch.ones(32, 32, device="mps"), k).item()
    b = photometric_loss(pred, gt, None, k).item()
    assert a == pytest.approx(b, rel=1e-6)


@mps
def test_unmasked_photometric_loss_reproduces_the_pre_existing_formula():
    """Regression: the refactor must not change what an unmasked run optimises.
    0.8 L1 + 0.2 (1 - SSIM), meaned, exactly as train.py computed it at 280843a.
    A non-square image is deliberate -- it catches a mask/SSIM-map broadcast that
    only works when H == W."""
    from metal_gauss.train import _gaussian_kernel, photometric_loss, ssim
    torch.manual_seed(2)
    gt, pred = torch.rand(48, 64, 3, device="mps"), torch.rand(48, 64, 3, device="mps")
    k = _gaussian_kernel(device="mps")
    legacy = (0.8 * (pred - gt).abs().mean() + 0.2 * (1.0 - ssim(pred, gt, k))).item()
    assert photometric_loss(pred, gt, None, k).item() == pytest.approx(legacy, rel=1e-6)
    # and the masked path must agree with it too when nothing is dropped
    ones = torch.ones(48, 64, device="mps")
    assert photometric_loss(pred, gt, ones, k).item() == pytest.approx(legacy, rel=1e-6)


@mps
def test_photometric_loss_masks_the_ssim_term_not_only_l1():
    """If only L1 were masked, the SSIM half would still see the wrecked band and the
    loss would sit near 0.2 * (1 - SSIM) rather than at zero. The margin construction
    makes that difference the whole signal."""
    from metal_gauss.train import _gaussian_kernel, photometric_loss, ssim_map
    torch.manual_seed(3)
    gt = torch.rand(48, 64, 3, device="mps")
    pred = gt.clone()
    pred[16:32] = 1.0 - pred[16:32]
    mask = torch.ones(48, 64, device="mps")
    mask[16 - SSIM_RADIUS:32 + SSIM_RADIUS] = 0
    k = _gaussian_kernel(device="mps")
    l1_only = 0.2 * (1.0 - ssim_map(pred, gt, k)).mean().item()
    assert l1_only > 0.02, "the unmasked SSIM term must be large, or this proves nothing"
    assert photometric_loss(pred, gt, mask, k).item() < 1e-6


@mps
def test_photometric_loss_does_NOT_divide_by_coverage_but_masked_mse_does():
    """The deliberate asymmetry, pinned. `photometric_loss` zeroes the numerator only
    (Brush `image_loss` semantics -- a training loss is a descent direction whose scale
    the learning rate absorbs); `masked_mse` divides, because a reported PSNR that did
    not would be inflated by -10 log10(coverage) and compared against a 24 dB gate.

    Making them "consistent" is the tempting wrong fix, and passes every other test in
    this file. Here the error is a uniform 0.2 offset over the whole frame and the mask
    keeps exactly half, so an undivided loss is HALF the unmasked one and a divided one
    is equal to it -- the two hypotheses are a factor of two apart.
    """
    from metal_gauss.train import _gaussian_kernel, masked_mse, photometric_loss
    torch.manual_seed(7)
    gt = torch.rand(48, 64, 3, device="mps") * 0.5
    pred = gt + 0.2                                     # exactly 0.2 of L1 everywhere
    mask = torch.zeros(48, 64, device="mps"); mask[:24] = 1.0
    k = _gaussian_kernel(device="mps")
    ratio = (photometric_loss(pred, gt, mask, k).item()
             / photometric_loss(pred, gt, None, k).item())
    assert 0.44 < ratio < 0.56, f"expected ~0.5 (numerator-only mask), got {ratio}"
    # ...while the METRIC on the same half-covered frame is coverage-corrected, so it
    # matches the unmasked MSE of a uniform error instead of halving it.
    err2 = (pred - gt) ** 2
    assert masked_mse(err2, mask).item() == pytest.approx(err2.mean().item(), rel=1e-5)


def test_masked_mse_divides_by_coverage():
    """Multiplying the error map by the mask and then .mean() over ALL pixels understates
    MSE by exactly the coverage and OVERSTATES PSNR by -10 log10(coverage): +2.3 dB at
    59% coverage (CLAUDE.md Stage 4). The fix divides by mean(mask)."""
    from metal_gauss.train import masked_mse
    err2 = torch.full((10, 10, 3), 0.01)
    mask = torch.zeros(10, 10); mask[:5] = 1
    err2[5:] = 100.0                                          # garbage where dropped
    assert masked_mse(err2, mask).item() == pytest.approx(0.01, rel=1e-6)
    assert masked_mse(err2, None).item() == pytest.approx((0.01 * 50 + 100.0 * 50) / 100,
                                                          rel=1e-6)


def test_masked_mse_without_the_coverage_divisor_would_inflate_psnr_by_23_db():
    """Pins the magnitude, not just the direction. At 59% coverage the undivided form
    reports +2.3 dB of PSNR that is not there."""
    from metal_gauss.train import masked_mse
    torch.manual_seed(0)
    err2 = torch.rand(100, 100, 3) * 0.01
    mask = (torch.rand(100, 100) < 0.59).float()
    cov = mask.mean().item()
    correct = masked_mse(err2, mask).item()
    undivided = (err2 * mask[..., None]).mean().item()
    inflation = -10 * math.log10(undivided) - (-10 * math.log10(correct))
    assert inflation == pytest.approx(-10 * math.log10(cov), abs=1e-6)
    assert 2.0 < inflation < 2.6


def test_masked_mse_on_a_fully_dropped_frame_is_finite():
    """An all-dropped frame must not produce an infinite PSNR and drag the mean up."""
    from metal_gauss.train import masked_mse
    err2 = torch.full((8, 8, 3), 0.25)
    v = masked_mse(err2, torch.zeros(8, 8)).item()
    assert math.isfinite(v) and v == pytest.approx(0.0)


@mps
def test_evaluate_reports_masked_psnr_and_coverage_from_the_view_masks():
    """The acceptance instrument, end to end: `evaluate` must return the mask mean as
    `coverage` and score `psnr_masked` over the kept pixels only. With the wrecked
    region entirely dropped, masked PSNR must be far above the unmasked one."""
    from metal_gauss.dataset import Scene, View
    from metal_gauss.train import evaluate

    H = W = 64
    K = torch.tensor([[W, 0.0, W / 2], [0.0, H, H / 2], [0.0, 0.0, 1.0]])
    vm = torch.eye(4); vm[2, 3] = 3.0
    # Render is black (no gaussians in front), so the render matches a black GT.
    img = torch.zeros(H, W, 3, dtype=torch.uint8)
    img[:16] = 255                                  # a bright band the render misses
    m = torch.full((H, W), 255, dtype=torch.uint8)
    m[:16] = 0                                      # ...and it is dropped
    v = View("e0", img, K, vm, mask=m)
    scene = Scene(train=[v], heldout=[v], points=np.zeros((0, 3), np.float32),
                  colors=np.zeros((0, 3), np.float32))
    n = 8
    p = {"means": torch.randn(n, 3, device="mps") * 0.01 + torch.tensor([0.0, 0.0, -50.0], device="mps"),
         "quats": torch.tensor([[1.0, 0, 0, 0]], device="mps").repeat(n, 1),
         "log_scales": torch.full((n, 3), -6.0, device="mps"),
         "logit_opac": torch.full((n,), -20.0, device="mps"),
         "sh_dc": torch.zeros(n, 1, 3, device="mps"),
         "sh_rest": torch.zeros(n, 15, 3, device="mps")}
    r = evaluate(p, scene, "mps", active=n)
    assert set(r) >= {"psnr_masked", "psnr", "coverage"}
    assert r["coverage"] == pytest.approx(0.75, abs=1e-4)      # 48 of 64 rows kept
    assert r["psnr_masked"] > r["psnr"] + 5.0


@mps
def test_evaluate_without_masks_reports_full_coverage_and_equal_psnrs():
    from metal_gauss.dataset import Scene, View
    from metal_gauss.train import evaluate
    H = W = 32
    K = torch.tensor([[W, 0.0, W / 2], [0.0, H, H / 2], [0.0, 0.0, 1.0]])
    vm = torch.eye(4); vm[2, 3] = 3.0
    img = torch.zeros(H, W, 3, dtype=torch.uint8); img[:8] = 255
    v = View("e0", img, K, vm)
    scene = Scene(train=[v], heldout=[v], points=np.zeros((0, 3), np.float32),
                  colors=np.zeros((0, 3), np.float32))
    n = 8
    p = {"means": torch.zeros(n, 3, device="mps") + torch.tensor([0.0, 0.0, -50.0], device="mps"),
         "quats": torch.tensor([[1.0, 0, 0, 0]], device="mps").repeat(n, 1),
         "log_scales": torch.full((n, 3), -6.0, device="mps"),
         "logit_opac": torch.full((n,), -20.0, device="mps"),
         "sh_dc": torch.zeros(n, 1, 3, device="mps"),
         "sh_rest": torch.zeros(n, 15, 3, device="mps")}
    r = evaluate(p, scene, "mps", active=n)
    assert r["coverage"] == 1.0
    assert r["psnr_masked"] == pytest.approx(r["psnr"], rel=1e-6)


# ------------------------------------------------- the VOID warning + reporting

def test_void_warning_fires_when_masks_were_supplied_but_coverage_is_100_pct():
    """The instrument itself. A run that supplied masks and still reports 100.0%
    coverage did not apply them -- that is splat_3840v2_full's 49.561 dB."""
    from metal_gauss.train import mask_void_warning
    w = mask_void_warning(masks_supplied=True, coverage=1.0)
    assert w is not None and "VOID" in w and "100.0%" in w


def test_void_warning_is_silent_when_no_masks_were_supplied():
    """An unmasked dataset legitimately reads 100.0%. Warning there would train
    operators to ignore the line, which is worse than not printing it."""
    from metal_gauss.train import mask_void_warning
    assert mask_void_warning(masks_supplied=False, coverage=1.0) is None


def test_void_warning_is_silent_on_a_genuinely_masked_run():
    from metal_gauss.train import mask_void_warning
    assert mask_void_warning(masks_supplied=True, coverage=0.935) is None


def test_void_warning_fires_on_coverage_that_rounds_to_100_pct():
    """A dataset whose masks reached one pixel of one frame is still a failure, and
    prints as `100.0%`. The threshold is on the value, not on its rendering."""
    from metal_gauss.train import mask_void_warning
    assert mask_void_warning(masks_supplied=True, coverage=0.99999) is not None
    assert mask_void_warning(masks_supplied=True, coverage=0.998) is None


def test_run_report_records_masked_psnr_and_coverage_and_keeps_legacy_psnr_unmasked():
    """`--report` is what gets cited; stdout is not. The legacy `psnr` key must keep
    meaning the UNMASKED number so readers written before masks existed stay correct."""
    import argparse
    from metal_gauss.train import _run_report
    args = argparse.Namespace(steps=10, budget=7, seed=0)
    log = [{"step": 10, "psnr": 19.5, "psnr_masked": 24.25, "coverage": 0.923,
            "active": 7}]
    r = _run_report(args, log, 12.5, 7)
    assert r["metrics"]["psnr"] == 19.5
    assert r["metrics"]["psnr_masked"] == 24.25
    assert r["metrics"]["coverage"] == 0.923
    assert r["final_psnr"] == 19.5
