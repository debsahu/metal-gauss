"""The three geometry losses fused into one pass, against the torch chain.

THE FIXTURE IS THE TEST. Three plausible errors -- `scale_alpha_not_nsum`,
`gate_prenorm_nr` and `no_alpha_on_depth` -- read 3e-16 on a production-like fixture
(alpha ~ 1, agreeing normals, tilted plane) and are therefore perfectly invisible there.
They only appear with alpha spread U(0.2,1), holes, and n_sum = a*(t*n1 + (1-t)*n2) so the
contributors DISAGREE. Measured: all three loud at 0.79-1.0 on the exposing fixture, and
gate_prenorm_nr exactly 0.000 on the production-like one.
Reference: research/depth-normal-loss-adjoint.md section 8.
"""
import pytest
import torch

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
GRAZING = (4.0, 5.0, -3.5, 2.0)      # fixture-2 intrinsics
PROD = (1000.0, 1000.0, 64.0, 48.0)


def exposing(h=40, w=52, seed=0, dev="mps"):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(h, w, generator=g) * 0.8 + 0.2
    a[torch.rand(h, w, generator=g) < 0.15] = 0.0
    n1 = torch.nn.functional.normalize(torch.randn(h, w, 3, generator=g), dim=-1)
    n2 = torch.nn.functional.normalize(torch.randn(h, w, 3, generator=g), dim=-1)
    t = torch.rand(h, w, 1, generator=g)
    n_sum = a[..., None] * (t * n1 + (1 - t) * n2)
    z = ((torch.rand(h, w, generator=g) * 2 + 0.5) * a)[..., None].expand(h, w, 3).contiguous()
    gt_d = torch.rand(h, w, generator=g) * 3 + 0.5
    gt_d[torch.rand(h, w, generator=g) < 0.12] = 0.0          # invalid depth prior
    gt_n = torch.nn.functional.normalize(torch.randn(h, w, 3, generator=g), dim=-1)
    gt_n[torch.rand(h, w, generator=g) < 0.12] = 0.0          # invalid normal prior
    return [x.to(dev) for x in (n_sum, z, a, gt_d, gt_n)]


def torch_chain(z_img, n_sum, alpha, gt_d, gt_n, K, space="disparity"):
    """Exactly what train.py's geometry_terms computes, alpha detached."""
    from metal_gauss.geometry_loss import (depth_loss, depth_normal_loss, normal_loss,
                                           normals_from_depth)
    a = alpha.detach().clamp_min(1e-10)
    ni = n_sum / a[..., None]
    ni = ni / ni.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    di = z_img[..., 0] / a
    nd = normals_from_depth(di, *K)
    return torch.stack([depth_loss(di, gt_d, space), normal_loss(ni, gt_n),
                        depth_normal_loss(nd, ni, alpha.detach())])


def _bars(got, ref, what):
    got, ref = got.detach().cpu(), ref.detach().cpu()
    rel = ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
    cos = torch.nn.functional.cosine_similarity(
        got.flatten().double(), ref.flatten().double(), dim=0).item()
    assert rel <= 1e-5 and cos >= 1 - 1e-6, f"{what}: rel {rel:.3e} cos {cos:.9f}"


@mps
@pytest.mark.parametrize("K", [GRAZING, PROD])
def test_fused_loss_VALUES_match_the_torch_chain(K):
    """A loss-value parity test that gradient tests can never replace: a sequential f32
    accumulate over 2.7M pixels errs 6.0e-4 on the VALUE while every gradient check still
    passes, because gradients see only 1/N."""
    from metal_gauss.geometry_loss import fused_geometry_losses
    n_sum, z, a, gt_d, gt_n = exposing()
    got = fused_geometry_losses(z, n_sum, a, gt_d, gt_n, K)
    ref = torch_chain(z, n_sum, a, gt_d, gt_n, K)
    _bars(got, ref, f"loss values K={K[0]}")


@mps
@pytest.mark.parametrize("K", [GRAZING, PROD])
def test_fused_loss_GRADIENTS_match_the_torch_chain_on_the_exposing_fixture(K):
    from metal_gauss.geometry_loss import fused_geometry_losses
    n_sum, z, a, gt_d, gt_n = exposing()
    w = torch.tensor([1.0, 0.2, 0.05], device="mps")
    zf, nf = z.clone().requires_grad_(True), n_sum.clone().requires_grad_(True)
    (fused_geometry_losses(zf, nf, a, gt_d, gt_n, K) * w).sum().backward()
    # TORCH f32 IS NOT GROUND TRUTH. At the grazing intrinsics (fx=4) the composed chain
    # carries a 1/L factor and both f32 paths drift: measured against an f64 CPU reference,
    # the fused kernel is CLOSER to truth than torch's own f32 chain (7.02e-6 vs 8.06e-6),
    # while their mutual difference is 1.19e-5 -- the sum of two independent errors. So the
    # comparison is against f64, which is the only thing here that is actually right.
    zc = z.cpu().double().requires_grad_(True)
    nc = n_sum.cpu().double().requires_grad_(True)
    (torch_chain(zc, nc, a.cpu().double(), gt_d.cpu().double(), gt_n.cpu().double(), K)
     * w.cpu().double()).sum().backward()
    _bars(zf.grad.cpu().double(), zc.grad, f"dL/dz_img vs f64 K={K[0]}")
    _bars(nf.grad.cpu().double(), nc.grad, f"dL/dn_sum vs f64 K={K[0]}")


@mps
def test_depth_cotangent_is_confined_to_channel_zero_and_alpha_gets_none():
    """z_img channels 1-2 carry no information, so their gradient must be EXACTLY zero.
    And alpha must receive nothing at all -- un-detaching it gives max|dL/dalpha| = 0.37,
    all from the depth branch."""
    from metal_gauss.geometry_loss import fused_geometry_losses
    n_sum, z, a, gt_d, gt_n = exposing(seed=3)
    af = a.clone().requires_grad_(True)
    zf, nf = z.clone().requires_grad_(True), n_sum.clone().requires_grad_(True)
    fused_geometry_losses(zf, nf, af, gt_d, gt_n, GRAZING).sum().backward()
    assert zf.grad[..., 1].abs().max().item() == 0.0
    assert zf.grad[..., 2].abs().max().item() == 0.0
    assert af.grad is None or af.grad.abs().max().item() == 0.0


@mps
def test_the_dn_term_contributes_nothing_where_alpha_is_at_or_below_half():
    """The gate is strict `>`, matching torch. A pixel at exactly 0.5 is EXCLUDED."""
    from metal_gauss.geometry_loss import fused_geometry_losses
    n_sum, z, a, gt_d, gt_n = exposing(seed=5)
    a = a.clone(); a[0, 0] = 0.5                       # exactly at the boundary
    ref = torch_chain(z, n_sum, a, gt_d, gt_n, GRAZING)
    got = fused_geometry_losses(z, n_sum, a, gt_d, gt_n, GRAZING)
    _bars(got, ref, "alpha == 0.5 boundary")


@mps
def test_metric_space_also_matches():
    from metal_gauss.geometry_loss import fused_geometry_losses
    n_sum, z, a, gt_d, gt_n = exposing(seed=7)
    zf, nf = z.clone().requires_grad_(True), n_sum.clone().requires_grad_(True)
    fused_geometry_losses(zf, nf, a, gt_d, gt_n, GRAZING, space="metric").sum().backward()
    zr, nr = z.clone().requires_grad_(True), n_sum.clone().requires_grad_(True)
    torch_chain(zr, nr, a, gt_d, gt_n, GRAZING, "metric").sum().backward()
    _bars(zf.grad, zr.grad, "metric dL/dz")
    _bars(nf.grad, nr.grad, "metric dL/dn_sum")
