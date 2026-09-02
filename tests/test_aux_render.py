"""Aux channels over the fused pass's own projection + tile lists.

Oracle: TWO torch_ref renders (colors=rgb, colors=aux) summed into one loss. The design
point is that the extra maps reuse ONE fused preprocess and ONE Metal binning; calling
`render(colors=...)` again instead re-projects and re-bins in torch, and its BACKWARD is
where the cost lands.
"""
import pytest
import torch

from metal_gauss import render

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def _K(W, H):
    f = 0.8 * max(W, H)
    K = torch.eye(3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = f, f, W / 2, H / 2
    return K


def _leaves(n=300, seed=11, dev="mps"):
    torch.manual_seed(seed)
    L = dict(m=torch.randn(n, 3) * 0.6 + torch.tensor([0., 0., 4.]), q=torch.randn(n, 4),
             s=torch.rand(n, 3) * 0.10 + 0.03, o=torch.rand(n) * 0.7 + 0.15,
             sh=torch.randn(n, 16, 3) * 0.25)
    return {k: v.to(dev).requires_grad_(True) for k, v in L.items()}


def _normals(m, q, s, vm):
    from metal_gauss.geometry_loss import splat_normals_cam
    return splat_normals_cam(m, q, s, vm)


def _z(m, vm):
    return (m @ vm[:3, :3].T + vm[:3, 3])[:, 2:3].expand(-1, 3)


def test_aux_maps_and_gradients_match_oracle():
    W, H = 64, 48
    K = _K(W, H)
    vm = torch.eye(4)
    vmd = vm.to("mps")

    def loss_of(backend, L):
        if backend == "metal":
            rgb, alpha, info = render(
                L["m"], L["q"], L["s"], L["o"], L["sh"][:, :1].contiguous(), K, vm, W, H,
                sh_degree=3, sh_rest=L["sh"][:, 1:].contiguous(), backend="metal",
                aux_colors=[_normals(L["m"], L["q"], L["s"], vmd), _z(L["m"], vmd)])
            nrm, dep = info["aux"]
        else:
            rgb, alpha, _ = render(L["m"], L["q"], L["s"], L["o"], L["sh"], K.to("mps"),
                                   vmd, W, H, sh_degree=3, backend="torch_ref")
            nrm = render(L["m"], L["q"], L["s"], L["o"], None, K.to("mps"), vmd, W, H,
                         colors=_normals(L["m"], L["q"], L["s"], vmd),
                         backend="torch_ref", background=None)[0]
            dep = render(L["m"], L["q"], L["s"], L["o"], None, K.to("mps"), vmd, W, H,
                         colors=_z(L["m"], vmd), backend="torch_ref", background=None)[0]
        depth = dep[..., 0] / alpha.detach().clamp_min(1e-10)
        return rgb.square().mean() + nrm.square().mean() + 0.1 * depth.mean(), (rgb, nrm, dep)

    Lm, Lr = _leaves(), _leaves()
    lm, (rgb_m, nrm_m, dep_m) = loss_of("metal", Lm); lm.backward()
    lr, (rgb_r, nrm_r, dep_r) = loss_of("torch_ref", Lr); lr.backward()
    for a, b, name in [(rgb_r, rgb_m, "rgb"), (nrm_r, nrm_m, "normal"), (dep_r, dep_m, "depth")]:
        assert (a - b).abs().max().item() < 2e-3, name
    for k in Lm:
        a, b = Lr[k].grad.cpu(), Lm[k].grad.cpu()
        rel = ((a - b).abs().max() / a.abs().max().clamp_min(1e-8)).item()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        assert torch.isfinite(b).all() and rel < 1e-5 and cos > 1 - 1e-6, (k, rel, cos)


def test_aux_maps_are_not_all_zero():
    """Guard against the oracle test passing because BOTH sides render nothing. An aux map
    of zeros matches an oracle of zeros to any tolerance."""
    W, H = 64, 48
    L = _leaves(seed=4)
    vm = torch.eye(4)
    _, alpha, info = render(L["m"], L["q"], L["s"], L["o"], L["sh"], _K(W, H), vm, W, H,
                            sh_degree=3, backend="metal",
                            aux_colors=[_normals(L["m"], L["q"], L["s"], vm.to("mps")),
                                        _z(L["m"], vm.to("mps"))])
    nrm, dep = info["aux"]
    assert alpha.max() > 0.5, "the scene did not render; nothing below is meaningful"
    assert nrm.abs().max() > 0.1 and dep.abs().max() > 1.0
    assert (dep[..., 0] > 0).any()


def test_aux_alpha_is_the_rgb_alpha_bit_for_bit():
    W, H = 64, 48
    L = _leaves(seed=3)
    vm = torch.eye(4)
    rgb, alpha, info = render(L["m"], L["q"], L["s"], L["o"], L["sh"], _K(W, H), vm, W, H,
                              sh_degree=3, backend="metal",
                              aux_colors=[_z(L["m"], vm.to("mps"))])
    # IDENTITY, not merely equality: the aux maps are divided by this, and it must be the
    # very tensor the RGB pass produced rather than a recomputed lookalike.
    assert info["aux_alpha"] is alpha
    assert torch.equal(info["aux_alpha"], alpha)


def test_aux_passes_do_not_vote_on_densification():
    """`absgrad_out` accumulates the screen-space gradient MCMC relocation samples on.
    Forwarding it to the aux passes would let a depth or normal map decide which gaussians
    get split -- a densification policy quietly driven by the geometry recipe, confounding
    every A/B in the measurement tier.

    THE AUX MAP MUST BE IN THE LOSS. Without that the aux pass's backward never runs, the
    buffer is untouched either way, and the test passes under the defect. A first version of
    this test made exactly that mistake and was caught only because it also demanded
    bit-exactness -- which the buffer cannot give, being accumulated through atomics whose
    order is not controllable (train.py: "~1e-10 a step").

    So the bar is relative, and calibrated against a measured floor: two identical runs
    disagree at ~2e-8, the correct implementation puts aux-vs-none at ~3.5e-8, and
    forwarding `absgrad_out` puts it at **30.7** -- nine orders of magnitude away.
    """
    W, H = 48, 32
    vm = torch.eye(4)

    def run(with_aux):
        L = _leaves(seed=23)
        buf = torch.zeros(L["m"].shape[0], device="mps")
        aux = [_z(L["m"], vm.to("mps"))] if with_aux else None
        rgb, _, info = render(L["m"], L["q"], L["s"], L["o"], L["sh"], _K(W, H), vm, W, H,
                              sh_degree=3, backend="metal", absgrad_out=buf, aux_colors=aux)
        loss = rgb.square().mean()
        if with_aux:
            loss = loss + info["aux"][0].square().mean()      # supervise it, or this is vacuous
        loss.backward()
        return buf.cpu()

    base, repeat, with_aux = run(False), run(False), run(True)
    assert base.abs().sum() > 0, "no densification signal accumulated; test is vacuous"
    rel = lambda x, y: ((x - y).abs().max() / y.abs().max()).item()
    floor = rel(repeat, base)
    assert floor < 1e-5, f"atomics floor unexpectedly large ({floor:.2e}); recalibrate"
    delta = rel(with_aux, base)
    assert delta < 1e-5, (
        f"aux passes polluted the densification signal: rel {delta:.3e} against an atomics "
        f"floor of {floor:.3e} (forwarding absgrad_out measures ~3e+01)")


def test_aux_alpha_is_the_PRE_composite_alpha_under_a_white_background():
    """`info['aux_alpha']` must be the coverage the aux maps were divided by, not something
    the background composite touched. Pinned under a non-zero background, where a bug that
    returned a composited quantity would show."""
    W, H = 32, 32
    L = _leaves(seed=9)
    vm = torch.eye(4)
    _, alpha, info = render(L["m"], L["q"], L["s"], L["o"], L["sh"], _K(W, H), vm, W, H,
                            sh_degree=3, backend="metal", background=(1.0, 1.0, 1.0),
                            aux_colors=[_z(L["m"], vm.to("mps"))])
    assert torch.equal(info["aux_alpha"], alpha)
    assert alpha.max() <= 1.0 and alpha.min() >= 0.0


def test_no_background_is_added_to_aux_maps():
    """A background composite on a depth map would add the background constant to every
    uncovered pixel and turn 'no measurement' into a plausible-looking depth."""
    W, H = 32, 32
    L = _leaves(seed=13)
    vm = torch.eye(4)
    aux = torch.full((L["m"].shape[0], 3), 7.0, device="mps")
    _, alpha, info = render(L["m"], L["q"], L["s"], L["o"], L["sh"], _K(W, H), vm, W, H,
                            sh_degree=3, backend="metal", background=(1.0, 1.0, 1.0),
                            aux_colors=[aux])
    uncovered = alpha < 1e-6
    assert uncovered.any(), "need an uncovered pixel for this test to mean anything"
    assert info["aux"][0][uncovered].abs().max().item() == 0.0


def test_aux_colors_empty_list_and_none_behave_identically():
    W, H = 32, 32
    L = _leaves(seed=17)
    vm = torch.eye(4)
    a = render(L["m"], L["q"], L["s"], L["o"], L["sh"], _K(W, H), vm, W, H,
               sh_degree=3, backend="metal")
    b = render(L["m"], L["q"], L["s"], L["o"], L["sh"], _K(W, H), vm, W, H,
               sh_degree=3, backend="metal", aux_colors=[])
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert b[2].get("aux") == []


def test_rgb_is_bit_identical_with_and_without_aux():
    """The aux passes must not perturb the RGB image at all -- otherwise every A/B in
    Task 10 is confounded by whether geometry terms were on."""
    W, H = 48, 32
    La, Lb = _leaves(seed=21), _leaves(seed=21)
    vm = torch.eye(4)
    rgb_a, alpha_a, _ = render(La["m"], La["q"], La["s"], La["o"], La["sh"], _K(W, H), vm,
                               W, H, sh_degree=3, backend="metal")
    rgb_b, alpha_b, _ = render(Lb["m"], Lb["q"], Lb["s"], Lb["o"], Lb["sh"], _K(W, H), vm,
                               W, H, sh_degree=3, backend="metal",
                               aux_colors=[_z(Lb["m"], vm.to("mps"))])
    assert torch.equal(rgb_a, rgb_b) and torch.equal(alpha_a, alpha_b)


def test_aux_on_explicit_colors_path_is_rejected():
    L = _leaves(seed=5)
    vm = torch.eye(4)
    with pytest.raises(ValueError, match="aux_colors"):
        render(L["m"], L["q"], L["s"], L["o"], None, _K(32, 32), vm, 32, 32,
               colors=L["m"], backend="metal", aux_colors=[L["m"]])


def test_aux_wrong_shape_is_rejected():
    L = _leaves(seed=6)
    vm = torch.eye(4)
    with pytest.raises(ValueError, match=r"\(N,3\)"):
        render(L["m"], L["q"], L["s"], L["o"], L["sh"], _K(32, 32), vm, 32, 32,
               sh_degree=3, backend="metal",
               aux_colors=[torch.zeros(L["m"].shape[0], 4, device="mps")])


def test_depth_from_preprocess_output_has_no_gradient_trap_is_documented():
    """_PreprocessMetal.backward DROPS d_depth (metal_backend.py:267-279). This test pins the
    trap: a loss on the preprocess's own `depth` output moves NOTHING. Aux depth must come
    from torch `z = (means @ R^T + t)[:, 2]` instead."""
    from metal_gauss import metal_backend as mb
    L = _leaves(seed=7)
    W = H = 32
    K = _K(W, H)
    vm = torch.eye(4)
    cam_center = -vm[:3, :3].T @ vm[:3, 3]
    uv, conic, depth, rxy, valid, colors = mb._PreprocessMetal.apply(
        L["m"], L["q"], L["s"], L["sh"], L["sh"], L["o"].detach(), vm.contiguous(),
        K[0, 0].item(), K[1, 1].item(), K[0, 2].item(), K[1, 2].item(), W, H, 0.01, 100.0,
        0.3, 32.0, cam_center, 3)
    depth.sum().backward()
    assert L["m"].grad is None or L["m"].grad.abs().sum() == 0
    # ...while the torch route this module uses instead DOES move means.
    L2 = _leaves(seed=7)
    _z(L2["m"], vm.to("mps")).sum().backward()
    assert L2["m"].grad is not None and L2["m"].grad.abs().sum() > 0


def test_unknown_kwargs_are_rejected_but_torch_ref_only_ones_are_tolerated():
    """`**_ignored` exists so `api.render(..., backend='metal')` can carry kwargs that only
    `torch_ref.render` takes (`max_per_tile`, `tile_chunk`, `slab`). It also swallowed
    `aux_colors=` before Task 7 implemented it, which made two tests in this file pass
    vacuously in their RED phase. Tolerate the known set; reject the typo."""
    L = _leaves(seed=31)
    vm = torch.eye(4)
    args = (L["m"], L["q"], L["s"], L["o"], L["sh"], _K(32, 32), vm, 32, 32)
    render(*args, sh_degree=3, backend="metal", max_per_tile=4096, tile_chunk=32, slab=256)
    with pytest.raises(TypeError, match="aux_colours"):
        render(*args, sh_degree=3, backend="metal", aux_colours=[])
