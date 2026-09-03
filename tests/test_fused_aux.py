"""Task 12: auxiliary channels composited INSIDE the RGB rasterisation pass.

Why fuse at all, from this project's own Task 11 measurement (500k splats, 1920x1440,
GPU exclusive): the two extra `_RasterizeMetal` passes cost 115.57 ms, of which the
FORWARD is only 29.3 ms. Roughly 86 ms -- three quarters -- is a second and third walk of
the tile lists in the BACKWARD. Fusing removes those extra traversals.

The forward is the easy half and it has one hard requirement: the RGB image must not
change at all. The aux composite uses the same weights `w = alpha * T` in the same
front-to-back order as a separate pass over the same tile lists, so the aux map must come
out BIT-IDENTICAL to the Tier 1 two-pass result -- not merely close. Anything looser would
hide a reordering.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def _scene(n=4000, W=96, H=64, seed=3):
    """uv / conic / opacity / colour and the tile lists, built exactly as render() does."""
    from metal_gauss import metal_backend as mb
    torch.manual_seed(seed)
    m = (torch.randn(n, 3) * 0.6 + torch.tensor([0., 0., 4.])).to("mps")
    q = torch.randn(n, 4).to("mps")
    s = (torch.rand(n, 3) * 0.06 + 0.02).to("mps")
    o = (torch.rand(n) * 0.7 + 0.2).to("mps")
    sh = (torch.randn(n, 16, 3) * 0.25).to("mps")
    K = torch.eye(3); K[0, 0] = K[1, 1] = 0.8 * max(W, H); K[0, 2], K[1, 2] = W / 2, H / 2
    vm = torch.eye(4)
    cam_center = -vm[:3, :3].T @ vm[:3, 3]
    uv, conic, depth, rxy, valid_i, colors = mb._PreprocessMetal.apply(
        m, q, s, sh, sh, o, vm.contiguous(),
        K[0, 0].item(), K[1, 1].item(), K[0, 2].item(), K[1, 2].item(),
        W, H, 0.01, 100.0, 0.3, 32.0 * max(W, H), cam_center, 3)
    tile = 16
    gauss_ids, tile_offsets, tiles_x = mb.build_tile_lists_metal(
        uv.detach(), rxy.detach(), depth.detach(), valid_i,
        conic.detach(), o.detach(), W, H, tile)
    return dict(uv=uv.detach(), conic=conic.detach(), opac=o.detach(),
                color=colors.detach(), gauss_ids=gauss_ids, tile_offsets=tile_offsets,
                W=W, H=H, tile=tile, tiles_x=tiles_x, n=n)


def _sep(sc, colour):
    """Tier 1 reference: a separate rasterisation pass over the SAME tile lists."""
    from metal_gauss import metal_backend as mb
    return mb._RasterizeMetal.apply(
        sc["uv"], sc["conic"], sc["opac"], colour.contiguous(),
        sc["gauss_ids"], sc["tile_offsets"], sc["W"], sc["H"], sc["tile"],
        sc["tiles_x"], None)


def test_fused_aux_map_is_BIT_IDENTICAL_to_a_separate_pass():
    """Same weights, same order, same lists -- so 'close' is not good enough. A tolerance
    here would accept a reordered traversal, which is exactly the bug a fused pass can
    introduce and the one hardest to see in a rendered image."""
    from metal_gauss import metal_backend as mb
    sc = _scene()
    aux = torch.randn(sc["n"], 4, device="mps")
    rgb_f, alpha_f, _, _, aux_f = mb.rasterize_fused_forward(
        sc["uv"], sc["conic"], sc["opac"], sc["color"], aux,
        sc["gauss_ids"], sc["tile_offsets"], sc["W"], sc["H"], sc["tile"], sc["tiles_x"])
    assert aux_f.shape == (sc["H"], sc["W"], 4)
    assert alpha_f.max() > 0.5, "nothing rendered; the comparison below would be vacuous"
    # Channels 0..2 against a separate 3-channel pass with the same values.
    ref3 = _sep(sc, aux[:, :3])[0]
    assert torch.equal(aux_f[..., :3], ref3), \
        f"aux channels 0-2 differ from the separate pass: max |d| {(aux_f[..., :3] - ref3).abs().max()}"
    # Channel 3 against a pass carrying it in slot 0.
    ref1 = _sep(sc, aux[:, 3:4].expand(-1, 3).contiguous())[0]
    assert torch.equal(aux_f[..., 3], ref1[..., 0])


def test_fused_rgb_is_BIT_IDENTICAL_with_and_without_aux():
    """The whole point of a fused pass is that the image does not move. If this ever fails,
    every A/B in the measurement tier is confounded by whether geometry terms were on."""
    from metal_gauss import metal_backend as mb
    sc = _scene(seed=5)
    args = (sc["uv"], sc["conic"], sc["opac"], sc["color"])
    tail = (sc["gauss_ids"], sc["tile_offsets"], sc["W"], sc["H"], sc["tile"], sc["tiles_x"])
    rgb_off, a_off, T_off, n_off, aux_off = mb.rasterize_fused_forward(
        *args, torch.empty(0, 4, device="mps"), *tail)
    rgb_on, a_on, T_on, n_on, _ = mb.rasterize_fused_forward(
        *args, torch.randn(sc["n"], 4, device="mps"), *tail)
    assert torch.equal(rgb_off, rgb_on)
    assert torch.equal(a_off, a_on) and torch.equal(T_off, T_on) and torch.equal(n_off, n_on)
    assert aux_off.numel() == 0, "no aux requested -> no aux map allocated"


def test_fused_rgb_matches_the_UNFUSED_kernel_exactly():
    """The fused kernel replaces the existing one; with aux off it must reproduce it to the
    bit, or Tier 2 has silently changed every photometric result."""
    from metal_gauss import metal_backend as mb
    sc = _scene(seed=7)
    rgb_ref, alpha_ref = _sep(sc, sc["color"])
    rgb_f, alpha_f, _, _, _ = mb.rasterize_fused_forward(
        sc["uv"], sc["conic"], sc["opac"], sc["color"],
        torch.empty(0, 4, device="mps"), sc["gauss_ids"], sc["tile_offsets"],
        sc["W"], sc["H"], sc["tile"], sc["tiles_x"])
    assert torch.equal(rgb_f, rgb_ref) and torch.equal(alpha_f, alpha_ref)


def test_fused_aux_rejects_a_wrong_shaped_aux_tensor():
    from metal_gauss import metal_backend as mb
    sc = _scene(seed=9)
    with pytest.raises(RuntimeError, match="aux"):
        mb.rasterize_fused_forward(
            sc["uv"], sc["conic"], sc["opac"], sc["color"],
            torch.randn(sc["n"], 3, device="mps"),          # 3 channels, not 4
            sc["gauss_ids"], sc["tile_offsets"], sc["W"], sc["H"], sc["tile"], sc["tiles_x"])


def test_fused_aux_is_zero_where_nothing_is_covered():
    """An uncovered pixel must read exactly 0 in every aux channel -- 'no measurement',
    not a background constant. A depth map with a background baked in reads as a plausible
    surface."""
    from metal_gauss import metal_backend as mb
    sc = _scene(seed=11)
    aux = torch.full((sc["n"], 4), 7.0, device="mps")
    _, alpha, _, _, aux_f = mb.rasterize_fused_forward(
        sc["uv"], sc["conic"], sc["opac"], sc["color"], aux,
        sc["gauss_ids"], sc["tile_offsets"], sc["W"], sc["H"], sc["tile"], sc["tiles_x"])
    uncovered = alpha < 1e-6
    assert uncovered.any(), "need an uncovered pixel for this test to mean anything"
    assert aux_f[uncovered].abs().max().item() == 0.0


# ---------------------------------------------------------------- Task 14: adjoint

def _fwd(sc, aux):
    from metal_gauss import metal_backend as mb
    return mb.rasterize_fused_forward(
        sc["uv"], sc["conic"], sc["opac"], sc["color"], aux,
        sc["gauss_ids"], sc["tile_offsets"], sc["W"], sc["H"], sc["tile"], sc["tiles_x"])


def _bwd(sc, aux, T, ncontrib, g_rgb, g_alpha, g_aux):
    from metal_gauss import metal_backend as mb
    ext = mb._load()
    return ext.rasterize_backward(
        sc["uv"], sc["conic"], sc["opac"], sc["color"], aux,
        sc["gauss_ids"], sc["tile_offsets"], T, ncontrib, g_rgb, g_alpha, g_aux,
        sc["W"], sc["H"], sc["tile"], sc["tiles_x"], False)


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-8)).item()


def test_fused_aux_value_gradient_matches_a_separate_pass():
    """d_aux must equal what a separate rasterisation of the same values produces as
    d_color: the aux VALUE gradient is `w * dL_daux` with `w = alpha*T`, exactly the
    colour lanes' form. Atomics make the reduction order uncontrolled, so this is a tight
    relative bound rather than bit-equality (unlike the forward, which has no atomics)."""
    from metal_gauss import metal_backend as mb
    sc = _scene(seed=21)
    torch.manual_seed(0)
    aux = torch.randn(sc["n"], 4, device="mps")
    g_rgb = torch.randn(sc["H"], sc["W"], 3, device="mps")
    g_alpha = torch.randn(sc["H"], sc["W"], device="mps")
    g_aux = torch.randn(sc["H"], sc["W"], 4, device="mps")
    _, _, T, nc, _ = _fwd(sc, aux)
    out = _bwd(sc, aux, T, nc, g_rgb, g_alpha, g_aux)
    d_aux = out[5]
    assert d_aux.shape == (sc["n"], 4)

    # Reference: a separate 3-channel pass carrying aux[:, :3], with the SAME grad.
    col3 = aux[:, :3].contiguous()
    _, _, T3, nc3, _ = mb.rasterize_fused_forward(
        sc["uv"], sc["conic"], sc["opac"], col3, torch.empty(0, 4, device="mps"),
        sc["gauss_ids"], sc["tile_offsets"], sc["W"], sc["H"], sc["tile"], sc["tiles_x"])
    ref = mb._load().rasterize_backward(
        sc["uv"], sc["conic"], sc["opac"], col3, torch.empty(0, 4, device="mps"),
        sc["gauss_ids"], sc["tile_offsets"], T3, nc3,
        g_aux[..., :3].contiguous(), torch.zeros_like(g_alpha),
        torch.empty(0, 4, device="mps"),
        sc["W"], sc["H"], sc["tile"], sc["tiles_x"], False)[3]
    assert _rel(d_aux[:, :3], ref) < 1e-5, _rel(d_aux[:, :3], ref)


def test_aux_gradients_DO_NOT_reach_the_blending_weights():
    """THE test for this tier. RGB lanes fold into the alpha VJP; AUX LANES MUST NOT.

    Brush drops the `dot_rgb` depth term deliberately (rasterize_backwards.rs:536-563) so
    a geometry loss cannot lower its error by changing opacity or footprint instead of
    moving the gaussian. Tier 1 got this by detaching uv/conic/opacity for the separate
    aux passes; inside ONE fused kernel that separation has to be explicit, because the
    RGB lanes legitimately need the coupling the aux lanes must not have.

    Getting it wrong reproduces the needle collapse: playroom R1 pre-fix went from
    in-plane aspect 0.2957 to 0.0659 and needle fraction 16.6% to 56.8%.

    So: hold the RGB gradient fixed, drive the aux gradient from zero to large, and the
    weight gradients must not move at all."""
    sc = _scene(seed=23)
    torch.manual_seed(1)
    aux = torch.randn(sc["n"], 4, device="mps")
    g_rgb = torch.randn(sc["H"], sc["W"], 3, device="mps")
    g_alpha = torch.randn(sc["H"], sc["W"], device="mps")
    _, _, T, nc, _ = _fwd(sc, aux)
    zero_aux = torch.zeros(sc["H"], sc["W"], 4, device="mps")
    big_aux = torch.randn(sc["H"], sc["W"], 4, device="mps") * 100.0
    a = _bwd(sc, aux, T, nc, g_rgb, g_alpha, zero_aux)
    b = _bwd(sc, aux, T, nc, g_rgb, g_alpha, big_aux)
    for i, name in ((0, "d_uv"), (1, "d_conic"), (2, "d_opacity")):
        assert a[i].abs().max() > 0, f"{name} is all zero; the test would be vacuous"
        r = _rel(b[i], a[i])
        assert r < 1e-6, (
            f"a 100x aux gradient moved {name} by rel {r:.3e} -- the aux lanes are folding "
            f"into the alpha VJP, which is the needle-collapse coupling")
    assert b[5].abs().max() > 0, "aux gradient must still reach the aux VALUE"


def test_rgb_gradients_are_unchanged_by_the_presence_of_aux():
    """The RGB adjoint must be untouched by the new lanes, or every photometric result
    silently depends on whether geometry terms were enabled."""
    sc = _scene(seed=25)
    torch.manual_seed(2)
    g_rgb = torch.randn(sc["H"], sc["W"], 3, device="mps")
    g_alpha = torch.randn(sc["H"], sc["W"], device="mps")
    aux = torch.randn(sc["n"], 4, device="mps")
    _, _, T, nc, _ = _fwd(sc, aux)
    with_aux = _bwd(sc, aux, T, nc, g_rgb, g_alpha,
                    torch.randn(sc["H"], sc["W"], 4, device="mps"))
    no_aux = _bwd(sc, torch.empty(0, 4, device="mps"), T, nc, g_rgb, g_alpha,
                  torch.empty(0, 4, device="mps"))
    for i, name in ((0, "d_uv"), (1, "d_conic"), (2, "d_opacity"), (3, "d_color")):
        r = _rel(with_aux[i], no_aux[i])
        assert r < 1e-6, f"{name} changed when aux was enabled: rel {r:.3e}"


def test_zero_aux_gradient_produces_zero_aux_value_gradient():
    sc = _scene(seed=27)
    aux = torch.randn(sc["n"], 4, device="mps")
    _, _, T, nc, _ = _fwd(sc, aux)
    out = _bwd(sc, aux, T, nc,
               torch.randn(sc["H"], sc["W"], 3, device="mps"),
               torch.randn(sc["H"], sc["W"], device="mps"),
               torch.zeros(sc["H"], sc["W"], 4, device="mps"))
    assert out[5].abs().max().item() == 0.0


# --------------------------------------------------- Task 15: the switch and its guards

def _leaves2(n=300, seed=61):
    torch.manual_seed(seed)
    L = dict(m=torch.randn(n, 3) * 0.6 + torch.tensor([0., 0., 4.]), q=torch.randn(n, 4),
             s=torch.rand(n, 3) * 0.10 + 0.03, o=torch.rand(n) * 0.7 + 0.15,
             sh=torch.randn(n, 16, 3) * 0.25)
    return {k: v.to("mps").requires_grad_(True) for k, v in L.items()}


def _render_pair(L, W=64, H=48, **kw):
    from metal_gauss import render
    from metal_gauss.geometry_loss import splat_normals_cam
    K = torch.eye(3); K[0, 0] = K[1, 1] = 0.8 * max(W, H); K[0, 2], K[1, 2] = W / 2, H / 2
    vm = torch.eye(4); vmd = vm.to("mps")
    aux = [splat_normals_cam(L["m"], L["q"], L["s"], vmd),
           (L["m"] @ vmd[:3, :3].T + vmd[:3, 3])[:, 2:3].expand(-1, 3)]
    return render(L["m"], L["q"], L["s"], L["o"], L["sh"][:, :1].contiguous(), K, vm, W, H,
                  sh_degree=3, sh_rest=L["sh"][:, 1:].contiguous(), backend="metal",
                  aux_colors=aux, aux_detach_weights=[True, True], **kw)


def test_fused_and_multipass_agree_within_the_preregistered_bars(monkeypatch):
    """Plan section 4: rel <= 1e-5 and cosine >= 1 - 1e-6 on every input that takes a
    gradient. The two paths are different kernels computing the same contract, so this is
    the equivalence that licenses switching the default."""
    outs = {}
    for name, mp in (("multipass", "1"), ("fused", "0")):
        monkeypatch.setenv("MG_AUX_MULTIPASS", mp)
        L = _leaves2()
        rgb, alpha, info = _render_pair(L)
        # Without this the test is VACUOUS: before the switch existed, both env settings
        # took the multipass path and it "passed" by comparing a path against itself.
        assert info["aux_path"] == name, f"expected the {name} path, got {info['aux_path']}"
        assert len(info["aux"]) == 2 and info["aux"][0].shape[-1] == 3
        (rgb.square().mean() + info["aux"][0].square().mean()
         + info["aux"][1].square().mean()).backward()
        outs[name] = ({k: L[k].grad.detach().cpu() for k in L},
                      rgb.detach().cpu(), info["aux"][0].detach().cpu(),
                      info["aux"][1].detach().cpu())
    gm, rgb_m, n_m, z_m = outs["multipass"]
    gf, rgb_f, n_f, z_f = outs["fused"]
    for a, b, nm in ((rgb_m, rgb_f, "rgb"), (n_m, n_f, "normal map"), (z_m, z_f, "z map")):
        assert (a - b).abs().max().item() < 2e-3, nm
    for k in gm:
        rel = ((gm[k] - gf[k]).abs().max() / gm[k].abs().max().clamp_min(1e-8)).item()
        cos = torch.nn.functional.cosine_similarity(
            gm[k].flatten(), gf[k].flatten(), dim=0).item()
        assert torch.isfinite(gf[k]).all() and rel < 1e-5 and cos > 1 - 1e-6, (k, rel, cos)


def test_fused_is_the_default_and_the_env_var_forces_the_old_path(monkeypatch):
    from metal_gauss import metal_backend as mb
    monkeypatch.delenv("MG_AUX_MULTIPASS", raising=False)
    assert mb._use_fused_aux([True, True], 2) is True
    monkeypatch.setenv("MG_AUX_MULTIPASS", "1")
    assert mb._use_fused_aux([True, True], 2) is False


def test_live_weight_aux_falls_back_to_the_multipass_path(monkeypatch):
    """The fused kernel ALWAYS drops the aux alpha VJP -- that is its contract. A caller
    asking for LIVE weights (Brush folds alpha in for the PGSR plane channels) cannot be
    served by it, so it must fall back rather than silently deliver the wrong contract.
    plane-aux (plan Task 20) depends on this."""
    from metal_gauss import metal_backend as mb
    monkeypatch.delenv("MG_AUX_MULTIPASS", raising=False)
    assert mb._use_fused_aux([False, False], 2) is False
    assert mb._use_fused_aux([True, False], 2) is False
    assert mb._use_fused_aux([True], 1) is False          # only the 2-map packing is fused
    assert mb._use_fused_aux([True, True, True], 3) is False


def test_live_weight_aux_still_gives_live_weight_gradients_after_the_switch(monkeypatch):
    """End to end: the fallback must actually preserve the distinction the per-channel API
    exists for, not merely choose a different code path."""
    from metal_gauss import render
    monkeypatch.delenv("MG_AUX_MULTIPASS", raising=False)
    W = H = 48
    K = torch.eye(3); K[0, 0] = K[1, 1] = 0.8 * W; K[0, 2] = K[1, 2] = W / 2
    vm = torch.eye(4)

    def opac_grad(detach):
        L = _leaves2(n=150, seed=63)
        z = (L["m"] @ vm[:3, :3].T.to("mps") + vm[:3, 3].to("mps"))[:, 2:3].expand(-1, 3)
        _, _, info = render(L["m"], L["q"], L["s"], L["o"], L["sh"], K, vm, W, H,
                            sh_degree=3, backend="metal",
                            aux_colors=[z], aux_detach_weights=[detach])
        info["aux"][0].square().mean().backward()
        return 0.0 if L["o"].grad is None else L["o"].grad.abs().max().item()

    assert opac_grad(True) < 1e-8
    assert opac_grad(False) > 1e-8
