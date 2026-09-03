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
