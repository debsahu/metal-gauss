"""Mask loading: both on-disk conventions, dataset-level polarity, NEAREST resize.

Every masks*/ dir in earthbyte/slam is 255 = DROP; alpha-baked RGBA is a = 0 -> DROP.
In memory metal-gauss carries ONE convention: (H,W) uint8, 255 = KEEP.
"""
import numpy as np
import pytest
import torch
from PIL import Image


def _write(p, arr, mode=None):
    Image.fromarray(arr, mode=mode).save(p)
    return p


def _only_view(scene):
    """views[::eval_split_every] ALWAYS contains index 0, so a one-view scene has an
    empty `train` list however large eval_split_every is. Read the view, not the split."""
    vs = scene.train + scene.heldout
    assert len(vs) == 1, f"expected exactly one view, got {len(vs)}"
    return vs[0]


def _colmap_single_view(root, W, H, name="v0.png"):
    """Minimal COLMAP text model: one PINHOLE camera, one identity-pose image, no points."""
    (root / "sparse").mkdir(exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    (root / "sparse" / "cameras.txt").write_text(
        f"1 PINHOLE {W} {H} {W} {H} {W / 2} {H / 2}\n")
    (root / "sparse" / "images.txt").write_text(f"1 1 0 0 0 0 0 0 1 {name}\n\n")
    (root / "sparse" / "points3D.txt").write_text("")


# ------------------------------------------------------------------ polarity

def test_polarity_auto_is_dataset_level_and_drop_when_mostly_black(tmp_path):
    """96% of real cube-face masks are entirely black (measured 2026-09-02 on
    osmo_playroom/cube4096: 4% of faces nonzero, mean white 0.50%). A per-frame
    decision is meaningless; the median over the sample decides."""
    from metal_gauss import masks
    d = tmp_path / "masks"; d.mkdir()
    for i in range(20):
        a = np.zeros((16, 16), np.uint8)
        if i == 0:
            a[:4] = 255                                  # one frame with an operator
        _write(d / f"f{i:02d}.png", a)
    pol, stats = masks.decide_polarity(d, sample=32)
    assert pol == "drop"
    assert stats["median_white_frac"] == 0.0 and stats["n_sampled"] == 20


def test_polarity_auto_keep_when_mostly_white(tmp_path):
    from metal_gauss import masks
    d = tmp_path / "masks"; d.mkdir()
    for i in range(5):
        a = np.full((16, 16), 255, np.uint8); a[:2] = 0
        _write(d / f"f{i}.png", a)
    assert masks.decide_polarity(d)[0] == "keep"


def test_a_mostly_white_frame_in_a_drop_directory_is_still_read_as_drop(tmp_path):
    """The teeth of 'dataset-level': one frame that would decide 'keep' on its own
    must be interpreted with the DIRECTORY's polarity. A per-frame heuristic
    inverts this frame and trains on the operator instead of the room."""
    from metal_gauss import masks
    d = tmp_path / "masks"; d.mkdir()
    for i in range(9):
        _write(d / f"f{i}.png", np.zeros((16, 16), np.uint8))
    odd = np.full((16, 16), 255, np.uint8); odd[:2] = 0   # 87.5% white on its own
    _write(d / "f9.png", odd)
    pol, _ = masks.decide_polarity(d)
    assert pol == "drop"
    m = masks.load_sidecar_mask(d / "f9.png", size_wh=(16, 16), polarity=pol)
    assert (m[:2] == 255).all() and (m[2:] == 0).all()    # drop: white -> 0


def test_polarity_at_exactly_half_white_is_keep(tmp_path):
    """The threshold is `median < 0.5 -> drop`, so exactly 0.5 falls on the keep side.
    Pinned so the boundary cannot drift silently."""
    from metal_gauss import masks
    d = tmp_path / "masks"; d.mkdir()
    for i in range(4):
        a = np.zeros((16, 16), np.uint8); a[:8] = 255     # exactly half
        _write(d / f"f{i}.png", a)
    pol, stats = masks.decide_polarity(d)
    assert stats["median_white_frac"] == pytest.approx(0.5)
    assert pol == "keep"


def test_polarity_stats_report_how_many_sampled_masks_were_nonzero(tmp_path):
    """A 35-of-276 sample of a set that is 96% all-black can legitimately draw zero
    nonzero masks, and `median_white_frac: 0.0` looks the same either way. The count
    is what lets an operator sanity-check the printed line instead of trusting it."""
    from metal_gauss import masks
    d = tmp_path / "masks"; d.mkdir()
    for i in range(10):
        a = np.zeros((16, 16), np.uint8)
        if i < 3:
            a[:1] = 255
        _write(d / f"f{i}.png", a)
    _, stats = masks.decide_polarity(d)
    assert stats["n_nonzero"] == 3
    assert stats["mean_white_frac"] == pytest.approx(3 * (1 / 16) / 10)


def test_decide_polarity_on_empty_dir_is_an_error(tmp_path):
    from metal_gauss import masks
    d = tmp_path / "masks"; d.mkdir()
    with pytest.raises(ValueError, match="no mask files"):
        masks.decide_polarity(d)


# ------------------------------------------------------------------ sidecar

def test_sidecar_drop_mask_becomes_keep_255_and_resizes_nearest(tmp_path):
    """255 = DROP on disk (every masks*/ dir in earthbyte/slam) -> 255 = KEEP in memory.
    4096-px masks beside 2048-px images exist in the wild (osmo_playroom/ds)."""
    from metal_gauss import masks
    a = np.zeros((32, 32), np.uint8); a[:16] = 255             # top half dropped
    p = _write(tmp_path / "m.png", a)
    m = masks.load_sidecar_mask(p, size_wh=(16, 16), polarity="drop")
    assert m.shape == (16, 16) and m.dtype == np.uint8
    assert set(np.unique(m)) == {0, 255}                       # NEAREST: no grey introduced
    assert (m[:8] == 0).all() and (m[8:] == 255).all()
    k = masks.load_sidecar_mask(p, size_wh=(16, 16), polarity="keep")
    assert (k[:8] == 255).all() and (k[8:] == 0).all()


def test_sidecar_resize_samples_one_pixel_and_does_not_average_the_neighbourhood(tmp_path):
    """The mask a 4x downscale produces must be the value of ONE source pixel, not the
    average of the 16 it covers. Here every 4x4 tile is 15/16 white with a single black
    pixel at the sampled position: NEAREST reads 0 everywhere, while BILINEAR (234-239)
    and LANCZOS (234-240) both threshold to white everywhere -- the OPPOSITE mask.
    Measured with Pillow 12.3 on 2026-09-02; it pins Pillow's pixel-centre phase
    (output x samples input floor((x+0.5)*scale)), which is what makes the tile
    construction land on the black pixel.
    """
    from metal_gauss import masks
    a = np.full((16, 16), 255, np.uint8)
    for ty in range(4):
        for tx in range(4):
            a[ty * 4 + 2, tx * 4 + 2] = 0
    p = _write(tmp_path / "m.png", a)
    m = masks.load_sidecar_mask(p, size_wh=(4, 4), polarity="drop")
    assert set(np.unique(m)) == {255}       # sampled pixel was black -> DROP=0 -> KEEP=255


def test_find_mask_is_ambiguous_when_two_suffixes_exist(tmp_path):
    from metal_gauss import masks
    d = tmp_path / "masks"; d.mkdir()
    _write(d / "v0.png", np.zeros((4, 4), np.uint8))
    _write(d / "v0.jpg", np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError, match="ambiguous"):
        masks.find_mask(d, "v0")


def test_load_sidecar_mask_refuses_unresolved_polarity(tmp_path):
    from metal_gauss import masks
    p = _write(tmp_path / "m.png", np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError, match="polarity"):
        masks.load_sidecar_mask(p, (4, 4), "auto")


# ------------------------------------------------------------------ loader

def test_alpha_baked_rgba_is_kept_by_loader(tmp_path):
    """dataset.py:83 used to .convert('RGB') and discard this. equirect_cube.py:408-415
    convention: a=0 -> DROP, real pixels still present in RGB.

    The RGB assertion is deliberate: the loader must NOT zero masked pixels. Masking
    is the LOSS's job (Task 3); zeroing here would green a red test with the wrong
    behaviour."""
    from metal_gauss.dataset import load_scene
    _colmap_single_view(tmp_path, W=32, H=32)
    rgb = np.zeros((32, 32, 3), np.uint8); rgb[..., 0] = 255
    alpha = np.full((32, 32), 255, np.uint8); alpha[:16] = 0
    _write(tmp_path / "images" / "v0.png", np.dstack([rgb, alpha]))
    sc = load_scene(tmp_path / "sparse", tmp_path / "images", max_resolution=32,
                    eval_split_every=1000)
    v = _only_view(sc)
    assert v.mask is not None and v.mask.dtype == torch.uint8
    assert (v.mask[:16] == 0).all() and (v.mask[16:] == 255).all()
    assert v.image[0, 0].tolist() == [255, 0, 0]            # RGB untouched


def test_rgb_image_without_masks_has_no_mask(tmp_path):
    from metal_gauss.dataset import load_scene
    _colmap_single_view(tmp_path, W=8, H=8)
    _write(tmp_path / "images" / "v0.png", np.zeros((8, 8, 3), np.uint8))
    sc = load_scene(tmp_path / "sparse", tmp_path / "images", max_resolution=8,
                    eval_split_every=1000)
    assert _only_view(sc).mask is None


def test_sidecar_mask_reaches_the_view_at_the_loaded_image_size(tmp_path):
    """The wild case: 32-px masks beside a 16-px loaded image (osmo_playroom/ds is
    4096 vs 2048). The mask must come back at the IMAGE's size, not the file's."""
    from metal_gauss.dataset import load_scene
    _colmap_single_view(tmp_path, W=16, H=16)
    _write(tmp_path / "images" / "v0.png", np.zeros((16, 16, 3), np.uint8))
    md = tmp_path / "masks"; md.mkdir()
    # 25% white, not 50%: at exactly half, `decide_polarity` resolves to "keep"
    # (the threshold is `median < 0.5 -> drop`), which is the boundary pinned above.
    a = np.zeros((32, 32), np.uint8); a[:8] = 255           # top quarter DROP
    _write(md / "v0.png", a)
    sc = load_scene(tmp_path / "sparse", tmp_path / "images", max_resolution=16,
                    eval_split_every=1000, masks_dir=md)
    m = _only_view(sc).mask
    assert m.shape == (16, 16)
    assert (m[:4] == 0).all() and (m[4:] == 255).all()


def test_alpha_and_sidecar_together_is_an_error(tmp_path):
    from metal_gauss.dataset import load_scene
    _colmap_single_view(tmp_path, W=8, H=8)
    _write(tmp_path / "images" / "v0.png", np.zeros((8, 8, 4), np.uint8))
    (tmp_path / "masks").mkdir()
    _write(tmp_path / "masks" / "v0.png", np.zeros((8, 8), np.uint8))
    with pytest.raises(ValueError, match="both"):
        load_scene(tmp_path / "sparse", tmp_path / "images", max_resolution=8,
                   eval_split_every=1000, masks_dir=tmp_path / "masks")


def test_downscaled_mask_is_a_strided_subsample_not_an_average(tmp_path):
    """A half/half mask cannot tell strided-NEAREST from area-interpolate -- the block
    boundaries align and both give the same answer. This one can: each 4x4 block is
    15/16 white with a black pixel exactly where the stride samples. Strided reads 0
    everywhere; `interpolate(mode='area')` reads 239 everywhere, which is neither a
    label nor even binary."""
    from metal_gauss.dataset import View, downscaled
    img = torch.zeros(32, 32, 3, dtype=torch.uint8)
    m = torch.full((32, 32), 255, dtype=torch.uint8)
    m[0::4, 0::4] = 0                                   # exactly the sampled positions
    v = View("x", img, torch.eye(3), torch.eye(4), mask=m)
    d = downscaled(v, 4)
    assert d.mask.shape == (8, 8)
    assert set(d.mask.unique().tolist()) == {0}


def test_downscaled_keeps_none_mask_none(tmp_path):
    from metal_gauss.dataset import View, downscaled
    v = View("x", torch.zeros(16, 16, 3, dtype=torch.uint8), torch.eye(3), torch.eye(4))
    assert downscaled(v, 2).mask is None
