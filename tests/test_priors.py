"""depth/ + normal/ prior loading on the prior_io contract.

Two rules this file exists to enforce, both of which cost real runs elsewhere:
  * a prior whose size differs from the size the IMAGE LOADED AT is a hard error, never
    a resize (Brush panics ~400 iterations in; we refuse at load);
  * the pyramid subsamples priors by STRIDE, never by interpolation -- an area-averaged
    depth blends the 0 invalid-sentinel into its valid neighbours and invents geometry.
"""
import numpy as np
import pytest
import torch
from PIL import Image


def _only_view(scene):
    vs = scene.train + scene.heldout
    assert len(vs) == 1, f"expected exactly one view, got {len(vs)}"
    return vs[0]


def _sparse(root, W, H):
    (root / "sparse").mkdir(exist_ok=True); (root / "images").mkdir(exist_ok=True)
    (root / "sparse" / "cameras.txt").write_text(f"1 PINHOLE {W} {H} {W} {H} {W / 2} {H / 2}\n")
    (root / "sparse" / "images.txt").write_text("1 1 0 0 0 0 0 0 1 v0.png\n\n")
    (root / "sparse" / "points3D.txt").write_text("")
    Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(root / "images" / "v0.png")


# ------------------------------------------------------------------ codec contract

def test_mixed_formats_load_and_decode_to_the_codec_contract(tmp_path):
    """Formats may mix per frame -- Brush dispatches on magic bytes, not extension, and a
    real dataset (cube3840v3) shipped float32-TIFF depth beside png-quantized normals."""
    from metal_gauss import prior_io, priors
    d, n = tmp_path / "depth", tmp_path / "normal"; d.mkdir(); n.mkdir()
    depth = np.full((6, 8), 1.2345, np.float32); depth[0, 0] = 0.0     # one invalid pixel
    prior_io.write_depth(d / "v0.tiff", depth, "tiff-f32")             # TIFF on disk...
    nrm = np.zeros((6, 8, 3), np.float32); nrm[..., 2] = -1.0; nrm[1, 1] = 0  # ...PNG on disk
    prior_io.write_normal(n / "v0.png", nrm, "png-quantized")
    dq, nq = priors.load_view_priors("v0", (8, 6), d, n)
    assert dq.dtype == torch.uint16 and nq.dtype == torch.uint8
    df = priors.decode_depth(dq); nf = priors.decode_normal(nq)
    assert df[0, 0] == 0.0 and abs(df[3, 3].item() - 1.2345) <= 0.0005 + 1e-6   # <= 0.5 mm
    assert torch.allclose(nf[2, 2], torch.tensor([0., 0., -1.]))
    assert (nf[1, 1] == 0).all()                                       # sentinel exact


def test_decoders_match_prior_io_bit_for_bit_over_the_whole_code_range(tmp_path):
    """The torch decoders are a REIMPLEMENTATION of the numpy codec. If they drift, the two
    trainers disagree about what a prior means and nothing errors. Checked over all 65536
    depth codes and all 256 normal codes, not on anchors."""
    from metal_gauss import prior_io, priors
    codes = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    want = prior_io.decode_depth_u16mm(codes)
    got = priors.decode_depth(torch.from_numpy(codes)).numpy()
    assert np.array_equal(got, want)
    nc = np.stack(np.meshgrid(np.arange(256), np.arange(256), indexing="ij"), -1)
    nc = np.concatenate([nc, np.full((256, 256, 1), 128)], -1).astype(np.uint8)
    assert np.array_equal(priors.decode_normal(torch.from_numpy(nc)).numpy(),
                          prior_io.decode_normal_u8(nc))


def test_normal_code_128_decodes_to_exactly_zero_per_channel():
    """Not per pixel. `decode_normal_u8` zeroes the CHANNEL whose code is 128, so a normal
    with one zero component keeps its other two. A per-pixel rule silently deletes them."""
    from metal_gauss import priors
    codes = torch.tensor([[[128, 255, 0]]], dtype=torch.uint8)
    out = priors.decode_normal(codes)
    assert out[0, 0, 0].item() == 0.0
    assert out[0, 0, 1].item() == pytest.approx(1.0)
    assert out[0, 0, 2].item() == pytest.approx(-1.0)


# ------------------------------------------------------------------ hard size check

def test_size_mismatch_is_a_hard_error_naming_both_sizes(tmp_path):
    from metal_gauss import prior_io, priors
    d = tmp_path / "depth"; d.mkdir()
    prior_io.write_depth(d / "v0.png", np.ones((12, 16), np.float32), "png-quantized")
    with pytest.raises(priors.PriorSizeError, match=r"16x12.*8x6.*max-resolution"):
        priors.load_view_priors("v0", (8, 6), d, None)


def test_size_mismatch_on_the_NORMAL_prior_is_also_a_hard_error(tmp_path):
    """The depth check is the one everybody writes; the normal check is the one that gets
    forgotten, and a wrong-size normal panics Brush at the same place."""
    from metal_gauss import prior_io, priors
    n = tmp_path / "normal"; n.mkdir()
    nrm = np.zeros((12, 16, 3), np.float32); nrm[..., 2] = -1.0
    prior_io.write_normal(n / "v0.png", nrm, "png-quantized")
    with pytest.raises(priors.PriorSizeError, match=r"normal.*16x12.*8x6"):
        priors.load_view_priors("v0", (8, 6), None, n)


def test_a_prior_that_is_one_pixel_off_is_still_refused(tmp_path):
    """Off-by-one is the realistic failure (a rounding difference in the resize), and it is
    the one a tolerant check would wave through."""
    from metal_gauss import prior_io, priors
    d = tmp_path / "depth"; d.mkdir()
    prior_io.write_depth(d / "v0.png", np.ones((6, 9), np.float32), "png-quantized")
    with pytest.raises(priors.PriorSizeError):
        priors.load_view_priors("v0", (8, 6), d, None)


def test_a_transposed_prior_is_refused(tmp_path):
    """Same pixel COUNT, wrong shape -- a generator that wrote (W,H) instead of (H,W).
    A check on `arr.size != H*W` waves this through, and the prior then supervises every
    pixel against the depth of a different pixel."""
    from metal_gauss import prior_io, priors
    d = tmp_path / "depth"; d.mkdir()
    prior_io.write_depth(d / "v0.png", np.ones((8, 6), np.float32), "png-quantized")
    with pytest.raises(priors.PriorSizeError, match=r"6x8.*8x6"):
        priors.load_view_priors("v0", (8, 6), d, None)


def test_ambiguous_stem_is_an_error(tmp_path):
    from metal_gauss import prior_io, priors
    d = tmp_path / "depth"; d.mkdir()
    prior_io.write_depth(d / "v0.png", np.ones((6, 8), np.float32), "png-quantized")
    prior_io.write_depth(d / "v0.tiff", np.ones((6, 8), np.float32), "tiff-f32")
    with pytest.raises(prior_io.PriorFormatError, match="ambiguous"):
        priors.load_view_priors("v0", (8, 6), d, None)


def test_missing_prior_is_none_not_error(tmp_path):
    from metal_gauss import priors
    d = tmp_path / "depth"; d.mkdir()
    assert priors.load_view_priors("v0", (8, 6), d, None) == (None, None)


# ------------------------------------------------------------------ residency

def test_float32_residency_escape_hatch_preserves_bits(tmp_path):
    """--prior-resident float32 must be lossless; the default is not, by design."""
    from metal_gauss import prior_io, priors
    d = tmp_path / "depth"; d.mkdir()
    depth = np.array([[1.23456789, 0.0], [2.0, 65.6]], np.float32)
    prior_io.write_depth(d / "v0.tiff", depth, "tiff-f32")
    dq, _ = priors.load_view_priors("v0", (2, 2), d, None, resident="float32")
    assert dq.dtype == torch.float32
    assert np.array_equal(priors.decode_depth(dq).numpy(), depth)
    dqz, _ = priors.load_view_priors("v0", (2, 2), d, None)          # default: quantized
    assert dqz.dtype == torch.uint16
    assert not np.array_equal(priors.decode_depth(dqz).numpy(), depth)


def test_quantized_residency_is_within_half_a_millimetre(tmp_path):
    from metal_gauss import prior_io, priors
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 10.0, (16, 16)).astype(np.float32)
    d = tmp_path / "depth"; d.mkdir()
    prior_io.write_depth(d / "v0.tiff", depth, "tiff-f32")
    dq, _ = priors.load_view_priors("v0", (16, 16), d, None)
    err = np.abs(priors.decode_depth(dq).numpy() - depth).max()
    assert err <= 0.0005 + 1e-6, err


# ------------------------------------------------------------------ dir resolution

def test_resolve_dirs_prefers_explicit_then_siblings_then_none(tmp_path):
    from metal_gauss import priors
    (tmp_path / "images").mkdir(); (tmp_path / "depth").mkdir()
    explicit = tmp_path / "elsewhere"; explicit.mkdir()
    d, n = priors.resolve_dirs(tmp_path / "images", None, None)
    assert d == tmp_path / "depth" and n is None            # sibling found, normal absent
    d, n = priors.resolve_dirs(tmp_path / "images", explicit, None)
    assert d == explicit                                     # explicit beats the sibling
    (tmp_path / "depth").rmdir()
    assert priors.resolve_dirs(tmp_path / "images", None, None) == (None, None)


# ------------------------------------------------------------------ loader + pyramid

def test_sibling_autodetect_survives_a_SYMLINKED_images_dir(tmp_path):
    """P-GEOM's `images` is a symlink into another tree entirely. `.resolve()` follows it,
    so the "sibling" becomes a directory of the LINK TARGET's parent and the depth/ beside
    the dataset is never found -- the run then trains with no priors and says nothing.
    Use `os.path.abspath()`, which normalises lexically without dereferencing."""
    from metal_gauss import priors
    real = tmp_path / "elsewhere" / "frames"; real.mkdir(parents=True)
    ds = tmp_path / "ds"; ds.mkdir()
    (ds / "depth").mkdir(); (ds / "normal").mkdir()
    (ds / "images").symlink_to(real, target_is_directory=True)
    d, n = priors.resolve_dirs(ds / "images", None, None)
    assert d == (ds / "depth").absolute(), f"symlink defeated sibling autodetect: {d}"
    assert n == (ds / "normal").absolute()


def test_sibling_autodetect_works_from_a_relative_images_path(tmp_path, monkeypatch):
    """`Path('images').parent` is `.`, which has no depth/ sibling. Normalisation must
    happen (via `os.path.abspath`), just without dereferencing."""
    from metal_gauss import priors
    (tmp_path / "images").mkdir(); (tmp_path / "depth").mkdir()
    monkeypatch.chdir(tmp_path)
    d, _ = priors.resolve_dirs("images", None, None)
    assert d == (tmp_path / "depth").absolute()


def test_scene_loader_attaches_priors_and_pyramid_is_strided(tmp_path):
    from metal_gauss import prior_io
    from metal_gauss.dataset import downscaled, load_scene
    _sparse(tmp_path, 16, 8)
    (tmp_path / "depth").mkdir()
    depth = np.arange(8 * 16, dtype=np.float32).reshape(8, 16) / 100 + 1.0
    depth[0, 1] = 0.0
    prior_io.write_depth(tmp_path / "depth" / "v0.png", depth, "png-quantized")
    sc = load_scene(tmp_path / "sparse", tmp_path / "images", max_resolution=16,
                    eval_split_every=1000, depth_dir=tmp_path / "depth")
    v = _only_view(sc)
    assert v.depth is not None and v.depth.shape == (8, 16)
    d2 = downscaled(v, 2)
    assert d2.depth.shape == (4, 8)
    assert torch.equal(d2.depth, v.depth[::2, ::2])          # strided NEAREST, sentinel-safe


def test_pyramid_never_blends_the_zero_sentinel_into_a_valid_neighbour(tmp_path):
    """The reason stride is mandatory. Here every sampled position is INVALID (depth 0) and
    its 3 neighbours are 5 m. Strided keeps 0 -- 'no measurement here'. Area interpolation
    returns 3.75 m, a depth that was never measured, and the depth loss then supervises
    every splat on that ray toward it."""
    from metal_gauss import prior_io
    from metal_gauss.dataset import downscaled, load_scene
    _sparse(tmp_path, 16, 8)
    (tmp_path / "depth").mkdir()
    depth = np.full((8, 16), 5.0, np.float32)
    depth[::2, ::2] = 0.0
    prior_io.write_depth(tmp_path / "depth" / "v0.png", depth, "png-quantized")
    sc = load_scene(tmp_path / "sparse", tmp_path / "images", max_resolution=16,
                    eval_split_every=1000, depth_dir=tmp_path / "depth")
    d2 = downscaled(_only_view(sc), 2)
    assert (d2.depth == 0).all(), "the invalid sentinel was blended away"


def test_normal_prior_reaches_the_view_and_survives_the_pyramid(tmp_path):
    from metal_gauss import prior_io
    from metal_gauss.dataset import downscaled, load_scene
    _sparse(tmp_path, 16, 8)
    (tmp_path / "normal").mkdir()
    nrm = np.zeros((8, 16, 3), np.float32); nrm[..., 2] = -1.0; nrm[0, 0] = 0.0
    prior_io.write_normal(tmp_path / "normal" / "v0.png", nrm, "png-quantized")
    sc = load_scene(tmp_path / "sparse", tmp_path / "images", max_resolution=16,
                    eval_split_every=1000, normal_dir=tmp_path / "normal")
    v = _only_view(sc)
    assert v.normal is not None and v.normal.shape == (8, 16, 3) and v.normal.dtype == torch.uint8
    d2 = downscaled(v, 2)
    assert d2.normal.shape == (4, 8, 3)
    assert torch.equal(d2.normal, v.normal[::2, ::2])


def test_loader_refuses_a_prior_that_does_not_match_the_LOADED_image_size(tmp_path):
    """End to end, the failure the plan cares about: --max-resolution silently downscales
    the image, and the prior beside it is then the wrong size. Refuse; never resize."""
    from metal_gauss import prior_io
    from metal_gauss.dataset import load_scene
    from metal_gauss.priors import PriorSizeError
    _sparse(tmp_path, 16, 8)
    (tmp_path / "depth").mkdir()
    prior_io.write_depth(tmp_path / "depth" / "v0.png", np.ones((8, 16), np.float32),
                         "png-quantized")
    with pytest.raises(PriorSizeError, match="max-resolution"):
        load_scene(tmp_path / "sparse", tmp_path / "images", max_resolution=8,
                   eval_split_every=1000, depth_dir=tmp_path / "depth")


# ------------------------------------------------------------------ opt-out + MPS

def test_no_priors_disables_sibling_autodetect(tmp_path):
    """Sibling auto-detect means a bare run on a dataset that HAS priors hard-refuses at any
    --max-resolution below the prior size. There has to be a way to say "train without
    priors" that is not "point the flag at an empty directory I made up"; the no-prior
    control arm in the measurement tier needs exactly this."""
    from metal_gauss import priors
    (tmp_path / "images").mkdir(); (tmp_path / "depth").mkdir(); (tmp_path / "normal").mkdir()
    assert priors.resolve_dirs(tmp_path / "images", None, None) != (None, None)
    assert priors.resolve_dirs(tmp_path / "images", None, None, enabled=False) == (None, None)


def test_no_priors_with_an_explicit_dir_is_a_contradiction(tmp_path):
    """Silently honouring one and dropping the other is how a run trains with supervision
    the operator thought they had turned off."""
    from metal_gauss import priors
    (tmp_path / "images").mkdir(); (tmp_path / "depth").mkdir()
    with pytest.raises(ValueError, match="no-priors"):
        priors.resolve_dirs(tmp_path / "images", tmp_path / "depth", None, enabled=False)


def test_scene_loader_honours_the_opt_out_on_a_dataset_that_has_priors(tmp_path):
    from metal_gauss import prior_io
    from metal_gauss.dataset import load_scene
    _sparse(tmp_path, 16, 8)
    (tmp_path / "depth").mkdir()
    prior_io.write_depth(tmp_path / "depth" / "v0.png", np.ones((4, 8), np.float32),
                         "png-quantized")          # deliberately the WRONG size
    with pytest.raises(Exception):                 # would refuse, were priors enabled
        load_scene(tmp_path / "sparse", tmp_path / "images", max_resolution=16,
                   eval_split_every=1000)
    sc = load_scene(tmp_path / "sparse", tmp_path / "images", max_resolution=16,
                   eval_split_every=1000, use_priors=False)
    assert _only_view(sc).depth is None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_decoders_run_on_real_mps_tensors():
    """Priors are decoded AFTER `.to(device)` in the training loop, and MPS is not CPU:
    torch 2.13 has no `uint16 == scalar` there (`eq_dense_scalar_cast_bool_ushort`), which
    is why `decode_depth` is a cast and a divide and only `decode_normal` compares -- on
    uint8, which does work. A CPU-only test cannot see any of that."""
    from metal_gauss import priors
    d = torch.tensor([[0, 1234, 65535]], dtype=torch.uint16).to("mps")
    out = priors.decode_depth(d)
    assert out.device.type == "mps" and out.dtype == torch.float32
    assert out.cpu().tolist()[0] == pytest.approx([0.0, 1.234, 65.535], abs=1e-5)
    n = torch.tensor([[[128, 255, 0]]], dtype=torch.uint8).to("mps")
    dn = priors.decode_normal(n).cpu()
    assert dn[0, 0, 0].item() == 0.0
    assert dn[0, 0, 1].item() == pytest.approx(1.0)
    assert dn[0, 0, 2].item() == pytest.approx(-1.0)
