"""COLMAP scene loader for training: images, poses, intrinsics, eval split."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from metal_gauss.masks import alpha_to_mask, decide_polarity, find_mask, load_sidecar_mask
from metal_gauss.priors import load_view_priors, resolve_dirs


@dataclass
class View:
    name: str
    image: torch.Tensor      # (H,W,3) uint8, cpu
    K: torch.Tensor          # (3,3)
    viewmat: torch.Tensor    # (4,4) world2cam
    mask: torch.Tensor | None = None    # (H,W) uint8, 255 = KEEP, cpu
    depth: torch.Tensor | None = None   # (H,W) uint16 mm (0 = invalid), or float32 metres
    normal: torch.Tensor | None = None  # (H,W,3) uint8 codes (128 = invalid), or float32
    # Downscale cache, per view. It used to be a module-level dict keyed by `id(v)`, which
    # is unsafe: CPython reuses a freed object's address for the next allocation of the
    # same size, so a View created after another was collected inherited its cached
    # downscale -- wrong image, wrong intrinsics, and since Task 5, wrong PRIORS.
    # Reproduced on trial 2 of 200. Holding the cache on the view makes the collision
    # impossible by construction and lets it die with the view.
    _pyramid: dict = field(default_factory=dict, repr=False, compare=False)


def downscaled(v: View, factor: int) -> View:
    """A view at 1/factor resolution, cached.

    Training the early steps at reduced resolution is most of why msplat is
    fast: its 7k average is 24.5 ms/step against 46 ms at full resolution. It
    is not a benchmarking trick either -- forced to full resolution its own
    held-out score DROPS (21.37 -> 20.78), so the coarse-to-fine schedule is
    doing real optimisation work, not just cheaper steps.

    Images are cached as uint8 and downsampled once per (view, factor); the
    pyramid costs ~33% more memory than the full-resolution set alone. The
    intrinsics scale with the image, and the principal point scales too -- it
    is not simply W/2 for a COLMAP camera.
    """
    if factor <= 1:
        return v
    hit = v._pyramid.get(factor)
    if hit is not None:
        return hit

    H, W = v.image.shape[:2]
    h, w = max(1, H // factor), max(1, W // factor)
    img = torch.nn.functional.interpolate(
        v.image.permute(2, 0, 1)[None].float(), size=(h, w),
        mode="area")[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)

    sx, sy = w / W, h / H
    K = v.K.clone()
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy

    # Strided NEAREST on the mask, never `interpolate`: a mask is labels, and an
    # area-averaged label is not a label. The slice bounds give exactly (h, w).
    # Strided NEAREST on mask and priors, never `interpolate`: a mask is labels, and an
    # area-averaged depth blends the 0 invalid-sentinel into its valid neighbours, which
    # invents a measurement the sensor never made. The slice bounds give exactly (h, w).
    def _sub(t):
        return None if t is None else t[:h * factor:factor, :w * factor:factor].contiguous()
    out = View(v.name, img.contiguous(), K, v.viewmat,
               mask=_sub(v.mask), depth=_sub(v.depth), normal=_sub(v.normal))
    v._pyramid[factor] = out
    return out


@dataclass
class Scene:
    train: list[View]
    heldout: list[View]
    points: np.ndarray       # (P,3) sparse init
    colors: np.ndarray       # (P,3) in [0,1]


def load_scene(colmap_dir: str | Path, images_dir: str | Path,
               max_resolution: int = 1600, eval_split_every: int = 8,
               masks_dir: str | Path | None = None,
               mask_polarity: str = "auto",
               depth_dir: str | Path | None = None,
               normal_dir: str | Path | None = None,
               prior_resident: str = "quantized",
               use_priors: bool = True) -> Scene:
    import pycolmap
    from PIL import Image

    rec = pycolmap.Reconstruction(str(colmap_dir))
    cam = list(rec.cameras.values())[0]

    # Resolved ONCE for the directory, before any frame is read -- see masks.py.
    polarity, mask_stats = "drop", None
    if masks_dir is not None:
        if mask_polarity == "auto":
            polarity, mask_stats = decide_polarity(masks_dir)
        else:
            polarity = mask_polarity
        print(f"masks: {masks_dir} polarity={polarity} {mask_stats}")

    d_dir, n_dir = resolve_dirs(images_dir, depth_dir, normal_dir, enabled=use_priors)
    if d_dir is not None or n_dir is not None:
        print(f"priors: depth={d_dir or 'none'} normal={n_dir or 'none'} "
              f"resident={prior_resident}")
    n_depth = n_normal = 0

    views = []
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        p = Path(images_dir) / im.name
        if not p.exists():
            continue
        # Capture the alpha BEFORE .convert("RGB") -- that call silently discards a
        # baked mask, which is how an alpha-masked dataset trains unmasked.
        src = Image.open(p)
        alpha = src.getchannel("A") if src.mode in ("RGBA", "LA") else None
        img = src.convert("RGB")
        scale = min(1.0, max_resolution / max(img.size))
        W, H = int(round(img.width * scale)), int(round(img.height * scale))
        img = img.resize((W, H), Image.LANCZOS)

        side = find_mask(masks_dir, Path(im.name).stem) if masks_dir is not None else None
        if alpha is not None and side is not None:
            raise ValueError(
                f"{im.name}: both a baked alpha channel and a sidecar mask {side} "
                f"-- pick one source")
        mask = None
        if alpha is not None:
            mask = alpha_to_mask(np.asarray(alpha.resize((W, H), Image.NEAREST)))
        elif side is not None:
            mask = load_sidecar_mask(side, (W, H), polarity)

        depth, normal = load_view_priors(Path(im.name).stem, (W, H), d_dir, n_dir,
                                        resident=prior_resident)
        n_depth += depth is not None
        n_normal += normal is not None

        sx, sy = W / cam.width, H / cam.height
        K = torch.tensor([[cam.params[0] * sx, 0, cam.params[2] * sx],
                          [0, cam.params[1] * sy, cam.params[3] * sy],
                          [0, 0, 1.0]], dtype=torch.float32)
        cfw = im.cam_from_world()
        vm = torch.eye(4)
        vm[:3, :3] = torch.as_tensor(cfw.rotation.matrix(), dtype=torch.float32)
        vm[:3, 3] = torch.as_tensor(np.asarray(cfw.translation), dtype=torch.float32)
        # uint8 storage: 162 full-res frames as float32 was 4x the memory and
        # OOMed a 24GB M5 alongside the 1M-splat Adam state. Convert per step.
        views.append(View(im.name,
                          torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()),
                          K, vm,
                          mask=None if mask is None else torch.from_numpy(mask),
                          depth=depth, normal=normal))

    if d_dir is not None or n_dir is not None:
        print(f"priors attached: depth {n_depth}/{len(views)}, normal {n_normal}/{len(views)}")

    heldout = views[::eval_split_every]
    heldout_names = {v.name for v in heldout}
    train = [v for v in views if v.name not in heldout_names]

    pts = np.array([p.xyz for p in rec.points3D.values()], np.float32)
    cols = np.array([p.color for p in rec.points3D.values()], np.float32) / 255.0
    return Scene(train, heldout, pts, cols)
