"""depth/ and normal/ prior discovery + loading, on the prior_io contract.

Discovery is by STEM (`prior_io.find_prior`; two files for one stem is a hard error, because
a dataset caught mid-format-migration can hold `x.tiff` beside `x.png` and picking either
silently is how a run trains on stale supervision).

SIZE IS CHECKED AGAINST THE SIZE THE IMAGE LOADED AT, AND A MISMATCH IS A HARD ERROR.
Never resize a prior. Area interpolation smears the 0 invalid-sentinel into valid pixels,
and a resized depth map is a different measurement, not the same one at another scale.
Brush panics here roughly 400 iterations in, after the dataset scan, which is a much worse
place to find out; we refuse at load and name both sizes.

RESIDENCY. Whatever the disk format, priors live in RAM as png-quantized codes -- depth
uint16 millimetres, normals uint8 -- so 5 bytes per pixel instead of 16. On the P-GEOM set
(196 x 1920x1440) that is 2.7 GB against 8.7 GB. The quantization is training-equivalent
per the WS-G gate: 0.0128 dB against the trainer's own 0.0317 dB same-seed repeat
(CLAUDE.md, "Prior compression"). A float32-TIFF prior is therefore NOT bit-preserved in
RAM; `--prior-resident float32` is the escape hatch and is deliberately not the default.

MPS NOTE (torch 2.13, measured 2026-09-02): `uint16 == scalar` is unsupported on MPS
(`eq_dense_scalar_cast_bool_ushort`). `decode_depth` therefore uses only a cast and a
divide, which do work there; `decode_normal`'s sentinel comparison is on uint8, which also
works. Do not add a uint16 comparison to this module.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from metal_gauss import prior_io


class PriorSizeError(RuntimeError):
    """A prior does not match the size its image loaded at. Never resize; regenerate."""


def resolve_dirs(images_dir, depth_dir, normal_dir,
                 enabled: bool = True) -> tuple[Path | None, Path | None]:
    """Explicit flags win; otherwise the siblings `<images>/../{depth,normal}` if they exist.

    `enabled=False` (the trainer's `--no-priors`) turns the whole mechanism off. Without it
    the sibling auto-detect makes a bare run on a dataset that HAS priors refuse at any
    `--max-resolution` below the prior size, and the only escape is inventing an empty
    directory to point the flag at. Combining it with an explicit `--depth-dir` is a
    contradiction and is refused rather than silently resolved one way.

    `abspath`, NOT `resolve()`. A dataset's `images/` is very often a symlink into another
    tree (P-GEOM's points at an entirely different capture directory), and `resolve()`
    dereferences it, so the "sibling" becomes a directory beside the LINK TARGET and the
    depth/ sitting right next to the dataset is never found. That failure is silent: the
    run trains with no priors and prints nothing, because there is nothing to print.
    `abspath` normalises lexically -- absolute, `..` collapsed, symlinks intact.
    """
    if not enabled:
        if depth_dir is not None or normal_dir is not None:
            raise ValueError(
                "--no-priors was given together with an explicit --depth-dir/--normal-dir; "
                "pick one -- refusing to guess which the operator meant")
        return None, None
    root = Path(os.path.abspath(images_dir)).parent
    d = Path(depth_dir) if depth_dir else (root / "depth" if (root / "depth").is_dir() else None)
    n = Path(normal_dir) if normal_dir else (root / "normal" if (root / "normal").is_dir() else None)
    return d, n


def _check(arr: np.ndarray, size_wh: tuple[int, int], path: Path, kind: str) -> None:
    W, H = size_wh
    if arr.shape[0] != H or arr.shape[1] != W:
        raise PriorSizeError(
            f"{kind} prior {path} is {arr.shape[1]}x{arr.shape[0]} but the image loaded at "
            f"{W}x{H}. Priors are never resized: regenerate them at the loaded size, or "
            f"raise --max-resolution so the image loads at the prior's size.")


def load_view_priors(stem: str, size_wh: tuple[int, int], depth_dir, normal_dir,
                     resident: str = "quantized"):
    """(depth, normal) for one view, or (None, None). Quantized codes unless overridden."""
    if resident not in ("quantized", "float32"):
        raise ValueError(f"resident must be quantized or float32, got {resident!r}")
    depth = normal = None
    if depth_dir is not None:
        p = prior_io.find_prior(depth_dir, stem)
        if p is not None:
            arr = prior_io.read_depth(p)
            _check(arr, size_wh, p, "depth")
            depth = (torch.from_numpy(prior_io.encode_depth_u16mm(arr))
                     if resident == "quantized" else torch.from_numpy(arr.copy()))
    if normal_dir is not None:
        p = prior_io.find_prior(normal_dir, stem)
        if p is not None:
            arr = prior_io.read_normal(p)
            _check(arr, size_wh, p, "normal")
            normal = (torch.from_numpy(prior_io.encode_normal_u8(arr))
                      if resident == "quantized" else torch.from_numpy(arr.copy()))
    return depth, normal


def decode_depth(t: torch.Tensor) -> torch.Tensor:
    """uint16 mm -> float32 metres, 0 = invalid. Op order pinned to
    `prior_io.decode_depth_u16mm` (`f32(mm) / 1000.0`), which Rust matches bit-for-bit on
    every one of the 65536 codes. A float32-resident tensor passes through untouched."""
    if t.dtype != torch.uint16:
        return t
    return t.to(torch.float32) / 1000.0


def decode_normal(t: torch.Tensor) -> torch.Tensor:
    """uint8 codes -> float32 unit normals. Code 128 -- and only 128 -- decodes to exactly
    0.0, PER CHANNEL, matching `prior_io.decode_normal_u8`. Per-channel matters: a normal
    with one zero component keeps its other two."""
    if t.dtype != torch.uint8:
        return t
    f = t.to(torch.float32) / 255.0 * 2.0 - 1.0
    return torch.where(t == 128, torch.zeros_like(f), f)
