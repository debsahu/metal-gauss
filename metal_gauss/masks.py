"""Mask sources for masked supervision. In memory a mask is (H,W) uint8, 255 = KEEP.

Two conventions exist on disk in earthbyte/slam, on the SAME capture:
  * sidecar `masks/<stem>.png`, 255 = DROP  (pano_pipeline/extract.py:18,
    osmo360/equirect_recipe/colmap_masks.py:32, mask_equirect_sam3.py:24,
    equirect_to_cube_faces.py:100)
  * RGBA alpha baked into the image, a = 0 -> DROP (equirect_cube.py:408-415)

Polarity of a sidecar directory is decided at DATASET level: 96% of real cube-face
masks are entirely black (measured 2026-09-02 on osmo_playroom/cube4096: 4% of faces
nonzero, mean white 0.50%), so any per-frame rule is a coin flip on the other 4%.
Drop-masks are a few percent white at the median; keep-masks are nearly all white
(the same heuristic as ingest/osmo360/equirect_recipe/build_osmo_ds.py).

Getting this backwards does not error: it trains on the operator and the monopod and
discards the room, reports a plausible loss curve, and reads `mean coverage` near 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

MASK_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
POLARITIES = ("auto", "drop", "keep")


def find_mask(masks_dir: str | Path, stem: str) -> Path | None:
    d = Path(masks_dir)
    hits = [d / f"{stem}{s}" for s in MASK_SUFFIXES if (d / f"{stem}{s}").exists()]
    if len(hits) > 1:
        raise ValueError(f"ambiguous mask for {stem!r}: {[str(h) for h in hits]}")
    return hits[0] if hits else None


def white_fraction(path: str | Path) -> float:
    a = np.asarray(Image.open(path).convert("L"))
    return float((a > 127).mean())


def decide_polarity(masks_dir: str | Path, sample: int = 32) -> tuple[str, dict]:
    """('drop'|'keep', stats) for a whole directory, from an evenly-spaced sample.

    The verdict is the MEDIAN white fraction: a drop-set is mostly black, a keep-set
    mostly white, and the two are separated by an enormous margin (0.5% vs ~99%), so
    the exact threshold is not delicate. The sample is thin on purpose -- these are
    4096-px PNGs and reading all 276 of them costs half a minute -- which means a set
    that is 96% all-black can legitimately draw zero nonzero masks. That does not
    change the verdict, but it does make `median_white_frac: 0.0` ambiguous between
    "a drop-set" and "I sampled nothing interesting", so `n_nonzero` and
    `mean_white_frac` are reported for the operator to sanity-check the printed line.
    """
    files = sorted(p for p in Path(masks_dir).iterdir()
                   if p.suffix.lower() in MASK_SUFFIXES)
    if not files:
        raise ValueError(f"no mask files in {masks_dir}")
    step = max(1, len(files) // sample)
    fr = [white_fraction(p) for p in files[::step]]
    med = float(np.median(fr))
    stats = {"n_files": len(files), "n_sampled": len(fr),
             "n_nonzero": int(sum(f > 0.0 for f in fr)),
             "median_white_frac": med, "mean_white_frac": float(np.mean(fr)),
             "min_white_frac": float(min(fr)), "max_white_frac": float(max(fr))}
    return ("drop" if med < 0.5 else "keep"), stats


def load_sidecar_mask(path: str | Path, size_wh: tuple[int, int],
                      polarity: str) -> np.ndarray:
    """One sidecar mask as (H,W) uint8, 255 = KEEP, at exactly `size_wh`.

    Resize is NEAREST because a mask is labels, not radiance. AREA or LANCZOS on a
    4096 -> 2048 downscale blends a one-pixel drop stripe into grey and the threshold
    then erases it; NEAREST subsamples it and the stripe survives.
    """
    if polarity not in ("drop", "keep"):
        raise ValueError(f"polarity must be resolved to drop/keep, got {polarity!r}")
    im = Image.open(path).convert("L")
    if im.size != tuple(size_wh):
        im = im.resize(tuple(size_wh), Image.NEAREST)
    a = np.asarray(im)
    keep = (a <= 127) if polarity == "drop" else (a > 127)
    return np.where(keep, 255, 0).astype(np.uint8)


def alpha_to_mask(alpha_u8: np.ndarray) -> np.ndarray:
    """RGBA alpha channel -> keep mask. a = 0 is DROP; anything else keeps."""
    return np.where(alpha_u8 > 0, 255, 0).astype(np.uint8)
