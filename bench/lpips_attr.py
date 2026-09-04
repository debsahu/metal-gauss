"""Task 21: shared machinery for the Stage 4 LPIPS attribution.

Three post-hoc photometric fitters, the Brush forms they reproduce, and the
scoring convention every number in this task is computed under. NOTHING HERE
TRAINS: the fitters operate on frozen rendered PNGs and never touch a splat.

Provenance of the forms, all Apache-2.0 Brush (compute/brush):
  bilateral grid   brush-appearance/src/bilagrid_kernels.rs:15-17,37-70,84-93,
                   120-152  (BT.601 guidance, aligned corners, border padding,
                   row-major 3x4 affine per cell)
                   brush-appearance/src/bilagrid.rs:340-348 (TV), :364-382
                   (identity init: channels 0, 5, 10)
                   brush-train/src/config.rs:996-1031 (dims 16,16,8; tv 10.0;
                   lr 2e-3; betas 0.9,0.999)
  PPISP            brush-appearance/src/ppisp_math.rs:330-347 (vignetting),
                   :360-382 (CRF), :386-397 (uv)
                   brush-appearance/src/ppisp_kernels.rs:75-168 (stage order)

WHY THE SCORING ROUNDS. The published LPIPS for every arm was computed by
scripts/lpips_eval.py off the uint8 PNGs in its --eval-dump. A fitted image is
therefore clamped and rounded the same way before scoring, so that a gain
existing only in un-rounded float -- which the delivery pipeline could never
realise -- is not counted as one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

KIND = "lpips_attribution"
SCHEMA = 1

# Brush defaults, brush-train/src/config.rs:996-1031.
BILAGRID_DIMS = (16, 16, 8)          # (x, y, guidance)
BILAGRID_TV_WEIGHT = 10.0
BILAGRID_LR = 2e-3
BILAGRID_BETAS = (0.9, 0.999)
BILAGRID_STEPS = 500

LUMA = (0.299, 0.587, 0.114)         # BT.601, bilagrid_kernels.rs:15-17


# --------------------------------------------------------------- bilateral grid

def bilagrid_identity(gl: int, gh: int, gw: int, device="cpu") -> torch.Tensor:
    """`[12, L, H, W]` identity affine: the diagonal of a ROW-MAJOR 3x4 is
    channels 0, 5 and 10 (bilagrid.rs:373-377)."""
    g = torch.zeros(12, gl, gh, gw, device=device)
    for c in (0, 5, 10):
        g[c] = 1.0
    return g


def _trilinear_coeffs(grid: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """The 12 affine coefficients per pixel, `[H, W, 12]`.

    Transcribed from `sample_point` / `interpolate`: aligned corners on x and y,
    BT.601 luminance scaled to `[0, L-1]` on the guidance axis, and corners
    clamped to the last cell (border padding).
    """
    c, gl, gh, gw = grid.shape
    assert c == 12, grid.shape
    h, w, _ = rgb.shape
    dev = rgb.device
    px = torch.arange(w, device=dev, dtype=rgb.dtype)
    py = torch.arange(h, device=dev, dtype=rgb.dtype)
    x = px * (gw - 1) / max(w - 1, 1)
    y = py * (gh - 1) / max(h - 1, 1)
    lum = LUMA[0] * rgb[..., 0] + LUMA[1] * rgb[..., 1] + LUMA[2] * rgb[..., 2]
    z = (lum * (gl - 1)).clamp(0.0, float(gl - 1))

    x0 = x.floor(); y0 = y.floor(); z0 = z.floor()
    tx = (x - x0)[None, :].expand(h, w)
    ty = (y - y0)[:, None].expand(h, w)
    tz = z - z0
    x0i = x0.long()[None, :].expand(h, w)
    y0i = y0.long()[:, None].expand(h, w)
    z0i = z0.long()
    x1i = (x0i + 1).clamp(max=gw - 1)
    y1i = (y0i + 1).clamp(max=gh - 1)
    z1i = (z0i + 1).clamp(max=gl - 1)

    flat = grid.reshape(12, -1)
    cells = gl * gh * gw
    out = torch.zeros(h, w, 12, device=dev, dtype=rgb.dtype)
    for corner in range(8):
        cx = x1i if corner & 1 else x0i
        cy = y1i if corner & 2 else y0i
        cz = z1i if corner & 4 else z0i
        wx = tx if corner & 1 else 1.0 - tx
        wy = ty if corner & 2 else 1.0 - ty
        wz = tz if corner & 4 else 1.0 - tz
        wgt = wx * wy * wz
        idx = (cz * gh + cy) * gw + cx                     # [H, W] into one cell plane
        for k in range(12):
            out[..., k] = out[..., k] + flat[k][idx.reshape(-1)].reshape(h, w) * wgt
    del cells
    return out


def bilagrid_apply(grid: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """Slice `grid` `[12, L, H, W]` by `rgb` `[H, W, 3]`. Row-major 3x4 affine
    applied to `(r, g, b, 1)` (bilagrid_kernels.rs:152-160)."""
    coef = _trilinear_coeffs(grid, rgb)
    ones = torch.ones_like(rgb[..., :1])
    col = torch.cat([rgb, ones], dim=-1)                   # [H, W, 4]
    m = coef.reshape(*coef.shape[:-1], 3, 4)
    return (m * col[..., None, :]).sum(-1)


def bilagrid_tv(grid: torch.Tensor) -> torch.Tensor:
    """Brush's TV: the MEAN squared first difference along each of x, y and the
    guidance axis, SUMMED over the three (bilagrid.rs:340-348)."""
    dx = (grid[..., 1:] - grid[..., :-1]) ** 2
    dy = (grid[..., 1:, :] - grid[..., :-1, :]) ** 2
    dz = (grid[..., 1:, :, :] - grid[..., :-1, :, :]) ** 2
    return dx.mean() + dy.mean() + dz.mean()


# --------------------------------------------------------------------- PPISP

def vig_uv(h: int, w: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """`[H, W, 2]`. ppisp_math.rs:386-397 -- pixel CENTRES, offset from the image
    centre, normalised by max(W, H) so the falloff is radial and not elliptical."""
    m = float(max(w, h))
    ux = (torch.arange(w, device=device, dtype=dtype) + 0.5 - w * 0.5) / m
    uy = (torch.arange(h, device=device, dtype=dtype) + 0.5 - h * 0.5) / m
    return torch.stack([ux[None, :].expand(h, w), uy[:, None].expand(h, w)], dim=-1)


def vig_falloff(uv: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """`1 + a0 r^2 + a1 r^4 + a2 r^6`, clamped to `[0, 1]`
    (ppisp_math.rs:332-347). The clamp forbids brightening, which is what keeps
    this a lens model rather than a free per-pixel gain.

    uv `[..., 2]`, params `[C, 5]` = (cx, cy, a0, a1, a2) -> `[..., C]`.
    """
    dx = uv[..., None, 0] - params[..., 0]
    dy = uv[..., None, 1] - params[..., 1]
    r2 = dx * dx + dy * dy
    f = 1.0 + params[..., 2] * r2 + params[..., 3] * r2 ** 2 + params[..., 4] * r2 ** 3
    return f.clamp(0.0, 1.0)


def _softplus(x):
    return F.softplus(x)


def crf_identity_raw(device="cpu") -> torch.Tensor:
    """The raw `(toe, shoulder, gamma, center)` that make `crf_apply` the exact
    identity. The raw ZEROS are not: softplus(0) = 0.693 gives toe = 0.993 and
    gamma = 0.793, a real curve."""
    t = math.log(math.expm1(0.7))          # 0.3 + softplus(t) == 1
    g = math.log(math.expm1(0.9))          # 0.1 + softplus(g) == 1
    return torch.tensor([t, t, g, 0.0], device=device)


def crf_apply(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Per-channel monotone toe-shoulder curve, ppisp_math.rs:360-382.
    x `[..., C]` already in [0,1]; params `[C, 4]` raw."""
    toe = 0.3 + _softplus(params[..., 0])
    shoulder = 0.3 + _softplus(params[..., 1])
    gamma = 0.1 + _softplus(params[..., 2])
    center = torch.sigmoid(params[..., 3])
    lerp = toe + center * (shoulder - toe)
    a = shoulder * center / lerp
    b = 1.0 - a
    xc = x.clamp(0.0, 1.0)
    lo = a * (xc / center).clamp_min(1e-12) ** toe
    hi = 1.0 - b * ((1.0 - xc) / (1.0 - center)).clamp_min(1e-12) ** shoulder
    y = torch.where(xc <= center, lo, hi)
    return y.clamp_min(0.0) ** gamma


# ------------------------------------------------------------------- the scoring

def quantize(img: torch.Tensor) -> torch.Tensor:
    """Clamp to [0,1] and round through uint8, exactly as a PNG round trip."""
    return (img.clamp(0.0, 1.0) * 255.0).round() / 255.0


def delta_lpips(metric, base, fit, gt) -> list[float]:
    """`LPIPS(base, gt) - LPIPS(fit, gt)` per view. POSITIVE = the fit is better.

    `fit[i] is None` means "identity fitter": both terms are still evaluated, so
    the identity control exercises the same code path and must return exactly 0.
    """
    out = []
    for b, f, g in zip(base, fit, gt):
        lo = float(metric(b, g))
        hi = float(metric(b if f is None else f, g))
        out.append(lo - hi)
    return out


# ------------------------------------------------------------------ result files

def write_json(path, obj: dict) -> Path:
    """Refuses to overwrite. Task 19's only defect that reached a number was a
    re-grade silently overwriting a different arm's verdict, both files
    well-formed (research/metal-gauss.md 12.5)."""
    p = Path(path)
    if p.exists():
        raise FileExistsError(f"refusing to overwrite an existing result: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    out = {"kind": KIND, "schema": SCHEMA, **obj}
    p.write_text(json.dumps(out, indent=2, default=str))
    return p


def read_result(path) -> dict:
    d = json.loads(Path(path).read_text())
    if d.get("kind") != KIND:
        raise ValueError(f"{path}: kind {d.get('kind')!r} is not {KIND!r}")
    if d.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {d.get('schema')!r} is not {SCHEMA}")
    return d
