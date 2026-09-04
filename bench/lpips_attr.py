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

    # All 12 coefficients are gathered in ONE index_select per corner rather than
    # twelve, which is the difference between 96 and 8 gathers per forward and
    # therefore between a fit that takes minutes and one that takes an hour.
    flat = grid.reshape(12, -1)
    acc = torch.zeros(12, h * w, device=dev, dtype=rgb.dtype)
    for corner in range(8):
        cx = x1i if corner & 1 else x0i
        cy = y1i if corner & 2 else y0i
        cz = z1i if corner & 4 else z0i
        wx = tx if corner & 1 else 1.0 - tx
        wy = ty if corner & 2 else 1.0 - ty
        wz = tz if corner & 4 else 1.0 - tz
        wgt = (wx * wy * wz).reshape(-1)
        idx = ((cz * gh + cy) * gw + cx).reshape(-1)
        acc = acc + flat.index_select(1, idx) * wgt
    return acc.reshape(12, h, w).permute(1, 2, 0)


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


# ------------------------------------------------------------------- the fitters
#
# All three fit POST HOC to a FROZEN render, minimising L2 to the ground truth.
# None of them is a training arm and none of them touches a splat. What they
# bound is the PHOTOMETRIC COMPONENT of the held-out residual -- the only
# component any appearance model could reduce by any route, direct or indirect.

def apply_ppisp(rgb: torch.Tensor, vig: torch.Tensor, crf: torch.Tensor) -> torch.Tensor:
    """PPISP's PER-CAMERA stages only: vignetting then CRF (ppisp_kernels.rs:82-168).

    The per-FRAME stages (exposure, colour homography) are deliberately absent.
    A per-frame stage evaluated on a held-out view is exactly the per-view cheat
    this trainer forbids, and including them would make (c) a per-view fitter
    wearing a shared model's name.
    """
    h, w, _ = rgb.shape
    f = vig_falloff(vig_uv(h, w, rgb.device, rgb.dtype), vig)
    return crf_apply((rgb * f).clamp(0.0, 1.0), crf)


def fit_affine(render: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Per-view global 3x4 affine, solved in CLOSED FORM.

    A global affine's L2 optimum is an ordinary least-squares solve, so there is
    no optimiser to converge and (a)'s ceiling is exact rather than
    optimisation-limited. That matters: the whole decision turns on a number
    being SMALL, and "small because the fitter did not converge" is the failure
    mode this task is most exposed to. Normal equations in float64 on the CPU --
    MPS has no float64 and a float32 sum over 2.8M pixels is not free.
    """
    h, w, _ = render.shape
    # .cpu() BEFORE .double(): MPS has no float64 and `.double()` on an MPS
    # tensor raises. research/metal-gauss.md 13.5 records the same defect found
    # by the same means -- only a test that renders can see it.
    x = render.reshape(-1, 3).cpu().double()
    y = gt.reshape(-1, 3).cpu().double()
    x1 = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], 1)     # [N, 4]
    a = x1.T @ x1
    b = x1.T @ y
    wmat = torch.linalg.solve(a + 1e-9 * torch.eye(4, dtype=a.dtype), b)  # [4, 3]
    fit = (x1 @ wmat).reshape(h, w, 3).to(render.dtype).to(render.device)
    return fit, {"fitter": "affine", "params": wmat.T.tolist(), "n_params": 12,
                 "mse_before": float(((render - gt) ** 2).mean()),
                 "mse_after": float(((fit - gt) ** 2).mean())}


def fit_bilagrid(render: torch.Tensor, gt: torch.Tensor, dims=BILAGRID_DIMS,
                 tv_weight: float = BILAGRID_TV_WEIGHT, steps: int = BILAGRID_STEPS,
                 lr: float = BILAGRID_LR, betas=BILAGRID_BETAS,
                 log_every: int = 100) -> tuple[torch.Tensor, dict]:
    """Per-view 16x16x8 bilateral grid of 3x4 affines, Brush's configuration."""
    gx, gy, gl = dims
    grid = bilagrid_identity(gl, gy, gx, device=render.device).requires_grad_(True)
    opt = torch.optim.Adam([grid], lr=lr, betas=tuple(betas))
    curve = []
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        out = bilagrid_apply(grid, render)
        mse = ((out - gt) ** 2).mean()
        loss = mse + tv_weight * bilagrid_tv(grid)
        loss.backward()
        opt.step()
        if s % log_every == 0 or s == steps - 1:
            curve.append([s, float(mse.detach()), float(loss.detach())])
    with torch.no_grad():
        fit = bilagrid_apply(grid, render)
    return fit.detach(), {"fitter": f"bilagrid_tv{tv_weight:g}", "dims": list(dims),
                          "tv_weight": tv_weight, "steps": steps, "lr": lr,
                          "n_params": 12 * gx * gy * gl, "mse_curve": curve,
                          "mse_before": float(((render - gt) ** 2).mean()),
                          "mse_after": float(((fit - gt) ** 2).mean())}


def fit_ppisp_shared(renders: list[torch.Tensor], gts: list[torch.Tensor],
                     steps: int = 500, lr: float = 2e-3, betas=(0.9, 0.999),
                     log_every: int = 100) -> tuple[list[torch.Tensor], dict]:
    """ONE vignetting + CRF for the whole scene: 15 + 12 = 27 parameters total.

    Gradients are accumulated view by view rather than in one batch -- identical
    to a full-batch step on the mean loss, and the only way 25 views at 2.8 Mpx
    fit in memory with a graph attached.
    """
    dev = renders[0].device
    vig = torch.zeros(3, 5, device=dev, requires_grad=True)
    crf = crf_identity_raw(dev).expand(3, 4).clone().requires_grad_(True)
    opt = torch.optim.Adam([vig, crf], lr=lr, betas=tuple(betas))
    n = len(renders)
    curve = []
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for r, g in zip(renders, gts):
            mse = ((apply_ppisp(r, vig, crf) - g) ** 2).mean() / n
            mse.backward()
            tot += float(mse.detach())
        opt.step()
        if s % log_every == 0 or s == steps - 1:
            curve.append([s, tot])
    with torch.no_grad():
        fits = [apply_ppisp(r, vig, crf) for r in renders]
    before = float(sum(((r - g) ** 2).mean() for r, g in zip(renders, gts)) / n)
    after = float(sum(((f - g) ** 2).mean() for f, g in zip(fits, gts)) / n)
    return fits, {"fitter": "ppisp_shared", "steps": steps, "lr": lr, "n_params": 27,
                  "n_views_shared_over": n, "mse_curve": curve,
                  "vignetting": vig.detach().cpu().tolist(),
                  "crf_raw": crf.detach().cpu().tolist(),
                  "mse_before": before, "mse_after": after}
