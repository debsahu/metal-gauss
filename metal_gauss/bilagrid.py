"""Per-view affine bilateral grids (BilaRF form), ported from Brush.

A phone camera rides auto-exposure and auto-white-balance through a walkthrough,
and the drift is not global: a window blows out one corner while the shaded side
of the room does not move. `AppearanceModel`'s `gain_bias` and `affine` modes are
global per frame and cannot express that. A bilateral grid can: it holds a 3x4
affine per cell of an `(x, y, guidance)` lattice and slices it per pixel, with
the pixel's own luminance as the third coordinate, so the correction varies both
across the frame and across the tone range.

WHAT THIS IS FOR, AND WHAT IT IS NOT. It exists so the gaussians do not have to
explain per-frame photometry. Held-out views get the IDENTITY -- `evaluate()`
never touches this module -- so any held-out gain is indirect, through cleaner
gaussians, and is strictly smaller than a post-hoc per-view fit's. Task 21
measured that post-hoc ceiling at +0.02667 LPIPS on playroom and +0.00888 on
ARKitScenes; those are upper bounds this cannot reach, not targets.

PROVENANCE -- all Apache-2.0 Brush (compute/brush), whose expression may be
copied. spirula-studio is GPL and was NOT read for this port.

  slice math      brush-appearance/src/bilagrid_kernels.rs:15-17 (BT.601 luma),
                  :37-70 (sample point: aligned corners, border clamp, guidance
                  clamp), :120-152 (8-corner trilinear), :158-171 (row-major 3x4
                  applied to (r, g, b, 1))
  identity init   brush-appearance/src/bilagrid.rs:364-382 -- channels 0, 5, 10
  TV              brush-appearance/src/bilagrid.rs:340-348 -- MEAN squared first
                  difference along each of x, y and guidance, SUMMED over the three
  per-view TV     brush-appearance/src/train_state.rs:352-366, 453-465 -- Brush
                  lifts only the ACTIVE view's grid and regularises THAT, not the
                  whole stack. Reproduced here; see `regulariser`.
  lr schedule     brush-appearance/src/train_state.rs:16-19, 44-53 and
                  lib.rs:159-174 -- warmup 1000 steps from 0.01x, then exponential
                  decay to 0.01x over the run
  defaults        brush-train/src/config.rs:1005-1031 -- dims 16,16,8; tv 10.0;
                  lr 2e-3; betas 0.9,0.999

THREE DELIBERATE DEVIATIONS FROM BRUSH, each one a decision and not an oversight:

 1. NO NON-FINITE FALLBACK. Brush's kernel writes 0.5 for a non-finite output
    (`bilagrid_kernels.rs:170`) and drops non-finite gradients (`:174-177`). We do
    not. A non-finite render is a defect this project needs to SEE -- CLAUDE.md's
    Stage 5 records a single NaN poisoning an entire SOG codebook and killing the
    viewer's host machine -- and a silent 0.5 is exactly the class of paper-over
    this repo exists to catch.
 2. NO ALPHA CHANNEL. Brush's kernel passes a 4th channel through untouched. Our
    render path hands the loss a bare `(H, W, 3)`; masks are a separate tensor and
    the aux maps are separate tensors again, so there is no alpha to preserve.
 3. NO EVAL APPLICATION. Brush's `apply_eval` (`train_state.rs:422`) applies the
    per-view grid at eval time. That is a per-view cheat under this trainer's
    discipline and is the single easiest way to produce a spectacular, worthless
    held-out number. It is not ported, and a test pins its absence.

WHY einsum AND NOT THE OBVIOUS FORM. The natural transcription materialises the
12 sliced coefficients as `[H, W, 12]` and broadcasts. Measured on this M5 Pro at
1440x1920, fwd+bwd, warmed: that form is 106.8 ms against 46.8 ms for the einsum
below, for a BIT-IDENTICAL result (max|delta| = 0.0). `grid_sample` itself is only
1.86 ms of either; the rest is permute/reshape traffic over a 33M-element tensor.
The einsum form is still far over the plan's 5%-of-step-time line -- see the
branch's pre-registration commit, which records the overrun and the decision to
measure the arm in torch before writing any Metal.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# BT.601, bilagrid_kernels.rs:15-17.
LUMA_R, LUMA_G, LUMA_B = 0.299, 0.587, 0.114

# brush-train/src/config.rs:1005-1031.
DEFAULT_DIMS = (16, 16, 8)          # (x, y, guidance) -- Brush's --bilagrid-dims order
DEFAULT_TV_WEIGHT = 10.0
DEFAULT_LR = 2e-3
LR_WARMUP_STEPS = 1000              # train_state.rs:16
LR_START_FACTOR = 0.01              # train_state.rs:18
LR_FINAL_FACTOR = 0.01              # train_state.rs:19

# The 3x4 affine is ROW-MAJOR, so its diagonal is coefficients 0, 5 and 10 --
# NOT 0, 4, 8, which is the column-major reading and is the one bug in this file
# that produces a plausible-looking image instead of an error.
IDENTITY_CHANNELS = (0, 5, 10)


def warmup_exp_lr(step: int, base: float, *, warmup: int = LR_WARMUP_STEPS,
                  start_factor: float = LR_START_FACTOR,
                  final_factor: float = LR_FINAL_FACTOR,
                  decay_steps: int = 30000) -> float:
    """Brush's `warmup_exp_lr` (brush-appearance/src/lib.rs:159-174).

    Linear warmup from `start_factor * base` over `warmup` steps, then exponential
    decay reaching `final_factor * base` at `decay_steps` steps PAST the warmup.
    Note the decay's exponent divides by `decay_steps`, not by `decay_steps -
    warmup`, so the final LR at `step == decay_steps` is slightly above
    `final_factor * base`. That is Brush's arithmetic and is reproduced rather
    than corrected: matching the reference is the point of a port.
    """
    if step < warmup:
        t = (step + 1.0) / warmup
        return base * (start_factor + (1.0 - start_factor) * t)
    return base * final_factor ** ((step - warmup) / max(decay_steps, 1))


def identity_grids(n_images: int, dims=DEFAULT_DIMS, device="mps",
                   dtype=torch.float32) -> torch.Tensor:
    """`[N, 12, L, H, W]` identity affines. `dims` is `(x, y, guidance)`."""
    gw, gh, gl = dims
    if min(gw, gh, gl) < 2:
        # Brush asserts the same thing (bilagrid.rs:52-59): a dimension of 1
        # divides by zero in both the interpolation and the TV normalisation.
        raise ValueError(f"bilagrid dims must each be >= 2, got {dims}")
    g = torch.zeros(n_images, 12, gl, gh, gw, device=device, dtype=dtype)
    for c in IDENTITY_CHANNELS:
        g[:, c] = 1.0
    return g


def sample_coords(rgb: torch.Tensor) -> torch.Tensor:
    """`[1, 1, H, W, 3]` normalised `grid_sample` coordinates for one render.

    Brush's kernel (bilagrid_kernels.rs:49-70) computes

        x = px * (gw - 1) / max(W - 1, 1)
        y = py * (gh - 1) / max(H - 1, 1)
        z = clamp(luma * (gl - 1), 0, gl - 1)

    and trilinearly interpolates with corners clamped to the last cell. In
    `grid_sample`'s normalised space that is EXACTLY `align_corners=True,
    padding_mode="border"` with

        xn = 2*px/(W-1) - 1,  yn = 2*py/(H-1) - 1,  zn = clamp(2*luma - 1, -1, 1)

    because align_corners maps [-1, 1] onto [0, D-1] and every `(g* - 1)` factor
    cancels. The grid's last axis is ordered (x, y, z) against the input's
    (W, H, D) -- the REVERSE of the tensor's own axis order. That reversal is the
    easiest thing here to get wrong and is why the reference test compares against
    an explicit 8-corner transcription rather than against another vectorised form.

    The `clamp` on `zn` is not cosmetic: it reproduces Brush's `guidance_active`
    gate (`:67`). `torch.clamp` has zero gradient outside its range, so a pixel
    whose luminance is off the end of the guidance axis contributes no gradient to
    the luminance -- which is precisely what that flag does in Brush's backward.
    """
    h, w, _ = rgb.shape
    dev, dt = rgb.device, rgb.dtype
    xn = 2.0 * torch.arange(w, device=dev, dtype=dt) / max(w - 1, 1) - 1.0
    yn = 2.0 * torch.arange(h, device=dev, dtype=dt) / max(h - 1, 1) - 1.0
    lum = LUMA_R * rgb[..., 0] + LUMA_G * rgb[..., 1] + LUMA_B * rgb[..., 2]
    zn = (2.0 * lum - 1.0).clamp(-1.0, 1.0)
    return torch.stack([xn[None, :].expand(h, w),
                        yn[:, None].expand(h, w), zn], dim=-1)[None, None]


def slice_apply(grid: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """Slice one grid `[1, 12, L, H, W]` by `rgb` `[H, W, 3]` -> `[H, W, 3]`.

    Row-major 3x4 affine applied to `(r, g, b, 1)` (bilagrid_kernels.rs:158-171).
    """
    if grid.dim() != 5 or grid.shape[0] != 1 or grid.shape[1] != 12:
        raise ValueError(f"grid must be [1, 12, L, H, W], got {tuple(grid.shape)}")
    h, w, c = rgb.shape
    if c != 3:
        raise ValueError(f"rgb must be [H, W, 3], got {tuple(rgb.shape)}")
    coef = F.grid_sample(grid, sample_coords(rgb), mode="bilinear",
                         padding_mode="border", align_corners=True).reshape(3, 4, h, w)
    col = torch.cat([rgb.permute(2, 0, 1),
                     torch.ones(1, h, w, device=rgb.device, dtype=rgb.dtype)], dim=0)
    return torch.einsum("rchw,chw->hwr", coef, col)


def tv_loss(grid: torch.Tensor) -> torch.Tensor:
    """Brush's TV (bilagrid.rs:340-348): the MEAN squared first difference along
    each of x, y and the guidance axis, SUMMED over the three.

    Note both halves: a SUM of three MEANS, not a mean of the concatenation and
    not a sum of sums. The three axes have different lengths, so the difference is
    a real reweighting and not a constant factor.
    """
    dx = (grid[..., 1:] - grid[..., :-1]) ** 2
    dy = (grid[..., 1:, :] - grid[..., :-1, :]) ** 2
    dz = (grid[..., 1:, :, :] - grid[..., :-1, :, :]) ** 2
    return dx.mean() + dy.mean() + dz.mean()


class BilateralGrid(torch.nn.Module):
    """One `[12, L, H, W]` affine grid per TRAINING view, identity at init."""

    def __init__(self, n_images: int, dims=DEFAULT_DIMS, device="mps"):
        super().__init__()
        self.dims = tuple(dims)
        self.grids = torch.nn.Parameter(identity_grids(n_images, dims, device))

    def forward(self, rgb: torch.Tensor, idx: int) -> torch.Tensor:
        # A SLICE, not an index: `self.grids[idx]` would drop the leading axis and
        # `grid_sample` needs the 5D form. The slice keeps the graph attached to
        # the whole parameter, so one view's photometric gradient lands in that
        # view's block and nowhere else.
        return slice_apply(self.grids[idx:idx + 1], rgb)

    def regulariser(self, idx: int | None = None) -> torch.Tensor:
        """TV over the ACTIVE view's grid, matching Brush.

        Brush lifts only `grid.view_grid(view_idx)` into autodiff each step and
        regularises that single `[1, 12, L, H, W]` block
        (train_state.rs:352-366, 453-465). Averaging over all N views instead
        would divide every view's TV gradient by N while its photometric gradient
        stayed put -- an N-fold weakening of a regulariser Task 21 measured to be
        LOAD-BEARING, not a tuning knob (the unregularised grid recovers a
        roughly scene-independent 7.4% of baseline on lego, i.e. nuisance
        capacity; the regularised one recovers 0.7%). With 196 views that is the
        difference between the model and the nuisance regime.

        `idx=None` returns the whole stack's TV, which is what a caller wants for
        reporting and NOT what the training loss uses.
        """
        g = self.grids if idx is None else self.grids[idx:idx + 1]
        return tv_loss(g)
