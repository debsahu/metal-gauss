"""Per-image appearance correction for captures with auto-exposure.

A phone camera rides its auto-exposure through a walkthrough, so the same wall
is a different brightness in different training frames. Without a way to
express that, the optimiser bakes the variation into the gaussians themselves
-- as colour drift, spurious geometry in shaded regions, or opacity soup -- and
held-out views inherit the damage.

This gives each TRAINING image a small learnable photometric transform (per
channel gain + bias, optionally a full 3x3 colour matrix) applied to the render
before the loss. The gaussians then only have to explain what is actually
scene-dependent. Held-out views get the identity transform: the model must
generalise without a per-view cheat, which is what makes the held-out PSNR gain
real rather than a fitting artefact.

Same idea as gsplat's bilateral grid / NeRF-W appearance embeddings, minus the
spatial grid -- exposure and white balance are global per frame, and the global
version costs 6 parameters per image instead of thousands.

The `bilagrid` mode is the one that DOES carry the spatial grid: a 3x4 affine per
cell of a 16x16x8 (x, y, guidance) lattice, sliced per pixel by the render's own
luminance. It is Brush's BilaRF form, lives in metal_gauss/bilagrid.py, and is
regularised by a total-variation term at Brush's weight 10.0 rather than by
`--appearance-reg`. That distinction is not cosmetic: Task 21 measured the TV
term to be load-bearing rather than a knob, so `reg_weight` is a property of the
MODE and is carried on the model instead of being chosen at the call site.
"""

from __future__ import annotations

import torch

from metal_gauss.bilagrid import BilateralGrid, DEFAULT_DIMS, DEFAULT_TV_WEIGHT


class AppearanceModel(torch.nn.Module):
    def __init__(self, n_images: int, mode: str = "gain_bias", device: str = "mps",
                 *, reg_weight: float = 1e-2, tv_weight: float = DEFAULT_TV_WEIGHT,
                 dims=DEFAULT_DIMS):
        super().__init__()
        self.mode = mode
        # The weight the training loss must multiply `regulariser()` by. Held here
        # rather than at the call site because it is a property of the mode: the
        # grid's is Brush's TV weight (10.0) and the global modes' is
        # `--appearance-reg` (1e-2), and picking the wrong one is silent -- the arm
        # runs and is the nuisance model instead of the ported one.
        self.reg_weight = float(tv_weight if mode == "bilagrid" else reg_weight)
        if mode == "bilagrid":                        # 24576 params / image at 16x16x8
            self.grid = BilateralGrid(n_images, dims, device=device)
        elif mode == "gain_bias":                     # 6 params / image
            self.gain = torch.nn.Parameter(torch.ones(n_images, 3, device=device))
            self.bias = torch.nn.Parameter(torch.zeros(n_images, 3, device=device))
        elif mode == "affine":                        # 12 params / image
            eye = torch.eye(3, device=device).expand(n_images, 3, 3).clone()
            self.matrix = torch.nn.Parameter(eye)
            self.bias = torch.nn.Parameter(torch.zeros(n_images, 3, device=device))
        else:
            raise ValueError(f"unknown appearance mode {mode!r}")

    def forward(self, rgb: torch.Tensor, idx: int) -> torch.Tensor:
        """rgb: (H,W,3) render -> photometrically corrected render."""
        if self.mode == "bilagrid":
            return self.grid(rgb, idx)
        if self.mode == "gain_bias":
            return rgb * self.gain[idx] + self.bias[idx]
        return rgb @ self.matrix[idx].T + self.bias[idx]

    def regulariser(self, idx: int | None = None) -> torch.Tensor:
        """Keep transforms near identity so they correct exposure, not content.

        `idx` is the view being drawn this step. The global modes ignore it -- their
        parameters are a fixed-size block and Brush's equivalent regularises all of
        it -- while the grid regularises ONLY that view's block, which is what Brush
        does (train_state.rs:352-366, 453-465) and matters at the N-fold level. See
        `BilateralGrid.regulariser`.
        """
        if self.mode == "bilagrid":
            return self.grid.regulariser(idx)
        if self.mode == "gain_bias":
            return ((self.gain - 1.0) ** 2).mean() + (self.bias ** 2).mean()
        eye = torch.eye(3, device=self.matrix.device)
        return ((self.matrix - eye) ** 2).mean() + (self.bias ** 2).mean()

    @torch.no_grad()
    def state_summary(self) -> dict:
        """What the model LEARNED, for the report. Not the configuration -- that is
        already in `resolved`. `max_abs_dev` is the number that separates "the grid
        trained" from "the grid was constructed", and it is the one a wiring test
        can assert on."""
        if self.mode == "bilagrid":
            g = self.grid.grids
            ident = torch.zeros_like(g)
            for c in (0, 5, 10):
                ident[:, c] = 1.0
            return {"mode": self.mode, "params": int(g.numel()),
                    "dims": list(self.grid.dims),
                    "max_abs_dev": float((g - ident).abs().max()),
                    "tv": float(self.grid.regulariser(None)),
                    "reg_weight": self.reg_weight}
        n = sum(x.numel() for x in self.parameters())
        return {"mode": self.mode, "params": int(n),
                "max_abs_dev": None, "tv": None, "reg_weight": self.reg_weight}
