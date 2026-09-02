"""Geometry terms of the earthbyte/slam Stage 3 recipe, ported from compute/brush
(brush-loss/src/lib.rs, brush-train/src/train.rs). Pure torch, device-agnostic.

The recipe these implement, measured in Brush and NOT yet in this trainer:
    --flatten-loss-weight 1.0 --depth-loss-weight 1.0
    --normal-loss-weight 0.2  --depth-normal-weight 0.05
"""
from __future__ import annotations

import torch


def flatten_loss(log_scales: torch.Tensor) -> torch.Tensor:
    """PlanarGS L_s: mean over splats of the SMALLEST ACTIVATED scale.

    Brush `train.rs`: `scales.min_dim(1).mean() * (flatten_loss_weight / metric_scale)`.
    We apply it as a CONSTANT weight with no metric normalisation and no ramp -- CLAUDE.md
    records `--normalize-metric-weights` as a measurable dilution on metric scenes
    (+1.1-1.4 deg of thin-axis on both test scenes), i.e. a weaker flatten wearing the
    name of a normalisation.

    `exp` is load-bearing: the term is a length in metres. On log scales, every sub-metre
    splat contributes a NEGATIVE number and minimising it inflates the thin axis instead.
    """
    return torch.exp(log_scales).min(dim=-1).values.mean()
