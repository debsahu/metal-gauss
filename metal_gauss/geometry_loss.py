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


from metal_gauss.torch_ref import quat_to_rotmat  # noqa: E402


def splat_normals_cam(means: torch.Tensor, quats: torch.Tensor, scales: torch.Tensor,
                      viewmat: torch.Tensor) -> torch.Tensor:
    """Per-splat camera-frame unit normal, made camera-facing by the PER-SPLAT VIEW RAY.

    Brush `train.rs` `splat_normals` + a world_to_cam rotation. The thinnest axis is
    COLUMN `argmin(scales)` of R(quat), because `build_cov3d` forms `R @ diag(s)`, i.e.
    scale i multiplies column i.

    ORIENTATION. "Facing the camera" is `dot(n, mean - cam_pos) <= 0`, evaluated against
    THAT SPLAT's own ray -- never `n_z <= 0`, which is what the ray test degenerates to
    only on the optical axis. In camera coordinates the ray from the camera to the splat
    IS `p_cam`, so the test is `dot(n_cam, p_cam) > 0 -> flip`. See
    research/normal-orientation-gate-defect.md: four generators in earthbyte/slam shipped
    the `n_z` form, and on 90-degree cube faces it emitted 20.594% of valid pixels facing
    backwards while looking like a working flip.

    The selector is a COMPARISON, not `sign()`. `sign(0) == 0` and multiplying by it
    annihilates an exactly-perpendicular splat's normal to (0,0,0), which every consumer
    here reads as "invalid" and silently drops from the loss. Brush spells the same thing
    `(facing < 0) * 2 - 1`, which keeps the tie NEGATED; this port flips on `not (dot < 0)`
    to match it exactly. Which sign an edge-on splat gets is arbitrary (the set has measure
    zero), but it is pinned so the two implementations cannot quietly diverge.

    Equivalently to Brush, which dots the WORLD normal with `mean - cam_pos`: a rotation
    preserves dot products, so `dot(R n, R (mean - cam)) = dot(n, mean - cam)`, and in
    camera coordinates `R (mean - cam)` IS `p_cam`.

    The argmin and the facing selector are DETACHED discrete choices: `scales` receives no
    gradient from this function (flatten is the scale-side pressure) and the flip does not
    differentiate through its own comparison.
    """
    R = quat_to_rotmat(quats)                                # (N,3,3), normalises internally
    idx = scales.detach().argmin(dim=1)
    n_world = R[torch.arange(len(R), device=R.device), :, idx]        # COLUMN idx of R
    Rc = viewmat[:3, :3].to(n_world)
    t = viewmat[:3, 3].to(n_world)
    n_cam = n_world @ Rc.T
    n_cam = n_cam / n_cam.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    p_cam = (means @ Rc.T + t).detach()
    facing = (n_cam.detach() * p_cam).sum(-1)
    facing_away = ~(facing < 0)[:, None]         # Brush: `(facing < 0) * 2 - 1`
    return torch.where(facing_away, -n_cam, n_cam)


def depth_loss(pred: torch.Tensor, gt: torch.Tensor,
               space: str = "disparity") -> torch.Tensor:
    """Brush `brush-loss` `depth_loss`, `DepthUncovered::Count`.

    Valid is `gt > 0` (0 is the prior codec's invalid sentinel). In disparity space the
    residual is `1/pred - 1/gt`, with an uncovered pixel (`pred <= 0`) scoring the FULL
    disparity `1/gt` rather than being skipped -- so failing to cover a surface at all is
    penalised, not ignored. The divisor is the VALID count, not the pixel count: dividing
    by H*W would silently down-weight a sparse prior against a dense one.

    Substitute-then-compute, never multiply-after. A masked pixel must contribute exactly
    nothing, and `x * 0` is NaN when x is inf or NaN -- one non-finite render pixel would
    otherwise NaN the step, and Adam writes NaN straight into the parameters.
    """
    gt_valid = gt > 0
    zero = torch.zeros((), dtype=pred.dtype, device=pred.device)
    one = torch.ones((), dtype=pred.dtype, device=pred.device)
    pred_s = torch.where(gt_valid, pred, zero)
    gt_s = torch.where(gt_valid, gt, zero)
    if space == "disparity":
        pred_pos = pred_s > 0
        disp_pred = torch.where(pred_pos, 1.0 / torch.where(pred_pos, pred_s, one), zero)
        disp_gt = torch.where(gt_valid, 1.0 / torch.where(gt_valid, gt_s, one), zero)
        residual = disp_pred - disp_gt
    elif space == "metric":
        residual = pred_s - gt_s
    else:
        raise ValueError(f"space must be disparity or metric, got {space!r}")
    valid = gt_valid.to(pred.dtype)
    return (residual.abs() * valid).sum() / valid.sum().clamp_min(1.0)


def normal_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Brush `normal_loss`: component-wise L1 over pixels whose prior is valid.

    L1 on components, NOT cosine. Cosine cannot separate two predictions that are both
    90 degrees from the target, and its gradient vanishes there.
    """
    valid = gt.norm(dim=-1) > 0.5                # (0,0,0) is the codec's invalid sentinel
    v3 = valid[..., None]
    pred_s = torch.where(v3, pred, torch.zeros_like(pred))
    gt_s = torch.where(v3, gt, torch.zeros_like(gt))
    err = (pred_s - gt_s).abs().sum(-1) * valid.to(pred.dtype)
    return err.sum() / (3.0 * valid.to(pred.dtype).sum()).clamp_min(1.0)


def normals_from_depth(depth: torch.Tensor, fx, fy, cx, cy) -> torch.Tensor:
    """Brush `brush-loss` `normals_from_depth`. Camera-frame, TOWARD-camera, unit.

    Integer pixel indices `(u - cx)/fx` (matching Brush and the cross-language fixture).
    Note the rasterizer samples pixel CENTRES at +0.5, so this carries a deliberate
    half-pixel shear against the render; it is kept for fixture compatibility and recorded
    here rather than silently fixed on one side of a cross-language contract.

    THERE IS NO ORIENTATION FLIP HERE, AND ADDING ONE WOULD BE A BUG. For the forward
    difference form, the exact discrete invariant is

        dot(cross(dPdu, dPdv), r) = z(u+1, v) * z(u, v+1) / (fx * fy)

    for ANY depth graph. Validity already requires both those depths positive, so that dot
    is strictly positive wherever a normal is emitted, and `cross(dPdv, dPdu)` -- the same
    vector negated -- is therefore toward-camera everywhere by construction. A per-pixel
    ray test would be correct but redundant; an `n_z` test would be wrong.
    """
    h, w = depth.shape
    out = torch.zeros(h, w, 3, dtype=depth.dtype, device=depth.device)
    if h < 2 or w < 2:
        return out
    u = (torch.arange(w, dtype=depth.dtype, device=depth.device) - cx) / fx
    v = (torch.arange(h, dtype=depth.dtype, device=depth.device) - cy) / fy
    P = torch.stack([depth * u[None, :], depth * v[:, None], depth], -1)
    base = P[:-1, :-1]
    du = P[:-1, 1:] - base
    dv = P[1:, :-1] - base
    cross = torch.cross(dv, du, dim=-1)
    # clamp BEFORE the sqrt: sqrt(0) has an infinite derivative, and 0 * inf is the NaN
    # Brush hit here.
    length = (cross * cross).sum(-1).clamp_min(1e-24).sqrt()
    n = cross / length[..., None]
    dpos = depth > 0
    valid = dpos[:-1, :-1] & dpos[:-1, 1:] & dpos[1:, :-1] & (length > 1e-12)
    out[:-1, :-1] = torch.where(valid[..., None], n, torch.zeros_like(n))
    return out


def depth_normal_loss(n_from_depth: torch.Tensor, n_rendered: torch.Tensor,
                      alpha: torch.Tensor) -> torch.Tensor:
    """PlanarGS depth-normal consistency: mean `1 - dot(n_d, n_r)` over covered, valid
    pixels. 0 when they agree, 1 at 90 degrees, 2 when anti-aligned -- so a sign-flipped
    prior blows this term up rather than hiding in it. Needs no prior data at all, which
    makes it the cheapest of the three to switch on."""
    covered = alpha > 0.5
    valid = covered & (n_from_depth.norm(dim=-1) > 0.5) & (n_rendered.norm(dim=-1) > 0.5)
    v3 = valid[..., None]
    nd = torch.where(v3, n_from_depth, torch.zeros_like(n_from_depth))
    nr = torch.where(v3, n_rendered, torch.zeros_like(n_rendered))
    err = (1.0 - (nd * nr).sum(-1)) * valid.to(nd.dtype)
    return err.sum() / valid.to(nd.dtype).sum().clamp_min(1.0)


def plane_features(means: torch.Tensor, quats: torch.Tensor, scales: torch.Tensor,
                   viewmat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PGSR per-splat plane features: camera-frame normal and plane offset.

    `d = n_world . (mean - cam_pos)`, which in camera coordinates is exactly
    `n_cam . p_cam` -- a rotation preserves dot products, so the two forms agree and this
    one needs no camera centre. Both outputs are live: `quats` learns through `n`, `means`
    through `d`. `scales` gets nothing (the thin-axis choice is a detached argmin).

    Composited over a pixel these give the plane whose intersection with the ray is the
    unbiased surface depth -- see `plane_depth_from_features`.
    """
    n_cam = splat_normals_cam(means, quats, scales, viewmat)
    Rc = viewmat[:3, :3].to(means)
    t = viewmat[:3, 3].to(means)
    p_cam = means @ Rc.T + t
    return n_cam, (n_cam * p_cam).sum(-1)


def plane_depth_from_features(feat_img: torch.Tensor, fx, fy, cx, cy,
                              min_alpha: float = 0.5, min_denom: float = 1e-3,
                              min_depth: float = 1e-3, max_depth: float = 1e3):
    """PGSR unbiased surface depth from composited plane features.

    `feat_img` is (H,W,5): composited `n_sum` (3), `offset_sum` (1), `alpha` (1). Port of
    Brush `plane_depth_from_features` (brush-loss/src/lib.rs:2426).

    NO ALPHA DIVISION. Numerator and denominator carry the SAME blending weights, so alpha
    cancels exactly -- unlike the centre-depth path, which must divide. Pixel-CENTRE rays
    `(u + 0.5 - cx)/fx` here, while `normals_from_depth` uses INTEGER indices; both match
    Brush, and both are kept deliberately, because changing one side of a cross-language
    contract to match the other is how the two silently diverge.

    Every channel is sanitised up front on the JOINT finite mask, BEFORE any division. One
    non-finite channel makes the whole pixel meaningless -- a NaN in `n_x` alone would
    otherwise decay into a plausible axis-aligned plane and be reported valid -- and a
    non-finite value masked out AFTER an op reappears as `0 * inf` in that op's VJP.

    Returns `(depth, normal, valid)`; invalid pixels are exactly 0 in both maps, which is
    what `depth_loss` scores as uncovered.
    """
    h, w, c = feat_img.shape
    if c != 5:
        raise ValueError(f"plane feature image must be (H,W,5) = n_sum(3)+offset(1)+alpha(1), got {c}")
    finite = torch.isfinite(feat_img).all(dim=-1)
    f = torch.where(finite[..., None], feat_img, torch.zeros_like(feat_img))
    n_sum, offset, alpha = f[..., :3], f[..., 3], f[..., 4].detach()

    u = (torch.arange(w, dtype=feat_img.dtype, device=feat_img.device) + 0.5 - cx) / fx
    v = (torch.arange(h, dtype=feat_img.dtype, device=feat_img.device) + 0.5 - cy) / fy
    denom = n_sum[..., 0] * u[None, :] + n_sum[..., 1] * v[:, None] + n_sum[..., 2]
    denom_ok = denom.abs() >= min_denom
    safe = torch.where(denom_ok, denom, torch.ones_like(denom))
    depth_raw = offset / safe
    in_range = (depth_raw >= min_depth) & (depth_raw <= max_depth)
    valid = (alpha >= min_alpha) & finite & denom_ok & in_range

    depth = torch.where(valid, depth_raw, torch.zeros_like(depth_raw))
    length = (n_sum * n_sum).sum(-1).clamp_min(1e-24).sqrt()
    normal = n_sum / length[..., None]
    normal = torch.where(valid[..., None], normal, torch.zeros_like(normal))
    return depth, normal, valid.to(feat_img.dtype)
