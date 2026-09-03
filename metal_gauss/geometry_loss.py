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


def _normals_from_depth_torch(depth: torch.Tensor, fx, fy, cx, cy) -> torch.Tensor:
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


class _NormalsFromDepthMetal(torch.autograd.Function):
    """Metal `normals_from_depth`, with the gather-form adjoint.

    research/normals-from-depth-adjoint.md, verified 31/31 against torch.autograd. The
    backward is one thread per INPUT pixel with no atomics -- each input is read by up to
    three output pixels and its own ray multiplies all three roles -- so unlike the
    rasteriser this lane is deterministic.
    """

    @staticmethod
    def forward(ctx, depth, fx, fy, cx, cy):
        from metal_gauss.metal_backend import _load
        ctx.save_for_backward(depth)
        ctx.K = (fx, fy, cx, cy)
        return _load().nfd_forward(depth.contiguous(), fx, fy, cx, cy)[0]

    @staticmethod
    def backward(ctx, g):
        from metal_gauss.metal_backend import _load
        (depth,) = ctx.saved_tensors
        fx, fy, cx, cy = ctx.K
        return _load().nfd_backward(depth.contiguous(), g.contiguous(),
                                    fx, fy, cx, cy)[0], None, None, None, None


def normals_from_depth(depth: torch.Tensor, fx, fy, cx, cy) -> torch.Tensor:
    """Camera-frame TOWARD-camera unit normals differentiated from a depth map.

    Dispatches to the Metal kernel for float32 MPS input and falls back to the torch
    reference otherwise (float64 fixtures, CPU, degenerate shapes). The two are pinned
    against each other and against torch.autograd in tests/test_nfd_kernel.py.
    """
    if (depth.device.type == "mps" and depth.dtype == torch.float32
            and depth.dim() == 2 and depth.shape[0] >= 2 and depth.shape[1] >= 2):
        return _NormalsFromDepthMetal.apply(depth, float(fx), float(fy),
                                            float(cx), float(cy))
    return _normals_from_depth_torch(depth, fx, fy, cx, cy)


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


class _FusedGeometryLosses(torch.autograd.Function):
    """The three geometry loss terms in one pass over the image.

    Replaces ~30 torch elementwise dispatches, each reading and writing 2.7M x 3 floats at
    1920x1440. Derivation and every rule honoured here:
    research/depth-normal-loss-adjoint.md (44/44 against torch.autograd).

    Leaves are `z_img` and `n_sum`. ALPHA IS NOT A LEAF and no cotangent is produced for
    it: un-detaching alpha gives max|dL/dalpha| = 0.37, all of it from the depth branch.

    DEVIATION FROM THE NOTE, stated: the note shows `ghat_n_d = -m n_r / N` can be formed
    per reader INSIDE the gather kernel, needing no intermediate buffer. This writes
    `g_nd` to a buffer and feeds the existing `nfd_backward`. One extra (H,W,3) buffer,
    reusing a kernel already verified 31/31, in exchange for not re-deriving the gather.
    Folding it in is the remaining optimisation, not a correctness gap.
    """

    @staticmethod
    def forward(ctx, z_img, n_sum, alpha, gt_depth, gt_normal, keep, K, space):
        from metal_gauss.metal_backend import _load
        ext = _load()
        a = alpha.detach().contiguous()
        di = (z_img[..., 0] / a.clamp_min(1e-10)).contiguous()
        n_d = ext.nfd_forward(di, *K)[0]
        k = (keep.to(a.dtype).contiguous() if keep is not None
             else torch.empty(0, device=a.device, dtype=a.dtype))
        num, cnt, depth_o, nr_o = ext.geom_loss_forward(
            z_img.contiguous(), n_sum.contiguous(), a, n_d,
            gt_depth.contiguous(), gt_normal.contiguous(), k, int(space))
        # Fixed-order host reduce of the per-threadgroup partials. The count is integer:
        # an f32 sum of ones is exact only to 2^24, which is exactly one 4096^2 face.
        tot = num.sum(0)
        N = cnt.to(torch.int64).sum(0)
        # normal_loss divides by 3 * valid.sum(): the denominator is the COMPONENT count,
        # not the pixel count, because the numerator is a component-wise L1. Brush:
        # `abs_err.sum() / (valid.sum() * 3.0).clamp_min(1.0)`.
        N = torch.stack([N[0], N[1] * 3, N[2]]).clamp_min(1)
        ctx.save_for_backward(depth_o, nr_o, n_sum, a, n_d, gt_depth, gt_normal, k)
        ctx.meta = (K, int(space), tuple(N.tolist()))
        return tot / N.to(tot.dtype)

    @staticmethod
    def backward(ctx, g):
        from metal_gauss.metal_backend import _load
        ext = _load()
        depth_o, nr_o, n_sum, a, n_d, gt_depth, gt_normal, k = ctx.saved_tensors
        K, space, N = ctx.meta
        # The UPSTREAM COTANGENT already carries the caller's per-term weight -- the
        # forward returns the three losses UNWEIGHTED. Multiplying by `weights` here too
        # applied them twice; the error was exactly the weight (0.2 -> rel 0.8,
        # 0.05 -> rel 0.95), which is what named the bug.
        w = g.detach().tolist()
        inv = [1.0 / n for n in N]
        g_depth, g_nd, g_nsum = ext.geom_loss_backward(
            depth_o, nr_o, n_sum.contiguous(), a, n_d,
            gt_depth.contiguous(), gt_normal.contiguous(), k, space, w, inv)
        # depth_img feeds BOTH the depth loss and normals_from_depth, so the two paths sum.
        g_depth = g_depth + ext.nfd_backward(depth_o, g_nd, *K)[0]
        g_z = torch.zeros_like(n_sum)
        g_z[..., 0] = g_depth / a.clamp_min(1e-10)      # channels 1-2 are exactly zero
        return g_z, g_nsum, None, None, None, None, None, None


def fused_geometry_losses(z_img, n_sum, alpha, gt_depth, gt_normal, K,
                          keep=None, space="disparity"):
    """(depth, normal, depth_normal) losses, UNWEIGHTED, as a length-3 tensor.

    The caller applies its own per-term weights to the returned tensor; they must not also
    be passed in here, or they are applied twice (once by the caller, once by the
    backward's use of the upstream cotangent)."""
    return _FusedGeometryLosses.apply(
        z_img, n_sum, alpha, gt_depth, gt_normal, keep, tuple(float(x) for x in K),
        0 if space == "disparity" else 1)
